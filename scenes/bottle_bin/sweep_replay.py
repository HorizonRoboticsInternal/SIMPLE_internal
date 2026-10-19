#!/usr/bin/env python
"""Batch replay of many real episodes in the bottle / trash-bin scene: headless, unpaced, parallel-safe.

Same control path as replay_in_scene.py (recorded upper-body targets + navigation command + base height fed to
PicoDecoupledAgent at 50 Hz, lower-body RL policy walking by itself) with three changes that make a 97-episode
sweep practical:

  * no video, no third-person renderer          -- only the numbers are kept
  * no wall-clock pacing                        -- the loop runs as fast as physics allows
  * no Pico TCP server / video streamer         -- several worker processes can run at once

One worker = one scene config (robot-to-edge, bin xy, bottle mass) and many (episode, bottle xy) jobs; the env and
the agent are built once and reused, the bottle is re-placed per job through its free joint.

  python sweep_replay.py --tag base3cm --robot-to-edge 0.03 --episodes 0 11 22
  python sweep_replay.py --tag grid --jobs jobs.json --shard 0/6
  python sweep_replay.py --tag base3cm --episodes ... --self-check      # run job 0 twice, compare

Writes replay/sweep/<tag>/ep<N>[_<variant>].json, one per job.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("BOTTLE_BIN_SWEEP", "1")
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]   # the SIMPLE checkout (scenes/<kit>/ lives inside it)
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
OUT = HERE / "replay" / "sweep"
HZ = 50
STABILIZE_TICKS = 150            # replaces replay_in_scene.py's "stabilized and 2 s of wall clock"
EXPORT_GATE = {"on": True}       # replay_isaac.py exports dataset frames only while this is on (settle + pre-pose stay out; 2026-09-27)
PREPOSE_TICKS = int(os.environ.get("REPLAY_PREPOSE_TICKS", "100"))   # 2026-09-25: after stabilising, 2 s with the first recorded upper-body pose (early hand closures)
SETTLE_S = float(os.environ.get("REPLAY_SETTLE_S", "1.5"))
UPPER_NAMES = ([f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
               + [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
               + [f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"])          # psi0 rows: the torso rpy command drives the waist

# 2026-09-29 (data check replay_phys): the psi0 ACTION orders the LEFT hand thumb_0..2, middle_0, middle_1, index_0, index_1 and the
# right hand thumb, index, middle (SIMPLE's src/simple/baselines/psi0.py: from_psi0_upper_joints / STATE_SLICES); the replays used
# UPPER_NAMES (index before middle on both hands) and drove the sim's left index with the middle command and vice versa.
# observation.hand_joints of the psi0 sets is index-before-middle on both hands, so STATE keeps UPPER_NAMES.
# REPLAY_PSI0_LEFT_HAND=legacy restores the old (swapped) mapping.
PSI0_ACT_NAMES = (list(UPPER_NAMES) if os.environ.get("REPLAY_PSI0_LEFT_HAND", "psi0") == "legacy" else
                  [f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")] + list(UPPER_NAMES[7:]))
ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
TRACE_EVERY = 5                  # decimation of the kept trace (10 Hz)


# ----------------------------------------------------------------- helpers (shared with replay_in_scene.py)
def yaw_of_wxyz(q):
    from scipy.spatial.transform import Rotation as R
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("zyx")[0])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def zero_stale_leading_rows(rel, max_lead=10):
    rel = rel.copy()
    z = np.where(np.abs(rel[:max_lead]) < 1e-9)[0]
    if len(z) and z[0] > 0:
        rel[: z[0]] = 0.0
    return rel


def tilt_deg(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


class VirtualClock:
    """A 50 Hz clock for PicoDecoupledAgent.

    The agent's WBC interpolation is driven by time.monotonic() (pico_decoupled_agent.py: t_now), so a replay that
    runs faster than real time would interpolate the arm targets differently from the wall-clock-paced
    replay_in_scene.py.  Feeding the agent a clock that advances exactly 1/HZ per control tick reproduces a
    perfectly paced run -- without the sleeps, and without the pacing jitter that makes two paced runs differ.
    """

    def __init__(self, dt=1.0 / HZ):
        import time as _t
        self.dt = dt
        self.t = _t.monotonic()      # start at the real clock: a consumer this misses sees seconds of drift, not a 1000 s jump

    def install(self):
        """Put every loaded control-path module on this clock.  g1_decoupled_whole_body_policy stamps last_goal_time
        with ITS OWN time_module.monotonic() and compares it with the time= the agent passes in; with only the agent
        patched, the difference is (virtual - real), which is whatever the machine's uptime happens to be.  On a
        freshly booted machine it exceeds the 1 s teleop safety timeout and the WBC injects a safe goal every tick:
        the arms track, the robot never walks."""
        import sys, time as _t
        patched = []
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, "__file__", "") or ""
            if not any(k in f for k in ("decoupled_wbc/control", "gear_sonic", "simple/agents")):
                continue
            for attr in ("time", "time_module", "_time"):
                if getattr(mod, attr, None) is _t:
                    setattr(mod, attr, self)
                    patched.append(f"{name}.{attr}")
        return patched

    def tick(self):
        self.t += self.dt

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def sleep(self, s):
        pass

    def __getattr__(self, name):                       # anything else falls through to the real module
        import time as _t
        return getattr(_t, name)


def no_pico(clock=None):
    """Stop PicoDecoupledAgent from binding the VR TCP port / starting the streaming thread (unused in replay),
    and (optionally) put it on a virtual clock."""
    import simple.agents.pico_decoupled_agent as P
    from simple.teleop.pico.streaming import FrameBuffer

    def _init(self):
        self._streaming = None
        self._frame_buffer = FrameBuffer()
    P.PicoDecoupledAgent._init_pico_streamer = _init
    P.PicoDecoupledAgent._poll_pico_buttons = lambda self: None
    if clock is not None:
        P.time = clock
    return P.PicoDecoupledAgent


THIRD = dict(lookat=[0.35, -0.65, 0.45], distance=3.9, azimuth=-38.0, elevation=-22.0)   # table, robot, bottle and bin


class Video:
    """The replay_in_scene.py video: MuJoCo third-person | MuJoCo D455 | the real recording's ego camera."""

    def __init__(self, path, model, data, mujoco, session, ep, fps_real):
        import av
        from PIL import ImageDraw  # noqa: F401  (used in add())
        self.mujoco = mujoco
        self.d = data
        self.r3 = mujoco.Renderer(model, height=480, width=640)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = THIRD["lookat"]; self.cam.distance = THIRD["distance"]
        self.cam.azimuth = THIRD["azimuth"]; self.cam.elevation = THIRD["elevation"]
        self.c = av.open(str(path), "w")
        self.vs = self.c.add_stream("libx264", rate=25); self.vs.pix_fmt = "yuv420p"
        self.vs.options = {"crf": "26", "preset": "veryfast", "movflags": "+faststart"}
        self.vs.width, self.vs.height = 3 * 640, 480
        real = DATA / "real_recordings" / session / f"videos/chunk-000/observation.images.ego_view/episode_{ep:06d}.mp4"
        if not real.exists():
            real = DATA / "real_recordings" / session / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4"     # psi0 layout
        self.rc = av.open(str(real)) if real.exists() else None
        self.real_iter = self.rc.decode(video=0) if self.rc else None
        self.rs = {"frame": None, "next": None, "next_t": -1.0}
        self.fps_real = fps_real

    def _real_at(self, t):
        if self.real_iter is None:
            return None
        while (self.rs["next_t"] <= t and self.rs["next"] is not None) or self.rs["next_t"] < 0:
            if self.rs["next"] is not None:
                self.rs["frame"] = self.rs["next"]
            try:
                fr = next(self.real_iter)
                self.rs["next"] = fr.to_ndarray(format="rgb24")
                self.rs["next_t"] = float(fr.time if fr.time is not None else self.rs["next_t"] + 1 / self.fps_real)
            except StopIteration:
                self.rs["next"] = None; self.rs["next_t"] = float("inf"); break
        return self.rs["frame"]

    def add(self, obs, label, t_real):
        import av
        import PIL.Image as I
        from PIL import ImageDraw
        self.r3.update_scene(self.d, self.cam)
        v3 = self.r3.render().copy()
        ego = np.asarray(obs["head_stereo_left"])
        tiles = [v3]
        for im in (ego, self._real_at(t_real)):
            if im is None:
                im = np.zeros_like(ego)
            pad = v3.shape[0] - im.shape[0]
            im = np.pad(im, ((pad // 2, pad - pad // 2), (0, 0), (0, 0))) if pad > 0 else im[: v3.shape[0]]
            if im.shape[1] != 640:
                im = np.asarray(I.fromarray(im).resize((640, im.shape[0])))
            tiles.append(im)
        fr = np.concatenate(tiles, axis=1)
        fr = fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2]
        im = I.fromarray(fr); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, fr.shape[1], 22], fill=(0, 0, 0)); dr.text((6, 4), label, fill=(255, 255, 255))
        for x, t in ((6, "MuJoCo third-person (bottle/bin scene)"), (646, "MuJoCo D455 (sim)"), (1286, "REAL D455 (recording)")):
            dr.rectangle([x, 26, x + 230, 44], fill=(0, 0, 0)); dr.text((x + 4, 29), t, fill=(255, 255, 0))
        for pkt in self.vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
            self.c.mux(pkt)

    def close(self):
        for pkt in self.vs.encode():
            self.c.mux(pkt)
        self.c.close()
        if self.rc:
            self.rc.close()
        self.r3.close()


# ----------------------------------------------------------------- episode data
class Episode:
    """One recorded episode, resampled onto 50 Hz controller ticks."""

    def __init__(self, session, ep):
        S = DATA / "real_recordings" / session
        info = json.load(open(S / "meta/info.json"))
        df = pd.read_parquet(S / f"data/chunk-000/episode_{ep:06d}.parquet")
        if "observation.state" not in info["features"]:
            # psi0-format LeRobot dataset (2026-09-25: data/real_recordings/psi0/ChipCanToTrash_0225_keep97, the 97 kept episodes of
            # session 02-25-56 in the deployment format): 36-D action = [0:7 left hand, 7:14 right hand, 14:21 left arm, 21:28 right arm,
            # 28:31 torso rpy, 31 height, 32:36 vx vy flag target_yaw]; no base pose, so the commanded relative yaw stands in for the heading
            A = np.stack(df["action"]).astype(float)
            self.act_names = list(PSI0_ACT_NAMES)
            self.ACT = np.concatenate([A[:, 0:14], A[:, 14:28], A[:, 28:31]], axis=1)
            self.NAV = A[:, 32:36].copy(); self.NAV[:, 3] = zero_stale_leading_rows(self.NAV[:, 3])
            self.H = A[:, 31].copy()
            self.real_heading_rel = self.NAV[:, 3].copy()
            self.psi0 = True
        else:
            self.act_names = info["features"]["action"]["names"]
            self.ACT = np.stack(df["action"]).astype(float)
            self.NAV = np.stack(df["teleop.navigate_command"]).astype(float)
            self.NAV[:, 3] = zero_stale_leading_rows(self.NAV[:, 3])
            self.H = np.stack(df["teleop.base_height_command"]).astype(float).ravel()
            BP = np.stack(df["observation.base_pose"]).astype(float)
            h = np.array([yaw_of_wxyz(q) for q in BP[:, 3:7]])
            self.real_heading_rel = wrap(h - h[0])
            self.psi0 = False
        self.fps = float(info["fps"])
        self.n_rows = len(df)
        self.n_ticks = int(np.ceil(self.n_rows * HZ / self.fps))
        self.ep = ep

    def row_at(self, t):
        return min(int(t * self.fps), self.n_rows - 1)

    def act_at(self, t):
        f = min(t * self.fps, self.n_rows - 1)
        i0 = int(f); i1 = min(i0 + 1, self.n_rows - 1); a = f - i0
        return (1 - a) * self.ACT[i0] + a * self.ACT[i1]


# ----------------------------------------------------------------- the runner
class Runner:
    """Holds one env + agent and replays any number of episodes through it."""

    def __init__(self, robot_to_edge, bin_xy, bottle_mass, bottle_xy=None, virtual_clock=True, fast_obs=False):
        os.environ["BOTTLE_BIN_ROBOT_TO_EDGE"] = f"{robot_to_edge:.4f}"
        os.environ["BOTTLE_BIN_BOTTLE_MASS"] = f"{bottle_mass:.4f}"
        if bin_xy is not None:
            os.environ["BOTTLE_BIN_BIN_XY"] = f"{bin_xy[0]:.4f},{bin_xy[1]:.4f}"
        if bottle_xy is not None:
            os.environ["BOTTLE_BIN_BOTTLE_XY"] = f"{bottle_xy[0]:.4f},{bottle_xy[1]:.4f}"
        import gymnasium as gym
        import mujoco
        import simple.tasks.g1_wholebody_bottle_bin_teleop as T
        sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import replay_physics; replay_physics.pin_task(T)   # the validated robot / reset configuration
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        self.mujoco = mujoco
        self.T = T
        cfg = SimLoopConfig().load_wbc_yaml(); cfg["ENV_NAME"] = "simple"
        self.env = gym.make(T.ENV_ID, sim_mode="mujoco", render_hz=HZ, physics_dt=cfg["SIMULATE_DT"], headless=True,
                            max_episode_steps=10 ** 6, sonic_config=cfg, target=T.TARGET, dr_level=0, success_criteria=1e9)
        self.task = self.env.unwrapped.task
        self.robot = self.task.robot
        self.clock = VirtualClock() if virtual_clock else None
        PicoDecoupledAgent = no_pico(self.clock)
        self.agent = PicoDecoupledAgent(self.robot)
        self.agent._wbc_policy.lower_body_policy.use_policy_action = True
        if self.clock is not None:
            self.clock_patched = self.clock.install()
            print(f"[sweep] virtual clock installed in {len(self.clock_patched)} modules:", ", ".join(self.clock_patched), flush=True)
        self.tp = self.agent._teleop_policy
        self.rm = self.agent._dwbc_robot_model
        inv = {v: k for k, v in self.rm.joint_to_dof_index.items()}
        self.up_names = [inv[i] for i in self.rm.get_joint_group_indices("upper_body")]
        self.default_up = np.asarray(self.rm.default_body_pose, dtype=float)[self.rm.get_joint_group_indices("upper_body")]
        self.cur = {"epi": None, "i": None}
        self.tp.get_action = self._replay_get_action
        self.bound = False
        self.scene = None
        self.fast_obs = fast_obs           # freeze the camera images: the WBC is proprioceptive, so a sweep without
        self._frozen = None                # video never looks at them, and rendering them is most of the step cost

    def _bind(self):
        """(Re-)look up the MuJoCo handles: the engine rebuilds the model on every env.reset()."""
        mj = self.mujoco
        self.m, self.d = self.robot.mjModel, self.robot.mjData
        self.T.recolor_hands(self.m)
        bid = lambda n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, n)
        self.tgt_label = self.task.layout.actors["target"].asset.label
        self.tgt_body = bid(self.tgt_label)
        self.bin_body = bid("trash_bin")
        self.palm_body = bid("right_wrist_yaw_link")
        self.tip_bodies = {n: bid(n) for n in ("right_hand_index_1_link", "right_hand_thumb_2_link")}
        self.tgt_qadr = self._free_joint_qposadr(self.tgt_body)
        self.feet = {n: bid(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")}
        pelvis = bid("pelvis")
        self.robot_root = int(self.m.body_rootid[pelvis])
        self.bin_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.bin_body} if self.bin_body >= 0 else set()
        _tb = bid("table")                                              # 2026-09-28: legs on the TABLE count too (walk completeness)
        self.table_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == _tb} if _tb >= 0 else set()
        self.bound = True
        if self.scene is None:
            self.scene = self._scene_checks()

    def _free_joint_qposadr(self, body):
        mj = self.mujoco
        self.tgt_dofadr = None
        for j in range(self.m.njnt):
            if self.m.jnt_bodyid[j] == body and self.m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                self.tgt_dofadr = int(self.m.jnt_dofadr[j])
                return int(self.m.jnt_qposadr[j])
        return None

    def _scene_checks(self):
        T = self.T; d = self.d
        table = self.task.layout.scene.table
        return {"robot_to_edge_feet": round(float(T.ROBOT_TO_EDGE), 4), "pelvis_to_edge": round(float(T.L["pelvis_to_edge"]), 4),
                "bottle_mass_kg": T.BOTTLE_MASS, "target_label": self.tgt_label,
                "layout_bottle_xy": [round(T.L["bottle_x"], 4), round(T.L["bottle_y"], 4)],
                "layout_bin_xy": [round(T.L["bin_x"], 4), round(T.L["bin_y"], 4)],
                "bin_wdh": [T.BS.BIN_W, T.BS.BIN_D, T.BS.BIN_H], "bin_wall": T.BS.BIN_T, "bottle_r": T.BS.BOTTLE_R,
                "bin_yaw_deg": T.BIN_YAW_DEG,
                "table_top_z": round(float(table.pose.position[2] + 0.5 * table.size[2]), 4),
                "table_near_edge_x": round(float(table.pose.position[0] - 0.5 * table.size[0]), 4)}

    # -------------------------------------------------------------- teleop hook
    def _replay_get_action(self):
        epi, i = self.cur["epi"], self.cur["i"]
        if epi is None or i is None:
            # settle phase: hold the FIRST recorded upper-body pose (not the controller default) so the arm is already where the
            # recording starts; 31 of the 35 episodes that never grasped close the hand before 3 s, while the arm was still
            # ramping from the default pose (2026-09-25)
            up = epi.act_at(0.0)[self.up_from_act].copy() if (epi is not None and getattr(self, "up_from_act", None) is not None) else self.default_up.copy()
            h = float(epi.H[0]) if epi is not None else 0.74
            return {"target_upper_body_pose": up, "navigate_cmd": [0.0, 0.0, 0.0, 0.0],
                    "base_height_command": h, "wrist_pose": np.zeros(14), "toggle_policy_action": False,
                    "toggle_data_collection": False, "toggle_data_abort": False}
        t = i / HZ; r = epi.row_at(t)
        up_from_act = self.up_from_act
        return {"target_upper_body_pose": epi.act_at(t)[up_from_act].copy(), "navigate_cmd": epi.NAV[r].tolist(),
                "base_height_command": float(epi.H[r]), "wrist_pose": np.zeros(14), "toggle_policy_action": False,
                "toggle_data_collection": False, "toggle_data_abort": False}

    # -------------------------------------------------------------- one job
    def place_bottle(self, xy):
        """Re-place the bottle (free joint) without rebuilding the scene; z and orientation are kept."""
        if xy is None or self.tgt_qadr is None:
            return
        a = self.tgt_qadr
        self.d.qpos[a: a + 2] = xy
        if self.tgt_dofadr is not None:
            self.d.qvel[self.tgt_dofadr: self.tgt_dofadr + 6] = 0.0
        self.mujoco.mj_forward(self.m, self.d)

    def spawn_contacts(self):
        """Geoms touching the can right after it is placed.  At the start pose the robot's open thumbs sit at
        (0.264, +-0.061, 0.888) -- inside the can's column for part of the table -- and a can spawned against a
        thumb is flung off the table, which is a placement that does not exist, not a failed grasp."""
        mj = self.mujoco
        cangeoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.tgt_body}
        hits = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            other = None
            if c.geom1 in cangeoms and c.geom2 not in cangeoms:
                other = c.geom2
            elif c.geom2 in cangeoms and c.geom1 not in cangeoms:
                other = c.geom1
            if other is None:
                continue
            nm = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[other])) or "world"
            hits.append({"body": nm, "dist": round(float(c.dist), 5)})
        return hits

    def robot_bin_leg_contact(self):
        """True while a leg / pelvis / torso geom touches the bin (the walk runs into it; hands and arms brushing the rim while
        dropping the can do not count -- same rule as bowl_sink's counters, 2026-09-27)."""
        import mujoco
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.bin_geoms or g2 in self.bin_geoms:
                other = g2 if g1 in self.bin_geoms else g1
                b = self.m.geom_bodyid[other]
                if int(self.m.body_rootid[b]) == self.robot_root and any(k in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "") for k in ("knee", "hip", "ankle", "pelvis", "torso", "waist", "foot")):
                    return True
        return False

    def robot_leg_contact(self, geoms):
        """True while a leg / pelvis / torso geom touches any of `geoms`."""
        import mujoco
        for i in range(self.d.ncon):
            c = self.d.contact[i]; g1, g2 = c.geom1, c.geom2
            if g1 in geoms or g2 in geoms:
                b = self.m.geom_bodyid[g2 if g1 in geoms else g1]
                if int(self.m.body_rootid[b]) == self.robot_root and any(k in (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "") for k in ("knee", "hip", "ankle", "pelvis", "torso", "waist", "foot", "toe")):
                    return True
        return False

    def robot_bin_contact(self):
        """True while any robot geom touches the bin."""
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.bin_geoms or g2 in self.bin_geoms:
                other = g2 if g1 in self.bin_geoms else g1
                if int(self.m.body_rootid[self.m.geom_bodyid[other]]) == self.robot_root:
                    return True
        return False

    def move_bin(self, xy, yaw_deg=None):
        """Re-place (and optionally turn) the static bin by editing its body pose."""
        self.bin_yaw_deg = float(yaw_deg) if yaw_deg is not None else float(self.T.BIN_YAW_DEG)
        if self.bin_body < 0:
            return
        if xy is not None:
            self.m.body_pos[self.bin_body][:2] = xy
        if yaw_deg is not None:
            h = np.radians(yaw_deg) / 2
            self.m.body_quat[self.bin_body] = [np.cos(h), 0.0, 0.0, np.sin(h)]
        self.mujoco.mj_forward(self.m, self.d)

    def tick(self):
        if self.clock is not None:
            self.clock.tick()
        elif self.pace_t0 is not None:                   # wall-clock pacing, as in replay_in_scene.py
            self.pace_i += 1
            sl = self.pace_t0 + self.pace_i / HZ - time.monotonic()
            if sl > 0:
                time.sleep(sl)

    def run(self, epi: Episode, bottle_xy=None, bin_xy=None, keep_trace=True, video_path=None, session=None, bin_yaw=None, grasp_assist=False, drop_assist=False):
        mj = self.mujoco
        T = self.T; env = self.env
        self.cur = {"epi": None, "i": None}
        obs, sinfo = env.reset()
        self._bind()
        if self.fast_obs and self._frozen is None:
            u = env.unwrapped
            self._frozen = u._render_frame()
            u._render_frame = lambda *args, **kw: self._frozen
        self.move_bin(bin_xy, bin_yaw)
        self.place_bottle(bottle_xy)
        hits = self.spawn_contacts()
        self.spawn_hits = [h for h in hits if not any(k in h["body"] for k in ("table", "floor", "world", "bin", "cover", "leg"))]
        self.up_from_act = np.array([epi.act_names.index(nm) for nm in self.up_names])
        if self.robot.elastic_band is not None:
            self.robot.elastic_band.enable = False
        self.agent._dropping = False
        self.agent.reset_policy()
        self.agent._wbc_policy.lower_body_policy.use_policy_action = True
        EXPORT_GATE["on"] = False
        self.tp.is_active = True                          # 2026-09-25: teleop active during the settle so the hook can hold the FIRST recorded pose
        self.cur["epi"] = epi; self.cur["i"] = None       # (i = None -> _replay_get_action returns row 0's upper body, zero nav)
        d = self.d
        self.pace_i, self.pace_t0 = 0, (None if self.clock is not None else time.monotonic())
        t_wall = time.monotonic()
        k_stab = None
        for k in range(STABILIZE_TICKS + PREPOSE_TICKS):  # let the robot settle on its feet before the recording starts ...
            self.tick()
            action = self.agent.get_action(obs, instruction=self.task.instruction, privileged_info=sinfo)
            obs, *_, sinfo = env.step(action)
            if k == 100 and PREPOSE_TICKS > 0 and hasattr(self.robot, "_stabilized"):
                self.robot._stabilized = True             # the base never gets below the velocity threshold while the policy stands: latch it so the
            if self.robot.stabilized and k > 100 and k_stab is None:   # agent stops returning the stabilise (default-pose) action and follows the hook
                k_stab = k                                # ... then keep stepping so the (now active) hook brings the arm to row 0's pose:
            if k_stab is not None and k >= k_stab + PREPOSE_TICKS:   # the agent returns the stabilise action (default pose) until stabilized
                break
        self.gate_t0 = (k + 1) / HZ                      # the task's gate clock started at reset; row 0 of the recording is here
        if os.environ.get("REPLAY_PREPOSE_DEBUG") == "1":
            _tgt = epi.act_at(0.0)[self.up_from_act]; _q = d.qpos[self.mj_arm] if hasattr(self, "mj_arm") else None
            print(f"[prepose] settle ended at k={k} (stabilized at {k_stab}); tp.active={self.tp.is_active} palm={d.xpos[self.palm_body].round(3).tolist()} "
                  f"row0 right-arm target={np.round(_tgt[21:28], 3).tolist()} actual right arm q={np.round(d.qpos[[self.m.jnt_qposadr[self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS[7:]]], 3).tolist()}", flush=True)
        self.tp.is_active = True
        heading0 = yaw_of_wxyz(d.qpos[3:7]); xy0 = d.qpos[:2].copy()
        tgt0 = d.xpos[self.tgt_body].copy()
        bin_xyz = d.xpos[self.bin_body].copy() if self.bin_body >= 0 else None
        vid = Video(video_path, self.m, d, mj, session, epi.ep, epi.fps) if video_path else None
        trace, z_hist, xyz_hist, tilt_hist = [], [], [], []
        contact_ticks, first_contact, feet_path, base_path = 0, None, [], []
        self._leg_ticks = 0; self._table_leg_ticks = 0
        self.cur["epi"] = epi
        n = epi.n_ticks
        # --- grasp assist (2026-09-27, as in coffee_cart / bowl_sink): job "grasp_assist": once the recorded hand has closed on the
        # bottle and a finger touches it (or the fingers are within 10 cm of its middle), the bottle rides with the palm (upright)
        # until the recorded release row; the result records "assist_captured".
        _ic = epi.act_names.index("right_hand_index_0_joint") if "right_hand_index_0_joint" in epi.act_names else 10
        _ri = epi.ACT[:, _ic]; _cl = np.where((_ri[1:] > 0.6) & (_ri[:-1] <= 0.6))[0]
        GRAB_ROW = int(_cl[0]) + 1 if len(_cl) else len(_ri) - 1
        _closed = np.where(_ri > 0.6)[0]; OPEN_ROW = int(_closed[-1]) + 1 if len(_closed) else len(_ri) - 1   # end of the LAST closed run
        tgt_jid = int(self.m.body_jntadr[self.tgt_body]); tgt_qadr = int(self.m.jnt_qposadr[tgt_jid]); tgt_dof = int(self.m.jnt_dofadr[tgt_jid])
        hand_geoms_ga = {g for g in range(self.m.ngeom) if "_hand_" in (mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]) or "")}
        tgt_geoms_ga = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.tgt_body}
        ga = {"on": False, "done": False, "captured_row": None, "off": None, "yaw_rel": 0.0}
        self._ga = ga; self._grab_row, self._open_row, self._grasp_assist = int(GRAB_ROW), int(OPEN_ROW), bool(grasp_assist); self._drop_assist = bool(drop_assist)
        if grasp_assist:
            print(f"[sweep] grasp assist on: recorded closure row {GRAB_ROW}, release row {OPEN_ROW}", flush=True)
        bin_top = float(T.BS.BIN_H)
        def assist(i):
            if ga.get("drop", 0) > 0:                  # 2026-09-27 drop assist: straight down, upright, until inside the bin (the opening
                ga["drop"] -= 1                        # fingers made the bottle tumble onto the rim in 57/88)
                d.qpos[tgt_qadr:tgt_qadr + 2] = ga["drop_xy"]; yw = yaw_of_wxyz(d.qpos[tgt_qadr + 3:tgt_qadr + 7])
                d.qpos[tgt_qadr + 3:tgt_qadr + 7] = [np.cos(yw / 2), 0.0, 0.0, np.sin(yw / 2)]
                d.qvel[tgt_dof:tgt_dof + 2] = 0.0; d.qvel[tgt_dof + 3:tgt_dof + 6] = 0.0
                if d.qpos[tgt_qadr + 2] < bin_top - 0.06: ga["drop"] = 0
                return
            if not grasp_assist or ga["done"]:
                return
            r = epi.row_at(i / HZ)
            if not ga["on"]:
                if r < GRAB_ROW or r > GRAB_ROW + int(2.5 * epi.fps):
                    return
                touched = any((c.geom1 in tgt_geoms_ga and c.geom2 in hand_geoms_ga) or (c.geom2 in tgt_geoms_ga and c.geom1 in hand_geoms_ga)
                              for c in d.contact[:d.ncon])
                mid = (d.xpos[self.tip_bodies["right_hand_index_1_link"]] + d.xpos[self.tip_bodies["right_hand_thumb_2_link"]]) / 2
                if not (touched or np.linalg.norm(mid - (d.xpos[self.tgt_body] + np.array([0.0, 0.0, 0.06]))) < 0.10):
                    return
                pyaw = yaw_of_wxyz(d.xquat[self.palm_body]); cp, sp = np.cos(-pyaw), np.sin(-pyaw)
                dv = d.qpos[tgt_qadr:tgt_qadr + 3] - d.xpos[self.palm_body]
                ga["off"] = np.array([cp * dv[0] - sp * dv[1], sp * dv[0] + cp * dv[1], dv[2]])
                ga["yaw_rel"] = yaw_of_wxyz(d.qpos[tgt_qadr + 3:tgt_qadr + 7]) - pyaw
                ga["on"] = True; ga["captured_row"] = int(r)
                print(f"[sweep] grasp assist: captured at row {r} (touched={touched}, offset {ga['off'].round(3).tolist()})", flush=True)
            if r >= OPEN_ROW:
                ga["on"] = False; ga["done"] = True; d.qvel[tgt_dof:tgt_dof + 6] = 0.0
                if drop_assist: ga["drop"] = int(1.5 * HZ); ga["drop_xy"] = d.qpos[tgt_qadr:tgt_qadr + 2].copy()
                print(f"[sweep] grasp assist: released at row {r}" + (" (straight drop)" if drop_assist else ""), flush=True)
                return
            pyaw = yaw_of_wxyz(d.xquat[self.palm_body]); cp, sp = np.cos(pyaw), np.sin(pyaw); o = ga["off"]
            d.qpos[tgt_qadr:tgt_qadr + 3] = d.xpos[self.palm_body] + np.array([cp * o[0] - sp * o[1], sp * o[0] + cp * o[1], o[2]])
            cy = pyaw + ga["yaw_rel"]; d.qpos[tgt_qadr + 3:tgt_qadr + 7] = [np.cos(cy / 2), 0.0, 0.0, np.sin(cy / 2)]
            d.qvel[tgt_dof:tgt_dof + 6] = 0.0
        EXPORT_GATE["on"] = True                          # the recorded rows start here: dataset frames from now on
        QREC = os.environ.get("REPLAY_QPOS_OUT"); qrec = []    # 2026-09-28: exact state recording for offline videos
        if QREC:
            QREC = QREC.replace("{ep}", f"{epi.ep:03d}"); os.makedirs(os.path.dirname(QREC) or ".", exist_ok=True); mj.mj_saveModel(self.m, QREC[:-4] + ".mjb", None)
        for i in range(n):
            self.cur["i"] = i
            self.tick()
            if i == 0:
                self.agent._wbc_policy.handle_keyboard_button("c")       # re-arm the relative yaw dial, like the A press
            action = self.agent.get_action(obs, instruction=self.task.instruction, privileged_info=sinfo)
            obs, _r, _te, _tr, sinfo = env.step(action)
            assist(i)
            tz = d.xpos[self.tgt_body].copy()
            z_hist.append(float(tz[2])); xyz_hist.append(tz); tilt_hist.append(tilt_deg(d.xquat[self.tgt_body]))
            if QREC and i % 2 == 0: qrec.append((d.qpos.copy(), i / HZ))
            if self.bin_geoms and self.robot_bin_contact():
                contact_ticks += 1
                if first_contact is None:
                    first_contact = i
                if self.robot_bin_leg_contact():
                    self._leg_ticks += 1
            if getattr(self, "table_geoms", None) and self.robot_leg_contact(self.table_geoms):
                self._table_leg_ticks += 1
            if i % TRACE_EVERY == 0:
                base_path.append([round(i / HZ, 2)] + [round(float(v), 3) for v in (d.qpos[:2] - xy0)] + [round(float(wrap(yaw_of_wxyz(d.qpos[3:7]) - heading0)), 3)])
                feet_path.append([round(i / HZ, 2)] + [round(float(v), 3) for f in self.feet.values() for v in d.xpos[f][:2]])
            if vid is not None and i % 2 == 0:
                ri = epi.row_at(i / HZ)
                vid.add(obs, f"{session} ep {epi.ep}  t={i / HZ:5.1f}s  vx {epi.NAV[ri, 0]:+.2f} vy {epi.NAV[ri, 1]:+.2f} "
                             f"turn {epi.NAV[ri, 2]:+.1f} rel yaw {epi.NAV[ri, 3]:+.2f} | bottle lift "
                             f"{d.xpos[self.tgt_body][2] - tgt0[2]:+.2f} m", i / HZ)
            if keep_trace and i % TRACE_EVERY == 0:
                t_i = i / HZ; ri = epi.row_at(t_i)
                trace.append({"t": round(t_i, 3), "sim_heading_rel": round(float(wrap(yaw_of_wxyz(d.qpos[3:7]) - heading0)), 4),
                              "feet": feet_path[-1][1:],
                              "real_heading_rel": round(float(epi.real_heading_rel[ri]), 4),
                              "sim_xy": (d.qpos[:2] - xy0).round(4).tolist(),
                              "bottle": np.round(tz, 4).tolist(), "tilt": round(tilt_hist[-1], 1),
                              "palm": d.xpos[self.palm_body].round(4).tolist()})
        n_settle = int(SETTLE_S * HZ)
        for j in range(n_settle):
            self.tick()
            action = self.agent.get_action(obs, instruction=self.task.instruction, privileged_info=sinfo)
            obs, _r, _te, _tr, sinfo = env.step(action)
            assist(n + j)
            tz = d.xpos[self.tgt_body].copy()
            z_hist.append(float(tz[2])); xyz_hist.append(tz); tilt_hist.append(tilt_deg(d.xquat[self.tgt_body]))
            if QREC and j % 2 == 0: qrec.append((d.qpos.copy(), (n + j) / HZ))
            if vid is not None and j % 2 == 0:
                vid.add(obs, f"{session} ep {epi.ep}  settle {j / HZ:4.1f}s after the last row | bottle "
                             f"{np.round(d.xpos[self.tgt_body], 2).tolist()} tilt {tilt_hist[-1]:.0f} deg", n / HZ)
            if keep_trace and (n + j) % TRACE_EVERY == 0:
                trace.append({"t": round((n + j) / HZ, 3), "settle": True, "bottle": np.round(tz, 4).tolist(),
                              "tilt": round(tilt_hist[-1], 1), "sim_xy": (d.qpos[:2] - xy0).round(4).tolist()})
        if vid is not None:
            vid.close()
        if QREC and qrec:
            np.savez_compressed(QREC, qpos=np.array([q for q, _ in qrec]), t=np.array([t for _, t in qrec]), ep=epi.ep, fps=25)
        wall = time.monotonic() - t_wall
        out = self._metrics(epi, np.array(z_hist), np.array(xyz_hist), np.array(tilt_hist), tgt0, bin_xyz, xy0,
                            heading0, trace, wall, bottle_xy, bin_xy, n)
        out["gates"] = sinfo.get("task_progress")                      # the task's own checkers (bottle_bin_task.py)
        out["gate_t0_offset_s"] = round(self.gate_t0, 2)               # subtract from the gates' t_* to get recording time
        out["robot_bin_contact_ticks"] = int(contact_ticks); out["robot_bin_leg_ticks"] = int(getattr(self, "_leg_ticks", 0)); out["robot_table_leg_ticks"] = int(getattr(self, "_table_leg_ticks", 0))
        out["robot_bin_contact_s"] = round(contact_ticks / HZ, 2)
        out["first_robot_bin_contact_t"] = None if first_contact is None else round(first_contact / HZ, 2)
        rel0 = out.get("release")
        out["contact_before_release"] = bool(first_contact is not None and (rel0 is None or first_contact / HZ < rel0["t"]))
        out["feet_path"] = feet_path                                   # [t, lx, ly, rx, ry] at 10 Hz, world = start frame
        rel = out.get("release")
        if rel:
            k = min(range(len(feet_path)), key=lambda j: abs(feet_path[j][0] - rel["t"]))
            out["feet_at_release"] = feet_path[k][1:]
            out["base_at_release"] = base_path[k][1:]                  # [x, y, heading_rel]
        out["base_path"] = base_path
        out["base_end_heading_rel"] = out.get("base_end_heading_rel")
        return out

    def _metrics(self, epi, z, xyz, tilt, tgt0, bin_xyz, xy0, heading0, trace, wall, bottle_xy, bin_xy, n_ticks):
        T = self.T; d = self.d
        lift = z - tgt0[2]
        k = int(lift.argmax())
        k_rel = None
        for j in range(k, len(z) - 5):                       # release = free fall: > 5 cm drop in 0.1 s
            if z[j] - z[j + 5] > 0.05:
                k_rel = j
                break
        release = None
        if k_rel is not None:
            release = {"t": round(k_rel / HZ, 3), "xy": [round(float(v), 4) for v in xyz[k_rel][:2]],
                       "z": round(float(xyz[k_rel][2]), 4)}
        end = xyz[-1]
        yaw = getattr(self, "bin_yaw_deg", T.BIN_YAW_DEG)
        along_x, along_y = (T.BS.BIN_W, T.BS.BIN_D) if abs(yaw) % 180 == 90 else (T.BS.BIN_D, T.BS.BIN_W)
        half = [along_x / 2 - T.BS.BIN_T - T.BS.BOTTLE_R, along_y / 2 - T.BS.BIN_T - T.BS.BOTTLE_R]
        out = {"ep": epi.ep, "n_ticks": int(n_ticks), "seconds": round(n_ticks / HZ, 2), "wall_s": round(wall, 1),
               "bottle_xy_cmd": list(bottle_xy) if bottle_xy is not None else None,
               "bin_xy_cmd": list(bin_xy) if bin_xy is not None else None,
               "bin_yaw_deg": yaw,
               "bottle_start": [round(float(v), 4) for v in tgt0],
               "bottle_end": [round(float(v), 4) for v in end], "bottle_end_tilt_deg": round(float(tilt[-1]), 1),
               "max_lift_m": round(float(lift[k]), 4), "t_max_lift": round(k / HZ, 2),
               "tilt_at_max_lift_deg": round(float(tilt[k]), 1),
               "grasp_ok": bool(lift[k] >= 0.10 and tilt[k] < 60),
               "lifted_20cm": bool(lift.max() >= 0.20),
               "release": release, "bin_xyz": [round(float(v), 4) for v in bin_xyz] if bin_xyz is not None else None,
               "half_opening": [round(float(v), 4) for v in half],
               "base_end_xy": (d.qpos[:2] - xy0).round(4).tolist(),
               "base_end_heading_rel": round(float(wrap(yaw_of_wxyz(d.qpos[3:7]) - heading0)), 4),
               "real_end_heading_rel": round(float(epi.real_heading_rel[-1]), 4),
               "assist_captured": self._ga["captured_row"], "grab_row": self._grab_row, "open_row": self._open_row, "grasp_assist": self._grasp_assist,
               "drop_assist": bool(getattr(self, "_drop_assist", False))}
        if bin_xyz is not None:
            bx, by = bin_xyz[:2]
            out["end_minus_bin"] = [round(float(end[0] - bx), 4), round(float(end[1] - by), 4), round(float(end[2]), 4)]
            out["in_bin"] = bool(abs(end[0] - bx) < along_x / 2 and abs(end[1] - by) < along_y / 2 and end[2] < T.BS.BIN_H)
            if release:
                rx, ry = release["xy"]
                out["release_inside_opening"] = bool(abs(rx - bx) <= half[0] and abs(ry - by) <= half[1])
        out["spawn_hits"] = self.spawn_hits
        out["spawn_ok"] = not self.spawn_hits             # False = the can was spawned inside the robot: not a valid placement
        out["success"] = bool(out.get("in_bin", False) and (out["grasp_ok"] or self._ga["captured_row"] is not None))   # 2026-09-27: assisted rides lift little
        if trace:
            out["trace"] = trace
        return out

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="2026-09-17-02-25-56-G1-sim")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--jobs", help="JSON list of {ep, bottle_xy?, bin_xy?, variant?}")
    ap.add_argument("--shard", default=None, help="i/n: take every n-th job starting at i")
    ap.add_argument("--robot-to-edge", type=float, default=0.03)
    ap.add_argument("--bottle-mass", type=float, default=0.5)
    ap.add_argument("--bottle-xy", type=float, nargs=2, default=None)
    ap.add_argument("--bin-xy", type=float, nargs=2, default=None)
    ap.add_argument("--bin-yaw", type=float, default=None, help="bin yaw in degrees (layout.json's -90 = 36 cm side toward the table)")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--fast-obs", action="store_true", help="freeze the rendered camera images (no video): ~3x faster")
    ap.add_argument("--video-dir", default=None, help="also write replay_<session>_ep<N>_<tag>.mp4 here")
    ap.add_argument("--wall-clock", action="store_true", help="pace at 50 Hz on the real clock (as replay_in_scene.py does) instead of the virtual clock")
    ap.add_argument("--self-check", action="store_true", help="run the first job twice and report the difference")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    jobs = []
    if a.jobs:
        jobs = json.load(open(a.jobs))
    else:
        jobs = [{"ep": e} for e in (a.episodes or [])]
    if a.shard:
        i, nsh = (int(v) for v in a.shard.split("/"))
        jobs = jobs[i::nsh]
    out_dir = OUT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def out_path(job):
        v = job.get("variant")
        return out_dir / (f"ep{job['ep']:03d}" + (f"_{v}" if v else "") + ".json")

    if a.skip_existing:
        jobs = [j for j in jobs if not out_path(j).exists()]
    if not jobs:
        print("[sweep] nothing to do", flush=True)
        os._exit(0)

    print(f"[sweep] tag={a.tag} jobs={len(jobs)} robot_to_edge={a.robot_to_edge} bin_xy={a.bin_xy} "
          f"bottle_xy={a.bottle_xy} mass={a.bottle_mass}", flush=True)
    R = Runner(a.robot_to_edge, a.bin_xy, a.bottle_mass, a.bottle_xy, virtual_clock=not a.wall_clock,
               fast_obs=a.fast_obs and not a.video_dir)
    t0 = time.monotonic()
    for k, job in enumerate(jobs):
        epi = Episode(a.session, job["ep"])
        bxy = job.get("bottle_xy")
        binxy = job.get("bin_xy", a.bin_xy)
        vpath = None
        if a.video_dir:
            vd = Path(a.video_dir); vd.mkdir(parents=True, exist_ok=True)
            vpath = vd / f"replay_{a.session.replace('/', '_')}_ep{job['ep']:03d}_{a.tag}.mp4"
        res = R.run(epi, bottle_xy=bxy, bin_xy=binxy, keep_trace=not a.no_trace, video_path=vpath, session=a.session,
                    bin_yaw=job.get("bin_yaw", a.bin_yaw), grasp_assist=bool(job.get("grasp_assist", os.environ.get("REPLAY_GRASP_ASSIST") == "1")),
                    drop_assist=bool(job.get("drop_assist", os.environ.get("REPLAY_DROP_ASSIST") == "1")))
        if vpath:
            res["video"] = vpath.name
        res["tag"] = a.tag; res["session"] = a.session; res["variant"] = job.get("variant")
        res["scene"] = R.scene
        if k == 0:
            print("[sweep] scene:", json.dumps(R.scene), flush=True)
        json.dump(res, open(out_path(job), "w"))
        brief = {kk: res[kk] for kk in ("ep", "seconds", "wall_s", "max_lift_m", "grasp_ok", "in_bin", "success") if kk in res}
        g = res.get("gates") or {}
        gates = {kk: g.get(kk) for kk in ("grasped", "at_bin", "placed", "robot_touched_bin", "t_grasped", "t_at_bin", "t_placed")}
        print(f"[sweep] {k + 1}/{len(jobs)} {brief} release={res['release']} gates={gates}", flush=True)
        if a.self_check and k == 0:
            res2 = R.run(epi, bottle_xy=bxy, bin_xy=binxy, keep_trace=False)
            print("[sweep] self-check repeat:", json.dumps({kk: res2[kk] for kk in ("max_lift_m", "grasp_ok", "release", "bottle_end")}), flush=True)
    print(f"[sweep] done {len(jobs)} jobs in {time.monotonic() - t0:.0f} s", flush=True)
    R.close()
    os._exit(0)                                              # EGL/SDK teardown aborts otherwise


if __name__ == "__main__":
    main()
