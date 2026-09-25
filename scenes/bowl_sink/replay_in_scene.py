#!/usr/bin/env python
"""Replay a real run through the decoupled-WBC stack in MuJoCo, inside the bowl-to-sink kitchen scene.

Two sources: a holobrain deployment client log (--client-log <actions_*.jsonl>, episodes by the client's episode numbers;
the executed 36-D rows at 30 Hz, the client's head-camera AVI by frame index, IMU yaw from probe/robot_watch.jsonl) or a
LeRobot recording (--session <name> under SIMPLE/data/real_recordings). The recorded upper-body targets, navigation command
and base height drive PicoDecoupledAgent tick by tick (rows resampled onto its 50 Hz ticks); the lower body walks by itself.

Outputs replay/: replay_<run>_ep<N>_<tag>.mp4 (third-person | sim D455 | real D455), trace_ep<N>_<tag>.json, index.html.
  python replay_in_scene.py --client-log .../runs/20260921_172723_bowltosink_c96/client/actions_taskX_20260921_172803.jsonl --episodes 7 8 10 12 14
  python replay_in_scene.py --probe
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, base64, io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image as I
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path(__file__).resolve().parents[2]   # the SIMPLE checkout (scenes/<kit>/ lives inside it)
OUT = HERE / "replay"
HZ = 50
ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
KEY_JOINTS = ["left_shoulder_pitch_joint", "left_elbow_joint", "right_shoulder_pitch_joint", "right_elbow_joint",
              "right_wrist_pitch_joint", "right_hand_index_0_joint", "left_hand_index_0_joint"]
THIRD = dict(lookat=[0.1, -0.5, 0.6], distance=4.6, azimuth=-35.0, elevation=-27.0)      # fixed: both counters, robot, bowl, sink
UPPER_NAMES = ([f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
               + [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
               + [f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"])          # the row's torso rpy command drives the waist
# the 36-D psi0 action row: [0:7 left hand, 7:14 right hand, 14:21 left arm, 21:28 right arm, 28:31 rpy, 31 height, 32:36 vx vy flag target_yaw]
SETTLE_S = float(os.environ.get("REPLAY_SETTLE_S", "1.5"))                                  # extra seconds after the last row
NAV_GAIN = float(os.environ.get("REPLAY_NAV_GAIN", "1.0"))                                  # scales the recorded vx, vy (the sim controller walks slower than the robot)
WALK_STRETCH = float(os.environ.get("REPLAY_WALK_STRETCH", "1.0"))                          # rows with a walking command last this much longer in sim (speed saturates)


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


class VirtualClock:
    """A 50 Hz clock for the control path (from ../bottle_bin/sweep_replay.py).  The agent's WBC interpolation runs on
    time.monotonic(), so a wall-clock-paced replay depends on machine load and two runs differ.  Advancing this clock by
    exactly 1/HZ per control tick reproduces a perfectly paced run, deterministically, and needs no sleeps."""

    def __init__(self, dt=1.0 / HZ):
        import time as _t
        self.dt = dt; self.t = _t.monotonic()

    def install(self):
        import sys as _s, time as _t
        patched = []
        for name, mod in list(_s.modules.items()):
            f = getattr(mod, "__file__", "") or ""
            if not any(k in f for k in ("decoupled_wbc/control", "gear_sonic", "simple/agents")):
                continue
            for attr in ("time", "time_module", "_time"):
                if getattr(mod, attr, None) is _t:
                    setattr(mod, attr, self); patched.append(f"{name}.{attr}")
        return patched

    def tick(self): self.t += self.dt
    def monotonic(self): return self.t
    def time(self): return self.t
    def sleep(self, s): pass
    def __getattr__(self, name):
        import time as _t
        return getattr(_t, name)


FAST = os.environ.get("REPLAY_FAST", "0") == "1"          # --fast: virtual clock, no pacing, no video


def make_env():
    import gymnasium as gym
    import bowl_sink_task as T
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    cfg = SimLoopConfig().load_wbc_yaml(); cfg["ENV_NAME"] = "simple"
    env = gym.make(T.ENV_ID, sim_mode="mujoco", render_hz=HZ, physics_dt=cfg["SIMULATE_DT"], headless=True,
                   max_episode_steps=10**6, sonic_config=cfg, target=T.TARGET, dr_level=0, success_criteria=1e9)
    return env, T


def third_person_camera(m):
    import mujoco
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = THIRD["lookat"]; cam.distance = THIRD["distance"]; cam.azimuth = THIRD["azimuth"]; cam.elevation = THIRD["elevation"]
    return cam


def scene_checks(env, T, obs):
    import mujoco
    robot = env.unwrapped.task.robot; m, d = robot.mjModel, robot.mjData
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_stereo_left")
    axis = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    tgt = env.unwrapped.task.layout.actors["target"].asset.label
    table = env.unwrapped.task.layout.scene.table
    out = {"robot_to_edge_feet": round(float(T.ROBOT_TO_EDGE), 3), "pelvis_to_edge": round(float(L_pelvis_to_edge(T)), 3), "bowl_mass_kg": T.BOWL_MASS,
           "pelvis": d.xpos[bid("pelvis")].round(3).tolist(),
           "table_top_z": round(float(table.pose.position[2] + 0.5 * table.size[2]), 3),
           "table_near_edge_x": round(float(table.pose.position[0] - 0.5 * table.size[0]), 3),
           "table_centre_y": round(float(table.pose.position[1]), 3),
           "target": {tgt: d.xpos[bid(tgt)].round(3).tolist()},
           "sink_counter": d.xpos[bid("sink_counter")].round(3).tolist() if bid("sink_counter") >= 0 else None,
           "basin_centre": [T.L["basin_cx"], T.L["basin_cy"], T.BS.SINK_H - T.BS.BASIN_DEPTH], "counter": d.xpos[bid("counter")].round(3).tolist() if bid("counter") >= 0 else None,
           "head_camera": {"pos": d.cam_xpos[cam].round(3).tolist(), "pitch_down_deg": round(float(np.degrees(np.arctan2(-axis[2], np.hypot(axis[0], axis[1])))), 1),
                           "fovy_deg": round(float(m.cam_fovy[cam]), 1), "image": list(np.asarray(obs["head_stereo_left"]).shape[:2][::-1])},
           "obs_keys": sorted(obs.keys()), "instruction": env.unwrapped.task.instruction}
    return out


def load_client_log(path: Path, ep: int):
    """Executed rows of one client episode: 28-D upper-body targets (UPPER_NAMES order), measured state, nav, height,
    row times, real heading (IMU yaw of probe/robot_watch.jsonl, 1 Hz, relative to the episode start), AVI frame indices."""
    cur = None; acts = []; obs = []; frames = []; t0 = None
    for line in open(path):
        r = json.loads(line); ty = r.get("type")
        if ty == "event":
            if r["event"] == "episode_start" and r.get("episode") == ep: cur = ep; t0 = r["t"]
            elif r["event"] == "episode_end" and cur == ep: break
        elif cur == ep:
            if ty == "action": acts.append((r["t"], r["a"]))
            elif ty == "obs": obs.append((r["t"], r["arm"], r["hand"]))
            elif ty == "frame": frames.append((r["t"], r["i"]))
    assert acts, f"episode {ep} has no action rows in {path}"
    ts = np.array([t for t, _ in acts]); A = np.array([a for _, a in acts], dtype=float)
    ACT = np.concatenate([A[:, 0:14], A[:, 14:28], A[:, 28:31]], axis=1)   # hands, arms, waist rpy = UPPER_NAMES order
    NAV = A[:, 32:36].copy(); NAV[:, 3] = zero_stale_leading_rows(NAV[:, 3]); H = A[:, 31].copy()
    ot = np.array([t for t, _, _ in obs]); O = np.array([np.concatenate([h, a]) for _, a, h in obs], dtype=float)   # hand then arm
    STATE = np.stack([np.interp(ts, ot, O[:, k]) for k in range(O.shape[1])], axis=1) if len(obs) else np.zeros((len(ts), 28))
    STATE = np.concatenate([STATE, np.zeros((len(ts), 3))], axis=1)          # no waist state in the client log
    watch = path.parent.parent / "probe" / "robot_watch.jsonl"
    if watch.exists():
        W = [json.loads(l) for l in open(watch) if l.strip()]; wt = np.array([w["t"] for w in W]); wy = np.unwrap(np.array([w["imu_rpy"][2] for w in W]))
        heading = np.interp(ts, wt, wy); real_heading_rel = wrap(heading - np.interp(t0, wt, wy))
    else:
        real_heading_rel = np.zeros(len(ts))
    ft = np.array([t for t, _ in frames]); fi = np.array([i for _, i in frames])
    frame_idx = np.interp(ts, ft, fi) if len(frames) else None
    real_video = next(path.parent.glob("*.avi"), None)
    print(f"[replay] client log ep {ep}: {len(acts)} rows over {ts[-1]-ts[0]:.1f} s, {len(obs)} obs, frames {int(fi.min()) if len(fi) else '-'}..{int(fi.max()) if len(fi) else '-'}, IMU turn {np.degrees(real_heading_rel[-1]):+.0f} deg", flush=True)
    return ACT, STATE, NAV, H, ts, real_heading_rel, frame_idx, real_video


def L_pelvis_to_edge(T):
    return T.L["pelvis_to_edge"]


def tilt_deg(quat_wxyz):
    w, x, y, z = quat_wxyz
    zz = 1 - 2 * (x * x + y * y)                       # world z component of the body z axis
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def probe(out_dir: Path):
    import mujoco
    env, T = make_env()
    obs, info = env.reset()
    robot = env.unwrapped.task.robot; m, d = robot.mjModel, robot.mjData
    T.recolor_hands(m)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = scene_checks(env, T, obs); print(json.dumps(checks, indent=1)); (out_dir / "probe.json").write_text(json.dumps(checks, indent=1))
    I.fromarray(np.asarray(obs["head_stereo_left"])).save(out_dir / "head_stereo_left.png")
    r3 = mujoco.Renderer(m, height=720, width=1280); r3.update_scene(d, third_person_camera(m)); I.fromarray(r3.render()).save(out_dir / "third_person.png"); r3.close()
    print("wrote", out_dir)
    os._exit(0)


def replay_episode(session, ep, video_out):
    import av, mujoco
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    if session.endswith(".jsonl"):                                  # holobrain deployment client log
        act_names = st_names = list(UPPER_NAMES)
        ACT, STATE, NAV, H, ts, real_heading_rel, frame_idx, real_video = load_client_log(Path(session), ep)
        fps = (len(ts) - 1) / (ts[-1] - ts[0]); n_rows = len(ts)
    elif "observation.state" not in json.load(open(ROOT / "data/real_recordings" / session / "meta/info.json"))["features"]:
        S = ROOT / "data/real_recordings" / session                   # psi0-format LeRobot dataset (36-D action, 32-D states)
        info = json.load(open(S / "meta/info.json")); fps = float(info["fps"])
        df = pd.read_parquet(S / f"data/chunk-000/episode_{ep:06d}.parquet")
        A = np.stack(df["action"]).astype(float); act_names = st_names = list(UPPER_NAMES)
        ACT = np.concatenate([A[:, 0:14], A[:, 14:28], A[:, 28:31]], axis=1)
        NAV = A[:, 32:36].copy(); NAV[:, 3] = zero_stale_leading_rows(NAV[:, 3]); H = A[:, 31].copy()
        STATE = np.concatenate([np.stack(df["observation.hand_joints"]).astype(float), np.stack(df["observation.arm_joints"]).astype(float), np.zeros((len(df), 3))], axis=1)
        real_heading_rel = NAV[:, 3].copy()                          # no base pose recorded: the commanded relative yaw stands in
        n_rows = len(df); frame_idx = None; ts = np.arange(n_rows) / fps
        real_video = S / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4"
    else:
        S = ROOT / "data/real_recordings" / session
        info = json.load(open(S / "meta/info.json"))
        df = pd.read_parquet(S / f"data/chunk-000/episode_{ep:06d}.parquet")
        act_names = info["features"]["action"]["names"]; st_names = info["features"]["observation.state"]["names"]
        ACT = np.stack(df["action"]).astype(float); STATE = np.stack(df["observation.state"]).astype(float)
        NAV = np.stack(df["teleop.navigate_command"]).astype(float); NAV[:, 3] = zero_stale_leading_rows(NAV[:, 3])
        H = np.stack(df["teleop.base_height_command"]).astype(float).ravel()
        BP = np.stack(df["observation.base_pose"]).astype(float)
        real_heading = np.array([yaw_of_wxyz(q) for q in BP[:, 3:7]]); real_heading_rel = wrap(real_heading - real_heading[0])
        fps = float(info["fps"]); n_rows = len(df); frame_idx = None; ts = np.arange(n_rows) / fps
        real_video = S / f"videos/chunk-000/observation.images.ego_view/episode_{ep:06d}.mp4"
    NAV[:, :2] *= NAV_GAIN
    walking = (np.abs(NAV[:, 0]) > 0.05) | (np.abs(NAV[:, 1]) > 0.05)
    row_dt = np.where(walking, WALK_STRETCH, 1.0) / fps                # sim time each recorded row lasts
    row_t = np.concatenate([[0.0], np.cumsum(row_dt)[:-1]])             # sim time at which each row starts
    total_t = float(row_t[-1] + row_dt[-1])
    n = int(np.ceil(total_t * HZ))                            # controller ticks at 50 Hz; rows are resampled onto them
    def row_at(t):
        return min(int(np.searchsorted(row_t, t, side="right") - 1), n_rows - 1)
    def act_at(t):                                            # linear interpolation of the recorded joint targets
        i0 = max(row_at(t), 0); i1 = min(i0 + 1, n_rows - 1); a = float(np.clip((t - row_t[i0]) / row_dt[i0], 0, 1)) if i1 > i0 else 0.0
        return (1 - a) * ACT[i0] + a * ACT[i1]
    print(f"[replay] {n_rows} rows at {fps:.1f} fps -> {n} ticks at {HZ} Hz (walk stretch {WALK_STRETCH:g}: {walking.sum()} walking rows, {total_t:.1f} s)", flush=True)

    env, T = make_env()
    try:
        task = env.unwrapped.task; robot = task.robot
        agent = PicoDecoupledAgent(robot); agent._poll_pico_buttons = lambda: None
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        tp = agent._teleop_policy; rm = agent._dwbc_robot_model
        inv = {v: k for k, v in rm.joint_to_dof_index.items()}
        up_names = [inv[i] for i in rm.get_joint_group_indices("upper_body")]
        up_from_act = np.array([act_names.index(nm) for nm in up_names])
        st_arm = np.array([st_names.index(nm) for nm in ARM_JOINTS])
        key_act = {nm: act_names.index(nm) for nm in KEY_JOINTS}; key_st = {nm: st_names.index(nm) for nm in KEY_JOINTS}

        obs, sinfo = env.reset(); m, d = robot.mjModel, robot.mjData
        T.recolor_hands(m)
        checks = scene_checks(env, T, obs); print("[replay] scene:", json.dumps(checks), flush=True)
        tgt_label = task.layout.actors["target"].asset.label
        tgt_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tgt_label)
        bin_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sink_counter")
        basin = np.array([T.L["basin_cx"], T.L["basin_cy"], T.BS.SINK_H - T.BS.BASIN_DEPTH])
        palm_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")      # the sonic MJCF has no palm body: palm geoms sit on the wrist-yaw link
        assert palm_body >= 0
        tip_bodies = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("right_hand_index_1_link", "right_hand_thumb_2_link")}
        mj_arm = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in ARM_JOINTS]
        mj_key = {nm: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in KEY_JOINTS}
        if robot.elastic_band is not None: robot.elastic_band.enable = False
        agent._dropping = False; agent.reset_policy(); agent._wbc_policy.lower_body_policy.use_policy_action = True
        img_key = "head_stereo_left"

        r3 = mujoco.Renderer(m, height=480, width=640); cam = third_person_camera(m)
        def third():
            r3.update_scene(d, cam); return r3.render().copy()

        cur = {"i": None}
        def replay_get_action():
            i = cur["i"]
            if i is None:
                return {"target_upper_body_pose": np.asarray(rm.default_body_pose, dtype=float)[rm.get_joint_group_indices("upper_body")],
                        "navigate_cmd": [0.0, 0.0, 0.0, 0.0], "base_height_command": 0.74, "wrist_pose": np.zeros(14),
                        "toggle_policy_action": False, "toggle_data_collection": False, "toggle_data_abort": False}
            t = i / HZ; r = row_at(t)
            return {"target_upper_body_pose": act_at(t)[up_from_act].copy(), "navigate_cmd": NAV[r].tolist(),
                    "base_height_command": float(H[r]), "wrist_pose": np.zeros(14),
                    "toggle_policy_action": False, "toggle_data_collection": False, "toggle_data_abort": False}
        tp.get_action = replay_get_action

        c = av.open(str(video_out), "w"); vs = c.add_stream("libx264", rate=25); vs.pix_fmt = "yuv420p"; vs.options = {"crf": "24", "preset": "veryfast"}
        vs.width, vs.height = 3 * 640, 480
        rc = av.open(str(real_video)); real_iter = rc.decode(video=0); real_state = {"frame": None, "next": None, "next_t": -1.0}
        i_first = 0; vfps = fps
        if frame_idx is not None:                                   # client AVI: seek to the episode's first frame index
            vs_in = rc.streams.video[0]; vfps = float(vs_in.average_rate); i_first = int(frame_idx[0])
            rc.seek(int(i_first / vfps / vs_in.time_base), stream=vs_in, backward=True, any_frame=False); real_iter = rc.decode(video=0)
            for fr in real_iter:
                if fr.pts is not None and int(round(float(fr.pts * vs_in.time_base) * vfps)) >= i_first - 1:
                    break
            real_state["next"] = fr.to_ndarray(format="rgb24"); real_state["next_t"] = 0.0
            frame_t = np.array([(fi - i_first) / vfps for fi in frame_idx])
        def real_at(t):
            """The real frame whose time is the latest <= t (frames are decoded once, in order)."""
            if frame_idx is not None:                               # sim time -> recorded row -> frame index -> AVI time
                t = float(frame_t[row_at(t)])
            while real_state["next_t"] <= t and real_state["next"] is not None or real_state["next_t"] < 0:
                if real_state["next"] is not None:
                    real_state["frame"] = real_state["next"]
                try:
                    fr = next(real_iter); real_state["next"] = fr.to_ndarray(format="rgb24")
                    real_state["next_t"] = (float(fr.time) - i_first / vfps) if frame_idx is not None else float(fr.time if fr.time is not None else real_state["next_t"] + 1 / fps)
                except StopIteration:
                    real_state["next"] = None; real_state["next_t"] = float("inf"); break
            return real_state["frame"]
        def emit(label, t_real):
            v3 = third(); ego = np.asarray(obs[img_key]); real = real_at(t_real)
            tiles = [v3]
            for im in (ego, real):
                if im is None: im = np.zeros_like(ego)
                Hh = v3.shape[0]; pad = Hh - im.shape[0]
                im = np.pad(im, ((pad // 2, pad - pad // 2), (0, 0), (0, 0))) if pad > 0 else im[:Hh]
                tiles.append(im)
            fr = np.concatenate(tiles, axis=1); fr = fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2]
            im = I.fromarray(fr); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, fr.shape[1], 22], fill=(0, 0, 0)); dr.text((6, 4), label, fill=(255, 255, 255))
            for x, t in ((6, "MuJoCo third-person (bowl/sink scene)"), (646, "MuJoCo D455 (sim)"), (1286, "REAL D455 (recording)")):
                dr.rectangle([x, 26, x + 230, 44], fill=(0, 0, 0)); dr.text((x + 4, 29), t, fill=(255, 255, 0))
            for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")): c.mux(pkt)

        clock = None
        if FAST:
            clock = VirtualClock(); patched = clock.install()
            print(f"[replay] fast mode: virtual clock in {len(patched)} modules, no pacing, no video", flush=True)
        t0 = time.monotonic(); tick = 0
        while not (robot.stabilized and (clock is not None and tick >= 150 or clock is None and time.monotonic() - t0 > 2.0)):
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=sinfo); obs, *_, sinfo = env.step(action)
            tick += 1
            if clock is not None: clock.tick()
            else:
                sl = t0 + tick / HZ - time.monotonic(); (sl > 0) and time.sleep(sl)
                if time.monotonic() - t0 > 20: break
            if clock is not None and tick > 1000: break
        tp.is_active = True
        heading0 = yaw_of_wxyz(d.qpos[3:7]); xy0 = d.qpos[:2].copy()
        tgt0 = d.xpos[tgt_body].copy()
        from bowl_sink_gates import BowlSinkGates, GateCfg, ContactProbe            # the four task gates, evaluated live
        gates = BowlSinkGates(GateCfg(basin_len=T.BS.BASIN_L, basin_across=T.BS.BASIN_ACROSS, bowl_base_r=T.BS.BOWL_R_BOT, sink_h=T.BS.SINK_H))
        probe = ContactProbe(m, str(task.target.asset.label))
        trace = []
        t0 = time.monotonic(); tick = 0
        for i in range(n):
            cur["i"] = i
            if i == 0:
                agent._wbc_policy.handle_keyboard_button("c")
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=sinfo)
            obs, _r, _te, _tr, sinfo = env.step(action)
            sim_h = wrap(yaw_of_wxyz(d.qpos[3:7]) - heading0)
            t_i = i / HZ; ri = row_at(t_i)
            gates.update(t_i, d.qpos[:2], yaw_of_wxyz(d.qpos[3:7]), d.xpos[tgt_body], basin, probe.hand_on(d),
                         base_speed=float(np.hypot(*d.qvel[:2])), bowl_tilt_deg=tilt_deg(d.xquat[tgt_body]))
            trace.append({"i": i, "row": ri, "t": t_i, "sim_heading_rel": float(sim_h), "real_heading_rel": float(real_heading_rel[ri]),
                          "cmd_rel_yaw": float(NAV[ri, 3]), "turn_flag": float(NAV[ri, 2]), "vx": float(NAV[ri, 0]), "vy": float(NAV[ri, 1]),
                          "sim_xy": (d.qpos[:2] - xy0).round(4).tolist(), "target_xyz": d.xpos[tgt_body].round(4).tolist(),
                          "target_tilt_deg": round(tilt_deg(d.xquat[tgt_body]), 1), "palm_r": d.xpos[palm_body].round(4).tolist(),
                          "index_tip_r": d.xpos[tip_bodies["right_hand_index_1_link"]].round(4).tolist(), "thumb_tip_r": d.xpos[tip_bodies["right_hand_thumb_2_link"]].round(4).tolist(),
                          "sim_arm": d.qpos[mj_arm].round(4).tolist(), "cmd_arm": act_at(t_i)[[act_names.index(nm) for nm in ARM_JOINTS]].round(4).tolist(),
                          "real_arm": STATE[ri, st_arm].round(4).tolist(),
                          "key_sim": {k: float(d.qpos[v]) for k, v in mj_key.items()}, "key_cmd": {k: float(ACT[ri, v]) for k, v in key_act.items()},
                          "key_real": {k: float(STATE[ri, v]) for k, v in key_st.items()}})
            if clock is not None: clock.tick()
            if i % 2 == 0 and clock is None:
                emit(f"{Path(session).name[:32]} ep {ep}  t={t_i:5.1f}s  vx {NAV[ri, 0]:+.2f} vy {NAV[ri, 1]:+.2f} turn {NAV[ri, 2]:+.1f} rel yaw {NAV[ri, 3]:+.2f} | heading sim {sim_h:+.2f} real {real_heading_rel[ri]:+.2f} | bowl lift {d.xpos[tgt_body][2] - tgt0[2]:+.2f} m", t_i)
            tick += 1
            if clock is None:
                sl = t0 + tick / HZ - time.monotonic()
                if sl > 0: time.sleep(sl)
        for j in range(int(SETTLE_S * HZ)):                       # hold the last row so a released bottle comes to rest
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=sinfo)
            obs, _r, _te, _tr, sinfo = env.step(action)
            gates.update(n / HZ + j / HZ, d.qpos[:2], yaw_of_wxyz(d.qpos[3:7]), d.xpos[tgt_body], basin, probe.hand_on(d),
                         base_speed=float(np.hypot(*d.qvel[:2])), bowl_tilt_deg=tilt_deg(d.xquat[tgt_body]))
            if clock is not None: clock.tick()
            if j % 2 == 0 and clock is None:
                emit(f"ep {ep}  settle {j / HZ:4.1f}s after the last row | bowl {np.round(d.xpos[tgt_body], 2).tolist()} tilt {tilt_deg(d.xquat[tgt_body]):.0f} deg", n / HZ)
            tick += 1
            if clock is None:
                sl = t0 + tick / HZ - time.monotonic()
                if sl > 0: time.sleep(sl)
        if clock is None:
            for pkt in vs.encode(): c.mux(pkt)
            c.close(); rc.close()
        lift = np.array([r["target_xyz"][2] - tgt0[2] for r in trace]); tilt = np.array([r["target_tilt_deg"] for r in trace])
        k = int(lift.argmax()); tip = next((r for r in trace if r["target_tilt_deg"] > 45), None)
        z = np.array([r["target_xyz"][2] for r in trace]); k_rel = None
        for j in range(k, len(z) - 5):                        # release = the bottle starts free-falling (> 0.5 m/s over 0.1 s)
            if z[j] - z[j + 5] > 0.05:
                k_rel = j; break
        release = {"t": trace[k_rel]["t"], "xy": trace[k_rel]["target_xyz"][:2], "z": trace[k_rel]["target_xyz"][2]} if k_rel is not None else None
        grasp = {"max_lift_m": round(float(lift[k]), 3), "t_max_lift": trace[k]["t"], "tilt_at_max_lift_deg": float(tilt[k]),
                 "first_tip_t": tip["t"] if tip else None,
                 "palm_minus_bottle_at_tip": (np.array(tip["palm_r"]) - np.array(tip["target_xyz"])).round(3).tolist() if tip else None,
                 "success": bool(lift[k] >= 0.10 and tilt[k] < 60), "release": release}
        ex, ey, ez = d.xpos[tgt_body]
        grasp["in_sink"] = bool(abs(ex - basin[0]) < T.BS.BASIN_L / 2 and abs(ey - basin[1]) < T.BS.BASIN_ACROSS / 2 and ez < T.BS.SINK_H)
        grasp["in_bin"] = grasp["in_sink"]
        grasp["bowl_end_minus_basin"] = [round(float(ex - basin[0]), 3), round(float(ey - basin[1]), 3), round(float(ez - basin[2]), 3)]
        summary = {"robot_to_edge_feet": checks["robot_to_edge_feet"], "target_start": tgt0.round(3).tolist(), "target_end": d.xpos[tgt_body].round(3).tolist(),
                   "basin": basin.round(3).tolist(),
                   "base_end_xy": (d.qpos[:2] - xy0).round(3).tolist(), "grasp": grasp, "scene": checks,
                   "gates": gates.P}
        print("[replay] gates:", gates.report(), flush=True)
        print("[replay] summary:", json.dumps(summary), flush=True)
        trace.append({"summary": summary})
    finally:
        try: r3.close()
        except Exception: pass
        env.close()
    return trace


def plots(ep, trace):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.array([r["t"] for r in trace]); out = {}
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, [r["real_heading_rel"] for r in trace], label="real robot heading (rel. to start)", lw=2)
    ax.plot(t, [r["sim_heading_rel"] for r in trace], label="MuJoCo heading (rel. to start)", lw=2)
    ax.plot(t, [r["cmd_rel_yaw"] for r in trace], "k--", label="recorded target_yaw (relative dial)", lw=1)
    ax.fill_between(t, -3.5, 3.5, where=np.abs([r["turn_flag"] for r in trace]) > 1e-6, color="orange", alpha=0.12, label="turn flag on")
    ax.set_ylim(min(-0.5, min(r["real_heading_rel"] for r in trace) - 0.3), max(0.5, max(r["real_heading_rel"] for r in trace) + 0.3))
    ax.set_xlabel("s"); ax.set_ylabel("rad"); ax.set_title(f"episode {ep}: heading"); ax.legend(fontsize=8, loc="best"); ax.grid(alpha=.3)
    b = io.BytesIO(); fig.tight_layout(); fig.savefig(b, format="png", dpi=110); plt.close(fig); out["heading"] = base64.b64encode(b.getvalue()).decode()
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.2)); axes = axes.ravel()
    for k, nm in enumerate(KEY_JOINTS):
        ax = axes[k]; ax.plot(t, [r["key_cmd"][nm] for r in trace], "k--", lw=1, label="recorded target")
        ax.plot(t, [r["key_real"][nm] for r in trace], lw=1.5, label="real measured"); ax.plot(t, [r["key_sim"][nm] for r in trace], lw=1.5, label="MuJoCo measured")
        ax.set_title(nm.replace("_joint", ""), fontsize=9); ax.grid(alpha=.3)
        if k == 0: ax.legend(fontsize=7)
    xy = np.array([r["sim_xy"] for r in trace]); tz = np.array([r["target_xyz"][2] for r in trace]); ax = axes[7]
    ax.plot(xy[:, 0], xy[:, 1], label="base path (m)"); ax.plot(0, 0, "go"); ax.plot(xy[-1, 0], xy[-1, 1], "ro")
    ax.set_title("MuJoCo base path (m), x toward the table", fontsize=9); ax.axis("equal"); ax.grid(alpha=.3)
    b = io.BytesIO(); fig.tight_layout(); fig.savefig(b, format="png", dpi=100); plt.close(fig); out["joints"] = base64.b64encode(b.getvalue()).decode()
    return out


def summarize(session, ep, tag):
    tr_all = json.load(open(OUT / f"trace_ep{ep}_{tag}.json")); tr = [r for r in tr_all if "summary" not in r]; summ = next((r["summary"] for r in tr_all if "summary" in r), {})
    sim_arm = np.array([r["sim_arm"] for r in tr]); cmd_arm = np.array([r["cmd_arm"] for r in tr]); real_arm = np.array([r["real_arm"] for r in tr])
    hs = np.array([r["sim_heading_rel"] for r in tr]); hr = np.array([r["real_heading_rel"] for r in tr]); tz = np.array([r["target_xyz"][2] for r in tr])
    return {"ep": ep, "tag": tag, "rows": len(tr), "seconds": round(len(tr) / HZ, 1), "video": f"replay_{Path(session).name[:40]}_ep{ep}_{tag}.mp4",
            "robot_to_edge_feet": summ.get("robot_to_edge_feet"), "grasp": summ.get("grasp"),
            "real_heading_change": round(float(hr[-1]), 3), "sim_heading_change": round(float(hs[-1]), 3),
            "heading_rms_diff": round(float(np.sqrt(np.mean(wrap(hs - hr) ** 2))), 3),
            "arm_rms_sim_vs_target": round(float(np.sqrt(np.mean((sim_arm - cmd_arm) ** 2))), 3),
            "arm_rms_real_vs_target": round(float(np.sqrt(np.mean((real_arm - cmd_arm) ** 2))), 3),
            "bottle_max_lift_m": round(float(tz.max() - tz[0]), 3), "bottle_end": summ.get("target_end"), "bin": summ.get("bin"),
            "base_end_xy": summ.get("base_end_xy"), "plots": plots(ep, tr)}


def build_html(session, episodes):
    import re
    runs = sorted((int(mm.group(1)), mm.group(2)) for f in OUT.glob("trace_ep*_*.json") for mm in [re.match(r"trace_ep(\d+)_(.+)\.json", f.name)] if mm)
    results = [summarize(session, ep, tag) for ep, tag in runs]
    rows = "".join(f"<tr><td>{r['ep']} {r['tag']}</td><td>{r['seconds']} s</td><td>{r['real_heading_change']:+.2f}</td><td>{r['sim_heading_change']:+.2f}</td><td>{r['heading_rms_diff']:.2f}</td><td>{r['arm_rms_real_vs_target']:.3f}</td><td>{r['arm_rms_sim_vs_target']:.3f}</td><td>{r['bottle_max_lift_m']:.2f} m</td><td>{'yes' if (r['grasp'] or {}).get('success') else 'no'}</td><td>{r['base_end_xy']}</td></tr>" for r in results)
    secs = "".join(f'''<section class=card><h2>Episode {r['ep']}, feet {r['robot_to_edge_feet']} m from the edge <span class=sub>{r['seconds']} s, {r['rows']} rows, grasp {r['grasp']}</span></h2>
<video controls preload=metadata src="{r['video']}"></video>
<div class=sub>Left: MuJoCo third-person view of the bottle/bin scene. Middle: MuJoCo D455 (sim). Right: the real recording's ego camera at the same row. Bottle end position {r['bottle_end']}, bin at {r['bin']}.</div>
<img src="data:image/png;base64,{r['plots']['heading']}"><img src="data:image/png;base64,{r['plots']['joints']}"></section>''' for r in results)
    html = f'''<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Bottle Bin Action Replay</title>
<style>body{{margin:0;background:#f4f4f1;color:#111;font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}main{{max-width:1180px;margin:0 auto;padding:22px 16px 60px}}h1{{font-size:22px;margin:0 0 6px}}h2{{font-size:17px;margin:0 0 4px}}.sub{{color:#444;font-size:12.5px;font-weight:400}}
.card{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px 14px;margin:12px 0;overflow-x:auto}}video,img{{width:100%;max-width:1150px;display:block;margin:8px 0;border-radius:6px}}
table{{border-collapse:collapse;font-size:13.5px}}th,td{{padding:5px 10px;border-top:1px solid #e5e5e5;text-align:right}}th{{font-size:11px;text-transform:uppercase;color:#666;border-top:0}}</style></head><body><main>
<h1>Real recording {session} replayed in the bottle / bin scene (MuJoCo)</h1>
<div class=sub>The recorded upper-body targets, navigation command and base height drive the decoupled whole-body stack in MuJoCo inside the new scene
(2.07 m table, cover board, bottle straight ahead, bin at the table's right end). Generated {time.strftime("%Y-%m-%d %H:%M")}.</div>
<section class=card><h2>Summary</h2><table><tr><th>ep</th><th>length</th><th>real heading change</th><th>sim heading change</th><th>heading RMS diff</th><th>arm RMS real vs target</th><th>arm RMS sim vs target</th><th>bottle max lift</th><th>grasp ok</th><th>base end xy (m)</th></tr>{rows}</table></section>
{secs}</main></body></html>'''
    (OUT / "index.html").write_text(html)
    json.dump([{k: v for k, v in r.items() if k != "plots"} for r in results], open(OUT / "summary.json", "w"), indent=1)
    print("wrote", OUT / "index.html", "| episodes:", [r["ep"] for r in results], flush=True)
    for r in results: print({k: v for k, v in r.items() if k != "plots"}, flush=True)


def main():
    import subprocess
    ap = argparse.ArgumentParser(); ap.add_argument("--session", default=None, help="LeRobot session name under SIMPLE/data/real_recordings")
    ap.add_argument("--client-log", default=str(Path("/home/Horizon/wrk/robot_orchard_deploy/holobrain_g1_deploy/runs/20260921_172723_bowltosink_c96/client/actions_taskX_20260921_172803.jsonl")))
    ap.add_argument("--episodes", type=int, nargs="+", default=[7])
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--html-only", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--robot-to-edge", type=float, default=0.30, help="front of the feet -> counter front edge (m)")
    ap.add_argument("--bottle-mass", "--bowl-mass", dest="bottle_mass", type=float, default=0.12, help="bowl mass in kg")
    ap.add_argument("--bin-xy", "--sink-along", dest="bin_xy", type=float, nargs=1, default=None, metavar="ALONG", help="basin centre: distance from the counter front edge toward the rear (m); default from layout.json")
    ap.add_argument("--bottle-xy", "--bowl-xy", dest="bottle_xy", type=float, nargs=2, default=None, metavar=("X", "Y"), help="bowl centre relative to the robot start (m)")
    ap.add_argument("--nav-gain", type=float, default=1.0, help="scale the recorded vx, vy (sim walks ~25%% slower than the robot)")
    ap.add_argument("--walk-stretch", type=float, default=1.0, help="rows with a walking command last this much longer in sim")
    ap.add_argument("--tag", default=None, help="output name tag")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.session is None: a.session = a.client_log
    os.environ["REPLAY_NAV_GAIN"] = f"{a.nav_gain:.3f}"; os.environ["REPLAY_WALK_STRETCH"] = f"{a.walk_stretch:.3f}"
    os.environ["BOWL_SINK_ROBOT_TO_EDGE"] = f"{a.robot_to_edge:.4f}"; os.environ["BOWL_SINK_BOWL_MASS"] = f"{a.bottle_mass:.4f}"
    if a.bin_xy: os.environ["BOWL_SINK_SINK_ALONG"] = f"{a.bin_xy[0]:.3f}"
    if a.bottle_xy: os.environ["BOWL_SINK_BOWL_XY"] = f"{a.bottle_xy[0]:.3f},{a.bottle_xy[1]:.3f}"
    tag = a.tag or (f"feet{round(a.robot_to_edge * 100):02d}cm" + (f"_sink{round(a.bin_xy[0] * 100):d}" if a.bin_xy else "") + (f"_gain{a.nav_gain:.2f}" if a.nav_gain != 1.0 else "") + (f"_stretch{a.walk_stretch:.2f}" if a.walk_stretch != 1.0 else ""))
    if a.probe:
        probe(OUT / "probe")
    if a.worker:
        ep = a.episodes[0]; print(f"\n[replay] === episode {ep} ===", flush=True)
        tr = replay_episode(a.session, ep, OUT / f"replay_{Path(a.session).name[:40]}_ep{ep}_{tag}.mp4")
        json.dump(tr, open(OUT / f"trace_ep{ep}_{tag}.json", "w")); print(f"[replay] episode {ep} done: {len(tr)} rows", flush=True)
        os._exit(0)
    if not a.html_only:
        for ep in a.episodes:       # one process per episode (Pico TCP port + SDK teardown), as in SIMPLE's script
            rc = subprocess.run([sys.executable, "-u", __file__, "--session", a.session, "--episodes", str(ep), "--worker",
                                 "--robot-to-edge", str(a.robot_to_edge), "--bottle-mass", str(a.bottle_mass), "--tag", tag, "--nav-gain", str(a.nav_gain), "--walk-stretch", str(a.walk_stretch)]
                                + (["--bin-xy", str(a.bin_xy[0])] if a.bin_xy else [])
                                + (["--bottle-xy", str(a.bottle_xy[0]), str(a.bottle_xy[1])] if a.bottle_xy else []), timeout=1800).returncode
            print(f"[replay] worker ep {ep} {tag}: exit {rc}, trace {'ok' if (OUT / f'trace_ep{ep}_{tag}.json').exists() else 'MISSING'}", flush=True)
    build_html(a.session, a.episodes)


if __name__ == "__main__":
    main()
