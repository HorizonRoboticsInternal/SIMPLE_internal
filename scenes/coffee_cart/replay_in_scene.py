#!/usr/bin/env python
"""Replay the real CartCoffeeCup recordings through the decoupled-WBC stack in MuJoCo, inside the coffee-cart scene.

Source: the psi0-format LeRobot dataset data/real_recordings/psi0/CartCoffeeCup_0919 (97 episodes, 30 fps).
The recorded upper-body targets, navigation command and base height drive PicoDecoupledAgent tick by tick (rows
resampled onto its 50 Hz ticks); the lower body walks by itself. Gravity compensation is ON so the arms hold the
commanded pose instead of sagging (~0.09 rad at the elbow with pure PD).

Two modes:
  --fast    no video, no wall-clock pacing -> for sweeps (also removes the run-to-run jitter the pacing causes)
  default   third-person | sim D455 | real D455 mp4 + trace json, wall-clock paced

  python replay_in_scene.py --probe
  python replay_in_scene.py --episodes 0 5 10 --fast
  python replay_in_scene.py --sweep-robot-to-cart 0.15 0.20 0.25 --sweep-route 3.0 3.5 4.0 --episodes 0 5 10 --fast
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
SESSION = "psi0/CartCoffeeCup_0919"
OUT = HERE / "replay"
HZ = 50
ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
KEY_JOINTS = ["left_shoulder_pitch_joint", "left_elbow_joint", "right_shoulder_pitch_joint", "right_elbow_joint",
              "right_wrist_pitch_joint", "right_hand_index_0_joint", "left_hand_index_0_joint"]
THIRD = dict(lookat=[2.0, -0.9, 0.6], distance=7.0, azimuth=-50.0, elevation=-24.0)
UPPER_NAMES = ([f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
               + [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
               + [f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
               + ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"])
# the 36-D psi0 action row: [0:7 L hand, 7:14 R hand, 14:21 L arm, 21:28 R arm, 28:31 rpy, 31 height, 32:36 vx vy flag target_yaw]
SETTLE_S = float(os.environ.get("REPLAY_SETTLE_S", "2.0"))
NAV_GAIN = float(os.environ.get("REPLAY_NAV_GAIN", "1.0"))
WALK_STRETCH = float(os.environ.get("REPLAY_WALK_STRETCH", "1.0"))
POST_TURN_GAIN = float(os.environ.get("REPLAY_POST_TURN_GAIN", "1.0"))   # scales vx/vy only AFTER the commanded turn: the sim
                                                                        # gait barely engages at the recorded 0.13 m/s approach
FAST = os.environ.get("REPLAY_FAST", "0") == "1"
SIM_CLOCK = {"t": 1000.0, "on": os.environ.get("REPLAY_SIM_CLOCK", "1") == "1"}


def _install_sim_clock():
    """The agent hands time.monotonic() to the WBC as the interpolation clock (target_time = now + 1/50), so under load the
    upper-body trajectory depends on how fast the machine happened to run. Give those modules a `time` shim that returns
    the tick count instead: the same run then reproduces bit for bit, on any load."""
    import types, time as _time
    import simple.agents.pico_decoupled_agent as PA
    import decoupled_wbc.control.policy.teleop_policy as TP
    shim = types.SimpleNamespace(monotonic=lambda: SIM_CLOCK["t"], time=_time.time, sleep=_time.sleep,
                                 perf_counter=lambda: SIM_CLOCK["t"], strftime=_time.strftime)
    PA.time = shim; TP.time = shim


def _tick_clock():
    SIM_CLOCK["t"] += 1.0 / HZ


def yaw_of_wxyz(q):
    from scipy.spatial.transform import Rotation as R
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("zyx")[0])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def tilt_deg(q):
    from scipy.spatial.transform import Rotation as R
    z = R.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, 1.0])
    return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))


def zero_stale_leading_rows(rel, max_lead=10):
    rel = rel.copy()
    z = np.where(np.abs(rel[:max_lead]) < 1e-9)[0]
    if len(z) and z[0] > 0:
        rel[: z[0]] = 0.0
    return rel


def make_env():
    """Env with gravity compensation ON: feed-forward g(q) on the arms so the commanded pose is the held pose."""
    import gymnasium as gym
    import coffee_cart_task as T
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    cfg = SimLoopConfig().load_wbc_yaml(); cfg["ENV_NAME"] = "simple"
    cfg["enable_gravity_compensation"] = True                      # user 2026-09-23: hold the pose we set, no sag
    cfg["gravity_compensation_joints"] = ["arms"]
    cfg["gravity_compensation_max_torque"] = float(os.environ.get("REPLAY_GRAVCOMP_MAX_TAU", "10.0"))
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
    L = T.L
    return {"robot_to_cart": round(float(T.ROBOT_TO_CART), 3), "route_forward": round(float(T.ROUTE_FORWARD), 3),
            "table_from_line": round(float(T.TABLE_FROM_LINE), 3), "cup_mass_kg": T.CUP_MASS, "cart_mass_kg": T.CART_MASS, "wheel_friction": T.WHEEL_FRICTION, "handle_top": T.BS.HANDLE_TOP,
            "gravity_compensation": True, "sim_clock": SIM_CLOCK["on"], "post_turn_gain": float(os.environ.get("REPLAY_POST_TURN_GAIN", "1.0")), "hand_kp_scale": os.environ.get("REPLAY_HAND_KP_SCALE", "1.0"), "wheel_priority_ok": None, "arm_kp_scale": os.environ.get("REPLAY_ARM_KP_SCALE", "1.0"), "bar_friction": os.environ.get("REPLAY_BAR_FRICTION"), "cart_attach": os.environ.get("REPLAY_CART_ATTACH", "none"), "waist_mode": os.environ.get("REPLAY_WAIST_MODE", "freeze_while_walking"),
            "pelvis": d.xpos[bid("pelvis")].round(3).tolist(),
            "cart": d.xpos[bid("cart")].round(3).tolist(),
            "target": {tgt: d.xpos[bid(tgt)].round(3).tolist()},
            "table_top_z": round(float(table.pose.position[2] + 0.5 * table.size[2]), 3),
            "table_near_edge_x": round(float(table.pose.position[0] - 0.5 * table.size[0]), 3),
            "table_centre": [round(float(v), 3) for v in table.pose.position[:2]],
            "head_camera": {"pos": d.cam_xpos[cam].round(3).tolist(),
                            "pitch_down_deg": round(float(np.degrees(np.arctan2(-axis[2], np.hypot(axis[0], axis[1])))), 1),
                            "fovy_deg": round(float(m.cam_fovy[cam]), 1)},
            "instruction": env.unwrapped.task.instruction}


def probe(out_dir: Path):
    import mujoco
    env, T = make_env()
    obs, info = env.reset()
    robot = env.unwrapped.task.robot; m, d = robot.mjModel, robot.mjData
    T.recolor_hands(m)
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = scene_checks(env, T, obs); print(json.dumps(checks, indent=1))
    (out_dir / "probe.json").write_text(json.dumps(checks, indent=1))
    r3 = mujoco.Renderer(m, height=720, width=1280); r3.update_scene(d, third_person_camera(m))
    I.fromarray(r3.render()).save(out_dir / "third_person.png"); r3.close()
    print("wrote", out_dir); sys.stdout.flush(); os._exit(0)


def load_episode(ep: int):
    """psi0 LeRobot episode -> (ACT 31-D upper-body targets, NAV, H, fps, n_rows, real_video)."""
    S = ROOT / "data/real_recordings" / SESSION
    info = json.load(open(S / "meta/info.json")); fps = float(info["fps"])
    df = pd.read_parquet(S / f"data/chunk-000/episode_{ep:06d}.parquet")
    A = np.stack(df["action"].values).astype(float)
    ACT = np.concatenate([A[:, 0:14], A[:, 14:28], A[:, 28:31]], axis=1)      # hands, arms, torso rpy -> UPPER_NAMES order
    NAV = A[:, 32:36].copy(); NAV[:, 3] = zero_stale_leading_rows(NAV[:, 3])
    H = A[:, 31].copy()
    STATE = np.concatenate([np.stack(df["observation.hand_joints"].values).astype(float),
                            np.stack(df["observation.arm_joints"].values).astype(float),
                            np.zeros((len(df), 3))], axis=1)
    real_video = S / f"videos/chunk-000/egocentric/episode_{ep:06d}.mp4"
    return ACT, STATE, NAV, H, fps, len(df), real_video


def replay_episode(ep, video_out, fast=False, video=None):
    video = (not fast) if video is None else video     # --video: record frames even in --fast (no pacing -> deterministic)
    import mujoco
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    ACT, STATE, NAV, H, fps, n_rows, real_video = load_episode(ep)
    act_names = st_names = list(UPPER_NAMES)
    nav_gain = float(os.environ.get("REPLAY_NAV_GAIN", "1.0"))            # read HERE: main() sets these after import
    walk_stretch = float(os.environ.get("REPLAY_WALK_STRETCH", "1.0"))
    post_turn_gain = float(os.environ.get("REPLAY_POST_TURN_GAIN", "1.0"))
    NAV[:, :2] *= nav_gain
    walking = (np.abs(NAV[:, 0]) > 0.05) | (np.abs(NAV[:, 1]) > 0.05)
    row_dt = np.where(walking, walk_stretch, 1.0) / fps
    row_t = np.concatenate([[0.0], np.cumsum(row_dt)[:-1]])
    total_t = float(row_t[-1] + row_dt[-1])
    n = int(np.ceil(total_t * HZ))

    def row_at(t):
        return min(int(np.searchsorted(row_t, t, side="right") - 1), n_rows - 1)

    def act_at(t):
        i0 = max(row_at(t), 0); i1 = min(i0 + 1, n_rows - 1)
        a = float(np.clip((t - row_t[i0]) / row_dt[i0], 0, 1)) if i1 > i0 else 0.0
        return (1 - a) * ACT[i0] + a * ACT[i1]

    _dy = np.abs(np.diff(NAV[:, 3]))
    RELEASE_ROW = int(np.where(_dy > 1e-6)[0][-1]) + 1 if (_dy > 1e-6).any() else n_rows
    print(f"[replay] ep {ep}: cart released at row {RELEASE_ROW} (t={RELEASE_ROW/fps:.1f}s, end of the commanded turn)", flush=True)
    print(f"[replay] ep {ep}: {n_rows} rows at {fps:.1f} fps -> {n} ticks at {HZ} Hz ({total_t:.1f} s, {walking.sum()} walking rows; "
          f"nav gain {nav_gain}, walk stretch {walk_stretch}, post-turn gain {post_turn_gain})", flush=True)

    env, T = make_env()
    r3 = None
    SIM_CLOCK["on"] = os.environ.get("REPLAY_SIM_CLOCK", "1") == "1"
    if SIM_CLOCK["on"]:
        _install_sim_clock()
    try:
        task = env.unwrapped.task; robot = task.robot
        agent = PicoDecoupledAgent(robot); agent._poll_pico_buttons = lambda: None
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        tp = agent._teleop_policy; rm = agent._dwbc_robot_model
        inv = {v: k for k, v in rm.joint_to_dof_index.items()}
        up_names = [inv[i] for i in rm.get_joint_group_indices("upper_body")]
        up_from_act = np.array([act_names.index(nm) for nm in up_names])
        waist_idx = np.array([j for j, nm in enumerate(up_names) if nm.startswith("waist")], dtype=int)
        waist_home = ACT[0][up_from_act][waist_idx].copy()      # the recorded torso rpy at row 0
        waist_mode = os.environ.get("REPLAY_WAIST_MODE", "freeze_while_walking")
        st_arm = np.array([st_names.index(nm) for nm in ARM_JOINTS])
        key_act = {nm: act_names.index(nm) for nm in KEY_JOINTS}; key_st = {nm: st_names.index(nm) for nm in KEY_JOINTS}

        obs, sinfo = env.reset(); m, d = robot.mjModel, robot.mjData
        T.recolor_hands(m)
        checks = scene_checks(env, T, obs); print("[replay] scene:", json.dumps(checks), flush=True)
        L = T.L
        tgt_label = task.layout.actors["target"].asset.label
        tgt_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tgt_label)
        cart_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cart")
        palm_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        tip_bodies = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                      ("right_hand_index_1_link", "right_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_thumb_2_link",
                       "left_wrist_yaw_link")}
        assert all(v >= 0 for v in tip_bodies.values()), tip_bodies
        mj_arm = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in ARM_JOINTS]
        mj_key = {nm: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)] for nm in KEY_JOINTS}
        cup_at_reset = d.xpos[tgt_body].copy()
        # --- grip / arm compliance knobs (applied on the compiled model; the sonic MJCF has stiff position actuators) ---
        hand_kp = float(os.environ.get("REPLAY_HAND_KP_SCALE", "1.0")); arm_kp = float(os.environ.get("REPLAY_ARM_KP_SCALE", "1.0"))
        bar_fr = os.environ.get("REPLAY_BAR_FRICTION")
        n_h = n_a = 0
        for a_ in range(m.nu):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a_, 0])) or ""
            sc = hand_kp if "_hand_" in jn else (arm_kp if any(k in jn for k in ("shoulder", "elbow", "wrist")) else 1.0)
            if sc != 1.0:
                m.actuator_gainprm[a_, 0] *= sc; m.actuator_biasprm[a_, 1] *= sc      # kp (biasprm[1] = -kp)
                n_h += "_hand_" in jn; n_a += "_hand_" not in jn
        # bar / wheel geoms by ROLE (geom cart_convex_<i> follows the cart_pieces() order), not by a size heuristic
        roles = T.cart_piece_roles()
        bar_geoms, wheel_geoms = set(), set()
        for g in range(m.ngeom):
            gn_ = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gn_.startswith("cart_convex_"):
                r_ = roles[int(gn_.split("_")[-1])]
                (bar_geoms if r_ == "bar" else wheel_geoms if r_ == "wheel" else set()).add(g)
        assert bar_geoms and len(wheel_geoms) == 4, (bar_geoms, wheel_geoms)
        wheel_fr_eff = [float(m.geom_friction[g, 0]) for g in wheel_geoms][0]; wheel_pri = int(m.geom_priority[list(wheel_geoms)[0]])
        if bar_fr is not None:
            for g in bar_geoms:
                m.geom_friction[g, 0] = float(bar_fr)
        hand_geoms_all = {g for g in range(m.ngeom) if "_hand_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")}
        print(f"[replay] compliance: hand kp x{hand_kp} ({n_h} act), arm kp x{arm_kp} ({n_a} act), bar geoms {sorted(bar_geoms)} friction {bar_fr}; "
              f"wheels {sorted(wheel_geoms)} friction {wheel_fr_eff} priority {wheel_pri}", flush=True)
        attach = os.environ.get("REPLAY_CART_ATTACH", "none") == "base"
        cart_jadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cart_joint")]
        cup_jadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{tgt_label}_joint")]
        if attach:
            # The WBC walk policy cannot push a cart: it jams against it (measured: 0.02 m of travel with the cart in
            # front, 2.30 m with it parked aside). Carry the cart rigidly with the base instead and take it out of the
            # robot's contact set, so the recorded gait plays out exactly as it does with no obstacle. The cart still
            # collides with the cup (contype/conaffinity bit 2), so the cup rides on the box and can be picked off it.
            for g in range(m.ngeom):
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == "cart":
                    m.geom_contype[g] = 2; m.geom_conaffinity[g] = 2
                elif (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == tgt_label:
                    m.geom_contype[g] = 3; m.geom_conaffinity[g] = 3
        if attach:                                   # rigid base->cart offset, captured once the robot has settled
            byaw0 = yaw_of_wxyz(d.qpos[3:7]); bxy0 = d.qpos[:2].copy()
            cxy0 = d.qpos[cart_jadr:cart_jadr + 2].copy(); cz0 = float(d.qpos[cart_jadr + 2])
            cyaw0 = yaw_of_wxyz(d.qpos[cart_jadr + 3:cart_jadr + 7])
            ca, sa = np.cos(-byaw0), np.sin(-byaw0)
            off = np.array([ca * (cxy0[0] - bxy0[0]) - sa * (cxy0[1] - bxy0[1]),
                            sa * (cxy0[0] - bxy0[0]) + ca * (cxy0[1] - bxy0[1])])
            dyaw = cyaw0 - byaw0
            cup_off = d.qpos[cup_jadr:cup_jadr + 3].copy() - np.array([cxy0[0], cxy0[1], cz0])
            cup_riding = {"on": True}
            cup_geoms = {g for g in range(m.ngeom)
                         if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == tgt_label}
            hand_geoms = {g for g in range(m.ngeom)
                          if "_hand_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")}

            held = {"on": True}

            def carry():
                if not held["on"]:
                    return
                if cur["i"] is not None and row_at(cur["i"] / HZ) >= RELEASE_ROW:
                    held["on"] = False                      # let go: the cart stays where it is, the robot walks in
                    for g in range(m.ngeom):
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == "cart":
                            m.geom_contype[g] = 1; m.geom_conaffinity[g] = 1
                    return
                yaw = yaw_of_wxyz(d.qpos[3:7]); bxy = d.qpos[:2]
                c, sn = np.cos(yaw), np.sin(yaw)
                cx = bxy[0] + c * off[0] - sn * off[1]
                cy = bxy[1] + sn * off[0] + c * off[1]
                cyaw = yaw + dyaw
                d.qpos[cart_jadr:cart_jadr + 3] = [cx, cy, cz0]
                d.qpos[cart_jadr + 3:cart_jadr + 7] = [np.cos(cyaw / 2), 0.0, 0.0, np.sin(cyaw / 2)]
                d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cart_joint")]:][:6] = 0.0
                if cup_riding["on"]:
                    touched = any((c.geom1 in cup_geoms and c.geom2 in hand_geoms)
                                  or (c.geom2 in cup_geoms and c.geom1 in hand_geoms)
                                  for c in d.contact[:d.ncon])
                    if touched:                                             # a hand reached it: stop pinning it
                        cup_riding["on"] = False
                    else:
                        dc = np.cos(cyaw - cyaw0), np.sin(cyaw - cyaw0)
                        d.qpos[cup_jadr:cup_jadr + 2] = [cx + dc[0] * cup_off[0] - dc[1] * cup_off[1],
                                                         cy + dc[1] * cup_off[0] + dc[0] * cup_off[1]]
                        d.qpos[cup_jadr + 2] = cz0 + cup_off[2]
        else:
            carry = lambda: None
        if robot.elastic_band is not None:
            robot.elastic_band.enable = False
        agent._dropping = False; agent.reset_policy(); agent._wbc_policy.lower_body_policy.use_policy_action = True

        if video:
            r3 = mujoco.Renderer(m, height=480, width=640); cam = third_person_camera(m)

        cur = {"i": None}
        def replay_get_action():
            i = cur["i"]
            if i is None:
                # settle into the RECORDED start posture (hands already on the handle), not the stack's default pose:
                # the default pose sweeps the arms straight through the cup and knocks it off the box.
                return {"target_upper_body_pose": ACT[0][up_from_act].copy(),
                        "navigate_cmd": [0.0, 0.0, 0.0, 0.0], "base_height_command": float(H[0]), "wrist_pose": np.zeros(14),
                        "toggle_policy_action": False, "toggle_data_collection": False, "toggle_data_abort": False}
            t = i / HZ; r = row_at(t)
            q = act_at(t)[up_from_act].copy()
            # The recorded waist (torso rpy) command fights the lower-body walk policy: driving it stalls the robot
            # (measured: 0.014 m of travel with it, 0.68 m with the waist held). Hold the waist while walking.
            if len(waist_idx) and (waist_mode == "freeze"
                                   or (waist_mode == "freeze_while_walking" and abs(float(NAV[r, 0])) > 0.1)):
                q[waist_idx] = waist_home
            nav = NAV[r].copy()
            if post_turn_gain != 1.0 and r >= RELEASE_ROW:
                nav[:2] *= post_turn_gain
            return {"target_upper_body_pose": q, "navigate_cmd": nav.tolist(),
                    "base_height_command": float(H[r]), "wrist_pose": np.zeros(14),
                    "toggle_policy_action": False, "toggle_data_collection": False, "toggle_data_abort": False}
        tp.get_action = replay_get_action

        emit = (lambda label, t_real: None)
        if video:
            import av
            c = av.open(str(video_out), "w"); vs = c.add_stream("libx264", rate=25); vs.pix_fmt = "yuv420p"
            vs.options = {"crf": "24", "preset": "veryfast"}; vs.width, vs.height = 3 * 640, 480
            rc = av.open(str(real_video)); real_iter = rc.decode(video=0)
            real_state = {"frame": None, "next": None, "next_t": -1.0}
            def real_at(t):
                while (real_state["next_t"] <= t and real_state["next"] is not None) or real_state["next_t"] < 0:
                    if real_state["next"] is not None:
                        real_state["frame"] = real_state["next"]
                    try:
                        fr = next(real_iter); real_state["next"] = fr.to_ndarray(format="rgb24")
                        real_state["next_t"] = float(fr.time if fr.time is not None else real_state["next_t"] + 1 / fps)
                    except StopIteration:
                        real_state["next"] = None; real_state["next_t"] = float("inf"); break
                return real_state["frame"]
            def emit(label, t_real):
                r3.update_scene(d, cam); v3 = r3.render().copy()
                ego = np.asarray(obs["head_stereo_left"]); real = real_at(t_real)
                tiles = [v3]
                for im in (ego, real):
                    if im is None: im = np.zeros_like(ego)
                    Hh = v3.shape[0]; pad = Hh - im.shape[0]
                    im = np.pad(im, ((pad // 2, pad - pad // 2), (0, 0), (0, 0))) if pad > 0 else im[:Hh]
                    tiles.append(im)
                fr = np.concatenate(tiles, axis=1); fr = fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2]
                im = I.fromarray(fr); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, fr.shape[1], 22], fill=(0, 0, 0)); dr.text((6, 4), label, fill=(255, 255, 255))
                for x, t in ((6, "MuJoCo third-person (coffee cart)"), (646, "MuJoCo D455 (sim)"), (1286, "REAL D455 (recording)")):
                    dr.rectangle([x, 26, x + 240, 44], fill=(0, 0, 0)); dr.text((x + 4, 29), t, fill=(255, 255, 0))
                for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")): c.mux(pkt)

        # stand up / stabilize before the recorded rows start
        t0 = time.monotonic(); tick = 0
        while not (robot.stabilized and (tick / HZ) > 2.0):
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=sinfo)
            obs, *_, sinfo = env.step(action)
            _tick_clock(); carry()
            tick += 1
            if not fast:
                sl = t0 + tick / HZ - time.monotonic(); (sl > 0) and time.sleep(sl)
            if tick / HZ > 20: break
        tp.is_active = True
        # press the policy-action toggle BEFORE entering replay mode, not on the first replay tick: pressing it once
        # the recorded rows are already streaming leaves the robot unable to walk (measured: 0.02 m vs 0.68 m of travel).
        agent._wbc_policy.handle_keyboard_button("c")
        heading0 = yaw_of_wxyz(d.qpos[3:7]); xy0 = d.qpos[:2].copy()
        tgt0 = d.xpos[tgt_body].copy(); cart0 = d.xpos[cart_body].copy()
        trace = []
        t0 = time.monotonic(); tick = 0
        for i in range(n):
            cur["i"] = i
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=sinfo)
            obs, _r, _te, _tr, sinfo = env.step(action)
            _tick_clock(); carry()
            if os.environ.get("REPLAY_DUMP") == "1" and i in (250, 500, 750):
                a2 = replay_get_action()
                print(f"[dump] i={i} nav={np.round(a2['navigate_cmd'],4).tolist()} h={a2['base_height_command']:.4f} "
                      f"pose={np.round(a2['target_upper_body_pose'],4).tolist()}", flush=True)
            if os.environ.get("REPLAY_DEBUG") == "1" and i % 100 == 0:
                lb = agent._wbc_policy.lower_body_policy
                nav = replay_get_action()["navigate_cmd"]
                print(f"[dbg] t={i/HZ:5.1f} nav={np.round(nav,3).tolist()} upa={getattr(lb,'use_policy_action',None)} "
                      f"tp_active={tp.is_active} stab={robot.stabilized} pelvis_x={d.qpos[0]:+.3f} "
                      f"cart_x={d.xpos[cart_body][0]:+.3f}", flush=True)
            sim_h = wrap(yaw_of_wxyz(d.qpos[3:7]) - heading0)
            t_i = i / HZ; ri = row_at(t_i)
            rec = {"i": i, "row": ri, "t": round(t_i, 3), "sim_heading_rel": round(float(sim_h), 4),
                   "cmd_rel_yaw": round(float(NAV[ri, 3]), 4), "vx": round(float(NAV[ri, 0]), 3),
                   "sim_xy": (d.qpos[:2] - xy0).round(4).tolist(),
                   "target_xyz": d.xpos[tgt_body].round(4).tolist(), "target_tilt_deg": round(tilt_deg(d.xquat[tgt_body]), 1),
                   "cart_xy": (d.xpos[cart_body][:2] - cart0[:2]).round(4).tolist(),
                   "palm_r": d.xpos[palm_body].round(4).tolist(),
                   "tips": {k: d.xpos[v].round(4).tolist() for k, v in tip_bodies.items()},
                   "base_yaw": round(float(yaw_of_wxyz(d.qpos[3:7])), 4), "base_xy_abs": d.qpos[:2].round(4).tolist(),
                   "task_in_hand": bool((sinfo.get("progress") or {}).get("cup_in_hand", False)),
                   "task_success": bool((sinfo.get("progress") or {}).get("success", False)),
                   "hand_bar_contacts": int(sum(1 for c in d.contact[:d.ncon]
                                                if (c.geom1 in bar_geoms and c.geom2 in hand_geoms_all) or (c.geom2 in bar_geoms and c.geom1 in hand_geoms_all))),
                   "bar_z": round(float(np.mean([d.geom_xpos[g][2] for g in bar_geoms])), 4) if bar_geoms else None}
            if not fast:
                rec.update({"sim_arm": d.qpos[mj_arm].round(4).tolist(),
                            "cmd_arm": act_at(t_i)[[act_names.index(nm) for nm in ARM_JOINTS]].round(4).tolist(),
                            "real_arm": STATE[ri, st_arm].round(4).tolist(),
                            "key_sim": {k: round(float(d.qpos[v]), 4) for k, v in mj_key.items()},
                            "key_cmd": {k: round(float(ACT[ri, v]), 4) for k, v in key_act.items()},
                            "key_real": {k: round(float(STATE[ri, v]), 4) for k, v in key_st.items()}})
            trace.append(rec)
            if video and i % 2 == 0:
                emit(f"ep {ep}  t={t_i:5.1f}s  vx {NAV[ri, 0]:+.2f} yaw cmd {NAV[ri, 3]:+.2f} | sim heading {sim_h:+.2f} | "
                     f"base {np.round(d.qpos[:2] - xy0, 2).tolist()} | cup lift {d.xpos[tgt_body][2] - tgt0[2]:+.2f} m", t_i)
            tick += 1
            if not fast:
                sl = t0 + tick / HZ - time.monotonic(); (sl > 0) and time.sleep(sl)
        for j in range(int(SETTLE_S * HZ)):
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=sinfo)
            obs, _r, _te, _tr, sinfo = env.step(action)
            _tick_clock(); carry()
            if video and j % 2 == 0:
                emit(f"ep {ep}  settle {j / HZ:4.1f}s | cup {np.round(d.xpos[tgt_body], 2).tolist()} tilt {tilt_deg(d.xquat[tgt_body]):.0f} deg", n / HZ)
            tick += 1
            if not fast:
                sl = t0 + tick / HZ - time.monotonic(); (sl > 0) and time.sleep(sl)
        if video:
            for pkt in vs.encode(): c.mux(pkt)
            c.close(); rc.close()

        # ---- outcome ------------------------------------------------------------------
        lift = np.array([r["target_xyz"][2] - tgt0[2] for r in trace])
        tilt = np.array([r["target_tilt_deg"] for r in trace])
        k = int(lift.argmax())
        ex, ey, ez = d.xpos[tgt_body]
        end_tilt = tilt_deg(d.xquat[tgt_body])
        # 0.035 m: the table top is 0.74 and the cart's box top is 0.80, so a cup still riding the cart must NOT pass
        on_table = bool(L["table_near_x"] - 0.02 <= ex <= L["table_far_x"] + 0.02
                        and L["table_right_y"] - 0.02 <= ey <= L["table_left_y"] + 0.02
                        and abs(ez - T.BS.TABLE_H) < 0.035 and end_tilt < 45)
        still_on_cart = bool(abs(ez - L["box_top"]) < 0.06 and end_tilt < 45
                             and L["box_near_x"] - 0.4 <= ex - (d.xpos[cart_body][0] - cart0[0]) <= L["box_far_x"] + 0.4)
        base_end = (d.qpos[:2] - xy0).round(3).tolist()
        task_pr = dict(sinfo.get("progress") or {})
        summary = {"ep": ep, "task_success": bool(task_pr.get("success", False)), "task_progress": task_pr,
                   "robot_to_cart": checks["robot_to_cart"], "route_forward": checks["route_forward"],
                   "table_from_line": checks["table_from_line"],
                   "cart_mass": checks["cart_mass_kg"], "wheel_friction": checks["wheel_friction"],
                   "grasped": bool(lift[k] >= 0.05 and tilt[k] < 60), "max_lift_m": round(float(lift[k]), 3),
                   "on_table": on_table, "still_on_cart": still_on_cart,
                   "cup_at_reset": cup_at_reset.round(3).tolist(),
                   "cup_settle_shift_mm": round(float(np.linalg.norm(tgt0 - cup_at_reset) * 1000), 1),
                   "cup_start": tgt0.round(3).tolist(), "cup_end": [round(float(ex), 3), round(float(ey), 3), round(float(ez), 3)],
                   "cup_end_tilt_deg": round(end_tilt, 1),
                   "cup_end_minus_table_centre": [round(float(ex - L["table_cx"]), 3), round(float(ey - L["table_cy"]), 3),
                                                  round(float(ez - T.BS.TABLE_H), 3)],
                   "base_end_xy": base_end, "base_travel_x": base_end[0],
                   "sim_heading_end": round(float(wrap(yaw_of_wxyz(d.qpos[3:7]) - heading0)), 3),
                   "cmd_heading_end": round(float(NAV[-1, 3]), 3),
                   "cart_end_xy": (d.xpos[cart_body][:2] - cart0[:2]).round(3).tolist(),
                   "table_near_x": L["table_near_x"], "table_centre": [L["table_cx"], L["table_cy"]],
                   "scene": checks}
        # A robot that never moved at all is a broken worker (the env's DDS channel factory fails when too many
        # processes share the domain), not an outcome: every real episode walks >= 1.3 m. Leave no summary so a
        # resumed batch reruns it.
        if abs(base_end[0]) < 0.05 and abs(base_end[1]) < 0.05 and abs(summary["sim_heading_end"]) < 0.05:
            print(f"[replay] ep {ep}: DEAD RUN (robot never moved) - no summary written", flush=True)
            summary["dead"] = True
        print("[replay] summary:", json.dumps(summary), flush=True)
        trace.append({"summary": summary})
    finally:
        try:
            if r3 is not None: r3.close()
        except Exception:
            pass
        env.close()
    return trace


def run_worker(a):
    tag = a.tag
    tr = replay_episode(a.episodes[0], OUT / f"replay_ep{a.episodes[0]}_{tag}.mp4", fast=a.fast, video=(a.video or not a.fast))
    json.dump(tr, open(OUT / f"trace_ep{a.episodes[0]}_{tag}.json", "w"))
    summ = next((r["summary"] for r in tr if "summary" in r), {})
    if summ.get("dead"):
        sys.stdout.flush(); os._exit(3)
    json.dump(summ, open(OUT / f"summary_ep{a.episodes[0]}_{tag}.json", "w"), indent=1)
    print(f"[replay] episode {a.episodes[0]} done", flush=True)
    sys.stdout.flush(); os._exit(0)


def spawn(ep, tag, cfg, fast, timeout=2400):
    """One subprocess per episode (SDK teardown + Pico port), as in the sibling scenes."""
    import subprocess
    cmd = [sys.executable, "-u", __file__, "--worker", "--episodes", str(ep), "--tag", tag,
           "--robot-to-cart", str(cfg["robot_to_cart"]), "--route-forward", str(cfg["route_forward"]),
           "--table-from-line", str(cfg["table_from_line"]), "--cup-mass", str(cfg["cup_mass"]),
           "--cart-mass", str(cfg["cart_mass"]), "--wheel-friction", str(cfg["wheel_friction"]), "--handle-top", str(cfg["handle_top"]),
           "--nav-gain", str(cfg["nav_gain"]), "--walk-stretch", str(cfg["walk_stretch"]), "--post-turn-gain", str(cfg.get("post_turn_gain", 1.0)),
           "--waist-mode", cfg.get("waist_mode", "freeze_while_walking"),
           "--cart-attach", cfg.get("cart_attach", "base")]
    if cfg.get("cup_xy"):
        cmd += ["--cup-xy", str(cfg["cup_xy"][0]), str(cfg["cup_xy"][1])]
    if fast:
        cmd.append("--fast")
    if cfg.get("video"):
        cmd.append("--video")
    if cfg.get("wall_clock"):
        cmd.append("--wall-clock")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "logs" / f"{tag}_ep{ep}.txt", "w")          # keep every worker's output: a dead robot explains itself here
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def run_batch(jobs, fast, jobs_parallel):
    """jobs: list of (ep, tag, cfg). Runs up to jobs_parallel subprocesses at a time."""
    skipped = [j for j in jobs if (OUT / f"summary_ep{j[0]}_{j[1]}.json").exists()]     # resume: keep finished runs
    pending = [j for j in jobs if j not in skipped]; running = []; done = len(skipped)
    if skipped:
        print(f"[batch] {len(skipped)}/{len(jobs)} already done, resuming the rest", flush=True)
    while pending or running:
        while pending and len(running) < jobs_parallel:
            ep, tag, cfg = pending.pop(0)
            running.append((ep, tag, spawn(ep, tag, cfg, fast), time.monotonic()))
        time.sleep(1.0)
        for item in list(running):
            ep, tag, p, t_start = item
            if p.poll() is not None:
                running.remove(item); done += 1
                ok = (OUT / f"summary_ep{ep}_{tag}.json").exists()
                print(f"[batch] {done}/{len(jobs)}  ep {ep} {tag}: exit {p.returncode} {'ok' if ok else 'NO SUMMARY'} "
                      f"({time.monotonic() - t_start:.0f}s)", flush=True)
            elif time.monotonic() - t_start > 2400:
                p.kill(); running.remove(item); done += 1
                print(f"[batch] {done}/{len(jobs)}  ep {ep} {tag}: TIMEOUT", flush=True)


def collect(tag):
    out = []
    for f in sorted(OUT.glob(f"summary_ep*_{tag}.json")):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fast", action="store_true", help="no wall-clock pacing (sweeps); deterministic")
    ap.add_argument("--wall-clock", action="store_true", help="let the agent read the real clock (the old, load-dependent behaviour)")
    ap.add_argument("--video", action="store_true", help="record the mp4 even with --fast (a deterministic presentation run)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel episode processes (keep the TOTAL on the machine <= 10: the env's DDS domain overflows beyond ~12 and the robot then ignores every command)")
    ap.add_argument("--robot-to-cart", type=float, default=None)
    ap.add_argument("--route-forward", type=float, default=None)
    ap.add_argument("--table-from-line", type=float, default=None)
    ap.add_argument("--cup-xy", type=float, nargs=2, default=None, metavar=("X", "Y"))
    ap.add_argument("--cup-mass", type=float, default=None)
    ap.add_argument("--cart-mass", type=float, default=None)
    ap.add_argument("--wheel-friction", type=float, default=None)
    ap.add_argument("--handle-top", type=float, default=None, help="grip bar height (m); default from layout.json")
    ap.add_argument("--cart-attach", default="none", choices=("base", "none"),
                    help="carry the cart rigidly with the base (the walk policy cannot push it)")
    ap.add_argument("--waist-mode", default="freeze_while_walking", choices=("track", "freeze", "freeze_while_walking"))
    ap.add_argument("--nav-gain", type=float, default=1.0)
    ap.add_argument("--post-turn-gain", type=float, default=1.0, help="scale vx/vy after the commanded turn only")
    ap.add_argument("--walk-stretch", type=float, default=1.0)
    ap.add_argument("--sweep-robot-to-cart", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-route", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-table-from-line", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-post-turn-gain", type=float, nargs="+", default=None)
    ap.add_argument("--table-pairs", type=float, nargs="+", default=None, metavar="V",
                    help="explicit (route, table_from_line) pairs instead of the cross product: r1 t1 r2 t2 ...")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)

    lay = json.load(open(HERE / "layout.json"))["layout"]
    base = dict(robot_to_cart=a.robot_to_cart if a.robot_to_cart is not None else lay["robot_to_cart"],
                route_forward=a.route_forward if a.route_forward is not None else lay["route_forward"],
                table_from_line=a.table_from_line if a.table_from_line is not None else lay["table_from_line"],
                cup_mass=a.cup_mass if a.cup_mass is not None else json.load(open(HERE / "layout.json"))["cup"]["mass"],
                cart_mass=a.cart_mass if a.cart_mass is not None else json.load(open(HERE / "layout.json"))["cart"]["mass"],
                wheel_friction=a.wheel_friction if a.wheel_friction is not None else json.load(open(HERE / "layout.json"))["cart"]["wheel_friction"],
                handle_top=a.handle_top if a.handle_top is not None else json.load(open(HERE / "layout.json"))["cart"]["handle_top"],
                nav_gain=a.nav_gain, walk_stretch=a.walk_stretch, post_turn_gain=a.post_turn_gain, cup_xy=a.cup_xy, waist_mode=a.waist_mode, cart_attach=a.cart_attach)

    os.environ["REPLAY_NAV_GAIN"] = f"{base['nav_gain']:.3f}"
    os.environ["REPLAY_WALK_STRETCH"] = f"{base['walk_stretch']:.3f}"
    os.environ["REPLAY_POST_TURN_GAIN"] = f"{base['post_turn_gain']:.3f}"
    os.environ["COFFEE_CART_ROBOT_TO_CART"] = f"{base['robot_to_cart']:.4f}"
    os.environ["COFFEE_CART_ROUTE_FORWARD"] = f"{base['route_forward']:.4f}"
    os.environ["COFFEE_CART_TABLE_FROM_LINE"] = f"{base['table_from_line']:.4f}"
    os.environ["COFFEE_CART_CUP_MASS"] = f"{base['cup_mass']:.4f}"
    os.environ["COFFEE_CART_CART_MASS"] = f"{base['cart_mass']:.4f}"
    os.environ["COFFEE_CART_WHEEL_FRICTION"] = f"{base['wheel_friction']:.4f}"
    os.environ["COFFEE_CART_HANDLE_TOP"] = f"{base['handle_top']:.4f}"
    if base["cup_xy"]:
        os.environ["COFFEE_CART_CUP_XY"] = f"{base['cup_xy'][0]:.4f},{base['cup_xy'][1]:.4f}"
    os.environ["REPLAY_WAIST_MODE"] = a.waist_mode
    os.environ["REPLAY_SIM_CLOCK"] = "0" if a.wall_clock else "1"
    os.environ["REPLAY_CART_ATTACH"] = a.cart_attach
    if a.fast:
        os.environ["REPLAY_FAST"] = "1"

    if a.probe:
        probe(OUT / "probe")
    if a.worker:
        run_worker(a); return

    def cfg_tag(c):
        n = lambda v: ("m" if v < 0 else "") + f"{abs(round(v * 100)):03d}"
        g = c.get("post_turn_gain", 1.0)
        return f"rc{n(c['robot_to_cart'])}_rt{n(c['route_forward'])}_tl{n(c['table_from_line'])}" + (f"_pg{round(g*10):02d}" if g != 1.0 else "")
    if a.sweep_robot_to_cart or a.sweep_route or a.sweep_table_from_line or a.sweep_post_turn_gain or a.table_pairs:
        rcs = a.sweep_robot_to_cart or [base["robot_to_cart"]]
        pgs = a.sweep_post_turn_gain or [base["post_turn_gain"]]
        if a.table_pairs:
            tables = list(zip(a.table_pairs[0::2], a.table_pairs[1::2]))
        else:
            tables = [(rt, tl) for rt in (a.sweep_route or [base["route_forward"]]) for tl in (a.sweep_table_from_line or [base["table_from_line"]])]
        jobs = []
        for rc in rcs:
            for pg in pgs:
                for rt, tl in tables:
                    c = dict(base, robot_to_cart=rc, route_forward=rt, table_from_line=tl, post_turn_gain=pg)
                    for ep in a.episodes:
                        jobs.append((ep, a.tag or cfg_tag(c), c))
        print(f"[sweep] {len(rcs)} rc x {len(pgs)} gains x {len(tables)} tables x {len(a.episodes)} episodes = {len(jobs)} runs, {a.jobs} at a time", flush=True)
        run_batch(jobs, a.fast, a.jobs)
        tags = sorted({t for _, t, _ in jobs})
        report = []
        for t in tags:
            rs = collect(t)
            if not rs: continue
            report.append({"tag": t, "n": len(rs),
                           "robot_to_cart": rs[0]["robot_to_cart"], "route_forward": rs[0]["route_forward"],
                           "table_from_line": rs[0]["table_from_line"], "post_turn_gain": rs[0]["scene"].get("post_turn_gain", 1.0),
                           "task_success": sum(r.get("task_success", False) for r in rs),
                           "cart_pushed": sum((r.get("task_progress") or {}).get("cart_pushed_ever", False) for r in rs),
                           "cup_lifted": sum((r.get("task_progress") or {}).get("cup_lifted_ever", False) for r in rs),
                           "grasped": sum(r["grasped"] for r in rs), "on_table": sum(r["on_table"] for r in rs),
                           "mean_base_travel_x": round(float(np.mean([r["base_travel_x"] for r in rs])), 3),
                           "mean_cart_dx": round(float(np.mean([r["cart_end_xy"][0] for r in rs])), 3),
                           "mean_heading_end": round(float(np.mean([r["sim_heading_end"] for r in rs])), 3)})
        report.sort(key=lambda r: (-r["task_success"], -r["on_table"], -r["grasped"]))
        json.dump(report, open(OUT / "sweep_report.json", "w"), indent=1)
        print("\n[sweep] RESULTS (best first):"); print(json.dumps(report, indent=1), flush=True)
        return

    tag = a.tag or cfg_tag(base)
    run_batch([(ep, tag, base) for ep in a.episodes], a.fast, a.jobs)
    rs = collect(tag)
    print(f"\n[replay] tag {tag}: {len(rs)} episodes, task success {sum(r.get('task_success', False) for r in rs)}, "
          f"cart pushed {sum((r.get('task_progress') or {}).get('cart_pushed_ever', False) for r in rs)}, "
          f"cup lifted {sum((r.get('task_progress') or {}).get('cup_lifted_ever', False) for r in rs)}, geometric on_table {sum(r['on_table'] for r in rs)}", flush=True)
    for r in rs:
        print(json.dumps({k: r[k] for k in ("ep", "grasped", "max_lift_m", "on_table", "cup_end", "base_end_xy", "sim_heading_end", "cmd_heading_end")}), flush=True)


if __name__ == "__main__":
    main()
