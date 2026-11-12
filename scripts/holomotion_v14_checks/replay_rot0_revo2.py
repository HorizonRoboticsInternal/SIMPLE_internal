#!/usr/bin/env python
"""Reference-action replay of rot0 teleop episodes on the Revo 2 robot (2026-10-23, user: "replay a few holomotion 50 ep to see if
the brainco hands were replaced, can replay still complete the tasks").

Closed loop as rerecord_holomotion_v14: HoloMotion v1.4 runs live on the recorded inputs (reference frames, PICO samples, clock,
frame-0 controller snapshot) in the recorded scene, but on data/robots/g1_rot0_teleop_revo2 (robot "g1_sonic_revo2"). The
recorded frame-0 physics state is mapped onto the new model joint by joint (root, 29 body joints, the objects); the hands start
open. The PICO grip bits drive the Revo 2 in grip mode (simple.robots.g1_sonic_revo2.revo2_targets) instead of the Dex3 poses.
Writes <out>/videos/ep<N>.mp4 (head camera | tracking camera, 25 fps), <out>/img/ep<N>_grasp.jpg (the right hand 0.5 s after the
first grip close) and <out>/results.json (gates, success, bottle lift, contacts, pelvis height) per episode.
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/replay_rot0_revo2.py <rot0 dataset dir> --episodes 0,10,20,30,40 --out docs/rot0_revo2
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--episodes", default="0,10,20,30,40"); ap.add_argument("--out", default="docs/rot0_revo2")
ap.add_argument("--robot-mjcf", default="robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_v3.xml"); ap.add_argument("--video-every", type=int, default=2)
ap.add_argument("--closeups", default="", help="comma list of times (s) at which to save a right-hand close-up, img/ep<N>_t<time>.jpg")
ap.add_argument("--touch-log", default="", help="log the right hand's Revo 2 Touch sensors (simple.sensors.revo2_touch) every frame to this JSON"); ap.add_argument("--contact-dump", default="", help="write every right-hand/bottle contact (body, point and normal in the body frame, normal force) per frame to this JSON"); ap.add_argument("--settle-s", type=float, default=2.0, help="after the last recorded frame keep stepping with the last action for this long (the recordings end right after the release; the placed gate needs 0.5 s in the bin)"); ap.add_argument("--stop-s", type=float, default=0.0, help="stop the episode after this many seconds (0 = full episode)")
ap.add_argument("--no-timeline", action="store_true", help="skip the per-frame bottle z / hand contact / grip bit / right-hand joint error in results.json")
ap.add_argument("--grip-poses", default="", help="JSON {open: [11], closed: [11], threshold} to use instead of the robot's grip poses (data/robots/g1_rot0_teleop_revo2/grip_poses.json or the module defaults)")
a = ap.parse_args(); SRC = Path(a.dataset).resolve(); OUT = (ROOT / a.out).resolve(); (OUT / "videos").mkdir(parents=True, exist_ok=True); (OUT / "img").mkdir(exist_ok=True)
EPS = [int(x) for x in a.episodes.split(",")]

import mujoco, pyarrow.parquet as pq, imageio.v2 as imageio
from PIL import Image
from simple.cli import teleop_holomotion_v14 as T
from simple.cli.replay_holomotion_v14 import _build_env, _reset_to
from simple.cli.rerecord_holomotion_v14 import _ReplaySubscriber, _pico_sample
from simple.teleop.holomotion_v14 import exact_log as XL, node_state as NS
from simple.teleop.holomotion_v14.robot_variants import tilt_head_sensor
from simple.robots.g1_sonic_revo2 import revo2_targets, OPEN_Q, CLOSED_Q, LEFT_REVO2_JOINTS, RIGHT_REVO2_JOINTS, set_grip_poses, GRIP_POSES
if a.grip_poses:
    _gp = json.load(open(a.grip_poses)); set_grip_poses(_gp["open"], _gp["closed"], _gp.get("threshold")); print(f"[revo2] grip poses from {a.grip_poses}", flush=True)
elif GRIP_POSES: print(f"[revo2] grip poses from {GRIP_POSES['path']}", flush=True)
print(f"[revo2] open {np.round(OPEN_Q, 3).tolist()}\n[revo2] closed {np.round(CLOSED_Q, 3).tolist()}", flush=True)


def joint_layout(m):
    out = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"__free{j}"
        nq = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(m.jnt_type[j]), 1)
        nv = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}.get(int(m.jnt_type[j]), 1)
        out[n] = (int(m.jnt_qposadr[j]), nq, int(m.jnt_dofadr[j]), nv)
    return out


def build_revo2_env(info, setup):
    """_build_env with the Revo 2 robot: same scene, env knobs and camera tilt; robot_uid g1_sonic_revo2 + its MJCF."""
    import gymnasium as gym
    from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene
    for k, v in info.get("env_knobs", {}).items(): os.environ[k] = v
    if info.get("scene_root") and "HOLOBRAIN_SIM_DIR" not in os.environ and os.path.isdir(info["scene_root"]): os.environ["HOLOBRAIN_SIM_DIR"] = info["scene_root"]
    mod = load_scene(info["scene"]); sonic_config = _load_sonic_config()
    env = gym.make(info["env_id"], sim_mode="mujoco", render_hz=info["render_hz"], physics_dt=info["physics_dt"], headless=True, max_episode_steps=10 ** 9,
                   sonic_config=sonic_config, target=getattr(mod, "TARGET", None), dr_level=info["dr_level"], success_criteria=0.9, robot_uid="g1_sonic_revo2")
    sonic_env = env.unwrapped; robot = sonic_env.task.robot; robot.mjcf_path = a.robot_mjcf
    tilt = float(info.get("head_tilt_deg", 0.0))
    if tilt: tilt_head_sensor(sonic_env.task, tilt)
    _reset_to(env, sonic_env, mod, info, setup)
    return env, sonic_env, mod


meta_lines = [json.loads(l) for l in (SRC / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
results = []; env = None; rec_layout = None
for ep in EPS:
    meta = meta_lines[ep]; log = XL.load(SRC / "replay" / f"episode_{ep:06d}.npz"); info, setup = dict(log["info"]), log["setup"]; snap = NS.unpack(log)
    t = pq.read_table(str(SRC / "data/chunk-000" / f"episode_{ep:06d}.parquet")); n = t.num_rows
    col = lambda c: np.stack(t[c].to_pandas().values).reshape(n, -1)  # noqa: E731
    pico, tick_time = col("holomotion.pico"), col("holomotion.tick_time")[:, 0]
    K = col("holomotion.MotionReference.frames").shape[1] // 65
    frames = col("holomotion.MotionReference.frames").reshape(n, K, 65); fmeta = col("holomotion.MotionReference.frames_meta").reshape(n, K, 2); cnt = col("holomotion.MotionReference.frames_this_tick")[:, 0].astype(int)
    init_pose = meta.get("init_pose"); init_pose = json.loads(init_pose) if isinstance(init_pose, str) else init_pose
    if rec_layout is None:                                       # the recorded (Dex3) model's joint layout, from a throw-away build of the recorded scene
        if info.get("policy_models_dir"): os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = info["policy_models_dir"]
        e0, s0, _ = _build_env(info, setup, check_legacy=False); m0 = s0.mujoco.mjModel
        rec_layout, rec_nq, rec_nv = joint_layout(m0), int(m0.nq), int(m0.nv)
        assert (rec_nq, rec_nv) == (info["sizes"]["nq"], info["sizes"]["nv"]), "recorded model rebuild differs from the log's sizes"
        e0.close(); del e0, s0
        from simple.tasks.registry import TaskRegistry                         # the task registry caches one instance per uid (and that
        TaskRegistry._instances.clear()                                        # instance holds the Dex3 robot): the Revo 2 env builds its own
    if env is None:
        env, sonic_env, mod = build_revo2_env(info, setup); robot = sonic_env.task.robot
        from simple.agents.holomotion_v14_agent import HoloMotionV14Agent
        agent = HoloMotionV14Agent(robot, enable_pico_stream=False, button_map="sim", quiet=True); agent.sub.close(); agent.sub = _ReplaySubscriber()
        from simple.teleop.holomotion_v14.sim_clock import ControllerClock
        clock = ControllerClock(t0=float(tick_time[0])); clock.install()
        print(f"[revo2] robot {robot.mjcf_path}: {sonic_env.mujoco.mjModel.nq} qpos ({len(robot.hand_names)} hand joints), recorded model {rec_nq} qpos", flush=True)
    else:
        _reset_to(env, sonic_env, mod, info, setup)
    task, m, d, eng = sonic_env.task, sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData, sonic_env.mujoco
    # ---- start: the live quick start, then the recorded frame-0 state mapped joint by joint
    agent.reset_policy(); T._stand_on_floor(sonic_env, agent, init_pose)
    st0 = log["state0"]; q0, v0 = st0[1:1 + rec_nq], st0[1 + rec_nq:1 + rec_nq + rec_nv]   # INTEGRATION state: time, qpos, qvel, ...
    lay = joint_layout(m); mapped, hands = 0, 0
    for name, (qa, nq, da, nv) in lay.items():
        if name in LEFT_REVO2_JOINTS or name in RIGHT_REVO2_JOINTS: continue
        if name in rec_layout:
            rqa, rnq, rda, rnv = rec_layout[name]
            if rnq == nq: d.qpos[qa:qa + nq] = q0[rqa:rqa + rnq]; d.qvel[da:da + nv] = v0[rda:rda + rnv]; mapped += 1
    for names in (LEFT_REVO2_JOINTS, RIGHT_REVO2_JOINTS):
        for jn, qv in zip(names, OPEN_Q): d.qpos[lay[jn][0]] = qv; d.qvel[lay[jn][2]] = 0.0; hands += 1
    mujoco.mj_forward(m, d)
    task._start_xy = np.asarray(d.qpos[:2], dtype=float).copy()
    XL.restore_band(robot, log["action_band_state"][0]); eng.render_step = int(log["render_step0"]); NS.restore(agent.node, snap)
    A = json.loads(str(snap["agent_json"])) if "agent_json" in snap else None
    if A is not None:
        agent.main_state, agent._last_bits, agent._dropping = A["main_state"], int(A["last_bits"]), bool(A["dropping"]); agent._state_t0 = float(tick_time[0]) - float(A.get("state_age", 0.0))
        arr = lambda v: None if v is None else np.asarray(v, dtype=np.float32)  # noqa: E731
        agent._left_hand, agent._right_hand = arr(A["left_hand"]), arr(A["right_hand"]); agent._hold_real, agent._mtd_from = arr(A["hold_real"]), arr(A["mtd_from"]); agent._quick = None
    else:
        agent.main_state, agent._state_t0, agent._quick, agent._dropping, agent._last_bits = "POLICY", 0.0, None, False, 0; agent._hold_real = agent._mtd_from = None
    observation, privileged_info = sonic_env._get_obs(), sonic_env._get_info()
    # ---- video: head camera | tracking camera; grasp still from a hand camera
    writer = imageio.get_writer(str(OUT / "videos" / f"ep{ep}.mp4"), fps=int(info["render_hz"]) // a.video_every, codec="libx264", quality=7, macro_block_size=1)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 1280); m.vis.global_.offheight = max(int(m.vis.global_.offheight), 720)
    rend = mujoco.Renderer(m, 360, 640); head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "hbvcam_left_pinhole")
    third = mujoco.MjvCamera(); third.type, third.distance, third.azimuth, third.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, 3.0, 150, -25; third.trackbodyid = m.body("pelvis").id
    vt = getattr(mod, "VIEW_THIRD", None) if "mod" in dir() else None
    if vt:                                                                              # the scene's own overview angle (kitchen, 2026-10-26)
        third.type = mujoco.mjtCamera.mjCAMERA_FREE; third.lookat[:] = vt["lookat"]; third.distance, third.azimuth, third.elevation = vt["distance"], vt["azimuth"], vt["elevation"]
    rwrist = m.body("right_wrist_yaw_link").id
    def hand_still(path):                                        # the three close-up views of dex3_grasp_stills.py (lookat 9 cm along the forearm)
        look = d.xpos[rwrist] + d.xmat[rwrist].reshape(3, 3) @ np.array([0.09, 0.0, 0.0]); ims = []
        for az, el, dist in [(150, -35, 0.45), (90, -80, 0.5), (200, -10, 0.45)]:
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = look; cam.distance, cam.azimuth, cam.elevation = dist, az, el
            hrend.update_scene(d, camera=cam); ims.append(hrend.render().copy())
        Image.fromarray(np.concatenate(ims, axis=1)).save(path, quality=88)
    hrend = mujoco.Renderer(m, 360, 480)
    closeup_frames = {int(round(float(x) * info["render_hz"])): float(x) for x in a.closeups.split(",") if x.strip()}
    stop_frame = int(round(a.stop_s * info["render_hz"])) if a.stop_s > 0 else n
    rq = [lay[j][0] for j in RIGHT_REVO2_JOINTS]; tl = None if a.no_timeline else dict(bottle_z=[], hand=[], grip=[], rhand_err=[], rhand_q=[], bottle_contacts=[], bottle_pos=[], bottle_tilt=[])
    bname = {b: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(m.nbody)}
    def bottle_touching():                                       # names of the non-bottle bodies in contact with the bottle this frame
        out = set()
        for ci in range(d.ncon):
            c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 == bid and b2 != bid: out.add(bname[b2])
            elif b2 == bid and b1 != bid: out.add(bname[b1])
        return sorted(out)
    def bottle_in_wrist():                                       # bottle centre in the right wrist_yaw frame (x = forearm axis, +y = palm side)
        R = d.xmat[rwrist].reshape(3, 3); return (R.T @ (d.xpos[bid] - d.xpos[rwrist])).round(3).tolist()
    wrist_notes = {}; cdump = []; tlog = []
    touch = None
    if a.touch_log:
        from simple.sensors.revo2_touch import Revo2Touch
        touch = Revo2Touch(m, d, "right")
    bid = m.body(str(task.target.asset.label)).id; pelvis = m.body("pelvis").id
    z0 = float(d.xpos[bid][2]); term_at = None; max_lift = 0.0; pel_min = 1.0; hand_s = 0.0; grip_close_frame = None; grasp_still = False; t0 = time.time()
    n_rec = min(n, stop_frame); settle_n = int(round(a.settle_s * info["render_hz"])) if a.settle_s > 0 else 0
    for i in range(n_rec + settle_n):                           # the recorded frames, then a settle tail: the recordings stop right after the
        live = i < n_rec; j = min(i, n_rec - 1)                  # operator's release, so the bottle gets time to land and the 0.5 s placed gate to fire
        if not live and term_at is not None: break
        if live:
            clock.now = float(tick_time[i]); kf = min(int(cnt[i]), K)
            agent.sub.set_tick([{"latest_obs": frames[i, j].astype(np.float32), "frame_index": np.array([int(fmeta[i, j, 0])]), "timestamp_ns": np.array([int(round(fmeta[i, j, 1] * 1e9))])} for j in range(kf)], _pico_sample(pico[i]))
            action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            P = getattr(action, "parameters", None)
            if P is not None:                                        # Revo 2 grip mode instead of the Dex3 poses
                P["left_hand_q"] = revo2_targets(pico[i][4], trigger=pico[i][6]); P["right_hand_q"] = revo2_targets(pico[i][5], trigger=pico[i][7])   # grip -> closed, trigger -> pinch
        else: clock.now = float(tick_time[n_rec - 1]) + (i - n_rec + 1) / info["render_hz"]   # hold the last action
        observation, reward, terminated, truncated, privileged_info = env.step(action)
        if terminated and term_at is None: term_at = i
        if touch is not None:
            ph = touch.physical(); tlog.append(dict(i=i, t=round(i / info["render_hz"], 3), grip=int(pico[j][5] > 0.5), sdk=touch.read(), physical=[dict(p, force_site=[round(x, 3) for x in p["force_site"]], normal_N=round(p["normal_N"], 3), tangential_N=round(p["tangential_N"], 3), direction_deg=None if p["direction_deg"] is None else round(p["direction_deg"], 1), proximity_m=None if p["proximity_m"] is None else round(p["proximity_m"], 4), touch_sensor_N=round(p["touch_sensor_N"], 3)) for p in ph], taxels_N=touch.taxels()[:, 0].round(3).tolist()))
        if a.contact_dump:
            f6 = np.zeros(6)
            for ci in range(d.ncon):
                c = d.contact[ci]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                other = b2 if b1 == bid else b1 if b2 == bid else None
                if other is None or not bname[other].startswith("right_hand"): continue
                mujoco.mj_contactForce(m, d, ci, f6); Rb = d.xmat[other].reshape(3, 3); nrm = c.frame[:3] * (1 if b1 == bid else -1)   # normal from the bottle into the hand body
                cdump.append(dict(i=i, body=bname[other], p=(Rb.T @ (c.pos - d.xpos[other])).round(4).tolist(), n=(Rb.T @ nrm).round(3).tolist(), fn=round(float(f6[0]), 3), dist=round(float(c.dist), 4)))
        if pico[j][5] > 0.5 and grip_close_frame is None: grip_close_frame = i; wrist_notes["bottle_in_wrist_at_close"] = bottle_in_wrist()
        if grip_close_frame is not None and i == grip_close_frame + 25: wrist_notes["bottle_in_wrist_close_plus_0.5s"] = bottle_in_wrist()
        max_lift = max(max_lift, float(d.xpos[bid][2]) - z0); pel_min = min(pel_min, float(d.xpos[pelvis][2]))
        task._bind_gate_ids(m); hand, _, _ = task._contacts(d); hand_s += (1.0 / info["render_hz"]) if hand else 0.0
        if tl is not None:
            tgt = revo2_targets(pico[j][5], trigger=pico[j][7]); q = d.qpos[rq]
            tl["bottle_z"].append(round(float(d.xpos[bid][2]), 4)); tl["bottle_pos"].append(d.xpos[bid].round(3).tolist()); tl["bottle_tilt"].append(round(float(np.degrees(np.arccos(np.clip(d.xmat[bid][8], -1, 1)))), 1)); tl["hand"].append(int(hand)); tl["grip"].append(int(pico[j][5] > 0.5)); tl["rhand_err"].append(round(float(np.abs(tgt - q).max()), 3)); tl["rhand_q"].append(q.round(3).tolist()); tl["bottle_contacts"].append(bottle_touching())
        if i in closeup_frames: hand_still(OUT / "img" / f"ep{ep}_t{closeup_frames[i]:.1f}.jpg")
        if i % a.video_every == 0:
            rend.update_scene(d, camera=head); h = rend.render(); rend.update_scene(d, camera=third); writer.append_data(np.concatenate([h, rend.render()], axis=1))
        if not grasp_still and grip_close_frame is not None and i == grip_close_frame + 25:
            hand_still(OUT / "img" / f"ep{ep}_grasp.jpg"); grasp_still = True
    writer.close()
    if touch is not None: json.dump(dict(episode=ep, hz=info["render_hz"], fingers=list(f["name"] for f in touch.fingers), frames=tlog), open(a.touch_log if len(EPS) == 1 else a.touch_log.replace(".json", f"_ep{ep}.json"), "w"))
    if a.contact_dump: json.dump(dict(episode=ep, hz=info["render_hz"], contacts=cdump), open(a.contact_dump if len(EPS) == 1 else a.contact_dump.replace(".json", f"_ep{ep}.json"), "w"))
    Pg = dict(getattr(task, "_progress", {}) or {}); gates = {g: bool(Pg.get(g)) for g in ("grasped", "at_bin", "placed") if g in Pg}
    has_bin = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trash_bin") >= 0
    if has_bin: bin_b = m.body("trash_bin").id; bin_pos = d.xpos[bin_b]
    else:                                                        # kitchen: the sink basin centre stands in for the bin
        sk = getattr(mod, "SINK", None); bin_pos = np.array([sk["cx"], sk["cy"], 0.0]) if sk else np.zeros(3)
    bottle_end = (d.xpos[bid] - bin_pos).round(3).tolist()
    try:                                                     # the task's own whole-bottle-in-bin test on the final state, with how far the bottle sticks out
        if not has_bin: raise LookupError("no trash_bin body")
        tp = np.concatenate([d.xpos[bid], d.xquat[bid]]); bq = d.xquat[bin_b]; w_, x_, y_, z_ = bq
        yaw = float(np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))); Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, tp[3:7])
        V = tp[:3] + task._g_target_verts @ Rm.reshape(3, 3).T; off = V[:, :2] - d.xpos[bin_b][:2]; cy, sy = np.cos(-yaw), np.sin(-yaw)
        lx = off[:, 0] * cy - off[:, 1] * sy; ly = off[:, 0] * sy + off[:, 1] * cy
        hx, hy = 0.24 / 2 - 0.008 + task.BIN_TOL, 0.36 / 2 - 0.008 + task.BIN_TOL
        end_in_bin = dict(in_bin=bool(task._in_bin(tp, d.xpos[bin_b], bq)), excess_cm=[round(100 * float(np.abs(lx).max() - hx), 1), round(100 * float(np.abs(ly).max() - hy), 1)],
                          lowest_minus_rim_cm=round(100 * float(V[:, 2].min() - d.xpos[bin_b][2] - 0.39), 1), highest_minus_rim_cm=round(100 * float(V[:, 2].max() - d.xpos[bin_b][2] - 0.39), 1))
    except Exception as ex: end_in_bin = dict(error=str(ex))
    bin_info = None if not has_bin else dict(body_pos=d.xpos[bin_b].round(3).tolist(), geoms=[dict(type=int(m.geom_type[g]), size=m.geom_size[g].round(3).tolist(), pos=d.geom_xpos[g].round(3).tolist(), zaxis=d.geom_xmat[g].reshape(3, 3)[:, 2].round(2).tolist()) for g in range(m.ngeom) if m.geom_bodyid[g] == bin_b])
    src_summary = ({r["new_episode"]: r for r in json.load(open(SRC / "meta/rerecord_summary.json"))["episodes"]}.get(ep, {}) if (SRC / "meta/rerecord_summary.json").exists() else {})
    row = dict(episode=ep, frames=int(n), settle_frames=int(settle_n), success=term_at is not None, success_frame=term_at, gates=gates, bottle_max_lift_m=round(max_lift, 3), bottle_end_minus_bin_m=bottle_end,
               bottle_end_z=round(float(d.xpos[bid][2]), 3), bottle_end_tilt_deg=round(float(np.degrees(np.arccos(np.clip(d.xmat[bid][8], -1, 1)))), 1), bottle_z0=round(z0, 3), end_in_bin=end_in_bin, pelvis_min_z=round(pel_min, 3), hand_on_bottle_s=round(hand_s, 2), grip_close_s=None if grip_close_frame is None else round(grip_close_frame / info["render_hz"], 2),
               joints_mapped_from_recording=mapped, hand_joints_set_open=hands, source_success=src_summary.get("success_on_last_frame"), source_saved_by_hand=src_summary.get("source_saved_by_hand"),
               video=f"videos/ep{ep}.mp4", grasp_still=f"img/ep{ep}_grasp.jpg" if grasp_still else None, closeups=[f"img/ep{ep}_t{t:.1f}.jpg" for f, t in sorted(closeup_frames.items()) if f < min(n, stop_frame)],
               timeline=tl, **wrist_notes, seconds=round(time.time() - t0, 1))
    results.append(row); json.dump(dict(robot=a.robot_mjcf, dataset=str(SRC), bin=bin_info, episodes=results), open(OUT / "results.json", "w"), indent=1)
    print(f"[revo2] ep {ep}: {n} frames, {'SUCCESS at frame %d' % term_at if term_at is not None else 'no success'}, gates {gates}, lift {max_lift:.2f} m, hand on bottle {hand_s:.1f} s, pelvis min {pel_min:.2f} m ({time.time() - t0:.0f} s)", flush=True)
print(f"[revo2] done: {sum(r['success'] for r in results)}/{len(results)} placed -> {OUT}", flush=True); sys.stdout.flush(); os._exit(0)
