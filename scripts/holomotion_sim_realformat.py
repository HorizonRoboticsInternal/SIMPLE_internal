"""Add the real G1 recorder's extra fields to a HoloMotion v14 sim-teleop LeRobot dataset.

    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_sim_realformat.py <dataset> [--episodes 0,1] [--out DIR]

<dataset> is a LeRobot v2.1 dir with replay/ (e.g. data/teleop_holomotion_v14/simple/G1WholebodyBottleBinTeleop-v0/level-3_pinhole).
The output (default <dataset>_realformat) is the same LeRobot dataset -- every original column, the videos and the replay
logs (hard links) -- plus the fields of the real recorder's HDF5 episodes (format_version 1.1) that the dataset does not
already have, one value per 50 fps frame, in the real files' conventions:

  joint_targets.kp / .kd                     PD gains of the frame's action
  states.robot.joint_velocity / joint_effort 29 joints, applied torque
  states.robot.root_pose                     pelvis position + quaternion xyzw (observation.base_pose is a different, wxyz estimate)
  states.robot.root_velocity                 world linear + body angular velocity of the pelvis
  states.robot.imu_*                         pelvis IMU at the stock imu_in_pelvis mount: quaternion xyzw, rpy (extrinsic xyz),
                                             gyroscope, accelerometer (with gravity)
  states.dex3.{left,right}.joint_velocity / joint_effort   real hand order (thumb_0-2, index_0-1, middle_0-1)
  reference_qpos                             [root_pos, root_quat wxyz, 29 dof_pos] of the current reference frame
  reference_actions                          11 blocks x 79 ([29 dof_pos, 29 dof_vel, root_pos, root_quat wxyz, 14 hand = 0]);
                                             block k = the current reference of frame t + k (last frame repeated at the end)
  holomotion_obs.*                           the three reference terms, computed as the real recorder does

Already in the dataset, so not duplicated (meta/realformat.json maps them): joint positions (observation.state), joint and hand
targets (action, policy.target_real), reference_frame_index (frame_index). Zero on the robot and not added: joint_vel_target,
joint_effort_target. Not produced: cameras/depth, tactile, neck, timestamps.

Physical values come from re-running each episode's bit-exact replay log (model hash checked, every frame compared with the
recording): frame i is the state before action i, exactly the state observation.state was recorded from.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

N_FUT = 10
JOINTS = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch",
          "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
          "left_wrist_pitch", "left_wrist_yaw",
          "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
          "right_wrist_pitch", "right_wrist_yaw"]
HAND = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]
PELVIS_IMU = np.array([0.04525, 0.0, -0.08339])          # stock G1 URDF imu_in_pelvis_joint (no rotation)
J = [n + "_joint" for n in JOINTS]
XYZ = ["x", "y", "z"]

FEATURES = {   # name -> (dim, names)
    "joint_targets.kp": (29, J), "joint_targets.kd": (29, J),
    "states.robot.joint_velocity": (29, J), "states.robot.joint_effort": (29, J),
    "states.robot.root_pose": (7, ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]),
    "states.robot.root_velocity": (6, ["lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]),
    "states.robot.imu_quaternion": (4, ["x", "y", "z", "w"]), "states.robot.imu_rpy": (3, ["roll", "pitch", "yaw"]),
    "states.robot.imu_gyroscope": (3, XYZ), "states.robot.imu_accelerometer": (3, XYZ),
    **{f"states.dex3.{s}.{q}": (7, [f"{s}_hand_{h}_joint" for h in HAND]) for s in ("left", "right")
       for q in ("joint_velocity", "joint_effort")},
    "reference_qpos": (36, ["root_pos_x", "root_pos_y", "root_pos_z", "root_quat_w", "root_quat_x", "root_quat_y",
                            "root_quat_z"] + J),
    "reference_actions": (869, None),
    "holomotion_obs.ref_future_root_ori_robot_frame_6d": (60, None),
    "holomotion_obs.ref_future_yaw_delta_sin_cos": (20, None),
    "holomotion_obs.ref_robot_yaw_error_sin_cos": (2, ["sin", "cos"]),
}
REALFORMAT = {
    "format_version": "1.1", "robot_name": "unitree_g1", "urdf_version": "29dof", "imu_frame": "pelvis_imu_link",
    "joint_names": JOINTS, "hand_joint_names": HAND, "n_fut_frames": N_FUT,
    "actions_layout": "[0:29] dof_pos, [29:58] dof_vel, [58:61] root_pos, [61:65] root_quat(wxyz), [65:79] hand_dofs(=0)",
    "reference_queue_slot": "block 0 = the frame's current reference; block k = the current reference of frame t + k",
    "sample_rate_hz": 50,
    "holomotion_obs": {
        "ref_future_yaw_delta_sin_cos": "sin/cos(yaw(block k) - yaw(block 0)), k = 1..10",
        "ref_robot_yaw_error_sin_cos": "sin/cos(yaw(block 0) - yaw(robot root quat)), no motion-entry alignment",
        "ref_future_root_ori_robot_frame_6d": "rotation of block k relative to block 0 (not the robot), first two matrix "
                                              "columns, column 0 then column 1 (the real recorder's layout)",
    },
    "already_in_dataset": {
        "states/robot/joint_position": "observation.state[0:29]",
        "states/dex3/left/joint_position": "observation.state[29:36] (thumb_0-2, index_0-1, middle_0-1)",
        "states/dex3/right/joint_position": "observation.state[36:43]",
        "joint_targets/joint_pos_target": "action[0:22] + action[29:36] (the 29 body joints), or policy.target_real",
        "states/dex3/left/cmd_position": "action[22:29], order index_0-1, middle_0-1, thumb_0-2",
        "states/dex3/right/cmd_position": "action[36:43], same order",
        "reference_frame_index": "frame_index",
    },
    "zero_on_robot_not_added": ["joint_targets/joint_vel_target", "joint_targets/joint_effort_target"],
    "not_produced": ["obs/* cameras and depth", "obs/tactile/*", "states/neck/*", "timestamps/*"],
}


# ------------------------------------------------------------------ quaternions (wxyz unless noted)
def std(q):
    return np.where(q[..., :1] < 0, -q, q)


def qmul(a, b):
    w0, x0, y0, z0 = np.moveaxis(a, -1, 0)
    w1, x1, y1, z1 = np.moveaxis(b, -1, 0)
    return np.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], -1)


def yaw(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot6d_colmajor(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    c0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    c1 = np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], -1)
    return np.concatenate([c0, c1], -1)


def rpy_xyz(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)), yaw(q)], -1)


def reference_columns(latest: np.ndarray, q_robot_wxyz: np.ndarray) -> dict[str, np.ndarray]:
    """latest (N, 65) = teleop.latest_obs per frame: [29 dof_pos, 29 dof_vel, root_pos, root_quat wxyz]."""
    n = len(latest)
    blocks = np.zeros((n, N_FUT + 1, 79))
    for k in range(N_FUT + 1):
        blocks[:, k, :65] = latest[np.minimum(np.arange(n) + k, n - 1)]
    q_cur, q_fut = std(blocks[:, 0, 61:65]), std(blocks[:, 1:, 61:65])
    yd = yaw(q_fut) - yaw(q_cur)[:, None]
    ye = yaw(q_cur) - yaw(std(q_robot_wxyz))
    rel = std(qmul((q_cur * np.array([1.0, -1, -1, -1]))[:, None, :], q_fut))
    return {
        "reference_qpos": np.concatenate([latest[:, 58:61], latest[:, 61:65], latest[:, 0:29]], 1),
        "reference_actions": blocks.reshape(n, -1),
        "holomotion_obs.ref_future_root_ori_robot_frame_6d": rot6d_colmajor(rel).reshape(n, -1),
        "holomotion_obs.ref_future_yaw_delta_sin_cos": np.stack([np.sin(yd), np.cos(yd)], -1).reshape(n, -1),
        "holomotion_obs.ref_robot_yaw_error_sin_cos": np.stack([np.sin(ye), np.cos(ye)], -1),
    }


def replay_columns(m, d, log) -> tuple[dict[str, np.ndarray], dict]:
    """Re-run the exact log; sample the state before every frame's action (state0, then after each frame)."""
    import mujoco
    from simple.teleop.holomotion_v14 import exact_log as XL

    qa = np.array([m.joint(j).qposadr[0] for j in J])
    va = np.array([m.joint(j).dofadr[0] for j in J])
    aa = np.array([m.actuator(j).id for j in J])
    hv = {s: np.array([m.joint(f"{s}_hand_{h}_joint").dofadr[0] for h in HAND]) for s in ("left", "right")}
    ha = {s: np.array([m.actuator(f"{s}_hand_{h}_joint").id for h in HAND]) for s in ("left", "right")}
    pelvis = m.body("pelvis").id
    imu_site, imu_adr = m.site("imu").id, m.sensor_adr[m.sensor("imu_acc").id]
    n = len(log["frame_calls"])
    rows = {k: [] for k in ("q", "dq", "tau", "hdq", "htau", "pos", "quat", "vlin", "w", "acc")}
    d2 = mujoco.MjData(m)
    chk = []

    def sample():
        XL.set_state(m, d2, XL.get_state(m, d, XL.INTEGRATION), XL.INTEGRATION)
        mujoco.mj_forward(m, d2)
        mujoco.mj_rnePostConstraint(m, d2)
        rows["q"].append(d2.qpos[qa].copy()); rows["dq"].append(d2.qvel[va].copy()); rows["tau"].append(d2.actuator_force[aa].copy())
        rows["hdq"].append(np.concatenate([d2.qvel[hv[s]] for s in ("left", "right")]))
        rows["htau"].append(np.concatenate([d2.actuator_force[ha[s]] for s in ("left", "right")]))
        rows["pos"].append(d2.xpos[pelvis].copy()); rows["quat"].append(d2.xquat[pelvis].copy())
        vw, vl, acc, a_site = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, d2, mujoco.mjtObj.mjOBJ_BODY, pelvis, vw, 0)
        mujoco.mj_objectVelocity(m, d2, mujoco.mjtObj.mjOBJ_BODY, pelvis, vl, 1)
        mujoco.mj_objectAcceleration(m, d2, mujoco.mjtObj.mjOBJ_BODY, pelvis, acc, 1)
        w = vl[:3]
        rows["vlin"].append(vw[3:].copy()); rows["w"].append(w.copy())
        rows["acc"].append(acc[3:] + np.cross(acc[:3], PELVIS_IMU) + np.cross(w, np.cross(w, PELVIS_IMU)))
        mujoco.mj_objectAcceleration(m, d2, mujoco.mjtObj.mjOBJ_SITE, imu_site, a_site, 1)   # method check on the torso IMU
        chk.append(np.abs(a_site[3:] - d2.sensordata[imu_adr:imu_adr + 3]).max())

    XL.set_state(m, d, log["state0"], XL.INTEGRATION)
    mujoco.mj_forward(m, d)
    rest_at = {int(i): v for i, v in zip(log["rest_idx"], log["rest_val"])}
    call, first_bad = 0, None
    for t, n_calls in enumerate(log["frame_calls"]):
        sample()                                         # the observation of frame t
        for _ in range(int(n_calls)):
            if call in rest_at:
                XL.set_state(m, d, rest_at[call], XL.REST)
            d.ctrl[:] = log["ctrl"][call]
            mujoco.mj_step(m, d, nstep=int(log["nstep"][call]))
            call += 1
        if first_bad is None and XL.get_state(m, d, XL.PHYSICS).tobytes() != log["physics"][t].tobytes():
            first_bad = t
    A = {k: np.asarray(v, np.float64) for k, v in rows.items()}
    xyzw = A["quat"][:, [1, 2, 3, 0]]
    cols = {
        "joint_targets.kp": log["action_kp"], "joint_targets.kd": log["action_kd"],
        "states.robot.joint_velocity": A["dq"], "states.robot.joint_effort": A["tau"],
        "states.robot.root_pose": np.concatenate([A["pos"], xyzw], 1),
        "states.robot.root_velocity": np.concatenate([A["vlin"], A["w"]], 1),
        "states.robot.imu_quaternion": xyzw, "states.robot.imu_rpy": rpy_xyz(A["quat"]),
        "states.robot.imu_gyroscope": A["w"], "states.robot.imu_accelerometer": A["acc"],
    }
    for i, s in enumerate(("left", "right")):
        cols[f"states.dex3.{s}.joint_velocity"] = A["hdq"][:, 7 * i:7 * i + 7]
        cols[f"states.dex3.{s}.joint_effort"] = A["htau"][:, 7 * i:7 * i + 7]
    return cols, dict(q=A["q"], quat=A["quat"], replay_bit_exact=first_bad is None, first_mismatch_frame=first_bad,
                      imu_method_check=float(max(chk)))


def stats(a: np.ndarray) -> dict:
    a = a.astype(np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
            "count": [int(len(a))]}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--episodes", default="", help="comma-separated indices (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="default <dataset>_realformat (replaced)")
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from simple.cli.replay_holomotion_v14 import _build_env, _reset_to
    from simple.teleop.holomotion_v14 import exact_log as XL

    src = a.dataset
    out = a.out or src.with_name(src.name + "_realformat")
    info = json.loads((src / "meta/info.json").read_text())
    episodes = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines()]
    ep_stats = {e["episode_index"]: e for e in map(json.loads, (src / "meta/episodes_stats.jsonl").read_text().splitlines())}
    wanted = {int(x) for x in a.episodes.split(",") if x} or {e["episode_index"] for e in episodes}
    episodes = [e for e in episodes if e["episode_index"] in wanted]
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    env = sonic_env = mod = first = None
    chunks = int(info.get("chunks_size", 1000))
    for e in episodes:
        ep = e["episode_index"]
        log = XL.load(src / e["exact_log"])
        linfo, setup = log["info"], log["setup"]
        if env is None:
            env, sonic_env, mod = _build_env(linfo, setup)
            first = linfo
        else:
            if linfo.get("scene") != first.get("scene") or linfo.get("backpack_kg") != first.get("backpack_kg"):
                sys.exit(f"episode {ep}: different scene/robot from the first episode; convert them separately")
            _reset_to(env, sonic_env, mod, linfo, setup)
        m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
        same_model = XL.model_sha256(m) == linfo["model_sha256"]
        cols, rep = replay_columns(m, d, log)

        rel = Path(info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep))
        table = pq.read_table(src / rel)
        latest = np.stack(table.column("teleop.latest_obs").to_numpy(zero_copy_only=False)).astype(np.float64)
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
        assert len(latest) == len(log["frame_calls"]), f"episode {ep}: parquet and replay lengths differ"
        state_diff = float(np.abs(rep["q"] - state[:, :29]).max())
        cols.update(reference_columns(latest, rep["quat"]))

        hf = json.loads(table.schema.metadata[b"huggingface"])
        live_diff = {}
        for name, (dim, _names) in FEATURES.items():
            arr = np.asarray(cols[name], np.float32).reshape(len(latest), dim)
            if name in table.column_names:                   # recorded live (teleop realformat): verify, keep the recorded column
                have = np.stack(table.column(name).to_numpy(zero_copy_only=False)).astype(np.float32).reshape(len(latest), dim)
                live_diff[name] = float(np.abs(have - arr).max())
                continue
            table = table.append_column(name, pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel(), pa.float32()), dim))
            hf["info"]["features"][name] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": dim, "_type": "Sequence"}
            ep_stats[ep]["stats"][name] = stats(arr)
        if live_diff:
            print(f"[realformat] episode {ep:06d}: {len(live_diff)} columns recorded live, kept; max |live - recomputed| = "
                  f"{max(live_diff.values()):.1e} ({max(live_diff, key=live_diff.get)})", flush=True)
        table = table.replace_schema_metadata({**table.schema.metadata, b"huggingface": json.dumps(hf).encode()})
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / rel)
        for vkey in [k for k, v in info["features"].items() if v.get("dtype") == "video"]:
            vrel = Path(info["video_path"].format(episode_chunk=ep // chunks, video_key=vkey, episode_index=ep))
            link_or_copy(src / vrel, out / vrel)
        link_or_copy(src / e["exact_log"], out / e["exact_log"])
        prev = e.get("realformat") or {}                  # the live recorder stores it as a JSON string
        prev = json.loads(prev) if isinstance(prev, str) else prev
        e["realformat"] = {**prev, "model_identical": bool(same_model), "replay_bit_exact": rep["replay_bit_exact"],
                           "live_columns_verified": {k: v for k, v in live_diff.items()} if live_diff else None,
                           "first_mismatch_frame": rep["first_mismatch_frame"],
                           "joint_position_vs_observation_state_max_abs": state_diff, "success": None,
                           "seed": setup.get("seed"), "model_sha256": linfo.get("model_sha256"),
                           "backpack_kg": linfo.get("backpack_kg"), "head_tilt_deg": linfo.get("head_tilt_deg")}
        print(f"[realformat] episode {ep:06d}: {len(latest)} frames, model {'identical' if same_model else 'DIFFERENT'}, "
              f"replay {'bit-exact' if rep['replay_bit_exact'] else 'differs from frame ' + str(rep['first_mismatch_frame'])}, "
              f"joints vs observation.state {state_diff:.1e}, IMU check {rep['imu_method_check']:.1e}", flush=True)

    for name, (dim, names) in FEATURES.items():
        info["features"][name] = {"dtype": "float32", "shape": [dim], "names": names}
    if len(episodes) != info["total_episodes"]:
        info["total_episodes"] = len(episodes)
        info["total_frames"] = int(sum(e["length"] for e in episodes))
        info["total_videos"] = len(episodes) * sum(v.get("dtype") == "video" for v in info["features"].values())
        info["splits"] = {"train": f"0:{len(episodes)}"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(ep_stats[e["episode_index"]]) + "\n" for e in episodes))
    for f in ("tasks.jsonl", "modality.json"):
        if (src / "meta" / f).exists():
            shutil.copy2(src / "meta" / f, out / "meta" / f)
    (out / "meta/realformat.json").write_text(json.dumps({**REALFORMAT, "source_dataset": str(src),
                                                          "added_features": list(FEATURES)}, indent=2))
    print(f"[realformat] {len(episodes)} episodes -> {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)                     # SIMPLE's env teardown can hang on exit


if __name__ == "__main__":
    main()
