"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

The real G1 recorder's extra fields (HDF5 format_version 1.1), recorded live by the v1.4 sim teleop.

scripts/holomotion_sim_realformat.py derives these after the fact by re-running an episode's bit-exact replay log.
Here the teleop computes the same values at recording time, with the same formulas, so a dataset has them from the
start (level-N_pinhole = the real format). Per 50 Hz frame, frame i = the state before action i (the state
observation.state is recorded from):

  joint_targets.kp / .kd                     PD gains of the frame's action (29, the robot's joint order = MJCF order)
  states.robot.joint_velocity / joint_effort 29 joints, applied actuator force
  states.robot.root_pose                     pelvis position + quaternion xyzw
  states.robot.root_velocity                 world linear + body angular velocity of the pelvis
  states.robot.imu_*                         pelvis IMU at the stock imu_in_pelvis mount: quaternion xyzw, rpy (extrinsic
                                             xyz), gyroscope (body angular velocity), accelerometer (with gravity)
  states.dex3.{left,right}.joint_velocity / joint_effort   real hand order (thumb_0-2, index_0-1, middle_0-1)
  reference_qpos                             [root_pos, root_quat wxyz, 29 dof_pos] of the frame's current reference
  reference_actions                          11 blocks x 79: block k = the current reference of frame t + k (the last
                                             frame repeated at the end), [29 dof_pos, 29 dof_vel, root_pos, root_quat,
                                             14 hand = 0]
  holomotion_obs.*                           the three reference terms, as the real recorder computes them

The physical values are sampled on a copy of the MuJoCo state (mj_forward + mj_rnePostConstraint on the copy), so the
live simulation is untouched and the replay log stays bit-exact. The reference columns need the next 10 frames, so
they are filled in when the episode is saved (``finish_episode``).
"""

from __future__ import annotations

from typing import Any

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

FEATURES: dict[str, tuple[int, list[str] | None]] = {   # name -> (dim, names)
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
REFERENCE_FEATURES = ("reference_qpos", "reference_actions", "holomotion_obs.ref_future_root_ori_robot_frame_6d",
                      "holomotion_obs.ref_future_yaw_delta_sin_cos", "holomotion_obs.ref_robot_yaw_error_sin_cos")
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
    "not_produced": ["obs/* cameras and depth (ego_view is the MuJoCo render; scripts/render_teleop_isaac.py re-renders it "
                     "in Isaac afterwards)", "obs/tactile/*", "states/neck/*", "timestamps/*"],
    "source": "recorded live by simple.cli.teleop_holomotion_v14 (the same values scripts/holomotion_sim_realformat.py "
              "derives from the replay log)",
}


def add_features(features: dict) -> None:
    for name, (dim, names) in FEATURES.items():
        features[name] = {"dtype": "float32", "shape": (dim,), "names": names}


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
    latest = np.asarray(latest, np.float64)
    n = len(latest)
    blocks = np.zeros((n, N_FUT + 1, 79))
    for k in range(N_FUT + 1):
        blocks[:, k, :65] = latest[np.minimum(np.arange(n) + k, n - 1)]
    q_cur, q_fut = std(blocks[:, 0, 61:65]), std(blocks[:, 1:, 61:65])
    yd = yaw(q_fut) - yaw(q_cur)[:, None]
    ye = yaw(q_cur) - yaw(std(np.asarray(q_robot_wxyz, np.float64)))
    rel = std(qmul((q_cur * np.array([1.0, -1, -1, -1]))[:, None, :], q_fut))
    return {
        "reference_qpos": np.concatenate([latest[:, 58:61], latest[:, 61:65], latest[:, 0:29]], 1),
        "reference_actions": blocks.reshape(n, -1),
        "holomotion_obs.ref_future_root_ori_robot_frame_6d": rot6d_colmajor(rel).reshape(n, -1),
        "holomotion_obs.ref_future_yaw_delta_sin_cos": np.stack([np.sin(yd), np.cos(yd)], -1).reshape(n, -1),
        "holomotion_obs.ref_robot_yaw_error_sin_cos": np.stack([np.sin(ye), np.cos(ye)], -1),
    }


# ------------------------------------------------------------------ live sampling
class LiveSampler:
    """The physical columns of the current (pre-step) MuJoCo state, on a scratch copy of the data."""

    def __init__(self, sonic_env) -> None:
        self.env = sonic_env
        self._model = None
        self.quats: list[np.ndarray] = []        # float64 pelvis quaternion (wxyz) of every RECORDED frame, in order
        self.last_quat: np.ndarray | None = None

    def _bind(self, m):
        import mujoco
        self.m, self._model = m, m
        self.d2 = mujoco.MjData(m)
        self.qa = np.array([m.joint(j).qposadr[0] for j in J])
        self.va = np.array([m.joint(j).dofadr[0] for j in J])
        self.aa = np.array([m.actuator(j).id for j in J])
        self.hv = np.concatenate([[m.joint(f"{s}_hand_{h}_joint").dofadr[0] for h in HAND] for s in ("left", "right")])
        self.ha = np.concatenate([[m.actuator(f"{s}_hand_{h}_joint").id for h in HAND] for s in ("left", "right")])
        self.pelvis = m.body("pelvis").id

    def sample(self, action) -> dict[str, np.ndarray]:
        """Call right before env.step (the state observation.state was taken from); action = the frame's ActionCmd."""
        import mujoco
        from simple.teleop.holomotion_v14 import exact_log as XL
        m, d = self.env.mujoco.mjModel, self.env.mujoco.mjData
        if self._model is not m:                               # a new scene compiled a new model
            self._bind(m)
        d2 = self.d2
        XL.set_state(m, d2, XL.get_state(m, d, XL.INTEGRATION), XL.INTEGRATION)
        mujoco.mj_forward(m, d2)
        mujoco.mj_rnePostConstraint(m, d2)
        vw, vl, acc = np.zeros(6), np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, d2, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, vw, 0)
        mujoco.mj_objectVelocity(m, d2, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, vl, 1)
        mujoco.mj_objectAcceleration(m, d2, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, acc, 1)
        w = vl[:3].copy()
        quat = d2.xquat[self.pelvis].copy()                    # wxyz
        xyzw = quat[[1, 2, 3, 0]]
        hdq, htau = d2.qvel[self.hv].copy(), d2.actuator_force[self.ha].copy()
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        out = {
            "joint_targets.kp": f32(action["kp"]), "joint_targets.kd": f32(action["kd"]),
            "states.robot.joint_velocity": f32(d2.qvel[self.va]), "states.robot.joint_effort": f32(d2.actuator_force[self.aa]),
            "states.robot.root_pose": f32(np.concatenate([d2.xpos[self.pelvis], xyzw])),
            "states.robot.root_velocity": f32(np.concatenate([vw[3:], w])),
            "states.robot.imu_quaternion": f32(xyzw), "states.robot.imu_rpy": f32(rpy_xyz(quat)),
            "states.robot.imu_gyroscope": f32(w),
            "states.robot.imu_accelerometer": f32(acc[3:] + np.cross(acc[:3], PELVIS_IMU) + np.cross(w, np.cross(w, PELVIS_IMU))),
            "states.dex3.left.joint_velocity": f32(hdq[:7]), "states.dex3.left.joint_effort": f32(htau[:7]),
            "states.dex3.right.joint_velocity": f32(hdq[7:]), "states.dex3.right.joint_effort": f32(htau[7:]),
        }
        for name in REFERENCE_FEATURES:                        # filled in at save time (they need the next 10 frames)
            out[name] = np.zeros(FEATURES[name][0], dtype=np.float32)
        self.last_quat = quat                                  # float64, for the yaw error at save time
        return out

    def frame_recorded(self) -> None:
        """Call after a frame built from the last sample() was added to the episode buffer."""
        self.quats.append(self.last_quat)

    def discard(self) -> None:
        self.quats = []


def finish_episode(episode_buffer: dict[str, Any], sampler: "LiveSampler | None" = None) -> None:
    """Fill the reference columns of a complete episode buffer (lists per feature, one entry per frame). The robot's
    root quaternion comes from the sampler's float64 record when it is aligned with the buffer (the converter's
    precision), else from the float32 root_pose column."""
    latest = np.stack([np.asarray(x) for x in episode_buffer["teleop.latest_obs"]])
    if sampler is not None and len(sampler.quats) == len(latest):
        q_wxyz = np.stack(sampler.quats)
    else:
        if sampler is not None:
            print(f"[Record] real-format: {len(sampler.quats)} float64 root quaternions for {len(latest)} frames; using root_pose")
        xyzw = np.stack([np.asarray(x)[3:7] for x in episode_buffer["states.robot.root_pose"]])
        q_wxyz = xyzw[:, [3, 0, 1, 2]].astype(np.float64)
    cols = reference_columns(latest, q_wxyz)
    for name in REFERENCE_FEATURES:
        arr = np.asarray(cols[name], dtype=np.float32).reshape(len(latest), FEATURES[name][0])
        episode_buffer[name] = [row for row in arr]
    if sampler is not None:
        sampler.discard()


def episode_meta(run_meta: dict, setup: dict, model_sha256: str) -> dict:
    return {"live": True, "model_sha256": model_sha256, "seed": setup.get("seed"), "backpack_kg": run_meta.get("backpack_kg"),
            "head_tilt_deg": run_meta.get("head_tilt_deg"), "success": None}
