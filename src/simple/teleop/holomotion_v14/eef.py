"""The decoupled-WBC end-effector columns for HoloMotion v1.4 recordings (2026-10-28).

The recordings keep the decoupled-WBC LeRobot schema, whose observation.eef_state, action.eef and teleop.base_height_command
were zero placeholders here until 2026-10-28. In the decoupled-WBC recordings (data/real_recordings/*) they hold:
  observation.eef_state  14 = left wrist pos (3) + quat w x y z (4), right wrist pos + quat: each *_wrist_yaw_link pose in
                         the pelvis frame (fitted on 2026-09-20-00-37-38-G1-sim: 0.0 mm / 0.0 deg from observation.state)
  action.eef             the same for the commanded wrists (the teleop IK targets)
  teleop.base_height_command  the commanded pelvis height (m)
HoloMotion has no wrist or height command of its own; the command it tracks is the motion reference, so action.eef is the
wrist poses of the reference frame the policy tracks this tick (reference_actions block 0, its dof_pos) and
base_height_command is that frame's root height. Before the 11-frame reference window has filled (block 0 all zero) both
take decoupled-WBC's own no-command defaults (decoupled_wbc/control/main/constants.py: DEFAULT_WRIST_POSE, DEFAULT_BASE_HEIGHT).
The arm chain is the same on every G1 variant here (stock, rot0, Revo 2): the hands hang below *_wrist_yaw_link.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

G1_BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
DEFAULT_WRIST_POSE = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2)       # x, y, z + w, x, y, z per wrist (decoupled_wbc)
DEFAULT_BASE_HEIGHT = 0.74
MJCF = Path(__file__).resolve().parents[4] / "data/robots/g1_rot0_teleop/g1_rot0_teleop.xml"


class WristFK:
    """Wrist poses in the pelvis frame from the 29 body joint angles (G1_BODY_JOINTS order), on a private MjData."""

    def __init__(self, mjcf: Path = MJCF):
        self.m = mujoco.MjModel.from_xml_path(str(mjcf)); self.d = mujoco.MjData(self.m)
        self.adr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in G1_BODY_JOINTS])
        self.pelvis = self.m.body("pelvis").id
        self.wrists = [self.m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]

    def poses(self, q29) -> np.ndarray:
        d = self.d; d.qpos[:] = self.m.qpos0; d.qpos[self.adr] = np.asarray(q29, dtype=np.float64)[:29]
        mujoco.mj_kinematics(self.m, d)
        R0 = d.xmat[self.pelvis].reshape(3, 3); p0 = d.xpos[self.pelvis]; out = np.zeros(14)
        for k, b in enumerate(self.wrists):
            q = np.zeros(4); mujoco.mju_mat2Quat(q, (R0.T @ d.xmat[b].reshape(3, 3)).reshape(-1))
            out[7 * k:7 * k + 3] = R0.T @ (d.xpos[b] - p0); out[7 * k + 3:7 * k + 7] = q if q[0] >= 0 else -q
        return out

    def command(self, ref_block0) -> tuple[np.ndarray, float]:
        """(action.eef, base_height_command) from the tracked reference frame: 65 = dof_pos 29, dof_vel 29, root_pos 3, root_rot_wxyz 4."""
        r = np.asarray(ref_block0, dtype=np.float64)
        if not np.any(r):
            return DEFAULT_WRIST_POSE.copy(), DEFAULT_BASE_HEIGHT
        return self.poses(r[:29]), float(r[60])


_FK: WristFK | None = None


def wrist_fk() -> WristFK:
    global _FK
    if _FK is None:
        _FK = WristFK()
    return _FK
