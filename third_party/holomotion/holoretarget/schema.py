"""Public HoloRetarget output schema."""

from __future__ import annotations


ROOT_POS_DIM = 3
ROOT_QUAT_DIM = 4
DOF_POS_DIM = 29
QPOS_DIM = ROOT_POS_DIM + ROOT_QUAT_DIM + DOF_POS_DIM

UNITREE_G1_29DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

if len(UNITREE_G1_29DOF_NAMES) != DOF_POS_DIM:
    raise RuntimeError(
        "HoloRetarget DoF schema must contain exactly 29 joints"
    )

# --- optional g1_comp extension: Dex3 hands + pan/tilt head servos ---------
# Canonical per-side Dex3 order: thumb_0, thumb_1, thumb_2, index_0, index_1,
# middle_0, middle_1. The 14-dim ``hand_joints`` payload field is left(7) then
# right(7) in this order.

DEX3_LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)

DEX3_RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

HAND_DOF_DIM = len(DEX3_LEFT_HAND_JOINT_NAMES) + len(DEX3_RIGHT_HAND_JOINT_NAMES)

# Fully-closed Dex3 pose reached at trigger value 1.0 (open pose is all zeros).
DEX3_LEFT_TRIGGER_CLOSE_POSE = (0.0, 0.6, 1.0, -0.9, -1.1, -0.9, -1.1)
DEX3_RIGHT_TRIGGER_CLOSE_POSE = (0.0, -0.6, -1.0, 0.9, 1.1, 0.9, 1.1)

# Two head servos: xl330_joint = pan (yaw), d455_joint = tilt (pitch). The
# 2-dim ``head_pan_tilt`` payload field is (pan, tilt) in radians.
HEAD_SERVO_JOINT_NAMES = ("xl330_joint", "d455_joint")
HEAD_DOF_DIM = len(HEAD_SERVO_JOINT_NAMES)
HEAD_PAN_LIMITS = (-0.8727, 0.8727)
HEAD_TILT_LIMITS = (-0.3491, 1.5708)

__all__ = [
    "DOF_POS_DIM",
    "QPOS_DIM",
    "ROOT_POS_DIM",
    "ROOT_QUAT_DIM",
    "UNITREE_G1_29DOF_NAMES",
    "DEX3_LEFT_HAND_JOINT_NAMES",
    "DEX3_RIGHT_HAND_JOINT_NAMES",
    "DEX3_LEFT_TRIGGER_CLOSE_POSE",
    "DEX3_RIGHT_TRIGGER_CLOSE_POSE",
    "HAND_DOF_DIM",
    "HEAD_SERVO_JOINT_NAMES",
    "HEAD_DOF_DIM",
    "HEAD_PAN_LIMITS",
    "HEAD_TILT_LIMITS",
]
