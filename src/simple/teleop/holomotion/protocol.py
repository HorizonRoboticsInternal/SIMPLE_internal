"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

HoloMotion v1.4 teleoperation contracts, vendored so SIMPLE runs standalone.

This module owns everything SIMPLE needs to speak to a HoloMotion reference
publisher without importing the HoloMotion package:

* the ZMQ wire format (fixed 1280-byte JSON header + packed binary fields),
* the ``reference_qpos`` layout and the HoloRetarget 29-dof joint order,
* the Dex3 hand contract (joint order, trigger close poses),
* the packed controller-button layout.

Keep these in sync with the HoloMotion repository:
``holoretarget/schema.py`` and ``deployment/holomotion_teleop/holomotion_teleop_node.py``.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np

# --- wire format ----------------------------------------------------------

HEADER_SIZE = 1280
DEFAULT_TOPIC = b"reference_qpos"

_DTYPE_BY_TAG = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}

_TAG_BY_KIND = {
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.uint8): "u8",
    np.dtype(bool): "bool",
}

# --- reference_qpos contract (holoretarget.schema) ------------------------

ROOT_POS_DIM = 3
ROOT_QUAT_DIM = 4
REF_DOF_DIM = 29
REF_QPOS_DIM = ROOT_POS_DIM + ROOT_QUAT_DIM + REF_DOF_DIM  # 36

#: HoloRetarget 29-dof order. Identical to the ``g1_sonic`` MJCF body-joint
#: order, so reference ``dof_pos`` and ``proprio["body_q"]`` share one layout.
HOLORETARGET_DOF_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
assert len(HOLORETARGET_DOF_NAMES) == REF_DOF_DIM

# --- actor observation dims (holomotion actor_observation.py) -------------

MOTION_ACTOR_CURRENT_DIM = 134
MOTION_ACTOR_FUTURE_DIM = 47


def motion_actor_observation_dim(n_fut_frames: int) -> int:
    return MOTION_ACTOR_CURRENT_DIM + int(n_fut_frames) * MOTION_ACTOR_FUTURE_DIM


# --- Dex3 hand contract ---------------------------------------------------

HAND_DOF_PER_SIDE = 7
HAND_DOF_DIM = 2 * HAND_DOF_PER_SIDE

#: Per-side order used by HoloMotion ("natural"): thumb_0..2, index_0/1, middle_0/1.
DEX3_LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
)
DEX3_RIGHT_HAND_JOINT_NAMES = tuple(n.replace("left_", "right_", 1) for n in DEX3_LEFT_HAND_JOINT_NAMES)

#: Fully-closed pose reached at trigger == 1.0 (open pose is all zeros).
DEX3_LEFT_TRIGGER_CLOSE_POSE = (0.0, 0.6, 1.0, -0.9, -1.1, -0.9, -1.1)
DEX3_RIGHT_TRIGGER_CLOSE_POSE = (0.0, -0.6, -1.0, 0.9, 1.1, 0.9, 1.1)

#: HoloMotion publishes thumb/index/middle; the g1_sonic MJCF orders the
#: fingers thumb/middle/index. Index with ``natural[DEX3_NATURAL_TO_MJCF]``.
DEX3_NATURAL_TO_MJCF = np.array([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)

# --- head servos ----------------------------------------------------------

HEAD_SERVO_JOINT_NAMES = ("xl330_joint", "d455_joint")
HEAD_DOF_DIM = 2

# --- controller buttons ---------------------------------------------------

#: Order of the packed ``buttons`` float32[8] payload field.
BUTTON_FIELDS = (
    "a_button", "b_button", "x_button", "y_button",
    "left_axis_click", "right_axis_click", "left_menu_button", "right_menu_button",
)

#: Scalar controller fields published alongside the packed buttons.
ANALOG_FIELDS = ("left_trigger", "right_trigger", "left_grip", "right_grip")


def pack_numpy_message(payload: dict, topic: bytes = DEFAULT_TOPIC, version: int = 1) -> bytes:
    """Serialize ``{name: ndarray}`` into the HoloMotion ZMQ packet format."""
    fields = []
    blobs = []
    for key, value in payload.items():
        if not isinstance(value, np.ndarray):
            continue
        tag = _TAG_BY_KIND.get(value.dtype)
        if tag is None:
            value = value.astype(np.float32)
            tag = "f32"
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        fields.append({"name": key, "dtype": tag, "shape": list(value.shape)})
        blobs.append(value.tobytes())

    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")
    ).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header)} > {HEADER_SIZE}")
    return topic + header.ljust(HEADER_SIZE, b"\x00") + b"".join(blobs)


def decode_numpy_message(packet: bytes, topic: bytes = DEFAULT_TOPIC) -> Dict[str, np.ndarray]:
    """Inverse of :func:`pack_numpy_message`."""
    if not packet.startswith(topic):
        raise ValueError("Unexpected ZMQ topic prefix")
    header_start = len(topic)
    header_end = header_start + HEADER_SIZE
    if len(packet) < header_end:
        raise ValueError("Packet too short for fixed-size header")
    header = json.loads(packet[header_start:header_end].rstrip(b"\x00").decode("utf-8"))
    payload = memoryview(packet[header_end:])
    result: Dict[str, np.ndarray] = {}
    offset = 0
    for field in header.get("fields", []):
        shape = tuple(field["shape"])
        dtype = _DTYPE_BY_TAG[field["dtype"]]
        size = int(np.prod(shape)) * dtype.itemsize
        result[field["name"]] = (
            np.frombuffer(payload[offset:offset + size], dtype=dtype).reshape(shape).copy()
        )
        offset += size
    return result


__all__ = [
    "HEADER_SIZE", "DEFAULT_TOPIC", "pack_numpy_message", "decode_numpy_message",
    "ROOT_POS_DIM", "ROOT_QUAT_DIM", "REF_DOF_DIM", "REF_QPOS_DIM",
    "HOLORETARGET_DOF_NAMES",
    "MOTION_ACTOR_CURRENT_DIM", "MOTION_ACTOR_FUTURE_DIM", "motion_actor_observation_dim",
    "HAND_DOF_PER_SIDE", "HAND_DOF_DIM",
    "DEX3_LEFT_HAND_JOINT_NAMES", "DEX3_RIGHT_HAND_JOINT_NAMES",
    "DEX3_LEFT_TRIGGER_CLOSE_POSE", "DEX3_RIGHT_TRIGGER_CLOSE_POSE", "DEX3_NATURAL_TO_MJCF",
    "HEAD_SERVO_JOINT_NAMES", "HEAD_DOF_DIM",
    "BUTTON_FIELDS", "ANALOG_FIELDS",
]
