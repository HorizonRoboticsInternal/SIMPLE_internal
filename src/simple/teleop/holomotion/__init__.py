"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

HoloMotion teleoperation support, self-contained inside SIMPLE.

* :mod:`simple.teleop.holomotion.protocol` - ZMQ wire format and the
  reference / Dex3 / button contracts shared with HoloMotion v1.4.
* :func:`resolve_motion_onnx` / :func:`resolve_reference_sample` - locate the
  artifacts vendored under ``data/holomotion``.

The reference publisher itself (PICO -> HoloRetarget -> ZMQ) is vendored under
``third_party/holomotion``; see that directory's README for its environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from .protocol import (  # noqa: F401
    ANALOG_FIELDS,
    BUTTON_FIELDS,
    DEFAULT_TOPIC,
    DEX3_LEFT_HAND_JOINT_NAMES,
    DEX3_LEFT_TRIGGER_CLOSE_POSE,
    DEX3_NATURAL_TO_MJCF,
    DEX3_RIGHT_HAND_JOINT_NAMES,
    DEX3_RIGHT_TRIGGER_CLOSE_POSE,
    HAND_DOF_DIM,
    HAND_DOF_PER_SIDE,
    HEAD_DOF_DIM,
    HEAD_SERVO_JOINT_NAMES,
    HEADER_SIZE,
    HOLORETARGET_DOF_NAMES,
    MOTION_ACTOR_CURRENT_DIM,
    MOTION_ACTOR_FUTURE_DIM,
    REF_DOF_DIM,
    REF_QPOS_DIM,
    decode_numpy_message,
    motion_actor_observation_dim,
    pack_numpy_message,
)

#: Repository root (…/SIMPLE), derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data" / "holomotion"
MODEL_DIR = DATA_ROOT / "models"
SAMPLE_DIR = DATA_ROOT / "samples"
PUBLISHER_ROOT = REPO_ROOT / "third_party" / "holomotion"

DEFAULT_REFERENCE_URI = os.environ.get("HOLOMOTION_REFERENCE_URI", "tcp://127.0.0.1:6001")


def resolve_motion_onnx(path: str | os.PathLike | None = None) -> str:
    """Locate the HoloMotion motion-tracking ONNX.

    Resolution order: explicit ``path`` -> ``$HOLOMOTION_MOTION_ONNX`` ->
    the newest ``*.onnx`` vendored under ``data/holomotion/models``.
    """
    candidate = str(path or os.environ.get("HOLOMOTION_MOTION_ONNX", "")).strip()
    if candidate:
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"HoloMotion ONNX not found: {candidate}")
        return candidate

    models = sorted(MODEL_DIR.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if models:
        return str(models[0])
    raise FileNotFoundError(
        "No HoloMotion motion-tracking ONNX found. Pass --onnx-path, set "
        f"HOLOMOTION_MOTION_ONNX, or place a model in {MODEL_DIR}."
    )


def resolve_reference_sample(path: str | os.PathLike | None = None) -> str:
    """Locate a recorded reference stream (``.npz``) for headset-free replay."""
    candidate = str(path or "").strip()
    if candidate:
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Reference recording not found: {candidate}")
        return candidate

    samples = sorted(SAMPLE_DIR.glob("*.npz"))
    if samples:
        return str(samples[0])
    raise FileNotFoundError(
        f"No reference recording found. Pass a path explicitly or place one in {SAMPLE_DIR}."
    )


__all__ = [
    "REPO_ROOT", "DATA_ROOT", "MODEL_DIR", "SAMPLE_DIR", "PUBLISHER_ROOT",
    "DEFAULT_REFERENCE_URI", "resolve_motion_onnx", "resolve_reference_sample",
    "pack_numpy_message", "decode_numpy_message", "motion_actor_observation_dim",
    "HEADER_SIZE", "DEFAULT_TOPIC", "REF_QPOS_DIM", "REF_DOF_DIM",
    "HOLORETARGET_DOF_NAMES", "MOTION_ACTOR_CURRENT_DIM", "MOTION_ACTOR_FUTURE_DIM",
    "HAND_DOF_DIM", "HAND_DOF_PER_SIDE", "DEX3_NATURAL_TO_MJCF",
    "DEX3_LEFT_HAND_JOINT_NAMES", "DEX3_RIGHT_HAND_JOINT_NAMES",
    "DEX3_LEFT_TRIGGER_CLOSE_POSE", "DEX3_RIGHT_TRIGGER_CLOSE_POSE",
    "HEAD_SERVO_JOINT_NAMES", "HEAD_DOF_DIM", "BUTTON_FIELDS", "ANALOG_FIELDS",
]
