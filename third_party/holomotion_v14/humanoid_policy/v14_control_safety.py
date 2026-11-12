"""Small fail-closed helpers for the HoloMotion v1.4 control path."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np


ARM_POSE_GATE_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


def sanitize_axis(value: float, *, limit: float = 1.0) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(-float(limit), min(float(limit), value))


class PicoAxisFilter:
    def __init__(
        self,
        *,
        smoothing: float = 0.1,
        deadzone: float = 0.05,
        limit: float = 1.0,
    ) -> None:
        if not 0.0 < float(smoothing) <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if float(deadzone) < 0.0:
            raise ValueError("deadzone must be >= 0")
        if float(limit) <= 0.0:
            raise ValueError("limit must be > 0")
        self.smoothing = float(smoothing)
        self.deadzone = float(deadzone)
        self.limit = float(limit)
        self._state = np.zeros(4, dtype=np.float32)

    def reset(self) -> None:
        self._state.fill(0.0)

    def update(self, left_axis, right_axis):
        raw = np.asarray(
            [
                left_axis[0] if len(left_axis) > 0 else 0.0,
                left_axis[1] if len(left_axis) > 1 else 0.0,
                right_axis[0] if len(right_axis) > 0 else 0.0,
                right_axis[1] if len(right_axis) > 1 else 0.0,
            ],
            dtype=np.float64,
        )
        sanitized = np.asarray(
            [sanitize_axis(value, limit=self.limit) for value in raw],
            dtype=np.float32,
        )
        sanitized[np.abs(sanitized) < self.deadzone] = 0.0
        self._state *= 1.0 - self.smoothing
        self._state += sanitized * self.smoothing
        self._state[np.abs(self._state) < 1.0e-4] = 0.0
        return (
            (float(self._state[0]), float(self._state[1])),
            (float(self._state[2]), float(self._state[3])),
        )


def finite_vector(values, *, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != int(expected_size):
        raise ValueError(
            f"{name} size mismatch: got {array.size}, expected {expected_size}"
        )
    if not np.isfinite(array).all():
        invalid = np.flatnonzero(~np.isfinite(array)).tolist()
        raise ValueError(f"{name} contains non-finite values at indices {invalid}")
    return array


def arm_pose_is_safe_for_walking(
    current_by_name,
    walking_default_by_name,
    *,
    max_error_rad: float = 0.45,
    joint_names=ARM_POSE_GATE_JOINTS,
) -> tuple[bool, float, str]:
    if float(max_error_rad) <= 0.0:
        raise ValueError("max_error_rad must be > 0")
    missing = [
        name
        for name in joint_names
        if name not in current_by_name or name not in walking_default_by_name
    ]
    if missing:
        return False, float("inf"), f"missing joints: {', '.join(missing)}"
    errors = np.asarray(
        [
            abs(float(current_by_name[name]) - float(walking_default_by_name[name]))
            for name in joint_names
        ],
        dtype=np.float64,
    )
    if not np.isfinite(errors).all():
        return False, float("inf"), "non-finite arm joint state"
    max_error = float(np.max(errors)) if errors.size else 0.0
    return max_error <= float(max_error_rad), max_error, ""


class ReferenceStreamWatchdog:
    def __init__(
        self,
        *,
        timeout_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if float(timeout_sec) <= 0.0:
            raise ValueError("timeout_sec must be > 0")
        self.timeout_sec = float(timeout_sec)
        self._clock = clock
        self._last_reference_time: float | None = None
        self._armed_since: float | None = None
        self._tripped = False

    def mark_reference(self, now: float | None = None) -> None:
        self._last_reference_time = self._clock() if now is None else float(now)

    def check(self, *, armed: bool, now: float | None = None) -> float | None:
        current_time = self._clock() if now is None else float(now)
        if not armed:
            self._armed_since = None
            self._tripped = False
            return None
        if self._armed_since is None:
            self._armed_since = current_time
        if self._tripped:
            return None
        baseline = self._armed_since
        if self._last_reference_time is not None:
            baseline = max(baseline, self._last_reference_time)
        silence = max(0.0, current_time - baseline)
        if silence <= self.timeout_sec:
            return None
        self._tripped = True
        return silence


__all__ = [
    "PicoAxisFilter",
    "ReferenceStreamWatchdog",
    "ARM_POSE_GATE_JOINTS",
    "arm_pose_is_safe_for_walking",
    "finite_vector",
    "sanitize_axis",
]
