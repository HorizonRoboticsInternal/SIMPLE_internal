"""Pico grip to Unitree Dex3 hand command targets."""

from __future__ import annotations

from dataclasses import dataclass


DEX3_DOF = 7

# Unitree Dex3-1 URDF limits from the official G1 Dex3 example.
LEFT_MIN_Q = (-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75)
LEFT_MAX_Q = (1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0)
RIGHT_MIN_Q = (-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0)
RIGHT_MAX_Q = (1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75)

OPEN_Q = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
STOP_Q = OPEN_Q

# SONIC's Pico teleop close preset.
# This closes more than Unitree's official midpoint grip while staying inside URDF limits.
SONIC_LEFT_MIDDLE_CLOSE_Q = (0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5)
SONIC_RIGHT_MIDDLE_CLOSE_Q = (0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5)


def _midpoint(min_q: tuple[float, ...], max_q: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((lo + hi) / 2.0 for lo, hi in zip(min_q, max_q))


# Unitree's official gripHand() example is kept for diagnostics/comparison.
UNITREE_LEFT_MID_Q = _midpoint(LEFT_MIN_Q, LEFT_MAX_Q)
UNITREE_RIGHT_MID_Q = _midpoint(RIGHT_MIN_Q, RIGHT_MAX_Q)

LEFT_CLOSE_Q = SONIC_LEFT_MIDDLE_CLOSE_Q
RIGHT_CLOSE_Q = SONIC_RIGHT_MIDDLE_CLOSE_Q


@dataclass(frozen=True)
class Dex3CommandConfig:
    grip_threshold: float = 0.5
    mode: int = 1
    timeout: int = 0
    kp: float = 1.5
    kd: float = 0.1


def dex3_motor_mode(motor_id: int, status: int = 1, timeout: int = 0) -> int:
    """Encode Unitree Dex3 RIS mode bits for one motor."""

    return (int(motor_id) & 0x0F) | ((int(status) & 0x07) << 4) | ((int(timeout) & 0x01) << 7)


def is_grip_closed(status: int, grip: float, threshold: float = 0.5) -> bool:
    return int(status) != 0 and float(grip) > float(threshold)


def target_q_for_grip(
    *,
    side: str,
    status: int,
    grip: float,
    threshold: float = 0.5,
) -> tuple[float, ...]:
    closed = is_grip_closed(status, grip, threshold)
    if not closed:
        return OPEN_Q
    if side == "left":
        return LEFT_CLOSE_Q
    if side == "right":
        return RIGHT_CLOSE_Q
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")
