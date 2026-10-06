"""Compact Pico controller state shared by v1.4.1 teleoperation consumers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from typing import Iterable


BUTTON_L3 = 1 << 0
BUTTON_X = 1 << 1
BUTTON_A = 1 << 2
BUTTON_R3 = 1 << 3
BUTTON_B = 1 << 4
BUTTON_Y = 1 << 5
BUTTON_LEFT_MENU = 1 << 6
BUTTON_RIGHT_MENU = 1 << 7
BUTTON_MASK = (
    BUTTON_L3
    | BUTTON_X
    | BUTTON_A
    | BUTTON_R3
    | BUTTON_B
    | BUTTON_Y
    | BUTTON_LEFT_MENU
    | BUTTON_RIGHT_MENU
)

JOY_AXIS_COUNT = 8
JOY_BUTTON_COUNT = 8


class PicoControlNotReady(RuntimeError):
    """Controller transport is present but has not produced a valid frame."""


def _finite_float(value: object, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _clamp(value: object, *, name: str, lower: float, upper: float) -> float:
    parsed = _finite_float(value, name=name)
    return min(max(parsed, lower), upper)


def _axis_pair(values: Iterable[object], *, name: str) -> tuple[float, float]:
    items = list(values)
    if len(items) < 2:
        raise ValueError(f"{name} must contain at least two values")
    return (
        _clamp(items[0], name=f"{name}[0]", lower=-1.0, upper=1.0),
        _clamp(items[1], name=f"{name}[1]", lower=-1.0, upper=1.0),
    )


@dataclass(frozen=True)
class PicoControlSample:
    timestamp_ns: int
    received_monotonic: float
    button_bits: int
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]
    left_trigger: float
    right_trigger: float
    left_grip: float
    right_grip: float
    emit_grip_axes: bool = True

    def joy_axes(self) -> list[float]:
        left_grip = self.left_grip if self.emit_grip_axes else 0.0
        right_grip = self.right_grip if self.emit_grip_axes else 0.0
        return [
            self.left_axis[0],
            self.left_axis[1],
            self.right_axis[0],
            self.right_axis[1],
            self.left_trigger,
            self.right_trigger,
            left_grip,
            right_grip,
        ]

    def joy_buttons(self) -> list[int]:
        return [int(bool(self.button_bits & (1 << index))) for index in range(JOY_BUTTON_COUNT)]

    def is_neutral(self, *, deadzone: float = 0.05) -> bool:
        threshold = _clamp(deadzone, name="deadzone", lower=0.0, upper=1.0)
        analog_values = (
            *self.left_axis,
            *self.right_axis,
            self.left_trigger,
            self.right_trigger,
            self.left_grip,
            self.right_grip,
        )
        return self.button_bits == 0 and all(
            abs(value) <= threshold for value in analog_values
        )


def read_pico_control_sample(
    sdk,
    *,
    received_monotonic: float | None = None,
    emit_grip_axes: bool | None = None,
) -> PicoControlSample:
    timestamp_ns = int(sdk.get_time_stamp_ns())
    if timestamp_ns == 0:
        raise PicoControlNotReady("Pico controller frame is not ready")
    if timestamp_ns < 0:
        raise ValueError("Pico controller timestamp must be positive")

    left_axis = _axis_pair(sdk.get_left_axis(), name="left_axis")
    right_axis = _axis_pair(sdk.get_right_axis(), name="right_axis")

    button_bits = 0
    button_bits |= int(bool(sdk.get_left_axis_click())) * BUTTON_L3
    button_bits |= int(bool(sdk.get_X_button())) * BUTTON_X
    button_bits |= int(bool(sdk.get_A_button())) * BUTTON_A
    button_bits |= int(bool(sdk.get_right_axis_click())) * BUTTON_R3
    button_bits |= int(bool(sdk.get_B_button())) * BUTTON_B
    button_bits |= int(bool(sdk.get_Y_button())) * BUTTON_Y
    button_bits |= int(bool(sdk.get_left_menu_button())) * BUTTON_LEFT_MENU
    button_bits |= int(bool(sdk.get_right_menu_button())) * BUTTON_RIGHT_MENU

    left_grip = _clamp(
        sdk.get_left_grip(),
        name="left_grip",
        lower=0.0,
        upper=1.0,
    )
    right_grip = _clamp(
        sdk.get_right_grip(),
        name="right_grip",
        lower=0.0,
        upper=1.0,
    )
    if emit_grip_axes is None:
        emit_grip_axes = (
            os.environ.get("HAND_CONTROL_MODE", "grip").strip().lower()
            != "hand"
        )

    return PicoControlSample(
        timestamp_ns=timestamp_ns,
        received_monotonic=(
            time.monotonic()
            if received_monotonic is None
            else _finite_float(received_monotonic, name="received_monotonic")
        ),
        button_bits=button_bits & BUTTON_MASK,
        left_axis=left_axis,
        right_axis=right_axis,
        left_trigger=_clamp(
            sdk.get_left_trigger(),
            name="left_trigger",
            lower=0.0,
            upper=1.0,
        ),
        right_trigger=_clamp(
            sdk.get_right_trigger(),
            name="right_trigger",
            lower=0.0,
            upper=1.0,
        ),
        left_grip=left_grip,
        right_grip=right_grip,
        emit_grip_axes=bool(emit_grip_axes),
    )
