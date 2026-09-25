#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pico/XRoboToolkit teleoperation service backed by HoloRetarget.

This deployment entrypoint owns only runtime I/O:

    XRoboToolkit body_poses[24, 7]
        -> holoretarget.HoloRetargeter
        -> qpos36 = [root_pos(3), root_rot_wxyz(4), dof_pos(29)]
        -> ZMQ PUB

The retarget algorithm itself lives under the repository-level ``holoretarget``
package so online deployment and future data production use the same code path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Optional

import numpy as np
import zmq


FILE_DIR = Path(__file__).resolve().parent
HOLOMOTION_ROOT = FILE_DIR.parents[1]
if str(HOLOMOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(HOLOMOTION_ROOT))

try:
    import xrobotoolkit_sdk as xrt
except ImportError:  # pragma: no cover - depends on deployment host
    xrt = None

from holoretarget import (  # noqa: E402
    HoloRetargetConfig,
    HoloRetargeter,
    QPOS_DIM,
    default_asset_root,
)
from holoretarget.schema import (  # noqa: E402
    DEX3_LEFT_TRIGGER_CLOSE_POSE,
    DEX3_RIGHT_TRIGGER_CLOSE_POSE,
    HAND_DOF_DIM,
    HEAD_DOF_DIM,
    HEAD_PAN_LIMITS,
    HEAD_TILT_LIMITS,
)


HEADER_SIZE = 1280
OUT_TOPIC = b"reference_qpos"
DEFAULT_HZ = 50.0
SOURCE_TIMEOUT_SEC = 0.1

SMPL_PELVIS_IDX = 0
SMPL_HEAD_IDX = 15
IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)
# Discrete controller buttons are packed into one ``buttons`` f32[8] payload
# field in this order (keeps the fixed-size packet header under HEADER_SIZE).
BUTTON_FIELDS = (
    "a_button",
    "b_button",
    "x_button",
    "y_button",
    "left_axis_click",
    "right_axis_click",
    "left_menu_button",
    "right_menu_button",
)
DETAIL_TIMING_KEYS = (
    "holoretarget.pico_to_smpl",
    "holoretarget.newton_targets",
    "holoretarget.newton_set_targets",
    "holoretarget.newton_root_seed",
    "holoretarget.solve",
    "holoretarget.output",
    "holoretarget.newton_project",
    "holoretarget.total",
)


def pack_numpy_message(payload: dict, topic: bytes = OUT_TOPIC, version: int = 1) -> bytes:
    fields = []
    binary_data = []

    for key, value in payload.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == np.uint8:
            dtype_str = "u8"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)

        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))

        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})
        binary_data.append(value.tobytes())

    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_bytes)} > {HEADER_SIZE}")
    return topic + header_bytes.ljust(HEADER_SIZE, b"\x00") + b"".join(binary_data)


def _safe_xrt_scalar(name: str, default: float = 0.0) -> float:
    """Read an optional scalar controller getter; 0.0 when the SDK lacks it."""
    getter = getattr(xrt, name, None)
    if getter is None:
        return default
    try:
        return float(getter())
    except Exception:
        return default


def _safe_xrt_axis(name: str) -> np.ndarray:
    """Read an optional thumbstick getter as float32[2] (x, y)."""
    getter = getattr(xrt, name, None)
    if getter is None:
        return np.zeros(2, dtype=np.float32)
    try:
        axis = np.asarray(getter(), dtype=np.float32).reshape(-1)
        out = np.zeros(2, dtype=np.float32)
        out[: min(2, axis.size)] = axis[:2]
        return out
    except Exception:
        return np.zeros(2, dtype=np.float32)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_to_rotvec_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    q = q / norm
    if q[3] < 0.0:
        q = -q
    w = min(1.0, float(q[3]))
    angle = 2.0 * float(np.arccos(w))
    sin_half = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if sin_half < 1e-8:
        return q[:3] * 2.0
    return q[:3] / sin_half * angle


def head_quat_relative_to_pelvis(body_poses: np.ndarray) -> np.ndarray:
    """Operator head orientation relative to the pelvis, as an xyzw quaternion.

    body_poses is (24, 7): [x, y, z, qx, qy, qz, qw] per SMPL-24 joint.
    Returns identity on malformed input.
    """
    try:
        bp = np.asarray(body_poses, dtype=np.float64)
        if bp.shape != (24, 7):
            return np.array(IDENTITY_QUAT_XYZW, dtype=np.float32)
        head = bp[SMPL_HEAD_IDX, 3:7]
        pelvis = bp[SMPL_PELVIS_IDX, 3:7]
        rel = _quat_mul_xyzw(_quat_conj_xyzw(pelvis), head)
        norm = float(np.linalg.norm(rel))
        if norm < 1e-12:
            return np.array(IDENTITY_QUAT_XYZW, dtype=np.float32)
        return (rel / norm).astype(np.float32)
    except Exception:
        return np.array(IDENTITY_QUAT_XYZW, dtype=np.float32)


class HandHeadController:
    """Map Pico controller/headset input to Dex3 hand joints and head servos.

    Hands: ``trigger`` linearly interpolates each hand from open (all zeros)
    to the Dex3 close pose using the controller trigger; ``joystick`` uses the
    thumbstick y instead (push up to close). Head: ``headset`` follows the
    operator's head orientation relative to a neutral pose calibrated on the
    first frame; ``joystick`` integrates the right thumbstick as pan/tilt
    rates. Outputs are clipped to the servo limits.
    """

    def __init__(
        self,
        *,
        hand_source: str = "trigger",
        head_source: str = "headset",
        head_follow_gain: float = 1.0,
        head_follow_yaw_axis: int = 1,
        head_follow_yaw_sign: float = 1.0,
        head_follow_pitch_axis: int = 0,
        head_follow_pitch_sign: float = -1.0,
        head_joystick_rate: float = 1.5,
        hz: float = DEFAULT_HZ,
    ) -> None:
        self.hand_source = hand_source
        self.head_source = head_source
        self.head_follow_gain = float(head_follow_gain)
        self.head_follow_yaw_axis = int(head_follow_yaw_axis)
        self.head_follow_yaw_sign = float(head_follow_yaw_sign)
        self.head_follow_pitch_axis = int(head_follow_pitch_axis)
        self.head_follow_pitch_sign = float(head_follow_pitch_sign)
        self.head_joystick_rate = float(head_joystick_rate)
        self._dt = 1.0 / float(hz) if hz > 0 else 0.02
        self._left_close = np.asarray(DEX3_LEFT_TRIGGER_CLOSE_POSE, dtype=np.float32)
        self._right_close = np.asarray(DEX3_RIGHT_TRIGGER_CLOSE_POSE, dtype=np.float32)
        self._head_base_quat: Optional[np.ndarray] = None
        self._joystick_pan = 0.0
        self._joystick_tilt = 0.0

    def reset_head_calibration(self) -> None:
        self._head_base_quat = None
        self._joystick_pan = 0.0
        self._joystick_tilt = 0.0

    def _hand_close_alphas(self, sample: dict) -> tuple[float, float]:
        if self.hand_source == "joystick":
            left_axis = np.asarray(sample.get("left_axis", (0.0, 0.0)), dtype=np.float32).reshape(-1)
            right_axis = np.asarray(sample.get("right_axis", (0.0, 0.0)), dtype=np.float32).reshape(-1)
            left = float(left_axis[1]) if left_axis.size > 1 else 0.0
            right = float(right_axis[1]) if right_axis.size > 1 else 0.0
        else:
            left = float(sample.get("left_trigger", 0.0))
            right = float(sample.get("right_trigger", 0.0))
        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _head_from_headset(self, sample: dict) -> tuple[float, float]:
        head_quat = np.asarray(
            sample.get("head_quat", IDENTITY_QUAT_XYZW), dtype=np.float64
        ).reshape(-1)
        if head_quat.size != 4:
            return 0.0, 0.0
        if self._head_base_quat is None:
            self._head_base_quat = head_quat.copy()
            return 0.0, 0.0
        delta = _quat_mul_xyzw(_quat_conj_xyzw(self._head_base_quat), head_quat)
        rotvec = _quat_to_rotvec_xyzw(delta)
        pan = (
            self.head_follow_gain
            * self.head_follow_yaw_sign
            * float(rotvec[self.head_follow_yaw_axis])
        )
        tilt = (
            self.head_follow_gain
            * self.head_follow_pitch_sign
            * float(rotvec[self.head_follow_pitch_axis])
        )
        return pan, tilt

    def _head_from_joystick(self, sample: dict) -> tuple[float, float]:
        right_axis = np.asarray(sample.get("right_axis", (0.0, 0.0)), dtype=np.float32).reshape(-1)
        axis_x = float(right_axis[0]) if right_axis.size > 0 else 0.0
        axis_y = float(right_axis[1]) if right_axis.size > 1 else 0.0
        # stick right pans right (negative xl330 yaw), stick up looks up
        # (negative d455 pitch); integrate as rates
        self._joystick_pan -= axis_x * self.head_joystick_rate * self._dt
        self._joystick_tilt -= axis_y * self.head_joystick_rate * self._dt
        self._joystick_pan = float(np.clip(self._joystick_pan, *HEAD_PAN_LIMITS))
        self._joystick_tilt = float(np.clip(self._joystick_tilt, *HEAD_TILT_LIMITS))
        return self._joystick_pan, self._joystick_tilt

    def process(self, sample: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
        sample = sample or {}
        hand_joints = np.zeros(HAND_DOF_DIM, dtype=np.float32)
        head_pan_tilt = np.zeros(HEAD_DOF_DIM, dtype=np.float32)

        if self.hand_source != "off":
            left_alpha, right_alpha = self._hand_close_alphas(sample)
            hand_joints[:7] = left_alpha * self._left_close
            hand_joints[7:] = right_alpha * self._right_close

        if self.head_source == "headset":
            pan, tilt = self._head_from_headset(sample)
        elif self.head_source == "joystick":
            pan, tilt = self._head_from_joystick(sample)
        else:
            pan, tilt = 0.0, 0.0
        head_pan_tilt[0] = np.clip(pan, *HEAD_PAN_LIMITS)
        head_pan_tilt[1] = np.clip(tilt, *HEAD_TILT_LIMITS)
        return hand_joints, head_pan_tilt


class PicoReader:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_stamp_ns: Optional[int] = None
        self._fps_ema = 0.0
        self._latest: Optional[dict] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        last_report = time.time()
        while not self._stop.is_set():
            if not xrt.is_body_data_available():
                time.sleep(0.001)
                continue

            stamp_ns = int(xrt.get_time_stamp_ns())
            prev_stamp_ns = self._last_stamp_ns
            if prev_stamp_ns is not None and stamp_ns == prev_stamp_ns:
                time.sleep(0.000001)
                continue

            device_dt = ((stamp_ns - prev_stamp_ns) * 1e-9) if prev_stamp_ns is not None else 0.0
            if device_dt > 0.0:
                inst_fps = 1.0 / device_dt
                self._fps_ema = inst_fps if self._fps_ema == 0.0 else (0.9 * self._fps_ema + 0.1 * inst_fps)
            self._last_stamp_ns = stamp_ns

            try:
                body_poses = np.asarray(xrt.get_body_joints_pose(), dtype=np.float32)
                if body_poses.shape != (24, 7):
                    print(f"[PicoReader] WARNING: unexpected body_poses shape: {body_poses.shape}")
                    continue

                sample = {
                    "body_poses_np": body_poses,
                    "head_quat": head_quat_relative_to_pelvis(body_poses),
                    "left_trigger": _safe_xrt_scalar("get_left_trigger"),
                    "right_trigger": _safe_xrt_scalar("get_right_trigger"),
                    "left_grip": _safe_xrt_scalar("get_left_grip"),
                    "right_grip": _safe_xrt_scalar("get_right_grip"),
                    "left_axis": _safe_xrt_axis("get_left_axis"),
                    "right_axis": _safe_xrt_axis("get_right_axis"),
                    "a_button": _safe_xrt_scalar("get_A_button"),
                    "b_button": _safe_xrt_scalar("get_B_button"),
                    "x_button": _safe_xrt_scalar("get_X_button"),
                    "y_button": _safe_xrt_scalar("get_Y_button"),
                    "left_axis_click": _safe_xrt_scalar("get_left_axis_click"),
                    "right_axis_click": _safe_xrt_scalar("get_right_axis_click"),
                    "left_menu_button": _safe_xrt_scalar("get_left_menu_button"),
                    "right_menu_button": _safe_xrt_scalar("get_right_menu_button"),
                    "timestamp_realtime": time.time(),
                    "timestamp_monotonic": time.monotonic(),
                    "timestamp_ns": stamp_ns,
                    "dt": float(device_dt),
                    "fps": float(self._fps_ema),
                }
                with self._lock:
                    self._latest = sample

                now = time.time()
                if now - last_report >= 5.0:
                    print(
                        f"[PicoReader] shape={body_poses.shape}, "
                        f"dt_ts={device_dt * 1000.0:.2f} ms, fps={self._fps_ema:.2f}"
                    )
                    last_report = now
            except Exception as exc:
                print(f"[PicoReader] read error: {exc}")


class ZmqReferenceSender:
    def __init__(self, uri: str, logger, topic: bytes = OUT_TOPIC, mode: str = "bind", conflate: bool = True):
        self.logger = logger
        self.topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        if conflate and hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1)

        if mode == "bind":
            self._socket.bind(uri)
        elif mode == "connect":
            self._socket.connect(uri)
        else:
            raise ValueError("mode must be 'bind' or 'connect'")

        self._last_send_time: Optional[float] = None
        self._send_freq_log: list[float] = []
        self._frame_count = 0
        self.logger.info(f"[ZMQOut] sender ready: mode={mode}, uri={uri}, topic={topic.decode('utf-8')}")

    def send(self, reference_qpos: np.ndarray, frame_index: int, sample_meta: dict) -> None:
        payload = {
            "reference_qpos": np.asarray(reference_qpos, dtype=np.float32),
            "frame_index": np.array([frame_index], dtype=np.int64),
            "timestamp_realtime": np.array([sample_meta["timestamp_realtime"]], dtype=np.float64),
            "timestamp_monotonic": np.array([sample_meta["timestamp_monotonic"]], dtype=np.float64),
            "timestamp_ns": np.array([sample_meta["timestamp_ns"]], dtype=np.int64),
            "source_timestamp_realtime": np.array(
                [sample_meta["source_timestamp_realtime"]], dtype=np.float64
            ),
            "source_timestamp_monotonic": np.array(
                [sample_meta["source_timestamp_monotonic"]], dtype=np.float64
            ),
            "source_timestamp_ns": np.array(
                [sample_meta["source_timestamp_ns"]], dtype=np.int64
            ),
            "pico_dt": np.array([sample_meta["dt"]], dtype=np.float32),
            "pico_fps": np.array([sample_meta["fps"]], dtype=np.float32),
        }

        hand_joints = sample_meta.get("hand_joints")
        if hand_joints is not None:
            payload["hand_joints"] = np.asarray(hand_joints, dtype=np.float32).reshape(
                HAND_DOF_DIM
            )
        head_pan_tilt = sample_meta.get("head_pan_tilt")
        if head_pan_tilt is not None:
            payload["head_pan_tilt"] = np.asarray(
                head_pan_tilt, dtype=np.float32
            ).reshape(HEAD_DOF_DIM)
        head_quat = sample_meta.get("head_quat")
        if head_quat is not None:
            payload["head_quat"] = np.asarray(head_quat, dtype=np.float32).reshape(4)
        for key in ("left_trigger", "right_trigger", "left_grip", "right_grip"):
            if key in sample_meta:
                payload[key] = np.array([float(sample_meta[key])], dtype=np.float32)
        if any(key in sample_meta for key in BUTTON_FIELDS):
            payload["buttons"] = np.array(
                [float(sample_meta.get(key, 0.0)) for key in BUTTON_FIELDS],
                dtype=np.float32,
            )
        for key in ("left_axis", "right_axis"):
            if key in sample_meta:
                payload[key] = np.asarray(sample_meta[key], dtype=np.float32).reshape(2)

        self._socket.send(pack_numpy_message(payload, topic=self.topic))

        now = time.time()
        if self._last_send_time is not None:
            dt = now - self._last_send_time
            if dt > 0:
                self._send_freq_log.append(1.0 / dt)
                self._frame_count += 1
                if self._frame_count >= 50:
                    avg_freq = sum(self._send_freq_log) / len(self._send_freq_log)
                    self.logger.info(f"Average ZMQ send rate: {avg_freq:.2f} Hz")
                    self._send_freq_log.clear()
                    self._frame_count = 0
        self._last_send_time = now

    def stop(self) -> None:
        self._socket.close(0)
        self._context.term()
        self.logger.info("ZMQ reference sender stopped")


def load_pico_raw_samples_csv(path: str, limit: int) -> dict[str, np.ndarray]:
    frames: list[np.ndarray] = []
    timestamp_realtime: list[float] = []
    timestamp_monotonic: list[float] = []
    timestamp_ns: list[int] = []
    pico_dt: list[float] = []
    pico_fps: list[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = np.empty((24, 7), dtype=np.float32)
            for i in range(24):
                for j in range(7):
                    frame[i, j] = float(row[f"body_pose_{i}_{j}"])
            frames.append(frame)
            timestamp_realtime.append(float(row.get("timestamp_realtime", "nan")))
            timestamp_monotonic.append(float(row.get("timestamp_monotonic", "nan")))
            timestamp_ns.append(int(row.get("timestamp_ns", "0") or 0))
            pico_dt.append(float(row.get("pico_dt", "nan")))
            pico_fps.append(float(row.get("pico_fps", "nan")))
            if len(frames) >= limit:
                break
    if not frames:
        raise RuntimeError(f"no Pico frames loaded from {path}")
    return {
        "frames": np.stack(frames, axis=0),
        "timestamp_realtime": np.asarray(timestamp_realtime, dtype=np.float64),
        "timestamp_monotonic": np.asarray(timestamp_monotonic, dtype=np.float64),
        "timestamp_ns": np.asarray(timestamp_ns, dtype=np.int64),
        "pico_dt": np.asarray(pico_dt, dtype=np.float64),
        "pico_fps": np.asarray(pico_fps, dtype=np.float64),
    }


def playback_dt_from_pico_samples(samples: dict[str, np.ndarray], fallback_hz: float) -> np.ndarray:
    n = int(samples["frames"].shape[0])
    fallback_dt = 1.0 / float(fallback_hz)
    playback_dt = np.full(n, fallback_dt, dtype=np.float64)

    pico_dt = samples.get("pico_dt")
    if pico_dt is not None:
        valid = np.isfinite(pico_dt) & (pico_dt > 1e-6) & (pico_dt < 0.2)
        if np.any(valid):
            fill = float(np.median(pico_dt[valid]))
            playback_dt[:] = fill
            playback_dt[valid] = pico_dt[valid]
            return playback_dt

    timestamp_ns = samples.get("timestamp_ns")
    if timestamp_ns is not None and timestamp_ns.shape[0] >= 2:
        diffs = np.diff(timestamp_ns.astype(np.int64)).astype(np.float64) * 1e-9
        valid = np.isfinite(diffs) & (diffs > 1e-6) & (diffs < 0.2)
        if np.any(valid):
            fill = float(np.median(diffs[valid]))
            playback_dt[:] = fill
            playback_dt[1:][valid] = diffs[valid]
            return playback_dt
    return playback_dt


def make_fake_pico_frame(t: float) -> np.ndarray:
    body = np.zeros((24, 7), dtype=np.float32)
    body[:, 6] = 1.0
    body[:, 1] = np.linspace(0.0, 1.6, 24, dtype=np.float32)
    body[0, :3] = np.array([0.0, 0.9, 0.0], dtype=np.float32)
    swing = 0.6 * np.sin(2.0 * np.pi * 0.5 * t)
    half = 0.5 * swing
    qz = np.float32(np.sin(half))
    qw = np.float32(np.cos(half))
    for joint_id in (16, 18, 20, 22):
        body[joint_id, 3:7] = np.array([0.0, 0.0, qz, qw], dtype=np.float32)
    return body


def install_stop_signal_handlers() -> threading.Event:
    stop_event = threading.Event()

    def request_stop(signum, frame):
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return stop_event


class FakePicoReader:
    def __init__(
        self,
        *,
        hz: float,
        samples: Optional[dict[str, np.ndarray]] = None,
        max_frames: int = 0,
        loop_frames: bool = True,
    ) -> None:
        self.hz = float(hz)
        self.samples = samples
        self.frames = None if samples is None else samples["frames"]
        self.playback_dt = None if samples is None else playback_dt_from_pico_samples(samples, self.hz)
        self.max_frames = max(0, int(max_frames))
        self.loop_frames = bool(loop_frames)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return self._latest

    def is_done(self) -> bool:
        return self._done.is_set()

    def _frame_at(self, index: int) -> np.ndarray:
        if self.frames is None:
            return make_fake_pico_frame(index / self.hz)
        if index < len(self.frames):
            return self.frames[index]
        if not self.loop_frames:
            return self.frames[-1]
        return self.frames[index % len(self.frames)]

    def _run(self) -> None:
        next_time = time.time()
        index = 0
        while not self._stop.is_set():
            if self.max_frames > 0 and index >= self.max_frames:
                self._done.set()
                return
            source_index = index if self.frames is None or index < len(self.frames) else index % len(self.frames)
            dt = float(self.playback_dt[source_index]) if self.playback_dt is not None else 1.0 / self.hz
            sample = {
                "body_poses_np": np.asarray(self._frame_at(index), dtype=np.float32).copy(),
                "timestamp_realtime": time.time(),
                "timestamp_monotonic": time.monotonic(),
                "timestamp_ns": int(time.time_ns()),
                "dt": dt,
                "fps": float(1.0 / dt),
            }
            with self._lock:
                self._latest = sample
            index += 1
            next_source_index = index if self.frames is None or index < len(self.frames) else index % len(self.frames)
            interval = float(self.playback_dt[next_source_index]) if self.playback_dt is not None else 1.0 / self.hz
            next_time += interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.time()


class HoloRetargetTeleopNode:
    def __init__(
        self,
        *,
        robot_zmq_uri: str,
        robot_zmq_mode: str,
        hz: float,
        timing_log_every: int,
        save_reference_path: str = "",
        debug_retarget_dump: str = "",
        asset_root: str = "",
        profile_timing: bool = False,
        enable_io: bool = True,
        start_loop: bool = True,
        hand_head_controller: Optional[HandHeadController] = None,
    ) -> None:
        self.hand_head = (
            hand_head_controller
            if hand_head_controller is not None
            else HandHeadController(hz=hz)
        )
        self.hz = float(hz)
        self.timing_log_every = max(1, int(timing_log_every))
        self.save_reference_path = save_reference_path
        self.debug_retarget_dump_path = debug_retarget_dump
        self.latest_sample: Optional[dict] = None
        self.latest_body_poses_np: Optional[np.ndarray] = None
        self.last_processed_timestamp_ns: Optional[int] = None
        self.frame_index = 0
        self.tick_count = 0
        self.timing_sums_ms: dict[str, float] = defaultdict(float)
        self.detail_timing_sums_ms: dict[str, float] = defaultdict(float)
        self.saved_references: list[np.ndarray] = []
        self.debug_records: list[dict] = []
        self.close_xrt_on_stop = bool(enable_io)
        self._loop_stop = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._retarget_rate_window_start = time.monotonic()
        self._retarget_completed_in_window = 0
        self._retarget_rate_log_interval_sec = 5.0
        self._source_timeout_sec = SOURCE_TIMEOUT_SEC

        self.retargeter = HoloRetargeter(
            HoloRetargetConfig(
                asset_root=asset_root,
                profile_timing=profile_timing,
            )
        )
        self.info(
            "HoloRetarget ready: "
            f"asset_root={self.retargeter.config.resolved_asset_root}, "
            f"table={self.retargeter.config.target_table}, "
            f"asset={self.retargeter.config.robot_asset}, "
            f"iter={self.retargeter.config.newton_iterations}, "
            f"step_size={self.retargeter.config.newton_step_size:.2f}, "
            f"root_seed={self.retargeter.config.root_seed_mode}, "
            f"max_joint_step={self.retargeter.config.max_joint_step:.3f}, "
            f"joint_limit_weight={self.retargeter.config.joint_limit_weight:.3f}"
        )

        self.reader = None
        self.sender = None
        if enable_io:
            self.reader = PicoReader()
            self.reader.start()
            self.sender = ZmqReferenceSender(
                uri=robot_zmq_uri, logger=self, mode=robot_zmq_mode
            )
            if start_loop:
                self.start_loop(self.hz)

    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def warning(self, msg: str) -> None:
        print(f"[WARN] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}")

    def push_body_poses(self, body_poses: np.ndarray) -> None:
        start = time.perf_counter()
        body_poses = np.asarray(body_poses, dtype=np.float32)
        if body_poses.shape != (24, 7):
            raise ValueError(f"body_poses must have shape (24, 7), got {body_poses.shape}")
        self.latest_body_poses_np = body_poses.copy()
        self._accumulate_timing("body_poses_input", start)

    def retarget_latest(self) -> Optional[np.ndarray]:
        if self.latest_body_poses_np is None:
            raise RuntimeError("no Pico body_poses frame available")
        start = time.perf_counter()
        reference_qpos = self.retargeter.retarget_qpos_from_body_poses(
            self.latest_body_poses_np
        )
        self._accumulate_timing("holoretarget", start)
        self._accumulate_detail_timing(self.retargeter.last_timing)
        return reference_qpos

    def start_loop(self, hz: float) -> None:
        self.info(f"Starting main loop at {hz:.1f} Hz")
        interval = 1.0 / float(hz)
        self._loop_stop.clear()

        def loop() -> None:
            next_time = time.monotonic()
            while not self._loop_stop.is_set():
                self._tick()
                next_time += interval
                sleep_time = next_time - time.monotonic()
                if sleep_time > 0:
                    self._loop_stop.wait(sleep_time)
                else:
                    next_time = time.monotonic()

        self._loop_thread = threading.Thread(target=loop, daemon=True)
        self._loop_thread.start()

    def _tick(self) -> None:
        tick_start = time.perf_counter()
        tick_realtime = time.time()
        tick_monotonic = time.monotonic()
        sample = self.reader.get_latest() if self.reader is not None else None
        try:
            sample_timestamp_ns = (
                int(sample["timestamp_ns"]) if sample is not None else None
            )
            is_new_sample = (
                sample is not None
                and sample_timestamp_ns != self.last_processed_timestamp_ns
            )
            if is_new_sample:
                self.latest_sample = sample
                self.push_body_poses(sample["body_poses_np"])
                self.last_processed_timestamp_ns = sample_timestamp_ns

            if self.latest_sample is not None:
                source_monotonic = float(
                    self.latest_sample.get("timestamp_monotonic", tick_monotonic)
                )
                source_age = max(0.0, tick_monotonic - source_monotonic)
                if source_age <= self._source_timeout_sec:
                    reference_qpos = self.retarget_latest()
                else:
                    reference_qpos = None
                if reference_qpos is not None:
                    self._retarget_completed_in_window += 1
                    sample_meta = self._make_output_sample_meta(
                        self.latest_sample,
                        tick_realtime=tick_realtime,
                        tick_monotonic=tick_monotonic,
                    )
                    hand_joints, head_pan_tilt = self.hand_head.process(
                        self.latest_sample
                    )
                    sample_meta["hand_joints"] = hand_joints
                    sample_meta["head_pan_tilt"] = head_pan_tilt
                    self._publish(
                        reference_qpos,
                        sample_meta=sample_meta,
                        body_poses=self.latest_body_poses_np,
                    )
        except Exception as exc:
            self.error(f"[tick_error] {exc}")
            self.error(traceback.format_exc())
            return
        self.tick_count += 1
        self._accumulate_timing("tick_total", tick_start)
        self._maybe_log_retarget_rate()
        self._maybe_log_timing()

    @staticmethod
    def _make_output_sample_meta(
        source_meta: dict,
        *,
        tick_realtime: float,
        tick_monotonic: float,
    ) -> dict:
        output_meta = dict(source_meta)
        output_meta["source_timestamp_realtime"] = float(
            source_meta["timestamp_realtime"]
        )
        output_meta["source_timestamp_monotonic"] = float(
            source_meta["timestamp_monotonic"]
        )
        output_meta["source_timestamp_ns"] = int(source_meta["timestamp_ns"])
        output_meta["timestamp_realtime"] = float(tick_realtime)
        output_meta["timestamp_monotonic"] = float(tick_monotonic)
        output_meta["timestamp_ns"] = int(tick_realtime * 1_000_000_000)
        return output_meta

    def _maybe_log_retarget_rate(self) -> None:
        now = time.monotonic()
        elapsed = now - self._retarget_rate_window_start
        if elapsed < self._retarget_rate_log_interval_sec:
            return
        actual_hz = self._retarget_completed_in_window / elapsed
        self.info(
            f"[Retarget] actual={actual_hz:.1f}Hz target={self.hz:.1f}Hz"
        )
        self._retarget_rate_window_start = now
        self._retarget_completed_in_window = 0

    def _publish(
        self,
        reference_qpos: np.ndarray,
        *,
        sample_meta: Optional[dict] = None,
        body_poses: Optional[np.ndarray] = None,
    ) -> None:
        reference_qpos = np.asarray(reference_qpos, dtype=np.float32)
        if reference_qpos.shape != (QPOS_DIM,):
            self.error(
                f"Output shape {reference_qpos.shape} != expected ({QPOS_DIM},)"
            )
            return
        if not np.isfinite(reference_qpos).all():
            self.error("NaN or Inf detected in reference qpos; skip publish")
            return
        self._record_debug(reference_qpos, body_poses=body_poses)
        if self.sender is not None:
            meta = self.latest_sample if sample_meta is None else sample_meta
            if meta is None:
                raise RuntimeError("sample metadata is required to publish reference qpos")
            self.sender.send(reference_qpos, self.frame_index, meta)
        if self.save_reference_path:
            self.saved_references.append(reference_qpos.copy())
        self.frame_index += 1

    def _accumulate_timing(self, name: str, start_time: float) -> float:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.timing_sums_ms[name] += elapsed_ms
        return elapsed_ms

    def _accumulate_detail_timing(self, timing: Optional[dict]) -> None:
        if not timing:
            return
        for key, value in timing.items():
            if isinstance(value, (int, float)):
                self.detail_timing_sums_ms[key] += float(value) * 1000.0

    def _maybe_log_timing(self) -> None:
        if self.tick_count <= 0 or self.tick_count % self.timing_log_every != 0:
            return
        keys = ("body_poses_input", "holoretarget", "postprocess_send", "tick_total")
        parts = [
            f"{key}={self.timing_sums_ms[key] / self.timing_log_every:.2f}ms"
            for key in keys
            if key in self.timing_sums_ms
        ]
        if parts:
            self.info("[Timing] " + ", ".join(parts))
        detail = [
            f"{key.split('.', 1)[1]}={self.detail_timing_sums_ms[key] / self.timing_log_every:.2f}ms"
            for key in DETAIL_TIMING_KEYS
            if key in self.detail_timing_sums_ms
        ]
        if detail:
            self.info("[Timing] holoretarget_detail: " + ", ".join(detail))
        self.timing_sums_ms.clear()
        self.detail_timing_sums_ms.clear()

    def _record_debug(
        self,
        reference_qpos: np.ndarray,
        *,
        body_poses: Optional[np.ndarray] = None,
    ) -> None:
        if not self.debug_retarget_dump_path:
            return
        record = {
            "frame_index": int(self.frame_index),
            "reference_qpos": reference_qpos.copy(),
        }
        source_body_poses = (
            self.latest_body_poses_np if body_poses is None else body_poses
        )
        if source_body_poses is not None:
            record["pico_body_poses"] = source_body_poses.copy()
        self.debug_records.append(record)

    def save_debug_retarget_dump(self) -> None:
        if not self.debug_retarget_dump_path or not self.debug_records:
            return
        save_path = Path(self.debug_retarget_dump_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for record in self.debug_records for key in record.keys()})
        payload = {}
        for key in keys:
            values = [record.get(key) for record in self.debug_records]
            non_null = [value for value in values if value is not None]
            if non_null and all(isinstance(value, np.ndarray) for value in non_null):
                shapes = {value.shape for value in non_null}
                if len(shapes) == 1 and len(non_null) == len(values):
                    payload[key] = np.stack(values, axis=0)
                    continue
            payload[key] = np.asarray(values, dtype=object)
        np.savez_compressed(save_path, **payload)
        self.info(f"[DebugRetarget] saved {len(self.debug_records)} frames to {save_path}")

    def save_references(self) -> None:
        if not self.save_reference_path or not self.saved_references:
            return
        save_path = Path(self.save_reference_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        reference_array = np.stack(self.saved_references, axis=0).astype(np.float32)
        if str(save_path).endswith(".npy"):
            np.save(save_path, reference_array)
        else:
            np.savez_compressed(save_path, reference_qpos=reference_array)
        self.info(
            f"[SaveReference] saved {reference_array.shape[0]} frames to {save_path}"
        )

    def stop(self) -> None:
        self._loop_stop.set()
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)
        if self.reader is not None:
            self.reader.stop()
        if self.sender is not None:
            self.sender.stop()
        self.save_references()
        self.save_debug_retarget_dump()
        try:
            if self.close_xrt_on_stop and xrt is not None and hasattr(xrt, "close"):
                xrt.close()
        except Exception:
            pass


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def format_stats(name: str, values: list[float]) -> str:
    return (
        f"{name}: mean={statistics.mean(values):.3f} ms, "
        f"p50={statistics.median(values):.3f} ms, "
        f"p95={percentile(values, 95):.3f} ms, "
        f"p99={percentile(values, 99):.3f} ms, "
        f"min={min(values):.3f} ms, max={max(values):.3f} ms"
    )


def run_fake_benchmark(args: argparse.Namespace) -> int:
    warmup = max(0, int(args.fake_warmup))
    frames = max(1, int(args.fake_frames))
    total = warmup + frames + 1
    node = HoloRetargetTeleopNode(
        robot_zmq_uri=args.robot_zmq_uri,
        robot_zmq_mode=args.robot_zmq_mode,
        hz=args.hz,
        timing_log_every=max(total + 1, int(args.timing_log_every)),
        save_reference_path="",
        debug_retarget_dump=args.debug_retarget_dump,
        asset_root=args.asset_root,
        profile_timing=args.profile_timing,
        enable_io=False,
        start_loop=False,
    )

    samples = None
    if args.fake_pico_csv:
        csv_path = str(Path(args.fake_pico_csv).expanduser().resolve())
        samples = load_pico_raw_samples_csv(csv_path, total)
        if samples["frames"].shape[0] < total:
            raise RuntimeError(f"CSV has {samples['frames'].shape[0]} frames, need {total}")
        print(f"[fake] loaded Pico raw CSV: {csv_path}, frames={samples['frames'].shape[0]}")

    playback_dt = None if samples is None else playback_dt_from_pico_samples(samples, args.hz)
    tick_ms: list[float] = []
    push_ms: list[float] = []
    retarget_ms: list[float] = []
    qpos_history: list[np.ndarray] = []
    detail_sums: dict[str, float] = defaultdict(float)
    detail_count = 0
    fallback_dt = 1.0 / float(args.hz)

    for i in range(total):
        frame = samples["frames"][i] if samples is not None else make_fake_pico_frame(i * fallback_dt)
        dt = float(playback_dt[i]) if playback_dt is not None else fallback_dt
        node.latest_sample = {
            "body_poses_np": frame,
            "timestamp_realtime": time.time(),
            "timestamp_monotonic": time.monotonic(),
            "timestamp_ns": int(time.time_ns()),
            "dt": dt,
            "fps": float(1.0 / dt),
        }
        t0 = time.perf_counter()
        node.push_body_poses(frame)
        t1 = time.perf_counter()
        reference_qpos = node.retarget_latest()
        t2 = time.perf_counter()
        if reference_qpos is None:
            continue
        if reference_qpos.shape != (QPOS_DIM,) or not np.isfinite(
            reference_qpos
        ).all():
            raise RuntimeError(
                f"invalid reference qpos at frame {i}: shape={reference_qpos.shape}"
            )
        node._record_debug(reference_qpos)
        if i > warmup:
            push_ms.append((t1 - t0) * 1000.0)
            retarget_ms.append((t2 - t1) * 1000.0)
            tick_ms.append((t2 - t0) * 1000.0)
            qpos_history.append(reference_qpos.copy())
            last = getattr(node.retargeter, "last_timing", None)
            if last:
                detail_count += 1
                for key, value in last.items():
                    if isinstance(value, (int, float)):
                        detail_sums[key] += float(value)

    qpos_array = np.stack(qpos_history, axis=0)
    print("\n=== HoloRetarget teleop fake benchmark ===")
    print(f"hz={args.hz:.1f}, frames={frames}, warmup={warmup}")
    print(format_stats("tick_total(push+retarget)", tick_ms))
    print(format_stats("push_body_poses", push_ms))
    print(format_stats("retarget_latest", retarget_ms))
    print(f"effective_rate={1000.0 / statistics.mean(tick_ms):.2f} Hz")
    print(
        f"qpos_shape={qpos_array.shape}, "
        f"all_finite={bool(np.isfinite(qpos_array).all())}"
    )
    if detail_count:
        parts = [
            f"{key.split('.', 1)[1]}={detail_sums[key] * 1000.0 / detail_count:.3f}ms"
            for key in DETAIL_TIMING_KEYS
            if key in detail_sums
        ]
        if parts:
            print("avg_profile: " + ", ".join(parts))
    node.save_debug_retarget_dump()
    return 0


def run_fake_stream(args: argparse.Namespace) -> int:
    stop_event = install_stop_signal_handlers()
    samples = None
    if args.fake_pico_csv:
        limit = int(args.fake_stream_frames) if int(args.fake_stream_frames) > 0 else 10000
        csv_path = str(Path(args.fake_pico_csv).expanduser().resolve())
        samples = load_pico_raw_samples_csv(csv_path, limit)
        playback_dt = playback_dt_from_pico_samples(samples, args.hz)
        print(f"[fake-stream] loaded Pico raw CSV: {csv_path}, frames={samples['frames'].shape[0]}")
        print(
            "[fake-stream] playback timing from CSV: "
            f"mean_dt={playback_dt.mean() * 1000.0:.2f}ms, "
            f"p50_dt={np.percentile(playback_dt, 50) * 1000.0:.2f}ms, "
            f"p95_dt={np.percentile(playback_dt, 95) * 1000.0:.2f}ms, "
            f"mean_fps={1.0 / playback_dt.mean():.2f}"
        )

    node = HoloRetargetTeleopNode(
        robot_zmq_uri=args.robot_zmq_uri,
        robot_zmq_mode=args.robot_zmq_mode,
        hz=args.hz,
        timing_log_every=args.timing_log_every,
        save_reference_path=args.save_reference_path,
        debug_retarget_dump=args.debug_retarget_dump,
        asset_root=args.asset_root,
        profile_timing=args.profile_timing,
        enable_io=False,
        start_loop=False,
        hand_head_controller=build_hand_head_controller(args),
    )
    node.reader = FakePicoReader(
        hz=args.hz,
        samples=samples,
        max_frames=args.fake_stream_frames,
        loop_frames=not args.fake_stream_no_loop,
    )
    node.reader.start()
    node.sender = ZmqReferenceSender(
        uri=args.robot_zmq_uri, logger=node, mode=args.robot_zmq_mode
    )
    node.start_loop(args.hz)

    try:
        if int(args.fake_stream_frames) > 0:
            while not stop_event.is_set() and not node.reader.is_done():
                time.sleep(0.05)
            time.sleep(0.2)
        else:
            while not stop_event.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        node.info("Interrupted by user")
    finally:
        node.stop()
    return 0


def init_xrt(start_service: bool = True) -> None:
    if xrt is None:
        raise ImportError("XRoboToolkit SDK not available. Install xrobotoolkit_sdk first.")
    if start_service:
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
    xrt.init()
    print("Waiting for body tracking data...")
    while not xrt.is_body_data_available():
        print("waiting for body data...")
        time.sleep(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XRT Pico -> HoloRetarget -> robot ZMQ(reference qpos)"
    )
    parser.add_argument("--robot-zmq-uri", default="tcp://*:6001")
    parser.add_argument("--robot-zmq-mode", choices=["bind", "connect"], default="bind")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--timing-log-every", type=int, default=200)
    parser.add_argument("--save-reference-path", default="")
    parser.add_argument("--debug-retarget-dump", default="")
    parser.add_argument("--skip-start-service", action="store_true")
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument("--asset-root", default=str(default_asset_root()))
    parser.add_argument("--fake-benchmark", action="store_true")
    parser.add_argument("--fake-pico-stream", action="store_true")
    parser.add_argument("--fake-pico-csv", default="")
    parser.add_argument("--fake-frames", type=int, default=200)
    parser.add_argument("--fake-warmup", type=int, default=30)
    parser.add_argument("--fake-stream-frames", type=int, default=0)
    parser.add_argument("--fake-stream-no-loop", action="store_true")
    parser.add_argument(
        "--hand-source",
        choices=["trigger", "joystick", "off"],
        default="trigger",
        help="Dex3 hand command source: controller triggers, thumbstick y (push up to close), or off",
    )
    parser.add_argument(
        "--head-source",
        choices=["headset", "joystick", "off"],
        default="headset",
        help="Head servo command source: operator headset orientation, right thumbstick rates, or off",
    )
    parser.add_argument("--head-follow-gain", type=float, default=1.0, help="scale applied to head pan/tilt")
    parser.add_argument("--head-follow-yaw-axis", type=int, default=1, help="head-delta rotvec axis driving pan (0=x,1=y,2=z); Pico frame is Y-up")
    parser.add_argument("--head-follow-yaw-sign", type=float, default=1.0, help="sign for pan (+1/-1)")
    parser.add_argument("--head-follow-pitch-axis", type=int, default=0, help="head-delta rotvec axis driving tilt")
    parser.add_argument("--head-follow-pitch-sign", type=float, default=-1.0, help="sign for tilt (+1/-1)")
    parser.add_argument("--head-joystick-rate", type=float, default=1.5, help="head pan/tilt rate (rad/s) in --head-source joystick mode")
    return parser


def build_hand_head_controller(args: argparse.Namespace) -> HandHeadController:
    return HandHeadController(
        hand_source=args.hand_source,
        head_source=args.head_source,
        head_follow_gain=args.head_follow_gain,
        head_follow_yaw_axis=args.head_follow_yaw_axis,
        head_follow_yaw_sign=args.head_follow_yaw_sign,
        head_follow_pitch_axis=args.head_follow_pitch_axis,
        head_follow_pitch_sign=args.head_follow_pitch_sign,
        head_joystick_rate=args.head_joystick_rate,
        hz=args.hz,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.fake_benchmark:
        return run_fake_benchmark(args)
    if args.fake_pico_stream:
        return run_fake_stream(args)

    stop_event = install_stop_signal_handlers()
    init_xrt(start_service=not args.skip_start_service)
    node = HoloRetargetTeleopNode(
        robot_zmq_uri=args.robot_zmq_uri,
        robot_zmq_mode=args.robot_zmq_mode,
        hz=args.hz,
        timing_log_every=args.timing_log_every,
        save_reference_path=args.save_reference_path,
        debug_retarget_dump=args.debug_retarget_dump,
        asset_root=args.asset_root,
        profile_timing=args.profile_timing,
        hand_head_controller=build_hand_head_controller(args),
    )
    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        node.stop()
        print("Program terminated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
