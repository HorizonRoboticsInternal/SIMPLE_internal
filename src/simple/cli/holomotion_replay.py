"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Replay a recorded HoloMotion reference stream over ZMQ.

Publishes ``reference_qpos`` packets in the HoloMotion v1.4 wire format, so
``teleop-holomotion`` can be exercised without a PICO headset or the
HoloMotion publisher:

    holomotion-replay                    # bundled sample, looping
    holomotion-replay recording.npz --hand-demo

Input is an ``.npz`` holding ``reference_qpos`` float32[T, 36] (as written by
``holomotion_teleop_node.py --save-reference-path``). An optional
``frame_index`` int64[T] upsamples sparse telemetry logs back onto the 50 Hz
frame clock by linear interpolation.
"""

from __future__ import annotations

import time
from typing import Annotated

import numpy as np
import typer
import zmq

from simple.teleop.holomotion import (
    BUTTON_FIELDS,
    DEX3_LEFT_TRIGGER_CLOSE_POSE,
    DEX3_RIGHT_TRIGGER_CLOSE_POSE,
    REF_QPOS_DIM,
    pack_numpy_message,
    resolve_reference_sample,
)


def load_frames(path: str) -> np.ndarray:
    """Load reference frames, upsampling onto the uniform frame clock if needed."""
    data = np.load(path)
    qpos = np.asarray(data["reference_qpos"], dtype=np.float32).reshape(-1, REF_QPOS_DIM)
    if "frame_index" not in data:
        return qpos
    fidx = np.asarray(data["frame_index"], dtype=np.int64)
    if fidx.size != qpos.shape[0] or np.all(np.diff(fidx) == 1):
        return qpos
    grid = np.arange(fidx[0], fidx[-1] + 1)
    up = np.stack([np.interp(grid, fidx, qpos[:, k]) for k in range(REF_QPOS_DIM)], axis=1).astype(np.float32)
    up[:, 3:7] /= np.linalg.norm(up[:, 3:7], axis=1, keepdims=True)
    return up


def main(
    npz: Annotated[str, typer.Argument(help="Recording; defaults to the bundled sample")] = "",
    uri: Annotated[str, typer.Option(help="ZMQ endpoint to bind")] = "tcp://*:6001",
    hz: Annotated[float, typer.Option(help="Publish rate (the policy expects 50 Hz)")] = 50.0,
    loop: Annotated[bool, typer.Option(help="Repeat the recording indefinitely")] = True,
    hand_demo: Annotated[bool, typer.Option(help="Publish a slow open/close Dex3 cycle")] = False,
):
    path = resolve_reference_sample(npz)
    frames = load_frames(path)
    left_close = np.asarray(DEX3_LEFT_TRIGGER_CLOSE_POSE, dtype=np.float32)
    right_close = np.asarray(DEX3_RIGHT_TRIGGER_CLOSE_POSE, dtype=np.float32)
    print(f"[replay] {frames.shape[0]} frames ({frames.shape[0] / hz:.1f}s) from {path}")
    print(f"[replay] publishing to {uri} at {hz:.0f} Hz (loop={loop}, hand_demo={hand_demo})")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(uri)
    time.sleep(0.3)

    t0 = time.monotonic()
    i = 0
    try:
        while loop or i < frames.shape[0]:
            close = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.25 * i / hz)) if hand_demo else 0.0
            now = time.time()
            payload = {
                "reference_qpos": frames[i % frames.shape[0]],
                "frame_index": np.array([i], dtype=np.int64),
                "timestamp_realtime": np.array([now], dtype=np.float64),
                "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
                "timestamp_ns": np.array([time.time_ns()], dtype=np.int64),
                "source_timestamp_realtime": np.array([now], dtype=np.float64),
                "source_timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
                "source_timestamp_ns": np.array([time.time_ns()], dtype=np.int64),
                "pico_dt": np.array([1.0 / hz], dtype=np.float32),
                "pico_fps": np.array([hz], dtype=np.float32),
                "hand_joints": np.concatenate([close * left_close, close * right_close]).astype(np.float32),
                "head_pan_tilt": np.zeros(2, dtype=np.float32),
                "left_trigger": np.array([close], dtype=np.float32),
                "right_trigger": np.array([close], dtype=np.float32),
                "left_grip": np.zeros(1, dtype=np.float32),
                "right_grip": np.zeros(1, dtype=np.float32),
                "buttons": np.zeros(len(BUTTON_FIELDS), dtype=np.float32),
                "left_axis": np.zeros(2, dtype=np.float32),
                "right_axis": np.zeros(2, dtype=np.float32),
            }
            socket.send(pack_numpy_message(payload))
            i += 1
            sleep = t0 + i / hz - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        context.term()
    print(f"[replay] published {i} frames")


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
