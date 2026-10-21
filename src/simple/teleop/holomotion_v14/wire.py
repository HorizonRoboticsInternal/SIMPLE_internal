"""
ZMQ channel between the v1.4 reference publisher and SIMPLE.

One PUB socket carries two topics, both packed with the robot's own codec
(``holomotion_teleop_ros2/latest_obs_zmq.py``: topic + 1280-byte JSON header + little-endian arrays):

* ``obs65``      the robot's latest_obs frame, same payload as ``ZmqLatestObsPublisher.send``
                 (latest_obs[65] = dof_pos[29] dof_vel[29] root_pos[3] root_rot_wxyz[4], frame_index, timestamps);
* ``pico_ctrl``  the PICO controller sample (v1.4.1 ``PicoControlSample``: button bits, sticks, triggers, grips).

Self-contained (numpy + pyzmq + the vendored codec) so the publisher can load it from the holomotion_teleop env
without importing the ``simple`` package.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_VENDOR = Path(os.environ.get("HOLOMOTION_V14_VENDOR_DIR", Path(__file__).resolve().parents[4] / "third_party" / "holomotion_v14"))
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
from holomotion_teleop_ros2.latest_obs_zmq import pack_numpy_message, unpack_numpy_message  # noqa: E402
from humanoid_policy.v141_pico_control import (  # noqa: E402
    BUTTON_A, BUTTON_B, BUTTON_L3, BUTTON_LEFT_MENU, BUTTON_R3, BUTTON_RIGHT_MENU, BUTTON_X, BUTTON_Y,
)

OBS65_TOPIC = b"obs65"
PICO_TOPIC = b"pico_ctrl"
BUTTONS = {"L3": BUTTON_L3, "X": BUTTON_X, "A": BUTTON_A, "R3": BUTTON_R3, "B": BUTTON_B, "Y": BUTTON_Y,
           "left_menu": BUTTON_LEFT_MENU, "right_menu": BUTTON_RIGHT_MENU}


def pico_payload(sample) -> dict:
    """A v1.4.1 PicoControlSample (or a dict with the same fields) -> wire payload."""
    g = (lambda k, d=0.0: sample.get(k, d)) if isinstance(sample, dict) else (lambda k, d=0.0: getattr(sample, k, d))
    return {
        "timestamp_ns": np.array([int(g("timestamp_ns", time.time_ns()))], dtype=np.int64),
        "button_bits": np.array([int(g("button_bits", 0))], dtype=np.int32),
        "left_axis": np.asarray(g("left_axis", (0.0, 0.0)), dtype=np.float32).reshape(2),
        "right_axis": np.asarray(g("right_axis", (0.0, 0.0)), dtype=np.float32).reshape(2),
        "triggers": np.array([g("left_trigger"), g("right_trigger")], dtype=np.float32),
        "grips": np.array([g("left_grip"), g("right_grip")], dtype=np.float32),
    }


def pico_sample(payload: dict) -> dict:
    """Wire payload -> the flat dict SimPolicyNode.feed_pico / the agent read (named buttons, sticks, triggers, grips)."""
    bits = int(payload["button_bits"].reshape(-1)[0])
    s = {name: int(bool(bits & bit)) for name, bit in BUTTONS.items()}
    s.update(button_bits=bits, timestamp_ns=int(payload["timestamp_ns"].reshape(-1)[0]),
             left_axis=tuple(payload["left_axis"].tolist()), right_axis=tuple(payload["right_axis"].tolist()),
             left_trigger=float(payload["triggers"][0]), right_trigger=float(payload["triggers"][1]),
             left_grip=float(payload["grips"][0]), right_grip=float(payload["grips"][1]),
             l_active=1, r_active=1, received_monotonic=time.monotonic())
    return s


class ReferencePublisher:
    def __init__(self, uri: str = "tcp://*:6001") -> None:
        import zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, 64)
        self._sock.bind(uri)
        self.uri = uri

    def send_obs65(self, latest_obs, frame_index: int, *, timestamp_ns: int = 0, pico_dt: float = 0.0,
                   pico_fps: float = 0.0) -> None:
        payload = {
            "latest_obs": np.asarray(latest_obs, dtype=np.float32).reshape(65),
            "frame_index": np.array([frame_index], dtype=np.int64),
            "timestamp_realtime": np.array([time.time()], dtype=np.float64),
            "timestamp_monotonic": np.array([time.monotonic()], dtype=np.float64),
            "timestamp_ns": np.array([timestamp_ns], dtype=np.int64),
            "pico_dt": np.array([pico_dt], dtype=np.float32),
            "pico_fps": np.array([pico_fps], dtype=np.float32),
        }
        self._sock.send(pack_numpy_message(payload, topic=OBS65_TOPIC))

    def send_pico(self, sample) -> None:
        self._sock.send(pack_numpy_message(pico_payload(sample), topic=PICO_TOPIC))

    def close(self) -> None:
        self._sock.close(0)


class ReferenceSubscriber:
    """Non-blocking reader: every obs65 frame since the last poll (in order) and the newest controller sample."""

    def __init__(self, uri: str = "tcp://127.0.0.1:6001") -> None:
        import zmq
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 256)
        self._sock.setsockopt(zmq.SUBSCRIBE, OBS65_TOPIC)
        self._sock.setsockopt(zmq.SUBSCRIBE, PICO_TOPIC)
        self._sock.connect(uri)
        self.uri = uri
        self.last_pico: dict | None = None
        self.last_obs_time: float | None = None
        self.n_obs = 0

    def poll(self, max_msgs: int = 512) -> list[dict]:
        frames = []
        for _ in range(max_msgs):
            try:
                packet = self._sock.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            if packet.startswith(OBS65_TOPIC):
                frames.append(unpack_numpy_message(packet, OBS65_TOPIC))
                self.last_obs_time = time.monotonic()
                self.n_obs += 1
            elif packet.startswith(PICO_TOPIC):
                self.last_pico = pico_sample(unpack_numpy_message(packet, PICO_TOPIC))
        return frames

    def pico_age(self) -> float:
        return float("inf") if self.last_pico is None else time.monotonic() - self.last_pico["received_monotonic"]

    def close(self) -> None:
        self._sock.close(0)
