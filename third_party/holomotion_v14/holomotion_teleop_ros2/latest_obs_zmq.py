from __future__ import annotations

import json
import time
from typing import Any

import numpy as np


HEADER_SIZE = 1280
DEFAULT_ZMQ_TOPIC = b"obs65"

_NUMPY_DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "bool": np.dtype("?"),
}


def decode_zmq_topic(topic_value: bytes | str) -> bytes:
    if isinstance(topic_value, bytes):
        return topic_value
    return str(topic_value).encode("utf-8")


def _dtype_name(value: np.ndarray) -> tuple[str, np.ndarray]:
    if value.dtype == np.float32:
        return "f32", value
    if value.dtype == np.float64:
        return "f64", value
    if value.dtype == np.int32:
        return "i32", value
    if value.dtype == np.int64:
        return "i64", value
    if value.dtype == np.uint8:
        return "u8", value
    if value.dtype == np.bool_:
        return "bool", value
    return "f32", value.astype(np.float32)


def pack_numpy_message(payload: dict[str, Any], topic: bytes = DEFAULT_ZMQ_TOPIC, version: int = 1) -> bytes:
    fields = []
    binary_data = []

    for key, raw_value in payload.items():
        if not isinstance(raw_value, np.ndarray):
            continue
        dtype_str, value = _dtype_name(raw_value)
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


def unpack_numpy_message(packet: bytes, expected_topic: bytes | str) -> dict[str, np.ndarray]:
    topic = decode_zmq_topic(expected_topic)
    if not packet.startswith(topic):
        raise ValueError("ZMQ packet topic does not match expected topic")

    header_start = len(topic)
    header_end = header_start + HEADER_SIZE
    if len(packet) < header_end:
        raise ValueError("ZMQ packet is shorter than its fixed header")

    raw_header = packet[header_start:header_end].rstrip(b"\x00")
    if not raw_header:
        raise ValueError("ZMQ packet header is empty")
    header = json.loads(raw_header.decode("utf-8"))
    if header.get("endian") != "le":
        raise ValueError("Only little-endian ZMQ packets are supported")

    payload: dict[str, np.ndarray] = {}
    offset = header_end
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        if dtype_name not in _NUMPY_DTYPES:
            raise ValueError(f"Unsupported ZMQ field dtype: {dtype_name}")
        shape = tuple(int(value) for value in field.get("shape", []))
        dtype = _NUMPY_DTYPES[dtype_name]
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        size = count * dtype.itemsize
        field_end = offset + size
        if field_end > len(packet):
            raise ValueError(f"ZMQ field {name} exceeds packet length")
        value = np.frombuffer(packet[offset:field_end], dtype=dtype, count=count)
        payload[name] = value.reshape(shape).copy() if shape else value.copy()
        offset = field_end
    return payload


class ZmqLatestObsPublisher:
    def __init__(
        self,
        uri: str,
        logger,
        topic: bytes | str = DEFAULT_ZMQ_TOPIC,
        mode: str = "bind",
        conflate: bool = True,
        log_every: int = 50,
    ) -> None:
        import zmq

        self.logger = logger
        self.topic = decode_zmq_topic(topic)
        self.mode = str(mode).strip().lower()
        self.uri = str(uri)
        self.log_every = max(0, int(log_every))
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        if conflate and hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1)

        if self.mode == "bind":
            self._socket.bind(self.uri)
        elif self.mode == "connect":
            self._socket.connect(self.uri)
        else:
            self._socket.close(0)
            self._context.term()
            raise ValueError("zmq_mode must be 'bind' or 'connect'")

        self._last_send_time: float | None = None
        self._send_freq_log: list[float] = []
        self._frame_count = 0
        self.logger.info(
            f"[ZMQOut] latest_obs publisher ready: mode={self.mode}, "
            f"uri={self.uri}, topic={self.topic.decode('utf-8', errors='replace')}"
        )

    def send(
        self,
        latest_obs: np.ndarray,
        frame_index: int,
        timestamp_realtime: float,
        timestamp_monotonic: float,
        timestamp_ns: int,
        pico_dt: float,
        pico_fps: float,
    ) -> None:
        payload = {
            "latest_obs": np.asarray(latest_obs, dtype=np.float32),
            "frame_index": np.array([frame_index], dtype=np.int64),
            "timestamp_realtime": np.array([timestamp_realtime], dtype=np.float64),
            "timestamp_monotonic": np.array([timestamp_monotonic], dtype=np.float64),
            "timestamp_ns": np.array([timestamp_ns], dtype=np.int64),
            "pico_dt": np.array([pico_dt], dtype=np.float32),
            "pico_fps": np.array([pico_fps], dtype=np.float32),
        }
        packet = pack_numpy_message(payload, topic=self.topic)
        self._socket.send(packet)
        self._record_send_frequency()

    def _record_send_frequency(self) -> None:
        if self.log_every <= 0:
            return
        now = time.time()
        if self._last_send_time is not None:
            dt = now - self._last_send_time
            if dt > 0.0:
                self._send_freq_log.append(1.0 / dt)
                self._frame_count += 1
                if self._frame_count >= self.log_every:
                    avg_freq = sum(self._send_freq_log) / len(self._send_freq_log)
                    self.logger.info(f"[ZMQOut] average send rate: {avg_freq:.2f} Hz")
                    self._send_freq_log.clear()
                    self._frame_count = 0
        self._last_send_time = now

    def stop(self) -> None:
        self._socket.close(0)
        self._context.term()
