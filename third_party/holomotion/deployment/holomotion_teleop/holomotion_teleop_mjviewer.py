#!/usr/bin/env python3
"""
Subscribe to reference qpos packets and visualize the robot pose in MuJoCo.

Expected packet layout:
- topic: b"reference_qpos"
- reference_qpos: float32[36] = root_pos[3] + root_rot_wxyz[4] + dof_pos[29]
- frame_index: int64[1]
- timestamp_realtime: float64[1]

Optional fields (published by the teleop node when hand/head control is on,
visualized with ``--model g1_comp``):
- hand_joints: float32[14] = Dex3 left(7) + right(7), per-side order
  thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1
- head_pan_tilt: float32[2] = xl330_joint (pan), d455_joint (tilt), radians

When those precomputed fields are absent but the stream carries the raw Pico
controller state (left/right_trigger, head_quat, ... — e.g. the Orin 6002
telemetry), the viewer derives hand_joints/head_pan_tilt locally with the same
trigger-interpolation and headset head-follow mapping as the teleop node.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import sys

import numpy as np
import zmq

# Vendored layout: .../third_party/holomotion/{holoretarget,deployment/...}
_VENDOR_ROOT = Path(__file__).resolve().parents[2]
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from holoretarget.schema import (
    DEX3_LEFT_HAND_JOINT_NAMES,
    DEX3_RIGHT_HAND_JOINT_NAMES,
    DOF_POS_DIM,
    HAND_DOF_DIM,
    HEAD_DOF_DIM,
    HEAD_SERVO_JOINT_NAMES,
    QPOS_DIM,
    UNITREE_G1_29DOF_NAMES,
)


HEADER_SIZE = 1280
DOF_DIM = DOF_POS_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = PROJECT_ROOT / "holoretarget" / "assets" / "unitree_g1"
MODEL_MJCF_PATHS = {
    "g1": ASSET_DIR / "g1_mocap_29dof.xml",
    "g1_comp": ASSET_DIR / "g1_comp_45dof.xml",
}
DEFAULT_MJCF_PATH = MODEL_MJCF_PATHS["g1"]


def dtype_from_str(dtype_str: str) -> np.dtype:
    mapping = {
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
        "i32": np.dtype("<i4"),
        "i64": np.dtype("<i8"),
        "u8": np.dtype("u1"),
        "bool": np.dtype("?"),
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype in message: {dtype_str}")
    return mapping[dtype_str]


def decode_numpy_message(packet: bytes, topic: bytes) -> Dict[str, np.ndarray]:
    if not packet.startswith(topic):
        raise ValueError("Unexpected ZMQ topic prefix")

    header_start = len(topic)
    header_end = header_start + HEADER_SIZE
    if len(packet) < header_end:
        raise ValueError("Packet too short for fixed-size header")

    header_bytes = packet[header_start:header_end]
    payload = memoryview(packet[header_end:])
    header_json = header_bytes.rstrip(b"\x00").decode("utf-8")
    header = json.loads(header_json)

    result: Dict[str, np.ndarray] = {}
    offset = 0
    for field in header.get("fields", []):
        name = field["name"]
        shape = tuple(field["shape"])
        dtype = dtype_from_str(field["dtype"])
        count = int(np.prod(shape))
        size_bytes = count * dtype.itemsize
        array = np.frombuffer(
            payload[offset : offset + size_bytes], dtype=dtype
        ).reshape(shape).copy()
        result[name] = array
        offset += size_bytes
    return result


class ReferenceReceiver:
    def __init__(self, uri: str, topic: str):
        self.uri = uri
        self.topic = topic.encode("utf-8")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        if hasattr(zmq, "CONFLATE"):
            self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(uri)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def recv_latest(self, timeout_ms: int) -> Dict[str, np.ndarray] | None:
        events = dict(self.poller.poll(timeout=timeout_ms))
        if self.socket not in events:
            return None

        packet = self.socket.recv()
        while True:
            events = dict(self.poller.poll(timeout=0))
            if self.socket not in events:
                break
            packet = self.socket.recv()

        return decode_numpy_message(packet, topic=self.topic)

    def close(self):
        self.socket.close(0)
        self.context.term()


class ReferenceMujocoViewer:
    """Minimal MuJoCo viewer for reference qpos packets.

    The model is a packaged Unitree G1 MJCF used by HoloRetarget: either the
    plain 29-dof mocap model or the g1_comp composite (Dex3 hands + pan/tilt
    head servos). Joint qpos addresses are resolved by name, so any model that
    contains the 29 canonical body joints works; hand/head joints are animated
    only when present in the model.
    """

    def __init__(self, mjcf_path: str | Path, viewer_fps: float, camera_follow: bool):
        import mujoco
        import mujoco.viewer

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).expanduser()))
        if self.model.nq < 7 + DOF_DIM:
            raise ValueError(f"MJCF has nq={self.model.nq}, expected at least {7 + DOF_DIM}")

        self.body_qpos_idx = self._qpos_indices(UNITREE_G1_29DOF_NAMES, required=True)
        hand_names = DEX3_LEFT_HAND_JOINT_NAMES + DEX3_RIGHT_HAND_JOINT_NAMES
        self.hand_qpos_idx = self._qpos_indices(hand_names, required=False)
        self.head_qpos_idx = self._qpos_indices(HEAD_SERVO_JOINT_NAMES, required=False)

        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer_fps = float(viewer_fps)
        self.camera_follow = bool(camera_follow)
        self._last_step_time = time.time()

    def _qpos_indices(self, joint_names, *, required: bool) -> np.ndarray | None:
        indices = []
        for name in joint_names:
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                if required:
                    raise ValueError(f"MJCF is missing required joint '{name}'")
                return None
            indices.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(indices, dtype=np.int64)

    @property
    def has_hands(self) -> bool:
        return self.hand_qpos_idx is not None

    @property
    def has_head(self) -> bool:
        return self.head_qpos_idx is not None

    def step(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        dof_pos: np.ndarray,
        hand_joints: np.ndarray | None = None,
        head_pan_tilt: np.ndarray | None = None,
        *,
        rate_limit: bool = True,
        follow_camera: bool = True,
    ) -> None:
        root_pos = np.asarray(root_pos, dtype=np.float64).reshape(3)
        root_rot = np.asarray(root_rot, dtype=np.float64).reshape(4)
        dof_pos = np.asarray(dof_pos, dtype=np.float64).reshape(DOF_DIM)

        quat_norm = float(np.linalg.norm(root_rot))
        if quat_norm > 1e-8:
            root_rot = root_rot / quat_norm
        else:
            root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot
        self.data.qpos[self.body_qpos_idx] = dof_pos
        if self.hand_qpos_idx is not None and hand_joints is not None:
            self.data.qpos[self.hand_qpos_idx] = np.asarray(
                hand_joints, dtype=np.float64
            ).reshape(HAND_DOF_DIM)
        if self.head_qpos_idx is not None and head_pan_tilt is not None:
            self.data.qpos[self.head_qpos_idx] = np.asarray(
                head_pan_tilt, dtype=np.float64
            ).reshape(HEAD_DOF_DIM)
        self.mujoco.mj_forward(self.model, self.data)

        if follow_camera and hasattr(self.viewer, "cam"):
            self.viewer.cam.lookat[:] = root_pos
        self.viewer.sync()

        if rate_limit and self.viewer_fps > 0.0:
            now = time.time()
            target_dt = 1.0 / self.viewer_fps
            sleep_time = target_dt - (now - self._last_step_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            self._last_step_time = time.time()

    def is_running(self) -> bool:
        return bool(self.viewer.is_running())

    def close(self) -> None:
        self.viewer.close()


def qpos_to_state(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if qpos.size != QPOS_DIM:
        raise ValueError(f"reference_qpos must have {QPOS_DIM} values, got {qpos.size}")
    root_pos = np.asarray(qpos[:3], dtype=np.float32)
    root_rot = np.asarray(qpos[3:7], dtype=np.float32)
    dof_pos = np.asarray(qpos[7:], dtype=np.float32)
    return root_pos, root_rot, dof_pos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize reference qpos ZMQ stream in MuJoCo"
    )
    parser.add_argument("--uri", default="tcp://127.0.0.1:6001", help="Publisher endpoint to connect to")
    parser.add_argument("--topic", default="reference_qpos", help="ZMQ topic to subscribe")
    parser.add_argument("--robot", default="unitree_g1", help="Deprecated; kept for script compatibility")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_MJCF_PATHS),
        default="g1",
        help="Packaged robot model: 'g1' (29 dof) or 'g1_comp' (Dex3 hands + pan/tilt head servos)",
    )
    parser.add_argument(
        "--mjcf",
        default=None,
        help="Explicit MuJoCo XML override; defaults to the packaged MJCF for --model",
    )
    parser.add_argument("--viewer-fps", type=float, default=55.0, help="Viewer refresh rate")
    parser.add_argument("--recv-timeout-ms", type=int, default=10, help="Polling timeout for new packets")
    parser.add_argument("--print-every", type=int, default=100, help="Print stream stats every N received frames")
    parser.add_argument("--no-follow-camera", action="store_true", help="Disable camera follow behavior")
    parser.add_argument("--dry-run", action="store_true", help="Receive and print packets without opening MuJoCo")
    return parser.parse_args()


def main():
    args = parse_args()
    mjcf_path = args.mjcf if args.mjcf else str(MODEL_MJCF_PATHS[args.model])

    receiver = ReferenceReceiver(uri=args.uri, topic=args.topic)
    print(
        f"[INFO] reference viewer connected: uri={args.uri}, "
        f"topic={args.topic}, model={args.model}, mjcf={mjcf_path}"
    )

    viewer = None
    latest_root_pos = np.array([0.0, 0.0, 0.793], dtype=np.float32)
    latest_root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    latest_dof_pos = np.zeros(DOF_DIM, dtype=np.float32)
    latest_hand_joints = np.zeros(HAND_DOF_DIM, dtype=np.float32)
    latest_head_pan_tilt = np.zeros(HEAD_DOF_DIM, dtype=np.float32)
    local_hand_head = None  # lazy fallback mapper for raw controller streams

    if not args.dry_run:
        viewer = ReferenceMujocoViewer(
            mjcf_path=mjcf_path,
            viewer_fps=args.viewer_fps,
            camera_follow=not args.no_follow_camera,
        )
        if viewer.has_hands or viewer.has_head:
            print(
                f"[INFO] composite model: hands={viewer.has_hands}, "
                f"head_servos={viewer.has_head}"
            )
        latest_root_pos = viewer.data.qpos[:3].copy().astype(np.float32)
        latest_root_rot = viewer.data.qpos[3:7].copy().astype(np.float32)
        latest_dof_pos = viewer.data.qpos[viewer.body_qpos_idx].copy().astype(np.float32)

    recv_count = 0
    last_recv_time = None
    recv_freq_log: list[float] = []

    try:
        while True:
            decoded = receiver.recv_latest(timeout_ms=args.recv_timeout_ms)
            if decoded is not None:
                reference_qpos = decoded.get("reference_qpos")
                if reference_qpos is None:
                    print("[WARN] packet does not contain reference_qpos")
                else:
                    reference_qpos = np.asarray(reference_qpos).reshape(-1)
                    latest_root_pos, latest_root_rot, latest_dof_pos = qpos_to_state(
                        reference_qpos
                    )
                    hand_joints = decoded.get("hand_joints")
                    head_pan_tilt = decoded.get("head_pan_tilt")
                    raw_controller = (
                        "left_trigger" in decoded
                        or "right_trigger" in decoded
                        or "head_quat" in decoded
                    )
                    if (hand_joints is None or head_pan_tilt is None) and raw_controller:
                        if local_hand_head is None:
                            from holomotion_teleop_node import HandHeadController

                            local_hand_head = HandHeadController()
                            print(
                                "[INFO] deriving hand/head targets locally from "
                                "raw controller fields"
                            )
                        sample = {}
                        for key in ("left_trigger", "right_trigger"):
                            if key in decoded:
                                sample[key] = float(decoded[key].reshape(-1)[0])
                        for key in ("left_axis", "right_axis", "head_quat"):
                            if key in decoded:
                                sample[key] = decoded[key].reshape(-1)
                        local_hand, local_head = local_hand_head.process(sample)
                        if hand_joints is None:
                            hand_joints = local_hand
                        if head_pan_tilt is None:
                            head_pan_tilt = local_head
                    if hand_joints is not None and hand_joints.size == HAND_DOF_DIM:
                        latest_hand_joints = np.asarray(
                            hand_joints, dtype=np.float32
                        ).reshape(HAND_DOF_DIM)
                    if head_pan_tilt is not None and head_pan_tilt.size == HEAD_DOF_DIM:
                        latest_head_pan_tilt = np.asarray(
                            head_pan_tilt, dtype=np.float32
                        ).reshape(HEAD_DOF_DIM)

                    recv_count += 1
                    now = time.time()
                    if last_recv_time is not None:
                        dt = now - last_recv_time
                        if dt > 0:
                            recv_freq_log.append(1.0 / dt)
                    last_recv_time = now

                    if recv_count % max(1, args.print_every) == 0:
                        frame_index = int(decoded.get("frame_index", np.array([-1], dtype=np.int64))[0])
                        avg_recv_hz = sum(recv_freq_log) / len(recv_freq_log) if recv_freq_log else 0.0
                        extras = ""
                        if hand_joints is not None:
                            extras += (
                                f", hand_close=[{latest_hand_joints[2]:+.2f}"
                                f" {latest_hand_joints[9]:+.2f}]"
                            )
                        if head_pan_tilt is not None:
                            extras += (
                                f", head_pan_tilt="
                                f"{np.array2string(latest_head_pan_tilt, precision=3, suppress_small=True)}"
                            )
                        print(
                            f"[reference] frame={frame_index}, recv_hz={avg_recv_hz:.2f}, "
                            f"root_pos={np.array2string(latest_root_pos, precision=4, suppress_small=True)}, "
                            f"root_rot_wxyz={np.array2string(latest_root_rot, precision=4, suppress_small=True)}"
                            f"{extras}"
                        )
                        recv_freq_log.clear()

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.step(
                    latest_root_pos,
                    latest_root_rot,
                    latest_dof_pos,
                    latest_hand_joints,
                    latest_head_pan_tilt,
                    rate_limit=True,
                    follow_camera=not args.no_follow_camera,
                )
            elif decoded is None:
                time.sleep(max(args.recv_timeout_ms, 1) / 1000.0)
    except KeyboardInterrupt:
        print("\n[INFO] viewer stopped by user")
    finally:
        receiver.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
