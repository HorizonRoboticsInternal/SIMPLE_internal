"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

HoloMotion (v1.4) teleoperation agent.

Pipeline, mirroring the on-robot HoloMotion v1.4 deployment:

    PICO / XRoboToolkit
      -> reference publisher (HoloRetarget; third_party/holomotion)
      -> ZMQ "reference_qpos": reference_qpos[36] + hand_joints[14] + buttons
      -> HoloMotionPicoAgent
           11-frame reference queue -> 604-D actor observation
           -> HoloMotion ONNX (KV cache) -> 29 joint targets
           -> ActionCmd("holomotion") -> G1Sonic PD torques (+ Dex3 hand PD)

The observation builder is a NumPy transliteration of
``holomotion/src/motion_tracking/actor_observation.py`` and
``reference_observation.py`` (the Warp kernels used by the Orin deployment),
so SIMPLE needs neither the ``holomotion`` package nor Warp. Only the ZMQ
reference stream and the exported ONNX policy are required; the contracts live
in :mod:`simple.teleop.holomotion`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import zmq

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.pico.streaming import FrameBuffer, StreamingThread
from simple.teleop.pico.tcp_server import TCPControlServer
from simple.teleop.pico.tcp_video_sender import TCPVideoSender

from .sonic_wbc_agent import SonicWbcAgent

# ---------------------------------------------------------------------------
# Contracts and wire format live in simple.teleop.holomotion (vendored)
# ---------------------------------------------------------------------------

from simple.teleop.holomotion import (  # noqa: E402
    BUTTON_FIELDS,
    DEFAULT_REFERENCE_URI,
    DEX3_NATURAL_TO_MJCF,
    HAND_DOF_DIM,
    HEADER_SIZE,
    HOLORETARGET_DOF_NAMES,
    MOTION_ACTOR_CURRENT_DIM,
    MOTION_ACTOR_FUTURE_DIM,
    REF_DOF_DIM,
    REF_QPOS_DIM,
    decode_numpy_message,
    resolve_motion_onnx,
)


class ReferenceReceiver:
    """Latest-only ZMQ subscriber for the HoloMotion reference stream."""

    def __init__(self, uri: str, topic: str = "reference_qpos"):
        self.uri = uri
        self.topic = topic.encode("utf-8")
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        if hasattr(zmq, "CONFLATE"):
            self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.connect(uri)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

    def recv_latest(self, timeout_ms: int = 0) -> Optional[Dict[str, np.ndarray]]:
        if self._sock not in dict(self._poller.poll(timeout=timeout_ms)):
            return None
        packet = self._sock.recv()
        while self._sock in dict(self._poller.poll(timeout=0)):
            packet = self._sock.recv()
        return decode_numpy_message(packet, self.topic)

    def close(self) -> None:
        self._sock.close(0)
        self._ctx.term()


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz), matching the Warp kernels exactly
# ---------------------------------------------------------------------------

def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1.0e-9)
    return -q if q[0] < 0.0 else q


def _quat_mul_wxyz(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, xyz0 = q0[0], q0[1:]
    w1, xyz1 = q1[0], q1[1:]
    xyz = w0 * xyz1 + w1 * xyz0 + np.cross(xyz0, xyz1)
    return np.array([w0 * w1 - np.dot(xyz0, xyz1), *xyz], dtype=np.float32)


def _quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)


def _quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = -q[1:4]
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def _projected_gravity(q: np.ndarray) -> np.ndarray:
    return np.array([
        2.0 * (-q[3] * q[1] + q[0] * q[2]),
        -2.0 * (q[3] * q[2] + q[0] * q[1]),
        1.0 - 2.0 * (q[0] * q[0] + q[3] * q[3]),
    ], dtype=np.float32)


def _rot6d_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z), 2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
    ], dtype=np.float32)


def _angular_velocity_wxyz(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    rw = q1[0] * q0[0] + q1[1] * q0[1] + q1[2] * q0[2] + q1[3] * q0[3]
    rx = -q1[0] * q0[1] + q1[1] * q0[0] - q1[2] * q0[3] + q1[3] * q0[2]
    ry = -q1[0] * q0[2] + q1[1] * q0[3] + q1[2] * q0[0] - q1[3] * q0[1]
    rz = -q1[0] * q0[3] - q1[1] * q0[2] + q1[2] * q0[1] + q1[3] * q0[0]
    if rw < 0.0:
        rw, rx, ry, rz = -rw, -rx, -ry, -rz
    inv = 1.0 / max(float(np.sqrt(rw * rw + rx * rx + ry * ry + rz * rz)), 1.0e-9)
    rw, rx, ry, rz = rw * inv, rx * inv, ry * inv, rz * inv
    mag = float(np.sqrt(rx * rx + ry * ry + rz * rz))
    scale = 2.0 * float(np.arctan2(mag, rw)) / mag if mag > 1.0e-7 else 2.0
    return np.array([rx, ry, rz], dtype=np.float32) * (scale / dt)


# ---------------------------------------------------------------------------
# Reference queue + kinematics (port of VrReference / GpuReferenceQueue)
# ---------------------------------------------------------------------------

class ReferenceQueue:
    """Fixed-rate queue of reference frames: [past, current, fut x n, tail].

    The policy tracks the *oldest* frame in the window, so the robot lags the
    operator by ``n_fut_frames`` ticks (~200 ms at 50 Hz) in exchange for a
    real future horizon. Sample times are uniform at ``fps``, as in the Orin
    GPU queue.
    """

    def __init__(self, n_fut_frames: int, fps: float = 50.0):
        self.n_fut = int(n_fut_frames)
        self.frames = self.n_fut + 3
        self.fps = float(fps)
        self.sample_time = np.arange(self.frames, dtype=np.float32) / self.fps
        self._buf = np.zeros((self.frames, REF_QPOS_DIM), dtype=np.float32)
        self.seen_frames = 0

    def reset(self) -> None:
        self._buf.fill(0.0)
        self.seen_frames = 0

    def store(self, qpos: np.ndarray) -> None:
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = qpos
        self.seen_frames += 1

    @property
    def ready(self) -> bool:
        return self.seen_frames >= self.frames

    def sequence(self) -> np.ndarray:
        return self._buf

    def current_qpos(self) -> np.ndarray:
        return self._buf[1]


def derive_reference_kinematics(qpos: np.ndarray, sample_time: np.ndarray):
    """Central-difference kinematics over a [T,36] window (NumPy port)."""
    frames = qpos.shape[0]
    dof_vel = np.zeros((frames, REF_DOF_DIM), dtype=np.float32)
    linvel_local = np.zeros((frames, 3), dtype=np.float32)
    angvel_local = np.zeros((frames, 3), dtype=np.float32)
    gravity = np.zeros((frames, 3), dtype=np.float32)
    for f in range(frames):
        lo = max(f - 1, 0)
        hi = min(f + 1, frames - 1)
        dt = max(float(sample_time[hi] - sample_time[lo]), 1.0e-6)
        lin_world = (qpos[hi, 0:3] - qpos[lo, 0:3]) / dt
        quat = qpos[f, 3:7]
        ang_world = _angular_velocity_wxyz(qpos[lo, 3:7], qpos[hi, 3:7], dt)
        linvel_local[f] = _quat_rotate_inverse_wxyz(quat, lin_world)
        angvel_local[f] = _quat_rotate_inverse_wxyz(quat, ang_world)
        gravity[f] = _quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        dof_vel[f] = (qpos[hi, 7:] - qpos[lo, 7:]) / dt
    return dof_vel, linvel_local, angvel_local, gravity


def build_motion_actor_observation(
    qpos: np.ndarray,                 # [T,36] window, oldest -> newest
    kin: tuple,                       # from derive_reference_kinematics(qpos)
    robot_root_quat_wxyz: np.ndarray,
    robot_root_angvel_local: np.ndarray,
    robot_dof_pos_onnx: np.ndarray,
    robot_dof_vel_onnx: np.ndarray,
    last_action_onnx: np.ndarray,
    default_dof_pos_onnx: np.ndarray,
    ref_to_onnx: np.ndarray,
    yaw_alignment_wxyz: Optional[np.ndarray],
    *,
    current_index: int = 1,
    n_fut: int = 10,
) -> np.ndarray:
    """NumPy port of ``_motion_actor_observation_kernel``."""
    dof_vel, linvel_local, angvel_local, gravity = kin
    cur = current_index
    fut = np.arange(cur + 1, cur + 1 + n_fut)

    robot_quat = _normalize_quat_wxyz(robot_root_quat_wxyz)
    ref_quat_cur = _normalize_quat_wxyz(qpos[cur, 3:7])

    def aligned(q: np.ndarray) -> np.ndarray:
        if yaw_alignment_wxyz is None:
            return q
        return _normalize_quat_wxyz(_quat_mul_wxyz(_normalize_quat_wxyz(yaw_alignment_wxyz), q))

    yaw_err = _yaw_from_quat_wxyz(aligned(ref_quat_cur)) - _yaw_from_quat_wxyz(robot_quat)

    current = np.concatenate([
        gravity[cur],                                        # 0:3
        linvel_local[cur],                                   # 3:6
        angvel_local[cur],                                   # 6:9
        qpos[cur, 7 + ref_to_onnx],                          # 9:38
        qpos[cur, 2:3],                                      # 38
        np.array([np.sin(yaw_err), np.cos(yaw_err)], dtype=np.float32),  # 39:41
        _projected_gravity(robot_quat),                      # 41:44
        np.asarray(robot_root_angvel_local, dtype=np.float32),          # 44:47
        robot_dof_pos_onnx - default_dof_pos_onnx,           # 47:76
        robot_dof_vel_onnx,                                  # 76:105
        last_action_onnx,                                    # 105:134
    ]).astype(np.float32)
    assert current.shape[0] == MOTION_ACTOR_CURRENT_DIM

    fut_quats = [_normalize_quat_wxyz(qpos[f, 3:7]) for f in fut]
    yaw_cur = _yaw_from_quat_wxyz(ref_quat_cur)
    yaw_delta = np.array([[np.sin(_yaw_from_quat_wxyz(q) - yaw_cur), np.cos(_yaw_from_quat_wxyz(q) - yaw_cur)]
                          for q in fut_quats], dtype=np.float32)
    rot6d = np.stack([
        _rot6d_wxyz(_normalize_quat_wxyz(_quat_mul_wxyz(_quat_inv_wxyz(robot_quat), aligned(q))))
        for q in fut_quats
    ])

    future = np.concatenate([
        qpos[fut][:, 7 + ref_to_onnx].reshape(-1),  # dof_pos_fut  (n_fut*29)
        qpos[fut, 2],                                # root_height_fut (n_fut)
        gravity[fut].reshape(-1),                    # gravity_projection_fut (n_fut*3)
        linvel_local[fut].reshape(-1),               # base_linvel_fut
        angvel_local[fut].reshape(-1),               # base_angvel_fut
        yaw_delta.reshape(-1),                       # future_yaw_delta_sin_cos (n_fut*2)
        rot6d.reshape(-1),                           # future_root_ori_robot_frame_6d (n_fut*6)
    ]).astype(np.float32)
    assert future.shape[0] == n_fut * MOTION_ACTOR_FUTURE_DIM
    return np.concatenate([current, future])


# ---------------------------------------------------------------------------
# ONNX policy (KV-cache protocol of the v1.4 deployment)
# ---------------------------------------------------------------------------

class HoloMotionOnnxPolicy:
    def __init__(self, onnx_path: str, *, use_gpu: bool = True, rope_reset_margin: int = 64):
        import onnxruntime as ort

        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"HoloMotion ONNX not found: '{onnx_path}'. Pass onnx_path= or set HOLOMOTION_MOTION_ONNX."
            )
        self.onnx_path = str(onnx_path)
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        self.provider = self.session.get_providers()[0]

        meta = self.session.get_modelmeta().custom_metadata_map
        parse = lambda s: np.array([float(v) for v in s.split(",") if v != ""], dtype=np.float32)
        self.joint_names = [n for n in meta["joint_names"].split(",") if n]
        self.action_scale = parse(meta["action_scale"])
        self.kps = parse(meta["joint_stiffness"])
        self.kds = parse(meta["joint_damping"])
        self.default_dof_pos = parse(meta["default_joint_pos"])
        self.rope_max_seq_len = int(meta.get("rope_max_seq_len", "0") or 0)
        self.rope_reset_margin = int(rope_reset_margin)
        self.num_actions = len(self.joint_names)
        assert self.num_actions == REF_DOF_DIM, f"expected 29 actions, got {self.num_actions}"

        self.obs_name, self.kv_in, self.step_in = None, None, None
        self.kv_shape, self.kv_dtype = None, np.float32
        for node in self.session.get_inputs():
            if "obs" in node.name:
                self.obs_name = node.name
                self.obs_dim = int(node.shape[-1])
            elif "past_key_values" in node.name:
                self.kv_in = node.name
                self.kv_shape = [d if isinstance(d, int) else 1 for d in node.shape]
                self.kv_dtype = np.float16 if "float16" in node.type else np.float32
            elif "step_idx" in node.name or node.name == "current_pos":
                self.step_in = node.name
        self.action_out, self.kv_out = None, None
        for node in self.session.get_outputs():
            if "present_key_values" in node.name:
                self.kv_out = node.name
            elif "actions" in node.name:
                self.action_out = node.name
        if self.action_out is None:
            self.action_out = self.session.get_outputs()[0].name
        self.use_kv = self.kv_in is not None and self.kv_shape is not None
        self.reset()

    def reset(self) -> None:
        self.step_idx = 0
        self.kv_cache = np.zeros(self.kv_shape, dtype=self.kv_dtype) if self.use_kv else None

    def _maybe_reset_rope_window(self) -> None:
        if not self.use_kv or self.step_in is None or self.rope_max_seq_len <= 0:
            return
        reset_at = self.rope_max_seq_len - self.rope_reset_margin
        if reset_at <= 0:
            reset_at = self.rope_max_seq_len
        if self.step_idx >= reset_at:
            self.kv_cache.fill(0)
            self.step_idx = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        feed = {self.obs_name: np.asarray(obs, dtype=np.float32).reshape(1, -1)}
        outputs = [self.action_out]
        if self.use_kv:
            self._maybe_reset_rope_window()
            feed[self.kv_in] = self.kv_cache
            if self.step_in is not None:
                feed[self.step_in] = np.array([self.step_idx], dtype=np.int64)
            if self.kv_out is not None:
                outputs.append(self.kv_out)
        result = self.session.run(outputs, feed)
        if self.use_kv and len(result) > 1:
            self.kv_cache = result[1]
        self.step_idx += 1
        return np.asarray(result[0], dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class HoloMotionPicoAgent(SonicWbcAgent):
    """Teleoperation agent driving G1Sonic with the HoloMotion v1.4 policy.

    The reference publisher (``third_party/holomotion``, run in its own
    Newton/Warp environment) owns PICO tracking, HoloRetarget and the
    controller buttons and publishes them over ZMQ. This agent subscribes,
    maintains the deployment-equivalent reference queue, builds the actor
    observation, runs the exported ONNX and emits joint targets.

    PICO conventions (same as the decoupled-WBC teleop):
      right axis click          -> drop robot / release elastic band
      left grip + right grip    -> reset environment
      left/right trigger        -> close Dex3 hands (mapped by the publisher)
    """

    def __init__(
        self,
        robot: G1Sonic,
        *,
        onnx_path: str = "",
        reference_uri: str = DEFAULT_REFERENCE_URI,
        n_fut_frames: int = 10,
        use_gpu: bool = True,
        reference_timeout_sec: float = 0.5,
        settle_sec: float = 1.0,
        enable_pico_stream: bool = True,
    ) -> None:
        super().__init__(robot)
        self.episodes_saved = 0
        self.num_episodes = 100
        self.image_publish_process = None
        self.sim_dt = float(self.robot.sonic_config["SIMULATE_DT"])
        self._control_dt = 4 * self.sim_dt
        self.fps = 1.0 / self._control_dt

        # Elastic band / reset (same conventions as PicoDecoupledAgent)
        self._dropping = False
        self._drop_rate = 0.15
        self._reset_requested = False
        self._drop_btn_last = False
        self._reset_btn_last = False

        # Policy + reference
        self.policy = HoloMotionOnnxPolicy(resolve_motion_onnx(onnx_path), use_gpu=use_gpu)
        self.n_fut = int(n_fut_frames)
        assert self.policy.obs_dim == MOTION_ACTOR_CURRENT_DIM + self.n_fut * MOTION_ACTOR_FUTURE_DIM, (
            f"ONNX obs dim {self.policy.obs_dim} != {MOTION_ACTOR_CURRENT_DIM}+{self.n_fut}*{MOTION_ACTOR_FUTURE_DIM}"
        )
        self.queue = ReferenceQueue(self.n_fut, fps=self.fps)
        self.receiver = ReferenceReceiver(reference_uri)
        self.reference_uri = reference_uri
        self.reference_timeout_sec = float(reference_timeout_sec)
        self.settle_sec = float(settle_sec)

        # Joint index maps: HoloRetarget/MJCF order <-> ONNX order
        self.ref_to_onnx = np.array([HOLORETARGET_DOF_NAMES.index(n) for n in self.policy.joint_names], dtype=np.int64)
        self.onnx_to_ref = np.empty(REF_DOF_DIM, dtype=np.int64)
        self.onnx_to_ref[self.ref_to_onnx] = np.arange(REF_DOF_DIM)
        assert list(self.robot.joint_names[:REF_DOF_DIM]) == list(HOLORETARGET_DOF_NAMES)
        self.kp_mjcf = self.policy.kps[self.onnx_to_ref]
        self.kd_mjcf = self.policy.kds[self.onnx_to_ref]
        self.default_mjcf = self.policy.default_dof_pos[self.onnx_to_ref]

        # Runtime state
        self._last_packet: Optional[Dict[str, np.ndarray]] = None
        self._last_packet_time: float = -1.0
        self._last_frame_index: int = -1
        self._last_action_onnx = np.zeros(REF_DOF_DIM, dtype=np.float32)
        self._target_mjcf = self.default_mjcf.copy()
        self._left_hand_mjcf = np.zeros(7, dtype=np.float32)
        self._right_hand_mjcf = np.zeros(7, dtype=np.float32)
        self._hand_joints_natural = np.zeros(HAND_DOF_DIM, dtype=np.float32)
        self._yaw_alignment: Optional[np.ndarray] = None
        self.tracking_active = False
        self._reset_time = time.monotonic()
        self._last_obs: Optional[np.ndarray] = None
        self._stats_next = time.monotonic() + 5.0
        self._stats_frames = 0

        self._streaming: Optional[StreamingThread] = None
        self._frame_buffer = FrameBuffer()
        if enable_pico_stream:
            self._init_pico_streamer()

        print(
            f"[HoloMotion] policy={os.path.basename(self.policy.onnx_path)} provider={self.policy.provider} "
            f"obs_dim={self.policy.obs_dim} n_fut={self.n_fut} kv={self.policy.use_kv} "
            f"reference={reference_uri}"
        )

    # ------------------------------------------------------------------
    # Pico VR camera streaming (identical to PicoDecoupledAgent)
    # ------------------------------------------------------------------

    def _init_pico_streamer(self):
        tcp_server = TCPControlServer("0.0.0.0:13579")

        def on_open_camera(camera_req):
            print(f"[HoloMotion] OPEN_CAMERA: {camera_req}")
            if self._streaming and self._streaming.is_running():
                return
            fps = camera_req.get("fps") or 60
            width = camera_req.get("width") or 2560
            height = camera_req.get("height") or 720
            bitrate = camera_req.get("bitrate") or 4_000_000
            hevc = bool(camera_req.get("enableMvHevc"))
            ip, port = camera_req.get("ip"), camera_req.get("port")
            if not ip or not port:
                print("[HoloMotion] OPEN_CAMERA missing ip/port, cannot stream")
                return
            try:
                sender = TCPVideoSender(ip=ip, port=port, width=width, height=height, fps=fps, bitrate=bitrate, hevc=hevc)
            except ConnectionRefusedError:
                print(f"[HoloMotion] Connection refused to {ip}:{port}")
                return
            self._streaming = StreamingThread(
                frame_buffer=self._frame_buffer, fps=fps, publishers=[sender],
                on_ended=lambda: tcp_server.close_client(),
            )
            self._streaming.start()

        def on_close_camera():
            print("[HoloMotion] CLOSE_CAMERA received")
            if self._streaming:
                self._streaming.stop()
                self._streaming = None
            tcp_server.close_client()

        tcp_server.on_open_camera = on_open_camera
        tcp_server.on_close_camera = on_close_camera
        tcp_server.start()
        self._tcp_server = tcp_server

    def update_render_caches(self, observation: dict):
        if self._streaming and self._streaming.is_running():
            left = observation.get("head_stereo_left")
            right = observation.get("head_stereo_right")
            if left is None or right is None:
                return observation
            left_bgr = np.ascontiguousarray(left[..., ::-1])
            right_bgr = np.ascontiguousarray(right[..., ::-1])
            text = f"{self.episodes_saved}/{self.num_episodes}"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x, y = left_bgr.shape[1] - tw - 100, th + 10
            cv2.putText(left_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
            cv2.putText(right_bgr, text, (x, y), font, scale, (0, 255, 0), thickness)
            self._frame_buffer.put(np.concatenate([left_bgr, right_bgr], axis=1))
        return observation

    # ------------------------------------------------------------------
    # Reference stream + buttons
    # ------------------------------------------------------------------

    def _poll_reference(self) -> None:
        packet = self.receiver.recv_latest(timeout_ms=0)
        if packet is not None and "reference_qpos" in packet:
            self._last_packet = packet
            self._last_packet_time = time.monotonic()
            self._last_frame_index = int(packet.get("frame_index", np.array([-1]))[0])
            self._stats_frames += 1
            hand = packet.get("hand_joints")
            if hand is not None and hand.size == HAND_DOF_DIM:
                self._hand_joints_natural = hand.astype(np.float32).reshape(HAND_DOF_DIM)
                self._left_hand_mjcf = self._hand_joints_natural[:7][DEX3_NATURAL_TO_MJCF]
                self._right_hand_mjcf = self._hand_joints_natural[7:][DEX3_NATURAL_TO_MJCF]
        # One store per control tick from the newest (non-stale) sample,
        # exactly like the Orin local-retarget source.
        if self._last_packet is not None and self.reference_age < self.reference_timeout_sec:
            self.queue.store(self._last_packet["reference_qpos"].astype(np.float32).reshape(REF_QPOS_DIM))

        now = time.monotonic()
        if now >= self._stats_next:
            print(f"[HoloMotion] reference {self._stats_frames / 5.0:.1f} Hz, frame={self._last_frame_index}, "
                  f"queue_ready={self.queue.ready}, tracking={self.tracking_active}")
            self._stats_frames = 0
            self._stats_next = now + 5.0

    @property
    def reference_age(self) -> float:
        if self._last_packet_time < 0:
            return float("inf")
        return time.monotonic() - self._last_packet_time

    def button(self, name: str) -> float:
        """Latest value of a controller field published by the teleop node."""
        if self._last_packet is None:
            return 0.0
        if name in self._last_packet:
            return float(np.asarray(self._last_packet[name]).reshape(-1)[0])
        packed = self._last_packet.get("buttons")
        if packed is not None and name in BUTTON_FIELDS:
            packed = np.asarray(packed).reshape(-1)
            idx = BUTTON_FIELDS.index(name)
            return float(packed[idx]) if idx < packed.size else 0.0
        return 0.0

    def _poll_pico_buttons(self) -> None:
        drop_btn = self.button("right_axis_click") > 0.5
        if drop_btn and not self._drop_btn_last:
            if self.robot.elastic_band and self.robot.elastic_band.enable and not self._dropping:
                self._dropping = True
                print("[HoloMotion] Controlled drop started (right stick click)")
        self._drop_btn_last = drop_btn

        reset_btn = self.button("left_grip") > 0.5 and self.button("right_grip") > 0.5
        if reset_btn and not self._reset_btn_last:
            self._reset_requested = True
            print("[HoloMotion] Environment reset requested (L-grip + R-grip)")
        self._reset_btn_last = reset_btn

    @property
    def reset_requested(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    # ------------------------------------------------------------------
    # Policy step
    # ------------------------------------------------------------------

    def _band_active(self) -> bool:
        return bool(self.robot.elastic_band and self.robot.elastic_band.enable and self.robot.use_floating_root_link)

    def _settled(self) -> bool:
        return self.robot.stabilized or (time.monotonic() - self._reset_time) >= self.settle_sec

    def _begin_tracking(self, proprio: dict) -> None:
        ref_quat = _normalize_quat_wxyz(self.queue.current_qpos()[3:7])
        robot_quat = _normalize_quat_wxyz(proprio["floating_base_pose"][3:7])
        yaw_offset = _yaw_from_quat_wxyz(robot_quat) - _yaw_from_quat_wxyz(ref_quat)
        self._yaw_alignment = _yaw_quat_wxyz(yaw_offset)
        self.policy.reset()
        self._last_action_onnx[:] = 0.0
        self.tracking_active = True
        print(f"[HoloMotion] Motion tracking started (yaw offset {np.degrees(yaw_offset):+.1f} deg)")

    def _policy_step(self, proprio: dict) -> None:
        qpos = self.queue.sequence()
        kin = derive_reference_kinematics(qpos, self.queue.sample_time)
        body_q = np.asarray(proprio["body_q"], dtype=np.float32)
        body_dq = np.asarray(proprio["body_dq"], dtype=np.float32)
        obs = build_motion_actor_observation(
            qpos, kin,
            robot_root_quat_wxyz=np.asarray(proprio["floating_base_pose"][3:7], dtype=np.float32),
            robot_root_angvel_local=np.asarray(proprio["floating_base_vel"][3:6], dtype=np.float32),
            robot_dof_pos_onnx=body_q[self.ref_to_onnx],
            robot_dof_vel_onnx=body_dq[self.ref_to_onnx],
            last_action_onnx=self._last_action_onnx,
            default_dof_pos_onnx=self.policy.default_dof_pos,
            ref_to_onnx=self.ref_to_onnx,
            yaw_alignment_wxyz=self._yaw_alignment,
            current_index=1,
            n_fut=self.n_fut,
        )
        action = self.policy(obs)
        self._last_obs = obs
        self._last_action_onnx = action
        target_onnx = self.policy.default_dof_pos + self.policy.action_scale * action
        self._target_mjcf = target_onnx[self.onnx_to_ref].astype(np.float32)

    def get_action(self, observation, instruction=None, **kwargs) -> ActionCmd:
        proprio = kwargs["privileged_info"]["proprio"]
        self._poll_reference()
        self._poll_pico_buttons()

        band = self._band_active()
        if self._dropping and band:
            self.robot.elastic_band.length -= self._drop_rate * self._control_dt
            if self.robot.elastic_band.length <= -0.25 and abs(self.robot.pelvis_vz) < 0.05:
                self.robot.elastic_band.enable = False
                self._dropping = False
                band = False
                print(f"[HoloMotion] Robot landed (pelvis Z={self.robot.pelvis_z:.3f} m)")

        can_track = self.queue.ready and self._settled() and not band
        if can_track and not self.tracking_active:
            self._begin_tracking(proprio)
        if self.tracking_active and self.reference_age >= self.reference_timeout_sec:
            # Stale stream: hold the last target (like the Orin stale gate).
            pass
        elif self.tracking_active:
            self._policy_step(proprio)
        else:
            self._target_mjcf = self.default_mjcf.copy()

        return ActionCmd(
            "holomotion",
            target_q=self._target_mjcf.copy(),            # (29,) MJCF/HoloRetarget order
            left_hand_q=self._left_hand_mjcf.copy(),      # (7,) MJCF order
            right_hand_q=self._right_hand_mjcf.copy(),    # (7,) MJCF order
            kp=self.kp_mjcf,
            kd=self.kd_mjcf,
            apply_elastic_band=band,
            dropping=self._dropping,
            # extras for recording
            tracking=self.tracking_active,
            reference_qpos=self.queue.current_qpos().copy(),
            reference_window=self.queue.sequence()[1:2 + self.n_fut].copy(),  # current + n_fut
            hand_joints=self._hand_joints_natural.copy(),  # natural order, as published
            raw_action=self._last_action_onnx.copy(),      # (29,) ONNX order
            policy_obs=None if self._last_obs is None else self._last_obs.copy(),
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_policy(self) -> None:
        self.queue.reset()
        self.policy.reset()
        self._last_action_onnx[:] = 0.0
        self._target_mjcf = self.default_mjcf.copy()
        self._yaw_alignment = None
        self.tracking_active = False
        self._dropping = False
        self._last_obs = None
        self._reset_time = time.monotonic()

    def publish_low_state(self, proprio):
        pass

    def close(self) -> None:
        try:
            self.receiver.close()
        except Exception:
            pass
        if self._streaming:
            self._streaming.stop()


__all__ = [
    "HoloMotionPicoAgent",
    "HoloMotionOnnxPolicy",
    "ReferenceQueue",
    "build_motion_actor_observation",
    "derive_reference_kinematics",
    "HOLORETARGET_DOF_NAMES",
    "DEX3_NATURAL_TO_MJCF",
]
