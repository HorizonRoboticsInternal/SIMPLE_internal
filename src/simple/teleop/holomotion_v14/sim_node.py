"""
The robot's HoloMotion v1.4 policy node, driven from the sim loop.

``build_sim_policy_node()`` returns an instance of the vendored ``HoloMotionPolicyNode`` (unchanged: model loading,
observation building, velocity/motion modes, VR reference queue, stale-reference freeze, action EMA, ONNX metadata
gains) with only the ROS plumbing replaced. Each control tick the caller hands it what the topics would carry:

    node.feed_lowstate(q, dq, quat_wxyz, gyro)     # /lowstate          (29 joints in the robot's motor order)
    node.feed_pico(sample)                         # /pico_bridge/vr_state (controller buttons and sticks)
    node.feed_reference(latest_obs, frame, stamp)  # MotionReference    (the 65-value latest_obs frame)
    node.run()                                     # the 50 Hz policy timer
    node.action_target, node.action_kps, node.action_kds   # /humanoid/action, /humanoid/kps, /humanoid/kds
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import numpy as np

from . import ROBOT_CONFIG, VENDOR_DIR, ensure_vendor_path
from . import ros_stubs


@dataclass
class _Motor:
    q: float = 0.0
    dq: float = 0.0
    tau_est: float = 0.0


@dataclass
class _Imu:
    quaternion: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    gyroscope: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accelerometer: list = field(default_factory=lambda: [0.0, 0.0, 9.81])
    rpy: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class SimLowState:
    """The fields of unitree_hg/LowState the policy reads."""

    imu_state: _Imu = field(default_factory=_Imu)
    motor_state: list = field(default_factory=lambda: [_Motor() for _ in range(35)])
    wireless_remote: bytes = bytes(40)


class _Controller:
    def __init__(self, s: dict, side: str) -> None:
        p = side[0]                                   # "l" / "r"
        self.status = int(s.get(f"{p}_active", 1))
        self.a_button = bool(s.get("A", 0)) if side == "right" else False
        self.b_button = bool(s.get("B", 0)) if side == "right" else False
        self.x_button = bool(s.get("X", 0)) if side == "left" else False
        self.y_button = bool(s.get("Y", 0)) if side == "left" else False
        axis = s.get(f"{side}_axis", (0.0, 0.0))
        self.axis_x, self.axis_y = float(axis[0]), float(axis[1])
        self.trigger = float(s.get(f"{side}_trigger", 0.0))
        self.grip = float(s.get(f"{side}_grip", 0.0))
        self.axis_click = bool(s.get(f"{'L3' if side == 'left' else 'R3'}", 0))
        self.menu_button = bool(s.get(f"{side}_menu", 0))


class _VRState:
    """robo_orchard_pico_msg_ros2/VRState, controllers only (the body goes through the reference publisher)."""

    def __init__(self, sample: dict) -> None:
        self.left_controller = _Controller(sample, "left")
        self.right_controller = _Controller(sample, "right")


def _read_meta_via_session(sessions: dict):
    """An ``onnx`` stand-in whose load() returns the metadata of an already-open ORT session.

    The vendored read_onnx_metadata() calls onnx.load() on the model file just to read metadata_props; for the 1.6 GB
    motion model that is a second full parse, and SIMPLE's venv has no ``onnx``. ORT exposes the same
    metadata_props as custom_metadata_map, so the node's own parsing code runs unchanged on them."""

    def load(path, *args, **kwargs):
        sess = sessions.get(str(path))
        if sess is None:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        props = [types.SimpleNamespace(key=k, value=v) for k, v in sess.get_modelmeta().custom_metadata_map.items()]
        return types.SimpleNamespace(metadata_props=props)

    return types.SimpleNamespace(load=load, __name__="onnx")


def build_sim_policy_node(*, quiet: bool = False, params: dict | None = None):
    """Construct the vendored policy node for use inside SIMPLE (loads both ONNX models; takes a few seconds)."""
    ensure_vendor_path()
    ros_stubs.install(share_dir=str(VENDOR_DIR))
    from humanoid_policy import policy_node_29dof as P

    overrides = dict(
        config_path=str(ROBOT_CONFIG),
        policy_input_source="pico",           # buttons/sticks from the PICO controllers, as in v1.4 collection
        inference_backend="onnx",             # the robot uses TensorRT on the Orin; same ONNX graphs here
        enable_teleop_reference=True,
        max_data_age=0.6,
        reference_delay_frames=0,
        release_motion_mode=False,
        policy_output_dry_run=False,
        policy_io_stats_enabled=False,
        timing_debug_enabled=False,
    )
    overrides.update(params or {})
    ros_stubs.Node.parameter_overrides = overrides
    ros_stubs.Node.quiet_logs = quiet

    class SimPolicyNode(P.HoloMotionPolicyNode):
        def _setup_timers(self) -> None:                      # the sim loop calls run() itself
            pass

        def _apply_onnx_metadata(self) -> None:
            sessions = {str(self.velocity_onnx_path): self.velocity_policy_session,
                        str(self.motion_onnx_path): self.motion_policy_session}
            real = sys.modules.get("onnx")
            try:
                import onnx  # noqa: F401  a real onnx package is used as is
                have_real = True
            except ImportError:
                have_real = False
            if not have_real:
                sys.modules["onnx"] = _read_meta_via_session(sessions)
            try:
                super()._apply_onnx_metadata()
            finally:
                if not have_real:
                    if real is None:
                        sys.modules.pop("onnx", None)
                    else:
                        sys.modules["onnx"] = real

        # ---- inputs, one call per sim control tick -------------------------------------------------------------
        def feed_lowstate(self, q_real, dq_real, quat_wxyz, gyro) -> None:
            ls = self._sim_lowstate
            ls.imu_state.quaternion = [float(v) for v in quat_wxyz]
            ls.imu_state.gyroscope = [float(v) for v in gyro]
            for i in range(self.num_actions):
                ls.motor_state[i].q = float(q_real[i])
                ls.motor_state[i].dq = float(dq_real[i])
            self._low_state_callback(ls)

        def feed_pico(self, sample: dict | None) -> None:
            if sample:
                self._pico_vr_state_callback(_VRState(sample))

        def feed_reference(self, latest_obs, frame_index: int, stamp_ns: int = 0) -> None:
            obs = np.asarray(latest_obs, dtype=np.float32).reshape(-1)
            n = self.num_actions
            msg = types.SimpleNamespace(
                dof_pos=obs[:n], dof_vel=obs[n:2 * n], root_pos=obs[58:61], root_rot_wxyz=obs[61:65],
                frame_index=int(frame_index),
                header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=int(stamp_ns // 1_000_000_000),
                                                                       nanosec=int(stamp_ns % 1_000_000_000))))
            self._motion_reference_callback(msg)

        def set_robot_state(self, state: str) -> None:        # /robot_state from the main (C++) node
            self._robot_state_callback(types.SimpleNamespace(data=state))

        # ---- outputs: what /humanoid/action, /humanoid/kps and /humanoid/kds carry ------------------------------
        @property
        def action_target(self):
            t = self.runtime.state.target_dof_pos_real
            return None if t is None else np.asarray(t, dtype=np.float32).copy()

        @property
        def action_kps(self) -> np.ndarray:
            last = self.kps_pub.last
            return np.asarray(last.data if last is not None else self.velocity_kps_real, dtype=np.float32)

        @property
        def action_kds(self) -> np.ndarray:
            last = self.kds_pub.last
            return np.asarray(last.data if last is not None else self.velocity_kds_real, dtype=np.float32)

    node = SimPolicyNode()
    node._sim_lowstate = SimLowState()
    node.setup()
    node._setup_completed = True
    return node


def robot_yaml() -> dict:
    """The vendored robot config (dof orders, default angles, move-to-default kp/kd)."""
    import yaml
    with open(ROBOT_CONFIG) as f:
        return yaml.safe_load(f)
