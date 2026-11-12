"""Runtime state machine and policy loop for the 29DOF policy node."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from humanoid_policy.v14_control_safety import arm_pose_is_safe_for_walking


@dataclass(frozen=True)
class PolicyInput:
    """Unified policy-control input from Unitree wireless remote or Pico."""

    enable_velocity: bool = False
    switch_to_motion: bool = False
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


def _default_button_states() -> dict[str, int]:
    return {
        "enable_velocity": 0,
        "switch_to_motion": 0,
    }


@dataclass
class PolicyRuntimeState:
    """Mutable policy runtime state kept outside the ROS node."""

    num_actions: int = 29
    policy_enabled: bool = False
    robot_state_ready: bool = False
    current_policy_mode: str = "velocity"
    motion_uses_vr_reference: bool = False
    motion_frame_idx: int = 0
    motion_step_idx: int = 0
    motion_in_progress: bool = False
    motion_reference_frozen: bool = False
    current_motion_clip_index: int = 0
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    target_dof_pos_real: np.ndarray | None = None
    actions_onnx: np.ndarray = field(init=False)
    target_dof_pos_onnx: np.ndarray = field(init=False)
    target_dof_pos_real_buffers: list[np.ndarray] = field(init=False)
    target_dof_pos_real_buffer: np.ndarray = field(init=False)
    target_dof_pos_real_buffer_idx: int = field(init=False)
    motion_filtered_actions_onnx: np.ndarray | None = None
    last_button_states: dict[str, int] = field(default_factory=_default_button_states)
    last_vr_status_log_time: float | None = None
    vr_queue_ready_logged: bool = False
    vr_fk_started_logged: bool = False
    vr_cold_start_logged: bool = False
    policy_slow_count: int = 0

    def __post_init__(self) -> None:
        self.resize_actions(self.num_actions)

    @property
    def latest_obs_flag(self) -> bool:
        """Compatibility alias for old node code.

        In Phase 3D this means "motion mode uses live VR reference"; it no longer
        means that the VR queue has merely become ready while still in velocity
        mode.
        """

        return bool(self.motion_uses_vr_reference)

    @latest_obs_flag.setter
    def latest_obs_flag(self, value: bool) -> None:
        self.motion_uses_vr_reference = bool(value)

    def resize_actions(self, num_actions: int) -> None:
        self.num_actions = int(num_actions)
        self.actions_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_onnx = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos_real_buffers = [
            np.zeros(self.num_actions, dtype=np.float32),
            np.zeros(self.num_actions, dtype=np.float32),
        ]
        self.target_dof_pos_real_buffer_idx = 0
        self.target_dof_pos_real_buffer = self.target_dof_pos_real_buffers[0]
        self.target_dof_pos_real = self.target_dof_pos_real_buffer
        self.motion_filtered_actions_onnx = None


class PolicyRuntime:
    """Owns mode transitions, VR gating, and policy-step orchestration."""

    def __init__(self, port: Any, num_actions: int = 29):
        self.port = port
        self.state = PolicyRuntimeState(num_actions=int(num_actions))
        self._velocity_input_feed: dict[str, np.ndarray] | None = None
        self._velocity_output_names: list[str] | None = None
        self._motion_input_feed: dict[str, np.ndarray] | None = None
        self._motion_output_names: list[str] | None = None
        self._motion_step_idx_array: np.ndarray | None = None

    def resize_actions(self, num_actions: int) -> None:
        self.state.resize_actions(int(num_actions))

    def _obs_evaluator(self):
        return getattr(self.port, "observation_evaluator", self.port)

    def set_robot_state(self, robot_state: str) -> None:
        if robot_state == "MOVE_TO_DEFAULT":
            self.state.robot_state_ready = True
        elif robot_state in {"ZERO_TORQUE", "EMERGENCY_STOP"}:
            self.state.robot_state_ready = False

    def handle_low_state(self, ls_msg: Any) -> None:
        """Store low-state input and refresh Unitree remote state when selected."""

        self.port._lowstate_msg = ls_msg
        if getattr(self.port, "policy_input_source", "unitree") == "unitree":
            self.port.remote_controller.set(ls_msg.wireless_remote)

    def handle_policy_input(self) -> None:
        control_input = self.port.read_policy_input()

        if (
            self._is_button_pressed("enable_velocity", control_input.enable_velocity)
            and self.state.robot_state_ready
        ):
            if self.state.current_policy_mode in {"motion", "hold"}:
                safe, max_error, detail = self._arms_safe_for_velocity()
                if not safe:
                    self.port.get_logger().warn(
                        "Walking transition blocked by arm-pose gate: "
                        f"max_error={max_error:.3f}rad "
                        f"limit={self._velocity_return_arm_max_error_rad():.3f}rad"
                        + (f" ({detail})" if detail else "")
                    )
                    return
            self.enable_velocity_policy()

        if (
            self._is_button_pressed("switch_to_motion", control_input.switch_to_motion)
            and self.state.robot_state_ready
            and self.state.policy_enabled
            and self.state.current_policy_mode in {"velocity", "hold"}
        ):
            self.switch_to_motion_mode()

        if self.state.current_policy_mode == "velocity":
            self.state.vx = control_input.vx
            self.state.vy = control_input.vy
            self.state.vyaw = control_input.vyaw
        else:
            self.state.vx, self.state.vy, self.state.vyaw = 0.0, 0.0, 0.0

    def enable_velocity_policy(self) -> None:
        self.state.policy_enabled = True
        self.state.current_policy_mode = "velocity"
        self.state.motion_uses_vr_reference = False
        self.state.motion_reference_frozen = False
        obs_eval = self._obs_evaluator()
        if hasattr(obs_eval, "clear_motion_yaw_alignment"):
            obs_eval.clear_motion_yaw_alignment()
        self.reset_motion_action_ema_filter()
        self.reset_counters()
        self.state.actions_onnx = np.zeros(self.port.num_actions, dtype=np.float32)
        self.state.target_dof_pos_onnx = (
            self.port.velocity_default_angles_onnx.copy()
        )
        self.port._publish_control_params()
        self.port.get_logger().info(
            f"Policy enabled in {self.state.current_policy_mode} tracking mode"
        )

    def switch_to_velocity_mode(self, reason: str = "") -> None:
        self.state.current_policy_mode = "velocity"
        self.state.motion_uses_vr_reference = False
        self.state.motion_reference_frozen = False
        self.state.motion_in_progress = False
        obs_eval = self._obs_evaluator()
        if hasattr(obs_eval, "clear_motion_yaw_alignment"):
            obs_eval.clear_motion_yaw_alignment()
        if hasattr(obs_eval, "clear_vr_fk_cache"):
            obs_eval.clear_vr_fk_cache()
        else:
            obs_eval._fk_vr_out = None
            obs_eval._use_fk_vr = False
        self.reset_motion_action_ema_filter()
        self.reset_counters()
        self.state.actions_onnx = np.zeros(self.port.num_actions, dtype=np.float32)
        self.state.target_dof_pos_onnx = (
            self.port.velocity_default_angles_onnx.copy()
        )
        self.port._publish_control_params()
        if reason:
            self.port.get_logger().info(f"Switched to velocity tracking mode ({reason})")
        else:
            self.port.get_logger().info("Switched to velocity tracking mode")

    def enter_hold_mode(self, reason: str) -> None:
        motor_state = getattr(self.port._lowstate_msg, "motor_state", None)
        if motor_state is None or len(motor_state) < self.port.num_actions:
            self.port.get_logger().error(
                f"Cannot enter HOLD without a complete low-state sample ({reason})"
            )
            self.state.policy_enabled = False
            return
        hold_target = np.asarray(
            [motor_state[index].q for index in range(self.port.num_actions)],
            dtype=np.float32,
        )
        if not np.isfinite(hold_target).all():
            self.port.get_logger().error(
                f"Cannot enter HOLD with non-finite joint positions ({reason})"
            )
            self.state.policy_enabled = False
            return
        self.state.current_policy_mode = "hold"
        self.state.motion_uses_vr_reference = False
        self.state.motion_reference_frozen = False
        self.state.motion_in_progress = False
        self.state.vx, self.state.vy, self.state.vyaw = 0.0, 0.0, 0.0
        self.state.target_dof_pos_real = hold_target
        self.port.get_logger().warn(f"Entered HOLD mode ({reason})")

    def _velocity_return_arm_max_error_rad(self) -> float:
        return float(getattr(self.port, "velocity_return_arm_max_error_rad", 0.45))

    def _arms_safe_for_velocity(self) -> tuple[bool, float, str]:
        if not bool(getattr(self.port, "velocity_return_arm_gate_enabled", False)):
            return True, 0.0, "gate disabled"
        obs_eval = self._obs_evaluator()
        current = getattr(obs_eval, "robot_dof_pos_by_name", {})
        defaults = {
            name: float(self.port.velocity_default_angles_onnx[index])
            for index, name in enumerate(self.port.velocity_dof_names_onnx)
        }
        return arm_pose_is_safe_for_walking(
            current,
            defaults,
            max_error_rad=self._velocity_return_arm_max_error_rad(),
        )

    def switch_to_motion_mode(self) -> bool:
        vr_data_available = self._vr_reference_available()
        vr_ready = self.port._is_vr_ready_for_motion()
        if self.port.enable_teleop_reference and not vr_ready:
            self.port.get_logger().warn(
                "VR teleoperation is enabled but the VR queue is not ready yet; keeping the current mode."
            )
            return False

        if not self.port.enable_teleop_reference:
            self.port.get_logger().warn(
                "Motion mode requires live MotionReference input in this ROS2 teleop package; keeping the current mode."
            )
            return False

        self.state.current_policy_mode = "motion"
        self.reset_motion_action_ema_filter()
        self.reset_counters()
        if self.port.use_kv_cache:
            self.port.get_logger().info("Motion KV-Cache reset.")
        self.port.get_logger().info("Motion Step Index reset to 0.")
        self.state.actions_onnx = np.zeros(self.port.num_actions, dtype=np.float32)
        self.state.target_dof_pos_onnx = self.port.motion_default_angles_onnx.copy()
        self.port._publish_control_params()

        self.state.motion_uses_vr_reference = bool(
            self.port.enable_teleop_reference and vr_data_available
        )
        self.state.motion_reference_frozen = False
        obs_eval = self._obs_evaluator()
        if hasattr(obs_eval, "begin_motion_yaw_alignment"):
            obs_eval.begin_motion_yaw_alignment()
        source_mode = (
            "ROS2 MotionReference"
            if self.state.motion_uses_vr_reference
            else "unavailable"
        )
        self.port.get_logger().info(
            f"Switched to motion tracking mode ({source_mode})"
        )
        if self.state.motion_uses_vr_reference:
            self.port.get_logger().info("[VR] Reference trajectory source: ROS2 MotionReference")
            obs_eval._warmup_fk_for_vr()
        self.state.motion_in_progress = True
        return True

    def reset_counters(self) -> None:
        self.state.motion_frame_idx = 0
        self.state.motion_step_idx = 0
        if self.port.use_kv_cache and self.port.motion_kv_cache is not None:
            self.port.motion_kv_cache.fill(0)

    def _maybe_reset_motion_rope_window(self) -> None:
        max_seq_len = int(getattr(self.port, "motion_rope_max_seq_len", 0) or 0)
        if max_seq_len <= 0 or self.port.motion_step_idx_input_name is None:
            return
        margin = int(getattr(self.port, "motion_rope_reset_margin", 0) or 0)
        reset_at = max_seq_len - margin
        if reset_at <= 0:
            reset_at = max_seq_len
        if self.state.motion_step_idx < reset_at:
            return

        old_step_idx = self.state.motion_step_idx
        if self.port.motion_kv_cache is not None:
            self.port.motion_kv_cache.fill(0)
        self.state.motion_step_idx = 0
        if self._motion_step_idx_array is not None:
            self._motion_step_idx_array[0] = 0
        self.port.get_logger().warn(
            "Motion RoPE step index reached "
            f"{old_step_idx}/{max_seq_len}; reset motion KV cache and step index "
            "to avoid exceeding exported RoPE cache."
        )

    def reset_motion_action_ema_filter(self) -> None:
        self.state.motion_filtered_actions_onnx = None

    def apply_motion_action_ema_filter(self, raw_actions: np.ndarray) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
        if not self.port.motion_action_ema_filter_enabled:
            return raw_actions.copy()

        if self.state.motion_filtered_actions_onnx is None:
            self.state.motion_filtered_actions_onnx = raw_actions.copy()
            return self.state.motion_filtered_actions_onnx

        alpha = float(self.port.motion_action_ema_filter_alpha)
        filtered_actions = self.state.motion_filtered_actions_onnx
        filtered_actions *= 1.0 - alpha
        filtered_actions += alpha * raw_actions
        return filtered_actions

    def mark_vr_queue_ready(self) -> None:
        if self.state.vr_fk_started_logged:
            return
        self.port.get_logger().info(
            "[VR] MotionReference data is ready; the main thread will build the reference trajectory from live ROS2 input."
        )
        self.state.vr_fk_started_logged = True

    def run(self) -> None:
        if not getattr(self.port, "_setup_completed", False):
            return

        t_loop_start = time.perf_counter()
        now = time.time()
        t_io = time.perf_counter()
        self.port._poll_motion_reference()
        self.handle_policy_input()
        self._log_vr_status(now)
        self._log_vr_ready(now)
        io_ms = self.port._timing_ms(t_io)

        policy_timing = self.run_policy_step()
        run_elapsed = 0.0
        if policy_timing is not None:
            run_elapsed = float(policy_timing.get("policy_total_ms", 0.0)) / 1000.0
        self._log_policy_latency(run_elapsed)

        if policy_timing is not None:
            sample = dict(policy_timing)
            sample["io_ms"] = io_ms
            sample["loop_total_ms"] = self.port._timing_ms(t_loop_start)
            self.port._record_timing_sample(sample)

    def run_policy_step(self) -> dict[str, float] | None:
        if self.port._lowstate_msg is None or not self.state.policy_enabled:
            return None

        timing_info = {
            "policy_total_ms": 0.0,
            "fk_ms": 0.0,
            "obs_ms": 0.0,
            "onnx_ms": 0.0,
            "post_ms": 0.0,
        }
        t_policy_start = time.perf_counter()

        if self.state.current_policy_mode == "hold":
            self.publish_current_action()
            self.port._publish_policy_mode()
            timing_info["post_ms"] = self.port._timing_ms(t_policy_start)
            timing_info["policy_total_ms"] = timing_info["post_ms"]
            return timing_info

        if self.state.current_policy_mode == "motion":
            if self.state.motion_uses_vr_reference:
                current_time = time.time()
                if self.port._vr_reference is None:
                    data_age = float("inf")
                else:
                    data_age = self.port._vr_reference.data_age(current_time)

                if data_age > self.port.max_data_age:
                    if not self.freeze_motion_reference(data_age):
                        self.publish_current_action()
                        self.port._publish_policy_mode()
                        timing_info["post_ms"] = self.port._timing_ms(t_policy_start)
                        timing_info["policy_total_ms"] = timing_info["post_ms"]
                        return timing_info
                elif self.state.motion_reference_frozen:
                    self.state.motion_reference_frozen = False
                    self.port.get_logger().info(
                        "MotionReference resumed; continuing live motion-policy tracking."
                    )

            if not self.state.motion_uses_vr_reference and (
                not hasattr(self.port, "n_motion_frames")
                or not hasattr(self.port, "ref_dof_pos")
            ):
                self.port.get_logger().warn(
                    "Live MotionReference is not active, skipping motion policy execution"
                )
                return None

            if (
                self.state.motion_uses_vr_reference
                and self.port._vr_reference is not None
            ):
                obs_eval = self._obs_evaluator()
                try:
                    n_fut = int(getattr(self.port, "n_fut_frames", 0))
                    if (
                        n_fut > 0
                        and getattr(obs_eval, "fk", None) is not None
                        and self.port._vr_reference.has_future_sequence(n_fut)
                    ):
                        t_fk = time.perf_counter()
                        cur_root_pos = obs_eval.ref_root_pos_raw.astype(np.float32)
                        cur_root_rot = self.port._vr_reference.current_root_rot()
                        if cur_root_rot is None:
                            obs_eval.clear_vr_fk_cache()
                            return None
                        if hasattr(obs_eval, "_compute_and_cache_vr_root_fk"):
                            obs_eval._compute_and_cache_vr_root_fk(
                                vr_reference=self.port._vr_reference,
                                cur_root_pos=cur_root_pos,
                                cur_root_rot=cur_root_rot,
                                n_fut=n_fut,
                                fps=float(1.0 / self.port.dt),
                            )
                        else:
                            cur_dof_pos = obs_eval.ref_dof_pos_raw.astype(np.float32)
                            root_pos_tensor, root_rot_tensor, dof_pos_tensor = (
                                obs_eval._prepare_vr_fk_tensors(
                                    vr_reference=self.port._vr_reference,
                                    cur_root_pos=cur_root_pos,
                                    cur_root_rot=cur_root_rot,
                                    cur_dof_pos=cur_dof_pos,
                                    n_fut=n_fut,
                                )
                            )
                            fk_out = obs_eval.fk(
                                root_pos=root_pos_tensor,
                                root_quat=root_rot_tensor,
                                dof_pos=dof_pos_tensor,
                                fps=float(1.0 / self.port.dt),
                                quat_format="wxyz",
                                vel_smoothing_sigma=0.0,
                                compute_velocity=False,
                            )
                            obs_eval._fk_vr_out = {
                                k: v.detach().cpu().numpy() for k, v in fk_out.items()
                            }
                        timing_info["fk_ms"] = self.port._timing_ms(t_fk)
                    else:
                        obs_eval.clear_vr_fk_cache()
                except Exception as exc:
                    self.port.get_logger().error(
                        f"VR FK computation failed; falling back to offline reference: {exc}"
                    )
                    obs_eval.clear_vr_fk_cache()

            self.port.obs_builder = self.port.motion_obs_builder
            current_action_scale = self.port.motion_action_scale_onnx
            current_default_angles = self.port.motion_default_angles_onnx
            current_onnx_to_real = self.port.motion_onnx_to_real
        else:
            self.port.obs_builder = self.port.velocity_obs_builder
            current_action_scale = self.port.velocity_action_scale_onnx
            current_default_angles = self.port.velocity_default_angles_onnx
            current_onnx_to_real = self.port.velocity_onnx_to_real

        t_obs = time.perf_counter()
        obs_eval = self._obs_evaluator()
        if hasattr(obs_eval, "cache_lowstate"):
            obs_eval.cache_lowstate(self.port._lowstate_msg, force=True)
        if self.state.current_policy_mode == "motion":
            if not getattr(obs_eval, "_fk_vr_cache_ready", False):
                obs_eval._cache_fk_vr_for_obs()
        policy_obs_base = self.port.obs_builder.build_policy_obs()
        if hasattr(self.port.obs_builder, "batch_view"):
            policy_obs_np = self.port.obs_builder.batch_view()
        else:
            policy_obs_np = policy_obs_base[None, :].astype(np.float32, copy=False)
        timing_info["obs_ms"] = self.port._timing_ms(t_obs)

        if (
            self.state.current_policy_mode == "motion"
            and self.state.motion_uses_vr_reference
        ):
            self.port._publish_motion_tracker_reference()

        t_onnx = time.perf_counter()
        onnx_output = self._run_current_policy(policy_obs_np)
        timing_info["onnx_ms"] = self.port._timing_ms(t_onnx)

        t_post = time.perf_counter()
        raw_actions_onnx = np.asarray(onnx_output[0], dtype=np.float32).reshape(-1)
        self.apply_policy_output(
            raw_actions_onnx,
            action_scale=current_action_scale,
            default_angles=current_default_angles,
            onnx_to_real=current_onnx_to_real,
            is_motion=self.state.current_policy_mode == "motion",
        )
        self.publish_current_action()
        self._mark_offline_motion_complete()
        self.port._publish_policy_mode()
        timing_info["post_ms"] = self.port._timing_ms(t_post)
        timing_info["policy_total_ms"] = self.port._timing_ms(t_policy_start)
        return timing_info

    def freeze_motion_reference(self, data_age: float) -> bool:
        if self.state.motion_reference_frozen:
            return True
        reference = self.port._vr_reference
        if reference is None or not reference.freeze_at_current_pose():
            self.enter_hold_mode(reason="MotionReference stale without a valid reference")
            return False
        self.port.get_logger().warn(
            f"MotionReference is stale: age={data_age*1000:.1f}ms > "
            f"{self.port.max_data_age*1000:.1f}ms; freezing the last valid "
            "pose and continuing motion-policy inference."
        )
        self.state.motion_reference_frozen = True
        return True

    def apply_policy_output(
        self,
        raw_actions_onnx: np.ndarray,
        *,
        action_scale: np.ndarray,
        default_angles: np.ndarray,
        onnx_to_real: Any,
        is_motion: bool,
    ) -> np.ndarray:
        if is_motion:
            if self.port.motion_action_ema_filter_enabled:
                actions = self.apply_motion_action_ema_filter(raw_actions_onnx)
            else:
                actions = np.asarray(raw_actions_onnx, dtype=np.float32).reshape(-1)
            np.copyto(self.state.actions_onnx, actions)
        else:
            actions = np.asarray(raw_actions_onnx, dtype=np.float32).reshape(-1)
            np.copyto(self.state.actions_onnx, actions)

        np.multiply(
            self.state.actions_onnx,
            action_scale,
            out=self.state.target_dof_pos_onnx,
        )
        np.add(
            self.state.target_dof_pos_onnx,
            default_angles,
            out=self.state.target_dof_pos_onnx,
        )
        self.state.target_dof_pos_real_buffer_idx = (
            1 - self.state.target_dof_pos_real_buffer_idx
        )
        target_real = self.state.target_dof_pos_real_buffers[
            self.state.target_dof_pos_real_buffer_idx
        ]
        np.take(self.state.target_dof_pos_onnx, onnx_to_real, out=target_real)
        self.state.target_dof_pos_real_buffer = target_real
        self.state.target_dof_pos_real = target_real
        return self.state.target_dof_pos_real

    def publish_current_action(self) -> None:
        self.port._publish_action_target(self.state.target_dof_pos_real)
        self.state.motion_frame_idx += 1

    def _run_current_policy(self, policy_obs_np: np.ndarray):
        if self.state.current_policy_mode == "velocity":
            if self._velocity_input_feed is None:
                self._velocity_input_feed = {self.port.velocity_input_name: policy_obs_np}
            else:
                self._velocity_input_feed[self.port.velocity_input_name] = policy_obs_np
            if self._velocity_output_names is None:
                self._velocity_output_names = [self.port.velocity_output_name]
            return self.port.velocity_policy_session.run(
                self._velocity_output_names,
                self._velocity_input_feed,
            )

        if self.port.use_kv_cache:
            if self.port.motion_kv_cache is None:
                shape = [
                    dim if isinstance(dim, int) else 1
                    for dim in self.port.motion_kv_shape
                ]
                self.port.motion_kv_cache = np.zeros(
                    shape, dtype=self.port.motion_kv_dtype
                )

            self._maybe_reset_motion_rope_window()

            if self._motion_input_feed is None:
                self._motion_input_feed = {
                    self.port.motion_input_name: policy_obs_np,
                    self.port.motion_kv_input_name: self.port.motion_kv_cache,
                }
            else:
                self._motion_input_feed[self.port.motion_input_name] = policy_obs_np
                self._motion_input_feed[self.port.motion_kv_input_name] = (
                    self.port.motion_kv_cache
                )
            if self.port.motion_step_idx_input_name is not None:
                if self._motion_step_idx_array is None:
                    self._motion_step_idx_array = np.zeros(1, dtype=np.int64)
                self._motion_step_idx_array[0] = self.state.motion_step_idx
                self._motion_input_feed[self.port.motion_step_idx_input_name] = (
                    self._motion_step_idx_array
                )

            if self._motion_output_names is None:
                self._motion_output_names = [self.port.motion_output_name]
                if self.port.motion_kv_output_name:
                    self._motion_output_names.append(self.port.motion_kv_output_name)
            onnx_output = self.port.motion_policy_session.run(
                self._motion_output_names,
                self._motion_input_feed,
            )
            if len(onnx_output) > 1:
                self.port.motion_kv_cache = onnx_output[1]
            self.state.motion_step_idx += 1
            return onnx_output

        if self._motion_input_feed is None:
            self._motion_input_feed = {self.port.motion_input_name: policy_obs_np}
        else:
            self._motion_input_feed[self.port.motion_input_name] = policy_obs_np
        if self._motion_output_names is None:
            self._motion_output_names = [self.port.motion_output_name]
        return self.port.motion_policy_session.run(
            self._motion_output_names,
            self._motion_input_feed,
        )

    def _is_button_pressed(self, button_name: str, current_state: bool) -> bool:
        current = 1 if current_state else 0
        last_state = self.state.last_button_states.get(button_name, 0)
        self.state.last_button_states[button_name] = current
        return current == 1 and last_state == 0

    def _vr_reference_available(self) -> bool:
        return bool(
            self.port.enable_teleop_reference
            and self.port._vr_reference is not None
            and self.port._vr_reference.has_latest_obs
        )

    def _log_vr_status(self, now: float) -> None:
        if self.state.current_policy_mode != "motion":
            return
        if self.state.last_vr_status_log_time is None:
            self.state.last_vr_status_log_time = now
            return
        if now - self.state.last_vr_status_log_time < 5.0:
            return

        vr_available = bool(
            self.port._vr_reference is not None and self.port._vr_reference.has_latest_obs
        )
        if vr_available:
            intervals = list(getattr(self.port, "_motion_reference_intervals", []))
            if intervals:
                avg_interval = float(np.mean(np.asarray(intervals, dtype=np.float64)))
                freq_text = f"expected_freq={1.0 / avg_interval:.1f}Hz" if avg_interval > 0 else "expected_freq=unknown"
            else:
                freq_text = "expected_freq=unknown"
            seen_frames = int(getattr(self.port._vr_reference, "seen_frames", 0))
            age_ms = self.port._vr_reference.data_age(now) * 1000.0
            self.port.get_logger().info(
                "[VR-STATUS] MotionReference streaming | "
                f"seen_frames={seen_frames} age={age_ms:.1f}ms {freq_text}"
            )
        else:
            self.port.get_logger().warn(
                "[VR-STATUS] No MotionReference received in the last 5 seconds; "
                "motion mode will stay unavailable until live reference data arrives."
            )
        self.state.last_vr_status_log_time = now

    def _log_vr_ready(self, now: float) -> None:
        del now
        if (
            not self.port.enable_teleop_reference
            or not self.state.policy_enabled
            or self.state.vr_queue_ready_logged
            or not self.port._is_vr_ready_for_motion()
        ):
            return
        seen_frames = (
            self.port._vr_reference.seen_frames
            if self.port._vr_reference is not None
            else 0
        )
        self.port.get_logger().info(
            f"[VR] VR queue is ready for motion mode (seen_frames={int(seen_frames)}, "
            f"n_fut={int(getattr(self.port, 'n_fut_frames', 0) or 0)}, "
            f"delay={int(getattr(self.port, 'reference_delay_frames', 0) or 0)})"
        )
        self.state.vr_queue_ready_logged = True

    def _log_policy_latency(self, run_elapsed: float) -> None:
        if not (
            self.state.current_policy_mode == "motion"
            and self.state.motion_uses_vr_reference
        ):
            return
        if run_elapsed > 0.5 and not self.state.vr_cold_start_logged:
            self.state.vr_cold_start_logged = True
            self.port.get_logger().info(
                "[VR] The first motion step is a cold start (FK/ONNX initialization) and may take about 1 second."
            )
        if run_elapsed > 1.15 * self.port.dt and run_elapsed <= 0.5:
            self.state.policy_slow_count += 1
            if (
                self.state.policy_slow_count == 1
                or self.state.policy_slow_count % 50 == 0
            ):
                self.port.get_logger().warn(
                    f"[VR] Policy step latency {run_elapsed*1000:.1f} ms exceeds the target "
                    f"{self.port.dt*1000:.1f} ms. Estimated /humanoid/action rate: "
                    f"{1.0/run_elapsed:.1f} Hz (target {1.0/self.port.dt:.0f} Hz). "
                    "The main bottleneck is usually FK or ONNX inference; if the system settles near 30 Hz, consider setting policy_freq to 30."
                )

    def _mark_offline_motion_complete(self) -> None:
        if self.state.current_policy_mode != "motion":
            return
        if not self.state.motion_uses_vr_reference and self.state.motion_in_progress:
            self.switch_to_velocity_mode(reason="live MotionReference unavailable")
