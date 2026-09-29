"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

HoloMotion v1.4 teleoperation agent (the teleop-collection branch's controller, run in MuJoCo).

    PICO headset + controllers
      -> reference publisher (holomotion_v14/publisher.py, holomotion_teleop env): body -> SMPL -> GMR -> obs65,
         controller sample -> pico_ctrl                                              [ZMQ, tcp://*:6001]
      -> HoloMotionV14Agent
           main-node states (the robot's C++ main_node): WAIT -> MOVE_TO_DEFAULT -> POLICY, ZERO_TORQUE, E-STOP
           policy node: the vendored HoloMotionPolicyNode (velocity model on A, motion model on B, Y/A back)
           Dex3 hands: the v1.4 grip gripper (grip > 0.5 -> close pose)
      -> ActionCmd("holomotion") -> G1Sonic PD torques with the policy's own kps/kds

Buttons are the robot's (v1.4.1 bit map). Sim-only actions (recording, reset) are read by the CLI and use combos the
robot never sees: while the left menu button is held, no button reaches the policy.
"""

from __future__ import annotations

import time

import numpy as np

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.holomotion.protocol import DEX3_NATURAL_TO_MJCF
from simple.teleop.holomotion_v14 import DEFAULT_REFERENCE_URI, ensure_vendor_path
from simple.teleop.holomotion_v14.sim_node import build_sim_policy_node, robot_yaml
from simple.teleop.holomotion_v14.wire import ReferenceSubscriber

from .sonic_wbc_agent import SonicWbcAgent

MOVE_TO_DEFAULT_SEC = 3.0          # main_node.cpp duration_
QUICK_SETTLE_SEC = 0.5             # quick start: walking policy on for this long before switching to motion tracking
ESTOP_DAMPING_SEC = 2.0            # main_node.cpp emergency_damping_duration_
PICO_TIMEOUT_SEC = 0.5             # the adapter's control_timeout_sec


class HoloMotionV14Agent(SonicWbcAgent):
    def __init__(
        self,
        robot: G1Sonic,
        *,
        reference_uri: str = DEFAULT_REFERENCE_URI,
        auto_stand: bool = False,
        enable_pico_stream: bool = True,
        stream_view: str = "single",
        quiet: bool = False,
    ) -> None:
        super().__init__(robot)
        ensure_vendor_path()
        from holomotion_peripherals_ros2.pico_dex3_gripper import target_q_for_grip
        self._grip_q = target_q_for_grip

        self.episodes_saved = 0
        self.num_episodes = 100
        self.image_publish_process = None
        self.sim_dt = float(self.robot.sonic_config["SIMULATE_DT"])
        self._control_dt = 4 * self.sim_dt
        self.auto_stand = bool(auto_stand)

        t0 = time.monotonic()
        self.node = build_sim_policy_node(quiet=quiet)
        print(f"[HoloMotion v1.4] policy node ready in {time.monotonic() - t0:.1f} s "
              f"(velocity {self.node.velocity_onnx_path}, motion {self.node.motion_onnx_path})")
        self.sub = ReferenceSubscriber(reference_uri)
        self.reference_uri = reference_uri

        # robot (real / motor) order <-> SIMPLE MJCF order, by name
        y = robot_yaml()
        real = list(self.node.real_dof_names)
        mjcf = list(self.robot.joint_names[:len(real)])
        missing = sorted(set(real) - set(mjcf))
        assert not missing, f"G1 model lacks joints the policy drives: {missing}"
        self.real_to_mjcf = np.array([real.index(n) for n in mjcf], dtype=np.int64)       # mjcf[i] = real[map[i]]
        self.mjcf_to_real = np.array([mjcf.index(n) for n in real], dtype=np.int64)       # real[i] = mjcf[map[i]]
        self.default_real = np.array([float(y["default_joint_angles"][n]) for n in real], dtype=np.float32)
        order = list(y.get("complete_dof_order", real))
        self.mtd_kp_real = np.array([float(y["kp"][order.index(n)]) for n in real], dtype=np.float32)
        self.mtd_kd_real = np.array([float(y["kd"][order.index(n)]) for n in real], dtype=np.float32)

        # Elastic band / reset (same conventions as the other SIMPLE teleop agents)
        self._dropping = False
        self._drop_rate = 0.15
        self._reset_requested = False

        self._streaming = None
        self._frame_buffer = None
        self.stream_view = stream_view                   # single | mono | stereo, see compose_stream_frame
        self.stream_status: list[tuple[str, tuple[int, int, int]]] = []   # overlay lines (text, BGR), set by the CLI
        if enable_pico_stream:
            from .holomotion_pico_agent import HoloMotionPicoAgent
            from simple.teleop.pico.streaming import FrameBuffer
            self._frame_buffer = FrameBuffer()
            HoloMotionPicoAgent._init_pico_streamer(self)        # same headset camera stream (port 13579)
        self.reset_policy()

    # ------------------------------------------------------------------ headset stream
    def update_render_caches(self, observation: dict):
        if self._streaming is not None and self._streaming.is_running():
            frame = compose_stream_frame(observation.get("head_stereo_left"), observation.get("head_stereo_right"),
                                         self.stream_status, self.stream_view)
            if frame is not None:
                self._frame_buffer.put(frame)
        return observation

    # ------------------------------------------------------------------ state
    def reset_policy(self) -> None:
        """A fresh start, as after re-launching the robot stack: runtime state, obs history, VR queue, KV cache."""
        from humanoid_policy.policy_runtime import PolicyRuntimeState
        from humanoid_policy.utils.remote_controller_filter import RemoteController
        n = self.node
        n.runtime.state = PolicyRuntimeState(num_actions=n.num_actions)
        n._init_obs_buffers()
        n.observation_evaluator.initialize_observation_state()
        for fn in ("clear_motion_yaw_alignment", "clear_vr_fk_cache"):
            if hasattr(n.observation_evaluator, fn):
                getattr(n.observation_evaluator, fn)()
        if n.motion_kv_cache is not None:
            n.motion_kv_cache.fill(0)
        n.pico_controller = RemoteController()
        n._pico_enable_velocity = n._pico_switch_to_motion = False
        n._pico_last_receive_time = None
        n.set_robot_state("ZERO_TORQUE")
        self.main_state = "WAIT"
        self._state_t0 = time.monotonic()
        self._mtd_from = None
        self._hold_real = None
        self._last_bits = 0
        self._dropping = False
        self._reset_time = time.monotonic()
        self._left_hand = np.zeros(7, dtype=np.float32)
        self._right_hand = np.zeros(7, dtype=np.float32)
        self._last = {}
        self._quick = None
        self._quick_ready = False

    # ------------------------------------------------------------------ quick start
    def quick_start(self) -> None:
        """Skip the leash and the stand-up: the robot is already standing on the floor (the CLI placed it, band off).
        Enters MOVE_TO_DEFAULT at the default pose, then presses A (walking policy) and, QUICK_SETTLE_SEC later, B
        (motion tracking) for the operator, through the same paths as the real buttons. quick_ready turns True once
        motion tracking is on."""
        self._quick = dict(phase="stand", ticks=0, t0=None, warned=False)
        self._quick_ready = False

    @property
    def quick_ready(self) -> bool:
        """True once per quick start, when motion tracking has taken over."""
        if self._quick_ready:
            self._quick_ready = False
            return True
        return False

    @property
    def quick_active(self) -> bool:
        return self._quick is not None

    def _quick_step(self, s: dict | None, q_real: np.ndarray) -> dict | None:
        from simple.teleop.holomotion_v14.wire import BUTTONS
        Q, n = self._quick, self.node
        if Q["phase"] == "stand":
            self._set_main_state("MOVE_TO_DEFAULT", q_real)
            self._mtd_from = self.default_real.copy()          # already there: hold the default pose
            Q["phase"], Q["ticks"] = "A", 0
        if self.main_state in ("ZERO_TORQUE", "EMERGENCY_STOP"):   # the operator pressed X / R3: stop helping
            self._quick = None
            return s
        if s is None or s.get("left_menu"):                    # no controller sample (or a sim combo held): wait
            if s is None and not Q["warned"]:
                print("[HoloMotion v1.4] quick start: waiting for the PICO controllers")
                Q["warned"] = True
            return s
        Q["ticks"] += 1
        press = None
        if Q["phase"] == "A":
            if self.main_state == "POLICY" and n.policy_enabled:
                Q["phase"], Q["ticks"] = "settle", 0
            else:
                press = "A" if Q["ticks"] % 4 else None        # 3 ticks pressed, 1 released: a fresh edge each time
        if Q["phase"] == "settle" and Q["ticks"] * self._control_dt >= QUICK_SETTLE_SEC:
            Q["phase"], Q["ticks"] = "B", 0
        if Q["phase"] == "B":
            if n.current_policy_mode == "motion" and n.policy_enabled:
                self._quick = None
                self._quick_ready = True
                print("[HoloMotion v1.4] quick start: standing, motion tracking on")
                return s
            press = "B" if Q["ticks"] % 4 else None
            if Q["ticks"] == int(5.0 / self._control_dt):
                print("[HoloMotion v1.4] quick start: motion tracking not ready yet -- is the body reference streaming?")
        if press:
            s = dict(s)
            s[press] = 1
            s["button_bits"] = int(s.get("button_bits", 0)) | BUTTONS[press]
        return s

    @property
    def reset_requested(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    @property
    def policy_mode(self) -> str:
        return f"{self.node.current_policy_mode}{'' if self.node.policy_enabled else '(off)'}"

    @property
    def tracking_active(self) -> bool:
        return self.main_state == "POLICY" and bool(self.node.policy_enabled)

    def pico(self) -> dict | None:
        """The newest controller sample, or None if older than the adapter's 0.5 s timeout."""
        s = self.sub.last_pico
        return s if s is not None and self.sub.pico_age() <= PICO_TIMEOUT_SEC else None

    # ------------------------------------------------------------------ main-node state machine
    def _set_main_state(self, state: str, q_real: np.ndarray) -> None:
        self.main_state = state
        self._state_t0 = time.monotonic()
        if state == "MOVE_TO_DEFAULT":
            self._mtd_from = q_real.copy()
            self.node.set_robot_state("MOVE_TO_DEFAULT")
        elif state in ("ZERO_TORQUE", "EMERGENCY_STOP"):
            self.node.set_robot_state(state)
        print(f"[HoloMotion v1.4] main state -> {state}")

    def _handle_main_buttons(self, s: dict | None, q_real: np.ndarray) -> None:
        bits = 0 if s is None or s.get("left_menu") else int(s.get("button_bits", 0))
        edge = bits & ~self._last_bits
        self._last_bits = bits
        from simple.teleop.holomotion_v14.wire import BUTTONS
        if edge & BUTTONS["R3"]:
            self._set_main_state("EMERGENCY_STOP", q_real)
        elif edge & BUTTONS["X"]:
            self._set_main_state("ZERO_TORQUE", q_real)
        elif edge & BUTTONS["L3"] and self.main_state in ("WAIT", "ZERO_TORQUE", "EMERGENCY_STOP"):
            self._set_main_state("MOVE_TO_DEFAULT", q_real)
        elif edge & BUTTONS["A"] and self.main_state == "MOVE_TO_DEFAULT":
            self._set_main_state("POLICY", q_real)
            if self._band_active():
                self._dropping = True                 # lower the sim band as the policy takes the weight

    def _band_active(self) -> bool:
        return bool(self.robot.elastic_band and self.robot.elastic_band.enable and self.robot.use_floating_root_link)

    # ------------------------------------------------------------------ step
    def get_action(self, observation, instruction=None, **kwargs) -> ActionCmd:
        proprio = kwargs["privileged_info"]["proprio"]
        body_q = np.asarray(proprio["body_q"], dtype=np.float32)
        body_dq = np.asarray(proprio["body_dq"], dtype=np.float32)
        base = np.asarray(proprio["floating_base_pose"], dtype=np.float32)
        base_vel = np.asarray(proprio["floating_base_vel"], dtype=np.float32)
        q_real, dq_real = body_q[self.mjcf_to_real], body_dq[self.mjcf_to_real]

        frames = self.sub.poll()
        s = self.pico()
        if self._quick is not None:
            s = self._quick_step(s, q_real)
        if self.auto_stand and self.main_state == "WAIT" and (self.robot.stabilized or time.monotonic() - self._reset_time > 1.0):
            self._set_main_state("MOVE_TO_DEFAULT", q_real)
        self._handle_main_buttons(s, q_real)

        # the policy node, exactly as on the robot: lowstate, controller, reference frames, then its 50 Hz timer
        n = self.node
        n.feed_lowstate(q_real, dq_real, base[3:7], base_vel[3:6])
        if s is not None and not s.get("left_menu"):
            s_fwd = dict(s)
            s_fwd["A"] = int(bool(s.get("A")) or bool(s.get("Y")))    # v1.4.1 bit map: Y -> back to velocity
            n.feed_pico(s_fwd)
        elif s is not None:
            n.feed_pico({**s, "A": 0, "B": 0, "X": 0, "Y": 0})         # left menu held: buttons belong to the CLI
        for f in frames:
            n.feed_reference(f["latest_obs"], int(f["frame_index"][0]), int(f.get("timestamp_ns", [0])[0]))
        t_run = time.perf_counter()
        n.run()
        run_ms = (time.perf_counter() - t_run) * 1e3

        # band
        band = self._band_active()
        if self._dropping and band:
            self.robot.elastic_band.length -= self._drop_rate * self._control_dt
            if self.robot.elastic_band.length <= -0.25 and abs(self.robot.pelvis_vz) < 0.05:
                self.robot.elastic_band.enable = False
                self._dropping = False
                band = False
                print(f"[HoloMotion v1.4] robot on its feet (pelvis z {self.robot.pelvis_z:.3f} m)")

        # joint targets and gains, as the main node applies them
        t = time.monotonic() - self._state_t0
        if self.main_state == "WAIT":
            if self._hold_real is None:
                self._hold_real = q_real.copy()
            target, kp, kd = self._hold_real, self.mtd_kp_real, self.mtd_kd_real
        elif self.main_state == "MOVE_TO_DEFAULT" or (self.main_state == "POLICY" and not n.policy_enabled):
            a = min(1.0, t / MOVE_TO_DEFAULT_SEC) if self._mtd_from is not None else 1.0
            src = self._mtd_from if self._mtd_from is not None else q_real
            target = (1 - a) * src + a * self.default_real
            kp, kd = self.mtd_kp_real, self.mtd_kd_real
        elif self.main_state == "POLICY":
            target = n.action_target if n.action_target is not None else self.default_real
            kp, kd = n.action_kps, n.action_kds
        elif self.main_state == "EMERGENCY_STOP" and t < ESTOP_DAMPING_SEC:
            target, kp, kd = q_real, np.zeros_like(q_real), self.mtd_kd_real
        else:                                              # ZERO_TORQUE, or E-stop after its damping phase
            target, kp, kd = q_real, np.zeros_like(q_real), np.zeros_like(q_real)

        # Dex3: the v1.4 grip gripper, independent of the body state
        if s is not None:
            self._left_hand = np.asarray(self._grip_q(side="left", status=1, grip=s["left_grip"]), dtype=np.float32)[DEX3_NATURAL_TO_MJCF]
            self._right_hand = np.asarray(self._grip_q(side="right", status=1, grip=s["right_grip"]), dtype=np.float32)[DEX3_NATURAL_TO_MJCF]

        st = n.runtime.state
        self._last = dict(run_ms=run_ms, frames=len(frames))
        return ActionCmd(
            "holomotion",
            target_q=np.asarray(target, dtype=np.float64)[self.real_to_mjcf],
            kp=np.asarray(kp, dtype=np.float64)[self.real_to_mjcf],
            kd=np.asarray(kd, dtype=np.float64)[self.real_to_mjcf],
            left_hand_q=self._left_hand.copy(),
            right_hand_q=self._right_hand.copy(),
            apply_elastic_band=band,
            dropping=self._dropping,
            # recording extras
            main_state=self.main_state,
            policy_mode=n.current_policy_mode,
            policy_enabled=bool(n.policy_enabled),
            navigate_cmd=np.array([st.vx, st.vy, st.vyaw, 0.0], dtype=np.float64),
            latest_obs=(np.zeros(65, np.float32) if n._vr_reference is None or n._vr_reference.latest_obs is None
                        else np.asarray(n._vr_reference.latest_obs, dtype=np.float32).reshape(-1)[:65].copy()),
            raw_action=np.asarray(st.actions_onnx, dtype=np.float32).copy(),
            target_real=np.asarray(target, dtype=np.float32).copy(),
        )

    def publish_low_state(self, proprio):
        pass

    def close(self) -> None:
        try:
            self.sub.close()
        except Exception:
            pass
        if self._streaming:
            self._streaming.stop()


def compose_stream_frame(left, right, status, view: str = "single"):
    """The headset frame (BGR), 2 x 16:9 wide as the XRoboToolkit app expects from a stereo camera. The app shows it
    as two pictures side by side by default; its right B toggles a single view that shows only the left half.

        single  the left head camera once, centred on a black frame: one picture in the app's default view
        mono    the left camera in both halves: one picture in the app's single view (two identical ones by default)
        stereo  left | right

    The status lines are drawn on the camera image (in both halves for mono/stereo)."""
    import cv2

    if left is None:
        return None
    left_bgr = np.ascontiguousarray(np.asarray(left)[..., :3][..., ::-1])
    if view == "stereo" and right is not None:
        right_bgr = np.ascontiguousarray(np.asarray(right)[..., :3][..., ::-1])
    else:
        right_bgr = left_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = left_bgr.shape[:2]
    scale = max(0.4, h / 720.0)
    x0, y = int(0.12 * w), int(0.12 * h)                          # inset: the headset crops the image edges
    for text, color in status:
        for img in (left_bgr, right_bgr):
            cv2.putText(img, text, (x0, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (x0, y), font, scale, color, 1, cv2.LINE_AA)
        y += int(22 * scale / 0.5)
    if view == "single":
        frame = np.zeros((h, 2 * w, 3), dtype=np.uint8)
        frame[:, w // 2:w // 2 + w] = left_bgr
        return frame
    return np.concatenate([left_bgr, right_bgr], axis=1)
