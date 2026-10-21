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
           policy node: the vendored HoloMotionPolicyNode (velocity model on A, motion model on Y; see button_map)
           Dex3 hands: the v1.4 grip gripper (grip > 0.5 -> close pose)
      -> ActionCmd("holomotion") -> G1Sonic PD torques with the policy's own kps/kds

Buttons (button_map): "sim" (default) A = walking, Y = motion tracking, B = nothing (it stays free for the PICO app's
view toggle); "robot" = the robot's v1.4.1 map, A = walking, B = motion tracking, Y = back to walking. Sim-only actions
(recording, reset) are read by the CLI and use combos the robot never sees: while the left menu button is held, no
button reaches the policy.
"""

from __future__ import annotations

import time

import numpy as np

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.holomotion.protocol import DEX3_NATURAL_TO_MJCF
from simple.teleop.holomotion_v14 import DEFAULT_REFERENCE_URI, ensure_vendor_path
from simple.teleop.holomotion_v14 import node_state as NS
from simple.teleop.holomotion_v14.sim_node import build_sim_policy_node, robot_yaml
from simple.teleop.holomotion_v14.wire import ReferenceSubscriber

from .sonic_wbc_agent import SonicWbcAgent

MOVE_TO_DEFAULT_SEC = 3.0          # main_node.cpp duration_
QUICK_SETTLE_SEC = 0.5             # quick start: walking policy on for this long before switching to motion tracking
# quick start before the PICO is connected: a controller sample with nothing pressed, to switch the walking policy on
NEUTRAL_PICO_SAMPLE = dict(A=0, B=0, X=0, Y=0, L3=0, R3=0, left_menu=0, right_menu=0, button_bits=0,
                           left_axis=[0.0, 0.0], right_axis=[0.0, 0.0], left_trigger=0.0, right_trigger=0.0,
                           left_grip=0.0, right_grip=0.0)
ESTOP_DAMPING_SEC = 2.0            # main_node.cpp emergency_damping_duration_
PICO_TIMEOUT_SEC = 0.5             # the adapter's control_timeout_sec


RAW_PICO_NAMES = ["left_x", "left_y", "right_x", "right_y", "left_grip", "right_grip", "left_trigger", "right_trigger",
                  "button_bits", "l_active", "r_active", "left_menu"]          # holomotion.pico column layout


VRSTATE_FIELDS = ["status", "a_button", "b_button", "x_button", "y_button", "axis_x", "axis_y", "trigger", "grip", "axis_click", "menu_button"]


def vrstate_vectors(fed: dict | None) -> dict:
    """robo_orchard_pico_msg_ros2/VRState as the node receives it: left_controller / right_controller fields (VRSTATE_FIELDS
    order), built with the same adapter the node is fed through (sim_node._VRState); NaN when nothing was fed this tick."""
    if fed is None:
        return {k: np.full(len(VRSTATE_FIELDS), np.nan, dtype=np.float32) for k in ("left_controller", "right_controller")}
    from simple.teleop.holomotion_v14.sim_node import _VRState
    vr = _VRState(fed)
    return {k: np.array([float(getattr(getattr(vr, k), f)) for f in VRSTATE_FIELDS], dtype=np.float32) for k in ("left_controller", "right_controller")}


def raw_pico_vector(s: dict | None) -> np.ndarray:
    """The controller sample of a tick as a fixed 12-vector (NaN when no sample was available)."""
    if not s:
        return np.full(len(RAW_PICO_NAMES), np.nan, dtype=np.float32)
    la, ra = s.get("left_axis", (0.0, 0.0)), s.get("right_axis", (0.0, 0.0))
    return np.array([la[0], la[1], ra[0], ra[1], s.get("left_grip", 0.0), s.get("right_grip", 0.0), s.get("left_trigger", 0.0),
                     s.get("right_trigger", 0.0), int(s.get("button_bits", 0)), int(bool(s.get("l_active", 0))),
                     int(bool(s.get("r_active", 0))), int(bool(s.get("left_menu", 0)))], dtype=np.float32)


class HoloMotionV14Agent(SonicWbcAgent):
    def __init__(
        self,
        robot: G1Sonic,
        *,
        reference_uri: str = DEFAULT_REFERENCE_URI,
        auto_stand: bool = False,
        enable_pico_stream: bool = True,
        button_map: str = "sim",
        stream_view: str = "single",
        stream_port: int = 13579,
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
        if button_map not in ("sim", "robot"):
            raise ValueError(f"button_map is 'sim' or 'robot', got {button_map!r}")
        self.button_map = button_map
        self.motion_button = "Y" if button_map == "sim" else "B"     # the button that switches to motion tracking

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
            from simple.teleop.pico.streaming import FrameBuffer
            self._frame_buffer = FrameBuffer()
            self._init_pico_streamer(stream_port)
        self.reset_policy()

    # ------------------------------------------------------------------ headset stream
    def _init_pico_streamer(self, port: int) -> None:
        """The XRoboToolkit video channel (as HoloMotionPicoAgent's, with the port as a parameter): the app sends
        OPEN_CAMERA to this port with where to send the H.264 stream; frames come from update_render_caches."""
        from simple.teleop.pico.streaming import StreamingThread
        from simple.teleop.pico.tcp_server import TCPControlServer
        from simple.teleop.pico.tcp_video_sender import TCPVideoSender
        tcp_server = TCPControlServer(f"0.0.0.0:{int(port)}")

        def on_open_camera(req):
            print(f"[HoloMotion v1.4] OPEN_CAMERA: {req}")
            if self._streaming and self._streaming.is_running():
                return
            ip, cport = req.get("ip"), req.get("port")
            if not ip or not cport:
                print("[HoloMotion v1.4] OPEN_CAMERA without ip/port: cannot stream")
                return
            fps = req.get("fps") or 60
            try:
                sender = TCPVideoSender(ip=ip, port=cport, width=req.get("width") or 2560, height=req.get("height") or 720,
                                        fps=fps, bitrate=req.get("bitrate") or 4_000_000, hevc=bool(req.get("enableMvHevc")))
            except ConnectionRefusedError:
                print(f"[HoloMotion v1.4] connection refused by {ip}:{cport}")
                return
            self._streaming = StreamingThread(frame_buffer=self._frame_buffer, fps=fps, publishers=[sender],
                                              on_ended=lambda: tcp_server.close_client())
            self._streaming.start()

        def on_close_camera():
            print("[HoloMotion v1.4] CLOSE_CAMERA")
            if self._streaming:
                self._streaming.stop()
                self._streaming = None
            tcp_server.close_client()

        tcp_server.on_open_camera = on_open_camera
        tcp_server.on_close_camera = on_close_camera
        tcp_server.start()
        self._tcp_server = tcp_server

    @property
    def streaming(self) -> bool:
        """The headset is connected and receiving the camera stream."""
        return self._streaming is not None and self._streaming.is_running()

    def update_render_caches(self, observation: dict):
        if self._streaming is not None and self._streaming.is_running():
            frame = compose_stream_frame(observation.get("head_stereo_left"), observation.get("head_stereo_right"),
                                         self.stream_status, self.stream_view)
            if frame is not None:
                self._frame_buffer.put(frame)
        return observation

    # ------------------------------------------------------------------ state
    def reset_policy(self) -> None:
        """A fresh start, as after re-launching the robot stack: runtime state, obs history, VR queue, KV cache
        (node_state.fresh_start, shared with the replay's restore so both start from the same node)."""
        n = self.node
        NS.fresh_start(n)
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
        self._quick_engaged = False
        self.quick_engaged_at = 0

    # ------------------------------------------------------------------ quick start
    def quick_start(self, hold_real: np.ndarray | None = None, auto_motion: bool = False) -> None:
        """Skip the leash and the stand-up: the robot is already standing on the floor (the CLI placed it, band off).
        Enters MOVE_TO_DEFAULT at the default pose and presses A (walking policy) for the operator, through the same path
        as the real button; the operator switches to motion tracking (Y) when ready. auto_motion: also press the
        motion-tracking button QUICK_SETTLE_SEC later. quick_ready turns True when the quick start is done (walking on
        and the controllers there; with auto_motion, motion tracking on). hold_real: the (randomized) pose the robot was
        placed in, held until A (default: the policy's default pose)."""
        self._quick = dict(phase="stand", ticks=0, calls=0, warned=False, engaged=False, auto_motion=bool(auto_motion),
                           hold=None if hold_real is None else np.asarray(hold_real, dtype=np.float32).copy())
        self._quick_ready = False
        self._quick_engaged = False
        self.quick_engaged_at = 0

    @property
    def quick_ready(self) -> bool:
        """True once per quick start, when motion tracking has taken over."""
        if self._quick_ready:
            self._quick_ready = False
            return True
        return False

    @property
    def quick_engaged(self) -> bool:
        """True once per quick start, on the first step the PICO controllers are there (the robot still in its start
        pose): the moment an episode that must begin from the start pose starts recording."""
        if self._quick_engaged:
            self._quick_engaged = False
            return True
        return False

    @property
    def quick_active(self) -> bool:
        return self._quick is not None

    def _quick_step(self, s: dict | None, q_real: np.ndarray) -> dict | None:
        """One tick of the quick start. The walking policy goes on at the first tick whether or not the PICO is
        connected yet (a neutral controller sample stands in for it), so the robot balances without the leash from the
        start. It is done once the operator's controllers are there (with auto_motion: once motion tracking is on)."""
        from simple.teleop.holomotion_v14.wire import BUTTONS
        Q, n = self._quick, self.node
        Q["calls"] += 1
        if Q["phase"] == "stand":
            self._set_main_state("MOVE_TO_DEFAULT", q_real)
            self._mtd_from = self.default_real.copy() if Q["hold"] is None else Q["hold"]   # already there: hold it
            Q["phase"], Q["ticks"] = "A", 0
        if self.main_state in ("ZERO_TORQUE", "EMERGENCY_STOP"):   # the operator pressed X / R3: stop helping
            self._quick = None
            return s
        real = s is not None
        if real and not Q["engaged"]:                          # the operator's controllers are there
            Q["engaged"] = self._quick_engaged = True
            self.quick_engaged_at = Q["calls"]
        operator = real and not s.get("left_menu")             # a held sim combo keeps the operator's buttons out
        Q["ticks"] += 1
        press = None
        if Q["phase"] == "A":
            if self.main_state == "POLICY" and n.policy_enabled:
                Q["phase"], Q["ticks"] = ("settle" if Q["auto_motion"] else "walk"), 0
            else:
                s = dict(s) if operator else dict(NEUTRAL_PICO_SAMPLE)
                press = "A" if Q["ticks"] % 4 else None        # 3 ticks pressed, 1 released: a fresh edge each time
        if Q["phase"] == "walk":                               # walking mode; the operator presses Y for motion
            if not real:
                if not Q["warned"]:
                    print("[HoloMotion v1.4] quick start: standing on the walking policy, waiting for the PICO controllers")
                    Q["warned"] = True
                return s
            self._quick = None
            self._quick_ready = True
            print(f"[HoloMotion v1.4] quick start: standing, walking mode ({self.motion_button}: motion tracking)")
            return s
        if Q["phase"] == "settle" and Q["ticks"] * self._control_dt >= QUICK_SETTLE_SEC:
            Q["phase"], Q["ticks"] = "B", 0
        if Q["phase"] == "B":
            if not operator:
                if not real and not Q["warned"]:
                    print("[HoloMotion v1.4] quick start: standing on the walking policy, waiting for the PICO controllers")
                    Q["warned"] = True
                return s
            if n.current_policy_mode == "motion" and n.policy_enabled:
                self._quick = None
                self._quick_ready = True
                print("[HoloMotion v1.4] quick start: standing, motion tracking on")
                return s
            press = self.motion_button if Q["ticks"] % 4 else None
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
    def agent_state(self) -> dict:
        """The agent's own state outside the policy node (main state, button-edge memory, hand targets, band drop): what
        a replay must set besides node_state.restore to continue from this tick (2026-10-18)."""
        arr = lambda v: None if v is None else np.asarray(v, dtype=np.float64).tolist()  # noqa: E731
        return dict(main_state=self.main_state, last_bits=int(self._last_bits), dropping=bool(self._dropping),
                    quick_active=self._quick is not None, left_hand=arr(self._left_hand), right_hand=arr(self._right_hand),
                    hold_real=arr(self._hold_real), mtd_from=arr(self._mtd_from),
                    state_age=float(time.monotonic() - self._state_t0))

    def get_action(self, observation, instruction=None, **kwargs) -> ActionCmd:
        self.tick_agent_state = self.agent_state()             # before this tick's buttons are handled
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
        # the node's internal state at the start of this tick, before any of its inputs (2026-10-16): the recorder keeps
        # the one of the tick a recording starts on, so the controller can be restored and replayed from frame 0
        self.tick_state = NS.snapshot(n)
        tick_time = NS.clock_now(self.tick_state)
        n.feed_lowstate(q_real, dq_real, base[3:7], base_vel[3:6])
        fed = None                                 # the controller sample the node receives this tick (recorded as VRState)
        if s is not None and not s.get("left_menu"):
            s_fwd = dict(s)
            if self.button_map == "sim":          # A walking, Y motion tracking, B free for the PICO app's view toggle
                s_fwd["A"], s_fwd["B"] = int(bool(s.get("A"))), int(bool(s.get("Y")))
            else:                                 # the robot's v1.4.1 map: A walking, B motion tracking, Y back to walking
                s_fwd["A"] = int(bool(s.get("A")) or bool(s.get("Y")))
            fed = s_fwd
        elif s is not None:
            fed = {**s, "A": 0, "B": 0, "X": 0, "Y": 0}                 # left menu held: buttons belong to the CLI
        if fed is not None:
            n.feed_pico(fed)
        for f in frames:
            n.feed_reference(f["latest_obs"], int(f["frame_index"][0]), int(f.get("timestamp_ns", [0])[0]))
        # raw inputs of this tick, for the recorder (2026-10-16): every reference frame received since the last tick,
        # the low state and the controller sample as fed to the node, then the policy's exact ONNX input / output
        raw_frames = [(np.asarray(f["latest_obs"], dtype=np.float32).reshape(-1)[:65].copy(), int(f["frame_index"][0]),
                       int(f.get("timestamp_ns", [0])[0])) for f in frames]
        raw_lowstate = np.concatenate([q_real, dq_real, base[3:7], base_vel[3:6]]).astype(np.float32)
        raw_pico = raw_pico_vector(s if s is not None else None)
        raw_vr = vrstate_vectors(fed)                                  # robo_orchard_pico_msg_ros2/VRState fields, as the node reads them
        n.last_policy_io = None
        t_run = time.perf_counter()
        n.run()
        run_ms = (time.perf_counter() - t_run) * 1e3
        pio = n.last_policy_io
        # the node's 11-frame reference window after this step's frames (2026-10-19): current + future 1..10, 65 each;
        # fed whole every 20 ms it reproduces the model input (holomotion.reference_window)
        vr = n._vr_reference
        if vr is not None and vr.prev_dof_pos is not None and vr.dof_pos_queue is not None:
            cur = np.concatenate([vr.prev_dof_pos, vr.prev_dof_vel, vr.prev_root_pos, vr.prev_root_rot]).astype(np.float32)
            fut = np.concatenate([vr.dof_pos_queue, vr.dof_vel_queue, vr.root_pos_queue, vr.root_rot_queue], axis=1).astype(np.float32)
            raw_window = np.vstack([cur[None, :], fut])
            raw_window_idx = np.concatenate([[-1 if vr.prev_frame_idx is None else int(vr.prev_frame_idx)], vr.frame_idx_queue]).astype(np.int64)
        else:                                                          # queue not filled yet (before the first 11 frames)
            raw_window = np.zeros((11, 65), dtype=np.float32); raw_window_idx = np.full(11, -1, dtype=np.int64)
        port = n.runtime.port                                          # the observation layouts, once each builder has run
        for mode_, b in (("motion", getattr(port, "motion_obs_builder", None)), ("velocity", getattr(port, "velocity_obs_builder", None))):
            if b is not None and mode_ not in n.obs_layouts and getattr(b, "_obs_slices", None):
                n.obs_layouts[mode_] = dict(terms=[(str(nm), int(st), int(en)) for nm, st, en, _h in b._obs_slices],
                                            dof_names_onnx=list(b.dof_names_onnx), dim=int(b._obs_buffer.shape[0]))

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
            # raw HoloMotion inputs (2026-10-16)
            raw_ref_frames=raw_frames,
            raw_lowstate=raw_lowstate,
            raw_pico=raw_pico,
            raw_vrstate=raw_vr,
            raw_policy_obs=None if pio is None else pio["obs"],
            raw_policy_out=None if pio is None else pio["out"],
            raw_policy_mode=None if pio is None else pio["mode"],
            raw_action_real=None if n.action_target is None else np.asarray(n.action_target, dtype=np.float32).copy(),
            raw_kps=np.asarray(n.action_kps, dtype=np.float32).copy(),
            raw_kds=np.asarray(n.action_kds, dtype=np.float32).copy(),
            raw_obs_layouts=n.obs_layouts,
            raw_tick_time=tick_time,
            raw_ref_window=raw_window,
            raw_ref_window_index=raw_window_idx,
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

    The status lines are drawn on the camera image (in both halves for mono/stereo). Built in one preallocated frame
    with cv2 colour conversion (~1 ms at 2560 x 720; numpy channel reversal took 6)."""
    import cv2

    if left is None:
        return None
    left = np.ascontiguousarray(np.asarray(left)[..., :3])
    h, w = left.shape[:2]
    frame = np.zeros((h, 2 * w, 3), dtype=np.uint8)
    if view == "single":
        frame[:, w // 2:w // 2 + w] = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        offsets = [w // 2]
    else:
        frame[:, :w] = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        second = right if (view == "stereo" and right is not None) else left
        frame[:, w:] = cv2.cvtColor(np.ascontiguousarray(np.asarray(second)[..., :3]), cv2.COLOR_RGB2BGR)
        offsets = [0, w]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, h / 720.0)
    x0, y = int(0.12 * w), int(0.12 * h)                          # inset: the headset crops the image edges
    for text, color in status:
        for off in offsets:
            cv2.putText(frame, text, (off + x0, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (off + x0, y), font, scale, color, 1, cv2.LINE_AA)
        y += int(22 * scale / 0.5)
    return frame
