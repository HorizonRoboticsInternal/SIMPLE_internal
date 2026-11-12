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

import os
import math
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
QUICK_HOLD_ARM_KP = 150.0          # quick_start(hold_pose): the arms kept at the placed pose under this PD stiffness in place of the walking
                                   # policy's arm actions during the ~0.3 s of walking mode before motion tracking takes the pose (sim only, 2026-11-03)
QUICK_HOLD_FILL = 12               # reference frames carrying the held pose fed before motion tracking is switched on (the node's window is 11)
QUICK_HOLD_LIVE_S = 0.5            # a held start hands over to the operator only while the headset's body frames are this fresh
QUICK_HOLD_FADE_S = 0.5            # the hand-over from the held pose to the operator's own pose after their motion button
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



def _quat_yaw(q) -> float:
    """Yaw of a wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul(a, b) -> np.ndarray:
    """wxyz quaternion product a * b."""
    aw, ax, ay, az = (float(v) for v in a); bw, bx, by, bz = (float(v) for v in b)
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dtype=np.float32)


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
        hand_source: str = "grip",
        hand_sides=("left", "right"),
        hand_calib_dir: str = "",
        hand_min_conf: float = 0.0,
        hand_smooth_hz: float = 10.0,
        hand_pinch_blend: bool = True,
        hand_calib: dict | None = None,
    ) -> None:
        super().__init__(robot)
        # 2026-11-03: --hand-source skeleton = the headset's hand tracking drives the Revo 2 fingers (hand_skeleton.py) instead of
        # the grip / trigger poses; the controllers may sleep, so pico() serves a neutral sample while hands arrive, and
        # pico_overlay (pico_keys.PicoKeys) lays the keyboard's buttons / the auto cycle's presses over it
        if hand_source not in ("grip", "skeleton"):
            raise ValueError(f"hand_source is 'grip' or 'skeleton', got {hand_source!r}")
        self.hand_source = hand_source
        self.pico_overlay = None
        self._skel = {}
        self.hand_meta = None
        if hand_source == "skeleton":
            from simple.teleop.holomotion_v14.hand_skeleton import build_hands
            self._skel = build_hands(tuple(hand_sides), hand_calib_dir, hand_min_conf, hand_smooth_hz, hand_pinch_blend, calib=hand_calib)
            from simple.teleop.holomotion_v14.hand_skeleton import hands_meta
            self.hand_meta = hands_meta(self._skel, hand_min_conf, hand_smooth_hz, hand_pinch_blend)
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
        self._arm_real_idx = np.array([i for i, n in enumerate(real) if any(k in n for k in ("shoulder", "elbow", "wrist"))], dtype=np.int64)
        self._real_names = list(real)
        self._arm_pin = None                                   # quick_start(pin_real_idx): joints kept pinned through the held start (A3's holding arm)
        ref_names = list(y.get("complete_dof_order", real))                                # the reference frames' joint order (publisher)
        self._real_to_ref = np.array([real.index(n) for n in ref_names], dtype=np.int64)  # ref[i] = real[map[i]]

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
                                        fps=fps, bitrate=int(os.environ.get("HOLOMOTION_STREAM_BITRATE", req.get("bitrate") or 4_000_000)),   # the app asks for 4 Mbps; the env overrides (Wi-Fi)
                                        hevc=bool(req.get("enableMvHevc")))
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

    # ------------------------------------------------------------------ hands
    def _is_revo2(self) -> bool:
        """G1SonicRevo2 (11 hand joints): the same PICO grip bit picks the Revo 2 open / closed pose
        (simple.robots.g1_sonic_revo2, data/robots/g1_rot0_teleop_revo2/grip_poses.json) instead of the Dex3 poses."""
        return int(getattr(self.robot, "num_hand_dof", 7)) == 11

    def _hand_open(self) -> np.ndarray:
        if self._is_revo2():
            from simple.robots.g1_sonic_revo2 import OPEN_Q
            return np.asarray(OPEN_Q, dtype=np.float32).copy()
        return np.zeros(7, dtype=np.float32)

    def _hand_target(self, side: str, grip: float, trigger: float = 0.0) -> np.ndarray:
        if self._is_revo2():                                   # grip button -> closed pose; trigger (index finger) -> the pinch pose (close2)
            from simple.robots.g1_sonic_revo2 import revo2_targets
            return revo2_targets(grip, trigger=trigger)
        return np.asarray(self._grip_q(side=side, status=1, grip=grip), dtype=np.float32)[DEX3_NATURAL_TO_MJCF]

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
        self._arm_hold = None
        self._arm_pin = None
        self._pose_hold = None
        self._ref_shift = None
        self._handover_started = False
        self._handover_blocked_t = 0.0
        self._last_bits = 0
        self._dropping = False
        self._reset_time = time.monotonic()
        self._left_hand = self._hand_open()
        self._right_hand = self._hand_open()
        self._last = {}
        self._quick = None
        self._quick_ready = False
        self._quick_engaged = False
        self.quick_engaged_at = 0

    # ------------------------------------------------------------------ quick start
    def arm_real_idx(self, side: str) -> np.ndarray:
        """The seven arm joints of one side as indices into the real joint order (quick_start's pin_real_idx)."""
        return np.array([i for i, n in enumerate(self._real_names) if n.startswith(f"{side}_") and any(k in n for k in ("shoulder", "elbow", "wrist"))], dtype=np.int64)

    def quick_start(self, hold_real: np.ndarray | None = None, auto_motion: bool = False, hold_pose: bool = False, pin_real_idx=None) -> None:
        """Skip the leash and the stand-up: the robot is already standing on the floor (the CLI placed it, band off).
        Enters MOVE_TO_DEFAULT at the default pose and presses A (walking policy) for the operator, through the same path
        as the real button; the operator switches to motion tracking (Y) when ready. auto_motion: also press the
        motion-tracking button QUICK_SETTLE_SEC later. quick_ready turns True when the quick start is done (walking on
        and the controllers there; with auto_motion, motion tracking on). hold_real: the (randomized) pose the robot was
        placed in, held until A (default: the policy's default pose).
        hold_pose (2026-11-03, the A2 start -- the robot must not touch anything at the beginning): the placed pose is kept
        until the operator takes over. The walking policy would pull the arms to its own default and sink the stance (arms
        reaching over the worktop went through it within 0.1 s), and neither a PD hold nor the walking policy with the arms
        pinned stands up (both fell within 2-6 s). So: the walking policy for QUICK_HOLD_FILL frames with the arms pinned at
        the placed targets (QUICK_HOLD_ARM_KP), while the reference frames fed to the node carry the placed joints (root as
        received); then motion tracking is switched on by the quick start itself and the motion policy holds the pose on its
        own balance. The operator's motion button (or the auto cycle's) starts a QUICK_HOLD_FADE_S hand-over from the held
        joints to the received ones; then the operator is in control. The frames are logged as fed and the arm pin is in
        agent_state (arm_hold), so a replay reproduces the hold from a frame 0 inside it. No body frame from the headset
        in a tick: the hold feeds its own (held joints, frozen root), so the robot stands whether or not the headset streams
        (11-04 12:22 live: body tracking at 0 Hz; the hold used to give up after 1 s into walking mode at the counter, and the
        robot toppled); the hand-over waits until body frames are QUICK_HOLD_LIVE_S fresh.
        pin_real_idx (2026-11-05, A3's level hold: the plate carried as in A2 ep41 with the arm raised ahead): these joints stay
        pinned at the placed targets (QUICK_HOLD_ARM_KP) THROUGH the motion-tracking hold -- the teleop's arm PD is 14 N m/rad and
        the motion policy holds a loaded arm 5-10 deg below its reference, which tips the held plate --, then fade to the policy's
        own targets and gains over the QUICK_HOLD_FADE_S hand-over. In agent_state as arm_pin."""
        self._quick = dict(phase="stand", ticks=0, calls=0, warned=False, engaged=False, auto_motion=bool(auto_motion),
                           hold=None if hold_real is None else np.asarray(hold_real, dtype=np.float32).copy(),
                           hold_pose=bool(hold_pose and hold_real is not None), last_motion=0, ready_told=False)
        self._arm_hold = np.asarray(hold_real, dtype=np.float32).copy() if (hold_pose and hold_real is not None) else None
        self._arm_pin = np.asarray(pin_real_idx, dtype=np.int64).copy() if (self._arm_hold is not None and pin_real_idx is not None and len(pin_real_idx)) else None
        self._pose_hold = (dict(q=np.asarray(hold_real, dtype=np.float32)[self._real_to_ref].copy(), fed=0, fade_t0=None, root=None, synth=0)
                           if (hold_pose and hold_real is not None) else None)
        self._ref_shift = None
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

    @property
    def handover_started(self) -> bool:
        """True once, on the tick a held start's hand-over to the operator begins (the recording's cue in place of the raw button)."""
        if self._handover_started:
            self._handover_started = False
            return True
        return False

    @property
    def handover_blocked(self) -> bool:
        """The operator's motion button was refused within the last 2 s because the headset streams no body frames."""
        return time.monotonic() - self._handover_blocked_t < 2.0

    @property
    def quick_holding(self) -> bool:
        """The placed start pose is still held (quick_start hold_pose: arms pinned, or motion tracking on the held frames)."""
        return self._arm_hold is not None or self._pose_hold is not None

    def _held_frames(self, frames: list) -> list:
        """quick_start(hold_pose): this tick's reference frames with the held pose as their joints and the root frozen at the first
        received root, upright (its heading only; index and stamp as received), so the robot neither moves nor leans after the
        operator before the hand-over.
        After the operator's motion button a QUICK_HOLD_FADE_S blend from the held joints to the received ones, and from then on
        the received root is re-based onto the frozen one (position and heading offset at the hand-over removed: _ref_shift), so
        the node sees a continuous root and keeps its yaw alignment; then the hold ends and the frames pass through the shift."""
        H = self._pose_hold
        k = len(H["q"])
        if not frames and H["fade_t0"] is None:                # no body frame from the headset this tick: the hold makes its own, so the
            if H["root"] is None:                              # robot stands whether or not the headset streams (11-04 12:22 live: body
                H["root"] = np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)   # tracking at 0 Hz -> the hold gave up -> walking
            H["synth"] += 1                                    # mode at the counter -> the robot toppled)
            obs = np.concatenate([H["q"], np.zeros(k, dtype=np.float32), H["root"]]).astype(np.float32)
            frames = [dict(latest_obs=obs, frame_index=np.array([self.sub_last_index() + H["synth"]], dtype=np.int64),
                           timestamp_ns=np.array([time.time_ns()], dtype=np.int64))]
        a = 0.0 if H["fade_t0"] is None else min(1.0, (time.monotonic() - H["fade_t0"]) / QUICK_HOLD_FADE_S)
        out = []
        for f in frames:
            obs = np.asarray(f["latest_obs"], dtype=np.float32).reshape(-1).copy()
            if H["root"] is None:                              # frozen root: the first received position, its heading alone (an operator
                H["root"] = obs[58:65].copy()                  # leaning over the keyboard would pitch the robot into the worktop, 11-04 live)
                yaw = _quat_yaw(H["root"][3:7])
                H["root"][3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            obs[:k] = (1.0 - a) * H["q"] + a * obs[:k]
            obs[k:2 * k] *= a
            if H["fade_t0"] is None:
                obs[58:65] = H["root"]
            else:
                if self._ref_shift is None:                    # the hand-over: re-base the live root onto the frozen one from here on
                    self._ref_shift = dict(p0=obs[58:61].copy(), base=H["root"][:3].copy(),
                                           dyaw=_quat_yaw(H["root"][3:7]) - _quat_yaw(obs[61:65]))
                obs = self._shift_frame(obs)
            out.append({**f, "latest_obs": obs})
        H["fed"] += len(out)
        if a >= 1.0:
            self._pose_hold = None
            print("[HoloMotion v1.4] quick start: hand-over done, the operator is in control")
        return out

    def sub_last_index(self) -> int:
        """The last body frame index received from the headset (0 before any)."""
        v = getattr(self.sub, "last_frame_index", None)
        return int(v) if v is not None else 0

    def _shift_frame(self, obs: np.ndarray) -> np.ndarray:
        """The received root re-based onto the held start's frozen root (see _held_frames): position deltas since the hand-over
        turned by the heading offset and added to the frozen position, the rotation turned by the same offset."""
        S = self._ref_shift
        c, s_ = math.cos(S["dyaw"]), math.sin(S["dyaw"])
        d = obs[58:61] - S["p0"]
        obs[58:61] = S["base"] + np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]], dtype=np.float32)
        obs[61:65] = _quat_mul(np.array([math.cos(S["dyaw"] / 2), 0.0, 0.0, math.sin(S["dyaw"] / 2)], dtype=np.float32), obs[61:65])
        return obs

    def _release_hold(self, why: str) -> None:
        self._arm_hold = self._arm_pin = self._pose_hold = self._ref_shift = None
        if self._quick is not None:
            self._quick["hold_pose"] = False
        print(f"[HoloMotion v1.4] quick start: start pose released ({why}); walking mode as usual")

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
        if Q["hold_pose"] and Q["phase"] == "B" and Q["ticks"] == int(5.0 / self._control_dt):
            print("[HoloMotion v1.4] quick start: motion tracking has not taken the held pose after 5 s; still trying (the hold feeds its own frames)")
        if Q["phase"] == "A":
            if self.main_state == "POLICY" and n.policy_enabled:
                Q["phase"], Q["ticks"] = ("fill" if Q["hold_pose"] else "settle" if Q["auto_motion"] else "walk"), 0
            else:
                s = dict(s) if operator else dict(NEUTRAL_PICO_SAMPLE)
                press = "A" if Q["ticks"] % 4 else None        # 3 ticks pressed, 1 released: a fresh edge each time
        if Q["phase"] == "fill":                               # held frames into the node's window, then motion tracking
            if self._pose_hold is not None and self._pose_hold["fed"] >= QUICK_HOLD_FILL:
                Q["phase"], Q["ticks"] = "B", 0
            else:
                return s
        if Q["phase"] == "motion_hold":                        # motion tracking holds the placed pose; the operator's button hands over
            if n.current_policy_mode != "motion" or not n.policy_enabled:
                self._release_hold("motion tracking went off")
                Q["phase"], Q["ticks"] = "walk", 0
            else:
                if real and not Q["ready_told"]:
                    Q["ready_told"] = self._quick_ready = True
                    print(f"[HoloMotion v1.4] quick start: standing, start pose held by motion tracking ({self.motion_button}: take over)")
                pressed = int(bool(operator and s.get(self.motion_button)))
                if operator:
                    s = dict(s); s[self.motion_button] = 0    # the button is ours here (sim map: the node would ignore it; robot map: walking)
                    s["button_bits"] = int(s.get("button_bits", 0)) & ~BUTTONS[self.motion_button]
                take = (pressed and not Q["last_motion"]) or (Q["auto_motion"] and Q["ticks"] * self._control_dt >= QUICK_SETTLE_SEC)
                Q["last_motion"] = pressed
                if take and self.sub.obs_age() > QUICK_HOLD_LIVE_S:   # no body frames from the headset: nothing to hand over to
                    take = False
                    self._handover_blocked_t = time.monotonic()
                    if time.monotonic() - Q.get("nolive_told", 0.0) > 2.0:
                        Q["nolive_told"] = time.monotonic()
                        print("[HoloMotion v1.4] quick start: the headset's body tracking is not streaming -- the hand-over waits for it (the robot keeps the start pose)")
                if take and self._pose_hold is not None:
                    self._pose_hold["fade_t0"] = time.monotonic()
                    self._handover_started = True
                    self._quick = None
                    print(f"[HoloMotion v1.4] quick start: the operator takes over ({QUICK_HOLD_FADE_S:g} s hand-over from the start pose)")
                return s
        if Q["phase"] == "walk":                               # walking mode; the operator presses Y for motion
            if not real:
                if not Q["warned"]:
                    print("[HoloMotion v1.4] quick start: standing on the walking policy, waiting for the PICO controllers")
                    Q["warned"] = True
                return s
            self._quick = None
            self._quick_ready = True
            print(f"[HoloMotion v1.4] quick start: standing, walking mode{' (arms held at the start pose)' if self._arm_hold is not None else ''} "
                  f"({self.motion_button}: motion tracking)")
            return s
        if Q["phase"] == "settle" and Q["ticks"] * self._control_dt >= QUICK_SETTLE_SEC:
            Q["phase"], Q["ticks"] = "B", 0
        if Q["phase"] == "B":
            if not operator and not Q["hold_pose"]:
                if not real and not Q["warned"]:
                    print("[HoloMotion v1.4] quick start: standing on the walking policy, waiting for the PICO controllers")
                    Q["warned"] = True
                return s
            if n.current_policy_mode == "motion" and n.policy_enabled:
                if Q["hold_pose"]:                             # the motion policy has the held frames: it keeps the pose from here
                    Q["phase"], Q["ticks"] = "motion_hold", 0
                    return s
                self._quick = None
                self._quick_ready = True
                print("[HoloMotion v1.4] quick start: standing, motion tracking on")
                return s
            if not operator:
                s = dict(NEUTRAL_PICO_SAMPLE)                  # (hold_pose: the held frames fill the window; no operator needed)
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
        """The newest controller sample, or None if older than the adapter's 0.5 s timeout. With the skeleton hand source a
        neutral sample stands in while the controllers sleep and hand frames arrive (the quick start waits for an operator);
        pico_overlay (keyboard / auto cycle) presses its buttons on top either way."""
        s = self.sub.last_pico
        s = s if s is not None and self.sub.pico_age() <= PICO_TIMEOUT_SEC else None
        if s is None and self.hand_source == "skeleton" and self.sub.hands_age() <= PICO_TIMEOUT_SEC:
            from simple.teleop.holomotion_v14.wire import BUTTONS
            now = time.monotonic()
            s = dict({n: 0 for n in BUTTONS}, button_bits=0, left_axis=(0.0, 0.0), right_axis=(0.0, 0.0), left_trigger=0.0, right_trigger=0.0,
                     left_grip=0.0, right_grip=0.0, l_active=1, r_active=1, timestamp_ns=time.time_ns(), received_monotonic=now)
        if self.pico_overlay is not None:
            s = self.pico_overlay.apply(s)
        return s

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
                    hold_real=arr(self._hold_real), mtd_from=arr(self._mtd_from), arm_hold=arr(self._arm_hold),
                    arm_pin=None if self._arm_pin is None else [int(i) for i in self._arm_pin],
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
        if self._pose_hold is not None:
            frames = self._held_frames(frames)                 # the held start pose as the reference's joints (quick_start hold_pose)
        elif self._ref_shift is not None:
            frames = [{**f, "latest_obs": self._shift_frame(np.asarray(f["latest_obs"], dtype=np.float32).reshape(-1).copy())} for f in frames]
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
            if self._arm_hold is not None:                     # quick_start(hold_pose): the arms pinned at the placed pose until motion tracking;
                motion_on = n.current_policy_mode == "motion" and n.policy_enabled   # pin_real_idx through the hold, fading out over the hand-over
                w = 1.0
                if motion_on and self._arm_pin is None:
                    self._arm_hold = None                      # the operator's arms from here
                elif motion_on:
                    H = self._pose_hold
                    w = 1.0 if (H is not None and H["fade_t0"] is None) else (max(0.0, 1.0 - (time.monotonic() - H["fade_t0"]) / QUICK_HOLD_FADE_S) if H is not None else 0.0)
                    if w <= 0.0:
                        self._arm_hold = self._arm_pin = None  # hand-over done: the operator's arm from here
                if self._arm_hold is not None:
                    a = self._arm_real_idx if not motion_on else self._arm_pin
                    target, kp, kd = target.copy(), kp.copy(), kd.copy()
                    target[a] = w * self._arm_hold[a] + (1.0 - w) * target[a]
                    kp[a] = w * np.maximum(kp[a], QUICK_HOLD_ARM_KP) + (1.0 - w) * kp[a]
                    kd[a] = w * np.maximum(kd[a], self.mtd_kd_real[a]) + (1.0 - w) * kd[a]
        elif self.main_state == "EMERGENCY_STOP" and t < ESTOP_DAMPING_SEC:
            target, kp, kd = q_real, np.zeros_like(q_real), self.mtd_kd_real
        else:                                              # ZERO_TORQUE, or E-stop after its damping phase
            target, kp, kd = q_real, np.zeros_like(q_real), np.zeros_like(q_real)

        # hands: the v1.4 grip gripper (Dex3 poses, or the Revo 2 grip poses on the Revo 2 robot), independent of the body state
        if s is not None:
            self._left_hand = self._hand_target("left", s["left_grip"], s.get("left_trigger", 0.0))
            self._right_hand = self._hand_target("right", s["right_grip"], s.get("right_trigger", 0.0))
        hands = self.sub.last_hands if self.hand_source == "skeleton" else None
        if self._skel and self._is_revo2():                                  # the headset's hand skeleton overrides the grip poses (hold policy inside)
            now_h = time.monotonic()
            for side, hand in self._skel.items():
                i = 0 if side == "left" else 1
                q = hand.update(None if hands is None else hands["hands"][i], 0 if hands is None else int(hands["active"][i]), now_h)
                if q is not None:
                    setattr(self, f"_{side}_hand", q.copy())
        from simple.teleop.holomotion_v14.hand_skeleton import hands_vector
        raw_pico_hands = hands_vector(hands) if self.hand_source == "skeleton" else None

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
            raw_pico_hands=raw_pico_hands,
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
    for line in status:
        text, color = line[0], line[1]
        k = float(line[2]) if len(line) > 2 else 1.0                      # (text, colour, scale): the auto cycle's countdown digit is 4x
        sc = scale * k
        for off in offsets:
            cv2.putText(frame, text, (off + x0, y + int((k - 1) * 22 * scale / 0.5)), font, sc, (0, 0, 0), max(3, int(3 * k)), cv2.LINE_AA)
            cv2.putText(frame, text, (off + x0, y + int((k - 1) * 22 * scale / 0.5)), font, sc, color, max(1, int(k)), cv2.LINE_AA)
        y += int(22 * scale * k / 0.5)
    return frame
