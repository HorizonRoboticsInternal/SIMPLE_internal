"""The recorded target_yaw must be RELATIVE to the heading at the A press (0 at episode start),
never the robot's absolute heading in the room.

Regression for the 2026-09-11/14 real recordings: G1GearWbcPolicy computed `target_yaw_rel`, but
G1DecoupledWholeBodyPolicy (the wrapper the control loop calls) rebuilt the action dict without
that key, so `_recorded_navigate_cmd` silently fell back to the absolute re-anchored dial and every
episode started at the robot's heading (-0.33 rad etc.) instead of 0.

Closed loop in MuJoCo with the real stack (TeleopPolicy -> decoupled WBC -> PD), the Pico replaced
by a scripted operator that can push the yaw stick:
  1. turn the robot ~1 rad with the stick, release the stick
  2. dispatch "c" (= Pico A, start recording) through the same wbc_policy.handle_keyboard_button
     the control loop's dispatcher uses -> recorded target_yaw must be 0 although heading != 0
  3. turn a bit more -> recorded target_yaw grows from 0 by the extra dial motion (relative)
  4. "c" again (save / next episode) -> recorded target_yaw back to 0

Run:  .venv/bin/python -m pytest tests/test_target_yaw_relative.py -v
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decoupled_wbc.control.utils.navigate_cmd import recorded_navigate_cmd as _recorded_navigate_cmd  # noqa: E402
from decoupled_wbc.control.teleop.streamers.base_streamer import StreamerOutput  # noqa: E402
from test_soft_reset import FakeOperator  # noqa: E402

HZ = 50


# ----------------------------------------------------------------------------- pure
def test_recorded_cmd_uses_relative_yaw_when_present():
    teleop = {"navigate_cmd": [0.1, 0.0, 1.0, 2.3]}                 # streamer: absolute dial 2.3
    wbc = {"navigate_cmd": np.array([0.1, 0.0, 1.0, 2.25]), "target_yaw_rel": 0.0}
    nav = _recorded_navigate_cmd(teleop, wbc)
    assert nav[:3] == [0.1, 0.0, 1.0] and nav[3] == 0.0


def test_recorded_cmd_fallback_is_the_absolute_dial():
    """Documents the trap: without target_yaw_rel the recorder writes the absolute dial."""
    teleop = {"navigate_cmd": [0.0, 0.0, 0.0, 2.3]}
    wbc = {"navigate_cmd": np.array([0.0, 0.0, 0.0, -0.33])}
    assert _recorded_navigate_cmd(teleop, wbc)[3] == pytest.approx(-0.33)


# ----------------------------------------------------------------------------- sim
class SteeringOperator(FakeOperator):
    """FakeOperator plus a yaw stick. Integrates target_yaw like PicoStreamer (50 Hz).
    wrap=True mimics the OLD streamer (dial wrapped to [-pi, pi]) to prove the policy-side guards."""

    def __init__(self, wrap: bool = False):
        super().__init__()
        self.vyaw = 0.0
        self.target_yaw = 0.0
        self.wrap = wrap

    def get(self) -> StreamerOutput:
        out = super().get()
        self.target_yaw = self.target_yaw + self.vyaw / HZ
        if self.wrap:
            self.target_yaw = float(np.arctan2(np.sin(self.target_yaw), np.cos(self.target_yaw)))
        out.control_data["navigate_cmd"] = [0.0, 0.0, self.vyaw, self.target_yaw]
        return out


def test_streamer_dial_is_continuous():
    from decoupled_wbc.control.teleop.streamers.pico_streamer import integrate_yaw_dial
    d = 0.0
    for _ in range(int(8 * HZ)):                       # 8 s at 1 rad/s: crosses pi
        d = integrate_yaw_dial(d, 1.0, 1.0 / HZ)
    assert d == pytest.approx(8.0, abs=1e-6), "the dial must not wrap"


def _heading(robot):
    from scipy.spatial.transform import Rotation as R
    q = robot.mjData.qpos[3:7]                       # MuJoCo free joint: w, x, y, z
    return float(R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])


@pytest.mark.sim
def test_recorded_target_yaw_is_relative_in_mujoco():
    import gymnasium as gym
    import simple.envs as _  # noqa: F401
    import decoupled_wbc.control.teleop.streamers.pico_streamer as pico_mod
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    fake = SteeringOperator()
    pico_mod.PicoStreamer = lambda *a, **k: fake
    cfg = SimLoopConfig().load_wbc_yaml(); cfg["ENV_NAME"] = "simple"
    env = gym.make("simple/G1WholebodyXMovePickTeleop-v0", sim_mode="mujoco", render_hz=HZ,
                   physics_dt=cfg["SIMULATE_DT"], headless=True, max_episode_steps=10**6,
                   sonic_config=cfg, target="graspnet1b:0", dr_level=0, success_criteria=1e9)
    try:
        task = env.unwrapped.task; robot = task.robot
        agent = PicoDecoupledAgent(robot)
        agent._poll_pico_buttons = lambda: None
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        wbc = agent._wbc_policy                                   # G1DecoupledWholeBodyPolicy, as on the robot
        gear = wbc.lower_body_policy                              # G1GearWbcPolicy

        last = {}
        orig_get_action = wbc.get_action
        def capture(*a, **k):
            out = orig_get_action(*a, **k); last["wbc_action"] = out; return out
        wbc.get_action = capture

        obs, info = env.reset()
        if robot.elastic_band is not None:
            robot.elastic_band.enable = False
        agent._dropping = False
        agent.reset_policy()
        agent._wbc_policy.lower_body_policy.use_policy_action = True

        recorded = []                                             # (t, phase, heading, recorded nav[3], abs dial, rel)
        t_start = time.monotonic(); tick = 0; phase = "settle"; phase_t0 = 0.0
        def rec_now():
            wa = last.get("wbc_action", {}); ta = getattr(agent, "_last_teleop_action", None) or {}
            nav = _recorded_navigate_cmd(ta, wa)
            return float(nav[3]), float(np.asarray(wa.get("navigate_cmd", [0, 0, 0, np.nan]))[3]), wa.get("target_yaw_rel", None)
        plan = {
            "settle":   (lambda: None,                              lambda t: robot.stabilized and t > 2.0, "activate"),
            "activate": (lambda: fake.press("activate"),            lambda t: t - phase_t0 > 3.0,           "turn1"),
            "turn1":    (lambda: setattr(fake, "vyaw", 0.5),        lambda t: t - phase_t0 > 3.0,           "coast1"),
            "coast1":   (lambda: setattr(fake, "vyaw", 0.0),        lambda t: t - phase_t0 > 2.0,           "pressA1"),
            "pressA1":  (lambda: wbc.handle_keyboard_button("c"),   lambda t: t - phase_t0 > 1.0,           "turn2"),
            "turn2":    (lambda: setattr(fake, "vyaw", 0.5),        lambda t: t - phase_t0 > 1.0,           "coast2"),
            "coast2":   (lambda: setattr(fake, "vyaw", 0.0),        lambda t: t - phase_t0 > 1.5,           "pressA2"),
            "pressA2":  (lambda: wbc.handle_keyboard_button("c"),   lambda t: t - phase_t0 > 1.0,           None),
        }
        plan[phase][0]()
        while phase is not None:
            t = time.monotonic() - t_start
            assert t < 60, "timeout"
            _, done, nxt = plan[phase]
            if done(t):
                phase, phase_t0 = nxt, t
                if phase is None:
                    break
                plan[phase][0]()
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=info)
            obs, _r, _te, _tr, info = env.step(action)
            r, absd, rel = rec_now()
            recorded.append((t, phase, _heading(robot), r, absd, rel))
            tick += 1
            sl = t_start + (tick / HZ) - time.monotonic()
            if sl > 0:
                time.sleep(sl)
    finally:
        env.close()

    P = np.array([r[1] for r in recorded]); H = np.array([r[2] for r in recorded]); REC = np.array([r[3] for r in recorded])
    ABS = np.array([r[4] for r in recorded]); REL = [r[5] for r in recorded]

    # the wrapper must forward the relative dial at all (this is the bug of 2026-09-11/14)
    assert all(v is not None for v in REL[-10:]), "G1DecoupledWholeBodyPolicy.get_action does not return target_yaw_rel"

    # 1. the robot really turned before the press, so absolute != relative is testable
    h_press1 = H[P == "pressA1"][0]
    assert abs(h_press1) > 0.3, f"robot did not turn enough before the press (heading {h_press1:.2f})"

    # 2. after "c" (Pico A) the RECORDED target_yaw is 0 although the heading is not
    w = (P == "pressA1"); rec1 = REC[w][2:]                        # allow the press tick + 1 to settle
    assert np.all(np.abs(rec1) < 0.02), f"recorded target_yaw after A is not 0: {rec1[:5]} (heading {h_press1:+.2f})"
    assert np.all(np.abs(ABS[w][2:] - h_press1) < 0.15), "absolute dial should equal the heading right after the re-arm"

    # 3. more stick input -> recorded value grows from 0 by the extra turn (relative), not to the heading
    extra = 0.5 * 1.0                                               # 0.5 rad/s for 1 s of stick
    rec2 = REC[P == "coast2"][-5:]
    assert np.all(rec2 > 0.2) and np.all(rec2 < extra + 0.15), f"relative dial after the second turn: {rec2}"
    assert np.all(np.abs(rec2 - ABS[P == "coast2"][-5:]) > 0.2), "recorded value collapsed back to the absolute dial"

    # 4. "c" again -> back to 0 (next episode starts at 0, no accumulation)
    rec3 = REC[P == "pressA2"][2:]
    assert np.all(np.abs(rec3) < 0.02), f"recorded target_yaw after the second A is not 0: {rec3[:5]}"


def _run_wrap_crossing(wrap_fake: bool):
    """After A, hold the yaw stick 5 s at 1 rad/s (dial passes pi): the recorded relative yaw must move
    at the stick rate only, and the robot must never turn the wrong way. Regression for the 2026-09-17
    sweeps (ep 4/5/9/12): dial wrapped at +-pi -> 2pi step -> interpolated into a ~1 s sweep."""
    import gymnasium as gym
    import simple.envs as _  # noqa: F401
    import decoupled_wbc.control.teleop.streamers.pico_streamer as pico_mod
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    fake = SteeringOperator(wrap=wrap_fake); pico_mod.PicoStreamer = lambda *a, **k: fake
    cfg = SimLoopConfig().load_wbc_yaml(); cfg["ENV_NAME"] = "simple"
    env = gym.make("simple/G1WholebodyXMovePickTeleop-v0", sim_mode="mujoco", render_hz=HZ, physics_dt=cfg["SIMULATE_DT"],
                   headless=True, max_episode_steps=10**6, sonic_config=cfg, target="graspnet1b:0", dr_level=0, success_criteria=1e9)
    try:
        task = env.unwrapped.task; robot = task.robot
        agent = PicoDecoupledAgent(robot); agent._poll_pico_buttons = lambda: None
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        wbc = agent._wbc_policy; last = {}
        # the robot deploy runs the command interpolator with --upper-body-joint-speed 5: that rate limit is
        # what turned a 2pi dial step into a ~1 s sweep; use the same value here or the bug cannot show
        wbc.upper_body_policy.max_change_rate = 5.0
        orig = wbc.get_action
        def capture(*a, **k):
            out = orig(*a, **k); last["wbc_action"] = out; return out
        wbc.get_action = capture
        obs, info = env.reset()
        if robot.elastic_band is not None: robot.elastic_band.enable = False
        agent._dropping = False; agent.reset_policy(); agent._wbc_policy.lower_body_policy.use_policy_action = True
        rec, head = [], []
        t_start = time.monotonic(); tick = 0; phase = "settle"; phase_t0 = 0.0
        plan = {"settle":   (lambda: None,                            lambda t: robot.stabilized and t > 2.0, "activate"),
                "activate": (lambda: fake.press("activate"),          lambda t: t - phase_t0 > 3.0,           "pressA"),
                "pressA":   (lambda: wbc.handle_keyboard_button("c"), lambda t: t - phase_t0 > 0.5,           "turn"),
                "turn":     (lambda: setattr(fake, "vyaw", 1.0),      lambda t: t - phase_t0 > 5.0,           "coast"),
                "coast":    (lambda: setattr(fake, "vyaw", 0.0),      lambda t: t - phase_t0 > 1.0,           None)}
        plan[phase][0]()
        while phase is not None:
            t = time.monotonic() - t_start; assert t < 60
            _, done, nxt = plan[phase]
            if done(t):
                phase, phase_t0 = nxt, t
                if phase is None: break
                plan[phase][0]()
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=info)
            obs, _r, _te, _tr, info = env.step(action)
            if phase in ("turn", "coast"):
                wa = last["wbc_action"]; ta = getattr(agent, "_last_teleop_action", None) or {}
                rec.append(float(_recorded_navigate_cmd(ta, wa)[3])); head.append(_heading(robot))
            tick += 1; sl = t_start + (tick / HZ) - time.monotonic()
            if sl > 0: time.sleep(sl)
    finally:
        env.close()
    # the RECORDED value is a wrapped angle in [-pi, pi] (it legitimately steps by 2pi once the commanded
    # turn passes 180 deg); unwrap it, then it must move at the stick rate only (a smeared 2pi shows as
    # ~0.16 rad/tick steps and fails here)
    rec = np.unwrap(np.array(rec)); head = np.unwrap(np.array(head))
    d = np.diff(rec)
    assert np.abs(d).max() < 0.05, f"recorded target_yaw stepped by {np.abs(d).max():.3f} rad in one tick (sweep / wrap glitch)"
    assert rec[-1] > 3.5, f"dial should be ~5 rad after 5 s at 1 rad/s, got {rec[-1]:.2f}"      # crossed pi
    dh = np.diff(head)
    assert dh.min() > -0.02, f"robot turned the wrong way ({dh.min():.3f} rad/tick): the target swept"
    assert head[-1] - head[0] > 2.5, f"robot did not follow the dial (turned {head[-1]-head[0]:.2f} rad)"


@pytest.mark.sim
def test_dial_crossing_pi_no_sweep_in_mujoco():
    _run_wrap_crossing(wrap_fake=False)


@pytest.mark.sim
def test_old_wrapping_streamer_is_guarded_in_mujoco():
    _run_wrap_crossing(wrap_fake=True)
