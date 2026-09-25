"""Unit tests for the `home_randomize` teleop option (randomized home/parked pose).

Two layers:
  1. pure tests of `sample_home_offset` (no simulator, milliseconds)
  2. a closed-loop MuJoCo test running the real stack (TeleopPolicy -> retargeting IK ->
     decoupled WBC -> PD) with a scripted operator: activate -> parked, A (release), A (save)
     -> parked, A (release), B (discard) -> parked. Checks every parked pose is a small,
     bounded, *different* perturbation of the canonical pose and that release is jump-free.

Run:  .venv/bin/python -m pytest tests/test_home_randomize.py -v          (all)
      .venv/bin/python -m pytest tests/test_home_randomize.py -v -m "not sim"   (fast only)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for test_soft_reset.FakeOperator

from decoupled_wbc.control.policy.teleop_policy import (  # noqa: E402
    HOME_RANDOM_CLIP_SIGMA,
    HOME_RANDOM_HAND_STD,
    PSI_REAL_INIT_POSE_STD,
    sample_home_offset,
)

ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in
              ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
HAND_JOINTS = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in
               ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
UPPER = ARM_JOINTS + HAND_JOINTS
HANDS_MASK = np.array([n in HAND_JOINTS for n in UPPER])


# ----------------------------------------------------------------------------- pure tests
def test_profile_covers_all_arm_joints():
    assert set(ARM_JOINTS) <= set(PSI_REAL_INIT_POSE_STD)
    assert all(0.005 < s < 0.1 for s in PSI_REAL_INIT_POSE_STD.values()), "sigma should be 'slight' (rad)"


def test_offset_is_bounded_and_hands_stay_open():
    rng = np.random.default_rng(0)
    for _ in range(200):
        off = sample_home_offset(UPPER, rng, scale=1.0, hands_mask=HANDS_MASK)
        assert off.shape == (len(UPPER),)
        assert np.all(off[HANDS_MASK] == 0.0)
        for i, n in enumerate(ARM_JOINTS):
            assert abs(off[i]) <= HOME_RANDOM_CLIP_SIGMA * PSI_REAL_INIT_POSE_STD[n] + 1e-12


def test_offset_statistics_match_profile():
    rng = np.random.default_rng(1)
    offs = np.array([sample_home_offset(UPPER, rng, hands_mask=HANDS_MASK) for _ in range(4000)])
    arm = offs[:, : len(ARM_JOINTS)]
    expected = np.array([PSI_REAL_INIT_POSE_STD[n] for n in ARM_JOINTS])
    # clipping at 2.5 sigma removes ~1.2% of the mass, so the empirical std is ~0.98 sigma
    assert np.allclose(arm.std(0), expected, rtol=0.12)
    assert np.all(np.abs(arm.mean(0)) < 0.2 * expected)


def test_scale_and_hands_flag():
    rng = np.random.default_rng(2)
    small = np.array([sample_home_offset(UPPER, rng, scale=0.25, hands_mask=HANDS_MASK) for _ in range(500)])
    rng = np.random.default_rng(2)
    full = np.array([sample_home_offset(UPPER, rng, scale=1.0, hands_mask=HANDS_MASK) for _ in range(500)])
    assert np.allclose(small, 0.25 * full)                     # same draws, scaled
    off = sample_home_offset(UPPER, np.random.default_rng(3), hands_mask=HANDS_MASK, randomize_hands=True)
    assert np.any(off[HANDS_MASK] != 0.0)
    assert np.all(np.abs(off[HANDS_MASK]) <= HOME_RANDOM_CLIP_SIGMA * HOME_RANDOM_HAND_STD + 1e-12)


def test_seed_is_deterministic_and_unknown_joints_untouched():
    a = sample_home_offset(UPPER + ["waist_yaw_joint"], np.random.default_rng(7), hands_mask=list(HANDS_MASK) + [False])
    b = sample_home_offset(UPPER + ["waist_yaw_joint"], np.random.default_rng(7), hands_mask=list(HANDS_MASK) + [False])
    assert np.array_equal(a, b)
    assert a[-1] == 0.0                                        # waist is not in the profile
    c = sample_home_offset(UPPER, np.random.default_rng(8), hands_mask=HANDS_MASK)
    assert not np.array_equal(a[: len(UPPER)], c)


# ------------------------------------------------------------------------- simulation test
HZ = 50
HOME_DURATION = 2.5          # short ramp keeps the test ~40 s; the real deploy default is 6 s
HOME_MAX_SPEED = 0.5
SCALE = 1.0


def _run_home_randomize_sim(tmp_path, random_arms: bool):
    import gymnasium as gym
    import mujoco
    import simple.envs as _  # noqa: F401
    import decoupled_wbc.control.teleop.streamers.pico_streamer as pico_mod
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent
    from test_soft_reset import FakeOperator

    fake = FakeOperator()
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
        tp = agent._teleop_policy
        tp.home_after_save, tp.home_duration, tp.home_max_joint_speed = True, HOME_DURATION, HOME_MAX_SPEED
        tp.home_randomize, tp.home_random_scale, tp.home_random_hands = True, SCALE, False
        tp.home_random_arms = random_arms                # arm jitter is opt-in (default: height only)
        tp._home_rng = np.random.default_rng(123)

        rm = agent._dwbc_robot_model
        up_idx = rm.get_joint_group_indices("upper_body")
        inv = {v: k for k, v in rm.joint_to_dof_index.items()}
        up_names = [inv[i] for i in up_idx]
        arm_pos = np.array([up_names.index(n) for n in ARM_JOINTS])
        hand_pos = np.array([i for i, n in enumerate(up_names) if n not in ARM_JOINTS and "waist" not in n])
        canonical = np.asarray(rm.default_body_pose, dtype=float)[up_idx]
        sigma = np.array([PSI_REAL_INIT_POSE_STD[n] for n in ARM_JOINTS]) * SCALE

        obs, info = env.reset()
        m, d = robot.mjModel, robot.mjData
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]
        if robot.elastic_band is not None:
            robot.elastic_band.enable = False
        agent._dropping = False
        agent.reset_policy()
        agent._wbc_policy.lower_body_policy.use_policy_action = True

        # state-driven script: three returns to home (activate, save, discard)
        parked_targets, parked_measured, release_first, ramp_steps = [], [], [], []
        parked_heights, parked_pelvis_z = [], []
        tp.home_random_height, tp.home_random_height_range = random_arms, (0.68, 0.74)   # height jitter only in the opt-in run
        t_start = time.monotonic(); tick = 0; phase = "settle"; phase_t0 = 0.0
        wait_state = lambda target, timeout: (lambda t: tp._state == target or (t - phase_t0) > timeout)
        plan = {
            "settle":        (lambda: None, lambda t: robot.stabilized and t > 2.0, "home1"),
            "home1":         (lambda: fake.press("activate"), wait_state("parked", 12.0), "hold1"),
            "hold1":         (lambda: None, lambda t: t - phase_t0 > 1.0, "rel1"),
            "rel1":          (lambda: fake.press("A"), lambda t: t - phase_t0 > 1.5, "home2"),
            "home2":         (lambda: fake.press("A"), wait_state("parked", 12.0), "hold2"),
            "hold2":         (lambda: None, lambda t: t - phase_t0 > 1.0, "rel2"),
            "rel2":          (lambda: fake.press("A"), lambda t: t - phase_t0 > 1.5, "home3"),
            "home3":         (lambda: fake.press("B"), wait_state("parked", 12.0), "hold3"),
            "hold3":         (lambda: None, lambda t: t - phase_t0 > 1.0, None),
        }
        plan[phase][0]()
        prev_state, prev_tgt = None, None
        while phase is not None:
            t = time.monotonic() - t_start
            assert t < 90, "test timed out"
            _, done, nxt = plan[phase]
            if done(t):
                if phase.startswith("hold"):
                    assert tp._state == "parked", f"{phase}: policy never reached 'parked' (state {tp._state})"
                    parked_targets.append(np.asarray(tp._home_to, dtype=float).copy())
                    parked_measured.append(d.qpos[qadr].copy())
                    parked_heights.append(float(tp._home_height) if tp._home_height is not None else 0.74); parked_pelvis_z.append(float(d.qpos[2]))
                phase, phase_t0 = nxt, t
                if phase is None:
                    break
                plan[phase][0]()
            action = agent.get_action(obs, instruction=task.instruction, privileged_info=info)
            obs, _r, _term, _tr, info = env.step(action)
            ta = getattr(agent, "_last_teleop_action", None) or {}
            tgt = np.asarray(ta.get("target_upper_body_pose", np.full(len(up_idx), np.nan)), dtype=float)
            st = tp._state if tp.is_active else "off"
            if prev_state == "parked" and st == "release":
                release_first.append(float(np.abs(tgt - prev_tgt).max()))        # first tracking target vs parked
            if prev_state in ("tracking", "free") and st == "homing" and prev_tgt is not None:
                ramp_steps.append(float(np.abs(tgt - prev_tgt).max()))           # continuity at ramp start
            prev_state, prev_tgt = st, tgt
            tick += 1
            sleep = t_start + (tick / HZ) - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    finally:
        env.close()

    # ------------------------------------------------------------------ assertions
    assert len(parked_targets) == 3, f"expected 3 parked poses, got {len(parked_targets)}"
    P = np.array(parked_targets)
    offs = P[:, arm_pos] - canonical[arm_pos]
    if random_arms:
        # 1. each home pose is a small, bounded perturbation of the canonical pose (arms only)
        assert np.all(np.abs(offs) <= HOME_RANDOM_CLIP_SIGMA * sigma + 1e-9), "offset exceeds 2.5 sigma"
        assert np.all(np.abs(offs).max(axis=1) > 0.005), "a home pose was not randomized"
        assert np.all(np.abs(offs).max(axis=1) < 0.2), "home pose is not 'slightly' different (> 0.2 rad)"
        # 3. consecutive home poses differ from each other
        for i in range(3):
            for j in range(i + 1, 3):
                assert np.abs(P[i] - P[j]).max() > 0.01, f"home poses {i} and {j} are identical"
    else:
        # default: the arms come back to the exact canonical pose every time
        assert np.all(np.abs(offs) < 1e-9), f"arms were jittered without home_random_arms: max {np.abs(offs).max():.4f}"
    # 2. hands stay open at home
    assert np.all(np.abs(P[:, hand_pos]) < 1e-9)
    # 4. the robot actually reached each randomized home pose (PD sag allowed)
    M = np.array(parked_measured)
    assert np.all(np.abs(M - P[:, arm_pos]).max(axis=1) < 0.15), "measured pose far from the randomized home target"
    # 5. release is jump-free: the first tracking target equals the parked (randomized) pose
    assert len(release_first) >= 2 and max(release_first) < 1e-3, f"release jump {release_first}"
    # 6. ramps back home start continuously from the last published target
    assert all(s < 0.02 for s in ramp_steps), f"step at ramp start {ramp_steps}"

    # 7. base height: fixed at 0.74 by default; with the opt-in, in range, different each time, and tracked by the robot
    H = np.array(parked_heights); Z = np.array(parked_pelvis_z)
    if random_arms:
        assert np.all((H >= 0.68 - 1e-9) & (H <= 0.74 + 1e-9)), f"home height out of range {H}"
        assert np.ptp(H) > 0.01, f"home heights not randomized {H}"
        assert np.corrcoef(H, Z)[0, 1] > 0.9 and np.ptp(Z) > 0.5 * np.ptp(H), f"pelvis height does not follow the command: cmd {H} measured {Z}"
    else:
        assert np.all(np.abs(H - 0.74) < 1e-9), f"home height changed without home_random_height: {H}"
        assert np.all(np.abs(Z - 0.74) < 0.03), f"pelvis not at the fixed 0.74 m home height: {Z}"
    (tmp_path / "home_randomize_results.txt").write_text(
        "\n".join(f"home {i}: max|offset| {np.abs(o).max():.3f} rad  height {h:.3f} m" for i, (o, h) in enumerate(zip(offs, H))))
    print("\n".join(f"home {i}: arms max|offset| {np.abs(o).max():.3f} rad, height {h:.3f} m" for i, (o, h) in enumerate(zip(offs, H))))


@pytest.mark.sim
def test_home_randomize_default_is_canonical_in_mujoco(tmp_path):
    """--home-randomize alone: arms exactly canonical, base height fixed at 0.74 m."""
    _run_home_randomize_sim(tmp_path, random_arms=False)


@pytest.mark.sim
def test_home_randomize_with_arms_in_mujoco(tmp_path):
    """--home-randomize --home-random-arms --home-random-height: arm joints and height both jittered."""
    _run_home_randomize_sim(tmp_path, random_arms=True)
