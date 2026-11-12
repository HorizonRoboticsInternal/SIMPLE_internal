"""Gravity feed-forward for the real G1 arms (used by G1Env when --enable-gravity-compensation)."""
import numpy as np
import pytest

from decoupled_wbc.control.envs.g1.utils.gravity_ff import gravity_feedforward_actuated
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

ARM_MOTORS = list(range(15, 29))      # body actuator order: 0-11 legs, 12-14 waist, 15-21 L arm, 22-28 R arm


@pytest.fixture(scope="module")
def rm():
    return instantiate_g1_robot_model()


def test_only_arm_motors_get_torque(rm):
    q = np.asarray(rm.default_body_pose, dtype=float)
    tau = gravity_feedforward_actuated(rm, q, ["arms"])
    assert tau.shape == (29,)
    assert np.all(tau[:15] == 0.0), "legs / waist must not receive gravity feed-forward"
    assert np.abs(tau[ARM_MOTORS]).max() > 1.0, "arms should need a few Nm even in the canonical pose"


def test_magnitudes_and_symmetry(rm):
    q = np.asarray(rm.default_body_pose, dtype=float)
    for n in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        q[rm.joint_to_dof_index[n]] = -1.4                       # arms raised forward, ~horizontal
    tau = gravity_feedforward_actuated(rm, q, ["arms"], max_torque=50.0)
    body = list(rm.supplemental_info.body_actuated_joints)
    lsp, rsp = body.index("left_shoulder_pitch_joint"), body.index("right_shoulder_pitch_joint")
    assert 3.0 < abs(tau[lsp]) < 12.0 and 3.0 < abs(tau[rsp]) < 12.0   # a ~2 kg arm+hand at ~0.3 m
    assert np.isclose(tau[lsp], tau[rsp], atol=0.05)                    # left/right symmetric
    assert np.abs(tau).max() < 12.0


def test_clip_and_default_groups(rm):
    q = np.asarray(rm.default_body_pose, dtype=float)
    tau = gravity_feedforward_actuated(rm, q, None, max_torque=1.0)   # None -> arms
    assert np.all(tau[:15] == 0.0) and np.abs(tau).max() <= 1.0 + 1e-12
    full = gravity_feedforward_actuated(rm, q, ["arms"], max_torque=50.0)
    assert np.any(np.abs(full[ARM_MOTORS]) > 1.0)                       # the clip above really clipped


def test_g1env_queue_action_sends_gravity_tau(rm, monkeypatch):
    """Drive G1Env.queue_action with a fake body: with the flag the motor tau is the gravity feed-forward, without it 0.
    G1Env imports ROS 2; outside the deploy container this test is skipped (it is run inside the image instead)."""
    pytest.importorskip("rclpy")
    import decoupled_wbc.control.envs.g1.g1_env as g1_env_mod
    sent = {}

    class FakeBody:
        def queue_action(self, a): sent.update(a)

    env = g1_env_mod.G1Env.__new__(g1_env_mod.G1Env)          # no SDK / channel init
    env.robot_model = rm; env.last_obs = None; env.with_hands = False; env._gravity_ff_warned = False
    env.body = lambda: FakeBody()
    q = np.asarray(rm.default_body_pose, dtype=float)
    env.enable_gravity_compensation = False
    env.gravity_compensation_joints = ["arms"]; env.gravity_compensation_max_torque = 10.0
    g1_env_mod.G1Env.queue_action(env, {"q": q})
    assert np.all(sent["body_tau"] == 0.0)
    env.enable_gravity_compensation = True
    g1_env_mod.G1Env.queue_action(env, {"q": q})
    expect = gravity_feedforward_actuated(rm, q, ["arms"], 10.0)
    assert np.allclose(sent["body_tau"], expect) and np.abs(sent["body_tau"][ARM_MOTORS]).max() > 1.0
    assert np.allclose(sent["body_q"], rm.get_body_actuated_joints(q))
