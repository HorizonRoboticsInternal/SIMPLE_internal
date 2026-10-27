# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest


def test_episode_robot_state_restore_preserves_objects_and_time():
    """Restore A/B/A by joint name, rejecting invalid states before mutation."""
    from copy import deepcopy
    from types import SimpleNamespace

    import mujoco

    from simple.envs.sonic_loco_manip import SonicLocoManipEnv
    from simple.robots.g1_sonic import WHOLE_BODY_JOINTS

    names = WHOLE_BODY_JOINTS
    links = "".join(
        f'<body pos="0 0 .01"><joint name="{name}"/>' '<geom size=".01" mass=".01"/>'
        for name in names
    ) + "</body>" * len(names)
    motors = "".join(f'<motor name="{name}" joint="{name}"/>' for name in names)
    # Put an object before the robot to catch accidental qpos[:7] writes.
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="2 0 1"><freejoint name="object"/>'
        '<geom size=".1"/></body><body pos="0 0 1"><freejoint name="root"/>'
        f'<geom size=".1"/>{links}</body></worldbody>'
        f"<actuator>{motors}</actuator></mujoco>"
    )
    data = mujoco.MjData(model)
    data.time = 1.25
    data.qvel[:] = 0.2
    object_qpos = data.joint("object").qpos.copy()
    object_qvel = data.joint("object").qvel.copy()
    env = SimpleNamespace(
        mjModel=model,
        mjData=data,
        task=SimpleNamespace(robot=SimpleNamespace(joint_names=names)),
    )
    state = {
        "version": 1,
        "joint_positions": dict.fromkeys(names, 0.03),
        "base_pose": [0.1, 0.2, 0.8, 1, 0, 0, 0],
        "base_height_command": 0.74,
    }
    restore = SonicLocoManipEnv._restore_initial_robot_state
    restore(env, state)
    other = deepcopy(state)
    other["joint_positions"] = dict.fromkeys(names, -0.1)
    restore(env, other)
    restore(env, state)
    np.testing.assert_array_equal(data.joint("root").qpos, state["base_pose"])
    np.testing.assert_array_equal(data.joint("object").qpos, object_qpos)
    np.testing.assert_array_equal(data.joint("object").qvel, object_qvel)
    for name in names:
        assert data.joint(name).qpos[0] == 0.03
        assert data.joint(name).qvel[0] == 0
    assert np.all(data.joint("root").qvel == 0)
    assert data.time == 1.25
    state["joint_positions"][names[0]] = 0.1
    assert env.initial_robot_state["joint_positions"][names[0]] == 0.03
    for key, value in (
        ("joint_positions", {}),
        ("base_pose", [0, 0, 1, 0, 0, 0, 0]),
        ("base_height_command", float("nan")),
    ):
        invalid = deepcopy(state)
        invalid[key] = value
        before = data.qpos.copy()
        with pytest.raises(ValueError):
            restore(env, invalid)
        np.testing.assert_array_equal(data.qpos, before)


def test_wbc_hand_observation_and_command_preserve_joint_identity():
    """MJCF traversal and WBC actuator order differ for index/middle fingers."""
    from types import SimpleNamespace

    import mujoco
    from decoupled_wbc.control.robot_model.instantiation.g1 import (
        instantiate_g1_robot_model,
    )

    from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

    names = [
        f"{side}_hand_{finger}_{i}_joint"
        for side in ("left", "right")
        for finger, count in (("thumb", 3), ("middle", 2), ("index", 2))
        for i in range(count)
    ]
    bodies = "".join(
        f'<body><joint name="{name}"/><geom size=".01" mass=".01"/></body>'
        for name in names
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    )
    agent = SonicDecoupledWbcAgent.__new__(SonicDecoupledWbcAgent)
    agent.robot = SimpleNamespace(
        mjModel=model,
        left_hand_index=np.arange(7),
        right_hand_index=np.arange(7, 14),
    )
    rm = instantiate_g1_robot_model()
    agent._dwbc_robot_model = rm
    observation = {
        "body_q": np.zeros(29),
        "body_dq": np.zeros(29),
        "floating_base_pose": np.array([0, 0, .74, 1, 0, 0, 0]),
        "floating_base_vel": np.zeros(6),
    }
    for side_index, side in enumerate(("left", "right")):
        for offset, suffix in enumerate(("q", "dq", "ddq", "tau_est")):
            observation[f"{side}_hand_{suffix}"] = (
                np.arange(7, dtype=float) + 1 + 10 * side_index + 100 * offset
            ) / 1000
    result = agent._build_wbc_observation(observation)
    for side_index, side in enumerate(("left", "right")):
        for suffix in ("q", "dq", "ddq", "tau_est"):
            expected = observation[f"{side}_hand_{suffix}"]
            for i, name in enumerate(names[side_index * 7:(side_index + 1) * 7]):
                assert result[suffix][rm.dof_index(name)] == expected[i]
            np.testing.assert_array_equal(agent._hand_command(result[suffix], side), expected)
