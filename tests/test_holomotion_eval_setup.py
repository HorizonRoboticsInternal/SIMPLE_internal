# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Physical initialization restores simulator state without a controller."""

import numpy as np


def test_physical_snapshot_restores_scene_without_controller(tmp_path):
    import json
    from types import SimpleNamespace

    import mujoco
    import pytest

    from simple.teleop.holomotion_v14 import exact_log as XL
    from simple.teleop.holomotion_v14.eval_setup import (
        export_episode_physics,
        restore_episode_physics,
    )

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qpos[:3] = [0.1, 0.2, 0.8]
    data.qvel[:] = np.arange(model.nv) * 0.01
    data.qfrc_applied[:] = 0.3
    band = SimpleNamespace(
        length=0.4, enable=False, point=np.array([0, 0, 1.0])
    )
    env = SimpleNamespace(
        mujoco=SimpleNamespace(mjModel=model, mjData=data, render_step=9),
        task=SimpleNamespace(
            robot=SimpleNamespace(elastic_band=band), _start_xy=None
        ),
    )
    state = XL.get_state(model, data, XL.INTEGRATION)
    replay = tmp_path / "replay.npz"
    np.savez(
        replay,
        state0=state,
        info_json=json.dumps(
            {
                "sizes": {
                    key: getattr(model, key)
                    for key in ("nq", "nv", "na", "nu", "nbody")
                }
            }
        ),
        action_band_state=np.array([[0.4, 0, 0, 0, 1.0]]),
        render_step0=9,
        rest_idx=np.array([0]),
        rest_val=np.array([XL.get_state(model, data, XL.REST)]),
    )
    snapshot = export_episode_physics(replay, tmp_path / "initial.npz")
    mujoco.mj_resetData(model, data)
    restore_episode_physics(env, snapshot)
    np.testing.assert_array_equal(
        XL.get_state(model, data, XL.INTEGRATION), state
    )
    np.testing.assert_array_equal(env.task._start_xy, [0.1, 0.2])
    assert env.mujoco.render_step == 9
    assert band.length == 0.4 and not band.enable

    # Reject a different scene before changing any physics state.
    with np.load(snapshot) as values:
        invalid = dict(values)
    invalid["model_sizes"] = invalid["model_sizes"] + 1
    np.savez(tmp_path / "invalid.npz", **invalid)
    with pytest.raises(ValueError, match="model sizes"):
        restore_episode_physics(env, tmp_path / "invalid.npz")
    np.testing.assert_array_equal(
        XL.get_state(model, data, XL.INTEGRATION), state
    )
