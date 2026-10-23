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

"""Versioned Holomotion assets and physical snapshot initialization."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np

ROT0_ROBOT_UID = "g1_rot0_teleop"
ROT0_ROBOT_XML = "robots/g1_rot0_teleop/20261019/g1_rot0_teleop.xml"


def register_rot0_robot() -> str:
    """Register the versioned rot0 asset through SIMPLE's standard data root."""
    from simple.robots.g1_sonic import G1Sonic
    from simple.robots.registry import RobotRegistry
    from simple.utils import resolve_data_path

    xml = Path(resolve_data_path(ROT0_ROBOT_XML)).resolve(strict=True)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nu != 43 or model.nq != 50:
        raise ValueError("Holomotion requires a floating G1 with 43 actuators")
    for camera in ("hbvcam_left_pinhole", "hbvcam_right_pinhole"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"Robot bundle lacks {camera}")

    @RobotRegistry.register(ROT0_ROBOT_UID)
    class G1Rot0Teleop(G1Sonic):
        # Keep the recorded spatial-state key; reuse G1 PD and joint mappings.
        uid = "g1_sonic"
        mjcf_path = str(xml)

    return str(xml)


def export_episode_physics(
    replay_path: str | Path, output_path: str | Path
) -> Path:
    """Extract a single physical initialization from a native exact replay log.

    The output contains no trajectory, PICO events or policy/node history.
    Model sizes detect incompatible scenes; callers must construct the same
    robot and scene ordering, as in native replay, before restoring this state.
    """
    import json

    with np.load(replay_path, allow_pickle=False) as log:
        info = json.loads(str(log["info_json"]))
        payload = {
            "schema_version": np.asarray(1),
            "state0": log["state0"].copy(),
            "band0": log["action_band_state"][0].copy(),
            "render_step0": log["render_step0"].copy(),
            "model_sizes": np.asarray(
                [
                    info["sizes"][key]
                    for key in ("nq", "nv", "na", "nu", "nbody")
                ],
                dtype=np.int64,
            ),
        }
        if len(log["rest_idx"]) and int(log["rest_idx"][0]) == 0:
            payload["rest0"] = log["rest_val"][0].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def restore_episode_physics(env: Any, snapshot_path: str | Path) -> None:
    """Restore replay's physical start after scene reset, before controller reset.

    Uses the same INTEGRATION, REST, band and render-step restoration as
    ``simple.cli.rerecord_holomotion_v14``. No controller object is accepted:
    PICO state, reference queues and policy history start fresh in evaluation.
    This requires the same constructed scene/robot ordering as native replay;
    matching state sizes alone does not prove matching model parameters.
    """
    import mujoco

    from simple.teleop.holomotion_v14 import exact_log as XL

    eng = env.mujoco
    model, data = eng.mjModel, eng.mjData
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        if int(snapshot["schema_version"]) != 1:
            raise ValueError("Unsupported physical initialization schema")
        sizes = np.asarray(
            [
                getattr(model, name)
                for name in ("nq", "nv", "na", "nu", "nbody")
            ]
        )
        if not np.array_equal(snapshot["model_sizes"], sizes):
            raise ValueError(
                "Physical initialization does not match scene/robot model sizes"
            )
        state = np.asarray(snapshot["state0"], dtype=np.float64)
        rest = (
            np.asarray(snapshot["rest0"], dtype=np.float64)
            if "rest0" in snapshot
            else None
        )
        band = np.asarray(snapshot["band0"], dtype=np.float64)
        render_step = int(snapshot["render_step0"])
        for name, value, size in (
            ("state0", state, mujoco.mj_stateSize(model, XL.INTEGRATION)),
            ("band0", band, 5),
        ):
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid physical initialization {name}")
        if rest is not None and (
            rest.shape != (mujoco.mj_stateSize(model, XL.REST),)
            or not np.isfinite(rest).all()
        ):
            raise ValueError("Invalid physical initialization rest0")
        if render_step < 0:
            raise ValueError("Invalid physical initialization render_step0")
    # Validate the complete snapshot before modifying the simulator.
    XL.set_state(model, data, state, XL.INTEGRATION)
    if rest is not None:
        XL.set_state(model, data, rest, XL.REST)
    mujoco.mj_forward(model, data)
    XL.restore_band(env.task.robot, band)
    eng.render_step = render_step
    if hasattr(env.task, "_start_xy"):
        env.task._start_xy = np.asarray(data.qpos[:2], dtype=float).copy()
