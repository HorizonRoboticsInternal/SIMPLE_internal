# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
from simple.robots.g1_sonic import G1Sonic
from simple.robots.registry import RobotRegistry
from simple.robots.g1comp_assets import (
    RUNTIME_VERSION,
    EGO_CAM_QUAT,
    build_mjcf,
    camera_position,
    add_isaac_head,
)
from simple.utils import resolve_data_path


@RobotRegistry.register("g1comp_sonic")
class G1CompSonic(G1Sonic):
    # Keep the spatial-state key used by existing eval datasets and controllers.
    uid = "g1_sonic"
    runtime_version = RUNTIME_VERSION
    isaac_head_cam_link = "torso_link/head_servo_link/xl330_link/d455_link"
    head_camera_mujoco_quaternion = EGO_CAM_QUAT

    def __init__(self, *args, **kwargs):
        self.mjcf_path = str(build_mjcf(resolve_data_path(G1Sonic.mjcf_path)))
        self.head_camera_mujoco_position = camera_position(self.mjcf_path).tolist()
        super().__init__(*args, **kwargs)

    @property
    def head_cam_link(self):
        return "d455_link"

    def add_isaac_visuals(self, stage, robot_path):
        add_isaac_head(stage, robot_path)
