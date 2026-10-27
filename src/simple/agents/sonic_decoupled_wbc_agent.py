# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations


import time
from simple.simulation_clock import controller_time, simulation_timed
import numpy as np
from simple.core.action import ActionCmd

from .wbc_agent import WholeBodyControlAgent
from simple.robots.g1_sonic import G1Sonic
from collections import deque
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig

class SonicDecoupledWbcAgent(WholeBodyControlAgent):
    
    @simulation_timed
    def __init__(self, robot: G1Sonic, sonic_config: dict):
        self.robot = robot
        self.sim_dt = sonic_config["SIMULATE_DT"]

        # --- Decoupled WBC pipeline (no teleop/streamer) ---
        self._init_decoupled_policy(sonic_config)
        self._action_queue = deque()

    def _init_decoupled_policy(self, sonic_config: dict):
        

        enable_waist = sonic_config.get("enable_waist", False)
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"

        self._dwbc_robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=sonic_config.get("high_elbow_pose", False),
        )

        dwbc_config = ControlLoopConfig(
            enable_waist=enable_waist,
            high_elbow_pose=sonic_config.get("high_elbow_pose", False),
        )
        wbc_config = dwbc_config.load_wbc_yaml()
        assert wbc_config["SIMULATE_DT"] == self.sim_dt

        initial_joints = sonic_config.get("initial_joint_positions", {})
        if initial_joints:
            self._dwbc_robot_model.set_initial_body_pose(
                np.asarray(list(initial_joints.values())),
                [self._dwbc_robot_model.dof_index(name) for name in initial_joints],
            )

        self._wbc_policy = get_wbc_policy(
            "g1", self._dwbc_robot_model, wbc_config,
            init_time=dwbc_config.upper_body_joint_speed,
        )

        self._default_initial_body_pose = self._dwbc_robot_model.initial_body_pose.copy()
        self._default_upper_policy = self._wbc_policy.upper_body_policy
        self._episode_base_height = None
        self._upper_body_indices = self._dwbc_robot_model.get_joint_group_indices(
            "upper_body"
        )
        self._control_frequency = dwbc_config.control_frequency
        self._control_dt = 4 * self.sim_dt  # 0.02 s (50 Hz main loop)
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._t_start = controller_time(self)

    @simulation_timed
    def initialize_from_episode(self, initial_robot_state: dict):
        """Seed a stationary controller state without stepping physics or inference.

        Call after environment and agent reset. Repeated static observations are an
        explicit history initialization, not a simulated warmup trajectory.
        """
        from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy
        from decoupled_wbc.control.policy.interpolation_policy import InterpolationPolicy
        from pinocchio import rpy
        import torch

        rm = self._dwbc_robot_model
        lower = self._wbc_policy.lower_body_policy
        upper = self._wbc_policy.upper_body_policy
        if not isinstance(lower, G1GearWbcPolicy) or not isinstance(upper, InterpolationPolicy):
            raise TypeError("Episode initialization requires Gear WBC and interpolation policies")
        joints = initial_robot_state["joint_positions"]
        if set(joints) != set(self.robot.joint_names):
            raise ValueError("Episode initialization requires every robot joint exactly once")
        values = np.asarray([joints[name] for name in self.robot.joint_names], dtype=float)
        height = float(initial_robot_state["base_height_command"])
        if not np.isfinite(values).all() or not np.isfinite(height) or height <= 0:
            raise ValueError("Episode joints and positive base height must be finite")
        observation = self._build_wbc_observation(self.robot.prepare_obs())
        indices = [rm.dof_index(name) for name in self.robot.joint_names]
        if not np.allclose(observation["q"][indices], values, atol=1e-6, rtol=0):
            mismatch = {
                name: (float(observation["q"][index]), float(value))
                for name, index, value in zip(self.robot.joint_names, indices, values)
                if abs(observation["q"][index] - value) > 1e-6
            }
            raise ValueError(f"Environment/controller joint ordering mismatch: {mismatch}")
        if not np.allclose(observation["dq"], 0, atol=1e-6):
            raise ValueError("Frozen episode initialization requires zero joint velocity")
        rm.set_initial_body_pose(values, indices)
        now = controller_time(self)
        self._action_queue.clear()
        self._episode_base_height = height
        self._wbc_policy.reset(init_time=now)
        pose = rm.get_initial_upper_body_pose().copy()
        self._wbc_policy.upper_body_policy = InterpolationPolicy(
            init_time=now,
            init_values={
                "target_upper_body_pose": pose,
                "base_height_command": np.array([height]),
                "navigate_cmd": np.zeros(4),
            },
            max_change_rate=upper.max_change_rate,
        )
        lower_indices = rm.get_joint_group_indices("lower_body")
        lower_q = observation["q"][lower_indices].copy()
        scale = np.asarray(lower.config["action_scale"])
        if lower_q.shape != lower.action.shape or np.any(scale == 0):
            raise ValueError("Unsupported lower-body action schema")
        lower.action = ((lower_q - lower.config["default_angles"]) / scale).astype(np.float32)
        lower.target_dof_pos = lower_q.copy()
        lower.height_cmd = height
        lower.cmd[:] = 0
        lower.target_yaw_cmd = 0.0
        # Match the torso command calculation in G1DecoupledWholeBodyPolicy.
        command_q = np.zeros(rm.num_dofs)
        command_q[self._upper_body_indices] = pose
        rm.cache_forward_kinematics(command_q, auto_clip=False)
        torso = rm.frame_placement("torso_link").rotation.copy()
        pelvis = rm.frame_placement("pelvis").rotation
        yaw = np.arctan2(pelvis[1, 0], pelvis[0, 0])
        torso_rpy = rpy.matrixToRpy(rpy.rpyToMatrix(0, 0, yaw).T @ torso)
        lower.roll_cmd, lower.pitch_cmd, lower.yaw_cmd = torso_rpy
        lower.use_policy_action = True
        lower.set_use_teleop_policy_cmd(True)
        self._wbc_policy.is_in_teleop_mode = True
        self._wbc_policy.last_goal_time = now
        # compute_observation updates gait phase; use zero frequency while seeding
        # so no synthetic control time passes.
        frequency = lower.freq_cmd
        lower.freq_cmd = 0.0
        try:
            single_obs, dim = lower.compute_observation(observation)
        finally:
            lower.freq_cmd = frequency
        count = lower.config["obs_history_len"]
        if dim * count != lower.config["num_obs"]:
            raise ValueError("Unsupported lower-body observation history schema")
        lower.observation = observation
        lower.obs_history = deque((single_obs.copy() for _ in range(count)), maxlen=count)
        lower.obs_buffer[:] = np.concatenate(list(lower.obs_history))
        lower.obs_tensor = torch.from_numpy(lower.obs_buffer).unsqueeze(0)
        self._cached_target_q = rm.get_body_actuated_joints(observation["q"]).copy()
        self._cached_left_hand_q = self._hand_command(observation["q"], "left")
        self._cached_right_hand_q = self._hand_command(observation["q"], "right")

    # ------------------------------------------------------------------
    # Observation / goal building
    # ------------------------------------------------------------------

    def _hand_order(self, side):
        """Indices mapping MuJoCo traversal order to WBC actuated-hand order."""
        mj_indices = getattr(self.robot, f"{side}_hand_index")
        mj_names = [self.robot.mjModel.joint(int(i)).name for i in mj_indices]
        dof_names = {index: name for name, index in self._dwbc_robot_model.joint_to_dof_index.items()}
        wbc_names = [dof_names[i] for i in self._dwbc_robot_model.get_hand_actuated_joint_indices(side)]
        return np.asarray([mj_names.index(name) for name in wbc_names])

    def _hand_command(self, q, side):
        values = self._dwbc_robot_model.get_hand_actuated_joints(q, side=side)
        return values[np.argsort(self._hand_order(side))].copy()

    def _build_wbc_observation(self, sim_obs: dict) -> dict:
        """Convert SIMPLE observation dict to decoupled_wbc observation format.

        Copied from PicoDecoupledAgent._build_wbc_observation.
        """
        rm = self._dwbc_robot_model
        # prepare_obs follows MJCF traversal (thumb/middle/index), while WBC
        # actuated hands are thumb/index/middle. Map by name in both directions.
        sim_obs = dict(sim_obs)
        for side in ("left", "right"):
            order = self._hand_order(side)
            for suffix in ("q", "dq", "ddq", "tau_est"):
                key = f"{side}_hand_{suffix}"
                if key in sim_obs:
                    sim_obs[key] = np.asarray(sim_obs[key])[order]
        obs = {}

        left_hand_q = sim_obs.get("left_hand_q", np.zeros(7))
        right_hand_q = sim_obs.get("right_hand_q", np.zeros(7))
        left_hand_dq = sim_obs.get("left_hand_dq", np.zeros(7))
        right_hand_dq = sim_obs.get("right_hand_dq", np.zeros(7))

        obs["q"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_q"],
            left_hand_actuated_joint_values=left_hand_q,
            right_hand_actuated_joint_values=right_hand_q,
        )
        obs["dq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_dq"],
            left_hand_actuated_joint_values=left_hand_dq,
            right_hand_actuated_joint_values=right_hand_dq,
        )
        obs["ddq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_ddq", np.zeros(29)),
            left_hand_actuated_joint_values=sim_obs.get("left_hand_ddq", np.zeros(7)),
            right_hand_actuated_joint_values=sim_obs.get("right_hand_ddq", np.zeros(7)),
        )
        obs["tau_est"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_tau_est", np.zeros(29)),
            left_hand_actuated_joint_values=sim_obs.get("left_hand_tau_est", np.zeros(7)),
            right_hand_actuated_joint_values=sim_obs.get("right_hand_tau_est", np.zeros(7)),
        )

        obs["floating_base_pose"] = sim_obs["floating_base_pose"]
        obs["floating_base_vel"] = sim_obs["floating_base_vel"]
        obs["floating_base_acc"] = sim_obs.get("floating_base_acc", np.zeros(6))

        obs["torso_quat"] = sim_obs.get("secondary_imu_quat", np.array([1, 0, 0, 0]))
        obs["torso_ang_vel"] = (
            sim_obs.get("secondary_imu_vel", np.zeros(6))[3:6]
            if "secondary_imu_vel" in sim_obs
            else np.zeros(3)
        )

        obs["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))
        return obs
        
    @simulation_timed
    def get_stabilize_action(self, observation) -> ActionCmd:
        """Run the WBC pipeline ramping to the WBC default pose — used during stabilization.

        Drives the upper body toward the WBC default configuration (forearms pointing
        forward), matching what teleop does before the operator activates the teleop policy.
        Does NOT read from episode data or advance _data_row_index.
        """
        # Always run the full WBC pipeline (main loop now at 50Hz)
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )

        t_now = controller_time(self)
        control_freq = self._control_frequency
        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        # Use the WBC's own default upper body pose (forearms forward), not the
        # MJCF keyframe defaults (arms down).  On the first step give a 2s ramp
        # window — matching teleop's initial `target_time = t_now + 2.0` — so the
        # arms move smoothly to the default rather than snapping.
        is_first_step = self._cached_target_q is None
        default_upper_body = self._dwbc_robot_model.get_initial_upper_body_pose()
        goal = {
            "target_upper_body_pose": default_upper_body,
            "navigate_cmd": np.asarray(DEFAULT_NAV_CMD),
            "base_height_command": np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT if self._episode_base_height is None else self._episode_base_height)),
            "target_time": t_now + (2.0 if is_first_step else 1 / control_freq),
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)

        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._hand_command(wbc_action["q"], "left")
        self._cached_right_hand_q = self._hand_command(wbc_action["q"], "right")

        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )
    
    def queue_action(self, action_cmd: ActionCmd):
        self._action_queue.append(action_cmd)

    def get_action(self, observation, instruction=None, **kwargs):
        try:
            if observation is not None and "agent" in observation:
                self._last_qpos = dict(zip(self.robot.joint_names, observation["agent"]))
            action = self._action_queue.popleft()
            self._last_pred_action = action
            return action
        except IndexError:
            raise StopIteration("No more queued actions in PrimitiveAgent.")
        
    def __len__(self):
        return len(self._action_queue)
    
    @simulation_timed
    def reset(self, **kwargs):
        # Clearing the cached joint targets re-arms get_stabilize_action's
        # 2-second ramp (it keys `is_first_step` off `_cached_target_q is None`),
        # and resetting the WBC policy clears its internal interpolation
        # setpoint.
        self._action_queue = deque()
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._episode_base_height = None
        self._dwbc_robot_model.set_initial_body_pose(self._default_initial_body_pose.copy())
        self._wbc_policy.upper_body_policy = self._default_upper_policy
        self._wbc_policy.reset(init_time=controller_time(self))
        return super().reset(**kwargs)
