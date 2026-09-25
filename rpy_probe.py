#!/usr/bin/env python
"""Empirical test: does enable_waist decide whether torso_rpy_cmd is non-zero?

Builds the real teleop stack (PicoDecoupledAgent -> G1DecoupledWholeBodyPolicy ->
G1GearWbcPolicy) in MuJoCo with the PICO streamer stubbed out, steps it, and
prints wbc_action["torso_rpy_cmd"] -- the exact value the exporter records as
observation.torso_rpy_command -- for enable_waist False and True.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
import simple.envs as _  # noqa: F401


class Stub:
    """Stands in for TeleopPolicy: navigate_cmd present => teleop mode active."""

    is_active = False

    def get_action(self):
        return {
            "wrist_pose": np.zeros(14),
            "navigate_cmd": [0.0, 0.0, 0.0, 0.0],
            "base_height_command": 0.74,
        }

    def reset(self):
        pass

    def close(self):
        return True


def run(enable_waist: bool, steps: int = 120):
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    cfg = SimLoopConfig().load_wbc_yaml()
    cfg["ENV_NAME"] = "simple"
    cfg["enable_waist"] = enable_waist

    env = gym.make(
        "simple/G1WholebodyXMovePickTeleop-v0",
        sim_mode="mujoco", render_hz=50, physics_dt=cfg["SIMULATE_DT"],
        headless=True, max_episode_steps=10**6, sonic_config=cfg,
        target="graspnet1b:0", dr_level=0, success_criteria=1e9,
    )
    task = env.unwrapped.task
    robot = task.robot
    agent = PicoDecoupledAgent(robot)
    agent._teleop_policy = Stub()
    agent._poll_pico_buttons = lambda: None
    agent._wbc_policy.lower_body_policy.use_policy_action = True

    obs, info = env.reset()
    if robot.elastic_band is not None:
        robot.elastic_band.enable = False
    agent._dropping = False
    agent.reset_policy()
    agent._wbc_policy.lower_body_policy.use_policy_action = True

    seen = []
    for _ in range(steps):
        action = agent.get_action(obs, instruction=task.instruction, privileged_info=info)
        obs, _r, _t, _tr, info = env.step(action)
        wa = getattr(agent, "_last_wbc_action", None)
        if wa is None:                      # read it straight off the policy
            wa = agent._wbc_policy.lower_body_policy
            seen.append([wa.roll_cmd, wa.pitch_cmd, wa.yaw_cmd])
    a = np.asarray(seen, dtype=float)
    env.close()
    return a


if __name__ == "__main__":
    for ew in (False, True):
        a = run(ew)
        print(f"enable_waist={ew!s:5s}  n={len(a)}  "
              f"max|rpy|={np.abs(a).max():.6f}  std={np.round(a.std(0), 6)}", flush=True)
