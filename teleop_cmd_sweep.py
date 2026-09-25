#!/usr/bin/env python
"""Command-envelope sweep for the sim teleop controller (decoupled WBC).

Drives the EXACT teleop stack (PicoDecoupledAgent -> InterpolationPolicy +
G1GearWbcPolicy Balance/Walk ONNX -> PD position) but replaces the PICO
streamer with scripted commands, so the locomotion command space
    navigate_cmd = [vx, vy, vyaw_flag, target_yaw],  base_height_command
can be swept programmatically. Each trial: stabilize -> elastic-band drop ->
enable lower-body RL (toggle_policy_action) -> hold the test command 5 s ->
stop 2 s. Records head+front camera video and measures achieved base motion.

Outputs: data/cmd_sweep/<trial>.mp4 + data/cmd_sweep/results.json
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import gymnasium as gym
import simple.envs as _  # noqa: F401  register envs

OUT = Path("data/cmd_sweep")
OUT.mkdir(parents=True, exist_ok=True)

CTRL_HZ = 50
DEFAULT_H = 0.74

# name, nav after landing [vx, vy, vyaw_flag, dyaw(rel target)], height, note
TRIALS = [
    ("baseline_stand",  [0.0, 0.0, 0.0, 0.0], DEFAULT_H, "no command - Balance policy hold"),
    ("vx_0.3",          [0.3, 0.0, 0.0, 0.0], DEFAULT_H, "forward, below teleop cap"),
    ("vx_0.5",          [0.5, 0.0, 0.0, 0.0], DEFAULT_H, "forward, = teleop MAX_LINEAR_VEL"),
    ("vx_0.8",          [0.8, 0.0, 0.0, 0.0], DEFAULT_H, "forward, beyond teleop cap"),
    ("vx_1.2",          [1.2, 0.0, 0.0, 0.0], DEFAULT_H, "forward, far beyond cap"),
    ("vx_1.8",          [1.8, 0.0, 0.0, 0.0], DEFAULT_H, "forward, extreme"),
    ("vx_-0.4",         [-0.4, 0.0, 0.0, 0.0], DEFAULT_H, "backward walk"),
    ("vy_0.4",          [0.0, 0.4, 0.0, 0.0], DEFAULT_H, "strafe left"),
    ("vy_0.8",          [0.0, 0.8, 0.0, 0.0], DEFAULT_H, "strafe, beyond cap"),
    ("turn_+90deg",     [0.0, 0.0, 0.5, 1.57], DEFAULT_H, "turn in place, +90 deg goal"),
    ("turn_fast_180",   [0.0, 0.0, 1.0, 3.10], DEFAULT_H, "fast 180 turn"),
    ("height_0.50",     [0.0, 0.0, 0.0, 0.0], 0.50, "crouch to 0.50 m"),
    ("height_0.30",     [0.0, 0.0, 0.0, 0.0], 0.30, "deep crouch 0.30 m (teleop clamp is 0.2)"),
    ("vx0.5_h0.50",     [0.5, 0.0, 0.0, 0.0], 0.50, "walk while crouched"),
]

CMD_SECS = 5.0
STOP_SECS = 2.0
SETTLE_SECS = 2.0


class ScriptedTeleop:
    """Stands in for TeleopPolicy: same get_action() dict, commands set by the harness."""

    def __init__(self):
        self.nav = [0.0, 0.0, 0.0, 0.0]
        self.height = DEFAULT_H
        self.toggle_once = False
        self.is_active = False  # arms stay at WBC default pose throughout

    def get_action(self):
        toggle = self.toggle_once
        self.toggle_once = False
        return {
            "wrist_pose": np.zeros(14),
            "navigate_cmd": [float(v) for v in self.nav],
            "base_height_command": float(self.height),
            "toggle_policy_action": toggle,
        }

    def reset(self):
        pass

    def close(self):
        return True


def yaw_of(robot):
    w, x, y, z = robot.mjData.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    sonic_config = SimLoopConfig().load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"

    env = gym.make(
        "simple/G1WholebodyXMovePickTeleop-v0",
        sim_mode="mujoco",
        render_hz=CTRL_HZ,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=True,
        max_episode_steps=1_000_000,
        sonic_config=sonic_config,
        target="graspnet1b:0",
        dr_level=0,
        success_criteria=1e9,  # never terminate on task success
    )
    task = env.unwrapped.task
    robot = task.robot

    agent = PicoDecoupledAgent(robot)
    stub = ScriptedTeleop()
    agent._teleop_policy = stub                 # replace the PICO source
    agent._poll_pico_buttons = lambda: None     # no headset buttons
    # The sim CLIs enable the lower-body RL policy at startup (the robot hangs,
    # drops, and lands under active balance). Toggling it on only after landing
    # does NOT work: Balance cannot recover from a statically-held stance.
    agent._wbc_policy.lower_body_policy.use_policy_action = True

    import av

    results = []
    legs_on = False

    def step(obs, info):
        action = agent.get_action(obs, instruction=task.instruction, privileged_info=info)
        obs, _r, _t, _tr, info = env.step(action)
        return obs, info

    _keys_logged = [False]

    def grab(obs, frames):
        l, f = obs.get("head_stereo_left"), obs.get("head_stereo_right")
        if l is None or f is None:
            try:
                fr = env.unwrapped.render()
                l = l if l is not None else fr.get("head_stereo_left")
                f = f if f is not None else fr.get("head_stereo_right")
                if not _keys_logged[0]:
                    print(f"[sweep] obs keys: {sorted(obs.keys())} | render keys: {sorted(fr.keys())}", flush=True)
                    _keys_logged[0] = True
            except Exception as e:
                if not _keys_logged[0]:
                    print(f"[sweep] render fallback failed: {e}", flush=True)
                    _keys_logged[0] = True
        if l is not None and f is not None:
            frames.append(np.concatenate([np.asarray(l), np.asarray(f)], axis=1))

    for name, nav, height, note in TRIALS:
        rec = {"trial": name, "nav_cmd": nav, "height_cmd": height, "note": note}
        frames = []
        try:
            obs, info = env.reset()
            # Mirror the CLI's recording-mode _on_episode_reset: NO elastic-band
            # drop. Spawn standing, reset the whole WBC pipeline to the default
            # pose, then engage the lower-body RL from that clean state.
            if robot.elastic_band is not None:
                robot.elastic_band.enable = False
            agent._dropping = False
            agent.reset_policy()
            agent._wbc_policy.lower_body_policy.use_policy_action = True
            stub.nav = [0.0, 0.0, 0.0, yaw_of(robot)]
            stub.height = DEFAULT_H

            # settle under active balance
            for _ in range(int(SETTLE_SECS * CTRL_HZ)):
                obs, info = step(obs, info)
                grab(obs, frames)

            # 4. command window
            yaw0 = yaw_of(robot)
            stub.nav = [nav[0], nav[1], nav[2], yaw0 + nav[3]]
            stub.height = height
            p_start = robot.mjData.qpos[:2].copy()
            z_min = robot.pelvis_z
            fell = False
            meas_p0 = None
            n_cmd = int(CMD_SECS * CTRL_HZ)
            for i in range(n_cmd):
                obs, info = step(obs, info)
                grab(obs, frames)
                z_min = min(z_min, robot.pelvis_z)
                if i == CTRL_HZ:  # skip 1 s ramp before measuring
                    meas_p0 = robot.mjData.qpos[:2].copy()
                    meas_yaw0 = yaw_of(robot)
                if robot.pelvis_z < 0.30 and height >= 0.45:
                    fell = True
                    break
            p_end = robot.mjData.qpos[:2].copy()
            yaw_end = yaw_of(robot)
            meas_T = (i + 1 - CTRL_HZ) / CTRL_HZ if meas_p0 is not None else None

            # 5. stop window
            stub.nav = [0.0, 0.0, 0.0, yaw_of(robot)]
            stub.height = DEFAULT_H
            if not fell:
                for _ in range(int(STOP_SECS * CTRL_HZ)):
                    obs, info = step(obs, info)
                    grab(obs, frames)
                    if robot.pelvis_z < 0.30:
                        fell = True
                        break

            disp = p_end - (meas_p0 if meas_p0 is not None else p_start)
            rec.update(
                fell=bool(fell),
                pelvis_z_min=round(float(z_min), 3),
                pelvis_z_end=round(float(robot.pelvis_z), 3),
                achieved_speed=(round(float(np.linalg.norm(disp) / meas_T), 3)
                                if meas_T and meas_T > 0.5 else None),
                achieved_yaw_rate=(round(float((yaw_end - meas_yaw0) / meas_T), 3)
                                   if meas_T and meas_T > 0.5 else None),
                frames=len(frames),
            )
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            stub.nav = [0.0, 0.0, 0.0, yaw_of(robot)]
            stub.height = DEFAULT_H

        if frames:
            path = OUT / f"{name}.mp4"
            container = av.open(str(path), "w")
            stream = container.add_stream("libx264", rate=CTRL_HZ)
            stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "24"}
            for fr in frames:
                for pkt in stream.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            rec["video"] = str(path)

        results.append(rec)
        print(f"[sweep] {name}: {json.dumps({k: v for k, v in rec.items() if k != 'nav_cmd'})}",
              flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

    env.close()
    print("[sweep] DONE")


if __name__ == "__main__":
    main()
