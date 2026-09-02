#!/usr/bin/env python
"""Motion-onset threshold sweep for the sim teleop controller.

teleop_cmd_sweep.py sampled the *upper* command envelope (0.3 .. 1.8) and so
cannot say where motion begins. This harness sweeps the low end finely to find
the threshold: the command magnitude below which the robot does not move and
above which it does.

Two candidate mechanisms, both measured here:
  1. Policy switch  -- g1_gear_wbc_policy.get_action() selects the Balance ONNX
     when ||[vx, vy, vyaw_flag]|| < 0.1, else the Walk ONNX. Predicts a hard
     step exactly at 0.1.
  2. Walk-policy floor -- even on the Walk policy a small vx may produce no net
     displacement, putting the true onset above 0.1.
Which one dominates is what the grid resolves. Every trial also counts Balance
vs Walk ONNX invocations directly, so the switch is observed, not inferred.

Turning is separate: vyaw is derived from the yaw error, and forced to 0 when
|vyaw_flag| < 0.1, so the flag is predicted to be effectively binary -- its
magnitude above 0.1 should not change the achieved turn rate.

Outputs data/cmd_threshold/results.json (+ optional videos).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import math
from pathlib import Path

import numpy as np

import gymnasium as gym
import simple.envs as _  # noqa: F401  register envs

OUT = Path("data/cmd_threshold")
OUT.mkdir(parents=True, exist_ok=True)

CTRL_HZ = 50
DEFAULT_H = 0.74

SETTLE_SECS = 2.0
CMD_SECS = 8.0      # long hold: 0.02 m/s still integrates to ~13 cm
RAMP_SECS = 1.5     # discarded before measuring
STOP_SECS = 1.0

# Fine grid bracketing the predicted 0.1 switch, plus repeats at 0 for the
# drift floor. Turning holds a large yaw goal so the error never runs out.
GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.09, 0.095, 0.10, 0.105, 0.11,
        0.12, 0.15, 0.20, 0.25, 0.30]


def build_trials(axes):
    trials = []
    for r in range(3):
        trials.append((f"zero_rep{r}", "zero", 0.0, [0.0, 0.0, 0.0, 0.0]))
    if "vx" in axes:
        for v in GRID:
            trials.append((f"vx_{v:g}", "vx", v, [v, 0.0, 0.0, 0.0]))
    if "vy" in axes:
        for v in GRID:
            trials.append((f"vy_{v:g}", "vy", v, [0.0, v, 0.0, 0.0]))
    if "vyaw" in axes:
        for v in GRID:
            # large relative yaw goal (+150 deg) so yaw_error never hits the
            # 0.01 rad dead zone during the window
            trials.append((f"vyaw_{v:g}", "vyaw", v, [0.0, 0.0, v, 2.62]))
    return trials


class ScriptedTeleop:
    """Stands in for TeleopPolicy: same get_action() dict, commands set by the harness."""

    def __init__(self):
        self.nav = [0.0, 0.0, 0.0, 0.0]
        self.height = DEFAULT_H
        self.toggle_once = False
        self.is_active = False

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


def _rle(seq):
    """Run-length encode the Balance/Walk trace: 'b170 W230 b50'."""
    out = []
    for ch in seq:
        if out and out[-1][0] == ch:
            out[-1][1] += 1
        else:
            out.append([ch, 1])
    return " ".join(f"{c}{n}" for c, n in out)


def yaw_of(robot):
    w, x, y, z = robot.mjData.qpos[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def foot_z(robot):
    """Max foot height above its own resting level -- detects stepping in place."""
    try:
        return max(
            float(robot.mjData.body("left_ankle_roll_link").xpos[2]),
            float(robot.mjData.body("right_ankle_roll_link").xpos[2]),
        )
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", default="vx,vy,vyaw",
                    help="comma list of vx,vy,vyaw")
    ap.add_argument("--video", default="",
                    help="comma list of trial names to record video for")
    ap.add_argument("--tag", default="", help="suffix for the results file")
    ap.add_argument("--grid", default="", help="override the sweep grid (comma list)")
    args = ap.parse_args()

    if args.grid:
        GRID[:] = [float(g) for g in args.grid.split(",")]

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    want_video = {v.strip() for v in args.video.split(",") if v.strip()}

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
        success_criteria=1e9,
    )
    task = env.unwrapped.task
    robot = task.robot

    agent = PicoDecoupledAgent(robot)
    stub = ScriptedTeleop()
    agent._teleop_policy = stub
    agent._poll_pico_buttons = lambda: None
    agent._wbc_policy.lower_body_policy.use_policy_action = True

    # --- instrument the ONNX switch: count Balance vs Walk invocations ---
    lb = agent._wbc_policy.lower_body_policy
    counts = {"balance": 0, "walk": 0}
    switch_trace = []

    def wrap(fn, key):
        def inner(x):
            counts[key] += 1
            switch_trace.append("W" if key == "walk" else "b")
            return fn(x)
        return inner

    lb.policy_1 = wrap(lb.policy_1, "balance")
    lb.policy_2 = wrap(lb.policy_2, "walk")

    trials = build_trials(axes)
    results = []
    import av

    def step(obs, info):
        action = agent.get_action(obs, instruction=task.instruction, privileged_info=info)
        obs, _r, _t, _tr, info = env.step(action)
        return obs, info

    for name, axis, value, nav in trials:
        rec = {"trial": name, "axis": axis, "value": value, "nav_cmd": nav}
        frames = []
        record = name in want_video
        try:
            obs, info = env.reset()
            if robot.elastic_band is not None:
                robot.elastic_band.enable = False
            agent._dropping = False
            agent.reset_policy()
            agent._wbc_policy.lower_body_policy.use_policy_action = True
            stub.nav = [0.0, 0.0, 0.0, yaw_of(robot)]
            stub.height = DEFAULT_H

            for _ in range(int(SETTLE_SECS * CTRL_HZ)):
                obs, info = step(obs, info)

            # command window
            counts["balance"] = counts["walk"] = 0
            switch_trace.clear()
            yaw0 = yaw_of(robot)
            stub.nav = [nav[0], nav[1], nav[2], yaw0 + nav[3]]
            p_ramp = None
            path_len = 0.0
            prev_p = robot.mjData.qpos[:2].copy()
            fz_max = -1e9
            fz_min = 1e9
            n_cmd = int(CMD_SECS * CTRL_HZ)
            n_ramp = int(RAMP_SECS * CTRL_HZ)
            fell = False
            for i in range(n_cmd):
                obs, info = step(obs, info)
                if record:
                    fr = env.unwrapped.render()
                    im = fr.get("head_stereo_left")
                    if im is not None:
                        frames.append(np.asarray(im))
                p = robot.mjData.qpos[:2].copy()
                if i == n_ramp:
                    p_ramp = p.copy()
                    yaw_ramp = yaw_of(robot)
                if i >= n_ramp:
                    path_len += float(np.linalg.norm(p - prev_p))
                    z = foot_z(robot)
                    fz_max = max(fz_max, z)
                    fz_min = min(fz_min, z)
                prev_p = p
                if robot.pelvis_z < 0.30:
                    fell = True
                    break

            p_end = robot.mjData.qpos[:2].copy()
            yaw_end = yaw_of(robot)
            T = (i + 1 - n_ramp) / CTRL_HZ

            stub.nav = [0.0, 0.0, 0.0, yaw_of(robot)]
            if not fell:
                for _ in range(int(STOP_SECS * CTRL_HZ)):
                    obs, info = step(obs, info)

            disp = p_end - (p_ramp if p_ramp is not None else p_end)
            dyaw = float(np.arctan2(np.sin(yaw_end - yaw_ramp), np.cos(yaw_end - yaw_ramp)))
            tot = counts["balance"] + counts["walk"]
            rec.update(
                fell=bool(fell),
                meas_secs=round(T, 2),
                net_disp_m=round(float(np.linalg.norm(disp)), 4),
                disp_xy=[round(float(disp[0]), 4), round(float(disp[1]), 4)],
                speed_mps=round(float(np.linalg.norm(disp) / T), 4) if T > 0.5 else None,
                path_len_m=round(path_len, 4),
                yaw_rate_rps=round(dyaw / T, 4) if T > 0.5 else None,
                foot_z_range=round(float(fz_max - fz_min), 4),
                walk_frac=round(counts["walk"] / tot, 3) if tot else None,
                cycles=tot,
                switch_trace=_rle(switch_trace),
                pelvis_z_end=round(float(robot.pelvis_z), 3),
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
        print(f"[thr] {json.dumps(rec)}", flush=True)
        (OUT / f"results{args.tag}.json").write_text(json.dumps(results, indent=2))

    env.close()
    print("[thr] DONE")


if __name__ == "__main__":
    main()
