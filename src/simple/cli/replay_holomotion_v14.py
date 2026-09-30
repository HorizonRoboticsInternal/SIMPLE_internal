"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Bit-exact replay of episodes recorded by ``simple.cli.teleop_holomotion_v14 --record`` (the exact log in
``<dataset>/replay/episode_XXXXXX.npz``).

    python -m simple.cli.replay_holomotion_v14 data/teleop_holomotion_v14/<env_id>/level-0 --episode 0
    python -m simple.cli.replay_holomotion_v14 <dataset> --all --mode action

Scene: rebuilt through SIMPLE from the saved setup (scene knobs, random setup, SIMPLE's environment_config), then
checked against the recorded model hash; ``--model mjb`` uses the saved compiled model instead (no SIMPLE rebuild, e.g.
after the scene code changed).

Modes:
  physics  the recorded ctrl and applied forces of every physics call, from the recorded start state
  action   the recorded actions (joint targets, kp, kd, Dex3 targets) through the robot's PD (G1Sonic.apply_action),
           checking that each call's ctrl comes out identical, then stepping; applied forces (object pseudo-gravity,
           the safety band while it is up) are inputs taken from the log

Either way every frame's qpos/qvel/act is compared bit for bit with the recording. --video renders the replay's head
and third-person views (the SIMPLE rebuild only).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import mujoco
import numpy as np
import typer

from simple.teleop.holomotion_v14 import exact_log as XL


def _episodes(dataset: Path, episode: int, all_: bool) -> list[Path]:
    rdir = dataset / "replay"
    if all_:
        return sorted(rdir.glob("episode_*.npz"))
    return [rdir / f"episode_{episode:06d}.npz"]


def _build_env(info: dict, setup: dict, headless: bool = True):
    """Rebuild the recorded scene through SIMPLE; returns (env, sonic_env, scene module)."""
    import gymnasium as gym
    from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene

    for k, v in info.get("env_knobs", {}).items():
        os.environ[k] = v
    if info.get("scene_root") and "HOLOBRAIN_SIM_DIR" not in os.environ and os.path.isdir(info["scene_root"]):
        os.environ["HOLOBRAIN_SIM_DIR"] = info["scene_root"]         # the scene code the episode was recorded with
    scene = info["scene"]
    mod = load_scene(scene)
    sonic_config = _load_sonic_config()
    env = gym.make(info["env_id"], sim_mode="mujoco", render_hz=info["render_hz"], physics_dt=info["physics_dt"],
                   headless=headless, max_episode_steps=10 ** 9, sonic_config=sonic_config,
                   target=getattr(mod, "TARGET", None), dr_level=info["dr_level"], success_criteria=0.9)
    sonic_env = env.unwrapped
    robot = sonic_env.task.robot
    from simple.teleop.holomotion_v14.robot_variants import REPO_ROOT, backpack_mjcf, teleop_mjcf, tilt_head_sensor
    tilt = float(info.get("head_tilt_deg", 0.0))
    if "robot" in info:                                   # rebuild the recorded robot model (written deterministically)
        robot.mjcf_path = teleop_mjcf(info["robot"], info.get("backpack_kg", 0.0), tilt_deg=tilt)
    if tilt:
        tilt_head_sensor(sonic_env.task, tilt)
    elif info.get("backpack_kg", 0) > 0:                  # episodes recorded before --robot existed
        robot.mjcf_path = backpack_mjcf(info["backpack_kg"])
    if info.get("robot_mjcf") and robot.mjcf_path != info["robot_mjcf"] and (REPO_ROOT / "data" / info["robot_mjcf"]).exists():
        robot.mjcf_path = info["robot_mjcf"]
    _reset_to(env, sonic_env, mod, info, setup)
    return env, sonic_env, mod


def _reset_to(env, sonic_env, mod, info: dict, setup: dict) -> None:
    """The recorded episode's scene: its setup into the scene module/task, then SIMPLE's saved DR state."""
    from simple.teleop.holomotion_v14.scene_setup import SceneSetups
    setups = SceneSetups(info["scene"], mod, sonic_env.task)
    setups.apply(setup if setup.get("randomized") else setups.nominal())
    env.reset(options={"state_dict": json.loads(info["environment_config"])})


def _replay_action(sonic_env, log: dict) -> dict:
    """Actions through G1Sonic.apply_action (PD recomputed); ctrl checked per call, physics per frame."""
    from simple.core.action import ActionCmd
    eng = sonic_env.mujoco
    m, d, robot = eng.mjModel, eng.mjData, sonic_env.task.robot
    XL.set_state(m, d, log["state0"], XL.INTEGRATION)
    mujoco.mj_forward(m, d)
    rest_at = {int(i): v for i, v in zip(log["rest_idx"], log["rest_val"])}
    call, first_bad, first_ctrl_bad, max_diff = 0, None, None, 0.0
    for t, n_calls in enumerate(log["frame_calls"]):
        cmd = ActionCmd("holomotion", target_q=log["action_target_q"][t], kp=log["action_kp"][t], kd=log["action_kd"][t],
                        left_hand_q=log["action_left_hand_q"][t], right_hand_q=log["action_right_hand_q"][t],
                        apply_elastic_band=False)
        for _ in range(int(n_calls)):
            if call in rest_at:
                XL.set_state(m, d, rest_at[call], XL.REST)
            robot.apply_action(cmd)
            if d.ctrl.tobytes() != log["ctrl"][call].tobytes() and first_ctrl_bad is None:
                first_ctrl_bad = call
            mujoco.mj_step(m, d, nstep=int(log["nstep"][call]))
            call += 1
        got = XL.get_state(m, d, XL.PHYSICS)
        if got.tobytes() != log["physics"][t].tobytes():
            max_diff = max(max_diff, float(np.max(np.abs(got - log["physics"][t]))))
            if first_bad is None:
                first_bad = t
    return dict(frames=len(log["frame_calls"]), exact=first_bad is None and first_ctrl_bad is None,
                first_mismatch=first_bad, first_ctrl_mismatch=first_ctrl_bad, max_abs_diff=max_diff)


def _video_writer(path: Path, fps: int):
    import imageio.v2 as imageio
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1)


def main(
    dataset: Annotated[Path, typer.Argument(help="the LeRobot dataset dir (…/<env_id>/level-N) holding replay/")],
    episode: Annotated[int, typer.Option(help="episode index")] = 0,
    all: Annotated[bool, typer.Option("--all", help="every episode with an exact log")] = False,
    mode: Annotated[str, typer.Option(help="physics | action")] = "physics",
    model: Annotated[str, typer.Option(help="sim (rebuild through SIMPLE, check the hash) | mjb (the saved model)")] = "sim",
    video: Annotated[str, typer.Option(help="write an mp4 of the replay (head camera | third person), sim model only")] = "",
):
    if mode not in ("physics", "action"):
        raise typer.BadParameter("--mode is physics or action")
    if mode == "action" and model != "sim":
        raise typer.BadParameter("--mode action needs the SIMPLE rebuild (--model sim)")
    paths = _episodes(dataset, episode, all)
    if not paths or not paths[0].exists():
        sys.exit(f"no exact log: {paths[0] if paths else dataset / 'replay'}")
    env = sonic_env = mod = None
    results = []
    for path in paths:
        log = XL.load(path)
        info, setup = log["info"], log["setup"]
        if model == "mjb":
            m = mujoco.MjModel.from_binary_path(str(path.with_suffix(".mjb")))
            d = mujoco.MjData(m)
            same_model = XL.model_sha256(m) == info["model_sha256"]
        else:
            if env is None:
                env, sonic_env, mod = _build_env(info, setup)
                first = info
            else:
                if info.get("scene") != first.get("scene") or info.get("backpack_kg") != first.get("backpack_kg"):
                    sys.exit("--all over episodes of different scenes / robots: replay them separately")
                _reset_to(env, sonic_env, mod, info, setup)
            m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
            same_model = XL.model_sha256(m) == info["model_sha256"]
        if not same_model:
            print(f"[replay] {path.name}: WARNING the rebuilt model differs from the recorded one "
                  f"(scene code changed?) -- try --model mjb")
        writer, render = None, None
        if video and model == "sim":
            writer = _video_writer(Path(video) if not all else Path(video).with_name(f"{Path(video).stem}_{path.stem}.mp4"),
                                   int(info["render_hz"]))
            renderer = mujoco.Renderer(m, 360, 640)
            third = mujoco.MjvCamera()
            third.type, third.distance, third.azimuth, third.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, 3.0, 150, -25
            third.trackbodyid = m.body("pelvis").id
            head = next((c for c in range(m.ncam) if "head" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) or "")), 0)

            def render(t):
                renderer.update_scene(d, camera=head)
                a = renderer.render()
                renderer.update_scene(d, camera=third)
                writer.append_data(np.concatenate([a, renderer.render()], axis=1))
        if mode == "physics":
            res = XL.replay_physics(m, d, log, on_frame=render)
        else:
            res = _replay_action(sonic_env, log)
        if writer is not None:
            writer.close()
        res.update(episode=path.stem, same_model=same_model, seed=setup.get("seed"))
        results.append(res)
        verdict = "BIT-EXACT" if res["exact"] else f"DIFFERS from frame {res['first_mismatch']} (max |diff| {res['max_abs_diff']:.3g})"
        print(f"[replay] {path.name}: {res['frames']} frames, {mode} replay, model {'identical' if same_model else 'DIFFERENT'}: {verdict}",
              flush=True)
    n_ok = sum(r["exact"] for r in results)
    print(f"[replay] {n_ok}/{len(results)} episodes bit-exact")
    if env is not None:
        env.close()
    os._exit(0 if n_ok == len(results) else 1)


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
