import typer
from typing_extensions import Annotated
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import simple.envs as _# import all envs
from simple.mp.curobo import CuRoboPlanner
from simple.agents.mp import MotionPlannerAgent
from simple.envs.lerobot import LerobotRecorder
import traceback
import sys
import os
os.environ["_TYPER_STANDARD_TRACEBACK"]="1"
import json
import shutil
from pathlib import Path
PRERESET_TASK_STATE = "" #"examples/demo_task_state_dict.json" # 


def _pin_base_materials(task, env_conf, keep_keys=("target", "container")):
    """Pin the training base's object/robot shader params so re-sampling only touches table/ground.

    Object shader params are stored as a list in layout order over ObjectActors, i.e. every
    actor except the tables and the robot. If the base carries no material state (some
    training sets were recorded without material randomization), the training look is the
    renderer default, so pin that."""
    DEFAULT = {"reflection_roughness_constant": 0.5, "metallic_constant": 0.0, "specular_level": 0.0}
    try:
        mat = (env_conf.get("dr_state_dict") or {}).get("material") or {}
        actors = (env_conf.get("layout") or {}).get("actors") or []
        keys = [a.get("key") or a.get("name") for a in actors] if isinstance(actors, list) else list(actors.keys())
        obj_keys = [k for k in keys if not str(k).startswith("table") and k != "robot" and not str(k).startswith("light")]
        params = mat.get("object_shader_params") or []
        keep = {k: v for k, v in zip(obj_keys, params) if k in keep_keys} if params else {}
        # The layout is authoritative: it stores the shader params each actor was actually
        # rendered with, even for training sets whose DR state has no material entry.
        if isinstance(actors, dict):
            for k in keep_keys:
                m = (actors.get(k) or {}).get("material")
                if isinstance(m, dict) and "reflection_roughness_constant" in m:
                    keep[k] = dict(m)
        for k in keep_keys:
            if k in obj_keys and k not in keep:
                keep[k] = dict(DEFAULT)
        rnd = task.dr.randomizers.get("material")
        rnd.cfg.keep_object_shader_params = keep
        rnd.cfg.keep_robot_shader_params = mat.get("robot_shader_params") or dict(DEFAULT)
        print(f"[dr] pinned shader params for {sorted(keep)}: {keep}")
        return keep
    except Exception as exc:  # noqa: BLE001
        print(f"[dr] could not pin base materials: {exc}")
        return {}


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/FrankaTabletopGrasp-v0",
    scene_uid: Annotated[str, typer.Option()] = "hssd:scene1",
    target_object: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    headless: Annotated[bool, typer.Option()] = False,
    webrtc: Annotated[bool, typer.Option()] = True,  # Enable WebRTC streaming
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 30, # FIXME
    data_format: Annotated[str, typer.Option()] = "lerobot",
    save_dir: Annotated[str, typer.Option()] = "data/datagen",
    num_episodes: Annotated[int, typer.Option()] = 100,
    shard_size: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    plan_batch_size: Annotated[int, typer.Option()] = 1,
    ignore_target_collision: Annotated[bool, typer.Option()] = False,
    debug: Annotated[bool, typer.Option()] = False,
    easy_motion_gen: Annotated[bool, typer.Option()] = False,
    env_config_dir: Annotated[str | None, typer.Option()] = None,
    keep_object_materials: Annotated[bool, typer.Option()] = False,
):
    # create environment
    make_kwargs = dict(
        scene_uid=scene_uid,
        target_object=target_object,
        sim_mode=sim_mode,
        headless=headless,
        webrtc=webrtc,
        max_episode_steps=max_episode_steps,
        render_hz=render_hz,
        # dr_level=dr_level,
    )
    if "Sonic" in env_id or "Teleop" in env_id:
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        sonic_config = SimLoopConfig().load_wbc_yaml()
        sonic_config["ENV_NAME"] = "simple"
        make_kwargs["sonic_config"] = sonic_config
    env = gym.make(env_id, **make_kwargs)
    task = env.unwrapped.task  # type: ignore

    env_configs = None

    def _load_env_configs(path: str) -> list[dict]:
        config_path = Path(path)
        if config_path.is_dir():
            candidate = config_path / "meta" / "episodes.jsonl"
            if candidate.exists():
                config_path = candidate
        if not config_path.exists():
            raise FileNotFoundError(f"env_config path not found: {config_path}")
        with open(config_path, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        configs = []
        for entry in lines:
            env_json = entry.get("environment_config")
            if env_json is None:
                continue
            if isinstance(env_json, str):
                configs.append(json.loads(env_json))
            else:
                configs.append(env_json)
        return configs

    if env_config_dir:
        env_configs = _load_env_configs(env_config_dir)
        if not env_configs:
            raise ValueError(f"No environment_config found in {env_config_dir}")
        if num_episodes > len(env_configs):
            print(
                f"[Eval] Requested {num_episodes} episodes but only "
                f"{len(env_configs)} env configs found; will loop."
            )

    recorder = LerobotRecorder(env=env, root_dir=save_dir, agent=None, debug=True, dr_level=dr_level)

    for i in tqdm(range(num_episodes), desc="Generating eval env configs"):
        if env_configs:
            env_conf = env_configs[i % len(env_configs)]
            if keep_object_materials:
                _pin_base_materials(task, env_conf)
            obs, info = env.reset(options={"state_dict": env_conf, "dr_level": dr_level})
            # Save the post-reset state (level-k components re-sampled), not the input base
            # config -- otherwise the written eval scene is just the training scene again.
            env_conf = task.state_dict()
        else:
            obs, info = env.reset(options={"state_dict": None, "dr_level": dr_level})
            env_conf = task.state_dict()
        recorder.dataset.clear_episode_buffer()

        # add a single dummy frame with all registered features
        frame = {}
        for feat_key, feat_info in recorder.dataset.meta.features.items():
            if feat_key in ("index", "episode_index", "frame_index", "timestamp", "task_index"):
                continue
            shape = tuple(feat_info["shape"])
            dtype = np.uint8 if feat_info["dtype"] == "video" else np.dtype(feat_info["dtype"])
            # use real observation if available
            obs_short = feat_key.replace("observation.rgb_", "").replace("observation.", "")
            if obs_short in obs:
                frame[feat_key] = obs[obs_short]
            else:
                frame[feat_key] = np.zeros(shape, dtype=dtype)

        recorder.dataset.add_frame(frame, task=task.instruction)
        recorder.dataset.save_episode()
        recorder.write_env_config(env_conf, i)
        print(f"[Eval] Generated env config {i + 1}/{num_episodes}")

    print(f"[Eval] Saved {num_episodes} env configs as lerobot dataset at {recorder.root_dir}")
    env.close()
    return


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
