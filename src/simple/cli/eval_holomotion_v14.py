"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

VLA evaluation with the HoloMotion v1.4 controller in the real-to-sim scenes (bottle_bin by default):

    python -m simple.cli.eval_holomotion_v14 --scene bottle_bin --host 127.0.0.1 --port 21000     # the 30 level-3 scenes

The loop is simple.cli.eval_decoupled_wbc's (the VLA behind HttpActionClient's POST /act, action chunks, success from
the task check, eval_stats.txt, one video per episode) with the controller swapped: instead of the decoupled WBC turning
upper-body targets + a navigate command into joint targets, HoloMotion v1.4's motion-tracking policy (model_22000, the
3.2 kg backpack model) tracks the reference frames the VLA returns (agents/holomotion_v14_vla_agent.py has the reply
layout). Everything else is the teleop's default collection setting, so the evaluation sees what the data was
recorded with:

    robot     G1 + Dex3 + HBVCAM stereo head camera + 3.2 kg backpack (teleop_mjcf("stereo", 3.2)), camera tilt 10 deg
    scenes    the kit's level-3 evaluation set (default: data/evals_scenes/<env>/dr-level-3, the 30 checked bottle_bin
              scenes), loaded exactly (environment_config) with the whole scene moved away from the robot so its feet
              start --start-distance (the teleop's 0.35 m) from the table instead of the replay-fit 3 cm; or fresh
              level-3 setups from --seed (--eval-set seeds); or recorded teleop episodes (--scenes-from)
    start     quick start: standing on the floor, no leash, a random start pose, walking policy on
    VLA image the rectified left eye, 1280 x 720 (what the teleop records as ego_view)

The controller runs on a clock that moves 20 ms per control step (teleop/holomotion_v14/sim_clock.py), so VLA latency
does not change the rollout. Each episode also gets a bit-exact replay log (replay/episode_N.npz), replayable with
    python -m simple.cli.replay_holomotion_v14 <run dir> --all --mode action
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import typer

from simple.cli.teleop_holomotion_v14 import (
    SCENE_ENV_PREFIX, SCENES, _draw_init_pose, _env_config_json, _load_sonic_config, _stand_on_floor, load_scene,
    scene_root,
)
from simple.teleop.holomotion_v14.scene_setup import SceneSetups

GATES = ("at_bowl", "cart_pushed", "grasped", "cup_lifted", "at_bin", "at_basin", "at_cart", "placed")


def _progress(info: dict) -> tuple[dict, dict]:
    """The task's gates (and the times they were met): bottle_bin / bowl_sink report info["task_progress"],
    coffee_cart info["progress"] (cart pushed, cup lifted, success = placed)."""
    p = info.get("task_progress")
    if p:
        return ({g: bool(p[g]) for g in GATES if g in p},
                {k: p[k] for k in p if k.startswith("t_") and p[k] is not None})
    p = info.get("progress")
    if p and "cart_pushed_ever" in p:
        return {"cart_pushed": bool(p["cart_pushed_ever"]), "cup_lifted": bool(p["cup_lifted_ever"]),
                "placed": bool(p["success"])}, {}
    return {}, {}
FELL_BELOW_M = 0.5            # pelvis height (m); the robot stands at ~0.76


def _fetch_policy_info(host: str, port: int, timeout: float = 5.0) -> dict:
    """GET /info from the policy server (optional endpoint; {} if absent), as eval_decoupled_wbc does."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/info", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[eval] no policy info from http://{host}:{port}/info ({e})")
        return {}


def _recorded_scenes(dataset: Path) -> list[dict]:
    """The setups, start poses and exact scenes (environment_config) of a teleop dataset's episodes."""
    out = []
    for line in open(dataset / "meta" / "episodes.jsonl"):
        e = json.loads(line)
        s, ip = e.get("scene_setup"), e.get("init_pose")
        s = json.loads(s) if isinstance(s, str) else s
        ip = json.loads(ip) if isinstance(ip, str) else ip
        if s:
            out.append(dict(label=f"episode_{int(e['episode_index'])}", index=int(e["episode_index"]), setup=s, init_pose=ip,
                            environment_config=e.get("environment_config")))
    return out


TABLE_KEY = {"bottle_bin": "table_cx", "bowl_sink": "counter_cx", "coffee_cart": "table_cx"}   # the SIMPLE table's x in L
# coffee_cart: standing in the v1.4 default pose the fingers reach the cart handle ~5 cm ahead of the nominal start, and
# 6 of the 30 level-3 scenes start the robot up to 10 cm ahead (2026-09-30 check). As the teleop's lv3 coffee_cart
# ranges (robot x -0.20..0), the robot starts 10 cm further back in those scenes; the rest of each scene is unchanged.
ROBOT_BACK = {"coffee_cart": 0.10}


def _eval_set_scenes(set_dir: Path, scene: str, setups: SceneSetups) -> list[dict]:
    """A level-3 eval set's scenes (make_levels / build_kit_lv3 output: meta/episodes.jsonl environment_config), moved
    along x so they sit where the teleop's start distance puts them: the robot keeps its level-3 start (near the
    origin, as in the teleop data), the table, target, distractors and (Isaac) lights shift by the difference between
    the kit layout as imported now and the table in the saved scene. The kit's own furniture (legs, cover board, bin)
    follows from the imported layout."""
    import copy
    robot_keys = ("g1_sonic", "g1_wholebody")
    out = []
    for line in open(set_dir / "meta" / "episodes.jsonl"):
        e = json.loads(line)
        cfg = json.loads(e["environment_config"]) if isinstance(e["environment_config"], str) else e["environment_config"]
        dr = cfg["dr_state_dict"]
        dx = float(setups.nominal_L[TABLE_KEY[scene]]) - float(dr["scene"]["table"]["pose"]["position"][0])
        cfg = copy.deepcopy(cfg)
        dr = cfg["dr_state_dict"]
        for k, v in (dr.get("spatial") or {}).items():
            if k not in robot_keys and isinstance(v, dict) and "position" in v:
                v["position"][0] = float(v["position"][0]) + dx
        dr["scene"]["table"]["pose"]["position"][0] = float(dr["scene"]["table"]["pose"]["position"][0]) + dx
        for v in (dr.get("lighting") or {}).values():
            if isinstance(v, dict) and isinstance(v.get("pose"), dict) and "position" in v["pose"]:
                v["pose"]["position"][0] = float(v["pose"]["position"][0]) + dx
        robot = next((dr["spatial"][k] for k in robot_keys if k in (dr.get("spatial") or {})), {"position": [0, 0, 0]})
        if ROBOT_BACK.get(scene) and "position" in robot:
            robot["position"][0] = float(robot["position"][0]) - ROBOT_BACK[scene]
        idx = int(e["episode_index"])
        setup = dict(setups.nominal(), eval_set=str(set_dir), eval_scene=idx, moved_m=round(dx, 4),
                     robot=dict(x=round(float(robot["position"][0]), 4), y=round(float(robot["position"][1]), 4), yaw_deg=0.0),
                     target_xy=[round(float(v["position"][0]), 4) for k, v in dr["spatial"].items() if k not in robot_keys][:1] +
                     [round(float(v["position"][1]), 4) for k, v in dr["spatial"].items() if k not in robot_keys][:1],
                     table_dz=round(float(dr["scene"]["table"]["pose"]["position"][2]) + float(dr["scene"]["table"]["size"][2]) / 2
                                    - float(setups._nominal_table_height.middle()), 3),
                     distractors=len([k for k in dr["spatial"] if k not in robot_keys]) - 1)
        out.append(dict(label=f"scene_{idx:02d}", index=idx, setup=setup, init_pose=None, environment_config=cfg, moved_m=dx))
    return out


def _third_person(sonic_env, L: dict, renderer_box: dict, size=(640, 360)) -> np.ndarray:
    """A fixed oblique view of the work area (table, bin), for the episode video."""
    import mujoco
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    if renderer_box.get("model") is not m:
        if renderer_box.get("r") is not None:
            renderer_box["r"].close()
        from simple.teleop.holomotion_v14.hbvcam_views import _renderer
        renderer_box.update(r=_renderer(m, size[1], size[0]), model=m)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [L.get("plan_cx", 0.6), L.get("plan_cy", 0.0), 0.5]
    cam.distance, cam.azimuth, cam.elevation = 3.4, 200.0, -38.0
    renderer_box["r"].update_scene(d, camera=cam)
    return renderer_box["r"].render()


def main(
    scene: Annotated[str, typer.Option(help=f"real-to-sim scene: {', '.join(SCENES)}")] = "bottle_bin",
    host: Annotated[str, typer.Option(help="VLA server (HttpActionClient: POST /act, optional GET /info)")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 21000,
    policy_name: Annotated[str, typer.Option(help="name for the output folder (the server's /info policy+timestamp, when given, prefixes the task)")] = "vla",
    eval_dir: Annotated[str, typer.Option()] = "data/evals_holomotion_v14",
    eval_set: Annotated[str, typer.Option(help="lv3 = data/evals_scenes/<env>/dr-level-3 (the level-3 scene set), a LeRobot eval-set dir, or seeds (fresh --setup-ranges draws from --seed)")] = "lv3",
    num_episodes: Annotated[int, typer.Option(help="episodes to run (-1 = the whole eval set; 16 with seeds)")] = -1,
    episode_start: Annotated[int, typer.Option(help="first episode (the set's order; with seeds, setup seed = --seed + index)")] = 0,
    seed: Annotated[int, typer.Option(help="seed base: episode i's start pose (and with --eval-set seeds its setup) uses seed + i")] = 10_000_000,
    setup_ranges: Annotated[str, typer.Option(help="lv3 (the level-3 eval ranges, the teleop default) or teleop")] = "lv3",
    scenes_from: Annotated[str, typer.Option(help="a teleop LeRobot dataset: evaluate on its episodes' exact scenes and start poses (overrides --eval-set)")] = "",
    random_init_pose: Annotated[bool, typer.Option(help="a random start pose per episode (seeded by the episode's setup seed), as the teleop default")] = True,
    init_pose_scale: Annotated[float, typer.Option()] = 1.0,
    start_distance: Annotated[float, typer.Option(help="robot start: feet to the table/counter/cart (m); -1 = the teleop default")] = -1.0,
    backpack_kg: Annotated[float, typer.Option()] = 3.2,
    motion_model: Annotated[str, typer.Option(help="backpack (model_22000) or public (model_16200)")] = "backpack",
    motion_model_bundle: Annotated[str, typer.Option(help="another backpack model bundle dir; overrides --motion-model")] = "",
    robot_kind: Annotated[str, typer.Option("--robot", help="stereo (G1 + HBVCAM head camera) or stock")] = "stereo",
    head_tilt_deg: Annotated[float, typer.Option(help="head cameras this many degrees further down, as the teleop default")] = 10.0,
    image_camera: Annotated[str, typer.Option(help="VLA image: pinhole (1280x720 rectified left eye, the teleop's recorded view) or fisheye (both eyes, 2560x720)")] = "pinhole",
    image_size: Annotated[str, typer.Option(help="resize the VLA image to WxH before sending (default: as rendered)")] = "",
    image_key: Annotated[str, typer.Option(help="the image's key in the request (the dataset's)")] = "observation.images.ego_view",
    instruction: Annotated[str, typer.Option(help="task instruction (default: the scene task's)")] = "",
    reference_hz: Annotated[float, typer.Option(help="rate of the VLA's reference frames; 50 = one per control step, as the dataset's teleop.latest_obs")] = 50.0,
    settle_s: Annotated[float, typer.Option(help="extra standing on the walking policy before the VLA is first asked (0: ask from the first step; walking mode lasts until the VLA's frames are valid)")] = 0.0,
    engage_timeout_s: Annotated[float, typer.Option(help="seconds to wait for valid VLA frames (walking) before the episode fails")] = 10.0,
    max_episode_steps: Annotated[int, typer.Option(help="control steps per episode (50 per second)")] = 1500,
    success_criteria: Annotated[float, typer.Option(help="task reward for success (bottle_bin: 1.0 placed, 0.6 grasped + at bin)")] = 0.9,
    save_video: Annotated[bool, typer.Option(help="per episode: the VLA image | a third-person view")] = True,
    exact_log: Annotated[bool, typer.Option(help="a bit-exact replay log per episode (replay/)")] = True,
    headless: Annotated[bool, typer.Option()] = True,
):
    import mujoco  # noqa: F401  (EGL context before the renderers)
    from simple.baselines.client import HttpActionClient
    from simple.teleop.holomotion_v14 import BACKPACK_MODELS_DIR, MODELS_DIR
    from simple.teleop.holomotion_v14.exact_log import ExactLog
    from simple.teleop.holomotion_v14.hbvcam_views import FisheyeView, PinholeView
    from simple.teleop.holomotion_v14.robot_variants import teleop_mjcf, tilt_head_sensor
    from simple.teleop.holomotion_v14.sim_clock import ControllerClock

    if scene not in SCENES:
        raise typer.BadParameter(f"--scene is one of {sorted(SCENES)}")
    if image_camera not in ("pinhole", "fisheye") or robot_kind not in ("stereo", "stock"):
        raise typer.BadParameter("--image-camera is pinhole or fisheye; --robot is stereo or stock")
    if robot_kind != "stereo":
        raise typer.BadParameter("the VLA image is the HBVCAM camera: --robot stereo")
    if motion_model_bundle:
        from simple.teleop.holomotion_v14.robot_variants import backpack_models_dir
        models_dir = backpack_models_dir(motion_model_bundle)
    else:
        models_dir = {"backpack": BACKPACK_MODELS_DIR, "public": MODELS_DIR}.get(motion_model)
        if models_dir is None or not (models_dir / "motion_tracking_model").is_dir():
            raise typer.BadParameter(f"--motion-model {motion_model}: not set up (see docs/holomotion_v14_teleop.md)")
    os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = str(models_dir)
    resize = tuple(int(v) for v in image_size.lower().split("x")) if image_size else None

    # ---- the eval set: its generation env (make_levels' scene_env.json) before the kit is imported, as eval_scene.py
    set_dir = None
    if not scenes_from and eval_set != "seeds":
        env_id_guess = {"bottle_bin": "G1WholebodyBottleBinTeleop-v0", "bowl_sink": "G1WholebodyBowlSinkTeleop-v0",
                        "coffee_cart": "G1WholebodyCoffeeCartTeleop-v0"}[scene]
        set_dir = (Path(__file__).resolve().parents[3] / "data" / "evals_scenes" / env_id_guess / "dr-level-3"
                   if eval_set == "lv3" else Path(eval_set))
        if not (set_dir / "meta" / "episodes.jsonl").exists():
            raise typer.BadParameter(f"no eval set at {set_dir}")
        meta = set_dir / "meta" / "scene_env.json"
        for k, v in (json.load(open(meta)).get("env", {}) if meta.exists() else {}).items():
            os.environ.setdefault(k, v)

    # ---- scene, robot, cameras: the teleop's defaults
    mod = load_scene(scene, None if start_distance < 0 else start_distance)
    sonic_config = _load_sonic_config()
    render_hz = 50
    dr_level = 3 if setup_ranges == "lv3" else 0
    env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=render_hz, physics_dt=sonic_config["SIMULATE_DT"],
                   headless=headless, max_episode_steps=10 ** 9, sonic_config=sonic_config,
                   target=getattr(mod, "TARGET", None), dr_level=dr_level, success_criteria=success_criteria)
    sonic_env = env.unwrapped
    task, robot = sonic_env.task, sonic_env.task.robot
    robot.mjcf_path = teleop_mjcf(robot_kind, backpack_kg, tilt_deg=head_tilt_deg)
    if head_tilt_deg:
        tilt_head_sensor(task, head_tilt_deg)
    control_dt = 4 * robot.sim_dt
    view = FisheyeView(sonic_env) if image_camera == "fisheye" else PinholeView(sonic_env)
    cache = {"step": None, "img": None}
    step_id = {"n": 0}

    def vla_image() -> np.ndarray:
        """The VLA image of the current (pre-step) state, rendered once per step."""
        if cache["step"] != step_id["n"]:
            img = np.hstack(view.render()) if image_camera == "fisheye" else view.render()
            if resize:
                import cv2
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            cache.update(step=step_id["n"], img=img)
        return cache["img"]

    # ---- VLA and controller
    from simple.agents.holomotion_v14_vla_agent import HoloMotionV14VlaAgent
    clock = ControllerClock()
    client = HttpActionClient(host, port)
    agent = HoloMotionV14VlaAgent(robot, client=client, image_fn=vla_image, clock=clock, reference_hz=reference_hz,
                                  settle_s=settle_s, engage_timeout_s=engage_timeout_s, image_key=image_key)
    clocked = clock.install()
    print(f"[eval] controller clock: {len(clocked)} modules on sim time; policies from {models_dir}")
    print(f"[eval] robot {robot.mjcf_path}; VLA image {image_camera}{' -> ' + image_size if resize else ''}; "
          f"reference frames at {reference_hz:g} Hz")

    # ---- episodes: seeded setups or a dataset's recorded ones
    last = None if num_episodes < 0 else episode_start + num_episodes
    if scenes_from:
        episodes = _recorded_scenes(Path(scenes_from))[episode_start:last]
        scene_tag = f"from_{Path(scenes_from).parent.name}_{Path(scenes_from).name}"
        setups = SceneSetups(scene, mod, task)
    elif set_dir is not None:
        setups = SceneSetups(scene, mod, task)
        episodes = _eval_set_scenes(set_dir, scene, setups)[episode_start:last]
        for ep in episodes:
            ep["init_seed"] = ((seed + ep["index"]) * 7919 + 17) % 2 ** 32 if random_init_pose else None
        scene_tag = f"{set_dir.name}" if eval_set != "lv3" else "dr-level-3"
        print(f"[eval] eval set {set_dir}: {len(episodes)} scenes; scene moved {episodes[0]['moved_m']:+.3f} m away from "
              f"the robot, robot {ROBOT_BACK.get(scene, 0.0):.2f} m further back" if episodes else f"[eval] eval set {set_dir}: no scenes")
    else:
        setups = SceneSetups(scene, mod, task, ranges=setup_ranges)
        episodes = []
        for i in range(episode_start, episode_start + (16 if num_episodes < 0 else num_episodes)):
            s = setups.sample(seed + i)
            episodes.append(dict(label=f"episode_{i}", index=i, setup=s, init_pose=None,
                                 init_seed=((seed + i) * 7919 + 17) % 2 ** 32 if random_init_pose else None))
        scene_tag = f"{setup_ranges}_seed{seed}"

    info = _fetch_policy_info(host, port)
    task_component = mod.ENV_ID.split("/")[-1]
    if info.get("policy") and info.get("timestamp"):
        task_component = f"{info['policy']}-{info['timestamp']}.{task_component}"
    out_dir = Path(eval_dir) / policy_name / task_component / scene_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {len(episodes)} episodes -> {out_dir}")

    prefix = SCENE_ENV_PREFIX.get(scene, "\0")
    run_meta = dict(scene=scene, scene_root=str(scene_root()), env_id=mod.ENV_ID, dr_level=dr_level,
                    env_knobs={k: v for k, v in os.environ.items() if k.startswith(prefix) or k.startswith("SIMPLE_")},
                    setup_ranges=None if scenes_from else setup_ranges, backpack_kg=backpack_kg, robot=robot_kind,
                    head_tilt_deg=head_tilt_deg, robot_mjcf=robot.mjcf_path, physics_dt=sonic_config["SIMULATE_DT"],
                    render_hz=render_hz, policy_models_dir=str(models_dir), velocity_onnx=str(agent.node.velocity_onnx_path),
                    motion_onnx=str(agent.node.motion_onnx_path), eval=dict(host=host, port=port, policy=policy_name,
                    server_info=info, image_camera=image_camera, image_size=image_size, reference_hz=reference_hz))
    xlog = ExactLog(sonic_env) if exact_log else None
    results_path = out_dir / "results.json"
    results = []
    third_box: dict = {}

    try:
        for ep in episodes:
            label = ep["label"]
            setups.apply(ep["setup"])
            if ep.get("environment_config"):                   # a recorded episode: its exact scene, as the replay loads it
                cfg = ep["environment_config"]
                observation, privileged_info = env.reset(options={"state_dict": json.loads(cfg) if isinstance(cfg, str) else cfg})
            else:
                observation, privileged_info = env.reset()
            if ep.get("init_seed") is not None:                # drawn after the reset: it needs the loaded robot's joints
                ep["init_pose"] = _draw_init_pose(np.random.RandomState(ep["init_seed"]), robot, init_pose_scale)
            placed = _stand_on_floor(sonic_env, agent, ep["init_pose"])
            observation, privileged_info = sonic_env._get_obs(), sonic_env._get_info()
            agent.reset_episode(placed[agent.mjcf_to_real], episode_index=int(ep["index"]))
            instr = instruction or task.instruction
            if ep is episodes[0]:
                print(f"[eval] instruction: {instr!r}")
            desc = SceneSetups.describe(ep["setup"])
            print(f"[eval] {label}: " + (f"level-3 set scene {ep['index']}: " + desc.split(': ', 1)[1] if "moved_m" in ep else desc))
            writer = None
            if save_video:
                import imageio.v2 as imageio
                (out_dir / "videos").mkdir(exist_ok=True)
                writer = imageio.get_writer(str(out_dir / "videos" / f"{label}.mp4"),
                                            fps=render_hz, codec="libx264", quality=7, macro_block_size=8,
                                            ffmpeg_params=["-pix_fmt", "yuv420p"])
            t_wall = time.perf_counter()
            success, steps, error = False, 0, None
            pelvis_min = float(privileged_info["proprio"]["floating_base_pose"][2])
            try:
                for steps in range(1, max_episode_steps + 1):
                    step_id["n"] += 1
                    if writer is not None:
                        vla_img = vla_image()
                    action = agent.get_action(observation, instruction=instr, privileged_info=privileged_info)
                    if xlog is not None:
                        xlog.before_step()
                        if steps == 1:
                            xlog.start()
                    observation, reward, terminated, truncated, privileged_info = env.step(action)
                    if xlog is not None:
                        xlog.add_frame(action)
                    clock.advance(control_dt)
                    pelvis_min = min(pelvis_min, float(privileged_info["proprio"]["floating_base_pose"][2]))
                    if not headless:
                        sonic_env.update_viewer()
                    if writer is not None:
                        import cv2
                        left = cv2.resize(vla_img[:, :vla_img.shape[1] // (2 if image_camera == "fisheye" else 1)], (640, 360),
                                          interpolation=cv2.INTER_AREA)
                        frame = np.hstack([left, _third_person(sonic_env, setups.mod.L, third_box)]).copy()
                        text = f"{label}  t={steps * control_dt:5.1f}s  {agent.phase}  " + " ".join(
                            g for g, v in _progress(privileged_info)[0].items() if v)
                        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        writer.append_data(frame)
                    if terminated:
                        success = True
                        break
            except Exception as e:  # noqa: BLE001  (a failing VLA / engage: the episode fails, the run goes on)
                import traceback
                traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            if writer is not None:
                writer.close()
            gates, gate_times = _progress(privileged_info)
            res = dict(episode=label, index=int(ep["index"]), success=bool(success), steps=int(steps),
                       sim_seconds=round(steps * control_dt, 2), wall_seconds=round(time.perf_counter() - t_wall, 1),
                       gates=gates,
                       pelvis_min_m=round(pelvis_min, 3), fell=bool(pelvis_min < FELL_BELOW_M),
                       gate_times=gate_times,
                       seed=ep["setup"].get("seed"), table_dz=ep["setup"].get("table_dz"), error=error, **agent.summary())
            if xlog is not None and xlog.recording:
                log_path = out_dir / "replay" / f"episode_{int(ep['index']):06d}.npz"
                xlog.save(log_path, ep["setup"], dict(run_meta, episode_index=int(ep["index"]), init_pose=ep["init_pose"],
                                                      environment_config=_env_config_json(task), result=res),
                          save_model=False)
                res["exact_log"] = str(log_path.relative_to(out_dir))
            results.append(res)
            with open(Path(eval_dir) / "eval_stats.txt", "a") as f:
                f.write(f"{label}: {success} \n")
            results_path.write_text(json.dumps(dict(run=run_meta, episodes=results), indent=1, default=str))
            gates = " ".join(g for g, v in res["gates"].items() if v) or "-"
            print(f"[eval] {label}: {'SUCCESS' if success else 'fail'}{' (FELL)' if res['fell'] else ''} after {res['sim_seconds']} s "
                  f"(gates: {gates}; {res['queries']} VLA queries, engaged at step {res['engaged_step']})"
                  + (f"; {error}" if error else ""), flush=True)
    except KeyboardInterrupt:
        print("[eval] interrupted")
    n = len(results)
    if n:
        rate = sum(r["success"] for r in results) / n
        gate_rates = {g: sum(r["gates"].get(g, False) for r in results) / n for g in GATES if any(g in r["gates"] for r in results)}
        print(f"[eval] {n} episodes: success {rate:.0%}; " + ", ".join(f"{g} {v:.0%}" for g, v in gate_rates.items()))
        print(f"[eval] results: {results_path}")
    sys.stdout.flush()
    os._exit(0)                                         # skip the slow interpreter / GL teardown


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
