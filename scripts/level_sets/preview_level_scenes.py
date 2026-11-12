#!/usr/bin/env python
"""MuJoCo third-person stills of a kit level set's scenes (exact reload, no Isaac), for scene review pages.

    MUJOCO_GL=egl .venv/bin/python scripts/level_sets/preview_level_scenes.py <kit> --src <set dir> --out <dir> [--scenes 0,1,2] [--cam top|iso]
The kit's env vars come from <src>/meta/scene_env.json (applied before the kit task is imported). Writes <out>/sceneNN_<cam>.jpg.
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
KITS = Path.home() / "wrk/robot_orchard_deploy/holobrain_g1_deploy/sim"
ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scenes", default=None); ap.add_argument("--cam", default="iso"); ap.add_argument("--size", default="960x540")
ap.add_argument("--lookat", default=None); ap.add_argument("--dist", type=float, default=None); ap.add_argument("--az", type=float, default=None); ap.add_argument("--el", type=float, default=None)
a = ap.parse_args(); SRC, OUT = Path(a.src), Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
se = json.load(open(SRC / "meta/scene_env.json"))
for k, v in se.get("env", {}).items(): os.environ.setdefault(k, v)
sys.path.insert(0, str(KITS / a.kit)); import importlib, numpy as np, gymnasium as gym, mujoco
from PIL import Image, ImageDraw
mod = importlib.import_module(f"{a.kit}_task")
from simple.cli.dr_decoupled_wbc import _make_sonic_config
eps = [json.loads(l) for l in open(SRC / "meta/episodes.jsonl")]
want = [int(x) for x in a.scenes.split(",")] if a.scenes else list(range(len(eps)))
W, H = (int(x) for x in a.size.split("x"))
env = gym.make(mod.ENV_ID, sim_mode="mujoco", headless=True, render_hz=50, max_episode_steps=10, sonic_config=_make_sonic_config())
renderer = None
for i in want:
    cfg = eps[i]["environment_config"]; cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    env.reset(seed=i, options={"state_dict": cfg, "dr_level": None})
    m, d = env.unwrapped.mujoco.mjModel, env.unwrapped.mujoco.mjData
    if renderer is None: renderer = mujoco.Renderer(m, height=H, width=W)
    else:
        renderer.close(); renderer = mujoco.Renderer(m, height=H, width=W)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    pelvis = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    # views: iso/top are robot-centred (bowl_sink counter + sink); route_iso/route_top frame the coffee route (start -> cart
    # -> table -> the wall 1.0 m beyond it, x ~3.85); --lookat/--dist/--az/--el override any of them
    views = {"top": ([pelvis[0] + 0.3, pelvis[1] - 0.6, 0.6], 5.0, 90, -89), "iso": ([pelvis[0] + 0.2, pelvis[1] - 0.5, 0.7], 4.6, 150, -32),
             "route_iso": ([1.4, -0.5, 0.7], 5.2, 35, -18), "route_top": ([2.0, -0.8, 0.6], 9.5, 90, -89)}   # route_iso = coffee_cart/probe_scene.py THIRD (behind the robot's right)
    look, dist, az, el = views[a.cam]
    if a.lookat: look = [float(x) for x in a.lookat.split(",")]
    cam.lookat[:] = look; cam.distance = a.dist or dist; cam.azimuth = a.az if a.az is not None else az; cam.elevation = a.el if a.el is not None else el
    opt = mujoco.MjvOption(); renderer.update_scene(d, camera=cam, scene_option=opt); img = renderer.render()
    im = Image.fromarray(img); dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 260, 22], fill=(0, 0, 0)); dr.text((6, 5), f"{a.kit} scene {i}  ({a.cam})", fill=(255, 255, 255))
    im.save(OUT / f"scene{i:02d}_{a.cam}.jpg", quality=85)
    print(f"[preview] scene {i} -> {OUT / f'scene{i:02d}_{a.cam}.jpg'}", flush=True)
env.close(); sys.stdout.flush(); os._exit(0)
