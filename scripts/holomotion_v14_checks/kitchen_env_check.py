#!/usr/bin/env python
"""Builds the kitchen teleop env with the Revo 2 robot (or the Dex3 one), resets it and saves a few views: the robot's
left pinhole eye, SIMPLE's head camera, an overview and a top view.  A quick check of the kit's placement and of the
attached kitchen before a teleop session.
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/kitchen_env_check.py [--robot revo2|stereo] [--out docs/kitchen_teleop/check] [--steps 100]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(); ap.add_argument("--robot", default="revo2"); ap.add_argument("--out", default="docs/kitchen_teleop/check"); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args(); OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
import mujoco, imageio.v2 as imageio
import gymnasium as gym
from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene
mod = load_scene("kitchen"); sonic_config = _load_sonic_config()
robot_uid = "g1_sonic_revo2" if a.robot == "revo2" else "g1_sonic"
t0 = time.time()
env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, physics_dt=sonic_config["SIMULATE_DT"], headless=True, max_episode_steps=10 ** 9,
               sonic_config=sonic_config, target=mod.TARGET, dr_level=0, success_criteria=0.9, robot_uid=robot_uid)
senv = env.unwrapped; robot = senv.task.robot
if a.robot != "revo2":
    from simple.teleop.holomotion_v14.robot_variants import teleop_mjcf
    robot.mjcf_path = teleop_mjcf("stereo", 3.2, tilt_deg=0.0, black_hands=True)
obs, info = env.reset(seed=a.seed)
m, d = senv.mujoco.mjModel, senv.mujoco.mjData
print(f"[check] env built + reset in {time.time() - t0:.1f} s: nbody {m.nbody} ngeom {m.ngeom} nmesh {m.nmesh} nq {m.nq} nu {m.nu}; robot {robot.mjcf_path}")
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
kb = [n for n in names if n.startswith("k_")]; other = [n for n in names if n and not n.startswith("k_") and not any(n.startswith(p) for p in ("left_", "right_", "pelvis", "torso", "waist", "head", "imu", "d455", "backpack", "logo"))]
print(f"[check] kitchen bodies {len(kb)} (e.g. {kb[:4]} ... {kb[-3:]}); other non-robot bodies: {other}")
tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(senv.task.target.asset.label))
exp = (mod.L['bottle_x'], mod.L['bottle_y'], mod.BS.WORKTOP_TOP) if senv.task.target.asset.label not in mod.BS.KITCHEN_OBJECTS else tuple(senv.task.layout.actors["target"].pose.position)
print(f"[check] target '{senv.task.target.asset.label}' at {d.xpos[tb].round(3).tolist()} (expected x ~{exp[0]:.3f}, y ~{exp[1]:+.3f}, z ~{exp[2]:.3f}; {'the kitchen file places it' if senv.task.target.asset.label in mod.BS.KITCHEN_OBJECTS else 'SIMPLE spatial DR'}); pelvis at {d.qpos[:3].round(3).tolist()}")
other = [m.body(i).name for i in range(m.nbody) if not m.body(i).name.startswith(("k_", "left_", "right_", "pelvis", "torso", "waist", "head", "imu")) and m.body(i).name not in ("world", senv.task.target.asset.label)]
print(f"[check] non-kitchen, non-robot bodies: {other[:12]}{' ...' if len(other) > 12 else ''}; SIMPLE bottle present: {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'bottle_500ml') >= 0}; distractor bodies: {sum(1 for i in range(m.nbody) if 'distractor' in m.body(i).name or 'graspnet' in m.body(i).name.lower())}")
print(f"[check] sink centre in the robot frame ({mod.L['sink_x']:.3f}, {mod.L['sink_y']:+.3f}); instruction: {senv.task.instruction!r}")
# settle a little with the robot's own controller holding its pose (no agent): just step physics a few times
for i in range(a.steps):
    mujoco.mj_step(m, d)
print(f"[check] after {a.steps} physics steps: target z {d.xpos[tb][2]:.3f} (worktop top {mod.BS.WORKTOP_TOP:.3f}), pelvis z {d.qpos[2]:.3f}")
m.vis.global_.offwidth = max(m.vis.global_.offwidth, 1920); m.vis.global_.offheight = max(m.vis.global_.offheight, 1080)
def shot(name, cam, W=1280, H=720):
    r = mujoco.Renderer(m, H, W); r.update_scene(d, camera=cam); img = r.render(); r.close(); imageio.imwrite(OUT / f"{name}.jpg", img, quality=88); return img
cams = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i): i for i in range(m.ncam)}
print(f"[check] cameras: {[c for c in cams if 'pinhole' in c or 'head' in c][:6]} (+{len(cams)} total)")
if "hbvcam_left_pinhole" in cams: shot("eye_left_pinhole", cams["hbvcam_left_pinhole"])
for cn in cams:
    if cn.startswith("head_stereo") or cn == "head_stereo_left":
        shot("simple_head_camera", cams[cn], 640, 360); break
free = mujoco.MjvCamera(); free.type = mujoco.mjtCamera.mjCAMERA_FREE
vt = getattr(mod, "VIEW_THIRD", dict(lookat=[0.5, -0.3, 0.9], distance=3.4, azimuth=215, elevation=-22)); free.lookat[:] = vt["lookat"]; free.distance = vt["distance"]; free.azimuth = vt["azimuth"]; free.elevation = vt["elevation"]; shot("overview", free)
free.lookat[:] = [0.3, -0.5, 0.5]; free.distance = 6.0; free.azimuth = 90; free.elevation = -89; shot("top", free)
free.lookat[:] = [0.0, -1.2, 0.8]; free.distance = 2.6; free.azimuth = 160; free.elevation = -20; shot("sink_side", free)
free.lookat[:] = [0.54, 0.175, 0.86]; free.distance = 0.9; free.azimuth = 25; free.elevation = -38; shot("bowl_small_ball", free)       # the ball in the small bowl (2026-10-26)
free.lookat[:] = [0.54, -0.935, 0.86]; free.distance = 1.0; free.azimuth = -20; free.elevation = -38; shot("bowl_large_ball", free)
free.lookat[:] = [mod.BIN["cx"], mod.BIN["cy"], 0.40]; free.distance = 1.7; free.azimuth = 5; free.elevation = -35; shot("pedal_bin", free)     # the open pedal bin
balls = {n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].round(4).tolist() for n in ("ball_small", "ball_large") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "k_bin_lid"); lid_deg = float(np.degrees(d.qpos[m.jnt_qposadr[lid]])) if lid >= 0 else None
print(f"[check] balls {balls}; pedal bin lid {lid_deg} deg; task {mod.TASK}")
json.dump(dict(robot=robot_uid, nbody=m.nbody, ngeom=m.ngeom, kitchen_bodies=len(kb), target=d.xpos[tb].round(4).tolist(), pelvis=d.qpos[:3].round(4).tolist(), L=mod.L, sink=mod.SINK,
               bin=mod.BIN, balls=balls, lid_deg=lid_deg, task=mod.TASK), open(OUT / "check.json", "w"), indent=1)
print(f"[check] views -> {OUT}")
os._exit(0)
