#!/usr/bin/env python
"""Applies a wrist-collar pose from the tuner page (docs/wrist_mount_tuner) to the rot0 Revo 2 robot (2026-10-26).

    .venv/bin/python scripts/holomotion_v14_checks/apply_wrist_mount_pose.py --json '<pose json>' | --file pose.json [--render]

The JSON is the page's object: pos [3] (left wrist_yaw frame, m), quat [w,x,y,z], cam_tilt_deg (camera tilt in the housing toward the forearm-axis side, i.e. toward the palm when the housing is on the back of the hand); the same local pose is used on the right
wrist (mirrored frame).  Written to data/robots/g1_rot0_teleop_revo2/wrist_mount_pose.json, then add_cameras_mounts.py --redo
and add_revo2_touch.py regenerate the robot file; --render saves wrist views + the two 150-deg camera views to
docs/wrist_mount_tuner/applied/.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; RD = ROOT / "data/robots/g1_rot0_teleop_revo2"; PY = str(ROOT / ".venv/bin/python")
ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--file"); ap.add_argument("--render", action="store_true"); a = ap.parse_args()
pose = json.loads(a.json) if a.json else json.loads(Path(a.file).read_text())
keep = {k: pose[k] for k in ("pos", "quat", "delta_mm", "delta_deg", "housing_side", "cam_tilt_deg", "set_at") if k in pose}; keep["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
(RD / "wrist_mount_pose.json").write_text(json.dumps(keep, indent=1)); print(f"[apply] {RD / 'wrist_mount_pose.json'}: {keep}")
subprocess.run([PY, str(ROOT / "scripts/robots/add_cameras_mounts.py"), "--redo"], check=True, cwd=ROOT)
subprocess.run([PY, str(ROOT / "scripts/robots/add_revo2_touch.py")], check=True, cwd=ROOT, env={**os.environ, "MUJOCO_GL": "egl"}, stdout=subprocess.DEVNULL)
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(str(RD / "g1_rot0_teleop_revo2.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
for s in ("left", "right"):
    b = m.body(f"{s}_wrist_cam_mount"); print(f"[apply] {s}_wrist_cam_mount pos {b.pos.round(5).tolist()} quat {b.quat.round(6).tolist()}")
if a.render:
    sys.path.insert(0, str(ROOT / "src"))
    from PIL import Image, ImageDraw
    import gymnasium as gym
    from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene
    from simple.teleop.holomotion_v14 import wrist_cams as WC
    mod = load_scene("kitchen"); sc = _load_sonic_config()
    env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, physics_dt=sc["SIMULATE_DT"], headless=True, max_episode_steps=10 ** 9, sonic_config=sc, target=mod.TARGET, dr_level=0, success_criteria=0.9, robot_uid="g1_sonic_revo2")
    senv = env.unwrapped; env.reset(seed=0); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    out = ROOT / "docs/wrist_mount_tuner/applied"; out.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, 600, 800); opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = 1; opt.geomgroup[2] = 1
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    for s, az in (("left", 120), ("right", 240)):
        lw = d.cam_xpos[m.camera(f"wrist_{s}").id]; cam.lookat[:] = lw + [-0.02, 0, 0]; cam.distance = 0.32; cam.azimuth, cam.elevation = az, -25
        r.update_scene(d, camera=cam, scene_option=opt); im = Image.fromarray(r.render()); ImageDraw.Draw(im).text((8, 8), f"applied {keep['applied_at']}: {s} wrist", fill=(255, 255, 0)); im.save(out / f"wrist_{s}.jpg", quality=86)
    w = WC.WristCams(senv, hz=0); v = w.render()
    for s in ("left", "right"): Image.fromarray(v[s]).save(out / f"cam_wrist_{s}.jpg", quality=88)
    print(f"[apply] views -> {out}")
