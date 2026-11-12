#!/usr/bin/env python
"""Applies a head-bracket pose from the tuner page (docs/head_mount_tuner) to the rot0 Revo 2 robot (2026-10-26).

    .venv/bin/python scripts/holomotion_v14_checks/apply_head_mount_pose.py --json '<pose json>' | --file pose.json [--render]

The JSON is the page's "Save"/"Copy JSON" object: pos [3] (torso_link frame, m), quat [w,x,y,z], move_module (bool).  It is
written to data/robots/g1_rot0_teleop_revo2/head_mount_pose.json, the mounts are re-placed (add_cameras_mounts.py --redo) and
the final robot file regenerated (add_revo2_touch.py); --render saves head views to docs/head_mount_tuner/applied/.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; RD = ROOT / "data/robots/g1_rot0_teleop_revo2"; PY = str(ROOT / ".venv/bin/python")
ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--file"); ap.add_argument("--render", action="store_true"); a = ap.parse_args()
pose = json.loads(a.json) if a.json else json.loads(Path(a.file).read_text())
keep = {k: pose[k] for k in ("pos", "quat", "move_module", "delta_mm", "delta_deg", "set_at") if k in pose}; keep["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
(RD / "head_mount_pose.json").write_text(json.dumps(keep, indent=1)); print(f"[apply] {RD / 'head_mount_pose.json'}: {keep}")
subprocess.run([PY, str(ROOT / "scripts/robots/add_cameras_mounts.py"), "--redo"], check=True, cwd=ROOT)
subprocess.run([PY, str(ROOT / "scripts/robots/add_revo2_touch.py")], check=True, cwd=ROOT, env={**__import__("os").environ, "MUJOCO_GL": "egl"}, stdout=subprocess.DEVNULL)
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path(str(RD / "g1_rot0_teleop_revo2.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
b = m.body("head_stereo_mount"); print(f"[apply] robot file ok: head_stereo_mount pos {b.pos.round(5).tolist()} quat {b.quat.round(6).tolist()}; left eye {m.camera('hbvcam_left_pinhole').pos.round(5).tolist()}")
if a.render:
    from PIL import Image, ImageDraw
    out = ROOT / "docs/head_mount_tuner/applied"; out.mkdir(parents=True, exist_ok=True)
    r = mujoco.Renderer(m, 600, 800); opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = 1
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; tor = m.body("torso_link").id; cam.lookat[:] = d.xpos[tor] + d.xmat[tor].reshape(3, 3) @ np.array([0.04, 0, 0.42]); cam.distance = 0.4
    for az, el, name in [(215, -25, "front_right_below"), (150, -5, "front_left"), (90, 5, "left_side")]:
        cam.azimuth, cam.elevation = az, el; r.update_scene(d, camera=cam, scene_option=opt); im = Image.fromarray(r.render()); ImageDraw.Draw(im).text((8, 8), f"applied {keep['applied_at']}: {name}", fill=(255, 255, 0)); im.save(out / f"{name}.jpg", quality=86)
    print(f"[apply] views -> {out}")
