#!/usr/bin/env python
"""The recorded Dex3 grasp, for comparison with replay_rot0_revo2.py: per episode, the recorded physics state 0.5 s after the
first right grip close is loaded into the recorded (Dex3) scene and the right hand is rendered from the same three close-up
views the Revo 2 replay uses; the bottle centre in the right wrist_yaw frame and the Dex3 finger angles are written to
<out>/dex3_grasp.json. The Revo 2 replay's grasp still is the same moment on the Revo 2 robot.
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/dex3_grasp_stills.py <rot0 dataset dir> --episodes 0,1,... --out docs/rot0_revo2
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, pyarrow.parquet as pq
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--episodes", default=",".join(str(i) for i in range(50))); ap.add_argument("--out", default="docs/rot0_revo2"); ap.add_argument("--after-s", type=float, default=0.5)
a = ap.parse_args(); SRC = Path(a.dataset).resolve(); OUT = (ROOT / a.out).resolve(); (OUT / "img").mkdir(parents=True, exist_ok=True)
from simple.cli.replay_holomotion_v14 import _build_env, _reset_to
from simple.teleop.holomotion_v14 import exact_log as XL
from simple.robots.g1_sonic import RIGHT_HAND_JOINTS
VIEWS = [(150, -35, 0.45), (90, -80, 0.5), (200, -10, 0.45)]        # azimuth, elevation, distance: the Revo 2 replay's hand_still views


def hand_views(m, d, rend, label):
    w = m.body("right_wrist_yaw_link").id; look = d.xpos[w] + d.xmat[w].reshape(3, 3) @ np.array([0.09, 0.0, 0.0])
    ims = []
    for az, el, dist in VIEWS:
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = look; cam.distance, cam.azimuth, cam.elevation = dist, az, el
        rend.update_scene(d, camera=cam); ims.append(rend.render().copy())
    im = Image.fromarray(np.concatenate(ims, axis=1)); ImageDraw.Draw(im).text((8, 8), label, fill=(255, 255, 0)); return im


env = None; out = {}
for ep in [int(x) for x in a.episodes.split(",")]:
    log = XL.load(SRC / "replay" / f"episode_{ep:06d}.npz"); info, setup = dict(log["info"]), log["setup"]
    t = pq.read_table(str(SRC / "data/chunk-000" / f"episode_{ep:06d}.parquet")); pico = np.stack(t["holomotion.pico"].to_pandas().values); n = t.num_rows
    close = next((i for i in range(n) if pico[i][5] > 0.5), None)
    if close is None: out[ep] = dict(grip_close_frame=None); continue
    f = min(close + int(round(a.after_s * info["render_hz"])), n - 1)
    if env is None:
        if info.get("policy_models_dir"): os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = info["policy_models_dir"]
        env, senv, mod = _build_env(info, setup, check_legacy=False); m, d = senv.mujoco.mjModel, senv.mujoco.mjData; rend = mujoco.Renderer(m, 360, 480)
    else:
        _reset_to(env, senv, mod, info, setup); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    XL.set_state(m, d, log["physics"][f], XL.PHYSICS); mujoco.mj_forward(m, d)
    bid = m.body(str(senv.task.target.asset.label)).id; w = m.body("right_wrist_yaw_link").id; R = d.xmat[w].reshape(3, 3)
    hand_views(m, d, rend, f"Dex3 as recorded, ep {ep}, {f / info['render_hz']:.1f} s ({a.after_s:.1f} s after grip close)").save(OUT / "img" / f"ep{ep}_dex3.jpg", quality=88)
    fq = [round(float(d.qpos[m.jnt_qposadr[m.joint(j).id]]), 3) for j in RIGHT_HAND_JOINTS]
    out[ep] = dict(grip_close_frame=int(close), still_frame=int(f), bottle_in_wrist=(R.T @ (d.xpos[bid] - d.xpos[w])).round(3).tolist(), dex3_right_q=fq, bottle_z=round(float(d.xpos[bid][2]), 3), still=f"img/ep{ep}_dex3.jpg")
    print(f"[dex3] ep {ep}: grip close {close / info['render_hz']:.2f} s, bottle in wrist {out[ep]['bottle_in_wrist']}", flush=True)
    json.dump(out, open(OUT / "dex3_grasp.json", "w"), indent=1)
print("[dex3] done", flush=True); sys.stdout.flush(); os._exit(0)
