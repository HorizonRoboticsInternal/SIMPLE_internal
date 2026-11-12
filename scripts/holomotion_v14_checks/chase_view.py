"""Third-person chase view of a recorded episode, rendered during its bit-exact replay (2026-10-27).

    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/chase_view.py <level dir> <episode> <out.mp4> [--mode clear|fixed]

--mode clear (default, user 2026-10-27: "the third person view is sometimes blocked by the wall during movement"): the camera
follows the robot's heading from behind-right, and every frame checks sight lines from the camera to the robot's pelvis,
head and both hands (mj_ray against the drawn geometry; the robot hiding part of itself is fine). When the current view
loses any of them it cuts to the clearest of a set of candidate views around the robot (other sides, higher, closer),
preferring the one nearest the current angle; cuts rather than sweeps, so the camera never travels through a wall, and at
most one cut per 0.6 s unless the view is blocked. --mode fixed: the old camera (2.6 m behind-right, 24 deg down).
Writes <out.mp4> (640x360, one frame per recorded frame) and <out>.json (per-frame visibility stats).
"""
import argparse, json, sys
import numpy as np, mujoco, imageio.v2 as iio
sys.path.insert(0, "/home/Horizon/wrk/SIMPLE/src")
from pathlib import Path
from simple.cli.replay_holomotion_v14 import _build_env
from simple.teleop.holomotion_v14 import exact_log as XL

# candidate views: (azimuth offset from the robot's heading, elevation, distance); the first is the default chase view
CANDS = [(-25, -24, 2.6), (25, -24, 2.6), (-25, -42, 2.4), (25, -42, 2.4), (-65, -30, 2.3), (65, -30, 2.3), (-110, -35, 2.2), (110, -35, 2.2),
         (180, -38, 2.2), (-25, -60, 2.2), (25, -60, 2.2), (-25, -42, 1.6), (25, -42, 1.6), (0, -80, 2.4)]
VIS_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)                       # what the renderer draws by default


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("level_dir"); ap.add_argument("episode", type=int); ap.add_argument("out")
    ap.add_argument("--mode", default="clear", choices=["clear", "fixed"]); ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    level_dir, ep = Path(a.level_dir), a.episode
    log = XL.load(level_dir / f"replay/episode_{ep:06d}.npz"); info, setup = log["info"], log["setup"]
    env, senv, mod = _build_env(info, setup); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    print("model identical:", XL.model_sha256(m) == info["model_sha256"], flush=True)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640); m.vis.global_.offheight = max(int(m.vis.global_.offheight), 360)
    r = None if a.no_video else mujoco.Renderer(m, 360, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    pel = m.body("pelvis").id
    robot = set()
    for b in range(m.nbody):
        k = b
        while k > 0:
            if k == pel: robot.add(b); break
            k = m.body_parentid[k]
    keys = [("pelvis", np.array([0, 0, 0.15])), ("torso_link", np.array([0, 0, 0.42])), ("left_wrist_yaw_link", np.zeros(3)), ("right_wrist_yaw_link", np.zeros(3))]
    keys = [(m.body(n).id, off) for n, off in keys if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
    w = None if a.no_video else iio.get_writer(a.out, fps=50, codec="libx264", quality=None, ffmpeg_params=["-crf", "20"], macro_block_size=1)
    gid = np.zeros(1, np.int32)
    st = dict(head=None, look=None, cand=0, last_cut=-999, cuts=0)
    stats = dict(frames=0, all_visible=0, pelvis_hidden=0, head_hidden=0, any_hidden=0)

    def eye_of(look, az, el, dist):
        az, el = np.radians(az), np.radians(el)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return look - dist * fwd

    def visible(eye):
        out = []
        for b, off in keys:
            p = d.xpos[b] + off; v = p - eye; L = float(np.linalg.norm(v))
            hit = mujoco.mj_ray(m, d, eye, v / L, VIS_GROUPS, 1, -1, gid)
            out.append(hit < 0 or hit >= L - 0.06 or int(m.geom_bodyid[gid[0]]) in robot)
        return out

    def on_frame(t):
        q = d.xquat[pel]; yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        if st["head"] is None: st["head"] = yaw
        st["head"] += 0.04 * ((yaw - st["head"] + 180) % 360 - 180)                     # heading followed over ~0.5 s
        look = d.xpos[pel] + np.array([0, 0, 0.15]); st["look"] = look if st["look"] is None else st["look"] + 0.15 * (look - st["look"])
        def view(i):
            dz, el, dist = CANDS[i]; return st["head"] + dz, el, dist
        if a.mode == "fixed":
            az, el, dist = view(0); vis = visible(eye_of(st["look"], az, el, dist))
        else:
            az, el, dist = view(st["cand"]); vis = visible(eye_of(st["look"], az, el, dist))
            may_cut = (t - st["last_cut"]) * 0.02 >= 0.6 or sum(vis) <= 2
            if not all(vis) and may_cut:
                cur_az = az; best = None
                for i in range(len(CANDS)):
                    azi, eli, di = view(i); vi = visible(eye_of(st["look"], azi, eli, di))
                    score = (sum(vi), vi[0] + vi[1], -abs((azi - cur_az + 180) % 360 - 180) / 180.0 - 0.02 * i)
                    if best is None or score > best[0]: best = (score, i, vi)
                if best[1] != st["cand"] and best[0][0] > sum(vis):
                    st["cand"], st["last_cut"], st["cuts"] = best[1], t, st["cuts"] + 1
                    az, el, dist = view(best[1]); vis = best[2]
            elif all(vis) and st["cand"] != 0 and (t - st["last_cut"]) * 0.02 >= 2.0:     # drift back to the default view when it is clear again
                azd, eld, dd = view(0)
                if all(visible(eye_of(st["look"], azd, eld, dd))):
                    st["cand"], st["last_cut"], st["cuts"] = 0, t, st["cuts"] + 1; az, el, dist = azd, eld, dd; vis = [True] * len(keys)
        stats["frames"] += 1; stats["all_visible"] += all(vis); stats["any_hidden"] += (not all(vis))
        stats["pelvis_hidden"] += (not vis[0]); stats["head_hidden"] += (not vis[1])
        if w is not None:
            cam.lookat[:] = st["look"]; cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(d, cam); w.append_data(r.render())

    res = XL.replay_physics(m, d, log, on_frame=on_frame)
    if w is not None: w.close()
    stats.update(cuts=st["cuts"], mode=a.mode, bit_exact=bool(res["exact"]))
    Path(str(a.out) + ".json").write_text(json.dumps(stats, indent=1))
    n = max(1, stats["frames"])
    print(f"replay {'BIT-EXACT' if res['exact'] else res}; mode {a.mode}: all four key points visible on {100 * stats['all_visible'] / n:.1f}% of frames, "
          f"pelvis hidden {100 * stats['pelvis_hidden'] / n:.1f}%, head hidden {100 * stats['head_hidden'] / n:.1f}%, cuts {st['cuts']}", flush=True)
    import os; os._exit(0)


if __name__ == "__main__":
    main()
