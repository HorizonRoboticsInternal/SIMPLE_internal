"""Page for an annotated episode: the process video, sub tasks, task-related slices, actions, 5 s segments with sampled frames,
a top view of the walk coloured by sub task, and the annotation files (2026-10-27).

    # 1. events from the bit-exact replay, 2. annotation, 3. process video + slice clips, 4. this page
    MUJOCO_GL=egl .venv/bin/python -u scripts/holomotion_v14_checks/episode_events.py <level dir> <ep> docs/_scan/<name>/events.npz
    .venv/bin/python scripts/holomotion_v14_checks/episode_annotate.py docs/_scan/<name>/events.npz <filled level dir> <ep> docs/_scan/<name> --names '{...}'
    .venv/bin/python scripts/holomotion_v14_checks/episode_process_video.py docs/_scan/<name> <head.mp4> <chase.mp4> docs/<name>/process.mp4 \
        --small docs/_scan/<name>/process_small.mp4 --slices docs/<name>/slices
    MUJOCO_GL=egl .venv/bin/python -u scripts/holomotion_v14_checks/episode_annotation_page.py <level dir> <ep> --name <name> --title "..." [--review <page name>] [--check <url>]

Writes docs/<name>/ (NAS: index.html, process.mp4, slices/, caption.json, metadata.json, slices.json, traj.json, topview.jpg) and
docs/_scan/<name>/page.html (artifact, with process_small.mp4 and the same json + topview.jpg). The top view is an orthographic render
of the scene at frame 0 with the robot hidden; traj.json gets its pixel mapping.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def topview(level_dir: Path, ep: int, traj: dict, out: Path):
    import mujoco
    from PIL import Image
    sys.path.insert(0, str(ROOT / "src"))
    from simple.cli.replay_holomotion_v14 import _build_env
    from simple.teleop.holomotion_v14 import exact_log as XL
    log = XL.load(level_dir / f"replay/episode_{ep:06d}.npz"); env, senv, mod = _build_env(log["info"], log["setup"]); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    xy = np.stack([traj["x"], traj["y"]], 1); lo, hi = xy.min(0) - 1.2, xy.max(0) + 1.2
    c = (lo + hi) / 2; span = hi - lo; W = 1000; H = int(round(W * span[1] / span[0]))
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    m.vis.global_.orthographic = 1; m.vis.global_.fovy = float(span[1])                  # orthographic: fovy = view height in metres
    pel = m.body("pelvis").id
    m.geom_rgba[[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == pel], 3] = 0  # hide the robot
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [c[0], c[1], 0.0]; cam.distance = 8.0; cam.elevation = -90; cam.azimuth = 90
    r.update_scene(d, cam); Image.fromarray(r.render()).save(out, quality=82)
    return dict(cx=float(c[0]), cy=float(c[1]), sx=float(span[0]), sy=float(span[1]), W=W, H=H)  # u = (x-cx)/sx*W + W/2, v = H/2 - (y-cy)/sy*H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("level_dir"); ap.add_argument("episode", type=int); ap.add_argument("--name", required=True)
    ap.add_argument("--title", default=""); ap.add_argument("--review", default=""); ap.add_argument("--check", default="")
    a = ap.parse_args()
    scan, nas = ROOT / "docs/_scan" / a.name, ROOT / "docs" / a.name
    traj = json.load(open(scan / "traj.json"))
    if "map" not in traj or not (scan / "topview.jpg").exists():
        traj["map"] = topview(Path(a.level_dir), a.episode, traj, scan / "topview.jpg")
        (scan / "traj.json").write_text(json.dumps(traj))
    C = json.load(open(scan / "caption.json")); meta = json.load(open(scan / "metadata.json"))
    sl_path = nas / "slices/slices.json"
    slices = json.load(open(sl_path)) if sl_path.exists() else json.load(open(scan / "slices.json"))
    facts = dict(title=a.title, review=a.review, check=a.check, episode=a.episode, dataset=C["config"]["source"]["dataset"].replace(str(ROOT) + "/", ""),
                 seconds=round(C["config"]["trim"]["original_frames"] / C["config"]["fps"], 2), meta=meta)
    tpl = (Path(__file__).resolve().parent / "episode_annotation_page.html").read_text()
    for f in ("caption.json", "metadata.json", "traj.json", "topview.jpg"):
        shutil.copy2(scan / f, nas / f)
    (nas / "slices.json").write_text(json.dumps(slices, indent=1)); (scan / "slices_page.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "clip"} for s in slices], indent=1))
    for target, video, sl_file, out in ((scan, "process_small.mp4", "slices_page.json", scan / "page.html"), (nas, "process.mp4", "slices.json", nas / "index.html")):
        page = tpl.replace("__TITLE__", a.title or f"Episode {a.episode} annotation").replace("__FACTS__", json.dumps(facts)).replace("__VIDEO__", video).replace("__SLICES__", sl_file)
        if target is nas:
            page = page.replace("<!--FONTS-->", "")
            page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n' + page + "\n</body></html>\n")
        else:
            page = page.replace("<!--FONTS-->", '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                                '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
        out.write_text(page)
    print(f"[page] {scan / 'page.html'} (artifact), {nas / 'index.html'} (NAS)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
