#!/usr/bin/env python
"""Install Isaac re-renders (scripts/render_teleop_isaac.py --eyes left,right) as a teleop dataset's ego-view videos.

    .venv/bin/python scripts/install_isaac_renders.py <dataset dir> <render dir> [--keep <dir>]

For every episode: <render>/episode_XXXXXX.mp4 -> videos/chunk-000/observation.images.ego_view_left/, episode_XXXXXX_right.mp4
-> .../ego_view_right/. The MuJoCo videos they replace are moved to --keep (default <dataset>/../mujoco_videos_<level>, outside
the dataset folder). Every render must exist, have no blank frames (its .json done-marker) and the episode's frame count; the
per-episode look goes to meta/isaac_render.json. Nothing else in the dataset changes.
"""
import argparse, json, shutil
from pathlib import Path
import av

ap = argparse.ArgumentParser(); ap.add_argument("dataset", type=Path); ap.add_argument("render", type=Path); ap.add_argument("--keep", type=Path, default=None); a = ap.parse_args()
info = json.loads((a.dataset / "meta/info.json").read_text()); eps = [json.loads(l) for l in (a.dataset / "meta/episodes.jsonl").read_text().splitlines()]
keep = a.keep or a.dataset.parent / f"mujoco_videos_{a.dataset.name}"
chunks = int(info.get("chunks_size", 1000)); vids = {k.split(".")[-1]: k for k, v in info["features"].items() if v.get("dtype") == "video"}
eyes = {"left": "ego_view_left", "right": "ego_view_right"}
assert all(v in vids for v in eyes.values()), vids


def n_frames(p):
    with av.open(str(p)) as c:
        return sum(1 for _ in c.decode(video=0))


plan, meta = [], {}
for e in eps:
    k = e["episode_index"]; marker = a.render / f"episode_{k:06d}.json"
    assert marker.exists(), f"episode {k}: no done-marker (render missing or had blank frames)"
    line = json.loads(marker.read_text()); assert line.get("blank_frames", 0) == 0 and line.get("frames") == e["length"], (k, line.get("frames"), e["length"])
    for eye, key in eyes.items():
        src = a.render / (f"episode_{k:06d}.mp4" if eye == "left" else f"episode_{k:06d}_right.mp4"); assert src.exists(), src
        nf = n_frames(src); assert nf == e["length"], f"episode {k} {eye}: {nf} frames in the render, {e['length']} in the dataset"
        dst = a.dataset / info["video_path"].format(episode_chunk=k // chunks, video_key=vids[key], episode_index=k)
        plan.append((src, dst, keep / dst.relative_to(a.dataset)))
    meta[str(k)] = {kk: line.get(kk) for kk in ("seed", "light_level", "light_gain", "room", "table_material", "ground_material", "lights", "light_intensity", "model_identical", "robot", "robot_mjcf", "black_hands", "frames")}
for src, dst, bak in plan:
    bak.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): shutil.move(str(dst), str(bak))
    shutil.copy2(src, dst)
(a.dataset / "meta/isaac_render.json").write_text(json.dumps({"renderer": "scripts/render_teleop_isaac.py (Isaac, MuJoCo robot + hbvcam pinhole eyes)", "episodes": meta}, indent=1))
print(f"[install] {len(plan)} videos installed into {a.dataset} ({len(eps)} episodes x {len(eyes)} eyes); MuJoCo originals kept in {keep}; meta/isaac_render.json written")
