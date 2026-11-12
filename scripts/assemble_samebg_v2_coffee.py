#!/usr/bin/env python
"""CartCoffeeCup_0919_sim_samebg_v2 = the same-background set's labels/observations + the wall-scene renders' videos.

The v2 renders replay the identical recorded states (forced at every step), so the frames are paired 1:1 with
CartCoffeeCup_0919_sim_samebg. The exported observation columns of the v2 run differ from the samebg set only through the
one free physics step after each forced state (contact-order chaos with the wall model: 8.5 % of frames > 0.05 rad); the
samebg parquet is kept so the three coffee variants (sim / samebg / samebg_v2) stay label-identical.

    .venv/bin/python scripts/assemble_samebg_v2_coffee.py      # after to_psi0 built psi0/CartCoffeeCup_0919_sim_samebg_v2 (74 eps)
"""
import json, shutil
from pathlib import Path

import cv2

P = Path("/home/Horizon/wrk/SIMPLE/data/replay_datasets_phys/psi0")
SRC, DST = P / "CartCoffeeCup_0919_sim_samebg", P / "CartCoffeeCup_0919_sim_samebg_v2"
es, ed = [json.loads(l) for l in open(SRC / "meta/episodes.jsonl")], [json.loads(l) for l in open(DST / "meta/episodes.jsonl")]
assert len(es) == len(ed) == 74, (len(es), len(ed))
for a, b in zip(es, ed):
    assert a["source"]["episode_index"] == b["source"]["episode_index"] and a["length"] == b["length"], (a["source"], b["source"], a["length"], b["length"])
# labels / observations / stats from the same-background set
shutil.rmtree(DST / "data"); shutil.copytree(SRC / "data", DST / "data")
for f in ("episodes_stats.jsonl", "stats.json", "stats_psi0.json", "tasks.jsonl", "modality.json", "lang_map.json", "relative_stats.json"):
    if (SRC / "meta" / f).exists(): shutil.copy2(SRC / "meta" / f, DST / "meta" / f)
rows = []
for a, b in zip(es, ed):
    r = json.loads(json.dumps(a)); ec = r["environment_config"]; ecb = b["environment_config"]
    ec["scene"] = ecb.get("scene", ec.get("scene")); ec["lighting"] = "bright"; ec["seed"] = 1000
    ec["scene_revision"] = "coffee_cart wall 2026-10-15: white wall 1.0 m ahead of the table's far edge (sim/coffee_cart build_scene WALL_FROM_TABLE 1.00; Isaac room shifted back 1.5 m), second ceiling light; the 0.5 m-wall render of 10-14 is kept as _backup_CartCoffeeCup_0919_sim_samebg_v2_wall0p5_20261015"
    ec["labels"] = "identical to CartCoffeeCup_0919_sim_samebg (same recorded states; the free-step observations of this render were not used)"
    rows.append(r)
(DST / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
# videos must match the kept parquet lengths
for k, r in enumerate(rows):
    cap = cv2.VideoCapture(str(DST / f"videos/chunk-000/egocentric/episode_{k:06d}.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    assert n == r["length"] and (w, h) == (640, 360), (k, n, r["length"], w, h)
info = json.load(open(DST / "meta/info.json")); si = json.load(open(SRC / "meta/info.json"))
assert info["total_episodes"] == si["total_episodes"] == 74 and info["total_frames"] == si["total_frames"], (info["total_frames"], si["total_frames"])
print(f"[v2] {DST.name}: {info['total_episodes']} episodes, {info['total_frames']} frames; labels = samebg, videos = wall-scene renders (640x360 @ 50, lengths checked)")
