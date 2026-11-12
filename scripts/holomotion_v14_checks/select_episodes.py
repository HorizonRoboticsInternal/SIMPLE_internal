#!/usr/bin/env python
"""Copy chosen episodes of recorded teleop datasets into a new LeRobot dataset (2026-11-10, the final A2 dataset: "find 30 ep
which the plate is most flat and save them into a new final A2 dataset").

    .venv/bin/python scripts/holomotion_v14_checks/select_episodes.py --selection <json> --out data/<new save dir>

The selection JSON: {"rule": ..., "episodes": [{"dataset": "<save dir under data/>", "episode": k, ...}, ...]} in the new order.
Per episode: the parquet with episode_index / index renumbered (frame_index, task_index kept: one task), the exact log
(replay/episode_*.npz, byte-identical: the replay checks physics, not the index), its meta/episodes.jsonl line (+ "source" and the
selection's own fields) and its episodes_stats line. Columns: the union of the sources' features; a column a source episode
lacks is added the way the recorder writes a missing sample -- NaN (holomotion.pico_hands: the controller-grip recordings have no
headset hand tracking). meta/info.json from the source with the most features, counts updated; the other meta files are copied
(they must be identical across the sources). The source datasets are only read. The selection is also written next to the
LeRobot root (<out>/selection.json; nothing extra inside the dataset folder).
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--selection", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--env", default="G1WholebodyKitchenSinkTeleop-v0"); ap.add_argument("--level", default="level-3_pinhole")
a = ap.parse_args()
SEL = json.load(open(a.selection)); eps = SEL["episodes"]
OUT_SAVE = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
OUT = OUT_SAVE / "simple" / a.env / a.level
if OUT_SAVE.exists():
    sys.exit(f"{OUT_SAVE} exists: pick a new --out (nothing is overwritten)")


def src_root(ds):
    return ROOT / "data" / ds / "simple" / a.env / a.level


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


SRC = sorted({e["dataset"] for e in eps})
infos = {ds: json.load(open(src_root(ds) / "meta/info.json")) for ds in SRC}
base_ds = max(SRC, key=lambda ds: len(infos[ds]["features"]))
features = dict(infos[base_ds]["features"])
for ds in SRC:                                                               # the union (a feature spec that differs between sources is an error)
    for k, v in infos[ds]["features"].items():
        if k in features and features[k] != v:
            sys.exit(f"feature {k} differs between {base_ds} and {ds}")
        features.setdefault(k, v)
side_files = ["holomotion_raw.json", "lean_record.json", "modality.json", "realformat.json", "tasks.jsonl"]
for f in side_files:
    texts = {ds: (src_root(ds) / "meta" / f).read_text() for ds in SRC if (src_root(ds) / "meta" / f).exists()}
    if len(set(texts.values())) > 1:
        sys.exit(f"meta/{f} differs between the sources {sorted(texts)}: merge it by hand")
(OUT / "meta").mkdir(parents=True); (OUT / "data/chunk-000").mkdir(parents=True); (OUT / "replay").mkdir(parents=True)
for f in side_files:
    if (src_root(base_ds) / "meta" / f).exists():
        shutil.copy2(src_root(base_ds) / "meta" / f, OUT / "meta" / f)
# the parquet schema (arrow types + the stored huggingface feature metadata) of the source with the most features: every
# episode is written with it (pandas would turn the fixed-size array columns into variable lists, which the loader rejects)
first_of = {ds: next(int(e["episode"]) for e in eps if e["dataset"] == ds) for ds in SRC}
SCHEMA = pq.read_schema(src_root(base_ds) / f"data/chunk-000/episode_{first_of[base_ds]:06d}.parquet")
rows = {ds: {r["episode_index"]: r for r in jsonl(src_root(ds) / "meta/episodes.jsonl")} for ds in SRC}
stats = {ds: {r["episode_index"]: r for r in jsonl(src_root(ds) / "meta/episodes_stats.jsonl")} for ds in SRC}
out_rows, out_stats, total = [], [], 0
for i, e in enumerate(eps):
    ds, k = e["dataset"], int(e["episode"]); S = src_root(ds)
    t = pq.read_table(S / f"data/chunk-000/episode_{k:06d}.parquet"); n = t.num_rows
    st = json.loads(json.dumps(stats[ds][k]))
    cols = []
    for fld in SCHEMA:
        if fld.name == "episode_index":
            cols.append(pa.array(np.full(n, i, dtype=np.int64), type=fld.type))
        elif fld.name == "index":
            cols.append(pa.array(np.arange(total, total + n, dtype=np.int64), type=fld.type))
        elif fld.name in t.column_names:
            cols.append(t.column(fld.name).cast(fld.type))
        else:                                                               # a column this source episode does not have: NaN, as the recorder writes a missing sample
            shape = int(np.prod(features[fld.name]["shape"]))
            flat = pa.array(np.full(n * shape, np.nan, dtype=features[fld.name]["dtype"]))
            cols.append(pa.FixedSizeListArray.from_arrays(flat, shape).cast(fld.type))
            st["stats"][fld.name] = dict(min=[float("nan")] * shape, max=[float("nan")] * shape, mean=[float("nan")] * shape, std=[float("nan")] * shape, count=[n])
    extra = [c for c in t.column_names if c not in SCHEMA.names]
    if extra:
        sys.exit(f"{ds} ep {k}: columns {extra} are not in the schema of {base_ds}")
    pq.write_table(pa.Table.from_arrays(cols, schema=SCHEMA), OUT / f"data/chunk-000/episode_{i:06d}.parquet")
    for f in S.glob(f"replay/episode_{k:06d}.*"):
        shutil.copy2(f, OUT / "replay" / f"episode_{i:06d}{f.suffix}")
    r = dict(rows[ds][k]); r["episode_index"] = i
    r["source"] = dict(dataset=ds, episode=k)
    r["selection"] = {kk: vv for kk, vv in e.items() if kk not in ("dataset", "episode")}
    out_rows.append(r); st["episode_index"] = i; out_stats.append(st)
    if "index" in st["stats"]:                                              # the global frame index moved with the renumbering
        st["stats"]["index"] = dict(min=[total], max=[total + n - 1], mean=[total + (n - 1) / 2], std=[float(np.std(np.arange(n)))], count=[n])
    if "episode_index" in st["stats"]:
        st["stats"]["episode_index"] = dict(min=[i], max=[i], mean=[float(i)], std=[0.0], count=[n])
    total += n
    print(f"[select] {i:2d} <- {ds} ep {k} ({n} frames)", flush=True)
info = json.loads(json.dumps(infos[base_ds])); info["features"] = features
info.update(total_episodes=len(eps), total_frames=total, total_chunks=1, total_videos=0, splits={"train": f"0:{len(eps)}"}, discarded_episode_indices=[])
(OUT / "meta/info.json").write_text(json.dumps(info, indent=4))
(OUT / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
(OUT / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_stats))
(OUT_SAVE / "selection.json").write_text(json.dumps(SEL, indent=1))
print(f"[select] {len(eps)} episodes, {total} frames -> {OUT}")
