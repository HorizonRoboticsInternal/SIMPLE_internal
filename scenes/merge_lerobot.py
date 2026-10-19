#!/usr/bin/env python3
"""Merge one-episode SIMPLE LeRobot datasets (Gr00t exporter layout, codebase v2.1) into one dataset.

    python sim/merge_lerobot.py --out data/replay_datasets/bowl_sink_merged data/replay_datasets/bowl_sink/ep_*
Episodes are renumbered in the given order; `index` is made contiguous; videos are renamed; meta totals are recomputed.
A provenance file meta/replay_episodes.jsonl records, per merged episode, the source dir and its replay_meta.json.
"""
import os
import argparse, json, shutil
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("srcs", nargs="+"); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(); out = Path(a.out)
    if out.exists():
        if not a.overwrite: raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    srcs = [Path(s) for s in a.srcs if (Path(s) / "meta/info.json").exists()]
    info0 = json.load(open(srcs[0] / "meta/info.json")); fps = info0["fps"]; chunk_size = info0.get("chunks_size", 1000)
    video_keys = [k for k, f in info0["features"].items() if f.get("dtype") == "video"]
    (out / "meta").mkdir(parents=True); episodes, stats, prov = [], [], []; tasks = {}; n_frames = 0; n_videos = 0
    for new_idx, s in enumerate(srcs):
        info = json.load(open(s / "meta/info.json")); ep_lines = [json.loads(l) for l in open(s / "meta/episodes.jsonl")]
        st_lines = [json.loads(l) for l in open(s / "meta/episodes_stats.jsonl")] if (s / "meta/episodes_stats.jsonl").exists() else []
        for t in (json.loads(l) for l in open(s / "meta/tasks.jsonl")):
            tasks.setdefault(t["task"], len(tasks))
        assert info["total_episodes"] == 1, f"{s}: {info['total_episodes']} episodes (expected 1)"
        old_idx = ep_lines[0]["episode_index"]; chunk = new_idx // chunk_size
        df = pd.read_parquet(s / info["data_path"].format(episode_chunk=old_idx // chunk_size, episode_index=old_idx))
        df["episode_index"] = new_idx; df["index"] = n_frames + df["frame_index"].astype(int)
        df["task_index"] = tasks[ep_lines[0]["tasks"][0]]
        dst = out / info["data_path"].format(episode_chunk=chunk, episode_index=new_idx); dst.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(dst)
        for k in video_keys:
            v = s / info["video_path"].format(episode_chunk=old_idx // chunk_size, video_key=k, episode_index=old_idx)
            if v.exists():
                d = out / info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=new_idx); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(v, d); n_videos += 1
        e = dict(ep_lines[0]); e["episode_index"] = new_idx; episodes.append(e)
        for st in st_lines: st = dict(st); st["episode_index"] = new_idx; stats.append(st)
        meta = json.load(open(s / "replay_meta.json")) if (s / "replay_meta.json").exists() else {}
        prov.append({"episode_index": new_idx, "source": str(s), "source_episode": meta.get("episode"), "seed": meta.get("seed"), "frames": len(df), "scene_env": meta.get("scene_env"), "result": meta.get("result")})
        n_frames += len(df)
    with open(out / "meta/episodes.jsonl", "w") as f:
        for e in episodes: f.write(json.dumps(e) + "\n")
    with open(out / "meta/episodes_stats.jsonl", "w") as f:
        for st in stats: f.write(json.dumps(st) + "\n")
    with open(out / "meta/tasks.jsonl", "w") as f:
        for t, i in sorted(tasks.items(), key=lambda kv: kv[1]): f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    with open(out / "meta/replay_episodes.jsonl", "w") as f:
        for p in prov: f.write(json.dumps(p) + "\n")
    if (srcs[0] / "meta/modality.json").exists(): shutil.copy2(srcs[0] / "meta/modality.json", out / "meta/modality.json")
    info = dict(info0); info.update(total_episodes=len(srcs), total_frames=n_frames, total_tasks=len(tasks), total_videos=n_videos,
                                    total_chunks=(len(srcs) - 1) // chunk_size + 1, splits={"train": f"0:{len(srcs)}"})
    json.dump(info, open(out / "meta/info.json", "w"), indent=2)
    print(f"merged {len(srcs)} episodes, {n_frames} frames, {n_videos} videos -> {out}")


if __name__ == "__main__":
    main()
