#!/usr/bin/env python3
"""Write a new psi0 dataset from replay_psi0.py outputs: every column re-recorded, wrist videos added.

    .venv/bin/python scripts/psi0_replay/update_dataset.py \
        --src  <psi0 dataset root> \
        --replays data/psi0_replay/<Task>   (ep_NNNNNN/ folders from replay_psi0.py) \
        --out  data/psi0_replay/<Task>_wrist

Per replayed episode (default: only those whose replay passed the task's own success check):
  * videos   observation.images.egocentric / .wrist_left / .wrist_right  = the replay's renders
  * columns  measured joints (hand/arm/leg joints, states 0:28, and the waist part where the source
             set stores the measured waist) from the replay; command-derived columns (action,
             prev_height, states[31], and the waist part where the source stores the command)
             copied from the recording. Frames the replay held after the recording ended
             (--pad-frames in replay_psi0.py) repeat the last command with walking zeroed.
  * meta     episodes.jsonl keeps environment_config/instruction and adds `source` + `replay`;
             info.json gets the wrist features and a {video_key} video_path; stats are recomputed;
             modality.json lists the wrist views.
The source dataset is never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_KEYS = {"egocentric": "observation.images.egocentric",
              "wrist_left": "observation.images.wrist_left",
              "wrist_right": "observation.images.wrist_right"}
STATE_SLICES = [(29, 32), (34, 36), (32, 34), (36, 43), (15, 22), (22, 29)]  # postprocess_psi0*.py


def stats_block(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    f = lambda x: np.asarray(x, dtype=np.float32).tolist()
    return {"mean": f(arr.mean(0)), "std": f(arr.std(0)), "min": f(arr.min(0)), "max": f(arr.max(0)),
            "q01": f(np.quantile(arr, 0.01, axis=0)), "q99": f(np.quantile(arr, 0.99, axis=0)),
            "count": [int(arr.shape[0])]}


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))


def measured_waist_convention(src_df) -> bool:
    """True when the source stores the MEASURED waist (roll, pitch, yaw) in states[28:31]
    (the SONIC teleop converter); False when it stores the commanded torso (the MP converter)."""
    S = np.stack(src_df["states"].to_numpy())
    L = np.stack(src_df["observation.leg_joints"].to_numpy())
    return float(np.abs(S[:, 28:31] - L[:, [13, 14, 12]]).max()) < 1e-5


def rebuild_rows(src_df, track, n_out: int):
    """psi0 columns for n_out frames: replayed joints + recorded (or held) commands."""
    q = track["joint_qpos"][:n_out].astype(np.float32)                 # (n, 43) G1 SONIC joint order
    A = np.stack(src_df["action"].to_numpy()).astype(np.float32)
    S_rec = np.stack(src_df["states"].to_numpy()).astype(np.float32)
    PR_rec = np.stack(src_df["observation.prev_torso_rpy"].to_numpy()).astype(np.float32)
    n_rec = len(A)
    # actions: recorded, then the held last command with walking zeroed (what the replay sent)
    acts = track["actions_sent"][:n_out].astype(np.float32) if "actions_sent" in track.files else None
    if acts is None:
        acts = np.concatenate([A, np.repeat(A[-1:], max(0, n_out - n_rec), 0)])[:n_out]
        acts[n_rec:, 32:35] = 0.0
    measured = measured_waist_convention(src_df)

    hand_obs = q[:, 29:43]                                              # L thumb, L index, L middle, R ...
    arm_obs = q[:, 15:29]
    leg_obs = q[:, 0:15]
    states = np.zeros((n_out, 32), np.float32)
    states[:, 0:28] = np.concatenate([q[:, s:e] for s, e in STATE_SLICES], axis=1)
    prev_height = np.concatenate([[S_rec[0, 31]], acts[:-1, 31]])[:n_out].astype(np.float32)
    states[:, 31] = prev_height
    if measured:
        states[:, 28:31] = q[:, [13, 14, 12]]
        prev_rpy = np.concatenate([np.zeros((1, 3), np.float32), states[:-1, 28:31]])
    else:
        idx = np.minimum(np.arange(n_out), n_rec - 1)
        states[:, 28:31] = S_rec[idx, 28:31]
        prev_rpy = PR_rec[idx]
    return dict(states=states, action=acts, hand=hand_obs, arm=arm_obs, leg=leg_obs,
                prev_rpy=prev_rpy.astype(np.float32), prev_height=prev_height.reshape(-1, 1),
                measured_waist=measured)


def copy_video(src: Path, dst: Path, n: int):
    """Copy, or re-encode to the first n frames when the render has more."""
    import av

    c = av.open(str(src))
    total = c.streams.video[0].frames
    c.close()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if total == n:
        shutil.copyfile(src, dst)
        return
    assert total > n, f"{src} has {total} frames < {n}"
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wrist_cams import VideoWriter

    w = VideoWriter(dst, fps=50)
    c = av.open(str(src))
    for k, f in enumerate(c.decode(video=0)):
        if k >= n:
            break
        w.write(f.to_ndarray(format="rgb24"))
    w.close()
    c.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--replays", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep-failed", action="store_true", help="also write replays that failed the task check")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = args.src, args.out
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    info = json.loads((src / "meta" / "info.json").read_text())
    src_rows = {int(r["episode_index"]): r for r in load_jsonl(src / "meta" / "episodes.jsonl")}

    eps = []
    for d in sorted(args.replays.glob("ep_*")):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if not s.get("wrist"):
            print(f"skip {d.name}: replayed without wrist cameras")
            continue
        if not s["task_success"] and not args.keep_failed:
            print(f"skip {d.name}: replay failed the task check (max reward {s.get('max_reward')})")
            continue
        eps.append((d, s))
    if not eps:
        raise SystemExit("no usable replays")

    (out / "meta").mkdir(parents=True)
    for name in ("tasks.jsonl", "lang_map.json", "relative_stats.json"):
        if (src / "meta" / name).exists():
            shutil.copyfile(src / "meta" / name, out / "meta" / name)

    chunks = int(info.get("chunks_size", 1000))
    new_rows, ep_stats, all_cols = [], [], {k: [] for k in ("states", "action", "timestamp", "frame_index",
                                                            "episode_index", "index", "task_index", "next.done")}
    index0 = 0
    for new_ep, (d, s) in enumerate(eps):
        src_ep = int(s["episode_index"])
        chunk = src_ep // chunks
        src_df = pq.read_table(src / info["data_path"].format(episode_chunk=chunk, episode_index=src_ep)).to_pandas()
        track = np.load(d / "track.npz", allow_pickle=True)
        n = int(s["frames_replayed"])
        cols = rebuild_rows(src_df, track, n)
        task_index = int(src_df["task_index"].iloc[0])
        fi = np.arange(n, dtype=np.int64)
        table = {
            "states": cols["states"].tolist(), "action": cols["action"].tolist(),
            "observation.hand_joints": cols["hand"].tolist(), "observation.arm_joints": cols["arm"].tolist(),
            "observation.leg_joints": cols["leg"].tolist(), "observation.prev_torso_rpy": cols["prev_rpy"].tolist(),
            "observation.prev_height": cols["prev_height"].tolist(),
            "timestamp": (fi / float(info["fps"])).astype(np.float32), "frame_index": fi,
            "episode_index": np.full(n, new_ep, np.int64), "index": index0 + fi,
            "task_index": np.full(n, task_index, np.int64), "next.done": fi == n - 1,
        }
        schema = pq.read_schema(src / info["data_path"].format(episode_chunk=chunk, episode_index=src_ep))
        pa_tab = pa.table({k: table[k] for k in schema.names}).cast(schema)
        out_chunk = new_ep // chunks
        p = out / info["data_path"].format(episode_chunk=out_chunk, episode_index=new_ep)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa_tab, p)
        for short, key in VIDEO_KEYS.items():
            copy_video(d / f"{short}.mp4",
                       out / "videos" / f"chunk-{out_chunk:03d}" / key / f"episode_{new_ep:06d}.mp4", n)

        row = dict(src_rows[src_ep])
        row.update(episode_index=new_ep, length=n, dataset_from_index=index0, dataset_to_index=index0 + n - 1)
        row["source"] = {"dataset": str(src), "episode_index": src_ep, "frames_recorded": s["frames_recorded"]}
        row["replay"] = {k: s.get(k) for k in ("sim_mode", "task_success", "max_reward", "success_criteria",
                                               "pad_frames_used", "tracking", "wrist", "sim_clock")}
        row["replay"]["waist_columns"] = "measured" if cols["measured_waist"] else "commanded (copied)"
        new_rows.append(row)
        st = {k: stats_block(np.asarray(table[k], dtype=np.float64)) for k in all_cols}
        ep_stats.append({"episode_index": new_ep, "stats": st})
        for k in all_cols:
            all_cols[k].append(np.asarray(table[k], dtype=np.float64).reshape(n, -1))
        index0 += n
        print(f"ep {new_ep} <- source {src_ep}: {n} frames ({s['frames_recorded']} recorded)")

    write_jsonl(out / "meta" / "episodes.jsonl", new_rows)
    write_jsonl(out / "meta" / "episodes_stats.jsonl", ep_stats)
    stats = {k: stats_block(np.concatenate(v)) for k, v in all_cols.items()}
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=1))
    (out / "meta" / "stats_psi0.json").write_text(json.dumps(stats, indent=1))

    w = eps[0][1]["wrist"]
    new_info = dict(info)
    new_info.update(total_episodes=len(new_rows), total_frames=index0, total_videos=len(new_rows) * len(VIDEO_KEYS),
                    total_chunks=(len(new_rows) - 1) // chunks + 1, splits={"train": f"0:{len(new_rows)}"},
                    video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    feats = dict(info["features"])
    vid = dict(feats["observation.images.egocentric"])
    vid["video_info"] = {"video.fps": float(info["fps"]), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "has_audio": False}
    feats["observation.images.egocentric"] = vid
    for key in ("observation.images.wrist_left", "observation.images.wrist_right"):
        feats[key] = {"dtype": "video", "shape": [w["height"], w["width"], 3], "names": ["height", "width", "channel"],
                      "video_info": dict(vid["video_info"])}
    new_info["features"] = feats
    new_info["wrist_cameras"] = w
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    mod_p = src / "meta" / "modality.json"
    if mod_p.exists():
        mod = json.loads(mod_p.read_text())
        mod.setdefault("video", {})
        mod["video"]["wrist_left"] = {"original_key": "observation.images.wrist_left"}
        mod["video"]["wrist_right"] = {"original_key": "observation.images.wrist_right"}
        (out / "meta" / "modality.json").write_text(json.dumps(mod, indent=4))
    print(f"wrote {out}: {len(new_rows)} episodes, {index0} frames")


if __name__ == "__main__":
    main()
