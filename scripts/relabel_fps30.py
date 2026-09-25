#!/usr/bin/env python
"""Relabel real-robot teleop datasets from the wrong 50 fps to their true 30 Hz row rate.

Why: the real-rig exporter writes one row per camera frame (blocking ZMQ recv on the 30 fps
D455 stream) but was launched with --data-collection-frequency 50, so every dataset says fps=50
while rows are ~33 ms apart. Content is untouched: only fps, the timestamp column (frame_index/30),
the timestamp statistics and the mp4 container frame rate change (no re-encode).

Datasets: raw sessions data/real_recordings/*-G1-* (with episodes), data/real_recordings/psi0/*,
and the four LMDB exports on nas28 (timestamp arrays + meta_data.fps).
Backups: data/real_recordings/_backup_fps50/<name>/ and /mnt/nas28/alan.jiang/_backup_fps50/<db>/.
Idempotent: datasets already at fps=30 are skipped. --check only verifies.
"""
import argparse, glob, json, os, pickle, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/Horizon/wrk/SIMPLE"); RR = ROOT / "data/real_recordings"; BK = RR / "_backup_fps50"
NEW_FPS = 30; OLD_FPS = 50; SCALE = OLD_FPS / NEW_FPS
FFMPEG = subprocess.run([sys.executable, "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"], capture_output=True, text=True).stdout.strip()
LMDBS = sorted(glob.glob("/mnt/nas28/alan.jiang/psi_g1_*_lmdb")); LMDB_BK = Path("/mnt/nas28/alan.jiang/_backup_fps50")


def video_rate_frames(p):
    import av
    c = av.open(str(p)); s = c.streams.video[0]; n = sum(1 for _ in c.decode(video=0)); r = float(s.average_rate); c.close(); return r, n


def remux_30(src: Path):
    """Frame-rate change by rewriting only the packet time base (x 5/3): identical packets, no
    re-encode, exact 30 fps. (A raw-H.264 round trip dropped frames on B-frame streams.)"""
    import av
    from fractions import Fraction
    r0, n0 = video_rate_frames(src)
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "v.mp4"
        inp = av.open(str(src)); ist = inp.streams.video[0]
        out = av.open(str(out_p), "w"); ost = out.add_stream_from_template(ist) if hasattr(out, "add_stream_from_template") else out.add_stream(template=ist)
        tb_new = ist.time_base * Fraction(OLD_FPS, NEW_FPS)          # same integer pts/dts, 5/3 longer ticks
        ost.time_base = tb_new
        for pkt in inp.demux(ist):
            if pkt.dts is None: continue
            pkt.stream = ost; pkt.time_base = tb_new
            out.mux(pkt)
        out.close(); inp.close()
        r, n = video_rate_frames(out_p)
        assert n == n0, f"frame count changed {n0}->{n} for {src}"
        assert abs(r - NEW_FPS) < 1e-3, f"rate {r} != {NEW_FPS} for {src}"
        shutil.copy2(out_p, src)


def scale_stats(d: dict):
    for k, v in d.items():
        if k == "count": continue
        d[k] = (np.asarray(v, dtype=float) * SCALE).tolist()
    return d


def relabel_lerobot(root: Path, check_only=False):
    info_p = root / "meta/info.json"; info = json.load(open(info_p))
    if info.get("total_episodes", 0) == 0:
        return "empty"
    if int(info["fps"]) == NEW_FPS:
        return "already 30"
    assert int(info["fps"]) == OLD_FPS, f"{root}: unexpected fps {info['fps']}"
    if check_only:
        return "would relabel"
    name = root.name if root.parent.name != "psi0" else "psi0__" + root.name
    bk = BK / name
    if not bk.exists():
        bk.mkdir(parents=True)
        for sub in ("meta", "data", "videos"):
            if (root / sub).exists(): shutil.copytree(root / sub, bk / sub)
    try:
        # 1. parquet timestamps
        n_rows = 0
        for pq in sorted(root.glob("data/chunk-*/episode_*.parquet")):
            df = pd.read_parquet(pq); dt = df["timestamp"].dtype
            df["timestamp"] = (df["frame_index"].to_numpy().astype(np.float64) / NEW_FPS).astype(dt)
            df.to_parquet(pq); n_rows += len(df)
        # 2. per-episode timestamp stats (recomputed from the new column)
        es_p = root / "meta/episodes_stats.jsonl"
        if es_p.exists():
            lines = [json.loads(l) for l in open(es_p) if l.strip()]
            by_ep = {}
            for pq in root.glob("data/chunk-*/episode_*.parquet"):
                df = pd.read_parquet(pq, columns=["episode_index", "timestamp"]); by_ep[int(df["episode_index"].iloc[0])] = df["timestamp"].to_numpy(dtype=float)
            for row in lines:
                ts = by_ep.get(int(row["episode_index"])); st = row["stats"].get("timestamp")
                if ts is None or st is None: continue
                new = {"min": [float(ts.min())], "max": [float(ts.max())], "mean": [float(ts.mean())], "std": [float(ts.std())], "count": st.get("count", [len(ts)])}
                if "q01" in st: new["q01"] = [float(np.quantile(ts, 0.01))]
                if "q99" in st: new["q99"] = [float(np.quantile(ts, 0.99))]
                row["stats"]["timestamp"] = {k: new[k] for k in st.keys() if k in new}
            open(es_p, "w").write("".join(json.dumps(r) + "\n" for r in lines))
        # 3. global stats (timestamps scale linearly)
        for f in ("stats.json", "stats_psi0.json"):
            p = root / "meta" / f
            if p.exists():
                s = json.load(open(p))
                if "timestamp" in s: s["timestamp"] = scale_stats(s["timestamp"])
                json.dump(s, open(p, "w"), indent=4)
        # 4. videos
        n_vid = 0
        for v in sorted(root.glob("videos/chunk-*/*/episode_*.mp4")):
            remux_30(v); n_vid += 1
        # 5. info
        info["fps"] = NEW_FPS
        json.dump(info, open(info_p, "w"), indent=4)
        # 6. verify: reload with lerobot (timestamp sync check runs inside)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id="relabel/" + name, root=str(root), video_backend="pyav")
        assert ds.meta.fps == NEW_FPS and ds.num_episodes == info["total_episodes"]
        _ = ds[0]
        return f"relabeled: {n_rows} rows, {n_vid} videos"
    except Exception:
        for sub in ("meta", "data", "videos"):
            if (bk / sub).exists():
                shutil.rmtree(root / sub, ignore_errors=True); shutil.copytree(bk / sub, root / sub)
        raise


def relabel_lmdb(db: str, check_only=False):
    import lmdb
    env = lmdb.open(db + "/meta", readonly=True, lock=False)
    with env.begin() as txn:
        keys = [k for k, _ in txn.cursor()]; eps = sorted(set(k.decode().split("/")[0] for k in keys))
        md = pickle.loads(txn.get(f"{eps[0]}/meta_data".encode()))
    map_size = env.info()["map_size"]; env.close()
    if md.get("fps") == NEW_FPS: return "already 30"
    assert md.get("fps") == OLD_FPS, f"{db}: fps {md.get('fps')}"
    if check_only: return f"would relabel {len(eps)} episodes"
    bk = LMDB_BK / Path(db).name
    if not bk.exists():
        bk.mkdir(parents=True); shutil.copytree(db + "/meta", bk / "meta")
    env = lmdb.open(db + "/meta", map_size=max(map_size, 64 << 20), lock=True)
    with env.begin(write=True) as txn:
        for e in eps:
            fi = pickle.loads(txn.get(f"{e}/frame_index".encode())); ts_old = pickle.loads(txn.get(f"{e}/timestamp".encode()))
            ts = (np.asarray(fi, dtype=np.float64) / NEW_FPS).astype(ts_old.dtype)
            txn.put(f"{e}/timestamp".encode(), pickle.dumps(ts, protocol=4))
            m = pickle.loads(txn.get(f"{e}/meta_data".encode())); m["fps"] = NEW_FPS
            txn.put(f"{e}/meta_data".encode(), pickle.dumps(m, protocol=4))
    env.sync(); env.close()
    env = lmdb.open(db + "/meta", readonly=True, lock=False)
    with env.begin() as txn:
        m = pickle.loads(txn.get(f"{eps[-1]}/meta_data".encode())); ts = pickle.loads(txn.get(f"{eps[-1]}/timestamp".encode()))
    env.close(); assert m["fps"] == NEW_FPS and abs(ts[1] - 1 / NEW_FPS) < 1e-9
    return f"relabeled {len(eps)} episodes"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--skip-lmdb", action="store_true")
    a = ap.parse_args()
    roots = [Path(p).parent.parent for p in sorted(glob.glob(str(RR / "*-G1-*/meta/info.json")))] + [Path(p).parent.parent for p in sorted(glob.glob(str(RR / "psi0/*/meta/info.json")))]
    if a.only: roots = [r for r in roots if r.name in a.only]
    report = {}
    for r in roots:
        try: report[str(r.relative_to(RR))] = relabel_lerobot(r, a.check)
        except Exception as e: report[str(r.relative_to(RR))] = f"FAILED (restored): {e}"
        print(f"  {r.relative_to(RR)}: {report[str(r.relative_to(RR))]}", flush=True)
    if not a.skip_lmdb:
        for db in LMDBS:
            try: report[db] = relabel_lmdb(db, a.check)
            except Exception as e: report[db] = f"FAILED: {e}"
            print(f"  {db}: {report[db]}", flush=True)
    json.dump(report, open(BK / ("check_report.json" if a.check else "relabel_report.json"), "w"), indent=1) if BK.exists() or not a.check else None
    return report


if __name__ == "__main__":
    BK.mkdir(parents=True, exist_ok=True)
    main()
