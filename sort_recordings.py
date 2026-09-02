#!/usr/bin/env python3
"""Classify + sort real-robot teleop episodes into review folders.

Signals (all from observation.state, which is measured — never the echoed action):
  grasp   : right-hand closure events. A close that later RE-OPENS is a failed
            attempt (operator released to retry); ending closed = success.
  legs    : smoothness of the 15 lower-body joints, scored against episode 112
            of the 2026-08-25 dataset (jerk = std of 2nd difference, and the
            worst per-joint spike).
  box     : whether the box is visible in the opening frames — filled in from
            vision review (box_in_sight.json), not guessed from signals.

Writes analysis to data/sorted/analysis.json; `--apply` then symlinks episodes
(parquet + mp4) into data/sorted/<category>/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = [
    ROOT / "data/real_recordings/2026-08-25-00-36-30-G1-sim",
    ROOT / "data/real_recordings/2026-08-24-21-34-53-G1-MovePick",
    ROOT / "data/real_recordings/2026-08-26-00-17-02-G1-sim",
]
OUT = ROOT / "data/sorted"
REF = ("2026-08-25-00-36-30-G1-sim", 112)   # smoothness reference episode

# Joint layout of the REAL recordings (from meta/info.json "names") is NOT
# simple.robots.g1_sonic.WHOLE_BODY_JOINTS. The real exporter interleaves
# arm/hand per side:
#   0:15 legs+waist | 15:22 left arm | 22:29 LEFT HAND | 29:36 RIGHT ARM | 36:43 RIGHT HAND
# (SIMPLE's own order is left arm, right arm, left hand, right hand -- different.)
# Grasping is done with the RIGHT hand.
HANDS = {"left": slice(22, 29), "right": slice(36, 43)}
ARMS = {"left": slice(15, 22), "right": slice(29, 36)}
GRASP_HAND = "right"
RH = HANDS[GRASP_HAND]
LEGS = slice(0, 15)     # legs + waist
CLOSE_FRAC = 0.45       # closure threshold, fraction of this episode's range
MIN_HOLD = 15           # frames (0.3 s) a closure must persist to count


def grasp_events(state: np.ndarray) -> tuple[int, bool, int | None]:
    """(number of closure attempts, ends_closed, first_closure_frame)."""
    rh = state[:, RH]
    dev = np.abs(rh - np.median(rh[:20], axis=0)).mean(1)   # deviation from open pose
    span = dev.max() - dev.min()
    if span < 0.15:                      # hand never meaningfully moved
        return 0, False, None
    closed = dev > dev.min() + CLOSE_FRAC * span
    # collapse to runs, drop runs shorter than MIN_HOLD
    runs, start = [], None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start >= MIN_HOLD:
                runs.append((start, i))
            start = None
    if start is not None and len(closed) - start >= MIN_HOLD:
        runs.append((start, len(closed)))
    if not runs:
        return 0, False, None
    ends_closed = runs[-1][1] >= len(closed) - 5
    return len(runs), bool(ends_closed), int(runs[0][0])


def leg_metrics(state: np.ndarray) -> dict:
    legs = state[:, LEGS]
    d2 = np.diff(legs, n=2, axis=0)
    return {
        "jerk": float(np.std(d2)),
        "jerk_max_joint": float(np.max(np.std(d2, axis=0))),
        "leg_range": float(np.max(legs.max(0) - legs.min(0))),
    }


def episode_files(ds: Path) -> dict[int, Path]:
    out = {}
    for f in sorted(ds.rglob("data/**/episode_*.parquet")):
        out[int(re.search(r"episode_(\d+)", f.name).group(1))] = f
    return out


def analyse() -> list[dict]:
    recs = []
    for ds in DATASETS:
        if not ds.is_dir():
            print(f"  skip missing {ds}")
            continue
        for ep, f in episode_files(ds).items():
            df = pd.read_parquet(
                f, columns=["observation.state", "teleop.navigate_command", "observation.base_vel"])
            st = np.stack(df["observation.state"].values)
            nav = np.stack(df["teleop.navigate_command"].values)
            # Torso wobble from the IMU gyro. On the real robot base_vel[0:3]
            # (linear) is all zeros — there is no odometry — so only the angular
            # part [3:6] carries information. This tracks perceived lower-body
            # smoothness far better than joint jerk (ep112 sits at the 50th
            # percentile here, vs the 82nd for jerk).
            ang = np.stack(df["observation.base_vel"].values)[:, 3:]
            n_close, ends_closed, first_close = grasp_events(st)
            rec = {
                "dataset": ds.name,
                "episode": ep,
                "frames": len(st),
                "n_close": n_close,
                "ends_closed": ends_closed,
                "first_close": first_close,
                "vx_max": round(float(nav[:, 0].max()), 3),
                "wobble": round(float(np.abs(ang).mean()), 5),
                **{k: round(v, 5) for k, v in leg_metrics(st).items()},
            }
            vids = list(ds.rglob(f"videos/**/episode_{ep:06d}.mp4"))
            rec["video"] = str(vids[0].relative_to(ROOT)) if vids else None
            rec["parquet"] = str(f.relative_to(ROOT))
            recs.append(rec)
    return recs


def classify(recs, box_in_sight, outcome=None, moved=None):
    outcome = outcome or {}
    moved = moved or {}
    ref = next(r for r in recs if r["dataset"] == REF[0] and r["episode"] == REF[1])
    thr = ref["wobble"]   # "as smooth as ep112" = torso wobble at or below its level
    for r in recs:
        key = f"{r['dataset']}/{r['episode']}"
        r["box_visible"] = box_in_sight.get(key)      # True / False / None(unreviewed)
        # Grasp outcome comes from VIDEO review (grasp_outcome.json): whether the
        # box ends up held in the hand or still sitting on the table. Proprioception
        # cannot tell these apart -- the hand closes identically on box or on air.
        # n_close (from hand-closure runs) only supplies the ATTEMPT COUNT.
        vis = outcome.get(key, "unclear")
        if vis == "held":
            r["grasp"] = "success_retry" if r["n_close"] >= 2 else "success_first"
        elif vis == "table":
            r["grasp"] = "failed"
        else:
            r["grasp"] = "unclear"
        r["box_end"] = vis
        r["box_moved"] = moved.get(key)
        r["smooth"] = r["wobble"] <= thr
    return recs


# 06 is a LABELLED category (data/sorted/box_moved.json: key -> true/false).
# It cannot be derived from the recordings: the box's position is not in the data
# (no object pose on the real robot) and every pixel proxy tried failed --
# blob tracking locks onto the robot's own black hands, and frame differencing is
# swamped by camera motion (only 15/159 episodes hold the camera still before the
# grasp, and there the arm entering the table dominates the difference).
CATEGORIES = {
    "01_box_not_in_sight_at_start": lambda r: r["box_visible"] is False,
    "02_grasp_failed_then_success": lambda r: r["grasp"] == "success_retry",
    "03_grasp_failed_all_attempts": lambda r: r["grasp"] == "failed",
    "04_good_and_smooth_legs": lambda r: (
        r["grasp"].startswith("success") and r["box_visible"] is True and r["smooth"]),
    "05_good_but_jerky_legs": lambda r: (
        r["grasp"].startswith("success") and r["box_visible"] is True and not r["smooth"]),
    "06_box_moved_before_grasp": lambda r: r.get("box_moved") is True,
}


def apply(recs: list[dict]):
    for cat in CATEGORIES:
        d = OUT / cat
        d.mkdir(parents=True, exist_ok=True)
        # clear stale links from a previous (possibly differently-labelled) run,
        # otherwise re-sorting leaves episodes in folders they no longer belong to
        for old in d.iterdir():
            if old.is_symlink() or old.is_file():
                old.unlink()
    counts = {c: 0 for c in CATEGORIES}
    for r in recs:
        for cat, pred in CATEGORIES.items():
            try:
                hit = pred(r)
            except Exception:
                hit = False
            if not hit:
                continue
            counts[cat] += 1
            tag = f"{r['dataset'][:10]}_ep{r['episode']:03d}"
            for kind in ("parquet", "video"):
                src = r.get(kind)
                if not src:
                    continue
                dst = OUT / cat / f"{tag}{Path(src).suffix}"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(ROOT / src)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="create the sorted symlink folders")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    bis_path = OUT / "box_in_sight.json"
    box_in_sight = json.loads(bis_path.read_text()) if bis_path.is_file() else {}

    out_path = OUT / "grasp_outcome.json"
    outcome = json.loads(out_path.read_text()) if out_path.is_file() else {}
    mv_path = OUT / "box_moved.json"
    moved = json.loads(mv_path.read_text()) if mv_path.is_file() else {}
    recs = classify(analyse(), box_in_sight, outcome, moved)
    (OUT / "analysis.json").write_text(json.dumps(recs, indent=2))

    from collections import Counter
    print("episodes:", len(recs))
    print("grasp:", dict(Counter(r["grasp"] for r in recs)))
    print("smooth:", dict(Counter(r["smooth"] for r in recs)))
    print("box reviewed:", sum(1 for r in recs if r["box_visible"] is not None), "/", len(recs))
    if a.apply:
        print("sorted:", apply(recs))
