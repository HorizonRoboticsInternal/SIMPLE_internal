#!/usr/bin/env python
"""Aggregate sweep_replay.py results and fit the scene's placement to them.

  python fit_layout.py replay/sweep/base_3cm                 # one config: counts, release cloud, best bin centre
  python fit_layout.py replay/sweep/grid3cm --by-variant     # a bottle grid: rank the bottle positions
  python fit_layout.py replay/sweep/base_3cm --out fit.json

"Success" of a replayed episode = the recorded arm motion lifted the bottle >= 10 cm without tipping it past 60 deg
(grasp_ok) AND the bottle ended inside the bin footprint below its rim (in_bin).  The bin fit is a 5 mm grid search
for the bin centre that catches the most RELEASE points inside the bin's inner opening -- the same criterion as
replay/bin_fit.py, and the one thing that can be re-evaluated for any bin position from a single sweep.
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(dirs, episodes=None):
    runs = []
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / "ep*.json"))):
            r = json.load(open(f))
            if episodes and r["ep"] not in episodes:
                continue
            r["_dir"] = Path(d).name
            r.pop("trace", None)
            runs.append(r)
    return runs


def key_of(r, by_variant):
    if not by_variant:
        return r["_dir"]
    b = r.get("bottle_xy_cmd") or r["scene"]["layout_bottle_xy"]
    return f"{r['_dir']}|bottle {b[0]:+.3f},{b[1]:+.3f}"


def bin_fit(points, half, step=0.005, pad=0.02):
    """Grid-search the bin centre that has the most release points inside its inner opening."""
    if len(points) == 0:
        return None
    E = np.asarray(points, float)
    lo, hi = E.min(0) - half - pad, E.max(0) + half + pad
    best = None
    for cx in np.arange(lo[0], hi[0] + 1e-9, step):
        for cy in np.arange(lo[1], hi[1] + 1e-9, step):
            inside = (np.abs(E[:, 0] - cx) <= half[0]) & (np.abs(E[:, 1] - cy) <= half[1])
            n = int(inside.sum())
            if n == 0:
                continue
            # tie-break on the worst margin of the points that ARE inside (a deeper-seated cloud is safer)
            marg = float(np.min(np.minimum(half[0] - np.abs(E[inside, 0] - cx), half[1] - np.abs(E[inside, 1] - cy))))
            k = (n, marg)
            if best is None or k > best[0]:
                best = (k, float(cx), float(cy), inside)
    (n, marg), cx, cy, inside = best
    return {"centre": [round(cx, 4), round(cy, 4)], "inside": n, "n_points": len(E), "worst_margin": round(marg, 4),
            "half_opening": [round(float(h), 4) for h in half]}


def stats(v):
    v = np.asarray(v, float)
    if not len(v):
        return None
    return {"n": len(v), "mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            "min": round(float(v.min()), 4), "p25": round(float(np.percentile(v, 25)), 4),
            "median": round(float(np.median(v)), 4), "p75": round(float(np.percentile(v, 75)), 4),
            "max": round(float(v.max()), 4)}


def outcome(r):
    """What happened to the can: it ended in the bin, on the floor, or never left the table."""
    if r.get("in_bin"):
        return "in_bin"
    return "on_table" if r["bottle_end"][2] > 0.60 else "on_floor"


def carried(r):
    """Bin-independent 'the replay picked the can up and walked off with it': it ended below the table top and
    more than 0.5 m from where it started.  Ranking bottle positions by in_bin alone would score them against
    whichever bin position the sweep happened to use."""
    s0, s1 = r["bottle_start"], r["bottle_end"]
    return bool(s1[2] < 0.60 and np.hypot(s1[0] - s0[0], s1[1] - s0[1]) > 0.5)


def summarize(runs, label):
    n = len(runs)
    grasp = [r for r in runs if r["grasp_ok"]]
    inbin = [r for r in runs if r.get("in_bin")]
    succ = [r for r in runs if r.get("success")]
    rel = np.array([r["release"]["xy"] for r in runs if r.get("release") and r["grasp_ok"]], float).reshape(-1, 2)
    half = np.array(runs[0]["half_opening"], float)
    fit = bin_fit(rel, half)
    out = {"label": label, "n": n, "grasp_ok": len(grasp), "in_bin": len(inbin), "success": len(succ),
           "on_table": len([r for r in runs if outcome(r) == "on_table"]),
           "on_floor": len([r for r in runs if outcome(r) == "on_floor"]),
           "carried": len([r for r in runs if carried(r)]),
           "spawn_bad": len([r for r in runs if r.get("spawn_ok") is False]),
           "not_carried_eps": sorted(r["ep"] for r in runs if not carried(r)),
           "in_bin_rate": round(len(inbin) / n, 3), "grasp_rate": round(len(grasp) / n, 3),
           "robot_to_edge": runs[0]["scene"]["robot_to_edge_feet"],
           "bottle_xy": runs[0].get("bottle_xy_cmd") or runs[0]["scene"]["layout_bottle_xy"],
           "bin_xy": runs[0]["bin_xyz"][:2] if runs[0].get("bin_xyz") else None,
           "max_lift": stats([r["max_lift_m"] for r in runs]),
           "release_x": stats(rel[:, 0]) if len(rel) else None, "release_y": stats(rel[:, 1]) if len(rel) else None,
           "bin_fit": fit,
           "never_picked_eps": sorted(r["ep"] for r in runs if outcome(r) == "on_table"),
           "dropped_outside_eps": sorted(r["ep"] for r in runs if outcome(r) == "on_floor"),
           "eps_in_bin": sorted(r["ep"] for r in inbin)}
    if fit:
        # what the success count would be with the fitted bin, judged on the release criterion
        cx, cy = fit["centre"]
        would = [r for r in runs if r["grasp_ok"] and r.get("release")
                 and abs(r["release"]["xy"][0] - cx) <= half[0] and abs(r["release"]["xy"][1] - cy) <= half[1]]
        out["success_with_fitted_bin_est"] = len(would)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--by-variant", action="store_true", help="group by bottle position instead of by directory")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--valid-only", action="store_true", help="drop configs where the can spawns inside the robot")
    ap.add_argument("--rank", default="carried", choices=("carried", "in_bin", "grasp_ok"))
    a = ap.parse_args()
    runs = load(a.dirs, set(a.episodes) if a.episodes else None)
    if not runs:
        raise SystemExit("no results found")
    groups = defaultdict(list)
    for r in runs:
        groups[key_of(r, a.by_variant)].append(r)
    res = [summarize(v, k) for k, v in sorted(groups.items())]
    if a.valid_only:
        res = [s for s in res if not s.get("spawn_bad")]
    res.sort(key=lambda s: (-s[a.rank], -s["in_bin"], -s["carried"]))
    print(f"{'config':40s} {'n':>3} {'CARRIED':>8} {'in bin':>7} {'table':>6} {'floor':>6} {'lift>10':>7} {'fit bin centre':>20} {'catch':>7}")
    for s in res[: a.top]:
        f = s["bin_fit"] or {}
        c = f.get("centre")
        bad = " CAN SPAWNED IN THE ROBOT" if s.get("spawn_bad") else ""
        print(f"{s['label'][:40]:40s} {s['n']:3d} {s['carried']:8d} {s['in_bin']:7d} {s['on_table']:6d} {s['on_floor']:6d} {s['grasp_ok']:7d} "
              f"{('(%+.3f, %+.3f)' % (c[0], c[1])) if c else '-':>20} {f.get('inside', 0):3d}/{f.get('n_points', 0):<3d}{bad}")
    if len(res) > a.top:
        print(f"... {len(res) - a.top} more configs")
    best = res[0]
    print("\nbest:", json.dumps({k: best[k] for k in ("label", "n", "grasp_ok", "carried", "in_bin", "on_table", "on_floor", "bottle_xy",
                                                      "bin_xy", "bin_fit", "success_with_fitted_bin_est")}, indent=1))
    if a.out:
        res = res
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
