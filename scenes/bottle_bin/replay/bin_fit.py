#!/usr/bin/env python
"""Release points of the replayed episodes -> the bin centre that catches all of them (or as many as possible).
   python bin_fit.py --tag scan [--episodes 0 11 ...] [--bin-yaw -90]"""
import argparse, glob, json, re, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); import build_scene as BS
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="scan"); ap.add_argument("--episodes", type=int, nargs="*"); ap.add_argument("--bin-yaw", type=float, default=-90.0)
a = ap.parse_args()
along_x, along_y = (BS.BIN_W, BS.BIN_D) if abs(a.bin_yaw) % 180 == 90 else (BS.BIN_D, BS.BIN_W)
hx, hy = along_x / 2 - BS.BIN_T - BS.BOTTLE_R, along_y / 2 - BS.BIN_T - BS.BOTTLE_R      # where the bottle axis may be at release to clear the walls
pts = {}
for f in sorted(glob.glob(str(HERE / f"trace_ep*_{a.tag}.json"))):
    ep = int(re.match(r"trace_ep(\d+)_", Path(f).name).group(1))
    if a.episodes and ep not in a.episodes: continue
    tr = json.load(open(f)); summ = next((r["summary"] for r in tr if "summary" in r), {}); g = summ.get("grasp", {})
    rel = g.get("release")
    pts[ep] = dict(release=rel, success=g.get("success"), in_bin=g.get("in_bin"), end=summ.get("target_end"), bin=summ.get("bin"), base_end=summ.get("base_end_xy"))
print(f"bin inner opening for the bottle axis: +-{hx:.3f} (x, toward the table) x +-{hy:.3f} (y)")
print(f"{'ep':>3} | grasp | release t    x      y      z   | end xyz | in bin")
P = []
for ep, d in sorted(pts.items()):
    r = d["release"]
    print(f"{ep:3d} | {'ok ' if d['success'] else 'NO '} | {'%5.1f  %+.3f %+.3f %.2f' % (r['t'], r['xy'][0], r['xy'][1], r['z']) if r else 'no release detected        '} | {np.round(d['end'], 2).tolist() if d['end'] else None} | {d['in_bin']}")
    if r and d["success"]: P.append((ep, *r["xy"]))
if not P: sys.exit("no release points")
E = np.array([p[1:] for p in P]); eps = [p[0] for p in P]
lo, hi = E.min(0), E.max(0); span = hi - lo
c_mid = (lo + hi) / 2
print(f"\nrelease points: n={len(P)} x in [{lo[0]:+.3f}, {hi[0]:+.3f}] (span {span[0]:.3f}), y in [{lo[1]:+.3f}, {hi[1]:+.3f}] (span {span[1]:.3f}); mean ({E[:,0].mean():+.3f}, {E[:,1].mean():+.3f})")
print(f"midpoint of the extremes ({c_mid[0]:+.3f}, {c_mid[1]:+.3f}); all inside needs span <= ({2*hx:.3f}, {2*hy:.3f})")
# best centre on a grid: maximise the number of release points inside, then minimise the worst margin
best = None
for cx in np.arange(lo[0] - hx, hi[0] + hx + 1e-9, 0.005):
    for cy in np.arange(lo[1] - hy, hi[1] + hy + 1e-9, 0.005):
        inside = (np.abs(E[:, 0] - cx) <= hx) & (np.abs(E[:, 1] - cy) <= hy)
        margin = np.min(np.minimum(hx - np.abs(E[:, 0] - cx), hy - np.abs(E[:, 1] - cy)))
        key = (inside.sum(), margin)
        if best is None or key > best[0]: best = (key, cx, cy, inside)
(nin, margin), cx, cy, inside = best
print(f"best centre ({cx:+.3f}, {cy:+.3f}): {nin}/{len(P)} release points inside, worst margin {margin:+.3f} m; outside: {[e for e, i in zip(eps, inside) if not i]}")
json.dump({"tag": a.tag, "release_points": {e: [float(x), float(y)] for e, x, y in P}, "best_centre": [float(cx), float(cy)], "inside": int(nin), "n": len(P), "worst_margin": float(margin), "half_opening": [hx, hy]}, open(HERE / f"bin_fit_{a.tag}.json", "w"), indent=1)
