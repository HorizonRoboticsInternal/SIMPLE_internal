#!/usr/bin/env python
"""Bottle replay validity (2026-09-28): ok = grasped by the fingers AND in the bin AND walk complete (<= 0.10 m from the free
walk, walk_gap_bottle) AND the bottle untouched until 1 s before the recorded closure (moves <= 2 cm). Usage: valid_bottle.py TAG
Results keyed by file stem (ep### or ep###_<variant>); prints the union of valid episodes."""
import json, glob, sys, numpy as np
from pathlib import Path
from walk_gap_bottle import gap
HERE = Path(__file__).resolve().parent; G = json.load(open(HERE / "episode_geometry_97.json"))["episodes"]
tag = sys.argv[1]; out = {}
for f in sorted(glob.glob(str(HERE / f"sweep/{tag}/*.json"))):
    r = json.load(open(f)); ep = r["ep"]; g, t = gap(r); tc = G[str(ep)]["closure"]["t"]
    T = r.get("trace") or []; b0 = np.array(r["bottle_start"]); pre = [x for x in T if x["t"] < tc - 1.0]
    mv = max([float(np.linalg.norm(np.array(x["bottle"]) - b0)) for x in pre] or [0.0])
    ok = bool(r.get("in_bin") and r.get("grasp_ok") and g is not None and g <= 0.10 and mv <= 0.02)
    out[Path(f).stem] = {"ep": ep, "variant": r.get("variant"), "ok": ok, "in_bin": r.get("in_bin"), "grasp_ok": r.get("grasp_ok"), "gap": g, "gap_t": t, "pre_moved": round(mv, 3),
               "leg": r.get("robot_bin_leg_ticks"), "job": {k: r.get(k) for k in ("bottle_xy_cmd", "bin_xy_cmd", "bin_yaw_deg")}}
json.dump(out, open(HERE / f"valid_{tag}.json", "w"), indent=1)
V = list(out.values()); eps = sorted({v["ep"] for v in V if v["ok"]})
print(f"{tag}: runs {len(V)}, grasped {sum(bool(v['grasp_ok']) for v in V)}, in bin {sum(bool(v['in_bin']) for v in V)}, walk complete {sum(1 for v in V if v['gap'] is not None and v['gap'] <= 0.10)}, "
      f"untouched before the grasp {sum(v['pre_moved'] <= 0.02 for v in V)}, valid runs {sum(v['ok'] for v in V)}, valid episodes {len(eps)}: {eps}")
