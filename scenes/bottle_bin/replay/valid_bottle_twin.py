#!/usr/bin/env python
"""Bottle validity against the CARRY TWIN (2026-09-28): the no-furniture free walk carries no 0.5 kg bottle, and a 15 s one-hand
carry drifts the gait 5-14 cm on its own (gradual from the grasp, no contact, the same for every bin position). So the walk check
compares a run with its twin = the same job with the bin 21 m away (the robot grasps and carries the bottle the same way):
ok = grasped AND in the bin AND bottle untouched before the grasp AND base path within 0.10 m of the twin AND legs on the
table <= 25 ticks AND legs on the bin <= 25 ticks. Usage: valid_bottle_twin.py TAG TWINTAG"""
import json, glob, sys, numpy as np
from pathlib import Path
from walk_gap_bottle import gap as gap_free
HERE = Path(__file__).resolve().parent; G = json.load(open(HERE / "episode_geometry_97.json"))["episodes"]
tag, twin = sys.argv[1], sys.argv[2]; out = {}
def path(r): return {round(p[0], 1): np.array(p[1:3]) for p in r.get("base_path") or []}
for f in sorted(glob.glob(str(HERE / f"sweep/{tag}/*.json"))):
    r = json.load(open(f)); ep = r["ep"]; tf = HERE / f"sweep/{twin}/{Path(f).name}"
    tw = json.load(open(tf)) if tf.exists() else None
    a, b = path(r), path(tw) if tw else {}; ks = sorted(set(a) & set(b))
    g = round(max(float(np.linalg.norm(a[k] - b[k])) for k in ks), 3) if ks else None
    tc = G[str(ep)]["closure"]["t"]; b0 = np.array(r["bottle_start"]); pre = [x for x in r.get("trace") or [] if x["t"] < tc - 1.0]
    mv = max([float(np.linalg.norm(np.array(x["bottle"]) - b0)) for x in pre] or [0.0])
    tl, bl = r.get("robot_table_leg_ticks"), r.get("robot_bin_leg_ticks") or 0
    ok = bool(r.get("in_bin") and r.get("grasp_ok") and mv <= 0.02 and g is not None and g <= 0.10 and tl is not None and tl <= 25 and bl <= 25)
    out[Path(f).stem] = {"ep": ep, "variant": r.get("variant"), "ok": ok, "in_bin": r.get("in_bin"), "grasp_ok": r.get("grasp_ok"), "gap": g, "gap_free": gap_free(r)[0],
                         "pre_moved": round(mv, 3), "leg": bl, "table_leg": tl, "twin_grasp": (tw or {}).get("grasp_ok"),
                         "job": {k: r.get(k) for k in ("bottle_xy_cmd", "bin_xy_cmd", "bin_yaw_deg")}}
json.dump(out, open(HERE / f"valid_{tag}.json", "w"), indent=1)
V = list(out.values()); eps = sorted({v["ep"] for v in V if v["ok"]})
print(f"{tag} vs twin {twin}: runs {len(V)}, grasp+bin {sum(bool(v['grasp_ok'] and v['in_bin']) for v in V)}, within 10 cm of the twin {sum(1 for v in V if v['gap'] is not None and v['gap'] <= 0.10)}, "
      f"legs clear of table+bin {sum(1 for v in V if (v['table_leg'] or 0) <= 25 and v['leg'] <= 25)}, valid episodes {len(eps)}: {eps}")
