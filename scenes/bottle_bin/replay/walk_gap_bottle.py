#!/usr/bin/env python
"""Bottle walk completeness: max distance between a run's base path and the free walk (table 6 m ahead, bin 21 m away) at the
same times (base_path rows [t, x, y, ...], 0.1 s). Usage: walk_gap_bottle.py TAG [TAG ...] -> prints per-episode gap."""
import json, glob, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
FW = {}
for f in glob.glob(str(HERE / "sweep/freewalk/*.json")):
    r = json.load(open(f)); FW[r["ep"]] = {round(p[0], 1): np.array(p[1:3]) for p in r["base_path"]}
def gap(r):
    fw = FW.get(r["ep"]); bp = {round(p[0], 1): np.array(p[1:3]) for p in (r.get("base_path") or [])}
    ks = sorted(set(fw or {}) & set(bp))
    if not ks: return None, None
    d = [float(np.linalg.norm(bp[k] - fw[k])) for k in ks]; i = int(np.argmax(d)); return round(d[i], 3), ks[i]
if __name__ == "__main__":
    out = {}
    for tag in sys.argv[1:]:
        for f in sorted(glob.glob(str(HERE / f"sweep/{tag}/*.json"))):
            r = json.load(open(f)); g, t = gap(r)
            leg = r.get("robot_bin_leg_ticks") if r.get("robot_bin_leg_ticks") is not None else r.get("robot_bin_contact_ticks")
            ok = bool(r.get("in_bin") and r.get("grasp_ok") and g is not None and g <= 0.10)
            out.setdefault(tag, {})[r["ep"]] = {"gap": g, "t": t, "in_bin": r.get("in_bin"), "grasp_ok": r.get("grasp_ok"), "leg": leg, "ok": ok}
            print(tag, r["ep"], "gap", g, "at t", t, "in_bin", r.get("in_bin"), "grasp", r.get("grasp_ok"), "leg ticks", leg, "OK" if ok else "")
    json.dump(out, open(HERE / "walkgap_last.json", "w"), indent=1)
