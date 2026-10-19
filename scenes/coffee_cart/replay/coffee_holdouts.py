#!/usr/bin/env python
"""Coffee holdouts after the closed-loop cup rounds (2026-09-29). Per episode, from its best closed-loop attempt:
  table: the cup is grasped and carried but released away from the table (the table came from an older free walk whose path
         differs) -> table under THIS run's release point: edge = min(pelvis_min_y - 0.14, cup_y + IN_Y), cup IN_X inside the near
         edge (variants move it further inside);
  box:   the wanted closure spot lies nearer the handle than the box top reaches (clipped in x) -> extend the box top toward the
         handle (COFFEE_CART_BOX_FROM_HANDLE, the real cart's tray reaches the handle); the cup at the wanted spot. The pre-grasp
         check still rejects a cup that rides in the hand.
Usage: coffee_holdouts.py OUT_PREFIX -> fit_<prefix>1..3.json (variants)"""
import json, sys, glob, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; L = json.load(open(HERE.parent / "layout.json"))["layout"]; CART0 = np.array([L["cart_cx"], L["cart_cy"]])
pre = sys.argv[1]
done = set()
for f in glob.glob(str(HERE / "walkgap_coffeevid*.json")):
    if not f.endswith("f.json"): done |= {int(e) for e, v in json.load(open(f)).items() if v["ok"]}
rest = [e for e in range(97) if e not in done and e != 77]
def attempts(e):
    out = []
    for f in sorted(glob.glob(str(HERE / "fitbatch_ccl??.jsonl"))):
        tag = Path(f).stem[len("fitbatch_"):]
        for l in open(f):
            r = json.loads(l)
            if r["ep"] == e: out.append((tag, r))
    return out
V = [dict(), dict(), dict()]; kinds = {}
for e in rest:
    A = attempts(e)
    if not A: continue
    good = [(t, r) for t, r in A if r.get("grasped") and r.get("cup_lifted") and (r.get("pre_grasp") or {}).get("ok", True)]
    if good:                                                     # table under this run's release
        tag, r = good[-1]; p = dict(r["params"]); t = [x for x in json.load(open(HERE / f"trace_ep{e}_{tag}.json")) if "row" in x]
        rw = np.array([x["row"] for x in t]); o = t[max(0, int(np.abs(rw - (p.get("cup_open_row") or r.get("cup_open_row"))).argmin()) - 3)]
        cup = np.array(o["target_xyz"][:2]); ymin = min(x["base_xy_abs"][1] for x in t if x["row"] >= (p.get("cup_close_row") or 0))
        for k, (inx, iny) in enumerate(((0.15, 0.15), (0.25, 0.20), (0.15, 0.25))):
            q = dict(p); left = min(ymin - 0.14, cup[1] + iny); flag = cup[1] > left - 0.05
            if flag: left = cup[1] + 0.05
            q.update(route_forward=round(float(cup[0] - inx + 0.6), 3), table_from_line=round(float(-left), 3), note=f"table under this run's release ({tag}): cup ({cup[0]:.2f}, {cup[1]:.2f}), pelvis min y {ymin:.2f}" + (" TIGHT" if flag else ""))
            V[k][str(e)] = q
        kinds[e] = "table"
    else:                                                        # box extension where the wanted spot was clipped in x
        tag, r = A[-1]; F = json.load(open(HERE / f"fit_{tag}.json")); p = dict(F[str(e)]); note = p.get("note", "")
        if "clipped [" not in note: continue
        cx, cy = [float(v) for v in note.split("clipped [")[1].split("]")[0].split(",")]
        if cx > -0.02: continue                                   # only x clipping toward the handle can be fixed by the tray
        for k, extra in enumerate((0.0, 0.02, 0.04)):
            q = dict(p); bfh = round(0.09 + cx - 0.02 - extra, 3); q["box_from_handle"] = bfh
            q["cup_xy"] = [round(float(p["cup_xy"][0] + cx - extra), 3), round(float(p["cup_xy"][1] + cy), 3)]
            q["note"] = f"box top extended toward the handle ({bfh:+.3f} m from the handle end), cup at the wanted closure spot ({tag}) - {extra}"
            V[k][str(e)] = q
        kinds[e] = "box"
for k, v in enumerate(V): json.dump(v, open(HERE / f"fit_{pre}{k + 1}.json", "w"), indent=1)
print("holdouts:", len(rest), "| table:", sorted(e for e, x in kinds.items() if x == "table"), "| box:", sorted(e for e, x in kinds.items() if x == "box"), "| no lever:", sorted(set(rest) - set(kinds)))
