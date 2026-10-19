#!/usr/bin/env python
"""Grid of small bowl shifts (base frame at the grasp) for episodes that did not lift the bowl, keeping the free-walk counter
edge and sink. Two-closure recordings (the operator re-grasped) get the bowl under the LAST closure that lasts to the release.
Writes fit_<prefix><k>.json, one file per grid cell (fit_batch keys are episodes)."""
import argparse, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
lay = json.load(open(HERE.parent / "layout.json"))["layout"]; TOE, BFE = lay["toe_offset"], 0.06
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--walkgap", required=True); ap.add_argument("--prefix", required=True)
ap.add_argument("--from", dest="frm", default="freewalk"); a = ap.parse_args()
base = json.load(open(a.base)); W = json.load(open(a.walkgap))
rot = lambda h, v: np.array([np.cos(h) * v[0] - np.sin(h) * v[1], np.sin(h) * v[0] + np.cos(h) * v[1]])
GRID = [(0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015), (0.015, 0.015), (0.015, -0.015), (0.03, 0.0), (-0.015, 0.015)]
OFF = np.array([0.06, 0.077])
def runs(tk):
    on, out = False, []
    for x in tk:
        v = x["key_real"]["right_hand_index_0_joint"]
        if v > 0.6 and not on: on, s = True, x["row"]
        if v < 0.4 and on: on = False; out.append((s, x["row"]))
    if on: out.append((s, tk[-1]["row"]))
    return sorted(set(out))
todo = [int(e) for e, v in W.items() if not v["placed"]]
cells = [dict() for _ in GRID]; recl = {}
for ep in todo:
    p = dict(base[str(ep)]); tk = [x for x in json.load(open(HERE / f"trace_ep{ep}_{a.frm}.json")) if "summary" not in x]
    R = runs(tk); h0 = None
    if len(R) >= 2:                                   # re-grasp: bowl under the last closure, counter edge from it
        s = R[-1][0]; g = min(tk, key=lambda x: abs(x["row"] - s)); mid = (np.array(g["index_tip_r"][:2]) + np.array(g["thumb_tip_r"][:2])) / 2
        b = mid + rot(g["sim_heading_rel"], OFF); p["bowl_xy"] = [round(float(b[0]), 3), round(float(b[1]), 3)]
        p["robot_to_edge"] = round(float(b[0] - BFE - TOE), 3); p["note"] = f"re-grasp: bowl under closure {R[-1]}"; recl[ep] = R
        g0 = g
    else:
        g0 = min(tk, key=lambda x: abs(x["row"] - R[0][0])) if R else tk[0]
    h = g0["sim_heading_rel"]
    for k, (dx, dy) in enumerate(GRID):
        q = dict(p); d = rot(h, (dx, dy)); q["bowl_xy"] = [round(p["bowl_xy"][0] + float(d[0]), 3), round(p["bowl_xy"][1] + float(d[1]), 3)]
        q["robot_to_edge"] = round(min(p["robot_to_edge"], q["bowl_xy"][0] - BFE - TOE), 3); q["grid"] = [dx, dy]
        q["sink_along"] = round(base[str(ep)]["sink_along"] + (q["robot_to_edge"] - base[str(ep)]["robot_to_edge"]), 3); cells[k][str(ep)] = q
reg = {str(e): dict(base[str(e)]) for e in recl}
for e in recl:
    s = recl[e][-1][0]; tk = [x for x in json.load(open(HERE / f"trace_ep{e}_{a.frm}.json")) if "summary" not in x]; g = min(tk, key=lambda x: abs(x["row"] - s))
    mid = (np.array(g["index_tip_r"][:2]) + np.array(g["thumb_tip_r"][:2])) / 2; b = mid + rot(g["sim_heading_rel"], OFF)
    reg[str(e)].update(bowl_xy=[round(float(b[0]), 3), round(float(b[1]), 3)], robot_to_edge=round(float(b[0] - BFE - TOE), 3), sink_along=round(base[str(e)]["sink_along"] + (float(b[0] - BFE - TOE) - base[str(e)]["robot_to_edge"]), 3), note=f"re-grasp: bowl under closure {recl[e][-1]}")
json.dump(reg, open(HERE / f"fit_{a.prefix}0.json", "w"), indent=1)
for k, c in enumerate(cells): json.dump(c, open(HERE / f"fit_{a.prefix}{k + 1}.json", "w"), indent=1)
print(f"{len(todo)} episodes x {len(GRID)} cells -> fit_{a.prefix}1..{len(GRID)}; re-grasp episodes {recl} -> fit_{a.prefix}0 (unshifted)")
