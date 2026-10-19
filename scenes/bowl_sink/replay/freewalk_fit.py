#!/usr/bin/env python
"""Bowl-to-sink placement fitted from the FREE WALK (2026-09-28, user: "the robot movement was not complete, it was blocked by
the counter"). For every episode the recorded walk is first replayed with the counters far away (tag freewalk). From that trace:
  - at the recorded closure: bowl under the fingers (hand-mid + (0.06, 0.07) in the base frame); the bowl counter's near edge
    0.06 m before the bowl (BOWL_FROM_EDGE) -> robot_to_edge = near_x - TOE;
  - at the recorded release: the basin centred under the carried bowl -> sink_along, robot_dy.
Checks (pelvis -> counter edge at the grasp / release stance >= MIN_CLEAR) flag episodes whose recorded reach is too short
for ANY counter. `--from TAG` refines bowl/basin from a placed run's own hand positions (counters kept)."""
import argparse, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
lay = json.load(open(HERE.parent / "layout.json"))["layout"]
TOE, BFE, BASIN_L, BASIN_CY0, FRONT0 = lay["toe_offset"], 0.06, 0.54, lay["basin_cy"], lay["sink_front_y"]
OFF = np.array([0.06, 0.07]); MIN_CLEAR = 0.10
ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="frm", default="freewalk"); ap.add_argument("--fits", default=None)
ap.add_argument("--out", required=True); a = ap.parse_args()
cl = {int(k): v for k, v in json.load(open(HERE / "closure_rows_0918.json")).items()}
prev = json.load(open(a.fits)) if a.fits else {}
rot = lambda h, v: np.array([np.cos(h) * v[0] - np.sin(h) * v[1], np.sin(h) * v[0] + np.cos(h) * v[1]])
out, flags = {}, {}
for ep, c in sorted(cl.items()):
    f = HERE / f"trace_ep{ep}_{a.frm}.json"
    if not f.exists(): continue
    tk = [x for x in json.load(open(f)) if "summary" not in x]; rows = np.array([x["row"] for x in tk])
    at = lambda r: tk[int(np.abs(rows - r).argmin())]
    g, rl = at(c["grasp_row"]), at(c["release_row"])
    mid = lambda x: (np.array(x["index_tip_r"][:2]) + np.array(x["thumb_tip_r"][:2])) / 2
    bowl = mid(g) + rot(g["sim_heading_rel"], OFF); rel = mid(rl) + rot(rl["sim_heading_rel"], OFF)
    p = dict(prev.get(str(ep), {}))
    if a.frm == "freewalk" or "robot_to_edge" not in p:
        near_x = float(bowl[0] - BFE); rte = near_x - TOE
    else:
        rte = float(p["robot_to_edge"]); near_x = rte + TOE
    sink_along = near_x - float(rel[0]) + BASIN_L / 2; robot_dy = BASIN_CY0 - float(rel[1])
    front_y = FRONT0 - robot_dy
    clear_g = near_x - float(g["sim_xy"][0]); clear_r = float(rl["sim_xy"][1]) - front_y
    out[str(ep)] = {"robot_to_edge": round(rte, 3), "bowl_xy": [round(float(bowl[0]), 3), round(float(bowl[1]), 3)],
                    "sink_along": round(sink_along, 3), "robot_dy": round(robot_dy, 3),
                    "note": f"free-walk fit ({a.frm}): pelvis->bowl-counter edge {clear_g:+.2f} m at the grasp, pelvis->sink edge {clear_r:+.2f} m at the release"}
    if clear_g < MIN_CLEAR or clear_r < MIN_CLEAR: flags[ep] = (round(clear_g, 2), round(clear_r, 2))
json.dump(out, open(a.out, "w"), indent=1)
print(len(out), "episodes ->", a.out, "| reach too short for a counter (grasp, release clearance m):", flags)
