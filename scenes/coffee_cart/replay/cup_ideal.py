#!/usr/bin/env python
"""Coffee: cup placed where the recorded hand closes (palm + rot(yaw, (0.123, 0.067)) at the closure = the median in the physical
successes), in the cart frame, kept on the normal box top - WITHOUT the x-only finger clearance clamp (it pushed all 60 misses
0.2-0.6 m past the palm). Whether the fingers disturb the cup during the push is left to the pre-grasp check (<= 100 touch
ticks, <= 2 cm, <= 20 deg). Table from the free walk afterwards (freewalk_table.py). Usage: cup_ideal.py FROMTAG FITS OUT [dx dy]"""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; L = json.load(open(HERE.parent / "layout.json"))["layout"]; CART0 = np.array([L["cart_cx"], L["cart_cy"]])
frm, fits, outp = sys.argv[1], sys.argv[2], sys.argv[3]; dxy = np.array([float(v) for v in sys.argv[4:6]]) if len(sys.argv) > 5 else np.zeros(2)
BOX = (L["box_near_x"] + 0.05, L["box_far_x"] - 0.06, L["box_right_y"] + 0.05, L["box_left_y"] - 0.05)
rs = {json.loads(l)["ep"]: json.loads(l) for l in open(HERE / f"fitbatch_{frm}.jsonl")}; F = json.load(open(HERE / fits)); out = {}; n = 0
rot = lambda h, v: np.array([np.cos(h) * v[0] - np.sin(h) * v[1], np.sin(h) * v[0] + np.cos(h) * v[1]])
for e, p in F.items():
    q = dict(p); r = rs.get(int(e))
    if r is None or r.get("success"): continue
    cl = p.get("cup_close_row") or r.get("cup_close_row"); t = [x for x in json.load(open(HERE / f"trace_ep{e}_{frm}.json")) if "row" in x]
    rw = np.array([x["row"] for x in t]); c = t[int(np.abs(rw - cl).argmin())]
    sx = float(p.get("robot_to_cart", L["robot_to_cart"])) - L["robot_to_cart"]; C0 = CART0 + np.array([sx, 0.0])
    ideal = np.array(c["palm_r"][:2]) + rot(c["base_yaw"], np.array([0.123, 0.067]) + dxy) - (C0 + np.array(c["cart_xy"]))
    cup = C0 + ideal; cup = np.array([np.clip(cup[0], BOX[0] + sx, BOX[1] + sx), np.clip(cup[1], BOX[2], BOX[3])])
    q["cup_xy"] = [round(float(cup[0]), 3), round(float(cup[1]), 3)]; q.pop("note_clear", None); q.pop("box_from_handle", None)
    q["note"] = f"cup where the hand closes ({frm})" + (f" + {dxy.tolist()}" if dxy.any() else ""); out[e] = q; n += 1
json.dump(out, open(HERE / outp, "w"), indent=1); print(n, "failing episodes re-placed ->", outp)
