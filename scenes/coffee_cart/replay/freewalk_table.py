#!/usr/bin/env python
"""Coffee table placement from the FREE WALK (2026-09-28, user: the walk must not be blocked by counters/desks).
From the free twin (same fit, table 12 m ahead, tag --free) of every episode:
  - cup at the recorded release = palm + rot(yaw, (0.092, 0.061))  (cup - palm at release in 65 free-walk carries, IQR 1-2 cm);
  - table left edge (toward the robot) = min(pelvis_min_y - BODY, cup_y + 0.15): the top slab (0.74 m, hip height) stays
    >= BODY = 0.14 m from the pelvis over the whole walk after the grasp, and the cup lands 0.15 m inside when that fits
    (>= 0.05 m inside otherwise; flagged 'tight' if even that needs the edge closer than BODY);
  - table x: the cup 0.15 m inside the near edge (depth 0.60 m); the left edge also stays right of the cart's path (deck
    half-width 0.235 + 0.05) where the table's x-span overlaps it.
Only route_forward / table_from_line change; cup, cart and robot placement are kept from --fits."""
import argparse, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
L = json.load(open(HERE.parent / "layout.json"))["layout"]; CART0 = np.array([L["cart_cx"], L["cart_cy"]])
ap = argparse.ArgumentParser(); ap.add_argument("--fits", required=True); ap.add_argument("--free", default="cfree"); ap.add_argument("--out", required=True)
ap.add_argument("--body", type=float, default=0.14); a = ap.parse_args()
CUP_REL = np.array([0.092, 0.061]); DEPTH = 0.60; IN_X = 0.15; IN_Y = 0.15; IN_MIN = 0.05
fits = json.load(open(a.fits)); res = {json.loads(l)["ep"]: json.loads(l) for l in open(HERE / f"fitbatch_{a.free}.jsonl")}
rot = lambda h, v: np.array([np.cos(h) * v[0] - np.sin(h) * v[1], np.sin(h) * v[0] + np.cos(h) * v[1]])
out, tight = {}, []
for ep_s, p in fits.items():
    q = dict(p); ep = int(ep_s); f = HERE / f"trace_ep{ep}_{a.free}.json"
    if "skip" in p or not f.exists(): out[ep_s] = q; continue
    r = res.get(ep, {}); cl = p.get("cup_close_row") or r.get("cup_close_row"); op = p.get("cup_open_row") or r.get("cup_open_row")
    t = [x for x in json.load(open(f)) if "row" in x]; rw = np.array([x["row"] for x in t])
    o = t[max(0, int(np.abs(rw - op).argmin()) - 5)]; cup = np.array(o["palm_r"][:2]) + rot(o["base_yaw"], CUP_REL)
    after = [x for x in t if x["row"] >= cl]; ymin = min(x["base_xy_abs"][1] for x in after)
    left = min(ymin - a.body, cup[1] + IN_Y); near_x = cup[0] - IN_X; far_x = near_x + DEPTH
    sx = float(p.get("robot_to_cart", L["robot_to_cart"])) - L["robot_to_cart"]; C0 = CART0 + np.array([sx, 0.0])
    cart = np.array([C0 + np.array(x["cart_xy"]) for x in t]); lane = cart[(cart[:, 0] > near_x - 0.36) & (cart[:, 0] < far_x + 0.36)]
    if len(lane): left = min(left, float(lane[:, 1].min()) - 0.235 - 0.05)
    flag = None
    if cup[1] > left - IN_MIN: left = float(cup[1] + IN_MIN); flag = "tight"; tight.append((ep, round(ymin - left, 3)))
    q.update(route_forward=round(float(far_x), 3), table_from_line=round(float(-left), 3))
    q["note_table"] = f"free-walk table: cup at release ({cup[0]:.2f}, {cup[1]:.2f}), pelvis min y {ymin:.2f}, edge {left:.2f}" + (f" TIGHT (pelvis {ymin - left:.2f} m from the edge)" if flag else "")
    out[ep_s] = q
json.dump(out, open(a.out, "w"), indent=1)
print(len(out), "episodes ->", a.out, "| tight (edge closer than", a.body, "m to the pelvis):", tight)
