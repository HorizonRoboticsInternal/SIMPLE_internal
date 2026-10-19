#!/usr/bin/env python
"""Coffee closed-loop cup placement (2026-09-29). cup_ideal.py mapped the closure spot into the cart frame WITHOUT the cart's yaw;
the cart turns during the push, so the misses had the cup 5-10 cm left of / ahead of the palm at the closure. Here, from a run's
own trace at the recorded closure: error = (cup - palm in the base frame) - (TARGET + offset), rotated to world, then into the
cart's reset frame by the cart yaw at the closure (from the cup's ride: angle of cup-cart now vs at reset, +/-45 deg), and the cup
start moves by -error (kept on the normal box top; 'clipped' noted). Successes keep their fit.
Usage: cup_closed_loop.py FROMTAG FITS OUT [dx dy]  (offset of the wanted closure position, base frame)"""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; L = json.load(open(HERE.parent / "layout.json"))["layout"]; CART0 = np.array([L["cart_cx"], L["cart_cy"]])
frm, fits, outp = sys.argv[1], sys.argv[2], sys.argv[3]; off = np.array([float(v) for v in sys.argv[4:6]]) if len(sys.argv) > 5 else np.zeros(2)
TARGET = np.array([0.123, 0.067]) + off; BOX = (L["box_near_x"] + 0.05, L["box_far_x"] - 0.06, L["box_right_y"] + 0.05, L["box_left_y"] - 0.05)
rs = {json.loads(l)["ep"]: json.loads(l) for l in open(HERE / f"fitbatch_{frm}.jsonl")}; F = json.load(open(HERE / fits)); out = {}; n = 0
rot = lambda h, v: np.array([np.cos(h) * v[0] - np.sin(h) * v[1], np.sin(h) * v[0] + np.cos(h) * v[1]])
ang = lambda v: float(np.arctan2(v[1], v[0]))
for e, p in F.items():
    q = dict(p); r = rs.get(int(e))
    if r is None: continue
    if r.get("success"): continue
    cl = p.get("cup_close_row") or r.get("cup_close_row"); t = [x for x in json.load(open(HERE / f"trace_ep{e}_{frm}.json")) if "row" in x]
    rw = np.array([x["row"] for x in t]); c = t[int(np.abs(rw - cl).argmin())]
    sx = float(p.get("robot_to_cart", L["robot_to_cart"])) - L["robot_to_cart"]; C0 = CART0 + np.array([sx, 0.0])
    rel = rot(-c["base_yaw"], np.array(c["target_xyz"][:2]) - np.array(c["palm_r"][:2])); err_w = rot(c["base_yaw"], rel - TARGET)
    v0 = np.array(t[0]["target_xyz"][:2]) - C0; vc = np.array(c["target_xyz"][:2]) - (C0 + np.array(c["cart_xy"]))
    psi = float(np.clip((ang(vc) - ang(v0) + np.pi) % (2 * np.pi) - np.pi, -np.pi / 4, np.pi / 4)) if c["target_xyz"][2] > 0.7 else 0.0
    new = np.array(p["cup_xy"]) - rot(-psi, err_w)
    cl_ = np.array([np.clip(new[0], BOX[0] + sx, BOX[1] + sx), np.clip(new[1], BOX[2], BOX[3])])
    q["cup_xy"] = [round(float(cl_[0]), 3), round(float(cl_[1]), 3)]
    q["note"] = f"closed-loop cup ({frm}): closure error {np.round(rel - TARGET, 3).tolist()} base frame, cart yaw {np.degrees(psi):+.0f} deg" + (f", clipped {np.round(new - cl_, 3).tolist()}" if np.linalg.norm(new - cl_) > 0.005 else "")
    out[e] = q; n += 1
json.dump(out, open(HERE / outp, "w"), indent=1); print(n, "failing episodes re-placed ->", outp)
