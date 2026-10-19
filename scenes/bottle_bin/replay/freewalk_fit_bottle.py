#!/usr/bin/env python
"""Bottle-to-bin placement fitted from the FREE WALK (2026-09-28; the walk must not be blocked by the table or the bin).
From sweep/freewalk (table 6 m ahead, bin 21 m away), per episode:
  - closure (episode_geometry_97 closure.t): bottle = palm + BOT_OFF (world; the robot has not turned yet), kept on the table top;
  - release (release.t): bin centre = palm + rot(heading, BIN_OFF), at least BIN_MIN_X ahead of the pelvis along the heading;
    bin yaw = heading (the layout's -90 deg bin with the robot facing -y).
Calibrated on the physical successes 2/8/46/56/80/88: bottle - palm at closure (0.098, 0.094); bin - palm at release
(0.20, 0.13) base frame; bin - pelvis >= 0.36 m, blocked at 0.355 with 23 leg ticks -> 0.40.
Usage: freewalk_fit_bottle.py --out jobs_fwb0.json [--shift dx dy --bin-shift dx dy]  (shifts in the base frame)"""
import argparse, json, glob, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--episodes", type=int, nargs="*")
ap.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0)); ap.add_argument("--bin-shift", type=float, nargs=2, default=(0.0, 0.0))
ap.add_argument("--variant", default=None); a = ap.parse_args()
G = json.load(open(HERE / "episode_geometry_97.json"))["episodes"]
BOT_OFF = np.array([0.098, 0.094]); BIN_OFF = np.array([0.20, 0.13]); BIN_MIN_X = 0.40; NEAR_X, R_BOT = 0.155, 0.033
rot = lambda h, v: np.array([np.cos(h) * v[0] - np.sin(h) * v[1], np.sin(h) * v[0] + np.cos(h) * v[1]])
jobs = []
for f in sorted(glob.glob(str(HERE / "sweep/freewalk/*.json"))):
    r = json.load(open(f)); ep = r["ep"]
    if a.episodes and ep not in a.episodes: continue
    g = G[str(ep)]; T = r["trace"]; bp = np.array(r["base_path"])
    at = lambda t: T[int(np.argmin([abs(x["t"] - t) for x in T]))]
    pc = at(g["closure"]["t"]); hc = bp[int(np.argmin(abs(bp[:, 0] - g["closure"]["t"]))), 3]
    b = np.array(pc["palm"][:2]) + rot(hc, BOT_OFF + np.array(a.shift)); b[0] = max(b[0], NEAR_X + R_BOT + 0.01)
    tr = g["release"]["t"]; pr = at(tr); j = int(np.argmin(abs(bp[:, 0] - tr))); base, h = bp[j, 1:3], bp[j, 3]
    binc = np.array(pr["palm"][:2]) + rot(h, BIN_OFF + np.array(a.bin_shift))
    along = float(rot(-h, binc - base)[0])
    if along < BIN_MIN_X: binc = binc + rot(h, (BIN_MIN_X - along, 0.0))
    jobs.append({"ep": ep, "bottle_xy": [round(float(v), 4) for v in b], "bin_xy": [round(float(v), 4) for v in binc], "bin_yaw": round(float(np.degrees(h)), 1),
                 **({"variant": a.variant} if a.variant else {}), "note": f"free-walk fit: bin {along:.2f} m ahead of the pelvis at the release (min {BIN_MIN_X})"})
json.dump(jobs, open(HERE / a.out, "w"), indent=0); print(len(jobs), "jobs ->", a.out)
