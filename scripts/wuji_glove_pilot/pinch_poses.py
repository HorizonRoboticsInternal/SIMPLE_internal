#!/usr/bin/env python
"""Robot pinch poses for the Revo 2 (2026-11-03): per finger, the (thumb rotation, thumb bend, finger) motor targets that
put the thumb pad on that finger's pad in the sim. The bone-angle retarget alone leaves the pads apart on a human pinch
(a 2-motor thumb cannot copy the human CMC), so the retarget blends toward these poses when the human thumb tip nears a
fingertip; the index pose is the tuned grip pose CLOSE2_Q, the others come from here.

Method, on the bench rig (fixed robot, the teleop's hand PD and solver): a kinematic grid over the three motors (qpos set
directly, other fingers open) keeps the poses whose pad geoms do not interpenetrate by more than MAX_PEN and ranks them
by the distance between the two pad centres (the touch sites); the best candidates then run through the hand PD from the
open hand for 1 s, and the one with the smallest settled pad-centre distance among those whose pads touch wins. Prints a
table per side and the PINCH_POSES dict for revo2_retarget.py (right-hand values; the left hand is checked against them).

  MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/pinch_poses.py [--step 0.05] [--side right]
  MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/pinch_poses.py --table      # thumb pose per finger CURL -> PINCH_KNOTS

--table (2026-11-03) is what the retarget uses now: for finger curls from 0.2 rad to the limit it finds the thumb (rotation,
bend) that lands on the finger's pad with the finger held at that curl, so the finger can follow the operator and the thumb
meets it; the gap column says where the thumb cannot reach (> 1 mm). The knots in revo2_retarget.PINCH_KNOTS are a
hand-smoothed reading of that table (a fixed thumb pose holds contact across most of a band; the middle finger needs more
thumb bend when strongly curled).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from bench import ROBOT_XML, Rig, build_model  # noqa: E402
from revo2_retarget import ACTIVE, MOTOR_UPPER, MOTORS, motors_to_joints  # noqa: E402
from simple.robots.g1_sonic_revo2 import CLOSE2_Q, OPEN_Q  # noqa: E402

FINGERS = ["index", "middle", "ring", "pinky"]
MAX_PEN_MM = 4.0            # kinematic candidates may interpenetrate up to this (the PD then stops at contact)
TOUCH_MM = 1.0              # dynamic: pads count as touching below this geom gap
N_DYN = 12                  # candidates per finger sent through the PD
SETTLE_TICKS = 50           # 1 s at 20 ms


def pad_gap_mm(rig: Rig, s: str, finger: str) -> float:
    g1, g2 = rig.m.geom(f"{s}_hand_thumb_touch_collision").id, rig.m.geom(f"{s}_hand_{finger}_touch_collision").id
    return 1e3 * float(mujoco.mj_geomDistance(rig.m, rig.d, g1, g2, 0.2, None))


def pad_centre_mm(rig: Rig, s: str, finger: str) -> float:
    p = lambda n: rig.d.site_xpos[rig.m.site(f"{s}_hand_{n}_touch_site").id]       # noqa: E731
    return 1e3 * float(np.linalg.norm(p("thumb") - p(finger)))


def set_hand(rig: Rig, s: str, motors: np.ndarray) -> None:
    rig.d.qpos[rig.qa[s]] = motors_to_joints(motors)
    rig.d.qvel[rig.va[s]] = 0.0
    mujoco.mj_forward(rig.m, rig.d)


def settle(rig: Rig, s: str, motors: np.ndarray) -> tuple[float, float, np.ndarray]:
    """From the open hand, the PD toward `motors` for SETTLE_TICKS: (pad gap mm, pad-centre distance mm, reached motors)."""
    other = "left" if s == "right" else "right"
    set_hand(rig, s, np.asarray(OPEN_Q)[ACTIVE])
    set_hand(rig, other, np.asarray(OPEN_Q)[ACTIVE])
    targets = {s: motors_to_joints(motors), other: np.asarray(OPEN_Q, dtype=np.float64)}
    for _ in range(SETTLE_TICKS):
        rig.step(targets)
    return pad_gap_mm(rig, s, FINGER), pad_centre_mm(rig, s, FINGER), rig.motors(s)


def search(rig: Rig, s: str, finger: str, step: float) -> dict:
    global FINGER
    FINGER = finger
    k = 2 + FINGERS.index(finger)
    open_m = np.asarray(OPEN_Q)[ACTIVE]
    grid = [np.arange(0.0, MOTOR_UPPER[i] + 1e-9, step) for i in (0, 1, k)]
    cands = []
    t0 = time.perf_counter()
    for rot, bend, f in itertools.product(*grid):
        m = open_m.copy()
        m[0], m[1], m[k] = rot, bend, f
        set_hand(rig, s, m)
        gap = pad_gap_mm(rig, s, finger)
        if gap > -MAX_PEN_MM:
            cands.append((pad_centre_mm(rig, s, finger), gap, m))
    n_grid = int(np.prod([len(g) for g in grid]))
    cands.sort(key=lambda c: c[0])
    best_kin = cands[0]
    dyn = []
    for centre, gap, m in cands[:N_DYN]:
        g, c, reached = settle(rig, s, m)
        dyn.append(dict(motors=m, kin_centre=centre, kin_gap=gap, gap=g, centre=c, reached=reached))
    touching = [d for d in dyn if d["gap"] <= TOUCH_MM]
    win = min(touching, key=lambda d: d["centre"]) if touching else min(dyn, key=lambda d: d["gap"])
    return dict(finger=finger, n_grid=n_grid, n_ok=len(cands), seconds=round(time.perf_counter() - t0, 1), best_kin=best_kin, win=win, dyn=dyn)


def thumb_table(rig: Rig, s: str, finger: str, step: float, n_curl: int = 14) -> list[tuple[float, float, float, float]]:
    """For finger motor values from 0.2 rad to the motor's limit: the (thumb rotation, thumb bend) that puts the thumb pad on
    the finger's pad with the finger held there (same kinematic grid -> PD settle as search(), thumb only). Returns
    (finger, rot, bend, settled pad gap mm) per curl value; the gap says where the thumb cannot reach (> TOUCH_MM)."""
    global FINGER
    FINGER = finger
    k = 2 + FINGERS.index(finger)
    open_m = np.asarray(OPEN_Q)[ACTIVE]
    grid = [np.arange(0.0, MOTOR_UPPER[i] + 1e-9, step) for i in (0, 1)]
    rows = []
    for v in np.linspace(0.2, MOTOR_UPPER[k], n_curl):
        cands = []
        for rot, bend in itertools.product(*grid):
            m = open_m.copy()
            m[0], m[1], m[k] = rot, bend, v
            set_hand(rig, s, m)
            gap = pad_gap_mm(rig, s, finger)
            if gap > -MAX_PEN_MM:
                cands.append((pad_centre_mm(rig, s, finger), gap, m))
        cands.sort(key=lambda c: c[0])
        dyn = []
        for centre, gap, m in cands[:6]:
            g, c, _ = settle(rig, s, m)
            dyn.append((m, g, c))
        touching = [x for x in dyn if x[1] <= TOUCH_MM]
        mm, g, c = min(touching, key=lambda x: x[2]) if touching else min(dyn, key=lambda x: x[1])
        rows.append((round(float(v), 3), round(float(mm[0]), 3), round(float(mm[1]), 3), round(float(g), 1)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05, help="motor grid step (rad)")
    ap.add_argument("--side", default="both", choices=["right", "left", "both"])
    ap.add_argument("--robot-mjcf", default=str(ROBOT_XML))
    ap.add_argument("--table", action="store_true", help="the thumb pose per finger CURL instead of one pinch pose per finger (PINCH_TABLE)")
    a = ap.parse_args()
    m, d = build_model(Path(a.robot_mjcf))
    rig = Rig(m, d)
    sides = ["right", "left"] if a.side == "both" else [a.side]
    if a.table:
        print(f"PINCH_TABLE = {{  # side -> finger -> [(finger motor, thumb rotation, thumb bend, settled pad gap mm)]; pinch_poses.py --table on {Path(a.robot_mjcf).name}")
        for s in sides:
            print(f"    {s!r}: {{")
            for f in FINGERS:
                t0 = time.perf_counter()
                rows = thumb_table(rig, s, f, a.step)
                print(f"        {f!r}: {rows},   # {time.perf_counter() - t0:.0f} s", flush=True)
            print("    },")
        print("}")
        return
    out = {}
    for s in sides:
        print(f"\n== {s} hand  ({Path(a.robot_mjcf).name}, grid step {a.step} rad)")
        # the tuned index pinch pose, for reference
        global FINGER
        FINGER = "index"
        g, c, _ = settle(rig, s, np.asarray(CLOSE2_Q)[ACTIVE])
        print(f"   CLOSE2_Q (rot {CLOSE2_Q[0]:.3f} bend {CLOSE2_Q[1]:.3f} index {CLOSE2_Q[3]:.3f}): settled pad gap {g:5.1f} mm, pad centres {c:5.1f} mm apart")
        out[s] = {}
        for f in FINGERS:
            r = search(rig, s, f, a.step)
            w, k = r["win"], 2 + FINGERS.index(f)
            out[s][f] = (round(float(w["motors"][0]), 3), round(float(w["motors"][1]), 3), round(float(w["motors"][k]), 3))
            print(f"   {f:6s} rot {w['motors'][0]:.3f} bend {w['motors'][1]:.3f} {f} {w['motors'][k]:.3f}  ->  settled pad gap {w['gap']:5.1f} mm, "
                  f"centres {w['centre']:5.1f} mm (kinematic {w['kin_gap']:5.1f} / {w['kin_centre']:5.1f} mm); reached rot {w['reached'][0]:.2f} bend {w['reached'][1]:.2f} "
                  f"{f} {w['reached'][k]:.2f}   [{r['n_ok']}/{r['n_grid']} grid poses ok, {r['seconds']} s]")
            if w["gap"] > TOUCH_MM:
                print(f"          WARNING: no candidate touched (best gap {w['gap']:.1f} mm)")
    if len(sides) == 2:
        for f in FINGERS:
            diff = np.abs(np.array(out["right"][f]) - np.array(out["left"][f])).max()
            if diff > 0.051:
                print(f"   NOTE {f}: left and right differ by {diff:.3f} rad")
    s0 = sides[0]
    print("\nPINCH_POSES = {  # (thumb rotation, thumb bend, finger) motor targets, rad; pinch_poses.py on " + Path(a.robot_mjcf).name)
    for f in FINGERS:
        print(f"    {f!r}: {out[s0][f]},")
    print("}")


if __name__ == "__main__":
    main()
