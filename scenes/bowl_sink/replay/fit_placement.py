#!/usr/bin/env python3
"""Where should the robot start, in the fixed kitchen, so that the most replayed episodes complete?

The simulated hand's path does not depend on what the scene contains, so one replay per episode at a given
walking gain gives, per episode, where the hand closes on the bowl (C) and where it opens over the sink (R), in
the start frame.  The kitchen is rigid: choosing the start pose (feet -> counter edge, pelvis -> counter's left
end) fixes both the bowl and the basin in that frame.  This script reads the traces of a replay batch, extracts
C and R, and grid-searches the start pose.

    python replay/fit_placement.py --tag g130 [--tol 0.09]
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
SC = "/tmp/claude-1000/-home-Horizon-wrk-SIMPLE/66a4ba73-f074-4719-b3c1-1268e8145abc/scratchpad"

L = json.load(open(SCENE / "layout.json"))
lay = L["layout"]
TOE = lay["toe_offset"]                       # pelvis -> front of the feet
BOWL_FROM_EDGE = lay["bowl_x"] - lay["near_x"]
ALONG = lay["sink_along"]                     # sink counter's start (= bowl counter's front edge) -> basin far edge
BASIN_L, BASIN_A = L["sink"]["basin"][0], L["sink"]["basin"][1]
COUNTER_W, SINK_D = L["counter"]["size"][1], L["sink"]["counter"][1]
BOWL_R = L["bowl"]["diameter"] / 2


def extract(tag):
    rows_tbl = {int(k): v for k, v in json.load(open(HERE / "closure_rows_0918.json")).items()}   # export episode -> rows
    rows = []
    for f in sorted(glob.glob(str(HERE / f"trace_ep*_{tag}.json"))):
        ep = int(re.search(r"trace_ep(\d+)_", f).group(1))
        tr = json.load(open(f)); tk = [r for r in tr if "summary" not in r]
        s = next((r["summary"] for r in tr if "summary" in r), {}); g = s.get("grasp", {}) or {}
        if not tk:
            continue
        mid = (np.array([r["index_tip_r"] for r in tk]) + np.array([r["palm_r"] for r in tk])) / 2
        ri = np.array([r["row"] for r in tk])
        grow = rows_tbl.get(ep, {}).get("grasp_row"); rrow = rows_tbl.get(ep, {}).get("release_row")
        if grow is None or rrow is None:
            continue
        rows.append(dict(ep=ep, lift=g.get("max_lift_m"), in_sink=bool(g.get("in_sink")),
                         C=[float(v) for v in mid[int(np.abs(ri - grow).argmin())]],
                         R=[float(v) for v in mid[int(np.abs(ri - rrow).argmin())]]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tol", type=float, default=0.09, help="hand-to-bowl distance at closure that still grasps (m)")
    a = ap.parse_args()
    rows = extract(a.tag)
    json.dump(rows, open(HERE / f"batch_{a.tag}_points.json", "w"), indent=1)
    C = np.array([r["C"][:2] for r in rows]); R = np.array([r["R"][:2] for r in rows]); n = len(rows)
    print(f"[{a.tag}] {n} episodes with traces")
    print(f"  hand at closure: x p10/50/90 {np.round(np.percentile(C[:,0],[10,50,90]),3)}  y {np.round(np.percentile(C[:,1],[10,50,90]),3)}")
    print(f"  hand at release: x p10/50/90 {np.round(np.percentile(R[:,0],[10,50,90]),3)}  y {np.round(np.percentile(R[:,1],[10,50,90]),3)}")
    d = R - C
    print(f"  carry (release - closure): x p50 {np.median(d[:,0]):+.3f}  y p50 {np.median(d[:,1]):+.3f}   kitchen needs x {-(ALONG - BASIN_L/2 + BOWL_FROM_EDGE):+.3f} to the basin centre")

    def score(rte, fl, tol=a.tol):
        near_x = rte + TOE
        bx, by = near_x + BOWL_FROM_EDGE, 0.0
        sx, sy = near_x - ALONG + BASIN_L / 2, (fl - COUNTER_W) - SINK_D / 2
        g = np.hypot(C[:, 0] - bx, C[:, 1] - by) < tol
        p = (np.abs(R[:, 0] - sx) < BASIN_L / 2 - BOWL_R) & (np.abs(R[:, 1] - sy) < BASIN_A / 2 - BOWL_R)
        return int((g & p).sum()), int(g.sum()), int(p.sum())

    rtes = np.arange(0.10, 0.80, 0.01); fls = np.arange(1.10, 2.40, 0.01)
    best = max(((score(r, f), r, f) for r in rtes for f in fls), key=lambda t: t[0])
    (cc, gg, pp), rte, fl = best
    print(f"\n  BEST start pose: feet {rte:.2f} m from the counter edge, pelvis {fl:.2f} m from its left end")
    print(f"    -> complete {cc}/{n}   grasp {gg}   drop-in {pp}")
    print(f"    bowl at {rte + TOE + BOWL_FROM_EDGE:.3f} m ahead; basin centre at ({rte + TOE - ALONG + BASIN_L/2:.3f}, {fl - COUNTER_W - SINK_D/2:.3f}) m")
    print("  sensitivity, feet->edge (from-left at best):", "  ".join(f"{r:.2f}:{score(r, fl)[0]}" for r in np.arange(max(0.10, rte - 0.15), rte + 0.16, 0.05)))
    print("  sensitivity, from-left (feet->edge at best):", "  ".join(f"{f:.2f}:{score(rte, f)[0]}" for f in np.arange(fl - 0.30, fl + 0.31, 0.10)))
    cur = score(lay["near_x"] - TOE, lay["left_y"])
    print(f"  scene as built (feet {lay['near_x'] - TOE:.2f}, from-left {lay['left_y']:.2f}): complete {cur[0]}  grasp {cur[1]}  drop-in {cur[2]}")
    json.dump(dict(tag=a.tag, n=n, robot_to_edge=float(rte), robot_from_left=float(fl), complete=cc, grasp=gg, drop_in=pp,
                   tol=a.tol, carry_x=float(np.median(d[:, 0])), carry_y=float(np.median(d[:, 1]))),
              open(HERE / f"fit_placement_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
