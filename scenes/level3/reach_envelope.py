#!/usr/bin/env python3
"""Reach geometry of scene sets vs the training demonstrations, per benchmark task.

Features per scene (world frame, robot yaw taken from its quaternion when stored):
  rel_x, rel_y   target position minus robot start, in the robot's heading frame (m)
  tz             target height (m);   top = table top height (m)
  cx, cy         container (if any) minus robot start, robot frame
    python3 reach_envelope.py [--json out.json]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import os
A = os.environ.get("LEVEL3_DEMO_ROOT", "/mnt/nas28/alan.jiang/psi_new_0828")          # the training demonstrations (100 per task)
ROOT = Path(__file__).resolve().parents[2]
DEMOS = {
    "G1WholebodyTabletopGraspMP-v0": f"{A}/simple/G1WholebodyTabletopGraspMP-v0",
    "G1WholebodyBendPickMP-v0": f"{A}/simple/G1WholebodyBendPickMP-v0",
    "G1WholebodyXMovePickTeleop-v0": f"{A}/simple-teleop/G1WholebodyXMovePickTeleop-v0/level-0",
    "G1WholebodyHandoverTeleop-v0": f"{A}/simple/G1WholebodyHandoverTeleop-v0",
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0": f"{A}/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    "G1WholebodyXMoveBendPickTeleop-v0": f"{A}/simple-archive/G1WholebodyXMoveBendPickTeleop-v0",
}
ROBOTS = ("g1_sonic", "g1_wholebody", "robot")
L3 = str(ROOT / "data/evals_scenes_benchmark/{t}/dr-level-3")
BASE = os.environ.get("LEVEL3_BASE_ROOT", "/mnt/nas28/alan.jiang/simple-eval-new20") + "/{t}/dr-level-0"


def load(p):
    out = []
    for l in open(Path(p) / "meta" / "episodes.jsonl"):
        if l.strip():
            c = json.loads(l).get("environment_config")
            while isinstance(c, str):
                c = json.loads(c)
            if c:
                out.append(c)
    return out


def yaw_of(r):
    q = r.get("quaternion") or r.get("orientation") or r.get("quat")
    if not q or len(q) != 4:
        return 0.0
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def feats(c):
    d = c["dr_state_dict"]; sp = d["spatial"]
    rk = next(k for k in sp if k in ROBOTS); r = sp[rk]
    rp = np.array(r["position"], float); yaw = yaw_of(r)
    uid = str((d.get("target") or {}).get("uid"))
    if uid not in sp:
        return None
    tp = np.array(sp[uid]["position"], float)
    R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    rel = R @ (tp[:2] - rp[:2])
    t = (d.get("scene") or {}).get("table") or {}
    top = t["pose"]["position"][2] + t["size"][2] / 2 if t else float("nan")
    f = {"rel_x": rel[0], "rel_y": rel[1], "tz": tp[2], "top": top, "yaw": math.degrees(yaw)}
    ck = next((k for k in sp if str(k).startswith("container")), None)
    if ck:
        cp = np.array(sp[ck]["position"], float); cr = R @ (cp[:2] - rp[:2]); f["cx"], f["cy"] = cr[0], cr[1]
    return f


def table(name, rows):
    ks = [k for k in ("rel_x", "rel_y", "tz", "top", "cx", "cy", "yaw") if all(k in r for r in rows)]
    a = {k: np.array([r[k] for r in rows]) for k in ks}
    return ks, a


def main():
    report = {}
    for t, dp in DEMOS.items():
        demo = [f for f in (feats(c) for c in load(dp)) if f]
        l3 = [f for f in (feats(c) for c in load(L3.format(t=t))) if f]
        ks, D = table("demo", demo); _, T = table("l3", l3)
        short = t.replace("G1Wholebody", "").replace("-v0", "")
        print(f"\n=== {short}: {len(demo)} demos, {len(l3)} level-3 scenes")
        print(f"{'':6s} {'demo min':>9s} {'demo max':>9s} | {'L3 min':>8s} {'L3 max':>8s}   (cm; yaw deg)")
        for k in ks:
            s = 1 if k == "yaw" else 100
            print(f"{k:6s} {D[k].min()*s:9.1f} {D[k].max()*s:9.1f} | {T[k].min()*s:8.1f} {T[k].max()*s:8.1f}")
        # per scene: outside the demo box on any reach axis, and distance to the nearest demo in (rel_x, rel_y, tz)
        X = np.stack([D["rel_x"], D["rel_y"], D["tz"]], 1); Y = np.stack([T["rel_x"], T["rel_y"], T["tz"]], 1)
        nn = np.sqrt(((Y[:, None, :] - X[None, :, :]) ** 2).sum(-1)).min(1)
        out = [(i + 1) for i in range(len(l3)) if any(not (D[k].min() - 1e-9 <= l3[i][k] <= D[k].max() + 1e-9) for k in ("rel_x", "rel_y", "tz"))]
        print(f"scenes outside the demo box (rel_x, rel_y, tz): {len(out)}/{len(l3)} -> {out}")
        print(f"nearest-demo distance (cm): median {np.median(nn)*100:.1f}, max {nn.max()*100:.1f}; > 2 cm: {int((nn > 0.02).sum())}, > 4 cm: {int((nn > 0.04).sum())}")
        report[t] = {"demo": {k: [float(D[k].min()), float(D[k].max())] for k in ks}, "outside": out, "nn_cm": [round(float(v) * 100, 1) for v in nn]}
    if "--json" in sys.argv:
        json.dump(report, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
