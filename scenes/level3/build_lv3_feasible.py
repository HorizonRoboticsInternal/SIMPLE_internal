#!/usr/bin/env python3
"""Feasible level-3 scenes for the five SIMPLE benchmark tasks.

Every scene keeps what the level-3 generator drew for it (distractors, table material, lighting, the target's pose on
the table) and gets a new layout offset -- robot start (dx forward, dy left) and table-top height dz -- chosen inside
the range where the G1 can still do the task from that start:
  * the target, seen from the robot's start, stays inside the task's reach box (rel_x forward, rel_y left, metres),
  * the table moves by dz inside the task's height band,
  * |dx| <= 10 cm, |dy| <= 5 cm (the level-3 limits).
Within the allowed interval each scene takes the value of a shuffled, evenly spread design (seeded), so the scenes spread
over the whole feasible range.  Objects standing on the moved table move with it; a second table does not.

    python3 build_lv3_feasible.py <task> --src <current dr-level-3 dir> --out <staging dir> [--seed 0] [--redo scenes.json]
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np

ROBOTS = ("g1_sonic", "g1_wholebody", "robot")
HERE = Path(__file__).resolve().parent
DESIGN = HERE / "lv3_30_design.json"            # the offsets the candidate generation applied (gen_candidates.sh)
LIM_DX, LIM_DY = 0.10, 0.05
# reach box (target minus robot start, robot frame) and table-height band per task; filled from the demonstrations and the
# motion-planner calibration (see README.md in this folder)
RANGES = json.load(open(HERE / "lv3_ranges.json"))


def load(p):
    rows = [json.loads(l) for l in open(Path(p) / "meta" / "episodes.jsonl") if l.strip()]
    cfgs = []
    for r in rows:
        c = r["environment_config"]
        while isinstance(c, str):
            c = json.loads(c)
        cfgs.append(c)
    return rows, cfgs


def robot_key(sp):
    return next(k for k in sp if k in ROBOTS)


def on_table(pos, tbl):
    tx, ty, tz = tbl["pose"]["position"]; sx, sy, sz = tbl["size"]
    return abs(pos[0] - tx) <= sx / 2 + 0.02 and abs(pos[1] - ty) <= sy / 2 + 0.02 and pos[2] > tz - 0.02


def interval(lo, hi, lim):
    a, b = max(lo, -lim), min(hi, lim)
    return (a, b) if a <= b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", default=None, help="JSON {scene: shrink} -- rebuild these scenes with the box shrunk toward its centre by this fraction")
    a = ap.parse_args()
    R = RANGES[a.task]
    X, Y, Z = R["rel_x"], R["rel_y"], R["dz"]
    D = json.load(open(DESIGN)); n = len(D["x"])
    rows, cfgs = load(a.src)
    assert len(cfgs) == n, f"{len(cfgs)} scenes, design has {n}"
    rng = np.random.RandomState(a.seed + 101)
    ux, uy, uz = (rng.permutation(np.linspace(0, 1, n)) for _ in range(3))
    redo = {int(k): float(v) for k, v in json.load(open(a.redo)).items()} if a.redo else {}
    prev = {}
    rep_path = Path(a.out) / "build_report.json"
    if a.redo and rep_path.exists():
        prev = {r["scene"]: r for r in json.load(open(rep_path))["scenes"]}
    out_rows, report = [], []
    for i, (row, c) in enumerate(zip(rows, cfgs)):
        d = c["dr_state_dict"]; sp = d["spatial"]; rk = robot_key(sp)
        tbl = (d.get("scene") or {}).get("table")
        r_l3 = np.array(sp[rk]["position"][:2], float)
        base_robot = r_l3 - np.array([D["x"][i], D["y"][i]])      # the level-3 generator added the design offset to the base start
        tgt = np.array(sp[str(d["target"]["uid"])]["position"], float)
        rel0 = tgt[:2] - base_robot
        s = redo.get(i, prev[i]["shrink"] if i in prev else 0.0)  # shrink toward the box centre (retries after a planner failure); kept scenes reproduce exactly
        cx, cy, cz = (X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2, (Z[0] + Z[1]) / 2
        Xs = (cx + (X[0] - cx) * (1 - s), cx + (X[1] - cx) * (1 - s)); Ys = (cy + (Y[0] - cy) * (1 - s), cy + (Y[1] - cy) * (1 - s))
        Zs = (cz + (Z[0] - cz) * (1 - s), cz + (Z[1] - cz) * (1 - s))
        ix = interval(rel0[0] - Xs[1], rel0[0] - Xs[0], LIM_DX)   # rel_x = rel0_x - dx must lie in Xs
        iy = interval(rel0[1] - Ys[1], rel0[1] - Ys[0], LIM_DY)
        flags = []
        if ix is None:
            dx = float(np.clip(rel0[0] - (Xs[0] + Xs[1]) / 2, -LIM_DX, LIM_DX)); flags.append("x_clipped")
        else:
            dx = ix[0] + ux[i] * (ix[1] - ix[0])
        if iy is None:
            dy = float(np.clip(rel0[1] - (Ys[0] + Ys[1]) / 2, -LIM_DY, LIM_DY)); flags.append("y_clipped")
        else:
            dy = iy[0] + uy[i] * (iy[1] - iy[0])
        dz = Zs[0] + uz[i] * (Zs[1] - Zs[0])
        dx, dy, dz = round(float(dx), 4), round(float(dy), 4), round(float(dz), 4)
        ddz = dz - D["dz"][i]                                      # the level-3 generator already moved the table by the design dz
        c2 = copy.deepcopy(c); d2 = c2["dr_state_dict"]; sp2 = d2["spatial"]; tbl2 = (d2.get("scene") or {}).get("table")
        sp2[rk]["position"][0] = float(base_robot[0] + dx); sp2[rk]["position"][1] = float(base_robot[1] + dy)
        moved = []
        if tbl2 is not None:
            for k, v in sp2.items():
                if k != rk and isinstance(v, dict) and on_table(v["position"], tbl):
                    v["position"][2] += ddz; moved.append(k)
            tbl2["pose"]["position"][2] += ddz
        lay = (c2.get("layout") or {}).get("actors") or {}
        if isinstance(lay, dict):
            for name, act in lay.items():
                pos = (act.get("pose") or {}).get("position") if isinstance(act, dict) else None
                if not pos:
                    continue
                if name == "robot":
                    pos[0] = float(base_robot[0] + dx); pos[1] = float(base_robot[1] + dy)
                elif name == "table" or (tbl is not None and on_table(pos, tbl)):
                    pos[2] += ddz
        rel = rel0 - np.array([dx, dy])
        top = (tbl["pose"]["position"][2] + tbl["size"][2] / 2 - D["dz"][i] + dz) if tbl else None
        row2 = json.loads(json.dumps(row)); row2["environment_config"] = json.dumps(c2)
        out_rows.append(row2)
        report.append({"scene": i, "dx": dx, "dy": dy, "dz": dz, "rel_x": round(float(rel[0]), 4), "rel_y": round(float(rel[1]), 4),
                       "table_top": None if top is None else round(float(top), 4), "moved_with_table": moved, "flags": flags, "shrink": s})
    out = Path(a.out)
    if not a.redo:
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(Path(a.src) / "data", out / "data")         # the decoupled-WBC loader wants the episode rows
        (out / "meta").mkdir(parents=True)
        for f in ("info.json", "tasks.jsonl"):
            if (Path(a.src) / "meta" / f).exists():
                shutil.copy(Path(a.src) / "meta" / f, out / "meta" / f)
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"task": a.task, "ranges": R, "scenes": report}, open(rep_path, "w"), indent=1)
    fl = [r["scene"] for r in report if r["flags"]]
    print(f"{a.task}: {len(report)} scenes -> {out}; rel_x {min(r['rel_x'] for r in report)*100:.1f}..{max(r['rel_x'] for r in report)*100:.1f} cm, "
          f"rel_y {min(r['rel_y'] for r in report)*100:.1f}..{max(r['rel_y'] for r in report)*100:.1f} cm, dz {min(r['dz'] for r in report)*100:.1f}..{max(r['dz'] for r in report)*100:.1f} cm; "
          f"clipped scenes {fl}")


if __name__ == "__main__":
    main()
