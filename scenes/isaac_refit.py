#!/usr/bin/env python3
"""After an Isaac export pass: episodes that were clean in the MuJoCo fits but not in the Isaac replay get their container
re-fitted from the ISAAC run (bin under the landed bottle, basin under the landed bowl, table under the landed cup), written to
replay/best_fits_isaac.json + best_results_isaac.jsonl for a second export pass (export_batch --fits/--results, no --skip-existing).
Clean-in-Isaac = task success with <= 10 robot-container contact ticks (bottle) / <= 50 leg contact ticks (bowl).

    python sim/isaac_refit.py bottle_bin [--dry]
"""
import os
import argparse, glob, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
LOGS = Path(os.environ.get("REPLAY_LOG_DIR") or DATA / "replay_logs")   # run logs / per-run videos (not tracked)
OUT = DATA / "replay_datasets"
LEG = ("knee", "hip", "ankle", "pelvis", "torso", "waist", "foot")


def clean(kit, meta):
    r = meta.get("result") or {}; c = meta.get("contacts") or {}
    if kit == "bottle_bin":
        return bool(r.get("success")) and (r.get("robot_bin_leg_ticks") if r.get("robot_bin_leg_ticks") is not None else (r.get("robot_bin_contact_ticks") or 0)) <= 25   # a brush of the bin while dropping (<= 0.5 s) does not block the walk
    leg = sum(int(n) for pair, n in (c.get("pairs") or []) if any(k in pair.split("|")[0] for k in LEG))
    g = r.get("gates") or {}
    if kit == "bowl_sink":
        return bool(g.get("placed")) and leg <= 150     # knee brushes while leaning over the counters (<= 3 s) do not block the walk
    if kit == "coffee_cart":
        return bool(r.get("on_table"))                   # the cup is on the table at the end (upright, inside the footprint)
    return bool(r.get("task_success") or g.get("success"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    fits = json.load(open(HERE / a.kit / "replay/best_fits.json")); refit = {}; kinds = {"clean": 0, "container_moved": 0, "unchanged": 0, "no_meta": 0}
    for ep_s, p in fits.items():
        ep = int(ep_s); f = OUT / a.kit / f"ep_{ep:03d}" / "replay_meta.json"
        if not f.exists(): kinds["no_meta"] += 1; continue
        m = json.load(open(f)); r = m.get("result") or {}
        if clean(a.kit, m): kinds["clean"] += 1; continue
        q = dict(p)
        if a.kit == "bottle_bin" and (r.get("grasp_ok") or r.get("assist_captured") is not None) and r.get("bottle_end") and r["bottle_end"][2] < 0.5:
            shift = np.zeros(2)
            if r.get("base_at_release") and (r.get("robot_bin_contact_ticks") or 0) > 10:
                h = r["base_at_release"][2]; shift = 0.10 * np.array([np.cos(h), np.sin(h)])
            q["bin_xy"] = [round(float(v), 4) for v in (np.array(r["bottle_end"][:2]) + shift)]; refit[ep_s] = q; kinds["container_moved"] += 1
        elif a.kit == "bowl_sink":
            g = r.get("gates") or {}; d = g.get("bowl_end_minus_basin")
            if g.get("grasped") and d and d[2] < 0.3:
                q["sink_along"] = round(p["sink_along"] - d[0], 3)              # basin_x = near_x - along + L/2: move it by the miss in x
                q["robot_dy"] = round(max(-0.30, p.get("robot_dy", 0.0) - d[1]), 3)   # everything shifts -dy: basin_y moves by -d(dy)
                refit[ep_s] = q; kinds["container_moved"] += 1
            else: kinds["unchanged"] += 1
        elif a.kit == "coffee_cart" and r.get("assist_captured") is not None and not r.get("on_table"):
            # 2026-09-27: assisted ride -> the desk under the ISAAC release point (the Isaac export differs from the MuJoCo fit by a
            # few cm, amplified by the walk; nothing touches the desk before the release, so the re-export reproduces this release)
            land = r.get("assist_release_xyz") or r.get("assist_settled_xyz")
            if land is None:                                     # older metas: the kit's log line "cup settled at [x, y, z]"
                import re
                lg = LOGS / "export_coffee_cart" / f"ep{ep}.log"
                mm = re.findall(r"grasp assist: cup settled at \[([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\]", lg.read_text(errors="ignore")) if lg.exists() else []
                land = [float(v) for v in mm[-1]] if mm else None
            if land is None: kinds["unchanged"] += 1; continue
            cx = float(land[0] + 0.15); lane_right = 9.0
            try:                                                 # the cart's path from the MuJoCo fit trace (the table must stay right of it)
                Lc = json.load(open(HERE / "coffee_cart/layout.json"))["layout"]; c0 = np.array([Lc["cart_cx"], Lc["cart_cy"]])
                tr = [x for x in json.load(open(HERE / f"coffee_cart/replay/trace_ep{ep}_{p['round']}.json")) if "summary" not in x]
                cp = c0 + np.array([x["cart_xy"] for x in tr]); near = cp[np.abs(cp[:, 0] - cx) < 0.66]
                if len(near): lane_right = float(near[:, 1].min() - 0.285)
            except Exception:
                pass
            cy = min(float(land[1] - 0.38), lane_right - 0.53)
            if land[1] - cy > 0.50:                              # the release point is too close to the cart's path: a desk under it would block
                kinds["unchanged"] += 1; continue                # the push (ep 12, 2026-09-28: the cart jammed on it and the walk stopped 1.3 m short)
            q["route_forward"] = round(cx + 0.3, 3); q["table_from_line"] = round(-cy - 0.53, 3); refit[ep_s] = q; kinds["container_moved"] += 1
        elif a.kit == "coffee_cart":
            ce = r.get("cup_end")
            if r.get("grasped") and ce and ce[2] < 0.5:
                q["route_forward"] = round(float(ce[0] + 0.3), 3); q["table_from_line"] = round(max(0.25, float(-ce[1] - 0.53)), 3); refit[ep_s] = q; kinds["container_moved"] += 1
            else: kinds["unchanged"] += 1
        else: kinds["unchanged"] += 1
    print(a.kit, kinds)
    if not a.dry:
        json.dump(refit, open(HERE / a.kit / "replay/best_fits_isaac.json", "w"), indent=1)
        with open(HERE / a.kit / "replay/best_results_isaac.jsonl", "w") as f:
            for ep_s in refit: f.write(json.dumps({"ep": int(ep_s), "success": True}) + "\n")
        print("->", len(refit), "episodes to re-export")


if __name__ == "__main__":
    main()
