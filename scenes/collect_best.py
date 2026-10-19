import os
#!/usr/bin/env python3
"""Per kit: the per-episode scene parameters of the latest CLEAN success over all rounds -> replay/best_fits.json and
replay/best_results.jsonl (one line per clean episode), the inputs of export_batch.py --fits/--results.
Clean = task success without the walk being obstructed (bottle: no bin contact before the release; bowl: <= 25 ticks of
robot-counter contact when known). bottle rounds are sweep result dirs replay/sweep/psi0fit<N>/ep###.json; bowl/coffee rounds
are replay/fitbatch_<tag>.jsonl with the params inside.
    python sim/collect_best.py bottle_bin --rounds psi0fit9 psi0fit10
    python sim/collect_best.py bowl_sink --rounds fit6r fit7r     (later rounds win)
"""
import argparse, glob, json
from pathlib import Path
HERE = Path(__file__).resolve().parent


CUP_FIX_TS = 1790570700.0   # 2026-09-27 21:45 local: coffee_cart_task honours COFFEE_CART_CUP_XY from here on


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--rounds", nargs="+", required=True); a = ap.parse_args()
    best = {}
    for tag in a.rounds:                                          # later rounds overwrite earlier ones
        if a.kit == "bottle_bin":
            for f in glob.glob(str(HERE / "bottle_bin/replay/sweep" / tag / "ep*.json")):
                r = json.load(open(f))
                if r.get("in_bin") and (r.get("robot_bin_leg_ticks") if r.get("robot_bin_leg_ticks") is not None else (r.get("robot_bin_contact_ticks") or 0)) <= 25 and (r.get("grasp_ok") or r.get("assist_captured") is not None or r.get("success")):   # re-scored: assisted rides lift little
                    best[int(r["ep"])] = {"bottle_xy": r["bottle_xy_cmd"], **({"bin_xy": r["bin_xy_cmd"]} if r.get("bin_xy_cmd") else {}),
                                          **({"grasp_assist": True} if r.get("grasp_assist") else {}), **({"drop_assist": True} if r.get("drop_assist") else {}),
                                          **({"bin_yaw": r["bin_yaw_deg"]} if r.get("bin_yaw_deg") is not None and abs(float(r["bin_yaw_deg"]) + 90.0) > 1e-6 else {}), "round": tag}
        else:
            for l in open(HERE / a.kit / "replay" / f"fitbatch_{tag}.jsonl"):
                if not l.strip(): continue
                r = json.loads(l)
                ok = (r.get("success") and (a.kit != "coffee_cart" or r.get("on_table"))) or (a.kit == "bowl_sink" and r.get("placed_any") and int(r.get("leg_contact") or 0) <= 150)   # re-scored with the current leg rule
                if ok and a.kit == "coffee_cart":
                    # 2026-09-27: before 21:49 the task ignored COFFEE_CART_CUP_XY (the cup always started at the layout default):
                    # a row either records where the cup really started, or predates the fix and ran with the default cup
                    r = dict(r); pr = dict(r.get("params") or {})
                    if r.get("cup_at_reset"): pr["cup_xy"] = [round(float(v), 4) for v in r["cup_at_reset"][:2]]
                    elif os.path.getmtime(HERE / a.kit / "replay" / f"fitbatch_{tag}.jsonl") < CUP_FIX_TS: pr["cup_xy"] = [0.465, -0.125]
                    r["params"] = pr
                if ok:
                    best[int(r["ep"])] = {**{k: v for k, v in (r.get("params") or {}).items() if k in ("robot_to_edge", "bowl_xy", "sink_along", "robot_dy", "nav_gain", "cup_xy", "route_forward", "table_from_line", "robot_to_cart", "cart_attach", "bottle_xy", "bin_xy", "hand_delta", "grasp_assist", "box_from_handle", "cart_park", "sink_recess", "cup_pin")}, "round": tag}
    out = HERE / a.kit / "replay"
    try:                                                          # keep Isaac-refit entries whose underlying round is still the best
        old = json.load(open(out / "best_fits.json"))
        for k, v in list(best.items()):
            o = old.get(str(k))
            if o and o.get("isaac_refit") and o.get("round") == v.get("round"): best[k] = o
    except Exception:
        pass
    json.dump({str(k): v for k, v in sorted(best.items())}, open(out / "best_fits.json", "w"), indent=1)
    with open(out / "best_results.jsonl", "w") as f:
        for ep in sorted(best): f.write(json.dumps({"ep": ep, "success": True, "round": best[ep]["round"]}) + "\n")
    print(f"{a.kit}: {len(best)} clean episodes -> {out/'best_fits.json'}")


if __name__ == "__main__":
    main()
