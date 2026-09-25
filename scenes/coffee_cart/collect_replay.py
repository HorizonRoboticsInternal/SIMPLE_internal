#!/usr/bin/env python3
"""Gather the replay sweep + all-episode baseline into replay/replay_results.json (what make_report.py's replay section reads).

    python collect_replay.py --baseline-tag rc020_rt280_tl040 [--video replay_ep84_show.mp4]
Sweep rows = every rc020_* tag with >= 6 episodes; baseline = the tag run over the whole set.
"""
import argparse, glob, json, re, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RP = HERE / "replay"
TABLE_H, CART_TOP = 0.74, 0.80


def classify(s):
    z = s["cup_end"][2]; tilt = s["cup_end_tilt_deg"]; x, y = s["cup_end"][:2]
    in_fp = (s["table_near_x"] - 0.02 <= x <= s["table_near_x"] + 0.62) and abs(y - s["table_centre"][1]) <= 0.55
    if s["on_table"]: return "on_table"
    if in_fp and abs(z - (TABLE_H + 0.045)) < 0.025 and tilt > 45: return "tipped"
    if abs(z - CART_TOP) < 0.03: return "on_cart"
    if z < 0.2: return "floor"
    return "other"


def summarise(rs, tag):
    c = collections.Counter(classify(s) for s in rs)
    pr = lambda s: s.get("task_progress") or {}
    return dict(tag=tag, n=len(rs), route_forward=rs[0]["route_forward"], table_from_line=rs[0]["table_from_line"],
                post_turn_gain=rs[0]["scene"].get("post_turn_gain", 1.0),
                task_success=sum(bool(s.get("task_success", False)) for s in rs),               # the task's own checker
                task_cart_pushed=sum(bool(pr(s).get("cart_pushed_ever", False)) for s in rs),
                task_cup_lifted=sum(bool(pr(s).get("cup_lifted_ever", False)) for s in rs),
                has_task_verdict=all("task_success" in s for s in rs),
                on_table=c["on_table"], tipped=c["tipped"],
                lifted=sum(s["max_lift_m"] > 0.05 for s in rs), floor=c["floor"], on_cart=c["on_cart"])


def release_points(tag):
    pts = []
    for f in glob.glob(str(RP / f"trace_ep*_{tag}.json")):
        tr = [r for r in json.load(open(f)) if "summary" not in r]
        z = np.array([r["target_xyz"][2] for r in tr]); lift = z - z[0]
        if lift.max() < 0.05: continue
        k = int(lift.argmax())
        for j in range(k, len(z) - 5):
            if z[j] - z[j + 5] > 0.05:
                pts.append(tr[j]["target_xyz"][:2]); break
    return np.array(pts) if pts else np.zeros((0, 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--baseline-tag", required=True); ap.add_argument("--video", default=None)
    ap.add_argument("--video-caption", default="episode 84, real contact: the robot grips the bar, pushes the cart, turns, and puts the cup on the table")
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for f in glob.glob(str(RP / "summary_ep*_rc020_*.json")):
        tag = re.search(r"_ep\d+_(rc020_.+)\.json", f.name if hasattr(f, "name") else f)[1]
        groups[tag].append(json.load(open(f)))
    sweep = sorted((summarise(rs, t) for t, rs in groups.items() if len(rs) >= 6),
                   key=lambda r: (r["route_forward"], r["table_from_line"], r["post_turn_gain"]))
    base_rs = groups[a.baseline_tag]; baseline = summarise(base_rs, a.baseline_tag)
    # "best" = the configuration with the highest completion rate, judged on the full set where one exists: an 8-episode
    # row beating a 96-episode row by one cup is sampling noise, not a better table position
    full = [r for r in sweep if r["n"] >= 50]
    pool = full if full else sweep
    best = max(pool, key=lambda r: ((r["task_success"] if r["has_task_verdict"] else r["on_table"]) / r["n"], r["tipped"] / r["n"], r["lifted"] / r["n"]))
    cart = np.array([s["cart_end_xy"][0] for s in base_rs]); head = np.array([abs(s["sim_heading_end"] - s["cmd_heading_end"]) for s in base_rs])
    P = release_points(a.baseline_tag)
    out = dict(baseline=baseline, best=best, sweep=sweep,
               push=dict(n=len(base_rs), pushed=int((cart > 1.0).sum()), cart_travel_mean=float(cart.mean()),
                         cart_travel_p10=float(np.percentile(cart, 10)), cart_travel_p90=float(np.percentile(cart, 90)),
                         heading_err_deg=float(np.degrees(head.mean()))),
               release=dict(n=int(len(P)), x_p10=float(np.percentile(P[:, 0], 10)) if len(P) else None, x_p90=float(np.percentile(P[:, 0], 90)) if len(P) else None,
                            x_med=float(np.median(P[:, 0])) if len(P) else None, y_med=float(np.median(P[:, 1])) if len(P) else None),
               video=a.video, video_caption=a.video_caption)
    (RP / "replay_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("baseline", "best", "push", "release")}, indent=1))
    print(f"\n{'route':>6} {'tfl':>5} {'gain':>5} {'n':>3} | {'TASK ok':>7} {'pushed':>6} {'lifted':>6} | {'geom on_table':>13} {'tipped':>6} {'floor':>5} {'cart':>4}")
    for r in sweep:
        ts = r["task_success"] if r["has_task_verdict"] else "-"
        print(f"{r['route_forward']:>6} {r['table_from_line']:>5} {r['post_turn_gain']:>5} {r['n']:>3} | {str(ts):>7} {r['task_cart_pushed']:>6} {r['task_cup_lifted']:>6} | {r['on_table']:>13} {r['tipped']:>6} {r['floor']:>5} {r['on_cart']:>4}")


if __name__ == "__main__":
    main()
