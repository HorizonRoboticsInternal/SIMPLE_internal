#!/usr/bin/env python
"""Bowl validity with a CARRY TWIN (2026-09-28): the empty-handed free walk cannot separate carry drift from a block late in
the walk, so: before the first recorded closure the path must stay within 0.10 m of the free walk (the bowl counter does not
block the approach); after it, within 0.10 m of the twin = the same run with the sink counter 20 m away (BOWL_SINK_SINK_FAR:
the bowl is grasped and carried the same way); legs on the counters <= 25 ticks; placed; bowl untouched before the grasp.
Usage: walk_gap_twin.py TAG TWINTAG -> walkgap_<TAG>.json (same format as walk_gap.py)"""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; tag, twin = sys.argv[1], sys.argv[2]
res = {json.loads(l)["ep"]: json.loads(l) for l in open(HERE / f"fitbatch_{tag}.jsonl")}; out = {}
xy = lambda f: {x["row"]: np.array(x["sim_xy"]) for x in json.load(open(f)) if "summary" not in x}
for ep, r in sorted(res.items()):
    try:
        tk = [x for x in json.load(open(HERE / f"trace_ep{ep}_{tag}.json")) if "summary" not in x]; pl = {x["row"]: np.array(x["sim_xy"]) for x in tk}
        fr, tw = xy(HERE / f"trace_ep{ep}_freewalk.json"), xy(HERE / f"trace_ep{ep}_{twin}.json")
        c0 = next((x["row"] for x in tk if x["key_real"]["right_hand_index_0_joint"] > 0.6), 10 ** 9)
        g_pre = max([float(np.linalg.norm(pl[k] - fr[k])) for k in pl if k in fr and k < c0] or [0.0])
        g_post = max([float(np.linalg.norm(pl[k] - tw[k])) for k in pl if k in tw and k >= c0] or [9.0])
        g_free = max(float(np.linalg.norm(pl[k] - fr[k])) for k in pl if k in fr)
        b0 = np.array(tk[0]["target_xyz"]); before = [x for x in tk if x["row"] < c0 - 30]
        mv = max([float(np.linalg.norm(np.array(x["target_xyz"]) - b0)) for x in before] or [0.0]); tl = max([x["target_tilt_deg"] for x in before] or [0.0])
        pre = {"ok": bool(mv <= 0.02 and tl <= 20.0), "moved": round(mv, 3), "tilt": round(float(tl), 1), "closure_row": int(c0)}
    except Exception as e:
        g_pre = g_post = g_free = None; pre = {"ok": False, "error": str(e)[:80]}
    leg = r.get("leg_contact") or 0
    ok = bool(r.get("placed_any")) and g_pre is not None and g_pre <= 0.10 and g_post <= 0.10 and leg <= 25 and bool(pre.get("ok"))
    out[ep] = {"placed": bool(r.get("placed_any")), "gap": None if g_post is None else round(max(g_pre, g_post), 3), "gap_pre_free": None if g_pre is None else round(g_pre, 3),
               "gap_post_twin": None if g_post is None else round(g_post, 3), "gap_free": None if g_free is None else round(g_free, 3), "leg": leg, "pre_grasp": pre, "ok": ok, "twin": True, "params": r.get("params")}
json.dump(out, open(HERE / f"walkgap_{tag}.json", "w"), indent=1)
print(f"{tag} vs twin {twin}: " + "; ".join(f"ep {e}: placed {v['placed']}, approach gap {v['gap_pre_free']}, after-grasp gap to twin {v['gap_post_twin']} (free walk {v['gap_free']}), legs {v['leg']}, ok {v['ok']}" for e, v in out.items()))
