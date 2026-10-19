#!/usr/bin/env python
"""Bowl: valid (walk_gap.py: placed + walk complete + untouched before the grasp) AND at the real end the hand is out of the bowl
(phys_render.bowl_end_ok). Usage: bowl_end_check.py TAG [TAG ...] -> prints per episode; best passing params -> fit_bowlvid5.json"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent.parent)); from phys_render import bowl_end_ok
best = {}
for tag in sys.argv[1:]:
    W = json.load(open(HERE / f"walkgap_{tag}.json"))
    for e, v in W.items():
        end = bowl_end_ok(json.load(open(HERE / f"trace_ep{e}_{tag}.json")), int(e)) if (HERE / f"trace_ep{e}_{tag}.json").exists() else False
        ok = bool(v["ok"] and end); print(tag, e, "placed", v["placed"], "gap", v["gap"], "end ok", end, "->", "VALID" if ok else "")
        if ok and (e not in best or v["gap"] < best[e]["gap"]): best[e] = {"gap": v["gap"], "p": dict(v["params"], round=tag)}
json.dump({e: b["p"] for e, b in best.items()}, open(HERE / "fit_bowlvid5.json", "w"), indent=1); print("valid:", sorted(best))
