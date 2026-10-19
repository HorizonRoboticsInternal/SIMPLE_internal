#!/usr/bin/env python
"""Coffee walk completeness: max distance between a run's base path and its free twin (same fit, table 12 m ahead) at the same
rows. Usage: walk_gap_coffee.py TAG FREETAG [--eps ...] -> walkgap_<TAG>.json; ok = fit_batch success (cup on the table, untouched
before the grasp) AND gap <= 0.10 m."""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; tag, free = sys.argv[1], sys.argv[2]
res = {json.loads(l)["ep"]: json.loads(l) for l in open(HERE / f"fitbatch_{tag}.jsonl")}; out = {}
for ep, r in sorted(res.items()):
    gap = at = None
    try:
        pl = {x["row"]: np.array(x["sim_xy"]) for x in json.load(open(HERE / f"trace_ep{ep}_{tag}.json")) if "row" in x}
        fr = {x["row"]: np.array(x["sim_xy"]) for x in json.load(open(HERE / f"trace_ep{ep}_{free}.json")) if "row" in x}
        ks = sorted(set(pl) & set(fr)); d = [float(np.linalg.norm(pl[k] - fr[k])) for k in ks]; i = int(np.argmax(d)); gap, at = round(d[i], 3), ks[i]
    except Exception as e:
        pass
    out[ep] = {"success": bool(r.get("success")), "gap": gap, "gap_row": at, "ok": bool(r.get("success")) and gap is not None and gap <= 0.10, "params": r.get("params")}
json.dump(out, open(HERE / f"walkgap_{tag}.json", "w"), indent=1)
print(f"{tag} vs {free}: success {sum(v['success'] for v in out.values())}, walk complete {sum(1 for v in out.values() if v['gap'] is not None and v['gap'] <= 0.10)}, "
      f"valid {sum(v['ok'] for v in out.values())} of {len(out)}; blocked successes: {sorted((e, v['gap']) for e, v in out.items() if v['success'] and not v['ok'])}")
