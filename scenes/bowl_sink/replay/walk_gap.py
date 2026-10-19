#!/usr/bin/env python
"""Bowl replay validity (2026-09-28): ok = bowl placed in the sink AND walk complete AND bowl untouched before the grasp.
walk complete: max distance between the placed run's base path and the free walk (counters far away) at the same rows <= 0.10 m.
pre-grasp: until 1 s (30 rows) before the first recorded right-hand closure the bowl moves <= 2 cm (xy and z) and tilts <= 20 deg."""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; tag = sys.argv[1]
res = {json.loads(l)["ep"]: json.loads(l) for l in open(HERE / f"fitbatch_{tag}.jsonl")}; out = {}
for ep, r in sorted(res.items()):
    gap = pre = None
    try:
        tk = [x for x in json.load(open(HERE / f"trace_ep{ep}_{tag}.json")) if "summary" not in x]
        pl = {x["row"]: np.array(x["sim_xy"]) for x in tk}
        fr = {x["row"]: np.array(x["sim_xy"]) for x in json.load(open(HERE / f"trace_ep{ep}_freewalk.json")) if "summary" not in x}
        common = sorted(set(pl) & set(fr)); gap = float(max(np.linalg.norm(pl[k] - fr[k]) for k in common))
        c0 = next((x["row"] for x in tk if x["key_real"]["right_hand_index_0_joint"] > 0.6), 10 ** 9)
        b0 = np.array(tk[0]["target_xyz"]); before = [x for x in tk if x["row"] < c0 - 30]
        mv = max([float(np.linalg.norm(np.array(x["target_xyz"]) - b0)) for x in before] or [0.0]); tl = max([x["target_tilt_deg"] for x in before] or [0.0])
        pre = {"ok": bool(mv <= 0.02 and tl <= 20.0), "moved": round(mv, 3), "tilt": round(float(tl), 1), "closure_row": int(c0)}
    except Exception as e:
        pre = {"ok": False, "error": str(e)[:80]}
    out[ep] = {"placed": bool(r.get("placed_any")), "gap": None if gap is None else round(gap, 3), "leg": r.get("leg_contact"), "pre_grasp": pre,
               "ok": bool(r.get("placed_any")) and gap is not None and gap <= 0.10 and bool(pre and pre.get("ok")), "params": r.get("params")}
json.dump(out, open(HERE / f"walkgap_{tag}.json", "w"), indent=1)
ok = [e for e, v in out.items() if v["ok"]]
print(f"{tag}: placed {sum(v['placed'] for v in out.values())}, walk complete {sum(1 for v in out.values() if v['gap'] is not None and v['gap'] <= 0.10)}, "
      f"bowl untouched before the grasp {sum(1 for v in out.values() if (v['pre_grasp'] or {}).get('ok'))}, valid {len(ok)} of {len(out)}: {sorted(ok)}")
