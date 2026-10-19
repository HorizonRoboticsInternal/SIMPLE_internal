#!/usr/bin/env python
"""Best VALID physical coffee run per episode (cup on the desk + untouched before the grasp + walk within 0.10 m of its free
twin) over the free-walk rounds -> phys_fw_best.json. Usage: collect_fw.py TAG [TAG ...] (walkgap_<TAG>.json)"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; best = {}
for tag in sys.argv[1:]:
    f = HERE / f"walkgap_{tag}.json"
    if not f.exists(): continue
    for ep, v in json.load(open(f)).items():
        if v.get("ok") and v.get("params") and (ep not in best or v["gap"] < best[ep]["gap"]):
            best[ep] = {"gap": v["gap"], "params": dict(v["params"], round=tag)}
out = {ep: b["params"] for ep, b in sorted(best.items(), key=lambda x: int(x[0]))}
json.dump(out, open(HERE / "phys_fw_best.json", "w"), indent=1); print(len(out), "valid coffee episodes -> phys_fw_best.json:", sorted(map(int, out)))
