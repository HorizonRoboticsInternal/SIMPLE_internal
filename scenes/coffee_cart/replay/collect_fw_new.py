#!/usr/bin/env python
"""Best valid coffee run per episode over TAGS for episodes NOT already valid in the coffeevid rerun -> phys_fw_best2.json"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; have = {e for e, v in json.load(open(HERE / "walkgap_coffeevid.json")).items() if v["ok"]}; best = {}
for tag in sys.argv[1:]:
    f = HERE / f"walkgap_{tag}.json"
    if not f.exists(): continue
    for ep, v in json.load(open(f)).items():
        if ep not in have and v.get("ok") and v.get("params") and (ep not in best or v["gap"] < best[ep]["gap"]): best[ep] = {"gap": v["gap"], "params": dict(v["params"], round=tag)}
out = {ep: b["params"] for ep, b in sorted(best.items(), key=lambda x: int(x[0]))}
json.dump(out, open(HERE / "phys_fw_best2.json", "w"), indent=1); print(len(out), "new valid coffee episodes -> phys_fw_best2.json:", sorted(map(int, out)))
