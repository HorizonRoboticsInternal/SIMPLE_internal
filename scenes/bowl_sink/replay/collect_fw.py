#!/usr/bin/env python
"""Best VALID physical bowl run per episode (placed + walk complete + untouched before the grasp) over the free-walk rounds ->
phys_fw_best.json {ep: fit params + round}. Usage: collect_fw.py TAG [TAG ...]"""
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
json.dump(out, open(HERE / "phys_fw_best.json", "w"), indent=1); print(len(out), "valid bowl episodes ->", "phys_fw_best.json:", sorted(map(int, out)))
