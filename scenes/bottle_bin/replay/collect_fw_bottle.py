#!/usr/bin/env python
"""Best VALID physical bottle run per episode over the free-walk sweeps (valid_<tag>.json from valid_bottle.py) ->
jobs_bottlevid.json (one job per episode: bottle_xy, bin_xy, bin_yaw). Usage: collect_fw_bottle.py TAG [TAG ...]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; best = {}
for tag in sys.argv[1:]:
    f = HERE / f"valid_{tag}.json"
    if not f.exists(): continue
    for k, v in json.load(open(f)).items():
        if v["ok"] and (v["ep"] not in best or v["gap"] < best[v["ep"]]["gap"]): best[v["ep"]] = dict(v, tag=tag)
jobs = [{"ep": ep, "bottle_xy": b["job"]["bottle_xy_cmd"], "bin_xy": b["job"]["bin_xy_cmd"], "bin_yaw": b["job"]["bin_yaw_deg"], "round": b["tag"] + (f":{b['variant']}" if b.get("variant") else "")}
        for ep, b in sorted(best.items())]
json.dump(jobs, open(HERE / "jobs_bottlevid.json", "w"), indent=0); print(len(jobs), "valid bottle episodes -> jobs_bottlevid.json:", [j["ep"] for j in jobs])
