#!/usr/bin/env python3
"""Fill replay_meta.json 'result' (gates) of already-exported bowl_sink episodes from the export logs (the first export pass saved
the meta inside env.close, before the kit returned its summary)."""
import os
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
LOGS = Path(os.environ.get("REPLAY_LOG_DIR") or DATA / "replay_logs")   # run logs / per-run videos (not tracked)
OUT = DATA / "replay_datasets"; L = LOGS
kit = sys.argv[1] if len(sys.argv) > 1 else "bowl_sink"; n_fix = n_placed = 0
for d in sorted((OUT / kit).glob("ep_*")):
    f = d / "replay_meta.json"
    if not f.exists(): continue
    m = json.load(open(f)); ep = m["episode"]
    if (m.get("result") or {}).get("gates"): continue
    log = L / f"export_{kit}" / f"ep{ep}.log"
    if not log.exists(): continue
    txt = log.read_text(errors="ignore"); g = re.findall(r"gates: ([^\n]{0,200})", txt)
    if not g: continue
    line = g[-1]; gates = {}
    for k in ("at_bowl", "grasped", "at_basin", "placed"):
        mm = re.search(rf"{k}@([0-9.]+)s", line); gates[k] = mm is not None; gates[f"t_{k}"] = float(mm.group(1)) if mm else None
    st = re.search(r"stalled at (\w+)", line); gates["stalled_at"] = None if (st is None or st.group(1) == "None") else st.group(1)
    lift = re.search(r"lift ([0-9.]+) m", line); gates["max_lift"] = float(lift.group(1)) if lift else None
    m["result"] = {"gates": gates, "from_log": True}; json.dump(m, open(f, "w"), indent=1, default=str); n_fix += 1; n_placed += int(gates["placed"])
print(f"{kit}: repaired {n_fix} metas, placed {n_placed}")
