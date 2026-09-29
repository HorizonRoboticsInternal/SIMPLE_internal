#!/usr/bin/env python3
"""Replace see-through table materials in a scene set's states (in place): glass, gems, liquids, anything whose MDL transmits
light, and thin-walled (translucent) fabrics. Isaac's real-time renderer draws an object that rests on such a surface wrongly
(the bowl's inside showed the table; bottles looked hollow). Each replaced table gets an opaque material already used by another
scene of the same set (deterministic). Visual only: MuJoCo's physics ignores table materials.

    python fix_tables.py <episodes.jsonl> [--report out.json]
"""
import argparse, json, os, re

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])   # the repo root
_cache = {}


def see_through(path: str) -> bool:
    if path not in _cache:
        p = path if os.path.isabs(path) else os.path.join(ROOT, path)
        s = open(p, errors="ignore").read() if os.path.exists(p) else ""
        cat = path.split("vMaterials_2/")[1].split("/")[0] if "vMaterials_2/" in path else ""
        _cache[path] = (cat in ("Glass", "Gems", "Liquids") or bool(re.search(r"scatter_transmit|scatter_reflect_transmit", s))
                        or bool(re.search(r"thin_walled:\s*true", s)))
    return _cache[path]


ap = argparse.ArgumentParser(); ap.add_argument("episodes"); ap.add_argument("--report", default=None)
a = ap.parse_args()
lines = [l for l in open(a.episodes) if l.strip()]
rows, cfgs, nests = [], [], []
for l in lines:
    r = json.loads(l); c = r["environment_config"]; n = 0
    while isinstance(c, str):
        c = json.loads(c); n += 1
    rows.append(r); cfgs.append(c); nests.append(n)
mats = [(c["dr_state_dict"].get("material") or {}).get("table_material") for c in cfgs]
pool = []
for m in mats:
    if m and not see_through(m["path"]) and m not in pool:
        pool.append(m)
changed = []
for i, (c, m) in enumerate(zip(cfgs, mats)):
    if m and see_through(m["path"]):
        new = pool[(i * 7) % len(pool)]
        c["dr_state_dict"]["material"]["table_material"] = dict(new)
        s = c
        for _ in range(nests[i]):
            s = json.dumps(s)
        rows[i]["environment_config"] = s
        changed.append({"scene": i, "from": m["name"], "to": new["name"]})
open(a.episodes, "w").write("".join(json.dumps(r) + "\n" for r in rows))
if a.report:
    json.dump(changed, open(a.report, "w"), indent=1)
print(f"{a.episodes}: {len(changed)} see-through tables replaced: " + ", ".join(f"#{x['scene'] + 1} {x['from']} -> {x['to']}" for x in changed))
