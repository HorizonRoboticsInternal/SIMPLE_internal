#!/usr/bin/env python
"""data/kitchen_plates_v2 (2026-11-06, user: "create some new plates reference this one [a white wide-rim pasta plate]; add to
the V2 randomization"): the lift set's 50 plates, unchanged and in the lift manifest's order, plus the pasta-family plates
(make_plates.py --family pasta) that pass the three plate checks, appended after them.

Index k < 50 is the same plate as in the lift set, so the A1 / A2 bank starts keep their recorded plate types, GRASPABLE's lift
indices stay valid, and a recorded episode's setup["plate"] / ["plate_b"] mean the same plate in both sets.  The lift set itself is
not touched (its episodes replay as recorded; a set's N changes the kit's plate_b = plate + 25 mod N for the slices that do not
draw plate B on their own).
    .venv/bin/python scripts/kitchen_plates/make_set_v2.py --pasta <candidates dir> --checks <dir with under_rim.json, pinch.json, rack_fit.json> \
        [--n-pasta 20] [--lift data/kitchen_plates_lift] [--lift-checks docs/kitchen_plates_lift] [--out data/kitchen_plates_v2] [--docs docs/kitchen_plates_v2]
Writes the plate directories, manifest.json (foot_spheres 4, family per plate, the sources) and the merged check results
(docs/<set>/{under_rim,pinch,rack_fit}.json, the lift results + the kept pasta results).
"""
import argparse, json, shutil
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--pasta", required=True); ap.add_argument("--checks", required=True); ap.add_argument("--n-pasta", type=int, default=20)
ap.add_argument("--lift", default="data/kitchen_plates_lift"); ap.add_argument("--lift-checks", default="docs/kitchen_plates_lift"); ap.add_argument("--out", default="data/kitchen_plates_v2")
ap.add_argument("--docs", default="docs/kitchen_plates_v2"); ap.add_argument("--made", default="2026-11-06"); ap.add_argument("--force", action="store_true", help="replace an existing --out")
a = ap.parse_args(); lift, pasta, checks, out, docs = Path(a.lift), Path(a.pasta), Path(a.checks), Path(a.out), Path(a.docs)
if out.exists():
    if not a.force:
        raise SystemExit(f"{out} exists (--force replaces it)")
    shutil.rmtree(out)
out.mkdir(parents=True); docs.mkdir(parents=True, exist_ok=True)
lman = json.loads((lift / "manifest.json").read_text()); pman = json.loads((pasta / "manifest.json").read_text())
res = {k: json.loads((checks / f"{k}.json").read_text()) for k in ("under_rim", "pinch", "rack_fit")}
lres = {k: json.loads((Path(a.lift_checks) / f"{k}.json").read_text()) for k in ("under_rim", "pinch", "rack_fit")}
by = {k: {r["id"]: r for r in v["plates"]} for k, v in res.items()}
ok_key = {"under_rim": "ok", "pinch": "ok", "rack_fit": "stands"}
passing = [p["id"] for p in pman["plates"] if all(by[k].get(p["id"], {}).get(ok_key[k]) for k in by)]
kept = passing[:a.n_pasta]
print(f"[v2] pasta candidates {len(pman['plates'])}: pass all three {len(passing)} -> keeping {len(kept)}: {kept}")
plates = []
for p in lman["plates"]:
    shutil.copytree(lift / p["id"], out / p["id"]); plates.append(dict(p, family="dinner"))
for p in pman["plates"]:
    if p["id"] in kept:
        shutil.copytree(pasta / p["id"], out / p["id"]); plates.append(dict(p, family="pasta"))
man = dict(made=a.made, n=len(plates), foot_spheres=4, plates=plates,
           note=(f"v2 = the lift set's {len(lman['plates'])} plates (data/kitchen_plates_lift, same order and ids) + {len(kept)} wide-rim pasta plates "
                 f"(make_plates.py --family pasta, seed {pman['seed']}, {len(pman['plates'])} candidates of which {len(passing)} pass check_under_rim, check_pinch and check_rack_fit; the first {len(kept)} kept). "
                 "Index k < 50 is the lift set's plate k."),
           sources=dict(lift=dict(dir=str(lift), n=len(lman["plates"]), seed=lman.get("seed"), note=lman.get("note")),
                        pasta=dict(seed=pman["seed"], n_candidates=len(pman["plates"]), min_clearance_mm=pman.get("min_clearance_mm"), min_lip_mm=pman.get("min_lip_mm"), passing=passing, kept=kept)))
(out / "manifest.json").write_text(json.dumps(man, indent=1))
for k in ("under_rim", "pinch", "rack_fit"):
    lp = {r["id"]: r for r in lres[k]["plates"]}
    merged = [lp[p["id"]] for p in lman["plates"] if p["id"] in lp] + [by[k][i] for i in kept if i in by[k]]
    d = dict(lres[k]); d["plates"] = merged; d["n"] = len(merged)
    if k == "rack_fit":
        d["stands"] = sum(bool(r["stands"]) for r in merged); d["slot_fit"] = sum(bool(r.get("slot_fit")) for r in merged)
    else:
        d["ok"] = sum(bool(r["ok"]) for r in merged)
        if k == "under_rim":
            d["fingers_ok"] = sum(bool(r["fingers"].get("ok")) for r in merged); d["thumb_ok"] = sum(bool(r["thumb"].get("ok")) for r in merged)
    d["made"] = a.made; d["sets"] = {"lift": str(Path(a.lift_checks) / f"{k}.json"), "pasta": str(checks / f"{k}.json")}
    (docs / f"{k}.json").write_text(json.dumps(d, indent=1))
for f in sorted(checks.glob("*.jpg")):                                       # the check renders of the kept pasta plates
    if any(i in f.name for i in kept):
        shutil.copy(f, docs / f.name)
print(f"[v2] {len(plates)} plates -> {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.0f} MB); checks -> {docs}")
