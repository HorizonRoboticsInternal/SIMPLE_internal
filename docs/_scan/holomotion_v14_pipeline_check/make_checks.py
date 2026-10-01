#!/usr/bin/env python
"""checks.json for the pipeline-check page: the replay checks (from replay_<task>.log) and the repeat check (repeat_a vs
repeat_b: the same scene twice through the live model, compared on their exact logs and results)."""
import glob, json, re, sys
from pathlib import Path

import numpy as np

pipe = Path(sys.argv[1])
out = {"replay": {}, "repeat": {}}
for log in sorted(pipe.glob("replay_*.log")):
    task = log.stem[len("replay_"):]
    lines = [l.rstrip() for l in log.read_text().splitlines() if l.startswith("[replay] episode_")]
    n_ok = sum("BIT-EXACT" in l for l in lines)
    ident = all("model identical" in l for l in lines) if lines else False
    out["replay"][task] = {"ok": bool(lines) and n_ok == len(lines),
                           "summary": f"{n_ok}/{len(lines)} episodes bit-exact (action replay, model {'identical' if ident else 'DIFFERENT'})",
                           "detail": "\n".join(lines)}



def _lat(e):
    """mean model latency of the episode, whichever key the results carry"""
    for k in ("latency_mean_s", "latency_mean", "query_seconds_mean"):
        if k in e: return e[k]
    return "?"

def repeat_check():
    a = sorted(glob.glob(str(pipe / "repeat_a" / "**" / "replay" / "*.npz"), recursive=True))
    b = sorted(glob.glob(str(pipe / "repeat_b" / "**" / "replay" / "*.npz"), recursive=True))
    if not a or not b:
        return None
    za, zb = np.load(a[0], allow_pickle=True), np.load(b[0], allow_pickle=True)
    ra = json.loads(Path(a[0]).parents[1].joinpath("results.json").read_text())
    rb = json.loads(Path(b[0]).parents[1].joinpath("results.json").read_text())
    ea, eb = ra["episodes"][0], rb["episodes"][0]
    ia, ib = json.loads(str(za["info_json"])), json.loads(str(zb["info_json"]))
    same = lambda k: za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k])
    keys = ["state0", "physics", "ctrl"] + [k for k in za.files if k.startswith("action_") and k in zb.files and k not in ("action_kp", "action_kd")]
    lines = [f"{k}: {'identical' if same(k) else 'DIFFERENT'}" for k in keys]
    first_diff = None
    if not same("physics"):
        n = min(len(za["physics"]), len(zb["physics"]))
        d = np.where(~np.all(za["physics"][:n] == zb["physics"][:n], axis=1))[0]
        first_diff = int(d[0]) + 1 if len(d) else None
    lines += [f"model hash equal: {ia.get('model_sha256') == ib.get('model_sha256')}",
              f"engaged at step {ea.get('engaged_step')} / {eb.get('engaged_step')}",
              f"model latency mean {_lat(ea)} / {_lat(eb)} s"]
    secs = ea.get("sim_seconds", "?"); q = ea.get("queries", "?")
    identical = all(same(k) for k in keys)
    summary = (f"{ia['scene']} scene 0 run twice through the live model ({secs} s, {q} queries each): "
               + ("the rollouts are identical" if identical else
                  f"the rollouts differ from frame {first_diff}: the model's replies are not deterministic (the chain up to the first reply is)"))
    return {"ok": identical, "summary": summary, "detail": "\n".join(lines)}


r = repeat_check()
if r:
    out["repeat"] = r
(pipe / "checks.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v.get("summary") if isinstance(v, dict) and "summary" in v else {t: x["summary"] for t, x in v.items()}) for k, v in out.items()}, indent=1))
