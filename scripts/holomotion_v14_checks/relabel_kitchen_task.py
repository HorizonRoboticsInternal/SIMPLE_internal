"""Relabel recorded kitchen episodes with another task and re-score them on its gates (2026-10-27).

    .venv/bin/python scripts/holomotion_v14_checks/relabel_kitchen_task.py <level dir> [--task load_dishwasher] [--write]

User, 2026-10-27: "correct the task as loading dishwasher" -- the kitchen sessions were recorded under the default task of the
time (bottle_sink, "pick up the red bottle ..."), while the operator loaded the dishwasher.  For every episode with an exact
log this replays it bit-exactly with KITCHEN_TASK=<task> and evaluates that task's gates on every frame (gates read the
simulation only; physics is unchanged), then with --write rewrites, in <level dir> and its _filled copy:
  meta/tasks.jsonl            the task text
  meta/episodes.jsonl         "tasks", realformat.success / success_frame, and "task_eval" (the gates and their frames)
after copying both files to meta/_before_relabel_<date>/.  The per-frame task_index stays 0 (one task per dataset).
Run one dataset per process (the scene module reads its knobs at import).
"""
from __future__ import annotations

import argparse, json, os, shutil, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("level_dir"); ap.add_argument("--task", default="load_dishwasher"); ap.add_argument("--write", action="store_true")
a = ap.parse_args()
os.environ["KITCHEN_TASK"] = a.task                                       # before the scene module is imported
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import mujoco, numpy as np
from simple.cli.replay_holomotion_v14 import _build_env, _reset_to
from simple.teleop.holomotion_v14 import exact_log as XL

src = Path(a.level_dir).resolve(); dirs = [src] + ([src.with_name(src.name + "_filled")] if src.with_name(src.name + "_filled").exists() else [])
env = senv = mod = None; results = {}
for path in sorted((src / "replay").glob("episode_*.npz")):
    ep = int(path.stem.split("_")[-1]); log = XL.load(path); info, setup = log["info"], log["setup"]
    if env is None:
        env, senv, mod = _build_env(info, setup)
    else:
        _reset_to(env, senv, mod, info, setup)
    m, d, task = senv.mujoco.mjModel, senv.mujoco.mjData, senv.task
    if mod.TASK != a.task:
        sys.exit(f"scene module task {mod.TASK} != {a.task}")
    first = {}; last = {}; door = [0.0]
    def on_frame(t):
        P = task.update_gates({}, senv.mujoco)
        for g in ("door_open", "grasped", "placed", "rack_in", "door_closed"):
            if P.get(g) and g not in first:
                first[g] = t
        door[0] = max(door[0], float(P.get("door_deg", 0.0))); last.clear(); last.update(P)
    res = XL.replay_physics(m, d, log, on_frame=on_frame)
    results[ep] = dict(bit_exact=bool(res["exact"]), frames=int(res["frames"]), gates={g: (first.get(g) is not None) for g in ("door_open", "grasped", "placed", "rack_in", "door_closed")},
                       frames_first={g: first.get(g) for g in ("door_open", "grasped", "placed", "rack_in", "door_closed")}, max_loaded=int(last.get("max_loaded", 0)),
                       loaded_at_end=int(last.get("loaded", 0)), loaded_dishes_at_end=last.get("loaded_dishes", ""), max_door_deg=round(door[0], 1))
    r = results[ep]
    print(f"[relabel] {src.parts[-4]} ep {ep}: {r['frames']} frames, replay {'BIT-EXACT' if r['bit_exact'] else 'DIFFERS'} | door_open {r['gates']['door_open']} "
          f"(frame {first.get('door_open')}), grasped {r['gates']['grasped']} (frame {first.get('grasped')}), placed {r['gates']['placed']} (frame {first.get('placed')}), "
          f"rack_in {r['gates']['rack_in']} (frame {first.get('rack_in')}), door_closed {r['gates']['door_closed']} (frame {first.get('door_closed')}) | "
          f"door max {r['max_door_deg']} deg, dishes loaded max {r['max_loaded']}, at the end {r['loaded_at_end']} ({r['loaded_dishes_at_end']})", flush=True)
instr = mod.INSTRUCTIONS[a.task]
if a.write:
    stamp = time.strftime("%Y%m%d")
    for D in dirs:
        meta = D / "meta"; bak = meta / f"_before_relabel_{stamp}"; bak.mkdir(exist_ok=True)
        for f in ("tasks.jsonl", "episodes.jsonl"):
            if not (bak / f).exists():
                shutil.copy2(meta / f, bak / f)
        tasks = [json.loads(l) for l in open(meta / "tasks.jsonl") if l.strip()]
        old = [t["task"] for t in tasks]
        for t in tasks:
            t["task"] = instr
        (meta / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
        eps = [json.loads(l) for l in open(meta / "episodes.jsonl") if l.strip()]
        for e in eps:
            fill = json.loads(e["fill"]) if isinstance(e.get("fill"), str) else (e.get("fill") or {})
            se = int(fill.get("source_episode", e["episode_index"])); r = results.get(se)
            e["tasks"] = [instr]
            if r is None:
                continue
            rf = json.loads(e["realformat"]) if isinstance(e.get("realformat"), str) else (e.get("realformat") or {})
            rf["success"] = r["gates"]["placed"]; rf["success_frame"] = r["frames_first"]["placed"]; e["realformat"] = json.dumps(rf)
            e["task_eval"] = json.dumps(dict(task=a.task, instruction=instr, previous_task=old[0] if old else "", evaluated=time.strftime("%Y-%m-%d %H:%M"),
                                             **{k: r[k] for k in ("gates", "frames_first", "max_door_deg", "max_loaded", "loaded_at_end", "loaded_dishes_at_end", "bit_exact")}))
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
        print(f"[relabel] wrote {meta} (task '{old[0] if old else ''}' -> '{instr}'; originals in {bak.name}/)", flush=True)
else:
    print("[relabel] dry run: pass --write to rewrite the metadata", flush=True)
os._exit(0)
