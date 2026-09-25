#!/usr/bin/env python
"""Run SIMPLE's eval_decoupled_wbc on a scene kit's task.

    python sim/eval_scene.py bowl_sink simple/G1WholebodyBowlSinkTeleop-v0 psi0_decoupled_wbc train \
        --data-format lerobot --data-dir data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-3 --port 21000 --headless --num-episodes 20

Registers the kit's task first, and re-applies the env vars stored in <data-dir>/meta/scene_env.json
(make_levels.py writes them) so a level-3 set is evaluated on the layout it was generated with.
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
kit = sys.argv[1]; rest = sys.argv[2:]
if "--data-dir" in rest:
    meta = Path(rest[rest.index("--data-dir") + 1]) / "meta" / "scene_env.json"
    if meta.exists():
        for k, v in json.load(open(meta)).get("env", {}).items():
            os.environ.setdefault(k, v)
        print(f"[eval_scene] applied {meta}: " + " ".join(f"{k}={v}" for k, v in json.load(open(meta)).get("env", {}).items()), flush=True)
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, str(HERE / kit))
__import__(f"{kit}_task")
sys.argv = [sys.argv[0]] + rest
from simple.cli.eval_decoupled_wbc import typer_main
typer_main()
