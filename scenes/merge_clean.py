#!/usr/bin/env python3
"""Merge every exported episode of a kit whose Isaac replay was clean (isaac_refit.clean) into data/replay_datasets/<kit>_replay_dataset."""
import os
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from isaac_refit import clean, OUT
kit = sys.argv[1]; eps = []
for d in sorted((OUT / kit).glob("ep_*")):
    f = d / "replay_meta.json"
    if f.exists() and (d / "dataset/meta/info.json").exists() and clean(kit, json.load(open(f))): eps.append(str(d / "dataset"))
print(f"[merge] {kit}: {len(eps)} clean Isaac episodes")
if eps:
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "merge_lerobot.py"), "--out", str(OUT / f"{kit}_replay_dataset"), "--overwrite", *eps], check=False)
