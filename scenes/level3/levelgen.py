#!/usr/bin/env python
"""Level-3 generation wrapper: runs SIMPLE's decoupled-WBC DR CLI or the scene kits' make_levels.py in this process with
(1) its own DDS domain (LEVELGEN_DDS_DOMAIN), so the CycloneDDS participant it creates does not count against domain 0,
    where another session's replay workers sit near the ~12-participant ceiling, and
(2) the base episodes cycled up to LEVELGEN_CYCLE_TO (dr_decoupled_wbc caps --num-episodes at the base set's size; the
    new-20 base sets have 20 episodes, level 3 wants 30: scenes 21-30 reuse base tables 1-10, everything else re-sampled).

    python levelgen.py dr_wbc  <dr_decoupled_wbc args...>
    python levelgen.py kit     <make_levels.py path> <kit> --_level 3 --episodes 30 --out <dir> --seed 0
"""
import os
import runpy
import sys

DOMAIN = int(os.environ.get("LEVELGEN_DDS_DOMAIN", "0"))
CYCLE = int(os.environ.get("LEVELGEN_CYCLE_TO", "0"))

import simple.cli.dr_decoupled_wbc as D  # noqa: E402

_orig_cfg = D._make_sonic_config


def _cfg():
    c = _orig_cfg()
    if DOMAIN:
        c["DOMAIN_ID"] = DOMAIN
    print(f"[levelgen] DDS domain {c.get('DOMAIN_ID')}", flush=True)
    return c


D._make_sonic_config = _cfg

if CYCLE:
    _oe, _oc = D._load_episodes, D._load_episode_configs

    def _cycle(d):
        ks = sorted(d)
        return {i: d[ks[i % len(ks)]] for i in range(max(CYCLE, len(ks)))}

    D._load_episodes = lambda p: _cycle(_oe(p))
    D._load_episode_configs = lambda p: _cycle(_oc(p))
    print(f"[levelgen] base episodes cycled to {CYCLE}", flush=True)

if os.environ.get("LEVELGEN_EXACT") == "1":
    # render the given scene states exactly: whatever --dr-level the CLI passes, load the whole stored state
    # (robot start, table pose, object poses, distractors, materials, lighting) instead of re-drawing parts of it
    from simple.dr.manager import DRManager
    _orig_load = DRManager.load_state_dict

    def _exact(self, state_dict, dr_level=None):
        return _orig_load(self, state_dict, dr_level=None)

    DRManager.load_state_dict = _exact
    print("[levelgen] exact reload: every scene state is loaded as stored", flush=True)

mode = sys.argv[1]
if mode == "dr":
    import typer
    from simple.cli import dr as DRCLI
    sys.argv = ["dr"] + sys.argv[2:]
    typer.run(DRCLI.main)
elif mode == "dr_wbc":
    import typer
    sys.argv = ["dr_decoupled_wbc"] + sys.argv[2:]
    typer.run(D.main)
elif mode == "kit":
    path = sys.argv[2]
    sys.argv = [path] + sys.argv[3:]
    runpy.run_path(path, run_name="__main__")
else:
    raise SystemExit(f"unknown mode {mode}")
