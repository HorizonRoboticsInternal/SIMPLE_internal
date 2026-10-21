#!/usr/bin/env python
"""Old vs new bottle_bin "placed" gate on recorded episodes (2026-10-21; the fix the gate-check page proposed, applied).

Replays each episode's exact log (bit-exact mj_step) and runs the task's gate logic with two "in the bin" rules:
    old   the bottle's body origin (its base) inside the opening shrunk by wall + bottle radius, below the rim
    new   bottle_bin_task._in_bin as patched today: every collision vertex within the inner walls (5 mm tolerance),
          lowest point below the rim
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/bottle_bin_gate_recheck.py <dataset dir> <out json> [episodes]
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from simple.teleop.holomotion_v14 import exact_log as XL
from simple.cli.replay_holomotion_v14 import _build_env, _reset_to

D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.parent.mkdir(parents=True, exist_ok=True)
summ = {r["source_episode"]: r for r in json.load(open(D / "meta" / "rerecord_summary.json"))["episodes"]}
metas = [json.loads(l) for l in open(D / "meta" / "episodes.jsonl")]
want = [int(e) for e in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(len(metas)))
env = se = mod = BS = None; rows = []; t0 = time.time()
for ep in want:
    log = XL.load(D / "replay" / f"episode_{ep:06d}.npz"); info, setup = log["info"], log["setup"]
    if env is None:
        env, se, mod = _build_env(info, setup); BS = sys.modules[mod.__name__].BS
    else:
        _reset_to(env, se, mod, info, setup)
    task, m, d = se.task, se.mujoco.mjModel, se.mujoco.mjData
    task._bind_gate_ids(m)
    assert hasattr(task, "_g_target_verts") and len(task._g_target_verts) > 8, "task not patched"

    def inside_old(target, bp):
        w, x, y, z = bp[3:7]; yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
        ax, ay = (BS.BIN_W, BS.BIN_D) if abs(round(yaw)) % 180 == 90 else (BS.BIN_D, BS.BIN_W)
        hx, hy = ax / 2 - BS.BIN_T - BS.BOTTLE_R, ay / 2 - BS.BIN_T - BS.BOTTLE_R
        return bool(abs(target[0] - bp[0]) <= hx and abs(target[1] - bp[1]) <= hy and target[2] < BS.BIN_H)

    rules = {"old": inside_old, "new": lambda target, bp: task._in_bin(target, bp[:3], bp[3:7])}
    st = {k: dict(grasped=None, placed=None, hold=0, inbin=0, first_inside=None) for k in rules}; init_h = [None]; frames = [0]

    def on_frame(t):
        inf = se._get_info(); target, bp = np.asarray(inf["target"], float), np.asarray(inf["trash_bin"], float)
        if init_h[0] is None: init_h[0] = float(target[2])
        hand, _touched, on_table = task._contacts(d)
        qw, qx, qy, qz = target[3:7]; tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1))))
        holding = hand and (target[2] - init_h[0] >= task.GRASP_LIFT or not on_table) and tilt < task.GRASP_TILT
        for k, fn in rules.items():
            s = st[k]; s["hold"] = s["hold"] + 1 if holding else 0
            if s["grasped"] is None and s["hold"] * task.STEP_DT >= task.GRASP_HOLD_S: s["grasped"] = t
            ins = s["grasped"] is not None and not hand and fn(target, bp)
            s["inbin"] = s["inbin"] + 1 if ins else 0
            if s["inbin"] == 1: s["first_inside"] = t
            if s["placed"] is None and s["inbin"] * task.STEP_DT >= task.PLACE_HOLD_S: s["placed"] = t
        frames[0] = t

    res = XL.replay_physics(m, d, log, on_frame=on_frame); n = int(res["frames"])
    # Settle: the old gate ended the recording 0.5 s after IT fired; if the new rule is true at the last frame but its own
    # 0.5 s hold is not complete, keep stepping with the last control held (robot standing, bottle free) for up to 1 s.
    settle = None
    if st["new"]["placed"] is None and st["new"]["inbin"] > 0:
        import mujoco
        nstep_last, calls_last = int(log["nstep"][-1]), int(log["frame_calls"][-1])
        for k in range(50):
            for _ in range(calls_last):
                mujoco.mj_step(m, d, nstep=nstep_last)
            on_frame(n + k)
            if st["new"]["placed"] is not None:
                settle = k + 1
                break
    row = dict(episode=ep, frames=n, physics_bitexact=bool(res["exact"]), source_saved_by_hand=summ[ep]["source_saved_by_hand"],
               recorded_success_on_last_frame=summ[ep]["success_on_last_frame"], recorded_success_frame=summ[ep]["success_frame"],
               old_placed=st["old"]["placed"], new_placed=st["new"]["placed"], new_end_streak=int(st["new"]["inbin"]), old_end_streak=int(st["old"]["inbin"]),
               new_first_inside=st["new"]["first_inside"], old_first_inside=st["old"]["first_inside"],
               old_grasped=st["old"]["grasped"], new_grasped=st["new"]["grasped"],
               new_placed_in_recording=st["new"]["placed"] is not None and st["new"]["placed"] < n, settle_frames=settle)
    rows.append(row)
    print(f"[gate-fix] ep {ep}: {n} frames, {'bit-exact' if res['exact'] else 'DIFFERS'}; recorded {'hand-saved' if row['source_saved_by_hand'] else 'auto'} success_frame {row['recorded_success_frame']}; "
          f"old placed {row['old_placed']} new placed {row['new_placed']} (new streak at end {row['new_end_streak']}{f', settled {settle} frames past the end' if settle else ''}) [{time.time() - t0:.0f} s]", flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
ok = sum(r["new_placed"] is not None for r in rows); same = sum((r["new_placed"] == r["old_placed"]) for r in rows if r["old_placed"] is not None)
print(f"[gate-fix] done: new gate passes {ok}/{len(rows)}; same frame as old on {same}/{sum(r['old_placed'] is not None for r in rows)} old passes; old passes {sum(r['old_placed'] is not None for r in rows)}", flush=True)
sys.stdout.flush(); os._exit(0)
