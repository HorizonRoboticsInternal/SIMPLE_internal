#!/usr/bin/env python3
"""Patch a psi0-format sim dataset IN PLACE (2026-09-29, user: fix only the flagged episodes, replace them in the psi0 data):
  1. label fixes on every episode, once (marker info.json "sim_label_fixes"): action left hand in the psi0 order by joint name
     (thumb, middle, index: swap [3:5] <-> [5:7] of the stored index-first label), bowl base command vx/vy divided by the replay
     nav gain (the operator's command), bottle task text "water bottle";
  2. replace episodes re-rendered by render_fix.py (--replace EP ...: replay episode numbers): parquet rebuilt from the new render
     (to_psi0.convert_episode, same length and global indices) and the video cut to the same frames;
  3. episodes_stats / stats / stats_psi0 recomputed. A backup of the dataset is kept next to it.
Usage: patch_psi0.py KIT [--replace EP ...] [--no-backup]"""
import os
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import to_psi0 as T
ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--replace", type=int, nargs="*", default=[]); ap.add_argument("--no-backup", action="store_true")
ap.add_argument("--dataset-dir", default=None, help="patch this copy instead (testing)"); a = ap.parse_args()
kit = a.kit; D = Path(a.dataset_dir) if a.dataset_dir else T.PHYS / "psi0" / T.OUTN[kit]
if not a.no_backup:
    B = T.PHYS / "psi0" / f"_backup_{T.OUTN[kit]}_20260929"
    if not B.exists(): shutil.copytree(D, B); print("backup ->", B)
info = json.load(open(D / "meta/info.json")); eps = [json.loads(l) for l in open(D / "meta/episodes.jsonl")]
by_src = {e["source"]["episode_index"]: e for e in eps}
done = set(info.get("sim_label_fixes", []))
FIX = "left_hand_psi0_order" not in done
NAVFIX = kit == "bowl_sink" and "operator_base_command" not in done


def write(tbl_cols, path, schema):
    arrays = [pa.array(tbl_cols[n], type=schema.field(n).type) for n in schema.names]
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)


replaced = {}
for ep in a.replace:
    e = by_src.get(ep)
    if e is None: print("replay episode", ep, "is not in the dataset"); continue
    k, n, i0 = e["episode_index"], e["length"], e["dataset_from_index"]
    d = T.PHYS / kit / f"ep_{ep:03d}"; meta = json.load(open(d / "replay_meta.json")); env = meta.get("scene_env", {})
    gain = float(env.get("REPLAY_NAV_GAIN", 1.5 if kit == "bowl_sink" else 1.0))
    tbl, n_new, st, ac = T.convert_episode(d, k, i0, keep=n, nav_gain=gain)
    assert n_new == n, (ep, n_new, n)
    pq.write_table(tbl, D / f"data/chunk-000/episode_{k:06d}.parquet")
    src = sorted((d / "dataset/videos").rglob("*.mp4"))[0]; nv = T.trim_video(src, D / f"videos/chunk-000/egocentric/episode_{k:06d}.mp4", n)
    assert nv == n, (ep, nv, n)
    m = (meta.get("env_config") or {}).get("dr_state_dict", {}).get("material", {})
    e["environment_config"]["rerendered_20260929"] = {"reason": "see-through / thin-walled table material (black or hollow object, light reflections)",
        "table_material": (m.get("table_material") or {}).get("path", "").split("vMaterials_2/")[-1], "ground_material": (m.get("ground_material") or {}).get("path", "").split("vMaterials_2/")[-1],
        "same_run": "forced from the recorded MuJoCo states of the original render; same seed and lighting"}
    replaced[k] = True; print(f"replaced dataset episode {k} (replay {ep}): {n} frames")
# label fixes on the episodes that were NOT rebuilt (rebuilt ones already carry them)
all_s, all_a, all_t, all_f, all_e, all_i, all_d, eps_stats = [], [], [], [], [], [], [], []
for e in eps:
    k = e["episode_index"]; p = D / f"data/chunk-000/episode_{k:06d}.parquet"; t = pq.read_table(p); cols = {n: t.column(n).to_pylist() for n in t.schema.names}
    A = np.array(cols["action"], dtype=np.float64)
    if not replaced.get(k):
        if FIX: A[:, [3, 4, 5, 6]] = A[:, [5, 6, 3, 4]]
        if NAVFIX: A[:, 32:34] /= float(e["environment_config"].get("nav_gain", e["environment_config"].get("nav_gain_in_sim", 1.5)))
        if FIX or NAVFIX: cols["action"] = [r.tolist() for r in A]; write(cols, p, t.schema)
    if kit == "bowl_sink": e["environment_config"]["label_base_command"] = "operator's (the sim executed it x nav_gain)"
    n = len(A); S = np.array(cols["states"], dtype=np.float64); ts = np.array(cols["timestamp"])
    eps_stats.append({"episode_index": k, "stats": {"action": dict(T.stats(A), count=[n]), "timestamp": dict(T.stats(ts), count=[n])}})
    all_s.append(S); all_a.append(A); all_t.append(ts); all_f.append(np.array(cols["frame_index"])); all_e.append(np.array(cols["episode_index"])); all_i.append(np.array(cols["index"])); all_d.append(np.array(cols["next.done"]))
if kit in T.TASK_TEXT:
    task = json.loads(open(D / "meta/tasks.jsonl").readline()); task.update(task=T.TASK_TEXT[kit], description=T.TASK_TEXT[kit])
    open(D / "meta/tasks.jsonl", "w").write(json.dumps(task) + "\n")
    for e in eps: e["instruction"] = {"task_index": 0, "task": T.TASK_TEXT[kit]}
cat = np.concatenate
st_all = {"states": T.stats(cat(all_s)), "action": T.stats(cat(all_a)), "timestamp": T.stats(cat(all_t)), "frame_index": T.stats(cat(all_f)), "episode_index": T.stats(cat(all_e)),
          "index": T.stats(cat(all_i)), "task_index": T.stats(np.zeros(len(cat(all_i)))), "next.done": T.stats(cat(all_d).astype(np.float32))}
json.dump(st_all, open(D / "meta/stats.json", "w"), indent=4); json.dump(st_all, open(D / "meta/stats_psi0.json", "w"), indent=4)
open(D / "meta/episodes_stats.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in eps_stats))
open(D / "meta/episodes.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in eps))
info["sim_label_fixes"] = sorted(done | {"left_hand_psi0_order"} | ({"operator_base_command"} if kit == "bowl_sink" else set()) | ({"task_text"} if kit in T.TASK_TEXT else set()))
json.dump(info, open(D / "meta/info.json", "w"), indent=4)
print(f"[patch-psi0] {kit}: {len(eps)} episodes, label fixes {'applied' if (FIX or NAVFIX) else 'already present'}, replaced {sorted(replaced)}")
