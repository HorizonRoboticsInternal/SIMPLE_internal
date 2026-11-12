#!/usr/bin/env python
"""Fill the decoupled-WBC wrist / height columns of an existing HoloMotion v1.4 LeRobot dataset (2026-10-28).

    .venv/bin/python scripts/holomotion_v14_checks/fill_eef_columns.py <level dir> [--write]

User, 2026-10-28, on kitchen5: "is it intended to leave empty?" for the all-zero columns. observation.eef_state, action.eef
and teleop.base_height_command were zero placeholders in the v1.4 recorder; the recorder now fills them
(simple.teleop.holomotion_v14.eef, which explains the conventions), and this script fills recordings made before that, from
columns they already have: observation.state (the 29 body joints) -> observation.eef_state, reference_actions block 0 (the
reference frame the policy tracked) -> action.eef and teleop.base_height_command. Same function as the recorder, so a filled
old dataset and a new recording agree.

Without --write it only reports. With --write: refuses columns that are not all zero (real data), copies each changed
parquet and meta/episodes_stats.jsonl to meta/_before_eef_fill_20261028/, rewrites the three columns in place (types and
schema metadata kept, every other column untouched) and their per-episode stats.
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from simple.teleop.holomotion_v14.eef import G1_BODY_JOINTS, wrist_fk  # noqa: E402

COLS = ("observation.eef_state", "action.eef", "teleop.base_height_command")
BACKUP = "_before_eef_fill_20261028"


def stats(v: np.ndarray) -> dict:
    return dict(min=v.min(0).tolist(), max=v.max(0).tolist(), mean=v.mean(0).tolist(), std=v.std(0).tolist(), count=[len(v)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("level_dir"); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    L = Path(a.level_dir); info = json.loads((L / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    if list(names[:29]) != G1_BODY_JOINTS:
        sys.exit("observation.state does not start with the 29 G1 body joints in G1_BODY_JOINTS order")
    for c in COLS + ("reference_actions",):
        if c not in info["features"]:
            sys.exit(f"{c} is not in this dataset")
    fk = wrist_fk(); new_stats = {}
    for f in sorted((L / "data").glob("chunk-*/episode_*.parquet")):
        ep = int(f.stem.split("_")[1]); t = pq.read_table(f)
        S = np.stack(t.column("observation.state").to_pylist()); R = np.stack(t.column("reference_actions").to_pylist()).reshape(len(S), 11, 65)
        old = {c: np.array([np.atleast_1d(x) for x in t.column(c).to_pylist()], dtype=np.float64) for c in COLS}
        eef = np.array([fk.poses(s[:29]) for s in S]); cmd = [fk.command(r[0]) for r in R]
        new = {"observation.eef_state": eef, "action.eef": np.array([c[0] for c in cmd]), "teleop.base_height_command": np.array([[c[1]] for c in cmd])}
        nocmd = int(sum(not np.any(r[0]) for r in R))
        live = [i for i, r in enumerate(R) if np.any(r[0])]
        err = np.linalg.norm(new["action.eef"][live][:, [0, 1, 2, 7, 8, 9]].reshape(-1, 2, 3) - eef[live][:, [0, 1, 2, 7, 8, 9]].reshape(-1, 2, 3), axis=2) if live else np.zeros((1, 2))
        print(f"[eef] episode {ep}: {len(S)} rows; old all zero: {', '.join(c for c in COLS if not old[c].any()) or 'none'}; "
              f"{nocmd} rows before the reference window filled (no-command defaults); commanded vs measured wrist position "
              f"median {np.median(err) * 100:.1f} cm, 95th pct {np.percentile(err, 95) * 100:.1f} cm; reference height "
              f"{new['teleop.base_height_command'][live].min():.3f}-{new['teleop.base_height_command'][live].max():.3f} m", flush=True)
        busy = [c for c in COLS if old[c].any()]
        if a.write and busy:
            sys.exit(f"episode {ep}: {busy} already hold data; not overwriting")
        if a.write:
            bk = L / "meta" / BACKUP / f.relative_to(L); bk.parent.mkdir(parents=True, exist_ok=True)
            if not bk.exists(): shutil.copy2(f, bk)
            for c in COLS:
                field = t.schema.field(c)                                       # fixed_size_list<double>[14], or a plain double (the height)
                vals = [list(v) for v in new[c]] if pa.types.is_fixed_size_list(field.type) or pa.types.is_list(field.type) else new[c][:, 0].tolist()
                t = t.set_column(t.schema.get_field_index(c), field, pa.array(vals, type=field.type))
            pq.write_table(t, f)
        new_stats[ep] = {c: stats(new[c]) for c in COLS}
    if a.write:
        sp = L / "meta/episodes_stats.jsonl"; bk = L / "meta" / BACKUP / "episodes_stats.jsonl"
        if not bk.exists(): shutil.copy2(sp, bk)
        rows = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()]
        for r in rows:
            if r["episode_index"] in new_stats: r["stats"].update(new_stats[r["episode_index"]])
        sp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[eef] wrote {len(new_stats)} episodes; originals in {L / 'meta' / BACKUP}", flush=True)


if __name__ == "__main__":
    main()
