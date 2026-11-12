#!/usr/bin/env python
"""Scene data for the Revo 2 grip-pose tuner page (docs/revo2_pose_tuner): for each recorded (Dex3) episode, the bottle and the
table top expressed in the right wrist_yaw frame at a few moments around the first right grip close (0.5 s before, 0.2 s
before, at close, 0.2 s after, 0.5 s after), the recorded Dex3 finger angles there, and which bodies touched the bottle.
Writes <out>/scenes.json. The page shows the Revo 2 hand in that same wrist frame, so a candidate open / closed pose can be
judged against where the bottle really was when the operator closed the grip.
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/revo2_pose_scenes.py <rot0 dataset dir> --episodes 0,1,... --out docs/revo2_pose_tuner
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--episodes", default=",".join(str(i) for i in range(50)))
ap.add_argument("--out", default="docs/revo2_pose_tuner"); ap.add_argument("--offsets", default="-25,-10,0,10,25", help="frames relative to the first right grip close (50 Hz)")
a = ap.parse_args(); SRC = Path(a.dataset).resolve(); OUT = (ROOT / a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
from simple.cli.replay_holomotion_v14 import _build_env, _reset_to
from simple.teleop.holomotion_v14 import exact_log as XL
from simple.robots.g1_sonic import RIGHT_HAND_JOINTS
OFFS = [int(x) for x in a.offsets.split(",")]
UP = np.array([0.0, 0.0, 1.0])


def r4(x): return np.round(np.asarray(x, dtype=float), 4).tolist()


def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(-1)); return q


def cluster(v, f, cell):
    """Vertex-clustering decimation (only used if the bottle mesh is large)."""
    key = np.floor(v / cell).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1)
    nv = int(inv.max()) + 1; acc = np.zeros((nv, 3)); cnt = np.zeros(nv)
    np.add.at(acc, inv, v); np.add.at(cnt, inv, 1); acc /= cnt[:, None]
    ff = inv[f]; ff = ff[(ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])]
    return acc, ff


def body_geoms(m, bid):
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid: continue
        gt = int(m.geom_type[g])
        e = dict(type=mujoco.mjtGeom(gt).name.replace("mjGEOM_", "").lower(), size=r4(m.geom_size[g]), pos=r4(m.geom_pos[g]), quat=r4(m.geom_quat[g]),
                 group=int(m.geom_group[g]), collides=bool(m.geom_contype[g] or m.geom_conaffinity[g]), rgba=r4(m.geom_rgba[g]))
        if gt == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g]); v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(int)
            if len(f) > 4000: v, f = cluster(v, f, 0.002)
            e["mesh"] = dict(v=r4(v), f=f.tolist(), faces_original=int(m.mesh_facenum[mid]))
        out.append(e)
    return out


def bottle_zmin(m, d, bid):
    """Lowest point of the bottle's geoms (world z) from their AABBs = the table top while the bottle rests on it."""
    z = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]; R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1): z.append(float((p + R @ (c + h * np.array([sx, sy, sz])))[2]))
    return min(z)


def touching(m, d, bid):
    names = set()
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 == bid and b2 != bid: names.add(m.body(b2).name)
        elif b2 == bid and b1 != bid: names.add(m.body(b1).name)
    return sorted(names)


env = None; out = dict(dataset=str(SRC), offsets=OFFS, episodes={})
for ep in [int(x) for x in a.episodes.split(",")]:
    log = XL.load(SRC / "replay" / f"episode_{ep:06d}.npz"); info, setup = dict(log["info"]), log["setup"]
    t = pq.read_table(str(SRC / "data/chunk-000" / f"episode_{ep:06d}.parquet")); pico = np.stack(t["holomotion.pico"].to_pandas().values); n = t.num_rows; hz = float(info["render_hz"])
    close = next((i for i in range(n) if pico[i][5] > 0.5), None)
    if close is None: out["episodes"][ep] = dict(grip_close_frame=None); continue
    if env is None:
        if info.get("policy_models_dir"): os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = info["policy_models_dir"]
        env, senv, mod = _build_env(info, setup, check_legacy=False); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
        out["render_hz"] = hz
    else:
        _reset_to(env, senv, mod, info, setup); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    label = str(senv.task.target.asset.label); bid = m.body(label).id; wid = m.body("right_wrist_yaw_link").id
    if "bottle" not in out: out["bottle"] = dict(label=label, geoms=body_geoms(m, bid))
    XL.set_state(m, d, log["physics"][0], XL.PHYSICS); mujoco.mj_forward(m, d); table_z = bottle_zmin(m, d, bid)
    world = {}; frames = {}
    for off in OFFS:
        f = int(min(max(close + off, 0), n - 1))
        XL.set_state(m, d, log["physics"][f], XL.PHYSICS); mujoco.mj_forward(m, d)
        pw, Rw = d.xpos[wid].copy(), d.xmat[wid].reshape(3, 3).copy(); pb, Rb = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
        world[off] = (pw, Rw)
        frames[off] = dict(frame=f, t=round(f / hz, 2), grip=int(pico[f][5] > 0.5), bottle_pos=r4(Rw.T @ (pb - pw)), bottle_quat=r4(mat2quat(Rw.T @ Rb)),
                           dex3_q=r4([d.qpos[m.jnt_qposadr[m.joint(j).id]] for j in RIGHT_HAND_JOINTS]), up=r4(Rw.T @ UP), table_d=round(float(table_z - pw[2]), 4),
                           wrist_world=r4(pw), wrist_above_table=round(float(pw[2] - table_z), 4), bottle_above_table=round(float(pb[2] - table_z), 4), contacts=touching(m, d, bid))
    if 0 in world and -10 in world:
        pw, Rw = world[0]; frames[0]["wrist_0p2s_before"] = r4(Rw.T @ (world[-10][0] - pw))
    out["episodes"][ep] = dict(grip_close_frame=int(close), grip_close_s=round(close / hz, 2), n=int(n), table_z=round(table_z, 4), frames={str(k): v for k, v in frames.items()})
    print(f"[scenes] ep {ep}: close {close / hz:.2f} s, bottle in wrist at close {frames[0]['bottle_pos']}, contacts {frames[0]['contacts']}", flush=True)
    json.dump(out, open(OUT / "scenes.json", "w"))
print("[scenes] done", flush=True); sys.stdout.flush(); os._exit(0)
