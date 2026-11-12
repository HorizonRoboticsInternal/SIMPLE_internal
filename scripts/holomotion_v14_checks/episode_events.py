"""Ground-truth event log of a recorded kitchen teleop episode, from its bit-exact replay (2026-10-27).

    MUJOCO_GL=egl .venv/bin/python -u scripts/holomotion_v14_checks/episode_events.py <level dir> <episode> <out.npz> [--sample-every 25]

Feeds the segment / caption / sub-task annotation (episode_annotate.py). Every recorded frame (50 Hz): robot base pose and
planar speed, every kitchen hinge/slide joint (dishwasher door and racks, drawers, sink, fridge, bin), pose of every free
prop, both palm positions, what each hand touches (names: the props, "dishwasher door", "upper rack", "lower rack",
"counter", other kitchen parts, the other hand) and what the rest of the robot touches ("body"), and the rack body poses. Every --sample-every frames (25 = 2 per second, the Horizon caption rate):
which objects the recorded head camera (hbvcam_left_pinhole, the rectified left eye) sees: centre inside the 640x360 image
and the sight line not blocked by something else (mj_ray), and whether each hand is in the picture. The replay must be bit-exact or the script stops.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from simple.cli.replay_holomotion_v14 import _build_env  # noqa: E402
from simple.teleop.holomotion_v14 import exact_log as XL  # noqa: E402

PROPS = {"k_plate_a": "plate A", "k_plate_b": "plate B", "k_bowl_small": "small bowl", "k_bowl_large": "large bowl", "bottle_red": "red bottle",
         "k_bottle_green": "green bottle", "k_carton_bags": "carton of bags", "k_carton_wrap": "carton of wrap"}
PARTS = {"k_dishwasher_door": "dishwasher door", "k_dishwasher_upper_rack": "upper rack", "k_dishwasher_lower_rack": "lower rack",
         "k_sink_left": "left sink", "k_sink_right": "right sink", "k_refrigerator_door": "fridge door", "k_freezer_door": "freezer door",
         "k_bin_lid": "bin lid", "k_pedal_bin": "pedal bin", "k_utility_drawer": "utility drawer"}
VIEW = {**PROPS, "k_dishwasher_door": "dishwasher door", "k_dishwasher_upper_rack": "upper rack", "k_dishwasher_lower_rack": "lower rack",
        "k_sink_left": "sink", "k_pedal_bin": "pedal bin", "k_refrigerator_door": "fridge"}
CAM, W, H = "hbvcam_left_pinhole", 640, 360


def part_box(m, d, body: str) -> dict | None:
    """World-axis box around one kitchen body's own geoms (the corners of each geom's local box), e.g. the pedal bin's footprint."""
    try:
        b = m.body(body).id
    except KeyError:
        return None
    pts = [d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (m.geom_aabb[g, :3] + np.array(s) * m.geom_aabb[g, 3:])
           for g in np.flatnonzero(m.geom_bodyid == b) for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
    return dict(lo=np.min(pts, 0).round(3).tolist(), hi=np.max(pts, 0).round(3).tolist()) if pts else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("level_dir"); ap.add_argument("episode", type=int); ap.add_argument("out")
    ap.add_argument("--sample-every", type=int, default=25)
    a = ap.parse_args()
    log = XL.load(Path(a.level_dir) / f"replay/episode_{a.episode:06d}.npz"); info = log["info"]
    env, senv, mod = _build_env(info, log["setup"]); m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""

    def top(b):                                                  # the kitchen part / prop / robot a body belongs to
        while b > 0:
            n = bname(b)
            if n in PROPS or n in PARTS: return n
            if n == "pelvis": return "robot"
            b = int(m.body_parentid[b])
        return "world"
    owner = [top(int(m.geom_bodyid[g])) for g in range(m.ngeom)]
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    hand_of = []
    for g in range(m.ngeom):
        n = bname(int(m.geom_bodyid[g]))
        hand_of.append("left" if n.startswith("left_hand_") else "right" if n.startswith("right_hand_") else "")
    counter = {g for g in range(m.ngeom) if any(s in gname(g) for s in ("peninsula", "worktop", "counter"))}

    def label(g):
        if g in counter: return "counter"
        o = owner[g]
        if o in PROPS: return PROPS[o]
        if o in PARTS: return PARTS[o]
        if o == "robot": return {"left": "left hand", "right": "right hand"}.get(hand_of[g], "robot body")
        return "floor" if gname(g) in ("floor", "ground") or int(m.geom_bodyid[g]) == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE else "kitchen"

    joints = [j for j in range(m.njnt) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("k_") and m.jnt_type[j] in (2, 3)]
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]; jadr = [int(m.jnt_qposadr[j]) for j in joints]
    props = [b for b in range(m.nbody) if bname(b) in PROPS]
    palm = {s: m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")}
    pel = m.body("pelvis").id; cam = m.camera(CAM).id
    view_bodies = [(bname(b), VIEW[bname(b)]) for b in range(m.nbody) if bname(b) in VIEW]
    robot_bodies = {b for b in range(m.nbody) if top(b) == "robot"}
    fovy = float(m.cam_fovy[cam]); f_px = (H / 2) / np.tan(np.radians(fovy) / 2)

    rec = dict(base=[], yaw=[], vxy=[], joints=[], prop_pos=[], prop_quat=[], palm=[], rack_pos=[], rack_mat=[])
    racks = [m.body(n).id for n in ("k_dishwasher_upper_rack", "k_dishwasher_lower_rack")]
    touch = []; views = {}
    gid = np.zeros(1, np.int32); grp = np.array([1, 1, 1, 0, 0, 0], np.uint8)

    def visible(t):
        cp = d.cam_xpos[cam].copy(); R = d.cam_xmat[cam].reshape(3, 3)          # camera looks along -z, x right, y up
        out = []
        for b, nm in view_bodies:
            p = d.xipos[m.body(b).id] if bname(m.body(b).id) in PROPS else d.xpos[m.body(b).id]
            if bname(m.body(b).id) in ("k_dishwasher_door",):                       # the door body origin is the hinge: aim at the panel middle
                p = d.xpos[m.body(b).id] + d.xmat[m.body(b).id].reshape(3, 3) @ np.array([0, 0, 0.35])
            v = R.T @ (p - cp)
            if v[2] >= -0.05: continue
            u, w = W / 2 + f_px * v[0] / -v[2], H / 2 - f_px * v[1] / -v[2]
            if not (0 <= u < W and 0 <= w < H): continue
            dvec = p - cp; L = float(np.linalg.norm(dvec))
            hit = mujoco.mj_ray(m, d, cp, dvec / L, grp, 1, -1, gid)
            hb = int(m.geom_bodyid[gid[0]]) if hit >= 0 else -1
            seen = hit < 0 or hit >= L - 0.08 or top(hb) == top(m.body(b).id)
            out.append(dict(name=nm, px=[round(u), round(w)], dist_m=round(L, 2), occluded=not seen, by=None if seen else (label(int(gid[0])))))
        hv = {}
        for side in ("left", "right"):                                          # the hand: palm centre ~8 cm past the wrist yaw link
            pb = palm[side]; p = d.xpos[pb] + d.xmat[pb].reshape(3, 3) @ np.array([0.08, 0, 0]); v = R.T @ (p - cp)
            hv[side] = bool(v[2] < -0.05 and 0 <= W / 2 + f_px * v[0] / -v[2] < W and 0 <= H / 2 - f_px * v[1] / -v[2] < H)
        views[t] = dict(objects=out, hands=hv)

    def on_frame(t):
        rec["base"].append(d.qpos[:3].copy()); q = d.qpos[3:7]
        rec["yaw"].append(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        rec["vxy"].append(np.hypot(*d.qvel[:2])); rec["joints"].append([d.qpos[i] for i in jadr])
        rec["prop_pos"].append(np.stack([d.xipos[b] for b in props])); rec["prop_quat"].append(np.stack([d.xquat[b] for b in props]))
        rec["palm"].append(np.stack([d.xpos[palm[s]] for s in ("left", "right")]))
        rec["rack_pos"].append(np.stack([d.xpos[b] for b in racks])); rec["rack_mat"].append(np.stack([d.xmat[b].reshape(3, 3) for b in racks]))
        tt = {"left": set(), "right": set(), "body": set()}
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
            for ga, gb in ((g1, g2), (g2, g1)):
                h = hand_of[ga]
                if h and owner[gb] != "robot" or (h and hand_of[gb] and hand_of[gb] != h):
                    tt[h].add(label(gb))
                elif owner[ga] == "robot" and not h and owner[gb] != "robot" and label(gb) not in ("floor", "kitchen"):
                    tt["body"].add(label(gb))                                   # arms, legs, torso on a kitchen part or prop
        touch.append({k: sorted(v) for k, v in tt.items()})
        if t % a.sample_every == 0:
            visible(t)

    res = XL.replay_physics(m, d, log, on_frame=on_frame)
    if not res["exact"]:
        sys.exit(f"replay is not bit-exact: {res}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, base=np.array(rec["base"]), yaw=np.array(rec["yaw"]), vxy=np.array(rec["vxy"]), joints=np.array(rec["joints"]),
                        prop_pos=np.array(rec["prop_pos"]), prop_quat=np.array(rec["prop_quat"]), palm=np.array(rec["palm"]),
                        rack_pos=np.array(rec["rack_pos"]), rack_mat=np.array(rec["rack_mat"]))
    meta = dict(level_dir=str(a.level_dir), episode=a.episode, racks=["upper rack", "lower rack"], frames=len(touch), fps=50, bit_exact=True, joint_names=jnames,
                props=[PROPS[bname(b)] for b in props], prop_bodies=[bname(b) for b in props], camera=CAM, image=[W, H], fovy=fovy,
                sample_every=a.sample_every, touch=touch, views={str(k): v for k, v in views.items()},
                scene=dict(rev=info.get("env_knobs", {}).get("KITCHEN_REV"), task=mod.TASK, rack_box=mod.RACK_BOX, door_open_rad=mod.DOOR_OPEN_RAD,
                           bin_box=part_box(m, d, "k_pedal_bin")))                 # the bin body does not move (lid and pedal are its children)
    Path(str(out.with_suffix("")) + ".json").write_text(json.dumps(meta))
    print(f"[events] {len(touch)} frames, replay BIT-EXACT; {len(views)} camera samples -> {out} + .json", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
