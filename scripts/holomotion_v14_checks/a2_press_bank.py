#!/usr/bin/env python
"""The A3 start bank from the A2 press-and-grasp episodes (2026-11-10, user: "use the 28 scenes to set the A3 init pose").

docs/_scan/a2_press_grasp/classify.json names the A2 episodes where one hand pressed one side of the plate down and the other
hand grasped the raised side and lifted (28 of 99). For each, the exact log's physics state at the slice's end event (the
plate 10 cm or more above its rest, held 0.5 s) is set in the rebuilt scene and read by name: the pelvis x, y, yaw (the
setups' robot frame), the 29 body joints and both hands' fingers, the held plate's pose in the holding hand's base frame (the
grasp, exact for that plate type), the holding hand's recorded command at that frame (what GripLatch holds until the operator
takes over), the other plate's spot (x, y, yaw on the worktop), both plate types and the plate set they index.

One process per recording setup (the kitchen module reads its knobs at import); --merge writes the bank:

    .venv/bin/python scripts/holomotion_v14_checks/a2_press_bank.py --dataset data/teleop_holomotion_v14_kitchen_a2_grip_v1 --tag grip_v1 --parts <dir>
    .venv/bin/python scripts/holomotion_v14_checks/a2_press_bank.py --merge --parts <dir> --out src/simple/teleop/holomotion_v14/start_banks/a2_press_end.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--dataset"); ap.add_argument("--tag"); ap.add_argument("--episodes", default="", help="lo-hi: only these (one recording setup per process)"); ap.add_argument("--parts", required=True)
ap.add_argument("--classify", default="docs/_scan/a2_press_grasp/classify.json")
ap.add_argument("--scene-joints", default="", help="comma list of scene joints whose recorded position the start restores (A5 from the A4 ends: k_bin_lid,k_bin_pedal)")
ap.add_argument("--held", default="", help="instead of --classify: a JSON list of {episode, plate, side[, source, press, selection]} for the episodes of --dataset "
                                          "(2026-11-10: the final A2 dataset, docs/_scan/a2_final/held.json)")
ap.add_argument("--merge", action="store_true"); ap.add_argument("--out", default="src/simple/teleop/holomotion_v14/start_banks/a2_press_end.json")
a = ap.parse_args()
PARTS = Path(a.parts); PARTS.mkdir(parents=True, exist_ok=True)
SET_NAMES = {"kitchen_plates": "v1", "kitchen_plates_lift": "lift", "kitchen_plates_v2": "v2"}

if a.merge:
    entries = []
    for f in sorted(PARTS.glob("*.json")):
        entries += json.loads(f.read_text())
    order = {"a2_hands": 0, "a2_grip": 1, "a2_grip_v1": 2, "a2_final": 3}
    entries.sort(key=lambda e: (order.get(e["dataset_short"], 9), e["episode"]))
    from collections import Counter
    bank = dict(meta=dict(slice="A2", made=time.strftime("%Y-%m-%d %H:%M"), n=len(entries), source=a.classify,
                          frame_rule="the exact log's physics state at slice.event_frame (the A2 gate held 0.5 s, the plate 10 cm or more up); pelvis = the floating base; scene frame = the setups' robot frame",
                          body_order="29 body joints in MJCF order: left leg (6), right leg (6), waist yaw/roll/pitch, left arm (7), right arm (7); fingers in grasp_start.FINGER_JOINTS order",
                          held="the held plate in the holding hand's base frame (rel_pos m, rel_quat wxyz), r_ref = that plate's radius (aabb), fingers = the holding hand at that frame, cmd = its recorded command (the latch target), palm_up_deg = the palm normal (+x of the hand base) from straight up (< 90 = palm facing up)",
                          plate_sets=dict(Counter(e["plate_set"] for e in entries))),
                entries=entries)
    out = Path(a.out); out.write_text(json.dumps(bank, indent=0, separators=(",", ":")))
    print(f"{len(entries)} entries -> {out} ({out.stat().st_size / 1e3:.0f} kB); plate sets {bank['meta']['plate_sets']}")
    sys.exit(0)

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from simple.teleop.holomotion_v14 import exact_log as XL
from simple.teleop.holomotion_v14.grasp_start import FINGER_JOINTS, plate_radius, hand_curl
from simple.cli.replay_holomotion_v14 import _build_env, _reset_to

lo_hi = [int(v) for v in a.episodes.split("-")] if a.episodes else [0, 10 ** 6]
if a.held:
    C = [dict(episode=h["episode"], plate=h["plate"], grasp_hand=h["side"], press=h.get("press"), source=h.get("source"), selection=h.get("selection"))
         for h in json.load(open(a.held)) if lo_hi[0] <= h["episode"] <= lo_hi[-1]]
else:
    C = [r for r in json.load(open(a.classify))["episodes"] if r["press_grasp"] and r["tag"] == a.tag and lo_hi[0] <= r["episode"] <= lo_hi[-1]]
DS = next((ROOT / a.dataset / "simple").glob("*/level-3_pinhole"))
rows = {json.loads(l)["episode_index"]: json.loads(l) for l in open(DS / "meta/episodes.jsonl")}
env = senv = mod = None; out = []


def quat_of(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


for r in C:
    ep = r["episode"]
    log = XL.load(DS / f"replay/episode_{ep:06d}.npz"); info, setup = log["info"], log["setup"]
    sm = rows[ep]["slice"]; sm = json.loads(sm) if isinstance(sm, str) else sm
    if env is None:
        env, senv, mod = _build_env(info, setup)
    else:
        _reset_to(env, senv, mod, info, setup)
    m, d = senv.mujoco.mjModel, senv.mujoco.mjData; robot = senv.task.robot
    assert XL.model_sha256(m) == info["model_sha256"], f"ep {ep}: the rebuilt model differs from the recorded one"
    f = int(sm["event_frame"])
    XL.set_state(m, d, log["physics"][f], XL.PHYSICS); mujoco.mj_forward(m, d)
    B = lambda n: m.body(n).id; Jq = lambda n: int(m.jnt_qposadr[m.joint(n).id])
    pel = B("pelvis"); ra = int(m.jnt_qposadr[m.body_jntadr[int(m.body_rootid[pel])]])
    root = d.qpos[ra:ra + 7].copy(); Rpel = d.xmat[pel].reshape(3, 3)
    yaw = math.atan2(Rpel[1, 0], Rpel[0, 0])
    body = d.qpos[m.jnt_qposadr[np.asarray(robot.body_joint_index, dtype=int)]].copy()
    fingers = {s: [round(float(d.qpos[Jq(f"{s}_hand_{n}_joint")]), 4) for n in FINGER_JOINTS] for s in ("left", "right")}
    plate, side = r["plate"], r["grasp_hand"]; other = "plate_b" if plate == "plate_a" else "plate_a"
    pb, hb, ob = B("k_" + plate), B(f"{side}_hand_base_link"), B("k_" + other)
    Rh, ph, Rp, pp = d.xmat[hb].reshape(3, 3), d.xpos[hb], d.xmat[pb].reshape(3, 3), d.xpos[pb]
    rel_pos = Rh.T @ (pp - ph); rel_quat = quat_of(Rh.T @ Rp)
    cmd = np.asarray(log[f"action_{side}_hand_q"][f], dtype=float)[:11]
    Ro, po = d.xmat[ob].reshape(3, 3), d.xpos[ob]
    o_yaw = math.degrees(math.atan2(Ro[1, 0], Ro[0, 0])); o_tilt = math.degrees(math.acos(np.clip(Ro[2, 2], -1, 1)))
    fwd = np.array([math.cos(yaw), math.sin(yaw)]); lft = np.array([-math.sin(yaw), math.cos(yaw)])
    rel = {p: dict(ahead=round(float(fwd @ (d.xpos[b][:2] - root[:2])), 4), left=round(float(lft @ (d.xpos[b][:2] - root[:2])), 4)) for p, b in ((plate, pb), (other, ob))}
    knobs = info.get("env_knobs", {}); pset = SET_NAMES.get(os.path.basename(knobs.get("KITCHEN_PLATES", "kitchen_plates").rstrip("/")), "?")
    e = dict(episode=ep, dataset=str(DS.parents[2].name), dataset_short=str(DS.parents[2].name).replace("teleop_holomotion_v14_kitchen_", ""), tag=a.tag, seed=setup.get("seed"),
             frame=f, frames=len(log["physics"]), hand_source=info.get("hand_source", "grip"), plate_set=pset, balls=knobs.get("KITCHEN_PLATE_BALLS"),
             robot=dict(x=round(float(root[0]), 4), y=round(float(root[1]), 4), yaw_deg=round(math.degrees(yaw), 2)), pelvis_z=round(float(root[2]), 4), root_quat=[round(float(v), 6) for v in root[3:7]],
             pelvis_tilt_deg=round(math.degrees(math.acos(np.clip(Rpel[2, 2], -1, 1))), 2),
             body=[round(float(v), 4) for v in body], fingers=fingers,
             plate=setup.get("plate"), plate_b=setup.get("plate_b"), target=plate, rel=rel,
             plate_spots={other: [round(float(po[0]), 4), round(float(po[1]), 4), round(o_yaw, 1)]}, other_tilt_deg=round(o_tilt, 2),
             held=dict(plate=plate, side=side, rel_pos=[round(float(v), 5) for v in rel_pos], rel_quat=[round(float(v), 6) for v in rel_quat],
                       r_ref=round(plate_radius(m, pb), 5), fingers=fingers[side], cmd=[round(float(v), 4) for v in cmd],
                       tilt_deg=round(math.degrees(math.acos(np.clip(Rp[2, 2], -1, 1))), 1), plate_z=round(float(pp[2]), 4),
                       curl=round(hand_curl(fingers[side]), 3), cmd_curl=round(hand_curl(cmd), 3),
                       palm_up_deg=round(math.degrees(math.acos(np.clip(Rh[2, 0], -1, 1))), 1)),   # the palm faces +x of the Revo 2 hand base (the fingers flex toward +x): its angle from straight up
             press=r["press"] if "press" in r else dict(hand=r["press_hand"], seconds=r["press_s"], max_n=r["press_max_n"], tip_deg=r["press_tilt_max_deg"]))
    if a.scene_joints:
        e["scene_joints"] = {n: round(float(d.qpos[Jq(n)]), 5) for n in a.scene_joints.split(",")}
    if r.get("source"):
        e["source"] = r["source"]
    if r.get("selection"):
        e["selection"] = r["selection"]
    out.append(e)
    print(f"[bank] {a.tag} ep {ep:2d} frame {f}: pelvis ({e['robot']['x']:+.3f}, {e['robot']['y']:+.3f}) {e['robot']['yaw_deg']:+.1f} deg, {plate} in the {side} hand "
          f"tilt {e['held']['tilt_deg']} deg, z {e['held']['plate_z']} m, curl {e['held']['curl']} cmd {e['held']['cmd_curl']}; {other} at {e['plate_spots'][other]} tilt {e['other_tilt_deg']}; set {pset} ids {e['plate']}/{e['plate_b']}", flush=True)
(PARTS / f"{a.tag}_{'_'.join(str(r['episode']) for r in C[:1])}_{len(C)}.json").write_text(json.dumps(out))
print("done", len(out))
