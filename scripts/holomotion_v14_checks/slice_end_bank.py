#!/usr/bin/env python
"""End states of a recorded slice -> a start bank for the next slice (2026-11-03; user: "use the A1 collected data to redesign
the A2 scene").

Reads a recorded kitchen slice dataset (meta/episodes.jsonl + data/chunk-*/episode_*.parquet, lean or filled) and, for every
complete episode, takes the robot's state at the slice's end event (``slice.event_frame``: the frame the end gate had held for
its hold time; ``--at gate`` / ``--at cut`` take the gate's first frame / the recording's cut instead): the pelvis x, y, yaw in
the scene frame (= the setups' robot frame), the 29 body joints and both hands' eleven finger joints (MJCF order, as
observation.state), the plate spots and plate types of that episode (from its saved scene setup; the plates do not move in a
Move slice), which plate satisfied the gate, and whether each hand was being squeezed at that frame (the operator holding the
grip) or had been squeezed at all. Writes one JSON bank that scene_setup's ``bank`` start kinds draw from:

    .venv/bin/python scripts/holomotion_v14_checks/slice_end_bank.py \\
        data/teleop_holomotion_v14_kitchen_a1/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole \\
        --out src/simple/teleop/holomotion_v14/start_banks/a1_end.json

Several dataset dirs can be given (more sessions of the same slice); episodes with ``slice.complete`` false or without an
event frame are skipped (the 1-frame ep0 of the first A1 session, an aborted episode).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+", help="<save-dir>/simple/<env>/level-3_pinhole directories of one recorded slice")
ap.add_argument("--out", required=True); ap.add_argument("--at", choices=("event", "gate", "cut"), default="event")
ap.add_argument("--gate-ahead", type=float, nargs=2, default=(0.15, 0.55), help="the at_counter gate's ahead window (m), to name the target plate")
ap.add_argument("--gate-side", type=float, default=0.30)
ap.add_argument("--no-fk", action="store_true", help="skip the forward kinematics (hands in the pelvis frame; needs the kitchen env)")
a = ap.parse_args()
os.environ.setdefault("MUJOCO_GL", "egl")
import pandas as pd
from simple.teleop.holomotion_v14.grasp_start import hand_curl, hand_kind


def J(v):
    return json.loads(v) if isinstance(v, str) else v


def yaw_of_xyzw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


entries, skipped, slice_id = [], [], None
for D in a.datasets:
    D = Path(D)
    for line in open(D / "meta/episodes.jsonl"):
        e = json.loads(line)
        sl, ss = J(e.get("slice")), J(e.get("scene_setup"))
        k = int(e["episode_index"])
        if not sl or not ss or not sl.get("complete") or sl.get("event_frame") is None:
            skipped.append(f"{D.parents[2].name} ep {k}"); continue
        slice_id = slice_id or sl["id"]
        f = {"event": sl["event_frame"], "gate": sl["gate_frame"], "cut": sl.get("cut_frame") or sl["event_frame"]}[a.at]
        df = pd.read_parquet(next(D.glob(f"data/chunk-*/episode_{k:06d}.parquet")))
        f = min(int(f), len(df) - 1)
        st = np.asarray(df["observation.state"].iloc[f], dtype=float)
        rp = np.asarray(df["states.robot.root_pose"].iloc[f], dtype=float)
        yaw = yaw_of_xyzw(rp[3:7])
        fwd = np.array([math.cos(yaw), math.sin(yaw)]); lft = np.array([-math.sin(yaw), math.cos(yaw)])
        spots = ss["preset"]["plate_spots"]
        rel = {}
        for name, sp in spots.items():
            v = np.asarray(sp[:2]) - rp[:2]
            rel[name] = dict(ahead=round(float(fwd @ v), 4), left=round(float(lft @ v), 4), in_gate=bool(a.gate_ahead[0] <= fwd @ v <= a.gate_ahead[1] and abs(lft @ v) <= a.gate_side))
        target = min(rel, key=lambda nm: (not rel[nm]["in_gate"], math.hypot(rel[nm]["ahead"], rel[nm]["left"])))
        grip = {s: np.asarray(df[f"action.revo2.{s}.grip_mode"].to_numpy(), dtype=float) for s in ("left", "right")}
        cmd = np.asarray(df["action"].iloc[f], dtype=float)
        entries.append(dict(
            episode=k, dataset=str(D.parents[2].name), seed=ss["seed"], frame=f, frames=len(df), at=a.at,
            robot=dict(x=round(float(rp[0]), 4), y=round(float(rp[1]), 4), yaw_deg=round(math.degrees(yaw), 2)), pelvis_z=round(float(rp[2]), 4),
            body=[round(float(v), 4) for v in st[:29]],
            fingers=dict(left=[round(float(v), 4) for v in st[29:40]], right=[round(float(v), 4) for v in st[40:51]]),
            hands=dict(left=dict(kind=hand_kind(st[29:40]), curl=round(hand_curl(st[29:40]), 3), squeezing=bool(hand_curl(cmd[22:33]) > 0.5), squeezed=bool(np.any(grip["left"] > 0))),
                       right=dict(kind=hand_kind(st[40:51]), curl=round(hand_curl(st[40:51]), 3), squeezing=bool(hand_curl(cmd[40:51]) > 0.5), squeezed=bool(np.any(grip["right"] > 0)))),
            plate_spots=spots, plate=ss.get("plate"), plate_b=ss.get("plate_b"), target=target, rel=rel))

entries.sort(key=lambda r: (r["dataset"], r["episode"]))

if not a.no_fk:                                                              # the hands in the pelvis frame (fingertip centroid, hand base) by forward kinematics on the robot model:
    import gymnasium as gym, mujoco                                          # the at_counter start kind keeps a new plate from under a low hand (scene_setup._hands_clear)
    from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene
    mod = load_scene("kitchen"); sc = _load_sonic_config()
    env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, physics_dt=sc["SIMULATE_DT"], headless=True, max_episode_steps=10 ** 9,
                   sonic_config=sc, target=getattr(mod, "TARGET", None), dr_level=3, success_criteria=0.9, robot_uid="g1_sonic_revo2")
    env.reset(); senv = env.unwrapped; m, d = senv.mujoco.mjModel, senv.mujoco.mjData; robot = senv.task.robot
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    PEL = B("pelvis"); ROOTB = int(m.body_rootid[PEL]); ROOTQ = int(m.jnt_qposadr[m.body_jntadr[ROOTB]])
    BQ = m.jnt_qposadr[np.asarray(robot.body_joint_index, dtype=int)]
    TIPS = {s: [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{s}_hand_") and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("_tip_link")] for s in ("left", "right")}
    HANDB = {s: B(f"{s}_hand_base_link") for s in ("left", "right")}
    for e in entries:
        d.qpos[ROOTQ:ROOTQ + 7] = [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]; d.qpos[BQ] = np.asarray(e["body"], dtype=np.float64)
        mujoco.mj_kinematics(m, d)
        R = d.xmat[PEL].reshape(3, 3); p0 = d.xpos[PEL]
        e["hands_pelvis"] = {s: dict(tip=[round(float(v), 4) for v in R.T @ (np.mean([d.xpos[b] for b in TIPS[s]], axis=0) - p0)],
                                     base=[round(float(v), 4) for v in R.T @ (d.xpos[HANDB[s]] - p0)]) for s in ("left", "right")}
    T = np.array([e["hands_pelvis"][s]["tip"] for e in entries for s in ("left", "right")])
    print("hands (fingertip centroid, pelvis frame): ahead %.2f..%.2f, aside %.2f..%.2f, up %.2f..%.2f m" % (T[:, 0].min(), T[:, 0].max(), T[:, 1].min(), T[:, 1].max(), T[:, 2].min(), T[:, 2].max()))

X = np.array([r["robot"]["x"] for r in entries]); Y = np.array([r["robot"]["y"] for r in entries]); W = np.array([r["robot"]["yaw_deg"] for r in entries])
B = np.array([r["body"] for r in entries])
bank = dict(
    meta=dict(slice=slice_id, at=a.at, made=time.strftime("%Y-%m-%d %H:%M"), datasets=[str(Path(d)) for d in a.datasets], n=len(entries), skipped=skipped,
              frame_rule="observation.state at slice.event_frame (the end gate held for its hold time); pelvis = states.robot.root_pose; scene frame = the setups' robot frame",
              body_order="29 body joints in MJCF order: left leg (6), right leg (6), waist yaw/roll/pitch, left arm (7), right arm (7); fingers in grasp_start.FINGER_JOINTS order; hands_pelvis = [ahead, left, up] of the fingertip centroid / hand base in the pelvis frame",
              summary=dict(x=[round(float(X.min()), 3), round(float(X.max()), 3)], y=[round(float(Y.min()), 3), round(float(Y.max()), 3)], yaw_deg=[round(float(W.min()), 1), round(float(W.max()), 1)],
                           body_min=[round(float(v), 3) for v in B.min(0)], body_max=[round(float(v), 3) for v in B.max(0)],
                           ahead_m=[round(min(r["rel"][r["target"]]["ahead"] for r in entries), 3), round(max(r["rel"][r["target"]]["ahead"] for r in entries), 3)],
                           left_m=[round(min(r["rel"][r["target"]]["left"] for r in entries), 3), round(max(r["rel"][r["target"]]["left"] for r in entries), 3)])),
    entries=entries)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(bank, indent=0, separators=(",", ":")))
print(f"{len(entries)} entries ({slice_id}, at {a.at}) -> {out} ({out.stat().st_size / 1e3:.0f} kB); skipped {skipped}")
print("pelvis x %.3f..%.3f  y %.3f..%.3f  yaw %.1f..%.1f deg; target plate ahead %s, left %s" % (X.min(), X.max(), Y.min(), Y.max(), W.min(), W.max(), bank["meta"]["summary"]["ahead_m"], bank["meta"]["summary"]["left_m"]))
print("hands: " + ", ".join(f"{s} squeezing at the end {sum(r['hands'][s]['squeezing'] for r in entries)}, never squeezed {sum(not r['hands'][s]['squeezed'] for r in entries)}" for s in ("left", "right")))
