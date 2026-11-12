#!/usr/bin/env python
"""Grip poses for the official-hand robot (2026-11-04): the legacy data/robots/g1_rot0_teleop_revo2/grip_poses.json (tuned in the
sim on the legacy hand) converted to BrainCo's v1.1.1 joint conventions, so the same physical hand poses come out:
thumb rotation - 3 deg and thumb bend - 9 deg (the official URDF moved those zeros into the joint origins; PIP - 6 deg follows
through the coupling), every distal joint = its finger's official coupling x proximal, clipped to the official ranges.
Checks the fingertips of both robots at each pose and writes <official dir>/grip_poses.json. Re-run after retuning the legacy
poses, or replace the file from the pose tuner run on the official robot.
    .venv/bin/python scripts/robots/convert_grip_poses_official.py
"""
import json, math
from pathlib import Path
import numpy as np, mujoco

ROOT = Path(__file__).resolve().parents[2]; LD = ROOT / "data/robots/g1_rot0_teleop_revo2"; OD = ROOT / "data/robots/g1_rot0_teleop_revo2_official"
F = ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal", "middle_proximal", "middle_distal", "ring_proximal", "ring_distal", "pinky_proximal", "pinky_distal"]
THUMB_SHIFT = (math.radians(3), math.radians(9))        # official zero offsets: abduction, MCP


def hand_info(m, side):
    J = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{f}_joint") for f in F]
    upper = np.array([m.jnt_range[j][1] for j in J]); coup = np.ones(11)
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj1id[e] in J: coup[J.index(int(m.eq_obj1id[e]))] = m.eq_data[e][1]
    return J, upper, coup


def enforce(q, coup):
    q = np.array(q, float)
    for p, d in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)): q[d] = coup[d] * q[p]
    return q


def tips(m, side, q):
    d = mujoco.MjData(m); J, _, coup = hand_info(m, side)
    for j, v in zip(J, enforce(q, coup)): d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link"); R = d.xmat[b].reshape(3, 3); p = d.xpos[b]
    return {f: R.T @ (d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_{f}_tip_link")] - p) for f in ("thumb", "index", "middle", "ring", "pinky")}


def convert(q, upper, coup):
    o = np.clip(np.array(q, float), 0.0, upper); o[0] -= THUMB_SHIFT[0]; o[1] -= THUMB_SHIFT[1]      # proximal values clipped to the official ranges first
    return np.clip(enforce(np.clip(o, 0.0, upper), coup), 0.0, upper)


def main():
    old = mujoco.MjModel.from_xml_path(str(LD / "g1_rot0_teleop_revo2_v3.xml")); new = mujoco.MjModel.from_xml_path(str(OD / "g1_rot0_teleop_revo2_official.xml"))
    leg = json.loads((LD / "grip_poses.json").read_text()); _, up, co = hand_info(new, "right")
    out = {k: convert(leg[k], up, co).round(4).tolist() for k in ("open", "closed", "close2") if k in leg}
    out["threshold"] = leg.get("threshold", 0.5)
    out["note"] = ("converted 2026-11-04 from the legacy grip_poses.json by scripts/robots/convert_grip_poses_official.py: thumb rotation -3 deg, thumb bend -9 deg "
                   "(BrainCo v1.1.1 zero offsets), distal = official coupling x proximal; same fingertip placement as the legacy poses to the mm (see the script's check). "
                   "Not yet re-tuned on this hand (docs/revo2_pose_tuner).")
    out["derived_from"] = str((LD / "grip_poses.json").relative_to(ROOT))
    (OD / "grip_poses.json").write_text(json.dumps(out, indent=1)); print(f"[grip] wrote {OD / 'grip_poses.json'}")
    for k in ("open", "closed", "close2"):
        if k not in leg: continue
        for side in ("right", "left"):
            a = tips(old, side, leg[k]); b = tips(new, side, out[k])
            print(f"  {k:7s} {side:5s} legacy {np.round(np.array(leg[k]), 3).tolist()}\n          -> official {out[k]}\n          fingertip shift legacy->official (mm): " + ", ".join(f"{f} {np.linalg.norm(a[f] - b[f]) * 1000:.1f}" for f in a))


if __name__ == "__main__":
    main()
