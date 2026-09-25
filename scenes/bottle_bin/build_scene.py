#!/usr/bin/env python3
"""Tabletop bottle / trash-bin scene for the g1comp (MuJoCo) -- successor of sim/tabletop_box/build_scene.py.

World frame as in the tabletop_box scene: x from the robot toward the table, z up, the robot's right = -y,
robot pelvis at the origin.  Left / right below are the robot's left / right (it faces the table).
  * table 2.07 m across (facing the robot) x 0.60 m deep, top 0.72 m above the floor, four 4 cm legs
  * cover board along the far edge, standing on the table top, top edge 0.92 m above the floor
    (--cover vertical, default; --cover shelf = a horizontal board at 0.92 m over the far --cover-depth m)
  * 500 ml plastic bottle (6.5 cm dia x 21.2 cm, generated mesh) standing upright, axis 56.4 cm from the
    LEFT table edge and 11.5 cm from the near edge (--bottle-ref near = the bottle's near surface instead)
  * robot g1comp facing the table: front of the feet 17 cm from the near edge (--robot-ref pelvis = the
    pelvis centre instead), pelvis centre 56 cm from the LEFT table edge (--robot-side right for the other
    edge; 2026-09-16: the user confirmed LEFT, so the bottle stands straight ahead of the robot)
  * trash bin 36 x 24 x 39 cm, black, 4 cm corner radius, open top, on the floor 1.53 m to the robot's right and
    0.36 m behind the pelvis start (--bin-xy; 2026-09-23: where a free-walking replay of the 97 kept episodes of 02-25-56 lets go;
    before that it stood 11 cm inside the table's right end, --bin-xy 0 -1.40), turned 90 deg clockwise
    (--bin-yaw -90): the 36 cm side points toward the table, the 24 cm side along the edge
  * pose 3 = the init/home pose of the 2026-09-17 recordings: every upper-body joint 0 except the elbows at -0.66,
    hands open, waist 0, pelvis 0.74 m, legs solved for flat feet (--pose 2 = arms hanging, 1 = Psi0 client pose);
    head servos pan 0 / tilt 10 deg down (114 ticks);
    D455 ego camera 90 deg HFOV, 16:9 (58.7 deg VFOV)

    MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py [--tag _x]

Writes scene.xml (keyframe default_pose), assets/*.obj (bottle, bin), layout.json and renders/*.png
(third-person views, the robot's D455 view, and plan.png = an orthographic top view with the dimensions drawn).
"""
from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
# g1comp robot model: G1_COMP_MJCF env var, else the copy bundled next to this file (robot/), else the holomotion checkout
_ROBOT_CANDIDATES = [os.environ.get("G1_COMP_MJCF"), HERE / "robot" / "g1_comp_45dof.xml",
                     "/home/Horizon/wrk/holomotion_v1.4/holoretarget/assets/unitree_g1/g1_comp_45dof.xml",
                     str(Path(__file__).resolve().parents[2] / "third_party/holomotion/holoretarget/assets/unitree_g1/g1_comp_45dof.xml")]
ROBOT_XML = next(Path(c) for c in _ROBOT_CANDIDATES if c and Path(c).exists())
PORTABLE = ROBOT_XML.is_relative_to(HERE)      # bundled copy -> scene.xml gets paths relative to its own folder

# ---- geometry (metres) ---------------------------------------------------------------
TABLE_W, TABLE_D, TABLE_H = 2.07, 0.60, 0.72      # across (faces the robot), depth, top height
TABLE_TOP_T, LEG_SIZE, LEG_INSET = 0.03, 0.04, 0.04
COVER_TOP_Z, COVER_T = 0.92, 0.018                # cover board: top edge above the floor, thickness
BOTTLE_FROM_LEFT, BOTTLE_FROM_FRONT = 0.564, 0.115  # 0.115 since 2026-09-23: 11.5 cm from the near edge = 27 cm ahead of the
#   pelvis, the centre of the plateau of a 46-point can-position sweep over all 97 kept episodes of session 02-25-56
#   (was 0.09; carried 75 -> 83 of 97).  Do not move it sideways: at the start pose the open thumbs sit at
#   (0.264, +-0.061, 0.888), so a can more than ~5 mm off the centre line spawns inside a thumb and is flung off the table.
ROBOT_TO_EDGE, ROBOT_FROM_EDGE = 0.03, 0.56           # feet -> near edge (spec 0.17; 0.03 since 2026-09-17: replays of session 02-25-56 grasp 7/10 there vs 5/10 at 0.07), pelvis -> side edge (--robot-side)
BIN_W, BIN_D, BIN_H, BIN_R, BIN_T = 0.36, 0.24, 0.39, 0.04, 0.008   # width (along the edge), depth, height, corner radius, wall
BIN_FROM_RIGHT_EDGE = 0.56 - 0.45                 # bin centre inside the table's right end (first build: robot 56 cm from that end, bin 45 cm right of it)
BIN_XY_DEFAULT = (-0.36, -1.53)                   # 2026-09-23 (second fit): the bin must be where a FREE walk ends.  With the bin
#   nearer ((-0.25,-0.84) and (-0.26,-0.80)) every one of the 97 replayed episodes walked into it and was stopped 0.6 m
#   short; the cans that "landed in it" were dropped by a blocked robot.  Replayed with no bin at all, the robot walks
#   0.90-1.31 m to its right, turns 78-111 deg (the recorded headings, tracked to +-3 deg) and lets go at a median
#   (-0.39, -1.40).  A 61-position scan, all 97 episodes at each, puts the best fixed bin at (-0.36, -1.53): 32 cans in
#   with the walk untouched, 12 walks shortened by > 6 cm, no contact longer than 0.5 s.  The remaining cans miss
#   because the operator steered by eye to a bin that was not at the same place relative to every episode's start.
# 500 ml PET bottle: body, shoulder, neck, cap (heights stack from the bottom)
BOTTLE_R, BOTTLE_BODY_H, BOTTLE_SHOULDER_H = 0.0325, 0.145, 0.040
BOTTLE_NECK_R, BOTTLE_NECK_H, BOTTLE_CAP_R, BOTTLE_CAP_H = 0.0125, 0.010, 0.015, 0.017
BOTTLE_H = BOTTLE_BODY_H + BOTTLE_SHOULDER_H + BOTTLE_NECK_H + BOTTLE_CAP_H
BOTTLE_MASS = 0.5                                 # kg: a full 500 ml bottle (the recordings' bottle has water in it; at 0.1 kg the replayed fingers tip it over)
PELVIS_HEIGHT = 0.74

# ---- pose --------------------------------------------------------------------------
ARM_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right")
              for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
POSE_2 = {j: 0.0 for j in ARM_JOINTS}                              # fixpose data start: arms hanging
POSE_2["left_shoulder_roll_joint"], POSE_2["right_shoulder_roll_joint"] = 0.2, -0.2
POSE_1 = dict(POSE_2, left_shoulder_pitch_joint=0.5, left_shoulder_roll_joint=0.0, left_shoulder_yaw_joint=0.2,
              left_elbow_joint=0.3, right_shoulder_pitch_joint=0.5, right_shoulder_roll_joint=0.0,
              right_shoulder_yaw_joint=-0.2, right_elbow_joint=0.3)   # Psi0 client default_dof_pos
POSE_3 = {j: 0.0 for j in ARM_JOINTS}                              # init/home pose of the 2026-09-17 recordings:
POSE_3["left_elbow_joint"] = POSE_3["right_elbow_joint"] = -0.66     # every upper-body joint 0, elbows -0.66, hands open
POSES = {1: POSE_1, 2: POSE_2, 3: POSE_3}
LEGS_WAIST = {"left_hip_pitch_joint": -0.1, "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2,
              "right_hip_pitch_joint": -0.1, "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2,
              "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0}
SERVO_DEG_PER_TICK = 0.088
HEAD_PAN = 0.0                                                    # ID1 2517 ticks = centred
HEAD_TILT_TICKS = 114                                             # ID0 goal 2200 = home 2086 + 114 -> 10 deg down

# teleop stack first_person_camera on d455_link (MuJoCo camera convention)
EGO_CAM_QUAT = "0.64367383 0.26523914 -0.27106013 -0.66472446"
EGO_HFOV = 90.0
EGO_W, EGO_H = 1280, 720                                          # 16:9 -> 58.7 deg VFOV (SIMPLE renders 640x360)
MOUNT_PITCH_AT_ZERO_DEG = 47.7                                    # optical axis below the torso x axis at d455_joint = 0

HAND_RGBA = "0.10 0.10 0.10 1"     # Dex3 hand links (the robot file has them light grey)
FLOOR_RGBA = "0.82 0.82 0.82 1"    # plain light grey floor (was a checker texture)
DIM = (183, 55, 31)          # dimension lines on the plan
EXT = (120, 120, 120)        # extension lines


# ---- generated meshes -----------------------------------------------------------------
def _write_obj(path: Path, verts, tris, comment=""):
    lines = [f"# {comment}"] + [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["f %d %d %d" % (a + 1, b + 1, c + 1) for a, b, c in tris]
    path.write_text("\n".join(lines) + "\n")


def revolve(profile, n=48):
    """Closed surface of revolution about z from a (r, z) profile that starts and ends on the axis (r = 0)."""
    verts, tris = [], []
    rings = []
    for r, z in profile:
        if r <= 0:
            rings.append([len(verts)]); verts.append((0.0, 0.0, z))
        else:
            ring = list(range(len(verts), len(verts) + n))
            verts += [(r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n), z) for k in range(n)]
            rings.append(ring)
    for lo, hi in zip(rings, rings[1:]):
        for k in range(n):
            k1 = (k + 1) % n
            if len(lo) == 1:                     # bottom pole, faces down
                tris.append((lo[0], hi[k1], hi[k]))
            elif len(hi) == 1:                   # top pole, faces up
                tris.append((hi[0], lo[k], lo[k1]))
            else:
                tris += [(lo[k], lo[k1], hi[k1]), (lo[k], hi[k1], hi[k])]
    return verts, tris


def bottle_meshes(adir: Path):
    z0 = BOTTLE_BODY_H
    shoulder = [(BOTTLE_NECK_R + (BOTTLE_R - BOTTLE_NECK_R) * np.sqrt(max(0.0, 1 - t * t)), z0 + t * BOTTLE_SHOULDER_H)
                for t in np.linspace(0, 1, 9)[1:]]
    z1 = z0 + BOTTLE_SHOULDER_H
    body = [(0, 0), (BOTTLE_R, 0), (BOTTLE_R, z0), *shoulder, (BOTTLE_NECK_R, z1 + BOTTLE_NECK_H), (0, z1 + BOTTLE_NECK_H)]
    z2 = z1 + BOTTLE_NECK_H
    cap = [(0, z2), (BOTTLE_CAP_R, z2), (BOTTLE_CAP_R, z2 + BOTTLE_CAP_H), (0, z2 + BOTTLE_CAP_H)]
    pb = adir / "bottle_500ml_body.obj"; _write_obj(pb, *revolve(body), comment="500 ml bottle body, bottom at z=0")
    pc = adir / "bottle_500ml_cap.obj"; _write_obj(pc, *revolve(cap), comment="500 ml bottle cap")
    return pb, pc


def rounded_rect(w, d, r, n=8):
    """CCW loop (x, y): w along x, d along y, corner radius r, n segments per corner."""
    hw, hd = w / 2, d / 2
    pts = []
    for cx, cy, a0 in ((hw - r, hd - r, 0), (-hw + r, hd - r, 90), (-hw + r, -hd + r, 180), (hw - r, -hd + r, 270)):
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / n)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def bin_mesh(adir: Path):
    """Open-top hollow bin with rounded vertical edges; local x = depth, y = width, bottom at z = 0."""
    P, Q = rounded_rect(BIN_D, BIN_W, BIN_R), rounded_rect(BIN_D - 2 * BIN_T, BIN_W - 2 * BIN_T, BIN_R - BIN_T)
    M = len(P)
    verts = [(x, y, 0.0) for x, y in P] + [(x, y, BIN_H) for x, y in P] + [(x, y, BIN_H) for x, y in Q] + [(x, y, BIN_T) for x, y in Q]
    ob, ot, it, ib = 0, M, 2 * M, 3 * M
    c0, c1 = len(verts), len(verts) + 1
    verts += [(0.0, 0.0, 0.0), (0.0, 0.0, BIN_T)]
    tris = []
    for i in range(M):
        j = (i + 1) % M
        tris += [(ob + i, ob + j, ot + j), (ob + i, ot + j, ot + i)]        # outer wall
        tris += [(ib + i, it + j, ib + j), (ib + i, it + i, it + j)]        # inner wall (faces inward)
        tris += [(ot + i, ot + j, it + j), (ot + i, it + j, it + i)]        # rim
        tris += [(c0, ob + j, ob + i), (c1, ib + i, ib + j)]                # outer bottom (down), inner bottom (up)
    p = adir / "bin_36x24x39.obj"
    _write_obj(p, verts, tris, comment="trash bin 36x24x39 cm, corner radius 4 cm, open top")
    return p


def bin_collision_geoms(parent, material):
    """Bottom slab, four straight walls and short boxes along each rounded corner (group 5 = hidden)."""
    common = dict(type="box", group="5", material=material, friction="1 0.005 0.0001")
    hw, hd, r, t, h = BIN_W / 2, BIN_D / 2, BIN_R, BIN_T, BIN_H
    ET.SubElement(parent, "geom", name="bin_bottom", size=f"{hd:.4f} {hw:.4f} {t/2:.4f}", pos=f"0 0 {t/2:.4f}", **common)
    # straight walls (local x = depth, y = width); the corners are covered separately
    for name, size, pos in (("bin_wall_front", (t / 2, hw - r, h / 2), (hd - t / 2, 0, h / 2)),
                            ("bin_wall_back", (t / 2, hw - r, h / 2), (-(hd - t / 2), 0, h / 2)),
                            ("bin_wall_left", (hd - r, t / 2, h / 2), (0, hw - t / 2, h / 2)),
                            ("bin_wall_right", (hd - r, t / 2, h / 2), (0, -(hw - t / 2), h / 2))):
        ET.SubElement(parent, "geom", name=name, size=" ".join(f"{v:.4f}" for v in size), pos=" ".join(f"{v:.4f}" for v in pos), **common)
    nseg = 4
    for ci, (cx, cy, a0) in enumerate(((hd - r, hw - r, 0), (-hd + r, hw - r, 90), (-hd + r, -hw + r, 180), (hd - r, -hw + r, 270))):
        for k in range(nseg):
            a = np.radians(a0 + 90 * (k + 0.5) / nseg)
            half = np.radians(90 / nseg / 2)
            rm = r - t / 2
            px, py = cx + rm * np.cos(a), cy + rm * np.sin(a)
            q = R.from_euler("z", a + np.pi / 2).as_quat()          # box long axis tangent to the arc
            ET.SubElement(parent, "geom", name=f"bin_corner{ci}_{k}",
                          size=f"{rm*np.sin(half)*1.05:.4f} {t/2:.4f} {h/2:.4f}", pos=f"{px:.4f} {py:.4f} {h/2:.4f}",
                          quat=f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}", **common)


# ---- scene ---------------------------------------------------------------------------
def cam_xyaxes(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return " ".join(f"{v:.5f}" for v in (*r, *u))


def ego_cam_pos_link() -> np.ndarray:
    """Camera origin in the d455_link frame: on the front face of the D455 mesh along the optical axis (+2 mm)."""
    w, x, y, z = (float(v) for v in EGO_CAM_QUAT.split())
    axis = R.from_quat([x, y, z, w]).apply([0, 0, -1.0])
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "d455_link")
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return axis * (float((v @ axis).max()) + 0.002)


def solve_legs(model, q_named):
    """Hip pitch / knee so the pelvis at PELVIS_HEIGHT has flat feet on the floor, the foot under the hip."""
    data = mujoco.MjData(model)
    jid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}

    def set_q(q):
        for n, v in q.items():
            data.qpos[model.jnt_qposadr[jid[n]]] = v

    foot = [g for g in range(model.ngeom)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) == "left_ankle_roll_link"
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]

    def lowest_and_x(q):
        set_q(q); mujoco.mj_kinematics(model, data)
        return min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in foot), float(np.mean([data.geom_xpos[g][0] for g in foot]))

    base = dict(q_named)
    straight = dict(base, **{f"{s}_{j}_joint": 0.0 for s in ("left", "right") for j in ("hip_pitch", "knee", "ankle_pitch")})
    _, x_ref = lowest_and_x(straight)

    def resid(v):
        h, k = v
        q = dict(base)
        for s in ("left", "right"):
            q[f"{s}_hip_pitch_joint"] = h; q[f"{s}_knee_joint"] = k; q[f"{s}_ankle_pitch_joint"] = -(h + k)
        z, x = lowest_and_x(q)
        return [z, x - x_ref]

    h, k = fsolve(resid, [-0.25, 0.45])
    out = dict(base)
    for s in ("left", "right"):
        out[f"{s}_hip_pitch_joint"] = float(h); out[f"{s}_knee_joint"] = float(k); out[f"{s}_ankle_pitch_joint"] = float(-(h + k))
    return out


def toe_front_x(model, data):
    """Front of the feet relative to the pelvis (sphere foot geoms on both ankle_roll links)."""
    xs = []
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if b in ("left_ankle_roll_link", "right_ankle_roll_link") and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            xs.append(data.geom_xpos[g][0] + model.geom_size[g][0])
    return max(xs) - data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][0]


def robot_pose(args):
    """Joint dict for the requested pose (legs solved on the robot file alone) + the toe offset."""
    q = dict(LEGS_WAIST, **POSES[args.pose])
    q["xl330_joint"] = HEAD_PAN
    q["d455_joint"] = np.radians(args.tilt_ticks * SERVO_DEG_PER_TICK)
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    tree = ET.parse(ROBOT_XML)
    pelvis_z = float(tree.getroot().find("./worldbody/body[@name='pelvis']").get("pos", "0 0 0").split()[2])
    d = mujoco.MjData(m)
    d.qpos[2] = PELVIS_HEIGHT                     # the robot file's own pelvis height differs
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
    global _PELVIS_QPOS_Z
    q = solve_legs_at(m, q, jid)
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[jid[n]]] = v
    mujoco.mj_kinematics(m, d)
    return q, toe_front_x(m, d), pelvis_z


def solve_legs_at(model, q_named, jid):
    """solve_legs with the pelvis free joint set to PELVIS_HEIGHT (the robot file's default may differ)."""
    orig = model.qpos0.copy()
    model.qpos0[2] = PELVIS_HEIGHT
    try:
        return solve_legs(model, q_named)
    finally:
        model.qpos0[:] = orig


def build_xml(args, out: Path, layout: dict, cam_pos, fovy) -> Path:
    adir = HERE / "assets"; adir.mkdir(exist_ok=True)
    body_obj, cap_obj = bottle_meshes(adir)
    bin_obj = bin_mesh(adir)

    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    root.set("model", "g1comp_tabletop_bottle_bin")
    comp = root.find("compiler")
    comp.set("angle", "radian")
    if PORTABLE:                       # meshdir = this folder; robot meshes under robot/meshes/, generated meshes under assets/
        comp.set("meshdir", ".")
        for m in root.find("asset").findall("mesh"):
            m.set("file", str(Path("robot") / "meshes" / Path(m.get("file")).name))
        mesh_ref = lambda p: str(Path(p).resolve().relative_to(HERE))
    else:
        comp.set("meshdir", str(ROBOT_XML.parent / "meshes"))
        mesh_ref = lambda p: str(p)

    d455 = root.find(".//body[@name='d455_link']")
    ET.SubElement(d455, "camera", name="ego", pos=" ".join(f"{v:.5f}" for v in cam_pos), quat=EGO_CAM_QUAT, fovy=f"{fovy:.3f}")
    w, x, y, z = (float(v) for v in EGO_CAM_QUAT.split())
    ax = R.from_quat([x, y, z, w]).apply([0, 0, -1.0])
    axis_len = 0.6
    mid = np.asarray(cam_pos) + ax * axis_len / 2
    ET.SubElement(d455, "geom", name="ego_axis", type="cylinder", size=f"0.003 {axis_len/2}",
                  pos=" ".join(f"{v:.5f}" for v in mid), quat=EGO_CAM_QUAT, rgba="0 0.8 0 1",
                  contype="0", conaffinity="0", group="4", mass="0")
    for bname in ("head_servo_link", "xl330_link", "d455_link"):
        for g in root.find(f".//body[@name='{bname}']").findall("geom"):
            if g.get("name") != "ego_axis":
                g.set("group", "3")
    for g in root.iter("geom"):
        if g.get("mesh") == "head_link":
            g.set("group", "3")
        if (g.get("mesh") or "").startswith(("left_hand_", "right_hand_")):      # Dex3 hands: black (user, 2026-09-16)
            g.set("rgba", HAND_RGBA)
    vis = root.find("visual") or ET.SubElement(root, "visual")
    ET.SubElement(vis, "quality", shadowsize="4096")
    glob = vis.find("global")
    if glob is not None:
        glob.set("offwidth", "2080"); glob.set("offheight", "1600")

    root.find("./worldbody/body[@name='pelvis']").set("pos", f"0 0 {PELVIS_HEIGHT}")
    wb = root.find("worldbody")
    for w_ in root.findall("worldbody"):
        for el in list(w_):
            if el.tag in ("geom", "light"):
                w_.remove(el)
    ET.SubElement(wb, "light", pos="0.4 0.5 2.6", dir="0 0 -1", directional="false", diffuse="0.75 0.75 0.75", specular="0.2 0.2 0.2")
    ET.SubElement(wb, "light", pos="-1.5 -1.0 2.2", dir="0.5 0.4 -0.75", directional="true", diffuse="0.4 0.4 0.4")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="5 5 0.05", material="floor_mat", friction="1 0.005 0.0001")

    L = layout
    table = ET.SubElement(wb, "body", name="table", pos=f"{L['table_cx']:.4f} {L['table_cy']:.4f} 0")
    ET.SubElement(table, "geom", name="table_top", type="box",
                  size=f"{TABLE_D/2} {TABLE_W/2} {TABLE_TOP_T/2}", pos=f"0 0 {TABLE_H - TABLE_TOP_T/2:.4f}",
                  material="table_mat", friction="1 0.005 0.0001")
    leg_h = TABLE_H - TABLE_TOP_T
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        ET.SubElement(table, "geom", name=f"table_leg{i}", type="box", size=f"{LEG_SIZE/2} {LEG_SIZE/2} {leg_h/2:.4f}",
                      pos=f"{sx*(TABLE_D/2-LEG_INSET-LEG_SIZE/2):.4f} {sy*(TABLE_W/2-LEG_INSET-LEG_SIZE/2):.4f} {leg_h/2:.4f}", material="leg_mat")
    if args.cover == "vertical":       # board standing on the slab along the far edge
        ET.SubElement(table, "geom", name="cover_board", type="box",
                      size=f"{COVER_T/2} {TABLE_W/2} {(args.cover_top - TABLE_H)/2:.4f}",
                      pos=f"{TABLE_D/2 - COVER_T/2:.4f} 0 {(TABLE_H + args.cover_top)/2:.4f}", material="board_mat")
    else:                              # horizontal board at cover_top over the far cover_depth, on two end panels
        cd = args.cover_depth
        ET.SubElement(table, "geom", name="cover_board", type="box",
                      size=f"{cd/2:.4f} {TABLE_W/2} {COVER_T/2}",
                      pos=f"{TABLE_D/2 - cd/2:.4f} 0 {args.cover_top - COVER_T/2:.4f}", material="board_mat")
        for i, sy in enumerate((1, -1)):
            ET.SubElement(table, "geom", name=f"cover_end{i}", type="box",
                          size=f"{cd/2:.4f} {COVER_T/2} {(args.cover_top - COVER_T - TABLE_H)/2:.4f}",
                          pos=f"{TABLE_D/2 - cd/2:.4f} {sy*(TABLE_W/2 - COVER_T/2):.4f} {(TABLE_H + args.cover_top - COVER_T)/2:.4f}", material="board_mat")

    bottle = ET.SubElement(wb, "body", name="bottle", pos=f"{L['bottle_x']:.4f} {L['bottle_y']:.4f} {TABLE_H:.4f}")
    ET.SubElement(bottle, "freejoint", name="bottle")
    ET.SubElement(bottle, "geom", name="bottle_body", type="mesh", mesh="bottle_body", material="bottle_mat",
                  mass=f"{args.bottle_mass*0.95:.4f}", friction="1 0.005 0.0001")
    ET.SubElement(bottle, "geom", name="bottle_cap", type="mesh", mesh="bottle_cap", material="cap_mat",
                  mass=f"{args.bottle_mass*0.05:.4f}", friction="1 0.005 0.0001")

    bq = R.from_euler("z", args.bin_yaw, degrees=True).as_quat()      # mesh: local x = 24 cm depth, y = 36 cm width
    bin_ = ET.SubElement(wb, "body", name="trash_bin", pos=f"{L['bin_x']:.4f} {L['bin_y']:.4f} 0",
                         quat=f"{bq[3]:.6f} {bq[0]:.6f} {bq[1]:.6f} {bq[2]:.6f}")
    ET.SubElement(bin_, "geom", name="bin_visual", type="mesh", mesh="bin_shell", material="bin_mat",
                  contype="0", conaffinity="0", group="1")
    bin_collision_geoms(bin_, "bin_mat")

    # third-person cameras (computed from a position and a look-at point)
    tcy = L["table_cy"]
    cams = {
        "overview": ((-2.6, tcy + 1.45, 2.3), (0.45, tcy - 0.1, 0.55)),
        "front_left": ((2.7, tcy + 1.85, 1.8), (0.25, tcy - 0.1, 0.6)),
        "front_right": ((2.5, tcy - 2.55, 1.7), (0.25, tcy - 0.2, 0.6)),
        "side": ((0.35, tcy - 4.0, 1.15), (0.35, tcy - 0.25, 0.62)),
        "bottle_closeup": ((L['bottle_x'] + 0.30, L['bottle_y'] + 0.62, 1.05), (L['bottle_x'], L['bottle_y'], 0.8)),
        "bin_closeup": ((L['bin_x'] - 0.75, L['bin_y'] - 0.7, 0.85), (L['bin_x'], L['bin_y'], 0.2)),
    }
    for name, (pos, tgt) in cams.items():
        ET.SubElement(wb, "camera", name=name, pos=" ".join(f"{v:.3f}" for v in pos), xyaxes=cam_xyaxes(pos, tgt))
    ET.SubElement(wb, "camera", name="plan", pos=f"{L['plan_cx']:.3f} {L['plan_cy']:.3f} 4", xyaxes="0 -1 0 1 0 0",
                  orthographic="true", fovy=f"{L['plan_extent']:.3f}")

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", name="bottle_body", file=mesh_ref(body_obj))
    ET.SubElement(asset, "mesh", name="bottle_cap", file=mesh_ref(cap_obj))
    ET.SubElement(asset, "mesh", name="bin_shell", file=mesh_ref(bin_obj))
    ET.SubElement(asset, "material", name="floor_mat", rgba=FLOOR_RGBA, reflectance="0.08", specular="0.1", shininess="0.1")
    ET.SubElement(asset, "material", name="table_mat", rgba="0.75 0.6 0.42 1")
    ET.SubElement(asset, "material", name="leg_mat", rgba="0.35 0.3 0.25 1")
    ET.SubElement(asset, "material", name="board_mat", rgba="0.88 0.86 0.80 1")
    ET.SubElement(asset, "material", name="bin_mat", rgba="0.06 0.06 0.06 1", specular="0.6", shininess="0.5")
    ET.SubElement(asset, "material", name="bottle_mat", rgba="0.62 0.80 0.95 0.55", specular="0.8", shininess="0.7")
    ET.SubElement(asset, "material", name="cap_mat", rgba="0.15 0.35 0.85 1")

    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode")
    return out


def compute_layout(args, toe_off):
    pelvis_to_edge = args.robot_to_edge + (toe_off if args.robot_ref == "toes" else 0.0)
    near_x = pelvis_to_edge
    if args.robot_side == "left":
        left_y = ROBOT_FROM_EDGE; right_y = left_y - TABLE_W
    else:
        right_y = -ROBOT_FROM_EDGE; left_y = right_y + TABLE_W
    bx = near_x + BOTTLE_FROM_FRONT + (BOTTLE_R if args.bottle_ref == "near" else 0.0)
    by = left_y - BOTTLE_FROM_LEFT if args.bottle_side == "left" else right_y + BOTTLE_FROM_LEFT
    L = dict(pelvis_to_edge=pelvis_to_edge, toe_offset=toe_off, near_x=near_x, far_x=near_x + TABLE_D,
             left_y=left_y, right_y=right_y, table_cx=near_x + TABLE_D / 2, table_cy=(left_y + right_y) / 2,
             bottle_x=bx, bottle_y=by, bottle_side=args.bottle_side, robot_side=args.robot_side,
             bin_x=(args.bin_xy[0] if args.bin_xy else 0.0),
             bin_y=(args.bin_xy[1] if args.bin_xy else (-args.bin_right_of_robot if args.bin_right_of_robot is not None else right_y + BIN_FROM_RIGHT_EDGE)),
             plan_cx=0.30, plan_cy=(left_y + min(right_y, args.bin_xy[1] if args.bin_xy else right_y) - 0.30) / 2, plan_extent=2.1, plan_w=1400, plan_h=1000)
    return L


def draw_plan(png: Path, L: dict, out: Path):
    im = Image.open(png).convert("RGB"); d = ImageDraw.Draw(im)
    W, H = im.size; s = H / L["plan_extent"]; cx, cy = L["plan_cx"], L["plan_cy"]
    P = lambda x, y: (W / 2 - (y - cy) * s, H / 2 - (x - cx) * s)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()

    def label(xy, text, color=DIM, f=None, anchor="mm"):
        f = f or font
        l, t, r, b = d.textbbox(xy, text, font=f, anchor=anchor)
        d.rectangle([l - 4, t - 2, r + 4, b + 2], fill=(255, 255, 255), outline=color)
        d.text(xy, text, fill=color, font=f, anchor=anchor)

    def ext(p0, p1):
        d.line([P(*p0), P(*p1)], fill=EXT, width=1)

    def dim(p0, p1, text, off=(0, 0), color=DIM):
        a, b = np.array(P(*p0)), np.array(P(*p1))
        d.line([tuple(a), tuple(b)], fill=color, width=3)
        n = np.array([-(b - a)[1], (b - a)[0]]); n = n / (np.linalg.norm(n) + 1e-9) * 7
        for p in (a, b):
            d.line([tuple(p - n), tuple(p + n)], fill=color, width=3)
        m = (a + b) / 2 + np.array(off)
        label(tuple(m), text, color)

    nx, fx, ly, ry = L["near_x"], L["far_x"], L["left_y"], L["right_y"]
    bx, by = L["bottle_x"], L["bottle_y"]
    toe_x = L["toe_offset"]
    # bottle
    bs, rs = L["bottle_side"], L["robot_side"]
    ey = ly if bs == "left" else ry
    ext((bx, ey), (bx, ey + (0.02 if bs == "left" else -0.02)))
    dim((bx, ey), (bx, by), f"56.4 cm from {bs} edge", off=(0, -26))
    ext((nx - 0.02, by), (bx + 0.02, by)); dim((nx, by), (bx, by), f"{(bx - nx) * 100:.1f} cm", off=(-70, 10))
    # robot vs its side edge / bin / near edge
    rey = ly if rs == "left" else ry
    ext((nx, rey), (-0.5, rey)); ext((toe_x, 0), (-0.5, 0))
    dim((-0.44, rey), (-0.44, 0.0), f"56 cm from {rs} edge", off=(0, 24))
    bxx, byy = L["bin_x"], L["bin_y"]
    dim((bxx - 0.30, 0.0), (bxx - 0.30, byy), f"{abs(byy)*100:.0f} cm right of the pelvis", off=(0, -22))
    ext((0.0, 0.0), (bxx - 0.32, 0.0)); ext((bxx, byy), (bxx - 0.32, byy))
    if abs(bxx) > 0.02:
        ext((0.0, 0.0), (0.0, byy - 0.30)); ext((bxx, byy), (bxx, byy - 0.30))
        dim((0.0, byy - 0.26), (bxx, byy - 0.26), f"{abs(bxx)*100:.0f} cm {'behind' if bxx < 0 else 'ahead of'} the pelvis", off=(0, 24))
    elif L["robot_side"] == "left":
        ext((bxx, ry), (-0.5, ry)); dim((-0.44, ry), (-0.44, byy), f"{(byy - ry)*100:.0f} cm", off=(0, 24))
    ext((toe_x, 0.14), (toe_x, 0.30)); ext((nx, 0.14), (nx, 0.30))
    dim((toe_x, 0.26), (nx, 0.26), f"{(nx - toe_x)*100:.0f} cm feet to edge", off=(-95, 0))
    # table
    ext((fx, ly), (fx + 0.1, ly)); ext((fx, ry), (fx + 0.1, ry))
    dim((fx + 0.08, ry), (fx + 0.08, ly), "table 2.07 m x 0.60 m, top 0.72 m", off=(0, -22))
    dy = (ry - 0.08) if rs == "left" else (ly + 0.08)          # depth dimension at the end away from the robot
    ext((nx, dy), (nx, dy + (0.08 if rs == "right" else -0.08))); ext((fx, dy), (fx, dy + (0.08 if rs == "right" else -0.08)))
    dim((nx, dy), (fx, dy), "0.60 m", off=(-58 if rs == "left" else 58, 0))
    label(P(fx - 0.01, L["table_cy"] - 0.35), "cover board along the far edge, top at 0.92 m", (60, 60, 60), small)
    label(P(L["bin_x"] - 0.22, L["bin_y"]), "bin 36 x 24 x 39 cm", (60, 60, 60), small)
    label(P(bx + 0.11, by), "500 ml bottle", (60, 60, 60), small)
    label(P(-0.62, 0.0), "g1comp, facing the table (+x)", (60, 60, 60), small)
    label((W - 12, H - 12), "plan view from above, robot's right = image right", (60, 60, 60), small, anchor="rd")
    label((12, 12), f"{1/s*100:.3f} cm / px", (60, 60, 60), small, anchor="la")
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scene.xml"))
    ap.add_argument("--renders", default=str(HERE / "renders"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--pose", type=int, default=3, choices=(1, 2, 3), help="3 = elbows raised -0.66, all else 0 (2026-09-17 recordings, default); 2 = arms hanging (fixpose start); 1 = Psi0 client pose")
    ap.add_argument("--tilt-ticks", type=float, default=HEAD_TILT_TICKS, help="D455 tilt servo from home in ticks (0.088 deg/tick, + = down)")
    ap.add_argument("--robot-ref", default="toes", choices=("toes", "pelvis"), help="what --robot-to-edge measures from")
    ap.add_argument("--robot-to-edge", type=float, default=ROBOT_TO_EDGE, help="front of the feet (or pelvis) -> near table edge (m); spec 0.17")
    ap.add_argument("--bottle-mass", type=float, default=BOTTLE_MASS, help="bottle mass (kg): 0.025 empty, ~0.5 full")
    ap.add_argument("--bottle-ref", default="center", choices=("center", "near"), help="what BOTTLE_FROM_FRONT (11.5 cm) measures to")
    ap.add_argument("--bottle-side", default="left", choices=("left", "right"), help="which table edge the 56.4 cm is measured from (robot's left/right)")
    ap.add_argument("--robot-side", default="left", choices=("left", "right"), help="which table edge the robot's 56 cm is measured from (robot's left/right)")
    ap.add_argument("--bin-right-of-robot", type=float, default=None, help="place the bin centre this far to the robot's right instead of 11 cm inside the table's right end")
    ap.add_argument("--bin-xy", type=float, nargs=2, default=list(BIN_XY_DEFAULT), metavar=("X", "Y"), help="bin centre in the pelvis frame (x toward the table, robot's right = -y); overrides the other bin options")
    ap.add_argument("--bin-yaw", type=float, default=-90.0, help="bin yaw in degrees, + = counter-clockwise from above; 0 = 36 cm side along the table edge, -90 = 36 cm side toward the table")
    ap.add_argument("--cover", default="vertical", choices=("vertical", "shelf"))
    ap.add_argument("--cover-top", type=float, default=COVER_TOP_Z, help="top of the cover board above the floor")
    ap.add_argument("--cover-depth", type=float, default=0.30, help="shelf mode: how much of the far side it covers")
    ap.add_argument("--ego-size", default=f"{EGO_W}x{EGO_H}")
    ap.add_argument("--settle", type=int, default=0, help="physics steps with the robot held (bottle settles on the slab)")
    args = ap.parse_args()

    ego_w, ego_h = (int(v) for v in args.ego_size.lower().split("x"))
    fovy = float(np.degrees(2 * np.arctan(np.tan(np.radians(EGO_HFOV / 2)) * ego_h / ego_w)))
    cam_pos = ego_cam_pos_link()

    q, toe_off, _ = robot_pose(args)
    L = compute_layout(args, toe_off)
    xml = build_xml(args, Path(args.out), L, cam_pos, fovy)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    jid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}
    for n, v in q.items():
        data.qpos[model.jnt_qposadr[jid[n]]] = v
    mujoco.mj_forward(model, data)
    if args.settle:
        hold = data.qpos.copy(); nq_bottle = model.jnt_qposadr[jid["bottle"]]
        for _ in range(args.settle):
            mujoco.mj_step(model, data)
            b = data.qpos[nq_bottle:nq_bottle + 7].copy(); data.qpos[:] = hold; data.qpos[nq_bottle:nq_bottle + 7] = b
            data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    tree = ET.parse(xml); root = tree.getroot()
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="default_pose", qpos=" ".join(f"{v:.6g}" for v in data.qpos))
    ET.indent(tree, space="  "); tree.write(xml, encoding="unicode")

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    cpos = data.cam_xpos[cid]; caxis = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    pitch = float(np.degrees(np.arctan2(-caxis[2], np.hypot(caxis[0], caxis[1]))))
    report = dict(layout={k: (round(v, 4) if isinstance(v, float) else v) for k, v in L.items()},
                  pose=dict(pose=args.pose, pelvis_height=PELVIS_HEIGHT, hip_pitch=q["left_hip_pitch_joint"], knee=q["left_knee_joint"],
                            ankle_pitch=q["left_ankle_pitch_joint"], head_pan_deg=0.0, head_tilt_deg=float(np.degrees(q["d455_joint"]))),
                  ego_camera=dict(pos=np.round(cpos, 4).tolist(), axis=np.round(caxis, 4).tolist(), pitch_down_deg=round(pitch, 2),
                                  hfov_deg=EGO_HFOV, fovy_deg=round(fovy, 2), size=[ego_w, ego_h]),
                  bodies={n: np.round(data.xpos[bid(n)], 4).tolist() for n in ("pelvis", "torso_link", "bottle", "trash_bin",
                                                                              "left_wrist_yaw_link", "right_wrist_yaw_link",
                                                                              "left_hand_index_1_link", "right_hand_index_1_link")},
                  bottle=dict(diameter=2 * BOTTLE_R, height=round(BOTTLE_H, 4), mass=args.bottle_mass),
                  bin=dict(w=BIN_W, d=BIN_D, h=BIN_H, corner_radius=BIN_R, wall=BIN_T, yaw_deg=args.bin_yaw,
                           along_x=BIN_W if abs(args.bin_yaw) % 180 == 90 else BIN_D, along_y=BIN_D if abs(args.bin_yaw) % 180 == 90 else BIN_W))
    # contacts other than feet-floor (there should be none: hands clear of the slab, bottle on the table only)
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])}:geom{g}"
    contacts = sorted({(gname(c.geom1), gname(c.geom2)) for c in data.contact[:data.ncon]})
    report["contacts"] = [f"{a} <-> {b}" for a, b in contacts if not ({a, b} & {"floor"} and ("ankle_roll" in a or "ankle_roll" in b))]
    (HERE / f"layout{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    rdir = Path(args.renders); rdir.mkdir(parents=True, exist_ok=True)
    opt = mujoco.MjvOption(); opt.geomgroup[3] = 1                   # third-person views show the head chain
    opt_axis = mujoco.MjvOption(); opt_axis.geomgroup[3] = 1; opt_axis.geomgroup[4] = 1   # + the optical-axis line
    ren = mujoco.Renderer(model, 900, 1600)
    for cam in ("overview", "front_left", "front_right", "side", "bottle_closeup", "bin_closeup"):
        ren.update_scene(data, camera=cam, scene_option=opt_axis if cam in ("overview", "front_left", "front_right", "side") else opt)
        Image.fromarray(ren.render()).save(rdir / f"{cam}{args.tag}.png")
    ren.close()
    ren = mujoco.Renderer(model, L["plan_h"], L["plan_w"])
    ren.update_scene(data, camera="plan", scene_option=opt)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    raw = rdir / f"plan_raw{args.tag}.png"
    Image.fromarray(ren.render()).save(raw); draw_plan(raw, L, rdir / f"plan{args.tag}.png"); raw.unlink()
    ren.close()
    ren = mujoco.Renderer(model, ego_h, ego_w)
    opt_ego = mujoco.MjvOption(); opt_ego.geomgroup[3] = 0             # hide the head chain from its own camera
    ren.update_scene(data, camera="ego", scene_option=opt_ego); Image.fromarray(ren.render()).save(rdir / f"ego_d455{args.tag}.png")
    ren.close()
    print("wrote", xml, "and renders to", rdir)


if __name__ == "__main__":
    main()
