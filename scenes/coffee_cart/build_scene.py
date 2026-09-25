#!/usr/bin/env python3
"""Serving-coffee-with-a-cart scene for the g1comp (MuJoCo), from the hand drawing ~/Downloads/servingcoffeewithcart.jpg.

Numbers on the sheet are cm unless it says mm; the "220mm" of the route is 220 cm (the same cm/mm slip as the bowl-sink
sheet), while "handle 30mm" and "cart wheel 120mm" really are mm (the sheet crosses out "30cm" and writes "30mm").

World frame: pelvis at the origin at the start pose, x = the way the robot faces (up the drawing), z up, robot's right = -y.
  * route (drawing "220" then "67"): the robot starts 2.20 m back from the table's FAR edge, on a line 0.67 m to the LEFT
    of the table's left end -- so it pushes the cart forward and then over to its right to reach the table
  * table 1.06 m across (facing the robot) x 0.60 m deep, top 0.74 m: drawing "106 x 60 x 74"
  * cart: 0.71 (along the push direction) x 0.47 m deck on 0.12 m casters, a 0.60 x 0.35 x 0.68 m box standing on the deck
    with 0.09 m of bare deck left at the handle end, a 30 mm push handle across that end
  * coffee cup 9 cm diameter x 14 cm high, standing on the box top (0.80 m), centred in the 10 x 10 cm square at the
    box top's near-right corner (drawing: the circle by the near edge, "10cm x 10cm at corner")
  * the robot starts 0.10 m behind the handle (drawing "10cm Robot")
  * pose 3 (elbows -0.66, everything else 0), head servo tilt 10 deg, D455 ego camera 90 deg HFOV, 16:9 -- as bottle_bin

    MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py [--tag _x]
Writes scene.xml (keyframe default_pose, paths relative to this folder), assets/*.obj, layout.json and renders/*.png
(third-person views, the robot's D455 view, plan.png = orthographic top view with the drawing's dimensions).
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
TABLE_W, TABLE_D, TABLE_H = 1.06, 0.60, 0.74             # drawing "106 x 60 x 74": across (y), deep (x), top height
TABLE_TOP_T = 0.03                                       # top slab; four legs below it
TABLE_LEG = 0.06
ROUTE_FORWARD = 2.80                                     # tuned by the replay sweep (README: route 2.8 / 0.4); the drawing said 2.20
TABLE_FROM_LINE = 0.40                                   # tuned by the replay sweep; the drawing said 0.67 (robot path line -> the table LEFT end, to the right = -y)

CART_L, CART_W = 0.71, 0.47                              # drawing "cart 71 x 47": along the push direction (x), across (y)
DECK_T = 0.02
CASTER_H = 0.12                                          # drawing "cart wheel: 120mm high" = floor -> deck TOP (what a caster is spec'd by)
DECK_TOP = CASTER_H                                      # deck slab spans DECK_TOP - DECK_T .. DECK_TOP
WHEEL_DIA = DECK_TOP - DECK_T                            # the wheel itself ends at the deck's underside: nothing intersects
BOX_L, BOX_W, BOX_H = 0.60, 0.35, 0.68                   # drawing "box 60 x 35 x 68"; top at DECK_TOP + BOX_H = 0.80
BOX_FROM_HANDLE_END = 0.09                               # drawing "9": bare deck between the handle end and the box
HANDLE_DIA = 0.030                                       # drawing "handle 30mm width"
HANDLE_TOP = 0.77                                        # measured: the recorded closed hands sit 0.772 m up during the push
                                                         # (user said "level with the box top" = 0.80: within 3 cm)
HANDLE_OVERHANG = 0.0                                    # the grip bar stands this far BEHIND the deck, as a real push handle does,
                                                         # so the operator (robot) has leg clearance from the deck
CART_MASS = 15.0                                         # loaded service cart; pushable once the wheels get contact priority (see below)
WHEEL_FRICTION = 0.015                                   # rolling resistance of real casters (0.08 was sliding friction: far too high)

CUP_R_TOP, CUP_R_BOT, CUP_H, CUP_T = 0.045, 0.037, 0.14, 0.003   # drawing "9cm diameter, 14cm high"
CUP_MASS = 0.35                                          # a full coffee cup (bottle_bin: SIMPLE's 0.1 kg default tips under a finger touch)
CUP_CORNER_BOX = 0.10                                    # drawing "10cm x 10cm at corner": the cup is centred in that square
ROBOT_TO_CART = 0.20                                     # drawing "10cm Robot"; user 2026-09-23: moved 10 cm further out
SHEET_ROBOT_TO_CART = 0.10                               # what the sheet actually says, for the report
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
HEAD_TILT_TICKS = 214                                             # ID0 goal 2300 = home 2086 + 214 -> 18.8 deg down (was 2200 / 114 / 10 deg until 2026-09-24)

# teleop stack first_person_camera on d455_link (MuJoCo camera convention)
EGO_CAM_QUAT = "0.64367383 0.26523914 -0.27106013 -0.66472446"
EGO_HFOV = 90.0
EGO_W, EGO_H = 1280, 720                                          # 16:9 -> 58.7 deg VFOV (SIMPLE renders 640x360)
MOUNT_PITCH_AT_ZERO_DEG = 47.7                                    # optical axis below the torso x axis at d455_joint = 0

HAND_RGBA = "0.10 0.10 0.10 1"     # Dex3 hand links (the robot file has them light grey)
FLOOR_RGBA = "0.82 0.82 0.82 1"    # plain light grey floor (no reflectance: a mirrored wheel reads as a sunken wheel)
DIM = (183, 55, 31)          # dimension lines on the plan
EXT = (120, 120, 120)        # extension lines
ROUTE = (35, 90, 160)        # the drawn route on the plan


# ---- generated meshes -----------------------------------------------------------------
def _write_obj(path: Path, verts, tris, comment=""):
    lines = [f"# {comment}"] + [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["f %d %d %d" % (a + 1, b + 1, c + 1) for a, b, c in tris]
    path.write_text("\n".join(lines) + "\n")


def revolve(profile, n=48):
    """Closed surface of revolution about z from a (r, z) profile that starts and ends on the axis (r = 0)."""
    verts, tris, rings = [], [], []
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
            if len(lo) == 1:
                tris.append((lo[0], hi[k1], hi[k]))
            elif len(hi) == 1:
                tris.append((hi[0], lo[k], lo[k1]))
            else:
                tris += [(lo[k], lo[k1], hi[k1]), (lo[k], hi[k1], hi[k])]
    return verts, tris


def cup_mesh(adir: Path):
    """Hollow coffee cup, bottom at z = 0: outer wall up, rim, inner wall down, inner floor."""
    t = CUP_T
    outer = [(0, 0), (CUP_R_BOT, 0), (CUP_R_TOP, CUP_H)]
    inner = [(CUP_R_TOP - t, CUP_H), (CUP_R_BOT - t, t), (0, t)]
    p = adir / "coffee_cup_9cm.obj"
    _write_obj(p, *revolve(outer + inner), comment="coffee cup 9 cm dia x 14 cm, 3 mm wall, bottom at z=0")
    return p


def cup_collision_geoms(parent, material, nseg=12):
    """Bottom disc + tilted wall segments (group 5 = hidden) so the fingers can close on the wall and it stays hollow."""
    common = dict(type="box", group="5", material=material, friction="1 0.005 0.0001")
    ET.SubElement(parent, "geom", name="cup_bottom", type="cylinder", size=f"{CUP_R_BOT:.4f} {CUP_T/2:.4f}",
                  pos=f"0 0 {CUP_T/2:.4f}", group="5", material=material, friction="1 0.005 0.0001",
                  mass=f"{CUP_MASS*0.35:.4f}")
    z0 = 0.006
    r0 = CUP_R_BOT + (CUP_R_TOP - CUP_R_BOT) * z0 / CUP_H
    slant = np.arctan2(CUP_R_TOP - CUP_R_BOT, CUP_H)                 # wall tilt from vertical
    wall_len = np.hypot(CUP_R_TOP - r0, CUP_H - z0)
    r_mid = (CUP_R_TOP + r0) / 2 - CUP_T / 2
    half_w = r_mid * np.sin(np.pi / nseg) * 1.05
    for k in range(nseg):
        a = 2 * np.pi * (k + 0.5) / nseg
        q = (R.from_euler("z", a) * R.from_euler("y", slant)).as_quat()   # box z along the wall, tilted outward
        ET.SubElement(parent, "geom", name=f"cup_wall{k}", size=f"{CUP_T/2:.4f} {half_w:.4f} {wall_len/2:.4f}",
                      pos=f"{r_mid*np.cos(a):.4f} {r_mid*np.sin(a):.4f} {(z0 + CUP_H)/2:.4f}",
                      quat=f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}", mass=f"{CUP_MASS*0.65/nseg:.5f}", **common)


def box_geom(parent, name, size, pos, material, **kw):
    ET.SubElement(parent, "geom", name=name, type="box", size=" ".join(f"{v/2:.4f}" for v in size),
                  pos=" ".join(f"{v:.4f}" for v in pos), material=material, **kw)


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


def solve_legs_at(model, q_named, jid):
    """solve_legs with the pelvis free joint set to PELVIS_HEIGHT (the robot file's default may differ)."""
    orig = model.qpos0.copy()
    model.qpos0[2] = PELVIS_HEIGHT
    try:
        return solve_legs(model, q_named)
    finally:
        model.qpos0[:] = orig


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
    q = solve_legs_at(m, q, jid)
    for n, v in q.items():
        d.qpos[m.jnt_qposadr[jid[n]]] = v
    mujoco.mj_kinematics(m, d)
    return q, toe_front_x(m, d), pelvis_z


def cart_geoms(parent, L):
    """Cart in its own frame: origin on the floor under the deck centre, +x toward the table, handle at -x."""
    hl, hw = CART_L / 2, CART_W / 2
    # deck slab, top at DECK_TOP
    box_geom(parent, "cart_deck", (CART_L, CART_W, DECK_T), (0, 0, DECK_TOP - DECK_T / 2), "cart_mat",
             friction="1 0.005 0.0001", mass=f"{CART_MASS*0.24:.4f}")
    # box standing on the deck, BOX_FROM_HANDLE_END of bare deck at the handle end
    box_cx = -hl + BOX_FROM_HANDLE_END + BOX_L / 2
    box_geom(parent, "cart_box", (BOX_L, BOX_W, BOX_H), (box_cx, 0, DECK_TOP + BOX_H / 2), "box_mat",
             friction="1 0.005 0.0001", mass=f"{CART_MASS*0.46:.4f}")
    # push handle: two uprights at the deck's near end, two struts running BACK, and the grip bar across their ends
    r = HANDLE_DIA / 2
    post_h = HANDLE_TOP - DECK_TOP
    ov = HANDLE_OVERHANG
    for s, y in (("l", hw - r - 0.01), ("r", -(hw - r - 0.01))):
        ET.SubElement(parent, "geom", name=f"cart_post_{s}", type="cylinder", size=f"{r:.4f} {post_h/2:.4f}",
                      pos=f"{-hl + r:.4f} {y:.4f} {DECK_TOP + post_h/2:.4f}", material="handle_mat",
                      mass=f"{CART_MASS*0.03:.4f}")
        if ov > 1e-3:                       # struts only when the bar actually stands behind the deck
            ET.SubElement(parent, "geom", name=f"cart_strut_{s}", type="cylinder", size=f"{r:.4f} {ov/2:.4f}",
                          pos=f"{-hl + r - ov/2:.4f} {y:.4f} {HANDLE_TOP:.4f}", quat="0.7071068 0 0.7071068 0",
                          material="handle_mat", mass=f"{CART_MASS*0.02:.4f}")
    ET.SubElement(parent, "geom", name="cart_handle", type="cylinder", size=f"{r:.4f} {(hw - r - 0.01):.4f}",
                  pos=f"{-hl + r - ov:.4f} 0 {HANDLE_TOP:.4f}", quat="0.7071068 0.7071068 0 0", material="handle_mat",
                  friction="1 0.005 0.0001", mass=f"{CART_MASS*0.04:.4f}")
    # four casters (one rigid body: low friction stands in for swivelling wheels)
    wr = WHEEL_DIA / 2
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ET.SubElement(parent, "geom", name=f"cart_wheel{i}", type="cylinder", size=f"{wr:.4f} 0.018",
                      pos=f"{sx*(hl - 0.09):.4f} {sy*(hw - 0.05):.4f} {wr:.4f}", quat="0.7071068 0.7071068 0 0",
                      material="wheel_mat", friction=f"{WHEEL_FRICTION} 0.002 0.0001", priority="1",
                      mass=f"{CART_MASS*0.045:.4f}")     # priority 1: MuJoCo pairs friction by MAX unless one geom outranks; floor is 1.0


def build_xml(args, out: Path, layout: dict, cam_pos, fovy) -> Path:
    adir = HERE / "assets"; adir.mkdir(exist_ok=True)
    cup_obj = cup_mesh(adir)

    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    root.set("model", "g1comp_coffee_cart")
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
    ET.SubElement(wb, "light", pos="1.0 -0.6 2.8", dir="0 0 -1", directional="false", diffuse="0.75 0.75 0.75", specular="0.2 0.2 0.2")
    ET.SubElement(wb, "light", pos="-1.2 1.4 2.4", dir="0.55 -0.45 -0.7", directional="true", diffuse="0.4 0.4 0.4")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="6 6 0.05", material="floor_mat", friction="1 0.005 0.0001")

    L = layout
    # table: top slab on four legs
    table = ET.SubElement(wb, "body", name="table", pos=f"{L['table_cx']:.4f} {L['table_cy']:.4f} 0")
    box_geom(table, "table_top", (TABLE_D, TABLE_W, TABLE_TOP_T), (0, 0, TABLE_H - TABLE_TOP_T / 2), "top_mat",
             friction="1 0.005 0.0001")
    leg_h = TABLE_H - TABLE_TOP_T
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        box_geom(table, f"table_leg{i}", (TABLE_LEG, TABLE_LEG, leg_h),
                 (sx * (TABLE_D / 2 - TABLE_LEG / 2 - 0.02), sy * (TABLE_W / 2 - TABLE_LEG / 2 - 0.02), leg_h / 2), "cabinet_mat")
    # cart (one rigid free body)
    cart = ET.SubElement(wb, "body", name="cart", pos=f"{L['cart_cx']:.4f} {L['cart_cy']:.4f} 0")
    ET.SubElement(cart, "freejoint", name="cart")
    cart_geoms(cart, L)
    # coffee cup on the box top
    cup = ET.SubElement(wb, "body", name="cup", pos=f"{L['cup_x']:.4f} {L['cup_y']:.4f} {L['box_top']:.4f}")
    ET.SubElement(cup, "freejoint", name="cup")
    ET.SubElement(cup, "geom", name="cup_visual", type="mesh", mesh="cup", material="cup_mat", contype="0", conaffinity="0", group="1", mass="0")
    cup_collision_geoms(cup, "cup_mat")

    # third-person cameras (computed from a position and a look-at point)
    tcx, tcy = L["table_cx"], L["table_cy"]
    ccx = L["cart_cx"]
    cams = {
        "overview": ((-1.7, 2.3, 2.5), (1.0, -0.65, 0.6)),
        "front_left": ((3.1, 1.5, 2.0), (1.1, -0.6, 0.7)),
        "from_table": ((tcx + 0.9, tcy - 0.4, 1.8), (0.4, -0.1, 0.8)),
        "side": ((ccx, 2.9, 1.25), (ccx, -0.3, 0.7)),
        "cart_closeup": ((ccx - 0.75, 1.05, 1.45), (ccx, 0.0, 0.72)),
        "cup_closeup": ((L["cup_x"] - 0.34, L["cup_y"] + 0.40, 1.14), (L["cup_x"], L["cup_y"], 0.86)),
        "table_closeup": ((tcx - 1.0, tcy + 0.95, 1.55), (tcx, tcy, 0.78)),
    }
    for name, (pos, tgt) in cams.items():
        ET.SubElement(wb, "camera", name=name, pos=" ".join(f"{v:.3f}" for v in pos), xyaxes=cam_xyaxes(pos, tgt))
    ET.SubElement(wb, "camera", name="plan", pos=f"{L['plan_cx']:.3f} {L['plan_cy']:.3f} 5", xyaxes="0 -1 0 1 0 0",
                  orthographic="true", fovy=f"{L['plan_extent']:.3f}")

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", name="cup", file=mesh_ref(cup_obj))
    ET.SubElement(asset, "material", name="floor_mat", rgba=FLOOR_RGBA, reflectance="0", specular="0.1", shininess="0.1")
    ET.SubElement(asset, "material", name="cabinet_mat", rgba="0.80 0.72 0.60 1")
    ET.SubElement(asset, "material", name="top_mat", rgba="0.88 0.80 0.68 1", specular="0.3", shininess="0.3")
    ET.SubElement(asset, "material", name="cart_mat", rgba="0.62 0.64 0.67 1", specular="0.4", shininess="0.4")
    ET.SubElement(asset, "material", name="box_mat", rgba="0.76 0.60 0.40 1")
    ET.SubElement(asset, "material", name="handle_mat", rgba="0.25 0.26 0.28 1", specular="0.6", shininess="0.6")
    ET.SubElement(asset, "material", name="wheel_mat", rgba="0.12 0.12 0.13 1")
    ET.SubElement(asset, "material", name="cup_mat", rgba="0.95 0.95 0.94 1", specular="0.5", shininess="0.5")

    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode")
    return out


def compute_layout(args, toe_off):
    # route: the pelvis start is args.route_forward back from the table's FAR edge; the table's LEFT end is
    # args.table_from_line to the robot's right (-y)
    table_far_x = args.route_forward
    table_near_x = table_far_x - TABLE_D
    table_left_y = -args.table_from_line
    table_right_y = table_left_y - TABLE_W
    # cart: handle args.robot_to_cart ahead of the front of the feet (or the pelvis), on the robot's centre line
    ref_x = toe_off if args.robot_ref == "toes" else 0.0
    handle_x = ref_x + args.robot_to_cart                      # the grip bar (what "10cm Robot" measures to)
    deck_near_x = handle_x + args.handle_overhang              # the deck itself starts further forward
    deck_far_x = deck_near_x + CART_L
    cart_cx = deck_near_x + CART_L / 2
    cart_cy = args.cart_y
    box_near_x = deck_near_x + BOX_FROM_HANDLE_END
    box_far_x = box_near_x + BOX_L
    box_top = DECK_TOP + BOX_H
    box_right_y = cart_cy - BOX_W / 2
    # cup: centred in the 10 x 10 cm square at the box top's near-right corner
    cup_x = args.cup_xy[0] if args.cup_xy else box_near_x + CUP_CORNER_BOX / 2
    cup_y = args.cup_xy[1] if args.cup_xy else box_right_y + CUP_CORNER_BOX / 2
    return dict(route_forward=args.route_forward, handle_overhang=args.handle_overhang, table_from_line=args.table_from_line,
                robot_to_cart=args.robot_to_cart, robot_ref=args.robot_ref, toe_offset=toe_off,
                table_near_x=table_near_x, table_far_x=table_far_x, table_left_y=table_left_y, table_right_y=table_right_y,
                table_cx=(table_near_x + table_far_x) / 2, table_cy=(table_left_y + table_right_y) / 2, table_top=TABLE_H,
                handle_x=handle_x, deck_near_x=deck_near_x, deck_far_x=deck_far_x, cart_cx=cart_cx, cart_cy=cart_cy,
                deck_top=DECK_TOP, box_near_x=box_near_x, box_far_x=box_far_x, box_right_y=box_right_y,
                box_left_y=cart_cy + BOX_W / 2, box_top=box_top, handle_top=HANDLE_TOP,
                cup_x=cup_x, cup_y=cup_y, cup_z=box_top,
                pelvis_to_table_near=table_near_x,
                plan_cx=1.05, plan_cy=-0.55, plan_extent=3.00, plan_w=1300, plan_h=1100)


def draw_plan(png: Path, L: dict, out: Path):
    im = Image.open(png).convert("RGB"); d = ImageDraw.Draw(im)
    W, H = im.size; s = H / L["plan_extent"]; cx, cy = L["plan_cx"], L["plan_cy"]
    P = lambda x, y: (W / 2 - (y - cy) * s, H / 2 - (x - cx) * s)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
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
        for q in (a, b):
            d.line([tuple(q - n), tuple(q + n)], fill=color, width=3)
        label(tuple((a + b) / 2 + np.array(off)), text, color)

    tnx, tfx = L["table_near_x"], L["table_far_x"]
    tly, try_ = L["table_left_y"], L["table_right_y"]
    toe_x, hx = L["toe_offset"], L["handle_x"]
    dnx, dfx = L["deck_near_x"], L["deck_far_x"]
    bnx, bfx = L["box_near_x"], L["box_far_x"]
    bry, bly = L["box_right_y"], L["box_left_y"]
    cy0 = L["cart_cy"]
    OUT = (70, 70, 70)

    def rect(x0, x1, y0, y1, color=OUT, width=2):
        d.line([P(x0, y0), P(x0, y1), P(x1, y1), P(x1, y0), P(x0, y0)], fill=color, width=width, joint="curve")

    # footprints (the render itself washes out from above, so outline everything)
    rect(tnx, tfx, tly, try_)
    rect(dnx, dfx, cy0 - CART_W / 2, cy0 + CART_W / 2)
    rect(bnx, bfx, bry, bly)
    rect(bnx, bnx + CUP_CORNER_BOX, bry, bry + CUP_CORNER_BOX, (150, 150, 150), 1)   # the 10 x 10 cm corner square
    cpx, cpy = P(L["cup_x"], L["cup_y"]); rr = CUP_R_TOP * s
    d.ellipse([cpx - rr, cpy - rr, cpx + rr, cpy + rr], outline=(150, 40, 20), width=3)

    # the route drawn on the sheet: forward, then across to the table's left end
    d.line([P(0.0, 0.0), P(tfx, 0.0)], fill=ROUTE, width=3)
    d.line([P(tfx, 0.0), P(tfx, tly)], fill=ROUTE, width=3)

    # 220 forward and 67 across
    ext((0.0, 0.0), (0.0, 0.36)); ext((tfx, 0.0), (tfx, 0.36))
    dim((0.0, 0.32), (tfx, 0.32), f"{L['route_forward']*100:.0f} cm  (drawing 220)", off=(-108, 0))
    ext((tfx, tly), (tfx + 0.17, tly))
    dim((tfx + 0.13, 0.0), (tfx + 0.13, tly), f"{L['table_from_line']*100:.0f} cm  (67)", off=(0, -22))
    # table
    dim((tfx + 0.13, tly), (tfx + 0.13, try_), f"table {TABLE_W*100:.0f} x {TABLE_D*100:.0f} x {TABLE_H*100:.0f} cm", off=(0, -22))
    ext((tnx, try_), (tnx, try_ - 0.18)); ext((tfx, try_), (tfx, try_ - 0.18))
    dim((tnx, try_ - 0.14), (tfx, try_ - 0.14), f"{TABLE_D*100:.0f} deep", off=(52, 0))
    # 10 cm feet -> handle
    ext((toe_x, 0.0), (toe_x, -0.30)); ext((hx, 0.0), (hx, -0.30))
    dim((toe_x, -0.26), (hx, -0.26), f"{(hx-toe_x)*100:.0f} cm feet to handle", off=(112, 0))
    # cart deck and the 9 cm of bare deck at the handle end
    ext((dnx, cy0 + CART_W / 2), (dnx, cy0 + 0.34)); ext((dfx, cy0 + CART_W / 2), (dfx, cy0 + 0.34))
    dim((dnx, cy0 + 0.30), (dfx, cy0 + 0.30), f"cart deck {CART_L*100:.0f}", off=(-62, 0))
    dim((dfx + 0.13, cy0 - CART_W / 2), (dfx + 0.13, cy0 + CART_W / 2), f"{CART_W*100:.0f}", off=(0, -20))
    dim((dnx, bry - 0.05), (bnx, bry - 0.05), f"{BOX_FROM_HANDLE_END*100:.0f}", off=(26, 0))
    # text
    label(P((bnx + bfx) / 2 + 0.09, cy0), f"box {BOX_L*100:.0f} x {BOX_W*100:.0f} x {BOX_H*100:.0f} cm, top {L['box_top']*100:.0f} cm", (60, 60, 60), small)
    label(P(dnx - 0.16, cy0 + 0.02), f"handle {HANDLE_DIA*1000:.0f} mm dia, bar at {L['handle_top']*100:.0f} cm", (60, 60, 60), small)
    label(P(bnx + 0.16, bry - 0.30), f"cup {CUP_R_TOP*200:.0f} cm dia x {CUP_H*100:.0f} cm, centred in the 10 x 10 cm corner square",
          (60, 60, 60), small)
    label(P(tnx - 0.20, (tly + try_) / 2), f"table top {TABLE_H*100:.0f} cm  (cup starts {(L['box_top']-TABLE_H)*100:.0f} cm higher)", (60, 60, 60), small)
    label(P(-0.28, 0.46), "g1comp start: facing +x, hands on the handle", (60, 60, 60), small)
    label((W - 12, H - 12), "plan view from above, robot's right = image right", (60, 60, 60), small, anchor="rd")
    label((12, 12), f"{1/s*100:.3f} cm / px", (60, 60, 60), small, anchor="la")
    label((12, 40), "blue = the route drawn on the sheet", ROUTE, small, anchor="la")
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scene.xml"))
    ap.add_argument("--renders", default=str(HERE / "renders"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--pose", type=int, default=3, choices=(1, 2, 3), help="3 = elbows raised -0.66 (default); 2 = arms hanging; 1 = Psi0 client pose")
    ap.add_argument("--tilt-ticks", type=float, default=HEAD_TILT_TICKS, help="D455 tilt servo from home in ticks (0.088 deg/tick, + = down)")
    ap.add_argument("--robot-ref", default="toes", choices=("toes", "pelvis"), help="what --robot-to-cart measures from")
    ap.add_argument("--robot-to-cart", type=float, default=ROBOT_TO_CART, help="front of the feet (or pelvis) -> the cart handle (m)")
    ap.add_argument("--route-forward", type=float, default=ROUTE_FORWARD, help="pelvis start -> the table's far edge (m; drawing 2.20)")
    ap.add_argument("--table-from-line", type=float, default=TABLE_FROM_LINE, help="robot's path line -> the table's left end, to the right (m; drawing 0.67)")
    ap.add_argument("--cart-y", type=float, default=0.0, help="cart centre line offset from the robot's (m, + = robot's left)")
    ap.add_argument("--cup-xy", type=float, nargs=2, default=None, metavar=("X", "Y"), help="cup centre in the pelvis frame (default: the corner square)")
    ap.add_argument("--cup-mass", type=float, default=CUP_MASS)
    ap.add_argument("--cart-mass", type=float, default=CART_MASS)
    ap.add_argument("--handle-top", type=float, default=HANDLE_TOP, help="push-bar height above the floor (m; not on the sheet)")
    ap.add_argument("--handle-overhang", type=float, default=HANDLE_OVERHANG, help="grip bar this far behind the deck (m)")
    ap.add_argument("--ego-size", default=f"{EGO_W}x{EGO_H}")
    ap.add_argument("--settle", type=int, default=0, help="physics steps with the robot held (the cup settles on the box)")
    args = ap.parse_args()
    globals()["CUP_MASS"] = args.cup_mass
    globals()["CART_MASS"] = args.cart_mass
    globals()["HANDLE_TOP"] = args.handle_top
    globals()["HANDLE_OVERHANG"] = args.handle_overhang

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
        hold = data.qpos.copy(); nq = model.jnt_qposadr[jid["cup"]]
        for _ in range(args.settle):
            mujoco.mj_step(model, data)
            b = data.qpos[nq:nq + 7].copy(); data.qpos[:] = hold; data.qpos[nq:nq + 7] = b; data.qvel[:] = 0
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
                  bodies={n: np.round(data.xpos[bid(n)], 4).tolist() for n in ("pelvis", "torso_link", "cart", "cup", "table",
                                                                              "left_wrist_yaw_link", "right_wrist_yaw_link",
                                                                              "left_hand_index_1_link", "right_hand_index_1_link")},
                  cup=dict(diameter=2 * CUP_R_TOP, height=CUP_H, mass=CUP_MASS),
                  cart=dict(deck=[CART_L, CART_W, DECK_T], deck_top=DECK_TOP, caster_h=CASTER_H, box=[BOX_L, BOX_W, BOX_H], box_top=L["box_top"],
                            box_from_handle_end=BOX_FROM_HANDLE_END, handle_dia=HANDLE_DIA, handle_top=HANDLE_TOP, handle_overhang=HANDLE_OVERHANG,
                            wheel_dia=WHEEL_DIA, mass=CART_MASS, wheel_friction=WHEEL_FRICTION),
                  table=dict(size=[TABLE_D, TABLE_W, TABLE_H], top_t=TABLE_TOP_T))
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])}:geom{g}"
    contacts = sorted({(gname(c.geom1), gname(c.geom2)) for c in data.contact[:data.ncon]})
    report["contacts"] = [f"{a} <-> {b}" for a, b in contacts if not ({a, b} & {"floor"} and ("ankle_roll" in a or "ankle_roll" in b or "cart_wheel" in a or "cart_wheel" in b))]
    (HERE / f"layout{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("layout", "ego_camera", "contacts")}, indent=1))

    rdir = Path(args.renders); rdir.mkdir(parents=True, exist_ok=True)
    opt = mujoco.MjvOption(); opt.geomgroup[3] = 1
    opt_axis = mujoco.MjvOption(); opt_axis.geomgroup[3] = 1; opt_axis.geomgroup[4] = 1
    wide = ("overview", "front_left", "from_table", "side")
    ren = mujoco.Renderer(model, 900, 1600)
    for cam in ("overview", "front_left", "from_table", "side", "cart_closeup", "cup_closeup", "table_closeup"):
        ren.update_scene(data, camera=cam, scene_option=opt_axis if cam in wide else opt)
        Image.fromarray(ren.render()).save(rdir / f"{cam}{args.tag}.png")
    ren.close()
    ren = mujoco.Renderer(model, L["plan_h"], L["plan_w"])
    ren.update_scene(data, camera="plan", scene_option=opt)
    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    raw = rdir / f"plan_raw{args.tag}.png"
    Image.fromarray(ren.render()).save(raw); draw_plan(raw, L, rdir / f"plan{args.tag}.png"); raw.unlink()
    ren.close()
    ren = mujoco.Renderer(model, ego_h, ego_w)
    opt_ego = mujoco.MjvOption(); opt_ego.geomgroup[3] = 0
    ren.update_scene(data, camera="ego", scene_option=opt_ego); Image.fromarray(ren.render()).save(rdir / f"ego_d455{args.tag}.png")
    ren.close()
    print("wrote", xml, "and renders to", rdir)


if __name__ == "__main__":
    main()
