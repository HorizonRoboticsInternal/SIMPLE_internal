#!/usr/bin/env python3
"""Bowl-to-sink kitchen scene for the g1comp (MuJoCo), from the hand drawing of 2026-09-22 (numbers in cm; the "mm" on the
sheet are cm) and the recordings of 2026-09-21 (holobrain run 20260921_172723_bowltosink_c96, DJI clips bowltosink_v2).

World frame: pelvis at the origin at the start pose, x toward the bowl counter, z up, the robot's right = -y.
  * bowl counter 2.60 m long (across, facing the robot) x 0.63 m deep, top 0.86 m; the robot stands 1.39 m from its LEFT end
    (= at the bowl's position along the counter), --robot-to-edge from the front of the feet to the counter's front edge
  * back unit 2.60 x 0.41 x 1.07 m behind the bowl counter (against the wall)
  * bowl: 15 cm diameter, 8 cm deep, hollow, green; --bowl-xy = its centre relative to the pelvis start (default: 0.20 m
    beyond the counter edge, 0.10 m to the robot's left; tuned against the head-camera frames)
  * sink counter 2.66 x 0.41 x 1.07 m perpendicular, along the wall on the robot's RIGHT: its front face at the bowl
    counter's right end, running from the wall toward the robot's rear; sink basin 0.54 (along) x 0.33 (across) x 0.10 m deep
    with a 4 cm rim on each side, its centre --sink-along m from the bowl counter's front edge (drawing: 1.27) toward the rear
  * pose 3 (elbows -0.66, everything else 0), head servo tilt 10 deg, D455 ego camera 90 deg HFOV, 16:9 -- as bottle_bin

    MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py [--tag _x]
Writes scene.xml (keyframe default_pose, paths relative to this folder), assets/*.obj, layout.json and renders/*.png
(third-person views, the robot's D455 view, plan.png = orthographic top view with the dimensions drawn).
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
COUNTER_W, COUNTER_D, COUNTER_H = 2.60, 0.63, 0.86      # bowl counter: along y (faces the robot), depth (x), top height
COUNTER_TOP_T = 0.03
BACK_D, BACK_H = 0.41, 1.07                             # unit behind the bowl counter (same length)
ROBOT_FROM_LEFT = 1.31                                  # pelvis -> the counter's left end. The drawing's 139 is to the BOWL; the robot
                                                        # stands 8 cm to its left of that (user 2026-09-23: centre the bowl in the view),
                                                        # so pelvis -> left end = 1.39 - 0.08
ROBOT_TO_EDGE = 0.30                                    # front of the feet -> counter front edge (head-camera frames; drawing has no value)
BOWL_R_TOP, BOWL_R_BOT, BOWL_H, BOWL_T = 0.075, 0.045, 0.08, 0.004   # 15 cm dia, 8 cm deep, 4 mm wall
BOWL_MASS = 0.12                                        # green plastic bowl
BOWL_XY_DEFAULT = (None, 0.00)                          # x = counter edge + BOWL_FROM_EDGE unless given; y = 0 = straight ahead
                                                        # (user 2026-09-23: the robot moved left so the bowl is centred in its view)
BOWL_FROM_EDGE = 0.15                                   # bowl centre beyond the counter's front edge (head-camera frames)
SINK_W, SINK_D, SINK_H = 2.08, 0.41, 0.86               # sink counter: along x (user 2026-09-23: 208 cm, not the sheet's 266), depth (y), top height
                                                        # (user 2026-09-23: the same 86 cm worktop as the bowl counter; the 107 on the
                                                        #  drawing is the height of the tall back unit only)
SINK_TOP_T = 0.10                                       # the basin depth = the slab it is cut into
BASIN_L, BASIN_ACROSS, BASIN_DEPTH, BASIN_RIM = 0.54, 0.33, 0.10, 0.04
BASIN_R = 0.03                                          # rounded corners of the basin (user 2026-09-23); --basin-radius
SINK_ALONG = 1.27                                       # drawing "127": bowl counter front edge -> the basin, along the sink counter
SINK_FROM_END = 0.63                                    # user 2026-09-23: the basin's far edge is 63 cm from the sink counter's FAR end
                                                        # (the end away from the wall, toward the robot's rear); --sink-from-end
SINK_START = "front-edge"                               # user 2026-09-23: the sink counter STARTS at the bowl counter's front-edge line
                                                        # and runs 2.66 m toward the robot's rear (it does not reach back to the wall); --sink-start wall = the old layout
SINK_REF = "far-edge"                                   # default anchor for the basin: the drawing's 127 from the sink counter's start (= the
                                                        # bowl counter's front edge) to the basin's far edge. "end" = SINK_FROM_END from the counter's far end.
                                                        # The older readings of the drawing's 127 remain available: "far-edge" / "near-edge" /
                                                        # "centre" = --sink-along from the bowl counter's front edge to that part of the basin
                                                        # edge FURTHER from the bowl counter; --sink-ref near-edge | centre
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


def bowl_mesh(adir: Path):
    """Hollow bowl, bottom at z = 0: outer wall up, rim, inner wall down, inner floor (a closed surface of revolution)."""
    t = BOWL_T
    outer = [(0, 0), (BOWL_R_BOT, 0), (BOWL_R_TOP, BOWL_H)]
    inner = [(BOWL_R_TOP - t, BOWL_H), (BOWL_R_BOT - t, t), (0, t)]
    p = adir / "bowl_15cm.obj"; _write_obj(p, *revolve(outer + inner), comment="bowl 15 cm dia x 8 cm, 4 mm wall, bottom at z=0")
    return p


def bowl_collision_geoms(parent, material, nseg=16):
    """Bottom disc + tilted wall segments (group 5 = hidden) so the hand can grasp the rim and objects stay inside."""
    common = dict(type="box", group="5", material=material, friction="1 0.005 0.0001")
    ET.SubElement(parent, "geom", name="bowl_bottom", type="cylinder", size=f"{BOWL_R_BOT:.4f} {BOWL_T/2:.4f}", pos=f"0 0 {BOWL_T/2:.4f}",
                  group="5", material=material, friction="1 0.005 0.0001", mass=f"{BOWL_MASS*0.4:.4f}")
    z0 = 0.008                                                          # wall pieces start above the floor disc so their tilted corners clear the counter
    r0 = BOWL_R_BOT + (BOWL_R_TOP - BOWL_R_BOT) * z0 / BOWL_H
    slant = np.arctan2(BOWL_R_TOP - BOWL_R_BOT, BOWL_H)                 # wall tilt from vertical
    wall_len = np.hypot(BOWL_R_TOP - r0, BOWL_H - z0)
    r_mid = (BOWL_R_TOP + r0) / 2 - BOWL_T / 2
    half_w = r_mid * np.sin(np.pi / nseg) * 1.05
    for k in range(nseg):
        a = 2 * np.pi * (k + 0.5) / nseg
        q = (R.from_euler("z", a) * R.from_euler("y", slant)).as_quat()   # box z along the wall, tilted outward
        ET.SubElement(parent, "geom", name=f"bowl_wall{k}", size=f"{BOWL_T/2:.4f} {half_w:.4f} {wall_len/2:.4f}",
                      pos=f"{r_mid*np.cos(a):.4f} {r_mid*np.sin(a):.4f} {(z0 + BOWL_H)/2:.4f}",
                      quat=f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}", mass=f"{BOWL_MASS*0.6/nseg:.5f}", **common)


def rounded_rect(w, d, r, n=8):
    """CCW loop (x, y): w along x, d along y, corner radius r, n segments per corner."""
    hw, hd = w / 2, d / 2
    pts = []
    for cx, cy, a0 in ((hw - r, hd - r, 0), (-hw + r, hd - r, 90), (-hw + r, -hd + r, 180), (hw - r, -hd + r, 270)):
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / n)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def basin_rim_mesh(adir: Path, w, d, r, t, n=8):
    """The four corner patches of the worktop that turn the square basin cut-out into a rounded one:
    each is the square corner minus the quarter disc of radius r, extruded down over the slab thickness t.
    Visual only (the walls below carry the collision)."""
    hw, hd = w / 2, d / 2
    verts, tris = [], []
    for cx, cy, a0, corner in ((hw - r, hd - r, 0, (hw, hd)), (-hw + r, hd - r, 90, (-hw, hd)),
                               (-hw + r, -hd + r, 180, (-hw, -hd)), (hw - r, -hd + r, 270, (hw, -hd))):
        arc = [(cx + r * np.cos(np.radians(a0 + 90 * k / n)), cy + r * np.sin(np.radians(a0 + 90 * k / n))) for k in range(n + 1)]
        poly = arc + [corner]                                     # simple polygon, fan from the square corner
        base = len(verts)
        verts += [(x, y, 0.0) for x, y in poly] + [(x, y, -t) for x, y in poly]
        m = len(poly); top = lambda i: base + i; bot = lambda i: base + m + i
        for k in range(len(arc) - 1):                             # top and bottom faces (fan from the corner vertex)
            tris.append((top(m - 1), top(k), top(k + 1)))
            tris.append((bot(m - 1), bot(k + 1), bot(k)))
        for k in range(m):                                        # side wall
            j = (k + 1) % m
            tris += [(top(k), bot(k), bot(j)), (top(k), bot(j), top(j))]
    path = adir / f"basin_rim_{int(w*100)}x{int(d*100)}_r{int(r*1000)}.obj"
    _write_obj(path, verts, tris, comment="worktop corner patches: square cut-out minus quarter discs")
    return path


def basin_wall_geoms(parent, bx, by, zc, w, d, depth, r, material, nseg=4, t=0.01):
    """Straight basin walls shortened by the corner radius + tangent box segments around each rounded corner."""
    hw, hd = w / 2, d / 2
    for name, size, pos in ((f"basin_wall_l", (t, d - 2 * r, depth), (bx - hw + t / 2, by, zc)),
                            (f"basin_wall_r", (t, d - 2 * r, depth), (bx + hw - t / 2, by, zc)),
                            (f"basin_wall_f", (w - 2 * r, t, depth), (bx, by - hd + t / 2, zc)),
                            (f"basin_wall_b", (w - 2 * r, t, depth), (bx, by + hd - t / 2, zc))):
        box_geom(parent, name, size, pos, material)
    for ci, (ccx, ccy, a0) in enumerate(((hw - r, hd - r, 0), (-hw + r, hd - r, 90), (-hw + r, -hd + r, 180), (hw - r, -hd + r, 270))):
        for k in range(nseg):
            a = np.radians(a0 + 90 * (k + 0.5) / nseg)
            half = np.radians(90 / nseg / 2)
            rm = r - t / 2
            px, py = bx + ccx + rm * np.cos(a), by + ccy + rm * np.sin(a)
            q = R.from_euler("z", a + np.pi / 2).as_quat()          # box long axis tangent to the arc
            ET.SubElement(parent, "geom", name=f"basin_corner{ci}_{k}", type="box", material=material,
                          size=f"{rm*np.sin(half)*1.15:.4f} {t/2:.4f} {depth/2:.4f}",
                          pos=f"{px:.4f} {py:.4f} {zc:.4f}",
                          quat=f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}")


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
    bowl_obj = bowl_mesh(adir)
    rim_obj = basin_rim_mesh(adir, BASIN_L, BASIN_ACROSS, args.basin_radius, SINK_TOP_T)

    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    root.set("model", "g1comp_bowl_to_sink")
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
    ET.SubElement(wb, "light", pos="0.2 -0.4 2.8", dir="0 0 -1", directional="false", diffuse="0.75 0.75 0.75", specular="0.2 0.2 0.2")
    ET.SubElement(wb, "light", pos="-1.5 -1.0 2.2", dir="0.5 0.4 -0.75", directional="true", diffuse="0.4 0.4 0.4")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="5 5 0.05", material="floor_mat", friction="1 0.005 0.0001")

    L = layout
    # bowl counter (solid cabinet + top slab) and the unit behind it
    cx, cy = L["counter_cx"], L["counter_cy"]
    counter = ET.SubElement(wb, "body", name="counter", pos=f"{cx:.4f} {cy:.4f} 0")
    box_geom(counter, "counter_body", (COUNTER_D, COUNTER_W, COUNTER_H - COUNTER_TOP_T), (0, 0, (COUNTER_H - COUNTER_TOP_T) / 2), "cabinet_mat")
    box_geom(counter, "counter_top", (COUNTER_D, COUNTER_W, COUNTER_TOP_T), (0, 0, COUNTER_H - COUNTER_TOP_T / 2), "top_mat", friction="1 0.005 0.0001")
    box_geom(counter, "back_unit", (BACK_D, COUNTER_W, BACK_H), (COUNTER_D / 2 + BACK_D / 2, 0, BACK_H / 2), "cabinet_mat")
    box_geom(counter, "back_unit_top", (BACK_D, COUNTER_W, 0.01), (COUNTER_D / 2 + BACK_D / 2, 0, BACK_H + 0.005), "top_mat")
    # sink counter along the right wall: cabinet, top slab in four pieces around the basin, basin floor + walls
    sx, sy = L["sink_cx"], L["sink_cy"]
    sink = ET.SubElement(wb, "body", name="sink_counter", pos=f"{sx:.4f} {sy:.4f} 0")
    body_h = SINK_H - SINK_TOP_T
    box_geom(sink, "sink_body", (SINK_W, SINK_D, body_h), (0, 0, body_h / 2), "cabinet_mat")
    bx, by = L["basin_cx"] - sx, L["basin_cy"] - sy                 # basin centre in the sink-counter frame
    zt = SINK_H - SINK_TOP_T / 2
    # slab pieces: left/right of the basin (along x), front/back strips (along y)
    xl0, xl1 = -SINK_W / 2, bx - BASIN_L / 2; xr0, xr1 = bx + BASIN_L / 2, SINK_W / 2
    box_geom(sink, "sink_top_a", (xl1 - xl0, SINK_D, SINK_TOP_T), ((xl0 + xl1) / 2, 0, zt), "top_mat")
    box_geom(sink, "sink_top_b", (xr1 - xr0, SINK_D, SINK_TOP_T), ((xr0 + xr1) / 2, 0, zt), "top_mat")
    yb0, yb1 = by - BASIN_ACROSS / 2, by + BASIN_ACROSS / 2
    box_geom(sink, "sink_top_c", (BASIN_L, yb0 + SINK_D / 2, SINK_TOP_T), (bx, (yb0 - SINK_D / 2) / 2, zt), "top_mat")
    box_geom(sink, "sink_top_d", (BASIN_L, SINK_D / 2 - yb1, SINK_TOP_T), (bx, (yb1 + SINK_D / 2) / 2, zt), "top_mat")
    box_geom(sink, "basin_floor", (BASIN_L, BASIN_ACROSS, 0.01), (bx, by, body_h + 0.005), "steel_mat", friction="1 0.005 0.0001")
    # basin walls with rounded corners (user 2026-09-23: the four corners of the sink are rounded)
    basin_wall_geoms(sink, bx, by, body_h + BASIN_DEPTH / 2, BASIN_L, BASIN_ACROSS, BASIN_DEPTH, args.basin_radius, "steel_mat")
    ET.SubElement(sink, "geom", name="basin_rim", type="mesh", mesh="basin_rim", material="top_mat",
                  pos=f"{bx:.4f} {by:.4f} {SINK_H:.4f}", contype="0", conaffinity="0", group="1")
    # bowl
    bowl = ET.SubElement(wb, "body", name="bowl", pos=f"{L['bowl_x']:.4f} {L['bowl_y']:.4f} {COUNTER_H:.4f}")
    ET.SubElement(bowl, "freejoint", name="bowl")
    ET.SubElement(bowl, "geom", name="bowl_visual", type="mesh", mesh="bowl", material="bowl_mat", contype="0", conaffinity="0", group="1")
    bowl_collision_geoms(bowl, "bowl_mat")

    # third-person cameras (computed from a position and a look-at point)
    cams = {
        "overview": ((-2.9, 2.2, 2.5), (0.2, -0.3, 0.6)),
        "front_left": ((2.2, 2.6, 1.9), (0.0, -0.4, 0.6)),
        "from_sink_side": ((-0.6, -3.4, 1.9), (0.1, 0.0, 0.7)),
        "side": ((0.35, 3.4, 1.15), (0.35, -0.3, 0.7)),
        "bowl_closeup": ((L['bowl_x'] - 0.35, L['bowl_y'] + 0.55, 1.25), (L['bowl_x'], L['bowl_y'], 0.9)),
        "sink_closeup": ((L['basin_cx'] + 0.2, L['basin_cy'] + 1.0, 1.7), (L['basin_cx'], L['basin_cy'], 1.0)),
    }
    for name, (pos, tgt) in cams.items():
        ET.SubElement(wb, "camera", name=name, pos=" ".join(f"{v:.3f}" for v in pos), xyaxes=cam_xyaxes(pos, tgt))
    ET.SubElement(wb, "camera", name="plan", pos=f"{L['plan_cx']:.3f} {L['plan_cy']:.3f} 4", xyaxes="0 -1 0 1 0 0",
                  orthographic="true", fovy=f"{L['plan_extent']:.3f}")

    asset = root.find("asset")
    ET.SubElement(asset, "mesh", name="bowl", file=mesh_ref(bowl_obj))
    ET.SubElement(asset, "mesh", name="basin_rim", file=mesh_ref(rim_obj))
    ET.SubElement(asset, "material", name="floor_mat", rgba=FLOOR_RGBA, reflectance="0.08", specular="0.1", shininess="0.1")
    ET.SubElement(asset, "material", name="cabinet_mat", rgba="0.93 0.93 0.91 1")
    ET.SubElement(asset, "material", name="top_mat", rgba="0.97 0.97 0.95 1", specular="0.3", shininess="0.3")
    ET.SubElement(asset, "material", name="steel_mat", rgba="0.72 0.74 0.76 1", specular="0.9", shininess="0.8", reflectance="0.2")
    ET.SubElement(asset, "material", name="bowl_mat", rgba="0.45 0.85 0.12 1", specular="0.5", shininess="0.5")

    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode")
    return out


def compute_layout(args, toe_off):
    pelvis_to_edge = args.robot_to_edge + (toe_off if args.robot_ref == "toes" else 0.0)
    near_x = pelvis_to_edge; far_x = near_x + COUNTER_D
    left_y = args.robot_from_left; right_y = left_y - COUNTER_W             # the counter's left end is on the robot's left (+y);
                                                                            # this also sets how far the sink counter's face is from the robot's line
    x_wall = far_x + BACK_D                                                 # the wall behind the bowl counter
    sink_cy = right_y - SINK_D / 2                                          # sink counter: front face at the bowl counter's right end
    sink_start_x = near_x if args.sink_start == "front-edge" else x_wall    # where the sink counter begins
    sink_cx = sink_start_x - SINK_W / 2                                     # ... running 2.66 m toward the robot's rear
    # the drawing's 127 runs from the bowl counter's front edge along the sink counter to the basin. --sink-ref says
    # which part of the basin it reaches: "far-edge" (default, the user's "sink right edge" = the edge away from the
    # bowl counter), "near-edge" (the edge toward it) or "centre".
    basin_ref_x = near_x - args.sink_along
    sink_x0 = sink_start_x - SINK_W                                         # the sink counter's far end
    basin_cx = {"end": sink_x0 + args.sink_from_end + BASIN_L / 2, "far-edge": basin_ref_x + BASIN_L / 2,
                "near-edge": basin_ref_x - BASIN_L / 2, "centre": basin_ref_x}[args.sink_ref]
    sink_along = near_x - (basin_cx - BASIN_L / 2)                            # always reported as front edge -> basin far edge
    bowl_x = args.bowl_xy[0] if args.bowl_xy else near_x + args.bowl_from_edge
    bowl_y = args.bowl_xy[1] if args.bowl_xy else BOWL_XY_DEFAULT[1]
    return dict(pelvis_to_edge=pelvis_to_edge, toe_offset=toe_off, near_x=near_x, far_x=far_x, x_wall=x_wall,
                left_y=left_y, right_y=right_y, counter_cx=near_x + COUNTER_D / 2, counter_cy=(left_y + right_y) / 2,
                bowl_x=bowl_x, bowl_y=bowl_y, sink_cx=sink_cx, sink_cy=sink_cy, sink_x0=sink_x0, sink_start_x=sink_start_x, sink_start=args.sink_start, sink_front_y=right_y,
                basin_cx=basin_cx, basin_cy=sink_cy, sink_along=sink_along, sink_ref=args.sink_ref, sink_from_end=basin_cx - BASIN_L / 2 - sink_x0,
                basin_far_x=basin_cx - BASIN_L / 2, basin_near_x=basin_cx + BASIN_L / 2, basin_radius=args.basin_radius,
                **_plan_view(near_x, far_x, x_wall, left_y, right_y, bowl_x, bowl_y, basin_cx, sink_cy, sink_x0))


def _plan_view(near_x, far_x, x_wall, left_y, right_y, bowl_x, bowl_y, basin_cx, sink_cy, sink_x0):
    """Frame the plan on what matters: the robot at the origin, the bowl, the basin and both counter faces."""
    xs = [0.0, near_x, far_x, bowl_x, basin_cx - 0.27, basin_cx + 0.27, x_wall, sink_x0]
    ys = [0.0, bowl_y, sink_cy - 0.205, left_y, right_y]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    extent = max(max(xs) - min(xs), (max(ys) - min(ys)) * 1000 / 1400) + 0.9
    return dict(plan_cx=round(cx, 3), plan_cy=round(cy, 3), plan_extent=round(extent, 2), plan_w=1400, plan_h=1000)


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

    nx, fx, ly, ry = L["near_x"], L["far_x"], L["left_y"], L["right_y"]
    OUT = (150, 150, 150)
    def rect(x0, y0, x1, y1, fill=None, outline=OUT, width=2):
        a, b = P(x0, y0), P(x1, y1)
        d.rectangle([min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], fill=fill, outline=outline, width=width)
    xw = L["x_wall"]; sy0_, sy1_ = L["sink_cy"] - SINK_D / 2, L["sink_cy"] + SINK_D / 2
    rect(nx, ry, fx, ly, fill=(246, 246, 243))                          # bowl counter
    rect(fx, ry, xw, ly, fill=(236, 236, 232))                          # back unit
    rect(L["sink_x0"], sy0_, L["sink_start_x"], sy1_, fill=(246, 246, 243))   # sink counter, from its start to its far end
    rect(L["basin_far_x"], L["basin_cy"] - BASIN_ACROSS / 2, L["basin_near_x"], L["basin_cy"] + BASIN_ACROSS / 2,
         fill=(205, 212, 218), outline=(110, 118, 126))                 # basin
    bx, by, toe_x = L["bowl_x"], L["bowl_y"], L["toe_offset"]
    # robot along the counter and to the edge
    ext((nx, ly), (-0.55, ly)); ext((toe_x, 0), (-0.55, 0))
    dim((-0.48, ly), (-0.48, 0.0), f"{ly*100:.0f} cm from the counter's left end", off=(0, 24))
    ext((toe_x, -0.14), (toe_x, -0.34)); ext((nx, -0.14), (nx, -0.34))
    dim((toe_x, -0.30), (nx, -0.30), f"{(nx-toe_x)*100:.0f} cm feet to edge", off=(120, -46))
    # bowl
    ext((nx - 0.02, by), (bx + 0.02, by)); dim((nx, by), (bx, by), f"{(bx-nx)*100:.0f} cm", off=(-58, 0))
    ext((bx, 0.0), (bx, by + 0.02)); dim((bx, 0.0), (bx, by), f"{abs(by)*100:.0f} cm {'left' if by > 0 else 'right'}", off=(0, -22))
    label(P(bx + 0.14, by), "bowl 15 cm dia, 8 cm deep", (60, 60, 60), small)
    # counters
    dim((fx + BACK_D + 0.08, ry), (fx + BACK_D + 0.08, ly), "bowl counter 2.60 x 0.63 m, top 0.86 m; back unit 0.41 m, top 1.07 m", off=(0, -22))
    label(P((nx + fx) / 2, (ly + ry) / 2 + 0.55), "counter top 0.86 m", (60, 60, 60), small)
    label(P(fx + BACK_D / 2, (ly + ry) / 2 + 0.55), "back unit 1.07 m", (60, 60, 60), small)
    sy0, sy1 = L["sink_cy"] - SINK_D / 2, L["sink_cy"] + SINK_D / 2
    dim((L["sink_x0"], sy0 - 0.10), (L["sink_start_x"], sy0 - 0.10), f"sink counter {SINK_W:.2f} x {SINK_D:.2f} m, top {SINK_H:.2f} m", off=(0, -40))
    label(P(L["sink_start_x"] - 0.30, L["sink_cy"]), "starts at the bowl counter's front-edge line", (60, 60, 60), small)
    # basin
    bcx, bcy = L["basin_cx"], L["basin_cy"]
    if L["sink_ref"] == "end":
        ext((L["sink_x0"], sy0), (L["sink_x0"], sy1 + 0.30)); ext((L["basin_far_x"], bcy), (L["basin_far_x"], sy1 + 0.30))
        dim((L["sink_x0"], sy1 + 0.24), (L["basin_far_x"], sy1 + 0.24), f"{L['sink_from_end']*100:.0f} cm from the sink counter's end to the basin", off=(0, -22))
        ext((nx, ry), (nx, sy1 + 0.20)); ext((L["basin_far_x"], bcy), (L["basin_far_x"], sy1 + 0.20))
        dim((nx, sy1 + 0.14), (L["basin_far_x"], sy1 + 0.14), f"= {L['sink_along']*100:.0f} cm behind the bowl counter's front edge", off=(140, 80))
    else:
        ref_x = {"far-edge": L["basin_far_x"], "near-edge": L["basin_near_x"], "centre": bcx}[L["sink_ref"]]
        ref_name = {"far-edge": "far edge", "near-edge": "near edge", "centre": "centre"}[L["sink_ref"]]
        sx_ = L["sink_start_x"]
        ext((sx_, ry), (sx_, sy1 + 0.22)); ext((ref_x, bcy), (ref_x, sy1 + 0.22))
        dim((sx_, sy1 + 0.16), (ref_x, sy1 + 0.16), f"{(sx_ - ref_x)*100:.0f} cm from the sink counter's start to the basin's {ref_name}", off=(-330, 0))
    label(P(bcx - 0.30, bcy), f"basin {BASIN_L*100:.0f} x {BASIN_ACROSS*100:.0f} x {BASIN_DEPTH*100:.0f} cm, {L['basin_radius']*100:.0f} cm rounded corners", (60, 60, 60), small)
    label(P(-0.75, 0.0), "g1comp start: facing the bowl counter (+x)", (60, 60, 60), small)
    label((W - 12, H - 12), "plan view from above, robot's right = image right", (60, 60, 60), small, anchor="rd")
    label((12, 12), f"{1/s*100:.3f} cm / px", (60, 60, 60), small, anchor="la")
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scene.xml"))
    ap.add_argument("--renders", default=str(HERE / "renders"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--pose", type=int, default=3, choices=(1, 2, 3), help="3 = elbows raised -0.66 (default); 2 = arms hanging; 1 = Psi0 client pose")
    ap.add_argument("--tilt-ticks", type=float, default=HEAD_TILT_TICKS, help="D455 tilt servo from home in ticks (0.088 deg/tick, + = down)")
    ap.add_argument("--robot-ref", default="toes", choices=("toes", "pelvis"), help="what --robot-to-edge measures from")
    ap.add_argument("--robot-to-edge", type=float, default=ROBOT_TO_EDGE, help="front of the feet (or pelvis) -> counter front edge (m)")
    ap.add_argument("--robot-from-left", type=float, default=ROBOT_FROM_LEFT, help="pelvis -> the bowl counter's LEFT end (m); also moves the sink counter, whose face sits at the counter's right end")
    ap.add_argument("--bowl-xy", type=float, nargs=2, default=None, metavar=("X", "Y"), help="bowl centre in the pelvis frame (x toward the counter, left = +y)")
    ap.add_argument("--bowl-from-edge", type=float, default=BOWL_FROM_EDGE, help="bowl centre beyond the counter's front edge (m); the base is 4.5 cm in radius, so 0.06 leaves it 1.5 cm inside the edge")
    ap.add_argument("--bowl-mass", type=float, default=0.12)
    ap.add_argument("--sink-along", type=float, default=SINK_ALONG, help="the drawing's 127: bowl counter front edge -> the basin, along the sink counter (m)")
    ap.add_argument("--sink-ref", default=SINK_REF, choices=("end", "far-edge", "near-edge", "centre"), help="how the basin is placed: end = --sink-from-end from the sink counter's far end (default); far-edge/near-edge/centre = --sink-along from the bowl counter's front edge to that part of the basin")
    ap.add_argument("--sink-from-end", type=float, default=SINK_FROM_END, help="with --sink-ref end: sink counter's far end -> the basin's far edge (m)")
    ap.add_argument("--sink-start", default=SINK_START, choices=("front-edge", "wall"), help="where the sink counter begins: front-edge = at the bowl counter's front-edge line (default), wall = flush with the back unit")
    ap.add_argument("--basin-radius", type=float, default=BASIN_R, help="corner radius of the basin (m)")
    ap.add_argument("--ego-size", default=f"{EGO_W}x{EGO_H}")
    ap.add_argument("--settle", type=int, default=0, help="physics steps with the robot held (the bowl settles on the counter)")
    args = ap.parse_args()
    globals()["BOWL_MASS"] = args.bowl_mass

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
        hold = data.qpos.copy(); nq = model.jnt_qposadr[jid["bowl"]]
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
                  bodies={n: np.round(data.xpos[bid(n)], 4).tolist() for n in ("pelvis", "torso_link", "bowl", "counter", "sink_counter",
                                                                              "left_wrist_yaw_link", "right_wrist_yaw_link",
                                                                              "left_hand_index_1_link", "right_hand_index_1_link")},
                  bowl=dict(diameter=2 * BOWL_R_TOP, depth=BOWL_H, mass=BOWL_MASS),
                  sink=dict(counter=[SINK_W, SINK_D, SINK_H], basin=[BASIN_L, BASIN_ACROSS, BASIN_DEPTH], basin_centre=[L["basin_cx"], L["basin_cy"], SINK_H - BASIN_DEPTH],
                            top=SINK_H, along=L["sink_along"], ref=args.sink_ref, from_end=L["sink_from_end"], corner_radius=args.basin_radius,
                            basin_far_x=L["basin_far_x"], basin_near_x=L["basin_near_x"]),
                  counter=dict(size=[COUNTER_D, COUNTER_W, COUNTER_H], back_unit=[BACK_D, COUNTER_W, BACK_H]))
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])}:geom{g}"
    contacts = sorted({(gname(c.geom1), gname(c.geom2)) for c in data.contact[:data.ncon]})
    report["contacts"] = [f"{a} <-> {b}" for a, b in contacts if not ({a, b} & {"floor"} and ("ankle_roll" in a or "ankle_roll" in b))]
    (HERE / f"layout{args.tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("layout", "ego_camera", "contacts")}, indent=1))

    rdir = Path(args.renders); rdir.mkdir(parents=True, exist_ok=True)
    opt = mujoco.MjvOption(); opt.geomgroup[3] = 1
    opt_axis = mujoco.MjvOption(); opt_axis.geomgroup[3] = 1; opt_axis.geomgroup[4] = 1
    ren = mujoco.Renderer(model, 900, 1600)
    for cam in ("overview", "front_left", "from_sink_side", "side", "bowl_closeup", "sink_closeup"):
        ren.update_scene(data, camera=cam, scene_option=opt_axis if cam in ("overview", "front_left", "from_sink_side", "side") else opt)
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
