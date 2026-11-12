"""Export the rot0 G1 + BrainCo Revo 2 teleop robot (MJCF) to URDF (2026-10-26).

    .venv/bin/python scripts/robots/export_urdf_revo2.py           # -> data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2.urdf
    .venv/bin/python scripts/robots/check_urdf_revo2.py            # FK of every link + depth renders vs the MJCF

The MJCF g1_rot0_teleop_revo2.xml is what the sim teleop runs and stays the source of truth; this URDF is generated from
it (re-run after any change to the MJCF: add_cameras_mounts.py, add_revo2_touch.py, the mount tuners). What it carries:

* kinematics: every MJCF body is a link (root = pelvis; the MJCF free joint is the commented floating base, as in the
  head-stereo G1 URDF); hinge joints -> revolute joints with the MJCF axis, range, damping and frictionloss; the finger
  distal joints follow the proximal ones through MJCF joint equalities -> <mimic multiplier offset>;
* inertia: each body's compiled mass, centre of mass and inertia (full tensor in the link frame, inertial rpy 0);
* geometry: visual = MuJoCo geom group 1, plus group-0 meshes that have no group-1 twin; collision = every colliding geom
  (contype/conaffinity != 0). Meshes by file (meshes/*.STL, the 3 mm-unit mount meshes with scale 0.001) at their
  authored placement (MuJoCo re-centres meshes at compile time; that offset is undone here), boxes / cylinders / spheres;
* frames: the MJCF sites as fixed links (head_stereo_{left,right}_optical_frame = the raw fisheye eyes,
  *_touch_site), the four recorded cameras as ROS optical frames (x right, y down, z forward; image-aligned: pixel u along
  +x, v along +y), the stock G1 sensor frames imu_link (= the MJCF site imu), d435_link, mid360_link copied from the
  head-stereo URDF, and pelvis_imu_link (the pelvis IMU the recorded states.robot.imu_* columns are defined at);
* effort limit = the MJCF actuator's ctrlrange (torque, N m); velocity limit from the source URDFs (G1 body joints:
  g1_29dof_with_hand_hbvcam_stereo.urdf; hand joints: g1_hand_brainco_module.urdf).

Not carried (no URDF field): joint armature, contact parameters, the 3 + 5 face cameras per wrist / eye (rendering
helpers), sensors, actuator gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ROBOT_DIR = ROOT / "data/robots/g1_rot0_teleop_revo2"
SRC = ROBOT_DIR / "g1_rot0_teleop_revo2_v3.xml"          # the current robot (v3, 2026-10-28); v1 = g1_rot0_teleop_revo2.xml
OUT = ROBOT_DIR / "g1_rot0_teleop_revo2.urdf"
G1_URDF = ROOT / "data/robots/g1_sonic/g1_29dof_with_hand_hbvcam_stereo.urdf"
HAND_URDF = ROOT / "data/robots/g1_revo2/g1_hand_brainco_module.urdf"
CAMERAS = ["hbvcam_left_pinhole", "hbvcam_right_pinhole", "wrist_left", "wrist_right"]
COPY_FRAMES = ["imu_link", "d435_link", "mid360_link"]
# skipped sites: com_marker (a viewer marker), *_touch_ray (proximity-ray helpers), imu (= the G1 URDF's imu_link, kept under
# that name), *_wrist_cam_optical (at the lens, but turned 180 deg about the viewing axis relative to the recorded wrist
# image; wrist_{left,right}_optical_frame below is the image-aligned frame)
SKIP_SITE = lambda n: n in ("com_marker", "imu") or n.endswith("_touch_ray") or n.endswith("_wrist_cam_optical")   # noqa: E731
RX_PI = np.diag([1.0, -1.0, -1.0])                                    # MuJoCo camera (-z forward, y up) -> ROS optical (z forward, y down)


def R_of(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, float)
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def rpy_of(R) -> np.ndarray:
    return Rot.from_matrix(np.asarray(R, float)).as_euler("xyz")      # URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)


def fmt(v) -> str:
    return " ".join("0" if abs(float(x)) < 5e-13 else f"{float(x):.12g}" for x in v)


def add_origin(el, p, R) -> None:
    ET.SubElement(el, "origin", xyz=fmt(p), rpy=fmt(rpy_of(R)))


def urdf_limits(path: Path, rename=lambda n: n) -> dict:
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        lim = j.find("limit")
        if lim is not None:
            out[rename(j.get("name"))] = {k: float(v) for k, v in lim.attrib.items()}
    return out


def build(src: Path = SRC) -> tuple[ET.Element, dict]:
    spec = mujoco.MjSpec.from_file(str(src))
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mesh_file = {ms.name: (ms.file, np.asarray(ms.scale, float)) for ms in spec.meshes}
    vel = urdf_limits(G1_URDF)
    vel.update(urdf_limits(HAND_URDF, rename=lambda n: n.replace("left_", "left_hand_", 1).replace("right_", "right_hand_", 1)
                           if any(f in n for f in ("thumb", "index", "middle", "ring", "pinky")) else n))
    report = dict(source=str(src.relative_to(ROOT)), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  links=0, revolute=0, fixed=0, mimic=0, visuals=0, collisions=0, frames=[], camera_frames={}, velocity_default=[])

    robot = ET.Element("robot", name="g1_rot0_teleop_revo2")
    robot.append(ET.Comment(
        f" Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/robots/export_urdf_revo2.py from {report['source']} "
        f"(sha256 {report['source_sha256'][:16]}). The MJCF is the source of truth: re-export after changing it. "))
    mj = ET.SubElement(robot, "mujoco")
    ET.SubElement(mj, "compiler", meshdir="meshes", discardvisual="false")
    robot.append(ET.Comment(' <joint name="floating_base_joint" type="floating"><parent link="world"/><child link="pelvis"/></joint> '))

    materials: dict[tuple, str] = {}

    def material(rgba) -> str:
        key = tuple(round(float(x), 4) for x in rgba)
        if key not in materials:
            materials[key] = f"mat_{len(materials):02d}"
            mel = ET.SubElement(robot, "material", name=materials[key])
            ET.SubElement(mel, "color", rgba=fmt(key))
        return materials[key]

    def geom_rgba(g):
        return m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]

    def authored_pose(g):
        """The geom's pose as written in the MJCF: for a mesh, undo the compiler's re-centring (mesh_pos / mesh_quat)."""
        p, R = m.geom_pos[g].copy(), R_of(m.geom_quat[g])
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            Ra = R @ R_of(m.mesh_quat[mid]).T
            return p - Ra @ m.mesh_pos[mid], Ra
        return p, R

    def geometry(parent_el, g):
        ge = ET.SubElement(parent_el, "geometry"); t = m.geom_type[g]; s = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            f, sc = mesh_file[m.mesh(m.geom_dataid[g]).name]
            attrs = dict(filename=f"meshes/{f}")
            if not np.allclose(sc, 1.0):
                attrs["scale"] = fmt(sc)
            ET.SubElement(ge, "mesh", **attrs)
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            ET.SubElement(ge, "box", size=fmt(2 * s[:3]))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ET.SubElement(ge, "cylinder", radius=fmt([s[0]]), length=fmt([2 * s[1]]))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ET.SubElement(ge, "sphere", radius=fmt([s[0]]))
        else:
            raise ValueError(f"geom {m.geom(g).name or g}: type {int(t)} has no URDF shape")

    def fixed_link(name, parent, p, R, kind):
        ET.SubElement(robot, "link", name=name)
        je = ET.SubElement(robot, "joint", name=f"{name}_joint", type="fixed")
        add_origin(je, p, R)
        ET.SubElement(je, "parent", link=parent); ET.SubElement(je, "child", link=name)
        report["frames"].append(dict(link=name, parent=parent, kind=kind, xyz=[round(float(v), 6) for v in p],
                                     rpy=[round(float(v), 6) for v in rpy_of(R)]))
        report["fixed"] += 1

    act_of = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    mimic = {int(m.eq_obj1id[e]): (int(m.eq_obj2id[e]), float(m.eq_data[e][0]), float(m.eq_data[e][1]))
             for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}

    for b in range(1, m.nbody):
        name = m.body(b).name
        link = ET.SubElement(robot, "link", name=name); report["links"] += 1
        if m.body_mass[b] > 0:                                         # full tensor in the link frame, inertial rpy 0: some readers
            ine = ET.SubElement(link, "inertial")                      # (MuJoCo's own URDF importer among them) ignore the inertial rpy
            ET.SubElement(ine, "origin", xyz=fmt(m.body_ipos[b]), rpy="0 0 0")
            ET.SubElement(ine, "mass", value=fmt([m.body_mass[b]]))
            Ri = R_of(m.body_iquat[b]); I = Ri @ np.diag(m.body_inertia[b]) @ Ri.T
            ET.SubElement(ine, "inertia", ixx=fmt([I[0, 0]]), ixy=fmt([I[0, 1]]), ixz=fmt([I[0, 2]]), iyy=fmt([I[1, 1]]), iyz=fmt([I[1, 2]]), izz=fmt([I[2, 2]]))
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        g1_meshes = {int(m.geom_dataid[g]) for g in geoms if m.geom_group[g] == 1 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
        for g in geoms:
            is_mesh = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            visible = geom_rgba(g)[3] > 0
            visual = visible and (m.geom_group[g] in (1, 2) or (m.geom_group[g] == 0 and is_mesh and int(m.geom_dataid[g]) not in g1_meshes))
            collide = bool(m.geom_contype[g] | m.geom_conaffinity[g])
            p, R = authored_pose(g)
            gname = m.geom(g).name
            if visual:
                ve = ET.SubElement(link, "visual", **({"name": gname} if gname else {}))
                add_origin(ve, p, R); geometry(ve, g)
                ET.SubElement(ve, "material", name=material(geom_rgba(g))); report["visuals"] += 1
            if collide:
                ce = ET.SubElement(link, "collision", **({"name": gname} if gname else {}))
                add_origin(ce, p, R); geometry(ce, g); report["collisions"] += 1
        parent = m.body_parentid[b]
        if parent == 0:
            continue                                                   # pelvis: the root (free joint in the MJCF)
        pj = m.body(parent).name
        hinge = [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if not hinge:
            je = ET.SubElement(robot, "joint", name=f"{name}_joint", type="fixed"); report["fixed"] += 1
        else:
            j = hinge[0]; jn = m.joint(j).name
            je = ET.SubElement(robot, "joint", name=jn, type="revolute" if m.jnt_limited[j] else "continuous"); report["revolute"] += 1
        add_origin(je, m.body_pos[b], R_of(m.body_quat[b]))
        ET.SubElement(je, "parent", link=pj); ET.SubElement(je, "child", link=name)
        if hinge:
            ET.SubElement(je, "axis", xyz=fmt(m.jnt_axis[j]))
            da = m.jnt_dofadr[j]
            a = act_of.get(j)
            effort = float(np.max(np.abs(m.actuator_ctrlrange[a]))) if a is not None and m.actuator_ctrllimited[a] else vel.get(jn, {}).get("effort", 0.0)
            v = vel.get(jn, {}).get("velocity")
            if v is None:
                v = 10.0; report["velocity_default"].append(jn)
            if m.jnt_limited[j]:
                ET.SubElement(je, "limit", lower=fmt([m.jnt_range[j, 0]]), upper=fmt([m.jnt_range[j, 1]]), effort=fmt([effort]), velocity=fmt([v]))
            ET.SubElement(je, "dynamics", damping=fmt([m.dof_damping[da]]), friction=fmt([m.dof_frictionloss[da]]))
            if j in mimic:
                j2, off, mult = mimic[j]
                ET.SubElement(je, "mimic", joint=m.joint(j2).name, multiplier=fmt([mult]), offset=fmt([off])); report["mimic"] += 1

    # sites -> fixed frames
    site_pose = {}
    for s in range(m.nsite):
        n = m.site(s).name
        if not n or SKIP_SITE(n):
            continue
        fixed_link(n, m.body(m.site_bodyid[s]).name, m.site_pos[s], R_of(m.site_quat[s]), "site")
        site_pose[n] = (d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy())
    # the recorded cameras as ROS optical frames, unless a site already is that frame
    for c in CAMERAS:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
        if cid < 0:
            continue
        wp, wR = d.cam_xpos[cid], d.cam_xmat[cid].reshape(3, 3) @ RX_PI
        same = next((n for n, (sp, sR) in site_pose.items() if np.allclose(sp, wp, atol=1e-6) and np.allclose(sR, wR, atol=1e-6)), None)
        if same:
            report["camera_frames"][c] = same
            continue
        fixed_link(f"{c}_optical_frame", m.body(m.cam_bodyid[cid]).name, m.cam_pos[cid], R_of(m.cam_quat[cid]) @ RX_PI, "camera")
        report["camera_frames"][c] = f"{c}_optical_frame"
    # stock G1 sensor frames (same torso as the head-stereo G1 URDF)
    g1 = ET.parse(G1_URDF).getroot()
    for j in g1.iter("joint"):
        child = j.find("child").get("link")
        if child in COPY_FRAMES:
            o = j.find("origin")
            p = np.array([float(v) for v in o.get("xyz").split()]); r = [float(v) for v in o.get("rpy").split()]
            fixed_link(child, j.find("parent").get("link"), p, Rot.from_euler("xyz", r).as_matrix(), "g1_urdf")
    # the frame the recorded real-format IMU columns are defined in (states.robot.imu_*, meta/realformat.json imu_frame):
    # the stock G1 imu_in_pelvis mount, as simple.teleop.holomotion_v14.realformat.PELVIS_IMU
    from simple.teleop.holomotion_v14.realformat import PELVIS_IMU
    fixed_link("pelvis_imu_link", "pelvis", np.asarray(PELVIS_IMU, float), np.eye(3), "realformat")
    # materials first (some URDF readers want them before use)
    mats = [e for e in robot if e.tag == "material"]
    for e in mats:
        robot.remove(e)
    for i, e in enumerate(mats):
        robot.insert(3 + i, e)
    return robot, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default=str(SRC)); ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    robot, rep = build(Path(a.src))
    ET.indent(robot, space="  ")
    Path(a.out).write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n")
    Path(a.out).with_suffix(".urdf.json").write_text(json.dumps(rep, indent=1))
    print(f"[urdf] {a.out}: {rep['links']} body links, {rep['revolute']} revolute ({rep['mimic']} mimic), {rep['fixed']} fixed joints, "
          f"{rep['visuals']} visuals, {rep['collisions']} collisions, {len(rep['frames'])} frames; cameras -> {rep['camera_frames']}"
          + (f"; velocity limit defaulted for {rep['velocity_default']}" if rep["velocity_default"] else ""))


if __name__ == "__main__":
    main()
