#!/usr/bin/env python
"""Build the rot0 teleop robot with BrainCo Revo 2 hands in place of the Dex3 hands (2026-10-23, user: "create one more robot
which you use brainco hand to replace the dex3 hands").

Sources: data/robots/g1_rot0_teleop/g1_rot0_teleop.xml (the frozen rot0 robot: HBVCAM stereo head at rot0, 3.2 kg backpack,
black Dex3 hands); the HoloMotion G1+BrainCo module data/robots/g1_revo2/g1_hand_brainco_module.urdf (HorizonRobotics/HoloMotion_assets
on Hugging Face: the Unitree wrist chain, then a fixed base at 0.0591 m from wrist_yaw and the Revo 2 finger chains with
limits, inertials and the distal = k x proximal couplings) and the Revo 2 meshes data/robots/g1_revo2/meshes (Unitree
xr_teleoperate, assets/brainco_hand/meshes). Under each <side>_wrist_yaw_link the Dex3 finger bodies and palm geoms are
removed and the Revo 2 is attached; names get the <side>_hand_ prefix (the engine finds hand joints by "left_hand"/"right_hand",
the kit gates find the right hand's geoms by the "right_hand_" body prefix). Every hand joint gets a torque actuator in tree
order (the engine writes PD torques by joint index), 11 per hand; the distal joints are coupled by equality constraints as on
the real hand. The adapter plate (base2, 23 g) has no mesh here: a black disc stands in. Output:
data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2.xml + meshes/ (rot0 meshes + Revo 2 meshes) + package.json.
    .venv/bin/python scripts/robots/make_g1_revo2.py                                     # legacy hand, unchanged output

2026-11-04 (user: "create a new version of the robot with the official hand ... make this new robot an option"): --hand official
takes the finger chains from BrainCo's own description package instead, data/robots/g1_revo2_official (= BrainCoTech/
revo2_description(_ros1) v1.1.1 of 2026-08-20, see its SOURCE.md). The mount is still the HoloMotion module's (adapter at
41.5 mm, hand base at 59.1 mm from wrist_yaw, same base frame: the module's joint origins equal BrainCo's to 0.01 mm); joints,
limits, the per-finger couplings (1.131 / 1.267 / 1.252 / 1.271, thumb 1.0945), the thumb zero offsets (+3 deg abduction,
+9 / +6 deg MCP / PIP), inertials, the re-exported meshes (visual + collision where BrainCo has both) and the fingertip pad
links (`*_touch_link`, placed as the fixed touch joints say; add_revo2_touch.py --pads-from-model adds the sensors) come
from the official URDFs. --src may be the legacy robot WITH its cameras (g1_rot0_teleop_revo2_notouch.xml): its Revo 2
hands (bodies, equalities, meshes, excludes) are swapped out and the camera mounts and the silver adapter ring are kept, so
steps 1 + 2 of the legacy chain collapse into one.
    .venv/bin/python scripts/robots/make_g1_revo2.py --hand official --src data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_notouch.xml \
        --out-dir data/robots/g1_rot0_teleop_revo2_official --out-name g1_rot0_teleop_revo2_official_notouch.xml
    .venv/bin/python scripts/robots/add_revo2_touch.py --robot-dir data/robots/g1_rot0_teleop_revo2_official --src g1_rot0_teleop_revo2_official_notouch.xml \
        --pads-from-model --out g1_rot0_teleop_revo2_official.xml
    .venv/bin/python scripts/robots/add_revo2_hand_selfcontact.py --src <that file> --out <that file>
"""
import argparse, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DEFAULT = ROOT / "data/robots/g1_rot0_teleop/g1_rot0_teleop.xml"; MOD = ROOT / "data/robots/g1_revo2/g1_hand_brainco_module.urdf"; MESH = ROOT / "data/robots/g1_revo2/meshes"
OFFICIAL = ROOT / "data/robots/g1_revo2_official"                     # BrainCo revo2_description v1.1.1 (urdf/ + meshes/), see SOURCE.md there
OUT_DEFAULT = ROOT / "data/robots/g1_rot0_teleop_revo2"
BLACK = "0.1 0.1 0.1 1"; SIDES = ("left", "right")
HAND_LABEL = {"legacy": "BrainCo Revo 2 (HoloMotion G1+BrainCo module kinematics; Unitree xr_teleoperate meshes)",
              "official": "BrainCo Revo 2 (BrainCoTech/revo2_description v1.1.1 URDFs + meshes, 2026-08-20; mount from the HoloMotion module)"}


def rpy_quat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)   # w x y z, URDF rpy = Rz(y) Ry(p) Rx(r)


def fmt(v): return " ".join(f"{float(x):.6g}" for x in v)


def hand_name(side, urdf_name):
    """left_thumb_metacarpal_Link -> left_hand_thumb_metacarpal_link; right_thumb_tip -> right_hand_thumb_tip_link"""
    rest = urdf_name[len(side) + 1:].lower()
    if rest.endswith("_link"): rest = rest[:-5]
    if rest.endswith("_joint"): return f"{side}_hand_{rest}"
    return f"{side}_hand_{rest}_link"


class HandSource:
    """The finger chains of both hands: `links` (URDF link elements by name), `joints` (URDF joint elements in file order) and
    where their mesh files live. legacy = the HoloMotion module URDF; official = BrainCo's revo2_{side}_hand.urdf (v1.1.1).
    The mount (base joint, adapter link) always comes from the module (`mod_links`, `mod_joints`)."""

    def __init__(self, variant: str):
        self.variant = variant; mod = ET.parse(MOD).getroot()
        self.mod_links = {l.get("name"): l for l in mod.findall("link")}; self.mod_joints = mod.findall("joint")
        if variant == "legacy":
            self.links, self.joints, self.mesh_dirs = self.mod_links, self.mod_joints, [MESH]
        else:
            self.links, self.joints = {}, []
            for side in SIDES:
                r = ET.parse(OFFICIAL / f"urdf/revo2_{side}_hand.urdf").getroot()
                self.links.update({l.get("name"): l for l in r.findall("link")}); self.joints += r.findall("joint")
            self.mesh_dirs = [OFFICIAL / "meshes"]

    def mesh_file(self, basename: str) -> Path:
        for d in self.mesh_dirs:
            hits = sorted(d.rglob(basename))
            if hits: return hits[0]
        raise FileNotFoundError(basename)

    def meshes(self, link_name: str):
        """(visual mesh files, collision mesh files) of a link; legacy: the link's first mesh for both, as before."""
        l = self.links[link_name]
        if self.variant == "legacy":
            f = os.path.basename(next(l.iter("mesh")).get("filename")); return [f], [f]
        vis = [os.path.basename(m.get("filename")) for v in l.findall("visual") for m in v.iter("mesh")]
        col = [os.path.basename(m.get("filename")) for c in l.findall("collision") for m in c.iter("mesh")]
        return vis, col


PINCH_N = 15.0          # Revo 2 single-finger pinch force (retailer specs: 15 N pinch, 50 N whole-hand grip, 2026-10-23 search)
CLOSE_RAD_S = 2.3       # the module URDF's finger velocity limit (rad/s): joint damping = torque limit / this


def scale_finger_limits(tree, xml_path):
    """Per finger joint: torque limit = PINCH_N x lever to that finger's tip (the module URDF's 2 N m would be over 30 N at the
    fingertip), never above the URDF's value; the thumb's metacarpal (abduction) keeps the URDF's limit. Damping = limit /
    CLOSE_RAD_S so a free finger closes at about the URDF's velocity limit instead of snapping shut. Writes the joint's
    actuatorfrcrange + damping and the actuator's ctrlrange. Returns {joint: (lever_m, limit_Nm)} for the right hand."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    jn = {j.get("name"): j for j in tree.getroot().iter("joint") if j.get("name")}; an = {a.get("joint"): a for a in tree.getroot().find("actuator")}
    out = {}
    for name, j in jn.items():
        if "_hand_" not in name or "metacarpal" in name: continue
        side, finger = name.split("_hand_")[0], name.split("_hand_")[1].rsplit("_", 2)[0]        # right_hand_index_proximal_joint -> index
        jid = m.joint(name).id; anchor = d.xanchor[jid]; tip = d.xpos[m.body(f"{side}_hand_{finger}_tip_link").id]
        lever = float(np.linalg.norm(tip - anchor)); urdf = float(j.get("actuatorfrcrange").split()[1])
        lim = round(min(PINCH_N * lever, urdf), 3); damp = round(lim / CLOSE_RAD_S, 3)
        j.set("actuatorfrcrange", f"-{lim:g} {lim:g}"); j.set("damping", f"{damp:g}"); an[name].set("ctrlrange", f"-{lim:g} {lim:g}")
        if side == "right": out[name] = dict(lever_m=round(lever, 4), limit_Nm=lim, damping=damp, urdf_effort=urdf)
    return out


def main():
    ap = argparse.ArgumentParser(description="rot0 teleop robot with BrainCo Revo 2 hands")
    ap.add_argument("--hand", default="legacy", choices=["legacy", "official"], help="legacy: HoloMotion module kinematics + Unitree meshes (the 10-23 robot); official: BrainCo revo2_description v1.1.1")
    ap.add_argument("--src", default=str(SRC_DEFAULT), help="robot to put the hands on: the rot0 Dex3 robot, or a Revo 2 robot whose hands get swapped (its cameras/mounts stay)")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT)); ap.add_argument("--out-name", default="g1_rot0_teleop_revo2.xml")
    a = ap.parse_args(); SRC = Path(a.src).resolve(); OUT = Path(a.out_dir).resolve(); OUT_XML = OUT / a.out_name; hs = HandSource(a.hand)
    tree = ET.parse(SRC); root = tree.getroot()
    asset = root.find("asset"); wb = root.find("worldbody"); act = root.find("actuator")
    bodies = {b.get("name"): b for b in root.iter("body")}
    removed = set(); mesh_added = []
    eq = root.find("equality")
    if eq is None: eq = ET.SubElement(root, "equality")
    for m in list(asset.findall("mesh")):                               # every old hand mesh asset (Dex3's, or a previous Revo 2's incl. pads): the new hand's are added fresh
        if "_hand_" in m.get("name", ""): asset.remove(m)
    wrist_joints = {f"{s}_wrist_{k}_joint" for s in SIDES for k in ("roll", "pitch", "yaw")}
    for side in SIDES:                                                     # pass 1: remove the old hands
        wrist = bodies[f"{side}_wrist_yaw_link"]
        # ---- remove the old hand: the Dex3 fingers and palm, or a previous Revo 2 (its adapter body is kept when it is there)
        for b in list(wrist.findall("body")):
            n = b.get("name", "")
            if n.startswith(f"{side}_hand_") and n != f"{side}_hand_base2_link":
                removed.update(x.get("name") for x in b.iter("body")); removed.update(x.get("name") for x in b.iter("joint")); wrist.remove(b)
        for g in list(wrist.findall("geom")):
            if g.get("mesh") == f"{side}_hand_palm_link": wrist.remove(g)
    # ---- sensors / contact excludes / keyframes / equalities that name the removed bodies or joints (a previous Revo 2's couplings
    # and excludes; the new hand reuses the names, so this runs before anything is added)
    for sec in ("sensor", "contact", "keyframe", "equality"):
        s_ = root.find(sec)
        if s_ is None: continue
        for e in list(s_):
            if any(v in removed for v in e.attrib.values()): s_.remove(e)
    for side in SIDES:                                                     # pass 2: the new hands
        wrist = bodies[f"{side}_wrist_yaw_link"]
        # ---- adapter plate (no mesh available): black disc + the module's mass; a source that already has one keeps it
        if wrist.find(f"body[@name='{side}_hand_base2_link']") is None:
            b2 = hs.mod_links[f"{side}_base2_link"]; ine = b2.find("inertial")
            plate = ET.SubElement(wrist, "body", name=f"{side}_hand_base2_link", pos="0.0415 0 0")
            ET.SubElement(plate, "inertial", pos=fmt(ine.find("origin").get("xyz").split()), mass=ine.find("mass").get("value"), diaginertia="2e-6 2e-6 2e-6")
            ET.SubElement(plate, "geom", type="cylinder", size="0.027 0.004", quat="0.7071068 0 0.7071068 0", rgba=BLACK, contype="0", conaffinity="0", group="1")
        # ---- hand base (the module's mount frame; the hand's own base link inertial + meshes)
        base_j = next(j for j in hs.mod_joints if j.get("name") == f"{side}_base_joint"); o = base_j.find("origin")
        q = rpy_quat([float(v) for v in o.get("rpy").split()])
        base = ET.SubElement(wrist, "body", name=f"{side}_hand_base_link", pos=fmt(o.get("xyz").split()), quat=fmt(q))
        add_inertial(base, hs.links[f"{side}_base_link"]); add_geoms(base, f"{side}_base_link", side, asset, mesh_added, hs)
        # ---- finger chains, depth first from the hand source's joint list (official: fixed tip / touch joints become plain bodies)
        body_of = {f"{side}_base_link": base}
        for j in hs.joints:
            name = j.get("name")
            if not name.startswith(side + "_") or name in wrist_joints: continue
            if j.get("type") == "fixed" and hs.variant == "legacy": continue
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent not in body_of: continue
            o = j.find("origin"); lim = j.find("limit"); ax = j.find("axis")
            xyz = o.get("xyz").split() if o is not None else ["0", "0", "0"]; rpy = [float(v) for v in o.get("rpy").split()] if o is not None else [0, 0, 0]
            nb = ET.SubElement(body_of[parent], "body", name=hand_name(side, child), pos=fmt(xyz), quat=fmt(rpy_quat(rpy)))
            lo, hi = (float(lim.get("lower")), float(lim.get("upper"))) if (lim is not None and j.get("type") != "fixed") else (0.0, 0.0)
            if hi > lo:                                                    # a real joint (tips have 0..0 or are fixed)
                ET.SubElement(nb, "joint", name=hand_name(side, name), **{"class": "finger_motor"}, pos="0 0 0", axis=ax.get("xyz"), range=f"{lo:g} {hi:g}",
                              actuatorfrcrange=f"-{float(lim.get('effort')):g} {float(lim.get('effort')):g}")
            add_inertial(nb, hs.links[child]); add_geoms(nb, child, side, asset, mesh_added, hs)
            body_of[child] = nb
        # ---- couplings (mimic joints)
        for j in hs.joints:
            m = j.find("mimic")
            if m is not None and j.get("name").startswith(side + "_"):
                ET.SubElement(eq, "joint", joint1=hand_name(side, j.get("name")), joint2=hand_name(side, m.get("joint")), polycoef=f"{float(m.get('offset', 0)):g} {float(m.get('multiplier', 1)):g} 0 0 0")
    # ---- actuators: one per joint in tree order (the engine writes torques by joint index); body ones copied, hand ones new
    old = {a_.get("joint"): a_ for a_ in act}
    for a_ in list(act): act.remove(a_)
    hand_joints = []
    for j in root.iter("joint"):
        n = j.get("name") or ""                                        # the pelvis free joint has no name
        if n in old and n not in removed: act.append(old[n])
        elif "_hand_" in n:
            e = float(j.get("actuatorfrcrange").split()[1]); ET.SubElement(act, "general", name=n, joint=n, ctrlrange=f"-{e:g} {e:g}"); hand_joints.append(n)
    # ---- self-collision excludes: MuJoCo collides each mesh as its convex hull, and the palm's hull fills the thumb's cut-out,
    # so the thumb proximal / distal links (grandchildren of the palm, not auto-excluded like parent-child pairs) are jammed
    # against the palm at every pose (2026-10-23: the thumb could not move between 0.42 and 0.72 rad). The real thumb sits
    # beside the palm; the fingertips keep colliding with the palm so a fist still stops there.
    contact = root.find("contact")
    if contact is None: contact = ET.SubElement(root, "contact")
    for side in SIDES:
        for link in ("thumb_proximal_link", "thumb_distal_link"):
            ET.SubElement(contact, "exclude", body1=f"{side}_hand_base_link", body2=f"{side}_hand_{link}")
        ET.SubElement(contact, "exclude", body1=f"{side}_hand_thumb_metacarpal_link", body2=f"{side}_hand_thumb_distal_link")
    # ---- meshes: the source robot's (minus every old hand mesh: Dex3's or a previous Revo 2's) + the new hand's
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "meshes").mkdir(exist_ok=True)
    for m in list(asset.findall("mesh")):
        if m.get("file") in mesh_added: continue
        src = SRC.parent / "meshes" / m.get("file")
        if not src.exists(): src = MESH / m.get("file")
        shutil.copy2(src, OUT / "meshes" / m.get("file"))
    for f in mesh_added: shutil.copy2(hs.mesh_file(f), OUT / "meshes" / f)
    ET.indent(tree, space="  "); tree.write(OUT_XML, encoding="unicode", xml_declaration=False)
    limits = scale_finger_limits(tree, OUT_XML)
    ET.indent(tree, space="  "); tree.write(OUT_XML, encoding="unicode", xml_declaration=False)
    # ---- check: compile, counts, mass, fingertip reach
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT_XML)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)]
    lh = [n for n in names if "left_hand" in n]; rh = [n for n in names if "right_hand" in n]
    act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    order_ok = all(act_names[i] == names[i + 1] for i in range(m.nu))     # joint 0 is the free joint
    hand_mass = {s: sum(float(m.body_mass[b]) for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{s}_hand_")) for s in SIDES}
    base = m.body("right_hand_base_link").id; tip = m.body("right_hand_index_tip_link").id; thumb = m.body("right_hand_thumb_tip_link").id
    ranges = {n: [round(float(x), 5) for x in m.jnt_range[m.joint(n).id]] for n in rh}
    coupl = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[e]): round(float(m.eq_data[e][1]), 4) for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}
    rep = dict(xml=str(OUT_XML.relative_to(ROOT)), nq=int(m.nq), nv=int(m.nv), nu=int(m.nu), nbody=int(m.nbody), njnt=int(m.njnt), left_hand_joints=lh, right_hand_joints=rh,
               actuators_in_joint_order=bool(order_ok), hand_mass_kg={k: round(v, 3) for k, v in hand_mass.items()}, total_mass_kg=round(float(m.body_subtreemass[m.body("pelvis").id]), 3),
               right_index_tip_from_base_m=np.round(d.xpos[tip] - d.xpos[base], 3).tolist(), right_thumb_tip_from_base_m=np.round(d.xpos[thumb] - d.xpos[base], 3).tolist(),
               equality_constraints=int(m.neq), joint_ranges_rad=ranges, couplings=coupl, removed=sorted(removed), meshes=len(list((OUT / "meshes").glob("*"))), finger_limits=limits)
    extra = {}
    if a.hand == "official":
        extra = dict(official_source=str((OFFICIAL / "SOURCE.md").relative_to(ROOT)), official_urdf_sha256={s: hashlib.sha256((OFFICIAL / f"urdf/revo2_{s}_hand.urdf").read_bytes()).hexdigest()[:16] for s in SIDES})
    json.dump(dict(name=OUT_XML.name, made="2026-10-23" if a.hand == "legacy" else time.strftime("%Y-%m-%d"), source_robot=str(SRC.relative_to(ROOT)),
                   hand=HAND_LABEL[a.hand], hand_variant=a.hand, **extra, **rep), open(OUT / "package.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("removed",)}, indent=1))


def rpy_mat(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]]); Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]]); return Rz @ Ry @ Rx


def add_inertial(body, link):
    """URDF inertial (tensor in its own origin frame) -> MJCF fullinertia expressed in the body frame (MuJoCo refuses a tensor
    plus an orientation), at the same centre of mass. A link without inertial / without mass (the official tip frames) gets
    a negligible 1 mg so the body compiles like the legacy 1 g tips."""
    ine = link.find("inertial")
    if ine is None or float(ine.find("mass").get("value")) <= 0:
        body.insert(0, ET.Element("inertial", pos="0 0 0", mass="1e-6", diaginertia="1e-9 1e-9 1e-9")); return
    o = ine.find("origin"); I = ine.find("inertia")
    ixx, iyy, izz, ixy, ixz, iyz = (float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
    R = rpy_mat([float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()])
    Ib = R @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ R.T
    body.insert(0, ET.Element("inertial", pos=fmt((o.get("xyz") if o is not None else "0 0 0").split()), mass=ine.find("mass").get("value"),
                              fullinertia=fmt([Ib[0, 0], Ib[1, 1], Ib[2, 2], Ib[0, 1], Ib[0, 2], Ib[1, 2]])))


def add_geoms(body, link_name, side, asset, mesh_added, hs: HandSource):
    """visual (group 1, no contact) + collision mesh geoms, black. The link's primary mesh (its collision mesh) is the asset
    <side>_hand_<part>_mesh as before; extra visual shells (BrainCo's *_visual_link.STL) get their own asset by file name.
    A link without meshes (the official tip frames) gets no geom."""
    vis, col = hs.meshes(link_name)
    if not vis and not col: return
    primary = (col or vis)[0]; primary_name = hand_name(side, link_name).replace("_link", "") + "_mesh"

    def ensure(f):
        name = primary_name if f == primary else hand_name(side, Path(f).stem).replace("_link", "") + "_mesh"
        if f not in mesh_added: mesh_added.append(f)
        if asset.find(f"mesh[@name='{name}']") is None: ET.SubElement(asset, "mesh", name=name, file=f)
        return name
    for f in (vis or col): ET.SubElement(body, "geom", type="mesh", mesh=ensure(f), rgba=BLACK, contype="0", conaffinity="0", group="1")
    for f in (col or vis): ET.SubElement(body, "geom", type="mesh", mesh=ensure(f), rgba=BLACK)


if __name__ == "__main__":
    main()
