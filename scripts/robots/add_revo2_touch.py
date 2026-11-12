#!/usr/bin/env python
"""Adds the BrainCo Revo 2 Touch tactile pads to the rot0 teleop robot with Revo 2 hands.

Real hand (Revo 2 Touch, capacitive): one 6-channel capacitive unit under a pad cover on each of the five fingertips, no
palm or finger-pad sensing. Per finger it reports normal force, tangential force, tangential direction, proximity and a
status word (0.01 N units, 0-25 N, >50 Hz). The pad covers are the `*_touch_link.STL` meshes of BrainCo's own MJCF
(Revo2_xml.zip), attached to the fingertip body (right hand) or to the distal link (left hand) exactly as in that file,
so the sensing area in the simulator is the vendor's pad geometry.

What this script adds to data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2.xml (input: *_notouch.xml):
  * meshes  <side>_hand_<finger>_touch_mesh            (10 STL files copied from the vendor zip into meshes/)
  * bodies  <side>_hand_<finger>_touch_link            fixed child carrying the pad: visual + collision geom (black, like the hand)
  * sites   <side>_hand_<finger>_touch_site            box site at the pad's centre of curvature, z pointing INTO the finger (touch_grid
                                                       faces -z), x = toward the fingertip (the real sensor's 0 deg direction)
            <side>_hand_<finger>_touch_ray             on the pad surface, z OUT of the pad, for the rangefinder
  * sensors <side>_hand_<finger>_touch_total   <touch>        total normal force on the pad [N]
            <side>_hand_<finger>_touch_prox    <rangefinder>  distance from the pad surface to the nearest object along the outward normal [m],
                                                       -1 = none within 3 cm
            <side>_hand_<finger>_touch_grid    touch_grid     3 x 3 taxels x (normal, tangential x, tangential y) [N] in the site frame
  * <extension><plugin plugin="mujoco.sensor.touch_grid"/>
simple.sensors.revo2_touch.Revo2Touch turns these into the SDK's TouchFingerItem fields.
    .venv/bin/python scripts/robots/add_revo2_touch.py [--robot-dir data/robots/g1_rot0_teleop_revo2] [--check]
"""
import argparse, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
# BrainCo left-hand MJCF: the touch geoms sit on the distal links with these offsets; right hand: on the tip bodies, identity.
# 2026-10-28 (user: "why is the left hand thumb tactile sensor not at the finger tip?"): the vendor's LEFT THUMB offset is defined
# for a thumb-distal frame that is not ours (HoloMotion module URDF: left and right thumb distal links are the same geometry in the
# same frame, tip at +26.5 mm y), so it put the pad 42.5 mm behind the fingertip, on the proximal side of the distal joint (the
# four finger offsets agree with ours to 0.1 mm). The left thumb pad now takes the right pad's placement (ICP of the vendor left
# mesh onto the right pad in the right tip frame, 24 start rotations; chamfer 0.25 mm, max 0.9 mm; the left mesh is the right one
# mirrored in y, the pad is x-symmetric to 0.06 mm), written in the distal frame, see LEFT_THUMB_POSE_FIXED. --legacy-left-thumb
# reproduces the old placement (the v1 / v2 robot files of recordings made before 2026-10-28).
# Left thumb pose = the right pad's placement expressed in the (identical) distal frame: T_tip_right (0 0.026529 -0.0017734 | 0.999443 -0.0333688 0 0)
# o T_fit (-2.8e-05 -0.013587 0.008592 | -0.000457 -0.001363 0.040141 0.999193, the ICP result in the right tip frame). It is NOT put on the
# left tip link because that link's frame is turned ~180 deg about z relative to the right one (a first attempt on 10-28 did that and
# floated the pad 11 mm beyond the fingertip; the user caught it in the viewer).
LEFT_THUMB_POSE_FIXED = ("-2.8e-05 0.0135453 0.00770572", "0.000502227 0.00134699 -0.0734605 -0.997297")
LEFT_PAD_POSE_LEGACY_THUMB = ("0 -0.013836 0.0076686", "0.99618 0.0873286 0 0")
LEFT_PAD_POSE = {"thumb": LEFT_THUMB_POSE_FIXED, "index": ("0.014497 0 0.023303", "0.971859 0 0.235565 0"),
                 "middle": ("0.012493 0 0.028393", "0.984807 0 0.17365 0"), "ring": ("0.012028 0 0.027796", "0.984807 0 0.17365 0"),
                 "pinky": ("0.01401 0 0.021124", "0.971859 0 0.235565 0")}
PAD_RGBA = "0.1 0.1 0.1 1"          # the teleop hands are black; the real pad cover is light blue (0.53 0.81 0.92)
SITE_RGBA = "0.53 0.81 0.92 0.35"
PAD_MASS = 0.002


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    import mujoco
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


def pad_frame(verts_pad, axis_pad, axis_point=None):
    """Frames for one pad, in the pad body's frame. verts_pad: pad mesh vertices; axis_pad: unit vector along the finger
    toward the fingertip; axis_point: a point ON the finger axis in the pad body frame (the distal link's origin; the pad
    body's own origin for the right-hand pads, which are the tip links).  The site sits on the finger axis at the pad's
    axial position (its centre of curvature) with z pointing INTO the finger (MuJoCo's touch_grid taxels face the site's
    -z axis), x toward the fingertip (the real sensor's 0 deg direction), y = z cross x. A second frame (ray) has z
    pointing OUT of the pad for the rangefinder.
    2026-10-26 fix: the left-hand pad bodies (vendor LEFT_PAD_POSE) have their origin OUTSIDE the finger, so measuring the
    radial direction from the body origin pointed the frames into the finger; the axis point is now the distal origin."""
    c = verts_pad.mean(0); a0 = np.zeros(3) if axis_point is None else np.asarray(axis_point, float)
    rel = c - a0; axial = a0 + (rel @ axis_pad) * axis_pad; radial = c - axial; out = radial / np.linalg.norm(radial)
    x = axis_pad - (axis_pad @ out) * out; x /= np.linalg.norm(x)
    z_site = -out; y_site = np.cross(z_site, x); R_site = np.stack([x, y_site, z_site], 1)
    z_ray = out; y_ray = np.cross(z_ray, x); R_ray = np.stack([x, y_ray, z_ray], 1)
    v = verts_pad - axial
    vx, vy, vo = v @ x, v @ y_site, v @ out
    radius = float(np.median(np.linalg.norm(v[vo > np.percentile(vo, 50)], axis=1)))        # outer half of the shell
    ang_x = np.degrees(np.arctan2(vx, np.maximum(vo, 1e-6))); ang_y = np.degrees(np.arctan2(vy, np.maximum(vo, 1e-6)))
    ext = dict(x=[float(np.percentile(ang_x, 1)), float(np.percentile(ang_x, 99))], y=[float(np.percentile(ang_y, 1)), float(np.percentile(ang_y, 99))])
    half = np.abs(v @ R_site).max(0) + 0.0015                   # box site half sizes enclosing the pad (+1.5 mm), the <touch> zone
    return dict(centre=axial, R=R_site, R_ray=R_ray, radius=radius, extent_deg=ext, box_half=half, area_mm2=None)


def main():
    import mujoco, trimesh
    ap = argparse.ArgumentParser(); ap.add_argument("--robot-dir", default="data/robots/g1_rot0_teleop_revo2"); ap.add_argument("--check", action="store_true", help="only compile and report")
    ap.add_argument("--out", default="g1_rot0_teleop_revo2.xml", help="output file name in the robot dir (v3 chain: g1_rot0_teleop_revo2_v3.xml, then add_revo2_hand_selfcontact.py in place)")
    ap.add_argument("--legacy-left-thumb", action="store_true", help="the pre-2026-10-28 left thumb pad placement (vendor offset on the distal link)")
    ap.add_argument("--src", default="g1_rot0_teleop_revo2_notouch.xml", help="input file name in the robot dir (the robot without pad sensors)")
    ap.add_argument("--pads-from-model", action="store_true", help="2026-11-04, the official-hand robot: the <side>_hand_<finger>_touch_link bodies (pose, mesh) are already in --src "
                    "(make_g1_revo2.py --hand official puts BrainCo's fixed touch joints there); keep them and only add the named collision geom, sites and sensors")
    ap.add_argument("--tip-axis", action="store_true", help="2026-11-05: also give the legacy LEFT-hand pads (on the distal link) a finger axis through the tip link, "
                    "like the right hand; tight box sites instead of v3's 17 mm radial boxes. Changes the output -> write a new file name (v4), never v3")
    a = ap.parse_args(); rd = ROOT / a.robot_dir; src = rd / a.src; dst = rd / a.out
    assert src.exists(), f"{src} missing (the pristine robot without pads)"
    tree = ET.parse(src); root = tree.getroot()
    asset = root.find("asset"); sensor = root.find("sensor")
    if root.find("extension") is None:
        ext = ET.Element("extension"); ET.SubElement(ext, "plugin", plugin="mujoco.sensor.touch_grid"); root.insert(list(root).index(asset), ext)
    bodies = {b.get("name"): b for b in root.iter("body")}; parent_of = {c: p for p in root.iter() for c in p}
    report = {}
    for side in ["right", "left"]:
        for f in FINGERS:
            mesh_name = f"{side}_hand_{f}_touch_mesh"; stl = f"{side}_{f}_touch_link.STL"
            existing = bodies.get(f"{side}_hand_{f}_touch_link") if a.pads_from_model else None
            if existing is not None:                                       # the hand source placed the pad: its parent (distal link), pose and mesh
                parent = parent_of[existing]; pos, quat = existing.get("pos", "0 0 0"), existing.get("quat", "1 0 0 0"); on_tip = False
                gm = existing.find("geom")
                if gm is not None: mesh_name = gm.get("mesh"); stl = asset.find(f"mesh[@name='{mesh_name}']").get("file")
            else:
                assert not a.pads_from_model, f"{side}_hand_{f}_touch_link not in {src.name}"
                on_tip = side == "right"
                if on_tip: parent = bodies[f"{side}_hand_{f}_tip_link"]; pos, quat = "0 0 0", "1 0 0 0"
                else: parent = bodies[f"{side}_hand_{f}_distal_link"]; pos, quat = LEFT_PAD_POSE_LEGACY_THUMB if (f == "thumb" and a.legacy_left_thumb) else LEFT_PAD_POSE[f]
            if asset.find(f"mesh[@name='{mesh_name}']") is None: ET.SubElement(asset, "mesh", name=mesh_name, file=stl)
            # finger axis (toward the tip) in the distal frame: fingers run along +z of the distal link, the legacy thumb along +y;
            # with the pads from the model it is the distal axis the tip link lies along (BrainCo's thumb runs along -y)
            if existing is not None:
                tp = np.array([float(v) for v in parent.find(f"body[@name='{side}_hand_{f}_tip_link']").get("pos").split()])
                k = int(np.argmax(np.abs(tp))); axis_distal = np.zeros(3); axis_distal[k] = np.sign(tp[k])
            else:
                axis_distal = np.array([0, 1.0, 0]) if f == "thumb" else np.array([0, 0, 1.0])
            if on_tip:
                tipq = np.array([float(v) for v in parent.get("quat", "1 0 0 0").split()]); R_dt = quat_to_mat(tipq)   # distal -> tip
                R_tp = quat_to_mat(np.array([float(v) for v in quat.split()]))                                       # tip -> pad body
                axis_pad = R_tp.T @ (R_dt.T @ axis_distal)
                axis_point = -R_tp.T @ np.array([float(v) for v in pos.split()])                 # the tip link's origin (on the finger axis), in the pad body frame
            else:
                R_dp = quat_to_mat(np.array([float(v) for v in quat.split()])); axis_pad = R_dp.T @ axis_distal
                pad_pos = np.array([float(v) for v in pos.split()])
                if existing is not None or a.tip_axis:
                    # 2026-11-05 fix: the finger axis must pass through the TIP link (as for the right-hand pads, which sit on it), not
                    # through the distal link's origin: the tip lies 17 deg off the distal +z, so a line from the distal origin missed
                    # the pad by 11 mm and the box site came out centred inside the finger with a 17 mm radial half size (v3 left hand,
                    # first official build). With the tip as the axis point the official fingers get the legacy right-hand boxes exactly.
                    tp_ = np.array([float(v) for v in parent.find(f"body[@name='{side}_hand_{f}_tip_link']").get("pos").split()])
                    axis_point = R_dp.T @ (tp_ - pad_pos)                                        # the tip link's origin, in the pad body frame
                else:
                    axis_point = -R_dp.T @ pad_pos                                                # v3 behaviour: the distal link's origin (kept so the v3 file reproduces)
            tm = trimesh.load(rd / "meshes" / stl, force="mesh"); V = np.asarray(tm.vertices, float)
            fr = pad_frame(V, axis_pad, axis_point); c = fr["centre"]; q = mat_to_quat(fr["R"])
            # the vendor meshes are closed shells: outer area ~ half the total area
            fr["area_mm2"] = round(float(tm.area) * 1e6 / 2, 0)
            name = f"{side}_hand_{f}_touch_link"
            old = parent.find(f"body[@name='{name}']")
            if old is not None: parent.remove(old)
            b = ET.SubElement(parent, "body", name=name, pos=pos, quat=quat)
            ET.SubElement(b, "inertial", pos=" ".join(f"{v:.6g}" for v in V.mean(0)), mass=str(PAD_MASS), diaginertia="2e-8 2e-8 2e-8")
            ET.SubElement(b, "geom", type="mesh", mesh=mesh_name, rgba=PAD_RGBA, contype="0", conaffinity="0", group="1")
            ET.SubElement(b, "geom", name=f"{side}_hand_{f}_touch_collision", type="mesh", mesh=mesh_name, rgba=PAD_RGBA)
            ET.SubElement(b, "site", name=f"{side}_hand_{f}_touch_site", type="box", pos=" ".join(f"{v:.6g}" for v in c), quat=" ".join(f"{v:.6g}" for v in q),
                          size=" ".join(f"{v:.5g}" for v in fr["box_half"]), group="4", rgba=SITE_RGBA)
            qr = mat_to_quat(fr["R_ray"]); out = fr["R_ray"][:, 2]
            hits, _, _ = tm.ray.intersects_location(c[None, :], out[None, :])                           # the pad's outer surface along the outward normal
            r_surf = float(((hits - c) @ out).max()) if len(hits) else fr["radius"]
            RAY_INSET = 0.001; ray_pos = c + out * (r_surf - RAY_INSET)    # 1 mm inside the pad shell: outside the finger's own links (which a rangefinder cannot exclude), never inside a touching object
            ET.SubElement(b, "site", name=f"{side}_hand_{f}_touch_ray", type="sphere", pos=" ".join(f"{v:.6g}" for v in ray_pos), quat=" ".join(f"{v:.6g}" for v in qr), size="0.001", group="4", rgba="0.9 0.5 0.1 0.6")
            # sensors
            for tag in ("touch", "rangefinder", "plugin"):
                for s in list(sensor.findall(f"{tag}")):
                    if (s.get("name") or "").startswith(f"{side}_hand_{f}_touch_"): sensor.remove(s)
            ET.SubElement(sensor, "touch", name=f"{side}_hand_{f}_touch_total", site=f"{side}_hand_{f}_touch_site")
            ET.SubElement(sensor, "rangefinder", name=f"{side}_hand_{f}_touch_prox", site=f"{side}_hand_{f}_touch_ray", cutoff="0.03")
            ex = fr["extent_deg"]; fov_x = min(90.0, max(20.0, 1.15 * max(abs(ex["x"][0]), abs(ex["x"][1])))); fov_y = min(90.0, max(20.0, 1.15 * max(abs(ex["y"][0]), abs(ex["y"][1]))))
            pl = ET.SubElement(sensor, "plugin", name=f"{side}_hand_{f}_touch_grid", plugin="mujoco.sensor.touch_grid", objtype="site", objname=f"{side}_hand_{f}_touch_site")
            for k, v in [("size", "3 3"), ("fov", f"{fov_x:.0f} {fov_y:.0f}"), ("gamma", "0"), ("nchannel", "3")]: ET.SubElement(pl, "config", key=k, value=v)
            report[name] = dict(parent=parent.get("name"), pad_pos=pos, pad_quat=quat, site_centre=np.round(c, 4).tolist(), site_quat=np.round(q, 4).tolist(), ray_pos=np.round(ray_pos, 5).tolist(), ray_quat=np.round(qr, 4).tolist(), ray_inset_m=RAY_INSET, surface_r_mm=round(1000 * r_surf, 2), pad_radius_mm=round(fr["radius"] * 1000, 1),
                                box_half_mm=np.round(fr["box_half"] * 1000, 1).tolist(), fov_deg=[round(fov_x), round(fov_y)], extent_deg={k: [round(v[0]), round(v[1])] for k, v in ex.items()},
                                outer_area_mm2=fr["area_mm2"], mesh_bounds_mm=np.round(np.c_[V.min(0), V.max(0)].T * 1000, 1).tolist())
    ET.indent(tree, space="  ")
    if not a.check: tree.write(dst, encoding="unicode", xml_declaration=False); print(f"[touch] wrote {dst}")
    m = mujoco.MjModel.from_xml_path(str(dst if not a.check else src)); print(f"[touch] compiles: nbody {m.nbody} ngeom {m.ngeom} nsensor {m.nsensor} nsensordata {m.nsensordata}")
    (rd / "touch_pads.json").write_text(json.dumps(report, indent=1)); print(f"[touch] pad report -> {rd / 'touch_pads.json'}")
    for k, v in report.items(): print(f"  {k}: on {v['parent']}, site centre {v['site_centre']} radius {v['pad_radius_mm']} mm, fov {v['fov_deg']}, outer area {v['outer_area_mm2']:.0f} mm2")


if __name__ == "__main__":
    main()
