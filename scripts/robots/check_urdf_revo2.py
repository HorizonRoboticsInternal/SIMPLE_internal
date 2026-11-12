"""Checks the URDF exported by export_urdf_revo2.py against the MJCF it came from (2026-10-26).

    MUJOCO_GL=egl .venv/bin/python scripts/robots/check_urdf_revo2.py [--configs 50] [--out <dir>]

1. kinematics: the URDF is loaded by MuJoCo's URDF importer (fixed links kept); at random joint configurations (distal =
   mimic of proximal) the world pose of every link and frame must equal the MJCF body / site / camera pose;
2. inertia: per link mass, world centre of mass and world inertia tensor;
3. cameras: red marker spheres placed at known points of each recorded camera's optical frame are rendered through the
   recorder's own image pipelines (rectified pinhole pair 1280x720 -> 640x360; wrist 150-deg equidistant fisheye) and
   must land where the documented intrinsics put them (meta/holomotion_raw.json "images");
4. geometry: depth renders of the visual geometry of both models from 6 viewpoints must agree.
Prints PASS/FAIL per check and writes <out>/urdf_check.json (+ depth / marker images).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ROBOT_DIR = ROOT / "data/robots/g1_rot0_teleop_revo2"
MJCF = ROBOT_DIR / "g1_rot0_teleop_revo2_v3.xml"
URDF = ROBOT_DIR / "g1_rot0_teleop_revo2.urdf"
RX_PI = np.diag([1.0, -1.0, -1.0])


def load_urdf_in_mujoco(path: Path) -> mujoco.MjModel:
    txt = path.read_text()
    txt = re.sub(r"<compiler [^>]*/>", f'<compiler meshdir="{ROBOT_DIR / "meshes"}" discardvisual="false" fusestatic="false" '
                                       f'balanceinertia="false"/>', txt, count=1)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(txt); tmp = f.name
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--configs", type=int, default=50); ap.add_argument("--out", default=str(ROBOT_DIR / "urdf_check"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mj = mujoco.MjModel.from_xml_path(str(MJCF)); dj = mujoco.MjData(mj)
    mu = load_urdf_in_mujoco(URDF); du = mujoco.MjData(mu)
    rep = dict(urdf=str(URDF.relative_to(ROOT)), mjcf=str(MJCF.relative_to(ROOT)), urdf_bodies=mu.nbody, urdf_joints=mu.njnt, urdf_eq=mu.neq)
    print(f"[check] URDF in MuJoCo: {mu.nbody} bodies, {mu.njnt} joints ({mu.neq} equality from <mimic>), {mu.ngeom} geoms")

    # ---- 1. kinematics -------------------------------------------------------------------------------------------
    hinge = [mj.joint(j).name for j in range(mj.njnt) if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    mimic = {mj.joint(mj.eq_obj1id[e]).name: (mj.joint(mj.eq_obj2id[e]).name, mj.eq_data[e][0], mj.eq_data[e][1]) for e in range(mj.neq)}
    rng = np.random.default_rng(0)
    frames = {}                                                         # urdf body name -> function(dj) -> (pos, R)
    for b in range(1, mj.nbody):
        frames[mj.body(b).name] = (lambda b: lambda: (dj.xpos[b], dj.xmat[b].reshape(3, 3)))(b)
    for s in range(mj.nsite):
        frames[mj.site(s).name] = (lambda s: lambda: (dj.site_xpos[s], dj.site_xmat[s].reshape(3, 3)))(s)
    for c in range(mj.ncam):
        frames[mj.camera(c).name + "_optical_frame"] = (lambda c: lambda: (dj.cam_xpos[c], dj.cam_xmat[c].reshape(3, 3) @ RX_PI))(c)
    common = [n for n in frames if mujoco.mj_name2id(mu, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
    worst_p = worst_R = 0.0
    for k in range(a.configs):
        q = {}
        for n in hinge:
            if n in mimic:
                continue
            j = mj.joint(n).id; lo, hi = mj.jnt_range[j] if mj.jnt_limited[j] else (-np.pi, np.pi)
            q[n] = rng.uniform(lo, hi) if k else 0.0
        for n, (src, off, mult) in mimic.items():
            q[n] = off + mult * q[src]
        dj.qpos[:] = mj.qpos0; dj.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        du.qpos[:] = mu.qpos0
        for n, v in q.items():
            dj.qpos[mj.jnt_qposadr[mj.joint(n).id]] = v
            du.qpos[mu.jnt_qposadr[mu.joint(n).id]] = v
        mujoco.mj_fwdPosition(mj, dj); mujoco.mj_fwdPosition(mu, du)          # (mj_kinematics alone leaves the cameras stale)
        for n in common:
            p, R = frames[n](); b = mu.body(n).id
            worst_p = max(worst_p, float(np.abs(du.xpos[b] - p).max())); worst_R = max(worst_R, float(np.abs(du.xmat[b].reshape(3, 3) - R).max()))
    ok1 = worst_p < 1e-9 and worst_R < 1e-9
    missing = sorted(set(n for n in frames if mujoco.mj_name2id(mu, mujoco.mjtObj.mjOBJ_BODY, n) < 0) - {"world"})
    rep["kinematics"] = dict(configs=a.configs, frames_compared=len(common), max_pos_err_m=worst_p, max_rot_err=worst_R, ok=ok1, not_in_urdf=missing)
    print(f"[check] 1 kinematics: {len(common)} links/frames x {a.configs} configs, max |dp| {worst_p:.2e} m, max |dR| {worst_R:.2e} -> {'PASS' if ok1 else 'FAIL'}")
    print(f"        not exported (by design): {missing}")

    # ---- 2. inertia -----------------------------------------------------------------------------------------------
    dj.qpos[:] = mj.qpos0; dj.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]; du.qpos[:] = mu.qpos0
    mujoco.mj_kinematics(mj, dj); mujoco.mj_kinematics(mu, du)
    wm = wc = wI = 0.0
    for b in range(1, mj.nbody):
        n = mj.body(b).name; bu = mu.body(n).id
        wm = max(wm, abs(mj.body_mass[b] - mu.body_mass[bu]))
        if mj.body_mass[b] > 0:
            wc = max(wc, float(np.abs(dj.xipos[b] - du.xipos[bu]).max()))
            Ij = dj.ximat[b].reshape(3, 3) @ np.diag(mj.body_inertia[b]) @ dj.ximat[b].reshape(3, 3).T
            Iu = du.ximat[bu].reshape(3, 3) @ np.diag(mu.body_inertia[bu]) @ du.ximat[bu].reshape(3, 3).T
            wI = max(wI, float(np.abs(Ij - Iu).max()))
    tot_j, tot_u = float(mj.body_subtreemass[1]), float(mu.body_subtreemass[mu.body("pelvis").id])
    ok2 = wm < 1e-9 and wc < 1e-9 and wI < 1e-7                       # (inertia: eigen-decomposition round-off ~1e-9)
    rep["inertia"] = dict(total_mass_mjcf=tot_j, total_mass_urdf=tot_u, max_mass_err=wm, max_com_err_m=wc, max_inertia_err=wI, ok=ok2)
    print(f"[check] 2 inertia: total {tot_j:.4f} vs {tot_u:.4f} kg; max |dm| {wm:.1e}, |dcom| {wc:.1e} m, |dI| {wI:.1e} -> {'PASS' if ok2 else 'FAIL'}")

    # ---- 3. cameras: markers through the recorder's image pipelines -------------------------------------------------
    from simple.teleop.holomotion_v14 import wrist_cams as WC
    from simple.teleop.holomotion_v14.hbvcam_views import PinholeView
    import cv2
    raw_imgs = json.loads((ROOT / "data/teleop_holomotion_v14_kitchen_plates_smoke/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole/meta/holomotion_raw.json").read_text())["images"]
    m0 = mujoco.MjModel.from_xml_path(str(MJCF)); d0 = mujoco.MjData(m0); d0.qpos[:7] = [0, 0, 1.0, 1, 0, 0, 0]; mujoco.mj_forward(m0, d0)
    head = [(0.15, 0.08, 0.7), (-0.2, -0.1, 0.9), (0.0, 0.12, 0.5), (0.3, -0.12, 0.8), (-0.35, 0.15, 0.7)]
    wrist = [(0.0, 0.0, 0.4), (0.2, 0.1, 0.3), (-0.3, -0.1, 0.25), (0.25, -0.15, 0.35), (-0.2, 0.15, 0.3), (0.0, -0.2, 0.3),
             (0.5, 0.05, 0.25), (-0.6, -0.1, 0.3), (0.45, -0.1, 0.3)]                   # the last three fall on the side faces (> 45 deg)
    pts = {"hbvcam_left_pinhole": head, "hbvcam_right_pinhole": head, "wrist_left": wrist, "wrist_right": wrist}
    key = {"hbvcam_left_pinhole": "observation.images.ego_view_left", "hbvcam_right_pinhole": "observation.images.ego_view_right",
           "wrist_left": "observation.images.wrist_left", "wrist_right": "observation.images.wrist_right"}
    res = {}; worst_px = 0.0
    for c, plist in pts.items():                                       # one camera at a time: only its own markers in the scene
        cid0 = m0.camera(c).id; p0, R0 = d0.cam_xpos[cid0], d0.cam_xmat[cid0].reshape(3, 3) @ RX_PI
        spec = mujoco.MjSpec.from_file(str(MJCF)); world = []
        for po in plist:
            w = p0 + R0 @ np.asarray(po); world.append(w)
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0], pos=list(w), rgba=[1, 0, 0, 1], contype=0, conaffinity=0)
        m1 = spec.compile(); d1 = mujoco.MjData(m1); d1.qpos[:7] = [0, 0, 1.0, 1, 0, 0, 0]; mujoco.mj_forward(m1, d1)
        for g in range(m1.ngeom):                                      # only the markers are drawn: the camera frames' kinematics
            if m1.geom_bodyid[g] != 0:                                 # are check 1; a marker partly hidden by the robot biases its
                m1.geom_group[g] = 5                                   # blob centre by pixels
        cid = m1.camera(c).id; cam_body = int(m1.cam_bodyid[cid])
        env = SimpleNamespace(mujoco=SimpleNamespace(mjModel=m1, mjData=d1))
        if c.startswith("hbvcam"):
            L, R = PinholeView(env).render_pair()
            img = cv2.resize(L if "left" in c else R, (640, 360), interpolation=cv2.INTER_AREA)
        else:
            wc = WC.WristCams(env, hz=0); img = wc.render()["left" if "left" in c else "right"]; wc.close()
        K = raw_imgs[key[c]]; rgb = img.astype(int)                    # the whole red silhouette: a brightness threshold keeps only the
        red = (rgb[..., 0] > 40) & (rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > 30)   # lit side and moves the centre ~2 px
        n_lab, lab, stats, cent = cv2.connectedComponentsWithStats(red.astype(np.uint8))
        found = [cent[i] for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= 3]
        rows = []
        for k, po in enumerate(plist):
            w_ = world[k]; vec = w_ - d1.cam_xpos[cid]; gid = np.full(1, -1, np.int32)
            mark = next(g for g in range(m1.ngeom) if m1.geom_bodyid[g] == 0 and np.allclose(d1.geom_xpos[g], w_, atol=1e-9))   # world geoms compile first
            mujoco.mj_ray(m1, d1, d1.cam_xpos[cid], vec / np.linalg.norm(vec), np.array([1, 1, 1, 0, 0, 0], np.uint8), 1, cam_body, gid)
            x, y, z = po
            if "fx" in K:
                u, v = K["fx"] * x / z + K["cx"], K["fy"] * y / z + K["cy"]
            else:
                th = np.arctan2(np.hypot(x, y), z); rr = K["f"] * th; ph = np.arctan2(y, x)
                u, v = K["cx"] + rr * np.cos(ph), K["cy"] + rr * np.sin(ph)
            if not (3 <= u <= 636 and 3 <= v <= 356):                  # outside the picture
                rows.append(dict(point_optical_m=list(po), predicted_px=[round(u, 2), round(v, 2)], visible=False)); continue
            if int(gid[0]) != mark:                                    # the robot's own body is in the way: not scored
                rows.append(dict(point_optical_m=list(po), predicted_px=[round(u, 2), round(v, 2)], visible=False)); continue
            near = min(found, key=lambda q: np.hypot(q[0] - u, q[1] - v)) if found else None
            err = float(np.hypot(near[0] - u, near[1] - v)) if near is not None else None
            if err is not None:
                worst_px = max(worst_px, err)
            rows.append(dict(point_optical_m=list(po), predicted_px=[round(u, 2), round(v, 2)], found_px=None if near is None else [round(float(near[0]), 2), round(float(near[1]), 2)], err_px=None if err is None else round(err, 2)))
        res[c] = rows
        cv2.imwrite(str(out / f"markers_{c}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ok4 = all(r["err_px"] is not None and r["err_px"] < 1.0 for rows in res.values() for r in rows if r.get("visible", True)) and \
        all(sum(1 for r in rows if r.get("visible", True)) >= 3 for rows in res.values())
    rep["cameras"] = dict(markers=res, max_err_px=worst_px, ok=ok4)
    for c, rows in res.items():
        print(f"[check] 3 {c}: " + "; ".join(f"pred {r['predicted_px']} found {r['found_px']} err {r['err_px']}" for r in rows if r.get("visible", True))
              + f" ({sum(1 for r in rows if not r.get('visible', True))} hidden by the body, skipped)")
    print(f"[check] 3 cameras: {sum(1 for r in res.values() for x in r if x.get('visible', True))} visible markers, max error {worst_px:.2f} px -> {'PASS' if ok4 else 'FAIL'}")
    # ---- 4. geometry: depth of the visual geometry ------------------------------------------------------------------
    exported_visual = set()
    import xml.etree.ElementTree as ET
    for v in ET.parse(URDF).getroot().iter("visual"):
        if v.get("name"):
            exported_visual.add(v.get("name"))
    def only_visual(m, keep):
        for g in range(m.ngeom):
            if not keep(g):
                m.geom_group[g] = 5
    g1_twins = {}
    for g in range(mj.ngeom):
        if mj.geom_group[g] == 1 and mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            g1_twins.setdefault(int(mj.geom_bodyid[g]), set()).add(int(mj.geom_dataid[g]))
    only_visual(mj, lambda g: mj.geom_group[g] in (1, 2) or (mj.geom_group[g] == 0 and mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                                                               and int(mj.geom_dataid[g]) not in g1_twins.get(int(mj.geom_bodyid[g]), set())))
    only_visual(mu, lambda g: not (mu.geom_contype[g] | mu.geom_conaffinity[g]))
    mu.stat.extent = mj.stat.extent                                    # same near / far planes (both scale with the extent)
    for m_ in (mj, mu):
        m_.vis.global_.offwidth, m_.vis.global_.offheight = 640, 480
    zfar = float(mj.vis.map.zfar * mj.stat.extent)
    opt = mujoco.MjvOption(); opt.sitegroup[:] = 0                     # sites (com_marker sphere, frames) are not geometry
    rj, ru = mujoco.Renderer(mj, 480, 640), mujoco.Renderer(mu, 480, 640)
    rj.enable_depth_rendering(); ru.enable_depth_rendering()
    views = [dict(lookat=[0, 0, 0.3], distance=2.2, azimuth=az, elevation=el) for az, el in ((180, -10), (90, -15), (0, -20), (-90, -15), (135, -50))]
    views.append(dict(lookat=[0.25, 0.25, 0.15], distance=0.6, azimuth=200, elevation=-25))      # the left hand and its wrist camera close up
    worst_d = 0.0; bad_px = 0; n_hit = 0
    import imageio.v2 as imageio
    for i, v in enumerate(views):
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = v["lookat"]
        cam.distance, cam.azimuth, cam.elevation = v["distance"], v["azimuth"], v["elevation"]
        rj.update_scene(dj, cam, scene_option=opt); ru.update_scene(du, cam, scene_option=opt)
        a_, b_ = rj.render(), ru.render()
        hit = (a_ < 0.99 * zfar) | (b_ < 0.99 * zfar)                   # robot pixels in either render
        diff = np.abs(a_ - b_)[hit]; n_hit += int(hit.sum())
        worst_d = max(worst_d, float(np.percentile(diff, 99.9)) if diff.size else 0.0); bad_px += int((diff > 1e-3).sum())
        far = 1.2 * float(np.percentile(a_[hit], 99)) if hit.any() else 1.0
        vis = np.clip(np.concatenate([np.where(a_ < 0.99 * zfar, a_, far) / far, np.where(b_ < 0.99 * zfar, b_, far) / far,
                                      np.where(hit, np.abs(a_ - b_) * 100, 0)], 1), 0, 1)        # left MJCF, middle URDF, right |diff| x 100
        imageio.imwrite(out / f"depth_{i}.png", (vis * 255).astype(np.uint8))
    ok3 = bad_px == 0
    rep["geometry"] = dict(views=len(views), p999_depth_diff_m=worst_d, pixels_over_1mm=bad_px, robot_pixels=int(n_hit), ok=ok3)
    print(f"[check] 4 geometry: depth of the visual geometry from {len(views)} views, {n_hit} robot pixels, differing > 1 mm: {bad_px} -> {'PASS' if ok3 else 'FAIL'}")
    from simple.teleop.holomotion_v14.hbvcam_views import close_renderer
    close_renderer(rj); close_renderer(ru)                              # (plain close() frees GL objects in whatever context is current)

    rep["all_ok"] = bool(ok1 and ok2 and ok3 and ok4)
    (out / "urdf_check.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"[check] all {'PASS' if rep['all_ok'] else 'FAIL'} -> {out / 'urdf_check.json'}")
    os._exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
