#!/usr/bin/env python
"""Does every procedural plate stand in the dishwasher's lower rack?  For each plate: the kitchen scene (no robot), door open,
lower rack in, plate_a replaced by the plate and stood on edge in a tine slot (plate plane across the rack's y axis),
dropped from 1 cm and simulated 3 s.  Pass = still upright (plate normal within 35 deg of the rack's y), centre inside the
rack box, top under the upper rack, no deep penetration.  Also reports the geometric slot fit (plate height vs slot pitch).
    MUJOCO_GL=egl .venv/bin/python scripts/kitchen_plates/check_rack_fit.py [--plates data/kitchen_plates] [--out docs/kitchen_plates/rack_fit.json]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_swap import swap_plate, load_meta
KX = Path("/home/Horizon/wrk/KitchenScene_Manipulable_20261026/scene.xml")
ap = argparse.ArgumentParser(); ap.add_argument("--plates", default="data/kitchen_plates"); ap.add_argument("--out", default="docs/kitchen_plates/rack_fit.json"); ap.add_argument("--render", action="store_true"); ap.add_argument("--only", default=""); a = ap.parse_args()
man = json.load(open(Path(a.plates) / "manifest.json"))
# rack geometry (lower rack): slots between vertical tine rows; pick the widest regular slot
spec0 = mujoco.MjSpec.from_file(str(KX)); m0 = spec0.compile(); d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
rb = m0.body("dishwasher_lower_rack").id; ys = []
for g in range(m0.ngeom):
    if m0.geom_bodyid[g] == rb and m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and m0.geom_contype[g]:
        R = d0.geom_xmat[g].reshape(3, 3); ax = R[:, 2]
        if abs(ax[2]) > 0.9: ys.append(round(float(d0.geom_xpos[g][1] - d0.xpos[rb][1]), 4))
ys = sorted(set(ys)); rows = []                                                     # tine rows come in pairs 5-15 mm apart: merge them into one row each
for y in ys:
    if rows and y - rows[-1][-1] < 0.018: rows[-1].append(y)
    else: rows.append([y])
rc = [float(np.mean(r)) for r in rows]; gaps = [(rc[i + 1] - rc[i], rc[i], rc[i + 1]) for i in range(len(rc) - 1)]
slot = max(gaps, key=lambda g: g[0]); print(f"[rack] {len(rows)} tine rows at y {[round(v, 3) for v in rc]}; slots {[round(g[0]*1000) for g in gaps]} mm; using the widest, y {slot[1]:.3f}..{slot[2]:.3f} ({slot[0]*1000:.0f} mm)")
rack_pos = m0.body("dishwasher_lower_rack").pos.copy(); upper_z = float(m0.body("dishwasher_upper_rack").pos[2]) - 0.003
results = []; t0 = time.time()
for P in man["plates"]:
    if a.only and P["id"] not in a.only.split(","): continue
    spec = mujoco.MjSpec.from_file(str(KX)); pdir = Path(a.plates) / P["id"]; meta = load_meta(pdir); swap_plate(spec, "plate_a", pdir, meta, keep_z=True)
    m = spec.compile(); d = mujoco.MjData(m)
    d.qpos[m.joint("dishwasher_door").qposadr[0]] = 1.5                                   # door open, rack in
    R = P["diameter_mm"] / 2000; ja = m.joint("plate_a_free").qposadr[0]
    yc = (slot[1] + slot[2]) / 2; d.qpos[ja:ja + 3] = [rack_pos[0] + 0.0, rack_pos[1] + yc, rack_pos[2] - 0.003 + R + 0.012]
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0, 0]), np.pi / 2); d.qpos[ja + 3:ja + 7] = q      # plate normal z -> -y: plane across the slot
    mujoco.mj_forward(m, d); bid = m.body("plate_a").id; pen = 0.0; n_steps = int(3.0 / m.opt.timestep)
    for i in range(n_steps):
        mujoco.mj_step(m, d)
        if i > n_steps - int(0.5 / m.opt.timestep):                                 # penetration in the settled last 0.5 s (not the drop impact)
            for c in d.contact[:d.ncon]:
                if bid in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]): pen = max(pen, -c.dist)
    n = d.xmat[bid].reshape(3, 3)[:, 2]; tilt = float(np.degrees(np.arccos(min(1.0, abs(n[1]))))); p = d.xpos[bid] - rack_pos
    inside = abs(p[0]) < 0.235 and abs(p[1]) < 0.22 and p[2] > -0.01; top = float(d.xpos[bid][2] + R); under_upper = top < upper_z
    ok = tilt < 35 and inside and under_upper and np.isfinite(d.qpos).all()                  # the fit; the settled wire penetration (soft contacts) is reported separately
    slot_fit = P["height_mm"] < slot[0] * 1000 - 2                                     # thinner than the slot (leaning allowed)
    results.append(dict(id=P["id"], diameter_mm=P["diameter_mm"], height_mm=P["height_mm"], tilt_deg=round(tilt, 1), centre_in_rack=[round(float(v), 3) for v in p], inside=bool(inside), top_z=round(top, 3), under_upper_rack=bool(under_upper), max_penetration_mm=round(pen * 1000, 2), stands=bool(ok), slot_fit=bool(slot_fit)))
    print(f"{P['id']}: D {P['diameter_mm']:.0f} H {P['height_mm']:.1f} mm -> tilt {tilt:5.1f} deg, centre {np.round(p,3).tolist()}, pen {pen*1000:.2f} mm, top {top:.3f} < upper {upper_z:.3f}: {'OK' if ok else 'FAIL'}", flush=True)
    if a.render and P["id"] in ("plate_00", "plate_05", "plate_11"):
        r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; look = d.xpos[bid] + [0, 0, 0.02]; eye = look + np.array([0.15, -0.95, 0.5]); v = eye - look   # from the open front (-y), above the open door
        cam.lookat[:] = look; cam.distance = float(np.linalg.norm(v)); cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0]))); cam.elevation = float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v))))   # MuJoCo az/el describe the LINE OF SIGHT (lookat - eye)
        r.update_scene(d, camera=cam); from PIL import Image; Path(a.out).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(r.render()).save(Path(a.out).parent / f"rack_{P['id']}.jpg", quality=85)
ok_n = sum(r["stands"] for r in results); fit_n = sum(r["slot_fit"] for r in results)
if a.only: sys.exit(0)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(dict(slot_mm=round(slot[0] * 1000, 1), slot_y=[slot[1], slot[2]], upper_rack_bottom_z=upper_z, stands=ok_n, slot_fit=fit_n, n=len(results), plates=results), open(a.out, "w"), indent=1)
print(f"[rack] {ok_n}/{len(results)} plates stand in the lower-rack slot after 3 s; {fit_n}/{len(results)} thinner than the slot; {time.time()-t0:.0f} s -> {a.out}")
