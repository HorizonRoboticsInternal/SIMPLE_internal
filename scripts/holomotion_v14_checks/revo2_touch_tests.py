#!/usr/bin/env python
"""Bench tests of the simulated Revo 2 Touch pads (scripts/robots/add_revo2_touch.py + simple.sensors.revo2_touch).
Each right-hand pad is mounted alone, outward normal up, and loaded with known weights and pushes:
  normal     a block of 1 / 5 / 10 / 25 / 30 N rests on the pad -> normal_force1 must read m*g in 0.01 N counts, saturating at 2500
  tangential a 5 N block is pushed sideways with 1 N along +tip, -tip, and the two side directions -> tangential_force1 = 100 counts,
             tangential_direction1 = 0 / 180 / 90 / 270 deg (0 = toward the fingertip, clockwise seen from outside the pad)
  proximity  a plate hovers 20 / 10 / 5 / 2 / 0.5 mm above the pad -> self_proximity1 counts per the page's mapping, 0 beyond 1 cm
  rate       cost of one read() in microseconds; the robot's physics step is 2 ms, the real sensor is polled at 100 Hz
Writes docs/revo2_touch/static_tests.json.   .venv/bin/python scripts/holomotion_v14_checks/revo2_touch_tests.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, mujoco
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from simple.sensors.revo2_touch import Revo2Touch, FINGERS, COUNTS_PER_N, FORCE_MAX_N, PROX_RANGE_M
RD = ROOT / "data/robots/g1_rot0_teleop_revo2"; PADS = json.loads((RD / "touch_pads.json").read_text()); OUT = ROOT / "docs/revo2_touch"; OUT.mkdir(exist_ok=True)
G = 9.81


def quat_to_mat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def mat_to_quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


def pad_model(finger, extra=""):
    """One right pad fixed in the world with its outward normal along +z and the fingertip direction along +x; the site centre at the origin."""
    pad = PADS[f"right_hand_{finger}_touch_link"]; c = np.array(pad["site_centre"]); R_site = quat_to_mat(pad["site_quat"])   # site axes in the pad body frame
    M = np.diag([1.0, -1.0, -1.0])                     # site (x tip, y, z into the finger) -> world (x tip, -y, -z): outward = +z world
    R_wb = M @ R_site.T; q_wb = mat_to_quat(R_wb); p_wb = -R_wb @ c
    half = " ".join(str(v) for v in np.array(pad["box_half_mm"]) / 1000); sq = " ".join(f"{v:.6g}" for v in pad["site_quat"]); sc = " ".join(f"{v:.6g}" for v in c)
    ray_q = mat_to_quat(R_site @ np.diag([1.0, -1.0, -1.0]))    # z out of the pad, x toward the tip
    xml = f'''<mujoco><compiler meshdir="{RD / 'meshes'}" angle="radian"/><option timestep="0.001" gravity="0 0 -{G}"/>
<extension><plugin plugin="mujoco.sensor.touch_grid"/></extension>
<asset><mesh name="pad" file="right_{finger}_touch_link.STL"/></asset>
<worldbody>
  <body name="right_hand_{finger}_touch_link" pos="{' '.join(f'{v:.6g}' for v in p_wb)}" quat="{' '.join(f'{v:.6g}' for v in q_wb)}">
    <geom name="right_hand_{finger}_touch_collision" type="mesh" mesh="pad" friction="0.7 0.005 0.0001" condim="3"/>
    <site name="right_hand_{finger}_touch_site" type="box" pos="{sc}" quat="{sq}" size="{half}" group="4"/>
    <site name="right_hand_{finger}_touch_ray" type="sphere" pos="{' '.join(f'{v:.6g}' for v in pad['ray_pos'])}" quat="{' '.join(f'{v:.6g}' for v in pad['ray_quat'])}" size="0.001" group="4"/>
  </body>
  {extra}
</worldbody>
<sensor><touch name="right_hand_{finger}_touch_total" site="right_hand_{finger}_touch_site"/><rangefinder name="right_hand_{finger}_touch_prox" site="right_hand_{finger}_touch_ray" cutoff="0.03"/>
<plugin name="right_hand_{finger}_touch_grid" plugin="mujoco.sensor.touch_grid" objtype="site" objname="right_hand_{finger}_touch_site"><config key="size" value="3 3"/><config key="fov" value="90 90"/><config key="gamma" value="0"/><config key="nchannel" value="3"/></plugin></sensor></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


class OneFinger(Revo2Touch):
    def __init__(self, m, d, finger):
        self.m, self.d, self.side = m, d, "right"; self._f6 = np.zeros(6); self._own_prefix = "right_hand_"
        base = f"right_hand_{finger}_touch"; from simple.sensors.revo2_touch import _sensor
        geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{base}_collision"); site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{base}_site")
        self.fingers = [dict(name=finger, geom=geom, body=int(m.geom_bodyid[geom]), site=site, total=_sensor(m, f"{base}_total"), prox=_sensor(m, f"{base}_prox"), grid=_sensor(m, f"{base}_grid"),
                             ray=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{base}_ray"),                 # the reader casts its own ray from this site (2026-10-26)
                             radius=PADS[f"right_hand_{finger}_touch_link"]["pad_radius_mm"] / 1000, inset=PADS[f"right_hand_{finger}_touch_link"]["ray_inset_m"])]


def pad_top_z(m):
    """Highest point of the pad collision mesh in the world (where a block rests)."""
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, [n for n in [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)] if n and "touch_collision" in n][0])
    mid = m.geom_dataid[g]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    W = (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]; return float(W[:, 2].max())


def settle(m, d, n):
    for _ in range(n): mujoco.mj_step(m, d)


def run_finger(finger):
    res = dict(finger=finger, pad=PADS[f"right_hand_{finger}_touch_link"], normal=[], tangential=[], proximity=[])
    top = pad_top_z(pad_model(finger))
    # --- normal force: a 40 x 40 x 10 mm block of mass F/g resting on the pad
    for F in [1.0, 5.0, 10.0, 25.0, 30.0]:
        m = pad_model(finger, f'<body name="blk" pos="0 0 {top + 0.0055:.5f}"><joint type="slide" axis="0 0 1"/><geom type="box" size="0.02 0.02 0.005" mass="{F / G:.6f}" friction="0.7 0.005 0.0001" condim="3"/></body>')
        d = mujoco.MjData(m); settle(m, d, 2000); s = OneFinger(m, d, finger); p = s.physical()[0]; sdk = s.read()[0]
        res["normal"].append(dict(applied_N=F, measured_N=round(p["normal_N"], 3), touch_sensor_N=round(p["touch_sensor_N"], 3), counts=sdk["normal_force1"],
                                  expected_counts=int(min(round(F * COUNTS_PER_N), FORCE_MAX_N * COUNTS_PER_N)), contacts=p["contacts"], taxels_N=s.taxels()[0, 0].round(3).tolist()))
    # --- tangential force and direction: 5 N block, 1 N sideways (friction limit 0.7 * 5 N = 3.5 N holds it)
    for label, fvec, expect in [("toward the fingertip", (1, 0), 0), ("away from the fingertip", (-1, 0), 180), ("clockwise side", (0, -1), 90), ("anticlockwise side", (0, 1), 270)]:
        ax = "1 0 0" if fvec[0] else "0 1 0"
        m = pad_model(finger, f'<body name="blk" pos="0 0 {top + 0.0055:.5f}"><joint type="slide" axis="0 0 1"/><joint type="slide" axis="{ax}"/><geom type="box" size="0.02 0.02 0.005" mass="{5 / G:.6f}" friction="0.7 0.005 0.0001" condim="3"/></body>')
        d = mujoco.MjData(m); settle(m, d, 1500); d.xfrc_applied[m.body("blk").id, :2] = fvec; settle(m, d, 1500); s = OneFinger(m, d, finger); p = s.physical()[0]; sdk = s.read()[0]
        res["tangential"].append(dict(push=label, applied_N=1.0, measured_N=round(p["tangential_N"], 3), counts=sdk["tangential_force1"], direction_deg=None if p["direction_deg"] is None else round(p["direction_deg"], 1),
                                      expected_direction_deg=expect, direction_counts=sdk["tangential_direction1"], normal_N=round(p["normal_N"], 3), block_speed_mm_s=round(1000 * float(np.linalg.norm(d.qvel[:3])), 2)))
    # --- proximity: a plate held at a distance above the pad (kinematic, gravity off)
    for mm in [20.0, 10.0, 5.0, 2.0, 0.5, 0.0]:
        rs = PADS[f"right_hand_{finger}_touch_link"]["surface_r_mm"] / 1000        # the pad surface point on the normal (the site centre is the origin, normal = +z)
        m = pad_model(finger, f'<body name="plate" pos="0 0 {rs + mm / 1000 + 0.002:.5f}"><geom type="box" size="0.03 0.03 0.002"/></body>')
        m.opt.gravity[:] = 0; d = mujoco.MjData(m); mujoco.mj_forward(m, d); s = OneFinger(m, d, finger); p = s.physical()[0]; sdk = s.read()[0]
        rf = float(d.sensordata[s.fingers[0]["prox"][0]])
        res["proximity"].append(dict(gap_mm=mm, rangefinder_m=round(rf, 4), distance_mm=None if p["proximity_m"] is None else round(1000 * p["proximity_m"], 2), counts=sdk["self_proximity1"]))
    # --- read cost
    m = pad_model(finger, f'<body name="blk" pos="0 0 {top + 0.0055:.5f}"><joint type="slide" axis="0 0 1"/><geom type="box" size="0.02 0.02 0.005" mass="{5 / G:.6f}"/></body>'); d = mujoco.MjData(m); settle(m, d, 1500); s = OneFinger(m, d, finger)
    t0 = time.perf_counter(); [s.read() for _ in range(500)]; res["read_us"] = round(1e6 * (time.perf_counter() - t0) / 500, 1)
    return res


if __name__ == "__main__":
    results = [run_finger(f) for f in FINGERS]
    ok = True
    for r in results:
        print(f"\n== {r['finger']} pad: outer area {r['pad']['outer_area_mm2']:.0f} mm2, radius {r['pad']['pad_radius_mm']} mm, read() {r['read_us']} us")
        for n in r["normal"]:
            good = abs(n["measured_N"] - n["applied_N"]) <= 0.02 * n["applied_N"] + 0.02 and n["counts"] == n["expected_counts"]; ok &= good
            print(f"   normal {n['applied_N']:5.1f} N -> {n['measured_N']:6.3f} N, {n['counts']:4d} counts (expect {n['expected_counts']}) {'ok' if good else 'MISMATCH'}")
        for t in r["tangential"]:
            good = abs(t["measured_N"] - 1.0) < 0.05 and t["direction_deg"] is not None and min(abs(t["direction_deg"] - t["expected_direction_deg"]), 360 - abs(t["direction_deg"] - t["expected_direction_deg"])) < 3; ok &= good
            print(f"   push 1 N {t['push']:24s} -> {t['measured_N']:.3f} N, {t['counts']} counts, direction {t['direction_deg']} deg (expect {t['expected_direction_deg']}) {'ok' if good else 'MISMATCH'}")
        for p in r["proximity"]:
            print(f"   plate {p['gap_mm']:4.1f} mm above the pad -> distance {p['distance_mm']} mm, {p['counts']} counts")
    (OUT / "static_tests.json").write_text(json.dumps(dict(all_ok=bool(ok), fingers=results), indent=1)); print(f"\nall_ok={ok} -> {OUT / 'static_tests.json'}")
