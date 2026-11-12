#!/usr/bin/env python
"""Can the Revo 2 pinch hold each procedural plate?  A floating right hand (the robot's hand subtree on a mocap carrier, its
own motors with the teleop's hand PD gains) starts 30 % closed (about 75 mm open) around the plate's rim lip (thumb under, index on top, 9 mm in
from the edge), closes to the pinch pose (CLOSE2_Q), the support under the plate is withdrawn, then the hand lifts 12 cm.
Pass = the plate rises with the hand (slip < 2 cm) and is still held at the end.
    MUJOCO_GL=egl .venv/bin/python scripts/kitchen_plates/check_pinch.py [--plates data/kitchen_plates] [--out docs/kitchen_plates/pinch.json] [--render]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_swap import swap_plate, load_meta
from simple.robots.g1_sonic_revo2 import OPEN_Q, CLOSE2_Q, RIGHT_REVO2_JOINTS, G1SonicRevo2
HAND_KP = np.asarray(G1SonicRevo2.HAND_KP, dtype=float)
ap = argparse.ArgumentParser(); ap.add_argument("--plates", default="data/kitchen_plates"); ap.add_argument("--out", default="docs/kitchen_plates/pinch.json"); ap.add_argument("--render", action="store_true"); ap.add_argument("--ids", default="", help="only these plate indices, e.g. 5,10"); ap.add_argument("--render-ids", default="plate_00,plate_07,plate_21"); a = ap.parse_args()
RENDER_IDS = set(a.render_ids.split(","))
man = json.load(open(Path(a.plates) / "manifest.json")); KD = 0.05; DT = 0.002


def hand_model(plate_dir=None, meta=None):
    robot = mujoco.MjSpec.from_file(str(ROOT / "data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2.xml"))
    spec = mujoco.MjSpec(); spec.option.timestep = DT; spec.option.gravity = [0, 0, -9.81]; spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST; spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; spec.option.impratio = 10; spec.option.noslip_iterations = 2
    spec.visual.global_.offwidth = 640; spec.visual.global_.offheight = 480
    spec.worldbody.add_light(pos=[0.3, 0.3, 1.0], dir=[-0.3, -0.3, -1]); spec.worldbody.add_light(pos=[-0.5, 0.2, 0.8], dir=[0.5, -0.2, -1])
    # the hand hangs on a free root body welded to a mocap carrier: attached straight under a mocap body the hand base would
    # be welded to the world and MuJoCo would stop filtering its contacts with the finger links (13 mm self-penetration)
    car = spec.worldbody.add_body(name="carrier", mocap=True)
    root = spec.worldbody.add_body(name="hand_root"); root.add_freejoint(); root.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0], mass=0.01, contype=0, conaffinity=0, group=4)
    fr = root.add_frame(); fr.attach_body(robot.body("right_hand_base_link"), "h_", "")
    eq = spec.add_equality(); eq.type = mujoco.mjtEq.mjEQ_WELD; eq.objtype = mujoco.mjtObj.mjOBJ_BODY; eq.name1 = "carrier"; eq.name2 = "hand_root"; eq.solref = [0.001, 1.0]; eq.solimp = [0.99, 0.999, 0.0005, 0.5, 2.0]
    for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY) if hasattr(spec.worldbody, "find_all") else spec.bodies:
        if b.name.startswith("h_") or b.name == "hand_root": b.gravcomp = 1.0                 # the arm carries the hand: no sag on the weld
    if plate_dir is not None:
        pb = spec.worldbody.add_body(name="plate", pos=[0, 0, 0.3]); pb.add_freejoint(); pb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0, group=4)
        swap_plate(spec, "plate", plate_dir, meta, keep_z=True)
        sup = spec.worldbody.add_body(name="support", mocap=True, pos=[0, 0, 0.3 - 0.01]); sup.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.01], rgba=[0.6, 0.6, 0.6, 0.5], condim=4, friction=[0.9, 0.005, 0.0001])
    return spec.compile()


def pads(m, d):
    th = d.site_xpos[m.site("h_right_hand_thumb_touch_site").id].copy(); ix = d.site_xpos[m.site("h_right_hand_index_touch_site").id].copy()
    nth = -d.site_xmat[m.site("h_right_hand_thumb_touch_site").id].reshape(3, 3)[:, 2]; nix = -d.site_xmat[m.site("h_right_hand_index_touch_site").id].reshape(3, 3)[:, 2]
    import json as _j; P = _j.load(open(ROOT / "data/robots/g1_rot0_teleop_revo2/touch_pads.json"))
    return th + nth * P["right_hand_thumb_touch_link"]["pad_radius_mm"] / 1000, ix + nix * P["right_hand_index_touch_link"]["pad_radius_mm"] / 1000


def set_hand(m, d, q):
    for j, v in zip(RIGHT_REVO2_JOINTS, q): d.qpos[m.joint("h_" + j).qposadr[0]] = v


# the pinch geometry in the hand base frame (carrier at identity): pad surfaces at the start pose and at the pinch pose
m0 = hand_model(); d0 = mujoco.MjData(m0); q_start = OPEN_Q + 0.30 * (CLOSE2_Q - OPEN_Q)                 # ~75 mm open: the thumb starts well below the lip, clear of the wall
set_hand(m0, d0, q_start); mujoco.mj_forward(m0, d0); th_s, ix_s = pads(m0, d0)                     # root free joint at identity: hand base frame = world
set_hand(m0, d0, CLOSE2_Q); mujoco.mj_forward(m0, d0); th_c, ix_c = pads(m0, d0)
mid_h = 0.5 * (th_c + ix_c); axis_h = (th_s - ix_s) / np.linalg.norm(th_s - ix_s)                       # the pads MEET at the pinch pose (the rim must be there); the opening axis from the start pose
wrist_h = np.zeros(3); fwd_h = mid_h - wrist_h; fwd_h -= (fwd_h @ axis_h) * axis_h; fwd_h /= np.linalg.norm(fwd_h)   # wrist -> pinch point, perpendicular to the pinch axis
print(f"[pinch] start pose pad gap {np.linalg.norm(th_s-ix_s)*1000:.1f} mm, pinch pose gap {np.linalg.norm(th_c-ix_c)*1000:.1f} mm; pinch point {np.round(mid_h*1000,1).tolist()} mm from the hand base")
# world targets: pinch axis -> -z (thumb below the lip), forward -> -x (reaching inward over the +x edge)
A = np.stack([axis_h, fwd_h, np.cross(axis_h, fwd_h)], 1); B = np.stack([[0, 0, -1.0], [-1.0, 0, 0], np.cross([0, 0, -1.0], [-1.0, 0, 0])], 1); R_wh = B @ A.T
q_wh = np.empty(4); mujoco.mju_mat2Quat(q_wh, R_wh.reshape(9))
acts = {m0.joint(m0.actuator_trnid[i][0]).name: i for i in range(m0.nu)}
results = []; t0 = time.time()
for pi, P in enumerate(man["plates"]):
    if a.ids and pi not in [int(v) for v in a.ids.split(",")]: continue
    pdir = Path(a.plates) / P["id"]; meta = load_meta(pdir); m = hand_model(pdir, meta); d = mujoco.MjData(m)
    acts = {m.joint(m.actuator_trnid[i][0]).name: i for i in range(m.nu)}; qadr = [m.joint("h_" + j).qposadr[0] for j in RIGHT_REVO2_JOINTS]; vadr = [m.joint("h_" + j).dofadr[0] for j in RIGHT_REVO2_JOINTS]; aidx = [acts["h_" + j] for j in RIGHT_REVO2_JOINTS]
    R_pl = meta["extents_mm"][0] / 2000; z0 = 0.3; lip_top = z0 + meta["H"]; t_rim = meta["t_rim"]        # the +x half-extent (ovals are longer along y, squircles along the diagonal)
    G = np.array([R_pl - 0.009, 0.0, lip_top - t_rim / 2])                                                # grasp point on the +x rim lip, 9 mm in from the edge
    set_hand(m, d, q_start); d.mocap_pos[0] = G - R_wh @ mid_h; d.mocap_quat[0] = q_wh
    jr = m.joint("hand_root").qposadr[0] if any(m.joint(i).name == "hand_root" for i in range(m.njnt)) else [m.jnt_qposadr[i] for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[i]).name == "hand_root"][0]
    d.qpos[jr:jr + 3] = d.mocap_pos[0]; d.qpos[jr + 3:jr + 7] = q_wh; mujoco.mj_forward(m, d)
    pen0 = [(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, round(c.dist * 1000, 1)) for c in d.contact[:d.ncon] if c.dist < -0.002]
    if pen0: print(f"   initial penetrations: {pen0[:4]}")
    jp = m.joint("plate_free").qposadr[0] if "plate_free" in [m.joint(i).name for i in range(m.njnt)] else [m.jnt_qposadr[i] for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
    pb = m.body("plate").id; sup = m.body("support").mocapid[0]; sup0 = d.mocap_pos[sup].copy(); car = m.body("carrier").mocapid[0]; car0 = d.mocap_pos[car].copy()
    def control(q_des):
        q = d.qpos[qadr]; dq = d.qvel[vadr]; tau = HAND_KP * (q_des - q) - KD * dq
        for k, i in enumerate(aidx): d.ctrl[i] = float(np.clip(tau[k], *m.actuator_ctrlrange[i]))
    T = dict(close=1.4, withdraw=0.5, hold=0.8, lift=1.0, hold2=1.0); t = 0.0; phase_log = {}; frames = []
    plate_z_before_lift = None; hand_z_before_lift = None
    while t < sum(T.values()):
        if t < T["close"]: q_des = q_start + (CLOSE2_Q - q_start) * min(1.0, t / 1.0)
        else: q_des = CLOSE2_Q
        t1 = T["close"]; t2 = t1 + T["withdraw"]; t3 = t2 + T["hold"]; t4 = t3 + T["lift"]
        if t1 <= t < t2: d.mocap_pos[sup] = sup0 + np.array([0, 0, -0.15 * (t - t1) / T["withdraw"]])
        if t3 <= t < t4:
            if plate_z_before_lift is None: plate_z_before_lift = float(d.xpos[pb][2]); hand_z_before_lift = float(d.mocap_pos[car][2])
            d.mocap_pos[car] = car0 + np.array([0, 0, 0.12 * (t - t3) / T["lift"]])
        control(q_des); mujoco.mj_step(m, d); t += DT
        if a.render and P["id"] in RENDER_IDS and int(t / DT) % 150 == 0:
            frames.append(d.xpos[pb].copy())
    if plate_z_before_lift is None: plate_z_before_lift = float(d.xpos[pb][2]); hand_z_before_lift = float(d.mocap_pos[car][2])
    rise = float(d.xpos[pb][2]) - plate_z_before_lift; hand_rise = float(d.mocap_pos[car][2]) - hand_z_before_lift; slip = hand_rise - rise
    held = any(pb in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and ("thumb" in m.body(m.geom_bodyid[c.geom1]).name + m.body(m.geom_bodyid[c.geom2]).name or "index" in m.body(m.geom_bodyid[c.geom1]).name + m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact[:d.ncon])
    q_end = d.qpos[qadr]; gap = float(np.linalg.norm(np.subtract(*pads(m, d)))); tilt = float(np.degrees(np.arccos(min(1.0, abs(d.xmat[pb].reshape(3, 3)[2, 2])))))
    ok = rise > 0.08 and slip < 0.02 and held and np.isfinite(d.qpos).all()
    results.append(dict(id=P["id"], mass_g=round(P["mass_kg"] * 1000), rim_mm=P["rim_thickness_mm"], rise_m=round(rise, 4), slip_m=round(slip, 4), held_at_end=bool(held), pad_gap_mm=round(gap * 1000, 1), plate_tilt_deg=round(tilt, 1), ok=bool(ok)))
    print(f"{P['id']}: {P['mass_kg']*1000:3.0f} g rim {P['rim_thickness_mm']:.1f} mm -> rise {rise*100:5.1f} cm (hand {hand_rise*100:.0f}), slip {slip*100:4.1f} cm, held {held}, gap {gap*1000:4.1f} mm, tilt {tilt:4.1f} deg: {'OK' if ok else 'FAIL'}", flush=True)
    if a.render and P["id"] in RENDER_IDS:
        from PIL import Image; r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = d.xpos[pb]; cam.distance = 0.45; cam.azimuth = 140; cam.elevation = -20
        opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = 1; r.update_scene(d, camera=cam, scene_option=opt); Path(a.out).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(r.render()).save(Path(a.out).parent / f"pinch_{P['id']}.jpg", quality=85)
n_ok = sum(r["ok"] for r in results); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(n=len(results), ok=n_ok, method="floating right hand (gravity-compensated, welded to a mocap carrier), teleop hand PD (HAND_KP, kd 0.05) within the motors' torque limits, start 30 % closed, rim lip pinched 9 mm in from the +x edge, support withdrawn, 12 cm lift", plates=results), open(a.out, "w"), indent=1)
print(f"[pinch] {n_ok}/{len(results)} plates held through a 12 cm lift; {time.time()-t0:.0f} s -> {a.out}")
