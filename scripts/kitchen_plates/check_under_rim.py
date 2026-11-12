#!/usr/bin/env python
"""Can the Revo 2 fingertips get UNDER each procedural plate's rim while the plate lies flat on the counter (2026-11-04, user:
"make sure the brainco finger can reach underneath it so i can lift it up horizontally")?

The floating right hand of check_pinch.py (the robot's hand subtree on a mocap carrier, the teleop's hand PD gains) with the
plate on a counter slab.  Two slides per plate, the link flat on the counter (its back 1 mm above it), pad up, tip 10 mm
outside the +x rim edge, pushed lip + 12 mm inward over 1 s:
  fingers  the four fingers straight (index 15 mm thick, the one that goes under in a pinch from above / a two-hand carry)
  thumb    the thumb straight (19 mm thick: the link under the lip in the operators' rim pinch, check_pinch.py)
Reported: how far the tip got under the rim edge and how far the plate was pushed away (a link that does not fit under the
lip shoves the plate instead of entering; one that fits drags it 1-2 cm while sliding in).  Pass = both tips >= UNDER_MM under
the rim; `shoved` flags a push of more than half the slide.
(The lift itself is check_pinch.py: 50/50 plates held through 12 cm once the lip is between the pads.)
    MUJOCO_GL=egl .venv/bin/python scripts/kitchen_plates/check_under_rim.py [--plates data/kitchen_plates] [--out docs/kitchen_plates/under_rim.json] [--ids 0,16,43] [--render]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, mujoco
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_swap import swap_plate, load_meta
from simple.robots.g1_sonic_revo2 import OPEN_Q, RIGHT_REVO2_JOINTS, G1SonicRevo2
HAND_KP = np.asarray(G1SonicRevo2.HAND_KP, dtype=float)
ap = argparse.ArgumentParser(); ap.add_argument("--plates", default="data/kitchen_plates"); ap.add_argument("--out", default="docs/kitchen_plates/under_rim.json")
ap.add_argument("--ids", default=""); ap.add_argument("--render", action="store_true"); ap.add_argument("--robot", default="data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_v3.xml")
ap.add_argument("--under-mm", type=float, default=15.0, help="pass: the tip at least this far under the rim edge after the slide")
a = ap.parse_args()
man = json.load(open(Path(a.plates) / "manifest.json")); KD = 0.05; DT = 0.002; Z_TOP = 0.30; UNDER_MM = a.under_mm
MODES = {
    "fingers": dict(links=[f"h_right_hand_{f}_distal_link" for f in ("index", "middle", "ring", "pinky")], site="h_right_hand_index_touch_site", prox="h_right_hand_index_proximal_link", pose="flat"),
    "thumb": dict(links=["h_right_hand_thumb_distal_link"], site="h_right_hand_thumb_touch_site", prox="h_right_hand_thumb_proximal_link", pose="thumb"),
}
ROLLS = tuple(float(v) for v in (0, -20, 20, -40, 40, -60, 60, -80, 80, -100, 100, -120, 120, -140, 140, -160, 160, 180))   # the hand swung about the sliding link's axis
PITCHES = (0.0, 10.0, 20.0, 30.0)                                                 # and its base end raised (the tip stays down) until the rest of it clears the counter


def hand_model(plate_dir=None, meta=None):
    """The floating right hand of check_pinch.py (free root welded to a mocap carrier, gravity-compensated) over a counter slab."""
    robot = mujoco.MjSpec.from_file(str(ROOT / a.robot))
    spec = mujoco.MjSpec(); spec.option.timestep = DT; spec.option.gravity = [0, 0, -9.81]; spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; spec.option.impratio = 10; spec.option.noslip_iterations = 2
    spec.visual.global_.offwidth = 640; spec.visual.global_.offheight = 480
    spec.worldbody.add_light(pos=[0.3, 0.3, 1.0], dir=[-0.3, -0.3, -1]); spec.worldbody.add_light(pos=[-0.5, 0.2, 0.8], dir=[0.5, -0.2, -1])
    car = spec.worldbody.add_body(name="carrier", mocap=True)
    root = spec.worldbody.add_body(name="hand_root"); root.add_freejoint(name="hand_root"); root.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0], mass=0.01, contype=0, conaffinity=0, group=4)
    fr = root.add_frame(); fr.attach_body(robot.body("right_hand_base_link"), "h_", "")
    eq = spec.add_equality(); eq.type = mujoco.mjtEq.mjEQ_WELD; eq.objtype = mujoco.mjtObj.mjOBJ_BODY; eq.name1 = "carrier"; eq.name2 = "hand_root"; eq.solref = [0.001, 1.0]; eq.solimp = [0.99, 0.999, 0.0005, 0.5, 2.0]
    for b in spec.bodies:
        if b.name.startswith("h_") or b.name == "hand_root": b.gravcomp = 1.0
    if plate_dir is not None:
        spec.worldbody.add_geom(name="counter", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.02], pos=[0, 0, Z_TOP - 0.02], rgba=[0.75, 0.72, 0.68, 1], condim=4, friction=[0.7, 0.005, 0.0001])
        pb = spec.worldbody.add_body(name="plate", pos=[0, 0, Z_TOP]); pb.add_freejoint(name="plate_free"); pb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0, group=4)
        swap_plate(spec, "plate", plate_dir, meta, keep_z=True)
    return spec.compile()


def set_hand(m, d, q):
    for j, v in zip(RIGHT_REVO2_JOINTS, q): d.qpos[m.joint("h_" + j).qposadr[0]] = v


def mesh_world_verts(m, d, g):
    mid = m.geom_dataid[g]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def link_points(m, d, links):
    pts = []
    for name in links:
        b = m.body(name).id
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                pts.append(mesh_world_verts(m, d, g))
    return np.concatenate(pts)


def other_penetrations(m, d, counter_g, links, thr=0.0005):
    """Hand parts other than the sliding links pressed into the counter (the placement must not rest the palm on it)."""
    out = []
    for c in d.contact[:d.ncon]:
        if c.dist >= -thr or counter_g not in (int(c.geom1), int(c.geom2)): continue
        b = int(m.geom_bodyid[c.geom1 if int(c.geom2) == counter_g else c.geom2]); n = m.body(b).name
        if n.startswith("h_") and n not in links: out.append((n, round(-c.dist * 1000, 1)))
    return out


def rot_x(deg):
    t = np.radians(deg); return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def rot_y(deg):
    t = np.radians(deg); return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


POSES = {"flat": OPEN_Q.copy(), "thumb": OPEN_Q.copy()}
POSES["flat"][3:] = 0.0                                                     # four fingers straight (thumb as the open pose has it)
POSES["thumb"][0:3] = 0.0; POSES["thumb"][3:] = 0.0                         # thumb straight too
# hand-frame geometry per mode (hand root at identity = hand base frame): link direction and pad normal -> fingers -x, pad up
m0 = hand_model(); d0 = mujoco.MjData(m0); FRAMES = {}
for mode, M in MODES.items():
    set_hand(m0, d0, POSES[M["pose"]]); mujoco.mj_forward(m0, d0); sid = m0.site(M["site"]).id
    fwd = d0.site_xpos[sid] - d0.xpos[m0.body(M["prox"]).id]; fwd /= np.linalg.norm(fwd)
    pad = -d0.site_xmat[sid].reshape(3, 3)[:, 2]; pad -= (pad @ fwd) * fwd; pad /= np.linalg.norm(pad)
    A = np.stack([fwd, pad, np.cross(fwd, pad)], 1); wx, wz = np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]); B = np.stack([wx, wz, np.cross(wx, wz)], 1)
    FRAMES[mode] = B @ A.T
    print(f"[under_rim] {mode}: link direction {np.round(fwd, 3).tolist()}, pad normal {np.round(pad, 3).tolist()} (hand frame) -> link along -x, pad up")

ids = [int(v) for v in a.ids.split(",")] if a.ids else list(range(len(man["plates"])))
results = []; t0 = time.time()
for pi in ids:
    P = man["plates"][pi]; pdir = Path(a.plates) / P["id"]; meta = load_meta(pdir)
    R_pl = meta["extents_mm"][0] / 2000; lip = meta["w"]
    res = dict(id=P["id"], diameter_mm=P["diameter_mm"], clearance_mm=P["rim_underside_clearance_mm"], lip_mm=P["rim_lip_mm"], mass_g=round(P["mass_kg"] * 1000))
    for mode, M in MODES.items():
        m = hand_model(pdir, meta); d = mujoco.MjData(m)
        acts = {m.joint(m.actuator_trnid[i][0]).name: i for i in range(m.nu)}; qadr = [m.joint("h_" + j).qposadr[0] for j in RIGHT_REVO2_JOINTS]
        vadr = [m.joint("h_" + j).dofadr[0] for j in RIGHT_REVO2_JOINTS]; aidx = [acts["h_" + j] for j in RIGHT_REVO2_JOINTS]
        jr = m.joint("hand_root").qposadr[0]; jp = m.joint("plate_free").qposadr[0]; pb = m.body("plate").id; car = m.body("carrier").mocapid[0]; counter_g = m.geom("counter").id
        Q = POSES[M["pose"]]

        def control(q_des):
            q = d.qpos[qadr]; dq = d.qvel[vadr]; tau = HAND_KP * (q_des - q) - KD * dq
            for k, i in enumerate(aidx): d.ctrl[i] = float(np.clip(tau[k], *m.actuator_ctrlrange[i]))

        def place(pos, quat):
            d.mocap_pos[car] = pos; d.mocap_quat[car] = quat; d.qpos[jr:jr + 3] = pos; d.qpos[jr + 3:jr + 7] = quat; d.qvel[:] = 0; mujoco.mj_forward(m, d)

        chosen = None
        for pitch in PITCHES:                                                # raise the base end, swing the hand about the sliding link, until nothing else rests on the counter
            for roll in ROLLS:
                Rw = rot_y(-pitch) @ rot_x(roll) @ FRAMES[mode]; qw = np.empty(4); mujoco.mju_mat2Quat(qw, Rw.reshape(9))   # link along -x: pitching about -y lifts its +x (base) end
                set_hand(m, d, Q); place([0.3, 0.0, Z_TOP + 0.05], qw)
                for _ in range(3):
                    F = link_points(m, d, M["links"])
                    shift = np.array([(R_pl + 0.010) - F[:, 0].min(), 0.0 - 0.5 * (F[:, 1].min() + F[:, 1].max()), (Z_TOP + 0.001) - F[:, 2].min()])
                    place(d.mocap_pos[car] + shift, qw)
                others = other_penetrations(m, d, counter_g, M["links"])
                if not others:
                    chosen = (roll, pitch); break
            if chosen: break
        if chosen is None:
            res[mode] = dict(ok=False, note="no roll / pitch of the hand keeps the rest of it off the counter", others=others[:3]); continue
        car0 = d.mocap_pos[car].copy(); plate0 = d.qpos[jp:jp + 3].copy(); D = 0.010 + lip + 0.012; t = 0.0; max_push = 0.0
        while t < 1.3:
            if t >= 0.3: d.mocap_pos[car] = car0 + np.array([-D * min(1.0, (t - 0.3) / 1.0), 0, 0])
            control(Q); mujoco.mj_step(m, d); t += DT
            max_push = max(max_push, float(np.hypot(*(d.qpos[jp:jp + 2] - plate0[:2]))))
        under = (d.qpos[jp] + R_pl - link_points(m, d, M["links"])[:, 0].min()) * 1000
        edge_up = float(d.qpos[jp + 2] - plate0[2]) * 1000 + R_pl * 1000 * abs(d.xmat[pb].reshape(3, 3)[2, 0])
        res[mode] = dict(under_mm=round(float(under), 1), pushed_mm=round(max_push * 1000, 1), edge_up_mm=round(edge_up, 1), roll_deg=chosen[0], pitch_deg=chosen[1], ok=bool(under >= UNDER_MM),
                         shoved=bool(max_push >= 0.5 * D))                    # the plate pushed more than half the slide: the link mostly shoved it instead of entering
        if a.render:
            from PIL import Image; rnd = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = [R_pl, 0, Z_TOP + 0.02]; cam.distance = 0.35; cam.azimuth = 120; cam.elevation = -12
            opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = 1; opt.geomgroup[1] = 1; rnd.update_scene(d, camera=cam, scene_option=opt)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True); Image.fromarray(rnd.render()).save(Path(a.out).parent / f"under_rim_{mode}_{P['id']}.jpg", quality=85); rnd.close()
    res["ok"] = bool(res["fingers"].get("ok") and res["thumb"].get("ok"))
    results.append(res); f, th = res["fingers"], res["thumb"]
    fmt = lambda r: (f"under {r['under_mm']:5.1f} mm pushed {r['pushed_mm']:4.1f}{' SHOVED' if r['shoved'] else ''} (roll {r['roll_deg']:+.0f} pitch {r['pitch_deg']:.0f}) {'ok' if r['ok'] else 'NO'}" if "under_mm" in r else r["note"])
    print(f"{P['id']}: D {P['diameter_mm']:3.0f} clear {P['rim_underside_clearance_mm']:4.1f} lip {P['rim_lip_mm']:4.1f} mm | fingers {fmt(f)} | thumb {fmt(th)} -> {'OK' if res['ok'] else 'FAIL'}", flush=True)
n_ok = sum(r["ok"] for r in results); n_f = sum(bool(r["fingers"].get("ok")) for r in results); n_t = sum(bool(r["thumb"].get("ok")) for r in results); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(n=len(results), ok=n_ok, fingers_ok=n_f, thumb_ok=n_t, under_mm=UNDER_MM, robot=a.robot, made="2026-11-04",
               method="floating right hand (check_pinch.py), plate flat on a counter slab; the four straight fingers / the straight thumb flat on the counter (back 1 mm up), pad up, tip 10 mm outside the +x rim edge, pushed lip + 12 mm inward over 1 s; the hand swung about the link until nothing else rests on the counter",
               plates=results), open(a.out, "w"), indent=1)
print(f"[under_rim] fingers {n_f}/{len(results)}, thumb {n_t}/{len(results)} get {UNDER_MM:g} mm under the rim; both {n_ok}/{len(results)}; {time.time() - t0:.0f} s -> {a.out}")
