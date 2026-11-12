#!/usr/bin/env python
"""Slice start states S1-S6 of the kitchen load-dishwasher task (2026-11-01, docs/long_horizon_slicing_scenes; the slices follow
kitchen7 episode 1 since 2026-11-02).

Samples N setups per slice, builds each in the kitchen teleop env (SceneSetups.apply + env.reset: the kit's SLICE_PRESET lands in
the compiled scene), stands the robot at its drawn start exactly as the teleop CLI does, and checks: robot-furniture and
plate penetrations, that the preset arrived (door angle, upper rack slide, plates standing in the rack, plate spots), and that
the scene holds still for 1 s of physics with the robot pinned (door / rack / lid / plate drift). Renders a top view, the
third-person view and a close-up of the first R setups per slice. Writes <out>/summary.json for the page.

    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/slice_start_check.py --slices S1,S2,S3,S4,S5,S6 --n 8 --render 2 \\
        --out docs/_scan/long_horizon_slicing_scenes/views
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
ap = argparse.ArgumentParser()
ap.add_argument("--slices", default="S1,S2,S3,S4,S5,S6,S7"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--render", type=int, default=2)
ap.add_argument("--seed", type=int, default=300); ap.add_argument("--out", default="docs/_scan/long_horizon_slicing_scenes/views")
ap.add_argument("--robot", default="revo2"); ap.add_argument("--hold-s", type=float, default=1.0)
a = ap.parse_args(); OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)

from simple.teleop.holomotion_v14.slices import SLICES, RACK_BOX
for _S in a.slices.split(","):                                               # the slices' scene knobs (the bin scenes: a pedal that springs back), before the kitchen module loads
    for _k, _v in (SLICES[_S].get("env") or {}).items():
        os.environ.setdefault(_k, _v)
import mujoco, imageio.v2 as imageio, gymnasium as gym
from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene, _stand_on_floor
from simple.teleop.holomotion_v14.scene_setup import SceneSetups, SLICE_STARTS, spawn_shift
from simple.teleop.holomotion_v14.sim_node import robot_yaml
from simple.teleop.holomotion_v14.grasp_start import FINGER_JOINTS, apply_pose, foot_on_pedal, hold_plate, plate_in_hand, nudge_clear, arms_to_walk_default, nudge_back_from_edge, NUDGE_WALK_CLEAR_M, latch_target, grasp_key, held_latch_target
from simple.robots.g1_sonic_revo2 import CLOSED_Q

mod = load_scene("kitchen"); sonic_config = _load_sonic_config()
env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, physics_dt=sonic_config["SIMULATE_DT"], headless=True, max_episode_steps=10 ** 9,
               sonic_config=sonic_config, target=getattr(mod, "TARGET", None), dr_level=3, success_criteria=0.9,
               robot_uid="g1_sonic_revo2" if a.robot == "revo2" else "g1_sonic")
senv = env.unwrapped; task = senv.task; robot = task.robot
env.reset()                                                                  # the robot's joint tables exist after the first reset
y = robot_yaml()
names = list(robot.joint_names[:len(robot.body_joint_index)])
agent = type("Stub", (), {})(); agent.default_real = np.array([float(y["default_joint_angles"][n]) for n in names], dtype=np.float32)
agent.real_to_mjcf = np.arange(len(names))                                 # the stub's "real" order is the MJCF order

CLOSE = {"S1": dict(lookat=[0.45, -0.42, 0.80], distance=3.0, azimuth=200, elevation=-26),       # the counter with both plates (kitchen7 ep1 order since 2026-11-02)
         "S2": dict(lookat=[0.25, 0.95, 0.80], distance=3.0, azimuth=215, elevation=-24),        # the approach to the bin, plate in the right hand
         "S3": dict(lookat=[-0.20, 0.10, 0.80], distance=4.6, azimuth=200, elevation=-30),       # the whole aisle, bin to door
         "S4": dict(lookat=[-0.95, -1.30, 0.55], distance=2.6, azimuth=215, elevation=-24),      # beside the open door, plate in the left hand (azimuth -35 is behind the back wall)
         "S5": dict(lookat=[-0.15, -0.80, 0.70], distance=3.4, azimuth=205, elevation=-26),      # the corridor between the dishwasher and the counter
         "S6": dict(lookat=[-0.30, -0.90, 0.75], distance=3.0, azimuth=205, elevation=-26),      # counter -> door with plate B
         "S7": dict(lookat=[-0.95, -1.30, 0.55], distance=2.6, azimuth=215, elevation=-24),
         # atomic slices (docs/atomic_slicing_scenes): the counter close-ups look along the aisle at the plate, the bin ones at the pedal
         "A1": dict(lookat=[0.20, -0.30, 0.80], distance=3.2, azimuth=200, elevation=-24),
         "A2": dict(lookat=[0.50, -0.35, 0.85], distance=2.4, azimuth=215, elevation=-42),      # from above the aisle, over the worktop edge (a low camera is blocked by the worktop)
         "A3": dict(lookat=[0.30, 0.30, 0.80], distance=3.2, azimuth=215, elevation=-26),
         "A4": dict(lookat=[0.72, 0.95, 0.70], distance=2.4, azimuth=235, elevation=-22),
         "A5": dict(lookat=[0.72, 1.00, 0.55], distance=2.0, azimuth=250, elevation=-18)}
TOP = dict(lookat=[-0.2, -0.5, 0.5], distance=6.2, azimuth=90, elevation=-89)


def shot(m, d, path, cam, W=1280, H=720):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    free = mujoco.MjvCamera(); free.type = mujoco.mjtCamera.mjCAMERA_FREE
    free.lookat[:] = cam["lookat"]; free.distance = cam["distance"]; free.azimuth = cam["azimuth"]; free.elevation = cam["elevation"]
    r = mujoco.Renderer(m, H, W); r.update_scene(d, camera=free); img = r.render(); r.close()
    imageio.imwrite(path, img, quality=88)


def ids(m):
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    return dict(door=m.jnt_qposadr[J("k_dishwasher_door")], upper=m.jnt_qposadr[J("k_dishwasher_upper_rack")], lower=m.jnt_qposadr[J("k_dishwasher_lower_rack")],
                lid=m.jnt_qposadr[J("k_bin_lid")], rack=B("k_dishwasher_upper_rack"), plates={p: B("k_" + p) for p in ("plate_a", "plate_b")}, pelvis=B("pelvis"))


def robot_mask(m, pelvis):
    root = int(m.body_rootid[pelvis]); return np.array([int(m.body_rootid[m.geom_bodyid[g]]) == root for g in range(m.ngeom)])


def penetrations(m, d, who: np.ndarray, floor: np.ndarray, min_mm: float = 0.5, skip: np.ndarray | None = None):
    """(other geom / body name, depth mm) of every contact deeper than min_mm between `who` and anything else but the floor
    (and but `skip`: the held plate against the hand that holds it is reported separately)."""
    out = []
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if who[g1] != who[g2] and not (floor[g1] or floor[g2]) and c.dist < -min_mm / 1000:
            other = g2 if who[g1] else g1
            if skip is not None and skip[other]:
                continue
            out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other])), round(-c.dist * 1000, 1)))
    return out


def hold_like_controller(m, d, robot, held_side: str | None, secs: float, kp_j: float = 150.0, kd_j: float = 5.0, held_target=None):
    """Hold the robot where it stands for `secs` the way a controller would, without resetting any state: the 29 body joints
    under PD toward their current angles, the pelvis under a 6-D virtual spring (xfrc), the holding hand's fingers under the
    hand PD toward the latched command (`held_target`; the closed grip by default, a grasp's own recorded command for the
    ep41 hold -- grasp_start.latch_target), the other hand's toward their current angles; plates, door, racks and lid free.
    (Resetting the robot's qpos every step, as the kit's settle does for furniture, breaks every contact solution: a plate in
    the hand then creeps 9 cm/s even from ep1's own recorded state, which the real controller held at 0.4 mm/s.)"""
    from simple.robots.g1_sonic_revo2 import CLOSED_Q
    held_q = np.asarray(CLOSED_Q[:11] if held_target is None else held_target, dtype=np.float64)[:11]
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    bj = [int(j) for j in robot.body_joint_index]; bq = np.array([int(m.jnt_qposadr[j]) for j in bj]); bv = np.array([int(m.jnt_dofadr[j]) for j in bj])
    ba = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in bj])
    hands = {}
    for side in ("left", "right"):
        js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{n}_joint") for n in FINGER_JOINTS]
        hands[side] = (np.array([int(m.jnt_qposadr[j]) for j in js]), np.array([int(m.jnt_dofadr[j]) for j in js]),
                       np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hand_{n}_joint") for n in FINGER_JOINTS]))
    q0 = d.qpos.copy(); p0, quat0 = d.xpos[pel].copy(), d.xquat[pel].copy()
    ctrl0, t0 = d.ctrl.copy(), float(d.time); vel = np.zeros(6); rot = np.zeros(3)
    for _ in range(int(secs / m.opt.timestep)):
        d.ctrl[:] = 0.0
        d.ctrl[ba] = kp_j * (q0[bq] - d.qpos[bq]) - kd_j * d.qvel[bv]
        for side, (fq, fv, fa) in hands.items():
            target = held_q if side == held_side else q0[fq]
            d.ctrl[fa] = robot.HAND_KP * (target - d.qpos[fq]) - robot.HAND_KD * d.qvel[fv]
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, pel, vel, 0)       # world frame: angular, linear
        mujoco.mju_subQuat(rot, quat0, d.xquat[pel])                                # rotation to go, in the pelvis frame
        R = d.xmat[pel].reshape(3, 3)
        d.xfrc_applied[pel, :3] = 3000.0 * (p0 - d.xpos[pel]) - 150.0 * vel[3:]
        d.xfrc_applied[pel, 3:] = 300.0 * (R @ rot) - 15.0 * vel[:3]
        mujoco.mj_step(m, d)
    d.xfrc_applied[pel] = 0.0; d.ctrl[:] = ctrl0; d.time = t0
    mujoco.mj_forward(m, d)


def rel_rack(m, d, I, b):
    R = d.xmat[I["rack"]].reshape(3, 3); return R.T @ (d.xpos[b] - d.xpos[I["rack"]])


def in_rack(m, d, I, b):
    r = rel_rack(m, d, I, b); return bool(abs(r[0]) <= RACK_BOX["x"] and abs(r[1]) <= RACK_BOX["y"] and RACK_BOX["z"][0] <= r[2] <= RACK_BOX["z"][1])


summary = {"made": time.strftime("%Y-%m-%d %H:%M"), "seed": a.seed, "n": a.n, "slices": {}}
if (OUT / "summary.json").exists():                                          # a rerun of some slices keeps the others' results
    summary["slices"] = json.loads((OUT / "summary.json").read_text()).get("slices", {})
t_all = time.time()
for S in a.slices.split(","):
    st = SLICE_STARTS["kitchen"][S]
    setups = SceneSetups("kitchen", mod, task, ranges="lv3", slice=S)
    rows = []
    for k in range(a.n):
        seed = a.seed + 10 * int(S[1]) + k
        s = setups.sample(seed); setups.apply(s)
        t0 = time.time(); env.reset()
        m, d = senv.mujoco.mjModel, senv.mujoco.mjData
        shift = spawn_shift(s) if (s.get("start") or {}).get("spawn_back_m") else None
        _stand_on_floor(senv, agent, None, shift)                            # a spawn-back start: moved forward to its stance, as the CLI does
        posed = apply_pose(senv, s.get("preset"))                            # an atomic start: ep1's whole-body pose at the boundary
        footed = foot_on_pedal(senv, s.get("preset"))                        # the Wipe: one foot onto the pressed pedal
        held = hold_plate(senv, s.get("preset"))                             # a plate-in-hand start: ep1's grasp, as the CLI does after standing
        nudged = None
        if SLICES[S].get("clean_start"):                                      # A2: the start as the CLI builds it (2026-11-04)
            if SLICES[S].get("start_mode") == "walk":
                arms_to_walk_default(senv); nudged = nudge_back_from_edge(senv, NUDGE_WALK_CLEAR_M)
            else:
                nudged = nudge_clear(senv, held["plate"] if held else None)
        I = ids(m); rm = robot_mask(m, I["pelvis"])
        name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        floor = np.array([m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or "floor" in name(g) for g in range(m.ngeom)])
        plate_mask = np.array([int(m.geom_bodyid[g]) in I["plates"].values() for g in range(m.ngeom)])
        held_mask = np.array([int(m.geom_bodyid[g]) == I["plates"][held["plate"]] for g in range(m.ngeom)]) if held else np.zeros(m.ngeom, bool)
        pen_robot = penetrations(m, d, rm, floor, skip=held_mask if held else (plate_mask if posed else None))   # the held plate against the hand is the grasp, reported below; a hands-at-the-rim pose may touch the plate (reported as plate_robot_pen_mm, > 5 mm fails)
        plate_robot_pen_mm = max([v for _, v in penetrations(m, d, plate_mask, floor, min_mm=0.0, skip=~rm)], default=0.0) if posed and not held else None
        pen_plate = penetrations(m, d, plate_mask, floor, min_mm=5.0, skip=rm if (held or posed) else None)   # a plate resting on a 2.8 mm tine capsule reads 3-4 mm with SIMPLE's soft contacts; the hands' own contact is reported apart
        deepest_plate_mm = max([v for _, v in penetrations(m, d, plate_mask, floor, min_mm=0.0, skip=rm if (held or posed) else None)], default=0.0)
        grasp = None
        if held:
            ok0, dist0, touch0 = plate_in_hand(m, d, held["plate"], held["side"])
            hand_pen = max([v for _, v in penetrations(m, d, held_mask, floor, min_mm=0.0) if True], default=0.0)
            grasp = dict(plate=held["plate"], side=held["side"], radius_cm=round(held["radius_m"] * 100, 1), in_hand=bool(ok0), dist_cm=round(dist0 * 100, 1), touching=bool(touch0),
                         plate_hand_pen_mm=round(hand_pen, 1), plate_z=round(float(d.xpos[I["plates"][held["plate"]]][2]), 3))
        P = s.get("preset") or {}
        state0 = dict(door_deg=round(math.degrees(d.qpos[I["door"]]), 1), upper_m=round(float(d.qpos[I["upper"]]), 3), lower_m=round(float(d.qpos[I["lower"]]), 3),
                      lid_rad=round(float(d.qpos[I["lid"]]), 3), plates={p: d.xpos[b].round(3).tolist() for p, b in I["plates"].items()},
                      in_rack=[p for p, b in I["plates"].items() if in_rack(m, d, I, b)])
        want = dict(door_deg=P.get("door_deg", 0.0), upper_m=P.get("rack_out_m", 0.0), in_rack=sorted(P.get("plates_in_rack", {})), spots=P.get("plate_spots", {}))
        ok_preset = (abs(state0["door_deg"] - want["door_deg"]) < 1.5 and abs(state0["upper_m"] - want["upper_m"]) < 0.01
                     and sorted(state0["in_rack"]) == want["in_rack"]
                     and all(np.hypot(*(np.array(state0["plates"][p][:2]) - np.array(xy[:2]))) < 0.02 for p, xy in want["spots"].items() if not (held and p == held["plate"])))
        # hold still for a while the way the controller would (joint PD, a virtual spring on the pelvis, the holding hand's
        # fingers toward the closed grip = the latched command); plates, door, racks and lid free
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        before = {p: d.xpos[b].copy() for p, b in I["plates"].items()}; jq0 = (d.qpos[I["door"]], d.qpos[I["upper"]], d.qpos[I["lid"]])
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{held['side']}_hand_base_link") if held else None
        rel0 = (d.xmat[hb].reshape(3, 3).T @ (d.xpos[I["plates"][held["plate"]]] - d.xpos[hb])) if held else None
        hold_like_controller(m, d, robot, held["side"] if held else None, a.hold_s, held_target=held_latch_target(P["plate_in_hand"]) if held else None)
        drift = dict(plates_cm={p: round(float(np.linalg.norm(d.xpos[b] - before[p])) * 100, 2) for p, b in I["plates"].items()},
                     door_deg=round(math.degrees(d.qpos[I["door"]] - jq0[0]), 2), upper_cm=round(float(d.qpos[I["upper"]] - jq0[1]) * 100, 2), lid_rad=round(float(d.qpos[I["lid"]] - jq0[2]), 3),
                     pelvis_cm=round(float(np.linalg.norm(d.xpos[I["pelvis"]] - q0[m.jnt_qposadr[m.body_jntadr[I["pelvis"]]]:m.jnt_qposadr[m.body_jntadr[I["pelvis"]]] + 3])) * 100, 2))
        if held:
            ok1, dist1, touch1 = plate_in_hand(m, d, held["plate"], held["side"])
            rel1 = d.xmat[hb].reshape(3, 3).T @ (d.xpos[I["plates"][held["plate"]]] - d.xpos[hb])
            grasp.update(in_hand_after_hold=bool(ok1), slip_in_hand_cm=round(float(np.linalg.norm(rel1 - rel0)) * 100, 2), dist_after_cm=round(dist1 * 100, 1))
        # 2026-11-03 (user: the robot must not touch anything at the beginning): any robot contact after the hold, the arms having
        # sagged under the PD -- the held plate against its hand excepted; the teleop redraws such a start (_start_touches)
        touch_after = penetrations(m, d, rm, floor, min_mm=-0.5, skip=held_mask if held else None)
        d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
        grasp_ok = grasp is None or (grasp["in_hand"] and grasp["in_hand_after_hold"] and grasp["slip_in_hand_cm"] < 2.0)
        pose_ok = plate_robot_pen_mm is None or plate_robot_pen_mm <= 10.0    # fingertips up to 1 cm into the rim: the first physics steps push them out and nudge the plate
        row = dict(nudge=None if nudged is None else dict(back_m=nudged['back_m'], arms_up_rad=nudged['arms_up_rad'], ok=nudged['ok']), seed=seed, setup=s, state0=state0, preset_ok=bool(ok_preset), robot_penetrations=pen_robot, touch_after_hold=touch_after, plate_penetrations=pen_plate, deepest_plate_mm=deepest_plate_mm, drift=drift,
                   grasp=grasp, grasp_ok=bool(grasp_ok), pose=posed, foot=footed, plate_robot_pen_mm=plate_robot_pen_mm, pose_ok=bool(pose_ok),
                   lid_after_hold=round(float(jq0[2] + drift["lid_rad"]), 3),
                   pelvis_z=round(float(d.xpos[I["pelvis"]][2]), 3), reset_s=round(time.time() - t0, 1))
        if k < a.render:
            for view, cam in (("top", TOP), ("third", mod.VIEW_THIRD), ("close", CLOSE[S])):
                shot(m, d, OUT / f"{S}_{k}_{view}.jpg", cam)
            row["views"] = {v: f"{S}_{k}_{v}.jpg" for v in ("top", "third", "close")}
        rows.append(row)
        flag = ("" if ok_preset else " PRESET MISMATCH") + (f" ROBOT TOUCHES {pen_robot}" if pen_robot else "") + (f" PLATE PEN {pen_plate}" if pen_plate else "") + ("" if grasp_ok else f" GRASP FAILED {grasp}") + ("" if pose_ok else f" HANDS {plate_robot_pen_mm} mm INTO THE PLATE")
        ptxt = (f" | start pose " + ", ".join(f"{k} {v}" for k, v in posed.items() if k not in ("pelvis_z", "source")) + f", pelvis {row['pelvis_z']} m, hands-plate pen {plate_robot_pen_mm} mm") if posed else ""
        ptxt += f" | {footed['foot']} foot on the pedal: IK {footed['err_mm']} mm / {footed['err_deg']} deg, lid after the hold {row['lid_after_hold']} rad" if footed else ""
        gtxt = (f" | {grasp['plate']} in the {grasp['side']} hand: {grasp['radius_cm'] * 2:.0f} cm plate, {grasp['dist_cm']} cm from the hand base, touching {grasp['touching']}, "
                f"plate-hand pen {grasp['plate_hand_pen_mm']} mm, after 1 s held {grasp['in_hand_after_hold']} slip {grasp['slip_in_hand_cm']} cm") if grasp else ""
        print(f"[{S} {k}] seed {seed}: {SceneSetups.describe(s)} | state door {state0['door_deg']} deg, rack {state0['upper_m']} m, in rack {state0['in_rack']} | "
              f"drift plates {drift['plates_cm']} cm, door {drift['door_deg']} deg, rack {drift['upper_cm']} cm, pelvis {drift['pelvis_cm']} cm{gtxt}{ptxt} | reset {row['reset_s']} s{flag}", flush=True)
    summary["slices"][S] = dict(spec=json.loads(json.dumps(st, default=list)), cfg=SLICES[S], rows=rows,
                                n_ok=sum(1 for r in rows if r["preset_ok"] and not r["robot_penetrations"] and not r["plate_penetrations"] and r["grasp_ok"] and r["pose_ok"] and not r.get("touch_after_hold")))
    n_touch = sum(1 for r in rows if r.get("touch_after_hold"))
    print(f"[{S}] {summary['slices'][S]['n_ok']}/{len(rows)} starts clean (preset as drawn, no robot / plate penetration at the start or after the {a.hold_s:g} s hold, the held plate stays in the hand)"
          + (f"; {n_touch} touched something after the hold: " + "; ".join(str(r["touch_after_hold"][:2]) for r in rows if r.get("touch_after_hold"))[:300] if n_touch else ""), flush=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(f"done in {time.time() - t_all:.0f} s -> {OUT}/summary.json")
