#!/usr/bin/env python
"""Randomization maps for the atomic slice scenes A1-A5 (user 2026-11-03: "for each scene, please also provide a location and
orientation randomization map so i can tell whether the randomization is enough, also for the robot pose").

Per scene: N pure draws of the start (SceneSetups.sample, no physics) -> robot start positions + headings and plate spots over an
orthographic top view of the kitchen, the drawn arm / waist joints against the mechanical joint ranges, the hands' fingertip
positions in the pelvis frame (forward kinematics on the robot model), the free-hand pose mix; and M full builds (apply + reset +
stand + pose + foot + hold, as the CLI does) rendered from a camera tied to the pelvis -> a grid of starts and one blended "pose fan".

    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/slice_rand_map.py --n 300 --m 12 --out docs/_scan/atomic_slicing_scenes/rand

Traps: the kitchen kit recompiles the model at every env.reset() (rebind() re-reads the handles and ids); the orthographic free
camera's pixel scale is fitted from a plate placed at known points (segmentation render); the setups' robot frame is the compiled
kitchen's world frame (checked: the pelvis lands at the setup's x, y to the mm).
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
ap = argparse.ArgumentParser()
ap.add_argument("--slices", default="A1,A2,A3,A4,A5"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--m", type=int, default=12)
ap.add_argument("--seed", type=int, default=5000); ap.add_argument("--out", default="docs/_scan/atomic_slicing_scenes/rand")
ap.add_argument("--skip-builds", action="store_true")
a = ap.parse_args(); OUT = ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)
SL = a.slices.split(",")

from simple.teleop.holomotion_v14.slices import SLICES
for _S in SL:
    for _k, _v in (SLICES[_S].get("env") or {}).items():
        os.environ.setdefault(_k, _v)
import mujoco, imageio.v2 as imageio, gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from scipy.ndimage import map_coordinates
from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene, _stand_on_floor
from simple.teleop.holomotion_v14.scene_setup import SceneSetups, SLICE_STARTS, spawn_shift, PEDAL_XY, load_bank, a1_envelope, A2_START
from simple.teleop.holomotion_v14.sim_node import robot_yaml
from simple.teleop.holomotion_v14.grasp_start import (ARM_JOINTS, FINGER_JOINTS, ARM_RANGE, ARM_RANGE_LOW, ARM_RANGE_NEAR, WAIST_RANGE, CARRY_JITTER,
                                                      CARRY_JITTER_LEFT, POSE_JITTER, GRASPS, POSES, grasp_key, apply_pose, foot_on_pedal, hold_plate)
from simple.robots.g1_sonic_revo2 import CLOSED_Q, OPEN_Q, CLOSE2_Q

COLOR = {"A1": "#2e8b57", "A2": "#1f6f5f", "A3": "#c98a1b", "A4": "#b5533c", "A5": "#8a5fbf"}
NAME = {"A1": "Move to the counter", "A2": "Pick the plate up", "A3": "Move the plate to the bin", "A4": "Step on the bin pedal", "A5": "Wipe the plate over the bin"}
EP1 = {"A1": (0.24, -0.21, -2.0, "ep1 3.2 s (end of the step-in)"), "A2": (0.24, -0.21, -2.0, "ep1 3.2 s (start of the reach)"),
       "A3": (0.18, -0.23, 10.5, "ep1 29.6 s (plate lifted; stance of the pick)"), "A4": (0.71, 1.20, -76.0, "ep1 51.5 s (pedal stance)"),
       "A5": (0.63, 1.21, -63.0, "ep1 53.0 s (mid-swipe)")}
EP1_START = {"A1": (0.06, 0.03, 0.0, "ep1 0 s (task start)")}
LEFT, RIGHT, PL_A, PL_B = "#2a6fdb", "#d6452b", "#1a9850", "#7b3294"
BANKC = "#7b3fa0"                                                            # the recorded A1 ends' span (the bank start kinds of A2, 2026-11-03)
try:
    A1E = load_bank("a1_end")["entries"]
except FileNotFoundError:
    A1E = []
A2_KINDS = {v[1] for v in SLICE_STARTS["kitchen"]["A2"].get("variants", [])}
A2_FROM_A1 = A2_START in ("a1", "a1_mix", "a1_end")

mod = load_scene("kitchen"); sonic_config = _load_sonic_config()
env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, physics_dt=sonic_config["SIMULATE_DT"], headless=True, max_episode_steps=10 ** 9,
               sonic_config=sonic_config, target=getattr(mod, "TARGET", None), dr_level=3, success_criteria=0.9, robot_uid="g1_sonic_revo2")
senv = env.unwrapped; task = senv.task; robot = task.robot
env.reset()
y = robot_yaml()
names = list(robot.joint_names[:len(robot.body_joint_index)])
agent = type("Stub", (), {})(); agent.default_real = np.array([float(y["default_joint_angles"][n]) for n in names], dtype=np.float32)
agent.real_to_mjcf = np.arange(len(names))
WORKTOP = float(mod.BS.WORKTOP_TOP)
NO_SITES = mujoco.MjvOption(); NO_SITES.sitegroup[0] = 0                     # the env's red com_marker site at the world origin (and the imu site) off the renders


def rebind():
    """The kitchen kit recompiles the model at every env.reset(): re-read the model / data handles and the id tables."""
    global m, d, J, B, PEL, ROOTB, ROOTQ, BJ, BQ, TIPS, HANDB, PLATEB, PLATEQ
    m, d = senv.mujoco.mjModel, senv.mujoco.mjData
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    PEL = B("pelvis"); ROOTB = int(m.body_rootid[PEL]); ROOTQ = int(m.jnt_qposadr[m.body_jntadr[ROOTB]])
    BJ = np.asarray(robot.body_joint_index, dtype=int); BQ = m.jnt_qposadr[BJ]
    TIPS = {s: [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{s}_hand_") and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("_tip_link")] for s in ("left", "right")}
    HANDB = {s: B(f"{s}_hand_base_link") for s in ("left", "right")}
    PLATEB = {p: B("k_" + p) for p in ("plate_a", "plate_b")}
    PLATEQ = {p: int(m.jnt_qposadr[J(f"k_{p}_free")]) for p in ("plate_a", "plate_b")}


rebind()
ROWS = [(f"{s} {n.replace('_', ' ')}", f"{s}_{n}_joint", s, k) for s in ("left", "right") for k, n in enumerate(ARM_JOINTS)] + \
       [(f"waist {n}", f"waist_{n}_joint", "waist", k) for k, n in enumerate(("yaw", "roll", "pitch"))]
JRANGE = {jn: (float(m.jnt_range[J(jn)][0]), float(m.jnt_range[J(jn)][1])) for _, jn, _, _ in ROWS}
QDEF = agent.default_real.astype(np.float64)
print(f"worktop top {WORKTOP:.3f} m; tips {[len(v) for v in TIPS.values()]}; joint ranges e.g. {JRANGE['left_shoulder_pitch_joint']}", flush=True)


# ------------------------------------------------------------------------------------------------------------ top view
def ortho_top(W=1600, H=1600, lookat=(0.2, 0.3, 0.45), dist=5.6):
    """An orthographic top view of the kitchen without the robot and plates, and the affine world (x, y) -> pixel (col, row)
    fitted from the plates put at known places (MuJoCo's orthographic free camera: `distance` is the view height)."""
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.orthographic = 1
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = 90; cam.elevation = -90
    q0 = d.qpos.copy()
    rgba0 = m.geom_rgba.copy()
    for g in range(m.ngeom):                                                 # small red markers in the aisle (SIMPLE's target / leash markers) off the map
        c = m.geom_rgba[g]
        if c[0] > 0.8 and c[1] < 0.3 and c[2] < 0.3 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_size[g][0] < 0.08:
            print(f"  hidden red marker geom {g} {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)!r} body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))!r} at {d.geom_xpos[g].round(2).tolist()}")
            m.geom_rgba[g, 3] = 0.0
    d.qpos[ROOTQ:ROOTQ + 2] = (40.0, 40.0)                                   # the robot out of the picture
    for p, qa in PLATEQ.items():
        d.qpos[qa:qa + 3] = (40.0, 40.0 + (0 if p == "plate_a" else 1), 1.0)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W); r.update_scene(d, camera=cam, scene_option=NO_SITES); img = r.render().copy()
    # calibration: plate_a at five known places over open floor / worktop (z 1.0 m), segmentation centroid
    r.enable_segmentation_rendering()
    pts = []
    for (x, yy) in ((-0.9, -1.0), (-0.9, 1.4), (1.2, -1.0), (1.2, 1.4), (0.2, 0.3)):
        d.qpos[PLATEQ["plate_a"]:PLATEQ["plate_a"] + 3] = (x, yy, 1.0); d.qpos[PLATEQ["plate_a"] + 3:PLATEQ["plate_a"] + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(m, d); r.update_scene(d, camera=cam, scene_option=NO_SITES); seg = r.render()
        gids = seg[..., 0]; mask = np.isin(gids, [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == PLATEB["plate_a"]]) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        assert mask.sum() > 50, f"calibration plate not seen at ({x}, {yy}): {mask.sum()} px"
        rr, cc = np.nonzero(mask); pts.append((x, yy, cc.mean(), rr.mean()))
    r.disable_segmentation_rendering(); r.close()
    m.geom_rgba[:] = rgba0
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
    P = np.array(pts); A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    col = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]; row = np.linalg.lstsq(A, P[:, 3], rcond=None)[0]
    res = np.hypot(A @ col - P[:, 2], A @ row - P[:, 3]).max()
    print(f"top view calibration: col = {col.round(2)} . (x, y, 1), row = {row.round(2)} . (x, y, 1); residual {res:.2f} px; px/m {np.hypot(col[0], row[0]):.1f} / {np.hypot(col[1], row[1]):.1f}", flush=True)
    return img, col, row


def resample(img, col, row, x0, x1, y0, y1, mpp=0.004):
    """The top view on a regular world grid (x right, y up), for imshow(extent=(x0, x1, y0, y1))."""
    xs = np.arange(x0, x1, mpp); ys = np.arange(y1, y0, -mpp)
    X, Y = np.meshgrid(xs, ys)
    C = col[0] * X + col[1] * Y + col[2]; R = row[0] * X + row[1] * Y + row[2]
    out = np.stack([map_coordinates(img[..., k].astype(np.float32), [R, C], order=1, mode="constant", cval=235.0) for k in range(3)], -1)
    return np.clip(out, 0, 255).astype(np.uint8), (x0, x1, y0, y1)


# ------------------------------------------------------------------------------------------------------------ draws
from simple.teleop.holomotion_v14.grasp_start import hand_kind              # flat / relaxed / curled (the resting-hand draw), closed grip, pinch


def fk_hands(setup):
    """Fingertip centroids and hand-base points of both hands in the pelvis frame for a drawn start (no physics)."""
    P = setup.get("preset") or {}
    q0 = d.qpos.copy()
    pz = P.get("pose")
    body = np.asarray((POSES[pz] if isinstance(pz, str) else pz)["body"], dtype=np.float64) if pz else QDEF.copy()   # a POSES name or an inline bank pose
    d.qpos[BQ] = body
    H = P.get("plate_in_hand")
    arms = dict(P.get("arms") or {})
    if H:
        arms[H["side"]] = H.get("arm") or list(GRASPS[grasp_key(H)]["arm"])
    for s, q in arms.items():
        for n, v in zip(ARM_JOINTS, q):
            d.qpos[m.jnt_qposadr[J(f"{s}_{n}_joint")]] = v
    if P.get("waist"):
        for n, v in zip(("yaw", "roll", "pitch"), P["waist"]):
            d.qpos[m.jnt_qposadr[J(f"waist_{n}_joint")]] = v
    mujoco.mj_kinematics(m, d)
    R = d.xmat[PEL].reshape(3, 3); p0 = d.xpos[PEL]
    out = {}
    for s in ("left", "right"):
        tip = np.mean([d.xpos[b] for b in TIPS[s]], axis=0)
        out[s] = dict(tip=(R.T @ (tip - p0)).round(4).tolist(), base=(R.T @ (d.xpos[HANDB[s]] - p0)).round(4).tolist())
    # the joint values the strips show
    jv = {}
    for label, jn, s, k in ROWS:
        jv[jn] = float(d.qpos[m.jnt_qposadr[J(jn)]])
    d.qpos[:] = q0; mujoco.mj_kinematics(m, d)
    return out, jv


def draws(S, n):
    setups = SceneSetups("kitchen", mod, task, ranges="lv3", slice=S)
    rows = []
    for k in range(n):
        seed = a.seed + 1000 * int(S[1]) + k
        s = setups.sample(seed)
        P = s.get("preset") or {}; st = s.get("start") or {}; H = P.get("plate_in_hand")
        hands, jv = fk_hands(s)
        rad = {p: setups._plate_radius(seed, H, p, SLICE_STARTS["kitchen"][S].get("plate_b")) for p in ("plate_a", "plate_b")}
        rows.append(dict(seed=seed, robot=s["robot"], variant=st.get("variant"), at_plate=st.get("at_plate") or st.get("plate"), ahead_m=st.get("ahead_m"), left_m=st.get("left_m"),
                         bank_episode=st.get("bank_episode"), spots=P.get("plate_spots") or {}, held=H, foot=P.get("foot_on_pedal"),
                         pose=(P["pose"].get("source") if isinstance(P.get("pose"), dict) else P.get("pose")),
                         hands={sd: hand_kind(q) for sd, q in (P.get("hands") or {}).items()}, waist=P.get("waist"), arms=P.get("arms"),
                         plate=s.get("plate"), plate_b=s.get("plate_b"), radius=rad, fk=hands, joints=jv))
    return rows


def spec_range(S, side, k, row):
    """The band the strip shows for a joint of an arm (or the waist): the spec's range (free arm / waist), the carry pose +-
    jitter (held arm), the pose +- jitter (whole-body pose), the recorded ends' span +- jitter (a bank kind) or nothing (not
    drawn, default pose)."""
    st = SLICE_STARTS["kitchen"][S]
    specs = [v[2] for v in st["variants"]] if "variants" in st else [st]
    bands = []
    for sp in specs:
        R = (sp.get("preset") or {}).get("rand") or {}
        held = (sp.get("preset") or {}).get("plate_in_hand")
        held_sides = []
        if held:
            held_sides = list(held["side"]) if isinstance(held["side"], (tuple, list)) else [held["side"]]
        if sp.get("a1_env"):                                                 # the A1-envelope start: the band each joint is drawn in
            env = a1_envelope(sp["a1_env"], **(sp.get("widen") or {}))
            lo, hi = env["waist"][k] if side == "waist" else env["arms"][side][k]
            bands.append(("bank", lo, hi))
            continue
        bank = sp.get("bank") or sp.get("bank_pose")
        if bank:                                                             # A2 from the A1 ends: the recorded span of that joint, plus the jitter
            E = load_bank(bank)["entries"]; jt = sp.get("jitter") or {}
            idx = 12 + k if side == "waist" else 15 + 7 * (side == "right") + k
            vals = [e["body"][idx] for e in E]; j = jt.get("waist_rad", 0.0) if side == "waist" else jt.get("arm_rad", 0.0)
            bands.append(("bank", min(vals) - j, max(vals) + j))
            continue
        if side == "waist":
            if R.get("waist"):
                bands.append(("free", *WAIST_RANGE[k]))
            continue
        if (sp.get("preset") or {}).get("pose"):
            base = POSES[sp["preset"]["pose"]]["body"][15 + 7 * (side == "right") + k]
            j = POSE_JITTER if R.get("pose_jitter") else 0.0
            bands.append(("pose", base - j, base + j))
            continue
        arms = R.get("arms"); arms = ("left", "right") if arms == "free" else (arms or ())
        if side in arms:
            rng = {"low": ARM_RANGE_LOW, "near": ARM_RANGE_NEAR}.get(R.get("arms_range", "full"), ARM_RANGE)[k]
            lo, hi = rng
            if side == "right" and k in (1, 2, 4, 6):
                lo, hi = -hi, -lo
            bands.append(("free", lo, hi))
        if side in held_sides and R.get("carry_jitter"):
            base = GRASPS[side]["arm"][k]; j = CARRY_JITTER[k] * (CARRY_JITTER_LEFT if side == "left" else 1.0)
            bands.append(("held", base - j, base + j))
    return bands


# ------------------------------------------------------------------------------------------------------------ figures
def draw_map(ax, S, rows, bg, ext, title=None, short=False):
    st = SLICE_STARTS["kitchen"][S]
    ax.imshow(bg, extent=ext, interpolation="bilinear", alpha=0.9)
    ax.set_xlim(-1.0, 1.4); ax.set_ylim(-1.15, 1.65); ax.set_aspect("equal")
    ax.set_xlabel("x (m), robot frame of the full task (+x toward the worktop)" if short else "x (m), the robot frame of the full task: +x = the task start's heading, toward the worktop"); ax.set_ylabel("y (m), +y = left")
    ax.grid(True, color="w", alpha=0.35, lw=0.6); ax.set_xticks(np.arange(-1.0, 1.41, 0.25)); ax.set_yticks(np.arange(-1.0, 1.51, 0.25))
    ax.tick_params(labelsize=8)
    # the spec's region
    specs = [v[2] for v in st["variants"]] if "variants" in st else [st]
    for sp in specs:
        if "box" in sp:
            b = sp["box"]; ax.add_patch(Rectangle((b["x"][0], b["y"][0]), b["x"][1] - b["x"][0], b["y"][1] - b["y"][0], fill=False, ec=COLOR[S], lw=1.6, ls="--"))
        if "at_counter" in sp:                                                # the A1 stance band (pelvis x and y limits)
            b = sp["at_counter"]; ax.add_patch(Rectangle((b["x_m"][0], b["y_m"][0]), b["x_m"][1] - b["x_m"][0], b["y_m"][1] - b["y_m"][0], fill=False, ec=COLOR[S] if sp.get("a1_env") else "#0b3d91", lw=1.6, ls="--"))
        for p, (xr, yr) in ((sp.get("preset") or {}).get("plate_spots") or {}).items():
            ax.add_patch(Rectangle((xr[0], yr[0]), xr[1] - xr[0], yr[1] - yr[0], fill=False, ec=PL_A if p == "plate_a" else PL_B, lw=1.2, ls=":"))
    # worktop edge, bin, pedal
    ax.plot([0.475, 0.475], [-1.43, 0.64], color="k", lw=0.8, alpha=0.5); ax.text(0.49, 0.60, "worktop edge", fontsize=7, color="k", alpha=0.7, rotation=90, va="top")
    ax.add_patch(Circle(PEDAL_XY, 0.03, fc="none", ec="k", lw=1.0)); ax.text(PEDAL_XY[0] + 0.05, PEDAL_XY[1], "pedal", fontsize=7.5, va="center")
    # plate spots
    for p, c in (("plate_a", PL_A), ("plate_b", PL_B)):
        pts = [(r["spots"][p], r["radius"][p]) for r in rows if p in r["spots"]]
        for (xy, rad) in pts:
            ax.add_patch(Circle(xy[:2], rad, fc=c, ec="none", alpha=0.045))
        if pts:
            xy = np.array([q[0][:2] for q in pts]); ax.scatter(xy[:, 0], xy[:, 1], s=5, c=c, alpha=0.75, lw=0, zorder=4)
            yaw = np.array([q[0][2] if len(q[0]) > 2 else 0.0 for q in pts]); L_ = 0.045
            ax.quiver(xy[:, 0], xy[:, 1], L_ * np.cos(np.radians(yaw)), L_ * np.sin(np.radians(yaw)), color=c, alpha=0.35, scale=1, scale_units="xy", angles="xy", width=0.0025, headwidth=2.5, headlength=3)
    # robot starts: arrow along the heading, coloured by variant / held side
    for r in rows:
        x, yy, yaw = r["robot"]["x"], r["robot"]["y"], math.radians(r["robot"]["yaw_deg"])
        c = COLOR[S]
        if r.get("variant") in ("rim", "a1_fit"): c = "#0b3d91"
        if r.get("held"): c = LEFT if r["held"]["side"] == "left" else RIGHT
        if S == "A5": c = LEFT if r.get("foot") == "left" else RIGHT
        ax.add_patch(FancyArrow(x, yy, 0.10 * math.cos(yaw), 0.10 * math.sin(yaw), width=0.004, head_width=0.03, head_length=0.03, fc=c, ec=c, alpha=0.6, lw=0.3))
    if S == "A2" and A2_FROM_A1 and A1E:                                     # where the 37 recorded A1 episodes actually ended
        ax.plot([e["robot"]["x"] for e in A1E], [e["robot"]["y"] for e in A1E], "+", color="k", ms=7, mew=1.3, alpha=0.85, lw=0, zorder=5)
    for key in ((S,),):
        for tup in (EP1.get(S), EP1_START.get(S)):
            if tup:
                x, yy, yaw, lab = tup; yaw = math.radians(yaw)
                ax.add_patch(Circle((x, yy), 0.045, fc="none", ec="k", lw=1.8, zorder=6)); ax.add_patch(FancyArrow(x, yy, 0.16 * math.cos(yaw), 0.16 * math.sin(yaw), width=0.006, head_width=0.045, head_length=0.045, fc="k", ec="k", zorder=6))
                ax.annotate(lab, (x, yy), (6, -14), textcoords="offset points", fontsize=7.5, color="k", zorder=7, bbox=dict(fc="w", ec="none", alpha=0.75, pad=1))
    leg = []
    if S == "A2" and A2_START == "a1":
        leg = [plt.Line2D([], [], color=COLOR[S], lw=3, label="start drawn in the A1 envelope (dashed = its x / y limits), validity-checked"),
               plt.Line2D([], [], color="k", marker="+", lw=0, ms=8, mew=1.3, label=f"the {len(A1E)} recorded A1 ends the envelope comes from")]
    elif S == "A2" and A2_FROM_A1:
        leg = [plt.Line2D([], [], color=COLOR[S], lw=3, label=f"kind a1_end: a recorded A1 end as it was (1 of {len(A1E)}, +-1 cm, +-3 deg)"),
               plt.Line2D([], [], color="#0b3d91", lw=3, label="kind a1_fit: the A1 stance band, new plates (dashed = its x / y limits)"),
               plt.Line2D([], [], color="k", marker="+", lw=0, ms=8, mew=1.3, label=f"the {len(A1E)} recorded A1 ends")]
    elif S == "A2": leg = [plt.Line2D([], [], color=COLOR[S], lw=3, label="start kind counter (7 of 10)"), plt.Line2D([], [], color="#0b3d91", lw=3, label="start kind hands at the rim (3 of 10)")]
    elif S in ("A3", "A4"): leg = [plt.Line2D([], [], color=RIGHT, lw=3, label="plate in the right hand"), plt.Line2D([], [], color=LEFT, lw=3, label="plate in the left hand")]
    elif S == "A5": leg = [plt.Line2D([], [], color=RIGHT, lw=3, label="right foot on the pedal"), plt.Line2D([], [], color=LEFT, lw=3, label="left foot on the pedal")]
    else: leg = [plt.Line2D([], [], color=COLOR[S], lw=3, label="robot start (arrow = heading)")]
    leg += [plt.Line2D([], [], color=PL_A, marker="o", lw=0, label="plate A centre (tick = its turn)"), plt.Line2D([], [], color=PL_B, marker="o", lw=0, label="plate B centre"),
            plt.Line2D([], [], color=COLOR[S], lw=1.6, ls="--", label="stance box of the spec"), plt.Line2D([], [], color="k", marker="o", mfc="none", lw=0, ms=9, label="ep1's own stance")]
    ax.legend(handles=leg, loc="lower left", fontsize=7.5, framealpha=0.9)
    ax.set_title(title or f"{S} {NAME[S]}: {len(rows)} drawn starts, top view", fontsize=9.5 if title else 11, loc="left")


def draw_rose(axr, S, rows):
    yaws = np.radians([r["robot"]["yaw_deg"] for r in rows])
    cnt, edges = np.histogram(yaws, bins=36, range=(-np.pi, np.pi))
    axr.bar((edges[:-1] + edges[1:]) / 2, cnt, width=np.diff(edges), color=COLOR[S], alpha=0.8, edgecolor="w", lw=0.4)
    axr.set_theta_zero_location("E"); axr.set_theta_direction(1); axr.set_xticks(np.radians(np.arange(0, 360, 45))); axr.tick_params(labelsize=7)
    axr.set_yticklabels([]); axr.set_title(f"robot heading (deg; 0 = +x): {np.degrees(yaws.min()):+.0f}..{np.degrees(yaws.max()):+.0f}, 10 deg bins", fontsize=8.5)
    e = EP1.get(S)
    if e: axr.plot([math.radians(e[2])] * 2, [0, cnt.max()], color="k", lw=1.5)


def draw_target(axp, S, rows):
    """Where the target (plate / pedal) is in the pelvis frame, or the stance x/y histogram."""
    if S in ("A2", "A5"):
        ah = np.array([r["ahead_m"] for r in rows if r.get("ahead_m") is not None]); lf = np.array([r["left_m"] for r in rows if r.get("left_m") is not None])
        kinds = [r.get("variant") or "" for r in rows]
        cs = ["#0b3d91" if v in ("rim", "a1_fit") else COLOR[S] for v in kinds] if S == "A2" else [LEFT if r.get("foot") == "left" else RIGHT for r in rows]
        axp.scatter(-lf, ah, s=9, c=cs, alpha=0.7, lw=0)
        if S == "A2" and A2_FROM_A1 and A1E:                                 # the nearer plate at the recorded A1 ends
            axp.scatter([-e["rel"][e["target"]]["left"] for e in A1E], [e["rel"][e["target"]]["ahead"] for e in A1E], marker="+", c="k", s=36, lw=1.3, zorder=5)
        axp.scatter([0], [0], marker="s", s=60, c="k"); axp.text(0.012, -0.012, "pelvis", fontsize=7.5)
        axp.set_xlabel("to the right of the pelvis (m)"); axp.set_ylabel("ahead of the pelvis (m)")
        axp.set_title("the plate's centre (A2) / the pedal (A5) in the pelvis frame", fontsize=8.5)
        if S == "A2":
            axp.plot([0.03], [0.38], "ko", mfc="none", ms=9); axp.text(0.035, 0.385, "ep1 3.2 s", fontsize=7); axp.plot([0.009], [0.366], "k+", ms=9); axp.text(0.014, 0.355, "ep1 16.4 s", fontsize=7)
        else:
            axp.plot([-0.02], [0.18], "ko", mfc="none", ms=9); axp.text(-0.015, 0.185, "ep1 53.0 s", fontsize=7)
        axp.set_aspect("equal"); axp.grid(True, alpha=0.3); axp.invert_xaxis() if False else None
        lim = max(0.55, ah.max() + 0.05); axp.set_ylim(-0.05, lim); axp.set_xlim((-0.32, 0.32) if S == "A2" and A2_FROM_A1 else (-0.2, 0.2))
    else:
        xs = np.array([r["robot"]["x"] for r in rows]); ys = np.array([r["robot"]["y"] for r in rows])
        axp.hist2d(xs, ys, bins=[10, 10], cmap="Greens" if S == "A1" else "Oranges" if S == "A3" else "Reds", cmin=0.5)
        axp.set_aspect("equal"); axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)"); axp.set_title("start positions per 10 x 10 cell of the box (count)", fontsize=8.5)
        for (i, j) in [(0, 0)]: pass


def fig_map(S, rows, bg, ext):
    fig = plt.figure(figsize=(12.5, 8.2), dpi=110)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1], wspace=0.22, hspace=0.3)
    draw_map(fig.add_subplot(gs[:, 0]), S, rows, bg, ext); draw_rose(fig.add_subplot(gs[0, 1], projection="polar"), S, rows); draw_target(fig.add_subplot(gs[1, 1]), S, rows)
    fig.text(0.01, 0.005, "SceneSetups.sample draws (lv3 ranges, --slice %s); the per-episode start jitter SIMPLE adds at reset (+-2 cm, +-3 deg) is not shown. Plate discs at their drawn radius (165-260 mm types)." % S, fontsize=7, color="#555")
    p = OUT / f"map_{S}.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p.name


def draw_strips(ax, S, rows, fs=8):
    st = SLICE_STARTS["kitchen"][S]
    for i, (label, jn, side, k) in enumerate(ROWS):
        yy = len(ROWS) - 1 - i
        lo, hi = JRANGE[jn]; ax.plot([lo, hi], [yy, yy], color="#d9d9d9", lw=7, solid_capstyle="butt", zorder=1)
        bands = spec_range(S, side, k, i)
        for kind, b0, b1 in bands:
            c = {"free": COLOR[S], "held": "#777", "pose": "#0b3d91", "bank": BANKC}[kind]
            ax.plot([b0, b1], [yy, yy], color=c, lw=7, alpha=0.55, solid_capstyle="butt", zorder=2)
        vals = np.array([r["joints"][jn] for r in rows])
        ax.scatter(vals, yy + np.random.RandomState(i).uniform(-0.22, 0.22, len(vals)), s=4, c="k", alpha=0.35, lw=0, zorder=3)
        dq = QDEF[names.index(jn)] if jn in names else 0.0
        ax.plot([dq], [yy], marker="|", color="#e6194b", ms=14, mew=2, zorder=4)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=fs)
    ax.set_xlabel("joint angle (rad)"); ax.grid(True, axis="x", alpha=0.3); ax.set_xlim(-3.2, 3.2)
    ax.set_title("drawn joints (dots), the spec's band (colour) inside the mechanical range (grey)", fontsize=8.5, loc="left")


def draw_tips_side(ax2, S, rows):
    for s, c in (("left", LEFT), ("right", RIGHT)):
        P = np.array([r["fk"][s]["tip"] for r in rows]); ax2.scatter(P[:, 0], P[:, 2], s=7, c=c, alpha=0.5, lw=0, label=f"{s} fingertips")
    ax2.axhline(WORKTOP - 0.761, color="k", ls="--", lw=0.8); ax2.text(-0.43, WORKTOP - 0.761 + 0.01, "worktop top (default stance)", fontsize=7)
    ax2.plot([0, 0], [-0.1, 0.45], color="k", lw=3, alpha=0.3); ax2.plot([0], [0], "ks", ms=6); ax2.text(0.01, -0.03, "pelvis", fontsize=7.5)
    if S == "A2" and A2_FROM_A1:
        ax2.axvspan(0.475 - 0.42, 0.475 - 0.26, color="#555", alpha=0.10); ax2.text(0.475 - 0.26 + 0.005, 0.42, "worktop edge over\nthe stance band\n(6-22 cm ahead)", fontsize=7, color="#555")
    elif S == "A2":
        ax2.axvline(0.22, color="#555", ls=":", lw=1); ax2.text(0.225, 0.42, "nearest worktop edge\n(counter kind)", fontsize=7, color="#555")
    if S == "A4":
        ax2.axvline(0.35, color="#555", ls=":", lw=1); ax2.text(0.355, 0.42, "nearest bin wall", fontsize=7, color="#555")
    ax2.set_xlabel("ahead of the pelvis (m)"); ax2.set_ylabel("above the pelvis (m)"); ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.45, 0.65); ax2.set_ylim(-0.2, 0.5); ax2.legend(fontsize=7.5, loc="upper left"); ax2.set_title("fingertip centroid of each hand, side view (pelvis frame)", fontsize=8.5)


def draw_tips_top(ax3, S, rows):
    for s, c in (("left", LEFT), ("right", RIGHT)):
        P = np.array([r["fk"][s]["tip"] for r in rows]); ax3.scatter(-P[:, 1], P[:, 0], s=7, c=c, alpha=0.5, lw=0)
    ax3.plot([0], [0], "ks", ms=6); ax3.plot([-0.14, 0.14], [0, 0], color="k", lw=3, alpha=0.3)
    ax3.set_xlabel("to the right of the pelvis (m)"); ax3.set_ylabel("ahead (m)"); ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.6, 0.6); ax3.set_ylim(-0.3, 0.65); ax3.set_title("fingertips, top view", fontsize=8.5)


def draw_hands(ax4, S, rows):
    kinds = ["flat", "relaxed", "curled", "closed grip", "pinch", "held grip"]
    width = 0.38
    for j, (s, c) in enumerate((("left", LEFT), ("right", RIGHT))):
        cnt = {k: 0 for k in kinds}
        for r in rows:
            if r.get("held") and r["held"]["side"] == s: cnt["held grip"] += 1
            elif s in r["hands"]: cnt[r["hands"][s] if r["hands"][s] in cnt else "curled"] += 1
            elif r.get("pose") and r["pose"] in POSES:
                k = hand_kind(POSES[r["pose"]]["fingers"][s]); cnt[k if k in cnt else "held grip"] += 1
            else: cnt["flat"] += 1
        vals = [100.0 * cnt[k] / len(rows) for k in kinds]
        ax4.bar(np.arange(len(kinds)) + (j - 0.5) * width, vals, width, color=c, alpha=0.85, label=f"{s} hand")
        for i, v in enumerate(vals):
            if v > 0: ax4.text(i + (j - 0.5) * width, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax4.set_xticks(range(len(kinds))); ax4.set_xticklabels(kinds, fontsize=7.5, rotation=25, ha="right"); ax4.set_ylabel("% of draws"); ax4.set_ylim(0, 100); ax4.legend(fontsize=7.5)
    ax4.set_title("hand start pose (latched until the operator squeezes)", fontsize=8.5); ax4.grid(True, axis="y", alpha=0.3)


BANDS_NOTE = "Bands: scene colour = a free arm's drawn range; grey = the holding arm around ep1's carry / hold pose (+- CARRY_JITTER); blue = ep1's whole-body pose +-0.015 rad; purple = the span of the 37 recorded A1 ends +- the jitter (A2's bank kinds). Red tick = the controller's default pose. Fingertip = mean of the five tip links."


def fig_pose(S, rows):
    fig = plt.figure(figsize=(13.5, 7.6), dpi=110)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1], wspace=0.26, hspace=0.36)
    draw_strips(fig.add_subplot(gs[:, 0]), S, rows); draw_tips_side(fig.add_subplot(gs[0, 1:]), S, rows); draw_tips_top(fig.add_subplot(gs[1, 1]), S, rows); draw_hands(fig.add_subplot(gs[1, 2]), S, rows)
    fig.suptitle(f"{S} {NAME[S]}: robot pose randomization over {len(rows)} draws", fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, BANDS_NOTE, fontsize=7, color="#555")
    p = OUT / f"pose_{S}.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p.name


def fig_single(S, rows, bg, ext, fan_path):
    """One figure per scene (user 2026-11-03: "provide a single figure in its section, not in a whole picture"): where the robot
    starts (map), which way it faces (rose), how it is posed (the built starts overlaid, the fingertip clouds, the joints)."""
    fig = plt.figure(figsize=(16.5, 9.4), dpi=110)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1.0], height_ratios=[1, 1], wspace=0.28, hspace=0.3, left=0.045, right=0.925, top=0.915, bottom=0.075)
    draw_map(fig.add_subplot(gs[:, 0]), S, rows, bg, ext, title=f"location: {len(rows)} drawn starts, arrow = heading (top view)", short=True)
    draw_rose(fig.add_subplot(gs[0, 1], projection="polar"), S, rows)
    axf = fig.add_subplot(gs[0, 2]); axf.axis("off")
    if fan_path and Path(fan_path).exists():
        axf.imshow(imageio.imread(fan_path)); axf.set_title("pose: the built starts overlaid, robot and held plate only\n(solid where they agree, a tint per start where they differ)", fontsize=8.5, loc="left")
    else:
        axf.text(0.5, 0.5, "no builds", ha="center", va="center")
    draw_tips_side(fig.add_subplot(gs[1, 1]), S, rows)
    axs = fig.add_subplot(gs[1, 2]); draw_strips(axs, S, rows, fs=7); axs.yaxis.tick_right()
    fig.suptitle(f"{S} {NAME[S]}: start randomization over {len(rows)} draws: location (left), heading (top middle), robot pose (right and bottom)", fontsize=12, x=0.01, y=0.975, ha="left")
    fig.text(0.01, 0.006, f"SceneSetups.sample draws (lv3 ranges, --slice {S}); the per-episode reset jitter (+-2 cm, +-3 deg) is not shown. " + BANDS_NOTE, fontsize=7, color="#555")
    p = OUT / f"rand_{S}.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig); return p.name


# ------------------------------------------------------------------------------------------------------------ builds
def robot_cam(S, yaw_deg, off=None, dist=None):
    """A camera tied to the pelvis: `off` degrees from straight ahead of the robot (negative = its front-left), looking back at it."""
    pel = d.xpos[PEL]
    off = -40 if off is None else off
    return dict(lookat=[pel[0], pel[1], pel[2] + 0.08], distance=(2.2 if S != "A5" else 2.3) if dist is None else dist, azimuth=yaw_deg + 180 + off, elevation=-18)


def keep_geoms(held):
    """Geom mask for the robot-only renders: the robot, both plates, the pedal bin."""
    k = np.zeros(m.ngeom, bool)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g]); bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        k[g] = int(m.body_rootid[b]) == ROOTB or b in PLATEB.values() or ("bin" in bn or "pedal" in bn)
    return k


def render_only(cam, keep, W=960, H=540):
    """Render with every geom but `keep` invisible, and the mask of the robot (+ held plate) pixels = where the image differs from
    the same render with the robot hidden too."""
    rgba0 = m.geom_rgba.copy()
    m.geom_rgba[~keep, 3] = 0.0
    img = render(cam, W, H)
    robot = np.array([int(m.body_rootid[m.geom_bodyid[g]]) == ROOTB for g in range(m.ngeom)])
    m.geom_rgba[robot, 3] = 0.0
    bg = render(cam, W, H)
    m.geom_rgba[:] = rgba0
    return img, (np.abs(img.astype(int) - bg.astype(int)).max(-1) > 10)


def good_cam(S, yaw_deg, robot_g):
    """The context camera: the first azimuth offset whose low-res frame shows the robot (>= 1.5 % of the pixels) and is not inside
    furniture (mean brightness >= 70): front-left, front-right, left, right, rear-left, rear-right, front, rear."""
    for off in ((-40, 40, -90, 90, -140, 140, 0, 180) if S != "A5" else (-75, 75, -40, 40, -110, 110)):
        cam = robot_cam(S, yaw_deg, off)                                     # the scene view: A5 from the robot's left side, the foot on the pedal in view
        seg = render(cam, 320, 180, seg=True); img = render(cam, 320, 180)
        frac = np.isin(seg[..., 0], robot_g).mean(); bright = img.mean()
        if frac >= 0.015 and bright >= 70:
            return cam, off
    return robot_cam(S, yaw_deg), None


def render(cam, W=960, H=540, seg=False):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    free = mujoco.MjvCamera(); free.type = mujoco.mjtCamera.mjCAMERA_FREE
    free.lookat[:] = cam["lookat"]; free.distance = cam["distance"]; free.azimuth = cam["azimuth"]; free.elevation = cam["elevation"]
    r = mujoco.Renderer(m, H, W)
    if seg: r.enable_segmentation_rendering()
    r.update_scene(d, camera=free, scene_option=NO_SITES); img = r.render().copy(); r.close()
    return img


def builds(S, M):
    setups = SceneSetups("kitchen", mod, task, ranges="lv3", slice=S)
    imgs, onlys, masks, caps, offs, offs_cam = [], [], [], [], [], []
    for k in range(M):
        seed = a.seed + 1000 * int(S[1]) + 500 + k
        s = setups.sample(seed); setups.apply(s); env.reset(); rebind()
        shift = spawn_shift(s) if (s.get("start") or {}).get("spawn_back_m") else None
        _stand_on_floor(senv, agent, None, shift)
        posed = apply_pose(senv, s.get("preset")); footed = foot_on_pedal(senv, s.get("preset")); held = hold_plate(senv, s.get("preset"))
        offs.append((float(d.xpos[PEL][0] - s["robot"]["x"]), float(d.xpos[PEL][1] - s["robot"]["y"])))
        robot_g = np.array([g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == ROOTB])
        cam, off = good_cam(S, s["robot"]["yaw_deg"], robot_g)
        ctx = render(cam)
        only, mask = render_only(robot_cam(S, s["robot"]["yaw_deg"]), keep_geoms(held))
        imgs.append(ctx); onlys.append(only); masks.append(mask); offs_cam.append(off)
        P = s.get("preset") or {}; st = s.get("start") or {}
        cap = [f"({s['robot']['x']:+.2f}, {s['robot']['y']:+.2f}) {s['robot']['yaw_deg']:+.0f} deg"]
        if st.get("variant"): cap.append(f"kind {st['variant']}" + (f" (A1 ep {st['bank_episode']})" if st.get("bank_episode") is not None else ""))
        if st.get("env"): cap.append(f"A1 envelope, legs of ep {st['legs_episode']}, draw {st['tries']}")
        if P.get("plate_in_hand"): cap.append(f"plate {P['plate_in_hand']['plate'][-1].upper()} in the {P['plate_in_hand']['side'][0].upper()} hand")
        if P.get("hands"): cap.append("hands " + " ".join(f"{sd[0].upper()}:{hand_kind(q)}" for sd, q in P["hands"].items()))
        if P.get("foot_on_pedal"): cap.append(f"{P['foot_on_pedal'][0].upper()} foot on the pedal")
        if P.get("waist"): cap.append(f"waist yaw {math.degrees(P['waist'][0]):+.0f}")
        caps.append(", ".join(cap[:2]) + "\n" + ", ".join(cap[2:]))
        print(f"  [{S} build {k}] {SceneSetups.describe(s)[:150]}", flush=True)
    print(f"[{S}] pelvis - setup offset (world vs robot frame): {np.round(np.mean(offs, 0), 3).tolist()} max |.| {np.abs(offs).max():.3f}; context camera offsets {offs_cam}", flush=True)
    # grids: the start in its scene (camera tied to the pelvis, turned away from furniture when needed) and the robot alone
    def grid(images, name, title, crop=None):
        cols = 4; rws = int(math.ceil(M / cols))
        fig, axs = plt.subplots(rws, cols, figsize=(cols * 4.0, rws * 2.75), dpi=100)
        for i, ax in enumerate(np.ravel(axs)):
            ax.axis("off")
            if i < M:
                im = images[i][crop] if crop else images[i]
                ax.imshow(im); ax.set_title(caps[i], fontsize=6.8, loc="left")
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.965)); g = OUT / name; fig.savefig(g, pil_kwargs=dict(quality=86)); plt.close(fig); return g.name
    g = grid(imgs, f"grid_{S}.jpg", f"{S} {NAME[S]}: {M} built starts in the scene (camera 2.2 m from the pelvis, turned to a free side when the default side is inside furniture)")
    Hh, Ww = onlys[0].shape[:2]; crop = (slice(int(Hh * 0.04), int(Hh * 0.98)), slice(int(Ww * 0.22), int(Ww * 0.78)))
    gp = grid(onlys, f"poses_{S}.jpg", f"{S} {NAME[S]}: the same {M} starts, robot, plates and bin only, always from the robot's front-left", crop)
    imgs = onlys
    # fan: robot + held plate pixels of every build over a neutral ground, tinted per build
    H, W = imgs[0].shape[:2]
    base = np.full((H, W, 3), 238.0)
    base[int(H * 0.74):] = 222.0
    tints = plt.get_cmap("tab20")(np.linspace(0, 1, M))[:, :3] * 255
    cover = np.zeros((H, W))
    for i in range(M):
        mk = masks[i]; px = imgs[i][mk].astype(float)
        col = 0.55 * px + 0.45 * tints[i]
        al = 0.5
        base[mk] = (1 - al) * base[mk] + al * col; cover[mk] += 1
    fan = np.clip(base, 0, 255).astype(np.uint8)
    fan = fan[int(H * 0.02):int(H * 0.98), int(W * 0.2):int(W * 0.8)]
    imageio.imwrite(OUT / f"fan_raw_{S}.jpg", fan, quality=88)                                   # the bare composite, for the single figure
    fig, ax = plt.subplots(figsize=(7.0, 6.1), dpi=100); ax.imshow(fan); ax.axis("off")
    ax.set_title(f"{S}: the {M} starts overlaid (robot and held plate only, one tint each; where they agree the colour is solid)", fontsize=9, loc="left")
    fig.tight_layout(); f = OUT / f"fan_{S}.jpg"; fig.savefig(f, pil_kwargs=dict(quality=88)); plt.close(fig)
    return g, gp, f.name, caps


# ------------------------------------------------------------------------------------------------------------ main
t0 = time.time()
img, col, row = ortho_top()
bg, ext = resample(img, col, row, -1.05, 1.45, -1.2, 1.7)
imageio.imwrite(OUT / "top_ortho.jpg", bg, quality=85)
print(f"background {bg.shape} in {time.time() - t0:.0f} s", flush=True)

coverage = {}
OLD = json.loads((OUT / "coverage.json").read_text()) if (OUT / "coverage.json").exists() else {}   # --skip-builds keeps the earlier builds' figures
ALL = {}
for S in SL:
    t1 = time.time()
    rows = draws(S, a.n); ALL[S] = rows
    (OUT / f"draws_{S}.json").write_text(json.dumps(rows))
    built = builds(S, a.m) if not a.skip_builds else None
    fm = fig_map(S, rows, bg, ext); fp = fig_pose(S, rows); fs = fig_single(S, rows, bg, ext, OUT / f"fan_raw_{S}.jpg")
    xs = np.array([r["robot"]["x"] for r in rows]); ys = np.array([r["robot"]["y"] for r in rows]); yw = np.array([r["robot"]["yaw_deg"] for r in rows])
    from scipy.spatial import cKDTree
    nn = cKDTree(np.column_stack([xs, ys])).query(np.column_stack([xs, ys]), k=2)[0][:, 1]
    # how much of each drawn joint's mechanical range the draws span
    span = {}
    for label, jn, side, k in ROWS:
        v = np.array([r["joints"][jn] for r in rows]); lo, hi = JRANGE[jn]
        span[label] = round(float((v.max() - v.min()) / (hi - lo)), 3)
    tipsx = {s: np.array([r["fk"][s]["tip"] for r in rows]) for s in ("left", "right")}
    kinds = {}
    for s in ("left", "right"):
        ks = [r["hands"].get(s) or ("held" if r.get("held") and r["held"]["side"] == s else ("pose" if r.get("pose") else "default open")) for r in rows]
        kinds[s] = {k: ks.count(k) for k in sorted(set(ks))}
    coverage[S] = dict(n=len(rows), x=(round(float(xs.min()), 2), round(float(xs.max()), 2)), y=(round(float(ys.min()), 2), round(float(ys.max()), 2)),
                       yaw=(round(float(yw.min()), 1), round(float(yw.max()), 1)), footprint_m2=round(float((xs.max() - xs.min()) * (ys.max() - ys.min())), 3),
                       std_xy_cm=(round(float(xs.std()) * 100, 1), round(float(ys.std()) * 100, 1)), std_yaw_deg=round(float(yw.std()), 1),
                       nn_cm=round(float(np.median(nn)) * 100, 1),
                       plate_types=dict(a=len({r["plate"] for r in rows}), b=len({r["plate_b"] for r in rows if r.get("plate_b") is not None})),
                       spots={p: dict(x=(round(min(r["spots"][p][0] for r in rows if p in r["spots"]), 2), round(max(r["spots"][p][0] for r in rows if p in r["spots"]), 2)),
                                      y=(round(min(r["spots"][p][1] for r in rows if p in r["spots"]), 2), round(max(r["spots"][p][1] for r in rows if p in r["spots"]), 2)), n=sum(1 for r in rows if p in r["spots"]))
                              for p in ("plate_a", "plate_b") if any(p in r["spots"] for r in rows)},
                       joint_span_frac=span, hands=kinds,
                       tips={s: dict(ahead=(round(float(tipsx[s][:, 0].min()), 2), round(float(tipsx[s][:, 0].max()), 2)), up=(round(float(tipsx[s][:, 2].min()), 2), round(float(tipsx[s][:, 2].max()), 2)),
                                     side=(round(float(tipsx[s][:, 1].min()), 2), round(float(tipsx[s][:, 1].max()), 2)),
                                     spread_cm=round(float(np.sqrt(((tipsx[s] - tipsx[s].mean(0)) ** 2).sum(1)).mean()) * 100, 1)) for s in ("left", "right")},
                       variants={v: sum(1 for r in rows if r.get("variant") == v) for v in {r.get("variant") for r in rows} if v},
                       held={f"{r['held']['plate']}/{r['held']['side']}": 0 for r in rows if r.get("held")},
                       foot={f: sum(1 for r in rows if r.get("foot") == f) for f in ("left", "right")} if S == "A5" else None,
                       figs=dict(map=fm, pose=fp, single=fs))
    for r in rows:
        if r.get("held"): coverage[S]["held"][f"{r['held']['plate']}/{r['held']['side']}"] += 1
    if built:
        g, gp, f, caps = built; coverage[S]["figs"].update(grid=g, poses=gp, fan=f); coverage[S]["build_caps"] = caps
    elif OLD.get(S):
        coverage[S]["figs"].update({k: v for k, v in OLD[S]["figs"].items() if k in ("grid", "poses", "fan")}); coverage[S]["build_caps"] = OLD[S].get("build_caps", [])
    print(f"[{S}] done in {time.time() - t1:.0f} s: {json.dumps({k: v for k, v in coverage[S].items() if k in ('x', 'y', 'yaw', 'footprint_m2', 'nn_cm', 'plate_types', 'variants', 'held', 'foot')})}", flush=True)

# all scenes on one map
fig, ax = plt.subplots(figsize=(9, 9.5), dpi=110); ax.imshow(bg, extent=ext, alpha=0.9)
for S in SL:
    rows = ALL[S][:150]
    for r in rows:
        x, yy, yaw = r["robot"]["x"], r["robot"]["y"], math.radians(r["robot"]["yaw_deg"])
        ax.add_patch(FancyArrow(x, yy, 0.09 * math.cos(yaw), 0.09 * math.sin(yaw), width=0.004, head_width=0.028, head_length=0.028, fc=COLOR[S], ec=COLOR[S], alpha=0.55, lw=0.3))
for S in SL:
    e = EP1.get(S); x, yy, yaw = e[0], e[1], math.radians(e[2])
    ax.add_patch(Circle((x, yy), 0.045, fc="none", ec="k", lw=1.6, zorder=6)); ax.add_patch(FancyArrow(x, yy, 0.14 * math.cos(yaw), 0.14 * math.sin(yaw), width=0.005, head_width=0.04, head_length=0.04, fc="k", ec="k", zorder=6))
ax.add_patch(Rectangle((0.61, -0.42), 0.15, 0.34, fill=False, ec=PL_A, lw=1.2, ls=":")); ax.add_patch(Rectangle((0.61, -0.68), 0.15, 0.20, fill=False, ec=PL_B, lw=1.2, ls=":"))
ax.set_xlim(-1.0, 1.4); ax.set_ylim(-1.15, 1.65); ax.set_aspect("equal"); ax.grid(True, color="w", alpha=0.35, lw=0.6)
ax.set_xlabel("x (m), robot frame of the full task"); ax.set_ylabel("y (m)"); ax.tick_params(labelsize=8)
ax.legend(handles=[plt.Line2D([], [], color=COLOR[S], lw=3, label=f"{S} {NAME[S]}") for S in SL] + [plt.Line2D([], [], color="k", marker="o", mfc="none", lw=0, ms=9, label="ep1's stances (3.2, 29.6, 51.5, 53.0 s)")], loc="lower left", fontsize=8, framealpha=0.9)
ax.set_title("Where the five scenes start: 150 drawn starts each (arrow = heading); dotted = plate A / plate B spot lanes", fontsize=10, loc="left")
fig.savefig(OUT / "map_all.png", bbox_inches="tight"); plt.close(fig)
coverage["_meta"] = dict(made=time.strftime("%Y-%m-%d %H:%M"), n=a.n, m=a.m, seed=a.seed, worktop_top=WORKTOP, joint_ranges=JRANGE)
(OUT / "coverage.json").write_text(json.dumps(coverage, indent=1))
print(f"all done in {time.time() - t0:.0f} s -> {OUT}")
