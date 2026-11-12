#!/usr/bin/env python
"""Wuji Glove pilot, stage 1 (2026-10-27): glove -> Revo 2 retarget -> the sim Revo 2 hand on its own (no PICO, no HoloMotion).

The rot0 Revo 2 robot (data/robots/g1_rot0_teleop_revo2) with its pelvis fixed and both hands raised in front of the
chest, palms toward the viewer, fingers up (the pose of an operator showing both palms to the headset). The
hands are driven exactly as in the teleop: per-joint PD with G1SonicRevo2.HAND_KP / HAND_KD, torques clipped at the
actuator ctrlrange, 5 ms physics steps, 4 per 20 ms tick, the teleop's solver options. Every tick takes the glove's newest
frame. The glove's skeleton is drawn as an orange ghost hand (scaled to the robot hand) next to the robot hand. The viewer
opens zoomed on the shown hand (--view hand, see VIEWS; --view free for MuJoCo's default camera).

Writes to --out: bench.mp4 (palm view | oblique view, motor bars: target outline, actual fill), metrics.json (glove rate,
dropouts, retarget time, per-motor tracking error and lag; with --glove synthetic also PASS/FAIL checks per scripted pose),
glove_log.bin (+ .json; replayable with --glove replay:<path>), timeline.npz.

  # no glove: the scripted synthetic hand, 2 cycles of 16 s
  MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove synthetic --seconds 32
  # with the glove (host NIC on 192.168.1.x/24): calibrate once, then live in the MuJoCo viewer, or headless with a video
  .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove wuji --side right --calibrate
  .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove wuji --side right --viewer
  MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove wuji --side right --seconds 60
  # a recorded glove session again (e.g. to compare retarget settings)
  MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove replay:<out>/glove_log.bin --no-pinch-blend
  # no glove: the PICO headset's hand tracking (pico_hand.py; XRoboToolkit PC service + hand tracking on in the headset app)
  .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove pico --side right --calibrate     # -> calib_right_pico.json
  .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove pico --side right --viewer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
if "--viewer" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from revo2_retarget import ACTIVE, FEATURES, MOTOR_UPPER, MOTORS, Calibration, GloveHand, Revo2Retargeter, features  # noqa: E402
from simple.robots.g1_sonic_revo2 import OPEN_Q, G1SonicRevo2  # noqa: E402
from wuji_glove import CYCLE_S, DROPOUT, GloveLog, make_source, script_pose  # noqa: E402

ROBOT_XML = ROOT / "data" / G1SonicRevo2.mjcf_path
CALIB_DIR = ROOT / "data" / "wuji_glove_pilot"
TICK, SUBSTEPS = 0.02, 4
# hands raised in front of the chest, palms toward the viewer, fingers up, 0.36 m apart (2026-11-03, the pose of an
# operator showing both palms to the headset): solved per arm by Powell over the 7 arm joints for palm normal = the
# robot's forward, fingers = up, hand base 0.32 m ahead and 0.30 m above the pelvis; mirror-symmetric, no body contact
ARM_POSE = {"right_shoulder_pitch_joint": -0.87, "right_shoulder_roll_joint": -0.23, "right_shoulder_yaw_joint": -0.19, "right_elbow_joint": 0.55,
            "right_wrist_roll_joint": -1.43, "right_wrist_pitch_joint": -0.03, "right_wrist_yaw_joint": -1.54,
            "left_shoulder_pitch_joint": -0.87, "left_shoulder_roll_joint": 0.23, "left_shoulder_yaw_joint": 0.19, "left_elbow_joint": 0.55,
            "left_wrist_roll_joint": 1.43, "left_wrist_pitch_joint": -0.03, "left_wrist_yaw_joint": 1.54}
# ghost hand placement in the palm frame (across toward the thumb, along the fingers, out of the palm): beside the hand on
# the pinky side, which in this pose is the outside of each hand
GHOST_BESIDE = (-0.13, 0.0, 0.0)
# viewer / video camera directions from the hand to the camera, in the palm frame (across toward the thumb side, along the
# fingers, out of the palm). "hand" = the palm from 20 deg toward the thumb and 15 deg toward the fingertips: all five
# fingers and the thumb's arc visible, the curl readable, the ghost hand beside the robot hand (chosen 2026-11-03 from 12
# candidates rendered in the open / fist / thumb-index / thumb-middle poses of a recorded session)
VIEWS = {"palm": (0.0, 0.0, 1.0), "oblique": (0.5, 0.6, 0.6), "hand": (0.35, 0.25, 0.9)}
HAND_VIEW_DISTANCE, HANDS_VIEW_DISTANCE = 0.30, 0.62          # viewer distance (m): one hand / both hands with their ghosts beside them
FINGERS5 = ["thumb", "index", "middle", "ring", "pinky"]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
         (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
# synthetic self-checks: (name, script time to sample (s), check(fractions, robot pinch gap mm) -> bool, what it means)
CHECKS = [("flat", 1.2, lambda f, g: f.max() < 0.1, "all six motors < 10 % closed"),
          ("fist", 4.3, lambda f, g: f[1:].min() > 0.85, "thumb bend and the four fingers > 85 % closed"),
          ("oppose", 9.3, lambda f, g: f[0] > 0.9 and f[2:].max() < 0.1, "thumb rotation > 90 %, fingers open"),
          ("pinch", 12.3, lambda f, g: g < 5.0, "index follows the hand, thumb from PINCH_KNOTS (blend 1.0); robot thumb and index pads within 5 mm")]


# ------------------------------------------------------------------------------------------------------------- model
def build_model(xml: Path):
    spec = mujoco.MjSpec.from_file(str(xml))
    for j in list(spec.body("pelvis").joints):              # fixed base: the bench only looks at the hands
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)
    spec.add_texture(name="bench_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.94, 0.95], rgb2=[0.72, 0.75, 0.78], width=256, height=1536)
    o = spec.option                                          # the teleop's solver settings
    o.timestep, o.integrator, o.cone, o.impratio, o.noslip_iterations = 0.005, mujoco.mjtIntegrator.mjINT_IMPLICITFAST, mujoco.mjtCone.mjCONE_ELLIPTIC, 10.0, 2
    m = spec.compile()
    if "com_marker" in [m.site(i).name for i in range(m.nsite)]:
        m.site_rgba[m.site("com_marker").id, 3] = 0.0           # the robot file's red CoM ball sits between the hands: hide it
    return m, mujoco.MjData(m)


class Rig:
    """Joint / actuator indices and the PD of the bench robot."""

    def __init__(self, m, d) -> None:
        self.m, self.d = m, d
        jid = lambda n: m.joint(n).id                         # noqa: E731
        self.hand = {s: [jid(f"{s}_hand_{f}_joint") for f in
                         ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal", "middle_proximal",
                          "middle_distal", "ring_proximal", "ring_distal", "pinky_proximal", "pinky_distal"]] for s in ("left", "right")}
        self.qa = {s: np.array([m.jnt_qposadr[j] for j in js]) for s, js in self.hand.items()}
        self.va = {s: np.array([m.jnt_dofadr[j] for j in js]) for s, js in self.hand.items()}
        act = {m.actuator_trnid[a, 0]: a for a in range(m.nu)}
        self.ha = {s: np.array([act[j] for j in js]) for s, js in self.hand.items()}
        hand_j = set(sum(self.hand.values(), []))
        self.body_j = [j for j in range(m.njnt) if j not in hand_j and j in act]
        self.bq = np.array([m.jnt_qposadr[j] for j in self.body_j]); self.bv = np.array([m.jnt_dofadr[j] for j in self.body_j])
        self.ba = np.array([act[j] for j in self.body_j])
        self.b0 = np.array([ARM_POSE.get(m.joint(j).name, 0.0) for j in self.body_j])
        self.lim = m.actuator_ctrlrange[:, 1].copy()
        d.qpos[self.bq] = self.b0
        mujoco.mj_forward(m, d)                                 # hands at 0 (fingers straight): the palm frame in each hand base
        self.forward = d.xmat[m.body("pelvis").id].reshape(3, 3)[:, 0].copy()   # the robot's forward (fixed base)
        self._palm = {}
        for s in ("left", "right"):
            x = lambda n: d.xpos[m.body(f"{s}_hand_{n}").id].copy()   # noqa: E731
            Rb = d.xmat[m.body(f"{s}_hand_base_link").id].reshape(3, 3)
            R = _frame(x("index_proximal_link") - x("pinky_proximal_link"), x("middle_tip_link") - x("middle_proximal_link"))
            self._palm[s] = (Rb.T @ R, float(np.linalg.norm(x("middle_proximal_link") - x("base_link"))))
            d.qpos[self.qa[s]] = OPEN_Q
        mujoco.mj_forward(m, d)

    def step(self, targets: dict) -> None:
        m, d = self.m, self.d
        for _ in range(SUBSTEPS):
            d.qpos[self.bq], d.qvel[self.bv] = self.b0, 0.0    # the arm is held kinematically (a soft PD sagged 15 cm); the hands are physics
            for s, q in targets.items():                      # = G1SonicRevo2._apply_hand_pd
                d.ctrl[self.ha[s]] = G1SonicRevo2.HAND_KP * (q - d.qpos[self.qa[s]]) - G1SonicRevo2.HAND_KD * d.qvel[self.va[s]]
            d.ctrl[:] = np.clip(d.ctrl, -self.lim, self.lim)
            mujoco.mj_step(m, d)

    def motors(self, s: str) -> np.ndarray:
        return self.d.qpos[self.qa[s]][ACTIVE].copy()

    def pinch_gap_mm(self, s: str, finger: str = "index", contact: str = "pad") -> float:
        """Thumb-to-finger distance (mm): "pad" = the two touch pads, "any" = any thumb distal / tip geom to any geom of the finger."""
        m = self.m
        if contact == "pad":
            pairs = [(m.geom(f"{s}_hand_thumb_touch_collision").id, m.geom(f"{s}_hand_{finger}_touch_collision").id)]
        else:
            col = lambda p: [g for g in range(m.ngeom) if m.geom(g).name.startswith(p) and (m.geom_contype[g] | m.geom_conaffinity[g])]   # noqa: E731
            thumb = col(f"{s}_hand_thumb_distal") + col(f"{s}_hand_thumb_tip") + col(f"{s}_hand_thumb_touch")
            pairs = [(a, b) for a in thumb for b in col(f"{s}_hand_{finger}")]
        return 1e3 * min(float(mujoco.mj_geomDistance(m, self.d, a, b, 0.2, None)) for a, b in pairs)

    def palm_frame(self, s: str):
        """Robot hand palm frame: origin (hand base), rotation [across, fwd, normal] (from the straight fingers), base -> knuckle length."""
        b = self.m.body(f"{s}_hand_base_link").id
        R_local, L = self._palm[s]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3) @ R_local, L


def _frame(across, fwd):
    f = fwd / np.linalg.norm(fwd)
    a = across - np.dot(across, f) * f
    a /= np.linalg.norm(a)
    return np.stack([a, f, np.cross(a, f)], axis=1)


def ghost_points(kp: np.ndarray, rig: Rig, s: str, offset=GHOST_BESIDE) -> np.ndarray:
    """The glove keypoints mapped into the robot palm frame, scaled to the robot hand, its wrist at `offset` (palm frame)."""
    kp = np.asarray(kp, dtype=np.float64)
    Rh, Lh = _frame(kp[5] - kp[17], kp[9] - kp[0]), np.linalg.norm(kp[9] - kp[0])
    o, Rr, Lr = rig.palm_frame(s)
    return o + Rr @ np.asarray(offset, dtype=np.float64) + (Rr @ Rh.T @ (kp - kp[0]).T).T * (Lr / max(Lh, 1e-6))


def add_ghost(scn, pts: np.ndarray, rgba=(1.0, 0.55, 0.1, 1.0)) -> None:
    for a, b in BONES:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0035, pts[a], pts[b])
        scn.ngeom += 1


def camera(rig: Rig, s: str, view: str, look, distance: float = 0.42):
    _, R, _ = rig.palm_frame(s)                              # columns: across (toward the index), fwd, out of the palm
    p = R @ np.asarray(VIEWS[view], dtype=np.float64)
    p = p / np.linalg.norm(p)
    cam = mujoco.MjvCamera()
    cam.type, cam.lookat[:], cam.distance = mujoco.mjtCamera.mjCAMERA_FREE, look, distance
    cam.azimuth, cam.elevation = np.degrees(np.arctan2(-p[1], -p[0])), np.degrees(np.arcsin(np.clip(-p[2], -1, 1)))
    return cam


def hand_look(rig: Rig, s: str) -> np.ndarray:
    """Point between the robot hand's centre (9 cm along the fingers) and the ghost hand beside it: the viewer's look-at."""
    o, R, _ = rig.palm_frame(s)
    return o + 0.09 * R[:, 1] - 0.065 * R[:, 0]


def bars(width: int, tgt: np.ndarray, act: np.ndarray, label: str):
    import cv2
    img = np.full((120, width, 3), 245, np.uint8)
    w = (width - 40) // 6
    cv2.putText(img, label, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    for i in range(6):
        x0, ft, fa = 20 + i * w, tgt[i] / MOTOR_UPPER[i], act[i] / MOTOR_UPPER[i]
        cv2.rectangle(img, (x0 + 10, 28), (x0 + w - 10, 46), (215, 215, 215), -1)
        cv2.rectangle(img, (x0 + 10, 28), (x0 + 10 + int((w - 20) * np.clip(fa, 0, 1)), 46), (90, 130, 60), -1)
        xt = x0 + 10 + int((w - 20) * np.clip(ft, 0, 1))
        cv2.line(img, (xt, 24), (xt, 50), (20, 110, 230), 2)
        cv2.putText(img, MOTORS[i], (x0 + 10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"{tgt[i]:.2f} / {act[i]:.2f}", (x0 + 10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "blue line = target, green = actual (rad)", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------------------------------------------- calibration
POSES = [("flat", "OPEN: fingers straight and together, thumb resting beside the index finger"),
         ("fist", "FIST: fingers fully curled, thumb bent across the middle of the fingers"),
         ("oppose", "OPPOSE: thumb tip on the base of the little finger, the other fingers straight")]
SYNTH_POSE_T = {"flat": 0.2, "fist": 3.3, "oppose": 8.7}          # synthetic: the script's hold windows


def calib_path(a, side: str) -> Path:
    """--calib, else per source: wuji and replay share calib_<side>.json, pico has its own calib_<side>_pico.json (the
    headset measures the hand differently from the glove), synthetic goes to --out."""
    if a.calib:
        return Path(a.calib)
    if a.glove == "synthetic":
        return Path(a.out) / f"calib_{side}_synthetic.json"
    return CALIB_DIR / (f"calib_{side}_pico.json" if a.glove == "pico" else f"calib_{side}.json")


def calibrate(a, side: str) -> None:
    src = make_source(a.glove, side, a.glove_address, a.glove_sn, dropouts=False)
    synthetic = a.glove == "synthetic"
    poses = {}
    for name, text in POSES:
        print(f"\n[calibrate] {text}", flush=True)
        if synthetic:                                             # jump the script clock to the pose's hold window
            src.latest(0.0)
            frames = [src.latest(float(t)) for t in SYNTH_POSE_T[name] + np.arange(0, 1.0, 1 / 120)]
        else:
            for k in range(3, 0, -1):
                print(f"  hold it ... {k}", flush=True)
                time.sleep(1.0)
            frames, t_end = [], time.monotonic() + 2.0
            while time.monotonic() < t_end:
                frames.append(src.latest(time.monotonic()))
                time.sleep(0.004)
        feats = [features(fr.kp) for fr in frames if fr is not None and float(np.mean(fr.conf)) >= a.min_conf]
        if len(feats) < 10:
            src.close()                                           # the PICO SDK aborts the process if left open at exit
            sys.exit(f"[calibrate] only {len(feats)} usable frames for {name}: is the glove streaming? (pico: the headset must be ON "
                     "and connected to the PC service; it drops off a few minutes after being taken off, and then every client gets zeros)")
        poses[name] = np.median(np.array(feats), axis=0)
        print("  " + "  ".join(f"{k} {v:.2f}" for k, v in zip(FEATURES, poses[name])), flush=True)
    src.close()
    c = Calibration.from_poses(poses, side)
    bad = [FEATURES[i] for i in range(6) if c["closed"][i] - c["open"][i] < 0.2]
    out = calib_path(a, side)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c, indent=1))
    print(f"\n[calibrate] wrote {out}" + (f"\n[calibrate] WARNING small range (< 0.2 rad) for {bad}: redo that pose" if bad else ""), flush=True)


# ---------------------------------------------------------------------------------------------------------------- run
def lag_ms(tgt: np.ndarray, act: np.ndarray, max_lag: int = 25) -> float | None:
    if tgt.std() < 0.05:
        return None
    a, b = tgt - tgt.mean(), act - act.mean()
    c = [np.dot(a[: len(a) - k], b[k:]) for k in range(max_lag + 1)]
    return float(np.argmax(c) * TICK * 1e3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glove", default="synthetic", help="wuji | pico (the headset's hand tracking) | synthetic | replay:<glove_log.bin>")
    ap.add_argument("--side", default="right", choices=["right", "left", "both"])
    ap.add_argument("--glove-address", default="", help="IP[:port] of the glove (default: the factory address of --side; one glove only)")
    ap.add_argument("--glove-sn", default="", help="connect by serial number instead (one glove only)")
    ap.add_argument("--calibrate", action="store_true", help="measure the open / fist / oppose poses and write the calibration JSON, then exit")
    ap.add_argument("--calib", default="", help=f"calibration JSON (default {CALIB_DIR.relative_to(ROOT)}/calib_<side>.json for wuji / replay, the built-in defaults for synthetic)")
    ap.add_argument("--seconds", type=float, default=2 * CYCLE_S, help="sim seconds to run headless (the viewer runs until closed)")
    ap.add_argument("--viewer", action="store_true", help="the interactive MuJoCo viewer in real time instead of the headless video")
    ap.add_argument("--view", default="hand", choices=[*VIEWS, "free"],
                    help="viewer camera: hand (zoomed on the shown hand, from the palm side toward the thumb and fingertips; the mouse still orbits and zooms, the look-at follows the hand) | palm | oblique | free (MuJoCo's default)")
    ap.add_argument("--view-distance", type=float, default=None, help=f"viewer camera distance (m; default {HAND_VIEW_DISTANCE} for one hand, {HANDS_VIEW_DISTANCE} for both)")
    ap.add_argument("--out", default="", help="output dir (default data/wuji_glove_pilot/bench_<time>)")
    ap.add_argument("--robot-mjcf", default=str(ROBOT_XML))
    ap.add_argument("--pinch-blend", action=argparse.BooleanOptionalAction, default=True, help="thumb tip near the index tip -> blend thumb + index toward the tuned pinch pose CLOSE2_Q")
    ap.add_argument("--smooth-hz", type=float, default=10.0, help="low-pass cutoff on the motor targets (0 = off)")
    ap.add_argument("--thumb-rot-min", type=float, default=0.0, help="thumb rotation (rad) at an open human thumb (1.515 = always opposed, like the tuned open grip pose)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="drop glove frames whose mean keypoint confidence is below this")
    ap.add_argument("--noise-mm", type=float, default=0.0, help="synthetic: keypoint noise (m RMS x 1e-3)")
    a = ap.parse_args()
    sides = ["left", "right"] if a.side == "both" else [a.side]
    a.out = a.out or str(CALIB_DIR / time.strftime("bench_%Y%m%d_%H%M%S"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.calibrate:
        for s in sides:
            calibrate(a, s)
        return

    m, d = build_model(Path(a.robot_mjcf))
    rig = Rig(m, d)
    log = GloveLog(Path(a.out) / "glove_log.bin", meta=dict(source=a.glove, sides=sides, bench=True))
    hands = {}
    for s in sides:
        cp = Path("-") if a.glove == "synthetic" and not a.calib else calib_path(a, s)       # synthetic: the built-in defaults
        calib = Calibration.load(cp)
        print(f"[bench] {s}: calibration {calib.path or 'built-in defaults'}", flush=True)
        rt = Revo2Retargeter(calib, pinch_blend=a.pinch_blend, smooth_hz=a.smooth_hz, thumb_rot_min=a.thumb_rot_min, side=s)
        hands[s] = GloveHand(s, make_source(a.glove, s, a.glove_address, a.glove_sn, a.noise_mm), rt, min_conf=a.min_conf, log=log)
    targets = {s: np.asarray(OPEN_Q, dtype=np.float64).copy() for s in ("left", "right")}
    show = sides[-1]
    latest: dict[str, dict] = {}                                # the newest result per gloved side (the viewer draws every ghost)
    T = {k: [] for k in ("t", "target", "actual", "fractions", "features", "pinch_w", "pinch_finger", "held", "fresh", "gap_mm", "touch_mm")}

    def tick(i: int, t: float) -> dict | None:
        res = {}
        for s, h in hands.items():
            r = h.update(t, i)
            if r is not None:
                targets[s] = r["q"]
                res[s] = latest[s] = r
        rig.step(targets)
        r = res.get(show)
        T["t"].append(t); T["target"].append(targets[show][ACTIVE].copy()); T["actual"].append(rig.motors(show))
        T["fractions"].append(r["fractions"] if r else np.zeros(6)); T["features"].append(r["features"] if r else np.zeros(6))
        T["pinch_w"].append(r["pinch_w"] if r else 0.0); T["held"].append(bool(r and r["held"])); T["fresh"].append(bool(r and r["fresh"]))
        finger = r["pinch_finger"] if r else "index"           # the robot's thumb to the finger being pinched: pad to pad, and any contact
        T["pinch_finger"].append(finger); T["gap_mm"].append(rig.pinch_gap_mm(show, finger)); T["touch_mm"].append(rig.pinch_gap_mm(show, finger, "any"))
        return r

    t_start = time.monotonic()
    if a.viewer:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(m, d) as v:
            i = 0
            while v.is_running():
                t = time.monotonic() - t_start
                r = tick(i, t)
                with v.lock():
                    v.user_scn.ngeom = 0
                    two = len(sides) == 2
                    for s_, r_ in latest.items():
                        add_ghost(v.user_scn, ghost_points(r_["frame"].kp, rig, s_, GHOST_BESIDE))
                    if a.view != "free":                       # angle and distance once (the mouse may change them), the look-at every tick
                        if two:                                # both hands: from in front of the robot, a little above, the ghosts beside them
                            look = np.mean([rig.palm_frame(s_)[0] + 0.08 * rig.palm_frame(s_)[1][:, 1] for s_ in sides], axis=0)
                            p = rig.forward + 0.15 * np.array([0.0, 0.0, 1.0])
                        else:                                  # one hand: its palm-frame view, the ghost beside it
                            look, p = hand_look(rig, show), rig.palm_frame(show)[1] @ np.asarray(VIEWS[a.view])
                        if i == 0:
                            p = p / np.linalg.norm(p)
                            dist = a.view_distance or (HANDS_VIEW_DISTANCE if two else HAND_VIEW_DISTANCE)
                            v.cam.type, v.cam.distance = mujoco.mjtCamera.mjCAMERA_FREE, dist
                            v.cam.azimuth, v.cam.elevation = np.degrees(np.arctan2(-p[1], -p[0])), np.degrees(np.arcsin(np.clip(-p[2], -1, 1)))
                        v.cam.lookat[:] = look
                v.sync()
                if i % 50 == 0 and r is not None:
                    print(f"[bench] t {t:6.1f}  target {np.round(targets[show][ACTIVE], 2)}  pinch_w {r['pinch_w']:.2f}{'  HELD' if r['held'] else ''}", flush=True)
                i += 1
                time.sleep(max(0.0, t_start + i * TICK - time.monotonic()))
    else:
        import imageio.v2 as iio
        r_ = mujoco.Renderer(m, 400, 480)
        writer = iio.get_writer(str(Path(a.out) / "bench.mp4"), fps=25, codec="libx264", quality=7, macro_block_size=8)
        live = a.glove in ("wuji", "pico")
        n = int(round(a.seconds / TICK))
        for i in range(n):
            t = (time.monotonic() - t_start) if live else i * TICK
            r = tick(i, t)
            if i % 2 == 0:
                o, R, _ = rig.palm_frame(show)
                look = o + 0.06 * R[:, 1] - 0.065 * R[:, 0]
                views = []
                for view in ("palm", "oblique"):
                    r_.update_scene(d, camera(rig, show, view, look))
                    if r is not None:
                        add_ghost(r_.scene, ghost_points(r["frame"].kp, rig, show))
                    views.append(r_.render())
                img = np.concatenate(views, axis=1)
                name = script_pose(t)[1] if a.glove == "synthetic" else a.glove
                label = f"{show} hand  t {t:5.2f} s  {name}  pinch blend {T['pinch_w'][-1]:.2f}  thumb-{T['pinch_finger'][-1]} gap {T['gap_mm'][-1]:5.1f} mm" + ("  HELD (no glove frame)" if T["held"][-1] else "")
                writer.append_data(np.concatenate([img, bars(img.shape[1], T["target"][-1], T["actual"][-1], label)], axis=0))
            if live:
                time.sleep(max(0.0, t_start + (i + 1) * TICK - time.monotonic()))
        writer.close()
        r_.close()
    for h in hands.values():
        h.source.close()
    log.close()

    # ---- metrics
    A = {k: np.asarray(v) for k, v in T.items()}
    err = np.abs(A["target"] - A["actual"])
    settle = A["t"] > 0.5
    met = dict(glove=a.glove, side=show, seconds=float(A["t"][-1] + TICK) if len(A["t"]) else 0.0, out=a.out,
               settings=dict(pinch_blend=a.pinch_blend, smooth_hz=a.smooth_hz, thumb_rot_min=a.thumb_rot_min, min_conf=a.min_conf,
                             calibration=hands[show].rt.calib.path or "built-in defaults"),
               glove_stats={s: h.summary() for s, h in hands.items()},
               tracking={MOTORS[k]: dict(mean_abs_err_rad=round(float(err[settle, k].mean()), 4), p95_abs_err_rad=round(float(np.percentile(err[settle, k], 95)), 4),
                                         lag_ms=lag_ms(A["target"][settle, k], A["actual"][settle, k])) for k in range(6)})
    if a.glove == "synthetic":
        checks, ph = [], (A["t"] % CYCLE_S)
        for name, ts, fn, what in CHECKS:
            idx = [i for i in range(len(A["t"])) if abs(ph[i] - ts) < TICK / 2 and A["t"][i] > 1.0] or [int(np.argmin(np.abs(A["t"] - ts)))]
            f, gap = A["fractions"][idx].mean(axis=0), float(A["gap_mm"][idx].mean())
            ok = fn(f, gap) and (name != "pinch" or not a.pinch_blend or float(A["pinch_w"][idx].min()) > 0.99)
            checks.append(dict(pose=name, ok=bool(ok), expect=what, pinch_w=round(float(A["pinch_w"][idx].mean()), 3), fractions=np.round(f, 3).tolist(), thumb_index_gap_mm=round(gap, 1),
                               target_rad=np.round(A["target"][idx].mean(axis=0), 3).tolist(), actual_rad=np.round(A["actual"][idx].mean(axis=0), 3).tolist()))
        dz = np.array([DROPOUT[0] + 0.05 <= p < DROPOUT[1] for p in ph])
        jump = float(np.abs(np.diff(A["target"][dz], axis=0)).max()) if dz.sum() > 1 else 0.0
        checks.append(dict(pose="dropout", ok=bool(jump < 1e-9 and A["held"][dz].any()), expect="no target change while no frames arrive; held flag set after 0.25 s",
                           max_target_step_rad=jump, held_ticks=int(A["held"][dz].sum())))
        met["checks"] = checks
        met["all_checks_pass"] = all(c["ok"] for c in checks)
    (Path(a.out) / "metrics.json").write_text(json.dumps(met, indent=1))
    np.savez_compressed(Path(a.out) / "timeline.npz", motors=np.array(MOTORS), features_names=np.array(FEATURES), **A)
    print(json.dumps({k: met[k] for k in ("glove_stats", "tracking")}, indent=1))
    if "checks" in met:
        for c in met["checks"]:
            print(f"[check] {c['pose']:8s} {'PASS' if c['ok'] else 'FAIL'}  {c['expect']}")
    print(f"[bench] wrote {a.out}/(bench.mp4, metrics.json, glove_log.bin, timeline.npz)", flush=True)


if __name__ == "__main__":
    main()
