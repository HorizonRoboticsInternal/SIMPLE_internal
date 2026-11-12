"""DexPilot-style vector retargeting for the Revo 2's 6 motors (2026-11-03), after dex-retargeting's DexPilotOptimizer
(github.com/dexsuite/dex-retargeting, arXiv 1910.03135) and AnyDexRetarget's pinch contact term. The bone-angle map in
revo2_retarget.py reads each finger on its own and needs the PINCH_KNOTS tables to make pinches meet; this solves all six
motors at once so that the robot's fingertip geometry matches the hand's:

  vectors   S1: thumb pad -> each finger pad (4); S2: finger pad -> finger pad (6); base: middle knuckle -> each pad (5).
            Human: the 21 MediaPipe keypoints (tips 4 8 12 16 20, knuckle 9), in the hand's palm frame, scaled by the
            robot / human middle-finger length. Robot: the pad centres (touch sites) from MuJoCo forward kinematics, in
            the robot palm frame (the same across / along / normal construction as bench.Rig.palm_frame).
  projection  a thumb-finger pair whose HUMAN distance drops under PROJECT_DIST (3 cm) is "projected": its reference
            vector becomes (almost) zero length, so the solver closes the robot pads on each other; it is released above
            ESCAPE_DIST (5 cm) (hysteresis). A finger-finger pair is projected when both fingers are pinched with the
            thumb and they are within 3 cm: its reference length is ETA2 (3 cm), keeping them apart. Projected pairs are
            weighted 200 (S1) / 400 (S2), the base vectors 15, the rest 1 (dex-retargeting's values).
  solver    bounded least squares (scipy trf, Huber delta 0.03 m, 2-point Jacobian) over the 6 motors with a 4e-3 pull
            toward the previous solution, warm-started from it. MuJoCo kinematics of the whole robot costs 2 us, a solve
            about a millisecond.

What this cannot do: make the thumb reach where the hand cannot (the Revo 2 thumb stops 13 mm short of the pinky pad);
there it returns the pose of closest approach, which is what the weights ask for.

Self-test (synthetic hand, no hardware): .venv/bin/python scripts/wuji_glove_pilot/dexpilot.py --selftest
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from revo2_retarget import MOTOR_UPPER, motors_to_joints  # noqa: E402
from simple.robots.g1_sonic_revo2 import OPEN_Q, G1SonicRevo2  # noqa: E402

ROBOT_XML = ROOT / "data" / G1SonicRevo2.mjcf_path
JOINTS = ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal", "middle_proximal",
          "middle_distal", "ring_proximal", "ring_distal", "pinky_proximal", "pinky_distal"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HUMAN_TIPS = [4, 8, 12, 16, 20]
PROJECT_DIST, ESCAPE_DIST = 0.03, 0.05         # human thumb-finger distance (m): pair projected below / released above
ETA1, ETA2 = 1e-4, 3e-2                        # reference length (m) of a projected thumb-finger / finger-finger pair
W_PROJ_S1, W_PROJ_S2, W_BASE = 200.0, 400.0, 15.0
HUBER_M, NORM_DELTA = 0.03, 4e-3
# vector list as (origin, task) over points [knuckle, thumb, index, middle, ring, pinky]; vec = p[task] - p[origin]
PAIRS_S1 = [(j, 1) for j in range(2, 6)]
PAIRS_S2 = [(j, i) for i in range(2, 5) for j in range(i + 1, 6)]
PAIRS_BASE = [(0, i) for i in range(1, 6)]
PAIRS = PAIRS_S1 + PAIRS_S2 + PAIRS_BASE
_MODELS: dict[str, mujoco.MjModel] = {}


def _frame(across, fwd):
    f = fwd / np.linalg.norm(fwd)
    a = across - np.dot(across, f) * f
    a /= np.linalg.norm(a)
    return np.stack([a, f, np.cross(a, f)], axis=1)


def _model(xml: Path) -> mujoco.MjModel:
    key = str(xml)
    if key not in _MODELS:
        _MODELS[key] = mujoco.MjModel.from_xml_path(key)
    return _MODELS[key]


class DexPilotSolver:
    def __init__(self, side: str, xml: Path | str = ROBOT_XML, project_dist: float = PROJECT_DIST, escape_dist: float = ESCAPE_DIST,
                 huber_m: float = HUBER_M, norm_delta: float = NORM_DELTA) -> None:
        self.side, self.project_dist, self.escape_dist = side, float(project_dist), float(escape_dist)
        self.huber_m, self.norm_delta = float(huber_m), float(norm_delta)
        m = self.m = _model(Path(xml))
        self.d = mujoco.MjData(m)
        self.qa = np.array([m.jnt_qposadr[m.joint(f"{side}_hand_{n}_joint").id] for n in JOINTS])
        self.base = m.body(f"{side}_hand_base_link").id
        self.knuckle = m.body(f"{side}_hand_middle_proximal_link").id
        self.sites = np.array([m.site(f"{side}_hand_{f}_touch_site").id for f in FINGERS])
        self.d.qpos[self.qa] = 0.0                              # fingers straight: the palm frame in the base link, the finger length
        mujoco.mj_kinematics(m, self.d)
        x = lambda n: self.d.xpos[m.body(f"{side}_hand_{n}").id]          # noqa: E731
        R = _frame(x("index_proximal_link") - x("pinky_proximal_link"), x("middle_tip_link") - x("middle_proximal_link"))
        self.R_local = self.d.xmat[self.base].reshape(3, 3).T @ R
        self.finger_len = float(np.linalg.norm(self.d.site_xpos[self.sites[2]] - x("middle_proximal_link")))
        self.projected = np.zeros(len(PAIRS_S1) + len(PAIRS_S2), dtype=bool)
        self.last = np.asarray(OPEN_Q, dtype=np.float64)[[0, 1, 3, 5, 7, 9]]
        self.stats = dict(solves=0, nfev=0, seconds=0.0)
        self._w = np.ones(len(PAIRS))

    def reset(self) -> None:
        self.projected[:] = False
        self.last = np.asarray(OPEN_Q, dtype=np.float64)[[0, 1, 3, 5, 7, 9]]

    # ------------------------------------------------------------------ geometry
    def robot_points(self, motors: np.ndarray) -> np.ndarray:
        """[knuckle, 5 pad centres] in the robot palm frame for 6 motor values (kinematics only)."""
        self.d.qpos[self.qa] = motors_to_joints(motors)
        mujoco.mj_kinematics(self.m, self.d)
        Rw = self.d.xmat[self.base].reshape(3, 3) @ self.R_local
        pts = np.concatenate([self.d.xpos[self.knuckle][None], self.d.site_xpos[self.sites]], axis=0)
        return (pts - self.d.xpos[self.base]) @ Rw

    @staticmethod
    def human_points(kp: np.ndarray) -> tuple[np.ndarray, float]:
        """[knuckle, 5 tips] in the hand's palm frame, and the middle-finger length (bone lengths, pose-invariant)."""
        kp = np.asarray(kp, dtype=np.float64).reshape(21, 3)
        Rh = _frame(kp[5] - kp[17], kp[9] - kp[0])
        pts = np.concatenate([kp[9][None], kp[HUMAN_TIPS]], axis=0)
        length = float(sum(np.linalg.norm(kp[i + 1] - kp[i]) for i in (9, 10, 11)))
        return (pts - kp[0]) @ Rh, length

    # ------------------------------------------------------------------ solve
    def solve(self, kp: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        hp, hlen = self.human_points(kp)
        scale = self.finger_len / max(hlen, 1e-6)
        hv = np.array([hp[t] - hp[o] for o, t in PAIRS])                      # human vectors, metres
        hd = np.linalg.norm(hv, axis=1)
        n1, n2 = len(PAIRS_S1), len(PAIRS_S2)
        p1 = self.projected[:n1]
        p1[hd[:n1] < self.project_dist] = True
        p1[hd[:n1] > self.escape_dist] = False
        for k, (j, i) in enumerate(PAIRS_S2):                                 # fingers i and j both pinched, and close to each other
            self.projected[n1 + k] = p1[i - 2] and p1[j - 2] and hd[n1 + k] <= 0.03
        ref = hv * scale
        unit = hv / (hd[:, None] + 1e-6)
        ref[:n1][p1] = unit[:n1][p1] * ETA1
        ref[n1:n1 + n2][self.projected[n1:]] = unit[n1:n1 + n2][self.projected[n1:]] * ETA2
        w = np.concatenate([np.where(p1, W_PROJ_S1, 1.0), np.where(self.projected[n1:], W_PROJ_S2, 1.0), np.full(len(PAIRS_BASE), W_BASE)])
        sw = np.sqrt(w / len(PAIRS))[:, None]
        last, reg = self.last.copy(), np.sqrt(self.norm_delta)

        def residual(x):
            rp = self.robot_points(x)
            rv = np.array([rp[t] - rp[o] for o, t in PAIRS])
            return np.concatenate([(sw * (rv - ref)).ravel(), reg * (x - last)])

        res = least_squares(residual, last, bounds=(np.zeros(6), MOTOR_UPPER), method="trf", loss="huber", f_scale=self.huber_m,
                            jac="2-point", xtol=1e-4, ftol=1e-6, max_nfev=60)
        self.last = np.clip(res.x, 0.0, MOTOR_UPPER)
        self.stats["solves"] += 1; self.stats["nfev"] += int(res.nfev); self.stats["seconds"] += time.perf_counter() - t0
        d_thumb = hd[:n1]
        nearest = int(np.argmin(d_thumb))
        info = dict(pinch_w=1.0 if p1[nearest] else float(np.clip((self.escape_dist - d_thumb[nearest]) / (self.escape_dist - self.project_dist), 0.0, 1.0)),
                    pinch_finger=FINGERS[1 + nearest], projected=self.projected.copy(), cost=float(res.cost), scale=scale)
        return self.last.copy(), info


# --------------------------------------------------------------------------------------------------------------- selftest
def _selftest() -> None:
    """The synthetic hand's poses, both sides. A flat hand gives thumb rotation 0 and open fingers but a thumb bend of about
    0.7: the robot's straight thumb points 10 cm out to the side (hitch-hiker), so the pad of a relaxed flat thumb, near
    the index base, is matched by bending it: that is the vector match, not an error."""
    from wuji_glove import CURL, FIST, FLAT, OPPOSE, PINCH, human_keypoints
    checks = (("flat", FLAT, lambda q: q[0] < 0.3 and q[2:].max() < 0.25), ("fist", FIST, lambda q: q[2:].min() > 1.0 and q[0] > 1.0),
              ("oppose", OPPOSE, lambda q: q[0] > 1.2 and q[2:].max() < 0.4), ("pinch", PINCH, lambda q: q[2] > 0.5),
              ("index curl", CURL(0), lambda q: q[2] > 1.0 and q[3:].max() < 0.25), ("pinky curl", CURL(3), lambda q: q[5] > 1.0 and q[2:5].max() < 0.25))
    for side in ("right", "left"):
        s = DexPilotSolver(side)
        for name, pose, check in checks:
            s.reset()
            for _ in range(3):
                q, info = s.solve(human_keypoints(pose, side))
            rp = s.robot_points(q)
            gap = np.linalg.norm(rp[1] - rp[2]) * 1000
            print(f"  {side:5s} {name:10s} motors {np.round(q, 2).tolist()}  thumb-index pad centres {gap:5.1f} mm  projected {info['projected'][:4].astype(int).tolist()}  {'OK' if check(q) else 'FAIL'}")
            assert check(q), f"{side} {name}"
            if name == "pinch":
                assert gap < 5.0 and info["projected"][0], "a pinch must close the pads"
    n = 200; t0 = time.perf_counter()
    for i in range(n):
        s.solve(human_keypoints(PINCH if i % 2 else FLAT, "left"))
    print(f"  {n} solves: {(time.perf_counter() - t0) / n * 1e3:.2f} ms each, {s.stats['nfev'] / s.stats['solves']:.0f} evaluations per solve")
    print("[dexpilot] selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
