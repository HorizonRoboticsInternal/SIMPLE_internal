"""Wuji Glove hand skeleton -> BrainCo Revo 2 hand targets (pilot, 2026-10-27).

The glove's `hand_skeleton` stream gives 21 MediaPipe keypoints (metres, in the glove's wrist frame, 120 Hz). The Revo 2 has
6 motors (thumb rotation, thumb bend, index, middle, ring, pinky) that drive its 11 joints; the distal joints follow their
proximal joint (thumb 1.0 x, fingers 1.155 x, simple.robots.g1_sonic_revo2.COUPLING). The retarget reduces the human hand to
6 scalars, each built from angles between bones, so it depends on neither the glove's wrist-frame axes nor the hand side:

  thumb_rot    palmar abduction of the thumb metacarpal (CMC -> MCP) out of the palm plane      -> thumb_metacarpal (rotation)
  thumb_bend   thumb MCP + IP flexion                                                            -> thumb_proximal (bend)
  index..pinky MCP flexion (in the finger's flexion plane, spread removed) + PIP + DIP            -> {finger}_proximal

Each scalar is mapped linearly from its calibrated [open, closed] values onto the motor's travel, clipped to 0..1 of it.
Options: the pinch blend (the human thumb tip near a fingertip: that finger keeps following the hand but closes at least to
the smallest curl where the Revo 2 thumb can reach its pad, and the thumb goes to the pose that meets the finger at its
curl, PINCH_KNOTS from pinch_poses.py --table; so a human pinch closes the Revo 2 pads that the bone-angle map alone does
not bring together, without opening a finger the hand has curled further; the nearest fingertip is taken, the pinky gets
the closest the thumb can come) and a first-order low-pass on the 6 motor targets.

GloveHand ties a glove source (wuji_glove.py) to a retargeter with the hold policy: a frame whose mean keypoint confidence is
below min_conf is dropped, and with no new frame for stale_s the last target is held (the hand never springs open on a
dropout, which would drop a held object).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from simple.robots.g1_sonic_revo2 import COUPLING, REVO2_UPPER

# MediaPipe landmarks: 0 wrist, then CMC/MCP, (MCP/PIP), (IP/DIP), TIP per finger
WRIST = 0
THUMB = (1, 2, 3, 4)
FINGER_LM = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FEATURES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
MOTORS = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
ACTIVE = [0, 1, 3, 5, 7, 9]                    # the motor (proximal) joints in the 11-joint MJCF order
MOTOR_UPPER = REVO2_UPPER[ACTIVE]

# human feature values (rad) for an open hand and for the closed end of each motor; starting values, the bench's
# --calibrate measures them per operator (flat hand / fist / thumb to the little finger's base)
DEFAULT_OPEN = np.array([0.10, 0.25, 0.25, 0.25, 0.25, 0.25])
DEFAULT_CLOSED = np.array([0.85, 1.60, 3.60, 3.60, 3.60, 3.60])
PINCH_NEAR, PINCH_FAR = 0.015, 0.035            # human thumb tip - fingertip distance (m): full blend / no blend
TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
# Where the Revo 2 thumb pad meets each finger's pad (pinch_poses.py --table on g1_rot0_teleop_revo2_v3.xml, 2026-11-03;
# the two hands are not mirror images in that file): per finger, knots of (finger motor, thumb rotation, thumb bend) that
# settle with the pads touching, from the smallest finger curl the thumb can reach to the largest; the thumb is
# interpolated between knots. A fixed thumb pose holds contact across a finger's band (gap <= 1 mm) except for a strongly
# curled middle finger, which needs more thumb bend. Pad-on-pad curls: index 0.69-0.98 (right) / 0.69-1.08 (left), middle
# 0.78-1.47. Ring: pad on pad only around 0.8-1.0, but the thumb TOUCHES the ring (tip or side, any geom) for every curl
# from 0.70 to the limit with rot 1.5 and bend 0.9-1.0, so its knots cover that whole range. Pinky: the thumb never reaches
# it (13 mm at best, thumb fully opposed and bent, pinky at 0.9); the model's ranges equal BrainCo's spec (thumb abduction
# 0-90 deg, thumb flex 0-59 deg, fingers 0-81 deg), so this is the hand, not the model: the pinky knot is the closest
# approach. The tuned grip pose CLOSE2_Q (rot 1.23, bend 0.67, index 0.705) settles 5.7 mm short and is no longer used here.
PINCH_KNOTS = {
    "right": {"index": [(0.69, 1.15, 0.90), (0.98, 1.15, 0.90)],
              "middle": [(0.78, 1.45, 0.90), (0.98, 1.45, 0.90), (1.17, 1.50, 1.00), (1.47, 1.50, 1.00)],
              "ring": [(0.80, 1.50, 0.875), (0.90, 1.50, 0.875), (0.95, 1.50, 1.00)],
              "pinky": [(0.90, 1.50, 1.00), (1.47, 1.50, 1.00)]},
    "left": {"index": [(0.69, 1.15, 0.75), (0.92, 1.15, 0.75), (1.08, 1.20, 0.70)],
             "middle": [(0.78, 1.35, 0.75), (0.98, 1.35, 0.75), (1.17, 1.50, 0.80), (1.47, 1.50, 0.95)],
             "ring": [(0.80, 1.45, 0.80), (0.90, 1.45, 0.80), (0.95, 1.50, 0.975)],
             "pinky": [(1.00, 1.50, 1.00), (1.47, 1.50, 1.00)]},
}
# Ring (2026-11-04): pad on pad is ROBUST only for ring curls 0.80-0.90 (a dozen thumb settings touch there, settled);
# outside it contact exists at isolated settings only, which showed as "sometimes touches". So during a ring pinch the
# ring is kept inside the band: closed to at least 0.80 and not past 0.95, with a thumb pose that touches across the
# whole band (right rot 1.5 bend 0.875, left rot 1.45 bend 0.80). Other fingers have no upper clip.
PINCH_HI = {"ring": 0.95}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0)))


def features(kp: np.ndarray) -> np.ndarray:
    """The 6 human scalars (rad, FEATURES order) from 21 MediaPipe keypoints (21 x 3, any right-handed frame, either hand)."""
    kp = np.asarray(kp, dtype=np.float64).reshape(21, 3)
    across = _unit(kp[5] - kp[17])                                   # little-finger MCP -> index MCP
    fwd = _unit(kp[9] - kp[WRIST])                                   # wrist -> middle MCP
    normal = _unit(np.cross(across, fwd))                            # palm-plane normal (sign unused)
    out = np.zeros(6)
    t = kp[THUMB[1]] - kp[THUMB[0]]
    out[0] = float(np.arcsin(np.clip(abs(np.dot(_unit(t), normal)), 0.0, 1.0)))
    out[1] = _angle(t, kp[THUMB[2]] - kp[THUMB[1]]) + _angle(kp[THUMB[2]] - kp[THUMB[1]], kp[THUMB[3]] - kp[THUMB[2]])
    for i, (mcp, pip, dip, tip) in enumerate(FINGER_LM.values()):
        meta, prox = kp[mcp] - kp[WRIST], kp[pip] - kp[mcp]
        flat = lambda v: v - np.dot(v, across) * across             # noqa: E731  remove the spread (abduction) component
        out[2 + i] = _angle(flat(meta), flat(prox)) + _angle(prox, kp[dip] - kp[pip]) + _angle(kp[dip] - kp[pip], kp[tip] - kp[dip])
    return out


def motors_to_joints(motors: np.ndarray) -> np.ndarray:
    """6 motor targets -> the 11 Revo 2 joint targets (MJCF order), distal joints coupled as on the hand."""
    q = np.zeros(11)
    q[ACTIVE] = motors
    q[2], q[4], q[6], q[8], q[10] = (COUPLING[k] * q[k - 1] for k in (2, 4, 6, 8, 10))
    return np.clip(q, 0.0, REVO2_UPPER)


class Calibration:
    """Per-operator [open, closed] feature values; JSON {"side", "features", "open": [6], "closed": [6], "poses": {...}}."""

    def __init__(self, open_=None, closed=None, side: str = "", path: str = "") -> None:
        self.open = np.asarray(DEFAULT_OPEN if open_ is None else open_, dtype=np.float64)
        self.closed = np.asarray(DEFAULT_CLOSED if closed is None else closed, dtype=np.float64)
        self.side, self.path = side, path

    @classmethod
    def load(cls, path) -> "Calibration":
        p = Path(path)
        if not p.is_file():
            return cls(path="")
        c = json.loads(p.read_text())
        return cls(c["open"], c["closed"], c.get("side", ""), str(p))

    @staticmethod
    def from_poses(poses: dict, side: str) -> dict:
        """flat -> open of all six; fist -> closed of thumb bend and the four fingers; oppose -> closed of thumb rotation."""
        flat, fist, oppose = (np.asarray(poses[k], dtype=np.float64) for k in ("flat", "fist", "oppose"))
        closed = fist.copy()
        closed[0] = oppose[0]
        return dict(side=side, features=FEATURES, open=flat.round(4).tolist(), closed=closed.round(4).tolist(),
                    poses={k: np.asarray(v).round(4).tolist() for k, v in poses.items()}, made=time.strftime("%Y-%m-%d %H:%M:%S"))


class Revo2Retargeter:
    def __init__(self, calib: Calibration | None = None, pinch_blend: bool = True, smooth_hz: float = 10.0,
                 thumb_rot_min: float = 0.0, side: str = "right") -> None:
        self.calib = calib or Calibration()
        self.pinch_blend, self.smooth_hz, self.thumb_rot_min = pinch_blend, float(smooth_hz), float(thumb_rot_min)
        self.side = side if side in PINCH_KNOTS else "right"
        self._pinch = {f: np.asarray(k, dtype=np.float64) for f, k in PINCH_KNOTS[self.side].items()}
        self._motors = None
        self._t = None

    def reset(self) -> None:
        self._motors = self._t = None

    def step(self, kp: np.ndarray, t: float) -> dict:
        """One keypoint frame at time t (s) -> features, 0..1 fractions, pinch weight, 6 motor and 11 joint targets."""
        kp = np.asarray(kp, dtype=np.float64).reshape(21, 3)
        f = features(kp)
        span = self.calib.closed - self.calib.open
        frac = np.clip((f - self.calib.open) / np.where(np.abs(span) < 1e-6, 1e-6, span), 0.0, 1.0)
        lo = np.zeros(6)
        lo[0] = min(self.thumb_rot_min, MOTOR_UPPER[0])
        raw = lo + frac * (MOTOR_UPPER - lo)
        w, finger = 0.0, "index"
        if self.pinch_blend:                           # the fingertip nearest the thumb tip: that finger keeps following the hand but
            dist = {f: float(np.linalg.norm(kp[4] - kp[i])) for f, i in TIPS.items()}      # closes at least to where the thumb
            finger = min(dist, key=dist.get)                                                # can reach it; the thumb goes to meet it
            w = float(np.clip((PINCH_FAR - dist[finger]) / (PINCH_FAR - PINCH_NEAR), 0.0, 1.0))
            if w > 0:
                k, knots = 2 + list(TIPS).index(finger), self._pinch[finger]
                v = max(raw[k], knots[0, 0])
                if finger in PINCH_HI:                 # the ring only touches the thumb pad inside a narrow curl band
                    v = min(v, PINCH_HI[finger])
                thumb = np.array([np.interp(v, knots[:, 0], knots[:, 1]), np.interp(v, knots[:, 0], knots[:, 2])])
                raw[k] = (1 - w) * raw[k] + w * v
                raw[:2] = (1 - w) * raw[:2] + w * thumb
        if self._motors is None or self.smooth_hz <= 0:
            self._motors = raw
        else:
            a = 1.0 - np.exp(-2 * np.pi * self.smooth_hz * max(t - self._t, 0.0))
            self._motors = self._motors + a * (raw - self._motors)
        self._t = t
        return dict(features=f, fractions=frac, pinch_w=w, pinch_finger=finger, motors=self._motors.copy(), q=motors_to_joints(self._motors))


class GloveHand:
    """A glove source + retargeter + hold policy for one hand. update(t) -> the latest result dict (with "held", "fresh",
    "frame") or None before the first usable frame. Every update is appended to `log` (wuji_glove.GloveLog) when given."""

    def __init__(self, side: str, source, retargeter: Revo2Retargeter, min_conf: float = 0.5, stale_s: float = 0.25, log=None) -> None:
        self.side, self.source, self.rt = side, source, retargeter
        self.min_conf, self.stale_s, self.log = float(min_conf), float(stale_s), log
        self.last = None
        self._last_new = None
        self._t_span = [None, None]
        self.stats = dict(updates=0, ticks_with_new_frame=0, low_conf=0, held=0, longest_gap_s=0.0, retarget_us=[])

    def update(self, t: float, tick: int = -1) -> dict | None:
        self.stats["updates"] += 1
        self._t_span = [t if self._t_span[0] is None else self._t_span[0], t]
        fr = self.source.latest(t)
        fresh = False
        if fr is not None:
            self.stats["ticks_with_new_frame"] += 1
            if self._last_new is not None:
                self.stats["longest_gap_s"] = max(self.stats["longest_gap_s"], t - self._last_new)
            self._last_new = t
            if float(np.mean(fr.conf)) < self.min_conf:
                self.stats["low_conf"] += 1
            else:
                t0 = time.perf_counter()
                out = self.rt.step(fr.kp, t)
                self.stats["retarget_us"].append((time.perf_counter() - t0) * 1e6)
                self.last = dict(out, frame=fr)
                fresh = True
        if self.last is None:
            return None
        held = not fresh and (self._last_new is None or t - self._last_new > self.stale_s or fr is not None)
        self.stats["held"] += int(held)
        res = dict(self.last, held=held, fresh=fresh)
        if self.log is not None:
            self.log.append(t=t, tick=tick, side=self.side, frame=res["frame"], fresh=fresh, held=held, out=res)
        return res

    def summary(self) -> dict:
        s = dict(self.stats)
        us = np.asarray(s.pop("retarget_us") or [0.0])
        span = (self._t_span[1] - self._t_span[0]) if self._t_span[0] is not None else 0.0
        s.update(retarget_us_mean=round(float(us.mean()), 1), retarget_us_p99=round(float(np.percentile(us, 99)), 1),
                 glove_frames_received=int(getattr(self.source, "received", -1)),
                 glove_rate_hz=round(getattr(self.source, "received", 0) / span, 1) if span > 0 else None)
        return s
