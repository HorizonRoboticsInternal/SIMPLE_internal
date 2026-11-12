"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

PICO hand skeleton -> BrainCo Revo 2 finger targets, as an option of the HoloMotion v1.4 teleop (2026-11-03, from the pilot in
scripts/wuji_glove_pilot/, docs/HANDOVER_PICO_SKELETON_REVO2_TELEOP.md). The headset tracks the operator's hands (26 OpenXR
joints per hand through the XRoboToolkit PC service); the publisher puts both hands' 26 x 7 arrays and the two isActive flags
on the wire next to the controller sample (wire.HANDS_TOPIC), the recorder keeps them (holomotion.pico_hands), and the agent
turns each hand into the 6 Revo 2 motor targets every 20 ms in place of the grip-button poses (--hand-source skeleton).

The retarget (unchanged from the pilot): 21 MediaPipe keypoints -> 6 bone-angle scalars (frame- and side-independent) ->
per-operator [open, closed] calibration -> motor travel, a pinch blend (the thumb goes to meet the pinched finger: PINCH_KNOTS
per side from pinch_poses.py), a 10 Hz low-pass; the distal joints follow their proximal (COUPLING). Hold policy: no new frame,
or a low-quality one, keeps the last target (a hand never springs open on a dropout: a held object would drop).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from simple.robots.g1_sonic_revo2 import COUPLING, REVO2_UPPER
from simple.robots import g1_sonic_revo2 as _revo2                           # HAND_VARIANT: which pinch knots apply

# OpenXR XR_HAND_JOINT order: 0 palm, 1 wrist, 2-5 thumb (metacarpal, proximal, distal, tip), then per finger metacarpal,
# proximal, intermediate, distal, tip: index 6-10, middle 11-15, ring 16-20, little 21-25 -> the 21 MediaPipe keypoints
XR_TO_MP = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
HANDS_SHAPE = (2, 26, 7)                                                     # left, right; 26 joints; x y z qx qy qz qw (m, headset world frame)
HANDS_COLUMN = 2 * 26 * 7 + 2                                                # holomotion.pico_hands: both arrays flat + the two isActive flags
LOW_CONF = 0.25                                                              # confidence of a frame the SDK marks not active (isActive 0)
SIDES = ("left", "right")

# MediaPipe landmarks: 0 wrist, then CMC/MCP, (MCP/PIP), (IP/DIP), TIP per finger
WRIST = 0
THUMB = (1, 2, 3, 4)
FINGER_LM = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FEATURES = ["thumb_rot", "thumb_bend", "index", "middle", "ring", "pinky"]
MOTORS = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
ACTIVE = [0, 1, 3, 5, 7, 9]                                                  # the motor (proximal) joints in the 11-joint MJCF order
MOTOR_UPPER = REVO2_UPPER[ACTIVE]                                            # import-time snapshot (legacy hand); the retargeter reads motor_upper()


def motor_upper() -> np.ndarray:
    """The 6 motors' upper limits (rad) of the SELECTED hand model: REVO2_UPPER is rewritten in place by
    g1_sonic_revo2.select_hand_variant (legacy 84 deg fingers, official 80.8 deg), so read it at call time."""
    return REVO2_UPPER[ACTIVE]
DEFAULT_OPEN = np.array([0.10, 0.25, 0.25, 0.25, 0.25, 0.25])               # human feature values (rad): open hand / closed end of each motor
DEFAULT_CLOSED = np.array([0.85, 1.60, 3.60, 3.60, 3.60, 3.60])
PINCH_NEAR, PINCH_FAR = 0.015, 0.035                                         # human thumb tip - fingertip distance (m): full blend / no blend
TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
# Where the Revo 2 thumb pad meets each finger's pad (scripts/wuji_glove_pilot/pinch_poses.py --table on g1_rot0_teleop_revo2_v3.xml,
# 2026-11-03; the two hands of that file are not mirror images): per finger, knots of (finger motor, thumb rotation, thumb bend)
# that settle with the pads touching, from the smallest finger curl the thumb can reach to the largest; the thumb is interpolated.
# The pinky cannot be reached by the thumb in this model (13 mm at best): its knot is the closest approach.
PINCH_KNOTS = {
    "right": {"index": [(0.69, 1.15, 0.90), (0.98, 1.15, 0.90)],
              "middle": [(0.78, 1.45, 0.90), (0.98, 1.45, 0.90), (1.17, 1.50, 1.00), (1.47, 1.50, 1.00)],
              "ring": [(0.70, 1.50, 0.90), (0.90, 1.50, 0.90), (1.00, 1.50, 1.00), (1.47, 1.50, 1.00)],
              "pinky": [(0.90, 1.50, 1.00), (1.47, 1.50, 1.00)]},
    "left": {"index": [(0.69, 1.15, 0.75), (0.92, 1.15, 0.75), (1.08, 1.20, 0.70)],
             "middle": [(0.78, 1.35, 0.75), (0.98, 1.35, 0.75), (1.17, 1.50, 0.80), (1.47, 1.50, 0.95)],
             "ring": [(0.70, 1.50, 1.00), (1.47, 1.50, 1.00)],
             "pinky": [(1.00, 1.50, 1.00), (1.47, 1.50, 1.00)]},
}
# 2026-11-04, the official-hand robot (g1_sonic_revo2.select_hand_variant("official"), BrainCo v1.1.1): measured the same way,
# pinch_poses.py --table --robot-mjcf data/robots/g1_rot0_teleop_revo2_official/g1_rot0_teleop_revo2_official.xml (table in
# docs/revo2_official_compare/pinch_table_official.log), hand-smoothed. The thumb values are lower than the legacy ones (BrainCo moved
# the thumb zeros by 3 / 9 deg into the joint origins) and the finger limit is 1.41. Right pinky: pads meet only at curl 0.85;
# left pinky: never (11 mm at best), its knot is the closest approach.
PINCH_KNOTS_OFFICIAL = {
    "right": {"index": [(0.67, 1.15, 0.75), (1.04, 1.15, 0.65)],
              "middle": [(0.67, 1.30, 0.75), (0.85, 1.15, 0.70), (1.04, 1.50, 0.60), (1.32, 1.45, 0.80)],
              "ring": [(0.76, 1.50, 0.75), (0.95, 1.55, 0.70), (1.04, 1.55, 0.70)],
              "pinky": [(0.85, 1.55, 0.65)]},
    "left": {"index": [(0.67, 1.15, 0.75), (1.04, 1.15, 0.65)],
             "middle": [(0.67, 1.40, 0.70), (0.85, 1.20, 0.60), (0.95, 1.45, 0.55), (1.13, 1.55, 0.65), (1.41, 1.35, 1.00)],
             "ring": [(0.76, 1.45, 0.70), (0.95, 1.55, 0.70)],
             "pinky": [(0.76, 1.55, 0.80)]},
}
PINCH_KNOTS_BY_HAND = {"legacy": PINCH_KNOTS, "official": PINCH_KNOTS_OFFICIAL}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0)))


def features(kp: np.ndarray) -> np.ndarray:
    """The 6 human scalars (rad, FEATURES order) from 21 MediaPipe keypoints (21 x 3, any right-handed frame, either hand)."""
    kp = np.asarray(kp, dtype=np.float64).reshape(21, 3)
    across = _unit(kp[5] - kp[17])                                           # little-finger MCP -> index MCP
    fwd = _unit(kp[9] - kp[WRIST])                                           # wrist -> middle MCP
    normal = _unit(np.cross(across, fwd))                                    # palm-plane normal (sign unused)
    out = np.zeros(6)
    t = kp[THUMB[1]] - kp[THUMB[0]]
    out[0] = float(np.arcsin(np.clip(abs(np.dot(_unit(t), normal)), 0.0, 1.0)))
    out[1] = _angle(t, kp[THUMB[2]] - kp[THUMB[1]]) + _angle(kp[THUMB[2]] - kp[THUMB[1]], kp[THUMB[3]] - kp[THUMB[2]])
    for i, (mcp, pip, dip, tip) in enumerate(FINGER_LM.values()):
        meta, prox = kp[mcp] - kp[WRIST], kp[pip] - kp[mcp]
        flat = lambda v: v - np.dot(v, across) * across                     # noqa: E731  remove the spread (abduction) component
        out[2 + i] = _angle(flat(meta), flat(prox)) + _angle(prox, kp[dip] - kp[pip]) + _angle(kp[dip] - kp[pip], kp[tip] - kp[dip])
    return out


def motors_to_joints(motors: np.ndarray) -> np.ndarray:
    """6 motor targets -> the 11 Revo 2 joint targets (MJCF order), distal joints coupled as on the hand."""
    q = np.zeros(11)
    q[ACTIVE] = motors
    q[2], q[4], q[6], q[8], q[10] = (COUPLING[k] * q[k - 1] for k in (2, 4, 6, 8, 10))
    return np.clip(q, 0.0, REVO2_UPPER)


class Calibration:
    """Per-operator [open, closed] feature values; JSON {"side", "features", "open": [6], "closed": [6], "poses": {...}}
    (scripts/wuji_glove_pilot/bench.py --glove pico --side <side> --calibrate writes calib_<side>_pico.json)."""

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


class Revo2Retargeter:
    def __init__(self, calib: Calibration | None = None, pinch_blend: bool = True, smooth_hz: float = 10.0,
                 thumb_rot_min: float = 0.0, side: str = "right") -> None:
        self.calib = calib or Calibration()
        self.pinch_blend, self.smooth_hz, self.thumb_rot_min = pinch_blend, float(smooth_hz), float(thumb_rot_min)
        knots = PINCH_KNOTS_BY_HAND.get(_revo2.HAND_VARIANT, PINCH_KNOTS)      # the selected hand model's knots (legacy / official)
        self.side = side if side in knots else "right"
        self._pinch = {f: np.asarray(k, dtype=np.float64) for f, k in knots[self.side].items()}
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
        lo = np.zeros(6); mu = motor_upper()
        lo[0] = min(self.thumb_rot_min, mu[0])
        raw = lo + frac * (mu - lo)
        w, finger = 0.0, "index"
        if self.pinch_blend:                                                 # the fingertip nearest the thumb tip: that finger keeps following
            dist = {f: float(np.linalg.norm(kp[4] - kp[i])) for f, i in TIPS.items()}      # the hand but closes at least to where the
            finger = min(dist, key=dist.get)                                                # thumb can reach it; the thumb goes to meet it
            w = float(np.clip((PINCH_FAR - dist[finger]) / (PINCH_FAR - PINCH_NEAR), 0.0, 1.0))
            if w > 0:
                k, knots = 2 + list(TIPS).index(finger), self._pinch[finger]
                v = max(raw[k], knots[0, 0])
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


class SkeletonHand:
    """One hand of the teleop: the SDK's 26 x 7 array and isActive flag per tick -> the 11 Revo 2 joint targets, with the hold
    policy. update(h, active, t) returns the current target (float32, MJCF order) or None before the first usable frame."""

    def __init__(self, side: str, retargeter: Revo2Retargeter, min_conf: float = 0.0, stale_s: float = 0.25) -> None:
        self.side, self.rt, self.min_conf, self.stale_s = side, retargeter, float(min_conf), float(stale_s)
        self.q: np.ndarray | None = None
        self.last: dict | None = None
        self._last_h = None
        self._t_new = None
        self.stats = dict(received=0, low_conf=0, held=0, updates=0)

    def update(self, h, active: int, t: float) -> np.ndarray | None:
        self.stats["updates"] += 1
        if h is not None:
            h = np.asarray(h, dtype=np.float32).reshape(26, 7)
            if np.any(h[:, :3]) and (self._last_h is None or not np.array_equal(h, self._last_h)):   # a new frame (the SDK stamps none)
                self._last_h = h.copy()
                self.stats["received"] += 1
                conf = 1.0 if int(active) == 1 else LOW_CONF
                if conf >= self.min_conf:
                    out = self.rt.step(h[XR_TO_MP, :3], t)
                    self.q = np.asarray(out["q"], dtype=np.float32)
                    self.last = out
                    self._t_new = t
                else:
                    self.stats["low_conf"] += 1
        if self.q is None:
            return None
        if self._t_new is None or t - self._t_new > self.stale_s:
            self.stats["held"] += 1
        return self.q

    def summary(self) -> dict:
        s = dict(self.stats)
        if self.last is not None:
            s["motors"] = np.round(self.last["motors"], 2).tolist()
            s["pinch"] = (self.last["pinch_finger"], round(float(self.last["pinch_w"]), 2))
        return s


def build_hands(sides, calib_dir: str | Path = "", min_conf: float = 0.0, smooth_hz: float = 10.0, pinch_blend: bool = True, calib: dict | None = None) -> dict:
    """The teleop's SkeletonHand per side; the operator calibration calib_<side>_pico.json from calib_dir when present
    (default data/wuji_glove_pilot), else the built-in defaults; `calib` = {side: {"open": [6], "closed": [6]}} overrides both
    (a re-record uses the calibration the recording's exact log kept)."""
    root = Path(__file__).resolve().parents[4]
    cdir = Path(calib_dir) if calib_dir else root / "data" / "wuji_glove_pilot"
    out = {}
    for s in sides:
        calib_obj = Calibration(calib[s]["open"], calib[s]["closed"], s, "recorded") if calib and s in calib else Calibration.load(cdir / f"calib_{s}_pico.json")
        out[s] = SkeletonHand(s, Revo2Retargeter(calib_obj, side=s, smooth_hz=smooth_hz, pinch_blend=pinch_blend), min_conf=min_conf)
        print(f"[hand_skeleton] {s} hand <- the headset's hand tracking (calibration {calib_obj.path or 'built-in defaults'}, "
              f"min_conf {min_conf}, smooth {smooth_hz} Hz, pinch blend {'on' if pinch_blend else 'off'})", flush=True)
    return out


def hands_meta(hands: dict, min_conf: float, smooth_hz: float, pinch_blend: bool) -> dict:
    """What the exact log keeps about the skeleton source, enough for a re-record to rebuild the same retarget."""
    return dict(sides=list(hands), min_conf=float(min_conf), smooth_hz=float(smooth_hz), pinch_blend=bool(pinch_blend),
                calib={s: dict(open=h.rt.calib.open.round(5).tolist(), closed=h.rt.calib.closed.round(5).tolist(), path=h.rt.calib.path) for s, h in hands.items()})


def hands_vector(sample: dict | None) -> np.ndarray:
    """The recorded holomotion.pico_hands row: both 26 x 7 arrays flat + the two isActive flags; NaN when no hands sample."""
    if sample is None:
        return np.full(HANDS_COLUMN, np.nan, dtype=np.float32)
    return np.concatenate([np.asarray(sample["hands"], dtype=np.float32).reshape(-1), np.asarray(sample["active"], dtype=np.float32).reshape(2)])


def hands_sample_from_vector(v) -> dict | None:
    """The inverse, for re-records (NaN = no sample on that tick)."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.shape[0] < HANDS_COLUMN or not np.isfinite(v[0]):
        return None
    return dict(hands=v[:HANDS_COLUMN - 2].reshape(HANDS_SHAPE), active=v[HANDS_COLUMN - 2:HANDS_COLUMN].astype(np.int8), received_monotonic=0.0)


# ------------------------------------------------------------------------------------------------------------- synthetic
from simple.teleop.holomotion_v14.hand_synth import synthetic_hands  # noqa: E402  (tests; the synthetic publisher loads hand_synth directly)


def _selftest() -> None:
    hands, active = synthetic_hands(0.0)
    h = {s: SkeletonHand(s, Revo2Retargeter(side=s)) for s in SIDES}
    q0 = {s: h[s].update(hands[i], active[i], 0.0) for i, s in enumerate(SIDES)}
    assert all(q is not None and q.shape == (11,) for q in q0.values()), q0
    assert all(h[s].update(hands[i], active[i], 0.02) is not None and h[s].stats["received"] == 1 for i, s in enumerate(SIDES)), "an unchanged array is not a new frame"
    fist, act = synthetic_hands(3.8)                                         # the fist of the script
    qf = {s: h[s].update(fist[i], act[i], 5.0) for i, s in enumerate(SIDES)}
    assert all(qf[s][3] > q0[s][3] + 0.5 for s in SIDES), (q0, qf)          # index proximal closes
    drop, act0 = synthetic_hands(5.2)                                        # the dropout: zeros, not active
    assert all(np.array_equal(h[s].update(drop[i], act0[i], 5.3), qf[s]) for i, s in enumerate(SIDES)), "hold on dropout"
    v = hands_vector(dict(hands=hands, active=active)); back = hands_sample_from_vector(v)
    assert v.shape == (HANDS_COLUMN,) and np.array_equal(back["hands"], hands) and list(back["active"]) == [1, 1]
    assert hands_sample_from_vector(hands_vector(None)) is None
    print("[hand_skeleton] selftest OK:", {s: h[s].summary() for s in SIDES})


if __name__ == "__main__":
    _selftest()
