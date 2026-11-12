"""Synthetic headset hands for tests (2026-11-03): both hands of the pilot's scripted human hand (flat, fist, thumb opposition,
thumb-index pinch, finger wave; a 0.4 s dropout per 16 s cycle) as the SDK would give them, (hands (2, 26, 7) float32, active
(2,) int8). Only numpy and the pilot's models in scripts/wuji_glove_pilot (wuji_glove.human_keypoints / script_pose,
pico_hand.mp_to_xr) -- no SIMPLE imports, so the publisher (holomotion_teleop env) can load it like wire.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HANDS_SHAPE = (2, 26, 7)
SIDES = ("left", "right")


def synthetic_hands(t: float):
    pilot = Path(__file__).resolve().parents[4] / "scripts" / "wuji_glove_pilot"
    if str(pilot) not in sys.path:
        sys.path.insert(0, str(pilot))
    from pico_hand import mp_to_xr                                           # noqa: E402  (lazy; the SDK is not imported)
    from wuji_glove import CYCLE_S, DROPOUT, human_keypoints, script_pose    # noqa: E402
    hands = np.zeros(HANDS_SHAPE, dtype=np.float32)
    if DROPOUT[0] <= t % CYCLE_S < DROPOUT[1]:
        return hands, np.zeros(2, dtype=np.int8)                             # no data (all zeros, not active) during the dropout
    pose, _ = script_pose(t)
    for i, side in enumerate(SIDES):
        hands[i] = mp_to_xr(human_keypoints(pose, side)).astype(np.float32)
    return hands, np.ones(2, dtype=np.int8)
