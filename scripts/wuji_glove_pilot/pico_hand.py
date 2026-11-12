"""PICO headset hand tracking as a hand source for the pilot (2026-11-03): the same latest(t) -> GloveFrame | None interface
as the glove sources in wuji_glove.py, so the retarget, the bench and the teleop take it unchanged (--glove pico).

The XRoboToolkit SDK gives the headset's 26 OpenXR hand joints per hand (x, y, z, qx, qy, qz, qw each; metres, the
headset's world frame, which the bone-angle retarget does not care about). The retarget takes the 21 MediaPipe keypoints,
so the palm joint and the four finger metacarpals are dropped (XR_TO_MP). The SDK has no per-sample hand timestamp: a
frame counts as new when the joint array changed. Confidence is the SDK's isActive flag (1 = high tracking quality) ->
1.0, else LOW_CONF, which is below the default --min-conf 0.5: a low-quality frame is held, not applied (--min-conf 0
takes it).

Needs the XRoboToolkit PC service and the headset's XRoboToolkit app with hand tracking enabled; the headset switches to
hand tracking when the controllers are put down. Both sides share one SDK session (init once, close with the last user).

Self-test with a fake SDK (no headset): .venv/bin/python scripts/wuji_glove_pilot/pico_hand.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"), str(Path(__file__).resolve().parent)]
from wuji_glove import GloveFrame  # noqa: E402

# OpenXR XR_HAND_JOINT order: 0 palm, 1 wrist, 2-5 thumb (metacarpal, proximal, distal, tip), then per finger
# metacarpal, proximal, intermediate, distal, tip: index 6-10, middle 11-15, ring 16-20, little 21-25.
XR_TO_MP = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
LOW_CONF = 0.25

_sdk = None
_users = 0


def _open_sdk(module=None):
    global _sdk, _users
    if _sdk is None:
        if module is None:
            import xrobotoolkit_sdk as module
            module.init()
        _sdk = module
    _users += 1
    return _sdk


def _close_sdk() -> None:
    global _sdk, _users
    _users -= 1
    if _users <= 0 and _sdk is not None:
        try:
            _sdk.close()
        except Exception:                             # noqa: BLE001
            pass
        _sdk, _users = None, 0


class PicoHandSource:
    def __init__(self, side: str, low_conf: float = LOW_CONF, sdk=None) -> None:
        self.side, self.low_conf = side, float(low_conf)
        self.xrt = _open_sdk(sdk)
        self._state = self.xrt.get_left_hand_tracking_state if side == "left" else self.xrt.get_right_hand_tracking_state
        self._active = self.xrt.get_left_hand_is_active if side == "left" else self.xrt.get_right_hand_is_active
        self._last = None
        self.received = 0
        self.low_quality = 0
        print(f"[pico_hand] {side}: headset hand tracking through the XRoboToolkit SDK (26 joints -> 21 keypoints)", flush=True)

    def latest(self, t: float) -> GloveFrame | None:
        h = np.asarray(self._state(), dtype=np.float64).reshape(26, 7)
        if not np.any(h[:, :3]):                      # all zeros: the headset has sent no hand data yet
            return None
        if self._last is not None and np.array_equal(h, self._last):
            return None
        self._last = h
        self.received += 1
        active = int(self._active()) == 1
        self.low_quality += int(not active)
        conf = 1.0 if active else self.low_conf
        return GloveFrame(h[XR_TO_MP, :3].astype(np.float32), np.full(21, conf, np.float32), self.received, int(t * 1e6), t)

    def close(self) -> None:
        _close_sdk()


# ------------------------------------------------------------------------------------------------------------- selftest
class _FakeSdk:
    """A stand-in for xrobotoolkit_sdk: holds one 26 x 7 array per side and the isActive flags."""

    def __init__(self) -> None:
        self.state = {"left": np.zeros((26, 7)), "right": np.zeros((26, 7))}
        self.active = {"left": 0, "right": 0}
        self.closed = 0

    def get_left_hand_tracking_state(self): return self.state["left"].tolist()
    def get_right_hand_tracking_state(self): return self.state["right"].tolist()
    def get_left_hand_is_active(self): return self.active["left"]
    def get_right_hand_is_active(self): return self.active["right"]
    def close(self): self.closed += 1


def mp_to_xr(kp: np.ndarray) -> np.ndarray:
    """21 MediaPipe keypoints -> a 26 x 7 OpenXR joint array (metacarpals interpolated, palm = mid-hand, identity quaternions)."""
    kp = np.asarray(kp, dtype=np.float64)
    h = np.zeros((26, 7))
    h[:, 6] = 1.0
    h[XR_TO_MP, :3] = kp
    for meta, mcp in ((6, 5), (11, 9), (16, 13), (21, 17)):
        h[meta, :3] = 0.5 * (kp[0] + kp[mcp])
    h[0, :3] = 0.5 * (kp[0] + kp[9])
    return h


def _selftest() -> None:
    from revo2_retarget import features
    from wuji_glove import FIST, FLAT, human_keypoints

    fake = _FakeSdk()
    right, left = PicoHandSource("right", sdk=fake), PicoHandSource("left", sdk=fake)
    assert right.latest(0.0) is None and left.latest(0.0) is None, "no data -> no frame"
    kp = human_keypoints(FLAT, "right")
    fake.state["right"], fake.active["right"] = mp_to_xr(kp), 1
    fr = right.latest(0.1)
    assert fr is not None and np.allclose(fr.kp, kp, atol=1e-6) and fr.conf.min() == 1.0 and fr.seq == 1, "26 -> 21 mapping"
    assert np.allclose(features(fr.kp), features(kp)), "the features of the mapped frame are the glove's"
    assert right.latest(0.2) is None, "an unchanged array is not a new frame"
    fake.state["right"], fake.active["right"] = mp_to_xr(human_keypoints(FIST, "right")), 0
    fr = right.latest(0.3)
    assert fr is not None and fr.seq == 2 and abs(float(fr.conf.mean()) - LOW_CONF) < 1e-6, "low quality -> LOW_CONF"
    assert features(fr.kp)[2] > 3.0, "the fist reaches the retarget"
    assert left.latest(0.3) is None, "the left hand is still empty"
    right.close()
    assert fake.closed == 0, "the SDK stays open while the other side uses it"
    left.close()
    assert fake.closed == 1, "the last user closes the SDK"
    print("[pico_hand] selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
