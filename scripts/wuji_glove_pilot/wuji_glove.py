"""Hand keypoint sources for the Wuji Glove pilot (2026-10-27), all with latest(t) -> GloveFrame | None (None = no new frame):

  WujiGloveSource       the real glove through wuji_sdk (pip install wuji-sdk): its hand_skeleton stream, 21 MediaPipe
                        keypoints in metres, 120 Hz over Ethernet. Found by the SDK scan (side from the serial number:
                        ...J... = left, ...K... = right), else the factory address 192.168.1.100 (left) / .101 (right), port
                        50001 (Wuji's docs say 50000; the glove answers on 50001). The host NIC needs a static 192.168.1.x/24.
  SyntheticGloveSource  a scripted human hand (flat, fist, thumb opposition, thumb-index pinch, finger wave; 16 s cycle,
                        120 Hz, one 0.4 s dropout per cycle at 5.0 s) for testing everything without the glove.
  ReplayGloveSource     the frames of a GloveLog (bench or teleop run), looped, on their recorded timing. A log keeps the
                        frame each 20 ms tick used, so a replay runs at 50 Hz, not the glove's 120 Hz.
  PicoHandSource        (pico_hand.py) the PICO headset's own hand tracking through the XRoboToolkit SDK, its 26 OpenXR
                        joints reduced to the 21 MediaPipe keypoints; no glove needed.

GloveLog: one fixed-width float64 row per hand update, appended and flushed at once (the teleop CLI ends with os._exit, so
nothing can be written at exit), plus <log>.json with the column layout. load_log(path) -> dict of columns.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MP_NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"] + [
    f"{f}_{j}" for f in ("index_finger", "middle_finger", "ring_finger", "pinky") for j in ("mcp", "pip", "dip", "tip")]
FACTORY_ADDRESS = {"left": "192.168.1.100:50001", "right": "192.168.1.101:50001"}
SIDE_LETTER = {"left": "J", "right": "K"}         # 4th character of the serial number (WG1KA... = a right glove)


@dataclass
class GloveFrame:
    kp: np.ndarray            # 21 x 3 float32, metres, MediaPipe order
    conf: np.ndarray          # 21 float32
    seq: int
    t_device_us: int
    t: float                  # the caller's clock when the frame was taken


# ---------------------------------------------------------------------------------------------------------- real glove
class WujiGloveSource:
    def __init__(self, side: str, address: str = "", sn: str = "") -> None:
        from wuji_sdk import SdkManager
        self.manager = SdkManager.instance()
        name = f"pilot_glove_{side}"
        if sn:
            via = sn
            self.dev = self.manager.connect(sn=sn, device_name=name)
        else:
            via = address or self._scan_address(side) or FACTORY_ADDRESS[side]
            self.dev = self.manager.connect(address=via if ":" in via else f"{via}:50001", device_name=name)
        self.sub = self.dev.hand_skeleton().subscribe()
        self._order = None
        self.received = 0
        print(f"[wuji_glove] {side}: connected {getattr(self.dev, 'serial_number', '?')} ({via})", flush=True)

    def _scan_address(self, side: str) -> str:
        """The address of the one glove of this side the SDK scan finds (about 2 s); '' when none or several."""
        try:
            found = [d for d in self.manager.scan() if len(d.sn) > 3 and d.sn[3] == SIDE_LETTER[side]]
        except Exception as e:                        # noqa: BLE001
            print(f"[wuji_glove] scan failed ({e}); using the factory address", flush=True)
            return ""
        return found[0].address if len(found) == 1 else ""

    def latest(self, t: float) -> GloveFrame | None:
        newest = None
        while True:                                   # recv() returns the OLDEST unread frame: drain to the newest
            f = self.sub.recv()
            if f is None:
                break
            newest = f
            self.received += 1
        if newest is None or len(newest.joints) != 21:
            return None
        if self._order is None:                       # MediaPipe order by name when the names are the canonical ones
            names = [str(j.name).lower() for j in newest.joints]
            self._order = [names.index(n) for n in MP_NAMES] if sorted(names) == sorted(MP_NAMES) else list(range(21))
        js = [newest.joints[i] for i in self._order]
        return GloveFrame(np.array([j.pose.position for j in js], dtype=np.float32), np.array([j.confidence for j in js], dtype=np.float32),
                          int(newest.header.seq), int(newest.header.timestamp_us), t)

    def close(self) -> None:
        try:
            self.sub.close()
            self.dev.disconnect()
        except Exception:                             # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------------ synthetic hand
# a right hand in a palm frame: x along the fingers, y toward the thumb, z out of the back of the hand (left = y mirrored)
_BASES = {"index": (0.092, 0.024), "middle": (0.095, 0.002), "ring": (0.089, -0.018), "pinky": (0.080, -0.036)}
_LENS = {"index": (0.042, 0.025, 0.020), "middle": (0.046, 0.029, 0.021), "ring": (0.043, 0.027, 0.021), "pinky": (0.034, 0.020, 0.019)}
_SPLIT = (0.38, 0.40, 0.22)                           # share of a finger's curl at MCP, PIP, DIP
_CMC, _THUMB_L = np.array([0.022, 0.022, -0.010]), (0.046, 0.032, 0.028)
_T0 = np.array([np.cos(np.radians(40)), np.sin(np.radians(40)), 0.0])      # thumb metacarpal in the palm plane (flat hand)
_T1 = np.array([0.55, -0.10, -0.83]) / np.linalg.norm([0.55, -0.10, -0.83])  # fully opposed: out of the palm, palm side
_THUMB_TOWARD = np.array([0.07, -0.02, -0.02])        # thumb flexion bends toward this point

# poses: [thumb opposition 0..1, thumb bend (rad), index, middle, ring, pinky curl (rad, MCP + PIP + DIP)]
FLAT = [0.0, 0.15, 0.10, 0.10, 0.10, 0.10]
FIST = [0.45, 1.45, 3.70, 3.80, 3.80, 3.70]
OPPOSE = [1.0, 0.35, 0.10, 0.10, 0.10, 0.10]
PINCH = [0.65, 0.40, 2.15, 0.10, 0.10, 0.10]          # grid search: thumb tip 1.4 mm from the index tip
CURL = lambda i: [0.0, 0.15] + [3.2 if k == i else 0.10 for k in range(4)]     # noqa: E731  one finger curled
SCRIPT = [(0.0, FLAT, "flat"), (1.5, FLAT, "flat"), (3.0, FIST, "fist"), (4.5, FIST, "fist"), (6.0, FLAT, "flat"),
          (7.0, FLAT, "flat"), (8.5, OPPOSE, "oppose"), (9.5, OPPOSE, "oppose"), (10.5, FLAT, "flat"), (11.5, PINCH, "pinch"),
          (12.5, PINCH, "pinch"), (13.3, FLAT, "flat"), (13.8, CURL(0), "index"), (14.3, CURL(1), "middle"),
          (14.8, CURL(2), "ring"), (15.3, CURL(3), "pinky"), (16.0, FLAT, "flat")]
CYCLE_S = SCRIPT[-1][0]
DROPOUT = (5.0, 5.4)                                  # no frames in this window of every cycle


def _u(v):
    return v / np.linalg.norm(v)


def human_keypoints(p, side: str = "right") -> np.ndarray:
    """21 MediaPipe keypoints (m) of the synthetic hand for pose p (see FLAT)."""
    opp, tb, *curls = p
    out, z = np.zeros((21, 3)), np.array([0.0, 0.0, 1.0])
    t = _u((1 - opp) * _T0 + opp * _T1)
    out[1] = _CMC
    out[2] = _CMC + _THUMB_L[0] * t
    b = _THUMB_TOWARD - out[2]
    b = _u(b - np.dot(b, t) * t)
    phi = 0.0
    for k, (share, L) in enumerate(zip((0.45, 0.55), _THUMB_L[1:])):
        phi += share * tb
        out[3 + k] = out[2 + k] + L * (np.cos(phi) * t + np.sin(phi) * b)
    for i, f in enumerate(_BASES):
        B = np.array([*_BASES[f], 0.0])
        m, phi = _u(B), 0.0
        out[5 + 4 * i] = B
        for k in range(3):
            phi += _SPLIT[k] * curls[i]
            out[6 + 4 * i + k] = out[5 + 4 * i + k] + _LENS[f][k] * (np.cos(phi) * m - np.sin(phi) * z)
    return out * np.array([1.0, -1.0, 1.0]) if side == "left" else out


def script_pose(t: float) -> tuple[np.ndarray, str]:
    """The scripted pose at t (cosine blend between key poses) and the name of the pose being approached / held."""
    tc = t % CYCLE_S
    for (t0, p0, _), (t1, p1, n1) in zip(SCRIPT, SCRIPT[1:]):
        if t0 <= tc <= t1:
            a = 0.5 - 0.5 * np.cos(np.pi * (tc - t0) / max(t1 - t0, 1e-9))
            return (1 - a) * np.asarray(p0) + a * np.asarray(p1), n1
    return np.asarray(FLAT, dtype=float), "flat"


class SyntheticGloveSource:
    def __init__(self, side: str, rate_hz: float = 120.0, noise_mm: float = 0.0, dropouts: bool = True, seed: int = 0) -> None:
        self.side, self.dt, self.noise = side, 1.0 / rate_hz, noise_mm * 1e-3
        self.dropouts, self.rng = dropouts, np.random.default_rng(seed)
        self._t0, self._last_k, self.received = None, -1, 0

    def _dropped(self, k: int) -> bool:
        return self.dropouts and DROPOUT[0] <= (k * self.dt) % CYCLE_S < DROPOUT[1]

    def latest(self, t: float) -> GloveFrame | None:
        if self._t0 is None:
            self._t0 = t
        k = int((t - self._t0) / self.dt + 1e-9)                    # newest device frame at or before t
        if k <= self._last_k:
            return None
        self.received += sum(not self._dropped(j) for j in range(self._last_k + 1, k + 1))
        self._last_k = k
        ts = k * self.dt
        if self._dropped(k):
            return None
        kp = human_keypoints(script_pose(ts)[0], self.side)
        if self.noise > 0:
            kp = kp + self.rng.normal(0.0, self.noise, kp.shape)
        return GloveFrame(kp.astype(np.float32), np.ones(21, np.float32), k, int(ts * 1e6), t)

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------------------------------------- replay
class ReplayGloveSource:
    def __init__(self, path: str, side: str) -> None:
        L = load_log(path)
        sel = (L["side"] == (0 if side == "left" else 1)) & (L["fresh"] > 0)
        if not sel.any():
            raise ValueError(f"{path}: no {side} glove frames")
        self.t = L["t"][sel] - L["t"][sel][0]
        if np.any(np.diff(self.t) < 0):
            raise ValueError(f"{path}: the clock goes backwards (more than one run in the file?)")
        self.kp, self.conf = L["kp"][sel].reshape(-1, 21, 3), L["conf"][sel]
        self.seq, self.dev = L["seq"][sel].astype(int), L["t_device_us"][sel].astype(int)
        self.period = float(self.t[-1] + np.median(np.diff(self.t))) if len(self.t) > 1 else 1.0
        self._t0, self._last, self.received = None, -1, 0

    def latest(self, t: float) -> GloveFrame | None:
        if self._t0 is None:
            self._t0 = t
        e = t - self._t0
        lap, tc = divmod(e, self.period)
        i = int(np.searchsorted(self.t, tc, side="right")) - 1
        k = int(lap) * len(self.t) + i
        if i < 0 or k <= self._last:
            return None
        self.received += k - self._last if self._last >= 0 else 1
        self._last = k
        return GloveFrame(self.kp[i].astype(np.float32), np.full(21, self.conf[i], np.float32), int(self.seq[i]), int(self.dev[i]), t)

    def close(self) -> None:
        pass


def make_source(kind: str, side: str, address: str = "", sn: str = "", noise_mm: float = 0.0, dropouts: bool = True):
    """kind: wuji | pico | synthetic | replay:<GloveLog path>"""
    if kind == "wuji":
        return WujiGloveSource(side, address=address, sn=sn)
    if kind == "pico":
        from pico_hand import PicoHandSource
        return PicoHandSource(side)
    if kind == "synthetic":
        return SyntheticGloveSource(side, noise_mm=noise_mm, dropouts=dropouts)
    if kind.startswith("replay:"):
        return ReplayGloveSource(kind.split(":", 1)[1], side)
    raise ValueError(f"glove source {kind!r}: wuji | pico | synthetic | replay:<log>")


# -------------------------------------------------------------------------------------------------------------- log
COLUMNS = [("t", 1), ("wall", 1), ("tick", 1), ("side", 1), ("fresh", 1), ("held", 1), ("seq", 1), ("t_device_us", 1),
           ("conf", 1), ("pinch_w", 1), ("kp", 63), ("features", 6), ("fractions", 6), ("motors", 6), ("q", 11)]
WIDTH = sum(n for _, n in COLUMNS)


class GloveLog:
    def __init__(self, path, meta: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        Path(str(self.path) + ".json").write_text(json.dumps(dict(columns=COLUMNS, dtype="float64", side_ids={"left": 0, "right": 1},
                                                                  t="the run's clock (s): sim time on the bench, time.monotonic in teleop",
                                                                  wall="time.time()", meta=meta or {}), indent=1))
        self._f = open(self.path, "wb")

    def append(self, t: float, tick: int, side: str, frame: GloveFrame, fresh: bool, held: bool, out: dict) -> None:
        row = np.concatenate([[t, time.time(), tick, 0 if side == "left" else 1, fresh, held, frame.seq, frame.t_device_us,
                               float(np.mean(frame.conf)), out["pinch_w"]], frame.kp.reshape(-1), out["features"],
                              out["fractions"], out["motors"], out["q"]]).astype(np.float64)
        self._f.write(row.tobytes())
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def load_log(path) -> dict:
    a = np.fromfile(path, dtype=np.float64)
    a = a[: len(a) // WIDTH * WIDTH].reshape(-1, WIDTH)
    out, c = {}, 0
    for name, n in COLUMNS:
        out[name] = a[:, c] if n == 1 else a[:, c:c + n]
        c += n
    return out
