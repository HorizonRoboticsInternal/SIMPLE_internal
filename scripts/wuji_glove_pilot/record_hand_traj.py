#!/usr/bin/env python
"""Record a labelled hand trajectory for developing the retarget (2026-11-03): a fixed list of poses, each announced with
a 3 s countdown and recorded for HOLD_S seconds, both hands at once, from any pilot source (pico, wuji). Writes a GloveLog
(the same format as the bench and the teleop, replayable with bench.py --glove replay:<dir>/glove_log.bin) plus
labels.json with the time span of every pose, so the frames of a pose can be pulled out by name. No sim, no retarget.

    .venv/bin/python scripts/wuji_glove_pilot/record_hand_traj.py --glove pico --speak    # -> data/wuji_glove_pilot/traj_<time>/
    .venv/bin/python scripts/wuji_glove_pilot/record_hand_traj.py --glove pico --hold 3 --poses quick

Keep the hands in the headset's view (about chest height, palms toward you), do the same pose with both hands at once,
and move into each pose during the countdown, then hold still. --poses full (default) takes about 4 minutes (6 with
--speak, which reads every instruction and the countdown aloud so the headset can stay on). The headset must stay on:
it drops off the PC service a few minutes after being taken off (then every client gets zeros) and reconnects within
seconds of being put on; the start-up wait says which of the two it sees.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from wuji_glove import GloveFrame, GloveLog, make_source  # noqa: E402

HOLD_S = 4.0
WAIT_S = 60.0               # how long to wait for both hands to stream before giving up (the headset reconnects within seconds of being put on)


def say(text: str, on: bool) -> None:
    """Speak through speech-dispatcher (spd-say), non-blocking, so the operator can follow with the headset on."""
    if on:
        try:
            import subprocess
            subprocess.Popen(["spd-say", "-r", "-20", "--", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:                             # noqa: BLE001
            pass


def headset_up(srcs) -> bool | None:
    """True when the PICO SDK reports a headset pose (the headset is connected to the PC service), False when all zeros,
    None for other sources."""
    for src in srcs.values():
        xrt = getattr(src, "xrt", None)
        if xrt is not None and hasattr(xrt, "get_headset_pose"):
            return any(abs(v) > 0 for v in xrt.get_headset_pose())
    return None
# (label, instruction). "full" covers the extremes of every feature, each finger alone, every thumb-fingertip pinch and
# gradual closing; "quick" is the calibration subset.
POSES_FULL = [
    ("open", "OPEN: fingers straight and together, thumb relaxed beside the index finger"),
    ("spread", "SPREAD: fingers straight and spread wide, thumb out"),
    ("fist", "FIST: fingers fully curled, thumb bent across them"),
    ("open2", "OPEN again"),
    ("half_curl", "HALF CURL: all four fingers bent about half way (like holding a cup), thumb relaxed"),
    ("hook", "HOOK: fingers bent at the middle and tip joints only, knuckles straight (claw)"),
    ("mcp_only", "KNUCKLES ONLY: fingers straight but bent 90 deg at the knuckles (a flat table top)"),
    ("open3", "OPEN again"),
    ("index_curl", "INDEX ONLY curled, the other fingers straight"),
    ("middle_curl", "MIDDLE ONLY curled, the other fingers straight (as far as you can)"),
    ("ring_curl", "RING ONLY curled (the pinky may follow), the other fingers straight"),
    ("pinky_curl", "PINKY ONLY curled, the other fingers straight"),
    ("ring_pinky_curl", "RING + PINKY curled together, index and middle straight"),
    ("open4", "OPEN again"),
    ("thumb_out", "THUMB OUT: thumb pointing away from the palm as far as it goes (hitch-hiker), fingers straight"),
    ("thumb_across", "THUMB ACROSS: thumb flat across the palm toward the pinky base, fingers straight"),
    ("oppose", "OPPOSE: thumb tip on the base of the little finger, fingers straight"),
    ("thumb_bend", "THUMB BENT: thumb tip bent down toward its own base, the thumb not rotated"),
    ("open5", "OPEN again"),
    ("pinch_index", "PINCH INDEX: thumb tip on the index fingertip (an OK sign), other fingers relaxed"),
    ("pinch_middle", "PINCH MIDDLE: thumb tip on the middle fingertip"),
    ("pinch_ring", "PINCH RING: thumb tip on the ring fingertip"),
    ("pinch_pinky", "PINCH PINKY: thumb tip on the little fingertip"),
    ("pinch_index_wide", "WIDE INDEX PINCH: thumb and index tips touching with the index almost straight (a big O)"),
    ("open6", "OPEN again"),
    ("slow_close", "SLOW CLOSE: from open, close all fingers and the thumb slowly into a fist over the whole hold"),
    ("slow_open", "SLOW OPEN: open slowly back to the flat hand over the whole hold"),
    ("wave", "FINGER WAVE: curl and release index, middle, ring, pinky one after the other, keep going"),
]
POSES_QUICK = [p for p in POSES_FULL if p[0] in ("open", "fist", "oppose", "pinch_index", "pinch_middle", "pinch_ring", "pinch_pinky", "slow_close")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default="pico", help="pico | wuji")
    ap.add_argument("--sides", default="left,right")
    ap.add_argument("--hold", type=float, default=HOLD_S, help="seconds recorded per pose")
    ap.add_argument("--poses", default="full", choices=["full", "quick"])
    ap.add_argument("--speak", action="store_true", help="speak each instruction and the countdown (spd-say): follow with the headset on")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sides = a.sides.split(",")
    out = Path(a.out) if a.out else ROOT / "data" / "wuji_glove_pilot" / time.strftime("traj_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    poses = POSES_FULL if a.poses == "full" else POSES_QUICK
    srcs = {s: make_source(a.glove, s, dropouts=False) for s in sides}
    log = GloveLog(out / "glove_log.bin", meta=dict(source=a.glove, sides=sides, traj=True, hold_s=a.hold, poses=[p[0] for p in poses]))
    zero = dict(pinch_w=0.0, features=np.zeros(6), fractions=np.zeros(6), motors=np.zeros(6), q=np.zeros(11))
    labels, counts = [], {s: 0 for s in sides}
    t0 = time.monotonic()

    def record(seconds: float, tick0: int) -> int:
        tick, t_end = tick0, time.monotonic() + seconds
        while time.monotonic() < t_end:
            t = time.monotonic() - t0
            for s, src in srcs.items():
                fr: GloveFrame | None = src.latest(time.monotonic())
                if fr is not None:
                    log.append(t=t, tick=tick, side=s, frame=fr, fresh=True, held=False, out=zero)
                    counts[s] += 1
            tick += 1
            time.sleep(0.004)
        return tick

    print(f"[record] {a.glove} -> {out}; {len(poses)} poses x {a.hold:.0f} s, both hands at once. Move during the countdown, hold still while recording.", flush=True)
    try:
        tick, waited = 0, 0.0
        while True:                                                # wait for both hands to stream (up to WAIT_S), 2 s at a time
            tick = record(2.0, tick); waited += 2.0
            missing = [s for s in sides if counts[s] == 0]
            if not missing:
                break
            up = headset_up(srcs)
            why = ("the headset is NOT connected to the PC service (asleep or taken off? it drops off a few minutes after removal and reconnects "
                   "within seconds of being put on)" if up is False else
                   "the headset is connected but sends no hand: hands in front of it? hand tracking on? controllers put down?" if up else
                   "is the source streaming?")
            if waited >= WAIT_S:
                sys.exit(f"[record] no frames from the {' and '.join(missing)} hand in {WAIT_S:.0f} s: {why}")
            print(f"   waiting for the {' and '.join(missing)} hand ({waited:.0f} s): {why}", flush=True)
            if waited % 10 == 0:
                say("waiting for the hands", a.speak)
        print(f"[record] streams ok ({', '.join(f'{s} {counts[s]}' for s in sides)} frames so far)", flush=True)
        say("streams ok. starting.", a.speak); time.sleep(2.0 if a.speak else 0.0)
        for i, (label, text) in enumerate(poses):
            print(f"\n[{i + 1}/{len(poses)}] {text}", flush=True)
            say(text.split(":")[0].lower().replace("_", " ") + ". " + text.split(":", 1)[-1].strip(), a.speak)
            time.sleep(3.0 if a.speak else 0.0)                    # time to hear the instruction
            for k in (3, 2, 1):
                print(f"   {k}", flush=True); say(str(k), a.speak); time.sleep(1.0)
            t_a = time.monotonic() - t0
            print(f"   recording {a.hold:.0f} s ...", flush=True); say("hold", a.speak)
            tick = record(a.hold, tick)
            say("ok", a.speak)
            labels.append(dict(label=label, text=text, t0=round(t_a, 3), t1=round(time.monotonic() - t0, 3)))
            (out / "labels.json").write_text(json.dumps(dict(hold_s=a.hold, glove=a.glove, sides=sides, poses=labels), indent=1))
        print(f"\n[record] done: {counts} frames, {len(labels)} poses -> {out}/glove_log.bin + labels.json", flush=True)
        say("done. all poses recorded.", a.speak)
    except KeyboardInterrupt:
        print(f"\n[record] stopped after {len(labels)} poses; what was recorded is in {out}", flush=True)
    finally:                                                       # the SDK must be closed on every exit, or its thread aborts the process
        log.close()
        for src in srcs.values():
            src.close()


if __name__ == "__main__":
    main()
