#!/usr/bin/env python
"""PICO check for the glove teleop (2026-11-03): does PICO body tracking still follow the arms when the controllers are
put down and the operator wears the Wuji gloves? The teleop takes the wrist poses from the headset's body tracking and
only the buttons from the controllers, so this is the one thing a two-glove recording depends on.

Run in the publisher env with the headset on (the XRoboToolkit PC service running, as for the teleop):

    ~/miniconda3/envs/holomotion_teleop/bin/python scripts/wuji_glove_pilot/pico_hands_probe.py --seconds 60

Twice a second it prints: body data available, the controllers' active flags from the v1.4.1 control sample, the
headset's own hand-tracking flags, and the two wrist positions of the 24-joint body pose (SMPL order: 20 = left wrist,
21 = right wrist) with how far each moved over the last 2 s. Put the controllers down, wave the arms: the wrists must
keep moving. The summary at the end says how many body frames came in and whether the wrists moved.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "holomotion_v14"))     # the vendored v1.4 packages (as the publisher)

L_WRIST, R_WRIST = 20, 21


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()

    import xrobotoolkit_sdk as xrt
    try:
        from humanoid_policy.v141_pico_control import PicoControlNotReady, read_pico_control_sample
    except Exception as e:                                        # noqa: BLE001
        print(f"[probe] v1.4.1 control-sample reader unavailable ({e}); controller flags not shown")
        PicoControlNotReady, read_pico_control_sample = Exception, None

    xrt.init()
    print("[probe] XRoboToolkit initialized; put the controllers down and wave the arms", flush=True)
    t0 = time.monotonic()
    last_stamp = None
    n_body = 0
    hist = collections.deque()                                   # (t, left_wrist, right_wrist)
    travel = {"left": 0.0, "right": 0.0}
    ctrl_active = collections.Counter()
    next_print = t0
    try:
        while time.monotonic() - t0 < a.seconds:
            now = time.monotonic()
            sample = None
            if read_pico_control_sample is not None:
                try:
                    sample = read_pico_control_sample(xrt)
                    ctrl_active[(int(sample.get("l_active", 0)), int(sample.get("r_active", 0)))] += 1
                except PicoControlNotReady:
                    ctrl_active["not ready"] += 1
            body = bool(xrt.is_body_data_available())
            lw = rw = None
            if body:
                stamp = int(xrt.get_time_stamp_ns())
                if stamp > 0 and stamp != last_stamp:
                    last_stamp = stamp
                    poses = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64).reshape(24, 7)
                    lw, rw = poses[L_WRIST, :3], poses[R_WRIST, :3]
                    if hist:
                        travel["left"] += float(np.linalg.norm(lw - hist[-1][1]))
                        travel["right"] += float(np.linalg.norm(rw - hist[-1][2]))
                    hist.append((now, lw, rw))
                    while hist and now - hist[0][0] > 2.0:
                        hist.popleft()
                    n_body += 1
            if now >= next_print:
                next_print = now + 0.5
                try:
                    hands = f"hand-tracking L={int(bool(xrt.get_left_hand_is_active()))} R={int(bool(xrt.get_right_hand_is_active()))}"
                except Exception:                                 # noqa: BLE001
                    hands = "hand-tracking n/a"
                ctrl = "-" if sample is None else f"L={int(sample.get('l_active', 0))} R={int(sample.get('r_active', 0))} bits={int(sample.get('button_bits', 0))}"
                if hist:
                    _, l0, r0 = hist[0]; _, l1, r1 = hist[-1]
                    move = f"moved 2 s: L {np.linalg.norm(l1 - l0) * 100:5.1f} cm  R {np.linalg.norm(r1 - r0) * 100:5.1f} cm"
                    pos = f"L wrist {np.round(l1, 2).tolist()}  R wrist {np.round(r1, 2).tolist()}"
                else:
                    move, pos = "no body frame yet", ""
                print(f"[{now - t0:5.1f}s] body={int(body)} frames={n_body:5d}  controllers {ctrl}  {hands}  {pos}  {move}", flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            xrt.close()
        except Exception:                                         # noqa: BLE001
            pass
    dt = time.monotonic() - t0
    print(f"\n[probe] {dt:.0f} s: {n_body} body frames ({n_body / dt:.0f} Hz); wrist travel L {travel['left']:.2f} m, R {travel['right']:.2f} m; "
          f"controller active flags seen: {dict(ctrl_active)}")
    print("[probe] PASS: body frames kept coming and both wrists travelled while the controllers were down."
          if n_body > 0 and min(travel.values()) > 0.3 else
          "[probe] CHECK: little or no wrist travel, or no body frames: body tracking needs the controllers (or was not running).")


if __name__ == "__main__":
    main()
