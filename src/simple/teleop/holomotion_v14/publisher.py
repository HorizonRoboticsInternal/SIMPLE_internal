#!/usr/bin/env python3
"""
HoloMotion v1.4 reference publisher for SIMPLE teleop (run in the ``holomotion_teleop`` conda env).

    --source pico       PICO body tracking -> SMPL -> GMR (the robot's PicoSmplGmrRetargeter) -> obs65, plus the
                        controller sample (v1.4.1 read_pico_control_sample) -> pico_ctrl. Needs the XRoboToolkit
                        PC service running with the headset connected and body tracking on.
    --source synthetic  no headset: a standing reference (the policy's default pose) and scripted controller input,
                        for testing the sim side, e.g.
                          --press 1:L3 --press 5:A --stick 7:12:left_y=0.8 --press 14:B --stick 16:18:right_x=0.5

Both publish on one ZMQ PUB socket (default tcp://*:6001, the robot node's endpoint); the SIMPLE agent subscribes.

    ~/miniconda3/envs/holomotion_teleop/bin/python src/simple/teleop/holomotion_v14/publisher.py --source pico
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("holomotion_v14_wire", HERE / "wire.py")
wire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wire)                         # also puts third_party/holomotion_v14 on sys.path
VENDOR = Path(wire._VENDOR)


def _default_smpl_dir() -> str:
    for c in (os.environ.get("HOLOMOTION_SMPL_DIR", ""), Path.home() / "wrk/holomotion_v1.4/assets/smpl",
              Path.home() / "wrk/HoloMotion_TELEOP/assets/smpl", Path.home() / "wrk/robot-locomanip/models/smpl"):
        if c and (Path(c) / "SMPL_NEUTRAL.pkl").is_file():
            return str(c)
    return ""


def _standing_obs65(root_z: float) -> np.ndarray:
    import yaml
    y = yaml.safe_load(open(VENDOR / "config" / "g1_29dof_holomotion.yaml"))
    dof = np.array([float(y["default_joint_angles"][n]) for n in y["complete_dof_order"]], dtype=np.float32)
    return np.concatenate([dof, np.zeros(29, np.float32), np.array([0.0, 0.0, root_z], np.float32),
                           np.array([1.0, 0.0, 0.0, 0.0], np.float32)])


class BodyTrackingWatch:
    """Says when the headset's body tracking stops and when it is back (2026-11-03: a teleop session froze for minutes with the
    publisher logging 'body frames 0.0 Hz' and the SDK 'device missing' while the controller samples kept coming; nothing said
    why). `body(t)` per body frame, `pico(t)` per controller sample, `report(t, ...)` every 5 s returns the log line."""

    def __init__(self, lost_after_s: float = 2.0) -> None:
        self.lost_after_s = lost_after_s
        self.last_body = self.last_pico = self.lost_since = None
        self.n_lost = 0

    def body(self, t: float) -> str:
        msg = ""
        if self.lost_since is not None:
            msg = f"[v14 publisher] {time.strftime('%H:%M:%S')} body tracking RESUMED after {t - self.lost_since:.0f} s"
            print(msg, flush=True)
            self.lost_since = None
        self.last_body = t
        return msg

    def pico(self, t: float) -> None:
        self.last_pico = t

    def report(self, t: float, n_body: int, fps: float, frame: int, hands: str = "") -> str:
        if self.last_body is None and frame == 0 and self.last_pico is not None:
            return (f"[v14 publisher] {time.strftime('%H:%M:%S')} NO BODY TRACKING YET (controller samples flowing{hands}): the headset is on and "
                    f"connected but sends no body pose -- the XRoboToolkit app must be in its full-body mode with the trackers calibrated, the headset on the head")
        if self.last_body is not None and t - self.last_body > self.lost_after_s:
            if self.lost_since is None:
                self.lost_since, self.n_lost = self.last_body, self.n_lost + 1
            pico = "still arrive" if self.last_pico is not None and t - self.last_pico < 2.0 else "stopped too"
            return (f"[v14 publisher] {time.strftime('%H:%M:%S')} NO BODY TRACKING for {t - self.last_body:.0f} s (controller samples {pico}; "
                    f"last body frame {frame - 1}): the headset is off the head or asleep, the app is in the background, the Wi-Fi link stalled, "
                    f"or another XRoboToolkit client changed the tracking mode; the teleop holds until frames return")
        return f"[v14 publisher] body frames {n_body / 5.0:.1f} Hz (PICO {fps:.1f} Hz), total {frame}{hands}"


class Script:
    """Timed synthetic controller input: --press T:BUTTON (held 0.2 s), --stick T0:T1:axis=value."""

    AXES = {"left_x": ("left_axis", 0), "left_y": ("left_axis", 1), "right_x": ("right_axis", 0),
            "right_y": ("right_axis", 1), "left_grip": ("left_grip", None), "right_grip": ("right_grip", None),
            "left_trigger": ("left_trigger", None), "right_trigger": ("right_trigger", None)}

    def __init__(self, presses: list[str], sticks: list[str]) -> None:
        self.presses = []
        for p in presses:
            t, names = p.split(":", 1)
            self.presses.append((float(t), [n.strip() for n in names.split("+")]))
        self.sticks = []
        for s in sticks:
            t0, t1, assign = s.split(":", 2)
            name, value = assign.split("=")
            self.sticks.append((float(t0), float(t1), name.strip(), float(value)))

    def sample(self, t: float) -> dict:
        s = dict(left_axis=[0.0, 0.0], right_axis=[0.0, 0.0], left_trigger=0.0, right_trigger=0.0,
                 left_grip=0.0, right_grip=0.0, button_bits=0, timestamp_ns=time.time_ns())
        for tp, names in self.presses:
            if tp <= t < tp + 0.2:
                for n in names:
                    s["button_bits"] |= wire.BUTTONS[n]
        for t0, t1, name, value in self.sticks:
            if t0 <= t < t1:
                key, idx = self.AXES[name]
                if idx is None:
                    s[key] = value
                else:
                    s[key][idx] = value
        return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("pico", "synthetic"), default="pico")
    ap.add_argument("--hands", action="store_true", help="also publish the headset's hand tracking (both 26 x 7 joint arrays + isActive flags, topic pico_hands) for the teleop's --hand-source skeleton; synthetic: the pilot's scripted human hands")
    ap.add_argument("--uri", default="tcp://*:6001")
    ap.add_argument("--hz", type=float, default=70.0, help="loop rate (the robot node's loop_hz)")
    ap.add_argument("--smpl-dir", default=_default_smpl_dir())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--press", action="append", default=[], help="synthetic: T:BUTTON[+BUTTON]  (L3 X A R3 B Y left_menu)")
    ap.add_argument("--stick", action="append", default=[], help="synthetic: T0:T1:axis=value  (left_x left_y right_x ... grips, triggers)")
    ap.add_argument("--root-z", type=float, default=0.78, help="synthetic: reference pelvis height")
    ap.add_argument("--motion", type=float, default=0.0, help="synthetic: from this time (s) on, move the reference (slow sine on the shoulders/elbows, a slow root drift) so the frames differ")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = run until Ctrl-C)")
    a = ap.parse_args()

    pub = wire.ReferencePublisher(a.uri)
    print(f"[v14 publisher] {a.source} -> {a.uri} (topics obs65, pico_ctrl) at {a.hz:g} Hz", flush=True)
    period = 1.0 / a.hz
    t_start = time.monotonic()
    frame = 0
    n_body = 0
    t_log = t_start + 5.0

    if a.source == "synthetic":
        script = Script(a.press, a.stick)
        if a.hands:
            _hs = importlib.util.spec_from_file_location("holomotion_v14_hand_synth", HERE / "hand_synth.py")
            hand_synth = importlib.util.module_from_spec(_hs); _hs.loader.exec_module(hand_synth)
            synthetic_hands = hand_synth.synthetic_hands
        obs = _standing_obs65(a.root_z)
        try:
            base = obs.copy()
            while not a.duration or time.monotonic() - t_start < a.duration:
                t = time.monotonic() - t_start
                pub.send_pico(script.sample(t))
                if a.hands:
                    hands, active = synthetic_hands(t)
                    pub.send_pico_hands(hands, active)
                if a.motion and t >= a.motion:                       # a moving reference: arms swing, root drifts (velocities consistent)
                    tt = t - a.motion; w = 2 * np.pi * 0.4
                    obs = base.copy()
                    for j, amp in ((15, 0.35), (22, -0.35), (18, 0.25), (25, 0.25)):     # shoulder pitch L/R, elbow L/R (complete_dof_order)
                        obs[j] = base[j] + amp * np.sin(w * tt); obs[29 + j] = amp * w * np.cos(w * tt)
                    obs[58] = base[58] + 0.05 * tt; obs[59] = base[59] + 0.02 * np.sin(0.5 * tt)
                pub.send_obs65(obs, frame, timestamp_ns=time.time_ns(), pico_dt=period, pico_fps=a.hz)
                frame += 1
                time.sleep(max(0.0, t_start + frame * period - time.monotonic()))
        except KeyboardInterrupt:
            pass
        pub.close()
        return

    import xrobotoolkit_sdk as xrt
    from holomotion_teleop_ros2.converter import PicoSmplGmrRetargeter
    from humanoid_policy.v141_pico_control import PicoControlNotReady, read_pico_control_sample

    if not a.smpl_dir:
        sys.exit("SMPL_NEUTRAL.pkl not found: pass --smpl-dir or set HOLOMOTION_SMPL_DIR")
    print(f"[v14 publisher] initializing SMPL/GMR (smpl_dir={a.smpl_dir}, device={a.device})", flush=True)
    retargeter = PicoSmplGmrRetargeter(smpl_asset_dir=a.smpl_dir, src_human="smplx", tgt_robot="unitree_g1",
                                       device=a.device)
    xrt.init()
    print("[v14 publisher] XRoboToolkit SDK initialized; waiting for body tracking ...", flush=True)
    last_stamp = None
    last_body_ns = None
    fps_ema = 0.0
    watch = BodyTrackingWatch()
    hands_warned = False
    n_hand_active = [0, 0]; n_hand_polls = 0                                  # hand tracking presence for the 5 s report
    if a.hands:
        print("[v14 publisher] publishing the headset's hand tracking too (pico_hands; hands track while the controllers are put down)", flush=True)
    try:
        while not a.duration or time.monotonic() - t_start < a.duration:
            tick = time.monotonic()
            try:
                pub.send_pico(read_pico_control_sample(xrt))
                watch.pico(tick)
            except PicoControlNotReady:
                pass
            if a.hands:                                                      # the headset's hand tracking, as the SDK gives it (zeros before any data)
                try:
                    act = [int(xrt.get_left_hand_is_active()), int(xrt.get_right_hand_is_active())]
                    pub.send_pico_hands(np.stack([np.asarray(xrt.get_left_hand_tracking_state(), dtype=np.float32).reshape(26, 7),
                                                  np.asarray(xrt.get_right_hand_tracking_state(), dtype=np.float32).reshape(26, 7)]), act)
                    n_hand_active[0] += act[0]; n_hand_active[1] += act[1]; n_hand_polls += 1
                except Exception as exc:                                     # noqa: BLE001  an SDK without hand tracking: say so once
                    if not hands_warned:
                        print(f"[v14 publisher] hand tracking not available from the SDK: {exc}", flush=True); hands_warned = True
            if xrt.is_body_data_available():
                stamp = int(xrt.get_time_stamp_ns())
                if stamp > 0 and stamp != last_stamp:
                    last_stamp = stamp
                    poses = np.asarray(xrt.get_body_joints_pose(), dtype=np.float32).reshape(24, 7)
                    try:
                        _, _, _, _, latest_obs = retargeter.retarget(poses, time.time())
                    except Exception as exc:                  # the robot node logs and drops the frame, too
                        print(f"[v14 publisher] retarget failed: {exc}", flush=True)
                    else:
                        dt = 0.0 if last_body_ns is None else (stamp - last_body_ns) * 1e-9
                        if dt > 0:
                            fps_ema = 1.0 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
                        last_body_ns = stamp
                        pub.send_obs65(latest_obs, frame, timestamp_ns=stamp, pico_dt=dt, pico_fps=fps_ema)
                        frame += 1
                        n_body += 1
                        watch.body(tick)
            if tick >= t_log:
                hands_txt = "" if not a.hands else (f"; hands active L {100 * n_hand_active[0] / n_hand_polls:.0f} % R {100 * n_hand_active[1] / n_hand_polls:.0f} %"
                                                    if n_hand_polls else "; hands: no polls")
                print(watch.report(tick, n_body, fps_ema, frame, hands_txt), flush=True)
                n_body = 0; n_hand_active = [0, 0]; n_hand_polls = 0
                t_log = tick + 5.0
            time.sleep(max(0.0, tick + period - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        try:
            xrt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
