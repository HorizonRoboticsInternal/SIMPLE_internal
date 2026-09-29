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
    ap.add_argument("--uri", default="tcp://*:6001")
    ap.add_argument("--hz", type=float, default=70.0, help="loop rate (the robot node's loop_hz)")
    ap.add_argument("--smpl-dir", default=_default_smpl_dir())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--press", action="append", default=[], help="synthetic: T:BUTTON[+BUTTON]  (L3 X A R3 B Y left_menu)")
    ap.add_argument("--stick", action="append", default=[], help="synthetic: T0:T1:axis=value  (left_x left_y right_x ... grips, triggers)")
    ap.add_argument("--root-z", type=float, default=0.78, help="synthetic: reference pelvis height")
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
        obs = _standing_obs65(a.root_z)
        try:
            while not a.duration or time.monotonic() - t_start < a.duration:
                t = time.monotonic() - t_start
                pub.send_pico(script.sample(t))
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
    try:
        while not a.duration or time.monotonic() - t_start < a.duration:
            tick = time.monotonic()
            try:
                pub.send_pico(read_pico_control_sample(xrt))
            except PicoControlNotReady:
                pass
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
            if tick >= t_log:
                print(f"[v14 publisher] body frames {n_body / 5.0:.1f} Hz (PICO {fps_ema:.1f} Hz), total {frame}", flush=True)
                n_body = 0
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
