#!/usr/bin/env python
"""Wuji Glove pilot, stage 2 (2026-10-27): the HoloMotion v1.4 sim teleop, unchanged, with the gloved hand's Revo 2 target
taken from the glove instead of the PICO grip / trigger.

Runs simple.cli.teleop_holomotion_v14 in this process after wrapping HoloMotionV14Agent.get_action: the agent computes its
command as always (body, buttons, the PICO grip poses for both hands), then the gloved side's left_hand_q / right_hand_q
(11 Revo 2 joints, MJCF order) is replaced by the glove target. Nothing in src/ is edited; the recorder, the exact log and both
replays (physics and --mode action) take the hand target from that command, so a recording made this way replays as usual.
A hand without a glove keeps the PICO grip / trigger poses. Needs --robot revo2 (with the Dex3 robot the command passes through).

Launch through scripts/wuji_glove_pilot/teleop_with_glove.sh (it uses scripts/teleop_holomotion_v14.sh with this file as the
SIMPLE python), or directly with a publisher already running:
    WUJI_GLOVE=synthetic .venv/bin/python -u scripts/wuji_glove_pilot/teleop_with_glove.py --scene kitchen --robot revo2 ...

Glove settings come from the environment (the CLI arguments all belong to the teleop CLI):
  WUJI_GLOVE          wuji | pico | synthetic | replay:<glove_log.bin>       (required; pico = the headset's hand tracking)
  WUJI_GLOVE_SIDES    right | left | left,right                       (default right)
  WUJI_GLOVE_ADDRESS_LEFT / _RIGHT, WUJI_GLOVE_SN_LEFT / _RIGHT       (default: the factory address of each side)
  WUJI_GLOVE_CALIB_DIR  dir with calib_<side>.json                    (default data/wuji_glove_pilot; missing = built-in defaults)
  WUJI_GLOVE_PINCH_BLEND (1) / WUJI_GLOVE_SMOOTH_HZ (10) / WUJI_GLOVE_THUMB_ROT_MIN (0) / WUJI_GLOVE_MIN_CONF (0.5)
  WUJI_GLOVE_LOG      log file (default <--save-dir>/wuji_glove/glove_log_<time>.bin, beside the dataset, never inside it;
                      without --save-dir data/wuji_glove_pilot/teleop_<time>/glove_log.bin)
  WUJI_GLOVE_KEYS     1 = the keyboard stands in for the PICO controllers' buttons and sticks (both hands gloved: nothing to
                      hold); with no controller sample a neutral one is supplied so the teleop starts. Keys: pico_keys.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from revo2_retarget import Calibration, GloveHand, Revo2Retargeter  # noqa: E402
from wuji_glove import GloveLog, make_source  # noqa: E402

STATUS_EVERY_S = 5.0


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _arg_value(args: list[str], flag: str) -> str:
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return ""


def build_hands(args: list[str]) -> dict:
    kind = _env("WUJI_GLOVE")
    if not kind:
        sys.exit("[wuji_glove] set WUJI_GLOVE=wuji | pico | synthetic | replay:<glove_log.bin>")
    sides = [s.strip() for s in _env("WUJI_GLOVE_SIDES", "right").split(",") if s.strip()]
    if not sides or any(s not in ("left", "right") for s in sides):
        sys.exit(f"[wuji_glove] WUJI_GLOVE_SIDES={_env('WUJI_GLOVE_SIDES')!r}: right, left or left,right")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = _arg_value(args, "--save-dir")
    log_path = Path(_env("WUJI_GLOVE_LOG") or (Path(save_dir) / "wuji_glove" / f"glove_log_{stamp}.bin" if save_dir
                                                else ROOT / "data" / "wuji_glove_pilot" / f"teleop_{stamp}" / "glove_log.bin"))
    calib_dir = Path(_env("WUJI_GLOVE_CALIB_DIR") or ROOT / "data" / "wuji_glove_pilot")
    settings = dict(pinch_blend=_env("WUJI_GLOVE_PINCH_BLEND", "1") not in ("0", "false", "no"),
                    smooth_hz=float(_env("WUJI_GLOVE_SMOOTH_HZ", "10")), thumb_rot_min=float(_env("WUJI_GLOVE_THUMB_ROT_MIN", "0")))
    min_conf = float(_env("WUJI_GLOVE_MIN_CONF", "0.5"))
    log = GloveLog(log_path, meta=dict(source=kind, sides=sides, teleop_args=args, settings=settings, min_conf=min_conf))
    hands = {}
    for s in sides:
        calib = Calibration.load(calib_dir / (f"calib_{s}_pico.json" if kind == "pico" else f"calib_{s}.json")) if kind != "synthetic" else Calibration()
        src = make_source(kind, s, address=_env(f"WUJI_GLOVE_ADDRESS_{s.upper()}"), sn=_env(f"WUJI_GLOVE_SN_{s.upper()}"))
        hands[s] = GloveHand(s, src, Revo2Retargeter(calib, side=s, **settings), min_conf=min_conf, log=log)
        print(f"[wuji_glove] {s} hand <- {kind} glove (calibration {calib.path or 'built-in defaults'}; {settings})", flush=True)
    print(f"[wuji_glove] log {log_path}", flush=True)
    return hands


def install(hands: dict, keys=None) -> None:
    """Wrap HoloMotionV14Agent.get_action: the gloved hands' targets replace the PICO grip targets in the returned command.
    With keys, HoloMotionV14Agent.pico (the controller sample the agent and the CLI's combos read each tick) gets the
    keyboard's buttons and sticks on top of the real sample, or on a neutral one while no controller sample arrives."""
    from simple.agents.holomotion_v14_agent import HoloMotionV14Agent
    if keys is not None:
        original_pico = HoloMotionV14Agent.pico
        HoloMotionV14Agent.pico = lambda self: keys.apply(original_pico(self))
    original = HoloMotionV14Agent.get_action
    state = dict(tick=0, warned=False, next_status=time.monotonic() + STATUS_EVERY_S)

    def get_action(self, observation, instruction=None, **kwargs):
        cmd = original(self, observation, instruction, **kwargs)
        if not self._is_revo2():
            if not state["warned"]:
                print("[wuji_glove] the robot is not the Revo 2 one (--robot revo2): glove input ignored", flush=True)
                state["warned"] = True
            return cmd
        now = time.monotonic()
        for s, h in hands.items():
            r = h.update(now, state["tick"])
            if r is None:                                      # no glove frame yet: the PICO grip target stays
                continue
            q = np.asarray(r["q"], dtype=np.float32)
            cmd.parameters[f"{s}_hand_q"] = q.copy()
            setattr(self, f"_{s}_hand", q.copy())
        state["tick"] += 1
        if now >= state["next_status"]:
            state["next_status"] = now + STATUS_EVERY_S
            for s, h in hands.items():
                g = h.summary()
                last = h.last
                print(f"[wuji_glove] {s}: {g['glove_rate_hz']} Hz, held {g['held']} ticks, low-conf {g['low_conf']}, "
                      f"motors {np.round(last['motors'], 2).tolist() if last else '-'}", flush=True)
        return cmd

    HoloMotionV14Agent.get_action = get_action


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["-u"]:                                 # called as SIMPLE_PY by scripts/teleop_holomotion_v14.sh
        args = args[1:]
    if args[:2] == ["-m", "simple.cli.teleop_holomotion_v14"]:
        args = args[2:]
    sys.stdout.reconfigure(line_buffering=True)
    hands = build_hands(args)
    keys = None
    if _env("WUJI_GLOVE_KEYS", "0") not in ("", "0", "false", "no"):
        from pico_keys import PicoKeys
        keys = PicoKeys.start()                            # None when stdin is not a terminal
    install(hands, keys)
    from simple.cli import teleop_holomotion_v14 as T
    sys.argv = ["teleop_holomotion_v14"] + args
    T.cli()


if __name__ == "__main__":
    main()
