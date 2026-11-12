#!/usr/bin/env python
"""Drive the REAL BrainCo Revo 2 hands from the PICO headset's hand tracking, with the same retarget as the simulation
(2026-11-04): pico_hand.PicoHandSource -> revo2_retarget.Revo2Retargeter (bone angles, calibration, PINCH_KNOTS, low-pass,
hold on dropout) -> the hands' six motors over Modbus RTU (bc-stark-sdk 2.0.9, FT4232H adapter).

    .venv/bin/python scripts/wuji_glove_pilot/real_revo2_pico.py status                 # find the hands, print state; moves nothing
    .venv/bin/python scripts/wuji_glove_pilot/real_revo2_pico.py run --dry-run          # PICO -> targets printed, nothing sent
    .venv/bin/python scripts/wuji_glove_pilot/real_revo2_pico.py run                    # both hands follow the operator; Ctrl-C opens them and stops
    .venv/bin/python scripts/wuji_glove_pilot/real_revo2_pico.py run --sides right --rate 30 --slew 1500
    .venv/bin/python scripts/wuji_glove_pilot/real_revo2_pico.py open                   # open both hands (one command), then exit

Motor map: the retarget's 6 motors (thumb rotation, thumb bend, index, middle, ring, pinky; rad, 0 = open) -> the SDK's
finger order [Thumb (flex), ThumbAux (rotation), Index, Middle, Ring, Pinky] as 0..1000 = motor / its limit
(1.047 thumb flex, 1.518 thumb rotation, 1.466 fingers; the model's limits equal BrainCo's spec: 59 / 90 / 81 deg).
0 = open, 1000 = closed, as in the SDK demos ("Open" = all 0).

Safety: the loop never sends a target that moves a finger faster than --slew units/s; with no PICO frame for 0.5 s the
last target is held (never sprung open: a held object would drop); Ctrl-C and errors send "open" and close the ports
(--no-open-on-exit keeps the last pose). The hands are found by serial number (BCXTL... left, BCXTR... right) on
/dev/ttyUSB*; --left / --right PORT:SLAVE pins them. Headset on, controllers down, hands in the headset's view.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from revo2_retarget import MOTOR_UPPER, Calibration, GloveHand, Revo2Retargeter  # noqa: E402
from wuji_glove import GloveLog  # noqa: E402

CALIB_DIR = ROOT / "data" / "wuji_glove_pilot"
SDK_ORDER = ["thumb_flex", "thumb_aux", "index", "middle", "ring", "pinky"]
MOTOR_TO_SDK = [1, 0, 2, 3, 4, 5]                   # retarget motor index for each SDK finger
SIDE_LETTER = {"left": "L", "right": "R"}          # BCXT<L|R>...
STALE_S = 0.5


def motors_to_sdk(motors: np.ndarray) -> list[int]:
    frac = np.clip(np.asarray(motors, dtype=np.float64)[MOTOR_TO_SDK] / MOTOR_UPPER[MOTOR_TO_SDK], 0.0, 1.0)
    return [int(round(1000 * f)) for f in frac]


# ------------------------------------------------------------------------------------------------------------ hands
class RealHand:
    """One Revo 2 over Modbus: open, read state, send positions with a slew limit, close."""

    def __init__(self, side: str, port: str, slave_id: int, baudrate) -> None:
        self.side, self.port, self.slave, self.baudrate = side, port, int(slave_id), baudrate
        self.ctx = None
        self.sent = None
        self.n_sent = 0

    async def open(self):
        from bc_stark_sdk import main_mod as sdk
        self.ctx = await sdk.modbus_open(self.port, self.baudrate)
        info = await self.ctx.get_device_info(self.slave)
        return info

    async def status(self) -> dict:
        st = await self.ctx.get_motor_status(self.slave)
        return dict(positions=list(st.positions), speeds=list(st.speeds), currents=list(st.currents))

    async def send(self, target: list[int], slew: float, dt: float) -> list[int]:
        """Move toward `target` by at most slew * dt units per finger; returns what was sent."""
        if self.sent is None:
            self.sent = await self._positions_now()
        step = max(1, int(round(slew * dt)))
        out = [int(np.clip(t, s - step, s + step)) for s, t in zip(self.sent, target)]
        await self.ctx.set_finger_positions(self.slave, out)
        self.sent = out; self.n_sent += 1
        return out

    async def _positions_now(self) -> list[int]:
        try:
            return [int(p) for p in (await self.status())["positions"]]
        except Exception:                             # noqa: BLE001
            return [0] * 6

    async def open_hand(self) -> None:
        await self.ctx.set_finger_positions(self.slave, [0] * 6)
        self.sent = [0] * 6

    async def close(self) -> None:
        from bc_stark_sdk import main_mod as sdk
        if self.ctx is not None:
            try:
                await sdk.modbus_close(self.ctx)
            except Exception:                         # noqa: BLE001
                pass
            self.ctx = None


async def detect(sides: list[str], pins: dict[str, str]) -> dict[str, RealHand]:
    """The hands on /dev/ttyUSB* by serial number (or pinned PORT:SLAVE), as RealHand objects (not opened)."""
    from bc_stark_sdk import main_mod as sdk
    found: dict[str, RealHand] = {}
    for s in sides:
        if pins.get(s):
            port, slave = pins[s].split(":")
            found[s] = RealHand(s, port, int(slave, 0), sdk.Baudrate.Baud460800)
    ports = [p for p in sorted(glob.glob("/dev/ttyUSB*")) if p not in {h.port for h in found.values()}]
    for port in ports:
        if all(s in found for s in sides):
            break
        try:
            devs = await asyncio.wait_for(sdk.auto_detect(scan_all=True, port=port, protocol="Modbus"), timeout=20)
        except Exception as e:                        # noqa: BLE001
            print(f"[revo2] {port}: scan failed ({e})", flush=True); continue
        for d in devs or []:                          # DetectedDevice: port_name, slave_id, baudrate, serial_number (BCXTR... / BCXTL...), sku_type (MediumRight)
            sn, sku = str(getattr(d, "serial_number", "") or ""), str(getattr(d, "sku_type", "") or "")
            m = re.search(r"BCXT([LR])", sn)
            letter = m.group(1) if m else ("R" if "Right" in sku else "L" if "Left" in sku else "")
            side = next((s for s in sides if s not in found and letter == SIDE_LETTER[s]), None)
            if side:
                found[side] = RealHand(side, getattr(d, "port_name", port), int(d.slave_id), d.baudrate)
                print(f"[revo2] {side} hand: {found[side].port} slave {int(d.slave_id)} ({sn}, {sku}, {getattr(d, 'hardware_type', '')}, fw {getattr(d, 'firmware_version', '')})", flush=True)
            else:
                print(f"[revo2] {port}: device {sn or d} not matched to a side", flush=True)
    missing = [s for s in sides if s not in found]
    if missing:
        print(f"[revo2] not found: {missing} (adapter plugged in? hands powered? /dev/ttyUSB*: {sorted(glob.glob('/dev/ttyUSB*'))})", flush=True)
    return found


async def open_all(hands: dict[str, RealHand]) -> None:
    for h in hands.values():
        info = await h.open()
        print(f"[revo2] {h.side}: opened {h.port} slave {h.slave}; firmware {getattr(info, 'firmware_version', '?')}, "
              f"hardware {getattr(info, 'hardware_type', '?')}, sn {getattr(info, 'serial_number', '?')}", flush=True)


async def close_all(hands: dict[str, RealHand], open_first: bool) -> None:
    for h in hands.values():
        if h.ctx is None:
            continue
        if open_first:
            try:
                await h.open_hand()
            except Exception as e:                    # noqa: BLE001
                print(f"[revo2] {h.side}: open-on-exit failed ({e})", flush=True)
        await h.close()


# -------------------------------------------------------------------------------------------------------------- modes
async def cmd_status(a) -> None:
    hands = await detect(a.sides, dict(left=a.left, right=a.right))
    await open_all(hands)
    try:
        for h in hands.values():
            st = await h.status()
            print(f"[revo2] {h.side}: positions {st['positions']} (0 open .. 1000 closed, {SDK_ORDER}); speeds {st['speeds']}; currents {st['currents']}", flush=True)
    finally:
        await close_all(hands, open_first=False)


async def cmd_open(a) -> None:
    hands = await detect(a.sides, dict(left=a.left, right=a.right))
    await open_all(hands)
    try:
        for h in hands.values():
            await h.open_hand(); print(f"[revo2] {h.side}: open sent", flush=True)
        await asyncio.sleep(1.5)
        for h in hands.values():
            print(f"[revo2] {h.side}: positions now {(await h.status())['positions']}", flush=True)
    finally:
        await close_all(hands, open_first=False)


async def cmd_run(a) -> None:
    from pico_hand import PicoHandSource
    hands = {} if a.dry_run else await detect(a.sides, dict(left=a.left, right=a.right))
    if not a.dry_run and set(hands) != set(a.sides):
        sys.exit("[revo2] not all requested hands found; use --dry-run to test the PICO side alone")
    out = CALIB_DIR / time.strftime("real_%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    log = GloveLog(out / "glove_log.bin", meta=dict(source="pico", sides=a.sides, real_revo2=True, dry_run=a.dry_run, rate_hz=a.rate, slew=a.slew))
    gloves = {}
    for s in a.sides:
        calib = Calibration.load(CALIB_DIR / f"calib_{s}_pico.json")
        rt = Revo2Retargeter(calib, pinch_blend=True, smooth_hz=a.smooth_hz, side=s)
        gloves[s] = GloveHand(s, PicoHandSource(s), rt, min_conf=a.min_conf, stale_s=STALE_S, log=log)
        print(f"[revo2] {s}: PICO -> retarget (calibration {calib.path or 'built-in defaults'})", flush=True)
    if not a.dry_run:
        await open_all(hands)
    dt = 1.0 / a.rate
    print(f"[revo2] running at {a.rate:g} Hz, slew {a.slew:g} units/s{' (DRY RUN: nothing sent)' if a.dry_run else ''}; Ctrl-C to stop"
          f"{'' if a.dry_run or not a.open_on_exit else ' (hands open on exit)'}", flush=True)
    t0 = time.monotonic(); tick = 0; last_print = t0; held = {s: 0 for s in a.sides}; frames = {s: 0 for s in a.sides}
    try:
        while True:
            t = time.monotonic()
            for s, g in gloves.items():
                r = g.update(t - t0, tick)
                if r is None:
                    continue                          # no frame yet: the hand stays where it is
                frames[s] += int(r["fresh"]); held[s] += int(r["held"])
                target = motors_to_sdk(r["motors"])
                if not a.dry_run and not r["held"]:   # a held frame = no new information: do not resend
                    await hands[s].send(target, a.slew, dt)
                if t - last_print >= 1.0:
                    print(f"[revo2] {t - t0:6.1f} s  {s:5s} {SDK_ORDER} = {target}  pinch {r['pinch_finger']} {r['pinch_w']:.2f}"
                          f"{'  HELD' if r['held'] else ''}  (frames {frames[s]}, held ticks {held[s]})", flush=True)
            if t - last_print >= 1.0:
                if not any(frames.values()):
                    print(f"[revo2] {t - t0:6.1f} s  waiting for PICO hand frames (headset on? controllers down? hands in view?)", flush=True)
                last_print = t
            tick += 1
            await asyncio.sleep(max(0.0, t0 + tick * dt - time.monotonic()))
    except (KeyboardInterrupt, asyncio.CancelledError):   # Ctrl-C under asyncio.run arrives as a cancellation inside the task
        print("\n[revo2] stopping", flush=True)
    finally:
        for g in gloves.values():
            g.source.close()
        log.close()
        if not a.dry_run:
            await close_all(hands, open_first=a.open_on_exit)
        print(f"[revo2] log -> {out}/glove_log.bin", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "open", "run"])
    ap.add_argument("--sides", default="left,right")
    ap.add_argument("--left", default="", help="pin the left hand: PORT:SLAVE, e.g. /dev/ttyUSB2:126")
    ap.add_argument("--right", default="", help="pin the right hand: PORT:SLAVE, e.g. /dev/ttyUSB1:127")
    ap.add_argument("--rate", type=float, default=30.0, help="command rate (Hz)")
    ap.add_argument("--slew", type=float, default=1500.0, help="max finger travel per second (0..1000 units)")
    ap.add_argument("--smooth-hz", type=float, default=10.0)
    ap.add_argument("--min-conf", type=float, default=0.0, help="PICO quality threshold (0 = take every frame)")
    ap.add_argument("--dry-run", action="store_true", help="run the PICO side, print targets, send nothing")
    ap.add_argument("--no-open-on-exit", dest="open_on_exit", action="store_false")
    a = ap.parse_args()
    a.sides = [s.strip() for s in a.sides.split(",") if s.strip()]
    try:
        asyncio.run({"status": cmd_status, "open": cmd_open, "run": cmd_run}[a.cmd](a))
    except KeyboardInterrupt:                         # the task already cleaned up (opened the hands, closed the ports)
        pass


if __name__ == "__main__":
    main()
