"""Keyboard stand-in for the PICO controllers (2026-11-03; moved from scripts/wuji_glove_pilot for the teleop's --hand-source skeleton and
--auto-cycle options): with both hands in Wuji gloves nothing can
hold a controller, but the teleop still takes its buttons and sticks from the controller sample. PicoKeys reads the keys
of the terminal the teleop runs in and lays them over that sample (HoloMotionV14Agent.pico, which the agent's state
machine and the CLI's record / save / reset combos both read); while no controller sample arrives at all, a neutral one is
supplied so the quick start sees an operator. The wrist poses still come from the headset's body tracking.

One key = one action, the sim combos included (a tap holds its buttons for HOLD_BUTTON_S, long enough for the CLI's
edge detection); the arrow keys deflect the sticks while held (the terminal's key repeat keeps them pressed).

    r  record / save (left menu + A)      b  abandon (left menu + B)      n  new setup (left menu + Y)
    0  reset the same setup (left menu + both triggers)
    a  walking mode   y  motion tracking   s  stand up (L3)   x  zero torque   e  emergency stop (R3)
    up / down  walk          left / right  turn          , / .  sidestep          h  this list

Self-test (no teleop): .venv/bin/python -m simple.teleop.holomotion_v14.pico_keys --selftest
"""
from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

from simple.teleop.holomotion_v14.wire import BUTTONS  # (button name -> bit of the v1.4.1 sample)

HOLD_BUTTON_S = 0.3          # a tap keeps its buttons pressed this long (the CLI and the agent poll at 50 Hz)
HOLD_AXIS_S = 0.6            # a held arrow key auto-repeats after about 0.5 s: bridge that gap so the stick stays deflected
STICK = 0.5                  # stick deflection of the walk / turn keys (the robot's speed scales with it)
WALK_SIGN, TURN_SIGN = +1.0, -1.0        # left_axis[1] > 0 = forward, right_axis[0] < 0 = turn left (v1.4.1 PicoController)

# key -> what it presses: buttons = names in the sample (also OR'd into button_bits), triggers = both triggers to 1.0,
# axis = (stick, index, value)
KEYS = {
    "r": dict(buttons=("left_menu", "A"), help="record / save (left menu + A)"),
    "b": dict(buttons=("left_menu", "B"), help="abandon the episode (left menu + B)"),
    "n": dict(buttons=("left_menu", "Y"), help="new random setup (left menu + Y)"),
    "0": dict(buttons=("left_menu",), triggers=True, help="reset the same setup (left menu + both triggers)"),
    "a": dict(buttons=("A",), help="walking mode"),
    "y": dict(buttons=("Y",), help="motion tracking"),
    "s": dict(buttons=("L3",), help="stand up"),
    "x": dict(buttons=("X",), help="zero torque"),
    "e": dict(buttons=("R3",), help="emergency stop"),
    "UP": dict(axis=("left_axis", 1, WALK_SIGN * STICK), help="walk forward"),
    "DOWN": dict(axis=("left_axis", 1, -WALK_SIGN * STICK), help="walk back"),
    "LEFT": dict(axis=("right_axis", 0, TURN_SIGN * STICK), help="turn left"),
    "RIGHT": dict(axis=("right_axis", 0, -TURN_SIGN * STICK), help="turn right"),
    ",": dict(axis=("left_axis", 0, TURN_SIGN * STICK), help="sidestep left"),
    ".": dict(axis=("left_axis", 0, -TURN_SIGN * STICK), help="sidestep right"),
}
EVENT_KEYS = {" ": "SPACE", "p": "SPACE"}       # not buttons: events the auto cycle reads (space or p: start / pause)
ESCAPES = {"\x1b[A": "UP", "\x1b[B": "DOWN", "\x1b[D": "LEFT", "\x1b[C": "RIGHT", "\x1bOA": "UP", "\x1bOB": "DOWN", "\x1bOD": "LEFT", "\x1bOC": "RIGHT"}
NEUTRAL = dict({n: 0 for n in BUTTONS}, button_bits=0, left_axis=(0.0, 0.0), right_axis=(0.0, 0.0), left_trigger=0.0,
               right_trigger=0.0, left_grip=0.0, right_grip=0.0, l_active=1, r_active=1)


def help_text() -> str:
    cols = [f"{k.lower():>5}  {v['help']}" for k, v in KEYS.items()]
    return "[pico_keys] keys stand in for the PICO controllers:\n" + "\n".join("   " + "   ".join(cols[i:i + 3]) for i in range(0, len(cols), 3))


class PicoKeys:
    """Key presses from a file descriptor in cbreak mode (no echo, no line buffer; Ctrl-C still works), applied to the
    controller sample. The press times are kept per key; apply() treats a key as held for HOLD_*_S after its last press."""

    def __init__(self, fd: int | None = None) -> None:
        self.fd = fd
        self._held: dict[str, float] = {}
        self._hold: dict[str, float | None] = {}
        self._events: dict[str, int] = {}
        self._lock = threading.Lock()
        self.presses = 0
        self.synthesized = 0                           # ticks with no controller sample, served a neutral one
        self._saved = None

    @classmethod
    def start(cls) -> "PicoKeys | None":
        if not sys.stdin.isatty():
            print("[pico_keys] stdin is not a terminal: no keyboard stand-in", flush=True)
            return None
        k = cls(sys.stdin.fileno())
        k._saved = termios.tcgetattr(k.fd)
        tty.setcbreak(k.fd)
        threading.Thread(target=k._reader, name="pico_keys", daemon=True).start()
        print(help_text(), flush=True)
        return k

    def restore(self) -> None:
        if self._saved is not None and self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    # ------------------------------------------------------------------ input
    def press(self, key: str, t: float | None = None) -> None:
        if key in KEYS:
            with self._lock:
                self._held[key] = time.monotonic() if t is None else t
                self.presses += 1
        elif key in EVENT_KEYS:
            with self._lock:
                self._events[EVENT_KEYS[key]] = self._events.get(EVENT_KEYS[key], 0) + 1
        elif key == "h":
            print(help_text(), flush=True)

    def press_code(self, code: int, from_cv: bool = False) -> None:
        """A key code from one of our windows (GLFW codes from the MuJoCo viewer's key callback; OpenCV waitKey codes from the
        headset-view window, from_cv=True): the same keys as the terminal's. Keyboard focus is wherever the operator last
        clicked, so SPACE must work from all three (2026-11-04: a live session's SPACE went to a window and nothing happened)."""
        code = int(code)
        arrows = {81: "LEFT", 82: "UP", 83: "RIGHT", 84: "DOWN"} if from_cv else {262: "RIGHT", 263: "LEFT", 264: "DOWN", 265: "UP"}
        if code in arrows:
            self.press(arrows[code])
            return
        c = code & 0xFF if from_cv else code
        if 32 <= c < 127:
            self.press(chr(c).lower())

    def take_event(self, name: str) -> int:
        """How many times the event key (SPACE) was pressed since the last call."""
        with self._lock:
            return self._events.pop(name, 0)

    def press_buttons(self, names, t: float | None = None, hold_s: float | None = None) -> None:
        """A programmatic press of controller buttons (the auto cycle's Y / left menu + A / left menu + B), held like a tap."""
        key = f"@{'+'.join(names)}"
        if key not in KEYS:
            KEYS[key] = dict(buttons=tuple(names), help="programmatic")
        with self._lock:
            self._held[key] = time.monotonic() if t is None else t
            self._hold[key] = hold_s
            self.presses += 1

    def _reader(self) -> None:
        buf = ""
        while True:
            r, _, _ = select.select([self.fd], [], [], 0.05 if buf else 0.5)
            if not r:
                if buf:                                 # a lone escape, or an unknown sequence: drop it
                    buf = ""
                continue
            try:
                ch = os.read(self.fd, 1).decode(errors="ignore")
            except OSError:
                return
            if not ch:
                return
            if buf or ch == "\x1b":
                buf += ch
                if buf in ESCAPES:
                    self.press(ESCAPES[buf]); buf = ""
                elif not any(e.startswith(buf) for e in ESCAPES):   # not an arrow key: drop the escape, the rest are plain keys
                    for c in buf[1:]:
                        self.press(c.lower())
                    buf = ""
                continue
            self.press(ch.lower())

    # ------------------------------------------------------------------ output
    def held(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        with self._lock:
            return [k for k, t in self._held.items() if now - t <= (self._hold.get(k) or (HOLD_AXIS_S if "axis" in KEYS[k] else HOLD_BUTTON_S))]

    def apply(self, s: dict | None, now: float | None = None) -> dict:
        """The controller sample with the held keys pressed on top; a neutral sample when there is none."""
        now = time.monotonic() if now is None else now
        held = self.held(now)
        if s is None:
            self.synthesized += 1
            s = dict(NEUTRAL, timestamp_ns=time.time_ns(), received_monotonic=now)
        if not held:
            return s
        out = dict(s)
        bits = int(out.get("button_bits", 0))
        la, ra = list(out.get("left_axis", (0.0, 0.0))), list(out.get("right_axis", (0.0, 0.0)))
        for k in held:
            spec = KEYS[k]
            for name in spec.get("buttons", ()):
                out[name] = 1
                bits |= BUTTONS[name]
            if spec.get("triggers"):
                out["left_trigger"] = out["right_trigger"] = 1.0
            if "axis" in spec:
                stick, i, v = spec["axis"]
                (la if stick == "left_axis" else ra)[i] = v
        out.update(button_bits=bits, left_axis=tuple(la), right_axis=tuple(ra))
        return out


def _selftest() -> None:
    k = PicoKeys()
    t = 100.0
    s = k.apply(None, now=t)
    assert s["button_bits"] == 0 and s["l_active"] == 1 and s["left_axis"] == (0.0, 0.0), s
    assert k.synthesized == 1
    k.press("r", t)
    s = k.apply(None, now=t + 0.1)
    assert s["left_menu"] == 1 and s["A"] == 1 and s["button_bits"] == (BUTTONS["left_menu"] | BUTTONS["A"]), s
    assert k.apply(None, now=t + 0.5)["A"] == 0, "a tap must end after HOLD_BUTTON_S"
    real = dict(NEUTRAL, B=1, button_bits=BUTTONS["B"], left_grip=0.7)
    k.press("0", t + 1.0)
    s = k.apply(real, now=t + 1.1)
    assert s["B"] == 1 and s["left_grip"] == 0.7 and s["left_menu"] == 1 and s["left_trigger"] == 1.0 == s["right_trigger"], s
    assert s["button_bits"] == (BUTTONS["B"] | BUTTONS["left_menu"]) and real["button_bits"] == BUTTONS["B"], "the real sample must not be modified"
    k.press("UP", t + 2.0); k.press("LEFT", t + 2.0)
    s = k.apply(None, now=t + 2.55)
    assert s["left_axis"] == (0.0, WALK_SIGN * STICK) and s["right_axis"] == (TURN_SIGN * STICK, 0.0), s
    assert k.apply(None, now=t + 2.7)["left_axis"] == (0.0, 0.0), "an arrow key must end after HOLD_AXIS_S"
    # the escape-sequence parser on a pipe
    rd, wr = os.pipe()
    k2 = PicoKeys(rd)
    threading.Thread(target=k2._reader, daemon=True).start()
    os.write(wr, b"\x1b[Ar\x1bx")                      # UP, r, then a lone escape followed by x
    time.sleep(0.25)
    assert set(k2._held) == {"UP", "r", "x"}, set(k2._held)
    k3 = PicoKeys(); k3.press(" "); k3.press("p")
    assert k3.take_event("SPACE") == 2 and k3.take_event("SPACE") == 0, "space / p are events, not buttons"
    k3.press_buttons(("Y",), t=10.0, hold_s=0.5)
    assert k3.apply(None, now=10.4)["Y"] == 1 and k3.apply(None, now=10.6)["Y"] == 0, "a programmatic press holds hold_s"
    print("[pico_keys] selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(help_text())
