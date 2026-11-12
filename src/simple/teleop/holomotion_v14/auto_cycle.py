"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

The teleop's auto cycle (2026-11-03, user: "after i press space, wait 5 seconds then do the enter motion tracking mode and
recording, after the gate is met and wait time is pass, auto save and reset a new scene, and wait another 5 s to record, i need
to see the count down from the stream"): for hand-tracking sessions, where no controller is held to press the buttons.

    SPACE on the keyboard arms the cycle -> a countdown on the headset view (--countdown-s, 5 s) -> the motion-tracking button
    is pressed for the operator (with --record-on-motion that also starts the recording) -> the operator does the slice -> the
    slice's end event + its overlap save the episode by themselves (--slice-auto-cut, forced on) and the scene resets to a new
    setup -> as soon as the robot stands again, the next countdown -> ... until --num-episodes. SPACE again pauses after the
    current episode (and resumes).

Every press goes through the keyboard stand-in (pico_keys.PicoKeys.press_buttons), i.e. into the controller sample the agent
and the CLI read and the recorder writes (holomotion.pico), so a re-record sees the same presses. The cycle never touches the
recorder directly.
"""
from __future__ import annotations

import math
import time

YELLOW, RED, GREY, WHITE, GREEN = (60, 220, 255), (60, 60, 255), (190, 190, 190), (255, 255, 255), (80, 220, 80)


class AutoCycle:
    def __init__(self, keys, countdown_s: float = 5.0, motion_button: str = "Y", start_armed: bool = False, max_record_s: float = 0.0) -> None:
        self.keys, self.countdown_s, self.motion_button = keys, float(countdown_s), motion_button
        self.max_record_s = float(max_record_s)
        self.armed = bool(start_armed)
        self.state = "wait_ready" if start_armed else "idle"               # idle | wait_ready | countdown | engaging | recording
        self.t_end = self.t_press = self.t_rec = 0.0
        self.presses = 0
        self.message = ""
        self._abandoned = False
        self.blocked = False

    def _say(self, text: str) -> None:
        self.message = text
        print(f"[AutoCycle] {time.strftime('%H:%M:%S')} {text}", flush=True)

    def step(self, now: float, rec_state: str, quick_ready: bool, episodes_saved: int, num_episodes: int, blocked: bool = False) -> None:
        """Once per control tick. rec_state: RecordingState.name (WAITING_FOR_LANDING / RECORDING / EPISODE_DONE). blocked: the
        held start refuses the hand-over because the headset streams no body frames -- keep pressing, do not give up."""
        self.blocked = bool(blocked)
        for _ in range(self.keys.take_event("SPACE") if self.keys is not None else 0):
            self.armed = not self.armed
            if self.armed:
                self._say("armed: the next episode starts after the countdown" if self.state in ("idle", "wait_ready") else "armed again")
                if self.state == "idle":
                    self.state = "wait_ready"
            else:
                self._say("paused: no new countdown; a running episode still saves itself" if self.state == "recording" else "paused")
                if self.state in ("wait_ready", "countdown"):
                    self.state = "idle"
        if self.state == "wait_ready":
            if episodes_saved >= num_episodes:
                self.state = "idle"
            elif rec_state == "WAITING_FOR_LANDING" and quick_ready:
                self.state, self.t_end = "countdown", now + self.countdown_s
                self._say(f"countdown {self.countdown_s:g} s, then {self.motion_button} (motion tracking + recording)")
        elif self.state == "countdown":
            if now >= self.t_end:
                self.keys.press_buttons((self.motion_button,), now, hold_s=0.3)
                self.presses += 1
                self.state, self.t_press = "engaging", now
        elif self.state == "engaging":
            if rec_state == "RECORDING":
                self.state, self.t_rec, self._abandoned = "recording", now, False
                self._say("recording; the slice's end event + overlap save it")
            elif self.blocked:
                if now - self.t_press > 1.5:                                 # the headset streams no body frames: press again, patiently
                    self.keys.press_buttons((self.motion_button,), now, hold_s=0.3); self.t_press = now
                    if now - getattr(self, "_blocked_told", 0.0) > 10.0:
                        self._blocked_told = now; self._say("waiting for the headset's body tracking before the hand-over")
            elif now - self.t_press > 1.5:
                if self.presses % 2:                                         # one more try
                    self.keys.press_buttons((self.motion_button,), now, hold_s=0.3); self.presses += 1; self.t_press = now
                else:
                    self.state = "idle"; self.armed = False
                    self._say(f"{self.motion_button} did not start a recording (is the robot standing? is the reference there?): paused, SPACE to retry")
        elif self.state == "recording":
            if rec_state != "RECORDING":                                     # saved (or abandoned) and reset: go round again
                self.state = "wait_ready" if self.armed else "idle"
                if not self.armed:
                    self._say("paused after this episode; SPACE to continue")
            elif self.max_record_s and now - self.t_rec > self.max_record_s and not self._abandoned:
                self.keys.press_buttons(("left_menu", "B"), now, hold_s=0.3); self._abandoned = True
                self._say(f"no end event within {self.max_record_s:g} s: episode abandoned (left menu + B), new setup")

    def scene_restarted(self) -> None:
        """The CLI reset the scene (a start that came to touch something was redrawn, an episode saved): a running countdown starts
        over when the new start is ready; a running episode's state is left to step()."""
        if self.state in ("countdown", "engaging"):
            self.state = "wait_ready"
            self._say("new start: the countdown runs again when the robot stands")

    def lines(self, now: float) -> list:
        """Overlay lines for the headset view: (text, colour[, scale])."""
        if self.state == "countdown":
            left = max(0.0, self.t_end - now)
            return [(f"{math.ceil(left)}", YELLOW, 4.0), (f"recording in {left:4.1f} s  (stand still)", YELLOW)]
        if self.state == "engaging":
            if getattr(self, "blocked", False):
                return [("waiting for the headset's body tracking (stand still)", YELLOW)]
            return [("starting motion tracking + recording ...", YELLOW)]
        if self.state == "recording":
            return [("AUTO: saves itself when the slice is complete" + ("" if self.armed else "  (paused after this one)"), GREEN if self.armed else GREY)]
        if self.state == "wait_ready":
            return [("AUTO: next episode when the robot stands", GREY)]
        return [("AUTO: press SPACE to start (terminal, sim window or headset-view window focused)", GREY)]
