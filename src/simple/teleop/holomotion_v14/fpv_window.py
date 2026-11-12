"""A desktop window with the first-person view of the HoloMotion v1.4 sim teleop: the frame the headset receives
(compose_stream_frame: the head camera picture with the status lines), shown with OpenCV at ~30 fps (2026-10-26, user:
"by default provide a first person view just like the one streamed to the PICO").  Single view: the one picture, 960 px
wide; mono / stereo: both halves side by side.  Silently disabled when no display is available."""
from __future__ import annotations
import numpy as np

TITLE = "HoloMotion v1.4: headset view"


class FpvWindow:
    def __init__(self, width: int = 960, on_key=None) -> None:
        import os, sys
        self.width = int(width); self._shown = False
        self.on_key = on_key                           # (opencv key code) -> None: the window's key presses go to the keyboard stand-in (2026-11-04)
        self.ok = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform != "linux")   # a Qt window with no display aborts the process, so never try
        if not self.ok:
            print("[HoloMotion v1.4] first-person window off (no display)")

    def show(self, left, right, status, view: str) -> None:
        if not self.ok or left is None:
            return
        try:
            import cv2
            from simple.agents.holomotion_v14_agent import compose_stream_frame
            frame = compose_stream_frame(left, right, status or [], view)
            if frame is None:
                return
            h, w2 = frame.shape[:2]; w = w2 // 2
            if view == "single":
                frame = frame[:, w // 2:w // 2 + w]                    # the picture sits centred in a black 2:1 frame
            scale = self.width / frame.shape[1]
            frame = cv2.resize(frame, (self.width, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            if not self._shown:
                cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL); cv2.resizeWindow(TITLE, frame.shape[1], frame.shape[0]); self._shown = True
            cv2.imshow(TITLE, frame)
            k = cv2.waitKey(1)
            if k >= 0 and self.on_key is not None:
                self.on_key(int(k))
        except Exception as e:                                          # no display / no GUI build: give up quietly
            print(f"[HoloMotion v1.4] first-person window off ({type(e).__name__}: {e})")
            self.ok = False

    def close(self) -> None:
        if self._shown:
            try:
                import cv2; cv2.destroyWindow(TITLE); cv2.waitKey(1)
            except Exception:
                pass
