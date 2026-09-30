"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Headset pictures from the G1's HBVCAM stereo head camera (robot_variants.teleop_mjcf("stereo")):

    PinholeView   the rectified left eye, one MuJoCo camera (1280 x 720, 104.7 x 72.2 deg)
    FisheyeView   both eyes as the real camera streams them: raw fisheye (OpenCV fisheye model of the 2026-09-10
                  calibration), left | right, 2560 x 720

MuJoCo cameras are pinhole only. FisheyeView renders each eye's 90-degree face cameras (hbvcam_{eye}_{face}, only the
faces the lens sees: front, left, right, up) into an atlas and remaps it with a per-pixel lookup built once from the
calibration's fisheye rays (third_party/hbvcam_stereo/hbvcam_fisheye.py, whose self-test checks the ray model against
cv2.fisheye to 1e-3 px). Both views set the off-screen buffer only while their renderer is created and put it back, so
the model stays byte-identical (the exact log's model hash).
"""

from __future__ import annotations

import sys

import numpy as np

from simple.teleop.holomotion_v14.robot_variants import (FISHEYE_PREFIX, PINHOLE_CAMERA, PINHOLE_H, PINHOLE_W,
                                                         STEREO_DIR)


def _renderer(model, height: int, width: int):
    import mujoco
    g = model.vis.global_
    saved = (int(g.offwidth), int(g.offheight))
    g.offwidth, g.offheight = max(saved[0], width), max(saved[1], height)
    try:
        return mujoco.Renderer(model, height, width)
    finally:
        g.offwidth, g.offheight = saved


def _hf():
    if str(STEREO_DIR) not in sys.path:
        sys.path.insert(0, str(STEREO_DIR))
    import hbvcam_fisheye
    return hbvcam_fisheye


class PinholeView:
    """The stereo camera's rectified left eye (PINHOLE_CAMERA), 1280 x 720."""

    def __init__(self, sonic_env) -> None:
        self.env, self._model, self._renderer = sonic_env, None, None

    def render(self) -> np.ndarray:
        m, d = self.env.mujoco.mjModel, self.env.mujoco.mjData
        if self._model is not m:                               # a new scene (every reset compiles a new model)
            if self._renderer is not None:
                self._renderer.close()
            self._renderer, self._model = _renderer(m, PINHOLE_H, PINHOLE_W), m
        self._renderer.update_scene(d, camera=PINHOLE_CAMERA)
        return self._renderer.render()


class FisheyeView:
    """Both eyes' raw fisheye images, 1280 x 720 each (left, right)."""

    FACE = 960                   # face render size: 480 px/rad at the face centre, about the lens's 471 px/rad

    def __init__(self, sonic_env, face_px: int = FACE) -> None:
        HF = _hf()
        self.env, self._model, self._renderer = sonic_env, None, None
        self._next_right, self._latest = False, {"left": None, "right": None}
        self.N = N = int(face_px)
        T = N + 2                                              # tile with a 1-px border (no bleeding across tiles)
        calib = HF.load_calib()
        names = list(HF.FACES)
        self.eyes = {}
        for eye in ("left", "right"):
            K, D, _, _ = HF.eye_calib(calib, eye)
            rays, valid, _ = HF.fisheye_rays(K, D)
            pick = np.argmax(np.stack([rays @ np.array(HF.FACES[n][0], float) for n in names], -1), -1)
            used = [n for i, n in enumerate(names) if ((pick == i) & valid).any()]
            mx = np.full(rays.shape[:2], -10.0, np.float32)    # outside the lens circle: black
            my = np.full(rays.shape[:2], -10.0, np.float32)
            for j, n in enumerate(used):
                fwd, right, down = (np.array(a, float) for a in HF.FACES[n])
                z = np.maximum(rays @ fwd, 1e-6)
                sel = (pick == names.index(n)) & valid
                mx[sel] = (((rays @ right) / z + 1) * N / 2 - 0.5)[sel] + j * T + 1   # as HF.remap_faces, in the atlas
                my[sel] = (((rays @ down) / z + 1) * N / 2 - 0.5)[sel] + 1
            self.eyes[eye] = dict(used=used, mx=mx, my=my, atlas=np.zeros((T, T * len(used), 3), np.uint8))

    def _ensure_renderer(self):
        m = self.env.mujoco.mjModel
        if self._model is not m:                               # a new scene (every reset compiles a new model)
            if self._renderer is not None:
                self._renderer.close()
            self._renderer, self._model = _renderer(m, self.N, self.N), m
        return self._renderer

    def render_eye(self, eye: str) -> np.ndarray:
        """One eye's fisheye image (1280 x 720)."""
        import cv2
        r, d = self._ensure_renderer(), self.env.mujoco.mjData
        N, T, E = self.N, self.N + 2, self.eyes[eye]
        a = E["atlas"]
        for j, face in enumerate(E["used"]):
            x0 = j * T
            r.update_scene(d, camera=f"{FISHEYE_PREFIX}_{eye}_{face}")
            # MuJoCo's headlight shines along each camera's axis, so the faces would be lit differently and show
            # seams: make it direction-free in the rendered scene (the model is not touched)
            sc = r.scene
            for i in range(sc.nlight):
                L = sc.lights[i]
                if L.headlight:
                    L.ambient = np.clip(np.asarray(L.ambient) + 0.6 * np.asarray(L.diffuse), 0.0, 1.0)
                    L.diffuse, L.specular = np.zeros(3), np.zeros(3)
            a[1:N + 1, x0 + 1:x0 + N + 1] = r.render()
            a[0, x0:x0 + T], a[N + 1, x0:x0 + T] = a[1, x0:x0 + T], a[N, x0:x0 + T]
            a[:, x0], a[:, x0 + T - 1] = a[:, x0 + 1], a[:, x0 + T - 2]
        return cv2.remap(a, E["mx"], E["my"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Both eyes now (left, right)."""
        return self.render_eye("left"), self.render_eye("right")

    def step(self) -> tuple[np.ndarray, np.ndarray] | None:
        """One eye per call, alternating (about 3.6 ms instead of 7 per sim step), and the newest left + newest right
        every call: a fresh stereo frame each sim step (the stream sends 30 fps), the two eyes at most one step
        (20 ms) apart. None until both eyes have been rendered once."""
        eye = "right" if self._next_right else "left"
        self._latest[eye] = self.render_eye(eye)
        self._next_right = not self._next_right
        if self._latest["left"] is None or self._latest["right"] is None:
            return None
        return self._latest["left"], self._latest["right"]
