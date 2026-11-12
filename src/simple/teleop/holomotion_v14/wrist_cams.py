"""The two HBVCAM-F2619HD wrist cameras of the rot0 Revo 2 robot in the HoloMotion v1.4 sim teleop recorder (2026-10-26).

The robot file carries MuJoCo cameras `wrist_left` / `wrist_right` (the 76-deg pinhole of the F2619HD) and the 90-deg faces
`wrist_<side>_front/left/right` (scripts/robots/add_cameras_mounts.py: in the printed collar's housing on the palm side of each
wrist-yaw link, looking along the forearm).  The recorder saves a 150-deg (diagonal) equidistant fisheye view per wrist at
640 x 360 (user 2026-10-26: "save the 150 fov view each"), assembled from the three faces with a per-pixel lookup, on the
pre-step state like the head eyes: observation.images.wrist_left / observation.images.wrist_right (LeRobot video features).  They are never streamed to the
PICO (user: "no need for displaying to pico").  The renderer re-binds whenever the env recompiles its model (new setup).
Shadows and reflections are off in these two renders (1.4 ms per tick for both instead of 3.8 ms in the kitchen scene).
"""
from __future__ import annotations
import numpy as np
import mujoco

NAMES = {"left": "wrist_left", "right": "wrist_right"}
W, H = 640, 360
EFL_MM, PIXEL_UM, SENSOR_W = 4.4, 3.0, 1920                 # F2619HD: EFL 4.4 mm, 3 um pixels, 1920 wide (the 76 deg lens)
FOV_DEG = 150.0                                             # user 2026-10-26: "save the 150 fov view each" -> diagonal FOV of an equidistant fisheye
FACE = 448                                                  # face render size: 224 px/rad at the face centre vs 280 px/rad in the output centre
NEAR_M = 0.006                                              # near clipping plane for the wrist faces (the palm starts ~13 mm from the lens)
FACES = ("front", "left", "right")                          # the 16:9 150-deg view never needs the up/down faces


def fisheye_f(fov_deg: float = FOV_DEG, w: int = W, h: int = H) -> float:
    """Equidistant focal length (px) that puts the diagonal FOV on the image diagonal."""
    return float(np.hypot(w / 2, h / 2) / np.radians(fov_deg / 2))


def fisheye_maps(fov_deg: float = FOV_DEG, w: int = W, h: int = H, n: int = FACE):
    """cv2.remap lookups from a horizontal atlas of the (front, left, right) 90-deg faces (each n x n with a 1-px border)."""
    f = fisheye_f(fov_deg, w, h); cx, cy = (w - 1) / 2, (h - 1) / 2
    v, u = np.mgrid[0:h, 0:w].astype(np.float64); dx, dy = u - cx, v - cy
    r = np.hypot(dx, dy); th = r / f; ph = np.arctan2(dy, dx)
    X, Y, Z = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)     # optical frame: X right, Y down, Z forward
    # each face's optical axes expressed in the front camera's optical frame: (forward, right, down)
    axes = {"front": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "left": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), "right": ((1, 0, 0), (0, 0, -1), (0, 1, 0))}
    rays = np.stack([X, Y, Z], -1); T = n + 2
    fwd = np.stack([rays @ np.array(axes[k][0], float) for k in FACES], -1); pick = np.argmax(fwd, -1)
    mx = np.full((h, w), -10.0, np.float32); my = np.full((h, w), -10.0, np.float32)
    for j, k in enumerate(FACES):
        fw, ri, dn = (np.array(a, float) for a in axes[k]); z = np.maximum(rays @ fw, 1e-6); sel = pick == j
        mx[sel] = (((rays @ ri) / z + 1) * n / 2 - 0.5)[sel] + j * T + 1
        my[sel] = (((rays @ dn) / z + 1) * n / 2 - 0.5)[sel] + 1
    valid = th <= np.radians(fov_deg / 2) + 1e-9
    mx[~valid] = -10.0; my[~valid] = -10.0                                     # outside the lens circle: black
    return mx, my, f


def features() -> dict:
    f = {f"observation.images.wrist_{s}": {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"]} for s in ("left", "right")}
    f["wrist_cams.fresh"] = {"dtype": "int8", "shape": (1,), "names": ["1_new_render_0_repeated_frame"]}
    return f


def image_spec(fov_deg: float = FOV_DEG) -> dict:
    f = fisheye_f(fov_deg); hf = np.degrees(2 * (W / 2) / f); vf = np.degrees(2 * (H / 2) / f)
    return {f"observation.images.wrist_{s}": dict(what=f"monocular wrist camera in the printed collar on the {s} wrist (palm side, looking along the forearm): "
                                                        f"equidistant fisheye view, {fov_deg:g} deg on the image diagonal ({hf:.1f} x {vf:.1f} deg), {W}x{H}, assembled from "
                                                        "three 90-deg MuJoCo faces (wrist_<side>_front/left/right); rendered on a 30 Hz schedule and repeated to the dataset's 50 fps "
                                                        "(wrist_cams.fresh = 1 on a new render)",
                                                   model="equidistant fisheye: r_px = f * theta", width=W, height=H, f=round(f, 3), cx=(W - 1) / 2, cy=(H - 1) / 2,
                                                   fov_diag_deg=fov_deg, hfov_deg=round(hf, 1), vfov_deg=round(vf, 1), mujoco_camera=NAMES[s]) for s in ("left", "right")}


def available(model) -> bool:
    return all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{n}_{k}") >= 0 for n in NAMES.values() for k in FACES)


class WristCams:
    """render() every control tick; new images are rendered on an `hz` schedule (user 2026-10-26: "30hz is enough") and
    repeated in between, so the dataset keeps one 50 fps stream per camera; the frame's `fresh` flag says which ticks
    carry a new render (column wrist_cams.fresh)."""

    def __init__(self, sonic_env, hz: float = 30.0, tick_dt: float = 0.02, fov_deg: float = FOV_DEG) -> None:
        self.env = sonic_env; self._model = None; self._renderer = None
        self.hz, self.tick_dt = float(hz), float(tick_dt); self._acc = 1.0; self._last = None
        self.fov_deg = float(fov_deg); self.mx, self.my, self.f = fisheye_maps(self.fov_deg)
        T = FACE + 2; self._atlas = np.zeros((T, T * len(FACES), 3), np.uint8)

    def _bind(self, m):
        if self._renderer is not None:
            from simple.teleop.holomotion_v14.hbvcam_views import close_renderer
            close_renderer(self._renderer)                   # never Renderer.close() with another view's context current (black pictures after a reset)
        g = m.vis.global_; saved = (int(g.offwidth), int(g.offheight)); g.offwidth, g.offheight = max(saved[0], FACE), max(saved[1], FACE)
        try:
            self._renderer = mujoco.Renderer(m, FACE, FACE)
        finally:
            g.offwidth, g.offheight = saved                      # the model stays byte-identical (exact-log hash)
        self._model = m
        self._ids = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"{n}_{k}") for k in FACES] for s, n in NAMES.items()}

    def render(self) -> dict:
        m, d = self.env.mujoco.mjModel, self.env.mujoco.mjData
        if self._model is not m:
            self._bind(m); self._last = None
        if self.hz > 0:
            self._acc += self.hz * self.tick_dt
            if self._acc < 1.0 and self._last is not None:
                return {**self._last, "fresh": False}
            self._acc -= 1.0
        import cv2
        out = {}; r = self._renderer; N = FACE; T = N + 2; a = self._atlas
        # MuJoCo's near clipping plane is vis.map.znear * stat.extent for every camera: in a room-sized scene that is
        # ~10 cm, which clips the palm (1-6 cm from the lens) out of the wrist views (2026-10-26).  Render these faces with
        # a 6 mm near plane and put the model's value back (byte-identical model for the exact-log hash).
        znear0 = float(m.vis.map.znear); m.vis.map.znear = NEAR_M / max(float(m.stat.extent), 1e-6)
        try:
            self._render_faces(d, out)
        finally:
            m.vis.map.znear = znear0
        self._last = out
        return {**out, "fresh": True}

    def _render_faces(self, d, out):
        import cv2
        r = self._renderer; N = FACE; T = N + 2; a = self._atlas
        for s, cids in self._ids.items():
            for j, cid in enumerate(cids):
                x0 = j * T; r.update_scene(d, camera=cid); sc = r.scene
                sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0; sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0   # cheaper passes (50 Hz loop budget)
                for i in range(sc.nlight):                       # the headlight follows each face camera: make it direction-free so the faces match
                    L = sc.lights[i]
                    if L.headlight:
                        L.ambient = np.clip(np.asarray(L.ambient) + 0.6 * np.asarray(L.diffuse), 0.0, 1.0); L.diffuse, L.specular = np.zeros(3), np.zeros(3)
                a[1:N + 1, x0 + 1:x0 + N + 1] = r.render()
                a[0, x0:x0 + T], a[N + 1, x0:x0 + T] = a[1, x0:x0 + T], a[N, x0:x0 + T]; a[:, x0], a[:, x0 + T - 1] = a[:, x0 + 1], a[:, x0 + T - 2]
            out[s] = cv2.remap(a, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def close(self):
        if self._renderer is not None:
            from simple.teleop.holomotion_v14.hbvcam_views import close_renderer
            close_renderer(self._renderer); self._renderer = None
