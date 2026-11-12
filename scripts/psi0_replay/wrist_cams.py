"""Two wrist cameras for the SIMPLE G1 (Dex3 hands), used by replay_psi0.py.

Placement = the printed wrist collar designed for the real robot (docs/wrist_mount_tuner,
data/robots/g1_rot0_teleop_revo2/cameras.json): collar on <side>_wrist_yaw_link at x = 36.5 mm,
lens 38.4 mm off the forearm axis looking along it toward the fingers, tilted `tilt_deg` toward the
hand's centre line about the pocket floor. On the Revo 2 robot the housing is on the PALM side; on
the Dex3 hand the default is the INDEX-finger side (+z): it sees the target in every frame where the
hand is near it (4 tasks), the back of the hand never does, and the Dex3 thumb swings into a
palm-side housing (docs/_scan/psi0_replay_wrist/mount_study.py). The right camera is the left one mirrored through the palm plane (y -> -y), image
mirrored so both stay right-handed. Only the camera is added, no collar mesh, so the head view of
the replay is the same robot as in the recording.

Frames:
  MuJoCo  camera on <side>_wrist_yaw_link (pos = palm offset + local pos)
  Isaac   camera prim under <robot>/<side>_hand_camera_base_link (the USD's own wrist-camera frame,
          = <side>_wrist_yaw_link + (0.0415, +-0.003, 0), identity rotation)
Both get the same pose: in the hand-camera frame, camera axes "world" (+x forward, +z up).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import transforms3d as t3d

# collar + housing (left wrist_yaw frame), from data/robots/g1_rot0_teleop_revo2/cameras.json
MOUNT_POS = np.array([0.0365, 0.0, 0.0])
MOUNT_QUAT = np.array([0.7071068, 0.0, 0.7071068, 0.0])      # +90 deg about y: +z_m = +x wrist (to the hand)
POCKET_FLOOR = np.array([0.0, -0.0384, 0.00605])             # tilt pivot, mount frame
LENS_ABOVE_FLOOR = 0.00795                                   # lens 7.95 mm above the pocket floor (+z_m)
PALM_OFFSET = {"left": np.array([0.0415, 0.003, 0.0]),       # hand_camera_base_link in <side>_wrist_yaw_link
               "right": np.array([0.0415, -0.003, 0.0])}

DEFAULT_TILT_DEG = 0.0       # the printed pocket holds the module at 0 deg
# where the housing sits around the forearm (left wrist frame; the right hand is the mirror image):
# palm = -y (the Revo 2 design), back = +y, index = +z (Dex3 index finger side), middle = -z
MOUNT_ROLL_DEG = {"palm": 0.0, "back": 180.0, "index": -90.0, "middle": 90.0}
DEFAULT_HFOV_DEG = 90.0      # = the psi0 head camera's 640x360 intrinsics (fx = fy = 320)

# MuJoCo camera (-z forward, +y up) <- "world" camera axes (+x forward, +z up), as in engines/mujoco.py
_Q_ISAAC_MUJOCO = t3d.quaternions.mat2quat(np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float))
_R_ISAAC_MUJOCO = t3d.quaternions.quat2mat(_Q_ISAAC_MUJOCO)


def _rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def left_camera_in_wrist(tilt_deg: float, side: str = "index") -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo-convention camera pose (pos, R) in left_wrist_yaw_link.

    side: where the housing sits around the forearm (MOUNT_ROLL_DEG): "palm" = as printed for the
    Revo 2 robot (-y), "back" (+y), "index" (+z, the Dex3 index-finger side), "middle" (-z).
    The tilt always turns the optical axis toward the hand's centre line.
    """
    R_m = _rx(MOUNT_ROLL_DEG[side]) @ t3d.quaternions.quat2mat(MOUNT_QUAT)   # roll the collar about the forearm
    R_t = _rx(-tilt_deg)                                   # optical axis (+z_m) turns toward +y_m
    lens_m = POCKET_FLOOR + R_t @ np.array([0.0, 0.0, LENS_ABOVE_FLOOR])
    R_cam_m = R_t @ t3d.quaternions.quat2mat([0.0, 0.0, 1.0, 0.0])   # MuJoCo cam looks +z_m, up +y_m
    return MOUNT_POS + R_m @ lens_m, R_m @ R_cam_m


def camera_in_hand_frame(side: str, tilt_deg: float, mount_side: str = "index") -> tuple[list[float], list[float]]:
    """(position, wxyz quaternion) in <side>_hand_camera_base_link, camera axes 'world'."""
    p, R = left_camera_in_wrist(tilt_deg, mount_side)
    if side == "right":
        M = np.diag([1.0, -1.0, 1.0])
        p, R = M @ p, M @ R @ np.diag([-1.0, 1.0, 1.0])      # mirror; flip image x to stay a rotation
    R_world = R @ _R_ISAAC_MUJOCO.T
    q = t3d.quaternions.mat2quat(R_world)
    q = q if q[0] >= 0 else -q
    return (p - PALM_OFFSET[side]).tolist(), q.tolist()


@dataclass
class WristCamSpec:
    tilt_deg: float = DEFAULT_TILT_DEG
    hfov_deg: float = DEFAULT_HFOV_DEG
    mount_side: str = "index"
    width: int = 640
    height: int = 360
    keys: tuple[str, ...] = ("wrist_left", "wrist_right")
    near: float = 0.005

    def cfg(self, side: str):
        from simple.sensors import CameraCfg

        pos, quat = camera_in_hand_frame(side, self.tilt_deg, self.mount_side)
        return CameraCfg(
            uid="HBVCAM_F2619HD_wrist",
            mount="eye_in_hand",
            width=self.width,
            height=self.height,
            focal_length=1.93,
            fov=np.deg2rad(self.hfov_deg),
            near=self.near,
            far=5.0,
            pose=dict(position=pos, quaternion=quat,
                      mujoco_body=f"{side}_wrist_yaw_link",
                      mujoco_offset=PALM_OFFSET[side].tolist(),
                      isaac_follow_mujoco=True),
        )

    def to_json(self) -> dict:
        out = dict(tilt_deg=self.tilt_deg, hfov_deg=self.hfov_deg, mount_side=self.mount_side,
                   width=self.width, height=self.height,
                   fx=self.width / (2 * np.tan(np.radians(self.hfov_deg) / 2)))
        for side in ("left", "right"):
            pos, quat = camera_in_hand_frame(side, self.tilt_deg, self.mount_side)
            out[f"{side}_in_hand_camera_base_link"] = dict(pos=pos, quat_wxyz=quat, axes="world (+x fwd, +z up)")
            pw, Rw = left_camera_in_wrist(self.tilt_deg, self.mount_side)
            if side == "right":
                M = np.diag([1.0, -1.0, 1.0]); pw = M @ pw
            out[f"{side}_lens_in_wrist_yaw_link_m"] = [round(float(v), 5) for v in pw]
        return out


def add_to_task_class(task_cls, spec: WristCamSpec) -> None:
    """Add wrist_left / wrist_right to a task class's sensor_cfgs (this process only).

    Must run before gym.make so the env's observation space includes them.
    """
    cfgs = dict(task_cls.sensor_cfgs)
    for key, side in zip(spec.keys, ("left", "right")):
        cfgs[key] = spec.cfg(side)
    task_cls.sensor_cfgs = cfgs


def fix_near_plane(m, near_m: float = 0.005) -> None:
    """MuJoCo's near plane is vis.map.znear * stat.extent for every camera (~10 cm in a room scene),
    which cuts the hand out of a wrist view. Lower it for this model."""
    m.vis.map.znear = min(float(m.vis.map.znear), near_m / max(float(m.stat.extent), 1e-6))


class VideoWriter:
    """H.264 mp4 via PyAV (yuv420p, the psi0 videos' pixel format)."""

    def __init__(self, path: Path, fps: int = 50, crf: int = 18):
        import av

        self.path = Path(path)
        self.fps = fps
        self.crf = crf
        self._c = None
        self._s = None
        self.n = 0

    def write(self, rgb: np.ndarray) -> None:
        import av

        rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8))
        if self._c is None:
            h, w = rgb.shape[:2]
            self._c = av.open(str(self.path), "w")
            self._s = self._c.add_stream("libx264", rate=self.fps)
            self._s.width, self._s.height, self._s.pix_fmt = w, h, "yuv420p"
            self._s.options = {"crf": str(self.crf), "preset": "medium"}
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for pkt in self._s.encode(frame):
            self._c.mux(pkt)
        self.n += 1

    def close(self) -> None:
        if self._c is not None:
            for pkt in self._s.encode():
                self._c.mux(pkt)
            self._c.close()
            self._c = None
