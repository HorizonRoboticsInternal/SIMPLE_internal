"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

The G1 + Dex3 with a backpack, and the v1.4.1 backpack motion model.

Backpack body: HoloMotion's G1 payload assets (branch ``feat/g1-payload-multi-urdf-dual-mujoco-eval``,
``assets/robots/unitree/G1/g1_with_hand_0715_backpack``): ``backpack_link`` fixed to ``torso_link`` at
(-0.094605, -0.0005, 0.165324), its COM at the body origin, 1.9 kg with the inertia below; mesh ``backpack_link.STL``
(vendored in ``third_party/holomotion_v14/backpack``). Their MuJoCo eval scene (``scene_u2_dex3_backpack_nocol_damped``)
has no backpack collision, and neither does this one. ``backpack_mjcf(kg)`` writes SIMPLE's G1 MJCF with that body at
``kg`` (inertia scaled with the mass); the v1.4.1 backpack policy was trained for a fixed 3.2 kg
(``v141.brainco_backpack_3p2.model.lock``).

Backpack motion model: ``model_22000`` (obs [1, 604], action [1, 29], the same interface as the public v1.4.1 motion
model; the velocity model is the public one). It is not in the repository -- the upstream build takes a bundle
directory (``config.yaml`` + ``model_22000.onnx``) and checks both hashes against the lock. ``backpack_models_dir``
does the same and lays the bundle out as the policy node's loader expects. It also takes the model folder copied out
of the robot's collection image, e.g. on the Orin (read-only, nothing runs):

    id=$(docker create holomotion-collect-unitree:v1.4.1-brainco-backpack-3p2-policy-freeze-20260911-r2)
    docker cp $id:/opt/holomotion/deployment/unitree_g1_ros2_29dof/install/humanoid_control/share/humanoid_control/models/HoloMotion_motion_tracking_model_v1.4.1 ./backpack_model
    docker rm $id
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from simple.teleop.holomotion_v14 import MODELS_DIR, REPO_ROOT, VENDOR_DIR

BACKPACK_DIR = VENDOR_DIR / "backpack"
BACKPACK_MESH = BACKPACK_DIR / "backpack_link.STL"
BACKPACK_LOCK = BACKPACK_DIR / "v141.brainco_backpack_3p2.model.lock"
BACKPACK_POS = (-0.094605, -0.0005, 0.165324)                          # in torso_link
BACKPACK_REF_KG = 1.9
BACKPACK_REF_INERTIA = (0.015510, 0.010909, 0.007288, 0.000017, 0.000563, 0.000059)   # ixx iyy izz ixy ixz iyz at 1.9 kg

BASE_MJCF = "robots/g1_sonic/g1_29dof_with_hand.xml"                  # G1Sonic.mjcf_path (under data/)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_backpack(spec, kg: float) -> None:
    import mujoco
    spec.add_mesh(name="backpack_link", file=str(BACKPACK_MESH))          # absolute: meshdir does not apply
    s = kg / BACKPACK_REF_KG
    body = spec.body("torso_link").add_body(name="backpack_link", pos=list(BACKPACK_POS))
    body.mass = float(kg)
    body.ipos = [0.0, 0.0, 0.0]
    body.fullinertia = [v * s for v in BACKPACK_REF_INERTIA]
    body.explicitinertial = True
    body.add_geom(name="backpack_link_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="backpack_link",
                  contype=0, conaffinity=0, group=1, density=0.0, rgba=[0.9, 0.9, 0.9, 1.0])


def _write_variant(spec, rel: str) -> str:
    spec.compile()                                                          # fail here, not inside the env
    xml = spec.to_xml()
    dst = REPO_ROOT / "data" / rel
    if not dst.exists() or dst.read_text() != xml:
        import os
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")    # atomic: a parallel run never reads a half-written file
        tmp.write_text(xml)
        os.replace(tmp, dst)
    return rel


def backpack_mjcf(kg: float, base: str = BASE_MJCF) -> str:
    """Write the G1 MJCF with a ``kg`` backpack next to the base one; returns its path relative to data/."""
    import mujoco

    tag = f"{kg:.2f}".replace(".", "p")
    rel = str(Path(base).with_name(f"{Path(base).stem}_backpack_{tag}kg.xml"))
    spec = mujoco.MjSpec.from_file(str(REPO_ROOT / "data" / base))
    _add_backpack(spec, kg)
    return _write_variant(spec, rel)


# ---- the G1 with the HBVCAM stereo head camera (third_party/hbvcam_stereo, the v1.4 collection robot's camera) ----
STEREO_DIR = REPO_ROOT / "third_party" / "hbvcam_stereo"
STEREO_MJCF = "robots/g1_sonic/g1_29dof_with_hand_hbvcam_stereo.xml"   # under data/, next to the stock meshes
PINHOLE_CAMERA = "hbvcam_left_pinhole"                                 # the rectified left eye, 1280 x 720
PINHOLE_W, PINHOLE_H = 1280, 720
FISHEYE_PREFIX = "hbvcam"                                              # face cameras hbvcam_{left,right}_{front,...}


def install_stereo_mjcf() -> str:
    """Copy the vendored stereo-camera G1 next to the stock G1 (its meshes are the stock ones); path under data/."""
    src, dst = STEREO_DIR / Path(STEREO_MJCF).name, REPO_ROOT / "data" / STEREO_MJCF
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        dst.write_bytes(src.read_bytes())
    return STEREO_MJCF


def rectified_left(calib: Path = STEREO_DIR / "calibration" / "stereo_calib.yaml"):
    """(R1, P1) of the stereo calibration: the rectified left eye is a pinhole camera (f 493.6 px, 104.7 x 72.2 deg)."""
    import cv2
    fs = cv2.FileStorage(str(calib), cv2.FILE_STORAGE_READ)
    R1, P1 = fs.getNode("R1").mat(), fs.getNode("P1").mat()
    size = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()))
    fs.release()
    assert size == (PINHOLE_W, PINHOLE_H), size
    return R1, P1


def _add_pinhole(spec) -> None:
    """The rectified left eye as a MuJoCo camera at the left optical frame: orientation R_opt R1^T (optical: x right,
    y down, z forward) turned to MuJoCo's camera axes (x right, y up, looking along -z); intrinsics from P1."""
    import mujoco
    site = next(s for s in spec.sites if s.name == "head_stereo_left_optical_frame")
    R_opt = np.zeros(9)
    mujoco.mju_quat2Mat(R_opt, np.asarray(site.quat, dtype=np.float64))
    R1, P1 = rectified_left()
    R_cam = R_opt.reshape(3, 3) @ R1.T @ np.diag([1.0, -1.0, -1.0])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_cam.reshape(-1))
    cam = site.parent.add_camera(name=PINHOLE_CAMERA, pos=list(site.pos), quat=list(quat))
    cam.resolution = [PINHOLE_W, PINHOLE_H]
    cam.sensor_size = [float(PINHOLE_W), float(PINHOLE_H)]                  # 1 length unit = 1 pixel
    cam.focal_pixel = [float(P1[0, 0]), float(P1[1, 1])]
    # MuJoCo's offset convention (measured): +x moves the principal point LEFT, +y moves it down, so OpenCV's
    # (cx, cy) is (W/2 - cx, cy - H/2); checked by projecting points, < 0.1 px
    cam.principal_pixel = [float(PINHOLE_W / 2 - P1[0, 2]), float(P1[1, 2] - PINHOLE_H / 2)]


def _tilt_cameras(spec, prefix: str, tilt_deg: float) -> None:
    """Pitch the cameras named ``prefix*`` down by ``tilt_deg`` (+ = down, - = up) about their body's y axis (the
    robot's left), each about its own position: the stereo rig as if mounted at another angle (both eyes lie on one
    line along y, so turning each about its own centre is turning the rig)."""
    import mujoco
    h = np.radians(tilt_deg) / 2.0
    qy = np.array([np.cos(h), 0.0, np.sin(h), 0.0])
    for cam in spec.cameras:
        if cam.name.startswith(prefix):
            if cam.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
                raise ValueError(f"camera {cam.name}: orientation not given as a quaternion")
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qy, np.asarray(cam.quat, dtype=np.float64))
            cam.quat = q


def _tilt_tag(tilt_deg: float) -> str:
    return "_tilt" + f"{tilt_deg:+g}".replace("+", "p").replace("-", "m").replace(".", "p")


def teleop_mjcf(robot: str = "stereo", backpack_kg: float = 3.2, tilt_deg: float = 0.0) -> str:
    """The teleop robot: "stereo" = the G1 with the HBVCAM stereo head camera plus its rectified left eye as the
    pinhole camera PINHOLE_CAMERA; "stock" = SIMPLE's G1. With a ``backpack_kg`` backpack if > 0 and the stereo
    camera pitched ``tilt_deg`` further down (+) or up (-). Path under data/."""
    import mujoco
    if robot not in ("stereo", "stock"):
        raise ValueError(f"robot is 'stereo' or 'stock', got {robot!r}")
    base = install_stereo_mjcf() if robot == "stereo" else BASE_MJCF
    if robot == "stock" and backpack_kg <= 0:
        return base
    spec = mujoco.MjSpec.from_file(str(REPO_ROOT / "data" / base))
    name = Path(base).stem
    if backpack_kg > 0:
        _add_backpack(spec, backpack_kg)
        name += "_backpack_" + f"{backpack_kg:.2f}".replace(".", "p") + "kg"
    if robot == "stereo":
        # the bundle's five 90-degree cameras per eye (head_stereo_{eye}_{face}), which the fisheye stream renders,
        # renamed hbvcam_{eye}_{face}: clear of SIMPLE's head_stereo sensor cameras
        for cam in [c for c in spec.cameras if c.name.startswith("head_stereo_")]:
            cam.name = FISHEYE_PREFIX + cam.name[len("head_stereo"):]
        _add_pinhole(spec)
        name += "_pinhole"
        if tilt_deg:
            _tilt_cameras(spec, FISHEYE_PREFIX, tilt_deg)                  # every hbvcam_* camera, the pinhole included
            name += _tilt_tag(tilt_deg)
    return _write_variant(spec, str(Path(base).with_name(name + ".xml")))


def tilt_head_sensor(task, tilt_deg: float, cam_id: str = "head_stereo") -> None:
    """Pitch the scene's head camera sensor (SIMPLE's ``head_stereo``, recorded as ego_view with --record-camera head)
    ``tilt_deg`` further down (+) or up (-), on top of the scene's own pose; call before env.reset. The scene kits
    apply a non-identity eye_in_head pose as a local offset of the stock mount (their _patch_head_camera_pose, the
    same composition as their <KIT>_HEAD_TRIM_DEG); SIMPLE's engine alone accepts only the identity there."""
    import transforms3d as t3d
    from simple.engines.mujoco import MujocoSimulator
    cfg = task.sensor_cfgs[cam_id]
    pose0 = task.__dict__.setdefault("_head_sensor_pose0", {k: list(v) for k, v in cfg.pose.items()})
    q = np.asarray(pose0["quaternion"], dtype=float)
    if tilt_deg:
        if not getattr(MujocoSimulator, "_tabletop_box_patched", False):
            raise ValueError("tilting the head camera needs one of the scene kits (their head-camera pose patch)")
        q = t3d.quaternions.qmult(q, t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], np.radians(-tilt_deg)))
    cfg.pose = dict(pose0, quaternion=[float(v) for v in q])


def read_lock(path: Path = BACKPACK_LOCK) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def backpack_models_dir(bundle: str | Path, check_hashes: bool = True) -> Path:
    """Lay out a backpack model bundle (config.yaml + model_22000.onnx) as the loader's models dir; returns it.

    The velocity model is the public one already in MODELS_DIR (the lock pins it to the same hash)."""
    lock = read_lock()
    bundle = Path(bundle).expanduser().resolve()
    name = lock["V141_BACKPACK_MODEL_FILENAME"]
    # the upstream bundle (config.yaml + model_22000.onnx) or the robot image's model folder (config.yaml + exported/)
    onnx = bundle / name if (bundle / name).is_file() else bundle / "exported" / name
    cfg = bundle / "config.yaml"
    if not onnx.is_file():
        raise FileNotFoundError(f"backpack bundle is missing {onnx.name}: {bundle}")
    if not cfg.is_file():                      # only the ONNX: the public v1.4.1 config (the observation layout is the
        cfg = (MODELS_DIR / "motion_tracking_model" / "config.yaml").resolve()     # same; gains etc. come from the ONNX)
        print(f"[HoloMotion v1.4] backpack bundle has no config.yaml: using the v1.4.1 config {cfg}")
        lock = dict(lock, V141_BACKPACK_CONFIG_SHA256=_sha256(cfg))
    vel = MODELS_DIR / "velocity_tracking_model"
    vel_onnx = next(iter(sorted((vel / "exported").glob("*.onnx"))), None) if (vel / "exported").is_dir() else None
    if check_hashes:
        checks = [(onnx, lock["V141_BACKPACK_MODEL_SHA256"]), (cfg, lock["V141_BACKPACK_CONFIG_SHA256"])]
        if vel_onnx is not None:
            checks.append((vel_onnx, lock["V141_BACKPACK_VELOCITY_MODEL_SHA256"]))
        for f, want in checks:
            got = _sha256(f)
            if got != want:
                raise ValueError(f"{f}: sha256 {got} does not match the lock ({want})")
    out = REPO_ROOT / "data" / "holomotion" / "v14_models_bundle"       # not the default BACKPACK_MODELS_DIR
    exp = out / "motion_tracking_model" / "exported"
    exp.mkdir(parents=True, exist_ok=True)
    for stale in exp.glob("*.onnx"):
        if stale.name != onnx.name:
            stale.unlink()
    links = {out / "velocity_tracking_model": vel.resolve(), out / "motion_tracking_model" / "config.yaml": cfg,
             exp / onnx.name: onnx}
    for link, target in links.items():
        if link.is_symlink() or link.exists():
            if link.is_symlink() and Path(os.readlink(link)) == target:
                continue
            link.unlink()
        link.symlink_to(target)
    return out
