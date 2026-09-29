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


def backpack_mjcf(kg: float, base: str = BASE_MJCF) -> str:
    """Write the G1 MJCF with a ``kg`` backpack next to the base one; returns its path relative to data/."""
    import mujoco

    data = REPO_ROOT / "data"
    src = data / base
    tag = f"{kg:.2f}".replace(".", "p")
    rel = str(Path(base).with_name(f"{Path(base).stem}_backpack_{tag}kg.xml"))
    dst = data / rel
    spec = mujoco.MjSpec.from_file(str(src))
    spec.add_mesh(name="backpack_link", file=str(BACKPACK_MESH))          # absolute: meshdir does not apply
    s = kg / BACKPACK_REF_KG
    body = spec.body("torso_link").add_body(name="backpack_link", pos=list(BACKPACK_POS))
    body.mass = float(kg)
    body.ipos = [0.0, 0.0, 0.0]
    body.fullinertia = [v * s for v in BACKPACK_REF_INERTIA]
    body.explicitinertial = True
    body.add_geom(name="backpack_link_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="backpack_link",
                  contype=0, conaffinity=0, group=1, density=0.0, rgba=[0.9, 0.9, 0.9, 1.0])
    spec.compile()                                                          # fail here, not inside the env
    xml = spec.to_xml()
    if not dst.exists() or dst.read_text() != xml:
        dst.write_text(xml)
    return rel


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
    for f in (onnx, cfg):
        if not f.is_file():
            raise FileNotFoundError(f"backpack bundle is missing {f.name}: {bundle}")
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
    out = REPO_ROOT / "data" / "holomotion" / "v14_models_backpack_3p2"
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
