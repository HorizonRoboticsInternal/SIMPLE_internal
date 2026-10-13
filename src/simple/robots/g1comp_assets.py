# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""Fixed visual g1comp head, from the private 2026-10-12 samebg bundle.

All generated files belong in SIMPLE_SCENE_CACHE_DIR, never the asset directory.
The transform and lens-origin calculation intentionally match the donor generator.
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

RUNTIME_VERSION = "g1comp-pan0-tilt10-v1"
HEAD_MESHES = ("head_servo_link.STL", "xl330_link.STL", "d455_link.STL")
EGO_CAM_QUAT = [0.64367383, 0.26523914, -0.27106013, -0.66472446]
HEAD_CHAIN = (
    ("head_servo_link", [0.0039635, 0, -0.047], [1, 0, 0, 0]),
    ("xl330_link", [0.030518, 0, 0.52486], [0.99980034, 0, 0.01998267, 0]),
    ("d455_link", [0.0295, 0, 0.013], [0.9961947, 0, 0.087155743, 0]),
)


def asset_directory():
    from simple.utils import resolve_data_path

    override = os.environ.get("SIMPLE_G1COMP_ASSET_DIR")
    return (
        Path(override)
        if override
        else Path(resolve_data_path("robots/g1comp/head_meshes"))
    )


def validate_assets():
    paths = tuple(asset_directory() / name for name in HEAD_MESHES)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing g1comp head mesh {path}; set SIMPLE_G1COMP_ASSET_DIR"
            )
    return paths


def build_mjcf(source):
    from simple.assets.scene_cache import scene_cache_dir

    source = Path(source).resolve()
    meshes = validate_assets()
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    meshdir = source.parent / compiler.get("meshdir", "")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
    compiler.attrib.pop("meshdir", None)
    asset = root.find("asset")
    parent = root.find(".//body[@name='torso_link']")
    if parent is None:
        raise ValueError("G1 SONIC MJCF has no torso_link")
    for (name, pos, quat), mesh in zip(HEAD_CHAIN, meshes):
        ET.SubElement(asset, "mesh", name=name, file=str(mesh.resolve()))
        parent = ET.SubElement(
            parent,
            "body",
            name=name,
            pos=" ".join(map(str, pos)),
            quat=" ".join(map(str, quat)),
        )
        ET.SubElement(
            parent,
            "geom",
            type="mesh",
            mesh=name,
            contype="0",
            conaffinity="0",
            group="1",
            density="0",
            rgba="0.9 0.9 0.9 1" if name == "d455_link" else "0.2 0.2 0.2 1",
        )
    output = (
        scene_cache_dir("g1comp", {"version": RUNTIME_VERSION, "source": str(source)})
        / "g1comp_sonic_pan0_tilt10.xml"
    )
    ET.ElementTree(root).write(output, encoding="unicode")
    return output


def camera_position(mjcf):
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "d455_link")
    vertices = model.mesh_vert[
        model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]
    ]
    # Preserve the donor calculation, including its use of compiled mesh vertices.
    w, x, y, z = EGO_CAM_QUAT
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    axis = rotation @ np.array([0.0, 0.0, -1.0])
    return axis * (float((vertices @ axis).max()) + 0.002)


def add_isaac_head(stage, robot_path):
    """Attach visual meshes to the animated torso; no USD physics schemas."""
    import numpy as np
    import trimesh
    from pxr import Gf, UsdGeom

    parent = robot_path + "/torso_link"
    if not stage.GetPrimAtPath(parent).IsValid():
        raise RuntimeError(f"Missing Isaac torso prim: {parent}")
    for (name, pos, quat), mesh_path in zip(HEAD_CHAIN, validate_assets()):
        parent += "/" + name
        xform = UsdGeom.Xform.Define(stage, parent)
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
        source = trimesh.load(str(mesh_path), force="mesh", process=False)
        mesh = UsdGeom.Mesh.Define(stage, parent + "/visual")
        mesh.CreatePointsAttr(np.asarray(source.vertices, dtype=np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(source.faces))
        mesh.CreateFaceVertexIndicesAttr(source.faces.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr(
            [(0.9, 0.9, 0.9) if name == "d455_link" else (0.2, 0.2, 0.2)]
        )
