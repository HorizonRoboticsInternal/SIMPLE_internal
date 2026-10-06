#!/usr/bin/env python3
"""SIMPLE task of the bottle / trash-bin scene for the sonic (decoupled-WBC) teleop stack, MuJoCo.

Same layout as build_scene.py / layout.json: pelvis at the origin, x toward the table, the robot's right = -y, floor at
z = 0 (sonic envs keep the ground at the origin). Table 2.07 x 0.60 x 0.72 with four legs and the cover board, the bottle
straight ahead on the table, the black bin at the table's right end (static bodies, the bottle free).

Registered on import: task uid  g1_wholebody_bottle_bin_teleop
                      gym id    simple/G1WholebodyBottleBinTeleop-v0   (SonicLocoManipEnv, robot g1_sonic)
Head camera: D455 as on the box task (640 x 360, 90 deg HFOV, servo tilt 10 deg, origin on the lens), obs key head_stereo_left.
BOTTLE_BIN_TARGET=graspnet1b:66 swaps the generated bottle for a GraspNet object (its tallest stable pose = standing).
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from simple.assets.scene_cache import fixed_asset, scene_cache_dir

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
from simple.tasks import g1_wholebody_bottle_bin_geometry as BS

TASK_UID = "g1_wholebody_bottle_bin_teleop"
ENV_ID = "simple/G1WholebodyBottleBinTeleop-v0"
ASSET_DIR = Path(os.environ.get("SIMPLE_DATA_DIR", "data")) / "real_scenes" / "bottle_bin" / "assets"
_LAYOUT_JSON = json.load(open(ASSET_DIR.parent / "layout.json"))
CACHE_DIR = scene_cache_dir("bottle_bin", _LAYOUT_JSON)
L = dict(_LAYOUT_JSON["layout"])                # from the last build_scene.py run (the confirmed layout), pelvis at the origin
# BOTTLE_BIN_ROBOT_TO_EDGE = front of the feet -> near table edge (m); the layout.json value is 0.17. A different value moves
# the robot along x, i.e. here (pelvis frame) the table, cover board, legs, bottle and bin all shift by the opposite amount.
ROBOT_TO_EDGE = float(os.environ.get("BOTTLE_BIN_ROBOT_TO_EDGE", L["pelvis_to_edge"] - L["toe_offset"]))
SHIFT = (ROBOT_TO_EDGE + L["toe_offset"]) - L["pelvis_to_edge"]
for _k in ("near_x", "far_x", "table_cx", "bottle_x", "bin_x"):
    L[_k] = L[_k] + SHIFT
L["pelvis_to_edge"] = L["pelvis_to_edge"] + SHIFT
# BOTTLE_BIN_BOTTLE_XY="x,y": bottle axis relative to the robot's start pose (overrides the layout's 9 cm / 56.4 cm placement)
# --- level knobs (2026-09-24): the same level 0/1/2/3 design as SIMPLE's benchmark sets (see ../make_levels.py) ---
ROBOT_DY = float(os.environ.get("BOTTLE_BIN_ROBOT_DY", "0"))          # robot start moved to its left (+y) by this: everything else moves -y (level 3)
for _k, _v in list(L.items()):
    if _k.endswith("y") and not _k.startswith("plan_") and isinstance(_v, (int, float)) and not isinstance(_v, bool):
        L[_k] = _v - ROBOT_DY
TARGET_JITTER = tuple(float(v) for v in os.environ.get("BOTTLE_BIN_TARGET_JITTER", "0,0").split(","))   # half-widths (x, y) of the bottle's start region (level 2/3)
NUM_DISTRACTORS = int(os.environ.get("BOTTLE_BIN_NUM_DISTRACTORS", "0"))                                   # GraspNet distractors on the table (SIMPLE uses 3)
if os.environ.get("BOTTLE_BIN_BOTTLE_XY"):
    L["bottle_x"], L["bottle_y"] = (float(v) for v in os.environ["BOTTLE_BIN_BOTTLE_XY"].split(","))
# BOTTLE_BIN_BIN_XY="x,y": bin centre relative to the robot's start pose (pelvis frame, x toward the table, right = -y)
if os.environ.get("BOTTLE_BIN_BIN_XY"):
    L["bin_x"], L["bin_y"] = (float(v) for v in os.environ["BOTTLE_BIN_BIN_XY"].split(","))
BIN_YAW_DEG = float(_LAYOUT_JSON["bin"]["yaw_deg"])
COVER_TOP_Z = BS.COVER_TOP_Z
TARGET = os.environ.get("BOTTLE_BIN_TARGET", "bottle_bin:bottle_500ml")
BOTTLE_MASS = float(os.environ.get("BOTTLE_BIN_BOTTLE_MASS", _LAYOUT_JSON["bottle"]["mass"]))   # kg, from build_scene.py (0.5 = full); SIMPLE's own default would be 0.1
INSTRUCTION = os.environ.get("BOTTLE_BIN_INSTRUCTION",
                             "pick up the bottle, move towards the trash bin, and place the bottle in the trash bin")
# 2026-09-29 (user): the bottle-bin kit's own task, as the real ChipCanToTrash recordings phrase it with the sim's bottle.
# The 2026-09-17 default "move the blue chip bag from the shelf to the plate on the counter" belonged to a different
# task and went into every recorded episode; set BOTTLE_BIN_INSTRUCTION to use another string.

# D455 lens pose of the scene (servo pan 0 / tilt 10 deg) as an offset of SIMPLE's stock eye_in_head mount, from
# sim/tabletop_box/g1_tabletop_box_simple_task.py (the engine's mount is the same for every G1 robot class)
CAM_LOCAL_POS = [0.01849, 0.03219, -0.00928]
CAM_LOCAL_QUAT = [0.996029, -0.088426, -0.007383, -0.007305]
HEAD_TILT_TRIM_DEG = float(os.environ.get("BOTTLE_BIN_HEAD_TRIM_DEG", "0"))
HEAD_HFOV_DEG, HEAD_W, HEAD_H = 90.0, 640, 360
HAND_RGBA = [float(v) for v in BS.HAND_RGBA.split()]


# ------------------------------------------------------------------ generated meshes (OBJ for MuJoCo, USDA for Isaac)
def _box(size, center=(0.0, 0.0, 0.0), yaw=0.0):
    sx, sy, sz = (s / 2 for s in size)
    v = np.array([(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float)
    if yaw:
        v = v @ R.from_euler("z", yaw).as_matrix().T
    v = v + np.asarray(center, dtype=float)
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]     # outward quads
    return v, f


def _write_obj(path: Path, pieces):
    lines, off = ["# generated by bottle_bin_task.py"], 0
    for v, f in pieces:
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
        lines += ["f " + " ".join(str(i + 1 + off) for i in q) for q in f]
        off += len(v)
    path.write_text("\n".join(lines) + "\n")


def _read_obj(path: Path):
    v, f = [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            v.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            f.append([int(w.split("/")[0]) - 1 for w in p[1:]])
    return np.array(v), f


def _write_usda(path: Path, prim: str, pieces, rgb):
    """SIMPLE's Isaac engine (engines/isaacsim.py __create_object) expects the object layout of its own assets:
    <root Xform>/Meshes (rigid body) with a visible "visual" mesh and a "collision" mesh child."""
    pts, counts, idx, off = [], [], [], 0
    for v, f in pieces:
        pts += [tuple(p) for p in v]
        for q in f:
            counts.append(len(q)); idx += [i + off for i in q]
        off += len(v)
    points = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in pts)
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "{prim}"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "{prim}"
{{
    def Xform "Meshes" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {{
        def Mesh "visual"
        {{
            int[] faceVertexCounts = {counts}
            int[] faceVertexIndices = {idx}
            point3f[] points = [{points}]
            color3f[] primvars:displayColor = [({rgb[0]}, {rgb[1]}, {rgb[2]})]
            uniform token subdivisionScheme = "none"
        }}
        def Mesh "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {{
            int[] faceVertexCounts = {counts}
            int[] faceVertexIndices = {idx}
            point3f[] points = [{points}]
            uniform token physics:approximation = "convexHull"
            uniform token purpose = "guide"
            uniform token subdivisionScheme = "none"
        }}
    }}
    def Scope "Looks"
    {{
    }}
}}
''')


def bin_pieces():
    """Bottom, four walls and 16 corner boxes in the bin's local frame (x = 24 cm depth, y = 36 cm width), bottom at z = 0."""
    hw, hd, r, t, h = BS.BIN_W / 2, BS.BIN_D / 2, BS.BIN_R, BS.BIN_T, BS.BIN_H
    pieces = [_box((BS.BIN_D, BS.BIN_W, t), (0, 0, t / 2))]
    pieces += [_box((t, 2 * (hw - r), h), (hd - t / 2, 0, h / 2)), _box((t, 2 * (hw - r), h), (-(hd - t / 2), 0, h / 2)),
               _box((2 * (hd - r), t, h), (0, hw - t / 2, h / 2)), _box((2 * (hd - r), t, h), (0, -(hw - t / 2), h / 2))]
    nseg, rm = 4, r - t / 2
    half = np.radians(90 / nseg / 2)
    for cx, cy, a0 in ((hd - r, hw - r, 0), (-hd + r, hw - r, 90), (-hd + r, -hw + r, 180), (hd - r, -hw + r, 270)):
        for k in range(nseg):
            a = np.radians(a0 + 90 * (k + 0.5) / nseg)
            pieces.append(_box((2 * rm * np.sin(half) * 1.05, t, h), (cx + rm * np.cos(a), cy + rm * np.sin(a), h / 2), yaw=a + np.pi / 2))
    return pieces


def ensure_assets() -> dict:
    """Write every mesh the task needs next to build_scene.py's own assets (idempotent)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    # bottle: build_scene.py's body + cap meshes (bottom at z = 0); one merged mesh for the placement / curobo checks
    body = fixed_asset(ASSET_DIR, "bottle_500ml_body.obj")
    cap = fixed_asset(ASSET_DIR, "bottle_500ml_cap.obj")
    merged = CACHE_DIR / "bottle_500ml.obj"
    if not merged.exists():
        _write_obj(merged, [_read_obj(body), _read_obj(cap)])
    usd = CACHE_DIR / "bottle_500ml.usda"
    if not usd.exists():
        _write_usda(usd, "bottle_500ml", [_read_obj(body), _read_obj(cap)], (0.62, 0.80, 0.95))
    files["bottle"] = dict(obj=merged, mujoco=[body, cap], usd=usd)
    # cover board: one box, origin at its centre
    cb = _box((BS.COVER_T, BS.TABLE_W, COVER_TOP_Z - BS.TABLE_H))
    p = CACHE_DIR / "cover_board.obj"; p.exists() or _write_obj(p, [cb])
    u = CACHE_DIR / "cover_board.usda"; u.exists() or _write_usda(u, "cover_board", [cb], (0.88, 0.86, 0.80))
    files["cover_board"] = dict(obj=p, mujoco=[p], usd=u)
    # table legs: four boxes in the table frame (origin = floor point under the table centre)
    leg_h = BS.TABLE_H - BS.TABLE_TOP_T
    legs = [_box((BS.LEG_SIZE, BS.LEG_SIZE, leg_h), (sx * (BS.TABLE_D / 2 - BS.LEG_INSET - BS.LEG_SIZE / 2),
                                                     sy * (BS.TABLE_W / 2 - BS.LEG_INSET - BS.LEG_SIZE / 2), leg_h / 2))
            for sx in (1, -1) for sy in (1, -1)]
    leg_objs = []
    for i, leg in enumerate(legs):
        q = CACHE_DIR / f"table_leg{i}.obj"; q.exists() or _write_obj(q, [leg]); leg_objs.append(q)
    p = CACHE_DIR / "table_legs.obj"; p.exists() or _write_obj(p, legs)
    u = CACHE_DIR / "table_legs.usda"; u.exists() or _write_usda(u, "table_legs", legs, (0.35, 0.30, 0.25))
    files["table_legs"] = dict(obj=p, mujoco=leg_objs, usd=u)
    # bin: 21 convex boxes for MuJoCo, the hollow shell (build_scene.py) for the USD
    pieces = bin_pieces()
    piece_objs = []
    for i, pc in enumerate(pieces):
        q = CACHE_DIR / f"bin_piece{i:02d}.obj"; q.exists() or _write_obj(q, [pc]); piece_objs.append(q)
    shell = fixed_asset(ASSET_DIR, "bin_36x24x39.obj")
    p = CACHE_DIR / "bin_pieces.obj"; p.exists() or _write_obj(p, pieces)
    u = CACHE_DIR / "bin_36x24x39.usda"; u.exists() or _write_usda(u, "trash_bin", [_read_obj(shell)], (0.06, 0.06, 0.06))
    files["trash_bin"] = dict(obj=p, mujoco=piece_objs, usd=u)
    return files


def standing_stable_index(asset_id: str) -> int:
    """GraspNet target: the stable pose with the largest height = standing."""
    import trimesh
    from simple.assets import AssetManager
    res, oid = asset_id.split(":")
    asset = AssetManager.get(res).load(oid)
    mesh = trimesh.load_mesh(asset.collision_mesh_curobo)
    best = None
    for k, sp in enumerate(asset.stable_poses):
        q = np.asarray(sp[3:7], float)
        h = float(np.ptp(R.from_quat([q[1], q[2], q[3], q[0]]).apply(mesh.vertices)[:, 2]))
        if best is None or h > best[0]:
            best = (h, k)
    return best[1]


# ------------------------------------------------------------------ engine patches
def _patch_static_objects():
    """Objects whose asset has static=True are welded to the world (no free joint); rgba colours them in MuJoCo."""
    import mujoco
    from simple.engines.mujoco import MujocoSimulator
    if getattr(MujocoSimulator, "_bottle_bin_patched", False):
        return
    orig = MujocoSimulator._build_object

    def build_object(self, mjSpec, mjWorld, actor):
        asset = actor.asset
        static, rgba, mass = getattr(asset, "static", False), getattr(asset, "rgba", None), getattr(asset, "mass", None) or 0.1
        if not static and rgba is None and mass == 0.1:
            return orig(self, mjSpec, mjWorld, actor)
        label = getattr(asset, "label", asset.uid)
        meshes = asset.collision_meshes_mujoco
        for i, f in enumerate(meshes):
            mjSpec.add_mesh(name=f"{label}_mesh_convex{i}", file=str(f))
        body = mjWorld.add_body(name=label, pos=actor.pose.position, quat=actor.pose.quaternion)
        for i in range(len(meshes)):
            body.add_geom(name=f"{label}_convex_{i}", meshname=f"{label}_mesh_convex{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          condim=4, mass=mass / len(meshes), friction=[0.8, 0.05, 0.005],
                          rgba=list(rgba or [1, 1, 1, 1]), solref=[0.005, 2])
        if not static:
            body.add_freejoint(name=f"{label}_joint")

    MujocoSimulator._build_object = build_object
    MujocoSimulator._bottle_bin_patched = True


def _patch_head_camera_pose():
    """Apply the configured eye_in_head pose as a local offset of SIMPLE's stock head mount (the engine only accepts
    an identity quaternion there). Copied from the tabletop_box task."""
    import transforms3d as t3d
    from simple.engines.mujoco import MujocoSimulator
    if getattr(MujocoSimulator, "_tabletop_box_patched", False):
        return
    orig_build = MujocoSimulator._build_camera

    def build_camera(self, cname, camera):
        q = np.asarray(camera.pose.quaternion, dtype=float)
        p = np.asarray(camera.pose.position, dtype=float)
        if camera.mount != "eye_in_head" or (np.allclose(q[1:], 0.0, atol=1e-6) and np.allclose(p, 0.0)):
            return orig_build(self, cname, camera)
        camera.pose.quaternion, camera.pose.position = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        orig_add = self._add_mujoco_camera

        def add_camera(parent, cam, **kw):
            r_stock = t3d.quaternions.quat2mat(kw["quat"])
            kw["pos"] = np.asarray(kw["pos"], dtype=float) + r_stock @ p
            kw["quat"] = t3d.quaternions.qmult(kw["quat"], q)
            return orig_add(parent, cam, **kw)

        self._add_mujoco_camera = add_camera
        try:
            return orig_build(self, cname, camera)
        finally:
            del self._add_mujoco_camera
            camera.pose.quaternion, camera.pose.position = q.tolist(), p.tolist()

    MujocoSimulator._build_camera = build_camera
    MujocoSimulator._tabletop_box_patched = True


def _patch_renderer_framebuffer():
    import mujoco
    if getattr(mujoco.Renderer, "_tabletop_box_patched", False):
        return
    orig_init = mujoco.Renderer.__init__

    def __init__(self, model, height=240, width=320, *args, **kwargs):
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
        orig_init(self, model, height, width, *args, **kwargs)

    mujoco.Renderer.__init__ = __init__
    mujoco.Renderer._tabletop_box_patched = True


def recolor_hands(model, rgba=HAND_RGBA, floor_rgba=None):
    """On the compiled model: Dex3 hand links black (the sonic MJCF has them light grey) and the ground plane a plain
    light grey instead of SIMPLE's blue checker (user, 2026-09-16)."""
    import mujoco
    floor_rgba = floor_rgba or [float(v) for v in BS.FLOOR_RGBA.split()]
    n = 0
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if "_hand_" in b:
            model.geom_rgba[g] = rgba; n += 1
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_matid[g] = -1; model.geom_rgba[g] = floor_rgba
    return n


# ------------------------------------------------------------------ task
def _material_cfg(parent, keep_keys):
    """SIMPLE's material randomizer draws random roughness / metallic / specular for EVERY object unless its key is
    pinned (a metallic bowl renders black in Isaac). Table and ground materials are still re-sampled (level 0+)."""
    cfg = deepcopy(parent.dr_cfgs["material"])
    cfg.material_mode = "rand_tableground"
    cfg.keep_object_shader_params = {k: {"reflection_roughness_constant": 0.5, "metallic_constant": 0.0, "specular_level": 0.0} for k in keep_keys}
    cfg.keep_robot_shader_params = {"reflection_roughness_constant": 0.5, "metallic_constant": 0.0, "specular_level": 0.0}
    return cfg

# ---- level-3 height variants (2026-09-24): the SIMPLE table height changes per scene (DRManager, SIMPLE_LEVEL3_TABLE_DZ);
# the kit's furniture follows through mesh variants named <asset>_p040 / _m040 (dz in mm, sign p/m) ------------------------
def _dz_tag(dz: float) -> str:
    return ("p" if dz >= 0 else "m") + f"{abs(dz) * 1000:03.0f}"


def _dz_from_id(asset_id: str, base: str) -> float:
    tag = asset_id[len(base) + 1:]
    return (1.0 if tag[0] == "p" else -1.0) * int(tag[1:]) / 1000.0


def _table_top(layout, fallback: float) -> float:
    """The SIMPLE table's actual top height in this reset (its pose may carry a level-3 offset)."""
    tb = layout.actors.get("table")
    if tb is None:
        return fallback
    size = getattr(tb, "size", None) or getattr(getattr(tb, "asset", None), "size", None)
    pos = getattr(getattr(tb, "pose", None), "position", None)
    if size is None or pos is None:
        return fallback
    return float(pos[2]) + float(size[2]) / 2.0


def _sync_extra(layout, am, name, asset_id, pos, quat):
    """Add / swap the kit's extra asset under key `name` so its asset uid is `asset_id`, then place it."""
    cur = layout.actors.get(name)
    if cur is not None and getattr(getattr(cur, "asset", None), "uid", name) != asset_id:
        layout.actors.pop(name)
    if name not in layout.actors:
        layout.add_object(name, am.load(asset_id))
    layout.actors[name].pose.position = list(pos)
    layout.actors[name].pose.quaternion = list(quat)


def legs_variant_files(dz: float) -> dict:
    """Table legs for a table top at TABLE_H + dz (level 3)."""
    tag = "_" + _dz_tag(dz); leg_h = BS.TABLE_H + dz - BS.TABLE_TOP_T
    legs = [_box((BS.LEG_SIZE, BS.LEG_SIZE, leg_h), (sx * (BS.TABLE_D / 2 - BS.LEG_INSET - BS.LEG_SIZE / 2),
                                                     sy * (BS.TABLE_W / 2 - BS.LEG_INSET - BS.LEG_SIZE / 2), leg_h / 2))
            for sx in (1, -1) for sy in (1, -1)]
    leg_objs = []
    for i, leg in enumerate(legs):
        q = CACHE_DIR / f"table_leg{i}{tag}.obj"; q.exists() or _write_obj(q, [leg]); leg_objs.append(q)
    pth = CACHE_DIR / f"table_legs{tag}.obj"; pth.exists() or _write_obj(pth, legs)
    u = CACHE_DIR / f"table_legs{tag}.usda"; u.exists() or _write_usda(u, f"table_legs{tag}", legs, (0.35, 0.30, 0.25))
    return dict(obj=pth, mujoco=leg_objs, usd=u)


def _define_task():
    import transforms3d as t3d
    from simple.assets import AssetManager
    from simple.core.asset import Asset
    from simple.core.object import SemanticAnnotated, SpatialAnnotated
    from simple.dr import CameraDRCfg, DistractorDRCfg, LanguageDRCfg, SpatialDRCfg, TabletopSceneDRCfg, TargetDRCfg
    from simple.dr.types import Box
    from simple.sensors import StereoCameraCfg
    from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
    from simple.tasks.registry import TaskRegistry

    class SceneAsset(Asset, SemanticAnnotated, SpatialAnnotated):
        def __init__(self, uid, name, usd_path, curobo_mesh, mujoco_meshes, stable_poses, static=False, rgba=None, mass=None):
            super().__init__(uid=uid, usd_path=str(usd_path), collision_mesh_curobo=str(curobo_mesh),
                             collision_meshes_mujoco=[str(m) for m in mujoco_meshes])
            self.name = name; self.label = uid; self.description = name
            self.stable_poses = stable_poses
            self.keypoints, self.axes, self.canonical_grasps, self.functional_grasps = {}, {}, [], {}
            self.static, self.rgba, self.mass = static, rgba, mass

        def to_dict(self):
            return {"res_id": "bottle_bin", "uid": self.uid, "label": self.label, "name": self.name,
                    "usd_path": self.usd_path, "description": self.description}

        def __repr__(self):
            return f"SceneAsset({self.uid})"

    if "bottle_bin" not in AssetManager._registry:
        @AssetManager.register("bottle_bin")
        class BottleBinAssetManager(AssetManager):
            def __init__(self):
                self.files = ensure_assets()

            def load(self, asset_id: str, *args, **kwargs):
                f = self.files
                if asset_id == "bottle_500ml":
                    return SceneAsset("bottle_500ml", "bottle", f["bottle"]["usd"], f["bottle"]["obj"], f["bottle"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=[0.62, 0.80, 0.95, 0.75], mass=BOTTLE_MASS)
                if asset_id == "cover_board":
                    return SceneAsset("cover_board", "cover board", f["cover_board"]["usd"], f["cover_board"]["obj"], f["cover_board"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.88, 0.86, 0.80, 1.0])
                if asset_id == "table_legs":
                    return SceneAsset("table_legs", "table legs", f["table_legs"]["usd"], f["table_legs"]["obj"], f["table_legs"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.35, 0.30, 0.25, 1.0])
                if asset_id.startswith("table_legs_"):
                    v = legs_variant_files(_dz_from_id(asset_id, "table_legs"))
                    return SceneAsset(asset_id, "table legs", v["usd"], v["obj"], v["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.35, 0.30, 0.25, 1.0])
                if asset_id == "trash_bin":
                    return SceneAsset("trash_bin", "trash bin", f["trash_bin"]["usd"], f["trash_bin"]["obj"], f["trash_bin"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.06, 0.06, 0.06, 1.0])
                raise ValueError(f"unknown bottle_bin asset {asset_id}")

            def sample(self, exclude=None):
                return self.load("bottle_500ml")

            def __len__(self):
                return 4

            def __iter__(self):
                yield self.load("bottle_500ml")

    cam_quat = t3d.quaternions.qmult(CAM_LOCAL_QUAT, t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], np.radians(-HEAD_TILT_TRIM_DEG))).tolist()
    stable_idx = 0 if TARGET.startswith("bottle_bin:") else standing_stable_index(TARGET)
    parent = G1WholebodyXMovePickTaskTeleop

    @TaskRegistry.register(TASK_UID)
    class G1WholebodyBottleBinTeleop(parent):
        uid: str = TASK_UID
        label: str = "G1 Bottle Bin Teleop"
        isaac_cameras_follow_mujoco = True          # the Isaac head camera copies the MuJoCo (patched, measured D455) camera pose every step
        description: str = ("G1 at the left end of a 2.07 x 0.60 x 0.72 m table with a cover board, a 500 ml bottle straight "
                            "ahead of it and a black trash bin on the floor at the table's right end.")
        metadata = dict(parent.metadata)
        sensor_cfgs = dict(
            head_stereo=StereoCameraCfg(uid="Realsense_D455", mount="eye_in_head", width=HEAD_W, height=HEAD_H,
                                        focal_length=1.93, fov=np.deg2rad(HEAD_HFOV_DEG), near=0.05, far=5, baseline=0.05,
                                        pose=dict(position=list(CAM_LOCAL_POS), quaternion=cam_quat)))
        dr_cfgs = dict(
            language=LanguageDRCfg(instructions=[INSTRUCTION]),
            target=TargetDRCfg(asset_id=TARGET),
            distractors=DistractorDRCfg(res_id="graspnet1b", number_of_distractors=NUM_DISTRACTORS, allow_duplicates=False, exclude=[]),
            spatial=SpatialDRCfg(
                spatial_mode="random",                                       # zero-width boxes = fixed placement
                robot_region=Box(low=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0]),
                target_region=Box(low=[L["bottle_x"] - TARGET_JITTER[0], L["bottle_y"] - TARGET_JITTER[1]], high=[L["bottle_x"] + TARGET_JITTER[0], L["bottle_y"] + TARGET_JITTER[1]]),
                distractors_region=Box(low=[L["near_x"] + 0.1, L["right_y"] + 0.3], high=[L["far_x"] - 0.1, L["left_y"] - 0.3]),
                target_stable_indices=[stable_idx], target_rotate_z=Box(low=0.0, high=0.0),
            ),
            camera=CameraDRCfg(cam_id="head_stereo"),
            scene=TabletopSceneDRCfg(
                scene_mode="fixed",
                table_size=Box(low=[BS.TABLE_D, BS.TABLE_W, BS.TABLE_TOP_T], high=[BS.TABLE_D, BS.TABLE_W, BS.TABLE_TOP_T]),
                table_position=Box(low=[L["table_cx"], L["table_cy"]], high=[L["table_cx"], L["table_cy"]]),
                table_height=Box(low=BS.TABLE_H, high=BS.TABLE_H),
                rotation_z=Box(low=0.0, high=0.0),
                room_choices=[os.environ.get("BOTTLE_BIN_ROOM", "hssd:scene2")], scene_manager="hssd", randomize_scene_pose=False,
            ),
            lighting=deepcopy(parent.dr_cfgs["lighting"]),
            material=_material_cfg(parent, ["target", "cover_board", "table_legs", "trash_bin"]),   # table + ground re-sampled per level; the target and the furniture keep a plain look
        )

        # ---- task gates (2026-09-23) -------------------------------------------------------------------
        # Three checks in the order the task happens, each latched once it is met; the task succeeds when the
        # third is met, as in SIMPLE's other tasks (check_success = reward >= success_criteria, default 0.9):
        #   grasped   the right hand touches the bottle, the bottle is off the table (lifted GRASP_LIFT, or no
        #             longer touching it -- some operators carry it a centimetre up) and it stays within
        #             GRASP_TILT of upright for GRASP_HOLD_S (a can that is knocked over and scooped up for a
        #             moment does not count)
        #   at_bin    after grasping, the pelvis has walked at least MIN_WALK from the start, is within
        #             AT_BIN_RADIUS of the bin centre and has stood still (< AT_BIN_SPEED) for AT_BIN_HOLD_S
        #   placed    after grasping, the bottle axis is inside the bin's inner opening, below the rim, the right
        #             hand has let go, and it stays there for PLACE_HOLD_S (a can that bounces off the rim and out
        #             does not count)
        # info["task_progress"] carries the flags, the times they were met and robot_touched_bin (a diagnostic:
        # the robot walked into the bin; not a gate).  The bin's pose is read from the simulator each step, so
        # moving or turning it at run time (sweep_replay.py) is respected.
        GRASP_LIFT = 0.03
        GRASP_HOLD_S = 0.5
        GRASP_TILT = 60.0                                  # deg, can axis from vertical
        PLACE_HOLD_S = 0.5
        MIN_WALK = 0.50
        AT_BIN_RADIUS = 0.80
        AT_BIN_SPEED = 0.15
        AT_BIN_HOLD_S = 0.4
        STEP_DT = 0.02                                     # env.step at render_hz = 50

        def reset(self, seed=None, options=None):
            super().reset(seed, options)
            self._progress = dict(grasped=False, at_bin=False, placed=False, robot_touched_bin=False,
                                  t_grasped=None, t_at_bin=None, t_placed=None, t_robot_touched_bin=None,
                                  max_lift=0.0, min_pelvis_to_bin=None)
            self._gate_steps = 0
            self._gate_info_id = None
            self._hold_steps = 0
            self._in_bin_steps = 0
            self._still_steps = 0
            self._start_xy = None
            self._gate_model_id = None
            am = AssetManager.get("bottle_bin")
            bq = R.from_euler("z", BIN_YAW_DEG, degrees=True).as_quat()
            dz = round(_table_top(self._layout, BS.TABLE_H) - BS.TABLE_H, 3); self._table_dz = dz      # level 3: table height offset
            extra = {
                "cover_board": ("cover_board", [L["far_x"] - BS.COVER_T / 2, L["table_cy"], (BS.TABLE_H + COVER_TOP_Z) / 2 + dz], [1.0, 0.0, 0.0, 0.0]),
                "table_legs": ("table_legs" if abs(dz) < 1e-3 else "table_legs_" + _dz_tag(dz), [L["table_cx"], L["table_cy"], 0.0], [1.0, 0.0, 0.0, 0.0]),
                "trash_bin": ("trash_bin", [L["bin_x"], L["bin_y"], 0.0], [float(bq[3]), float(bq[0]), float(bq[1]), float(bq[2])]),
            }
            for name, (asset_id, pos, quat) in extra.items():
                _sync_extra(self._layout, am, name, asset_id, pos, quat)
            self._instruction = INSTRUCTION

        # ---- gate helpers ----------------------------------------------------------------------------
        def _bind_gate_ids(self, m):
            import mujoco
            if self._gate_model_id == id(m):
                return
            bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            self._g_target = bid(str(self.target.asset.label))
            self._g_bin = bid("trash_bin")
            self._g_target_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self._g_target}
            self._g_bin_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self._g_bin}
            names = {b: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(m.nbody)}
            self._g_rhand_geoms = {g for g in range(m.ngeom) if names[m.geom_bodyid[g]].startswith("right_hand_")}
            self._g_table_geoms = {g for g in range(m.ngeom) if "table" in names[m.geom_bodyid[g]]}
            pelvis = bid("pelvis")
            root = int(m.body_rootid[pelvis]) if pelvis >= 0 else -1
            self._g_robot_geoms = {g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root}
            self._gate_model_id = id(m)

        def _contacts(self, d):
            """(right hand touches the bottle, robot touches the bin, bottle touches the table) from the contact list."""
            hand, bin_, table = False, False, False
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                t1, t2 = g1 in self._g_target_geoms, g2 in self._g_target_geoms
                if (t1 and g2 in self._g_rhand_geoms) or (t2 and g1 in self._g_rhand_geoms):
                    hand = True
                if (t1 and g2 in self._g_table_geoms) or (t2 and g1 in self._g_table_geoms):
                    table = True
                if (g1 in self._g_bin_geoms and g2 in self._g_robot_geoms) or (g2 in self._g_bin_geoms and g1 in self._g_robot_geoms):
                    bin_ = True
            return hand, bin_, table

        def _in_bin(self, target_xyz, bin_xyz, bin_quat_wxyz):
            """Bottle axis inside the inner opening and below the rim.  BIN_W (0.36) lies along world x when the
            bin is turned +-90 deg, along world y at 0 deg (as build_scene.py and replay_in_scene.py take it)."""
            w, x, y, z = bin_quat_wxyz
            yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
            along_x, along_y = (BS.BIN_W, BS.BIN_D) if abs(round(yaw)) % 180 == 90 else (BS.BIN_D, BS.BIN_W)
            hx, hy = along_x / 2 - BS.BIN_T - BS.BOTTLE_R, along_y / 2 - BS.BIN_T - BS.BOTTLE_R
            dx, dy = target_xyz[0] - bin_xyz[0], target_xyz[1] - bin_xyz[1]
            return bool(abs(dx) <= hx and abs(dy) <= hy and target_xyz[2] < BS.BIN_H)

        def update_gates(self, info, mujoco_env=None):
            if mujoco_env is None or "target" not in info or "trash_bin" not in info:
                return self._progress
            if self._gate_info_id == id(info):                 # env.step calls compute_reward and then check_success
                return self._progress                          # on the same info dict: evaluate once per step
            self._gate_info_id = id(info)
            m, d = mujoco_env.mjModel, mujoco_env.mjData
            self._bind_gate_ids(m)
            P = self._progress
            self._gate_steps += 1
            t = round(self._gate_steps * self.STEP_DT, 2)
            target = np.asarray(info["target"], dtype=float)
            bin_pose = np.asarray(info["trash_bin"], dtype=float)
            if self._init_target_height is None:
                self._init_target_height = float(target[2])
            lift = float(target[2] - self._init_target_height)
            P["max_lift"] = max(P["max_lift"], round(lift, 4))
            hand, touched, on_table = self._contacts(d)
            if touched and not P["robot_touched_bin"]:
                P["robot_touched_bin"], P["t_robot_touched_bin"] = True, t
            pelvis_xy = np.asarray(d.qpos[:2], dtype=float)
            if self._start_xy is None:
                self._start_xy = pelvis_xy.copy()
            # 1. grasped: held up, upright, for a while
            qw, qx, qy, qz = target[3:7]
            tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1))))
            holding = hand and (lift >= self.GRASP_LIFT or not on_table) and tilt < self.GRASP_TILT
            self._hold_steps = self._hold_steps + 1 if holding else 0
            if not P["grasped"] and self._hold_steps * self.STEP_DT >= self.GRASP_HOLD_S:
                P["grasped"], P["t_grasped"] = True, t
            # 2. stopped at the bin
            dist = float(np.hypot(*(pelvis_xy - bin_pose[:2])))
            P["min_pelvis_to_bin"] = round(dist, 3) if P["min_pelvis_to_bin"] is None else min(P["min_pelvis_to_bin"], round(dist, 3))
            if P["grasped"] and not P["at_bin"]:
                speed = float(np.hypot(*np.asarray(d.qvel[:2], dtype=float)))
                walked = float(np.hypot(*(pelvis_xy - self._start_xy)))
                self._still_steps = self._still_steps + 1 if speed < self.AT_BIN_SPEED else 0
                if dist <= self.AT_BIN_RADIUS and walked >= self.MIN_WALK and self._still_steps * self.STEP_DT >= self.AT_BIN_HOLD_S:
                    P["at_bin"], P["t_at_bin"] = True, t
            # 3. placed: in the bin, let go of, and still there half a second later
            inside = P["grasped"] and not hand and self._in_bin(target, bin_pose[:3], bin_pose[3:7])
            self._in_bin_steps = self._in_bin_steps + 1 if inside else 0
            if not P["placed"] and self._in_bin_steps * self.STEP_DT >= self.PLACE_HOLD_S:
                P["placed"], P["t_placed"] = True, t
            P["t"] = t                                         # seconds since reset
            info["task_progress"] = dict(P)
            return P

        def compute_reward(self, info, *args, **kwargs):
            P = self.update_gates(info, kwargs.get("mujoco_env"))
            if P["placed"]:
                return 1.0
            return 0.3 * float(P["grasped"]) + 0.3 * float(P["at_bin"])

        def check_success(self, info, *args, **kwargs):
            """The task succeeds when the bottle has been placed in the bin (reward 1.0 >= success_criteria)."""
            return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    return G1WholebodyBottleBinTeleop


def register():
    import gymnasium as gym
    import simple.envs  # noqa: F401  stock registrations
    _patch_renderer_framebuffer()
    _patch_head_camera_pose()
    _patch_static_objects()
    task_cls = _define_task()
    return task_cls


G1WholebodyBottleBinTeleop = register()
