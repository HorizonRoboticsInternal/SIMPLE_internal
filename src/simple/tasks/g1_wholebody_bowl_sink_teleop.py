#!/usr/bin/env python3
"""SIMPLE task of the bowl-to-sink kitchen scene for the sonic (decoupled-WBC) teleop stack, MuJoCo.

Same layout as build_scene.py / layout.json: pelvis at the origin, x toward the bowl counter, the robot's right = -y, floor
at z = 0. SIMPLE's table primitive is the bowl counter's top slab; the cabinet below it, the back unit and the sink counter
(cabinet, slab pieces, basin) are static assets; the green bowl (15 cm, hollow) is the free target.

Registered on import: task uid  g1_wholebody_bowl_sink_teleop
                      gym id    simple/G1WholebodyBowlSinkTeleop-v0   (SonicLocoManipEnv, robot g1_sonic)
Head camera: D455 as on the box task (640 x 360, 90 deg HFOV, servo tilt 10 deg), obs key head_stereo_left.
Env vars: BOWL_SINK_ROBOT_TO_EDGE, BOWL_SINK_BOWL_XY="x,y", BOWL_SINK_SINK_ALONG, BOWL_SINK_BOWL_MASS, BOWL_SINK_INSTRUCTION.
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
from simple.tasks import g1_wholebody_bowl_sink_geometry as BS
from simple.tasks.g1_wholebody_bowl_sink_gates import BowlSinkGates, GateCfg, ContactProbe, yaw_wxyz  # noqa: E402  the four task gates (2026-09-23)

TASK_UID = "g1_wholebody_bowl_sink_teleop"
ENV_ID = "simple/G1WholebodyBowlSinkTeleop-v0"
ASSET_DIR = Path(os.environ.get("SIMPLE_DATA_DIR", "data")) / "real_scenes" / "bowl_sink" / "assets"
_LAYOUT_JSON = json.load(open(ASSET_DIR.parent / "layout.json"))
CACHE_DIR = scene_cache_dir("bowl_sink", _LAYOUT_JSON)
L = dict(_LAYOUT_JSON["layout"])                # from the last build_scene.py run, pelvis at the origin
ROBOT_TO_EDGE = float(os.environ.get("BOWL_SINK_ROBOT_TO_EDGE", L["pelvis_to_edge"] - L["toe_offset"]))
SHIFT = (ROBOT_TO_EDGE + L["toe_offset"]) - L["pelvis_to_edge"]       # a different start distance moves everything but the robot
for _k in ("near_x", "far_x", "x_wall", "counter_cx", "bowl_x", "sink_cx", "sink_x0", "basin_cx"):
    L[_k] = L[_k] + SHIFT
L["pelvis_to_edge"] = L["pelvis_to_edge"] + SHIFT
# --- level knobs (2026-09-24): the same level 0/1/2/3 design as SIMPLE's benchmark sets (see ../make_levels.py) ---
ROBOT_DY = float(os.environ.get("BOWL_SINK_ROBOT_DY", "0"))          # robot start moved to its left (+y) by this: everything else moves -y (level 3)
for _k, _v in list(L.items()):
    if _k.endswith("y") and not _k.startswith("plan_") and isinstance(_v, (int, float)) and not isinstance(_v, bool):
        L[_k] = _v - ROBOT_DY
TARGET_JITTER = tuple(float(v) for v in os.environ.get("BOWL_SINK_TARGET_JITTER", "0,0").split(","))   # half-widths (x, y) of the bowl's start region (level 2/3)
NUM_DISTRACTORS = int(os.environ.get("BOWL_SINK_NUM_DISTRACTORS", "0"))                                   # GraspNet distractors on the table (SIMPLE uses 3)
if os.environ.get("BOWL_SINK_BOWL_XY"):
    L["bowl_x"], L["bowl_y"] = (float(v) for v in os.environ["BOWL_SINK_BOWL_XY"].split(","))
if os.environ.get("BOWL_SINK_SINK_ALONG"):
    L["sink_along"] = float(os.environ["BOWL_SINK_SINK_ALONG"])
    _ref = os.environ.get("BOWL_SINK_SINK_REF", L.get("sink_ref", "far-edge"))       # what the 127 reaches
    _rx = L["near_x"] - L["sink_along"]
    L["basin_cx"] = {"far-edge": _rx + BS.BASIN_L / 2, "near-edge": _rx - BS.BASIN_L / 2, "centre": _rx, "end": _rx + BS.BASIN_L / 2}[_ref]   # BOWL_SINK_SINK_ALONG is always front edge -> far edge
    L["sink_ref"] = _ref
TARGET = os.environ.get("BOWL_SINK_TARGET", "bowl_sink:bowl_15cm")
BOWL_MASS = float(os.environ.get("BOWL_SINK_BOWL_MASS", _LAYOUT_JSON["bowl"]["mass"]))
INSTRUCTION = os.environ.get("BOWL_SINK_INSTRUCTION",
                             "pick up the green bowl from the counter, turn right, move towards the sink, and place the bowl in the sink")

# D455 lens pose of the scene (servo pan 0 / tilt 10 deg) as an offset of SIMPLE's stock eye_in_head mount, from
# sim/tabletop_box/g1_tabletop_box_simple_task.py (the engine's mount is the same for every G1 robot class)
CAM_LOCAL_POS = [0.01849, 0.03219, -0.00928]
CAM_LOCAL_QUAT = [0.996029, -0.088426, -0.007383, -0.007305]
HEAD_TILT_TRIM_DEG = float(os.environ.get("BOWL_SINK_HEAD_TRIM_DEG", "0"))
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
    lines, off = ["# generated by bowl_sink_task.py"], 0
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


def _tilted_box(size, center, yaw, tilt_y):
    """Box rotated by tilt about its own y axis, then yaw about z, then translated (the bowl wall pieces)."""
    v, f = _box(size)
    v = v @ (R.from_euler("z", yaw) * R.from_euler("y", tilt_y)).as_matrix().T + np.asarray(center)
    return v, f


def bowl_pieces(nseg=16):
    """Same pieces as build_scene.bowl_collision_geoms: a floor disc (as a 24-gon prism) + tilted wall boxes."""
    pieces = []
    r, t = BS.BOWL_R_BOT, BS.BOWL_T
    n = 24
    disc_v = [(r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n), z) for z in (0.0, t) for k in range(n)]
    disc_f = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for k in range(n):
        k1 = (k + 1) % n; disc_f.append((k, k1, n + k1, n + k))
    pieces.append((np.array(disc_v), disc_f))
    z0 = 0.008; r0 = BS.BOWL_R_BOT + (BS.BOWL_R_TOP - BS.BOWL_R_BOT) * z0 / BS.BOWL_H
    slant = np.arctan2(BS.BOWL_R_TOP - BS.BOWL_R_BOT, BS.BOWL_H); wall_len = np.hypot(BS.BOWL_R_TOP - r0, BS.BOWL_H - z0)
    r_mid = (BS.BOWL_R_TOP + r0) / 2 - t / 2; half_w = r_mid * np.sin(np.pi / nseg) * 1.05
    for k in range(nseg):
        a = 2 * np.pi * (k + 0.5) / nseg
        pieces.append(_tilted_box((t, 2 * half_w, wall_len), (r_mid * np.cos(a), r_mid * np.sin(a), (z0 + BS.BOWL_H) / 2), a, slant))
    return pieces


def sink_pieces(h=None):
    """Sink counter in its own frame (centre at the floor under the counter centre): cabinet, four slab pieces, basin floor + walls.
    h = the counter top height (default BS.SINK_H; level 3 passes SINK_H + dz)."""
    Lq = L
    H = BS.SINK_H if h is None else float(h)
    body_h = H - BS.SINK_TOP_T; zt = H - BS.SINK_TOP_T / 2
    bx, by = Lq["basin_cx"] - Lq["sink_cx"], Lq["basin_cy"] - Lq["sink_cy"]
    pcs = [_box((BS.SINK_W, BS.SINK_D, body_h), (0, 0, body_h / 2))]
    xl0, xl1 = -BS.SINK_W / 2, bx - BS.BASIN_L / 2; xr0, xr1 = bx + BS.BASIN_L / 2, BS.SINK_W / 2
    pcs.append(_box((xl1 - xl0, BS.SINK_D, BS.SINK_TOP_T), ((xl0 + xl1) / 2, 0, zt)))
    pcs.append(_box((xr1 - xr0, BS.SINK_D, BS.SINK_TOP_T), ((xr0 + xr1) / 2, 0, zt)))
    yb0, yb1 = by - BS.BASIN_ACROSS / 2, by + BS.BASIN_ACROSS / 2
    pcs.append(_box((BS.BASIN_L, yb0 + BS.SINK_D / 2, BS.SINK_TOP_T), (bx, (yb0 - BS.SINK_D / 2) / 2, zt)))
    pcs.append(_box((BS.BASIN_L, BS.SINK_D / 2 - yb1, BS.SINK_TOP_T), (bx, (yb1 + BS.SINK_D / 2) / 2, zt)))
    pcs.append(_box((BS.BASIN_L, BS.BASIN_ACROSS, 0.01), (bx, by, body_h + 0.005)))
    d = BS.BASIN_DEPTH
    r = float(Lq.get("basin_radius", BS.BASIN_R))          # rounded basin corners (user 2026-09-23)
    hw, ha = BS.BASIN_L / 2, BS.BASIN_ACROSS / 2
    pcs += [_box((0.01, BS.BASIN_ACROSS - 2 * r, d), (bx - hw + 0.005, by, body_h + d / 2)),
            _box((0.01, BS.BASIN_ACROSS - 2 * r, d), (bx + hw - 0.005, by, body_h + d / 2)),
            _box((BS.BASIN_L - 2 * r, 0.01, d), (bx, by - ha + 0.005, body_h + d / 2)),
            _box((BS.BASIN_L - 2 * r, 0.01, d), (bx, by + ha - 0.005, body_h + d / 2))]
    for ccx, ccy, a0 in ((hw - r, ha - r, 0), (-hw + r, ha - r, 90), (-hw + r, -ha + r, 180), (hw - r, -ha + r, 270)):
        for k in range(4):                                  # tangent segments around each corner arc
            a = np.radians(a0 + 90 * (k + 0.5) / 4); half = np.radians(90 / 8); rm = r - 0.005
            pcs.append(_box((2 * rm * np.sin(half) * 1.15, 0.01, d),
                            (bx + ccx + rm * np.cos(a), by + ccy + rm * np.sin(a), body_h + d / 2),
                            yaw=float(a + np.pi / 2)))
    return pcs


def ensure_assets() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    bowl = fixed_asset(ASSET_DIR, "bowl_15cm.obj")
    pieces = bowl_pieces(); objs = []
    for i, pc in enumerate(pieces):
        q = CACHE_DIR / f"bowl_piece{i:02d}.obj"; q.exists() or _write_obj(q, [pc]); objs.append(q)
    u = CACHE_DIR / "bowl_15cm.usda"; u.exists() or _write_usda(u, "bowl_15cm", [_read_obj(bowl)], (0.45, 0.85, 0.12))
    files["bowl"] = dict(obj=bowl, mujoco=objs, usd=u)
    # bowl counter cabinet (below SIMPLE's table slab) + back unit, in the counter frame (floor point under the counter centre)
    body_h = BS.COUNTER_H - BS.COUNTER_TOP_T
    cab = [_box((BS.COUNTER_D, BS.COUNTER_W, body_h), (0, 0, body_h / 2)),
           _box((BS.BACK_D, BS.COUNTER_W, BS.BACK_H), (BS.COUNTER_D / 2 + BS.BACK_D / 2, 0, BS.BACK_H / 2))]
    objs = []
    for i, pc in enumerate(cab):
        q = CACHE_DIR / f"counter_piece{i}.obj"; q.exists() or _write_obj(q, [pc]); objs.append(q)
    p = CACHE_DIR / "counter.obj"; p.exists() or _write_obj(p, cab)
    u = CACHE_DIR / "counter.usda"; u.exists() or _write_usda(u, "counter", cab, (0.93, 0.93, 0.91))
    files["counter"] = dict(obj=p, mujoco=objs, usd=u)
    sp = sink_pieces(); objs = []
    for i, pc in enumerate(sp):
        q = CACHE_DIR / f"sink_piece{i:02d}.obj"; _write_obj(q, [pc]); objs.append(q)     # always rewritten: depends on the basin position
    p = CACHE_DIR / "sink_counter.obj"; _write_obj(p, sp)
    u = CACHE_DIR / "sink_counter.usda"; _write_usda(u, "sink_counter", sp, (0.93, 0.93, 0.91))
    files["sink"] = dict(obj=p, mujoco=objs, usd=u)
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
    if getattr(MujocoSimulator, "_bowl_sink_patched", False):
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
    MujocoSimulator._bowl_sink_patched = True


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


def counter_variant_files(dz: float) -> dict:
    """Bowl-counter cabinet for a slab at COUNTER_H + dz (the back unit keeps its 1.07 m)."""
    tag = "_" + _dz_tag(dz); body_h = BS.COUNTER_H + dz - BS.COUNTER_TOP_T
    cab = [_box((BS.COUNTER_D, BS.COUNTER_W, body_h), (0, 0, body_h / 2)),
           _box((BS.BACK_D, BS.COUNTER_W, BS.BACK_H), (BS.COUNTER_D / 2 + BS.BACK_D / 2, 0, BS.BACK_H / 2))]
    objs = []
    for i, pc in enumerate(cab):
        q = CACHE_DIR / f"counter_piece{i}{tag}.obj"; q.exists() or _write_obj(q, [pc]); objs.append(q)
    pth = CACHE_DIR / f"counter{tag}.obj"; pth.exists() or _write_obj(pth, cab)
    u = CACHE_DIR / f"counter{tag}.usda"; u.exists() or _write_usda(u, f"counter{tag}", cab, (0.93, 0.93, 0.91))
    return dict(obj=pth, mujoco=objs, usd=u)


def sink_variant_files(dz: float) -> dict:
    """Sink counter for a top at SINK_H + dz (basin and rim move with it)."""
    tag = "_" + _dz_tag(dz); sp = sink_pieces(BS.SINK_H + dz); objs = []
    for i, pc in enumerate(sp):
        q = CACHE_DIR / f"sink_piece{i:02d}{tag}.obj"; _write_obj(q, [pc]); objs.append(q)
    pth = CACHE_DIR / f"sink_counter{tag}.obj"; _write_obj(pth, sp)
    u = CACHE_DIR / f"sink_counter{tag}.usda"; _write_usda(u, f"sink_counter{tag}", sp, (0.93, 0.93, 0.91))
    return dict(obj=pth, mujoco=objs, usd=u)


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
            return {"res_id": "bowl_sink", "uid": self.uid, "label": self.label, "name": self.name,
                    "usd_path": self.usd_path, "description": self.description}

        def __repr__(self):
            return f"SceneAsset({self.uid})"

    if "bowl_sink" not in AssetManager._registry:
        @AssetManager.register("bowl_sink")
        class BowlSinkAssetManager(AssetManager):
            def __init__(self):
                self.files = ensure_assets()

            def load(self, asset_id: str, *args, **kwargs):
                f = self.files
                if asset_id == "bowl_15cm":
                    return SceneAsset("bowl_15cm", "green bowl", f["bowl"]["usd"], f["bowl"]["obj"], f["bowl"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=[0.45, 0.85, 0.12, 1.0], mass=BOWL_MASS)
                if asset_id == "counter":
                    return SceneAsset("counter", "counter cabinet and back unit", f["counter"]["usd"], f["counter"]["obj"], f["counter"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.93, 0.93, 0.91, 1.0])
                if asset_id == "sink_counter":
                    return SceneAsset("sink_counter", "sink counter", f["sink"]["usd"], f["sink"]["obj"], f["sink"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.90, 0.91, 0.90, 1.0])
                if asset_id.startswith("counter_"):
                    v = counter_variant_files(_dz_from_id(asset_id, "counter"))
                    return SceneAsset(asset_id, "counter cabinet and back unit", v["usd"], v["obj"], v["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.93, 0.93, 0.91, 1.0])
                if asset_id.startswith("sink_counter_"):
                    v = sink_variant_files(_dz_from_id(asset_id, "sink_counter"))
                    return SceneAsset(asset_id, "sink counter", v["usd"], v["obj"], v["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.90, 0.91, 0.90, 1.0])
                raise ValueError(f"unknown bowl_sink asset {asset_id}")

            def sample(self, exclude=None):
                return self.load("bowl_15cm")

            def __len__(self):
                return 3

            def __iter__(self):
                yield self.load("bowl_15cm")

    cam_quat = t3d.quaternions.qmult(CAM_LOCAL_QUAT, t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], np.radians(-HEAD_TILT_TRIM_DEG))).tolist()
    stable_idx = 0 if TARGET.startswith("bowl_sink:") else standing_stable_index(TARGET)
    parent = G1WholebodyXMovePickTaskTeleop

    @TaskRegistry.register(TASK_UID)
    class G1WholebodyBowlSinkTeleop(parent):
        isaac_hidden_scene_groups = ("furniture",)
        uid: str = TASK_UID
        label: str = "G1 Bowl Sink Teleop"
        isaac_cameras_follow_mujoco = True          # the Isaac head camera copies the MuJoCo (patched, measured D455) camera pose every step
        description: str = ("G1 facing a 2.60 x 0.63 x 0.86 m kitchen counter with a green 15 cm bowl on it; a sink counter of the same "
                            "0.86 m height runs along the wall on its right, its basin's far edge 1.27 m behind the counter's front edge.")
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
                target_region=Box(low=[L["bowl_x"] - TARGET_JITTER[0], L["bowl_y"] - TARGET_JITTER[1]], high=[L["bowl_x"] + TARGET_JITTER[0], L["bowl_y"] + TARGET_JITTER[1]]),
                distractors_region=Box(low=[L["near_x"] + 0.1, L["right_y"] + 0.3], high=[L["far_x"] - 0.1, L["left_y"] - 0.3]),
                target_stable_indices=[stable_idx], target_rotate_z=Box(low=0.0, high=0.0),
            ),
            camera=CameraDRCfg(cam_id="head_stereo"),
            scene=TabletopSceneDRCfg(
                scene_mode="fixed",
                table_size=Box(low=[BS.COUNTER_D, BS.COUNTER_W, BS.COUNTER_TOP_T], high=[BS.COUNTER_D, BS.COUNTER_W, BS.COUNTER_TOP_T]),
                table_position=Box(low=[L["counter_cx"], L["counter_cy"]], high=[L["counter_cx"], L["counter_cy"]]),
                table_height=Box(low=BS.COUNTER_H, high=BS.COUNTER_H),
                rotation_z=Box(low=0.0, high=0.0),
                room_choices=[os.environ.get("BOWL_SINK_ROOM", "hssd:scene2")], scene_manager="hssd", randomize_scene_pose=False,
            ),
            lighting=deepcopy(parent.dr_cfgs["lighting"]),
            material=_material_cfg(parent, ["target", "counter", "sink_counter"]),   # table + ground re-sampled per level; the target and the furniture keep a plain look
        )

        def reset(self, seed=None, options=None):
            super().reset(seed, options)
            dz = round(_table_top(self._layout, BS.COUNTER_H) - BS.COUNTER_H, 3); self._table_dz = dz    # level 3: counter height offset
            self._gates = BowlSinkGates(GateCfg(basin_len=BS.BASIN_L, basin_across=BS.BASIN_ACROSS, bowl_base_r=BS.BOWL_R_BOT, sink_h=BS.SINK_H + dz))
            self._gate_probe = None
            self._gate_steps = 0
            self._basin_offset = (L["basin_cx"] - L["sink_cx"], L["basin_cy"] - L["sink_cy"], BS.SINK_H + dz - BS.BASIN_DEPTH)
            am = AssetManager.get("bowl_sink")
            sfx = "" if abs(dz) < 1e-3 else "_" + _dz_tag(dz)
            extra = {
                "counter": ("counter" + sfx, [L["counter_cx"], L["counter_cy"], 0.0], [1.0, 0.0, 0.0, 0.0]),
                "sink_counter": ("sink_counter" + sfx, [L["sink_cx"], L["sink_cy"], 0.0], [1.0, 0.0, 0.0, 0.0]),
            }
            for name, (asset_id, pos, quat) in extra.items():
                _sync_extra(self._layout, am, name, asset_id, pos, quat)
            self._instruction = INSTRUCTION

        # ---- task gates (2026-09-23): moved-to-bowl, grasped, moved-to-basin, placed --------------------------
        # update_gates() runs every env.step through compute_reward(info, mujoco_env=...): the SIMPLE env passes every
        # actor's pose in info (name -> [xyz, wxyz]) plus the MuJoCo handle.  The basin centre is the sink counter's
        # pose plus its layout offset, so a moved sink counter is respected.  Progress lands in info["task_progress"];
        # reward = 0.25 per gate, 1.0 once placed; success (check_success) = placed.
        BASIN_OFFSET = (L["basin_cx"] - L["sink_cx"], L["basin_cy"] - L["sink_cy"], BS.SINK_H - BS.BASIN_DEPTH)

        def update_gates(self, info, mujoco_env=None):
            if mujoco_env is None or "target" not in info or "sink_counter" not in info:
                return self._gates.P
            m, d = mujoco_env.mjModel, mujoco_env.mjData
            # compute_reward() and check_success() both land here every env.step (check_success calls compute_reward):
            # advance the gate clock once per simulated instant, otherwise every hold time runs twice as fast and the
            # logged times double (found 2026-09-24 on the deployed-model pipeline test: hand_on_bowl_s 73.9 in a 50 s episode)
            key = (id(d), float(d.time))
            if getattr(self, "_gate_last_key", None) == key:
                return self._gates.P
            self._gate_last_key = key
            if self._gate_probe is None or self._gate_probe.model_id != id(m):
                self._gate_probe = ContactProbe(m, str(self.target.asset.label))
            self._gate_steps += 1
            t = self._gate_steps * self._gates.cfg.step_dt
            sink = np.asarray(info["sink_counter"], dtype=float)
            basin = sink[:3] + np.asarray(getattr(self, "_basin_offset", self.BASIN_OFFSET))
            target = np.asarray(info["target"], dtype=float)
            tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (target[4] ** 2 + target[5] ** 2), -1, 1))))
            P = self._gates.update(t, d.qpos[:2], yaw_wxyz(d.qpos[3:7]), target[:3], basin, self._gate_probe.hand_on(d),
                                   base_speed=float(np.hypot(*np.asarray(d.qvel[:2], dtype=float))), bowl_tilt_deg=tilt)
            info["task_progress"] = dict(P)
            return P

        def compute_reward(self, info, *args, **kwargs):
            self.update_gates(info, kwargs.get("mujoco_env"))
            return self._gates.reward()

        def check_success(self, info, *args, **kwargs):
            """Success = the bowl placed in the basin (gate 4): reward 1.0 >= success_criteria."""
            return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    return G1WholebodyBowlSinkTeleop


def register():
    import gymnasium as gym
    import simple.envs  # noqa: F401  stock registrations
    _patch_renderer_framebuffer()
    _patch_head_camera_pose()
    _patch_static_objects()
    task_cls = _define_task()
    return task_cls


G1WholebodyBowlSinkTeleop = register()
