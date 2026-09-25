#!/usr/bin/env python3
"""SIMPLE task of the serving-coffee-with-a-cart scene for the sonic (decoupled-WBC) teleop stack, MuJoCo.

Same layout as build_scene.py / layout.json: pelvis at the origin, x the way the robot faces, the robot's right = -y,
floor at z = 0. SIMPLE's table primitive is the serving table's top slab; its four legs are a static asset; the cart
(deck, box, handle, four casters -- one rigid free body, low friction on the wheel geoms so it rolls rather than drags)
and the coffee cup are free objects. The cup starts on the cart's box top, not on the table, so reset() puts it there.

Registered on import: task uid  g1_wholebody_coffee_cart_teleop
                      gym id    simple/G1WholebodyCoffeeCartTeleop-v0   (SonicLocoManipEnv, robot g1_sonic)
Head camera: D455 as on the box task (640 x 360, 90 deg HFOV, servo goal 2300 = 18.8 deg down), obs key head_stereo_left.
Env vars: COFFEE_CART_ROBOT_TO_CART, COFFEE_CART_ROUTE_FORWARD, COFFEE_CART_TABLE_FROM_LINE, COFFEE_CART_CUP_XY="x,y",
          COFFEE_CART_CUP_MASS, COFFEE_CART_CART_MASS, COFFEE_CART_TARGET, COFFEE_CART_INSTRUCTION.
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_scene as BS  # noqa: E402  geometry constants

TASK_UID = "g1_wholebody_coffee_cart_teleop"
ENV_ID = "simple/G1WholebodyCoffeeCartTeleop-v0"
ASSET_DIR = HERE / "assets"
_LAYOUT_JSON = json.load(open(HERE / "layout.json"))
L = dict(_LAYOUT_JSON["layout"])                # from the last build_scene.py run, pelvis at the origin

# --- knobs: the cart (and the cup on it) move with --robot-to-cart; the table moves with the route ---
ROBOT_TO_CART = float(os.environ.get("COFFEE_CART_ROBOT_TO_CART", L["robot_to_cart"]))
CART_SHIFT = (ROBOT_TO_CART + L["toe_offset"]) - L["handle_x"]
for _k in ("handle_x", "deck_near_x", "deck_far_x", "cart_cx", "box_near_x", "box_far_x", "cup_x"):
    L[_k] = L[_k] + CART_SHIFT
L["robot_to_cart"] = ROBOT_TO_CART
ROUTE_FORWARD = float(os.environ.get("COFFEE_CART_ROUTE_FORWARD", L["route_forward"]))
ROUTE_SHIFT = ROUTE_FORWARD - L["route_forward"]
for _k in ("table_near_x", "table_far_x", "table_cx"):
    L[_k] = L[_k] + ROUTE_SHIFT
L["route_forward"] = ROUTE_FORWARD
TABLE_FROM_LINE = float(os.environ.get("COFFEE_CART_TABLE_FROM_LINE", L["table_from_line"]))
_TSHIFT = -TABLE_FROM_LINE - L["table_left_y"]
for _k in ("table_left_y", "table_right_y", "table_cy"):
    L[_k] = L[_k] + _TSHIFT
L["table_from_line"] = TABLE_FROM_LINE
# --- level knobs (2026-09-24): the same level 0/1/2/3 design as SIMPLE's benchmark sets (see ../make_levels.py) ---
ROBOT_DY = float(os.environ.get("COFFEE_CART_ROBOT_DY", "0"))          # robot start moved to its left (+y) by this: everything else moves -y (level 3)
for _k, _v in list(L.items()):
    if _k.endswith("y") and not _k.startswith("plan_") and isinstance(_v, (int, float)) and not isinstance(_v, bool):
        L[_k] = _v - ROBOT_DY
TARGET_JITTER = tuple(float(v) for v in os.environ.get("COFFEE_CART_TARGET_JITTER", "0,0").split(","))   # half-widths (x, y) of the cup's start region (level 2/3)
# A randomised cup must stay on the box top: the measured spot is at the box's near-right corner (0.5 cm inside), so the
# region is clipped to the top minus the cup radius and a 1.5 cm margin. COFFEE_CART_TARGET_REGION="x0,x1,y0,y1" sets it outright.
_SAFE = (L["box_near_x"] + BS.CUP_R_TOP + 0.015, L["box_far_x"] - BS.CUP_R_TOP - 0.015, L["box_right_y"] + BS.CUP_R_TOP + 0.015, L["box_left_y"] - BS.CUP_R_TOP - 0.015)
if os.environ.get("COFFEE_CART_TARGET_REGION"):
    _x0, _x1, _y0, _y1 = (float(v) for v in os.environ["COFFEE_CART_TARGET_REGION"].split(","))
else:
    _x0, _x1, _y0, _y1 = L["cup_x"] - TARGET_JITTER[0], L["cup_x"] + TARGET_JITTER[0], L["cup_y"] - TARGET_JITTER[1], L["cup_y"] + TARGET_JITTER[1]
if TARGET_JITTER != (0.0, 0.0) or os.environ.get("COFFEE_CART_TARGET_REGION"):
    _x0, _x1, _y0, _y1 = max(_x0, _SAFE[0]), min(_x1, _SAFE[1]), max(_y0, _SAFE[2]), min(_y1, _SAFE[3])
    if _x0 > _x1: _x0 = _x1 = min(max(L["cup_x"], _SAFE[0]), _SAFE[1])
    if _y0 > _y1: _y0 = _y1 = min(max(L["cup_y"], _SAFE[2]), _SAFE[3])
TARGET_LO, TARGET_HI = (_x0, _y0), (_x1, _y1)
NUM_DISTRACTORS = int(os.environ.get("COFFEE_CART_NUM_DISTRACTORS", "0"))                                   # GraspNet distractors on the table (SIMPLE uses 3)
if os.environ.get("COFFEE_CART_CUP_XY"):
    L["cup_x"], L["cup_y"] = (float(v) for v in os.environ["COFFEE_CART_CUP_XY"].split(","))

TARGET = os.environ.get("COFFEE_CART_TARGET", "coffee_cart:coffee_cup")
CUP_MASS = float(os.environ.get("COFFEE_CART_CUP_MASS", _LAYOUT_JSON["cup"]["mass"]))
CART_MASS = float(os.environ.get("COFFEE_CART_CART_MASS", _LAYOUT_JSON["cart"]["mass"]))
WHEEL_FRICTION = float(os.environ.get("COFFEE_CART_WHEEL_FRICTION", BS.WHEEL_FRICTION))
# handle geometry lives in layout.json (build_scene.py may have been run with --handle-top / --handle-overhang)
BS.HANDLE_TOP = float(os.environ.get("COFFEE_CART_HANDLE_TOP", _LAYOUT_JSON["cart"]["handle_top"]))
BS.HANDLE_OVERHANG = float(os.environ.get("COFFEE_CART_HANDLE_OVERHANG", _LAYOUT_JSON["cart"].get("handle_overhang", 0.30)))
# --- success checkers (SIMPLE-style: compute_reward / check_success on the env's info dict) ---
CHK_CART_PUSHED_M = float(os.environ.get("COFFEE_CART_CHK_CART_PUSHED_M", "0.5"))   # cart moved this far from its start
CHK_CUP_LIFT_M = float(os.environ.get("COFFEE_CART_CHK_CUP_LIFT_M", "0.05"))          # cup this high above its rest on the box
CHK_TABLE_Z_TOL = float(os.environ.get("COFFEE_CART_CHK_TABLE_Z_TOL", "0.035"))      # cup base within this of the table top (0.74 vs the 0.80 box)
CHK_UPRIGHT_DEG = float(os.environ.get("COFFEE_CART_CHK_UPRIGHT_DEG", "45"))         # tilt of the cup axis from vertical
CHK_HOLD_STEPS = int(os.environ.get("COFFEE_CART_CHK_HOLD_STEPS", "25"))             # resting + released this many env steps (0.5 s at 50 Hz)
CHK_REQUIRE_RELEASE = os.environ.get("COFFEE_CART_CHK_REQUIRE_RELEASE", "1") == "1"    # 0: a cup resting upright on the table counts even with the hand still on it
INSTRUCTION = os.environ.get("COFFEE_CART_INSTRUCTION",
                             "push the cart to the table, then pick up the coffee cup and place it on the table")

# D455 lens pose of the scene (servo pan 0 / tilt 10 deg) as an offset of SIMPLE's stock eye_in_head mount, from
# sim/tabletop_box/g1_tabletop_box_simple_task.py (the engine's mount is the same for every G1 robot class)
CAM_LOCAL_POS = [0.01849, 0.03219, -0.00928]
CAM_LOCAL_QUAT = [0.996029, -0.088426, -0.007383, -0.007305]
ROOM_DX = float(os.environ.get("COFFEE_CART_ROOM_DX", "1.0"))    # Isaac room pushed this far ahead of its stock placement (the room's own counter sits at
                                                                  # the robot's start). hssd:scene31 (2026-09-24): the wall ahead then sits ~5.1 m out, 1.3 m past
                                                                  # the cart's nose at the end of the push, and the side the robot turns to is open to ~4.4 m -
                                                                  # in scene2 a counter stood 1.65 m to the right at the stop, straight in the head camera.
                                                                  # inside the far wall (black head-camera frames); 0.7 keeps the third-person camera inside
HEAD_TILT_TRIM_DEG = float(os.environ.get("COFFEE_CART_HEAD_TRIM_DEG", "8.8"))   # servo goal 2300: +8.8 deg (100 ticks) on the encoded 2200 baseline, axis 66.5 deg below horizontal (2026-09-24)
HEAD_HFOV_DEG, HEAD_W, HEAD_H = 90.0, 640, 360
HAND_RGBA = [float(v) for v in BS.HAND_RGBA.split()]

GRIPPY = [0.8, 0.05, 0.005]                              # SIMPLE's default object friction
SLIPPY = [WHEEL_FRICTION, 0.002, 0.0001]                 # the casters: one rigid body, so low friction stands in for rolling


# ------------------------------------------------------------------ generated meshes (OBJ for MuJoCo, USDA for Isaac)
def _box(size, center=(0.0, 0.0, 0.0), yaw=0.0):
    sx, sy, sz = (s / 2 for s in size)
    v = np.array([(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float)
    if yaw:
        v = v @ R.from_euler("z", yaw).as_matrix().T
    v = v + np.asarray(center, dtype=float)
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]     # outward quads
    return v, f


def _cyl(r, half_len, center, axis="z", n=16):
    """Convex prism standing in for a cylinder, its length along `axis`."""
    circ = [(r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n)) for k in range(n)]
    v = []
    for s in (-half_len, half_len):
        for a, b in circ:
            v.append([a, b, s] if axis == "z" else ([a, s, b] if axis == "y" else [s, a, b]))
    v = np.array(v, dtype=float) + np.asarray(center, dtype=float)
    f = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for k in range(n):
        k1 = (k + 1) % n
        f.append((k, k1, n + k1, n + k))
    return v, f


def _write_obj(path: Path, pieces):
    lines, off = ["# generated by coffee_cart_task.py"], 0
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
    """Box rotated by tilt about its own y axis, then yaw about z, then translated (the cup wall pieces)."""
    v, f = _box(size)
    v = v @ (R.from_euler("z", yaw) * R.from_euler("y", tilt_y)).as_matrix().T + np.asarray(center)
    return v, f


def cup_pieces(nseg=12):
    """Same pieces as build_scene.cup_collision_geoms: a floor disc (as a 24-gon prism) + tilted wall boxes."""
    pieces = []
    r, t = BS.CUP_R_BOT, BS.CUP_T
    pieces.append(_cyl(r, t / 2, (0.0, 0.0, t / 2), "z", n=24))
    z0 = 0.006
    r0 = BS.CUP_R_BOT + (BS.CUP_R_TOP - BS.CUP_R_BOT) * z0 / BS.CUP_H
    slant = np.arctan2(BS.CUP_R_TOP - BS.CUP_R_BOT, BS.CUP_H)
    wall_len = np.hypot(BS.CUP_R_TOP - r0, BS.CUP_H - z0)
    r_mid = (BS.CUP_R_TOP + r0) / 2 - t / 2
    half_w = r_mid * np.sin(np.pi / nseg) * 1.05
    for k in range(nseg):
        a = 2 * np.pi * (k + 0.5) / nseg
        pieces.append(_tilted_box((t, 2 * half_w, wall_len), (r_mid * np.cos(a), r_mid * np.sin(a), (z0 + BS.CUP_H) / 2), a, slant))
    return pieces


def cart_pieces(box_dz: float = 0.0):
    """Cart in its own frame (origin on the floor under the deck centre, +x toward the table, handle at -x).
    Returns (pieces, masses, frictions) -- the wheels get the low friction that stands in for rolling casters.
    box_dz: level-3 change of the box height (the cup's support)."""
    hl, hw = BS.CART_L / 2, BS.CART_W / 2
    BOX_H = BS.BOX_H + box_dz
    box_cx = -hl + BS.BOX_FROM_HANDLE_END + BS.BOX_L / 2
    r = BS.HANDLE_DIA / 2
    post_h = BS.HANDLE_TOP - BS.DECK_TOP
    wr = BS.WHEEL_DIA / 2
    bar_half = hw - r - 0.01
    pieces = [_box((BS.CART_L, BS.CART_W, BS.DECK_T), (0, 0, BS.DECK_TOP - BS.DECK_T / 2)),
              _box((BS.BOX_L, BS.BOX_W, BOX_H), (box_cx, 0, BS.DECK_TOP + BOX_H / 2))]
    fr = [GRIPPY, GRIPPY]
    frac = [0.22, 0.46]
    roles = ["deck", "box"]
    ov = BS.HANDLE_OVERHANG
    for y in (bar_half, -bar_half):
        pieces.append(_cyl(r, post_h / 2, (-hl + r, y, BS.DECK_TOP + post_h / 2), "z"))
        fr.append(GRIPPY); frac.append(0.03); roles.append("post")
        if ov > 1e-3:                                                                   # strut back to the grip bar, if any
            pieces.append(_cyl(r, ov / 2, (-hl + r - ov / 2, y, BS.HANDLE_TOP), "x"))
            fr.append(GRIPPY); frac.append(0.02); roles.append("strut")
    pieces.append(_cyl(r, bar_half, (-hl + r - ov, 0.0, BS.HANDLE_TOP), "y"))          # the grip bar itself
    fr.append(GRIPPY); frac.append(0.04); roles.append("bar")
    tot = sum(frac) + 4 * 0.045; frac = [f / tot for f in frac]; wheel_frac = 0.045 / tot   # renormalise
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        pieces.append(_cyl(wr, 0.018, (sx * (hl - 0.09), sy * (hw - 0.05), wr), "y"))
        fr.append(SLIPPY); frac.append(wheel_frac); roles.append("wheel")
    return pieces, [CART_MASS * f for f in frac], fr, roles


def cart_piece_roles():
    return cart_pieces()[3]


def table_leg_pieces(dz: float = 0.0):
    """The four legs in the table frame (SIMPLE's table primitive is the top slab above them)."""
    leg_h = BS.TABLE_H + dz - BS.TABLE_TOP_T
    dx = BS.TABLE_D / 2 - BS.TABLE_LEG / 2 - 0.02
    dy = BS.TABLE_W / 2 - BS.TABLE_LEG / 2 - 0.02
    return [_box((BS.TABLE_LEG, BS.TABLE_LEG, leg_h), (sx * dx, sy * dy, leg_h / 2))
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))]


def ensure_assets() -> dict:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    cup = ASSET_DIR / "coffee_cup_9cm.obj"
    if not cup.exists():
        BS.cup_mesh(ASSET_DIR)
    objs = []
    for i, pc in enumerate(cup_pieces()):
        q = ASSET_DIR / f"cup_piece{i:02d}.obj"; q.exists() or _write_obj(q, [pc]); objs.append(q)
    u = ASSET_DIR / "coffee_cup_9cm.usda"; u.exists() or _write_usda(u, "coffee_cup_9cm", [_read_obj(cup)], (0.95, 0.95, 0.94))
    files["cup"] = dict(obj=cup, mujoco=objs, usd=u)
    pcs, masses, frictions, roles = cart_pieces()
    objs = []
    for i, pc in enumerate(pcs):
        q = ASSET_DIR / f"cart_piece{i:02d}.obj"; _write_obj(q, [pc]); objs.append(q)     # always rewritten: depends on the knobs
    p = ASSET_DIR / "cart.obj"; _write_obj(p, pcs)
    u = ASSET_DIR / "cart.usda"; _write_usda(u, "cart", pcs, (0.62, 0.64, 0.67))
    # MuJoCo combines two geoms' friction with MAX unless one has higher priority: the floor is 1.0, so the wheels
    # need priority 1 for their rolling-resistance friction to be the one that counts
    files["cart"] = dict(obj=p, mujoco=objs, usd=u, masses=masses, frictions=frictions,
                         priorities=[1 if r == "wheel" else 0 for r in roles], roles=roles)
    lp = table_leg_pieces(); objs = []
    for i, pc in enumerate(lp):
        q = ASSET_DIR / f"table_leg{i}.obj"; q.exists() or _write_obj(q, [pc]); objs.append(q)
    p = ASSET_DIR / "table_legs.obj"; p.exists() or _write_obj(p, lp)
    u = ASSET_DIR / "table_legs.usda"; u.exists() or _write_usda(u, "table_legs", lp, (0.80, 0.72, 0.60))
    files["table_legs"] = dict(obj=p, mujoco=objs, usd=u)
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
def _patch_scene_objects():
    """Honour asset.static (welded, no free joint), .rgba, .mass and -- new here -- .piece_masses / .piece_frictions,
    so the cart can carry slippery wheel geoms and a grippy deck in one rigid body."""
    import mujoco
    from simple.engines.mujoco import MujocoSimulator
    if getattr(MujocoSimulator, "_coffee_cart_patched", False):
        return
    orig = MujocoSimulator._build_object

    def build_object(self, mjSpec, mjWorld, actor):
        asset = actor.asset
        static = getattr(asset, "static", False)
        rgba = getattr(asset, "rgba", None)
        mass = getattr(asset, "mass", None) or 0.1
        piece_masses = getattr(asset, "piece_masses", None)
        piece_frictions = getattr(asset, "piece_frictions", None)
        piece_priorities = getattr(asset, "piece_priorities", None)
        if not static and rgba is None and mass == 0.1 and piece_masses is None and piece_frictions is None:
            return orig(self, mjSpec, mjWorld, actor)
        label = getattr(asset, "label", asset.uid)
        meshes = asset.collision_meshes_mujoco
        for i, f in enumerate(meshes):
            mjSpec.add_mesh(name=f"{label}_mesh_convex{i}", file=str(f))
        body = mjWorld.add_body(name=label, pos=actor.pose.position, quat=actor.pose.quaternion)
        for i in range(len(meshes)):
            g = body.add_geom(name=f"{label}_convex_{i}", meshname=f"{label}_mesh_convex{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                              condim=4, mass=(piece_masses[i] if piece_masses else mass / len(meshes)),
                              friction=(piece_frictions[i] if piece_frictions else [0.8, 0.05, 0.005]),
                              rgba=list(rgba or [1, 1, 1, 1]), solref=[0.005, 2])
            if piece_priorities and piece_priorities[i]:
                g.priority = int(piece_priorities[i])
        if not static:
            body.add_freejoint(name=f"{label}_joint")

    MujocoSimulator._build_object = build_object
    MujocoSimulator._coffee_cart_patched = True


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


def cart_variant_files(dz: float) -> dict:
    tag = "_" + _dz_tag(dz); pcs, masses, frictions, roles = cart_pieces(dz); objs = []
    for i, pc in enumerate(pcs):
        q = ASSET_DIR / f"cart_piece{i:02d}{tag}.obj"; _write_obj(q, [pc]); objs.append(q)
    pth = ASSET_DIR / f"cart{tag}.obj"; _write_obj(pth, pcs)
    u = ASSET_DIR / f"cart{tag}.usda"; _write_usda(u, f"cart{tag}", pcs, (0.62, 0.64, 0.67))
    return dict(obj=pth, mujoco=objs, usd=u, masses=masses, frictions=frictions, priorities=[1 if r == "wheel" else 0 for r in roles], roles=roles)


def legs_variant_files(dz: float) -> dict:
    tag = "_" + _dz_tag(dz); lp = table_leg_pieces(dz); objs = []
    for i, pc in enumerate(lp):
        q = ASSET_DIR / f"table_leg{i}{tag}.obj"; q.exists() or _write_obj(q, [pc]); objs.append(q)
    pth = ASSET_DIR / f"table_legs{tag}.obj"; pth.exists() or _write_obj(pth, lp)
    u = ASSET_DIR / f"table_legs{tag}.usda"; u.exists() or _write_usda(u, f"table_legs{tag}", lp, (0.80, 0.72, 0.60))
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
        def __init__(self, uid, name, usd_path, curobo_mesh, mujoco_meshes, stable_poses, static=False, rgba=None,
                     mass=None, piece_masses=None, piece_frictions=None, piece_priorities=None):
            super().__init__(uid=uid, usd_path=str(usd_path), collision_mesh_curobo=str(curobo_mesh),
                             collision_meshes_mujoco=[str(m) for m in mujoco_meshes])
            self.name = name; self.label = uid; self.description = name
            self.stable_poses = stable_poses
            self.keypoints, self.axes, self.canonical_grasps, self.functional_grasps = {}, {}, [], {}
            self.static, self.rgba, self.mass = static, rgba, mass
            self.piece_masses, self.piece_frictions, self.piece_priorities = piece_masses, piece_frictions, piece_priorities

        def to_dict(self):
            return {"res_id": "coffee_cart", "uid": self.uid, "label": self.label, "name": self.name,
                    "usd_path": self.usd_path, "description": self.description}

        def __repr__(self):
            return f"SceneAsset({self.uid})"

    if "coffee_cart" not in AssetManager._registry:
        @AssetManager.register("coffee_cart")
        class CoffeeCartAssetManager(AssetManager):
            def __init__(self):
                self.files = ensure_assets()

            def load(self, asset_id: str, *args, **kwargs):
                f = self.files
                if asset_id == "coffee_cup":
                    return SceneAsset("coffee_cup", "white coffee cup", f["cup"]["usd"], f["cup"]["obj"], f["cup"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=[0.95, 0.95, 0.94, 1.0], mass=CUP_MASS)
                if asset_id == "cart":
                    return SceneAsset("cart", "service cart with a box", f["cart"]["usd"], f["cart"]["obj"], f["cart"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=[0.68, 0.60, 0.48, 1.0],
                                      mass=CART_MASS, piece_masses=f["cart"]["masses"], piece_frictions=f["cart"]["frictions"],
                                      piece_priorities=f["cart"]["priorities"])
                if asset_id == "table_legs":
                    return SceneAsset("table_legs", "table legs", f["table_legs"]["usd"], f["table_legs"]["obj"], f["table_legs"]["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.80, 0.72, 0.60, 1.0])
                if asset_id.startswith("cart_"):
                    v = cart_variant_files(_dz_from_id(asset_id, "cart"))
                    return SceneAsset(asset_id, "service cart with a box", v["usd"], v["obj"], v["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=[0.68, 0.60, 0.48, 1.0],
                                      mass=CART_MASS, piece_masses=v["masses"], piece_frictions=v["frictions"], piece_priorities=v["priorities"])
                if asset_id.startswith("table_legs_"):
                    v = legs_variant_files(_dz_from_id(asset_id, "table_legs"))
                    return SceneAsset(asset_id, "table legs", v["usd"], v["obj"], v["mujoco"],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], static=True, rgba=[0.80, 0.72, 0.60, 1.0])
                raise ValueError(f"unknown coffee_cart asset {asset_id}")

            def sample(self, exclude=None):
                return self.load("coffee_cup")

            def __len__(self):
                return 3

            def __iter__(self):
                yield self.load("coffee_cup")

    cam_quat = t3d.quaternions.qmult(CAM_LOCAL_QUAT, t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], np.radians(-HEAD_TILT_TRIM_DEG))).tolist()
    stable_idx = 0 if TARGET.startswith("coffee_cart:") else standing_stable_index(TARGET)
    parent = G1WholebodyXMovePickTaskTeleop

    @TaskRegistry.register(TASK_UID)
    class G1WholebodyCoffeeCartTeleop(parent):
        uid: str = TASK_UID
        label: str = "G1 Coffee Cart Teleop"
        isaac_cameras_follow_mujoco = True          # the Isaac head camera copies the MuJoCo (patched, measured D455) camera pose every step
        isaac_hidden_scene_groups = ("furniture",)  # the HSSD room's own furniture is hidden in Isaac: a 1 m high island of hssd:scene31 spans the cart route and a cabinet row sits where the desk stands (user: nothing between cart and desk, nothing blocking the cart at the end)
        description: str = ("G1 standing behind a 0.71 x 0.47 m service cart whose box top carries a coffee cup; it pushes the cart "
                            f"{L['route_forward']:.2f} m forward and {L['table_from_line']:.2f} m to its right to a "
                            f"{BS.TABLE_W:.2f} x {BS.TABLE_D:.2f} x {BS.TABLE_H:.2f} m table, and puts the cup on it.")
        metadata = dict(parent.metadata)
        sensor_cfgs = dict(
            head_stereo=StereoCameraCfg(uid="Realsense_D455", mount="eye_in_head", width=HEAD_W, height=HEAD_H,
                                        focal_length=1.93, fov=np.deg2rad(HEAD_HFOV_DEG), near=0.05, far=8, baseline=0.05,
                                        pose=dict(position=list(CAM_LOCAL_POS), quaternion=cam_quat)))
        dr_cfgs = dict(
            language=LanguageDRCfg(instructions=[INSTRUCTION]),
            target=TargetDRCfg(asset_id=TARGET),
            distractors=DistractorDRCfg(res_id="graspnet1b", number_of_distractors=NUM_DISTRACTORS, allow_duplicates=False, exclude=[]),
            spatial=SpatialDRCfg(
                spatial_mode="random",                                       # zero-width boxes = fixed placement
                robot_region=Box(low=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0]),
                target_region=Box(low=[TARGET_LO[0], TARGET_LO[1]], high=[TARGET_HI[0], TARGET_HI[1]]),
                distractors_region=Box(low=[L["table_near_x"] + 0.1, L["table_right_y"] + 0.2],
                                       high=[L["table_far_x"] - 0.1, L["table_left_y"] - 0.2]),
                target_stable_indices=[stable_idx], target_rotate_z=Box(low=0.0, high=0.0),
            ),
            camera=CameraDRCfg(cam_id="head_stereo"),
            scene=TabletopSceneDRCfg(
                scene_mode="fixed",
                table_size=Box(low=[BS.TABLE_D, BS.TABLE_W, BS.TABLE_TOP_T], high=[BS.TABLE_D, BS.TABLE_W, BS.TABLE_TOP_T]),
                table_position=Box(low=[L["table_cx"], L["table_cy"]], high=[L["table_cx"], L["table_cy"]]),
                table_height=Box(low=BS.TABLE_H, high=BS.TABLE_H),
                rotation_z=Box(low=0.0, high=0.0),
                room_choices=[os.environ.get("COFFEE_CART_ROOM", "hssd:scene31")], scene_manager="hssd", randomize_scene_pose=False,
            ),
            lighting=deepcopy(parent.dr_cfgs["lighting"]),
            material=_material_cfg(parent, ["target", "cart", "table_legs"]),   # table + ground re-sampled per level; the target and the furniture keep a plain look
        )

        def reset(self, seed=None, options=None):
            super().reset(seed, options)
            am = AssetManager.get("coffee_cart")
            dz = round(_table_top(self._layout, BS.TABLE_H) - BS.TABLE_H, 3); self._table_dz = dz     # level 3: the delivery table AND the cart box change by dz
            self._table_h = BS.TABLE_H + dz; self._box_top = L["box_top"] + dz
            sfx = "" if abs(dz) < 1e-3 else "_" + _dz_tag(dz)
            extra = {
                "cart": ("cart" + sfx, [L["cart_cx"], L["cart_cy"], 0.0], [1.0, 0.0, 0.0, 0.0]),
                "table_legs": ("table_legs" + sfx, [L["table_cx"], L["table_cy"], 0.0], [1.0, 0.0, 0.0, 0.0]),
            }
            for name, (asset_id, pos, quat) in extra.items():
                _sync_extra(self._layout, am, name, asset_id, pos, quat)
            # the cup starts on the cart's box top, not on the table SIMPLE placed it against
            if "target" in self._layout.actors:
                _p = self._layout.actors["target"].pose.position          # keep the x, y SIMPLE sampled in the (jittered) target region
                _ok = TARGET_LO[0] - 1e-6 <= _p[0] <= TARGET_HI[0] + 1e-6 and TARGET_LO[1] - 1e-6 <= _p[1] <= TARGET_HI[1] + 1e-6
                _x, _y = (float(_p[0]), float(_p[1])) if _ok else (L["cup_x"], L["cup_y"])
                if abs(_x - L["cup_x"]) < 1e-6 and abs(_y - L["cup_y"]) < 1e-6:
                    _x, _y = L["cup_x"], L["cup_y"]                     # no jitter: the EXACT nominal. The sampler returns float32, and its
                                                                          # 4e-9 rounding alone flipped a marginal replay (ep 84, 2026-09-24)
                self._layout.actors["target"].pose.position = [_x, _y, self._box_top]
            self._instruction = INSTRUCTION
            sc = self.dr.get_randomizer("scene")._inner_state          # shift the (visual-only) Isaac room, idempotently: absolute = room middle + ROOM_DX
            if sc is not None and hasattr(sc, "center_offset") and isinstance(getattr(sc, "conf", None), dict):
                mid = [(u + d) / 2.0 for u, d in zip(sc.conf["center_offset_limit_up"], sc.conf["center_offset_limit_down"])]
                sc.center_offset = [mid[0] + ROOM_DX, mid[1], mid[2]]
            self._chk = dict(armed=False, cup_z0=None, cart_xy0=None, cart_pushed_ever=False, cup_lifted_ever=False,
                             max_lift_m=0.0, cart_travel_m=0.0, on_table_steps=0, success=False, geoms=None)

        # ------------------------------------------------------------------ checkers
        def _contact_geoms(self, m):
            """Cup and hand geom ids, cached per compiled model."""
            import mujoco
            c = self._chk
            if c["geoms"] is not None and c["geoms"][0] is id(m):
                return c["geoms"][1], c["geoms"][2]
            tgt = self.layout.actors["target"].asset.label
            cup = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") == tgt}
            hand = {g for g in range(m.ngeom) if "_hand_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")}
            c["geoms"] = (id(m), cup, hand)
            return cup, hand

        def progress(self, info, mujoco_env=None) -> dict:
            """Sub-goal checkers on the env's info dict (actor poses as [xyz, wxyz]); sticky flags survive later steps."""
            c = self._chk
            cup = np.asarray(info["target"][:3], float); q = np.asarray(info["target"][3:7], float)
            cart = np.asarray(info["cart"][:3], float)
            if not c["armed"]:
                c["cup_z0"], c["cart_xy0"], c["armed"] = float(cup[2]), cart[:2].copy(), True
            zaxis = R.from_quat([q[1], q[2], q[3], q[0]]).apply([0.0, 0.0, 1.0])
            tilt_deg = float(np.degrees(np.arccos(np.clip(zaxis[2], -1.0, 1.0))))
            lift = float(cup[2] - c["cup_z0"]); c["max_lift_m"] = max(c["max_lift_m"], lift)
            travel = float(np.linalg.norm(cart[:2] - c["cart_xy0"])); c["cart_travel_m"] = max(c["cart_travel_m"], travel)
            in_footprint = bool(L["table_near_x"] - 0.02 <= cup[0] <= L["table_far_x"] + 0.02
                                and L["table_right_y"] - 0.02 <= cup[1] <= L["table_left_y"] + 0.02)
            upright = tilt_deg < CHK_UPRIGHT_DEG
            resting_on_table = bool(in_footprint and upright and abs(float(cup[2]) - getattr(self, "_table_h", BS.TABLE_H)) < CHK_TABLE_Z_TOL)
            in_hand = False
            if mujoco_env is not None and hasattr(mujoco_env, "mjModel"):
                m, d = mujoco_env.mjModel, mujoco_env.mjData
                cup_g, hand_g = self._contact_geoms(m)
                in_hand = any((ct.geom1 in cup_g and ct.geom2 in hand_g) or (ct.geom2 in cup_g and ct.geom1 in hand_g)
                              for ct in d.contact[:d.ncon])
            c["cart_pushed_ever"] |= travel >= CHK_CART_PUSHED_M
            c["cup_lifted_ever"] |= lift >= CHK_CUP_LIFT_M
            c["on_table_steps"] = c["on_table_steps"] + 1 if (resting_on_table and (not in_hand or not CHK_REQUIRE_RELEASE)) else 0
            if c["on_table_steps"] >= CHK_HOLD_STEPS:
                c["success"] = True                                       # latched: placed and let go
            # plain Python types only: this dict goes straight into json.dump downstream
            return dict(
                cart_travel_m=round(travel, 3), cart_pushed=bool(travel >= CHK_CART_PUSHED_M), cart_pushed_ever=bool(c["cart_pushed_ever"]),
                cup_lift_m=round(lift, 3), cup_lifted=bool(lift >= CHK_CUP_LIFT_M), cup_lifted_ever=bool(c["cup_lifted_ever"]),
                max_lift_m=round(float(c["max_lift_m"]), 3), cup_tilt_deg=round(tilt_deg, 1), cup_upright=bool(upright),
                cup_in_table_footprint=bool(in_footprint), cup_resting_on_table=resting_on_table, cup_in_hand=bool(in_hand),
                cup_tipped_on_table=bool(in_footprint and not upright and abs(float(cup[2]) - (getattr(self, "_table_h", BS.TABLE_H) + BS.CUP_R_TOP)) < 0.03),
                cup_on_floor=bool(float(cup[2]) < 0.2),
                cup_still_on_cart=bool(abs(float(cup[2]) - getattr(self, "_box_top", L["box_top"])) < 0.03 and float(np.linalg.norm(cup[:2] - cart[:2])) < 0.5),
                on_table_steps=int(c["on_table_steps"]), success=bool(c["success"]))

        def compute_reward(self, info, *args, mujoco_env=None, **kwargs) -> float:
            """0.25 cart pushed + 0.25 cup lifted + 0.5 placed; 1.0 exactly when the cup has been placed and released."""
            pr = self.progress(info, mujoco_env)
            info["progress"] = pr; info["success"] = pr["success"]
            return 0.25 * pr["cart_pushed_ever"] + 0.25 * pr["cup_lifted_ever"] + 0.5 * pr["success"]

        def check_success(self, info, *args, **kwargs) -> bool:
            # the env calls compute_reward() first every step; evaluating progress() a second time here made the
            # on_table_steps hold counter run twice per step (CHK_HOLD_STEPS reached in half the time; found 2026-09-24)
            if "progress" not in info:
                self.compute_reward(info, *args, **kwargs)
            pr = info["progress"]
            reward = 0.25 * pr["cart_pushed_ever"] + 0.25 * pr["cup_lifted_ever"] + 0.5 * pr["success"]
            return bool(pr["success"]) and reward >= min(self.success_criteria, 1.0)

    return G1WholebodyCoffeeCartTeleop


def register():
    import gymnasium as gym
    import simple.envs  # noqa: F401  stock registrations
    _patch_renderer_framebuffer()
    _patch_head_camera_pose()
    _patch_scene_objects()
    task_cls = _define_task()
    if ENV_ID not in gym.envs.registry:
        gym.register(id=ENV_ID, entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv", kwargs={"task": TASK_UID})
    return task_cls


if __name__ != "__main__":
    try:
        register()
    except Exception as e:  # noqa: BLE001
        print(f"[coffee_cart] task not registered on import: {e}", file=sys.stderr)
