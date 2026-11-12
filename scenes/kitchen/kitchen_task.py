"""HoloMotion v1.4 sim teleop kit `kitchen` (2026-10-26): the manipulable kitchen with the G1 (Dex3 or BrainCo Revo 2 hands).

Task: pick the 500 ml bottle standing on the peninsula worktop in front of the robot, turn right, walk to the sink and
put the bottle in the sink basin.  Gates, latched in order (same scheme as bottle_bin_task): grasped, at_sink, placed.

How the kitchen gets into SIMPLE: the whole authored MJCF (/home/Horizon/wrk/KitchenScene_Manipulable_20261026/scene.xml,
16 mechanism joints, 8 free props) is attached to the compiled SIMPLE scene with MjSpec.attach, rotated 180 deg and placed
so that the robot (SIMPLE's origin, facing +x) stands in the aisle KITCHEN_ROBOT_TO_EDGE in front of the worktop edge,
lifted 2.5 cm so the kitchen floor meets SIMPLE's ground.  SIMPLE's own table primitive and HSSD room objects are not built;
the target bottle (the bottle_bin kit's mesh) and GraspNet distractors are placed by SIMPLE's spatial randomizer on the
worktop (table_height = the worktop top).

Env knobs (all optional): KITCHEN_ROBOT_TO_EDGE (m, default 0.35), KITCHEN_ROBOT_Y (kitchen y of the start, default 1.4),
KITCHEN_TARGET_JITTER "x,y" half-widths (only for SIMPLE's bottle), KITCHEN_NUM_DISTRACTORS (default 0: the kitchen has its own props),
KITCHEN_INSTRUCTION, KITCHEN_XML (another kitchen file).
Run:  scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2   (or --robot stereo for the Dex3 G1)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import mujoco

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_scene as BS  # noqa: E402

TASK_UID = "g1_wholebody_kitchen_sink_teleop"
ENV_ID = "simple/G1WholebodyKitchenSinkTeleop-v0"
BOTTLE_BIN_ASSETS = HERE.parent / "bottle_bin" / "assets"          # the 500 ml bottle meshes of the bottle_bin kit
_LAYOUT_JSON = json.load(open(HERE / "layout.json"))
L = dict(_LAYOUT_JSON["layout"])
ROBOT_TO_EDGE = float(os.environ.get("KITCHEN_ROBOT_TO_EDGE", L["pelvis_to_edge"] - L["toe_offset"]))
ROBOT_Y_K = float(os.environ.get("KITCHEN_ROBOT_Y", L["robot_y_k"]))
TARGET_JITTER = tuple(float(v) for v in os.environ.get("KITCHEN_TARGET_JITTER", "0,0").split(","))
NUM_DISTRACTORS = int(os.environ.get("KITCHEN_NUM_DISTRACTORS", "0"))       # none: the kitchen's props are the clutter
TARGET = os.environ.get("KITCHEN_TARGET", "kitchen:bottle_red")           # the kitchen's own red bottle; kitchen:bottle_500ml = SIMPLE's bottle as before
# tasks (2026-10-26, user: "add a small ball in each of the bowls, i want to grasp the bowl, move to the trash bin and throw the
# small ball inside it into the trash bin" ... "just as an option, the default should be the previous empty one"):
# KITCHEN_TASK picks the gates and the instruction. The balls are an option: on with KITCHEN_TASK=ball_bin or
# KITCHEN_BALLS=1, off by default (empty bowls, as before). The pedal bin's lid is closed (KITCHEN_BIN_LID=open starts it open).
# load_dishwasher (2026-10-27, user: "correct the task as loading dishwasher") is the default: what the kitchen sessions
# record; bottle_sink and ball_bin stay options. The task only sets the instruction and the progress gates (no physics),
# so it does not change a recording's replay.
TASK = os.environ.get("KITCHEN_TASK", "load_dishwasher")
INSTRUCTIONS = {"load_dishwasher": "open the dishwasher, load the plates and bowls from the counter into its racks, push the rack back in and close the door",
                "bottle_sink": "pick up the red bottle from the counter, walk to the sink, and put it in the sink",
                "ball_bin": "pick up a bowl with a ball in it, walk to the pedal bin, and tip the ball into the bin"}
DISHES = ("plate_a", "plate_b", "bowl_small", "bowl_large")             # the kitchen's dishes (k_ bodies after attach)
RACKS = ("dishwasher_lower_rack", "dishwasher_upper_rack")
RACK_BOX = dict(x=0.25, y=0.24, z=(-0.03, 0.30))                       # a dish centre inside a rack, in the rack body's frame (m)
DOOR_OPEN_RAD = 1.2                                                     # ~69 deg: open far enough to pull a rack out
RACK_IN_M, DOOR_CLOSED_RAD, CLOSED_HOLD_S = 0.05, float(np.radians(5.0)), 1.0   # 2026-10-30 (user: "after plate placed it should not stop, i need to pull
                                                                        # back the rack and close the dishwasher"): the task ends with every rack within 5 cm
                                                                        # of home and the door within 5 deg of shut, a dish still loaded, held 1 s
if TASK not in INSTRUCTIONS:
    raise ValueError(f"KITCHEN_TASK={TASK!r}: one of {sorted(INSTRUCTIONS)}")
INSTRUCTION = os.environ.get("KITCHEN_INSTRUCTION", INSTRUCTIONS[TASK])
# scene revision (2026-10-26): 1 = the kitchen as recorded before this evening (bowls authored 4 mm above the worktop, 18 mm
# bin floor, the cling-wrap carton leaning on the back step and jittering); 2 = bowls at their resting height, 10 cm bin
# floor (same top face), carton 3 cm toward the aisle. The knobs below are put into os.environ so a recording's env_knobs
# carry them and its replay rebuilds the same scene; replay_holomotion_v14 rebuilds a recording without KITCHEN_REV as rev 1.
# Rev 3 (2026-10-27) adds the dishwasher door fixes of _dishwasher_rev3 (solid front, handle where it is drawn, counterbalanced hinge).
REV = int(os.environ.get("KITCHEN_REV", "3"))             # 3 (2026-10-27): the dishwasher door is solid and stays where it is left (_dishwasher_rev3)
BALLS_ON = os.environ.get("KITCHEN_BALLS", "1" if TASK == "ball_bin" else "0") != "0"
if TASK == "ball_bin" and not BALLS_ON:
    print("[kitchen] KITCHEN_TASK=ball_bin with KITCHEN_BALLS=0: the bowls are empty, the task cannot succeed", file=sys.stderr)
BIN_LID = os.environ.get("KITCHEN_BIN_LID", "closed")                     # user 2026-10-26: closed by default (the pedal or a hand opens it);
if BIN_LID not in ("open", "closed"):                                       # "open" starts it open and a spring holds it there
    raise ValueError(f"KITCHEN_BIN_LID={BIN_LID!r}: open or closed")
DOOR_DEG = float(os.environ.get("KITCHEN_DOOR_DEG", "0"))                 # start angle of the dishwasher door, deg (2026-10-30, slice presets:
if not 0.0 <= DOOR_DEG <= 90.0:                                            # S2 starts with it open; 0 = closed as authored); kitchen_after_compile sets it
    raise ValueError(f"KITCHEN_DOOR_DEG={DOOR_DEG!r}: 0..90")
PEDAL_RETURN = os.environ.get("KITCHEN_PEDAL_RETURN", "0") != "0"        # 2026-11-02 (atomic slices, user: "one foot should be on the pedal so that it drives
                                                                           # the lid up"): a return spring on the pedal, so the lid falls shut when the foot leaves,
                                                                           # as a real pedal bin does. Off by default: as authored the pedal has no spring and a tap
                                                                           # leaves the lid up (kitchen4-7 were recorded that way, kitchen7 ep1 wiped with both feet down).
PLATE_BALLS = int(os.environ.get("KITCHEN_PLATE_BALLS", "0"))           # 2026-11-04 (user: "two free small balls on the plate, free to move around: when I lift the
PLATE_BALL_MM = float(os.environ.get("KITCHEN_PLATE_BALL_MM", "10"))    # plate I need to make it horizontal, a rolling ball tells me"): this many loose balls (radius mm)
PLATE_BALL_ROLL = float(os.environ.get("KITCHEN_PLATE_BALL_ROLL", "0.0003"))  # on every plate lying on the worktop (not in a rack, not in a hand), in a row across the well
PLATE_BALL_SOLREF = 0.02                                                  # centre, settled on its floor (_place_plate_balls). 0 = none (default; the CLI's --plate-balls N).
if PLATE_BALLS < 0 or PLATE_BALLS > 4:                                    # Rolling friction 0.0003 m (the threshold is roll / radius): measured in this scene with the plate
    raise ValueError(f"KITCHEN_PLATE_BALLS={PLATE_BALLS}: 0..4")        # held still, a 10 mm ball moves 2 mm in 2 s at 1 deg of tilt and reaches the wall at 2 deg.
for _k, _v in (("KITCHEN_REV", str(REV)), ("KITCHEN_BALLS", "1" if BALLS_ON else "0"), ("KITCHEN_BIN_LID", BIN_LID),   # (0.001 only moved from 3 deg here, unlike a bare MuJoCo scene.)
               ("KITCHEN_DOOR_DEG", f"{DOOR_DEG:g}"), ("KITCHEN_PEDAL_RETURN", "1" if PEDAL_RETURN else "0"),
               ("KITCHEN_PLATE_BALLS", str(PLATE_BALLS)), ("KITCHEN_PLATE_BALL_MM", f"{PLATE_BALL_MM:g}"), ("KITCHEN_PLATE_BALL_ROLL", f"{PLATE_BALL_ROLL:g}")):
    os.environ.setdefault(_k, _v)
KITCHEN_XML = Path(os.environ.get("KITCHEN_XML", str(BS.KITCHEN_XML)))
# procedural plates (2026-10-26, user: 50 different plates, a new one every rollout): data/kitchen_plates/plate_NN made by
# scripts/kitchen_plates/make_plates.py; the scene setups pick PLATE_ID = seed % N_PLATES for every sampled setup (the
# nominal setup keeps the authored paper plates); KITCHEN_PLATE_IDS="a,b" pins both plates; KITCHEN_PLATES="" turns it off
PLATES_DIR = Path(os.environ.get("KITCHEN_PLATES", str(Path(__file__).resolve().parents[4] / "SIMPLE/data/kitchen_plates"))) if os.environ.get("KITCHEN_PLATES", "x") != "" else None
if PLATES_DIR is not None and not (PLATES_DIR / "manifest.json").exists():
    # the folder is not there (a recording's KITCHEN_PLATES knob holds the path of the machine it was made on): the set of the
    # same name in this repo's data/ (scenes/kitchen -> the repo root), else the authoring workstation's, else the default set
    _set = PLATES_DIR.name
    PLATES_DIR = next((c for c in (HERE.parents[1] / "data" / _set, Path("/home/Horizon/wrk/SIMPLE/data") / _set,
                                   HERE.parents[1] / "data" / "kitchen_plates", Path("/home/Horizon/wrk/SIMPLE/data/kitchen_plates")) if (c / "manifest.json").exists()), None)
    if PLATES_DIR is not None:
        print(f"[kitchen] plate set {_set}: {PLATES_DIR}", flush=True)
PLATE_MANIFEST = json.loads((PLATES_DIR / "manifest.json").read_text()) if PLATES_DIR is not None else None
N_PLATES = len(PLATE_MANIFEST["plates"]) if PLATE_MANIFEST else 0
PLATE_ID = None                                                                     # set by SceneSetups.apply (setup["plate"]) before each reset
PLATE_ID_B = None                                                                   # plate B's own type when a setup draws it (atomic slices, 2026-11-02); None = PLATE_ID + 25
PLATE_B_OFFSET = 25                                                                 # plate_b gets plate (PLATE_ID + 25) % N: two types per episode, each type twice in 50 episodes
# 2026-11-04: sphere feet under the plates (plate_swap feet=): 0 = the flat foot cylinder as recorded so far (it rattles in place under
# SIMPLE's solver, 40-70 deg/s of angular jitter without moving, which sets the indicator balls creeping); the lift set's plates carry
# foot_spheres 4 in their meta; KITCHEN_PLATE_FEET overrides (the CLI sets 4 with --plate-balls). Recorded with the knobs.
PLATE_FEET = int(os.environ.get("KITCHEN_PLATE_FEET", str(int((PLATE_MANIFEST or {}).get("foot_spheres", 0)))))
os.environ.setdefault("KITCHEN_PLATE_FEET", str(PLATE_FEET))
# slice start presets (2026-11-01, docs/long_horizon_slicing_scenes; user: "create each s1,2,3,4,5 teleop scene"): what
# scene_setup.SceneSetups.apply writes here for the episode (setup["preset"]) and kitchen_after_compile puts into the compiled
# scene: door_deg (dishwasher door angle), rack_out_m (upper rack slide, 0..0.43), plates_in_rack {plate_a|plate_b: slot}
# (a plate standing in the upper rack, RACK_SLOTS), plate_spots {plate: (x, y)} (a plate elsewhere on the worktop, robot frame,
# same height), bin_lid "open" (the pedal bin's lid up, as after a pedal press; 2026-11-02). qpos only: the compiled model is the
# same for every preset, so a recording's replay (which rebuilds the model from the setup and restores state 0) is unaffected.
# None = the plain kitchen (KITCHEN_DOOR_DEG and KITCHEN_BIN_LID still apply).
SLICE_PRESET: dict | None = None
# resting poses of a plate standing in the upper rack, in the rack body's frame (pos m, quat wxyz): kitchen7 episode 1's two
# plates (docs/_scan/kitchen7_ep1_annotation/events.npz, 2026-11-02; until then kitchen5 ep1's): "near" = plate A (plate_27,
# 219 mm) as it stood after its place at 110 s, leaning 60 deg from flat, the slot nearer the robot's stance beside the door;
# "far" = plate B (plate_02, 247 mm) at the episode's end, 40 deg. Other plate types are lifted clear and settle from these.
RACK_SLOTS = {"near": dict(pos=(-0.085, -0.098, 0.099), quat=(-0.001, -0.020, 0.496, -0.868)),
              "far": dict(pos=(0.073, -0.038, 0.133), quat=(-0.937, 0.292, -0.181, 0.054))}
PLATE_FREE_JOINTS = {"plate_a": "k_plate_a_free", "plate_b": "k_plate_b_free"}
SETTLE_S = 1.0                                                                      # props moved by a preset settle this long before the scene is handed over (0.6 s left a dropped plate still creeping)
# the third-person view of the kitchen (user's viewer angle, 2026-10-26): from the aisle behind the robot's right shoulder, high
# and wide, the peninsula, the sink counter, the fridge and the shelf all in frame. Used by the viewer at launch, the replay
# videos' third-person pane and the env check's overview. MuJoCo free-camera convention (azimuth/elevation = line of sight).
VIEW_THIRD = dict(lookat=[0.3, -0.4, 0.4], distance=7.3, azimuth=-110.0, elevation=-35.0)
HEAD_HFOV_DEG, HEAD_W, HEAD_H = 90.0, 640, 360
CAM_LOCAL_POS = [0.01849, 0.03219, -0.00928]                      # as bottle_bin_task: the D455 pose offset of the stock eye_in_head mount
CAM_LOCAL_QUAT = [0.996029, -0.088426, -0.007383, -0.007305]

# ---- geometry in the robot frame --------------------------------------------------------------------------------------
START_K = BS.robot_start_kitchen(ROBOT_TO_EDGE, L["toe_offset"], ROBOT_Y_K)        # pelvis in the kitchen frame
EDGE_X = ROBOT_TO_EDGE + L["toe_offset"]                                            # worktop front edge ahead of the pelvis
L.update(dict(edge_x=round(EDGE_X, 4), far_x=round(EDGE_X + (BS.PENINSULA_FRONT_X - BS.PENINSULA_BACK_X), 4),
              bottle_x=round(EDGE_X + L["bottle_behind_edge"], 4), bottle_y=float(L["bottle_y"]),       # SIMPLE's bottle (KITCHEN_TARGET=kitchen:bottle_500ml); overwritten below for a kitchen bottle
              left_y=round(START_K[1] - BS.PENINSULA_Y[0], 4), right_y=round(START_K[1] - BS.PENINSULA_Y[1], 4),
              counter_top=round(BS.WORKTOP_TOP, 4)))
_sc = BS.kitchen_to_robot([BS.SINK_CENTER_K[0], BS.SINK_CENTER_K[1], 0.0], START_K)
SINK = dict(cx=float(_sc[0]), cy=float(_sc[1]), hx=(BS.SINK_X[1] - BS.SINK_X[0]) / 2, hy=(BS.SINK_Y[1] - BS.SINK_Y[0]) / 2,
            bottom=BS.SINK_BOTTOM_K + BS.FLOOR_LIFT, rim=BS.SINK_RIM_K + BS.FLOOR_LIFT)
L.update(sink_x=round(SINK["cx"], 4), sink_y=round(SINK["cy"], 4))
_bc = BS.kitchen_to_robot([BS.BIN_CENTER_K[0], BS.BIN_CENTER_K[1], 0.0], START_K)
BIN = dict(cx=float(_bc[0]), cy=float(_bc[1]), hx=BS.BIN_HALF[0], hy=BS.BIN_HALF[1], floor=BS.BIN_FLOOR_K + BS.FLOOR_LIFT, rim=BS.BIN_RIM_K + BS.FLOOR_LIFT)
L.update(bin_x=round(BIN["cx"], 4), bin_y=round(BIN["cy"], 4))
_own = TARGET.split(":")[-1]
if _own in BS.KITCHEN_OBJECTS:                                                      # the kitchen's own bottle stays where the kitchen puts it: its spot, in the robot frame
    _bp = BS.kitchen_to_robot(list(BS.KITCHEN_OBJECTS[_own]["pos"]), START_K)
    L.update(bottle_x=round(float(_bp[0]), 4), bottle_y=round(float(_bp[1]), 4), target=_own)


# ---- the kitchen in the compiled scene -------------------------------------------------------------------------------
def _bottle_feet(child, n_feet: int = 4, r: float = 0.0045, lift_cyl: float = 0.0015) -> None:
    """Sphere feet under the kitchen bottles' base cylinders (2026-10-26, user: "why is the green bottle shaking?").

    A cylinder standing on a box gets ONE contact point from MuJoCo's general convex collider, so the bottle balances on a
    point; the kitchen file is fine at its own 2 ms / impratio 1, but SIMPLE's solver (impratio 10, noslip 2) pumps that
    point contact and the bottle rocks more and more (3 deg after 30 s, visible shaking).  Four spheres on the base rim give
    four contact points: measured 0.0 cm/s at rest for 30 s, still after a 0.15 m/s kick, tips and rolls after a 1.2 m/s one.
    The base cylinder's bottom face is lifted 1.5 mm so it never touches the worktop while the bottle stands; the tipping
    radius shrinks from R to R - r (2.08 -> 1.63 cm).  The authored kitchen file is not changed.
    """
    for b in BS.KITCHEN_OBJECTS:
        body = child.body(b); cyl = next((g for g in body.geoms if g.name == f"{b}_collision_000"), None)
        if cyl is None or cyl.type != mujoco.mjtGeom.mjGEOM_CYLINDER:
            continue
        R, hz, z0 = float(cyl.size[0]), float(cyl.size[1]), float(cyl.pos[2]); bottom = z0 - hz
        cyl.size[1] = hz - lift_cyl / 2; cyl.pos[2] = z0 + lift_cyl / 2
        for k in range(n_feet):
            a = 2 * np.pi * k / n_feet; rc = R - r
            body.add_geom(name=f"{b}_foot_{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], pos=[rc * np.cos(a), rc * np.sin(a), bottom + r],
                          contype=1, conaffinity=1, mass=0, group=3, rgba=[0, 0, 0, 0], condim=cyl.condim, friction=cyl.friction, solref=cyl.solref, solimp=cyl.solimp, margin=cyl.margin)


def plate_ids_for(plate_id):
    """(plate_a, plate_b) plate indices for a setup's plate id; KITCHEN_PLATE_IDS="a,b" pins them."""
    pin = os.environ.get("KITCHEN_PLATE_IDS")
    if pin:
        a, b = (int(v) for v in pin.split(",")); return a % N_PLATES, b % N_PLATES
    if plate_id is None:
        return None
    if PLATE_ID_B is not None:
        return int(plate_id) % N_PLATES, int(PLATE_ID_B) % N_PLATES
    return int(plate_id) % N_PLATES, (int(plate_id) + PLATE_B_OFFSET) % N_PLATES


def _swap_plates(child) -> None:
    ids = plate_ids_for(PLATE_ID) if N_PLATES else None
    if not ids:
        return
    sys.path.insert(0, str(next((c for c in (HERE.parents[1] / "scripts" / "kitchen_plates", Path("/home/Horizon/wrk/SIMPLE/scripts/kitchen_plates")) if (c / "plate_swap.py").exists()), HERE)))
    from plate_swap import swap_plate, load_meta
    for body, pid in zip(("plate_a", "plate_b"), ids):
        pdir = PLATES_DIR / PLATE_MANIFEST["plates"][pid]["id"]; swap_plate(child, body, pdir, load_meta(pdir), feet=PLATE_FEET)
    print(f"[kitchen] plates: plate_a = {PLATE_MANIFEST['plates'][ids[0]]['id']}, plate_b = {PLATE_MANIFEST['plates'][ids[1]]['id']} (setup plate {PLATE_ID})", flush=True)


PLATE_BALL_RGBA = ((0.85, 0.12, 0.10, 1.0), (0.12, 0.35, 0.90, 1.0), (0.95, 0.80, 0.10, 1.0), (0.10, 0.65, 0.30, 1.0))   # red, blue, yellow, green


def _ball_plates() -> list:
    """The plates that get the indicator balls this episode: lying on the worktop (not standing in a rack, not in a hand)."""
    P = SLICE_PRESET or {}
    held = (P.get("plate_in_hand") or {}).get("plate")
    return [p for p in ("plate_a", "plate_b") if p not in (P.get("plates_in_rack") or {}) and p != held]


def _add_plate_balls(child) -> None:
    """KITCHEN_PLATE_BALLS free spheres per worktop plate (glass-marble mass), added to the kitchen spec; placed on the plate's
    floor after compile by _place_plate_balls (the plates move with the preset). Names k_ball_<plate>_<k> after the attach."""
    if PLATE_BALLS <= 0:
        return
    r = PLATE_BALL_MM / 1000.0; mass = 2500.0 * 4.0 / 3.0 * np.pi * r ** 3
    for plate in _ball_plates():
        pb = child.body(plate)
        for k in range(PLATE_BALLS):
            b = child.worldbody.add_body(name=f"ball_{plate}_{k}", pos=[float(pb.pos[0]) + (k - (PLATE_BALLS - 1) / 2.0) * (2 * r + 0.01), float(pb.pos[1]), float(pb.pos[2]) + 0.05 + r])
            b.add_freejoint(name=f"ball_{plate}_{k}_free")
            b.add_geom(name=f"ball_{plate}_{k}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], mass=mass, rgba=list(PLATE_BALL_RGBA[k % len(PLATE_BALL_RGBA)]),
                       condim=6, friction=[0.6, 0.005, PLATE_BALL_ROLL], solref=[PLATE_BALL_SOLREF, 1.0])


def _place_plate_balls(m, d) -> None:
    """Rest the indicator balls on their plate's inner floor wherever the preset left the plate: a row across the well centre
    along the plate's own x, each dropped from 5 mm above the plate's top surface under it (a downward ray, skipping anything
    that is not the plate, e.g. a hand of the reset pose), then settled 0.5 s with everything else held still."""
    if PLATE_BALLS <= 0:
        return
    r = PLATE_BALL_MM / 1000.0; joints = []
    mujoco.mj_forward(m, d)
    for plate in _ball_plates():
        pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "k_" + plate)
        if pb < 0:
            continue
        R, p = d.xmat[pb].reshape(3, 3), d.xpos[pb].copy()
        for k in range(PLATE_BALLS):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"k_ball_{plate}_{k}_free")
            if j < 0:
                continue
            xy = p + R @ np.array([(k - (PLATE_BALLS - 1) / 2.0) * (2 * r + 0.01), 0.0, 0.0])
            z_top, pnt, geomid = p[2] + 0.03, np.array([xy[0], xy[1], p[2] + 0.15]), np.array([-1], dtype=np.int32)
            for _ in range(5):
                dist = mujoco.mj_ray(m, d, pnt, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
                if dist < 0 or geomid[0] < 0:
                    break
                if int(m.geom_bodyid[geomid[0]]) == pb:
                    z_top = float(pnt[2] - dist); break
                pnt[2] -= dist + 0.001                                       # something above the plate (a hand): look on below it
            a, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            d.qpos[a:a + 3] = [xy[0], xy[1], z_top + r + 0.005]; d.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]; d.qvel[v:v + 6] = 0.0
            joints.append(j)
    if joints:
        _settle_free(m, d, joints, seconds=0.5)


def attach_kitchen(spec, task):
    """Attach the authored kitchen to SIMPLE's MjSpec right before it compiles (see _patch_mujoco_scene)."""
    import mujoco
    child = mujoco.MjSpec.from_file(str(KITCHEN_XML))
    child.meshdir = str(KITCHEN_XML.parent / "assets")                 # absolute: the attached spec compiles from SIMPLE's paths
    child.texturedir = str(KITCHEN_XML.parent / "assets")
    if any(tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for tex in spec.textures):    # only one skybox per model: keep SIMPLE's if it has one
        for tex in list(child.textures):
            if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                child.delete(tex)
    if os.environ.get("KITCHEN_KEEP_FLOOR", "0") != "1":                 # the kitchen's concrete floor mesh lands exactly on SIMPLE's ground plane and
        for g in list(child.geoms):                                         # z-fights with it; SIMPLE's blue checker floor stays (user, 2026-10-26)
            if g.name in ("room_shell_concrete_world",):
                child.delete(g)
    jug_dx = float(os.environ.get("KITCHEN_JUG_DX", "0.10"))             # the water jug stands against the bar-top step and looks as if it leaned over it from
    if jug_dx:                                                              # above (user, 2026-10-26): move it this far toward the aisle (kitchen +x)
        for g in child.geoms:
            if g.name in ("water_jug_water_world", "water_jug_blue_world"):
                g.pos = [float(g.pos[0]) + jug_dx, float(g.pos[1]), float(g.pos[2])]
    carton_dx = float(os.environ.get("KITCHEN_CARTON_DX", "0.03" if REV >= 2 else "0"))   # the cling-wrap carton is authored leaning on the peninsula's back step (2 contacts on
    if carton_dx:                                                           # the step, tilt 4.7 deg) and jitters at 3 cm/s under SIMPLE's solver; 3 cm toward the
        cb = child.body("carton_wrap")                                      # aisle (kitchen +x) puts it flat on the worktop: 0.0 cm/s (2026-10-26)
        cb.pos = [float(cb.pos[0]) + carton_dx, float(cb.pos[1]), float(cb.pos[2])]
    _bottle_feet(child)                                                    # multi-point bases, see _bottle_feet
    _swap_plates(child)                                                    # the episode's procedural plates (PLATE_ID), see above
    _add_plate_balls(child)                                                # KITCHEN_PLATE_BALLS loose balls per worktop plate (2026-11-04), placed after compile
    if BIN_LID == "open":                                                   # the pedal bin's lid starts open (kitchen_after_compile) and a soft spring
        lid = child.joint("bin_lid"); lid.stiffness = 3.0; lid.springref = BS.BIN_LID_OPEN   # holds it there (its own gravity would shut it)
    if PEDAL_RETURN:                                                        # the pedal springs back when the foot leaves: 3.6 N m at full press (the lid's 0.08 N m of
        ped = child.joint("bin_pedal"); ped.stiffness = 20.0; ped.springref = 0.0   # friction counts 8x at the pedal through the pedal_opens_lid equality, lid = 8 x pedal;
                                                                            # 3 N m/rad did not move it); a foot holds it down with about 25 N
    if REV >= 3:
        _dishwasher_rev3(child)
    if REV >= 2:
        floor = child.geom("bin_floor")                                     # a ball dropped from bowl height (4 m/s) punched through the 18 mm bin floor at
        top = float(floor.pos[2]) + float(floor.size[2])                    # 5 ms steps: make the floor 10 cm thick downwards (same top face, invisible below)
        floor.size[2] = 0.05; floor.pos[2] = top - 0.05
        for b in BS.KITCHEN_BOWLS:                                          # the bowls at their resting height (authored 4 mm above the worktop)
            body = child.body(b); body.pos = [float(body.pos[0]), float(body.pos[1]), BS.BOWL_REST_Z_K]
    tgt = task.layout.actors.get("target") if getattr(task, "layout", None) is not None else None
    own = tgt.asset.label if (tgt is not None and tgt.asset.label in BS.KITCHEN_OBJECTS) else None
    if own:                                                                 # the kitchen's own bottle is the target: it stays where the kitchen puts it (user, 2026-10-26)
        bp = np.asarray(child.body(own).pos, dtype=float)                   # tell SIMPLE's layout where it is (robot frame) instead of moving it to the sampled spot
        tgt.pose.position = [float(START_K[0] - bp[0]), float(START_K[1] - bp[1]), float(bp[2] + BS.FLOOR_LIFT)]
    pos, quat = BS.kitchen_frame_in_robot(START_K)
    frame = spec.worldbody.add_frame(pos=pos, quat=quat)
    spec.attach(child, frame=frame, prefix="k_")
    if own:                                                                 # the engine binds the target by its asset label (mjData.body(label))
        spec.body("k_" + own).name = own
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), 1920)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), 1080)
    return spec


DISHWASHER_DOOR = dict(stiffness=12.4, springref=-0.06, frictionloss=3.5, damping=2.0)


def _dishwasher_rev3(child) -> None:
    """The dishwasher door (2026-10-27, user: "make everything rigid parts, when I grasp the handle the hand overlaps with the
    panel ... the door should stay where it is if no pull force from the hand", video Feishu20261027-211631).

    Collision, in the door frame (x along the door, -y = the room side, z up when closed): the authored door only collided
    through an inner slab at y +36..+64 mm, while the drawn stainless front is at y -15 mm, so fingers sank 51 mm into it; the
    handle capsule (r 15 mm at y -25 mm) filled the finger gap behind the drawn chrome bar (r 8 mm at y -41 mm).  Now: a box
    for the front (y -15..+29 mm, the full door), one for the control strip on top (with its chrome trim), the handle bar
    where it is drawn, its two end posts, and the inner panel collides.  All new geoms are invisible (group 3) and massless.

    Hinge: the 5 kg door pulls open with up to 17.7 N m. A real door is spring-counterbalanced and stays at any angle; here a
    joint spring (12.4 N m/rad, springref -0.06 rad: the best linear fit to the gravity torque over 0..90 deg, residual
    <= 2.6 N m) plus 3.5 N m hinge friction holds it anywhere; opening takes ~5.4 N m (~8 N at the handle)."""
    import mujoco
    door = child.body("dishwasher_door")
    j = child.joint("dishwasher_door")
    for k, v in DISHWASHER_DOOR.items():
        setattr(j, k, v)
    inv = dict(contype=1, conaffinity=1, group=3, rgba=[0, 0, 0, 0], mass=0)
    h = child.geom("dishwasher_handle"); h.fromto = [-0.236, -0.041, 0.700, 0.236, -0.041, 0.700]; h.size = [0.008, 0, 0]
    for nm, x in (("dishwasher_handle_post_left", -0.230), ("dishwasher_handle_post_right", 0.230)):
        door.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[x, -0.033, 0.700, x, -0.012, 0.700], size=[0.008, 0, 0], **inv)
    # front y -15..+29 mm (the left tub wall starts at +30 mm), control strip y -19..+9 mm (the counter starts at +10 mm):
    # touching nothing when closed, so the door is not pushed open
    door.add_geom(name="dishwasher_front_collision", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.007, 0.3705], size=[0.28, 0.022, 0.3675], **inv)
    door.add_geom(name="dishwasher_control_strip_collision", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -0.005, 0.743], size=[0.26, 0.014, 0.013], **inv)
    ip = child.geom("dishwasher_inner_panel"); ip.contype = 1; ip.conaffinity = 1


def _settle_free(m, d, joints, seconds: float = SETTLE_S) -> None:
    """Let the given free-jointed props settle under gravity while everything else stays put (its qpos restored and its qvel
    zeroed after every step), then leave them at rest with the clock and the controls as they were."""
    import mujoco
    keep_q, keep_v = np.ones(m.nq, dtype=bool), np.ones(m.nv, dtype=bool)
    for j in joints:
        keep_q[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 7] = False
        keep_v[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = False
    q0, t0, ctrl0 = d.qpos.copy(), float(d.time), d.ctrl.copy()
    d.ctrl[:] = 0.0
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        d.qpos[keep_q] = q0[keep_q]
        d.qvel[keep_v] = 0.0
    d.qvel[:] = 0.0
    d.time, d.ctrl[:] = t0, ctrl0
    mujoco.mj_forward(m, d)


def _depenetrate(m, d, j, max_iter: int = 6, clearance: float = 0.001) -> float:
    """Raise a free-jointed prop until its deepest contact is shallower than `clearance` m: a resting pose recorded with one plate
    type puts a bigger plate through the rack floor (13-16 mm for the "near" slot's 165 mm plate under a 230 mm one), and the
    solver leaves it wedged there. Returns the lift (m); the settle then drops it onto the tines."""
    import mujoco
    a, body, lift = int(m.jnt_qposadr[j]), int(m.jnt_bodyid[j]), 0.0
    for _ in range(max_iter):
        mujoco.mj_forward(m, d)
        pen = 0.0
        for i in range(int(d.ncon)):
            c = d.contact[i]
            if body in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])) and c.dist < -clearance:
                pen = max(pen, -float(c.dist))
        if pen <= clearance:
            break
        d.qpos[a + 2] += pen + clearance
        lift += pen + clearance
    return lift


def kitchen_after_compile(sim) -> None:
    """Right after MujocoSimulator compiles the scene (see _patch_mujoco_scene): the pedal bin's lid open, the pedal down with
    it (KITCHEN_BIN_LID=open); the dishwasher door at its start angle (KITCHEN_DOOR_DEG, 2026-10-30; the rev-3 hinge holds it);
    the episode's slice preset (SLICE_PRESET, 2026-11-01): door angle, upper rack slide, plates standing in the upper rack and
    plates moved on the worktop, the moved plates settled for SETTLE_S."""
    import mujoco
    m, d = sim.mjModel, sim.mjData
    P = SLICE_PRESET or {}

    def _jid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _set(name, q):
        j = _jid(name)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q

    door_deg = float(P.get("door_deg", DOOR_DEG))
    if door_deg:
        _set("k_dishwasher_door", float(np.radians(door_deg)))
    if P.get("rack_out_m"):
        _set("k_dishwasher_upper_rack", float(P["rack_out_m"]))
    if P.get("bin_lid", BIN_LID) == "open":                                       # the knob, or the slice preset's lid
        for name, q in (("k_bin_lid", BS.BIN_LID_OPEN), ("k_bin_pedal", BS.BIN_LID_OPEN / 8.0)):  # lid = 8 x pedal (equality pedal_opens_lid)
            _set(name, q)
    moved = []
    for plate, xy in (P.get("plate_spots") or {}).items():                  # a plate elsewhere on the worktop: same height, flat, [x, y(, yaw deg)]
        j = _jid(PLATE_FREE_JOINTS[plate])
        if j >= 0:
            a = int(m.jnt_qposadr[j]); d.qpos[a:a + 2] = [float(xy[0]), float(xy[1])]
            h = np.radians(float(xy[2])) / 2.0 if len(xy) > 2 else 0.0
            d.qpos[a + 3:a + 7] = [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))]
            moved.append(j)
    if P.get("plates_in_rack"):
        mujoco.mj_forward(m, d)                                                  # the upper rack's world pose at its slide
        rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "k_dishwasher_upper_rack")
        Rr, pr, qr = d.xmat[rb].reshape(3, 3).copy(), d.xpos[rb].copy(), np.zeros(4)
        mujoco.mju_mat2Quat(qr, d.xmat[rb])
        for plate, slot in P["plates_in_rack"].items():
            j = _jid(PLATE_FREE_JOINTS[plate])
            if j < 0:
                continue
            s, a, q = RACK_SLOTS[slot], int(m.jnt_qposadr[j]), np.zeros(4)
            d.qpos[a:a + 3] = pr + Rr @ np.asarray(s["pos"], dtype=float)
            mujoco.mju_mulQuat(q, qr, np.asarray(s["quat"], dtype=float)); d.qpos[a + 3:a + 7] = q
            moved.append(j)
    if moved:
        # The robot is still in SIMPLE's reset pose here (arms held forward at worktop height, hands 0.32-0.37 m ahead of the
        # pelvis), not yet stood at its start: a start close to the counter swept plate_a off its spot during the settle
        # (2026-11-02, S1 starts at x >= 0.13: the right index finger pushed the plate 8-17 cm). Park the robot 3 m up meanwhile.
        rj = _robot_free_joint(m)
        if rj is not None:
            ra = int(m.jnt_qposadr[rj]); z_robot = float(d.qpos[ra + 2]); d.qpos[ra + 2] += 3.0
        for j in moved:
            _depenetrate(m, d, j)
        _settle_free(m, d, moved)
        if rj is not None:
            d.qpos[ra + 2] = z_robot
            mujoco.mj_forward(m, d)
    _place_plate_balls(m, d)                                                     # the indicator balls onto the (settled) worktop plates


def _robot_free_joint(m):
    """The free joint of the robot (the kinematic tree that holds the body "pelvis"), or None."""
    import mujoco
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pel < 0:
        return None
    root = int(m.body_rootid[pel])
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.body_rootid[m.jnt_bodyid[j]]) == root:
            return j
    return None


def _sync_extra(layout, am, name, asset_id, pos, quat):
    """Add / swap the kit's extra object under key `name` (as bottle_bin_task), then place it."""
    cur = layout.actors.get(name)
    if cur is not None and getattr(getattr(cur, "asset", None), "uid", name) != asset_id:
        layout.actors.pop(name)
    if name not in layout.actors:
        layout.add_object(name, am.load(asset_id))
    layout.actors[name].pose.position = list(pos)
    layout.actors[name].pose.quaternion = list(quat)


def _patch_mujoco_scene():
    """MujocoSimulator: no SIMPLE table, no HSSD room objects, the kitchen attached before compile."""
    from simple.engines.mujoco import MujocoSimulator
    if getattr(MujocoSimulator, "_kitchen_patched", False):
        return
    from simple.assets.graspnet import Graspnet1BAsset
    orig_build_object = MujocoSimulator._build_object

    def build_object(self, mjSpec, mjWorld, actor):
        if isinstance(actor.asset, SceneAsset) and actor.asset.label in BS.KITCHEN_OBJECTS:
            return None                                                 # the kitchen's own bottle: already in the attached kitchen, at its authored spot (attach_kitchen renames it)
        if isinstance(actor.asset, SceneAsset) and actor.asset.label in BS.BALLS:      # a ball in a bowl (2026-10-26): one sphere, a free body
            import mujoco
            spec = BS.BALLS[actor.asset.label]
            body = mjWorld.add_body(name=actor.asset.label, pos=list(actor.pose.position), quat=list(actor.pose.quaternion))
            body.add_geom(name=f"{actor.asset.label}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[spec["radius"], 0, 0], mass=spec["mass"],
                          rgba=list(spec["rgba"]), condim=BS.BALL_CONDIM, friction=list(BS.BALL_FRICTION), solref=[0.02, 1.0])
            body.add_freejoint(name=f"{actor.asset.label}_joint")
            return None
        if isinstance(actor.asset, (SceneAsset, Graspnet1BAsset)):      # SIMPLE's bottle (KITCHEN_TARGET=kitchen:bottle_500ml) and GraspNet distractors
            return orig_build_object(self, mjSpec, mjWorld, actor)
        return None                                                     # HSSD room furniture: the kitchen is the room

    MujocoSimulator._build_object = build_object
    MujocoSimulator._build_primitive = lambda self, mjSpec, mjWorld, actor, table_name="table": None
    original = MujocoSimulator._setup_scene
    src = textwrap.dedent(inspect.getsource(original))
    anchor = "self.mjModel=mjSpec.compile()"
    assert src.count(anchor) == 1, "MujocoSimulator._setup_scene changed; update kitchen_task._patch_mujoco_scene"
    src = src.replace(anchor, "mjSpec = attach_kitchen(mjSpec, self.task)\n    " + anchor)
    anchor2 = "    # CHECK\n    mujoco.mj_forward(self.mjModel, self.mjData)"
    assert src.count(anchor2) == 1, "MujocoSimulator._setup_scene changed; update kitchen_task._patch_mujoco_scene (after-compile hook)"
    src = src.replace(anchor2, "    kitchen_after_compile(self)\n" + anchor2)
    scope = dict(original.__globals__); scope["attach_kitchen"] = attach_kitchen; scope["kitchen_after_compile"] = kitchen_after_compile
    exec(compile(src, str(HERE / "kitchen_task.py") + "::_setup_scene", "exec"), scope)
    MujocoSimulator._setup_scene = scope["_setup_scene"]
    MujocoSimulator._kitchen_patched = True


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


# ---- assets ----------------------------------------------------------------------------------------------------------
SceneAsset = None


def _define_assets():
    global SceneAsset
    from simple.assets import AssetManager
    from simple.core.asset import Asset
    from simple.core.object import SemanticAnnotated, SpatialAnnotated

    class _SceneAsset(Asset, SemanticAnnotated, SpatialAnnotated):
        def __init__(self, uid, name, usd_path, curobo_mesh, mujoco_meshes, stable_poses, static=False, rgba=None, mass=None):
            super().__init__(uid=uid, usd_path=str(usd_path), collision_mesh_curobo=str(curobo_mesh), collision_meshes_mujoco=[str(m) for m in mujoco_meshes])
            self.name = name; self.label = uid; self.description = name
            self.stable_poses = stable_poses
            self.keypoints, self.axes, self.canonical_grasps, self.functional_grasps = {}, {}, [], {}
            self.static, self.rgba, self.mass = static, rgba, mass

        def to_dict(self):
            return {"res_id": "kitchen", "uid": self.uid, "label": self.label, "name": self.name, "usd_path": self.usd_path, "description": self.description}

        def __repr__(self):
            return f"SceneAsset({self.uid})"

    SceneAsset = _SceneAsset
    if "kitchen" not in AssetManager._registry:
        @AssetManager.register("kitchen")
        class KitchenAssetManager(AssetManager):
            def __init__(self):
                body, cap, merged, usd = (BOTTLE_BIN_ASSETS / n for n in ("bottle_500ml_body.obj", "bottle_500ml_cap.obj", "bottle_500ml.obj", "bottle_500ml.usda"))
                self.files = dict(bottle=dict(obj=merged, mujoco=[body, cap], usd=usd))      # SIMPLE's bottle, only with KITCHEN_TARGET=kitchen:bottle_500ml

            def load(self, asset_id: str, *args, **kwargs):
                if asset_id in BS.KITCHEN_OBJECTS:                       # a kitchen free body: meshes only for SIMPLE's bookkeeping, the kitchen file builds it
                    spec = BS.KITCHEN_OBJECTS[asset_id]; col = BS.KITCHEN_DIR / "assets" / "collision" / f"{asset_id}_collision_002.obj"
                    usd = BOTTLE_BIN_ASSETS / "bottle_500ml.usda"
                    return SceneAsset(asset_id, asset_id.replace("_", " "), usd, col, [col], stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=[0.8, 0.1, 0.1, 1.0], mass=spec["mass"])
                if asset_id in BS.BALLS:                                 # the balls in the bowls: bookkeeping only, build_object makes the sphere
                    spec = BS.BALLS[asset_id]; col = BS.KITCHEN_DIR / "assets" / "collision" / f"{spec['bowl']}_collision_002.obj"
                    return SceneAsset(asset_id, asset_id.replace("_", " "), BOTTLE_BIN_ASSETS / "bottle_500ml.usda", col, [col],
                                      stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], rgba=list(spec["rgba"]), mass=spec["mass"])
                if asset_id == "bottle_500ml":
                    f = self.files["bottle"]
                    return SceneAsset("bottle_500ml", "bottle", f["usd"], f["obj"], f["mujoco"], stable_poses=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                                      rgba=[0.62, 0.80, 0.95, 0.75], mass=BS.BOTTLE_MASS)
                raise ValueError(f"unknown kitchen asset {asset_id}")

            def sample(self, exclude=None):
                return self.load(TARGET.split(":")[-1])

            def __len__(self):
                return 1 + len(BS.KITCHEN_OBJECTS)

            def __iter__(self):
                yield self.load(TARGET.split(":")[-1])


# ---- the task ----------------------------------------------------------------------------------------------------------
def _define_task():
    import transforms3d as t3d
    from simple.dr import CameraDRCfg, DistractorDRCfg, LanguageDRCfg, MaterialDRCfg, SpatialDRCfg, TabletopSceneDRCfg, TargetDRCfg
    from simple.dr.types import Box
    from simple.sensors import StereoCameraCfg
    from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
    from simple.tasks.registry import TaskRegistry

    parent = G1WholebodyXMovePickTaskTeleop
    cam_quat = [1.0, 0.0, 0.0, 0.0]                                   # SIMPLE's eye_in_head mount needs an identity orientation (no D455 pose patch here)

    @TaskRegistry.register(TASK_UID)
    class G1WholebodyKitchenSinkTeleop(parent):
        viewer_cam = VIEW_THIRD                                             # the viewer opens at the kitchen overview angle
        uid: str = TASK_UID
        label: str = "G1 Kitchen Sink Teleop"
        isaac_cameras_follow_mujoco = True
        description: str = ("G1 in the aisle of a kitchen, facing the peninsula worktop; the sink is to the robot's right, the pedal bin "
                            "ahead-left. bottle_sink: pick the red bottle and put it in the sink. ball_bin: pick up a bowl with a ball in it "
                            "and tip the ball into the open pedal bin.")
        metadata = dict(parent.metadata)
        sensor_cfgs = dict(
            head_stereo=StereoCameraCfg(uid="Realsense_D455", mount="eye_in_head", width=HEAD_W, height=HEAD_H, focal_length=1.93,
                                        fov=np.deg2rad(HEAD_HFOV_DEG), near=0.05, far=8, baseline=0.05,
                                        pose=dict(position=list(CAM_LOCAL_POS), quaternion=cam_quat)))
        dr_cfgs = dict(
            language=LanguageDRCfg(instructions=[INSTRUCTION]),
            target=TargetDRCfg(asset_id=TARGET),
            distractors=DistractorDRCfg(res_id="graspnet1b", number_of_distractors=NUM_DISTRACTORS, allow_duplicates=False, exclude=[]),
            spatial=SpatialDRCfg(
                spatial_mode="random",
                robot_region=Box(low=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0]),
                target_region=Box(low=[L["bottle_x"] - TARGET_JITTER[0], L["bottle_y"] - TARGET_JITTER[1]], high=[L["bottle_x"] + TARGET_JITTER[0], L["bottle_y"] + TARGET_JITTER[1]]),
                distractors_region=Box(low=[L["edge_x"] + 0.10, -0.55], high=[L["edge_x"] + 0.55, 0.55]),
                target_stable_indices=[0], target_rotate_z=Box(low=0.0, high=0.0),
            ),
            camera=CameraDRCfg(cam_id="head_stereo"),
            scene=TabletopSceneDRCfg(                                     # describes the worktop for SIMPLE's placement; the table itself is not built
                scene_mode="fixed",
                table_size=Box(low=[BS.TABLE_D, BS.TABLE_W, BS.TABLE_TOP_T], high=[BS.TABLE_D, BS.TABLE_W, BS.TABLE_TOP_T]),
                table_position=Box(low=[L["edge_x"] + BS.TABLE_D / 2, 0.0], high=[L["edge_x"] + BS.TABLE_D / 2, 0.0]),
                table_height=Box(low=BS.TABLE_H, high=BS.TABLE_H),
                rotation_z=Box(low=0.0, high=0.0),
                room_choices=[os.environ.get("KITCHEN_ROOM", "hssd:scene2")], scene_manager="hssd", randomize_scene_pose=False,
            ),
            lighting=deepcopy(parent.dr_cfgs["lighting"]),
            material=MaterialDRCfg(material_mode="fixed"),
        )

        # gates: grasped (hand on the bottle, off the counter, upright, 0.5 s) -> at_sink (pelvis within AT_SINK_RADIUS of the
        # sink, walked, still 0.4 s) -> placed (bottle base inside the basin, below the rim, hand off, 0.5 s)
        GRASP_LIFT, GRASP_HOLD_S, GRASP_TILT = 0.03, 0.5, 60.0
        AT_SINK_RADIUS, AT_SINK_SPEED, AT_SINK_HOLD_S, MIN_WALK = 0.90, 0.15, 0.4, 0.30
        PLACE_HOLD_S, STEP_DT = 0.5, 0.02

        def reset(self, seed=None, options=None):
            super().reset(seed, options)
            if TASK == "load_dishwasher":
                self._progress = dict(door_open=False, grasped=False, placed=False, rack_in=False, door_closed=False, t_door_open=None, t_grasped=None,
                                      t_placed=None, t_rack_in=None, t_door_closed=None, max_lift=0.0, loaded=0, loaded_dishes="", max_loaded=0,
                                      door_deg=0.0, rack_out_m=0.0, max_rack_out_m=0.0)
                self._dish_z0 = None; self._rest_steps = {}; self._closed_steps = 0
            elif TASK == "ball_bin":
                self._progress = dict(grasped=False, at_bin=False, placed=False, t_grasped=None, t_at_bin=None, t_placed=None, max_lift=0.0,
                                      min_pelvis_to_bin=None, bowl="", ball_in_bin="")
            else:
                self._progress = dict(grasped=False, at_sink=False, placed=False, t_grasped=None, t_at_sink=None, t_placed=None, max_lift=0.0, min_pelvis_to_sink=None)
            self._gate_steps = 0; self._hold_steps = 0; self._in_sink_steps = 0; self._still_steps = 0
            self._start_xy = None; self._gate_model_id = None; self._gate_info_id = None; self._bowl_z0 = None
            self._instruction = INSTRUCTION
            if BALLS_ON:                                                    # a ball in each bowl, resting on its inner floor
                from simple.assets import AssetManager
                am = AssetManager.get("kitchen")
                for name, spec in BS.BALLS.items():
                    bp = BS.KITCHEN_BOWLS[spec["bowl"]]["pos"]; rest = BS.BALL_REST[name]
                    pos = BS.kitchen_to_robot(list(bp), START_K) + np.array([rest[0], rest[1], 0.0]); pos[2] = BS.BOWL_REST_Z_K + BS.FLOOR_LIFT + rest[2]
                    _sync_extra(self._layout, am, name, name, [float(v) for v in pos], [1.0, 0.0, 0.0, 0.0])

        def _bind_gate_ids(self, m):
            import mujoco
            if self._gate_model_id == id(m):
                return
            bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            self._g_target = bid(str(self.target.asset.label))
            self._g_target_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self._g_target}
            names = {b: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(m.nbody)}
            self._g_rhand_geoms = {g for g in range(m.ngeom) if names[m.geom_bodyid[g]].startswith("right_hand_")}
            gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            self._g_table_geoms = {g for g in range(m.ngeom) if any(s in gname(g) for s in ("peninsula", "worktop", "counter"))}
            self._g_lhand_geoms = {g for g in range(m.ngeom) if names[m.geom_bodyid[g]].startswith("left_hand_")}
            self._g_bowls = {n: bid("k_" + n) for n in BS.KITCHEN_BOWLS if bid("k_" + n) >= 0}          # ball_bin: the bowls and the balls
            self._g_bowl_geoms = {n: {g for g in range(m.ngeom) if m.geom_bodyid[g] == b} for n, b in self._g_bowls.items()}
            self._g_balls = {n: bid(n) for n in BS.BALLS if bid(n) >= 0}
            self._g_dishes = {n: bid("k_" + n) for n in DISHES if bid("k_" + n) >= 0}                  # load_dishwasher
            self._g_dish_geoms = {n: {g for g in range(m.ngeom) if m.geom_bodyid[g] == b} for n, b in self._g_dishes.items()}
            self._g_racks = [bid("k_" + r) for r in RACKS if bid("k_" + r) >= 0]
            jd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "k_dishwasher_door"); self._g_door_q = int(m.jnt_qposadr[jd]) if jd >= 0 else None
            jr = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "k_" + r) for r in RACKS]                 # the racks' slide joints (0 = home)
            self._g_rack_q = [int(m.jnt_qposadr[j]) for j in jr if j >= 0]
            self._gate_model_id = id(m)

        def _contacts(self, d):
            """(right hand touches the bottle, False, bottle touches the counter): the bottle_bin shape, for the replay tools."""
            hand, table = False, False
            for i in range(d.ncon):
                c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
                t1, t2 = g1 in self._g_target_geoms, g2 in self._g_target_geoms
                if (t1 and g2 in self._g_rhand_geoms) or (t2 and g1 in self._g_rhand_geoms): hand = True
                if (t1 and g2 in self._g_table_geoms) or (t2 and g1 in self._g_table_geoms): table = True
            return hand, False, table

        def _bowl_contacts(self, d):
            """{bowl: (a hand touches it, it touches the counter)}."""
            out = {n: [False, False] for n in self._g_bowls}
            hands = self._g_rhand_geoms | self._g_lhand_geoms
            for i in range(d.ncon):
                c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
                for n, gs in self._g_bowl_geoms.items():
                    other = g2 if g1 in gs else g1 if g2 in gs else None
                    if other is None:
                        continue
                    if other in hands:
                        out[n][0] = True
                    if other in self._g_table_geoms:
                        out[n][1] = True
            return {n: tuple(v) for n, v in out.items()}

        @staticmethod
        def _in_bin(p):
            """A ball centre inside the pedal bin: within its floor footprint (minus the ball), above the floor, below the rim."""
            x, y, z = (float(v) for v in p[:3]); r = 0.02
            return abs(x - BIN["cx"]) <= BIN["hx"] - r and abs(y - BIN["cy"]) <= BIN["hy"] - r and BIN["floor"] - 0.02 < z < BIN["rim"]

        def _gates_load_dishwasher(self, info, m, d, t):
            """load_dishwasher: door_open (door >= 69 deg) / grasped (a hand on a dish that is 3 cm up or off the counter, 0.5 s)
            / placed (a dish resting in a rack, no hand on it, 0.5 s) / rack_in (after placed: every rack within RACK_IN_M of home)
            / door_closed (racks in, door within DOOR_CLOSED_RAD of shut, a dish still loaded, CLOSED_HOLD_S; success since 2026-10-30,
            before that placed was). loaded = dishes resting in a rack now; rack_out_m = the furthest rack's slide."""
            P = self._progress
            if self._dish_z0 is None:
                self._dish_z0 = {n: float(d.xipos[b][2]) for n, b in self._g_dishes.items()}
            if self._g_door_q is not None:
                P["door_deg"] = round(float(np.degrees(d.qpos[self._g_door_q])), 1)
                if not P["door_open"] and d.qpos[self._g_door_q] >= DOOR_OPEN_RAD:
                    P["door_open"], P["t_door_open"] = True, t
            hands = self._g_rhand_geoms | self._g_lhand_geoms
            touch, on_table = {n: False for n in self._g_dishes}, {n: False for n in self._g_dishes}
            for i in range(d.ncon):
                c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
                for n, gs in self._g_dish_geoms.items():
                    other = g2 if g1 in gs else g1 if g2 in gs else None
                    if other is None:
                        continue
                    if other in hands: touch[n] = True
                    if other in self._g_table_geoms: on_table[n] = True
            lifted = {n: float(d.xipos[b][2]) - self._dish_z0[n] for n, b in self._g_dishes.items()}
            P["max_lift"] = max(P["max_lift"], round(max(lifted.values(), default=0.0), 4))
            holding = [n for n in self._g_dishes if touch[n] and (lifted[n] >= self.GRASP_LIFT or not on_table[n])]
            self._hold_steps = self._hold_steps + 1 if holding else 0
            if not P["grasped"] and self._hold_steps * self.STEP_DT >= self.GRASP_HOLD_S:
                P["grasped"], P["t_grasped"] = True, t
            loaded = []
            for n, b in self._g_dishes.items():
                inside = False
                for rb in self._g_racks:
                    loc = d.xmat[rb].reshape(3, 3).T @ (d.xipos[b] - d.xpos[rb])
                    if abs(loc[0]) <= RACK_BOX["x"] and abs(loc[1]) <= RACK_BOX["y"] and RACK_BOX["z"][0] <= loc[2] <= RACK_BOX["z"][1]:
                        inside = True; break
                self._rest_steps[n] = self._rest_steps.get(n, 0) + 1 if (inside and not touch[n]) else 0
                if self._rest_steps[n] * self.STEP_DT >= self.PLACE_HOLD_S:
                    loaded.append(n)
            P["loaded"], P["loaded_dishes"] = len(loaded), ",".join(loaded); P["max_loaded"] = max(P["max_loaded"], len(loaded))
            if not P["placed"] and loaded:
                P["placed"], P["t_placed"] = True, t
            slide = [float(d.qpos[q]) for q in self._g_rack_q]
            P["rack_out_m"] = round(max(slide, default=0.0), 3); P["max_rack_out_m"] = max(P["max_rack_out_m"], P["rack_out_m"])
            racks_in = all(v <= RACK_IN_M for v in slide)
            if P["placed"] and not P["rack_in"] and racks_in:
                P["rack_in"], P["t_rack_in"] = True, t
            door_shut = self._g_door_q is not None and float(d.qpos[self._g_door_q]) <= DOOR_CLOSED_RAD
            self._closed_steps = self._closed_steps + 1 if (P["placed"] and racks_in and door_shut and loaded) else 0
            if not P["door_closed"] and self._closed_steps * self.STEP_DT >= CLOSED_HOLD_S:
                P["door_closed"], P["t_door_closed"] = True, t
            P["t"] = t
            info["task_progress"] = dict(P)
            return P

        def _gates_ball_bin(self, info, m, d, t):
            """ball_bin: grasped (a hand on a bowl that is off the counter or 3 cm up, 0.5 s) -> at_bin (pelvis within
            AT_SINK_RADIUS of the bin, walked, still 0.4 s) -> placed (a ball inside the bin for 0.5 s)."""
            P = self._progress
            if self._bowl_z0 is None:
                self._bowl_z0 = {n: float(d.xpos[b][2]) for n, b in self._g_bowls.items()}
            touch = self._bowl_contacts(d)
            lifted = {n: float(d.xpos[b][2]) - self._bowl_z0[n] for n, b in self._g_bowls.items()}
            P["max_lift"] = max(P["max_lift"], round(max(lifted.values(), default=0.0), 4))
            holding = [n for n in self._g_bowls if touch[n][0] and (lifted[n] >= self.GRASP_LIFT or not touch[n][1])]
            self._hold_steps = self._hold_steps + 1 if holding else 0
            if not P["grasped"] and self._hold_steps * self.STEP_DT >= self.GRASP_HOLD_S:
                P["grasped"], P["t_grasped"], P["bowl"] = True, t, holding[0]
            pelvis_xy = np.asarray(d.qpos[:2], dtype=float)
            if self._start_xy is None:
                self._start_xy = pelvis_xy.copy()
            dist = float(np.hypot(pelvis_xy[0] - BIN["cx"], pelvis_xy[1] - BIN["cy"]))
            P["min_pelvis_to_bin"] = round(dist, 3) if P["min_pelvis_to_bin"] is None else min(P["min_pelvis_to_bin"], round(dist, 3))
            if P["grasped"] and not P["at_bin"]:
                speed = float(np.hypot(*np.asarray(d.qvel[:2], dtype=float))); walked = float(np.hypot(*(pelvis_xy - self._start_xy)))
                self._still_steps = self._still_steps + 1 if speed < self.AT_SINK_SPEED else 0
                if dist <= self.AT_SINK_RADIUS and walked >= self.MIN_WALK and self._still_steps * self.STEP_DT >= self.AT_SINK_HOLD_S:
                    P["at_bin"], P["t_at_bin"] = True, t
            inside = [n for n, b in self._g_balls.items() if self._in_bin(d.xpos[b])]
            P["ball_in_bin"] = inside[0] if inside else ""
            self._in_sink_steps = self._in_sink_steps + 1 if inside else 0
            if not P["placed"] and self._in_sink_steps * self.STEP_DT >= self.PLACE_HOLD_S:
                P["placed"], P["t_placed"] = True, t
            P["t"] = t
            info["task_progress"] = dict(P)
            return P

        @staticmethod
        def _in_sink(target_pose):
            x, y, z = (float(v) for v in target_pose[:3])
            return (abs(x - SINK["cx"]) <= SINK["hx"] - BS.BOTTLE_R and abs(y - SINK["cy"]) <= SINK["hy"] - BS.BOTTLE_R
                    and SINK["bottom"] - 0.02 < z < SINK["rim"] - 0.01)

        def update_gates(self, info, mujoco_env=None):
            if mujoco_env is None or not hasattr(mujoco_env, "mjModel"):
                return self._progress
            m, d = mujoco_env.mjModel, mujoco_env.mjData
            key = (id(info), float(d.time))                              # env.step calls compute_reward and then check_success on the same info: evaluate once
            if self._gate_info_id == key:                                # per step (the sim time too: a freed info dict's id comes back on the next one)
                return self._progress
            self._gate_info_id = key
            self._bind_gate_ids(m)
            P = self._progress; self._gate_steps += 1; t = round(self._gate_steps * self.STEP_DT, 2)
            if TASK == "load_dishwasher":
                return self._gates_load_dishwasher(info, m, d, t)
            if TASK == "ball_bin":
                return self._gates_ball_bin(info, m, d, t)
            target = np.asarray(info["target"], dtype=float)
            if self._init_target_height is None:
                self._init_target_height = float(target[2])
            lift = float(target[2] - self._init_target_height); P["max_lift"] = max(P["max_lift"], round(lift, 4))
            hand, _, on_table = self._contacts(d)
            pelvis_xy = np.asarray(d.qpos[:2], dtype=float)
            if self._start_xy is None:
                self._start_xy = pelvis_xy.copy()
            qw, qx, qy, qz = target[3:7]
            tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1))))
            holding = hand and (lift >= self.GRASP_LIFT or not on_table) and tilt < self.GRASP_TILT
            self._hold_steps = self._hold_steps + 1 if holding else 0
            if not P["grasped"] and self._hold_steps * self.STEP_DT >= self.GRASP_HOLD_S:
                P["grasped"], P["t_grasped"] = True, t
            dist = float(np.hypot(pelvis_xy[0] - SINK["cx"], pelvis_xy[1] - SINK["cy"]))
            P["min_pelvis_to_sink"] = round(dist, 3) if P["min_pelvis_to_sink"] is None else min(P["min_pelvis_to_sink"], round(dist, 3))
            if P["grasped"] and not P["at_sink"]:
                speed = float(np.hypot(*np.asarray(d.qvel[:2], dtype=float))); walked = float(np.hypot(*(pelvis_xy - self._start_xy)))
                self._still_steps = self._still_steps + 1 if speed < self.AT_SINK_SPEED else 0
                if dist <= self.AT_SINK_RADIUS and walked >= self.MIN_WALK and self._still_steps * self.STEP_DT >= self.AT_SINK_HOLD_S:
                    P["at_sink"], P["t_at_sink"] = True, t
            inside = P["grasped"] and not hand and self._in_sink(target)
            self._in_sink_steps = self._in_sink_steps + 1 if inside else 0
            if not P["placed"] and self._in_sink_steps * self.STEP_DT >= self.PLACE_HOLD_S:
                P["placed"], P["t_placed"] = True, t
            P["t"] = t
            info["task_progress"] = dict(P)
            return P

        def compute_reward(self, info, *args, **kwargs):
            P = self.update_gates(info, kwargs.get("mujoco_env"))
            if TASK == "load_dishwasher":                                   # 2026-10-30: success = the door shut on the loaded rack, not the first dish
                if P["door_closed"]:
                    return 1.0
                return 0.3 * float(P["door_open"]) + 0.2 * float(P["grasped"]) + 0.2 * float(P["placed"]) + 0.1 * float(P["rack_in"])
            if P["placed"]:
                return 1.0
            return 0.3 * float(P["grasped"]) + 0.3 * float(P.get("at_sink", False) or P.get("at_bin", False) or P.get("door_open", False))

        def check_success(self, info, *args, **kwargs):
            return self.compute_reward(info, *args, **kwargs) >= self.success_criteria

    return G1WholebodyKitchenSinkTeleop


def register():
    import gymnasium as gym
    import simple.envs  # noqa: F401
    _patch_renderer_framebuffer()
    _define_assets()
    _patch_mujoco_scene()
    task_cls = _define_task()
    if ENV_ID not in gym.envs.registry:
        gym.register(id=ENV_ID, entry_point="simple.envs.sonic_loco_manip:SonicLocoManipEnv", kwargs={"task": TASK_UID})
    print(f"[kitchen] robot start {ROBOT_TO_EDGE:.3f} m from the worktop edge (kitchen frame {np.round(START_K, 3).tolist()}, yaw 180 deg); "
          f"bottle at ({L['bottle_x']:.3f}, {L['bottle_y']:+.3f}), sink at ({L['sink_x']:.3f}, {L['sink_y']:+.3f}), worktop top {BS.WORKTOP_TOP:.3f} m; "
          f"task {TASK}, scene rev {REV}, pedal bin at ({L['bin_x']:.3f}, {L['bin_y']:+.3f}) lid {BIN_LID}, balls {'in both bowls' if BALLS_ON else 'off'}"
          + (f", {PLATE_BALLS} loose {PLATE_BALL_MM:g} mm balls on every worktop plate (KITCHEN_PLATE_BALLS)" if PLATE_BALLS else ""), flush=True)
    return task_cls


if __name__ != "__main__":
    try:
        register()
    except Exception as e:  # noqa: BLE001
        print(f"[kitchen] task not registered on import: {e}", file=sys.stderr)
