"""Kitchen scene constants (BS) for the HoloMotion v1.4 sim teleop kit `kitchen`.

The kitchen is the authored, manipulable MuJoCo kitchen of /home/Horizon/wrk/KitchenScene_Manipulable_20261026
(manipulable_v3: 16 mechanism joints, 8 free props, 1068 geoms). It is attached whole to the SIMPLE scene, so there is
nothing to build here; this module only records the numbers the task and the scene-setup sampler need.

Frames: the kitchen file is z-up metres with its collision floor at z = FLOOR_Z; SIMPLE's ground is at z = 0, so the
kitchen is lifted by FLOOR_LIFT. The robot starts at SIMPLE's origin facing +x; the kitchen is rotated 180 deg and placed
so that the peninsula worktop's front edge (kitchen x = PENINSULA_FRONT_X, facing +x in the kitchen) is ahead of the
robot and the sink is to its right. kitchen_to_robot() does the conversion.
"""
import os
from pathlib import Path
import numpy as np

_WS_KITCHEN = Path("/home/Horizon/wrk/KitchenScene_Manipulable_20261026")      # the authoring workstation's copy (used when present)
# KITCHEN_SCENE_DIR if set, else the workstation copy, else the copy in this repo (scenes/kitchen/kitchen_scene: scene.xml + assets/)
KITCHEN_DIR = Path(os.environ.get("KITCHEN_SCENE_DIR") or (_WS_KITCHEN if (_WS_KITCHEN / "scene.xml").exists() else Path(__file__).resolve().parent / "kitchen_scene"))
KITCHEN_XML = KITCHEN_DIR / "scene.xml"
FLOOR_Z = -0.025                      # the kitchen's floor plane (scene.xml geom "floor")
FLOOR_LIFT = -FLOOR_Z                 # +0.025 m: kitchen -> SIMPLE ground
# peninsula worktop (collider "peninsula_worktop": centre (-0.86, 1.27, 0.831), half (0.35, 1.035, 0.016))
PENINSULA_FRONT_X = -0.51             # the edge the robot faces; drawers open toward +x
PENINSULA_BACK_X = -1.21
PENINSULA_Y = (0.235, 2.305)
WORKTOP_TOP_K = 0.847                 # kitchen z of the worktop surface
WORKTOP_TOP = WORKTOP_TOP_K + FLOOR_LIFT   # 0.872 in SIMPLE's frame
# sink bowl (invisible guide boxes sink_bowl_*: bottom top face z 0.68, walls to z 0.83, inner x 0.095..0.645, y 2.445..2.835)
SINK_X = (0.095, 0.645)
SINK_Y = (2.445, 2.835)
SINK_BOTTOM_K, SINK_RIM_K = 0.68, 0.83
SINK_CENTER_K = ((SINK_X[0] + SINK_X[1]) / 2, (SINK_Y[0] + SINK_Y[1]) / 2)      # (0.37, 2.64) = the file's sink_target site
# pedal bin (site bin_target) and the fridge handle, for reference
BIN_K = (-0.8, 0.065)
# the robot's start along the peninsula (kitchen y) and the kitchen_workspace site
ROBOT_Y_K_DEFAULT = 0.875                # a clear stretch of worktop between the small bowl (kitchen y 0.70) and plate_a (1.10)
# SIMPLE bottle (the bottle_bin kit's 500 ml bottle)
BOTTLE_R, BOTTLE_H, BOTTLE_MASS = 0.027, 0.22, 0.25        # the kitchen's own red condiment bottle (objects.json), the task target since 2026-10-26
KITCHEN_OBJECTS = {                   # kitchen free bodies usable as the task target (objects.json: authored kitchen-frame pos, base of the bottle on the worktop)
    "bottle_red": dict(pos=(-0.99, 1.91, 0.847), radius=0.027, height=0.22, mass=0.25),
    "bottle_green": dict(pos=(-0.91, 1.93, 0.847), radius=0.026, height=0.23, mass=0.25),
}
# the two bowls (objects.json; base on the worktop) and the ball in each (2026-10-26, user: "add a small ball in each of the
# bowls, i want to grasp the bowl, move to the trash bin and throw the small ball inside it into the trash bin"). A 4 cm
# ball rests 26 mm above a bowl's base (inner floor 6 mm + radius; measured, 0.0 cm/s at rest under SIMPLE's solver).
KITCHEN_BOWLS = {
    "bowl_small": dict(pos=(-0.7, 0.7, 0.851), radius=0.07, height=0.055, mass=0.18),
    "bowl_large": dict(pos=(-0.7, 1.81, 0.851), radius=0.085, height=0.07, mass=0.28),
}
BALLS = {
    "ball_small": dict(bowl="bowl_small", radius=0.02, mass=0.03, rgba=(1.0, 0.45, 0.05, 1.0)),      # orange, in the small bowl (17 cm left of the robot's start)
    "ball_large": dict(bowl="bowl_large", radius=0.02, mass=0.03, rgba=(1.0, 0.85, 0.10, 1.0)),      # yellow, in the large bowl (94 cm to the right, by the red bottle)
}
BALL_REST_ABOVE_BASE = 0.026
# the ball starts on the bowl's axis, 25.8 mm above the base (robot frame dx, dy, dz). The bowls' inner floors are faceted
# hulls: a ball with torsional friction only crept 4-6 cm across them in 30 s, so the balls get rolling friction (condim 6,
# 0.003 m: a sphere of 2 cm radius holds on slopes up to 8 deg and rolls freely once the bowl is tipped further)
BALL_REST = {"ball_small": (0.0, 0.0, 0.0258), "ball_large": (0.0, 0.0, 0.0258)}
BALL_FRICTION, BALL_CONDIM = (0.8, 0.02, 0.003), 6
BOWL_REST_Z_K = 0.8471                # the bowls are authored 4 mm above the worktop (0.851 vs 0.847) and drop at the start: placed at their resting height instead
# pedal bin (body pedal_bin): walls bin_wall_* on a rounded 0.36 x 0.23 m footprint up to kitchen z 0.558, floor bin_floor
# (centre (-0.80, 0.083, 0.041), half (0.17, 0.11, 0.009)); lid hinge (-0.8, 0.225, 0.57) range 0..1.45 rad, the pedal
# opens it (equality pedal_opens_lid: lid = 8 x pedal). In the robot frame the bin stands 0.64 m ahead, 0.79 m to the left.
BIN_CENTER_K = (-0.80, 0.083)
BIN_HALF = (0.17, 0.11)
BIN_FLOOR_K, BIN_RIM_K = 0.05, 0.558
BIN_LID_OPEN = 1.45
HAND_RGBA = "0.10 0.10 0.10 1"
# the SIMPLE table the scene randomizer still describes (not built: the kitchen worktop takes its place)
TABLE_D, TABLE_W, TABLE_TOP_T, TABLE_H = 0.70, 2.07, 0.032, WORKTOP_TOP


def robot_start_kitchen(robot_to_edge: float, toe_offset: float, robot_y_k: float = ROBOT_Y_K_DEFAULT):
    """Pelvis position of the robot in the kitchen frame: `robot_to_edge` from the front of the feet to the worktop edge."""
    return np.array([PENINSULA_FRONT_X + robot_to_edge + toe_offset, robot_y_k, 0.0])


def kitchen_to_robot(p_k, start_k):
    """A kitchen-frame point -> SIMPLE/robot frame (robot at the origin facing +x, kitchen rotated 180 deg and lifted)."""
    p = np.asarray(p_k, float); s = np.asarray(start_k, float)
    return np.array([s[0] - p[0], s[1] - p[1], (p[2] if p.shape[0] > 2 else 0.0) + FLOOR_LIFT])


def kitchen_frame_in_robot(start_k):
    """pos, quat (w x y z) of the kitchen's origin in the robot frame, for MjSpec.attach."""
    s = np.asarray(start_k, float)
    return [float(s[0]), float(s[1]), FLOOR_LIFT], [0.0, 0.0, 0.0, 1.0]
