"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Per-episode random setups for the three real-to-sim teleop scenes (bottle_bin, bowl_sink, coffee_cart).

A setup is drawn from its own seed and holds exact values (not ranges), so the scene it describes can be rebuilt later:

    robot        start x, y (m) and yaw (deg) in the scene frame (pelvis of the nominal start at the origin, x forward)
    target_xy    the bottle / bowl / cup (x, y)
    L            layout overrides the scene module reads at reset (bin, cart and cup positions)
    distractors  number of GraspNet distractors (their pick and poses come from SIMPLE's DR, seeded by the setup)
    table_dz     (level-3 ranges only) the table / counter / cart-box height offset (m, whole mm)

``apply(setup)`` writes these into the live task before ``env.reset``: the spatial randomizer's robot / target regions
become zero-width boxes, the scene module's layout dict ``L`` gets the overrides (the tasks read it in ``reset``), the
distractor randomizer gets its count, the scene randomizer the table height, and numpy's and Python's global RNGs (SIMPLE's
DR draws from both: numpy for poses, lighting and table jitter; ``random`` for the distractor pick, their turn, the
materials and stable poses) are seeded. The scene task files are not changed. Until 2026-09-30 only numpy was seeded,
so a setup's seed did not fix which distractors appeared; recorded episodes keep their exact scene in
environment_config, which is what the replay (and eval --scenes-from) loads.

Two sets of ranges, offsets from each scene's nominal layout (after the teleop start distance):

    teleop  (RANGES) chosen so the task stays doable: the table/counter/cart is still in reach after walking in, the bin
            stays clear of the walk (see the 2026-09-23 free-walk fit), the cup stays on the box top
    lv3     (LV3_RANGES) the kits' level-3 evaluation sets (holobrain_g1_deploy/sim/make_levels.py): robot start
            +-10 cm back/forward and +-5 cm sideways (no turn), table / counter / cart-box height +-4 cm, the kit's
            level-2/3 target region, 3 distractors; the bin and the cart stay put, as in those sets. make_levels
            spreads 10 scenes evenly over the box; here every setup is a uniform draw from it. coffee_cart's robot
            span is shifted 10 cm back (see LV3_RANGES).
"""

from __future__ import annotations

import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

SETUP_VERSION = 1
# A2 (Pick) starts from the recorded A1 ends since 2026-11-03 (user: "use the A1 collected data to redesign the A2 scene");
# since 2026-11-04 EXACTLY (user: "use the same location and pose seed from A1 recorded data"): bank entry seed % 37, no jitter,
# that episode's plates; KITCHEN_A2_START=a1 = the 11-03 envelope kind, v1 = the 2026-11-02 kinds. The hands are never taken
# from the record: 22 of the 74 A1 hands were closed because the operator was holding the grip, and 14 of those (curl 0.96,
# fingertips in the palm) lock after a qpos teleport exactly like the old closed-grip draw (scratchpad bank_hands_open_test.py);
# every free hand gets the resting draw (user 2026-11-03: open for the next grasp, a regular human hand).
A3_START = os.environ.get("KITCHEN_A3_START", "a2_final")                # a2_final (the 30 ends of the final A2 dataset, data/teleop_holomotion_v14_kitchen_a2_final; user 2026-11-10 night: "use the final one to create A3 scenes") | a2_press (the recorded A2 press-and-grasp ends, in seed order; 2026-11-10 "use the 28 scenes to set the A3 init pose") | a2e41 (A2 ep41's level hold, randomized; 2026-11-05)
A3_PALM = os.environ.get("KITCHEN_A3_PALM", "up")                         # a2_press: only the ends with the holding palm facing up (user 2026-11-10: "only for those right palm is facing upward"; 23 of 28) | any (all 28)
A4_START = os.environ.get("KITCHEN_A4_START", "a3_end")                  # a3_end (the recorded A3 ends at the bin, plate in the right hand, in seed order; user 2026-11-10 night: "A3 done, please adjust A4 based on A3") | v1 (the stance box at the bin with ep1's carry grasp, walking mode)
A5_START = os.environ.get("KITCHEN_A5_START", "a4_end")                  # a4_end (the recorded A4 ends: right foot on the pedal, lid open, plate in the right hand, in seed order; user 2026-11-10 night: "A4 is done do A5") | v1 (ep1's swipe stance + pose, the foot placed on the pedal by IK)
A2_START = os.environ.get("KITCHEN_A2_START", "a1_end")                  # a1_end (the recorded A1 ends as they were, in seed order; user 2026-11-04: "use the same location and pose from the A1 recorded data") | a1 (the A1 envelope, validity-checked) | a1_mix (a recorded end / the band, 50:50) | v1
_BANKS: dict[str, dict] = {}
WORKTOP_TOP, WORKTOP_EDGE_X = 0.872, 0.475                                  # the kitchen's peninsula worktop in the setups' frame (kit log; worktop x 0.475..1.175)
BODY_JOINTS = tuple(f"{s}_{n}" for s in ("left", "right") for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")) + \
              ("waist_yaw", "waist_roll", "waist_pitch") + tuple(f"{s}_{n}" for s in ("left", "right") for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))


class _RobotFK:
    """Kinematics of the teleop robot alone (its MJCF, loaded once): for 29 body joints, each hand's fingertip centroid, hand
    base and elbow in the pelvis frame. Used to check a drawn start pose before it is accepted (the A1-envelope start)."""

    def __init__(self) -> None:
        import mujoco
        from simple.robots.g1_sonic_revo2 import G1SonicRevo2
        from simple.utils import get_data_dir
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(Path(get_data_dir()) / G1SonicRevo2.mjcf_path)); self.d = mujoco.MjData(self.m)
        J = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        B = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.q = np.array([self.m.jnt_qposadr[J(f"{n}_joint")] for n in BODY_JOINTS])
        self.pel = B("pelvis"); root = int(self.m.body_rootid[self.pel]); self.rootq = int(self.m.jnt_qposadr[self.m.body_jntadr[root]])
        self.pts = {s: dict(tips=[B(f"{s}_hand_{f}_tip_link") for f in ("thumb", "index", "middle", "ring", "pinky")], base=B(f"{s}_hand_base_link"),
                            wrist=B(f"{s}_wrist_yaw_link"), elbow=B(f"{s}_elbow_link")) for s in ("left", "right")}

    def hand_rot(self, body29) -> dict:
        """Per side the hand base link's rotation in the pelvis frame (the pelvis level) for 29 body joints."""
        d = self.d
        d.qpos[:] = 0.0; d.qpos[self.rootq:self.rootq + 7] = [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]; d.qpos[self.q] = np.asarray(body29, dtype=np.float64)
        self.mj.mj_kinematics(self.m, d)
        R = d.xmat[self.pel].reshape(3, 3)
        return {s: R.T @ d.xmat[P["base"]].reshape(3, 3) for s, P in self.pts.items()}

    def default29(self) -> np.ndarray:
        """The robot's default standing joint angles (the teleop's yaml), in BODY_JOINTS order."""
        if not hasattr(self, "_default29"):
            from simple.teleop.holomotion_v14.sim_node import robot_yaml
            y = robot_yaml()["default_joint_angles"]
            self._default29 = np.array([float(y[f"{n}_joint"]) for n in BODY_JOINTS], dtype=np.float64)
        return self._default29.copy()

    def points(self, body29) -> dict:
        """Per side, in the pelvis frame: `tips` (the five fingertip links), `tip` (their centroid), `base` (hand base), `wrist`
        (the forearm's end) and `elbow`."""
        d = self.d
        d.qpos[:] = 0.0; d.qpos[self.rootq:self.rootq + 7] = [0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0]; d.qpos[self.q] = np.asarray(body29, dtype=np.float64)
        self.mj.mj_kinematics(self.m, d)
        R = d.xmat[self.pel].reshape(3, 3); p0 = d.xpos[self.pel]
        rel = lambda p: R.T @ (p - p0)
        out = {}
        for s, P in self.pts.items():
            tips = [rel(d.xpos[b]) for b in P["tips"]]
            out[s] = dict(tips=tips, tip=np.mean(tips, axis=0), base=rel(d.xpos[P["base"]]), wrist=rel(d.xpos[P["wrist"]]), elbow=rel(d.xpos[P["elbow"]]))
        return out


_FK: list = []


def robot_fk() -> _RobotFK:
    if not _FK:
        _FK.append(_RobotFK())
    return _FK[0]


# A level hold (grasp_start.GRASPS with `level_deg`, A2 ep41's grasp for A3, 2026-11-05): the holding arm's jitter and the
# drawn waist are accepted only while the plate they would hold stays within this tilt from flat (robot kinematics alone,
# the legs at the default stance, the pelvis level); redrawn up to LEVEL_TRIES times, else the flattest draw is kept. With
# CARRY_JITTER as it is, the waist pitch alone (up to 0.22 rad forward) tilted the placed plate 15-20 deg and two of eight
# held starts lost the plate (a3_hold_diag, 2026-11-05); the balls roll to the wall at 2 deg, so a start is "level" at <= 5.
LEVEL_MAX_DEG = 5.0
LEVEL_TRIES = 300
LEVEL_WAIST_RANGE = ((-0.30, 0.30), (-0.05, 0.05), (-0.10, 0.22))           # WAIST_RANGE with the pitch reaching ep41's own slight lean-back (-0.067)


def held_plate_tilt_deg(side: str, key: str, arm, waist=None) -> float:
    """How far from flat (deg) a plate held with grasp_start.GRASPS[key] would be for this holding arm (7 joints) and waist
    (3, else the default), the legs at the default stance and the pelvis level."""
    from simple.teleop.holomotion_v14.grasp_start import GRASPS, _quat_mat
    fk = robot_fk(); body = fk.default29()
    if waist is not None:
        body[12:15] = waist
    k0 = 15 if side == "left" else 22
    body[k0:k0 + 7] = arm
    n = (fk.hand_rot(body)[side] @ _quat_mat(GRASPS[key]["rel_quat"]))[:, 2]
    return float(np.degrees(np.arccos(min(1.0, abs(float(n[2]))))))


def load_bank(name: str) -> dict:
    """A start bank: the end states of a recorded slice (scripts/holomotion_v14_checks/slice_end_bank.py), start_banks/<name>.json."""
    if name not in _BANKS:
        p = Path(__file__).with_name("start_banks") / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"start bank {name!r} not found at {p}")
        _BANKS[name] = json.loads(p.read_text())
    return _BANKS[name]


_ENV: dict[str, dict] = {}


def a1_envelope(bank: str = "a1_end", arm_widen: float = 0.03, waist_widen: float = 0.02) -> dict:
    """The randomization ranges a bank spans: per joint of each arm and of the waist its recorded min..max widened by a margin
    (rad), and the recorded leg stances (12 joints + pelvis height) to draw whole. The A1-envelope start draws inside these."""
    key = f"{bank}:{arm_widen}:{waist_widen}"
    if key not in _ENV:
        E = load_bank(bank)["entries"]; B = np.array([e["body"] for e in E], dtype=float)
        _ENV[key] = dict(source=bank, n=len(E),
                         arms={s: [(round(float(B[:, 15 + 7 * i + k].min() - arm_widen), 3), round(float(B[:, 15 + 7 * i + k].max() + arm_widen), 3)) for k in range(7)]
                               for i, s in enumerate(("left", "right"))},
                         waist=[(round(float(B[:, 12 + k].min() - waist_widen), 3), round(float(B[:, 12 + k].max() + waist_widen), 3)) for k in range(3)],
                         legs=[(int(e["episode"]), [float(v) for v in e["body"][:12]], float(e.get("pelvis_z", 0.78))) for e in E])
    return _ENV[key]


# Validity of a drawn start (the A1-envelope kind), on the robot's own kinematics (link origins, so the margins cover the
# forearm / hand geometry): every hand point (the five fingertips, the hand base, the wrist) over the worktop (x past the edge
# less a margin) is at least over_top_m above its top; over a plate (within radius + plate_near_m of its centre) a fingertip
# is at least over_plate_tip_m and the base / wrist at least over_plate_m above the worktop top (plates are 21-25 mm high; a
# wrist link 40 mm up went 21 mm into a plate, a fingertip centroid 75 mm up touched one by 4.5 mm in the checks); an elbow
# is at least elbow_margin_m behind the edge or above the top (the arm is 4-5 cm thick: with a 4 cm margin an elbow and an
# upper arm met the worktop's front edge by 2-3 mm when the robot yawed toward that side); the hands stay on their own side
# of the body. The plate rule is stricter than the recorded ends on purpose (36 of the 37 hover a hand low over a plate: the
# start the user refused); the torso is kept off the worktop front by the pelvis x limit (it meets the front at x 0.417).
# forearm_over_top_m: the elbow -> wrist segment where it crosses the worktop's front edge (or its lower end when both are over
# the worktop) at least that high above the top: a forearm crossing 36 mm up touched the edge by 3 mm (the arm is ~4 cm thick);
# the recorded ends cross at 33, 40, then >= 51 mm, so two of them are excluded on purpose.
# 2026-11-03 night (user, with a headset picture of a start whose hands sat on the plates: "i dont want this happen"): a hand point
# within plate_near_m 0.05 of a plate's rim must be at least 0.18 above the worktop top (15 cm above the rim: clearly above the
# plate, not resting on it; 0.15 still looked close in the headset) and a hand over the bare worktop at least 0.04 above it. The
# recorded ends hover a hand low over a plate in most episodes; that is exactly the start the user does not want, so the envelope
# leaves it out (the headset grid of drawn starts is on the atomic page, A2 card).
# 2026-11-03 22:15 (user, live start with the hands on the plates again: "the robot should not touch anything at the beginning"): the
# margins below are for the drawn pose; the arms SAG 2-5 cm once the controller holds them, so the teleop also checks the real contacts
# after the start and 1.5 s later and redraws a touching start (teleop CLI _start_touches). over_top 0.08 / forearm 0.06 / props 0.06.
A1_VALID = dict(over_top_m=0.08, edge_margin_m=0.04, elbow_margin_m=0.07, forearm_over_top_m=0.06, plate_near_m=0.05, over_plate_m=0.18, over_plate_tip_m=0.18,
                prop_clear_m=0.06, hands_apart_m=0.10, midline_m=0.05)
# The kitchen's fixed worktop props beside the plate lanes (read from the compiled model, 2026-11-03): centre x, y, footprint
# radius, top above the worktop. A hand point within radius + plate_near_m of one must be prop_clear_m above its top -- a physical
# clearance, not the plates' visual one (a hand base touched the large bowl, 19 cm tall, in a physics check).
WORKTOP_PROPS = {"bowl_large": (0.665, -0.935, 0.125, 0.19), "bowl_small": (0.665, 0.175, 0.103, 0.154), "bottle_green": (0.875, -1.055, 0.06, 0.24),
                 "carton_wrap": (0.987, -0.387, 0.15, 0.27), "carton_bags": (0.996, 0.404, 0.15, 0.30)}


def _valid_start(pts: dict, x: float, y: float, h: float, pz: float, spots: dict, rad: dict, V: dict) -> str:
    """"" when the pose `pts` (robot_fk().points) standing at (x, y) facing `h` rad with the pelvis `pz` high is valid among the
    plates at `spots` (radii `rad`), else the rule it breaks."""
    c, s_ = math.cos(h), math.sin(h)
    for side in ("left", "right"):
        for (a, l, u), tip in [(t, True) for t in pts[side]["tips"]] + [(pts[side]["base"], False), (pts[side]["wrist"], False)]:   # every fingertip, the hand base, the wrist
            wx, wy, wz = x + a * c - l * s_, y + a * s_ + l * c, pz + u
            if wx >= WORKTOP_EDGE_X - V["edge_margin_m"] and wz < WORKTOP_TOP + V["over_top_m"]:
                return f"{side} hand into the worktop"
            if wz < WORKTOP_TOP + (V["over_plate_tip_m"] if tip else V["over_plate_m"]) and any(math.hypot(sp[0] - wx, sp[1] - wy) < rad[p] + V["plate_near_m"] for p, sp in spots.items()):
                return f"{side} hand too low over a plate"
            for name, (px, py, pr, ptop) in WORKTOP_PROPS.items():
                if math.hypot(px - wx, py - wy) < pr + V["plate_near_m"] and wz < WORKTOP_TOP + ptop + V.get("prop_clear_m", 0.08):
                    return f"{side} hand over the {name}"
        a, l, u = pts[side]["elbow"]; ex, ez = x + a * c - l * s_, pz + u
        if ex >= WORKTOP_EDGE_X - V.get("elbow_margin_m", V["edge_margin_m"]) and ez < WORKTOP_TOP + V["over_top_m"]:
            return f"{side} elbow into the worktop"
        a, l, u = pts[side]["wrist"]; wx_, wz_ = x + a * c - l * s_, pz + u          # the forearm (elbow -> wrist) where it passes the worktop's front edge
        fo = V.get("forearm_over_top_m", 0.0)
        if fo and ex < WORKTOP_EDGE_X <= wx_ and ez + (WORKTOP_EDGE_X - ex) / (wx_ - ex) * (wz_ - ez) < WORKTOP_TOP + fo:
            return f"{side} forearm on the worktop edge"
        if fo and ex >= WORKTOP_EDGE_X and min(ez, wz_) < WORKTOP_TOP + fo:
            return f"{side} forearm on the worktop"
    if pts["left"]["tip"][1] - pts["right"]["tip"][1] < V["hands_apart_m"] or pts["left"]["tip"][1] < -V["midline_m"] or pts["right"]["tip"][1] > V["midline_m"]:
        return "hands crossed"
    return ""

# (low, high) offsets from the nominal layout. Distances in m, angles in deg.
RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "bottle_bin": {
        "robot_dx": (-0.08, 0.08), "robot_dy": (-0.10, 0.10), "robot_yaw": (-10.0, 10.0),
        "target_dx": (-0.04, 0.10), "target_dy": (-0.12, 0.12),         # bottle, from 11.5 cm behind the table's front edge
        "bin_dx": (-0.12, 0.12), "bin_dy": (-0.12, 0.12),
        "distractors": (0, 3),
    },
    "bowl_sink": {
        "robot_dx": (-0.08, 0.08), "robot_dy": (-0.10, 0.10), "robot_yaw": (-10.0, 10.0),
        "target_dx": (0.0, 0.10), "target_dy": (-0.15, 0.15),           # bowl, from 6 cm behind the counter's front edge
        "distractors": (0, 3),
    },
    "kitchen": {                                                               # the kitchen's own red bottle stays where the kitchen puts it (no target jitter, no distractors)
        "robot_dx": (-0.08, 0.08), "robot_dy": (-0.10, 0.10), "robot_yaw": (-10.0, 10.0),
        "target_dx": (0.0, 0.0), "target_dy": (0.0, 0.0),         # bottle, 11.5 cm behind the worktop's front edge; the kitchen's own bowl and plate are 17-22 cm to either side
        "distractors": (0, 0),                                          # the kitchen has its own props
    },
    "coffee_cart": {
        "cart_dx": (-0.05, 0.05), "cart_dy": (-0.08, 0.08),             # the cart, the box on it and the cup move together
        "robot_dx": (-0.04, 0.05), "robot_dy": (-0.05, 0.05), "robot_yaw": (-6.0, 6.0),   # relative to the moved cart
        "cup_x_span": (0.0, 0.20),                                      # cup: first 20 cm of the box top from the handle side
        "distractors": (0, 3),
    },
}

# The kits' level-3 design (make_levels.py: ROBOT_BACK 0.10, ROBOT_LEFT 0.05, TABLE_DZ 0.04, NUM_DISTRACTORS 3, and
# KITS[kit]["jitter"] / ["region"] for the target).
LV3_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "bottle_bin": {
        "robot_dx": (-0.10, 0.10), "robot_dy": (-0.05, 0.05),
        "target_dx": (-0.03, 0.03), "target_dy": (-0.08, 0.08),         # jitter "0.03,0.08"
        "table_dz": (-0.04, 0.04), "distractors": (3, 3),
    },
    "bowl_sink": {
        "robot_dx": (-0.10, 0.10), "robot_dy": (-0.05, 0.05),
        "target_dx": (0.0, 0.0), "target_dy": (-0.08, 0.08),            # jitter "0,0.08": the bowl stays on the edge line
        "table_dz": (-0.04, 0.04), "distractors": (3, 3),               # the counter and the sink move together
    },
    "kitchen": {
        "robot_dx": (-0.10, 0.10), "robot_dy": (-0.05, 0.05),
        "target_dx": (0.0, 0.0), "target_dy": (0.0, 0.0), "distractors": (0, 0),   # the worktop height is the kitchen's: no table_dz; +-5 cm keeps clear of the kitchen's plate and bowl
    },
    "coffee_cart": {
        # level 3's 20 cm back/forward span, centred 10 cm further back: standing in the v1.4 default pose the fingers
        # reach the cart handle about 5 cm ahead of the nominal start (2026-09-30 check), so start clear and walk in
        "robot_dx": (-0.20, 0.0), "robot_dy": (-0.05, 0.05),
        # region "0.475,0.65,-0.115,0.115" = the box top from its near edge + cup radius + 1.5 cm (0.475) to 17.5 cm
        # further, across the whole safe width
        "cup_x_span": (0.0, 0.175),
        "table_dz": (-0.04, 0.04), "distractors": (3, 3),               # the delivery table and the cart box together
    },
}
RANGE_SETS = {"teleop": RANGES, "lv3": LV3_RANGES}

# Slice starts (docs/long_horizon_slicing_scenes, simple.teleop.holomotion_v14.slices; S1 2026-10-30 and S2-S6 2026-11-01 from
# kitchen5 episode 1, rebuilt 2026-11-02 from kitchen7 episode 1 -- user: "use these as demo slicing, replace the kitchen5
# one"). Every start is a procedural preset drawn here (the kitchen kit's SLICE_PRESET puts it into the scene after compile)
# from kitchen7 ep1's own states at the slice boundaries (docs/_scan/kitchen7_ep1_annotation/events.npz; the robot frame of
# the full task: pelvis at the origin facing +x; the random init pose adds its own 2 cm / 3 deg):
#   path   a corridor along ep1's walk: `before_m` of arc length before its end, up to `side_m` aside, heading +-yaw_deg
#          ("to_end": facing where the walk ends instead of along the local segment)
#   box    a stance box: x, y ranges and a yaw range (robot frame, deg)
#   preset door_deg / rack_out_m ranges; plates_in_rack = (probability, {plate: slot}) choices (kitchen_task.RACK_SLOTS "near" /
#          "far" = ep1's own resting poses); plate_spots = {plate: (x range, y range)} on the worktop; bin_lid "open"
# kitchen7 ep1 (268.7 s): picks plate A at the counter from x 0.22-0.32, y -0.24..-0.18 (0-29.6 s); carries it to the bin,
# pedal + swipe from (0.66, 1.20) facing -67 deg (29.6-55.7 s); carries it to the dishwasher, passes it to the left hand at
# (-0.93, -0.97) and opens the door with the right from x -0.93..-0.59, y -1.13..-0.99 (55.7-86.9 s); steps beside the open
# door (82.9 deg), pulls the rack (43 cm) and stands the plate in it from x -0.71..-0.54, y -1.00..-0.91 (86.9-110.2 s); walks
# back, fetches plate B from (0.17, -0.67) and loads it from the far end of the door (110.2-204.3 s); pushes the rack in over
# three goes and closes the door (204.3-268.7 s). The lid stayed open (1.45 rad) after the pedal. Counter: plate_a (0.665, -0.225),
# plate_b (0.665, -0.575), small bowl (0.665, 0.175, r 0.10), worktop x 0.475..1.175, y 0.64..-1.43, plates 0.16-0.26 m across;
# door x -0.93..-1.49, top edge y -0.74 when open; the door-side box keeps the feet off the door and the hands out of the sink
# cabinet (y -1.22 put one in).
KITCHEN_S1_PATH = [(0.14, -0.18), (0.08, -0.32), (0.01, -0.49), (-0.19, -0.50), (-0.37, -0.62), (-0.55, -0.69),
                   (-0.70, -0.68), (-0.82, -0.83), (-0.90, -0.95), (-0.88, -1.05)]   # kitchen5 ep1's counter -> door approach, the 2026-10-30 S1 "open the dishwasher"; no slice uses it since 2026-11-02, kept for the record
# Since 2026-11-02 (user: "S2 should be S1 continued work, then the plate is already in the robot's hand"): every slice starts
# where the previous one ends, the plate in the hand where ep1 had it (preset plate_in_hand -> grasp_start.hold_plate: ep1's
# own grasp, the arm in its carry pose, the grip held closed until the operator squeezes). Seven slices: S1 pick, S2 carry to
# the bin + wipe, S3 carry to the dishwasher + door with the plate in hand, S4 rack out + load, S5 fetch plate B, S6 carry
# plate B + load into the half-full rack, S7 push in + close.
KITCHEN_S2_PATH = [(-0.40, 0.68), (-0.39, 0.97), (-0.20, 0.92), (0.12, 1.03), (0.32, 1.22), (0.50, 1.25), (0.66, 1.20)]   # ep1 29.6-51.5 s: the last 1.5 m of the carry to the bin, ending in the bin stance
KITCHEN_S3_PATH = [(0.65, 1.20), (0.37, 1.03), (0.10, 0.69), (-0.06, 0.20), (-0.11, -0.22), (-0.06, -0.60), (-0.25, -0.72), (-0.52, -0.52),
                   (-0.70, -0.52), (-0.89, -0.84), (-0.93, -0.97)]                     # ep1 55.7-72.0 s: bin -> the hand-over spot by the door (4.9 m walked, straightened to 3.3 m)
KITCHEN_S5_PATH = [(-0.72, -0.94), (-0.58, -0.91), (-0.31, -0.98), (0.00, -0.97), (0.07, -0.83), (0.17, -0.67)]   # ep1 110.2-127.3 s, dishwasher -> counter: its first 1.1 m (ep1 then wandered before the pick)
KITCHEN_S6_PATH = [(0.15, -0.68), (-0.10, -0.74), (-0.35, -0.82), (-0.55, -0.96), (-0.65, -1.05)]   # counter -> the door-side stance, straight (ep1's second carry wandered 9.8 m with two hand-overs)
_DOOR_SIDE = dict(x=(-0.70, -0.45), y=(-1.15, -0.98), yaw_deg=(-165.0, -120.0))    # beside the open door, facing the dishwasher (ep1 pulled the rack and pushed it from x -0.75..-0.55, y -1.21..-0.96, facing -178..-84 deg)
_COUNTER = dict(x=(-0.05, 0.05), y=(-0.75, -0.10), yaw_deg=(-15.0, 15.0))            # in front of the plates, one step back from ep1's pick stances (x 0.13-0.32, y -0.73..-0.18, facing -14..+19 deg): at x >= 0.08 SIMPLE's reset pose (arms forward at worktop height, fingertips 0.40 m ahead of the pelvis) kicks plate_a during the reset steps
_A_SPOT, _B_SPOT = ((0.62, 0.72), (-0.27, -0.18)), ((0.62, 0.72), (-0.63, -0.55))    # the authored spots +-5 / +-4.5 cm (2 cm apart at worst)
_DOOR_OPEN, _RACK_OUT = dict(door_deg=(70.0, 90.0)), dict(rack_out_m=(0.30, 0.43))   # ep1: door 81-90 deg, rack 43 cm out
_A_RIGHT, _A_LEFT, _B_RIGHT = dict(plate="plate_a", side="right"), dict(plate="plate_a", side="left"), dict(plate="plate_b", side="right")   # ep1 carried plate A in the right hand, passed it to the left for the door
SLICE_STARTS = {"kitchen": {
    "S1": dict(box=_COUNTER, preset=dict(plate_spots={"plate_a": _A_SPOT, "plate_b": _B_SPOT})),
    "S2": dict(path=KITCHEN_S2_PATH, before_m=(0.3, 1.4), side_m=0.15, yaw_deg=15.0,
               preset=dict(plate_in_hand=_A_RIGHT, plate_spots={"plate_b": _B_SPOT})),
    "S3": dict(path=KITCHEN_S3_PATH, before_m=(0.3, 3.3), side_m=0.15, yaw_deg=15.0,
               preset=dict(bin_lid="open", plate_in_hand=_A_RIGHT, plate_spots={"plate_b": _B_SPOT})),
    "S4": dict(box=_DOOR_SIDE, preset=dict(**_DOOR_OPEN, bin_lid="open", plate_in_hand=_A_LEFT, plate_spots={"plate_b": _B_SPOT})),
    "S5": dict(path=KITCHEN_S5_PATH, before_m=(0.3, 1.1), side_m=0.15, yaw_deg=15.0, heading="to_end",
               preset=dict(**_DOOR_OPEN, **_RACK_OUT, bin_lid="open", plates_in_rack=[(1.0, {"plate_a": "near"})], plate_spots={"plate_b": _B_SPOT})),
    "S6": dict(path=KITCHEN_S6_PATH, before_m=(0.3, 0.9), side_m=0.10, yaw_deg=15.0,
               preset=dict(**_DOOR_OPEN, **_RACK_OUT, bin_lid="open", plates_in_rack=[(1.0, {"plate_a": "near"})], plate_in_hand=_B_RIGHT)),
    "S7": dict(box=_DOOR_SIDE, preset=dict(**_DOOR_OPEN, **_RACK_OUT, bin_lid="open",
                                           plates_in_rack=[(0.5, {"plate_a": "near", "plate_b": "far"}), (0.5, {"plate_a": "near"})],
                                           plate_spots={"plate_b": _B_SPOT})),
}}

# Atomic slices A1-A6 (2026-11-02 evening, docs/atomic_slicing_scenes): S1 + S2 (the task up to the first plate wiped clean)
# as RoboAtom (arXiv 2610.40207) atomic skills -- a segment per dominant skill, "arm reaching, lifting, lowering, alignment and
# retraction belong to the neighboring manipulation": Move to the counter (0-3.2 s of kitchen7 ep1; whole-body locomotion is
# its own dominant skill, user 2026-11-02: "the first atomic action should be move close to the counter"), Pick (3.2-29.6 s),
# Step on the pedal (51.5-51.9 s; a foot skill, not RoboAtom's hand Press), Wipe (51.9-55.5 s) = `--slice A1..A5`, one scene per skill (user: "why there are 3 atomic
# action scene for a single pick"): the Pick scene draws its start kind (7 of 10 at the counter stance, 3 of 10 in ep1's
# hands-at-the-rim pose), both ending with the plate held up; the annotation's finer events (hand on the plate, lift onset) are
# milestones in the episode meta. The Wipe starts with one foot on the pressed pedal and KITCHEN_PEDAL_RETURN=1 (a return
# spring on the pedal, so the lid is up only while the foot presses; user: "one foot should be on the pedal so that it drives
# the lid up"; as authored the pedal has no spring, ep1 tapped it and wiped with both feet down). The grasp -> lift boundary itself (the plate 3 cm
# up between two moving hands, ep1 at 21.5 s) is a transient: placed as ep1 had it and held like the controller holds, the
# plate fell flat within 0.5 s for 49 of the 50 plate types (scratchpad grasp_sweep_pinch2.py), so it is not a start.
# Start kinds: `at_plate` stands the robot where ep1 stood relative to the drawn plate spot (the plate `ahead_m` ahead and
# `left_m` to the left in the pelvis frame -- ranges are drawn --, facing yaw_deg +- jitter; `rim_m` = the radius of the plate
# ep1 stood at, so a bigger plate puts the robot further back by the difference and the hands land at the rim); `spawn_back_m`
# spawns the robot that far behind its start for SIMPLE's reset steps (its reset pose, arms forward at worktop height, would
# sweep the plate from a stance 0.3-0.4 m from the worktop) and the CLI moves it forward before the quick start.
# Randomization (user 2026-11-02: "higher than usual"): robot location (boxes 2-4x the S-slices', at_plate ranges), the whole
# upper body when nothing is in a hand (`rand`: arms drawn in grasp_start.ARM_RANGE / the lower "low" range near furniture,
# waist, each free hand a resting curl (grasp_start.draw_hand; no closed grip or pinch: set by qpos they lock, 2026-11-03) latched until the operator squeezes), the holding arm
# jittered around ep1's carry pose, the whole-body rim pose jittered; plate spots over the free worktop (plate A -0.42..-0.08,
# plate B -0.78..-0.48 in y, x 0.61-0.76, drawn apart; the target plate of an at-the-plate start stays at x 0.60-0.66 so the
# knees stay off the drawer fronts), a random yaw per plate, both plate types drawn independently, which plate is picked /
# held (A or B) and which hand holds it (right as ep1, or left), the bin lid closed or already open for the carry.
# ep1: step-in ended at pelvis (0.24, -0.21) facing -2 deg with plate A 0.38 m ahead, 3 cm left; the clean reach ended at
# (0.18, -0.23) facing 10 deg, plate 0.37 m ahead (16.4 s); the bin stance (0.71, 1.20) facing -76 deg with the bin 0.44 m
# ahead; the lid went 0 -> 1.39 rad in 0.4 s and stayed up. The default standing pose has the knees further forward than
# ep1's, so the counter stance (A2) is 0.40-0.50 m from the plate (x >= 0.27 touched the drawer fronts).
_APPROACH = dict(x=(-0.60, -0.10), y=(-0.80, 0.20), yaw_deg=(-40.0, 40.0))         # one or two steps back from the counter stance, roughly facing it
_COUNTER_WIDE = dict(x=(-0.15, 0.05), y=(-0.85, 0.15), yaw_deg=(-35.0, 35.0))       # the task's counter start, wider; x <= 0.05 keeps the reset pose off the plates
_BIN_STANCE = dict(x=(0.55, 0.80), y=(1.20, 1.36), yaw_deg=(-95.0, -45.0))          # around ep1's stance at the pedal (0.71, 1.20) facing -76 deg; y < 1.20 put a foot on the pedal or a shin on the bin wall
_A_WIDE, _B_WIDE = ((0.61, 0.76), (-0.42, -0.08)), ((0.61, 0.76), (-0.68, -0.48))   # the free worktop between the small bowl (y 0.175, r 0.10) and the large bowl (y -0.935, r 0.12); x >= 0.61 keeps a 26 cm plate on the top
_A_NEAR, _B_NEAR = ((0.60, 0.66), (-0.42, -0.08)), ((0.60, 0.66), (-0.68, -0.48))   # the target plate of an at-the-plate start: not further than 0.66 (knees off the drawer fronts)
_EITHER = ("plate_a", "plate_b")
_HELD = dict(plate=_EITHER, side=("right", "left"))                                  # either plate, in either hand (ep1: plate A, right)
_HELD_A2E41 = dict(plate=_EITHER, side=("right", "left"), grasp="a2e41")             # A3 since 2026-11-05: A2 ep41's level hold (grasp_start.GRASPS right_a2e41 / mirrored left), either plate, either hand
_RAND_FREE = dict(arms=("left", "right"), arms_range="full", hands=("left", "right"), waist=True, plate_yaw=True)
_RAND_NEAR = dict(arms=("left", "right"), arms_range="near", hands=("left", "right"), waist=True, plate_yaw=True)
_RAND_CARRY = dict(arms="free", arms_range="low", hands="free", waist=True, carry_jitter=True, plate_yaw=True)
_RAND_CARRY_NEAR = dict(arms="free", arms_range="near", hands="free", waist=True, carry_jitter=True, plate_yaw=True)   # at the bin: the "low" range put the free hand into the closed lid
_RAND_POSE = dict(pose_jitter=True, plate_yaw=True)
SPAWN_BACK = 0.35
PEDAL_XY = (0.765, 1.024)                                                          # the pedal tread's centre in the robot frame (the kitchen is fixed; the bin centre is (0.765, 0.792))
_PICK_NEAR = dict(at_plate=dict(plate=_EITHER, ahead_m=(0.40, 0.50), left_m=(-0.12, 0.06), yaw_deg=0.0, jitter_deg=15.0), spawn_back_m=SPAWN_BACK,
                  preset=dict(plate_spots={"plate_a": _A_NEAR, "plate_b": _B_NEAR}, rand=_RAND_NEAR))
_PICK_RIM = dict(at_plate=dict(plate=_EITHER, ahead_m=0.366, left_m=-0.009, yaw_deg=10.5, jitter_m=0.02, jitter_deg=4.0, rim_m=0.1096), spawn_back_m=SPAWN_BACK,
                 preset=dict(pose="k7_f819", plate_spots={"plate_a": _A_NEAR, "plate_b": _B_NEAR}, rand=_RAND_POSE))
_RAND_WIPE = dict(hands=("left",), pose_jitter=True, plate_yaw=True)                 # ep1's swipe pose, the free hand drawn
# A2 from the recorded A1 ends (2026-11-03, 37 complete episodes of data/teleop_holomotion_v14_kitchen_a1 -> start_banks/a1_end.json;
# user: "use the A1 collected data to redesign the A2 scene"). Where the operators actually stopped is not where the 2026-11-02
# counter kind stood: pelvis x 0.26-0.42 m (6-22 cm from the worktop edge at 0.475; the counter kind stood at 0.10-0.26), y
# -0.59..-0.04, facing the counter within -10..+12 deg, the nearer plate 0.24-0.45 m ahead and up to 26 cm to either side
# (31 of 37 stood between the two plates with both inside the gate); the arms are held OUT FORWARD, nearly straight (shoulder
# pitch -0.15..0.12, elbow -0.37..-0.10 rad): the fingertips 0.27-0.38 m ahead of the pelvis and 0.13-0.39 m above it, over
# the worktop edge at chest height -- the operator's controller-holding posture, nothing like the NEAR range's bent arms --,
# the waist within +-0.1, the legs in the controller's own stance (knee 0.23-0.51, the pelvis at 0.78 m), which is why x 0.4
# is clear of the drawer fronts where the default stance's knees were not. Two start kinds, drawn half and half:
#   a1_end  one of the 37 recorded ends as it was: that episode's plate spots and plate types, the pelvis +-1 cm / +-3 deg
#           (x capped at 0.405: the torso meets the worktop front at 0.417), all 29 body joints (the arms +-0.03 rad), the
#           hands drawn (resting) -- the exact chaining;
#   a1_fit  the same stance distribution with NEW plates: pelvis x 0.26-0.40 (less 0.20 x |sin yaw|: an elbow met the worktop
#           at 0.377 / -10 deg), the picked plate (A or B) -26..+26 cm aside, facing 0 +-12 deg, plate spots over A1's own
#           lanes with a random turn, plate types from the seed; the standing pose (legs, waist, arms) of one recorded end,
#           the arms +-0.05 and the waist +-0.02 rad; the hands drawn. Redrawn while a hand of that pose lower than 0.22 m
#           above the pelvis would sit within 6 cm of a plate's rim (the hands are over the worktop: a wrist went 2 cm into a
#           plate drawn under it).
# Both spawn 0.35 m back for SIMPLE's reset steps as before. The rim kind (ep1's hands at the rim) is not an A1 end: v1 only.
_A1_END_JITTER = dict(xy_m=0.0, yaw_deg=0.0, arm_rad=0.0)                # 2026-11-04 user: the A1 end's location and pose exactly (was +-1 cm, +-3 deg, arms +-0.03, x <= 0.405)
_A1_FIT_JITTER = dict(arm_rad=0.05, waist_rad=0.02)
_PICK_A1_END = dict(bank="a1_end", jitter=_A1_END_JITTER, spawn_back_m=SPAWN_BACK, preset=dict(rand=dict(hands=("left", "right"))))
_PICK_A1_FIT = dict(at_counter=dict(plate=_EITHER, x_m=(0.26, 0.40), x_per_sin_yaw=0.20, left_m=(-0.26, 0.26), y_m=(-0.70, 0.05), yaw_deg=0.0, jitter_deg=12.0,
                                    hand_clear_m=0.06, hand_low_m=0.22),
                    bank_pose="a1_end", jitter=_A1_FIT_JITTER, spawn_back_m=SPAWN_BACK,
                    preset=dict(plate_spots={"plate_a": _A_WIDE, "plate_b": _B_WIDE}, rand=dict(hands=("left", "right"), plate_yaw=True)))
# The default since 2026-11-03 late (user: "starts from the 37 recorded A1 ends with randomization based on it, like pose,
# location, orientation etc, but it should be a valid randomization range"): ONE start kind drawn inside the envelope of the
# 37 ends -- location: pelvis x 0.26-0.40 (less 0.20 x |sin yaw|), the picked plate -26..+26 cm aside, y within -0.70..0.05;
# orientation 0 +-12 deg; pose: every arm joint and waist joint uniform in its recorded min..max (+-0.03 / +-0.02 rad), the
# legs one recorded controller stance whole (a per-joint leg draw is not a stance), the hands the resting draw; plates over
# A1's lanes with new types. Each draw is checked on the robot's kinematics (A1_VALID: hands above the worktop and clear of
# the plates, elbows behind the edge, hands uncrossed) and redrawn until valid (up to 300 tries).
_PICK_A1 = dict(at_counter=dict(plate=_EITHER, x_m=(0.26, 0.40), x_per_sin_yaw=0.20, left_m=(-0.26, 0.26), y_m=(-0.70, 0.05), yaw_deg=0.0, jitter_deg=12.0),
                a1_env="a1_end", widen=dict(arm_widen=0.03, waist_widen=0.02), valid=A1_VALID, spawn_back_m=SPAWN_BACK,
                preset=dict(plate_spots={"plate_a": _A_WIDE, "plate_b": _B_WIDE}, rand=dict(hands=("left", "right"), plate_yaw=True)))
_A2_V1 = dict(variants=[(0.7, "counter", _PICK_NEAR), (0.3, "rim", _PICK_RIM)], plate_b="independent")       # 2026-11-02: the counter stance / ep1's hands-at-the-rim pose
_A2_MIX = dict(variants=[(0.5, "a1_end", _PICK_A1_END), (0.5, "a1_fit", _PICK_A1_FIT)], plate_b="independent")  # 11-03 evening, first cut: a recorded end as it was / the band with a recorded pose
_A2_END = dict(_PICK_A1_END, plate_b="independent")                                                            # the recorded ends as they were (default since 2026-11-04; the plates where that episode had them)
_A2_A1 = dict(_PICK_A1, plate_b="independent")
_A3_A2E41 = dict(box=_COUNTER_WIDE, plate_b="independent",
                 preset=dict(plate_in_hand=_HELD_A2E41, plate_spots={"plate_a": _A_WIDE, "plate_b": _B_WIDE}, rand=_RAND_CARRY))
# A3 from the recorded A2 ends (2026-11-10, user: "use the 28 scenes to set the A3 init pose"; the 28 A2 episodes in which one
# hand pressed one side of the plate down and the other grasped the raised side and lifted it, docs/a2_press_grasp ->
# start_banks/a2_press_end.json by scripts/holomotion_v14_checks/a2_press_bank.py): the start IS that episode's state when the A2
# gate completed (the plate 10 cm or more up, held 0.5 s), in seed order: the pelvis x, y, heading, all 29 body joints, the
# holding hand's fingers and the plate in it exactly as recorded (grasp "bank": the plate pose in the hand base frame, the
# operator's hand command there as the latch target), the other plate where it lay, both plate types as recorded (never drawn,
# KITCHEN_BANK_PLATES ignored: the grasp is exact for that plate only); the free hand drawn resting (a recorded closed hand
# locks after a teleport). Only the entries recorded with plates that exist in the loaded plate set are drawn (a v1 entry needs
# --plate-set v1; lift entries also serve v2, v2 entries with both types < 50 also serve lift). Spawned 0.35 m back for SIMPLE's
# reset steps, as the A2 bank start.
# 2026-11-10 later (user: "filter one more time ... only for those right palm is facing upward"): 23 of the 28 ends hold the plate
# with the palm facing up (palm normal 9-56 deg from straight up, the plate resting on the hand); the other 5 (a2_hands ep 0, 4, 5,
# 37, 66) clasp the rim from above, palm down (114-136 deg). KITCHEN_A3_PALM=up (default) draws only the first; any = all 28.
_A3_PRESS = dict(bank="a2_press_end", held_bank=True, bank_plates="recorded", palm=A3_PALM, jitter=dict(xy_m=0.0, yaw_deg=0.0, arm_rad=0.0), spawn_back_m=SPAWN_BACK,
                 plate_b="independent", preset=dict(rand=dict(hands=("left", "right"))))

# The default since 2026-11-10 night (user: "use the final one to create A3 scenes"): the same start kind over the 30 ends of the
# final A2 dataset (the flattest 30 of the 23 palm-up press-and-grasp picks + 16 new grip picks; start_banks/a2_final_end.json,
# a2_press_bank.py --held docs/_scan/a2_final/held.json): 21 recorded with the v1 plates, 9 with v2; all palm up.
_A3_FINAL = dict(_A3_PRESS, bank="a2_final_end")
# A4 from the recorded A3 ends (2026-11-10 night, user: "A3 done, please adjust A4 based on A3"): the 29 episodes of
# data/teleop_holomotion_v14_kitchen_a3_final whose plate was still in the right hand when the A3 gate completed (at the bin,
# standing still 0.5 s; ep 4 had put the plate down) -> start_banks/a3_end.json (a2_press_bank.py --held
# docs/_scan/a3_final_ends/held.json). The start is that state exactly, in seed order, as the A3 bank start: the robot at the bin
# with its pose, the plate in the right hand, the other plate on the worktop, the recorded plate types (v1), the free hand resting;
# held by motion tracking until Y (slices A4 start_mode "hold"). No spawn-back (the old A4 started at the bin as well).
# A5 from the recorded A4 ends (2026-11-10 night, user: "A4 is done do A5"): the 30 episodes of
# data/teleop_holomotion_v14_kitchen_a4_final whose plate was still in the right hand when the A4 gate completed (the lid past 1.3 rad
# for 0.3 s; ep 12 had laid the plate on the lid) -> start_banks/a4_end.json (a2_press_bank.py --held docs/_scan/a4_final_ends/held.json
# --scene-joints k_bin_lid,k_bin_pedal). The start is that state exactly, in seed order: the robot at the bin with the RIGHT FOOT on
# the pressed pedal (all 30; ankle roll or pitch link on the tread), the pedal (0.18 rad, its stop) and the lid (1.45 rad) as recorded
# (apply_pose scene_joints; the preset's bin_lid "open" has the kit open them at compile first), the plate in the right hand, the other
# plate, recorded types (v1), the free hand resting; held by motion tracking until Y (slices A5 start_mode "hold").
_A5_A4END = dict(bank="a4_end", held_bank=True, bank_plates="recorded", jitter=dict(xy_m=0.0, yaw_deg=0.0, arm_rad=0.0),
                 plate_b="independent", preset=dict(bin_lid="open", rand=dict(hands=("left", "right"))))
_A4_A3END = dict(bank="a3_end", held_bank=True, bank_plates="recorded", jitter=dict(xy_m=0.0, yaw_deg=0.0, arm_rad=0.0),
                 plate_b="independent", preset=dict(rand=dict(hands=("left", "right"))))


def bank_fits(e: dict, set_name: str) -> bool:
    """Is a held-plate bank entry usable with the loaded plate set: recorded with the same set, with lift (whose 50 types are v2's
    types 0-49) under v2, or with v2 under lift when both of its types are below 50."""
    es, ids = e.get("plate_set"), [i for i in (e.get("plate"), e.get("plate_b")) if i is not None]
    if es == set_name or (es == "lift" and set_name == "v2"):
        return True
    return es == "v2" and set_name == "lift" and all(i < 50 for i in ids)
SLICE_STARTS["kitchen"].update({
    "A1": dict(box=_APPROACH, plate_b="independent",
               preset=dict(plate_spots={"plate_a": _A_WIDE, "plate_b": _B_WIDE}, rand=_RAND_FREE)),
    # one Pick scene (user 2026-11-02: one scene per atomic action); since 2026-11-03 its start is drawn in the envelope of the recorded A1 ends
    "A2": {"v1": _A2_V1, "a1_mix": _A2_MIX, "a1_end": _A2_END}.get(A2_START, _A2_A1),
    # A3 (2026-11-05, user: "use A2 ep41 hand pose and arm pose as reference holding, and adjust the A3 30 setup based on that,
    # include similar randomization rules"): the held plate sits as in A2 ep41's level pick (2.9 deg from flat, four pads, the
    # arm raised 0.45 m ahead) instead of ep1's steep carry; the start is held by motion tracking (slices start_mode "hold") so
    # the raised arm survives until the operator takes over; everything else as before (box, the free arm in the low range,
    # the free hand resting, the waist, the holding arm +- CARRY_JITTER, plate spots + yaw, either plate in either hand, plate
    # types from the grasp's graspable list of the loaded plate set).
    "A3": {"a2_final": _A3_FINAL, "a2_press": _A3_PRESS}.get(A3_START, _A3_A2E41),
    "A4": _A4_A3END if A4_START == "a3_end" else dict(box=_BIN_STANCE, plate_b="independent",
               preset=dict(plate_in_hand=_HELD, plate_spots={"plate_a": _A_WIDE, "plate_b": _B_WIDE}, rand=_RAND_CARRY_NEAR)),
    # Wipe: ep1's swipe stance and whole-body pose, the plate in the right hand as ep1 held it, one foot (drawn) on the pressed pedal
    "A5": _A5_A4END if A5_START == "a4_end" else dict(at_pedal=dict(ahead_m=(0.17, 0.23), left_m=(-0.03, 0.06), yaw_deg=-62.0, jitter_deg=12.0), plate_b="independent",
               preset=dict(pose="k7_f2650", plate_in_hand=dict(plate=_EITHER, side="right"), foot_on_pedal=("left", "right"), bin_lid="open",
                           plate_spots={"plate_a": _A_WIDE, "plate_b": _B_WIDE}, rand=_RAND_WIPE)),
})


def _pick(rng, v):
    """A fixed value, or one of a tuple of choices."""
    return v[int(rng.randint(len(v)))] if isinstance(v, (tuple, list)) else v


def _draw_preset(rng, spec: dict, radius_of=None) -> dict:
    """One episode's slice preset from its ranges (exact values, 0.1 mm / 0.1 deg). `radius_of(plate, held)` gives a plate's
    radius for the spot spacing (the plate types follow from the seed and the held plate)."""
    from simple.teleop.holomotion_v14.grasp_start import ARM_JOINTS, CARRY_JITTER, GRASPS, POSES, POSE_JITTER, WAIST_RANGE, draw_arm, draw_hand, grasp_key
    out = {}
    R = spec.get("rand") or {}
    if "door_deg" in spec:
        out["door_deg"] = round(float(rng.uniform(*spec["door_deg"])), 1)
    if "rack_out_m" in spec:
        out["rack_out_m"] = round(float(rng.uniform(*spec["rack_out_m"])), 3)
    if "plates_in_rack" in spec:
        w = np.array([p for p, _ in spec["plates_in_rack"]], dtype=float)
        out["plates_in_rack"] = dict(spec["plates_in_rack"][int(rng.choice(len(w), p=w / w.sum()))][1])
    if "pose" in spec:                                                      # fixed: ep1's whole-body pose at a frame (grasp_start.POSES, apply_pose)
        out["pose"] = spec["pose"]
    if "plate_in_hand" in spec:                                             # {plate, side[, grasp]}: plate / side may be choices (grasp_start.hold_plate puts it there after the robot stands)
        H = dict(spec["plate_in_hand"]); H["plate"] = _pick(rng, H["plate"]); H["side"] = _pick(rng, H["side"])
        if spec.get("pose"):                                                # the held arm is the pose's arm, not the carry grasp's
            base = POSES[spec["pose"]]["body"]; k0 = 15 if H["side"] == "left" else 22
            H["arm"] = [round(float(q + rng.uniform(-POSE_JITTER, POSE_JITTER)), 3) for q in base[k0:k0 + 7]] if R.get("pose_jitter") else list(base[k0:k0 + 7])
        if R.get("carry_jitter"):                                           # the holding arm around the grasp's recorded arm pose (ep1's carry / hold, ep41's level hold; half the jitter for ep1's left hold)
            from simple.teleop.holomotion_v14.grasp_start import carry_jitter_scale
            key = grasp_key(H); G = GRASPS[key]; base = G["arm"]; k = carry_jitter_scale(key)
            level = G.get("level_deg") is not None                          # a level hold: the jitter (and the waist, drawn here) must keep the plate flat (LEVEL_MAX_DEG, FK)
            best = None; tilt = None; waist = None
            for tries in range(1, LEVEL_TRIES + 1):
                arm = [round(float(q + rng.uniform(-j * k, j * k)), 3) for q, j in zip(base, CARRY_JITTER)]
                if not level:
                    break
                waist = [round(float(rng.uniform(lo, hi)), 3) for lo, hi in LEVEL_WAIST_RANGE] if R.get("waist") else None
                tilt = held_plate_tilt_deg(H["side"], key, arm, waist)
                if best is None or tilt < best[0]:
                    best = (tilt, arm, waist)
                if tilt <= LEVEL_MAX_DEG:
                    break
            else:
                tilt, arm, waist = best                                     # nothing within the limit in LEVEL_TRIES draws: the flattest one
            H["arm"] = arm
            if level:
                H["level_tilt_deg"] = round(float(tilt), 1); H["level_tries"] = int(tries)
                if waist is not None:
                    out["waist"] = waist
        out["plate_in_hand"] = H
    held = out.get("plate_in_hand", {})
    if "plate_spots" in spec:                                               # a plate standing in the rack or held in the hand keeps no worktop spot
        # (a pinched plate keeps its spot drawn: the at-the-plate stance is computed from it, the plate itself then goes into the hand)
        spots = {}
        for p, (xr, yr) in spec["plate_spots"].items():
            if p in out.get("plates_in_rack", {}) or (p == held.get("plate") and held.get("grasp") != "pinch"):
                continue
            xy = [_r(rng.uniform(*xr)), _r(rng.uniform(*yr))]
            if radius_of is not None:                                       # drawn apart: at least both radii + 2 cm between the centres
                for _ in range(40):
                    if all(np.hypot(xy[0] - q[0], xy[1] - q[1]) >= radius_of(p, held) + radius_of(o, held) + 0.02 for o, q in spots.items()):
                        break
                    xy = [_r(rng.uniform(*xr)), _r(rng.uniform(*yr))]
            if R.get("plate_yaw"):
                xy.append(round(float(rng.uniform(-180.0, 180.0)), 1))
            spots[p] = xy
        out["plate_spots"] = spots
    if "foot_on_pedal" in spec:                                             # which foot stands on the pressed pedal (grasp_start.foot_on_pedal after the pose)
        out["foot_on_pedal"] = _pick(rng, spec["foot_on_pedal"])
    if "bin_lid" in spec:                                                   # fixed, or (probability, state) choices
        if isinstance(spec["bin_lid"], str):
            out["bin_lid"] = spec["bin_lid"]
        else:
            w = np.array([p for p, _ in spec["bin_lid"]], dtype=float)
            out["bin_lid"] = spec["bin_lid"][int(rng.choice(len(w), p=w / w.sum()))][1]
    # the upper body when nothing is in that hand (user 2026-11-02: higher start randomization): arms, waist, finger poses
    free = [sd for sd in ("left", "right") if sd != held.get("side")]
    arms = R.get("arms"); arms = free if arms == "free" else (arms or ())
    if arms:
        out["arms"] = {sd: draw_arm(rng, sd, R.get("arms_range", "full")) for sd in arms if sd in free}
    if R.get("waist") and "waist" not in out:                               # (a level hold drew its waist with the holding arm above)
        out["waist"] = [round(float(rng.uniform(lo, hi)), 3) for lo, hi in WAIST_RANGE]
    hands = R.get("hands"); hands = free if hands == "free" else (hands or ())
    if hands:
        out["hands"] = {sd: draw_hand(rng) for sd in hands if sd in free}
    if R.get("pose_jitter") and out.get("pose"):                            # the whole-body pose's arms, jittered
        base = POSES[out["pose"]]["body"]
        out["arms"] = {sd: [round(float(q + rng.uniform(-POSE_JITTER, POSE_JITTER)), 3) for q in base[15 + 7 * k:22 + 7 * k]] for k, sd in enumerate(("left", "right"))}
    return out


def _bank_pose(rng, e: dict, bank: str, arm_rad: float = 0.0, waist_rad: float = 0.0) -> dict:
    """A recorded end state as an inline whole-body pose for grasp_start.apply_pose (a dict instead of a POSES name): the 29
    body joints as recorded, the 14 arm joints and the 3 waist joints jittered. The fingers are not part of it: the free hands
    are drawn separately (a recorded closed hand locks after a teleport, see the module header)."""
    body = [float(v) for v in e["body"]]
    for i in range(15, 29):
        body[i] = round(body[i] + float(rng.uniform(-arm_rad, arm_rad)), 4)
    for i in range(12, 15):
        body[i] = round(body[i] + float(rng.uniform(-waist_rad, waist_rad)), 4)
    out = dict(body=body, source=f"{bank} ep {e['episode']} frame {e['frame']} ({e['dataset']})")
    if e.get("root_quat") is not None:                                     # a held-plate end (A3 from the A2 ends): the pelvis as recorded (leaning 1-5 deg), not upright
        out["root_quat"] = list(e["root_quat"])
    if e.get("scene_joints"):                                              # scene joints as recorded (A5 from the A4 ends: the pressed pedal and the open lid)
        out["scene_joints"] = dict(e["scene_joints"])
    return out


def _hands_clear(e: dict, x: float, y: float, h: float, spots: dict, rad: dict, clear_m: float, low_m: float) -> bool:
    """No hand of the recorded body pose `e` over a plate at rim height when the pelvis stands at (x, y) facing `h` rad: every
    hand point (fingertip centroid, hand base; bank `hands_pelvis`) lower than `low_m` above the pelvis keeps a horizontal
    distance of at least that plate's radius + `clear_m` from every plate centre."""
    H = e.get("hands_pelvis")
    if not H:
        return True
    c, s_ = math.cos(h), math.sin(h)
    for side in ("left", "right"):
        for key in ("tip", "base"):
            a, l, u = H[side][key]
            if u >= low_m:
                continue
            px, py = x + a * c - l * s_, y + a * s_ + l * c
            if any(math.hypot(sp[0] - px, sp[1] - py) < rad[p] + clear_m for p, sp in spots.items()):
                return False
    return True


def _point_before_end(path, dist: float):
    """The point `dist` m of arc length before the polyline's end, and the path heading there (deg)."""
    pts = [np.asarray(p, dtype=float) for p in path]
    for i in range(len(pts) - 1, 0, -1):
        a, b = pts[i - 1], pts[i]
        seg = b - a
        length = float(np.linalg.norm(seg))
        if dist <= length or i == 1:
            u = 1.0 - min(dist, length) / length if length > 0 else 0.0
            return a + u * seg, math.degrees(math.atan2(seg[1], seg[0]))
        dist -= length
    raise ValueError("empty path")

_CART_X_KEYS = ("handle_x", "deck_near_x", "deck_far_x", "cart_cx", "box_near_x", "box_far_x", "cup_x")
_CART_Y_KEYS = ("cart_cy", "box_right_y", "box_left_y", "cup_y")


def _yaw_quat(yaw_deg: float) -> list[float]:
    h = math.radians(yaw_deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]                            # w, x, y, z


def _r(v: float) -> float:
    return round(float(v), 4)                                               # 0.1 mm: exact, readable values in the metadata


class SceneSetups:
    """Draws and applies setups for one live scene module + task."""

    def __init__(self, scene: str, module, task, max_distractors: int | None = None, ranges: str = "teleop",
                 slice: str = "") -> None:
        if ranges not in RANGE_SETS:
            raise ValueError(f"unknown ranges {ranges!r}; one of {sorted(RANGE_SETS)}")
        if scene not in RANGE_SETS[ranges]:
            raise ValueError(f"no randomization ranges for scene {scene!r}")
        if slice and slice not in SLICE_STARTS.get(scene, {}):
            raise ValueError(f"no slice start {slice!r} for scene {scene!r}; built: {sorted(SLICE_STARTS.get(scene, {}))}")
        self.scene, self.mod, self.task, self.range_set, self.slice = scene, module, task, ranges, slice
        self.ranges = deepcopy(RANGE_SETS[ranges][scene])
        if max_distractors is not None:
            lo, hi = self.ranges["distractors"]
            if hi > 0:                                                      # a scene whose range is (0, 0) wants none (kitchen: its own props are the clutter)
                self.ranges["distractors"] = (min(lo, max_distractors), max_distractors)
        # the layout as imported (start distance applied) and the task's own regions, kept on the module / task the
        # first time: a later SceneSetups on the same objects must not take an applied setup for the nominal one
        self.nominal_L = module.__dict__.setdefault("_scene_setup_nominal_L", dict(module.L))
        spatial = task.dr.get_randomizer("spatial")
        nominal = task.__dict__.setdefault("_scene_setup_nominal", dict(
            regions={k: deepcopy(getattr(spatial.cfg, k)) for k in ("robot_region", "robot_orientation_region", "target_region")},
            distractors=int(task.dr.get_randomizer("distractors").cfg.number_of_distractors),
            table_height=deepcopy(task.dr.get_randomizer("scene").cfg.table_height)))
        self._nominal_regions, self._nominal_distractors = nominal["regions"], nominal["distractors"]
        self._nominal_table_height = nominal["table_height"]

    # ------------------------------------------------------------------ sampling
    def nominal(self) -> dict[str, Any]:
        """The scene exactly as the module lays it out (no randomization)."""
        L = self.nominal_L
        tx, ty = {"bottle_bin": ("bottle_x", "bottle_y"), "bowl_sink": ("bowl_x", "bowl_y"),
                  "coffee_cart": ("cup_x", "cup_y"), "kitchen": ("bottle_x", "bottle_y")}[self.scene]
        return dict(version=SETUP_VERSION, scene=self.scene, seed=None, randomized=False,
                    robot=dict(x=0.0, y=0.0, yaw_deg=0.0), target_xy=[_r(L[tx]), _r(L[ty])], L={},
                    distractors=self._nominal_distractors)

    def sample(self, seed: int) -> dict[str, Any]:
        rng = np.random.RandomState(seed)
        R, L = self.ranges, self.nominal_L

        def u(k: str) -> float:
            return float(rng.uniform(*R.get(k, (0.0, 0.0))))                # a quantity the range set leaves out stays nominal

        s = dict(version=SETUP_VERSION, scene=self.scene, seed=int(seed), randomized=True, ranges=self.range_set, L={})
        if self.scene == "bottle_bin":
            s["robot"] = dict(x=_r(u("robot_dx")), y=_r(u("robot_dy")), yaw_deg=_r(u("robot_yaw")))
            s["target_xy"] = [_r(L["bottle_x"] + u("target_dx")), _r(L["bottle_y"] + u("target_dy"))]
            bin_dx, bin_dy = u("bin_dx"), u("bin_dy")
            if "bin_dx" in R or "bin_dy" in R:
                s["L"] = {"bin_x": _r(L["bin_x"] + bin_dx), "bin_y": _r(L["bin_y"] + bin_dy)}
        elif self.scene == "bowl_sink":
            s["robot"] = dict(x=_r(u("robot_dx")), y=_r(u("robot_dy")), yaw_deg=_r(u("robot_yaw")))
            s["target_xy"] = [_r(L["bowl_x"] + u("target_dx")), _r(L["bowl_y"] + u("target_dy"))]
        elif self.scene == "kitchen":
            s["robot"] = dict(x=_r(u("robot_dx")), y=_r(u("robot_dy")), yaw_deg=_r(u("robot_yaw")))
            st = SLICE_STARTS["kitchen"].get(self.slice) if self.slice else None
            variant = plate_ids = None
            if st is not None and "variants" in st:                         # one scene, several start kinds (A2 Pick: at the counter / hands at the rim)
                w = np.array([v[0] for v in st["variants"]], dtype=float)
                _, variant, spec = st["variants"][int(rng.choice(len(w), p=w / w.sum()))]
                st = dict(spec, plate_b=st.get("plate_b"))
            if st is not None:                                              # a slice's start instead of the task's counter start
                radius_of = (lambda plate, held: self._plate_radius(seed, held, plate, st.get("plate_b"))) if st.get("plate_b") else None
                if "at_pedal" in st:                                        # where ep1 stood at the pedal: the tread ahead / left in the pelvis frame
                    ap = st["at_pedal"]
                    yaw = ap["yaw_deg"] + float(rng.uniform(-ap["jitter_deg"], ap["jitter_deg"])); h = math.radians(yaw)
                    ahead = float(rng.uniform(*ap["ahead_m"])); left = float(rng.uniform(*ap["left_m"]))
                    p = np.asarray(PEDAL_XY, dtype=float) - (ahead * np.array([math.cos(h), math.sin(h)]) + left * np.array([-math.sin(h), math.cos(h)]))
                    s["robot"] = dict(x=_r(p[0]), y=_r(p[1]), yaw_deg=_r(yaw))
                    s["start"] = dict(at_pedal=True, ahead_m=_r(ahead), left_m=_r(left))
                elif "bank" in st:                                            # one recorded end of the previous slice, as it was (A2 from the A1 ends, 2026-11-03)
                    E = load_bank(st["bank"])["entries"]
                    if st.get("held_bank"):                                     # A3 from the A2 ends: only the ends recorded with plates the loaded set has
                        from simple.teleop.holomotion_v14.grasp_start import plate_set_name
                        sn = plate_set_name(); E = [x for x in E if bank_fits(x, sn)]
                        if st.get("palm") == "up":                             # the holding palm facing up (angle from straight up < 90 deg)
                            E = [x for x in E if (x.get("held") or {}).get("palm_up_deg", 0.0) < 90.0]
                        if not E:
                            raise ValueError(f"start bank {st['bank']}: no entry was recorded with plates of the loaded plate set {sn!r} (use --plate-set v1 or v2)")
                    e = E[int(seed) % len(E)]; jt = st.get("jitter") or {}   # in seed order: consecutive setups walk through every recorded end
                    s["preset"] = _draw_preset(rng, st.get("preset", {}), None)
                    s["preset"]["plate_spots"] = deepcopy(e["plate_spots"])
                    s["preset"]["pose"] = _bank_pose(rng, e, st["bank"], jt.get("arm_rad", 0.0))
                    j = float(jt.get("xy_m", 0.0)); jy = float(jt.get("yaw_deg", 0.0))
                    s["robot"] = dict(x=_r(min(e["robot"]["x"] + rng.uniform(-j, j), jt.get("x_max", np.inf))), y=_r(e["robot"]["y"] + rng.uniform(-j, j)),
                                      yaw_deg=_r(e["robot"]["yaw_deg"] + rng.uniform(-jy, jy)))
                    s["start"] = dict(bank=st["bank"], bank_episode=int(e["episode"]), plate=e["target"],
                                      ahead_m=_r(e["rel"][e["target"]]["ahead"]), left_m=_r(e["rel"][e["target"]]["left"]))
                    plate_ids = (e.get("plate"), e.get("plate_b"))
                    if e.get("held"):                                         # the plate in the hand as recorded (grasp_start.hold_plate, inline grasp)
                        Hh = e["held"]; side = Hh["side"]; k0 = 15 if side == "left" else 22
                        s["preset"]["plate_in_hand"] = dict(plate=Hh["plate"], side=side, grasp="bank", rel_pos=list(Hh["rel_pos"]), rel_quat=list(Hh["rel_quat"]),
                                                            fingers=list(Hh["fingers"]), cmd=list(Hh["cmd"]), r_ref=Hh["r_ref"], arm=list(s["preset"]["pose"]["body"][k0:k0 + 7]),
                                                            source=f"{e.get('dataset_short', e.get('dataset'))} ep {e['episode']} frame {e['frame']}, tilt {Hh.get('tilt_deg')} deg")
                        (s["preset"].get("hands") or {}).pop(side, None)       # the holding hand is the grasp's, not a drawn one
                        s["preset"]["plate_spots"] = {p: v for p, v in s["preset"]["plate_spots"].items() if p != Hh["plate"]}
                        src = e.get("source")
                        here = (e.get("dataset_short", e.get("dataset")), e["episode"])
                        s["start"]["bank_source"] = f"{here[0]} ep {here[1]}" + (f" ({src['dataset']} ep {src['episode']})" if src and (src.get("dataset"), src.get("episode")) != here else "")
                        s["start"]["bank_plates"] = st.get("bank_plates", "recorded")
                elif "at_counter" in st:                                      # at the counter as the A1 operators stood: pelvis x in a band, a plate `left` aside, facing the counter
                    from simple.teleop.holomotion_v14.grasp_start import R_REF
                    ac = st["at_counter"]
                    s["preset"] = _draw_preset(rng, st.get("preset", {}), radius_of)
                    target = _pick(rng, ac["plate"]); spots = s["preset"]["plate_spots"]; spot = spots[target]
                    rad = {p: (radius_of(p, None) if radius_of else R_REF) for p in spots}

                    def stance():                                            # the pelvis: x in the band (less the yaw term), the picked plate `left` aside, facing the counter
                        yaw = ac["yaw_deg"] + float(rng.uniform(-ac["jitter_deg"], ac["jitter_deg"])); h = math.radians(yaw)
                        x_hi = ac["x_m"][1] - float(ac.get("x_per_sin_yaw", 0.0)) * abs(math.sin(h))
                        x = float(rng.uniform(ac["x_m"][0], x_hi)); left = float(rng.uniform(*ac["left_m"]))
                        ahead = (spot[0] - x + left * math.sin(h)) / math.cos(h)   # plate = pelvis + ahead * forward + left * leftward, with the pelvis at x
                        return yaw, h, x, left, ahead, spot[1] - ahead * math.sin(h) - left * math.cos(h)

                    if st.get("a1_env"):                                     # the A1 envelope: arms + waist per joint in the recorded span, legs a recorded stance, checked on the kinematics
                        env = a1_envelope(st["a1_env"], **(st.get("widen") or {})); fk = robot_fk(); V = dict(A1_VALID, **(st.get("valid") or {}))
                        reason, total = "", 0
                        for attempt in range(6):                             # a plate layout that defeats the stance loop is redrawn
                            if attempt:
                                s["preset"] = _draw_preset(rng, st.get("preset", {}), radius_of); spots = s["preset"]["plate_spots"]; spot = spots[target]
                            for tries in range(1, 101):                      # 1 ms a try
                                total += 1
                                yaw, h, x, left, ahead, yy = stance()
                                if not (ac["y_m"][0] <= yy <= ac["y_m"][1]):
                                    reason = "pelvis y out of the band"
                                    continue
                                ep, legs, pz = env["legs"][int(rng.randint(env["n"]))]
                                body = [round(v, 4) for v in legs] + [round(float(rng.uniform(lo, hi)), 4) for lo, hi in env["waist"]] \
                                    + [round(float(rng.uniform(lo, hi)), 4) for sd in ("left", "right") for lo, hi in env["arms"][sd]]
                                reason = _valid_start(fk.points(body), x, yy, h, pz, spots, rad, V)
                                if not reason:
                                    break
                            if not reason:
                                break
                        s["robot"] = dict(x=_r(x), y=_r(yy), yaw_deg=_r(yaw))
                        s["start"] = dict(at_counter=target, plate=target, ahead_m=_r(ahead), left_m=_r(left), env=st["a1_env"], legs_episode=ep, tries=total, plate_draws=attempt + 1)
                        if reason:
                            s["start"]["unchecked"] = reason                 # no valid draw in 6 plate layouts x 100 stances: the last one, flagged
                        s["preset"]["pose"] = dict(body=body, source=f"the A1 envelope ({st['a1_env']}: arms and waist drawn in the recorded span, legs of ep {ep})")
                    else:                                                    # a1_mix's fitted kind: the stance band with one recorded body pose, no low hand over a plate
                        E = load_bank(st["bank_pose"])["entries"] if st.get("bank_pose") else None
                        e = None
                        for _ in range(60):
                            yaw, h, x, left, ahead, yy = stance()
                            e = E[int(rng.randint(len(E)))] if E else None
                            if not (ac["y_m"][0] <= yy <= ac["y_m"][1]):
                                continue
                            if e is None or _hands_clear(e, x, yy, h, spots, rad, float(ac.get("hand_clear_m", 0.0)), float(ac.get("hand_low_m", 0.0))):
                                break
                        s["robot"] = dict(x=_r(x), y=_r(yy), yaw_deg=_r(yaw))
                        s["start"] = dict(at_counter=target, plate=target, ahead_m=_r(ahead), left_m=_r(left))
                        if e is not None:                                    # the standing pose (legs, waist, arms) of that recorded end, jittered
                            jt = st.get("jitter") or {}
                            s["preset"]["pose"] = _bank_pose(rng, e, st["bank_pose"], jt.get("arm_rad", 0.0), jt.get("waist_rad", 0.0))
                            s["start"]["pose_episode"] = int(e["episode"])
                elif "at_plate" in st:                                        # where ep1 stood relative to the plate: the preset (plate spot) first
                    ap = st["at_plate"]
                    s["preset"] = _draw_preset(rng, st.get("preset", {}), radius_of)
                    target = _pick(rng, ap["plate"])
                    spot = s["preset"]["plate_spots"][target]
                    yaw = ap["yaw_deg"] + float(rng.uniform(-ap["jitter_deg"], ap["jitter_deg"]))
                    h = math.radians(yaw)
                    ahead = float(rng.uniform(*ap["ahead_m"])) if isinstance(ap["ahead_m"], (tuple, list)) else ap["ahead_m"]
                    left = float(rng.uniform(*ap["left_m"])) if isinstance(ap["left_m"], (tuple, list)) else ap["left_m"]
                    if ap.get("rim_m"):                                     # the hands at the rim: a bigger plate than ep1's puts the robot further back
                        ahead += self._plate_radius(seed, s["preset"].get("plate_in_hand"), target, st.get("plate_b")) - ap["rim_m"]
                    p = np.asarray(spot[:2], dtype=float) - (ahead * np.array([math.cos(h), math.sin(h)]) + left * np.array([-math.sin(h), math.cos(h)]))
                    j = ap.get("jitter_m", 0.0)
                    p = p + rng.uniform(-j, j, size=2)
                    s["robot"] = dict(x=_r(p[0]), y=_r(p[1]), yaw_deg=_r(yaw))
                    s["start"] = dict(at_plate=target, ahead_m=_r(ahead), left_m=_r(left))
                elif "path" in st:
                    before = float(rng.uniform(*st["before_m"]))
                    side = float(rng.uniform(-st["side_m"], st["side_m"]))
                    p, heading = _point_before_end(st["path"], before)
                    if st.get("heading") == "to_end":                       # face where the walk ends instead of along the local segment
                        end = np.asarray(st["path"][-1], dtype=float)
                        heading = math.degrees(math.atan2(end[1] - p[1], end[0] - p[0]))
                    h = math.radians(heading)
                    p = p + side * np.array([-math.sin(h), math.cos(h)])  # +side = to the left of the path
                    s["robot"] = dict(x=_r(p[0]), y=_r(p[1]), yaw_deg=_r(heading + rng.uniform(-st["yaw_deg"], st["yaw_deg"])))
                    s["start"] = dict(path_before_m=_r(before), side_m=_r(side))
                else:
                    b = st["box"]
                    s["robot"] = dict(x=_r(rng.uniform(*b["x"])), y=_r(rng.uniform(*b["y"])), yaw_deg=_r(rng.uniform(*b["yaw_deg"])))
                    s["start"] = dict(box=True)
                if st.get("spawn_back_m"):                                  # spawned that far behind the start for SIMPLE's reset steps, moved forward by the CLI
                    s["start"]["spawn_back_m"] = float(st["spawn_back_m"])
                s["slice"] = self.slice
                if variant:
                    s["start"]["variant"] = variant
                if st.get("preset") and "preset" not in s:
                    s["preset"] = _draw_preset(rng, st["preset"], radius_of)
            s["target_xy"] = [_r(L["bottle_x"] + u("target_dx")), _r(L["bottle_y"] + u("target_dy"))]
            n_pl = int(getattr(self.mod, "N_PLATES", 0) or 0)                 # the procedural plate set (2026-10-26): a new plate type every rollout
            if n_pl:
                held = (s.get("preset") or {}).get("plate_in_hand")
                a_id, b_id = self._plate_ids(seed, held, (st or {}).get("plate_b"))
                s["plate"] = a_id
                if b_id is not None:                                        # plate B's type drawn on its own (atomic slices); else the kit's id + 25
                    s["plate_b"] = b_id
                if plate_ids is not None and plate_ids[0] is not None:     # a bank start: the plate types of the recorded episode
                    if os.environ.get("KITCHEN_BANK_PLATES", "recorded") == "draw" and (st or {}).get("bank_plates") != "recorded":   # ... or drawn from the loaded set (2026-11-06: --plate-set v2 sets it, so the
                        s["start"]["plates"] = "drawn"                       # pasta plates reach the A2 starts; the recorded location / pose / spots stay)
                    else:
                        s["plate"] = int(plate_ids[0])
                        if plate_ids[1] is not None:
                            s["plate_b"] = int(plate_ids[1])
        else:                                                               # coffee_cart
            cdx, cdy = u("cart_dx"), u("cart_dy")
            over = {k: L[k] + cdx for k in _CART_X_KEYS}
            over.update({k: L[k] + cdy for k in _CART_Y_KEYS})
            # the cup: on the box top minus its radius and a 1.5 cm margin (as coffee_cart_task's safe region), in the
            # first cup_x_span of the top from the handle side, anywhere across it
            m = self.mod.BS.CUP_R_TOP + 0.015
            x0, x1 = over["box_near_x"] + m, over["box_far_x"] - m
            y0, y1 = over["box_right_y"] + m, over["box_left_y"] - m
            lo, hi = R["cup_x_span"]
            over["cup_x"] = x0 + float(rng.uniform(lo, min(hi, x1 - x0)))
            over["cup_y"] = float(rng.uniform(y0, y1))
            s["L"] = {k: _r(v) for k, v in over.items()}
            s["robot"] = dict(x=_r(cdx + u("robot_dx")), y=_r(cdy + u("robot_dy")), yaw_deg=_r(u("robot_yaw")))
            s["target_xy"] = [s["L"]["cup_x"], s["L"]["cup_y"]]
        lo, hi = R["distractors"]
        s["distractors"] = int(rng.randint(int(lo), int(hi) + 1))
        if "table_dz" in R:                                                 # drawn last: the teleop ranges' draws are unchanged
            s["table_dz"] = round(float(rng.uniform(*R["table_dz"])), 3) + 0.0   # whole mm, as the kits name their variants
        return s

    def _plate_ids(self, seed: int, held: dict | None, plate_b_mode: str | None) -> tuple[int, int | None]:
        """(plate_a id, plate_b id or None) for a seed: the held plate's type comes from its grasp's graspable list
        (grasp_start.GRASPABLE); plate B's type is drawn on its own when the slice says `plate_b="independent"`, else None
        (the kit uses id + 25)."""
        from simple.teleop.holomotion_v14.grasp_start import PLATE_B_OFFSET, graspable_ids, grasp_key
        n_pl = int(getattr(self.mod, "N_PLATES", 0) or 0)
        if not n_pl:
            return 0, None
        a = int(seed) % n_pl
        b = int(np.random.RandomState((int(seed) * 2654435761 + 97) % 2 ** 32).randint(n_pl)) if plate_b_mode == "independent" else None
        if held:
            ok = graspable_ids(grasp_key(held)) or list(range(n_pl))        # the grasp's list for the loaded plate set
            k = ok[int(seed) % len(ok)]
            if held["plate"] == "plate_a":
                a = k % n_pl
            elif b is None:
                a = (k - PLATE_B_OFFSET) % n_pl
            else:
                b = k % n_pl
        return a, b

    def _plate_radius(self, seed: int, held: dict | None, plate: str, plate_b_mode: str | None) -> float:
        """Half the width of the plate type this seed draws for `plate` (the kitchen plate manifest)."""
        from simple.teleop.holomotion_v14.grasp_start import PLATE_B_OFFSET, R_REF
        man = getattr(self.mod, "PLATE_MANIFEST", None); n_pl = int(getattr(self.mod, "N_PLATES", 0) or 0)
        if not man or not n_pl:
            return R_REF
        a, b = self._plate_ids(seed, held, plate_b_mode)
        k = a if plate == "plate_a" else (b if b is not None else (a + PLATE_B_OFFSET) % n_pl)
        return float(man["plates"][k]["diameter_mm"]) / 2000.0

    # ------------------------------------------------------------------ applying
    def apply(self, setup: dict[str, Any]) -> None:
        """Write a setup into the live module/task; call right before env.reset()."""
        from simple.dr.types import Box
        if setup.get("scene") != self.scene:
            raise ValueError(f"setup is for {setup.get('scene')!r}, the scene is {self.scene!r}")
        L = self.mod.L
        L.clear()
        L.update(self.nominal_L)
        L.update(setup.get("L", {}))
        spatial = self.task.dr.get_randomizer("spatial")
        for k, v in self._nominal_regions.items():
            setattr(spatial.cfg, k, deepcopy(v))
        if setup.get("randomized"):
            r = setup["robot"]
            back = float((setup.get("start") or {}).get("spawn_back_m", 0.0) or 0.0)
            dx, dy = spawn_shift(setup) if back else (0.0, 0.0)             # spawn behind the start; the CLI moves the robot forward by (dx, dy) before the quick start
            spatial.cfg.robot_region = Box(low=[r["x"] - dx, r["y"] - dy, 0.0], high=[r["x"] - dx, r["y"] - dy, 0.0])
            q = _yaw_quat(r["yaw_deg"])
            spatial.cfg.robot_orientation_region = Box(low=list(q), high=list(q))
            tx, ty = setup["target_xy"]
            spatial.cfg.target_region = Box(low=[tx, ty], high=[tx, ty])
        self.task.dr.get_randomizer("distractors").cfg.number_of_distractors = int(setup.get("distractors", 0))
        # the table / counter / cart-box height: the scene randomizer builds the table at this top height, SIMPLE places
        # the objects on it, and the kit moves its own furniture with it (level-3 mesh variants, per whole mm)
        th = deepcopy(self._nominal_table_height)
        dz = float(setup.get("table_dz", 0.0))
        if dz != 0.0:
            h = float(th.middle()) + dz
            th = Box(low=h, high=h)
        self.task.dr.get_randomizer("scene").cfg.table_height = th
        if hasattr(self.mod, "PLATE_ID"):                                   # kitchen: which procedural plate this setup uses (None = the authored paper plates)
            self.mod.PLATE_ID = setup.get("plate")
        if hasattr(self.mod, "PLATE_ID_B"):                                 # kitchen: plate B's own type (atomic slices), None = plate + 25
            self.mod.PLATE_ID_B = setup.get("plate_b")
        if hasattr(self.mod, "SLICE_PRESET"):                               # kitchen: the slice's start preset (door, rack, plates), None for a plain setup
            self.mod.SLICE_PRESET = deepcopy(setup.get("preset")) or None
        if setup.get("seed") is not None:
            np.random.seed(int(setup["seed"]) % (2 ** 32))                 # SIMPLE's DR draws from both
            random.seed(int(setup["seed"]))

    @staticmethod
    def describe(setup: dict[str, Any]) -> str:
        from simple.teleop.holomotion_v14.grasp_start import hand_curl, hand_kind
        r = setup.get("robot", {})
        parts = [f"robot ({r.get('x', 0):+.3f}, {r.get('y', 0):+.3f}) {r.get('yaw_deg', 0):+.1f} deg",
                 f"target ({setup['target_xy'][0]:.3f}, {setup['target_xy'][1]:+.3f})"]
        if setup.get("plate") is not None:
            parts.append(f"plate {setup['plate']:02d}" + (f" + {setup['plate_b']:02d}" if setup.get("plate_b") is not None else ""))
        if setup.get("slice"):
            st = setup.get("start", {})
            if "path_before_m" in st:
                parts.insert(0, f"slice {setup['slice']} start {st['path_before_m']:.2f} m before the end of the walk, "
                                f"{st.get('side_m', 0) * 100:+.0f} cm aside")
            elif "bank" in st:
                parts.insert(0, f"slice {setup['slice']} start from {st['bank']} {st.get('bank_source') or 'ep ' + str(st['bank_episode'])} as recorded ({st.get('plate')} {st.get('ahead_m', 0):.2f} m ahead, "
                                f"{st.get('left_m', 0) * 100:+.0f} cm aside)" + (f", variant {st['variant']}" if st.get("variant") else ""))
            elif "at_counter" in st:
                parts.insert(0, f"slice {setup['slice']} start at the counter ({st['at_counter']} {st.get('ahead_m', 0):.2f} m ahead, {st.get('left_m', 0) * 100:+.0f} cm aside"
                                + (f", standing as a1_end ep {st['pose_episode']}" if st.get("pose_episode") is not None else "")
                                + (f"; the {st['env']} envelope, legs of ep {st['legs_episode']}, valid draw {st['tries']}" if st.get("env") else "")
                                + (f", UNCHECKED: {st['unchecked']}" if st.get("unchecked") else "") + ")" + (f", variant {st['variant']}" if st.get("variant") else ""))
            elif "at_plate" in st:
                parts.insert(0, f"slice {setup['slice']} start at {st['at_plate']} ({st.get('ahead_m', 0):.2f} m ahead)" + (f", variant {st['variant']}" if st.get("variant") else ""))
            elif "at_pedal" in st:
                parts.insert(0, f"slice {setup['slice']} start at the pedal ({st.get('ahead_m', 0):.2f} m ahead)")
            else:
                parts.insert(0, f"slice {setup['slice']} start in its stance box")
            if st.get("spawn_back_m"):
                parts.append(f"spawned {st['spawn_back_m']:.2f} m behind")
            P = setup.get("preset") or {}
            if P.get("pose"):
                parts.append(f"pose {P['pose'] if isinstance(P['pose'], str) else P['pose'].get('source', 'inline')}")
            if "door_deg" in P:
                parts.append(f"door {P['door_deg']:g} deg")
            if "rack_out_m" in P:
                parts.append(f"upper rack out {P['rack_out_m'] * 100:.0f} cm")
            if P.get("plates_in_rack"):
                parts.append("in the rack: " + ", ".join(f"{p} ({slot})" for p, slot in P["plates_in_rack"].items()))
            for p, xy in (P.get("plate_spots") or {}).items():
                parts.append(f"{p} at ({xy[0]:.3f}, {xy[1]:+.3f})" + (f" yaw {xy[2]:.0f}" if len(xy) > 2 else ""))
            if P.get("plate_in_hand"):
                H = P["plate_in_hand"]
                how = {"pinch": "pinched by", "a2e41": "held level (A2 ep41's grasp) in", "bank": "held as recorded (" + str(H.get("source", "")) + ") in"}.get(H.get("grasp"), "in")
                parts.append(f"{H['plate']} {how} the {H['side']} hand" + (" (arm jittered)" if H.get("arm") and H.get("grasp") != "bank" else ""))
            if P.get("bin_lid"):
                parts.append(f"bin lid {P['bin_lid']}")
            if P.get("foot_on_pedal"):
                parts.append(f"{P['foot_on_pedal']} foot on the pedal")
            if P.get("arms"):
                parts.append("arms drawn: " + ", ".join(P["arms"]))
            if P.get("hands"):
                parts.append("hands drawn: " + ", ".join(f"{sd} {hand_kind(q)} ({hand_curl(q):.0%} shut)" for sd, q in P["hands"].items()))
            if P.get("waist"):
                parts.append(f"waist ({P['waist'][0]:+.2f}, {P['waist'][1]:+.2f}, {P['waist'][2]:+.2f})")
        Lo = setup.get("L", {})
        if "bin_x" in Lo:
            parts.append(f"bin ({Lo['bin_x']:+.3f}, {Lo['bin_y']:+.3f})")
        if "cart_cx" in Lo:
            parts.append(f"cart ({Lo['cart_cx']:.3f}, {Lo['cart_cy']:+.3f})")
        if "table_dz" in setup:
            parts.append(f"table {setup['table_dz'] * 100:+.1f} cm")
        parts.append(f"{setup.get('distractors', 0)} distractors")
        seed = setup.get("seed")
        return ("seed " + str(seed) + ": " if seed is not None else "nominal: ") + ", ".join(parts)


def spawn_shift(setup: dict[str, Any]) -> tuple[float, float]:
    """(dx, dy) from where a spawn-back slice setup spawns the robot to its start: spawn_back_m along the start heading.
    SceneSetups.apply spawns the robot at start - (dx, dy); the teleop CLI moves it by + (dx, dy) before the quick start."""
    back = float((setup.get("start") or {}).get("spawn_back_m", 0.0) or 0.0)
    if not back:
        return 0.0, 0.0
    h = math.radians(float(setup["robot"]["yaw_deg"]))
    return back * math.cos(h), back * math.sin(h)


def dumps(setup: dict[str, Any]) -> str:
    return json.dumps(setup, sort_keys=True)
