"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Plate-in-hand starts for the kitchen slices (2026-11-02; user: "S2 should be S1 continued work, then I would assume the plate is
already in the robot's hand"): OmniReset's "Grasped" tier done procedurally instead of from a checkpoint.

A slice preset carries ``plate_in_hand = {"plate": "plate_a" | "plate_b", "side": "left" | "right"}``. After the teleop CLI has
stood the robot at its start (``_stand_on_floor``), ``hold_plate`` puts that arm into kitchen7 episode 1's carry pose, the
eleven finger joints into ep1's grasp, and the plate into the hand where ep1 held it: the plate centre 15 cm from the hand
base, its rim between the thumb and the index / middle / ring fingers, leaning about 70 deg from flat. The grasp was read
from ep1's exact log (frame 2000 = 40.0 s for the right hand carrying plate A to the bin; frame 4400 = 88.0 s for the left
hand after the hand-over, holding plate A while the right hand opened the door). Other plate types are shifted along the
radial direction so the same rim point sits in the fingers (plate_27 of ep1 is 219 mm across; the set runs 165-260 mm).

The operator's grip is open when a setup loads, which would open the hand and drop the plate, so ``GripLatch`` keeps that
hand's command at the closed grip pose until the operator has pressed the grip (or the trigger) once; from then on the
hand follows the operator as always. The latched command is what the recorder and the exact log see, so the replay is
bit-exact without knowing about the latch.

Atomic slices (2026-11-02 evening, docs/atomic_slicing_scenes; user: "slice the task until first cleaning plate into atomic
action and create the teleop scene for each"): one more start kind from ep1's states. ``pose`` = all 29 body joints and both
hands' fingers at one ep1 frame (``POSES``; ``apply_pose`` sets them and puts the feet back on the floor): A3 "grasp and lift"
starts with the hands at the plate's rim as at 16.4 s. ``plate_in_hand.grasp = "pinch"`` (``GRASPS["left_pinch"]``, ep1's
left-hand rim pinch at 21.5 s with the plate 3 cm up and leaning 10 deg) is built but NOT used by any slice: placed and held
like the controller holds (fingers toward the closed grip), the plate fell flat on the counter within 0.5 s and kept sliding
under the fingers for 49 of the 50 plate types (scratchpad grasp_sweep_pinch.py / grasp_sweep_pinch2.py, 2026-11-02) -- the
grasp -> lift boundary is a transient between two moving hands, not a state a slice can start from.

A3 since 2026-11-05 (user: "use A2 ep41 hand pose and arm pose as reference holding, and adjust the A3 30 setup based on
that ... include similar randomization rules"): ``plate_in_hand.grasp = "a2e41"`` -> ``GRASPS["right_a2e41"]`` / the mirrored
``"left_a2e41"``, the LEVEL hold of the one A2 pick that kept the plate flattest with the indicator balls on it (page
docs/a2_grasp_poses). Such a grasp has its own ``r_ref`` (the recorded plate's radius for the radial shift), its own latch
target ``cmd`` (``latch_target``: the operator's recorded hand command rather than the closed grip) and a ``jitter`` scale for
CARRY_JITTER. The raised arm is kept by the held start (slices A3 ``start_mode = "hold"``: motion tracking on the placed
joints until the operator's Y, as A2), the rest of A3's randomization is unchanged (box, free arm / hand / waist, plate spots
and yaw, which plate in which hand, plate types from the grasp's graspable list of the loaded plate set).

A3 since 2026-11-10 (user: "use the 28 scenes to set the A3 init pose"): ``plate_in_hand.grasp = "bank"`` carries its grasp
INLINE, read from a recorded A2 end (start_banks/a2_press_end.json, the A2 episodes where one hand pressed one side of the plate
and the other grasped the raised side): ``rel_pos`` / ``rel_quat`` (the plate in the hand base frame), ``fingers``, ``cmd`` (the
operator's recorded hand command, the latch target), ``r_ref`` (that plate's radius: the same plate type, so no radial shift)
and ``source``. ``grasp_spec`` returns the GRASPS entry or that inline one; ``held_latch_target`` the command GripLatch holds.
"""

from __future__ import annotations

import math

import numpy as np

HAND_W = 11
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
FINGER_JOINTS = ("thumb_metacarpal", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal", "middle_proximal", "middle_distal",
                 "ring_proximal", "ring_distal", "pinky_proximal", "pinky_distal")
PLATE_FREE_JOINTS = {"plate_a": "k_plate_a_free", "plate_b": "k_plate_b_free"}
PLATE_BODIES = {"plate_a": "k_plate_a", "plate_b": "k_plate_b"}
R_REF = 0.1096                                                               # plate_27's radius (219.2 mm across): the plate ep1 carried
# The fingers start exactly at ep1's angles and the live hand PD (the latched closed command) takes over from the first tick:
# that is ep1's own state, which held the plate under the controller for minutes (creep 0.4 mm/s). A "clamp" pre-step that
# pinned the plate and closed the fingers on it was tried on 2026-11-02 and dropped: a plate whose rim sits a few mm into the
# thumb base pushed the pinned-plate thumb open (its actuator saturates at 0.5 N m), and pinning any body by resetting qpos
# every step makes a held plate creep 9 cm/s in the check even from ep1's own recorded state.

# kitchen7 ep1's grasps: the plate pose in the {side}_hand_base_link frame (pos m, quat wxyz), the finger joints (MJCF order,
# rad) and the arm joints (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, rad) at that frame
GRASPS = {
    "right": dict(source="kitchen7 ep1 frame 2000 (40.0 s), carrying plate A to the bin",
                  rel_pos=(0.1110, 0.0094, 0.1021), rel_quat=(0.2462, -0.6652, 0.7048, 0.0096),
                  fingers=(1.467, 0.421, 0.422, 0.691, 0.798, 0.792, 0.914, 0.852, 0.983, 1.384, 1.598),
                  arm=(-0.067, -0.316, 0.171, 0.085, 0.282, -0.068, 0.024)),
    "left": dict(source="kitchen7 ep1 frame 4400 (88.0 s), holding plate A after the hand-over, door just opened",
                 rel_pos=(0.1159, -0.0441, 0.0913), rel_quat=(0.1194, 0.7108, 0.6928, 0.0235),
                 fingers=(1.515, 0.293, 0.294, 1.191, 1.374, 1.054, 1.218, 1.24, 1.432, 1.384, 1.598),
                 arm=(-0.267, 0.579, 0.059, -0.32, -0.656, -0.113, -0.091)),
    # the pinch that starts ep1's lift: plate A 3.4 cm above its rest, leaning 9.8 deg, held between the left ring / pinky distals
    # and the thumb base, the right thumb resting on the far rim (the right arm comes from POSES["k7_f1077"]). Not a stable
    # start (see the module docstring); kept for the record and for a checkpoint-style start later.
    "left_pinch": dict(source="kitchen7 ep1 frame 1077 (21.5 s), the rim pinched by the left hand as the lift begins",
                       rel_pos=(0.1247, 0.0766, 0.1269), rel_quat=(0.6975, 0.4622, 0.436, -0.3312),
                       fingers=(1.519, 0.598, 0.6, 1.399, 1.615, 0.731, 0.845, 0.202, 0.232, 0.14, 0.162),
                       arm=(-0.1282, 0.4335, -0.2516, -0.2397, -0.6263, 0.1201, -0.0162)),
    # The LEVEL hold of A2 episode 41 (2026-11-05; user: "use A2 ep41 hand pose and arm pose as reference holding" for the A3
    # carry). docs/a2_grasp_poses: of the 44 recorded A2 picks, ep41 kept the plate within 3 deg of flat at the gate with both
    # indicator balls on it, the index / middle / ring pads under the rim and the thumb pad on top (four pads, the grip that
    # passed; the steep rim pinches failed). Read from its exact log at frame 376 (7.52 s, 2.9 deg from flat, the plate 10 cm
    # up, 0.45 m ahead / 0.10 m right / 0.21 m above the pelvis; plate 23 of the lift set, 183 mm across = r_ref 0.0914, by
    # plate_radius). The arm is raised and turned out (shoulder pitch -0.58, roll -0.83, yaw 0.44, elbow 0.19, wrist roll
    # -0.80), nothing like the hanging carry: a start from it is HELD by motion tracking (slices A3 start_mode "hold"), the
    # walking policy would swing the arm down to its default. `cmd` = the skeleton hand's command at that frame, 0.1-0.26 rad
    # beyond the fingers the plate stopped: the latch holds THAT, not the closed grip (0.75-1.05 rad beyond: a different grip
    # that turns the plate). The left-hand version is the mirror image (hand frame y -> -y: rel y, quat x / z and the arm's
    # roll / yaw signs flipped); the Revo 2 left hand is the right hand's mirror.
    "right_a2e41": dict(source="A2 ep41 frame 376 (7.52 s, teleop_holomotion_v14_kitchen_a2_hands): the plate just lifted, 2.9 deg from flat, index / middle / ring pads under the rim, thumb on top",
                        rel_pos=(0.0958, 0.0062, 0.1383), rel_quat=(-0.2893, -0.5863, 0.0323, 0.7560), r_ref=0.0914, level_deg=2.9,
                        fingers=(1.2434, 0.6421, 0.643, 0.5599, 0.6465, 0.6943, 0.8016, 0.7133, 0.8238, 0.6384, 0.7376),
                        cmd=(1.45, 0.9, 0.9, 0.6568, 0.7585, 0.87, 1.0048, 0.8025, 0.9269, 0.5827, 0.673),
                        arm=(-0.5832, -0.8257, 0.4352, 0.1916, -0.8017, -0.3013, -0.2509), jitter=1.0),
    "left_a2e41": dict(source="A2 ep41 frame 376 mirrored for the left hand (the plate level, three pads under the rim, thumb on top)",
                       rel_pos=(0.0958, -0.0062, 0.1383), rel_quat=(-0.2893, 0.5863, 0.0323, -0.7560), r_ref=0.0914, level_deg=2.9,
                       fingers=(1.2434, 0.6421, 0.643, 0.5599, 0.6465, 0.6943, 0.8016, 0.7133, 0.8238, 0.6384, 0.7376),
                       cmd=(1.45, 0.9, 0.9, 0.6568, 0.7585, 0.87, 1.0048, 0.8025, 0.9269, 0.5827, 0.673),
                       arm=(-0.5832, 0.8257, -0.4352, 0.1916, 0.8017, -0.3013, 0.2509), jitter=1.0, mirrored=True),
}

# Whole-body poses at ep1 frames (the 29 body joints in MJCF order: legs, waist, left arm, right arm; both hands' eleven
# finger joints), for the atomic slices whose start is "hands at the plate" rather than "plate in the hand". The legs are
# ep1's own standing pose at that moment (pelvis 0.78 m, 2 cm taller than the default stance), so the hands land at ep1's
# height above the worktop.
POSES = {
    "k7_f819": dict(source="kitchen7 ep1 frame 819 (16.4 s): both hands at plate A's rim, open, the right ring finger touching it",
                    body=(-0.1645, 0.0972, 0.1648, 0.3153, -0.1676, -0.047, -0.0898, -0.1143, -0.2135, 0.3031, -0.2221, 0.0589,
                          0.0872, 0.0117, -0.0085,
                          0.0521, 0.4543, -0.0633, -0.1507, -0.6142, 0.0259, 0.0416,
                          0.0158, -0.4733, 0.0612, -0.1849, 0.3681, 0.0905, 0.0902),
                    fingers=dict(left=(1.519, 0.044, 0.044, 0.087, 0.101, 0.087, 0.101, 0.086, 0.099, 0.078, 0.09),
                                 right=(1.489, -0.001, -0.001, 0.045, 0.052, 0.006, 0.007, 0.031, 0.036, 0.0, 0.0))),
    "k7_f2650": dict(source="kitchen7 ep1 frame 2650 (53.0 s): swiping, plate A in the right hand over the open bin, the left hand mid-pass",
                     body=(-0.0897, 0.0032, -0.0006, 0.3541, -0.1988, -0.0191, -0.2661, -0.08, -0.0159, 0.3938, -0.072, 0.0203,
                           -0.0026, -0.0781, 0.0328,
                           -0.2155, 0.4524, -0.4646, -0.3007, -0.6135, -0.3017, -0.0584,
                           -0.1459, -0.5421, -0.0312, 0.0519, 0.2221, -0.0856, -0.1011),
                     fingers=dict(left=(1.487, 0.042, 0.042, 0.087, 0.101, 0.087, 0.101, 0.087, 0.1, 0.087, 0.1),
                                  right=(1.487, 0.374, 0.375, 0.749, 0.865, 0.825, 0.953, 0.862, 0.995, 1.384, 1.598))),
    "k7_f1077": dict(source="kitchen7 ep1 frame 1077 (21.5 s): the rim pinched by the left hand, the right thumb on the far rim, the lift beginning",
                     body=(-0.1443, 0.1055, 0.2184, 0.3745, -0.2573, -0.0382, -0.0764, -0.1167, -0.1611, 0.3098, -0.2582, 0.073,
                           0.0386, -0.0032, -0.0595,
                           -0.1282, 0.4335, -0.2516, -0.2397, -0.6263, 0.1201, -0.0162,
                           -0.2066, -0.4697, 0.1638, 0.1173, 0.238, -0.0336, 0.0562),
                     fingers=dict(left=(1.519, 0.598, 0.6, 1.399, 1.615, 0.731, 0.845, 0.202, 0.232, 0.14, 0.162),
                                  right=(1.467, -0.0, -0.0, 0.045, 0.052, 0.006, 0.007, 0.004, 0.005, 0.019, 0.022))),
}


# Start randomization for the atomic slices (2026-11-02 evening; user: "the init randomization higher than usual, robot pose
# including hand open and close pose when there is nothing in hand, robot location, plates location on the table, plate type").
# Arm joints are drawn uniformly in these ranges (rad, MJCF order shoulder pitch / roll / yaw, elbow, wrist roll / pitch / yaw;
# the right arm mirrors roll and yaw), the waist in WAIST_RANGE (yaw, roll, pitch), a free hand anywhere between open and the
# closed grip or the thumb-index pinch. The ranges keep the hands in front of and beside the body (no self-collision model is
# checked); the quick start holds the drawn pose until the operator engages, and the drawn hand pose is latched like a held plate.
ARM_RANGE = ((-0.55, 0.45), (0.12, 0.75), (-0.35, 0.35), (0.15, 1.15), (-0.55, 0.55), (-0.35, 0.35), (-0.35, 0.35))
ARM_RANGE_LOW = ((-0.10, 0.50), (0.12, 0.60), (-0.30, 0.30), (0.20, 1.10), (-0.50, 0.50), (-0.35, 0.35), (-0.35, 0.35))   # one step from the worktop / at the bin: the hand stays low
ARM_RANGE_NEAR = ((0.15, 0.70), (0.12, 0.50), (-0.30, 0.30), (0.80, 1.40), (-0.50, 0.50), (-0.35, 0.35), (-0.35, 0.35))   # at the counter stance (0.22 m from the drawer fronts): fingertips <= 0.17 m ahead of the pelvis (measured: pitch +0.15 / elbow 0.8 reaches 0.17 m; the default pose 0.28 m; pitch -0.2 / elbow 0.2 0.46 m)
CARRY_JITTER_LEFT = 0.5                                                      # the left hold grasp (ep1 frame 4400, arm bent close to the body) slips at the full jitter: half of it
WAIST_RANGE = ((-0.30, 0.30), (-0.05, 0.05), (-0.05, 0.22))
CARRY_JITTER = (0.08, 0.06, 0.10, 0.10, 0.10, 0.10, 0.10)                     # +- around ep1's carry / hold pose when a plate is in that hand (0.15 on the wrist roll slipped a left-hand grasp 5.7 cm)
POSE_JITTER = 0.015                                                          # +- on every arm joint of a whole-body pose (hands at the rim: 0.04 put fingertips 1-2 cm into the worktop)


def draw_arm(rng, side: str, rng_name: str = "full") -> list[float]:
    q = [float(rng.uniform(lo, hi)) for lo, hi in {"low": ARM_RANGE_LOW, "near": ARM_RANGE_NEAR}.get(rng_name, ARM_RANGE)]
    if side == "right":
        q[1], q[2], q[4], q[6] = -q[1], -q[2], -q[4], -q[6]
    return [round(v, 3) for v in q]


# A free hand's start pose is a RESTING human hand, not a grasp (user 2026-11-03: "consider the regular human hand pose when doing
# the work, don't just random everything without considering how human hand and finger will behave"). One curl per hand, 0 = flat,
# 1 = the closed grip, drawn Beta(2, 4) x HAND_CURL_MAX (mean 0.15: most hands slightly curled, a few flat, none half shut), the
# fingers curling a little more toward the little finger as a resting hand does (index x0.85 ... pinky x1.15), the distal joint
# following the proximal, the thumb curled half as much with its opposition eased a little toward the pinch, +-0.04 rad per joint.
# Capped at 0.45 of the closure because the middle fingertip meets the thumb pad at 0.5 (hand_teleport_test.py). The old kinds
# (HAND_MIX: closed grip 20 %, pinch 15 %, "between" up to 85 %, set by qpos) put the fingertips into the palm / onto the thumb,
# where the hand PD could not open them again after the operator's first squeeze (kitchen_a2 ep0, 2026-11-03).
HAND_CURL_MAX = 0.45
HAND_CURL_WEIGHTS = {"index": 0.85, "middle": 0.95, "ring": 1.05, "pinky": 1.15}
HAND_CURL_BINS = ((0.08, "flat"), (0.25, "relaxed"), (1.01, "curled"))       # how a drawn hand is described (hand_curl)


def draw_hand(rng) -> list[float]:
    from simple.robots.g1_sonic_revo2 import CLOSED_Q, OPEN_Q, REVO2_UPPER
    span = np.asarray(CLOSED_Q, dtype=np.float64) - np.asarray(OPEN_Q, dtype=np.float64)
    c = float(rng.beta(2.0, 4.0)) * HAND_CURL_MAX
    q = np.asarray(OPEN_Q, dtype=np.float64).copy()
    for i, name in enumerate(FINGER_JOINTS):
        finger = name.split("_")[0]
        if name == "thumb_metacarpal":
            q[i] = OPEN_Q[i] + float(rng.uniform(-0.15, 0.0))                     # opposition: between the grip's 1.51 and the pinch's 1.23
        elif finger == "thumb":
            q[i] = OPEN_Q[i] + 0.5 * c * span[i] + float(rng.uniform(-0.03, 0.03))
        else:
            q[i] = OPEN_Q[i] + min(c * HAND_CURL_WEIGHTS[finger], HAND_CURL_MAX) * span[i] + float(rng.uniform(-0.04, 0.04))
    upper = np.asarray([REVO2_UPPER[0], REVO2_UPPER[1], REVO2_UPPER[1], *([REVO2_UPPER[3]] * 8)], dtype=np.float64)[:len(q)]
    q = np.clip(q, 0.0, upper)
    return [round(float(v), 3) for v in q[:HAND_W]]


def hand_curl(q) -> float:
    """How far a hand pose is from flat toward the closed grip, 0..1 (mean over the eight finger joints)."""
    from simple.robots.g1_sonic_revo2 import CLOSED_Q, OPEN_Q
    q = np.asarray(q, dtype=np.float64)[:HAND_W]; span = (CLOSED_Q - OPEN_Q)[:HAND_W]
    f = [i for i, n in enumerate(FINGER_JOINTS) if not n.startswith("thumb")]
    return float(np.mean((q[f] - OPEN_Q[f]) / span[f]))


def hand_kind(q) -> str:
    """flat / relaxed / curled by hand_curl, or the grip poses by name."""
    from simple.robots.g1_sonic_revo2 import CLOSE2_Q, CLOSED_Q, OPEN_Q
    q = np.asarray(q, dtype=np.float64)[:HAND_W]
    for name, ref in (("closed grip", CLOSED_Q), ("pinch", CLOSE2_Q)):
        if np.allclose(q, np.asarray(ref)[:HAND_W], atol=2e-3):
            return name
    c = hand_curl(q)
    return next(k for lim, k in HAND_CURL_BINS if c < lim)


def grasp_key(P: dict) -> str:
    """GRASPS / GRASPABLE key of a plate_in_hand preset: the side for ep1's carry grasp, "<side>_<grasp>" for another."""
    g = P.get("grasp", "carry")
    return P["side"] if g in ("carry", None, "") else f"{P['side']}_{g}"


def latch_target(key: str) -> np.ndarray:
    """The hand command GripLatch holds for a plate-in-hand start until the operator squeezes: the grasp's own recorded
    command when it has one (ep41's skeleton hand, 0.1-0.26 rad beyond the fingers the plate stopped), else the closed grip
    (ep1's carry, whose operator held the grip button)."""
    from simple.robots.g1_sonic_revo2 import CLOSED_Q
    cmd = (GRASPS.get(key) or {}).get("cmd")
    return np.asarray(cmd if cmd is not None else CLOSED_Q, dtype=np.float64)[:HAND_W].copy()


def grasp_spec(P: dict) -> dict:
    """The grasp of a plate_in_hand preset: the GRASPS entry of its key, or the preset's own inline grasp (``grasp="bank"``: a
    recorded end state carries rel_pos, rel_quat, fingers, cmd, r_ref and source)."""
    if P.get("rel_pos") is not None:
        return dict(rel_pos=P["rel_pos"], rel_quat=P["rel_quat"], fingers=P["fingers"], cmd=P.get("cmd"), r_ref=P.get("r_ref", R_REF),
                    arm=P.get("arm"), source=P.get("source", "inline grasp"), inline=True)
    return GRASPS[grasp_key(P)]


def held_latch_target(P: dict) -> np.ndarray:
    """The hand command GripLatch holds for the hand that starts with the plate: an inline grasp's recorded command, else the
    grasp table's (latch_target)."""
    if P.get("cmd") is not None:
        return np.asarray(P["cmd"], dtype=np.float64)[:HAND_W].copy()
    return latch_target(grasp_key(P))


def carry_jitter_scale(key: str) -> float:
    """How much of CARRY_JITTER the holding arm of this grasp gets (per grasp; ep1's left hold half of it)."""
    G = GRASPS.get(key) or {}
    return float(G.get("jitter", CARRY_JITTER_LEFT if key.startswith("left") else 1.0))


# Plate types ep1's grasp holds, per hand (scratchpad grasp_sweep.py 2026-11-02: every one of the 50 types placed in the hand
# and held 1 s under a controller-like hold, kept when still in the hand with < 2 cm of slip; 32 of 50 in the right hand, 24 in
# the left). The other rims do not fit the finger spacing of the one recorded grasp (they drop or creep 2-15 cm/s). A
# plate-in-hand slice draws its held plate from the list of its side (scene_setup.SceneSetups.sample); None = all types.
GRASPABLE: dict[str, list[int] | dict[str, list[int]]] | None = {
    "right": [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 19, 22, 23, 24, 26, 27, 28, 30, 31, 33, 34, 35, 36, 38, 41, 42, 44, 47, 48, 49],
    "left": [1, 2, 3, 4, 5, 7, 8, 9, 13, 14, 15, 18, 23, 26, 27, 31, 33, 35, 36, 38, 41, 42, 43, 46],
    # ep41's level hold, per plate set (2026-11-05 sweep, scratchpad a2e41_sweep.py: every type as plate A in each hand, built as
    # the A3 start is built with ep41's own waist, held 1 s like the controller -- legs / waist at the robot's PD gains, the arm
    # stiff at 150 so the grasp is what is tested, the fingers at the grasp's latch target --; kept when still in the hand, the
    # centre moved < 2 cm in the hand frame and the plate within 15 deg of flat afterwards; ep41's own type 23 of the lift set
    # ends at 13 deg: the plate pivots on the pads under the weak finger PD, as it did in the recording, 3 -> 8 deg over the hold).
    # 48 of 50 lift types stay in the hand, 26-27 within 2 cm; the tilt cut makes the list. The v1 rims are shallower: fewer.
    # v2 (2026-11-06) = the lift set's 50 plates at the same indices + 20 wide-rim pasta plates as 50-69: the lift lists carry over;
    # none of the pasta plates fits ep41's hold (the sweep over 50-69: the index / middle distal links end 16-20 mm inside the
    # bowl wall, which starts 45-65 mm in from the edge; slid 3.5-5 cm further out of the hand (FAMILY_OUT_M) the fingers clear
    # the bowl but the thumb no longer clamps the rim and the plate drops), so A3 keeps drawing lift types under --plate-set v2.
    "right_a2e41": {"lift": [0, 1, 3, 6, 8, 12, 15, 23, 26, 28, 29, 32, 38, 39, 43, 49], "v2": [0, 1, 3, 6, 8, 12, 15, 23, 26, 28, 29, 32, 38, 39, 43, 49], "v1": [3, 4, 5, 18, 29, 38, 39, 40]},
    "left_a2e41": {"lift": [0, 1, 6, 10, 13, 15, 23, 26, 28, 29, 32, 38, 39, 43, 49], "v2": [0, 1, 6, 10, 13, 15, 23, 26, 28, 29, 32, 38, 39, 43, 49], "v1": [0, 2, 3, 4, 17, 18, 29, 37, 40]},
}
PLATE_SET_NAMES = {"kitchen_plates": "v1", "kitchen_plates_lift": "lift", "kitchen_plates_v2": "v2"}   # data/<dir> -> the --plate-set name (v2 = lift + pasta plates, 2026-11-06)


def plate_set_name() -> str:
    """Which procedural plate set the kitchen loads (KITCHEN_PLATES, set by the CLI's --plate-set before the kit imports): v1 / lift / the directory name."""
    import os
    d = os.path.basename(os.environ.get("KITCHEN_PLATES", "").rstrip("/")) or "kitchen_plates"
    return PLATE_SET_NAMES.get(d, d)


def graspable_ids(key: str) -> list[int] | None:
    """The plate types this grasp holds (GRASPABLE[key]; a per-plate-set dict is resolved by plate_set_name()), None = all."""
    ok = (GRASPABLE or {}).get(key)
    if isinstance(ok, dict):
        ok = ok.get(plate_set_name())
    return list(ok) if ok else None


PLATE_B_OFFSET = 25                                                          # kitchen_task: plate_b = (plate + 25) % N


def pick_plate_id(seed: int, held_plate: str, side: str, n_plates: int, grasp: str = "carry") -> int:
    """The setup's plate id for a plate-in-hand slice: the seed-th graspable type of that hand (and grasp) as the held plate
    (plate_a = id, plate_b = id + 25); the other plate follows from the id as always."""
    ok = graspable_ids(grasp_key(dict(side=side, grasp=grasp)))
    if not ok or not n_plates:
        return int(seed) % max(n_plates, 1)
    k = ok[int(seed) % len(ok)]
    return k % n_plates if held_plate == "plate_a" else (k - PLATE_B_OFFSET) % n_plates


def _quat_mat(q):
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64)); return R.reshape(3, 3)


def plate_radius(m, body: int) -> float:
    """Half the plate's width across, from its geoms' bounding boxes (the procedural plates are flat: the largest horizontal half extent)."""
    r = 0.0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == body:
            r = max(r, float(np.max(m.geom_aabb[g, 3:5])))
    return r if r > 0 else R_REF


FAMILY_OUT_M = {"pasta": 0.0}                                                 # extra outward shift of a held plate per plate family (set below the sweep)


def plate_family(m, pb: int) -> str:
    """The procedural plate family of a plate body from its geom names (plate_swap tags them <body>_<id>_...): pasta / dinner."""
    import mujoco
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == pb and "pasta_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
            return "pasta"
    return "dinner"


def grasp_pose(key, radius: float, out_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(rel_pos, rel_quat) of a plate of this radius in the hand base frame: the recorded grasp (GRASPS[key]), the centre
    shifted along the plate's radial direction toward / away from the hand so the same rim point sits between the fingers
    (the recorded plate's radius is the grasp's r_ref, ep1's plate_27 by default); `out_m` slides the plate that much further
    out of the hand (the fingers less deep under the rim: the pasta family's bowl wall starts 45-65 mm in from the edge)."""
    G = key if isinstance(key, dict) else GRASPS[key]                     # a GRASPS key, or a grasp dict (grasp_spec: an inline bank grasp)
    R = _quat_mat(G["rel_quat"]); t = np.asarray(G["rel_pos"], dtype=np.float64)
    h = -R.T @ t                                                          # the hand base in the plate frame
    u = np.array([h[0], h[1], 0.0]); u /= max(np.linalg.norm(u), 1e-9)    # radial direction toward the grasped rim point
    return t + R @ ((float(G.get("r_ref", R_REF)) - radius - float(out_m)) * u), np.asarray(G["rel_quat"], dtype=np.float64)


def hold_plate(sonic_env, preset: dict | None, arm_pose: bool = True) -> dict | None:
    """Put the preset's plate into the robot's hand (see the module docstring). Call after the robot stands at its start and
    before the controller's quick start; returns what was done (for the log) or None when the preset asks for nothing."""
    import mujoco
    P = (preset or {}).get("plate_in_hand")
    if not P:
        return None
    side, plate, key = P["side"], P["plate"], grasp_key(P)
    inline = P.get("rel_pos") is not None                                 # a recorded end's own grasp (A3 from the A2 press-and-grasp ends)
    if side not in ("left", "right") or (not inline and key not in GRASPS) or plate not in PLATE_FREE_JOINTS:
        raise ValueError(f"plate_in_hand: side {side!r} / grasp {P.get('grasp', 'carry')!r} / plate {plate!r} unknown")
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    G = grasp_spec(P)

    def jadr(name):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise RuntimeError(f"plate_in_hand: joint {name!r} is not in the scene")
        return j, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])

    if arm_pose:
        for n, q in zip(ARM_JOINTS, P.get("arm") or G["arm"]):             # ep1's carry pose, or the setup's jittered copy of it
            _, qa, va = jadr(f"{side}_{n}_joint"); d.qpos[qa] = q; d.qvel[va] = 0.0
    for n, q in zip(FINGER_JOINTS, G["fingers"]):
        _, qa, va = jadr(f"{side}_hand_{n}_joint"); d.qpos[qa] = q; d.qvel[va] = 0.0
    mujoco.mj_forward(m, d)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link")
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLATE_BODIES[plate])
    radius = plate_radius(m, pb); family = plate_family(m, pb)
    out_m = 0.0 if inline else float(FAMILY_OUT_M.get(family, 0.0))       # a recorded grasp of this very plate type needs no family slide
    rel_pos, rel_quat = grasp_pose(G, radius, out_m)
    Rh, ph = d.xmat[hb].reshape(3, 3), d.xpos[hb]
    qh = np.zeros(4); mujoco.mju_mat2Quat(qh, d.xmat[hb]); qp = np.zeros(4); mujoco.mju_mulQuat(qp, qh, rel_quat)
    j, qa, va = jadr(PLATE_FREE_JOINTS[plate])
    d.qpos[qa:qa + 3] = ph + Rh @ rel_pos; d.qpos[qa + 3:qa + 7] = qp; d.qvel[va:va + 6] = 0.0
    mujoco.mj_forward(m, d)
    return dict(plate=plate, side=side, grasp=P.get("grasp", "carry"), radius_m=round(radius, 4), family=family, out_m=out_m, plate_xyz=d.xpos[pb].round(3).tolist(), source=G["source"])


def _refloor(m, d) -> None:
    """The lowest foot sphere 1 mm above the floor (as _stand_on_floor leaves it), after the legs changed."""
    import mujoco
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); root = int(m.body_rootid[pel])
    a = int(m.jnt_qposadr[m.body_jntadr[root]])
    mujoco.mj_forward(m, d)
    feet = [g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and (m.geom_contype[g] or m.geom_conaffinity[g])]
    low = min(float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in feet)
    d.qpos[a + 2] += 0.001 - low
    mujoco.mj_forward(m, d)


def _set_joints(m, d, names, values) -> None:
    import mujoco
    for n, q in zip(names, values):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0:
            raise RuntimeError(f"joint {n!r} is not in the scene")
        lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf))
        d.qpos[m.jnt_qposadr[j]] = float(np.clip(q, lo, hi)); d.qvel[m.jnt_dofadr[j]] = 0.0


def apply_pose(sonic_env, preset: dict | None) -> dict | None:
    """Put the robot into the preset's start pose, feet back on the floor: ``pose`` (a POSES name: 29 body joints + both hands'
    fingers; or an inline dict {body, fingers?, source} -- a recorded end state from a start bank, scene_setup._bank_pose),
    then ``arms`` {side: 7 joints}, ``waist`` (yaw, roll, pitch) and ``hands`` {side: 11 finger joints} on top.
    Call after the robot stands at its start and before hold_plate; None when the preset asks for nothing."""
    import mujoco
    P = preset or {}
    name = P.get("pose")
    if not (name or P.get("arms") or P.get("waist") or P.get("hands")):
        return None
    m, d, robot = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData, sonic_env.task.robot
    out = {}
    if name:
        if isinstance(name, dict):
            W = name; name = W.get("source", "inline")
        elif name in POSES:
            W = POSES[name]
        else:
            raise ValueError(f"pose {name!r} unknown; built: {', '.join(POSES)}")
        bj = np.asarray(robot.body_joint_index, dtype=int)
        if len(bj) != len(W["body"]):
            raise RuntimeError(f"pose {name}: {len(W['body'])} joints for a robot with {len(bj)} body joints")
        d.qpos[m.jnt_qposadr[bj]] = np.asarray(W["body"], dtype=np.float64); d.qvel[m.jnt_dofadr[bj]] = 0.0
        for jn, q in (W.get("scene_joints") or {}).items():               # a recorded end's scene joints (A5: the pressed pedal and the open lid)
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = float(q); d.qvel[m.jnt_dofadr[j]] = 0.0
        if W.get("root_quat") is not None:                                 # a recorded end's pelvis orientation (the heading is the setup's, equal to it for a bank start)
            pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); ra = int(m.jnt_qposadr[m.body_jntadr[int(m.body_rootid[pel])]])
            d.qpos[ra + 3:ra + 7] = np.asarray(W["root_quat"], dtype=np.float64)
        for side in ("left", "right"):
            if W.get("fingers") and side in W["fingers"]:
                _set_joints(m, d, [f"{side}_hand_{n}_joint" for n in FINGER_JOINTS], W["fingers"][side])
        out.update(pose=name, source=W.get("source", name))
    for side, q in (P.get("arms") or {}).items():
        _set_joints(m, d, [f"{side}_{n}_joint" for n in ARM_JOINTS], q); out.setdefault("arms", []).append(side)
    if P.get("waist"):
        _set_joints(m, d, ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"], P["waist"]); out["waist"] = [round(float(v), 3) for v in P["waist"]]
    for side, q in (P.get("hands") or {}).items():
        _set_joints(m, d, [f"{side}_hand_{n}_joint" for n in FINGER_JOINTS], q); out.setdefault("hands", []).append(side)
    _refloor(m, d)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    out["pelvis_z"] = round(float(d.xpos[pel][2]), 3)
    return out



# ---------------------------------------------------------------------------------------------------------------------------
# A placed start adjusted a little until nothing of the robot is near the scene (user 2026-11-04: "adjust the A1 seed a little bit
# and make it no contact to the scene"): the A2 start is a recorded A1 end, and a few of those ends stand with the torso on the
# worktop front or a hand on a plate rim. Clearance, not just contact: under the hold the fingertips come down up to 4 cm.
NUDGE_CLEAR_M = 0.04
NUDGE_STEPS = ((0.02, 0.0), (0.0, 0.05), (0.02, 0.05), (0.04, 0.0), (0.04, 0.05), (0.04, 0.10), (0.06, 0.05), (0.06, 0.10),
               (0.08, 0.10), (0.10, 0.10), (0.10, 0.15), (0.12, 0.15))                    # (back along the heading m, shoulders up rad)
# Walking mode (user 2026-11-04: "make sure each pose the robot can stand still in the walking mode"): the walking policy owns the
# arms (its default pose: forearms forward, the hands 10 cm below a worktop top the pelvis stands 6-22 cm from) and its stance wanders
# up to 9 cm with a zero command, so the start is moved back only, 2 cm at a time, until nothing is within 10 cm.
NUDGE_WALK_CLEAR_M = 0.10


def nudge_back_from_edge(sonic_env, clear_m: float = NUDGE_WALK_CLEAR_M, step_m: float = 0.02, max_back_m: float = 0.60) -> dict | None:
    """A walking-mode start at the counter moved back along its heading, step_m at a time, until every robot geom (feet excepted)
    ends at least clear_m behind the worktop's front edge (scene_setup.WORKTOP_EDGE_X; the robot faces it within +-12 deg, so the
    arms, knees and torso all clear it together). Geometry only: a contact pass with a 10 cm margin on every robot geom overflows
    MuJoCo's arena (robot-to-robot pairs) and crashed the CLI (2026-11-04). None when clear as placed; else dict(back_m, ok)."""
    import mujoco
    from simple.teleop.holomotion_v14.scene_setup import WORKTOP_EDGE_X
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    root, rob = _robot_geoms(m)
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""   # noqa: E731
    rob = np.array([g for g in rob if "ankle_roll" not in bname(int(m.geom_bodyid[g]))], dtype=int)
    free = next(j for j in range(m.njnt) if int(m.body_rootid[m.jnt_bodyid[j]]) == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    a = int(m.jnt_qposadr[free]); w, x, y, z = d.qpos[a + 3:a + 7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    front = lambda: float(np.max(d.geom_xpos[rob, 0] + m.geom_rbound[rob]))   # noqa: E731  the robot's foremost point (+x = toward the worktop)
    mujoco.mj_forward(m, d)
    f0 = front()
    if f0 <= WORKTOP_EDGE_X - clear_m:
        return None
    back = 0.0
    while back < max_back_m:
        back = round(back + step_m, 3)
        d.qpos[a] -= step_m * math.cos(yaw); d.qpos[a + 1] -= step_m * math.sin(yaw); d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        if front() <= WORKTOP_EDGE_X - clear_m:
            return dict(back_m=back, arms_up_rad=0.0, before=[("front", "worktop edge", round((WORKTOP_EDGE_X - f0) * 1000, 1))], ok=True)
    return dict(back_m=back, arms_up_rad=0.0, before=[("front", "worktop edge", round((WORKTOP_EDGE_X - f0) * 1000, 1))], ok=False)


def arms_to_walk_default(sonic_env) -> None:
    """The 14 arm joints to the walking policy's default pose (robot_yaml default_joint_angles), the rest of the placed pose kept:
    a walking-mode start -- the policy drives the arms there within a second whatever they were placed at, and a recorded arm
    pose held out over the plates would be swept down through them on the way."""
    import mujoco
    from simple.teleop.holomotion_v14.sim_node import robot_yaml
    y = robot_yaml()["default_joint_angles"]
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    for side in ("left", "right"):
        names = [f"{side}_{n}_joint" for n in ARM_JOINTS]
        _set_joints(m, d, names, [float(y[n]) for n in names])
    mujoco.mj_forward(m, d)


def _robot_geoms(m):
    import mujoco
    root = int(m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")])
    return root, np.array([g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root], dtype=int)


def clearance_contacts(sonic_env, skip_body: str | None = None, clearance_m: float = NUDGE_CLEAR_M) -> list:
    """Robot parts closer than clearance_m to anything in the scene but the floor plane, the robot's own feet (ankle_roll links)
    and the held plate `skip_body`: (robot part, what, separation mm; negative = inside), nearest first. One forward pass with the
    robot's geom margins set to clearance_m, so the collision pipeline reports exactly those pairs with their distance; the
    margins are restored and the data recomputed."""
    import mujoco
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    root, rob = _robot_geoms(m)
    saved = m.geom_margin[rob].copy(); saved_type = m.geom_contype[rob].copy()
    m.geom_margin[rob] = clearance_m
    m.geom_contype[rob] = 0                                                  # no robot-to-robot pairs (they explode with a margin; the scene's
    try:                                                                     # conaffinity still meets the robot's contype-free geoms via its contype)
        mujoco.mj_forward(m, d)
        bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"   # noqa: E731
        skip_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, skip_body) if skip_body else -1
        out = []
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = int(m.body_rootid[b1]) == root, int(m.body_rootid[b2]) == root
            if r1 == r2 or c.dist >= clearance_m:
                continue
            rb, ob, og = (b1, b2, int(c.geom2)) if r1 else (b2, b1, int(c.geom1))
            if m.geom_type[og] == mujoco.mjtGeom.mjGEOM_PLANE or ob == skip_b or "ankle_roll" in bname(rb):
                continue
            out.append((bname(rb).replace("_link", ""), bname(ob).replace("k_", ""), round(float(c.dist) * 1000, 1)))
    finally:
        m.geom_margin[rob] = saved; m.geom_contype[rob] = saved_type
        mujoco.mj_forward(m, d)
    return sorted(out, key=lambda t: t[2])


def nudge_clear(sonic_env, skip_body: str | None = None, clearance_m: float = NUDGE_CLEAR_M, steps=NUDGE_STEPS) -> dict | None:
    """Move the placed start a little until nothing of the robot is within clearance_m of the scene: the first of `steps` (back
    along the heading, both shoulders pitched so the hands rise; smallest first) that clears it. None when the start is clear as
    placed; otherwise dict(back_m, arms_up_rad, before, ok) -- ok False: no step cleared it and the robot is left as placed."""
    import mujoco
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    before = clearance_contacts(sonic_env, skip_body, clearance_m)
    if not before:
        return None
    q0 = d.qpos.copy()
    root = int(m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")])
    free = next(j for j in range(m.njnt) if int(m.body_rootid[m.jnt_bodyid[j]]) == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    a = int(m.jnt_qposadr[free]); w, x, y, z = q0[a + 3:a + 7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    sj = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_shoulder_pitch_joint") for s in ("left", "right")]
    sp = [int(m.jnt_qposadr[j]) for j in sj]
    hb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_base_link") for s in ("left", "right")]
    zs = []
    for sgn in (-1.0, 1.0):                                                      # which way the shoulder pitch raises the hands on this model
        d.qpos[:] = q0; d.qpos[sp] += sgn * 0.1; mujoco.mj_forward(m, d); zs.append(sum(float(d.xpos[b][2]) for b in hb))
    sign = -1.0 if zs[0] >= zs[1] else 1.0
    for back, up in steps:
        d.qpos[:] = q0
        d.qpos[a] -= back * math.cos(yaw); d.qpos[a + 1] -= back * math.sin(yaw)
        for j, qi in zip(sj, sp):
            v = q0[qi] + sign * up
            if m.jnt_limited[j]:
                v = min(max(v, float(m.jnt_range[j][0])), float(m.jnt_range[j][1]))
            d.qpos[qi] = v
        d.qvel[:] = 0.0
        if not clearance_contacts(sonic_env, skip_body, clearance_m):
            return dict(back_m=back, arms_up_rad=up, before=before[:4], ok=True)
    d.qpos[:] = q0; d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
    return dict(back_m=0.0, arms_up_rad=0.0, before=before[:4], ok=False)

LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
FOOT_SOLE = np.array([0.035, 0.0, -0.03])                                   # ankle_roll link frame: the mid-sole point at the foot spheres' centre height (spheres at x -0.05 / 0.12, z -0.03, r 5 mm)


def foot_on_pedal(sonic_env, preset: dict | None) -> dict | None:
    """Put the preset's foot (``foot_on_pedal`` = "left" / "right") flat on the pedal bin's tread, the pedal pressed (the preset's
    bin_lid "open" sets the pedal down and the lid up): a damped least-squares IK on that leg's six joints, the pelvis where it
    stands (user 2026-11-02: "when wipe the plate the robot one foot should be on the pedal so that it drives the lid up").
    Call after apply_pose (the other foot is on the floor) and before hold_plate; None when the preset asks for nothing."""
    import mujoco
    side = (preset or {}).get("foot_on_pedal")
    if not side:
        return None
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    tread = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "k_pedal_tread")
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if tread < 0 or body < 0:
        raise RuntimeError("foot_on_pedal: no pedal tread / ankle link in the scene")
    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{n}_joint") for n in LEG_JOINTS]
    qa = np.array([int(m.jnt_qposadr[j]) for j in joints]); va = np.array([int(m.jnt_dofadr[j]) for j in joints])
    lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.inf for j in joints]); hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.inf for j in joints])
    mujoco.mj_forward(m, d)
    n = d.geom_xmat[tread].reshape(3, 3)[:, 2].copy()                       # the tread's normal (tilted with the pressed pedal)
    c = d.geom_xpos[tread] + n * (float(m.geom_size[tread][2]) + 0.005)     # the foot spheres' centres on the tread top
    fwd = d.xmat[pel].reshape(3, 3)[:, 0]; fwd = fwd - n * float(fwd @ n); fwd /= max(np.linalg.norm(fwd), 1e-9)
    R_t = np.column_stack([fwd, np.cross(n, fwd), n])                        # foot x along the pelvis heading, sole on the tread
    p_t = c - R_t @ FOOT_SOLE
    q_t = np.zeros(4); mujoco.mju_mat2Quat(q_t, R_t.ravel())
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv)); rot = np.zeros(3); qe = np.zeros(4); qinv = np.zeros(4)
    for _ in range(200):
        mujoco.mj_forward(m, d)
        e_p = p_t - d.xpos[body]
        mujoco.mju_negQuat(qinv, d.xquat[body]); mujoco.mju_mulQuat(qe, q_t, qinv); mujoco.mju_quat2Vel(rot, qe, 1.0)   # world-frame rotation body -> target
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(rot) < 1e-3:
            break
        mujoco.mj_jacBody(m, d, jacp, jacr, body)
        J = np.vstack([jacp[:, va], 0.3 * jacr[:, va]]); e = np.concatenate([e_p, 0.3 * rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        dq = np.clip(dq, -0.08, 0.08)
        d.qpos[qa] = np.clip(d.qpos[qa] + dq, lo, hi)
    d.qvel[va] = 0.0
    mujoco.mj_forward(m, d)
    err_mm = float(np.linalg.norm(p_t - d.xpos[body])) * 1000
    mujoco.mju_negQuat(qinv, d.xquat[body]); mujoco.mju_mulQuat(qe, q_t, qinv); mujoco.mju_quat2Vel(rot, qe, 1.0)
    return dict(foot=side, err_mm=round(err_mm, 1), err_deg=round(float(np.degrees(np.linalg.norm(rot))), 1),
                leg_q=[round(float(v), 3) for v in d.qpos[qa]], tread_z=round(float(c[2]), 3))


def body_joint_qpos(sonic_env) -> np.ndarray:
    """The robot's 29 body joints as they stand now (MJCF order): the pose the quick start holds (agent.quick_start(hold_real=...))."""
    m, d, robot = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData, sonic_env.task.robot
    return d.qpos[m.jnt_qposadr[robot.body_joint_index]].copy()


def plate_in_hand(m, d, plate: str, side: str, max_m: float = 0.25) -> tuple[bool, float, bool]:
    """(held, distance hand base -> plate centre, a hand geom touches the plate): the check scripts' view of the grasp."""
    import mujoco
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base_link")
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLATE_BODIES[plate])
    dist = float(np.linalg.norm(d.xpos[pb] - d.xpos[hb]))
    hand = set()
    for b in range(m.nbody):
        k = b
        while k > 0:
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, k) or "").startswith(f"{side}_hand_"):
                hand.add(b); break
            k = int(m.body_parentid[k])
    touch = any((int(m.geom_bodyid[c.geom1]) == pb and int(m.geom_bodyid[c.geom2]) in hand) or (int(m.geom_bodyid[c.geom2]) == pb and int(m.geom_bodyid[c.geom1]) in hand)
                for c in d.contact[:d.ncon])
    return dist <= max_m and touch, dist, touch


LATCH_TAKE_MARGIN = 0.05                                                     # a held plate is handed to the operator when their hand command is at least this close (curl, 0..1) to the grasp's


class GripLatch:
    """Keeps a hand's command at its start pose until the operator takes over: the drawn start pose of a free hand the preset
    posed (``hands``) until that hand's command is anything but open (grip, trigger, or the first tracked hand pose); the
    hand that starts with the plate until the operator has CLOSED their hand at least as far as the grasp (hand_curl within
    LATCH_TAKE_MARGIN of the latch target's: the grip button, or a tracked hand closing on the plate). With hand tracking
    (--hand-source skeleton) a command is never exactly open, so the old "anything but open" rule handed the plate over on
    the first tracked frame and the open tracked hand dropped it (A3 smoke, 2026-11-05)."""

    def __init__(self) -> None:
        self.held: dict[str, bool] = {}
        self.target: dict[str, np.ndarray] = {}
        self.kind: dict[str, str] = {}

    def arm(self, preset: dict | None) -> None:
        from simple.robots.g1_sonic_revo2 import OPEN_Q
        P = preset or {}
        self.held, self.target, self.kind = {}, {}, {}
        for side, q in (P.get("hands") or {}).items():
            q = np.asarray(q, dtype=np.float64)[:HAND_W]
            if not np.allclose(q, OPEN_Q[:HAND_W], atol=1e-3):           # an open start needs no latch
                self.held[side], self.target[side], self.kind[side] = True, q, "pose"
        H = P.get("plate_in_hand")
        if H:
            self.held[H["side"]], self.target[H["side"]], self.kind[H["side"]] = True, held_latch_target(H), "plate"   # the closed grip, or the grasp's own recorded command

    def takes(self, side: str, q) -> bool:
        """Has the operator taken this hand: a free hand on any command but open; a held plate once the hand is closed at least as far as the grasp."""
        from simple.robots.g1_sonic_revo2 import OPEN_Q
        q = np.asarray(q, dtype=np.float64)[:HAND_W]
        if self.kind.get(side) == "plate":
            return hand_curl(q) >= hand_curl(self.target[side]) - LATCH_TAKE_MARGIN
        return not np.allclose(q, OPEN_Q[:HAND_W], atol=1e-3)

    def apply(self, action) -> list[str]:
        """Rewrite the action's hand command in place; returns the sides latched on this tick."""
        if not any(self.held.values()):
            return []
        params = getattr(action, "parameters", action)
        out = []
        for side in [s for s, h in self.held.items() if h]:
            key = f"{side}_hand_q"
            if not self.takes(side, params[key]):                       # the operator has not taken the hand yet: hold the start pose
                params[key] = self.target[side].astype(np.float32).copy(); out.append(side)
            else:                                                       # the operator has the hand from now on
                self.held[side] = False
        return out

    def active(self) -> list[str]:
        return [s for s, h in self.held.items() if h]
