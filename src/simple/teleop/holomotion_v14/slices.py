"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Slice recording for the long-horizon kitchen task (2026-10-30, docs/long_horizon_slicing; user: "provide me the teleop cmd
and scene following your S1 scene and rules for me to test it first"; S2-S6 2026-11-01; rebuilt 2026-11-02 on kitchen7
episode 1 as the demo -- user: "use these as demo slicing, replace the kitchen5 one" -- docs/long_horizon_slicing_scenes).

A slice is one state-defined piece of the load-dishwasher task recorded as its own episode: every setup starts in the
slice's start state (scene_setup.SLICE_STARTS draws it, the kitchen kit's SLICE_PRESET puts it into the scene), the
operator plays until the slice's end event, and ``overlap_s`` after the event the slice is complete -- the headset says
"save now" and the operator saves with left menu + A (user 2026-10-30: no automatic stop or reset; ``--slice-auto-cut``
restores the automatic save). ``realformat.success`` of a slice episode means its end event was reached; the task's own
gates are evaluated by the replay (replay_holomotion_v14 --gates) as for every recording.

The seven slices follow kitchen7 ep1's order and chain: each starts where the previous one ends, the plate already in the
hand where ep1 had it (preset plate_in_hand, grasp_start.hold_plate; user 2026-11-02: "S2 should be S1 continued work"):

    S1  Pick up a plate                 start in front of the plates at the counter; door closed, rack in, lid closed.
                                        End: a counter plate lifted >= 3 cm above where it rested, held for 0.5 s.
    S2  Carry to the bin, wipe          start on ep1's carry path 0.3-1.4 m before the bin, plate_a in the right hand, lid closed.
                                        End: lid >= 1.3 rad and a plate over the open bin above its rim, held for 0.5 s
                                        (the swipe passes are labelled by the annotation, not here).
    S3  Carry to the dishwasher, door   start on ep1's bin -> door carry, plate_a in the right hand, lid open, door closed.
                                        End: door >= 1.2 rad while a hand holds a plate, then no hand on the door for 0.5 s
                                        (ep1 passed the plate to the left hand and pulled with the right).
    S4  Pull the rack out, load         start beside the open door facing it, plate_a in the left hand, rack in.
                                        End: a plate newly inside the upper rack's box, then no hand on any plate for 0.5 s.
    S5  Fetch the second plate          start on ep1's dishwasher -> counter walk, door open, rack out, plate_a in the rack.
                                        End: a counter plate lifted >= 3 cm, held 0.5 s (the plate in the rack does not count).
    S6  Load the second plate           start on the counter -> door path, plate_b in the right hand, plate_a in the rack.
                                        End: as S4.
    S7  Push the rack in, close         start beside the open door, rack out, 1-2 plates in it.
                                        End: both racks within 5 cm of home and the door within 5 deg of shut, then no
                                        hand on the door for 0.5 s.

The end checks read the joints, body poses and contacts of the compiled scene directly (numpy, no task gates in the loop)
and are advisory: they tell the operator when the slice is done. The 2 s overlap is recorded after the event. Plates
already standing in the rack when the recording starts never count as lifted / loaded (2026-11-02: before that a
half-loaded start fired the old S5's gate on frame 0).

Atomic slices A1-A6 (2026-11-02 evening, docs/atomic_slicing_scenes): S1 + S2 -- the task up to the first plate wiped clean --
cut at the annotation's own action boundaries of kitchen7 ep1 (caption.json "Current Action"):

    A1  Move to the counter           start 1-2 steps back from the counter, upper body drawn. End: a counter plate 0.15-0.55 m
                                      ahead of the pelvis and within 0.30 m sideways, the pelvis slower than 0.15 m/s, for 0.5 s
                                      (ep1 0-3.2 s; whole-body locomotion is its own dominant skill, as the carry to the bin is).
    A2  Pick the plate up             start 7 of 10 where ep1 stood for the pick (0.40-0.50 m from the plate), upper body drawn, 3 of 10
                                      in ep1's whole-body pose at 16.4 s (hands at the rim). End: a plate 10 cm up for 0.5 s;
                                      milestones a hand on a plate, lift onset (3 cm)  (ep1 3.2-29.6 s).
    A3  Move the plate to the bin     start at the counter, the plate in a hand. End: the bin 0.2-0.65 m ahead of the pelvis, within
                                      0.35 m sideways, the pelvis slower than 0.15 m/s, for 0.5 s  (29.6-51.5 s).
    A4  Step on the pedal             start at the bin stance, the plate in a hand, lid closed. End: lid >= 1.3 rad for 0.3 s  (51.5-51.9 s).
                                      (skill "Step": a lower-body skill we add to RoboAtom's hand vocabulary; its "Press" is a hand action)
                                      The pedal springs back (KITCHEN_PEDAL_RETURN=1), so the foot has to stay on it.
    A5  Wipe the plate                start in ep1's swipe stance and pose, the plate in the right hand over the open bin, one foot on
                                      the pressed pedal (leg IK onto the tread). End: 3 passes of the free hand over the held plate
                                      while it is over the open bin (a pass: the nearest fingertip within 16 cm of the plate centre,
                                      then out past 24 cm; ep1's three passes came within 11-15 cm and went out to 27-41 cm, never
                                      touching the plate)  (51.9-55.5 s).
    The skill names and the one-segment-per-dominant-skill rule follow RoboAtom (arXiv 2610.40207): arm reaching, lifting and
    alignment belong to the neighbouring manipulation, so both Pick tiers run to the plate held up; walking to the counter is
    whole-body locomotion, a dominant skill of its own (Move), like the carry to the bin.

2026-11-02 fix: the bin footprint for "over the bin" was read from the wall geoms' body-local positions (0.8 m off); it is the
world footprint now (plate_over_bin, at_bin, swipe_passes).
"""

from __future__ import annotations

import os

import numpy as np

HZ = 50                                                                     # recorded frames per second (the 20 ms control tick)
import os as _os
# KITCHEN_SLICE_GATE_TEST_S=<seconds>: the end gate counts as met from that recording time on, whatever the scene does. For
# headless tests of --auto-cycle only (a synthetic operator meets no gate); printed loudly by the CLI when set.
_GATE_TEST_FRAME = int(float(_os.environ["KITCHEN_SLICE_GATE_TEST_S"]) * HZ) if _os.environ.get("KITCHEN_SLICE_GATE_TEST_S") else None
RACK_BOX = dict(x=0.25, y=0.24, z=(-0.03, 0.30))                           # a dish centre inside the upper rack, rack body frame (kitchen_task.RACK_BOX)
PLATE_BODIES = ("k_plate_a", "k_plate_b")

SLICES = {
    "S1": dict(name="Pick up a plate from the counter", instruction="pick up a plate from the counter",
               end_gate="plate_lifted", gate=("plate_lifted", 0.03), hold_s=0.5, overlap_s=2.0,
               end_text="a counter plate lifted 3 cm or more, held for 0.5 s"),
    "S2": dict(name="Carry the plate to the pedal bin and wipe it clean",
               instruction="carry the plate to the pedal bin, step on the pedal and wipe the plate clean over the open bin",
               end_gate="plate_over_open_bin", gate=("plate_over_bin", 1.3), hold_s=0.5, overlap_s=2.0,
               end_text="bin lid open (pedal) with a plate held over the bin above its rim, for 0.5 s"),
    "S3": dict(name="Carry the plate to the dishwasher and open the door",
               instruction="carry the plate to the dishwasher and open the door with the plate in hand",
               end_gate="door_open_holding", gate=("door_ge_holding", 1.2), hands_off="k_dishwasher_door", hands_off_s=0.5, overlap_s=2.0,
               end_text="door open past 69 deg while a hand holds a plate, then no hand on the door for 0.5 s"),
    "S4": dict(name="Pull the rack out and load the plate",
               instruction="pull the upper rack out and stand the plate in it",
               end_gate="plate_in_rack", gate=("plate_in_rack",), hands_off="plates", hands_off_s=0.5, overlap_s=2.0,
               end_text="a plate newly inside the upper rack, then no hand on any plate for 0.5 s"),
    "S5": dict(name="Fetch the second plate", instruction="walk to the counter and pick up the remaining plate",
               end_gate="plate_lifted", gate=("plate_lifted", 0.03), hold_s=0.5, overlap_s=2.0,
               end_text="a counter plate lifted 3 cm or more, held for 0.5 s"),
    "S6": dict(name="Load the second plate",
               instruction="carry the plate to the dishwasher and stand it in the upper rack next to the first",
               end_gate="plate_in_rack", gate=("plate_in_rack",), hands_off="plates", hands_off_s=0.5, overlap_s=2.0,
               end_text="a plate newly inside the upper rack, then no hand on any plate for 0.5 s"),
    "S7": dict(name="Push the rack in and close the door", instruction="push the upper rack back into the dishwasher and close the door",
               end_gate="door_closed", gate=("racks_in_door_shut", 0.05, 5.0), hands_off="k_dishwasher_door", hands_off_s=0.5, overlap_s=2.0,
               end_text="both racks within 5 cm of home and the door within 5 deg of shut, then no hand on the door for 0.5 s"),
    # atomic slices of S1 + S2 (docs/atomic_slicing_scenes), RoboAtom skills (arXiv 2610.40207), one scene per skill: Move to the
    # counter, Pick (two start kinds drawn inside the scene), Move to the bin, Step (foot), Wipe; the annotation's finer events are
    # milestones (first frame each, in the episode meta); overlap 1 s. The bin scenes run with KITCHEN_PEDAL_RETURN=1 (env).
    "A1": dict(name="Move to the counter", skill="Move", instruction="walk up to the counter and stand in front of the plate",
               end_gate="at_counter", gate=("at_counter", 0.15, 0.55, 0.30, 0.15), hold_s=0.5, overlap_s=1.0,
               end_text="a counter plate 0.15-0.55 m ahead of the pelvis and within 0.30 m sideways, the robot standing still (< 0.15 m/s), for 0.5 s"),
    "A2": dict(name="Pick the plate up", skill="Pick", instruction="pick up the plate from the counter",
               end_gate="plate_up", gate=("plate_lifted", 0.10), hold_s=0.5, overlap_s=1.0,
               clean_start=True,          # user 2026-11-03: the robot must not touch anything at the beginning (A2 only, not the other slices)
               start_mode="hold",         # user 2026-11-04 13:30: arms close to the A1 end pose, standing, as close to the counter as possible -> the held
                                          # start (motion tracking on the recorded joints); "walk" = walking mode moved back from the counter (CLI --no-start-hold)
               milestones=(("hand_on_plate", ("hand_on_plate",)), ("lift_onset", ("plate_lifted", 0.03))),
               end_text="the plate 10 cm or more above where it rested, for 0.5 s"),
    "A3": dict(name="Move the plate to the pedal bin", skill="Move", instruction="carry the plate to the pedal bin and stand in front of it",
               end_gate="at_bin", gate=("at_bin", 0.20, 0.65, 0.35, 0.15), hold_s=0.5, overlap_s=1.0, env={"KITCHEN_PEDAL_RETURN": "1"},
               start_mode="hold",         # 2026-11-05: the plate held level as in A2 ep41 with the arm raised ahead (since 2026-11-10 by default a recorded A2
                                          # press-and-grasp end, scene_setup KITCHEN_A3_START=a2_press) -> the placed pose is kept by motion
                                          # tracking until the operator's Y (as A2); walking mode would swing the arm to its hanging default and spill the plate
               hold_pin="held_arm",       # ... and the holding arm stays pinned at its placed joints through the hold (the motion policy holds a loaded arm
                                          # 5-10 deg low at its 14 N m/rad), fading to the operator over the 0.5 s hand-over
               end_text="the bin 0.2-0.65 m ahead of the pelvis and within 0.35 m sideways, the robot standing still (< 0.15 m/s), for 0.5 s"),
    "A4": dict(name="Step on the bin pedal", skill="Step", instruction="step on the pedal and hold it down so the bin lid opens",      # a foot skill (lower body), not RoboAtom's hand Press (user 2026-11-03)
               end_gate="lid_open", gate=("lid_ge", 1.3), hold_s=0.3, overlap_s=1.0, env={"KITCHEN_PEDAL_RETURN": "1"},
               **(dict(start_mode="hold", hold_pin="held_arm") if os.environ.get("KITCHEN_A4_START", "a3_end") == "a3_end" else {}),   # 2026-11-10 night: a recorded
               # A3 end (the plate raised in the right hand at the bin) kept by motion tracking until Y, the holding arm pinned, as A3
               end_text="the bin lid open past 1.3 rad for 0.3 s (the foot holding the pedal down)"),
    "A5": dict(name="Wipe the plate over the bin", skill="Wipe", instruction="keep the foot on the pedal, hold the plate over the open bin and swipe it clean with the other hand",
               end_gate="swiped", gate=("swipe_passes", 3, 1.3, 0.16, 0.24), hold_s=0.0, overlap_s=1.0, env={"KITCHEN_PEDAL_RETURN": "1"},
               **(dict(start_mode="hold", hold_pin="held_arm") if os.environ.get("KITCHEN_A5_START", "a4_end") == "a4_end" else {}),   # 2026-11-10 night: a recorded
               # A4 end (right foot on the pedal, lid open, plate in the right hand) kept by motion tracking until Y, the holding arm pinned
               end_text="three passes of the free hand over the held plate while it is over the open bin (lid up: the foot on the pedal)"),
}


def _hand_side(m, body: int) -> str:
    """"left" / "right" when the body is part of a hand (bodies left_hand_* / right_hand_*: Dex3 and Revo 2 alike), else ""."""
    import mujoco
    while body > 0:
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if n.startswith("left_hand_"):
            return "left"
        if n.startswith("right_hand_"):
            return "right"
        body = int(m.body_parentid[body])
    return ""


def _subtree_geoms(m, bodies: list[int]) -> np.ndarray:
    sub = np.zeros(m.nbody, dtype=bool)
    for b in range(m.nbody):
        k = b
        while k > 0:
            if k in bodies:
                sub[b] = True
                break
            k = int(m.body_parentid[k])
    return sub[np.asarray(m.geom_bodyid)]


class SliceCut:
    """Watches one recording for the slice's end event and says when to cut.

    ``bind(model)`` after every env.reset (the compiled model changes between episodes: plate swap), ``reset()`` when a
    recording starts or is abandoned, ``update(progress, data, frame)`` once per recorded frame.
    """

    def __init__(self, slice_id: str) -> None:
        if slice_id not in SLICES:
            raise ValueError(f"unknown slice {slice_id!r}; built: {', '.join(SLICES)}")
        self.id, self.cfg = slice_id, SLICES[slice_id]
        self.hands_off_ticks = int(round(self.cfg.get("hands_off_s", 0.0) * HZ))
        self.hold_ticks = int(round(self.cfg.get("hold_s", 0.0) * HZ))
        self.overlap_ticks = int(round(self.cfg["overlap_s"] * HZ))
        self.hand = self.target = None
        self.reset()

    # ------------------------------------------------------------------ binding to the compiled scene
    def bind(self, m) -> None:
        import mujoco
        self.m = m
        sides = [_hand_side(m, int(m.geom_bodyid[g])) for g in range(m.ngeom)]
        self.hand = np.array([bool(s) for s in sides])
        self.hand_side = np.array([{"": 0, "left": 1, "right": 2}[s] for s in sides], dtype=np.int8)
        self.tips = {side: [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{side}_hand_")
                            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("_tip_link")] for side in ("left", "right")}
        self.b_pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.v_root = int(m.jnt_dofadr[m.body_jntadr[int(m.body_rootid[self.b_pelvis])]]) if self.b_pelvis >= 0 else None

        def jq(name):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            return int(m.jnt_qposadr[j]) if j >= 0 else None

        def bid(name):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            return b if b >= 0 else None

        self.q_door, self.q_upper, self.q_lower, self.q_lid = jq("k_dishwasher_door"), jq("k_dishwasher_upper_rack"), jq("k_dishwasher_lower_rack"), jq("k_bin_lid")
        self.b_rack = bid("k_dishwasher_upper_rack")
        self.b_plates = [b for b in (bid(n) for n in PLATE_BODIES) if b is not None]
        self.plate_geoms = _subtree_geoms(m, self.b_plates)
        ho = self.cfg.get("hands_off")
        if ho == "plates":
            self.target = _subtree_geoms(m, self.b_plates)
        elif ho:
            tb = bid(ho)
            if tb is None:
                raise RuntimeError(f"slice {self.id}: body {ho!r} is not in the scene")
            self.target = _subtree_geoms(m, [tb])
        else:
            self.target = None
        # the pedal bin's footprint and rim from its wall geoms: world positions (the bin is a static body, so a forward pass at
        # qpos0 gives them; m.geom_pos is body-local and sat 0.8 m off until 2026-11-02 evening)
        walls = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("k_bin_wall")]
        if walls:
            d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
            p = np.asarray(d0.geom_xpos)[walls]
            self.bin_lo, self.bin_hi = p[:, :2].min(0) - 0.05, p[:, :2].max(0) + 0.05
            self.bin_rim = float((p[:, 2] + np.asarray(m.geom_size)[walls][:, 2]).max())
            self.bin_xy = p[:, :2].mean(0)
        else:
            self.bin_lo = self.bin_hi = self.bin_rim = self.bin_xy = None
        self.reset()

    def reset(self) -> None:
        self.gate_frame = self.event_frame = self.cut_frame = None
        self.off_ticks = self.hold_count = 0
        self.touching = False
        self.plate_z0 = self.plate_free = None
        self.passes, self.pass_in = 0, False                              # swipe_passes: passes counted, the free hand currently in
        self.milestones: dict[str, int] = {}                              # first frame of each milestone event (cfg "milestones")
        self._test_frame = -1

    # ------------------------------------------------------------------ the robot, for the stance gates
    def _in_front(self, d, xy, lo: float, hi: float, side: float) -> bool:
        """A world point `lo`..`hi` m ahead of the pelvis and within `side` m of its centre line (pelvis yaw)."""
        p = d.xpos[self.b_pelvis]; R = d.xmat[self.b_pelvis].reshape(3, 3)
        f = np.array([R[0, 0], R[1, 0]]); f /= max(np.linalg.norm(f), 1e-9)
        v = np.asarray(xy, dtype=float) - p[:2]
        ahead, left = float(f @ v), float(-f[1] * v[0] + f[0] * v[1])
        return lo <= ahead <= hi and abs(left) <= side

    def _pelvis_speed(self, d) -> float:
        return float(np.linalg.norm(d.qvel[self.v_root:self.v_root + 2])) if self.v_root is not None else 0.0

    def _sides_on(self, d, mask) -> set[str]:
        """Which hands touch a geom of `mask`."""
        n = int(d.ncon)
        if n == 0 or mask is None:
            return set()
        g1, g2 = np.asarray(d.contact.geom1[:n]), np.asarray(d.contact.geom2[:n])
        s = np.concatenate([self.hand_side[g1][mask[g2]], self.hand_side[g2][mask[g1]]])
        return {("", "left", "right")[int(k)] for k in np.unique(s) if k}

    def _over_bin(self, d, b: int) -> bool:
        p = d.xpos[b]
        return bool(self.bin_lo is not None and np.all(p[:2] >= self.bin_lo) and np.all(p[:2] <= self.bin_hi) and p[2] >= self.bin_rim)

    def _swipe_passes(self, d, lid_min: float, near: float, far: float) -> int:
        """Counts passes of the free hand over the held plate while it is over the open bin: the nearest fingertip of the hand
        not holding the plate comes within `near` of the plate centre (in), then goes out past `far` (one pass)."""
        if self.q_lid is None or float(d.qpos[self.q_lid]) < lid_min:
            self.pass_in = False
            return self.passes
        for b, free in zip(self.b_plates, self.plate_free):
            if not free or not self._over_bin(d, b):
                continue
            holding = self._sides_on(d, np.asarray(self.m.geom_bodyid) == b)
            if len(holding) != 1:
                continue
            other = "right" if "left" in holding else "left"
            tips = self.tips.get(other) or []
            if not tips:
                continue
            dist = min(float(np.linalg.norm(d.xpos[t] - d.xpos[b])) for t in tips)
            if not self.pass_in and dist <= near:
                self.pass_in = True
            elif self.pass_in and dist >= far:
                self.pass_in = False; self.passes += 1
            return self.passes
        self.pass_in = False
        return self.passes

    # ------------------------------------------------------------------ the end gate, from the scene state
    def _plate_rel_rack(self, d, b: int) -> np.ndarray:
        R = d.xmat[self.b_rack].reshape(3, 3)
        return R.T @ (d.xpos[b] - d.xpos[self.b_rack])

    def _in_rack(self, d, b: int) -> bool:
        if self.b_rack is None:
            return False
        r = self._plate_rel_rack(d, b)
        return bool(abs(r[0]) <= RACK_BOX["x"] and abs(r[1]) <= RACK_BOX["y"] and RACK_BOX["z"][0] <= r[2] <= RACK_BOX["z"][1])

    def _first_frame(self, d) -> None:
        """Where the plates are when the recording starts: their resting height, and which of them are not in the rack yet
        (a plate the preset already stood in the rack never counts as lifted or loaded)."""
        if self.plate_z0 is None:
            self.plate_z0 = np.array([float(d.xpos[b][2]) for b in self.b_plates])
            self.plate_free = np.array([not self._in_rack(d, b) for b in self.b_plates], dtype=bool)

    def _lifted(self, d, dz: float) -> np.ndarray:
        z = np.array([float(d.xpos[b][2]) for b in self.b_plates])
        return self.plate_free & (z >= self.plate_z0 + dz)

    def gate_now(self, d) -> bool:
        if _GATE_TEST_FRAME is not None:                                     # headless tests of the auto cycle (no operator to meet the gate)
            return self._test_frame >= _GATE_TEST_FRAME
        return self._gate(d, self.cfg["gate"])

    def _gate(self, d, spec) -> bool:
        kind, *arg = spec
        self._first_frame(d)
        if kind == "door_ge":
            return self.q_door is not None and float(d.qpos[self.q_door]) >= arg[0]
        if kind == "rack_ge":
            return self.q_upper is not None and float(d.qpos[self.q_upper]) >= arg[0]
        if kind == "racks_in_door_shut":
            racks = [q for q in (self.q_upper, self.q_lower) if q is not None]
            return (bool(racks) and all(float(d.qpos[q]) <= arg[0] for q in racks) and self.q_door is not None
                    and float(d.qpos[self.q_door]) <= np.radians(arg[1]))
        if kind == "plate_lifted":
            return bool(np.any(self._lifted(d, arg[0])))
        if kind == "door_ge_holding":                                      # the door open while a hand holds a plate (hand-plate contact)
            return self.q_door is not None and float(d.qpos[self.q_door]) >= arg[0] and self._touching(d, self.plate_geoms)
        if kind == "plate_over_bin":
            if self.q_lid is None or self.bin_lo is None or float(d.qpos[self.q_lid]) < arg[0]:
                return False
            return any(free and self._over_bin(d, b) for b, free in zip(self.b_plates, self.plate_free))
        if kind == "plate_in_rack":
            return any(free and self._in_rack(d, b) for b, free in zip(self.b_plates, self.plate_free))
        if kind == "at_counter":                                           # a free counter plate lo..hi m ahead, within side m, standing still
            lo, hi, side, vmax = arg
            return self._pelvis_speed(d) <= vmax and any(free and self._in_front(d, d.xpos[b][:2], lo, hi, side) for b, free in zip(self.b_plates, self.plate_free))
        if kind == "hand_on_plate":                                        # a hand touches a plate that is free (on the counter)
            mask = self.plate_geoms & np.isin(np.asarray(self.m.geom_bodyid), [b for b, free in zip(self.b_plates, self.plate_free) if free])
            return self._touching(d, mask)
        if kind == "at_bin":                                               # the bin lo..hi m ahead, within side m, standing still
            lo, hi, side, vmax = arg
            return self.bin_xy is not None and self._pelvis_speed(d) <= vmax and self._in_front(d, self.bin_xy, lo, hi, side)
        if kind == "lid_ge":
            return self.q_lid is not None and float(d.qpos[self.q_lid]) >= arg[0]
        if kind == "swipe_passes":
            n, lid_min, near, far = arg
            return self._swipe_passes(d, lid_min, near, far) >= n
        raise ValueError(f"slice {self.id}: unknown gate {kind!r}")

    def _touching(self, d, mask) -> bool:
        n = int(d.ncon)
        if n == 0 or self.hand is None or mask is None:
            return False
        g1, g2 = np.asarray(d.contact.geom1[:n]), np.asarray(d.contact.geom2[:n])
        return bool(np.any((self.hand[g1] & mask[g2]) | (self.hand[g2] & mask[g1])))

    def touching_target(self, d) -> bool:
        return self._touching(d, self.target)

    # ------------------------------------------------------------------ per frame
    def update(self, prog: dict, d, frame: int) -> str | None:
        """Once per recorded frame (``frame`` = its index). Returns "gate", "event" or "cut" on the tick it happens, else None."""
        self._test_frame = frame
        if _GATE_TEST_FRAME is not None:
            prog = {}                                                        # the forced gate, not the task's
        if self.cut_frame is not None:
            return None
        for name, spec in self.cfg.get("milestones", ()):                 # the annotation's finer events, labelled as they happen
            if name not in self.milestones and self._gate(d, spec):
                self.milestones[name] = frame
        out = None
        if self.event_frame is None:
            gate = prog.get(self.cfg["end_gate"]) if prog else None
            if gate is None:                                               # no live task gates (2026-10-30): the scene state decides
                gate = self.gate_now(d)
            if self.gate_frame is None:
                if not gate:
                    return None
                self.gate_frame, self.hold_count, out = frame, 1, "gate"
            elif self.target is None:                                      # a "held" gate: it must stay true for hold_s (else re-armed)
                if not gate:
                    self.gate_frame, self.hold_count = None, 0
                    return None
                self.hold_count += 1
            if self.target is not None:
                self.touching = self.touching_target(d)
                self.off_ticks = 0 if self.touching else self.off_ticks + 1
                if self.off_ticks >= self.hands_off_ticks:
                    self.event_frame, out = frame, "event"
            elif self.hold_count >= max(1, self.hold_ticks):
                self.event_frame, out = frame, "event"
            return out
        if frame - self.event_frame >= self.overlap_ticks:
            self.cut_frame = frame
            return "cut"
        return out

    # ------------------------------------------------------------------ words
    def _target_word(self) -> str:
        ho = self.cfg.get("hands_off", "")
        return "plate" if ho == "plates" else "rack" if "rack" in ho else "door"

    def after_gate_text(self) -> str:
        """What is still awaited once the gate fired (log lines)."""
        if self.target is None:
            return f"holding for {self.cfg.get('hold_s', 0):g} s"
        return f"waiting for the hands to leave the {self._target_word()}"

    def event_text(self) -> str:
        if self.target is None:
            return f"{self.cfg['end_gate']} held {self.cfg.get('hold_s', 0):g} s"
        return f"hands off the {self._target_word()}"

    def status(self, frame: int, auto: bool = False) -> tuple[str, bool]:
        """One headset / window line: (text, past the event). ``auto``: the recorder saves by itself at the cut."""
        g = self.cfg["end_gate"]
        second = "hands off" if self.target is not None else "held"
        tail = "-> auto-save" if auto else "-> then save (L-menu + A)"
        ms = "".join(f" {n}[{'x' if n in self.milestones else ' '}]" for n, _ in self.cfg.get("milestones", ()))
        if self.gate_frame is None:
            return f"{self.id}{ms} {g} [ ]  {second} [ ]  {tail}", False
        if self.event_frame is None:
            ok = (not self.touching) if self.target is not None else (self.hold_count >= self.hold_ticks)
            return f"{self.id} {g} [x]  {second} [{'x' if ok else ' '}]  {tail}", False
        left = max(0.0, (self.overlap_ticks - (frame - self.event_frame)) / HZ)
        if auto:
            return f"{self.id} done: saving in {left:.1f} s", True
        return (f"{self.id} done in {left:.1f} s" if left > 0 else f"{self.id} COMPLETE: save now (L-menu + A)"), True

    def meta(self, n_frames: int) -> dict:
        return dict(id=self.id, name=self.cfg["name"], skill=self.cfg.get("skill"), end_gate=self.cfg["end_gate"], end_text=self.cfg["end_text"], gate_frame=self.gate_frame,
                    event_frame=self.event_frame, hands_off_s=self.cfg.get("hands_off_s"), hold_s=self.cfg.get("hold_s"), overlap_s=self.cfg["overlap_s"],
                    cut_frame=self.cut_frame, frames=int(n_frames), complete=self.event_frame is not None, milestones=dict(self.milestones))

    def success(self, n_frames: int) -> dict:
        """For realformat: the slice succeeded when its end event was reached; success_frame = the end gate's frame."""
        ok = self.event_frame is not None
        return dict(success=ok, success_frame=self.gate_frame if ok else None)
