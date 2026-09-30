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
import random
from copy import deepcopy
from typing import Any

import numpy as np

SETUP_VERSION = 1

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

_CART_X_KEYS = ("handle_x", "deck_near_x", "deck_far_x", "cart_cx", "box_near_x", "box_far_x", "cup_x")
_CART_Y_KEYS = ("cart_cy", "box_right_y", "box_left_y", "cup_y")


def _yaw_quat(yaw_deg: float) -> list[float]:
    h = math.radians(yaw_deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]                            # w, x, y, z


def _r(v: float) -> float:
    return round(float(v), 4)                                               # 0.1 mm: exact, readable values in the metadata


class SceneSetups:
    """Draws and applies setups for one live scene module + task."""

    def __init__(self, scene: str, module, task, max_distractors: int | None = None, ranges: str = "teleop") -> None:
        if ranges not in RANGE_SETS:
            raise ValueError(f"unknown ranges {ranges!r}; one of {sorted(RANGE_SETS)}")
        if scene not in RANGE_SETS[ranges]:
            raise ValueError(f"no randomization ranges for scene {scene!r}")
        self.scene, self.mod, self.task, self.range_set = scene, module, task, ranges
        self.ranges = deepcopy(RANGE_SETS[ranges][scene])
        if max_distractors is not None:
            lo, hi = self.ranges["distractors"]
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
                  "coffee_cart": ("cup_x", "cup_y")}[self.scene]
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
            spatial.cfg.robot_region = Box(low=[r["x"], r["y"], 0.0], high=[r["x"], r["y"], 0.0])
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
        if setup.get("seed") is not None:
            np.random.seed(int(setup["seed"]) % (2 ** 32))                 # SIMPLE's DR draws from both
            random.seed(int(setup["seed"]))

    @staticmethod
    def describe(setup: dict[str, Any]) -> str:
        r = setup.get("robot", {})
        parts = [f"robot ({r.get('x', 0):+.3f}, {r.get('y', 0):+.3f}) {r.get('yaw_deg', 0):+.1f} deg",
                 f"target ({setup['target_xy'][0]:.3f}, {setup['target_xy'][1]:+.3f})"]
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


def dumps(setup: dict[str, Any]) -> str:
    return json.dumps(setup, sort_keys=True)
