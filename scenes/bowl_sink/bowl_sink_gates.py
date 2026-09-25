"""Four staged gates for the bowl -> sink task, shared by the SIMPLE task (live) and the replay (per episode).

    1. at_bowl   the robot has moved to the bowl and stopped there: walked at least MIN_WALK_BOWL since the start, the
                 bowl is within AT_BOWL_RADIUS of the pelvis and within AT_BOWL_BEARING of the heading, and the base has
                 been still (speed < STILL_SPEED) for HOLD_S.  Must happen before the grasp.
    2. grasped   the right hand is on the bowl and the bowl has been lifted GRASP_LIFT above where it started, and stays
                 lifted with the hand on it for GRASP_HOLD_S (a knock does not count).
    3. at_basin  after grasping: walked at least MIN_WALK_BASIN since the grasp, the basin centre within AT_BASIN_RADIUS
                 of the pelvis and within AT_BASIN_BEARING of the heading, base still for HOLD_S.
    4. placed    after at_basin: the hand has let go, the bowl's centre is inside the basin (footprint shrunk by the bowl's
                 base radius) and below the rim.  success = placed.  The gates are strictly ordered: each one can only
                 latch after the previous one has.  `settled` is added at the end if the bowl also came
                 to rest there (speed < STILL_SPEED for HOLD_S).

Every gate latches with the time it was met.  Diagnostics: max_lift, tilt at grasp, min pelvis-to-bowl / -to-basin
distances, where the bowl ended relative to the basin, and which gate the episode stalled at.

The class is pure numpy: callers feed it poses per step (`update`), or a replay trace (`from_trace`, which uses the
palm-to-bowl distance as the contact proxy because traces carry no contact list).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GateCfg:
    step_dt: float = 0.02            # env.step period (50 Hz)
    still_speed: float = 0.15        # m/s: "stopped"
    hold_s: float = 0.40             # how long the base must be still to count as stopped
    min_walk_bowl: float = 0.05      # m: the robot must have moved at all to "move to the bowl"
    at_bowl_radius: float = 0.60     # m: pelvis -> bowl (the bowl sits at the counter edge; the hand reaches ~0.32 m)
    at_bowl_bearing: float = 60.0    # deg: bowl within this of the heading
    grasp_lift: float = 0.05         # m above the bowl's start height
    grasp_hold_s: float = 0.30       # lifted with the hand on it for this long
    min_walk_basin: float = 0.20     # m walked since the grasp (the robot has to leave the bowl counter for the sink)
    at_basin_radius: float = 0.80    # m: pelvis -> basin centre
    at_basin_bearing: float = 75.0   # deg
    contact_proxy_dist: float = 0.13 # m palm -> bowl centre that counts as "hand on the bowl" when no contacts are given
    basin_len: float = 0.54          # along the sink counter (x)
    basin_across: float = 0.33       # across (y)
    bowl_base_r: float = 0.045       # the bowl's base radius: its centre must be this far inside the basin walls
    sink_h: float = 0.86             # rim height: the bowl's centre must be below it to be "in"


class BowlSinkGates:
    ORDER = ("at_bowl", "grasped", "at_basin", "placed")

    def __init__(self, cfg: GateCfg | None = None):
        self.cfg = cfg or GateCfg()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        c = self.cfg
        self.P = dict(at_bowl=False, grasped=False, at_basin=False, placed=False, settled=False, success=False,
                      t_at_bowl=None, t_grasped=None, t_at_basin=None, t_placed=None,
                      max_lift=0.0, tilt_at_grasp=None, min_pelvis_to_bowl=None, min_pelvis_to_basin=None,
                      bowl_end_minus_basin=None, stalled_at=None, hand_on_bowl_s=0.0, steps=0)
        self._start_xy = None; self._bowl_z0 = None; self._grasp_xy = None
        self._still = 0; self._held = 0; self._settle = 0
        self._last_xy = None; self._last_bowl = None

    def _bearing_ok(self, pelvis_xy, yaw, target_xy, limit_deg):
        dx, dy = target_xy[0] - pelvis_xy[0], target_xy[1] - pelvis_xy[1]
        rel = math.degrees(math.atan2(dy, dx) - yaw)
        rel = (rel + 180.0) % 360.0 - 180.0
        return abs(rel) <= limit_deg

    # ------------------------------------------------------------------ one step
    def update(self, t: float, pelvis_xy, yaw: float, bowl_xyz, basin_xyz, hand_on_bowl: bool,
               base_speed: float | None = None, bowl_tilt_deg: float | None = None, bowl_speed: float | None = None):
        """Feed one control step.  pelvis_xy / bowl_xyz / basin_xyz in the same world frame; yaw = heading (rad)."""
        c, P = self.cfg, self.P
        pelvis_xy = np.asarray(pelvis_xy, float)[:2]; bowl = np.asarray(bowl_xyz, float); basin = np.asarray(basin_xyz, float)
        P["steps"] += 1
        if self._start_xy is None:
            self._start_xy = pelvis_xy.copy(); self._bowl_z0 = float(bowl[2])
        if base_speed is None:                        # derive from the last position if the caller has no velocity
            base_speed = 0.0 if self._last_xy is None else float(np.hypot(*(pelvis_xy - self._last_xy)) / c.step_dt)
        if bowl_speed is None:
            bowl_speed = 0.0 if self._last_bowl is None else float(np.linalg.norm(bowl - self._last_bowl) / c.step_dt)
        self._last_xy, self._last_bowl = pelvis_xy.copy(), bowl.copy()
        self._still = self._still + 1 if base_speed < c.still_speed else 0
        still = self._still * c.step_dt >= c.hold_s
        walked = float(np.hypot(*(pelvis_xy - self._start_xy)))
        lift = float(bowl[2] - self._bowl_z0)
        P["max_lift"] = max(P["max_lift"], round(lift, 4))
        if hand_on_bowl:
            P["hand_on_bowl_s"] = round(P["hand_on_bowl_s"] + c.step_dt, 2)
        d_bowl = float(np.hypot(*(pelvis_xy - bowl[:2]))); d_basin = float(np.hypot(*(pelvis_xy - basin[:2])))
        P["min_pelvis_to_bowl"] = round(d_bowl, 3) if P["min_pelvis_to_bowl"] is None else min(P["min_pelvis_to_bowl"], round(d_bowl, 3))
        P["min_pelvis_to_basin"] = round(d_basin, 3) if P["min_pelvis_to_basin"] is None else min(P["min_pelvis_to_basin"], round(d_basin, 3))
        t = round(float(t), 2)

        # 1. moved to the bowl and stopped there (before the grasp)
        if not P["at_bowl"] and not P["grasped"]:
            if walked >= c.min_walk_bowl and d_bowl <= c.at_bowl_radius and still and self._bearing_ok(pelvis_xy, yaw, bowl[:2], c.at_bowl_bearing):
                P["at_bowl"], P["t_at_bowl"] = True, t
        # 2. grasped: hand on the bowl and lifted, held
        if not P["grasped"]:
            self._held = self._held + 1 if (hand_on_bowl and lift >= c.grasp_lift) else 0
            if self._held * c.step_dt >= c.grasp_hold_s and P["at_bowl"]:
                P["grasped"], P["t_grasped"] = True, t
                self._grasp_xy = pelvis_xy.copy()
                P["tilt_at_grasp"] = None if bowl_tilt_deg is None else round(float(bowl_tilt_deg), 1)
        # 3. moved to the basin and stopped there (after the grasp)
        if P["grasped"] and not P["at_basin"]:
            since_grasp = float(np.hypot(*(pelvis_xy - self._grasp_xy)))
            if since_grasp >= c.min_walk_basin and d_basin <= c.at_basin_radius and still and self._bearing_ok(pelvis_xy, yaw, basin[:2], c.at_basin_bearing):
                P["at_basin"], P["t_at_basin"] = True, t
        # 4. placed: let go, inside the basin, below the rim
        inside = (abs(bowl[0] - basin[0]) <= c.basin_len / 2 - c.bowl_base_r and abs(bowl[1] - basin[1]) <= c.basin_across / 2 - c.bowl_base_r
                  and bowl[2] < c.sink_h)
        if P["at_basin"] and not P["placed"] and not hand_on_bowl and inside:
            P["placed"], P["t_placed"] = True, t
            P["success"] = True
        if P["placed"]:
            self._settle = self._settle + 1 if (inside and bowl_speed < c.still_speed) else 0
            P["settled"] = self._settle * c.step_dt >= c.hold_s
        P["bowl_end_minus_basin"] = [round(float(bowl[i] - basin[i]), 3) for i in range(3)]
        P["stalled_at"] = next((g for g in self.ORDER if not P[g]), None)
        return P

    def reward(self) -> float:
        P = self.P
        return 1.0 if P["placed"] else 0.25 * (float(P["at_bowl"]) + float(P["grasped"]) + float(P["at_basin"]))

    def report(self) -> str:
        P = self.P
        parts = [f"{g}{'@%.1fs' % P['t_' + g] if P[g] else ' -'}" for g in self.ORDER]
        return " | ".join(parts) + (f" | settled" if P["settled"] else "") + f" | lift {P['max_lift']:.2f} m | stalled at {P['stalled_at']}"

    # ------------------------------------------------------------------ replay traces
    @classmethod
    def from_trace(cls, trace, basin_xyz, cfg: GateCfg | None = None, bowl_key="target_xyz", palm_key="palm_r",
                   tip_key="index_tip_r", xy_key="sim_xy", yaw_key="sim_heading_rel", tilt_key="target_tilt_deg", heading0=0.0, xy0=(0.0, 0.0)):
        """Evaluate a replay trace (list of per-tick dicts; entries with a 'summary' key are skipped).
        Contact is not recorded, so 'hand on the bowl' = palm or index tip within contact_proxy_dist of the bowl centre."""
        g = cls(cfg)
        c = g.cfg
        basin = np.asarray(basin_xyz, float)
        for r in trace:
            if "summary" in r:
                continue
            bowl = np.asarray(r[bowl_key], float)
            palm = np.asarray(r[palm_key], float); tip = np.asarray(r.get(tip_key, r[palm_key]), float)
            on = min(np.linalg.norm(palm - bowl), np.linalg.norm(tip - bowl)) <= c.contact_proxy_dist
            g.update(r["t"], np.asarray(r[xy_key], float) + np.asarray(xy0, float), heading0 + float(r[yaw_key]), bowl, basin, on, bowl_tilt_deg=r.get(tilt_key))
        return g

    def to_dict(self):
        return dict(self.P, cfg=asdict(self.cfg))


class ContactProbe:
    """Live contact test: does any right-hand geom touch the bowl?  Binds geom ids once per MuJoCo model."""

    def __init__(self, m, target_body: str, hand_prefix: str = "right_hand_"):
        import mujoco
        self.model_id = id(m)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_body)
        if bid < 0:
            raise KeyError(f"no body named {target_body!r}")
        names = {b: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(m.nbody)}
        self.target_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
        self.hand_geoms = {g for g in range(m.ngeom) if names[m.geom_bodyid[g]].startswith(hand_prefix)}

    def hand_on(self, d) -> bool:
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.target_geoms and g2 in self.hand_geoms) or (g2 in self.target_geoms and g1 in self.hand_geoms):
                return True
        return False


def yaw_wxyz(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
