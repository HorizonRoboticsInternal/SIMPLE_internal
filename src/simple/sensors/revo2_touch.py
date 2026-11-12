"""BrainCo Revo 2 Touch (capacitive fingertip tactile sensors) emulated in MuJoCo.

The real hand has one capacitive unit under a pad cover on each of the five fingertips (nothing on the palm or the finger
pads). Per finger the SDK (`bc_stark_sdk` TouchFingerItem, ROS2 ros2_stark_msgs/TouchStatusItem) reports
    normal_force1          uint16, 0.01 N units, 0-25 N  (2500 counts at full scale, resolution 0.1 N)
    tangential_force1      uint16, 0.01 N units
    tangential_direction1  uint16, degrees 0-359, 0 = toward the fingertip, clockwise, 0xFFFF = no tangential force
    self_proximity1        uint32, capacitance counts, rises when an object is within ~1 cm
    status                 uint16, 0 normal / 1 abnormal / 2 communication error
and zeros in normal_force2/3, tangential_force2/3, tangential_direction2/3, self_proximity2, mutual_proximity.
Finger order thumb, index, middle, ring, pinky. Sampling >50 Hz (SDK polls at 100 Hz).

The simulated pads are the vendor's `*_touch_link.STL` covers, added to the robot by scripts/robots/add_revo2_touch.py
together with a box site at each pad's centre of curvature (z into the finger, x toward the fingertip), a `touch`
sensor, a `rangefinder` (site z out of the pad) and a `mujoco.sensor.touch_grid` plugin sensor (3 x 3 taxels, channels
= force on the pad along the site z, x, y). `Revo2Touch.read()` computes the SDK fields from the contacts on the pad
geom (exact normal and friction forces), `taxels()` returns the 3 x 3 grid, `physical()` the SI values.
Model assumptions, stated on the page docs/revo2_touch: direction is measured looking at the pad from outside (site x
toward the tip, y clockwise from it); proximity counts = PROX_FULL_COUNTS * (1 - d / 1 cm) for the distance d from the
pad surface (the real scale is not published; one field report gives ~529 000 counts at ~5 mm).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import mujoco

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COUNTS_PER_N = 100.0            # uint16 in 0.01 N
FORCE_MAX_N = 25.0              # sensor range 0-25 N; saturates at 2500 counts
MIN_TANGENTIAL_N = 0.1          # below the 0.1 N resolution the direction reads invalid
DIRECTION_INVALID = 0xFFFF
PROX_RANGE_M = 0.010            # proximity distance 0-1 cm (manual)
PROX_FULL_COUNTS = 1_058_000    # counts at contact; 529 000 at 5 mm (field report), linear in (1 - d / 1 cm)
STATUS_NORMAL, STATUS_ABNORMAL, STATUS_COMM_ERROR = 0, 1, 2
RATE_HZ = 100.0                 # the SDK's default poll rate; the sim can be read every physics step


def _sensor(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0: raise KeyError(f"sensor {name!r} not in the model; run scripts/robots/add_revo2_touch.py")
    return int(model.sensor_adr[sid]), int(model.sensor_dim[sid])


class Revo2Touch:
    """Reads the five fingertip sensors of one hand."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, side: str = "right", pads_json: str | Path | None = None):
        self.m, self.d, self.side = model, data, side
        radii = {}
        if pads_json is None:                              # the selected robot's folder first (2026-11-04: legacy or official hand), then the legacy one
            cands = []
            try:
                from simple.robots.g1_sonic_revo2 import G1SonicRevo2
                from simple.utils import get_data_dir
                cands.append(Path(get_data_dir()) / Path(G1SonicRevo2.mjcf_path).parent / "touch_pads.json")
            except Exception:
                pass
            cands.append(Path(__file__).resolve().parents[3] / "data/robots/g1_rot0_teleop_revo2/touch_pads.json")
            for cand in cands:
                if cand.exists(): pads_json = cand; break
        insets = {}
        if pads_json:
            pads = json.loads(Path(pads_json).read_text()); radii = {k: v["pad_radius_mm"] / 1000.0 for k, v in pads.items()}; insets = {k: v.get("ray_inset_m", 0.001) for k, v in pads.items()}
        self.fingers = []
        for f in FINGERS:
            base = f"{side}_hand_{f}_touch"
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{base}_collision")
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{base}_site")
            if geom < 0 or site < 0: raise KeyError(f"{base}: pad geom or site missing; run scripts/robots/add_revo2_touch.py")
            self.fingers.append(dict(name=f, geom=geom, body=int(model.geom_bodyid[geom]), site=site, total=_sensor(model, f"{base}_total"),
                                     prox=_sensor(model, f"{base}_prox"), grid=_sensor(model, f"{base}_grid"), radius=radii.get(f"{base}_link", 0.006), inset=insets.get(f"{base}_link", 0.001)))
        self._f6 = np.zeros(6); self._own_prefix = f"{side}_hand_"
        for fg in self.fingers:
            fg["ray"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_{fg['name']}_touch_ray")

    # ---- physical quantities -------------------------------------------------------------------------------------
    def physical(self) -> list[dict]:
        """Per finger: normal_N, tangential_N, direction_deg (0 = toward the tip, clockwise seen from outside, None if
        no tangential force), proximity_m (distance from the pad surface to the nearest object, None if > 1 cm or none),
        contacts (count), force_site (3-vector of the total contact force on the pad in the site frame), touch_sensor_N
        (the model's <touch> sensor, a cross-check of normal_N)."""
        m, d = self.m, self.d; out = []
        for fg in self.fingers:
            R = d.site_xmat[fg["site"]].reshape(3, 3)                       # site axes in world (columns)
            fn_sum = 0.0; F_world = np.zeros(3); n = 0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if c.geom1 != fg["geom"] and c.geom2 != fg["geom"]: continue
                if c.efc_address < 0: continue
                mujoco.mj_contactForce(m, d, ci, self._f6)
                fr = c.frame.reshape(3, 3)                                    # rows: normal (geom1 -> geom2), tangent 1, tangent 2
                f_on_geom2 = self._f6[0] * fr[0] + self._f6[1] * fr[1] + self._f6[2] * fr[2]
                f_on_pad = f_on_geom2 if c.geom2 == fg["geom"] else -f_on_geom2
                fn_sum += float(self._f6[0]); F_world += f_on_pad; n += 1
            F_site = R.T @ F_world                                            # x toward the tip, y, z into the finger
            Ft = float(np.hypot(F_site[0], F_site[1]))
            direction = (round(float(np.degrees(np.arctan2(F_site[1], F_site[0]))), 2) % 360.0) if Ft >= MIN_TANGENTIAL_N else None
            dist = self._proximity(fg)
            out.append(dict(finger=fg["name"], normal_N=fn_sum, tangential_N=Ft, direction_deg=direction, proximity_m=dist, contacts=n,
                            force_site=F_site.tolist(), touch_sensor_N=float(d.sensordata[fg["total"][0]])))
        return out

    def _proximity(self, fg) -> float | None:
        """Distance from the pad surface to the nearest object along the outward normal (m), None beyond 1 cm. The ray
        starts 1 mm inside the pad shell (so a touching object never swallows it) and skips the hand's own links, which
        the model's <rangefinder> sensor cannot do."""
        m, d = self.m, self.d; ray = fg["ray"]; pnt = d.site_xpos[ray].copy(); vec = d.site_xmat[ray].reshape(3, 3)[:, 2].copy()
        travelled = 0.0; geomid = np.zeros(1, dtype=np.int32)
        for _ in range(6):
            dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, fg["body"], geomid)
            if dist < 0: return None
            g = int(geomid[0]); name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if name.startswith(self._own_prefix):                             # own hand: step past it and look again
                travelled += dist + 1e-4; pnt = pnt + vec * (dist + 1e-4); continue
            total = travelled + dist - fg["inset"]
            return None if total > PROX_RANGE_M else max(0.0, total)
        return None

    def taxels(self) -> np.ndarray:
        """(5, 3, 3, 3) touch_grid taxels per finger: [finger, channel (z=into the finger, x, y), row (+y), col (+x)] in N."""
        return np.stack([self.d.sensordata[a:a + n].reshape(3, 3, 3) for a, n in (fg["grid"] for fg in self.fingers)])

    # ---- SDK format ----------------------------------------------------------------------------------------------
    @staticmethod
    def to_counts(force_N: float) -> int:
        return int(min(round(max(force_N, 0.0) * COUNTS_PER_N), FORCE_MAX_N * COUNTS_PER_N))

    @staticmethod
    def proximity_counts(distance_m: float | None) -> int:
        if distance_m is None or distance_m >= PROX_RANGE_M: return 0
        return int(round(PROX_FULL_COUNTS * (1.0 - distance_m / PROX_RANGE_M)))

    def read(self) -> list[dict]:
        """TouchFingerItem per finger (thumb, index, middle, ring, pinky), the fields of bc_stark_sdk / ros2_stark_msgs."""
        items = []
        for p in self.physical():
            items.append(dict(normal_force1=self.to_counts(p["normal_N"]), normal_force2=0, normal_force3=0,
                              tangential_force1=self.to_counts(p["tangential_N"]), tangential_force2=0, tangential_force3=0,
                              tangential_direction1=DIRECTION_INVALID if p["direction_deg"] is None else int(round(p["direction_deg"])) % 360,
                              tangential_direction2=0, tangential_direction3=0,
                              self_proximity1=self.proximity_counts(p["proximity_m"]), self_proximity2=0, mutual_proximity=0, status=STATUS_NORMAL))
        return items

    def read_arrays(self) -> dict:
        """EtherCAT TxPDO 0x6010 layout: arrays per quantity, 5 entries in finger order."""
        it = self.read()
        return dict(force_normal=[i["normal_force1"] for i in it], force_tangential=[i["tangential_force1"] for i in it],
                    force_direction=[i["tangential_direction1"] for i in it], proximity=[i["self_proximity1"] for i in it], touch_status=[i["status"] for i in it])


def sensordata_layout(model: mujoco.MjModel, side: str = "right") -> dict:
    """Where the raw MuJoCo sensors of one hand live in data.sensordata: name -> (address, dim)."""
    out = {}
    for f in FINGERS:
        for suffix in ("total", "prox", "grid"):
            name = f"{side}_hand_{f}_touch_{suffix}"
            try: out[name] = _sensor(model, name)
            except KeyError: pass
    return out
