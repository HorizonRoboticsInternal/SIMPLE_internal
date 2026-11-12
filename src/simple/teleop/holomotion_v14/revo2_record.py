"""Recording extras for the G1 with BrainCo Revo 2 hands in the HoloMotion v1.4 sim teleop (2026-10-26).

With `--robot revo2 --record` the LeRobot dataset keeps the Dex3 recordings' columns, with the hand slots of
observation.state / action widened from 7 to 11 joints, and every frame gains the Revo 2 joint and motor columns and the
fingertip tactile pads (simple.sensors.revo2_touch), all sampled on the pre-step state like observation.state.

observation.state (51): the robot's joint order = 29 body joints (legs, waist, left arm, right arm), left hand 11, right hand 11.
action (51): the decoupled-WBC layout of the Dex3 datasets = legs + waist + left arm [0:22], left hand [22:33], right arm
[33:40], right hand [40:51]; hand values in MJCF order (thumb metacarpal, thumb proximal, thumb distal, index proximal,
index distal, middle, ring, pinky proximal/distal).

Extra columns, per side s in left, right (thumb, index, middle, ring, pinky for the pads):
  states.revo2.{s}.joint_position   11 float32 rad       states.revo2.{s}.motor_position  6 float32 rad, the 6 motors
  action.revo2.{s}.cmd_position     11 float32 rad       action.revo2.{s}.motor_target    6 float32 rad
  action.revo2.{s}.grip_mode         1 int64: 0 open, 1 closed (the power grasp), 2 close2 (the second closing pose: the pinch
                                     until 2026-11-11, then the thumb-only close); which button gives which is in run_meta revo2_grip
  tactile.{s}.force_normal           5 int32 (0.01 N)    tactile.{s}.force_normal_N       5 float32 N
  tactile.{s}.force_tangential       5 int32 (0.01 N)    tactile.{s}.force_tangential_N   5 float32 N
  tactile.{s}.force_direction        5 int32 deg, 65535 = none     tactile.{s}.proximity_m  5 float32 m, -1 = nothing within 1 cm
  tactile.{s}.proximity              5 int64 counts      tactile.{s}.status               5 int32
  tactile.{s}.contacts               5 int32 contacts on the pad   tactile.{s}.taxels_N   135 float32 (finger, channel z/x/y, row, col)
The hands' joint_velocity / joint_effort come with the real-format columns (states.revo2.{s}.*, realformat.py).
The 6 motors are the proximal joints: thumb rotation (metacarpal), thumb bend, index, middle, ring, pinky; the distal joints follow them.
"""
from __future__ import annotations
import numpy as np
import mujoco
from simple.robots.g1_sonic_revo2 import BODY_JOINTS, CLOSE2_Q, CLOSED_Q, FINGERS, LEFT_REVO2_JOINTS, RIGHT_REVO2_JOINTS
from simple.sensors.revo2_touch import FINGERS as PADS, Revo2Touch

HAND_W = 11
ACTIVE = [0, 1, 3, 5, 7, 9]
MOTORS = ["thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"]
SIDES = ("left", "right")
TACTILE = [("force_normal", "int32", 5), ("force_tangential", "int32", 5), ("force_direction", "int32", 5), ("proximity", "int64", 5), ("status", "int32", 5),
           ("force_normal_N", "float32", 5), ("force_tangential_N", "float32", 5), ("proximity_m", "float32", 5), ("contacts", "int32", 5), ("taxels_N", "float32", 135)]


def state_names() -> list[str]:
    return list(BODY_JOINTS) + list(LEFT_REVO2_JOINTS) + list(RIGHT_REVO2_JOINTS)


def action_names() -> list[str]:
    return list(BODY_JOINTS[:22]) + list(LEFT_REVO2_JOINTS) + list(BODY_JOINTS[22:]) + list(RIGHT_REVO2_JOINTS)


def features(tactile: bool = True) -> dict:
    """The extra columns; tactile=False (--lean-record / --no-tactile) leaves the tactile.* pads out."""
    f = {}
    for s in SIDES:
        jn = [f"{s}_hand_{x}_joint" for x in FINGERS]
        f[f"states.revo2.{s}.joint_position"] = {"dtype": "float32", "shape": (HAND_W,), "names": jn}
        f[f"states.revo2.{s}.motor_position"] = {"dtype": "float32", "shape": (6,), "names": list(MOTORS)}
        f[f"action.revo2.{s}.cmd_position"] = {"dtype": "float32", "shape": (HAND_W,), "names": jn}
        f[f"action.revo2.{s}.motor_target"] = {"dtype": "float32", "shape": (6,), "names": list(MOTORS)}
        f[f"action.revo2.{s}.grip_mode"] = {"dtype": "int64", "shape": (1,), "names": ["0_open_1_closed_2_pinch"]}
        for k, dt, n in (TACTILE if tactile else []):
            f[f"tactile.{s}.{k}"] = {"dtype": dt, "shape": (n,), "names": list(PADS) if n == 5 else None}
    return f


def tactile_features() -> dict:
    return {k: v for k, v in features(tactile=True).items() if k.startswith("tactile.")}


def patch_schema(feats: dict, modality: dict) -> None:
    """Widen the Dex3 schema's state/action to the Revo 2 robot and fix the modality slices."""
    sn, an = state_names(), action_names()
    feats["observation.state"] = {**feats["observation.state"], "shape": (len(sn),), "names": sn}
    feats["action"] = {**feats["action"], "shape": (len(an),), "names": an}
    st, ac = modality.get("state", {}), modality.get("action", {})
    for key, (a, b) in [("left_arm", (15, 22)), ("right_arm", (22, 29)), ("left_hand", (29, 40)), ("right_hand", (40, 51))]:
        if key in st: st[key].update(start=a, end=b)
    for key, (a, b) in [("left_arm", (15, 22)), ("left_hand", (22, 33)), ("right_arm", (33, 40)), ("right_hand", (40, 51))]:
        if key in ac: ac[key].update(start=a, end=b)


class Revo2Recorder:
    """Per-frame Revo 2 columns; call sample(action) right before env.step, on the state observation.state was taken from."""

    def __init__(self, sonic_env, tactile: bool = True) -> None:
        self.env = sonic_env; self._model = None; self.tactile = bool(tactile)

    def _bind(self, m, d):
        self.touch = {s: Revo2Touch(m, d, s) for s in SIDES} if self.tactile else {}
        self.qa = {"left": np.array([m.joint(j).qposadr[0] for j in LEFT_REVO2_JOINTS]), "right": np.array([m.joint(j).qposadr[0] for j in RIGHT_REVO2_JOINTS])}
        self._model = m

    @staticmethod
    def _params(action) -> dict:
        return getattr(action, "parameters", action)

    def action_vector(self, action) -> np.ndarray:
        P = self._params(action)
        tq = np.asarray(P["target_q"], dtype=np.float64)
        lh = np.asarray(P["left_hand_q"], dtype=np.float64)[:HAND_W]; rh = np.asarray(P["right_hand_q"], dtype=np.float64)[:HAND_W]
        return np.concatenate([tq[:22], lh, tq[22:29], rh])

    def sample(self, action) -> dict:
        m, d = self.env.mujoco.mjModel, self.env.mujoco.mjData
        if self._model is not m:
            self._bind(m, d)
        P = self._params(action); f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        cols = {"_action": self.action_vector(action)}
        for s in SIDES:
            q = d.qpos[self.qa[s]]; cmd = np.asarray(P[f"{s}_hand_q"], dtype=np.float64)[:HAND_W]
            cols[f"states.revo2.{s}.joint_position"] = f32(q); cols[f"states.revo2.{s}.motor_position"] = f32(q[ACTIVE])
            cols[f"action.revo2.{s}.cmd_position"] = f32(cmd); cols[f"action.revo2.{s}.motor_target"] = f32(cmd[ACTIVE])
            mode = 1 if np.allclose(cmd, CLOSED_Q[:HAND_W], atol=1e-3) else 2 if np.allclose(cmd, CLOSE2_Q[:HAND_W], atol=1e-3) else 0
            cols[f"action.revo2.{s}.grip_mode"] = np.array([mode], dtype=np.int64)
            if self.tactile:
                cols.update(self._tactile(s))
        return cols

    def _tactile(self, s: str) -> dict:
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        tp = self.touch[s]; ph = tp.physical(); arrays = tp.read_arrays(); cols = {}
        cols[f"tactile.{s}.force_normal"] = np.asarray(arrays["force_normal"], dtype=np.int32)
        cols[f"tactile.{s}.force_tangential"] = np.asarray(arrays["force_tangential"], dtype=np.int32)
        cols[f"tactile.{s}.force_direction"] = np.asarray(arrays["force_direction"], dtype=np.int32)
        cols[f"tactile.{s}.proximity"] = np.asarray(arrays["proximity"], dtype=np.int64)
        cols[f"tactile.{s}.status"] = np.asarray(arrays["touch_status"], dtype=np.int32)
        cols[f"tactile.{s}.force_normal_N"] = f32([p["normal_N"] for p in ph]); cols[f"tactile.{s}.force_tangential_N"] = f32([p["tangential_N"] for p in ph])
        cols[f"tactile.{s}.proximity_m"] = f32([-1.0 if p["proximity_m"] is None else p["proximity_m"] for p in ph])
        cols[f"tactile.{s}.contacts"] = np.asarray([p["contacts"] for p in ph], dtype=np.int32)
        cols[f"tactile.{s}.taxels_N"] = f32(tp.taxels().reshape(-1))
        return cols

    def tactile_sample(self) -> dict:
        """The tactile.* columns alone, on the current state (the replay's --fill of a --lean-record dataset)."""
        m, d = self.env.mujoco.mjModel, self.env.mujoco.mjData
        if self._model is not m or not self.touch:
            self.tactile = True; self._bind(m, d)
        cols = {}
        for s in SIDES:
            cols.update(self._tactile(s))
        return cols
