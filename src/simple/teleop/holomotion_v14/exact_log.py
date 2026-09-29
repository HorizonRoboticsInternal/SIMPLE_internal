"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Bit-exact episode log for sim teleop: everything MuJoCo integrates from, so an episode can be replayed to the last bit.

MuJoCo's documented contract: ``mj_step`` is a deterministic function of the model and the *integration state*
(``mjSTATE_INTEGRATION``: time, qpos, qvel, act, warm start, and the user inputs ctrl, qfrc_applied, xfrc_applied,
eq_active, mocap, userdata). So the log keeps

    state0      the full integration state right before the episode's first env.step
    ctrl        ctrl before every physics call (env.step makes 4 calls at 50 Hz, each mj_step at 200 Hz)
    rest        the other user inputs (applied forces: the object pseudo-gravity, the safety band; mocap, ...),
                stored only when they change
    nstep       mj_step count of each call
    physics     qpos/qvel/act after every env.step (mjSTATE_PHYSICS), the reference a replay is compared against
    action      the ActionCmd fields of every frame (target_q, kp, kd, hand targets, band) plus the band state before
                it, for replaying through env.step (the PD torques recomputed) instead of the raw ctrl
    model hash  sha256 of the compiled model (mj_saveModel bytes); a replay first checks it rebuilt the same model

Replay (simple.cli.replay_holomotion_v14): rebuild the scene from the saved setup, check the model hash, mj_setState
state0, then set ctrl/rest and mj_step for every call and compare physics bit for bit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

S = mujoco.mjtState
INTEGRATION = int(S.mjSTATE_INTEGRATION)
PHYSICS = int(S.mjSTATE_PHYSICS)
REST = int(S.mjSTATE_USER) & ~int(S.mjSTATE_CTRL)        # qfrc/xfrc_applied, eq_active, mocap, userdata

LOG_VERSION = 1


def get_state(m, d, spec: int) -> np.ndarray:
    out = np.empty(mujoco.mj_stateSize(m, spec), dtype=np.float64)
    mujoco.mj_getState(m, d, out, spec)
    return out


def set_state(m, d, state: np.ndarray, spec: int) -> None:
    mujoco.mj_setState(m, d, np.ascontiguousarray(state, dtype=np.float64), spec)


def model_bytes(m) -> bytes:
    buf = np.empty(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return buf.tobytes()


def model_sha256(m) -> str:
    return hashlib.sha256(model_bytes(m)).hexdigest()


def band_state(robot) -> np.ndarray:
    b = getattr(robot, "elastic_band", None)
    if b is None:
        return np.zeros(5)
    return np.array([float(b.length), float(bool(b.enable)), *np.asarray(b.point, dtype=np.float64)[:3]])


def restore_band(robot, v: np.ndarray) -> None:
    b = getattr(robot, "elastic_band", None)
    if b is not None:
        b.length, b.enable = float(v[0]), bool(v[1] > 0.5)
        b.point = np.asarray(v[2:5], dtype=np.float64).copy()


class ExactLog:
    """Hooks the env's MuJoCo engine and records the current episode."""

    def __init__(self, sonic_env) -> None:
        self.env = sonic_env
        self.eng = sonic_env.mujoco
        self._orig_step = self.eng.step
        self.eng.step = self._step                        # instance attribute: env.step's 4 calls land here
        self._calls: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._pre: dict[str, Any] | None = None
        self.recording = False
        self._clear()

    # ------------------------------------------------------------------ hooks
    def _step(self, *args, **kwargs):
        m, d = self.eng.mjModel, self.eng.mjData
        t0 = float(d.time)
        ctrl, rest = d.ctrl.copy(), get_state(m, d, REST)
        out = self._orig_step(*args, **kwargs)
        self._calls.append((ctrl, rest, int(round((float(d.time) - t0) / m.opt.timestep))))
        return out

    def before_step(self) -> None:
        """Call right before env.step: the state an episode starting at this step starts from."""
        m, d = self.eng.mjModel, self.eng.mjData
        self._calls = []
        self._pre = dict(state=get_state(m, d, INTEGRATION), render_step=int(self.eng.render_step),
                         band=band_state(self.env.task.robot))

    # ------------------------------------------------------------------ episode
    def _clear(self) -> None:
        self.recording = False
        self.state0 = None
        self.render_step0 = 0
        self.ctrl, self.nstep, self.frame_calls = [], [], []
        self.rest_idx, self.rest_val = [], []
        self.physics, self.time = [], []
        self.act = {k: [] for k in ("target_q", "kp", "kd", "left_hand_q", "right_hand_q", "band", "band_state")}
        self._last_rest = None
        self.model_sha = ""

    def start(self) -> None:
        """The frame being added next is the episode's first: its pre-step state is the start state."""
        assert self._pre is not None, "before_step() was not called"
        self._clear()
        self.recording = True
        self.state0 = self._pre["state"].copy()
        self.render_step0 = self._pre["render_step"]
        self.model_sha = model_sha256(self.eng.mjModel)

    def add_frame(self, action) -> None:
        if not self.recording:
            return
        m, d = self.eng.mjModel, self.eng.mjData
        n_before = len(self.ctrl)
        for ctrl, rest, n in self._calls:
            if self._last_rest is None or not np.array_equal(rest, self._last_rest):
                self.rest_idx.append(len(self.ctrl))
                self.rest_val.append(rest)
                self._last_rest = rest
            self.ctrl.append(ctrl)
            self.nstep.append(n)
        self.frame_calls.append(len(self.ctrl) - n_before)
        self.physics.append(get_state(m, d, PHYSICS))
        self.time.append(float(d.time))
        for k in ("target_q", "kp", "kd", "left_hand_q", "right_hand_q"):
            self.act[k].append(np.asarray(action[k], dtype=np.float64))
        self.act["band"].append(bool(action["apply_elastic_band"]))
        self.act["band_state"].append(self._pre["band"])

    def discard(self) -> None:
        self._clear()

    def save(self, path: Path, setup: dict[str, Any], meta: dict[str, Any], save_model: bool = True) -> Path:
        """Write episode_XXXXXX.npz (and the compiled model, .mjb) next to the dataset."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m = self.eng.mjModel
        info = dict(version=LOG_VERSION, mujoco=mujoco.__version__, model_sha256=self.model_sha, timestep=float(m.opt.timestep),
                    frames=len(self.physics), calls=len(self.ctrl), sizes=dict(nq=m.nq, nv=m.nv, nu=m.nu, na=m.na, nbody=m.nbody),
                    **meta)
        arrays = dict(
            state0=self.state0, render_step0=np.int64(self.render_step0),
            ctrl=np.asarray(self.ctrl, dtype=np.float64), nstep=np.asarray(self.nstep, dtype=np.int32),
            frame_calls=np.asarray(self.frame_calls, dtype=np.int32),
            rest_idx=np.asarray(self.rest_idx, dtype=np.int64), rest_val=np.asarray(self.rest_val, dtype=np.float64),
            physics=np.asarray(self.physics, dtype=np.float64), time=np.asarray(self.time, dtype=np.float64),
            setup_json=np.array(json.dumps(setup, sort_keys=True)), info_json=np.array(json.dumps(info, sort_keys=True)),
            **{f"action_{k}": np.asarray(v) for k, v in self.act.items()},
        )
        np.savez_compressed(path, **arrays)
        if save_model:
            mujoco.mj_saveModel(m, str(path.with_suffix(".mjb")), None)
        self._clear()
        return path


def load(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["setup"] = json.loads(str(out.pop("setup_json")))
    out["info"] = json.loads(str(out.pop("info_json")))
    return out


def replay_physics(m, d, log: dict[str, Any], on_frame=None) -> dict[str, Any]:
    """Replay the raw inputs from state0; compare every frame's physics state bit for bit."""
    set_state(m, d, log["state0"], INTEGRATION)
    mujoco.mj_forward(m, d)                                # derived quantities only; the integration state is untouched
    rest_at = {int(i): v for i, v in zip(log["rest_idx"], log["rest_val"])}
    ctrl, nstep, phys = log["ctrl"], log["nstep"], log["physics"]
    call, first_bad, max_diff = 0, None, 0.0
    for t, n_calls in enumerate(log["frame_calls"]):
        for _ in range(int(n_calls)):
            if call in rest_at:
                set_state(m, d, rest_at[call], REST)
            d.ctrl[:] = ctrl[call]
            mujoco.mj_step(m, d, nstep=int(nstep[call]))
            call += 1
        got = get_state(m, d, PHYSICS)
        if got.tobytes() != phys[t].tobytes():
            max_diff = max(max_diff, float(np.max(np.abs(got - phys[t]))))
            if first_bad is None:
                first_bad = t
        if on_frame is not None:
            on_frame(t)
    frames = len(log["frame_calls"])
    return dict(frames=frames, exact=first_bad is None, first_mismatch=first_bad, max_abs_diff=max_diff)
