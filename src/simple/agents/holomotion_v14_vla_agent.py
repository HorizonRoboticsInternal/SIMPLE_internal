"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

HoloMotion v1.4 (model_22000 by default) driven by a VLA instead of the PICO operator.

The teleop agent (holomotion_v14_agent.HoloMotionV14Agent) gets two inputs from the headset side: the body reference
(latest_obs frames, 65 numbers: dof_pos[29] dof_vel[29] root_pos[3] root_rot_wxyz[4], joints in the robot's order) and
the controller sample (buttons, and the grips that close the Dex3 hands). Here the VLA supplies both:

    VLA reply, one row per reference frame:
        [0:65]   latest_obs -- what model_22000 tracks (required)
        [65:67]  left, right grip in [0, 1] (the v1.4 grip gripper: > 0.5 closes the hand)        (D = 67)
     or [65:79]  left hand q[7], right hand q[7] in SIMPLE's MJCF order (as ActionCmd)               (D = 79)
        (D = 65: the hands stay open)

Each control step (50 Hz) feeds reference_hz / 50 frames (default one) to the policy node, right before the node's
step, exactly where the teleop feeds the frames the publisher sent; a new chunk is queried from the VLA when the
current one runs out. An episode goes: quick start (standing, walking policy on, as in teleop) -> walking mode until
the VLA has returned valid frames: they stream in while the robot stands, and once the node has n_fut_frames + 1 of
them (11) the motion-tracking button is pressed, as the operator's Y -> motion tracking on the VLA's frames until the
episode ends. Until then the walking policy stands the robot in place, arms and wrists as it moves them (as in teleop
before the operator's Y).

A reply is valid when it parses, has shape (T, 65 | 67 | 79) with T >= 1, is finite, and its root quaternions have unit
norm (+-10 %). An invalid reply or a failed query feeds nothing: before motion tracking the robot keeps walking (standing)
and the next step asks again; during it the node treats the missing frames as it treats a stalled stream on the robot
(it freezes the reference once it is older than max_data_age). No valid frames within engage_timeout_s -> the episode
fails with that reason.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Callable

import time as _wall

import numpy as np

from simple.core.action import ActionCmd
from simple.robots.g1_sonic import G1Sonic

from .holomotion_v14_agent import NEUTRAL_PICO_SAMPLE, HoloMotionV14Agent

LATEST_OBS_DIM = 65


def _invalid_reason(pred: np.ndarray) -> str | None:
    """Why a VLA reply cannot be fed to the controller, or None if it can."""
    if pred.ndim != 2 or pred.shape[1] not in (LATEST_OBS_DIM, 67, 79) or len(pred) == 0:
        return f"shape {pred.shape}, expected (T, 65 | 67 | 79)"
    if not np.all(np.isfinite(pred)):
        return "non-finite values"
    qn = np.linalg.norm(pred[:, 61:65], axis=1)
    if np.any(np.abs(qn - 1.0) > 0.1):
        return f"root quaternion norm {float(qn.min()):.3f}..{float(qn.max()):.3f}"
    return None


class _VlaFeed:
    """Stands in for wire.ReferenceSubscriber: the frames queued for this step and the synthetic controller sample."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.last_pico: dict | None = None
        self.n_obs = 0

    def poll(self) -> list[dict]:
        out, self.frames = self.frames, []
        self.n_obs += len(out)
        return out

    def pico_age(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


class HoloMotionV14VlaAgent(HoloMotionV14Agent):
    def __init__(
        self,
        robot: G1Sonic,
        *,
        client,
        image_fn: Callable[[], np.ndarray],
        clock,
        reference_hz: float = 50.0,
        settle_s: float = 0.0,
        engage_timeout_s: float = 10.0,
        image_key: str = "observation.images.ego_view",
        dataset_name: str = "simple",
    ) -> None:
        super().__init__(robot, reference_uri="tcp://127.0.0.1:1", enable_pico_stream=False, button_map="sim", quiet=True)
        self.sub.close()                                      # no publisher: the VLA feeds the node
        self.sub = _VlaFeed()
        self.client, self.image_fn, self.clock = client, image_fn, clock
        self.reference_hz, self.settle_s, self.engage_timeout_s = float(reference_hz), float(settle_s), float(engage_timeout_s)
        self.image_key, self.dataset_name = image_key, dataset_name
        self._session_idx = 0
        self.reset_episode(None)

    # ------------------------------------------------------------------ episode
    def reset_episode(self, hold_real: np.ndarray | None, episode_index: int = -1) -> None:
        """A fresh controller for a new episode; the robot was just placed standing (hold_real: its body pose in the
        robot's joint order)."""
        self.reset_policy()
        self.quick_start(hold_real=hold_real, auto_motion=False)
        self.phase = "stand"
        self._phase_steps = 0
        self._chunk: deque = deque()
        self._acc = 0.0
        self._frame_index = 0
        self._grips = (0.0, 0.0)
        self._hands: tuple[np.ndarray, np.ndarray] | None = None
        self._session_idx += 1
        self._session_id = f"holomotion-v14-{os.getpid()}-{self._session_idx}"
        self._reset_history = True
        self._episode_index = int(episode_index)
        self.steps = 0
        self.timeline: list[tuple[int, str, str]] = []                  # (step, main state, policy) at every change
        self.engaged_step: int | None = None
        self.queries: list[dict] = []
        self.invalid: list[dict] = []                                    # rejected replies / failed queries
        self.first_valid_step: int | None = None
        self._obs = None
        self._proprio = None
        self._instruction = None

    # ------------------------------------------------------------------ VLA
    def _query(self) -> bool:
        n = self.node
        ref = n._vr_reference
        latest = (np.zeros(LATEST_OBS_DIM, np.float32) if ref is None or ref.latest_obs is None
                  else np.asarray(ref.latest_obs, np.float32).reshape(-1)[:LATEST_OBS_DIM].copy())
        p = self._proprio
        state = {
            "observation.state": np.asarray(self._obs["joint_qpos"], dtype=np.float64),
            "observation.base_pose": np.asarray(p["floating_base_pose"], dtype=np.float64),
            "observation.base_vel": np.asarray(p["floating_base_vel"], dtype=np.float64),
            "teleop.latest_obs": latest,
            "policy.mode": np.array([1 if n.current_policy_mode == "motion" and n.policy_enabled else 0], np.int64),
        }
        history = {"session_id": self._session_id, "episode_index": self._episode_index, "step_index": int(self.steps),
                   "frame_index": int(self._frame_index)}
        if self._reset_history:
            history["reset"] = True
            self._reset_history = False
        t0 = _wall.perf_counter()
        try:
            pred, *_ = self.client.query_action({self.image_key: self.image_fn()}, self._instruction, state, {},
                                                history=history, dataset=self.dataset_name)
            pred = np.asarray(pred, dtype=np.float64)
            if pred.ndim == 1:
                pred = pred[None]
            why = _invalid_reason(pred)
        except Exception as e:  # noqa: BLE001  (server down, HTTP error, unparsable reply: not valid input)
            pred, why = None, f"{type(e).__name__}: {str(e)[:120]}"
        dt = _wall.perf_counter() - t0
        if why:
            self.invalid.append(dict(step=int(self.steps), reason=why))
            if len(self.invalid) <= 3 or len(self.invalid) % 50 == 0:
                print(f"[HoloMotion v1.4 eval] step {self.steps}: VLA reply not used ({why}); "
                      f"{'still walking' if self.phase != 'motion' else 'the reference holds'}")
            return False
        self._chunk.extend(pred)
        if self.first_valid_step is None:
            self.first_valid_step = int(self.steps)
        self.queries.append(dict(step=int(self.steps), frames=int(len(pred)), dim=int(pred.shape[1]), seconds=round(dt, 4)))
        return True

    def _next_frame(self) -> bool:
        if not self._chunk and not self._query():
            return False
        row = self._chunk.popleft()
        self.sub.frames.append({"latest_obs": row[:LATEST_OBS_DIM].astype(np.float32),
                                "frame_index": np.array([self._frame_index]),
                                "timestamp_ns": np.array([int(round(self.clock.now * 1e9))])})
        self._frame_index += 1
        if len(row) == 67:
            self._grips = (float(row[65]), float(row[66]))
        elif len(row) == 79:
            self._hands = (row[65:72].astype(np.float32), row[72:79].astype(np.float32))
        return True

    # ------------------------------------------------------------------ step
    def get_action(self, observation, instruction=None, **kwargs) -> ActionCmd:
        from simple.teleop.holomotion_v14.wire import BUTTONS
        n = self.node
        self._obs, self._proprio = observation, kwargs["privileged_info"]["proprio"]
        self._instruction = instruction
        self._phase_steps += 1
        press = None
        if self.phase == "stand" and self._quick is None and self.main_state == "POLICY" and n.policy_enabled:
            self.phase, self._phase_steps = "settle", 0                     # walking policy on: the robot stands
        if self.phase == "settle" and self._phase_steps * self._control_dt >= self.settle_s:
            self.phase, self._phase_steps = "prefill", 0
        if self.phase in ("prefill", "switch", "motion"):
            self._acc += self.reference_hz * self._control_dt
            k = int(self._acc + 1e-9)
            self._acc -= k
            for _ in range(k):
                if not self._next_frame():
                    break                                              # nothing valid this step: ask again next step
        if self.phase == "prefill" and n._vr_reference is not None and n._is_vr_ready_for_motion():
            self.phase, self._phase_steps = "switch", 0
        if self.phase == "switch":
            if n.current_policy_mode == "motion" and n.policy_enabled:
                self.phase, self.engaged_step = "motion", int(self.steps)
                print(f"[HoloMotion v1.4 eval] motion tracking on the VLA's frames (step {self.steps})")
            else:
                press = self.motion_button if self._phase_steps % 4 else None   # a fresh edge every 4 ticks
        if self.phase in ("prefill", "switch") and self._phase_steps * self._control_dt > self.engage_timeout_s:
            why = (f"no valid VLA frames in {self.engage_timeout_s:g} s ({len(self.invalid)} replies not used; last: "
                   f"{self.invalid[-1]['reason'] if self.invalid else '-'})" if self.first_valid_step is None else
                   f"motion tracking did not engage within {self.engage_timeout_s:g} s of the first valid frames")
            raise RuntimeError(why)

        # the controller sample the node and the quick start read: nothing pressed but the switch, the VLA's grips
        s = dict(NEUTRAL_PICO_SAMPLE, left_grip=self._grips[0], right_grip=self._grips[1],
                 l_active=1, r_active=1, timestamp_ns=int(round(self.clock.now * 1e9)))
        if press:
            s[press] = 1
            s["button_bits"] = int(s["button_bits"]) | BUTTONS[press]
        self.sub.last_pico = s
        action = super().get_action(observation, instruction, **kwargs)
        if self._hands is not None:                                        # D = 79: hand joints straight from the VLA
            action = ActionCmd(action.type, **{**action.parameters, "left_hand_q": self._hands[0].copy(),
                                               "right_hand_q": self._hands[1].copy()})
        st = (self.main_state, n.current_policy_mode if n.policy_enabled else "off")
        if not self.timeline or self.timeline[-1][1:] != st:
            self.timeline.append((int(self.steps), *st))
        self.steps += 1
        return action

    def summary(self) -> dict[str, Any]:
        q = [x["seconds"] for x in self.queries]
        return dict(phase=self.phase, engaged_step=self.engaged_step, frames_fed=int(self._frame_index),
                    queries=len(self.queries), query_seconds_mean=round(float(np.mean(q)), 4) if q else None,
                    query_seconds_max=round(float(np.max(q)), 4) if q else None,
                    reply_dim=self.queries[0]["dim"] if self.queries else None,
                    first_valid_step=self.first_valid_step, replies_not_used=len(self.invalid),
                    not_used_reasons=sorted({x["reason"].split(":")[0] for x in self.invalid}),
                    controller=[f"step {t}: {m} / {p}" for t, m, p in self.timeline])
