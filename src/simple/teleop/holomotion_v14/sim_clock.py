"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

A controller clock for the HoloMotion v1.4 stack in evaluation: the time the policy node and the agent read advances
by exactly one control period per sim step, whatever the wall clock does.

The vendored node reads ``time.time()`` for the reference's data age (frozen above 0.6 s), the controller-input timeout
and its receive bookkeeping, and the agent reads ``time.monotonic()`` for its main-state timers. In teleop both run at
wall-clock pace. In an evaluation a VLA query stalls the loop for a variable time, so on the wall clock the controller
would see a different history on every run. ``ControllerClock.install()`` swaps the ``time`` module those modules
imported for a proxy on this clock (as ``simple.simulation_clock.install_dependency_clocks`` does for the
decoupled-WBC policy modules); ``perf_counter`` stays real (timing diagnostics only).
"""

from __future__ import annotations

import sys
import time as _time
from typing import Any

# every module of the stack that does `import time`
MODULE_PREFIXES = ("humanoid_policy.", "holomotion_policy_core.", "holomotion_peripherals_ros2.",
                   "simple.agents.holomotion_v14_agent", "simple.agents.holomotion_v14_vla_agent",
                   "simple.teleop.holomotion_v14.sim_node")


class _TimeProxy:
    def __init__(self, clock: "ControllerClock") -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.now

    def monotonic(self) -> float:
        return self._clock.now

    def time_ns(self) -> int:
        return int(round(self._clock.now * 1e9))

    def monotonic_ns(self) -> int:
        return self.time_ns()

    def __getattr__(self, name: str) -> Any:        # perf_counter, sleep, strftime, ...: the real ones
        return getattr(_time, name)


class ControllerClock:
    """Starts at ``t0`` seconds (an epoch-like value, so stamps look like wall time) and moves only by ``advance``."""

    def __init__(self, t0: float = 1.0e9) -> None:
        self.now = float(t0)
        self._proxy = _TimeProxy(self)

    def advance(self, dt: float) -> None:
        self.now += float(dt)

    def install(self) -> list[str]:
        """Point every loaded stack module's ``time`` at this clock; returns the modules changed. Call after the
        policy node is built (its modules are imported then)."""
        changed = []
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith(MODULE_PREFIXES):
                continue
            if getattr(module, "time", None) is _time:
                module.time = self._proxy
                changed.append(name)
        return changed
