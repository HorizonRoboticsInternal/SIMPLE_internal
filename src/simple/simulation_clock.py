"""SIMPLE: SIMulation-based Policy Learning and Evaluation.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import math
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

from simple.determinism import enabled

_CURRENT = ContextVar("psi_controller_simulation_time", default=None)
_REAL_MONOTONIC = time.monotonic
_REAL_TIME = time.time
_REAL_PERF = time.perf_counter


def monotonic() -> float:
    value = _CURRENT.get()
    return _REAL_MONOTONIC() if value is None else value


def wall_time() -> float:
    value = _CURRENT.get()
    return _REAL_TIME() if value is None else value


def perf_counter() -> float:
    value = _CURRENT.get()
    return _REAL_PERF() if value is None else value


class _ScopedTime:
    monotonic = staticmethod(monotonic)
    time = staticmethod(wall_time)
    perf_counter = staticmethod(perf_counter)

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


_PROXY = _ScopedTime()


def install_dependency_clocks() -> list[str]:
    """Redirect only imported clocks in loaded decoupled-WBC policy modules."""
    changed = []
    aliases = [
        (_REAL_MONOTONIC, monotonic),
        (_REAL_TIME, wall_time),
        (_REAL_PERF, perf_counter),
    ]
    for name, module in list(sys.modules.items()):
        if not name.startswith("decoupled_wbc.control.policy.") or module is None:
            continue
        for key, value in list(vars(module).items()):
            replacement = (
                _PROXY
                if value is time
                else next((new for old, new in aliases if value is old), None)
            )
            if replacement is not None:
                setattr(module, key, replacement)
                changed.append(name + "." + key)
    return changed


@contextmanager
def simulation_time(value: float) -> Iterator[None]:
    """Supply one clock to interpolation and timeout bookkeeping in this call."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Simulation time must be finite and nonnegative")
    install_dependency_clocks()
    token = _CURRENT.set(value)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def controller_time(controller: Any) -> float:
    """Use actual MuJoCo time, including stabilization, without a rollout reset."""
    if not enabled():
        return _REAL_MONOTONIC()
    robot = getattr(controller, "robot", None)
    data = getattr(robot, "mjData", None)
    return float(data.time) if data is not None else 0.0


def simulation_timed(method: Callable) -> Callable:
    """Apply only to the job-local evaluation classes patched to import this."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not enabled():
            return method(self, *args, **kwargs)
        robot = getattr(self, "robot", None)
        if method.__name__ == "__init__" and robot is None:
            robot = kwargs.get("robot", args[0] if args else None)
        data = getattr(robot, "mjData", None)
        with simulation_time(float(data.time) if data is not None else 0.0):
            result = method(self, *args, **kwargs)
            install_dependency_clocks()  # Cover dependencies loaded by construction.
            return result

    return wrapped
