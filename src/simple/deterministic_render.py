"""SIMPLE: SIMulation-based Policy Learning and Evaluation.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import json
import os
import random
import time
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from typing import Any

import numpy as np

from simple.determinism import enabled

SETTINGS = {
    "/rtx/rendermode": "PathTracing",
    "/rtx/pathtracing/spp": 16,
    "/rtx/pathtracing/totalSpp": 16,
    "/rtx/pathtracing/clampSpp": 16,
    "/rtx/pathtracing/optixDenoiser/enabled": False,
    "/rtx/pathtracing/adaptiveSampling/enabled": False,
    "/rtx/pathtracing/fireflyFilter/enabled": False,
    "/rtx/pathtracing/lightcache/cached/enabled": False,
    "/rtx/pathtracing/cached/enabled": False,
    "/rtx/pathtracing/mgpu/enabled": False,
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/aa/op": 0,
    "/rtx/post/dlss/execMode": 0,
    "/rtx/post/motionblur/enabled": False,
    "/rtx/post/dof/enabled": False,
    "/rtx/post/lensFlares/enabled": False,
    "/rtx/post/chromaticAberration/enabled": False,
}


def configure() -> None:
    """Apply settings idempotently and log readback; registration is not GPU proof."""
    import carb.settings

    settings = carb.settings.get_settings()
    spp = int(os.environ.get("SIMPLE_ISAAC_SPP", "16"))
    if not 1 <= spp <= 32:
        raise ValueError("Path-tracing spp must be in [1, 32]")
    render_settings = dict(SETTINGS)
    render_settings["/rtx/pathtracing/optixDenoiser/enabled"] = enabled(
        "SIMPLE_ISAAC_OPTIX_DENOISER"
    )
    effective = {}
    for key, default in render_settings.items():
        value = (
            spp
            if key
            in (
                "/rtx/pathtracing/spp",
                "/rtx/pathtracing/totalSpp",
                "/rtx/pathtracing/clampSpp",
            )
            else default
        )
        if settings.get(key) != value:
            settings.set(key, value)
        effective[key] = settings.get(key)
        if effective[key] != value:
            raise RuntimeError(f"Render setting readback mismatch: {key}")
    print(
        "PSI_PATHTRACING_SETTINGS " + json.dumps(effective, sort_keys=True), flush=True
    )


def warmup(env: Any, passes: int) -> None:
    """Wait for stage loading, then render through synchronized poses at fixed physics."""
    import mujoco
    import omni.kit.app
    import omni.timeline
    from omni.isaac.core.utils.stage import is_stage_loading

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    deadline = time.monotonic() + 180.0
    stable = 0
    while stable < 3:
        if time.monotonic() > deadline:
            raise TimeoutError("Isaac stage remained busy during path-tracing warmup")
        omni.kit.app.get_app().update()
        stable = 0 if is_stage_loading() else stable + 1
    model, data = env.mujoco.mjModel, env.mujoco.mjData
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    for _ in range(passes):
        env.isaac.step(env.mujoco)
    after = np.empty_like(state)
    mujoco.mj_getState(model, data, after, mujoco.mjtState.mjSTATE_INTEGRATION)
    if not np.array_equal(state, after):
        raise RuntimeError("Render warmup changed MuJoCo integration state")


def warmed_reset(reset: Callable) -> Callable:
    """Discard first reset without consuming the real episode's global RNG stream."""

    @wraps(reset)
    def wrapped(
        self: Any, *, seed: int | None = None, options: dict | None = None
    ) -> tuple:
        if self.isaac is None or not enabled("SIMPLE_ISAAC_DETERMINISTIC"):
            return reset(self, seed=seed, options=options)
        if not getattr(self, "_psi_pathtracing_bootstrapped", False):
            import torch

            py_state, np_state = random.getstate(), np.random.get_state()
            torch_state = torch.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            )
            try:
                reset(self, seed=seed, options=deepcopy(options))
                warmup(self, 60)
                self._render_frame()
                self._render_frame()
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)
            self._psi_pathtracing_bootstrapped = True
        reset(self, seed=seed, options=deepcopy(options))
        warmup(self, 8)
        observation, info = self._get_obs(), self._get_info()
        heads = {k: v for k, v in observation.items() if "head_stereo" in k}
        if not heads:
            raise RuntimeError("No head-camera images after path-tracing reset")
        for name, value in heads.items():
            image = np.asarray(value)
            if image.size == 0 or float(image.std()) < 1 or float(image.mean()) < 1:
                raise RuntimeError(f"Blank path-tracing reset camera: {name}")
        return observation, info

    return wrapped
