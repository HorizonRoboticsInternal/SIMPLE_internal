"""Shared Isaac SimulationApp creation helpers."""

from __future__ import annotations
import os
import sys
from pathlib import Path

from simple.utils import env_flag

HYDRA_WAIT_IDLE = "/app/hydraEngine/waitIdle"
HYDRA_RENDER_COMPLETE = "/app/updateOrder/checkForHydraRenderComplete"
THROTTLING_ENABLE_ASYNC = "/exts/isaacsim.core.throttling/enable_async"


def _compact_dict(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def create_simulation_app(
    simulation_app_cls,
    *,
    headless: bool,
    renderer: str = "RayTracedLighting",
    width: int | None = None,
    height: int | None = None,
    anti_aliasing: int | None = None,
    hide_ui: bool | None = None,
    multi_gpu: bool = False,
):
    experience = os.getenv("SIMPLE_ISAAC_EXPERIENCE", "").strip()
    zero_delay = env_flag("SIMPLE_ISAAC_ZERO_DELAY", default=True)
    disable_throttling_async = env_flag("SIMPLE_ISAAC_DISABLE_THROTTLING_ASYNC", default=True)

    settings: list[tuple[str, object, object]] = []
    if zero_delay:
        settings.append((HYDRA_WAIT_IDLE, 1, True))
        settings.append((HYDRA_RENDER_COMPLETE, 1000, 1000))
    if disable_throttling_async:
        settings.append((THROTTLING_ENABLE_ASYNC, "false", False))
    extra_args = [f"--{key}={arg_value}" for key, arg_value, _ in settings]

    sim_cfg = _compact_dict({
        "headless": headless,
        "renderer": renderer,
        "multi_gpu": multi_gpu,
        "anti_aliasing": anti_aliasing,
        "hide_ui": hide_ui,
        "width": width,
        "height": height,
        "experience": experience,
    })
    if extra_args:
        sim_cfg["extra_args"] = extra_args

    portable_root = os.getenv("SIMPLE_ISAAC_PORTABLE_ROOT", "").strip()
    original_argv: list[str] | None = None
    has_portable_root = any(
        arg == "--portable-root" or arg.startswith("--portable-root=")
        for arg in sys.argv
    )
    if portable_root and not has_portable_root:
        Path(portable_root).mkdir(parents=True, exist_ok=True)
        original_argv = sys.argv.copy()
        sys.argv.extend(["--portable-root", portable_root])

    try:
        app = simulation_app_cls(sim_cfg)
    finally:
        if original_argv is not None:
            sys.argv[:] = original_argv

    for key, _, runtime_value in settings:
        app.set_setting(key, runtime_value)

    return app
