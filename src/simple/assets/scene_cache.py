# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Keep generated scene geometry separate from shared, read-only assets."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def scene_cache_dir(scene: str, layout: dict) -> Path:
    parameters = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(scene.upper() + "_")
    }
    digest = hashlib.sha256(
        json.dumps(
            {"layout": layout, "parameters": parameters}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    root = Path(
        os.environ.get(
            "SIMPLE_SCENE_CACHE_DIR",
            os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir()),
        )
    )
    path = root / "simple-scene-meshes" / str(os.getpid()) / scene / digest
    if path.resolve().is_relative_to("/horizon-bucket"):
        raise ValueError("Generated scene meshes require a local cache directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def fixed_asset(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed scene asset: {path}")
    return path
