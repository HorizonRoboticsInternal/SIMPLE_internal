"""SIMPLE: SIMulation-based Policy Learning and Evaluation.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import random

import numpy as np

RUNTIME_VERSION = 1


def enabled(name: str = "SIMPLE_DETERMINISTIC_RUNTIME") -> bool:
    """Read an explicit runtime switch; invalid values fail rather than disable it."""
    value = os.environ.get(name, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0/1 or false/true")
    return value in {"1", "true"}


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Derive an episode stream independent of previous rollout lengths."""
    if (
        not isinstance(base_seed, int) or isinstance(base_seed, bool)
    ) or not 0 <= base_seed < 2**32:
        raise ValueError("Base seed must be an integer in [0, 2**32)")
    if (
        not isinstance(episode_index, int) or isinstance(episode_index, bool)
    ) or episode_index < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    return int(np.random.SeedSequence([base_seed, episode_index]).generate_state(1)[0])


def seed_episode(base_seed: int, episode_index: int) -> int:
    """Seed simulator Python, NumPy and Torch generators for one episode."""
    import torch

    seed = episode_seed(base_seed, episode_index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
