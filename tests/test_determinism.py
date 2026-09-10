"""SIMPLE: SIMulation-based Policy Learning and Evaluation.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import os
import random
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simple import determinism, simulation_clock
from simple.deterministic_render import warmed_reset


class TestDeterminism(unittest.TestCase):
    def test_episode_stream_does_not_depend_on_history(self):
        determinism.seed_episode(0, 2)
        a = (random.random(), np.random.rand(), torch.rand(3))
        determinism.seed_episode(0, 1)
        torch.rand(50)
        determinism.seed_episode(0, 2)
        b = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(a[:2], b[:2])
        self.assertTrue(torch.equal(a[2], b[2]))
        with self.assertRaises(ValueError):
            determinism.episode_seed(0, -1)

    def test_clock_scope_and_default_passthrough(self):
        name = "decoupled_wbc.control.policy._test"
        dependency = ModuleType(name)
        dependency.clock = time
        dependency.now = time.monotonic
        real = time.monotonic
        owner = SimpleNamespace(
            robot=SimpleNamespace(mjData=SimpleNamespace(time=4.25))
        )

        @simulation_clock.simulation_timed
        def read_clock(self):
            return simulation_clock.controller_time(self), dependency.now()

        with (
            patch.dict(sys.modules, {name: dependency}),
            patch.dict(os.environ, {"SIMPLE_DETERMINISTIC_RUNTIME": "1"}),
        ):
            self.assertEqual(read_clock(owner), (4.25, 4.25))
            with simulation_clock.simulation_time(6):
                with simulation_clock.simulation_time(7):
                    self.assertEqual(dependency.clock.monotonic(), 7)
                self.assertEqual(dependency.now(), 6)
            self.assertGreaterEqual(dependency.now(), real() - 1)
            self.assertIs(time.monotonic, real)
        with patch.dict(os.environ, {"SIMPLE_DETERMINISTIC_RUNTIME": "0"}):
            self.assertGreaterEqual(read_clock(owner)[0], real() - 1)

    def test_discarded_reset_preserves_rng_and_options(self):
        # Unit test reset orchestration; Isaac propagation has separate GPU evidence.
        class Env:
            isaac = object()
            count = 0

            def _render_frame(self):
                return None

            def _get_obs(self):
                return {"head_stereo_left": np.arange(12).reshape(2, 2, 3)}

            def _get_info(self):
                return {}

            @warmed_reset
            def reset(self, *, seed=None, options=None):
                self.count += 1
                self.sample = (random.random(), np.random.rand(), torch.rand(2))
                options["value"] = "changed"
                return self._get_obs(), {}

        determinism.seed_episode(3, 0)
        expected = (random.random(), np.random.rand(), torch.rand(2))
        determinism.seed_episode(3, 0)
        options = {"value": "original"}
        with (
            patch.dict(os.environ, {"SIMPLE_ISAAC_DETERMINISTIC": "1"}),
            patch("simple.deterministic_render.warmup"),
        ):
            env = Env()
            env.reset(seed=3, options=options)
            self.assertEqual(env.count, 2)
            self.assertEqual(env.sample[:2], expected[:2])
            self.assertTrue(torch.equal(env.sample[2], expected[2]))
            self.assertEqual(options["value"], "original")
            env.reset(seed=3, options=options)
            self.assertEqual(env.count, 3)
        with patch.dict(os.environ, {"SIMPLE_ISAAC_DETERMINISTIC": "0"}):
            env = Env()
            env.reset(options={})
            self.assertEqual(env.count, 1)


if __name__ == "__main__":
    unittest.main()
