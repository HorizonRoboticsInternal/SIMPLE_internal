# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

"""Head changes must preserve dynamics and stay out of shared asset directories."""

import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEAD = runpy.run_path(str(ROOT / "src/simple/robots/g1comp_assets.py"))


class G1CompAssetsTest(unittest.TestCase):
    def test_massless_head_preserves_robot(self):
        source = ROOT / "data/robots/g1_sonic/g1_29dof_with_hand.xml"
        assets = ROOT / "data/robots/g1comp/head_meshes"
        if not source.exists() or not assets.exists():
            self.skipTest("requires standard G1 SONIC and g1comp head assets")
        before = source.read_bytes()
        with (
            tempfile.TemporaryDirectory() as cache,
            patch.dict(
                os.environ,
                {
                    "SIMPLE_SCENE_CACHE_DIR": cache,
                    "SIMPLE_G1COMP_ASSET_DIR": str(assets),
                },
            ),
        ):
            output = HEAD["build_mjcf"](source)
            self.assertTrue(output.is_relative_to(cache))
            original = mujoco.MjModel.from_xml_path(str(source))
            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(
                (original.nq, original.nv, original.nu), (model.nq, model.nv, model.nu)
            )
            self.assertEqual(
                [original.joint(i).name for i in range(original.njnt)],
                [model.joint(i).name for i in range(model.njnt)],
            )
            for i in range(original.nbody):
                body = model.body(original.body(i).name)
                np.testing.assert_allclose(body.mass, original.body(i).mass)
                np.testing.assert_allclose(body.inertia, original.body(i).inertia)
            for name, _, _ in HEAD["HEAD_CHAIN"]:
                self.assertEqual(float(model.body(name).mass[0]), 0)
            np.testing.assert_allclose(
                HEAD["camera_position"](output),
                [0.03548974, -0.00095636, -0.036035],
                atol=1e-8,
            )
        self.assertEqual(source.read_bytes(), before)

if __name__ == "__main__":
    unittest.main()
