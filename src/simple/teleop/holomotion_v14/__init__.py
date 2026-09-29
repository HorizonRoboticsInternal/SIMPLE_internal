"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

HoloMotion v1.4 (teleop-collection branch) for SIMPLE teleoperation.

The robot's own policy code is vendored unchanged in ``third_party/holomotion_v14`` (see its SOURCE.md). This package
only supplies what ROS supplies on the robot:

* ``ros_stubs``   minimal stand-ins for rclpy and the ROS message types, so ``policy_node_29dof`` imports without ROS;
* ``sim_node``    ``SimPolicyNode``: the robot's ``HoloMotionPolicyNode`` fed from MuJoCo instead of topics;
* ``wire``        the ZMQ channel from the reference publisher (``obs65`` latest_obs frames, as on the robot, plus the
                  PICO controller sample);
* ``publisher``   the reference publisher: PICO body tracking -> SMPL -> GMR -> ``obs65`` (runs in the
                  ``holomotion_teleop`` conda env, which has SMPL/GMR).

Operator buttons are the robot's (v1.4.1 bit map): left stick click = stand (MOVE_TO_DEFAULT), right A = policy on in
walking mode, right B = motion tracking, Y = back to walking, X = zero torque, right stick click = emergency stop,
joysticks = walk/turn, grips = close the Dex3 hands. Sim-only actions use combos the robot never reads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
VENDOR_DIR = Path(os.environ.get("HOLOMOTION_V14_VENDOR_DIR", REPO_ROOT / "third_party" / "holomotion_v14"))
MODELS_DIR = Path(os.environ.get("HOLOMOTION_V14_MODELS_DIR", REPO_ROOT / "data" / "holomotion" / "v14_models"))
ROBOT_CONFIG = VENDOR_DIR / "config" / "g1_29dof_holomotion.yaml"

DEFAULT_REFERENCE_URI = "tcp://127.0.0.1:6001"     # the robot node's zmq_uri (it binds tcp://*:6001)
OBS65_TOPIC = b"obs65"
PICO_TOPIC = b"pico_ctrl"


def ensure_vendor_path() -> None:
    """Make the vendored v1.4 packages importable (humanoid_policy, holomotion_policy_core, ...)."""
    path = str(VENDOR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("HOLOMOTION_UNITREE_POLICY_MODELS_DIR", str(MODELS_DIR))
