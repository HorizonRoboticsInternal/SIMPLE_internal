"""Standard SIMPLE MP client for HoloBrain through the Psi0 HTTP server."""

import numpy as np

from simple.baselines.psi0 import Psi0Agent


class HolobrainSimpleAgent(Psi0Agent):
    """Add the full named G1 joint state required by HoloBrain."""

    def _build_state_dict(self, states, proprio, info):
        if proprio.shape != (1, 43):
            raise ValueError(
                f"Expected G1 joint state shape (1, 43), got {proprio.shape}."
            )
        return {
            "states": states,
            "joint_qpos_43": proprio.astype(np.float32),
            "joint_names": list(self.robot.joint_names),
        }
