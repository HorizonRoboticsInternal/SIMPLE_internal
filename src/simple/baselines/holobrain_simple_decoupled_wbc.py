"""Decoupled-WBC SIMPLE client for HoloBrain through the Psi0 HTTP server."""

import numpy as np
from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent


class HolobrainSimpleDecoupledWbcAgent(Psi0DecoupledWbcAgent):
    def _build_state_dict(self, states, proprio, info):
        if proprio.shape != (1, 43):
            raise ValueError(
                f"Expected G1 joint state shape (1, 43), got {proprio.shape}."
            )
        state = {
            "states": states,
            "joint_qpos_43": proprio.astype(np.float32),
            "joint_names": list(self.robot.joint_names),
        }
        if (
            info is not None
            and "floating_base_pose" in info.get("proprio", {})
        ):
            state["floating_base_pose"] = np.asarray(
                info["proprio"]["floating_base_pose"], dtype=np.float32
            )
        return state
