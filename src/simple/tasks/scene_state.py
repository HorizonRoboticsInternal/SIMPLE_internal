"""Restore kit furniture, or move it with an explicitly randomized table."""

import numpy as np
from scipy.spatial.transform import Rotation


def restore_furniture(layout, options, name, asset_id, position, quaternion, *, follows_height=False):
    state = (options or {}).get("state_dict") or {}
    actor = state.get("layout", {}).get("actors", {}).get(name)
    if actor is None:
        return asset_id, position, quaternion
    pose = actor["pose"]
    if (options or {}).get("dr_level") is None:
        return actor["asset"]["uid"], pose["position"], pose["quaternion"]

    # DR currently changes table height, but also support a planar layout change.
    saved = state["dr_state_dict"]["scene"]["table"]
    table = layout.actors["table"]
    old_q = saved["pose"]["quaternion"]
    new_q = table.pose.quaternion
    rotation = Rotation.from_quat(np.roll(new_q, -1)) * Rotation.from_quat(np.roll(old_q, -1)).inv()
    offset = np.asarray(pose["position"]) - saved["pose"]["position"]
    position = rotation.apply(offset) + table.pose.position
    position[2] = pose["position"][2]
    if follows_height:
        old_top = saved["pose"]["position"][2] + saved["size"][2] / 2
        new_top = table.pose.position[2] + table.size[2] / 2
        position[2] += new_top - old_top
    quaternion = np.roll((rotation * Rotation.from_quat(np.roll(pose["quaternion"], -1))).as_quat(), 1)
    # Keep the newly selected height variant, not the saved pre-DR asset.
    return asset_id, position.tolist(), quaternion.tolist()
