"""Init / home pose presets and the teleop defaults (init pose, return-to-init after save, gravity feed-forward)."""
import json
import numpy as np
import pytest

from decoupled_wbc.control.robot_model.instantiation.g1 import INIT_POSE_PRESETS, apply_init_pose, instantiate_g1_robot_model

ARM = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
HAND = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
USER_JSON = {"left_hip_pitch_joint": 0, "right_hip_pitch_joint": 0, "waist_yaw_joint": 0, "left_knee_joint": 0,
             "left_shoulder_pitch_joint": 0, "right_shoulder_pitch_joint": 0, "left_shoulder_roll_joint": 0, "right_shoulder_roll_joint": 0,
             "left_shoulder_yaw_joint": 0, "right_shoulder_yaw_joint": 0, "left_elbow_joint": -0.66, "right_elbow_joint": -0.66,
             "left_wrist_roll_joint": 0, "right_wrist_roll_joint": 0, "left_wrist_pitch_joint": 0, "right_wrist_pitch_joint": 0,
             "left_wrist_yaw_joint": 0, "right_wrist_yaw_joint": 0,
             **{h: 0 for h in HAND}}


def _upper(rm):
    inv = {v: k for k, v in rm.joint_to_dof_index.items()}
    return {inv[i]: float(rm.default_body_pose[i]) for i in rm.get_joint_group_indices("upper_body")}


def test_default_is_elbows_raised(monkeypatch):
    monkeypatch.delenv("DECOUPLED_WBC_INIT_POSE", raising=False)
    rm = instantiate_g1_robot_model()
    q = _upper(rm)
    assert rm.init_pose_name == "elbows_raised"
    assert q["left_elbow_joint"] == pytest.approx(-0.66) and q["right_elbow_joint"] == pytest.approx(-0.66)
    assert all(abs(q[j]) < 1e-9 for j in ARM + HAND if "elbow" not in j), "every other upper-body joint must be 0"
    assert np.array_equal(rm.initial_body_pose, rm.default_body_pose)


def test_user_json_matches_preset(tmp_path):
    f = tmp_path / "pose.json"; f.write_text(json.dumps(USER_JSON))
    rm = instantiate_g1_robot_model(init_pose=str(f))
    rm2 = instantiate_g1_robot_model(init_pose="elbows_raised")
    assert np.allclose(rm.default_body_pose, rm2.default_body_pose), "the user's JSON (legs/waist entries ignored) == the preset"


def test_canonical_preset_and_env_override(monkeypatch):
    rm = instantiate_g1_robot_model(init_pose="canonical"); q = _upper(rm)
    assert q["left_shoulder_roll_joint"] == pytest.approx(0.2) and q["right_shoulder_roll_joint"] == pytest.approx(-0.2) and q["left_elbow_joint"] == 0.0
    monkeypatch.setenv("DECOUPLED_WBC_INIT_POSE", "canonical")
    assert instantiate_g1_robot_model().init_pose_name == "canonical"


def test_out_of_limit_values_are_clipped_and_legs_ignored(capsys):
    rm = instantiate_g1_robot_model(init_pose="canonical")
    apply_init_pose(rm, {"left_elbow_joint": -3.0, "left_knee_joint": 0.5}, "custom")
    q = _upper(rm)
    assert q["left_elbow_joint"] == pytest.approx(float(rm.lower_joint_limits[rm.joint_to_dof_index["left_elbow_joint"] - (len(rm.default_body_pose) - len(rm.lower_joint_limits))]))
    assert "clipped" in capsys.readouterr().out
    assert rm.default_body_pose[rm.joint_to_dof_index["left_knee_joint"]] == 0.0


def test_shoulder_roll_zero_is_inside_limits():
    rm = instantiate_g1_robot_model()
    off = len(rm.default_body_pose) - len(rm.lower_joint_limits)
    for j, sign in (("left_shoulder_roll_joint", 1), ("right_shoulder_roll_joint", -1)):
        i = rm.joint_to_dof_index[j] - off
        assert rm.lower_joint_limits[i] <= 0.0 <= rm.upper_joint_limits[i]
    assert INIT_POSE_PRESETS["elbows_raised"] == {"left_elbow_joint": -0.66, "right_elbow_joint": -0.66}


def test_teleop_defaults_return_home_with_gravity_comp():
    """Every save -> ramp back to the init pose, with the arm gravity feed-forward on: both must be the DEFAULT."""
    from decoupled_wbc.control.main.teleop.configs.configs import BaseConfig
    d = {k: f.default for k, f in BaseConfig.__dataclass_fields__.items()}
    assert d["init_pose"] == "elbows_raised"
    assert d["home_after_save"] is True
    assert d["enable_gravity_compensation"] is True
    assert d["gravity_compensation_joints"] in (None, ["arms"])       # None -> ["arms"] in G1Env
