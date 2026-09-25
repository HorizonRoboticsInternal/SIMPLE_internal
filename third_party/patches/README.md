# Local patches to the decoupled_wbc submodule

Base commit: `5b304e4d643fe89cd000068dcbba7e7519539caa` (remote: git@github.com:songlin/decoupled_wbc.git)

`decoupled_wbc_local.patch` carries every real-robot fix that REAL_ROBOT_RUNBOOK.md
depends on (realsense.py camera module, deploy_g1 flag forwarding + --no-add_stereo_camera,
HAND_VELOCITY_LIMIT 50->80, exporter base_pose/base_vel mapping, and related config/ROS
utils). A submodule update erases them; re-apply with:

    git -C third_party/decoupled_wbc apply ../patches/decoupled_wbc_local.patch
