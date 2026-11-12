# Vendored HoloMotion v1.4 teleop-collection code

- source: robot-lab-internal/open-source/holomotion, branch `feat/v14-teleop-collection`
- commit: `a465fd83127e0e2d96f1d00ce0b9681881d2d851` (2026-09-22)
- synced: 2026-09-28 by `third_party/holomotion_v14/sync.sh`

Unchanged copies of the ROS-free modules; do not edit them here -- re-run sync.sh instead.
SIMPLE's glue (ROS stubs, sim lowstate, PICO input, ZMQ reference) is in `src/simple/teleop/holomotion_v14/`.

| vendored | from |
|---|---|
| `humanoid_policy/` | `deployment/packages/humanoid_control/humanoid_policy/` (policy node, runtime, evaluator, reference queue, PICO control) |
| `holomotion_policy_core/` | `deployment/packages/holomotion_policy_core/` (obs builder, ONNX loading, root-only FK, remote controller) |
| `holomotion_peripherals_ros2/pico_dex3_gripper.py` | Dex3 grip open/close poses |
| `holomotion_teleop_ros2/{converter,latest_obs_zmq}.py` | PICO body -> SMPL -> GMR retargeter, ZMQ `obs65` wire format |
| `config/g1_29dof_holomotion.yaml` | robot config (dof orders, default angles, move-to-default gains) |
| `backpack/v141.brainco_backpack_3p2.model.lock` | `deployment/images/locks/unitree/` (hashes of the backpack motion model `model_22000`, its config, the velocity model) |
| `backpack/backpack_link.STL` | branch `feat/g1-payload-multi-urdf-dual-mujoco-eval` (commit `cfa436cd`), `assets/robots/unitree/G1/g1_with_hand_0715_backpack/meshes/` |
