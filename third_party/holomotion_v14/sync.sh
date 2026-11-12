#!/usr/bin/env bash
# Vendor the ROS-free parts of HoloMotion's v1.4 teleop-collection deployment into SIMPLE.
#
#   third_party/holomotion_v14/sync.sh [/path/to/holomotion checkout]     (default: ~/wrk/holomotion_v14_teleop)
#
# The checkout must be on robot-lab-internal/open-source/holomotion, branch feat/v14-teleop-collection
# (a sparse checkout of deployment/ is enough). Files are copied unchanged; SIMPLE supplies the ROS glue
# (src/simple/teleop/holomotion_v14/), so the policy maths, mode state machine, reference queue and
# PICO->SMPL->GMR retargeting stay identical to the robot's.
set -euo pipefail
SRC="${1:-$HOME/wrk/holomotion_v14_teleop}"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$SRC/deployment/packages"
[[ -d "$PKG/humanoid_control/humanoid_policy" ]] || { echo "not a v1.4 deployment checkout: $SRC" >&2; exit 1; }

rm -rf "$DST/humanoid_policy" "$DST/holomotion_policy_core" "$DST/holomotion_peripherals_ros2" "$DST/holomotion_teleop_ros2" "$DST/config"
mkdir -p "$DST/humanoid_policy/obs_builder" "$DST/humanoid_policy/utils" "$DST/holomotion_peripherals_ros2" \
         "$DST/holomotion_teleop_ros2" "$DST/config"

HP="$PKG/humanoid_control/humanoid_policy"
for f in __init__.py config.py cpu_affinity.py holomotion_fk_root_only.py launch_profile.py observation_evaluator.py onnx_policy.py \
         policy_node_29dof.py policy_runtime.py reference_contract.py v141_pico_control.py v14_control_safety.py \
         vr_reference.py; do
  cp "$HP/$f" "$DST/humanoid_policy/$f"
done
cp "$HP"/obs_builder/*.py "$DST/humanoid_policy/obs_builder/"
cp "$HP"/utils/*.py "$DST/humanoid_policy/utils/"
cp -r "$PKG/holomotion_policy_core/holomotion_policy_core" "$DST/holomotion_policy_core"
cp "$PKG/holomotion_peripherals_ros2/holomotion_peripherals_ros2/__init__.py" \
   "$PKG/holomotion_peripherals_ros2/holomotion_peripherals_ros2/pico_dex3_gripper.py" "$DST/holomotion_peripherals_ros2/"
cp "$PKG/holomotion_teleop_ros2/holomotion_teleop_ros2/__init__.py" \
   "$PKG/holomotion_teleop_ros2/holomotion_teleop_ros2/converter.py" \
   "$PKG/holomotion_teleop_ros2/holomotion_teleop_ros2/latest_obs_zmq.py" "$DST/holomotion_teleop_ros2/"
cp "$PKG/humanoid_control/config/g1_29dof_holomotion.yaml" "$DST/config/"
find "$DST" -name "__pycache__" -type d -prune -exec rm -rf {} +
# backpack: the v1.4.1 backpack policy's model lock (this branch) and HoloMotion's G1 backpack mesh (payload branch)
PAYLOAD_REF="${PAYLOAD_REF:-origin/feat/g1-payload-multi-urdf-dual-mujoco-eval}"
mkdir -p "$DST/backpack"
cp "$SRC/deployment/images/locks/unitree/v141.brainco_backpack_3p2.model.lock" "$DST/backpack/"
if git -C "$SRC" cat-file -e "$PAYLOAD_REF:assets/robots/unitree/G1/g1_with_hand_0715_backpack/meshes/backpack_link.STL" 2>/dev/null; then
  git -C "$SRC" show "$PAYLOAD_REF:assets/robots/unitree/G1/g1_with_hand_0715_backpack/meshes/backpack_link.STL" > "$DST/backpack/backpack_link.STL"
else
  echo "note: $PAYLOAD_REF not fetched; kept the existing backpack/backpack_link.STL" >&2
fi

{
  echo "# Vendored HoloMotion v1.4 teleop-collection code"
  echo
  echo "- source: robot-lab-internal/open-source/holomotion, branch \`$(git -C "$SRC" rev-parse --abbrev-ref HEAD)\`"
  echo "- commit: \`$(git -C "$SRC" rev-parse HEAD)\` ($(git -C "$SRC" log -1 --format='%cd' --date=short))"
  echo "- synced: $(date +%Y-%m-%d) by \`third_party/holomotion_v14/sync.sh\`"
  echo
  echo "Unchanged copies of the ROS-free modules; do not edit them here -- re-run sync.sh instead."
  echo "SIMPLE's glue (ROS stubs, sim lowstate, PICO input, ZMQ reference) is in \`src/simple/teleop/holomotion_v14/\`."
  echo
  echo "| vendored | from |"
  echo "|---|---|"
  echo "| \`humanoid_policy/\` | \`deployment/packages/humanoid_control/humanoid_policy/\` (policy node, runtime, evaluator, reference queue, PICO control) |"
  echo "| \`holomotion_policy_core/\` | \`deployment/packages/holomotion_policy_core/\` (obs builder, ONNX loading, root-only FK, remote controller) |"
  echo "| \`holomotion_peripherals_ros2/pico_dex3_gripper.py\` | Dex3 grip open/close poses |"
  echo "| \`holomotion_teleop_ros2/{converter,latest_obs_zmq}.py\` | PICO body -> SMPL -> GMR retargeter, ZMQ \`obs65\` wire format |"
  echo "| \`config/g1_29dof_holomotion.yaml\` | robot config (dof orders, default angles, move-to-default gains) |"
  echo "| \`backpack/v141.brainco_backpack_3p2.model.lock\` | \`deployment/images/locks/unitree/\` (hashes of the backpack motion model \`model_22000\`, its config, the velocity model) |"
  echo "| \`backpack/backpack_link.STL\` | branch \`feat/g1-payload-multi-urdf-dual-mujoco-eval\` (commit \`$(git -C "$SRC" rev-parse --short=8 "$PAYLOAD_REF" 2>/dev/null || echo unknown)\`), \`assets/robots/unitree/G1/g1_with_hand_0715_backpack/meshes/\` |"
} > "$DST/SOURCE.md"
echo "vendored into $DST from $(git -C "$SRC" rev-parse --short HEAD)"
