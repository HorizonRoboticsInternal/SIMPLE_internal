#!/usr/bin/env bash
# Minimal local example of SIMPLE's SCRIPTED data collection
# (automated motion planning, no VR headset, no human in the loop).
#
# Task: simple/FrankaTabletopGraspMP-v0
#   OpenGripper -> GraspObject (GraspNet cached grasp + CuRobo plan) -> CloseGripper -> Lift
#
# Output is a LeRobot-format dataset under $OUT.
#
# SIM=mujoco       (default) physics only, no Isaac. Fast, but the recorded RGB is an
#                  untextured MuJoCo render — fine for checking the pipeline, not for training.
# SIM=mujoco_isaac Isaac renders photorealistic frames in the SAME pass. Same dataset
#                  schema, real pixels. First launch pulls ~1.5 GB of scene/material
#                  assets and spends a few minutes booting Isaac.
# Scripted data needs no replay stage; that is teleop-only.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# --- CUDA toolkit used to build/run CuRobo on this box (no system CUDA) ---
export CUDA_HOME=/home/Horizon/miniconda3/envs/bodex
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# --- headless rendering / Isaac EULA ---
export MUJOCO_GL=egl
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TASK=${TASK:-FrankaTabletopGraspMP-v0}
# graspnet1b:5 is the target the task's own DR config defaults to, and it plans
# reliably here. Object 0 has two stable poses with empty grasp caches, so a share of
# resets fail before planning even starts.
OBJ=${OBJ:-graspnet1b:5}
N=${N:-2}
SIM=${SIM:-mujoco}
OUT=${OUT:-data/datagen_example}

source .venv/bin/activate

# LeRobotDataset.create refuses to write into an existing directory.
rm -rf "$OUT"

# Two flags the CLI defaults get wrong for this task:
#   --plan-batch-size 40  matches docs/source/tutorials/data_gen.md; the default of 1
#                         hands CuRobo a single grasp candidate per attempt.
#   --easy-motion-gen     restores CuRoboPlanner's own default. The CLI passes False,
#                         which builds the full collision world and fails IK constantly.
python -m simple.cli.datagen "simple/$TASK" \
  --sim-mode "$SIM" \
  --headless \
  --no-webrtc \
  --target-object "$OBJ" \
  --num-episodes "$N" \
  --plan-batch-size 40 \
  --easy-motion-gen \
  --save-dir "$OUT"

echo
echo "Dataset written under $OUT/simple/$TASK/level-0/"
