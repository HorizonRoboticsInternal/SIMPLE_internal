#!/usr/bin/env bash
set -euo pipefail

SIMPLE_ROOT=${SIMPLE_ROOT:-/mnt/nas26/kerou.zhang/robotics/SIMPLE}
PSI_ROOT=${PSI_ROOT:-/mnt/nas26/kerou.zhang/robotics/Psi0}
BENCH_ROOT=${BENCH_ROOT:-/mnt/nas26/kerou.zhang/robotics/simple-benchmark-reproduction}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# ===== Frequently changed evaluation configuration =====
JOB_DIR=${JOB_DIR:-${1:-/mnt/nas26/kerou.zhang/robotics/Psi0/.runs/finetune/simple-14tasks.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2609022352}}
CKPT_STEP=${CKPT_STEP:-${2:-30000}}
EPISODES=${EPISODES:-3}
SEED=${SEED:-}
SETTING=${SETTING:-psi0}
GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
TASKS=(
  G1WholebodyBendHandoverTeleop-v0
  G1WholebodyBendPickTeleop-v0
  G1WholebodyCloseDoorTeleop-v0
  G1WholebodyOpenFaucetTeleop-v0
  G1WholebodyOpenOvenTeleop-v0
  G1WholebodyOpenTrashCanTeleop-v0
  G1WholebodyPickAndPlaceAndHugContainerTeleop-v0
  G1WholebodyPushOfficeChairTeleop-v0
)
DATA_ROOT=${DATA_ROOT:-/mnt/nas28/kerou.zhang/datasets/SIMPLE_data/simple-0721/simple-eval}
# =======================================================

if [[ -z "$JOB_DIR" || $# -gt 2 ]]; then
  echo "Usage: $0 JOB_DIR [CKPT_STEP]" >&2
  exit 2
fi
JOB_DIR=$(readlink -f "$JOB_DIR")
OUTPUT_PREFIX=$(basename "$JOB_DIR")
OUT_ROOT=${OUT_ROOT:-$SIMPLE_ROOT/data/closed_loop_evals/old_eval_data_0721/${OUTPUT_PREFIX}_other_old_tasks_ckpt${CKPT_STEP}_$STAMP}

PSI_SERVER=${PSI_SERVER:-$PSI_ROOT/.venv/bin/serve_psi0}
SIMPLE_PYTHON=${SIMPLE_PYTHON:-$SIMPLE_ROOT/.venv/bin/python}
CACHE_ROOT=${CACHE_ROOT:-$BENCH_ROOT/cache/other-old-tasks}
TMP_ROOT=${TMP_ROOT:-$BENCH_ROOT/tmp/other-old-tasks-$STAMP}
BASE_PORT=${BASE_PORT:-22200}
SESSION=${TMUX_SESSION:-psi0-other-old-$STAMP}
ATTACH=${ATTACH:-1}
read -r -a GPUS <<< "$GPU_LIST"

tmux new-session -d -s "$SESSION" -n evaluation

for worker in "${!GPUS[@]}"; do
  GPU=${GPUS[$worker]}
  PORT=$((BASE_PORT + GPU))
  CACHE="$CACHE_ROOT/gpu-$GPU"
  TMP="$TMP_ROOT/gpu-$GPU"
  ASSIGNED_TASKS=()
  for ((task_index = worker; task_index < ${#TASKS[@]}; task_index += ${#GPUS[@]})); do
    ASSIGNED_TASKS+=("${TASKS[$task_index]}")
  done
  TASK_NAMES=${ASSIGNED_TASKS[*]}

  read -r -d '' COMMAND <<EOF || true
set -e
if [[ -n "$SEED" ]]; then
  export PYTHONHASHSEED="$SEED"
  export PSI0_EVAL_SEED="$SEED"
  export PSI_EVAL_SEED="$SEED"
  export SIMPLE_EVAL_SEED="$SEED"
else
  unset PYTHONHASHSEED PSI0_EVAL_SEED PSI_EVAL_SEED SIMPLE_EVAL_SEED
fi
mkdir -p "$OUT_ROOT" "$TMP/server" "$TMP/eval" "$CACHE/optix" "$CACHE/cuda" "$CACHE/xdg"

cd "$PSI_ROOT"
env -u LD_LIBRARY_PATH \\
  CUDA_VISIBLE_DEVICES=$GPU \\
  TMPDIR="$TMP/server" \\
  PYTHONUNBUFFERED=1 \\
  "$PSI_SERVER" \\
    --host 127.0.0.1 \\
    --port $PORT \\
    --device cuda:0 \\
    --policy psi0 \\
    --run-dir "$JOB_DIR" \\
    --ckpt-step $CKPT_STEP \\
    --action-exec-horizon 24 \\
    --rtc >"$OUT_ROOT/server_gpu-$GPU.log" 2>&1 &
SERVER_PID=\$!
trap 'kill -TERM \$SERVER_PID 2>/dev/null || true' EXIT
until curl --silent --fail http://127.0.0.1:$PORT/health >/dev/null; do sleep 2; done

cd "$SIMPLE_ROOT"
for TASK in $TASK_NAMES; do
  mkdir -p "$OUT_ROOT/\$TASK"
  for LEVEL in level-0 level-1 level-2; do
    env -u LD_LIBRARY_PATH \\
      CUDA_VISIBLE_DEVICES=$GPU \\
      TMPDIR="$TMP/eval" \\
      MUJOCO_GL=egl \\
      PYOPENGL_PLATFORM=egl \\
      OPTIX_CACHE_PATH="$CACHE/optix" \\
      CUDA_CACHE_PATH="$CACHE/cuda" \\
      XDG_CACHE_HOME="$CACHE/xdg" \\
      PYTHONUNBUFFERED=1 \\
      OMNI_KIT_ACCEPT_EULA=Y \\
      TASK_NAME="\$TASK" \\
      "$SIMPLE_PYTHON" "$SIMPLE_ROOT/src/simple/cli/eval_decoupled_wbc.py" \\
        "simple/\$TASK" \\
        psi0_decoupled_wbc \\
        "\$LEVEL" \\
        --data-format lerobot \\
        --data-dir "$DATA_ROOT/\$TASK/\$LEVEL" \\
        --eval-dir "$OUT_ROOT/\$TASK/\$LEVEL" \\
        --host 127.0.0.1 \\
        --port $PORT \\
        --sim-mode mujoco_isaac \\
        --headless \\
        --num-episodes $EPISODES \\
        --save-video
  done
done
EOF

  if ((worker == 0)); then
    PANE=$(tmux display-message -p -t "$SESSION:0.0" '#{pane_id}')
    tmux respawn-pane -k -t "$PANE" bash -c "$COMMAND"
  else
    PANE=$(tmux split-window -d -P -F '#{pane_id}' -t "$SESSION:0" bash -c "$COMMAND")
  fi
  tmux select-pane -t "$PANE" -T "GPU $GPU: $TASK_NAMES"
  tmux select-layout -t "$SESSION:0" tiled >/dev/null
done

tmux new-window -d -t "$SESSION" -n summary \
  "$SIMPLE_PYTHON" "$SIMPLE_ROOT/scripts/summarize_psi0_eval.py" \
    --wait \
    --out-root "$OUT_ROOT" \
    --episodes "$EPISODES" \
    --setting "$SETTING" \
    "${TASKS[@]}"

tmux set-option -t "$SESSION" remain-on-exit on
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format ' #{pane_title} '

echo "Results: $OUT_ROOT"
if [[ "$ATTACH" != 1 ]]; then
  echo "Attach with: tmux attach -t $SESSION"
elif [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "$SESSION"
else
  tmux attach-session -t "$SESSION"
fi
