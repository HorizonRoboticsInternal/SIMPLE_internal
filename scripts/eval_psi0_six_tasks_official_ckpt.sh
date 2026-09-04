#!/usr/bin/env bash
set -euo pipefail

SIMPLE_ROOT=${SIMPLE_ROOT:-/mnt/nas26/kerou.zhang/robotics/SIMPLE}
PSI_ROOT=${PSI_ROOT:-/mnt/nas26/kerou.zhang/robotics/Psi0}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# ===== Frequently changed evaluation configuration =====
# Leave empty to use RUN_DIRS (one official checkpoint per task).
# Set it to a job directory to evaluate one checkpoint on every task.
JOB_DIR=${JOB_DIR:-}
CKPT_STEP=${CKPT_STEP:-40000}
EPISODES=${EPISODES:-10}
SEED=${SEED:-}
SETTING=${SETTING:-psi0}
TASKS=(
  G1WholebodyXMovePickTeleop-v0
  G1WholebodyBendPickMP-v0
  G1WholebodyHandoverTeleop-v0
  G1WholebodyLocomotionPickBetweenTablesTeleop-v0
  G1WholebodyTabletopGraspMP-v0
  G1WholebodyXMoveBendPickTeleop-v0
)
RUN_DIRS=(
  "$PSI_ROOT/.runs/psi0/simple-checkpoints/g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205"
  "$PSI_ROOT/.runs/psi0/simple-checkpoints/g1wholebodybendpick-v0.simple.flow1000.cosine.lr1.0e-04.b256.gpus8.2603151312"
  "$PSI_ROOT/.runs/psi0/simple-checkpoints/g1wholebodyhandover-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604071507"
  "$PSI_ROOT/.runs/psi0/simple-checkpoints/g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126"
  "$PSI_ROOT/.runs/psi0/simple-checkpoints/g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503"
  "$PSI_ROOT/.runs/psi0/simple-checkpoints/g1wholebodyxmovebendpickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b112.gpus7.2604100422"
)
DATA_ROOT=${DATA_ROOT:-/mnt/nas28/kerou.zhang/datasets/SIMPLE_data/simple-0721/simple-eval}
if [[ -n "$JOB_DIR" ]]; then
  OUTPUT_PREFIX=$(basename "$JOB_DIR")
else
  OUTPUT_PREFIX=6_official_ckpts
fi
OUT_ROOT=${OUT_ROOT:-$SIMPLE_ROOT/data/closed_loop_evals/old_eval_data_0721/${OUTPUT_PREFIX}_ckpt${CKPT_STEP}_$STAMP}
# =======================================================

BENCH_ROOT=${BENCH_ROOT:-/mnt/nas26/kerou.zhang/robotics/simple-benchmark-reproduction}
PSI_SERVER=${PSI_SERVER:-$PSI_ROOT/.venv/bin/serve_psi0}
SIMPLE_PYTHON=${SIMPLE_PYTHON:-$SIMPLE_ROOT/.venv/bin/python}
CACHE_ROOT=${CACHE_ROOT:-$BENCH_ROOT/cache/six-tasks}
TMP_ROOT=${TMP_ROOT:-$BENCH_ROOT/tmp/six-tasks-$STAMP}
SESSION=${TMUX_SESSION:-psi0-eval-$STAMP}
ATTACH=${ATTACH:-1}

tmux new-session -d -s "$SESSION" -n evaluation

PANE_COUNT=0
for i in "${!TASKS[@]}"; do
  TASK=${TASKS[$i]}
  RUN_DIR=${JOB_DIR:-${RUN_DIRS[$i]}}
  RUN="$OUT_ROOT/$TASK"

  if [[ "$TASK" == *Teleop-v0 ]]; then
    EVAL_SCRIPT=src/simple/cli/eval_decoupled_wbc.py
    POLICY=psi0_decoupled_wbc
  else
    EVAL_SCRIPT=src/simple/cli/eval.py
    POLICY=psi0
  fi

  if [[ "$TASK" == G1WholebodyLocomotionPickBetweenTablesTeleop-v0 ]]; then
    LEVEL_GROUPS=(level-0 level-1 level-2)
    WORKER_GPUS=(3 6 7)
  else
    LEVEL_GROUPS=("level-0 level-1 level-2")
    WORKER_GPUS=("$i")
  fi

  for worker in "${!LEVEL_GROUPS[@]}"; do
    LEVELS=${LEVEL_GROUPS[$worker]}
    GPU=${WORKER_GPUS[$worker]}
    PORT=$((22100 + GPU))
    CACHE="$CACHE_ROOT/$TASK/gpu-$GPU"
    TMP="$TMP_ROOT/$TASK/gpu-$GPU"

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
mkdir -p "$RUN" "$TMP/server" "$TMP/eval" "$CACHE/optix" "$CACHE/cuda" "$CACHE/xdg"

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
    --run-dir "$RUN_DIR" \\
    --ckpt-step $CKPT_STEP \\
    --action-exec-horizon 24 \\
    --rtc >"$RUN/server_gpu-$GPU.log" 2>&1 &
SERVER_PID=\$!
trap 'kill -TERM \$SERVER_PID 2>/dev/null || true' EXIT
until curl --silent --fail http://127.0.0.1:$PORT/health >/dev/null; do sleep 2; done

cd "$SIMPLE_ROOT"
for LEVEL in $LEVELS; do
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
    TASK_NAME="$TASK" \\
    "$SIMPLE_PYTHON" "$SIMPLE_ROOT/$EVAL_SCRIPT" \\
      "simple/$TASK" \\
      "$POLICY" \\
      "\$LEVEL" \\
      --data-format lerobot \\
      --data-dir "$DATA_ROOT/$TASK/\$LEVEL" \\
      --eval-dir "$RUN/\$LEVEL" \\
      --host 127.0.0.1 \\
      --port $PORT \\
      --sim-mode mujoco_isaac \\
      --headless \\
      --num-episodes $EPISODES \\
      --save-video
done
EOF

    if ((PANE_COUNT == 0)); then
      PANE=$(tmux display-message -p -t "$SESSION:0.0" '#{pane_id}')
      tmux respawn-pane -k -t "$PANE" bash -c "$COMMAND"
    else
      PANE=$(tmux split-window -d -P -F '#{pane_id}' -t "$SESSION:0" bash -c "$COMMAND")
    fi
    tmux select-pane -t "$PANE" -T "$TASK ($LEVELS)"
    tmux select-layout -t "$SESSION:0" tiled >/dev/null
    PANE_COUNT=$((PANE_COUNT + 1))
  done
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
