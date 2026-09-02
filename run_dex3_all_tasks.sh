#!/usr/bin/env bash
# Generate 2 episodes (=2 videos per camera) for every G1 Dex3 scripted MP task.
#
# Requires the patched G1 MJCF (position actuators on the Dex3 hand); with the
# upstream <motor> actuators every episode ends as "motion plan exhausted".
#
# Dex3 tasks must use --plan-batch-size 1: the BoDex planner's batch>1 branch in
# src/simple/mp/curobo.py is an unimplemented `...` stub.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_HOME=/home/Horizon/miniconda3/envs/bodex
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

TASK_LIST=${TASK_LIST:-/tmp/claude-1000/-home-Horizon-wrk-SIMPLE/66a4ba73-f074-4719-b3c1-1268e8145abc/scratchpad/dex3_tasks.txt}
OUT=${OUT:-data/dex3_all}
N=${N:-2}
PER_TASK_TIMEOUT=${PER_TASK_TIMEOUT:-900}
LOGDIR=${LOGDIR:-/tmp/claude-1000/-home-Horizon-wrk-SIMPLE/66a4ba73-f074-4719-b3c1-1268e8145abc/scratchpad/dex3_logs}
SUMMARY=${SUMMARY:-$LOGDIR/summary.tsv}

source .venv/bin/activate
mkdir -p "$LOGDIR"
: > "$SUMMARY"

while read -r TASK; do
  [ -z "$TASK" ] && continue
  short="${TASK#simple/}"
  dir="$OUT/$short"
  rm -rf "$dir"
  timeout "$PER_TASK_TIMEOUT" python -u -m simple.cli.datagen "$TASK" \
      --sim-mode mujoco --headless --no-webrtc \
      --num-episodes "$N" --plan-batch-size 1 \
      --save-dir "$dir" > "$LOGDIR/$short.log" 2>&1
  rc=$?
  eps=$(find "$dir" -name '*.parquet' 2>/dev/null | wc -l)
  vids=$(find "$dir" -name '*.mp4' 2>/dev/null | wc -l)
  printf '%s\t%s\t%s\t%s\n' "$short" "$rc" "$eps" "$vids" >> "$SUMMARY"
  echo "[dex3-all] $short rc=$rc episodes=$eps videos=$vids"
done < "$TASK_LIST"

echo "[dex3-all] DONE"
awk -F'\t' '{e+=$3; v+=$4; n++} END{printf "[dex3-all] %d tasks, %d episodes, %d videos\n", n, e, v}' "$SUMMARY"
