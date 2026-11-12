#!/usr/bin/env bash
# The six SIMPLE psi0 tasks, one example episode each: Isaac replay with wrist cameras, then the
# updated dataset. Episodes picked from MuJoCo test replays that pass each task's own check.
#   PSI0_ROOT  folder holding the six psi0 sets (default: the NAS-28 simple-0721 copy)
#   OUT        output root (default data/psi0_replay)
set -u
cd "$(dirname "$0")/../.."
PSI0_ROOT=${PSI0_ROOT:-/mnt/nas28/kerou.zhang/datasets/SIMPLE_data/simple-0721/simple}
OUT=${OUT:-data/psi0_replay}
TASKS=(
  "G1WholebodyXMovePickTeleop-v0 0"
  "G1WholebodyXMoveBendPickTeleop-v0 4"
  "G1WholebodyHandoverTeleop-v0 27"
  "G1WholebodyLocomotionPickBetweenTablesTeleop-v0 5"
  "G1WholebodyBendPickMP-v0 0"
  "G1WholebodyTabletopGraspMP-v0 0"
)
for row in "${TASKS[@]}"; do
  set -- $row; task=$1; eps=$2
  src=${SRC_OVERRIDE:-$PSI0_ROOT/$task}
  [ -n "${LOCAL_MAP:-}" ] && src=$(python3 -c "import sys,json; print(json.load(open('$LOCAL_MAP'))['$task'])")
  echo "== $task episodes $eps  ($src)"
  for try in 1 2; do   # Isaac can abort in its first world reset under heavy CPU load: retry once
    scripts/psi0_replay/run_isaac.sh --data-dir "$src" --env-id "simple/$task" --episodes "$eps" \
        --wrist-cams --out "$OUT/$task" --skip-existing > "$OUT/$task.replay.log" 2>&1
    grep "^\[replay\]" "$OUT/$task.replay.log" && break
    echo "attempt $try failed:"; grep -E "Error|Aborted|terminate" "$OUT/$task.replay.log" | grep -v -i -E "lula|motion_gen|franka" | tail -3
  done
  .venv/bin/python scripts/psi0_replay/update_dataset.py --src "$src" --replays "$OUT/$task" \
      --out "$OUT/${task}_wrist" --overwrite
done
