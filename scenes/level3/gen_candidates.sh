#!/usr/bin/env bash
# Step 1: 30 level-3 candidate scenes per benchmark task, one Isaac instance at a time (MuJoCo + Isaac, SIMPLE's DR level 3).
# Each candidate draws new distractors, table material, lighting and object pose; the robot start and table height get the
# offsets of lv3_30_design.json (SIMPLE_LEVEL3_*). These offsets are NOT reachability-checked yet: build_lv3_feasible.py
# replaces them per scene. Output: $WORK/candidates/<task>/dr-level-3.
#   bash scenes/level3/gen_candidates.sh [task ...]        (default: the five benchmark tasks)
# env: LEVEL3_BASE_ROOT (the 20-scene level-0 base sets, default /mnt/nas28/alan.jiang/simple-eval-new20), WORK (default data/level3_work)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"
PY=${PY:-.venv/bin/python}; WORK=${WORK:-$ROOT/data/level3_work}; N=30
BASE=${LEVEL3_BASE_ROOT:-/mnt/nas28/alan.jiang/simple-eval-new20}
export MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES LEVELGEN_DDS_DOMAIN=${LEVELGEN_DDS_DOMAIN:-37}     # own DDS domain: never competes with other workers' participants
export SIMPLE_LEVEL3_EPISODES=$N SIMPLE_LEVEL3_BACK=0.10 SIMPLE_LEVEL3_SIDE=0.05 SIMPLE_LEVEL3_SEED=0 SIMPLE_LEVEL3_TABLE_DZ=0.04
TASKS=("$@"); [ ${#TASKS[@]} -gt 0 ] || TASKS=(G1WholebodyTabletopGraspMP-v0 G1WholebodyBendPickMP-v0 G1WholebodyXMovePickTeleop-v0 G1WholebodyHandoverTeleop-v0 G1WholebodyLocomotionPickBetweenTablesTeleop-v0)
mkdir -p $WORK/logs
ok() { [ -f "$1/meta/episodes.jsonl" ] && [ "$(grep -c . "$1/meta/episodes.jsonl")" -ge $N ]; }
for t in "${TASKS[@]}"; do
  dst=$WORK/candidates/$t/dr-level-3; ok $dst && { echo "skip $t (done)"; continue; }
  for try in 1 2 3; do
    TMP=$WORK/gen/$t; rm -rf $TMP; mkdir -p $TMP; echo "=== candidates $t try $try $(date +%T)"
    if [[ $t == *MP-v0 ]]; then        # MP tasks: AMO robot, no DDS; the base has 20 scenes and dr.py loops over them
      timeout 3600 $PY -m simple.cli.dr simple/$t --env-config-dir $BASE/$t/dr-level-0 --dr-level 3 --num-episodes $N --sim-mode mujoco_isaac \
        --render-hz 50 --headless --no-webrtc --keep-object-materials --save-dir "$TMP" > $WORK/logs/gen_$t.log 2>&1
      src=$(ls -d "$TMP/$t/level-3" "$TMP/simple/$t/level-3" "$TMP/simple/$t/dr-level-3" "$TMP/$t/dr-level-3" 2>/dev/null | head -1)
    else                               # decoupled-WBC tasks: the CLI caps at the base size, levelgen cycles the 20 base scenes to 30
      LEVELGEN_CYCLE_TO=$N timeout 3600 $PY $HERE/levelgen.py dr_wbc simple/$t --data-dir $BASE/$t/dr-level-0 --dr-level 3 --num-episodes $N \
        --headless --keep-object-materials --save-dir "$TMP" > $WORK/logs/gen_$t.log 2>&1
      src=$(ls -d "$TMP/simple/$t/dr-level-3" "$TMP/$t/dr-level-3" 2>/dev/null | head -1)
    fi
    if [ -n "$src" ] && ok "$src"; then mkdir -p $(dirname $dst); rm -rf $dst; mv "$src" $dst; echo "candidates $t ok $(date +%T)"; break; fi
    echo "candidates $t try $try failed (log $WORK/logs/gen_$t.log)"; sleep 10
  done
done
