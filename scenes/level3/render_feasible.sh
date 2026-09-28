#!/usr/bin/env bash
# Step 4: render the built (and, for the MP tasks, planner-checked) scene states EXACTLY -- levelgen.py LEVELGEN_EXACT=1 loads
# every stored state instead of re-drawing parts of it -- one Isaac instance at a time, and install them as
# data/evals_scenes_benchmark/<task>/dr-level-3 with meta/feasibility_build.json (+ meta/feasibility_planner.jsonl).
#   bash scenes/level3/render_feasible.sh [task ...]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"
PY=${PY:-.venv/bin/python}; WORK=${WORK:-$ROOT/data/level3_work}; OUT=data/evals_scenes_benchmark; N=30
export MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES LEVELGEN_DDS_DOMAIN=${LEVELGEN_DDS_DOMAIN:-37} LEVELGEN_EXACT=1
TASKS=("$@"); [ ${#TASKS[@]} -gt 0 ] || TASKS=(G1WholebodyTabletopGraspMP-v0 G1WholebodyBendPickMP-v0 G1WholebodyXMovePickTeleop-v0 G1WholebodyHandoverTeleop-v0 G1WholebodyLocomotionPickBetweenTablesTeleop-v0 G1WholebodyXMoveBendPickTeleop-v0)
mkdir -p $WORK/logs
ok() { [ -f "$1/meta/episodes.jsonl" ] && [ "$(grep -c . "$1/meta/episodes.jsonl")" -ge $N ]; }
for t in "${TASKS[@]}"; do
  stage=$WORK/stage/$t; dst=$OUT/$t/dr-level-3
  [ -f $stage/meta/episodes.jsonl ] || { echo "no built set for $t (run build_lv3_feasible.py)"; continue; }
  for try in 1 2 3; do
    TMP=$WORK/render/$t; rm -rf $TMP; mkdir -p $TMP; echo "=== render $t try $try $(date +%T)"
    if [[ $t == *MP-v0 ]]; then
      timeout 3600 $PY $HERE/levelgen.py dr simple/$t --env-config-dir $stage --dr-level 3 --num-episodes $N --sim-mode mujoco_isaac --render-hz 50 \
        --headless --no-webrtc --keep-object-materials --save-dir "$TMP" > $WORK/logs/render_$t.log 2>&1
      src=$(ls -d "$TMP/$t/level-3" "$TMP/simple/$t/level-3" "$TMP/simple/$t/dr-level-3" "$TMP/$t/dr-level-3" 2>/dev/null | head -1)
    else
      timeout 3600 $PY $HERE/levelgen.py dr_wbc simple/$t --data-dir $stage --dr-level 3 --num-episodes $N --headless --keep-object-materials \
        --save-dir "$TMP" > $WORK/logs/render_$t.log 2>&1
      src=$(ls -d "$TMP/simple/$t/dr-level-3" "$TMP/$t/dr-level-3" 2>/dev/null | head -1)
    fi
    if [ -n "$src" ] && ok "$src"; then
      mkdir -p $OUT/$t; rm -rf $dst; mv "$src" $dst; cp $stage/build_report.json $dst/meta/feasibility_build.json
      [ -f $WORK/feas/final_$t.jsonl ] && cp $WORK/feas/final_$t.jsonl $dst/meta/feasibility_planner.jsonl
      echo "render $t ok $(date +%T)"; break
    fi
    echo "render $t try $try failed (log $WORK/logs/render_$t.log)"; sleep 10
  done
done
