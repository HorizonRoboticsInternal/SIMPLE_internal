#!/usr/bin/env bash
# The planner-twin grasp check of the four teleop tasks' level-3 sets and the stays-put check of all six sets.
#   bash scenes/level3/check_teleop.sh            (results in data/level3_work/feas/)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"
PY=${PY:-.venv/bin/python}; OUT=${WORK:-$ROOT/data/level3_work}/feas; B=data/evals_scenes_benchmark; mkdir -p $OUT
export MUJOCO_GL=egl KIT_DDS_DOMAIN=${KIT_DDS_DOMAIN:-38}
for t in G1WholebodyXMovePickTeleop-v0 G1WholebodyLocomotionPickBetweenTablesTeleop-v0 G1WholebodyHandoverTeleop-v0; do
  $PY $HERE/feas_loco.py $B/$t/dr-level-3 --task $t --tries 2 --out $OUT/loco_$t.jsonl > $OUT/loco_$t.log 2>&1
done
for t in G1WholebodyTabletopGraspMP-v0 G1WholebodyBendPickMP-v0; do            # motion-planner tasks: MP env, stand 20 s
  $PY $HERE/bench_settle.py simple/$t $B/$t/dr-level-3 --out $OUT/settle_$t.jsonl --hold-s 20 > $OUT/settle_$t.log 2>&1
done
for t in G1WholebodyXMovePickTeleop-v0 G1WholebodyLocomotionPickBetweenTablesTeleop-v0 G1WholebodyHandoverTeleop-v0; do   # teleop: Sonic env
  $PY $HERE/kit_settle.py simple/$t $B/$t/dr-level-3 --out $OUT/settle_$t.jsonl --hold-s 20 > $OUT/settle_$t.log 2>&1
done
grep -h -E "^scene .*(infeasible|MOVED)" $OUT/*.log || echo "all scenes passed"
