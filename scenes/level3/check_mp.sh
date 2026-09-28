#!/usr/bin/env bash
# Step 3 (TabletopGrasp / BendPick only): check every built scene with the motion planner that generated the task's training
# data, then rebuild the failing scenes with their range shrunk toward the centre (0.5, then 1.0) and check those again.
# Results: $WORK/feas/final_<task>.jsonl (one line per scene; feasible = grasp + lift executed).
#   bash scenes/level3/check_mp.sh G1WholebodyTabletopGraspMP-v0 [G1WholebodyBendPickMP-v0]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); cd "$ROOT"
PY=${PY:-.venv/bin/python}; WORK=${WORK:-$ROOT/data/level3_work}
export MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $WORK/feas
for t in "$@"; do
  RH=""; [[ $t == G1WholebodyBendPickMP-v0 ]] && RH="--render-hz 50"      # BendPick asserts render/physics parity (50 Hz)
  F=$WORK/feas/final_$t.jsonl; STAGE=$WORK/stage/$t
  $PY $HERE/feas_mp.py $STAGE --env-id simple/$t $RH --tries 4 --out $F > $WORK/feas/check_$t.log 2>&1
  for shrink in 0.5 1.0; do
    n_bad=$(python3 -c "import json; print(sum(not json.loads(l)['feasible'] for l in open('$F') if l.strip()))")
    [ "$n_bad" = 0 ] && break
    python3 - "$F" "$shrink" "$WORK/redo_$t.json" <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = sorted({r["scene"] for r in rs if not r["feasible"]})
json.dump({str(i): float(sys.argv[2]) for i in bad}, open(sys.argv[3], "w"))
open(sys.argv[1], "w").write("".join(json.dumps(r) + "\n" for r in rs if r["feasible"]))
print(f"redo {len(bad)} scenes with shrink {sys.argv[2]}: {bad}")
PY
    $PY $HERE/build_lv3_feasible.py $t --src $WORK/candidates/$t/dr-level-3 --out $STAGE --redo $WORK/redo_$t.json | tail -1
    $PY $HERE/feas_mp.py $STAGE --env-id simple/$t $RH --tries 4 --out $F >> $WORK/feas/check_$t.log 2>&1
  done
  python3 -c "import json; rs=[json.loads(l) for l in open('$F') if l.strip()]; print('$t:', sum(r['feasible'] for r in rs), '/', len(rs), 'solved; failing', sorted(r['scene'] for r in rs if not r['feasible']))"
done
