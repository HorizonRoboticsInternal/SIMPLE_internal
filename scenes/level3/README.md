# Level-3 scenes the G1 can complete

Tools that made the 30 level-3 scenes of each of the six benchmark tasks in `data/evals_scenes_benchmark/<task>/dr-level-3`.
Level 3 changes everything at once in every scene: distractors, table material, lighting, object pose, robot start and
table-top height. The robot start and the table height are chosen **per scene, after its object pose is drawn**, inside
the range where the G1 can still do the task.

| step | file | what it does |
|---|---|---|
| 1 | `gen_candidates.sh` | 30 candidate scenes per task with SIMPLE's DR level 3 (Isaac): new distractors, material, lighting, object pose; robot start and height from `lv3_30_design.json` (unchecked) |
| 2 | `build_lv3_feasible.py` | keeps each candidate's visuals and object pose, replaces its robot start (dx, dy) and table height dz so the target, seen from the start, stays in the task's reach box of `lv3_ranges.json`; objects on the moved table move with it |
| 3 | `check_mp.sh` → `feas_mp.py` | TabletopGrasp and BendPick: every scene must be solved by the motion planner that generated the training data (cuRobo, grasp + lift executed in MuJoCo, up to 4 random grasp draws); failing scenes are rebuilt with the range shrunk toward its centre and checked again |
| 4 | `render_feasible.sh` → `levelgen.py` | renders the built states exactly (`LEVELGEN_EXACT=1` loads every stored state instead of re-drawing parts of it) and installs them with `meta/feasibility_build.json` and `meta/feasibility_planner.jsonl` |
| - | `make_lv3_grids.py`, `reach_envelope.py` | the README grids; target-vs-robot geometry of a set compared with the 100 training demonstrations |

```bash
bash scenes/level3/gen_candidates.sh
for t in G1WholebodyTabletopGraspMP-v0 G1WholebodyBendPickMP-v0 G1WholebodyXMovePickTeleop-v0 G1WholebodyHandoverTeleop-v0 \
         G1WholebodyLocomotionPickBetweenTablesTeleop-v0 G1WholebodyXMoveBendPickTeleop-v0; do
  .venv/bin/python scenes/level3/build_lv3_feasible.py $t --src data/level3_work/candidates/$t/dr-level-3 --out data/level3_work/stage/$t
done
bash scenes/level3/check_mp.sh G1WholebodyTabletopGraspMP-v0 G1WholebodyBendPickMP-v0
bash scenes/level3/render_feasible.sh
```

## Where the ranges come from

**Motion-planner calibration** (`feas_mp.py --shifts=...` on four training scenes per task, one change at a time; scenes
still solvable after the change):

| change | TabletopGrasp | BendPick |
|---|---|---|
| table 4 / 2 cm lower | 3/4 · 3/4 | 0/4 · 3/4 |
| table 2 / 4 cm higher | 0/4 · 0/4 | 3/4 · 0/4 |
| robot 6 / 3 cm back | 0/4 · 1/4 | 4/4 · 3/4 |
| robot 3 / 6 cm forward | 4/4 · 4/4 | 4/4 · 4/4 |
| robot 4 cm right / left | 2/4 · 3/4 | 4/4 · 0/4 |

Positive control: 10/10 training scenes solved per task (at most two grasp draws). TabletopGrasp fails with a higher table
because the arms' start pose then hits it, and fails from further back because the can is out of reach.

**Ranges used** (`lv3_ranges.json`; target ahead / left of the robot start, table height change; cm):

| task | grasp | target ahead | target left | table height | checked by |
|---|---|---|---|---|---|
| TabletopGrasp | standing | 27.4 to 39.8 | −7.0 to 2.7 | −4 to 0 | motion planner, 30/30 |
| BendPick | bend | 22.2 to 42.6 | −8.0 to 0 | −2 to +2 | motion planner, 30/30 |
| Handover | standing | 24.9 to 29.8 | −3.9 to 2.0 | −3 to 0 | inside the demonstrations' reach |
| LocoPickBetweenTables | standing, then walk | 30.5 to 33.5 | −3.9 to 4.0 | −3 to 0 | inside the demonstrations' reach |
| XMovePick | walk, then standing | 48.0 to 74.7 | −8.0 to −4.1 | −4 to 0 | inside the demonstrations' reach |
| XMoveBendPick | walk, then bend | 61.3 to 74.4 | −8.0 to −4.0 | −2 to +2 | inside the demonstrations' reach |

The four teleop tasks have no planner: their targets stay inside the reach box of the 100 successful demonstrations, and
their table heights only move the way the planner found safe for the same kind of grasp. For the walking tasks the start
distance is absorbed by the walk, so it ranges between the closest eval start and the farthest demonstration.

**Manual fix.** TabletopGrasp scene 30 still failed at the centre of its range: a distractor stood 14.7 cm behind the can,
in the grasp path. It was moved 10 cm further back (recorded in that scene's `flags` in `meta/feasibility_build.json`);
the planner then solved it.

**Notes.** Batch planning (`plan_batch_size` > 1) is broken for the dexterous-hand grasps ('trajs' referenced before
assignment), and each target has only one or two cached grasps, so retries mainly re-draw cuRobo's IK seeds. BendPick needs
`--render-hz 50`. `levelgen.py` gives the level generator its own DDS domain (`LEVELGEN_DDS_DOMAIN`) so it never competes
with other decoupled-WBC workers for participant slots. The training demonstrations and the level-0 base sets are read from
`LEVEL3_DEMO_ROOT` / `LEVEL3_BASE_ROOT` (lab NAS by default).
