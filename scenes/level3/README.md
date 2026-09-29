# Level-3 scenes the G1 can complete

Tools that made the 30 level-3 scenes of each of the five benchmark tasks in `data/evals_scenes_benchmark/<task>/dr-level-3`.
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
         G1WholebodyLocomotionPickBetweenTablesTeleop-v0; do
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
| Handover | standing | 24.9 to 29.8 | −3.9 to 2.0 | −3 to 0 | planner twin: right-hand grasp 30/30, left-hand takeover 19/30 |
| LocoPickBetweenTables | standing, then walk | 30.5 to 33.5 | −3.9 to 4.0 | −3 to 0 | planner twin, 30/30 |
| XMovePick | walk, then standing | 48.0 to 74.7 | −8.0 to −4.1 | −4 to 0 | planner twin, 30/30 |

The three teleop tasks have no planner: their targets stay inside the reach box of the 100 successful demonstrations, and
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

## Teleop tasks: the planner-twin check (added 2026-09-28)

The three teleop tasks have no planner of their own, but SIMPLE has motion-planning twins with the same G1 and planner
(XMoveAndPickMP, LocomotionPickBetweenTablesMP, TabletopHandoverMP). `feas_loco.py` copies each teleop scene
into its twin (table top at z = 0, the robot's pelvis at the teleop height above it; the basket of Handover is left out) and
places the robot where the teleop demonstrations stood when they lifted the object (from their recorded `observation.base_pose`
and `observation.object_poses`: XMovePick 30.7 cm ahead / 7.8 cm right, 109 demos).
It tries the straight-ahead stance first, then the demos' sidestep. A scene passes when cuRobo grasps the object and it ends >= 5 cm above its start; for Handover the
right-hand grasp and the left-hand takeover are recorded separately.

| task | level-0 control (first 6 scenes) | level 3 | changed to get there |
|---|---|---|---|
| XMovePick | 5/6 (scene 1: the box stands 0.5 cm from the edge and topples before the grasp) | 30/30 | none |
| LocoPickBetweenTables | 6/6 | 30/30 | 6 scenes recentred (box closest to the pelvis on a lower table) |
| Handover | grasp 6/6, takeover 3/6 | grasp 30/30, takeover 19/30 | 7 scenes recentred (box left of centre on a lower table), 1 distractor moved out of the arm's path |

The twin's left-hand takeover fails in half of SIMPLE's own level-0 scenes, so it is a limit of the twin, not of the scenes.

## Stays put (20 s, robot standing)

`bench_settle.py` (motion-planner tasks) and `kit_settle.py` (teleop tasks) load every scene exactly and let the robot stand for
20 s: the object must move < 1-1.5 cm, tilt < 5 deg and not drop. All 150 retained scenes pass. TabletopGrasp needed 4 scenes recentred: the
right hand pushed the can 2.4 cm or left it leaning on a finger at start-up (SIMPLE's own level 0 nudges it up to 1.3 cm).

## See-through table materials

`fix_tables.py` replaces glass, gem, liquid, light-transmitting and thin-walled (translucent fabric) table materials by an opaque
material of the same set: Isaac's real-time renderer drew objects resting on them hollow. Visual only (MuJoCo ignores it); the
replaced ones are listed in each set's `meta/table_material_fix.json`.

`check_teleop.sh` re-runs both checks.
