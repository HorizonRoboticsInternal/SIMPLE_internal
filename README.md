# Scenes for the G1: level-3 scenes, three real kitchens, and the HoloMotion controller

What was built, where it lives, and the one command that runs each. This is the README of the
`feat/new-simple-eval-tasks` branch; the upstream SIMPLE README is kept as [README_SIMPLE.md](README_SIMPLE.md).
The videos open in GitHub's player from `scenes/readme_vid/`; the served copy of this page, with every scene grid, is at
http://10.40.11.11:8899/simple_scenes_readme/index.html (lab network only).

## What is on this branch

| where | what |
|---|---|
| `scenes/bottle_bin/`, `scenes/bowl_sink/`, `scenes/coffee_cart/` | the three real-world scenes as SIMPLE tasks (`<kit>_task.py` registers `simple/G1Wholebody{BottleBin,BowlSink,CoffeeCart}Teleop-v0` on import), their MuJoCo scene (`build_scene.py` → `scene.xml`, `layout.json`, `assets/`), the replay tool (`replay_in_scene.py`, scored by the same gates as the task), renders and a README each |
| `scenes/make_levels.py` | writes the level 0–3 evaluation sets of a kit (LeRobot format, one `environment_config` per scene) into `data/evals_scenes/<env>/dr-level-<N>/`, plus `meta/scene_env.json` with the kit knobs the set was generated with |
| `scenes/eval_scene.py` | runs SIMPLE's `eval_decoupled_wbc` on a kit's task: registers the task, re-applies `scene_env.json`, then the standard evaluator |
| `scenes/replay_isaac.py`, `scenes/compose_third.py`, `scenes/screen_episodes.sh` | replay a recorded episode in a level scene with Isaac rendering (third person + head camera + real head camera in one video); screen episodes in MuJoCo only |
| `data/evals_scenes_benchmark/<task>/dr-level-3/` | the six original tasks' level-3 sets, 30 reachability-checked scenes each (section 1) |
| `scenes/level3/` | the tools that built and checked them: candidate generation, per-scene reach fitting, motion-planner check, exact render ([README](scenes/level3/README.md)) |
| `data/evals_scenes/…/dr-level-{0,1,2}/` | the three kitchens' level 0–2 sets (ten scenes each, 2026-09-24), committed so the evaluator runs without regeneration |
| `data/evals_scenes/…/dr-level-3/` | the three kitchens' level-3 sets, 30 scenes each (2026-09-28): bottle_bin and bowl_sink rebuilt from the replay fits and checked graspable, coffee_cart the ±10 / ±5 / ±4 cm draw; used by the HoloMotion v1.4 VLA evaluation (section 4) |
| `scenes/readme_img/` | the pictures of this README |
| `third_party/holomotion/`, `src/simple/teleop/holomotion/`, `src/simple/agents/holomotion_pico_agent.py`, `src/simple/cli/{teleop_holomotion,holomotion_replay}.py` | the HoloMotion v1.4.1 controller (section 3) |
| `docs/TELEOP_CONTROL_LOOP_SPEC.md`, `docs/holomotion_teleop.md`, `REAL_ROBOT_RUNBOOK.md` | the teleop stack's control-loop spec, the HoloMotion guide, the real-robot runbook |
| `src/simple/cli/{teleop,replay,eval}_holomotion_v14.py`, `src/simple/agents/holomotion_v14_{agent,vla_agent}.py`, `src/simple/teleop/holomotion_v14/`, `third_party/holomotion_v14/`, `third_party/hbvcam_stereo/`, `scripts/teleop_holomotion_v14.sh`, `scripts/holomotion_v14_episode_server.py` | HoloMotion v1.4 (teleop-collection branch) on the G1 with the 3.2 kg backpack and the HBVCAM stereo fisheye head camera: PICO sim teleop with bit-exact replay, and the VLA evaluation (section 4) |
| `docs/holomotion_v14_teleop.md`, `docs/holomotion_v14_eval.md`, `PICO_SIM_TELEOP_RUNBOOK.md` | their guides and the PICO runbook |

Setup notes: the scene robot is the `g1comp` G1 (D455 head camera, Dex3 hands) in `scenes/bottle_bin/robot/g1_comp_45dof.xml`; its
`robot/meshes` is a symlink to `data/robots/g1/meshes`, which comes from SIMPLE's `robots_g1.zip` like every other robot asset
(the other two kits' `robot/` links point at bottle_bin's). The replay tools read the real recordings from `data/real_recordings/`,
which are not in the repository. `make_report.py` in each kit rebuilds that kit's report page and still points at the drawings
and the ffmpeg of the workstation it was written on.

Rebuild a kit's level sets (Isaac; `frames/ep<i>.png` previews are written next to them but not committed) and replay a
recorded episode inside a level scene:

```bash
python scenes/make_levels.py bowl_sink --levels 0 1 2 3 --episodes 20 --out data/evals_scenes
python scenes/replay_isaac.py bowl_sink --episode 41 --nav-gain 1.5 --out scenes/bowl_sink/replay/isaac_level0_ep41.mp4
```

Evaluate a kit's task the standard way (the Psi-0 server on `--port`, Isaac rendering, one level set):

```bash
python scenes/eval_scene.py bowl_sink simple/G1WholebodyBowlSinkTeleop-v0 psi0_decoupled_wbc train \
    --data-format lerobot --data-dir data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-0 \
    --port 21000 --headless --num-episodes 10
```

## 1 · Level-3 scenes for SIMPLE

The six original tasks in new scenes: 30 level-3 scenes each, rendered in Isaac, every one inside the range where the G1
can still do the task. Every level-3 scene changes all of these at once:

1. **Scene**: new distractors and a new table material
2. **Lighting**: new lighting
3. **Objects**: new object poses
4. **Layout**: a different robot start and table-top height, chosen per scene inside the range where the G1 can still do
   the task (within ±10 cm forward/back and ±5 cm sideways at most)

**How the ranges were set.** The robot start is chosen after the scene's object pose is drawn, so that the target, seen from
the start, stays where the G1 can reach it. For the two motion-planner tasks the reach and height limits come from the
planner that generated their training data, and every final scene was solved by it (grasp and lift executed, up to four
random grasp draws). The four teleop tasks were checked with their motion-planning twins (same G1 and planner): the robot
stands where the teleop demonstrations stood when they lifted the object (from their recorded base and object poses) and
must grasp and lift it at least 5 cm; Handover's twin also hands the object to the left hand. Scenes that failed a check
were moved toward the middle of their range and re-checked. Every scene was also stepped for 20 s with the robot standing:
the object must stay put. See-through table materials (glass, gems, water, thin fabrics) were replaced by opaque ones:
Isaac drew objects resting on them hollow. Tools, steps and the calibration are in
[scenes/level3/README.md](scenes/level3/README.md).

Level 3 ranges per task (cm; forward = toward the table; + = forward / left / up):

| task | grasp | robot start, forward | robot start, sideways | table height | target ahead of the start | target left of the start | checked by |
|---|---|---|---|---|---|---|---|
| TabletopGrasp | standing grasp | −3.0 to 9.2 | −2.7 to 5.0 | −4.0 to 0 | 27.4 to 39.8 | −7.0 to 2.7 | motion planner, 30/30 solved; can stays put 30/30 |
| BendPick | bend grasp | −7.4 to 8.2 | −5.0 to 2.9 | −2.0 to 2.0 | 22.2 to 42.6 | −8.0 to 0 | motion planner, 30/30 solved; box stays put 30/30 |
| XMovePick | walk, then standing grasp | −10.0 to 5.1 | −2.7 to 3.5 | −4.0 to 0 | 48.0 to 74.7 | −8.0 to −4.1 | planner twin, walk then grasp: 30/30 |
| Handover | standing grasp | −3.4 to 4.4 | −4.8 to 4.5 | −3.0 to 0 | 24.9 to 29.8 | −3.9 to 2.0 | planner twin: right-hand grasp 30/30, left-hand takeover 19/30 (3/6 on level 0) |
| LocoPickBetweenTables | standing grasp, then walk | −2.5 to 1.9 | −5.0 to 5.0 | −3.0 to 0 | 30.5 to 33.5 | −3.9 to 4.0 | planner twin, grasp from the start: 30/30 |
| XMoveBendPick | walk, then bend grasp | −9.7 to 7.9 | −3.8 to 3.4 | −2.0 to 2.0 | 61.3 to 74.4 | −8.0 to −4.0 | planner twin, walk then bend grasp: 30/30 |

Motion-planner calibration: training scenes still solvable after one change.

| change | TabletopGrasp | BendPick |
|---|---|---|
| table 4 / 2 cm lower | 3/4 · 3/4 | 0/4 · 3/4 |
| table 2 / 4 cm higher | 0/4 · 0/4 | 3/4 · 0/4 |
| robot 6 / 3 cm back | 0/4 · 1/4 | 4/4 · 3/4 |
| robot 3 / 6 cm forward | 4/4 · 4/4 | 4/4 · 4/4 |
| robot 4 cm right / left | 2/4 · 3/4 | 4/4 · 0/4 |

TabletopGrasp cannot start further away or with a higher table: the can is then out of reach, or the arms' start pose hits
the raised table. Where a combination of changes still failed a check, that scene was moved toward the middle of its range
and re-checked; in two scenes a distractor standing in the grasp path was moved back (TabletopGrasp 30, Handover 24). BendPick
boxes sit 0.5–3.4 cm behind the table edge, as in SIMPLE's own level 0 (0.6–3.3 cm); none moved in the 20 s check. Each scene
is built on one of the task's 20 base layouts; scenes 21–30 reuse layouts 1–10 with everything else drawn anew. The sets are
in `data/evals_scenes_benchmark/<task>/dr-level-3`: per-scene offsets in `meta/feasibility_build.json`, planner results in
`meta/feasibility_planner.jsonl` (planner tasks) or `meta/feasibility_loco.jsonl` (planner twin of the teleop tasks), the
stays-put check in `meta/feasibility_settle.jsonl`, replaced table materials in `meta/table_material_fix.json`. They have not
been evaluated with Ψ0 yet.

Evaluate a task on its set the standard way (Ψ0 server on `--port`, Isaac rendering):

```bash
python -m simple.cli.eval_decoupled_wbc simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train --data-format lerobot \
    --data-dir data/evals_scenes_benchmark/G1WholebodyXMovePickTeleop-v0/dr-level-3 --port 21000 --sim-mode mujoco_isaac --headless --num-episodes 30
python -m simple.cli.eval simple/G1WholebodyTabletopGraspMP-v0 psi0 train --data-format lerobot \
    --data-dir data/evals_scenes_benchmark/G1WholebodyTabletopGraspMP-v0/dr-level-3 --port 21000 --headless --num-episodes 30
```

<details><summary><b>TabletopGrasp</b> · 30 level-3 scenes · planner-checked</summary>

![TabletopGrasp: 30 level-3 Isaac scenes](scenes/readme_img/grid_TabletopGraspMP_lv3_30.jpg)

Per scene (cm):

| scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | +7.4 | +4.3 | −0.5 | +4.5 | +3.9 | +5.7 | +9.2 | +4.7 | +8.6 | +5.2 |
| start, sideways | −2.6 | +2.0 | −1.9 | −0.6 | +1.3 | +4.0 | +0.1 | +2.0 | −0.5 | −2.0 |
| table height | −0.1 | −2.0 | −4.0 | −2.0 | −2.0 | −0.8 | −1.2 | −2.3 | −0.5 | −1.5 |
| target ahead | 34.0 | 33.6 | 36.8 | 33.6 | 33.6 | 28.3 | 33.2 | 33.2 | 30.8 | 30.7 |
| target left | 0 | −2.1 | 1.3 | −2.1 | −2.1 | −2.2 | −2.9 | −0.1 | −1.7 | −0.1 |
| planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | +1.2 | +5.7 | −0.3 | +3.9 | +4.7 | +1.8 | −3.0 | +2.6 | +3.1 | +7.1 |
| start, sideways | +3.0 | +0.3 | +3.1 | +2.4 | +2.3 | +3.8 | +4.8 | +0.8 | −1.5 | +1.8 |
| table height | −3.7 | −2.0 | −1.9 | −3.3 | −2.0 | −2.0 | −2.1 | −2.9 | −2.5 | −2.3 |
| target ahead | 35.5 | 33.6 | 36.5 | 31.7 | 33.6 | 33.6 | 37.7 | 33.8 | 36.7 | 32.2 |
| target left | −2.4 | −2.1 | −0.2 | −0.9 | −2.1 | −2.1 | −2.9 | 1.4 | −0.2 | −0.2 |
| planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | +1.1 | +1.0 | +3.9 | +2.2 | +1.0 | +2.8 | +8.7 | −1.1 | +5.9 | +3.3 |
| start, sideways | 0 | +1.6 | +0.1 | +0.6 | −2.7 | −0.5 | +1.9 | +1.8 | −0.9 | +3.2 |
| table height | −3.6 | −1.5 | −2.0 | −3.5 | 0 | −2.0 | −1.9 | −1.4 | −2.5 | −2.0 |
| target ahead | 38.4 | 36.4 | 33.6 | 36.3 | 33.4 | 33.6 | 35.0 | 37.2 | 33.9 | 33.6 |
| target left | 0.8 | −1.0 | −2.1 | 0.7 | 2.7 | −2.1 | 0.8 | 0.5 | −0.4 | −2.1 |
| planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Notes: recentred toward the middle of the range after a failed check: scenes 2, 4, 5, 10, 12, 13, 15, 16, 19, 20, 23, 26, 30 · scene 1: robot start moved -3.4 cm sideways so the can is on its centre line: the right hand pushed it 2.4 cm at start-up · scene 30: distractor 46 moved 10 cm further back: it stood 14.7 cm behind the can in the grasp path

</details>

<details><summary><b>BendPick</b> · 30 level-3 scenes · planner-checked</summary>

![BendPick: 30 level-3 Isaac scenes](scenes/readme_img/grid_BendPickMP_lv3_30.jpg)

Per scene (cm):

| scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | +0.3 | +4.9 | −5.0 | +2.6 | −6.7 | +4.7 | +8.2 | +0.1 | +7.1 | +4.6 |
| start, sideways | +0.2 | −4.3 | −3.9 | +1.7 | +1.6 | −2.5 | −1.0 | −2.1 | −0.8 | −4.8 |
| table height | +0.9 | +0.2 | −2.0 | −0.2 | +0.9 | +0.6 | +0.8 | −0.3 | +1.5 | +0.5 |
| target ahead | 30.5 | 29.1 | 37.8 | 30.5 | 41.4 | 28.0 | 27.1 | 30.4 | 24.9 | 27.7 |
| target left | −4.5 | −2.1 | −2.7 | −7.2 | −7.5 | −5.5 | −4.7 | −4.9 | −3.6 | −2.6 |
| planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −1.7 | −6.8 | −7.4 | +0.4 | +3.8 | −1.2 | −7.0 | −0.5 | −6.0 | +4.8 |
| start, sideways | +0.8 | +1.7 | −3.6 | −1.2 | −0.7 | +0.4 | +2.9 | −3.3 | −3.6 | −3.1 |
| table height | −0.9 | −0.8 | +0.2 | −0.7 | −0.6 | +0.8 | −0.1 | −0.9 | −0.5 | −0.6 |
| target ahead | 34.0 | 40.7 | 42.0 | 30.8 | 29.8 | 33.3 | 36.5 | 32.4 | 37.5 | 28.9 |
| target left | −4.9 | −6.1 | −2.7 | −4.3 | −4.8 | −6.0 | −7.7 | −2.8 | −2.4 | −1.1 |
| planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −5.6 | −4.2 | −1.5 | −0.3 | +1.8 | +3.5 | +5.9 | −5.9 | +1.7 | −2.4 |
| start, sideways | −2.9 | −1.1 | +2.4 | −1.3 | −5.0 | −0.4 | −3.0 | −2.1 | −2.6 | −0.7 |
| table height | −0.8 | +0.5 | +1.7 | −0.7 | +2.0 | −0.9 | +0.1 | +0.6 | −0.5 | +1.3 |
| target ahead | 36.1 | 37.2 | 32.6 | 34.3 | 33.8 | 27.3 | 26.9 | 38.1 | 30.8 | 34.6 |
| target left | −3.1 | −4.3 | −6.9 | −3.2 | −0.8 | −5.2 | −4.8 | −3.9 | −3.4 | −5.7 |
| planner | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Notes: recentred toward the middle of the range after a failed check: scenes 1, 2, 6, 10, 11, 14, 15, 16, 19, 21, 24, 26

</details>

<details><summary><b>XMovePick</b> · 30 level-3 scenes · planner-checked</summary>

![XMovePick: 30 level-3 Isaac scenes](scenes/readme_img/grid_XMovePickTeleop_lv3_30.jpg)

Per scene (cm):

| scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | 0 | +2.3 | −6.7 | −1.9 | −8.9 | +4.9 | +5.1 | −3.2 | +2.0 | +2.5 |
| start, sideways | −0.5 | −1.5 | −1.8 | +3.5 | +3.2 | −0.2 | −1.3 | +1.0 | −1.0 | −2.7 |
| table height | −0.1 | −1.7 | −4.0 | −2.2 | −1.1 | −0.8 | −1.2 | −2.3 | −0.5 | −1.0 |
| target ahead | 52.5 | 50.6 | 58.5 | 52.9 | 62.6 | 49.1 | 49.7 | 53.5 | 49.9 | 48.4 |
| target left | −6.5 | −4.2 | −4.8 | −7.6 | −7.7 | −7.3 | −6.2 | −6.0 | −5.9 | −4.4 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −4.3 | −8.6 | −9.5 | +0.1 | +2.9 | −3.0 | −7.4 | −4.2 | −10.0 | +2.2 |
| start, sideways | +2.7 | +2.4 | −2.6 | +2.1 | −0.5 | +0.4 | +0.6 | −1.8 | 0 | −2.1 |
| table height | −3.7 | −2.8 | −1.8 | −3.3 | −3.2 | −0.4 | −2.1 | −2.9 | −3.0 | −2.6 |
| target ahead | 58.9 | 59.7 | 60.9 | 53.3 | 52.1 | 57.9 | 60.4 | 54.2 | 63.5 | 52.6 |
| target left | −6.9 | −7.0 | −4.9 | −6.4 | −6.8 | −8.0 | −7.9 | −5.0 | −4.5 | −4.6 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −7.7 | −5.6 | −2.7 | −5.6 | −2.1 | +2.3 | +2.0 | −7.1 | −2.6 | −4.1 |
| start, sideways | −1.6 | +0.2 | +0.8 | −0.2 | −0.9 | 0 | −1.9 | +0.3 | −0.7 | +1.3 |
| table height | −3.6 | −1.5 | −0.3 | −3.5 | 0 | −3.9 | −1.9 | −1.4 | −2.5 | −0.7 |
| target ahead | 62.5 | 59.6 | 57.0 | 57.8 | 55.4 | 48.0 | 51.1 | 59.0 | 53.2 | 57.6 |
| target left | −5.2 | −6.1 | −7.5 | −5.3 | −4.1 | −7.2 | −5.6 | −5.7 | −5.4 | −6.7 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

</details>

<details><summary><b>Handover</b> · 30 level-3 scenes · planner-checked</summary>

![Handover: 30 level-3 Isaac scenes](scenes/readme_img/grid_HandoverTeleop_lv3_30.jpg)

Per scene (cm):

| scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | +2.1 | +1.6 | −1.0 | −1.6 | −1.5 | +4.4 | −0.7 | −0.2 | +0.4 | −1.7 |
| start, sideways | +1.5 | −2.3 | −0.1 | +4.5 | +2.1 | +0.2 | −1.9 | +1.2 | +0.2 | +0.5 |
| table height | −0.1 | −1.5 | −1.5 | −1.7 | −0.8 | −0.6 | −0.9 | −1.8 | −0.4 | −1.5 |
| target ahead | 26.4 | 27.4 | 27.4 | 26.8 | 29.5 | 25.2 | 25.4 | 27.1 | 25.6 | 27.4 |
| target left | −1.7 | −0.9 | −0.9 | −2.8 | −3.5 | −2.9 | −1.4 | −0.9 | −0.6 | −0.9 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| handed over | ✓ | · | · | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −2.3 | −0.2 | −3.3 | +1.1 | +2.3 | −0.9 | −0.3 | −0.6 | −1.0 | +0.2 |
| start, sideways | −0.8 | +0.7 | −4.0 | +1.3 | −1.3 | +0.9 | +0.4 | −0.7 | +2.2 | +1.7 |
| table height | −2.8 | −2.1 | −1.3 | −2.5 | −2.4 | −0.3 | −1.6 | −1.5 | −1.5 | −1.5 |
| target ahead | 28.1 | 29.3 | 29.6 | 26.6 | 26.1 | 27.8 | 28.9 | 27.4 | 27.4 | 27.4 |
| target left | −2.3 | −2.5 | 0 | −1.5 | −2.2 | −3.9 | −3.7 | −0.9 | −0.9 | −0.9 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| handed over | · | ✓ | ✓ | · | ✓ | ✓ | ✓ | ✓ | · | · |

| scene | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −0.2 | −2.6 | −1.1 | −1.1 | +0.1 | +3.8 | −0.2 | −2.7 | −0.7 | −1.3 |
| start, sideways | −1.5 | −0.5 | +4.3 | −2.0 | −3.5 | +3.8 | −2.8 | −2.3 | −2.6 | +1.6 |
| table height | −2.7 | −1.1 | −0.2 | −1.5 | 0 | −2.9 | −1.5 | −1.0 | −1.9 | −0.5 |
| target ahead | 29.1 | 28.4 | 27.6 | 27.4 | 27.3 | 24.9 | 25.9 | 28.8 | 26.9 | 27.9 |
| target left | 0.4 | −1.1 | −2.5 | −0.9 | 2.0 | −2.5 | −0.3 | −0.4 | 0 | −1.9 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| handed over | · | · | ✓ | ✓ | ✓ | ✓ | · | · | · | ✓ |

Notes: recentred toward the middle of the range after a failed check: scenes 2, 3, 10, 18, 19, 20, 24 · scene 24: distractor 1 (sugar box) moved from 14 cm behind-right of the box to 20 cm: it stood in the right arm's grasp path

</details>

<details><summary><b>LocoPickBetweenTables</b> · 30 level-3 scenes · planner-checked</summary>

![LocoPickBetweenTables: 30 level-3 Isaac scenes](scenes/readme_img/grid_LocomotionPickBetweenTablesTeleop_lv3_30.jpg)

Per scene (cm):

| scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −1.2 | +0.4 | −0.7 | +0.6 | −0.6 | +0.7 | −0.1 | +0.9 | −0.9 | −1.4 |
| start, sideways | −1.0 | −1.4 | −3.7 | +1.1 | +1.2 | +1.8 | −2.3 | +1.8 | +1.5 | +1.8 |
| table height | −0.1 | −1.5 | −3.0 | −1.7 | −0.8 | −1.5 | −1.5 | −1.8 | −0.4 | −1.5 |
| target ahead | 31.4 | 32.0 | 32.8 | 31.6 | 33.3 | 32.0 | 32.0 | 31.9 | 30.9 | 32.0 |
| target left | −1.5 | 0.1 | 2.5 | −3.2 | −3.4 | 0.1 | 0.1 | 1.0 | 1.1 | 0.1 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −0.5 | −1.6 | −0.3 | +0.4 | +1.0 | +1.0 | −0.7 | −0.4 | −2.2 | +0.2 |
| start, sideways | −1.3 | +3.4 | −2.2 | +2.5 | +3.2 | +5.0 | +0.6 | −3.1 | −4.5 | −2.0 |
| table height | −2.8 | −2.1 | −1.3 | −2.5 | −1.5 | −0.3 | −1.6 | −2.2 | −2.3 | −2.0 |
| target ahead | 32.5 | 33.2 | 33.4 | 31.5 | 32.0 | 32.2 | 33.0 | 32.0 | 33.5 | 31.3 |
| target left | −2.5 | −1.1 | 2.4 | 0.4 | 0.1 | −1.4 | −3.7 | 1.9 | 0.5 | 2.9 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −2.5 | −1.2 | −0.9 | −0.6 | +0.6 | −0.2 | +0.8 | +0.1 | −0.9 | −1.1 |
| start, sideways | −0.3 | −1.4 | +1.6 | −3.1 | −5.0 | −1.6 | +0.3 | −0.8 | +1.6 | +0.6 |
| table height | −2.7 | −1.1 | −0.2 | −2.6 | 0 | −1.5 | −1.5 | −1.0 | −1.9 | −0.5 |
| target ahead | 33.1 | 32.7 | 32.1 | 32.6 | 31.9 | 32.0 | 31.1 | 32.9 | 31.7 | 32.4 |
| target left | 2.0 | −0.5 | −2.8 | 0.3 | 2.2 | 0.1 | 1.1 | 0.7 | 2.2 | −1.2 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Notes: recentred toward the middle of the range after a failed check: scenes 2, 6, 7, 10, 15, 26

</details>

<details><summary><b>XMoveBendPick</b> · 30 level-3 scenes · planner-checked</summary>

![XMoveBendPick: 30 level-3 Isaac scenes](scenes/readme_img/grid_XMoveBendPickTeleop_lv3_30.jpg)

Per scene (cm):

| scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −0.4 | +2.8 | −7.8 | −4.2 | −0.4 | +0.2 | +7.9 | −3.1 | +4.6 | +4.3 |
| start, sideways | +0.1 | −3.8 | −0.6 | −1.6 | +3.4 | −0.3 | −1.4 | −0.5 | +1.5 | −1.9 |
| table height | +1.9 | +0.3 | −2.0 | 0 | +0.9 | +1.2 | +0.8 | −0.3 | +1.5 | +1.0 |
| target ahead | 65.4 | 63.5 | 68.3 | 67.8 | 73.6 | 62.1 | 62.7 | 66.9 | 63.1 | 61.8 |
| target left | −6.5 | −4.1 | −4.7 | −6.0 | −7.7 | −7.3 | −6.2 | −5.9 | −5.8 | −4.3 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −4.8 | −5.5 | −9.7 | +1.3 | −4.6 | −3.7 | −1.4 | −3.6 | −3.5 | −0.1 |
| start, sideways | +0.7 | −0.1 | −0.4 | +2.0 | −0.8 | +0.3 | +2.4 | +0.4 | −0.3 | −0.7 |
| table height | −1.7 | 0 | +0.2 | −1.3 | 0 | +1.6 | −0.1 | −0.9 | −1.0 | −0.6 |
| target ahead | 69.9 | 67.8 | 70.7 | 65.8 | 67.8 | 69.0 | 72.1 | 68.1 | 74.4 | 64.9 |
| target left | −6.9 | −6.0 | −4.8 | −6.3 | −6.0 | −8.0 | −7.9 | −5.0 | −4.4 | −4.5 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| scene | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| start, forward | −8.3 | −5.6 | −3.3 | −3.8 | +1.6 | −4.5 | +7.8 | −7.8 | +0.8 | −4.6 |
| start, sideways | −1.1 | −1.2 | +0.2 | −2.3 | −3.5 | +1.6 | −2.0 | +1.6 | +0.2 | +1.2 |
| table height | −1.6 | +0.5 | +1.7 | −1.5 | +2.0 | 0 | +0.1 | +0.6 | −0.5 | +1.3 |
| target ahead | 71.6 | 70.8 | 68.5 | 70.3 | 67.6 | 67.8 | 65.8 | 69.7 | 66.7 | 69.4 |
| target left | −5.1 | −6.1 | −7.4 | −5.2 | −4.0 | −6.0 | −5.5 | −5.7 | −5.4 | −6.6 |
| grasp (twin) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Notes: recentred toward the middle of the range after a failed check: scenes 4, 12, 15, 26 · scene 4: box moved 2.0 cm away from the table edge (with the robot start): it tipped off when grasped · scene 12: box moved 0.3 cm away from the table edge (with the robot start): it tipped off when grasped

</details>

## 2 · Real-world scenes rebuilt in MuJoCo

Each is a SIMPLE task plus a replay tool: the real teleop recordings are played back inside the scene and scored by **gates**,
ordered checks that latch when met, each only after the one before. Reward grows per gate; the last gate is success. One success
each below. Each scene also ships as SIMPLE evaluation sets at **levels 0–3**, rendered in Isaac, ten scenes per level (2026-09-24).
The same gate code scores the replays and the SIMPLE task. The level-3 tables below describe the first, ten-scene level 3
(2026-09-24). Level 3 now has 30 scenes per kit (2026-09-28):
- bottle_bin and bowl_sink were rebuilt from the replay fits and checked graspable;
- coffee_cart is the unchecked ±10 cm / ±5 cm / ±4 cm draw.

Section 4 evaluates on those 30.

On 2026-09-24 each scene was also run once with its deployed HoloBrain model through the standard evaluator
(`simple.cli.eval_decoupled_wbc` with SIMPLE's `psi0_decoupled_wbc` agent, HTTP `/act`, a whole chunk per query, 8 demasking
steps, horizon 15, level-0 set, Isaac rendering). None succeeded, which was expected; the gate readings below show the checker
responding to what actually happened. Those runs used `sim/pipeline_test.sh` of the `holobrain_g1_deploy` package, which starts the
model server.

### Bottle → bin

![bottle_bin scene](scenes/readme_img/scene_bottle_bin.jpg)
_Scene: table, 500 ml bottle, trash bin behind the robot's right._

- **task** Pick up the bottle, turn, drop it in the trash bin. `simple/G1WholebodyBottleBinTeleop-v0`
- **gates**
  1. **grasped**: the right hand touches the bottle and it is 3 cm off the table, upright, for 0.5 s
  2. **at the bin**: after the grasp, walked ≥ 0.5 m, within 0.8 m of the bin, standing still 0.4 s
  3. **placed**: released inside the bin's opening, below the rim, and still there 0.5 s later
- **built from** the user's measurements; bin fitted to the release points of 97 real episodes
- **run** `cd scenes/bottle_bin && MUJOCO_GL=egl python replay_in_scene.py --session 2026-09-17-02-25-56-G1-sim --episodes 12`
- **success, rendered in Isaac** (level 0, scene 0): [video](scenes/readme_vid/isaac_bottle_bin.mp4),
  the recorded episode (session 02-25-56 episode 10) replayed in the level-0 scene: left Isaac third person, middle Isaac head
  camera, right the real head camera, in step
- **deployed model** `chipcan_nativec9` (job `qwen3_2b_posttrain_96d_g1_teleop_chipcantotrash_base_native_tokenizer_norm_20260917_131811`,
  checkpoint_9 of 0..19; stage-2 24-task ckpt4 init, lr 3e-5, pretraining tokenizer quantiles, ChipCanToTrash 02-25-56 session, 97
  episodes): [video](scenes/readme_vid/pipetest_bottle_bin_isaac.mp4). Gating on this episode:
  grasped, at bin, placed all never met. The bottle rose at most 10.3 cm but never counted as grasped (that needs a hand on it,
  the bottle within 60° of upright and 3 cm up for 0.5 s), so it was knocked about rather than held; the robot came within 1.55 m
  of the bin (0.8 m and a 0.4 s stop count as at the bin). Checker evaluated at every one of the 1701 control steps (34 s at
  50 Hz, the 5 s stand-up included); no success.

Levels 0–3 in Isaac, ten scenes each. Level 0: three GraspNet distractors and a new table material. Level 1: + new lighting.
Level 2: + new object pose, the bottle moved ±3 cm across and ±8 cm along the table. Level 3: + new layout, everything above
re-randomised in every scene and the table top height and the robot's start change per scene.

| level 3, scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| robot start, forward (cm) | −6 | +8 | −1 | +10 | −8 | +3 | +6 | −3 | −10 | +1 |
| robot start, sideways (cm) | −2 | +1 | −4 | −3 | +5 | +4 | −5 | +2 | +3 | −1 |
| table top height (cm) | +3 | 0 | −4 | −2 | −3 | +4 | +2 | −1 | +1 | 0 |

Starts are inside ±10 cm forward/back and ±5 cm sideways (+ = forward / left); heights inside ±4 cm. The ten offsets are spread
evenly over each range and shuffled by the seed, so every scene gets a distinct pair.

![bottle_bin levels](scenes/readme_img/grid_bottle_bin_rows.jpg)
_One row per level, ten scenes each._ [Scene page](http://10.40.11.11:8899/holobrain_bottle_bin_scene/index.html)

### Bowl → sink

![bowl_sink scene](scenes/readme_img/scene_bowl_sink.jpg)
_Scene: L-shaped kitchen, green bowl at the counter edge, sink on the right._

- **task** Pick up the bowl, turn right, walk to the sink, place it in the basin. `simple/G1WholebodyBowlSinkTeleop-v0`
- **gates**
  1. **at the bowl**: moved ≥ 5 cm, bowl within 0.6 m and 60° ahead, standing still 0.4 s
  2. **grasped**: hand on the bowl, lifted 5 cm and held 0.3 s
  3. **at the sink**: walked ≥ 0.2 m since the grasp, basin within 0.8 m and 75°, still 0.4 s
  4. **placed**: released, bowl inside the basin footprint, below the rim
- **built from** a hand drawing, corrected on the bench; start pose tuned on 55 real episodes
- **run** `cd scenes/bowl_sink && MUJOCO_GL=egl REPLAY_FAST=1 python replay_in_scene.py --session psi0/BowlToSink_0918 --episodes 7 --nav-gain 1.3`
- **success, rendered in Isaac** (level 0, scene 0): [video](scenes/readme_vid/isaac_bowl_sink.mp4),
  the recorded episode (BowlToSink_0918 episode 41, walking gain 1.5, robot start 0.38 m from the counter edge, moved back on
  2026-09-24 so the walk-in stops short of the cabinet) replayed in the level-0 scene: left Isaac third person, middle Isaac head
  camera, right the real head camera, in step
- **deployed model** `bowltosink_c9` (job `qwen3_2b_posttrain_96d_grouped_diffusion_bowltosink_20260918_103225`, checkpoint_9,
  last of 0..9; stage-2 24-task ckpt4 init, lr 1e-4, BowlToSink_0918, 55 episodes):
  [video](scenes/readme_vid/pipetest_bowl_sink_isaac.mp4). Gating on this episode: at bowl met at
  9.1 s; grasped, at basin, placed never met. The robot was within reach of the bowl from 9.1 s, a hand touched the bowl for 17.4 s
  in total, the bowl rose at most 4.5 cm (below the 5 cm a grasp needs, held 0.3 s); the ladder stalled at grasped. Checker
  evaluated at every one of the 2501 control steps (50 s at 50 Hz, the 5 s stand-up included); no success; closest pelvis-bowl
  0.268 m, closest pelvis-basin 0.865 m.

Levels 0–3 in Isaac, ten scenes each. Level 0: three GraspNet distractors and a new table material. Level 1: + new lighting.
Level 2: + new object pose, the bowl moved ±8 cm along the counter edge, never toward it. Level 3: + new layout, everything above
re-randomised in every scene and the counter top height and the robot's start change per scene.

| level 3, scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| robot start, forward (cm) | −6 | +8 | −1 | +10 | −8 | +3 | +6 | −3 | −10 | +1 |
| robot start, sideways (cm) | −2 | +1 | −4 | −3 | +5 | +4 | −5 | +2 | +3 | −1 |
| counter top height (cm) | +3 | 0 | −4 | −2 | −3 | +4 | +2 | −1 | +1 | 0 |

Same ranges and shuffling as above.

![bowl_sink levels](scenes/readme_img/grid_bowl_sink_rows.jpg)
_One row per level, ten scenes each._ [Scene page](http://10.40.11.11:8899/holobrain_bowl_sink_scene/index.html) ·
[Success videos](http://10.40.11.11:8899/holobrain_bowl_sink_scene/index.html#success) ·
[The four gates](http://10.40.11.11:8899/holobrain_bowl_sink_scene/index.html#gates)

### Coffee cart → desk

![coffee_cart scene](scenes/readme_img/scene_coffee_cart.jpg)
_Scene: service cart with a box and a coffee cup, table to deliver to._

- **task** Push the cart to the desk, take the cup, put it on the desk. `simple/G1WholebodyCoffeeCartTeleop-v0`
- **gates**
  1. **cart pushed**: the cart has travelled ≥ 0.5 m from its start
  2. **cup lifted**: the cup is 5 cm above its rest on the box
  3. **placed**: the cup stands upright on the table, released, for 0.5 s
- **built from** a hand drawing; tuned on all 97 real episodes
- **run** `cd scenes/coffee_cart && MUJOCO_GL=egl python replay_in_scene.py --fast --video --episodes 84`
- **success, rendered in Isaac** (level 0, scene 0): [video](scenes/readme_vid/isaac_coffee_cart.mp4),
  the recorded episode (CartCoffeeCup_0919 episode 64) replayed in the level-0 scene (room hssd:scene31 with its furniture hidden
  since 2026-09-24: open floor between the cart and the desk and beyond it, no wall at the end of the push): left Isaac third
  person, middle Isaac head camera, right the real head camera, in step
- **deployed model** `cart_c19` (job `qwen3_2b_posttrain_96d_grouped_diffusion_cartcoffeecup_20260920_122057`, checkpoint_19, last
  of 0..19; stage-2 24-task ckpt4 init, no camera model, lr 1e-4, CartCoffeeCup_0919, 97 episodes):
  [video](scenes/readme_vid/pipetest_coffee_cart_isaac.mp4). Gating on this episode: cart pushed,
  cup lifted, placed all never met. Hand contact with the cup from 7.5 s, the cup rose at most 4.1 cm (5 cm counts as lifted) and
  ended on its side (tilt 87°) off its rest on the cart, not on the table; the cart moved 0.05 m (0.5 m counts as pushed). Checker
  evaluated at every one of the 3201 control steps (64 s at 50 Hz, the 5 s stand-up included); no success.

Levels 0–3 in Isaac, ten scenes each. Level 0: three GraspNet distractors and a new table material. Level 1: + new lighting.
Level 2: + new object pose, the cup placed anywhere on the near half of the box top. Level 3: + new layout, everything above
re-randomised in every scene and the cart-box top height and the robot's start change per scene.

| level 3, scene | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| robot start, forward (cm) | −6 | +8 | −1 | +10 | −8 | +3 | +6 | −3 | −10 | +1 |
| robot start, sideways (cm) | −2 | +1 | −4 | −3 | +5 | +4 | −5 | +2 | +3 | −1 |
| cart-box top height (cm) | +3 | 0 | −4 | −2 | −3 | +4 | +2 | −1 | +1 | 0 |

Same ranges and shuffling as above.

![coffee_cart levels](scenes/readme_img/grid_coffee_cart_rows.jpg)
_One row per level, ten scenes each._ [Scene page](http://10.40.11.11:8899/holobrain_coffee_cart_scene/index.html)

## 3 · HoloMotion v1.4.1 controller

A second way to drive the G1 in SIMPLE: the HoloMotion motion-tracking policy instead of the decoupled whole-body controller,
with the publisher vendored so no separate checkout is needed.

PICO headset → HoloRetarget publisher → reference stream, 50 Hz → HoloMotion ONNX → 29 joint targets

- **teleop and record**: `third_party/holomotion/run_publisher.sh` then
  `teleop-holomotion simple/G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop-v0`
- **replay without a headset**: `holomotion-replay --hand-demo` or `holomotion-replay path/to/recording.npz`

Lives in `third_party/holomotion/`, `src/simple/teleop/holomotion/`, `src/simple/agents/holomotion_pico_agent.py`.
Full guide: [docs/holomotion_teleop.md](docs/holomotion_teleop.md).

## 4 · HoloMotion v1.4: evaluating a VLA on the three kitchens

The G1 with Dex3 hands, HoloMotion's 3.2 kg backpack and the HBVCAM stereo fisheye head camera
(`third_party/hbvcam_stereo`), driven by HoloMotion v1.4's motion-tracking policy `model_22000` (the backpack model). The
loop is SIMPLE's `eval_decoupled_wbc`: the VLA behind `HttpActionClient`'s `POST /act`, action chunks, task-check success,
`eval_stats.txt` and a video per episode. The one change is the controller: the VLA returns the reference frames
model_22000 tracks, the same frames the PICO headset streams in teleop.

**Once per machine.** The model is not in git (1.6 GB). Copy it from NAS-28:

```bash
rsync -a /mnt/nas28/alan.jiang/holomotion_models/v14_models_backpack_3p2/ data/holomotion/v14_models_backpack_3p2/
(cd data/holomotion/v14_models_backpack_3p2 && sha256sum -c SHA256SUMS)
```

**Run.** Start your VLA server on port 21000, then one command per task. Each runs that task's 30 level-3 scenes.

```bash
python -m simple.cli.eval_holomotion_v14 --scene bottle_bin  --host 127.0.0.1 --port 21000
python -m simple.cli.eval_holomotion_v14 --scene bowl_sink   --host 127.0.0.1 --port 21000
python -m simple.cli.eval_holomotion_v14 --scene coffee_cart --host 127.0.0.1 --port 21000
```

**What the VLA sends and receives:**

| | |
|---|---|
| request | `image["observation.images.ego_view"]` = the HBVCAM rectified left eye, 1280 × 720, tilted 10° down; the task instruction; `state` named like the teleop dataset (`observation.state`, `observation.base_pose`, `observation.base_vel`, `teleop.latest_obs`, `policy.mode`) |
| reply | `action` (T, D), one row per frame at 50 Hz: `[0:65]` the reference frame (`dof_pos[29]`, `dof_vel[29]`, `root_pos[3]`, `root_rot_wxyz[4]`), exactly the dataset's `teleop.latest_obs`; then either 2 grips (D = 67) or 14 hand joint targets (D = 79), or nothing (D = 65, hands open) |

**How an episode runs:**
- the robot stands on the floor in a random start pose, on the walking policy, which moves the arms to its own
  posture within 0.2 s, as in the teleop recordings before the operator's Y;
- the VLA is asked from the first step; the robot keeps walking (standing in place) until its replies are valid
  (right shape, finite, unit root quaternions), and a failed or unusable reply is simply asked again;
- once the policy node has 11 valid frames, motion tracking takes over automatically, as the operator's Y does in
  teleop (step 14, 0.28 s, with a VLA that answers from the start);
- success is the task's last gate (item placed), within 30 s by default.

The controller runs on a clock that advances 20 ms per step, so VLA latency cannot change a rollout. That was
checked: a server delayed 0.3 s per reply gave a bit-identical run.

**Scenes.** `data/evals_scenes/<env>/dr-level-3`, loaded exactly, with two changes for the v1.4 standing pose (its
hands sit at table height):
- bottle_bin: the scenes were built with the feet 3 cm from the table, so each scene is moved 0.32 m away from the
  robot, to the teleop's 0.35 m;
- coffee_cart: the robot starts 10 cm further back; 6 of 30 scenes had the fingers inside the cart handle;
- bowl_sink: used as built.

All 90 scenes were checked on 2026-09-30: each loads, the robot stands 3 s without falling and more than 3 cm from the
furniture.

**Outputs.** `data/evals_holomotion_v14/<policy>/<env>/dr-level-3/` holds:
- `results.json`: per episode, success, every gate, falls, VLA query count and latency;
- `videos/`: the VLA image beside a third-person view;
- `replay/`: a bit-exact replay log per episode, played with
  `python -m simple.cli.replay_holomotion_v14 <that folder> --all --mode action`.

**Checking the pipeline without a VLA.** `scripts/holomotion_v14_episode_server.py <teleop dataset> --episode N --port
21000` serves a recorded teleop episode's own reference frames, for a pipeline check.

**Pipeline check with a real model (2026-09-30).** The three HoloBrain G1 deploy models were run in the loop through
`scripts/holomotion_v14_vla_bridge.py`, which gives the model its training format and fakes HoloMotion reference frames
from its decoupled-WBC rows (legs at the standing pose, waist and arms from the model, root from its walking command).
15 episodes, 987 queries: every stage passed, all replay logs bit-exact; the models sample, so a scene is not
repeatable. Report: http://10.40.11.11:8899/holomotion_v14_pipeline_check/index.html (generator
`docs/_scan/holomotion_v14_pipeline_check/`); details in [docs/holomotion_v14_eval.md](docs/holomotion_v14_eval.md).

**Scene code.** The kit code comes from `~/wrk/robot_orchard_deploy/holobrain_g1_deploy/sim` when present, else from
`scenes/`. The `scenes/bowl_sink` copy here predates the sink-cabinet toe space, so its image differs under the counter;
the physics at the start is the same.

**Full guide:** [docs/holomotion_v14_eval.md](docs/holomotion_v14_eval.md). Collecting the teleop data in the same
setting: [docs/holomotion_v14_teleop.md](docs/holomotion_v14_teleop.md) and
[PICO_SIM_TELEOP_RUNBOOK.md](PICO_SIM_TELEOP_RUNBOOK.md):
`scripts/teleop_holomotion_v14.sh --scene bottle_bin --record`.
