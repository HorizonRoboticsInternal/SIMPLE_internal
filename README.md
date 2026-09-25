# Scenes for the G1: new level 0–3 scenes, three real kitchens, and the HoloMotion controller

What was built, where it lives, and the one command that runs each. This is the README of the
`feat/new-simple-eval-tasks` branch; the upstream SIMPLE README is kept as [README_SIMPLE.md](README_SIMPLE.md).
The served copy of this page, with every video, is at http://10.40.11.11:8899/simple_scenes_readme/index.html.

## What is on this branch

| where | what |
|---|---|
| `scenes/bottle_bin/`, `scenes/bowl_sink/`, `scenes/coffee_cart/` | the three real-world scenes as SIMPLE tasks (`<kit>_task.py` registers `simple/G1Wholebody{BottleBin,BowlSink,CoffeeCart}Teleop-v0` on import), their MuJoCo scene (`build_scene.py` → `scene.xml`, `layout.json`, `assets/`), the replay tool (`replay_in_scene.py`, scored by the same gates as the task), renders and a README each |
| `scenes/make_levels.py` | writes the level 0–3 evaluation sets of a kit (LeRobot format, one `environment_config` per scene) into `data/evals_scenes/<env>/dr-level-<N>/`, plus `meta/scene_env.json` with the kit knobs the set was generated with |
| `scenes/eval_scene.py` | runs SIMPLE's `eval_decoupled_wbc` on a kit's task: registers the task, re-applies `scene_env.json`, then the standard evaluator |
| `scenes/replay_isaac.py`, `scenes/compose_third.py`, `scenes/screen_episodes.sh` | replay a recorded episode in a level scene with Isaac rendering (third person + head camera + real head camera in one video); screen episodes in MuJoCo only |
| `data/evals_scenes/…/dr-level-{0,1,2,3}/` | the twelve level sets (ten scenes each), committed so the evaluator runs without regeneration |
| `scenes/readme_img/` | the pictures of this README |
| `third_party/holomotion/`, `src/simple/teleop/holomotion/`, `src/simple/agents/holomotion_pico_agent.py`, `src/simple/cli/{teleop_holomotion,holomotion_replay}.py` | the HoloMotion v1.4.1 controller (section 3) |
| `docs/TELEOP_CONTROL_LOOP_SPEC.md`, `docs/holomotion_teleop.md`, `REAL_ROBOT_RUNBOOK.md` | the teleop stack's control-loop spec, the HoloMotion guide, the real-robot runbook |

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

## 1 · New level 0/1/2/3 scenes for SIMPLE

The six original tasks in new scenes: ten at each of the four levels, rendered in Isaac and evaluated once with the official
Ψ0 checkpoints. Each level adds to the one before it; every level is ten scenes per task:

1. **Level 0**, new scene: new distractors and a new table material
2. **Level 1**: + new lighting
3. **Level 2**: + new object poses
4. **Level 3**: + new layout, on top of levels 0–2 re-randomised in every scene, as the report's alternative layouts did:
   table top height (ten offsets inside ±4 cm, one per scene) and robot start (ten offsets inside ±10 cm forward/back and
   ±5 cm sideways, one per scene)

**149 / 220** completed over the four levels (level 3 not evaluated for 2 of the 6 tasks). Levels 0–2: 137/180, level 3: 12/40;
levels 0–2 are the first ten of the 20-scene sets ([20-scene report with every video](http://10.40.11.11:8899/psi0_new20_report/index.html)).

| task | level 0 | level 1 | level 2 | level 3 | the ten scenes of each level |
|---|---|---|---|---|---|
| TabletopGrasp | 9/10 | 7/10 | 6/10 | 2/10 | [lv0](http://10.40.11.11:8899/simple_scenes_readme/img/grid_TabletopGraspMP_lv0.jpg) · [lv1](http://10.40.11.11:8899/simple_scenes_readme/img/grid_TabletopGraspMP_lv1.jpg) · [lv2](http://10.40.11.11:8899/simple_scenes_readme/img/grid_TabletopGraspMP_lv2.jpg) · [lv3](http://10.40.11.11:8899/simple_scenes_readme/img/grid_TabletopGraspMP_lv3.jpg) |
| BendPick | 9/10 | 9/10 | 8/10 | 3/10 | [lv0](http://10.40.11.11:8899/simple_scenes_readme/img/grid_BendPickMP_lv0.jpg) · [lv1](http://10.40.11.11:8899/simple_scenes_readme/img/grid_BendPickMP_lv1.jpg) · [lv2](http://10.40.11.11:8899/simple_scenes_readme/img/grid_BendPickMP_lv2.jpg) · [lv3](http://10.40.11.11:8899/simple_scenes_readme/img/grid_BendPickMP_lv3.jpg) |
| XMovePick | 10/10 | 10/10 | 9/10 | 4/10 | [lv0](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMovePickTeleop_lv0.jpg) · [lv1](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMovePickTeleop_lv1.jpg) · [lv2](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMovePickTeleop_lv2.jpg) · [lv3](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMovePickTeleop_lv3.jpg) |
| Handover | 10/10 | 7/10 | 4/10 | 3/10 | [lv0](http://10.40.11.11:8899/simple_scenes_readme/img/grid_HandoverTeleop_lv0.jpg) · [lv1](http://10.40.11.11:8899/simple_scenes_readme/img/grid_HandoverTeleop_lv1.jpg) · [lv2](http://10.40.11.11:8899/simple_scenes_readme/img/grid_HandoverTeleop_lv2.jpg) · [lv3](http://10.40.11.11:8899/simple_scenes_readme/img/grid_HandoverTeleop_lv3.jpg) |
| LocoPickBetweenTables | 3/10 | 9/10 | 4/10 | not evaluated | [lv0](http://10.40.11.11:8899/simple_scenes_readme/img/grid_LocomotionPickBetweenTablesTeleop_lv0.jpg) · [lv1](http://10.40.11.11:8899/simple_scenes_readme/img/grid_LocomotionPickBetweenTablesTeleop_lv1.jpg) · [lv2](http://10.40.11.11:8899/simple_scenes_readme/img/grid_LocomotionPickBetweenTablesTeleop_lv2.jpg) · [lv3](http://10.40.11.11:8899/simple_scenes_readme/img/grid_LocomotionPickBetweenTablesTeleop_lv3.jpg) |
| XMoveBendPick | 7/10 | 8/10 | 8/10 | not evaluated | [lv0](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMoveBendPickTeleop_lv0.jpg) · [lv1](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMoveBendPickTeleop_lv1.jpg) · [lv2](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMoveBendPickTeleop_lv2.jpg) · [lv3](http://10.40.11.11:8899/simple_scenes_readme/img/grid_XMoveBendPickTeleop_lv3.jpg) |

Level 0 is new distractors + table material, level 1 adds lighting, level 2 adds object poses, level 3 is levels 0+1+2 combined
plus table top z (±4 cm) and robot start (±10 / ±5 cm) in every scene.

## 2 · Real-world scenes rebuilt in MuJoCo

Each is a SIMPLE task plus a replay tool: the real teleop recordings are played back inside the scene and scored by **gates**,
ordered checks that latch when met, each only after the one before. Reward grows per gate; the last gate is success. One success
each below. Each scene also ships as SIMPLE evaluation sets at **levels 0–3**, rendered in Isaac (20 scenes at levels 0–2, 10 at
level 3), the same ladder as section 1. The same gate code scores the replays and the SIMPLE task.

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
- **success, rendered in Isaac** (level 0, scene 0): [video](http://10.40.11.11:8899/simple_scenes_readme/vid/isaac_bottle_bin.mp4),
  the recorded episode (session 02-25-56 episode 10) replayed in the level-0 scene: left Isaac third person, middle Isaac head
  camera, right the real head camera, in step
- **deployed model** `chipcan_nativec9` (job `qwen3_2b_posttrain_96d_g1_teleop_chipcantotrash_base_native_tokenizer_norm_20260917_131811`,
  checkpoint_9 of 0..19; stage-2 24-task ckpt4 init, lr 3e-5, pretraining tokenizer quantiles, ChipCanToTrash 02-25-56 session, 97
  episodes): [video](http://10.40.11.11:8899/simple_scenes_readme/vid/pipetest_bottle_bin_isaac.mp4). Gating on this episode:
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
- **success, rendered in Isaac** (level 0, scene 0): [video](http://10.40.11.11:8899/simple_scenes_readme/vid/isaac_bowl_sink.mp4),
  the recorded episode (BowlToSink_0918 episode 41, walking gain 1.5, robot start 0.38 m from the counter edge, moved back on
  2026-09-24 so the walk-in stops short of the cabinet) replayed in the level-0 scene: left Isaac third person, middle Isaac head
  camera, right the real head camera, in step
- **deployed model** `bowltosink_c9` (job `qwen3_2b_posttrain_96d_grouped_diffusion_bowltosink_20260918_103225`, checkpoint_9,
  last of 0..9; stage-2 24-task ckpt4 init, lr 1e-4, BowlToSink_0918, 55 episodes):
  [video](http://10.40.11.11:8899/simple_scenes_readme/vid/pipetest_bowl_sink_isaac.mp4). Gating on this episode: at bowl met at
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
- **success, rendered in Isaac** (level 0, scene 0): [video](http://10.40.11.11:8899/simple_scenes_readme/vid/isaac_coffee_cart.mp4),
  the recorded episode (CartCoffeeCup_0919 episode 64) replayed in the level-0 scene (room hssd:scene31 with its furniture hidden
  since 2026-09-24: open floor between the cart and the desk and beyond it, no wall at the end of the push): left Isaac third
  person, middle Isaac head camera, right the real head camera, in step
- **deployed model** `cart_c19` (job `qwen3_2b_posttrain_96d_grouped_diffusion_cartcoffeecup_20260920_122057`, checkpoint_19, last
  of 0..19; stage-2 24-task ckpt4 init, no camera model, lr 1e-4, CartCoffeeCup_0919, 97 episodes):
  [video](http://10.40.11.11:8899/simple_scenes_readme/vid/pipetest_coffee_cart_isaac.mp4). Gating on this episode: cart pushed,
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
