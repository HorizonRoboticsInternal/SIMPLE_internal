# Serving coffee with a cart (g1comp, MuJoCo) + SIMPLE task + real-recording replay

Built 2026-09-23 from the hand drawing `~/Downloads/servingcoffeewithcart.jpg` (cm unless it says mm; the route's "220mm" is
220 cm, the same slip as the bowl-sink sheet) and tuned against all 97 episodes of
`SIMPLE/data/real_recordings/psi0/CartCoffeeCup_0919`. Same recipe as `../bottle_bin/` and `../bowl_sink/`.

    MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py          # scene.xml, layout.json, renders/, plan.png (defaults = the tuned route 2.8 / 0.4 since 2026-09-24)
    MUJOCO_GL=egl  ~/wrk/SIMPLE/.venv/bin/python probe_scene.py                     # the task inside SIMPLE -> probe/
    MUJOCO_GL=egl  ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --fast --jobs 4 --episodes 0 12 24   # replay (deterministic)
    MUJOCO_GL=egl  ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --fast --video --episodes 84         # + mp4, same result
    python collect_replay.py --baseline-tag rc020_rt280_tl040 --video replay_ep84.mp4 && python make_report.py

World frame: pelvis at the origin at the start pose, x the way the robot faces, z up, the robot's right = -y.

| item | value | source |
|---|---|---|
| table | 1.06 across x 0.60 deep, top 0.74 m, slab on four legs | drawing "106 x 60 x 74" |
| route | table's far edge 2.8 m ahead, its left end 0.4 m to the robot's right | drawing 2.2 / 0.67; moved by the replay sweep (below) |
| cart | 0.71 x 0.47 deck, top 0.12 m on casters, 15 kg, one rigid free body | drawing "cart 71 x 47", "wheel 120mm" |
| box on it | 0.60 x 0.35 x 0.68, 9 cm of bare deck at the handle end, top 0.80 m | drawing "box 60x35x68", "9" |
| handle | 30 mm bar 0.325 m ahead of the pelvis, 0.77 m high | drawing "10cm Robot" +10 (user); height measured from the recorded closed hands |
| cup | 9 cm dia x 14 cm, 0.35 kg, near-right corner of the box top | drawing "9cm diameter, 14cm high", "10cm x 10cm at corner" |

**Physics that matters.** MuJoCo pairs two geoms' friction by MAX unless one outranks the other: the floor is mu = 1.0, so the wheel
geoms carry `priority=1` with rolling resistance 0.015 (MJCF `priority`, mjSpec `geom.priority`, per-piece `piece_priorities` in the
task's engine patch). Without it the robot cannot push the cart at any mass. Gravity compensation is on in the replay
(`enable_gravity_compensation` on the sonic config) so the arms hold the commanded pose.

**Isaac room.** `COFFEE_CART_ROOM` (default `hssd:scene31`) with `COFFEE_CART_ROOM_DX` (1.0 m ahead of the stock placement). SIMPLE puts the
room's own counter at the robot's start and the route runs away from it; scene31 leaves ~5.1 m ahead and ~4.4 m on the turn side
(scene2 had a counter 1.65 m to the right at the stop). The `dr-level-*` sets pin the room in their state: regenerate them after changing it.

**Task** (`coffee_cart_task.py`, `simple/G1WholebodyCoffeeCartTeleop-v0`, sonic teleop stack): SIMPLE's table primitive is the
serving table top; legs static; cart and cup free objects. Checkers in `compute_reward` / `check_success`, written to
`info["progress"]` every step: `cart_pushed` (>= 0.5 m, sticky), `cup_lifted` (>= 5 cm, sticky), `cup_upright` (< 45 deg),
`cup_resting_on_table` (footprint, upright, base within 3.5 cm of the top), `cup_in_hand` (hand-cup contact), the failure states
`cup_tipped_on_table` / `cup_on_floor` / `cup_still_on_cart`, and `success` = resting on the table and out of the hand for 25
steps (sticky). Reward 0.25 pushed + 0.25 lifted + 0.5 success. Env vars: `COFFEE_CART_ROBOT_TO_CART`, `_ROUTE_FORWARD`,
`_TABLE_FROM_LINE`, `_CUP_XY`, `_CUP_MASS`, `_CART_MASS`, `_WHEEL_FRICTION`, `_HANDLE_TOP`, `_INSTRUCTION`, `_CHK_*` (`_CHK_REQUIRE_RELEASE=0` counts a cup resting upright on the table even with the hand still on it: 9/96 instead of 5/96).

**Replay** (`replay_in_scene.py`): the recorded 36-D psi0 rows drive `PicoDecoupledAgent` at 50 Hz through real contact; the
agent and teleop policy read a sim clock instead of `time.monotonic()` (`--wall-clock` restores the old behaviour), so runs are
bit-reproducible and the `--fast --video` mp4 is the same run as the sweep. Keep the machine-wide worker count <= 4-10: the
env's DDS domain overflows beyond that and the robot silently ignores commands (a run that never moves exits 3, no summary).
Knobs: `--robot-to-cart`, `--route-forward`, `--table-from-line`, `--cup-xy`, `--cart-mass`, `--wheel-friction`, `--handle-top`,
`--post-turn-gain`, `--waist-mode`, `--cart-attach` (default none), sweeps via `--sweep-*` / `--table-pairs`.

Results: see `site/index.html` (make_report.py) - route 2.8 / 0.4: cart pushed in 95/96 episodes (2.21 m mean), cup lifted 39/96,
cup placed upright on the table 9/96 (+11 tipped onto it). Episode 77 is a 2.3 s aborted recording. The ceiling is the open-loop
push distance spread (release x 1.8-2.8 m against a 0.6 m table depth); a post-turn nav gain does not help.

## Levels 0-3 in Isaac (2026-09-24)

The same level design as SIMPLE's benchmark sets, rendered in Isaac in the plain HSSD room `scene2` (no living-room furniture; `../make_levels.py`; the Isaac renderer needs the
`Meshes/visual` + `Meshes/collision` layout in the generated `assets/*.usda`, which `_write_usda` now writes):

| level | what changes from the base scene | knob |
|---|---|---|
| 0 | 3 GraspNet distractors on the table and a new table + ground material (object and furniture shader params pinned: a random metallic bowl rendered black) | `COFFEE_CART_NUM_DISTRACTORS` |
| 1 | + new lighting | |
| 2 | + the cup anywhere on the near half of the box top (`COFFEE_CART_TARGET_REGION="0.475,0.65,-0.115,0.115"`, clipped to the top minus the cup radius + 1.5 cm so it cannot fall) | `COFFEE_CART_TARGET_JITTER` / region env var, see `../make_levels.py` |
| 3 | + alternative layout: 10 scenes, each with a different robot start (up to 10 cm back/forward, 5 cm sideways) AND a different table height (±4 cm; SIMPLE's DR manager shifts the table, the cart box (the cup's support) and table legs follow through per-height mesh variants `<asset>_p031`/`_m040`) | `ROBOT_BACK`, `ROBOT_LEFT`, `LEVEL3_EPISODES` in `../make_levels.py`; the start is stored in each scene's state, so eval needs no env var |

```
python sim/make_levels.py coffee_cart --levels 0 1 2 3 --episodes 10 --out data/evals_scenes
    -> data/evals_scenes/G1WholebodyCoffeeCartTeleop-v0/dr-level-<n>/  (LeRobot set as data/evals_new20; frames/ep*.png = Isaac head camera AFTER the same stabilisation the eval runs, i.e. the controller's start pose;
       meta/scene_env.json = the env vars used, re-applied by eval_scene.py)
python sim/eval_scene.py coffee_cart simple/G1WholebodyCoffeeCartTeleop-v0 psi0_decoupled_wbc train --data-format lerobot \
    --data-dir data/evals_scenes/G1WholebodyCoffeeCartTeleop-v0/dr-level-3 --port 21000 --headless --num-episodes 20
```

## Head tilt: servo goal 2300 (2026-09-24)

`build_scene.py` now defaults to `--tilt-ticks 214` (goal 2300 = home 2086 + 214) and `coffee_cart_task.py` to
`COFFEE_CART_HEAD_TRIM_DEG=8.8` on its encoded 2200 baseline: the D455 axis is 66.5 deg below horizontal (was 57.7 at 2200).
Caveat: a real first frame of `CartCoffeeCup_0919` (handle bar at the bottom of the frame, box top mid-frame) looks less
pitched than 66.5 deg, so those recordings were probably not taken at 2300; for replays against them set
`COFFEE_CART_HEAD_TRIM_DEG` / `--tilt-ticks` back to the goal they used.

Isaac head camera: the kit's D455 pose is a MuJoCo-side patch, so the task sets `isaac_cameras_follow_mujoco = True` and SIMPLE's Isaac engine copies the MuJoCo camera's world pose every step (`_sync_cameras_from_mujoco`); without it the Isaac view is narrower and higher than the verified MuJoCo/real view.

Isaac-rendered replay (`../replay_isaac.py`): runs this kit's `replay_in_scene.py` unchanged inside the SIMPLE level-N scene in
`mujoco_isaac` mode (the env is reset to one scene of `data/evals_scenes`, so the distractors, table material and lighting are
the set's; the middle video panel is then the Isaac head camera). The wall-clock-paced replays get bowl_sink's VirtualClock
installed from outside and ticked per `env.step`, so the controller runs deterministically at 50 Hz despite Isaac's slow steps.
```
python sim/replay_isaac.py coffee_cart --episode 64 --scene 0 --ext-clock   # ep 84 (the MuJoCo showcase) does not survive the level-0 scene; 64/70/93 do      # -> replay/isaac_level0_ep<N>.mp4
```

Isaac third person (final recipe, 2026-09-24): two passes per episode plus a separate composer, because nothing can run after a kit
replay inside the Isaac process (SimulationApp shutdown segfaults). Pass A `--no-third` writes the kit video (middle panel = Isaac head
camera). Pass B `--hide-shell` keeps the kit's own free-camera pose (it stands outside the HSSD room, so the room's walls / ceilings /
openings prims are hidden) and saves an Isaac third-person + head JPEG every control tick to `replay/third_coffee_cart_ep64/`. The composer
then swaps the left panel in; the panels are frame-synchronous by construction: kit frame k (written every 2nd tick after the settle)
is tick S + 2k with S = M - 2N (M captured ticks, N kit frames), no image matching.
```
python sim/replay_isaac.py coffee_cart --episode 64 --scene 0 --ext-clock --no-third                                   # pass A -> replay/isaac_level0_ep64.mp4
python sim/replay_isaac.py coffee_cart --episode 64 --scene 0 --ext-clock --hide-shell --out sim/coffee_cart/replay/passB_ep64.mp4   # pass B -> replay/third_coffee_cart_ep64/f*.jpg h*.jpg
python sim/compose_third.py sim/coffee_cart/replay/isaac_level0_ep64.mp4 sim/coffee_cart/replay/third_coffee_cart_ep64 sim/coffee_cart/replay/isaac_level0_ep64_3p.mp4 sim/coffee_cart/replay/isaac_level0_ep64_3p.jpg 0
```

Isaac room furniture: the task sets `isaac_hidden_scene_groups = ("furniture",)`, so SIMPLE's Isaac engine hides the HSSD room's own furniture group (a 1 m high island of hssd:scene31 spans the whole cart route and a cabinet row stands where the desk is; with it visible the head camera showed "objects between cart and desk" and a "wall" at the end of the push). Visual only: MuJoCo has no room. Probe: scratchpad isaac_smoke/room31_probe.py prints every furniture / wall bbox and flags the ones inside the corridor.
Isaac room: `COFFEE_CART_ROOM_DX` (default 0.7 m) pushes the HSSD room ahead at reset so the 2.8 m route does not end inside its far wall (visual only; the head camera rendered black there).
Level-3 heights: `SIMPLE_LEVEL3_TABLE_DZ` (set by make_levels.py, 0.04) is the half-range of the per-scene table offset; the kit reads the actual table top at reset and loads the matching furniture variant.
