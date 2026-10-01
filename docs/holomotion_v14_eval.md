# VLA evaluation with the HoloMotion v1.4 controller

`python -m simple.cli.eval_holomotion_v14` evaluates a VLA in the three real-to-sim scenes (bottle → bin, bowl →
sink, coffee cart → table) with the controller the teleop data was collected with. The loop is `simple.cli.eval_decoupled_wbc`'s: the VLA sits behind
the same HTTP client (`simple.baselines.client.HttpActionClient`, `POST /act`, optional `GET /info`), returns action
chunks, and the task check decides success. The controller is the one change. Instead of the decoupled WBC turning
upper-body targets and a navigate command into joint targets, HoloMotion v1.4's motion-tracking policy (`model_22000`,
the 3.2 kg backpack model) tracks the reference frames the VLA returns.

```bash
# 1. your VLA server, answering POST /act on port 21000
# 2. the evaluation, 30 level-3 scenes per task
python -m simple.cli.eval_holomotion_v14 --scene bottle_bin  --host 127.0.0.1 --port 21000
python -m simple.cli.eval_holomotion_v14 --scene bowl_sink   --host 127.0.0.1 --port 21000
python -m simple.cli.eval_holomotion_v14 --scene coffee_cart --host 127.0.0.1 --port 21000
```

Instructions sent to the VLA:
- bottle_bin: "pick up the bottle, move towards the trash bin, and place the bottle in the trash bin";
- bowl_sink: "pick up the green bowl from the counter, turn right, move towards the sink, and place the bowl in the sink";
- coffee_cart: "push the cart to the table, then pick up the coffee cup and place it on the table".

Success is the task's last gate at reward ≥ 0.9: the bottle in the bin, the bowl in the sink, the cup placed on the
table. `results.json` also records every gate:
- bottle_bin: grasped, at_bin, placed;
- bowl_sink: at_bowl, grasped, at_basin, placed;
- coffee_cart: cart_pushed, cup_lifted, placed.

The motion model is not in git. Copy it once from the NAS:

```bash
rsync -a /mnt/nas28/alan.jiang/holomotion_models/v14_models_backpack_3p2/ data/holomotion/v14_models_backpack_3p2/
(cd data/holomotion/v14_models_backpack_3p2 && sha256sum -c SHA256SUMS)
```

## What is fixed to the teleop defaults

| | |
|---|---|
| robot | G1 + Dex3 + HBVCAM stereo head camera + 3.2 kg backpack, camera tilted 10° down |
| controller | HoloMotion v1.4, motion model `model_22000` (`--motion-model public` for `model_16200`) |
| scenes | each task's level-3 eval set, `data/evals_scenes/<env>/dr-level-3`: 30 scenes, each with its own robot start, item spot, table / counter / cart-box height, 3 distractors and look (below) |
| start | quick start: standing on the floor, no leash, a random start pose, walking policy on |
| VLA image | the HBVCAM rectified left eye, 1280 × 720 (the teleop's recorded `ego_view`); `--image-camera fisheye`, `--image-size 640x360` |
| success | the task check at reward ≥ 0.9 (bottle_bin: placed in the bin). The original eval's default of 0.5 would count grasped + at bin. |

**The level-3 sets, adapted to the v1.4 start.** Each task has 30 scenes:
- bottle_bin and bowl_sink: rebuilt on 09-28 from the replay fits and checked graspable;
- coffee_cart: the make_levels draw, ±10 / ±5 cm and ±4 cm.

Two need a start adjustment, because the v1.4 controller stands with its hands at table height. bowl_sink's are used
as built.

*bottle_bin.* The scenes were built from the decoupled-WBC replay fits, with the robot's feet 3 cm from the table. The v1.4 controller stands with its hands at table height, so at 3 cm they land on
the table; the teleop data starts at 0.35 m (±10 cm). Each scene is loaded exactly (its environment_config, as
`eval_decoupled_wbc --data-format lerobot` and the replay load it), with one change: the table, bottle, distractors and
(Isaac) lights move 0.32 m away from the robot. The kit's legs, cover board and bin follow from the layout, imported at
the teleop start distance (`--start-distance`), and the robot keeps its level-3 start near the origin, as in the
teleop data. MuJoCo's one light is fixed at the origin, so the image is lit as in the teleop data. Checked on scenes
0 and 1:
- the feet start 0.350 m from the table;
- the table heights (−1.7, +0.3 cm) and bottle spots match the set's build record;
- legs, cover board and bin line up with the moved table.

*coffee_cart.* The fingers reach the cart handle about 5 cm ahead of the nominal start. With the scenes as built, 12 of
30 had the hands within 3 cm of the cart and 6 had them inside it. The robot therefore starts 10 cm further back in every
coffee_cart scene. Its starts then span x −0.20 to 0, as the teleop's lv3 coffee_cart setups do. The rest of each scene
is unchanged.

*All 90 scenes, checked (2026-09-30):*
- each loads through the eval;
- the robot stands 3 s on the walking policy: none fell (lowest pelvis 0.752 m), no errors;
- the standing robot is more than 3 cm from the furniture in every scene.

`results.json` records `pelvis_min_m` and `fell` (pelvis below 0.5 m) per episode.

**Scene code.** The kits are imported from `~/wrk/robot_orchard_deploy/holobrain_g1_deploy/sim` when that folder
exists, else from the repository's `scenes/` (`HOLOBRAIN_SIM_DIR` overrides both). Checked on 2026-09-30 with the
branch's `scenes/` copies:
- all three tasks load, send the right instruction and stand;
- over 150 steps the physics is bit-identical to the workstation's kits;
- the VLA image is the same for bottle_bin and coffee_cart;
- for bowl_sink it differs under the counter: the branch copy predates the 09-27 sink-cabinet toe space.

Every scene gets a random start pose seeded by `--seed` + scene index. `--episode-start` and `--num-episodes` pick
scenes (default all).

Other scene sources:
- `--eval-set <dir>`: another LeRobot eval set (its `meta/scene_env.json` is applied before the kit is imported, as
  in `eval_scene.py`).
- `--eval-set seeds`: fresh level-3 setups from `--seed` + index (16 by default).
- `--scenes-from <teleop dataset>`: recorded teleop episodes, with their exact scenes and start poses.

## The VLA protocol

**Request** (`POST /act`, the HttpActionClient message; numpy arrays base64-encoded):

- `image`: `{"observation.images.ego_view": uint8 (720, 1280, 3)}`. Set the key with `--image-key`.
- `instruction`: the task's instruction ("pick up the bottle, move towards the trash bin, and place the bottle in the
  trash bin").
- `state`: named like the teleop dataset's columns:
  - `observation.state`: the 43 joint positions;
  - `observation.base_pose`: 7 values;
  - `observation.base_vel`: 6 values;
  - `teleop.latest_obs`: the reference frame the controller is on now (65);
  - `policy.mode`: 0 walking, 1 motion tracking.
- `history`: `session_id` (new per episode), `episode_index`, `step_index`, `frame_index`, and `reset: true` on an
  episode's first query.
- `dataset_name`: `"simple"`.

**Reply** `action`: shape (T, D), one row per reference frame, at 50 Hz by default (`--reference-hz`):

| columns | what |
|---|---|
| 0–64 | the reference frame model_22000 tracks: `dof_pos[29]`, `dof_vel[29]` in the robot's joint order (`complete_dof_order` in `third_party/holomotion_v14/config/g1_29dof_holomotion.yaml`), `root_pos[3]`, `root_rot_wxyz[4]`. This is exactly the dataset's `teleop.latest_obs`. |
| 65–66 (D = 67) | left and right grip, 0–1: the teleop's Dex3 grip gripper, where more than 0.5 closes the hand |
| 65–78 (D = 79) | left and right hand joint targets, 7 each, in SIMPLE's MJCF order (`G1Sonic.joint_names[29:43]`). These are the replay logs' `action_left_hand_q` / `action_right_hand_q`; the parquet `action` holds the same values in the decoupled-WBC order. |
| D = 65 | the hands stay open |

## How an episode runs

1. **Reset and stand.** The setup is applied, the scene reset, and the robot placed standing in its (random) start
   pose. The walking policy switches on at the first step, as in the teleop quick start.
2. **Walking mode until the VLA's frames are valid.** The VLA is asked from the first step. Its frames stream into the
   policy node, one per 50 Hz control step, right before the node's step, where the teleop fed the publisher's frames.
   A new chunk is queried when the current one runs out. Meanwhile the walking policy, with no walking command, keeps
   the robot standing in place. It moves the arms and wrists to its own posture within about 0.2 s, exactly as in
   teleop before the operator's Y.
   - A reply is used only if it parses, has shape (T, 65 | 67 | 79), is finite, and its root quaternions have unit norm.
   - A failed query or an unusable reply feeds nothing: the robot keeps walking and the next step asks again.
3. **Switch to motion tracking, automatically.** The node needs 11 valid frames (its 10-frame window plus one). Then
   the eval presses the motion-tracking button, as the operator's Y. With a VLA that answers from the start, that is
   step 14 (0.28 s).
4. **Run.** Motion tracking follows the VLA's frames. If a reply is unusable, no frame goes in that step. The node then
   treats it as a stalled stream on the robot: the reference holds, and freezes once it is older than 0.6 s.
5. **End.** The episode ends the moment the task check passes (bottle_bin: the bottle in the bin and released), at
   `--max-episode-steps` (1500 = 30 s), or with a "no valid VLA frames" error if none arrive within
   `--engage-timeout-s` (10 s).

`results.json` records each episode's controller timeline (for example `step 0: POLICY / velocity | step 14: POLICY /
motion`), the first valid VLA step, and how many replies were not used and why. `--settle-s` adds standing time before
the first query (default 0).

Checked 2026-09-30 with the stand-in VLA:

| case | first valid frames | motion tracking from | until then |
|---|---|---|---|
| answers from the start (bottle_bin) | step 2 | step 14 | walking, standing in place |
| first 10 replies NaN (bowl_sink) | step 12 (10 replies not used) | step 24 | walking |
| no server (coffee_cart, 2 scenes) | none | never | walking; each episode ends after the 2 s test timeout with the reason |

None fell, and the episodes replay bit-exact. Standing in walking mode for 6 s with no VLA (5 start poses), the robot
moved 1.2–2.2 cm and turned less than 1°. Holding the arms at the start pose instead, against the walking policy, was
tried and dropped: the robot drifted 29–90 cm and fell once.

The controller stack (the vendored node and the agent) runs on a clock that moves exactly 20 ms per control step
(`teleop/holomotion_v14/sim_clock.py`). On the wall clock, VLA latency would age the reference (the node freezes it
past 0.6 s) and shift the node's timers. With this clock, a rollout depends only on the scene, the seed and the VLA's
replies.

## Outputs

Everything goes to `<eval_dir>/<policy>/<env>/<scenes>/`. When the server has `/info`, `<env>` is prefixed with its
policy and timestamp, as in the original eval.

| File | Contents |
|---|---|
| `results.json` | per episode: success, gates (grasped / at_bin / placed) and their times, steps, VLA queries and latency, the setup seed, table offset, errors |
| `videos/episode_N.mp4` | the VLA image beside a third-person view, with phase and gates overlaid |
| `replay/episode_N.npz` | the bit-exact replay log. Replay with `python -m simple.cli.replay_holomotion_v14 <that folder> --all --mode action`. |

`eval_stats.txt` in `--eval-dir` gets `episode_N: True/False` lines, as in the original.

## Checks (2026-09-30)

A stand-in VLA, `scripts/holomotion_v14_episode_server.py`, serves a recorded teleop episode's own reference frames
and hand targets. It checks the pipeline, not a policy.

```bash
.venv/bin/python scripts/holomotion_v14_episode_server.py <teleop dataset> --episode 9 --port 21090 [--delay 0.3]
python -m simple.cli.eval_holomotion_v14 --scenes-from <teleop dataset> --episode-start 9 --num-episodes 1 --port 21090
```

- **The recorded scene:** the eval rebuilt the scene of teleop episode 9 as a byte-identical model.
- **The motion:** motion tracking engaged at step 40 (with the first version's fixed 0.5 s settle; now step 14). The robot walked to the table, reached around the bottle, then
  turned and walked to the bin. Open-loop, the grasp missed; the bottle ended up on its side.
- **Replay:** the eval's replay log replays bit-exact (900 frames).
- **VLA latency:** with the server delayed 0.3 s per reply (0.316 s mean vs 0.013 s), the rollout was bit-identical in
  start state, physics, controls and targets. The same held for a seeded lv3 scene.

**Found on the way:** SIMPLE's scene randomizers draw the distractor pick, their turn, the materials and stable poses
from Python's `random`, which the setups did not seed. Seeding both numpy and `random` since 2026-09-30 makes a setup
seed fix the whole scene. Recorded episodes are unaffected, because they keep their exact scene.

## Pipeline check with a real model (2026-09-30)

The three HoloBrain G1 deploy models (`~/wrk/robot_orchard_deploy/models`, presets `chipcan_nativec9`, `bowltosink_c9`,
`cart_c19`, served by `holobrain_g1_deploy/scripts/serve.sh`) were put in the loop, one per task. They command the
decoupled WBC (upper-body targets + a walking command), so `scripts/holomotion_v14_vla_bridge.py` sits between the eval
and the model server: it sends the model its training format (`rgb_head_stereo_left` 640 × 360, `states` 32) and
turns each reply (24 × 36 rows at 50 Hz) into 24 × 79 reference frames: legs at the default standing pose, waist and
arms from the model, root integrated from its vx / vy / vyaw, hands reordered. A check of the pipeline, not of the models.

```bash
cd ~/wrk/robot_orchard_deploy/holobrain_g1_deploy && PORT=8014 bash scripts/serve.sh chipcan_nativec9 --steps 8 --replan 15
cd ~/wrk/SIMPLE && .venv/bin/python scripts/holomotion_v14_vla_bridge.py --upstream-port 8014 --port 21000 --preset chipcan_nativec9 --log bridge.jsonl
python -m simple.cli.eval_holomotion_v14 --scene bottle_bin --port 21000 --image-size 640x360 --num-episodes 5
```

Result, 5 level-3 scenes per task (report: http://10.40.11.11:8899/holomotion_v14_pipeline_check/index.html, generator
`docs/_scan/holomotion_v14_pipeline_check/build.py`):
- every query reached the model with a 640 × 360 image and a finite 32-state; every reply was 24 × 36 and finite;
- every bridge frame passed the eval's validity check (0 of 977 replies rejected);
- motion tracking engaged at step 15 in all 15 episodes; all ran the full 30 s, no errors, no falls;
- model latency median 766 ms (three servers on one GPU; 420 ms alone);
- all 15 replay logs replay bit-exact;
- the same scene run twice differs from the first reply on: the models sample (grouped diffusion); each run is still
  reproducible from its log.
- Task gates: bowl_sink reached `at_bowl` in 3 of 5 scenes (its model walks toward the counter); nothing else, as
  expected with faked references, an unfamiliar camera and, for bottle_bin, a model trained on the chip can.

**Not the same as `eval_decoupled_wbc`:** one worker per process, MuJoCo only (no Isaac rendering), no policy-specific
agents. The VLA protocol above replaces the per-baseline agents.
