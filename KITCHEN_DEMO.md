# Kitchen demo branch (`alan_kitchen_demo`)

A snapshot of Alan's local SIMPLE working branch on 2026-11-11 (`alan/Simple` at `54718ab` plus the uncommitted work on
top of it), together with the kitchen scene kit and the assets it needs. It is here so the kitchen demo can be developed
into the eval pipeline. It is **not** based on `feat/new-simple-eval-tasks`: file layout and defaults differ from that
branch (tasks, data directories, robot variants), so expect a port, not a merge.

The kitchen demo: a G1 with BrainCo Revo 2 hands in a manipulable kitchen (16 mechanism joints such as the dishwasher
door, racks and the pedal bin; 8 free props), driven by the HoloMotion v1.4 controller. The long task "load the
dishwasher" is cut into slices; the atomic slices A1-A5 (walk to the counter, pick the plate up, carry it to the pedal
bin, step on the pedal, wipe the plate over the bin) have recorded reference data.

## What is where

| Path | What |
|---|---|
| `scenes/kitchen/` | The scene kit: `kitchen_task.py` (registers `simple/G1WholebodyKitchenSinkTeleop-v0`, task uid `g1_wholebody_kitchen_sink_teleop`; tasks via `KITCHEN_TASK`, default `load_dishwasher`), `build_scene.py` (frames and constants), `layout.json` |
| `scenes/kitchen/kitchen_scene/` | The authored kitchen: `scene.xml` + `assets/` (490 meshes, 11 textures). Attached whole to the SIMPLE scene |
| `scenes/bottle_bin/assets/bottle_500ml*` | The bottle the kitchen kit uses as SIMPLE's target object |
| `data/robots/g1_rot0_teleop_revo2/` | The robot: G1 + head stereo bracket + wrist cameras + Revo 2 hands with tactile pads (`g1_rot0_teleop_revo2_v3.xml`, the file every recording used), URDF, grip poses |
| `data/kitchen_plates`, `data/kitchen_plates_lift`, `data/kitchen_plates_v2` | The procedural plate sets (50 / 50 / 70 plates); a scene setup picks two plates from one set |
| `src/simple/cli/teleop_holomotion_v14.py` | Teleop + recorder (`--scene kitchen --robot revo2 --slice A1..A5`) |
| `src/simple/cli/replay_holomotion_v14.py` | Bit-exact replay of a recording, `--fill` (renders the cameras and tactile pads into a full dataset), `--gates` |
| `src/simple/cli/rerecord_holomotion_v14.py`, `eval_holomotion_v14.py` | Closed-loop re-record from recorded controller inputs; the VLA eval loop for the three older scenes (see "Eval pipeline" below) |
| `src/simple/teleop/holomotion_v14/` | Controller node wrapper, exact log, slices and their gates (`slices.py`), random scene setups and start banks (`scene_setup.py`, `start_banks/`), plate-in-hand starts (`grasp_start.py`), Revo 2 recording, wrist cameras, real-format columns |
| `src/simple/robots/g1_sonic_revo2.py`, `src/simple/sensors/revo2_touch.py` | The Revo 2 robot class and its tactile sensors |
| `third_party/holomotion_v14/`, `third_party/hbvcam_stereo/` | The vendored HoloMotion v1.4 robot code and the head stereo camera bundle |
| `scripts/kitchen_plates/`, `scripts/robots/`, `scripts/holomotion_v14_checks/` | Plate generator and swap, robot builders, check and analysis scripts |
| `docs/holomotion_v14_teleop.md` | The full how-to and change log (kitchen, slices, plates, recording, replay) |

## Setup

1. Install SIMPLE as usual (`README.md`), with git-lfs: the meshes and textures (`*.obj`, `*.png`) are LFS files.
2. `third_party/decoupled_wbc` is a submodule with local fixes that are not in its remote: apply
   `third_party/patches/decoupled_wbc_local.patch` as `third_party/patches/README.md` says.
3. Controller models are **not** in this repo. The default motion model (`model_22000`) and the walking model are
   expected in `data/holomotion/v14_models_backpack_3p2/`; `docs/holomotion_v14_teleop.md` ("Setup") gives the source
   and the checksums. Replaying recorded physics needs no model.
4. No workstation paths are required: the kit uses `scenes/kitchen/kitchen_scene`, `data/kitchen_plates*` and
   `scripts/kitchen_plates` of this checkout. Overrides: `KITCHEN_SCENE_DIR`, `KITCHEN_PLATES`, `HOLOBRAIN_SIM_DIR`
   (the folder that holds the scene kits).

## Run

Teleop and record an atomic slice (PICO headset + the reference publisher, see the doc):

```bash
scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --slice A2 --setup-ranges lv3 --lean-record \
    --save-dir data/<run>
```

Hand buttons (Revo 2, per hand, since 2026-11-11): the side (grip) button closes only the thumb, the four fingers stay
straight; the trigger gives the full close; both together give the full close. Poses and the button assignment are in
`data/robots/g1_rot0_teleop_revo2/grip_poses.json` (`"buttons"`); `docs/holomotion_v14_teleop.md` ("Revo 2 buttons") has
the details. Slices that start with the plate in the hand (A3-A5) hand it over on the first full close (the trigger).

Replay a recording bit for bit, and render its cameras and tactile pads into a full dataset:

```bash
python -m simple.cli.replay_holomotion_v14 <dataset>/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole --all --no-gates
python -m simple.cli.replay_holomotion_v14 <dataset>/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole --all --fill
```

A recording stores the path of its plate set as it was on the recording machine (`KITCHEN_PLATES`); when that path does
not exist, the kit takes the set of the same name from this checkout's `data/` and says so.

## Checked on this branch

From a copy of this branch with the workstation's kitchen scene, kit, plate folders, robot folder and teleop source
hidden: two reference A2 recordings (one per plate set) replayed **bit-exact** (`simple.cli.replay_holomotion_v14`).
The replay prints "model DIFFERENT": the model hash includes asset path strings; the physics is identical.

Not checked on this branch: live teleop, `--fill` rendering, the eval CLI.

## Eval pipeline: what exists and what does not

- `simple.cli.eval_holomotion_v14` runs a VLA against the HoloMotion controller on the three older scenes
  (`bottle_bin`, `bowl_sink`, `coffee_cart`). It accepts `--scene kitchen`, but that path has not been run: the
  Revo 2 robot, the 51-wide state / action, the slice starts (plate in hand, foot on the pedal) and the slice gates
  are not wired into it.
- What an eval of a slice needs is already in the teleop code and can be reused: the start of each slice
  (`scene_setup.py` + `start_banks/*.json`, applied by `grasp_start.apply_pose`), and its success test
  (`slices.SliceCut`: the end gate, the hold time and the end event per slice; `replay --gates` evaluates it offline).
- The A5 gate (three passes of the free hand over the plate) is met by few recordings; treat it as open.

## Not in this branch

Recorded datasets and renders, controller model weights, the generated HTML pages under `docs/`, the workstation's
one-off run scripts, the real-robot deploy code, the BrainCo SDK clone, the older HoloMotion v1.3 teleop's vendored
code (`third_party/holomotion`; `simple.cli.teleop_holomotion` needs it), and the second Revo 2 robot variant
(`--revo2-hand official`; `scripts/robots/make_g1_revo2.py --hand official` builds it).
