# Scene evaluation

The flat modules in `src/simple/tasks/` register these tasks through the normal SIMPLE package imports:

| Kit | Environment |
|---|---|
| `bottle_bin` | `simple/G1WholebodyBottleBinTeleop-v0` |
| `bowl_sink` | `simple/G1WholebodyBowlSinkTeleop-v0` |
| `coffee_cart` | `simple/G1WholebodyCoffeeCartTeleop-v0` |

This directory contains scene-building, generation, and replay tools. Task success checks live in `src/simple/tasks/`; layouts and fixed assets live in `data/real_scenes/`.
Fixed evaluation inputs live in `data/evals_scenes/<environment>/dr-level-{0,1,2,3}`.
Levels 0–2 have ten scenes each; level 3 has thirty. Keep the dataset videos because
the LeRobot metadata references them. Reports and preview images are generated outputs.

Run a policy server, then evaluate a fixed scene set:

```bash
python -m simple.cli.eval_decoupled_wbc simple/G1WholebodyBowlSinkTeleop-v0 psi0_decoupled_wbc train \
    --data-format lerobot --data-dir data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-3 \
    --port 21000 --headless --num-episodes 30
```

The standard evaluator applies `meta/scene_env.json` before environment creation. Robot and scene assets are supplied through the normal data directory.

Regenerate scene sets with Isaac rendering:

```bash
python scenes/make_levels.py bowl_sink --levels 0 1 2 3 --episodes 30 --out data/evals_scenes
```

See [level3/README.md](level3/README.md) for benchmark reachability and stability validation.

## Source files versus generated resources

This directory retains dataset-generation, replay, scene-building tools, and
compatibility entrypoints. Runtime tasks are the flat `g1_wholebody_*_teleop`
modules under `src/simple/tasks/`, following the maintained snapshot layout.
Fixed layouts and assets live under `data/real_scenes/<kit>/`; robot assets
live under `data/robots/g1_comp/`. Scene builders write XML, OBJ/USD and previews
under the scene data directory by default, not under this tools directory.
Keep other generated tool outputs under `data/` using their output options.
See [data/README.md](../data/README.md) for Git tracking and output conventions.

There are no duplicate `<scene>_task.py` runtime entrypoints here. Tools import
`simple.tasks.g1_wholebody_<scene>_teleop` directly; `build_scene.py` and
`eval_scene.py` remain command-line conveniences.
