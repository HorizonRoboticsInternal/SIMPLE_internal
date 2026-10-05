# Scene evaluation

The three scene kits register these tasks when imported:

| Kit | Environment |
|---|---|
| `bottle_bin` | `simple/G1WholebodyBottleBinTeleop-v0` |
| `bowl_sink` | `simple/G1WholebodyBowlSinkTeleop-v0` |
| `coffee_cart` | `simple/G1WholebodyCoffeeCartTeleop-v0` |

Each kit includes its layout, task success checks, scene builder, and required assets.
Fixed evaluation inputs live in `data/evals_scenes/<environment>/dr-level-{0,1,2,3}`.
Levels 0–2 have ten scenes each; level 3 has thirty. Keep the dataset videos because
the LeRobot metadata references them. Reports and preview images are generated outputs.

Run a policy server, then evaluate a fixed scene set:

```bash
python scenes/eval_scene.py bowl_sink simple/G1WholebodyBowlSinkTeleop-v0 psi0_decoupled_wbc train \
    --data-format lerobot --data-dir data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-3 \
    --port 21000 --headless --num-episodes 30
```

The wrapper loads `meta/scene_env.json` before task registration so the layout matches
the dataset. Robot meshes are supplied by SIMPLE's normal robot assets download.

Regenerate scene sets with Isaac rendering:

```bash
python scenes/make_levels.py bowl_sink --levels 0 1 2 3 --episodes 30 --out data/evals_scenes
```

See [level3/README.md](level3/README.md) for benchmark reachability and stability validation.
