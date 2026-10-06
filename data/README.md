# SIMPLE data and generated resources

Runtime code and tools live under `src/`, `scripts/`, and `scenes/`. Scene XML,
OBJ/USD meshes, layouts, datasets, and generated outputs belong under `data/`
(or the root selected with `SIMPLE_DATA_DIR`), not next to tool source files.

- `real_scenes/<scene>/layout.json` and `assets/`: fixed scene resources.
- `robots/g1_comp/`: robot XML, meshes, and original licenses.
- `real_scenes/<scene>/scene.xml`: default scene-builder output.
- `real_scenes/<scene>/renders/`: generated previews, ignored by Git.
- `evals_scenes/`: existing fixed evaluation datasets.

The scene-builder compatibility entrypoints in `scenes/<scene>/build_scene.py`
write into the scene's data directory by default. Build locally with
`SIMPLE_DATA_DIR=data`; published bucket resources are read-only during eval.
Evaluation-generated parameterized meshes use a separate local
`SIMPLE_SCENE_CACHE_DIR`/`XDG_CACHE_HOME` cache.

## Adding a new data component

The root `.gitignore` explicitly lists existing downloaded data/dataset
components. It no longer ignores every new directory under `data/`.
Consequently a new fixed-resource component is visible to `git status` without
`git add -f`. Existing ignored downloads stay ignored, and files already
tracked remain tracked.

Before adding a component, inspect `git status --short -- data/<component>`
and `git check-ignore -v --no-index <file>`. Track its fixed resources, layout,
license, and provenance. Add an explicit ignore rule for new large downloaded
content, datasets, or generated caches; do not commit those by accident.
Render/replay/probe/report outputs and old height-variant meshes remain ignored.
