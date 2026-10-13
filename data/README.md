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

## samebg evaluation publication

Git tracks the robot implementation and `robots/g1comp/README.md` provenance.
The eval dataset `evals_scenes_3real_samebg/`, original donor bundle,
`robots/g1comp/head_meshes/`, and `publish_staging/` remain ignored.
Evaluation reads the published bucket copy through the existing launcher.
Git ignore rules do not define bucket upload contents.

Publish only these paths under the existing siqi SIMPLE data root:

| Local source | Bucket path relative to the data root |
|---|---|
| `evals_scenes_3real_samebg/` | `evals/evals_scenes_3real_samebg/` |
| `robots/g1comp/head_meshes/` | `robots/g1comp/head_meshes/` |
| `robots/g1comp/README.md` | `robots/g1comp/README.md` |

Keep the complete LeRobot `meta/`, `data/`, and `videos/` directories. Do not
upload the donor `psi0/`, `_states/`, `code/`, URDFs, generated XML, logs, caches,
or render outputs. Reuse existing standard assets and evaluation sets; inspect
missing dependencies before adding any other files to the publication list.
Use `rsync --inplace`, first with `--dry-run`, and verify checksums afterwards.
Never sync this entire data directory or use `--delete` for this publication.
Source code is published separately as a new versioned SIMPLE snapshot.

The local `evals_scenes_3real_samebg/<task>/dr-level-3/` directories copy the
donor `scene_sets/<task>/samebg-seed1000/` without changing file contents.
`dr-level-3` is the directory convention only: these are still fixed seed-1000
states, not newly sampled level-3 scenes. Publish the normalized local directory,
not the original donor `scene_sets/` tree.
