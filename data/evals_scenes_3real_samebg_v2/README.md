# Same-background evaluation v2

Each task contains 30 distinct 3real initial spatial states, paired by episode index.
Robot/task geometry and parquet states are preserved. Distractors are removed.
Room transform, materials and lights use the original samebg fixed-look template.
Use g1comp pan0/tilt10 and restore stored states without a runtime DR override.

Videos are unchanged 3real reference clips, NOT new v2 previews. No rendering was performed.
This data conversion does not change runtime joint restoration or warmup behavior.
validation.json records data-level checks; simulator validation is separate.

Original samebg backup: /home/users/kerou.zhang-labs/workspace/SIMPLE/data/evals_scenes_3real_samebg_backup_20261014_122238
