# g1comp visual head assets

Source: private `samebg_sim_datasets_20261012/robot/head_meshes/` bundle
provided by Alan Jiang (2026-10-12). These are the unmodified donor STL files.
They are not assets downloaded by SIMPLE's standard download script.

`simple.robots.g1comp_assets` combines these three meshes with the existing
`robots/g1_sonic/g1_29dof_with_hand.xml` at runtime. The chain is fixed at pan
0 degrees and tilt 10 degrees, with no mass, collision or additional joints.
The camera mount and lens-origin calculation match the donor generator.
Isaac uses the existing G1 USD plus these same meshes as visual-only children
of the torso. No additional URDF, prebuilt MJCF or USD asset is required.

Set `SIMPLE_G1COMP_ASSET_DIR` to the `head_meshes/` subdirectory when assets are stored separately.
Generated MJCF is written only to `SIMPLE_SCENE_CACHE_DIR` (or the local cache
fallback). Keep that cache off shared bucket mounts.

SHA-256:

- `d455_link.STL`: `ae00356d5505dd26c71141030cc2f738e7bafb694074f7141e089624ef9897b7`
- `head_servo_link.STL`: `c6e39a7e66ab82cd486486652cb3ee249acd41bc544b48fc102fac5e422b123c`
- `xl330_link.STL`: `3a4207216ec90bb76394760583a2056012ff6b9097e10c22937e59f3d51057e5`
