# Three-real-scene robot configuration

BottleBin, BowlSink and CoffeeCart use the `g1comp_sonic` registered robot.
The stored spatial-state UID remains `g1_sonic`, so existing scene datasets
keep their original robot placements; `runtime_version` identifies the new
`g1comp-pan0-tilt10-v1` visual/camera configuration. Other tasks select the
original G1 SONIC registration.

The head is a massless, collision-free fixed chain at pan 0 / tilt 10 degrees.
The D455 camera is attached to `d455_link`; the policy observation name remains
`head_stereo_left`, 640 x 360. Existing task timing is unchanged. Isaac copies
the MuJoCo camera world pose through the existing task camera synchronization.
The new head is also added to the Isaac USD torso as real visual mesh geometry.

Deployment requires the updated SIMPLE source snapshot, the existing base G1
SONIC MJCF/meshes and G1 USD, plus these three files in
`robots/g1comp/head_meshes` (or `SIMPLE_G1COMP_ASSET_DIR`):

- `head_servo_link.STL`
- `xl330_link.STL`
- `d455_link.STL`

The source is the private 2026-10-12 samebg bundle. The local assets README
records original checksums. No new head URDF or generated XML/USD needs upload.
Set `SIMPLE_SCENE_CACHE_DIR` to worker-local writable storage; generated files
must never be placed beside read-only shared robot assets.

Both ordinary and samebg real-scene evaluation use this robot. The two datasets
remain separate; changing the head does not select or alter episode states.
