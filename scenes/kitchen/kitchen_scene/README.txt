KITCHEN: EIGHT INDEPENDENTLY MOVABLE OBJECTS
26 October 2026

Project: /home/Horizon/wrk/KitchenScene_Manipulable_20261026
Report: http://10.40.11.11:8899/kitchen_scene_20261025/manipulable_v3/
Predecessor: /home/Horizon/wrk/KitchenScene_Objects_20261026

WHAT THIS REVISION DOES
The two loose blue-rim plates, red/green condiment bottles and two labelled
cartons are now independent free rigid bodies. Their authored appearance is
preserved. Two new hollow ceramic bowls are added as task objects; these bowls
are not claimed to be reconstructed from the capture. Other background props,
including the stacked plates, cup stacks and appliances, remain fixed.

Each selected object has a free joint, visual meshes, separate contact geometry,
explicit mass/center of mass/inertia, friction and origin/grasp/support sites.
There are no object welds or object-motion actuators. Runtime object motion is
ordinary MuJoCo contact dynamics. The existing 16 mechanism coordinates and 393
environment contact geometries are preserved, including the dishwasher tines.

ENTRY POINTS
scene.xml         Default runtime: passive environment, 8 free objects, 0 actuators.
scene_passive.xml Same as scene.xml, kept as an explicit passive entry point.
scene_demo.xml    Adds the predecessor's 15 mechanism demonstration actuators.
scene_contacts.xml Collision debug view; same passive dynamics.

The model contains 24 joints: 16 mechanism joints and 8 free joints (72 qpos).
Assets use relative paths. Copy the entire extracted folder to another machine.
No scan archive, Blender installation or predecessor scene is needed to run it.

QUICK START
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m mujoco.viewer --mjcf=scene.xml

Python reset/state helper:
    import sys
    sys.path.insert(0, 'scripts')
    from object_api import KitchenObjects
    kitchen = KitchenObjects()
    kitchen.reset()  # optional poses: {id: {position: [x,y,z], quaternion_wxyz: [...]}}
    kitchen.step(1.0)
    print(kitchen.states()['bottle_red'])

Object placement is reset-time only. Choose collision-free poses on a supported
surface. During a rollout use mj_step and robot forces, not qpos edits. This helper
does not itself validate arbitrary reset layouts or provide a robot controller.

OBJECT CONFIGURATION
objects.json is the editable configuration. object_physics.json adds the generated
collision inventory, COM/inertia and diagnostic-grasp configuration.

ID             MASS     CONTACT GEOMETRY
plate_a        0.030 kg  Base cylinder + 96 convex rim sectors (23 cm diameter).
plate_b        0.030 kg  Same geometry, independently movable.
bottle_red     0.250 kg  6 pieces: base/body, shoulder/neck segments and cap.
bottle_green   0.250 kg  6 pieces; independent shape and height.
carton_bags    0.160 kg  Box collider (17 x 25 x 16 cm).
carton_wrap    0.180 kg  Box collider (17 x 31 x 10.5 cm).
bowl_small     0.180 kg  Base + 96 wall sectors (14 cm diameter, 5.5 cm high).
bowl_large     0.280 kg  Base + 96 wall sectors (17 cm diameter, 7 cm high).

All selected-object geoms currently use MuJoCo friction = 0.7, 0.005, 0.0001
(sliding, torsional, rolling) and condim=4. The rolling coefficient is stored but
is inactive at condim=4. Masses and friction are engineering priors, not measured
material properties. Inertia is an additive collider-volume approximation,
normalized to the configured mass; overlapping parts introduce approximation.
Edit objects.json then run scripts/build_physics.py to regenerate physics.
Changing shape/pose also requires updating/rebuilding the authored visual geometry;
the physics configuration alone is not a generic visual asset generator.

PHYSICS ACCEPTANCE RESULTS
validation.json is the machine-readable evidence. It records:
- 8/8: ten-second countertop settling, supported and low velocity.
- 8/8: 18 cm free drop to floor, supported and stable after three seconds.
- 2/2: a free 9 mm-radius ball falls through each bowl opening and settles inside.
- 8/8: contact-only adjustable parallel-pad lift/hold/release checks.
- Predecessor hashes, 16 mechanism joint definitions and 393 contacts unchanged.

These are individual deterministic diagnostic trials, not a randomized benchmark.
The diagnostic gripper uses three actuated translation joints and two bounded
finger actuators. It closes, lifts about 14 cm, holds for 0.8 seconds, lowers,
opens, withdraws and lets the object settle. Objects receive forces only through
contact; they are never welded or attached to the gripper. Actual peak lift was
13.6-15.1 cm. Both finger pads were in contact at the end of the hold. Grasp force
is limited to 3 N per pad for plates and 12 N per pad for other objects.

This is NOT G1/Dex3 validation, a learned-policy rollout or successful MimicGen
deployment. The diagnostic gripper changes aperture to suit each object. In
particular the carton width may require a different Dex3 strategy or two hands.
Robot reach, finger collisions and grasp feasibility need a separate integration.

Video evidence: eight grasp checks, four representative drop checks and two bowl
interior checks. MP4s replay saved simulated joint states at 1280 x 720, 30 fps;
stills are native MuJoCo at 1920 x 1080. No original video is used as a background.
The HTML's WebGL model is for geometric inspection, not a physics simulator.
The predecessor report's native Unreal/Isaac renders remain available but predate
these free objects; new renderer packs must be exported before using this revision.

REBUILD / CHECK
1. Blender 4.5.4 --background --python scripts/build_objects.py
2. python scripts/build_physics.py
3. python scripts/validate_objects.py
   The source-preservation comparison expects the predecessor at the recorded
   path. On a standalone machine use --skip-source-comparison; results then
   explicitly record that source comparison was not performed.
4. MUJOCO_GL=egl python scripts/render_objects.py
5. python scripts/export_viewer.py
6. python scripts/build_report.py

The bundle contains authored meshes/textures, the editable kitchen_objects.blend,
object metadata, scripts and recorded diagnostic states. Generated test XMLs use
relative ../assets paths in the portable bundle. Validation creates fresh local
test XMLs when run again. No robot models or model weights are bundled.

LIMITATIONS / NEXT INTEGRATION
Plates are rigid approximations of paper plates; no bending. Bottles are sealed
rigid assemblies; no cap articulation, squeezing or liquid simulation. Cartons
are closed rigid boxes. Bowls do not break. Material parameters need measurement
for quantitative real-world contact fidelity. No validated robot task reset,
success gates, teleoperation mappings or MimicGen phase annotations are supplied
by this object-readiness revision.
