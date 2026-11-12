"""Put one of the procedural plates (data/kitchen_plates/plate_NN) onto a kitchen plate body of an MjSpec (2026-10-26).

swap_plate(spec, body_name, plate_dir, meta, prefix="") replaces the body's geoms with the plate's visual mesh (textured) and its
convex collision wedges + foot cylinder, sets the body's inertial to the plate's mass/inertia and lifts the body so the plate's
foot rests where the old plate's bottom was.  Mesh/texture/material names get a unique prefix so several plates can live in
one model.  Used by the kitchen kit (per-episode plate) and by the plate checks.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import mujoco

COLL = dict(contype=1, conaffinity=1, group=3, condim=4, friction=[0.7, 0.005, 0.0001], solref=[0.006, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0], margin=0.0001)


def load_meta(plate_dir: Path) -> dict:
    return json.loads((Path(plate_dir) / "meta.json").read_text())


FEET_R, FEET_LIFT = 0.0045, 0.0015                                       # sphere feet (as the kitchen bottles'): radius, and how far the flat foot cylinder is lifted off the counter


def swap_plate(spec: mujoco.MjSpec, body_name: str, plate_dir: Path, meta: dict | None = None, tag: str | None = None, keep_z: bool = False, feet: int | None = None):
    """feet: this many small spheres under the foot ring carry the plate instead of the flat foot cylinder's face (None = the plate's
    meta 'foot_spheres', 0 as before). 2026-11-04: a flat foot on the worktop rattles in place under SIMPLE's solver (40-70 deg/s of
    angular jitter, no net motion), which sets loose balls on the plate creeping; on sphere feet it is still. Same resting height."""
    plate_dir = Path(plate_dir).resolve(); meta = meta or load_meta(plate_dir); tag = tag or f"{body_name}_{meta['id']}"     # absolute: specs with a meshdir would prefix a relative path
    feet = int(meta.get("foot_spheres", 0)) if feet is None else int(feet)
    body = spec.body(body_name)
    for g in list(body.geoms):
        spec.delete(g)
    # assets: absolute file paths (the kitchen spec's meshdir points elsewhere)
    tex = spec.add_texture(name=f"{tag}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(plate_dir / "texture.png"))
    mat = spec.add_material(name=f"{tag}_mat", specular=0.4, shininess=0.6, reflectance=0.05); mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{tag}_tex"
    vis = spec.add_mesh(name=f"{tag}_vis", file=str(plate_dir / "visual.obj"))
    body.add_geom(name=f"{tag}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=vis.name, material=mat.name, contype=0, conaffinity=0, group=1, mass=0)
    for k in range(int(meta["n_coll"])):
        mesh = spec.add_mesh(name=f"{tag}_c{k:02d}", file=str(plate_dir / f"coll_{k:02d}.obj"), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        body.add_geom(name=f"{tag}_coll_{k:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, rgba=[0, 0, 0, 0], mass=0, **COLL)
    fc = meta["foot_cyl"]
    if feet > 0:
        h = fc["h"] - FEET_LIFT                                          # the cylinder keeps its top (the well floor), its face lifted off the counter
        body.add_geom(name=f"{tag}_foot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[fc["r"], h / 2, 0], pos=[0, 0, FEET_LIFT + h / 2], rgba=[0, 0, 0, 0], mass=0, **COLL)
        rf = float(meta["r_foot"]) + 0.5 * float(meta["foot_w"])
        for k in range(feet):
            ang = 2 * math.pi * (k + 0.5) / feet
            body.add_geom(name=f"{tag}_feet{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[FEET_R, 0, 0], pos=[rf * math.cos(ang), rf * math.sin(ang), FEET_R], rgba=[0, 0, 0, 0], mass=0, **COLL)
    else:
        body.add_geom(name=f"{tag}_foot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[fc["r"], fc["h"] / 2, 0], pos=[0, 0, fc["h"] / 2], rgba=[0, 0, 0, 0], mass=0, **COLL)
    I = meta["inertia"]; body.mass = float(meta["mass_kg"]); body.ipos = [0, 0, float(meta["com_z"])]; body.iquat = [1, 0, 0, 0]
    body.fullinertia = [I["ixx"], I["iyy"], I["izz"], 0.0, 0.0, 0.0]; body.inertia = [0.0, 0.0, 0.0]; body.explicitinertial = True   # the kitchen bodies carry fullinertia: set that form (both set = compile error)
    if not keep_z:                                                   # the authored paper plates sit with their bottom 1 mm under the body origin; ours start at z = 0
        body.pos = [float(body.pos[0]), float(body.pos[1]), float(body.pos[2]) - 0.001 + 0.0015]
    return tag
