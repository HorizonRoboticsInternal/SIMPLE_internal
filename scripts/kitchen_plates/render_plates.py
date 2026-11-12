#!/usr/bin/env python
"""Thumbnails of procedural plates (2026-11-06): each plate on a neutral slab, lit, seen from a 3/4 view like a product photo,
plus a cross-section drawing from its meta (the generator's profile).  Used by the plate-set pages.
    MUJOCO_GL=egl .venv/bin/python scripts/kitchen_plates/render_plates.py --plates data/kitchen_plates_v2 --out docs/kitchen_plates_v2/img [--ids pasta_00,pasta_03] [--size 480x360]
Writes <out>/<id>.jpg (the render) and <out>/<id>_section.svg (the section, 1 px = 0.5 mm, the rim lip / well / foot labelled).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_swap import swap_plate, load_meta
from make_plates import profile

ap = argparse.ArgumentParser(); ap.add_argument("--plates", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids", default="")
ap.add_argument("--size", default="480x360"); ap.add_argument("--elevation", type=float, default=-32.0); ap.add_argument("--azimuth", type=float, default=135.0)
a = ap.parse_args(); W, H = (int(v) for v in a.size.split("x")); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
man = json.load(open(Path(a.plates) / "manifest.json")); ids = set(a.ids.split(",")) if a.ids else None


def render(pdir: Path, meta: dict, path: Path):
    spec = mujoco.MjSpec(); spec.visual.global_.offwidth = W; spec.visual.global_.offheight = H
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]; spec.visual.headlight.diffuse = [0.18, 0.18, 0.18]; spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.worldbody.add_light(pos=[0.4, -0.3, 0.9], dir=[-0.4, 0.3, -0.9], diffuse=[0.45, 0.45, 0.45], specular=[0.25, 0.25, 0.25], castshadow=True)
    spec.worldbody.add_light(pos=[-0.5, 0.4, 0.6], dir=[0.5, -0.4, -0.6], diffuse=[0.22, 0.22, 0.22], specular=[0.0, 0.0, 0.0], castshadow=False)
    tex = spec.add_texture(name="slab", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[0.82, 0.80, 0.77], width=8, height=8)
    mat = spec.add_material(name="slab", reflectance=0.0, specular=0.1, shininess=0.1); mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "slab"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.6, 0.01], pos=[0, 0, -0.01], material="slab")
    pb = spec.worldbody.add_body(name="plate", pos=[0, 0, 0]); pb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0, group=4)
    swap_plate(spec, "plate", pdir, meta, keep_z=True)
    m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, meta["H"] * 0.4]; cam.distance = meta["R"] * 3.6; cam.azimuth = a.azimuth; cam.elevation = a.elevation
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[0] = 1; opt.geomgroup[1] = 1
    r.update_scene(d, camera=cam, scene_option=opt); Image.fromarray(r.render()).save(path, quality=88); r.close()


def section_svg(meta: dict, path: Path):
    """The closed profile mirrored about the axis, 1 px = 0.5 mm; well depth and rim width marked."""
    top, under = profile(meta); prof = top + under; S = 2000.0
    R = meta["R"]; Hm = meta["H"]; pad = 14; w = int(2 * R * S) + 2 * pad; h = int(Hm * S) + 2 * pad + 18
    def P(r, z): return f"{pad + (R + r) * S:.1f},{pad + (Hm - z) * S:.1f}"
    def Q(r, z): return f"{pad + (R - r) * S:.1f},{pad + (Hm - z) * S:.1f}"
    right = " ".join(P(r, z) for r, z in prof); left = " ".join(Q(r, z) for r, z in reversed(prof))
    marks = []
    if meta.get("family") == "pasta":
        r_in = R - meta["w"]; zf = meta["h_foot"] + meta["t"]; zr = Hm - meta["rim_rise"]
        marks.append(f'<line x1="{pad + (R + r_in) * S:.1f}" y1="{pad:.1f}" x2="{pad + (R + R) * S:.1f}" y2="{pad:.1f}" class="m"/>')
        marks.append(f'<text x="{pad + (R + r_in + meta["w"] / 2) * S:.1f}" y="{pad - 3:.1f}" class="t" text-anchor="middle">rim {meta["w"] * 1000:.0f} mm</text>')
        marks.append(f'<line x1="{pad + R * S:.1f}" y1="{pad + (Hm - zf) * S:.1f}" x2="{pad + R * S:.1f}" y2="{pad + (Hm - zr) * S:.1f}" class="m"/>')
        marks.append(f'<text x="{pad + R * S + 4:.1f}" y="{pad + (Hm - (zf + zr) / 2) * S + 4:.1f}" class="t">well {(zr - zf) * 1000:.0f} mm</text>')
    marks.append(f'<text x="{pad:.1f}" y="{h - 4:.1f}" class="t">D {meta["diameter_mm"]:.0f} mm · H {meta["height_mm"]:.0f} mm · foot {meta["r_foot"] * 2000:.0f} mm</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
           f'<style>.p{{fill:var(--plate,#eceae4);stroke:var(--ink,#333);stroke-width:1.2;stroke-linejoin:round}}.m{{stroke:var(--accent,#b5542a);stroke-width:1}}.t{{font:10px system-ui,sans-serif;fill:var(--ink,#333)}}</style>'
           f'<polygon class="p" points="{left} {right}"/>{"".join(marks)}</svg>')
    path.write_text(svg)


for P in man["plates"]:
    if ids and P["id"] not in ids:
        continue
    pdir = Path(a.plates) / P["id"]; meta = load_meta(pdir)
    render(pdir, meta, out / f"{P['id']}.jpg"); section_svg(meta, out / f"{P['id']}_section.svg")
    print(P["id"], flush=True)
print(f"-> {out}")
