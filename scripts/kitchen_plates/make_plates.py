#!/usr/bin/env python
"""50 procedural dinner plates for the kitchen scene (2026-10-26, user: "make 50 different plate meshes, graspable / pinchable by
the BrainCo hands, fitting the dishwasher slots; a new plate type every teleop rollout").

Each plate is a solid of revolution (optionally squircle / oval in plan) with a foot ring, a shallow well, a sloped wall and a
rim lip, generated from a random but bounded parameter set:
  diameter 170-270 mm, overall height 20-34 mm (stands on edge in the lower rack's slots), rim lip 12-35 mm wide,
  rim thickness 4-9 mm (the Revo 2 pinch closes to a 3 mm pad gap and opens to 106 mm), wall 4.5-8 mm, foot ring at
  40-58 % of the radius and 4-8 mm high (the rim underside clears the counter by >= 10 mm at the edge, so a fingertip can
  get under it), mass 60-450 g (paper / melamine / stoneware / porcelain densities, clamped).
Outputs data/kitchen_plates/plate_NN/: visual.obj (with planar UVs), texture.png, coll_*.obj (convex wedges: 24 x 2 radial
rings + a foot cylinder in meta), meta.json; and data/kitchen_plates/manifest.json.
    .venv/bin/python scripts/kitchen_plates/make_plates.py [--n 50] [--seed 7] [--out data/kitchen_plates]

`--family pasta` (2026-11-06, from the user's reference photo of a white wide-rim pasta plate): a deep bowl-shaped well (20-30 mm
below the rim, a flat floor >= 50 mm across for the indicator balls, a superellipse wall that steepens into the rim corner) on a
foot ring under the well's centre (22-30 % of the radius), a short near-vertical wall and a wide flat rim (38-49 % of the radius,
43-65 mm) with a rounded edge; round plan, diameter 224-265 mm, height 33-46 mm, mostly plain whites. The rim underside sits
30-40 mm above the counter, so both the fingertips and the thumb slide under it. Three collision rings per sector (lower bowl,
upper bowl, rim) keep the well hollow. `--id-prefix pasta` names them pasta_NN; `--foot-spheres 4` writes the sphere feet into
the metas (as the lift set carries them). The dinner family (default) is unchanged.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw

N_ANG = 96                                              # angular resolution of the visual mesh
N_SECT = 24                                             # convex collision wedges per ring
DENS = {"paper": 320.0, "melamine": 1450.0, "stoneware": 2250.0, "porcelain": 2400.0, "glass": 2500.0}
MASS_MIN, MASS_MAX = 0.06, 0.40                          # a real porcelain dinner plate is 500-800 g; the Revo 2 pinch holds less, so walls are thin and heavy plates are clamped


def profile(p):
    """Closed (r, z) polyline of the plate's cross-section, from the inner floor centre around the outside and back under.
    Returns the outer-top points first (centre -> rim edge), then the underside (rim edge -> foot -> centre).
    Both families give 22 top points (centre, well edge, 8 wall, 5 lip, 7 edge) and 15 underside points (rim edge, lip inner,
    8 wall, foot outer top, foot outer bottom, foot inner bottom, recess edge, centre), so the hull rings index the same."""
    if p.get("family") == "pasta":
        return profile_pasta(p)
    R, H, w, t_rim, t, r_foot, h_foot, foot_w, well_dz, edge = (p[k] for k in ("R", "H", "w", "t_rim", "t", "r_foot", "h_foot", "foot_w", "well_dz", "edge"))
    r_well = r_foot + p["well_margin"]                  # the flat floor's edge
    z_floor = h_foot + t                                # inner floor height (above the recessed bottom plate)
    z_rim = H - p["rim_rise"]                           # the rim lip's top reaches H at the outer edge (H = overall height)
    r_rim_in = R - w
    top = [(0.0, z_floor), (r_well, z_floor + well_dz)]
    # wall: a smooth curve from the well edge up to the rim's inner edge (ease-in-out)
    for i in range(1, 9):
        s = i / 9.0; e = s * s * (3 - 2 * s)
        top.append((r_well + (r_rim_in - r_well) * s, z_floor + well_dz + (z_rim - z_floor - well_dz) * e))
    # rim lip: from the inner edge to the outer edge, rising by rim_slope
    for i in range(0, 5):
        s = i / 4.0; top.append((r_rim_in + (R - edge - r_rim_in) * s, z_rim + p["rim_rise"] * s))
    # rounded outer edge (half circle of radius edge, centred at (R - edge, z_rim + rim_rise - t_rim/2 ... )
    zc = z_rim + p["rim_rise"] - t_rim / 2
    for i in range(1, 8):
        a = math.pi / 2 - math.pi * i / 8; top.append((R - edge + edge * math.cos(a), zc + (t_rim / 2) * math.sin(a)))
    under = [(R - edge, zc - t_rim / 2)]                # underside of the rim lip, back to the wall
    under.append((r_rim_in + p["lip_under"], zc - t_rim / 2 - p["under_drop"]))
    # under the wall, offset by t below the top curve
    for i in range(8, 0, -1):
        s = i / 9.0; e = s * s * (3 - 2 * s)
        under.append((r_well + (r_rim_in - r_well) * s, z_floor + well_dz + (z_rim - z_floor - well_dz) * e - t))
    under.append((r_foot + foot_w, h_foot))             # outside of the foot ring, top
    under.append((r_foot + foot_w, 0.0))                # foot bottom outer
    under.append((r_foot, 0.0))                         # foot bottom inner
    under.append((r_foot, p["recess"]))                 # recessed bottom
    under.append((0.0, p["recess"]))
    return top, under


def _bowl(s, k):
    """The pasta well's wall as a superellipse quadrant in angle form: (radial fraction, depth fraction) and the tangent at
    angle s * 90 deg. k = 2 is a circle; larger k flattens the floor and steepens the wall into the rim corner."""
    phi = s * math.pi / 2; e = 2.0 / k
    r = math.sin(phi) ** e; z = 1.0 - math.cos(phi) ** e
    dr = e * math.sin(phi) ** (e - 1) * math.cos(phi); dz = e * math.cos(phi) ** (e - 1) * math.sin(phi)
    return r, z, dr, dz


def profile_pasta(p):
    """The wide-rim pasta plate (2026-11-06): the flat floor at the well's centre, the bowl wall up to the rim corner, the flat
    lip out to the rounded edge; under: the lip's underside, the bowl's outer face (the wall offset by its thickness along the
    normal, so the near-vertical part keeps its thickness), the foot ring under the well, the recessed bottom."""
    R, H, w, t_rim, t, r_foot, h_foot, foot_w, edge, k = (p[k] for k in ("R", "H", "w", "t_rim", "t", "r_foot", "h_foot", "foot_w", "edge", "bowl_k"))
    r_well = r_foot + p["well_margin"]                  # the flat floor's edge (>= 40 mm across: the two indicator balls sit on it)
    z_floor = h_foot + t
    z_rim = H - p["rim_rise"]                           # the rim's inner corner; the lip rises rim_rise to the outer edge
    r_rim_in = R - w; depth = z_rim - z_floor; dr_w = r_rim_in - r_well
    top = [(0.0, z_floor), (r_well, z_floor)]
    for i in range(1, 9):
        fr, fz, _, _ = _bowl(i / 9.0, k); top.append((r_well + dr_w * fr, z_floor + depth * fz))
    for i in range(0, 5):
        s = i / 4.0; top.append((r_rim_in + (R - edge - r_rim_in) * s, z_rim + p["rim_rise"] * s))
    zc = z_rim + p["rim_rise"] - t_rim / 2
    for i in range(1, 8):
        a = math.pi / 2 - math.pi * i / 8; top.append((R - edge + edge * math.cos(a), zc + (t_rim / 2) * math.sin(a)))
    under = [(R - edge, zc - t_rim / 2), (r_rim_in + p["lip_under"], zc - t_rim / 2 - p["under_drop"])]   # lip_under > t: outside the wall's outer face
    for i in range(8, 0, -1):
        fr, fz, dfr, dfz = _bowl(i / 9.0, k); tr, tz = dr_w * dfr, depth * dfz        # tangent (outward, up)
        n = np.array([tz, -tr]); n /= np.linalg.norm(n)                                  # outward-down normal
        under.append((r_well + dr_w * fr + t * n[0], z_floor + depth * fz + t * n[1]))
    under.append((r_foot + foot_w, h_foot))             # outside of the foot ring, top
    under.append((r_foot + foot_w, 0.0))
    under.append((r_foot, 0.0))
    under.append((r_foot, p["recess"]))
    under.append((0.0, p["recess"]))
    return top, under


def shape_scale(theta, p):
    if p["plan"] == "round":
        return np.ones_like(theta)
    if p["plan"] == "squircle":
        n = p["plan_n"]; return 1.0 / (np.abs(np.cos(theta)) ** n + np.abs(np.sin(theta)) ** n) ** (1.0 / n) * p["plan_norm"]
    a, b = 1.0, p["plan_b"]                            # oval: x semi-axis 1, y semi-axis b
    return a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)


def revolve(prof, p, uv=True):
    """Triangle mesh of the closed profile revolved with the plan-shape radial scale; planar UVs for the texture."""
    th = np.linspace(0, 2 * np.pi, N_ANG, endpoint=False); s = shape_scale(th, p)
    V, UV, F = [], [], []
    ring_idx = []
    for (r, z) in prof:
        if r < 1e-9:
            V.append([0, 0, z]); UV.append([0.5, 0.5]); ring_idx.append(("pole", len(V) - 1))
        else:
            base = len(V)
            for k in range(N_ANG):
                x, y = s[k] * r * np.cos(th[k]), s[k] * r * np.sin(th[k]); V.append([x, y, z]); UV.append([0.5 + x / (2.2 * p["R"]), 0.5 + y / (2.2 * p["R"])])
            ring_idx.append(("ring", base))
    for i in range(len(ring_idx) - 1):
        (ta, a), (tb, b) = ring_idx[i], ring_idx[i + 1]
        for k in range(N_ANG):
            k2 = (k + 1) % N_ANG
            if ta == "pole" and tb == "ring": F.append([a, b + k, b + k2])
            elif ta == "ring" and tb == "pole": F.append([a + k, b, a + k2])
            elif ta == "ring" and tb == "ring": F += [[a + k, b + k, b + k2], [a + k, b + k2, a + k2]]
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    if m.volume < 0: m.invert()
    return m, np.array(UV)


def wedge_hulls(top, under, p):
    """Convex collision pieces: for each of N_SECT sectors, two hulls (wall ring, rim ring) from the outer AND inner surface
    points at the sector's bounding and middle angles; plus a central foot/well cylinder (returned as a size)."""
    th_edges = np.linspace(0, 2 * np.pi, N_SECT + 1)
    prof_rim = top[10:] + under[:3]                     # rim lip top, rounded edge, rim underside
    if p.get("family") == "pasta":                      # the deep well in two rings (a single hull would fill the bowl to its chord)
        rings = (top[1:6] + under[6:11], top[5:11] + under[2:7], prof_rim)   # lower bowl (floor edge .. 40 deg + foot top), upper bowl (40 deg .. rim corner), rim
    else:
        rings = (top[1:11] + under[2:11], prof_rim)     # well edge -> rim inner (top) and the corresponding underside; the rim
    pieces = []
    for k in range(N_SECT):
        angs = np.array([th_edges[k], 0.5 * (th_edges[k] + th_edges[k + 1]), th_edges[k + 1]]); s = shape_scale(angs, p)
        for prof in rings:
            pts = []
            for (r, z) in prof:
                for a, sc in zip(angs, s): pts.append([sc * r * np.cos(a), sc * r * np.sin(a), z])
            pieces.append(trimesh.convex.convex_hull(np.array(pts)))
    r_cyl = p["r_foot"] + p["well_margin"] + 0.004; h_cyl = p["h_foot"] + p["t"] + p["well_dz"]
    return pieces, dict(r=r_cyl, h=h_cyl)


def texture(p, rng, path):
    W = 512; im = Image.new("RGB", (W, W), tuple(int(255 * c) for c in p["base_rgb"])); dr = ImageDraw.Draw(im)
    c = W / 2; Rpx = W / 2.2 * (p["R"] / p["R"])         # plate edge radius in px: UV maps x/(2.2R) -> edge at 0.5 + 1/2.2
    Rpx = W * (1 / 2.2) * 1.0; acc = tuple(int(255 * v) for v in p["accent_rgb"])
    style = p["decor"]
    if style == "rim_band":
        for wdt, frac in ((p["band_px"], 1.0), (max(2, p["band_px"] // 3), 0.88)):
            r = Rpx * (1 - p["band_off"]) * frac; dr.ellipse([c - r, c - r, c + r, c + r], outline=acc, width=wdt)
    elif style == "dots":
        n = rng.integers(40, 140)
        for _ in range(n):
            rr = Rpx * math.sqrt(rng.uniform(0.02, 0.92)); a = rng.uniform(0, 2 * math.pi); d = rng.uniform(4, 11)
            dr.ellipse([c + rr * math.cos(a) - d, c + rr * math.sin(a) - d, c + rr * math.cos(a) + d, c + rr * math.sin(a) + d], fill=acc)
    elif style == "radial":
        n = int(rng.integers(12, 40))
        for i in range(n):
            a = 2 * math.pi * i / n; r0, r1 = Rpx * p["band_off"] * 3, Rpx * 0.97
            dr.line([c + r0 * math.cos(a), c + r0 * math.sin(a), c + r1 * math.cos(a), c + r1 * math.sin(a)], fill=acc, width=int(rng.integers(2, 6)))
    elif style == "speckle":
        for _ in range(3000):
            rr = Rpx * math.sqrt(rng.uniform(0, 0.98)); a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0.8, 2.2)
            dr.ellipse([c + rr * math.cos(a) - d, c + rr * math.sin(a) - d, c + rr * math.cos(a) + d, c + rr * math.sin(a) + d], fill=acc)
    elif style == "rings":
        for k in range(int(rng.integers(3, 8))):
            r = Rpx * rng.uniform(0.15, 0.95); dr.ellipse([c - r, c - r, c + r, c + r], outline=acc, width=int(rng.integers(2, 7)))
    elif style == "floral":
        for _ in range(int(rng.integers(6, 16))):
            rr = Rpx * math.sqrt(rng.uniform(0.3, 0.9)); a = rng.uniform(0, 2 * math.pi); x0, y0 = c + rr * math.cos(a), c + rr * math.sin(a)
            for j in range(6):
                b = a + j * math.pi / 3; d = rng.uniform(9, 16); dr.ellipse([x0 + 10 * math.cos(b) - d / 2, y0 + 10 * math.sin(b) - d / 2, x0 + 10 * math.cos(b) + d / 2, y0 + 10 * math.sin(b) + d / 2], fill=acc)
            dr.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill=tuple(min(255, v + 60) for v in acc))
    im.save(path)


PALETTE = [((0.96, 0.96, 0.94), (0.15, 0.35, 0.65)), ((0.98, 0.97, 0.92), (0.75, 0.2, 0.2)), ((0.93, 0.95, 0.97), (0.1, 0.5, 0.45)), ((0.97, 0.93, 0.88), (0.55, 0.3, 0.1)),
           ((0.86, 0.92, 0.86), (0.1, 0.4, 0.2)), ((0.92, 0.88, 0.95), (0.4, 0.2, 0.6)), ((0.2, 0.22, 0.25), (0.85, 0.85, 0.8)), ((0.55, 0.7, 0.8), (0.95, 0.95, 0.95)),
           ((0.95, 0.85, 0.75), (0.8, 0.45, 0.2)), ((0.99, 0.99, 0.99), (0.2, 0.2, 0.2)), ((0.85, 0.3, 0.25), (0.98, 0.95, 0.9)), ((0.3, 0.45, 0.35), (0.9, 0.9, 0.8))]


def sample(i, rng, clear_min=0.016, lip_min=0.012):
    plan = rng.choice(["round", "round", "round", "squircle", "oval"]); n = float(rng.uniform(3.2, 6.0)); b = float(rng.uniform(1.06, 1.18))
    R = rng.uniform(0.088, 0.130 if plan != "oval" else 0.135 / b); t_rim = rng.uniform(0.004, 0.008)
    # rim underside >= clear_min off the counter at the edge, lip >= lip_min wide. v1 (2026-10-26): 16 mm / 12 mm. 2026-11-04 (user: the BrainCo
    # fingers must get under the rim to lift the plate flat; measured with check_under_rim.py: the index distal link is 15 mm thick, the
    # thumb's 19 mm): --min-clearance-mm 24 --min-lip-mm 18 -> data/kitchen_plates_lift, both links slide >= 15 mm under every plate
    H = rng.uniform(t_rim + clear_min, max(0.034, t_rim + clear_min + 0.008)); w = rng.uniform(lip_min, max(lip_min + 0.004, min(0.035, 0.3 * R)))
    t = rng.uniform(0.004, 0.006); r_foot = R * rng.uniform(0.40, 0.58); h_foot = rng.uniform(0.004, 0.008); foot_w = rng.uniform(0.004, 0.007)
    mat = rng.choice(["porcelain", "stoneware", "melamine", "paper", "glass"], p=[0.30, 0.20, 0.35, 0.10, 0.05])
    base, acc = PALETTE[int(rng.integers(len(PALETTE)))]
    p = dict(id=f"plate_{i:02d}", R=R, H=H, w=w, t_rim=t_rim, t=t, r_foot=r_foot, h_foot=h_foot, foot_w=foot_w, well_dz=rng.uniform(0.0, 0.003), well_margin=rng.uniform(0.004, 0.02),
             edge=t_rim / 2, rim_rise=rng.uniform(0.0, 0.004), lip_under=rng.uniform(0.0, 0.004), under_drop=rng.uniform(0.0, 0.003), recess=rng.uniform(0.001, 0.003),
             plan=str(plan), plan_n=n, plan_norm=1.0, plan_b=b, material=str(mat), base_rgb=list(base), accent_rgb=list(acc),
             decor=str(rng.choice(["plain", "rim_band", "rim_band", "dots", "radial", "speckle", "rings", "floral"])), band_px=int(rng.integers(6, 22)), band_off=float(rng.uniform(0.02, 0.10)))
    if p["plan"] == "squircle":                        # keep the mean radius ~R so the diameter stays in range
        th = np.linspace(0, 2 * np.pi, 720); s = 1.0 / (np.abs(np.cos(th)) ** n + np.abs(np.sin(th)) ** n) ** (1.0 / n); p["plan_norm"] = float(1.0 / s.mean())
    return p


PASTA_BASES = [(0.99, 0.99, 0.99), (0.98, 0.98, 0.96), (0.97, 0.97, 0.95), (0.98, 0.97, 0.92), (0.96, 0.96, 0.94), (0.95, 0.96, 0.97)]   # whites, as the reference plate
PASTA_ACCENTS = [(0.15, 0.35, 0.65), (0.75, 0.2, 0.2), (0.1, 0.5, 0.45), (0.2, 0.2, 0.2), (0.55, 0.3, 0.1), (0.6, 0.6, 0.62), (0.8, 0.65, 0.3)]


def sample_pasta(i, rng, prefix="pasta", clear_min=0.024, lip_min=0.018):
    """One wide-rim pasta plate (the reference photo: well about half the diameter, rim a quarter, foot a third; white)."""
    R = float(rng.uniform(0.112, 0.1325)); w = R * float(rng.uniform(0.38, 0.49)); t_rim = float(rng.uniform(0.0045, 0.0070)); t = float(rng.uniform(0.0040, 0.0055))
    r_foot = R * float(rng.uniform(0.22, 0.30)); h_foot = float(rng.uniform(0.006, 0.009)); foot_w = float(rng.uniform(0.005, 0.008))
    depth = float(rng.uniform(0.020, 0.030)); rim_rise = float(rng.uniform(0.0, 0.002)); H = h_foot + t + depth + rim_rise
    assert H - t_rim >= clear_min and w >= lip_min, (H, t_rim, w)
    mat = rng.choice(["porcelain", "stoneware", "melamine"], p=[0.5, 0.3, 0.2])
    base = PASTA_BASES[int(rng.integers(len(PASTA_BASES)))]; acc = PASTA_ACCENTS[int(rng.integers(len(PASTA_ACCENTS)))]
    return dict(id=f"{prefix}_{i:02d}", family="pasta", R=R, H=H, w=w, t_rim=t_rim, t=t, r_foot=r_foot, h_foot=h_foot, foot_w=foot_w, well_dz=0.0, well_margin=float(rng.uniform(0.002, 0.008)),
                edge=t_rim / 2, rim_rise=rim_rise, lip_under=t + float(rng.uniform(0.001, 0.003)), under_drop=float(rng.uniform(0.0, 0.002)), recess=float(rng.uniform(0.001, 0.003)),
                bowl_k=float(rng.uniform(2.0, 2.8)), well_depth=depth, plan="round", plan_n=4.0, plan_norm=1.0, plan_b=1.0, material=str(mat), base_rgb=list(base), accent_rgb=list(acc),
                decor=str(rng.choice(["plain", "rim_band", "rings"], p=[0.6, 0.25, 0.15])), band_px=int(rng.integers(4, 14)), band_off=float(rng.uniform(0.02, 0.08)))


def build(p, out: Path, foot_spheres: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    top, under = profile(p); prof = top + under
    m, UV = revolve(prof, p)
    vol = float(m.volume); mass = min(MASS_MAX, max(MASS_MIN, vol * DENS[p["material"]]))
    # OBJ with UVs
    with open(out / "visual.obj", "w") as f:
        f.write(f"# {p['id']} procedural plate\n")
        for v in m.vertices: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in UV: f.write(f"vt {uv[0]:.5f} {uv[1]:.5f}\n")
        for tri in m.faces: f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")
    pieces, cyl = wedge_hulls(top, under, p)
    for k, h in enumerate(pieces): h.export(out / f"coll_{k:02d}.obj")
    texture(p, np.random.default_rng(abs(hash(p["id"])) % 2 ** 32), out / "texture.png")
    ext = m.bounds[1] - m.bounds[0]
    meta = dict(p, n_coll=len(pieces), foot_cyl=cyl, mass_kg=round(mass, 4), volume_cm3=round(vol * 1e6, 1), extents_mm=[round(float(v) * 1000, 1) for v in ext],
                diameter_mm=round(float(max(ext[0], ext[1])) * 1000, 1), height_mm=round(float(ext[2]) * 1000, 1), rim_thickness_mm=round(p["t_rim"] * 1000, 1),
                rim_underside_clearance_mm=round((p["H"] - p["t_rim"]) * 1000, 1), mass_raw_kg=round(vol * DENS[p["material"]], 4), rim_lip_mm=round(p["w"] * 1000, 1), faces=int(len(m.faces)),
                inertia=dict(ixx=round(0.25 * mass * p["R"] ** 2 + mass * p["H"] ** 2 / 12, 7), iyy=round(0.25 * mass * p["R"] ** 2 + mass * p["H"] ** 2 / 12, 7), izz=round(0.5 * mass * p["R"] ** 2, 7)),
                com_z=round(float(m.center_mass[2]), 5), watertight=bool(m.is_watertight))
    if foot_spheres:
        meta["foot_spheres"] = int(foot_spheres)
    if p.get("family") == "pasta":
        meta["well_depth_mm"] = round(p["well_depth"] * 1000, 1); meta["well_floor_mm"] = round((p["r_foot"] + p["well_margin"]) * 2000, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1)); return meta


MANIFEST_KEYS = ("id", "diameter_mm", "height_mm", "rim_thickness_mm", "rim_lip_mm", "rim_underside_clearance_mm", "mass_kg", "material", "plan", "decor", "n_coll", "foot_cyl", "inertia", "base_rgb", "accent_rgb")
MANIFEST_OPT = ("family", "well_depth_mm", "well_floor_mm", "foot_spheres")


def manifest_entry(m: dict) -> dict:
    return dict([(k, m[k]) for k in MANIFEST_KEYS] + [(k, m[k]) for k in MANIFEST_OPT if k in m])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", default="data/kitchen_plates")
    ap.add_argument("--min-clearance-mm", type=float, default=16.0, help="rim underside above the counter at the edge, at least (v1: 16; the lift set: 24)")
    ap.add_argument("--min-lip-mm", type=float, default=12.0, help="rim lip width, at least (v1: 12; the lift set: 18)"); ap.add_argument("--made", default="2026-10-26")
    ap.add_argument("--family", default="dinner", choices=("dinner", "pasta"), help="dinner (v1 / lift: the shallow plate) or pasta (2026-11-06: deep well, wide rim)")
    ap.add_argument("--id-prefix", default=None, help="plate ids <prefix>_NN (default plate, pasta for --family pasta)"); ap.add_argument("--id-start", type=int, default=0)
    ap.add_argument("--foot-spheres", type=int, default=0, help="write this many sphere feet into the metas (the lift set carries 4; plate_swap reads it)")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); out = Path(a.out); metas = []
    prefix = a.id_prefix or ("pasta" if a.family == "pasta" else "plate")
    for i in range(a.id_start, a.id_start + a.n):
        if a.family == "pasta":
            p = sample_pasta(i, rng, prefix=prefix, clear_min=a.min_clearance_mm / 1000, lip_min=a.min_lip_mm / 1000)
        else:
            p = sample(i, rng, clear_min=a.min_clearance_mm / 1000, lip_min=a.min_lip_mm / 1000); p["id"] = f"{prefix}_{i:02d}"
        meta = build(p, out / p["id"], foot_spheres=a.foot_spheres); metas.append(meta)
        print(f"{p['id']}: D {meta['diameter_mm']:.0f} mm H {meta['height_mm']:.1f} mm rim {meta['rim_thickness_mm']:.1f} mm lip {meta['rim_lip_mm']:.0f} mm {p['plan']:8s} {p['material']:9s} {p['decor']:8s} mass {meta['mass_kg']*1000:.0f} g faces {meta['faces']} watertight {meta['watertight']}")
    man = dict(made=a.made, seed=a.seed, n=a.n, min_clearance_mm=a.min_clearance_mm, min_lip_mm=a.min_lip_mm, plates=[manifest_entry(m) for m in metas])
    if a.family != "dinner":
        man["family"] = a.family
    if a.foot_spheres:
        man["foot_spheres"] = int(a.foot_spheres)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"-> {out}/manifest.json")


if __name__ == "__main__":
    main()
