"""Revo 2 hands: low-friction self-contact (2026-10-27) -> g1_rot0_teleop_revo2_v2.xml.

    .venv/bin/python scripts/robots/add_revo2_hand_selfcontact.py [--friction 0.2]

User, 2026-10-27: "sometimes after the hand closes with an object and places it somewhere, the hand keeps the close pose;
after I release the close button the hand should always go back to the open pose".  Cause (kitchen4 episode 2, 143-155 s,
replayed bit-exactly): closing the grip on little or nothing drives the opposed thumb tip into the gap between the index
and ring fingertips; the parts then press on each other at 50-75 N, and with friction 1.0 (and SIMPLE's impratio 10 +
2 noslip iterations) the fingers are frictionally locked: every opening command, in any order, left all joints where they
were for 3 s.  With hand-on-hand friction 0.5 or lower the same state opens in 0.7 s.

This writes a new robot file next to the old one (the old one stays byte-identical, so recordings made with it keep
replaying bit-exactly): every hand-on-hand contact that can lock -- thumb links against the four fingers, neighbouring
fingers against each other, the palm against the fingertips and the thumb tip -- becomes an explicit <pair> with
friction FRICTION (the matching body pairs are <exclude>d so MuJoCo does not also generate them with the geoms'
friction). Hand-on-object contacts keep the geoms' friction 1.0, so grasping is unchanged.  The pairs carry the contact
stiffness SIMPLE's engine gives every hand geom (MujocoSimulator: solref 0.005 1, solimp 0.9 0.99 0.0001), because
explicit pairs do not take the geoms' values.  Unnamed colliding hand geoms get names (<body>_collision).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2.xml"
OUT = ROOT / "data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_v2.xml"
SOLREF, SOLIMP = "0.005 1", "0.9 0.99 0.0001 0.5 2"            # = MujocoSimulator's hand-geom override
FINGERS = ("index", "middle", "ring", "pinky")


def links(side: str, digit: str, parts=("proximal", "distal", "tip", "touch")) -> list[str]:
    return [f"{side}_hand_{digit}_{p}_link" for p in parts]


def body_pairs(side: str) -> list[tuple[str, str]]:
    thumb = links(side, "thumb")
    finger = {f: links(side, f) for f in FINGERS}
    pairs = [(a, b) for a in thumb for f in FINGERS for b in finger[f]]                         # thumb vs the fingers
    for f1, f2 in zip(FINGERS[:-1], FINGERS[1:]):                                               # neighbouring fingers
        pairs += list(itertools.product(finger[f1], finger[f2]))
    palm = f"{side}_hand_base_link"                                                              # palm vs fingertips, thumb tip
    pairs += [(palm, b) for f in FINGERS for b in links(side, f, ("distal", "tip", "touch"))]
    pairs += [(palm, b) for b in links(side, "thumb", ("tip", "touch"))]
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--friction", type=float, default=0.2); ap.add_argument("--src", default=str(SRC)); ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    m = mujoco.MjModel.from_xml_path(str(src))
    tree = ET.parse(src); root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    # name the colliding geoms of every hand body (XML child order = compiled order within a body)
    geom_names: dict[str, list[str]] = {}
    for bname, bel in bodies.items():
        if not bname or "_hand_" not in bname:
            continue
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        compiled = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        xml_geoms = [e for e in bel if e.tag == "geom"]
        assert len(compiled) == len(xml_geoms), (bname, len(compiled), len(xml_geoms))
        names = []
        for k, (g, e) in enumerate(zip(compiled, xml_geoms)):
            if not (m.geom_contype[g] | m.geom_conaffinity[g]):
                continue
            if not e.get("name"):
                e.set("name", f"{bname}_collision" + ("" if k == 0 else f"{k}"))
            names.append(e.get("name"))
        geom_names[bname] = names
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    fr = f"{a.friction:g} {a.friction:g} 0.005 0.0001 0.0001"
    contact.append(ET.Comment(f" hand self-contact: low friction ({a.friction:g}) so the fingers cannot lock on the thumb or on each "
                              f"other (scripts/robots/add_revo2_hand_selfcontact.py, 2026-10-27) "))
    n_ex = n_pair = 0
    for side in ("left", "right"):
        for b1, b2 in body_pairs(side):
            g1s, g2s = geom_names.get(b1, []), geom_names.get(b2, [])
            if not g1s or not g2s:
                continue
            ET.SubElement(contact, "exclude", body1=b1, body2=b2); n_ex += 1
            for g1 in g1s:
                for g2 in g2s:
                    ET.SubElement(contact, "pair", geom1=g1, geom2=g2, condim="3", friction=fr, solref=SOLREF, solimp=SOLIMP); n_pair += 1
    root.insert(0, ET.Comment(f" {out.name}: {src.name if src.resolve() != out.resolve() else 'the same file before this step'} (sha256 {hashlib.sha256(src.read_bytes()).hexdigest()[:16]}) + low-friction hand "
                              f"self-contact, generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/robots/add_revo2_hand_selfcontact.py "))
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=False)
    m2 = mujoco.MjModel.from_xml_path(str(out))
    same = (m2.nbody, m2.njnt, m2.ngeom, m2.nu, m2.nsensor, m2.ncam) == (m.nbody, m.njnt, m.ngeom, m.nu, m.nsensor, m.ncam)
    print(f"[selfcontact] {out}: {n_ex} body excludes, {n_pair} pairs at friction {a.friction:g}; npair {m2.npair}, nexclude {m2.nexclude}; "
          f"bodies/joints/geoms/actuators/sensors/cameras unchanged: {same}")


if __name__ == "__main__":
    main()
