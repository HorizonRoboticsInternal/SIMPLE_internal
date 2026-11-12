#!/usr/bin/env python
"""RoboAtom-style slicing of a recorded kitchen teleop episode WITH lower-body skills (user 2026-11-03: "we should consider the
lowbody key skills, for example A4 should be step not press ... create a new full ep roboatom slicing for kitchen demo").

RoboAtom (arXiv 2610.40207) labels one segment per dominant skill from a hand-skill vocabulary and says "Move means physical
robot-base/chassis translation only"; it has no foot or posture skills (its embodiments are arms on wheeled bases). For a legged
humanoid we keep its hand classes for the upper body and add a lower-body lane: Move (theirs), Turn, Step, Squat; the whole-body
label of a segment is the manipulation when the hands do more than hold, otherwise the lower-body skill.

Input: the episode's annotation scan folder (events.npz / events.json from episode_events.py, traj.json from episode_annotate.py)
and the page's caption.json. Output: <out>/segments.json (RoboAtom segment format, two lanes + whole-body), timeline.png, path.png.

    .venv/bin/python scripts/holomotion_v14_checks/roboatom_slice_episode.py --scan docs/_scan/kitchen7_ep0_annotation \
        --page docs/kitchen7_ep0_annotation --out docs/_scan/roboatom_k7ep0
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle

ap = argparse.ArgumentParser()
ap.add_argument("--scan", default="docs/_scan/kitchen7_ep0_annotation"); ap.add_argument("--page", default="docs/kitchen7_ep0_annotation")
ap.add_argument("--out", default="docs/_scan/roboatom_k7ep0")
ap.add_argument("--move-net", type=float, default=0.30, help="a base translation run is a Move when its net displacement is at least this (m); smaller = a stance adjustment absorbed into the neighbour")
ap.add_argument("--turn-deg", type=float, default=45.0, help="a rotation in place (net < move-net) is a Turn from this angle on, unless it leads into a walk")
ap.add_argument("--squat-drop", type=float, default=0.06, help="a Squat is the pelvis this far below its standing height (m) for >= 0.5 s")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[2]
SCAN, PAGE, OUT = ROOT / a.scan, ROOT / a.page, ROOT / a.out; OUT.mkdir(parents=True, exist_ok=True)

z = np.load(SCAN / "events.npz"); E = json.load(open(SCAN / "events.json")); TR = json.load(open(PAGE / "traj.json")); CAP = json.load(open(PAGE / "caption.json"))
FPS = int(E["fps"]); T = int(E["frames"]); t = np.arange(T) / FPS
base, yaw, vxy, J = z["base"], z["yaw"], z["vxy"], z["joints"]
ji = {n: i for i, n in enumerate(E["joint_names"])}
door, rack, lid, pedal = J[:, ji["k_dishwasher_door"]], J[:, ji["k_dishwasher_upper_rack"]], J[:, ji["k_bin_lid"]], J[:, ji["k_bin_pedal"]]
yawu = np.unwrap(yaw)
k = FPS // 2; ker = np.ones(k) / k
v_s = np.convolve(vxy, ker, "same"); w_s = np.degrees(np.convolve(np.gradient(yawu) * FPS, ker, "same")); z_s = np.convolve(base[:, 2], ker, "same")
touch = E["touch"]
foot_bin = np.array([("pedal bin" in tt["body"]) for tt in touch])
Z_STAND = float(np.median(base[:, 2]))
PLATE_NAMES = {"plate A": None, "plate B": None}
MAIN = CAP["config"]["Main Task"]
# plate names as the annotation calls them ("green-ringed plate")
for seg in CAP["segments"]:
    for ac in seg["Current Action"]:
        for p in PLATE_NAMES:
            pass
names_txt = " ".join(ac["label"] for seg in CAP["segments"] for ac in seg["Current Action"])
import re
cands = re.findall(r"(?:Reach for|Carry|Lift) the ([a-z\- ]+plate)", names_txt)
seen = []
for c in cands:
    if c not in seen: seen.append(c)
PLATE_TXT = {"plate A": seen[0] if seen else "plate A", "plate B": seen[1] if len(seen) > 1 else "plate B"}


def runs(mask, gap_s=1.0, min_s=0.5):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = []; s = idx[0]; p = idx[0]
    for i in idx[1:]:
        if i - p > gap_s * FPS:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return [(int(x), int(y)) for x, y in out if (y - x) / FPS >= min_s]


# ------------------------------------------------------------------ lower-body phases from the ground truth
walks = []
for s, e in runs(v_s > 0.15, gap_s=1.0, min_s=0.5):
    net = float(np.hypot(*(base[e, :2] - base[s, :2]))); path = float(vxy[s:e].sum() / FPS); turn = float(np.degrees(yawu[e] - yawu[s]))
    walks.append(dict(s=s, e=e, net=net, path=path, turn=turn, vmax=float(vxy[s:e].max())))
turns = []
for s, e in runs(np.abs(w_s) > 30, gap_s=0.6, min_s=0.4):
    net = float(np.hypot(*(base[e, :2] - base[s, :2]))); turns.append(dict(s=s, e=e, deg=float(np.degrees(yawu[e] - yawu[s])), net=net))
squats = [dict(s=s, e=e, zmin=float(base[s:e, 2].min())) for s, e in runs(z_s < Z_STAND - a.squat_drop, gap_s=0.5, min_s=0.5)]
presses = [dict(s=s, e=e, pedal=float(pedal[s:e].max()), lid_from=float(lid[s]), lid_to=float(lid[min(e + FPS, T - 1)])) for s, e in runs(foot_bin, gap_s=0.3, min_s=0.05)]
# the Step segment: from the first foot contact with the bin to the last frame the foot is on it (presses within 5 s of each other are one Step)
steps = []
for p in presses:
    if steps and p["s"] - steps[-1]["e"] < 5 * FPS:
        steps[-1]["e"] = p["e"]; steps[-1]["presses"].append(p)
    else:
        steps.append(dict(s=p["s"], e=p["e"], presses=[p]))
for st in steps:
    st["lid_after"] = float(lid[min(st["e"] + FPS, T - 1)]); st["n_press"] = sum(1 for p in st["presses"] if p["pedal"] > 0.05)

# ------------------------------------------------------------------ the annotation's actions and handling events
ACTS = [dict(label=x["label"], s=int(x["s"]), e=int(x["e"]), rel=x.get("rel", True)) for x in TR["acts"]]
H = TR["events"]["handling"]; DOOR = TR["events"]["door"]
hold = {}
for h in H:
    hold[h["dish"]] = dict(contact=h["s"], lift=h["lift"], release=h["release"], hands=h.get("hold_hands"), passes=h.get("passes", []), pick_hands=h.get("pick_hands"), place_hands=h.get("place_hands"), tries=h.get("tries", 0), setdowns=h.get("setdowns", 0), where=h["where"])


def act_between(s, e):
    """The annotation's actions that fall (mostly) inside a segment: more than half of the action, or more than half of the segment."""
    return [x for x in ACTS if x["s"] < e and x["e"] > s and (min(x["e"], e) - max(x["s"], s) > 0.5 * (x["e"] - x["s"]) or min(x["e"], e) - max(x["s"], s) > 0.5 * (e - s))]


def fmt(fr):
    return f"{fr / FPS:.1f}"


# ------------------------------------------------------------------ build the two lanes
# Upper lane: RoboAtom hand skills from the annotation's events (Pick = contact run that leads to the lift .. the lift done and the
# free hand off; Place = lowering .. release; Handover; Open / Close; Pull / Push), everything else in between is "hold" or nothing.
upper = []
for dish, h in hold.items():
    pick_acts = [x for x in ACTS if x["label"].startswith(("Reach for", "Grasp", "Lift")) and dish[-1] in x["label"][-40:] or False]
    # the Pick: from the first Reach of this dish to the end of its Lift action
    reach = [x for x in ACTS if x["label"].startswith("Reach for the") and PLATE_TXT[dish] in x["label"]]
    lift = [x for x in ACTS if x["label"].startswith("Lift the") and PLATE_TXT[dish] in x["label"]]
    ps = reach[0]["s"] if reach else h["contact"]; pe = lift[-1]["e"] if lift else h["lift"]
    retry = f"; the hands set it back down once and took it again" if h.get("setdowns") else (f"; one regrip" if h.get("tries") else "")
    upper.append(dict(skill="Pick", s=ps, e=pe, obj=dish, desc=f"Pick up the {PLATE_TXT[dish]} from the counter: both hands reach, pinch the rim and lift; the {('left' if (h.get('hands') or ['right'])[0] == 'right' else 'right')} hand lets go, the plate stays in the {(h.get('hands') or ['right'])[0]} hand{retry}"))
    for p in h.get("passes", []):
        upper.append(dict(skill="Handover", s=p["s"], e=p["e"], obj=dish, desc=f"Pass the {PLATE_TXT[dish]} from the {p['frm']} hand to the {p['to']} hand"))
    lower_acts = [x for x in ACTS if x["label"].startswith("Lower the") and PLATE_TXT[dish] in x["label"]]
    rel = [x for x in ACTS if x["label"].startswith("Release the") and PLATE_TXT[dish] in x["label"]]
    ls = lower_acts[0]["s"] if lower_acts else h["release"] - 2 * FPS; le = rel[-1]["e"] if rel else h["release"]
    upper.append(dict(skill="Place", s=ls, e=le, obj=dish, desc=f"Place the {PLATE_TXT[dish]} upright in the {h['where']} with the {(h.get('place_hands') or ['?'])[0]} hand and let go"))
for d in DOOR:
    if d["kind"] == "open_door":
        upper.append(dict(skill="Open", s=d["s"], e=d["e"], obj="dishwasher door", desc=f"Pull the dishwasher door open with the {d['hands'][0]} hand, {d['deg'][0]} to {d['deg'][1]} deg; the held plate stays in the other hand"))
    elif d["kind"] == "close_door":
        upper.append(dict(skill="Close", s=d["s"], e=d["e"], obj="dishwasher door", desc=f"Lift the dishwasher door shut with the {d['hands'][0]} hand, {d['deg'][0]} to {d['deg'][1]} deg"))
    elif d["kind"] == "pull_rack":
        rr = [x for x in ACTS if x["label"].startswith("Reach for the upper rack")]
        upper.append(dict(skill="Pull", s=rr[0]["s"] if rr else d["s"], e=d["e"], obj=d["rack"], desc=f"Pull the {d['rack']} out {d['m'][1] - d['m'][0]:.2f} m with the {d['hands'][0]} hand; the held plate stays in the other hand"))
    elif d["kind"] == "push_rack":
        upper.append(dict(skill="Push", s=d["s"], e=d["e"], obj=d["rack"], desc=f"Push the {d['rack']} back in with the {d['hands'][0]} hand, in {d.get('taps', 1)} pushes"))
# Wipe (RoboAtom hand class): the free hand's passes across the face of the held plate over the open bin, from the annotation's bin stop
for b in TR["events"].get("bin_stops", []):
    if b.get("swipes") and b.get("sw"):
        dish = b["dish"]; where = "over the open bin" if b.get("over_bin") else "beside the open bin"
        upper.append(dict(skill="Wipe", s=int(b["sw"][0]), e=int(b["sw"][1]), obj=dish,
                          desc=f"Wipe the {PLATE_TXT[dish]} clean {where} with the {b['other']} hand: {b['swipes']} passes across its face while the "
                               f"{' and '.join(b['hold'])} hand holds it tipped to {b['tilt'][1]} deg"))
upper.sort(key=lambda x: x["s"])
# a Reach that precedes an Open / Close / Pull is absorbed into it (RoboAtom rule: arm reaching belongs to the neighbouring manipulation)
for u in upper:
    if u["skill"] in ("Close", "Open"):
        rr = [x for x in ACTS if x["label"].startswith("Reach for the dishwasher door") and x["e"] <= u["s"] + 2 and x["s"] > u["s"] - 6 * FPS]
        if rr: u["s"] = rr[0]["s"]
# resolve overlaps in the upper lane (Place of plate B overlaps the Push by the annotation's 0.3 s): first come, first served
for i in range(1, len(upper)):
    if upper[i]["s"] < upper[i - 1]["e"]:
        upper[i]["s"] = upper[i - 1]["e"]
# held state per frame (for the figure and the whole-body tie-break)
holding = np.zeros(T, dtype=object); holding[:] = ""
for dish, h in hold.items():
    holding[h["lift"]:h["release"]] = dish

# Lower lane
lower = []
for w in walks:
    if w["net"] >= a.move_net:
        lower.append(dict(skill="Move", **w))
    else:
        lower.append(dict(skill="adjust", **w))
for st in steps:
    lower.append(dict(skill="Step", s=st["s"], e=st["e"], n_press=st["n_press"], lid_after=st["lid_after"], presses=st["presses"]))
for sq in squats:
    lower.append(dict(skill="Squat", s=sq["s"], e=sq["e"], zmin=sq["zmin"]))
# turns that stand alone (not inside / leading into a Move within 1 s) are Turn segments
moves = [x for x in lower if x["skill"] == "Move"]
for tu in turns:
    if abs(tu["deg"]) < a.turn_deg or tu["net"] >= a.move_net:
        continue
    if any(m["s"] - FPS <= tu["e"] and tu["s"] <= m["e"] + FPS for m in lower if m["skill"] in ("Move", "adjust")):
        continue                                          # the turn is part of a walk or of a stance shuffle, not a turn in place
    lower.append(dict(skill="Turn", s=tu["s"], e=tu["e"], deg=tu["deg"], net=tu["net"]))
lower.sort(key=lambda x: x["s"])
# a Move's consecutive runs separated by a short pause (< 3 s, no manipulation, still holding the same thing) are one Move (RoboAtom:
# "do not split tiny transitions"); a longer stop (ep0 stands 6 s at the bin with plate B) ends the Move: two base-motion phases
merged = []
for x in lower:
    if merged and x["skill"] == "Move" and merged[-1]["skill"] == "Move" and x["s"] - merged[-1]["e"] < 3 * FPS and not any(u["s"] < x["s"] and u["e"] > merged[-1]["e"] for u in upper) and holding[x["s"]] == holding[merged[-1]["e"]]:
        m = merged[-1]; m["e"] = x["e"]; m["path"] += x["path"]; m["net"] = float(np.hypot(*(base[m["e"], :2] - base[m["s"], :2]))); m["turn"] += x["turn"]; m["pauses"] = m.get("pauses", 0) + 1
    else:
        merged.append(dict(x))
lower = merged
# a walk that overlaps a Step is cut at the Step (the foot is the dominant thing); a shuffle inside a Step is part of it
for st in [x for x in lower if x["skill"] == "Step"]:
    for x in lower:
        if x["skill"] in ("Move", "adjust") and x["s"] < st["e"] and x["e"] > st["s"]:
            if x["s"] < st["s"]: x["e"] = st["s"]
            else: x["s"] = st["e"]
lower = [x for x in lower if x["e"] - x["s"] > 0.2 * FPS]
lower.sort(key=lambda x: x["s"])

# ------------------------------------------------------------------ whole-body sequence (one label per segment, RoboAtom style)
# boundaries = every upper-lane skill boundary + every Move / Step / Turn boundary of the lower lane (Squat and adjustments stay inside
# the manipulation they serve). Label: the upper-lane skill if one is active, else the lower-lane skill, else the gap is absorbed.
for u in [u for u in upper if u["skill"] == "Wipe"]:
    for st in [x for x in lower if x["skill"] == "Step"]:
        if st["s"] < u["e"] <= st["e"] and st["e"] - u["e"] <= 1.5 * FPS: u["e"] = st["e"]      # the foot leaves the pedal right after the last pass
cuts = sorted({0, T} | {u["s"] for u in upper} | {u["e"] for u in upper} | {x["s"] for x in lower if x["skill"] in ("Move", "Step", "Turn")} | {x["e"] for x in lower if x["skill"] in ("Move", "Step", "Turn")})
pieces = []
for s, e in zip(cuts[:-1], cuts[1:]):
    mid = (s + e) // 2
    up = next((u for u in upper if u["s"] <= mid < u["e"]), None)
    lo = next((x for x in lower if x["s"] <= mid < x["e"] and x["skill"] in ("Move", "Step", "Turn")), None)
    pieces.append(dict(s=s, e=e, up=up, lo=lo))
# merge pieces: an unlabeled piece (nothing but standing / holding) goes to the segment it leads into, the tail to the last one
# (RoboAtom: "absorb preparation, transitions and idle frames into the nearest meaningful action"); consecutive pieces of one
# source are one segment (a manipulation that starts inside a walk ends that Move; what is left of the walk after it is a new Move)
segments = []
gap_start = None
for p in pieces:
    if p["up"] is None and p["lo"] is None:
        if gap_start is None: gap_start = p["s"]
        continue
    src = p["up"] if p["up"] is not None else p["lo"]; lane = "upper" if p["up"] is not None else "lower"
    s = gap_start if gap_start is not None else p["s"]; gap_start = None
    if segments and segments[-1]["src"] is src:
        segments[-1]["e"] = p["e"]
    else:
        segments.append(dict(s=s, e=p["e"], label=src["skill"], src=src, lane=lane))
if gap_start is not None and segments:
    segments[-1]["e"] = T

# descriptions for the lower-lane segments
def lower_desc(x, holding_txt):
    if x["skill"] == "Move":
        where = ""
        return f"Walk {x['path']:.1f} m ({x['net']:.1f} m as the crow flies, {abs(x['turn']):.0f} deg turned){holding_txt}"
    if x["skill"] == "Step":
        return f"Step on the bin pedal: {x['n_press']} presses, the foot on it {(x['e'] - x['s']) / FPS:.1f} s; the lid ends at {x['lid_after']:.2f} rad{holding_txt}"
    if x["skill"] == "Turn":
        return f"Turn in place {x['deg']:+.0f} deg{holding_txt}"
    if x["skill"] == "Squat":
        return f"Squat: the pelvis down to {x['zmin']:.2f} m (standing {Z_STAND:.2f} m)"
    return "stance adjustment"


# where a Move goes: the place nearest to where it ends (the bin from the scene, the counter = where the picks touched the plates,
# the dishwasher = where the robot stood at the end of the door events)
BB = E["scene"]["bin_box"]; ANCH = {"the pedal bin": np.array([(BB["lo"][0] + BB["hi"][0]) / 2, (BB["lo"][1] + BB["hi"][1]) / 2])}
for d_, h_ in hold.items():
    ANCH.setdefault("the counter", base[h_["contact"], :2])
for d_ in DOOR:
    if d_["kind"] in ("open_door", "close_door"): ANCH.setdefault("the dishwasher", base[d_["e"], :2])


def destination(e):
    p = base[min(e, T - 1), :2]
    name, pt = min(ANCH.items(), key=lambda kv: float(np.hypot(*(kv[1] - p))))
    return f" to {name}" if float(np.hypot(*(pt - p))) < 0.9 else ""


# the whole-body sequence stops at the annotation's trim (the last task action); what follows is Idle (RoboAtom: kept with an empty skill)
T_KEEP = int(CAP["config"].get("trim", {}).get("kept_frames", T))
segments = [sg for sg in segments if sg["s"] < T_KEEP]
if segments and segments[-1]["e"] > T_KEEP:
    segments[-1]["e"] = T_KEEP
IDLE = [dict(start_s=round(T_KEEP / FPS, 2), end_s=round(T / FPS, 2), skill="", text=CAP["config"].get("trim", {}).get("rule", "after the last task action"))] if T_KEEP < T else []


SEG = []
for i, sg in enumerate(segments, 1):
    src = sg["src"]; s, e = sg["s"], sg["e"]
    held = holding[(s + e) // 2]
    held_txt = f", the {PLATE_TXT[held]} in a hand" if held else ""
    if sg["lane"] == "upper":
        desc = src["desc"]
    else:
        if src["skill"] == "Step":
            desc = lower_desc(dict(src, s=s, e=e), held_txt)
            if src["e"] > e + 0.5 * FPS:
                nxt = next((u for u in upper if u["s"] <= e < u["e"]), None)
                desc += f"; the foot stays on the pedal {(src['e'] - e) / FPS:.1f} s more inside the {nxt['skill'] if nxt else 'next segment'} that follows"
        elif src["skill"] == "Move":
            dest = destination(src["e"]); verb = "step up" if src["path"] < 0.6 else "walk"
            desc = (f"Carry the {PLATE_TXT[held]}{dest}: {verb}" if held else f"{verb.capitalize()}{dest}:") + f" {src['path']:.1f} m ({src['net']:.1f} m net, {abs(src['turn']):.0f} deg turned{', ' + str(src['pauses']) + ' pause' if src.get('pauses') else ''})"
        else:
            desc = lower_desc(src, held_txt)
    lo_in = [x for x in lower if min(x["e"], e) - max(x["s"], s) >= 0.5 * FPS and x["skill"] in ("Squat", "adjust", "Turn", "Step", "Move") and x is not src]
    SEG.append(dict(index=i, skill=src["skill"], lane=sg["lane"], start_frame=int(s) + 1, end_frame=int(e), start_s=round(s / FPS, 2), end_s=round(e / FPS, 2), seconds=round((e - s) / FPS, 2),
                    description=desc, object=src.get("obj") or ("bin pedal" if src["skill"] == "Step" else None), holding=held or None,
                    lower_body_inside=[dict(skill=x["skill"], start_s=round(x["s"] / FPS, 1), end_s=round(x["e"] / FPS, 1), **({"net_m": round(x["net"], 2)} if "net" in x else {}), **({"turn_deg": round(x["turn"])} if "turn" in x else {}), **({"deg": round(x["deg"])} if "deg" in x else {}), **({"zmin": round(x["zmin"], 2)} if "zmin" in x else {})) for x in lo_in],
                    annotation_actions=[x["label"] for x in act_between(s, e)]))

# ------------------------------------------------------------------ export
VOCAB = dict(upper_body=["Pick", "Place", "Handover", "Move", "Open", "Close", "Push", "Pull", "Press", "Rotate", "Screw", "Plug", "Pour", "Fill", "Scoop", "Stir", "Dip", "Shake", "Fold", "Tie", "Wipe", "Sweep", "Wash", "Brush", "Cut", "Peel", "Strike", "Scan"],
             lower_body={"Move": "base translation (RoboAtom's own class: 'Move means physical robot-base/chassis translation only'); a walk, with the turns that start or end it inside",
                         "Turn": f"a rotation in place of {a.turn_deg:.0f} deg or more with less than {a.move_net:.2f} m of translation that does not lead into a walk",
                         "Step": "a foot on an object (the bin pedal): from the first foot contact to the foot leaving, repeated presses inside",
                         "Squat": f"the pelvis {100 * a.squat_drop:.0f} cm or more below its standing height for 0.5 s or more; stays inside the manipulation it serves"},
             rules=["one segment per dominant skill; a new segment starts only when the dominant skill changes",
                    "arm reaching, lifting, lowering, alignment and retraction belong to the neighbouring manipulation (RoboAtom rule 5)",
                    f"a base translation under {a.move_net:.2f} m net is a stance adjustment and belongs to the neighbouring manipulation too",
                    "whole-body label = the hand skill when the hands do more than hold; otherwise the lower-body skill (a carry is a Move, a pedal press with the plate in hand is a Step)",
                    "a Squat and a stance adjustment never cut a hand skill; a Step cuts a carry or a hold (the foot is the dominant thing while the hands only hold), and a hand skill done with the foot on the pedal (a Wipe over the bin) labels the whole body with the Step inside it",
                    "retries stay inside the segment (RoboAtom's Recover is an intermediate label resolved to the skill performed)"])
export = dict(config=dict(source=str(PAGE.relative_to(ROOT)), fps=FPS, frames=T, frame_base=1, main_task=MAIN, standing_pelvis_z=round(Z_STAND, 3), thresholds=dict(move_net_m=a.move_net, turn_deg=a.turn_deg, squat_drop_m=a.squat_drop, walk_speed=0.15, turn_rate_deg_s=30), vocabulary=VOCAB),
              segments=SEG, idle=IDLE,
              lanes=dict(upper=[dict(skill=u["skill"], start_s=round(u["s"] / FPS, 2), end_s=round(u["e"] / FPS, 2), object=u.get("obj"), description=u["desc"]) for u in upper],
                         lower=[dict(skill=x["skill"], start_s=round(x["s"] / FPS, 2), end_s=round(x["e"] / FPS, 2), **{kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in x.items() if kk not in ("s", "e", "skill", "presses")}) for x in lower],
                         holding=[dict(dish=d, lift_s=round(h["lift"] / FPS, 2), release_s=round(h["release"] / FPS, 2)) for d, h in hold.items()]),
              lower_body_phases=dict(walks=[dict(start_s=round(w["s"] / FPS, 1), end_s=round(w["e"] / FPS, 1), net_m=round(w["net"], 2), path_m=round(w["path"], 2), turn_deg=round(w["turn"]), vmax=round(w["vmax"], 2)) for w in walks],
                                     turns=[dict(start_s=round(x["s"] / FPS, 1), end_s=round(x["e"] / FPS, 1), deg=round(x["deg"]), net_m=round(x["net"], 2)) for x in turns],
                                     squats=[dict(start_s=round(x["s"] / FPS, 1), end_s=round(x["e"] / FPS, 1), zmin=round(x["zmin"], 3)) for x in squats],
                                     presses=[dict(start_s=round(p["s"] / FPS, 1), end_s=round(p["e"] / FPS, 1), pedal_rad=round(p["pedal"], 3), lid_from=round(p["lid_from"], 2), lid_to=round(p["lid_to"], 2)) for p in presses]),
              annotation=dict(sub_tasks=len(TR["subs"]), actions=len([x for x in ACTS if x.get("rel", True)]), actions_all=len(ACTS)))
(OUT / "segments.json").write_text(json.dumps(export, indent=1))
print(f"{len(SEG)} whole-body segments, upper lane {len(upper)}, lower lane {len(lower)} ({sum(1 for x in lower if x['skill'] == 'adjust')} adjustments)")
for sg in SEG:
    print(f"  {sg['index']:2d} {sg['start_s']:6.1f}-{sg['end_s']:6.1f} {sg['skill']:9s} [{sg['lane'][0]}] {sg['description'][:110]}  | inside: {[(x['skill'], x['start_s']) for x in sg['lower_body_inside']]}")

# ------------------------------------------------------------------ figures
SC = {"Move": "#2e8b57", "Pick": "#1f6f5f", "Step": "#b5533c", "Wipe": "#8a5fbf", "Handover": "#0b3d91", "Open": "#c98a1b", "Pull": "#d9772a", "Place": "#6a3d9a",
      "Push": "#e0a84f", "Close": "#8c6d31", "Squat": "#c2185b", "Turn": "#3b8ea5", "adjust": "#aab2b8", "hold": "#dfe4e8"}
END = T / FPS


def bars(ax, items, y, h, key="skill", text=True, fs=7.5, alpha=1.0, hatch=None, label_min=1.5):
    for x in items:
        s, e = x["s"] / FPS, x["e"] / FPS; c = SC.get(x[key], "#999")
        ax.add_patch(Rectangle((s, y), e - s, h, fc=c, ec="w", lw=0.6, alpha=alpha, hatch=hatch))
        if text and e - s >= label_min:
            lab = x[key] if x[key] != "adjust" else ""
            ax.text((s + e) / 2, y + h / 2, lab, ha="center", va="center", fontsize=fs, color="w" if x[key] not in ("adjust", "hold", "Push") else "#333", fontweight="bold")


fig = plt.figure(figsize=(17, 11.5), dpi=110)
gs = fig.add_gridspec(9, 1, height_ratios=[1.25, 1.0, 1.0, 0.9, 0.7, 0.7, 0.7, 0.7, 0.6], hspace=0.12)
ax0 = fig.add_subplot(gs[0]); axes = [ax0] + [fig.add_subplot(gs[i], sharex=ax0) for i in range(1, 9)]
# 1 whole-body
ax = axes[0]
for sg in SEG:
    s, e = sg["start_s"], sg["end_s"]; c = SC[sg["skill"]]
    ax.add_patch(Rectangle((s, 0), e - s, 1, fc=c, ec="w", lw=0.8))
    if e - s >= 2.0:
        ax.text((s + e) / 2, 0.62, sg["skill"], ha="center", va="center", fontsize=8.5 if e - s >= 7 else 6.5, color="w" if sg["skill"] != "Push" else "#333", fontweight="bold")
        ax.text((s + e) / 2, 0.25, f"{sg['index']}", ha="center", va="center", fontsize=7, color="w" if sg["skill"] != "Push" else "#333")
for idl in IDLE:
    ax.add_patch(Rectangle((idl["start_s"], 0), idl["end_s"] - idl["start_s"], 1, fc="#d0d4d8", ec="w", lw=0.8, hatch="xx"))
ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("whole body\n(one label)", fontsize=8.5, rotation=0, ha="right", va="center")
ax.set_title(f"{PAGE.name.replace('_annotation', '').replace('_', ' ')}: RoboAtom slicing with the lower-body lane · {len(SEG)} segments over {END:.0f} s", fontsize=11, loc="left")
# 2 upper lane
ax = axes[1]
for d, h in hold.items():
    ax.add_patch(Rectangle((h["lift"] / FPS, 0.15), (h["release"] - h["lift"]) / FPS, 0.7, fc=SC["hold"], ec="none"))
    ax.text(h["lift"] / FPS + 1.0, 0.5, f"holding the {PLATE_TXT[d]} →", ha="left", va="center", fontsize=6.5, color="#555")
bars(ax, upper, 0, 1, fs=6.5)
ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("upper body\n(hand skills)", fontsize=8.5, rotation=0, ha="right", va="center")
# 3 lower lane
ax = axes[2]
bars(ax, [x for x in lower if x["skill"] in ("Move", "Step", "Turn")], 0, 1)
bars(ax, [x for x in lower if x["skill"] == "Squat"], 0, 1)
bars(ax, [x for x in lower if x["skill"] == "adjust"], 0.2, 0.6, hatch="////", alpha=0.9)
ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel("lower body\n(Move · Step · Squat)", fontsize=8.5, rotation=0, ha="right", va="center")
# 4 annotation actions
ax = axes[3]
for i, x in enumerate(ACTS):
    s, e = x["s"] / FPS, x["e"] / FPS
    ax.add_patch(Rectangle((s, 0.1 + 0.42 * (i % 2)), e - s, 0.38, fc="#9fb3c8" if x.get("rel", True) else "#ccc", ec="w", lw=0.5))
ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_ylabel(f"annotation\n{len(ACTS)} actions", fontsize=8.5, rotation=0, ha="right", va="center")
# 5 speed
ax = axes[4]; ax.fill_between(t, 0, vxy, color="#2e8b57", alpha=0.35, lw=0); ax.plot(t, v_s, color="#2e8b57", lw=0.9); ax.axhline(0.15, color="k", lw=0.5, ls=":")
ax.set_ylim(0, 0.95); ax.set_ylabel("base speed\n(m/s)", fontsize=8.5, rotation=0, ha="right", va="center"); ax.tick_params(labelsize=7)
# 6 yaw rate
ax = axes[5]; ax.fill_between(t, 0, np.abs(w_s), color="#3b8ea5", alpha=0.4, lw=0); ax.axhline(30, color="k", lw=0.5, ls=":")
ax.set_ylim(0, 150); ax.set_ylabel("|yaw rate|\n(deg/s)", fontsize=8.5, rotation=0, ha="right", va="center"); ax.tick_params(labelsize=7)
# 7 pelvis height
ax = axes[6]; ax.plot(t, base[:, 2], color="#c2185b", lw=0.9); ax.axhline(Z_STAND - a.squat_drop, color="k", lw=0.5, ls=":")
ax.set_ylim(0.5, 0.85); ax.set_ylabel("pelvis height\n(m)", fontsize=8.5, rotation=0, ha="right", va="center"); ax.tick_params(labelsize=7)
# 8 bin: pedal, lid, foot
ax = axes[7]; ax.plot(t, lid, color="#b5533c", lw=0.9, label="lid (rad)"); ax.plot(t, pedal * 5, color="#555", lw=0.8, label="pedal ×5 (rad)")
ax.fill_between(t, 0, foot_bin * 1.6, color="#b5533c", alpha=0.25, lw=0, step="mid", label="foot on the bin")
ax.set_ylim(0, 1.7); ax.legend(fontsize=6.5, loc="upper right", ncol=3, framealpha=0.8); ax.set_ylabel("pedal bin", fontsize=8.5, rotation=0, ha="right", va="center"); ax.tick_params(labelsize=7)
# 9 door / rack
ax = axes[8]; ax.plot(t, door, color="#c98a1b", lw=0.9, label="door (rad)"); ax.plot(t, rack, color="#d9772a", lw=0.9, label="upper rack (m)")
ax.set_ylim(0, 1.7); ax.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.8); ax.set_ylabel("dishwasher", fontsize=8.5, rotation=0, ha="right", va="center"); ax.tick_params(labelsize=7)
ax.set_xlabel("time (s)", fontsize=9)
for axx in axes:
    axx.set_xlim(0, END); axx.grid(True, axis="x", alpha=0.25)
    for sg in SEG:
        axx.axvline(sg["start_s"], color="k", lw=0.4, alpha=0.25)
for axx in axes[:-1]:
    plt.setp(axx.get_xticklabels(), visible=False)
handles = [Patch(fc=SC[k], label=k) for k in ("Move", "Pick", "Handover", "Open", "Pull", "Place", "Push", "Close", "Step", "Squat")] + [Patch(fc=SC["adjust"], hatch="////", label="stance adjustment (absorbed)"), Patch(fc=SC["hold"], label="holding a plate (state, not a skill)")]
fig.legend(handles=handles, loc="lower center", ncol=12, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.text(0.005, 0.005, f"Whole body: a hand skill wins while the hands do more than hold; otherwise the lower-body skill. Thresholds: walk = smoothed speed > 0.15 m/s, Move if net >= {a.move_net:.2f} m; Turn >= {a.turn_deg:.0f} deg in place; Squat = pelvis >= {100 * a.squat_drop:.0f} cm below {Z_STAND:.2f} m; Step = foot on the bin.", fontsize=6.5, color="#555")
fig.savefig(OUT / "timeline.png", bbox_inches="tight"); plt.close(fig)

# path coloured by the lower-body lane
M = TR["map"]; img = plt.imread(PAGE / "topview.jpg")
def uv(x, y):
    return (x - M["cx"]) / M["sx"] * M["W"] + M["W"] / 2, M["H"] / 2 - (y - M["cy"]) / M["sy"] * M["H"]
fig, ax = plt.subplots(figsize=(8.2, 9.2), dpi=110); ax.imshow(img); ax.axis("off")
u, v = uv(base[:, 0], base[:, 1])
ax.plot(u, v, color="#777", lw=1.0, alpha=0.6)
for x in lower:
    s, e = x["s"], x["e"]; c = SC[x["skill"]]
    if x["skill"] == "Move":
        ax.plot(u[s:e], v[s:e], color=c, lw=3.2, solid_capstyle="round", alpha=0.95)
    elif x["skill"] == "adjust":
        ax.plot(u[s:e], v[s:e], color="#666", lw=2.2, alpha=0.8)
# the whole-body segment numbers at the midpoint of each walk (the table has the rest), Step / Squat ringed
offs = {}
for i, sg in enumerate(segments, 1):
    src = sg["src"]
    if sg["lane"] == "lower" and src["skill"] == "Move":
        m = (src["s"] + src["e"]) // 2
        ax.annotate(str(i), (u[m], v[m]), ha="center", va="center", fontsize=8, color="w", fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc=SC["Move"], ec="w", lw=1.2))
for x in lower:
    s, e = x["s"], x["e"]; c = SC[x["skill"]]
    if x["skill"] in ("Step", "Squat", "Turn"):
        i = (s + e) // 2; ax.add_patch(Circle((u[i], v[i]), 16, fc="none", ec=c, lw=2.6))
        ax.annotate(f"{x['skill']} {s / FPS:.0f}–{e / FPS:.0f} s", (u[i], v[i]), (16, 4 if x["skill"] == "Step" else -18), textcoords="offset points", fontsize=7.5, color=c, fontweight="bold", bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.2))
for d, h in hold.items():
    for key, lab in (("lift", "Pick"), ("release", "Place")):
        i = h[key]; ax.plot(u[i], v[i], marker="o", ms=7, color=SC[lab], mec="w", mew=1)
ax.plot(u[0], v[0], marker="s", ms=8, color="k"); ax.annotate("start", (u[0], v[0]), (6, -12), textcoords="offset points", fontsize=7.5)
ax.legend(handles=[plt.Line2D([], [], color=SC["Move"], lw=3, marker="o", ms=9, mfc=SC["Move"], mec="w", label="Move (walk), numbered as in the table"), plt.Line2D([], [], color="#666", lw=2.2, label="stance adjustment, absorbed"), plt.Line2D([], [], color=SC["Step"], lw=2.6, marker="o", mfc="none", ms=9, ls="", label="Step (foot on the pedal)"),
                   plt.Line2D([], [], color=SC["Squat"], lw=2.6, marker="o", mfc="none", ms=9, ls="", label="Squat (inside the Close)"), plt.Line2D([], [], color=SC["Pick"], marker="o", ls="", label="Pick done (plate lifted)"), plt.Line2D([], [], color=SC["Place"], marker="o", ls="", label="Place done (released)")],
          loc="lower left", fontsize=7.5, framealpha=0.9)
ax.set_title("The pelvis path, coloured by the lower-body lane", fontsize=10, loc="left")
fig.savefig(OUT / "path.png", bbox_inches="tight", pil_kwargs=dict(quality=90)); plt.close(fig)
print("figures ->", OUT)
