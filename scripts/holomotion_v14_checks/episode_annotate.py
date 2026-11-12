"""Segment / caption / sub-task annotation of a recorded kitchen dishwasher episode, in the Horizon caption.json format (2026-10-27).

    .venv/bin/python scripts/holomotion_v14_checks/episode_annotate.py <events.npz> <level dir> <episode> <out dir> \
        [--names '{"plate A": "white plate", "plate B": "green-rimmed plate"}']

User, 2026-10-27: "for this ep, please do a horizon_ego_0324_trimmed like segment and caption and subtask and task related
slicing and annotation". The Horizon set (http://10.40.11.11:8899/horizon_ego_0324_trimmed/) was captioned by Gemini from
video; here every boundary and state comes from the simulation itself (episode_events.py: the bit-exact replay's contacts,
joint angles, poses and camera visibility), so frame ranges are exact, and the sentences are written from those events.

Writes <out dir>/caption.json (the Horizon layout: config with Main Task and trim; 5 s segments with Sub Task, Current Action
[label, description, task_relevant, start_frame, end_frame, in_main_task], 2 fps sampled frames with objects, interaction
and hands [visible, holding, object, grasp_state, same_object_as_other_hand]; global_summary; video_caption), metadata.json
(task labels, scene, duration, trim), slices.json (the task-related slices: one per sub task, contiguous, with frame ranges
and seconds) and traj.json (top-view base path coloured by sub task, for the page). Frames are 1-based as in the Horizon
files: frame n is dataset frame_index n-1 (50 fps).
2026-10-28, user: a carry that stops at the pedal bin is four sub tasks, not one: carry to the bin, step on the pedal to open
the lid (first foot press -> lid >= 1.3 rad), swipe the dish clean over the bin (-> the robot walks on; swipes = passes of the
free hand's palm in toward the dish and back, >= 10 cm), carry on to the dishwasher. Needs scene.bin_box from episode_events.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 50
SEG = 250                      # 5 s segments
SAMPLE = 25                    # 2 samples per second
DISHES = ("plate A", "plate B", "small bowl", "large bowl")
PARTS_NICE = {"dishwasher door": "the dishwasher door", "upper rack": "the upper rack", "lower rack": "the lower rack", "left sink": "the sink",
              "right sink": "the sink", "counter": "the counter", "pedal bin": "the pedal bin"}


def runs(mask, min_len=1, gap=0):
    """[(start, end)) runs of True, merging runs separated by <= gap frames, keeping runs >= min_len frames."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b - 1 <= gap:
            b = i
        else:
            out.append((int(a), int(b) + 1)); a = b = i
    out.append((int(a), int(b) + 1))
    return [(a, b) for a, b in out if b - a >= min_len]


def settle(sig, i, thr, hold=FPS):
    """First frame >= i after which |d sig/dt| stays below thr for `hold` frames."""
    v = np.abs(np.gradient(sig)) * FPS
    for k in range(i, len(sig) - hold):
        if v[k:k + hold].max() < thr:
            return k
    return len(sig) - 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("events"); ap.add_argument("level_dir"); ap.add_argument("episode", type=int); ap.add_argument("out")
    ap.add_argument("--names", default="{}", help="JSON: friendlier names for props in sentences, e.g. {\"plate A\": \"white plate\"}")
    a = ap.parse_args()
    E = np.load(a.events); M = json.load(open(str(Path(a.events).with_suffix("")) + ".json")); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    nice = json.loads(a.names); NM = lambda n: nice.get(n, n)                  # display name ("green-rimmed plate")
    THE = lambda n: "the " + NM(n) if not NM(n).startswith("the ") else NM(n)
    T = M["frames"]; touch = M["touch"]; J = {n: i for i, n in enumerate(M["joint_names"])}
    base, vxy, yaw = E["base"], E["vxy"], E["yaw"]
    door = E["joints"][:, J["k_dishwasher_door"]]
    rack = {"upper rack": E["joints"][:, J["k_dishwasher_upper_rack"]], "lower rack": E["joints"][:, J["k_dishwasher_lower_rack"]]}
    pedal = E["joints"][:, J["k_bin_pedal"]] if "k_bin_pedal" in J else np.zeros(T); lid = E["joints"][:, J["k_bin_lid"]] if "k_bin_lid" in J else np.zeros(T)
    props = M["props"]; P = E["prop_pos"]; pidx = {n: i for i, n in enumerate(props)}
    dishes = [n for n in DISHES if n in pidx]; z0 = {n: float(P[0, pidx[n], 2]) for n in dishes}
    pspeed = np.vstack([np.zeros((1, len(props))), np.linalg.norm(np.diff(P, axis=0), axis=2) * FPS])
    has = lambda side, name: np.array([name in touch[i][side] for i in range(T)])
    anyhand = lambda name: has("left", name) | has("right", name)
    df = pd.read_parquet(Path(a.level_dir) / f"data/chunk-000/episode_{a.episode:06d}.parquet", columns=[f"action.revo2.{s}.grip_mode" for s in ("left", "right")])
    grip = {s: np.stack(df[f"action.revo2.{s}.grip_mode"].values).ravel().astype(int) for s in ("left", "right")}
    RB = M["scene"]["rack_box"]

    def in_rack(n, i):
        p = P[i, pidx[n]]
        for k, rn in enumerate(M["racks"]):
            loc = E["rack_mat"][i, k].T @ (p - E["rack_pos"][i, k])
            if abs(loc[0]) <= RB["x"] and abs(loc[1]) <= RB["y"] and RB["z"][0] <= loc[2] <= RB["z"][1]:
                return rn
        return None

    walk = runs(vxy > 0.15, min_len=25, gap=50)
    walking = np.zeros(T, bool)
    for s, e in walk: walking[s:e] = True
    def sides_on(name, s, e):                                                  # hands that really took part: >= 0.5 s of contact, else the busiest
        n = {h: int(has(h, name)[s:e].sum()) for h in ("left", "right")}
        hs = [h for h in ("left", "right") if n[h] >= 25]
        return hs or ([max(n, key=n.get)] if max(n.values()) else [])
    hands_txt = lambda hs: "both hands" if len(hs) == 2 else f"the {hs[0]} hand" if hs else "the robot"
    vb = lambda hs, sing, plur: plur if len(hs) == 2 else sing                # verb agreement with hands_txt
    times = lambda n: "once" if n == 1 else "twice" if n == 2 else f"{n} times"
    feet_on_pedal = np.array(["pedal bin" in touch[i]["body"] for i in range(T)])
    dw_xy = E["rack_pos"][0, 0, :2]
    def leg(s, e):                                                              # metres walked from s to e, and whether that is a detour
        path = float(np.sum(np.linalg.norm(np.diff(base[s:e, :2], axis=0), axis=1))) if e > s + 1 else 0.0
        straight = float(np.linalg.norm(base[e - 1, :2] - base[s, :2])) if e > s else 0.0
        return path, path > 2 * max(straight, 0.5) and path - straight > 2.0   # twice the straight line and 2 m extra

    # ---- primitive events ---------------------------------------------------------------------------------------------------------
    ev = []                                                                    # core task events, each a dict(kind, s, e, ...)
    for s, e in runs(anyhand("dishwasher door"), min_len=5, gap=150):
        delta = float(door[min(e + 150, T - 1)] - door[s])
        if abs(delta) < 0.3:
            continue
        end = settle(door, e, 0.03)
        ev.append(dict(kind="open_door" if delta > 0 else "close_door", s=s, e=max(end, e), hands=sides_on("dishwasher door", s, e),
                       deg=(round(math.degrees(door[s])), round(math.degrees(door[max(end, e)])))))
    for rn, q in rack.items():
        mv = runs(np.abs(np.gradient(q)) * FPS > 0.02, min_len=3, gap=200)
        for s, e in mv:
            delta = float(q[e - 1] - q[s]); hs = sides_on(rn, max(0, s - 25), min(T, e + 25))
            if abs(delta) < 0.05 or not hs:
                continue                                                       # nudges from a plate, or pushed by the closing door
            c = np.flatnonzero(anyhand(rn)[max(0, s - 25):min(T, e + 25)])
            ev.append(dict(kind="pull_rack" if delta > 0 else "push_rack", s=int(max(0, s - 25) + c[0]), e=e, rack=rn, hands=hs,
                           m=(round(float(q[s]), 2), round(float(q[e - 1]), 2)), taps=len(runs(anyhand(rn)[s:e], gap=5))))
    # ---- handling legs (2026-10-30, kitchen6 ep1): pick -> carry -> set down / place, one leg per rest with the hands off. A dish is
    # "off its rest" when it is 3 cm above or below, or 10 cm aside of, where it last rested (a plate carried hanging low never rose
    # 3 cm above the counter), a rest is 1 s with no hand on it and the dish still, and where it rests names the set-down ------------
    BB = M["scene"].get("bin_box")
    bin_c = np.array([(BB["lo"][0] + BB["hi"][0]) / 2, (BB["lo"][1] + BB["hi"][1]) / 2]) if BB else None
    cnt_xy0 = np.mean([P[0, pidx[n], :2] for n in dishes], axis=0)
    still = pspeed < 0.02

    def place_of(n, i):
        """Where a dish rests at frame i: a rack, the lid of the pedal bin, the floor, the counter or the dishwasher."""
        r = in_rack(n, i)
        if r:
            return r
        q = P[i, pidx[n]]
        if BB and BB["lo"][0] - 0.05 <= q[0] <= BB["hi"][0] + 0.05 and BB["lo"][1] - 0.05 <= q[1] <= BB["hi"][1] + 0.05 and abs(q[2] - BB["hi"][2]) < 0.10:
            return "lid of the pedal bin"
        if q[2] < 0.15:
            return "floor"
        if abs(q[2] - z0[n]) < 0.03:
            return "counter"
        return "dishwasher" if np.linalg.norm(q[:2] - dw_xy) < np.linalg.norm(q[:2] - cnt_xy0) else "counter"

    def near_bin(xy_):
        return bin_c is not None and float(np.linalg.norm(np.asarray(xy_)[:2] - bin_c)) < 0.6

    handling = []
    for n in dishes:
        hc = anyhand(n); zi = P[:, pidx[n], 2]; xy = P[:, pidx[n], :2]
        rest_i, start, prev_place, legs = 0, 0, "counter", 0
        while start < T:
            tr = runs(hc[start:], min_len=5, gap=50)                           # hands on the dish (a regrip within 1 s is the same leg)
            if not tr:
                break
            s = start + tr[0][0]
            off = (np.abs(zi - zi[rest_i]) > 0.03) | (np.linalg.norm(xy - xy[rest_i], axis=1) > 0.10)
            offr = [(x, y) for x, y in runs(off, min_len=12, gap=10) if y > s and x >= s - 10 and hc[max(0, x - 10):x + 10].any()
                    and not (in_rack(n, x) and any(abs(rack[rn][min(T - 1, x + 5)] - rack[rn][max(0, x - 5)]) > 0.01 for rn in rack))]   # moved by a hand, not by the sliding rack
            if not offr:                                                        # touched, never moved: look past this contact run
                start = start + tr[0][1]
                continue
            end = next((x for x, y in runs((~hc) & still[:, pidx[n]], min_len=50, gap=0) if x >= offr[0][0]), T)   # the first rest with the hands off
            final = [r_ for r_ in offr if r_[0] < end] or offr[:1]
            lift = final[-1][0]                                                 # the move that ends at this rest; earlier ones were set back down
            cand = [x for x, y in tr if start + x <= lift + 10]                 # the pick is the contact run that leads to this move (kitchen7 ep1: a
            s = start + cand[-1] if cand else s                                 # 0.2 s brush of plate B two minutes before its pick is not its pick)
            setdowns = max(0, len([r_ for r_ in final if r_[0] >= s - 10]) - 1)
            wk = [w for w in walk if lift <= w[0] < end]
            wk0 = wk[0][0] if wk else lift
            soon = {h: np.convolve(has(h, n).astype(int), np.ones(26, int))[25:25 + T] > 0 for h in ("left", "right")}   # touching now or within 0.5 s
            cl = {h: (grip[h][s + 1:wk0 + 1] != 0) & (grip[h][s:wk0] == 0) & soon[h][s + 1:wk0 + 1] for h in ("left", "right")}
            tries = max(0, sum(int(c.sum()) for c in cl.values()) - sum(int(c.any()) for c in cl.values()))   # beyond one per hand = regrips
            contact = np.flatnonzero(hc[s:end]) + s
            release = int(contact[-1]) + 1 if len(contact) else int(end)
            if release <= lift:                                                 # touched before it moved, never after: not a handling
                start = max(int(end), lift + 1)
                continue
            st = settle(P[:, pidx[n], 0] + P[:, pidx[n], 1] + zi, release, 0.02, hold=25)
            settle_i = int(min(max(st, release), max(end, release), T - 1))
            carry_s = int(wk0)
            before = [w for w in walk if w[1] <= release and w[0] >= carry_s]
            carry_e = int(before[-1][1]) if before else carry_s
            if carry_e <= carry_s and float(np.linalg.norm(xy[max(lift, release - 1)] - xy[lift])) > 0.5:   # moved > 0.5 m without a walk (slid along the counter)
                carry_s, carry_e = int(lift), int(max(lift + 1, release - 25))
            hold_hands = [h for h in ("left", "right") if has(h, n)[carry_s:carry_e].mean() > 0.5] if carry_e > carry_s else sides_on(n, lift, max(release, lift + 1))
            pick_hands = sides_on(n, s, max(carry_s, lift + 1))
            place_hands = sides_on(n, max(carry_e, lift), max(release, carry_e + 1, lift + 1)) or hold_hands or pick_hands
            let_go = [h for h in pick_hands if h not in hold_hands]
            let_go_t = {h: int(np.flatnonzero(has(h, n)[s:carry_s])[-1] + s) for h in let_go if has(h, n)[s:carry_s].any()}
            hl, hr = has("left", n), has("right", n)                            # hand-overs on the way: the sole holder changes for >= 0.5 s
            holder = np.where(hl & ~hr, 1, np.where(hr & ~hl, 2, 0))
            passes, last, last_seen = [], 0, lift
            for i in range(max(lift, carry_s, max(let_go_t.values(), default=0) + 1), release):   # after the pick and its let-go; each side alone >= 0.5 s
                if holder[i] == 0 or not (holder[i:i + 25] == holder[i]).all():
                    continue
                if last and holder[i] != last:
                    passes.append(dict(frm="left" if last == 1 else "right", to="left" if holder[i] == 1 else "right", s=int(last_seen), e=int(i) + 1))
                last, last_seen = int(holder[i]), i
            wi = min(T - 1, settle_i + 25); where = place_of(n, wi)
            path = float(np.sum(np.linalg.norm(np.diff(base[carry_s:carry_e, :2], axis=0), axis=1))) if carry_e > carry_s else 0.0
            straight = float(np.linalg.norm(base[carry_e - 1, :2] - base[carry_s, :2])) if carry_e > carry_s else 0.0
            handling.append(dict(dish=n, s=int(s), lift=int(lift), tries=tries, setdowns=setdowns, carry_s=carry_s, carry_e=carry_e, release=release, settle=settle_i,
                                 where=where, in_rack=where in M["racks"], from_place=prev_place, leg=legs, near_bin=bool(near_bin(xy[wi])),
                                 pick_hands=pick_hands, hold_hands=hold_hands, place_hands=place_hands, let_go=let_go_t, passes=passes,
                                 path=round(path, 2), straight=round(straight, 2)))
            legs += 1; prev_place = where; rest_i = wi; start = max(int(end), settle_i + 1)
    handling.sort(key=lambda h: h["s"])
    pedal_ev = [dict(s=s, e=e, lid_open=bool(lid[min(T - 1, s):min(T, e + 50)].max() > 0.5 and lid[max(0, s - 5)] < 0.5)) for s, e in runs(feet_on_pedal, min_len=3, gap=250)]
    brace = []
    for side in ("left", "right"):
        for nm in ("left sink", "right sink"):
            for s, e in runs(has(side, nm), min_len=10, gap=50):
                brace.append(dict(side=side, s=s, e=e))

    # ---- bin stops: a carry that stops at the pedal bin, opens the lid with a foot and swipes the dish clean over it -------------------
    BB = M["scene"].get("bin_box"); LID_OPEN = 1.3                              # the lid's travel ends at about 1.45 rad
    def tilt(i, n):                                                             # the dish's own z axis from vertical (deg): 0 = flat
        w, x, y, z = E["prop_quat"][i, pidx[n]]
        return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    foot_runs = runs(feet_on_pedal, min_len=3, gap=5)
    bin_stops = []
    for h in handling:
        n = h["dish"]; presses = [r for r in foot_runs if h["carry_s"] <= r[0] < h["carry_e"]]
        full = next((i for i in range(presses[0][0], h["carry_e"]) if lid[i] >= LID_OPEN), None) if presses else None
        if full is None and BB:
            # no press in this carry: a stop with the dish over the OPEN bin (the lid left up by an earlier press; kitchen7 ep0's second
            # plate was wiped over it without a press) is a bin stop too, from the first frame the dish is inside the bin box
            over = runs(np.array([BB["lo"][0] <= P[i, pidx[n], 0] <= BB["hi"][0] and BB["lo"][1] <= P[i, pidx[n], 1] <= BB["hi"][1] and lid[i] >= LID_OPEN
                                  for i in range(h["carry_s"], h["carry_e"])]), min_len=FPS, gap=25)
            if over: full = h["carry_s"] + over[0][0]
        if full is None:
            continue
        # the stop ends at the first REAL walk after the lid opened (>= 0.30 m net, the Move rule); a shuffle beside the bin is part of
        # the stop (kitchen7 ep0: a 0.08 m shuffle at 37.8 s used to end it before the third press and the three swipes at 41.9-44.9 s)
        s_p = presses[0][0] if presses else full; leave = next((w[0] for w in walk if full < w[0] < h["carry_e"] and float(np.linalg.norm(base[w[1] - 1, :2] - base[w[0], :2])) >= 0.30), h["carry_e"])
        tl = [tilt(i, n) for i in range(full, leave)]; other = [s for s in ("left", "right") if s not in h["hold_hands"]]
        b = dict(dish=n, s=s_p, full=full, leave=leave, presses=len([r for r in presses if r[0] < full]),
                 fell_shut=len([r for r in runs(lid[s_p:full] > 0.3) if r[1] < full - s_p]), foot_holds=bool(feet_on_pedal[full:leave].mean() > 0.8),
                 over_bin=bool(BB) and bool(np.mean([BB["lo"][0] <= P[i, pidx[n], 0] <= BB["hi"][0] and BB["lo"][1] <= P[i, pidx[n], 1] <= BB["hi"][1] for i in range(full, leave)]) > 0.5),
                 tilt=(round(tl[0]), round(max(tl))), tip_e=None, hold=h["hold_hands"], other=other[0] if len(other) == 1 else None, swipes=0, sw=None, touches=0)
        if b["tilt"][1] - b["tilt"][0] >= 20:                                   # the holding hand tips the dish over the bin
            b["tip_e"] = full + next(k for k, v in enumerate(tl) if v >= tl[0] + 0.8 * (max(tl) - tl[0]))
        if b["other"] and leave - full > FPS:
            # the free hand's palm relative to the dish, along its main direction of motion: each swipe is one pass in toward the dish and
            # back out; turning points need 10 cm of travel, and the inner ones (nearer the dish) count the swipes
            rel = E["palm"][full:leave, 0 if b["other"] == "left" else 1] - P[full:leave, pidx[n]]
            c = rel - rel.mean(0); x = c @ np.linalg.svd(c, full_matrices=False)[2][0]; dist = np.linalg.norm(rel, axis=1)
            piv, d, lo, hi, ext = [0], 0, 0, 0, 0
            for i in range(1, len(x)):
                if d == 0:
                    lo, hi = (i if x[i] < x[lo] else lo), (i if x[i] > x[hi] else hi)
                    if x[i] - x[lo] >= 0.10: piv, d, ext = [lo], 1, i
                    elif x[hi] - x[i] >= 0.10: piv, d, ext = [hi], -1, i
                elif (x[i] - x[ext]) * d > 0: ext = i
                elif (x[ext] - x[i]) * d >= 0.10: piv.append(ext); d, ext = -d, i
            piv = piv + [ext] if d else piv
            inward = -np.sign(np.corrcoef(x, dist)[0, 1])                       # +1: the dish lies toward +x
            # an inner turning point counts as a pass only when the palm really came to the dish: within 10 cm of its closest approach
            # (kitchen7 ep0: a turn 0.45 m from the plate at 38.9 s is the hand moving beside the bin, the passes at 41.9-44.3 s are 0.29 m)
            near = [k for j, k in enumerate(piv) if all((x[k] - x[piv[jj]]) * inward > 0 for jj in (j - 1, j + 1) if 0 <= jj < len(piv)) and dist[k] <= dist.min() + 0.10]
            if near and len(piv) >= 2:                                          # a swipe needs a real pass
                j0, j1 = piv.index(near[0]), piv.index(near[-1])               # the window: from the turn before the first pass to the turn after the last
                b.update(swipes=len(near), sw=(full + piv[max(0, j0 - 1)], full + piv[min(len(piv) - 1, j1 + 1)]),
                         touches=len(runs(has(b["other"], n)[full:leave], gap=10)))
        bin_stops.append(b)

    # ---- sub tasks: core events in time order; a carry is cut around door / rack events done with the dish in hand (kitchen6 ep1:
    # the door opened and the rack pulled out while the plate hung in the other hand); gaps become walks / approaches -------------
    fixed = [dict(kind=x["kind"], s=x["s"], e=x["e"], ref=x) for x in ev]
    core = []
    for h in handling:
        b = next((x for x in bin_stops if x["dish"] == h["dish"] and h["carry_s"] <= x["s"] < h["carry_e"]), None)
        core.append(dict(kind="pick", s=h["s"], e=max(h["carry_s"], h["lift"] + 1), ref=h))
        dest = "dishwasher" if h["in_rack"] else "pedal bin" if (h["near_bin"] or h["where"] == "lid of the pedal bin") else "counter"
        if b:                                                                   # carry -> to the bin, pedal, swipe, on to the dishwasher
            fixed += ([dict(kind="pedal", s=b["s"], e=b["full"], ref=h, bin=b)] if b["presses"] else []) + [dict(kind="swipe", s=b["full"], e=b["leave"], ref=h, bin=b)]   # no press at an open bin: swipe only
            carries = [dict(kind="to_bin", s=h["carry_s"], e=b["s"], ref=h, to="pedal bin"), dict(kind="carry", s=b["leave"], e=h["carry_e"], ref=h, bin=b, to=dest)]
        else:
            carries = [dict(kind="carry", s=h["carry_s"], e=h["carry_e"], ref=h, to=dest)]
        for c in carries:
            cuts = sorted((x["s"], x["e"]) for x in fixed if x["s"] < c["e"] and x["e"] > c["s"] and x["kind"] in ("open_door", "close_door", "pull_rack", "push_rack"))
            t0, pieces = c["s"], []
            for cs, ce in cuts:
                if cs > t0:
                    pieces.append((t0, cs))
                t0 = max(t0, ce)
            if c["e"] > t0:
                pieces.append((t0, c["e"]))
            for j, (ps, pe) in enumerate(pieces):
                if pe - ps < 25:
                    continue
                net = float(np.linalg.norm(base[pe - 1, :2] - base[ps, :2]))
                if j > 0 and net < 0.6:                                         # after the door / rack: stepping up to the open rack
                    core.append(dict(kind="position", s=ps, e=pe, to="place", at="dishwasher", ref=h))
                else:
                    core.append(dict(c, s=ps, e=pe))
        core.append(dict(kind="place" if h["in_rack"] else "setdown", s=max(h["carry_e"], h["lift"]), e=h["settle"], ref=h))
    key_t = {"pick": lambda r: r["lift"], "place": lambda r: r["release"], "setdown": lambda r: r["release"]}
    for c in core:                                                              # a door / rack worked in the middle of a pick / place (kitchen7 ep0: the rack
        if c["kind"] not in key_t:                                              # pulled out with the plate already held over it): the handling resumes after it
            continue
        kf = key_t[c["kind"]](c["ref"])
        for x in fixed:
            if x["kind"] in ("open_door", "close_door", "pull_rack", "push_rack") and x["s"] < c["e"] and x["e"] > c["s"]:
                if x["e"] <= kf:
                    c["s"] = max(c["s"], x["e"])
                elif x["s"] >= kf:
                    c["e"] = min(c["e"], x["s"])
    core += fixed
    core = [c for c in core if c["e"] > c["s"]]
    core.sort(key=lambda c: (c["s"], c["e"]))
    for i in range(1, len(core)):                                               # no overlaps: a later event starts where the earlier ends
        core[i]["s"] = max(core[i]["s"], core[i - 1]["e"])
    core = [c for c in core if c["e"] > c["s"]]

    def place_name(c):                                                          # where the next event happens, for the walks / approaches
        k, r = c["kind"], c.get("ref", {})
        if k == "pick":
            if r.get("from_place") == "lid of the pedal bin" or (r.get("leg") and near_bin(P[c["s"], pidx[r["dish"]], :2])):
                return "pedal bin"
            return "counter"
        if k in ("pedal", "swipe", "to_bin"):
            return "pedal bin"
        if k == "setdown":
            return "pedal bin" if r.get("near_bin") or r.get("where") == "lid of the pedal bin" else r.get("where", "counter")
        return c.get("at", "dishwasher")
    subs, t = [], 0
    for c in core:
        if c["s"] > t:
            g = [w for w in walk if w[0] < c["s"] and w[1] > t]
            wtime = sum(min(w[1], c["s"]) - max(w[0], t) for w in g)
            if wtime >= FPS:
                we = min(max(w[1] for w in g), c["s"]); net = float(np.linalg.norm(base[we - 1, :2] - base[t, :2]))
                if net >= 0.6:
                    subs.append(dict(kind="walk", s=t, e=we, to=place_name(c)))
                else:
                    subs.append(dict(kind="position", s=t, e=we, to=c["kind"], at=place_name(c)))
                t = we
            c = dict(c, s=t)                                                    # the rest of the gap: the approach to this event
        else:
            c = dict(c, s=max(c["s"], t))
        subs.append(c); t = c["e"]
    if subs and t < T < t + 25:                                                # less than 0.5 s left: no step-back slice
        subs[-1]["e"] = t = T
    task_end = t
    if t < T:
        subs.append(dict(kind="tail", s=t, e=T))

    def sub_label(c):
        k = c["kind"]; r = c.get("ref", {}); at = c.get("at", "dishwasher"); d = THE(r.get("dish", ""))
        return {"walk": f"Walk to the {c.get('to')}",
                "position": "Step into position at the dishwasher door" if c.get("to") == "close_door" else f"Step into position at the {at}",
                "open_door": "Open the dishwasher door", "close_door": "Close the dishwasher door", "pull_rack": f"Pull out the {r.get('rack')}",
                "push_rack": f"Push the {r.get('rack')} back in",
                "pick": f"Pick up {d} from the counter" if not r.get("leg") else f"Pick {d} up again from the {r.get('from_place')}",
                "carry": f"Carry {d} to the {c.get('to', 'dishwasher')}", "place": f"Place {d} in the {r.get('where') or 'dishwasher'}",
                "setdown": f"Set {d} down on the {r.get('where')}" + (" by the pedal bin" if r.get("where") == "counter" and r.get("near_bin") else ""),
                "to_bin": f"Carry {d} to the pedal bin", "pedal": "Step on the bin pedal to open the lid",
                "swipe": f"Swipe {d} clean over the bin" if c.get("bin", {}).get("swipes") else f"Hold {d} {'over' if c.get('bin', {}).get('over_bin') else 'beside'} the open bin",
                "tail": "Step back from the dishwasher"}[k]
    for c in subs:
        c["label"] = sub_label(c); c["task_relevant"] = c["kind"] != "tail"; c["in_main_task"] = c["kind"] != "tail"
    # ---- actions (finer, with frame ranges) -------------------------------------------------------------------------------------------
    def held_txt(s_, e_):                                                        # a dish kept in the other hand while the door / rack is worked
        hd = [n for n in dishes if anyhand(n)[s_:e_].mean() > 0.5]
        if not hd:
            return ""
        hs = sides_on(hd[0], s_, e_)
        return f" {THE(hd[0]).capitalize()} stays in the {hs[0]} hand the whole time." if hs else f" {THE(hd[0]).capitalize()} stays in hand."

    acts = []
    A = lambda label, s, e, desc, rel=True, main=True: acts.append(dict(label=label, description=desc, task_relevant=rel, start_frame=int(s), end_frame=int(e), in_main_task=main))
    for c in subs:
        k, s, e, r = c["kind"], c["s"], c["e"], c.get("ref", {})
        if k == "walk":
            A(f"Walk to the {c['to']}", s, e, f"The robot walks to the {c['to']} with empty hands." if not any(touch[i][h] for i in range(s, e, 25) for h in ("left", "right")) else f"The robot walks to the {c['to']}.")
        elif k == "position":
            low = bool((E["palm"][s:e, :, 2].max(axis=1) < 0.55).any())
            if c.get("to") == "close_door":
                A("Reposition in front of the open door", s, e, "The robot backs away from the open door and repositions in front of it" + (", lowering both hands beside the door to get under the door panel." if low else "."))
            else:
                A("Step into position", s, e, "The robot shuffles its feet to stand square in front of the open dishwasher; its knees brush the open door." if any("dishwasher door" in touch[i]["body"] for i in range(s, e)) else f"The robot adjusts its stance in front of the {c.get('at', 'dishwasher')}.")
        elif k in ("open_door", "close_door"):
            hs = r["hands"]; first = r["s"]
            if first - s > 25: A("Reach for the dishwasher door handle" if k == "open_door" else "Reach for the dishwasher door", s, first, f"{hands_txt(hs).capitalize()} {vb(hs, 'reaches', 'reach')} for the {'door handle' if k == 'open_door' else 'door'}.")
            if k == "open_door":
                A("Pull the dishwasher door open", first, e, f"{hands_txt(hs).capitalize()} {vb(hs, 'hooks', 'hook')} the handle and {vb(hs, 'pulls', 'pull')} the dishwasher door down until it rests open, about {r['deg'][1]} degrees from closed." + held_txt(s, e))
            else:
                A("Lift the dishwasher door closed", first, e, f"{hands_txt(hs).capitalize()} {vb(hs, 'lifts', 'lift')} the dishwasher door from about {r['deg'][0]} degrees until it shuts." + held_txt(s, e))
        elif k in ("pull_rack", "push_rack"):
            hs = r["hands"]; first = r["s"]
            if first - s > 25: A(f"Reach for the {r['rack']}", s, first, f"{hands_txt(hs).capitalize()} {vb(hs, 'reaches', 'reach')} into the open dishwasher toward the {r['rack']}.")
            if k == "pull_rack":
                A(f"Pull out the {r['rack']}", first, e, f"{hands_txt(hs).capitalize()} {vb(hs, 'grips', 'grip')} the front of the {r['rack']} and {vb(hs, 'slides', 'slide')} it out about {round(100 * (r['m'][1] - r['m'][0]))} cm." + held_txt(s, e))
            else:
                A(f"Push the {r['rack']} back in", first, e, f"{hands_txt(hs).capitalize()} {vb(hs, 'pushes', 'push')} the {r['rack']} back into the dishwasher in {r['taps']} short pushes." + held_txt(s, e))
        elif k == "pick":
            first = r["s"]; hs = r["pick_hands"]; frm = "the counter" if not r["leg"] else "the " + r["from_place"]; again = " again" if r["leg"] else ""
            if first - s > 25: A(f"Reach for {THE(r['dish'])}", s, first, f"{hands_txt(hs).capitalize()} reach{'' if len(hs) == 2 else 'es'} for {THE(r['dish'])} on {frm}.")
            A(f"Grasp {THE(r['dish'])}{again}", first, r["lift"], f"{hands_txt(hs).capitalize()} {vb(hs, 'pinches', 'pinch')} the rim of {THE(r['dish'])} and {vb(hs, 'works', 'work')} it loose from {frm}" +
              (f", setting it back down {times(r['setdowns'])}" if r["setdowns"] else "") + ".")
            A(f"Lift {THE(r['dish'])}", r["lift"], max(e, min(r["lift"] + 25, r["release"])), f"{THE(r['dish']).capitalize()} comes up off {frm}" + (f"; the hands regrip it {times(r['tries'])} while steadying it" if r["tries"] else "") + (
                "; " + " and ".join(f"the {h} hand lets go" for h in r["let_go"]) + f", leaving it in the {r['hold_hands'][0]} hand." if r["let_go"] and r["hold_hands"] else "."))
        elif k in ("to_bin", "carry"):
            hs = sides_on(r["dish"], s, e) or r["hold_hands"]; path, long = leg(s, e); dest = "the " + (c.get("to") or "dishwasher")
            ps_ = [p_ for p_ in r.get("passes", []) if s <= p_["s"] < e]        # a hand-over on the way: the first hand carries, not "both hands"
            if ps_:
                hs = [ps_[0]["frm"]]
            A(f"Carry {THE(r['dish'])} to {dest}", s, e, f"{hands_txt(hs).capitalize()} {vb(hs, 'carries', 'carry')} {THE(r['dish'])} while the robot walks " +
              (f"a winding {path:.1f} m route to {dest}." if long else f"to {dest} ({path:.1f} m).") + (f" On the way it passes {THE(r['dish'])} to the {ps_[-1]['to']} hand." if ps_ else ""))
        elif k == "pedal":
            b = c["bin"]
            A("Press the bin pedal with a foot", s, e, (f"A foot presses the pedal of the bin {times(b['presses'])}; the lid lifts and drops shut {times(b['fell_shut'])} before it opens fully."
                                                       if b["fell_shut"] else f"A foot presses the pedal of the bin {times(b['presses'])}; only the last press opens the lid fully."
                                                       if b["presses"] > 1 else f"A foot presses the pedal of the bin and the lid opens fully."))
        elif k == "swipe":
            b = c["bin"]; held = hands_txt(b["hold"]); where = "over the open bin" if b["over_bin"] else "beside the open bin"
            if b["tip_e"]:
                A(f"Tip {THE(r['dish'])} over the open bin", s, b["tip_e"], f"{held.capitalize()} {vb(b['hold'], 'turns', 'turn')} {THE(r['dish'])} until it is tipped about {b['tilt'][1]} degrees from flat {where}.")
            if b["sw"]:
                A(f"Swipe {THE(r['dish'])} with the {b['other']} hand", b["sw"][0], b["sw"][1],
                  f"The {b['other']} hand swipes across the face of {THE(r['dish'])} and back {times(b['swipes'])} while {held} {vb(b['hold'], 'holds', 'hold')} it {where}"
                  + (", touching it on every pass" if b["touches"] >= b["swipes"] else f", touching it on {b['touches']} of the {b['swipes']} passes" if b["touches"] else "")
                  + ("; a foot keeps the pedal down the whole time." if b["foot_holds"] else "."))
            else:
                A(f"Hold {THE(r['dish'])} {where.replace(' open', '')}", s, e, f"{held.capitalize()} {vb(b['hold'], 'holds', 'hold')} {THE(r['dish'])} {where}.")
        elif k == "place":
            hs = r["place_hands"]
            A(f"Lower {THE(r['dish'])} into the {r['where']}", s, r["release"], f"{hands_txt(hs).capitalize()} {vb(hs, 'lowers', 'lower')} {THE(r['dish'])} upright into the {r['where']}.")
            A(f"Release {THE(r['dish'])}", max(s + 1, r["release"] - 25), e, f"{hands_txt(hs).capitalize()} {vb(hs, 'opens and lets', 'open and let')} go; {THE(r['dish'])} settles in the {r['where']}.")
        elif k == "setdown":
            hs = r["place_hands"]; where = "the " + r["where"]; by = " beside the pedal bin" if r["where"] == "counter" and r["near_bin"] else ""
            A(f"Lower {THE(r['dish'])} onto {where}", s, r["release"], f"{hands_txt(hs).capitalize()} {vb(hs, 'lowers', 'lower')} {THE(r['dish'])} onto {where}{by}.")
            A(f"Release {THE(r['dish'])}", max(s + 1, r["release"] - 25), e, f"{hands_txt(hs).capitalize()} {vb(hs, 'opens and lets', 'open and let')} go; {THE(r['dish'])} lies on {where}.")
        elif k == "tail":
            A("Step back from the dishwasher", s, e, "The robot lowers its hands and steps back from the closed dishwasher.", rel=False, main=False)
    in_stop = lambda p: any(p["s"] < b["leave"] and p["e"] > b["s"] for b in bin_stops)   # overlaps a bin stop (its presses are that stop's sub task)
    for p in pedal_ev:
        if in_stop(p):
            continue                                                            # part of a bin stop's sub tasks
        A("Step on the pedal bin's pedal", p["s"], p["e"], "Walking past the pedal bin, a foot presses its pedal and the lid flips open." if p["lid_open"] else "A foot brushes the pedal of the bin.", rel=False, main=False)
    for b in brace:
        A(f"Rest the {b['side']} hand on the sink", b["s"], b["e"], f"The {b['side']} hand rests on the edge of the sink for support.", rel=False, main=False)
    for h in handling:                                                          # hand-overs on the way (kitchen6 ep1: plate A right -> left before the door)
        for p_ in h["passes"]:
            A(f"Pass {THE(h['dish'])} to the {p_['to']} hand", p_["s"], p_["e"], f"The {p_['to']} hand takes {THE(h['dish'])} from the {p_['frm']} hand, which lets go.")
    for s_, e_ in runs(np.gradient(lid) * FPS < -0.5, min_len=3, gap=25):        # the bin lid falling shut on its own
        if not any(b["s"] <= s_ < b["leave"] for b in bin_stops) and lid[max(0, s_ - 5)] > 0.5 and not feet_on_pedal[max(0, s_ - 25):s_ + 1].any():
            A("The bin lid falls shut", s_, min(T, e_ + 10), "The lid of the pedal bin, left open, drops shut on its own.", rel=False, main=False)
    acts.sort(key=lambda x: (x["start_frame"], -x["end_frame"]))

    # ---- per-sample states -----------------------------------------------------------------------------------------------------------
    sub_at = np.zeros(T, int)
    for i, c in enumerate(subs): sub_at[c["s"]:c["e"]] = i

    def door_state(i):
        dg = math.degrees(door[i]); dv = (door[min(T - 1, i + 50)] - door[max(0, i - 50)]) * FPS / 100
        return "opening" if dv > 0.02 else "closing" if dv < -0.02 else "closed" if dg < 5 else "open" if dg > 60 else "partly open"

    def rack_state(rn, i):
        q = rack[rn]; dv = (q[min(T - 1, i + 25)] - q[max(0, i - 25)]) * FPS / 50
        return "sliding out" if dv > 0.01 else "sliding in" if dv < -0.01 else "pulled out" if q[i] > 0.3 else "pushed in" if q[i] < 0.02 else "partly out"

    def holding(side, i):
        for n in dishes:
            if n in touch[i][side] and (grip[side][i] != 0 or P[i, pidx[n], 2] > z0[n] + 0.02):
                return n
        return None

    def hand(side, i, vis):
        tt = [x for x in touch[i][side] if x not in ("floor", "kitchen", "left hand", "right hand", "robot body")]
        held = holding(side, i); obj = held or next((x for x in tt if x in dishes), None) or next((x for x in tt if x in PARTS_NICE and x != "counter"), None)
        state = None
        if held:
            h_ = next((h for h in handling if h["dish"] == held and h["s"] <= i <= h["settle"]), None)
            state = "grasping" if h_ is None or i < h_["lift"] else "holding" if i < h_["carry_e"] else "placing"
        elif obj in dishes:
            state = "touching"
        elif obj == "dishwasher door":
            d = door_state(i); de = next((x["kind"] for x in ev if x["kind"] in ("open_door", "close_door") and x["s"] <= i <= x["e"]), None)
            state = "pulling open" if d == "opening" or de == "open_door" else "lifting closed" if d == "closing" or de == "close_door" else "touching"
        elif obj in rack:
            d = rack_state(obj, i); state = "pulling" if d == "sliding out" else "pushing" if d == "sliding in" else "touching"
        elif obj in ("left sink", "right sink", "pedal bin"):
            state = "resting on"
        bs = next((b for b in bin_stops if b["sw"] and b["other"] == side and b["sw"][0] <= i <= b["sw"][1]), None)
        if bs and grip[side][i] == 0:                                           # open hand passing over the dish: swiping, even when it brushes it
            held, obj, state = None, bs["dish"], "swiping"
        if not obj:
            prev = [x for k in range(max(0, i - 25), i) for x in touch[k][side] if x in dishes]
            nxt = next((x for k in range(i, min(T, i + 75)) for x in touch[k][side] if x in dishes or x in ("dishwasher door", "upper rack", "lower rack")), None)
            if prev and holding("left" if side == "right" else "right", i) != prev[-1]:
                obj, state = prev[-1], "releasing"
            elif nxt:
                obj, state = nxt, "reaching for"
        return dict(visible=bool(vis), holding=bool(held), object=(NM(obj) if obj in pidx else obj) if obj else None, grasp_state=state, same_object_as_other_hand=False, _id=obj)

    def objects(i):
        res = []
        for o in M["views"].get(str(i), {}).get("objects", []):
            if o["occluded"]:
                continue
            n = o["name"]
            if n in pidx:
                st = "held" if any(holding(h, i) == n for h in ("left", "right")) else (f"in the {in_rack(n, i)}" if n in dishes and in_rack(n, i) else "moving" if pspeed[i, pidx[n]] > 0.02 else "static")
            elif n == "dishwasher door": st = door_state(i)
            elif n in rack: st = rack_state(n, i)
            elif n == "pedal bin": st = "lid open" if lid[i] > 0.5 else "lid closed"
            else: st = "static"
            res.append(dict(name=NM(n), state=st))
        return res

    def sentence(i, H):
        L, R = H["left"], H["right"]
        verb = {"placing": "lowers", "holding": "holds", "grasping": "grips", "touching": "touches", "pulling open": "pulls open", "lifting closed": "lifts closed", "pulling": "pulls out",
                "pushing": "pushes in", "resting on": "rests on", "releasing": "lets go of", "reaching for": "reaches for", "swiping": "swipes across"}
        objtxt = lambda h: ("the " + h["object"]) if h["object"] and not h["object"].startswith("the ") and h["_id"] not in PARTS_NICE else PARTS_NICE.get(h["_id"], h["object"])
        if L["_id"] and L["_id"] == R["_id"] and L["grasp_state"] == R["grasp_state"]:
            parts = [f"Both hands {verb[L['grasp_state']].replace('holds', 'hold').replace('grips', 'grip').replace('touches', 'touch').replace('reaches', 'reach').replace('lets', 'let').replace('pulls', 'pull').replace('lifts', 'lift').replace('pushes', 'push').replace('rests', 'rest').replace('lowers', 'lower')} {objtxt(L)}"]
        else:
            parts = [f"The {s} hand {verb[h['grasp_state']]} {objtxt(h)}" for s, h in (("right", R), ("left", L)) if h["grasp_state"]]
        c = subs[sub_at[i]]
        move = ""
        if walking[i]:
            move = "the robot walks " + ("toward the " + (c.get("to") or "dishwasher") if c["kind"] in ("walk", "carry", "to_bin") else "a few steps")
        elif abs(np.gradient(np.unwrap(yaw))[i] * FPS) > 0.6:
            move = "the robot turns on the spot"
        if not parts:
            near = "counter" if np.linalg.norm(base[i, :2] - cnt_xy) < np.linalg.norm(base[i, :2] - dw_xy) else "dishwasher"
            s = "No hand interaction; " + (move if move else f"the robot stands at the {near}") + "."
        else:
            s = ", and ".join(parts[:1] + [p[0].lower() + p[1:] for p in parts[1:]]) + (f" while {move}" if move else "") + "."
        if feet_on_pedal[i]:
            s += " A foot is on the pedal of the bin and its lid is open." if lid[i] > 0.5 else " A foot is on the pedal of the bin."
        return s[0].upper() + s[1:]

    cnt_xy = np.mean([P[0, pidx[n], :2] for n in dishes], axis=0)
    samples = {}
    for i in range(0, T, SAMPLE):
        hv = M["views"].get(str(i), {}).get("hands", {"left": False, "right": False})
        H = {s: hand(s, i, hv.get(s)) for s in ("left", "right")}
        same = H["left"]["_id"] is not None and H["left"]["_id"] == H["right"]["_id"]
        for s in H: H[s]["same_object_as_other_hand"] = bool(same)
        samples[i] = dict(objects=objects(i), interaction=sentence(i, H), hands={s: {k: v for k, v in H[s].items() if k != "_id"} for s in H})

    # ---- segments in the Horizon layout ------------------------------------------------------------------------------------------------
    F1 = lambda i: int(i) + 1                                                      # 1-based frames like the Horizon files
    segs = []
    for k in range(math.ceil(T / SEG)):
        s, e = k * SEG, min(T, (k + 1) * SEG) - 1
        ov = {}
        for c in subs:
            o = min(e + 1, c["e"]) - max(s, c["s"])
            if o > 0: ov[c["label"]] = ov.get(c["label"], 0) + o
        main_sub = max(ov, key=ov.get); csub = next(c for c in subs if c["label"] == main_sub and min(e + 1, c["e"]) > max(s, c["s"]))
        ca = [dict(x, task_relevant=x["task_relevant"], start_frame=F1(x["start_frame"]), end_frame=x["end_frame"]) for x in acts if x["start_frame"] <= e and x["end_frame"] > s]
        sf = list(range(s, e + 1, SAMPLE))
        segs.append({"segment_id": k, "start_frame": F1(s), "end_frame": F1(e), "Sub Task": main_sub, "Current Action": ca,
                     "sampled_frames": {f"Image {j + 1}": F1(f) for j, f in enumerate(sf)},
                     "objects": {str(F1(f)): samples[f]["objects"] for f in sf}, "interaction": {str(F1(f)): samples[f]["interaction"] for f in sf},
                     "hands": {str(F1(f)): samples[f]["hands"] for f in sf}, "sub_task_start_frame": F1(csub["s"]), "sub_task_end_frame": csub["e"],
                     "seconds": [round(s / FPS, 2), round((e + 1) / FPS, 2)]})

    # ---- task text, summary, caption (the sentences follow the order things happened; 2026-10-30) ----------------------------------
    placed = [h for h in handling if h["in_rack"]]
    n_pl = len({h["dish"] for h in placed})
    did_close = any(c["kind"] == "close_door" for c in subs)
    wiped = [b for b in bin_stops if b["swipes"] and any(h["dish"] == b["dish"] for h in placed)]
    in_hand = lambda c: [n for n in dishes if anyhand(n)[c["s"]:c["e"]].mean() > 0.5]
    steps, owner, reps = [], [], []
    GOES = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    def step(txt, own=None):
        if steps and own is None and steps[-1] == txt:                          # the same step again (kitchen7 ep1: the rack pushed in in three goes)
            reps[-1] += 1
            return
        steps.append(txt); owner.append(own); reps.append(1)
    for c in subs:
        k, r = c["kind"], c.get("ref", {}); d = THE(r["dish"]) if r.get("dish") else ""
        same = bool(owner) and r.get("dish") is not None and owner[-1] == r.get("dish")
        if k == "open_door": step("opens the dishwasher" + (f" with {THE(in_hand(c)[0])} in hand" if in_hand(c) else ""))
        elif k == "pull_rack": step(f"pulls out the {r['rack']}")
        elif k == "pick": step(f"carries {d} from the counter" if not r["leg"] else f"picks {d} up again from the {r['from_place']}", r["dish"])
        elif k == "to_bin" and same: steps[-1] += " to the pedal bin" if steps[-1].startswith("carries") else ", carries it to the pedal bin"
        elif k == "swipe" and same: steps[-1] += (", swipes it clean over the open bin" if c["bin"]["swipes"] else f", holds it {'over' if c['bin']['over_bin'] else 'beside'} the open bin")
        elif k == "setdown":
            if same and r["where"] == r["from_place"] == "counter" and r["carry_e"] <= r["carry_s"] and steps[-1].startswith("carries "):
                steps[-1] = f"lifts {d} off the counter and sets it straight back down"   # an aborted lift (kitchen7 ep1: 1.6 s in the air)
            elif same: steps[-1] += f" and sets it down on the {r['where']}" + (" beside the bin" if r["where"] == "counter" and r["near_bin"] else "")
            else: step(f"sets {d} down on the {r['where']}", r["dish"])
        elif k == "place":
            if same: steps[-1] += f" and stands it in the {r['where']}"
            else: step(f"stands {d} in the {r['where']}", r["dish"])
        elif k == "push_rack": step(f"pushes the {r['rack']} back in")
        elif k == "close_door": step("closes the door")
    kinds = [c["kind"] for c in subs if c["kind"] in ("open_door", "pull_rack", "pick")]
    regular = (n_pl > 1 and all(h["leg"] == 0 and h["in_rack"] for h in handling) and len(wiped) == n_pl and kinds[:2] == ["open_door", "pull_rack"]
               and not any(c["kind"] == "setdown" for c in subs))                 # the plain run: open, rack out, each plate counter -> bin -> rack
    if regular:
        racks_out = [c["ref"]["rack"] for c in subs if c["kind"] == "pull_rack"]; racks_in = sorted({c["ref"]["rack"] for c in subs if c["kind"] == "push_rack"})
        steps = ["opens the dishwasher"] + [f"pulls out the {r_}" for r_ in racks_out] + \
                [f"carries {'two' if n_pl == 2 else n_pl} plates from the counter one at a time", "swipes each one clean over the pedal bin", f"stands them in the {placed[0]['where']}"] + \
                [f"pushes the {r_} back in" for r_ in racks_in] + (["closes the door"] if did_close else [])
    else:
        steps = [t_ + (f" over {GOES.get(n_, str(n_))} goes" if n_ > 1 else "") for t_, n_ in zip(steps, reps)]
    main_task = "Loading the dishwasher. The robot " + ((", ".join(steps[:-1]) + (", and " if len(steps) > 1 else "") + steps[-1]) if steps else "handles the dishes") + "."
    summary = "A humanoid robot with dexterous hands loads " + ("two plates" if n_pl == 2 else f"{n_pl} plate(s)") + " from a kitchen counter into a dishwasher" + \
              (", swiping each plate clean over the pedal bin on the way," if wiped and len(wiped) == n_pl else f", swiping {GOES.get(len(wiped), str(len(wiped)))} of them clean over the pedal bin," if wiped else "") + (" and closes it." if did_close else ".")
    para, pushed, ORD = [], {}, {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    first_walk = next((c for c in subs if c["kind"] == "walk"), None)
    para.append("Filmed from the robot's head camera in a first-person view, the video opens in a kitchen with a counter of dishes, bottles and cartons. "
                + (f"The robot turns away from the counter and walks to the dishwasher at the far side of the kitchen. " if first_walk and first_walk["s"] == 0 else ""))
    for c in subs:
        r = c.get("ref", {}); k = c["kind"]; d = THE(r["dish"]) if r.get("dish") else "the plate"
        ps = [p_ for p_ in r.get("passes", []) if c["s"] <= p_["s"] < c["e"]]
        if k == "open_door":
            hd = in_hand(c)
            txt = (f"With {THE(hd[0])} still in the other hand, " if hd else "") + f"{hands_txt(r['hands'])} {vb(r['hands'], 'hooks', 'hook')} the handle and {vb(r['hands'], 'pulls', 'pull')} the door down until it rests open. "
            para[-1] += txt[0].upper() + txt[1:]
        elif k == "pull_rack":
            br = [b for b in brace if b["s"] < c["e"] and b["e"] > c["s"]]
            txt = ("With the left hand resting on the edge of the sink, " if br else "") + f"{hands_txt(r['hands'])} {vb(r['hands'], 'grips', 'grip')} the {r['rack']} and {vb(r['hands'], 'slides', 'slide')} it out. "
            para[-1] += txt[0].upper() + txt[1:]
        elif k == "pick":
            h = r
            if not h["leg"]:
                txt = f"At the counter, {hands_txt(h['pick_hands'])} pinch{'' if len(h['pick_hands']) == 2 else 'es'} {d}"
                txt += (f", regripping it {times(h['tries'])} before it comes up off the counter" if h["tries"] else " and lift it" if len(h["pick_hands"]) == 2 else " and lifts it")
                txt += (", then the " + " and ".join(h["let_go"]) + f" hand lets go and the {h['hold_hands'][0]} hand carries it alone. " if h["let_go"] and h["hold_hands"] else ". ")
                para.append(txt)
            else:
                para[-1] += f"{hands_txt(h['pick_hands']).capitalize()} {vb(h['pick_hands'], 'picks', 'pick')} {d} up again from the {h['from_place']}. "
        elif k == "to_bin":
            path_, long_ = leg(c["s"], c["e"])
            para[-1] += f"Holding the plate out in front of it, the robot {'takes a winding route to' if long_ else 'walks to' if path_ >= 1.0 else 'steps up to'} the pedal bin. "
        elif k == "pedal":
            b = c["bin"]
            para[-1] += "A foot presses the bin's pedal" + (f" {times(b['presses'])}; the lid drops shut {times(b['fell_shut'])} before it stays open. " if b["fell_shut"] else
                                                          f" {times(b['presses'])}; only the last press opens the lid. " if b["presses"] > 1 else " and the lid opens. ")
        elif k == "swipe":
            b = c["bin"]
            para[-1] += (f"The {b['hold'][0]} hand tips the plate over the open bin and the {b['other']} hand swipes across it {times(b['swipes'])} to clean it"
                         + (", the foot still holding the pedal down. " if b["foot_holds"] else ". ")) if b["swipes"] else f"The robot holds the plate {'over' if b['over_bin'] else 'beside'} the open bin. "
        elif k == "carry" and c.get("bin"):
            para[-1] += f"Then it carries the plate on to the dishwasher{' by a winding route' if leg(c['s'], c['e'])[1] else ''}. "
        elif k == "carry":
            long = leg(c["s"], c["e"])[1]; dest = c.get("to", "dishwasher")
            pe = [p for p in pedal_ev if c["s"] <= p["s"] < c["e"] and not in_stop(p)]
            para[-1] += (f"The robot takes a winding route to the {dest}, " if long else f"The robot walks to the {dest}, ") + "holding the plate out in front of it" + \
                        (", and as it passes the pedal bin a foot presses the pedal and the bin's lid flips open. " if pe and pe[0]["lid_open"] else ". ")
        elif k == "setdown":
            if r["where"] == r["from_place"] == "counter" and r["carry_e"] <= r["carry_s"] and para[-1].rstrip().endswith("it."):
                para[-1] = para[-1].rstrip()[:-1] + ", but sets it straight back down. "   # an aborted lift
            else:
                para[-1] += f"It sets {d} down on the {r['where']}" + (" beside the pedal bin. " if r["where"] == "counter" and r["near_bin"] else ". ")
        elif k == "place":
            h = r
            para[-1] += f"At the dishwasher it lowers {d} upright into the {h['where']} and opens its hand; the plate settles between the tines" + (" beside the first one. " if placed and h is not placed[0] else ". ")
        elif k == "push_rack":
            pushed[r["rack"]] = pushed.get(r["rack"], 0) + 1
            if pushed[r["rack"]] > 1:
                para[-1] += f"A {ORD.get(pushed[r['rack']], 'further')} push sends the {r['rack']} further in. "
            else:
                para.append(f"With {'both plates' if n_pl == 2 else 'the plates' if n_pl > 2 else 'the plate'} in place, {hands_txt(r['hands'])} {vb(r['hands'], 'pushes', 'push')} the {r['rack']} back into the dishwasher in a few short shoves. ")
        elif k == "position":
            if c.get("to") == "close_door":
                para[-1] += "It then backs away from the open door and repositions in front of it, lowering both hands beside the door to find a way under it. "
            elif c.get("to") == "pull_rack":
                para[-1] += "Stepping closer, its knees brush the open door. "
        elif k == "close_door":
            rq = max(float(rack[rn][c["s"]]) for rn in rack)
            para[-1] += f"{hands_txt(r['hands']).capitalize()} {vb(r['hands'], 'gets', 'get')} under the door and {vb(r['hands'], 'lifts', 'lift')} it until it shuts" + (", which also pushes the rack the last few centimetres in. " if rq > 0.03 else ". ")
        elif k == "tail":
            para[-1] += "The video ends with the robot stepping back from the closed dishwasher."
        if ps:
            para[-1] += f"On the way it passes the plate to its {ps[0]['to']} hand" + "".join(f", then {'back ' if p_['to'] == ps[0]['frm'] else ''}to its {p_['to']} hand" for p_ in ps[1:]) + ". "
    video_caption = "\n\n".join(p.strip() for p in para)
    lint = [x for x in acts if x["end_frame"] <= x["start_frame"]]

    kept = task_end
    caption = {"config": {"api_version": None, "provider": "simulation ground truth", "model": "rules over the bit-exact replay (scripts/holomotion_v14_checks/episode_events.py + episode_annotate.py)",
                          "segment_sec": SEG / FPS, "sample_fps": FPS / SAMPLE, "fps": FPS, "frame_base": 1, "Main Task": main_task,
                          "_lint": {"empty_actions": len(lint)},
                          "trim": {"kept_frames": kept, "original_frames": T, "rule": "cut at the end of the last task action (the dishwasher door shut and still), dropping the step back",
                                   "cut_after": next(c["label"] for c in reversed(subs) if c["kind"] != "tail"), "note": "frames are 1-based: frame n = dataset frame_index n-1"},
                          "source": {"dataset": str(Path(a.level_dir)), "episode": a.episode, "replay": "bit-exact"}},
               "segments": segs, "global_summary": summary, "video_caption": video_caption}
    (out / "caption.json").write_text(json.dumps(caption, indent=1))
    meta = {"episode": a.episode, "task_labels": {"level_1": "Object manipulation", "level_2": "Transport and placement", "level_3": main_task},
            "scene": "Kitchen (SIMPLE MuJoCo, manipulable kitchen rev %s)" % M["scene"].get("rev"), "duration_seconds": round(T / FPS, 2),
            "device": {"robot": "Unitree G1 rot0 with Revo 2 hands", "teleop": "PICO + HoloMotion v1.4"}, "trim": caption["config"]["trim"]}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    slices = [dict(index=i + 1, sub_task=c["label"], kind=c["kind"], start_frame=F1(c["s"]), end_frame=c["e"], start_s=round(c["s"] / FPS, 2), end_s=round(c["e"] / FPS, 2),
                   seconds=round((c["e"] - c["s"]) / FPS, 2), task_relevant=c["task_relevant"], in_main_task=c["in_main_task"], kept=c["e"] <= kept,
                   actions=[x["label"] for x in acts if x["start_frame"] < c["e"] and x["end_frame"] > c["s"]]) for i, c in enumerate(subs)]
    (out / "slices.json").write_text(json.dumps(slices, indent=1))
    traj = dict(T=T, fps=FPS, x=[round(float(v), 3) for v in base[::5, 0]], y=[round(float(v), 3) for v in base[::5, 1]], sub=[int(sub_at[i]) for i in range(0, T, 5)],
                subs=[c["label"] for c in subs], acts=[dict(label=x["label"], s=x["start_frame"], e=x["end_frame"], rel=x["task_relevant"]) for x in acts],
                events=dict(handling=[{k: v for k, v in h.items()} for h in handling], pedal=pedal_ev, brace=brace, bin_stops=bin_stops,
                            door=[{k: v for k, v in x.items()} for x in ev]))
    (out / "traj.json").write_text(json.dumps(traj))
    print(f"[annotate] {T} frames: {len(subs)} sub tasks, {len(acts)} actions, {len(segs)} segments, {len(samples)} sampled frames; trim keeps {kept} of {T}")
    for s_ in slices:
        print(f"  {s_['index']:2d} {s_['start_s']:6.1f}-{s_['end_s']:6.1f} s {s_['sub_task']}{'' if s_['task_relevant'] else '  (not task-relevant)'}")
    print("main task:", main_task)


if __name__ == "__main__":
    main()
