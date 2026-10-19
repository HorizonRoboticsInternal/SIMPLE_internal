#!/usr/bin/env python3
"""Build the next fitting round for a kit from the last round's results/traces.

    python sim/fit_next.py bowl_sink   --from fit2 --to fit3        # reads replay/fit_psi0_<n>.json + fitbatch_fit2.jsonl + traces -> fit_psi0_<n+1>.json
    python sim/fit_next.py coffee_cart --from fit2 --to fit3
    python sim/fit_next.py bottle_bin  --from psi0fit3 --to psi0fit4   # reads replay/jobs_psi0_fit3.json + sweep/psi0fit3/ep*.json -> jobs_psi0_fit4.json
Rules: success -> keep. Grasped but not placed -> move the container under where the object landed. Not grasped -> put the object
where THIS run's hand closed (+ the calibrated offset); if it already was there, nudge it by a round-dependent 2-3 cm step.
"""
import argparse, glob, json, os, re, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
NUDGES = [(-0.02, 0.02), (0.02, -0.02), (0.0, 0.03), (-0.03, 0.0), (0.03, 0.02), (-0.02, -0.03)]


def nudge(round_idx):
    return np.array(NUDGES[round_idx % len(NUDGES)])


def bowl(a):
    sys.path.insert(0, str(HERE / "bowl_sink/replay")); import fit_placement as fp
    cl = {int(k): v for k, v in json.load(open(HERE / "bowl_sink/replay/closure_rows_0918.json")).items()}
    n_from = int(re.sub(r"\D", "", a.frm)); fits = json.load(open(a.fits_file or (HERE / f"bowl_sink/replay/fit_psi0_{n_from}.json")))
    res = {r["ep"]: r for r in (json.loads(l) for l in open(a.results_file or (HERE / f"bowl_sink/replay/fitbatch_{a.frm}.jsonl")) if l.strip())}
    OFF = np.array([0.06, 0.07]); TOE = fp.TOE; R_B = fp.BOWL_R; COUNTER_D = 0.63; RTE_MIN, RTE_MAX = 0.15, 0.38
    out = {}; kinds = {"placed": 0, "basin_moved": 0, "bowl_moved": 0, "nudged": 0, "kept": 0}
    for ep_s, p in fits.items():
        ep = int(ep_s); r = res.get(ep, {}); q = dict(p); q["rounds_failed"] = int(p.get("rounds_failed", 0)) + (0 if r.get("success") else 1)
        if r.get("success"):
            kinds["placed"] += 1; q["note"] = f"placed in {a.frm}"; out[ep_s] = q; continue
        PHYSB = os.environ.get("FIT_PHYSICAL") == "1"            # 2026-09-28: physical replays only (user) -> no grasp assist
        if PHYSB: q.pop("grasp_assist", None)
        else: q["grasp_assist"] = True                            # 2026-09-27: bowl rides with the palm from the recorded closure to the release
        if not PHYSB and r.get("assist_captured") is not None and r.get("basin") and r.get("open_row") is not None:
            try:                                                   # basin under the assisted release point (bowl position 5 ticks after the release row)
                tr = json.load(open(HERE / f"bowl_sink/replay/trace_ep{ep}_{a.frm}.json")); tk = [x for x in tr if "summary" not in x]; rows = np.array([x["row"] for x in tk])
                o = tk[min(len(tk) - 1, int(np.abs(rows - r["open_row"]).argmin()) + 5)]; land = np.array(o["target_xyz"][:2]); bsn = np.array(r["basin"][:2])
                dxy = land - bsn; near_x = p["robot_to_edge"] + TOE; bx, by = bsn[0] + dxy[0], bsn[1] + dxy[1]
                q["sink_along"] = round(near_x - bx + fp.BASIN_L / 2, 3); q["robot_dy"] = round(max(-0.90, p["robot_dy"] + (bsn[1] - by)), 3)
                q["note"] = f"basin under the assisted release (landing {land.round(3).tolist()})" + (" ; robot_dy clamped (sink lane)" if p["robot_dy"] + (bsn[1] - by) < -0.90 else ""); kinds["basin_moved"] += 1; out[ep_s] = q; continue
            except Exception as e:
                q["note"] = f"assist basin kept ({e!r})"; kinds["kept"] += 1; out[ep_s] = q; continue
        if r.get("grasped") and r.get("bowl_end_minus_basin") and r["bowl_end_minus_basin"][2] < 0.3:
            d = r["bowl_end_minus_basin"]; near_x = p["robot_to_edge"] + TOE; bx, by = r["basin"][0] + d[0], r["basin"][1] + d[1]
            q["sink_along"] = round(near_x - bx + fp.BASIN_L / 2, 3); q["robot_dy"] = round(p["robot_dy"] + (r["basin"][1] - by), 3)
            q["note"] = f"basin moved by {np.round(d[:2], 3).tolist()}"; kinds["basin_moved"] += 1; out[ep_s] = q; continue
        try:
            tr = json.load(open(HERE / f"bowl_sink/replay/trace_ep{ep}_{a.frm}.json")); tk = [x for x in tr if "summary" not in x]; rows = np.array([x["row"] for x in tk])
            s = next(x["summary"] for x in tr if "summary" in x); x = tk[int(np.abs(rows - cl[ep]["grasp_row"]).argmin())]
            mid = (np.array(x["index_tip_r"][:2]) + np.array(x["palm_r"][:2])) / 2; b0 = np.array(s["target_start"][:2]); want = mid + OFF
            if np.linalg.norm(want - b0) > 0.02: bowl_xy = want; kinds["bowl_moved"] += 1; q["note"] = "bowl re-placed from this run's hand path"
            else: bowl_xy = b0 + nudge(q["rounds_failed"]); kinds["nudged"] += 1; q["note"] = f"bowl nudged {nudge(q['rounds_failed']).tolist()}"
            rte = float(np.clip(min(RTE_MAX, bowl_xy[0] - R_B - 0.01 - TOE), RTE_MIN, RTE_MAX)); near_x = rte + TOE
            bowl_x = float(np.clip(bowl_xy[0], near_x + R_B + 0.01, near_x + COUNTER_D - R_B))
            old_near = p["robot_to_edge"] + TOE; q["sink_along"] = round(p["sink_along"] + (near_x - old_near), 3)      # basin stays put relative to the robot
            q.update(robot_to_edge=round(rte, 3), bowl_xy=[round(bowl_x, 3), round(float(bowl_xy[1]), 3)])
        except Exception as e:
            q["note"] = f"kept ({e!r})"; kinds["kept"] += 1
        # the sink counter (front at y = -0.56 - robot_dy) must stay clear of the robot's start and walk: robot_dy >= -0.30
        if q.get("robot_dy", 0.0) < -0.30:
            q["sink_along"] = round(q["sink_along"], 3); q["robot_dy"] = -0.30; q["note"] = (q.get("note", "") + "; robot_dy clamped (sink lane)").strip("; ")
        out[ep_s] = q
    n_to = int(re.sub(r"\D", "", a.to)); dst = a.out_file or (HERE / f"bowl_sink/replay/fit_psi0_{n_to}.json"); json.dump(out, open(dst, "w"), indent=1)
    print("bowl_sink", a.to, kinds, "->", str(dst))


def coffee(a):
    import pandas as pd
    L = json.load(open(HERE / "coffee_cart/layout.json"))["layout"]; CART0 = np.array([L["cart_cx"], L["cart_cy"]]); CUP0 = np.array([L["cup_x"], L["cup_y"]])
    BOX = (L["box_near_x"] + 0.05, L["box_far_x"] - 0.06, L["box_right_y"] + 0.05, L["box_left_y"] - 0.05); OFF_CUP = np.array([0.124, 0.054]); TFL_MIN = 0.25
    n_from = int(re.sub(r"\D", "", a.frm)); fits = json.load(open(a.fits_file or (HERE / f"coffee_cart/replay/fit_psi0_{n_from}.json")))
    res = {r["ep"]: r for r in (json.loads(l) for l in open(a.results_file or (HERE / f"coffee_cart/replay/fitbatch_{a.frm}.jsonl")) if l.strip())}
    HD_REF = np.array([0.028, 0.016, -0.108]); HD_GAIN = 0.8; HD_MAX = 0.20   # cup - finger-midpoint (base frame) at closure in the successful replays
    out = {}; kinds = {"success": 0, "table_moved": 0, "cup_moved": 0, "nudged": 0, "kept": 0, "hand_delta": 0}
    for ep_s, p in fits.items():
        if "skip" in p: out[ep_s] = p; continue
        ep = int(ep_s); r = res.get(ep, {}); q = dict(p); q["rounds_failed"] = int(p.get("rounds_failed", 0)) + (0 if r.get("success") else 1)
        if not (r.get("success") and r.get("on_table")) and (abs(float(p.get("robot_to_cart", L["robot_to_cart"])) - L["robot_to_cart"]) > 0.15 or abs(float(p.get("robot_dy", 0.0))) > 0.35 or not (1.5 < float(p.get("route_forward", 3.0)) < 6.0)):
            q.pop("robot_to_cart", None); q["robot_dy"] = 0.0; q["cup_xy"] = [L["cup_x"], L["cup_y"]]; q.pop("hand_delta", None); q.pop("box_from_handle", None)
            q["route_forward"] = round(max(2.0, min(5.0, float(p.get("route_forward", 3.0)))), 3); q["table_from_line"] = round(max(0.25, min(1.5, float(p.get("table_from_line", 0.5)))), 3)
            q["note"] = "reset (runaway cart/table fit)"; kinds["kept"] += 1; out[ep_s] = q; continue
        if r.get("success") and ("on_table" not in r or r.get("on_table")):   # coffee: latched success with the cup off the table at the end is a failure
            kinds["success"] += 1; q["note"] = f"succeeded in {a.frm}"; out[ep_s] = q; continue
        ce = r.get("cup_end"); tc = r.get("table_centre")
        PHYS = os.environ.get("FIT_PHYSICAL") == "1"             # 2026-09-28: physical replays only (user) -> no grasp assist, no arm correction
        if PHYS:
            for k_ in ("grasp_assist", "cup_pin", "cart_park", "cart_attach", "hand_delta"): q.pop(k_, None)
            if q.pop("box_from_handle", None) is not None and q.get("cup_xy"):   # no box-top extension: the cup back onto the normal top
                _sx = float(q.get("robot_to_cart", L["robot_to_cart"])) - L["robot_to_cart"]
                q["cup_xy"] = [round(max(float(q["cup_xy"][0]), BOX[0] + _sx), 3), q["cup_xy"][1]]
        else:
            q["grasp_assist"] = True                               # 2026-09-27: the Dex3 pinch on the cup is not reproducible -> cup rides with the palm
        rtc = float(p.get("robot_to_cart", L["robot_to_cart"])); sx = rtc - L["robot_to_cart"]          # this episode's cart shift (x)
        C0 = CART0 + np.array([sx, 0.0]); BOXe = (BOX[0] + sx, BOX[1] + sx, BOX[2], BOX[3])
        if PHYS:
            try:                                                   # HANDLE FIT: the bar must sit in the palms when the recorded hand closes on it
                A_ = np.stack(pd.read_parquet(DATA / f"real_recordings/psi0/CartCoffeeCup_0919/data/chunk-000/episode_{ep:06d}.parquet")["action"]).astype(float)
                cl_ = np.where(A_[:, 10] > 0.6)[0]; hrow = int(cl_[0]) + 15 if len(cl_) else None
                tr_ = [x for x in json.load(open(HERE / f"coffee_cart/replay/trace_ep{ep}_{a.frm}.json")) if "summary" not in x]
                rows_ = np.array([x["row"] for x in tr_]); x_ = tr_[int(np.abs(rows_ - hrow).argmin())]
                bar_x = L["handle_x"] + sx + float(x_["cart_xy"][0])
                dxm = 0.5 * (float(x_["palm_r"][0]) + float(x_["tips"]["left_wrist_yaw_link"][0])) - bar_x
                cc = np.array([y.get("hand_bar_contacts", 0) for y in tr_]); pu = np.array([abs(y["vx"]) > 0.05 for y in tr_])
                share = float((cc[pu] > 0).mean()) if pu.any() else 1.0
                HANDLE_DX = -0.14                                   # palm (wrist-yaw origin) behind the bar in the replays that hold it
                if abs(dxm - HANDLE_DX) > 0.03 and share < 0.6:
                    delta = float(np.clip(dxm - HANDLE_DX, -0.15, 0.15)); new_rtc = float(np.clip(rtc + delta, 0.02, 0.35)); delta = new_rtc - rtc
                    q["robot_to_cart"] = round(new_rtc, 3)
                    if q.get("cup_xy"): q["cup_xy"] = [round(float(q["cup_xy"][0]) + delta, 3), q["cup_xy"][1]]
                    else: q["cup_xy"] = [round(L["cup_x"] + sx + delta, 3), L["cup_y"]]
                    q["note"] = f"handle fit: palms {dxm:+.3f} m from the bar (want {HANDLE_DX:+.2f}), hand-bar contact {share:.0%} of the push -> cart {delta:+.3f} m"
                    kinds["handle_fit"] = kinds.get("handle_fit", 0) + 1; out[ep_s] = q; continue
            except Exception as e:
                q["note_handle"] = f"handle fit skipped ({e!r})"
        if not PHYS and r.get("assist_captured") is not None and not r.get("on_table"):
            try:                                                   # table under the assisted release point (the cart lane is allowed beyond the cart's end)
                tr = [x for x in json.load(open(HERE / f"coffee_cart/replay/trace_ep{ep}_{a.frm}.json")) if "summary" not in x]; rows = np.array([x["row"] for x in tr])
                cart_path = CART0 + np.array([x["cart_xy"] for x in tr])
                o = tr[min(len(tr) - 1, int(np.abs(rows - (p.get("cup_open_row") or r.get("cup_open_row"))).argmin()) + 5)]
                land = np.array(o["target_xyz"][:2]); cx = float(land[0] + 0.15)          # x: the landing sits near the middle of the 0.6 m depth
                # y: the table's left edge must stay right of the cart's PATH (deck half-width 0.235 + 0.05) where the table's x-span is
                near = cart_path[np.abs(cart_path[:, 0] - cx) < 0.30 + 0.36]
                lane_right = float(near[:, 1].min() - 0.235 - 0.05) if len(near) else 9.0
                cy = min(float(land[1] - 0.38), lane_right - 0.53)   # preferred: the cup 0.15 m inside the near edge
                if land[1] - cy > 0.50: cy = float(land[1] - 0.50)   # keep the landing inside (3 cm margin) even if the table brushes the lane
                q.update(route_forward=round(cx + 0.3, 3), table_from_line=round(-cy - 0.53, 3))
                q["note"] = f"table under the assisted release (landing {land.round(3).tolist()}, lane right {lane_right:.2f})"; kinds["table_moved"] += 1; out[ep_s] = q; continue
            except Exception as e:
                q["note"] = f"assist table kept ({e!r})"; kinds["kept"] += 1; out[ep_s] = q; continue
        if r.get("assist_captured") is None and r.get("rounds_failed", 0) >= 0 and not r.get("grasped"):
            pass                                                   # no capture yet: keep the position fit below, the assist is now on
        if r.get("grasped") and ce and tc and ce[2] < 0.5 and not r.get("on_table") and not (PHYS and -ce[1] - 0.53 < TFL_MIN):
            # the cup was carried and let go somewhere: put the table centre there (lane kept clear). PHYS: a cup that fell INSIDE
            # the cart's lane was dropped during the carry (a grasp problem) - a table cannot go there, fall through to the cup fit
            route = float(ce[0] + 0.3); tfl = max(TFL_MIN, float(-ce[1] - 0.53))
            q.update(route_forward=round(route, 3), table_from_line=round(tfl, 3)); q["note"] = "table moved under the landed cup"; kinds["table_moved"] += 1; out[ep_s] = q; continue
        try:
            if "cup_close_row" not in p:                          # closure rows from the recording (the last hand-close event is the cup, the first the handle)
                A = np.stack(pd.read_parquet(DATA / f"real_recordings/psi0/CartCoffeeCup_0919/data/chunk-000/episode_{ep:06d}.parquet")["action"]).astype(float)
                ev = []; closed = False
                for i_, v_ in enumerate(A[:, 10]):
                    if not closed and v_ > 0.6: closed = True; ev.append(("close", i_))
                    elif closed and v_ < 0.3: closed = False; ev.append(("open", i_))
                closes = [i_ for k_, i_ in ev if k_ == "close"]; opens = [i_ for k_, i_ in ev if k_ == "open"]
                p = dict(p); p["cup_close_row"] = closes[-1] if closes else len(A) - 1; p["cup_open_row"] = next((o for o in opens if o > p["cup_close_row"]), len(A) - 1); q.update(cup_close_row=p["cup_close_row"], cup_open_row=p["cup_open_row"])
            tr = [x for x in json.load(open(HERE / f"coffee_cart/replay/trace_ep{ep}_{a.frm}.json")) if "summary" not in x]; rows = np.array([x["row"] for x in tr])
            c = tr[int(np.abs(rows - p["cup_close_row"]).argmin())]; cart_c = C0 + np.array(c["cart_xy"]); cup_c = np.array(c["target_xyz"][:2]); rel0 = CUP0 - CART0
            tips = c.get("tips") or {}
            if not PHYS and c["target_xyz"][2] > 0.5 and "right_hand_index_1_link" in tips and "right_hand_thumb_2_link" in tips:
                # closed loop on the HAND: where the fingers closed relative to the cup, vs where they close in the successful replays
                cup3 = np.array(c["target_xyz"]); mid = (np.array(tips["right_hand_index_1_link"]) + np.array(tips["right_hand_thumb_2_link"])) / 2
                yw = float(c["base_yaw"]); cy, sy = np.cos(-yw), np.sin(-yw); dv = cup3 - mid
                err = np.array([cy * dv[0] - sy * dv[1], sy * dv[0] + cy * dv[1], dv[2]]) - HD_REF
                if np.hypot(err[0], err[1]) > 0.06 or p.get("hand_delta") and max(abs(v) for v in p["hand_delta"]) >= 0.19:
                    # a big miss (or a saturated hand delta): move the CUP to the fingers (cart frame) and drop the hand delta
                    yaw_c = np.arctan2(*(cup_c - cart_c)[::-1]) - np.arctan2(rel0[1], rel0[0])      # cart yaw from the cup's ride
                    Rm = np.array([[np.cos(-yaw_c), -np.sin(-yaw_c)], [np.sin(-yaw_c), np.cos(-yaw_c)]])
                    cup_new = CART0 + Rm @ (mid[:2] - cart_c); q.pop("hand_delta", None)              # the cup goes where the fingers closed
                    bfh = float(p.get("box_from_handle", 0.09)); near_lim = BOX[0] + (bfh - 0.09)   # the box top may already be extended
                    clipped = np.array([np.clip(cup_new[0], near_lim, BOX[1]), np.clip(cup_new[1], BOX[2], BOX[3])]); over = cup_new - clipped
                    if over[0] < -0.02:                           # the fingers close NEARER than the box top's near edge: extend the top toward the handle
                        bfh = max(-0.20, round(bfh + float(over[0]) - 0.03, 3)); q["box_from_handle"] = bfh
                        clipped[0] = max(float(cup_new[0]), BOX[0] + (bfh - 0.09))   # the cup stays ON the (extended) top
                        q["note_box"] = f"box top extended to {bfh:+.2f} m from the handle end"
                    q["cup_xy"] = [round(float(clipped[0]), 3), round(float(clipped[1]), 3)]
                    q["note"] = f"cup moved to the fingers (miss {err.round(3).tolist()})"; kinds["cup_moved"] += 1; out[ep_s] = q; continue
                hd = np.clip(np.array(p.get("hand_delta", [0.0, 0.0, 0.0]), dtype=float) + HD_GAIN * err, -HD_MAX, HD_MAX)
                q["hand_delta"] = hd.round(3).tolist(); q["note"] = f"hand delta {hd.round(3).tolist()} (finger error {err.round(3).tolist()})"; kinds["hand_delta"] += 1
                o = tr[int(np.abs(rows - p["cup_open_row"]).argmin())]; tab = np.array(o["palm_r"][:2]) + np.array([0.20, -0.38])
                q.update(route_forward=round(float(tab[0] + 0.3), 3), table_from_line=round(max(TFL_MIN, float(-tab[1] - 0.53)), 3))
                out[ep_s] = q; continue
            yaw = np.arctan2(*(cup_c - cart_c)[::-1]) - np.arctan2(rel0[1], rel0[0]); palm_c = np.array(c["palm_r"][:2])
            if PHYS:                                               # cup - palm at the closure in the 39 physical successes (base frame): (0.123, 0.067)
                byw = float(c["base_yaw"]); want = palm_c + np.array([np.cos(byw) * 0.123 - np.sin(byw) * 0.067, np.sin(byw) * 0.123 + np.cos(byw) * 0.067])
            else:
                want = palm_c + OFF_CUP
            Rm = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]]); cup_new = C0 + Rm @ (want - cart_c)
            cup_old = np.array(p["cup_xy"])
            if np.linalg.norm(cup_new - cup_old) > 0.02: kinds["cup_moved"] += 1; q["note"] = "cup re-placed from this run's hand path (cart frame)"
            else: cup_new = cup_old + nudge(q["rounds_failed"]); kinds["nudged"] += 1; q["note"] = f"cup nudged {nudge(q['rounds_failed']).tolist()}"
            near_lim = BOXe[0] + (float(p.get("box_from_handle", 0.09)) - 0.09)
            if PHYS:
                # 2026-09-28 (user: "the cup is stuck to the hand"): no box-top extension, and the cup stays clear of the right
                # hand's fingers while they hold the handle (cart frame, rows before the recorded grasp): cup radius + 2 cm
                near_lim = BOX[0] + sx                                 # the normal box top (no extension)
                pre = [x for x in tr if x["row"] < p["cup_close_row"] - 30]
                if pre:
                    tips_rel = [max(float(x["tips"]["right_hand_index_1_link"][0]), float(x["tips"]["right_hand_thumb_2_link"][0])) - (C0[0] + float(x["cart_xy"][0])) for x in pre]
                    clear = C0[0] + float(np.percentile(tips_rel, 99)) + 0.045 + 0.02
                    if clear > near_lim: near_lim = clear; q["note_clear"] = f"cup kept {clear - C0[0]:+.3f} m (cart frame) clear of the fingers on the handle"
            q["cup_xy"] = [round(float(np.clip(cup_new[0], near_lim, BOXe[1])), 3), round(float(np.clip(cup_new[1], BOXe[2], BOXe[3])), 3)]
            # the push is deterministic for a given cup position and changes little for a small cup shift: also re-place the
            # table from THIS run's release point so the next round can succeed in one go
            o = tr[int(np.abs(rows - p["cup_open_row"]).argmin())]; tab = np.array(o["palm_r"][:2]) + np.array([0.20, -0.38])
            q.update(route_forward=round(float(tab[0] + 0.3), 3), table_from_line=round(max(TFL_MIN, float(-tab[1] - 0.53)), 3))
        except Exception as e:
            q["note"] = f"kept ({e!r})"; kinds["kept"] += 1
        out[ep_s] = q
    n_to = int(re.sub(r"\D", "", a.to)); dst = a.out_file or (HERE / f"coffee_cart/replay/fit_psi0_{n_to}.json"); json.dump(out, open(dst, "w"), indent=1)
    print("coffee_cart", a.to, kinds, "->", dst)


def bottle(a):
    g = json.load(open(HERE / "bottle_bin/replay/episode_geometry_97.json"))["episodes"]
    n_from = int(re.sub(r"\D", "", a.frm)); jobs = {j["ep"]: j for j in json.load(open(HERE / f"bottle_bin/replay/jobs_psi0_fit{n_from}.json"))}
    res = {json.load(open(f))["ep"]: json.load(open(f)) for f in glob.glob(str(HERE / f"bottle_bin/replay/sweep/{a.frm}/ep*.json"))}
    OFF = np.array([0.098, 0.094]); BIN_OFF = np.array([0.03, -0.07]); NEAR_X = 0.155; R_BOT = 0.033
    out = []; kinds = {"success": 0, "bin_moved": 0, "bin_shifted": 0, "bottle_moved": 0, "nudged": 0, "kept": 0}
    def clean(r): return bool(r.get("in_bin")) and (r.get("robot_bin_leg_ticks") if r.get("robot_bin_leg_ticks") is not None else (r.get("robot_bin_contact_ticks") or 0)) <= 25 and (r.get("grasp_ok") or r.get("assist_captured") is not None)
    for ep, j0 in sorted(jobs.items()):
        r = res.get(ep); j = dict(j0); j.pop("variant", None)
        if r is None: kinds["kept"] += 1; out.append(j); continue
        j["rounds_failed"] = int(j0.get("rounds_failed", 0)) + (0 if clean(r) else 1)
        if clean(r): kinds["success"] += 1; out.append(j); continue
        if os.environ.get("FIT_PHYSICAL") == "1": j.pop("grasp_assist", None); j.pop("drop_assist", None)   # 2026-09-28: physical only
        else: j["grasp_assist"] = True                            # 2026-09-27: bottle rides with the palm from the recorded closure to the release
        if (r.get("robot_bin_contact_ticks") or 0) > 300 and (r.get("max_lift_m") or 0) < 0.02:   # the bin sits in the walk (a nonsense fit): back to the layout default
            j["bin_xy"] = [-0.30, -1.45]; j["note"] = "bin reset (blocked the walk)"; kinds["bin_moved"] += 1; out.append(j); continue
        if r.get("success"):                              # in the bin, but the walk ran into it: push the bin 0.12 m further along the heading at release
            bx, by, h = r["base_at_release"]; j["bin_xy"] = [round(float(v), 4) for v in (np.array(r["bin_xyz"][:2]) + 0.12 * np.array([np.cos(h), np.sin(h)]))]
            kinds["bin_shifted"] += 1; out.append(j); continue
        if r.get("grasp_ok") or r.get("assist_captured") is not None:   # assisted rides lift little: the landing point is what matters
            end = r.get("bottle_end"); rel = r.get("release"); shift = np.zeros(2)
            if r.get("base_at_release") and (r.get("robot_bin_contact_ticks") or 0) > 25:
                h = r["base_at_release"][2]; shift = 0.10 * np.array([np.cos(h), np.sin(h)])
            if rel and r.get("assist_captured") is not None: j["bin_xy"] = [round(float(v), 4) for v in (np.array(rel["xy"]) + BIN_OFF + shift)]   # ride: the bottle is in the fingers at the release
            elif end and end[2] < 0.5: j["bin_xy"] = [round(float(v), 4) for v in (np.array(end[:2]) + shift)]
            elif rel: j["bin_xy"] = [round(float(v), 4) for v in (np.array(rel["xy"]) + BIN_OFF + shift)]
            kinds["bin_moved"] += 1; out.append(j); continue
        tr = r["trace"]; t_c = g[str(ep)]["closure"]["t"]; i = int(np.argmin([abs(row["t"] - t_c) for row in tr]))
        palm_c = np.array(tr[i]["palm"][:2]); b0 = np.array(r["bottle_start"][:2]); want = palm_c + OFF
        if np.linalg.norm(want - b0) > 0.02: bxy = want; kinds["bottle_moved"] += 1
        else: bxy = b0 + nudge(j["rounds_failed"]); kinds["nudged"] += 1
        bxy[0] = max(bxy[0], NEAR_X + R_BOT + 0.01); j["bottle_xy"] = [round(float(v), 4) for v in bxy]
        rel = r.get("release")
        if rel and "bin_xy" not in j: j["bin_xy"] = [round(float(v), 4) for v in (np.array(rel["xy"]) + BIN_OFF)]
        out.append(j)
    n_to = int(re.sub(r"\D", "", a.to)); json.dump(out, open(HERE / f"bottle_bin/replay/jobs_psi0_fit{n_to}.json", "w"))
    print("bottle_bin", a.to, kinds, "->", f"jobs_psi0_fit{n_to}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--from", dest="frm", required=True); ap.add_argument("--to", required=True)
    ap.add_argument("--fits-file", default=None); ap.add_argument("--results-file", default=None); ap.add_argument("--out-file", default=None)
    a = ap.parse_args(); {"bowl_sink": bowl, "coffee_cart": coffee, "bottle_bin": bottle}[a.kit](a)
