"""Per-episode summary + failure cause of a replay_rot0_revo2.py run (results.json with the per-frame timeline:
bottle_z / bottle_pos / bottle_tilt / hand / grip / bottle_contacts), for the tuner page (docs/revo2_pose_tuner) and the
apply script. Also extracts the end frame of every video as the card poster (videos/ep<N>_end.jpg).

Outcome labels (one per episode; "placed" when the task's whole-bottle-in-bin gate fired):
  bottle knocked over before the grasp     tilted > 30 deg before the grip bit closed
  bottle pushed away before the grasp      moved > 4 cm on the table before the grip bit closed
  hand closed beside the bottle            grip closed, the right hand never touched the bottle afterwards
  grasp missed, bottle never lifted        hand touched it but it never left the table by 2 cm
  slipped out, bottle back on the table    left the closed hand, ended on the table
  slipped out during the carry             left the closed hand away from the bin, ended on the floor
  slipped out at the bin                   left the closed hand within 30 cm of the bin, not inside
  released before the bin                  grip opened with the bottle not over the bin
  released over the bin, lies across the rim / leaning out of the bin / ended beside the bin
  still in the hand at the end             never let go
    python revo2_replay_summary.py <replay dir>      (prints the table; the apply script and add_variant.py import summarise)
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

HZ = 50.0                         # replay / render rate (frames per second)
BIN_W, BIN_D, BIN_H, BIN_T = 0.36, 0.24, 0.39, 0.008   # build_scene.py: inner opening 0.344 x 0.224 m, rim 0.39 m above the bin base
TABLE_NAMES = ("table",)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def end_frame(video: Path, out: Path) -> bool:
    if out.is_file(): return True
    if not video.is_file(): return False
    r = subprocess.run([_ffmpeg(), "-loglevel", "error", "-y", "-sseof", "-0.3", "-i", str(video), "-frames:v", "1", "-q:v", "4", str(out)], capture_output=True)
    return r.returncode == 0 and out.is_file()


def bin_frame(bin_info):
    """(centre xy, local x axis, local y axis, rim z) from the logged bin geoms (yaw-agnostic: the long walls sit ~0.116 m
    from the centre along the bin's local x)."""
    c = np.asarray(bin_info["body_pos"], float)
    ax = None
    for g in bin_info.get("geoms", []):
        off = np.asarray(g["pos"], float)[:2] - c[:2]; n = float(np.linalg.norm(off))
        if 0.09 <= n <= 0.14 and g["size"][2] > 0.05: ax = off / n; break
    if ax is None: ax = np.array([1.0, 0.0])
    ay = np.array([-ax[1], ax[0]])
    return c, ax, ay, float(c[2]) + BIN_H


def in_bin_xy(p, c, ax, ay, margin=0.0):
    d = np.asarray(p, float)[:2] - c[:2]
    return abs(float(d @ ax)) <= BIN_D / 2 - BIN_T + margin and abs(float(d @ ay)) <= BIN_W / 2 - BIN_T + margin


def _hum(body):                       # right_hand_thumb_proximal_link -> thumb proximal
    return body.replace("right_hand_", "").replace("_link", "").replace("_", " ")


def classify(e: dict, bin_info: dict | None) -> dict:
    """Returns dict(cause, detail, lost_s, end_state) for one episode row (cause None when placed)."""
    tl = e.get("timeline") or {}
    z = np.asarray(tl.get("bottle_z", []), float); hand = np.asarray(tl.get("hand", []), int); grip = np.asarray(tl.get("grip", []), int)
    tilt = np.asarray(tl.get("bottle_tilt", []), float); pos = np.asarray(tl.get("bottle_pos", []), float); con = tl.get("bottle_contacts", [])
    n = len(z); gates = e.get("gates") or {}
    if n == 0: return dict(cause=None if e.get("success") else "not placed", detail="no timeline in results.json", lost_s=None, end_state=None)
    have_pose = len(tilt) == n and len(pos) == n
    if not have_pose: tilt = np.zeros(n); pos = np.zeros((n, 3)); pos[:, 2] = z
    z0 = float(e.get("bottle_z0", z[0])); lift = z - z0
    ic = int(np.argmax(grip > 0)) if (grip > 0).any() else None
    c = ax = ay = rim_z = None
    if bin_info: c, ax, ay, rim_z = bin_frame(bin_info)
    table = np.array([("table" in ci) for ci in con] + [False] * (n - len(con)), bool)
    rhand = [[nm for nm in ci if nm.startswith("right_hand")] for ci in con] + [[]] * (n - len(con))
    bin_d = lambda i: float(np.hypot(*(pos[i, :2] - c[:2]))) if c is not None and have_pose else float("nan")
    inside_bin = lambda i: c is not None and have_pose and in_bin_xy(pos[i], c, ax, ay) and float(pos[i, 2]) < rim_z - 0.10
    end_con = set(con[-1]) if con else set(); end_tilt = float(tilt[-1]); end_d = bin_d(n - 1)
    if np.isnan(end_d):
        er = np.asarray(e.get("bottle_end_minus_bin_m", [9, 9, 9]), float); end_d = float(np.hypot(er[0], er[1]))
    if inside_bin(n - 1): end_state = "in the bin" + (", lying on its side" if end_tilt > 45 else "")
    elif "table" in end_con: end_state = "back on the table"
    elif "trash_bin" in end_con and end_tilt > 45 and (rim_z is None or float(pos[-1, 2]) > 0.5 * BIN_H): end_state = "lying across the rim"
    elif "trash_bin" in end_con: end_state = "leaning against the outside of the bin"
    elif end_d < 0.35: end_state = "on the floor beside the bin"
    else: end_state = f"on the floor {end_d:.1f} m from the bin"
    if e.get("success"):
        return dict(cause=None, detail=f"placed at {e['success_frame'] / HZ:.1f} s" if e.get("success_frame") is not None else "placed", lost_s=None, end_state="in the bin")
    if ic is None:
        return dict(cause="grip never closed", detail=f"the grip bit never went above the threshold in this recording; ended {end_state}", lost_s=None, end_state=end_state)
    t_close = ic / HZ
    ft = next((i for i in range(ic) if rhand[i]), None)
    first_touch = f" (first touched by the {_hum(sorted(rhand[ft], key=len)[0])} at {ft / HZ:.1f} s)" if ft is not None else ""
    if have_pose and ic > 0:
        tp = float(tilt[:ic].max()); it = int(np.argmax(tilt[:ic]))
        if tp > 30:
            return dict(cause="bottle knocked over before the grasp", detail=f"the bottle tipped to {tp:.0f} deg at {it / HZ:.1f} s{first_touch}; the grip closed at {t_close:.1f} s; ended {end_state}", lost_s=None, end_state=end_state)
        mv = float(np.linalg.norm(pos[ic, :2] - pos[0, :2]))
        if mv > 0.04 and lift[:ic].max() < 0.02:
            return dict(cause="bottle pushed away before the grasp", detail=f"the bottle moved {100 * mv:.0f} cm on the table before the grip closed at {t_close:.1f} s{first_touch}; ended {end_state}", lost_s=None, end_state=end_state)
    wr = e.get("bottle_in_wrist_at_close")
    where = f"; at the close the bottle base was {100 * wr[0]:.0f} cm along the forearm, {100 * wr[1]:.0f} cm to the palm side and {100 * abs(wr[2]):.0f} cm {'below' if wr[2] < 0 else 'above'} the wrist" if wr else ""
    carried = [i for i in range(ic, n) if hand[i] and not table[i]]
    if not carried:
        touch_s = float(hand[ic:].sum() / HZ)
        if touch_s < 0.1:
            return dict(cause="hand closed beside the bottle", detail=f"grip closed at {t_close:.1f} s, the right hand never touched the bottle afterwards{where}; ended {end_state}", lost_s=None, end_state=end_state)
        return dict(cause="grasp did not lift the bottle", detail=f"the hand touched the bottle for {touch_s:.1f} s after the close but it never left the table (max lift {100 * float(lift[ic:].max()):.1f} cm){where}; ended {end_state}", lost_s=None, end_state=end_state)
    i_first = carried[0]; i_last = int(np.where(hand > 0)[0].max())
    if i_last >= n - 2:
        return dict(cause="still in the hand at the end", detail=f"the bottle was still in the right hand when the clip ended (max lift {100 * float(lift.max()):.0f} cm)", lost_s=None, end_state="in the hand")
    lost_s = i_last / HZ; hold_s = (i_last - i_first) / HZ; grip_at = int(grip[min(i_last + 2, n - 1)])
    d_lost = bin_d(i_last); h_lost = (float(pos[i_last, 2]) - rim_z) if rim_z is not None else float("nan"); tilt_lost = float(tilt[i_last])
    at = f"at {lost_s:.1f} s" + ("" if np.isnan(d_lost) else f", {100 * d_lost:.0f} cm from the bin centre") + ("" if np.isnan(h_lost) else f", {100 * h_lost:.0f} cm above the rim") + (f", bottle tilted {tilt_lost:.0f} deg" if have_pose else "")
    gate_note = "" if gates.get("grasped") else "; the task's grasp gate never fired (it needs the bottle held upright, under 60 deg, off the table for 0.5 s)"
    if end_state.startswith("in the bin"):
        cause = "in the bin, not scored: carried tilted, no valid grasp" if not gates.get("grasped") else "in the bin, not scored"
        return dict(cause=cause, detail=f"the bottle ended {end_state}; carried {hold_s:.1f} s, let go {at}; placed needs a valid grasp first and 0.5 s in the bin free of the hand{gate_note}", lost_s=lost_s, end_state=end_state)
    if grip_at:
        if hold_s < 0.5: cause, verb = "grasp did not hold, dropped within 0.5 s", "fell out of the closed hand"
        elif not np.isnan(d_lost) and d_lost < 0.45: cause, verb = "slipped out at the bin", "slipped out of the closed hand"
        else: cause, verb = "slipped out during the carry", "slipped out of the closed hand"
    else:
        over = c is not None and have_pose and in_bin_xy(pos[i_last], c, ax, ay, 0.05)
        if over: cause, verb = ("released tilted over the bin" if tilt_lost >= 25 else "released over the bin, bounced out"), "released over the bin"
        elif not np.isnan(d_lost) and d_lost < 0.45: cause, verb = "released next to the bin", "the grip opened"
        else: cause, verb = "released before the bin", "the grip opened"
    return dict(cause=cause, detail=f"carried {hold_s:.1f} s; {verb} {at}; ended {end_state}{gate_note}", lost_s=lost_s, end_state=end_state)


def summarise(rdir: Path, root: Path, posters: bool = True) -> dict:
    rdir = Path(rdir); res = json.load(open(rdir / "results.json"))
    bin_info = res.get("bin") if isinstance(res, dict) else None
    eps = res["episodes"] if isinstance(res, dict) else res
    per = []
    for r in eps:
        cl = classify(r, bin_info)
        vid = Path(r["video"]).name if r.get("video") else None
        poster = None
        if vid and posters:
            pf = rdir / "videos" / f"ep{r['episode']}_end.jpg"
            if end_frame(rdir / "videos" / vid, pf): poster = f"{rdir.name}/videos/{pf.name}"
        per.append(dict(episode=r["episode"], placed=bool(r.get("success")), cause=cl["cause"], detail=cl["detail"], end_state=cl["end_state"], lost_s=None if cl["lost_s"] is None else round(cl["lost_s"], 1),
                        lift_cm=round(100 * float(r.get("bottle_max_lift_m", 0)), 1), hand_s=round(float(r.get("hand_on_bottle_s", 0)), 2), grip_close_s=r.get("grip_close_s"),
                        grasped=bool((r.get("gates") or {}).get("grasped")), at_bin=bool((r.get("gates") or {}).get("at_bin")),
                        video=f"{rdir.name}/videos/{vid}" if vid else None, poster=poster, still=f"{rdir.name}/{r['grasp_still']}" if r.get("grasp_still") else None))
    causes = {}
    for p in per: causes.setdefault("placed" if p["placed"] else p["cause"], []).append(p["episode"])
    try: rel = str(rdir.resolve().relative_to(Path(root).resolve()))
    except ValueError: rel = str(rdir)
    return dict(dir=rel, episodes=len(per), placed=sum(p["placed"] for p in per), grasped=sum(p["grasped"] for p in per), held_0p5s=sum(p["hand_s"] >= 0.5 for p in per),
                causes=[dict(cause=k, count=len(v), episodes=v) for k, v in sorted(causes.items(), key=lambda kv: (kv[0] != "placed", -len(kv[1])))], per_episode=per)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]; s = summarise(Path(sys.argv[1]), root, posters="--no-posters" not in sys.argv)
    print(f"{s['dir']}: {s['placed']}/{s['episodes']} placed, {s['grasped']} grasp gate, {s['held_0p5s']} held >= 0.5 s")
    for c in s["causes"]: print(f"  {c['count']:3d}  {c['cause']}: {c['episodes']}")
    if "-v" in sys.argv:
        for p in s["per_episode"]: print(f"  ep {p['episode']:2d} {'PLACED' if p['placed'] else p['cause']}: {p['detail']}")
