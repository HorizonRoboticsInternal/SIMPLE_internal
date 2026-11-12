#!/usr/bin/env python
"""Align the bowl_sink level-3 eval set's counter/sink geometry with the BowlToSink_0918 recordings (2026-10-21, user:
"the eval scene sink location is different from the recording ... the eval scene should align the dataset scene").

The recordings (replay_datasets_phys/bowl_sink/ep_*/replay_meta.json scene_env, per-episode fits) put the bowl near the
counter's centre and the basin ~1.13 m behind the counter's front-edge line; the eval set used the kit defaults (bowl 0.74 m
right of the counter centre, next to the sink end; basin 1.27 m behind the edge). The robot start and the bowl are kept as
built (the fitted grasp pose); the counter, its distractors and the sink move so the bowl-to-counter and bowl-to-basin
relations take the recordings' medians:
    BOWL_SINK_ROBOT_DY   (counter + sink shift to the robot's right)
    BOWL_SINK_SINK_ALONG (counter front edge -> basin far edge)
Per scene: dr_state_dict.scene.table (the counter top is the DR "table") and the distractors move -DY in y with the counter;
a distractor closer than MIN_TO_BOWL to the bowl is slid along the counter (+y) until clear. Output: <set>_aligned_stage/
(meta/episodes.jsonl with the new configs, scene_env.json with the knobs, build notes) for rerender_level_set.py.
    .venv/bin/python scripts/level_sets/align_bowl_lv3.py [--set data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-3]
"""
import argparse, json, shutil, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = Path.home() / "wrk/robot_orchard_deploy/holobrain_g1_deploy/sim/bowl_sink"
BASIN_L, TOE, CNT_W, CNT_D, MIN_TO_BOWL, MIN_TO_OTHER, EDGE_MARGIN = 0.54, 0.125, 2.60, 0.63, 0.30, 0.12, 0.10

ap = argparse.ArgumentParser(); ap.add_argument("--set", default="data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-3"); ap.add_argument("--dataset", default="data/replay_datasets_phys/psi0/BowlToSink_0918_sim")
a = ap.parse_args(); SRC = ROOT / a.set; OUT = SRC.parent / (SRC.name + "_aligned_stage")
L = json.load(open(KIT / "layout.json"))["layout"]

# 1. the recordings' geometry (per-episode fitted knobs -> counter frame)
rec = []
for l in open(ROOT / a.dataset / "meta/episodes.jsonl"):
    e = json.loads(l); src = json.loads(e["source"]) if isinstance(e["source"], str) else e["source"]
    se = json.load(open(ROOT / f"data/replay_datasets_phys/bowl_sink/ep_{src['episode_index']:03d}/replay_meta.json"))["scene_env"]
    rte, dy, sa = float(se["BOWL_SINK_ROBOT_TO_EDGE"]), float(se["BOWL_SINK_ROBOT_DY"]), float(se["BOWL_SINK_SINK_ALONG"]); bx, by = map(float, se["BOWL_SINK_BOWL_XY"].split(","))
    rec.append(dict(bowl_left_of_counter_centre=by - (L["counter_cy"] - dy), basin_far_edge_behind_edge=sa, start_feet_to_edge=rte))
med_off, med_sa = st.median(r["bowl_left_of_counter_centre"] for r in rec), st.median(r["basin_far_edge_behind_edge"] for r in rec)

# 2. the set's scenes
eps = [json.loads(l) for l in open(SRC / "meta/episodes.jsonl")]
cfgs = [json.loads(e["environment_config"]) if isinstance(e["environment_config"], str) else e["environment_config"] for e in eps]
bowl_y = [c["dr_state_dict"]["spatial"]["bowl_15cm"]["position"][1] for c in cfgs]
DY = round(L["counter_cy"] - (st.median(bowl_y) - med_off), 3); SA = round(med_sa, 3)
cnt_cy = L["counter_cy"] - DY; near_x = L["near_x"]; basin = (near_x - SA + BASIN_L / 2, L["basin_cy"] - DY)
print(f"[align] recordings: bowl {med_off:+.3f} m left of the counter centre (median), basin far edge {med_sa:.3f} m behind the edge")
print(f"[align] knobs: BOWL_SINK_ROBOT_DY={DY} BOWL_SINK_SINK_ALONG={SA} -> counter centre y {cnt_cy:+.3f}, basin centre {basin[0]:+.3f},{basin[1]:+.3f}")

def dist(p, q): return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
notes = []
for i, c in enumerate(cfgs):
    ds = c["dr_state_dict"]; sp = ds["spatial"]
    ds["scene"]["table"]["pose"]["position"][1] = round(ds["scene"]["table"]["pose"]["position"][1] - DY, 4)
    for k in ("low", "high"):
        c["dr_cfgs"]["scene"]["table_position"][k][1] = round(c["dr_cfgs"]["scene"]["table_position"][k][1] - DY, 4)
        c["dr_cfgs"]["spatial"]["distractors_region"][k][1] = round(c["dr_cfgs"]["spatial"]["distractors_region"][k][1] - DY, 4)
    bowl = sp["bowl_15cm"]["position"]; moved = []
    others = []
    for k, v in sp.items():
        if k in ("g1_sonic", "bowl_15cm"): continue
        p = v["position"]; p[1] = round(p[1] - DY, 4); start = list(p)
        while dist(p, bowl) < MIN_TO_BOWL or any(dist(p, o) < MIN_TO_OTHER for o in others):
            p[1] = round(p[1] + 0.05, 4)
        p[1] = min(p[1], cnt_cy + CNT_W / 2 - EDGE_MARGIN)
        if p[1] != start[1]: moved.append(dict(uid=k, from_xy=[round(start[0], 3), round(start[1], 3)], to_xy=[round(p[0], 3), round(p[1], 3)], bowl_dist=round(dist(p, bowl), 3)))
        others.append(p)
        if "layout" in c and k in {v2.get("uid"): k2 for k2, v2 in c["layout"].get("actors", {}).items()}: pass
    # the informational layout block (rewritten by the re-render; kept consistent here)
    acts = c.get("layout", {}).get("actors", {})
    for name, act in acts.items():
        if name in ("table", "counter", "sink_counter") or name.startswith("distractor_"):
            uid = act.get("uid"); pos = act["pose"]["position"]
            if name.startswith("distractor_") and uid in sp: pos[0], pos[1] = sp[uid]["position"][0], sp[uid]["position"][1]
            else: pos[1] = round(pos[1] - DY, 4)
    notes.append(dict(scene=i, bowl_xy=[round(bowl[0], 3), round(bowl[1], 3)], bowl_left_of_counter_centre=round(bowl[1] - cnt_cy, 3),
                      basin_from_bowl=[round(basin[0] - bowl[0], 3), round(basin[1] - bowl[1], 3)], distractors_moved=moved))
    eps[i]["environment_config"] = json.dumps(c)

# 3. write the stage
if OUT.exists(): shutil.rmtree(OUT)
(OUT / "meta").mkdir(parents=True)
open(OUT / "meta/episodes.jsonl", "w").write("".join(json.dumps(e) + "\n" for e in eps))
for f in SRC.glob("meta/*.json*"):
    if f.name not in ("episodes.jsonl",): shutil.copy2(f, OUT / "meta" / f.name)
se = json.load(open(OUT / "meta/scene_env.json")); se["env"].update({"BOWL_SINK_ROBOT_DY": str(DY), "BOWL_SINK_SINK_ALONG": str(SA)})
se["geometry_aligned"] = dict(date="2026-10-21", reason="user: the eval sink location must match the BowlToSink_0918 recordings (replay_datasets_phys/psi0/BowlToSink_0918_sim)",
                              recordings_median=dict(bowl_left_of_counter_centre_m=round(med_off, 3), basin_far_edge_behind_counter_edge_m=round(med_sa, 3), start_feet_to_edge_m=round(st.median(r["start_feet_to_edge"] for r in rec), 3)),
                              before=dict(bowl_left_of_counter_centre_m=round(st.median(bowl_y) - L["counter_cy"], 3), basin_far_edge_behind_counter_edge_m=L["sink_along"]),
                              kept="robot start and bowl as built (the fitted grasp pose); lighting as stored", moved="counter top, distractors (with the counter; slid along it when within %.2f m of the bowl), sink counter and basin" % MIN_TO_BOWL,
                              scenes=notes)
json.dump(se, open(OUT / "meta/scene_env.json", "w"), indent=1)
n_moved = sum(len(n["distractors_moved"]) for n in notes)
print(f"[align] staged {len(eps)} scenes -> {OUT}; distractors slid along the counter: {n_moved}; bowl left of counter centre after: "
      f"{min(n['bowl_left_of_counter_centre'] for n in notes):+.3f}..{max(n['bowl_left_of_counter_centre'] for n in notes):+.3f} (recordings {min(r['bowl_left_of_counter_centre'] for r in rec):+.3f}..{max(r['bowl_left_of_counter_centre'] for r in rec):+.3f})")
