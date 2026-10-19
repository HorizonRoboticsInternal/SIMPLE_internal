#!/usr/bin/env python3
"""Stage 2 of the replay export: turn a raw export folder (replay_meta.json + frames.npz + images/*.jpg, written by
replay_isaac.py --export inside Isaac) into a one-episode SIMPLE LeRobot dataset at <folder>/dataset using the Gr00t exporter.

    python sim/build_dataset.py data/replay_datasets/bowl_sink/ep_041 [more folders...]
"""
import os
import json, sys, time, shutil
from pathlib import Path
import numpy as np


def build(folder: Path) -> Path:
    import cv2
    from simple.cli.dr_decoupled_wbc import _init_exporter, _save_episode_env_config, _make_sonic_config
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    meta = json.load(open(folder / "replay_meta.json")); arrays = np.load(folder / "frames.npz")
    n = int(meta["frames"]); assert n == len(arrays["action"]), (n, len(arrays["action"]))
    cfg = _make_sonic_config(); waist = "lower_and_upper_body" if cfg.get("enable_waist", False) else "lower_body"
    rm = instantiate_g1_robot_model(waist_location=waist, high_elbow_pose=cfg.get("high_elbow_pose", False))
    out = folder / "dataset"
    if out.exists(): shutil.rmtree(out)
    exporter = _init_exporter(str(out), meta["instruction"], rm, meta["obj_names"], meta["joint_names"], tuple(meta["image_shape"]))
    keys = list(arrays.keys())
    for k in range(n):
        img = cv2.cvtColor(cv2.imread(str(folder / "images" / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB)
        frame = {"observation.images.ego_view": img}
        for key in keys:
            v = arrays[key][k]
            frame[key] = np.asarray(v, dtype=np.float32) if key == "observation.img_state_delta" else np.asarray(v, dtype=np.float64)
        exporter.add_frame(frame)
    time.sleep(0.5)
    exporter.save_episode()
    _save_episode_env_config(exporter, meta.get("env_config", {}), 0)
    exporter.stop_video_writers()
    (out / "replay_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "env_config"}, indent=1, default=str))
    return out


if __name__ == "__main__":
    for f in sys.argv[1:]:
        f = Path(f)
        try:
            out = build(f); print(f"[build] {f} -> {out}", flush=True)
        except Exception as exc:
            print(f"[build] {f} FAILED: {exc!r}", flush=True)
