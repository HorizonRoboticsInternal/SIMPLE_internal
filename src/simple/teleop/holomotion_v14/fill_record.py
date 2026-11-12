"""--fill for simple.cli.replay_holomotion_v14 (2026-10-26).

A ``--lean-record`` teleop dataset holds states, actions and the controller's inputs, but no images and no tactile pads
(they cost most of the 20 ms control tick in a big scene). This module replays such a dataset bit-exactly and writes a
complete copy (``<dataset>_filled`` by default) in which every frame also carries

  * the recorded camera's images (observation.images.ego_view_left / ego_view_right, or ego_view), rendered exactly as the
    teleop would have: the rectified pinhole pair at 1280 x 720, area-averaged to the recorded size,
  * the wrist cameras (observation.images.wrist_left / wrist_right + wrist_cams.fresh) on the recording's 30 Hz schedule,
  * the Revo 2 fingertip tactile pads (tactile.*),

all taken on the frame's pre-step state, the state observation.state was sampled from (exact_log.replay_physics's
on_frame_start). On a full recording the recomputed columns are checked against the recorded ones instead (tactile:
max |diff|; videos: PSNR against the decoded recording).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

AUTO = ("timestamp", "frame_index", "episode_index", "index", "task_index")
NP = {"float32": np.float32, "float64": np.float64, "int64": np.int64, "int32": np.int32, "int16": np.int16, "int8": np.int8,
      "uint8": np.uint8, "bool": np.bool_}


def _jsonl(p: Path) -> list:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def default_out(dataset: Path) -> Path:
    return dataset.with_name(dataset.name + "_filled")


class Filler:
    def __init__(self, src: Path, out: Path, sonic_env, info: dict, check: bool = True) -> None:
        import cv2
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        from simple.cli.teleop_holomotion_v14 import _init_exporter
        from simple.teleop.holomotion_v14 import wrist_cams as WC
        from simple.teleop.holomotion_v14.hbvcam_views import FisheyeView, PinholeView

        self.cv2, self.src, self.out, self.env, self.check = cv2, Path(src), Path(out), sonic_env, check
        if self.out.exists() and any(self.out.iterdir()) and not os.environ.get("HOLOMOTION_FILL_APPEND"):   # (a replay over several recording setups appends to it)
            raise SystemExit(f"[fill] {self.out} exists and is not empty: pick another --fill-out")
        meta = self.src / "meta"
        self.info_json = json.loads((meta / "info.json").read_text())
        self.src_features = self.info_json["features"]
        self.lean = json.loads((meta / "lean_record.json").read_text()) if (meta / "lean_record.json").exists() else {}
        raw = json.loads((meta / "holomotion_raw.json").read_text()) if (meta / "holomotion_raw.json").exists() else {}
        tasks = _jsonl(meta / "tasks.jsonl")
        if not tasks:
            raise SystemExit(f"[fill] {self.src}: no meta/tasks.jsonl (no episode saved)")
        self.task_prompt = tasks[0]["task"]
        self.episodes_meta = {int(e["episode_index"]): e for e in _jsonl(meta / "episodes.jsonl")}
        self.record_camera = self.lean.get("record_camera") or info.get("record_camera", "pinhole")
        if self.record_camera not in ("pinhole", "fisheye"):
            raise SystemExit("[fill] only pinhole / fisheye recordings of the stereo camera are filled (record_camera head is the env's own render)")
        if self.lean:
            rec_w, rec_h = self.lean["record_size"]
        else:
            key = "observation.images.ego_view_left" if "observation.images.ego_view_left" in self.src_features else "observation.images.ego_view"
            h, w = self.src_features[key]["shape"][:2]
            rec_w, rec_h = (w // 2 if self.record_camera == "fisheye" else w), h
        self.rec_w, self.rec_h = int(rec_w), int(rec_h)
        hand = "revo2" if info.get("robot") == "revo2" else "dex3"
        m = sonic_env.mujoco.mjModel

        def src_has(prefix: str) -> bool:
            return any(k.startswith(prefix) for k in self.src_features)

        self.do_wrist = bool(WC.available(m) and (self.lean.get("wrist_cams", True) if self.lean else src_has("observation.images.wrist_")))
        self.do_tactile = bool(hand == "revo2" and (self.lean.get("tactile", True) if self.lean else src_has("tactile.")))
        self.wrist_hz = float(self.lean.get("wrist_hz", info.get("wrist_hz", 30.0)))
        wrist_fov = float(self.lean.get("wrist_fov", info.get("wrist_fov", 150.0)))

        robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        names = (self.src_features.get("observation.object_poses") or {}).get("names") or []
        obj_names = list(dict.fromkeys(n.rsplit(".", 1)[0] for n in names))
        joint_names = self.src_features["observation.state"]["names"]
        policy_obs_dim = int(raw.get("policy_obs_dim_max", 0))
        realformat = (meta / "realformat.json").exists()
        ego_shape = (self.rec_h, 2 * self.rec_w, 3) if self.record_camera == "fisheye" else (self.rec_h, self.rec_w, 3)
        self.exporter = _init_exporter(str(self.out), self.task_prompt, robot_model, obj_names, joint_names, ego_shape, realformat,
                                       right_eye=(self.record_camera == "pinhole"), policy_obs_dim=policy_obs_dim, hand=hand,
                                       wrist=self.do_wrist, wrist_fov=wrist_fov, images=True, tactile=self.do_tactile, probe=True)
        want = {k for k in self.exporter if k not in AUTO}
        have = {k for k in self.src_features if k not in AUTO}
        self.added = sorted(want - have)
        missing = sorted(have - want)
        passthrough = {k: dict(dtype=self.src_features[k]["dtype"], shape=tuple(int(v) for v in self.src_features[k]["shape"]), names=self.src_features[k].get("names"))
                       for k in missing if self.src_features[k]["dtype"] not in ("image", "video")}   # a plain column the source has and this version does not write (an optional
        if sorted(passthrough) != missing:                                                                # recorder column such as holomotion.pico_hands): copied through as it is
            raise SystemExit(f"[fill] the source has image columns this recorder version does not write: {sorted(set(missing) - set(passthrough))[:6]} -- cannot copy it")
        self.passthrough = sorted(passthrough)
        self.exporter = _init_exporter(str(self.out), self.task_prompt, robot_model, obj_names, joint_names, ego_shape, realformat,
                                       right_eye=(self.record_camera == "pinhole"), policy_obs_dim=policy_obs_dim, hand=hand,
                                       wrist=self.do_wrist, wrist_fov=wrist_fov, images=True, tactile=self.do_tactile, extra_features=passthrough)
        want = {k for k in self.exporter.features if k not in AUTO}
        for k in want & have:
            if [int(v) for v in self.src_features[k]["shape"]] != [int(v) for v in self.exporter.features[k]["shape"]]:
                raise SystemExit(f"[fill] column {k}: source shape {self.src_features[k]['shape']} vs {self.exporter.features[k]['shape']}")
        self.image_keys = [k for k, f in self.exporter.features.items() if f["dtype"] in ("image", "video")]
        self.recomputed = sorted(set(self.image_keys) | ({"wrist_cams.fresh"} if self.do_wrist else set())
                                 | {k for k in self.exporter.features if k.startswith("tactile.")})
        self.compare = sorted(set(self.recomputed) & have) if check else []
        self.view = PinholeView(sonic_env) if self.record_camera == "pinhole" else FisheyeView(sonic_env)
        self.wc = WC.WristCams(sonic_env, hz=self.wrist_hz, fov_deg=wrist_fov) if self.do_wrist else None
        self.r2 = None
        if self.do_tactile:
            from simple.teleop.holomotion_v14.revo2_record import Revo2Recorder
            self.r2 = Revo2Recorder(sonic_env, tactile=True)
        self.rows, self.n, self.src_episode = {}, 0, -1
        self.cmp_rows, self.cmp_video, self.stats = {}, {}, {}
        print(f"[fill] {self.src} -> {self.out}\n[fill] adds {len(self.added)} columns: images ({self.record_camera}, {self.rec_w} x {self.rec_h})"
              f"{', wrist cameras at ' + format(self.wrist_hz, 'g') + ' Hz' if self.do_wrist else ''}{', tactile pads' if self.do_tactile else ''}"
              + (f"; {len(self.compare)} recorded columns will be checked against the recomputed ones" if self.compare else ""), flush=True)

    # -- per episode -------------------------------------------------------------------------------------------------
    def begin(self, src_episode: int, n_frames: int) -> None:
        import pandas as pd
        chunk = src_episode // int(self.info_json.get("chunks_size", 1000))
        pq = self.src / self.info_json["data_path"].format(episode_chunk=chunk, episode_index=src_episode)
        df = pd.read_parquet(pq)
        if len(df) != n_frames:
            raise SystemExit(f"[fill] episode {src_episode}: {len(df)} rows in {pq.name} but {n_frames} frames in the exact log")
        self.rows = {k: df[k].to_numpy() for k in df.columns if k not in AUTO and k not in self.recomputed}
        self.src_episode, self.n = int(src_episode), int(n_frames)
        self.cmp_rows = {k: df[k].to_numpy() for k in self.compare if k in df.columns}
        for cap in self.cmp_video.values():
            cap.release()
        self.cmp_video = {}
        for k in self.compare:
            if k in self.image_keys:
                vp = self.src / self.info_json["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=src_episode)
                if vp.exists():
                    self.cmp_video[k] = self.cv2.VideoCapture(str(vp))
        self.stats = {k: [] for k in self.compare}
        if self.wc is not None:                                 # the wrist schedule starts fresh with the episode's frame 0
            self.wc._acc, self.wc._last = 1.0, None

    def _small(self, img):
        if (img.shape[1], img.shape[0]) == (self.rec_w, self.rec_h):
            return img
        return self.cv2.resize(img, (self.rec_w, self.rec_h), interpolation=self.cv2.INTER_AREA)

    def frame_start(self, t: int) -> None:
        """Before frame t's physics: the recorded row + the images and pads of this state."""
        F = self.exporter.features
        frame = {}
        for k, col in self.rows.items():
            f = F[k]
            frame[k] = np.asarray(col[t], dtype=NP[f["dtype"]]).reshape(tuple(int(v) for v in f["shape"]))
        if self.record_camera == "pinhole":
            left, right = self.view.render_pair()
            frame["observation.images.ego_view_left"] = self._small(left)
            frame["observation.images.ego_view_right"] = self._small(right)
        else:
            left, right = self.view.render()
            frame["observation.images.ego_view"] = np.hstack([self._small(left), self._small(right)])
        if self.wc is not None:
            w = self.wc.render()
            frame["observation.images.wrist_left"], frame["observation.images.wrist_right"] = w["left"], w["right"]
            frame["wrist_cams.fresh"] = np.array([1 if w.get("fresh", True) else 0], dtype=np.int8)
        if self.r2 is not None:
            frame.update(self.r2.tactile_sample())
        for k in self.compare:                                  # a full recording: how far the recomputation is from it
            if k in self.cmp_video:
                ok, img = self.cmp_video[k].read()
                if ok:
                    a, b = img[..., ::-1].astype(np.float64), np.asarray(frame[k], dtype=np.float64)
                    if a.shape == b.shape:
                        mse = float(np.mean((a - b) ** 2))
                        self.stats[k].append(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
            elif k in self.cmp_rows:
                rec = np.asarray(self.cmp_rows[k][t], dtype=np.float64).reshape(np.shape(frame[k]))
                self.stats[k].append(float(np.max(np.abs(rec - np.asarray(frame[k], dtype=np.float64)))))
        self.exporter.add_frame(frame)

    def end(self, res: dict) -> list[str]:
        from simple.cli.teleop_holomotion_v14 import _save_episode_meta
        ep = int(self.exporter.episode_buffer["episode_index"])
        self.exporter.save_episode()
        src_meta = self.episodes_meta.get(self.src_episode, {})
        extra = {k: v for k, v in src_meta.items() if k not in ("episode_index", "tasks", "length")}
        extra["fill"] = dict(source=str(self.src), source_episode=self.src_episode, bit_exact=bool(res.get("exact")), frames=self.n, added=self.added)
        _save_episode_meta(self.exporter, ep, extra)
        for f in (self.src / "replay").glob(f"episode_{self.src_episode:06d}.*"):
            dst = self.out / "replay" / f"episode_{ep:06d}{f.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        summary = []
        for k, vals in self.stats.items():
            if not vals:
                continue
            short = k.replace("observation.images.", "")
            if k in self.cmp_video:
                a = np.asarray(vals)
                summary.append(f"{short} PSNR median {np.median(a):.1f} dB, min {a.min():.1f}")
            else:
                summary.append(f"{short} max|diff| {max(vals):.3g}")
        if ep != self.src_episode:
            print(f"[fill] note: source episode {self.src_episode} is episode {ep} of the filled dataset")
        print(f"[fill] episode {ep}: {self.n} frames written, {'bit-exact' if res.get('exact') else 'NOT bit-exact (images of a diverged replay)'}"
              + ("; recorded vs recomputed: " + "; ".join(summary) if summary else ""), flush=True)
        return summary

    def finish(self) -> None:
        src_raw, out_raw = self.src / "meta" / "holomotion_raw.json", self.out / "meta" / "holomotion_raw.json"
        if src_raw.exists() and out_raw.exists():
            s, o = json.loads(src_raw.read_text()), json.loads(out_raw.read_text())
            if "obs_layout_live" in s:
                o["obs_layout_live"] = s["obs_layout_live"]
                out_raw.write_text(json.dumps(o, indent=2))
        (self.out / "meta" / "lean_record.json").write_text(json.dumps({**self.lean, "lean_record": False, "filled": True, "filled_from": str(self.src),
                                                                        "added": self.added}, indent=2))
        for cap in self.cmp_video.values():
            cap.release()
        for v in (self.view, self.wc):
            if v is not None and hasattr(v, "close"):
                v.close()
        print(f"[fill] done: {self.out}", flush=True)
