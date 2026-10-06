#!/usr/bin/env python
"""Pipeline check of simple.cli.eval_holomotion_v14 with a REAL model in the loop: the HoloBrain G1 deploy server
(holobrain_g1_deploy, scripts/serve.sh <preset>, SIMPLE's PSI0 /act protocol) behind a bridge that turns its
decoupled-WBC rows into the reference frames the HoloMotion v1.4 controller tracks.

    .venv/bin/python scripts/holomotion_v14_vla_bridge.py --upstream-port 8014 --port 21000 --preset chipcan_nativec9 \
        --log <dir>/bridge_bottle_bin.jsonl
    python -m simple.cli.eval_holomotion_v14 --scene bottle_bin --port 21000 --image-size 640x360

What flows (every eval query):
  eval -> bridge   image observation.images.ego_view (the HBVCAM rectified left eye), instruction, state
                   (observation.state 43 joints, observation.base_pose, ...), history (session_id, reset, step)
  bridge -> model  the PSI0 request SIMPLE's psi0_decoupled_wbc agent sends: image rgb_head_stereo_left, states (1, 32)
                   = Dex3 hands 14 (L thumb0-2, middle0-1, index0-1; R thumb0-2, index0-1, middle0-1), arms 14 (motors
                   15..28), waist roll/pitch/yaw, last commanded base height
  model -> bridge  action (24, 36) at 50 Hz: hands 14 (same order), arms 14, torso roll/pitch/yaw, height, vx vy vyaw
                   target_yaw (the server interpolates its chunk to SIMPLE's 24 frames)
  bridge -> eval   action (24, 79): per row a HoloMotion reference frame latest_obs[65] + the 14 hand targets
                   (SIMPLE MJCF order), the layout agents/holomotion_v14_vla_agent.py accepts

The reference frame is FAKED from the model's intent -- these models command a decoupled WBC (upper-body targets +
a walking command), not a whole-body motion:
  dof_pos  legs 0..11 = HoloMotion's default standing angles; waist 12..14 = the model's torso yaw/roll/pitch;
           arms 15..28 = the model's arm targets (the same motor order)
  dof_vel  finite differences of dof_pos along the chunk (legs 0)
  root     integrated from the model's vx, vy (heading frame) and vyaw, 20 ms per row, starting at the robot's base
           pose on the session's first query; z = that pose's z + (height command - its first value)
  rot      yaw-only quaternion (w, 0, 0, z) of the integrated heading
So the controller walks where the model's velocity command points and holds the model's arm/waist pose -- a check of
the pipeline (model in the loop, protocol, timing, outputs), not of the model.

GET /info reports the preset and a start stamp (the eval names its output folder after them); GET /health proxies
the model server. --log writes one JSON line per query (latency, the model's command, the frames' validity).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests
import yaml


from simple.baselines.client import HttpActionClient, ResponseMessage, convert_numpy_in_dict, numpy_deserialize  # noqa: E402

from simple.teleop.holomotion_v14 import ROBOT_CONFIG
ROBOT_YAML = ROBOT_CONFIG
DT = 0.02
# SIMPLE joint_qpos (43) -> the PSI0 hand14 (Dex3 order) + arm14
HAND_FROM_QPOS = [29, 30, 31, 34, 35, 32, 33, 36, 37, 38, 39, 40, 41, 42]
ARM_FROM_QPOS = list(range(15, 29))
# the model's hand14 (Dex3 order) -> SIMPLE MJCF hands (L thumb0-2 index0-1 middle0-1, R the same)
LEFT_MJCF_FROM_MODEL = [0, 1, 2, 5, 6, 3, 4]
RIGHT_MJCF_FROM_MODEL = [7, 8, 9, 10, 11, 12, 13]


def yaw_of(q_wxyz) -> float:
    w, x, y, z = (float(v) for v in q_wxyz)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Bridge:
    def __init__(self, upstream: HttpActionClient, preset: str, log_path: Path | None) -> None:
        self.up, self.preset = upstream, preset
        self.stamp = time.strftime("%Y%m%dT%H%M%S")
        y = yaml.safe_load(open(ROBOT_YAML))
        self.dof_names = list(y["complete_dof_order"])
        self.default = np.array([float(y["default_joint_angles"][n]) for n in self.dof_names])
        self.sessions: dict[str, dict] = {}
        self.n = 0
        self.log = open(log_path, "a") if log_path else None

    # ---- the fake reference
    def frames(self, sess: dict, rows: np.ndarray) -> np.ndarray:
        T = len(rows)
        dof = np.tile(self.default, (T, 1))
        dof[:, 12] = rows[:, 30]                      # waist yaw
        dof[:, 13] = rows[:, 28]                      # waist roll
        dof[:, 14] = rows[:, 29]                      # waist pitch
        dof[:, 15:29] = rows[:, 14:28]                # arms, motors 15..28
        vel = np.gradient(dof, DT, axis=0) if T > 1 else np.zeros_like(dof)
        vel[:, :12] = 0.0
        root = np.zeros((T, 3))
        rot = np.zeros((T, 4))
        x, y, z0, yaw = sess["root"]
        for t in range(T):
            vx, vy, vyaw = (float(v) for v in rows[t, 32:35])
            x += (vx * np.cos(yaw) - vy * np.sin(yaw)) * DT
            y += (vx * np.sin(yaw) + vy * np.cos(yaw)) * DT
            yaw += vyaw * DT
            root[t] = [x, y, z0 + float(rows[t, 31]) - sess["h0"]]
            rot[t] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        sess["root"] = (x, y, z0, yaw)
        sess["h_cmd"] = float(rows[-1, 31])
        hands = np.concatenate([rows[:, LEFT_MJCF_FROM_MODEL], rows[:, RIGHT_MJCF_FROM_MODEL]], axis=1)
        return np.concatenate([dof, vel, root, rot, hands], axis=1).astype(np.float32)   # (T, 79)

    # ---- one eval query
    def act(self, payload: dict) -> np.ndarray:
        t0 = time.perf_counter()
        img = np.asarray(next(iter(payload["image"].values())))
        st = payload["state"]
        hist = payload.get("history") or {}
        sid = str(hist.get("session_id", "default"))
        q = np.asarray(st["observation.state"], dtype=np.float64).reshape(-1)
        base = np.asarray(st["observation.base_pose"], dtype=np.float64).reshape(-1)
        if hist.get("reset") or sid not in self.sessions:
            self.sessions[sid] = dict(root=(float(base[0]), float(base[1]), float(base[2]), yaw_of(base[3:7])),
                                      h0=0.74, h_cmd=0.74, queries=0)
        sess = self.sessions[sid]
        states = np.concatenate([q[HAND_FROM_QPOS], q[ARM_FROM_QPOS], [q[13], q[14], q[12]], [sess["h_cmd"]]]).astype(np.float32)[None]
        t1 = time.perf_counter()
        rows, err, _ = self.up.query_action({"rgb_head_stereo_left": img}, payload.get("instruction") or "", {"states": states}, {},
                                            history={"reset": bool(hist.get("reset")), "session_id": sid,
                                                     "step_index": int(hist.get("step_index", 0))}, dataset="simple")
        t_model = time.perf_counter() - t1
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 36 or not np.isfinite(rows).all():
            raise ValueError(f"model reply {rows.shape}, finite={bool(np.isfinite(rows).all())}")
        if sess["queries"] == 0:
            sess["h0"] = float(rows[0, 31])
        out = self.frames(sess, rows)
        sess["queries"] += 1
        self.n += 1
        rec = dict(n=self.n, session=sid, step=int(hist.get("step_index", -1)), reset=bool(hist.get("reset")),
                   image=list(img.shape), model_s=round(t_model, 4), total_s=round(time.perf_counter() - t0, 4),
                   rows=int(len(rows)), cmd=dict(vx=round(float(rows[:, 32].mean()), 3), vy=round(float(rows[:, 33].mean()), 3),
                   vyaw=round(float(rows[:, 34].mean()), 3), height=round(float(rows[:, 31].mean()), 3),
                   right_hand_closed=bool(rows[:, 7:14].mean() > 0.5)),
                   out=dict(shape=list(out.shape), finite=bool(np.isfinite(out).all()),
                            quat_norm=round(float(np.linalg.norm(out[:, 61:65], axis=1).mean()), 4)))
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8014)
    ap.add_argument("--port", type=int, default=21000)
    ap.add_argument("--preset", default="holobrain")
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()
    up = HttpActionClient(a.upstream_host, a.upstream_port)
    health = requests.get(f"http://{a.upstream_host}:{a.upstream_port}/health", timeout=5).json()
    bridge = Bridge(up, a.preset, a.log)
    print(f"[bridge] model server :{a.upstream_port} = {Path(str(health.get('run_dir', '?'))).name} {health.get('ckpt_step')}; "
          f"serving the HoloMotion eval on :{a.port} as policy {a.preset}-{bridge.stamp}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/info":
                self._send({"policy": a.preset, "timestamp": bridge.stamp, "model": {k: health.get(k) for k in ("run_dir", "ckpt_step", "instruction_override")},
                            "queries": bridge.n})
            elif self.path == "/health":
                self._send(requests.get(f"http://{a.upstream_host}:{a.upstream_port}/health", timeout=5).json())
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/act":
                self.send_error(404)
                return
            payload = convert_numpy_in_dict(json.loads(self.rfile.read(int(self.headers["Content-Length"]))), numpy_deserialize)
            try:
                out = bridge.act(payload)
            except Exception as e:  # noqa: BLE001
                print(f"[bridge] /act failed: {type(e).__name__}: {e}", flush=True)
                self._send({"status": "error", "detail": str(e)}, 500)
                return
            self._send(ResponseMessage(action=out, err=0.0).serialize())

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
