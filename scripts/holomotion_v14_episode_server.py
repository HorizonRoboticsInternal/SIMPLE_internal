#!/usr/bin/env python
"""A stand-in VLA for simple.cli.eval_holomotion_v14: serves a recorded teleop episode's own controller inputs.

    .venv/bin/python scripts/holomotion_v14_episode_server.py <teleop dataset dir> --episode 12 --port 21000
    python -m simple.cli.eval_holomotion_v14 --scenes-from <teleop dataset dir> --episode-start 12 --num-episodes 1

Each POST /act returns the next --chunk rows of that episode (the HttpActionClient wire format): the reference frame
the controller had at each recorded step (teleop.latest_obs, 65) and the hand targets it applied (the replay log's
action_left/right_hand_q, MJCF order) -> rows of 79. After the last frame it repeats the last one. A request with
history.reset starts over. With the recorded scene (--scenes-from, same episode), the eval replays the teleop episode
open-loop through the eval's controller path: a check of the pipeline, not of a policy.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from simple.baselines.client import ResponseMessage, convert_numpy_in_dict, numpy_deserialize


def load_rows(dataset: Path, episode: int) -> np.ndarray:
    t = pq.read_table(dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet", columns=["teleop.latest_obs"]).to_pandas()
    ref = np.stack(t["teleop.latest_obs"].to_numpy()).astype(np.float64)
    log = dataset / "replay" / f"episode_{episode:06d}.npz"
    if not log.exists():
        return ref
    z = np.load(log)
    n = min(len(ref), len(z["action_left_hand_q"]))
    return np.concatenate([ref[:n], z["action_left_hand_q"][:n], z["action_right_hand_q"][:n]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=21000)
    ap.add_argument("--chunk", type=int, default=16, help="rows per reply")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before each reply (a slow VLA)")
    a = ap.parse_args()
    rows = load_rows(a.dataset, a.episode)
    print(f"[episode server] {a.dataset} episode {a.episode}: {len(rows)} rows of {rows.shape[1]} on :{a.port}", flush=True)
    cursors: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/info":
                self._send({"policy": f"teleop-episode-{a.episode}", "timestamp": "recorded"})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/act":
                self.send_error(404)
                return
            req = convert_numpy_in_dict(json.loads(self.rfile.read(int(self.headers["Content-Length"]))), numpy_deserialize)
            hist = req.get("history") or {}
            sid = str(hist.get("session_id", ""))
            if hist.get("reset") or sid not in cursors:
                cursors[sid] = 0
                img = next(iter(req["image"].values()))
                print(f"[episode server] session {sid}: image {getattr(img, 'shape', None)}, state keys {sorted(req['state'])}", flush=True)
            if a.delay > 0:
                import time
                time.sleep(a.delay)
            c = cursors[sid]
            idx = np.minimum(np.arange(c, c + a.chunk), len(rows) - 1)
            cursors[sid] = c + a.chunk
            self._send(ResponseMessage(action=rows[idx].astype(np.float32), err=0.0).serialize())

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
