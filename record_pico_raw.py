#!/usr/bin/env python
"""Record raw PICO / XRoboToolkit (xrt) controller input to a local file.

Taps the same XrClient the teleop stack uses, but bypasses PicoStreamer so
nothing is consumed, filtered, or integrated away: every sample holds the raw
SDK values AND the derived teleop command computed from them, so a later
analysis can see exactly what the dead zone and the walk threshold did to the
operator's input.

    record:   python record_pico_raw.py [--hz 50] [--duration 120] [--hands]
    analyze:  python record_pico_raw.py --analyze data/pico_raw/<file>.jsonl

Output is JSONL, one flat object per sample, flushed continuously — a crash or
Ctrl+C keeps everything recorded so far. Load with:

    import pandas as pd
    df = pd.read_json("data/pico_raw/<file>.jsonl", lines=True)

Requires the PICO service to be running. If you are not already in a teleop
session, pass --start-service (or run /opt/apps/roboticsservice/runService.sh
yourself); do NOT pass it while teleop is live, it starts a second copy.
"""
import argparse
import json
import math
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data/pico_raw"

# Mirrors the constants in pico_streamer._generate_unified_raw_data (they are
# function-locals there, so they cannot be imported).
DEAD_ZONE = 0.1
MAX_LINEAR_VEL = 0.5   # m/s
MAX_ANGULAR_VEL = 1.0  # rad/s
# Balance/Walk ONNX switch in g1_gear_wbc_policy.get_action(). Measured onset:
# nothing below this moves the robot (see cmd_threshold_report.html).
WALK_THRESHOLD = 0.1

BUTTONS = ["A", "B", "X", "Y", "left_menu_button", "right_menu_button",
           "left_axis_click", "right_axis_click"]
KEYS = ["left_trigger", "right_trigger", "left_grip", "right_grip"]
POSES = ["headset", "left_controller", "right_controller"]


def _local_dead_zone(value, dead_zone):
    """Byte-for-byte copy of PicoStreamer._apply_dead_zone (rescales, not clips)."""
    if abs(value) < dead_zone:
        return 0.0
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - dead_zone) / (1.0 - dead_zone)


def get_dead_zone_fn():
    """Prefer the real implementation so this can never drift from the stack."""
    try:
        from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer
        fn = lambda v, dz: PicoStreamer._apply_dead_zone(None, v, dz)  # noqa: E731
        for probe in (0.0, 0.05, 0.1, 0.5, -0.37, 1.0):
            assert abs(fn(probe, DEAD_ZONE) - _local_dead_zone(probe, DEAD_ZONE)) < 1e-12
        return fn, "pico_streamer"
    except Exception as e:  # noqa: BLE001
        print(f"[rec] using local dead-zone copy ({type(e).__name__}: {e})")
        return _local_dead_zone, "local"


SERVICE_PORT = 60061  # RoboticsServiceProcess <-> xrobotoolkit_sdk


class KeyMarker:
    """Read single keypresses from the terminal without waiting for Enter.

    cbreak leaves ISIG on, so Ctrl+C still raises SIGINT normally. Degrades to
    a no-op when stdin is not a tty (piped, nohup, systemd), so the recorder
    still runs headless -- only spacebar marking is lost.
    """

    def __init__(self):
        self.fd = None
        self.saved = None

    def __enter__(self):
        try:
            if sys.stdin.isatty():
                self.fd = sys.stdin.fileno()
                self.saved = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
        except Exception:  # noqa: BLE001
            self.fd = None
        return self

    @property
    def active(self):
        return self.fd is not None

    def poll(self):
        """Return the pending keypress, or None. Never blocks."""
        if self.fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *exc):
        if self.fd is not None and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False


def port_listening(port):
    """True if anything LISTENs on `port`. Reads /proc — opens no connection,
    so it cannot disturb the service's own client session."""
    want = f"{port:04X}"
    for proc in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in Path(proc).read_text().splitlines()[1:]:
                f = line.split()
                if len(f) > 3 and f[3] == "0A" and f[1].split(":")[1].upper() == want:
                    return True
        except OSError:
            pass
    return False


def preflight(client, force, timeout=5.0):
    """The SDK returns zeros forever when the service is absent, instead of
    failing. Verify the device clock actually advances before recording.

    The stream needs ~0.6 s after init() before the first frame lands, so this
    polls until the clock moves rather than sampling once -- a single early
    probe reports a perfectly healthy headset as dead.
    """
    deadline = time.monotonic() + timeout
    t_start = time.monotonic()
    prev, live, waited = None, False, 0.0
    while time.monotonic() < deadline:
        ts = client.get_timestamp_ns()
        if ts and prev and ts != prev:
            live, waited = True, time.monotonic() - t_start
            break
        if ts:
            prev = ts
        time.sleep(0.05)
    listening = port_listening(SERVICE_PORT)

    print(f"[rec] preflight: service on :{SERVICE_PORT} "
          f"{'LISTENING' if listening else 'NOT LISTENING'} | "
          f"device clock {f'advancing after {waited:.2f}s' if live else 'STOPPED'}")
    if live:
        head = None
        try:
            head = [float(v) for v in client.get_pose_by_name("headset")]
        except Exception:  # noqa: BLE001
            pass
        if head and any(abs(v) > 1e-9 for v in head[:3]):
            print(f"[rec] headset tracking live at "
                  f"({head[0]:+.2f}, {head[1]:+.2f}, {head[2]:+.2f})")
        else:
            print("[rec] NOTE: headset pose is all zeros — tracking may not be running")
        return True

    print("[rec] ERROR: no live PICO data — every sample would be zeros.")
    if not listening:
        print("[rec]   RoboticsServiceProcess is not running. Start it with")
        print("[rec]   --start-service, or: bash /opt/apps/roboticsservice/runService.sh &")
    else:
        print("[rec]   Service is up but the headset is not streaming to it.")
        print("[rec]   Check the headset is awake, on this LAN, and paired to this PC.")
    if force:
        print("[rec]   --force given, recording anyway.")
        return True
    print("[rec]   Pass --force to record regardless.")
    return False


def stick_for_command(cmd_value, max_vel):
    """Inverse of the dead zone: stick deflection needed to reach cmd_value."""
    return DEAD_ZONE + (cmd_value / max_vel) * (1.0 - DEAD_ZONE)


# --------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------
def record(args):
    from decoupled_wbc.control.teleop.device.pico.xr_client import XrClient

    dead_zone, dz_src = get_dead_zone_fn()
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    service = None
    if args.start_service:
        service = subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        print(f"[rec] started pico service pid={service.pid}")
        time.sleep(2.0)

    client = XrClient()

    if not preflight(client, args.force):
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        if service is not None:
            service.terminate()
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(args.out) if args.out else OUT_DIR / f"pico_raw_{stamp}.jsonl"
    fh = path.open("w", buffering=1)

    meta = {
        "_meta": True,
        "started_wall": time.time(),
        "started_iso": datetime.now().isoformat(timespec="seconds"),
        "hz": args.hz,
        "dead_zone": DEAD_ZONE,
        "max_linear_vel": MAX_LINEAR_VEL,
        "max_angular_vel": MAX_ANGULAR_VEL,
        "walk_threshold": WALK_THRESHOLD,
        "dead_zone_impl": dz_src,
        "hands": bool(args.hands),
        "trackers": bool(args.trackers),
        "note": args.note,
        "mark_button": args.mark_button,
        "auto_segment": bool(args.auto),
        "auto_gap": args.gap,
        "labels": labels,
    }
    fh.write(json.dumps(meta) + "\n")

    stop = {"flag": False}

    def on_sigint(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    print(f"[rec] writing {path}")
    print(f"[rec] polling at {args.hz} Hz — Ctrl+C to stop"
          + (f", auto-stop after {args.duration}s" if args.duration else ""))

    dt = 1.0 / args.hz
    n = 0
    t0 = time.monotonic()
    target_yaw = 0.0  # integrated the same way PicoStreamer does
    next_t = t0
    last_ts, last_change, stall_warned, n_stalls = None, t0, False, 0

    # ---- movement segmentation -------------------------------------------
    segments = []
    seg_id = 0
    seg_start = 0.0
    seg_n = 0
    active_since = None   # auto mode: when the current activity bout began
    idle_since = None     # auto mode: when the robot went quiet
    mark_btn_last = False

    def seg_label(i):
        return labels[i] if i < len(labels) else f"seg_{i:03d}"

    def close_segment(now, reason, active_end=None):
        """End the current segment and open the next.

        t_start/t_end are the label boundary and always match the span of
        samples carrying this seg_id. `active_end` (auto mode) additionally
        records where the movement itself stopped, with the trailing stillness
        trimmed off -- it must NOT move the boundary, or every later segment's
        range drifts away from the data it labels.
        """
        nonlocal seg_id, seg_start, seg_n
        a_end = active_end if active_end is not None else now
        segments.append({
            "id": seg_id,
            "label": seg_label(seg_id),
            "t_start": round(seg_start, 3),
            "t_end": round(now, 3),
            "duration": round(now - seg_start, 3),
            "t_active_end": round(a_end, 3),
            "duration_active": round(a_end - seg_start, 3),
            "samples": seg_n,
            "closed_by": reason,
        })
        print(f"\n[rec] segment {seg_id} '{seg_label(seg_id)}' closed "
              f"({now - seg_start:.1f}s, {seg_n} samples, {reason}) "
              f"→ now recording '{seg_label(seg_id + 1)}'", flush=True)
        seg_id += 1
        seg_start = now
        seg_n = 0

    def safe(fn, default=None):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return default

    keys = KeyMarker()
    try:
        keys.__enter__()
        if keys.active:
            print("[rec] SPACE = end this movement and start the next · "
                  "q = stop recording")
        else:
            print("[rec] stdin is not a tty — spacebar marking disabled"
                  + (f", using {args.mark_button}" if args.mark_button else ""))
        if args.mark_button:
            print(f"[rec] controller button '{args.mark_button}' also marks a boundary")
        if args.auto:
            print(f"[rec] auto-segmenting on activity (gap {args.gap}s of stillness)")
        print(f"[rec] recording '{seg_label(0)}'", flush=True)

        while not stop["flag"]:
            now = time.monotonic()
            t_rel = now - t0  # segment bookkeeping is all in relative seconds
            if args.duration and t_rel >= args.duration:
                break

            # ---- spacebar / keyboard marking
            while True:
                ch = keys.poll()
                if ch is None:
                    break
                if ch == " ":
                    close_segment(t_rel, "space")
                elif ch in ("q", "Q"):
                    stop["flag"] = True
                elif ch == "\x03":  # Ctrl+C if ISIG somehow off
                    stop["flag"] = True

            lj = safe(lambda: list(client.get_joystick_state("left")), [None, None])
            rj = safe(lambda: list(client.get_joystick_state("right")), [None, None])

            rec = {
                "i": n,
                "t_wall": time.time(),
                "t_mono": now - t0,
                "xrt_ts_ns": safe(client.get_timestamp_ns),
                "lx": lj[0], "ly": lj[1],
                "rx": rj[0], "ry": rj[1],
            }
            for k in KEYS:
                rec[k] = safe(lambda k=k: float(client.get_key_value_by_name(k)))
            for b in BUTTONS:
                rec[b] = safe(lambda b=b: int(bool(client.get_button_state_by_name(b))))
            for p in POSES:
                rec[p] = safe(lambda p=p: [float(v) for v in client.get_pose_by_name(p)])

            # ---- derived: exactly what PicoStreamer would emit from this input
            if lj[0] is not None and rj[0] is not None:
                fwd_bwd, strafe, yaw_in = lj[1], -lj[0], -rj[0]
                vx = dead_zone(fwd_bwd, DEAD_ZONE) * MAX_LINEAR_VEL
                vy = dead_zone(strafe, DEAD_ZONE) * MAX_LINEAR_VEL
                vyaw_flag = dead_zone(yaw_in, DEAD_ZONE) * MAX_ANGULAR_VEL
                target_yaw += vyaw_flag * dt
                target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
                norm = math.sqrt(vx * vx + vy * vy + vyaw_flag * vyaw_flag)
                rec.update(
                    cmd_vx=vx, cmd_vy=vy, cmd_vyaw_flag=vyaw_flag,
                    cmd_target_yaw=target_yaw,
                    cmd_norm=norm,
                    would_walk=int(norm >= WALK_THRESHOLD),
                )

            if args.hands:
                for side in ("left", "right"):
                    st = safe(lambda s=side: client.get_hand_tracking_state(s))
                    rec[f"{side}_hand"] = (
                        [[float(v) for v in row] for row in st] if st is not None else None
                    )
            if args.trackers:
                td = safe(client.get_motion_tracker_data, {}) or {}
                rec["n_trackers"] = len(td)
                rec["tracker_serials"] = sorted(td.keys())

            # ---- controller-button marking (rising edge)
            if args.mark_button:
                pressed = bool(rec.get(args.mark_button))
                if pressed and not mark_btn_last:
                    close_segment(t_rel, f"button:{args.mark_button}")
                mark_btn_last = pressed

            # ---- auto segmentation on movement bouts
            if args.auto:
                sticks = [abs(v) for v in (rec["lx"], rec["ly"], rec["rx"], rec["ry"])
                          if v is not None]
                moving = bool(sticks) and max(sticks) >= DEAD_ZONE
                if moving:
                    idle_since = None
                    if active_since is None:
                        active_since = t_rel
                else:
                    if idle_since is None:
                        idle_since = t_rel
                    # close only a segment that actually contained movement,
                    # and trim the trailing stillness off its end
                    if active_since is not None and (t_rel - idle_since) >= args.gap:
                        close_segment(t_rel, "auto", active_end=idle_since)
                        active_since = None

            rec["seg_id"] = seg_id
            rec["seg_label"] = seg_label(seg_id)
            seg_n += 1

            # Watchdog: the device clock freezing means the headset stopped
            # streaming and everything from here is zeros.
            ts_now = rec.get("xrt_ts_ns")
            if ts_now and ts_now != last_ts:
                last_ts, last_change, stall_warned = ts_now, now, False
            elif (now - last_change) > 3.0 and not stall_warned:
                print(f"\n[rec] WARNING: device clock frozen for "
                      f"{now - last_change:.0f}s at t={now - t0:.0f}s — "
                      f"recording zeros until it resumes", flush=True)
                stall_warned = True
                n_stalls += 1
            rec["stalled"] = int(stall_warned)

            fh.write(json.dumps(rec) + "\n")
            n += 1

            if args.verbose and n % args.hz == 0:
                w = rec.get("would_walk")
                print(f"\r[rec] {n:7d} samples  {now - t0:7.1f}s  "
                      f"L=({rec['lx']:+.2f},{rec['ly']:+.2f}) "
                      f"R=({rec['rx']:+.2f},{rec['ry']:+.2f}) "
                      f"cmd_norm={rec.get('cmd_norm', float('nan')):.3f} "
                      f"{'WALK' if w else 'idle'}   ", end="", flush=True)

            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # fell behind; resync rather than spiral
    finally:
        # Close the segment still in progress, then persist the segment index.
        if seg_n:
            close_segment(time.monotonic() - t0, "end-of-recording")
        keys.__exit__(None, None, None)
        fh.close()
        seg_path = path.with_suffix(".segments.json")
        seg_path.write_text(json.dumps(segments, indent=2))
        # Without this the SDK's destructor aborts the process on exit (SIGABRT).
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        if service is not None:
            service.terminate()
        elapsed = time.monotonic() - t0
        print(f"\n[rec] {n} samples in {elapsed:.1f}s "
              f"({n / elapsed if elapsed else 0:.1f} Hz effective)")
        if n_stalls:
            print(f"[rec] WARNING: {n_stalls} stream stall(s) — see the "
                  f"'stalled' column")
        print(f"[rec] {path}")
        print(f"[rec] {len(segments)} segment(s) → {seg_path.name}")
        if args.split and segments:
            split_segments(path)
        if n:
            analyze(argparse.Namespace(analyze=str(path)))


def split_segments(path):
    """Write one JSONL per segment, carrying the header into each."""
    path = Path(path)
    meta, by_seg = None, {}
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("_meta"):
                meta = o
            else:
                by_seg.setdefault(o.get("seg_id", 0), []).append(o)
    for sid, rows in sorted(by_seg.items()):
        label = rows[0].get("seg_label", f"seg_{sid:03d}")
        out = path.with_name(f"{path.stem}.{sid:03d}_{label}.jsonl")
        with out.open("w") as fh:
            if meta:
                fh.write(json.dumps({**meta, "segment": sid, "segment_label": label}) + "\n")
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[rec]   {out.name}  ({len(rows)} samples)")


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------
def per_segment_table(rows):
    """One row per marked movement: how hard it was pushed and whether it moved."""
    segs = {}
    for r in rows:
        segs.setdefault(r.get("seg_id", 0), []).append(r)
    if len(segs) <= 1 and 0 in segs:
        return  # unsegmented recording, the axis table already says it all

    print(f"\n{'seg':>3} {'label':16} {'secs':>6} {'n':>6} "
          f"{'|lx|':>6} {'|ly|':>6} {'|rx|':>6} {'walk%':>6}  peak axis")
    for sid, rs in sorted(segs.items()):
        span = rs[-1]["t_mono"] - rs[0]["t_mono"]
        peaks = {}
        for ax in ("lx", "ly", "rx", "ry"):
            v = [abs(r[ax]) for r in rs if r.get(ax) is not None]
            peaks[ax] = max(v) if v else 0.0
        w = [r.get("would_walk") for r in rs if r.get("would_walk") is not None]
        walk = 100 * sum(w) / len(w) if w else 0.0
        top = max(peaks, key=peaks.get)
        print(f"{sid:>3} {rs[0].get('seg_label', ''):16} {span:>6.1f} {len(rs):>6} "
              f"{peaks['lx']:>6.2f} {peaks['ly']:>6.2f} {peaks['rx']:>6.2f} "
              f"{walk:>5.0f}%  {top} ({peaks[top]:.2f})")


def analyze(args):
    path = Path(args.analyze)
    rows, meta = [], {}
    with path.open() as fh:
        for line in fh:
            o = json.loads(line)
            (meta.update(o) if o.get("_meta") else rows.append(o))
    if not rows:
        print("[an] no samples")
        return

    n = len(rows)
    span = rows[-1]["t_mono"] - rows[0]["t_mono"]
    print(f"\n=== {path.name} ===")
    print(f"samples {n}  span {span:.1f}s  effective {n / span if span else 0:.1f} Hz "
          f"(requested {meta.get('hz')})")

    ts = [r.get("xrt_ts_ns") for r in rows if r.get("xrt_ts_ns")]
    if len(ts) > 1:
        stale = sum(1 for a, b in zip(ts, ts[1:]) if a == b)
        if stale:
            print(f"stale device frames: {stale}/{len(ts) - 1} "
                  f"({100 * stale / (len(ts) - 1):.1f}%) — polling faster than "
                  f"the device updates; lower --hz or dedupe on xrt_ts_ns")
        else:
            print("stale device frames: none")
    else:
        print("stale device frames: n/a (no device timestamps)")

    walk_stick = stick_for_command(WALK_THRESHOLD, MAX_LINEAR_VEL)
    turn_stick = stick_for_command(WALK_THRESHOLD, MAX_ANGULAR_VEL)

    print(f"\nthresholds: |stick| >= {walk_stick:.3f} to walk, "
          f">= {turn_stick:.3f} to turn (dead zone {DEAD_ZONE})")
    print(f"\n{'axis':6} {'min':>7} {'max':>7} {'|.|p50':>7} {'|.|p95':>7} "
          f"{'zero%':>7} {'dead%':>7} {'live%':>7}")

    for axis, thr, label in (("lx", walk_stick, "strafe"), ("ly", walk_stick, "fwd"),
                             ("rx", turn_stick, "yaw"), ("ry", turn_stick, "unused")):
        v = [r[axis] for r in rows if r.get(axis) is not None]
        if not v:
            continue
        a = sorted(abs(x) for x in v)
        p = lambda q: a[min(len(a) - 1, int(q * len(a)))]  # noqa: E731
        zero = sum(1 for x in a if x < DEAD_ZONE) / len(a)
        dead = sum(1 for x in a if DEAD_ZONE <= x < thr) / len(a)
        live = sum(1 for x in a if x >= thr) / len(a)
        print(f"{axis:6} {min(v):>7.3f} {max(v):>7.3f} {p(0.5):>7.3f} {p(0.95):>7.3f} "
              f"{100*zero:>6.1f}% {100*dead:>6.1f}% {100*live:>6.1f}%   ({label})")

    print("\n  zero% = below the dead zone, command is exactly 0")
    print("  dead% = NON-ZERO command that is still below the walk threshold:")
    print("          logged and recorded, but the robot does not move")
    print("  live% = actually commands motion")

    w = [r.get("would_walk") for r in rows if r.get("would_walk") is not None]
    if w:
        cross = sum(1 for a, b in zip(w, w[1:]) if a != b)
        print(f"\nwould_walk: {100 * sum(w) / len(w):.1f}% of samples, "
              f"{cross} threshold crossings "
              f"({cross / span * 60 if span else 0:.1f}/min)")

    pressed = {b: sum(r.get(b) or 0 for r in rows) for b in BUTTONS}
    pressed = {k: v for k, v in pressed.items() if v}
    print(f"buttons held (samples): {pressed or 'none'}")

    per_segment_table(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hz", type=float, default=50.0, help="poll rate (default 50, = teleop loop)")
    ap.add_argument("--duration", type=float, default=0.0, help="auto-stop after N seconds")
    ap.add_argument("--out", default="", help="output path (default data/pico_raw/pico_raw_<ts>.jsonl)")
    ap.add_argument("--hands", action="store_true", help="also record 27x7 hand tracking (large)")
    ap.add_argument("--trackers", action="store_true", help="also record motion-tracker presence")
    ap.add_argument("--start-service", action="store_true",
                    help="launch the pico service first (NOT while teleop is running)")
    ap.add_argument("--labels", default="",
                    help="comma list naming the movements in order, "
                         "e.g. fwd,back,strafe_l,strafe_r,turn_l,turn_r")
    ap.add_argument("--mark-button", default="",
                    help=f"controller button that also marks a boundary, "
                         f"one of: {', '.join(BUTTONS)} "
                         f"(right_axis_click is unused by teleop)")
    ap.add_argument("--auto", action="store_true",
                    help="also close a segment automatically after --gap of stillness")
    ap.add_argument("--gap", type=float, default=1.0,
                    help="seconds of stillness that ends a movement in --auto (default 1.0)")
    ap.add_argument("--split", action="store_true",
                    help="also write one JSONL per segment next to the recording")
    ap.add_argument("--force", action="store_true",
                    help="record even if the preflight finds no live PICO data")
    ap.add_argument("--note", default="", help="free-text note stored in the file header")
    ap.add_argument("--quiet", dest="verbose", action="store_false", help="no live status line")
    ap.add_argument("--analyze", default="", help="analyze an existing recording instead")
    args = ap.parse_args()

    if args.analyze:
        analyze(args)
        return 0
    return record(args) or 0


if __name__ == "__main__":
    sys.exit(main())
