"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Bit-exact replay of episodes recorded by ``simple.cli.teleop_holomotion_v14 --record`` (the exact log in
``<dataset>/replay/episode_XXXXXX.npz``).

    python -m simple.cli.replay_holomotion_v14 data/teleop_holomotion_v14/<env_id>/level-0 --episode 0
    python -m simple.cli.replay_holomotion_v14 <dataset> --all --mode action

Scene: rebuilt through SIMPLE from the saved setup (scene knobs, random setup, SIMPLE's environment_config), then
checked against the recorded model hash; ``--model mjb`` uses the saved compiled model instead (no SIMPLE rebuild, e.g.
after the scene code changed).

Modes:
  physics  the recorded ctrl and applied forces of every physics call, from the recorded start state
  action   the recorded actions (joint targets, kp, kd, Dex3 targets) through the robot's PD (G1Sonic.apply_action),
           checking that each call's ctrl comes out identical, then stepping; applied forces (object pseudo-gravity,
           the safety band while it is up) are inputs taken from the log

Either way every frame's qpos/qvel/act is compared bit for bit with the recording. --video renders the replay's head
and third-person views (the SIMPLE rebuild only).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import mujoco
import numpy as np
import typer

from simple.teleop.holomotion_v14 import exact_log as XL


def _episodes(dataset: Path, episode: int, all_: bool, only: str = "") -> list[Path]:
    rdir = dataset / "replay"
    if only:                                                     # --episodes 0,3,5-9
        ks = []
        for part in only.split(","):
            lo, _, hi = part.strip().partition("-")
            ks += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
        return [rdir / f"episode_{k:06d}.npz" for k in ks]
    if all_:
        return sorted(rdir.glob("episode_*.npz"))
    return [rdir / f"episode_{episode:06d}.npz"]


def _setup_key(info: dict) -> str:
    """What one replay process can rebuild: the kitchen kit reads its knobs (plate set, balls, ...) once at import, so episodes
    recorded with other knobs, scene code or robot need a process of their own."""
    return json.dumps(dict(knobs=info.get("env_knobs", {}), scene=info.get("scene"), scene_root=info.get("scene_root"), robot=info.get("robot"),
                           robot_mjcf=info.get("robot_mjcf"), backpack=info.get("backpack_kg"), tilt=info.get("head_tilt_deg"), black=info.get("black_hands")), sort_keys=True)


def _replay_by_setup(dataset: Path, paths: list[Path], argv_tail: list[str], fill_out: Path | None) -> int | None:
    """A dataset whose episodes were recorded with different setups (2026-11-10: the final A2 dataset mixes v1 / v2 plates,
    with and without the balls): replay each setup's episodes in a child process of its own, one after another (a --fill
    appends to the same output). Returns the exit code, or None when one process covers every episode."""
    import subprocess
    groups: dict[str, list[int]] = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            info = json.loads(str(z["info_json"]))
        groups.setdefault(_setup_key(info), []).append(int(p.stem.split("_")[-1]))
    if len(groups) < 2:
        return None
    codes = []
    for i, (key, eps) in enumerate(groups.items()):
        env = {k: v for k, v in os.environ.items() if not k.startswith("KITCHEN_")}   # the child takes the recorded knobs, nothing inherited
        env["HOLOMOTION_REPLAY_GROUP"] = "1"
        if fill_out is not None and i > 0:
            env["HOLOMOTION_FILL_APPEND"] = "1"
        knobs = json.loads(key)["knobs"]
        print(f"[replay] setup {i + 1}/{len(groups)}: episodes {','.join(map(str, eps))} (plates {os.path.basename(knobs.get('KITCHEN_PLATES', 'kitchen_plates'))}, "
              f"balls {knobs.get('KITCHEN_PLATE_BALLS', '0')})", flush=True)
        cmd = [sys.executable, "-m", "simple.cli.replay_holomotion_v14", str(dataset), "--episodes", ",".join(map(str, eps))] + argv_tail
        codes.append(subprocess.run(cmd, env=env).returncode)
    print(f"[replay] {len(groups)} recording setups, each in its own process: " + ", ".join("ok" if c == 0 else f"exit {c}" for c in codes), flush=True)
    return 0 if all(c == 0 for c in codes) else 1


def _build_env(info: dict, setup: dict, headless: bool = True, check_legacy: bool = True):
    """Rebuild the recorded scene through SIMPLE; returns (env, sonic_env, scene module)."""
    import gymnasium as gym
    from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene

    for k, v in info.get("env_knobs", {}).items():
        os.environ[k] = v
    if info.get("scene") == "kitchen" and "KITCHEN_REV" not in info.get("env_knobs", {}):
        os.environ["KITCHEN_REV"] = "1"          # recorded before the kitchen kit had revisions (2026-10-26): the kitchen without the balls
    if info.get("scene_root") and "HOLOBRAIN_SIM_DIR" not in os.environ and os.path.isdir(info["scene_root"]):
        os.environ["HOLOBRAIN_SIM_DIR"] = info["scene_root"]         # the scene code the episode was recorded with
    scene = info["scene"]
    mod = load_scene(scene)
    sonic_config = _load_sonic_config()
    revo2 = info.get("robot") == "revo2"                 # the rot0 G1 with BrainCo Revo 2 hands (2026-10-26): its own robot class and file
    if revo2:
        from simple.tasks.registry import TaskRegistry
        TaskRegistry._instances.clear()                   # the registry caches one task per uid: a fresh one takes the other robot
        from simple.robots.g1_sonic_revo2 import hand_variant_of, select_hand_variant
        _hv = select_hand_variant(hand_variant_of(info.get("robot_mjcf")))   # 2026-11-04: legacy or official hand, as recorded (class file, limits, couplings, grip poses, pinch knots)
        print(f"[replay] Revo 2 hand: {_hv['variant']} ({_hv['mjcf']})", flush=True)
        from simple.robots.g1_sonic_revo2 import use_recorded_grip
        use_recorded_grip(info)                           # 2026-11-11: the grip poses + button assignment the recording was made with
    env = gym.make(info["env_id"], sim_mode="mujoco", render_hz=info["render_hz"], physics_dt=info["physics_dt"],
                   headless=headless, max_episode_steps=10 ** 9, sonic_config=sonic_config,
                   target=getattr(mod, "TARGET", None), dr_level=info["dr_level"], success_criteria=0.9,
                   **({"robot_uid": "g1_sonic_revo2"} if revo2 else {}))
    sonic_env = env.unwrapped
    robot = sonic_env.task.robot
    from simple.teleop.holomotion_v14.robot_variants import REPO_ROOT, backpack_mjcf, teleop_mjcf, tilt_head_sensor
    tilt = float(info.get("head_tilt_deg", 0.0))
    if revo2:
        pass                                              # the class default file (or the recorded robot_mjcf below)
    elif "robot" in info:                                 # rebuild the recorded robot model (written deterministically)
        robot.mjcf_path = teleop_mjcf(info["robot"], info.get("backpack_kg", 0.0), tilt_deg=tilt,
                                      black_hands=bool(info.get("black_hands", False)))   # recorded since 2026-10-18
    elif info.get("backpack_kg", 0) > 0:                  # episodes recorded before --robot existed
        robot.mjcf_path = backpack_mjcf(info["backpack_kg"])
    # (until 2026-10-18 the elif above hung off `if tilt:`, so a tilt-0 episode got the plain backpack robot and only
    # the robot_mjcf override below put the recorded file back)
    if tilt:
        tilt_head_sensor(sonic_env.task, tilt)
    if info.get("robot_mjcf") and robot.mjcf_path != info["robot_mjcf"] and (REPO_ROOT / "data" / info["robot_mjcf"]).exists():
        robot.mjcf_path = info["robot_mjcf"]
    _reset_to(env, sonic_env, mod, info, setup)
    # 2026-10-16: the stereo robot changed (rot0 bracket, both rectified eyes). Episodes recorded before carry the old
    # model's hash: rebuild them with the legacy stereo robot (rot15 base, left pinhole only) when the hash disagrees.
    if check_legacy and info.get("robot") == "stereo" and info.get("model_sha256") and not info.get("black_hands"):
        from simple.teleop.holomotion_v14 import exact_log as XL
        if XL.model_sha256(sonic_env.mujoco.mjModel) != info["model_sha256"]:
            robot.mjcf_path = teleop_mjcf("stereo", info.get("backpack_kg", 0.0), tilt_deg=tilt, legacy=True)
            _reset_to(env, sonic_env, mod, info, setup)
            print(f"[replay] rebuilt with the legacy stereo robot (rot15 bracket): model hash "
                  f"{'matches' if XL.model_sha256(sonic_env.mujoco.mjModel) == info['model_sha256'] else 'STILL DIFFERS'}", flush=True)
    return env, sonic_env, mod


def _reset_to(env, sonic_env, mod, info: dict, setup: dict) -> None:
    """The recorded episode's scene: its setup into the scene module/task, then SIMPLE's saved DR state."""
    from simple.teleop.holomotion_v14.scene_setup import SceneSetups
    setups = SceneSetups(info["scene"], mod, sonic_env.task)
    setups.apply(setup if setup.get("randomized") else setups.nominal())
    env.reset(options={"state_dict": json.loads(info["environment_config"])})


def _replay_action(sonic_env, log: dict) -> dict:
    """Actions through G1Sonic.apply_action (PD recomputed); ctrl checked per call, physics per frame."""
    from simple.core.action import ActionCmd
    eng = sonic_env.mujoco
    m, d, robot = eng.mjModel, eng.mjData, sonic_env.task.robot
    XL.set_state(m, d, log["state0"], XL.INTEGRATION)
    mujoco.mj_forward(m, d)
    rest_at = {int(i): v for i, v in zip(log["rest_idx"], log["rest_val"])}
    call, first_bad, first_ctrl_bad, max_diff = 0, None, None, 0.0
    for t, n_calls in enumerate(log["frame_calls"]):
        cmd = ActionCmd("holomotion", target_q=log["action_target_q"][t], kp=log["action_kp"][t], kd=log["action_kd"][t],
                        left_hand_q=log["action_left_hand_q"][t], right_hand_q=log["action_right_hand_q"][t],
                        apply_elastic_band=False)
        for _ in range(int(n_calls)):
            if call in rest_at:
                XL.set_state(m, d, rest_at[call], XL.REST)
            robot.apply_action(cmd)
            if d.ctrl.tobytes() != log["ctrl"][call].tobytes() and first_ctrl_bad is None:
                first_ctrl_bad = call
            mujoco.mj_step(m, d, nstep=int(log["nstep"][call]))
            call += 1
        got = XL.get_state(m, d, XL.PHYSICS)
        if got.tobytes() != log["physics"][t].tobytes():
            max_diff = max(max_diff, float(np.max(np.abs(got - log["physics"][t]))))
            if first_bad is None:
                first_bad = t
    return dict(frames=len(log["frame_calls"]), exact=first_bad is None and first_ctrl_bad is None,
                first_mismatch=first_bad, first_ctrl_mismatch=first_ctrl_bad, max_abs_diff=max_diff)


def _video_writer(path: Path, fps: int):
    import imageio.v2 as imageio
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1)


def main(
    dataset: Annotated[Path, typer.Argument(help="the LeRobot dataset dir (…/<env_id>/level-N) holding replay/")],
    episode: Annotated[int, typer.Option(help="episode index")] = 0,
    all: Annotated[bool, typer.Option("--all", help="every episode with an exact log")] = False,
    episodes: Annotated[str, typer.Option("--episodes", help="these episodes, e.g. 0,3,5-9 (instead of --episode / --all). Episodes of different recording setups (plate set, balls, robot) are replayed in one process per setup, one after another")] = "",
    mode: Annotated[str, typer.Option(help="physics | action")] = "physics",
    model: Annotated[str, typer.Option(help="sim (rebuild through SIMPLE, check the hash) | mjb (the saved model)")] = "sim",
    video: Annotated[str, typer.Option(help="write an mp4 of the replay (head camera | third person), sim model only")] = "",
    fill: Annotated[bool, typer.Option("--fill", help="write a complete copy of the dataset (<dataset>_filled) with the images (ego_view, wrist cameras) and the "
                                                     "tactile pads rendered / read on the replayed states: what --lean-record left out (meta/lean_record.json). "
                                                     "On a full recording the recomputed columns are checked against the recorded ones. --model sim --mode physics only.")] = False,
    fill_out: Annotated[str, typer.Option(help="--fill: the output dataset dir (default <dataset>_filled)")] = "",
    gates: Annotated[bool, typer.Option("--gates/--no-gates", help="run the task's gates on the replayed physics, frame by frame, and write the result into meta/episodes.jsonl: task_eval (every gate with its first frame, the final state) and realformat.success / success_frame (the task's last gate; a --slice episode keeps its slice success). The original line is kept once in meta/_before_gates_<date>/. The --fill copy gets the same. Since 2026-10-30 the teleop loop does not evaluate the gates (user: no gate checks in the online pipeline), so this is where the labels come from. Physics replay through the SIMPLE rebuild only")] = True,
):
    if mode not in ("physics", "action"):
        raise typer.BadParameter("--mode is physics or action")
    if mode == "action" and model != "sim":
        raise typer.BadParameter("--mode action needs the SIMPLE rebuild (--model sim)")
    if fill and (mode != "physics" or model != "sim"):
        raise typer.BadParameter("--fill needs --mode physics --model sim")
    filler = None
    paths = _episodes(dataset, episode, all, episodes)
    if model == "sim" and len(paths) > 1 and not os.environ.get("HOLOMOTION_REPLAY_GROUP"):
        tail = ["--mode", mode, "--model", model, "--gates" if gates else "--no-gates"] + (["--video", video] if video else [])
        out_fill = None
        if fill:
            from simple.teleop.holomotion_v14.fill_record import default_out
            out_fill = Path(fill_out) if fill_out else default_out(dataset)
            tail += ["--fill", "--fill-out", str(out_fill)]
        code = _replay_by_setup(dataset, paths, tail, out_fill)
        if code is not None:
            sys.exit(code)
    if not paths or not paths[0].exists():
        sys.exit(f"no exact log: {paths[0] if paths else dataset / 'replay'}")
    env = sonic_env = mod = None
    results = []
    evals: dict[int, dict] = {}
    for path in paths:
        log = XL.load(path)
        info, setup = log["info"], log["setup"]
        if model == "mjb":
            m = mujoco.MjModel.from_binary_path(str(path.with_suffix(".mjb")))
            d = mujoco.MjData(m)
            same_model = XL.model_sha256(m) == info["model_sha256"]
        else:
            if env is None:
                env, sonic_env, mod = _build_env(info, setup)
                first = info
            else:
                if info.get("scene") != first.get("scene") or info.get("backpack_kg") != first.get("backpack_kg"):
                    sys.exit("--all over episodes of different scenes / robots: replay them separately")
                _reset_to(env, sonic_env, mod, info, setup)
            m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
            same_model = XL.model_sha256(m) == info["model_sha256"]
        if not same_model:
            print(f"[replay] {path.name}: WARNING the rebuilt model differs from the recorded one "
                  f"(scene code changed?) -- try --model mjb")
        if fill and filler is None:
            from simple.teleop.holomotion_v14.fill_record import Filler, default_out
            filler = Filler(dataset, Path(fill_out) if fill_out else default_out(dataset), sonic_env, info)
        if filler is not None:
            filler.begin(int(path.stem.split("_")[-1]), len(log["frame_calls"]))
        writer, render = None, None
        if video and model == "sim":
            writer = _video_writer(Path(video) if not all else Path(video).with_name(f"{Path(video).stem}_{path.stem}.mp4"),
                                   int(info["render_hz"]))
            renderer = mujoco.Renderer(m, 360, 640)
            third = mujoco.MjvCamera()
            vt = getattr(mod, "VIEW_THIRD", None)
            if vt:                                                   # the scene's own overview (kitchen, 2026-10-26)
                third.type = mujoco.mjtCamera.mjCAMERA_FREE; third.lookat[:] = vt["lookat"]; third.distance, third.azimuth, third.elevation = vt["distance"], vt["azimuth"], vt["elevation"]
            else:
                third.type, third.distance, third.azimuth, third.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, 3.0, 150, -25
                third.trackbodyid = m.body("pelvis").id
            head = next((c for c in range(m.ncam) if "head" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) or "")), 0)

            def render(t):
                renderer.update_scene(d, camera=head)
                a = renderer.render()
                renderer.update_scene(d, camera=third)
                writer.append_data(np.concatenate([a, renderer.render()], axis=1))
        gate_run = GateRun(sonic_env, mod) if (gates and mode == "physics" and model == "sim") else None
        if gate_run is not None:
            render_cb = render

            def on_frame(t, _g=gate_run, _r=render_cb):
                _g.frame(t)
                if _r is not None:
                    _r(t)
        else:
            on_frame = render
        if mode == "physics":
            res = XL.replay_physics(m, d, log, on_frame=on_frame, on_frame_start=filler.frame_start if filler is not None else None)
        else:
            res = _replay_action(sonic_env, log)
        if writer is not None:
            writer.close()
        if gate_run is not None:
            ev = gate_run.result(res)
            src_ep = int(path.stem.split("_")[-1])
            evals[src_ep] = ev
            if filler is not None and src_ep in filler.episodes_meta:        # the fill copy's line is built from this record
                _apply_gate_result(filler.episodes_meta[src_ep], ev)
            latched = ", ".join(f"{k} @{ev['frames_first'][k]}" for k, v in ev["gates"].items() if v) or "none"
            print(f"[gates] {path.name}: {latched}; success ({ev['final_gate']}) {ev['success']}", flush=True)
        if filler is not None:
            filler.end(res)
        res.update(episode=path.stem, same_model=same_model, seed=setup.get("seed"))
        results.append(res)
        verdict = "BIT-EXACT" if res["exact"] else f"DIFFERS from frame {res['first_mismatch']} (max |diff| {res['max_abs_diff']:.3g})"
        print(f"[replay] {path.name}: {res['frames']} frames, {mode} replay, model {'identical' if same_model else 'DIFFERENT'}: {verdict}",
              flush=True)
    n_ok = sum(r["exact"] for r in results)
    print(f"[replay] {n_ok}/{len(results)} episodes bit-exact")
    if evals:
        _write_gate_results(dataset, evals)
    if filler is not None:
        filler.finish()
    if env is not None:
        env.close()
    os._exit(0 if n_ok == len(results) else 1)


class GateRun:
    """The task's gates over one replayed episode (2026-10-30: the labels are made here, not in the teleop loop).

    ``frame(t)`` after frame t's physics (``replay_physics`` on_frame); ``result()`` -> the task_eval record: every boolean
    gate with the first frame it latched, the final gate (door_closed when the task has it, else placed) as success, and the
    last progress values (door_deg, loaded, max_lift, ...; the t_* stamps left out)."""

    def __init__(self, sonic_env, mod) -> None:
        self.env, self.task, self.mod = sonic_env, sonic_env.task, mod
        self.first, self.last, self.max_door = {}, {}, 0.0

    def frame(self, t: int) -> None:
        P = self.task.update_gates({}, self.env.mujoco)
        for k, v in P.items():
            if isinstance(v, bool) and v and k not in self.first:
                self.first[k] = int(t)
        if "door_deg" in P:
            self.max_door = max(self.max_door, float(P["door_deg"]))
        self.last = dict(P)

    def result(self, res: dict) -> dict:
        import time
        P = self.last
        bools = [k for k, v in P.items() if isinstance(v, bool)]
        final = "door_closed" if "door_closed" in P else "placed" if "placed" in P else (bools[-1] if bools else None)
        out = dict(task=getattr(self.mod, "TASK", None) or getattr(self.mod, "ENV_ID", ""), instruction=str(getattr(self.task, "instruction", "")),
                   evaluated=time.strftime("%Y-%m-%d %H:%M"), source="replay", bit_exact=bool(res.get("exact")), frames=int(res.get("frames", 0)),
                   gates={k: bool(P[k]) for k in bools}, frames_first={k: self.first.get(k) for k in bools}, final_gate=final,
                   success=bool(P.get(final)) if final else False, success_frame=self.first.get(final) if final else None,
                   end={k: v for k, v in P.items() if not isinstance(v, bool) and not k.startswith("t_") and k != "t"})
        if "door_deg" in P:                                                 # the kitchen fields relabel_kitchen_task.py wrote, same names
            out.update(max_door_deg=round(self.max_door, 1), max_loaded=int(P.get("max_loaded", 0)), loaded_at_end=int(P.get("loaded", 0)),
                       loaded_dishes_at_end=P.get("loaded_dishes", ""))
        return out


def _apply_gate_result(line: dict, ev: dict) -> None:
    """task_eval + realformat.success into one episodes.jsonl line (a --slice episode keeps its slice success)."""
    line["task_eval"] = json.dumps(ev)
    if line.get("slice"):
        return
    rf = json.loads(line["realformat"]) if isinstance(line.get("realformat"), str) else dict(line.get("realformat") or {})
    rf["success"], rf["success_frame"] = ev["success"], ev["success_frame"]
    line["realformat"] = json.dumps(rf)


def _write_gate_results(dataset: Path, evals: dict[int, dict]) -> None:
    import shutil
    import time
    meta = dataset / "meta"
    f = meta / "episodes.jsonl"
    if not f.exists() or not evals:
        return
    bak = meta / f"_before_gates_{time.strftime('%Y%m%d')}"
    bak.mkdir(exist_ok=True)
    if not (bak / f.name).exists():
        shutil.copy2(f, bak / f.name)
    lines = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    by_index = {int(e["episode_index"]): e for e in lines}
    n = 0
    for ep, ev in evals.items():
        if ep in by_index:
            _apply_gate_result(by_index[ep], ev)
            n += 1
    f.write_text("".join(json.dumps(e) + "\n" for e in lines))
    print(f"[gates] {n} episode line(s) of {f} updated (task_eval, realformat.success; originals in {bak.name}/)", flush=True)


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
