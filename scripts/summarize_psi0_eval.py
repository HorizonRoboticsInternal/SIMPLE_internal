#!/usr/bin/env python3
import argparse
import re
import time
from pathlib import Path


TASK_TITLES = {
    "G1WholebodyXMovePickTeleop-v0": "XMove Pick Teleop",
    "G1WholebodyBendPickMP-v0": "BendPick MP",
    "G1WholebodyHandoverTeleop-v0": "Handover Teleop",
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0": (
        "Locomotion Pick Between Tables"
    ),
    "G1WholebodyTabletopGraspMP-v0": "Tabletop Grasp MP",
    "G1WholebodyXMoveBendPickTeleop-v0": "XMove BendPick Teleop",
    "G1WholebodyBendHandoverTeleop-v0": "Bend Handover Teleop",
    "G1WholebodyBendPickTeleop-v0": "BendPick Teleop",
    "G1WholebodyCloseDoorTeleop-v0": "Close Door Teleop",
    "G1WholebodyOpenFaucetTeleop-v0": "Open Faucet Teleop",
    "G1WholebodyOpenOvenTeleop-v0": "Open Oven Teleop",
    "G1WholebodyOpenTrashCanTeleop-v0": "Open Trash Can Teleop",
    "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0": (
        "Pick And Place And Hug Container Teleop"
    ),
    "G1WholebodyPushOfficeChairTeleop-v0": "Push Office Chair Teleop",
}
EPISODE_RE = re.compile(r"^episode_(\d+): (True|False)\s*$")


def latest_episode_results(stats_path: Path) -> dict[int, bool]:
    if not stats_path.exists():
        return {}
    results: dict[int, bool] = {}
    for line in stats_path.read_text().splitlines():
        if line.strip() == "================":
            results = {}
            continue
        match = EPISODE_RE.match(line)
        if match:
            results[int(match.group(1))] = match.group(2) == "True"
    return results


def collect(out_root: Path, tasks: list[str]) -> dict[str, list[dict[int, bool]]]:
    return {
        task: [
            latest_episode_results(
                out_root / task / f"level-{level}" / "eval_stats.txt"
            )
            for level in range(3)
        ]
        for task in tasks
    }


def is_complete(results: dict[str, list[dict[int, bool]]], episodes: int) -> bool:
    return all(
        len(level_results) >= episodes
        for task_results in results.values()
        for level_results in task_results
    )


def write_markdown(
    output: Path,
    setting: str,
    tasks: list[str],
    results: dict[str, list[dict[int, bool]]],
    episodes: int,
) -> None:
    headers = ["**Setting (old eval data)**"]
    headers.extend(f"**{TASK_TITLES.get(task, task)}**" for task in tasks)
    headers.append("**Sum**")

    successes_by_task = [
        [sum(level_results.values()) for level_results in results[task]]
        for task in tasks
    ]
    total_successes = sum(sum(levels) for levels in successes_by_task)
    total_episodes = len(tasks) * 3 * episodes
    cells = [setting.replace("|", "\\|")]
    cells.extend(
        " \\| ".join(str(successes) for successes in levels)
        for levels in successes_by_task
    )
    cells.append(f"**{total_successes}/{total_episodes}**")

    alignments = [":---:"] * len(headers)
    markdown = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(alignments) + " |",
            "| " + " | ".join(cells) + " |",
            "",
        ]
    )
    output.write_text(markdown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--setting", default="psi0")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()

    stats_root = args.stats_root or args.out_root
    while True:
        results = collect(stats_root, args.tasks)
        if is_complete(results, args.episodes):
            break
        if not args.wait:
            raise SystemExit("Evaluation is not complete")
        completed = sum(
            len(level_results)
            for task_results in results.values()
            for level_results in task_results
        )
        expected = len(args.tasks) * 3 * args.episodes
        print(f"Waiting for evaluation: {completed}/{expected} episodes", flush=True)
        time.sleep(10)

    output = args.out_root / "summary.md"
    write_markdown(output, args.setting, args.tasks, results, args.episodes)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
