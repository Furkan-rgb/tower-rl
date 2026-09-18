#!/usr/bin/env python3
"""Report named arms against each other from evaluations already collected.

The second stage of the two-stage protocol, and the last thing Milestone 2 runs.
Each argument names one arm and the directory of actor records `run_actors.py`
left for it, so the learned arm here is the checkpoint `select_checkpoint.py`
chose on set A, re-evaluated on a set B of its own:

    uv run python scripts/report_arms.py \\
        random=/tmp/eval-random scripted=/tmp/eval-scripted \\
        stacked-dqn=/tmp/eval-selected

Three things are printed and nothing is concluded. The interquartile mean of
each arm with its stratified interval, the pairwise bootstrap differences
between them, and the per-wave comparison of every pair, which is the sharp
instrument: a final wave's variance is dominated by how many waves an episode
survives, so half a wave needs hundreds of episodes to see, while what one wave
index cost carries a fraction of that variance (`experiment/wave_statistics.py`).

This decides no verdict. It prints intervals, and an interval that contains zero
means this sample could not tell the arms apart - which is not the same as their
being equal, and is never reported as if it were.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.experiment.arm_evaluation import (  # noqa: E402
    STATISTICS,
    ArmEvaluation,
    pooled_report,
    read_arm_evaluation,
    statistic_line,
)
from tower_rl.experiment.comparison import compare, stratified_bootstrap  # noqa: E402
from tower_rl.experiment.wave_statistics import analyse_reports  # noqa: E402


def named_arms(requested: list[str]) -> dict[str, Path]:
    """`name=<directory>` pairs, in the order they were given."""
    arms: dict[str, Path] = {}
    for item in requested:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory:
            raise SystemExit(f"expected name=<directory>, not {item!r}")
        if name in arms:
            raise SystemExit(f"arm {name!r} is named twice")
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise SystemExit(f"no directory at {path} for arm {name!r}")
        arms[name] = path
    if len(arms) < 2:
        raise SystemExit("a report needs at least two arms")
    return arms


def read(arms: dict[str, Path]) -> list[ArmEvaluation]:
    evaluations = []
    for name, directory in arms.items():
        try:
            evaluations.append(read_arm_evaluation(directory, name=name))
        except ValueError as failure:
            raise SystemExit(f"arm {name!r}: {failure}") from failure
    return evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arms",
        nargs="+",
        help="name=<directory>, one per arm; e.g. scripted=/tmp/eval-scripted",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/tmp/tower-rl-arms"),
        help="where each arm's pooled episodes are written for the per-wave comparison",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-arms.json"))
    arguments = parser.parse_args()

    evaluations = read(named_arms(arguments.arms))
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "resamples": arguments.resamples,
        "seed": arguments.seed,
        "arms": {},
        "differences": [],
        "per_wave": {},
    }

    print("interquartile mean, stratified by actor:", flush=True)
    for statistic in STATISTICS:
        print(f"\n{statistic}:", flush=True)
        for evaluation in evaluations:
            point, low, high = stratified_bootstrap(
                evaluation.strata[statistic],
                resamples=arguments.resamples,
                seed=arguments.seed,
            )
            entry = report["arms"].setdefault(
                evaluation.name,
                {
                    "directory": str(evaluation.directory),
                    "policy_identity": evaluation.policy_identity,
                    "valid_episodes": evaluation.valid_episodes,
                    "actors": len(evaluation.strata[statistic]),
                },
            )
            entry[statistic] = {
                "iqm": round(point, 3),
                "low": round(low, 3),
                "high": round(high, 3),
            }
            print(
                "  "
                + statistic_line(
                    evaluation.name, point, low, high, evaluation.valid_episodes
                ),
                flush=True,
            )

    print("\npairwise difference in mean final wave:", flush=True)
    differences = compare(
        {evaluation.name: evaluation.values("final_wave") for evaluation in evaluations},
        seed=arguments.seed,
    )
    for difference in differences:
        print(f"  {difference.describe()}", flush=True)
        report["differences"].append(
            {
                "left": difference.left,
                "right": difference.right,
                "difference": round(difference.difference, 3),
                "interval": [round(difference.low, 3), round(difference.high, 3)],
                "effect_size": round(difference.effect_size, 3),
                "separated": difference.separated,
            }
        )

    # The per-wave comparison reads report JSONs by path, so each arm's episodes
    # are pooled into one file of the shape it already takes. Written beside the
    # report rather than kept in memory: the same files are what a later reading
    # of this comparison is reproduced from.
    pooled = {}
    for evaluation in evaluations:
        path = arguments.output_directory / f"{evaluation.name}.json"
        path.write_text(json.dumps(pooled_report(evaluation), indent=2))
        pooled[evaluation.name] = path

    print("\nper wave index:", flush=True)
    names = [evaluation.name for evaluation in evaluations]
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            # Each pooled file is named for its arm, which is where
            # `analyse_reports` takes the arm names from.
            rendered = analyse_reports(str(pooled[left]), str(pooled[right]))
            print(rendered, flush=True)
            report["per_wave"][f"{left} vs {right}"] = rendered

    arguments.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
