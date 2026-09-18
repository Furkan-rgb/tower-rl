#!/usr/bin/env python3
"""Report named arms against each other from evaluations already collected.

The second stage of the two-stage protocol, and the last thing Milestone 2 runs.
Each argument names one arm and the directory of actor records `run_actors.py`
left for it, so the learned arm here is the checkpoint `select_checkpoint.py`
chose on set A, re-evaluated on a set B of its own. Pass `--selection
<run>/selection.json` and that is checked rather than assumed: the records say
which model played, the selection says which was chosen, and a mismatch is
refused by name.

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
import re
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
from tower_rl.experiment.tracking import TrackedRun, open_tracked_run  # noqa: E402
from tower_rl.experiment.wave_statistics import analyse_reports  # noqa: E402

#: What an arm may be called. The name becomes a file name under
#: `--output-directory` and a metric key on the tracked run, so anything outside
#: this would either write somewhere it was not asked to - `../` is a name - or
#: produce a key the store refuses halfway through a report.
ARM_NAME = re.compile(r"[A-Za-z0-9_-]+")


def named_arms(requested: list[str]) -> dict[str, Path]:
    """`name=<directory>` pairs, in the order they were given."""
    arms: dict[str, Path] = {}
    for item in requested:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory:
            raise SystemExit(f"expected name=<directory>, not {item!r}")
        if not ARM_NAME.fullmatch(name):
            # Refused here, before anything is read or written, rather than by
            # whatever the name is later used as.
            raise SystemExit(
                f"arm name {name!r} may hold only letters, digits, underscores "
                "and dashes; it is used as a file name and as a metric key"
            )
        if name in arms:
            raise SystemExit(f"arm {name!r} is named twice")
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise SystemExit(f"no directory at {path} for arm {name!r}")
        arms[name] = path
    if len(arms) < 2:
        raise SystemExit("a report needs at least two arms")
    return arms


def require_selection(selection_path: Path, evaluations: list[ArmEvaluation]) -> dict[str, Any]:
    """Refuse a set B that did not play the model set A chose.

    The two stages are separate commands over separate directories, so the one
    thing that can silently go wrong between them is reporting the wrong model:
    a stale directory, a re-run that overwrote one, the right run's wrong
    checkpoint. The selection file says which model was chosen and the records
    say which model played, and this is where the two are made to agree.
    """
    selection: dict[str, Any] = json.loads(selection_path.read_text())
    wanted_identity = selection["checkpoint_identity"]
    wanted_name = Path(str(selection["checkpoint"])).stem
    played = [
        evaluation
        for evaluation in evaluations
        if (evaluation.policy_identity or {}).get("checkpoint_identity")
    ]
    if not played:
        raise SystemExit(
            f"{selection_path} selected {wanted_name}, but no arm here played a "
            "checkpoint at all; name the selected checkpoint's directory as an arm"
        )
    for evaluation in played:
        identity = evaluation.policy_identity or {}
        if identity.get("checkpoint_identity") != wanted_identity:
            raise SystemExit(
                f"arm {evaluation.name!r} played a checkpoint of another run "
                f"({identity.get('run_id')!r}), not the one {selection_path} selected "
                f"({selection['run_id']!r})"
            )
        if identity.get("name") != wanted_name:
            raise SystemExit(
                f"arm {evaluation.name!r} played {identity.get('name')!r}, but "
                f"{selection_path} selected {wanted_name!r} of the same run"
            )
    return selection


def read(arms: dict[str, Path]) -> list[ArmEvaluation]:
    evaluations = []
    for name, directory in arms.items():
        try:
            evaluations.append(read_arm_evaluation(directory, name=name))
        except ValueError as failure:
            raise SystemExit(f"arm {name!r}: {failure}") from failure
    return evaluations


def tracked_run(arguments: argparse.Namespace) -> TrackedRun | None:
    """The run these results are added to, or nothing if none was named."""
    if not arguments.mlflow_run:
        return None
    try:
        return open_tracked_run(
            arguments.mlflow_run,
            run_dir=arguments.run_dir,
            experiment=arguments.experiment,
        )
    except ImportError as missing:
        raise SystemExit(
            f"--mlflow-run needs MLflow installed ({missing}). "
            "Install it with `uv sync --extra tracking`, or drop the flag."
        ) from missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arms",
        nargs="+",
        help="name=<directory>, one per arm; e.g. scripted=/tmp/eval-scripted",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help=(
            "the selection.json select_checkpoint.py left in the run directory; "
            "the arm that played a checkpoint is checked to be the one it chose"
        ),
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("/tmp/tower-rl-arms"),
        help="where each arm's pooled episodes are written for the per-wave comparison",
    )
    parser.add_argument(
        "--mlflow-run",
        default=None,
        help=(
            "an existing tracked run id to add these results to, so the greedy "
            "curve lands on the page of the training run it is about"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.home() / ".local/state/tower-rl/runs",
        help="where the tracking store lives; only read with --mlflow-run",
    )
    parser.add_argument(
        "--experiment",
        default="tower-rl-training",
        help="the MLflow experiment the run belongs to; only read with --mlflow-run",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-arms.json"))
    arguments = parser.parse_args()

    evaluations = read(named_arms(arguments.arms))
    # Before a single interval is computed: a report of the wrong model is
    # worse than no report, and this is the one check that can catch it.
    selection = (
        None if arguments.selection is None
        else require_selection(arguments.selection, evaluations)
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "resamples": arguments.resamples,
        "seed": arguments.seed,
        "selection": selection,
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

    tracked = tracked_run(arguments)
    if tracked is not None:
        # One measurement about a finished run rather than a point on its
        # budget, so it sits at the origin of the same axis: the training run's
        # page then carries what its selected checkpoint was worth against the
        # floors, beside the curve that produced it.
        for name, entry in report["arms"].items():
            tracked.log_metrics(
                {
                    f"report_{name}_final_wave_iqm": float(entry["final_wave"]["iqm"]),
                    f"report_{name}_final_wave_ci_low": float(entry["final_wave"]["low"]),
                    f"report_{name}_final_wave_ci_high": float(entry["final_wave"]["high"]),
                },
                decisions=0,
            )
        print(f"\nlogged to tracked run {tracked.run_id}", flush=True)

    report["mlflow_run"] = arguments.mlflow_run
    arguments.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
