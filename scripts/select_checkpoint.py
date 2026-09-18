#!/usr/bin/env python3
"""Choose one checkpoint of a training run from evaluations already collected.

The first stage of the two-stage protocol. Selecting a checkpoint on the same
episodes it is then reported on is how a benchmark reports its own selection
noise as a result, so selection happens here, on set A, and the chosen
checkpoint is afterwards reported on a fresh set B - which is nothing more than
another `run_actors.py` run of that one checkpoint into an empty directory.

This reads only what is already on disk. No emulator is started, no episode is
played, nothing is trained: each argument after the run directory is one
evaluation directory as `run_actors.py --policy checkpoint:<path>` leaves it,
holding one JSON record per actor.

    uv run python scripts/select_checkpoint.py \\
        ~/.local/state/tower-rl/runs/session-.../stacked-dqn-... \\
        /tmp/eval-0100000 /tmp/eval-0200000 /tmp/eval-0300000

The selection is by the highest interquartile mean of the final wave. The
intervals are printed beside it, and when the leaders' intervals overlap that is
said out loud: the selection is still made - some checkpoint has to be reported
on set B - but a difference the sample could not resolve is not a finding.
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
    overlaps,
    read_arm_evaluation,
    statistic_line,
)
from tower_rl.experiment.comparison import stratified_bootstrap  # noqa: E402

#: The statistic the selection is made on. `decisions` is reported beside it and
#: chooses nothing: a checkpoint that survives longer per episode is interesting,
#: but the arm is judged on how far it got.
SELECT_ON = "final_wave"


def run_checkpoints(run_directory: Path) -> dict[str, Path]:
    """The numbered checkpoints a run left, by file name."""
    checkpoints = sorted((run_directory / "checkpoints").glob("checkpoint-*.pt"))
    if not checkpoints:
        raise SystemExit(
            f"{run_directory} left no numbered checkpoints; train with "
            "--checkpoint-every-decisions to produce candidates to choose among"
        )
    return {path.name: path for path in checkpoints}


def candidate(directory: Path, checkpoints: dict[str, Path]) -> ArmEvaluation:
    """One evaluation, checked to be an evaluation of a checkpoint of this run.

    An evaluation directory that played some other run's checkpoint is refused
    by name. Silently including it would put a model the run never produced into
    the selection, and the selected file would then be reported as this run's.
    """
    try:
        evaluation = read_arm_evaluation(directory)
    except ValueError as failure:
        raise SystemExit(str(failure)) from failure
    played = evaluation.checkpoint
    if played is None:
        raise SystemExit(
            f"{directory} did not play a checkpoint; it played "
            f"{(evaluation.policy_identity or {}).get('name', 'an unnamed arm')!r}"
        )
    if Path(played).name not in checkpoints:
        raise SystemExit(
            f"{directory} played {Path(played).name}, which is not a numbered "
            f"checkpoint of this run ({sorted(checkpoints)})"
        )
    return evaluation


def score(
    evaluation: ArmEvaluation, *, resamples: int, seed: int
) -> dict[str, dict[str, float | int]]:
    """Every reported statistic of one checkpoint, as a point and an interval."""
    scored: dict[str, dict[str, float | int]] = {}
    for statistic in STATISTICS:
        point, low, high = stratified_bootstrap(
            evaluation.strata[statistic], resamples=resamples, seed=seed
        )
        scored[statistic] = {
            "iqm": round(point, 3),
            "low": round(low, 3),
            "high": round(high, 3),
            "episodes": evaluation.valid_episodes,
            "actors": len(evaluation.strata[statistic]),
        }
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory",
        type=Path,
        help="the run whose checkpoints are being chosen among",
    )
    parser.add_argument(
        "evaluations",
        type=Path,
        nargs="+",
        help="one directory of actor records per checkpoint, as run_actors.py writes them",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tower-rl-selection.json"))
    arguments = parser.parse_args()

    checkpoints = run_checkpoints(arguments.run_directory)
    candidates = [candidate(directory, checkpoints) for directory in arguments.evaluations]
    scored = [
        (evaluation, score(evaluation, resamples=arguments.resamples, seed=arguments.seed))
        for evaluation in candidates
    ]
    # In the order the run produced them, which is the order the file names sort
    # in, so the table reads as a curve rather than as the command line's order.
    scored.sort(key=lambda item: Path(str(item[0].checkpoint)).name)

    print(f"{len(scored)} checkpoints of {arguments.run_directory.name}", flush=True)
    for statistic in STATISTICS:
        print(f"\n{statistic}:", flush=True)
        for evaluation, numbers in scored:
            entry = numbers[statistic]
            print(
                "  "
                + statistic_line(
                    Path(str(evaluation.checkpoint)).name,
                    float(entry["iqm"]),
                    float(entry["low"]),
                    float(entry["high"]),
                    int(entry["episodes"]),
                ),
                flush=True,
            )

    ranked = sorted(scored, key=lambda item: float(item[1][SELECT_ON]["iqm"]), reverse=True)
    best, best_numbers = ranked[0]
    selected = checkpoints[Path(str(best.checkpoint)).name]
    contested = [
        Path(str(evaluation.checkpoint)).name
        for evaluation, numbers in ranked[1:]
        if overlaps(
            (float(best_numbers[SELECT_ON]["low"]), float(best_numbers[SELECT_ON]["high"])),
            (float(numbers[SELECT_ON]["low"]), float(numbers[SELECT_ON]["high"])),
        )
    ]

    print(f"\nselected {selected}", flush=True)
    if contested:
        # Said out loud rather than left to be read off the table: the selection
        # still has to be made, but this is not a measured difference.
        print(
            "  its interval overlaps " + ", ".join(contested) + "; the selection is a choice "
            "among checkpoints this sample could not separate",
            flush=True,
        )
    print(
        "  report it on a fresh set: run_actors.py --policy "
        f"checkpoint:{selected} --output-directory <new directory>",
        flush=True,
    )

    report: dict[str, Any] = {
        "run_directory": str(arguments.run_directory),
        "selected_on": SELECT_ON,
        "selected_checkpoint": str(selected),
        "selection_contested_by": contested,
        "resamples": arguments.resamples,
        "seed": arguments.seed,
        "candidates": [
            {
                "checkpoint": evaluation.checkpoint,
                "evaluation_directory": str(evaluation.directory),
                "policy_identity": evaluation.policy_identity,
                **numbers,
            }
            for evaluation, numbers in scored
        ],
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
