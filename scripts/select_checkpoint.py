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
        state/runs/session-.../stacked-dqn-... \\
        /tmp/eval-0100000 /tmp/eval-0200000 /tmp/eval-0300000

The selection is by the highest interquartile mean of the final wave. The
intervals are printed beside it, and when the leaders' intervals overlap that is
said out loud: the selection is still made - some checkpoint has to be reported
on set B - but a difference the sample could not resolve is not a finding.

What was chosen is written to `<run directory>/selection.json`, beside the run
rather than carried by hand into the next command: `report_arms.py --selection`
reads it back and refuses a set B that did not play that model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.environment.project_state import state_directory  # noqa: E402
from tower_rl.experiment.arm_evaluation import (  # noqa: E402
    STATISTICS,
    ArmEvaluation,
    overlaps,
    read_arm_evaluation,
    statistic_line,
)
from tower_rl.experiment.comparison import stratified_bootstrap  # noqa: E402
from tower_rl.experiment.tracking import TrackedRun, open_tracked_run  # noqa: E402

#: The statistic the selection is made on. `decisions` is reported beside it and
#: chooses nothing: a checkpoint that survives longer per episode is interesting,
#: but the arm is judged on how far it got.
SELECT_ON = "final_wave"


@dataclass(frozen=True)
class Candidate:
    """One checkpoint of this run, and the evaluation that scored it.

    The four facts travel together because every one of them is read from the
    actor records rather than from where a file happens to sit: which run
    produced it, which of that run's checkpoints it is, and which file on disk
    that names.
    """

    evaluation: ArmEvaluation
    #: The file in this run's own checkpoints directory.
    checkpoint: Path
    #: Decisions spent when it was written, from the name the record carries.
    #: The budget axis every other metric of the run is keyed by, so the greedy
    #: curve lands above the exploring one rather than beside it.
    decisions: int
    #: `CheckpointIdentity` hashed: the run, the profile and the three schemas.
    #: It names the *run*, not the checkpoint - every checkpoint of one run
    #: hashes identically, because identity holds nothing that changes as the
    #: run proceeds. `decisions` is what separates them within the run.
    identity_hash: str

    @property
    def name(self) -> str:
        return self.checkpoint.name


def run_checkpoints(run_directory: Path) -> dict[str, Path]:
    """The numbered checkpoints a run left, by file name."""
    checkpoints = sorted((run_directory / "checkpoints").glob("checkpoint-*.pt"))
    if not checkpoints:
        raise SystemExit(
            f"{run_directory} left no numbered checkpoints; train with "
            "--checkpoint-every-decisions to produce candidates to choose among"
        )
    return {path.name: path for path in checkpoints}


def candidate(directory: Path, run_directory: Path, checkpoints: dict[str, Path]) -> Candidate:
    """One evaluation, checked to be an evaluation of a checkpoint of this run.

    Which run produced the model played here is read from the records, not from
    the file name they mention: `run_id` and the identity hash come from the
    checkpoint's own `CheckpointIdentity`, and the run directory is named for
    its run id. Two runs at the same `--checkpoint-every-decisions` leave files
    called exactly the same thing, so a name check would accept another run's
    evaluation as this one's and the selected file would then be reported as
    this run's work.

    The file name is still what says *which* checkpoint of the run it is, and it
    has to be: the identity hash is constant across a run, so only the decisions
    in the name separate one candidate from another.
    """
    try:
        evaluation = read_arm_evaluation(directory)
    except ValueError as failure:
        raise SystemExit(str(failure)) from failure
    identity = evaluation.policy_identity or {}
    if evaluation.checkpoint is None:
        raise SystemExit(
            f"{directory} did not play a checkpoint; it played "
            f"{identity.get('name', 'an unnamed arm')!r}"
        )
    played_run = identity.get("run_id")
    if played_run != run_directory.name:
        raise SystemExit(
            f"{directory} played a checkpoint of run {played_run!r}, not of "
            f"{run_directory.name!r}; two runs leave identically named files, "
            "so the run id is what says whose checkpoint this is"
        )
    digest = identity.get("checkpoint_identity")
    if not digest:
        raise SystemExit(f"{directory} records no checkpoint identity to verify")
    name = Path(str(evaluation.checkpoint)).name
    if name not in checkpoints:
        raise SystemExit(
            f"{directory} played {name}, which is not a numbered checkpoint "
            f"this run left ({sorted(checkpoints)})"
        )
    return Candidate(
        evaluation=evaluation,
        checkpoint=checkpoints[name],
        decisions=checkpoint_decisions(name),
        identity_hash=str(digest),
    )


def one_run(candidates: list[Candidate]) -> str:
    """The identity every candidate agrees on, or a refusal naming the ones that do not."""
    digests = {item.identity_hash for item in candidates}
    if len(digests) > 1:
        raise SystemExit(
            "these evaluations do not agree on what produced them: "
            + ", ".join(f"{item.name} -> {item.identity_hash}" for item in candidates)
        )
    return digests.pop()


def checkpoint_decisions(name: str) -> int:
    """The decisions a numbered checkpoint's file name records."""
    digits = name.removeprefix("checkpoint-").removesuffix(".pt")
    if not digits.isdigit():
        raise SystemExit(f"{name} does not name the decisions behind it")
    return int(digits)


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
        default=state_directory() / "runs",
        help="where the tracking store lives; only read with --mlflow-run",
    )
    parser.add_argument(
        "--experiment",
        default="tower-rl-training",
        help="the MLflow experiment the run belongs to; only read with --mlflow-run",
    )
    parser.add_argument(
        "--output", type=Path, default=state_directory() / "records" / "selection.json"
    )
    arguments = parser.parse_args()

    checkpoints = run_checkpoints(arguments.run_directory)
    candidates = [
        candidate(directory, arguments.run_directory, checkpoints)
        for directory in arguments.evaluations
    ]
    identity = one_run(candidates)
    scored = [
        (item, score(item.evaluation, resamples=arguments.resamples, seed=arguments.seed))
        for item in candidates
    ]
    # In the order the run produced them, by the decisions behind each one
    # rather than by how its name happens to sort: the zero padding only orders
    # correctly while every name is the same width, and a ten-million-decision
    # run is three days of collection, not a hypothetical.
    scored.sort(key=lambda item: item[0].decisions)

    print(f"{len(scored)} checkpoints of {arguments.run_directory.name}", flush=True)
    for statistic in STATISTICS:
        print(f"\n{statistic}:", flush=True)
        for item, numbers in scored:
            entry = numbers[statistic]
            print(
                "  "
                + statistic_line(
                    item.name,
                    float(entry["iqm"]),
                    float(entry["low"]),
                    float(entry["high"]),
                    int(entry["episodes"]),
                ),
                flush=True,
            )

    ranked = sorted(scored, key=lambda item: float(item[1][SELECT_ON]["iqm"]), reverse=True)
    best, best_numbers = ranked[0]
    selected = best.checkpoint
    contested = [
        item.name
        for item, numbers in ranked[1:]
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

    tracked = tracked_run(arguments)
    if tracked is not None:
        # Onto the training run's own page, on its own budget axis: the greedy
        # curve is the one question the exploring curve cannot answer, and a
        # second run holding it would have to be found by hand.
        for item, numbers in scored:
            entry = numbers[SELECT_ON]
            tracked.log_metrics(
                {
                    "greedy_final_wave_iqm": float(entry["iqm"]),
                    "greedy_final_wave_ci_low": float(entry["low"]),
                    "greedy_final_wave_ci_high": float(entry["high"]),
                },
                decisions=item.decisions,
            )
        print(f"  logged to tracked run {tracked.run_id}", flush=True)

    # What set B has to be reported on, written where the run itself is rather
    # than carried by hand between two commands. `report_arms.py --selection`
    # reads it back and refuses records that did not play this model, which is
    # what closes the gap between choosing on A and reporting on B.
    selection = {
        "run_id": arguments.run_directory.name,
        "checkpoint": str(selected),
        "decisions": best.decisions,
        "checkpoint_identity": identity,
        "selected_on": SELECT_ON,
        "iqm": best_numbers[SELECT_ON]["iqm"],
        "interval": [best_numbers[SELECT_ON]["low"], best_numbers[SELECT_ON]["high"]],
        "selected_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    selection_path = arguments.run_directory / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2))
    print(f"  selection written to {selection_path}", flush=True)

    report: dict[str, Any] = {
        "run_directory": str(arguments.run_directory),
        "mlflow_run": arguments.mlflow_run,
        "selection": selection,
        "selection_contested_by": contested,
        "resamples": arguments.resamples,
        "seed": arguments.seed,
        "candidates": [
            {
                "checkpoint": str(item.checkpoint),
                "decisions": item.decisions,
                "evaluation_directory": str(item.evaluation.directory),
                "policy_identity": item.evaluation.policy_identity,
                # Nested rather than spread: one of the reported statistics is
                # itself called `decisions`, and spreading it here would
                # overwrite the budget position this candidate sits at with a
                # bootstrap of how long its episodes ran.
                "statistics": numbers,
            }
            for item, numbers in scored
        ],
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
