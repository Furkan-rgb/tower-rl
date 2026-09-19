#!/usr/bin/env python3
"""Measure the plasticity-loss signature across the checkpoints of one run.

A flat greedy curve across a run's checkpoints is consistent with several
things, and plasticity loss (Nikishin et al. 2022; Sokar et al. 2023; Kumar et
al. 2021) is only one of them. This reads checkpoints already on disk and asks
whether that particular signature is there, so an expensive reset A/B is
designed against a measurement rather than against a hypothesis. No emulator is
started and nothing is trained: it is minutes of CPU.

    uv run python scripts/diagnose_plasticity.py \\
        --output ~/.local/state/tower-rl/m2-run1/plasticity/diagnosis.json \\
        --observations ~/.local/state/tower-rl/m2-run1/plasticity/batch.pt \\
        ~/.local/state/tower-rl/runs/.../checkpoints/checkpoint-*.pt

Three quantities per checkpoint, ordered by the decisions the checkpoint was
written at:

* the tau-dormant fraction of each measured layer. Sokar et al. 2023 score
  neuron `i` of a layer by `s_i = E|h_i| / mean_k E|h_k|`, the expectation
  taken over the observation batch, and call it tau-dormant when `s_i <= tau`.
  Reported at tau = 0.025 and tau = 0.1, the two thresholds that paper uses.
* the rank of the features the heads are handed: the stable rank
  `||F||_F^2 / ||F||_2^2`, and `srank_99` in the sense of Kumar et al. 2021 -
  the fewest leading singular values whose sum reaches 99% of the total.
* the L2 norm of every parameter tensor, and of all of them together.

The parameter norms need no input at all, so `--observations` is optional: with
it omitted the norms alone are reported. Dormancy and rank are properties of a
layer's response to real states and are simply absent without a batch.

The layers measured are `trunk.row_encoder`, `trunk.scalar_encoder` and `core`
of `StackedPolicyNetwork`. Each is an `nn.Sequential` ending in its activation,
so the module's own output is the post-activation output the dormancy score is
defined on. `core` is also the pooled feature matrix the rank is taken of: it
is the last representation before the dueling heads.

The observation batch is a `torch.save`d dict of three tensors, shaped the way
the network consumes a stored sequence - time contiguous, so the stacked
history window it builds is the one a real decision saw:

    {"scalars": float32 [batch, time, SCALAR_COUNT],
     "rows":    float32 [batch, time, ROW_COUNT, ROW_WIDTH],
     "mask":    bool    [batch, time, ACTION_COUNT]}

Every step of every sequence is one observation, so `batch * time` is the
number the dormancy expectation is taken over. The mask is carried because the
network needs one to produce Q-values; it does not enter any measurement here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT  # noqa: E402
from tower_rl.environment.run_actions import RUN_ACTIONS  # noqa: E402
from tower_rl.learning.checkpoint import identity_hash, load  # noqa: E402
from tower_rl.learning.network import StackedPolicyNetwork  # noqa: E402
from tower_rl.learning.policies import checkpoint_policy  # noqa: E402

torch.set_num_threads(1)

#: The layers whose response is measured, named as they are reached from
#: `StackedPolicyNetwork`. Two encoders and the core: where a representation
#: could collapse without the heads showing it.
MEASURED_LAYERS: tuple[str, ...] = ("trunk.row_encoder", "trunk.scalar_encoder", "core")

#: The layer whose output *is* the feature matrix the heads see.
FEATURE_LAYER = "core"

#: Sokar et al. 2023: tau = 0.1 for the ReDo benchmarks, tau = 0.025 also tested.
TAUS: tuple[float, ...] = (0.025, 0.1)

#: Kumar et al. 2021 take the fewest leading singular values summing to this
#: share of the total.
SRANK_SHARE = 0.99


class ObservationBatchError(ValueError):
    """The observation batch is not the three tensors this reads."""


def read_observations(path: Path) -> dict[str, Tensor]:
    """Read and shape-check one observation batch.

    Checked rather than trusted: a batch built against a different feature
    schema would still load, run, and produce dormancy numbers that mean
    nothing.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ObservationBatchError(f"{path} does not hold a dict of tensors")
    missing = {"scalars", "rows", "mask"} - set(payload)
    if missing:
        raise ObservationBatchError(f"{path} is missing {', '.join(sorted(missing))}")
    scalars = payload["scalars"].to(torch.float32)
    rows = payload["rows"].to(torch.float32)
    mask = payload["mask"].to(torch.bool)
    if scalars.dim() != 3 or scalars.shape[2] != SCALAR_COUNT:
        raise ObservationBatchError(f"scalars must be [batch, time, {SCALAR_COUNT}]")
    if rows.shape[:2] != scalars.shape[:2] or rows.shape[2:] != (ROW_COUNT, ROW_WIDTH):
        raise ObservationBatchError(
            f"rows must be [batch, time, {ROW_COUNT}, {ROW_WIDTH}] over the same steps"
        )
    if mask.shape[:2] != scalars.shape[:2] or mask.shape[2] != len(RUN_ACTIONS):
        raise ObservationBatchError(
            f"mask must be [batch, time, {len(RUN_ACTIONS)}] over the same steps"
        )
    if not bool(mask.any(dim=-1).all()):
        raise ObservationBatchError("every observation must leave at least one action valid")
    return {"scalars": scalars, "rows": rows, "mask": mask}


def activations(
    network: StackedPolicyNetwork, batch: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Run the batch through one network and keep what each measured layer emitted.

    Returned flattened to `[observations, width]`: dormancy and rank are both
    defined over a layer's units, and everything in front of the last dimension
    is one more sample of them. For `trunk.row_encoder` that means one sample
    per upgrade row per step, which is exactly what the shared row encoder is
    asked to encode.
    """
    modules = dict(network.named_modules())
    captured: dict[str, Tensor] = {}
    handles = []

    def keep(name: str) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: Tensor) -> None:
            captured[name] = output.detach().reshape(-1, output.shape[-1])

        return hook

    for name in MEASURED_LAYERS:
        handles.append(modules[name].register_forward_hook(keep(name)))
    try:
        with torch.no_grad():
            network(batch["scalars"], batch["rows"], batch["mask"])
    finally:
        for handle in handles:
            handle.remove()
    return captured


def dormant_fraction(layer: Tensor, tau: float) -> float:
    """The share of a layer's units that are tau-dormant over this batch.

    A layer whose every unit is silent has no mean to normalise by; it is
    wholly dormant, and reporting a division by zero instead would hide the
    most extreme case the measurement exists to catch.
    """
    scores = layer.abs().mean(dim=0)
    average = float(scores.mean())
    if average <= 0.0:
        return 1.0
    return float((scores / average <= tau).to(torch.float32).mean())


def stable_rank(features: Tensor) -> float:
    """`||F||_F^2 / ||F||_2^2`: how many directions the features really use."""
    values = torch.linalg.svdvals(features.to(torch.float64))
    largest = float(values[0])
    if largest <= 0.0:
        return 0.0
    return float((values**2).sum()) / (largest**2)


def srank(features: Tensor, share: float = SRANK_SHARE) -> int:
    """Kumar et al. 2021: the fewest leading singular values summing to `share`."""
    values = torch.linalg.svdvals(features.to(torch.float64))
    total = float(values.sum())
    if total <= 0.0:
        return 0
    cumulative = torch.cumsum(values, dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(share, dtype=torch.float64))) + 1


def parameter_norms(network: StackedPolicyNetwork) -> tuple[dict[str, float], float]:
    """Every parameter tensor's L2 norm, and the norm of all of them together.

    The total is the norm of the whole parameter vector, not the sum of the
    per-tensor norms, so it is the quantity a growth curve is usually drawn of.
    """
    norms = {
        name: float(parameter.detach().norm()) for name, parameter in network.named_parameters()
    }
    total = float(torch.tensor(list(norms.values())).norm()) if norms else 0.0
    return norms, total


def diagnose(path: Path, batch: dict[str, Tensor] | None) -> dict[str, Any]:
    """Everything this reports about one checkpoint."""
    checkpoint = load(path)
    backbone, identity = checkpoint_policy(path)
    network = backbone.online
    norms, total = parameter_norms(network)
    result: dict[str, Any] = {
        "checkpoint": str(path),
        "decisions": checkpoint.progress.environment_decisions,
        "optimisation_steps": checkpoint.progress.optimisation_steps,
        "episodes": checkpoint.progress.episodes,
        "run_id": identity.run_id,
        "identity_hash": identity_hash(identity),
        "parameter_norms": norms,
        "parameter_norm_total": total,
    }
    if batch is None:
        return result

    layers = activations(network, batch)
    result["dormant_fraction"] = {
        name: {f"{tau}": dormant_fraction(layers[name], tau) for tau in TAUS}
        for name in MEASURED_LAYERS
    }
    features = layers[FEATURE_LAYER]
    result["rank"] = {
        "layer": FEATURE_LAYER,
        "width": int(features.shape[1]),
        "observations": int(features.shape[0]),
        "stable_rank": stable_rank(features),
        f"srank_{int(SRANK_SHARE * 100)}": srank(features),
    }
    return result


def table(rows: Sequence[dict[str, Any]], *, with_observations: bool) -> list[str]:
    """One line per checkpoint, in the order the run wrote them."""
    header = ["decisions", "steps", "|theta|"]
    if with_observations:
        for name in MEASURED_LAYERS:
            short = name.split(".")[-1]
            header += [f"{short}@.025", f"{short}@.1"]
        header += ["srank", f"srank{int(SRANK_SHARE * 100)}"]

    lines = [" ".join(f"{column:>12}" for column in header)]
    for row in rows:
        cells = [
            f"{row['decisions']:>12,}",
            f"{row['optimisation_steps']:>12,}",
            f"{row['parameter_norm_total']:>12.3f}",
        ]
        if with_observations:
            for name in MEASURED_LAYERS:
                cells += [f"{row['dormant_fraction'][name][f'{tau}']:>12.3f}" for tau in TAUS]
            cells += [
                f"{row['rank']['stable_rank']:>12.2f}",
                f"{row['rank'][f'srank_{int(SRANK_SHARE * 100)}']:>12d}",
            ]
        lines.append(" ".join(cells))
    return lines


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint files to read")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="where the JSON report is written; keep it outside the repository",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="observation batch; without it only the parameter norms are reported",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    batch = read_observations(arguments.observations) if arguments.observations else None
    if batch is None:
        print("no observation batch: reporting parameter norms only", flush=True)
    else:
        count = int(batch["scalars"].shape[0] * batch["scalars"].shape[1])
        print(f"{count} observations from {arguments.observations}", flush=True)

    rows = sorted(
        (diagnose(path.expanduser(), batch) for path in arguments.checkpoints),
        key=lambda row: int(row["decisions"]),
    )
    for line in table(rows, with_observations=batch is not None):
        print(line, flush=True)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "observations": (
            None
            if batch is None
            else {
                "path": str(arguments.observations),
                "count": int(batch["scalars"].shape[0] * batch["scalars"].shape[1]),
                "sequences": int(batch["scalars"].shape[0]),
                "steps_per_sequence": int(batch["scalars"].shape[1]),
            }
        ),
        "layers": list(MEASURED_LAYERS),
        "taus": list(TAUS),
        "checkpoints": rows,
    }
    output = arguments.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"written to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
