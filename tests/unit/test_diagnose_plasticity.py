"""The offline plasticity diagnostic, and the capture that feeds it.

The measurements are held against networks whose answer is known by
construction - a layer whose units are deliberately silenced has a known
dormant fraction, and a feature matrix built from a known number of
directions has a known rank - because a diagnostic that is only checked for plausibility cannot be
used to decide against a hypothesis. The entry point itself is exercised
end-to-end over synthetic checkpoints, with and without an observation batch.
"""

from __future__ import annotations

import json
from pathlib import Path

import diagnose_plasticity as diagnostic
import pytest
import run_episodes
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.checkpoint import CheckpointIdentity, TrainingProgress, write_checkpoint
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig

#: Narrow on purpose: the diagnostic must rebuild the shape the checkpoint was
#: written with rather than the default one.
NETWORK = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

LEARNER = StackedDqnConfig(history_length=4, n_step=3, seed=11)


def identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        run_id="stacked-dqn-20260101-000000-abcdef",
        backbone="stacked-dqn",
        profile_id="fake-profile-v1",
        observation_schema="observation-v1",
        action_schema="run-action-v1",
        reward_schema="reward-v1",
        source_revision="test",
    )


def resolved() -> dict[str, object]:
    return {
        "backbone": "stacked-dqn",
        "history_length": LEARNER.history_length,
        "n_step": LEARNER.n_step,
        "discount": LEARNER.discount,
        "learning_rate": LEARNER.learning_rate,
        "target_ema_decay": LEARNER.target_ema_decay,
        "network_identity_capacity": NETWORK.identity_capacity,
        "network_identity_dim": NETWORK.identity_dim,
        "network_hidden": NETWORK.hidden,
        "network_core_hidden": NETWORK.core_hidden,
    }


def written_checkpoint(directory: Path, decisions: int, *, seed: int) -> Path:
    """One checkpoint of a synthetic run at a given decision count."""
    torch.manual_seed(seed)
    backbone = StackedDqnBackbone(
        config=StackedDqnConfig(
            history_length=LEARNER.history_length, n_step=LEARNER.n_step, seed=seed
        ),
        network_config=NETWORK,
        device=torch.device("cpu"),
    )
    path = directory / f"checkpoint-{decisions:07d}.pt"
    write_checkpoint(
        path,
        identity=identity(),
        progress=TrainingProgress(
            environment_decisions=decisions, optimisation_steps=decisions // 4
        ),
        backbone_state=backbone.state_dict(),
        resolved_config=resolved(),
        replay_provenance={},
    )
    return path


def written_batch(path: Path, *, sequences: int = 3, steps: int = 12) -> Path:
    """An observation batch of the documented shape, with a valid mask."""
    generator = torch.Generator().manual_seed(5)
    mask = torch.rand(sequences, steps, len(RUN_ACTIONS), generator=generator) > 0.5
    # WAIT is always available, which both matches the game and guarantees the
    # batch satisfies the reader's "some action is valid" rule.
    mask[..., 0] = True
    torch.save(
        {
            "scalars": torch.rand(sequences, steps, SCALAR_COUNT, generator=generator),
            "rows": torch.rand(sequences, steps, ROW_COUNT, ROW_WIDTH, generator=generator),
            "mask": mask,
        },
        path,
    )
    return path


def test_the_dormant_fraction_counts_the_units_below_the_threshold() -> None:
    """Sokar's score is relative to the layer's own mean, not to an absolute size."""
    # Eight units: six carrying one unit of activation, two carrying a
    # thousandth of it. The mean score is 1, so the two quiet ones score 0.001
    # and are dormant at both thresholds while the loud ones are dormant at
    # neither.
    layer = torch.tensor([[1.0] * 6 + [0.001, 0.001]]).repeat(20, 1)
    assert diagnostic.dormant_fraction(layer, 0.1) == pytest.approx(2 / 8)
    assert diagnostic.dormant_fraction(layer, 0.025) == pytest.approx(2 / 8)
    # Scaling the whole layer changes nothing: the score is normalised.
    assert diagnostic.dormant_fraction(layer * 1000.0, 0.1) == pytest.approx(2 / 8)


def test_a_wholly_silent_layer_is_wholly_dormant() -> None:
    """The extreme case has no mean to normalise by and must not divide by zero."""
    assert diagnostic.dormant_fraction(torch.zeros(10, 4), 0.025) == 1.0


def test_the_ranks_report_how_many_directions_the_features_use() -> None:
    """Both statistics are held against a matrix whose rank is known."""
    # One direction, repeated: stable rank 1, and one singular value carries
    # everything.
    single = torch.ones(32, 6)
    assert diagnostic.stable_rank(single) == pytest.approx(1.0, abs=1e-6)
    assert diagnostic.srank(single) == 1

    # Six orthogonal directions of equal size: every direction is used equally,
    # so the stable rank is the full six and 99% of the singular mass needs all
    # six of them.
    full = torch.eye(6).repeat(5, 1)
    assert diagnostic.stable_rank(full) == pytest.approx(6.0, abs=1e-6)
    assert diagnostic.srank(full) == 6


def test_the_parameter_total_is_the_norm_of_the_whole_vector() -> None:
    """Not the sum of the per-tensor norms, which would be a different curve."""
    backbone = StackedDqnBackbone(
        config=LEARNER, network_config=NETWORK, device=torch.device("cpu")
    )
    norms, total = diagnostic.parameter_norms(backbone.online)
    assert set(norms) == {name for name, _ in backbone.online.named_parameters()}
    flat = torch.cat([p.detach().reshape(-1) for p in backbone.online.parameters()])
    assert total == pytest.approx(float(flat.norm()), rel=1e-5)


def test_every_measured_layer_is_reached_and_flattened_to_its_units(tmp_path: Path) -> None:
    """The hooks must fire on the modules named, at the widths they emit."""
    backbone = StackedDqnBackbone(
        config=LEARNER, network_config=NETWORK, device=torch.device("cpu")
    )
    batch = diagnostic.read_observations(written_batch(tmp_path / "batch.pt"))
    layers = diagnostic.activations(backbone.online, batch)

    assert set(layers) == set(diagnostic.MEASURED_LAYERS)
    steps = int(batch["scalars"].shape[0] * batch["scalars"].shape[1])
    assert layers["trunk.scalar_encoder"].shape == (steps, NETWORK.hidden)
    assert layers["core"].shape == (steps, NETWORK.core_hidden)
    # One sample per upgrade row per step: the row encoder is shared over rows.
    assert layers["trunk.row_encoder"].shape == (steps * ROW_COUNT, NETWORK.hidden)


def test_a_batch_of_the_wrong_shape_is_refused_rather_than_measured(tmp_path: Path) -> None:
    """A batch from another feature schema would produce numbers that mean nothing."""
    path = tmp_path / "wrong.pt"
    torch.save({"scalars": torch.zeros(2, 3, SCALAR_COUNT + 1)}, path)
    with pytest.raises(diagnostic.ObservationBatchError):
        diagnostic.read_observations(path)

    torch.save(
        {
            "scalars": torch.zeros(2, 3, SCALAR_COUNT),
            "rows": torch.zeros(2, 3, ROW_COUNT, ROW_WIDTH),
            "mask": torch.zeros(2, 3, len(RUN_ACTIONS), dtype=torch.bool),
        },
        path,
    )
    with pytest.raises(diagnostic.ObservationBatchError):
        diagnostic.read_observations(path)


def test_the_run_reports_every_checkpoint_in_budget_order(tmp_path: Path) -> None:
    """The curve is read down the page, so the rows are ordered by decisions."""
    late = written_checkpoint(tmp_path, 200, seed=3)
    early = written_checkpoint(tmp_path, 100, seed=4)
    output = tmp_path / "out" / "diagnosis.json"

    assert (
        diagnostic.main(
            [
                str(late),
                str(early),
                "--output",
                str(output),
                "--observations",
                str(written_batch(tmp_path / "batch.pt")),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text())
    assert [row["decisions"] for row in report["checkpoints"]] == [100, 200]
    assert report["observations"]["count"] == 36
    for row in report["checkpoints"]:
        assert set(row["dormant_fraction"]) == set(diagnostic.MEASURED_LAYERS)
        assert set(row["dormant_fraction"]["core"]) == {"0.025", "0.1"}
        assert row["rank"]["layer"] == "core"
        assert 0.0 < row["rank"]["stable_rank"] <= NETWORK.core_hidden
        assert 1 <= row["rank"]["srank_99"] <= NETWORK.core_hidden
        assert row["parameter_norm_total"] > 0.0


def test_the_parameter_norms_alone_need_no_observations(tmp_path: Path) -> None:
    """The one measurement that is a property of the weights and nothing else."""
    path = written_checkpoint(tmp_path, 100, seed=5)
    output = tmp_path / "diagnosis.json"

    assert diagnostic.main([str(path), "--output", str(output)]) == 0

    report = json.loads(output.read_text())
    assert report["observations"] is None
    (row,) = report["checkpoints"]
    assert "dormant_fraction" not in row
    assert "rank" not in row
    assert row["parameter_norms"]["core.0.weight"] > 0.0


class CountingPolicy:
    """A policy that always waits and counts the episodes it was started for."""

    def __init__(self) -> None:
        self.episodes = 0

    def initial_state(self) -> int:
        self.episodes += 1
        return 0

    def act(self, features: StateFeatures, state: int, *, epsilon: float = 0.0) -> tuple[int, int]:
        return 0, state + 1


def one_state(value: float) -> StateFeatures:
    """A state distinguishable from every other, so order can be asserted."""
    return StateFeatures(
        scalars=(value,) * SCALAR_COUNT,
        rows=(value,) * (ROW_COUNT * ROW_WIDTH),
        mask=(True,) * len(RUN_ACTIONS),
    )


def play(recorder: run_episodes.RecordingPolicy, decisions: int, *, value: float = 0.0) -> None:
    """One episode of `decisions` states, driven the way an actor drives a policy."""
    state = recorder.initial_state()
    for step in range(decisions):
        _action, state = recorder.act(one_state(value + step), state, epsilon=0.0)


def test_a_recorded_window_never_straddles_two_episodes(tmp_path: Path) -> None:
    """The stacked history returns to zeros at an episode start.

    Two episodes of a window and a half produce one whole window each and not
    three: the tail of an episode is dropped rather than joined to the start of
    the next, which would offer the diagnostic a history no decision ever had.
    """
    inner = CountingPolicy()
    recorder = run_episodes.RecordingPolicy(inner, tmp_path / "batch.pt", window=8)
    play(recorder, 12)
    play(recorder, 12)

    windows = recorder.windows()
    assert [len(window) for window in windows] == [8, 8]
    assert inner.episodes == 2
    # In decision order, from each episode's first decision.
    assert windows[0][0].scalars[0] == 0.0
    assert windows[1][0].scalars[0] == 0.0


def test_the_batch_is_written_after_every_episode(tmp_path: Path) -> None:
    """A session cut short still leaves the episodes it finished."""
    path = tmp_path / "batch.pt"
    recorder = run_episodes.RecordingPolicy(CountingPolicy(), path, window=8)
    play(recorder, 8)
    assert not path.exists()  # nothing is complete until the episode ends

    play(recorder, 8)  # starting the second episode flushes the first
    assert diagnostic.read_observations(path)["scalars"].shape[0] == 1

    assert recorder.write() == 16
    assert diagnostic.read_observations(path)["scalars"].shape[0] == 2


def test_the_recorded_batch_is_what_the_diagnostic_reads(tmp_path: Path) -> None:
    """The format is the contract between the capture and the measurement."""
    path = tmp_path / "batch.pt"
    recorder = run_episodes.RecordingPolicy(CountingPolicy(), path, window=8)
    play(recorder, 20)
    assert recorder.write() == 16

    batch = diagnostic.read_observations(path)
    assert batch["scalars"].shape == (2, 8, SCALAR_COUNT)
    assert batch["rows"].shape == (2, 8, ROW_COUNT, ROW_WIDTH)
    assert batch["mask"].shape == (2, 8, len(RUN_ACTIONS))
    # Row features are flattened in action order by `StateFeatures`; the batch
    # must present them as the network's [row, feature] grid, not transposed.
    assert batch["rows"][0, 3].tolist() == [[3.0] * ROW_WIDTH] * ROW_COUNT
