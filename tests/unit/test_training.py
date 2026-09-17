from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.evaluator import EvaluationReport  # noqa: E402
from tower_rl.application.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.application.training import (  # noqa: E402
    TrainingConfig,
    TrainingRun,
    episode_budget,
)
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig  # noqa: E402
from tower_rl.ports.run_port import RunPortError  # noqa: E402

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


def _run(**overrides: object) -> TrainingRun:
    environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    backbone = RecurrentQBackbone(config=RecurrentQConfig(seed=0), network_config=SMALL)
    replay = PrioritizedSequenceReplay(capacity=64, seed=0)
    actor = Actor(
        environment=environment,
        policy=backbone,
        config=ActorConfig(sequence_length=6, burn_in=1, stride=3),
        replay=replay,
    )
    settings: dict[str, object] = {
        "budget_decisions": 150,
        "warmup_sequences": 2,
        "batch_size": 2,
        "gradient_steps_per_decision": 0.2,
    }
    settings.update(overrides)
    return TrainingRun(
        actor=actor,
        replay=replay,
        backbone=backbone,
        config=TrainingConfig(**settings),  # type: ignore[arg-type]
    )


def test_a_run_spends_its_decision_budget_and_learns() -> None:
    training = _run()

    report = training.run()

    assert report.decisions >= 150
    assert report.episodes > 0
    assert report.optimisation_steps > 0
    assert training.backbone.model_version == report.optimisation_steps
    assert report.recent_losses and all(loss >= 0 for loss in report.recent_losses)


def test_the_budget_is_counted_in_decisions_not_episodes() -> None:
    """A stronger policy survives longer; an episode budget would flatter it."""
    short = _run(budget_decisions=60).run()
    long = _run(budget_decisions=240).run()

    assert long.decisions > short.decisions
    assert long.decisions >= 240 and short.decisions >= 60
    # Episodes are a consequence of the budget, never the budget itself.
    assert episode_budget(TrainingConfig(budget_decisions=240), 40.0) == 6


def test_exploration_anneals_across_the_budget() -> None:
    config = TrainingConfig(budget_decisions=1000, epsilon_start=1.0, epsilon_end=0.05)

    assert config.epsilon(0) == pytest.approx(1.0)
    assert config.epsilon(500) == pytest.approx(0.525)
    assert config.epsilon(1000) == pytest.approx(0.05)
    # Past the budget it clamps rather than going negative.
    assert config.epsilon(5000) == pytest.approx(0.05)


def test_importance_sampling_correction_anneals_the_other_way() -> None:
    config = TrainingConfig(budget_decisions=100, beta_start=0.4, beta_end=1.0)

    assert config.beta(0) == pytest.approx(0.4)
    assert config.beta(100) == pytest.approx(1.0)


def test_no_optimisation_happens_before_the_buffer_is_warm() -> None:
    training = _run(warmup_sequences=10_000, budget_decisions=80)

    report = training.run()

    assert report.optimisation_steps == 0
    assert training.backbone.model_version == 0


def test_a_gradient_debt_is_not_banked_while_the_buffer_fills() -> None:
    """Otherwise the first warm episode triggers a burst on almost no data."""
    training = _run(warmup_sequences=3, gradient_steps_per_decision=1.0, budget_decisions=120)

    report = training.run()

    # With a debt banked, steps would far exceed decisions seen since warm-up.
    assert report.optimisation_steps <= report.decisions


def test_configuration_refuses_impossible_budgets() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        TrainingConfig(budget_decisions=0)
    with pytest.raises(ValueError, match="gradient steps"):
        TrainingConfig(budget_decisions=10, gradient_steps_per_decision=0.0)
    with pytest.raises(ValueError, match="decisions per episode"):
        episode_budget(TrainingConfig(budget_decisions=10), 0)


def test_each_episode_is_reported_as_it_completes() -> None:
    seen: list[int] = []
    training = _run(budget_decisions=100)
    training.on_episode = lambda report: seen.append(report.episodes)

    report = training.run()

    assert seen == list(range(1, report.episodes + 1))


def test_evaluation_and_checkpointing_run_on_their_periods() -> None:
    from tower_rl.application.evaluator import EvaluationReport, WaveDistribution

    evaluations = 0

    def evaluate() -> EvaluationReport:
        nonlocal evaluations
        evaluations += 1
        return EvaluationReport(
            policy="RecurrentQBackbone",
            profile_id="fake-profile-v1",
            model_version=0,
            game_speed=8.0,
            valid_episodes=2,
            invalid_episodes=0,
            distribution=WaveDistribution.of([4, 5]),
        )

    training = _run(budget_decisions=200)
    training.config = TrainingConfig(
        budget_decisions=200,
        warmup_sequences=2,
        batch_size=2,
        gradient_steps_per_decision=0.2,
        evaluate_every_episodes=2,
        checkpoint_every_episodes=3,
    )
    written: list[int] = []
    training.evaluate = evaluate
    training.checkpoint = lambda report: written.append(report.episodes)

    report = training.run()

    assert evaluations == report.episodes // 2
    assert len(report.evaluations) == evaluations
    assert written == [episode for episode in range(1, report.episodes + 1) if episode % 3 == 0]
    assert report.checkpoints_written == len(written)


def test_periodic_hooks_are_off_when_their_period_is_zero() -> None:
    calls: list[str] = []
    training = _run(budget_decisions=80)
    training.evaluate = lambda: calls.append("evaluate")  # type: ignore[assignment,return-value]
    training.checkpoint = lambda report: calls.append("checkpoint")

    training.run()

    assert calls == [], "a zero period must disable the hook entirely"


def test_evaluation_does_not_consume_the_decision_budget() -> None:
    """Evaluation is measurement, not experience."""
    from tower_rl.application.evaluator import EvaluationReport, WaveDistribution

    training = _run(budget_decisions=120)
    training.config = TrainingConfig(
        budget_decisions=120,
        warmup_sequences=2,
        batch_size=2,
        gradient_steps_per_decision=0.2,
        evaluate_every_episodes=1,
    )
    training.evaluate = lambda: EvaluationReport(
        policy="p", profile_id="fake-profile-v1", model_version=0, game_speed=8.0,
        valid_episodes=99, invalid_episodes=0, distribution=WaveDistribution.of([9, 10]),
    )

    report = training.run()

    # The budget counts collected decisions only; evaluation episodes are not in it.
    assert report.decisions == sum(s.decisions for s in report.episode_summaries)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_run_on_an_accelerator_builds_its_batches_there() -> None:
    """A CPU batch handed to a CUDA model fails on the first optimisation step."""
    training = _run()
    training.backbone = RecurrentQBackbone(
        config=RecurrentQConfig(seed=0),
        network_config=SMALL,
        device=torch.device("cuda"),
    )
    training.actor.policy = training.backbone

    report = training.run()

    assert report.optimisation_steps > 0


def test_a_run_advances_in_blocks_and_carries_its_progress() -> None:
    """Arms sharing one device take turns, so a run must be resumable mid-budget."""
    training = _run(budget_decisions=2000)

    first = training.advance(40)
    after_first = (first.decisions, first.episodes, first.optimisation_steps)

    assert 0 < after_first[0] < 2000
    assert not training.finished

    second = training.advance(40)

    assert second is training.report, "progress is carried, not restarted"
    assert second.decisions > after_first[0]
    assert second.episodes > after_first[1]
    # Gradient steps earned in one block but not yet taken carry into the next.
    assert second.optimisation_steps > after_first[2]


def test_advancing_never_overruns_the_budget() -> None:
    training = _run(budget_decisions=60)

    training.advance(10_000)

    assert training.finished
    # The limit lands on an episode boundary, so the last episode may carry the
    # count past the budget; it may never stop short of it.
    assert training.report.decisions >= 60


def test_a_block_must_buy_at_least_one_decision() -> None:
    with pytest.raises(ValueError, match="at least one decision"):
        _run().advance(0)


def test_the_loss_window_is_reported_and_empty_before_any_step() -> None:
    training = _run(budget_decisions=2000)

    assert training.report.mean_recent_loss is None

    report = training.advance(40)

    assert report.optimisation_steps > 0
    assert report.mean_recent_loss is not None and report.mean_recent_loss >= 0.0


def _failing_run(**overrides: object) -> TrainingRun:
    """A run whose port refuses the episodes named in `refuse_episodes`."""
    port = FakeRunPort(damage_per_second=2.0, refuse_episodes=frozenset({2, 3}))
    training = _run(**overrides)
    training.actor.environment = InstrumentedRunEnvironment(
        port=port,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    return training


def test_an_episode_the_port_refuses_is_counted_and_the_run_continues() -> None:
    """A failed episode is an outcome, not the end of hours of collection."""
    training = _failing_run(budget_decisions=150)

    report = training.run()

    assert report.failed_episodes == 2
    assert len(report.episode_failures) == 2
    assert report.decisions >= 150, "the budget is still spent"
    # Attempts are counted in `episodes`; only the ones that produced a record
    # leave a summary behind.
    assert report.episodes == len(report.episode_summaries) + report.failed_episodes


def test_an_instance_that_fails_every_episode_stops_the_run() -> None:
    """Continuing against a broken instance would spin without collecting."""
    training = _run(budget_decisions=150, max_consecutive_episode_failures=3)
    training.actor.environment = InstrumentedRunEnvironment(
        port=FakeRunPort(refuse_to_start=True),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )

    with pytest.raises(RunPortError):
        training.run()

    assert training.report.failed_episodes == 3


def test_an_evaluation_that_cannot_be_scored_does_not_lose_the_run() -> None:
    """Evaluation is measurement; losing a measurement must not lose the run."""

    def refuse() -> EvaluationReport:
        raise ValueError("no valid episode was produced; the arm cannot be scored")

    training = _run(budget_decisions=100, evaluate_every_episodes=1)
    training.evaluate = refuse

    report = training.run()

    assert report.decisions >= 100
    assert report.evaluations == []
    assert report.evaluation_failures and "cannot be scored" in report.evaluation_failures[0]
