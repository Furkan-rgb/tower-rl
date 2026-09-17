from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
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

SMALL = NetworkConfig(hidden=16, recurrent_hidden=16, identity_dim=4)


def _run(**overrides: object) -> TrainingRun:
    environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(slice_game_ms=250, max_quiet_game_ms=1000),
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
