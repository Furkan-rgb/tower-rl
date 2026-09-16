from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.evaluator import (  # noqa: E402
    WaveDistribution,
    evaluate,
    to_record,
)
from tower_rl.application.policies import (  # noqa: E402
    CheapestFirstPolicy,
    RandomPolicy,
    WaitOnlyPolicy,
)
from tower_rl.application.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402

PROFILE = "fake-profile-v1"


def _environment(**port_kwargs: object) -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(**port_kwargs),  # type: ignore[arg-type]
        builder=RunStateBuilder(profile_id=PROFILE),
        cadence=CadenceConfig(slice_game_ms=250, max_quiet_game_ms=1000),
    )


def test_an_actor_plays_an_episode_and_emits_sequences() -> None:
    replay = PrioritizedSequenceReplay(capacity=64, seed=0)
    actor = Actor(
        environment=_environment(damage_per_second=1.0),
        policy=CheapestFirstPolicy(),
        config=ActorConfig(sequence_length=8, burn_in=2, stride=4),
        replay=replay,
    )

    result = actor.run_episode()

    assert result.summary.decisions > 0
    assert result.sequences_accepted > 0
    assert len(replay) == result.sequences_accepted
    stored = replay._items[0]
    assert len(stored.steps) == 8 and stored.burn_in == 2
    assert stored.metadata.profile_id == PROFILE
    assert stored.metadata.observation_schema == "observation-v1"


def test_buying_survives_longer_than_never_buying() -> None:
    """The scripted floor must actually be a harder floor than doing nothing."""
    buyer = Actor(environment=_environment(damage_per_second=1.0), policy=CheapestFirstPolicy())
    waiter = Actor(environment=_environment(damage_per_second=1.0), policy=WaitOnlyPolicy())

    bought = buyer.run_episode().summary
    waited = waiter.run_episode().summary

    assert bought.purchases > 0 and waited.purchases == 0
    assert bought.final_wave > waited.final_wave


def test_an_actor_without_replay_still_reports_its_episode() -> None:
    actor = Actor(environment=_environment(damage_per_second=2.0), policy=RandomPolicy(seed=1))

    result = actor.run_episode()

    assert result.sequences_offered == 0 and result.sequences_accepted == 0
    assert result.summary.decisions > 0


def test_evaluation_reports_a_distribution_not_a_best_run() -> None:
    report = evaluate(
        _environment(damage_per_second=1.0),
        CheapestFirstPolicy(),
        episodes=4,
        profile_id=PROFILE,
    )

    assert report.valid_episodes == 4
    assert report.distribution.episodes == 4
    assert report.distribution.minimum <= report.distribution.median
    assert report.distribution.median <= report.distribution.maximum
    # The one-line summary must carry the spread, not just the centre.
    line = report.summary_line()
    assert "sd" in line and "range" in line and "mean" in line


def test_evaluation_never_explores_even_if_configured_to() -> None:
    report = evaluate(
        _environment(damage_per_second=1.0),
        CheapestFirstPolicy(),
        episodes=2,
        profile_id=PROFILE,
        actor_config=ActorConfig(epsilon=1.0),
    )

    # A deterministic policy under a forced epsilon of zero must be repeatable.
    again = evaluate(
        _environment(damage_per_second=1.0),
        CheapestFirstPolicy(),
        episodes=2,
        profile_id=PROFILE,
        actor_config=ActorConfig(epsilon=1.0),
    )
    assert report.distribution.mean == again.distribution.mean


def test_a_single_episode_reports_undefined_spread_rather_than_zero() -> None:
    distribution = WaveDistribution.of([7])

    assert distribution.episodes == 1
    assert math.isnan(distribution.stdev), "one episode implies no certainty"


def test_distribution_summarizes_shape() -> None:
    distribution = WaveDistribution.of([4, 6, 8, 10, 12])

    assert distribution.mean == 8.0
    assert distribution.median == 8
    assert distribution.minimum == 4 and distribution.maximum == 12
    assert distribution.lower_quartile <= distribution.median


def test_an_arm_with_no_valid_episode_cannot_be_scored() -> None:
    with pytest.raises(ValueError, match="no valid episode"):
        evaluate(
            _environment(refuse_to_start=False, damage_per_second=1.0),
            CheapestFirstPolicy(),
            episodes=0,
            profile_id=PROFILE,
        )


def test_records_carry_everything_needed_to_compare_arms_later() -> None:
    report = evaluate(
        _environment(damage_per_second=1.0),
        CheapestFirstPolicy(),
        episodes=3,
        profile_id=PROFILE,
        model_version=17,
    )

    record = to_record(report)

    assert record["profile_id"] == PROFILE
    assert record["model_version"] == 17
    assert record["valid_episodes"] == 3
    assert "mean_final_wave" in record and "stdev_final_wave" in record
    assert "total_decisions" in record and "game_speed" in record


def test_invalid_episodes_carry_their_validator_reason() -> None:
    """A rate without reasons cannot be fixed; M1B-E007 needed the text."""
    from tower_rl.application.evaluator import EvaluationReport, WaveDistribution, to_record

    report = EvaluationReport(
        policy="CheapestFirstPolicy",
        profile_id=PROFILE,
        model_version=0,
        game_speed=64.0,
        valid_episodes=39,
        invalid_episodes=1,
        distribution=WaveDistribution.of([9, 10, 11]),
        invalid_by_reason={"observation_invalid": 1},
        invalid_detail={"state: health exceeds maximum": 1},
    )

    record = to_record(report)

    assert record["invalid_detail"] == {"state: health exceeds maximum": 1}
    assert record["invalid_rate"] == 0.025
