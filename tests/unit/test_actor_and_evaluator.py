from __future__ import annotations

import math

import pytest
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.episode import (
    EpisodeSummary,
    RunTransition,
    TerminationOutcome,
    WaveRecord,
)
from tower_rl.environment.run_actions import RunActionId
from tower_rl.environment.run_environment import (
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.experiment.comparison import bootstrap_difference, cohens_d
from tower_rl.learning.actor import Actor, ActorConfig
from tower_rl.learning.evaluator import (
    WaveDistribution,
    episode_record,
    evaluate,
    to_record,
)
from tower_rl.learning.policies import (
    CheapestFirstPolicy,
    RandomPolicy,
    WaitOnlyPolicy,
)
from tower_rl.learning.replay import PrioritizedSequenceReplay, ReplayStep

PROFILE = "fake-profile-v1"


def _environment(**port_kwargs: object) -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(**port_kwargs),  # type: ignore[arg-type]
        builder=RunStateBuilder(profile_id=PROFILE),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
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
    assert stored.metadata.observation_schema == "observation-v2"


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


def test_a_wholly_failed_arm_still_says_why_every_episode_failed() -> None:
    """The reasons are all a fully-failed arm has to report; it must report them."""
    with pytest.raises(ValueError, match="no_answer_from_the_bridge"):
        evaluate(
            _environment(
                damage_per_second=1.0,
                ambiguous_advance_episodes=frozenset({1, 2, 3}),
            ),
            CheapestFirstPolicy(),
            episodes=3,
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


def test_the_report_carries_decision_density_and_the_speed_up() -> None:
    """These are the numbers the stepping design is accepted or rejected on.

    Decisions per episode is compared against the 1x reference of 89.3
    (`M1B-E014`); measured game seconds — the round clock — over wall seconds is
    what moving the advance loop into the bridge was for.
    """
    report = evaluate(
        _environment(damage_per_second=1.0),
        CheapestFirstPolicy(),
        episodes=3,
        profile_id=PROFILE,
    )

    assert report.total_frames > 0
    assert report.total_budgeted_game_seconds > 0.0
    # The game's own clock beside the budgeted game time: the design assumes they
    # are the same number, so the report has to show both.
    assert report.total_round_seconds == pytest.approx(report.total_budgeted_game_seconds)
    assert report.total_advance_wall_seconds > 0.0
    assert report.decisions_per_episode == (
        report.decisions_in_valid_episodes / report.valid_episodes
    )
    assert report.decisions_per_wave == (
        report.decisions_per_episode / report.distribution.mean
    )
    # Wall time is rounded to a hundredth, so a fast fake run can report zero;
    # the guard, not the ratio, is what matters then.
    expected = (
        report.total_round_seconds / report.total_wall_seconds
        if report.total_wall_seconds > 0
        else 0.0
    )
    assert report.speedup == expected

    record = to_record(report)
    for key in (
        "decisions_per_episode",
        "decisions_per_wave",
        "total_frames",
        "total_budgeted_game_seconds",
        "total_round_seconds",
        "total_advance_wall_seconds",
        "advances_cut_short",
        "speedup",
    ):
        assert key in record


def test_decision_density_is_a_mean_over_the_valid_episodes_alone() -> None:
    """An invalid episode's decisions must not be divided by the valid count.

    Two valid episodes of 100 decisions and one failed episode that managed 40
    is a density of 100, not 120: the failed episode is an environment failure,
    and charging its decisions to the episodes that survived would flatter
    exactly the arms that failed most.
    """
    from tower_rl.learning.evaluator import EvaluationReport

    report = EvaluationReport(
        policy="p",
        profile_id=PROFILE,
        model_version=0,
        game_speed=1.0,
        valid_episodes=2,
        invalid_episodes=1,
        distribution=WaveDistribution.of([10, 10]),
        total_decisions=240,
        decisions_in_valid_episodes=200,
    )

    assert report.decisions_per_episode == 100.0
    assert report.decisions_per_wave == 10.0


def test_the_new_metrics_are_guarded_against_an_empty_denominator() -> None:
    from tower_rl.learning.evaluator import EvaluationReport

    empty = EvaluationReport(
        policy="p",
        profile_id=PROFILE,
        model_version=0,
        game_speed=1.0,
        valid_episodes=0,
        invalid_episodes=2,
        distribution=WaveDistribution.of([0]),
    )

    assert empty.decisions_per_episode == 0.0
    assert empty.decisions_per_wave == 0.0
    assert empty.speedup == 0.0


def test_invalid_episodes_carry_their_validator_reason() -> None:
    """A rate without reasons cannot be fixed; M1B-E007 needed the text."""
    from tower_rl.learning.evaluator import EvaluationReport, WaveDistribution, to_record

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


def _summary(**overrides: object) -> EpisodeSummary:
    defaults: dict[str, object] = dict(
        episode_id="ep",
        profile_id=PROFILE,
        final_wave=5,
        decisions=10,
        purchases=2,
        termination=TerminationOutcome.GAME_OVER,
        elapsed_wall_seconds=1.0,
        game_speed=8.0,
        invalid_transitions=0,
        frames=100,
        game_ms=1000.0,
        round_ms=1000.0,
        advance_wall_seconds=0.5,
        advances_cut_short=0,
        termination_detail=(),
        recovered_transients=0,
        starting_wave=1,
    )
    defaults.update(overrides)
    return EpisodeSummary(**defaults)  # type: ignore[arg-type]


def test_per_episode_records_cover_valid_and_invalid_episodes_alike() -> None:
    """Every attempted episode is carried through, not only the valid ones.

    This is the gap that used to force a throwaway observer wrapper around
    every statistical comparison: the aggregates alone cannot feed a bootstrap
    interval or Cohen's d.
    """
    from tower_rl.learning.evaluator import EvaluationReport

    valid = _summary(episode_id="valid", final_wave=7, starting_wave=1)
    invalid = _summary(
        episode_id="invalid",
        final_wave=2,
        termination=TerminationOutcome.OBSERVATION_INVALID,
        termination_detail=("state: health exceeds maximum",),
        starting_wave=3,
    )
    report = EvaluationReport(
        policy="p",
        profile_id=PROFILE,
        model_version=0,
        game_speed=8.0,
        valid_episodes=1,
        invalid_episodes=1,
        distribution=WaveDistribution.of([7]),
        episodes=(valid, invalid),
        episodes_not_started_fresh=1,
    )

    record = to_record(report)

    assert record["episodes_not_started_fresh"] == 1
    episodes = record["episodes"]
    assert len(episodes) == 2

    first, second = episodes
    assert first["episode_index"] == 0
    assert first["valid"] is True
    assert first["final_wave"] == 7
    assert first["invalid_reasons"] == ()
    assert first["starting_wave"] == 1
    for key in (
        "decisions",
        "purchases",
        "frames",
        "budgeted_game_ms",
        "round_ms",
        "advance_wall_seconds",
        "elapsed_wall_seconds",
        "termination_detail",
        "advances_cut_short",
        "pin_restarts",
        "recovered_transients",
    ):
        assert key in first

    assert second["episode_index"] == 1
    assert second["valid"] is False
    assert second["final_wave"] == 2
    assert second["invalid_reasons"] == ("state: health exceeds maximum",)
    assert second["termination_detail"] == ("state: health exceeds maximum",)
    assert second["starting_wave"] == 3


def test_the_episode_record_carries_the_restarts_its_boundary_needed() -> None:
    """`#57`: a pin the port had to restart the boundary for is on the episode.

    The episode itself is ordinary - the restart happened before it began - so
    nothing about it says the instance needed help unless the count does.
    """
    record = episode_record(0, _summary(pin_restarts=2))

    assert record["pin_restarts"] == 2


def test_episode_record_matches_the_episode_summary_it_wraps() -> None:
    summary = _summary(recovered_transients=1)

    record = episode_record(3, summary)

    assert record["episode_index"] == 3
    assert record["recovered_transients"] == 1
    assert record["budgeted_game_ms"] == summary.game_ms


def test_starting_wave_flags_an_episode_that_did_not_start_fresh() -> None:
    """A fresh run always starts at wave 1; anything higher continued a leftover run."""
    report = evaluate(
        _environment(damage_per_second=1.0, starting_wave=3),
        CheapestFirstPolicy(),
        episodes=2,
        profile_id=PROFILE,
    )

    assert all(summary.starting_wave == 3 for summary in report.episodes)
    assert report.episodes_not_started_fresh == len(report.episodes)


def test_a_fresh_run_is_not_counted_as_contaminated() -> None:
    report = evaluate(
        _environment(damage_per_second=1.0),
        CheapestFirstPolicy(),
        episodes=2,
        profile_id=PROFILE,
    )

    assert all(summary.starting_wave == 1 for summary in report.episodes)
    assert report.episodes_not_started_fresh == 0


def test_per_episode_records_feed_the_comparison_protocol_directly() -> None:
    """No throwaway observer wrapper: the report's own episodes are enough.

    `bootstrap_difference` and `cohens_d` both consume per-episode samples;
    this proves `report.episodes` is that sample without any adapter between
    the evaluator and `comparison.py`.
    """
    scripted = evaluate(
        _environment(damage_per_second=2.0),
        CheapestFirstPolicy(),
        episodes=6,
        profile_id=PROFILE,
    )
    waiting = evaluate(
        _environment(damage_per_second=2.0),
        WaitOnlyPolicy(),
        episodes=6,
        profile_id=PROFILE,
    )

    scripted_waves = [summary.final_wave for summary in scripted.episodes if summary.valid]
    waiting_waves = [summary.final_wave for summary in waiting.episodes if summary.valid]

    observed, low, high = bootstrap_difference(scripted_waves, waiting_waves, seed=0)
    effect_size = cohens_d(scripted_waves, waiting_waves)

    assert observed > 0, "buying survives longer, so its waves should lead"
    assert low <= observed <= high
    assert effect_size > 0


def test_every_episodes_wave_rows_partition_its_totals() -> None:
    """Per-wave rows are a partition of the episode, not a second measurement.

    If they ever stopped summing to the episode's own measured round time and
    decision count, the per-wave gate would be comparing something the episode
    record does not claim to have spent.
    """
    report = evaluate(
        _environment(damage_per_second=0.4, seconds_per_wave=4.0),
        CheapestFirstPolicy(),
        episodes=4,
        profile_id=PROFILE,
    )

    assert report.episodes
    for summary in report.episodes:
        assert summary.waves
        assert [wave.wave for wave in summary.waves] == sorted(
            wave.wave for wave in summary.waves
        )
        assert sum(wave.game_ms for wave in summary.waves) == pytest.approx(summary.round_ms)
        assert sum(wave.decisions for wave in summary.waves) == summary.decisions
        # Only the wave the episode ended in is a fragment.
        assert [wave.completed for wave in summary.waves] == [True] * (len(summary.waves) - 1) + [
            False
        ]


def test_the_episode_record_carries_the_wave_rows_the_analysis_reads() -> None:
    """The keys are `experiment.wave_statistics.wave_observations` reads, exactly."""
    wave = WaveRecord(
        wave=2, completed=True, game_ms=30_000.0, decisions=9, advances=14,
        health_fraction=0.8, cash_log=4.1,
    )

    record = episode_record(0, _summary(waves=(wave,)))

    assert record["waves"] == [
        {
            "wave": 2,
            "completed": True,
            "game_ms": 30_000.0,
            "decisions": 9,
            "advances": 14,
            "health_fraction": 0.8,
            "cash_log": 4.1,
        }
    ]


def test_an_episode_that_died_before_any_choice_is_scored_not_failed() -> None:
    """A zero-decision episode is an episode (ADR 0009).

    Under choice points a run can end before it ever offers a purchase. Nothing
    was decided, so nothing reaches replay - but the tower did die at a wave,
    which is what evaluation measures, so the episode is scored rather than
    counted as an environment failure.
    """
    settings: dict[str, object] = {
        "offered": {"attack": 1, "defense": 0, "utility": 0},
        "start_cash": 0.0,
        "cash_per_second": 0.0,
        "damage_per_second": 4.0,
        "max_health": 1.0,
    }
    buffer = PrioritizedSequenceReplay(capacity=64, seed=0)
    actor = Actor(
        environment=_environment(**settings), policy=CheapestFirstPolicy(), replay=buffer
    )

    result = actor.run_episode()

    assert result.summary.valid and result.summary.decisions == 0
    assert result.sequences_offered == 0 and result.sequences_accepted == 0
    assert len(buffer) == 0, "there is no decision to learn from"

    report = evaluate(
        _environment(**settings), CheapestFirstPolicy(), episodes=3, profile_id=PROFILE
    )

    assert report.valid_episodes == 3 and report.invalid_episodes == 0
    assert report.total_decisions == 0
    assert report.decisions_per_episode == 0.0
    assert report.distribution.mean >= 1.0


def test_each_stored_step_carries_the_game_time_its_transition_spanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T7: the actor keeps `transition.game_ms`; a purchase spans none of it."""
    environment = _environment(damage_per_second=1.0)
    spans: list[float] = []
    step = environment.step

    def recorded(action: RunActionId) -> RunTransition:
        transition = step(action)
        spans.append(transition.game_ms)
        return transition

    monkeypatch.setattr(environment, "step", recorded)
    actor = Actor(environment=environment, policy=CheapestFirstPolicy())
    stored: list[ReplayStep] = []
    monkeypatch.setattr(
        actor, "_emit", lambda steps, summary: (stored.extend(steps), (0, 0))[1]
    )

    result = actor.run_episode()

    assert [step.game_ms for step in stored] == spans
    assert 0.0 in spans, "a confirmed purchase takes no game time"
    assert any(span > 0.0 for span in spans), "a wait does"
    assert result.reward_bearing_transitions == sum(1 for s in stored if s.reward != 0.0)
    assert 0 <= result.wave_change_ended_span <= result.reward_bearing_transitions
