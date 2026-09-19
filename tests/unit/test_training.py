from __future__ import annotations

import functools
import statistics

import pytest
import torch
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.episode import (
    EpisodeSummary,
    TerminationOutcome,
)
from tower_rl.environment.run_environment import (
    BRIDGE_EVENT_DIVERGENCE,
    GAME_TIME_INFLATED,
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.actor import Actor, ActorConfig
from tower_rl.learning.evaluator import EvaluationReport
from tower_rl.learning.exploration import ExplorationSchedule
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.replay import PrioritizedSequenceReplay
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig
from tower_rl.learning.training import (
    STALE_OR_DUPLICATE,
    CollectedEpisode,
    TrainingConfig,
    TrainingRun,
    action_distribution,
    collection_windows,
    episode_budget,
    episode_health,
)

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


def _run(*, device: torch.device | None = None, **overrides: object) -> TrainingRun:
    environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    backbone = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2),
        network_config=SMALL,
        device=device or torch.device("cpu"),
    )
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
        actors=[actor],
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
    assert report.recent_weighted_losses and all(
        loss >= 0 for loss in report.recent_weighted_losses
    )


def test_the_budget_is_counted_in_decisions_not_episodes() -> None:
    """A stronger policy survives longer; an episode budget would flatter it."""
    short = _run(budget_decisions=60).run()
    long = _run(budget_decisions=240).run()

    assert long.decisions > short.decisions
    assert long.decisions >= 240 and short.decisions >= 60
    # Episodes are a consequence of the budget, never the budget itself.
    assert episode_budget(TrainingConfig(budget_decisions=240), 40.0) == 6


def test_exploration_anneals_over_its_horizon_and_then_holds() -> None:
    """Over a horizon in decisions, never over the budget.

    Annealing across the whole budget is what left the first run collecting at a
    mean epsilon of 0.525: more than half of it was near-random data, and the
    collection episodes could not be read as a policy's performance at all.
    """
    config = TrainingConfig(
        budget_decisions=20_000,
        exploration=ExplorationSchedule(
            epsilon_start=1.0, epsilon_end=0.05, anneal_decisions=10_000
        ),
    )
    epsilon = functools.partial(config.exploration.epsilon_for, 0)

    assert epsilon(0) == pytest.approx(1.0)
    assert epsilon(5_000) == pytest.approx(0.525)
    assert epsilon(10_000) == pytest.approx(0.05)
    # Held for the whole second half of the budget, not annealed further.
    assert epsilon(15_000) == pytest.approx(0.05)
    assert epsilon(20_000) == pytest.approx(0.05)


def test_the_anneal_horizon_does_not_move_with_the_budget() -> None:
    """The same horizon means the same exploration whatever the budget is."""
    schedule = ExplorationSchedule(anneal_decisions=10_000)
    short = TrainingConfig(budget_decisions=12_000, exploration=schedule)
    long = TrainingConfig(budget_decisions=200_000, exploration=schedule)

    assert short.exploration.epsilon_for(0, 5_000) == pytest.approx(
        long.exploration.epsilon_for(0, 5_000)
    )
    assert long.exploration.epsilon_for(0, 10_001) == pytest.approx(
        schedule.epsilon_end
    )


def test_a_horizon_of_no_decisions_is_refused() -> None:
    with pytest.raises(ValueError, match="anneal horizon"):
        TrainingConfig(
            budget_decisions=100, exploration=ExplorationSchedule(anneal_decisions=0)
        )


def test_importance_sampling_correction_anneals_the_other_way() -> None:
    config = TrainingConfig(budget_decisions=100, beta_start=0.4, beta_end=1.0)

    assert config.beta(0) == pytest.approx(0.4)
    assert config.beta(100) == pytest.approx(1.0)


def test_the_run_publishes_the_exploration_and_importance_values_it_used() -> None:
    """The schedules are the run's, so the values it drew are on its report.

    Whatever records a run - a checkpoint, a curve point - needs the epsilon the
    fleet actually acted at and the beta the learner actually sampled at. Read
    from the report, so nothing outside `learning` has to evaluate a schedule of
    its own and arrive at a value the run never used.
    """
    training = _run(
        budget_decisions=150,
        exploration=ExplorationSchedule(
            epsilon_start=1.0, epsilon_end=0.05, anneal_decisions=150
        ),
    )

    # Before a decision is spent: exactly where the schedules start.
    assert training.report.epsilon == pytest.approx(1.0)
    assert training.report.importance_beta == pytest.approx(training.config.beta_start)

    report = training.run()

    assert report.optimisation_steps > 0
    # The value of the last episode it started, which is the last one it acted
    # at: drawn once per episode, so it lags the schedule by that episode.
    def annealed(decisions: int) -> float:
        return training.config.exploration.epsilon_for(0, decisions)
    assert annealed(report.decisions) <= report.epsilon < 1.0
    assert report.epsilon == pytest.approx(
        annealed(report.decisions - report.collected[-1].summary.decisions)
    )
    assert training.config.beta_start <= report.importance_beta <= training.config.beta_end
    assert report.importance_beta > training.config.beta_start


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
    from tower_rl.learning.evaluator import EvaluationReport, WaveDistribution

    evaluations = 0

    def evaluate() -> EvaluationReport:
        nonlocal evaluations
        evaluations += 1
        return EvaluationReport(
            policy="StackedDqnBackbone",
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
    from tower_rl.learning.evaluator import EvaluationReport, WaveDistribution

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
    training = _run(device=torch.device("cuda"))

    report = training.run()

    assert report.optimisation_steps > 0
    # And the copy the actor acts from is on the device the learner is on; a
    # copy left on the host would fail its first forward pass instead.
    acting = training.acting[training.actors[0].config.actor_id]
    assert acting.device == training.backbone.device


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

    assert training.report.mean_recent_weighted_loss is None

    report = training.advance(40)

    assert report.optimisation_steps > 0
    assert (
        report.mean_recent_weighted_loss is not None
        and report.mean_recent_weighted_loss >= 0.0
    )


def _failing_run(**overrides: object) -> TrainingRun:
    """A run whose port refuses the episodes named in `refuse_episodes`."""
    port = FakeRunPort(damage_per_second=2.0, refuse_episodes=frozenset({2, 3}))
    training = _run(**overrides)
    training.actors[0].environment = InstrumentedRunEnvironment(
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


def test_a_stale_sequence_costs_one_episode_and_not_the_run() -> None:
    """M1B-E024: the failure that used to kill an unattended overnight run.

    A command refused because it no longer binds the bridge's latest
    observation reaches the run as a port failure, so the episode is classified,
    counted and left behind while collection carries on. It is never retried:
    a retry would hide a stranded sequence rather than report it.
    """
    training = _run(budget_decisions=150)
    training.actors[0].environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0, stale_advance_episodes=frozenset({2, 3})),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )

    report = training.run()

    assert report.failed_episodes == 2
    assert all("stale" in failure for failure in report.episode_failures)
    assert report.decisions >= 150, "the run was not abandoned"
    assert report.valid_episodes > 0, "collection did not continue"


def test_an_instance_that_fails_every_episode_stops_the_run() -> None:
    """Continuing against a broken instance would spin without collecting."""
    training = _run(budget_decisions=150, max_consecutive_episode_failures=3)
    training.actors[0].environment = InstrumentedRunEnvironment(
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


def _collected(
    wave: int,
    *,
    decisions: int = 100,
    waits: int = 50,
    purchases: int = 10,
    valid: bool = True,
    termination_detail: tuple[str, ...] = (),
    advances_cut_short: int = 0,
    starting_wave: int = 0,
    game_ms: float = 0.0,
    round_ms: float = 0.0,
    actor_id: str = "actor-0",
) -> CollectedEpisode:
    """One collected episode, as the report keeps it."""
    return CollectedEpisode(
        summary=EpisodeSummary(
            episode_id=f"episode-{wave}",
            profile_id="fake-profile-v1",
            final_wave=wave,
            decisions=decisions,
            purchases=purchases,
            termination=(
                TerminationOutcome.GAME_OVER if valid else TerminationOutcome.DEVICE_FAILED
            ),
            elapsed_wall_seconds=1.0,
            game_speed=8.0,
            invalid_transitions=0,
            termination_detail=termination_detail,
            advances_cut_short=advances_cut_short,
            starting_wave=starting_wave,
            game_ms=game_ms,
            round_ms=round_ms,
        ),
        wait_decisions=waits,
        actor_id=actor_id,
    )


def test_the_collection_curve_is_cut_into_non_overlapping_windows() -> None:
    """The curve the run is read from: the collected episodes themselves."""
    collected = [_collected(wave) for wave in (4, 6, 8, 10)]

    windows = collection_windows(collected, size=2)

    assert [window.index for window in windows] == [0, 1]
    assert [window.episodes for window in windows] == [2, 2]
    assert [window.mean_final_wave for window in windows] == [5.0, 9.0]
    # Each window is placed where the budget had been spent to at its last
    # episode, and carries the decisions spent inside it.
    assert [window.decisions_at_end for window in windows] == [200, 400]
    assert [window.decisions for window in windows] == [200, 200]
    assert windows[0].stdev_final_wave == pytest.approx(statistics.stdev([4, 6]))
    assert windows[0].standard_error == pytest.approx(
        statistics.stdev([4, 6]) / 2 ** 0.5
    )


def test_a_window_that_has_not_closed_is_not_a_point() -> None:
    """A point averaged over fewer episodes has a different standard error."""
    windows = collection_windows([_collected(4), _collected(6), _collected(8)], size=2)

    assert len(windows) == 1
    assert windows[0].mean_final_wave == 5.0


def test_an_invalid_episode_costs_decisions_without_scoring_a_window() -> None:
    """It has no final wave to average, but its decisions were still spent."""
    collected = [_collected(4), _collected(99, valid=False), _collected(6)]

    windows = collection_windows(collected, size=2)

    assert len(windows) == 1
    assert windows[0].mean_final_wave == 5.0
    assert windows[0].decisions_at_end == 300


def test_a_window_reports_what_the_policy_did_in_it() -> None:
    """A collapsed policy waits out every decision and buys nothing."""
    collected = [
        _collected(4, decisions=100, waits=95, purchases=0),
        _collected(6, decisions=100, waits=85, purchases=2),
    ]

    windows = collection_windows(collected, size=2)

    assert windows[0].wait_fraction == pytest.approx(0.9)
    assert windows[0].purchases_per_episode == pytest.approx(1.0)


def test_a_window_is_also_cut_per_actor_and_over_the_near_greedy_ones() -> None:
    """The pooled mean of a laddered fleet is nobody's performance.

    Actors exploring at rates two orders of magnitude apart are averaged into
    the pooled series, so the window carries each actor's own mean beside it and
    a pooled mean over the actors that are near-greedy - the one a readout of
    what the policy itself reaches can cite.
    """
    collected = [
        _collected(2, actor_id="searcher"),
        _collected(4, actor_id="greedy-a"),
        _collected(8, actor_id="greedy-b"),
        _collected(6, actor_id="greedy-a"),
    ]

    windows = collection_windows(
        collected, size=4, near_greedy_actor_ids={"greedy-a", "greedy-b"}
    )

    window = windows[0]
    assert window.mean_final_wave == pytest.approx(5.0), "the pooled series is unchanged"
    assert window.mean_final_wave_by_actor == {
        "searcher": pytest.approx(2.0),
        "greedy-a": pytest.approx(5.0),
        "greedy-b": pytest.approx(8.0),
    }
    # Pooled over the near-greedy episodes, not averaged over per-actor means:
    # the actors need not have played the same number of episodes.
    assert window.near_greedy_episodes == 3
    assert window.near_greedy_mean_final_wave == pytest.approx(6.0)


def test_a_uniform_fleet_s_near_greedy_series_is_its_pooled_series() -> None:
    """Every actor draws the one rate, so naming none of them names all of them."""
    collected = [_collected(4, actor_id="actor-0"), _collected(6, actor_id="actor-1")]

    window = collection_windows(collected, size=2)[0]

    assert window.near_greedy_episodes == window.episodes == 2
    assert window.near_greedy_mean_final_wave == pytest.approx(window.mean_final_wave)


def test_a_window_pools_health_over_every_episode_attempted_in_it() -> None:
    """Not only the valid episodes: the invalid one attempted inside it too.

    A health problem must be locatable in time, not only in the run's total, so
    a window's health is pooled over every episode attempted while it was
    filling - including the invalid one that cost decisions but scored nothing.
    """
    collected = [
        _collected(4),
        _collected(99, valid=False, termination_detail=("device offline",)),
        _collected(6),
    ]

    windows = collection_windows(collected, size=2)

    assert len(windows) == 1
    health = windows[0].health
    assert health.episodes == 3
    assert health.valid_episodes == 2
    assert health.invalid_episodes == 1
    assert health.invalid_detail == {"device offline": 1}


def test_episode_health_names_a_lifecycle_reason_in_the_aggregate() -> None:
    """A failure of this exact shape must not survive only as free text on one episode."""
    reason = "the game did not honour speed_down: lifecycle_timeout"
    summaries = [
        _collected(4).summary,
        _collected(6).summary,
        _collected(99, valid=False, termination_detail=(reason,)).summary,
    ]

    health = episode_health(summaries)

    assert health.episodes == 3
    assert health.valid_episodes == 2
    assert health.invalid_episodes == 1
    assert health.invalid_by_reason == {TerminationOutcome.DEVICE_FAILED.value: 1}
    assert health.invalid_detail == {reason: 1}


def test_episode_health_pools_the_round_budgeted_ratio_and_keeps_the_worst() -> None:
    summaries = [
        _collected(4, game_ms=1000.0, round_ms=1100.0).summary,
        _collected(6, game_ms=1000.0, round_ms=1300.0).summary,
        # No measurable game time: excluded from both the pool and the worst.
        _collected(8, game_ms=0.0, round_ms=0.0).summary,
    ]

    health = episode_health(summaries)

    assert health.round_budgeted_ratio == pytest.approx(1.2)
    assert health.worst_round_budgeted_ratio == pytest.approx(1.3)


def test_episode_health_has_no_ratio_before_any_episode_measured_game_time() -> None:
    health = episode_health([_collected(4).summary])

    assert health.round_budgeted_ratio is None
    assert health.worst_round_budgeted_ratio is None


def test_episode_health_counts_the_named_bridge_and_device_failures() -> None:
    summaries = [
        _collected(4, valid=False, termination_detail=(BRIDGE_EVENT_DIVERGENCE,)).summary,
        _collected(
            6, valid=False, termination_detail=(f"rejected: {STALE_OR_DUPLICATE}",)
        ).summary,
        _collected(
            8,
            valid=False,
            termination_detail=(f"{GAME_TIME_INFLATED}: round clock ran 1.4x",),
        ).summary,
    ]

    health = episode_health(summaries)

    assert health.bridge_event_divergence == 1
    assert health.stale_or_duplicate == 1
    assert health.game_time_inflated == 1
    assert health.advances_cut_short == 0
    assert health.episodes_not_started_fresh == 0


def test_episode_health_counts_a_leftover_run_and_cut_short_advances() -> None:
    summaries = [
        _collected(4, starting_wave=3).summary,
        _collected(6, advances_cut_short=2).summary,
    ]

    health = episode_health(summaries)

    assert health.episodes_not_started_fresh == 1
    assert health.advances_cut_short == 2


def test_the_action_distribution_is_none_before_any_episode() -> None:
    assert action_distribution([]) is None
    distribution = action_distribution([_collected(4, decisions=10, waits=2, purchases=3)])
    assert distribution is not None
    assert distribution.wait_fraction == pytest.approx(0.2)
    assert distribution.purchases_per_episode == pytest.approx(3.0)


def test_a_run_records_the_action_distribution_of_what_it_collected() -> None:
    report = _run(budget_decisions=100).run()

    distribution = action_distribution(report.collected)

    assert distribution is not None
    assert distribution.episodes == len(report.collected)
    assert 0.0 <= distribution.wait_fraction <= 1.0
    assert distribution.decisions == report.decisions


def test_the_weighted_loss_and_the_unweighted_td_error_are_reported_apart() -> None:
    """Two signals, two names: the weighted one moves with the beta schedule."""
    report = _run(budget_decisions=150).run()

    assert report.optimisation_steps > 0
    assert len(report.recent_weighted_losses) == len(report.recent_unweighted_td_errors)
    assert report.mean_recent_weighted_loss is not None
    assert report.mean_recent_unweighted_absolute_td_error is not None
    assert report.mean_recent_unweighted_absolute_td_error >= 0.0
    # The value fit is reported whenever a batch held completed episodes; when
    # none did it is absent rather than zero.
    fit = report.mean_recent_value_fit_correlation
    assert fit is None or -1.0 <= fit <= 1.0
