"""What a training run leaves behind, and whether it can be read afterwards.

The run itself is exercised end to end against the fake port in
`test_train_entry_point.py`; what is under test here is the record
`experiment/training_report.py` and `experiment/metrics.py` produce from it -
the learning curve and the checkpoint each point names, the collection curve,
every collected episode's own record, the pooled health counters, and the
per-actor account a fleet is read by.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import train
from test_train_entry_point import PROFILE, SMALL_NETWORK, arguments, environment, session

from tower_rl.experiment.metrics import health_counters
from tower_rl.experiment.run_identity import SCRIPTED_REFERENCE
from tower_rl.learning.checkpoint import fingerprint, load

#: Long enough that the run plays more than one episode, which is what closes a
#: window of the collection curve.
TRAINING_BUDGET = "300"

#: Two fake instances, which is the fleet arrangement a device run takes.
FLEET_BUDGET = "300"


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One short single-actor session, read by several checks."""
    return session(tmp_path_factory.mktemp("runs"), budget=TRAINING_BUDGET)


@pytest.fixture(scope="module")
def fleet_trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return session(tmp_path_factory.mktemp("fleet"), budget=FLEET_BUDGET, actors=2)


#: A whole learning curve point, as the report must carry it.
POINT_KEYS = {
    "decisions",
    "episodes",
    "wall_seconds",
    "model_version",
    "mean_final_wave",
    "stdev_final_wave",
    "final_waves",
    "valid_episodes",
    "invalid_episodes",
    "invalid_by_reason",
    "versus_scripted_reference",
    "checkpoint_fingerprint",
    "checkpoint_path",
    "weighted_loss",
    "unweighted_mean_absolute_td_error",
    "gradient_norm",
    "value_fit_correlation",
    "collection_wait_fraction",
    "collection_purchases_per_episode",
    "pre_registered_final",
}

#: One point of the collection curve, as the report must carry it.
WINDOW_KEYS = {
    "index",
    "episodes",
    "decisions",
    "decisions_at_end",
    "mean_final_wave",
    "stdev_final_wave",
    "standard_error",
    "wait_fraction",
    "purchases_per_episode",
    "health",
    "mean_final_wave_by_actor",
    "near_greedy_episodes",
    "near_greedy_mean_final_wave",
}

#: `EpisodeHealth`'s own fields, pooled at whatever scope names it: the whole
#: run, one actor, or one collection window.
HEALTH_KEYS = {
    "episodes",
    "valid_episodes",
    "invalid_episodes",
    "invalid_by_reason",
    "invalid_detail",
    "bridge_event_divergence",
    "stale_or_duplicate",
    "game_time_inflated",
    "advances_cut_short",
    "episodes_not_started_fresh",
    "round_budgeted_ratio",
    "worst_round_budgeted_ratio",
}

#: The health counters expected at zero on the fixtures below: fake runs with
#: no injected refusal, divergence, stall or leftover run.
ZERO_HEALTH_COUNTERS = {
    "invalid_episodes",
    "advances_cut_short",
    "episodes_not_started_fresh",
    "bridge_event_divergence",
    "stale_or_duplicate",
    "game_time_inflated",
}

#: Health counters that sum straightforwardly across actors. The two free-text
#: mappings merge by reason instead, and the two ratios pool rather than sum.
ADDITIVE_HEALTH_COUNTERS = ZERO_HEALTH_COUNTERS | {"episodes", "valid_episodes"}


def test_the_report_carries_a_well_formed_learning_curve(trained: dict[str, Any]) -> None:
    arm = trained["arm"]
    curve = arm["learning_curve"]

    assert curve, "a run with evaluations must produce curve points"
    assert all(set(point) == POINT_KEYS for point in curve)
    decisions = [point["decisions"] for point in curve]
    assert decisions == sorted(decisions), "a curve is placed on the budget in order"
    for point in curve:
        assert point["wall_seconds"] >= 0.0
        assert point["valid_episodes"] + point["invalid_episodes"] == 2
        assert len(point["final_waves"]) == point["valid_episodes"]
        assert point["mean_final_wave"] > 0
        # Two evaluation episodes have a spread; it is reported beside the mean.
        assert point["stdev_final_wave"] is not None
        assert point["versus_scripted_reference"] == pytest.approx(
            point["mean_final_wave"] - SCRIPTED_REFERENCE, abs=1e-3
        )


#: Fields `evaluator.episode_record` promises for every episode, valid or not.
EPISODE_RECORD_KEYS = {
    "episode_index",
    "valid",
    "final_wave",
    # The protocol the episode was played under, on the row itself: a pooled
    # row that cannot say which cadence asked it or which upgrade rows it could
    # buy from is a wave with no decision problem attached (ADR 0009, 0011).
    "decision_cadence",
    "upgrade_availability",
    "decisions",
    "advances",
    "purchases",
    "frames",
    "budgeted_game_ms",
    "round_ms",
    "advance_wall_seconds",
    "elapsed_wall_seconds",
    "invalid_reasons",
    "termination_detail",
    "advances_cut_short",
    "recovered_transients",
    "starting_wave",
    "waves",
}


def test_collected_episodes_are_persisted_with_the_evaluator_shape(
    trained: dict[str, Any],
) -> None:
    """A 4.3-hour run cannot be certified honest without every episode's record."""
    arm = trained["arm"]
    records = arm["collected_episodes"]

    assert len(records) == arm["episodes"] - arm["failed_episodes"]
    assert all(EPISODE_RECORD_KEYS | {"actor_id"} == set(record) for record in records)
    assert [record["episode_index"] for record in records] == list(range(len(records)))
    assert any(record["valid"] for record in records)
    actor_id = arm["resolved_config"]["actor_ids"][0]
    assert all(record["actor_id"] == actor_id for record in records)


def test_evaluation_episodes_are_persisted_with_the_same_shape(trained: dict[str, Any]) -> None:
    arm = trained["arm"]
    for evaluation in arm["evaluations"]:
        records = evaluation["episodes"]
        assert records
        assert all(set(record) == EPISODE_RECORD_KEYS for record in records)


def test_the_health_aggregate_matches_the_collected_episode_records(
    trained: dict[str, Any],
) -> None:
    """The aggregate is pooled from the same records the report persists."""
    arm = trained["arm"]
    records = arm["collected_episodes"]
    health = arm["health"]

    assert health["episodes"] == len(records)
    assert health["valid_episodes"] == sum(1 for record in records if record["valid"])
    assert health["invalid_episodes"] == sum(
        1 for record in records if not record["valid"]
    )
    assert health["advances_cut_short"] == sum(
        record["advances_cut_short"] for record in records
    )
    assert health["episodes_not_started_fresh"] == sum(
        1 for record in records if record["starting_wave"] > 1
    )
    detail: dict[str, int] = {}
    for record in records:
        if record["valid"]:
            continue
        for text in record["termination_detail"]:
            detail[text] = detail.get(text, 0) + 1
    assert health["invalid_detail"] == detail


def test_the_curve_is_readable_against_the_measured_baselines(trained: dict[str, Any]) -> None:
    """The comparison has to be in the artefact, not in another document."""
    for reference in (trained["reference_final_waves"], trained["arm"]["reference_final_waves"]):
        assert reference["scripted"] == 5.57
        assert reference["random"] == 5.35
        assert reference["wait"] == 1.87
        assert reference["source"]


def test_each_point_names_a_checkpoint_that_holds_the_weights_it_scored(
    trained: dict[str, Any],
) -> None:
    arm = trained["arm"]
    curve = arm["learning_curve"]
    for point in curve:
        # The file still holds the weights the point scored: a point writes
        # its own checkpoint rather than sharing the overwritten resume one.
        stored = load(Path(point["checkpoint_path"]))
        assert fingerprint(stored.backbone_state) == point["checkpoint_fingerprint"]
        assert stored.progress.environment_decisions == point["decisions"]
    # The fingerprint moves as the model learns, or the curve could not be
    # attributed to anything.
    versions = {point["model_version"] for point in curve}
    digests = {point["checkpoint_fingerprint"] for point in curve}
    assert len(digests) == len(versions)


def test_everything_the_run_writes_lands_outside_the_repository(
    trained: dict[str, Any],
) -> None:
    repository = Path(train.__file__).resolve().parents[1]
    session_dir = Path(trained["session"])
    summary = json.loads((session_dir / "summary.json").read_text())

    assert repository not in session_dir.parents
    assert summary["arm"]["learning_curve"] == trained["arm"]["learning_curve"]
    arm = trained["arm"]
    run_dir = Path(arm["checkpoint_path"]).parents[1]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    # The checkpoint round-trips: written atomically, checksummed on read.
    checkpoint = load(Path(arm["checkpoint_path"]))
    assert checkpoint.identity.backbone == arm["backbone"]
    assert checkpoint.progress.environment_decisions == arm["decisions"]
    assert checkpoint.progress.environment_game_ms / 1000 == pytest.approx(
        arm["game_seconds"]
    )


def test_the_report_carries_the_collection_curve(trained: dict[str, Any]) -> None:
    """The series the run is read from: collected episodes, in closed windows."""
    arm = trained["arm"]
    curve = arm["collection_curve"]

    assert curve, "a run of several episodes closes at least one window"
    assert all(set(window) == WINDOW_KEYS for window in curve)
    assert [window["index"] for window in curve] == list(range(len(curve)))
    assert arm["collection_window_episodes"] == 2
    placements = [window["decisions_at_end"] for window in curve]
    assert placements == sorted(placements)
    for window in curve:
        assert window["episodes"] == 2
        assert window["mean_final_wave"] > 0
        assert window["standard_error"] is not None
        assert 0.0 <= window["wait_fraction"] <= 1.0
        assert window["purchases_per_episode"] >= 0.0
    # The windows do not overlap, so their episodes sum to what was scored
    # without double counting; a trailing partial window is not a point.
    scored = sum(window["episodes"] for window in curve)
    assert scored <= arm["valid_episodes"] < scored + 2


def test_the_run_ends_on_one_pre_registered_exploration_free_evaluation(
    trained: dict[str, Any],
) -> None:
    arm = trained["arm"]
    final = arm["final_evaluation"]

    assert final is not None and final["pre_registered_final"] is True
    # It scores the final weights, after the budget was spent.
    assert final["decisions"] == arm["decisions"]
    assert final["model_version"] == arm["optimisation_steps"]
    assert final["versus_scripted_reference"] == pytest.approx(
        final["mean_final_wave"] - SCRIPTED_REFERENCE, abs=1e-3
    )
    # Exactly one, and it is the last point on the curve.
    headline = [
        point for point in arm["learning_curve"] if point["pre_registered_final"]
    ]
    assert headline == [final] == [arm["learning_curve"][-1]]


def test_the_record_names_the_pre_registered_point_rather_than_its_caller(
    tmp_path: Path,
) -> None:
    """`record_point` owns `final_point`: the headline is named where it is made.

    It used to be assigned by the closure that ran the evaluation, so a second
    caller recording a final point left the report naming the first one. The
    report records results; it does not run evaluations and does not depend on
    whoever asked for one to finish the record off.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        arm, run_evaluation = train.build_arm(
            train.BACKBONE,
            arguments(tmp_path, **{"--budget-game-seconds": "20"}),
            instances=[train.ActorInstance(serial="fake-0", environment=environment())],
            device=torch.device("cpu"),
            profile_id=PROFILE,
            parent=tmp_path,
            revision="test",
            started=0.0,
            tracker=train.NoExperimentTracker(),
            tags={},
        )

    assert not hasattr(arm, "evaluation"), "the report holds no evaluation to run"
    assert arm.final_point is None

    run_evaluation(False)
    assert arm.final_point is None, "a mid-run point is not the headline"

    run_evaluation(True)
    assert arm.final_point is arm.learning_curve[-1]
    assert arm.final_point is not None and arm.final_point.pre_registered_final


def test_a_checkpoint_carries_the_schedules_the_run_actually_used(
    trained: dict[str, Any],
) -> None:
    """Epsilon and beta are the run's to publish, not the report's to recompute.

    The report used to evaluate the training config's schedules itself, which is
    a learning decision taken in the record of one. It reads what the run
    published instead, so a checkpoint restores the values collection and
    sampling actually ran at.
    """
    arm = trained["arm"]
    stored = load(Path(arm["checkpoint_path"]))
    resolved = arm["resolved_config"]

    assert resolved["epsilon_end"] <= stored.progress.epsilon <= resolved["epsilon_start"]
    # Annealed away from where it started: the run drew it per episode.
    assert stored.progress.epsilon < resolved["epsilon_start"]
    assert resolved["beta_start"] <= stored.progress.importance_beta <= resolved["beta_end"]


def test_the_learner_diagnostics_travel_with_every_point(trained: dict[str, Any]) -> None:
    """Without them a flat curve cannot be told from a broken learner."""
    arm = trained["arm"]
    point = arm["final_evaluation"]

    assert point["weighted_loss"] is not None
    assert point["unweighted_mean_absolute_td_error"] is not None
    # Two names, because they are two quantities: the loss carries the
    # importance-sampling weights and moves with the beta schedule.
    assert point["weighted_loss"] != point["unweighted_mean_absolute_td_error"]
    assert point["gradient_norm"] is not None
    fit = point["value_fit_correlation"]
    assert fit is None or -1.0 <= fit <= 1.0
    assert 0.0 <= point["collection_wait_fraction"] <= 1.0
    assert point["collection_purchases_per_episode"] >= 0.0
    distribution = arm["action_distribution"]
    assert distribution["episodes"] == arm["episodes"] - arm["failed_episodes"]
    assert distribution["decisions"] == arm["decisions"]


def test_the_report_accounts_for_every_actor_and_for_the_fleet(
    fleet_trained: dict[str, Any],
) -> None:
    """An aggregate that cannot name a stalled instance cannot report one."""
    arm = fleet_trained["arm"]
    actors = arm["actors"]

    assert [actor["actor_id"] for actor in actors] == arm["resolved_config"][
        "actor_ids"
    ]
    assert sum(actor["decisions"] for actor in actors) == arm["decisions"]
    assert sum(actor["game_seconds"] for actor in actors) == pytest.approx(
        arm["game_seconds"], abs=0.01
    )
    assert sum(actor["episodes"] for actor in actors) == arm["episodes"]
    assert arm["actors_withdrawn"] == 0
    assert arm["episodes_per_hour"] > 0 and arm["decisions_per_hour"] > 0
    # Game seconds an hour is the comparable throughput: it does not move with
    # how often the environment happened to ask for a decision.
    assert arm["game_seconds_per_hour"] > 0
    assert all(actor["game_seconds_per_hour"] > 0 for actor in actors)
    for actor in actors:
        assert actor["episodes"] > 0 and actor["decisions"] > 0
        assert actor["valid_episodes"] + actor["invalid_episodes"] <= actor["episodes"]
        assert actor["withdrawn"] is None
        assert actor["episodes_per_hour"] > 0
        assert set(health_counters([])) == HEALTH_KEYS
        # The health counters a fleet is watched by, per actor: clean on
        # this fixture's fake run, so every one of them is zero.
        for counter in ZERO_HEALTH_COUNTERS:
            assert actor[counter] == 0
        assert actor["invalid_by_reason"] == {}
        assert actor["invalid_detail"] == {}
    for counter in ADDITIVE_HEALTH_COUNTERS:
        assert arm["health"][counter] == sum(actor[counter] for actor in actors)


def test_the_collection_curve_survives_a_fleet(fleet_trained: dict[str, Any]) -> None:
    """One series in completion order, cut into windows exactly as before."""
    arm = fleet_trained["arm"]
    curve = arm["collection_curve"]

    assert curve and all(set(window) == WINDOW_KEYS for window in curve)
    assert all(set(window["health"]) == HEALTH_KEYS for window in curve)
    assert [window["index"] for window in curve] == list(range(len(curve)))
    placements = [window["decisions_at_end"] for window in curve]
    assert placements == sorted(placements)
    assert all(window["episodes"] == 2 for window in curve)
    scored = sum(window["episodes"] for window in curve)
    assert scored <= arm["valid_episodes"] < scored + 2


def test_a_fleet_attributes_collected_episodes_to_their_actor(
    fleet_trained: dict[str, Any],
) -> None:
    """A record without its actor cannot say which emulator to look at."""
    arm = fleet_trained["arm"]
    records = arm["collected_episodes"]
    actor_ids = set(arm["resolved_config"]["actor_ids"])

    assert records
    assert {record["actor_id"] for record in records} <= actor_ids
    # Every actor's own episode count matches the records attributed to it.
    for actor in arm["actors"]:
        attributed = [
            record for record in records if record["actor_id"] == actor["actor_id"]
        ]
        assert len(attributed) == actor["episodes"] - actor["failed_episodes"]


def test_the_summary_reports_the_budget_in_game_time(trained: dict[str, Any]) -> None:
    """What the run spent, in the unit it was budgeted in, and what it overshot by.

    The budget is accounted at episode granularity, so the run stops after the
    episode that crossed it. The overshoot is reported rather than rounded
    away: it is how far apart two arms equalised on one budget can be.
    """
    arm = trained["arm"]
    budget = int(arm["resolved_config"]["budget_game_seconds"])

    assert arm["game_seconds"] >= budget, "the budget is spent, never stopped short of"
    assert arm["game_seconds"] == pytest.approx(
        sum(float(episode["round_ms"]) for episode in arm["collected_episodes"]) / 1000
    )
    overshoot = arm["budget_overshoot_game_ms"]
    assert overshoot == pytest.approx(arm["game_seconds"] * 1000 - budget * 1000, abs=1)
    # At most the last episode of the one actor this fixture collects with.
    assert 0 <= overshoot <= float(arm["collected_episodes"][-1]["round_ms"])
    # Decisions are still counted beside it - the replay ratio and the
    # exploration anneal are in decisions by design - but they are no longer
    # what the run is spent against.
    assert arm["decisions"] > 0 and arm["decisions_per_hour"] > 0


def test_a_run_that_spent_its_budget_says_it_did_not_stop_early(
    trained: dict[str, Any],
) -> None:
    """Recorded whether or not early stopping was on, so two runs compare."""
    early = trained["arm"]["early_stopping"]

    assert early["patience_periods"] == 0 and early["early_stopped"] is False
    assert early["stopped_at_period"] is None
    assert early["tracker_restored_from_parent"] is False
    # And the thresholds the run was judged under travel as params too.
    assert trained["arm"]["resolved_config"]["early_stop_min_improvement"] == 0.2


def test_the_summary_carries_the_periods_the_run_judged_itself_on(
    tmp_path: Path,
) -> None:
    """One record per numbered-checkpoint crossing, on the run's own axis."""
    report = session(
        tmp_path,
        budget="1200",
        settings={"--checkpoint-every-game-seconds": "400"},
    )
    arm = report["arm"]
    periods = arm["checkpoint_periods"]

    assert periods, "a 1,200-second budget crosses a 400-second period"
    assert [period["index"] for period in periods] == list(range(1, len(periods) + 1))
    assert [period["game_seconds_at_end"] for period in periods] == [
        400 * (index + 1) for index in range(len(periods))
    ]
    # Every period of this fixture is collected by the one near-greedy actor of
    # a uniform schedule, so each carries a mean and it is a real wave.
    assert all(period["near_greedy_episodes"] > 0 for period in periods)
    assert all(period["mean_final_wave"] >= 1 for period in periods)
    assert arm["early_stopping"]["periods_closed"] == len(periods)
    # The bar is the mean of some period that cleared it, which is one of these
    # and never above the highest of them - not necessarily the highest itself.
    best = arm["early_stopping"]["best_period_near_greedy_mean_final_wave"]
    assert best in [period["mean_final_wave"] for period in periods]
    assert best <= max(period["mean_final_wave"] for period in periods)
    assert arm["early_stopping"]["closing_period_near_greedy_mean_final_wave"] == (
        periods[-1]["mean_final_wave"]
    )
