"""The learner thread and the debt that holds the replay ratio (ADR 0017).

The first half drives `LearnerThread` directly with a step that is only a
counter, so the debt's arithmetic is exact and the waits are the test's own.
The second half runs a fleet on fake ports and reads what a run does with it:
who takes the steps, what a checkpoint sees, and whether an actor is still
kept waiting by a slow step. No emulator, no adb, no bridge, and nothing any of
these fakes produce reaches a replay buffer outside the test that made it.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import torch
from test_fleet_training import (
    DECISION_SECONDS,
    LEARN_SECONDS,
    Overlap,
    environment,
    fleet,
)

from tower_rl.environment.decision_time import BLOCKED, DecisionTimeProfile
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch
from tower_rl.learning.learner import Learner, LearnerThread
from tower_rl.learning.training import TrainingProgressReport

METRICS = LearnMetrics(
    weighted_loss=0.0,
    unweighted_mean_absolute_td_error=0.0,
    gradient_norm=0.0,
    td_errors=(),
)


def cpu_learner() -> Learner:
    """A learner whose network is never touched: the step below is the test's."""
    return Learner(backbone=cast(Any, SimpleNamespace(device=torch.device("cpu"))))


class CountedStep:
    """A gradient step that only counts, and can be held shut or made slow."""

    def __init__(self, seconds: float = 0.0) -> None:
        self.seconds = seconds
        self.taken = 0
        self.inside = threading.Event()
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self) -> LearnMetrics:
        self.inside.set()
        self.gate.wait()
        time.sleep(self.seconds)
        self.taken += 1
        self.inside.clear()
        return METRICS


def never() -> bool:
    return False


def test_the_bound_pauses_an_actor_until_the_learner_drains_the_debt() -> None:
    step = CountedStep()
    step.gate.clear()
    learner = LearnerThread(cpu_learner(), step, steps_per_decision=1.0, bound_decisions=4)
    learner.start()
    learner.credit(7)  # three over the bound, and the learner is stuck in its first step
    assert step.inside.wait(5)
    returned = threading.Event()
    profile = DecisionTimeProfile()

    def actor() -> None:
        learner.make_room(profile, never)
        returned.set()

    threading.Thread(target=actor, daemon=True).start()

    assert not returned.wait(0.3), "the actor went on with the learner seven steps behind"
    step.gate.set()
    assert returned.wait(5), "the actor stayed paused after the debt drained"
    assert learner.load().owed_steps <= learner.bound_steps
    learner.finish(drain=True)

    load = learner.load()
    assert step.taken == load.gradient_steps == 7, "the drain paid everything owed"
    assert load.owed_steps == 0
    assert load.paused_actor_seconds >= 0.3
    assert profile.snapshot().buckets[BLOCKED].wall_seconds >= 0.3


def test_the_ratio_is_held_within_the_bound_over_every_window() -> None:
    """Three actors credit half a step per decision against a learner slower than them.

    Each actor credits before it checks the bound, so at any moment the debt is
    at most the bound plus one decision's worth per actor; and the steps taken
    over any window are the ratio of the decisions credited in it, short by at
    most that. The drain at the end pays the rest.
    """
    ratio, bound, actors, decisions = 0.5, 16, 3, 150
    step = CountedStep(seconds=0.002)
    learner = LearnerThread(
        cpu_learner(), step, steps_per_decision=ratio, bound_decisions=bound
    )
    ceiling = bound * ratio + actors * ratio
    readings: list[tuple[int, float]] = []
    credited = [0]
    lock = threading.Lock()

    def actor() -> None:
        profile = DecisionTimeProfile()
        for _ in range(decisions):
            learner.credit(1)
            with lock:
                credited[0] += 1
            learner.make_room(profile, never)
            with lock:
                readings.append((credited[0], learner.load().owed_steps))

    learner.start()
    threads = [threading.Thread(target=actor) for _ in range(actors)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    learner.finish(drain=True)

    assert max(owed for _, owed in readings) <= ceiling
    assert learner.load().paused_actor_seconds > 0, "the learner was slow enough to bind"
    assert step.taken == int(ratio * actors * decisions)


def test_holding_the_learner_still_waits_out_the_step_in_flight_and_starts_none() -> None:
    step = CountedStep(seconds=0.05)
    learner = LearnerThread(cpu_learner(), step, steps_per_decision=1.0, bound_decisions=100)
    learner.start()
    learner.credit(20)
    assert step.inside.wait(5)

    with learner.held():
        assert not step.inside.is_set(), "the step in flight finished before the hold began"
        held_at = step.taken
        time.sleep(0.2)
        assert step.taken == held_at, "a step began while the learner was held still"
    learner.finish(drain=True)

    assert step.taken == 20


def test_a_step_that_raises_ends_the_thread_and_is_kept_for_the_run() -> None:
    def broken() -> LearnMetrics:
        raise RuntimeError("the step failed")

    learner = LearnerThread(cpu_learner(), broken, steps_per_decision=1.0, bound_decisions=1)
    learner.start()
    learner.credit(5)
    # An actor paused on the bound is not left waiting on a learner that is gone.
    learner.make_room(DecisionTimeProfile(), never)
    learner.finish(drain=True)

    assert isinstance(learner.failure, RuntimeError)
    assert learner.load().gradient_steps == 0


def test_an_abort_ends_after_the_step_in_flight_without_paying_the_debt() -> None:
    step = CountedStep(seconds=0.05)
    learner = LearnerThread(cpu_learner(), step, steps_per_decision=1.0, bound_decisions=100)
    learner.start()
    learner.credit(50)
    assert step.inside.wait(5)
    # A drain asked for first and an abort after it, as a second SIGINT does.
    finisher = threading.Thread(target=learner.finish, kwargs={"drain": True})
    finisher.start()
    learner.finish(drain=False)
    finisher.join()

    assert 1 <= step.taken < 50


def test_no_actor_thread_ever_takes_a_gradient_step() -> None:
    training = fleet([environment() for _ in range(3)], budget_decisions=250)
    threads: set[str] = set()
    learn = training.learner.learn

    def recorded(batch: SequenceBatch) -> LearnMetrics:
        threads.add(threading.current_thread().name)
        return learn(batch)

    training.learner.learn = recorded  # type: ignore[method-assign]

    report = training.run()

    assert report.optimisation_steps > 0
    assert threads == {"learner"}


def test_a_run_ends_every_block_owing_less_than_one_step() -> None:
    """The block's end drains the debt, so the ratio holds over the whole run.

    Every episode counted with the buffer warm earned its decisions, as it did
    when the finishing actor took them itself: the steps are that ratio of
    those decisions, to the step.
    """
    training = fleet([environment() for _ in range(2)], budget_decisions=300)
    warm_decisions: list[int] = []

    def counted(report: TrainingProgressReport) -> None:
        if training._warmed:
            warm_decisions.append(report.collected[-1].summary.decisions)

    training.on_episode = counted

    report = training.run()

    ratio = training.config.gradient_steps_per_decision
    assert report.optimisation_steps == int(ratio * sum(warm_decisions))
    assert training.learner_load().owed_steps < 1.0


def test_a_checkpoint_is_written_with_the_learner_held_still() -> None:
    """What a resume point saves is one completed step, counted as such (ADR 0014)."""
    training = fleet(
        [environment() for _ in range(2)],
        budget_decisions=300,
        checkpoint_every_episodes=1,
    )
    stepping = threading.Event()
    learn = training.learner.learn

    def watched(batch: SequenceBatch) -> LearnMetrics:
        stepping.set()
        try:
            time.sleep(DECISION_SECONDS)
            return learn(batch)
        finally:
            stepping.clear()

    training.learner.learn = watched  # type: ignore[method-assign]
    seen: list[tuple[int, int]] = []

    def checkpoint(report: TrainingProgressReport) -> None:
        version = training.backbone.model_version
        assert not stepping.is_set(), "a step was in flight during the save"
        time.sleep(DECISION_SECONDS * 3)
        assert not stepping.is_set(), "a step began during the save"
        assert training.backbone.model_version == version
        seen.append((report.optimisation_steps, version))

    training.checkpoint = checkpoint

    training.run()

    assert any(steps > 0 for steps, _ in seen)
    assert all(steps == version for steps, version in seen), (
        "the counted steps and the weights' own count disagreed at a save"
    )


def test_a_slow_learner_pauses_no_actor_and_no_actor_takes_a_step() -> None:
    """A gradient step ten times a decision costs a fleet's actors nothing but refreshes.

    Before ADR 0017 the finishing actor took every step its episode earned, so
    the fleet's actors spent the learner's whole stepping time waiting on it.
    Now every step runs on the learner thread, and the bound pauses nobody: the
    whole run credits 15 steps against a bound of 25.6, so the debt cannot pass
    it however slow the learner is. Counted, not timed.
    """
    training = fleet(
        [environment(Overlap()) for _ in range(3)],
        budget_decisions=300,
        gradient_steps_per_decision=0.05,
        parameter_sync_decisions=10,
    )
    assert training.learner_thread.bound_steps > 300 * 0.05
    threads: list[str] = []
    learn = training.backbone.learn

    def slow(batch: SequenceBatch) -> LearnMetrics:
        # Inside `Learner.lock`, so a refresh really has a step to wait on.
        threads.append(threading.current_thread().name)
        time.sleep(LEARN_SECONDS)
        return learn(batch)

    training.backbone.learn = slow  # type: ignore[method-assign]

    report = training.run()

    assert len(threads) == report.optimisation_steps >= 5
    assert set(threads) == {"learner"}, "an actor thread took a gradient step"
    assert training.learner_load().paused_actor_seconds == 0.0, "an actor paused on the bound"


def test_a_refresh_never_waits_for_the_step_in_flight_and_loads_the_last_finished_one() -> None:
    """An actor reads the last published snapshot; a step in flight holds nothing it needs.

    Ordered by events, not by time: the refresh runs, and returns, while the
    step is held shut inside, and it loads what the previous step published.
    """
    inside, release = threading.Event(), threading.Event()

    class Network:
        device = torch.device("cpu")
        version = 0

        def learn(self, batch: object) -> LearnMetrics:
            inside.set()
            release.wait()
            self.version += 1
            return METRICS

        def state_dict(self) -> dict[str, object]:
            return {"version": torch.tensor(self.version)}

    loaded: list[int] = []
    acting = SimpleNamespace(load_state_dict=lambda state: loaded.append(int(state["version"])))
    learner = Learner(backbone=cast(Any, Network()))
    learner.publish()
    thread = LearnerThread(
        learner,
        lambda: learner.learn(cast(Any, None)),
        steps_per_decision=1.0,
        bound_decisions=100,
    )
    thread.start()
    thread.credit(1)  # one step owed, so no second can begin behind the first
    assert inside.wait(5)

    refresher = threading.Thread(
        target=lambda: learner.publish_to(cast(Any, acting)), daemon=True
    )
    refresher.start()
    refresher.join(5)
    stuck = refresher.is_alive()
    release.set()
    thread.finish(drain=True)

    assert not stuck, "a refresh waited for the step in flight"
    assert loaded == [0], "the refresh did not load the last finished step"
    assert learner.publish_to(cast(Any, acting)) == learner.published == 2
    assert loaded == [0, 1], "the finished step was not published"
