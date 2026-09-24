"""A fleet of actors collecting into one learner, without a device.

No emulator, no adb, no bridge: every actor drives its own `FakeRunPort`, so
what is under test is the plumbing a fleet adds - concurrency, one shared
buffer, a budget counted across actors, per-actor failure isolation, and the
guard that keeps a forward pass out of the middle of an optimisation step.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import chain, repeat
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from fakes.backbone_equality import parameters_are_equal
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.episode import EpisodeSummary, TerminationOutcome
from tower_rl.environment.features import encode_state
from tower_rl.environment.run_environment import (
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.actor import Actor, ActorConfig, EpisodeResult
from tower_rl.learning.backbone import (
    LearnMetrics,
    SequenceBatch,
    acting_copy,
)
from tower_rl.learning.checkpoint import (
    CheckpointIdentity,
    TrainingProgress,
    resume_state,
    write_checkpoint,
)
from tower_rl.learning.exploration import ExplorationSchedule, ape_x_floors
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.replay import PrioritizedSequenceReplay
from tower_rl.learning.stacked_dqn import (
    StackedDqnBackbone,
    StackedDqnConfig,
)
from tower_rl.learning.training import (
    KillBar,
    NearGreedyPlateau,
    TrainingConfig,
    TrainingProgressReport,
    TrainingRun,
    collection_windows,
)
from tower_rl.simulation.instrumented_bridge import BridgeTimeoutError
from tower_rl.simulation.instrumented_run_adapter import (
    InstrumentedRunAdapter,
)

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

#: Any schedule at all: nothing in this file is about exploration, and the rates
#: a real run uses are resolved from `train.py`'s parser rather than defaulted.
SCHEDULE = ExplorationSchedule(
    epsilon_start=1.0, epsilon_end=0.05, anneal_decisions=10_000
)


#: Long enough that two actors overlap in it and short enough that the suite
#: stays fast. A real decision costs hundreds of milliseconds on device.
DECISION_SECONDS = 0.002

#: An optimisation step wide enough to contain a whole decision, which is what
#: makes "the actor did not wait for the learner" an assertion rather than a
#: hope. On the GPU a real step is about 20 ms, so this is the real ratio.
LEARN_SECONDS = 0.02


@dataclass
class Overlap:
    """How many actors were inside a decision at once, at their busiest."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    peak: int = 0
    #: When each decision started and finished, so it can be placed against an
    #: optimisation step.
    spans: list[tuple[float, float]] = field(default_factory=list)

    @contextmanager
    def deciding(self) -> Any:
        started = time.monotonic()
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1
                self.spans.append((started, time.monotonic()))


class PacedEnvironment(InstrumentedRunEnvironment):
    """A fake instance whose decisions take measurable time, as a real one does.

    Without this every fake episode finishes in microseconds and concurrency is
    unobservable: the actors would be genuinely parallel and still never be seen
    inside a decision together.
    """

    overlap: Overlap

    def step(self, action: Any) -> Any:
        with self.overlap.deciding():
            time.sleep(DECISION_SECONDS)
            return super().step(action)


def environment(overlap: Overlap | None = None, **port: Any) -> InstrumentedRunEnvironment:
    settings: dict[str, Any] = {"damage_per_second": 2.0}
    settings.update(port)
    if overlap is None:
        return InstrumentedRunEnvironment(
            port=FakeRunPort(**settings),
            builder=RunStateBuilder(profile_id="fake-profile-v1"),
            cadence=CadenceConfig(max_quiet_game_ms=1000),
        )
    paced = PacedEnvironment(
        port=FakeRunPort(**settings),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    paced.overlap = overlap
    return paced


def fleet(
    environments: list[InstrumentedRunEnvironment],
    *,
    backbone: Any = None,
    report: TrainingProgressReport | None = None,
    **overrides: Any,
) -> TrainingRun:
    """One arm collecting on these instances, one actor each.

    `report` is what a resumed segment is built with: the counters a parent
    left, handed to the run at construction because the run reads its
    checkpoint cadence off them.
    """
    learner = backbone or StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    replay = PrioritizedSequenceReplay(capacity=256, seed=0)
    actors = [
        Actor(
            environment=item,
            policy=learner,
            config=ActorConfig(
                actor_id=f"fake-{index}:stacked-dqn",
                sequence_length=6,
                burn_in=1,
                stride=3,
            ),
            replay=replay,
        )
        for index, item in enumerate(environments)
    ]
    settings: dict[str, Any] = {
        "budget_decisions": 375,
        "warmup_sequences": 2,
        "batch_size": 2,
        "gradient_steps_per_decision": 0.2,
        "exploration": SCHEDULE,
    }
    settings.update(overrides)
    return TrainingRun(
        actors=actors,
        replay=replay,
        backbone=learner,
        config=TrainingConfig(**settings),
        report=report or TrainingProgressReport(),
    )


class _SilentBridgeClient:
    """A bridge that has stopped answering, exactly as a dead emulator's does.

    The adapter in front of it is the real one, so what is under test is the
    path a bridge failure actually takes out of the client and into the fleet.
    """

    def read_state(self) -> Any:
        raise BridgeTimeoutError("bridge liveness expired: silent for 60s")

    def send_command(self, message: Any) -> Any:
        raise BridgeTimeoutError("bridge liveness expired: silent for 60s")


def dead_bridge_environment() -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=InstrumentedRunAdapter(client=_SilentBridgeClient()),  # type: ignore[arg-type]
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )


def test_a_bridge_that_stops_answering_withdraws_its_actor_and_not_the_run() -> None:
    """The likeliest failure of all, and the one that used to end the whole run.

    A bridge error was not a `RunPortError`, so it bypassed the withdrawal path
    entirely and re-raised out of `advance` - a 100,000-decision fleet run died
    with three healthy actors still collecting.
    """
    withdrawn: list[tuple[str, str]] = []
    training = fleet(
        [environment(), dead_bridge_environment(), environment()],
        budget_decisions=250,
        max_consecutive_episode_failures=3,
    )
    training.on_withdrawal = lambda progress: withdrawn.append(
        (progress.actor_id, progress.withdrawn or "")
    )

    report = training.run()

    dead = report.actors["fake-1:stacked-dqn"]
    assert dead.withdrawn is not None and "liveness expired" in dead.withdrawn
    assert dead.failed_episodes == 3 and dead.decisions == 0
    # Named as it happened, with the instance it names and why it left.
    assert withdrawn == [("fake-1:stacked-dqn", dead.withdrawn)]
    alive = [progress for progress in report.actors.values() if progress.withdrawn is None]
    assert len(alive) == 2 and all(progress.valid_episodes > 0 for progress in alive)
    assert report.decisions >= 250


def test_a_fleet_whose_bridges_have_all_died_still_ends_the_run() -> None:
    """Nothing left collecting is a dead environment however it died."""
    training = fleet(
        [dead_bridge_environment() for _ in range(2)],
        budget_decisions=250,
        max_consecutive_episode_failures=2,
    )

    with pytest.raises(RunPortError, match="liveness expired"):
        training.run()

    assert all(
        progress.withdrawn is not None for progress in training.report.actors.values()
    )


def test_the_actors_of_a_fleet_collect_at_the_same_time() -> None:
    """The whole point: N instances stepping at once, not N arms taking turns."""
    overlap = Overlap()
    training = fleet([environment(overlap) for _ in range(3)], budget_decisions=250)

    training.run()

    assert overlap.peak > 1, "the actors never overlapped, so nothing was parallel"
    assert all(progress.episodes > 0 for progress in training.report.actors.values())


def test_every_actor_writes_into_the_one_replay_buffer() -> None:
    """One buffer and one learner: experience is pooled, not split per instance."""
    training = fleet([environment() for _ in range(3)], budget_decisions=250)

    report = training.run()

    stored = {sequence.metadata.actor_id for sequence in training.replay._items}
    assert stored == {progress.actor_id for progress in report.actors.values()}
    assert report.sequences_accepted == training.replay.stats.added


def test_the_budget_is_counted_across_the_fleet() -> None:
    """Three actors spend one budget, and the learner's steps track all of it."""
    training = fleet([environment() for _ in range(3)], budget_decisions=250)

    report = training.run()

    per_actor = [progress.decisions for progress in report.actors.values()]
    game_time = [progress.game_ms for progress in report.actors.values()]
    assert report.decisions >= 250
    assert sum(game_time) == pytest.approx(report.game_ms)
    assert sum(per_actor) == report.decisions
    assert all(spent < report.game_ms for spent in game_time), (
        "no single actor spent the whole budget; the fleet did"
    )
    # Gradient steps follow the fleet's decisions rather than one actor's. The
    # first episode of each actor fills the buffer before any step is owed, so
    # the floor is taken over what was collected after the buffer went warm.
    warming = sum(
        episode.summary.decisions
        for episode in report.collected[: len(training.actors)]
    )
    ratio = training.config.gradient_steps_per_decision
    assert report.optimisation_steps >= ratio * (report.decisions - warming) - 1
    assert report.optimisation_steps > ratio * max(per_actor)


def test_the_collection_series_is_the_fleet_in_completion_order() -> None:
    """Windows are cut over episodes as they ended, whichever actor ended them."""
    overlap = Overlap()
    training = fleet([environment(overlap) for _ in range(3)], budget_decisions=250)

    report = training.run()
    windows = collection_windows(report.collected, size=2)

    assert windows, "a run of several episodes closes at least one window"
    assert [window.index for window in windows] == list(range(len(windows)))
    placements = [window.decisions_at_end for window in windows]
    assert placements == sorted(placements), "a window is placed where it closed"
    assert sum(window.episodes for window in windows) <= report.valid_episodes
    # The series is one fleet-wide sequence, so consecutive episodes come from
    # different actors; a series cut per actor could not interleave like this.
    collectors = [episode.actor_id for episode in report.collected]
    assert any(first != second for first, second in zip(collectors, collectors[1:], strict=False))
    assert sum(len(report.episodes_of(actor)) for actor in set(collectors)) == len(
        report.collected
    )


def test_one_dead_instance_costs_an_actor_and_not_the_run() -> None:
    """A fleet-level limit: one emulator dying is not a dead environment."""
    training = fleet(
        [environment(), environment(refuse_to_start=True), environment()],
        budget_decisions=250,
        max_consecutive_episode_failures=3,
    )

    report = training.run()

    dead = report.actors["fake-1:stacked-dqn"]
    assert dead.withdrawn is not None and dead.failed_episodes == 3
    assert dead.decisions == 0 and dead.valid_episodes == 0
    assert report.failed_episodes == 3 and len(report.episode_failures) == 3
    # The survivors spent the whole budget between them.
    alive = [progress for progress in report.actors.values() if progress.withdrawn is None]
    assert len(alive) == 2 and all(progress.valid_episodes > 0 for progress in alive)
    assert report.decisions >= 250


def test_a_withdrawn_actor_is_not_asked_again_in_a_later_block() -> None:
    """Its instance failed every episode the limit allows; retrying would spin."""
    training = fleet(
        [environment(), environment(refuse_to_start=True)],
        budget_decisions=500,
        max_consecutive_episode_failures=2,
    )

    training.advance(100)
    dead = training.report.actors["fake-1:stacked-dqn"]
    failures_after_first_block = dead.failed_episodes

    training.advance(100)

    assert dead.withdrawn is not None
    assert dead.failed_episodes == failures_after_first_block == 2
    assert training.report.decisions >= 200


def test_a_fleet_with_nothing_left_collecting_stops_the_run() -> None:
    """Every instance broken is a dead environment, which is what ends a run."""
    training = fleet(
        [environment(refuse_to_start=True) for _ in range(2)],
        budget_decisions=250,
        max_consecutive_episode_failures=2,
    )

    with pytest.raises(RunPortError):
        training.run()

    assert training.report.failed_episodes == 4
    assert all(
        progress.withdrawn is not None for progress in training.report.actors.values()
    )
    # And a run with no actor left refuses to be advanced again.
    with pytest.raises(RunPortError, match="nothing is left collecting"):
        training.advance(50)


def test_two_actors_may_not_share_one_identity() -> None:
    """Per-actor reporting is keyed by identity; a shared name hides a dead one."""
    training = fleet([environment(), environment()])
    for actor in training.actors:
        actor.config = ActorConfig(actor_id="same", sequence_length=6, burn_in=1, stride=3)

    with pytest.raises(ValueError, match="id of its own"):
        TrainingRun(
            actors=training.actors,
            replay=training.replay,
            backbone=training.backbone,
            config=training.config,
        )


def test_a_run_needs_at_least_one_actor() -> None:
    with pytest.raises(ValueError, match="at least one actor"):
        fleet([])


def test_a_fleet_refuses_to_evaluate_on_an_instance_that_is_still_collecting() -> None:
    """The single-environment invariant, asserted where it would be broken.

    An evaluation borrows an actor's environment. With a fleet, the other actors
    are still collecting, so the borrowed instance would be driven from two
    threads at once. The CLI refuses the flag combination, but nothing stops a
    caller composing a run in Python, so the run refuses it itself - before a
    thread is started, not halfway through a block.
    """
    training = fleet([environment(), environment()], evaluate_every_episodes=1)
    training.evaluate = lambda: cast(Any, None)

    with pytest.raises(ValueError, match="cannot run while a fleet"):
        training.advance(50)

    assert training.report.episodes == 0, "nothing was collected before the refusal"


def test_a_fleet_of_one_may_evaluate_between_its_own_episodes() -> None:
    """The same hook is legal for one actor: the hook runs on that actor's thread."""
    training = fleet([environment()], evaluate_every_episodes=1, budget_decisions=25)
    evaluations = 0

    def evaluate_now() -> Any:
        nonlocal evaluations
        evaluations += 1
        raise ValueError("no arm to score; the point is that it was called")

    training.evaluate = evaluate_now
    training.advance(20)

    assert evaluations > 0


@dataclass
class WatchedBackbone:
    """A backbone that records who touched it and when, so the sharing is visible.

    One of these stands in for the learner's own network. The copies the actors
    act from are deepcopies of it and keep records of their own, so what this
    instance records is exactly what reached the learner itself.
    """

    inner: StackedDqnBackbone
    #: Breaches of the one rule the learner's lock is still there for.
    torn: list[str] = field(default_factory=list)
    #: Model versions observed by `act`, in order.
    versions: list[int] = field(default_factory=list)
    #: Publications received, and where in this copy's own series of decisions
    #: each of them landed - which is how a publication inside an episode would
    #: be caught.
    publications: int = 0
    publish_positions: list[int] = field(default_factory=list)
    updating: bool = False
    publishing: bool = False

    @property
    def acts(self) -> int:
        return len(self.versions)

    @property
    def model_version(self) -> int:
        return self.inner.model_version

    @property
    def device(self) -> torch.device:
        return self.inner.device

    def act(self, features: Any, state: Any, *, epsilon: float) -> tuple[int, Any]:
        if self.updating:
            self.torn.append("an actor read the network mid-update")
        self.versions.append(self.inner.model_version)
        return self.inner.act(features, state, epsilon=epsilon)

    def initial_state(self) -> Any:
        return self.inner.initial_state()

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        if self.publishing:
            self.torn.append("the learner stepped while its parameters were being read")
        self.updating = True
        try:
            # Wide enough that an unguarded publication would land inside it.
            time.sleep(DECISION_SECONDS)
            return self.inner.learn(batch)
        finally:
            self.updating = False

    def state_dict(self) -> dict[str, Any]:
        if self.updating:
            self.torn.append("a publication read the network mid-update")
        self.publishing = True
        try:
            time.sleep(DECISION_SECONDS)
            return self.inner.state_dict()
        finally:
            self.publishing = False

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.publications += 1
        self.publish_positions.append(self.acts)
        self.inner.load_state_dict(state)


def watched_fleet(instances: int, **overrides: Any) -> tuple[TrainingRun, WatchedBackbone]:
    """A fleet whose learner and acting copies both keep a record of themselves."""
    watched = WatchedBackbone(
        StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    )
    training = fleet([environment() for _ in range(instances)], backbone=watched, **overrides)
    return training, watched


def copies(training: TrainingRun) -> list[WatchedBackbone]:
    """The acting copies, which are deepcopies of the watched learner."""
    return [cast(WatchedBackbone, copy) for copy in training.acting.values()]


def test_a_ladder_puts_every_actor_of_the_fleet_on_a_rate_of_its_own() -> None:
    """Ape-X's arrangement: one fleet searches and reports at the same time.

    The anneal is spent here, so every actor is at its own rung - which is what
    a run collects under for all but the first few thousand decisions.
    """
    schedule = ExplorationSchedule.for_option(
        "ladder",
        actors=3,
        epsilon_start=0.9,
        epsilon_end=SCHEDULE.epsilon_end,
        anneal_decisions=1,
    )
    training = fleet(
        [environment() for _ in range(3)], exploration=schedule, budget_decisions=250
    )

    training.run()

    rungs = ape_x_floors(3)
    rates = [actor.config.epsilon for actor in training.actors]
    # Exploration is drawn once per episode, so an actor whose only episode
    # began at decision zero is still at the start of the anneal; every other
    # actor is at its own rung and at nobody else's.
    assert rates == [
        pytest.approx(rung) if rate != 0.9 else 0.9 for rate, rung in zip(rates, rungs, strict=True)
    ]
    assert any(rate != 0.9 for rate in rates), "the anneal is one decision long"


def test_a_ladder_built_for_another_fleet_is_refused() -> None:
    """Read against the wrong fleet, actor 3 of 7 would act at actor 3 of 4's rate."""
    with pytest.raises(ValueError, match="exploration ladder"):
        fleet(
            [environment() for _ in range(2)],
            exploration=ExplorationSchedule.for_option(
                "ladder",
                actors=7,
                epsilon_start=SCHEDULE.epsilon_start,
                epsilon_end=SCHEDULE.epsilon_end,
                anneal_decisions=SCHEDULE.anneal_decisions,
            ),
        )


def test_every_actor_acts_from_a_copy_of_its_own_and_not_from_the_learner() -> None:
    """The point of the whole arrangement: no forward pass touches shared state."""
    training, watched = watched_fleet(3, budget_decisions=250)

    report = training.run()

    assert report.optimisation_steps > 0
    # Nothing ever asked the learner's own network for an action.
    assert watched.acts == 0
    acting = copies(training)
    assert len({id(copy) for copy in acting}) == 3
    assert all(copy is not training.backbone for copy in acting)
    assert all(copy.acts > 0 for copy in acting)
    # Every decision the fleet spent was taken on some actor's own copy.
    assert sum(copy.acts for copy in acting) == report.decisions


def test_an_actor_does_not_wait_for_the_learner_to_finish_a_step() -> None:
    """A decision now runs inside an optimisation step; it used to queue behind it.

    The learner's step is deliberately slow here, and a decision is timed. Under
    the single shared network a decision could never be contained by an update -
    the lock held both - and this is the assertion that says so.
    """
    overlap = Overlap()
    watched = WatchedBackbone(
        StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    )
    updates: list[tuple[float, float]] = []
    inner_learn = watched.learn

    def timed(batch: SequenceBatch) -> LearnMetrics:
        started = time.monotonic()
        try:
            time.sleep(LEARN_SECONDS)
            return inner_learn(batch)
        finally:
            updates.append((started, time.monotonic()))

    watched.learn = timed  # type: ignore[method-assign]
    training = fleet(
        [environment(overlap) for _ in range(3)], backbone=watched, budget_decisions=250
    )

    report = training.run()

    assert report.optimisation_steps > 0 and updates
    assert any(
        start <= decided and finished <= end
        for start, end in updates
        for decided, finished in overlap.spans
    ), "no decision was taken while the learner was inside a step"


def test_a_publication_gives_an_actor_the_learner_s_current_parameters() -> None:
    """Synchronisation is the whole contract: afterwards the copy is the learner."""
    training, _ = watched_fleet(1, budget_decisions=250)
    acting = copies(training)[0]

    report = training.run()

    assert report.optimisation_steps > 0
    # The run ends with learning after the last episode, so the copy is behind.
    assert not parameters_are_equal(acting.inner.online, training.backbone.inner.online)

    training.learner.publish_to(acting)

    assert parameters_are_equal(acting.inner.online, training.backbone.inner.online)
    assert parameters_are_equal(acting.inner.target, training.backbone.inner.target)
    assert acting.model_version == training.backbone.model_version


def test_no_actor_is_ever_given_half_of_an_optimisation_step() -> None:
    """Stale parameters are fine; half-updated ones are no policy at all."""
    training, watched = watched_fleet(3, budget_decisions=250)

    report = training.run()

    assert report.optimisation_steps > 0, "nothing would be guarded without updates"
    assert watched.publications == 0, "the learner is published from, never into"
    assert sum(copy.publications for copy in copies(training)) > 0
    assert watched.torn == []


def test_a_fleet_of_one_starts_every_episode_from_the_learner_s_parameters() -> None:
    """`--actors 1` must be the run it always was, or the references move.

    A single actor used to act from the learner's own network, which only ever
    moved between its episodes. At the default cadence of one episode its copy
    is refreshed at exactly those moments, so what it acts from at every episode
    is what it would have acted from before.
    """
    watched = WatchedBackbone(
        StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    )
    matched: list[bool] = []
    instance = environment()
    training = fleet([instance], backbone=watched, budget_decisions=250)
    acting = copies(training)[0]
    opened = instance.reset

    def reset_and_witness() -> Any:
        matched.append(
            parameters_are_equal(acting.inner.online, watched.inner.online)
            and acting.model_version == watched.model_version
        )
        return opened()

    instance.reset = reset_and_witness  # type: ignore[method-assign]

    report = training.run()

    assert report.optimisation_steps > 0 and len(matched) > 1
    assert all(matched), "an episode began on parameters the learner had moved past"


def test_a_fleet_of_one_learns_only_between_its_own_episodes() -> None:
    """The single-actor path is the loop it always was: collect, then learn.

    One actor takes its own gradient steps between its own episodes, so the
    parameters it acts from never move inside an episode. That is not true of a
    fleet, where another actor's learning lands mid-episode, and it is what
    keeps a run configured with `--actors 1` reproducible.
    """
    training, watched = watched_fleet(1, budget_decisions=250)
    acting = copies(training)[0]
    boundaries: list[int] = []
    training.on_episode = lambda _: boundaries.append(acting.acts)

    report = training.run()

    assert report.optimisation_steps > 0 and len(boundaries) > 1
    start = 0
    for end in boundaries:
        within = set(acting.versions[start:end])
        assert len(within) <= 1, "the parameters moved inside a single episode"
        start = end
    assert len(set(acting.versions)) > 1, "and they did move between episodes"
    # And no refresh ever landed inside an episode, so the history window the
    # actor carries through one was produced by the parameters it still holds.
    assert acting.publish_positions
    assert set(acting.publish_positions) <= {0, *boundaries}


def test_the_synchronisation_cadence_is_counted_in_an_actor_s_own_episodes() -> None:
    """A bounded lag, set explicitly: one refresh every three episodes, not more."""
    training, _ = watched_fleet(1, budget_decisions=400, parameter_sync_episodes=3)
    acting = copies(training)[0]

    report = training.run()

    assert report.episodes >= 4
    assert acting.publications == 1 + (report.episodes - 1) // 3
    assert acting.publications < report.episodes


def test_a_publication_leaves_the_state_an_actor_carries_through_an_episode_alone() -> None:
    """The carried history window is the actor's, not the network's, and outlives a refresh."""
    training, _ = watched_fleet(1, budget_decisions=250)
    acting = copies(training)[0]
    instance = training.actors[0].environment

    training.run()

    features = encode_state(instance.reset())
    _, carried = acting.act(features, acting.initial_state(), epsilon=0.0)
    before = carried.clone()

    training.learner.publish_to(acting)

    assert torch.equal(carried, before), (
        "a refresh of the parameters disturbed the state the episode was carrying"
    )
    action, resumed = acting.act(features, carried, epsilon=0.0)
    assert features.mask[action]
    assert resumed.shape == before.shape


def test_the_actors_of_a_fleet_do_not_explore_in_lockstep() -> None:
    """Copies of one backbone would otherwise share the stream they were copied from."""
    learner = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    first = acting_copy(learner)
    second = acting_copy(learner, exploration_seed="fake-1:stacked-dqn")
    third = acting_copy(learner, exploration_seed="fake-2:stacked-dqn")

    draws = [
        [copy._random.random() for _ in range(8)]  # type: ignore[attr-defined]
        for copy in (first, second, third)
    ]

    assert draws[0] != draws[1] and draws[1] != draws[2] and draws[0] != draws[2]
    # The first actor carries on the learner's own stream, so a fleet of one
    # explores exactly as a single actor acting from the learner did.
    assert draws[0] == [learner._random.random() for _ in range(8)]


def test_an_acting_copy_is_not_trained_and_shares_nothing_with_the_learner() -> None:
    learner = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )

    acting = acting_copy(learner)

    assert acting is not learner
    assert acting.device == learner.device
    assert parameters_are_equal(acting.online, learner.online)
    assert not any(parameter.requires_grad for parameter in acting.online.parameters())
    assert not acting.online.training
    with torch.no_grad():
        learner.online.state_dict()["core.0.weight"].add_(1.0)
    assert not parameters_are_equal(acting.online, learner.online), (
        "the copy moved with the learner, so it shares its storage"
    )


# --- Stopping early on a near-greedy curve that has stopped improving -------
#
# What a period is worth is what the fleet collected in it, so a test of the
# stopping rule has to fix what the episodes are worth. The actors below are
# the real ones, driven by the real run, with only `run_episode` scripted: the
# period cutting, the plateau counting and the stop itself are the run's own.

#: Decisions one scripted episode spends, so a selection period of
#: `PERIOD_DECISIONS` is exactly `PERIOD_EPISODES` episodes and a crossing lands
#: where the test says.
EPISODE_DECISIONS = 4
PERIOD_EPISODES = 5
PERIOD_DECISIONS = PERIOD_EPISODES * EPISODE_DECISIONS
EPISODE_ROUND_MS = 1000.0


def scripted_episode(wave: int) -> EpisodeResult:
    """One delivered episode worth `wave`, spending four decisions."""
    return EpisodeResult(
        summary=EpisodeSummary(
            episode_id=f"scripted-{wave}",
            profile_id="fake-profile-v1",
            final_wave=wave,
            decisions=EPISODE_DECISIONS,
            purchases=0,
            termination=TerminationOutcome.GAME_OVER,
            elapsed_wall_seconds=0.0,
            game_speed=8.0,
            invalid_transitions=0,
            game_ms=EPISODE_ROUND_MS,
            round_ms=EPISODE_ROUND_MS,
        ),
        sequences_offered=0,
        sequences_accepted=0,
        total_reward=0.0,
        wait_decisions=0,
    )


def play(training: TrainingRun, waves: Sequence[int], *, actor: int = 0) -> None:
    """Give one actor the final waves its episodes will be worth, in order.

    The last wave repeats for ever, so an actor that outlives its script keeps
    collecting at that level rather than raising out of its own thread.
    """
    script = chain(waves, repeat(waves[-1]))
    training.actors[actor].run_episode = lambda: scripted_episode(  # type: ignore[method-assign]
        next(script)
    )


def period_means(means: Sequence[int]) -> list[int]:
    """A wave script whose consecutive periods average exactly these."""
    return [wave for wave in means for _ in range(PERIOD_EPISODES)]


def scripted_fleet(waves: Sequence[int], **overrides: Any) -> TrainingRun:
    """A fleet of one whose episodes are worth exactly `waves`, in order."""
    settings: dict[str, Any] = {
        "budget_decisions": 20 * PERIOD_DECISIONS,
        "selection_period_decisions": PERIOD_DECISIONS,
        "early_stop_patience_periods": 2,
        "early_stop_min_improvement": 0.2,
    }
    settings.update(overrides)
    training = fleet([environment()], **settings)
    play(training, waves)
    return training


def test_a_plateaued_fleet_stops_after_exactly_the_patience_it_was_given() -> None:
    """The rule itself: two periods that add nothing, and the budget is not spent."""
    training = scripted_fleet(period_means([10, 10, 10, 10, 10]))

    report = training.run()

    plateau = report.plateau
    # Period 1 set the baseline; periods 2 and 3 failed to improve on it, and
    # the second of those is the patience the run was given.
    assert plateau.stopped_at_period == 3
    assert plateau.periods_closed == 3
    assert plateau.best_mean_final_wave == 10
    assert training.stopped_early and training.finished
    assert [period.mean_final_wave for period in report.selection_periods] == [10, 10, 10]
    assert [period.decisions_at_end for period in report.selection_periods] == [20, 40, 60]
    # Three whole periods, and not an episode past the crossing that stopped it.
    assert report.episodes == 3 * PERIOD_EPISODES
    assert report.decisions < training.config.budget_decisions


def test_a_fleet_that_keeps_improving_spends_its_whole_budget() -> None:
    """The curve is still going up, so the budget is still buying something."""
    # Six periods of budget and six improving periods: the run is still
    # learning at every crossing it is judged on.
    training = scripted_fleet(
        period_means([10, 11, 12, 13, 14, 15]),
        budget_decisions=6 * PERIOD_DECISIONS,
    )

    report = training.run()

    assert not training.stopped_early
    assert len(report.selection_periods) == 6
    assert report.plateau.stopped_at_period is None
    assert report.plateau.periods_without_improvement == 0
    assert report.decisions >= training.config.budget_decisions


def test_a_period_inside_the_threshold_has_not_improved_on_the_best() -> None:
    """Half a wave on a noisy curve is not progress, and must not buy patience."""
    training = scripted_fleet(period_means([10, 11, 12, 13]), early_stop_min_improvement=3.0)

    report = training.run()

    # Every period beat the one before it by a wave and none of them cleared
    # the threshold, so the run stopped on a curve that was still moving up.
    assert report.plateau.stopped_at_period == 3
    # The bar stayed where the curve last really moved: the first period's own
    # mean, not the highest the run had seen by then.
    assert report.plateau.best_mean_final_wave == 10


def test_patience_zero_never_stops_a_run() -> None:
    """The default: every run measured so far spent its whole budget."""
    training = scripted_fleet(period_means([10, 10, 10, 10]), early_stop_patience_periods=0)

    report = training.run()

    assert not training.stopped_early
    assert report.decisions >= training.config.budget_decisions
    # The periods were still closed and still measured: the curve is reported
    # whether or not the run is allowed to stop itself on it.
    assert len(report.selection_periods) == 20
    assert report.plateau.periods_without_improvement > 0


def test_the_first_period_sets_the_baseline_and_cannot_stop_the_run() -> None:
    """There must be something to fail to improve on before a run has stopped."""
    training = scripted_fleet(period_means([10, 10, 10]), early_stop_patience_periods=1)

    report = training.run()

    assert report.plateau.stopped_at_period == 2, "the first period stopped the run"
    assert report.selection_periods[0].mean_final_wave == 10


def test_a_period_is_measured_over_the_near_greedy_actors_alone() -> None:
    """Under a ladder the searching actors are not the policy's performance.

    The searching actor here reaches far higher waves than the near-greedy
    ones. Pooled, the periods would sit well above the series a readout cites;
    the run has to judge itself on the series that reads as the policy.
    """
    schedule = ExplorationSchedule.for_option(
        "ladder",
        actors=3,
        epsilon_start=0.9,
        epsilon_end=SCHEDULE.epsilon_end,
        anneal_decisions=1,
    )
    training = fleet(
        [environment() for _ in range(3)],
        exploration=schedule,
        # A hundred episodes, twenty periods: enough that the racing threads
        # always leave two near-greedy periods after the best one.
        budget_decisions=100 * EPISODE_DECISIONS,
        selection_period_decisions=PERIOD_DECISIONS,
        early_stop_patience_periods=2,
    )
    # The bottom two rungs of a ladder of three are near-greedy; actor 0, at
    # 0.4, is searching.
    assert training.near_greedy_actor_ids == {"fake-1:stacked-dqn", "fake-2:stacked-dqn"}
    play(training, [40], actor=0)
    play(training, [7], actor=1)
    play(training, [7], actor=2)

    report = training.run()

    assert training.stopped_early
    assert report.plateau.best_mean_final_wave == 7
    # 7 wherever a near-greedy actor ended an episode inside the period, and
    # nothing at all where none did - never the 40 the searching actor reached.
    assert {period.mean_final_wave for period in report.selection_periods} <= {7.0, None}
    assert any(period.near_greedy_episodes for period in report.selection_periods)


def test_the_plateau_a_run_resumes_from_is_the_one_its_parent_left(tmp_path: Path) -> None:
    """A run trained in two sittings is judged on one curve, not on two.

    The tracker travels in the checkpoint, so the second segment stops on the
    period that completes the parent's patience instead of counting again from
    zero and spending the rest of the budget on a curve that already plateaued.
    """
    parent = scripted_fleet(period_means([10, 10]), budget_decisions=2 * PERIOD_DECISIONS)

    parent.run()

    # Two periods closed and one of them without improvement: one short of the
    # patience of two, so the parent itself did not stop.
    assert not parent.stopped_early
    assert parent.report.plateau.periods_closed == 2
    assert parent.report.plateau.periods_without_improvement == 1

    path = tmp_path / "latest.pt"
    write_checkpoint(
        path,
        identity=CheckpointIdentity(
            run_id="parent",
            backbone="stacked-dqn",
            profile_id="fake-profile-v1",
            observation_schema="1",
            action_schema="1",
            reward_schema="1",
            source_revision="test",
        ),
        progress=TrainingProgress(
            environment_decisions=parent.report.decisions,
            environment_game_ms=parent.report.game_ms,
            episodes=parent.report.episodes,
            checkpoint_periods_closed=parent.report.plateau.periods_closed,
            best_period_near_greedy_mean=parent.report.plateau.best_mean_final_wave,
            periods_without_improvement=parent.report.plateau.periods_without_improvement,
        ),
        backbone_state={"weight": torch.zeros(2)},
        resolved_config={},
        replay_provenance={},
    )
    state = resume_state(path)
    assert state.periods_closed == 2 and state.best_period_near_greedy_mean == 10

    # Exactly the progress `train.py` builds a resumed segment with.
    child = scripted_fleet(
        period_means([10, 10, 10]),
        budget_decisions=20 * PERIOD_DECISIONS,
        report=TrainingProgressReport(
            decisions=state.decisions,
            game_ms=state.game_ms,
            episodes=state.episodes,
            plateau=NearGreedyPlateau(
                periods_closed=state.periods_closed or 0,
                best_mean_final_wave=state.best_period_near_greedy_mean,
                periods_without_improvement=state.periods_without_improvement,
                restored=state.periods_closed is not None,
            ),
        ),
    )

    report = child.run()

    assert report.plateau.restored
    # One more period without improvement completes the patience, so the child
    # stops at the run's third period rather than at its own second.
    assert report.plateau.stopped_at_period == 3
    assert child.stopped_early
    assert len(report.selection_periods) == 1, "the child closed one period of its own"


def test_a_checkpoint_written_before_early_stopping_restores_no_tracker() -> None:
    """Absence is read as absence, not as a run that had closed no period."""
    progress = TrainingProgress(environment_decisions=10, environment_game_ms=1000.0)

    assert progress.checkpoint_periods_closed is None
    assert progress.best_period_near_greedy_mean is None


def test_a_curve_creeping_up_below_the_threshold_is_never_stopped() -> None:
    """The bar is where the curve last really moved, not the highest it reached.

    Raising it on every new maximum would raise it by exactly what a creeping
    curve gained, so a run gaining a tenth of a wave a period would stop while
    one gaining nothing at all carried on. Held at the last counted
    improvement, the creep clears the threshold every other period and the run
    goes on collecting.
    """
    plateau = NearGreedyPlateau()

    for mean in (5.0, 5.1, 5.2, 5.3):
        plateau.close_period(mean, min_improvement=0.2)
        assert not plateau.plateaued(2)

    # 5.2 cleared 5.0 + 0.2 and reset the streak; 5.3 did not clear 5.2 + 0.2.
    assert plateau.best_mean_final_wave == 5.2
    assert plateau.periods_without_improvement == 1


def test_a_period_that_measured_nothing_leaves_the_plateau_where_it_was() -> None:
    """No near-greedy episode in it is no evidence either way.

    It is still a period that closed - the ordinals must not skip - but it
    neither resets the streak, which would buy a plateaued run more budget for
    having collected nothing, nor moves the best, which nothing measured.
    """
    plateau = NearGreedyPlateau()
    plateau.close_period(5.0, min_improvement=0.2)
    plateau.close_period(5.1, min_improvement=0.2)
    before = (plateau.best_mean_final_wave, plateau.periods_without_improvement)

    plateau.close_period(None, min_improvement=0.2)

    assert (plateau.best_mean_final_wave, plateau.periods_without_improvement) == before
    assert plateau.periods_closed == 3, "the period still closed"


def test_a_checkpoint_is_written_on_the_cadence_and_at_every_period_close() -> None:
    """The cadence and the selection period are independent.

    Checkpoints every 12 decisions and periods every 20: 20 is not a multiple
    of 12, yet every period close still has a checkpoint named by its own
    decisions, and a crossing both land on is written once.
    """
    training = scripted_fleet(
        period_means([10, 11, 12]),
        budget_decisions=3 * PERIOD_DECISIONS,
        checkpoint_every_decisions=12,
        early_stop_patience_periods=0,
    )
    written: list[int] = []
    training.numbered_checkpoint = lambda report: written.append(report.decisions)

    report = training.run()

    assert written == [12, 20, 24, 36, 40, 48, 60]
    assert report.checkpoints_written == len(written)
    closes = [period.decisions_at_end for period in report.selection_periods]
    assert closes == [20, 40, 60]
    assert set(closes) <= set(written)


def test_without_a_cadence_checkpoints_are_written_only_where_periods_close() -> None:
    training = scripted_fleet(
        period_means([10, 11, 12]),
        budget_decisions=3 * PERIOD_DECISIONS,
        early_stop_patience_periods=0,
    )
    written: list[int] = []
    training.numbered_checkpoint = lambda report: written.append(report.decisions)

    training.run()

    assert written == [20, 40, 60]


# --- Stopping below a pre-registered kill bar on the decision axis -----------
#
# Each scripted episode is four decisions, so a single actor's episodes end at
# decisions 4, 8, 12, ... and a bar's window holds exactly the episodes the
# test counts into it.


def kill_bar_fleet(waves: Sequence[int], *bars: KillBar, **overrides: Any) -> TrainingRun:
    """A fleet of one with only kill bars to stop it: the plateau rule is off."""
    settings: dict[str, Any] = {"budget_decisions": 120, "kill_bars": bars}
    settings.update(overrides)
    training = fleet([environment()], **settings)
    play(training, waves)
    return training


def test_a_run_below_its_kill_bar_stops_at_the_bar() -> None:
    bar = KillBar(at_decisions=40, window_start_decisions=20, min_mean_final_wave=6.0)
    training = kill_bar_fleet([5], bar)

    report = training.run()

    assert training.stopped_early and training.finished
    assert training.killed_by == report.kill_bar_checks[0]
    [check] = report.kill_bar_checks
    # The episodes ending at 24, 28, 32, 36 and 40: the window is (20, 40].
    assert (check.near_greedy_episodes, check.mean_final_wave) == (5, 5.0)
    assert check.decisions == 40 and check.stopped
    # Stopped at the boundary that reached the bar, not an episode later.
    assert report.decisions == 40
    assert report.plateau.stopped_at_period is None


def test_a_run_that_clears_its_kill_bar_carries_on_and_records_the_check() -> None:
    """Only the window counts: the poor episodes before its start do not."""
    bar = KillBar(at_decisions=40, window_start_decisions=20, min_mean_final_wave=6.0)
    # Five episodes (decisions 4..20) at wave 1, then wave 8 from decision 24 on.
    training = kill_bar_fleet([1] * 5 + [8], bar)

    report = training.run()

    assert not training.stopped_early and training.killed_by is None
    assert report.decisions >= training.config.budget_decisions
    [check] = report.kill_bar_checks
    assert (check.near_greedy_episodes, check.mean_final_wave, check.stopped) == (5, 8.0, False)


def test_a_kill_bar_whose_window_measured_nothing_does_not_stop_the_run() -> None:
    """No valid near-greedy episode in the window: recorded, and not a stop."""
    bar = KillBar(at_decisions=40, window_start_decisions=20, min_mean_final_wave=6.0)
    training = kill_bar_fleet([5], bar)
    invalid = scripted_episode(5)
    invalid = replace(
        invalid,
        summary=replace(invalid.summary, termination=TerminationOutcome.OBSERVATION_INVALID),
    )
    training.actors[0].run_episode = lambda: invalid  # type: ignore[method-assign]

    report = training.run()

    assert not training.stopped_early
    [check] = report.kill_bar_checks
    assert (check.near_greedy_episodes, check.mean_final_wave, check.stopped) == (0, None, False)


def test_a_kill_bar_reads_the_near_greedy_actors_alone() -> None:
    """The searching actor's high waves cannot hold a failing policy up."""
    schedule = ExplorationSchedule.for_option(
        "ladder",
        actors=3,
        epsilon_start=0.9,
        epsilon_end=SCHEDULE.epsilon_end,
        anneal_decisions=1,
    )
    bar = KillBar(at_decisions=60, window_start_decisions=0, min_mean_final_wave=6.0)
    training = fleet(
        [environment() for _ in range(3)],
        exploration=schedule,
        budget_decisions=400,
        kill_bars=(bar,),
    )
    play(training, [40], actor=0)
    play(training, [5], actor=1)
    play(training, [5], actor=2)

    report = training.run()

    [check] = report.kill_bar_checks
    assert check.stopped and check.mean_final_wave == 5.0
    assert training.stopped_early


def test_a_resumed_run_skips_the_bars_its_parent_already_passed() -> None:
    passed = KillBar(at_decisions=20, window_start_decisions=0, min_mean_final_wave=99.0)
    ahead = KillBar(at_decisions=60, window_start_decisions=40, min_mean_final_wave=6.0)
    training = kill_bar_fleet(
        [8],
        passed,
        ahead,
        report=TrainingProgressReport(decisions=40, game_ms=10_000.0),
    )

    report = training.run()

    # The parent answered the bar at 20 - it would have stopped there had it
    # failed - and this segment's episodes are placed from 40 on, not from 0.
    [check] = report.kill_bar_checks
    assert check.bar == ahead
    assert (check.near_greedy_episodes, check.mean_final_wave, check.stopped) == (5, 8.0, False)


def test_a_kill_bar_window_must_end_after_it_starts() -> None:
    with pytest.raises(ValueError, match="window"):
        KillBar(at_decisions=8000, window_start_decisions=12000, min_mean_final_wave=8.6)
