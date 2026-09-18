"""A fleet of actors collecting into one learner, without a device.

No emulator, no adb, no bridge: every actor drives its own `FakeRunPort`, so
what is under test is the plumbing a fleet adds - concurrency, one shared
buffer, a budget counted across actors, per-actor failure isolation, and the
guard that keeps a forward pass out of the middle of an optimisation step.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
import torch
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.features import encode_state
from tower_rl.environment.run_environment import (
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.infrastructure.instrumented_bridge import BridgeTimeoutError
from tower_rl.infrastructure.instrumented_run_adapter import (
    InstrumentedRunAdapter,
)
from tower_rl.learning.actor import Actor, ActorConfig
from tower_rl.learning.backbone import (
    LearnMetrics,
    SequenceBatch,
    acting_copy,
    parameters_are_equal,
)
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.replay import PrioritizedSequenceReplay
from tower_rl.learning.stacked_dqn import (
    StackedDqnBackbone,
    StackedDqnConfig,
)
from tower_rl.learning.training import (
    TrainingConfig,
    TrainingRun,
    collection_windows,
)

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

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
    **overrides: Any,
) -> TrainingRun:
    """One arm collecting on these instances, one actor each."""
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
        "budget_decisions": 300,
        "warmup_sequences": 2,
        "batch_size": 2,
        "gradient_steps_per_decision": 0.2,
    }
    settings.update(overrides)
    return TrainingRun(
        actors=actors,
        replay=replay,
        backbone=learner,
        config=TrainingConfig(**settings),
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
        budget_decisions=200,
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
    assert report.decisions >= 200


def test_a_fleet_whose_bridges_have_all_died_still_ends_the_run() -> None:
    """Nothing left collecting is a dead environment however it died."""
    training = fleet(
        [dead_bridge_environment() for _ in range(2)],
        budget_decisions=200,
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
    training = fleet([environment(overlap) for _ in range(3)], budget_decisions=200)

    training.run()

    assert overlap.peak > 1, "the actors never overlapped, so nothing was parallel"
    assert all(progress.episodes > 0 for progress in training.report.actors.values())


def test_every_actor_writes_into_the_one_replay_buffer() -> None:
    """One buffer and one learner: experience is pooled, not split per instance."""
    training = fleet([environment() for _ in range(3)], budget_decisions=200)

    report = training.run()

    stored = {sequence.metadata.actor_id for sequence in training.replay._items}
    assert stored == {progress.actor_id for progress in report.actors.values()}
    assert report.sequences_accepted == training.replay.stats.added


def test_the_budget_is_counted_across_the_fleet() -> None:
    """Three actors spend one budget, and the learner's steps track all of it."""
    training = fleet([environment() for _ in range(3)], budget_decisions=200)

    report = training.run()

    per_actor = [progress.decisions for progress in report.actors.values()]
    assert report.decisions >= 200
    assert sum(per_actor) == report.decisions
    assert all(decisions < report.decisions for decisions in per_actor), (
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
    training = fleet([environment(overlap) for _ in range(3)], budget_decisions=200)

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
        budget_decisions=200,
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
    assert report.decisions >= 200


def test_a_withdrawn_actor_is_not_asked_again_in_a_later_block() -> None:
    """Its instance failed every episode the limit allows; retrying would spin."""
    training = fleet(
        [environment(), environment(refuse_to_start=True)],
        budget_decisions=400,
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
        budget_decisions=200,
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
    training = fleet([environment()], evaluate_every_episodes=1, budget_decisions=20)
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


def test_every_actor_acts_from_a_copy_of_its_own_and_not_from_the_learner() -> None:
    """The point of the whole arrangement: no forward pass touches shared state."""
    training, watched = watched_fleet(3, budget_decisions=200)

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
        [environment(overlap) for _ in range(3)], backbone=watched, budget_decisions=200
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
    training, _ = watched_fleet(1, budget_decisions=200)
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
    training, watched = watched_fleet(3, budget_decisions=200)

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
    training = fleet([instance], backbone=watched, budget_decisions=200)
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
    training, watched = watched_fleet(1, budget_decisions=200)
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
    training, _ = watched_fleet(1, budget_decisions=900, parameter_sync_episodes=3)
    acting = copies(training)[0]

    report = training.run()

    assert report.episodes >= 4
    assert acting.publications == 1 + (report.episodes - 1) // 3
    assert acting.publications < report.episodes


def test_a_publication_leaves_the_state_an_actor_carries_through_an_episode_alone() -> None:
    """The carried history window is the actor's, not the network's, and outlives a refresh."""
    training, _ = watched_fleet(1, budget_decisions=200)
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
