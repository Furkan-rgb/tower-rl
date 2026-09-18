"""A fleet of actors collecting into one learner, without a device.

No emulator, no adb, no bridge: every actor drives its own `FakeRunPort`, so
what is under test is the plumbing a fleet adds - concurrency, one shared
buffer, a budget counted across actors, per-actor failure isolation, and the
guard that keeps a forward pass out of the middle of an optimisation step.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

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
    collection_windows,
)
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.infrastructure.instrumented_bridge import BridgeTimeoutError  # noqa: E402
from tower_rl.infrastructure.instrumented_run_adapter import (  # noqa: E402
    InstrumentedRunAdapter,
)
from tower_rl.learning.backbone import LearnMetrics, SequenceBatch  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.recurrent_q import RecurrentQBackbone, RecurrentQConfig  # noqa: E402
from tower_rl.ports.run_port import RunPortError  # noqa: E402

torch.set_num_threads(1)

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

#: Long enough that two actors overlap in it and short enough that the suite
#: stays fast. A real decision costs hundreds of milliseconds on device.
DECISION_SECONDS = 0.002


@dataclass
class Overlap:
    """How many actors were inside a decision at once, at their busiest."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    active: int = 0
    peak: int = 0

    @contextmanager
    def deciding(self) -> Any:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1


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
    learner = backbone or RecurrentQBackbone(
        config=RecurrentQConfig(seed=0), network_config=SMALL
    )
    replay = PrioritizedSequenceReplay(capacity=256, seed=0)
    actors = [
        Actor(
            environment=item,
            policy=learner,
            config=ActorConfig(
                actor_id=f"fake-{index}:recurrent-q",
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

    dead = report.actors["fake-1:recurrent-q"]
    assert dead.withdrawn is not None and "liveness expired" in dead.withdrawn
    assert dead.failed_episodes == 3 and dead.decisions == 0
    # Named as it happened, with the instance it names and why it left.
    assert withdrawn == [("fake-1:recurrent-q", dead.withdrawn)]
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

    dead = report.actors["fake-1:recurrent-q"]
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
    dead = training.report.actors["fake-1:recurrent-q"]
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


@dataclass
class WatchedBackbone:
    """A backbone that says when it is mid-update, and who read it during one."""

    inner: RecurrentQBackbone
    torn_reads: list[str] = field(default_factory=list)
    #: Model versions observed by `act`, and where episodes ended in that series.
    versions: list[int] = field(default_factory=list)
    updating: bool = False

    @property
    def model_version(self) -> int:
        return self.inner.model_version

    @property
    def device(self) -> torch.device:
        return self.inner.device

    def act(self, features: Any, state: Any, *, epsilon: float) -> tuple[int, Any]:
        if self.updating:
            self.torn_reads.append("an actor read the network mid-update")
        self.versions.append(self.inner.model_version)
        return self.inner.act(features, state, epsilon=epsilon)

    def initial_state(self) -> Any:
        return self.inner.initial_state()

    def stored_recurrent_state(self, state: Any) -> Any:
        return self.inner.stored_recurrent_state(state)

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        self.updating = True
        try:
            # Wide enough that an unguarded actor would land inside it.
            time.sleep(DECISION_SECONDS)
            return self.inner.learn(batch)
        finally:
            self.updating = False

    def state_dict(self) -> dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.inner.load_state_dict(state)


def test_no_actor_reads_the_network_while_the_learner_updates_it() -> None:
    """Stale parameters are fine; half-updated ones are no policy at all."""
    watched = WatchedBackbone(
        RecurrentQBackbone(config=RecurrentQConfig(seed=0), network_config=SMALL)
    )
    training = fleet(
        [environment() for _ in range(3)], backbone=watched, budget_decisions=200
    )

    report = training.run()

    assert report.optimisation_steps > 0, "nothing would be guarded without updates"
    assert watched.torn_reads == []


def test_a_fleet_of_one_learns_only_between_its_own_episodes() -> None:
    """The single-actor path is the loop it always was: collect, then learn.

    One actor takes its own gradient steps between its own episodes, so the
    parameters never move while it is inside an episode. That is not true of a
    fleet, where another actor's learning lands mid-episode, and it is what
    keeps a run configured with `--actors 1` reproducible.
    """
    watched = WatchedBackbone(
        RecurrentQBackbone(config=RecurrentQConfig(seed=0), network_config=SMALL)
    )
    boundaries: list[int] = []
    training = fleet([environment()], backbone=watched, budget_decisions=200)
    training.on_episode = lambda _: boundaries.append(len(watched.versions))

    report = training.run()

    assert report.optimisation_steps > 0 and len(boundaries) > 1
    start = 0
    for end in boundaries:
        within = set(watched.versions[start:end])
        assert len(within) <= 1, "the parameters moved inside a single episode"
        start = end
    assert len(set(watched.versions)) > 1, "and they did move between episodes"
