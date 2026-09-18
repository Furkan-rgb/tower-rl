"""The decision-time decomposition: it must add up, and it must not be shared.

Two properties are worth tests of their own. The buckets have to sum to the
wall time actually measured, or an unaccounted cost hides in a number nobody
questions; and each actor has to accumulate into its own profile, or the
measurement of contention would itself contend.

The end-to-end case runs a fleet on `FakeRunPort`, which is a test double: its
transitions exist to exercise the plumbing and never reach a training run.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes.fake_run_port import FakeRunPort  # noqa: E402

from tower_rl.application.actor import Actor, ActorConfig  # noqa: E402
from tower_rl.application.decision_time import (  # noqa: E402
    BLOCKED,
    BRIDGE_ROUND_TRIP,
    BUCKETS,
    OBSERVATION_DECODE,
    POLICY_FORWARD,
    RESIDUAL,
    DecisionTimeProfile,
    pooled,
)
from tower_rl.application.replay import PrioritizedSequenceReplay  # noqa: E402
from tower_rl.application.run_environment import (  # noqa: E402
    CadenceConfig,
    InstrumentedRunEnvironment,
)
from tower_rl.application.training import (  # noqa: E402
    TrainingConfig,
    TrainingRun,
)
from tower_rl.domain.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.network import NetworkConfig  # noqa: E402
from tower_rl.learning.stacked_dqn import (  # noqa: E402
    StackedDqnBackbone,
    StackedDqnConfig,
)

torch.set_num_threads(1)

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

#: Float noise over a few thousand additions, and nothing more. The residual is
#: computed by subtraction, so anything above this is a real accounting fault.
TOLERANCE_SECONDS = 1e-6


def test_the_buckets_sum_to_the_measured_total() -> None:
    profile = DecisionTimeProfile()
    with profile.collecting():
        for _ in range(3):
            with profile.span(BRIDGE_ROUND_TRIP):
                time.sleep(0.005)
            with profile.span(OBSERVATION_DECODE):
                sum(range(10_000))
        breakdown = profile.snapshot()

    total = sum(breakdown.buckets[name].wall_seconds for name in BUCKETS)
    assert abs(total - breakdown.elapsed_seconds) < TOLERANCE_SECONDS
    assert abs(breakdown.accounting_error_seconds) < TOLERANCE_SECONDS
    # Nothing was double counted, so the remainder is a real remainder.
    assert breakdown.residual.wall_seconds >= -TOLERANCE_SECONDS
    assert breakdown.buckets[BRIDGE_ROUND_TRIP].count == 3


def test_sleeping_and_computing_are_told_apart() -> None:
    """The discriminator the whole measurement rests on.

    A bucket that sleeps spends wall time and no CPU; a bucket that computes
    spends both. Without this pair of clocks a slow bucket cannot be read as
    either an idle host or a contended one.
    """
    profile = DecisionTimeProfile()
    with profile.collecting():
        with profile.span(BRIDGE_ROUND_TRIP):
            time.sleep(0.05)
        with profile.span(OBSERVATION_DECODE):
            deadline = time.perf_counter() + 0.05
            while time.perf_counter() < deadline:
                sum(range(1000))
        breakdown = profile.snapshot()

    waiting = breakdown.buckets[BRIDGE_ROUND_TRIP]
    computing = breakdown.buckets[OBSERVATION_DECODE]
    assert waiting.cpu_seconds < waiting.wall_seconds / 2
    assert computing.cpu_seconds > computing.wall_seconds / 2


def test_waiting_for_a_lock_is_charged_to_blocked() -> None:
    """Only the wait is `blocked` - the work done holding the lock is not.

    Charging the body to `blocked` too would read as contention that is really
    the holder's own work, so the two are separated by making them very
    different lengths and checking the wait alone lands in the bucket.
    """
    wait_seconds, body_seconds = 0.10, 0.30
    held = threading.Lock()
    taken = threading.Event()

    def hold_it_for_a_known_time() -> None:
        held.acquire()
        taken.set()
        time.sleep(wait_seconds)
        held.release()

    holder = threading.Thread(target=hold_it_for_a_known_time)
    holder.start()
    taken.wait()

    profile = DecisionTimeProfile()
    with profile.collecting():
        with profile.acquiring(held):
            time.sleep(body_seconds)
        breakdown = profile.snapshot()
    holder.join()

    blocked = breakdown.buckets[BLOCKED]
    assert blocked.count == 1
    # The wait, and nothing like the body: charging the body here would read as
    # about 0.4 s rather than about 0.1 s.
    assert wait_seconds * 0.5 <= blocked.wall_seconds < wait_seconds + body_seconds * 0.5
    # The wait was off the CPU, which is what separates it from busy work.
    assert blocked.cpu_seconds < 0.02
    # The body's time is accounted for, just not as contention: it opened no
    # bucket of its own, so it is the remainder.
    assert breakdown.elapsed_seconds >= (wait_seconds + body_seconds) * 0.9
    assert breakdown.residual.wall_seconds >= body_seconds * 0.8


def test_two_profiles_do_not_see_each_other() -> None:
    """Per-actor isolation, at the accumulator itself: no shared state at all."""
    first, second = DecisionTimeProfile(), DecisionTimeProfile()
    barrier = threading.Barrier(2)

    def work(profile: DecisionTimeProfile, bucket: str) -> None:
        barrier.wait()
        with profile.collecting():
            for _ in range(5):
                with profile.span(bucket):
                    time.sleep(0.001)

    threads = [
        threading.Thread(target=work, args=(first, BRIDGE_ROUND_TRIP)),
        threading.Thread(target=work, args=(second, POLICY_FORWARD)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert first.snapshot().buckets[BRIDGE_ROUND_TRIP].count == 5
    assert first.snapshot().buckets[POLICY_FORWARD].count == 0
    assert second.snapshot().buckets[POLICY_FORWARD].count == 5
    assert second.snapshot().buckets[BRIDGE_ROUND_TRIP].count == 0


def test_a_delta_describes_the_interval_between_two_snapshots() -> None:
    profile = DecisionTimeProfile()
    with profile.collecting():
        with profile.span(POLICY_FORWARD):
            time.sleep(0.005)
        earlier = profile.snapshot()
        with profile.span(POLICY_FORWARD):
            time.sleep(0.005)
        later = profile.snapshot()
    interval = later.since(earlier)
    assert interval.buckets[POLICY_FORWARD].count == 1
    assert interval.decisions == 1
    accounted = sum(interval.buckets[name].wall_seconds for name in BUCKETS)
    assert abs(accounted - interval.elapsed_seconds) < TOLERANCE_SECONDS


def environment() -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )


def fleet(count: int) -> TrainingRun:
    learner = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=2), network_config=SMALL
    )
    replay = PrioritizedSequenceReplay(capacity=256, seed=0)
    actors = [
        Actor(
            environment=environment(),
            policy=learner,
            config=ActorConfig(
                actor_id=f"fake-{index}", sequence_length=6, burn_in=1, stride=3
            ),
            replay=replay,
        )
        for index in range(count)
    ]
    return TrainingRun(
        actors=actors,
        replay=replay,
        backbone=learner,
        config=TrainingConfig(
            budget_decisions=200,
            warmup_sequences=2,
            batch_size=2,
            gradient_steps_per_decision=0.2,
        ),
    )


def test_a_fleet_reports_where_each_actor_spent_its_time() -> None:
    """End to end on the fake port: every actor's buckets populate and add up."""
    training = fleet(3)
    report = training.run()

    assert report.decisions >= 200
    breakdowns = []
    for progress in report.actors.values():
        breakdown = progress.decision_time
        assert breakdown is not None, progress.actor_id
        breakdowns.append(breakdown)
        # Every actor's own decisions, counted where its policy was called.
        assert breakdown.decisions == progress.decisions
        assert breakdown.buckets[BRIDGE_ROUND_TRIP].wall_seconds > 0
        assert breakdown.buckets[OBSERVATION_DECODE].wall_seconds > 0
        assert breakdown.buckets[POLICY_FORWARD].wall_seconds > 0
        accounted = sum(breakdown.buckets[name].wall_seconds for name in BUCKETS)
        assert abs(accounted - breakdown.elapsed_seconds) < TOLERANCE_SECONDS
        # A negative residual would mean two buckets charged the same instant.
        assert breakdown.buckets[RESIDUAL].wall_seconds >= -TOLERANCE_SECONDS
        assert breakdown.decisions_per_hour > 0

    # Per-actor isolation through the real loop: the fleet's decisions are the
    # actors' decisions, each counted once.
    assert sum(item.decisions for item in breakdowns) == sum(
        progress.decisions for progress in report.actors.values()
    )
    fleet_total = pooled(breakdowns)
    assert abs(fleet_total.accounting_error_seconds) < TOLERANCE_SECONDS * len(breakdowns)


def test_the_record_a_run_serialises_is_json_shaped() -> None:
    profile = DecisionTimeProfile()
    with profile.collecting(), profile.span(POLICY_FORWARD):
        time.sleep(0.001)
    record: dict[str, Any] = profile.snapshot().as_record()
    assert set(record["buckets"]) == set(BUCKETS)
    assert record["decisions"] == 1
    assert record["buckets"][POLICY_FORWARD]["wall_ms_per_decision"] > 0
