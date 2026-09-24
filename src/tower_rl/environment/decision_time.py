"""Where one actor's wall time goes, measured on the actor's own thread.

Per-actor throughput decays as actors are added (9,498 decisions/hour at four
actors, 8,479 at eight) while per-emulator CPU stays flat near 96%, and nothing
recorded could say whether the host was *idle waiting on emulators* or *busy and
contended inside Python*.  This module answers that, and only that.

One decision's wall time is cut into disjoint buckets - the bridge round trip,
host-side observation decoding, the policy forward pass, learner work, time
blocked acquiring a shared lock, and a residual that makes the rest add up to
the measured total.  Each bucket carries two clocks: elapsed wall time
(`perf_counter`) and the thread's own CPU time (`thread_time`).  That pair is
the discriminator the decay hypothesis needed:

* wall high, CPU near zero, inside `bridge_round_trip` - the host is idle,
  waiting for an emulator to simulate;
* wall high, CPU near wall - the host is genuinely computing;
* wall high, CPU far below wall, inside a bucket that holds no lock and does no
  I/O (`observation_decode`, `policy_forward`) - the thread was runnable but not
  running, which on CPython means it was waiting for the interpreter lock;
* wall high inside `blocked` - the thread was waiting for one of *this* code's
  locks, which is a different failure with a different fix.

Nothing here is shared between actors.  Each actor accumulates into its own
profile on its own thread, and a run publishes an immutable snapshot of it at
episode boundaries under the lock it already takes - so measuring the fleet
cannot itself serialise the fleet.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import TracebackType

#: Issuing a bridge command and receiving its reply: the emulator simulating.
BRIDGE_ROUND_TRIP = "bridge_round_trip"
#: Turning a bridge reply into the observation the policy consumes - building
#: the run state and encoding its features. Pure host CPU by construction.
OBSERVATION_DECODE = "observation_decode"
#: Action selection.
POLICY_FORWARD = "policy_forward"
#: Gradient steps and parameter publication, taken on the actor's own thread.
LEARNER_STEP = "learner_step"
#: Waiting to acquire a lock the fleet shares. The acquisition only; whatever
#: the caller then does under the lock is charged where it belongs.
BLOCKED = "blocked"
#: Measured total minus everything above, so nothing can hide unaccounted.
RESIDUAL = "residual"

MEASURED_BUCKETS = (
    BRIDGE_ROUND_TRIP,
    OBSERVATION_DECODE,
    POLICY_FORWARD,
    LEARNER_STEP,
    BLOCKED,
)
BUCKETS = (*MEASURED_BUCKETS, RESIDUAL)


@dataclass(frozen=True)
class BucketTime:
    """One bucket's two clocks, and how many times it was entered."""

    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    count: int = 0

    @property
    def off_cpu_seconds(self) -> float:
        """Wall time this bucket spent not executing on a CPU.

        In `bridge_round_trip` that is the emulator working. Anywhere else it is
        the thread waiting for something - a lock, or the interpreter lock.
        """
        return self.wall_seconds - self.cpu_seconds

    def __sub__(self, earlier: BucketTime) -> BucketTime:
        return BucketTime(
            wall_seconds=self.wall_seconds - earlier.wall_seconds,
            cpu_seconds=self.cpu_seconds - earlier.cpu_seconds,
            count=self.count - earlier.count,
        )

    def __add__(self, other: BucketTime) -> BucketTime:
        return BucketTime(
            wall_seconds=self.wall_seconds + other.wall_seconds,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            count=self.count + other.count,
        )


@dataclass(frozen=True)
class DecisionTimeBreakdown:
    """One actor's (or one fleet's) time decomposition over some span.

    Immutable, so a snapshot published by the thread that owns the profile can
    be read by any other thread without a second lock of its own.
    """

    decisions: int
    #: Wall time the actor's thread was collecting, which is what the buckets
    #: decompose. `residual` is defined as this minus the measured buckets, so
    #: the six always sum to it exactly.
    elapsed_seconds: float
    #: That same span's CPU time on that thread.
    cpu_seconds: float
    buckets: dict[str, BucketTime]

    @property
    def residual(self) -> BucketTime:
        return self.buckets[RESIDUAL]

    @property
    def accounting_error_seconds(self) -> float:
        """Measured total minus the buckets. Zero up to float rounding."""
        return self.elapsed_seconds - sum(item.wall_seconds for item in self.buckets.values())

    @property
    def decisions_per_hour(self) -> float:
        """Decisions per hour *of collecting*, not per wall hour: `elapsed_seconds`
        counts only the time inside collecting blocks, so an actor that also runs
        periodic evaluation reports the rate it collects at while collecting.
        """
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.decisions / self.elapsed_seconds * 3600

    @property
    def busy_fraction(self) -> float:
        """The share of the span this thread was actually executing Python.

        The span is collecting time, as `decisions_per_hour` notes; time spent
        in `measured_apart` is in neither the numerator nor the denominator.
        """
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.cpu_seconds / self.elapsed_seconds

    def since(self, earlier: DecisionTimeBreakdown) -> DecisionTimeBreakdown:
        """This cumulative snapshot minus an earlier one: the span between them."""
        return DecisionTimeBreakdown(
            decisions=self.decisions - earlier.decisions,
            elapsed_seconds=self.elapsed_seconds - earlier.elapsed_seconds,
            cpu_seconds=self.cpu_seconds - earlier.cpu_seconds,
            buckets={name: self.buckets[name] - earlier.buckets[name] for name in BUCKETS},
        )

    def __add__(self, other: DecisionTimeBreakdown) -> DecisionTimeBreakdown:
        """Pool two actors. Wall time then counts thread-seconds, not fleet seconds."""
        return DecisionTimeBreakdown(
            decisions=self.decisions + other.decisions,
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
            cpu_seconds=self.cpu_seconds + other.cpu_seconds,
            buckets={name: self.buckets[name] + other.buckets[name] for name in BUCKETS},
        )

    def as_record(self) -> dict[str, object]:
        """The JSON shape: the decomposition, per decision as well as in total."""
        per_decision = 1.0 / self.decisions if self.decisions else 0.0
        return {
            "decisions": self.decisions,
            "decisions_per_hour": round(self.decisions_per_hour, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "busy_fraction": round(self.busy_fraction, 4),
            "accounting_error_seconds": round(self.accounting_error_seconds, 9),
            "buckets": {
                name: {
                    "wall_seconds": round(bucket.wall_seconds, 4),
                    "cpu_seconds": round(bucket.cpu_seconds, 4),
                    "off_cpu_seconds": round(bucket.off_cpu_seconds, 4),
                    "wall_ms_per_decision": round(bucket.wall_seconds * per_decision * 1000, 3),
                    "cpu_ms_per_decision": round(bucket.cpu_seconds * per_decision * 1000, 3),
                    "count": bucket.count,
                }
                for name, bucket in ((name, self.buckets[name]) for name in BUCKETS)
            },
        }


EMPTY_BREAKDOWN = DecisionTimeBreakdown(
    decisions=0,
    elapsed_seconds=0.0,
    cpu_seconds=0.0,
    buckets={name: BucketTime() for name in BUCKETS},
)


class _Span:
    """Charges one region of the owning thread's time to one bucket."""

    __slots__ = ("_bucket", "_cpu", "_profile", "_wall")

    def __init__(self, profile: DecisionTimeProfile, bucket: str) -> None:
        self._profile = profile
        self._bucket = bucket
        self._wall = 0.0
        self._cpu = 0.0

    def __enter__(self) -> _Span:
        self._wall = time.perf_counter()
        self._cpu = time.thread_time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Charged whether or not the region raised: the time was spent either way.
        self._profile.charge(
            self._bucket,
            time.perf_counter() - self._wall,
            time.thread_time() - self._cpu,
        )


class _Acquire:
    """Takes a shared lock, charging only the wait to `blocked`.

    The body under the lock is not charged here - it is charged wherever it
    belongs - so this never nests inside another bucket and the residual stays
    a real remainder rather than a subtraction artefact.
    """

    __slots__ = ("_lock", "_profile")

    def __init__(self, profile: DecisionTimeProfile, lock: threading.Lock) -> None:
        self._profile = profile
        self._lock = lock

    def __enter__(self) -> _Acquire:
        wall = time.perf_counter()
        cpu = time.thread_time()
        self._lock.acquire()
        self._profile.charge(
            BLOCKED, time.perf_counter() - wall, time.thread_time() - cpu
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()


class _Collecting:
    """The span of thread time the buckets decompose: one block of collection."""

    __slots__ = ("_profile",)

    def __init__(self, profile: DecisionTimeProfile) -> None:
        self._profile = profile

    def __enter__(self) -> _Collecting:
        self._profile.open_block()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._profile.close_block()


@dataclass
class DecisionTimeProfile:
    """One actor's accumulator, mutated only by that actor's own thread.

    There is deliberately no lock and nothing shared here: a counter shared
    across the fleet would serialise exactly the contention it was built to
    measure. What crosses threads is `snapshot`, an immutable value the owning
    thread publishes at its own episode boundaries.
    """

    _wall: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(MEASURED_BUCKETS, 0.0)
    )
    _cpu: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(MEASURED_BUCKETS, 0.0)
    )
    _count: dict[str, int] = field(default_factory=lambda: dict.fromkeys(MEASURED_BUCKETS, 0))
    #: Collecting time from blocks that have already ended. A run can be
    #: advanced in blocks, and an actor gets a fresh thread for each one.
    _closed_wall: float = 0.0
    _closed_cpu: float = 0.0
    #: The open block's origins on both clocks, or None between blocks.
    _origin_wall: float | None = None
    _origin_cpu: float = 0.0

    def collecting(self) -> _Collecting:
        """Measure the total this actor's buckets are a decomposition of."""
        return _Collecting(self)

    def span(self, bucket: str) -> _Span:
        return _Span(self, bucket)

    def acquiring(self, lock: threading.Lock) -> _Acquire:
        return _Acquire(self, lock)

    def charge(self, bucket: str, wall_seconds: float, cpu_seconds: float) -> None:
        self._wall[bucket] += wall_seconds
        self._cpu[bucket] += cpu_seconds
        self._count[bucket] += 1

    @property
    def block_open(self) -> bool:
        """Whether a collecting block is open right now."""
        return self._origin_wall is not None

    def open_block(self) -> None:
        self._origin_wall = time.perf_counter()
        self._origin_cpu = time.thread_time()

    def close_block(self) -> None:
        if self._origin_wall is None:
            return
        self._closed_wall += time.perf_counter() - self._origin_wall
        self._closed_cpu += time.thread_time() - self._origin_cpu
        self._origin_wall = None

    def snapshot(self) -> DecisionTimeBreakdown:
        """This actor's decomposition so far, as an immutable value.

        Taken on the owning thread. The residual is computed here rather than
        accumulated, which is what guarantees the buckets sum to the measured
        total: it is defined as the part of the total that no bucket claimed.
        """
        elapsed, cpu = self._closed_wall, self._closed_cpu
        if self._origin_wall is not None:
            elapsed += time.perf_counter() - self._origin_wall
            cpu += time.thread_time() - self._origin_cpu
        buckets = {
            name: BucketTime(self._wall[name], self._cpu[name], self._count[name])
            for name in MEASURED_BUCKETS
        }
        decisions = buckets[POLICY_FORWARD].count
        buckets[RESIDUAL] = BucketTime(
            wall_seconds=elapsed - sum(item.wall_seconds for item in buckets.values()),
            cpu_seconds=cpu - sum(item.cpu_seconds for item in buckets.values()),
            count=decisions,
        )
        return DecisionTimeBreakdown(
            decisions=decisions,
            elapsed_seconds=elapsed,
            cpu_seconds=cpu,
            buckets=buckets,
        )
