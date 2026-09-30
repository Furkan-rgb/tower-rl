"""The learner: the one training copy of the network, and the thread that trains it.

Gradient steps are taken on one thread of their own, beside the actors, never
on an actor's thread (ADR 0017). Before, the actor whose episode ended took the
steps that episode earned, under the run's progress lock: its emulator idled
for them and every other actor finishing an episode queued behind them
(`M3-P015`/`M3-P016`: learn 24-32 ms and blocked 0-110 ms per decision beside
a 157-230 ms bridge round trip).

**The debt.** The replay ratio is held by a debt counted in gradient steps.
Each unit of data collected - a credit - earns the learner `steps_per_credit`
steps; the learner takes a step whenever one whole step is owed; and an actor
that finds more than `bound_credits` credits' worth owed pauses before its
next decision until the learner has brought the debt back under the bound.
So over any span the steps taken are the configured ratio of the data
collected, short by at most the bound (plus one credit per actor, each of
which credits before it checks), and the parameters an actor acts from are at
most that far behind the data it is collecting. Under stacked-dqn and
DreamerV3 a credit is a decision, credited as it is taken; under R2D2 it is a
replay item, credited as its episode ends (`TrainingRun`).

**Holding the learner still.** `held` lets no step begin and waits out the one
in flight, so whatever reads the training network meanwhile - a checkpoint, a
periodic evaluation, the run's last resume point - reads one completed step.

**The GPU.** On CUDA the learner issues its work on a stream of its own
(`Learner.stream`), while actors act on the default stream. PyTorch's side
streams do not synchronise with the default stream, so a forward pass is not
queued behind a gradient step's kernels; the two share the device's time. The
price is that nothing orders the two streams for us, so `Learner` does it
where they meet. A step ends by cloning the network into a new snapshot and
synchronising its stream before it publishes it, so the snapshot an actor
copies from is finished; and a copy synchronises the actor's stream before it
lets the snapshot go, so its memory is not reused while still being read.
What reads the network under `held` - a save, an evaluation - runs on the
default stream and waits on the host for what it read (a save copies to host
memory, an evaluation reads each action back), so the step after the hold
cannot overtake it either.

**Publication.** Parameters reach the actors as the official agent's do
(`embodied/jax/agent.py` 243-247, 279-282): every completed step publishes a
snapshot of the network with a number, and an actor whose copy is older loads
the latest before a decision: before every one under DreamerV3, on the refresh
cadence under stacked-dqn and R2D2 (`TrainingConfig.parameter_sync_decisions`). The
actor's carried state is its own and survives the load. A snapshot is never
written once published - each step makes a new one - so `Learner.lock` guards
only the moment a snapshot is handed over or loaded, never a step: an actor
never waits out a gradient step to refresh.

**Locks, in the one order they are ever taken:** the run's progress lock
(`TrainingRun._lock`), then `LearnerThread.held`, then the replay buffer's
lock, then `Learner.lock`. The debt's condition is a leaf: nothing else is
taken while it is held. The learner thread takes the replay lock and
`Learner.lock` one at a time and never the progress lock, which is why a hook
running under the progress lock can hold it still without deadlock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import torch

from tower_rl.environment.decision_time import BLOCKED, DecisionTimeProfile
from tower_rl.learning.backbone import Backbone, LearnMetrics, SequenceBatch

#: How often an actor paused on the bound looks at the run's stop again. The
#: stop is an event a signal handler sets, and a signal handler must not take
#: the condition's lock to notify it - the main thread may hold that lock at
#: the moment the signal lands - so a paused actor polls for it instead.
STOP_POLL_SECONDS = 0.1

#: Per-step metrics kept for the run to read, newest last: the run's own
#: windows are a hundred steps long, so older ones would be discarded unread.
RECENT_METRICS = 100


@dataclass
class Learner:
    """The one training copy of the network, and how its parameters reach actors.

    No actor acts through this. Each acts from its own copy (`acting_copy`), so
    a forward pass contends with neither the learner nor another actor. What
    reaches a copy is the snapshot the last completed step published
    (`published`), so a copy is always the parameters of some completed step
    and never half of one, and taking it never waits for the step in flight.
    """

    backbone: Backbone
    #: Guards `_snapshot` and `published`: held to hand a new snapshot over and
    #: to load one into a copy, never across a step.
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: The CUDA stream gradient steps are issued on, or None off the GPU.
    stream: torch.cuda.Stream | None = field(default=None, init=False)
    #: The number of the latest snapshot; it rises by one per publication. An
    #: acting copy that last loaded an older number is behind.
    published: int = field(default=0, init=False)
    #: The backbone's state as of the last publication, never written after.
    _snapshot: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        device = self.backbone.device
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]

    @contextmanager
    def issuing(self) -> Iterator[None]:
        """Issue the calling thread's GPU work on the learner's stream.

        The current stream is per thread in PyTorch, so this moves the learner
        thread alone: an actor acting meanwhile stays on the default stream.
        Whatever the default stream still has queued when the thread starts -
        a resume's `load_state_dict`, say - is ordered before its first step.
        """
        if self.stream is None:
            yield
            return
        self.stream.wait_stream(torch.cuda.default_stream(self.backbone.device))
        with torch.cuda.stream(self.stream):
            yield

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        """Take one gradient step and publish the parameters it left."""
        metrics = self.backbone.learn(batch)
        self.publish()
        return metrics

    def publish(self) -> None:
        """Publish the backbone as it stands as the next snapshot.

        Called at the end of every step, on the learner thread, and by the run
        as a collection block starts, with the learner not running - which is
        what picks up a checkpoint loaded after the run was built. The clone is
        finished before it is handed over, so a copy taken from it on another
        stream reads a whole step. It is the network state alone
        (`Backbone.network_state_dict`): an acting copy never reads the
        optimizer's moments, so none are cloned or held for it.
        """
        snapshot = _cloned(self.backbone.network_state_dict())
        if self.stream is not None:
            torch.cuda.current_stream(self.backbone.device).synchronize()
        with self.lock:
            self._snapshot = snapshot
            self.published += 1

    def publish_to(self, acting: Backbone) -> int:
        """Load the latest snapshot into one actor's acting copy; return its number.

        Called on that actor's own thread between two of its decisions, so no
        forward pass can see the copy half written, and the snapshot is never
        written after its publication, so what is loaded is one completed step.
        Every decision is therefore taken on one complete version. The lock is
        held against a publication only, which is a hand-over of a finished
        clone: an actor never waits for a step here.
        """
        with self.lock:
            if self._snapshot is None:
                raise RuntimeError("nothing has been published yet")
            acting.load_network_state_dict(self._snapshot)
            if self.stream is not None:
                # The load ran on this thread's stream; finish it before the
                # snapshot may be let go and its memory reused.
                torch.cuda.current_stream(self.backbone.device).synchronize()
            return self.published


def _cloned(state: Any) -> Any:
    """A state dict with every tensor cloned, so later steps cannot change it."""
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        return {key: _cloned(value) for key, value in state.items()}
    if isinstance(state, list | tuple):
        return type(state)(_cloned(value) for value in state)
    return state


@dataclass(frozen=True)
class LearnerLoad:
    """What the learner thread has done, cumulatively, read at one moment."""

    gradient_steps: int
    #: Wall time inside gradient steps, and wall time the thread was running.
    stepping_seconds: float
    running_seconds: float
    #: Gradient steps owed when read, and the most that may be owed before
    #: actors pause. Readings, not counters: `since` keeps the later one.
    owed_steps: float
    bound_steps: float
    #: Actor-thread seconds spent paused on the bound, summed over the fleet.
    paused_actor_seconds: float

    @property
    def utilization(self) -> float:
        """The share of its running time the learner spent stepping."""
        if self.running_seconds <= 0:
            return 0.0
        return self.stepping_seconds / self.running_seconds

    def since(self, earlier: LearnerLoad) -> LearnerLoad:
        """The span between an earlier reading and this one."""
        return replace(
            self,
            gradient_steps=self.gradient_steps - earlier.gradient_steps,
            stepping_seconds=self.stepping_seconds - earlier.stepping_seconds,
            running_seconds=self.running_seconds - earlier.running_seconds,
            paused_actor_seconds=self.paused_actor_seconds - earlier.paused_actor_seconds,
        )

    def as_record(self) -> dict[str, float]:
        return {
            "gradient_steps": self.gradient_steps,
            "stepping_seconds": round(self.stepping_seconds, 4),
            "running_seconds": round(self.running_seconds, 4),
            "utilization": round(self.utilization, 4),
            "owed_steps": round(self.owed_steps, 3),
            "bound_steps": round(self.bound_steps, 3),
            "paused_actor_seconds": round(self.paused_actor_seconds, 4),
        }


IDLE_LOAD = LearnerLoad(
    gradient_steps=0,
    stepping_seconds=0.0,
    running_seconds=0.0,
    owed_steps=0.0,
    bound_steps=0.0,
    paused_actor_seconds=0.0,
)


class _Ending(Enum):
    """How a collection block asked the learner thread to end."""

    #: Take every whole step still owed, then end: the orderly end of a block.
    DRAIN = "drain"
    #: End after the step in flight: an exception is on its way out.
    ABORT = "abort"


class LearnerThread:
    """The thread that takes the gradient steps the fleet's collection earns.

    One per run, started for each collection block and ended with it; its
    counters and its debt outlive the block. `step` takes one gradient step and
    returns what it found; it is called only on this thread, one at a time.
    """

    def __init__(
        self,
        learner: Learner,
        step: Callable[[], LearnMetrics],
        *,
        steps_per_credit: float,
        bound_credits: int,
        carried_debt_steps: float = 0.0,
    ) -> None:
        self._learner = learner
        self._step = step
        self._steps_per_credit = steps_per_credit
        # Never under one step: the learner steps only on a whole one owed, so
        # a bound below it would pause actors on a debt nothing can pay.
        self.bound_steps = max(bound_credits * steps_per_credit, 1.0)
        #: Guards everything below. A leaf: nothing else is taken under it.
        self._condition = threading.Condition()
        #: Credits, net of any taken back, and steps taken. The debt is derived
        #: from the two rather than kept as a running float, so the steps over
        #: any span are the ratio of its credits exactly, with no rounding
        #: carried from one credit to the next.
        self._credits = 0
        #: Of those, the credits of episodes still being played: credited as
        #: their decisions were taken, not yet counted by the run (`settle`).
        self._unsettled = 0
        #: Steps already owed when the thread was made: a resumed run's parent's
        #: debt (`counted_debt_steps`), which the counters above start without.
        self._carried_debt_steps = carried_debt_steps
        self._holds = 0
        self._stepping = False
        self._ending: _Ending | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._failure: Exception | None = None
        #: Steps taken, and the metrics of the latest, not yet read by the run.
        self._unread_steps = 0
        self._recent: deque[LearnMetrics] = deque(maxlen=RECENT_METRICS)
        self._steps = 0
        self._stepping_seconds = 0.0
        self._running_seconds = 0.0
        self._started_at = 0.0
        self._paused_actor_seconds = 0.0

    @property
    def failure(self) -> Exception | None:
        """What ended the thread, if a step raised."""
        return self._failure

    def start(self) -> None:
        """Start serving the debt, for one collection block."""
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("the learner thread is already running")
            self._ending = None
            self._failure = None
            self._running = True
            self._started_at = time.perf_counter()
            self._thread = threading.Thread(target=self._serve, name="learner")
        self._thread.start()

    def finish(self, *, drain: bool) -> None:
        """End the thread and wait for it: draining the debt, or after the step in flight.

        An abort overrides a drain already asked for, which is how a second
        SIGINT arriving during the drain cuts it short.
        """
        thread = self._thread
        if thread is None:
            return
        with self._condition:
            if not drain or self._ending is None:
                self._ending = _Ending.DRAIN if drain else _Ending.ABORT
            self._condition.notify_all()
        thread.join()
        self._thread = None

    def credit(self, credits: int) -> None:
        """Owe the learner what `credits` earn, as an episode's decisions are taken."""
        with self._condition:
            self._credits += credits
            self._unsettled += credits
            if self._owed() >= 1.0:
                self._condition.notify_all()

    def settle(self, credited: int, counted: int) -> None:
        """Square an episode's credit with what the run counted of it.

        `credited` is what its decisions credited as they were taken; `counted`
        is what the run now owes the learner for it - its decisions, or its
        replay items under R2D2, or none for an episode that never counted
        (abandoned, failed) or was counted before the buffer was warm.
        """
        with self._condition:
            self._unsettled -= credited
            self._credits += counted - credited
            if self._owed() >= 1.0:
                self._condition.notify_all()

    def counted_debt_steps(self) -> float:
        """The steps owed on the episodes the run has counted, and not on those in flight.

        What a checkpoint records and a resume carries, so the steps of a run
        resumed with its buffer are the steps of one that never stopped: the
        episodes in flight are lost by a stop, and their credit with them. Read
        with the learner held still, or the steps taken move under it.
        """
        with self._condition:
            settled = self._credits - self._unsettled
            return self._carried_debt_steps + settled * self._steps_per_credit - self._steps

    def make_room(self, profile: DecisionTimeProfile, abandoned: Callable[[], bool]) -> None:
        """Pause the calling actor while more than the bound is owed.

        Returns once the learner has brought the debt back under the bound, or
        early - leaving the caller to find out why - once `abandoned` says so or
        the learner has stopped serving. The wait is charged to the actor's
        `blocked` bucket and to the fleet's paused seconds.
        """
        with self._condition:
            if self._owed() <= self.bound_steps:
                return
        with profile.span(BLOCKED):
            started = time.perf_counter()
            with self._condition:
                while (
                    self._owed() > self.bound_steps
                    and self._running
                    and self._ending is not _Ending.ABORT
                    and not abandoned()
                ):
                    self._condition.wait(STOP_POLL_SECONDS)
                self._paused_actor_seconds += time.perf_counter() - started

    @contextmanager
    def held(self) -> Iterator[None]:
        """Hold the learner still: no step begins, and the one in flight is waited out."""
        with self._condition:
            self._holds += 1
            while self._stepping:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._holds -= 1
                self._condition.notify_all()

    def take_metrics(self) -> tuple[int, list[LearnMetrics]]:
        """The steps taken since the last call, and the latest of their metrics."""
        with self._condition:
            steps, recent = self._unread_steps, list(self._recent)
            self._unread_steps = 0
            self._recent.clear()
        return steps, recent

    def load(self) -> LearnerLoad:
        with self._condition:
            running = self._running_seconds
            if self._running:
                running += time.perf_counter() - self._started_at
            return LearnerLoad(
                gradient_steps=self._steps,
                stepping_seconds=self._stepping_seconds,
                running_seconds=running,
                owed_steps=self._owed(),
                bound_steps=self.bound_steps,
                paused_actor_seconds=self._paused_actor_seconds,
            )

    def _owed(self) -> float:
        """Gradient steps owed now; the caller holds the condition."""
        return (
            self._carried_debt_steps + self._credits * self._steps_per_credit - self._steps
        )

    def _serve(self) -> None:
        try:
            with self._learner.issuing():
                while self._next_step():
                    self._take_step()
        except Exception as failure:  # noqa: BLE001 - handed to the run, which raises it
            with self._condition:
                self._failure = failure
        finally:
            with self._condition:
                self._running = False
                self._running_seconds += time.perf_counter() - self._started_at
                self._condition.notify_all()

    def _next_step(self) -> bool:
        """Wait until a step is owed and allowed; False once the thread should end."""
        with self._condition:
            while True:
                if self._ending is _Ending.ABORT:
                    return False
                if self._owed() >= 1.0 and not self._holds:
                    self._stepping = True
                    return True
                if self._ending is _Ending.DRAIN and self._owed() < 1.0:
                    return False
                self._condition.wait()

    def _take_step(self) -> None:
        began = time.perf_counter()
        metrics: LearnMetrics | None = None
        try:
            metrics = self._step()
        finally:
            with self._condition:
                self._stepping = False
                self._stepping_seconds += time.perf_counter() - began
                if metrics is not None:
                    self._steps += 1
                    self._unread_steps += 1
                    self._recent.append(metrics)
                self._condition.notify_all()
