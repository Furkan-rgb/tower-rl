"""Bounded prioritized sequence replay.

Stores sequences rather than isolated transitions, because a recurrent learner
needs contiguous history with burn-in.  It holds encoded features and never game
state, so it knows nothing about The Tower: what it does know is which schema and
which profile a sequence came from, and it refuses to mix them.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from tower_rl.domain.features import StateFeatures
from tower_rl.domain.run_actions import RUN_ACTIONS

DEFAULT_PRIORITY_EXPONENT = 0.9
"""R2D2 mixes the maximum and mean absolute TD error of a sequence, so one
surprising step matters without a single outlier dominating the whole sequence."""


class ReplayRejected(ValueError):
    """A payload was refused; it is counted, never silently coerced."""


@dataclass(frozen=True)
class SequenceMetadata:
    """Everything needed to decide whether a sequence may be learned from."""

    episode_id: str
    actor_id: str
    profile_id: str
    observation_schema: str
    action_schema: str
    reward_schema: str
    model_version: int
    epsilon: float
    game_speed: float

    def compatibility_key(self) -> tuple[str, str, str, str]:
        return (
            self.profile_id,
            self.observation_schema,
            self.action_schema,
            self.reward_schema,
        )


@dataclass(frozen=True)
class ReplayStep:
    """One stored decision. The mask is persisted from the first logged episode."""

    features: StateFeatures
    action_index: int
    reward: float
    done: bool
    admissible: bool
    #: Filler that carries no experience. An episode shorter than one window is
    #: padded up to a full window rather than dropped, which is the only way its
    #: terminal step can reach replay at all. A padded step is never a training
    #: target and never contributes a TD error; see `Actor._emit`.
    padding: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.action_index < len(RUN_ACTIONS):
            raise ReplayRejected(f"action index {self.action_index} is outside the schema")


@dataclass(frozen=True)
class ReplaySequence:
    """A contiguous slice of one episode, with its burn-in prefix."""

    metadata: SequenceMetadata
    steps: tuple[ReplayStep, ...]
    burn_in: int

    def __post_init__(self) -> None:
        if not self.steps:
            raise ReplayRejected("a sequence must contain at least one step")
        if not 0 <= self.burn_in < len(self.steps):
            raise ReplayRejected("burn-in must leave at least one learning step")
        if all(step.padding for step in self.steps[self.burn_in :]):
            raise ReplayRejected("padding alone is not a learning window")

    @property
    def learn_length(self) -> int:
        return len(self.steps) - self.burn_in


@dataclass
class ReplayStats:
    added: int = 0
    rejected: int = 0
    evicted: int = 0
    sampled: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1


@dataclass
class PrioritizedSequenceReplay:
    """An in-memory ring buffer sampled by priority.

    Capacity is counted in sequences and bounded, so a long run cannot grow
    without limit. Compatibility is fixed by the first accepted sequence: mixing
    profiles or schema versions would silently train one model on two different
    environments.
    """

    capacity: int
    priority_exponent: float = DEFAULT_PRIORITY_EXPONENT
    #: Sampling exponent. 0 is uniform; 1 is fully proportional to priority.
    alpha: float = 0.6
    seed: int | None = None

    _items: deque[ReplaySequence] = field(default_factory=deque, init=False)
    _priorities: deque[float] = field(default_factory=deque, init=False)
    _compatibility: tuple[str, str, str, str] | None = field(default=None, init=False)
    #: Evictions seen when the last batch was sampled, so a stale index cannot be
    #: mistaken for a live one.
    _evictions_at_sample: int = field(default=0, init=False)
    _random: random.Random = field(init=False)
    stats: ReplayStats = field(default_factory=ReplayStats, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("replay capacity must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        self._random = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def compatibility(self) -> tuple[str, str, str, str] | None:
        return self._compatibility

    def add(self, sequence: ReplaySequence) -> bool:
        """Accept one sequence, or reject and count it. Never coerces."""
        key = sequence.metadata.compatibility_key()
        if self._compatibility is None:
            self._compatibility = key
        elif key != self._compatibility:
            self.stats.reject("incompatible_profile_or_schema")
            return False
        if any(not step.admissible for step in sequence.steps):
            self.stats.reject("inadmissible_transition")
            return False

        if len(self._items) == self.capacity:
            self._items.popleft()
            self._priorities.popleft()
            self.stats.evicted += 1
        self._items.append(sequence)
        # A new sequence enters at the highest priority the buffer currently
        # holds, so it is sampled at least once before its real TD error is
        # known. Schaul 2016 says *current* maximum, not highest ever seen: one
        # large early error would otherwise pin insertion priority forever and
        # sampling would degenerate towards recency. The scan is over live
        # sequences only and happens once per stored sequence, which is far
        # rarer than the identical scan `sample` already does every batch.
        self._priorities.append(max(self._priorities, default=1.0))
        self.stats.added += 1
        return True

    def sample(
        self, batch_size: int, *, beta: float = 0.4
    ) -> tuple[tuple[int, ...], tuple[ReplaySequence, ...], tuple[float, ...]]:
        """Sample sequences by priority with importance-sampling weights."""
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self._items:
            raise ReplayRejected("replay is empty")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be within [0, 1]")

        weights = [priority**self.alpha for priority in self._priorities]
        total = sum(weights)
        indices = tuple(
            self._random.choices(range(len(self._items)), weights=weights, k=batch_size)
        )
        smallest = min(weights) / total
        # Importance sampling corrects the bias that prioritization introduces,
        # normalized by the largest correction so weights never exceed one.
        corrections = []
        for index in indices:
            probability = weights[index] / total
            corrections.append((smallest / probability) ** beta)
        self.stats.sampled += batch_size
        self._evictions_at_sample = self.stats.evicted
        return indices, tuple(self._items[index] for index in indices), tuple(corrections)

    def update_priorities(
        self, indices: tuple[int, ...], td_errors: tuple[tuple[float, ...], ...]
    ) -> None:
        """Fold learner feedback back into sampling priorities.

        An index is a position in the buffer, and eviction shifts every position
        down by one. So an update must reach the buffer before anything is added
        to a full buffer; otherwise it would land on a different sequence, which
        is worse than not landing at all. The training loop samples, learns and
        updates without collecting in between, and this states that rather than
        assuming it.
        """
        if len(indices) != len(td_errors):
            raise ValueError("each index needs its own sequence of TD errors")
        if self.stats.evicted != self._evictions_at_sample:
            raise ReplayRejected(
                "eviction has shifted every index since these were sampled; "
                "priorities must be updated before more sequences are added"
            )
        for index, errors in zip(indices, td_errors, strict=True):
            if not errors:
                raise ValueError("a priority update needs at least one TD error")
            if not 0 <= index < len(self._priorities):
                # An index the buffer never held. Dropping it is correct; the
                # dangerous case, an index that still lands but on the wrong
                # sequence, is refused above rather than tolerated here.
                continue
            magnitudes = [abs(error) for error in errors]
            priority = (
                self.priority_exponent * max(magnitudes)
                + (1.0 - self.priority_exponent) * (sum(magnitudes) / len(magnitudes))
            )
            priority = max(priority, 1e-6)
            self._priorities[index] = priority

    def snapshot(self) -> dict[str, object]:
        """Metadata a checkpoint needs to state what replay it resumed with."""
        return {
            "sequences": len(self._items),
            "capacity": self.capacity,
            "compatibility": list(self._compatibility) if self._compatibility else None,
            "added": self.stats.added,
            "rejected": self.stats.rejected,
            "evicted": self.stats.evicted,
            "sampled": self.stats.sampled,
            "rejections_by_reason": dict(self.stats.rejections_by_reason),
        }
