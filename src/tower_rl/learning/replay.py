"""Bounded prioritized sequence replay.

Stores sequences rather than isolated transitions, because the learner
needs contiguous history with burn-in.  It holds encoded features and never game
state, so it knows nothing about The Tower: what it does know is which schema and
which profile a sequence came from, and it refuses to mix them.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS

#: R2D2's prioritized sequence replay (Kapturowski et al. 2019, ICLR; the values
#: of DeepMind's Acme reference `r2d2/config.py`: `priority_exponent`,
#: `importance_sampling_exponent`, `max_priority_weight`). R2D2 rather than
#: Ape-X or Schaul 2016 because what is stored here is a sequence, not a
#: transition. Fixed, not options: stacked-dqn always samples by them, and
#: DreamerV3 always samples uniformly (`uniform`), so neither can be silently
#: run without the replay its recipe names. Decided 2026-09-26 (board #85).
#:
#: Priority exponent alpha: a sequence is sampled with probability p ** alpha
#: over the buffer's total.
R2D2_PRIORITY_EXPONENT = 0.9
#: Importance-sampling exponent beta, held fixed as R2D2 and Ape-X hold it
#: rather than annealed to 1 as Schaul 2016 does: an anneal over the budget
#: moves the weighted loss on its own, which is what made the first run's loss
#: unreadable.
R2D2_IMPORTANCE_SAMPLING_EXPONENT = 0.6
#: Priority mix eta: a sequence's priority is eta * max |TD| + (1 - eta) *
#: mean |TD| over its steps, so one surprising step matters without a single
#: outlier dominating the whole sequence.
R2D2_PRIORITY_MIX = 0.9
#: The smallest priority a sequence can hold, so one the learner currently
#: fits exactly is still sampled now and then (the role of Schaul 2016's
#: epsilon in p = |delta| + epsilon; here a floor, a project choice).
PRIORITY_FLOOR = 1e-6


#: The layout of a saved buffer (`PrioritizedSequenceReplay.save_to`). A dump
#: in any other layout is refused rather than read as this one.
REPLAY_DUMP_FORMAT_VERSION = 1
#: The dump's one metadata file, beside its arrays.
REPLAY_DUMP_METADATA = "replay.json"


class ReplayRejected(ValueError):
    """A payload was refused; it is counted, never silently coerced."""


class ReplayDumpError(RuntimeError):
    """A saved buffer could not be written, read, or is not this buffer's to load."""


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
    #: The game time the transition spanned, in ms: 0 for a purchase. Required,
    #: because a default of 0 would silently mean "no discount" to a learner
    #: that discounts by game time, at every call site that forgot it.
    game_ms: float = field(kw_only=True)

    def __post_init__(self) -> None:
        if not 0 <= self.action_index < len(RUN_ACTIONS):
            raise ReplayRejected(f"action index {self.action_index} is outside the schema")
        if not math.isfinite(self.game_ms) or self.game_ms < 0.0:
            raise ReplayRejected(f"game time {self.game_ms} ms is not finite and non-negative")


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

    Nothing here is thread-safe on its own. One buffer is shared by a fleet of
    actors and one learner, and `lock` is what they share it under; see the
    field for why that discipline is the caller's rather than each method's.
    """

    capacity: int
    seed: int | None = None
    #: Sampling exponent. 0 is uniform; 1 is fully proportional to priority.
    alpha: float = R2D2_PRIORITY_EXPONENT
    #: Importance-sampling exponent: how much of the bias prioritization
    #: introduces the weights correct. Irrelevant under uniform sampling, where
    #: every weight is exactly one.
    beta: float = R2D2_IMPORTANCE_SAMPLING_EXPONENT

    #: Held by every caller that adds, samples or updates priorities, because
    #: several actors write into one buffer while the learner reads it. It is
    #: not taken inside the methods below: `update_priorities` refuses indices
    #: an eviction has shifted, so the learner must hold this across `sample`,
    #: `learn` and `update_priorities` together rather than around each of them,
    #: and a lock already held by the caller could not be taken again here. An
    #: actor holds it for the sequences of one episode.
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

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
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be within [0, 1]")
        self._random = random.Random(self.seed)

    @classmethod
    def uniform(cls, capacity: int, *, seed: int | None = None) -> PrioritizedSequenceReplay:
        """A buffer that samples every live sequence equally: DreamerV3's replay.

        The official DreamerV3 loop samples uniformly (docs/solution.md 9.4c).
        Priorities are still kept and updated, but at alpha 0 they are never
        read, and every importance-sampling weight is exactly one.
        """
        return cls(capacity=capacity, seed=seed, alpha=0.0, beta=0.0)

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
        # rarer than the pass over every priority `sample` makes every batch.
        self._priorities.append(max(self._priorities, default=1.0))
        self.stats.added += 1
        return True

    def sample(
        self, batch_size: int
    ) -> tuple[tuple[int, ...], tuple[ReplaySequence, ...], tuple[float, ...]]:
        """Sample sequences by priority with importance-sampling weights.

        Sequence i is drawn with probability P(i) = p_i ** alpha / sum_k p_k ** alpha,
        with replacement. Its importance-sampling weight is (N * P(i)) ** -beta
        divided by the largest weight in the batch, so the rarest sequence drawn
        weighs exactly one and none weighs more (Schaul 2016, section 3.4).

        The normaliser is the batch's, as in the R2D2 reference learners (Acme
        `r2d2/learning.py`, SEED RL), not the whole buffer's as in the baselines
        convention this buffer used while it only ever ran at alpha 0, where the
        choice made no difference. The buffer's largest weight
        belongs to its single lowest-priority sequence, so one sequence the
        learner fits almost exactly - down at `PRIORITY_FLOOR` - would shrink
        every weight of every batch by orders of magnitude, and with it the
        gradient that clipping and the optimizer see.
        """
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self._items:
            raise ReplayRejected("replay is empty")

        weights = [priority**self.alpha for priority in self._priorities]
        indices = tuple(
            self._random.choices(range(len(self._items)), weights=weights, k=batch_size)
        )
        # (N * P(i)) ** -beta over (N * P(rarest)) ** -beta: N and the total
        # cancel, leaving the ratio of the two sampling weights.
        rarest = min(weights[index] for index in indices)
        corrections = tuple((rarest / weights[index]) ** self.beta for index in indices)
        self.stats.sampled += batch_size
        self._evictions_at_sample = self.stats.evicted
        return indices, tuple(self._items[index] for index in indices), corrections

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
            priority = R2D2_PRIORITY_MIX * max(magnitudes) + (1.0 - R2D2_PRIORITY_MIX) * (
                sum(magnitudes) / len(magnitudes)
            )
            self._priorities[index] = max(priority, PRIORITY_FLOOR)

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

    def check_dump(self, metadata: Mapping[str, Any]) -> None:
        """Refuse a saved buffer this one could not continue from.

        The layout, the capacity and how it samples must all be this buffer's:
        sequences saved under R2D2's priorities and continued uniformly, or the
        other way round, would be a replay neither recipe names.
        """
        if metadata.get("format_version") != REPLAY_DUMP_FORMAT_VERSION:
            raise ReplayDumpError(
                f"replay dump format {metadata.get('format_version')} is not "
                f"{REPLAY_DUMP_FORMAT_VERSION}"
            )
        saved = (metadata.get("capacity"), metadata.get("alpha"), metadata.get("beta"))
        if saved != (self.capacity, self.alpha, self.beta):
            raise ReplayDumpError(
                f"replay dump holds capacity {saved[0]} sampled at alpha {saved[1]}, "
                f"beta {saved[2]}; this buffer is capacity {self.capacity} at alpha "
                f"{self.alpha}, beta {self.beta}"
            )

    def save_to(self, directory: Path, *, run: Mapping[str, Any]) -> int:
        """Write the whole buffer to `directory` atomically; return the bytes written.

        The caller holds `lock` and keeps whatever else it must still - the dump
        is a snapshot of one moment. `run` is the caller's account of that
        moment (its decision count and identity), stored beside the buffer's own
        so a resume can refuse a dump from any other one.

        Everything sampling depends on is saved: every stored sequence, its
        priority, the order they were inserted in - the buffer is a FIFO held
        oldest first, so that order is its cursor and says what is evicted next
        - the compatibility key and the counters. The sampling stream is not:
        a resumed buffer draws from its own seed.

        Plain `.npy` arrays and one JSON file, never a pickle. Overlapping
        windows share their steps in memory, so each distinct step is written
        once to a step table and sequences hold indices into it; that also
        keeps the reload from doubling the buffer's size. Each array is written
        row by row through a memory map, so no second copy of the buffer is
        made in memory. The files go to a sibling directory that is renamed
        into place only once all of them are on disk, so a dump that exists is
        a complete one.
        """
        steps: dict[int, int] = {}
        distinct_steps: list[ReplayStep] = []
        episodes: dict[int, int] = {}
        distinct_episodes: list[SequenceMetadata] = []
        step_index: list[int] = []
        sequence_episode: list[int] = []
        for sequence in self._items:
            # By object identity: the steps and the metadata are alive in the
            # buffer throughout, so an id cannot be reused while this runs.
            episode = episodes.setdefault(id(sequence.metadata), len(distinct_episodes))
            if episode == len(distinct_episodes):
                distinct_episodes.append(sequence.metadata)
            sequence_episode.append(episode)
            for step in sequence.steps:
                index = steps.setdefault(id(step), len(distinct_steps))
                if index == len(distinct_steps):
                    distinct_steps.append(step)
                step_index.append(index)

        temporary = directory.with_name(directory.name + ".partial")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            for name, (dtype, width, value) in _STEP_FIELDS.items():
                _write_rows(temporary / f"step_{name}.npy", dtype, width, distinct_steps, value)
            for name, (dtype, width, value) in _EPISODE_FIELDS.items():
                _write_rows(
                    temporary / f"episode_{name}.npy", dtype, width, distinct_episodes, value
                )
            sequences = {
                "steps": [len(sequence.steps) for sequence in self._items],
                "burn_in": [sequence.burn_in for sequence in self._items],
                "episode": sequence_episode,
                "step_index": step_index,
            }
            for name, values in sequences.items():
                numpy.save(temporary / f"sequence_{name}.npy", numpy.asarray(values, numpy.int64))
            numpy.save(
                temporary / "sequence_priority.npy",
                numpy.asarray(self._priorities, numpy.float64),
            )
            metadata = {
                "format_version": REPLAY_DUMP_FORMAT_VERSION,
                "capacity": self.capacity,
                "alpha": self.alpha,
                "beta": self.beta,
                "sequences": len(self._items),
                "steps": len(distinct_steps),
                "step_slots": len(step_index),
                "episodes": len(distinct_episodes),
                "compatibility": list(self._compatibility) if self._compatibility else None,
                "stats": asdict(self.stats),
                "run": dict(run),
            }
            (temporary / REPLAY_DUMP_METADATA).write_text(json.dumps(metadata, indent=2))
            size = 0
            for path in temporary.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                size += path.stat().st_size
            _replace_directory(temporary, directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return size

    def load_from(self, directory: Path) -> None:
        """Restore a buffer `save_to` wrote into this empty one.

        Checked by size and shape rather than by hash: the atomic rename already
        guarantees a dump that exists was written whole, and what is left - a
        file truncated or swapped by hand - changes a shape or a length, which
        is checked here, while hashing several gigabytes would cost as long
        again as reading them. The arrays are read through memory maps and
        turned back into sequences row by row, so the files are never held in
        memory beside the buffer they rebuild.
        """
        if self._items:
            raise ReplayDumpError("a dump is only loaded into an empty buffer")
        metadata = read_replay_metadata(directory)
        self.check_dump(metadata)
        arrays: dict[str, Any] = {}
        tables = (
            ("step", _STEP_FIELDS, metadata["steps"]),
            ("episode", _EPISODE_FIELDS, metadata["episodes"]),
        )
        for table, fields, rows in tables:
            for name, (_dtype, width, _value) in fields.items():
                shape = (rows,) if width is None else (rows, width)
                arrays[f"{table}_{name}"] = _read_array(directory, f"{table}_{name}", shape)
        sequences = metadata["sequences"]
        for name in ("steps", "burn_in", "episode", "priority"):
            arrays[f"sequence_{name}"] = _read_array(directory, f"sequence_{name}", (sequences,))
        step_index = _read_array(directory, "sequence_step_index", (metadata["step_slots"],))
        if int(arrays["sequence_steps"].sum()) != len(step_index):
            raise ReplayDumpError("replay dump's sequence lengths do not add up to its steps")
        if len(step_index) and not 0 <= int(step_index.min()) <= int(step_index.max()) < (
            metadata["steps"]
        ):
            raise ReplayDumpError("replay dump's sequences index steps it does not hold")
        if sequences > self.capacity:
            raise ReplayDumpError(f"replay dump holds {sequences} sequences, over capacity")

        episodes = [
            SequenceMetadata(
                **{
                    name: arrays[f"episode_{name}"][row].item()
                    for name in _EPISODE_FIELDS
                }
            )
            for row in range(metadata["episodes"])
        ]
        steps = [_step_from(arrays, row) for row in range(metadata["steps"])]
        items: deque[ReplaySequence] = deque()
        start = 0
        for row in range(sequences):
            end = start + int(arrays["sequence_steps"][row])
            items.append(
                ReplaySequence(
                    metadata=episodes[int(arrays["sequence_episode"][row])],
                    steps=tuple(steps[index] for index in step_index[start:end].tolist()),
                    burn_in=int(arrays["sequence_burn_in"][row]),
                )
            )
            start = end
        self._items = items
        self._priorities = deque(arrays["sequence_priority"].tolist())
        compatibility = metadata["compatibility"]
        self._compatibility = None if compatibility is None else tuple(compatibility)
        self.stats = ReplayStats(**metadata["stats"])
        # No batch is outstanding: the next update follows the next sample.
        self._evictions_at_sample = self.stats.evicted


def read_replay_metadata(directory: Path) -> dict[str, Any]:
    """The metadata of a saved buffer, read without touching its arrays."""
    try:
        metadata: dict[str, Any] = json.loads((directory / REPLAY_DUMP_METADATA).read_text())
    except (OSError, ValueError) as error:
        raise ReplayDumpError(f"cannot read replay dump {directory}: {error}") from error
    return metadata


#: One array per field of a stored step, and how to read that field off one:
#: its dtype and its row width (None for a scalar). float64 because that is
#: what the step holds, so a restored step equals the one saved.
_STEP_FIELDS: dict[str, tuple[type, int | None, Callable[[Any], object]]] = {
    "scalars": (numpy.float64, SCALAR_COUNT, lambda step: step.features.scalars),
    "rows": (numpy.float64, ROW_COUNT * ROW_WIDTH, lambda step: step.features.rows),
    "mask": (numpy.bool_, len(RUN_ACTIONS), lambda step: step.features.mask),
    "action_index": (numpy.int64, None, lambda step: step.action_index),
    "reward": (numpy.float64, None, lambda step: step.reward),
    "done": (numpy.bool_, None, lambda step: step.done),
    "admissible": (numpy.bool_, None, lambda step: step.admissible),
    "padding": (numpy.bool_, None, lambda step: step.padding),
    "game_ms": (numpy.float64, None, lambda step: step.game_ms),
}

#: The same for the metadata of an episode, which all of its sequences share.
#: Strings are fixed-width unicode arrays, which `.npy` holds without a pickle.
_EPISODE_FIELDS: dict[str, tuple[type, int | None, Callable[[Any], object]]] = {
    "episode_id": (numpy.str_, None, lambda meta: meta.episode_id),
    "actor_id": (numpy.str_, None, lambda meta: meta.actor_id),
    "profile_id": (numpy.str_, None, lambda meta: meta.profile_id),
    "observation_schema": (numpy.str_, None, lambda meta: meta.observation_schema),
    "action_schema": (numpy.str_, None, lambda meta: meta.action_schema),
    "reward_schema": (numpy.str_, None, lambda meta: meta.reward_schema),
    "model_version": (numpy.int64, None, lambda meta: meta.model_version),
    "epsilon": (numpy.float64, None, lambda meta: meta.epsilon),
    "game_speed": (numpy.float64, None, lambda meta: meta.game_speed),
}


def _write_rows(
    path: Path,
    dtype: type,
    width: int | None,
    rows: list[Any],
    value: Callable[[Any], object],
) -> None:
    """One field of a table, written row by row into a memory-mapped `.npy`."""
    if dtype is numpy.str_:
        # Episodes are few, and a string array's width is its longest value.
        numpy.save(path, numpy.asarray([value(row) for row in rows], numpy.str_))
        return
    shape = (len(rows),) if width is None else (len(rows), width)
    array = numpy.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    for index, row in enumerate(rows):
        array[index] = value(row)
    array.flush()
    del array


def _read_array(directory: Path, name: str, shape: tuple[int, ...]) -> Any:
    """One array of a dump, memory-mapped and checked against the shape it must have."""
    try:
        array = numpy.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ReplayDumpError(f"replay dump array {name} is unreadable: {error}") from error
    if array.shape != shape:
        raise ReplayDumpError(
            f"replay dump array {name} has shape {array.shape}, expected {shape}"
        )
    return array


def _step_from(arrays: Mapping[str, Any], row: int) -> ReplayStep:
    return ReplayStep(
        features=StateFeatures(
            scalars=tuple(arrays["step_scalars"][row].tolist()),
            rows=tuple(arrays["step_rows"][row].tolist()),
            mask=tuple(arrays["step_mask"][row].tolist()),
        ),
        action_index=int(arrays["step_action_index"][row]),
        reward=float(arrays["step_reward"][row]),
        done=bool(arrays["step_done"][row]),
        admissible=bool(arrays["step_admissible"][row]),
        padding=bool(arrays["step_padding"][row]),
        game_ms=float(arrays["step_game_ms"][row]),
    )


def _replace_directory(source: Path, target: Path) -> None:
    """Rename `source` over `target`; a directory cannot simply be replaced."""
    previous = target.with_name(target.name + ".previous")
    shutil.rmtree(previous, ignore_errors=True)
    if target.exists():
        os.replace(target, previous)
    os.replace(source, target)
    shutil.rmtree(previous, ignore_errors=True)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
