"""R2D2's replay: fixed-length items over stored steps, sampled by priority.

A port of the replay R2D2 is published with (Kapturowski et al. 2019, section
2.3, section 3, Table 2), where the paper is silent as Acme builds it
(google-deepmind/acme at 4949d3ce: `agents/jax/r2d2/{builder,config,learning}.py`
and the `StructuredAdder`, `adders/reverb/structured.py`). What it follows:

- **Items.** An item is `R2D2_ITEM_LENGTH` = 40 burn-in + 80 trace + 1
  bootstrap = 121 consecutive steps of one episode (builder.py 121-122), and
  never crosses an episode boundary.
- **Where items start.** On the grid 0, 40, 80, ... of the episode's steps
  (`sequence_period` 40, `create_sequence_config` 278-285). At the episode's
  end Acme's TRUNCATE rule adds the grid-aligned remainder, fewer than 121
  steps, and an episode shorter than 121 steps is one item of all its steps
  (structured.py 303-358). A short item is right-padded with zero steps when
  sampled (builder.py `_zero_pad`, 63-94). `item_layout` is that rule. So an
  episode's first 40 steps are burn-in only, and a short episode's terminal
  step is still in replay.
- **Stored state.** Each item carries the actor's LSTM state (h, c) from
  before its first step, the only state kept (`_build_sequence`, 53-60); the
  zero state for an item at an episode's start.
- **Priorities.** A new item enters at priority 1.0 (`StructuredAdder`,
  structured.py 58). An item is drawn with probability p^alpha / sum p^alpha,
  with replacement (Reverb `Prioritized(0.9)`), and weighted by (1 / (P +
  1e-6))^beta over the batch's largest (learning.py 147-151). The learner's
  priority is eta * max |delta| + (1 - eta) * mean |delta| over the item's
  trace (learning.py 153-157). Priorities are kept as given, with no floor.
- **Capacity.** `max_replay_size` 100,000 items, oldest removed first
  (`selectors.Fifo`). An episode's arrays are freed when its last item goes.
- **Storage.** Each step is stored once, in its episode's arrays, and read by
  every item that overlaps it, as Reverb shares chunks between items.

What differs, each forced:

- **Whole-episode insertion.** Acme inserts items as an episode streams in;
  here an episode's items are inserted when it ends, so a counted episode is
  in replay whole or not at all (ADR 0014's resume pair, ADR 0017's debt), as
  in `dreamer_replay.py`. The actor cuts the stream at its first inadmissible
  transition (docs/environment-contract.md) before it gets here.
- **Pad steps are excluded from priorities**, as the learner excludes them
  from the loss: a pad step's action mask is empty, so its Q is undefined.
  An item with no valid trace step - an episode of 41 steps or fewer - takes
  priority 0 at its first update and is not drawn again.
- **Minimum size, samples per insert.** See `R2D2_MIN_REPLAY_ITEMS` and
  `R2D2_SAMPLES_PER_INSERT`: both forced by the protocol's budget.
- **The rate limiter** is ADR 0017's learner debt, credited per item
  inserted (`R2D2_LEARNER_STEPS_PER_ITEM`, `R2D2_LEARNER_DEBT_BOUND_ITEMS`).
  An episode's items are inserted, and credited, as it ends.

A step is in the shared layout (`StepArrays`): its observation, the action
taken at it, and the reward, termination and game time of the transition into
it. That is Acme's observation-action-reward input as it stands, the previous
action aside, which a sample carries as `previous_action`.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH
from tower_rl.learning.backbone import SequenceBatch
from tower_rl.learning.replay import (
    R2D2_IMPORTANCE_SAMPLING_EXPONENT,
    R2D2_PRIORITY_EXPONENT,
    REPLAY_DUMP_METADATA,
    ReplayDumpError,
    ReplayRejected,
    ReplayStats,
    SequenceMetadata,
    read_replay_metadata,
)
from tower_rl.learning.step_arrays import StepArrays, read_rows, write_rows

#: P Table 2, Acme `burn_in_length`.
R2D2_BURN_IN = 40
#: P Table 2, Acme `trace_length`.
R2D2_TRACE_LENGTH = 80
#: Burn-in, trace and the one step the last trace step bootstraps from
#: (builder.py 121-122).
R2D2_ITEM_LENGTH = R2D2_BURN_IN + R2D2_TRACE_LENGTH + 1
#: P section 2.3 (adjacent items overlap by 40), Acme `sequence_period`.
R2D2_SEQUENCE_PERIOD = 40
#: P Table 2 (4e6 observations in items of 80), Acme `max_replay_size`.
R2D2_REPLAY_CAPACITY = 100_000
#: P Appendix, Acme `hk.LSTM(512)`: each of h and c.
R2D2_STATE_SIZE = 512
#: StructuredAdder's priority for every new item (structured.py 58).
R2D2_INITIAL_PRIORITY = 1.0
#: P Table 2, Acme `batch_size`.
R2D2_BATCH_SIZE = 64
#: Acme `min_replay_size` is 50,000 items, about 2,000,000 decisions at one
#: item per 40: twice the protocol's 1,000,000-decision budget (solution
#: 9.2b), so learning would never start. Forced lower, to 1,250 items: 50,000
#: decisions, Acme's number read in transitions.
R2D2_MIN_REPLAY_ITEMS = 1_250
#: Acme's `samples_per_insert` is 4: about 1,563 learner steps in 1,000,000
#: decisions, under one 2,500-step target period. Forced up by the budget to
#: 320, about 125,000 steps and 50 target copies.
R2D2_SAMPLES_PER_INSERT = 320
#: Acme's rate limiter (`SampleToInsertRatio`) held as ADR 0017's learner debt,
#: credited per item actually inserted, as Reverb counts inserts: 320 samples
#: per insert / 64 samples per learner step = 5 learner steps per item.
R2D2_LEARNER_STEPS_PER_ITEM = R2D2_SAMPLES_PER_INSERT / R2D2_BATCH_SIZE
#: The debt bound, in items: Acme's `error_buffer`, `min_replay_size x
#: samples_per_insert x samples_per_insert_tolerance_rate` = 1,250 x 320 x 0.1
#: = 40,000 samples = 625 learner steps = 125 items. It meets ADR 0017's rule
#: for the bound - cover the longest ordinary hold, a resume-point save of
#: 11.5 s - with room: an episode's items are credited at once as it ends
#: (about 12 items, 60 steps, from a 521-decision episode), so the worst hold
#: is every actor ending an episode inside it, 8 x 60 = 480 steps, under 625.
#: Its lag, 625 steps, is a quarter of one 2,500-step target period. Reverb
#: also offsets the ratio by `min_replay_size x samples_per_insert`, so the
#: first 1,250 items earn no steps; here the items before the buffer is warm
#: earn none and the warming episode's earn theirs (ADR 0017): a difference
#: of one episode's items.
R2D2_LEARNER_DEBT_BOUND_ITEMS = 125
#: The layout of a saved R2D2 replay. 1 and 2 were the removed stacked-dqn's
#: window buffer and 3 is DreamerV3's step replay; all are refused, since no
#: earlier buffer holds items with stored states.
R2D2_REPLAY_DUMP_FORMAT_VERSION = 4


def item_layout(step_count: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The (start offsets, unpadded lengths) of the items of an episode of `step_count` steps.

    Acme's TRUNCATE configs (structured.py 278-358), stated by where items
    start: an item at every grid position with at least `length - period + 1`
    steps from it - the full items and the grid-aligned remainder - or, for an
    episode shorter than one item, one item of the whole episode.
    """
    if step_count < R2D2_ITEM_LENGTH:
        return numpy.zeros(1, numpy.int64), numpy.full(1, step_count, numpy.int64)
    last_start = step_count - (R2D2_ITEM_LENGTH - R2D2_SEQUENCE_PERIOD + 1)
    starts = numpy.arange(0, last_start + 1, R2D2_SEQUENCE_PERIOD, dtype=numpy.int64)
    return starts, numpy.minimum(R2D2_ITEM_LENGTH, step_count - starts)


@dataclass
class _Episode:
    number: int
    metadata: SequenceMetadata
    steps: StepArrays
    #: The (h, c) each of its items starts from, in item order. float32
    #: [items, 2, state].
    states: numpy.ndarray
    #: Its items still in replay.
    live_items: int


@dataclass(frozen=True)
class R2D2Sample:
    """One sampled batch: its arrays, and the items it was drawn from, for their priorities."""

    #: The items' serial numbers, `R2D2Replay.update_priorities`'s keys.
    keys: numpy.ndarray  # int64 [batch]
    #: Each item's sampling probability and importance-sampling weight.
    probabilities: numpy.ndarray  # float64 [batch]
    weights: numpy.ndarray  # float32 [batch]
    #: Per step [batch, length, ...]: the `StepArrays` fields, the action taken
    #: at the step before (0 at an episode's start, as Acme's
    #: `ObservationActionRewardWrapper` begins), `padding`, `first` (an
    #: episode's first step) and `last` (its last); per item [batch, state]:
    #: `h` and `c`. A pad step is zero in every field and True in `padding`.
    arrays: Mapping[str, numpy.ndarray]

    def batch(self, device: torch.device | None = None) -> SequenceBatch:
        """The sample as a `SequenceBatch` in the step layout, starting from its stored states."""
        a = {name: _moved(value, device) for name, value in self.arrays.items()}
        size, length = a["action"].shape
        return SequenceBatch(
            scalars=a["scalars"],
            rows=a["rows"].reshape(size, length, ROW_COUNT, ROW_WIDTH),
            mask=a["mask"],
            actions=a["action"],
            rewards=a["reward"],
            dones=a["terminal"],
            padding=a["padding"],
            game_ms=a["game_ms"],
            weights=_moved(self.weights, device),
            burn_in=R2D2_BURN_IN,
            first=a["first"],
            last=a["last"],
            context=(a["h"], a["c"]),
            previous_actions=a["previous_action"],
        )


def _moved(array: numpy.ndarray, device: torch.device | None) -> torch.Tensor:
    """`array` on `device`; to a GPU through pinned memory, so the copy does not block."""
    tensor = torch.from_numpy(numpy.ascontiguousarray(array))
    if device is None or device.type != "cuda":
        return tensor.to(device) if device is not None else tensor
    return tensor.pin_memory().to(device, non_blocking=True)


@dataclass
class R2D2Replay:
    """Prioritized replay of fixed-length items with stored states (module docstring).

    `lock` is shared by the actors and the learner exactly as
    `PrioritizedSequenceReplay.lock` is, and no method takes it itself.

    Items live in a ring of `capacity` slots, oldest first; an item's key is
    its serial number, the count of items inserted before it, and its slot is
    that modulo the capacity. A key older than the oldest live item names an
    evicted one.
    """

    capacity: int = R2D2_REPLAY_CAPACITY
    seed: int | None = None
    state_size: int = R2D2_STATE_SIZE
    #: What `TrainingRun` and the manifest read off a buffer.
    alpha: float = field(default=R2D2_PRIORITY_EXPONENT, init=False)
    beta: float = field(default=R2D2_IMPORTANCE_SAMPLING_EXPONENT, init=False)

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    stats: ReplayStats = field(default_factory=ReplayStats, init=False)
    _episodes: dict[int, _Episode] = field(default_factory=dict, init=False)
    _next_episode: int = field(default=0, init=False)
    #: Items ever inserted, and those live: the live keys are
    #: [inserted - live, inserted).
    _inserted: int = field(default=0, init=False)
    _live: int = field(default=0, init=False)
    #: The item table, by slot. A free slot has priority 0, so it is never drawn.
    _episode: numpy.ndarray = field(init=False)
    _offset: numpy.ndarray = field(init=False)
    _length: numpy.ndarray = field(init=False)
    _priority: numpy.ndarray = field(init=False)
    #: `_priority ** alpha`, kept beside it so a draw does not raise every
    #: priority to alpha.
    _scaled: numpy.ndarray = field(init=False)
    _compatibility: tuple[str, str, str, str] | None = field(default=None, init=False)
    _random: numpy.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("replay capacity must be positive")
        self._episode = numpy.zeros(self.capacity, numpy.int64)
        self._offset = numpy.zeros(self.capacity, numpy.int64)
        self._length = numpy.zeros(self.capacity, numpy.int64)
        self._priority = numpy.zeros(self.capacity, numpy.float64)
        self._scaled = numpy.zeros(self.capacity, numpy.float64)
        self._random = numpy.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self._live

    @property
    def compatibility(self) -> tuple[str, str, str, str] | None:
        return self._compatibility

    @property
    def steps_held(self) -> int:
        return sum(len(episode.steps) for episode in self._episodes.values())

    def add(self, metadata: SequenceMetadata, steps: StepArrays, states: numpy.ndarray) -> bool:
        """Insert one ended episode's items; each enters at priority 1.0.

        `states` [decisions / 40 rounded up, 2, state] is the actor's (h, c)
        before each decision at 0, 40, 80, ...: the zero state first. Each
        item keeps the one at its start.
        """
        key = metadata.compatibility_key()
        if self._compatibility is None:
            self._compatibility = key
        elif key != self._compatibility:
            self.stats.reject("incompatible_profile_or_schema")
            return False
        if len(steps) < 2:
            raise ReplayRejected("an episode needs at least one decision")
        grid = math.ceil((len(steps) - 1) / R2D2_SEQUENCE_PERIOD)
        if states.shape != (grid, 2, self.state_size):
            raise ReplayRejected(
                f"an episode of {len(steps)} steps needs states of shape "
                f"{(grid, 2, self.state_size)}, not {states.shape}"
            )
        starts, lengths = item_layout(len(steps))
        number = self._next_episode
        self._next_episode += 1
        self._episodes[number] = _Episode(
            number,
            metadata,
            steps,
            numpy.array(states[starts // R2D2_SEQUENCE_PERIOD], numpy.float32),
            len(starts),
        )
        for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
            if self._live == self.capacity:
                self._evict()
            slot = self._inserted % self.capacity
            self._episode[slot] = number
            self._offset[slot] = start
            self._length[slot] = length
            self._set_priority(slot, R2D2_INITIAL_PRIORITY)
            self._inserted += 1
            self._live += 1
        self.stats.added += 1
        return True

    def _evict(self) -> None:
        slot = (self._inserted - self._live) % self.capacity
        self._set_priority(slot, 0.0)
        self._live -= 1
        self.stats.evicted += 1
        episode = self._episodes[int(self._episode[slot])]
        episode.live_items -= 1
        if not episode.live_items:
            del self._episodes[episode.number]

    def sample(self, batch_size: int) -> R2D2Sample:
        """`batch_size` items drawn by priority with replacement, gathered and padded."""
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self._live:
            raise ReplayRejected("replay is empty")
        # Inverse-CDF draws over the cumulative p^alpha; a free or zero-priority
        # slot has zero width and is never landed on.
        cumulative = numpy.cumsum(self._scaled)
        total = cumulative[-1]
        # Below `total` even where the product rounds up to it.
        draws = numpy.minimum(
            self._random.random(batch_size) * total, numpy.nextafter(total, 0.0)
        )
        slots = numpy.searchsorted(cumulative, draws, side="right")
        probabilities = self._scaled[slots] / total
        weights = (1.0 / (probabilities + 1e-6)) ** self.beta
        weights /= weights.max()
        oldest = self._inserted - self._live
        keys = oldest + (slots - oldest) % self.capacity
        self.stats.sampled += batch_size
        return R2D2Sample(
            keys=keys,
            probabilities=probabilities,
            weights=weights.astype(numpy.float32),
            arrays=self._gather(slots),
        )

    def _gather(self, slots: numpy.ndarray) -> dict[str, numpy.ndarray]:
        """The items at `slots` as [batch, length] arrays, right-padded with zero steps.

        One slice copy per item and field: the rows move in numpy, never as
        Python values.
        """
        size, length = len(slots), R2D2_ITEM_LENGTH
        episodes = [self._episodes[number] for number in self._episode[slots].tolist()]
        reference = episodes[0].steps
        names = [name.name for name in fields(StepArrays)]
        arrays = {
            name: numpy.zeros(
                (size, length, *getattr(reference, name).shape[1:]),
                getattr(reference, name).dtype,
            )
            for name in names
        }
        previous = numpy.zeros((size, length), reference.action.dtype)
        states = numpy.empty((size, 2, self.state_size), numpy.float32)
        offsets = self._offset[slots]
        counts = self._length[slots]
        for row, (episode, start, count) in enumerate(
            zip(episodes, offsets.tolist(), counts.tolist(), strict=True)
        ):
            steps = episode.steps
            for name in names:
                arrays[name][row, :count] = getattr(steps, name)[start : start + count]
            previous[row, int(start == 0) : count] = steps.action[
                max(start - 1, 0) : start + count - 1
            ]
            states[row] = episode.states[start // R2D2_SEQUENCE_PERIOD]
        position = offsets[:, None] + numpy.arange(length)
        ends = numpy.array([len(episode.steps) - 1 for episode in episodes])[:, None]
        arrays["previous_action"] = previous
        arrays["padding"] = numpy.arange(length) >= counts[:, None]
        arrays["first"] = position == 0
        arrays["last"] = position == ends
        arrays["h"] = states[:, 0]
        arrays["c"] = states[:, 1]
        return arrays

    def update_priorities(self, keys: numpy.ndarray, priorities: numpy.ndarray) -> None:
        """Set each sampled item's priority, as the learner computed it (`r2d2.item_priorities`).

        An item evicted since it was sampled is skipped, as Reverb skips a
        missing key.
        """
        if keys.shape != priorities.shape:
            raise ValueError("each key needs exactly one priority")
        oldest = self._inserted - self._live
        live = (keys >= oldest) & (keys < self._inserted)
        slots = keys[live] % self.capacity
        self._priority[slots] = priorities[live]
        self._scaled[slots] = self._priority[slots] ** self.alpha

    def _set_priority(self, slot: int, priority: float) -> None:
        self._priority[slot] = priority
        self._scaled[slot] = priority**self.alpha

    # -- persistence -------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        return {
            "sequences": self._live,
            "steps": self.steps_held,
            "capacity": self.capacity,
            "compatibility": list(self._compatibility) if self._compatibility else None,
            "added": self.stats.added,
            "rejected": self.stats.rejected,
            "evicted": self.stats.evicted,
            "sampled": self.stats.sampled,
            "rejections_by_reason": dict(self.stats.rejections_by_reason),
        }

    def check_dump(self, metadata: Mapping[str, Any]) -> None:
        """Refuse a dump of another layout, capacity, item geometry or state size."""
        if metadata.get("format_version") != R2D2_REPLAY_DUMP_FORMAT_VERSION:
            raise ReplayDumpError(
                f"replay dump format {metadata.get('format_version')} is not R2D2's item "
                f"replay ({R2D2_REPLAY_DUMP_FORMAT_VERSION}); no earlier buffer holds "
                "items with stored states"
            )
        saved = {name: metadata.get(name) for name in _GEOMETRY}
        if saved != self._geometry():
            raise ReplayDumpError(
                f"replay dump holds {saved}; this buffer is {self._geometry()}"
            )

    def _geometry(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "item_length": R2D2_ITEM_LENGTH,
            "burn_in": R2D2_BURN_IN,
            "period": R2D2_SEQUENCE_PERIOD,
            "state_size": self.state_size,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def image(self) -> R2D2ReplayImage:
        """The buffer at this moment; the caller holds `lock`.

        The item table's live rows are copied, oldest first - a few megabytes
        at capacity - since slots are reused and priorities change. The
        episodes' steps and states are shared, not copied: nothing writes
        them after `add`, so the image stays this moment's however the buffer
        moves on while it is written.
        """
        slots = (self._inserted - self._live + numpy.arange(self._live)) % self.capacity
        return R2D2ReplayImage(
            geometry=self._geometry(),
            episodes=tuple(
                (episode.number, episode.metadata, episode.steps, episode.states)
                for episode in self._episodes.values()
            ),
            item_episode=self._episode[slots],
            item_offset=self._offset[slots],
            item_length=self._length[slots],
            item_priority=self._priority[slots],
            inserted=self._inserted,
            next_episode=self._next_episode,
            compatibility=self._compatibility,
            stats=asdict(self.stats),
            sampler_state=self._random.bit_generator.state,
        )

    def save_to(self, directory: Path, *, run: Mapping[str, Any]) -> int:
        return self.image().write(directory, run=run)

    def load_from(self, directory: Path) -> None:
        """Restore a dump `R2D2ReplayImage.write` wrote into this empty buffer."""
        if self._live or self._episodes:
            raise ReplayDumpError("a dump is only loaded into an empty buffer")
        metadata = read_replay_metadata(directory)
        self.check_dump(metadata)
        lengths = [int(count) for count in metadata["episode_lengths"]]
        items = [int(count) for count in metadata["episode_items"]]
        step_arrays = {}
        for name in fields(StepArrays):
            array = read_rows(directory, name.name)
            if len(array) != sum(lengths):
                raise ReplayDumpError(f"replay dump array {name.name} has {len(array)} rows")
            step_arrays[name.name] = array
        states = read_rows(directory, "states")
        if len(states) != sum(items):
            raise ReplayDumpError(f"replay dump holds {len(states)} states for {sum(items)} items")
        table = {name: read_rows(directory, f"item_{name}") for name in _ITEM_COLUMNS}
        live = len(table["priority"])
        if live > self.capacity or any(len(column) != live for column in table.values()):
            raise ReplayDumpError("replay dump's item table is not one of this buffer's")
        numbers = [int(number) for number in metadata["episode_numbers"]]
        live_items = dict.fromkeys(numbers, 0)
        for number in table["episode"].tolist():
            if number not in live_items:
                raise ReplayDumpError("replay dump's items index episodes it does not hold")
            live_items[number] += 1
        step, state = 0, 0
        for index, number in enumerate(numbers):
            self._episodes[number] = _Episode(
                number,
                SequenceMetadata(**metadata["episode_metadata"][index]),
                StepArrays(
                    **{
                        name: numpy.array(rows[step : step + lengths[index]])
                        for name, rows in step_arrays.items()
                    }
                ),
                numpy.array(states[state : state + items[index]]),
                live_items[number],
            )
            step += lengths[index]
            state += items[index]
        self._inserted = int(metadata["inserted"])
        self._live = live
        slots = (self._inserted - live + numpy.arange(live)) % self.capacity
        self._episode[slots] = table["episode"]
        self._offset[slots] = table["offset"]
        self._length[slots] = table["length"]
        self._priority[slots] = table["priority"]
        self._scaled[slots] = self._priority[slots] ** self.alpha
        self._next_episode = int(metadata["next_episode"])
        compatibility = metadata["compatibility"]
        self._compatibility = None if compatibility is None else tuple(compatibility)
        self.stats = ReplayStats(**metadata["stats"])
        self._random.bit_generator.state = metadata["sampler_state"]


#: The dump's metadata keys that must match the loading buffer's.
_GEOMETRY = ("capacity", "item_length", "burn_in", "period", "state_size", "alpha", "beta")
#: The item table's columns, each one `.npy` of the live items, oldest first.
_ITEM_COLUMNS = ("episode", "offset", "length", "priority")


@dataclass(frozen=True)
class R2D2ReplayImage:
    """An R2D2 replay as it stood at one moment, written without the buffer's lock."""

    geometry: Mapping[str, Any]
    #: (number, metadata, steps, states) of every episode with a live item.
    episodes: tuple[tuple[int, SequenceMetadata, StepArrays, numpy.ndarray], ...]
    #: The live items, oldest first.
    item_episode: numpy.ndarray
    item_offset: numpy.ndarray
    item_length: numpy.ndarray
    item_priority: numpy.ndarray
    inserted: int
    next_episode: int
    compatibility: tuple[str, str, str, str] | None
    stats: Mapping[str, Any]
    sampler_state: Mapping[str, Any]

    def write(self, directory: Path, *, run: Mapping[str, Any]) -> int:
        """Write atomically: one `.npy` per step field, the states, the item table and one JSON.

        Each episode's stored states are written whole, including those of
        its items already evicted, so the states file is indexed by episode
        as the steps are. Returns the bytes written.
        """
        if directory.exists():
            raise ReplayDumpError(f"a replay dump already exists at {directory}")
        temporary = directory.with_name(directory.name + ".partial")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        total = sum(len(steps) for _, _, steps, _ in self.episodes)
        try:
            for name in fields(StepArrays):
                parts = [getattr(steps, name.name) for _, _, steps, _ in self.episodes]
                write_rows(temporary / f"{name.name}.npy", total, parts)
            states = [states for *_, states in self.episodes]
            write_rows(temporary / "states.npy", sum(len(s) for s in states), states)
            columns = (self.item_episode, self.item_offset, self.item_length, self.item_priority)
            for column_name, column in zip(_ITEM_COLUMNS, columns, strict=True):
                numpy.save(temporary / f"item_{column_name}.npy", column)
            metadata = {
                "format_version": R2D2_REPLAY_DUMP_FORMAT_VERSION,
                **self.geometry,
                "steps": total,
                "episode_numbers": [number for number, *_ in self.episodes],
                "episode_lengths": [len(steps) for _, _, steps, _ in self.episodes],
                "episode_items": [len(states) for *_, states in self.episodes],
                "episode_metadata": [asdict(metadata) for _, metadata, _, _ in self.episodes],
                "inserted": self.inserted,
                "next_episode": self.next_episode,
                "compatibility": list(self.compatibility) if self.compatibility else None,
                "stats": dict(self.stats),
                "sampler_state": dict(self.sampler_state),
                "run": dict(run),
            }
            (temporary / REPLAY_DUMP_METADATA).write_text(json.dumps(metadata))
            size = 0
            for path in temporary.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                size += path.stat().st_size
            os.rename(temporary, directory)
            descriptor = os.open(directory.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return size
