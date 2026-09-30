"""DreamerV3's replay: step streams, one item per step, with stored latents.

A port of the official `embodied/core/replay.py` (danijar/dreamerv3 at
e3f02248) as `dreamerv3/main.py` `make_replay` configures it, in place of the
window buffer stacked-dqn samples (`replay.py`). What it follows, by line:

- Each actor writes one stream of steps, episode after episode. Every stream
  position becomes an item - the start of one window of `length` steps - as
  soon as `length` steps follow it, so windows start at every step and run
  across episode boundaries, where `is_first` resets the model
  (`Replay.add`, 77-118).
- `length` is the batch length plus the replay context, 64 + 1
  (`main.py` 187). The context step is read only for its stored latent,
  which is where the window's recurrent state starts (`agent.py`
  `_apply_replay_context`, 312-340).
- Items are sampled uniformly and evicted oldest first past `capacity`
  (`_insert`, `_remove`, 171-191; `selectors.Uniform`).
- The online queue: every `length`-th item of a stream is also queued, and a
  training sample takes queued items first, oldest first, each once
  (`add` 114-118, `_sample` 158-160).
- After each training step the learner's posterior latents of the window's
  trained steps are written back over the stored ones (`agent.py` 144-150,
  `Replay.update` 140-149); an item evicted meanwhile is skipped.
- A sampled window's first step is marked `is_first` and any step followed
  by an `is_first` is marked `is_last` (`_annotate_batch`, 278-292).

Two things differ, and both are forced (docs/solution.md 9.4c). Steps are
added a whole episode at a time, when the episode ends: a counted episode is
in replay whole or not at all, which the resume pair (ADR 0014) and the
learner's debt (ADR 0017) are both built on. And a stream is cut at its first
inadmissible transition, whose observation may not be valid: the
environment's admissibility contract (docs/environment-contract.md) gates
replay, which the official replay has no notion of.

A step is stored in Dreamer's own layout: its observation, the action taken
at it, and the reward, termination and game time of the transition *into* it.
An episode's last step is its final observation - the environment's terminal
one when the run died - with no action taken.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import SequenceBatch
from tower_rl.learning.replay import (
    REPLAY_DUMP_METADATA,
    ReplayDumpError,
    ReplayRejected,
    ReplayStats,
    SequenceMetadata,
    read_replay_metadata,
)

#: `configs.yaml` `replay.size`: items, which is steps.
DREAMER_REPLAY_CAPACITY = 5_000_000
#: `configs.yaml` `replay_context`.
DREAMER_REPLAY_CONTEXT = 1
#: The layout of a saved step replay. 1 and 2 are the window buffer's
#: (`replay.REPLAY_DUMP_FORMAT_VERSION`), which this buffer refuses: a DreamerV3
#: run saved before stored latents cannot continue under them.
DREAMER_REPLAY_DUMP_FORMAT_VERSION = 3


@dataclass(frozen=True)
class EpisodeSteps:
    """One episode's steps in Dreamer's layout, one row per step.

    The latent is the acting policy's posterior at that step, `deter` and the
    stochastic state as class indices: its sample is one-hot, so the indices
    hold it exactly in 1/64 of the official float32 one-hot's bytes.
    """

    scalars: numpy.ndarray  # float32 [n, SCALAR_COUNT]
    rows: numpy.ndarray  # float32 [n, ROW_COUNT * ROW_WIDTH]
    mask: numpy.ndarray  # bool [n, actions]
    #: The action taken at the step; 0 at the last step, where none is.
    action: numpy.ndarray  # int64 [n]
    #: The transition into the step: 0, False and 0 at the first.
    reward: numpy.ndarray  # float32 [n]
    terminal: numpy.ndarray  # bool [n]
    game_ms: numpy.ndarray  # float32 [n]
    deter: numpy.ndarray  # float32 [n, deter]
    stoch: numpy.ndarray  # int8 [n, stoch]

    def __post_init__(self) -> None:
        count = len(self.action)
        if count < 1:
            raise ReplayRejected("an episode needs at least one step")
        if any(len(getattr(self, name.name)) != count for name in fields(self)):
            raise ReplayRejected("every field of an episode needs one row per step")

    def __len__(self) -> int:
        return len(self.action)


@dataclass
class _Episode:
    number: int
    actor_id: str
    metadata: SequenceMetadata
    steps: EpisodeSteps
    #: The stream position of the episode's first step.
    stream_start: int
    #: The same actor's next episode, once it has one.
    successor: int | None = None


@dataclass(frozen=True)
class DreamerSample:
    """One sampled batch: its arrays, and where each window's steps live, for the write-back."""

    #: Per window, the (episode, offset, count) runs its steps were read from.
    runs: tuple[tuple[tuple[int, int, int], ...], ...]
    arrays: Mapping[str, numpy.ndarray]

    def batch(self, device: torch.device | None = None) -> SequenceBatch:
        """The sample as a `SequenceBatch` with context, in Dreamer's layout."""
        a = {name: torch.as_tensor(value, device=device) for name, value in self.arrays.items()}
        size, length = a["action"].shape
        return SequenceBatch(
            scalars=a["scalars"],
            rows=a["rows"].reshape(size, length, ROW_COUNT, ROW_WIDTH),
            mask=a["mask"],
            actions=a["action"],
            rewards=a["reward"],
            dones=a["terminal"],
            padding=torch.zeros(size, length, dtype=torch.bool, device=device),
            game_ms=a["game_ms"],
            weights=torch.ones(size, device=device),
            burn_in=DREAMER_REPLAY_CONTEXT,
            first=a["first"],
            last=a["last"],
            context=(a["deter"][:, 0], a["stoch"][:, 0].to(torch.int64)),
        )


@dataclass
class DreamerReplay:
    """Uniform step replay with an online queue and latents written back (module docstring).

    `lock` is shared by the actors and the learner exactly as
    `PrioritizedSequenceReplay.lock` is, and no method takes it itself.
    """

    capacity: int = DREAMER_REPLAY_CAPACITY
    #: Steps per item: the batch length plus the context.
    length: int = 64 + DREAMER_REPLAY_CONTEXT
    seed: int | None = None
    #: Uniform: what `TrainingRun` and the manifest read off a buffer.
    alpha: float = field(default=0.0, init=False)
    beta: float = field(default=0.0, init=False)

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    stats: ReplayStats = field(default_factory=ReplayStats, init=False)
    _episodes: dict[int, _Episode] = field(default_factory=dict, init=False)
    _next_episode: int = field(default=0, init=False)
    #: Per actor: the stream length, and its newest episode.
    _stream_length: dict[str, int] = field(default_factory=dict, init=False)
    _newest: dict[str, int] = field(default_factory=dict, init=False)
    #: Per actor: the (episode, offset) of every stream position not yet an item.
    _pending: dict[str, deque[tuple[int, int]]] = field(default_factory=dict, init=False)
    #: Live items, oldest first: (episode, offset).
    _items: deque[tuple[int, int]] = field(default_factory=deque, init=False)
    _queue: deque[tuple[int, int]] = field(default_factory=deque, init=False)
    _compatibility: tuple[str, str, str, str] | None = field(default=None, init=False)
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("replay capacity must be positive")
        if self.length < 2:
            raise ValueError("an item needs its context and at least one step")
        self._random = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def compatibility(self) -> tuple[str, str, str, str] | None:
        return self._compatibility

    @property
    def steps_held(self) -> int:
        return sum(len(episode.steps) for episode in self._episodes.values())

    def add(self, actor_id: str, metadata: SequenceMetadata, steps: EpisodeSteps) -> bool:
        """Append one episode to its actor's stream; every position it completes becomes an item."""
        key = metadata.compatibility_key()
        if self._compatibility is None:
            self._compatibility = key
        elif key != self._compatibility:
            self.stats.reject("incompatible_profile_or_schema")
            return False
        number = self._next_episode
        self._next_episode += 1
        start = self._stream_length.get(actor_id, 0)
        self._episodes[number] = _Episode(number, actor_id, metadata, steps, start)
        if actor_id in self._newest:
            self._episodes[self._newest[actor_id]].successor = number
        self._newest[actor_id] = number
        self._stream_length[actor_id] = start + len(steps)
        pending = self._pending.setdefault(actor_id, deque())
        for offset in range(len(steps)):
            pending.append((number, offset))
            # `Replay.add`: the oldest pending position becomes an item once
            # `length` steps have followed it into the stream.
            if len(pending) >= self.length:
                item = pending.popleft()
                self._insert(item)
                position = self._episodes[item[0]].stream_start + item[1]
                # `lengths[worker] % length == 0` at the step that completes
                # the item, the step `length - 1` after its start.
                if (position + self.length - 1) % self.length == 0:
                    self._queue.append(item)
        self.stats.added += 1
        return True

    def _insert(self, item: tuple[int, int]) -> None:
        while len(self._items) >= self.capacity:
            self._remove()
        self._items.append(item)

    def _remove(self) -> None:
        number, offset = self._items.popleft()
        self.stats.evicted += 1
        episode = self._episodes[number]
        if offset == len(episode.steps) - 1:
            # No live item starts in it or before it in its stream any more;
            # later items start in its successors and never read back into it.
            del self._episodes[number]

    def sample(self, batch_size: int) -> DreamerSample:
        """`Replay.sample(batch, 'train')`: queued items first, then uniform."""
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not self._items:
            raise ReplayRejected("replay is empty")
        windows: list[tuple[tuple[int, int, int], ...]] = []
        while len(windows) < batch_size:
            if self._queue:
                item = self._queue.popleft()
            else:
                item = self._items[self._random.randrange(len(self._items))]
            runs = self._runs(*item)
            if runs is None:
                continue  # its steps were evicted since it was queued
            windows.append(runs)
        self.stats.sampled += batch_size
        return DreamerSample(tuple(windows), self._assemble(windows))

    def _runs(self, number: int, offset: int) -> tuple[tuple[int, int, int], ...] | None:
        """The (episode, offset, count) runs of the window starting at one item."""
        runs = []
        remaining = self.length
        current: int | None = number
        while remaining:
            if current is None or current not in self._episodes:
                return None
            steps = len(self._episodes[current].steps)
            count = min(remaining, steps - offset)
            runs.append((current, offset, count))
            remaining -= count
            current, offset = self._episodes[current].successor, 0
        return tuple(runs)

    def _assemble(self, windows: Sequence[tuple[tuple[int, int, int], ...]]) -> dict[str, Any]:
        """`_assemble_batch` then `_annotate_batch`: the windows as [batch, length] arrays."""
        names = [name.name for name in fields(EpisodeSteps)]
        parts: dict[str, list[numpy.ndarray]] = {name: [] for name in (*names, "first", "last")}
        for runs in windows:
            for number, offset, count in runs:
                steps = self._episodes[number].steps
                for name in names:
                    parts[name].append(getattr(steps, name)[offset : offset + count])
                positions = numpy.arange(offset, offset + count)
                parts["first"].append(positions == 0)
                parts["last"].append(positions == len(steps) - 1)
        size = len(windows)
        arrays = {
            name: numpy.concatenate(values).reshape(size, self.length, *values[0].shape[1:])
            for name, values in parts.items()
        }
        arrays["first"][:, 0] = True
        following = numpy.roll(arrays["first"], -1, axis=1)
        following[:, -1] = False
        arrays["last"] |= following
        return arrays

    def write_back(self, sample: DreamerSample, deter: numpy.ndarray, stoch: numpy.ndarray) -> None:
        """`Replay.update`: the trained steps' posterior latents over the stored ones.

        `deter` [batch, length - context, deter] and `stoch` [batch, length -
        context, stoch] class indices, for every step of each window after its
        context. A window evicted since it was sampled is skipped, a run at a
        time, as the official `KeyError` is.
        """
        for window, runs in enumerate(sample.runs):
            written = -DREAMER_REPLAY_CONTEXT
            for number, offset, count in runs:
                skip = max(0, -written)
                episode = self._episodes.get(number)
                if episode is not None and count > skip:
                    rows = slice(offset + skip, offset + count)
                    source = slice(written + skip, written + count)
                    episode.steps.deter[rows] = deter[window, source]
                    episode.steps.stoch[rows] = stoch[window, source]
                written += count

    # -- persistence -------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        return {
            "sequences": len(self._items),
            "steps": self.steps_held,
            "queued": len(self._queue),
            "capacity": self.capacity,
            "compatibility": list(self._compatibility) if self._compatibility else None,
            "added": self.stats.added,
            "rejected": self.stats.rejected,
            "evicted": self.stats.evicted,
            "sampled": self.stats.sampled,
            "rejections_by_reason": dict(self.stats.rejections_by_reason),
        }

    def check_dump(self, metadata: Mapping[str, Any]) -> None:
        """Refuse a dump of another layout, capacity or item length."""
        if metadata.get("format_version") != DREAMER_REPLAY_DUMP_FORMAT_VERSION:
            raise ReplayDumpError(
                f"replay dump format {metadata.get('format_version')} is not DreamerV3's "
                f"step replay ({DREAMER_REPLAY_DUMP_FORMAT_VERSION}); a dump from before "
                "stored latents cannot continue under them"
            )
        saved = (metadata.get("capacity"), metadata.get("length"))
        if saved != (self.capacity, self.length):
            raise ReplayDumpError(
                f"replay dump holds capacity {saved[0]} and length {saved[1]}; this buffer "
                f"is capacity {self.capacity} and length {self.length}"
            )

    def image(self) -> DreamerReplayImage:
        """The buffer at this moment; the caller holds `lock` and the learner still.

        The image shares the episodes' arrays rather than copying them, which
        would double the latents' memory for every save. That is sound only
        while nothing writes them until the image is written, after the lock
        is released: an actor only appends whole episodes, but the write-back
        changes the latents in place, and it runs only inside a learner step.
        So the learner must be held still (`LearnerThread.held`) from here
        until the image is written, as both callers do: a resume point is
        written under `TrainingRun._learner_still` and the run's last one
        under `TrainingRun.held_still`. Each episode's record is copied, as
        its `successor` is set when its actor adds the next one.
        """
        episodes = tuple(replace(episode) for episode in self._episodes.values())
        return DreamerReplayImage(
            capacity=self.capacity,
            length=self.length,
            episodes=episodes,
            items=tuple(self._items),
            queue=tuple(self._queue),
            pending={actor: tuple(items) for actor, items in self._pending.items()},
            stream_length=dict(self._stream_length),
            newest=dict(self._newest),
            next_episode=self._next_episode,
            compatibility=self._compatibility,
            stats=asdict(self.stats),
            sampler_state=self._random.getstate(),
        )

    def save_to(self, directory: Path, *, run: Mapping[str, Any]) -> int:
        return self.image().write(directory, run=run)

    def load_from(self, directory: Path) -> None:
        """Restore a dump `DreamerReplayImage.write` wrote into this empty buffer."""
        if self._items or self._episodes:
            raise ReplayDumpError("a dump is only loaded into an empty buffer")
        metadata = read_replay_metadata(directory)
        self.check_dump(metadata)
        total = int(metadata["steps"])
        lengths = metadata["episode_lengths"]
        if sum(lengths) != total:
            raise ReplayDumpError("replay dump's episode lengths do not add up to its steps")
        arrays = {}
        for name in fields(EpisodeSteps):
            array = _read(directory, name.name)
            if len(array) != total:
                raise ReplayDumpError(f"replay dump array {name.name} has {len(array)} rows")
            arrays[name.name] = array
        start = 0
        for index, count in enumerate(lengths):
            steps = EpisodeSteps(
                **{name: numpy.array(rows[start : start + count]) for name, rows in arrays.items()}
            )
            number = int(metadata["episode_numbers"][index])
            successor = metadata["episode_successors"][index]
            self._episodes[number] = _Episode(
                number,
                metadata["episode_actors"][index],
                SequenceMetadata(**metadata["episode_metadata"][index]),
                steps,
                int(metadata["episode_stream_starts"][index]),
                None if successor is None else int(successor),
            )
            start += count
        self._items = _pairs(metadata["items"])
        self._queue = _pairs(metadata["queue"])
        self._pending = {actor: _pairs(items) for actor, items in metadata["pending"].items()}
        for number, _offset in (*self._items, *(i for p in self._pending.values() for i in p)):
            if number not in self._episodes:
                raise ReplayDumpError("replay dump's items index episodes it does not hold")
        self._stream_length = {k: int(v) for k, v in metadata["stream_length"].items()}
        self._newest = {k: int(v) for k, v in metadata["newest"].items()}
        self._next_episode = int(metadata["next_episode"])
        compatibility = metadata["compatibility"]
        self._compatibility = None if compatibility is None else tuple(compatibility)
        self.stats = ReplayStats(**metadata["stats"])
        version, internal, gauss = metadata["sampler_state"]
        self._random.setstate((version, tuple(internal), gauss))


@dataclass(frozen=True)
class DreamerReplayImage:
    """A step replay as it stood at one moment, written without the buffer's lock."""

    capacity: int
    length: int
    episodes: tuple[_Episode, ...]
    items: tuple[tuple[int, int], ...]
    queue: tuple[tuple[int, int], ...]
    pending: Mapping[str, tuple[tuple[int, int], ...]]
    stream_length: Mapping[str, int]
    newest: Mapping[str, int]
    next_episode: int
    compatibility: tuple[str, str, str, str] | None
    stats: Mapping[str, Any]
    sampler_state: tuple[Any, ...]

    def write(self, directory: Path, *, run: Mapping[str, Any]) -> int:
        """Write atomically, one `.npy` per step field and one JSON; return the bytes written.

        Each field's array is streamed episode by episode, so no second copy of
        the buffer is built in memory; the directory is renamed into place only
        once every file is on disk (as `ReplayImage.write`).
        """
        if directory.exists():
            raise ReplayDumpError(f"a replay dump already exists at {directory}")
        temporary = directory.with_name(directory.name + ".partial")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        total = sum(len(episode.steps) for episode in self.episodes)
        try:
            for name in fields(EpisodeSteps):
                _write(temporary / f"{name.name}.npy", total, self.episodes, name.name)
            metadata = {
                "format_version": DREAMER_REPLAY_DUMP_FORMAT_VERSION,
                "capacity": self.capacity,
                "length": self.length,
                "steps": total,
                "episode_numbers": [episode.number for episode in self.episodes],
                "episode_lengths": [len(episode.steps) for episode in self.episodes],
                "episode_actors": [episode.actor_id for episode in self.episodes],
                "episode_metadata": [asdict(episode.metadata) for episode in self.episodes],
                "episode_stream_starts": [episode.stream_start for episode in self.episodes],
                "episode_successors": [episode.successor for episode in self.episodes],
                "items": [list(item) for item in self.items],
                "queue": [list(item) for item in self.queue],
                "pending": {
                    actor: [list(item) for item in items] for actor, items in self.pending.items()
                },
                "stream_length": dict(self.stream_length),
                "newest": dict(self.newest),
                "next_episode": self.next_episode,
                "compatibility": list(self.compatibility) if self.compatibility else None,
                "stats": dict(self.stats),
                "sampler_state": list(self.sampler_state),
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


def _write(path: Path, total: int, episodes: Sequence[_Episode], name: str) -> None:
    """One step field of every episode as one `.npy`, written an episode at a time."""
    if not episodes:
        numpy.save(path, numpy.zeros((0,), numpy.float32))
        return
    reference = getattr(episodes[0].steps, name)
    with path.open("wb") as stream:
        numpy.lib.format.write_array_header_1_0(
            stream,
            {
                "descr": reference.dtype.str,
                "fortran_order": False,
                "shape": (total, *reference.shape[1:]),
            },
        )
        for episode in episodes:
            stream.write(numpy.ascontiguousarray(getattr(episode.steps, name)).tobytes())
        stream.flush()
        os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def _pairs(values: Sequence[Sequence[int]]) -> deque[tuple[int, int]]:
    return deque((int(first), int(second)) for first, second in values)


def _read(directory: Path, name: str) -> Any:
    try:
        return numpy.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ReplayDumpError(f"replay dump array {name} is unreadable: {error}") from error


def episode_steps(
    scalars: Sequence[Sequence[float]],
    rows: Sequence[Sequence[float]],
    mask: Sequence[Sequence[bool]],
    action: Sequence[int],
    reward: Sequence[float],
    terminal: Sequence[bool],
    game_ms: Sequence[float],
    deter: Sequence[numpy.ndarray],
    stoch: Sequence[numpy.ndarray],
) -> EpisodeSteps:
    """An `EpisodeSteps` at the stored dtypes, from per-step values."""
    return EpisodeSteps(
        scalars=numpy.asarray(scalars, numpy.float32).reshape(len(action), SCALAR_COUNT),
        rows=numpy.asarray(rows, numpy.float32).reshape(len(action), ROW_COUNT * ROW_WIDTH),
        mask=numpy.asarray(mask, numpy.bool_).reshape(len(action), len(RUN_ACTIONS)),
        action=numpy.asarray(action, numpy.int64),
        reward=numpy.asarray(reward, numpy.float32),
        terminal=numpy.asarray(terminal, numpy.bool_),
        game_ms=numpy.asarray(game_ms, numpy.float32),
        deter=numpy.stack(deter).astype(numpy.float32),
        stoch=numpy.stack(stoch).astype(numpy.int8),
    )

