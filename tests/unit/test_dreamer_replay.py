"""DreamerV3's step replay against the official `embodied/core/replay.py` it ports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy
import pytest

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.dreamer_replay import DreamerReplay, EpisodeSteps, episode_steps
from tower_rl.learning.replay import REPLAY_DUMP_METADATA, ReplayDumpError, SequenceMetadata

METADATA = SequenceMetadata(
    episode_id="e", actor_id="a", profile_id="p",
    observation_schema="observation-v1", action_schema="run-action-v1",
    reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
)


def _episode(first: int, count: int) -> EpisodeSteps:
    """`count` steps whose reward is their stream position, from `first` on."""
    positions = range(first, first + count)
    return episode_steps(
        scalars=[[0.0] * SCALAR_COUNT for _ in positions],
        rows=[[0.0] * (ROW_COUNT * ROW_WIDTH) for _ in positions],
        mask=[[True] * len(RUN_ACTIONS) for _ in positions],
        action=[1 for _ in positions],
        reward=[float(p) for p in positions],
        terminal=[False for _ in positions],
        game_ms=[1000.0 for _ in positions],
        deter=[numpy.full(2, float(p)) for p in positions],
        stoch=[numpy.full(1, p % 4) for p in positions],
    )


def _replay(capacity: int = 64) -> DreamerReplay:
    """One stream of episodes of 3 and 4 steps, positions 0-6, in items of 3 steps."""
    replay = DreamerReplay(capacity=capacity, length=3, seed=0)
    assert replay.add("a", METADATA, _episode(0, 3))
    assert replay.add("a", METADATA, _episode(3, 4))
    return replay


def _positions(replay: DreamerReplay, sample_size: int) -> list[list[float]]:
    return replay.sample(sample_size).arrays["reward"].tolist()


def test_every_step_starts_an_item_once_its_window_is_complete() -> None:
    """`Replay.add`: an item per stream position with `length` steps from it, across episodes."""
    replay = _replay()
    assert len(replay) == 5  # positions 0-4; 5 and 6 await the next episode
    replay.add("a", METADATA, _episode(7, 2))
    assert len(replay) == 7


def test_a_window_is_annotated_as_the_official_batch_is() -> None:
    """`_annotate_batch`: the first step is `is_first`; a step before an `is_first` is `is_last`."""
    replay = _replay()
    arrays = replay._assemble([replay._runs(0, 1)])  # type: ignore[list-item]
    assert arrays["reward"].tolist() == [[1.0, 2.0, 3.0]]
    assert arrays["first"].tolist() == [[True, False, True]]
    assert arrays["last"].tolist() == [[False, True, False]]


def test_training_samples_take_the_online_queue_first_then_uniform() -> None:
    """Every `length`-th item is queued (`add` 114-118) and sampled first, once (`_sample`)."""
    replay = _replay()
    sample = replay.sample(3)
    # Queued: the items completed when the stream length was a multiple of 3,
    # positions 1 (completed by step 3) and 4 (by step 6).
    assert [window[0] for window in sample.arrays["reward"].tolist()][:2] == [1.0, 4.0]
    # Each once: the queue is spent, and what follows is uniform.
    assert not replay._queue


def test_items_are_evicted_oldest_first_and_their_episode_with_its_last() -> None:
    replay = _replay(capacity=2)
    assert len(replay) == 2
    assert replay.stats.evicted == 3
    # The first episode's last item (position 2) is gone, so is the episode.
    assert replay.steps_held == 4
    # Its queued item (position 1) is skipped.
    assert {window[0] for window in _positions(replay, 6)} <= {3.0, 4.0}


def test_the_learners_latents_replace_the_stored_ones_after_the_context() -> None:
    """`Replay.update`: the trained steps' posterior, across an episode boundary."""
    replay = _replay()
    sample = replay.sample(1)  # position 1: steps 1, 2 | 3
    deter = numpy.array([[[20.0, 20.0], [30.0, 30.0]]], numpy.float32)
    stoch = numpy.array([[[2], [3]]], numpy.int8)
    replay.write_back(sample, deter, stoch)
    first, second = replay._episodes[0].steps, replay._episodes[1].steps
    assert first.deter[:, 0].tolist() == [0.0, 1.0, 20.0]
    assert second.deter[0, 0] == 30.0 and second.stoch[0, 0] == 3
    assert first.stoch[:, 0].tolist() == [0, 1, 2]


def test_a_write_back_skips_what_was_evicted_since_the_sample() -> None:
    replay = DreamerReplay(capacity=2, length=3, seed=0)
    replay.add("a", METADATA, _episode(0, 3))
    sample = replay.sample(1)  # position 0, the only item
    replay.add("a", METADATA, _episode(3, 4))  # evicts the first episode
    replay.write_back(sample, numpy.zeros((1, 2, 2), numpy.float32), numpy.zeros((1, 2, 1)))
    assert replay._episodes[1].steps.deter[0, 0] == 3.0


def test_an_episode_of_another_profile_or_schema_is_refused() -> None:
    replay = _replay()
    other = replace(METADATA, observation_schema="observation-v0")
    assert not replay.add("a", other, _episode(7, 3))
    assert replay.stats.rejections_by_reason == {"incompatible_profile_or_schema": 1}


def test_a_dump_restores_the_buffer_and_its_sampler(tmp_path: Path) -> None:
    replay = _replay()
    replay.sample(1)
    replay.save_to(tmp_path / "dump", run={"run": "r"})
    restored = DreamerReplay(capacity=64, length=3, seed=5)
    restored.load_from(tmp_path / "dump")
    assert restored.snapshot() == replay.snapshot()
    assert _positions(restored, 6) == _positions(replay, 6)
    replay.add("a", METADATA, _episode(7, 2))
    restored.add("a", METADATA, _episode(7, 2))
    assert restored._queue == replay._queue


def test_a_dump_of_another_format_or_shape_is_refused(tmp_path: Path) -> None:
    _replay().save_to(tmp_path / "dump", run={"run": "r"})
    with pytest.raises(ReplayDumpError, match="capacity"):
        DreamerReplay(capacity=32, length=3).load_from(tmp_path / "dump")
    path = tmp_path / "dump" / REPLAY_DUMP_METADATA
    metadata = json.loads(path.read_text())
    path.write_text(json.dumps({**metadata, "format_version": 2}))
    with pytest.raises(ReplayDumpError, match="format 2"):
        DreamerReplay(capacity=64, length=3).load_from(tmp_path / "dump")


def test_an_image_shares_the_latents_and_keeps_its_own_episode_records() -> None:
    """A save holds the learner still, so nothing writes the arrays: they are not copied.

    An actor may still add an episode after the lock is released, which sets
    its predecessor's successor, so the records themselves are the image's own.
    """
    replay = DreamerReplay(capacity=64, length=3, seed=0)
    replay.add("a", METADATA, _episode(0, 3))
    image = replay.image()
    replay.add("a", METADATA, _episode(3, 4))

    (episode,) = image.episodes
    assert episode.steps.deter is replay._episodes[episode.number].steps.deter
    assert episode.successor is None
    assert replay._episodes[episode.number].successor is not None
