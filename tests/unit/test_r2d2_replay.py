"""R2D2's item replay against the Acme replay it ports (`r2d2_replay.py`)."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy
import pytest

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.dreamer_replay import DreamerReplay, episode_steps
from tower_rl.learning.r2d2_replay import (
    R2D2_BURN_IN,
    R2D2_ITEM_LENGTH,
    R2D2_LEARNER_DEBT_BOUND_ITEMS,
    R2D2_LEARNER_STEPS_PER_ITEM,
    R2D2_SEQUENCE_PERIOD,
    R2D2Replay,
    item_layout,
    state_count,
)
from tower_rl.learning.replay import (
    REPLAY_DUMP_METADATA,
    ReplayDumpError,
    ReplayRejected,
    SequenceMetadata,
    read_replay_metadata,
)
from tower_rl.learning.step_arrays import StepArrays

METADATA = SequenceMetadata(
    episode_id="e", actor_id="a", profile_id="p",
    observation_schema="observation-v1", action_schema="run-action-v1",
    reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
)
STATE = 4


def _steps(count: int, first: int = 0) -> StepArrays:
    """`count` steps: step i has reward, scalar 0 and game time `first + i`, action i % 60 + 1."""
    positions = numpy.arange(first, first + count)
    mask = numpy.ones((count, len(RUN_ACTIONS)), numpy.bool_)
    mask[-1] = False
    action = (numpy.arange(count) % 60 + 1).astype(numpy.int64)
    action[-1] = 0
    scalars = numpy.zeros((count, SCALAR_COUNT), numpy.float32)
    scalars[:, 0] = positions
    return StepArrays(
        scalars=scalars,
        rows=numpy.ones((count, ROW_COUNT * ROW_WIDTH), numpy.float32),
        mask=mask,
        action=action,
        reward=positions.astype(numpy.float32),
        terminal=numpy.arange(count) == count - 1,
        game_ms=positions.astype(numpy.float32),
    )


def _states(count: int, size: int = STATE) -> numpy.ndarray:
    """The actor's (h, c) before decisions 0, 40, ...: h = k, c = -k at the k-th."""
    grid = state_count(count)
    marks = numpy.arange(grid, dtype=numpy.float32)[:, None, None]
    return numpy.broadcast_to(marks * numpy.array([1.0, -1.0])[:, None], (grid, 2, size)).astype(
        numpy.float32
    )


def _replay(*counts: int, capacity: int = 1000, seed: int = 0) -> R2D2Replay:
    replay = R2D2Replay(capacity=capacity, seed=seed, state_size=STATE)
    for count in counts:
        assert replay.add(METADATA, _steps(count), _states(count))
    return replay


def _acme_items(step_count: int) -> list[tuple[int, int]]:
    """(start, length) of every item Acme's TRUNCATE configs write (structured.py 278-358).

    The configs' conditions run literally over the step indices; each writes
    the last `length` steps ending at the current one.
    """
    length, period = R2D2_ITEM_LENGTH, R2D2_SEQUENCE_PERIOD
    target = (length - 1) % period
    items = []
    for index in range(step_count):
        written = []
        if index >= length - 1 and index % period == target:
            written.append(length)
        if index == step_count - 1:
            for x in range(period):
                if x != target and index % period == x and index >= length:
                    written.append(length - (target - x) % period)
            for x in range(1, length):
                if index == x - 1:
                    written.append(x)
        items += [(index - n + 1, n) for n in written]
    return items


@pytest.mark.parametrize("count", [2, 41, 120, 121, 122, 160, 161, 162, 522])
def test_items_are_those_acmes_truncate_adder_writes(count: int) -> None:
    starts, lengths = item_layout(count)
    assert list(zip(starts.tolist(), lengths.tolist(), strict=True)) == _acme_items(count)


def test_item_layout_matches_acme_for_every_length_to_600() -> None:
    for count in range(1, 600):
        starts, lengths = item_layout(count)
        assert list(zip(starts.tolist(), lengths.tolist(), strict=True)) == _acme_items(count)


def test_items_start_every_40_steps_and_hold_121() -> None:
    starts, lengths = item_layout(522)
    assert starts.tolist() == list(range(0, 441, 40))
    assert lengths.tolist() == [121] * 10 + [121, 82]
    assert R2D2_ITEM_LENGTH == 40 + 80 + 1


def test_an_episodes_first_40_steps_are_never_a_trace_step() -> None:
    for count in (41, 121, 161, 522):
        starts, lengths = item_layout(count)
        for start, length in zip(starts.tolist(), lengths.tolist(), strict=True):
            trace = range(start + R2D2_BURN_IN, start + length - 1)
            assert not set(trace) & set(range(R2D2_BURN_IN))


def test_a_short_episode_is_one_item_padded_on_the_right() -> None:
    replay = _replay(50)
    assert len(replay) == 1
    arrays = replay.sample(1).arrays
    assert arrays["reward"][0, :50].tolist() == list(range(50))
    assert not arrays["padding"][0, :50].any() and arrays["padding"][0, 50:].all()
    for name in ("scalars", "rows", "mask", "action", "reward", "terminal", "game_ms"):
        assert not arrays[name][0, 50:].any(), name
    assert arrays["terminal"][0, 49] and arrays["last"][0, 49]
    assert arrays["last"][0].sum() == 1


def test_an_episode_of_one_step_is_one_item_of_that_step() -> None:
    """Acme's TRUNCATE writes the one-step episode too (structured.py 350-358)."""
    replay = _replay(1)
    assert len(replay) == 1
    arrays = replay.sample(1).arrays
    assert arrays["padding"][0].tolist() == [False] + [True] * (R2D2_ITEM_LENGTH - 1)
    assert arrays["last"][0, 0] and arrays["first"][0, 0]


def test_states_must_cover_every_40th_decision() -> None:
    replay = R2D2Replay(capacity=10, state_size=STATE)
    with pytest.raises(ReplayRejected):
        replay.add(METADATA, _steps(122), _states(81))


def test_a_sampled_item_is_its_episodes_steps_from_its_start() -> None:
    replay = _replay(162)  # items at 0 (121 steps), 40 (121) and 80 (82)
    sample = replay.sample(64)
    arrays = sample.arrays
    for row in range(64):
        start = int(arrays["reward"][row, 0])
        count = int((~arrays["padding"][row]).sum())
        assert (start, count) in {(0, 121), (40, 121), (80, 82)}
        assert arrays["reward"][row, :count].tolist() == list(range(start, start + count))
        assert arrays["first"][row].tolist() == [start == 0] + [False] * 120
        assert bool(arrays["last"][row, count - 1]) == (start == 80)
        expected = (numpy.arange(start - 1, start + count - 1) % 60 + 1).tolist()
        if start == 0:
            expected[0] = 0
        assert arrays["previous_action"][row, :count].tolist() == expected
        assert not arrays["previous_action"][row, count:].any()


def test_an_item_starts_from_the_state_stored_at_its_first_step() -> None:
    replay = _replay(522)
    arrays = replay.sample(256).arrays
    for row in range(256):
        grid = int(arrays["reward"][row, 0]) // R2D2_SEQUENCE_PERIOD
        assert arrays["h"][row].tolist() == [float(grid)] * STATE
        assert arrays["c"][row].tolist() == [-float(grid)] * STATE


def test_the_batch_carries_the_stored_state_and_the_padding() -> None:
    batch = _replay(162).sample(8).batch()
    assert batch.scalars.shape == (8, 121, SCALAR_COUNT)
    assert batch.rows.shape == (8, 121, ROW_COUNT, ROW_WIDTH)
    assert batch.burn_in == R2D2_BURN_IN
    assert batch.context is not None and batch.context[0].shape == (8, STATE)
    assert batch.previous_actions is not None and batch.previous_actions.shape == (8, 121)
    assert batch.padding.shape == (8, 121) and batch.weights.shape == (8,)


def test_fifo_evicts_the_oldest_item_and_frees_an_episode_with_its_last() -> None:
    replay = _replay(162, capacity=4)  # 3 items
    assert len(replay) == 3 and replay.steps_held == 162
    replay.add(METADATA, _steps(122), _states(122))  # 2 items: one of the first goes
    assert len(replay) == 4 and replay.stats.evicted == 1
    assert replay.steps_held == 162 + 122
    starts = {int(value) for value in replay.sample(200).arrays["reward"][:, 0]}
    assert starts == {40, 80, 0}  # the first episode's 40 and 80, the second's 0 and 40
    replay.add(METADATA, _steps(122), _states(122))
    assert len(replay) == 4 and replay.steps_held == 122 + 122


def test_new_items_enter_at_priority_one() -> None:
    replay = _replay(121)
    sample = replay.sample(1)
    replay.update_priorities(sample.keys, numpy.array([0.9 * 7 + 0.1 * 6]))
    replay.add(METADATA, _steps(121), _states(121))
    image = replay.image()
    assert image.item_priority.tolist() == [pytest.approx(0.9 * 7 + 0.1 * 6), 1.0]


def test_sampling_probabilities_and_weights_are_acmes() -> None:
    replay = _replay(121, 121, 121, seed=3)
    keys = numpy.array([0, 1, 2])
    replay.update_priorities(keys, numpy.array([1.0, 2.0, 4.0]))
    sample = replay.sample(64)
    scaled = numpy.array([1.0, 2.0, 4.0]) ** 0.9
    chances = scaled / scaled.sum()
    assert sample.probabilities.tolist() == pytest.approx(chances[sample.keys].tolist())
    raw = (1.0 / (chances[sample.keys] + 1e-6)) ** 0.6
    assert sample.weights.tolist() == pytest.approx((raw / raw.max()).tolist(), rel=1e-6)
    assert sample.weights.max() == 1.0


def test_importance_weights_are_normalised_by_the_batchs_largest_not_the_buffers() -> None:
    """The least likely item is not drawn, so the batch's largest weight is not the buffer's."""
    replay = _replay(121, 121, 121, seed=5)
    replay.update_priorities(numpy.array([0, 1, 2]), numpy.array([1.0, 2.0, 1e-6]))
    sample = replay.sample(8)
    assert 2 not in sample.keys.tolist()
    raw = (1.0 / (sample.probabilities + 1e-6)) ** 0.6
    assert sample.weights.tolist() == pytest.approx((raw / raw.max()).tolist(), rel=1e-6)
    assert sample.weights.max() == 1.0


def test_an_item_with_priority_zero_is_not_drawn_again() -> None:
    replay = _replay(30, 121)
    replay.update_priorities(numpy.array([0]), numpy.array([0.0]))
    assert set(replay.sample(100).keys.tolist()) == {1}


def test_when_every_priority_is_zero_the_draw_is_uniform_over_the_live_items() -> None:
    """Reverb's PrioritizedSelector::Sample does the same when its total weight is 0."""
    replay = _replay(30, 30, 30, capacity=2, seed=1)  # keys 1 and 2 are live
    replay.update_priorities(numpy.array([1, 2]), numpy.array([0.0, 0.0]))
    sample = replay.sample(200)
    assert set(sample.keys.tolist()) == {1, 2}
    assert sample.probabilities.tolist() == [0.5] * 200
    assert sample.weights.tolist() == [1.0] * 200
    assert sample.arrays["padding"].shape[0] == 200


def test_a_priority_update_skips_an_item_evicted_since_its_sample() -> None:
    replay = _replay(121, capacity=1)
    sample = replay.sample(1)
    replay.add(METADATA, _steps(121), _states(121))
    replay.update_priorities(sample.keys, numpy.array([9.0]))
    assert replay.image().item_priority.tolist() == [1.0]


def test_the_rate_limiter_is_five_learner_steps_per_item_within_acmes_error_buffer() -> None:
    """320 samples per insert / 64 per batch; 1,250 x 320 x 0.1 samples of slack."""
    assert R2D2_LEARNER_STEPS_PER_ITEM == 320 / 64 == 5.0
    assert R2D2_LEARNER_DEBT_BOUND_ITEMS * 320 == 1_250 * 320 * 0.1
    # An ordinary 521-decision episode inserts 12 items, 60 steps' worth at
    # once; eight actors ending one inside a single hold stay under the bound.
    assert len(item_layout(522)[0]) == 12
    assert 8 * 12 * R2D2_LEARNER_STEPS_PER_ITEM < R2D2_LEARNER_DEBT_BOUND_ITEMS * 5


def test_a_dump_round_trips_items_priorities_states_and_the_sampler(tmp_path: Path) -> None:
    replay = _replay(162, 50, 522, capacity=12)
    first = replay.sample(4)
    replay.update_priorities(first.keys, numpy.array([1.5, 2.725, 0.0, 2.0]))
    replay.save_to(tmp_path / "dump", run={"decisions": 7})
    loaded = R2D2Replay(capacity=12, seed=99, state_size=STATE)
    loaded.load_from(tmp_path / "dump")
    assert len(loaded) == len(replay) == 12
    assert loaded.steps_held == replay.steps_held
    before, after = replay.image(), loaded.image()
    for name in ("item_episode", "item_offset", "item_length", "item_priority"):
        assert getattr(after, name).tolist() == getattr(before, name).tolist(), name
    assert after.inserted == before.inserted
    for (number, _, steps, states), (other, _, loaded_steps, loaded_states) in zip(
        before.episodes, after.episodes, strict=True
    ):
        assert number == other
        assert numpy.array_equal(states, loaded_states)
        assert numpy.array_equal(steps.rows, loaded_steps.rows)
    ours, theirs = replay.sample(16), loaded.sample(16)
    assert ours.keys.tolist() == theirs.keys.tolist()
    for name, value in ours.arrays.items():
        assert numpy.array_equal(value, theirs.arrays[name]), name
    replay.add(METADATA, _steps(121), _states(121))
    loaded.add(METADATA, _steps(121), _states(121))
    assert replay.image().item_episode.tolist() == loaded.image().item_episode.tolist()


@pytest.mark.parametrize("version", [1, 2, 3])
def test_an_older_dump_format_is_refused(tmp_path: Path, version: int) -> None:
    replay = _replay(121)
    replay.save_to(tmp_path / "dump", run={})
    path = tmp_path / "dump" / REPLAY_DUMP_METADATA
    metadata = json.loads(path.read_text())
    path.write_text(json.dumps({**metadata, "format_version": version}))
    with pytest.raises(ReplayDumpError, match="not R2D2's"):
        R2D2Replay(capacity=1000, state_size=STATE).load_from(tmp_path / "dump")


def test_a_dreamer_dump_is_refused(tmp_path: Path) -> None:
    dreamer = DreamerReplay(capacity=8, length=2, seed=0)
    dreamer.add(
        "a",
        METADATA,
        episode_steps(
            scalars=[[0.0] * SCALAR_COUNT] * 3, rows=[[0.0] * (ROW_COUNT * ROW_WIDTH)] * 3,
            mask=[[True] * len(RUN_ACTIONS)] * 3, action=[1, 1, 0], reward=[0.0] * 3,
            terminal=[False] * 3, game_ms=[0.0] * 3,
            deter=[numpy.zeros(2)] * 3, stoch=[numpy.zeros(1)] * 3,
        ),
    )
    dreamer.save_to(tmp_path / "dump", run={})
    with pytest.raises(ReplayDumpError, match="format 3"):
        R2D2Replay(capacity=8, state_size=STATE).load_from(tmp_path / "dump")


def test_a_dump_of_another_geometry_is_refused(tmp_path: Path) -> None:
    _replay(121).save_to(tmp_path / "dump", run={})
    with pytest.raises(ReplayDumpError, match="capacity"):
        R2D2Replay(capacity=999, state_size=STATE).load_from(tmp_path / "dump")


def test_a_step_costs_about_2_51_kilobytes() -> None:
    """Spec 2.1: 2,406 B of step (2,413 with an int64 action) and 4,096 B of (h, c) per item."""
    replay = R2D2Replay(capacity=1000, state_size=512)
    for _ in range(4):
        replay.add(METADATA, _steps(522), _states(522, 512))
    image = replay.image()
    step_bytes = sum(
        getattr(steps, name).nbytes for _, _, steps, _ in image.episodes for name in (
            "scalars", "rows", "mask", "action", "reward", "terminal", "game_ms"
        )
    )
    state_bytes = sum(states.nbytes for *_, states in image.episodes)
    per_step = (step_bytes + state_bytes) / replay.steps_held
    print(f"bytes per step: {per_step:.0f}")
    assert 2_450 < per_step < 2_600


def test_sampling_a_64_by_121_batch_at_100k_items_is_quick() -> None:
    """100,000 items over 64 distinct 522-step episodes (80 MB), cycled; the median of 20 draws.

    About 4 ms on the workstation's CPU, most of it the 19 MB copy. The bound
    is loose for a loaded host; what it catches is a per-value Python collate,
    which cost 40 ms or more.
    """
    replay = R2D2Replay(capacity=100_000, seed=0, state_size=STATE)
    episodes = [_steps(522, first=1000 * index) for index in range(64)]
    states = _states(522)
    index = 0
    while len(replay) < 100_000:
        replay.add(METADATA, episodes[index % 64], states)
        index += 1
    times = []
    for _ in range(20):
        began = time.perf_counter()
        replay.sample(64)
        times.append(time.perf_counter() - began)
    median = sorted(times)[10]
    print(f"sample 64x121 at 100k items: {median * 1000:.2f} ms")
    assert median < 0.025


def test_incompatible_profile_or_schema_is_rejected_and_counted() -> None:
    replay = _replay(121)
    for field_name, value in (
        ("profile_id", "profile-v2"),
        ("observation_schema", "observation-v2"),
        ("action_schema", "run-action-v2"),
        ("reward_schema", "reward-v2"),
    ):
        other = dataclasses.replace(METADATA, **{field_name: value})
        assert not replay.add(other, _steps(121), _states(121)), field_name
    assert len(replay) == 1, "the first episode's one item, and none since"
    assert replay.stats.rejected == 4
    assert replay.stats.rejections_by_reason["incompatible_profile_or_schema"] == 4
    # A different model version or epsilon is ordinary off-policy data, not a
    # compatibility break.
    other = dataclasses.replace(METADATA, model_version=99, epsilon=0.9)
    assert replay.add(other, _steps(121), _states(121))
    assert len(replay) == 2


def test_an_empty_buffer_saves_and_reloads_empty(tmp_path: Path) -> None:
    R2D2Replay(capacity=5, state_size=STATE).save_to(tmp_path / "replay", run={})
    loaded = R2D2Replay(capacity=5, state_size=STATE)
    loaded.load_from(tmp_path / "replay")
    assert len(loaded) == 0 and loaded.compatibility is None and loaded.steps_held == 0


def test_a_save_never_overwrites_an_existing_dump(tmp_path: Path) -> None:
    replay = _replay(121)
    replay.save_to(tmp_path / "replay", run={"decisions": 1})
    replay.add(METADATA, _steps(50), _states(50))
    with pytest.raises(ReplayDumpError, match="already exists"):
        replay.save_to(tmp_path / "replay", run={"decisions": 2})
    assert read_replay_metadata(tmp_path / "replay")["run"] == {"decisions": 1}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["replay"]


def test_a_dump_is_only_loaded_into_an_empty_buffer(tmp_path: Path) -> None:
    _replay(121).save_to(tmp_path / "replay", run={})
    with pytest.raises(ReplayDumpError, match="empty buffer"):
        _replay(50).load_from(tmp_path / "replay")


@pytest.mark.parametrize("array", ["rows", "action", "states", "item_priority"])
def test_a_dump_whose_array_lost_rows_is_refused(tmp_path: Path, array: str) -> None:
    _replay(121, 50).save_to(tmp_path / "replay", run={})
    path = tmp_path / "replay" / f"{array}.npy"
    numpy.save(path, numpy.load(path)[:-1])
    with pytest.raises(ReplayDumpError, match="rows|states|item table|index episodes"):
        R2D2Replay(capacity=1000, state_size=STATE).load_from(tmp_path / "replay")


def test_a_failed_save_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    replay = _replay(121, 50)
    monkeypatch.setattr(numpy, "save", refuse)
    with pytest.raises(OSError, match="disk full"):
        replay.save_to(tmp_path / "replay", run={})
    assert list(tmp_path.iterdir()) == []
