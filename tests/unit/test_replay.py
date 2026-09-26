from __future__ import annotations

import pytest

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, encode_state
from tower_rl.environment.run_actions import RUN_ACTIONS, WAIT, action_index, upgrade_action
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.replay import (
    PRIORITY_FLOOR,
    R2D2_IMPORTANCE_SAMPLING_EXPONENT,
    R2D2_PRIORITY_EXPONENT,
    R2D2_PRIORITY_MIX,
    PrioritizedSequenceReplay,
    ReplayRejected,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.simulation.instrumented_bridge import BridgeObservation, UpgradeInventoryEntry

BUILDER = RunStateBuilder(profile_id="profile-v1")


def _reading(sequence: int = 1, cash: float = 100.0) -> BridgeObservation:
    entries = tuple(
        UpgradeInventoryEntry(
            family=family,
            index=index,
            cost=10.0,
            level=1,
            max_level=50,
            unlocked=index < 2,
            tier_unlocked=False,
            maxed=False,
        )
        for family in ("attack", "defense", "utility")
        for index in range(20)
    )
    return BridgeObservation(
        sequence=sequence,
        lifecycle="active",
        wave=3,
        cash=cash,
        health=4.0,
        max_health=5.0,
        terminal=False,
        round_active=True,
        game_speed=8.0,
        play_time=10.0,
        upgrades=entries,
    )


def _features(cash: float = 100.0):
    return encode_state(BUILDER.build(_reading(cash=cash), captured_at_monotonic=1.0))


def _metadata(**overrides: object) -> SequenceMetadata:
    base: dict[str, object] = {
        "episode_id": "episode-1",
        "actor_id": "actor-0",
        "profile_id": "profile-v1",
        "observation_schema": "observation-v1",
        "action_schema": "run-action-v1",
        "reward_schema": "reward-v1",
        "model_version": 3,
        "epsilon": 0.1,
        "game_speed": 8.0,
    }
    base.update(overrides)
    return SequenceMetadata(**base)  # type: ignore[arg-type]


def _sequence(length: int = 4, burn_in: int = 1, **metadata: object) -> ReplaySequence:
    steps = tuple(
        ReplayStep(
            features=_features(),
            action_index=action_index(WAIT),
            reward=1.0,
            done=step == length - 1,
            admissible=True,
            game_ms=1000.0,
        )
        for step in range(length)
    )
    return ReplaySequence(metadata=_metadata(**metadata), steps=steps, burn_in=burn_in)


def test_encoding_layout_matches_the_action_space() -> None:
    features = _features()

    assert len(features.scalars) == SCALAR_COUNT
    assert len(features.rows) == ROW_COUNT * ROW_WIDTH
    assert len(features.mask) == len(RUN_ACTIONS)
    # Row i describes action index i + 1, because index 0 is always WAIT.
    assert len(RUN_ACTIONS) - 1 == ROW_COUNT
    assert features.mask[action_index(upgrade_action("attack", 0))] is True
    assert features.mask[action_index(upgrade_action("utility", 5))] is False


def test_sequences_need_a_learning_step_after_burn_in() -> None:
    with pytest.raises(ReplayRejected, match="burn-in"):
        _sequence(length=3, burn_in=3)
    with pytest.raises(ReplayRejected, match="at least one step"):
        ReplaySequence(metadata=_metadata(), steps=(), burn_in=0)


def test_incompatible_profile_or_schema_is_rejected_and_counted() -> None:
    replay = PrioritizedSequenceReplay(capacity=8, seed=1)

    assert replay.add(_sequence())
    assert not replay.add(_sequence(profile_id="profile-v2"))
    assert not replay.add(_sequence(observation_schema="observation-v2"))
    assert not replay.add(_sequence(action_schema="run-action-v2"))

    assert len(replay) == 1
    assert replay.stats.rejected == 3
    assert replay.stats.rejections_by_reason["incompatible_profile_or_schema"] == 3
    # A different model version or epsilon is ordinary off-policy data, not a
    # compatibility break.
    assert replay.add(_sequence(model_version=99, epsilon=0.9))


def test_inadmissible_transitions_never_enter_replay() -> None:
    replay = PrioritizedSequenceReplay(capacity=4, seed=1)
    steps = (
        ReplayStep(_features(), action_index(WAIT), 1.0, False, admissible=True, game_ms=0.0),
        ReplayStep(_features(), action_index(WAIT), 0.0, False, admissible=False, game_ms=0.0),
    )

    assert not replay.add(ReplaySequence(_metadata(), steps, burn_in=0))
    assert replay.stats.rejections_by_reason["inadmissible_transition"] == 1
    assert len(replay) == 0


def test_the_buffer_is_bounded_and_evicts_oldest_first() -> None:
    replay = PrioritizedSequenceReplay(capacity=3, seed=1)
    for index in range(5):
        replay.add(_sequence(episode_id=f"episode-{index}"))

    assert len(replay) == 3
    assert replay.stats.evicted == 2
    episodes = {sequence.metadata.episode_id for sequence in replay._items}
    assert episodes == {"episode-2", "episode-3", "episode-4"}


def test_priority_updates_bias_sampling_towards_surprise() -> None:
    replay = PrioritizedSequenceReplay(capacity=4, alpha=1.0, seed=7)
    for index in range(4):
        replay.add(_sequence(episode_id=f"episode-{index}"))

    replay.update_priorities((0, 1, 2, 3), ((0.001,), (0.001,), (0.001,), (50.0,)))
    _, sequences, _ = replay.sample(200)

    surprising = sum(1 for item in sequences if item.metadata.episode_id == "episode-3")
    assert surprising > 150, "the high-error sequence must dominate sampling"


def test_importance_weights_are_normalized_and_favour_rare_samples() -> None:
    replay = PrioritizedSequenceReplay(capacity=4, alpha=1.0, seed=3)
    for index in range(2):
        replay.add(_sequence(episode_id=f"episode-{index}"))
    replay.update_priorities((0, 1), ((0.01,), (10.0,)))

    indices, _, weights = replay.sample(50)

    assert all(0.0 < weight <= 1.0 for weight in weights)
    rare = [weight for index, weight in zip(indices, weights, strict=True) if index == 0]
    common = [weight for index, weight in zip(indices, weights, strict=True) if index == 1]
    if rare and common:
        assert max(common) <= min(rare), "over-sampled sequences must be down-weighted"


def test_priority_mixes_maximum_and_mean_error() -> None:
    replay = PrioritizedSequenceReplay(capacity=2, seed=1)
    replay.add(_sequence())

    replay.update_priorities((0,), ((10.0, 0.0, -0.0, 0.0),))

    expected = 0.9 * 10.0 + 0.1 * 2.5
    assert replay._priorities[0] == pytest.approx(expected)
    # Absolute errors: a negative TD error is as surprising as a positive one.
    replay.update_priorities((0,), ((-10.0, 0.0, 0.0, 0.0),))
    assert replay._priorities[0] == pytest.approx(expected)


def test_the_published_r2d2_constants_are_what_stacked_dqn_samples_by() -> None:
    """Kapturowski et al. 2019 (R2D2), as Acme's `r2d2/config.py` states them.

    alpha 0.9 (`priority_exponent`), beta 0.6 (`importance_sampling_exponent`),
    eta 0.9 (`max_priority_weight`). A buffer built without arguments is the
    one stacked-dqn trains from, so these are its values.
    """
    assert (R2D2_PRIORITY_EXPONENT, R2D2_IMPORTANCE_SAMPLING_EXPONENT) == (0.9, 0.6)
    assert R2D2_PRIORITY_MIX == 0.9
    replay = PrioritizedSequenceReplay(capacity=4)
    assert (replay.alpha, replay.beta) == (0.9, 0.6)


def test_uniform_replay_samples_evenly_and_weighs_every_sequence_as_one() -> None:
    """DreamerV3's replay: priorities are kept, but never read."""
    replay = PrioritizedSequenceReplay.uniform(4, seed=5)
    assert (replay.alpha, replay.beta) == (0.0, 0.0)
    for index in range(4):
        replay.add(_sequence(episode_id=f"episode-{index}"))
    replay.update_priorities((0, 1, 2, 3), ((0.001,), (0.001,), (0.001,), (1000.0,)))

    indices, _, weights = replay.sample(4000)

    assert set(weights) == {1.0}
    for index in range(4):
        assert indices.count(index) / 4000 == pytest.approx(0.25, abs=0.03)


def test_sampling_is_proportional_to_priority_to_the_alpha() -> None:
    """P(i) = p_i ** 0.9 / sum_k p_k ** 0.9, drawn with replacement."""
    replay = PrioritizedSequenceReplay(capacity=3, seed=11)
    for index in range(3):
        replay.add(_sequence(episode_id=f"episode-{index}"))
    # A single error gives priority exactly |error|: max and mean agree.
    replay.update_priorities((0, 1, 2), ((1.0,), (4.0,), (10.0,)))

    draws = 30_000
    indices, _, _ = replay.sample(draws)

    powered = [priority**0.9 for priority in (1.0, 4.0, 10.0)]
    for index, weight in enumerate(powered):
        expected = weight / sum(powered)
        assert indices.count(index) / draws == pytest.approx(expected, abs=0.01)


def test_importance_weights_are_the_published_formula_normalised_by_the_batch() -> None:
    """w_i = (N P(i)) ** -beta over the batch's largest, so the rarest weighs one."""
    replay = PrioritizedSequenceReplay(capacity=3, seed=2)
    for index in range(3):
        replay.add(_sequence(episode_id=f"episode-{index}"))
    replay.update_priorities((0, 1, 2), ((1.0,), (4.0,), (10.0,)))

    indices, _, weights = replay.sample(64)

    powered = [priority**0.9 for priority in (1.0, 4.0, 10.0)]
    probabilities = [weight / sum(powered) for weight in powered]
    raw = [(3 * probabilities[index]) ** -0.6 for index in indices]
    expected = [weight / max(raw) for weight in raw]
    assert weights == pytest.approx(expected)
    assert max(weights) == pytest.approx(1.0)


def test_one_sequence_at_the_floor_does_not_shrink_every_other_weight() -> None:
    """The normaliser is the batch's: a buffer-wide one would be the floor's."""
    replay = PrioritizedSequenceReplay(capacity=64, seed=4)
    for index in range(64):
        replay.add(_sequence(episode_id=f"episode-{index}"))
    replay.update_priorities(
        tuple(range(64)), ((0.0,),) + tuple((1.0,) for _ in range(63))
    )
    assert replay._priorities[0] == PRIORITY_FLOOR

    indices, _, weights = replay.sample(8)

    assert 0 not in indices, "the floor is sampled almost never"
    assert weights == (1.0,) * 8, "equal priorities weigh equally, at one"


def test_learner_feedback_reaches_the_sequence_it_was_sampled_from_in_a_full_buffer() -> None:
    """Sample, update, add: the update lands before eviction shifts anything.

    End to end at capacity, as a production buffer spends almost all of its
    life: every add evicts the oldest sequence, and the priority learned for
    a sampled sequence must stay with that sequence while it lives.
    """
    replay = PrioritizedSequenceReplay(capacity=4, seed=9)
    for index in range(4):
        replay.add(_sequence(episode_id=f"episode-{index}"))

    for step in range(4, 12):
        indices, sequences, _ = replay.sample(2)
        replay.update_priorities(indices, tuple((float(step),) for _ in indices))
        for index, sequence in zip(indices, sequences, strict=True):
            assert replay._items[index] is sequence
            assert replay._priorities[index] == pytest.approx(float(step))
        # The next add evicts the oldest and shifts every index down by one;
        # a priority travels with its sequence rather than with its position.
        kept = {id(item): priority for item, priority in zip(
            replay._items, replay._priorities, strict=True
        )}
        replay.add(_sequence(episode_id=f"episode-{step}"))
        assert len(replay) == 4 and replay.stats.evicted == step - 3
        for item, priority in list(zip(replay._items, replay._priorities, strict=True))[:-1]:
            assert kept[id(item)] == priority
        # A new sequence enters at the current maximum.
        assert replay._priorities[-1] == max(replay._priorities)


def test_updates_for_indices_the_buffer_never_held_are_dropped() -> None:
    replay = PrioritizedSequenceReplay(capacity=1, seed=1)
    replay.add(_sequence())

    replay.update_priorities((5,), ((1.0,),))  # index beyond the buffer

    assert len(replay._priorities) == 1


def test_priorities_cannot_be_updated_once_eviction_has_shifted_every_index() -> None:
    """A stale index still lands - on the wrong sequence. That is refused."""
    replay = PrioritizedSequenceReplay(capacity=2, seed=1)
    replay.add(_sequence(episode_id="episode-0"))
    replay.add(_sequence(episode_id="episode-1"))
    indices, _, _ = replay.sample(1)

    replay.add(_sequence(episode_id="episode-2"))  # evicts, shifting indices down

    with pytest.raises(ReplayRejected, match="eviction"):
        replay.update_priorities(indices, ((1.0,),))


def test_a_new_sequence_enters_at_the_current_maximum_not_a_historical_one() -> None:
    """Schaul 2016 inserts at the current maximum.

    A monotone one would let a single large early TD error pin insertion
    priority forever, so late in training every new sequence would enter far
    above the steady state and sampling would degenerate towards recency.
    """
    replay = PrioritizedSequenceReplay(capacity=8, seed=1)
    replay.add(_sequence(episode_id="episode-0"))
    replay.update_priorities((0,), ((50.0,),))  # one early surprise

    replay.add(_sequence(episode_id="episode-1"))
    assert replay._priorities[1] == pytest.approx(replay._priorities[0])

    replay.update_priorities((0, 1), ((0.1,), (0.1,)))  # the surprise settles
    replay.add(_sequence(episode_id="episode-2"))

    assert replay._priorities[2] == pytest.approx(0.1), "the spike must not persist"


def test_an_empty_replay_refuses_to_sample() -> None:
    replay = PrioritizedSequenceReplay(capacity=2, seed=1)

    with pytest.raises(ReplayRejected, match="empty"):
        replay.sample(1)


def test_snapshot_reports_what_a_checkpoint_must_state() -> None:
    replay = PrioritizedSequenceReplay(capacity=2, seed=1)
    replay.add(_sequence())
    replay.add(_sequence(profile_id="other"))

    snapshot = replay.snapshot()

    assert snapshot["sequences"] == 1
    assert snapshot["capacity"] == 2
    assert snapshot["compatibility"] == [
        "profile-v1",
        "observation-v1",
        "run-action-v1",
        "reward-v1",
    ]
    assert snapshot["rejected"] == 1


@pytest.mark.parametrize("game_ms", [-1.0, float("nan"), float("inf")])
def test_a_step_whose_game_time_is_negative_or_not_finite_is_refused(game_ms: float) -> None:
    """T7: a learner that discounts by game time would read it as a discount."""
    with pytest.raises(ReplayRejected, match="game time"):
        ReplayStep(_features(), action_index(WAIT), 0.0, False, admissible=True, game_ms=game_ms)


def test_a_step_must_say_how_much_game_time_it_spanned() -> None:
    """No default: a forgotten call site must fail, not silently discount nothing."""
    with pytest.raises(TypeError, match="game_ms"):
        ReplayStep(_features(), action_index(WAIT), 0.0, False, admissible=True)  # type: ignore[call-arg]
