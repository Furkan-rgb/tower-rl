"""Which steps of an episode actually reach replay.

Under `reward-v1` the reward is a wave delta, so the step that ends the episode
carries the whole of the negative signal, and an episode too short to fill one
window is an early death - the most informative failure there is. Both used to be
dropped silently: nothing in the loss, the TD errors or the gradient norms shows
experience that was never stored. These are the regression tests for that.
"""

from __future__ import annotations

from typing import cast

import pytest

from tower_rl.application.actor import Actor, ActorConfig
from tower_rl.application.policies import Policy
from tower_rl.application.replay import (
    PrioritizedSequenceReplay,
    ReplayRejected,
    ReplaySequence,
    ReplayStep,
)
from tower_rl.application.run_environment import InstrumentedRunEnvironment
from tower_rl.domain.episode import EpisodeSummary, TerminationOutcome
from tower_rl.domain.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures

LENGTH, BURN_IN, STRIDE = 8, 4, 4


def _features() -> StateFeatures:
    return StateFeatures(
        scalars=(0.5,) * SCALAR_COUNT,
        rows=(0.5,) * (ROW_COUNT * ROW_WIDTH),
        mask=tuple(index < 3 for index in range(61)),
    )


def _episode(decisions: int) -> list[ReplayStep]:
    return [
        ReplayStep(
            features=_features(),
            action_index=0,
            reward=float(step),
            done=step == decisions - 1,
            admissible=True,
        )
        for step in range(decisions)
    ]


def _actor(replay: PrioritizedSequenceReplay) -> Actor:
    return Actor(
        environment=cast(InstrumentedRunEnvironment, None),
        policy=cast(Policy, None),
        config=ActorConfig(sequence_length=LENGTH, burn_in=BURN_IN, stride=STRIDE),
        replay=replay,
    )


def _summary() -> EpisodeSummary:
    return EpisodeSummary(
        episode_id="episode-1",
        profile_id="profile-v1",
        final_wave=3,
        decisions=1,
        purchases=0,
        termination=TerminationOutcome.GAME_OVER,
        elapsed_wall_seconds=1.0,
        game_speed=8.0,
        invalid_transitions=0,
    )


def _stored(decisions: int) -> list[ReplaySequence]:
    replay = PrioritizedSequenceReplay(capacity=256, seed=0)
    offered, accepted = _actor(replay)._emit(_episode(decisions), _summary())

    assert offered == accepted, "no window may be refused by replay"
    return list(replay._items)


@pytest.mark.parametrize("decisions", list(range(1, 3 * LENGTH + 1)))
def test_every_episode_length_stores_its_terminal_step(decisions: int) -> None:
    """Including lengths shorter than one window, which used to store nothing."""
    stored = _stored(decisions)

    assert stored, "an episode of any length must reach replay"
    terminal = [
        (sequence, index)
        for sequence in stored
        for index, step in enumerate(sequence.steps)
        if step.done and not step.padding
    ]
    assert terminal, "the step that ended the episode must be stored"
    for sequence, index in terminal:
        assert index >= sequence.burn_in, "a terminal step inside burn-in is never learned"


@pytest.mark.parametrize("decisions", list(range(1, 3 * LENGTH + 1)))
def test_every_stored_window_is_a_full_window_of_real_experience(decisions: int) -> None:
    stored = _stored(decisions)

    for sequence in stored:
        assert len(sequence.steps) == LENGTH
        # Padding fills the front only, and only for an episode too short to
        # fill a window on its own.
        padded = [step.padding for step in sequence.steps]
        assert padded == sorted(padded, reverse=True)
        assert not any(padded) or decisions < LENGTH
        assert sum(padded) == max(0, LENGTH - decisions)


def test_a_short_episode_keeps_its_real_steps_at_the_end_of_the_window() -> None:
    (sequence,) = _stored(3)

    rewards = [step.reward for step in sequence.steps[-3:]]
    assert rewards == [0.0, 1.0, 2.0]
    assert all(step.padding for step in sequence.steps[:-3])
    assert not any(step.padding for step in sequence.steps[-3:])


def test_padding_alone_is_not_a_learning_window() -> None:
    """The learner needs at least one real step, or it has no TD error to give."""
    padded = Actor._left_padded(_episode(1), LENGTH)
    metadata = _stored(1)[0].metadata

    # Everything but the last step is filler, so this window holds no experience.
    with pytest.raises(ReplayRejected, match="padding alone"):
        ReplaySequence(metadata, padded[:-1], BURN_IN)


def test_windows_overlap_rather_than_lose_the_end_of_a_long_episode() -> None:
    """A length that is not a multiple of the stride still ends on the last step."""
    stored = _stored(LENGTH + STRIDE + 1)

    assert stored[-1].steps[-1].done
    # The end-aligned window repeats part of its predecessor; duplicated
    # experience is harmless, a dropped terminal step is not.
    assert stored[-1].steps[0].reward < stored[-1].steps[-1].reward


def test_an_episode_with_no_decisions_emits_nothing() -> None:
    replay = PrioritizedSequenceReplay(capacity=4, seed=0)

    assert _actor(replay)._emit([], _summary()) == (0, 0)
    assert len(replay) == 0
