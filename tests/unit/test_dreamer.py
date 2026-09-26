"""DreamerV3 as a backbone: shapes, the action mask, padding, resume, streams, a run.

A small configuration on the CPU throughout; the published sizes are what
`DreamerConfig()` defaults to and are exercised by the step-time measurement
recorded in `docs/experiments.md`, not here.
"""

from __future__ import annotations

import io
import math
from dataclasses import replace

import pytest
import torch
from fakes.backbone_equality import parameters_are_equal
from fakes.fake_run_port import FakeRunPort

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.actor import Actor, ActorConfig
from tower_rl.learning.backbone import SequenceBatch, acting_copy, collate
from tower_rl.learning.dreamer import WAIT_INDEX, DreamerBackbone, DreamerConfig
from tower_rl.learning.exploration import ExplorationSchedule
from tower_rl.learning.replay import (
    PrioritizedSequenceReplay,
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.learning.training import TrainingConfig, TrainingRun

ACTIONS = len(RUN_ACTIONS)
LENGTH = 6
SMALL = DreamerConfig(
    deter=16, hidden=8, classes=4, units=8, stoch=4, blocks=2,
    batch_size=2, batch_length=LENGTH, warmup=2, seed=0,
)


def _backbone(config: DreamerConfig = SMALL) -> DreamerBackbone:
    return DreamerBackbone(config=config)


def _features(*, valid: tuple[int, ...] = (0, 1, 2), seed: float = 0.5) -> StateFeatures:
    return StateFeatures(
        scalars=tuple([seed] * SCALAR_COUNT),
        rows=tuple([seed] * (ROW_COUNT * ROW_WIDTH)),
        mask=tuple(index in valid for index in range(ACTIONS)),
    )


def _sequence(*, padding: int = 0, filler: float = 0.0, done: bool = True) -> ReplaySequence:
    """One window; padded steps carry `filler` in their features and action.

    Their reward is 0 and they never end an episode, as the actor pads: that is
    the reward and termination the episode's first real step is trained on,
    matching the official `is_first` target.
    """
    steps = tuple(
        ReplayStep(
            features=_features(seed=filler if index < padding else 0.1 * index),
            action_index=int(filler * 7) % 3 if index < padding else index % 3,
            reward=0.0 if index < padding else 1.0 + index,
            done=done and index == LENGTH - 1,
            admissible=True,
            game_ms=1000.0,
            padding=index < padding,
        )
        for index in range(LENGTH)
    )
    return ReplaySequence(
        SequenceMetadata(
            episode_id="e", actor_id="a", profile_id="p",
            observation_schema="observation-v1", action_schema="run-action-v1",
            reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
        ),
        steps,
        0,
    )


def _batch(*, padding: int = 0, filler: float = 0.0) -> SequenceBatch:
    return collate(
        (_sequence(padding=padding, filler=filler), _sequence(done=False)), (1.0, 1.0)
    )


def _learn(backbone: DreamerBackbone, batch: SequenceBatch, *, seed: int) -> float:
    torch.manual_seed(seed)
    return backbone.learn(batch).weighted_loss


def test_acting_carries_the_recurrent_state_and_learning_reports_every_sequence() -> None:
    backbone = _backbone()
    c = SMALL
    state = backbone.initial_state()
    assert [tuple(part.shape) for part in state] == [
        (1, c.deter), (1, c.stoch * c.classes), (1, ACTIONS)
    ]
    action, state = backbone.act(_features(), state, epsilon=0.0)
    assert action in (0, 1, 2)
    deter, stoch, previous = state
    assert tuple(deter.shape) == (1, c.deter)
    # The stochastic state is one one-hot per latent.
    assert stoch.view(c.stoch, c.classes).sum(-1).tolist() == [1.0] * c.stoch
    assert previous.argmax().item() == action and previous.sum().item() == 1.0

    metrics = backbone.learn(_batch(padding=2))
    assert backbone.model_version == 1
    assert len(metrics.td_errors) == 2 and all(metrics.td_errors)
    assert math.isfinite(metrics.weighted_loss) and math.isfinite(metrics.gradient_norm)
    assert metrics.gradient_norm > 0.0


def test_the_published_configuration_is_the_one_documented() -> None:
    published = DreamerConfig()
    assert (published.deter, published.hidden, published.classes, published.units) == (
        2048, 256, 16, 256,
    )
    assert (published.batch_size, published.batch_length) == (16, 64)
    assert published.gradient_steps_per_decision == 0.5
    assert published.discount == pytest.approx(1 - 1 / 333)


def test_learning_refuses_a_batch_of_another_shape_or_with_burn_in() -> None:
    backbone = _backbone()
    batch = _batch()
    with pytest.raises(ValueError, match="burn-in"):
        backbone.learn(replace(batch, burn_in=1))
    single = collate((_sequence(),), (1.0,))
    with pytest.raises(ValueError, match="batches"):
        backbone.learn(single)


def test_acting_never_samples_an_invalid_action() -> None:
    backbone = _backbone()
    # A strongly preferring actor, so an unmasked policy would pick the invalid action.
    with torch.no_grad():
        backbone.actor[-1].bias.zero_()
        backbone.actor[-1].bias[5] = 50.0
    valid = (0, 7, 33)
    state = backbone.initial_state()
    seen = set()
    for _ in range(300):
        action, state = backbone.act(_features(valid=valid), state, epsilon=0.0)
        seen.add(action)
    assert seen <= set(valid)
    # 1% uniform over the valid actions reaches every one of them.
    assert seen == set(valid)


def test_imagination_samples_only_what_the_decoded_mask_allows() -> None:
    backbone = _backbone()
    allowed = {WAIT_INDEX, 4, 9}
    with torch.no_grad():
        head = backbone.world_model.decode_mask
        head.weight.zero_()
        head.bias.fill_(-1.0)
        for index in (4, 9):
            head.bias[index] = 1.0
        backbone.actor[-1].bias.zero_()
        backbone.actor[-1].bias[20] = 50.0
    c = SMALL
    torch.manual_seed(0)
    with torch.no_grad():
        features, actions, masks = backbone._imagine(
            torch.randn(64, c.deter), torch.zeros(64, c.stoch * c.classes)
        )
    assert tuple(features.shape) == (64, c.imagination_horizon + 1, c.deter + c.stoch * c.classes)
    assert set(actions.unique().tolist()) <= allowed
    assert masks.gather(-1, actions.unsqueeze(-1)).all()
    assert masks[..., WAIT_INDEX].all()


def test_padded_steps_contribute_nothing_to_the_update() -> None:
    """Whatever padding holds, the loss and every parameter after it are identical."""
    left, right = _backbone(), _backbone()
    for step in range(3):
        assert _learn(left, _batch(padding=3, filler=0.0), seed=step) == _learn(
            right, _batch(padding=3, filler=9.0), seed=step
        )
    for name in ("world_model", "actor", "critic", "slow_critic"):
        assert parameters_are_equal(getattr(left, name), getattr(right, name)), name
    # And the steps did move the parameters: the warm-up rate is zero only at first.
    assert not parameters_are_equal(left.world_model, _backbone().world_model)


def test_a_reloaded_backbone_resumes_exactly() -> None:
    original = _backbone()
    for step in range(3):
        _learn(original, _batch(padding=2), seed=step)
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    resumed = _backbone(replace(SMALL, seed=99))
    resumed.load_state_dict(torch.load(buffer, weights_only=False))
    assert resumed.model_version == original.model_version

    for step in range(3, 6):
        assert _learn(original, _batch(), seed=step) == _learn(resumed, _batch(), seed=step)
    for name in ("world_model", "actor", "critic", "slow_critic"):
        assert parameters_are_equal(getattr(original, name), getattr(resumed, name)), name
    assert [t.item() for t in original.return_normaliser.stats()] == [
        t.item() for t in resumed.return_normaliser.stats()
    ]


def test_bfloat16_compute_changes_rounding_not_the_update() -> None:
    """The first update's loss within bfloat16 tolerance of float32's; state stays float32.

    On the CPU the learner defaults to float32 and no compilation; mixed
    precision is forced here to exercise the bfloat16 path (the CPU's autocast).
    """
    reference = _backbone()
    assert reference.mixed_precision is False and reference.compiled is False
    mixed = DreamerBackbone(config=SMALL, mixed_precision=True)
    mixed.load_state_dict(reference.state_dict())
    batch = _batch(padding=2)
    expected = _learn(reference, batch, seed=0)
    assert _learn(mixed, batch, seed=0) == pytest.approx(expected, rel=1e-2)
    for name in ("world_model", "actor", "critic", "slow_critic"):
        for parameter in getattr(mixed, name).parameters():
            assert parameter.dtype == torch.float32, name
    for state in mixed.optimizer.state.values():
        assert state["nu"].dtype == state["mu"].dtype == torch.float32
    assert all(math.isfinite(_learn(mixed, batch, seed=seed)) for seed in (1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_cuda_learner_is_compiled_bfloat16_and_saves_a_plain_checkpoint() -> None:
    """On CUDA both are on by default; the update and the checkpoint are float32's."""
    cuda = torch.device("cuda")
    reference = DreamerBackbone(config=SMALL, device=cuda, mixed_precision=False, compiled=False)
    optimised = DreamerBackbone(config=SMALL, device=cuda)
    assert optimised.mixed_precision is True and optimised.compiled is True
    optimised.load_state_dict(reference.state_dict())
    batch = collate(
        (_sequence(padding=2), _sequence(done=False)), (1.0, 1.0), device=cuda
    )
    expected = _learn(reference, batch, seed=0)
    assert _learn(optimised, batch, seed=0) == pytest.approx(expected, rel=1e-2)
    assert all(math.isfinite(_learn(optimised, batch, seed=seed)) for seed in (1, 2))

    # Compiled functions, not compiled modules: no `_orig_mod.` in any key, and
    # the checkpoint loads into a CPU backbone, which acts from it.
    state = optimised.state_dict()
    assert state["world_model"].keys() == reference.state_dict()["world_model"].keys()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    evaluating = _backbone()
    evaluating.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=False))
    assert evaluating.model_version == 3
    action, _ = evaluating.act(_features(), evaluating.initial_state(), epsilon=0.0)
    assert action in (0, 1, 2)


def _actions(backbone: DreamerBackbone, count: int = 40) -> list[int]:
    state = backbone.initial_state()
    chosen = []
    for index in range(count):
        action, state = backbone.act(
            _features(valid=tuple(range(ACTIONS)), seed=0.01 * index), state, epsilon=0.0
        )
        chosen.append(action)
    return chosen


def test_each_acting_copy_samples_from_a_stream_of_its_own() -> None:
    learner = _backbone()
    first = acting_copy(learner, exploration_seed="actor-0")
    again = acting_copy(learner, exploration_seed="actor-0")
    other = acting_copy(learner, exploration_seed="actor-1")
    assert isinstance(first, DreamerBackbone) and isinstance(other, DreamerBackbone)
    assert isinstance(again, DreamerBackbone)
    torch_stream = torch.random.get_rng_state()
    assert _actions(first) == _actions(again)
    assert _actions(first) != _actions(other)
    # Acting draws nothing from torch's stream, which the learner samples from.
    assert torch.equal(torch.random.get_rng_state(), torch_stream)


def test_a_short_training_run_on_the_fake_port_takes_finite_optimisation_steps() -> None:
    environment = InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=2.0),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )
    backbone = _backbone()
    replay = PrioritizedSequenceReplay(capacity=64, seed=0)
    actor = Actor(
        environment=environment,
        policy=acting_copy(backbone),
        config=ActorConfig(sequence_length=LENGTH, burn_in=0, stride=LENGTH // 2),
        replay=replay,
    )
    training = TrainingRun(
        actors=[actor],
        replay=replay,
        backbone=backbone,
        config=TrainingConfig(
            budget_decisions=120,
            warmup_sequences=2,
            batch_size=SMALL.batch_size,
            gradient_steps_per_decision=0.25,
            exploration=ExplorationSchedule(
                epsilon_start=0.0, epsilon_end=0.0, anneal_decisions=1
            ),
        ),
    )
    report = training.run()
    assert report.decisions >= 120
    assert report.optimisation_steps > 0
    assert backbone.model_version == report.optimisation_steps
    assert all(math.isfinite(loss) for loss in report.recent_weighted_losses)
