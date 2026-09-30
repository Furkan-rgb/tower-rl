"""DreamerV3 as a backbone: shapes, the action mask, replay context, resume, streams, a run.

A small configuration on the CPU throughout; the published sizes are what
`DreamerConfig()` defaults to and are exercised by the step-time measurement
recorded in `docs/experiments.md`, not here.
"""

from __future__ import annotations

import io
import math
import threading
from dataclasses import replace
from typing import Any, cast

import numpy
import pytest
import torch
from fakes.backbone_equality import parameters_are_equal
from fakes.fake_run_port import FakeRunPort
from torch.nn import functional

from tower_rl.environment.episode import TerminationOutcome
from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.learning.actor import Actor
from tower_rl.learning.backbone import SequenceBatch, acting_copy, collate
from tower_rl.learning.dreamer import WAIT_INDEX, DreamerBackbone, DreamerConfig
from tower_rl.learning.dreamer_math import symlog
from tower_rl.learning.dreamer_replay import (
    DreamerReplay,
    DreamerSample,
    EpisodeSteps,
    episode_steps,
)
from tower_rl.learning.exploration import ExplorationSchedule
from tower_rl.learning.replay import (
    ReplaySequence,
    ReplayStep,
    SequenceMetadata,
)
from tower_rl.learning.training import TrainingConfig, TrainingRun

ACTIONS = len(RUN_ACTIONS)
LENGTH = 6
SMALL = DreamerConfig(
    deter=16, hidden=8, classes=4, units=8, stoch=4, blocks=2,
    batch_size=2, batch_length=LENGTH, warmup=2, seed=0, discount_per_game_second=0.999,
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


def _metadata() -> SequenceMetadata:
    return _sequence().metadata


def _episode(
    count: int = LENGTH + 1,
    *,
    died: bool = True,
    game_ms: tuple[float, ...] | None = None,
    reward: float | None = None,
) -> EpisodeSteps:
    """One episode in Dreamer's layout: step t carries the transition into it.

    The first step has none; the last is the final observation, the death when
    `died`. Latents are arbitrary but fixed, as the acting policy stored them.
    """
    generator = torch.Generator().manual_seed(count)
    return episode_steps(
        scalars=[[0.1 * index] * SCALAR_COUNT for index in range(count)],
        rows=[[0.1 * index] * (ROW_COUNT * ROW_WIDTH) for index in range(count)],
        mask=[[i in (0, 1, 2) for i in range(ACTIONS)] for _ in range(count)],
        action=[index % 3 if index < count - 1 else 0 for index in range(count)],
        reward=[
            0.0 if index == 0 else (1.0 + index if reward is None else reward)
            for index in range(count)
        ],
        terminal=[died and index == count - 1 for index in range(count)],
        game_ms=[
            0.0 if index == 0 else (1000.0 if game_ms is None else game_ms[index - 1])
            for index in range(count)
        ],
        deter=list(torch.randn(count, SMALL.deter, generator=generator).numpy()),
        stoch=list(torch.randint(0, SMALL.classes, (count, SMALL.stoch), generator=generator)
                   .numpy()),
    )


def _windows(*episodes: EpisodeSteps, device: torch.device | None = None) -> SequenceBatch:
    """The window at the start of each episode, as `DreamerReplay` batches it."""
    replay = DreamerReplay(capacity=64, length=LENGTH + 1)
    for index, episode in enumerate(episodes):
        replay.add(f"actor-{index}", _metadata(), episode)
    runs = tuple(((index, 0, LENGTH + 1),) for index in range(len(episodes)))
    return DreamerSample(runs, replay._assemble(runs)).batch(device)


def _batch(device: torch.device | None = None) -> SequenceBatch:
    """A window that ends in a death, and one that runs on."""
    return _windows(_episode(), _episode(LENGTH + 3, died=False), device=device)


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

    metrics = backbone.learn(_batch())
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
    # The discount is the task's, set by the run (ADR 0013), not a code default.
    assert published.discount_per_game_second is None
    assert not published.survival_time_reward


def test_learning_refuses_a_config_without_the_game_time_discount() -> None:
    backbone = _backbone(replace(SMALL, discount_per_game_second=None))
    with pytest.raises(ValueError, match="game-time discount"):
        backbone.learn(_batch())
    with pytest.raises(ValueError, match="survival-time reward needs"):
        replace(SMALL, discount_per_game_second=None, survival_time_reward=True)


def test_learning_refuses_a_batch_of_another_shape_or_without_stored_latents() -> None:
    backbone = _backbone()
    batch = _batch()
    with pytest.raises(ValueError, match="latents"):
        backbone.learn(replace(batch, context=None))
    with pytest.raises(ValueError, match="latents"):
        backbone.learn(collate((_sequence(), _sequence()), (1.0, 1.0)))
    with pytest.raises(ValueError, match="batches"):
        backbone.learn(_windows(_episode()))


def test_the_actor_mixes_in_no_uniform_as_the_official_categorical_head_does() -> None:
    """`heads.py` `Head.categorical` builds `outs.Categorical(logits)` with no unimix."""
    assert DreamerConfig().actor_unimix == 0.0


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
    # The preferred action is masked out, and the valid ones share what is left.
    assert seen == set(valid)


def test_imagination_samples_only_what_the_decoded_mask_allows() -> None:
    backbone = _backbone()
    allowed = {WAIT_INDEX, 4, 9}
    with torch.no_grad():
        # Two classes per action, invalid then valid: only WAIT, 4 and 9 decode valid.
        head = backbone.world_model.decode_mask
        head.weight.zero_()
        bias = head.bias.view(ACTIONS, 2)
        bias[:, 0] = 1.0
        bias[:, 1] = -1.0
        for index in (4, 9):
            bias[index] = torch.tensor([-1.0, 1.0])
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


def test_the_mask_is_a_two_class_key_as_the_official_code_takes_a_boolean() -> None:
    """`elements/space.py` 15-16, 42-43: bool is discrete with 2 classes.

    The encoder takes it one-hot (`nets.py` 488-493), the decoder predicts a
    categorical per action (`rssm.py` 299-300) whose loss sums over actions
    (`heads.py` `Agg`), and the decoded mask is its argmax.
    """
    world = _backbone().world_model
    scalars, rows = torch.full((1, SCALAR_COUNT), 0.5), torch.full((1, ROW_COUNT, ROW_WIDTH), 0.5)
    mask = torch.tensor([[index in (0, 3) for index in range(ACTIONS)]])
    first = world.encoder[0]
    assert first.in_features == SCALAR_COUNT + ROW_COUNT * ROW_WIDTH + 2 * ACTIONS
    flat = torch.cat(
        (
            symlog(scalars),
            symlog(rows.flatten(-2)),
            functional.one_hot(mask.long(), 2).flatten(-2).float(),
        ),
        -1,
    )
    assert torch.equal(world.encode(scalars, rows, mask), world.encoder(flat))

    logits = torch.randn(1, ACTIONS, 2)
    expected = -functional.log_softmax(logits, -1)[0, torch.arange(ACTIONS), mask[0].long()]
    assert world.mask_loss(logits, mask).item() == pytest.approx(expected.sum().item(), rel=1e-6)
    valid = world.valid_actions(logits)
    assert torch.equal(valid[0, 1:], (logits[0, 1:, 1] > logits[0, 1:, 0]))
    assert valid[0, WAIT_INDEX]


def test_a_backbone_from_before_the_one_hot_mask_keeps_its_0_1_mask() -> None:
    """Its network is shaped for the 0/1 input and binary head, and still acts on them."""
    backbone = _backbone(replace(SMALL, mask_one_hot=False))
    world = backbone.world_model
    assert world.encoder[0].in_features == SCALAR_COUNT + ROW_COUNT * ROW_WIDTH + ACTIONS
    logits = torch.tensor([[-1.0] * ACTIONS])
    logits[0, 5] = 1.0
    assert world.valid_actions(logits)[0].nonzero().flatten().tolist() == [WAIT_INDEX, 5]
    action, _ = backbone.act(_features(), backbone.initial_state(), epsilon=0.0)
    assert action in (0, 1, 2)

def _with_context(batch: SequenceBatch, **changes: Any) -> SequenceBatch:
    """The batch with its context step's (index 0) fields changed."""
    fields = {}
    for name, value in changes.items():
        tensor = getattr(batch, name).clone()
        tensor[:, 0] = value
        fields[name] = tensor
    return replace(batch, **fields)


def _gradient(batch: SequenceBatch) -> torch.Tensor:
    """Every parameter's gradient from one update of a fresh backbone.

    The heads start at zero output, as the official ones do, so their loss is
    the same for any target, and the learning rate warms up from 0: the
    gradient is what a target reaches.
    """
    backbone = _backbone()
    _learn(backbone, batch, seed=0)
    return torch.cat([
        parameter.grad.flatten()
        for group in backbone.optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ])


def test_the_context_step_enters_the_update_only_through_its_stored_latent() -> None:
    """`_apply_replay_context`: the carry is the context's latent; its data is not a window step.

    Its action is the previous action of the window's first step, so that is read too.
    """
    batch = _batch()
    reference = _gradient(batch)
    changed = _with_context(batch, scalars=9.0, rewards=5.0, dones=True, game_ms=7000.0)
    assert torch.equal(_gradient(changed), reference)
    assert batch.context is not None
    moved = replace(batch, context=(batch.context[0] + 1.0, batch.context[1]))
    assert not torch.equal(_gradient(moved), reference)


def test_the_first_window_steps_reward_and_continue_are_trained() -> None:
    """`_annotate_batch` marks only the context step `is_first`; step 1 keeps its targets."""
    batch = _batch()
    reference = _gradient(batch)
    for name, value in (("rewards", 40.0), ("dones", True)):
        tensor = getattr(batch, name).clone()
        tensor[:, 1] = value
        assert not torch.equal(_gradient(replace(batch, **{name: tensor})), reference), name


def test_a_reloaded_backbone_resumes_exactly() -> None:
    original = _backbone()
    for step in range(3):
        _learn(original, _batch(), seed=step)
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
    batch = _batch()
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
    batch = _batch(device=cuda)
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
    replay = DreamerReplay(capacity=256, length=LENGTH + 1, seed=0)
    actor = Actor(environment=environment, policy=acting_copy(backbone), replay=replay)
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



def test_an_acting_copy_takes_each_finished_step_mid_episode_and_keeps_its_latent() -> None:
    """`embodied/jax/agent.py` 243-247, 279-282: the next policy call after a step acts on it.

    The actor's hook before a decision loads the learner's last finished step,
    carries the episode's latent across, and returns while a step is held shut
    inside the learner rather than waiting it out.
    """
    backbone = _backbone()
    replay = DreamerReplay(capacity=256, length=LENGTH + 1, seed=0)
    actor = Actor(environment=_fake_environment(), policy=acting_copy(backbone), replay=replay)
    training = TrainingRun(
        actors=[actor],
        replay=replay,
        backbone=backbone,
        config=TrainingConfig(
            budget_decisions=1,
            warmup_sequences=2,
            batch_size=SMALL.batch_size,
            gradient_steps_per_decision=0.5,
            parameter_sync_decisions=1,
            exploration=ExplorationSchedule(
                epsilon_start=0.0, epsilon_end=0.0, anneal_decisions=1
            ),
        ),
    )
    acting = cast(DreamerBackbone, training.acting[actor.config.actor_id])
    before_decision = actor.before_decision
    assert before_decision is not None
    training.learner.publish()
    before_decision()
    _, latent = acting.act(_features(), acting.initial_state(), epsilon=0.0)
    kept = tuple(part.clone() for part in latent)

    inside, release = threading.Event(), threading.Event()
    learn = backbone.learn

    def held_shut(batch: SequenceBatch) -> Any:
        inside.set()
        release.wait()
        return learn(batch)

    backbone.learn = held_shut  # type: ignore[method-assign]
    stepping = threading.Thread(target=training.learner.learn, args=(_batch(),))
    stepping.start()
    assert inside.wait(5)
    deciding = threading.Thread(target=before_decision)
    deciding.start()
    deciding.join(5)
    waited = deciding.is_alive()
    release.set()
    stepping.join(5)

    assert not waited, "a decision waited out the learner's step in flight"
    assert acting.model_version == 0
    before_decision()
    assert acting.model_version == backbone.model_version == 1
    assert parameters_are_equal(acting.world_model, backbone.world_model)
    assert parameters_are_equal(acting.actor, backbone.actor)
    assert all(torch.equal(part, old) for part, old in zip(latent, kept, strict=True)), (
        "the swap disturbed the latent the episode carries"
    )
    action, _ = acting.act(_features(), latent, epsilon=0.0)
    assert action in (0, 1, 2)

# -- the game-time discount (ADR 0013) -----------------------------------------

#: Game time of each transition a window trains on: a purchase, a 17 s WAIT, a
#: purchase, 2 s, 1 s, and 1 s into the death that ends the window.
SPANS_MS = (0.0, 17000.0, 0.0, 2000.0, 1000.0, 1000.0)


def _timed_batch(*, wave_reward: float = 0.0) -> SequenceBatch:
    """Two windows whose transitions span `SPANS_MS`; the first ends its episode."""
    return _windows(
        _episode(game_ms=SPANS_MS, reward=wave_reward),
        _episode(LENGTH + 3, died=False, game_ms=(*SPANS_MS, 1000.0, 1000.0), reward=wave_reward),
    )


def _timed_sequences(*, wave_reward: float = 0.0) -> SequenceBatch:
    """`_timed_batch`'s transitions in stacked-dqn's replay: each step's is the one out of it."""
    def window(done: bool) -> ReplaySequence:
        sequence = _sequence(done=done)
        steps = tuple(
            replace(step, game_ms=span, reward=wave_reward)
            for step, span in zip(sequence.steps, SPANS_MS, strict=True)
        )
        return replace(sequence, steps=steps)

    return collate((window(True), window(False)), (1.0, 1.0))


def _continue_targets(
    backbone: DreamerBackbone, batch: SequenceBatch, monkeypatch: pytest.MonkeyPatch
) -> torch.Tensor:
    """The continue head's target in one update: the only [B, T] cross-entropy target."""
    seen: list[torch.Tensor] = []
    original = functional.binary_cross_entropy_with_logits

    def recording(logits: torch.Tensor, target: torch.Tensor, **options: object) -> torch.Tensor:
        if tuple(target.shape) == (batch.batch_size, LENGTH):
            seen.append(target.detach().clone())
        return original(logits, target, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(functional, "binary_cross_entropy_with_logits", recording)
    backbone.learn(batch)
    assert len(seen) == 1
    return seen[0]


def test_the_continue_target_carries_each_transitions_own_game_time_discount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """contdisc per transition: 1 for a purchase, 0.999 ** t for t s, 0 for a death.

    Dreamer's step t carries the transition into it; the window's last step
    in an episode that ends there is the stored terminal observation.
    """
    targets = _continue_targets(_backbone(), _timed_batch(), monkeypatch)
    ended, running = targets.tolist()
    expected = [1.0, 0.999**17, 1.0, 0.999**2, 0.999, 0.999]
    assert running == pytest.approx(expected, rel=1e-6)
    assert ended == pytest.approx([*expected[:-1], 0.0], rel=1e-6)


def _learn_rewards(learn: Any, batch: SequenceBatch, module: Any) -> torch.Tensor:
    """The per-transition rewards a learner's `learn` takes its return from, in replay's layout.

    Both backbones hand exactly these to `value_fit_correlation`.
    """
    seen: list[torch.Tensor] = []
    original = module.value_fit_correlation

    def recording(values: Any, mask: Any, rewards: torch.Tensor, *rest: Any, **options: Any) -> Any:
        seen.append(rewards.detach().clone())
        return original(values, mask, rewards, *rest, **options)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "value_fit_correlation", recording)
        learn(batch)
    return seen[0]


@pytest.mark.parametrize("survival", [True, False])
def test_dreamer_learns_from_exactly_stacked_dqns_reward(survival: bool) -> None:
    """One task reward (ADR 0013): (1 - d) * V_REF, or the wave change d * r, for the same spans."""
    from tower_rl.learning import dreamer, stacked_dqn, value_learning

    batch = _timed_batch(wave_reward=1.0)
    sequences = _timed_sequences(wave_reward=1.0)
    stacked = stacked_dqn.StackedDqnBackbone(
        config=stacked_dqn.StackedDqnConfig(
            history_length=1, discount_per_game_second=0.999, survival_time_reward=survival
        )
    )
    learner = _backbone(replace(SMALL, survival_time_reward=survival))

    ours = _learn_rewards(learner.learn, batch, dreamer)
    theirs = _learn_rewards(stacked.learn, sequences, stacked_dqn)
    # The same reward per transition. Dreamer's fit starts one later: its
    # window's first value is the one after the context's transition in.
    assert torch.equal(ours, theirs[:, 1:])
    discounts = 0.999 ** (torch.tensor(SPANS_MS, dtype=torch.float64) / 1000.0)
    expected = (1.0 - discounts) * value_learning.V_REF if survival else discounts
    assert torch.allclose(theirs[0].double(), expected, rtol=1e-6, atol=0.0)
    # A purchase earns no survival time; its wave change is not discounted.
    assert theirs[0, 0].item() == (0.0 if survival else 1.0)


def test_the_diagnostics_read_the_continue_head_and_the_decoded_mask() -> None:
    """Implied game time log(c) / log 0.999 against the stored, and the mask's two error rates."""
    backbone = _backbone()
    predicted = torch.tensor([[0.5, 0.999**2, 0.999**3, 0.3]])
    target = torch.tensor([[1.0, 0.999, 0.999**2, 0.0]])
    # Two real transitions and a death; the first step is not trained.
    transition = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    terminal = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    seconds = torch.tensor([[0.0, 1.0, 2.0, 5.0]])
    # Two steps, both allowing WAIT and 4; the decoded mask adds 9 to both
    # and drops 4 from the second. Of the 5 decoded-valid entries (WAIT, 4, 9;
    # WAIT, 9) 2 are invalid; of the 4 valid ones 1 is dropped.
    mask = torch.zeros(1, 2, ACTIONS, dtype=torch.bool)
    mask[..., [WAIT_INDEX, 4]] = True
    # Two classes per action, invalid then valid; the argmax is read.
    logits = torch.zeros(1, 2, ACTIONS, 2)
    logits[..., 0] = 1.0
    logits[0, 0, [4, 9], 1] = 2.0
    logits[0, 1, 9, 1] = 2.0

    checks = backbone._checks(
        torch.logit(predicted.double()).float(), target, seconds, terminal, transition,
        logits, mask,
    )
    measured = {name: value.item() for name, value in checks.items()}
    assert measured["dreamer_implied_dt_seconds"] == pytest.approx(2.5, rel=1e-4)
    # |2 - 1| / 1 and |3 - 2| / 2: the per-transition error the ratio of means hides.
    assert measured["dreamer_implied_dt_relative_error"] == pytest.approx(0.75, rel=1e-3)
    assert measured["dreamer_true_dt_seconds"] == pytest.approx(1.5)
    assert measured["dreamer_predicted_continue"] == pytest.approx(
        (0.999**2 + 0.999**3 + 0.3) / 3, rel=1e-6
    )
    assert measured["dreamer_true_continue"] == pytest.approx((0.999 + 0.999**2) / 3, rel=1e-6)
    assert measured["dreamer_mask_false_valid_rate"] == pytest.approx(2 / 5)
    assert measured["dreamer_mask_false_invalid_rate"] == pytest.approx(1 / 4)


def test_learning_reports_the_diagnostics_and_their_ratio() -> None:
    metrics = _backbone().learn(_timed_batch())
    assert set(metrics.diagnostics) == {
        "dreamer_implied_dt_seconds",
        "dreamer_implied_dt_relative_error",
        "dreamer_true_dt_seconds",
        "dreamer_implied_to_true_dt",
        "dreamer_predicted_continue",
        "dreamer_true_continue",
        "dreamer_mask_false_valid_rate",
        "dreamer_mask_false_invalid_rate",
    }
    assert metrics.diagnostics["dreamer_implied_to_true_dt"] == pytest.approx(
        metrics.diagnostics["dreamer_implied_dt_seconds"]
        / metrics.diagnostics["dreamer_true_dt_seconds"]
    )
    assert all(math.isfinite(value) for value in metrics.diagnostics.values())


# -- replay context: the window starts where acting was --------------------------


def _acted_episode(backbone: DreamerBackbone, count: int) -> EpisodeSteps:
    """`count` observations acted on in turn, each step's latent as the actor stores it."""
    policy = acting_copy(backbone)
    assert isinstance(policy, DreamerBackbone)
    state = policy.initial_state()
    observations = [_features(seed=0.37 * index % 1.0) for index in range(count)]
    actions, latents = [], []
    for features in observations:
        action, state = policy.act(features, state, epsilon=0.0)
        actions.append(action)
        latents.append(policy.replay_entry(state))
    return episode_steps(
        scalars=[o.scalars for o in observations],
        rows=[o.rows for o in observations],
        mask=[o.mask for o in observations],
        action=[*actions[:-1], 0],
        reward=[0.0] * count,
        terminal=[False] * count,
        game_ms=[0.0, *[1000.0] * (count - 1)],
        deter=[deter for deter, _ in latents],
        stoch=[stoch for _, stoch in latents],
    )


def test_a_window_restores_the_latent_acting_reached_at_its_context_step() -> None:
    """`_apply_replay_context`: the window's first step is the actor's, not a fresh start.

    Its deterministic state is a function of the context's latent and action
    alone, so it is the one the actor reached there; the later steps depend on
    the stochastic draws, which acting and learning take apart.
    """
    backbone = _backbone()
    episode = _acted_episode(backbone, 2 * LENGTH)
    replay = DreamerReplay(capacity=64, length=LENGTH + 1)
    replay.add("a", _metadata(), episode)
    starts = (3, 4)
    runs = tuple(((0, start, LENGTH + 1),) for start in starts)
    batch = DreamerSample(runs, replay._assemble(runs)).batch()
    metrics = backbone.learn(batch)
    assert metrics.latents is not None
    deter, _ = metrics.latents
    for window, start in enumerate(starts):
        assert torch.allclose(
            deter[window, 0], torch.from_numpy(episode.deter[start + 1]), atol=1e-5
        )


# -- the actor's stream -----------------------------------------------------------


def _fake_environment(damage_per_second: float = 50.0) -> InstrumentedRunEnvironment:
    return InstrumentedRunEnvironment(
        port=FakeRunPort(damage_per_second=damage_per_second),
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
    )


def test_an_episode_is_stored_in_dreamers_layout_ending_on_its_terminal_observation() -> None:
    """The driver's layout: each step the transition into it, and the death's observation last.

    The official driver stores the environment's terminal observation as a
    step of its own, with the action masked to zero; the transition into it
    carries the death.
    """
    environment = _fake_environment()
    backbone = _backbone()
    replay = DreamerReplay(capacity=256, length=LENGTH + 1, seed=0)
    actor = Actor(environment=environment, policy=acting_copy(backbone), replay=replay)
    result = actor.run_episode()
    (episode,) = replay._episodes.values()
    steps = episode.steps
    assert len(steps) == result.summary.decisions + 1
    assert steps.terminal.tolist() == [False] * result.summary.decisions + [True]
    assert steps.action[-1] == 0 and steps.reward[0] == 0.0 and steps.game_ms[0] == 0.0
    assert steps.reward[1:].sum() == pytest.approx(result.total_reward)
    # Every acted step's latent as acting reached it (the first, from the zero
    # state and no action, is itself zero); the final one is never read.
    assert (numpy.abs(steps.deter[1:-1]).sum(-1) > 0).all()
    assert not steps.deter[-1].any()


def test_an_episodes_stream_stops_at_its_first_inadmissible_transition() -> None:
    """What an inadmissible transition led to may not be valid; its own observation is kept."""
    environment = _fake_environment()
    environment.reset()
    actor = Actor(environment=environment, policy=acting_copy(_backbone()), replay=None)
    replay = DreamerReplay(capacity=256, length=LENGTH + 1, seed=0)
    steps = [
        ReplayStep(
            features=_features(seed=0.1 * index),
            action_index=1,
            reward=float(index),
            done=False,
            admissible=index != 3,
            game_ms=1000.0,
        )
        for index in range(5)
    ]
    latents = [(numpy.ones(SMALL.deter), numpy.zeros(SMALL.stoch))] * 5
    summary = environment.summarize(TerminationOutcome.OBSERVATION_INVALID)
    assert actor._emit_stream(replay, steps, latents, None, summary) == (1, 1)
    (episode,) = replay._episodes.values()
    assert episode.steps.reward.tolist() == [0.0, 0.0, 1.0, 2.0]
    assert episode.steps.action.tolist() == [1, 1, 1, 0]
    assert replay.stats.rejections_by_reason == {"inadmissible_transition": 1}
    # Inadmissible at once: nothing to keep.
    first = [replace(steps[0], admissible=False)]
    assert actor._emit_stream(replay, first, latents, None, summary) == (1, 0)
