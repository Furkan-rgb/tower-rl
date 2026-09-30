"""R2D2's network and learn step against the paper, Acme and rlax (`learning/r2d2.py`)."""

from __future__ import annotations

import copy
import dataclasses
import math
import time

import numpy
import pytest
import torch

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import Backbone, SequenceBatch, acting_copy
from tower_rl.learning.network import dueling_masked_q
from tower_rl.learning.r2d2 import (
    R2D2Backbone,
    R2D2Config,
    signed_hyperbolic,
    signed_parabolic,
    trace_targets,
)
from tower_rl.learning.r2d2_replay import (
    R2D2_BURN_IN,
    R2D2_ITEM_LENGTH,
    R2D2_SEQUENCE_PERIOD,
    R2D2_STATE_SIZE,
    R2D2Replay,
    sequence_priority,
)
from tower_rl.learning.replay import SequenceMetadata
from tower_rl.learning.step_arrays import StepArrays
from tower_rl.learning.value_learning import V_REF

GAMMA = 0.999
ACTIONS = len(RUN_ACTIONS)
METADATA = SequenceMetadata(
    episode_id="e", actor_id="a", profile_id="p",
    observation_schema="observation-v1", action_schema="run-action-v1",
    reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
)


def _backbone(**overrides: object) -> R2D2Backbone:
    settings: dict[str, object] = {"discount_per_game_second": GAMMA, "seed": 0, **overrides}
    return R2D2Backbone(R2D2Config(**settings))  # type: ignore[arg-type]


def _episode(count: int, seed: int = 0) -> StepArrays:
    """`count` random steps; WAIT always valid, the last step terminal with an empty mask."""
    generator = numpy.random.default_rng(seed)
    mask = generator.random((count, ACTIONS)) < 0.3
    mask[:, 0] = True
    mask[-1] = False
    action = numpy.array(
        [generator.choice(numpy.flatnonzero(row)) if row.any() else 0 for row in mask]
    )
    game_ms = generator.choice([0.0, 500.0, 1850.0, 4000.0], size=count).astype(numpy.float32)
    game_ms[0] = 0.0
    return StepArrays(
        scalars=generator.normal(size=(count, SCALAR_COUNT)).astype(numpy.float32),
        rows=generator.normal(size=(count, ROW_COUNT * ROW_WIDTH)).astype(numpy.float32),
        mask=mask,
        action=action.astype(numpy.int64),
        reward=numpy.zeros(count, numpy.float32),
        terminal=numpy.arange(count) == count - 1,
        game_ms=game_ms,
    )


def _states(count: int, seed: int = 0) -> numpy.ndarray:
    grid = math.ceil((count - 1) / R2D2_SEQUENCE_PERIOD)
    states = numpy.random.default_rng(seed).normal(size=(grid, 2, R2D2_STATE_SIZE)) * 0.1
    states[0] = 0.0
    return states.astype(numpy.float32)


def _batch(count: int, size: int = 2, seed: int = 0) -> SequenceBatch:
    """`size` draws from a replay holding one episode of `count` steps."""
    replay = R2D2Replay(capacity=100, seed=seed)
    replay.add(METADATA, _episode(count, seed), _states(count, seed))
    return replay.sample(size).batch()


# -- value rescaling -------------------------------------------------------------


def test_h_and_its_inverse_round_trip_and_match_rlax() -> None:
    x = torch.tensor(
        [-1e4, -50.0, -1.0, -1e-3, 0.0, 1e-3, 0.5, 9.51, 1e3, 1e5], dtype=torch.float64
    )
    assert torch.allclose(signed_parabolic(signed_hyperbolic(x)), x, rtol=1e-9, atol=1e-9)
    assert torch.allclose(signed_hyperbolic(signed_parabolic(x)), x, rtol=1e-9, atol=1e-9)
    # sqrt(10.51) - 1 + 0.00951: the survival return's bound V_REF, rescaled.
    assert float(signed_hyperbolic(torch.tensor(9.51, dtype=torch.float64))) == pytest.approx(
        2.2514, abs=1e-4
    )


# -- targets -----------------------------------------------------------------------


def _reference_targets(
    online_q: torch.Tensor,
    target_q: torch.Tensor,
    mask: torch.Tensor,
    game_ms: list[float],
    dones: list[bool],
    real: int,
    n: int,
) -> list[float]:
    """The rescaled n-step double-Q target of every step of one trace, one step at a time."""
    steps = len(game_ms)
    d_into = [GAMMA ** (ms / 1000.0) for ms in game_ms]
    reward = [(1.0 - d_into[t + 1]) * V_REF if t + 1 < real else 0.0 for t in range(steps - 1)]
    discount = [
        d_into[t + 1] * (0.0 if dones[t + 1] else 1.0) if t + 1 < real else 1.0
        for t in range(steps - 1)
    ]
    values = []
    for k in range(steps):
        valid = [a for a in range(mask.shape[-1]) if mask[k, a]]
        best = max(valid, key=lambda a: float(online_q[k, a])) if valid else None
        values.append(0.0 if best is None else float(target_q[k, best]))
    targets = []
    for t in range(steps - 1):
        total, factor = 0.0, 1.0
        for k in range(n):
            if t + k < steps - 1:
                total += factor * reward[t + k]
                factor *= discount[t + k]
        bootstrap = values[min(t + n, real - 1)]
        total += factor * float(signed_parabolic(torch.tensor(bootstrap, dtype=torch.float64)))
        targets.append(float(signed_hyperbolic(torch.tensor(total, dtype=torch.float64))))
    return targets


@pytest.mark.parametrize("real", [9, 7])
def test_the_n_step_game_time_target_matches_a_hand_computation(real: int) -> None:
    """Varied game time, a purchase (0 ms), a terminal inside the window, and rlax's end.

    With 9 real steps the trace runs to its end: the last steps' returns are
    shortened to bootstrap from the last value (rlax). With 7 the episode dies
    into step 6 and the rest is padding.
    """
    steps = 9
    game_ms = [0.0, 2000.0, 0.0, 1500.0, 3000.0, 1000.0, 500.0, 2500.0, 700.0]
    dones = [False] * steps
    padding = [False] * steps
    if real < steps:
        dones[real - 1] = True
        for index in range(real, steps):
            game_ms[index] = 0.0
            padding[index] = True
    generator = torch.Generator().manual_seed(0)
    mask = torch.rand(steps, ACTIONS, generator=generator) < 0.4
    mask[:, 0] = True
    if real < steps:
        mask[real - 1 :] = False
    advantages = torch.randn(steps, ACTIONS, generator=generator)
    online_q = dueling_masked_q(torch.randn(steps, 1, generator=generator), advantages, mask)
    target_q = dueling_masked_q(
        torch.randn(steps, 1, generator=generator),
        torch.randn(steps, ACTIONS, generator=generator),
        mask,
    )

    targets, valid = trace_targets(
        online_q[None],
        target_q[None],
        mask[None],
        torch.tensor([game_ms]),
        torch.tensor([dones]),
        torch.tensor([padding]),
        discount_per_game_second=GAMMA,
        n=5,
    )

    expected = _reference_targets(online_q, target_q, mask, game_ms, dones, real, n=5)
    assert valid[0].tolist() == [t + 1 < real for t in range(steps - 1)]
    for t in range(real - 1):
        assert float(targets[0, t]) == pytest.approx(expected[t], rel=1e-5, abs=1e-6)


def test_the_bootstrap_argmax_skips_invalid_actions_and_an_empty_state_is_worth_zero() -> None:
    """The online network's favourite is invalid; the target values its best valid one."""
    mask = torch.tensor([[[True, True, False], [True, True, False], [False, False, False]]])
    online_q = dueling_masked_q(
        torch.zeros(1, 3, 1),
        torch.tensor([[[0.0, 0.0, 9.0], [1.0, 5.0, 9.0], [1.0, 2.0, 3.0]]]),
        mask,
    )
    target_q = dueling_masked_q(
        torch.zeros(1, 3, 1),
        torch.tensor([[[0.0, 0.0, 0.0], [3.0, -7.0, 99.0], [4.0, 5.0, 6.0]]]),
        mask,
    )
    unmoving = torch.zeros(1, 3, dtype=torch.bool)
    targets, _ = trace_targets(
        online_q, target_q, mask, torch.zeros(1, 3), unmoving, unmoving,
        discount_per_game_second=GAMMA, n=1,
    )
    # Step 1: online Q is (-2, 2, -inf), so action 1, which the target values
    # at -5 - not the target's own best, 5, nor anything of the invalid 2.
    assert float(targets[0, 0]) == pytest.approx(-5.0, abs=1e-5)
    # Step 2 has no valid action: worth 0, not -inf or NaN.
    assert not torch.isfinite(target_q[0, 2]).any()
    assert float(targets[0, 1]) == 0.0


def test_dueling_centres_over_valid_actions_only() -> None:
    mask = torch.tensor([True, True, True, False])
    q = dueling_masked_q(torch.tensor([5.0]), torch.tensor([1.0, 2.0, 3.0, 100.0]), mask)
    assert q[:3].tolist() == [4.0, 5.0, 6.0]
    assert q[3] == float("-inf")
    assert float(q[:3].mean()) == 5.0


# -- network and learn step ---------------------------------------------------------


def test_the_network_is_initialised_as_haiku_initialises_it() -> None:
    online = _backbone().online
    core = dict(online.core.named_parameters())
    hidden = online.core.hidden_size
    assert torch.equal(core["bias_ih_l0"][hidden : 2 * hidden], torch.ones(hidden))
    assert core["bias_ih_l0"][:hidden].abs().sum() == 0
    assert core["bias_hh_l0"].abs().sum() == 0 and not core["bias_hh_l0"].requires_grad
    torso = online.torso[0]
    weight = torso.weight.detach()
    assert torso.bias.abs().sum() == 0
    bound = 2.0 / math.sqrt(torso.in_features)
    assert float(weight.abs().max()) <= bound
    assert float(weight.std()) == pytest.approx(0.88 / math.sqrt(torso.in_features), rel=0.05)


def test_the_optimiser_is_adam_at_the_papers_values_without_decay() -> None:
    backbone = _backbone()
    optimizer = backbone.optimizer
    assert type(optimizer) is torch.optim.Adam
    (group,) = optimizer.param_groups
    assert group["lr"] == 1e-4 and group["eps"] == 1e-3
    assert group["betas"] == (0.9, 0.999) and group["weight_decay"] == 0
    trained = [p for p in backbone.online.parameters() if p.requires_grad]
    assert len(group["params"]) == len(trained)


def test_burn_in_steps_receive_no_gradient() -> None:
    backbone = _backbone()
    batch = _batch(200)
    batch.scalars.requires_grad_(True)
    backbone.learn(batch)
    gradient = batch.scalars.grad
    assert gradient is not None
    assert gradient[:, :R2D2_BURN_IN].abs().sum() == 0
    assert gradient[:, R2D2_BURN_IN:].abs().sum() > 0
    assert all(p.grad is None for p in backbone.target.parameters())


def test_pad_steps_and_the_terminal_step_touch_neither_loss_nor_priority() -> None:
    """A 60-step episode: 19 targets (steps 40-58); step 59 is terminal, 60-120 padding."""
    batch = _batch(60)
    assert batch.padding[:, 60:].all() and not batch.padding[:, :60].any()
    first = _backbone()
    second = copy.deepcopy(first)
    metrics = first.learn(batch)
    assert [len(errors) for errors in metrics.td_errors] == [19, 19]

    noise = torch.randn_like(batch.scalars[:, 59:])
    changed = dataclasses.replace(
        batch,
        scalars=torch.cat((batch.scalars[:, :59], noise), dim=1),
        rows=torch.cat((batch.rows[:, :59], torch.randn_like(batch.rows[:, 59:])), dim=1),
        actions=torch.cat(
            (batch.actions[:, :59], torch.full_like(batch.actions[:, 59:], 3)), dim=1
        ),
    )
    again = second.learn(changed)
    assert again.weighted_loss == pytest.approx(metrics.weighted_loss, rel=1e-6)
    numpy.testing.assert_allclose(
        numpy.array(again.td_errors), numpy.array(metrics.td_errors), rtol=1e-5
    )


def test_an_items_priority_mixes_the_max_and_mean_of_its_valid_steps() -> None:
    metrics = _backbone().learn(_batch(150))
    for errors in metrics.td_errors:
        # A 150-step episode's items start at 0 (121 steps) and 40 (110 steps).
        assert len(errors) in (80, 69)
        assert all(math.isfinite(error) for error in errors)
        magnitudes = numpy.abs(errors)
        assert sequence_priority(errors) == pytest.approx(
            0.9 * magnitudes.max() + 0.1 * magnitudes.mean()
        )


def test_the_target_is_copied_at_step_2500_and_not_before() -> None:
    backbone = _backbone()
    initial = copy.deepcopy(backbone.target.state_dict())
    batch = _batch(130)
    backbone._steps = 2_498
    backbone.learn(batch)
    assert backbone.model_version == 2_499
    assert all(torch.equal(initial[k], v) for k, v in backbone.target.state_dict().items())
    backbone.learn(batch)
    assert backbone.model_version == 2_500
    online = backbone.online.state_dict()
    assert all(torch.equal(online[k], v) for k, v in backbone.target.state_dict().items())
    assert not all(torch.equal(initial[k], v) for k, v in backbone.target.state_dict().items())


def test_learning_on_one_small_batch_lowers_its_loss() -> None:
    backbone = _backbone()
    batch = _batch(200)
    losses = [backbone.learn(batch).weighted_loss for _ in range(8)]
    assert losses[-1] < 0.5 * losses[0]


def test_one_learn_step_at_64_by_121_on_the_cpu() -> None:
    """Timed for the record; the bound only catches a pathological slowdown.

    The CPU learner is not the one a run uses (docs spec: 28-36 ms on the
    GPU); on the CPU the LSTM's forward and backward dominate.
    """
    backbone = _backbone()
    replay = R2D2Replay(capacity=100, seed=0)
    replay.add(METADATA, _episode(522), _states(522))
    batch = replay.sample(64).batch()
    assert batch.scalars.shape[:2] == (64, R2D2_ITEM_LENGTH)
    began = time.perf_counter()
    backbone.learn(batch)
    elapsed = time.perf_counter() - began
    print(f"R2D2 learn step 64x121 on CPU ({torch.get_num_threads()} threads): {elapsed:.2f} s")
    assert elapsed < 60.0


# -- acting --------------------------------------------------------------------------


def _features(steps: StepArrays, index: int) -> StateFeatures:
    return StateFeatures(
        scalars=tuple(steps.scalars[index].tolist()),
        rows=tuple(steps.rows[index].tolist()),
        mask=tuple(bool(flag) for flag in steps.mask[index]),
    )


def test_acting_carries_the_state_the_learner_unrolls_from() -> None:
    """The state an actor hands replay before decision 40 is the learner's burn-in end state.

    The actor acts through an episode; its stored states and actions go into
    replay; the learner's online unroll over the item at 0's first 40 steps,
    from its zero state, reaches the state stored for the item at 40.
    """
    backbone = _backbone()
    actor = acting_copy(backbone, exploration_seed=1)
    assert isinstance(actor, R2D2Backbone)
    episode = _episode(122, seed=3)
    state = actor.initial_state()
    entries, actions = [], []
    for decision in range(len(episode) - 1):
        if decision % R2D2_SEQUENCE_PERIOD == 0:
            entries.append(actor.replay_entry(state))
        action, state = actor.act(_features(episode, decision), state, epsilon=0.3)
        assert episode.mask[decision, action]
        actions.append(action)
        with pytest.raises(ValueError, match="after_transition"):
            actor.act(_features(episode, 0), state, epsilon=0.0)
        state = actor.after_transition(state, float(episode.game_ms[decision + 1]))
    acted = dataclasses.replace(episode, action=numpy.array([*actions, 0], numpy.int64))
    replay = R2D2Replay(capacity=10, seed=0)
    replay.add(METADATA, acted, numpy.stack(entries))
    arrays = replay.sample(32).arrays
    at_zero = int(numpy.flatnonzero(arrays["first"][:, 0])[0])
    at_forty = int(numpy.flatnonzero(~arrays["first"][:, 0])[0])

    rewards = (1.0 - GAMMA ** (torch.as_tensor(arrays["game_ms"][at_zero]) / 1000.0)) * V_REF
    with torch.no_grad():
        _, (h, c) = backbone.online(
            torch.as_tensor(arrays["scalars"][at_zero, None, :R2D2_BURN_IN]),
            torch.as_tensor(arrays["rows"][at_zero, None, :R2D2_BURN_IN]).view(
                1, R2D2_BURN_IN, ROW_COUNT, ROW_WIDTH
            ),
            torch.as_tensor(arrays["mask"][at_zero, None, :R2D2_BURN_IN]),
            torch.as_tensor(arrays["previous_action"][at_zero, None, :R2D2_BURN_IN]),
            rewards[None, :R2D2_BURN_IN].float(),
            (torch.zeros(1, 1, R2D2_STATE_SIZE), torch.zeros(1, 1, R2D2_STATE_SIZE)),
        )
    assert torch.allclose(h[0, 0], torch.as_tensor(arrays["h"][at_forty]), atol=1e-5)
    assert torch.allclose(c[0, 0], torch.as_tensor(arrays["c"][at_forty]), atol=1e-5)


def test_the_published_network_state_carries_no_optimizer_state() -> None:
    learner: Backbone = _backbone()
    assert isinstance(learner, R2D2Backbone)
    learner.learn(_batch(130))
    published = learner.network_state_dict()
    assert set(published) == {"online", "steps"}
    actor = acting_copy(_backbone(seed=5))
    assert isinstance(actor, R2D2Backbone)
    actor.load_network_state_dict(published)
    assert actor.model_version == 1
    for key, value in learner.online.state_dict().items():
        assert torch.equal(actor.online.state_dict()[key], value)


def test_a_checkpoint_resumes_the_learner_exactly() -> None:
    batch = _batch(130)
    learner = _backbone()
    learner.learn(batch)
    resumed = _backbone(seed=7)
    resumed.load_state_dict(copy.deepcopy(learner.state_dict()))
    assert resumed.model_version == 1
    assert learner.learn(batch).weighted_loss == pytest.approx(
        resumed.learn(batch).weighted_loss, rel=1e-6
    )
