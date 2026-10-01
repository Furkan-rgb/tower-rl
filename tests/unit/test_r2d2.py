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
    GRADIENT_CLIP_NORM,
    R2D2Backbone,
    R2D2Config,
    R2D2Network,
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


@pytest.mark.parametrize(("real", "died"), [(9, False), (7, True), (7, False)])
def test_the_n_step_game_time_target_matches_a_hand_computation(real: int, died: bool) -> None:
    """Varied game time, a purchase (0 ms), a terminal inside the window, and rlax's end.

    With 9 real steps the trace runs to its end: the last steps' returns are
    shortened to bootstrap from the last value (rlax). With 7 that died, the
    episode dies into step 6 and the rest is padding. With 7 that did not,
    the stream was cut there: step 6 is a live observation, and the returns
    running past it bootstrap from its value, not from a pad's.
    """
    steps = 9
    game_ms = [0.0, 2000.0, 0.0, 1500.0, 3000.0, 1000.0, 500.0, 2500.0, 700.0]
    dones = [False] * steps
    padding = [False] * steps
    if real < steps:
        dones[real - 1] = died
        for index in range(real, steps):
            game_ms[index] = 0.0
            padding[index] = True
    generator = torch.Generator().manual_seed(0)
    mask = torch.rand(steps, ACTIONS, generator=generator) < 0.4
    mask[:, 0] = True
    if real < steps:
        mask[real if not died else real - 1 :] = False
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


def test_a_backbones_seed_fixes_its_parameters_and_does_not_reseed_torch() -> None:
    def built(global_seed: int, seed: int) -> tuple[R2D2Backbone, torch.Tensor]:
        torch.manual_seed(global_seed)
        backbone = _backbone(seed=seed)
        return backbone, torch.rand(3)

    first, after_first = built(1, seed=7)
    second, after_second = built(2, seed=7)
    other, _ = built(1, seed=8)
    for name, value in first.online.state_dict().items():
        assert torch.equal(second.online.state_dict()[name], value), name
    assert any(
        not torch.equal(other.online.state_dict()[name], value)
        for name, value in first.online.state_dict().items()
    ), "another seed must start elsewhere"
    assert not torch.equal(after_first, after_second), "seeding reseeded torch's global stream"


def test_the_row_identity_embedding_takes_haikus_embed_default() -> None:
    # hk.Embed: truncated normal at std 1, cut at 2 std. Not torch's N(0, 1).
    table = R2D2Network().trunk.identity.weight.detach()
    assert float(table.abs().max()) <= 2.0
    assert float(table.std()) == pytest.approx(0.88, rel=0.05)
    assert abs(float(table.mean())) < 0.05


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
    assert metrics.priorities is not None and metrics.priorities.shape == (2,)

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
    assert again.priorities is not None
    numpy.testing.assert_allclose(again.priorities, metrics.priorities, rtol=1e-5)


def test_an_item_with_no_valid_trace_step_gets_priority_zero() -> None:
    """A 41-step episode is one item whose trace is its terminal step alone."""
    metrics = _backbone().learn(_batch(41))
    assert metrics.priorities is not None and metrics.priorities.tolist() == [0.0, 0.0]
    assert metrics.weighted_loss == 0.0 and metrics.taken_q_max is None


def test_the_gradient_norm_is_clipped_at_40() -> None:
    """Ape-X's clip, through Table 2: the step applies the clipped gradient."""
    backbone = _backbone()
    batch = _batch(200)
    metrics = backbone.learn(dataclasses.replace(batch, weights=batch.weights * 1e4))
    assert metrics.gradient_norm > GRADIENT_CLIP_NORM
    applied = torch.nn.utils.get_total_norm(
        [p.grad for p in backbone.online.parameters() if p.grad is not None]
    )
    assert float(applied) == pytest.approx(GRADIENT_CLIP_NORM, rel=1e-4)


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
    assert losses[-1] < 0.75 * losses[0]


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


# -- the learn step's wiring, against an independent reference --------------------


def _wiring_batch() -> SequenceBatch:
    """Three items built by hand, each exercising one path of `learn`.

    Item 0 is real throughout; item 1 dies into step 10 (padding after); item
    2 is a stream cut at step 11 - real, alive, with no done - and padded
    after. The stored states are non-zero, the IS weights unequal, the wave
    `reward` non-zero (the network must never see it), the game times long
    enough that the survival reward is far outside tanh's linear range, and
    the previous actions are not the actions shifted.
    """
    generator = torch.Generator().manual_seed(11)
    size, length, burn_in = 3, 14, 4
    padding = torch.zeros(size, length, dtype=torch.bool)
    padding[1, 11:] = True
    padding[2, 12:] = True
    dones = torch.zeros(size, length, dtype=torch.bool)
    dones[1, 10] = True
    mask = torch.rand(size, length, ACTIONS, generator=generator) < 0.3
    mask[..., 0] = True
    mask[1, 10:] = False
    mask[2, 12:] = False
    choices = torch.tensor([0.0, 500.0, 30_000.0, 900_000.0])
    game_ms = choices[torch.randint(0, 4, (size, length), generator=generator)]
    game_ms[padding] = 0.0

    def valid_actions() -> torch.Tensor:
        scores = torch.rand(size, length, ACTIONS, generator=generator)
        return torch.where(mask, scores, -1.0).argmax(dim=-1)

    return SequenceBatch(
        scalars=torch.randn(size, length, SCALAR_COUNT, generator=generator),
        rows=torch.randn(size, length, ROW_COUNT, ROW_WIDTH, generator=generator),
        mask=mask,
        actions=valid_actions(),
        rewards=torch.rand(size, length, generator=generator) * 3.0 + 1.0,
        dones=dones,
        padding=padding,
        game_ms=game_ms,
        weights=torch.tensor([1.0, 0.3, 0.6]),
        burn_in=burn_in,
        context=(
            torch.randn(size, R2D2_STATE_SIZE, generator=generator) * 2.0,
            torch.randn(size, R2D2_STATE_SIZE, generator=generator) * 2.0,
        ),
        previous_actions=valid_actions(),
    )


def _reference_q(
    network: R2D2Network,
    batch: SequenceBatch,
    span: slice,
    state: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """R2D2's network written out from its parts, not through `R2D2Network.forward`."""
    encoded_rows, encoded_scalars = network.trunk.encode(
        batch.scalars[:, span], batch.rows[:, span]
    )
    torso_linear = network.torso[0]
    assert isinstance(torso_linear, torch.nn.Linear)
    torso = torch.relu(torso_linear(torch.cat((encoded_rows.flatten(2), encoded_scalars), -1)))
    discounts = GAMMA ** (batch.game_ms[:, span].double() / 1000.0)
    previous_reward = ((1.0 - discounts) * V_REF).float()
    assert batch.previous_actions is not None
    embedded = torch.cat(
        (
            torso,
            torch.nn.functional.one_hot(batch.previous_actions[:, span], ACTIONS).float(),
            torch.tanh(previous_reward).unsqueeze(-1),
        ),
        -1,
    )
    core, after = network.core(embedded, state)
    q = dueling_masked_q(network.value(core), network.advantage(core), batch.mask[:, span])
    return q, after


def _reference_loss(
    backbone: R2D2Backbone, batch: SequenceBatch, n: int = 5
) -> tuple[float, list[float]]:
    """The loss and priorities of one learn step, computed one item and one step at a time."""
    online, target = copy.deepcopy(backbone.online), copy.deepcopy(backbone.target)
    burn, length = batch.burn_in, batch.scalars.shape[1]
    assert batch.context is not None
    start = (batch.context[0][None], batch.context[1][None])
    with torch.no_grad():
        target_q, _ = _reference_q(target, batch, slice(0, length), start)
        _, burnt = _reference_q(online, batch, slice(0, burn), start)
        online_q, _ = _reference_q(online, batch, slice(burn, length), burnt)
    target_q = target_q[:, burn:]
    losses, priorities = [], []
    for item in range(batch.batch_size):
        real = int((~batch.padding[item, burn:]).sum())
        game_ms = batch.game_ms[item, burn:].tolist()
        dones = batch.dones[item, burn:].tolist()
        mask = batch.mask[item, burn:]
        targets = _reference_targets(
            online_q[item], target_q[item], mask, game_ms, dones, real, n
        )
        errors = []
        for t in range(real - 1):
            action = int(batch.actions[item, burn + t])
            errors.append(targets[t] - float(online_q[item, t, action]))
        magnitudes = numpy.abs(errors) if errors else numpy.zeros(1)
        losses.append(float(batch.weights[item]) * 0.5 * float(numpy.square(errors).sum()))
        priorities.append(0.9 * magnitudes.max() + 0.1 * magnitudes.mean())
    return float(numpy.mean(losses)), priorities


def test_the_learn_step_is_wired_as_the_reference_computes_it() -> None:
    """Stored state for both networks, the online burn-in, OAR inputs, h, IS weights, 0.5."""
    backbone = _backbone()
    batch = _wiring_batch()
    expected_loss, expected_priorities = _reference_loss(backbone, batch)
    metrics = backbone.learn(batch)
    assert metrics.weighted_loss == pytest.approx(expected_loss, rel=1e-5)
    assert metrics.priorities is not None
    assert metrics.priorities.tolist() == pytest.approx(expected_priorities, rel=1e-5)


def test_the_vectorised_value_fit_is_the_shared_one() -> None:
    """`_value_fit` against `value_learning.value_fit_correlation`'s loop over time."""
    from tower_rl.learning.r2d2 import _value_fit
    from tower_rl.learning.value_learning import value_fit_correlation

    generator = torch.Generator().manual_seed(3)
    size, time_steps = 4, 12
    values = torch.randn(size, time_steps, ACTIONS, generator=generator)
    mask = torch.rand(size, time_steps, ACTIONS, generator=generator) < 0.5
    mask[..., 0] = True
    mask[0, 7:] = False
    rewards = torch.rand(size, time_steps, generator=generator, dtype=torch.float64)
    discounts = torch.rand(size, time_steps, generator=generator, dtype=torch.float64)
    dones = torch.zeros(size, time_steps, dtype=torch.bool)
    dones[0, 6] = dones[1, 9] = dones[2, 3] = True
    valid = torch.ones(size, time_steps, dtype=torch.bool)
    valid[0, 7:] = False
    correlation, count = _value_fit(values, mask, rewards, discounts, dones, valid)
    expected = value_fit_correlation(
        values, mask, rewards.float(), dones, valid.float(), discounts=discounts
    )
    assert expected is not None and int(count) == 7 + 10 + 4
    assert float(correlation) == pytest.approx(expected, rel=1e-5)


# -- acting ----------------------------------------------------------------------


def _masked(valid: tuple[int, ...]) -> StateFeatures:
    """Features with random scalars and rows, and exactly the actions `valid` offered."""
    episode = _episode(2, seed=11)
    return StateFeatures(
        scalars=tuple(episode.scalars[0].tolist()),
        rows=tuple(episode.rows[0].tolist()),
        mask=tuple(index in valid for index in range(ACTIONS)),
    )


def test_acting_stays_inside_the_mask_greedy_and_exploring() -> None:
    backbone = _backbone()
    features = _masked((0, 5, 9))
    state = backbone.initial_state()
    for epsilon in (0.0, 0.5, 1.0):
        for _ in range(60):
            action, _ = backbone.act(features, state, epsilon=epsilon)
            assert action in (0, 5, 9), epsilon


def test_acting_refuses_a_state_with_no_valid_action() -> None:
    backbone = _backbone()
    with pytest.raises(ValueError, match="no action is available"):
        backbone.act(_masked(()), backbone.initial_state(), epsilon=0.0)


def test_acting_explores_epsilon_greedily_over_the_valid_actions() -> None:
    backbone = _backbone()
    valid = (0, 5, 9, 20)
    features, state = _masked(valid), backbone.initial_state()
    greedy, _ = backbone.act(features, state, epsilon=0.0)
    assert {backbone.act(features, state, epsilon=0.0)[0] for _ in range(20)} == {greedy}

    draws = 1000
    random_actions = [backbone.act(features, state, epsilon=1.0)[0] for _ in range(draws)]
    assert set(random_actions) == set(valid)
    assert all(abs(random_actions.count(a) / draws - 0.25) < 0.06 for a in valid)

    # At epsilon a draw leaves the greedy action only when exploring picks another:
    # epsilon * (1 - 1/4) of the time.
    moved = sum(backbone.act(features, state, epsilon=0.4)[0] != greedy for _ in range(draws))
    assert abs(moved / draws - 0.4 * 0.75) < 0.06
