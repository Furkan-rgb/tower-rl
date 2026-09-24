"""BBF's components on the stacked-DQN backbone, against the official code.

The reference is google-research/bigger_better_faster: `bbf/configs/BBF.gin`
and `bbf/agents/spr_agent.py`. Where a number is pinned here it is the number
that code produces; `docs/solution.md` "BBF recipe" lists each component.
"""

from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path
from typing import Any

import numpy
import pytest
import torch
from fakes.backbone_equality import parameters_are_equal

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.backbone import SequenceBatch, acting_copy, collate
from tower_rl.learning.checkpoint import (
    Checkpoint,
    CheckpointIdentity,
    TrainingProgress,
    load,
    save,
)
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork
from tower_rl.learning.replay import ReplaySequence, ReplayStep, SequenceMetadata
from tower_rl.learning.stacked_dqn import (
    PERTURB_FACTOR,
    SHRINK_FACTOR,
    StackedDqnBackbone,
    StackedDqnConfig,
    reset_seed,
)
from tower_rl.learning.training import Learner

SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)

#: BBF's own settings, as `BBF.gin` states them.
BBF = {
    "n_step": 10,
    "n_step_final": 3,
    "n_step_anneal_steps": 10_000,
    "discount_initial": 0.97,
    "discount": 0.997,
    "learning_rate": 1e-4,
    "weight_decay": 0.1,
    "weight_decay_on_vectors": False,
    "adam_eps": 1.5e-4,
    "target_ema_decay": 0.995,
}


# --- The official scheduler, verbatim but for names (spr_agent.py 311-350) ----


def official_exponential_decay_scheduler(
    decay_period: float,
    warmup_steps: int,
    initial_value: float,
    final_value: float,
    reverse: bool = False,
) -> Any:
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = numpy.log(initial_value)
    end = numpy.log(final_value)

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step: int) -> float:
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = numpy.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = numpy.exp(new_value)
        if reverse:
            new_value = 1 - new_value
        return float(new_value)

    return scheduler


def official_n_step(step: int) -> int:
    """`update_horizon_scheduler` (spr_agent.py 1198-1204) at BBF.gin's values."""
    schedule = official_exponential_decay_scheduler(10_000, 0, 1, 3 / 10)
    return int(numpy.round(schedule(step) * 10))


def official_gamma(step: int) -> float:
    """`gamma_scheduler` (spr_agent.py 1254-1260) at BBF.gin's values."""
    return float(official_exponential_decay_scheduler(10_000, 0, 0.97, 0.997, reverse=True)(step))


@pytest.mark.parametrize(
    ("step", "n", "gamma"),
    [(0, 10, 0.97), (2_500, 7, 0.98313), (5_000, 5, 0.990513), (10_000, 3, 0.997)],
)
def test_the_anneals_match_the_fixture_table(step: int, n: int, gamma: float) -> None:
    config = StackedDqnConfig(**BBF)  # type: ignore[arg-type]

    assert config.n_step_at(step) == official_n_step(step) == n
    assert config.discount_at(step) == pytest.approx(gamma, abs=5e-7)


def test_the_anneals_match_the_official_scheduler_at_every_step() -> None:
    config = StackedDqnConfig(**BBF)  # type: ignore[arg-type]

    for step in range(0, 12_001, 7):
        assert config.n_step_at(step) == official_n_step(step), step
        assert math.isclose(config.discount_at(step), official_gamma(step), abs_tol=1e-12), step


def test_without_a_discount_anneal_the_discount_is_fixed() -> None:
    config = StackedDqnConfig(n_step_final=3, n_step_anneal_steps=100)

    assert {config.discount_at(step) for step in (0, 50, 100, 10**6)} == {config.discount}


def test_a_discount_anneal_needs_the_n_step_anneal_it_runs_over() -> None:
    with pytest.raises(ValueError, match="discount anneal"):
        StackedDqnConfig(discount_initial=0.97)


# --- Reset timing: every interval, none within one interval of the budget ----


def test_resets_fall_every_interval_and_never_within_one_of_the_end() -> None:
    config = StackedDqnConfig(reset_every_steps=40_000, no_resets_after_steps=121_000)

    resets = [step for step in range(0, 130_001) if config.resets_after(step)]

    # 120,000 would leave 1,000 steps to recover in: BBF skips it.
    assert resets == [40_000, 80_000]


def test_a_reset_with_exactly_one_interval_left_is_not_taken() -> None:
    """No reset in the final interval, as BBF's 100k run skips its fourth.

    The official code counts environment steps with `reset_offset=1`, so its
    resets fall near 36k, 76k and 116k gradient steps; the strict bound here is
    what gives the same outcome in gradient steps.
    """
    config = StackedDqnConfig(reset_every_steps=40_000, no_resets_after_steps=160_000)

    assert [step for step in range(0, 160_001) if config.resets_after(step)] == [
        40_000,
        80_000,
    ]


def test_there_are_no_resets_by_default() -> None:
    config = StackedDqnConfig()

    assert not any(config.resets_after(step) for step in range(0, 200_001, 40_000))


def test_resets_need_the_budget_they_are_timed_against() -> None:
    with pytest.raises(ValueError, match="budget"):
        StackedDqnConfig(reset_every_steps=40_000)


# --- The reset itself --------------------------------------------------------


def _sequence(*, length: int = 16, burn_in: int = 3, shift: float = 0.0) -> ReplaySequence:
    actions = len(RUN_ACTIONS)
    steps = tuple(
        ReplayStep(
            features=StateFeatures(
                scalars=tuple([0.1 * index + shift] * SCALAR_COUNT),
                rows=tuple([0.05 * index - shift] * (ROW_COUNT * ROW_WIDTH)),
                mask=tuple(slot in (0, 1, 2) for slot in range(actions)),
            ),
            action_index=index % 3,
            reward=1.0 + shift,
            done=index == length - 1,
            admissible=True,
        )
        for index in range(length)
    )
    metadata = SequenceMetadata(
        episode_id="e", actor_id="a", profile_id="p",
        observation_schema="observation-v1", action_schema="run-action-v1",
        reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
    )
    return ReplaySequence(metadata, steps, burn_in)


def _batches(count: int) -> list[SequenceBatch]:
    return [
        collate((_sequence(shift=0.01 * index), _sequence(shift=-0.02 * index)), (1.0, 1.0))
        for index in range(count)
    ]


def _bbf(**overrides: object) -> StackedDqnBackbone:
    settings: dict[str, object] = {
        **BBF,
        "seed": 0,
        "history_length": 4,
        "n_step_anneal_steps": 2,
        "reset_every_steps": 3,
        "no_resets_after_steps": 100,
    }
    settings.update(overrides)
    return StackedDqnBackbone(
        config=StackedDqnConfig(**settings),  # type: ignore[arg-type]
        network_config=SMALL,
    )


def _fresh(backbone: StackedDqnBackbone, reset: int, role: str) -> StackedPolicyNetwork:
    """The initialisation a reset should draw, built independently of `_reset`."""
    torch.manual_seed(reset_seed(backbone._reset_seed, reset, role))
    return StackedPolicyNetwork(SMALL, history_length=backbone.config.history_length)


def _parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def test_shrink_and_perturb_is_exactly_half_old_half_fresh_on_the_trunk() -> None:
    backbone = _bbf(reset_every_steps=0, no_resets_after_steps=0)
    for batch in _batches(2):
        backbone.learn(batch)
    online_trunk = _parameters(backbone.online.trunk)
    target_trunk = _parameters(backbone.target.trunk)

    backbone._reset()

    fresh_online = _fresh(backbone, 1, "online")
    fresh_target = _fresh(backbone, 1, "target")
    assert (SHRINK_FACTOR, PERTURB_FACTOR) == (0.5, 0.5)
    for old, fresh, now in zip(
        online_trunk, fresh_online.trunk.parameters(), backbone.online.trunk.parameters(),
        strict=True,
    ):
        assert torch.equal(now, old * 0.5 + fresh.detach() * 0.5)
    for old, fresh, now in zip(
        target_trunk, fresh_target.trunk.parameters(), backbone.target.trunk.parameters(),
        strict=True,
    ):
        assert torch.equal(now, old * 0.5 + fresh.detach() * 0.5)


def test_the_core_heads_and_target_are_freshly_initialised() -> None:
    backbone = _bbf(reset_every_steps=0, no_resets_after_steps=0)
    for batch in _batches(2):
        backbone.learn(batch)

    backbone._reset()

    fresh_online = _fresh(backbone, 1, "online")
    fresh_target = _fresh(backbone, 1, "target")
    for part in ("core", "heads"):
        assert parameters_are_equal(getattr(backbone.online, part), getattr(fresh_online, part))
        assert parameters_are_equal(getattr(backbone.target, part), getattr(fresh_target, part))
    # The target draws an initialisation of its own, as `jit_reset` splits its key.
    assert not parameters_are_equal(fresh_online.core, fresh_target.core)


def test_a_reset_neither_reads_nor_moves_the_global_generator() -> None:
    backbone = _bbf(reset_every_steps=0, no_resets_after_steps=0)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)

    backbone._reset()

    assert torch.equal(torch.rand(3), expected)


def test_the_whole_optimizer_state_is_cleared_at_a_reset() -> None:
    """BBF's optax chain of masked states defeats `copy_params`' keys, so the
    fresh state replaces all of it, encoder included."""
    backbone = _bbf(reset_every_steps=0, no_resets_after_steps=0)
    for batch in _batches(3):
        backbone.learn(batch)
    state = backbone.optimizer.state
    parameters = list(backbone.online.parameters())
    assert all(state[p] for p in parameters), "every parameter has moments before the reset"

    backbone._reset()

    assert all(p not in state or not state[p] for p in parameters)

    backbone.learn(_batches(1)[0])

    assert all(float(state[p]["step"]) == 1.0 for p in parameters)


def test_learning_resets_on_schedule_and_restarts_both_anneals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tower_rl.learning.stacked_dqn as module

    seen: list[tuple[int, float]] = []
    real = module.n_step_targets

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen.append((int(kwargs["n_step"]), float(kwargs["discount"])))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "n_step_targets", recording)
    backbone = _bbf(reset_every_steps=3, no_resets_after_steps=10, n_step_anneal_steps=2)

    for batch in _batches(9):
        backbone.learn(batch)

    config = backbone.config
    cycle = [(config.n_step_at(t), config.discount_at(t)) for t in range(3)]
    # Resets after steps 3 and 6; step 9 would leave no interval to recover in.
    assert seen == cycle * 3
    assert (backbone.model_version, backbone._resets, backbone._cycle_steps) == (9, 2, 3)


# --- The optimizer -----------------------------------------------------------


def test_weight_decay_skips_one_dimensional_parameters() -> None:
    backbone = _bbf()
    groups = backbone.optimizer.param_groups

    decayed, undecayed = groups
    assert decayed["weight_decay"] == 0.1 and undecayed["weight_decay"] == 0.0
    assert all(p.ndim != 1 for p in decayed["params"])
    assert all(p.ndim == 1 for p in undecayed["params"])
    assert len(decayed["params"]) + len(undecayed["params"]) == len(
        list(backbone.online.parameters())
    )
    assert all(group["eps"] == 1.5e-4 and group["lr"] == 1e-4 for group in groups)


def test_decay_is_applied_to_matrices_and_not_to_vectors() -> None:
    """With zero gradients Adam moves nothing, so what is left is the decay."""
    backbone = _bbf()
    before = _parameters(backbone.online)
    for parameter in backbone.online.parameters():
        parameter.grad = torch.zeros_like(parameter)

    backbone.optimizer.step()

    for old, now in zip(before, backbone.online.parameters(), strict=True):
        expected = old if old.ndim == 1 else old * (1.0 - 1e-4 * 0.1)
        assert torch.allclose(now, expected, rtol=0, atol=1e-12)


def test_run_four_keeps_one_group_decaying_everything() -> None:
    backbone = StackedDqnBackbone(config=StackedDqnConfig(seed=0), network_config=SMALL)

    [group] = backbone.optimizer.param_groups
    assert (group["weight_decay"], group["eps"]) == (1e-5, 1e-8)


# --- The target ---------------------------------------------------------------


def test_the_target_moves_a_two_hundredth_of_the_way_every_step() -> None:
    backbone = _bbf(reset_every_steps=0, no_resets_after_steps=0)
    target = _parameters(backbone.target)

    backbone.learn(_batches(1)[0])

    assert 1.0 - backbone.config.target_ema_decay == pytest.approx(0.005)
    for old, online, now in zip(
        target, backbone.online.parameters(), backbone.target.parameters(), strict=True
    ):
        assert torch.equal(now, old.mul(0.995).add(online.detach(), alpha=1.0 - 0.995))


def _states(count: int) -> list[StateFeatures]:
    draw = random.Random(0)
    return [
        StateFeatures(
            scalars=tuple(draw.uniform(-1.0, 1.0) for _ in range(SCALAR_COUNT)),
            rows=tuple(draw.uniform(-1.0, 1.0) for _ in range(ROW_COUNT * ROW_WIDTH)),
            mask=tuple(True for _ in RUN_ACTIONS),
        )
        for _ in range(count)
    ]


def _greedy(network: StackedPolicyNetwork, features: StateFeatures) -> int:
    scalars = torch.tensor([[list(features.scalars)]], dtype=torch.float32)
    rows = torch.tensor([[list(features.rows)]], dtype=torch.float32).view(
        1, 1, ROW_COUNT, ROW_WIDTH
    )
    mask = torch.tensor([[list(features.mask)]], dtype=torch.bool)
    with torch.no_grad():
        q, _ = network(scalars, rows, mask, None)
    return int(q[0, 0].argmax().item())


def test_actors_act_with_the_target_after_an_update() -> None:
    """BBF's `target_action_selection=True`, through the path actors are fed by."""
    # A large learning rate pulls the online network well away from the target,
    # so there are states where the two choose differently.
    backbone = _bbf(act_with_target=True, reset_every_steps=0, learning_rate=0.5)
    learner = Learner(backbone)
    actor = acting_copy(backbone)
    for batch in _batches(3):
        learner.learn(batch)
    learner.publish_to(actor)

    states = _states(32)
    differing = [
        state
        for state in states
        if _greedy(backbone.target, state) != _greedy(backbone.online, state)
    ]
    assert differing, "the online and target networks never disagreed"
    for state in states:
        action, _ = actor.act(state, None, epsilon=0.0)
        assert action == _greedy(backbone.target, state)

    # The control: the same copy, acting with the online network as run 4 does.
    assert isinstance(actor, StackedDqnBackbone)
    online_actor = StackedDqnBackbone(
        config=dataclasses.replace(actor.config, act_with_target=False), network_config=SMALL
    )
    online_actor.load_state_dict(actor.state_dict())
    action, _ = online_actor.act(differing[0], None, epsilon=0.0)
    assert action == _greedy(backbone.online, differing[0])
    assert action != _greedy(backbone.target, differing[0])


# --- Gradient clipping --------------------------------------------------------


def test_bbf_does_not_clip_the_gradient() -> None:
    clipped = _bbf(reset_every_steps=0, gradient_clip=1e-3)
    unclipped = _bbf(reset_every_steps=0, gradient_clip=None)
    batch = _batches(1)[0]

    clipped_norm = clipped.learn(batch).gradient_norm
    unclipped_norm = unclipped.learn(batch).gradient_norm

    def norm(backbone: StackedDqnBackbone) -> float:
        grads = [p.grad for p in backbone.online.parameters() if p.grad is not None]
        return float(torch.nn.utils.get_total_norm(grads))

    # Both report the norm before clipping, and it is the same gradient.
    assert unclipped_norm == pytest.approx(clipped_norm)
    assert unclipped_norm > 1e-3
    # Only the clipped learner stepped with a gradient scaled down to its limit.
    assert norm(unclipped) == pytest.approx(unclipped_norm)
    assert norm(clipped) == pytest.approx(1e-3, rel=1e-3)


def test_a_clip_must_be_positive() -> None:
    with pytest.raises(ValueError, match="clip"):
        StackedDqnConfig(gradient_clip=0.0)


# --- Persistence --------------------------------------------------------------


def _round_trip(backbone: StackedDqnBackbone, path: Path) -> dict[str, Any]:
    identity = CheckpointIdentity(
        run_id="r", backbone="stacked-dqn", profile_id="p",
        observation_schema="o", action_schema="a", reward_schema="w", source_revision="s",
    )
    save(
        Checkpoint(
            identity=identity,
            progress=TrainingProgress(),
            backbone_state=backbone.state_dict(),
        ),
        path,
    )
    return dict(load(path).backbone_state)


def test_a_resumed_learner_is_the_same_learner_across_a_reset(tmp_path: Path) -> None:
    """Split before the reset at step 3 and resumed from the file, it ends identical."""
    batches = _batches(6)
    # Unseeded, so the reset seed has to come back from the file.
    whole = _bbf(seed=None, reset_every_steps=3, no_resets_after_steps=7)
    for batch in batches[:2]:
        whole.learn(batch)
    saved = _round_trip(whole, tmp_path / "latest.pt")
    for batch in batches[2:]:
        whole.learn(batch)

    resumed = _bbf(seed=None, reset_every_steps=3, no_resets_after_steps=7)
    resumed.load_state_dict(saved)
    for batch in batches[2:]:
        resumed.learn(batch)

    assert whole._resets == resumed._resets == 1
    assert whole._cycle_steps == resumed._cycle_steps == 3
    assert parameters_are_equal(whole.online, resumed.online)
    assert parameters_are_equal(whole.target, resumed.target)
    for mine, theirs in zip(
        whole.optimizer.state_dict()["state"].values(),
        resumed.optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        assert all(torch.equal(mine[key], theirs[key]) for key in mine)


def test_a_state_from_before_resets_reads_as_one_cycle() -> None:
    """Run 4's checkpoints carry no cycle: their anneal read the total count."""
    parent = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=4, n_step_final=3, n_step_anneal_steps=4),
        network_config=SMALL,
    )
    parent.learn(_batches(1)[0])
    parent.learn(_batches(1)[0])
    state = parent.state_dict()
    for key in ("cycle_steps", "resets", "reset_seed"):
        del state[key]

    child = StackedDqnBackbone(
        config=StackedDqnConfig(seed=0, history_length=4, n_step_final=3, n_step_anneal_steps=4),
        network_config=SMALL,
    )
    child.load_state_dict(state)

    assert (child._cycle_steps, child._resets) == (2, 0)
