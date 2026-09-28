from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from fakes.backbone_equality import parameters_are_equal

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, SCALAR_COUNT, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning import stacked_dqn
from tower_rl.learning.backbone import collate
from tower_rl.learning.network import NetworkConfig, StackedPolicyNetwork
from tower_rl.learning.replay import ReplaySequence, ReplayStep, SequenceMetadata
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig
from tower_rl.learning.value_learning import n_step_targets

ACTIONS = len(RUN_ACTIONS)
SMALL = NetworkConfig(hidden=16, core_hidden=16, identity_dim=4)


def _features(*, valid: tuple[int, ...] = (0, 1, 2), seed: float = 0.5) -> StateFeatures:
    mask = tuple(index in valid for index in range(ACTIONS))
    return StateFeatures(
        scalars=tuple([seed] * SCALAR_COUNT),
        rows=tuple([seed] * (ROW_COUNT * ROW_WIDTH)),
        mask=mask,
    )


def _sequence(*, length: int = 8, burn_in: int = 4) -> ReplaySequence:
    steps = tuple(
        ReplayStep(
            features=_features(seed=0.1 * index),
            action_index=index % 3,
            reward=1.0,
            done=index == length - 1,
            admissible=True,
            game_ms=1000.0,
        )
        for index in range(length)
    )
    return ReplaySequence(
        SequenceMetadata(
            episode_id="e", actor_id="a", profile_id="p",
            observation_schema="observation-v1", action_schema="run-action-v1",
            reward_schema="reward-v1", model_version=0, epsilon=0.0, game_speed=8.0,
        ),
        steps,
        burn_in,
    )


def _backbone(**overrides: object) -> StackedDqnBackbone:
    settings: dict[str, object] = {"seed": 0, "history_length": 4}
    settings.update(overrides)
    return StackedDqnBackbone(
        config=StackedDqnConfig(**settings),  # type: ignore[arg-type]
        network_config=SMALL,
    )


def test_the_window_carries_the_previous_steps_not_the_current_one() -> None:
    network = StackedPolicyNetwork(SMALL, history_length=3)
    # Three steps of distinct scalar vectors, however wide the schema is.
    scalars = torch.arange(3 * SCALAR_COUNT, dtype=torch.float32).view(1, 3, SCALAR_COUNT)
    state = network.initial_state(1)

    stacked = network.stack(scalars, state)

    assert stacked.shape == (1, 3, SCALAR_COUNT * 3)
    # The first step has no history at all, so its window is padded with zeros
    # and ends with the step itself.
    assert torch.equal(stacked[0, 0, -SCALAR_COUNT:], scalars[0, 0])
    assert torch.equal(stacked[0, 0, :-SCALAR_COUNT], torch.zeros(SCALAR_COUNT * 2))
    # By the third step the window is full and ordered oldest to newest.
    assert torch.equal(stacked[0, 2], scalars[0].flatten())


def test_carrying_state_across_calls_matches_one_long_call() -> None:
    """Acting step by step must see the same window as a batched forward pass."""
    network = StackedPolicyNetwork(SMALL, history_length=3)
    scalars = torch.randn(1, 4, SCALAR_COUNT)
    rows = torch.randn(1, 4, ROW_COUNT, ROW_WIDTH)
    mask = torch.ones(1, 4, ACTIONS, dtype=torch.bool)

    whole, _ = network(scalars, rows, mask)

    state = network.initial_state(1)
    stepwise = []
    for index in range(4):
        q, state = network(
            scalars[:, index : index + 1], rows[:, index : index + 1],
            mask[:, index : index + 1], state,
        )
        stepwise.append(q)

    assert torch.allclose(whole, torch.cat(stepwise, dim=1), atol=1e-6)


def test_history_of_one_is_the_no_history_ablation() -> None:
    network = StackedPolicyNetwork(SMALL, history_length=1)
    scalars = torch.randn(2, 3, SCALAR_COUNT)
    state = network.initial_state(2)

    assert torch.equal(network.stack(scalars, state), scalars)
    assert network.initial_state(2).shape == (2, 0, SCALAR_COUNT)


def test_learning_runs_and_only_the_window_contributes_errors() -> None:
    backbone = _backbone()
    batch = collate((_sequence(length=8, burn_in=4),), (1.0,))

    metrics = backbone.learn(batch)

    assert len(metrics.td_errors[0]) == 4, "eight steps minus four burn-in"
    assert metrics.weighted_loss >= 0.0
    assert backbone.model_version == 1


def test_a_burn_in_too_short_to_fill_the_window_is_refused() -> None:
    """Silently padding here would train on history acting never sees."""
    backbone = _backbone(history_length=6)
    batch = collate((_sequence(length=8, burn_in=2),), (1.0,))

    with pytest.raises(ValueError, match="cannot fill a window"):
        backbone.learn(batch)


def test_the_target_follows_the_online_network_without_ever_matching_it() -> None:
    backbone = _backbone()
    before = [parameter.clone() for parameter in backbone.target.parameters()]

    backbone.learn(collate((_sequence(),), (1.0,)))

    assert not parameters_are_equal(backbone.online, backbone.target), "EMA, not a copy"
    moved = any(
        not torch.equal(old, new)
        for old, new in zip(before, backbone.target.parameters(), strict=True)
    )
    assert moved, "an EMA target must move on every step"


def test_acting_respects_the_mask_and_threads_its_window() -> None:
    backbone = _backbone()
    state = backbone.initial_state()

    action, state = backbone.act(_features(valid=(0, 5)), state, epsilon=0.0)

    assert action in (0, 5)
    assert state.shape == (1, 3, SCALAR_COUNT)


def test_state_round_trips_exactly() -> None:
    backbone = _backbone()
    backbone.learn(collate((_sequence(),), (1.0,)))
    saved = backbone.state_dict()

    restored = _backbone()
    restored.load_state_dict(saved)

    assert restored.model_version == backbone.model_version
    assert parameters_are_equal(restored.online, backbone.online)
    assert parameters_are_equal(restored.target, backbone.target)


# --- The n-step anneal: n from 10 down to 3 over the first gradient steps


def _annealed(**overrides: object) -> StackedDqnBackbone:
    settings: dict[str, object] = {
        "n_step": 10,
        "n_step_final": 3,
        "n_step_anneal_steps": 10_000,
    }
    settings.update(overrides)
    return _backbone(**settings)


def test_the_n_step_anneal_is_exponential_and_then_holds() -> None:
    config = StackedDqnConfig(n_step=10, n_step_final=3, n_step_anneal_steps=10_000)

    assert config.n_step_at(0) == 10
    # Halfway is the geometric mean, 10 * 0.3 ** 0.5 = 5.48, not the linear 6.5.
    assert config.n_step_at(5_000) == 5
    assert config.n_step_at(10_000) == 3
    assert config.n_step_at(50_000) == 3
    schedule = [config.n_step_at(step) for step in range(0, 10_001, 100)]
    assert schedule == sorted(schedule, reverse=True), "the anneal never lengthens n"


def test_without_an_anneal_n_is_fixed_at_every_step() -> None:
    """The default is every run before run 4: n = n_step throughout."""
    config = StackedDqnConfig()

    assert config.n_step_final is None and config.n_step_anneal_steps == 0
    assert {config.n_step_at(step) for step in (0, 1, 10_000, 10**7)} == {config.n_step}


@pytest.mark.parametrize(
    "settings",
    [
        {"n_step_final": 3},
        {"n_step_anneal_steps": 100},
        {"n_step_final": 0, "n_step_anneal_steps": 100},
    ],
)
def test_a_half_configured_anneal_is_refused(settings: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        StackedDqnConfig(**settings)  # type: ignore[arg-type]


def _record_n_steps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every n the learner hands the shared target, in the order it did."""
    import tower_rl.learning.stacked_dqn as module

    seen: list[int] = []
    real = module.n_step_targets

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen.append(int(kwargs["n_step"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "n_step_targets", recording)
    return seen


def test_learning_builds_its_target_with_the_n_of_its_own_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _record_n_steps(monkeypatch)
    backbone = _annealed(n_step_anneal_steps=2)
    batch = collate((_sequence(length=16, burn_in=4),), (1.0,))

    for _ in range(4):
        backbone.learn(batch)

    # t = 0, 1, 2 (= T), 3 (> T): 10, 10 * 0.3 ** 0.5 = 5.48, 3, 3.
    assert seen == [10, 5, 3, 3]


def test_a_resumed_learner_continues_the_anneal_where_it_left_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _annealed(n_step_anneal_steps=4)
    batch = collate((_sequence(length=16, burn_in=4),), (1.0,))
    for _ in range(2):
        parent.learn(batch)

    seen = _record_n_steps(monkeypatch)
    resumed = _annealed(n_step_anneal_steps=4)
    resumed.load_state_dict(parent.state_dict())
    resumed.learn(batch)

    # Step two of four, not step zero again: 10 * 0.3 ** 0.5.
    assert seen == [parent.config.n_step_at(2)] == [5]


def _timed_sequence(spans_ms: tuple[float, ...], *, burn_in: int = 4) -> ReplaySequence:
    """A sequence whose steps span these game times, a purchase at 0 ms."""
    steps = tuple(
        ReplayStep(
            features=_features(seed=0.1 * index),
            action_index=index % 3,
            reward=1.0 if index % 2 else 0.0,
            done=index == len(spans_ms) - 1,
            admissible=True,
            game_ms=game_ms,
        )
        for index, game_ms in enumerate(spans_ms)
    )
    return ReplaySequence(_sequence().metadata, steps, burn_in)


def test_learning_under_the_game_time_discount_is_finite() -> None:
    """T9: learn() discounting per game-second: finite loss, TD error and value fit."""
    backbone = _backbone(discount_per_game_second=0.997)
    spans = (2000.0, 0.0, 5000.0, 0.0, 1733.0, 0.0, 17000.0, 2300.0, 0.0, 900.0)
    batch = collate((_timed_sequence(spans), _timed_sequence(spans[::-1])), (1.0, 1.0))

    for _ in range(3):
        metrics = backbone.learn(batch)
        assert torch.isfinite(torch.tensor(metrics.weighted_loss))
        assert torch.isfinite(torch.tensor(metrics.unweighted_mean_absolute_td_error))
        assert metrics.value_fit_correlation is not None
        assert -1.0 <= metrics.value_fit_correlation <= 1.0


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.1])
def test_a_discount_per_game_second_outside_zero_one_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="per game-second"):
        StackedDqnConfig(discount_per_game_second=value)


# --- The survival-time reward (board #82)


def test_the_survival_time_reward_is_refused_without_the_game_time_discount() -> None:
    """Per decision a span has no length for the reward to be integrated over."""
    with pytest.raises(ValueError, match="survival-time reward"):
        StackedDqnConfig(survival_time_reward=True)


def test_a_span_of_no_game_time_survives_exactly_nothing() -> None:
    config = StackedDqnConfig(discount_per_game_second=0.997, survival_time_reward=True)
    discounts = config.transition_discounts(torch.zeros(1, 3))

    assert torch.equal(config.survival_rewards(discounts), torch.zeros(1, 3, dtype=torch.float64))


def test_a_seventeen_second_span_earns_its_discounted_share_of_a_wave() -> None:
    """(1 - g^17) / (-ln g * 35): the longest span M3-P003 saw, about half a wave."""
    config = StackedDqnConfig(discount_per_game_second=0.997, survival_time_reward=True)
    discounts = config.transition_discounts(torch.tensor([[17000.0]]))

    assert config.survival_rewards(discounts).item() == pytest.approx(0.47352, rel=1e-5)


def test_the_survival_reward_at_0_997_is_the_unscaled_reward_of_every_run_before() -> None:
    """(1 - d) * V_REF equals M3-P004..M3-P014's (1 - d) / (-ln 0.997 * 35) at 0.997."""
    config = StackedDqnConfig(discount_per_game_second=0.997, survival_time_reward=True)
    seconds = torch.tensor([[0.0, 0.5, 1.85, 17.0, 35.0, 1000.0, 1e6]], dtype=torch.float64)
    discounts = config.transition_discounts(seconds * 1000.0)

    unscaled = (1.0 - discounts) / (-math.log(0.997) * 35.0)
    assert abs(stacked_dqn.V_REF - 9.51) < 0.005
    assert torch.allclose(config.survival_rewards(discounts), unscaled, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("discount_per_game_second", [0.997, 0.999, 0.9994])
def test_the_survival_return_is_bounded_by_v_ref_whatever_the_discount(
    discount_per_game_second: float,
) -> None:
    """A never-ending life's return, summed span by span, approaches V_REF from below."""
    config = StackedDqnConfig(
        discount_per_game_second=discount_per_game_second, survival_time_reward=True
    )
    # 200,000 spans of 1.85 s, about 10,600 waves: gamma ** T is negligible.
    spans = torch.full((200_000,), 1850.0, dtype=torch.float64)
    discounts = config.transition_discounts(spans)
    rewards = config.survival_rewards(discounts)
    # Each reward valued at the start of its own span, discounted back to 0.
    before = torch.cat((torch.ones(1, dtype=torch.float64), discounts.cumprod(0)[:-1]))
    maximum = float((before * rewards).sum())

    assert maximum <= stacked_dqn.V_REF * (1.0 + 1e-9)
    assert maximum == pytest.approx(stacked_dqn.V_REF, rel=1e-6)


def test_a_learning_step_reports_the_largest_taken_action_q() -> None:
    """The live read of the value against V_REF: a finite number, from real steps."""
    spans = (2000.0, 0.0, 5000.0, 0.0, 1733.0, 0.0, 17000.0, 2300.0, 0.0, 900.0)
    metrics = _backbone().learn(collate((_timed_sequence(spans),), (1.0,)))

    assert metrics.taken_q_max is not None and math.isfinite(metrics.taken_q_max)


@pytest.mark.parametrize("discount_per_game_second", [None, 0.997])
def test_learning_with_the_survival_time_reward_off_is_bit_identical(
    discount_per_game_second: float | None,
) -> None:
    """Off, the flag changes nothing: same metrics and same weights, bit for bit."""
    spans = (2000.0, 0.0, 5000.0, 0.0, 1733.0, 0.0, 17000.0, 2300.0, 0.0, 900.0)
    batch = collate((_timed_sequence(spans), _timed_sequence(spans[::-1])), (1.0, 0.5))
    default = _backbone(discount_per_game_second=discount_per_game_second)
    explicit = _backbone(
        discount_per_game_second=discount_per_game_second, survival_time_reward=False
    )

    for _ in range(3):
        assert default.learn(batch) == explicit.learn(batch)

    for left, right in ((default.online, explicit.online), (default.target, explicit.target)):
        left_state, right_state = left.state_dict(), right.state_dict()
        assert left_state.keys() == right_state.keys()
        assert all(torch.equal(left_state[key], right_state[key]) for key in left_state)


def test_learning_with_the_survival_time_reward_on_learns_from_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On, the target is built from the survival reward in place of the wave reward."""
    spans = (2000.0, 0.0, 5000.0, 0.0, 1733.0, 0.0, 17000.0, 2300.0, 0.0, 900.0)
    batch = collate((_timed_sequence(spans),), (1.0,))
    survival = _backbone(discount_per_game_second=0.997, survival_time_reward=True)
    captured: list[torch.Tensor] = []

    def capturing(rewards: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        captured.append(rewards.clone())
        return n_step_targets(rewards, *args, **kwargs)

    monkeypatch.setattr(stacked_dqn, "n_step_targets", capturing)
    metrics = survival.learn(batch)

    config = survival.config
    expected = config.survival_rewards(
        config.transition_discounts(batch.game_ms[:, batch.burn_in :])
    ).to(batch.rewards.dtype)
    assert len(captured) == 1
    assert torch.equal(captured[0], expected)
    assert not torch.equal(expected, batch.rewards[:, batch.burn_in :])
    assert torch.isfinite(torch.tensor(metrics.weighted_loss))


# -- ez-greedy (board #83) ---------------------------------------------------

#: Masks the golden run cycles through: WAIT alone, and one without WAIT.
GOLDEN_MASKS = ((0, 1, 2), (0, 2), (0, 1, 2, 3), (0,), (1, 2))

#: What acting at epsilon 0.5 chose on the code before ez-greedy, and the next
#: uniform its stream then gave: off, both must stay exactly these.
GOLDEN_ACTIONS = [
    0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 1, 0, 2, 0, 0, 0, 0, 2,
]  # fmt: skip
GOLDEN_NEXT_UNIFORM = 0.5512672460905512


def test_acting_with_ez_greedy_off_is_bit_identical() -> None:
    """The same actions and the same stream position as before the flag existed."""
    backbone = _backbone(ez_greedy=False)
    state = backbone.initial_state()
    actions = []
    for index in range(40):
        features = _features(valid=GOLDEN_MASKS[index % 5], seed=0.05 * index)
        action, state = backbone.act(features, state, epsilon=0.5)
        actions.append(action)

    assert actions == GOLDEN_ACTIONS
    assert backbone._random.random() == GOLDEN_NEXT_UNIFORM
    assert (backbone.options_started, backbone.longest_option) == (0, 0)


class _Durations:
    """`zeta_duration` replaced by fixed lengths, counting its draws."""

    def __init__(self, *lengths: int) -> None:
        self.lengths = list(lengths)
        self.draws = 0

    def __call__(self, stream: object) -> int:
        self.draws += 1
        return self.lengths.pop(0)


def _ez(monkeypatch: pytest.MonkeyPatch, *lengths: int) -> tuple[StackedDqnBackbone, _Durations]:
    durations = _Durations(*lengths)
    monkeypatch.setattr(stacked_dqn, "zeta_duration", durations)
    return _backbone(ez_greedy=True), durations


def test_an_option_repeats_its_action_for_exactly_n_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drawing decision is the first of n; the (n + 1)th flips a fresh coin."""
    backbone, durations = _ez(monkeypatch, 3, 1)
    state = backbone.initial_state()
    actions = []
    for index in range(3):
        action, state = backbone.act(_features(seed=0.1 * index), state, epsilon=1.0)
        actions.append(action)

    assert len(set(actions)) == 1 and durations.draws == 1
    backbone.act(_features(), state, epsilon=1.0)
    assert durations.draws == 2
    assert (backbone.options_started, backbone.longest_option) == (2, 3)


def test_a_masked_option_waits_counts_down_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone, durations = _ez(monkeypatch, 4, 1)
    state = backbone.initial_state()
    # Only row 2 is legal when the option starts, so it is the option's action.
    action, state = backbone.act(_features(valid=(2,)), state, epsilon=1.0)
    assert action == 2
    # Unaffordable: WAIT, and the decision still counts.
    action, state = backbone.act(_features(valid=(0, 1)), state, epsilon=1.0)
    assert action == stacked_dqn.WAIT_INDEX
    # Legal again: the purchase resumes for the option's last two decisions.
    for _ in range(2):
        action, state = backbone.act(_features(valid=(0, 1, 2)), state, epsilon=1.0)
        assert action == 2
    assert durations.draws == 1 and backbone.longest_option == 4
    backbone.act(_features(), state, epsilon=1.0)
    assert durations.draws == 2


def test_a_masked_option_refuses_a_state_where_wait_is_masked_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WAIT is always legal in an active run; a state without it is loud, not guessed."""
    backbone, _ = _ez(monkeypatch, 3)
    state = backbone.initial_state()
    backbone.act(_features(valid=(2,)), state, epsilon=1.0)

    with pytest.raises(ValueError, match="WAIT is not available"):
        backbone.act(_features(valid=(1,)), state, epsilon=1.0)


def test_an_episode_boundary_ends_an_option(monkeypatch: pytest.MonkeyPatch) -> None:
    backbone, durations = _ez(monkeypatch, 5, 5)
    state = backbone.initial_state()
    backbone.act(_features(), state, epsilon=1.0)

    state = backbone.initial_state()
    assert (backbone.options_started, backbone.longest_option) == (0, 0)
    backbone.act(_features(), state, epsilon=1.0)
    assert durations.draws == 2 and backbone.options_started == 1


def test_epsilon_zero_never_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluation stays greedy, even with an option left running."""
    backbone, durations = _ez(monkeypatch, 5)
    greedy = _backbone()
    state, greedy_state = backbone.initial_state(), greedy.initial_state()
    backbone.act(_features(), state, epsilon=1.0)
    for index in range(4):
        features = _features(seed=0.2 * index)
        action, state = backbone.act(features, state, epsilon=0.0)
        expected, greedy_state = greedy.act(features, greedy_state, epsilon=0.0)
        assert action == expected
    assert durations.draws == 1 and backbone.longest_option == 1


def test_the_window_advances_during_a_repeat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every decision of an option still runs the forward pass the window rides on."""
    backbone, _ = _ez(monkeypatch, 6)
    greedy = _backbone()
    state, greedy_state = backbone.initial_state(), greedy.initial_state()
    for index in range(6):
        features = _features(seed=0.1 * (index + 1))
        _, state = backbone.act(features, state, epsilon=1.0)
        _, greedy_state = greedy.act(features, greedy_state, epsilon=0.0)
        assert torch.equal(state, greedy_state)
    assert backbone.longest_option == 6


# --- The optimiser's epsilon and the recipe's discount (board #93, reverted #85)


def test_the_first_step_is_taken_at_an_adam_epsilon_of_1e_8() -> None:
    """Read off the step itself, not the setting.

    AdamW's first step moves a parameter p with (clipped) gradient g by
    -lr * wd * p - lr * g / (|g| + eps): the bias-corrected moments are g and
    g squared. Where |g| is far below eps, 1.5e-4 and torch's 1e-8 differ by
    orders of magnitude - about lr itself, far above float32's rounding of
    the step - so the step says which one it was taken at.
    """
    backbone = _backbone()
    config = backbone.config
    before = [parameter.detach().clone() for parameter in backbone.online.parameters()]

    backbone.learn(collate((_sequence(), _sequence()), (1.0, 1.0)))

    def expected(epsilon: float) -> list[torch.Tensor]:
        return [
            old
            - config.learning_rate * config.weight_decay * old
            - config.learning_rate * parameter.grad / (parameter.grad.abs() + epsilon)
            for old, parameter in zip(before, backbone.online.parameters(), strict=True)
            if parameter.grad is not None
        ]

    after = [
        parameter.detach()
        for parameter in backbone.online.parameters()
        if parameter.grad is not None
    ]
    small = torch.cat(
        [
            parameter.grad.abs().flatten()
            for parameter in backbone.online.parameters()
            if parameter.grad is not None
        ]
    )
    assert (small < 1.5e-5).any(), "no gradient small enough to tell the two apart"
    assert all(
        torch.allclose(actual, predicted, rtol=0.0, atol=1e-6)
        for actual, predicted in zip(after, expected(1e-8), strict=True)
    )
    assert not all(
        torch.allclose(actual, predicted, rtol=0.0, atol=1e-6)
        for actual, predicted in zip(after, expected(1.5e-4), strict=True)
    )


def test_a_game_time_discount_of_0_999_reaches_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discounts and rewards the n-step target is built from, at the recipe's value.

    0.999 ** seconds per transition, and under the survival-time reward
    (1 - d) * V_REF: a policy that never dies is worth V_REF, about 9.51 waves,
    as it was at 0.997, where the unscaled reward made it 28.6 (ADR 0013).
    """
    spans = (2000.0, 0.0, 5000.0, 0.0, 1733.0, 0.0, 17000.0, 2300.0, 0.0, 900.0)
    batch = collate((_timed_sequence(spans),), (1.0,))
    backbone = _backbone(discount_per_game_second=0.999, survival_time_reward=True)
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capturing(rewards: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        captured.append((rewards.clone(), kwargs["discounts"].clone()))
        return n_step_targets(rewards, *args, **kwargs)

    monkeypatch.setattr(stacked_dqn, "n_step_targets", capturing)
    backbone.learn(batch)

    ((rewards, discounts),) = captured
    seconds = torch.tensor(spans[batch.burn_in :], dtype=torch.float64) / 1000.0
    assert torch.allclose(discounts[0], 0.999**seconds)
    assert torch.allclose(
        rewards[0].double(), (1.0 - 0.999**seconds) * stacked_dqn.V_REF, atol=1e-7
    )


# --- SR-SPR-style resets: core and heads re-initialised, the trunk shrunk


def _resetting(**overrides: object) -> StackedDqnBackbone:
    settings: dict[str, object] = {"reset_every_steps": 2, "last_reset_step": 10}
    settings.update(overrides)
    return _backbone(**settings)


def test_a_reset_reinitialises_core_and_heads_and_shrinks_the_trunk() -> None:
    batch = collate((_sequence(length=16, burn_in=4),), (1.0,))
    plain, resetting = _backbone(), _resetting()
    # The fresh network the first reset draws: seeded by the run seed and a
    # reset count of 0, which `plain` shares.
    fresh = plain._fresh_network()
    for _ in range(2):
        plain.learn(batch)
        resetting.learn(batch)

    old = dict(plain.online.named_parameters())
    new = dict(fresh.named_parameters())
    assert resetting.resets == 1
    for name, parameter in resetting.online.named_parameters():
        if name.split(".")[0] in ("core", "heads"):
            assert torch.equal(parameter, new[name]), name
        else:
            assert name.startswith("trunk."), name
            expected = 0.8 * old[name] + 0.2 * new[name]
            assert torch.allclose(parameter, expected, atol=1e-7), name
    assert parameters_are_equal(resetting.target, resetting.online)


def test_a_reset_clears_the_optimizer_state_of_the_reset_parameters_only() -> None:
    batch = collate((_sequence(length=16, burn_in=4),), (1.0,))
    plain, resetting = _backbone(), _resetting()
    for _ in range(2):
        plain.learn(batch)
        resetting.learn(batch)

    kept = {
        name: plain.optimizer.state[parameter]
        for name, parameter in plain.online.named_parameters()
    }
    for name, parameter in resetting.online.named_parameters():
        state = resetting.optimizer.state.get(parameter)
        if name.startswith("trunk."):
            assert state is not None, name
            assert torch.equal(state["exp_avg"], kept[name]["exp_avg"]), name
            assert torch.equal(state["exp_avg_sq"], kept[name]["exp_avg_sq"]), name
        else:
            assert not state, name
    # The next step runs, starting the reset parameters' moments over.
    resetting.learn(batch)


def test_the_fresh_network_is_seeded_by_the_run_seed_and_the_reset_count() -> None:
    first, second = _backbone(), _backbone()
    assert parameters_are_equal(first._fresh_network(), second._fresh_network())
    second.resets = 1
    assert not parameters_are_equal(first._fresh_network(), second._fresh_network())
    assert not parameters_are_equal(_backbone(seed=1)._fresh_network(), first._fresh_network())


def test_fresh_network_leaves_the_cpu_rng_state_unchanged() -> None:
    backbone = _backbone()
    before = torch.get_rng_state()
    backbone._fresh_network()
    after = torch.get_rng_state()
    assert torch.equal(before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fresh_network_leaves_the_cuda_rng_state_unchanged() -> None:
    backbone = _backbone()
    before = torch.cuda.get_rng_state()
    backbone._fresh_network()
    after = torch.cuda.get_rng_state()
    assert torch.equal(before, after)


def test_a_reset_restarts_the_n_step_anneal(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_n_steps(monkeypatch)
    backbone = _annealed(n_step_anneal_steps=2, reset_every_steps=3, last_reset_step=3)
    batch = collate((_sequence(length=16, burn_in=4),), (1.0,))

    for _ in range(7):
        backbone.learn(batch)

    # A reset after step 3, and none after step 6, which is past the last.
    assert seen == [10, 5, 3, 10, 5, 3, 3]
    assert backbone.resets == 1


def test_no_reset_happens_after_the_last_reset_step() -> None:
    backbone = _resetting(reset_every_steps=1, last_reset_step=2)
    batch = collate((_sequence(),), (1.0,))
    for _ in range(5):
        backbone.learn(batch)

    assert backbone.resets == 2
    assert backbone.state_dict()["steps_at_reset"] == 2


def test_resets_are_off_by_default() -> None:
    assert StackedDqnConfig().reset_every_steps == 0
    batch = collate((_sequence(),), (1.0,))
    default, explicit = _backbone(), _backbone(reset_every_steps=0, last_reset_step=10**6)
    for _ in range(4):
        default.learn(batch)
        explicit.learn(batch)

    assert default.resets == 0 and default.state_dict()["steps_at_reset"] == 0
    assert parameters_are_equal(default.online, explicit.online)


def test_a_resumed_learner_keeps_its_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = collate((_sequence(length=16, burn_in=4),), (1.0,))
    parent = _annealed(n_step_anneal_steps=4, reset_every_steps=3, last_reset_step=100)
    for _ in range(4):
        parent.learn(batch)

    seen = _record_n_steps(monkeypatch)
    resumed = _annealed(n_step_anneal_steps=4, reset_every_steps=3, last_reset_step=100)
    resumed.load_state_dict(parent.state_dict())
    resumed.learn(batch)

    assert (resumed.resets, resumed.state_dict()["steps_at_reset"]) == (1, 3)
    # One step past the reset, not four past the start.
    assert seen == [parent.config.n_step_at(1)]
    # The next reset draws the fresh network the uninterrupted run would.
    assert parameters_are_equal(resumed._fresh_network(), parent._fresh_network())


def test_a_state_from_before_resets_loads_as_never_reset() -> None:
    parent = _backbone()
    parent.learn(collate((_sequence(),), (1.0,)))
    state = parent.state_dict()
    del state["steps_at_reset"], state["resets"]

    restored = _resetting()
    restored.load_state_dict(state)

    assert restored.resets == 0 and restored.state_dict()["steps_at_reset"] == 0
    assert parameters_are_equal(restored.online, parent.online)


@pytest.mark.parametrize("settings", [{"reset_every_steps": -1}, {"last_reset_step": -1}])
def test_a_negative_reset_setting_is_refused(settings: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        StackedDqnConfig(**settings)  # type: ignore[arg-type]
