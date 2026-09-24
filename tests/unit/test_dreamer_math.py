"""DreamerV3's numerical parts against the official formulas, with worked numbers.

Every oracle here is either a hand-computed value or an independent re-statement
of the official code (danijar/dreamerv3 at e3f02248) in NumPy, so a port that
drifted from the source fails against arithmetic rather than against itself.
"""

from __future__ import annotations

import math

import numpy
import pytest
import torch

from tower_rl.learning.dreamer_math import (
    LaProp,
    PercentileNormaliser,
    categorical_kl,
    lambda_return,
    masked_entropy,
    masked_policy_log_probs,
    sample_index,
    symexp,
    symlog,
    twohot_bins,
    twohot_loss,
    twohot_mean,
    unimix_probs,
)


def test_symlog_and_symexp_are_inverse_and_match_the_formula() -> None:
    e = math.e
    values = torch.tensor([-(e**2 - 1), -(e - 1), 0.0, e - 1, e**2 - 1], dtype=torch.float64)
    assert symlog(values).tolist() == pytest.approx([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert symexp(torch.tensor([-2.0, 1.0], dtype=torch.float64)).tolist() == pytest.approx(
        [-(e**2 - 1), e - 1]
    )
    wide = torch.linspace(-1e4, 1e4, 101, dtype=torch.float64)
    assert torch.allclose(symexp(symlog(wide)), wide)


def test_the_twohot_bins_are_the_official_symmetric_symexp_bins() -> None:
    bins = twohot_bins(255)
    assert bins.shape == (255,)
    assert bins[127].item() == 0.0
    assert torch.equal(bins, -bins.flip(0))
    assert bins[-1].item() == pytest.approx(math.expm1(20.0), rel=1e-6)
    # `heads.py`: half = symexp(linspace(-20, 0, 128)); bins = [half, -half[:-1][::-1]].
    half = numpy.sign(numpy.linspace(-20, 0, 128)) * numpy.expm1(
        numpy.abs(numpy.linspace(-20, 0, 128))
    )
    official = numpy.concatenate([half, -half[:-1][::-1]])
    # float32 against float64 arithmetic: equal to single precision.
    assert numpy.allclose(bins.numpy(), official, rtol=1e-6, atol=0.0)


def test_twohot_loss_encodes_between_the_two_neighbouring_bins() -> None:
    bins = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    uniform = torch.zeros(4)
    # Anything encoded against uniform logits costs log 4.
    assert twohot_loss(uniform, torch.tensor(0.5), bins).item() == pytest.approx(math.log(4))
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    logp = torch.log_softmax(logits, -1)
    cases = {
        0.25: {1: 0.75, 2: 0.25},
        0.5: {1: 0.5, 2: 0.5},
        1.0: {2: 1.0},  # exactly on a bin: all weight there
        5.0: {3: 1.0},  # past the last bin: all weight on it
        -3.0: {0: 1.0},
    }
    for target, weights in cases.items():
        expected = -sum(weight * logp[index].item() for index, weight in weights.items())
        assert twohot_loss(logits, torch.tensor(target), bins).item() == pytest.approx(expected)


def test_the_twohot_mean_of_an_encoding_recovers_its_target() -> None:
    bins = twohot_bins(255)
    targets = torch.tensor([-100.0, -3.7, -0.01, 0.0, 0.2, 1.0, 42.0])
    for target in targets:
        # The loss is the cross-entropy against the encoding, so the encoding
        # is recovered as the gradient of the loss at zero logits plus 1/n.
        logits = torch.zeros(255, requires_grad=True)
        twohot_loss(logits, target, bins).backward()  # type: ignore[no-untyped-call]
        assert logits.grad is not None
        encoded = 1.0 / 255 - logits.grad
        recovered = twohot_mean(encoded.clamp(min=0).log(), bins)
        assert recovered.item() == pytest.approx(target.item(), rel=1e-4, abs=1e-5)


def test_uniform_probabilities_over_symmetric_bins_predict_exactly_zero() -> None:
    assert twohot_mean(torch.zeros(3, 255), twohot_bins(255)).tolist() == [0.0, 0.0, 0.0]


def test_the_lambda_return_matches_hand_computed_values() -> None:
    reward = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    value = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    zeros = torch.zeros(1, 4)
    plain = lambda_return(zeros, zeros, reward, value, 0.5, 0.5)
    assert plain.tolist() == [[9.8125, 15.25, 23.0]]
    # A terminal at step 2 ends the return of step 1 with its reward.
    term = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    assert lambda_return(zeros, term, reward, value, 0.5, 0.5).tolist() == [
        [6.5, 2.0, 23.0]
    ]
    # A last flag at step 2 makes step 1 bootstrap from step 2's value alone.
    last = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    assert lambda_return(last, zeros, reward, value, 0.5, 0.5).tolist() == [
        [10.25, 17.0, 23.0]
    ]


def test_the_lambda_return_matches_the_recursive_definition() -> None:
    generator = numpy.random.default_rng(0)
    batch, time, discount, lam = 3, 9, 0.9, 0.8
    reward = generator.normal(size=(batch, time))
    boot = generator.normal(size=(batch, time))
    term = generator.random((batch, time)) < 0.2
    last = generator.random((batch, time)) < 0.2
    expected = numpy.zeros((batch, time - 1))
    for row in range(batch):
        following = boot[row, -1]
        for step in reversed(range(time - 1)):
            live = (1 - term[row, step + 1]) * discount
            mix = (1 - last[row, step + 1]) * lam
            following = reward[row, step + 1] + live * (
                (1 - mix) * boot[row, step + 1] + mix * following
            )
            expected[row, step] = following
    actual = lambda_return(
        torch.tensor(last),
        torch.tensor(term),
        torch.tensor(reward),
        torch.tensor(boot),
        discount,
        lam,
    )
    assert numpy.allclose(actual.numpy(), expected)


def test_the_percentile_normaliser_tracks_5th_and_95th_percentiles() -> None:
    normaliser = PercentileNormaliser()
    values = torch.arange(101, dtype=torch.float32)
    normaliser.update(values)
    offset, scale = normaliser.stats()
    assert offset.item() == pytest.approx(0.05)
    # 0.95 - 0.05 is below the limit of 1, so returns are never scaled up.
    assert scale.item() == pytest.approx(1.0)
    for _ in range(2000):
        normaliser.update(values)
    offset, scale = normaliser.stats()
    assert offset.item() == pytest.approx(5.0, rel=1e-4)
    assert scale.item() == pytest.approx(90.0, rel=1e-4)
    restored = PercentileNormaliser()
    restored.load_state_dict(normaliser.state_dict())
    assert [t.item() for t in restored.stats()] == [t.item() for t in normaliser.stats()]


def test_latent_unimix_floors_every_class_at_one_percent_over_n() -> None:
    logits = torch.tensor([[50.0, 0.0, -50.0, 0.0]])
    probs = unimix_probs(logits, 0.01)
    assert probs.sum().item() == pytest.approx(1.0)
    assert probs.min().item() >= 0.01 / 4 - 1e-9
    same = categorical_kl(probs, probs)
    assert same.item() == pytest.approx(0.0, abs=1e-7)
    other = unimix_probs(torch.zeros(1, 4), 0.01)
    expected = float((probs * (probs.log() - other.log())).sum())
    assert categorical_kl(probs, other).item() == pytest.approx(expected)


def test_the_masked_policy_gives_invalid_actions_exactly_zero() -> None:
    logits = torch.tensor([[5.0, 100.0, -3.0, 0.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False, True]])
    log_probs = masked_policy_log_probs(logits, mask, 0.01)
    probs = log_probs.exp()
    assert probs[0, 1].item() == 0.0 and probs[0, 3].item() == 0.0
    assert probs.sum().item() == pytest.approx(1.0)
    # 1% over the three valid actions is the floor of each.
    assert probs[0, 2].item() >= 0.01 / 3
    masked_entropy(log_probs, mask).sum().backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    uniform = masked_policy_log_probs(torch.zeros(1, 5), mask, 0.01)
    assert masked_entropy(uniform, mask).item() == pytest.approx(math.log(3))


def test_inverse_cdf_sampling_never_draws_a_zero_probability_entry() -> None:
    probs = torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0]])
    for u, expected in ((0.0, 1), (0.49, 1), (0.5, 3), (0.999999, 3), (1.0, 3)):
        assert sample_index(probs, torch.tensor([u])).item() == expected
    many = torch.rand(10_000)
    drawn = sample_index(probs.expand(10_000, 5), many)
    assert set(drawn.tolist()) == {1, 3}
    assert abs((drawn == 1).float().mean().item() - 0.5) < 0.03


def _official_laprop(
    parameter: numpy.ndarray,
    gradients: list[numpy.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    agc: float,
    pmin: float,
    warmup: int,
) -> numpy.ndarray:
    """`clip_by_agc`, `scale_by_rms`, `scale_by_momentum`, warm-up schedule (opt.py)."""
    nu = numpy.zeros_like(parameter)
    mu = numpy.zeros_like(parameter)
    for count, gradient in enumerate(gradients):
        upper = agc * max(pmin, numpy.linalg.norm(parameter))
        update = gradient / max(1.0, numpy.linalg.norm(gradient) / upper)
        step = count + 1
        nu = beta2 * nu + (1 - beta2) * update * update
        update = update / (numpy.sqrt(nu / (1 - beta2**step)) + eps)
        mu = beta1 * mu + (1 - beta1) * update
        rate = lr * min(count, warmup) / warmup
        parameter = parameter - rate * mu / (1 - beta1**step)
    return parameter


def test_laprop_matches_the_official_optimiser_chain() -> None:
    generator = numpy.random.default_rng(1)
    start = generator.normal(size=(3, 4))
    # Large and small gradients, so AGC clips some steps and not others.
    gradients = [generator.normal(size=(3, 4)) * scale for scale in (5.0, 0.01, 2.0, 0.1, 1.0)]
    settings = dict(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-20, agc=0.3, warmup=3)
    expected = _official_laprop(start, gradients, pmin=1e-3, **settings)  # type: ignore[arg-type]

    parameter = torch.nn.Parameter(torch.tensor(start))
    optimiser = LaProp([parameter], agc_floor=1e-3, **settings)  # type: ignore[arg-type]
    moved_on_first_step = None
    for gradient in gradients:
        parameter.grad = torch.tensor(gradient)
        optimiser.step()
        if moved_on_first_step is None:
            moved_on_first_step = not numpy.array_equal(parameter.detach().numpy(), start)
    # The warm-up starts at a learning rate of exactly zero.
    assert moved_on_first_step is False
    assert numpy.allclose(parameter.detach().numpy(), expected, rtol=1e-10, atol=1e-12)
