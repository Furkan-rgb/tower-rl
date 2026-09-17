"""What every value-based backbone in the comparison shares.

The n-step double-Q target is one concept, not one per algorithm. Keeping a
single definition means a subtle bootstrapping mistake cannot be fixed in one
backbone and left in another, which would silently make a comparison between
them a comparison of two different target definitions.
"""

from __future__ import annotations

import torch
from torch import Tensor


def evaluated_next_values(online_q: Tensor, target_q: Tensor, mask: Tensor) -> Tensor:
    """Double Q: the online network chooses, the target network values.

    That split is what stops one optimistic network from bootstrapping its own
    overestimate. A state with no valid action is terminal and must contribute
    zero rather than the negative infinity the mask would otherwise carry.
    """
    best = online_q.argmax(dim=-1, keepdim=True)
    evaluated = target_q.gather(-1, best).squeeze(-1)
    evaluated = torch.where(mask.any(dim=-1), evaluated, torch.zeros_like(evaluated))
    return torch.where(torch.isfinite(evaluated), evaluated, torch.zeros_like(evaluated))


def n_step_targets(
    rewards: Tensor,
    dones: Tensor,
    online_q: Tensor,
    target_q: Tensor,
    mask: Tensor,
    *,
    discount: float,
    n_step: int,
) -> tuple[Tensor, Tensor]:
    """Double Q n-step targets, and which steps have a well-defined one.

    The bootstrap comes from the state `n` steps ahead inside the same sequence,
    so a step whose window runs past the end is not learnable here. Termination is
    per sequence element, never collapsed across the batch.
    """
    batch, time = rewards.shape
    evaluated = evaluated_next_values(online_q, target_q, mask)

    targets = torch.zeros_like(rewards)
    learnable = torch.zeros_like(rewards)
    terminal = dones.to(rewards.dtype)
    for step in range(time):
        accumulated = torch.zeros(batch, device=rewards.device)
        alive = torch.ones(batch, device=rewards.device)
        factor = 1.0
        ended_inside_window = torch.zeros(batch, device=rewards.device)
        for offset in range(n_step):
            index = step + offset
            if index >= time:
                break
            accumulated = accumulated + alive * factor * rewards[:, index]
            factor *= discount
            alive = alive * (1.0 - terminal[:, index])
            ended_inside_window = torch.maximum(ended_inside_window, terminal[:, index])
        bootstrap = step + n_step
        if bootstrap < time:
            accumulated = accumulated + alive * factor * evaluated[:, bootstrap]
            learnable[:, step] = 1.0
        else:
            # Without a bootstrap state the return is only complete if the
            # episode actually ended inside the window.
            learnable[:, step] = ended_inside_window
        targets[:, step] = accumulated
    return targets, learnable


def weighted_sequence_loss(
    chosen: Tensor,
    targets: Tensor,
    learnable: Tensor,
    weights: Tensor,
    *,
    huber_delta: float,
) -> Tensor:
    """Huber loss over the learnable steps only, weighted per sequence.

    Steps whose n-step window runs past the end of the sequence have no
    well-defined target, and padded steps are not experience at all. Both are
    excluded from the sum and from the count it is divided by, so neither can
    contribute a zero that dilutes the loss of the steps that are real.
    """
    loss_per_step = torch.nn.functional.huber_loss(
        chosen, targets, reduction="none", delta=huber_delta
    )
    counted = learnable.sum(dim=1).clamp(min=1.0)
    return ((loss_per_step * learnable).sum(dim=1) / counted * weights).mean()


def real_step_td_errors(absolute: Tensor, real: Tensor) -> tuple[tuple[float, ...], ...]:
    """Per-sequence absolute TD errors over real steps only.

    Padding is filler, not a step the learner was wrong about. Reporting a zero
    for it would drag the mean term of a sequence's priority down and make a
    padded sequence look duller than it is, which is exactly the sequence - a
    short episode, an early death - that prioritization should be surfacing.
    """
    kept = real.detach().cpu().bool()
    return tuple(
        tuple(row[flags].tolist())
        for row, flags in zip(absolute.detach().cpu(), kept, strict=True)
    )
