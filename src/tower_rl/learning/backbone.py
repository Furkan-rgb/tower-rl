"""The interface every candidate algorithm implements.

A backbone is addressed only through this protocol, so the environment,
observation, action space and evaluation path stay outside it and a change of
algorithm cannot quietly change what it is measured against.
"""

from __future__ import annotations

import copy as copying
import random
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from tower_rl.environment.features import ROW_COUNT, ROW_WIDTH, StateFeatures
from tower_rl.learning.replay import ReplaySequence


@dataclass(frozen=True)
class LearnMetrics:
    """What one optimisation step reports, for tracking and for priorities.

    The two scalars a run's health is read from are named for whether the
    importance-sampling weights are in them, because that distinction is what
    made the first run unreadable: the weighted loss fell over the run largely
    because beta annealed the weights upwards, not because the learner improved.
    """

    #: The optimised quantity: Huber loss scaled by the per-sequence
    #: importance-sampling weights, and therefore confounded with the beta
    #: schedule and the priority exponent.
    weighted_loss: float
    #: The same batch's mean absolute TD error with no weighting of any kind.
    #: This is the one to read a learning curve against.
    unweighted_mean_absolute_td_error: float
    gradient_norm: float
    #: Per-sequence absolute TD errors, in the order the batch was sampled.
    td_errors: tuple[tuple[float, ...], ...]
    #: Correlation between V(s_t) and the realised discounted return over the
    #: steps of this batch whose episode ended inside the stored sequence.
    #: `None` when the batch holds too few such steps to correlate.
    value_fit_correlation: float | None = None


@dataclass(frozen=True)
class SequenceBatch:
    """One collated batch of equal-length sequences."""

    scalars: Tensor  # [batch, time, scalars]
    rows: Tensor  # [batch, time, rows, width]
    mask: Tensor  # [batch, time, actions]
    actions: Tensor  # [batch, time]
    rewards: Tensor  # [batch, time]
    dones: Tensor  # [batch, time]
    #: True where a step is filler rather than experience. [batch, time]
    padding: Tensor
    weights: Tensor  # [batch]
    burn_in: int

    @property
    def batch_size(self) -> int:
        return int(self.scalars.shape[0])


class Backbone(Protocol):
    """One learning algorithm, addressed identically to every other."""

    @property
    def model_version(self) -> int:
        """Increments on every applied optimisation step."""
        ...

    @property
    def device(self) -> torch.device:
        """Where this backbone's parameters live, so a batch can be built there."""
        ...

    def act(
        self, features: StateFeatures, state: Any, *, epsilon: float
    ) -> tuple[int, Any]:
        """Choose one valid action index and return the carried state."""
        ...

    def initial_state(self) -> Any:
        """The carried state an episode starts from."""
        ...

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        """Apply one optimisation step and report what it found."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to resume this backbone exactly."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...


def acting_copy(backbone: Backbone, *, exploration_seed: int | str | None = None) -> Backbone:
    """One actor's own copy of a backbone, to act from without touching the learner.

    This is the actor-learner arrangement of Ape-X (Horgan et al. 2018) and R2D2
    (Kapturowski et al. 2019): actors act from copies of the network and the
    learner publishes its parameters into them periodically. Acting on
    parameters a few optimisation steps old is the accepted, intended cost of
    it. The alternative - every actor reading the one live network - serialises
    every forward pass against every gradient step, which caps a fleet's
    throughput as soon as the fleet is large.

    The copy lives on the same device as the original and shares no tensor with
    it, so a publication into it (`Learner.publish_to`) is invisible to the
    learner and its acting is invisible to every other actor. Its modules are
    put in evaluation mode with gradients switched off: nothing here is ever
    trained, and the acting path must build no graph.

    `exploration_seed` reseeds the copy's epsilon-greedy stream, if it keeps one
    in a `random.Random` as the learned backbone does. Left `None` the copy
    carries on the stream it was copied with, which is what keeps a fleet of one
    drawing exactly the exploration a single actor draws today; a fleet gives
    every actor after the first a seed of its own, or they would all explore in
    lockstep from the same copied stream.
    """
    acting = copying.deepcopy(backbone)
    for value in vars(acting).values():
        if isinstance(value, nn.Module):
            value.eval()
            value.requires_grad_(False)
    stream = getattr(acting, "_random", None)
    if exploration_seed is not None and isinstance(stream, random.Random):
        stream.seed(exploration_seed)
    return acting


def collate(
    sequences: tuple[ReplaySequence, ...],
    weights: tuple[float, ...],
    *,
    device: torch.device | None = None,
) -> SequenceBatch:
    """Stack equal-length sequences into tensors.

    Equal length is required: the actor emits fixed length windows, padding the
    front of an episode too short to fill one rather than dropping it. Padding is
    carried as its own mask, separate from the action mask, and the only thing it
    ever does is remove a step from the learning window - a padded step is never a
    target and never contributes a TD error.
    """
    if not sequences:
        raise ValueError("a batch needs at least one sequence")
    length = len(sequences[0].steps)
    burn_in = sequences[0].burn_in
    if any(len(sequence.steps) != length for sequence in sequences):
        raise ValueError("all sequences in a batch must have the same length")
    if any(sequence.burn_in != burn_in for sequence in sequences):
        raise ValueError("all sequences in a batch must share one burn-in length")

    def _features(step_features: StateFeatures) -> tuple[list[float], list[float], list[bool]]:
        return list(step_features.scalars), list(step_features.rows), list(step_features.mask)

    scalars, rows, masks, actions, rewards, dones, padding = [], [], [], [], [], [], []
    for sequence in sequences:
        collected = [_features(step.features) for step in sequence.steps]
        scalars.append([item[0] for item in collected])
        rows.append([item[1] for item in collected])
        masks.append([item[2] for item in collected])
        actions.append([step.action_index for step in sequence.steps])
        rewards.append([step.reward for step in sequence.steps])
        dones.append([step.done for step in sequence.steps])
        padding.append([step.padding for step in sequence.steps])

    row_tensor = torch.tensor(rows, dtype=torch.float32, device=device)
    return SequenceBatch(
        scalars=torch.tensor(scalars, dtype=torch.float32, device=device),
        rows=row_tensor.view(len(sequences), length, ROW_COUNT, ROW_WIDTH),
        mask=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.int64, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.bool, device=device),
        padding=torch.tensor(padding, dtype=torch.bool, device=device),
        weights=torch.tensor(weights, dtype=torch.float32, device=device),
        burn_in=burn_in,
    )
