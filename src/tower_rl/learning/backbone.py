"""The interface every candidate algorithm implements.

A backbone is addressed only through this protocol, so the environment,
observation, action space and evaluation path stay outside it and a change of
algorithm cannot quietly change what it is measured against.
"""

from __future__ import annotations

import copy as copying
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy
import torch
from torch import Tensor, nn

from tower_rl.environment.features import StateFeatures


@dataclass(frozen=True)
class LearnMetrics:
    """What one optimisation step reports, for tracking and for priorities.

    The two scalars a run's health is read from are named for whether the
    importance-sampling weights are in them, because that distinction is what
    made the first run unreadable: the weighted loss fell over the run largely
    because beta annealed the weights upwards, not because the learner improved.
    """

    #: The optimised quantity, scaled by the per-sequence importance-sampling
    #: weights where the replay has them, and therefore confounded with the
    #: priorities the batch was sampled by.
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
    #: The largest online Q of a taken action over the batch's real steps,
    #: before this step's update: what a value overshooting its bound (the
    #: survival-time reward's `V_REF`) is read from live. `None` for a learner
    #: that does not report it.
    taken_q_max: float | None = None
    #: A backbone's own monitors of this batch, by name: DreamerV3's checks of
    #: its continue head and decoded mask (`DreamerBackbone.learn`). Empty for
    #: a learner that has none.
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    #: The posterior latents of the batch's trained steps, `deter` [batch,
    #: time, deter] and the stochastic state's class indices [batch, time,
    #: stoch], on the CPU: what DreamerV3 writes back into replay after each
    #: step (`DreamerReplay.write_back`). None for a learner that stores none.
    latents: tuple[Tensor, Tensor] | None = None
    #: Each sampled item's replay priority [batch], on the CPU, from a learner
    #: that computes them itself as Acme's R2D2 learner does
    #: (`R2D2Replay.update_priorities`); None for one that reports
    #: `td_errors` instead.
    priorities: numpy.ndarray | None = None


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
    #: Game time each transition spanned, in ms; 0 for a purchase and for
    #: padding. [batch, time]
    game_ms: Tensor
    weights: Tensor  # [batch]
    burn_in: int
    #: A batch is in DreamerV3's layout (`DreamerReplay`):
    #: `rewards`, `dones` and `game_ms` describe the transition *into* each
    #: step and `actions` the action taken at it; the first `burn_in` steps
    #: are the replay context, read only for `context`, the stored latent of
    #: the last of them - `deter` [batch, deter] and stochastic class indices
    #: [batch, stoch]. `first` and `last` [batch, time] are `is_first` and
    #: `is_last`.
    #: R2D2's replay (`R2D2Replay`) uses the same step layout, with its own
    #: meanings: `burn_in` is the steps unrolled before the trace, `first`
    #: and `last` mark an episode's first and last steps, and `context` is
    #: the stored LSTM state (h, c), each [batch, state], from before the
    #: window's first step.
    first: Tensor | None = None
    last: Tensor | None = None
    context: tuple[Tensor, Tensor] | None = None
    #: R2D2's observation-action-reward input: the action taken at the step
    #: before each step, 0 at an episode's start. [batch, time] or None.
    previous_actions: Tensor | None = None

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

    def network_state_dict(self) -> dict[str, Any]:
        """What acting reads - the networks and the step they are from - and no optimizer state.

        What the learner publishes to the acting copies (`Learner.publish`),
        so a copy never holds optimizer moments it does not use.
        """
        ...

    def load_network_state_dict(self, state: dict[str, Any]) -> None:
        """Load `network_state_dict` into an acting copy."""
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
