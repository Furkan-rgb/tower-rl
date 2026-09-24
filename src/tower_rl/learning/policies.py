"""Policies that share the learner's interface, including the non-learned floors.

Random and scripted play are not scaffolding: without them, "it learned" is
unfalsifiable.  They implement the same protocol as a backbone so the actor,
evaluator and reporting path are identical for every arm of the comparison.

A trained checkpoint is an arm on the same terms (`checkpoint_policy`): it is
rebuilt into the backbone that wrote it and handed back as a policy, so an
evaluation of a checkpoint and an evaluation of the scripted floor differ in
nothing but which policy is asked for an action.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Protocol

import torch

from tower_rl.environment.features import ROW_FEATURES, ROW_WIDTH, StateFeatures
from tower_rl.environment.run_actions import RUN_ACTIONS
from tower_rl.learning.checkpoint import Checkpoint, CheckpointIdentity, load
from tower_rl.learning.dreamer import DREAMERV3, DreamerBackbone, DreamerConfig
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig


class Policy(Protocol):
    """Anything that can choose a valid action, learned or not."""

    def initial_state(self) -> Any: ...

    def act(self, features: StateFeatures, state: Any, *, epsilon: float) -> tuple[int, Any]: ...


def valid_actions(features: StateFeatures) -> list[int]:
    return [index for index, allowed in enumerate(features.mask) if allowed]


@dataclass
class RandomPolicy:
    """Uniform over currently valid actions. The floor every arm must clear."""

    seed: int | None = None
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def initial_state(self) -> None:
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        choices = valid_actions(features)
        if not choices:
            raise ValueError("no action is available in this state")
        return self._random.choice(choices), None


@dataclass
class CheapestFirstPolicy:
    """Buy the cheapest affordable upgrade, otherwise wait.

    This is the scripted policy measured in `M1B-E003`, reaching wave eight to ten
    where buying nothing dies at wave two. It is the harder floor: beating random
    proves very little, beating this proves something.
    """

    #: Index of `cost_log` inside a row, which orders identically to raw cost.
    #: Derived from the schema rather than written as a number: the row layout
    #: grew in `observation-v2`, and a hand-kept 0 would have gone on naming
    #: whichever feature happened to land first and quietly compared the wrong
    #: column - a floor that is silently not the cheapest-first floor.
    cost_feature: int = field(default_factory=lambda: ROW_FEATURES.index("cost_log"))

    def initial_state(self) -> None:
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        choices = valid_actions(features)
        if not choices:
            raise ValueError("no action is available in this state")
        purchases = [index for index in choices if index != 0]
        if not purchases:
            return 0, None
        cheapest = min(purchases, key=lambda index: self._cost(features, index))
        return cheapest, None

    def _cost(self, features: StateFeatures, action_index: int) -> float:
        # Action index 0 is WAIT, so row `i` backs action index `i + 1`.
        row = (action_index - 1) * ROW_WIDTH
        return features.rows[row + self.cost_feature]


@dataclass
class WaitOnlyPolicy:
    """Never buys. The degenerate reference that dies at wave two."""

    def initial_state(self) -> None:
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        if not features.mask[0]:
            raise ValueError("WAIT is not available in this state")
        return 0, None


def checkpoint_policy(
    path: Path,
    *,
    decision_cadence: str,
    upgrade_availability: str,
    device: torch.device | None = None,
    sampling_seed: str | None = None,
) -> tuple[StackedDqnBackbone | DreamerBackbone, CheckpointIdentity]:
    """Rebuild the backbone a checkpoint holds, as a policy to evaluate.

    Everything needed to reconstruct it is in the checkpoint: its identity says
    which backbone and which schemas, and the resolved config it was written
    with says how the learner and the network were shaped. Nothing is taken from
    the caller, so a checkpoint cannot be evaluated under settings it was not
    trained under - a network rebuilt a layer wider would fail to load its own
    weights, and one rebuilt with a shorter history would load them and act on a
    window the run never saw.

    The two settings that are *not* in the weights are taken from the caller and
    checked: the cadence the policy will be asked at (ADR 0009) and the upgrade
    rows it will be offered (ADR 0011). Neither changes a tensor, so neither
    would fail to load - a checkpoint trained on the image's six purchasable
    rows would quietly play a fully unlocked run, and the wave it scored would
    be read against floors measured under something else. They are required
    arguments rather than optional ones because a caller that forgot to say is
    exactly the silent case; this is the refusal the resume path already has
    (`checkpoint.load(expected=...)`), through the same `incompatibilities`.

    On the CPU by default. Evaluation is one forward pass per decision against
    an emulator that takes orders of magnitude longer to answer, so a GPU buys
    nothing and a fleet of evaluating actors would contend for one.

    The identity comes back beside the policy because a record of what this
    played has to name which checkpoint played it.
    """
    checkpoint = load(path)
    played = replace(
        checkpoint.identity,
        decision_cadence=str(decision_cadence),
        upgrade_availability=str(upgrade_availability),
    )
    # Every other field is copied from the checkpoint, so only these two can
    # differ: `incompatibilities` stays the one place that says what a
    # difference means, and its reasons name both values.
    refusals = checkpoint.identity.incompatibilities(played)
    if refusals:
        raise ValueError(f"{path} cannot be played here: {'; '.join(refusals)}")
    if checkpoint.identity.backbone == DREAMERV3:
        return _dreamer_policy(checkpoint, device, sampling_seed), checkpoint.identity

    settings = checkpoint.resolved_config
    if checkpoint.identity.backbone != "stacked-dqn":
        raise ValueError(
            f"{path} holds a {checkpoint.identity.backbone!r} backbone; "
            "only stacked-dqn can be rebuilt as a policy"
        )
    defaults = NetworkConfig()
    learner_defaults = StackedDqnConfig()
    backbone = StackedDqnBackbone(
        config=StackedDqnConfig(
            history_length=int(settings["history_length"]),
            n_step=int(settings["n_step"]),
            # Absent from a checkpoint written before the n-step anneal, which
            # held n fixed - what these defaults rebuild.
            n_step_final=(
                None
                if settings.get("n_step_final") is None
                else int(settings["n_step_final"])
            ),
            n_step_anneal_steps=int(settings.get("n_step_anneal_steps", 0)),
            discount=float(settings["discount"]),
            learning_rate=float(settings["learning_rate"]),
            target_ema_decay=float(settings["target_ema_decay"]),
            # Absent from a checkpoint written before the BBF recipe, whose
            # optimizer was built from these defaults. They decide how the
            # optimizer is split into groups, which its state must match.
            weight_decay=float(settings.get("weight_decay", learner_defaults.weight_decay)),
            weight_decay_on_vectors=bool(
                settings.get(
                    "weight_decay_on_vectors", learner_defaults.weight_decay_on_vectors
                )
            ),
            adam_eps=float(settings.get("adam_eps", learner_defaults.adam_eps)),
            # The network the run acted with is the one its checkpoints are
            # evaluated with: the target under BBF's recipe, else the online.
            act_with_target=bool(
                settings.get("act_with_target", learner_defaults.act_with_target)
            ),
        ),
        network_config=NetworkConfig(
            identity_capacity=int(
                settings.get("network_identity_capacity", defaults.identity_capacity)
            ),
            identity_dim=int(settings.get("network_identity_dim", defaults.identity_dim)),
            hidden=int(settings.get("network_hidden", defaults.hidden)),
            core_hidden=int(settings.get("network_core_hidden", defaults.core_hidden)),
        ),
        device=device or torch.device("cpu"),
    )
    backbone.load_state_dict(dict(checkpoint.backbone_state))
    # Nothing here is ever trained again. `act` already builds no graph; this
    # says so at the object as well, so a policy that leaked into a learner
    # would fail rather than quietly accumulate gradients.
    for network in (backbone.online, backbone.target):
        network.eval()
        network.requires_grad_(False)
    return backbone, checkpoint.identity


def _dreamer_policy(
    checkpoint: Checkpoint, device: torch.device | None, sampling_seed: str | None
) -> DreamerBackbone:
    """A DreamerV3 checkpoint as a policy: its config is the `dreamer_*` keys it recorded.

    DreamerV3 samples its policy, so every evaluating instance needs a stream of
    its own (`sampling_seed`, e.g. its serial). Left None, every instance would
    draw the same uniforms from the run's seed.
    """
    settings = checkpoint.resolved_config
    config = DreamerConfig(
        **{name.name: settings[f"dreamer_{name.name}"] for name in fields(DreamerConfig)}
    )
    backbone = DreamerBackbone(config=config, device=device or torch.device("cpu"))
    backbone.load_state_dict(dict(checkpoint.backbone_state))
    for value in vars(backbone).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
            value.requires_grad_(False)
    if sampling_seed is not None:
        backbone._random.seed(sampling_seed)
    return backbone


def describe(policy: Policy) -> str:
    """A stable name for reports, so arms stay identifiable across runs."""
    return type(policy).__name__


assert len(RUN_ACTIONS) > 1, "the action space must contain WAIT plus upgrades"
