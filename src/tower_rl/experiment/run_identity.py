"""What one training run is, and what it was configured with.

A run that cannot name itself - which code produced it, which floors its curve
is read against, exactly which settings it was fixed with - cannot be compared
with the next one, so identity is resolved once, up front, and travels with
every artefact the run writes.

The checkpoint carries its own identity (`learning/checkpoint.CheckpointIdentity`)
because a checkpoint has to refuse an incompatible resume on its own; this
module resolves the run id and the configuration snapshot that identity is built
from.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields

import torch

from tower_rl.environment.episode import REWARD_SCHEMA_VERSION
from tower_rl.environment.run_actions import ACTION_SCHEMA_VERSION
from tower_rl.environment.run_environment import (
    CadenceConfig,
    DecisionCadence,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import OBSERVATION_SCHEMA_VERSION
from tower_rl.learning.checkpoint import CheckpointIdentity
from tower_rl.learning.dreamer import DreamerConfig
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.stacked_dqn import StackedDqnConfig
from tower_rl.learning.training import TrainingConfig

#: The measured floors a learning curve has to be read against, carried in every
#: report so the curve is legible without a second document. Mean final wave over
#: 23 valid episodes per arm, `M1B-E021` at commit 86fcf3c.
REFERENCE_FINAL_WAVES: dict[str, object] = {
    "metric": "mean final wave",
    "episodes_per_arm": 23,
    "source": "M1B-E021 at commit 86fcf3c",
    "scripted": 5.57,
    "random": 5.35,
    "wait": 1.87,
}

#: The floor a learned arm has to clear to mean anything.
SCRIPTED_REFERENCE = 5.57


def source_revision() -> str:
    """Bind every artifact to the code that produced it."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def new_run_id(name: str) -> str:
    """A run id that sorts by when it was started and collides with nothing."""
    return f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class RunIdentity:
    """Who produced a run: which arm, on which device profile, at which code.

    Everything a run is named by that a caller actually chooses. The schemas a
    checkpoint is keyed on are not among them: they are properties of the code
    this run is, which is why `checkpoint_identity` reads them here rather than
    taking them from a script.
    """

    run_id: str
    backbone: str
    profile_id: str
    source_revision: str
    #: Which cadence this run collects under (ADR 0009). A choice the operator
    #: makes, like the arm and the profile, which is why it lives here and not
    #: with the schema versions below.
    decision_cadence: DecisionCadence = DecisionCadence.CHOICE_POINTS
    #: Which upgrade rows this run plays with (ADR 0011). The same kind of
    #: choice as the cadence - the operator makes it, and it changes what the
    #: episodes mean - so it is named here beside it. The profile id stays the
    #: image's either way: availability is applied at each round start, not
    #: baked into the image.
    upgrade_availability: UpgradeAvailability = UpgradeAvailability.IMAGE

    @classmethod
    def started_now(
        cls,
        backbone: str,
        *,
        profile_id: str,
        source_revision: str,
        decision_cadence: DecisionCadence = DecisionCadence.CHOICE_POINTS,
        upgrade_availability: UpgradeAvailability = UpgradeAvailability.IMAGE,
    ) -> RunIdentity:
        """A fresh identity for a run about to start, with a new run id."""
        return cls(
            run_id=new_run_id(backbone),
            backbone=backbone,
            profile_id=profile_id,
            source_revision=source_revision,
            decision_cadence=decision_cadence,
            upgrade_availability=upgrade_availability,
        )


def checkpoint_identity(identity: RunIdentity) -> CheckpointIdentity:
    """The compatibility key the run's checkpoints are written under.

    The one place the observation, action and reward schema versions are read
    into an identity. `CheckpointIdentity.incompatibilities` deliberately
    ignores `run_id`: a checkpoint from an earlier run may be resumed into this
    one as long as the arm, the profile and all three schemas match.
    """
    return CheckpointIdentity(
        run_id=identity.run_id,
        backbone=identity.backbone,
        profile_id=identity.profile_id,
        observation_schema=OBSERVATION_SCHEMA_VERSION,
        action_schema=ACTION_SCHEMA_VERSION,
        reward_schema=REWARD_SCHEMA_VERSION,
        source_revision=identity.source_revision,
        decision_cadence=identity.decision_cadence,
        upgrade_availability=identity.upgrade_availability,
    )


def resolved_config(
    name: str,
    arguments: argparse.Namespace,
    *,
    actor_ids: Sequence[str],
    config: TrainingConfig,
    learner: StackedDqnConfig,
    network: NetworkConfig,
    cadence: CadenceConfig,
    decision_cadence: DecisionCadence,
    upgrade_availability: UpgradeAvailability,
    burn_in: int,
    stride: int,
    device: torch.device,
    parent_checkpoint: str | None = None,
) -> dict[str, object]:
    """Everything the run was actually fixed with, as one flat snapshot.

    Taken from what was built rather than from what was asked for on the command
    line - the cadence the environment holds, the window the training config
    holds - because those are what the run was collected under.
    """
    return {
        "backbone": name,
        "budget_decisions": config.budget_decisions,
        # The checkpoint this run continues, as `learning.checkpoint.ResumeState`
        # cites it: the file and the identity hash of the run that wrote it.
        # None for a run that started from scratch. The budget beside it is the
        # whole run's, not this segment's - a resume continues a budget, it does
        # not start a second one.
        "parent_checkpoint": parent_checkpoint,
        # The fleet this arm actually collected with, and the instances it
        # addressed - one actor per emulator instance.
        "actors": len(actor_ids),
        "actor_ids": list(actor_ids),
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "warmup_sequences": config.warmup_sequences,
        "gradient_steps_per_decision": arguments.gradient_steps_per_decision,
        "sequence_length": arguments.sequence_length,
        # The burn-in this arm was built with: exactly what fills the window.
        "burn_in": burn_in,
        "stride": stride,
        "history_length": arguments.history_length if name == "stacked-dqn" else None,
        # The shape of the network, not only its hyperparameters: a checkpoint
        # whose snapshot cannot say how wide its layers were cannot be rebuilt
        # into the policy that wrote it, which is what an evaluation of a
        # numbered checkpoint has to do.
        "network_identity_capacity": network.identity_capacity,
        "network_identity_dim": network.identity_dim,
        "network_hidden": network.hidden,
        "network_core_hidden": network.core_hidden,
        "n_step": learner.n_step,
        # The n-step anneal, if any: `n_step` is where it starts. None and 0
        # hold `n_step` fixed, which is what every run before run 4 used.
        "n_step_final": learner.n_step_final,
        "n_step_anneal_steps": learner.n_step_anneal_steps,
        # Only the discount the target is built with: under the game-time
        # discount the per-decision one is not read, and is recorded as None
        # rather than as a value that played no part.
        "discount": (
            None if learner.discount_per_game_second is not None else learner.discount
        ),
        # None discounts per decision, which is every run before board #81.
        "discount_per_game_second": learner.discount_per_game_second,
        # False learns from the wave reward, which is every run before board #82.
        "survival_time_reward": learner.survival_time_reward,
        # False explores one decision at a time, which is every run before
        # board #83.
        "ez_greedy": learner.ez_greedy,
        "learning_rate": learner.learning_rate,
        "target_ema_decay": (
            arguments.target_ema_decay if name == "stacked-dqn" else None
        ),
        "epsilon_start": config.exploration.epsilon_start,
        "epsilon_end": config.exploration.epsilon_end,
        "epsilon_anneal_decisions": config.exploration.anneal_decisions,
        # Which exploration this arm collected under, and the per-actor floors
        # it resolved to: under a ladder the fleet's actors sit at rates two
        # orders of magnitude apart, and a curve read months later cannot be
        # told from a uniform one without them. Empty under `uniform`, which has
        # no per-actor floor at all.
        "exploration": config.exploration.option,
        "exploration_epsilon_floors": list(config.exploration.floors),
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "priority_alpha": arguments.priority_alpha,
        "replay_capacity": arguments.replay_capacity,
        "collection_window_episodes": config.collection_window_episodes,
        "evaluate_every_episodes": arguments.evaluate_every_episodes,
        "evaluation_episodes": arguments.evaluation_episodes,
        "checkpoint_every_episodes": arguments.checkpoint_every_episodes,
        "checkpoint_every_decisions": config.checkpoint_every_decisions,
        "selection_period_decisions": config.selection_period_decisions,
        # What the run was allowed to stop itself on. A run that ended before
        # its budget has to be readable as a decision rather than as an
        # interruption, and these are the thresholds that decision was made
        # under. Zero patience is off, which is what every run so far spent its
        # whole budget under.
        "early_stop_patience_periods": config.early_stop_patience_periods,
        "early_stop_min_improvement": config.early_stop_min_improvement,
        # The pre-registered kill bars, as (at, window start, minimum mean).
        "kill_bars": [
            [bar.at_decisions, bar.window_start_decisions, bar.min_mean_final_wave]
            for bar in config.kill_bars
        ],
        # The parameter lag the fleet acted under, which a later reading of the
        # collection curve needs as much as the replay ratio.
        "parameter_sync_episodes": config.parameter_sync_episodes,
        # The cadence the environment was actually built with, not what was
        # asked for on the command line.
        "frame_game_ms": cadence.frame_game_ms,
        "max_quiet_game_ms": cadence.max_quiet_game_ms,
        "health_change_fraction": cadence.health_change_fraction,
        # Which of those cadence stops the policy was actually asked about.
        # What a decision means depends on it: run 1 counted slices, a
        # choice-point run counts choices (ADR 0009).
        "decision_cadence": str(decision_cadence),
        # Which upgrade rows the policy could actually buy. `image` is the
        # profile image's own six; `all` reopens every real row at each round
        # start, which is a different decision problem and a different set of
        # baselines (ADR 0011).
        "upgrade_availability": str(upgrade_availability),
        "device": str(device),
        # The guest rate this arm actually collected at: a fleet run raises
        # every instance to it, and a single actor's is whatever the operator
        # brought their instance up at, so a run's identity carries the rate
        # its curve was measured under rather than leaving it to be inferred.
        "frame_rate_hz": arguments.frame_rate_hz,
    }


def dreamer_resolved_config(
    resolved: dict[str, object], config: DreamerConfig, *, mixed_precision: bool
) -> dict[str, object]:
    """A DreamerV3 run's snapshot: `resolved_config`'s, with DreamerV3's own settings.

    The stacked-dqn learner and network settings do not describe this run and
    are recorded as None. Every `DreamerConfig` value is recorded under
    `dreamer_<field>`, which is what `checkpoint_policy` rebuilds the policy from.
    `dreamer_compute_dtype` is the learner's compute precision (bfloat16 on
    CUDA, `DreamerBackbone.mixed_precision`); it is no `DreamerConfig` field,
    so the rebuild never reads it.
    """
    stacked = {item.name for item in fields(StackedDqnConfig)} - {"seed"}
    stacked |= {f"network_{item.name}" for item in fields(NetworkConfig)}
    return {
        **{key: None if key in stacked else value for key, value in resolved.items()},
        **{f"dreamer_{key}": value for key, value in asdict(config).items()},
        "dreamer_compute_dtype": "bfloat16" if mixed_precision else "float32",
    }


def tracked_params(resolved: dict[str, object]) -> dict[str, object]:
    """The resolved settings plus the floors, as the tracker records them.

    The floors the curve is read against travel with the run, so a comparison
    opened months later is self-contained.
    """
    params: dict[str, object] = dict(resolved)
    params.update({f"reference_{key}": value for key, value in REFERENCE_FINAL_WAVES.items()})
    return params
