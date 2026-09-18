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

import torch

from tower_rl.application.training import TrainingConfig
from tower_rl.environment.run_environment import CadenceConfig
from tower_rl.learning.stacked_dqn import StackedDqnConfig

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


def resolved_config(
    name: str,
    arguments: argparse.Namespace,
    *,
    actor_ids: Sequence[str],
    config: TrainingConfig,
    learner: StackedDqnConfig,
    cadence: CadenceConfig,
    burn_in: int,
    stride: int,
    device: torch.device,
) -> dict[str, object]:
    """Everything the run was actually fixed with, as one flat snapshot.

    Taken from what was built rather than from what was asked for on the command
    line - the cadence the environment holds, the window the training config
    holds - because those are what the run was collected under.
    """
    return {
        "backbone": name,
        "budget_decisions": arguments.budget_decisions,
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
        "n_step": learner.n_step,
        "discount": learner.discount,
        "learning_rate": learner.learning_rate,
        "target_ema_decay": (
            arguments.target_ema_decay if name == "stacked-dqn" else None
        ),
        "epsilon_start": config.epsilon_start,
        "epsilon_end": config.epsilon_end,
        "epsilon_anneal_decisions": config.epsilon_anneal_decisions,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "priority_alpha": arguments.priority_alpha,
        "replay_capacity": arguments.replay_capacity,
        "collection_window_episodes": config.collection_window_episodes,
        "evaluate_every_episodes": arguments.evaluate_every_episodes,
        "evaluation_episodes": arguments.evaluation_episodes,
        "checkpoint_every_episodes": arguments.checkpoint_every_episodes,
        # The parameter lag the fleet acted under, which a later reading of the
        # collection curve needs as much as the replay ratio.
        "parameter_sync_episodes": config.parameter_sync_episodes,
        # The cadence the environment was actually built with, not what was
        # asked for on the command line.
        "frame_game_ms": cadence.frame_game_ms,
        "max_quiet_game_ms": cadence.max_quiet_game_ms,
        "health_change_fraction": cadence.health_change_fraction,
        "block_decisions": arguments.block_decisions,
        "device": str(device),
    }


def tracked_params(resolved: dict[str, object]) -> dict[str, object]:
    """The resolved settings plus the floors, as the tracker records them.

    The floors the curve is read against travel with the run, so a comparison
    opened months later is self-contained.
    """
    params: dict[str, object] = dict(resolved)
    params.update({f"reference_{key}": value for key, value in REFERENCE_FINAL_WAVES.items()})
    return params


__all__ = [
    "REFERENCE_FINAL_WAVES",
    "SCRIPTED_REFERENCE",
    "new_run_id",
    "resolved_config",
    "source_revision",
    "tracked_params",
]
