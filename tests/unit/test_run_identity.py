"""What a run is, and what it records itself as having been configured with.

`experiment/run_identity.py` resolves the identity every artefact of a run is
bound to: the run id, the revision that produced it, and the flat snapshot of
everything the run was actually fixed with. The snapshot is checked through the
run it is taken from, because what matters is not that the function copies its
arguments but that the settings the run was built with are the ones it records.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train  # noqa: E402
from test_train_entry_point import PROFILE, SMALL_NETWORK, arguments, environment  # noqa: E402

from tower_rl.experiment.run_identity import (  # noqa: E402
    REFERENCE_FINAL_WAVES,
    new_run_id,
    source_revision,
    tracked_params,
)

torch.set_num_threads(1)


def _arm(run_dir: Path, **overrides: str) -> Any:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "NetworkConfig", lambda: SMALL_NETWORK)
        return train.build_arm(
            train.BACKBONE,
            arguments(run_dir, **overrides),
            instances=[train.ActorInstance(serial="fake-0", environment=environment())],
            device=torch.device("cpu"),
            profile_id=PROFILE,
            parent=run_dir,
            revision="test",
            started=0.0,
            tracker=train.NoExperimentTracker(),
            tags={},
        )


def test_every_flag_reaches_the_thing_it_configures(tmp_path: Path) -> None:
    """A flag that reaches nothing is worse than no flag: it looks like a knob."""
    arm = _arm(
        tmp_path,
        **{
            "--n-step": "3",
            "--discount": "0.9",
            "--learning-rate": "0.002",
            "--target-ema-decay": "0.9",
            "--warmup-sequences": "7",
            "--epsilon-start": "0.8",
            "--epsilon-end": "0.02",
            "--epsilon-anneal-decisions": "77",
            "--priority-alpha": "0.3",
            "--collection-window-episodes": "5",
            "--gradient-steps-per-decision": "0.25",
            "--batch-size": "4",
            "--parameter-sync-episodes": "4",
        },
    )

    learner = arm.backbone.config
    assert (learner.n_step, learner.discount, learner.learning_rate) == (3, 0.9, 0.002)
    assert learner.target_ema_decay == 0.9
    assert arm.replay.alpha == 0.3
    config = arm.training.config
    assert config.warmup_sequences == 7
    assert (config.epsilon_start, config.epsilon_end) == (0.8, 0.02)
    assert config.epsilon_anneal_decisions == 77
    assert config.collection_window_episodes == 5
    assert (config.batch_size, config.gradient_steps_per_decision) == (4, 0.25)
    assert config.parameter_sync_episodes == 4
    # And the run records what it was actually built with.
    resolved = arm.resolved
    assert resolved["n_step"] == 3 and resolved["discount"] == 0.9
    assert resolved["priority_alpha"] == 0.3
    assert resolved["epsilon_anneal_decisions"] == 77
    assert resolved["target_ema_decay"] == 0.9
    assert resolved["parameter_sync_episodes"] == 4


def test_the_burn_in_the_arm_is_built_with_is_the_one_that_fills_the_window(
    tmp_path: Path,
) -> None:
    """Burn-in only fills the history window; anything longer throws steps away."""
    stacked = _arm(tmp_path / "stacked", **{"--stacked-burn-in": "3"})

    assert stacked.training.actors[0].config.burn_in == 3
    assert stacked.resolved["burn_in"] == 3


def test_a_run_id_names_its_backbone_and_collides_with_nothing() -> None:
    """Two runs started in the same second are still two runs."""
    first, second = new_run_id(train.BACKBONE), new_run_id(train.BACKBONE)

    assert first.startswith(f"{train.BACKBONE}-") and first != second


def test_every_artifact_is_bound_to_the_code_that_produced_it() -> None:
    """A revision that cannot be resolved says so rather than being omitted."""
    revision = source_revision()

    assert revision and " " not in revision


def test_the_floors_the_curve_is_read_against_travel_with_the_run() -> None:
    """A comparison opened months later must be self-contained."""
    params = tracked_params({"seed": 0})

    assert params["seed"] == 0
    assert params["reference_scripted"] == REFERENCE_FINAL_WAVES["scripted"]
    assert params["reference_source"] == REFERENCE_FINAL_WAVES["source"]
