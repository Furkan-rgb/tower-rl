"""What the step and item replays share: their errors, metadata and counters.

The buffers themselves are `dreamer_replay.py` and `r2d2_replay.py`. Replay holds
encoded features and never game state, so it knows nothing about The Tower: what
it does know is which schema and which profile an episode came from, and it
refuses to mix them (`SequenceMetadata.compatibility_key`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: R2D2's prioritized sequence replay (Kapturowski et al. 2019, ICLR; the values
#: of DeepMind's Acme reference `r2d2/config.py`: `priority_exponent`,
#: `importance_sampling_exponent`, `max_priority_weight`). Fixed, not options:
#: R2D2 always samples by them, and DreamerV3 always samples uniformly, so
#: neither can be silently run without the replay its recipe names.
#:
#: Priority exponent alpha: a sequence is sampled with probability p ** alpha
#: over the buffer's total.
R2D2_PRIORITY_EXPONENT = 0.9
#: Importance-sampling exponent beta, held fixed as R2D2 and Ape-X hold it
#: rather than annealed to 1 as Schaul 2016 does: an anneal over the budget
#: moves the weighted loss on its own, which is what made the first run's loss
#: unreadable.
R2D2_IMPORTANCE_SAMPLING_EXPONENT = 0.6
#: Priority mix eta: a sequence's priority is eta * max |TD| + (1 - eta) *
#: mean |TD| over its steps, so one surprising step matters without a single
#: outlier dominating the whole sequence.
R2D2_PRIORITY_MIX = 0.9

#: The dump's one metadata file, beside its arrays.
REPLAY_DUMP_METADATA = "replay.json"


class ReplayRejected(ValueError):
    """A payload was refused; it is counted, never silently coerced."""


class ReplayDumpError(RuntimeError):
    """A saved buffer could not be written, read, or is not this buffer's to load."""


@dataclass(frozen=True)
class SequenceMetadata:
    """Everything needed to decide whether a sequence may be learned from."""

    episode_id: str
    actor_id: str
    profile_id: str
    observation_schema: str
    action_schema: str
    reward_schema: str
    model_version: int
    epsilon: float
    game_speed: float

    def compatibility_key(self) -> tuple[str, str, str, str]:
        return (
            self.profile_id,
            self.observation_schema,
            self.action_schema,
            self.reward_schema,
        )


@dataclass
class ReplayStats:
    added: int = 0
    rejected: int = 0
    evicted: int = 0
    sampled: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1


def read_replay_metadata(directory: Path) -> dict[str, Any]:
    """The metadata of a saved buffer, read without touching its arrays."""
    try:
        metadata: dict[str, Any] = json.loads((directory / REPLAY_DUMP_METADATA).read_text())
    except (OSError, ValueError) as error:
        raise ReplayDumpError(f"cannot read replay dump {directory}: {error}") from error
    return metadata
