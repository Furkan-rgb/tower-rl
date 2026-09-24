"""One arm's evaluation, as the fleet runner leaves it on disk.

`run_actors.py` writes one JSON record per instance into a directory, each one
holding that actor's episodes and the arm that played them. Reading a directory
back is the first step of both things done afterwards - choosing among the
checkpoints of a run, and reporting the chosen one against the floors - so it
happens here rather than twice in two scripts.

The actor is the stratum. Episodes collected on one emulator instance are not
exchangeable with episodes collected on another: instances differ in host
contention and in whatever state their overlay accumulated, and an interval that
resampled the pool flat would let one instance crowd out another. `seed` will be
the stratum when several training seeds are compared; the shape here is the same.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The per-episode quantities an arm is summarised on. `final_wave` is the
#: outcome; `decisions` says how much of an episode the policy actually got to
#: decide, which is what separates an arm that survived from one that waited.
STATISTICS: tuple[str, ...] = ("final_wave", "decisions")


@dataclass(frozen=True)
class ArmEvaluation:
    """What one arm did, kept by the actor that produced each episode."""

    name: str
    directory: Path
    #: Whatever the actors recorded about the policy they played: the arm name,
    #: and for a checkpoint its path and the hash of its identity. `None` for a
    #: record written before arms carried their own identity.
    policy_identity: dict[str, Any] | None
    #: Every valid episode's record, pooled across actors, for the per-wave
    #: comparison that reads them whole.
    episodes: tuple[dict[str, Any], ...]
    #: Per statistic, the values of each actor's valid episodes, keyed by actor.
    strata: Mapping[str, Mapping[str, tuple[float, ...]]]

    @property
    def valid_episodes(self) -> int:
        return len(self.episodes)

    def values(self, statistic: str) -> tuple[float, ...]:
        """One statistic pooled over every actor, in actor order."""
        by_actor = self.strata[statistic]
        return tuple(value for actor in sorted(by_actor) for value in by_actor[actor])

    @property
    def checkpoint(self) -> str | None:
        """The checkpoint file this arm played, if it was a checkpoint at all."""
        if not self.policy_identity:
            return None
        path = self.policy_identity.get("checkpoint_path")
        return None if path is None else str(path)


def read_arm_evaluation(directory: Path, *, name: str | None = None) -> ArmEvaluation:
    """Read every actor record in one directory as a single arm's evaluation.

    A directory holds one arm. Records that disagree about which arm they played
    are refused by name rather than pooled: an accidental mix of two checkpoints
    would report an interval over a policy that never existed.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        # A fleet aggregate written beside the actor records is not one of them.
        if not isinstance(payload, dict) or "episodes" not in payload:
            continue
        records[path.stem] = payload
    if not records:
        raise ValueError(f"{directory} holds no actor records")

    played = {
        json.dumps(record.get("policy_identity"), sort_keys=True) for record in records.values()
    }
    if len(played) > 1:
        raise ValueError(
            f"{directory} mixes arms: {sorted(played)}; one directory holds one arm"
        )
    identity = next(iter(records.values())).get("policy_identity")

    strata: dict[str, dict[str, tuple[float, ...]]] = {statistic: {} for statistic in STATISTICS}
    episodes: list[dict[str, Any]] = []
    for actor in sorted(records):
        valid = [episode for episode in records[actor]["episodes"] if episode.get("valid", True)]
        if not valid:
            # An actor that scored nothing is not a stratum with no values; it
            # is an actor that contributed no episodes, and resampling an empty
            # stratum is undefined.
            continue
        episodes.extend(valid)
        for statistic in STATISTICS:
            strata[statistic][actor] = tuple(float(episode[statistic]) for episode in valid)
    if not episodes:
        raise ValueError(f"{directory} holds no valid episode; the arm cannot be scored")

    return ArmEvaluation(
        name=name or directory.name,
        directory=directory,
        policy_identity=identity,
        episodes=tuple(episodes),
        strata=strata,
    )


def pooled_report(evaluation: ArmEvaluation) -> dict[str, Any]:
    """The arm's episodes in the one-arm report shape `wave_statistics` reads.

    `episode_records` takes either report shape and this is the flat one, so the
    per-wave comparison needs no knowledge that the episodes came from several
    actors of a fleet.
    """
    return {
        "policy_identity": evaluation.policy_identity,
        "valid_episodes": evaluation.valid_episodes,
        "episodes": list(evaluation.episodes),
    }


def statistic_line(name: str, point: float, low: float, high: float, episodes: int) -> str:
    """One arm's IQM with its interval, formatted so the spread is unavoidable."""
    return f"{name:<28} IQM {point:6.2f}  [{low:6.2f}, {high:6.2f}]  n={episodes}"


def overlaps(first: Sequence[float], second: Sequence[float]) -> bool:
    """Whether two intervals share any value, given as (low, high) pairs."""
    return first[0] <= second[1] and second[0] <= first[1]
