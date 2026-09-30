"""Pipeline throughput under one fixed policy: where a decision's wall time goes.

Training decisions per hour follow the policy's game time per decision, which
ranged 1.5-4.3 s across arms, so they cannot say whether a change to the
pipeline made it faster (ADR 0016, "Throughput and comparability"). This reads
the per-episode records a fixed scripted policy left and reports what can:

- frames per second, over the bridge's own advance time and over the episode;
- a decision's wall time cut into the advance (the bridge's `wall_micros`), the
  transport (the host's advance round trip minus that), purchases, the policy,
  and the rest of the host's time;
- the behaviour fingerprint - final wave, decisions per wave and the
  round-clock ratio - which a change that claims to be robustness-only must
  leave unchanged.

Only valid episodes are measured: an invalid one ends wherever its failure
fell, and its timing belongs to no protocol. They are counted instead. Every
interval is a stratified bootstrap over episodes with the actor as the
stratum, so a comparison between two reports reads interval against interval.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tower_rl.experiment.comparison import stratified_bootstrap

Episode = Mapping[str, Any]

#: Per-episode readings, each a function of one valid episode record that
#: took at least one decision. Milliseconds are per decision.
EPISODE_READINGS: dict[str, Callable[[Episode], float]] = {
    "decision_ms": lambda e: 1000 * e["elapsed_wall_seconds"] / e["decisions"],
    "advance_ms": lambda e: 1000 * e["advance_wall_seconds"] / e["decisions"],
    "transport_ms": lambda e: (
        1000 * (e["advance_round_trip_seconds"] - e["advance_wall_seconds"]) / e["decisions"]
    ),
    # The same per advance command: a property of the pipeline rather than of
    # how many advances the policy's decisions take.
    "transport_ms_per_advance": lambda e: (
        1000 * (e["advance_round_trip_seconds"] - e["advance_wall_seconds"]) / e["advances"]
    ),
    "transport_cpu_ms": lambda e: 1000 * e["advance_round_trip_cpu_seconds"] / e["decisions"],
    "purchase_ms": lambda e: 1000 * e["purchase_round_trip_seconds"] / e["decisions"],
    # Everything in the decision that is neither a command's round trip:
    # the policy, observation decoding, and the environment's bookkeeping.
    "host_ms": lambda e: (
        1000
        * (
            e["elapsed_wall_seconds"]
            - e["advance_round_trip_seconds"]
            - e["purchase_round_trip_seconds"]
        )
        / e["decisions"]
    ),
    "frames_per_advance_second": lambda e: e["frames"] / e["advance_wall_seconds"],
    "frames_per_episode_second": lambda e: e["frames"] / e["elapsed_wall_seconds"],
    "frames_per_advance": lambda e: e["frames"] / e["advances"],
    "advances_per_decision": lambda e: e["advances"] / e["decisions"],
    "purchases_per_decision": lambda e: e["purchases"] / e["decisions"],
    # The fingerprint.
    "final_wave": lambda e: float(e["final_wave"]),
    "decisions_per_wave": lambda e: e["decisions"] / e["final_wave"],
    "round_clock_ratio": lambda e: e["budgeted_game_ms"] / e["round_ms"],
    "game_seconds_per_decision": lambda e: e["round_ms"] / 1000 / e["decisions"],
}


def measurable(episode: Episode) -> bool:
    """Whether an episode carries every reading: valid, and with a decision in it."""
    return bool(episode["valid"]) and episode["decisions"] > 0 and episode["advances"] > 0


def interval(strata: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """The mean over every stratum's values, with its 95% bootstrap interval."""
    mean, low, high = stratified_bootstrap(strata, statistics.fmean, seed=0)
    return {"mean": round(mean, 4), "low": round(low, 4), "high": round(high, 4)}


def purchase_round_trip_ms(episodes: Sequence[Episode]) -> float | None:
    """Mean wall time of one purchase command, or None when nothing was bought."""
    purchases = sum(int(episode["purchases"]) for episode in episodes)
    if purchases == 0:
        return None
    seconds = sum(float(episode["purchase_round_trip_seconds"]) for episode in episodes)
    return round(1000 * seconds / purchases, 3)


def actor_throughput(record: Mapping[str, Any]) -> dict[str, Any]:
    """One actor's readings as plain means, and what it spent between episodes."""
    attempted = record["episodes"]
    episodes = [episode for episode in attempted if measurable(episode)]
    decisions = sum(episode["decisions"] for episode in episodes)
    in_episodes = sum(episode["elapsed_wall_seconds"] for episode in attempted)
    readings = {
        name: round(statistics.fmean(reading(episode) for episode in episodes), 4)
        for name, reading in (EPISODE_READINGS.items() if episodes else ())
    }
    policy_seconds = float(record.get("total_policy_seconds", 0.0))
    return {
        "valid_episodes": int(record["valid_episodes"]),
        "invalid_episodes": int(record["invalid_episodes"]),
        "measured_episodes": len(episodes),
        "decisions": decisions,
        **readings,
        # Over every attempted episode's decisions, which is what the policy
        # was timed over.
        "policy_ms": round(
            1000 * policy_seconds / max(1, sum(e["decisions"] for e in attempted)), 4
        ),
        "purchase_round_trip_ms": purchase_round_trip_ms(episodes),
        # Reset, retirement and round start: the actor's wall time outside
        # every episode, per attempted episode.
        "boundary_seconds_per_episode": (
            round((float(record["wall_seconds"]) - in_episodes) / len(attempted), 3)
            if attempted
            else None
        ),
    }


def throughput(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The fleet's throughput report from its actors' records, keyed by actor.

    `records` holds each actor's `run_episodes.py` record under its serial.
    """
    strata = {
        actor: [episode for episode in record["episodes"] if measurable(episode)]
        for actor, record in records.items()
    }
    strata = {actor: episodes for actor, episodes in strata.items() if episodes}
    actors = {actor: actor_throughput(record) for actor, record in records.items()}
    measured = [episode for episodes in strata.values() for episode in episodes]
    invalid_by_reason: Counter[str] = Counter()
    for record in records.values():
        invalid_by_reason.update(record.get("invalid_by_reason", {}))
    report: dict[str, Any] = {
        "validity": {
            "valid_episodes": sum(int(r["valid_episodes"]) for r in records.values()),
            "invalid_episodes": sum(int(r["invalid_episodes"]) for r in records.values()),
            "invalid_by_reason": dict(invalid_by_reason),
            "measured_episodes": len(measured),
        },
        "actors": actors,
    }
    if not measured:
        return report
    per_decision = {
        name: interval(
            {actor: [reading(episode) for episode in group] for actor, group in strata.items()}
        )
        for name, reading in EPISODE_READINGS.items()
    }
    policy_ms = statistics.fmean(actor["policy_ms"] for actor in actors.values())
    report["readings"] = per_decision
    # The split the benchmark exists for, in ms per decision: the parts sum to
    # `decision_ms` up to the bootstrap's rounding.
    report["decision_split_ms"] = {
        "advance": per_decision["advance_ms"]["mean"],
        "transport": per_decision["transport_ms"]["mean"],
        "purchase": per_decision["purchase_ms"]["mean"],
        "policy": round(policy_ms, 4),
        "other_host": round(per_decision["host_ms"]["mean"] - policy_ms, 4),
    }
    # Actors run at once, so their in-episode rates add.
    report["fleet_frames_per_episode_second"] = round(
        sum(actor.get("frames_per_episode_second", 0.0) for actor in actors.values()), 2
    )
    report["purchase_round_trip_ms"] = purchase_round_trip_ms(measured)
    boundaries = [
        actor["boundary_seconds_per_episode"]
        for actor in actors.values()
        if actor["boundary_seconds_per_episode"] is not None
    ]
    report["boundary_seconds_per_episode"] = (
        round(statistics.fmean(boundaries), 3) if boundaries else None
    )
    waves = Counter(int(episode["final_wave"]) for episode in measured)
    report["fingerprint"] = {
        "final_wave": per_decision["final_wave"],
        "final_wave_stdev": (
            round(statistics.stdev(e["final_wave"] for e in measured), 4)
            if len(measured) > 1
            else None
        ),
        "final_wave_counts": {str(wave): waves[wave] for wave in sorted(waves)},
        "decisions_per_wave": per_decision["decisions_per_wave"],
        "round_clock_ratio": per_decision["round_clock_ratio"],
        "game_seconds_per_decision": per_decision["game_seconds_per_decision"],
    }
    return report
