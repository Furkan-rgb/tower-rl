#!/usr/bin/env python3
"""Run N episodes of one policy against the private instrumented clone.

Private device runner for the instrumented-training profile. It wires the real
bridge adapter to the environment and evaluator and writes its report outside the
repository. It never touches the canonical evaluation AVD, and it never taps:
since the round boundary moved into the bridge, nothing in this path reads a
pixel or touches the screen.

    ./scripts/run_episodes.py --episodes 50

`--policy` names the arm: one of the non-learned floors, or `checkpoint:<path>`
for a checkpoint a training run left behind, which is rebuilt into the backbone
that wrote it and played greedily. Every record says which it was.

    ./scripts/run_episodes.py --episodes 50 \\
        --policy checkpoint:state/runs/.../checkpoint-gs0100000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# One torch thread per actor process. A fleet runs N of these at once against N
# emulators on one host, and torch sizes its intra-op pool from the core count,
# so N processes each claiming every core thrash instead of collecting. Acting
# is a single-sample forward pass through a small network and has nothing to
# gain from a pool anyway. Set before torch is first imported, because OpenMP
# and MKL read them when their runtimes initialise; `tests/conftest.py` does the
# same thing for the same reason.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

from tower_rl.environment.features import (  # noqa: E402
    ROW_COUNT,
    ROW_WIDTH,
    StateFeatures,
)
from tower_rl.environment.project_state import state_directory  # noqa: E402
from tower_rl.environment.run_environment import (  # noqa: E402
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import RunStateBuilder  # noqa: E402
from tower_rl.learning.actor import ActorConfig  # noqa: E402
from tower_rl.learning.checkpoint import CheckpointError, identity_hash  # noqa: E402
from tower_rl.learning.evaluator import EvaluationReport, evaluate, to_record  # noqa: E402
from tower_rl.learning.policies import (  # noqa: E402
    CheapestFirstPolicy,
    Policy,
    RandomPolicy,
    WaitOnlyPolicy,
    checkpoint_policy,
)
from tower_rl.simulation.bridge import bridge_build_directory, compatibility  # noqa: E402
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    InstrumentedBridgeClient,
    UpgradeSlotLabel,
)
from tower_rl.simulation.instrumented_run_adapter import (  # noqa: E402
    InstrumentedRunAdapter,
)

torch.set_num_threads(1)

POLICIES = {
    "scripted": CheapestFirstPolicy,
    "random": RandomPolicy,
    "wait": WaitOnlyPolicy,
}

#: How a checkpoint is named as an arm, beside the names above.
CHECKPOINT_SELECTOR = "checkpoint:"

#: Steps in one recorded observation window. The stored sequence length the run
#: trained on, so a window offers the diagnostic the same stacked history the
#: learner and the actor both saw.
OBSERVATION_WINDOW = 80


@dataclass
class RecordingPolicy:
    """Plays exactly what it wraps, and keeps the states it was asked about.

    No observation batch exists anywhere in this project: an actor record holds
    episode summaries, a checkpoint holds `replay_provenance` rather than the
    buffer, and `DecisionView` deliberately carries no observation. The offline
    plasticity diagnostic (`diagnose_plasticity.py`) needs real states to push
    through a network, so one short session writes them here.

    A wrapper rather than a hook inside the learner: `Policy` is a protocol, so
    this sees exactly the `StateFeatures` the wrapped policy acted on, in
    decision order, without `learning` or `environment` growing a seam for a
    diagnostic. The only thing it changes downstream is `EvaluationReport.policy`,
    which names the acting class and carries no arm identity anyway - the arm
    travels in `policy_identity`, which is built from the selector.
    """

    policy: Policy
    path: Path
    window: int = OBSERVATION_WINDOW
    #: One list per episode, in the order the episodes ran.
    episodes: list[list[StateFeatures]] = field(default_factory=list)

    def initial_state(self) -> Any:
        # An episode start is where the stacked history returns to zeros, so a
        # window must never straddle two episodes. This is also the only
        # boundary this wrapper sees, so it is where the batch so far is
        # flushed: a session cut short still leaves the episodes it finished.
        self.write()
        self.episodes.append([])
        return self.policy.initial_state()

    def act(self, features: StateFeatures, state: Any, *, epsilon: float) -> tuple[int, Any]:
        self.episodes[-1].append(features)
        return self.policy.act(features, state, epsilon=epsilon)

    def windows(self) -> list[list[StateFeatures]]:
        """Each episode cut into whole windows from its start; the tail is dropped.

        Non-overlapping, so no state is counted twice in an expectation taken
        over the batch, and left-aligned, so every window begins where a real
        history window began rather than part way through one.
        """
        return [
            episode[start : start + self.window]
            for episode in self.episodes
            for start in range(0, len(episode) - self.window + 1, self.window)
        ]

    def write(self) -> int:
        """Write the batch `diagnose_plasticity.py` reads; return its observations."""
        windows = self.windows()
        if not windows:
            return 0
        rows = torch.tensor(
            [[state.rows for state in window] for window in windows], dtype=torch.float32
        )
        torch.save(
            {
                "scalars": torch.tensor(
                    [[state.scalars for state in window] for window in windows],
                    dtype=torch.float32,
                ),
                "rows": rows.view(len(windows), self.window, ROW_COUNT, ROW_WIDTH),
                "mask": torch.tensor(
                    [[state.mask for state in window] for window in windows], dtype=torch.bool
                ),
            },
            self.path,
        )
        return len(windows) * self.window


def policy_from(selector: str) -> tuple[Policy, dict[str, object]]:
    """The arm this run plays, and the identity every record of it carries.

    A selector is one of the non-learned floors by name, or `checkpoint:<path>`.
    The two are the same kind of thing to everything downstream - the actor, the
    evaluator, the per-episode records - which is the point: a checkpoint is
    measured by exactly the protocol its floors are.

    The identity travels with the record rather than being inferred from the
    directory a file happens to sit in. `run_actors.py` writes one file per
    actor and a selection reads many of those directories at once, so a record
    that could not name its own arm could only be attributed by convention.
    """
    if selector in POLICIES:
        return POLICIES[selector](), {"name": selector}
    if not selector.startswith(CHECKPOINT_SELECTOR):
        raise SystemExit(
            f"unknown policy {selector!r}; choose from {sorted(POLICIES)} "
            f"or {CHECKPOINT_SELECTOR}<path>"
        )
    path = Path(selector[len(CHECKPOINT_SELECTOR) :]).expanduser()
    try:
        policy, identity = checkpoint_policy(path)
    except (CheckpointError, ValueError) as failure:
        raise SystemExit(f"cannot play {path} as an arm: {failure}") from failure
    return policy, {
        # Named for the file, which is named for the game time behind it, so an
        # actor id and a report line say which checkpoint of the run this is.
        "name": path.stem,
        "checkpoint_path": str(path.resolve()),
        # The path is where the file is today; this is what it holds.
        "checkpoint_identity": identity_hash(identity),
        "run_id": identity.run_id,
    }


def actor_record(
    report: EvaluationReport,
    identity: Mapping[str, object],
    *,
    frame_game_ms: float,
    max_quiet_game_ms: int,
    decision_cadence: DecisionCadence,
    upgrade_availability: UpgradeAvailability,
    wall_seconds: float,
    labels: Sequence[UpgradeSlotLabel] = (),
) -> dict[str, Any]:
    """One actor's durable record: the episodes, and which arm produced them.

    `run_actors.py` writes one of these per instance and a later selection reads
    whole directories of them, so the arm has to be inside the record. The
    evaluator's own `policy` field names the class that acted, which is the same
    class for every checkpoint of every run.
    """
    record = to_record(report)
    record["policy_identity"] = dict(identity)
    record["frame_game_ms"] = frame_game_ms
    record["max_quiet_game_ms"] = max_quiet_game_ms
    # Which protocol these episodes were played under. Decisions mean different
    # things under the two, so a record that could not say which cadence
    # produced it could not be compared with anything (ADR 0009).
    record["decision_cadence"] = str(decision_cadence)
    # Which upgrade rows these episodes could buy from. A record collected on
    # the image's six rows and one collected on every real row are measurements
    # of two different decision problems (ADR 0011).
    record["upgrade_availability"] = str(upgrade_availability)
    record["wall_seconds"] = round(wall_seconds, 1)
    # What the game calls each slot the actions address, so the human reading
    # this record afterwards can tell what `attack:3` was. Never an input: the
    # policy addresses a slot by index, and a renamed row must not change what a
    # checkpoint means.
    record["upgrade_rows"] = [
        {"family": label.family, "index": label.index, "name": label.name,
         "description": label.description}
        for label in labels
        if label.name
    ]
    record["episodes_per_hour"] = (
        round(report.valid_episodes / wall_seconds * 3600, 1) if wall_seconds > 0 else 0.0
    )
    return record


def add_cadence_arguments(parser: argparse.ArgumentParser) -> None:
    """The cadence is game time throughout; the only wall clock is a hang deadline.

    There is no speed argument. The game's own multiplier is pinned at 1x inside
    the adapter, and speed comes from the bridge stepping frames, so a speed knob
    here could only reintroduce the coarsening it was removed for (M1B-E012).
    """
    parser.add_argument(
        "--frame-game-ms",
        type=float,
        default=1000.0 / 60.0,
        help="game time one rendered frame is worth; the floor on decision granularity",
    )
    parser.add_argument(
        "--max-quiet-game-ms",
        type=int,
        default=2000,
        help="game time one advance may spend before returning a decision anyway",
    )
    parser.add_argument(
        "--max-episode-wall-seconds",
        type=float,
        default=600.0,
        help="hang deadline in wall seconds; wall time is not bounded by game time",
    )
    parser.add_argument(
        "--decision-cadence",
        choices=[cadence.value for cadence in DecisionCadence],
        default=DecisionCadence.CHOICE_POINTS.value,
        help="which cadence stops the policy is asked about: choice-points is "
        "the contract, every-slice reproduces run 1's protocol (ADR 0009)",
    )


def add_upgrade_availability_argument(parser: argparse.ArgumentParser) -> None:
    """Which upgrade rows the run is played with (ADR 0011).

    Separate from the cadence arguments because it is a separate thing: the
    cadence says when the policy is asked, this says what it may buy. `image` is
    the default and is what every baseline so far was measured under.
    """
    parser.add_argument(
        "--upgrade-availability",
        choices=[availability.value for availability in UpgradeAvailability],
        default=UpgradeAvailability.IMAGE.value,
        help="which upgrade rows are purchasable: image is what the profile "
        "image offers, all reopens every real row at each round start (ADR 0011)",
    )


def upgrade_availability_from(arguments: argparse.Namespace) -> UpgradeAvailability:
    """The availability this invocation collects or plays under."""
    return UpgradeAvailability(arguments.upgrade_availability)


def cadence_from(arguments: argparse.Namespace) -> CadenceConfig:
    return CadenceConfig(
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
        max_episode_wall_seconds=arguments.max_episode_wall_seconds,
    )


def decision_cadence_from(arguments: argparse.Namespace) -> DecisionCadence:
    """The named cadence policy this invocation collects or plays under."""
    return DecisionCadence(arguments.decision_cadence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--policy",
        default="scripted",
        help=(
            f"one of {sorted(POLICIES)}, or {CHECKPOINT_SELECTOR}<path> to play a "
            "checkpoint a training run left behind"
        ),
    )
    parser.add_argument("--serial", default="emulator-5556")
    parser.add_argument("--port", type=int, default=47652)
    add_cadence_arguments(parser)
    add_upgrade_availability_argument(parser)
    parser.add_argument(
        "--output", type=Path, default=state_directory() / "records" / "episodes.json"
    )
    parser.add_argument(
        "--record-observations",
        type=Path,
        default=None,
        help="write the states this session acted on, as the observation batch "
        "`diagnose_plasticity.py` reads; keep it outside the repository",
    )
    arguments = parser.parse_args()

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to run against the canonical evaluation AVD")

    # Before the device is touched: a checkpoint that cannot be rebuilt should
    # fail now, not after an emulator has been brought up for it.
    policy, identity = policy_from(arguments.policy)
    recorder: RecordingPolicy | None = None
    if arguments.record_observations is not None:
        path = arguments.record_observations.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        recorder = RecordingPolicy(policy, path)
        policy = recorder

    expected = compatibility(bridge_build_directory())

    client = InstrumentedBridgeClient(
        "127.0.0.1",
        arguments.port,
        expected_compatibility=expected,
        connect_timeout=5.0,
        read_timeout=120.0,
        heartbeat_timeout=60.0,
    )
    handshake = client.connect()
    print(
        f"bridge {handshake.bridge_version} profile {handshake.compatibility.profile_id} "
        f"speed {handshake.game_speed}",
        flush=True,
    )

    adapter = InstrumentedRunAdapter(client=client)
    # Before the first round: a command of the adapter's own initiative belongs
    # to the episode boundary, and these are constant for the build.
    labels = adapter.slot_labels()
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=cadence_from(arguments),
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
    )

    started = time.monotonic()
    try:
        report = evaluate(
            environment,
            policy,
            episodes=arguments.episodes,
            profile_id=expected.profile_id,
            actor_config=ActorConfig(actor_id=f"{arguments.serial}:{identity['name']}"),
        )
    finally:
        adapter.release()
        client.close()

    if recorder is not None:
        print(f"{recorder.write()} observations written to {recorder.path}", flush=True)

    record = actor_record(
        report,
        identity,
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
        wall_seconds=time.monotonic() - started,
        labels=labels,
    )
    arguments.output.write_text(json.dumps(record, indent=2))
    print(report.summary_line(), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
