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
        --policy checkpoint:state/runs/<run name>/checkpoints/checkpoint-d0015000.pt

A checkpoint's record is filed with the run that wrote it by default, in
`<run folder>/evaluations/<name>/<serial>.json`; `--evaluation-name` names it
(default: the checkpoint and the UTC time) and `--output` puts it anywhere else.
A floor's record goes to `state/records/episodes.json` unless `--output` says
otherwise.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
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

from tower_rl.console_timestamp import timestamped_print as print  # noqa: E402
from tower_rl.environment.project_state import state_directory  # noqa: E402
from tower_rl.environment.run_environment import (  # noqa: E402
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_state import RunStateBuilder  # noqa: E402
from tower_rl.environment.upgrade_setup import (  # noqa: E402
    UpgradeSetup,
    UpgradeSetupReference,
    UpgradeSetupRefused,
)
from tower_rl.environment.workshop import WORKSHOP_OFF, workshop_rows  # noqa: E402
from tower_rl.experiment.run_folder import evaluation_directory, utc_stamp  # noqa: E402
from tower_rl.learning.actor import ActorConfig  # noqa: E402
from tower_rl.learning.checkpoint import CheckpointError, identity_hash  # noqa: E402
from tower_rl.learning.evaluator import EvaluationReport, evaluate, to_record  # noqa: E402
from tower_rl.learning.policies import (  # noqa: E402
    LATE_GAME_BUILDS,
    BindsRowNames,
    BlenderBuildPolicy,
    CheapestFirstPolicy,
    LateGameBuildPolicy,
    Policy,
    RandomPolicy,
    TurtlePolicy,
    WaitOnlyPolicy,
    checkpoint_policy,
)
from tower_rl.simulation.bridge import bridge_build_directory, compatibility  # noqa: E402
from tower_rl.simulation.instrumented_bridge import (  # noqa: E402
    MAX_WORKSHOP_LEVEL,
    BridgeCompatibility,
    InstrumentedBridgeClient,
    UpgradeSlotLabel,
)
from tower_rl.simulation.instrumented_run_adapter import (  # noqa: E402
    InstrumentedRunAdapter,
)

torch.set_num_threads(1)

POLICIES: dict[str, Callable[[], Policy]] = {
    "scripted": CheapestFirstPolicy,
    "random": RandomPolicy,
    "wait": WaitOnlyPolicy,
    "turtle": TurtlePolicy,
    **{
        f"build-{build}": functools.partial(LateGameBuildPolicy, build)
        for build in LATE_GAME_BUILDS
    },
    "build-blender": BlenderBuildPolicy,
}

#: How a checkpoint is named as an arm, beside the names above.
CHECKPOINT_SELECTOR = "checkpoint:"

def policy_from(
    selector: str,
    *,
    decision_cadence: DecisionCadence,
    upgrade_availability: UpgradeAvailability,
    workshop_level: int,
    sampling_seed: str | None = None,
) -> tuple[Policy, dict[str, object]]:
    """The arm this run plays, and the identity every record of it carries.

    A selector is one of the non-learned floors by name, or `checkpoint:<path>`.
    The two are the same kind of thing to everything downstream - the actor, the
    evaluator, the per-episode records - which is the point: a checkpoint is
    measured by exactly the protocol its floors are.

    Which is also why the protocol this session will run is passed in: a floor
    plays whatever it is given, but a checkpoint learned one cadence and one set
    of purchasable rows, and playing it under the others measures something the
    run it came from never posed. `checkpoint_policy` refuses that by name.

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
        policy, identity = checkpoint_policy(
            path,
            decision_cadence=str(decision_cadence),
            upgrade_availability=str(upgrade_availability),
            workshop_level=workshop_level,
            # A checkpoint that samples its policy draws from a stream of this
            # instance's own, not one every instance of a fleet shares.
            sampling_seed=sampling_seed,
        )
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
        # The upgrade setup the checkpoint's run was played on, which this
        # session's first episode must reproduce; None for a checkpoint written
        # before the setup was recorded.
        "upgrade_setup_digest": identity.upgrade_setup_digest,
    }


def bind_row_names(policy: Policy, labels: Sequence[UpgradeSlotLabel]) -> None:
    """Give a policy that buys rows by name the game's labels; any other is left alone.

    `turtle` and the `build-*` policies address rows by the game's own names,
    resolved from these labels, and a name the game does not report stops the
    session here, loudly.
    """
    if isinstance(policy, BindsRowNames):
        policy.bind_row_names(labels)


def add_evaluation_name_argument(parser: argparse.ArgumentParser) -> None:
    """`--evaluation-name`: the directory a checkpoint's evaluation is filed under."""
    parser.add_argument(
        "--evaluation-name",
        default=None,
        help=(
            "with a checkpoint policy and no output given, the evaluation is "
            "written to <its run folder>/evaluations/<this name>/; default "
            "<checkpoint>-<UTC time>"
        ),
    )


def checkpoint_evaluation_directory(selector: str, name: str | None) -> Path | None:
    """Where an evaluation of a checkpoint arm is filed by default; None for a floor.

    The run is the folder the checkpoint sits in (`tower_rl.experiment.run_folder`).
    The default name carries the time, so two evaluations of one checkpoint
    never share a directory: a directory is read back as one arm's records, and
    two sets pooled into it would be read as one.
    """
    if not selector.startswith(CHECKPOINT_SELECTOR):
        return None
    checkpoint = Path(selector[len(CHECKPOINT_SELECTOR) :]).expanduser()
    return evaluation_directory(checkpoint, name or f"{checkpoint.stem}-{utc_stamp()}")


def refuse_an_unused_evaluation_name(arguments: argparse.Namespace, *outputs: str) -> None:
    """`--evaluation-name` names the default directory only; given otherwise, it is refused."""
    if arguments.evaluation_name is None:
        return
    if not arguments.policy.startswith(CHECKPOINT_SELECTOR):
        raise SystemExit("--evaluation-name names an evaluation of a checkpoint policy")
    given = [output for output in outputs if getattr(arguments, output) is not None]
    if given:
        raise SystemExit(
            f"--evaluation-name names the default output directory, which "
            f"--{given[0].replace('_', '-')} replaces; give one or the other"
        )


def settle_output(arguments: argparse.Namespace) -> None:
    """Fill in where the record goes, when `--output` was not given.

    `<run folder>/evaluations/<name>/<serial>.json` for a checkpoint, the file
    `run_actors.py` would have named for this instance; `state/records/` for a
    floor.
    """
    refuse_an_unused_evaluation_name(arguments, "output")
    if arguments.output is not None:
        return
    directory = checkpoint_evaluation_directory(arguments.policy, arguments.evaluation_name)
    arguments.output = (
        state_directory() / "records" / "episodes.json"
        if directory is None
        else directory / f"{arguments.serial}.json"
    )


def actor_record(
    report: EvaluationReport,
    identity: Mapping[str, object],
    *,
    frame_game_ms: float,
    max_quiet_game_ms: int,
    decision_cadence: DecisionCadence,
    upgrade_availability: UpgradeAvailability,
    workshop_level: int,
    wall_seconds: float,
    labels: Sequence[UpgradeSlotLabel] = (),
    upgrade_setup: UpgradeSetup | None = None,
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
    # And on which Workshop setup: baseline v1's bare account at 0, the runway
    # profile's rows at this level otherwise (ADR 0012).
    record["workshop_level"] = workshop_level
    record["workshop_rows"] = list(workshop_rows(workshop_level))
    # What the game actually held for these episodes, as it read it back, once
    # for the file: each episode below carries only the digest, and one that
    # drifted from this setup carries its own.
    record["upgrade_setup"] = None if upgrade_setup is None else upgrade_setup.to_record()
    record["upgrade_setup_digest"] = None if upgrade_setup is None else upgrade_setup.digest
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
    """The cadence is game time throughout.

    There is no speed argument. The game's own multiplier is pinned at 1x inside
    the adapter, and speed comes from the bridge stepping frames, so a speed knob
    here could only reintroduce the coarsening it was removed for (M1B-E012).

    There is also no episode-length argument: an episode ends only on the game
    dying or on the environment's own liveness guard finding the game clock has
    stopped advancing (`STALLED`, `#88`), never on a fixed decision count or wall
    time - so nothing here needs to configure one.
    """
    parser.add_argument(
        "--frame-game-ms",
        type=float,
        # 100 ms, adopted in M1B-E018 and shown equivalent and 2.6-3.0x faster under M2 in M2-S001.
        default=100.0,
        help="game time one rendered frame is worth; the floor on decision granularity",
    )
    parser.add_argument(
        "--max-quiet-game-ms",
        type=int,
        default=2000,
        help="game time one advance may spend before returning a decision anyway",
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


def _workshop_level(text: str) -> int:
    level = int(text)
    if not WORKSHOP_OFF <= level <= MAX_WORKSHOP_LEVEL:
        raise argparse.ArgumentTypeError(
            f"a Workshop level is {WORKSHOP_OFF} to {MAX_WORKSHOP_LEVEL}, the bridge's own bound"
        )
    return level


def add_workshop_level_argument(parser: argparse.ArgumentParser) -> None:
    """The Workshop runway profile's level (ADR 0012).

    Beside the availability because it is the same kind of choice: fixed by the
    operator for the whole run, applied by the environment, never chosen by the
    policy. 0 is the default and is baseline v1, the account as the image holds
    it; nothing is written.
    """
    parser.add_argument(
        "--workshop-level",
        type=_workshop_level,
        default=WORKSHOP_OFF,
        help="set the Workshop runway profile's rows to this level before every "
        "round, on the disposable instance only; 0 (the default) writes nothing "
        "(ADR 0012)",
    )


def workshop_level_from(arguments: argparse.Namespace) -> int:
    """The Workshop runway profile's level this invocation plays on."""
    return int(arguments.workshop_level)


def cadence_from(arguments: argparse.Namespace) -> CadenceConfig:
    return CadenceConfig(
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
    )


def decision_cadence_from(arguments: argparse.Namespace) -> DecisionCadence:
    """The named cadence policy this invocation collects or plays under."""
    return DecisionCadence(arguments.decision_cadence)


def emulator_command_lines(serial: str, proc: Path = Path("/proc")) -> list[str]:
    """The command line of every emulator process that holds `serial`'s console port.

    Only a process whose `/proc/<pid>/exe` is the emulator or a `qemu-system-*`
    binary counts, as in `spectate.running_emulators`: any other process can
    quote an emulator's argv (a shell, an editor, this script's own caller). Of
    those, the ones whose `/proc/<pid>/cmdline` carries `-port <console port>`,
    the reading `run_stage.sh` takes. Empty when the serial is not an
    emulator's or no emulator holds the port.
    """
    port = serial.removeprefix("emulator-")
    if port == serial or not port.isdigit():
        return []
    try:
        entries = sorted(proc.iterdir())
    except OSError:
        return []
    found: list[str] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            executable = Path(os.readlink(entry / "exe")).name
            raw = (entry / "cmdline").read_bytes()
        except OSError:  # gone, or not ours to look at
            continue
        if not (executable.startswith("qemu-system") or executable == "emulator"):
            continue
        text = " " + raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if f" -port {port} " in text:
            found.append(text)
    return found


def refuse_an_unconfined_workshop_write(
    serial: str, level: int, proc: Path = Path("/proc")
) -> None:
    """Refuse a Workshop write unless the instance was launched `-read-only` (ADR 0012).

    The runway profile is written into a live game whose save could carry it;
    only a read-only instance guarantees that nothing it does outlives it. At
    level 0 nothing is written, so there is nothing to confine.
    """
    if level == WORKSHOP_OFF:
        return
    command_lines = emulator_command_lines(serial, proc)
    if not command_lines:
        raise SystemExit(
            f"WORKSHOP_NOT_CONFINED: no emulator process holds {serial}'s port, so it "
            "cannot be shown to be read-only; refusing a Workshop write"
        )
    if any(" -read-only " not in command_line for command_line in command_lines):
        raise SystemExit(
            f"WORKSHOP_NOT_CONFINED: {serial} was not launched -read-only; refusing a "
            "Workshop write"
        )


def open_environment(
    serial: str, port: int, expected: BridgeCompatibility, arguments: argparse.Namespace
) -> tuple[InstrumentedBridgeClient, InstrumentedRunAdapter, InstrumentedRunEnvironment]:
    """Connect the bridge and build the adapter and environment on top of it.

    The wiring every collection entry point (`run_episodes.py`, `train.py`,
    `spectate.py`) needs identically: the same client timeouts, then the
    adapter, then the environment built from this invocation's cadence,
    upgrade-availability and Workshop arguments. A Workshop level above 0 is
    refused before anything connects unless `serial` runs `-read-only`. The
    caller keeps its own connect-time side effects (printing the handshake,
    reading slot labels, wrapping the result) and its own `release()`/`close()`
    in `finally`.
    """
    refuse_an_unconfined_workshop_write(serial, workshop_level_from(arguments))
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        port,
        expected_compatibility=expected,
        connect_timeout=5.0,
        read_timeout=120.0,
        heartbeat_timeout=60.0,
    )
    client.connect()
    adapter = InstrumentedRunAdapter(client=client)
    environment = InstrumentedRunEnvironment(
        port=adapter,
        builder=RunStateBuilder(profile_id=expected.profile_id),
        cadence=cadence_from(arguments),
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
        workshop_level=workshop_level_from(arguments),
    )
    return client, adapter, environment


def print_workshop_rows(client: InstrumentedBridgeClient, level: int) -> None:
    """The Workshop rows as the game names them, for a device session to read.

    At 0 this only reads. Above 0 it applies the runway profile once, exactly
    as a round start would, and prints the report the write came back with:
    each family's implemented count, and every row's in-run name, Workshop
    maximum, and Workshop level before and after. Whether the level took hold
    in combat is read elsewhere: from the tower stats in the first observation
    of a round (`damage`, `attackSpeed`, `thornDamage`, ...), against level 0.
    """
    sequence = client.read_state().sequence
    if level > WORKSHOP_OFF:
        report = client.set_workshop_levels(
            level, workshop_rows(level), expected_sequence=sequence
        )
    else:
        report = client.read_workshop_levels(expected_sequence=sequence)
    print(json.dumps(
        {
            "wrote": report.wrote,
            "implemented": dict(report.implemented),
            "rows": [vars(row) for row in report.rows],
        },
        indent=2,
    ), flush=True)


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
    add_workshop_level_argument(parser)
    parser.add_argument(
        "--list-workshop-rows",
        action="store_true",
        help="print every Workshop row as the game names it, with its level and "
        "ceiling, and exit without playing; with --workshop-level above 0, apply "
        "the runway profile once first and print its before/after report (ADR 0012)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "where the record is written; default <run folder>/evaluations/"
            "<name>/<serial>.json for a checkpoint, state/records/episodes.json "
            "for a floor"
        ),
    )
    add_evaluation_name_argument(parser)
    arguments = parser.parse_args()

    if arguments.serial == "emulator-5554":
        raise SystemExit("refusing to run against the canonical evaluation AVD")
    settle_output(arguments)

    # Before the device is touched: a checkpoint that cannot be rebuilt should
    # fail now, not after an emulator has been brought up for it.
    policy, identity = policy_from(
        arguments.policy,
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
        workshop_level=workshop_level_from(arguments),
        sampling_seed=arguments.serial,
    )
    expected = compatibility(bridge_build_directory())

    client, adapter, environment = open_environment(
        arguments.serial, arguments.port, expected, arguments
    )
    handshake = client.handshake
    print(
        f"bridge {handshake.bridge_version} profile {handshake.compatibility.profile_id} "
        f"speed {handshake.game_speed}",
        flush=True,
    )
    # Before the first round: a command of the adapter's own initiative belongs
    # to the episode boundary, and these are constant for the build.
    labels = adapter.slot_labels()
    if arguments.list_workshop_rows:
        try:
            print_workshop_rows(client, workshop_level_from(arguments))
        finally:
            adapter.release()
            client.close()
        return 0

    # A checkpoint is played only on the setup its run was played on; its first
    # episode here must reproduce the digest it recorded.
    expected_setup = identity.get("upgrade_setup_digest")
    environment.setup_reference = UpgradeSetupReference(
        expected=expected_setup if isinstance(expected_setup, str) else None,
        expected_from=str(identity.get("checkpoint_path", identity["name"])),
    )
    started = time.monotonic()
    try:
        bind_row_names(policy, labels)
        report = evaluate(
            environment,
            policy,
            episodes=arguments.episodes,
            profile_id=expected.profile_id,
            actor_config=ActorConfig(actor_id=f"{arguments.serial}:{identity['name']}"),
        )
    except UpgradeSetupRefused as refused:
        raise SystemExit(f"cannot play {identity['name']} here: {refused}") from refused
    finally:
        adapter.release()
        client.close()

    record = actor_record(
        report,
        identity,
        frame_game_ms=arguments.frame_game_ms,
        max_quiet_game_ms=arguments.max_quiet_game_ms,
        decision_cadence=decision_cadence_from(arguments),
        upgrade_availability=upgrade_availability_from(arguments),
        workshop_level=workshop_level_from(arguments),
        wall_seconds=time.monotonic() - started,
        labels=labels,
        upgrade_setup=environment.setup_reference.first,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(record, indent=2))
    print(report.summary_line(), flush=True)
    print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
