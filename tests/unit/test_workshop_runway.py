"""The Workshop runway profile (ADR 0012): written before a round, held through it.

The controller writes one Workshop level into a fixed set of rows, by name,
before `begin_episode`, and reads the rows back once the round has started. A
level that did not land refuses the episode; a level the round start put back
invalidates it by name. Off - level 0 - issues no Workshop command at all.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import run_episodes
from fakes.fake_run_port import FakeRunPort
from test_instrumented_bridge import _connected_client, _observation

from tower_rl.environment.episode import TerminationOutcome
from tower_rl.environment.run_actions import WAIT
from tower_rl.environment.run_environment import (
    WORKSHOP_NOT_APPLIED,
    WORKSHOP_REVERTED,
    CadenceConfig,
    DecisionCadence,
    InstrumentedRunEnvironment,
    UpgradeAvailability,
)
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import RunStateBuilder
from tower_rl.environment.workshop import WORKSHOP_OFF, WORKSHOP_RUNWAY_ROWS, workshop_rows
from tower_rl.experiment.run_identity import (
    REFERENCE_FINAL_WAVES,
    SCRIPTED_REFERENCE,
    RunIdentity,
    checkpoint_identity,
    reference_final_waves,
    scripted_reference,
    tracked_params,
)
from tower_rl.learning.checkpoint import identity_hash
from tower_rl.learning.evaluator import episode_record
from tower_rl.simulation.instrumented_bridge import (
    BridgeProtocolError,
    decode_command,
    decode_workshop_state,
    encode_frame,
    read_frame,
)
from tower_rl.simulation.instrumented_run_adapter import InstrumentedRunAdapter


def _environment(
    level: int, **port_kwargs: object
) -> tuple[InstrumentedRunEnvironment, FakeRunPort]:
    port = FakeRunPort(**port_kwargs)  # type: ignore[arg-type]
    environment = InstrumentedRunEnvironment(
        port=port,
        builder=RunStateBuilder(profile_id="fake-profile-v1"),
        cadence=CadenceConfig(max_quiet_game_ms=1000),
        decision_cadence=DecisionCadence.CHOICE_POINTS,
        upgrade_availability=UpgradeAvailability.IMAGE,
        workshop_level=level,
    )
    return environment, port


def test_the_profile_is_eleven_named_rows_or_nothing() -> None:
    assert workshop_rows(WORKSHOP_OFF) == ()
    assert workshop_rows(5) == WORKSHOP_RUNWAY_ROWS
    assert len(WORKSHOP_RUNWAY_ROWS) == 11 and len(set(WORKSHOP_RUNWAY_ROWS)) == 11
    with pytest.raises(ValueError):
        workshop_rows(-1)


def test_off_issues_no_workshop_command() -> None:
    environment, port = _environment(WORKSHOP_OFF)

    environment.reset()

    assert port.workshop_commands == [("begin", 0, ())]


def test_the_levels_are_written_before_the_round_and_read_back_after_it() -> None:
    environment, port = _environment(5)

    state = environment.reset()

    assert port.workshop_commands == [
        ("set", 5, WORKSHOP_RUNWAY_ROWS),
        ("begin", 0, ()),
        ("read", 0, ()),
    ]
    assert state.valid
    held = {row.name: row.after for row in port.workshop_levels()}
    assert all(held[name] == 5 for name in WORKSHOP_RUNWAY_ROWS)
    # The rows the ADR holds at zero are left alone.
    assert held["Orbs"] == 0 and held["Death Defy"] == 0 and held["Interest / Wave"] == 0


def test_a_level_the_round_start_put_back_invalidates_the_episode_by_name() -> None:
    environment, _ = _environment(5, workshop_reverts_at_round_start=True)

    state = environment.reset()

    assert not state.valid
    assert any(reason.startswith(WORKSHOP_REVERTED) for reason in state.invalid_reasons)
    transition = environment.step(WAIT)
    assert transition.termination is TerminationOutcome.OBSERVATION_INVALID
    assert not transition.admissible


def test_a_name_the_game_does_not_have_refuses_the_episode() -> None:
    names = {"attack": ("Damage",), "defense": ("Health",), "utility": ("Cash Bonus",)}
    environment, port = _environment(5, workshop_names=names)

    with pytest.raises(RunPortError, match=WORKSHOP_NOT_APPLIED) as refused:
        environment.reset()

    assert "workshop_row_unknown" in str(refused.value)
    assert ("begin", 0, ()) not in port.workshop_commands


def test_a_level_above_a_rows_maximum_refuses_the_episode() -> None:
    environment, _ = _environment(5, workshop_max_level=3)

    with pytest.raises(RunPortError, match="workshop_level_above_max"):
        environment.reset()


def test_the_episode_record_and_the_arm_say_which_profile_they_played() -> None:
    environment, _ = _environment(5)
    environment.reset()
    summary = environment.summarize(TerminationOutcome.OPERATOR_STOP)

    record = episode_record(0, summary)
    assert record["workshop_level"] == 5
    assert record["workshop_rows"] == list(WORKSHOP_RUNWAY_ROWS)

    v1 = checkpoint_identity(
        RunIdentity.started_now("arm", profile_id="p", source_revision="abc")
    )
    v2 = replace(v1, workshop_level=5)
    assert any("workshop_level" in reason for reason in v2.incompatibilities(v1))
    # The level is refused, not hashed: existing checkpoint keys keep resolving.
    assert identity_hash(v1) == identity_hash(v2)


def _workshop_message(*, wrote: bool) -> dict[str, object]:
    return {
        "type": "workshop_state", "protocol_version": 2, "wrote": wrote,
        "rows": [
            {"family": "attack", "index": 0, "name": "Damage", "max_level": 5000,
             "before": 0, "after": 5 if wrote else 0},
            {"family": "defense", "index": 9, "name": "Orbs", "max_level": 5,
             "before": 0, "after": 0},
        ],
        "families": [
            {"family": "attack", "implemented": 17},
            {"family": "defense", "implemented": 18},
            {"family": "utility", "implemented": 13},
        ],
    }


def test_the_workshop_report_decodes_every_row_and_each_implemented_count() -> None:
    report = decode_workshop_state(_workshop_message(wrote=True))

    assert report.wrote
    assert [(row.name, row.before, row.after) for row in report.rows] == [
        ("Damage", 0, 5), ("Orbs", 0, 0)
    ]
    assert dict(report.implemented) == {"attack": 17, "defense": 18, "utility": 13}
    missing = {**_workshop_message(wrote=True), "families": []}
    with pytest.raises(BridgeProtocolError):
        decode_workshop_state(missing)


def test_a_held_check_that_cannot_read_refuses_the_episode() -> None:
    """The write landed, but the round start's read did not: never assumed held."""
    environment, port = _environment(5, refuse_workshop_read=True)

    with pytest.raises(RunPortError, match=WORKSHOP_NOT_APPLIED):
        environment.reset()

    assert [command[0] for command in port.workshop_commands] == ["set", "begin", "read"]


_QEMU = "/opt/android-sdk/emulator/qemu/linux-x86_64/qemu-system-x86_64"


def _proc(root: Path, *processes: tuple[str, str]) -> Path:
    """A `/proc` holding these (executable, argv) processes, the shape the check reads."""
    for pid, (executable, command_line) in enumerate(processes, start=4242):
        entry = root / str(pid)
        entry.mkdir(parents=True)
        (entry / "exe").symlink_to(executable)
        (entry / "cmdline").write_bytes(command_line.replace(" ", "\0").encode() + b"\0")
    (root / "self").mkdir(parents=True)
    return root


def test_a_workshop_write_is_refused_unless_the_instance_is_read_only(tmp_path: Path) -> None:
    writable = _proc(tmp_path / "a", (_QEMU, "qemu-system-x86_64 -avd clone -port 5556 -no-window"))
    confined = _proc(tmp_path / "b", (_QEMU, "qemu-system-x86_64 -avd clone -port 5556 -read-only"))
    # A shell quoting a read-only argv is not an emulator, so it confines nothing.
    quoted = _proc(tmp_path / "c", ("/usr/bin/bash", "bash -c emulator -port 5556 -read-only"))
    # One of two emulators on the port is writable: the write is refused.
    mixed = _proc(
        tmp_path / "d",
        (_QEMU, "qemu-system-x86_64 -avd clone -port 5556 -read-only"),
        (_QEMU, "qemu-system-x86_64 -avd clone -port 5556 -no-window"),
    )

    with pytest.raises(SystemExit, match="WORKSHOP_NOT_CONFINED.*not launched -read-only"):
        run_episodes.refuse_an_unconfined_workshop_write("emulator-5556", 5, writable)
    with pytest.raises(SystemExit, match="WORKSHOP_NOT_CONFINED.*no emulator process"):
        run_episodes.refuse_an_unconfined_workshop_write("emulator-5558", 5, confined)
    with pytest.raises(SystemExit, match="WORKSHOP_NOT_CONFINED.*no emulator process"):
        run_episodes.refuse_an_unconfined_workshop_write("emulator-5556", 5, quoted)
    with pytest.raises(SystemExit, match="WORKSHOP_NOT_CONFINED.*not launched -read-only"):
        run_episodes.refuse_an_unconfined_workshop_write("emulator-5556", 5, mixed)
    run_episodes.refuse_an_unconfined_workshop_write("emulator-5556", 5, confined)
    # Level 0 writes nothing, so it needs no confinement.
    run_episodes.refuse_an_unconfined_workshop_write("emulator-5556", 0, writable)


def test_the_cli_level_is_bounded_like_the_bridge() -> None:
    parser = argparse.ArgumentParser()
    run_episodes.add_workshop_level_argument(parser)

    assert parser.parse_args(["--workshop-level", "9999"]).workshop_level == 9999
    for bad in ("-1", "10000"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--workshop-level", bad])


def test_a_runway_run_is_given_no_v1_floor() -> None:
    assert reference_final_waves(0) == REFERENCE_FINAL_WAVES
    assert scripted_reference(0) == SCRIPTED_REFERENCE
    assert reference_final_waves(5) is None and scripted_reference(5) is None
    assert "reference_scripted" in tracked_params({"workshop_level": 0})
    assert not any(key.startswith("reference_") for key in tracked_params({"workshop_level": 5}))


def test_the_write_command_is_held_to_the_native_parser() -> None:
    command = {
        "type": "command", "protocol_version": 2, "request_id": "w-1",
        "expected_observation_sequence": 1, "kind": "set_workshop_levels",
        "level": 5, "rows": ["Damage", "Defense %"],
    }
    assert decode_command(command).rows == ("Damage", "Defense %")
    for bad in ({"level": 0}, {"rows": []}, {"rows": ['Da"mage']}, {"family": "attack"}):
        with pytest.raises(BridgeProtocolError):
            decode_command({**command, **bad})


def test_the_adapter_writes_in_one_round_trip_and_reports_every_row() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        requests: list[dict[str, object]] = []

        def bridge() -> None:
            """Answer the way the bridge does: report, state, result."""
            requests.append(read_frame(peer, timeout=2.0))
            peer.sendall(encode_frame(_workshop_message(wrote=True)))
            peer.sendall(encode_frame(_observation(2)))
            peer.sendall(
                encode_frame(
                    {
                        "type": "command_result", "protocol_version": 2,
                        "request_id": str(requests[0]["request_id"]),
                        "outcome": "confirmed", "reason": "workshop_levels_applied",
                        "observation_sequence": 2,
                    }
                )
            )

        responder = threading.Thread(target=bridge)
        responder.start()
        adapter = InstrumentedRunAdapter(client=client)
        rows = adapter.set_workshop_levels(5, ("Damage",))
        responder.join(timeout=5.0)

        assert requests[0]["kind"] == "set_workshop_levels"
        assert requests[0]["level"] == 5 and requests[0]["rows"] == ["Damage"]
        assert {row.name: row.after for row in rows} == {"Damage": 5, "Orbs": 0}
    finally:
        client.close()
        peer.close()
