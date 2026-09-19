from __future__ import annotations

import select
import socket
import struct
import threading
import time
from contextlib import suppress

import pytest

from tower_rl.environment.run_environment import CadenceConfig, InstrumentedRunEnvironment
from tower_rl.environment.run_state import LIVE_WIRE_NAMES, RunStateBuilder
from tower_rl.learning.evaluator import evaluate
from tower_rl.simulation.instrumented_bridge import (
    ADVANCE_WALL_CEILING_SECONDS,
    DEFAULT_READ_TIMEOUT_SECONDS,
    PAUSE_SETTLE_SECONDS,
    BridgeCompatibility,
    BridgeCompatibilityError,
    BridgeDisconnectedError,
    BridgeProtocolError,
    BridgeRunUnavailable,
    BridgeRunUnavailableError,
    BridgeStaleObservationError,
    BridgeTimeoutError,
    InstrumentedBridgeClient,
    decode_command,
    decode_command_result,
    decode_handshake,
    decode_observation,
    decode_slot_labels,
    decode_unlock_state,
    encode_frame,
    read_frame,
)
from tower_rl.simulation.instrumented_run_adapter import InstrumentedRunAdapter

EXPECTED = BridgeCompatibility(
    package_version="29.0.3",
    package_version_code=1199,
    official_signer_sha256="a" * 64,
    original_libunity_sha256="b" * 64,
    libil2cpp_sha256="c" * 64,
    unity_version="6000.3.15f1",
    il2cpp_metadata_version=39,
    bridge_version="tower-bridge-v1",
    profile_id="private-training-v1",
)


def _handshake(**overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "handshake",
        "protocol_version": 2,
        "bridge_version": "tower-bridge-v1",
        "mode": "instrumented_training",
        "command_capability": "semantic-v2",
        "compatibility": {
            "package_version": "29.0.3",
            "package_version_code": 1199,
            "official_signer_sha256": "a" * 64,
            "original_libunity_sha256": "b" * 64,
            "libil2cpp_sha256": "c" * 64,
            "unity_version": "6000.3.15f1",
            "il2cpp_metadata_version": 39,
            "profile_id": "private-training-v1",
        },
        "game_speed": 1.0,
    }
    message.update(overrides)
    return message


#: The inventory the domain requires of a real reading: every family, every slot.
#: Most of this file decodes one entry and says so; anything driving the whole
#: pipeline - the environment, the evaluator - needs the real shape.
FULL_INVENTORY = [
    {
        "family": family, "index": index, "cost": 5.0, "level": 2, "max_level": 10,
        "unlocked": True, "tier_unlocked": True, "maxed": False,
    }
    for family in ("attack", "defense", "utility")
    for index in range(20)
]


#: Every `observation-v2` live reading a state message must carry. The values
#: are the resting ones a device produced for an unexercised stat (board #39);
#: what matters here is that the whole declared set is present, because a
#: message missing one is a bridge that does not speak this schema.
LIVE_READINGS = {**dict.fromkeys(LIVE_WIRE_NAMES, 0.0), "damage": 12.09, "waveTimer": 4.5}


def _observation(
    sequence: int = 1,
    wave: int = 7,
    *,
    terminal: bool = False,
    speed: float = 1.5,
    full_inventory: bool = False,
    live: dict[str, float] | None = None,
) -> dict[str, object]:
    if full_inventory:
        return {**_observation(sequence, wave, terminal=terminal, speed=speed, live=live),
                "upgrades": FULL_INVENTORY}
    return {
        "live": LIVE_READINGS if live is None else live,
        "type": "observation",
        "sequence": sequence,
        "lifecycle": "terminal" if terminal else "active",
        "wave": wave,
        "cash": 123.5,
        "health": 0.0 if terminal else 95.0,
        "max_health": 100.0,
        "terminal": terminal,
        "round_active": not terminal,
        "game_speed": speed,
        "play_time": 3546.9,
        "upgrades": [
            {
                "family": "attack",
                "index": 0,
                "cost": 5.0,
                "level": 2,
                "max_level": 10,
                "unlocked": True,
                "tier_unlocked": True,
                "maxed": False,
            }
        ],
    }


def _connected_client() -> tuple[InstrumentedBridgeClient, socket.socket]:
    client_socket, peer_socket = socket.socketpair()
    client = InstrumentedBridgeClient(
        "127.0.0.1", 47651, expected_compatibility=EXPECTED, read_timeout=0.1
    )
    client._socket = client_socket
    client._handshake = decode_handshake(_handshake(), EXPECTED)
    return client, peer_socket


def test_frame_round_trip_accepts_one_complete_object() -> None:
    client, peer = socket.socketpair()
    try:
        peer.sendall(encode_frame({"type": "heartbeat", "last_observation_sequence": 3}))

        assert read_frame(client, timeout=0.1) == {
            "type": "heartbeat",
            "last_observation_sequence": 3,
        }
    finally:
        client.close()
        peer.close()


def test_malformed_and_oversized_frames_fail_closed() -> None:
    client, peer = socket.socketpair()
    try:
        peer.sendall(struct.pack("!I", 5) + b"[1,2]")
        with pytest.raises(BridgeProtocolError, match="root"):
            read_frame(client, timeout=0.1)

        peer.sendall(struct.pack("!I", 11))
        with pytest.raises(BridgeProtocolError, match="outside allowed bounds"):
            read_frame(client, timeout=0.1, max_frame_size=10)
    finally:
        client.close()
        peer.close()


def test_version_or_compatibility_mismatch_is_rejected() -> None:
    with pytest.raises(BridgeCompatibilityError, match="protocol version mismatch"):
        decode_handshake(_handshake(protocol_version=1), EXPECTED)

    incompatible = _handshake(
        compatibility={
            "package_version": "29.0.4",
            "package_version_code": 1199,
            "official_signer_sha256": "a" * 64,
            "original_libunity_sha256": "b" * 64,
            "libil2cpp_sha256": "c" * 64,
            "unity_version": "6000.3.15f1",
            "il2cpp_metadata_version": 39,
            "profile_id": "private-training-v1",
        }
    )
    with pytest.raises(BridgeCompatibilityError, match="compatibility mismatch"):
        decode_handshake(incompatible, EXPECTED)


def test_handshake_rejects_bad_or_drifted_immutable_hashes() -> None:
    malformed = _handshake()
    malformed["compatibility"] = {
        **malformed["compatibility"],  # type: ignore[arg-type]
        "original_libunity_sha256": "B" * 64,
    }
    with pytest.raises(BridgeProtocolError, match="SHA-256"):
        decode_handshake(malformed, EXPECTED)

    drifted_bridge = _handshake(bridge_version="tower-bridge-v2")
    with pytest.raises(BridgeCompatibilityError, match="compatibility mismatch"):
        decode_handshake(drifted_bridge, EXPECTED)


def test_valid_observation_decodes_and_stale_sequence_is_rejected() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(4)))
        observation = client.read_observation()

        assert observation.sequence == 4
        assert observation.upgrades[0].family == "attack"
        assert observation.upgrades[0].level == 2

        peer.sendall(encode_frame(_observation(4)))
        with pytest.raises(BridgeStaleObservationError, match="not newer"):
            client.read_observation()
    finally:
        client.close()
        peer.close()


def test_a_state_message_must_carry_exactly_the_live_readings_the_schema_declares() -> None:
    """Never a silent zero: a missing reading is a bridge that is not v2.

    The values themselves pass through raw - scaling and the range invariant are
    the environment's - so what the wire owes is the whole declared set, no more
    and no less.
    """
    whole = decode_observation(_observation(1))

    assert set(whole.live) == set(LIVE_WIRE_NAMES)
    assert whole.live["damage"] == pytest.approx(12.09)

    missing = {name: 0.0 for name in LIVE_WIRE_NAMES if name != "waveTimer"}
    with pytest.raises(BridgeProtocolError, match="missing live readings"):
        decode_observation(_observation(1, live=missing))

    extra = {**dict.fromkeys(LIVE_WIRE_NAMES, 0.0), "cellsEarnedThisWave": 1.0}
    # Named, not counted: "one too many" leaves the reader to diff two lists of
    # thirty-seven to find out which field the bridge sent.
    with pytest.raises(BridgeProtocolError, match="cannot place.*cellsEarnedThisWave"):
        decode_observation(_observation(1, live=extra))

    with pytest.raises(BridgeProtocolError, match="no live readings object"):
        decode_observation({k: v for k, v in _observation(1).items() if k != "live"})


def test_the_upgrade_row_labels_decode_with_the_games_own_empty_tail() -> None:
    labels = decode_slot_labels(
        {
            "type": "slot_labels",
            "protocol_version": 2,
            "labels": [
                {"family": "attack", "index": 0, "name": "Damage", "description": "Tower damage"},
                # The game's own trailing empties, carried so the slot indices
                # stay aligned with the action schema.
                {"family": "attack", "index": 19, "name": "", "description": ""},
            ],
        }
    )

    assert labels[0].name == "Damage" and labels[1].name == ""

    with pytest.raises(BridgeProtocolError, match="duplicate slot label"):
        decode_slot_labels(
            {
                "type": "slot_labels",
                "protocol_version": 2,
                "labels": [
                    {"family": "attack", "index": 0, "name": "a", "description": "b"},
                    {"family": "attack", "index": 0, "name": "a", "description": "b"},
                ],
            }
        )


def test_the_slot_label_command_is_one_round_trip_and_the_answer_is_remembered() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        requests: list[dict[str, object]] = []

        def bridge() -> None:
            """Answer the command the way the bridge does: labels, state, result."""
            requests.append(read_frame(peer, timeout=2.0))
            peer.sendall(
                encode_frame(
                    {
                        "type": "slot_labels", "protocol_version": 2,
                        "labels": [
                            {"family": "utility", "index": 1, "name": "Cash / Wave",
                             "description": "Cash each wave"},
                        ],
                    }
                )
            )
            peer.sendall(encode_frame(_observation(2)))
            peer.sendall(
                encode_frame(
                    {
                        "type": "command_result", "protocol_version": 2,
                        "request_id": str(requests[0]["request_id"]),
                        "outcome": "confirmed", "reason": "slot_labels_reported",
                        "observation_sequence": 2,
                    }
                )
            )

        responder = threading.Thread(target=bridge)
        responder.start()
        adapter = InstrumentedRunAdapter(client=client)
        labels = adapter.slot_labels()
        responder.join(timeout=5.0)

        assert labels[0].name == "Cash / Wave"
        # Constant for the build, so asking again costs no round trip at all.
        assert adapter.slot_labels() is labels
        assert requests[0]["kind"] == "slot_labels"
        assert requests[0]["protocol_version"] == 2
    finally:
        client.close()
        peer.close()


def test_heartbeat_must_confirm_the_latest_observation() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        assert client.read_observation().sequence == 1

        peer.sendall(encode_frame({"type": "heartbeat", "last_observation_sequence": 1}))
        peer.sendall(encode_frame(_observation(2)))
        assert client.read_observation().sequence == 2

        client, peer = _connected_client()
        peer.sendall(encode_frame({"type": "heartbeat", "last_observation_sequence": 3}))
        with pytest.raises(BridgeStaleObservationError, match="does not match"):
            client.read_observation()
    finally:
        client.close()
        peer.close()


def test_timeout_and_eof_are_distinct() -> None:
    client, peer = socket.socketpair()
    try:
        with pytest.raises(BridgeTimeoutError, match="timed out"):
            read_frame(client, timeout=0.01)
        peer.close()
        with pytest.raises(BridgeDisconnectedError, match="closed"):
            read_frame(client, timeout=0.1)
    finally:
        client.close()


def test_command_contract_rejects_malformed_and_stale_requests() -> None:
    malformed_advance = {
        "type": "command", "protocol_version": 2, "request_id": "a",
        "expected_observation_sequence": 1, "kind": "advance", "index": 0,
        "budget_game_ms": 2000, "frame_game_ms": 16.0, "health_change_fraction": 0.05,
    }
    with pytest.raises(BridgeProtocolError, match="upgrade target"):
        decode_command(malformed_advance)
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        client.read_observation()
        stale_lifecycle = {
            "type": "command", "protocol_version": 2, "request_id": "a",
            "expected_observation_sequence": 2, "kind": "lifecycle", "action": "pause",
        }
        with pytest.raises(BridgeStaleObservationError, match="latest"):
            client.send_command(stale_lifecycle)
        other_result = {
            "type": "command_result", "protocol_version": 2, "request_id": "b",
            "outcome": "confirmed", "reason": "run_active", "observation_sequence": 1,
        }
        peer.sendall(encode_frame(other_result))
        mine = {
            "type": "command", "protocol_version": 2, "request_id": "a",
            "expected_observation_sequence": 1, "kind": "lifecycle", "action": "pause",
        }
        with pytest.raises(BridgeProtocolError, match="request id"):
            client.send_command(mine)
    finally:
        client.close()
        peer.close()


def test_command_wire_format_matches_the_native_parser_contract() -> None:
    """The native parser reads fixed canonical offsets, so the encoding is pinned."""
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        client.read_observation()
        result = {
            "type": "command_result", "protocol_version": 2, "request_id": "buy-1",
            "outcome": "confirmed", "reason": "confirmed_state_change",
            "observation_sequence": 1,
        }
        peer.sendall(encode_frame(result))
        client.send_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "buy-1",
                "expected_observation_sequence": 1, "kind": "buy_upgrade",
                "family": "attack", "index": 3,
            }
        )
        payload = read_frame(peer, timeout=0.1)
        raw = encode_frame(payload)[4:].decode("utf-8")

        prefix = '{"type":"command","protocol_version":2,"request_id":"'
        sequence_key = '","expected_observation_sequence":'
        assert raw.startswith(prefix)
        assert len(prefix) == 53
        assert len(sequence_key) == 34
        assert raw[len(prefix) :].startswith("buy-1" + sequence_key)
        assert raw.endswith('"kind":"buy_upgrade","family":"attack","index":3}')
    finally:
        client.close()
        peer.close()


def test_read_observation_returns_the_newest_buffered_snapshot() -> None:
    client, peer = _connected_client()
    try:
        for sequence in (1, 2, 3):
            peer.sendall(encode_frame(_observation(sequence)))
        peer.sendall(encode_frame({"type": "heartbeat", "last_observation_sequence": 3}))

        assert client.read_observation().sequence == 3
    finally:
        client.close()
        peer.close()


def test_no_initialized_run_is_state_not_an_observation() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(
            encode_frame(
                {"type": "run_unavailable", "sequence": 3, "reason": "no_initialized_run"}
            )
        )
        state = client.read_state()

        assert state == BridgeRunUnavailable(3, "no_initialized_run")

        peer.sendall(
            encode_frame(
                {"type": "run_unavailable", "sequence": 4, "reason": "no_initialized_run"}
            )
        )
        with pytest.raises(BridgeRunUnavailableError, match="no exact run state"):
            client.read_observation()
    finally:
        client.close()
        peer.close()


def test_state_sequence_must_advance_across_both_state_kinds() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(5)))
        assert client.read_state().sequence == 5

        peer.sendall(
            encode_frame(
                {"type": "run_unavailable", "sequence": 5, "reason": "no_initialized_run"}
            )
        )
        with pytest.raises(BridgeStaleObservationError, match="not newer"):
            client.read_state()
    finally:
        client.close()
        peer.close()


def test_lifecycle_commands_are_separate_from_policy_actions() -> None:
    lifecycle = decode_command(
        {
            "type": "command", "protocol_version": 2, "request_id": "r",
            "expected_observation_sequence": 2, "kind": "lifecycle", "action": "start_round",
        }
    )

    assert lifecycle.kind == "lifecycle"
    assert lifecycle.action == "start_round"
    assert lifecycle.family is None and lifecycle.index is None

    with pytest.raises(BridgeProtocolError, match="unsupported lifecycle action"):
        decode_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "r",
                "expected_observation_sequence": 2, "kind": "lifecycle", "action": "buy_gems",
            }
        )
    with pytest.raises(BridgeProtocolError, match="must not contain an upgrade target"):
        decode_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "r",
                "expected_observation_sequence": 2, "kind": "lifecycle",
                "action": "start_round", "family": "attack", "index": 0,
            }
        )
    with pytest.raises(BridgeProtocolError, match="only a lifecycle command"):
        decode_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "r",
                "expected_observation_sequence": 2, "kind": "advance", "action": "start_round",
                "budget_game_ms": 2000, "frame_game_ms": 16.0,
                "health_change_fraction": 0.05,
            }
        )


def test_lifecycle_wire_format_matches_the_native_parser_contract() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        client.read_state()
        peer.sendall(
            encode_frame(
                {
                    "type": "command_result", "protocol_version": 2, "request_id": "life-1",
                    "outcome": "confirmed", "reason": "run_active", "observation_sequence": 1,
                }
            )
        )
        client.send_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "life-1",
                "expected_observation_sequence": 1, "kind": "lifecycle",
                "action": "start_round",
            }
        )
        raw = encode_frame(read_frame(peer, timeout=0.1))[4:].decode("utf-8")

        assert raw.endswith('"kind":"lifecycle","action":"start_round"}')
    finally:
        client.close()
        peer.close()


def test_advance_wire_format_matches_the_native_parser_contract() -> None:
    """The native parser reads these three fields at fixed offsets, in this order."""
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        client.read_state()
        peer.sendall(
            encode_frame(
                {
                    "type": "command_result", "protocol_version": 2, "request_id": "adv-1",
                    "outcome": "confirmed", "reason": "budget_exhausted",
                    "observation_sequence": 1, "frames": 120, "game_ms": 2000,
                    "round_ms": 1998, "wall_micros": 57_000,
                }
            )
        )
        client.send_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "adv-1",
                "expected_observation_sequence": 1, "kind": "advance",
                "budget_game_ms": 2000, "frame_game_ms": 16.5,
                "health_change_fraction": 0.05,
            }
        )
        raw = encode_frame(read_frame(peer, timeout=0.1))[4:].decode("utf-8")

        assert raw.endswith(
            '"kind":"advance","budget_game_ms":2000,"frame_game_ms":16.5,'
            '"health_change_fraction":0.05}'
        )
    finally:
        client.close()
        peer.close()


def test_advance_command_bounds_every_field_it_carries() -> None:
    """One advance replaces the host's slicing loop, so its bounds are the contract."""
    advance = decode_command(
        {
            "type": "command", "protocol_version": 2, "request_id": "a",
            "expected_observation_sequence": 1, "kind": "advance",
            "budget_game_ms": 2000, "frame_game_ms": 1000 / 60,
            "health_change_fraction": 0.05,
        }
    )

    assert advance.kind == "advance"
    assert advance.budget_game_ms == 2000
    assert advance.frame_game_ms == 1000 / 60
    assert advance.health_change_fraction == 0.05

    for field, out_of_range in (
        ("budget_game_ms", 9),
        ("budget_game_ms", 10_001),
        ("frame_game_ms", 0.9),
        ("frame_game_ms", 250.1),
        ("health_change_fraction", -0.1),
        ("health_change_fraction", 1.1),
    ):
        message = {
            "type": "command", "protocol_version": 2, "request_id": "a",
            "expected_observation_sequence": 1, "kind": "advance",
            "budget_game_ms": 2000, "frame_game_ms": 16.0,
            "health_change_fraction": 0.05,
        }
        message[field] = out_of_range
        with pytest.raises(BridgeProtocolError):
            decode_command(message)


def test_the_step_and_wait_commands_no_longer_exist() -> None:
    """Slicing from the host is gone, and `WAIT` is an advance; both kinds are dead."""
    for dead in (
        {"kind": "step", "game_ms": 250},
        {"kind": "wait"},
    ):
        with pytest.raises(BridgeProtocolError, match="unsupported command kind"):
            decode_command(
                {
                    "type": "command", "protocol_version": 2, "request_id": "s",
                    "expected_observation_sequence": 1, **dead,
                }
            )


def test_a_command_result_carries_what_the_advance_cost() -> None:
    """Frames, both clocks and wall time are how the speed-up is measured at all.

    The native encoder writes these in one fixed order - `frames`, `game_ms`,
    `round_ms`, `wall_micros` - and `game_ms` against `round_ms` is what shows
    whether the game time each frame was told to be worth actually passed.
    `round_ms` comes from the game's own per-round clock; `playTime` runs at wall
    rate whatever the game clock does, so it could never witness this.
    """
    result = decode_command_result(
        {
            "type": "command_result", "protocol_version": 2, "request_id": "a",
            "outcome": "confirmed", "reason": "event:wave_changed",
            "observation_sequence": 7, "frames": 120, "game_ms": 2000.0,
            "round_ms": 1998.0, "wall_micros": 57_000,
        }
    )

    assert result.reason == "event:wave_changed"
    assert (result.frames, result.game_ms, result.round_ms, result.wall_micros) == (
        120, 2000.0, 1998.0, 57_000,
    )
    # Nothing bound this result to a state, so it honestly carries none.
    assert result.state is None

    # Commands that burn no game time omit them, and zero is the honest answer.
    bare = decode_command_result(
        {
            "type": "command_result", "protocol_version": 2, "request_id": "b",
            "outcome": "confirmed", "reason": "confirmed_state_change",
            "observation_sequence": 7,
        }
    )

    assert (bare.frames, bare.game_ms, bare.round_ms, bare.wall_micros) == (0, 0.0, 0.0, 0)


def test_a_result_carries_the_observation_the_bridge_sent_with_it() -> None:
    """The settled observation arrives with the result, so no second read is needed."""
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        client.read_state()
        peer.sendall(encode_frame(_observation(2, wave=9)))
        peer.sendall(
            encode_frame(
                {
                    "type": "command_result", "protocol_version": 2, "request_id": "adv-2",
                    "outcome": "confirmed", "reason": "event:wave_changed",
                    "observation_sequence": 2, "frames": 60, "game_ms": 1000,
                    "round_ms": 1000, "wall_micros": 21_000,
                }
            )
        )
        result = client.send_command(
            {
                "type": "command", "protocol_version": 2, "request_id": "adv-2",
                "expected_observation_sequence": 1, "kind": "advance",
                "budget_game_ms": 2000, "frame_game_ms": 16.5,
                "health_change_fraction": 0.05,
            }
        )

        assert result.state is not None
        assert result.state.sequence == 2
        assert result.state.wave == 9
    finally:
        client.close()
        peer.close()


def test_any_inbound_frame_proves_the_bridge_is_alive() -> None:
    """A heartbeat is not the only proof of life, and demanding one kills healthy runs.

    With one command in flight per decision the bridge never reaches the idle
    branch that emits heartbeats, so a stream of observations and command results
    is all a working run produces. Liveness therefore has to be renewed by any
    inbound frame; otherwise the deadline trips in the middle of healthy play
    (M1B-E017 - every unattended run died at about sixty seconds).
    """
    client_socket, peer = socket.socketpair()
    client = InstrumentedBridgeClient(
        "127.0.0.1", 47651, expected_compatibility=EXPECTED,
        read_timeout=0.5, heartbeat_timeout=0.05,
    )
    client._socket = client_socket
    client._handshake = decode_handshake(_handshake(), EXPECTED)
    try:
        for sequence in range(1, 8):
            # Each gap is inside the deadline; the whole exchange is well past it.
            time.sleep(0.02)
            peer.sendall(encode_frame(_observation(2 * sequence - 1)))
            assert client.read_state().sequence == 2 * sequence - 1
            peer.sendall(encode_frame(_observation(2 * sequence)))
            peer.sendall(
                encode_frame(
                    {
                        "type": "command_result", "protocol_version": 2,
                        "request_id": f"adv-{sequence}", "outcome": "confirmed",
                        "reason": "budget_exhausted",
                        "observation_sequence": 2 * sequence, "frames": 3,
                        "game_ms": 300, "round_ms": 300, "wall_micros": 50_000,
                    }
                )
            )
            result = client.send_command(
                {
                    "type": "command", "protocol_version": 2,
                    "request_id": f"adv-{sequence}",
                    "expected_observation_sequence": 2 * sequence - 1, "kind": "advance",
                    "budget_game_ms": 2000, "frame_game_ms": 100.0,
                    "health_change_fraction": 0.05,
                }
            )
            assert result.outcome.value == "confirmed"
            read_frame(peer, timeout=0.1)
    finally:
        client.close()
        peer.close()


def test_the_default_read_timeout_covers_the_bridge_advance_ceiling() -> None:
    """A read that gave up first would call a working advance a timeout."""
    assert DEFAULT_READ_TIMEOUT_SECONDS >= ADVANCE_WALL_CEILING_SECONDS + PAUSE_SETTLE_SECONDS
    assert DEFAULT_READ_TIMEOUT_SECONDS >= 20.0
    client = InstrumentedBridgeClient("127.0.0.1", 47651, expected_compatibility=EXPECTED)
    assert client.read_timeout == DEFAULT_READ_TIMEOUT_SECONDS


def test_upgrade_level_above_its_own_maximum_is_rejected() -> None:
    message = _observation(1)
    message["upgrades"] = [
        {
            "family": "attack", "index": 0, "cost": 5.0, "level": 11, "max_level": 10,
            "unlocked": True, "tier_unlocked": True, "maxed": False,
        }
    ]
    with pytest.raises(BridgeProtocolError, match="exceeds its own maximum"):
        decode_observation(message)


class _IdleStreamBridge:
    """The bridge's own stream loop, run against a real socket in a thread.

    It mirrors `ServeClient` in `native/tower_bridge/tower_bridge.cpp`: one
    monotonic sequence, an idle tick that emits state, and the guard that rejects
    a command bound to anything but the standing sequence as `stale_or_duplicate`.
    `holds_the_sequence_while_paused` is the behaviour under test - the bridge now
    emits a heartbeat rather than a fresh observation while the world is paused,
    because a paused world has nothing new to say.
    """

    def __init__(
        self,
        peer: socket.socket,
        *,
        holds_the_sequence_while_paused: bool,
        idle_interval: float = 0.05,
        run_ends_under_advance: bool = False,
        run_ends_in_pause_settle: bool = False,
        full_inventory: bool = False,
    ) -> None:
        self._peer = peer
        self._holds = holds_the_sequence_while_paused
        self._idle_interval = idle_interval
        # Only a test that drives the domain pipeline needs the real inventory
        # shape; the protocol tests are clearer with one entry.
        self._full_inventory = full_inventory
        self._run_ends_under_advance = run_ends_under_advance
        self._run_ends_in_pause_settle = run_ends_in_pause_settle
        self._run_active = True
        self._paused = False
        self._sequence = 0
        self._last_request_id = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> _IdleStreamBridge:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def restart_run(self) -> None:
        """The game starts another round, which the bridge only ever observes.

        On the device this is what `start_round` - the home screen's own BATTLE
        control - causes. The bridge dispatches that control and then waits for
        the game's own state, so its stream simply starts reporting an active
        run again, which it can only do if the ended run left the sequence free
        to move.
        """
        self._run_active = True

    def _state(self) -> dict[str, object]:
        return _observation(
            self._sequence,
            terminal=not self._run_active,
            speed=1.0,
            full_inventory=self._full_inventory,
        )

    def _send(self, message: dict[str, object]) -> None:
        with suppress(OSError):
            self._peer.sendall(encode_frame(message))

    def _result(self, request_id: str, outcome: str, reason: str) -> dict[str, object]:
        return {
            "type": "command_result", "protocol_version": 2, "request_id": request_id,
            "outcome": outcome, "reason": reason, "observation_sequence": self._sequence,
            "frames": 20, "game_ms": 2000, "round_ms": 2000, "wall_micros": 40_000,
        }

    def _serve(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([self._peer], [], [], self._idle_interval)
            if not readable:
                if self._paused and self._holds:
                    self._send(
                        {"type": "heartbeat", "last_observation_sequence": self._sequence}
                    )
                else:
                    self._sequence += 1
                    self._send(self._state())
                continue
            try:
                command = read_frame(self._peer, timeout=1.0)
            except (OSError, BridgeProtocolError, BridgeDisconnectedError, BridgeTimeoutError):
                return
            request_id = str(command["request_id"])
            if (
                command["expected_observation_sequence"] != self._sequence
                or request_id == self._last_request_id
            ):
                self._send(self._result(request_id, "rejected", "stale_or_duplicate"))
                continue
            self._last_request_id = request_id
            was_active = self._run_active
            self._apply_pause_rule(command)
            self._sequence += 1
            self._send(self._state())
            # A real bridge names the decision event it stopped on, and the
            # environment fails a transition where the two disagree: a run that
            # ended under this advance stopped on the health that ended it.
            ended = was_active and not self._run_active
            self._send(
                self._result(
                    request_id,
                    "confirmed",
                    "event:health_changed" if ended else "budget_exhausted",
                )
            )

    def _apply_pause_rule(self, command: dict[str, object]) -> None:
        """Exactly the rule `ServeClient` applies to `world_paused`.

        An advance presses `Pause` when the run is still active as its frame
        loop ends, but what it reports is read after the pause has settled: the
        world is standing still only if the run is still in progress then. A
        tower that dies inside that settle window - the case the real bridge
        deadlocked on - was paused for nothing, and its screens keep changing.
        A lifecycle command holds the world only when a `pause` is confirmed,
        and there too the flag survives only while the run is in progress.
        Every other command - a purchase, a speed change - leaves the pause
        exactly as it found it, which is why a purchase cannot resume the stream.
        """
        kind = command["kind"]
        if kind == "advance":
            if self._run_ends_under_advance:
                self._run_active = False
            pressed_pause = self._run_active
            if pressed_pause and self._run_ends_in_pause_settle:
                self._run_active = False
            self._paused = pressed_pause and self._run_active
        elif kind == "lifecycle":
            if command.get("action") == "start_round":
                self.restart_run()
            self._paused = command.get("action") == "pause" and self._run_active


def _slow_bridge_client() -> tuple[InstrumentedBridgeClient, socket.socket]:
    client_socket, peer_socket = socket.socketpair()
    client = InstrumentedBridgeClient(
        "127.0.0.1", 47651, expected_compatibility=EXPECTED, read_timeout=2.0
    )
    client._socket = client_socket
    client._handshake = decode_handshake(_handshake(), EXPECTED)
    return client, peer_socket


def _advance(request_id: str, sequence: int) -> dict[str, object]:
    return {
        "type": "command", "protocol_version": 2, "request_id": request_id,
        "expected_observation_sequence": sequence, "kind": "advance",
        "budget_game_ms": 2000, "frame_game_ms": 100.0, "health_change_fraction": 0.05,
    }


def test_a_paused_world_holds_the_sequence_across_a_slow_decision() -> None:
    """A policy that thinks for longer than the idle interval is still in time.

    A forward pass plus a learning step routinely costs more than the bridge's
    idle interval. While the world is paused nothing can change, so the sequence
    the host binds must still stand when the command arrives; a device run where
    it did not lost 15 of 35 advances to `stale_or_duplicate`.
    """
    client, peer = _slow_bridge_client()
    try:
        with _IdleStreamBridge(peer, holds_the_sequence_while_paused=True):
            state = client.read_state()
            first = client.send_command(_advance("adv-1", state.sequence))
            assert first.outcome.value == "confirmed"

            # The policy thinking: many idle intervals, and the hot path takes no
            # further read, because the advance already carried its settled state.
            time.sleep(0.4)
            second = client.send_command(_advance("adv-2", first.observation_sequence))

            assert second.outcome.value == "confirmed"
            # A read at an episode boundary must still answer while paused: the
            # heartbeat stands for the state the bridge has already sent.
            standing = client.read_state()
            assert standing.sequence == second.observation_sequence
    finally:
        client.close()
        peer.close()


def test_closing_forgets_the_sequence_so_a_reconnect_can_start_over() -> None:
    """The sequence belongs to the connection, not to the client.

    A bridge that restarts begins counting again, so a client still holding the
    old high-water mark would reject the new stream's first observations as not
    newer than a stream that no longer exists.
    """
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(9)))
        assert client.read_state().sequence == 9
        client.close()

        client._socket, peer_again = socket.socketpair()
        client._handshake = decode_handshake(_handshake(), EXPECTED)
        peer_again.sendall(encode_frame(_observation(1)))

        assert client.read_state().sequence == 1
    finally:
        client.close()
        peer.close()


def test_a_purchase_does_not_resume_a_paused_stream() -> None:
    """Only an advance or a lifecycle command decides whether the world is paused.

    A purchase presses an upgrade button; it never unpauses anything. A double
    that resumed the stream on one would hide the very rejections the pause was
    introduced to prevent.
    """
    client, peer = _slow_bridge_client()
    try:
        with _IdleStreamBridge(peer, holds_the_sequence_while_paused=True):
            state = client.read_state()
            advance = client.send_command(_advance("adv-1", state.sequence))
            purchase = client.send_command(
                {
                    "type": "command", "protocol_version": 2, "request_id": "buy-1",
                    "expected_observation_sequence": advance.observation_sequence,
                    "kind": "buy_upgrade", "family": "attack", "index": 0,
                }
            )
            assert purchase.outcome.value == "confirmed"

            time.sleep(0.4)
            after = client.send_command(_advance("adv-2", purchase.observation_sequence))

            assert after.outcome.value == "confirmed"
    finally:
        client.close()
        peer.close()


def test_a_run_that_ended_under_an_advance_keeps_streaming() -> None:
    """The episode boundary needs fresh state, and that run was never paused."""
    client, peer = _slow_bridge_client()
    try:
        with _IdleStreamBridge(
            peer, holds_the_sequence_while_paused=True, run_ends_under_advance=True
        ):
            state = client.read_state()
            ended = client.send_command(_advance("adv-1", state.sequence))

            time.sleep(0.2)
            boundary = client.read_state()

            assert boundary.terminal
            assert boundary.sequence > ended.observation_sequence, (
                "a run that ended under an advance is never paused, so its screens "
                "keep changing and its state must keep streaming"
            )
    finally:
        client.close()
        peer.close()


def test_a_run_that_died_inside_the_pause_settle_window_keeps_streaming() -> None:
    """The boundary deadlock: `Pause` was pressed, and then the tower died.

    The last advance before a death always sits in the settle window, because a
    health change is what ends the frame loop. The bridge used to report the
    pause it had intended rather than the state it settled on, so it withheld
    every observation after that terminal one and the host could never see the
    world move again: roughly one device episode boundary in seven died here.
    """
    client, peer = _slow_bridge_client()
    try:
        with _IdleStreamBridge(
            peer, holds_the_sequence_while_paused=True, run_ends_in_pause_settle=True
        ):
            state = client.read_state()
            ended = client.send_command(_advance("adv-1", state.sequence))
            assert ended.outcome.value == "confirmed"

            time.sleep(0.2)
            boundary = client.read_state()

            assert boundary.terminal
            assert boundary.sequence > ended.observation_sequence, (
                "a run that ended while its pause was landing is not a world "
                "standing still, so the sequence must not be held"
            )
    finally:
        client.close()
        peer.close()


def test_the_episode_boundary_completes_after_a_death_under_an_advance() -> None:
    """The whole boundary, over the stream rule that deadlocked it.

    The previous episode ended terminally while its advance was pausing. The
    adapter must then see a fresh terminal reading, close the finished run and
    press the game's own start control, and see the new run - which is only
    possible if the bridge kept streaming. With the sequence held,
    `begin_episode` polled a cached terminal reading until `RunPortError` killed
    the run.
    """
    client, peer = _slow_bridge_client()
    try:
        with _IdleStreamBridge(
            peer, holds_the_sequence_while_paused=True, run_ends_in_pause_settle=True
        ):
            adapter = InstrumentedRunAdapter(client=client, episode_start_timeout=10.0)
            state = client.read_state()
            adapter.advance_until_event(
                expected_sequence=state.sequence,
                budget_game_ms=2000,
                frame_game_ms=1000 / 60,
                health_change_fraction=0.05,
            )

            adapter.begin_episode()

            assert not client.read_state().terminal
    finally:
        client.close()
        peer.close()


def test_a_stream_that_ticked_on_while_paused_rejected_that_decision() -> None:
    """Why the bridge holds the sequence: the failure this replaced.

    With the idle tick emitting fresh state through the pause, every decision
    slower than one interval bound a sequence the bridge had already left behind.
    """
    client, peer = _slow_bridge_client()
    try:
        with _IdleStreamBridge(peer, holds_the_sequence_while_paused=False):
            state = client.read_state()
            first = client.send_command(_advance("adv-1", state.sequence))
            assert first.outcome.value == "confirmed"
            time.sleep(0.4)
            second = client.send_command(_advance("adv-2", first.observation_sequence))

            assert second.outcome.value == "rejected"
            assert second.reason == "stale_or_duplicate"
    finally:
        client.close()
        peer.close()


def _idle_tolerant_client(
    *, heartbeat_timeout: float, read_timeout: float = 2.0
) -> tuple[InstrumentedBridgeClient, socket.socket]:
    client_socket, peer_socket = socket.socketpair()
    client = InstrumentedBridgeClient(
        "127.0.0.1",
        47651,
        expected_compatibility=EXPECTED,
        read_timeout=read_timeout,
        heartbeat_timeout=heartbeat_timeout,
    )
    client._socket = client_socket
    client._handshake = decode_handshake(_handshake(), EXPECTED)
    return client, peer_socket


def test_a_client_left_idle_against_a_live_bridge_does_not_expire() -> None:
    """Liveness is the bridge's silence, never the host's inattention.

    A fleet connects each actor's client as its own instance comes up and then
    leaves it unread while the remaining instances cold-boot. Measured from the
    client's own last read, that idleness looked exactly like a dead bridge and
    killed two actors of four on their first episode - while the bridge's frames
    were sitting unread in the socket the whole time.
    """
    client, peer = _idle_tolerant_client(heartbeat_timeout=0.05)
    try:
        with _IdleStreamBridge(peer, holds_the_sequence_while_paused=True, idle_interval=0.01):
            # What a bring-up leaves behind: a client that has heard from its
            # bridge once and is then left alone while the fleet comes up.
            client.read_state()
            # Ten times the liveness window, and a stream's worth of frames
            # waiting in the socket by the end of it.
            time.sleep(0.5)

            state = client.read_state()

            # The read caught up to the bridge's present rather than answering
            # from the backlog: a command bound to what it returned is accepted,
            # which is the only thing that makes the state usable.
            accepted = client.send_command(_advance("adv-after-idle", state.sequence))
            assert accepted.outcome.value == "confirmed"
    finally:
        client.close()
        peer.close()


def test_a_bridge_that_has_gone_silent_still_expires_promptly() -> None:
    """The deadline must still catch a bridge that has genuinely stopped.

    Nothing is served on the peer, so the stream is silent from the first read:
    the client waits one liveness window and no longer, well inside the much
    larger read timeout a device run is configured with.
    """
    client, peer = _idle_tolerant_client(heartbeat_timeout=0.2, read_timeout=10.0)
    try:
        started = time.monotonic()
        with pytest.raises(BridgeTimeoutError, match="liveness expired"):
            client.read_state()
        elapsed = time.monotonic() - started

        assert 0.2 <= elapsed < 2.0, "a dead bridge must not be waited out for the read timeout"
    finally:
        client.close()
        peer.close()


def test_the_final_evaluation_survives_a_client_left_idle_by_the_rest_of_the_fleet() -> None:
    """The run's headline number is taken on instance 0 after the fleet stops.

    By then that client has been idle for as long as the slowest actor took to
    finish, which killed the pre-registered evaluation of a diagnostic run
    outright. The whole path is exercised here - a real client, a
    real adapter and the real evaluator - across an idle gap several times the
    liveness window.
    """

    class FirstAllowed:
        def initial_state(self) -> None:
            return None

        def act(
            self, features: object, state: None, *, epsilon: float
        ) -> tuple[int, None]:
            mask = features.mask  # type: ignore[attr-defined]
            return next(index for index, allowed in enumerate(mask) if allowed), None

    client, peer = _idle_tolerant_client(heartbeat_timeout=0.05, read_timeout=5.0)
    try:
        with _IdleStreamBridge(
            peer,
            holds_the_sequence_while_paused=True,
            idle_interval=0.01,
            run_ends_under_advance=True,
            full_inventory=True,
        ):
            environment = InstrumentedRunEnvironment(
                port=InstrumentedRunAdapter(client=client, episode_start_timeout=10.0),
                builder=RunStateBuilder(profile_id=EXPECTED.profile_id),
                cadence=CadenceConfig(frame_game_ms=100.0, max_quiet_game_ms=2000),
            )
            # This instance's own last collection episode, and then the fleet
            # finishing without it.
            client.read_state()
            time.sleep(0.4)

            report = evaluate(
                environment, FirstAllowed(), episodes=2, profile_id=EXPECTED.profile_id
            )

            assert report.valid_episodes == 2
    finally:
        client.close()
        peer.close()


def _unlock_state_frame(*, attack: int, defense: int, utility: int, wrote: bool) -> dict:
    """The frame the diagnostics bridge sends before the state and the result."""
    return {
        "type": "unlock_state",
        "protocol_version": 2,
        "wrote": wrote,
        "families": [
            {"family": "attack", "length": 20, "true_count": attack},
            {"family": "defense", "length": 12, "true_count": defense},
            {"family": "utility", "length": 14, "true_count": utility},
        ],
    }


def test_the_unlock_state_decodes_as_a_length_and_a_true_count_per_family() -> None:
    wrote, families = decode_unlock_state(
        _unlock_state_frame(attack=3, defense=0, utility=14, wrote=False)
    )

    assert wrote is False
    assert [family.family for family in families] == ["attack", "defense", "utility"]
    assert families[0].length == 20 and families[0].true_count == 3
    assert families[2].true_count == families[2].length

    # More trues than slots is not a state the arrays can be in, so it is a
    # protocol error rather than something to report to the operator.
    overflowing = _unlock_state_frame(attack=3, defense=0, utility=14, wrote=False)
    overflowing["families"][0]["true_count"] = 21
    with pytest.raises(BridgeProtocolError, match="out of bounds"):
        decode_unlock_state(overflowing)

    # A report that names one family twice has not reported the other, and a
    # trial that read it would be comparing counts it does not have.
    duplicated = _unlock_state_frame(attack=3, defense=0, utility=14, wrote=False)
    duplicated["families"][1]["family"] = "attack"
    with pytest.raises(BridgeProtocolError, match="duplicate unlock family state"):
        decode_unlock_state(duplicated)

    missing = _unlock_state_frame(attack=3, defense=0, utility=14, wrote=False)
    del missing["families"][2]
    with pytest.raises(BridgeProtocolError, match="every upgrade family"):
        decode_unlock_state(missing)


def test_the_unlock_commands_carry_only_the_sequence_they_bind() -> None:
    for kind in ("unlock_state", "unlock_all_upgrades"):
        command = decode_command(
            {
                "type": "command",
                "protocol_version": 2,
                "request_id": "unlock-1",
                "expected_observation_sequence": 4,
                "kind": kind,
            }
        )

        assert command.kind == kind and command.family is None and command.index is None

        with pytest.raises(BridgeProtocolError, match="must not contain an upgrade target"):
            decode_command(
                {
                    "type": "command",
                    "protocol_version": 2,
                    "request_id": "unlock-1",
                    "expected_observation_sequence": 4,
                    "kind": kind,
                    "family": "attack",
                    "index": 0,
                }
            )


def test_writing_every_unlock_reports_what_the_arrays_then_hold() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        assert client.read_observation().sequence == 1
        requests: list[dict[str, object]] = []

        def bridge() -> None:
            """Answer the way the diagnostics bridge does: report, state, result."""
            requests.append(read_frame(peer, timeout=2.0))
            peer.sendall(
                encode_frame(_unlock_state_frame(attack=20, defense=12, utility=14, wrote=True))
            )
            peer.sendall(encode_frame(_observation(2)))
            peer.sendall(
                encode_frame(
                    {
                        "type": "command_result", "protocol_version": 2,
                        "request_id": str(requests[0]["request_id"]),
                        "outcome": "confirmed", "reason": "unlock_all_applied",
                        "observation_sequence": 2,
                    }
                )
            )

        responder = threading.Thread(target=bridge)
        responder.start()
        families = client.unlock_all_upgrades(expected_sequence=1)
        responder.join(timeout=5.0)

        assert requests[0]["kind"] == "unlock_all_upgrades"
        # Every slot in every family, which is the whole point of the write.
        assert all(family.true_count == family.length for family in families)
    finally:
        client.close()
        peer.close()


def test_an_unlock_read_rejected_for_a_stale_sequence_is_an_error_not_an_empty_report() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        assert client.read_observation().sequence == 1
        requests: list[dict[str, object]] = []

        def bridge() -> None:
            """A diagnostics bridge that knows the command but refuses this one.

            `stale_or_duplicate` is what a superseded sequence or a repeated
            request id earns. It is not what a production bridge answers - that
            one cannot parse the kind at all, which the next test covers.
            """
            requests.append(read_frame(peer, timeout=2.0))
            peer.sendall(encode_frame(_observation(2)))
            peer.sendall(
                encode_frame(
                    {
                        "type": "command_result", "protocol_version": 2,
                        "request_id": str(requests[0]["request_id"]),
                        "outcome": "rejected", "reason": "stale_or_duplicate",
                        "observation_sequence": 2,
                    }
                )
            )

        responder = threading.Thread(target=bridge)
        responder.start()
        with pytest.raises(BridgeProtocolError, match="no unlock state"):
            client.read_unlock_state(expected_sequence=1)
        responder.join(timeout=5.0)

        assert requests[0]["kind"] == "unlock_state"
    finally:
        client.close()
        peer.close()


def test_a_production_bridge_cannot_parse_an_unlock_command_and_the_client_gives_up() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        assert client.read_observation().sequence == 1

        def bridge() -> None:
            """What the production build actually does: no such kind exists.

            Its parser reads the canonical encoding at fixed offsets and knows
            nothing of `unlock_state`, so the frame fails to parse, the bridge
            answers `protocol_error` and drops the connection. There is no
            command result at all.
            """
            read_frame(peer, timeout=2.0)
            peer.sendall(
                encode_frame(
                    {
                        "type": "error", "protocol_version": 2,
                        "code": "protocol_error", "message": "malformed command",
                    }
                )
            )

        responder = threading.Thread(target=bridge)
        responder.start()
        with pytest.raises(BridgeCompatibilityError, match="protocol_error"):
            client.read_unlock_state(expected_sequence=1)
        responder.join(timeout=5.0)

        # The connection is gone, not merely unhappy: a bridge that dropped the
        # socket cannot be asked anything else, and the sequence belongs to the
        # connection, so a reconnect must start counting again.
        assert client._socket is None
        assert client._last_observation_sequence == 0
    finally:
        client.close()
        peer.close()


def test_an_unlock_report_that_contradicts_the_command_it_answers_is_refused() -> None:
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        assert client.read_observation().sequence == 1
        requests: list[dict[str, object]] = []

        def bridge() -> None:
            """A read command answered by a report claiming it wrote."""
            requests.append(read_frame(peer, timeout=2.0))
            peer.sendall(
                encode_frame(_unlock_state_frame(attack=20, defense=12, utility=14, wrote=True))
            )
            peer.sendall(encode_frame(_observation(2)))
            peer.sendall(
                encode_frame(
                    {
                        "type": "command_result", "protocol_version": 2,
                        "request_id": str(requests[0]["request_id"]),
                        "outcome": "confirmed", "reason": "unlock_state_reported",
                        "observation_sequence": 2,
                    }
                )
            )

        responder = threading.Thread(target=bridge)
        responder.start()
        # The two ends disagree about whether the game's state just changed,
        # which is never something to reconcile quietly.
        with pytest.raises(BridgeProtocolError, match="does not match the unlock_state command"):
            client.read_unlock_state(expected_sequence=1)
        responder.join(timeout=5.0)
    finally:
        client.close()
        peer.close()
