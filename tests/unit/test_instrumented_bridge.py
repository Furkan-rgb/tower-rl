from __future__ import annotations

import socket
import struct

import pytest

from tower_rl.infrastructure.instrumented_bridge import (
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
    encode_frame,
    read_frame,
)

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
        "protocol_version": 1,
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


def _observation(sequence: int = 1, wave: int = 7) -> dict[str, object]:
    return {
        "type": "observation",
        "sequence": sequence,
        "lifecycle": "active",
        "wave": wave,
        "cash": 123.5,
        "health": 95.0,
        "max_health": 100.0,
        "terminal": False,
        "round_active": True,
        "game_speed": 1.5,
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
        decode_handshake(_handshake(protocol_version=2), EXPECTED)

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
        "type": "command", "protocol_version": 1, "request_id": "a",
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
            "type": "command", "protocol_version": 1, "request_id": "a",
            "expected_observation_sequence": 2, "kind": "lifecycle", "action": "pause",
        }
        with pytest.raises(BridgeStaleObservationError, match="latest"):
            client.send_command(stale_lifecycle)
        other_result = {
            "type": "command_result", "protocol_version": 1, "request_id": "b",
            "outcome": "confirmed", "reason": "run_active", "observation_sequence": 1,
        }
        peer.sendall(encode_frame(other_result))
        mine = {
            "type": "command", "protocol_version": 1, "request_id": "a",
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
            "type": "command_result", "protocol_version": 1, "request_id": "buy-1",
            "outcome": "confirmed", "reason": "confirmed_state_change",
            "observation_sequence": 1,
        }
        peer.sendall(encode_frame(result))
        client.send_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "buy-1",
                "expected_observation_sequence": 1, "kind": "buy_upgrade",
                "family": "attack", "index": 3,
            }
        )
        payload = read_frame(peer, timeout=0.1)
        raw = encode_frame(payload)[4:].decode("utf-8")

        prefix = '{"type":"command","protocol_version":1,"request_id":"'
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
            "type": "command", "protocol_version": 1, "request_id": "r",
            "expected_observation_sequence": 2, "kind": "lifecycle", "action": "retry",
        }
    )

    assert lifecycle.kind == "lifecycle"
    assert lifecycle.action == "retry"
    assert lifecycle.family is None and lifecycle.index is None

    with pytest.raises(BridgeProtocolError, match="unsupported lifecycle action"):
        decode_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "r",
                "expected_observation_sequence": 2, "kind": "lifecycle", "action": "buy_gems",
            }
        )
    with pytest.raises(BridgeProtocolError, match="must not contain an upgrade target"):
        decode_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "r",
                "expected_observation_sequence": 2, "kind": "lifecycle",
                "action": "retry", "family": "attack", "index": 0,
            }
        )
    with pytest.raises(BridgeProtocolError, match="only a lifecycle command"):
        decode_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "r",
                "expected_observation_sequence": 2, "kind": "advance", "action": "retry",
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
                    "type": "command_result", "protocol_version": 1, "request_id": "life-1",
                    "outcome": "confirmed", "reason": "run_active", "observation_sequence": 1,
                }
            )
        )
        client.send_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "life-1",
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
                    "type": "command_result", "protocol_version": 1, "request_id": "adv-1",
                    "outcome": "confirmed", "reason": "budget_exhausted",
                    "observation_sequence": 1, "frames": 120, "game_ms": 2000,
                    "play_ms": 1998, "wall_micros": 57_000,
                }
            )
        )
        client.send_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "adv-1",
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
            "type": "command", "protocol_version": 1, "request_id": "a",
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
            "type": "command", "protocol_version": 1, "request_id": "a",
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
                    "type": "command", "protocol_version": 1, "request_id": "s",
                    "expected_observation_sequence": 1, **dead,
                }
            )


def test_a_command_result_carries_what_the_advance_cost() -> None:
    """Frames, both clocks and wall time are how the speed-up is measured at all.

    The native encoder writes these in one fixed order - `frames`, `game_ms`,
    `play_ms`, `wall_micros` - and `game_ms` against `play_ms` is what shows
    whether the game time each frame was told to be worth actually passed.
    """
    result = decode_command_result(
        {
            "type": "command_result", "protocol_version": 1, "request_id": "a",
            "outcome": "confirmed", "reason": "event:wave_changed",
            "observation_sequence": 7, "frames": 120, "game_ms": 2000.0,
            "play_ms": 1998.0, "wall_micros": 57_000,
        }
    )

    assert result.reason == "event:wave_changed"
    assert (result.frames, result.game_ms, result.play_ms, result.wall_micros) == (
        120, 2000.0, 1998.0, 57_000,
    )
    # Nothing bound this result to a state, so it honestly carries none.
    assert result.state is None

    # Commands that burn no game time omit them, and zero is the honest answer.
    bare = decode_command_result(
        {
            "type": "command_result", "protocol_version": 1, "request_id": "b",
            "outcome": "confirmed", "reason": "confirmed_state_change",
            "observation_sequence": 7,
        }
    )

    assert (bare.frames, bare.game_ms, bare.play_ms, bare.wall_micros) == (0, 0.0, 0.0, 0)


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
                    "type": "command_result", "protocol_version": 1, "request_id": "adv-2",
                    "outcome": "confirmed", "reason": "event:wave_changed",
                    "observation_sequence": 2, "frames": 60, "game_ms": 1000,
                    "play_ms": 1000, "wall_micros": 21_000,
                }
            )
        )
        result = client.send_command(
            {
                "type": "command", "protocol_version": 1, "request_id": "adv-2",
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
