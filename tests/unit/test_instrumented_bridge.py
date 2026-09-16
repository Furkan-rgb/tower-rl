from __future__ import annotations

import socket
import struct

import pytest

from tower_rl.infrastructure.instrumented_bridge import (
    BridgeCompatibility,
    BridgeCompatibilityError,
    BridgeDisconnectedError,
    BridgeProtocolError,
    BridgeStaleObservationError,
    BridgeTimeoutError,
    InstrumentedBridgeClient,
    decode_command,
    decode_handshake,
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
        "command_capability": "semantic-v1",
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


def _observation(sequence: int = 1) -> dict[str, object]:
    return {
        "type": "observation",
        "sequence": sequence,
        "lifecycle": "active",
        "wave": 7,
        "cash": 123.5,
        "health": 95.0,
        "max_health": 100.0,
        "terminal": False,
        "round_active": True,
        "upgrades": [
            {
                "family": "attack",
                "index": 0,
                "cost": 5.0,
                "level": 2,
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
    malformed_wait = {
        "type": "command", "protocol_version": 1, "request_id": "a",
        "expected_observation_sequence": 1, "kind": "wait", "index": 0,
    }
    with pytest.raises(BridgeProtocolError, match="upgrade target"):
        decode_command(malformed_wait)
    client, peer = _connected_client()
    try:
        peer.sendall(encode_frame(_observation(1)))
        client.read_observation()
        stale_wait = {
            "type": "command", "protocol_version": 1, "request_id": "a",
            "expected_observation_sequence": 2, "kind": "wait",
        }
        with pytest.raises(BridgeStaleObservationError, match="latest"):
            client.send_command(stale_wait)
        other_result = {
            "type": "command_result", "protocol_version": 1, "request_id": "b",
            "outcome": "confirmed", "reason": "wait_elapsed", "observation_sequence": 1,
        }
        peer.sendall(encode_frame(other_result))
        wait = {
            "type": "command", "protocol_version": 1, "request_id": "a",
            "expected_observation_sequence": 1, "kind": "wait",
        }
        with pytest.raises(BridgeProtocolError, match="request id"):
            client.send_command(wait)
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
