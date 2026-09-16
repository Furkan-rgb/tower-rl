"""Strict client for the private instrumented-training bridge.

This module owns only the bounded local socket protocol: framing, compatibility,
stream ordering, and the semantic command round trip.  It deliberately does not
translate bridge data into domain observations or decide game actions.
"""

from __future__ import annotations

import json
import math
import select
import socket
import struct
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 1
MIN_STEP_GAME_MS = 10
MAX_STEP_GAME_MS = 5000
MIN_REQUESTED_SPEED = 0.5
MAX_REQUESTED_SPEED = 10.0
COMMAND_CAPABILITY = "semantic-v2"
LIFECYCLE_ACTIONS = frozenset(
    {
        "start_round",
        "retry",
        "go_home",
        "enable_auto_restart",
        "speed_max",
        "speed_down",
        "pause",
        "unpause",
    }
)
DEFAULT_MAX_FRAME_SIZE = 65_536
DEFAULT_MAX_UPGRADE_ENTRIES = 192
MAX_DRAINED_OBSERVATIONS = 64


class InstrumentedBridgeError(RuntimeError):
    """Base error for a bridge stream that must not be trusted."""


class BridgeProtocolError(InstrumentedBridgeError):
    """The peer sent an invalid or unexpected protocol message."""


class BridgeCompatibilityError(InstrumentedBridgeError):
    """The bridge does not match the selected private training profile."""


class BridgeDisconnectedError(InstrumentedBridgeError):
    """The peer closed the stream before a complete message arrived."""


class BridgeTimeoutError(InstrumentedBridgeError):
    """A bounded bridge read or heartbeat deadline expired."""


class BridgeStaleObservationError(InstrumentedBridgeError):
    """An observation or heartbeat moved backwards or repeated a sequence."""


class BridgeRunUnavailableError(InstrumentedBridgeError):
    """No run is initialized, so exact run state does not exist yet."""


class CommandOutcome(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class BridgeCompatibility:
    """The exact public compatibility record required by a host profile."""

    package_version: str
    package_version_code: int
    official_signer_sha256: str
    original_libunity_sha256: str
    libil2cpp_sha256: str
    unity_version: str
    il2cpp_metadata_version: int
    bridge_version: str
    profile_id: str

    def __post_init__(self) -> None:
        if not self.package_version or self.package_version_code < 1:
            raise ValueError("package version and positive package version code are required")
        if not self.unity_version or not self.bridge_version or not self.profile_id:
            raise ValueError("unity version, bridge version, and profile id are required")
        if self.il2cpp_metadata_version < 1:
            raise ValueError("il2cpp metadata version must be positive")
        for name, value in (
            ("official_signer_sha256", self.official_signer_sha256),
            ("original_libunity_sha256", self.original_libunity_sha256),
            ("libil2cpp_sha256", self.libil2cpp_sha256),
        ):
            _validate_lower_sha256(name, value)


@dataclass(frozen=True)
class BridgeHandshake:
    protocol_version: int
    bridge_version: str
    compatibility: BridgeCompatibility
    game_speed: float


@dataclass(frozen=True)
class UpgradeInventoryEntry:
    """One read-only in-run upgrade entry reported by the native bridge."""

    family: str
    index: int
    cost: float
    level: int
    unlocked: bool
    tier_unlocked: bool
    maxed: bool


@dataclass(frozen=True)
class BridgeObservation:
    """A raw exact-state snapshot; normalization belongs above this adapter."""

    sequence: int
    lifecycle: str
    wave: int
    cash: float
    health: float
    max_health: float
    terminal: bool
    round_active: bool
    game_speed: float
    upgrades: tuple[UpgradeInventoryEntry, ...]


@dataclass(frozen=True)
class BridgeRunUnavailable:
    """The game holds no initialized run; only a lifecycle command applies."""

    sequence: int
    reason: str


@dataclass(frozen=True)
class BridgeCommand:
    request_id: str
    expected_observation_sequence: int
    kind: str
    family: str | None = None
    index: int | None = None
    action: str | None = None
    value: float | None = None
    game_ms: int | None = None


@dataclass(frozen=True)
class BridgeCommandResult:
    request_id: str
    outcome: CommandOutcome
    reason: str
    observation_sequence: int


def encode_frame(
    message: Mapping[str, object], *, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE
) -> bytes:
    """Encode one JSON message with an unsigned big-endian length prefix."""
    _validate_max_frame_size(max_frame_size)
    try:
        payload = json.dumps(
            message, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BridgeProtocolError(f"message cannot be encoded as JSON: {error}") from error
    if not payload or len(payload) > max_frame_size:
        raise BridgeProtocolError(f"frame payload size {len(payload)} is outside allowed bounds")
    return struct.pack("!I", len(payload)) + payload


def read_frame(
    stream: socket.socket, *, timeout: float, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE
) -> dict[str, Any]:
    """Read exactly one bounded JSON object from an already-connected socket."""
    _validate_max_frame_size(max_frame_size)
    header = _read_exact(stream, 4, timeout)
    (payload_size,) = struct.unpack("!I", header)
    if payload_size == 0 or payload_size > max_frame_size:
        raise BridgeProtocolError(f"frame payload size {payload_size} is outside allowed bounds")
    payload = _read_exact(stream, payload_size, timeout)
    try:
        decoded = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BridgeProtocolError(f"invalid JSON frame: {error}") from error
    if not isinstance(decoded, dict):
        raise BridgeProtocolError("frame root must be a JSON object")
    return decoded


def decode_handshake(
    message: Mapping[str, Any], expected: BridgeCompatibility
) -> BridgeHandshake:
    """Validate the first bridge message and require exact compatibility."""
    _require_message_type(message, "handshake")
    protocol_version = _int(message, "protocol_version", minimum=1)
    if protocol_version != PROTOCOL_VERSION:
        raise BridgeCompatibilityError(
            f"protocol version mismatch: expected {PROTOCOL_VERSION}, got {protocol_version}"
        )
    bridge_version = _string(message, "bridge_version")
    if (
        message.get("mode") != "instrumented_training"
        or message.get("command_capability") != COMMAND_CAPABILITY
    ):
        raise BridgeCompatibilityError(
            f"bridge must advertise instrumented training and {COMMAND_CAPABILITY} commands"
        )
    compatibility_value = message.get("compatibility")
    if not isinstance(compatibility_value, Mapping):
        raise BridgeProtocolError("handshake compatibility must be an object")
    try:
        compatibility = BridgeCompatibility(
            package_version=_string(compatibility_value, "package_version"),
            package_version_code=_int(compatibility_value, "package_version_code", minimum=1),
            official_signer_sha256=_string(compatibility_value, "official_signer_sha256"),
            original_libunity_sha256=_string(compatibility_value, "original_libunity_sha256"),
            libil2cpp_sha256=_string(compatibility_value, "libil2cpp_sha256"),
            unity_version=_string(compatibility_value, "unity_version"),
            il2cpp_metadata_version=_int(compatibility_value, "il2cpp_metadata_version", minimum=1),
            bridge_version=bridge_version,
            profile_id=_string(compatibility_value, "profile_id"),
        )
    except ValueError as error:
        raise BridgeProtocolError(f"invalid handshake compatibility: {error}") from error
    if compatibility != expected:
        raise BridgeCompatibilityError(
            f"compatibility mismatch: expected {expected!r}, got {compatibility!r}"
        )
    return BridgeHandshake(
        protocol_version=protocol_version,
        bridge_version=bridge_version,
        compatibility=compatibility,
        game_speed=_finite_number(message, "game_speed"),
    )


def decode_observation(
    message: Mapping[str, Any], *, max_upgrade_entries: int = DEFAULT_MAX_UPGRADE_ENTRIES
) -> BridgeObservation:
    """Decode a complete raw snapshot and reject invalid inventory shapes."""
    _require_message_type(message, "observation")
    if max_upgrade_entries < 1:
        raise ValueError("max_upgrade_entries must be positive")
    upgrades_value = message.get("upgrades")
    if not isinstance(upgrades_value, list) or len(upgrades_value) > max_upgrade_entries:
        raise BridgeProtocolError("upgrade inventory is missing, malformed, or exceeds its bound")
    entries: list[UpgradeInventoryEntry] = []
    identities: set[tuple[str, int]] = set()
    for value in upgrades_value:
        if not isinstance(value, Mapping):
            raise BridgeProtocolError("upgrade inventory entry must be an object")
        entry = UpgradeInventoryEntry(
            family=_upgrade_family(value),
            index=_int(value, "index", minimum=0),
            cost=_finite_number(value, "cost"),
            level=_int(value, "level", minimum=0),
            unlocked=_bool(value, "unlocked"),
            tier_unlocked=_bool(value, "tier_unlocked"),
            maxed=_bool(value, "maxed"),
        )
        identity = (entry.family, entry.index)
        if identity in identities:
            raise BridgeProtocolError(f"duplicate upgrade inventory entry: {identity!r}")
        identities.add(identity)
        entries.append(entry)
    lifecycle = _string(message, "lifecycle")
    if lifecycle not in {"idle", "active", "terminal", "invalid"}:
        raise BridgeProtocolError(f"unsupported lifecycle: {lifecycle!r}")
    return BridgeObservation(
        sequence=_int(message, "sequence", minimum=1),
        lifecycle=lifecycle,
        wave=_int(message, "wave", minimum=0),
        cash=_finite_number(message, "cash"),
        health=_finite_number(message, "health"),
        max_health=_finite_number(message, "max_health"),
        terminal=_bool(message, "terminal"),
        round_active=_bool(message, "round_active"),
        game_speed=_finite_number(message, "game_speed"),
        upgrades=tuple(entries),
    )


def decode_command(message: Mapping[str, Any]) -> BridgeCommand:
    _require_message_type(message, "command")
    if _int(message, "protocol_version", minimum=1) != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported command protocol version")
    request_id = _string(message, "request_id")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if len(request_id) > 64 or any(character not in allowed for character in request_id):
        raise BridgeProtocolError("request_id must be bounded ASCII")
    kind = _string(message, "kind")
    family = message.get("family")
    index = message.get("index")
    action = message.get("action")
    if kind in {"wait", "lifecycle", "set_speed", "step"} and (
        family is not None or index is not None
    ):
        raise BridgeProtocolError(f"{kind} command must not contain an upgrade target")
    if kind != "lifecycle" and action is not None:
        raise BridgeProtocolError("only a lifecycle command carries an action")
    if kind == "wait":
        return BridgeCommand(
            request_id, _int(message, "expected_observation_sequence", minimum=1), kind
        )
    if kind == "step":
        game_ms = _int(message, "game_ms", minimum=MIN_STEP_GAME_MS)
        if game_ms > MAX_STEP_GAME_MS:
            raise BridgeProtocolError("step game time is outside the allowed range")
        return BridgeCommand(
            request_id,
            _int(message, "expected_observation_sequence", minimum=1),
            kind,
            game_ms=game_ms,
        )
    if kind == "set_speed":
        value = _finite_number(message, "value")
        if not MIN_REQUESTED_SPEED <= value <= MAX_REQUESTED_SPEED:
            raise BridgeProtocolError("requested speed is outside the allowed training range")
        return BridgeCommand(
            request_id,
            _int(message, "expected_observation_sequence", minimum=1),
            kind,
            value=value,
        )
    if kind == "lifecycle":
        if not isinstance(action, str) or action not in LIFECYCLE_ACTIONS:
            raise BridgeProtocolError("unsupported lifecycle action")
        return BridgeCommand(
            request_id,
            _int(message, "expected_observation_sequence", minimum=1),
            kind,
            action=action,
        )
    if kind != "buy_upgrade" or not isinstance(family, str):
        raise BridgeProtocolError("unsupported command kind")
    return BridgeCommand(
        request_id,
        _int(message, "expected_observation_sequence", minimum=1),
        kind,
        _upgrade_family({"family": family}),
        _int(message, "index", minimum=0),
    )


def decode_command_result(message: Mapping[str, Any]) -> BridgeCommandResult:
    _require_message_type(message, "command_result")
    if _int(message, "protocol_version", minimum=1) != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported command-result protocol version")
    try:
        outcome = CommandOutcome(_string(message, "outcome"))
    except ValueError as error:
        raise BridgeProtocolError("unsupported command outcome") from error
    return BridgeCommandResult(
        _string(message, "request_id"), outcome, _string(message, "reason"),
        _int(message, "observation_sequence", minimum=1),
    )


class InstrumentedBridgeClient:
    """One bounded host connection to a read-only loopback bridge.

    The caller owns quarantining an actor after any exception from this class.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        expected_compatibility: BridgeCompatibility,
        connect_timeout: float = 2.0,
        read_timeout: float = 2.0,
        heartbeat_timeout: float = 5.0,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
        max_upgrade_entries: int = DEFAULT_MAX_UPGRADE_ENTRIES,
    ) -> None:
        if not host or not 0 < port < 65_536:
            raise ValueError("host and TCP port must be valid")
        if connect_timeout <= 0 or read_timeout <= 0 or heartbeat_timeout <= 0:
            raise ValueError("all bridge timeouts must be positive")
        _validate_max_frame_size(max_frame_size)
        if max_upgrade_entries < 1:
            raise ValueError("max_upgrade_entries must be positive")
        self.host = host
        self.port = port
        self.expected_compatibility = expected_compatibility
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.heartbeat_timeout = heartbeat_timeout
        self.max_frame_size = max_frame_size
        self.max_upgrade_entries = max_upgrade_entries
        self._socket: socket.socket | None = None
        self._handshake: BridgeHandshake | None = None
        self._last_observation_sequence = 0
        self._last_heartbeat_at: float | None = None

    @property
    def handshake(self) -> BridgeHandshake:
        if self._handshake is None:
            raise BridgeDisconnectedError("bridge is not connected")
        return self._handshake

    def connect(self) -> BridgeHandshake:
        """Connect and accept only a valid initial handshake."""
        if self._socket is not None:
            raise BridgeProtocolError("bridge is already connected")
        try:
            self._socket = socket.create_connection((self.host, self.port), self.connect_timeout)
            message = self._read_message(self.read_timeout)
            self._handshake = decode_handshake(message, self.expected_compatibility)
            self._last_heartbeat_at = time.monotonic()
            return self._handshake
        except TimeoutError as error:
            self.close()
            raise BridgeTimeoutError(f"timed out connecting to bridge: {error}") from error
        except OSError as error:
            self.close()
            raise BridgeDisconnectedError(f"cannot connect to bridge: {error}") from error
        except InstrumentedBridgeError:
            self.close()
            raise

    def read_observation(self) -> BridgeObservation:
        """Return the freshest exact run state, or fail when no run exists."""
        state = self.read_state()
        if isinstance(state, BridgeRunUnavailable):
            raise BridgeRunUnavailableError(f"no exact run state: {state.reason}")
        return state

    def read_state(self) -> BridgeObservation | BridgeRunUnavailable:
        """Return the freshest state the bridge has already sent.

        The bridge streams state at a fixed cadence. A caller that consumed one
        queued frame per decision would fall progressively behind and bind its
        commands to an already-superseded sequence, so buffered frames are
        drained and only the newest state is returned. Between episodes the game
        holds no initialized run, which is reported as its own state rather than
        as invented run values.
        """
        try:
            state = self._read_state()
            for _ in range(MAX_DRAINED_OBSERVATIONS):
                if not self._has_buffered_frame():
                    break
                buffered = self._consume_message(self._read_message(self.read_timeout))
                if buffered is not None:
                    state = buffered
            return state
        except InstrumentedBridgeError:
            self.close()
            raise

    def _has_buffered_frame(self) -> bool:
        if self._socket is None:
            return False
        readable, _, _ = select.select([self._socket], [], [], 0)
        return bool(readable)

    def _read_state(self) -> BridgeObservation | BridgeRunUnavailable:
        if self._socket is None or self._handshake is None:
            raise BridgeDisconnectedError("bridge is not connected")
        self._check_heartbeat()
        deadline = time.monotonic() + self.read_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeTimeoutError("timed out waiting for bridge state")
            state = self._consume_message(self._read_message(remaining))
            if state is not None:
                return state

    def _consume_message(
        self, message: Mapping[str, Any]
    ) -> BridgeObservation | BridgeRunUnavailable | None:
        """Apply one inbound stream message, returning bridge state when it is one."""
        message_type = message.get("type")
        if message_type == "observation":
            observation = decode_observation(message, max_upgrade_entries=self.max_upgrade_entries)
            self._advance_sequence(observation.sequence)
            self._check_heartbeat()
            return observation
        if message_type == "run_unavailable":
            unavailable = BridgeRunUnavailable(
                _int(message, "sequence", minimum=1), _string(message, "reason")
            )
            self._advance_sequence(unavailable.sequence)
            self._check_heartbeat()
            return unavailable
        if message_type == "heartbeat":
            sequence = _int(message, "last_observation_sequence", minimum=0)
            if sequence != self._last_observation_sequence:
                raise BridgeStaleObservationError(
                    "heartbeat sequence does not match the latest observation"
                )
            self._last_heartbeat_at = time.monotonic()
            return None
        if message_type == "error":
            code = _string(message, "code")
            detail = _string(message, "message")
            if code in {"compatibility_error", "protocol_error"}:
                raise BridgeCompatibilityError(f"bridge {code}: {detail}")
            raise BridgeProtocolError(f"bridge {code}: {detail}")
        raise BridgeProtocolError(f"unexpected bridge message type: {message_type!r}")

    def send_command(self, message: Mapping[str, object]) -> BridgeCommandResult:
        """Submit one sequence-bound semantic command and await its bounded result."""
        if self._socket is None or self._handshake is None:
            raise BridgeDisconnectedError("bridge is not connected")
        command = decode_command(message)
        if command.expected_observation_sequence != self._last_observation_sequence:
            raise BridgeStaleObservationError("command does not bind the latest observation")
        try:
            canonical: dict[str, object] = {
                "type": "command",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": command.request_id,
                "expected_observation_sequence": command.expected_observation_sequence,
                "kind": command.kind,
            }
            if command.kind == "buy_upgrade":
                canonical["family"] = command.family
                canonical["index"] = command.index
            elif command.kind == "lifecycle":
                canonical["action"] = command.action
            elif command.kind == "set_speed":
                canonical["value"] = command.value
            elif command.kind == "step":
                canonical["game_ms"] = command.game_ms
            self._write_message(canonical)
            deadline = time.monotonic() + self.read_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeTimeoutError("timed out waiting for command result")
                message_in_flight = self._read_message(remaining)
                if message_in_flight.get("type") == "command_result":
                    decoded = decode_command_result(message_in_flight)
                    if decoded.request_id != command.request_id:
                        raise BridgeProtocolError("command result request id mismatch")
                    return decoded
                self._consume_message(message_in_flight)
        except InstrumentedBridgeError:
            self.close()
            raise

    def close(self) -> None:
        """Close the local stream without leaking a socket descriptor."""
        stream, self._socket = self._socket, None
        self._handshake = None
        self._last_heartbeat_at = None
        if stream is not None:
            with suppress(OSError):
                stream.shutdown(socket.SHUT_RDWR)
            stream.close()

    def _read_message(self, timeout: float) -> dict[str, Any]:
        if self._socket is None:
            raise BridgeDisconnectedError("bridge is not connected")
        return read_frame(self._socket, timeout=timeout, max_frame_size=self.max_frame_size)

    def _write_message(self, message: Mapping[str, object]) -> None:
        if self._socket is None:
            raise BridgeDisconnectedError("bridge is not connected")
        payload = encode_frame(message, max_frame_size=self.max_frame_size)
        self._socket.settimeout(self.read_timeout)
        try:
            self._socket.sendall(payload)
        except TimeoutError as error:
            raise BridgeTimeoutError("timed out writing command") from error
        except OSError as error:
            raise BridgeDisconnectedError(f"bridge socket write failed: {error}") from error

    def _advance_sequence(self, sequence: int) -> None:
        if sequence <= self._last_observation_sequence:
            raise BridgeStaleObservationError(
                "state sequence is not newer than the previous state"
            )
        self._last_observation_sequence = sequence

    def _check_heartbeat(self) -> None:
        if (
            self._last_heartbeat_at is not None
            and time.monotonic() - self._last_heartbeat_at > self.heartbeat_timeout
        ):
            raise BridgeTimeoutError("bridge heartbeat expired")


def _read_exact(stream: socket.socket, size: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    remaining = size
    previous_timeout = stream.gettimeout()
    try:
        while remaining:
            read_timeout = deadline - time.monotonic()
            if read_timeout <= 0:
                raise BridgeTimeoutError("timed out while reading bridge frame")
            stream.settimeout(read_timeout)
            try:
                chunk = stream.recv(remaining)
            except TimeoutError as error:
                raise BridgeTimeoutError("timed out while reading bridge frame") from error
            except OSError as error:
                raise BridgeDisconnectedError(f"bridge socket read failed: {error}") from error
            if not chunk:
                raise BridgeDisconnectedError("bridge closed the stream")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        stream.settimeout(previous_timeout)
    return b"".join(chunks)


def _require_message_type(message: Mapping[str, Any], expected: str) -> None:
    if message.get("type") != expected:
        raise BridgeProtocolError(f"expected {expected!r} message")


def _string(message: Mapping[str, Any], name: str) -> str:
    value = message.get(name)
    if not isinstance(value, str) or not value:
        raise BridgeProtocolError(f"{name} must be a non-empty string")
    return value


def _int(message: Mapping[str, Any], name: str, *, minimum: int) -> int:
    value = message.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BridgeProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(message: Mapping[str, Any], name: str) -> float:
    value = message.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise BridgeProtocolError(f"{name} must be a finite number")
    return float(value)


def _bool(message: Mapping[str, Any], name: str) -> bool:
    value = message.get(name)
    if not isinstance(value, bool):
        raise BridgeProtocolError(f"{name} must be a boolean")
    return value


def _upgrade_family(message: Mapping[str, Any]) -> str:
    family = _string(message, "family")
    if family not in {"attack", "defense", "utility"}:
        raise BridgeProtocolError(f"unsupported upgrade family: {family!r}")
    return family


def _validate_max_frame_size(max_frame_size: int) -> None:
    if not 1 <= max_frame_size <= 16 * 1024 * 1024:
        raise ValueError("max_frame_size must be between 1 and 16777216 bytes")


def _validate_lower_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


__all__ = [
    "DEFAULT_MAX_FRAME_SIZE",
    "DEFAULT_MAX_UPGRADE_ENTRIES",
    "PROTOCOL_VERSION",
    "BridgeCommand",
    "BridgeCommandResult",
    "BridgeCompatibility",
    "BridgeCompatibilityError",
    "BridgeDisconnectedError",
    "BridgeHandshake",
    "BridgeObservation",
    "BridgeProtocolError",
    "BridgeStaleObservationError",
    "BridgeTimeoutError",
    "CommandOutcome",
    "InstrumentedBridgeClient",
    "InstrumentedBridgeError",
    "UpgradeInventoryEntry",
    "decode_handshake",
    "decode_command",
    "decode_command_result",
    "decode_observation",
    "encode_frame",
    "read_frame",
]
