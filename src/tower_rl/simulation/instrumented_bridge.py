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
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

#: The one thing this module takes from the domain: which `Main` fields a v2
#: state message must carry. It is a wire contract, not a translation - the
#: values are passed through raw and scaled in `environment/run_state.py` - and
#: importing the list is what keeps a second copy of thirty-seven field names
#: from drifting away from the schema that declares them.
from tower_rl.environment.run_state import LIVE_WIRE_NAMES

#: Bumped to 2 by `observation-v2`: the state message carries the game's live
#: `Main` readings and the bridge answers a `slot_labels` command, so a v1
#: bridge and a v2 host cannot talk to each other and must not try.
PROTOCOL_VERSION = 2
#: Bounds on one `advance` request. The budget is the game time the bridge may
#: burn before handing a decision back even when nothing happened; the frame is
#: how much game time each rendered frame is worth, which is what decouples
#: decision granularity from the wall clock (see solution.md 9.2c).
MIN_ADVANCE_BUDGET_GAME_MS = 10
MAX_ADVANCE_BUDGET_GAME_MS = 10_000
MIN_FRAME_GAME_MS = 1.0
MAX_FRAME_GAME_MS = 250.0
MIN_REQUESTED_SPEED = 0.5
MAX_REQUESTED_SPEED = 64.0
#: The bridge's own wall-clock ceiling on one advance and the settling window it
#: then spends waiting for the pause to land. Both mirror `kAdvanceWallBudgetMicros`
#: and `kPauseSettleMicros` in `native/tower_bridge/tower_bridge.cpp`; they cannot
#: be shared across the language boundary, so they are cross-referenced instead
#: and the client's default read timeout is derived from them rather than guessed.
ADVANCE_WALL_CEILING_SECONDS = 15.0
PAUSE_SETTLE_SECONDS = 0.5
#: A read that gave up before the bridge's own ceiling would report a timeout for
#: an advance that was still going to answer, so the default covers the ceiling,
#: the settle, and a margin for the round trip itself.
DEFAULT_READ_TIMEOUT_SECONDS = ADVANCE_WALL_CEILING_SECONDS + PAUSE_SETTLE_SECONDS + 4.5
COMMAND_CAPABILITY = "semantic-v2"
#: The game's three upgrade families, in the order the bridge reports them.
UPGRADE_FAMILIES = ("attack", "defense", "utility")
#: Commands whose whole payload is the observation sequence they bind.
#: `unlock_state` and `unlock_all_upgrades` are answered by the diagnostics
#: build only (board #54's profile-v2 trial instrument). A production bridge's
#: parser has no such kind at all, so it answers `protocol_error` and drops the
#: connection rather than rejecting a command it understands - which is what
#: keeps a measured run unable to change what the game offers a policy.
SEQUENCE_ONLY_COMMAND_KINDS = frozenset({"slot_labels", "unlock_state", "unlock_all_upgrades"})
LIFECYCLE_ACTIONS = frozenset(
    {
        "start_round",
        "go_home",
        "speed_max",
        "speed_down",
        "pause",
        "unpause",
    }
)
DEFAULT_MAX_FRAME_SIZE = 65_536
DEFAULT_MAX_UPGRADE_ENTRIES = 192


class InstrumentedBridgeError(RuntimeError):
    """Base error for a bridge stream that must not be trusted."""


class BridgeProtocolError(InstrumentedBridgeError):
    """The peer sent an invalid or unexpected protocol message."""


class BridgeCompatibilityError(InstrumentedBridgeError):
    """The bridge does not match the selected private training profile."""


class BridgeDisconnectedError(InstrumentedBridgeError):
    """The peer closed the stream before a complete message arrived."""


class BridgeTimeoutError(InstrumentedBridgeError):
    """A bounded bridge read or liveness deadline expired."""


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
    max_level: int
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
    play_time: float
    upgrades: tuple[UpgradeInventoryEntry, ...]
    #: Every `LIVE_WIRE_NAMES` reading, raw and unscaled, keyed by the game's own
    #: `Main` field name. Scaling is the environment's, not this adapter's.
    live: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class UpgradeSlotLabel:
    """What the player reads on one upgrade row: its name and its description.

    Read once per session by the `slot_labels` command, never per snapshot: the
    arrays are constant for a build. These are for humans - the spectate panel
    and the records a developer reads afterwards - and are deliberately not part
    of the observation tensor, which addresses a slot by its stable index.
    """

    family: str
    index: int
    name: str
    description: str


@dataclass(frozen=True)
class UnlockFamilyState:
    """How much of one family's in-run availability array is true, and how long it is.

    The profile-v2 trial instrument (board #54) asks one question - does a
    bridge write to `upgrade*Unlocked` land, and does it survive - so the answer
    it needs is a count, not a per-slot picture. Reported by the diagnostics
    build only; a production bridge answers neither unlock command.
    """

    family: str
    length: int
    true_count: int


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
    budget_game_ms: int | None = None
    frame_game_ms: float | None = None
    health_change_fraction: float | None = None


@dataclass(frozen=True)
class BridgeCommandResult:
    request_id: str
    outcome: CommandOutcome
    reason: str
    observation_sequence: int
    #: What an `advance` actually cost. Every other command reports zeroes,
    #: because none of them burns game time; that is the honest value, not a
    #: placeholder.
    frames: int = 0
    #: Budget accounting: frames times `frame_game_ms`, the game time the advance
    #: asked for.
    game_ms: float = 0.0
    #: The game's own per-round clock across the same advance. Reported beside
    #: `game_ms` so the 1:1 mapping between them can be checked, not assumed.
    #: `playTime` cannot serve here: it advances at wall rate whatever the game
    #: clock does, so it witnesses the host's speed-up, not the game's time.
    round_ms: float = 0.0
    wall_micros: int = 0
    #: The state message the bridge sent immediately before this result, which
    #: for an `advance` is the settled observation taken after the pause landed.
    #: `None` when the bridge sent no state bound to this result - a rejection
    #: reports the sequence that already stood - or when the run is unavailable,
    #: which the caller reads exactly as it reads an absent run.
    state: BridgeObservation | None = None


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
            max_level=_int(value, "max_level", minimum=0),
            unlocked=_bool(value, "unlocked"),
            tier_unlocked=_bool(value, "tier_unlocked"),
            maxed=_bool(value, "maxed"),
        )
        if entry.max_level and entry.level > entry.max_level:
            raise BridgeProtocolError(
                f"upgrade level exceeds its own maximum: {entry.family}[{entry.index}]"
            )
        identity = (entry.family, entry.index)
        if identity in identities:
            raise BridgeProtocolError(f"duplicate upgrade inventory entry: {identity!r}")
        identities.add(identity)
        entries.append(entry)
    lifecycle = _string(message, "lifecycle")
    if lifecycle not in {"idle", "active", "terminal", "invalid"}:
        raise BridgeProtocolError(f"unsupported lifecycle: {lifecycle!r}")
    live_value = message.get("live")
    if not isinstance(live_value, Mapping):
        raise BridgeProtocolError("state message carries no live readings object")
    # Exactly the declared set, no more and no less: an absent field is a bridge
    # that does not speak this schema, and an extra one is a bridge sending
    # something the host has no transform for. Either way the host must not
    # guess, and a protocol error costs the episode rather than the meaning of
    # every observation after it.
    missing = [name for name in LIVE_WIRE_NAMES if name not in live_value]
    if missing:
        raise BridgeProtocolError(f"state message is missing live readings: {missing}")
    extra = sorted(set(live_value) - set(LIVE_WIRE_NAMES))
    if extra:
        raise BridgeProtocolError(
            f"state message carries live readings this schema cannot place: {extra}"
        )
    live = {name: _finite_number(live_value, name) for name in LIVE_WIRE_NAMES}
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
        play_time=_finite_number(message, "play_time"),
        upgrades=tuple(entries),
        live=live,
    )


def decode_slot_labels(
    message: Mapping[str, Any], *, max_labels: int = DEFAULT_MAX_UPGRADE_ENTRIES
) -> tuple[UpgradeSlotLabel, ...]:
    """Decode the once-per-session upgrade-row names the player reads."""
    _require_message_type(message, "slot_labels")
    if _int(message, "protocol_version", minimum=1) != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported slot-label protocol version")
    labels_value = message.get("labels")
    if not isinstance(labels_value, list) or len(labels_value) > max_labels:
        raise BridgeProtocolError("slot labels are missing, malformed, or exceed their bound")
    labels: list[UpgradeSlotLabel] = []
    seen: set[tuple[str, int]] = set()
    for value in labels_value:
        if not isinstance(value, Mapping):
            raise BridgeProtocolError("slot label must be an object")
        label = UpgradeSlotLabel(
            family=_upgrade_family(value),
            index=_int(value, "index", minimum=0),
            name=_label_text(value, "name"),
            description=_label_text(value, "description"),
        )
        if (label.family, label.index) in seen:
            raise BridgeProtocolError(f"duplicate slot label: {label.family}[{label.index}]")
        seen.add((label.family, label.index))
        labels.append(label)
    return tuple(labels)


def decode_unlock_state(
    message: Mapping[str, Any], *, max_entries: int = DEFAULT_MAX_UPGRADE_ENTRIES
) -> tuple[bool, tuple[UnlockFamilyState, ...]]:
    """Decode whether the bridge wrote, and each array's length and true-count.

    The `wrote` flag comes back so the caller can hold the bridge to the command
    it sent: a report claiming a write for a read command, or the reverse, means
    the two ends disagree about what just happened to the game's state, which is
    not something to reconcile silently.
    """
    _require_message_type(message, "unlock_state")
    if _int(message, "protocol_version", minimum=1) != PROTOCOL_VERSION:
        raise BridgeProtocolError("unsupported unlock-state protocol version")
    families_value = message.get("families")
    if not isinstance(families_value, list) or len(families_value) != len(UPGRADE_FAMILIES):
        raise BridgeProtocolError("unlock state must report every upgrade family exactly once")
    families: list[UnlockFamilyState] = []
    seen: set[str] = set()
    for value in families_value:
        if not isinstance(value, Mapping):
            raise BridgeProtocolError("unlock family state must be an object")
        family = _upgrade_family(value)
        length = _int(value, "length", minimum=0)
        true_count = _int(value, "true_count", minimum=0)
        if length > max_entries or true_count > length:
            raise BridgeProtocolError(f"unlock state for {family} is out of bounds")
        if family in seen:
            raise BridgeProtocolError(f"duplicate unlock family state: {family}")
        seen.add(family)
        families.append(UnlockFamilyState(family=family, length=length, true_count=true_count))
    return _bool(message, "wrote"), tuple(families)


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
    if kind in {"lifecycle", "set_speed", "advance", *SEQUENCE_ONLY_COMMAND_KINDS} and (
        family is not None or index is not None
    ):
        raise BridgeProtocolError(f"{kind} command must not contain an upgrade target")
    if kind != "lifecycle" and action is not None:
        raise BridgeProtocolError("only a lifecycle command carries an action")
    if kind == "advance":
        budget = _int(message, "budget_game_ms", minimum=MIN_ADVANCE_BUDGET_GAME_MS)
        if budget > MAX_ADVANCE_BUDGET_GAME_MS:
            raise BridgeProtocolError("advance budget is outside the allowed range")
        frame = _finite_number(message, "frame_game_ms")
        if not MIN_FRAME_GAME_MS <= frame <= MAX_FRAME_GAME_MS:
            raise BridgeProtocolError("advance frame game time is outside the allowed range")
        fraction = _finite_number(message, "health_change_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise BridgeProtocolError("health change fraction must be a fraction")
        return BridgeCommand(
            request_id,
            _int(message, "expected_observation_sequence", minimum=1),
            kind,
            budget_game_ms=budget,
            frame_game_ms=frame,
            health_change_fraction=fraction,
        )
    if kind in SEQUENCE_ONLY_COMMAND_KINDS:
        # Each carries nothing but the sequence it binds. `slot_labels` is asked
        # once because its answer is the same for the whole build; the two
        # unlock kinds address all three families at once by definition.
        return BridgeCommand(
            request_id,
            _int(message, "expected_observation_sequence", minimum=1),
            kind,
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
        frames=_int(message, "frames", minimum=0) if "frames" in message else 0,
        game_ms=_finite_number(message, "game_ms") if "game_ms" in message else 0.0,
        round_ms=_finite_number(message, "round_ms") if "round_ms" in message else 0.0,
        wall_micros=_int(message, "wall_micros", minimum=0) if "wall_micros" in message else 0,
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
        read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
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
        # The liveness deadline: how long the host waits on a silent stream
        # before calling the bridge dead. It is spent waiting, never counted
        # while the host is busy elsewhere - see `_await_bridge_frame`.
        self.heartbeat_timeout = heartbeat_timeout
        self.max_frame_size = max_frame_size
        self.max_upgrade_entries = max_upgrade_entries
        self._socket: socket.socket | None = None
        self._handshake: BridgeHandshake | None = None
        self._last_observation_sequence = 0
        self._last_state: BridgeObservation | BridgeRunUnavailable | None = None
        self._slot_labels: tuple[UpgradeSlotLabel, ...] = ()
        self._unlock_state: tuple[UnlockFamilyState, ...] = ()
        self._unlock_wrote: bool | None = None

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

        While the world is paused the bridge deliberately streams no new state,
        because a paused world has none: a heartbeat then stands for the state
        already sent, and that is what this returns.
        """
        try:
            return self._drain_to_newest(self._read_state())
        except InstrumentedBridgeError:
            self.close()
            raise

    def _drain_to_newest(
        self, state: BridgeObservation | BridgeRunUnavailable
    ) -> BridgeObservation | BridgeRunUnavailable:
        """Consume every frame already buffered, so the state returned is the present.

        A client that has not read for a while has a whole stream's worth of
        frames waiting for it, and stopping short of the end would leave it
        acting on a sequence the bridge has already left behind - which the
        bridge then refuses as `stale_or_duplicate`. The drain is bounded by the
        read timeout rather than by a frame count, because the backlog is
        whatever the bridge sent while nobody was reading: only a bridge
        producing frames faster than the host can consume them could keep this
        from ending, and that is what the deadline is for.
        """
        deadline = time.monotonic() + self.read_timeout
        while self._has_buffered_frame():
            if time.monotonic() > deadline:
                raise BridgeTimeoutError("the bridge stream never paused long enough to catch up")
            buffered = self._consume_message(self._read_message(self.read_timeout))
            if buffered is not None:
                state = buffered
        return state

    def _has_buffered_frame(self) -> bool:
        if self._socket is None:
            return False
        readable, _, _ = select.select([self._socket], [], [], 0)
        return bool(readable)

    def _read_state(self) -> BridgeObservation | BridgeRunUnavailable:
        if self._socket is None or self._handshake is None:
            raise BridgeDisconnectedError("bridge is not connected")
        deadline = time.monotonic() + self.read_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeTimeoutError("timed out waiting for bridge state")
            self._await_bridge_frame(remaining)
            message = self._read_message(remaining)
            state = self._consume_message(message)
            if state is not None:
                return state
            # A heartbeat says the state the bridge has already sent still stands.
            # While the world is paused that is the whole truth about it, and the
            # bridge sends no further observation, so waiting for one would wait
            # out the read timeout for a world that cannot change.
            if message.get("type") == "heartbeat" and self._last_state is not None:
                return self._last_state

    def _consume_message(
        self, message: Mapping[str, Any]
    ) -> BridgeObservation | BridgeRunUnavailable | None:
        """Apply one inbound stream message, returning bridge state when it is one."""
        message_type = message.get("type")
        if message_type == "observation":
            observation = decode_observation(message, max_upgrade_entries=self.max_upgrade_entries)
            self._advance_sequence(observation.sequence)
            self._last_state = observation
            return observation
        if message_type == "run_unavailable":
            unavailable = BridgeRunUnavailable(
                _int(message, "sequence", minimum=1), _string(message, "reason")
            )
            self._advance_sequence(unavailable.sequence)
            self._last_state = unavailable
            return unavailable
        if message_type == "slot_labels":
            # Session-scoped, so the connection owns them: the arrays are
            # constant for a build and are asked for once, before the first
            # round. They are not state - nothing binds a sequence to them.
            self._slot_labels = decode_slot_labels(message, max_labels=self.max_upgrade_entries)
            return None
        if message_type == "unlock_state":
            # Not state: nothing binds a sequence to it, and it is the answer to
            # the command in flight rather than something the bridge streams.
            self._unlock_wrote, self._unlock_state = decode_unlock_state(
                message, max_entries=self.max_upgrade_entries
            )
            return None
        if message_type == "heartbeat":
            sequence = _int(message, "last_observation_sequence", minimum=0)
            if sequence != self._last_observation_sequence:
                raise BridgeStaleObservationError(
                    "heartbeat sequence does not match the latest observation"
                )
            return None
        if message_type == "error":
            code = _string(message, "code")
            detail = _string(message, "message")
            if code in {"compatibility_error", "protocol_error"}:
                raise BridgeCompatibilityError(f"bridge {code}: {detail}")
            raise BridgeProtocolError(f"bridge {code}: {detail}")
        raise BridgeProtocolError(f"unexpected bridge message type: {message_type!r}")

    def read_slot_labels(self, *, expected_sequence: int) -> tuple[UpgradeSlotLabel, ...]:
        """Ask the game once what each upgrade row is called, and what it does.

        For humans only. The policy addresses a slot by its stable index, so a
        renamed row must not change what a checkpoint means; these labels reach
        the spectate panel and the session records a developer reads, and never
        the observation tensor.
        """
        result = self.send_command(
            {
                "type": "command",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": f"labels-{time.monotonic_ns() % 1_000_000_000}",
                "expected_observation_sequence": expected_sequence,
                "kind": "slot_labels",
            }
        )
        if result.outcome != CommandOutcome.CONFIRMED or not self._slot_labels:
            raise BridgeProtocolError(f"the bridge reported no slot labels: {result.reason}")
        return self._slot_labels

    def read_unlock_state(self, *, expected_sequence: int) -> tuple[UnlockFamilyState, ...]:
        """Ask how much of each family's in-run availability array is true.

        Reads nothing else and writes nothing. Answered by a diagnostics bridge
        only. A production bridge does not reject the command, it cannot parse
        it: its parser has no such kind, so it answers `protocol_error` and
        drops the connection, which surfaces here as a
        `BridgeCompatibilityError` and a closed client.
        """
        return self._unlock_command("unlock_state", expected_sequence=expected_sequence)

    def unlock_all_upgrades(self, *, expected_sequence: int) -> tuple[UnlockFamilyState, ...]:
        """Set every in-run availability flag true, and report what then stands.

        The trial instrument for board #54, and nothing else: it changes what
        the game offers inside the live process, so it exists only in the
        diagnostics build and must never be pointed at the canonical profile.
        The returned counts are read back out of the arrays after the write, so
        a write that did not take reports as one.
        """
        return self._unlock_command("unlock_all_upgrades", expected_sequence=expected_sequence)

    def _unlock_command(
        self, kind: str, *, expected_sequence: int
    ) -> tuple[UnlockFamilyState, ...]:
        self._unlock_state = ()
        self._unlock_wrote = None
        result = self.send_command(
            {
                "type": "command",
                "protocol_version": PROTOCOL_VERSION,
                "request_id": f"unlock-{time.monotonic_ns() % 1_000_000_000}",
                "expected_observation_sequence": expected_sequence,
                "kind": kind,
            }
        )
        if result.outcome != CommandOutcome.CONFIRMED or not self._unlock_state:
            raise BridgeProtocolError(f"the bridge reported no unlock state: {result.reason}")
        if self._unlock_wrote != (kind == "unlock_all_upgrades"):
            raise BridgeProtocolError(
                f"the unlock report does not match the {kind} command it answers"
            )
        return self._unlock_state

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
            elif command.kind == "advance":
                canonical["budget_game_ms"] = command.budget_game_ms
                canonical["frame_game_ms"] = command.frame_game_ms
                canonical["health_change_fraction"] = command.health_change_fraction
            self._write_message(canonical)
            deadline = time.monotonic() + self.read_timeout
            # The bridge emits the state a result describes immediately before the
            # result itself. Keeping it here is what lets one advance cost one
            # round trip: the caller has the settled observation already and does
            # not have to wait for the next free-running tick to see it.
            bound: BridgeObservation | None = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BridgeTimeoutError("timed out waiting for command result")
                message_in_flight = self._read_message(remaining)
                if message_in_flight.get("type") == "command_result":
                    decoded = decode_command_result(message_in_flight)
                    if decoded.request_id != command.request_id:
                        raise BridgeProtocolError("command result request id mismatch")
                    if bound is not None and bound.sequence == decoded.observation_sequence:
                        return replace(decoded, state=bound)
                    return decoded
                in_flight = self._consume_message(message_in_flight)
                if isinstance(in_flight, BridgeObservation):
                    bound = in_flight
        except InstrumentedBridgeError:
            self.close()
            raise

    def close(self) -> None:
        """Close the local stream without leaking a socket descriptor."""
        stream, self._socket = self._socket, None
        self._handshake = None
        self._last_state = None
        # The sequence belongs to the connection, not to the client: a reconnect
        # starts a fresh one, and keeping the old high-water mark would reject
        # the new stream's first observations as not newer.
        self._last_observation_sequence = 0
        if stream is not None:
            with suppress(OSError):
                stream.shutdown(socket.SHUT_RDWR)
            stream.close()

    def _read_message(self, timeout: float) -> dict[str, Any]:
        if self._socket is None:
            raise BridgeDisconnectedError("bridge is not connected")
        # A heartbeat only exists to prove the bridge is alive, and any frame it
        # decodes to is strictly stronger proof, so every inbound frame - an
        # observation, a command result, a heartbeat - is proof of life. Under
        # one command in flight per decision the bridge has no idle moment in
        # which to emit a heartbeat, and requiring one would kill a healthy run.
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

    def _await_bridge_frame(self, timeout: float) -> None:
        """Wait for the bridge to speak, judging its silence rather than the host's.

        Liveness belongs to the bridge: it expires when nothing arrives while
        the host is actually waiting to hear something. Measured instead from
        the host's own last read it reported an idle *client* as a dead bridge,
        which killed the first actors of a fleet - each one connected at its own
        bring-up and then left unread while the remaining instances cold-booted,
        with the bridge's frames sitting in the socket the whole time.
        Anything already buffered is therefore proof of life and answers at once,
        and a bridge that has genuinely stopped is still caught one
        `heartbeat_timeout` after the host first waits on it.
        """
        if self._socket is None:
            raise BridgeDisconnectedError("bridge is not connected")
        wait = min(timeout, self.heartbeat_timeout)
        readable, _, _ = select.select([self._socket], [], [], wait)
        if readable:
            return
        if wait >= self.heartbeat_timeout:
            raise BridgeTimeoutError(
                f"bridge liveness expired: silent for {self.heartbeat_timeout:g}s"
            )
        raise BridgeTimeoutError("timed out waiting for bridge state")


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


#: Enough for the longest upgrade description the game carries, and short enough
#: that sixty of them cannot approach the frame bound.
MAX_LABEL_CHARACTERS = 256


def _label_text(message: Mapping[str, Any], name: str) -> str:
    """One human-facing label, which may legitimately be empty.

    The game's own name arrays are twenty wide per family with an empty tail -
    the slots no tier ever offers - so an empty label is the game's answer, not
    a truncated read, and is carried as such to keep the slot indices aligned.
    """
    value = message.get(name)
    if not isinstance(value, str) or len(value) > MAX_LABEL_CHARACTERS:
        raise BridgeProtocolError(f"{name} must be a string of at most {MAX_LABEL_CHARACTERS}")
    return value


def _bool(message: Mapping[str, Any], name: str) -> bool:
    value = message.get(name)
    if not isinstance(value, bool):
        raise BridgeProtocolError(f"{name} must be a boolean")
    return value


def _upgrade_family(message: Mapping[str, Any]) -> str:
    family = _string(message, "family")
    if family not in UPGRADE_FAMILIES:
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
    "ADVANCE_WALL_CEILING_SECONDS",
    "DEFAULT_MAX_FRAME_SIZE",
    "DEFAULT_MAX_UPGRADE_ENTRIES",
    "DEFAULT_READ_TIMEOUT_SECONDS",
    "PAUSE_SETTLE_SECONDS",
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
    "UnlockFamilyState",
    "UpgradeInventoryEntry",
    "UpgradeSlotLabel",
    "decode_handshake",
    "decode_command",
    "decode_command_result",
    "decode_observation",
    "decode_slot_labels",
    "decode_unlock_state",
    "encode_frame",
    "read_frame",
]
