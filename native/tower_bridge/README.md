# Tower bridge (M1B instrumented-training adapter)

This is original ARM64 source for the private `instrumented-training` profile.
It is not used by official evaluation or watch mode. It dynamically discovers the
unnamespaced IL2CPP `Main` class and allowlisted field names, then exposes exact
observations and a bounded semantic command path on device loopback TCP port
`47651`. No game bytes, offsets, assets, signatures, dumps, or runtime outputs
belong in this directory.

## Ownership and protocol

The bridge owns dynamic IL2CPP lookup, raw state snapshots, and main-thread
command dispatch. The Python client in
`src/tower_rl/infrastructure/instrumented_bridge.py` owns TCP framing, time
limits, compatibility validation, stream ordering, and clean disconnects. Neither
component owns game rules or normalizes observations for an RL policy.

Frames are a four-byte unsigned big-endian payload length followed by at most
65,536 bytes of UTF-8 JSON. The first message is a `handshake` containing
protocol version `1`, bridge version, configured package version and version
code, official signer SHA-256, original `libunity.so` SHA-256,
`libil2cpp.so` SHA-256, Unity/metadata/profile compatibility values, and the
current `game_speed`. All hashes must be lowercase 64-hex values. It advertises
`mode: "instrumented_training"` and `command_capability: "semantic-v1"`.

Subsequent `observation` messages have strictly increasing `sequence` values and
include `lifecycle`, `wave`, `cash`, `health`, `max_health`, `terminal`,
`round_active`, plus bounded `upgrades` entries:

```json
{"family":"attack","index":0,"cost":5.0,"level":2,
 "unlocked":true,"tier_unlocked":true,"maxed":false}
```

Families are `attack`, `defense`, and `utility`; each is capped at 64 entries.
The bridge sends a heartbeat with the latest observation sequence at least once a
second.

## Command path

One command may be in flight. A command binds the latest observation sequence and
carries a bounded ASCII `request_id`; a repeated id or a superseded sequence is
`rejected` with `stale_or_duplicate`. The native parser is deliberately minimal
and reads the canonical encoding at fixed offsets, so the client's exact key
order and compact separators are part of the contract and are pinned by
`tests/unit/test_instrumented_bridge.py`.

`WAIT` performs no Unity call. It elapses a bounded interval and answers with a
fresh observation.

`buy_upgrade` validates game-owned preconditions first: the entry must be
`unlocked`, not `maxed`, priced above zero, and affordable from current cash.
`tier_unlocked` is reported state, not a precondition — live 29.0.3 reports it
false for every upgrade the game actually offers. The bridge then writes the
aligned static `IntSelect.upgradeSelect` mailbox and queues the game's own
`UpgradeButton`, `UpgradeDefenseButton`, or `UpgradeUtilityButton` through the
official `libunity.so` export `UnitySendMessage`, so Unity executes the purchase
on its main thread. `il2cpp_runtime_invoke` is never called from the socket
thread.

Only the game's own level increment confirms a purchase. In-run cash rises
continuously from kills, so a cash delta is neither confirmation nor
contradiction. A level that moves by anything other than one, or availability
that regresses, is `ambiguous` with `contradictory_state_change`; exhausting the
confirmation window is `ambiguous` with `confirmation_timeout`. Both are
quarantine-worthy and never count as a purchase.

## Known live behavior

IL2CPP resolution happens on the first client connection and is then cached.
Resolving at library-load time crashes the game process, because `libil2cpp.so`
is loadable well before its runtime is usable; a connecting host client is the
evidence that the game has had time to initialize.

A family's cost array is only populated after that family's tab has been opened
during the run (see `M1B-E001`). Entries without a positive cost are rejected,
and an actor must open each family tab once per run until the game's own refresh
path is identified.

`tower_bridge.cpp` uses only exported IL2CPP APIs for fields and arrays:
`il2cpp_field_get_value`, `il2cpp_field_static_get_value`,
`il2cpp_field_static_set_value`, `il2cpp_array_length`,
`il2cpp_array_get_byte_length`, and `il2cpp_array_object_header_size`. It does
not embed game offsets. The verified semantic fields include `towerMaxHealth`
(not `towerHealthMax`) and all three upgrade-family arrays.

## Private build and deployment

Use a private Android NDK build directory outside this repository. The supplied
CMake project expects an ARM64 Android target, for example:

```sh
cmake -S native/tower_bridge -B "$TOWER_BRIDGE_BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-35
cmake --build "$TOWER_BRIDGE_BUILD_DIR"
```

Before a live build, replace the `unconfigured` package/profile compile
definitions and all three SHA-256 values in private build configuration with an
allowlisted profile. An unconfigured native build emits a compatibility error
instead of a handshake.

`scripts/instrumented_bridge.sh` deploys to, verifies, and cleans the private
rooted clone. It expects the build directory to hold both `libtower_bridge.so`
and the patched `libunity-bridge.so` whose only change is an added `DT_NEEDED`
entry for the bridge. Do not commit the resulting `.so`, extracted libraries,
overlays, APKs, device data, or logs.

A live deployment still needs the remaining M1B gates: family cost coverage,
normal-speed parity against the visible controller, pixel-watchdog and
quarantine behavior, and the speed equivalence gate.
