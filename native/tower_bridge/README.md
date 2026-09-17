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
`mode: "instrumented_training"` and `command_capability: "semantic-v2"`.

Subsequent `observation` messages have strictly increasing `sequence` values and
include `lifecycle`, `wave`, `cash`, `health`, `max_health`, `terminal`,
`round_active`, plus bounded `upgrades` entries:

```json
{"family":"attack","index":0,"cost":5.0,"level":2,"max_level":79,
 "unlocked":true,"tier_unlocked":true,"maxed":false}
```

Families are `attack`, `defense`, and `utility`; each is capped at 64 entries.
Each observation also carries the current `game_speed` and `play_time`. The
latter is the game's own account-lifetime clock: it advances at wall-clock rate
at every game speed, so it is liveness evidence that the process is still
running, not an in-run game clock and not a policy feature. The bridge sends a
heartbeat with the latest observation sequence at least once a second, including
while a slow lifecycle transition is in flight.

Between episodes the game holds no initialized run. That is reported as a
`run_unavailable` message carrying the same monotonic sequence, so a controller
can still bind and send a command, and no invented run values are ever presented
as observations.

## Command path

Policy actions are `advance` and `buy_upgrade`; the policy's own `WAIT` is an
`advance`, and there is no separate `wait` command kind. Navigation and speed are
separate controller-owned kinds and can never become learned actions:

- `lifecycle` with an allowlisted `action` dispatches one of the game's own
  parameterless entry points and waits for the game's own state to agree;
- `set_speed` writes the game's `gameSpeed` and dispatches its own
  `GameSpeedModifier`, confirmed against the observed `game_speed`;
- `advance` runs the world frame by frame and pauses again, returning the settled
  observation. It carries `budget_game_ms`, `frame_game_ms`, and
  `health_change_fraction`.

`advance` is how the simulation is decoupled from the wall clock. While
`Time.captureDeltaTime` is set, one rendered frame advances exactly
`frame_game_ms` of game time however long it took to render, so the game time
between decisions depends on neither the host's speed nor the game's own speed
multiplier. That multiplier is pinned at 1x by standing decision and is never a
speed-up mechanism: a faster game clock makes each frame worth more game time,
which coarsens decision moments instead of preserving them (`solution.md` 9.2c,
`M1B-E016`).

The loop stops at the first decision event, or when the budget is spent. The
events and their precedence mirror the host's own predicate exactly -
`event:run_ended`, `event:wave_changed`, `event:newly_affordable`,
`event:health_changed`, else `budget_exhausted` - so one socket round trip buys
one policy decision rather than one time slice.

**The observation bound to an `advance` result is settled, and the host uses
it.** `Pause` is dispatched to Unity's main thread and lands a frame or two after
it is sent, so the state the instant the loop breaks is mid-frame.
`captureDeltaTime` therefore stays at `frame_game_ms` until the pause has landed
- the build resolves no game-owned pause flag, so landing is observed as two
further rendered frames or 500 ms of wall clock, whichever comes first - and
those tail frames are counted into `frames` and `game_ms` at the same weight as
every other frame. Only then is the state read. The readings taken inside the
loop decide *when* to stop; this settled reading decides what is *reported*, so
the `reason` in the result and the observation sent immediately before it always
describe the same moment. The host binds that observation to the result rather
than waiting for the next free-running stream tick, which is what keeps one
decision at one round trip. Advancing with the budget set to
one frame is the single-frame case; there is no separate step command and no
wall-clock sleep fallback. If no frame renders within the bridge's wall-clock
ceiling the result is `ambiguous` with `no_frame_rendered`, and if the engine
clock cannot be resolved it is `ambiguous` with `clock_unavailable`. Only the
engine leaf icalls `Time::get_frameCount` and `Time::set_captureDeltaTime` are
called directly, both required to be attributable to `libunity.so`.

Every `command_result` carries `frames`, `game_ms`, `play_ms`, and `wall_micros`,
in that order; they are zero for the commands that advance no frames. `game_ms`
is budget accounting - frames times `frame_game_ms` - while `play_ms` is measured
from the game's own `playTime` clock across the same advance, so the intended 1:1
mapping between them is checkable rather than assumed. `wall_micros` is real
elapsed `CLOCK_MONOTONIC` time, which is what makes the bridge's 15-second
advance ceiling a real ceiling; the host's default read timeout is derived from
that ceiling plus the settling window. Availability inside the loop is
read exactly as the host masks it - active run, `unlocked`, not `maxed`, priced
above zero, and affordable - over the first twenty slots of each family, which is
the width of the host's action schema.

The free-running stream cadence and `WAIT` remain game-time quantities scaled by
the observed speed; a fixed wall-clock cadence starves the policy of decisions
per game second (see `M1B-E002`). With the multiplier pinned at 1x that scaling
is the identity.

One command may be in flight. A command binds the latest observation sequence and
carries a bounded ASCII `request_id`; a repeated id or a superseded sequence is
`rejected` with `stale_or_duplicate`. The native parser is deliberately minimal
and reads the canonical encoding at fixed offsets, so the client's exact key
order and compact separators are part of the contract and are pinned by
`tests/unit/test_instrumented_bridge.py`.

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

A family's cost array is only populated after that family has been displayed, so
the bridge dispatches the game's own `UpgradeCostCalc`, `UpgradeDefenseCostCalc`,
and `UpgradeUtilityCostCalc` when a run becomes active and after each confirmed
purchase. Entries without a positive cost are still rejected.

`Main` exists only inside the battle scene, so no `Main` method starts a run from
the home screen, and the known restart entry points are progression-gated at this
baseline. The episode boundary therefore still needs one bridge-gated tap. The build keeps no live in-run clock: `roundTime`, `gameplayTimeThisRound`, and
`realTimeThisRound` all read 0.0 throughout a run, so the controller owns run
time.

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
