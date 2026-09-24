# Tower bridge (M1B instrumented-training adapter)

This is original ARM64 source for the private `instrumented-training` profile.
It is not used by official evaluation. Watch mode (`scripts/spectate.py`) runs
on the instrumented clone with this bridge deployed, at 60 Hz through
`-gpu lavapipe`. It dynamically discovers the
unnamespaced IL2CPP `Main` class and allowlisted field names, then exposes exact
observations and a bounded semantic command path on device loopback TCP port
`47651`. No game bytes, offsets, assets, signatures, dumps, or runtime outputs
belong in this directory.

## Ownership and protocol

The bridge owns dynamic IL2CPP lookup, raw state snapshots, and main-thread
command dispatch. The Python client in
`src/tower_rl/simulation/instrumented_bridge.py` owns TCP framing, time
limits, compatibility validation, stream ordering, and clean disconnects. Neither
component owns game rules or normalizes observations for an RL policy.

Frames are a four-byte unsigned big-endian payload length followed by at most
65,536 bytes of UTF-8 JSON. The first message is a `handshake` containing
protocol version `2`, bridge version, configured package version and version
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

Each observation also carries a `live` object: every `Main` field
`observation-v2` shows the policy, under the game's own field name, read at the
width its own declared IL2CPP type names. The list is `kLiveFieldNames` in
`tower_bridge.cpp` and `LIVE_WIRE_NAMES` in `environment/run_state.py`, which
are one schema in two languages; the host refuses a state message that does not
carry exactly that set. A field the class does not carry, or one declared at a
type this bridge cannot read, fails initialization by name — a silent zero for a
field the schema declares would be wrong in every observation after it, which is
the M1B-E017 failure repeated thirty-seven times.

Each observation also carries the current `game_speed` and `play_time`.
`game_speed` is the multiplier the game holds at that instant and is reported,
never gated on: every observation the host reads is taken from a world the
bridge has paused, where the field reads `0.0` whatever the running world would
do (M1B-E009). What the world really advanced at is read from `round_ms` against
`game_ms` on an `advance`, below. The latter is the game's own account-lifetime
clock: it advances at wall-clock rate
at every game speed, so it is liveness evidence that the process is still
running, not an in-run game clock and not a policy feature. The bridge sends a
heartbeat with the latest observation sequence at least once a second, including
while a slow lifecycle transition is in flight.

**While the world is paused the sequence does not move.** The sequence exists to
stop the host acting on a stale view; a paused world produces no new information,
so its view cannot go stale. Between commands — the advance leaves a still-active
run paused — the idle tick emits a heartbeat instead of a fresh observation, and
the heartbeat names the sequence that still stands. Without that, a policy whose
forward pass and learning step take longer than one stream interval had every
command rejected as `stale_or_duplicate`, even straight after a fresh read. A run
that has ENDED is never that world: it keeps producing screens, so the screens
between episodes keep streaming and the episode boundary still sees its lifecycle
transitions.

The sequence is therefore held only when the world is standing still, which takes
both halves: this bridge pressed `Pause`, **and** the run is still in progress in
the settled state the host is about to be sent. The advance reports the pair
itself, because only it knows it pressed the control; a lifecycle `pause` counts
only once the game's own state confirms it; and the flag is confirmed once more
against the state about to be sent, which can clear it but never set it. Reporting
the intent to pause instead is what deadlocked the episode boundary: a tower that
died inside the pause-settle window — where the last advance before a death always
sits — left a terminal observation with the stream held behind it, and the host
polled that one reading until it timed out, at roughly one boundary in seven.
Re-deriving the flag from scratch would be the opposite error: a round that
started in between — the game's own auto-restart does that — would mark a running
world as paused, which is the one way the host could be left acting on a view the
bridge had stopped refreshing.

Between episodes the game holds no initialized run. That is reported as a
`run_unavailable` message carrying the same monotonic sequence, so a controller
can still bind and send a command, and no invented run values are ever presented
as observations. Its `reason` separates the two states that reach it:

- `no_initialized_run` — `Main` is alive but its scalars do not describe a run.
  The game is up and idle at the home screen (`M1B-E015`).
- `main_unavailable` — `Main` is not alive yet. The game has not finished
  starting: the splash, or the Firebase OFFLINE modal.

Host bring-up reads that difference as its readiness signal, which is why the
bridge reports it instead of answering `run_unavailable` for both.

## Command path

Policy actions are `advance` and `buy_upgrade`; the policy's own `WAIT` is an
`advance`, and there is no separate `wait` command kind. Navigation and speed are
separate controller-owned kinds and can never become learned actions:

- `lifecycle` with an allowlisted `action` dispatches one of the game's own
  parameterless entry points and waits for the game's own state to agree;
- `set_speed` writes the game's `gameSpeed` and dispatches its own
  `GameSpeedModifier`. Its `speed_requested` confirmation says the slot holds
  the requested value, not that the world runs at it: the read-back is of the
  field the bridge just wrote, and no readable field reports the effective rate.
  The verification is the next `advance`'s `round_ms` against `game_ms`;
- `slot_labels` answers once with the game's own name and description for every
  upgrade row, in a `slot_labels` frame sent before the state and the result.
  The arrays are constant for a build, so the host asks once per session and
  before the first round; the labels are for humans and never enter the
  observation tensor;
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
one policy decision rather than one time slice. The loop also has a hard
wall-time ceiling, and an advance it cuts off reports `wall_ceiling` rather than
`budget_exhausted`: the settled state cannot tell the two apart, and the host
fails an episode on the second because a truncation by host load is not part of
the decision problem.

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

Every `command_result` carries `frames`, `game_ms`, `round_ms`, and
`wall_micros`, in that order; they are zero for the commands that advance no
frames. `game_ms` is budget accounting - frames times `frame_game_ms` - while
`round_ms` is measured from the game's own per-round clock
(`Main.gameplayTimeThisRound`, a `float`) across the same advance, so the
intended 1:1 mapping between them is checkable rather than assumed. `playTime`
cannot serve as that witness: it is the account-lifetime clock and advances at
wall rate whatever `captureDeltaTime` does, so a ratio built on it only
reproduces one over the speed-up (M1B-E017). A ratio that runs away from 1 is the
one report that a world is simulating more time per frame than it was told to -
the shape a speed multiplier left applied has - and the host fails such an
episode by name rather than rescaling the frame's worth to match (M1B-E023).
`round_ms` is zero when the settled
state could not be read at all, which the outcome and reason on the same result
already report, and zero when the round clock reset under the advance. `wall_micros` is real
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
`rejected` with `stale_or_duplicate`; because the paused stream holds the
sequence, host latency alone can no longer supersede it. The native parser is deliberately minimal
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

`unlock_state` and `unlock_all_upgrades` are how upgrade availability is applied
(ADR 0011). They are in the **production** build: the game recomputes its real
rows' availability at every round start, so a run played under
`--upgrade-availability all` issues `unlock_all_upgrades` at each round start,
and a command an ordinary measured run depends on cannot live behind a
diagnostics flag. `unlock_state` reads `upgradeUnlocked`,
`upgradeDefenseUnlocked` and `upgradeUtilityUnlocked` and reports each array's
length and how many of its elements are true; `unlock_all_upgrades` sets every
element true through `WritePrimitiveArray`, the exact mirror of
`ReadPrimitiveArray` (same bounds and element-size checks, same pointer
arithmetic, the `memcpy` reversed), and then reports the same pairs read back
out of the arrays, so a write that did not take reports as one. The report is a
separate `unlock_state` frame sent before the state and the result, exactly as
the slot labels are, and carries a `wrote` flag the client holds against the
command it sent.

All three families are resolved and every element proven readable before
anything is written, so a drifted schema refuses the command whole and leaves
the game exactly as it was. A write that fails *after* that pre-read can still
leave the arrays part way; the frame is emitted either way and shows what the
arrays then hold, which is the only honest report of a partial state. Each of
the three passes re-reads the field rather than reusing the array pointer the
first one found, and refuses if it is handed a different object: a held pointer
would write to — and then dutifully read back — an array the game had swapped
out, reporting a success the game never saw.

Moving the pair into production changed the production digest, which is the
identity every deployment is checked against; the new digest is recorded in
`docs/setup.md` beside the one it replaces, and `state/bridge/current` has to be
reinstalled for it. The protocol version is unchanged: both kinds were already
part of protocol 2's command set on the host side, and the `unlock_state` frame
is additive.

Nothing here writes unless it is asked to: `unlock_state` reads and reports, and
`unlock_all_upgrades` is issued only by an environment configured
`--upgrade-availability all`. The write is in-memory only — nothing here calls a
save.

## Known live behavior

IL2CPP resolution happens on the first client connection and is then cached.
Resolving at library-load time crashes the game process, because `libil2cpp.so`
is loadable well before its runtime is usable; a connecting host client is the
evidence that the game has had time to initialize.

A family's cost array is only populated after that family has been displayed, so
the bridge dispatches the game's own `UpgradeCostCalc`, `UpgradeDefenseCostCalc`,
and `UpgradeUtilityCostCalc` when a run becomes active and after each confirmed
purchase. Entries without a positive cost are still rejected.

The episode boundary touches no screen. `start_round` is the home screen's own
BATTLE control, `BattlePanelUI.StartNewRound` on the GameObject named
`BattlePanel`, and it is the only lifecycle action delivered anywhere but
`Main`. `Main.StartNewRoundFunction` and `Main.AutoRetryBattle` are both
delivered - `Button_GameEndPanelGoHome` proves delivery to `Main` from the same
state - and both do nothing, which is why they are gone (`M1B-E022`).
`BattlePanel` only exists while the home screen is up, so a finished run is
closed with `go_home` first; that sequence is what `begin_episode` performs, and
it needs no retry control.

The build does keep live in-run clocks: `roundTime`, `gameplayTimeThisRound`, and
`realTimeThisRound` are `float` fields, and the earlier reading that they "all
read 0.0" came from reading them as `double`. Read as singles they advance
together with the round and are the game-owned witness `round_ms` reports
(M1B-E017). The controller still owns run *boundaries*, which is a separate
question.

`tower_bridge.cpp` uses only exported IL2CPP APIs for fields and arrays:
`il2cpp_field_get_value`, `il2cpp_field_static_get_value`,
`il2cpp_field_static_set_value`, `il2cpp_array_length`,
`il2cpp_array_get_byte_length`, and `il2cpp_array_object_header_size`. It does
not embed game offsets. The verified semantic fields include `towerMaxHealth`
(not `towerHealthMax`) and all three upgrade-family arrays.

## Private build and deployment

The supplied CMake project expects an ARM64 Android target, for example:

```sh
cmake -S native/tower_bridge -B "$TOWER_BRIDGE_BUILD_DIR" \
  -C state/bridge/config/profile.cmake \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-35
cmake --build "$TOWER_BRIDGE_BUILD_DIR"
```

The build is reproducible: no host path reaches the binary, so the same source,
NDK and `profile.cmake` give the same `libtower_bridge.so` from any build
directory or worktree, and the digest the install is filed under identifies the
source rather than the machine it was built on. The flags that buy that are in
`CMakeLists.txt` with the reasoning beside them; `docs/setup.md` says how to
check it.

**The private build configuration lives at `state/bridge/config/profile.cmake`**
— the package version and version code, the official signer SHA-256, the
original `libunity.so` and `libil2cpp.so` SHA-256 values, and the profile id, as
CMake cache values. It is machine-local and is never committed; `state/` is
git-ignored in full. The configuration the deployed bridge was actually built
with is readable from the `CMakeCache.txt` installed beside it
(`state/bridge/current/CMakeCache.txt`), which is the reference to rebuild
against. A build left at the `unconfigured` defaults compiles and then emits a
compatibility error instead of a handshake.

`docs/setup.md` has the full build-install-verify recipe, including the digest
the installed directory is named for.

### Render-off experiment build

`-DTOWER_BRIDGE_RENDER_FRAME_INTERVAL=N` (an integer in 1..60, default 1) builds
a variant in which the game renders and presents one player-loop frame in N
(#27). The loop itself — scripts, physics, `Time.frameCount`, the
`captureDeltaTime` step — still runs every frame; what is skipped is rendering.
At the default nothing is defined and the production artifact is byte for byte
the same, which `docs/setup.md` records as a digest pair.

Above 1 the bridge resolves the static `m_RenderFrameInterval` (`System.Int32`)
of `UnityEngine.Rendering.OnDemandRendering` at initialization, and on every
connection writes N into it before the handshake with
`il2cpp_field_static_set_value` — the engine polls that field each frame, and
its public setter is managed code this thread must not run. The value is read
back; a class, field or type that does not resolve fails initialization, and a
read-back that is not N answers `compatibility_error` ("render frame interval
not applied") instead of a handshake, so the build never runs rendered under
the render-off label. It logs `render interval=… readback=…
effective_render_fps=… target=…` once per connection (the effective rate is
`OnDemandRendering::GetEffectiveRenderFrameRate`, reported and never gated on),
and one `renderprobe frames=… rendered=… wall_us=…` line per advance, from
`Time::get_renderedFrameCount` across the advance — the direct evidence that
about one frame in N rendered. The interval is an in-memory static of the game
process: nothing is saved, and a force-stop drops it.

The variant is selected per run by `TOWER_BRIDGE_BUILD_DIR` pointing at a build
directory that also holds the unchanged `libunity-bridge.so` from
`state/bridge/current` and the variant's own `CMakeCache.txt`; the N=16 build is
at `state/bridge/builds/render-interval-16/`. It is never installed as
`state/bridge/current`. **Spectate and recording must never use it**: the screen
shows one frame in N.

`scripts/instrumented_bridge.sh` deploys to, verifies, and cleans the private
rooted clone. It expects the build directory to hold both `libtower_bridge.so`
and the patched `libunity-bridge.so` whose only change is an added `DT_NEEDED`
entry for the bridge, and defaults to the installed bridge at
`state/bridge/current`. Do not commit the resulting `.so`, extracted libraries,
overlays, APKs, device data, or logs.
