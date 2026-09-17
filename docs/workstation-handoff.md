# Tower-RL — Workstation Handoff

## START HERE — current state, 2026-09-17

This section is the one place that says where the project actually is. Everything
below it is historical and dated; read this first and treat older sections as
context rather than as current truth.

### The goal

A reproducible benchmark on the real game in which several RL backbones are
trained and compared under one identical budgeted protocol, where the best
learned model reproducibly beats the random and scripted baselines, with all
evidence in `docs/experiments.md`.

### What is proven and working

- **The environment.** The game is observed and controlled through its own
  runtime, with no OCR in the decision loop. 1,000 consecutive scripted episodes
  ran at 100 percent validity (`M1B-E009`). That clears M2's *reliability*
  clauses. **M2 is not complete**: its speed/actor-count comparison clause and
  its visual-evidence clause are both open.
- **The learning pipeline, end to end on the real game.** Both backbones,
  interleaved on one device, 54 episodes, 614 optimisation steps, no replay
  rejections, checkpoints round-tripping with identity and checksums
  (`M1B-E011`). This proves plumbing, not learning.
- **Two backbones behind one contract suite**: `recurrent-q` and `stacked-dqn`
  (the rank-1 candidate from `docs/rl-candidates.md`). They share the trunk,
  the dueling heads and the n-step double-Q targets, so only the core differs.
- **The comparison machinery**: interleaved scheduling, bootstrap intervals,
  Cohen's d, and `required_episodes` for power.
- **Frame-exact stepping** (`M1B-E016`) and **the advance loop inside the
  bridge** (`M1B-E017`), described below.

### Standing decisions a new agent must not re-litigate

1. **The game's own speed multiplier is pinned at 1x and is not a speed-up
   mechanism.** A faster game clock makes each rendered frame worth more game
   time, which coarsens the agent's decisions in proportion to the speed gained:
   4.8 decisions per wave at 64x against 12.2 at 1x (`M1B-E012`). Encoded in
   `instrumented_run_adapter.py` as `GAME_SPEED = 1.0` with `_pin_game_speed`
   restoring it at every episode start, and asserted by test. Speed is not a
   parameter anywhere: not in the adapter, the cadence, or any runner's
   arguments.
2. **Speed comes from stepping frames faster.** `Time.captureDeltaTime` makes one
   rendered frame worth a fixed amount of game time however long it took to
   render, so decision moments are identical at any speed *by construction*. The
   bridge implements this; a 250 ms step costs about 57 ms of wall clock whether
   the multiplier is 1 or 16 (`M1B-E016`).
3. **The clone must be offline during automation, verified by interface.**
   `airplane_mode_on` reads 1 while the wifi radio is up and was never evidence
   of anything (`M1B-E010`). Use `svc wifi disable` / `svc data disable` and
   confirm nothing but `lo` holds an IPv4 address. `instrumented_bridge.sh
   deploy` now refuses otherwise.
4. **The game needs a network to *start*, not to play.** It blocks on a Firebase
   check and an OFFLINE modal. `scripts/clone_session.py start` performs the
   launch-online, reach-home, cut-radios sequence and verifies the result.

### Done: the advance loop is inside the bridge

Commit `611667d`, measured on the device in `M1B-E017`. One `advance` command per
decision, 5 of 5 episodes valid at 100 ms per frame, speed-up 5.013, 60.4 fps,
274 ms per advance, zero `advances_cut_short`, and zero `BRIDGE_EVENT_DIVERGENCE`
— the bridge's stopping conditions and the host's `_events_between` agreed on
every decision, which is the invariant that keeps the host definition
authoritative. Two defects that run exposed are fixed in the commit that carries
`M1B-E017`: the host now treats **any** inbound frame as liveness (no heartbeat
reaches it while a command is always in flight, so every unattended run died at
about 60 seconds), and the game-time witness is now the game's own per-round
clock, `round_ms`, because `playTime` runs at wall rate and witnessed nothing.

### Done: the `frame_game_ms` sweep, and the standing speed decision

`M1B-E018`. Five arms (16.7, 50, 100, 100 repeat, 250 ms), 8 episodes each,
sequential not interleaved. Decision density (dec/wave) is flat at 20.9–21.1
across 16.7/50/100/250 ms — frame size does not move it over this range. 250 ms
shows a statistically detected final-wave difference from the 16.7 ms reference
and is rejected as not faithful to it, regardless of its (favourable) direction.
**Standing decision: the benchmark runs at `frame_game_ms = 100`.** The prior
target of matching 89.3 decisions/episode (`M1B-E014`) is withdrawn as an error:
that figure is from the old, pre-`M1B-E016` code path, and every arm of the
`M1B-E018` sweep lands at 124–167 decisions/episode through the current path,
regardless of frame size. `M1B-E018` also flags that eight episodes per arm
cannot detect a one-wave fidelity difference at 80% power (97/arm would be
needed), so absence of a detected difference at 50/100 ms is not equivalence,
only an absence of evidence at this sample size.

### Done: the game-time accounting question is closed for reporting purposes

`M1B-E019`. The round clock stays the authoritative witness for reported
speedup, already in effect since `c07ae35`; the advance loop's budget stays on
frame arithmetic (`frames × frame_game_ms`), for the reasons recorded there —
the budget bounds quiet game time rather than measuring it, and advances stop
on events rather than on the budget. The measured law is
`round_delta ≈ 1.07 · frame_game_ms · (loop_frames − 1)`: the game credits
about 7% more simulated time per frame than requested, and about one frame per
advance goes uncredited. Two mechanism questions are open and unchased — the
1.07 factor, and the uncredited frame — see `M1B-E019` for the candidates
considered.

### The immediate next slice

**The `stale_or_duplicate` sequence race**, which blocks training. `M1B-E019`
found the bridge's roughly 250 ms idle observation stream races any
sequence-bound command once host latency approaches it (15 of 35 advances
rejected in a deliberate-latency test) — a trained network's forward pass plus
learning step routinely exceeds 250 ms, so this must be fixed before training
can run unattended. It surfaces as `ACTION_PIPELINE_FAILED` (lost episodes, not
silent corruption). A fix may be landing concurrently with this note; check
`M1B-E019` and recent commits for its status before starting new work here.

**Episode-boundary overhead** stays queued after the sequence race. Advance
share — the fraction of wall clock spent on genuine advances rather than
boundary — is 0.834 at 100 ms and falls further at coarser frames, so the
fixed per-episode boundary already dominates throughput above 100 ms. The
6-second result-panel settle and the gated boundary tap (item 2 below) are the
two known contributors and are the next thing to cut once the race is closed.

### After that, in order

1. `instrumented_bridge.sh deploy`'s cold-launch gap. It force-stops and
   cold-launches, which lands on the OFFLINE modal when the device is offline:
   the `M1B-E017` run had to bring the radios back up, force-stop, relaunch, wait
   for home, and cut the radios again before it could deploy. Every device run
   pays this until it is fixed.
2. The boundary tap. `Main.Instance` *is* alive at the home screen with a
   non-zero native handle (`M1B-E015`), so the long-held premise that `Main`
   exists only in the battle scene is wrong and the receiver hunt was aimed at a
   problem that does not exist. Why `UnitySendMessage` does not take is the open
   question: the method may not be on the component attached to that object, the
   object may be inactive and so invisible to `GameObject.Find` semantics,
   preconditions may be unmet, or the transition may exceed the 30-second wait.
   Removing the tap also removes the 6-second result-panel settle — 6 of the 7.3
   seconds of per-episode boundary that is 48 percent of wall clock.
3. Save an already-started offline snapshot with `clone_session.py snapshot` and
   verify `restore` does not re-run the Firebase check.
4. Revisit the renderer only once nothing in the loop reads a pixel. `-gpu host`
   was withdrawn for corrupting the frame (`M1B-E004`), which broke screen
   classification; game logic was never affected. Under frame stepping with the
   loop moved into the bridge, frame rate governs throughput, so this matters
   again — for fps, not for pixels.
5. Longer training runs, the comparison floor (scripted, random, wait arms), and
   the speed equivalence gate.

### A small gap worth closing: `run_episodes.py` writes no per-episode data

It writes one aggregate JSON record per invocation (`--output`, default
`/tmp/tower-rl-episodes.json`) and nothing per episode. The `M1B-E018` sweep
needed per-episode waves, decision counts, and timing to compute its bootstrap
intervals and had to bootstrap that data through a scratchpad observer wrapper
around the runner rather than reading it from the runner itself. Worth closing
in the runner: writing a per-episode sidecar (or extending the aggregate record
with a per-episode array) would remove the need for an ad hoc wrapper the next
time a sweep or comparison needs episode-level data.

### Claims this session corrected — do not reinstate them

- `M1B-E006`'s decisions-per-wave table (63/79/69) **does not reproduce** and
  must never be quoted as current.
- 8x was adopted then withdrawn as a training speed. It does match 1x density,
  but only because 8x sits below the frame limit by coincidence.
- "A property setter from the socket thread crashed the game twice" was
  **misattributed**. The recorded double crash was `il2cpp_domain_get` at
  library-load time before any client connected. No entry records a
  `runtime_invoke` crash. The rule is: never execute managed game code or Unity
  scene-graph code from the socket thread; engine leaf accessors, attributed to
  `libunity.so` by `dladdr` first, are a different category and are proven safe.
- `Main` existing only inside the battle scene: **disproved** by `M1B-E015`.

### Where things are tracked

| Document | Holds |
| --- | --- |
| `docs/task.md` | Authoritative scope, milestones and gate criteria |
| `docs/solution.md` | Design decisions, including 9.2c on decision moments |
| `docs/experiments.md` | Every finding, newest first, including negatives |
| `docs/rl-candidates.md` | The RL algorithm study and its ranking |
| `docs/adr/` | Architecture decisions 0005-0008 |
| This section | Current state and what to do next |

### Device state

Nothing is running. The last stage was closed with
`instrumented_bridge.sh cleanup`: original `libunity.so` SHA-256
`ffc1f3ef…dd0040` verified, package identity unchanged (`versionCode 1199`,
`29.0.3`, installer `com.android.vending`), zero mounts, artifacts removed,
device offline, no emulator running.

To resume: launch with `clone_session.py start`, then
`TOWER_BRIDGE_BUILD_DIR=<private build dir> ./scripts/instrumented_bridge.sh
deploy`. The private build directory holds `libtower_bridge.so` and the patched
`libunity-bridge.so`; the NDK is at `~/.local/share/android-sdk/ndk/29.0.14206865`
and the bridge is rebuilt with `cmake --build <build dir>`.

**`/tmp/tower-bridge-live.latest` is not to be trusted without checking.** Before
the `M1B-E017` run it pointed at a build whose `CMakeCache` read `unconfigured`
for every compatibility value and which had no patched `libunity-bridge.so`, so
it could neither deploy nor handshake. That run rebuilt it, with the package
version, version code, signer and library hashes verified against the live
device, and repointed the file. Verify the cache values and the presence of both
libraries before relying on the pointer; `/tmp` does not survive a reboot.

The logcat tag is
`tower_bridge`. Always finish with `cleanup`.

### Host quirks worth knowing

- Background tasks are killed by a low-memory watchdog that reads `free` rather
  than `available`; it killed two runs on a machine with 92 GB available. Run
  device measurements in the foreground or detached with `setsid nohup`.
- Piping a command through `tail` masks its exit status. It hid a mypy failure
  at `1be263a` and a crashed training run behind exit code 0. Check each check's
  own status.


## What exists on this Mac

The validated account-bearing snapshot is stored outside the repository at:

```text
${ANDROID_AVD_HOME:-$HOME/.android/avd}/tower_rl_api36_play_arm64.avd/snapshots/tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914/
```

The snapshot is approximately 2.9 GiB on disk (`ram.bin` and renderer
textures). The AVD is API 36 `google_apis_playstore`, ARM64, Pixel 2, 1080×1920,
420 dpi, and was created with Android Emulator 37.1.11. The game is the
Play-installed The Tower 29.0.3 (`versionCode=1199`), not the local 29.0.1
XAPK.

It is intentionally not checked into Git. It contains Android user data,
account/session state, renderer state, and a local game save. The same applies to
the other snapshots under:

```text
${ANDROID_AVD_HOME:-$HOME/.android/avd}/tower_rl_api36_play_arm64.avd/snapshots/
```

## Reopen the validated Mac baseline

On this Mac, with the AVD already provisioned:

```text
emulator @tower_rl_api36_play_arm64 \
  -gpu lavapipe \
  -no-audio \
  -no-boot-anim \
  -snapshot tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914 \
  -no-snapshot-save
```

Then verify:

```text
uv run tower-rl doctor \
  --xapk local/the-tower-29-0-1.xapk \
  --serial emulator-5554
uv run tower-rl probe --serial emulator-5554
```

The expected state is Battle home, Tier 1 selected, highest wave 2, 53 coins,
0 gems, x1.00 total coin bonus, Labs locked, airplane mode enabled, and no
external route.

## Prepare the RTX 4090 workstation

Treat the workstation as a new device profile. The observed workstation is
Ubuntu 26.04.1 x86_64 with an i9-14900, 125 GiB RAM, RTX 4090, and accessible
KVM. Do not assume this Mac snapshot
will boot there: an RTX 4090 machine is commonly x86_64, while this snapshot is
from an ARM64 AVD and is also tied to the pinned Lavapipe/Swangle renderer.

1. Clone the repository and run `uv sync --all-groups`.
2. Run the repository preflight before changing device state:

   ```text
   mkdir -p runtime
   ./scripts/workstation_preflight.py --json > runtime/workstation-preflight.json
   ```

   Characterize the host, Android SDK, emulator version, virtualization, ABI,
   storage, and renderer before selecting an AVD.
3. Copy `configs/workstation.example.yaml` to `configs/local.yaml` and fill in
   the discovered values. Create a compatible API 36 Google Play AVD for the
   workstation ABI with the repository helper:

   ```text
   ./scripts/create_avd.sh tower_rl_api36_play_x86_64 \
     'system-images;android-36;google_apis_playstore;x86_64' pixel_2
   ```

   Use the same logical display profile where possible; record any difference in
   `docs/environment-profile.yaml`.
4. Launch the AVD with the renderer selected for that host:

   ```text
   ./scripts/launch_avd.sh tower_rl_api36_play_x86_64 host
   ```

5. Launch the official game from its Google Play listing with the user's manual
   sign-in. Do not automate credentials, purchases, advertisements, or legal
   consent. The local XAPK remains metadata/reference input only.
6. Recreate the semantic baseline manually (Tier 1 selected, no permanent
   Workshop spending, Labs locked), take the device offline only after the game
   is running, and create a new workstation-local snapshot with a unique name.
   Airplane mode is **not** sufficient and never was: the setting reads 1 while
   the wifi radio stays up with a route (`M1B-E010`). Use `adb shell svc wifi
   disable` and `adb shell svc data disable`, then confirm that `ip -o -4 addr
   show` lists nothing but `lo`.
7. Verify that new snapshot with `tower-rl probe --navigate
   --restore-snapshot <workstation-snapshot>` before any actor or learner work.

The workstation snapshot should be created with the renderer that is actually
validated there. Do not copy or reuse the Mac's account-bearing snapshot merely
to save setup time. If a same-OS, same-ABI private transfer is ever attempted,
keep it outside the repository and treat it as an experimental artifact rather
than a supported bootstrap path.

## Portable versus machine-local material

Portable through Git:

- source code, tests, `uv.lock`, and sanitized documentation;
- the XAPK inspection metadata and renderer/profile requirements;
- the probe and restore commands.

Machine-local only:

- XAPK/APK bytes and extracted game files;
- AVD directories, emulator snapshots, Android user data, and account state;
- screenshots, logs, replay, checkpoints, and trained models.

The first workstation milestone is therefore **reprovision and revalidate**, not
snapshot copying. Once it passes, add the workstation profile as a separate
environment version and only then begin multi-actor isolation work.

## Current continuation state — 2026-09-15

Workstation reprovisioning is complete. The authoritative local runtime is the
Play-installed The Tower 29.0.3 build (`versionCode 1199`) on
`tower_rl_api36_play_x86_64`, serial `emulator-5554`, using `-gpu lavapipe` and
the validated ANGLE/Swangle path. The AVD now has `hw.ramSize=6144`. The new
golden candidate is
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_6gb_offline_home_20260914_workstation`.
The prior 2 GiB snapshot
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation`
is retained unchanged. The emulator is left at the new snapshot with airplane
mode enabled and no route.

The 6 GiB candidate passes the profile probe, bounded navigation/restore, and
one 4/4 natural-death diagnostic. It does **not** pass M1 reliability: the fresh
100-consecutive run stopped after seven valid episodes with an explicit device
failure and restored the baseline. The subsequent Wave 3 stall interpretation
has been rejected by bounded progression evidence: an offline sparse-capture
run advanced through Wave 6 and died naturally at about 377 seconds. The old
120-second deadline expired during valid gameplay, and the first post-baseline
death's `New Highest Wave!` layout moved the Game Stats HOME-button border and
was classified as a modal. Production now permits 600 seconds, polls at the
configured one-second interval, and recognizes both result layouts. A live
1/1 natural-death smoke passed in 351.3 seconds and restored this same snapshot.
No renderer/profile change or replacement snapshot was required. Do not declare
M1 complete until the unchanged 6 GiB baseline passes the required 100 episodes.

A subsequent 10-episode qualification stopped after three valid episodes because
the result-to-home transition completed just after the controller exhausted its
three five-second waits. The captured failure frame was already valid Battle
home, and there was no device/lifecycle failure. Those bounded waits are now 12
seconds; a production-path 2/2 natural-death verification passed in 687.0
seconds and restored the offline baseline. The 100 gate has not been restarted.

The fresh 10-episode retry also stopped after three valid episodes. Evidence then
showed a delayed generic modal-close tap crossing the result-to-home transition,
opening Home Settings, and allowing subsequent lifecycle taps to reach an
unlinked-cloud-save account warning. No confirmation or account action was
selected, and recovery restored the baseline. Generic modal taps are now
disabled: lifecycle waits observe a modal without input and either continue when
it settles or fail explicitly. The narrow remaining blocker is positive subtype
recognition and a destination-safe action for any modal that truly requires
dismissal. Do not start the 100 gate before that path passes a fresh 10/10 run.

The 2 GiB configuration backup is private local state at
`$HOME/.local/state/tower-rl/avd-config-backups/20260914-workstation-2gb/`.
To launch the 6 GiB candidate explicitly:

```text
./scripts/launch_avd.sh tower_rl_api36_play_x86_64 lavapipe \
  tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_6gb_offline_home_20260914_workstation
```

The XAPK is no longer required for this workstation runtime. Its sanitized
metadata remains in `docs/environment-profile.yaml`; do not copy proprietary
archive bytes into the repository.

M0 is complete: host/device characterization, Play installation, renderer
selection, offline baseline, snapshot restore, and bounded navigation probe are
recorded in `docs/experiments.md` (M0-E001 through M0-E014). M1 implementation
is in `src/tower_rl/vision.py`, `src/tower_rl/infrastructure/adb_device.py`,
and `src/tower_rl/application/controller.py`. The repeatable gate is
`scripts/m1_reliability.py`; reports belong under `/tmp` or another ignored
machine-local directory. The post-diagnosis one-episode smoke passed, but the
earlier 100-episode run did not. A future run must finish with `passed: true`,
`valid_episodes: 100`, and `baseline_restored: true` before M1 is declared
complete.

Before changing code, read `AGENTS.md`, `docs/task.md`, `docs/solution.md`,
relevant ADRs, and current controller/vision tests. After any live run, restore
the golden snapshot and verify with `uv run tower-rl probe --serial
emulator-5554 --restore-snapshot <snapshot-name>`.

## Resolved — screen calibration after progression drift — 2026-09-17

Stage B is blocked on one calibration question, recorded rather than guessed at.

The clone's account has drifted from the documented baseline by playing: Highest
Wave 2 to 11, 53 coins to 909, with a `MILESTONES` button and a gem/video widget
now on the home screen. The Battle-home classifier samples pixel (10, 200), which
was background at the baseline and now falls inside the new widget, so the screen
no longer classifies and the adapter correctly refuses the boundary tap.

Resolved in `M1B-E005`. The gate now lives in
`src/tower_rl/infrastructure/visual_profile.py`, calibrated against sixty-three
live frames labelled by the game's own lifecycle, with seven anchors for home,
four for the result panel and three for an active run. The result gate anchors on
the RETRY button itself, and the adapter settles six seconds before classifying
because the panel animates in. Only anchor values are recorded, never
screenshots.

Also open: `M1B-E003` measured throughput under `-gpu host`, which
`M1B-E004` then rejected for rendering the frame incorrectly. Those numbers are
an upper bound until re-measured under `-gpu lavapipe`.

## Experimental no-OCR continuation — 2026-09-15

The scalability investigation in `M1-E007` established a viable behind-the-GUI
path, but it has not replaced the V1 visual contract yet. Two private disposable
AVDs exist outside Git:

- `tower_rl_instrumented_api36` is a rooted clone with the unchanged
  Play-installed 29.0.3 package.
- `tower_rl_gadget_api36` is a throwaway re-signed-XAPK clone used only to
  isolate native instrumentation behavior.

Frida server attachment works at the x86 process level but cannot enumerate the
translated ARM64 IL2CPP module. ARM64 Frida Gadget aborts under
`libndk_translation`. Do not spend another iteration on those routes unless the
host or Android ABI changes.

A small custom ARM64 dependency does work under translation. It starts inside
the Unity process, resolves IL2CPP exports, finds `Main` dynamically, and reads
`Main.gameSpeed`. It was proven first in a temporary re-signed XAPK and then in
the Play-installed package by placing a reversible Magisk bind mount over the
extracted `libunity.so`. In the second proof, package version 29.0.3,
`installerPackageName=com.android.vending`, signed APK bytes, and app data all
remained unchanged. The mount was removed and the bridge/probe files were deleted
afterward; no emulator is intentionally left running.

ADR 0006 now defines the separate instrumented-training and official-evaluation
profiles. The first production bridge slice lives under `native/tower_bridge/`
with its strict host client in
`src/tower_rl/infrastructure/instrumented_bridge.py`. A live 29.0.3 run passed
the exact compatibility handshake and returned both active and terminal
observations with all 60 in-run upgrades (20 Attack, 20 Defense, 20 Utility).
Startup observations fail closed until `Main.Instance` and run scalar state are
initialized. The temporary overlay was removed, a clean reboot restored the
original `libunity.so` hash, and no emulator is intentionally left running.

Step 3 of the six-step sequence in `M1-E007` is now proven live and recorded in
`M1B-E001`. On the rooted clone, `WAIT`, duplicate/stale rejection, locked and
unpriced rejection, and confirmed `attack` and `defense` purchases all execute
through Unity's main thread with game-owned before/after evidence, corroborated
by sparse pixels. Utility is unavailable at this fixed baseline, with evidence.
Five defects in the previously unexecuted command slice were corrected first; the
important ones are that IL2CPP resolution must happen on the first client
connection rather than at library load, `tier_unlocked` never gates a purchase,
and cash is not a confirmation signal.

`M1B-E002` then took the loop to whole episodes. In-run control needs no screen
at all: a greedy scripted policy reaches wave 7 to 10 with 20 to 26 confirmed
purchases per episode, driven only through the bridge. Costs are refreshed by the
game's own `Upgrade*CostCalc` methods, so no tab interaction is needed. The
episode boundary still needs one bridge-gated tap, because `Main` exists only in
the battle scene and the known restart entry points are progression-gated here;
`BattlePanelUI.StartNewRound` is the likely receiver but its GameObject name is
unknown. Speed above the account's own 1.5 ceiling applies through the game's
`GameSpeedModifier`, and decision cadence must scale with it or the policy is
silently starved. Pause and unpause freeze the world exactly, but stepped mode
currently costs about 430 ms per decision and is slower than free running; that
overhead needs profiling. ADR 0007 records the decision that the instrumented
clone is now the primary training and evaluation environment, with a bounded
official cross-check at promotion.

Continue from step 4. The open gates are family cost coverage without an
undocumented manual step (a family's cost array only populates after its tab has
been displayed), deterministic normal-speed scripted parity against the visible
controller, protocol-loss and thread-affinity quarantine behavior, and the speed
equivalence gate. Only after parity should higher `Time.timeScale` values or
actor-count scaling be tested, and no instrumented transition may enter replay
before those gates pass. Keep the canonical emulator and all evaluation
unchanged, unrooted, normal-speed, and pixel-observed.

Operate the private clone with `scripts/instrumented_bridge.sh`
(`verify`/`deploy`/`cleanup`) and `TOWER_BRIDGE_BUILD_DIR` pointing at the private
NDK build directory that holds `libtower_bridge.so` and the patched
`libunity-bridge.so`. Launch that clone with `-gpu lavapipe`; `swiftshader_indirect`
produced an unusable System UI ANR on this host. Always finish with `cleanup` and
confirm the original `libunity.so` SHA-256, unchanged package identity, no
remaining mounts, and no running emulator. Never send a tap without first
classifying the screen.

Private/local-only material includes the copied rooted system image, disposable
AVDs, temporary XAPK/APK extraction, IL2CPP dumps, patched native libraries,
temporary signing key, save bytes, logs, and account-bearing state. None may be
committed. The only durable repository evidence is the sanitized experiment and
this handoff.

## Open questions — 2026-09-17

Recorded rather than guessed at, so the next slice does not quietly decide them.

**Resuming an interrupted training run.** `scripts/train.py` writes a complete,
identity-bound checkpoint every 25 episodes and once at the end, and
`learning/checkpoint.py` will refuse to load one whose profile or schema differs.
Nothing reads one back yet: `TrainingRun.run()` always starts from zero
decisions, so resuming would need it to begin from a restored progress counter,
and the replay buffer is in memory and is lost with the process either way. A
multi-hour run interrupted at hour three therefore restarts. The question is
whether resume should restore replay as well — a checkpoint that restores
weights but not replay resumes into a very different learning problem from the
one it stopped in, and recording that honestly matters more than the
convenience. Not blocking: the budget is counted in decisions, so an interrupted
run is a shorter run rather than a corrupt one.

**Where the stacked agent's window length should sit.** `docs/rl-candidates.md`
3.1 treats `k` as a tuned hyperparameter between 4 and 16 and section 5 names
`k = 1` as the ablation that settles whether history is needed at all. The
default here is 8, chosen as the midpoint and nothing more. The ablation is a
comparison arm like any other and belongs after the comparison floor (E), not
before it.

**Actor-count scaling.** Still unmeasured. One emulator sustains about 236
episodes per hour at 99.3 percent validity (M1B-E008). Whether two or four
instrumented clones on this host multiply that or contend for the GPU is an
empirical question, and the number that settles it is aggregate *valid* episodes
per hour, not episodes per hour. Measure it before committing to long training
runs, because it changes what an equal decision budget costs in wall-clock time.

## Directed next steps — 2026-09-17

Three items the developer set after `M1B-E010`, in dependency order. All need the
device, which is busy with the speed equivalence gate until it finishes.

**1. A snapshot of the game already running, offline.** The startup network
window exists only because the game cold-launches into a Firebase check. Saving
an emulator snapshot while the game sits at `battle_home_tier_1` with the radios
already down removes that window entirely: every later run restores an
already-started, already-offline game and never connects at all. Precedent is
good — `solution.md` 8.1 records that in-place restore preserved the game
process, the Battle-home state and a clean post-restore fingerprint on the
canonical AVD. What must be verified is that a *restored* process does not
re-run the online check.

**2. Remove the last tap by finding the right receiver.** This is the blocker
`M1B-E004` left open, and it gates item 3. The bridge dispatches every lifecycle
method through `UnitySendMessage` to the GameObject named `Main`, and `Main`
only exists inside the battle scene, so `StartNewRoundFunction`, `AutoRetryBattle`
and `Button_ToggleAutoRestartBattle` all no-op from the home screen. Two cheap
experiments before any native work:

- Re-test `enable_auto_restart`. It was judged progression-gated at the
  documented baseline, but the clone has since drifted to Highest Wave 11 and
  909 coins (`M1B-E005`). If auto-restart is now unlocked the game restarts
  rounds by itself and the boundary disappears without finding any receiver.
- Re-test `retry` dispatched at the result panel rather than from home, and
  record whether `Main` still exists at that moment.

If both fail, the identified work is to locate the GameObject that owns
`BattlePanelUI.StartNewRound` and address it by name. That needs a main-thread
trampoline that exists on the home screen, because `UnitySendMessage` is the only
main-thread entry point the bridge has and it addresses objects by name.

**3. Then the renderer is free.** `-gpu host` was withdrawn in `M1B-E004`
because it corrupts the frame — smearing, magenta and cyan banding, ghosted text
— which broke screen classification. Game logic was never affected, because the
bridge reads exact state rather than pixels. The frame matters for exactly one
thing: the gated tap. Remove the tap and nothing in the training loop reads a
pixel, so host rendering becomes admissible again and the `M1B-E003` throughput
figures taken under it become relevant rather than an unusable upper bound.
Until then lavapipe stays, because a corrupt frame with a live tap is the
`M1-E005` failure waiting to happen.


## The frame is the floor on decision granularity — 2026-09-17

A constraint worth stating separately, because it bounds what any amount of
engineering can achieve and it reframes the speed choice.

Unity advances game time per rendered frame. A step shorter than one frame passes
no world time at all, so the finest decision granularity available at speed `s`
is one frame, and one frame is worth `frame_wall_seconds x s` of game time. At
30 fps that is 33 ms of wall clock, so:

| Speed | Game time in one frame |
| --- | --- |
| 1x | 0.03 s |
| 8x | 0.27 s |
| 16x | 0.53 s |
| 64x | 2.1 s |

The environment asks for a 250 ms slice, which a frame exceeds above about 8x.
`M1B-E006` reported decisions per wave of 63, 79 and 69 at 1.5x, 4x and 8x
falling to 39, 20 and 16 above, which appeared to confirm this. **That table does
not reproduce.** Measured fresh in `M1B-E012`, 8x gives 13.4 decisions per wave
and 64x gives 4.8: the direction holds, the magnitude does not, and 8x is not a
speed at which the requirement is met.

Two consequences:

1. **There is no speed that satisfies the requirement.** 8x is 2.8 times better
   than 64x on decision density and 3.6 times worse on throughput, and it still
   reaches only 13.4 decisions per wave (`M1B-E012`). Lowering the speed shrinks
   the violation; it does not remove it, because speed and density are traded
   against each other by construction.
2. **Getting both requires decoupling game time per frame from wall time per
   frame**, which is what `Time.captureDeltaTime` does. `captureDeltaTime` is a
   native-backed property rather than a field, so `field_static_set_value` cannot
   reach it — but it does not need a main-thread receiver either. It is a leaf
   engine binding reachable through `il2cpp_resolve_icall`, as are
   `Time.frameCount`, `QualitySettings.vSyncCount` and
   `Application.targetFrameRate`.

   **A correction to what this section previously claimed.** It said a Unity
   property setter from the socket thread "is the pattern that crashed the game
   twice already". That misattributed the crash. The recorded double crash
   (`M1B-E001`) was `il2cpp_domain_get` called at library-load time, four seconds
   after `libil2cpp.so` appears and *before any client ever connected* — premature
   runtime resolution, not a threading violation and not a property setter. No
   entry in this repository records a `runtime_invoke` crash at all. The rule is
   still right, but its scope is narrower than stated: **never execute managed
   game code or Unity scene-graph code from the socket thread.** Engine leaf
   accessors on `Time` and `QualitySettings` are a different category.

   So the two problems decouple. Only the boundary tap needs an out-of-battle
   receiver.
