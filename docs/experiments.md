# Tower-RL — Experiments and Evidence

This document records feasibility work, benchmarks, failed approaches, and
contrary evidence. An entry records what was observed; it does not advance a
milestone unless the corresponding gate in `task.md` is satisfied.

Do not add proprietary package bytes, extracted assets, account/save state,
personal screenshots, bulk logs, replay, or model artifacts.

## M1B-E001 — Live semantic command path and family cost coverage

**Date:** 2026-09-16
**Status:** `WAIT` and earned-cash purchases proven live; utility unavailable at
this baseline; family cost coverage requires tab activation

The command slice written in M1-E007 step 3 compiled but had never executed
against the running game. Five defects were found and corrected before any live
claim could be made. The native command parser derived one field offset from a
literal length by hand and was one byte short, so every well-formed command would
have been rejected as malformed. Purchase confirmation required a cash decrease
and treated any cash change as a contradiction, but in-run cash rises
continuously from kills; a real purchase would have been reported ambiguous. The
host client consumed one queued observation per decision while the bridge streams
at a fixed cadence, so it fell progressively behind and bound its commands to
superseded sequences. IL2CPP resolution had been moved to library-load time,
four seconds after `libil2cpp.so` merely appears; that is far earlier than the
proven path and `il2cpp_domain_get` killed the game process twice before a client
ever connected. Finally, the precondition required `tier_unlocked`, which live
29.0.3 reports as false for every upgrade the game actually offers, so no
purchase could ever pass.

Resolution is now deferred to the first client connection and cached, so a
reconnect does not repeat the stabilization delay and the game is demonstrably
initialized before the runtime is touched. In-run availability is `unlocked`,
`maxed`, and a positive cost against current cash; `tier_unlocked` is reported
state, not a gate. Confirmation is the game's own level increment.

On the private rooted clone running the unchanged Play-installed 29.0.3 package
with the reversible overlay, the version-locked handshake passed and exact
observations reported lifecycle, wave, cash, tower health, terminal state, and 60
upgrade entries. The following command outcomes were observed live:

- `WAIT` returned `confirmed` / `wait_elapsed` bound to a strictly newer
  observation;
- a repeated request id returned `rejected` / `stale_or_duplicate`;
- a locked upgrade returned `rejected` / `precondition_failed`;
- an unpriced entry returned `rejected` / `precondition_failed`;
- `attack[2]` returned `confirmed` / `confirmed_state_change` with level 0 to 1,
  cost 4.0 to 6.0, and cash 104.0 to 100.0; and
- `defense[1]` returned `confirmed` / `confirmed_state_change` with level 0 to 1,
  cost 5.0 to 7.0, and cash 97.0 to 92.0.

Sparse pixel evidence agreed with the bridge in both directions. A terminal
bridge observation reporting wave 2 and cash 110.0 matched the visible result
screen, and the confirmed attack purchase was visible as Critical Chance moving
from 1.00% at $4 to 2.00% at $6. One purchase showed cash falling by three while
its cost was four, which is kill income arriving inside the confirmation window
and is exactly why cash is not a confirmation signal.

A material observation gap was found. A family's cost array is only populated
once that family's tab has been opened during the run. Before any tab
interaction the run reported 17 priced attack entries, 0 priced defense entries,
and 0 priced utility entries; opening the defense tab produced 18 priced defense
entries and opening the utility tab produced 13 priced utility entries. Reading
the arrays alone therefore does not satisfy the first ADR 0006 acceptance gate.
Until the game's own refresh path is identified, an actor must open each family
tab once at run start, and any entry without a positive cost must remain masked
and rejected rather than treated as free.

Utility is unavailable at this fixed baseline rather than unsupported: with all
tabs opened, utility reported 13 priced entries and 0 offered entries, while
attack offered 4 and defense offered 2. A utility purchase therefore cannot be
demonstrated from this baseline and is recorded as unavailable with evidence.

Contrary evidence and failures are part of this entry. The clone first ran under
`swiftshader_indirect`, which produced a System UI ANR and an unusable screen;
`-gpu lavapipe` with eight cores ran the same package normally and was used for
all reported results. A deploy that removed the overlay's backing file before
unmounting left the target path resolving to a deleted inode, so the next mount
failed; the deploy and cleanup paths now unmount until the target's SHA-256
matches the original library. One blind coordinate tap, sent without classifying
the screen first, opened Settings and then the Account panel. No account,
credential, link, logout, or cloud action was selected, both panels were closed
with verification between taps, and the account remained not linked. This repeats
the M1-E005 lesson: every tap must follow a positive screen classification, and
the live helpers used for this entry classify the screen before acting.

Reversibility was verified after the run. The overlay was unmounted, the staged
bridge and library files were removed, `libunity.so` again matched its original
SHA-256, and Package Manager still reported version 29.0.3, version code 1199,
and `installerPackageName=com.android.vending`. Airplane mode was re-enabled and
no emulator was left running. The patched library, bridge binary, private build
directory, and live helper scripts remain machine-local and uncommitted.

Remaining before M1B can be claimed: family cost coverage without manual tab
activation or an explicit documented actor step, deterministic normal-speed
scripted parity against the visible controller, protocol-loss and
thread-affinity quarantine behavior, and the speed equivalence gate. No
instrumented transition may enter replay until those pass.

## M1-E006 — Exact-state and accelerated-runtime feasibility review

**Date:** 2026-09-15
**Status:** Static review complete; live reliability work intentionally paused

The user stopped the active M1 reliability retry because six-minute real-time
episodes plus screenshot recognition are not an acceptable production training
path without a stronger scaling result. The runner received a graceful
interrupt, restored the 6 GiB offline baseline, and a final probe verified the
Battle home screen, Tower foreground, airplane mode enabled, no external route,
and no observation errors. The emulator was then stopped through its console.
The scheduled continuation and its active implementation worker were also
stopped so they cannot restart Android implicitly. The interrupted partial run
is not acceptance evidence.

A field review found three materially different integration classes:

1. DeepMind's AndroidEnv supports ordinary unmodified Android applications with
   pixels and touchscreen actions, but its own documentation states that Android
   remains a real-time simulation whose speed cannot be increased. This validates
   actor parallelism and observation reduction as black-box optimizations, not a
   hidden route to faster game time.
2. Unity ML-Agents supports direct structured observations/actions, concurrent
   environments, `time_scale`, and graphics-disabled execution when the Unity
   project/build has been instrumented for ML-Agents. Those controls cannot be
   attached to an arbitrary signed production APK without developer integration
   or modifying/rebuilding the game. The clean exact-and-accelerated architecture
   therefore requires a TechTreeGames-provided training/debug build or supported
   telemetry/control bridge.
3. Tower-specific community tools expose useful but narrower precedents.
   WaveTrace watches the rendered game and performs event-triggered OCR when the
   wave advances. TheTowerSDK decodes `playerInfo.dat`; it documents permanent
   state, completed-run history/battle reports, and a read-only ADB watcher. This
   could replace visual extraction for baseline/progression verification and
   terminal evaluation if a controlled cadence test confirms when the game writes
   the required fields. No reviewed evidence establishes exact high-frequency
   live cash, health, visible run-upgrade state, or an action interface.

Android UI Automator remains worth one bounded audit because it can read and act
on accessibility nodes in release applications. Unity/custom-rendered controls
only appear as useful semantic nodes when the application developer supplies an
accessibility hierarchy, so this is a feasibility check rather than an assumed
solution.

The next decision gate is deliberately bounded and keeps the emulator off until
run: (a) inspect the live accessibility hierarchy, (b) measure save-file write
cadence and fields during one short controlled run without retaining save bytes,
(c) benchmark stable hardware-renderer and headless profiles for aggregate actor
density, and (d) seek a developer-supported bridge if exact faster-than-real-time
simulation is required. Save parsing must remain read-only, local, runtime-only,
and separated from TheTowerSDK's mechanics/formulas; training on those formulas
would create a synthetic approximation and violate the authoritative real-APK
objective.

Sources reviewed:

- https://github.com/google-deepmind/android_env
- https://unity-technologies.github.io/ml-agents/Training-ML-Agents/
- https://unity-technologies.github.io/ml-agents/Python-LLAPI/
- https://developer.android.com/training/testing/other-components/ui-automator
- https://developer.android.com/guide/topics/ui/accessibility/views/custom-views
- https://developer.android.com/studio/run/emulator-commandline
- https://developer.android.com/studio/run/emulator-acceleration
- https://github.com/sbrants/wavetrace
- https://github.com/TmRxJD/TheTowerSDK

## M1-E007 — XAPK/save/runtime instrumentation feasibility audit

**Date:** 2026-09-15
**Status:** Exact read-only production bridge proven live; command and parity gates pending

The locally supplied 29.0.1 XAPK is an official four-split ARM64 Unity IL2CPP
package, not an independently signed repack. Its signing certificate matches the
Play-installed 29.0.3 package exactly. Modifying and re-signing that XAPK would
therefore discard the official signature and risks breaking Play licensing and
signature-bound Google/Firebase integrations. Tower-RL will not bypass those
checks or claim a re-signed package is the official runtime.

Static metadata version 39 was decoded locally with `il2cpp_dumper` 0.7.0. The
29.0.3 release exposes exact run state in `Main.Instance`, including cash, wave,
tower health, game-over and round-active flags, all three run-upgrade cost/level
arrays, and the game-speed fields. Its own attack, defense, and utility purchase
methods can accept the semantic selection used by the UI. `PlayerData` also
contains exact terminal/run fields and upgrade arrays. These findings establish
technical feasibility for a local event-driven bridge that observes the real
compiled game state and invokes the game's normal purchase methods, without
making learned screen coordinates part of the policy.

The game's speed modifier ultimately sets Unity `Time.timeScale`; the method has
an uncapped branch above the normal 5x UI range. An instrumented runtime could
experimentally set a higher speed and invoke that same method. This remains a
speed-altered runtime, not automatically equivalent acceptance evidence. Any
accelerated profile must be compared with normal-speed official evaluation using
fixed scripted actions and available seed/state controls; speed is capped or
rejected when transition order, outcomes, or distributions diverge.

Two tempting passive alternatives failed the live audit:

- Android UI Automator exposed only the full-screen Unity surface and no semantic
  text or controls, so accessibility cannot replace image recognition.
- `playerInfo.dat` decoded successfully and is useful for baseline and terminal
  verification, but during a live Tier-1 run its size, modification time, and hash
  were unchanged after 12 seconds while the controller observed wave 1 and cash
  80. Static disassembly also showed a normal 300-second autosave interval and a
  direct game-over save. Save watching is therefore not a decision-frequency
  observation source, and the file remains read-only.

The current Google Play emulator image is a production build: `adb root` is
disabled and the app is neither debuggable nor profileable. A separate disposable
clone was therefore created from a private copy of the system image and rooted
with Magisk 30.7. The canonical AVD, its snapshots, and the SDK-managed system
image were not modified. The unchanged Play-installed package launched on the
visibly rooted clone without a distinct integrity rejection, although a cold
offline launch still stopped at the already-known entitlement/network panel.

Two generic instrumentation routes were then separated experimentally:

- An x86_64 Frida 17.18 server could enumerate and attach to the ARM64 game
  process through Android's native translation layer. The injected x86 agent
  could not enumerate the translated ARM64 `libil2cpp.so`, so normal Frida module
  lookup and IL2CPP hooks are not viable on this x86_64 AVD.
- An ARM64 Frida Gadget added to a disposable, re-signed XAPK was loaded by the
  ARM linker but aborted in its constructor inside `libndk_translation`. This is
  an observed Frida/native-translation incompatibility, not evidence that custom
  ARM64 code cannot run.

A minimal custom ARM64 shared library proved the narrower mechanism. It was
loaded as a `DT_NEEDED` dependency of `libunity.so`, started a background thread,
resolved `il2cpp_domain_get` and the other required IL2CPP exports, found the
unnamespaced `Main` class dynamically, and read the static `Main.gameSpeed`
field. The re-signed XAPK process remained alive and reported `00.00` on its
pre-game screen. This establishes an exact, non-OCR state path through the real
game runtime without relying on Frida.

The same bridge was then tested against the unchanged Play-installed 29.0.3
package. A Magisk bind mount over only the extracted `libunity.so` supplied the
added dependency while leaving the signed split APKs, package version, installer
identity, and app data unchanged. The process remained alive, package manager
still reported `installerPackageName=com.android.vending`, and the bridge again
read `Main.gameSpeed`. The overlay was removed after the proof; reboot would also
remove it. This is the preferred workstation training route because it preserves
the official package identity and avoids re-signing. It still requires a private
rooted actor clone and is not the canonical evaluation runtime.

The repository bridge implementation was subsequently built for ARM64 with NDK
29.0.14206865 and validated on the rooted 29.0.3 clone. Its version-locked
handshake matched package version/code, official signer, original `libunity.so`,
`libil2cpp.so`, Unity version, metadata version, bridge version, and training
profile. During an active Tier-1 run it returned `lifecycle=active`, wave, cash,
current/max health, round flags, and 60 complete upgrade entries: 20 Attack, 20
Defense, and 20 Utility. A second sample after natural death returned
`lifecycle=terminal`, zero health, and the same complete inventory. These two
samples prove exact active-to-terminal state coverage and the full three-family
inventory shape without OCR.

The bridge also failed closed during startup: before `Main.Instance` existed it
reported an unavailable observation, and before a run initialized all scalar
state it rejected the snapshot rather than emitting plausible defaults. The
initial four-second IL2CPP stabilization delay means the current host client
must allow more than four seconds for its first handshake; subsequent snapshots
are emitted at the configured 250 ms cadence. Startup-state modeling and a
shorter connection path remain implementation work.

Cleanup was verified after a clean reboot. The temporary overlay and bridge
files were absent, `libunity.so` again matched its original SHA-256
`ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
and Package Manager still reported version 29.0.3, version code 1199, and
`installerPackageName=com.android.vending`. No emulator was left running.

The read-only bridge emits versioned observations over an ADB-forwarded local
socket. Wave start, state changes, purchase result, death, and stall/invalid
outcomes remain the target protocol once commands are added. Read-only IL2CPP
access and complete inventory decoding are now proven. Main-thread command
dispatch, earned-cash purchase confirmation, active-run event cadence, reset
handling, and accelerated-time equivalence are not yet proven and must fail
closed until they are. The intended action path queues a semantic command,
executes it on Unity's main thread, sets the game's own upgrade selection, and
invokes the corresponding game method so cost and validity remain owned by the
game.

The implementation sequence resulting from this audit is:

1. Record a product/architecture decision that introduces a separate
   `instrumented-training` profile without weakening official normal-speed
   evaluation.
2. Build the small versioned ARM64 bridge and host socket protocol; expose exact
   lifecycle, wave, cash, health, upgrade-cost/level, and terminal observations.
3. Add a Unity-main-thread command queue for `WAIT` and every supported semantic
   in-run purchase, with before/after confirmation and explicit rejection.
4. Validate normal death/reset and deterministic scripted episodes against the
   existing visible controller, retaining sparse pixels as a watchdog rather
   than decision-frequency OCR.
5. Sweep higher `Time.timeScale` values and actor counts. Accept a speed only
   when scripted transition order and outcome distributions remain equivalent to
   the normal-speed official reference; otherwise cap it.
6. Keep exploration-free best-model evaluation and headed watch mode on the
   unchanged official profile. Instrumented training data and checkpoints record
   the exact game hash, bridge version, speed, and actor profile.

Save inspection remains diagnostic only. `playerInfo.dat` is too infrequently
updated for decisions, and neither the save nor cloud/account state is edited.
No root hiding, integrity bypass, purchase/ad/competitive automation, save
editing, or cloud operation is part of the route. Pixels remain a fail-closed
watchdog and the official headed evaluation path; OCR leaves the training
decision loop only after bridge parity passes.

All proprietary APKs, metadata dumps, save bytes, patched binaries, signing keys,
and runtime artifacts used for this audit remained outside the repository in
private temporary storage. Both disposable test AVDs and the canonical emulator
were stopped after the bounded test.

## M1-E005 — Delayed modal-tap account-UI safety failure

**Date:** 2026-09-15
**Status:** 10-attempt retry stopped after 3 valid episodes; 100 gate not started

The fresh 10-consecutive retry stopped after three valid episodes in 1,118.3
seconds. It failed while waiting for the next active run with a supported modal;
the failure frame showed the Home Settings panel. Tower remained foreground,
the runner restored the offline baseline, and the lifecycle log contained no
crash, ANR, LMK, or process death. The 100-consecutive gate was not started.

A bounded four-episode reproduction again reached three valid episodes and then
failed during result-to-home navigation. Its pre-recovery frame showed Settings,
the Account panel, and an unlinked-cloud-save logout warning. No confirmation,
credential, link, logout, or other account action was selected. This proves the
generic top-right modal-close tap can cross a slow scene transition, land on the
Home Settings control, and allow later lifecycle taps to cascade into unsafe
account UI. Merely delaying the close tap by two modal classifications did not
remove the race.

Lifecycle waits no longer send generic modal taps. A modal may disappear without
input and allow the expected state to be observed; otherwise the bounded wait
fails explicitly. Any future automatic dismissal requires a positively
identified, profile-owned modal subtype and a destination-safe action. Private
reports and frames are `/tmp/tower-rl-m1-10-homewait-retry-20260915.json` and
`/tmp/tower-rl-m1-4-home-no-modal-tap-20260915.json`, with corresponding
filtered logcat files. Both failed runs restored the unchanged 6 GiB offline
baseline.

## M1-E004 — Long-gate result-to-home timing failure

**Date:** 2026-09-15
**Status:** 10-attempt gate stopped after 3 valid episodes; bounded fix passed 2/2

The first 10-consecutive run using the corrected 600-second natural-death
deadline stopped after three valid episodes in 823.8 seconds. The explicit
failure was `result-to-home transition did not reach Battle home`, not a device
failure. The failure frame captured immediately after the controller exhausted
three five-second waits was already a valid Battle home frame. Tower remained
foreground, the filtered lifecycle log contained no crash, ANR, LMK, or process
death, and the runner restored the 6 GiB offline baseline. This proves a late
supported scene transition crossing the controller's deadline rather than
navigation to an unknown screen.

The existing bounded result-to-home attempts now allow 12 seconds each. A
two-episode production-path verification then passed 2/2 genuine WAIT-policy
natural deaths in 687.0 seconds and restored the same baseline with airplane
mode enabled and no route. Reports and lifecycle artifacts are private files at
`/tmp/tower-rl-m1-10-long-20260915.json` and
`/tmp/tower-rl-m1-2-home-wait-20260915.json`, with corresponding failure frame
and filtered logcat files. The required 100-consecutive gate was not started.

## M1-E003 — Wave 3 timeout diagnosis and death-layout correction

**Date:** 2026-09-15
**Status:** Runtime stall rejected; production correction passed one live episode

Bounded host-side traces against the 6 GiB workstation snapshot disproved the
working Wave 3 freeze hypothesis. During an offline sparse-capture control, the
screen remained a valid active run while wave advanced 1 → 2 → 3 → 4 → 5 → 6,
cash advanced 80 → 107, and visible health fell from approximately 98% to 29%.
The run reached the genuine Game Stats death panel after approximately 377
seconds. At Wave 3, Tower remained foreground and input-responsive, its process
CPU counters advanced, RSS was approximately 1.42 GiB, the Unity SurfaceView
and buffers remained present, and ActivityManager/logcat contained no Tower
LMK, OOM, ANR, AndroidRuntime fatal exception, or Unity crash. Because this
complete run occurred with airplane mode enabled and no route, no online or
alternate-renderer variant was needed.

The apparent stall had two controller/profile causes. The fixed 120-second
natural-death deadline expired while a healthy run was still progressing. In
addition, a first post-baseline death adds `New Highest Wave!`, moving the Game
Stats HOME-button border below the previously calibrated normal-result
position; that genuine result classified as `supported_modal`. The controller
now allows 600 seconds and samples at the configured one-second decision
interval. The visual profile recognizes both Game Stats button layouts while
retaining the existing generic-modal rejection.

A production-path one-episode WAIT-policy smoke then passed 1/1 in 351.3
seconds, recognized genuine death, and restored the unchanged 6 GiB golden
snapshot. The final probe was valid Battle home with Tower foreground,
airplane mode enabled, and no route. Private evidence is under
`/tmp/tower-rl-wave3-stall-diag-20260914/`,
`/tmp/tower-rl-wave3-passive-control-20260914/`,
`/tmp/tower-rl-wave3-passive-to-death-20260914/`, and
`/tmp/tower-rl-m1-wave3-fix-live-20260915.json`; logs and screenshots remain
outside the repository. No snapshot or renderer configuration changed, and the
100-consecutive M1 gate was not started.

## M1-E002 — Workstation 6 GiB profile repair and reliability failure

**Date:** 2026-09-14
**Status:** Profile/probe passed; M1 reliability gate failed

The workstation has 125 GiB host RAM, with approximately 93 GiB available at
the start of the repair. The exact AVD is `tower_rl_api36_play_x86_64` and its
configuration is under the user-local Android AVD directory. Before changing
it, the 2 GiB configuration and AVD pointer file were copied to the private
user-local Tower-RL state directory. The only AVD configuration change was
`hw.ramSize=2G` to `hw.ramSize=6144`; the running guest subsequently reported
6,072,056 kB total RAM. The previous snapshot was retained unchanged.

The AVD was stopped through the emulator console and cold-booted with the
pinned `lavapipe` renderer, which selected llvmpipe Vulkan and ANGLE/Swangle
GLES. The official Play-installed 29.0.3 app required ordinary connectivity to
pass its offline startup panel. No sign-in, credential entry, legal acceptance,
purchase, or advertisement was automated. After reaching Battle home,
connectivity was disabled again; airplane mode was enabled and the guest route
table was empty.

The visible baseline remained Tier 1, highest wave 2, 55 coins, 0 gems, x1.00
total coin bonus, and Labs locked. Snapshot
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_6gb_offline_home_20260914_workstation`
was saved separately from the 2 GiB snapshot. It occupies approximately 6.0
GiB locally. The profile probe and bounded Home → active run → result → restored
Home probe passed offline.

A four-episode natural-death diagnostic passed 4/4 in 164.6 seconds with no
failure and restored the new baseline. The subsequent fresh 100-consecutive
gate failed after seven valid episodes with an explicit `DeviceFailureError`;
the failure frame showed the Pixel launcher and the runner restored the new
baseline. The report is
`/tmp/tower-rl-m1-100-6gb-20260914.json` and remains outside the repository.

Follow-up bounded diagnostics did not reproduce the original 2 GiB LMK kill.
Instead, the game repeatedly remained alive and foreground on a visually frozen
Wave 3 active-run frame until the 120-second natural-death deadline. At these
failures Tower RSS was approximately 1.35–2.28 GiB, guest available memory was
approximately 2.15–3.12 GiB, and logs contained no Tower LMK kill, OOM, ANR,
AndroidRuntime fatal exception, or Unity crash. The stall reproduced both from
a freshly launched 6 GiB snapshot and after a cold launch without loading a
snapshot. Reducing death-poll capture frequency from 250 ms to the configured
one-second interval also reproduced the stall, so no controller change was
made from that rejected hypothesis.

The 6 GiB profile therefore fixes the proven guest-capacity deficiency but is
not reliability-qualified. M1 remains open: the complete gate has 7 valid
episodes followed by one `DEVICE_FAILURE`, and the subsequent diagnostic
taxonomy is an active-run stall/timeout. All failure artifacts and lifecycle
logs are private `/tmp` files, and the emulator is left at the new offline
baseline.

## M1-E001 — Workstation actor control slice

**Date:** 2026-09-14
**Status:** In progress; M1 gate remains open

The first real-game controller slice was exercised against the Play-installed
29.0.3 package on the fixed `lavapipe`/ANGLE software-rendered profile. It
restored the golden Tier-1 home snapshot, started a run, extracted wave/cash/
health plus the visible upgrade cards, and recognized `WAIT` and all five
supported purchase actions (`BUY_HEALTH`, `BUY_DAMAGE`, `BUY_ATTACK_SPEED`,
`BUY_CRITICAL_CHANCE`, and `BUY_CRITICAL_FACTOR`). A real critical-chance tap
was confirmed by the observed level/cash transition. Snapshot recovery returned
the emulator to the golden home baseline after the probe.

The implementation is fail-closed on unknown frames and records frame-backed
readings. OCR uses bounded retries because Unity text recognition was
intermittent under the software renderer; the retry path is intentionally local
to visible numeric fields. Focused checks pass: ruff, mypy, and 11 unit tests.

The required M1 evidence is not complete: normal death/result reset and 100
consecutive valid scripted episodes still need to be run and recorded. No
permanent game state was intentionally changed; the workstation AVD is left at
the golden snapshot.

The repeatable gate runner is `scripts/m1_reliability.py`; it checks the complete
purchase action mask, follows a deterministic WAIT policy to genuine game
death, starts the next run through the result/home path, and restores the golden
snapshot in its finalizer. Its output is an explicitly supplied local path and
must remain outside the public repository.

The first 100-attempt run stopped after one valid episode because the result
screen exposed a supported wave-information modal during the result wait. After
that handling was added, a three-attempt run stopped after one valid episode
because the same transient modal appeared during the Home wait. These are
classified UI states, not silent failures; the controller now dismisses the
profile-known modal while waiting for either RESULT or HOME. The 100-episode
gate must be rerun after this fix.

A subsequent gate attempt also encountered a transient `adb: device ... not
found` during the first episode's menu tap. The runner recovered the snapshot;
the ADB adapter now retries bounded transport errors (`not found`, `offline`,
`closed`, or `no devices`) before classifying the attempt as failed. A clean
two-episode run passed after this change, and the 100-episode gate is running
again.

The next three-episode run completed two episodes, then failed when the third
end-run confirmation remained in Active Run. End-run now retries the complete
profile-owned menu/confirmation sequence up to three times and still requires a
classified RESULT frame. The short reliability run should be repeated before
starting the 100-episode gate again.

The reliability runner now also settles for bounded intervals after starting
and resetting runs, so rapid Unity scene transitions are not mistaken for a
ready control surface.

The settle-adjusted three-episode run still completed two valid episodes, then
failed to reach RESULT on the third end-run attempt despite the bounded retry
sequence. This remains an open M1 reliability issue; the actor is not yet
approved for the 100-episode gate. The emulator was restored to the golden
baseline after the failure.

After extending the RESULT wait and adding episode-boundary settling, a fresh
three-episode run passed: all three episodes had the complete action mask,
reached RESULT, reset through the result/home path, and completed with the
golden baseline restored. This is a short reliability check; it does not yet
replace the required 100-episode M1 gate.

The captured result screenshot confirms that the return control is the stable
`HOME` button at the pinned lower-right result-panel location. The controller
now lets the result panel settle for two seconds before retrying that control;
the next short-gate run must verify this timing adjustment.

The diagnostic failure frame exposed the underlying false boundary: a generic
Attack Speed information modal had matched the old broad RESULT heuristic. The
classifier now requires the distinctive Game Stats HOME-button border for
RESULT and classifies other overlays as MODAL. The reliability runner now uses
the deterministic WAIT policy until natural death instead of forcing End Round.
A two-episode natural-death run passed with both valid episodes and the golden
baseline restored. The full 100-episode gate remains outstanding.

The probe restore path was hardened to wait for a valid Battle-home frame after
snapshot load rather than relying on a fixed two-second delay. Final restore
verification passed with airplane mode enabled, no route, foreground The Tower,
and screen `battle_home_tier_1`.

## M0-E001 — Development-host and XAPK characterization

**Date:** 2026-09-14
**Status:** Superseded by M0-E002 after first-run consent
**Purpose:** Establish whether the supplied XAPK can be analyzed, installed, and
launched on the single-device development host.

### Host evidence

- macOS 26.6.2 on Apple M2 Pro (`arm64`)
- 10 physical/logical CPU cores reported
- 16 GiB system memory
- Apple hardware virtualization available
- approximately 88 GiB free before Android SDK installation
- Python, `uv`, and Java present
- Android SDK tools were initially absent

This is the single-device development host. It is not the later 28 GB/RTX 4090
training workstation, whose OS, CPU, virtualization, and storage remain to be
characterized.

### Package evidence

The ignored local archive was inspected without modifying or committing it.

- Archive SHA-256:
  `6496e4f07904723c190c9728e47e621da9bdb40d24b66e1800af1f85211bf0c0`
- Package: `com.TechTreeGames.TheTower`
- Version: `29.0.1` (`versionCode` 1178)
- Minimum SDK: 27
- Target SDK: 36
- Required graphics API: OpenGL ES 3.0
- Launch activity: `com.unity3d.player.UnityPlayerActivity`
- Native ABI supplied: `arm64-v8a`
- All four APKs verify under APK Signature Schemes v2 and v3 with one common
  signing certificate (`SHA-256 b6c646d31fc34415445c6901450fe0a6690d413f9ef67af5ca5a64ce4ae2ee52`)

The archive contains one base APK, its required ARM64 configuration split, a
`gpdeku` install-time feature split, and that feature's ARM64 configuration. All
four share the same package/version identity and form the tested installation
set. Exact safe metadata and per-file checksums are in
`docs/environment-profile.yaml`.

### Android setup and result

Installed:

- Android command-line tools 15.8 / build 15859902
- Android Emulator 37.1.11
- Platform Tools 37.0.1
- Android API 36 platform
- Google APIs API 36 ARM64 system image

Created isolated AVD `tower_rl_api36_arm64` from the Pixel 2 device definition.
The guest reported API 36 and ABI `arm64-v8a`.

The four APKs installed atomically with `adb install-multiple`. Android reported
the expected package, version, SDK range, and primary ABI. A cold launch completed
successfully in approximately 1.6 seconds according to Activity Manager and
reached the real game's first-run EULA/privacy screen. The first captured frame
was 1080×1920 portrait and showed Android's one-time immersive-mode notice above
the game.

### Safety and retained artifacts

- No APK or extracted game file was written into a tracked repository path.
- Temporary extracted APKs and the first screenshot remain outside the repository.
- The screenshot is not suitable as a public test fixture and is not committed.
- The local XAPK remains ignored by Git.

### Blocker and next validation

Accepting the game's EULA/privacy policy is a user-owned legal interaction. No
automation accepted it. The user later completed that step; initialization then
stalled as recorded in M0-E002.

M0 is not complete: the Tier-1 manual-start, fixed baseline, device profile, and
production-host parts of the gate remain open.

## M0-E002 — Google APIs image purchaser stall

**Date:** 2026-09-14
**Status:** Both anonymous emulator trials blocked at purchaser initialization
**Purpose:** Diagnose why the game stopped advancing after first-run consent.

### Observation

The game's visible breadcrumb panel showed successful Firebase initialization,
GDPR acceptance, anonymous PlayFab account creation, cloud-save load, session
creation, remote-settings load, and minimum-version check. The final breadcrumb
remained `Initializing Purchaser` for more than six minutes.

The guest network was connected and Android marked it `INTERNET`, `VALIDATED`,
and unrestricted. The app remained foreground and did not crash.

### Device evidence

The API 36 `google_apis` image contained:

- Google Play services and Google Services Framework;
- a minimal `com.android.vending` version 1.8 package;
- no service resolving
  `com.android.vending.billing.InAppBillingService.BIND`.

The app's log repeatedly reported BillingClient failures while
`Purchaser.Initialize` was active, including attempts to unbind a service that
had not registered. This matches the visible stall and rejects the plain Google
APIs image for The Tower 29.0.1.

### Corrective trial

The incompatible AVD was stopped without deleting its user data. A separate
`tower_rl_api36_play_arm64` AVD was created with the API 36 ARM64 Google Play
image. Its Play Store package resolves the required billing service. The same
four-split XAPK set installed and cold-launched successfully. The user accepted
the EULA/privacy screen on this replacement AVD.

After consent, the replacement also remained at purchaser initialization. Its
log reported `In-app billing API version 3 is not supported on this device`,
followed by billing-service disconnect/death messages. The Play Store updated
itself from version 45.3.21 to 53.0.27 during the trial, but a clean app restart
after that update produced the same result. No Google account was configured on
the AVD, and the game was installed by the ADB shell rather than acquired by that
account through Google Play.

The replacement initially fell back to software graphics because the Mac had
less than the emulator's requested free-memory threshold at launch. Renderer
performance is therefore still uncharacterized and cannot yet support a profile
selection claim.

### Next validation

The next smallest authorized compatibility test is user-owned Play Store
provisioning:

1. sign into the Play Store with a dedicated user-owned Google account;
2. ensure that account can legitimately acquire The Tower from its production
   listing, without making an in-app purchase;
3. relaunch the installed, correctly signed game and observe purchaser setup;
4. if it succeeds, confirm the home/tutorial flow and preserve a named
   post-consent state at a stable screen;
5. if it fails, test a lower supported Play Store API image and record the result.

Do not falsify the installer identity, bypass Play licensing, or automate account
credentials. Whether sign-in/entitlement resolves the error remains a hypothesis,
not a completed compatibility result.

## M0-E003 — Play entitlement and offline snapshot-resume trial

**Date:** 2026-09-14
**Status:** One offline resumed retry cycle passed; offline cold launch failed
**Purpose:** Determine whether a production Play installation resolves purchaser
initialization and whether a post-consent AVD snapshot can support a network-
isolated Tier-1 sample.

### Play installation result

The user signed into Google Play manually. No credentials were automated or
captured. The ADB-installed copy was removed, and the game was acquired and
installed from its production Play listing. Android reported:

- package `com.TechTreeGames.TheTower`;
- version `29.0.3` (`versionCode=1199`);
- installer `com.android.vending`;
- ARM64 base, configuration, `gpdeku`, and `gpdeku` ARM64 configuration splits.

This installation completed purchaser initialization and entered a real Tier-1
run. With no actions, the run ended at wave 2 and displayed the normal result
screen. This confirms that the earlier purchaser stall was specific to the
unentitled ADB-installed path on the tested profile, not a general inability of
the API 36 Play Store image to run the game.

### Recovery snapshots

At the stable wave-2 result screen, the following named emulator snapshots were
created in the local AVD storage, outside the repository:

- `tower_post_consent_play_29_0_3_online_20260914`;
- `tower_post_consent_play_29_0_3_offline_running_20260914`.

The second snapshot was created only after Android airplane mode was enabled,
Wi-Fi and mobile data were disabled, and an external ping failed with `Network is
unreachable`. The running Unity process and result screen survived the transition.

### Offline behavior

A force-stop followed by an offline launcher start did **not** reopen the game.
Google Play displayed its app/licensing panel and the game process exited. A
durable offline cold-start path is therefore not established.

Restoring the online snapshot and immediately disabling connectivity preserved
the already-running game. From that state, a diagnostic tap on `Retry` dismissed
the result panel, ran another short Tier-1 attempt, and returned to the result
panel while airplane mode remained enabled and no route was available. The
untouched offline-running snapshot was restored after the trial.

### Interpretation and remaining gate

This is evidence for exactly one **snapshot-resumed, network-isolated** retry
cycle. It does not establish that cached Play entitlement or sessions survive
long delays, that a force-stopped actor recovers offline, that cloned actors have
independent identities or random streams, or that parallel execution is safe and
reliable. The Play-installed 29.0.3 package is now the runtime candidate; the
locally supplied 29.0.1 XAPK remains characterization input and is not the
candidate runtime package.

M0 remains open pending a fixed baseline, home-to-Tier-1 navigation inventory,
renderer validation, offline snapshot aging/recovery tests, and a controlled
two-actor isolation experiment.

## M0-E004 — Initial Tier-1 golden-baseline candidate

**Date:** 2026-09-14
**Status:** Snapshot created and transport-restored; visual restore gate pending
**Purpose:** Establish the first fixed, no-permanent-spending Tier-1 baseline and
test named-snapshot recovery.

### Baseline establishment

Starting from the offline-running post-consent snapshot, the result screen was
navigated to Battle home. The first-run tutorial required opening Workshop and
claiming an unavoidable 50-coin onboarding grant. The grant increased the visible
unspent balance from 3 to 53 coins. No coins were spent and no permanent combat
upgrade was purchased.

The visible fixed state was inventoried as:

- Tier 1 selected, highest wave 2, total coin bonus `x1.00`;
- 53 unspent coins and 0 gems;
- Workshop attack values: damage 3, attack speed 1.00, critical chance 1.00%,
  critical factor x1.20, with range upgrades locked;
- Workshop defense values: health 5 and health regeneration 0.00/sec, with
  additional defense upgrades locked;
- utility cash bonuses locked;
- Ultimate Weapons informational screen observed, with no weapon selected or
  purchased;
- all post-Workshop progression tabs visibly locked, so no Lab research can be
  active.

The device profile was 1080x1920, 420 dpi, portrait rotation 0, `en-US`, and
approximately 60 Hz. Android airplane mode remained enabled and no external
route was available. SHA-256 hashes of the four Play-installed 29.0.3 APK splits
were recorded in `environment-profile.yaml`; no package bytes were copied into
the repository.

### Snapshot and restore result

The named local snapshot
`tower_golden_t1_v1_play_29_0_3_offline_running_20260914` was created at the
Battle home screen with Tier 1 selected. The emulator reported an approximate
snapshot size of 1.5 GiB. It is stored only in ignored AVD data.

Loading the snapshot restored the game process in the foreground, preserved
airplane mode, and preserved the absence of an external route. The ADB transport
then became unstable when a post-restore screenshot was requested. Reconnecting
ADB restored command access, but another screenshot request reproduced the
disconnect. Earlier captures in the same session succeeded before this restore.

### Interpretation

This artifact is a **golden-baseline candidate**, not yet an admitted M1 recovery
baseline. Snapshot load and nonvisual invariants passed, but the required visual
fingerprint verification did not. The current software-rendered emulator profile
also emitted gfxstream/color-buffer errors and had already shown transient ADB
disconnects. Renderer/capture stability must be fixed or a different validated
graphics profile selected before this snapshot can be promoted to the trusted
golden baseline.

## M0-E005 — Pinned-renderer golden snapshot validation

**Date:** 2026-09-14
**Status:** Passed for the single-device M0 recovery check
**Purpose:** Rebuild the golden snapshot under a pinned renderer and verify that
the visible baseline survives an in-place snapshot restore.

### Procedure and evidence

The AVD was restarted with `-gpu lavapipe`, which resolves to Lavapipe Vulkan and
ANGLE/SwiftShader (`swangle`) GLES. The earlier 1.5-GiB snapshot was not
compatible with alternate renderer combinations; it was retained unchanged. With
connectivity temporarily enabled, the Play-installed 29.0.3 game was relaunched
and reached the same Battle home state. Connectivity was then disabled with
airplane mode, Wi-Fi, and mobile data disabled; an external ping returned
`Network is unreachable`.

A replacement snapshot,
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914`, was
created at Battle home. Its current local AVD footprint is approximately 2.9 GiB
(`ram.bin` plus renderer textures); this account-bearing state is not portable
repository data.

To test restore, the device was navigated away from the baseline and the named
snapshot was loaded in place. After restore:

- the game process was alive and foreground;
- airplane mode remained enabled;
- external network remained unreachable;
- `adb exec-out screencap -p` succeeded without the earlier pull/transport
  failure;
- the restored frame visibly showed Battle home, Tier 1 selected, highest wave 2,
  53 unspent coins, 0 gems, and `x1.00` total coin bonus;
- the representative restored-frame SHA-256 was
  `32ffeb8f5dd1a2d5b5da03b0e694059debe0f49c4ae46162ad2b9e2f44084a25`.

The screenshot is temporary evidence only and is not stored in the repository.

### Interpretation

The pinned-renderer snapshot passes the current single-device golden-baseline
restore gate. It is the canonical `tower-t1-initial-v1` artifact for subsequent
controller work. It remains a local account-bearing AVD snapshot and must not be
copied into Git or treated as server-state rewind. Offline cold launch after a
force-stop remains unsupported; recovery must resume the running snapshot or
temporarily restore connectivity. Parallel identity/randomness isolation and the
M1/M2 reliability gates remain outstanding.

## M0-E006 — Bounded semantic navigation probe

**Date:** 2026-09-14
**Status:** Passed for the single-device smoke path
**Purpose:** Verify that the pinned visual profile can drive the game's safe
Home → Tier 1 → result → Home path without selecting an upgrade or meta action.

The new `tower-rl probe` command captures PNG frames with `adb exec-out`, checks
the foreground package, airplane mode, external route, fixed 1080×1920 frame
size, and a conservative renderer-specific screen profile. It recognizes Home,
active run, the transient Wave Info modal, result, and unknown states. Unknown
or contradictory states fail closed.

The bounded sequence used only the calibrated controller operations: tap Battle,
open the in-run menu, select End Round, confirm, close a possible Wave Info
modal, and tap Home. It completed successfully while offline. The command then
restored the canonical golden snapshot, and a follow-up probe returned the exact
golden representative frame SHA-256
`32ffeb8f5dd1a2d5b5da03b0e694059debe0f49c4ae46162ad2b9e2f44084a25` with the
game foreground and no external route.

Starting a run can award ordinary run coins before End Round is processed, so
navigation probes must always use `--restore-snapshot` when run against the
account-bearing golden device. The probe exposes no purchase or permanent-
progression action. This is a smoke/navigation gate, not the M2 reliability
gate; repeated episodes, death detection, full observation extraction, and
two-actor isolation remain outstanding.

## M0-E007 — Workstation artifact portability assessment

**Date:** 2026-09-14
**Status:** Handoff documented; cross-host transfer intentionally not attempted
**Purpose:** Make the validated Mac setup reproducible on the future RTX 4090
workstation without publishing account-bearing state.

The canonical snapshot resides in the Mac-local AVD snapshot directory under
`$ANDROID_AVD_HOME` (or `$HOME/.android/avd`), in the named snapshot directory
recorded in `docs/workstation-handoff.md`. Its current on-disk footprint is
approximately 2.9 GiB, including `ram.bin`, renderer textures, hardware metadata,
and the snapshot manifest. This is Android user/account state and remains outside
Git.

The workstation ABI, host OS, emulator version, and renderer are not yet known.
Because the current artifact is ARM64 and renderer-pinned to Lavapipe/Swangle,
copying it to an x86_64 RTX 4090 host would not be a supported bootstrap path.
The handoff therefore specifies Play reprovisioning, manual baseline setup, a
new workstation-local snapshot, and probe validation before actor work. No
proprietary package bytes, credentials, screenshots, or emulator data were added
to the repository.

## M0-E008 — Workstation-independent environment contract scaffold

**Date:** 2026-09-14
**Status:** Unit and contract checks passed; real-device extraction remains open
**Purpose:** Make the Android-independent portion of the environment ready for
workstation integration.

Added the versioned `observation-v1` and `run-action-v1` schemas, typed action
outcomes and termination reasons, fail-closed temporal/logical observation
validation, and the `AndroidDevice` protocol boundary. Added a sanitized local
configuration example and contract tests. The policy action set remains semantic
(`WAIT`, `BUY_HEALTH`, and observed attack upgrades); coordinates and navigation
are confined to the controller/device layer.

This scaffold deliberately does not claim M1: there is no full OCR/structured
extractor, purchase confirmation, death detector, or 100-episode reliability
run yet. Those components can now be integrated on the workstation without
changing the policy-facing contract.

## M0-E009 — Repository workstation bootstrap helpers

**Date:** 2026-09-14
**Status:** Read-only preflight and idempotent AVD helpers validated on the Mac
**Purpose:** Move repeatable host setup into the repository while keeping
account-bearing actions manual.

Added `scripts/workstation_preflight.py` for host/SDK/tool/AVD inventory,
`scripts/create_avd.sh` for safe idempotent API-image AVD creation, and
`scripts/launch_avd.sh` for renderer/snapshot launch. Added
`configs/workstation.example.yaml` and documented the complete handoff flow.
The helpers do not install the game, sign into Play, accept legal consent, or
copy snapshots. The Mac preflight reports the expected ARM64/API 36 tools and
AVDs; all repository checks remain green.

## M0-E010 — Domain-driven package boundaries

**Date:** 2026-09-14
**Status:** Refactor verified; behavior preserved
**Purpose:** Keep game concepts independent from Android and process details as
the workstation integration grows.

The canonical domain model now lives under `tower_rl.domain`, ports under
`tower_rl.ports`, use-case orchestration under `tower_rl.application`, and ADB
adapters under `tower_rl.infrastructure`. The CLI composes these layers. Root
imports remain compatibility shims only. Unit, lint, type, and live baseline
probe checks pass after the refactor.

## M0-E011 — RTX workstation characterization

**Date:** 2026-09-14
**Status:** Host gate passed; Android provisioning remains open
**Purpose:** Characterize the training workstation before selecting an Android
backend or actor count.

### Observed host

- Ubuntu 26.04.1 LTS, kernel `7.0.0-31-generic`, x86_64;
- Intel Core i9-14900, 24 physical cores / 32 logical CPUs;
- 125 GiB RAM and approximately 1.1 TiB free on the workspace filesystem;
- NVIDIA RTX 4090, driver `595.91.07`, CUDA `13.2`;
- Intel VT-x is enabled; `/dev/kvm` is readable and writable by the current user
  through an explicit device ACL.

### Missing prerequisites

The preflight found no Android SDK root, `adb`, emulator, `sdkmanager`, or
`avdmanager`. No Android device or AVD is currently connected or discoverable.
The expected local XAPK is also absent from the repository's ignored `local/`
directory and the searched local paths, so package metadata cannot be freshly
verified on this host yet.

### Verification

`uv sync --all-groups`, `uv run ruff check .`, `uv run mypy`, and `uv run pytest`
all pass (11 tests). The read-only preflight was run through
`scripts/workstation_preflight.py --json`; the CLI doctor independently reports
the same missing Android tools and XAPK prerequisites.

### Next smallest action

Provision Android command-line tools/emulator and the API 36 Google Play x86_64
image, then place the authorized reference archive at the documented ignored
path (or provide its actual path). Recreate the Play-installed game baseline
manually on this host; the ARM64 Mac snapshot is not portable or supported here.
M0 remains open until one workstation AVD launches the real game, Tier 1 is
manually started, and a new local baseline snapshot passes the probe.

## M0-E012 — Workstation Android stack provisioning

**Date:** 2026-09-14
**Status:** Toolchain and AVD ready; Play provisioning requires user action
**Purpose:** Prepare a compatible Android execution target on the RTX
workstation without touching account state.

Installed under the user's local data directories, without sudo:

- Eclipse Temurin JDK 17.0.20.1;
- Android command-line tools 15.0;
- platform-tools 37.0.1, emulator 37.1.11, API 36 platform/build tools;
- API 36 `google_apis_playstore` x86_64 system image.

Created and booted `tower_rl_api36_play_x86_64` from the Pixel 2 definition.
The device reports x86_64, 1080×1920, 420 dpi, and `sys.boot_completed=1`.
The emulator selected the RTX 4090 through the host renderer; renderer and game
stability remain unvalidated.

At the time of this provisioning entry the Play Store was unauthenticated;
subsequent user-owned sign-in, official game installation/entitlement, legal
consent, and first-run setup are recorded in M0-E013.

## M0-E013 — Workstation renderer and golden-baseline restore

**Date:** 2026-09-14
**Status:** Passed for the workstation single-device M0 recovery check
**Purpose:** Validate a cleanly rendered real game and establish a separate
account-bearing baseline on the x86_64 workstation.

The initial `host` renderer produced visibly corrupted Unity text and UI. The
same AVD was relaunched with pinned `lavapipe`, which selected the llvmpipe
Vulkan device and ANGLE/Swangle GLES. The Play-installed The Tower 29.0.3
(`versionCode=1199`) then rendered cleanly at 1080×1920, 420 dpi, portrait.

At the stable Battle home screen, Tier 1 was selected, highest wave was 2,
unspent coins were 55, gems were 0, total coin bonus was x1.00, and Labs were
locked. Networking was disabled inside the guest; airplane mode was 1 and
`ip route` was empty. The repository probe passed with no reasons.

Snapshot `tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation`
was saved in the local AVD storage. Loading it back into the running emulator
passed: the game remained foreground, the probe passed again, and the restored
PNG SHA-256 exactly matched the pre-restore frame
(`f00ab790b75c44862baa39f84f2ebf0767d600e8af9a6939def1266816f1f643`). The
snapshot directory occupies approximately 2.9 GiB and is ignored local state.

This validates the workstation baseline/recovery artifact, not M1. The XAPK
bytes are still absent from this checkout; previously recorded compatibility
metadata remains in `environment-profile.yaml`, while fresh local XAPK
inspection is still optional reference verification.

## M0-E014 — Workstation bounded Tier-1 navigation probe

**Date:** 2026-09-14
**Status:** Passed; baseline restored
**Purpose:** Verify the safe workstation Home → Tier 1 → result → Home path
without buying upgrades or invoking permanent-progression controls.

`uv run tower-rl probe --serial emulator-5554 --navigate --restore-snapshot
tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation`
completed with valid initial Battle home, active Tier-1 run, result, and final
Battle-home observations. Airplane mode remained enabled and the external route
remained absent throughout. The command restored the workstation snapshot after
the probe; a follow-up probe returned the canonical frame hash
`f00ab790b75c44862baa39f84f2ebf0767d600e8af9a6939def1266816f1f643`.

This closes the workstation-specific M0 navigation/start smoke check. The
available in-run action inventory, speed profile, offline aging behavior, and
multi-actor isolation remain open before M1/M2 work.
