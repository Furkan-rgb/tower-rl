# Tower-RL — Experiments and Evidence

This document records feasibility work, benchmarks, failed approaches, and
contrary evidence. An entry records what was observed; it does not advance a
milestone unless the corresponding gate in `task.md` is satisfied.

Do not add proprietary package bytes, extracted assets, account/save state,
personal screenshots, bulk logs, replay, or model artifacts.

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
