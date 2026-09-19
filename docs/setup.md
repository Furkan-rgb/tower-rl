# Tower-RL — Local Setup

## 1. Current support status

The validated single-device M0 baseline is:

- Apple M2 Pro host with hardware virtualization;
- Android Emulator 37.1.11;
- API 36 Google Play ARM64 system image;
- Pixel 2 device definition at 1080×1920 portrait;
- The Tower 29.0.3 (`versionCode=1199`) acquired through Google Play on the
  user-provisioned test account. The local 29.0.1 XAPK is retained for metadata
  and compatibility inspection only; it is not the runtime installation.

The operator completed first-run legal consent and signed in through Play. The
game reaches the Tier-1 Battle home, and the canonical golden snapshot
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914` has been
restored and visually verified with networking disabled. The snapshot is a
recoverable running-state baseline; a force-stopped offline cold launch still
re-enters Play licensing and is not supported. The production 28 GB/RTX 4090
workstation has not been characterized.

For moving this work to that workstation, use
[`workstation-handoff.md`](workstation-handoff.md). The account-bearing AVD
snapshot is machine-local and must be recreated and revalidated on a different
host.

## 2. Repository environment

From the repository root:

```text
uv sync --all-groups
uv run ruff check .
uv run mypy
uv run pytest
```

Python 3.12 is selected by the project metadata. The proprietary XAPK must remain
under `local/`, which is ignored by Git.

There is no configuration file. Every entry point under `scripts/` is configured
by its own command-line arguments, and the private bridge build directory is
named by the `TOWER_BRIDGE_BUILD_DIR` environment variable. Keep credentials,
account state and snapshots out of the repository.

### Where project state lives

Everything this project writes lives under `state/` at the repository root:

```text
state/bridge/<sha256>/   one installed bridge build, named for its own digest
state/bridge/current     symlink to the bridge that is deployed
state/bridge/config/     the private build configuration (never committed)
state/runs/              training runs, checkpoints and reports
state/mlflow.db          the MLflow store, with artifacts in state/mlartifacts/
state/records/           evaluation records: actors, arms, episodes, selection
state/recordings/        spectate recordings and their per-run records
```

`state/` is git-ignored in full — the artifacts in it are far above GitHub's
file limit and this repository is public — and no project state is kept anywhere
else on the host. The location is resolved from the package's own file location
(`tower_rl.environment.project_state.state_directory`), never from the current
directory, so an entry point started from anywhere finds the same tree. There is
no environment variable that moves it.

A host that still has the old `~/.local/state/tower-rl` tree brings it in once,
with no emulator running:

```text
uv run python scripts/migrate_state.py
```

It renames each entry into `state/`, re-points `state/bridge/current` at its
sibling relatively, prints what moved where, and leaves the old location absent.

### Building and installing the bridge

The bridge is ARM64 source under `native/tower_bridge/` built against the
Android NDK (`~/.local/share/android-sdk/ndk/29.0.14206865` on the workstation).
Its compatibility identity — package version and version code, official signer,
original `libunity.so` and `libil2cpp.so` digests, profile id — is private and is
supplied as CMake cache values from `state/bridge/config/profile.cmake`, which is
not committed. **The reference for those values is the `CMakeCache.txt` beside
the installed bridge**: it records exactly what the deployed artifact was
configured with, so a lost `profile.cmake` is rewritten from
`state/bridge/current/CMakeCache.txt` (`grep TOWER_BRIDGE_ state/bridge/current/CMakeCache.txt`)
rather than guessed at.

```text
build=$(mktemp -d)
cmake -S native/tower_bridge -B "$build" -C state/bridge/config/profile.cmake \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-35
cmake --build "$build"
sha=$(sha256sum "$build/libtower_bridge.so" | cut -d' ' -f1)
install -D -t state/bridge/$sha \
  "$build/libtower_bridge.so" "$build/libunity-bridge.so" "$build/CMakeCache.txt"
sha256sum state/bridge/$sha/libtower_bridge.so    # must print $sha
ln -sfn $sha state/bridge/current
```

Copy, verify, *then* move the pointer, in that order. The patched
`libunity-bridge.so` — an added `DT_NEEDED` entry and nothing else — is installed
beside it, and `CMakeCache.txt` goes with them because the handshake identity is
read back out of it rather than hard-coded in Python.

The digest check is over the *installed* artifact against the directory name it
is filed under, which is what makes the layout self-verifying and is what
`bridge.installed_bridge_directory` refuses on. It is not a claim that a rebuild
reproduces an earlier digest: a different NDK, build path or configuration
produces a different `libtower_bridge.so`, which is a new directory and a new
`current`, not a failure. An unconfigured build (the `unconfigured` defaults in
`CMakeLists.txt`) compiles but answers a compatibility error instead of a
handshake.

`TOWER_BRIDGE_BUILD_DIR` overrides all of this and deploys straight out of a
build tree, which is how a bridge under development is run; every ordinary run
leaves it unset and takes the installed one.

## 3. Android tooling on Apple silicon

The M0 development host used Homebrew's Android command-line tools:

```text
brew install --cask android-commandlinetools
sdkmanager --licenses
sdkmanager \
  "platform-tools" \
  "emulator" \
  "build-tools;36.0.0" \
  "platforms;android-36" \
  "system-images;android-36;google_apis_playstore;arm64-v8a"
```

Review and accept Android's SDK licenses interactively. On this Homebrew setup,
the SDK root is `/opt/homebrew/share/android-commandlinetools`. The relevant tools
are not necessarily all linked into `PATH`:

```text
export ANDROID_SDK_ROOT=/opt/homebrew/share/android-commandlinetools
export PATH="$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
```

Create the current candidate AVD once:

```text
printf 'no\n' | avdmanager create avd \
  --name tower_rl_api36_play_arm64 \
  --package 'system-images;android-36;google_apis_playstore;arm64-v8a' \
  --device pixel_2
```

Do not place AVD data, snapshots, or Android user data in the repository.

## 4. Inspect and boot

Run the metadata-only checks before installation:

There is no `tower-rl` console script. The checks are
`tower_rl.doctor.run_doctor(xapk, serial)`, rendered with
`tower_rl.doctor.render_json`; host tooling alone is
`uv run python scripts/workstation_preflight.py`.

Boot the candidate visibly for first-run characterization:

```text
emulator @tower_rl_api36_play_arm64 \
  -gpu host \
  -no-audio \
  -no-boot-anim \
  -no-snapshot
```

Wait for `adb shell getprop sys.boot_completed` to return `1`.

### Which renderer

`-gpu host` is the standing renderer for every fleet, training and collection
run. Renderer equivalence is `M1B-E026` Gate B, five episodes each: game-time
ratio 1.0108 (host) against 1.0118 (lavapipe), decisions per wave 21.0 against
20.73, mean final wave 6.2 against 6.0; Gate C then ran 25/25 valid episodes
unattended on host with every health counter at zero, and the conclusion there
was to adopt `-gpu host` as the mandatory training renderer.

The cost is that it cannot snapshot. The emulator refuses to save a snapshot of
a Vulkan app under `-gpu host` (`KO: Snapshot save is skipped. Reason:
UNSUPPORTED_VK_APP`), so under `-gpu host` every instance takes the cold
bring-up path by name: `prepare_pinned_snapshot` in
`src/tower_rl/simulation/fleet.py` checks the renderer against
`SNAPSHOT_CAPABLE_RENDERER` (`lavapipe`), prints `renderer 'host' cannot
snapshot; nothing to pin`, and pins nothing.

So `-gpu lavapipe` survives only where a snapshot is actually saved or restored:
`prepare_pinned_snapshot` and `bring_up`'s restore path, the golden-baseline
restore in section 7 below, and manual visual review. The `--renderer` default
in the scripts is still `lavapipe` for that reason; a fleet run passes
`--renderer host` explicitly.

## 5. Optional XAPK metadata/reference inspection

The supplied XAPK is a ZIP archive containing four APKs. It is reference-only
for this baseline; the validated runtime was installed by Google Play. Inspect
its metadata without installing or copying proprietary bytes into the repo:

```text
unzip -l local/the-tower-29-0-1.xapk
```

Confirm the Play-installed running setup with `run_doctor(xapk, serial)` against
the running serial.

The M0 `probe` command and the visual profile it validated no longer exist. The
environment is read through the instrumented bridge instead
(`src/tower_rl/simulation/instrumented_bridge.py`), and no part of the RL loop
reads a pixel. To bring one instance up and inspect it:

```text
uv run python scripts/clone_session.py up --renderer host --cores 4
```

Launch the installed app:

```text
adb -s emulator-5554 shell am start -W \
  -n com.TechTreeGames.TheTower/com.unity3d.player.UnityPlayerActivity
```

## 6. First-run boundary

The game presents its EULA and privacy policy on first launch. The operator must
review those documents and personally decide whether to accept them. Tower-RL
does not automate legal consent, sign-in, purchases, advertisements, or cloud
account setup.

Do not substitute the plain `google_apis` API 36 image for this game version. It
has a minimal Play Store package with no billing service, causing initialization
to stall at `Purchaser.Initialize` even when networking is healthy. This does not
authorize purchases or Play Store sign-in; the compatible image supplies an API
the game expects during startup.

The signed-in Play image supplies the billing service the game expects: the
production listing completed purchaser initialization and reached Tier 1. No
credentials, purchases, advertisements, or account setup are automated by
Tower-RL.

After accepted first-run setup, M0 continues with:

1. inventorying home, tutorial, Tier-select, run, result, and known modal screens;
2. manually starting Tier 1;
3. selecting fixed resolution, density, locale, renderer, and in-game speed;
4. enumerating the account's visible in-run actions;
5. establishing the fixed permanent baseline;
6. capturing only deliberately sanitized recognition fixtures.

## 7. Restore the golden baseline

Use the pinned software renderer when restoring the snapshot. This avoids the
host-renderer/Vulkan feature mismatch observed with `-gpu host` or `-gpu auto`:

```text
emulator @tower_rl_api36_play_arm64 \
  -gpu lavapipe \
  -no-audio \
  -no-boot-anim \
  -snapshot tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914 \
  -no-snapshot-save
```

The restored process should be foreground at the Battle home with Tier 1
selected, 53 coins, 0 gems, x1.00 speed, and Labs still locked. Airplane mode
must remain enabled; verify that no external route exists before collecting
fixtures. Do not treat a force-stopped offline cold launch as a valid reset:
the game's Play licensing path currently requires network access.

## 8. Stop the development emulator

When no longer needed:

```text
adb -s emulator-5554 emu kill
```

Do not delete the AVD after account setup; its user data may be needed to create
the later golden recovery baseline and must remain outside Git.

## 9. Watching an agent

`scripts/spectate.py` is the one path that exists for a human rather than for a
measurement: one clone instance comes up **with a window**, the chosen arm plays
in it, and a terminal panel beside the window shows what the agent is doing.

```text
uv run python scripts/spectate.py --policy checkpoint:<path-to-checkpoint.pt>
```

`--policy` takes the same arms every other runner takes: `scripted`, `random`,
`wait`, or `checkpoint:<path>`.

The default session is **one episode**: the agent plays a single run from wave 1
until the tower dies. `--episodes N` plays N, and `--episodes 0` plays until you
stop it. When the last episode ends the panel holds the final state — the wave
reached, the termination outcome, the last actions — and waits for a keypress
before the instance is torn down, so the death is not the moment the window
disappears.

The panel shows the current episode, wave, cash and health, the last 20 actions
(each upgrade slot bought, or `wait`), episodes played, the running mean final
wave, and decisions per minute.

Keys: `q` stops at the next decision; Ctrl-C does the same, and both keep the
episodes that had already finished. Any key ends the hold at the end, and the
hold ends by itself after `--hold-seconds` either way, so a session nobody came
back to still puts its emulator down. Whatever happens, the bridge and the
emulator are put down through the same teardown path the fleet uses.

`--no-panel` is the log-friendly mode: it prints one line per decision instead
of drawing a terminal panel, which is what you want when the session is
unattended, redirected to a file, or read afterwards rather than watched. There
`--hold-seconds` (default 10) simply waits, since there is no key to press.

**60 Hz is real time.** That is the default and it is the point: one game second
per wall second, the speed the game is actually played at. The fleet runs at 120
Hz to make an advance cheap in wall time, which buys throughput and nothing a
human wants; `--frame-rate-hz 120` is accepted if you want to watch it at that
rate.

**Spectating takes the host to itself.** The script refuses to start while any
emulator is running — `adb devices` non-empty, or a `qemu-system` process
present — and says so. A windowed real-time session must never share a host with
a training run or a measurement, whose throughput is what the host is for.

`--record session.mp4` records the guest screen with `adb shell screenrecord`
and pulls the file back at the end. Android's `screenrecord` stops itself after
**three minutes**, which no flag lifts, so a longer session is recorded as
consecutive numbered chunks (`session-000.mp4`, `session-001.mp4`, …) with about
a second lost at each seam. The recording covers through the death and the hold.

**The guest renders through `-gpu lavapipe` by default**, not the host renderer
the fleet trains on: the host renderer glitches the picture on this machine,
which makes a recording of it useless. `--renderer host` is accepted if you
want the fleet's own renderer instead. The renderer in use is named on the
panel's own title line.

A relative `--record` filename, and the per-run episode JSON `--output-directory`
writes, both land under `state/recordings/` by default — one home for spectate
output, resolved from the package's own location rather than the current
directory. All of `state/` is git-ignored: an mp4 is far above GitHub's file
limit, and this repo is public.

`--output-directory` writes the episodes the session played as the same
per-episode records the fleet writes. Omit it and nothing is kept: the panel is
a view, not a measurement.
