# Tower-RL — Local Setup

## 1. Current support status

The supported host is the RTX 4090 workstation, and the validated
single-device baseline on it is:

- Android Emulator 37.1.11;
- API 36 Google Play x86_64 system image, which the ARM64 game runs on through
  the image's own native-bridge translation;
- Pixel 2 device definition at 1080×1920 portrait;
- The Tower 29.0.3 (`versionCode=1199`) acquired through Google Play on the
  user-provisioned test account. The local 29.0.1 XAPK is retained for metadata
  and compatibility inspection only; it is not the runtime installation.

The operator completed first-run legal consent and signed in through Play. The
game reaches the Tier-1 Battle home, and the canonical golden snapshot
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914` has been
restored and visually verified with networking disabled. The snapshot is a
recoverable running-state baseline; a force-stopped offline cold launch still
re-enters Play licensing and is not supported.

[`workstation-handoff.md`](workstation-handoff.md) holds the workstation's own
state and procedures. The account-bearing AVD snapshot is machine-local and
must be recreated and revalidated on a different host.

## 2. Repository environment

From the repository root:

```text
uv sync --all-groups
uv run ruff check .
uv run mypy
uv run pytest
```

`uv run pytest` enforces a 120 s per-test timeout (`pytest-timeout`, thread
method) so a hang fails loudly instead of running unbounded.

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
state/bridge/builds/     experiment bridge variants, selected only by TOWER_BRIDGE_BUILD_DIR
state/bridge/config/     the private build configuration (never committed)
state/runs/              training runs, checkpoints and reports
state/mlflow.db          the MLflow store, with artifacts in state/mlartifacts/
state/records/           evaluation records: actors, arms, episodes, selection
state/recordings/        spectate recordings and their per-run records
state/logs/              each emulator's own captured output, by serial
```

`state/` is git-ignored in full — the artifacts in it are far above GitHub's
file limit and this repository is public — and no project state is kept anywhere
else on the host. The location is resolved from the package's own file location
(`tower_rl.environment.project_state.state_directory`), never from the current
directory, so an entry point started from anywhere finds the same tree. There is
no environment variable that moves it. A linked git worktree shares this same
`state/`: the resolver follows the worktree's `.git` file back to the main
checkout so a run started from a worktree still finds the one bridge install
and the one set of runs.

A host that still has the old `~/.local/state/tower-rl` tree brings it in once,
with no emulator running:

```text
uv run python scripts/migrate_state.py
```

It renames each entry into `state/`, re-points `state/bridge/current` at its
sibling relatively, writes `state/bridge/config/profile.cmake` out of the
installed build's `CMakeCache.txt` (below), prints what moved where, and leaves
the old location absent.

### Building and installing the bridge

The bridge is ARM64 source under `native/tower_bridge/` built against the
Android NDK (`~/.local/share/android-sdk/ndk/29.0.14206865` on the workstation).
Its compatibility identity — package version and version code, official signer,
original `libunity.so` and `libil2cpp.so` digests, profile id — is private and is
supplied as CMake cache values from `state/bridge/config/profile.cmake`, which is
not committed. **The reference for those values is the `CMakeCache.txt` beside
the installed bridge**: it records exactly what the deployed artifact was
configured with. `scripts/migrate_state.py` writes `profile.cmake` out of it, and
`migrate_state.write_private_build_configuration` rewrites it at any time from
`state/bridge/current/CMakeCache.txt`, so the configuration is never guessed at
and never retyped from a shell history.

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
`bridge.installed_bridge_directory` refuses on. An unconfigured build (the
`unconfigured` defaults in `CMakeLists.txt`) compiles but answers a
compatibility error instead of a handshake.

**The build is reproducible: the same source, NDK and `profile.cmake` produce
the same `libtower_bridge.so`, byte for byte, from any build directory and from
any checkout or worktree.** That is what lets a digest name a *source* rather
than the directory someone happened to build in, which is the whole basis for
confirming the deployed bridge by digest. `native/tower_bridge/CMakeLists.txt`
buys it with `-ffile-prefix-map` and `-ffile-compilation-dir` (no host path
reaches the binary), `-Wl,--strip-all` (the debug sections, which carried the
NDK's own absolute include paths, are dropped; the dynamic symbols a crash stack
can name are kept), and `-Wl,--build-id=sha1` (the build id hashes the output
instead of the path). Checking it is two builds and one comparison:

```text
a=$(mktemp -d); b=$(mktemp -d)/nested/deeper; mkdir -p "$b"
for d in "$a" "$b"; do
  cmake -S native/tower_bridge -B "$d" -C state/bridge/config/profile.cmake \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-35 >/dev/null
  cmake --build "$d" >/dev/null
done
sha256sum "$a/libtower_bridge.so" "$b/libtower_bridge.so"   # the two digests must match
```

Two digests that differ mean something outside the source got in, and the
installed directory name has stopped identifying what is in it. A *different*
NDK or a different `profile.cmake` legitimately produces a different
`libtower_bridge.so`, which is a new directory and a new `current`, not a
failure — but the build path no longer does.

#### Production digests, and the one that has to be reinstalled

The digest is a property of the source, this NDK and this `profile.cmake`, so a
source change moves it. The current ones, each built with the recipe above,
host cross-compile only, nothing installed:

```text
662cba0974d701c471fe0e7c6cbdeda08c14a668509e8da123a738bfa4f8902b   installed at state/bridge/current until 2026-09-20
b9852e6494056fedaf1b142af2368dbaa4ac975c7dfea2ffbff44b16f1d76284   source before ADR 0011
f9d5f161c33b3af98787d161c9e73f26b1286f519b1648c41b167bffd62a96c3   source with ADR 0011, 2026-09-20; installed at state/bridge/current
7b5e97014b37c63fc0172c5aa3212ca2975ef1fb1722431437b0ca9d902aa228   render-off experiment, TOWER_BRIDGE_RENDER_FRAME_INTERVAL=16 (#27), 2026-09-23
```

`7b5e9701…` is an **experiment variant, not a production digest**. It is the
same source built with `-DTOWER_BRIDGE_RENDER_FRAME_INTERVAL=16` added to the
recipe above, which makes the game render and present one player-loop frame in
16 (`native/tower_bridge/README.md`, "Render-off experiment build"). The default
of that option is 1, which defines nothing, so the production build of the same
source is still `f9d5f161…` byte for byte. The variant is kept out of the
digest-named layout, at `state/bridge/builds/render-interval-16/` beside an
unchanged copy of `state/bridge/current/libunity-bridge.so` and its own
`CMakeCache.txt`, and is selected for one run only by
`TOWER_BRIDGE_BUILD_DIR=state/bridge/builds/render-interval-16`. It is never
installed as `state/bridge/current`, and spectate and recording must never run
on it: what it puts on screen is one frame in sixteen.

`f9d5f161…` is the production build with `unlock_state` and
`unlock_all_upgrades` in it, which is how `--upgrade-availability all` is
applied at each round start ([ADR 0011](adr/0011-upgrade-availability-is-applied-at-round-start.md)).
The move changed the production artifact and nothing else: the diagnostics build
is byte-for-byte what it was (`1a8d2467b2a1f73a0e3e2ca7e6e8fb780a0b8cbec9d97e005e448c3fa5777289`,
the digest `M2-E008` ran on), because the code only left an `#ifdef` it was
inside.

**`state/bridge/current` had to be reinstalled** before any run used the new
commands — install and repoint per the recipe above, copy, verify, then move the
pointer. That was done on 2026-09-20: `current` points at `f9d5f161…`.

`TOWER_BRIDGE_BUILD_DIR` overrides all of this and deploys straight out of a
build tree, which is how a bridge under development is run; every ordinary run
leaves it unset and takes the installed one.

## 3. Android tooling and the AVDs

The workstation's SDK lives at `~/.local/share/android-sdk`, which is one of the
roots `tower_rl.simulation.android_sdk.sdk_roots` searches, so an unattended run
finds `adb` and `emulator` without a shell exporting anything. `ANDROID_HOME` or
`ANDROID_SDK_ROOT` overrides the search. The packages are:

```text
sdkmanager --licenses
sdkmanager \
  "platform-tools" \
  "emulator" \
  "build-tools;36.0.0" \
  "platforms;android-36" \
  "system-images;android-36;google_apis_playstore;x86_64"
```

Review and accept Android's SDK licenses interactively. If the tools are not on
`PATH`:

```text
export ANDROID_SDK_ROOT="$HOME/.local/share/android-sdk"
export PATH="$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
```

Two AVDs exist, and nothing in the project confuses them:

- **`tower_rl_api36_play_x86_64`** — the canonical, account-bearing AVD
  (`CANONICAL_AVD` in `src/tower_rl/simulation/instance.py`), on
  `emulator-5554`. It holds the Play sign-in and the golden snapshot, and
  `CloneInstance` and `run_episodes.py` refuse to touch it. Create it once, with
  the defaults `scripts/create_avd.sh` carries:

```text
./scripts/create_avd.sh
```

  The script is idempotent and takes optional positional overrides — name,
  system image, device — so another image can be named explicitly; unqualified,
  it creates `tower_rl_api36_play_x86_64` from
  `system-images;android-36;google_apis_playstore;x86_64` on `pixel_2`. It
  never installs the game and never touches account data.

- **`tower_rl_instrumented_api36`** — the disposable rooted clone every
  instrumented run uses (`CLONE_AVD`), starting at `emulator-5556` for index 0
  and taking the next even console port per further index. It is a machine-local
  rooted copy of the canonical AVD, recreated per host rather than carried
  between them, and it is what `scripts/clone_session.py` and
  `scripts/instrumented_bridge.sh` address.

Do not place AVD data, snapshots, or Android user data in the repository.

## 4. Inspect and boot

Run the metadata-only checks before installation:

There is no `tower-rl` console script. The checks are
`tower_rl.doctor.run_doctor(xapk, serial)`, rendered with
`tower_rl.doctor.render_json`; host tooling alone is
`uv run python scripts/workstation_preflight.py`.

Boot the canonical AVD visibly, which is what `scripts/launch_avd.sh` does —
`-gpu <renderer> -no-audio -no-boot-anim` and, with no snapshot named,
`-no-snapshot`:

```text
./scripts/launch_avd.sh tower_rl_api36_play_x86_64 host
```

The renderer defaults to `host` (`TOWER_RL_RENDERER` overrides it), and a third
argument names a snapshot to restore read-only (`TOWER_RL_SNAPSHOT`). Wait for
`adb shell getprop sys.boot_completed` to return `1`.

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
in `clone_session.py` and `spectate.py` is `lavapipe` for that reason, while
`train.py` and `run_actors.py` default to `--renderer host`, the renderer every
measured fleet run was taken under.

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
./scripts/launch_avd.sh tower_rl_api36_play_x86_64 lavapipe \
  tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914
```

A named snapshot is restored with `-no-snapshot-save`, so the baseline cannot be
written over by the session that reads it.

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
(`Damage → L4/20  -120` for a purchase — the row, the level it reached and the
cash it cost — and `Hold  2.0s` for a wait), episodes played, the running mean final
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
per-episode records the fleet writes, under `state/recordings/records/` unless
another directory is named.

### What the agent did, in the recording

A recording of the guest screen shows the game and not the agent: the same
`--record` session also writes `<stem>.decisions.jsonl` beside the video, one
JSON object per decision, so what the agent did and when is readable afterwards.
Each line carries `video_s` (seconds since the recording began), `chunk` and
`chunk_s` (which chunk the decision is in and how far into it — the placement a
seam cannot move), `episode`, `decision`, `wave`, `cash`, `health_fraction`,
`action` (`wait` or `attack:3`), `label` (what the game calls that upgrade row,
or `Hold`), `purchase` (the row bought, the level it reached, its maximum and
the cash it cost — `null` for a hold, and absent from tracks written before it
was recorded, which still render), `held_s` (game time the choice was held
for), `hud` (the same
readings the terminal panel shows), and `ended` — with `reason` — on the
decision the episode died on. The session record under
`state/recordings/records/` names the video, the track and the monotonic anchor
both are timed from.

`scripts/render_recording.py` composes the two into one video:

```text
uv run python scripts/render_recording.py --recording state/recordings/session
```

It concatenates the chunks in order and pads the picture to the right with a
panel drawn by `libass` — a header with wave, cash, health and the HUD block,
updated per decision, and a scrolling history of the last twelve actions in
the same form the live panel uses, the current one picked out. The result is
`<stem>-panel.mp4` beside the input. The guest picture is never scaled: the
output is the source resolution plus the 560-pixel panel, encoded x264 CRF 20,
`veryfast`, `yuv420p`, no audio. ffmpeg and a monospace font are required and
the render fails by name if either is missing (Debian/Ubuntu: `sudo apt install
ffmpeg fonts-dejavu-core`).

**The panel leads the picture slightly.** Times are anchored on the monotonic
clock at the instant the host asks the guest to record, and the guest's first
frame arrives an adb round trip and a `screenrecord` process start later — a
few hundred milliseconds, under a second. Each chunk seam loses about a second
the same way, which is why a decision is placed by its chunk rather than by
`video_s`. Close enough to watch; not something to measure from.

## 10. Running a device stage unattended

A stage — a training seed, an evaluation batch, a recording session — is hours
of device time, so it is launched once and left alone rather than watched:

```text
nohup ./scripts/run_stage.sh --name m2-run2-train-seed1 --instances 7 -- \
  uv run --extra tracking python scripts/train.py \
      --actors 7 --renderer host --frame-rate-hz 120 \
      --decision-cadence choice-points --exploration ladder \
      --budget-game-seconds 360000 --block-game-seconds 4000 \
      --checkpoint-every-game-seconds 60000 \
      --epsilon-anneal-decisions 2500 \
      --early-stop-patience-periods 2 --early-stop-min-improvement 0.2 \
      --seed 1 > /dev/null 2>&1 &
```

That is `M2-P002`'s option-B training line for seed 1, unchanged, with
`run_stage.sh` in front of it; `nohup`'s own stdout goes nowhere because the
script already writes everything to the log below.

The stage command after `--` is run exactly as written; `run_stage.sh` does not
bring the fleet up, because `train.py` and `run_actors.py` bring up and tear
down their own instances. What it adds is the guarantee on the way out. On
**every** exit path — success, failure, `SIGINT`, `SIGTERM` — it interrupts the
stage command so the runner can tear its own fleet down, then runs
`scripts/instrumented_bridge.sh cleanup` on each of the stage's serials that is
still live, kills every emulator that is still attached, and verifies the host
is empty: no qemu process (counted through `/proc/*/exe`) and no adb device. An
instance the runner's own teardown is still killing is given up to 60 s to
either answer or go — it exits during the teardown far more often than it needs
cleaning, and neither outcome is a failure; one that is still attached and
still not answering after that is killed and reported as unverifiable, which is
a failure, because its cleanup was never run.
Before it launches anything it refuses a host that is already running something
it should not. Every **attached** instance is checked, not only the ones adb
reports as `device`: one that is not `tower_rl_instrumented_api36`, not
`-read-only`, not on an even console port from 5556, still holding a routable
interface, or attached in a state that cannot be asked about its interfaces at
all, is a refusal — as is `emulator-5554` or the canonical evaluation AVD
anywhere. The canonical AVD is never killed either: if one is running when the
stage ends, the summary says the cleanup failed and it is left for you.

Everything the stage and the script write goes to `state/logs/<name>-<timestamp>.log`
and to stdout, ending in one summary line:

```text
stage m2-run2-train-seed1: exit 0, cleanup ok, instances 7/7 cleaned, 0 exited during teardown, wall 08:12:44
```

Once the teardown has begun, `SIGINT` and `SIGTERM` are **ignored**, so a second
Ctrl-C cannot leave the device half cleaned; each instance's cleanup is bounded
at 120 s so that ignoring them cannot hang the stage. `SIGKILL` of the
supervisor is the one signal that abandons the teardown, and it leaves the
cleanup and the verification to be run by hand.

The exit status is the stage command's, or non-zero if cleanup or the
verification failed while the stage itself succeeded — so a stage that left the
device dirty cannot be read as a stage that passed.

It needs **bash 5.1 or newer** (the workstation runs 5.3) and refuses to start
otherwise: it waits on the stage and on the grace timer at once through
`wait -n -p`, and a shell that cannot do that would read a running stage as a
finished one and clean the device up underneath it.
