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

For a host-specific configuration, copy `configs/local.example.yaml` to
`configs/local.yaml` and edit only device/profile values. Keep credentials,
account state, snapshots, and generated runtime paths out of that file.

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

```text
uv run tower-rl doctor \
  --xapk local/the-tower-29-0-1.xapk
```

Boot the candidate visibly for first-run characterization:

```text
emulator @tower_rl_api36_play_arm64 \
  -gpu lavapipe \
  -no-audio \
  -no-boot-anim \
  -no-snapshot
```

Wait for `adb shell getprop sys.boot_completed` to return `1`.

## 5. Optional XAPK metadata/reference inspection

The supplied XAPK is a ZIP archive containing four APKs. It is reference-only
for this baseline; the validated runtime was installed by Google Play. Inspect
its metadata without installing or copying proprietary bytes into the repo:

```text
unzip -l local/the-tower-29-0-1.xapk
```

Confirm the Play-installed running setup:

```text
uv run tower-rl doctor \
  --xapk local/the-tower-29-0-1.xapk \
  --serial emulator-5554

Validate the pinned visual profile and, when desired, run the bounded no-upgrade
navigation smoke flow. The navigation form restores the canonical snapshot after
the run so earned in-run coins or transient state cannot drift the baseline:

```text
uv run tower-rl probe \
  --serial emulator-5554 \
  --navigate \
  --restore-snapshot tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914
```
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
