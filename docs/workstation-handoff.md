# Tower-RL — Workstation Handoff

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

Treat the workstation as a new device profile. Do not assume this Mac snapshot
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
   Workshop spending, Labs locked), enable airplane mode only after the game is
   running, and create a new workstation-local snapshot with a unique name.
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
