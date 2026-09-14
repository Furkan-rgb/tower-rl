#!/usr/bin/env bash
set -euo pipefail

# Safe/idempotent AVD creation. This never installs the game or touches account data.
avd_name="${1:-tower_rl_api36_play_x86_64}"
system_image="${2:-system-images;android-36;google_apis_playstore;x86_64}"
device_name="${3:-pixel_2}"

if ! command -v avdmanager >/dev/null 2>&1; then
  echo "avdmanager is not on PATH; install Android command-line tools first" >&2
  exit 1
fi
if ! command -v emulator >/dev/null 2>&1; then
  echo "emulator is not on PATH; install the Android emulator first" >&2
  exit 1
fi

if emulator -list-avds | grep -Fxq "$avd_name"; then
  echo "AVD already exists: $avd_name"
  exit 0
fi

echo "Creating AVD $avd_name from $system_image ($device_name)"
printf 'no\n' | avdmanager create avd \
  --name "$avd_name" \
  --package "$system_image" \
  --device "$device_name"
