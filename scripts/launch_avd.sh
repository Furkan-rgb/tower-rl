#!/usr/bin/env bash
set -euo pipefail

# Launch only; Play sign-in, consent, and snapshot creation remain manual.
avd_name="${1:?usage: launch_avd.sh <avd-name> [renderer] [snapshot-name]}"
renderer="${2:-${TOWER_RL_RENDERER:-host}}"
snapshot_name="${3:-${TOWER_RL_SNAPSHOT:-}}"

if ! command -v emulator >/dev/null 2>&1; then
  echo "emulator is not on PATH; install the Android emulator first" >&2
  exit 1
fi

snapshot_args=(-no-snapshot)
if [[ -n "$snapshot_name" ]]; then
  snapshot_args=(-snapshot "$snapshot_name" -no-snapshot-save)
fi

exec emulator "@$avd_name" \
  -gpu "$renderer" \
  -no-audio \
  -no-boot-anim \
  "${snapshot_args[@]}"
