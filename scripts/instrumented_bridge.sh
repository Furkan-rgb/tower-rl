#!/usr/bin/env bash
set -euo pipefail

# Private instrumented-training profile only (ADR 0006). This script never runs
# against the canonical official-evaluation emulator: it mounts a reversible
# libunity.so view on a disposable rooted clone so the versioned bridge loads as
# an ordinary DT_NEEDED dependency. It changes no APK bytes, package identity,
# installer identity, app data, or account state.
#
#   verify   report package identity, libunity hash, and bridge-artifact state
#   deploy   install the built bridge and mount the overlay, then start the game
#   cleanup  stop the game, unmount, remove artifacts, reset the frame-rate
#            override, and re-verify identity
#
# Several clone instances can run at once, so the target instance is an argument:
#
#   instrumented_bridge.sh <verify|deploy|cleanup> [serial] [host_port]
#
# One host port per instance, since every instance forwards to the same device
# port. Left out, it follows the serial: emulator-5556 -> 47652, 5558 -> 47653.
#
# Machine-local inputs (never committed):
#   TOWER_BRIDGE_SERIAL     adb serial of the rooted clone (default emulator-5556)
#   TOWER_BRIDGE_BUILD_DIR  private NDK build dir holding libtower_bridge.so and
#                           the patched libunity-bridge.so
#   TOWER_BRIDGE_HOST_PORT  host port forwarded to device port 47651

command="${1:?usage: instrumented_bridge.sh <verify|deploy|cleanup> [serial] [host_port]}"
serial="${2:-${TOWER_BRIDGE_SERIAL:-emulator-5556}}"
package="com.TechTreeGames.TheTower"
device_port=47651
first_console_port=5556
first_host_port=47652
canonical_avd="tower_rl_api36_play_x86_64"

derived_host_port() {
  local console="${serial#emulator-}"
  case "$console" in
    ''|*[!0-9]*) echo "$first_host_port" ;;
    *) echo $(( first_host_port + (console - first_console_port) / 2 )) ;;
  esac
}

build_dir="${TOWER_BRIDGE_BUILD_DIR:-$(cat /tmp/tower-bridge-live.latest 2>/dev/null || true)}"
host_port="${3:-${TOWER_BRIDGE_HOST_PORT:-$(derived_host_port)}}"
adb="${ANDROID_SDK_ROOT:-$HOME/.local/share/android-sdk}/platform-tools/adb"

device() { "$adb" -s "$serial" "$@"; }
su_device() { device shell "su -c '$1'"; }

# The canonical evaluation AVD is never instrumented, whatever serial it is on.
running_avd="$(device emu avd name 2>/dev/null | head -n 1 | tr -d '\r' || true)"
if [ "$running_avd" = "$canonical_avd" ]; then
  echo "refusing: $serial is the canonical evaluation AVD $canonical_avd" >&2
  exit 1
fi

lib_path() {
  device shell pm path "$package" | sed -n '1s#package:##p' | tr -d '\r' |
    sed 's#/base.apk#/lib/arm64/libunity.so#'
}

# The overlay's backing file must never be removed while the bind mount is live:
# the target path then resolves to a deleted inode and can no longer be mounted.
original_libunity_sha256() {
  [ -n "$build_dir" ] && [ -f "$build_dir/CMakeCache.txt" ] || return 1
  sed -n 's/^TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256:STRING=//p' "$build_dir/CMakeCache.txt"
}

target_sha256() {
  su_device "sha256sum $1" | awk '{print $1}' | tr -d '\r'
}

unmount_overlay() {
  local target="$1" original
  original="$(original_libunity_sha256 || true)"
  if [ -z "$original" ]; then
    su_device "umount $target" > /dev/null 2>&1 || true
    return 0
  fi
  for _ in 1 2 3 4 5; do
    [ "$(target_sha256 "$target")" = "$original" ] && return 0
    su_device "umount $target" > /dev/null 2>&1 || true
  done
  [ "$(target_sha256 "$target")" = "$original" ]
}

report_identity() {
  local target="$1"
  echo "libunity_sha256: $(su_device "sha256sum $target" | awk '{print $1}')"
  device shell dumpsys package "$package" |
    grep -E 'versionName=|versionCode=|installerPackageName=' | head -n 3 | sed 's/^ *//'
  su_device "mount | grep -c libunity.so || true" | tr -d '\r' | sed 's/^/libunity_mounts: /'
}

# Airplane mode alone does not take this emulator offline: the setting can read 1
# while the wifi radio is still up with a route to the host NAT, which is how the
# clone ran online through M1B-E009. The interface is what decides, so that is
# what is checked, and deploy refuses rather than warns.
require_offline() {
  local addresses
  addresses="$(device shell ip -o -4 addr show 2>/dev/null | tr -d '\r' | grep -v ' lo ' || true)"
  if [ -n "$addresses" ]; then
    echo "refusing to deploy: $serial still has a routable interface" >&2
    echo "$addresses" >&2
    echo "disable the radios first: adb -s $serial shell svc wifi disable && adb -s $serial shell svc data disable" >&2
    exit 1
  fi
}

target="$(lib_path)"
[ -n "$target" ] || { echo "package $package is not installed on $serial" >&2; exit 1; }

case "$command" in
  verify)
    report_identity "$target"
    device shell ip -o -4 addr show 2>/dev/null | tr -d '\r' | grep -v ' lo ' |
      sed 's/^/routable_interface: /' || echo "routable_interfaces: none"
    su_device "test ! -e /data/user/0/$package/files/libtower_bridge.so && test ! -e /data/local/tmp/libunity-tower-bridge.so && echo bridge_artifacts: none"
    ;;

  deploy)
    [ -n "$build_dir" ] || { echo "TOWER_BRIDGE_BUILD_DIR is required" >&2; exit 1; }
    require_offline
    bridge="$build_dir/libtower_bridge.so"
    overlay="$build_dir/libunity-bridge.so"
    for file in "$bridge" "$overlay"; do
      [ -f "$file" ] || { echo "missing private build artifact: $file" >&2; exit 1; }
    done
    uid="$(device shell dumpsys package "$package" | sed -n 's/.*\(appId\|userId\)=\([0-9]\+\).*/\2/p' | head -n 1 | tr -d '\r')"
    [ -n "$uid" ] || { echo "cannot resolve the package uid" >&2; exit 1; }
    device shell am force-stop "$package"
    # A previous deploy leaves root-owned, relabelled staging files in place.
    unmount_overlay "$target" || { echo "cannot unmount the previous overlay" >&2; exit 1; }
    su_device "rm -f /data/local/tmp/libtower_bridge.so /data/local/tmp/libunity-tower-bridge.so"
    device push "$bridge" /data/local/tmp/libtower_bridge.so > /dev/null
    device push "$overlay" /data/local/tmp/libunity-tower-bridge.so > /dev/null
    su_device "cp /data/local/tmp/libtower_bridge.so /data/user/0/$package/files/libtower_bridge.so && chown $uid:$uid /data/user/0/$package/files/libtower_bridge.so && chmod 0555 /data/user/0/$package/files/libtower_bridge.so && restorecon -F /data/user/0/$package/files/libtower_bridge.so && chcon u:object_r:apk_data_file:s0 /data/local/tmp/libunity-tower-bridge.so && chmod 0555 /data/local/tmp/libunity-tower-bridge.so && mount -o bind /data/local/tmp/libunity-tower-bridge.so $target && mount | grep libunity.so"
    device forward "tcp:$host_port" "tcp:$device_port" > /dev/null
    device logcat -c
    device shell monkey -p "$package" -c android.intent.category.LAUNCHER 1 > /dev/null
    echo "deployed: overlay mounted, host port $host_port forwarded to device $device_port"
    ;;

  cleanup)
    device shell am force-stop "$package"
    # Bring-up pins the game's frame rate through GameManagerService (see
    # `clone_session.pin_game_frame_rate`); that override is device state, so it
    # is reset here and no instance is left modified by a run.
    device shell cmd game reset "$package" > /dev/null
    echo "game_frame_rate_override: reset"
    unmount_overlay "$target" || echo "warning: the overlay is still mounted" >&2
    su_device "rm -f /data/user/0/$package/files/libtower_bridge.so /data/local/tmp/libtower_bridge.so /data/local/tmp/libunity-tower-bridge.so"
    device forward --remove "tcp:$host_port" 2> /dev/null || true
    report_identity "$target"
    su_device "test ! -e /data/user/0/$package/files/libtower_bridge.so && test ! -e /data/local/tmp/libunity-tower-bridge.so && echo bridge_artifacts: removed"
    ;;

  *)
    echo "unknown command: $command" >&2
    exit 1
    ;;
esac
