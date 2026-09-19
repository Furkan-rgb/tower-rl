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
#            override, and re-verify identity. Every step is attempted whatever
#            an earlier one found, and the exit status is non-zero if any check
#            failed: a caller that reads it must not be told an instance is
#            clean when it is not.
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
#                           the patched libunity-bridge.so; defaults to the
#                           installed bridge under <repo>/state/bridge
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

# The bridge this project deploys: an explicit build tree while one is being
# developed, otherwise the installed one. `current` is a symlink to a directory
# named for the SHA-256 of the `libtower_bridge.so` in it, under the project's
# git-ignored `state/`, so it survives the reboot that a /tmp build directory
# does not. The root comes from this script's own location, never the cwd, and
# is kept in step with `bridge.BRIDGE_STATE_DIRECTORY`, which is what every
# scripted path resolves through; this is the hand-run path.
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
installed_bridge="$repository_root/state/bridge/current"
build_dir="${TOWER_BRIDGE_BUILD_DIR:-$installed_bridge}"
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

# What the device must read back as once cleanup is done. Three of the four come
# from the private build's own `CMakeCache.txt` — the same file
# `bridge.compatibility` reads and the one place that says which game build this
# profile is for — rather than being written here a second time to drift from it.
cmake_cache_value() {
  [ -n "$build_dir" ] && [ -f "$build_dir/CMakeCache.txt" ] || return 1
  sed -n "s/^$1:STRING=//p" "$build_dir/CMakeCache.txt"
}

# The overlay's backing file must never be removed while the bind mount is live:
# the target path then resolves to a deleted inode and can no longer be mounted.
original_libunity_sha256() { cmake_cache_value TOWER_BRIDGE_ORIGINAL_LIBUNITY_SHA256; }
expected_version_name() { cmake_cache_value TOWER_BRIDGE_PACKAGE_VERSION; }
expected_version_code() { cmake_cache_value TOWER_BRIDGE_PACKAGE_VERSION_CODE; }

#: The one expectation the build configuration does not carry: the app is the
#: one Google Play installed, and no part of the instrumented profile may change
#: that. `deploy` mounts a view of a file; it never touches installer identity.
expected_installer="com.android.vending"

package_dump() { device shell dumpsys package "$package" 2> /dev/null | tr -d '\r'; }

dump_field() {
  printf '%s\n' "$2" | sed -n "s/.*$1=\([^ ]*\).*/\1/p" | head -n 1
}

libunity_mount_count() {
  su_device "mount | grep -c libunity.so || true" | tr -d '\r' | head -n 1
}

#: Cleanup's checks are counted rather than acted on where they fail: the device
#: is put back whatever an earlier step found, and the count decides the exit
#: status at the end. Before this, a still-mounted overlay and a frame-rate
#: override that never reset were printed as warnings and exited 0, so a caller
#: that read the status — `fleet.tear_down_instance`, `run_stage.sh` — was told
#: the instance was clean when it was not.
failures=0
fail() {
  echo "check failed: $1" >&2
  failures=$((failures + 1))
}

#: A reading that came back empty is not a reading, and a report of one is
#: worse than no report: `verify` printing `versionCode=` at exit 0 says the
#: identity was confirmed when the device answered nothing at all.
require_reading() {
  [ -n "$2" ] || fail "$1 read back empty on $serial"
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

# What `cleanup` leaves behind, read back from the device and compared against
# what it must be — not printed for somebody to notice. The readings are the
# same ones `report_identity` prints, in the same shapes, because they are the
# evidence a cleanup log is read for; the difference is that a mismatch here is
# a failure rather than a line.
check_identity() {
  local expected
  # The readings are `report_identity`'s: one place takes them and prints them,
  # and this adds what each one has to be. Reading the device twice would let
  # the evidence in the log and the value that was checked disagree.
  report_identity "$1"
  if expected="$(original_libunity_sha256)" && [ -n "$expected" ]; then
    [ "$identity_sha256" = "$expected" ] ||
      fail "libunity_sha256 on $serial is $identity_sha256, not the original $expected"
  else
    fail "nothing to check libunity_sha256 against: $build_dir holds no CMakeCache.txt"
  fi
  if expected="$(expected_version_name)" && [ -n "$expected" ]; then
    [ "$identity_version_name" = "$expected" ] ||
      fail "versionName is $identity_version_name, not $expected"
  else
    fail "nothing to check versionName against: $build_dir holds no CMakeCache.txt"
  fi
  if expected="$(expected_version_code)" && [ -n "$expected" ]; then
    [ "$identity_version_code" = "$expected" ] ||
      fail "versionCode is $identity_version_code, not $expected"
  else
    fail "nothing to check versionCode against: $build_dir holds no CMakeCache.txt"
  fi
  [ "$identity_installer" = "$expected_installer" ] ||
    fail "installerPackageName is $identity_installer, not $expected_installer"
  [ "$identity_mounts" = "0" ] ||
    fail "$identity_mounts libunity.so mount(s) survive on $serial"
}

#: The readings `report_identity` last took. `check_identity` compares these
#: rather than reading the device again, so what a log shows and what was
#: checked are the same numbers.
identity_sha256=""
identity_version_name=""
identity_version_code=""
identity_installer=""
identity_mounts=""

report_identity() {
  local target="$1" dump
  identity_sha256="$(target_sha256 "$target")"
  dump="$(package_dump)"
  identity_version_name="$(dump_field versionName "$dump")"
  identity_version_code="$(dump_field versionCode "$dump")"
  identity_installer="$(dump_field installerPackageName "$dump")"
  identity_mounts="$(libunity_mount_count)"
  echo "libunity_sha256: $identity_sha256"
  echo "versionName=$identity_version_name"
  echo "versionCode=$identity_version_code"
  echo "installerPackageName=$identity_installer"
  echo "libunity_mounts: $identity_mounts"
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
    # `verify` reports rather than decides — except about its own readings. An
    # empty one means the device answered nothing, and reporting that as
    # identity would be a confirmation nobody made.
    require_reading libunity_sha256 "$identity_sha256"
    require_reading versionName "$identity_version_name"
    require_reading versionCode "$identity_version_code"
    require_reading installerPackageName "$identity_installer"
    device shell ip -o -4 addr show 2>/dev/null | tr -d '\r' | grep -v ' lo ' |
      sed 's/^/routable_interface: /' || echo "routable_interfaces: none"
    su_device "test ! -e /data/user/0/$package/files/libtower_bridge.so && test ! -e /data/local/tmp/libunity-tower-bridge.so && echo bridge_artifacts: none" ||
      fail "bridge artifacts survive on $serial"
    if [ "$failures" -gt 0 ]; then
      echo "verify_checks: $failures failed on $serial" >&2
      exit 1
    fi
    ;;

  deploy)
    [ -d "$build_dir" ] || {
      echo "no bridge to deploy: $build_dir is not a directory" >&2
      echo "install one under $repository_root/state/bridge/<sha256>/ and point current at it, or set TOWER_BRIDGE_BUILD_DIR" >&2
      exit 1
    }
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
    # Every step runs to the end whatever an earlier one found: a device cleaned
    # half way is worse than one every step was attempted on. What a failed check
    # changes is the exit status, through `fail`.
    device shell am force-stop "$package" || fail "am force-stop returned nonzero"
    # Bring-up pins the game's frame rate through GameManagerService (see
    # `clone_session.raise_frame_rate`); that override is device state, so it is
    # reset here and no instance is left modified by a run.
    #
    # Unguarded, this line ended cleanup: `set -e` plus adb's propagation of the
    # remote exit code meant one nonzero `cmd game` return left the overlay
    # mounted, the bridge deployed and identity never re-verified — the state
    # cleanup exists to prevent. And the restore is read back rather than
    # asserted: every other line printed here is evidence, so this one may not
    # claim a reset it never observed.
    device shell cmd game reset "$package" > /dev/null 2>&1 ||
      echo "warning: cmd game reset returned nonzero" >&2
    uid="$(device shell dumpsys package "$package" 2> /dev/null |
      sed -n 's/.*\(appId\|userId\)=\([0-9]\+\).*/\2/p' | head -n 1 | tr -d '\r' || true)"
    override=""
    if [ -n "$uid" ]; then
      override="$(device shell dumpsys SurfaceFlinger 2> /dev/null | tr -d '\r' |
        sed -n "s/.*{$uid, \([0-9]\+\) [0-9]\+}.*/\1/p" | head -n 1 || true)"
    fi
    if [ -z "$override" ]; then
      # The game is force-stopped above, so SurfaceFlinger may hold no per-uid
      # entry at all; absence is not a reading and is not reported as one.
      echo "game_frame_rate_override: reset-issued (unverified)"
    elif [ "$override" = "0" ] || [ "$override" = "60" ]; then
      # Observed on device: a reset leaves the per-uid gameModeOverride at 0,
      # which is GameManagerService holding no override at all; 60 is the stock
      # `ro.surface_flinger.game_default_frame_rate_override`. Either is the
      # state bring-up found, and neither is the rate bring-up pinned.
      echo "game_frame_rate_override: reset"
    else
      echo "game_frame_rate_override: NOT-reset, still $override" >&2
      fail "the game frame-rate override on $serial is still $override"
    fi
    unmount_overlay "$target" || {
      echo "warning: the overlay is still mounted" >&2
      fail "the overlay is still mounted on $serial"
    }
    su_device "rm -f /data/user/0/$package/files/libtower_bridge.so /data/local/tmp/libtower_bridge.so /data/local/tmp/libunity-tower-bridge.so" ||
      fail "the bridge artifacts could not be removed from $serial"
    device forward --remove "tcp:$host_port" 2> /dev/null || true
    check_identity "$target"
    # The device's own answer rather than its exit status: `adb shell` has
    # propagated the remote status since platform-tools 24, but here the remote
    # command is `su -c '<test> && <echo>'`, so the status that arrives is su's
    # and depends on the su build. The marker it echoes is its own evidence and
    # needs no such assumption.
    artifacts="$(su_device "test ! -e /data/user/0/$package/files/libtower_bridge.so && test ! -e /data/local/tmp/libunity-tower-bridge.so && echo bridge_artifacts: removed" 2> /dev/null | tr -d '\r' | head -n 1 || true)"
    if [ "$artifacts" = "bridge_artifacts: removed" ]; then
      echo "$artifacts"
    else
      fail "bridge artifacts survive on $serial"
    fi
    if [ "$failures" -gt 0 ]; then
      echo "cleanup_checks: $failures failed on $serial" >&2
      exit 1
    fi
    echo "cleanup_checks: all passed"
    ;;

  *)
    echo "unknown command: $command" >&2
    exit 1
    ;;
esac
