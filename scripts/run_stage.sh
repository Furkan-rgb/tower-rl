#!/usr/bin/env bash
set -euo pipefail

# One device stage, from one invocation, ending with the device verified clean.
#
#   run_stage.sh --name <stage> --instances <N> [--shutdown-grace <s>] -- <command...>
#
# A stage is a multi-hour device command: a training seed, an evaluation batch, a
# recording session. Launched once in the background, this script supervises it
# and guarantees the teardown on *every* exit path — success, failure, SIGINT,
# SIGTERM — so the safety-critical part of a stage is not left to whoever
# happens to be watching the log.
#
# It does not bring the fleet up. The fleet runners own their own bring-up and
# their own teardown (`bring_up_fleet`/`tear_down_fleet` in
# `src/tower_rl/simulation/fleet.py`; `scripts/train.py` and
# `scripts/run_actors.py` call them), so a wrapper that launched emulators would
# collide with the stage command rather than serve it. What is missing there is
# the abort path: a killed or crashed runner leaves emulators running with the
# overlay mounted, and nothing then re-verifies the device. That is this
# script's whole job — supervise, then clean up and *verify*, whatever happened.
#
# Cleanup itself is `scripts/instrumented_bridge.sh cleanup`, per serial, and its
# post-conditions are that script's: the original `libunity.so` digest, the
# version code, the installer identity, zero mounts, the artifacts removed, the
# frame-rate override read back. Nothing here reimplements or reinterprets them;
# what is added here is that they are run on every abort path and that the host
# is checked afterwards for a surviving qemu process or adb device.
#
# Device-safety invariants are refused, never warned about: only the
# `tower_rl_instrumented_api36` AVD, even console ports from 5556, every
# instance `-read-only`, and offline verified by interface. Never
# `emulator-5554`, never `tower_rl_api36_play_x86_64`.
#
# Two paths are read from the environment so a test can point them somewhere
# harmless; both default to the real thing and neither is a runtime option:
#   TOWER_STAGE_PROC_ROOT      the /proc to count qemu processes in
#   TOWER_STAGE_LOG_DIRECTORY  where the stage log is written

clone_avd="tower_rl_instrumented_api36"
canonical_avd="tower_rl_api36_play_x86_64"
canonical_serial="emulator-5554"
first_console_port=5556
#: How long the host is given to come back empty after the last kill.
verify_timeout=30
#: How long the stage command is given to tear its own fleet down after SIGINT,
#: before it is killed outright. A seven-instance fleet's teardown is minutes,
#: not seconds: each instance is force-stopped, unmounted and re-verified.
default_shutdown_grace=900

name=""
instances=0
shutdown_grace="$default_shutdown_grace"
stage_command=()

while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="${2:?--name needs a value}"; shift 2 ;;
    --instances) instances="${2:?--instances needs a value}"; shift 2 ;;
    --shutdown-grace) shutdown_grace="${2:?--shutdown-grace needs a value}"; shift 2 ;;
    --) shift; stage_command=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$name" ] || { echo "usage: run_stage.sh --name <stage> --instances <N> -- <command...>" >&2; exit 2; }
case "$instances" in ''|*[!0-9]*) echo "--instances must be a count" >&2; exit 2 ;; esac
[ "$instances" -ge 1 ] || { echo "--instances must be at least 1" >&2; exit 2; }
[ "${#stage_command[@]}" -gt 0 ] || { echo "nothing to run: give the stage command after --" >&2; exit 2; }

# The repository root comes from this script's own location, never the cwd —
# the same rule `instrumented_bridge.sh` follows, so a stage launched from
# anywhere finds the same bridge, the same logs and the same scripts.
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_directory="${TOWER_STAGE_LOG_DIRECTORY:-$repository_root/state/logs}"
proc_root="${TOWER_STAGE_PROC_ROOT:-/proc}"

# Whatever is on PATH wins, so a test can put a stub in front of both; the
# installed SDK and the script beside this one are what production resolves to.
adb="$(command -v adb || true)"
[ -n "$adb" ] && [ -x "$adb" ] ||
  adb="${ANDROID_SDK_ROOT:-$HOME/.local/share/android-sdk}/platform-tools/adb"
bridge_script="$(command -v instrumented_bridge.sh || true)"
[ -n "$bridge_script" ] && [ -x "$bridge_script" ] ||
  bridge_script="$repository_root/scripts/instrumented_bridge.sh"

mkdir -p "$log_directory"
log_path="$log_directory/$name-$(date +%Y%m%d-%H%M%S).log"
# Everything this script and the stage command write goes to both the log and
# stdout, so the summary line is in the place a reader looks either way.
exec > >(tee -a "$log_path") 2>&1

serial_for() { echo "emulator-$((first_console_port + 2 * $1))"; }

expected_serials=()
for index in $(seq 0 $((instances - 1))); do
  expected_serials+=("$(serial_for "$index")")
done

#: The serials adb reports as `device` right now. A serial in any other state is
#: not something cleanup can be run through, and is reported rather than used.
live_serials() {
  "$adb" devices 2>/dev/null | awk '$2 == "device" { print $1 }'
}

#: Every serial adb knows about, whatever state it is in.
attached_serials() {
  "$adb" devices 2>/dev/null | awk 'NR > 1 && NF >= 2 { print $1 }'
}

#: The emulator's own command line, from the process holding that console port.
#: Read from /proc rather than asked of the emulator, because `-read-only` is a
#: launch argument and nothing on the device reports it.
emulator_command_line() {
  local port="$1" file text
  for file in "$proc_root"/[0-9]*/cmdline; do
    [ -r "$file" ] || continue
    text="$(tr '\0' ' ' < "$file" 2>/dev/null || true)"
    case "$text" in
      *" -port $port "*) echo "$text"; return 0 ;;
    esac
  done
  return 1
}

#: qemu processes, counted through /proc/*/exe. Never `pgrep -f`: a pattern
#: matched against command lines also matches the agent's own shell that carries
#: the pattern, which is how a fleet once read as still running after it was gone.
qemu_process_count() {
  local entry target count=0
  for entry in "$proc_root"/[0-9]*/exe; do
    target="$(readlink "$entry" 2>/dev/null || true)"
    case "$target" in *qemu*) count=$((count + 1)) ;; esac
  done
  echo "$count"
}

# ---------------------------------------------------------------------------
# Before anything is launched.
#
# The fleet runners bring their own instances up, so the ordinary case is a host
# with nothing running on it and nothing here to check. What is checked is every
# instance that *is* already up — the single-instance stages connect to one that
# was brought up for them — and the check is a refusal, not a warning.
# ---------------------------------------------------------------------------
preflight() {
  local serial port command_line addresses refusals=0

  for serial in $(live_serials); do
    if [ "$serial" = "$canonical_serial" ]; then
      echo "refusing: $canonical_serial is running; that is the canonical evaluation AVD's serial" >&2
      refusals=$((refusals + 1))
      continue
    fi
    local expected=no
    for candidate in "${expected_serials[@]}"; do
      [ "$candidate" = "$serial" ] && expected=yes
    done
    if [ "$expected" = no ]; then
      echo "refusing: $serial is running but is not one of this stage's $instances instances" >&2
      refusals=$((refusals + 1))
      continue
    fi

    port="${serial#emulator-}"
    if [ "$((port % 2))" -ne 0 ] || [ "$port" -lt "$first_console_port" ]; then
      echo "refusing: $serial is not an even console port at or above $first_console_port" >&2
      refusals=$((refusals + 1))
      continue
    fi

    if ! command_line="$(emulator_command_line "$port")"; then
      echo "refusing: no emulator process holds port $port, so $serial cannot be shown to be read-only" >&2
      refusals=$((refusals + 1))
      continue
    fi
    case "$command_line" in
      *"@$canonical_avd"*)
        echo "refusing: $serial is running the canonical evaluation AVD $canonical_avd" >&2
        refusals=$((refusals + 1)); continue ;;
    esac
    case "$command_line" in
      *"@$clone_avd"*) ;;
      *) echo "refusing: $serial is not running $clone_avd" >&2
         refusals=$((refusals + 1)); continue ;;
    esac
    case "$command_line" in
      *" -read-only "*) ;;
      *) echo "refusing: $serial was not launched -read-only" >&2
         refusals=$((refusals + 1)); continue ;;
    esac

    # Airplane mode reads 1 while the radio is still up, so the interface is
    # what decides — the same reading `instrumented_bridge.sh deploy` refuses on.
    addresses="$("$adb" -s "$serial" shell ip -o -4 addr show 2>/dev/null | tr -d '\r' | grep -v ' lo ' || true)"
    if [ -n "$addresses" ]; then
      echo "refusing: $serial still has a routable interface" >&2
      echo "$addresses" >&2
      refusals=$((refusals + 1))
      continue
    fi

    echo "preflight: $serial is $clone_avd, read-only, offline"
  done

  [ "$refusals" -eq 0 ]
}

# ---------------------------------------------------------------------------
# After the stage, however it ended.
# ---------------------------------------------------------------------------
cleaned=0
cleanup_ok=yes

# The stage command is interrupted rather than killed, and then waited for: the
# fleet runners tear their own fleet down on SIGINT, and that teardown is the
# one that knows which instances the run actually brought up. This script's own
# cleanup below is the backstop for what that teardown did not reach.
#
# Waiting is `wait -n -p` (bash 5.1 or newer) against the stage and a timer,
# rather than a poll on `kill -0`: a child that has exited but not been reaped
# is a zombie, and `kill -0` reports a zombie as alive, so a poll waits out the
# whole grace period against a stage that finished immediately.
stop_stage() {
  local finished="" timer
  if ! kill -0 "$stage_pid" 2>/dev/null; then
    wait "$stage_pid" 2>/dev/null || true
    return 0
  fi
  echo "stage $name: interrupting the stage command (pid $stage_pid) so it can tear its own fleet down"
  kill -INT "$stage_pid" 2>/dev/null || true
  sleep "$shutdown_grace" &
  timer=$!
  wait -n -p finished "$stage_pid" "$timer" 2>/dev/null || true
  if [ "$finished" = "$timer" ]; then
    echo "stage $name: the stage command did not exit within ${shutdown_grace}s; killing it" >&2
    kill -KILL "$stage_pid" 2>/dev/null || true
    wait "$stage_pid" 2>/dev/null || true
    cleanup_ok=no
  else
    kill "$timer" 2>/dev/null || true
    wait "$timer" 2>/dev/null || true
  fi
}

clean_instances() {
  local serial live
  for serial in "${expected_serials[@]}"; do
    live=no
    for candidate in $(live_serials); do
      [ "$candidate" = "$serial" ] && live=yes
    done
    if [ "$live" = no ]; then
      # Either the stage never brought it up, or the stage's own teardown
      # already put it down. Reported, because a stage that was meant to run N
      # instances and cleaned fewer is a thing the summary has to say.
      echo "cleanup: $serial is not live; not cleaned"
      continue
    fi
    if "$bridge_script" cleanup "$serial"; then
      cleaned=$((cleaned + 1))
    else
      echo "cleanup: $serial did not clean up" >&2
      cleanup_ok=no
    fi
    "$adb" -s "$serial" emu kill > /dev/null 2>&1 || true
  done

  # Whatever else is still attached goes down too: leaving an emulator running
  # is a device-safety failure, not an inconvenience. The canonical serial is
  # the one thing automation never touches, so it is reported and left alone.
  for serial in $(attached_serials); do
    if [ "$serial" = "$canonical_serial" ]; then
      echo "cleanup: $canonical_serial is attached and was left untouched; put it down by hand" >&2
      cleanup_ok=no
      continue
    fi
    "$adb" -s "$serial" emu kill > /dev/null 2>&1 || true
  done
}

verify_host_clean() {
  local waited=0 qemu devices
  while :; do
    qemu="$(qemu_process_count)"
    devices="$(attached_serials | wc -l)"
    if [ "$qemu" -eq 0 ] && [ "$devices" -eq 0 ]; then
      echo "verified: no qemu process, no adb device"
      return 0
    fi
    [ "$waited" -ge "$verify_timeout" ] && break
    sleep 2
    waited=$((waited + 2))
  done
  echo "verification failed: $qemu qemu process(es), $devices adb device(s) after ${verify_timeout}s" >&2
  "$adb" devices >&2 || true
  return 1
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  [ -n "${stage_status:-}" ] || stage_status="$status"
  stop_stage
  clean_instances
  verify_host_clean || cleanup_ok=no
  local exit_code="$stage_status"
  if [ "$exit_code" -eq 0 ] && [ "$cleanup_ok" = no ]; then
    exit_code=1
  fi
  local wall=$SECONDS
  printf 'stage %s: exit %d, cleanup %s, instances %d/%d cleaned, wall %02d:%02d:%02d\n' \
    "$name" "$stage_status" "$([ "$cleanup_ok" = yes ] && echo ok || echo failed)" \
    "$cleaned" "$instances" "$((wall / 3600))" "$((wall % 3600 / 60))" "$((wall % 60))"
  exit "$exit_code"
}

if ! preflight; then
  echo "stage $name: refused before launch; nothing was started and nothing was cleaned" >&2
  exit 2
fi

echo "stage $name: $instances instance(s), log $log_path"
echo "stage $name: ${stage_command[*]}"

# The stage runs in the background and is waited on, rather than in the
# foreground: a foreground child holds every trap until it returns, and the
# whole point of this script is that a signal reaches the teardown promptly.
trap 'exit 130' INT
trap 'exit 143' TERM
trap finish EXIT

# Job control, so that the stage can be interrupted at all. A shell without it
# starts every asynchronous command with SIGINT ignored, and a signal ignored on
# entry cannot be trapped by the child either — so `kill -INT` reached a
# training run that could not act on it, and its own fleet teardown, the one
# that knows what it brought up, never ran.
set -m

"${stage_command[@]}" &
stage_pid=$!
set +e
wait "$stage_pid"
stage_status=$?
set -e
