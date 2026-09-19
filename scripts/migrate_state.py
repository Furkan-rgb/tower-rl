#!/usr/bin/env python3
"""Move this host's `~/.local/state/tower-rl` tree into the project's `state/`.

One-shot: the project keeps everything it writes under `<repo>/state/` now
(`tower_rl.environment.project_state`), and this is the single step that brings
an existing host's tree in. Run it once; afterwards the old location is gone and
this script has nothing left to do.

The move is `os.rename` per top-level entry, not a copy: the two locations are
on the same filesystem on this host, a rename is atomic and instant, and a copy
of a 35 GB bridge-and-runs tree is both slow and a chance to end up with two
divergent copies. A rename across filesystems fails loudly rather than falling
back to a copy nobody asked for.

Two refusals, because both are states in which a move silently corrupts
something:

- an emulator is running — a live instance holds an installed bridge open and a
  training run writes into `runs/` and the MLflow store while we move them;
- `state/` already has content — a second run, or a half-finished first one,
  must not merge two trees into each other.

`bridge/current` is recreated as a relative symlink. It pointed at an absolute
path under the home directory, which names nothing after the move, and a
relative `current -> <sha256>` is what keeps the installed-bridge layout valid
wherever the checkout is.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.environment.project_state import state_directory  # noqa: E402

#: Where project state lived before this move.
FORMER_STATE_DIRECTORY = Path.home() / ".local" / "state" / "tower-rl"


class MigrationRefused(Exception):
    """The move was not attempted, and the reason is in the message."""


def running_emulators(proc: Path = Path("/proc")) -> list[str]:
    """Every live process whose executable is a `qemu-system-*`, by pid.

    Read from `/proc/<pid>/exe`, the kernel's own answer to "what is this
    process running". Not `pgrep -f`, which matches a *command line*: it would
    report this script for naming the emulator in an argument and miss a qemu
    whose argv was rewritten. `spectate.py` holds the same reading for its own
    refusal; it is repeated here rather than imported because that copy sits
    behind a curses-and-torch entry point that a filesystem move must not have
    to load.
    """
    found: list[str] = []
    try:
        entries = sorted(proc.iterdir())
    except OSError:  # no procfs to read
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            executable = os.readlink(entry / "exe")
        except OSError:  # gone, or not ours to look at
            continue
        if Path(executable).name.startswith("qemu-system"):
            found.append(f"{entry.name} {executable}")
    return found


def relative_current_symlink(bridge: Path) -> None:
    """Re-point `bridge/current` at the sibling directory it names, relatively.

    Only the last component survives: `current` has always named a directory
    beside it, so the absolute path it held under the home directory carries no
    information the name does not, and it is the one part of the moved tree that
    would otherwise still point outside the project.
    """
    current = bridge / "current"
    if not current.is_symlink():
        return
    target = Path(os.readlink(current)).name
    current.unlink()
    current.symlink_to(target)


def migrate(source: Path, destination: Path) -> list[tuple[Path, Path]]:
    """Move every entry of `source` into `destination`; return the mapping.

    The destination is created, each top-level entry is renamed into it, and the
    emptied source directory is removed, so the old location is absent when this
    returns — a directory left behind is a second place to look for state, which
    is the thing this move exists to end.
    """
    if not source.exists():
        return []
    occupied = sorted(destination.iterdir()) if destination.is_dir() else []
    if occupied:
        raise MigrationRefused(
            f"{destination} already has content ({', '.join(p.name for p in occupied)}); "
            "this move is one-shot and will not merge two state trees"
        )
    destination.mkdir(parents=True, exist_ok=True)
    moved: list[tuple[Path, Path]] = []
    for entry in sorted(source.iterdir()):
        target = destination / entry.name
        try:
            os.rename(entry, target)
        except OSError as error:
            raise MigrationRefused(
                f"cannot move {entry} to {target}: {error} "
                "(a rename is only possible within one filesystem; this script does not copy)"
            ) from error
        moved.append((entry, target))
    relative_current_symlink(destination / "bridge")
    source.rmdir()
    return moved


def main() -> int:
    emulators = running_emulators()
    if emulators:
        print(
            "refusing to move project state while an emulator is running: "
            + "; ".join(emulators),
            file=sys.stderr,
        )
        return 1
    source, destination = FORMER_STATE_DIRECTORY, state_directory()
    try:
        moved = migrate(source, destination)
    except MigrationRefused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1
    if not moved:
        print(f"nothing to move: {source} does not exist")
        return 0
    for old, new in moved:
        print(f"{old} -> {new}")
    print(f"{source} is gone; project state is under {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
