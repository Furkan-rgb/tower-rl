#!/usr/bin/env python3
"""Read-only host, SDK, package, and Android-device diagnostics.

`tower_rl.doctor` holds every check; this is its one command, so there is
exactly one `doctor` mode rather than two overlapping half-doctors. Both the
XAPK and the device serial are optional, so this is the first command a fresh
checkout runs, before `local/*.xapk` is even in place:

    ./scripts/doctor.py
    ./scripts/doctor.py --xapk local/the-tower-29-0-1.xapk --serial emulator-5554
    ./scripts/doctor.py --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tower_rl.doctor import CheckStatus, render_json, run_doctor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xapk", type=Path, default=None, help="reference XAPK to inspect")
    parser.add_argument("--serial", default=None, help="ADB serial to check, if any")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    arguments = parser.parse_args()

    results = run_doctor(arguments.xapk, arguments.serial)

    if arguments.json:
        print(render_json(results))
    else:
        for result in results:
            print(f"[{result.status.value.upper():4}] {result.name}: {result.message}")
            for key, value in sorted(result.details.items()):
                print(f"       {key}: {value}")

    return 0 if not any(result.status is CheckStatus.FAIL for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
