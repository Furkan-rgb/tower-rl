"""Run the M1 scripted-controller reliability gate on one provisioned AVD."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tower_rl.application.controller import ControllerError, TowerController
from tower_rl.domain import RunAction, ScreenState
from tower_rl.infrastructure.adb_device import AdbDevice


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = AdbDevice(args.serial)
    controller = TowerController(device)
    report: dict[str, object] = {
        "episodes_requested": args.episodes,
        "valid_episodes": 0,
        "failures": [],
        "started_at_epoch": time.time(),
    }
    try:
        controller.recover(args.snapshot)
        observation = controller.start_episode()
        for episode in range(1, args.episodes + 1):
            if observation.screen is not ScreenState.ACTIVE_RUN:
                raise ControllerError(f"episode {episode} did not start in active run")
            expected = {action for action in RunAction if action is not RunAction.WAIT}
            if not expected.issubset(set(observation.action_mask)):
                raise ControllerError(f"episode {episode} has incomplete action mask")
            result = controller.await_death()
            if result.screen is not ScreenState.RESULT:
                raise ControllerError(f"episode {episode} did not reach result")
            report["valid_episodes"] = episode
            if episode < args.episodes:
                observation = controller.reset_episode(scan_upgrades=False)
                time.sleep(2.0)
        report["passed"] = True
    except Exception as error:
        failures = report["failures"]
        assert isinstance(failures, list)
        failures.append({"type": type(error).__name__, "message": str(error)})
        try:
            frame = device.screenshot()
            failure_frame = args.output.with_suffix(".failure.png")
            failure_frame.write_bytes(frame.png_bytes)
            report["failure_frame"] = str(failure_frame)
        except Exception as capture_error:
            report["failure_frame_error"] = str(capture_error)
        report["passed"] = False
    finally:
        try:
            controller.recover(args.snapshot)
            report["baseline_restored"] = True
        except Exception as error:
            report["baseline_restored"] = False
            failures = report["failures"]
            assert isinstance(failures, list)
            failures.append({"type": type(error).__name__, "message": str(error)})
        report["finished_at_epoch"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report.get("passed") is True and report.get("baseline_restored") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
