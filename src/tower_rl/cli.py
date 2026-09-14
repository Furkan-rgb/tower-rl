"""Tower-RL command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tower_rl.application import ProbeService
from tower_rl.doctor import CheckStatus, render_json, run_doctor
from tower_rl.infrastructure import AdbProbe, ProbeError

app = typer.Typer(no_args_is_help=True, help="Control and train Tower-RL.")


@app.callback()
def main() -> None:
    """Control and train Tower-RL."""


@app.command()
def doctor(
    xapk: Annotated[
        Path,
        typer.Option(
            "--xapk",
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the locally supplied XAPK.",
        ),
    ] = Path("local/the-tower.xapk"),
    serial: Annotated[
        str | None,
        typer.Option("--serial", help="ADB serial required for this check."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable report."),
    ] = False,
) -> None:
    """Validate host tools, XAPK metadata, and an optional Android device."""

    results = run_doctor(xapk, serial)
    if json_output:
        typer.echo(render_json(results))
    else:
        for result in results:
            typer.echo(f"{result.status.value.upper():4}  {result.name:16} {result.message}")
    if any(result.status is CheckStatus.FAIL for result in results):
        raise typer.Exit(code=1)


@app.command()
def probe(
    serial: Annotated[str, typer.Option("--serial", help="ADB serial to probe.")],
    navigate: Annotated[
        bool,
        typer.Option(
            "--navigate/--no-navigate",
            help="Run the bounded Home -> Tier 1 -> result -> Home smoke flow.",
        ),
    ] = False,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Optional path for the captured PNG."),
    ] = None,
    restore_snapshot: Annotated[
        str | None,
        typer.Option(
            "--restore-snapshot",
            help="Restore this local AVD snapshot after a navigation probe.",
        ),
    ] = None,
) -> None:
    """Capture and validate the pinned single-device visual baseline."""

    try:
        payload = ProbeService(AdbProbe(serial)).execute(
            navigate=navigate, frame_path=frame, restore_snapshot=restore_snapshot
        )
    except ProbeError as error:
        typer.echo(f"PROBE FAILED  {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
