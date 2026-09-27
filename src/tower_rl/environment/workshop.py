"""The Workshop runway profile: fixed permanent levels an operator gives a run (ADR 0012).

The agent cannot earn permanent upgrades, so a run may be played with a
modest, fixed Workshop setup instead of none. It is environment configuration
the controller owns - chosen once per run by the operator, applied before every
round on the disposable instance, never persisted and never a policy action -
and the agent still has to earn every in-run upgrade on top of it.

The game calls the Workshop "Enhancement". Its rows are addressed here by the
name the game gives them and resolved against the live name arrays by the
bridge, so a guessed index can never write the wrong row.
"""

from __future__ import annotations

#: The rows the runway profile raises, by their in-game names. Every other
#: Workshop row stays where the account has it. Why these and not the others is
#: ADR 0012's: Orbs one-shot normal enemies, Interest is worth $0, the Enemy
#: Level Skips remove the enemies' own scaling, and Death Defy, Recovery and the
#: Wall are extra lives.
#:
#: The names are the in-run upgrade rows' names, read off the device
#: (`slot_labels`); that the Workshop names its rows identically is not yet
#: confirmed on the device. The bridge refuses a name it cannot find by name
#: (`workshop_row_unknown:<name>`), and `run_episodes.py --list-workshop-rows`
#: prints the names it has.
WORKSHOP_RUNWAY_ROWS: tuple[str, ...] = (
    "Damage",
    "Attack Speed",
    "Critical Chance",
    "Critical Factor",
    "Health",
    "Health Regen",
    "Defense %",
    "Defense Absolute",
    "Thorn Damage",
    "Cash Bonus",
    "Cash / Wave",
)

#: The level that means "no profile": nothing is written, and the run is played
#: on the account exactly as the image holds it (baseline v1).
WORKSHOP_OFF = 0


def workshop_rows(level: int) -> tuple[str, ...]:
    """The rows a run at `level` writes: the profile's rows, or none when it is off."""
    if level < WORKSHOP_OFF:
        raise ValueError(f"a Workshop level cannot be negative: {level}")
    return WORKSHOP_RUNWAY_ROWS if level > WORKSHOP_OFF else ()
