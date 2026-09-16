"""The port an instrumented run environment drives.

The environment owns cadence, transitions, reward, and episode classification.
The adapter behind this port owns transport, the protocol, and the choice between
pausing and free running.  Keeping that choice in the adapter is deliberate: it
depends on the current game speed and the wire protocol, neither of which the
environment should know about.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tower_rl.domain.run_state import ExactRunReadingLike


@runtime_checkable
class CommandResultLike(Protocol):
    """The game-owned verdict on one requested command."""

    outcome: str
    reason: str


class RunPortError(RuntimeError):
    """The port could not complete a request; the caller classifies the episode."""


class RunPort(Protocol):
    """One instrumented game instance, addressed semantically."""

    def read_state(self) -> ExactRunReadingLike | None:
        """Return the freshest exact reading, or None when no run is initialized."""
        ...

    def begin_episode(self) -> None:
        """Bring the instance into an active run, raising `RunPortError` if it cannot."""
        ...

    def buy_upgrade(self, family: str, slot: int, *, expected_sequence: int) -> CommandResultLike:
        """Request one earned-cash purchase bound to the state it was decided from."""
        ...

    def advance(self, *, expected_sequence: int, game_ms: int) -> CommandResultLike:
        """Advance approximately `game_ms` of game time and return with fresh state."""
        ...
