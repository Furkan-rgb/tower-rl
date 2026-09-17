"""The port an instrumented run environment drives.

The environment owns cadence, transitions, reward, and episode classification.
The adapter behind this port owns transport, the protocol, and how frames are
actually stepped.  Keeping that in the adapter is deliberate: it depends on the
wire protocol, which the environment should not know about.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tower_rl.domain.run_state import ExactRunReadingLike


@runtime_checkable
class CommandResultLike(Protocol):
    """The game-owned verdict on one requested command.

    Read-only, because the environment only ever inspects a verdict. Declaring
    these as settable attributes would make them invariant, which would reject an
    adapter that reports a `StrEnum` outcome even though a `StrEnum` is a `str`.
    """

    @property
    def outcome(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def frames(self) -> int: ...

    @property
    def game_ms(self) -> float: ...


@runtime_checkable
class AdvanceResultLike(CommandResultLike, Protocol):
    """One advance's verdict together with the state it ended on.

    The port returns the reading the game settled at when it stopped advancing,
    so the environment does not pay a second round trip to look at a state the
    port already held.
    """

    @property
    def round_ms(self) -> float: ...

    @property
    def wall_micros(self) -> int: ...

    @property
    def state(self) -> ExactRunReadingLike | None: ...


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

    def buy_upgrade(self, family: str, slot: int, *, expected_sequence: int) -> AdvanceResultLike:
        """Request one earned-cash purchase bound to the state it was decided from.

        Like `advance_until_event`, the bridge sends the settled observation
        immediately before the result, whatever the outcome, so the result
        carries the state the caller would otherwise pay a second round trip
        to read.
        """
        ...

    def advance_until_event(
        self,
        *,
        expected_sequence: int,
        budget_game_ms: int,
        frame_game_ms: float,
        health_change_fraction: float,
    ) -> AdvanceResultLike:
        """Advance frames until a decision event occurs or the budget is spent.

        One call per decision, not one per slice. The port evaluates the same
        decision conditions the environment does and stops at the first one, so
        the round trip is paid once; `reason` says which condition stopped it.
        The environment re-checks the returned state against its own predicate
        and treats a disagreement as an invalid transition, so this is an
        optimisation of *when* to look, never the definition of what counts.

        `state` is the settled reading the advance ended on, not a fresh read:
        the port has already paused the world and let the pause land, so waiting
        for another reading would cost a round trip and show the same state.
        """
        ...
