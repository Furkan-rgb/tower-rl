"""The port an instrumented run environment drives.

The environment owns cadence, transitions, reward, and episode classification.
The adapter behind this port owns transport, the protocol, and how frames are
actually stepped.  Keeping that in the adapter is deliberate: it depends on the
wire protocol, which the environment should not know about.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from tower_rl.environment.run_state import ExactRunReadingLike


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


@runtime_checkable
class UpgradeSlotLabelLike(Protocol):
    """What the game calls one upgrade row.

    The environment wants one thing from a label: whether the row exists at
    all. The three name arrays are twenty slots wide and their tails are empty,
    so a slot with no name is a slot the game does not offer - which is what
    `upgrade availability` has to be read against (ADR 0011).
    """

    @property
    def family(self) -> str: ...

    @property
    def index(self) -> int: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class UnlockFamilyStateLike(Protocol):
    """How much of one family's in-run availability array stands true.

    A length and a count, read back out of the game after the write, never a
    restatement of what was asked for: a write that did not take reports as one.
    """

    @property
    def family(self) -> str: ...

    @property
    def length(self) -> int: ...

    @property
    def true_count(self) -> int: ...


class RunPortError(RuntimeError):
    """The port could not complete a request; the caller classifies the episode."""


class RunPort(Protocol):
    """One instrumented game instance, addressed semantically."""

    @property
    def pin_restarts(self) -> int:
        """Boundaries this port restarted because the speed pin was not held.

        Cumulative over the port's life. The environment reports the episode's
        own share of it, so a recovery the port made silently still shows up in
        the episode record it made room for (`#57`).
        """
        ...

    def read_state(self) -> ExactRunReadingLike | None:
        """Return the freshest exact reading, or None when no run is initialized."""
        ...

    def begin_episode(self) -> None:
        """Bring the instance into an active run, raising `RunPortError` if it cannot."""
        ...

    def slot_labels(self) -> Sequence[UpgradeSlotLabelLike]:
        """What the game calls each upgrade row; constant for a build.

        Asked for before a round rather than inside one, and answered from the
        port's own cache thereafter.
        """
        ...

    def unlock_all_upgrades(self) -> Sequence[UnlockFamilyStateLike]:
        """Make every in-run upgrade row purchasable, and report what then stands.

        Issued at a round start under `UpgradeAvailability.ALL` and never
        otherwise. The game recomputes its real rows' availability whenever a
        round begins (`M2-E008`), so this is a round-scoped capability rather
        than a property of the image, and the counts come back read out of the
        game after the write so a write that did not take is visible here.
        """
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
