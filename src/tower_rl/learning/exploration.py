"""How much of the fleet's collection is spent off the greedy policy, per actor.

One run has one anneal - the fleet's exploration rate falls from `epsilon_start`
to `epsilon_end` over a horizon in decisions and is held there - and, optionally,
a floor of its own for each actor.  The two compose by `max`: an actor never
explores less than the fleet's current rate, and never less than its own floor,
so a segment resumed past the end of the anneal collects at the floors alone.

The floors are Ape-X's (Horgan et al. 2018, arXiv:1803.00933): actor `i` of `N`
acts at `epsilon_i = 0.4 ** (1 + 7 * i / (N - 1))`, spanning 0.4 down to 0.00066
for seven actors.  One fleet then both searches and reports: the high actors play
build orders the greedy policy would never reach, while the near-greedy ones -
at or under `NEAR_GREEDY_EPSILON` - keep producing a collection curve that can
still be read as the policy's own performance.

A uniform schedule has no per-actor floors at all: every actor draws the one
annealed rate, which is what every run before this one collected under, and the
whole fleet is therefore near-greedy in the only sense the curve cares about -
the episodes are all of one policy at one exploration rate.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The rate at or under which an actor's episodes are read as the policy's own
#: performance rather than as search.
NEAR_GREEDY_EPSILON = 0.02

#: Ape-X's two constants, verbatim from the paper: `epsilon = 0.4`, `alpha = 7`.
APE_X_EPSILON = 0.4
APE_X_ALPHA = 7.0

#: What `--exploration` accepts, resolved by `ExplorationSchedule.for_option`.
UNIFORM = "uniform"
LADDER = "ladder"
EXPLORATION_OPTIONS = (UNIFORM, LADDER)


def ape_x_floors(actors: int) -> tuple[float, ...]:
    """The Ape-X ladder for a fleet of `actors`, lowest index highest rate.

    `i / (N - 1)` is undefined for a fleet of one, which the paper never
    considers; it is taken as 0 here, so a single actor gets the base rate 0.4
    and the ladder degenerates to the search end of itself rather than to a
    division by zero.
    """
    if actors < 1:
        raise ValueError("a fleet needs at least one actor")
    span = max(1, actors - 1)
    return tuple(
        APE_X_EPSILON ** (1.0 + APE_X_ALPHA * index / span) for index in range(actors)
    )


@dataclass(frozen=True)
class ExplorationSchedule:
    """The rate each actor explores at, at each point of the budget.

    `floors` is empty for a uniform schedule and holds one rate per actor for a
    ladder; nothing else distinguishes the two, and no caller branches on which
    it holds.
    """

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    anneal_decisions: int = 10_000
    floors: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.anneal_decisions < 1:
            raise ValueError("the epsilon anneal horizon must be positive")

    @classmethod
    def for_option(
        cls,
        option: str,
        *,
        actors: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        anneal_decisions: int = 10_000,
    ) -> ExplorationSchedule:
        """Resolve `--exploration` once, where the command line is read."""
        if option not in EXPLORATION_OPTIONS:
            raise ValueError(f"unknown exploration option: {option}")
        return cls(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            anneal_decisions=anneal_decisions,
            floors=() if option == UNIFORM else ape_x_floors(actors),
        )

    @property
    def option(self) -> str:
        """Which of `EXPLORATION_OPTIONS` this schedule is, for the record."""
        return UNIFORM if not self.floors else LADDER

    def annealed(self, decisions: int) -> float:
        """The fleet's rate: anneal over the horizon, then hold.

        Deliberately not over the budget - annealing across the whole budget
        spent over half the first run above 0.5, so most of what it collected
        was near-random and its collection curve could not be read as a policy's
        performance at all.
        """
        fraction = min(1.0, decisions / self.anneal_decisions)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * fraction

    def floor_for(self, actor_index: int) -> float:
        """This actor's own rate, below which it never drops."""
        return self.epsilon_end if not self.floors else self.floors[actor_index]

    def epsilon_for(self, actor_index: int, decisions: int) -> float:
        """What actor `actor_index` acts at, this far into the budget."""
        return max(self.annealed(decisions), self.floor_for(actor_index))

    def is_near_greedy(self, actor_index: int) -> bool:
        """Whether this actor's episodes are read as performance, not search.

        Every actor of a uniform schedule is: they share the one rate, so the
        near-greedy series and the pooled series are the same episodes.
        """
        return not self.floors or self.floors[actor_index] <= NEAR_GREEDY_EPSILON


__all__ = [
    "APE_X_ALPHA",
    "APE_X_EPSILON",
    "EXPLORATION_OPTIONS",
    "LADDER",
    "NEAR_GREEDY_EPSILON",
    "UNIFORM",
    "ExplorationSchedule",
    "ape_x_floors",
]
