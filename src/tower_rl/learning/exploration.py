"""How much of the fleet's collection is spent off the greedy policy, per actor.

One run has one anneal: every actor's rate falls linearly from `epsilon_start`
over `anneal_decisions` and is held afterwards.  What it falls *to* is the
actor's own floor.  Under the uniform schedule that floor is `epsilon_end` for
every actor, which is the one rate every run so far collected under; under a
ladder each actor has a floor of its own and `epsilon_end` is not used at all.

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

    #: Named by whoever resolved the command line and by nobody else: there is
    #: no default here, so a run's exploration cannot be half `train.py`'s
    #: parser and half this file's.
    epsilon_start: float
    epsilon_end: float
    anneal_decisions: int
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
        epsilon_start: float,
        epsilon_end: float,
        anneal_decisions: int,
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

    @property
    def reported_actor(self) -> int:
        """Whose rate stands for the run's when only one number can be carried.

        The lowest-index near-greedy actor: actor 0 of a uniform fleet, where
        every actor draws that one rate anyway, and the top of the near-greedy
        rungs under a ladder. Informational only - under a ladder no single
        number is the fleet's, and anything measuring exploration per episode
        asks `epsilon_for` for the actor that played it.
        """
        return next(
            (index for index in range(len(self.floors)) if self.is_near_greedy(index)),
            0,
        )

    def reported_epsilon(self, decisions: int) -> float:
        """The schedule position a checkpoint and a progress report carry."""
        return self.epsilon_for(self.reported_actor, decisions)

    def floor_for(self, actor_index: int) -> float:
        """The rate this actor anneals to and is then held at.

        `epsilon_end` for every actor of a uniform schedule; its own rung of the
        ladder otherwise, in which case `epsilon_end` plays no part at all.
        """
        return self.epsilon_end if not self.floors else self.floors[actor_index]

    def epsilon_for(self, actor_index: int, decisions: int) -> float:
        """What actor `actor_index` acts at, this far into the budget.

        The horizon is in decisions and is deliberately not the budget:
        annealing across the whole budget spent over half of run 1 above 0.5, so
        most of what it collected was near-random and its collection curve could
        not be read as a policy's performance at all.
        """
        fraction = min(1.0, decisions / self.anneal_decisions)
        floor = self.floor_for(actor_index)
        return self.epsilon_start + (floor - self.epsilon_start) * fraction

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
