"""Policies that share the learner's interface, including the non-learned floors.

Random and scripted play are not scaffolding: without them, "it learned" is
unfalsifiable.  They implement the same protocol as a backbone so the actor,
evaluator and reporting path are identical for every arm of the comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from tower_rl.domain.features import ROW_WIDTH, StateFeatures
from tower_rl.domain.run_actions import RUN_ACTIONS


class Policy(Protocol):
    """Anything that can choose a valid action, learned or not."""

    def initial_state(self) -> Any: ...

    def act(self, features: StateFeatures, state: Any, *, epsilon: float) -> tuple[int, Any]: ...

    def stored_recurrent_state(self, state: Any) -> Any:
        """The carried state as replay must keep it, or `None` to keep nothing.

        R2D2 section 2.3 stores the recurrent state a window began from so the
        learner burns in from it rather than from zeros. Only a policy that
        carries a recurrent state has anything to store; every other policy says
        so by returning `None`, and then no window carries a state nothing reads.
        An implementation that does store must detach and move to CPU, because a
        replayed state outlives both the graph and the device that produced it.
        """
        ...


def valid_actions(features: StateFeatures) -> list[int]:
    return [index for index, allowed in enumerate(features.mask) if allowed]


@dataclass
class RandomPolicy:
    """Uniform over currently valid actions. The floor every arm must clear."""

    seed: int | None = None
    _random: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def initial_state(self) -> None:
        return None

    def stored_recurrent_state(self, state: None) -> None:
        """Carries no state across steps, so a window stores none."""
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        choices = valid_actions(features)
        if not choices:
            raise ValueError("no action is available in this state")
        return self._random.choice(choices), None


@dataclass
class CheapestFirstPolicy:
    """Buy the cheapest affordable upgrade, otherwise wait.

    This is the scripted policy measured in `M1B-E003`, reaching wave eight to ten
    where buying nothing dies at wave two. It is the harder floor: beating random
    proves very little, beating this proves something.
    """

    #: Index of `cost_log` inside a row, which orders identically to raw cost.
    cost_feature: int = 0

    def initial_state(self) -> None:
        return None

    def stored_recurrent_state(self, state: None) -> None:
        """Carries no state across steps, so a window stores none."""
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        choices = valid_actions(features)
        if not choices:
            raise ValueError("no action is available in this state")
        purchases = [index for index in choices if index != 0]
        if not purchases:
            return 0, None
        cheapest = min(purchases, key=lambda index: self._cost(features, index))
        return cheapest, None

    def _cost(self, features: StateFeatures, action_index: int) -> float:
        # Action index 0 is WAIT, so row `i` backs action index `i + 1`.
        row = (action_index - 1) * ROW_WIDTH
        return features.rows[row + self.cost_feature]


@dataclass
class WaitOnlyPolicy:
    """Never buys. The degenerate reference that dies at wave two."""

    def initial_state(self) -> None:
        return None

    def stored_recurrent_state(self, state: None) -> None:
        """Carries no state across steps, so a window stores none."""
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        if not features.mask[0]:
            raise ValueError("WAIT is not available in this state")
        return 0, None


def describe(policy: Policy) -> str:
    """A stable name for reports, so arms stay identifiable across runs."""
    return type(policy).__name__


assert len(RUN_ACTIONS) > 1, "the action space must contain WAIT plus upgrades"
