"""A deterministic in-process stand-in for one instrumented game instance.

This is a TEST DOUBLE and lives under `tests/` on purpose. It exists so the
environment, replay, learner and actor can be exercised without a device. It is
not a model of The Tower, it is never importable from `src/`, and no transition
it produces may ever enter training replay: the project's objective requires that
primary experience come from the real game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tower_rl.domain.run_actions import SLOTS_PER_FAMILY
from tower_rl.infrastructure.instrumented_bridge import BridgeObservation, UpgradeInventoryEntry
from tower_rl.ports.run_port import RunPortError

FAMILIES = ("attack", "defense", "utility")


@dataclass
class _Slot:
    cost: float
    level: int = 0
    max_level: int = 50
    unlocked: bool = False

    @property
    def maxed(self) -> bool:
        return self.level >= self.max_level


@dataclass
class FakeCommandResult:
    outcome: str
    reason: str


@dataclass
class FakeRunPort:
    """A tiny, fully deterministic run: cash accrues, health decays, waves pass."""

    #: How many slots per family the game offers at this fake baseline.
    offered: dict[str, int] = field(
        default_factory=lambda: {"attack": 4, "defense": 2, "utility": 0}
    )
    cash_per_second: float = 8.0
    damage_per_second: float = 0.25
    seconds_per_wave: float = 6.0
    max_health: float = 5.0
    game_speed: float = 8.0
    start_cash: float = 80.0
    #: Set to raise from `begin_episode`, to exercise failure classification.
    refuse_to_start: bool = False

    sequence: int = field(default=0, init=False)
    wave: int = field(default=0, init=False)
    cash: float = field(default=0.0, init=False)
    health: float = field(default=0.0, init=False)
    elapsed_ms: int = field(default=0, init=False)
    active: bool = field(default=False, init=False)
    slots: dict[tuple[str, int], _Slot] = field(default_factory=dict, init=False)
    advances: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._build_slots()

    def _build_slots(self) -> None:
        self.slots = {}
        for family in FAMILIES:
            for index in range(SLOTS_PER_FAMILY):
                self.slots[(family, index)] = _Slot(
                    cost=5.0 + 5.0 * index,
                    unlocked=index < self.offered[family],
                )

    # -- RunPort -----------------------------------------------------------

    def begin_episode(self) -> None:
        if self.refuse_to_start:
            raise RunPortError("fake instance refused to start")
        self._build_slots()
        self.wave = 1
        self.cash = self.start_cash
        self.health = self.max_health
        self.elapsed_ms = 0
        self.active = True
        self.sequence += 1

    def read_state(self) -> BridgeObservation | None:
        if not self.active and self.health > 0.0:
            return None
        self.sequence += 1
        return BridgeObservation(
            sequence=self.sequence,
            lifecycle="active" if self.active else "terminal",
            wave=self.wave,
            cash=round(self.cash, 3),
            health=max(0.0, round(self.health, 3)),
            max_health=self.max_health,
            terminal=not self.active,
            round_active=self.active,
            game_speed=self.game_speed,
            play_time=100.0 + self.elapsed_ms / 1000.0,
            upgrades=tuple(
                UpgradeInventoryEntry(
                    family=family,
                    index=index,
                    cost=slot.cost,
                    level=slot.level,
                    max_level=slot.max_level,
                    unlocked=slot.unlocked,
                    tier_unlocked=False,
                    maxed=slot.maxed,
                )
                for (family, index), slot in self.slots.items()
            ),
        )

    def buy_upgrade(
        self, family: str, slot_index: int, *, expected_sequence: int
    ) -> FakeCommandResult:
        if expected_sequence != self.sequence:
            return FakeCommandResult("rejected", "stale_or_duplicate")
        slot = self.slots[(family, slot_index)]
        if not slot.unlocked or slot.maxed or slot.cost <= 0 or slot.cost > self.cash:
            return FakeCommandResult("rejected", "precondition_failed")
        self.cash -= slot.cost
        slot.level += 1
        slot.cost = round(slot.cost * 1.5, 3)
        # Each purchase buys a little survivability, so a buying policy outlives
        # one that only waits. That ordering is what the tests rely on.
        self.damage_per_second = max(0.02, self.damage_per_second * 0.82)
        self.sequence += 1
        return FakeCommandResult("confirmed", "confirmed_state_change")

    def advance(self, *, expected_sequence: int, game_ms: int) -> FakeCommandResult:
        if expected_sequence != self.sequence:
            return FakeCommandResult("rejected", "stale_or_duplicate")
        self.advances += 1
        if not self.active:
            return FakeCommandResult("confirmed", "step_elapsed")
        seconds = game_ms / 1000.0
        self.elapsed_ms += game_ms
        self.cash += self.cash_per_second * seconds
        self.health -= self.damage_per_second * seconds
        self.wave = 1 + int(self.elapsed_ms / 1000.0 / self.seconds_per_wave)
        if self.health <= 0.0:
            self.health = 0.0
            self.active = False
        self.sequence += 1
        return FakeCommandResult("confirmed", "step_elapsed")
