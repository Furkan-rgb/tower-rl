"""A deterministic in-process stand-in for one instrumented game instance.

This is a TEST DOUBLE and lives under `tests/` on purpose. It exists so the
environment, replay, learner and actor can be exercised without a device. It is
not a model of The Tower, it is never importable from `src/`, and no transition
it produces may ever enter training replay: the project's objective requires that
primary experience come from the real game.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tower_rl.environment.run_actions import SLOTS_PER_FAMILY
from tower_rl.environment.run_port import RunPortError
from tower_rl.environment.run_state import LIVE_WIRE_NAMES, NO_ENEMY_DISTANCE

FAMILIES = ("attack", "defense", "utility")


@dataclass(frozen=True)
class FakeUpgradeReading:
    """One upgrade slot as this instance reports it (`UpgradeEntryLike`)."""

    family: str
    index: int
    cost: float
    level: int
    max_level: int
    unlocked: bool
    tier_unlocked: bool
    maxed: bool


@dataclass(frozen=True)
class FakeRunReading:
    """One exact reading of this instance (`ExactRunReadingLike`).

    Deliberately declared here from the environment's protocols rather than
    reusing the bridge's own observation type: a double that reached into
    `infrastructure` would make the environment untestable without the
    transport it is supposed to be independent of. The field set still mirrors
    what the bridge transmits, `terminal`, `round_active` and `play_time`
    included, so a test can substitute either shape for the other.
    """

    sequence: int
    lifecycle: str
    wave: int
    cash: float
    health: float
    max_health: float
    terminal: bool
    round_active: bool
    game_speed: float
    play_time: float
    upgrades: tuple[FakeUpgradeReading, ...]
    #: Every `observation-v2` live reading, raw, under the game's own `Main`
    #: field name. The whole set is always present, because the real bridge
    #: refuses to start without it and a double that omitted one would let a
    #: missing reading pass every test.
    live: Mapping[str, float]


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
    frames: int = 0
    game_ms: float = 0.0
    round_ms: float = 0.0
    wall_micros: int = 0
    #: The settled reading an advance ended on, exactly as the real bridge sends
    #: the observation its result describes.
    state: FakeRunReading | None = None


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
    #: Episode ordinals, counting from one, whose `begin_episode` fails. Unlike
    #: `refuse_to_start`, which refuses every episode, this makes one episode in
    #: the middle of a run fail the way a boundary that will not open does.
    refuse_episodes: frozenset[int] = frozenset()
    #: Episode ordinals whose first advance comes back ambiguous: the shape of an
    #: action pipeline that cannot say what the world did, which the environment
    #: classifies as `ACTION_PIPELINE_FAILED`.
    ambiguous_advance_episodes: frozenset[int] = frozenset()
    #: Episode ordinals whose first advance is refused because the command no
    #: longer binds the bridge's latest observation. The adapter reports that
    #: refusal as a port failure, which is what lets one lost episode cost an
    #: episode rather than the whole run (M1B-E024).
    stale_advance_episodes: frozenset[int] = frozenset()
    #: How much game time this world really simulates per millisecond of game
    #: time an advance budgets for. 1.0 is the world the bridge asks for. Setting
    #: it above 1.0 simulates the defect a speed multiplier left applied
    #: produces: every frame is worth more than `frame_game_ms`, so the round
    #: clock outruns the budget and the run progresses faster than the record
    #: can account for (M1B-E023).
    world_time_scale: float = 1.0
    #: The wave `begin_episode` starts at. A real fresh run always starts at 1;
    #: setting this above 1 simulates continuing a leftover run, to exercise the
    #: episode-independence check without a second fake port.
    starting_wave: int = 1
    #: The real bridge's round clock resets with the round, so the one advance
    #: that ends a run reports zero round time for it even though game time was
    #: spent reaching the end. Off by default because most fakes have no reason
    #: to model it; a test of the round-clock guard's lower bound turns it on to
    #: exercise that legitimate zero without it being mistaken for a defect.
    round_clock_resets_on_death: bool = False

    sequence: int = field(default=0, init=False)
    #: Episodes begun, the ordinal the two failure sets are matched against.
    episodes: int = field(default=0, init=False)
    _ambiguous_seen: set[int] = field(default_factory=set, init=False)
    _stale_seen: set[int] = field(default_factory=set, init=False)
    wave: int = field(default=0, init=False)
    cash: float = field(default=0.0, init=False)
    health: float = field(default=0.0, init=False)
    elapsed_ms: float = field(default=0.0, init=False)
    active: bool = field(default=False, init=False)
    slots: dict[tuple[str, int], _Slot] = field(default_factory=dict, init=False)
    advances: int = field(default=0, init=False)
    #: How many times the environment asked for a state of its own accord. One
    #: decision must not cost one of these on top of its advance.
    reads: int = field(default=0, init=False)

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
        self.episodes += 1
        if self.refuse_to_start or self.episodes in self.refuse_episodes:
            raise RunPortError("fake instance refused to start")
        self._build_slots()
        self.wave = self.starting_wave
        self.cash = self.start_cash
        self.health = self.max_health
        self.elapsed_ms = 0.0
        self.active = True
        self.sequence += 1

    def read_state(self) -> FakeRunReading | None:
        self.reads += 1
        return self._observe()

    def _observe(self) -> FakeRunReading | None:
        if not self.active and self.health > 0.0:
            return None
        self.sequence += 1
        return FakeRunReading(
            sequence=self.sequence,
            lifecycle="active" if self.active else "terminal",
            wave=self.wave,
            cash=round(self.cash, 3),
            health=max(0.0, round(self.health, 3)),
            max_health=self.max_health,
            terminal=not self.active,
            round_active=self.active,
            # The real game stops time on death, so a terminal reading carries
            # speed zero (M1B-E009). The double has to do the same, or code that
            # samples speed at the wrong moment looks correct here.
            game_speed=self.game_speed if self.active else 0.0,
            play_time=100.0 + self.elapsed_ms / 1000.0,
            live=self._live(),
            upgrades=tuple(
                FakeUpgradeReading(
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

    def _live(self) -> dict[str, float]:
        """Every live reading this world has, and a resting zero for the rest.

        The few this double genuinely models move with its own run, so a test of
        the schema sees values that change the way the game's do. Everything
        else reads zero, which is exactly what the device saw for every upgrade
        a run never bought (board #39) and is therefore a legitimate reading
        rather than a hole.
        """
        seconds = self.elapsed_ms / 1000.0
        wave_seconds = seconds % self.seconds_per_wave
        live = dict.fromkeys(LIVE_WIRE_NAMES, 0.0)
        live["damage"] = 3.0 + sum(slot.level for slot in self.slots.values())
        live["attackSpeed"] = 1.0
        # Percent on the wire, as `criticalChance` reads on the device.
        live["criticalChance"] = 1.0
        live["criticalMult"] = 1.2
        live["multishotTargets"] = 2.0
        live["towerRangeDistance"] = 2.7 if self.active else 0.0
        live["wallHealth"] = 0.2 * self.max_health
        live["currentWaveBaseHealth"] = 2.35 * 1.29 ** (self.wave - 1)
        live["currentWaveBaseDamage"] = 1.176 * 1.18 ** (self.wave - 1)
        live["currentWaveBaseKillCash"] = 1.0
        live["enemiesSpawnedThisWave"] = float(int(wave_seconds))
        live["enemiesKilledThisWave"] = float(int(wave_seconds))
        live["estimatedEnemiesToSpawnThisWave"] = float(20 + self.wave)
        # The game stores a sentinel rather than an absence, and an idle world
        # has no enemy at all.
        live["closestEnemyDistance"] = 1.5 if self.active else NO_ENEMY_DISTANCE
        live["waveTimer"] = wave_seconds
        live["waveLengthSeconds"] = self.seconds_per_wave
        live["waveCooldownSeconds"] = 1.0
        live["cashEarnedThisWave"] = self.cash_per_second * wave_seconds
        live["gameplayTimeThisRound"] = seconds
        return live

    def buy_upgrade(
        self, family: str, slot_index: int, *, expected_sequence: int
    ) -> FakeCommandResult:
        # The real bridge sends the settled observation immediately before every
        # command result, whatever the outcome, so this double binds one too:
        # a test that only reads `result.state` must see what the bridge would
        # actually hand back rather than what a fresh `read_state()` would show.
        if expected_sequence != self.sequence:
            return FakeCommandResult("rejected", "stale_or_duplicate", state=self._observe())
        slot = self.slots[(family, slot_index)]
        if not slot.unlocked or slot.maxed or slot.cost <= 0 or slot.cost > self.cash:
            return FakeCommandResult("rejected", "precondition_failed", state=self._observe())
        self.cash -= slot.cost
        slot.level += 1
        slot.cost = round(slot.cost * 1.5, 3)
        # Each purchase buys a little survivability, so a buying policy outlives
        # one that only waits. That ordering is what the tests rely on.
        self.damage_per_second = max(0.02, self.damage_per_second * 0.82)
        self.sequence += 1
        return FakeCommandResult("confirmed", "confirmed_state_change", state=self._observe())

    def advance_until_event(
        self,
        *,
        expected_sequence: int,
        budget_game_ms: int,
        frame_game_ms: float,
        health_change_fraction: float,
    ) -> FakeCommandResult:
        """Step frame by frame until a decision event or the budget, as the bridge does.

        The reason it reports is derived from its own transitions, so a test that
        wants the environment's divergence check to fire has to make this double
        lie on purpose.
        """
        if expected_sequence != self.sequence:
            return FakeCommandResult("rejected", "stale_or_duplicate")
        if (
            self.episodes in self.stale_advance_episodes
            and self.episodes not in self._stale_seen
        ):
            # Once per named episode: the sequence the environment was holding
            # no longer exists, so the port can only refuse.
            self._stale_seen.add(self.episodes)
            raise RunPortError("the bridge refused a stale command: does not bind the latest")
        if (
            self.episodes in self.ambiguous_advance_episodes
            and self.episodes not in self._ambiguous_seen
        ):
            # Once per named episode: the bridge could not say how far it got.
            self._ambiguous_seen.add(self.episodes)
            return FakeCommandResult("ambiguous", "no_answer_from_the_bridge")
        self.advances += 1
        if not self.active:
            return FakeCommandResult("confirmed", "event:run_ended", state=self._observe())
        wave = self.wave
        health_fraction = self._transmitted_health_fraction()
        affordable = self._affordable()
        round_time_before = self.elapsed_ms

        frames = 0
        spent = 0.0
        reason = "budget_exhausted"
        while spent < budget_game_ms:
            self._step_one_frame(frame_game_ms)
            frames += 1
            spent += frame_game_ms
            if not self.active:
                reason = "event:run_ended"
                break
            if self.wave != wave:
                reason = "event:wave_changed"
                break
            if self._affordable() - affordable:
                reason = "event:newly_affordable"
                break
            if (
                abs(self._transmitted_health_fraction() - health_fraction)
                >= health_change_fraction
            ):
                reason = "event:health_changed"
                break
        round_ms = self.elapsed_ms - round_time_before
        if reason == "event:run_ended" and self.round_clock_resets_on_death:
            round_ms = 0.0
        return FakeCommandResult(
            "confirmed",
            reason,
            frames=frames,
            game_ms=spent,
            round_ms=round_ms,
            wall_micros=frames * 100,
            state=self._observe(),
        )

    def _step_one_frame(self, frame_game_ms: float) -> None:
        seconds = frame_game_ms * self.world_time_scale / 1000.0
        self.elapsed_ms += frame_game_ms * self.world_time_scale
        self.cash += self.cash_per_second * seconds
        self.health -= self.damage_per_second * seconds
        self.wave = self.starting_wave + int(self.elapsed_ms / 1000.0 / self.seconds_per_wave)
        if self.health <= 0.0:
            self.health = 0.0
            self.active = False

    def _transmitted_health_fraction(self) -> float:
        """Exactly what the environment will see as health, rounding included.

        The stopping predicate has to read the value that `_observe` transmits,
        not the internal float: a delta that rounds to just under the threshold
        would otherwise be reported as `event:health_changed`, and the
        environment would correctly call that a bridge-event divergence.
        """
        return max(0.0, round(self.health, 3)) / self.max_health

    def _affordable(self) -> set[tuple[str, int]]:
        """Exactly what the environment will see as available, rounding included."""
        cash = round(self.cash, 3)
        return {
            key
            for key, slot in self.slots.items()
            if slot.unlocked and not slot.maxed and 0 < slot.cost <= cash
        }
