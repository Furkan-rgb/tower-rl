"""Policies that share the learner's interface, including the non-learned floors.

Random and scripted play are not scaffolding: without them, "it learned" is
unfalsifiable.  They implement the same protocol as a backbone so the actor,
evaluator and reporting path are identical for every arm of the comparison.

A trained checkpoint is an arm on the same terms (`checkpoint_policy`): it is
rebuilt into the backbone that wrote it and handed back as a policy, so an
evaluation of a checkpoint and an evaluation of the scripted floor differ in
nothing but which policy is asked for an action.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Protocol

import torch

from tower_rl.environment.features import (
    ROW_FEATURES,
    ROW_WIDTH,
    SCALAR_FEATURES,
    StateFeatures,
)
from tower_rl.environment.run_actions import RUN_ACTIONS, action_index, upgrade_action
from tower_rl.environment.run_port import UpgradeSlotLabelLike
from tower_rl.learning.checkpoint import Checkpoint, CheckpointIdentity, load
from tower_rl.learning.dreamer import DREAMERV3, DreamerBackbone, DreamerConfig
from tower_rl.learning.network import NetworkConfig
from tower_rl.learning.stacked_dqn import StackedDqnBackbone, StackedDqnConfig


class Policy(Protocol):
    """Anything that can choose a valid action, learned or not."""

    def initial_state(self) -> Any: ...

    def act(self, features: StateFeatures, state: Any, *, epsilon: float) -> tuple[int, Any]: ...


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
    #: Derived from the schema rather than written as a number: the row layout
    #: grew in `observation-v2`, and a hand-kept 0 would have gone on naming
    #: whichever feature happened to land first and quietly compared the wrong
    #: column - a floor that is silently not the cheapest-first floor.
    cost_feature: int = field(default_factory=lambda: ROW_FEATURES.index("cost_log"))

    def initial_state(self) -> None:
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


#: The in-run rows the turtle build buys, by the game's own row label
#: (`slot_labels`; recorded in `state/records/m3-p004/eval-arm/*.json`). The
#: policy is handed the labels the game reports and resolves these names to
#: slots itself, so it never assumes a slot index.
DEFENSE_ABSOLUTE = "Defense Absolute"
THORN_DAMAGE = "Thorn Damage"
DEFENSE_PERCENT = "Defense %"
HEALTH = "Health"
CASH_PER_WAVE = "Cash / Wave"
KNOCKBACK_CHANCE = "Knockback Chance"
KNOCKBACK_FORCE = "Knockback Force"
ORBS = "Orbs"
ORB_SPEED = "Orb Speed"
TURTLE_ROWS: tuple[str, ...] = (
    DEFENSE_ABSOLUTE,
    THORN_DAMAGE,
    DEFENSE_PERCENT,
    HEALTH,
    CASH_PER_WAVE,
    KNOCKBACK_CHANCE,
    KNOCKBACK_FORCE,
    ORBS,
    ORB_SPEED,
)

#: Each successive hit from the same enemy is this much stronger ("heat-up";
#: fandom Beginner Guide and fandom "Defense Absolute" page, game-vault guide).
HEAT_UP_PER_HIT = 1.04
#: Thorn Damage, in percent, at which a normal enemy dies one hit sooner:
#: 11 % -> 10 hits, 21 -> 5, 26 -> 4, 34 -> 3, 51 -> 2 (fandom Beginner Guide,
#: Discord T1 guide: "keep 1 % above a fraction").
THORN_BREAKPOINTS_PERCENT: tuple[float, ...] = (11.0, 21.0, 26.0, 34.0, 51.0)
#: Hits-to-kill used when Thorns is (near) zero and would otherwise be
#: unbounded; the spec's cap. Reached below 2 % Thorns.
HITS_TO_KILL_CAP = 50
#: Cash / Wave is bought through this wave (fandom Beginner Guide: economy in
#: the first 5-9 waves).
EARLY_ECONOMY_LAST_WAVE = 8
#: Health fraction below which damage is taken to be getting through Defense
#: Absolute, and the build switches to blender (fandom Beginner Guide's in-run
#: trigger, made observable).
SWITCH_HEALTH_FRACTION = 0.8


def hits_to_kill(thorn_percent: float) -> int:
    """Tower hits a normal enemy survives Thorns for: ceil(1 / thorn), capped."""
    if thorn_percent <= 0.0:
        return HITS_TO_KILL_CAP
    return min(HITS_TO_KILL_CAP, math.ceil(100.0 / thorn_percent))


def defense_absolute_margin(thorn_percent: float) -> float:
    """How far above the base hit Defense Absolute must stand to hold.

    An enemy's last hit before Thorns kills it has heated up `hits - 1` times,
    so the margin comes from the same heat-up the guides cite, not a tuned
    constant.
    """
    return HEAT_UP_PER_HIT ** (hits_to_kill(thorn_percent) - 1)


def next_thorn_breakpoint(thorn_percent: float) -> float | None:
    """The next Thorns breakpoint above the current reading; None past the last."""
    return next((bp for bp in THORN_BREAKPOINTS_PERCENT if thorn_percent < bp), None)


def turtle_row_indices(labels: Sequence[UpgradeSlotLabelLike]) -> dict[str, int]:
    """Every `TURTLE_ROWS` name resolved to its action index, from the game's labels.

    Fails loudly on a name the game does not report, or reports twice: a
    missing row must never become a silent fallback to some other purchase.
    """
    rows: dict[str, int] = {}
    for label in labels:
        if label.name not in TURTLE_ROWS:
            continue
        if label.name in rows:
            raise ValueError(f"the game names two rows {label.name!r}")
        rows[label.name] = action_index(upgrade_action(label.family, label.index))
    missing = [name for name in TURTLE_ROWS if name not in rows]
    if missing:
        raise ValueError(f"the game reports no upgrade row named {missing}; turtle cannot play")
    return rows


@dataclass(frozen=True)
class TurtleReading:
    """What the turtle rule reads off one state, in the units the player sees.

    `observation-v2` log-scales all three combat readings (`log1p`), so each is
    un-scaled here. `currentWaveBaseDamage` and `defenseAbs` are flat damage;
    `thornDamage` is the game's percent of enemy max health (5.0 is 5 %: the
    row's label reads "Deals % of Enemy Max Health", and Workshop Thorns 5
    reads 5.0 at wave 1).
    """

    wave: int
    health_fraction: float
    base_damage: float
    defense_absolute: float
    thorn_percent: float

    @classmethod
    def of(cls, features: StateFeatures) -> TurtleReading:
        def hud(name: str) -> float:
            # Rounded so an exact 51 % does not come back as 50.99999999.
            return round(math.expm1(features.scalars[SCALAR_FEATURES.index(name)]), 6)

        return cls(
            wave=round(hud("wave_log")),
            health_fraction=features.scalars[SCALAR_FEATURES.index("health_fraction")],
            base_damage=hud("wave_base_damage_log"),
            defense_absolute=hud("defense_absolute_log"),
            thorn_percent=hud("thorn_damage_log"),
        )

    @property
    def defense_absolute_holds(self) -> bool:
        margin = defense_absolute_margin(self.thorn_percent)
        return self.defense_absolute >= margin * self.base_damage


@dataclass
class TurtlePolicy:
    """A hand-written turtle-then-blender build: the community's reference play.

    Turtle: Defense Absolute until it holds against the heated-up base hit, Cash
    / Wave through wave 8, Thorns to its last breakpoint, then the cheaper of
    Defense % and Health. Once Defense Absolute stops holding at the start of
    two consecutive waves, or health falls below 0.8, it switches for good to
    blender: Thorns to 51 %, Knockback and Orbs, then Health and Defense %. It
    buys no Damage or Attack Speed in either phase, deliberately.

    It addresses rows by name, so it has to be given the game's labels
    (`bind_row_names`) before it can act. Its phase is per-episode memory kept
    on the instance and reset by `initial_state`, like `StackedDqnBackbone`'s
    option counts; `episode_detail` is what the episode record carries.
    """

    rows: dict[str, int] | None = None
    #: The wave the build switched to blender in, this episode; None if it did not.
    switch_wave: int | None = field(default=None, init=False)
    _last_wave_seen: int | None = field(default=None, init=False)
    _last_wave_failed: int | None = field(default=None, init=False)

    def bind_row_names(self, labels: Sequence[UpgradeSlotLabelLike]) -> None:
        self.rows = turtle_row_indices(labels)

    @property
    def episode_detail(self) -> dict[str, Any]:
        return {"switch_wave": self.switch_wave}

    def initial_state(self) -> None:
        self.switch_wave = None
        self._last_wave_seen = None
        self._last_wave_failed = None
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        if self.rows is None:
            raise RuntimeError("turtle was never given the game's row names (bind_row_names)")
        if not valid_actions(features):
            raise ValueError("no action is available in this state")
        reading = TurtleReading.of(features)
        if self.switch_wave is None:
            self._check_switch(reading)
        chosen = (
            self._turtle(features, reading)
            if self.switch_wave is None
            else self._blender(features, reading)
        )
        return chosen, None

    def _check_switch(self, reading: TurtleReading) -> None:
        # The first decision seen in a wave is its start as this policy sees it.
        if reading.wave != self._last_wave_seen:
            self._last_wave_seen = reading.wave
            if reading.defense_absolute_holds:
                self._last_wave_failed = None
            else:
                if self._last_wave_failed == reading.wave - 1:
                    self.switch_wave = reading.wave
                self._last_wave_failed = reading.wave
        if reading.health_fraction < SWITCH_HEALTH_FRACTION:
            self.switch_wave = reading.wave

    def _turtle(self, features: StateFeatures, reading: TurtleReading) -> int:
        if not reading.defense_absolute_holds and self._offered(features, DEFENSE_ABSOLUTE):
            return self._buy(features, DEFENSE_ABSOLUTE)
        if reading.wave <= EARLY_ECONOMY_LAST_WAVE and self._offered(features, CASH_PER_WAVE):
            return self._buy(features, CASH_PER_WAVE)
        if next_thorn_breakpoint(reading.thorn_percent) is not None and self._offered(
            features, THORN_DAMAGE
        ):
            return self._buy(features, THORN_DAMAGE)
        return self._buy(features, DEFENSE_PERCENT, HEALTH)

    def _blender(self, features: StateFeatures, reading: TurtleReading) -> int:
        if next_thorn_breakpoint(reading.thorn_percent) is not None and self._offered(
            features, THORN_DAMAGE
        ):
            return self._buy(features, THORN_DAMAGE)
        affordable = [
            name
            for name in (KNOCKBACK_CHANCE, KNOCKBACK_FORCE, ORBS, ORB_SPEED)
            if features.mask[self._index(name)]
        ]
        if affordable:
            return self._buy(features, *affordable)
        return self._buy(features, HEALTH, DEFENSE_PERCENT)

    def _buy(self, features: StateFeatures, *names: str) -> int:
        """The cheapest offered row of `names`, or WAIT when it is not affordable."""
        offered = [self._index(name) for name in names if self._offered(features, name)]
        if not offered:
            return 0
        cheapest = min(offered, key=lambda index: self._row(features, index, "cost_log"))
        return cheapest if features.mask[cheapest] else 0

    def _offered(self, features: StateFeatures, name: str) -> bool:
        """Whether the row can be bought at all this run: unlocked and not maxed."""
        index = self._index(name)
        return bool(self._row(features, index, "unlocked")) and not self._row(
            features, index, "maxed"
        )

    def _index(self, name: str) -> int:
        assert self.rows is not None
        return self.rows[name]

    @staticmethod
    def _row(features: StateFeatures, action: int, feature: str) -> float:
        # Action index 0 is WAIT, so row `i` backs action index `i + 1`.
        return features.rows[(action - 1) * ROW_WIDTH + ROW_FEATURES.index(feature)]


@dataclass
class WaitOnlyPolicy:
    """Never buys. The degenerate reference that dies at wave two."""

    def initial_state(self) -> None:
        return None

    def act(
        self, features: StateFeatures, state: None, *, epsilon: float = 0.0
    ) -> tuple[int, None]:
        if not features.mask[0]:
            raise ValueError("WAIT is not available in this state")
        return 0, None


def checkpoint_policy(
    path: Path,
    *,
    decision_cadence: str,
    upgrade_availability: str,
    workshop_level: int,
    device: torch.device | None = None,
    sampling_seed: str | None = None,
) -> tuple[StackedDqnBackbone | DreamerBackbone, CheckpointIdentity]:
    """Rebuild the backbone a checkpoint holds, as a policy to evaluate.

    Everything needed to reconstruct it is in the checkpoint: its identity says
    which backbone and which schemas, and the resolved config it was written
    with says how the learner and the network were shaped. Nothing is taken from
    the caller, so a checkpoint cannot be evaluated under settings it was not
    trained under - a network rebuilt a layer wider would fail to load its own
    weights, and one rebuilt with a shorter history would load them and act on a
    window the run never saw.

    The settings that are *not* in the weights are taken from the caller and
    checked: the cadence the policy will be asked at (ADR 0009), the upgrade
    rows it will be offered (ADR 0011) and the Workshop runway profile it will
    play on (ADR 0012). None changes a tensor, so none
    would fail to load - a checkpoint trained on the image's six purchasable
    rows would quietly play a fully unlocked run, and the wave it scored would
    be read against floors measured under something else. They are required
    arguments rather than optional ones because a caller that forgot to say is
    exactly the silent case; this is the refusal the resume path already has
    (`checkpoint.load(expected=...)`), through the same `incompatibilities`.

    On the CPU by default. Evaluation is one forward pass per decision against
    an emulator that takes orders of magnitude longer to answer, so a GPU buys
    nothing and a fleet of evaluating actors would contend for one.

    The identity comes back beside the policy because a record of what this
    played has to name which checkpoint played it.
    """
    checkpoint = load(path)
    played = replace(
        checkpoint.identity,
        decision_cadence=str(decision_cadence),
        upgrade_availability=str(upgrade_availability),
        workshop_level=workshop_level,
    )
    # Every other field is copied from the checkpoint, so only these three can
    # differ: `incompatibilities` stays the one place that says what a
    # difference means, and its reasons name both values. The Workshop runway
    # profile is the third setting that changes no tensor (ADR 0012).
    refusals = checkpoint.identity.incompatibilities(played)
    if refusals:
        raise ValueError(f"{path} cannot be played here: {'; '.join(refusals)}")
    if checkpoint.identity.backbone == DREAMERV3:
        return _dreamer_policy(checkpoint, device, sampling_seed), checkpoint.identity

    settings = checkpoint.resolved_config
    if checkpoint.identity.backbone != "stacked-dqn":
        raise ValueError(
            f"{path} holds a {checkpoint.identity.backbone!r} backbone; "
            "only stacked-dqn can be rebuilt as a policy"
        )
    defaults = NetworkConfig()
    backbone = StackedDqnBackbone(
        config=StackedDqnConfig(
            history_length=int(settings["history_length"]),
            n_step=int(settings["n_step"]),
            # Absent from a checkpoint written before the n-step anneal, which
            # held n fixed - what these defaults rebuild.
            n_step_final=(
                None
                if settings.get("n_step_final") is None
                else int(settings["n_step_final"])
            ),
            n_step_anneal_steps=int(settings.get("n_step_anneal_steps", 0)),
            # Acting reads neither discount. A game-time run records the
            # per-decision one as None, and a run before it records no
            # per-game-second key at all; both rebuild on the defaults.
            discount=(
                StackedDqnConfig.discount
                if settings.get("discount") is None
                else float(settings["discount"])
            ),
            discount_per_game_second=settings.get("discount_per_game_second"),
            learning_rate=float(settings["learning_rate"]),
            target_ema_decay=float(settings["target_ema_decay"]),
        ),
        network_config=NetworkConfig(
            identity_capacity=int(
                settings.get("network_identity_capacity", defaults.identity_capacity)
            ),
            identity_dim=int(settings.get("network_identity_dim", defaults.identity_dim)),
            hidden=int(settings.get("network_hidden", defaults.hidden)),
            core_hidden=int(settings.get("network_core_hidden", defaults.core_hidden)),
        ),
        device=device or torch.device("cpu"),
    )
    backbone.load_state_dict(dict(checkpoint.backbone_state))
    # Nothing here is ever trained again. `act` already builds no graph; this
    # says so at the object as well, so a policy that leaked into a learner
    # would fail rather than quietly accumulate gradients.
    backbone.online.eval()
    backbone.online.requires_grad_(False)
    return backbone, checkpoint.identity


def _dreamer_policy(
    checkpoint: Checkpoint, device: torch.device | None, sampling_seed: str | None
) -> DreamerBackbone:
    """A DreamerV3 checkpoint as a policy: its config is the `dreamer_*` keys it recorded.

    DreamerV3 samples its policy, so every evaluating instance needs a stream of
    its own (`sampling_seed`, e.g. its serial). Left None, every instance would
    draw the same uniforms from the run's seed.
    """
    settings = checkpoint.resolved_config
    config = DreamerConfig(
        **{name.name: settings[f"dreamer_{name.name}"] for name in fields(DreamerConfig)}
    )
    backbone = DreamerBackbone(config=config, device=device or torch.device("cpu"))
    backbone.load_state_dict(dict(checkpoint.backbone_state))
    for value in vars(backbone).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
            value.requires_grad_(False)
    if sampling_seed is not None:
        backbone._random.seed(sampling_seed)
    return backbone


def describe(policy: Policy) -> str:
    """A stable name for reports, so arms stay identifiable across runs."""
    return type(policy).__name__


assert len(RUN_ACTIONS) > 1, "the action space must contain WAIT plus upgrades"
