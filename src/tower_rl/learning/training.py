"""The training loop that ties actor, replay, learner and checkpointing together.

Progress has one unit: decisions, cumulative across the fleet.  The budget, the
checkpoint cadence, the selection periods, the exploration anneal, the
importance-sampling beta and the kill bars are all positions on that one axis.
Learning happens per decision - the replay ratio is gradient steps per
decision - so a budget in decisions fixes the amount of learning, where a
budget in game time did not: the game time a policy spends per decision moves
with its strength (run 5 spent 6.98 game-seconds per decision against run 4's
4.42 over matched decisions, `M2-P005` diagnostic (c)) and with the game's
speed.  Game time and wall time are still measured and reported beside it, as
statistics.  The budget
is accounted at episode granularity - an episode is played to its classified
end - so a run stops after the episode that crossed it.

The decision axis is cut into selection periods (`selection_period_decisions`,
15,000 by default).  When a period closes the run writes a numbered checkpoint
and reads the mean final wave of the near-greedy actors' valid episodes that
ended inside it; the arm is chosen from those periods by the rule in
`docs/solution.md` 9.2b, applied by hand.  A curve that has not improved on
the level it last really moved to for `early_stop_patience_periods` periods in a row has
stopped learning, and the rest of the budget buys nothing, so the run ends
there.  Numbered checkpoints may also be written more often than periods close
(`checkpoint_every_decisions`), for a learning curve; that cadence selects
nothing.

A run may also be given kill bars: pre-registered floors at points on the
decision axis, matched against an earlier run.  When the fleet first reaches a
bar's decision count, the near-greedy actors' valid episodes that ended inside
its window must average at least its threshold, or the run stops there.

A run collects with one actor or with a fleet of them, and the budget is the
fleet's: N actors, each on its own emulator instance, collect concurrently into
one replay buffer and one learner, so the gradient steps a run takes track the
decisions the whole fleet collected.  Collection scales linearly to four
instances on this host (M1B-E028), and an actor spends nearly all of its time
waiting on a socket, so the actors are threads: they share the replay buffer
directly and nothing has to be serialised between processes.  What they do not
share is the network they act from.  Each actor holds its own copy of it and the
learner publishes into that copy between the actor's episodes, which is the
actor-learner arrangement of Ape-X and R2D2: a forward pass then contends with
nothing, where every actor reading the one live network would have put fifty
decisions a second and a dozen gradient steps a second through one lock.  What
that costs is the discipline in this file - the run's progress is mutated only
under `_lock`, the buffer only under the replay's own lock, and the learner's
parameters are read only through `Learner.publish_to`.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from tower_rl.environment.decision_time import (
    LEARNER_STEP,
    DecisionTimeBreakdown,
    DecisionTimeProfile,
)
from tower_rl.environment.episode import EpisodeSummary
from tower_rl.environment.run_environment import BRIDGE_EVENT_DIVERGENCE, GAME_TIME_INFLATED
from tower_rl.environment.run_port import RunPortError
from tower_rl.learning.actor import Actor, EpisodeResult
from tower_rl.learning.backbone import (
    Backbone,
    LearnMetrics,
    SequenceBatch,
    acting_copy,
    collate,
)
from tower_rl.learning.evaluator import EvaluationReport
from tower_rl.learning.exploration import ExplorationSchedule
from tower_rl.learning.replay import PrioritizedSequenceReplay

#: The device's own rejection reason for a stale or duplicate command, carried
#: into an episode's `termination_detail` free text exactly as it comes off the
#: bridge. `scripts/run_actors.py` names the identical literal
#: (`STALE_OR_DUPLICATE`) for its own report; it is not imported from there
#: because the learning package does not depend on a script.
STALE_OR_DUPLICATE = "stale_or_duplicate"


@dataclass(frozen=True)
class KillBar:
    """A floor the near-greedy curve must clear by a point on the decision axis.

    Pre-registered against an earlier run at matched fleet decisions: when the
    fleet's cumulative decisions first reach `at_decisions`, the near-greedy
    actors' valid episodes that ended in (`window_start_decisions`,
    `at_decisions`] must average at least `min_mean_final_wave`. A run that has
    fallen that far behind its comparator stops rather than spending the rest of
    its budget confirming it. Absolute rather than relative to the run's own
    best, which is what separates it from the plateau rule.
    """

    at_decisions: int
    window_start_decisions: int
    min_mean_final_wave: float

    def __post_init__(self) -> None:
        if not 0 <= self.window_start_decisions < self.at_decisions:
            raise ValueError("a kill bar's window must end after it starts, at or after zero")


@dataclass(frozen=True)
class KillBarCheck:
    """What one kill bar found when the run reached it."""

    bar: KillBar
    #: Where the run stood when it was checked: the first episode boundary at
    #: or past the bar, so never below `bar.at_decisions`.
    decisions: int
    near_greedy_episodes: int
    #: None when no near-greedy actor ended a valid episode in the window. That
    #: measures nothing, so it does not stop the run; it is recorded as that.
    mean_final_wave: float | None
    stopped: bool


def _mean(values: list[float]) -> float | None:
    """The mean of a health window, or None before it holds anything."""
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True)
class EpisodeHealth:
    """Environment-honesty counters pooled over a span of episodes.

    A 4.3-hour run finished with `advances_cut_short`, `episodes_not_started_fresh`
    and the round/budgeted ratio unrecoverable, because nothing persisted them
    per episode or aggregated them - the run could not be certified as honest for
    its whole duration. These counters are what that certification is read from,
    at whatever span they are pooled over: the whole run, one actor, or one
    collection window.
    """

    episodes: int
    valid_episodes: int
    invalid_episodes: int
    #: Invalid episodes by `TerminationOutcome`, coarse-grained.
    invalid_by_reason: dict[str, int]
    #: Invalid episodes by their free-text `termination_detail`, verbatim. A
    #: reason like "the game did not honour speed_down: lifecycle_timeout" is
    #: counted and named here rather than surviving only as text on one episode.
    invalid_detail: dict[str, int]
    bridge_event_divergence: int
    stale_or_duplicate: int
    game_time_inflated: int
    advances_cut_short: int
    #: Speed-pin failures the port recovered from at an episode boundary,
    #: pooled over the span. Recovered, so no episode is lost to one - which is
    #: exactly why it has to be pooled somewhere a long run is read from (`#57`).
    pin_restarts: int
    episodes_not_started_fresh: int
    #: The game's round clock over the budgeted game time, pooled across every
    #: episode that spent measurable game time. None until one has.
    round_budgeted_ratio: float | None
    #: The single most inflated per-episode ratio in the span. Only inflation is
    #: judged (see `GAME_TIME_INFLATED`'s own note), so the worst is the maximum
    #: observed, not the extreme in either direction.
    worst_round_budgeted_ratio: float | None


def episode_health(summaries: Sequence[EpisodeSummary]) -> EpisodeHealth:
    """Pool the honesty counters over a span of episodes, valid and invalid alike.

    One definition shared by run, actor and collection-window reporting, so a
    health problem can be found at any of those scopes without a second way of
    counting it.
    """
    valid = sum(1 for summary in summaries if summary.valid)
    by_reason: dict[str, int] = {}
    detail: dict[str, int] = {}
    for summary in summaries:
        if summary.valid:
            continue
        by_reason[summary.termination.value] = by_reason.get(summary.termination.value, 0) + 1
        for text in summary.termination_detail or (summary.termination.value,):
            detail[text] = detail.get(text, 0) + 1
    all_detail = [text for summary in summaries for text in summary.termination_detail]
    measured = [
        (summary.round_ms, summary.game_ms) for summary in summaries if summary.game_ms > 0
    ]
    pooled_ratio = (
        sum(round_ms for round_ms, _ in measured) / sum(game_ms for _, game_ms in measured)
        if measured
        else None
    )
    worst_ratio = max((round_ms / game_ms for round_ms, game_ms in measured), default=None)
    return EpisodeHealth(
        episodes=len(summaries),
        valid_episodes=valid,
        invalid_episodes=len(summaries) - valid,
        invalid_by_reason=by_reason,
        invalid_detail=detail,
        bridge_event_divergence=sum(1 for text in all_detail if BRIDGE_EVENT_DIVERGENCE in text),
        stale_or_duplicate=sum(1 for text in all_detail if STALE_OR_DUPLICATE in text),
        game_time_inflated=sum(1 for text in all_detail if GAME_TIME_INFLATED in text),
        advances_cut_short=sum(summary.advances_cut_short for summary in summaries),
        pin_restarts=sum(summary.pin_restarts for summary in summaries),
        episodes_not_started_fresh=sum(1 for summary in summaries if summary.starting_wave > 1),
        round_budgeted_ratio=pooled_ratio,
        worst_round_budgeted_ratio=worst_ratio,
    )


@dataclass
class Learner:
    """The one training copy of the network, and how its parameters reach actors.

    No actor acts through this. Each acts from its own copy (`acting_copy`), so
    a forward pass contends with neither the learner nor another actor - the
    whole reason a fleet of twenty can act at all. What is left shared is the
    moment a copy is refreshed, and that is what the lock is still for: an
    optimisation step and a publication never overlap, so what an actor copies
    out is always the parameters of some completed step and never half of one.
    """

    backbone: Backbone
    lock: threading.Lock = field(default_factory=threading.Lock)

    def learn(self, batch: SequenceBatch) -> LearnMetrics:
        with self.lock:
            return self.backbone.learn(batch)

    def publish_to(self, acting: Backbone) -> None:
        """Copy the learner's parameters into one actor's acting copy.

        Called on that actor's own thread between its episodes, which is the
        other half of the no-torn-read guarantee: the lock keeps the source
        still while it is read, and an actor that is copying is by construction
        not acting, so no forward pass can see the copy half written.
        """
        with self.lock:
            acting.load_state_dict(self.backbone.state_dict())


@dataclass(frozen=True)
class TrainingConfig:
    """One arm's budget and schedules, recorded with its result."""

    #: The budget: cumulative decisions across the whole fleet. Counted at
    #: episode granularity, because an episode is played to its end: the run
    #: stops after the episode that crosses it.
    budget_decisions: int
    #: What each actor explores at, at each point of the budget: the anneal and,
    #: under a ladder, the rung each actor anneals to. Required, like the budget:
    #: the rates a run explores at are resolved from the command line, and a
    #: default here would be a second source of them.
    exploration: ExplorationSchedule
    #: Sequences required before the first optimisation step. About 35 episodes
    #: at this geometry: enough that the first gradient steps see more than a
    #: handful of episodes of one policy.
    warmup_sequences: int = 100
    batch_size: int = 8
    #: Gradient steps per environment decision: the replay ratio. What matters
    #: is the transitions replayed per transition generated, which is this times
    #: the learnable steps in a batch - at 80-step sequences, burn-in 7, n-step
    #: 10 and batch 8 that is about 504 per step, so 0.25 puts the run at 126:1,
    #: above SPR's 64:1. The 2.0 of the first run was 1087:1.
    gradient_steps_per_decision: float = 0.25
    #: Episodes per point of the collection curve. The curve is read from the
    #: collection episodes themselves rather than from exploration-free
    #: evaluations: at epsilon 0.05 they are almost on-policy, they cost no
    #: extra device time, and 100 episodes put the standard error near 0.2
    #: waves where a 5-episode evaluation point sits near 0.9.
    collection_window_episodes: int = 100
    #: Importance-sampling correction anneals the other way, as is conventional.
    beta_start: float = 0.4
    beta_end: float = 1.0
    #: Zero disables the periodic hook entirely; a positive value is a period in
    #: episodes. Evaluation is exploration-free and never writes to replay.
    evaluate_every_episodes: int = 0
    checkpoint_every_episodes: int = 0
    #: Decisions between extra numbered checkpoints, for a learning curve. Zero
    #: writes numbered checkpoints only where a selection period closes. This
    #: cadence chooses nothing: the arm is always a period's checkpoint.
    checkpoint_every_decisions: int = 0
    #: Decisions per selection period. A period's near-greedy mean is what the
    #: arm is chosen on and what early stopping counts; a numbered checkpoint
    #: is written wherever one closes. 15,000 is run 4's average period
    #: (60,356 decisions / 4 periods), not a measured length.
    selection_period_decisions: int = 15_000
    #: How many selection periods in a row may close without the near-greedy
    #: curve improving before the run stops itself, after the checkpoint of the
    #: period that closed the last of them. Zero is off.
    early_stop_patience_periods: int = 0
    #: What counts as an improvement, in waves. A period whose near-greedy mean
    #: does not reach the best period mean so far plus this much has not
    #: improved on it. 0.2 waves is about the standard error of a hundred
    #: episode window, so a period inside it is noise rather than progress.
    early_stop_min_improvement: float = 0.2
    #: How many episodes in a row may fail at the port before an actor gives up.
    #: A single failed episode is an ordinary event on a real device and must not
    #: end a run that has hours of experience in it; a device that fails every
    #: episode is a broken instance, and continuing would spin without collecting
    #: anything. The limit is what separates the two, and it is counted per
    #: actor: one dead emulator out of four is one withdrawn actor, not a dead
    #: environment, and the run ends only when every actor has withdrawn.
    max_consecutive_episode_failures: int = 5
    #: Episodes one actor plays between refreshes of the copy it acts from, its
    #: parameter lag. One means every actor starts each episode from the
    #: learner's current parameters, which is exactly what a single actor did
    #: when it acted from the learner's network directly - the reason it is the
    #: default is that it leaves `--actors 1` unchanged against the runs already
    #: measured. It is also well inside published practice: Ape-X and R2D2
    #: actors refresh every few hundred environment steps, and an episode here
    #: is about 121 decisions. Raising it trades freshness for fewer
    #: publications; the lag it buys is bounded by this many of the actor's own
    #: episodes, never by the fleet's rate.
    parameter_sync_episodes: int = 1
    #: Pre-registered floors on the decision axis the run stops itself on; see
    #: `KillBar`. Empty is off, which is every run before run 4.
    kill_bars: tuple[KillBar, ...] = ()

    def __post_init__(self) -> None:
        if self.budget_decisions < 1:
            raise ValueError("budget must be positive")
        if self.batch_size < 1 or self.warmup_sequences < 1:
            raise ValueError("batch size and warm-up must be positive")
        if self.gradient_steps_per_decision <= 0:
            raise ValueError("gradient steps per decision must be positive")
        if self.collection_window_episodes < 1:
            raise ValueError("a collection window needs at least one episode")
        if self.max_consecutive_episode_failures < 1:
            raise ValueError("at least one episode failure must be survivable")
        if self.parameter_sync_episodes < 1:
            raise ValueError("actors must be synchronised at least every episode")
        if self.checkpoint_every_decisions < 0:
            raise ValueError("a checkpoint cadence cannot be negative")
        if self.selection_period_decisions < 1:
            raise ValueError("a selection period must be positive")
        if self.early_stop_patience_periods < 0:
            raise ValueError("early-stopping patience cannot be negative")

    def beta(self, decisions: int) -> float:
        """The importance exponent at this point of the budget: `beta_end` at its end."""
        fraction = min(1.0, decisions / self.budget_decisions)
        return self.beta_start + (self.beta_end - self.beta_start) * fraction


@dataclass(frozen=True)
class CollectedEpisode:
    """One episode of collection, with what the policy did in it.

    The summary says what the game did; `wait_decisions` says what the policy
    asked for. Both are needed to tell a policy that is learning slowly from one
    that has collapsed onto `WAIT`, which the final wave alone cannot.
    """

    summary: EpisodeSummary
    wait_decisions: int
    #: Which actor played it. The episodes of a fleet are one series in
    #: completion order, and this is how a per-actor account is taken of it.
    actor_id: str = "actor-0"

    @property
    def wait_fraction(self) -> float:
        decisions = self.summary.decisions
        return 0.0 if decisions == 0 else self.wait_decisions / decisions


@dataclass(frozen=True)
class ActionDistribution:
    """What a policy spent its decisions on over a span of episodes.

    The random baseline buys 18.7 upgrades per episode; a policy that has
    collapsed waits out more than nine decisions in ten and buys nothing.
    """

    episodes: int
    decisions: int
    wait_fraction: float
    purchases_per_episode: float


@dataclass(frozen=True)
class CollectionWindow:
    """One point of the collection curve: a block of episodes as they were played.

    Read in preference to exploration-free evaluation points. These episodes are
    collected at the held epsilon anyway, so the window costs no device time, and
    a hundred of them put the standard error of the mean near 0.2 waves - small
    enough that a real improvement of half a wave is visible, where the five
    episode evaluation points of the first run could not resolve less than about
    three waves.
    """

    #: Zero-based ordinal of the window within the run.
    index: int
    #: Valid episodes in the window; always the configured size.
    episodes: int
    #: Decisions spent inside the window, and the budget position at its end.
    decisions: int
    decisions_at_end: int
    mean_final_wave: float
    #: None only for a one-episode window, which has no spread.
    stdev_final_wave: float | None
    standard_error: float | None
    wait_fraction: float
    purchases_per_episode: float
    #: The environment-honesty counters over every episode attempted while this
    #: window was filling, valid and invalid alike - not only the valid ones the
    #: wave statistics above are over. This is what locates a health problem in
    #: time rather than only in the whole run's total.
    health: EpisodeHealth
    #: The same window per actor, by actor id: an actor with no valid episode in
    #: it is absent rather than zero. Under an exploration ladder the pooled mean
    #: above is an average over actors exploring at rates two orders of magnitude
    #: apart, which is not any policy's performance; these are.
    mean_final_wave_by_actor: dict[str, float] = field(default_factory=dict)
    #: The window over the near-greedy actors only - the series a readout that
    #: asks what the policy itself reaches has to cite. Equal to the pooled
    #: numbers under a uniform schedule, where every actor is near-greedy.
    near_greedy_episodes: int = 0
    near_greedy_mean_final_wave: float | None = None


@dataclass(frozen=True)
class SelectionPeriod:
    """One selection period, as it closed.

    The collection window beside this is cut in episodes and is the curve the
    run is read from; a period is cut in decisions, and a numbered checkpoint is
    written where it closes, so its mean belongs to that checkpoint - which is
    what choosing the arm and deciding whether the run still improves compare.
    """

    #: The period's ordinal over the whole run, counting from one.
    index: int
    #: Where the run stood on the decision axis when the period closed, which
    #: is also the number the period's checkpoint is named by.
    decisions_at_end: int
    #: Valid episodes the near-greedy actors ended inside the period, and their
    #: mean final wave. None when no near-greedy actor finished a valid episode
    #: in it, which measures nothing rather than measuring zero.
    near_greedy_episodes: int
    mean_final_wave: float | None
    #: The level the next period is judged against: the mean of the last
    #: period that improved on it by the threshold, which is not the highest
    #: mean the run has seen - see `NearGreedyPlateau.close_period`.
    best_mean_final_wave: float | None


@dataclass
class NearGreedyPlateau:
    """Whether the near-greedy curve has stopped improving, period by period.

    The run's own read of its learning curve, and the only state behind its
    decision to stop early. It is counted over the whole run rather than over
    one segment of it: the counters travel in the checkpoint, so a run trained
    in two sittings is judged on one curve. What a period's mean is over - the
    near-greedy actors' valid episodes, which under a uniform schedule is every
    actor's - belongs to `TrainingRun`; this only remembers what the periods
    have been worth.
    """

    #: Periods closed so far, over the whole run.
    periods_closed: int = 0
    #: The level every later period is judged against: the mean of the last
    #: period that cleared it by `min_improvement`, which is deliberately not
    #: the highest mean the run has seen. None until a period has produced one.
    best_mean_final_wave: float | None = None
    #: Periods in a row that failed to improve on it.
    periods_without_improvement: int = 0
    #: The period the run stopped itself at, or None while it is still running.
    stopped_at_period: int | None = None
    #: Whether these counters came back from a parent checkpoint. False for a
    #: fresh run and for a resume from a checkpoint written before early
    #: stopping existed, whose tracker starts over - which the report says
    #: rather than leaving a fresh baseline to look like a continued one.
    restored: bool = False

    def close_period(self, mean: float | None, *, min_improvement: float) -> None:
        """Record what the period just closed was worth.

        The first period with a mean sets the baseline and cannot count against
        the run: there has to be something to fail to improve on before a run
        can be said to have stopped improving. A period no near-greedy actor
        finished a valid episode in is counted neither way - it measures
        nothing - though it is still a period that closed.

        The baseline moves only on an improvement that counted, never on a mere
        new maximum. A curve creeping up by less than `min_improvement` a
        period would otherwise raise the bar it is judged against by exactly
        what it gained, and a run gaining a tenth of a wave an hour would stop
        while one gaining nothing at all carried on - the opposite of what the
        threshold is for. Held here, the creep is judged against where the
        curve last really moved and keeps the run alive as long as it clears
        the threshold now and then.
        """
        self.periods_closed += 1
        if mean is None:
            return
        best = self.best_mean_final_wave
        if best is None or mean >= best + min_improvement:
            self.periods_without_improvement = 0
            self.best_mean_final_wave = mean
        else:
            self.periods_without_improvement += 1

    def plateaued(self, patience_periods: int) -> bool:
        """Whether the curve has failed to improve for `patience_periods` in a row."""
        return bool(patience_periods) and self.periods_without_improvement >= patience_periods


def action_distribution(episodes: Sequence[CollectedEpisode]) -> ActionDistribution | None:
    """What the policy did over these episodes, or None if there are none."""
    if not episodes:
        return None
    decisions = sum(episode.summary.decisions for episode in episodes)
    waits = sum(episode.wait_decisions for episode in episodes)
    purchases = sum(episode.summary.purchases for episode in episodes)
    return ActionDistribution(
        episodes=len(episodes),
        decisions=decisions,
        wait_fraction=0.0 if decisions == 0 else waits / decisions,
        purchases_per_episode=purchases / len(episodes),
    )


def _mean_final_wave_by_actor(
    episodes: Sequence[CollectedEpisode],
) -> dict[str, float]:
    """Mean final wave per actor over these episodes, in first-seen order."""
    waves: dict[str, list[int]] = {}
    for episode in episodes:
        waves.setdefault(episode.actor_id, []).append(episode.summary.final_wave)
    return {actor_id: statistics.fmean(values) for actor_id, values in waves.items()}


def collection_windows(
    collected: Sequence[CollectedEpisode],
    *,
    size: int,
    spent_before: int = 0,
    near_greedy_actor_ids: Collection[str] | None = None,
) -> list[CollectionWindow]:
    """Cut the collection episodes into consecutive non-overlapping windows.

    Windows are counted in valid episodes, because an invalid episode has no
    final wave to average; the decisions of every episode still count towards
    the budget position a window is placed at, since they were all spent. A
    trailing partial window is not emitted at all: a point averaged over fewer
    episodes than the rest has a different standard error and would be read as
    if it did not.

    `near_greedy_actor_ids` names the actors whose episodes are read as the
    policy's own performance rather than as search; None means every actor is,
    which is what a uniform schedule's fleet is.

    `spent_before` is the budget position these episodes start from, which is
    not zero for a run resumed from a checkpoint: the episode list is this
    segment's, but `decisions_at_end` is a position on the whole run's budget,
    and a window keyed segment-relatively would land on the same axis as the
    parent's points and beneath them.
    """
    if size < 1:
        raise ValueError("a collection window needs at least one episode")
    windows: list[CollectionWindow] = []
    current: list[CollectedEpisode] = []
    #: Every episode attempted while this window was filling, valid and invalid
    #: alike - the span `health` is pooled over, wider than `current`.
    attempted: list[CollectedEpisode] = []
    spent = spent_before
    window_decisions = 0
    for episode in collected:
        spent += episode.summary.decisions
        window_decisions += episode.summary.decisions
        attempted.append(episode)
        if not episode.summary.valid:
            continue
        current.append(episode)
        if len(current) < size:
            continue
        waves = [item.summary.final_wave for item in current]
        near_greedy = [
            item
            for item in current
            if near_greedy_actor_ids is None or item.actor_id in near_greedy_actor_ids
        ]
        distribution = action_distribution(current)
        assert distribution is not None  # a full window is never empty
        stdev = statistics.stdev(waves) if len(waves) > 1 else None
        windows.append(
            CollectionWindow(
                index=len(windows),
                episodes=len(current),
                decisions=window_decisions,
                decisions_at_end=spent,
                mean_final_wave=statistics.fmean(waves),
                stdev_final_wave=stdev,
                standard_error=None if stdev is None else stdev / len(waves) ** 0.5,
                wait_fraction=distribution.wait_fraction,
                purchases_per_episode=distribution.purchases_per_episode,
                mean_final_wave_by_actor=_mean_final_wave_by_actor(current),
                near_greedy_episodes=len(near_greedy),
                near_greedy_mean_final_wave=(
                    statistics.fmean(item.summary.final_wave for item in near_greedy)
                    if near_greedy
                    else None
                ),
                health=episode_health([item.summary for item in attempted]),
            )
        )
        current = []
        attempted = []
        window_decisions = 0
    return windows


@dataclass
class ActorProgress:
    """What one actor of the fleet contributed, and whether it is still alive.

    The aggregate counts on `TrainingProgressReport` say what the run collected;
    these say who collected it. Four actors averaging the same episode rate and
    one actor that stopped an hour ago look identical in the aggregate, which is
    exactly the failure a fleet has to be able to report.
    """

    actor_id: str
    #: Episodes attempted, including the ones the port could not deliver.
    episodes: int = 0
    decisions: int = 0
    #: Game time this actor collected, measured: the game's own round clock
    #: summed over the episodes it delivered. A statistic, not the budget.
    game_ms: float = 0.0
    valid_episodes: int = 0
    invalid_episodes: int = 0
    failed_episodes: int = 0
    failures: list[str] = field(default_factory=list)
    #: Failures since this actor's last delivered episode, which is what the
    #: per-actor limit is measured against.
    consecutive_failures: int = 0
    #: Where this actor's wall time went, cumulative over the run and published
    #: by the actor itself at its own episode boundaries. None until it has
    #: finished an episode. See `environment/decision_time.py`: this is what
    #: separates an actor idle on its emulator from one contending in Python.
    decision_time: DecisionTimeBreakdown | None = None
    #: Why this actor stopped collecting, or None while it is still collecting.
    #: A withdrawn actor is never restarted: its instance failed every episode
    #: the limit allows, and the fleet carries on without it.
    withdrawn: str | None = None


@dataclass
class TrainingProgressReport:
    """What a run produced, enough to compare arms and to resume."""

    #: The budget position: decisions across every actor.
    decisions: int = 0
    #: Measured game time across every actor, summed over the episodes they
    #: delivered. Reported beside the decisions; the run is not spent in it.
    game_ms: float = 0.0
    episodes: int = 0
    optimisation_steps: int = 0
    sequences_accepted: int = 0
    #: `EpisodeResult`'s end-of-span counters, summed over this segment.
    reward_bearing_transitions: int = 0
    wave_change_ended_span: int = 0
    wall_seconds: float = 0.0
    #: Every episode collected, in the order it was played. The collection curve
    #: is read from this; evaluation is the headline, not the curve.
    collected: list[CollectedEpisode] = field(default_factory=list)
    #: The optimised quantity, which carries the importance-sampling weights in
    #: it and therefore moves with the beta schedule whether or not the learner
    #: improves. Named for that, and never reported without the unweighted TD
    #: error beside it.
    recent_weighted_losses: list[float] = field(default_factory=list)
    #: The signal to read instead: absolute TD error with no weighting at all.
    recent_unweighted_td_errors: list[float] = field(default_factory=list)
    recent_gradient_norms: list[float] = field(default_factory=list)
    #: Correlation between predicted value and realised return, per step that
    #: could compute one. The strongest evidence that the learner works at all.
    recent_value_fits: list[float] = field(default_factory=list)
    evaluations: list[EvaluationReport] = field(default_factory=list)
    checkpoints_written: int = 0
    #: Where the exploration schedule had reached and the importance-sampling
    #: exponent the run last sampled at. Published here by the run that draws
    #: them from its schedules, so a checkpoint or a report carries the value
    #: the run actually used rather than re-evaluating a schedule of its own.
    #: Under a ladder the actors are at rates of their own and this is
    #: `ExplorationSchedule.reported_epsilon` - informational, and never what a
    #: per-episode or per-actor measurement should be read from.
    epsilon: float = 0.0
    importance_beta: float = 0.0
    #: Episodes the port could not produce at all - a boundary that would not
    #: settle, an instance that would not start. They are counted in `episodes`
    #: like any other attempt, but they leave no summary behind, so the two
    #: counts differ exactly by this one.
    failed_episodes: int = 0
    episode_failures: list[str] = field(default_factory=list)
    #: Periodic evaluations that could not be scored. Training continues: an
    #: evaluation is measurement, and losing a measurement must not lose the run.
    evaluation_failures: list[str] = field(default_factory=list)
    #: The fleet, keyed by actor id in the order the actors were started. A run
    #: with one actor holds exactly one entry.
    actors: dict[str, ActorProgress] = field(default_factory=dict)
    #: Every selection period this segment closed, in order. The episodes
    #: behind them are this segment's - the collected list is not restored on a
    #: resume - while the plateau below is counted over the whole run.
    selection_periods: list[SelectionPeriod] = field(default_factory=list)
    #: The run's read of its own near-greedy curve, and the state a resume
    #: restores so a run trained in two sittings is judged on one curve.
    plateau: NearGreedyPlateau = field(default_factory=NearGreedyPlateau)
    #: Every kill bar this segment reached, in the order it reached them.
    kill_bar_checks: list[KillBarCheck] = field(default_factory=list)

    @property
    def game_seconds(self) -> float:
        return self.game_ms / 1000.0

    @property
    def episode_summaries(self) -> list[EpisodeSummary]:
        """The collected episodes as the environment classified them."""
        return [episode.summary for episode in self.collected]

    @property
    def valid_episodes(self) -> int:
        return sum(1 for episode in self.collected if episode.summary.valid)

    @property
    def final_waves(self) -> list[int]:
        return [
            episode.summary.final_wave for episode in self.collected if episode.summary.valid
        ]

    @property
    def mean_recent_weighted_loss(self) -> float | None:
        """Mean of the optimised loss over the last hundred steps, or None before any.

        Reported beside the outcome because section 9.7 asks for it: a loss that
        stops moving while episodes keep arriving is a learner problem, and it is
        invisible in the final-wave distribution alone. It is weighted, so it
        falls as beta anneals even when nothing is learned; that is what made the
        first run's apparent progress an artefact, and why it is never reported
        without `mean_recent_unweighted_absolute_td_error`.
        """
        return _mean(self.recent_weighted_losses)

    @property
    def mean_recent_unweighted_absolute_td_error(self) -> float | None:
        """Mean absolute TD error over the same window, with no weighting in it."""
        return _mean(self.recent_unweighted_td_errors)

    @property
    def mean_recent_gradient_norm(self) -> float | None:
        """Mean gradient norm over the same window, or None before any step."""
        return _mean(self.recent_gradient_norms)

    @property
    def mean_recent_value_fit_correlation(self) -> float | None:
        """Mean value-fit correlation over the same window, or None before any."""
        return _mean(self.recent_value_fits)

    def episodes_of(self, actor_id: str) -> list[CollectedEpisode]:
        """The episodes one actor collected, in the order it completed them."""
        return [episode for episode in self.collected if episode.actor_id == actor_id]


@contextmanager
def measured_apart(profile: DecisionTimeProfile) -> Iterator[None]:
    """Run something on an actor's thread without charging it to that actor.

    The hooks an episode owes - periodic evaluation, checkpointing, reporting -
    run on whichever actor's thread finished the episode, but they are
    measurement rather than collection. The collecting block is closed around
    them and reopened after, so the total the buckets decompose is collection
    alone and the residual keeps meaning "Python time this actor could not
    account for".

    Only a block that was open is reopened. Called outside one - which nothing
    does today - reopening unconditionally would leave a block open that nobody
    closes, and every later snapshot would charge it the wall time since.
    """
    collecting = profile.block_open
    profile.close_block()
    try:
        yield
    finally:
        if collecting:
            profile.open_block()


@dataclass
class TrainingRun:
    """One arm: a fleet of actors, one replay buffer, one backbone, one budget.

    The fleet is `actors`, one actor per emulator instance. Whatever their
    number there is one of everything else: one buffer they all write into, one
    network they all act from, and one budget counted across all of them, so the
    replay ratio the run is configured with is the ratio the fleet trains at.
    """

    actors: list[Actor]
    replay: PrioritizedSequenceReplay
    backbone: Backbone
    config: TrainingConfig
    on_episode: Callable[[TrainingProgressReport], None] | None = None
    #: Called once for the actor that has just left the fleet, with the progress
    #: naming it and the failure that withdrew it. A withdrawal is silent in the
    #: aggregate - the fleet simply collects a little slower - so an unattended
    #: run has to be told about it when it happens, not only in the summary.
    on_withdrawal: Callable[[ActorProgress], None] | None = None
    #: Runs exploration-free episodes on the same device. Its cost comes out of
    #: wall-clock time, never out of the budget, because evaluation is
    #: measurement rather than experience. It borrows an instance, so it may only
    #: run while the fleet is not collecting.
    evaluate: Callable[[], EvaluationReport] | None = None
    checkpoint: Callable[[TrainingProgressReport], None] | None = None
    #: Called when a selection period closes or the fleet crosses a multiple of
    #: `checkpoint_every_decisions`, to write a checkpoint under its own name.
    #: Separate from `checkpoint` above, which keeps the one resume point: a
    #: numbered checkpoint is a candidate arm, so it must survive the next one
    #: being written.
    numbered_checkpoint: Callable[[TrainingProgressReport], None] | None = None
    #: Progress so far. Instance state rather than a local because a run can be
    #: advanced in blocks, and a resumed run is constructed with its parent's.
    report: TrainingProgressReport = field(default_factory=TrainingProgressReport)
    #: The training copy of the network, and the only thing that updates it.
    learner: Learner = field(init=False)
    #: One acting copy per actor, keyed by actor id: what that actor actually
    #: chooses its actions from, refreshed from the learner on the configured
    #: cadence. Built here and handed to the actors so that no caller can put an
    #: actor back on the learner's own network.
    acting: dict[str, Backbone] = field(init=False)
    #: Guards everything the fleet shares except the buffer and the network: the
    #: progress report, the gradient debt and the hooks. An actor holds it
    #: between episodes and never while it is collecting, so at the cadence a
    #: real instance runs at it costs nothing.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    #: Gradient steps earned but not yet taken, carried across blocks.
    _owed: float = field(default=0.0, init=False)
    #: The decisions the periodic hook last looked at. Episodes end whole, so
    #: the counter jumps past a multiple of a period rather than landing on it;
    #: a multiple lying between this and the current count is a crossing, and
    #: each crossing is answered exactly once.
    _periodic_at: int = field(default=0, init=False)
    #: Where in `report.collected` the period now open began. A period's mean is
    #: over the episodes that ended inside it, and this is the cursor that
    #: separates them from the ones the previous period was already judged on.
    _period_start: int = field(default=0, init=False)
    #: Where each actor stands in the fleet, by id. The order actors were given
    #: in is the order the exploration ladder is read in and the order a
    #: per-actor metric series is keyed by, so it is resolved once here rather
    #: than re-derived by everything that reports per actor.
    actor_index: dict[str, int] = field(default_factory=dict, init=False)
    #: Episodes each actor has played since its copy was last refreshed. Starts
    #: at the cadence so every actor publishes before its first episode, which
    #: is also what picks up a checkpoint loaded into the backbone after the run
    #: was built. Each actor touches only its own entry of a dict whose keys are
    #: all present from construction, so it needs no lock of its own.
    _since_sync: dict[str, int] = field(default_factory=dict, init=False)
    #: Where on the decision axis this segment's `report.collected` starts: zero
    #: for a fresh run, the parent's count for a resumed one. A kill bar places
    #: each episode on the whole run's axis from here.
    _segment_start_decisions: int = field(default=0, init=False)
    #: Kill bars still to be reached, by index into `config.kill_bars`.
    _bars_pending: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not self.actors:
            raise ValueError("a run needs at least one actor")
        self._segment_start_decisions = self.report.decisions
        # A bar the parent already passed was answered by the parent: it would
        # have stopped there had it failed.
        self._bars_pending = [
            index
            for index, bar in enumerate(self.config.kill_bars)
            if bar.at_decisions > self.report.decisions
        ]
        identities = [actor.config.actor_id for actor in self.actors]
        if len(set(identities)) != len(identities):
            # Per-actor reporting is keyed by identity; two actors under one
            # name would report as one instance and hide a dead one.
            raise ValueError("every actor of a fleet needs an id of its own")
        self.actor_index = {
            actor.config.actor_id: index for index, actor in enumerate(self.actors)
        }
        floors = self.config.exploration.floors
        if floors and len(floors) != len(self.actors):
            # A ladder is built for a fleet of a particular size: read against a
            # different one, actor 3 of 7 would act at actor 3 of 4's rate, and
            # the run's own record of what it explored at would be wrong.
            raise ValueError(
                f"the exploration ladder has {len(floors)} rates for "
                f"{len(self.actors)} actors"
            )
        self.learner = Learner(self.backbone)
        # The cadences are continued rather than restarted: a run resumed at
        # 50,123 decisions has already answered the multiple of 15,000 at
        # 45,000, and the next period it owes closes at 60,000.
        self._periodic_at = self.report.decisions
        self.report.epsilon = self.config.exploration.reported_epsilon(
            self.report.decisions
        )
        self.report.importance_beta = self.config.beta(self.report.decisions)
        self.acting = {}
        for index, actor in enumerate(self.actors):
            actor_id = actor.config.actor_id
            # The first actor carries on the learner's own exploration stream,
            # so a fleet of one draws the epsilon sequence it has always drawn;
            # the rest are seeded from their identities, or every copy would
            # explore in lockstep from the one stream they were copied from.
            copy = acting_copy(
                self.backbone, exploration_seed=None if index == 0 else actor_id
            )
            self.acting[actor_id] = copy
            actor.policy = copy
            # One time-accounting profile per actor, shared with the instance it
            # drives so a decision's environment time and its policy time land
            # in the same buckets. Wired here like the acting copy above, for
            # the same reason: the run owns what an actor is attached to.
            actor.environment.profile = actor.profile
            self._since_sync[actor_id] = self.config.parameter_sync_episodes
            self.report.actors.setdefault(actor_id, ActorProgress(actor_id))

    @property
    def near_greedy_actor_ids(self) -> frozenset[str]:
        """The actors whose episodes read as the policy's performance, not search.

        Every actor of a uniform schedule, which draws the one annealed rate;
        under a ladder, the ones at or under `NEAR_GREEDY_EPSILON`. Resolved
        here because the fleet's order is the run's: the ladder is read in it,
        and anything asking which actors are near-greedy is asking about this
        fleet.
        """
        exploration = self.config.exploration
        return frozenset(
            actor_id
            for actor_id, index in self.actor_index.items()
            if exploration.is_near_greedy(index)
        )

    @property
    def killed_by(self) -> KillBarCheck | None:
        """The kill bar the run stopped itself on, or None."""
        return next((check for check in self.report.kill_bar_checks if check.stopped), None)

    @property
    def stopped_early(self) -> bool:
        """Whether the run stopped itself: on a plateau, or below a kill bar."""
        return self.report.plateau.stopped_at_period is not None or self.killed_by is not None

    @property
    def finished(self) -> bool:
        """Whether nothing is left to collect: the budget is spent, or it stopped itself."""
        return self.report.decisions >= self.config.budget_decisions or self.stopped_early

    def run(self) -> TrainingProgressReport:
        """Collect and learn until the decision budget is spent."""
        return self.advance(self.config.budget_decisions)

    def advance(self, decisions: int) -> TrainingProgressReport:
        """Collect with every live actor until `decisions` more are spent.

        The limit is the fleet's and lands on an episode boundary for each actor:
        an episode in progress is played to its classified end, because a half
        episode is not experience - so the fleet overshoots the limit by at most
        one episode per collecting actor.

        One actor per thread. An actor is waiting on its emulator's socket for
        almost all of its life and torch releases the interpreter lock around the
        work that is not waiting, so threads collect from four instances at once
        while sharing the buffer and the network directly - which is the whole
        reason this needs no parameter server and no queues of tensors.
        """
        if decisions < 1:
            raise ValueError("a block must be at least one decision")
        report = self.report
        target = min(report.decisions + decisions, self.config.budget_decisions)
        collecting = [
            actor
            for actor in self.actors
            if report.actors[actor.config.actor_id].withdrawn is None
        ]
        if not collecting:
            raise RunPortError("every actor has withdrawn; nothing is left collecting")
        self._refuse_evaluation_during_collection(len(collecting))

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(collecting)) as pool:
            futures = [pool.submit(self._collect, actor, target) for actor in collecting]
            # Leaving the pool joins every actor, so the block ends with no
            # episode in flight and nothing still writing to the report.
        report.wall_seconds = round(report.wall_seconds + time.monotonic() - started, 2)

        withdrawals: list[RunPortError] = []
        for future in futures:
            error = future.exception()
            if error is None:
                continue
            if not isinstance(error, RunPortError):
                raise error
            withdrawals.append(error)
        if withdrawals and all(
            progress.withdrawn is not None for progress in report.actors.values()
        ):
            # One dead instance costs the run an actor and is reported as that.
            # A fleet with nothing left collecting is a dead environment, which
            # is what the failure limit exists to end the run on.
            raise withdrawals[-1]
        return report

    def _refuse_evaluation_during_collection(self, collecting: int) -> None:
        """Refuse a periodic evaluation that would share an instance with an actor.

        Evaluation borrows an actor's environment and plays whole episodes on
        it. With one actor that is safe: the hook runs on that actor's own
        thread, between its episodes. With a fleet, every other actor is still
        collecting, and the borrowed environment would be driven from two
        threads at once - which the port cannot detect and which corrupts both
        the evaluation and the episode it interleaved with. Checked here, before
        a single thread is started, rather than trusted to whatever composed the
        run.
        """
        if collecting < 2 or self.evaluate is None:
            return
        if not self.config.evaluate_every_episodes:
            return
        raise ValueError(
            "periodic evaluation borrows an instance and cannot run while a fleet "
            f"of {collecting} actors is collecting; leave evaluate_every_episodes "
            "at 0 for a fleet, or evaluate after the budget is spent"
        )

    def _collect(self, actor: Actor, target: int) -> None:
        """One actor's thread: collect episodes, and learn from what it collected.

        The learning an episode earns is taken here, on the collecting thread,
        under `_lock`, so exactly one episode is being accounted for at a time
        while the other actors carry on playing. That is also what makes a fleet
        of one identical to the loop this file had before there were fleets: the
        single actor takes its own gradient steps between its own episodes, in
        the same order, against a lock nothing else ever holds.
        """
        actor_id = actor.config.actor_id
        progress = self.report.actors[actor_id]
        acting = self.acting[actor_id]
        profile = actor.profile
        with profile.collecting():
            self._collect_until(actor, progress, acting, profile, target)

    def _collect_until(
        self,
        actor: Actor,
        progress: ActorProgress,
        acting: Backbone,
        profile: DecisionTimeProfile,
        target: int,
    ) -> None:
        """The collection loop itself, with its time charged to `profile`.

        Split out only so the total the buckets decompose is the whole of this
        loop: the wall time of this actor's thread, from its first lock to its
        last episode, and nothing else.
        """
        actor_id = actor.config.actor_id
        while True:
            with profile.acquiring(self._lock):
                if self.report.decisions >= target or self.stopped_early:
                    # The budget, or the run's own decision to stop: a plateau
                    # is answered at the episode boundary after the crossing
                    # that found it, so every actor finishes the episode it is
                    # in and none of them starts another.
                    return
                # This actor's own rate, which under a ladder is not the rate
                # any other actor is drawing - and beside it the one number the
                # run publishes for itself, which is a schedule position rather
                # than any actor's rate.
                exploration = self.config.exploration
                epsilon = exploration.epsilon_for(
                    self.actor_index[actor_id], self.report.decisions
                )
                self.report.epsilon = exploration.reported_epsilon(
                    self.report.decisions
                )
            # Refreshed between episodes and never inside one: the copy's
            # parameters hold still for a whole episode, and the history window
            # the actor carries through that episode was produced by exactly the
            # parameters it is still acting from. `_lock` is released first, so
            # the only order locks are ever taken in is progress, then replay,
            # then learner.
            if self._since_sync[actor_id] >= self.config.parameter_sync_episodes:
                with profile.span(LEARNER_STEP):
                    self.learner.publish_to(acting)
                self._since_sync[actor_id] = 0
            # Exploration is set per episode rather than per step, so a stored
            # sequence has one epsilon and its provenance stays meaningful.
            actor.config = replace(actor.config, epsilon=epsilon)
            # The version of the parameters this episode is actually played
            # with, which is the copy's rather than the learner's: a sequence
            # must be stamped with the policy that produced it.
            actor.model_version = acting.model_version
            self._since_sync[actor_id] += 1
            try:
                result = actor.run_episode()
            except RunPortError as failure:
                # The port could not deliver an episode. That is a counted
                # outcome, not the end of the run: an episode classified
                # invalid by the environment already continues, and an episode
                # the port refused outright must not be treated more harshly.
                with profile.acquiring(self._lock):
                    self._record_failure(progress, failure)
                    progress.decision_time = profile.snapshot()
                    withdrawn = progress.withdrawn is not None
                    if withdrawn:
                        if self.on_withdrawal is not None:
                            self.on_withdrawal(progress)
                    else:
                        self._after_episode(profile)
                if withdrawn:
                    raise
                continue
            with profile.acquiring(self._lock):
                self._record_episode(progress, result)
                with profile.span(LEARNER_STEP):
                    self._learn(result.summary.decisions)
                # Published before the hooks, so a hook reading the fleet's
                # decomposition sees this episode's learning in it.
                progress.decision_time = profile.snapshot()
                self._after_episode(profile)
                barren = progress.withdrawn
                if barren is not None and self.on_withdrawal is not None:
                    self.on_withdrawal(progress)
            if barren is not None:
                # Episode after episode that reaches no choice point, or that
                # advances no game time at all: the actor is collecting nothing
                # to learn from and nothing the budget is counted in, so it
                # leaves the fleet exactly as one on a failing port does.
                raise RunPortError(barren)

    def _record_failure(self, progress: ActorProgress, failure: RunPortError) -> None:
        """Count an episode the port could not deliver, against run and actor."""
        report = self.report
        report.episodes += 1
        report.failed_episodes += 1
        report.episode_failures.append(str(failure))
        progress.episodes += 1
        progress.failed_episodes += 1
        progress.failures.append(str(failure))
        progress.consecutive_failures += 1
        if progress.consecutive_failures >= self.config.max_consecutive_episode_failures:
            progress.withdrawn = str(failure)

    def _record_episode(self, progress: ActorProgress, result: EpisodeResult) -> None:
        """Place one collected episode on the run's series and on its actor's."""
        report = self.report
        summary = result.summary
        report.episodes += 1
        report.decisions += summary.decisions
        # The measured round clock, reported beside the decisions.
        report.game_ms += summary.round_ms
        report.sequences_accepted += result.sequences_accepted
        report.reward_bearing_transitions += result.reward_bearing_transitions
        report.wave_change_ended_span += result.wave_change_ended_span
        # Completion order across the fleet: an episode joins the series when it
        # ends, which is what makes a window of them a slice of one wall-clock
        # span of collection rather than of one instance's history.
        report.collected.append(
            CollectedEpisode(
                summary, wait_decisions=result.wait_decisions, actor_id=progress.actor_id
            )
        )
        progress.episodes += 1
        progress.decisions += summary.decisions
        progress.game_ms += summary.round_ms
        if summary.decisions and summary.round_ms:
            progress.consecutive_failures = 0
        else:
            # Two barren outcomes, both delivered episodes rather than port
            # failures, and each named for what it was. A run that ended before
            # it offered a single choice is a real, valid episode (ADR 0009),
            # and an actor whose every run ends that way is collecting nothing
            # to learn from - and, the budget being decisions, one that would
            # never reach a target. An episode that advanced no game time at
            # all is a broken instance. Both count toward the same streak a
            # failing port does, so the condition is loud rather than an
            # unattended run that never finishes.
            progress.consecutive_failures += 1
            if progress.consecutive_failures >= self.config.max_consecutive_episode_failures:
                reason = (
                    "advanced no game time"
                    if not summary.round_ms
                    else "ended before a choice point"
                )
                progress.withdrawn = (
                    f"{progress.consecutive_failures} consecutive episodes {reason}"
                )
        if summary.valid:
            progress.valid_episodes += 1
        else:
            progress.invalid_episodes += 1

    def _after_episode(self, profile: DecisionTimeProfile) -> None:
        """The hooks one episode owes, on the thread that collected it.

        Charged to nobody: see `measured_apart`. A periodic evaluation plays
        whole episodes, and left inside the collecting span its wall time would
        land in this actor's residual and read as contention that never
        happened.
        """
        with measured_apart(profile):
            self._periodic(self.report)
            if self.on_episode is not None:
                self.on_episode(self.report)

    def _learn(self, decisions: int) -> None:
        """Take the gradient steps these decisions earned.

        The debt is the fleet's: every actor's decisions credit the one counter,
        so four actors buy four times the gradient steps in an hour and the
        configured replay ratio is what the run actually trains at.
        """
        report = self.report
        self._owed += decisions * self.config.gradient_steps_per_decision
        while self._owed >= 1.0 and self._warm():
            metrics = self._optimise(report.decisions)
            report.optimisation_steps += 1
            report.recent_weighted_losses.append(metrics.weighted_loss)
            report.recent_unweighted_td_errors.append(
                metrics.unweighted_mean_absolute_td_error
            )
            report.recent_gradient_norms.append(metrics.gradient_norm)
            if metrics.value_fit_correlation is not None:
                # A batch with too few completed episodes in it reports no
                # correlation rather than a zero that would look like a
                # learner predicting nothing.
                report.recent_value_fits.append(metrics.value_fit_correlation)
            for window in (
                report.recent_weighted_losses,
                report.recent_unweighted_td_errors,
                report.recent_gradient_norms,
                report.recent_value_fits,
            ):
                del window[:-100]
            self._owed -= 1.0
        if not self._warm():
            # Do not bank a debt of gradient steps while the buffer fills, or
            # the first warm episode would be followed by a burst of updates
            # on almost no data.
            self._owed = 0.0

    def _warm(self) -> bool:
        """Whether the buffer holds enough sequences for the first step."""
        with self.replay.lock:
            return len(self.replay) >= self.config.warmup_sequences

    def _periodic(self, report: TrainingProgressReport) -> None:
        """Evaluate, checkpoint, close a selection period and check kill bars.

        A numbered checkpoint is written where a selection period closes, so
        every period has the checkpoint its mean belongs to, and additionally
        on the `checkpoint_every_decisions` cadence. Each is answered once per
        crossing: an episode is far shorter than either period.
        """
        period = self.config.evaluate_every_episodes
        if self.evaluate is not None and period and report.episodes % period == 0:
            try:
                report.evaluations.append(self.evaluate())
            except (RunPortError, ValueError) as failure:
                # `evaluate` refuses to score an arm that produced no valid
                # episode, and the port can fail under it exactly as it can
                # under collection. Either way the point is lost, not the run.
                report.evaluation_failures.append(str(failure))
        period = self.config.checkpoint_every_episodes
        if self.checkpoint is not None and period and report.episodes % period == 0:
            self.checkpoint(report)
            report.checkpoints_written += 1
        before, self._periodic_at = self._periodic_at, report.decisions

        def crossed(every: int) -> bool:
            return bool(every) and report.decisions // every > before // every

        period_closed = crossed(self.config.selection_period_decisions)
        if self.numbered_checkpoint is not None and (
            period_closed or crossed(self.config.checkpoint_every_decisions)
        ):
            self.numbered_checkpoint(report)
            report.checkpoints_written += 1
        if period_closed:
            # After the checkpoint, so a run that stops here has written the
            # model the period it stopped on produced.
            self._close_period(report)
        self._check_kill_bars(report)

    def _check_kill_bars(self, report: TrainingProgressReport) -> None:
        """Check every kill bar the fleet's decisions have now reached.

        Each episode is placed on the decision axis where it ended, in the
        fleet's completion order - which is how the comparator run's episodes
        were placed too. A resumed segment holds only its own episodes, so a
        window that straddles the resume point is read over the part of it this
        segment collected.
        """
        reached = [
            index
            for index in self._bars_pending
            if self.config.kill_bars[index].at_decisions <= report.decisions
        ]
        if not reached:
            return
        near_greedy_ids = self.near_greedy_actor_ids
        for index in reached:
            self._bars_pending.remove(index)
            bar = self.config.kill_bars[index]
            waves: list[int] = []
            ended_at = self._segment_start_decisions
            for episode in report.collected:
                ended_at += episode.summary.decisions
                if (
                    bar.window_start_decisions < ended_at <= bar.at_decisions
                    and episode.summary.valid
                    and episode.actor_id in near_greedy_ids
                ):
                    waves.append(episode.summary.final_wave)
            mean = statistics.fmean(waves) if waves else None
            report.kill_bar_checks.append(
                KillBarCheck(
                    bar=bar,
                    decisions=report.decisions,
                    near_greedy_episodes=len(waves),
                    mean_final_wave=mean,
                    stopped=mean is not None and mean < bar.min_mean_final_wave,
                )
            )

    def _close_period(self, report: TrainingProgressReport) -> None:
        """Close the selection period this crossing ends, and stop on a plateau.

        The period's mean is the near-greedy actors' valid episodes that ended
        inside it: the series that reads as the policy's own performance rather
        than as search, which under a uniform schedule is every episode the
        fleet collected. A run whose best period mean is not improved on for
        `early_stop_patience_periods` periods in a row is not learning any more,
        and the rest of its budget buys nothing.
        """
        episodes = report.collected[self._period_start :]
        self._period_start = len(report.collected)
        near_greedy_ids = self.near_greedy_actor_ids
        near_greedy = [
            episode.summary.final_wave
            for episode in episodes
            if episode.summary.valid and episode.actor_id in near_greedy_ids
        ]
        mean = statistics.fmean(near_greedy) if near_greedy else None
        plateau = report.plateau
        plateau.close_period(
            mean, min_improvement=self.config.early_stop_min_improvement
        )
        report.selection_periods.append(
            SelectionPeriod(
                index=plateau.periods_closed,
                decisions_at_end=report.decisions,
                near_greedy_episodes=len(near_greedy),
                mean_final_wave=mean,
                best_mean_final_wave=plateau.best_mean_final_wave,
            )
        )
        if plateau.plateaued(self.config.early_stop_patience_periods):
            plateau.stopped_at_period = plateau.periods_closed

    def _optimise(self, decisions: int) -> LearnMetrics:
        # The buffer's lock is held across sampling, learning and the priority
        # update together: an actor adding to a full buffer in between would
        # evict a sequence and shift every index this batch was sampled at,
        # which replay refuses outright rather than applying to the wrong one.
        beta = self.config.beta(decisions)
        self.report.importance_beta = beta
        with self.replay.lock:
            indices, sequences, weights = self.replay.sample(
                self.config.batch_size, beta=beta
            )
            # Built where the parameters are: a CPU batch handed to a CUDA model
            # fails on the first optimisation step, which is the worst place to
            # discover it after an hour of collection.
            metrics = self.learner.learn(
                collate(sequences, weights, device=self.backbone.device)
            )
            self.replay.update_priorities(indices, metrics.td_errors)
        return metrics
