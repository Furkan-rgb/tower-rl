"""The training loop that ties actor, replay, learner and checkpointing together.

Budget is counted in environment decisions rather than episodes.  That choice
matters for fairness: a better policy survives longer, so an episode budget would
quietly hand the stronger arm more real experience and flatter it.  Decisions are
what the environment actually costs.

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
from collections.abc import Callable, Iterator, Sequence
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
from tower_rl.learning.actor import Actor, ActorConfig, EpisodeResult
from tower_rl.learning.backbone import (
    Backbone,
    LearnMetrics,
    SequenceBatch,
    acting_copy,
    collate,
)
from tower_rl.learning.evaluator import EvaluationReport
from tower_rl.learning.replay import PrioritizedSequenceReplay

#: The device's own rejection reason for a stale or duplicate command, carried
#: into an episode's `termination_detail` free text exactly as it comes off the
#: bridge. `scripts/run_actors.py` names the identical literal
#: (`STALE_OR_DUPLICATE`) for its own report; it is not imported from there
#: because the learning package does not depend on a script.
STALE_OR_DUPLICATE = "stale_or_duplicate"


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

    #: The equalised budget. Every arm of a comparison gets the same number.
    budget_decisions: int
    #: Sequences required before the first optimisation step. About 35 episodes
    #: at this geometry: enough that the first gradient steps see more than a
    #: handful of episodes of one policy.
    warmup_sequences: int = 100
    batch_size: int = 8
    #: Gradient steps per environment decision: the replay ratio. What matters
    #: is the transitions replayed per transition generated, which is this times
    #: the learnable steps in a batch - at 80-step sequences, burn-in 7, n-step
    #: 10 and batch 8 that is about 504 per step, so 0.25 puts the run at 126:1,
    #: between SPR (64) and BBF (256). The 2.0 of the first run was 1087:1.
    gradient_steps_per_decision: float = 0.25
    #: Exploration anneals from start to end over `epsilon_anneal_decisions` and
    #: is held at `epsilon_end` afterwards.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    #: The horizon of the anneal, in decisions, and deliberately not the budget:
    #: annealing across the whole budget spent over half the first run at an
    #: epsilon above 0.5, so most of what was collected was near-random and the
    #: collection curve could not be read as a policy's performance at all.
    epsilon_anneal_decisions: int = 10_000
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
    #: Decisions between the numbered checkpoints a later evaluation chooses
    #: among. Zero leaves only the resume point, which is overwritten as the run
    #: proceeds and therefore names no particular model. A period in decisions
    #: rather than in episodes because decisions are the budget unit the arms
    #: are equalised on: two runs whose episodes differ in length still produce
    #: checkpoints at the same points of their budgets.
    checkpoint_every_decisions: int = 0
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

    def __post_init__(self) -> None:
        if self.budget_decisions < 1:
            raise ValueError("budget must be positive")
        if self.batch_size < 1 or self.warmup_sequences < 1:
            raise ValueError("batch size and warm-up must be positive")
        if self.gradient_steps_per_decision <= 0:
            raise ValueError("gradient steps per decision must be positive")
        if self.epsilon_anneal_decisions < 1:
            raise ValueError("the epsilon anneal horizon must be positive")
        if self.collection_window_episodes < 1:
            raise ValueError("a collection window needs at least one episode")
        if self.max_consecutive_episode_failures < 1:
            raise ValueError("at least one episode failure must be survivable")
        if self.parameter_sync_episodes < 1:
            raise ValueError("actors must be synchronised at least every episode")
        if self.checkpoint_every_decisions < 0:
            raise ValueError("a checkpoint period in decisions cannot be negative")

    def progress(self, decisions: int) -> float:
        return min(1.0, decisions / self.budget_decisions)

    def epsilon(self, decisions: int) -> float:
        """Anneal over the horizon, then hold - never over the whole budget."""
        fraction = min(1.0, decisions / self.epsilon_anneal_decisions)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * fraction

    def beta(self, decisions: int) -> float:
        fraction = self.progress(decisions)
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


def collection_windows(
    collected: Sequence[CollectedEpisode], *, size: int
) -> list[CollectionWindow]:
    """Cut the collection episodes into consecutive non-overlapping windows.

    Windows are counted in valid episodes, because an invalid episode has no
    final wave to average; the decisions of every episode still count towards
    the budget position a window is placed at, since they were all spent. A
    trailing partial window is not emitted at all: a point averaged over fewer
    episodes than the rest has a different standard error and would be read as
    if it did not.
    """
    if size < 1:
        raise ValueError("a collection window needs at least one episode")
    windows: list[CollectionWindow] = []
    current: list[CollectedEpisode] = []
    #: Every episode attempted while this window was filling, valid and invalid
    #: alike - the span `health` is pooled over, wider than `current`.
    attempted: list[CollectedEpisode] = []
    spent = 0
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

    decisions: int = 0
    episodes: int = 0
    optimisation_steps: int = 0
    sequences_accepted: int = 0
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
    #: The exploration rate the run last acted at and the importance-sampling
    #: exponent it last sampled at. Published here by the run that draws them
    #: from its schedules, so a checkpoint or a report carries the value the run
    #: actually used rather than re-evaluating a learning schedule of its own.
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
    #: wall-clock time, never out of the decision budget, because evaluation is
    #: measurement rather than experience. It borrows an instance, so it may only
    #: run while the fleet is not collecting.
    evaluate: Callable[[], EvaluationReport] | None = None
    checkpoint: Callable[[TrainingProgressReport], None] | None = None
    #: Called when the fleet crosses a multiple of `checkpoint_every_decisions`,
    #: to write a checkpoint under its own name. Separate from `checkpoint`
    #: above, which keeps the one resume point: a numbered checkpoint is a
    #: candidate a later evaluation may choose, so it must survive the next one
    #: being written.
    numbered_checkpoint: Callable[[TrainingProgressReport], None] | None = None
    #: Progress so far. It is instance state rather than a local because a run
    #: can be advanced in blocks: several arms sharing one device take turns, so
    #: whatever drifts on the device lands on all of them equally.
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
    #: The last multiple of `checkpoint_every_decisions` a numbered checkpoint
    #: was written for. Episodes end whole, so the counter jumps past a multiple
    #: rather than landing on it, and several multiples can fall inside one long
    #: episode; this records which of them have already been answered so a
    #: crossing produces exactly one checkpoint.
    _numbered_at: int = field(default=0, init=False)
    #: Episodes each actor has played since its copy was last refreshed. Starts
    #: at the cadence so every actor publishes before its first episode, which
    #: is also what picks up a checkpoint loaded into the backbone after the run
    #: was built. Each actor touches only its own entry of a dict whose keys are
    #: all present from construction, so it needs no lock of its own.
    _since_sync: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.actors:
            raise ValueError("a run needs at least one actor")
        identities = [actor.config.actor_id for actor in self.actors]
        if len(set(identities)) != len(identities):
            # Per-actor reporting is keyed by identity; two actors under one
            # name would report as one instance and hide a dead one.
            raise ValueError("every actor of a fleet needs an id of its own")
        self.learner = Learner(self.backbone)
        # The cadence is continued rather than restarted: a run resumed at
        # 50,123 decisions of a 100,000-decision period has already answered the
        # multiple at 50,000, and the next checkpoint it owes is the one at
        # 100,000. Derived from the report it was constructed with, so a fresh
        # run - whose counter is zero - is unaffected.
        if self.config.checkpoint_every_decisions:
            period = self.config.checkpoint_every_decisions
            self._numbered_at = self.report.decisions // period * period
        self.report.epsilon = self.config.epsilon(self.report.decisions)
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
    def finished(self) -> bool:
        return self.report.decisions >= self.config.budget_decisions

    def run(self) -> TrainingProgressReport:
        """Collect and learn until the decision budget is spent."""
        return self.advance(self.config.budget_decisions)

    def advance(self, decisions: int) -> TrainingProgressReport:
        """Collect with every live actor and learn until `decisions` more are spent.

        The limit is the fleet's and lands on an episode boundary for each actor:
        an episode in progress is played to its classified end, because a half
        episode is not experience.

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
                if self.report.decisions >= target:
                    return
                epsilon = self.config.epsilon(self.report.decisions)
                self.report.epsilon = epsilon
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
        report.sequences_accepted += result.sequences_accepted
        # Completion order across the fleet: an episode joins the series when it
        # ends, which is what makes a window of them a slice of one wall-clock
        # span of collection rather than of one instance's history.
        report.collected.append(
            CollectedEpisode(
                summary, wait_decisions=result.wait_decisions, actor_id=progress.actor_id
            )
        )
        progress.consecutive_failures = 0
        progress.episodes += 1
        progress.decisions += summary.decisions
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
        """Evaluate and checkpoint on their episode periods, if configured."""
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
        decisions_period = self.config.checkpoint_every_decisions
        if self.numbered_checkpoint is not None and decisions_period:
            # The multiple this many decisions has reached, which is what the
            # crossing is counted by: the counter lands past a multiple, not on
            # it, because an episode is played to its end.
            reached = report.decisions // decisions_period * decisions_period
            if reached > self._numbered_at:
                self._numbered_at = reached
                self.numbered_checkpoint(report)
                report.checkpoints_written += 1

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


def episode_budget(config: TrainingConfig, decisions_per_episode: float) -> int:
    """Roughly how many episodes a decision budget buys, for planning only.

    Reported rather than used: a stronger policy survives longer and therefore
    spends the same decision budget over fewer episodes, which is exactly why the
    budget is counted in decisions.
    """
    if decisions_per_episode <= 0:
        raise ValueError("decisions per episode must be positive")
    return max(1, round(config.budget_decisions / decisions_per_episode))


__all__ = [
    "ActionDistribution",
    "ActorConfig",
    "ActorProgress",
    "CollectedEpisode",
    "CollectionWindow",
    "EpisodeHealth",
    "action_distribution",
    "collection_windows",
    "episode_health",
    "Learner",
    "TrainingConfig",
    "TrainingProgressReport",
    "TrainingRun",
    "episode_budget",
]
