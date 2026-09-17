"""The training loop that ties actor, replay, learner and checkpointing together.

Budget is counted in environment decisions rather than episodes.  That choice
matters for fairness: a better policy survives longer, so an episode budget would
quietly hand the stronger arm more real experience and flatter it.  Decisions are
what the environment actually costs.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from tower_rl.application.actor import Actor, ActorConfig
from tower_rl.application.evaluator import EvaluationReport
from tower_rl.application.replay import PrioritizedSequenceReplay
from tower_rl.domain.episode import EpisodeSummary
from tower_rl.learning.backbone import Backbone, LearnMetrics, collate
from tower_rl.ports.run_port import RunPortError


def _mean(values: list[float]) -> float | None:
    """The mean of a health window, or None before it holds anything."""
    if not values:
        return None
    return sum(values) / len(values)


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
    #: How many episodes in a row may fail at the port before the run gives up.
    #: A single failed episode is an ordinary event on a real device and must not
    #: end a run that has hours of experience in it; a device that fails every
    #: episode is a broken instance, and continuing would spin without collecting
    #: anything. The limit is what separates the two.
    max_consecutive_episode_failures: int = 5

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
    spent = 0
    window_decisions = 0
    for episode in collected:
        spent += episode.summary.decisions
        window_decisions += episode.summary.decisions
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
            )
        )
        current = []
        window_decisions = 0
    return windows


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
    #: Episodes the port could not produce at all - a boundary that would not
    #: settle, an instance that would not start. They are counted in `episodes`
    #: like any other attempt, but they leave no summary behind, so the two
    #: counts differ exactly by this one.
    failed_episodes: int = 0
    episode_failures: list[str] = field(default_factory=list)
    #: Periodic evaluations that could not be scored. Training continues: an
    #: evaluation is measurement, and losing a measurement must not lose the run.
    evaluation_failures: list[str] = field(default_factory=list)

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


@dataclass
class TrainingRun:
    """One arm: one actor, one replay buffer, one backbone, one budget."""

    actor: Actor
    replay: PrioritizedSequenceReplay
    backbone: Backbone
    config: TrainingConfig
    on_episode: Callable[[TrainingProgressReport], None] | None = None
    #: Runs exploration-free episodes on the same device. Its cost comes out of
    #: wall-clock time, never out of the decision budget, because evaluation is
    #: measurement rather than experience.
    evaluate: Callable[[], EvaluationReport] | None = None
    checkpoint: Callable[[TrainingProgressReport], None] | None = None
    #: Progress so far. It is instance state rather than a local because a run
    #: can be advanced in blocks: several arms sharing one device take turns, so
    #: whatever drifts on the device lands on all of them equally.
    report: TrainingProgressReport = field(default_factory=TrainingProgressReport)
    #: Gradient steps earned but not yet taken, carried across blocks.
    _owed: float = field(default=0.0, init=False)
    #: Episodes the port failed in a row, carried across blocks for the same
    #: reason: an instance that fails every episode must not look healthy again
    #: merely because the arms took turns.
    _consecutive_failures: int = field(default=0, init=False)

    @property
    def finished(self) -> bool:
        return self.report.decisions >= self.config.budget_decisions

    def run(self) -> TrainingProgressReport:
        """Collect and learn until the decision budget is spent."""
        return self.advance(self.config.budget_decisions)

    def advance(self, decisions: int) -> TrainingProgressReport:
        """Collect and learn for up to `decisions` more, never past the budget.

        The limit lands on an episode boundary: an episode in progress is played
        to its classified end, because a half-episode is not experience.
        """
        if decisions < 1:
            raise ValueError("a block must be at least one decision")
        report = self.report
        target = min(report.decisions + decisions, self.config.budget_decisions)
        started = time.monotonic()
        owed = self._owed

        while report.decisions < target:
            # Exploration is set per episode rather than per step, so a stored
            # sequence has one epsilon and its provenance stays meaningful.
            self.actor.config = replace(
                self.actor.config, epsilon=self.config.epsilon(report.decisions)
            )
            self.actor.model_version = self.backbone.model_version
            try:
                result = self.actor.run_episode()
            except RunPortError as failure:
                # The port could not deliver an episode. That is a counted
                # outcome, not the end of the run: an episode classified
                # invalid by the environment already continues, and an episode
                # the port refused outright must not be treated more harshly.
                report.episodes += 1
                report.failed_episodes += 1
                report.episode_failures.append(str(failure))
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.max_consecutive_episode_failures:
                    raise
                self._periodic(report)
                if self.on_episode is not None:
                    self.on_episode(report)
                continue
            self._consecutive_failures = 0

            report.episodes += 1
            report.decisions += result.summary.decisions
            report.sequences_accepted += result.sequences_accepted
            report.collected.append(
                CollectedEpisode(result.summary, wait_decisions=result.wait_decisions)
            )

            owed += result.summary.decisions * self.config.gradient_steps_per_decision
            while owed >= 1.0 and len(self.replay) >= self.config.warmup_sequences:
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
                owed -= 1.0
            if len(self.replay) < self.config.warmup_sequences:
                # Do not bank a debt of gradient steps while the buffer fills, or
                # the first warm episode would be followed by a burst of updates
                # on almost no data.
                owed = 0.0

            self._periodic(report)
            if self.on_episode is not None:
                self.on_episode(report)

        self._owed = owed
        report.wall_seconds = round(report.wall_seconds + time.monotonic() - started, 2)
        return report

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

    def _optimise(self, decisions: int) -> LearnMetrics:
        indices, sequences, weights = self.replay.sample(
            self.config.batch_size, beta=self.config.beta(decisions)
        )
        # Built where the parameters are: a CPU batch handed to a CUDA model
        # fails on the first optimisation step, which is the worst place to
        # discover it after an hour of collection.
        metrics = self.backbone.learn(
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
    "CollectedEpisode",
    "CollectionWindow",
    "action_distribution",
    "collection_windows",
    "TrainingConfig",
    "TrainingProgressReport",
    "TrainingRun",
    "episode_budget",
]
