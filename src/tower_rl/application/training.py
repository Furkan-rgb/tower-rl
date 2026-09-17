"""The training loop that ties actor, replay, learner and checkpointing together.

Budget is counted in environment decisions rather than episodes.  That choice
matters for fairness: a better policy survives longer, so an episode budget would
quietly hand the stronger arm more real experience and flatter it.  Decisions are
what the environment actually costs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from tower_rl.application.actor import Actor, ActorConfig
from tower_rl.application.evaluator import EvaluationReport
from tower_rl.application.replay import PrioritizedSequenceReplay
from tower_rl.domain.episode import EpisodeSummary
from tower_rl.learning.backbone import Backbone, LearnMetrics, collate


@dataclass(frozen=True)
class TrainingConfig:
    """One arm's budget and schedules, recorded with its result."""

    #: The equalised budget. Every arm of a comparison gets the same number.
    budget_decisions: int
    #: Sequences required before the first optimisation step.
    warmup_sequences: int = 16
    batch_size: int = 8
    #: Gradient steps per environment decision. Above one this is the replay ratio
    #: that the data-efficient literature raises; it costs GPU rather than device
    #: time, which is the resource we are not short of.
    gradient_steps_per_decision: float = 0.5
    #: Exploration anneals from start to end across the budget.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    #: Importance-sampling correction anneals the other way, as is conventional.
    beta_start: float = 0.4
    beta_end: float = 1.0
    #: Zero disables the periodic hook entirely; a positive value is a period in
    #: episodes. Evaluation is exploration-free and never writes to replay.
    evaluate_every_episodes: int = 0
    checkpoint_every_episodes: int = 0

    def __post_init__(self) -> None:
        if self.budget_decisions < 1:
            raise ValueError("budget must be positive")
        if self.batch_size < 1 or self.warmup_sequences < 1:
            raise ValueError("batch size and warm-up must be positive")
        if self.gradient_steps_per_decision <= 0:
            raise ValueError("gradient steps per decision must be positive")

    def progress(self, decisions: int) -> float:
        return min(1.0, decisions / self.budget_decisions)

    def epsilon(self, decisions: int) -> float:
        fraction = self.progress(decisions)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * fraction

    def beta(self, decisions: int) -> float:
        fraction = self.progress(decisions)
        return self.beta_start + (self.beta_end - self.beta_start) * fraction


@dataclass
class TrainingProgressReport:
    """What a run produced, enough to compare arms and to resume."""

    decisions: int = 0
    episodes: int = 0
    optimisation_steps: int = 0
    sequences_accepted: int = 0
    wall_seconds: float = 0.0
    episode_summaries: list[EpisodeSummary] = field(default_factory=list)
    recent_losses: list[float] = field(default_factory=list)
    evaluations: list[EvaluationReport] = field(default_factory=list)
    checkpoints_written: int = 0

    @property
    def valid_episodes(self) -> int:
        return sum(1 for summary in self.episode_summaries if summary.valid)

    @property
    def final_waves(self) -> list[int]:
        return [summary.final_wave for summary in self.episode_summaries if summary.valid]


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

    def run(self) -> TrainingProgressReport:
        """Collect and learn until the decision budget is spent."""
        report = TrainingProgressReport()
        started = time.monotonic()
        owed = 0.0

        while report.decisions < self.config.budget_decisions:
            # Exploration is set per episode rather than per step, so a stored
            # sequence has one epsilon and its provenance stays meaningful.
            self.actor.config = replace(
                self.actor.config, epsilon=self.config.epsilon(report.decisions)
            )
            self.actor.model_version = self.backbone.model_version
            result = self.actor.run_episode()

            report.episodes += 1
            report.decisions += result.summary.decisions
            report.sequences_accepted += result.sequences_accepted
            report.episode_summaries.append(result.summary)

            owed += result.summary.decisions * self.config.gradient_steps_per_decision
            while owed >= 1.0 and len(self.replay) >= self.config.warmup_sequences:
                metrics = self._optimise(report.decisions)
                report.optimisation_steps += 1
                report.recent_losses.append(metrics.loss)
                del report.recent_losses[:-100]
                owed -= 1.0
            if len(self.replay) < self.config.warmup_sequences:
                # Do not bank a debt of gradient steps while the buffer fills, or
                # the first warm episode would be followed by a burst of updates
                # on almost no data.
                owed = 0.0

            self._periodic(report)
            if self.on_episode is not None:
                self.on_episode(report)

        report.wall_seconds = round(time.monotonic() - started, 2)
        return report

    def _periodic(self, report: TrainingProgressReport) -> None:
        """Evaluate and checkpoint on their episode periods, if configured."""
        period = self.config.evaluate_every_episodes
        if self.evaluate is not None and period and report.episodes % period == 0:
            report.evaluations.append(self.evaluate())
        period = self.config.checkpoint_every_episodes
        if self.checkpoint is not None and period and report.episodes % period == 0:
            self.checkpoint(report)
            report.checkpoints_written += 1

    def _optimise(self, decisions: int) -> LearnMetrics:
        indices, sequences, weights = self.replay.sample(
            self.config.batch_size, beta=self.config.beta(decisions)
        )
        metrics = self.backbone.learn(collate(sequences, weights))
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
    "ActorConfig",
    "TrainingConfig",
    "TrainingProgressReport",
    "TrainingRun",
    "episode_budget",
]
