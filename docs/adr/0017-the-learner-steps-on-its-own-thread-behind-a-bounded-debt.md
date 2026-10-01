# ADR 0017 — The learner steps on its own thread, behind a bounded debt

**Status:** adopted in code, 2026-09-30. Board `#102`. Verified on fake ports
and on the workstation GPU with fake ports; the device A/B against the
fixed-policy benchmark (`#101`) is still to run.

**Partly superseded by ADR 0018:** parameter publication and policy lag (the
*Policy lag* and *Publication* points of the decision below) - DreamerV3 now
loads the last completed step before every decision and carries its latent
across the swap. The learner thread, the bounded debt and the rest stand.

**Partly superseded by board #111 (R2D2):** the debt is counted in credits, not
decisions. A credit is one decision for DreamerV3 (0.5 steps, bound 512) and
one inserted item for R2D2 (5 steps per item, bound 125 items = 625 steps,
Acme's `error_buffer`; `R2D2_LEARNER_DEBT_BOUND_ITEMS`). The 512-step bound and
the 519-step lag below are the removed stacked-dqn's figures (commit
`cb2f324`). R2D2's worst hold is every actor ending an episode inside a 11.5 s
save, 8 x 60 = 480 steps, under 625.

## Context

Until now the actor whose episode ended took every gradient step that episode
earned, on its own thread and under the run's progress lock
(`TrainingRun._lock`). Its emulator idled for them, and any other actor
finishing an episode queued behind them. `M3-P016`'s 7-actor segments show
the cost on the timing line: learn 28–36 ms and blocked 0–107 ms per decision
beside a 137–259 ms bridge round trip, at 9,168–17,549 decisions/hour per
actor (`state/runs/m3-p016-dreamerv3-v2-20260929T201403Z/segments/1/train.log`
and `…-20260930T062633Z/…`, local).

## Decision

**One learner thread takes the gradient steps while the actors collect. The
replay ratio is held by a debt the actors' decisions credit, and an actor
pauses only while the debt is over a bound.**

- **The debt.** Each decision credits the learner `gradient_steps_per_decision`
  steps as it is taken, once the buffer is warm. The debt is derived from two
  integers, decisions credited and steps taken, so over any span the steps are
  the ratio of its decisions exactly, short by at most the bound. An episode is
  squared when it is counted: the episode that warms the buffer earns all its
  decisions, as every episode ending warm always did, and an episode abandoned
  or failed takes its credit back, as it never counted. A block ends by paying
  every whole step still owed, so it ends owing under one step.
- **Across a resume.** A checkpoint records the debt the learner owes on the
  episodes counted so far (`progress.learner_debt_steps`); the episodes in
  flight when a run stops are lost, and their credit with them. A resume that
  reloads its parent's buffer carries that debt and pays it first, so over the
  parent's segments and the resumed one the steps taken are the ratio of the
  decisions counted since the first warm episode, to within the under-one-step
  remainder a block ends on: what the old rule gave a run that never stopped.
  A checkpoint written mid-block records what was owed at that moment, up to
  the bound, and the resume pays it. A resume that re-warms an empty buffer
  starts owing nothing, since the debt was owed on a buffer it no longer has.
- **The bound.** `learner_debt_bound_decisions`, default **512 decisions**
  (512 steps for stacked-dqn at 1.0, 256 for DreamerV3 at 0.5).
  - *Principle:* the smallest debt that never pauses an actor while the
    learner keeps up on average. It must cover the longest ordinary stretch in
    which the learner is held still.
  - *Quantity:* that stretch is a resume-point save, which holds the learner:
    11.5 s at 1,000,000 replay steps (`docs/experiments.md`, "Crash-safe
    resume point"). A 7-actor fleet at up to 17,549 decisions/hour per actor
    collects about 34 decisions/s, so about 390 decisions.
  - *Value:* 512, the next power of two above that, with 30% to spare. It is
    below the 700-step lag the default refresh of 100 decisions already
    accepts at 7 actors (§6.10).
- **Policy lag.** The learner is at most `bound + actors` decisions behind
  collection, since each actor credits before it checks: 519 decisions at 7
  actors, 519 steps for stacked-dqn and 260 for DreamerV3. An acting copy adds
  its refresh cadence on top, unchanged: 7 × 10 × 1.0 = 70 steps for the
  stacked-dqn recipe, one episode's fleet steps for DreamerV3. Before this,
  the steps an in-flight episode earned were not taken until it ended, so the
  fleet already acted about one episode per actor behind the ideal.
- **Holding still.** A checkpoint, a periodic evaluation and the run's last
  resume point hold the learner still (`LearnerThread.held`): no step begins
  and the one in flight is waited out, so a saved or scored network is one
  completed step and `optimisation_steps` equals the weights' own count. The
  learner never takes the progress lock, so the progress lock's holder can
  hold it. Lock order: progress lock, learner held, replay lock,
  `Learner.lock`; the debt's condition is a leaf.
- **Replay.** The learner holds the replay lock to sample and again to update
  priorities, not across the step, so an actor adding its episode never waits
  on a step. `update_priorities` shifts each sampled index by the evictions
  since the sample and drops an evicted one; it used to refuse any eviction.
- **Publication.** Unchanged in shape: an actor copies the learner's
  parameters into its own copy, on its own thread, under `Learner.lock`, which
  a step also holds. A copy therefore waits for at most the one step in flight,
  per refresh. That is the one place an actor still waits on a learn step,
  charged to `learner_step`.
- **GPU.** On CUDA the learner issues on a stream of its own; actors act on the
  default stream, so a forward pass does not queue behind a step's kernels. A
  step synchronises its stream before releasing `Learner.lock`, a publication
  synchronises the actor's stream before releasing it, and the learner's stream
  waits on the default stream when the thread starts (a resume's load).
- **Stopping.** SIGINT's stop abandons episodes as before; the block then
  drains the debt (at most the bound: about 15 s of DreamerV3 steps, 512
  stacked-dqn steps) before the last resume point. A second SIGINT or an
  exception ends the learner after the step in flight. A step that raises ends
  the thread, abandons the actors' episodes and is raised by the run.
- **Reporting.** The per-decision timing line gains the learner's steps,
  utilisation (stepping time over running time), the debt when read against
  the bound, and actors' paused ms per decision; MLflow gains
  `learner_utilization`, `learner_debt_steps` and `learner_paused_ms`.

## Determinism

A fleet of one is no longer the loop it was before fleets: its parameters now
move inside its episodes, and which transitions a batch is drawn from depends
on when the learner reaches it. Same seed, same fleet, same result was already
untrue for fleets of more than one; it is now untrue for one as well. The
fleet-of-one tests that pinned the old behaviour were replaced with one that
shows the new behaviour.

## Consequences

- The checkpoint format is 6. A checkpoint from before this (format 5 or
  earlier) was trained with its steps taken on the actors' threads; a resume
  of it is refused by name, because continuing it on the learner thread would
  make one run of two learners. It still loads for evaluation and selection.
  `M3-P016`'s checkpoints are format 4 and 5, so that run cannot be continued;
  it can only be started again. The bound is recorded in the resolved config
  from now on, not compared on resume, and has no flag.
- The gain is capped by the learner. At DreamerV3's ~60 ms per step under load
  and 0.5 steps per decision, the learner serves about 33 decisions/s, close
  to what the fleet collects; past that the bound paces the actors, and the
  timing line's paused figure says so.
- A reset (`--reset-every-steps`) is logged at the first episode that sees it;
  those taken in a block's closing drain have no episode after them, and are
  logged when the block ends (`TrainingRun.on_block_end`).
- Weights are no longer a function of the decision count: the closing drain
  steps after the last episode's hooks. A curve point's checkpoint is
  therefore named `decisions-<d>-v<model version>.pt`, so the final
  evaluation no longer overwrites a periodic point's file at the same count.
