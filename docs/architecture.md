# Tower-RL — Architecture

This is the structural map of the code that exists. It describes the four
packages under `src/tower_rl/`, what each owns, where state lives, and the two
flows a run actually takes. It describes nothing that is not implemented; a
planned design belongs in `docs/solution.md` until it is built.

`docs/task.md` is authoritative for scope and gates. `docs/solution.md` is
authoritative for the technical approach. `docs/environment-contract.md` defines
the observation, action, episode and fidelity schemas. Where this document and
the code disagree, the code wins and this document is wrong.

## 1. The dependency rule

Four packages, one direction:

```text
scripts/            composition only; imports packages, and no package imports it
  experiment/       may import environment and learning
  learning/         may import environment
  simulation/       may import environment
environment/        the root; imports nothing else in tower_rl
```

`environment` is the root because it defines what a run *is* — the actions, the
state, the episode, the port a run is driven through, and what a decision costs.
It is the one package that can be reasoned about without a device, a learner or
an experiment.

`simulation` and `learning` are siblings and must not know about each other. The
join between them is `environment.run_port.RunPort`: `simulation` supplies an
adapter that satisfies it, `learning` drives an `InstrumentedRunEnvironment`
that consumes it, and neither imports the other. That is what lets a learner run
against a fake port in a unit test, and what keeps host and emulator detail out
of everything that is not device-facing (`find_android_tool` moved down into
`simulation` because of this rule rather than the rule being widened to let it
stay in `doctor`).

`experiment` observes both and drives neither device nor script.

`scripts/` is the composition root. A script wires a bridge client to an
adapter, an adapter to an environment, an environment to actors and a learner,
and reports the result. Packages never import a script module, so nothing in
`src/` can only be run from one entry point.

### Enforcement

`tests/unit/test_import_contracts.py` is the enforcement, not this document. It
parses every `.py` file under `src/tower_rl/` and under `tests/` with `ast` and
reports each import that crosses a boundary it may not. It reads from the source
rather than from a runtime import graph, so a module nobody imports is still
held to the rule.

Each rule is one constant plus one `offences(...)` call. Learning, simulation
and the tests are stated as *allowances* ("only the environment") rather than as
lists of forbidden neighbours, so a package added later is forbidden by default
instead of by somebody remembering to name it.

To change the rules, change the constants in that file and say why in the
docstring beside them. A rule that is relaxed to let one import compile is the
failure mode this test exists to make visible.

## 2. `environment` — what a run is

Owns the decision problem, and nothing about how a device is reached.

- `run_actions.py` — `RunActionId`, the `RUN_ACTIONS` inventory (`WAIT` plus one
  buy per upgrade family and slot), `action_index`/`action_at`, and
  `ACTION_SCHEMA_VERSION`.
- `run_state.py` — `RunState` and `RunStateBuilder`, which turn one bridge
  reading into a validated observation; `OBSERVATION_SCHEMA_VERSION`,
  `validate_transition`. It also holds `LIVE_FIELDS`, the
  single declaration of what `observation-v2` shows the policy beyond the
  upgrade grid — one `Main` field per row, its observed unit, and the transform
  that rescales it — together with `scale_live_reading`, which enforces each
  field's range invariant, and `hud_readings`, its inverse, which puts a reading
  back in the unit a human can check against the game's own HUD. The bridge
  decoder requires exactly these wire names and `features.SCALAR_FEATURES` takes
  its order from them, so the schema is declared once and read everywhere.
- `features.py` — `encode_state` to a `StateFeatures` (scalars plus one row per
  upgrade action). The only place raw state becomes network input. The widths
  the network is built from derive from `SCALAR_FEATURES`/`ROW_FEATURES`, which
  is why growing the schema needs no change in `learning/network.py`.
- `episode.py` — `TerminationOutcome`, `ActionOutcome`, `DecisionEvent`,
  `DecisionView`, `RunTransition`, `WaveRecord`, `EpisodeSummary`,
  `wave_progress_reward`, `REWARD_SCHEMA_VERSION`. `DecisionEvent` is the
  cadence condition an advance stopped on; `DecisionView` is the unrelated
  thing a spectator reads — one decision as a human sees it, emitted by
  `InstrumentedRunEnvironment.on_decision`.
- `run_port.py` — the `RunPort` protocol and `RunPortError`: the whole surface a
  run is driven through, including the upgrade-row labels and the round-start
  unlock availability is applied with.
- `run_environment.py` — `InstrumentedRunEnvironment`, `CadenceConfig`,
  `DecisionCadence` and `UpgradeAvailability`. It decides when a decision is
  due, charges the game clock, checks fidelity
  (`MIN_ROUND_CLOCK_RATIO`/`MAX_ROUND_CLOCK_RATIO`, `GAME_TIME_INFLATED`,
  `GAME_TIME_DEFLATED`, `ADVANCE_TRUNCATED_BY_WALL`, `BRIDGE_EVENT_DIVERGENCE`)
  and recovers the death-boundary transient. It also owns upgrade availability
  (ADR 0011): under `UpgradeAvailability.ALL` it reopens every real upgrade row
  at each round start through the port and holds the episode to it
  (`UNLOCK_NOT_APPLIED`, `UNLOCK_REVERTED`).
- `decision_time.py` — `DecisionTimeProfile` and `DecisionTimeBreakdown`: where
  a decision's wall time went, by bucket.
- `project_state.py` — `repository_root` and `state_directory`: the git-ignored
  `<repo>/state/` every artifact this project writes lives under, resolved from
  the package's own location rather than the cwd. It is here because this is the
  root package, so simulation (bridge builds), experiment (the MLflow store) and
  the scripts (runs, records, recordings) may all import it and none of them
  spells the location out for itself.

**Stepping.** One `step` executes the semantic action and then advances the
world until the policy has something to choose again: under the default
`DecisionCadence.CHOICE_POINTS` it keeps advancing while the settled observation
offers only `WAIT` (`RunState.is_choice_point`), and stops at the first choice
point, at the end of the run, or at the first thing that makes the transition
inadmissible. The reset that produces an episode's first observation runs the
same loop. Each internal advance is an ordinary one — same `CadenceConfig`, same
`_events_between` divergence check, same wave and episode tallies — so what is
withheld is the decision, never an advance's accounting; the transition that
comes back carries the span's `advances`, its measured `game_ms`, its events and
the wave progress across all of it. `DecisionCadence.EVERY_SLICE` asks at every
stop instead, which is what run 1 collected under, and exists to reproduce it
(ADR 0009).

**State.** The per-episode tally (`_EpisodeTally`, `_WaveTally`) lives inside
one `InstrumentedRunEnvironment` instance and leaves it only as an immutable
`EpisodeSummary`. Everything else here is a frozen dataclass.

## 3. `simulation` — reaching one instance

Owns emulator lifecycle, the wire protocol, and the adapter behind `RunPort`. It
knows nothing about the run being driven on it.

- `android_sdk.py` — locating `adb`, `emulator`, `sdkmanager`, `apkanalyzer`.
- `instance.py` — `CloneInstance` (index → serial, console port, forwarded
  bridge port), `adb`, `launch_emulator`, `wait_for_boot`, `kill_emulator`, and
  the pinned constants (`CLONE_AVD`, `GUEST_FRAME_RATE_HZ`).
- `bring_up.py` — one instance from cold or from a snapshot to a game at home,
  offline, with the bridge deployed: `bring_up`, `cold_bring_up`, `restore`,
  `save_snapshot`, `require_offline`, `require_game_activity`, `wait_until_ready`.
  `bridge_key`/`keyed_snapshot_name` pin a snapshot to the exact bridge build it
  was taken with.
- `frame_rate.py` — `raise_frame_rate` and `confirm_frame_rate`, which refuse an
  instance whose applied surface rate does not agree with both levers.
- `bridge.py` — deploying and cleaning up the instrumented bridge through
  `scripts/instrumented_bridge.sh`; `compatibility` reads the build's identity.
- `instrumented_bridge.py` — `InstrumentedBridgeClient` and the framed JSON
  protocol (version 2): handshake, compatibility, observation, command, advance,
  slot labels, the in-run availability arrays (`unlock_state`,
  `unlock_all_upgrades`), and the typed errors for every way it can fail. The one thing it
  takes from the domain is `LIVE_WIRE_NAMES`, the set of `Main` fields a v2
  state message owes; it validates their presence and passes the values through
  raw, and does not scale them.
- `instrumented_run_adapter.py` — `InstrumentedRunAdapter`, the `RunPort`
  implementation, plus `slot_labels()`, the once-per-session read of what the
  game calls each upgrade row, and `unlock_all_upgrades()`, the round-start
  write `UpgradeAvailability.ALL` is applied with. This is the join to
  `environment`.
- `fleet.py` — many instances: `stagger_bring_up`, `bring_up_fleet`,
  `prepare_pinned_snapshot`, `tear_down_instance`, `tear_down_fleet`.

**State.** A `CloneInstance` is frozen identity, not a live handle; the live
state is the emulator process and the socket inside an `InstrumentedBridgeClient`.
`stagger_bring_up` owns three parallel lists, one entry per instance: `gates`
(readiness, set when that instance's bring-up concludes), `begun` (set when it
starts) and `begun_at` (when it started). The `begun`/`begun_at` pair exists
because every actor's thread starts at fleet start, so the backstop has to be
timed from the predecessor's *own* launch rather than from fleet start — timing
it from fleet start let a 7-instance cold fleet expire instances 3–6's windows
before their predecessors had launched, and four emulators booted at once, the
exact defect the function exists to prevent.

**The one rule in `fleet.py`:** emulators must not boot at the same instant.
Four cold-booting together pushed host load to 10.71 and left the last unable to
reach home inside its timeout, while steady-state collection uses 563% of 3,200%
available CPU (`M1B-E028`) — the contention is entirely in the boot.
`stagger_bring_up` gates each bring-up on the previous one for actors that then
collect concurrently; `bring_up_fleet` is the same rule where the loop is
already sequential. Teardown is never conditional: a failure is reported after
the rest of the fleet is down, never instead of it.

## 4. `learning` — everything about learning

Owns policies, replay, actors, the learner and the training run. It knows the
environment and nothing that observes or drives it.

- `policies.py` — the `Policy` protocol and the scripted baselines
  (`RandomPolicy`, `CheapestFirstPolicy`, `WaitOnlyPolicy`).
- `network.py` — `TowerTrunk`, `DuelingHeads`, `StackedPolicyNetwork`.
- `backbone.py` — the `Backbone` protocol, `SequenceBatch`, `LearnMetrics`,
  `collate`, `acting_copy`.
- `stacked_dqn.py` — `StackedDqnBackbone`, the default backbone.
- `dreamer.py` — `DreamerBackbone` and `DreamerConfig`: DreamerV3 (world model,
  imagination actor-critic) behind the same `Backbone` protocol, chosen with
  `scripts/train.py --backbone dreamerv3` (`docs/solution.md` §9.4c).
- `dreamer_math.py` — DreamerV3's network-free parts: symlog, twohot, the
  masked categorical, λ-returns, the percentile return normaliser, LaProp.
- `value_learning.py` — n-step targets, the weighted sequence loss, TD errors.
- `replay.py` — `PrioritizedSequenceReplay` over `ReplaySequence`.
- `actor.py` — `Actor`, which plays one episode against one
  `InstrumentedRunEnvironment` and emits sequences plus an `EpisodeSummary`.
- `training.py` — `Learner`, `TrainingConfig`, `TrainingRun`,
  `TrainingProgressReport`, `episode_health`, `collection_windows`,
  `SelectionPeriod`, `NearGreedyPlateau`, and
  `KillBar`/`KillBarCheck`.
- `exploration.py` — `ExplorationSchedule` and `ape_x_floors`: what each actor
  explores at, at each point of the budget.
- `evaluator.py` — `evaluate`, exploration-free and replay-free, producing an
  `EvaluationReport`.
- `checkpoint.py` — `Checkpoint`, `CheckpointIdentity`, `save`/`load`, the
  checksum sidecar and `write_manifest`.

**Exploration.** Every actor's rate falls linearly from `epsilon_start` over
`--epsilon-anneal-decisions` and is held afterwards; what it falls *to* is the
actor's own floor. `ExplorationSchedule` owns those numbers - nothing else in
`learning` holds an exploration rate - and `TrainingRun` asks it per actor, once
per episode. The default, `uniform`, has no per-actor floors at all and anneals
every actor to `--epsilon-end`, which is what every run so far collected under.
`--exploration ladder` replaces that destination per actor, so `--epsilon-end`
is the uniform schedule's floor only and passing it with a ladder is refused.
The ladder is Ape-X's (Horgan et al. 2018): actor `i` of `N` anneals to
`0.4 ** (1 + 7 i / (N - 1))`,
so one fleet both searches - the top actors play build orders the greedy policy
would never reach - and reports, because the near-greedy actors at the bottom
still produce a collection curve that reads as the policy's own performance.
`CollectionWindow` carries that split: the pooled window, each actor's own mean
final wave, and a mean pooled over the near-greedy actors alone, which under a
uniform schedule is every actor and therefore the pooled series itself. In the
tracking store those are `collection_window_near_greedy_mean_final_wave` for the
episode-cut window, and `selection_period_near_greedy_mean_final_wave` with
`selection_period_best_near_greedy_mean_final_wave` for the decision-cut
selection periods the arm is chosen on and early stopping is judged on.

**Selection periods and the arm.** The decision axis is cut into selection
periods of `--selection-period-decisions` (default 15,000), independent of the
checkpoint cadence. At the episode that crosses a period's multiple
`TrainingRun` closes the period just ended: it takes the mean final wave of the
near-greedy actors' valid episodes that ended inside it — under a uniform
schedule that is every actor — records it as a `SelectionPeriod`, and writes a
numbered checkpoint there, so every period close is a file on disk. The
summary lists them as `selection_periods`. No code picks the arm: it is chosen
by hand from that list by the one rule in `docs/solution.md` 9.2b — the
checkpoint at the close of the period with the highest near-greedy mean, period
1 excluded, a period with no mean ineligible, ties to the earlier period. A
kill-bar stop does not change which checkpoint is the arm; the
pre-registration decides whether a killed run's arm is evaluated.

**Stopping early.** A run may end before its budget is spent. At each period
close the mean goes to `NearGreedyPlateau`, which keeps
the level the curve last really moved to and how many periods in a row have
failed to reach it plus `--early-stop-min-improvement` waves. That level moves
only on a period that clears the threshold, never on a mere new maximum: a
curve creeping up by less than the threshold would otherwise raise the bar it
is judged against by exactly what it gained, so a run gaining a tenth of a wave
a period would stop while one gaining nothing carried on. The first period sets the
baseline and cannot trigger a stop. When `--early-stop-patience-periods`
periods in a row have failed to improve, the run stops after that period's
checkpoint is written: the actors finish the episodes they are in and start no
more, `finished` is true, and the summary's `early_stopping` block records the
stop, the period it happened at, the best period mean and the closing period's.
The default patience of 0 is off, which is what every measured run so far
collected under. The three counters travel in the checkpoint's
`TrainingProgress` — optional fields within format 3 — so a resumed run is
judged on one curve rather than counting again from zero; a resume from a file
written before they existed starts the tracker fresh and says so, in the log
and in `early_stopping.tracker_restored_from_parent`.

A run may also stop on a `KillBar` (`--kill-bar`): at the first episode
boundary where the fleet's cumulative decisions reach the bar, `TrainingRun`
takes the near-greedy mean over the bar's decision window and stops if it is
below the bar's minimum. Each check is a `KillBarCheck` on the report
(`early_stopping.kill_bar_checks`); `stopped_early` covers both kinds of stop
and `killed_by` names the bar. `scripts/train.py` skips the final evaluation
after a kill-bar stop. A bar the parent run already passed is not rechecked
on resume.

**State.** `TrainingRun` owns everything the fleet shares: one
`PrioritizedSequenceReplay` all actors write into, one `Backbone` inside
`Learner`, one acting copy per actor in `acting`, the `TrainingProgressReport`,
the gradient debt `_owed`, and `_since_sync`. Three locks, each guarding one
thing:

- `Learner.lock` guards the training network. It is what keeps an optimisation
  step and a parameter publication from overlapping, so what an actor copies out
  is always some completed step and never half of one.
- `TrainingRun._lock` guards the progress report, the gradient debt and the
  hooks. An actor holds it between its episodes and never while collecting.
- `PrioritizedSequenceReplay.lock` guards the buffer, and is the one taken by
  the *caller* rather than inside the methods: `update_priorities` refuses
  indices an eviction has shifted, so the learner must hold it across `sample`,
  `learn` and `update_priorities` together.

`_since_sync` needs no lock: each actor touches only its own entry of a dict
whose keys are all present from construction.

## 5. `experiment` — observing a run

Owns run identity, metrics, tracking and statistics. It reaches for no adapter
and no script.

- `run_identity.py` — `RunIdentity`, `new_run_id`, `source_revision`,
  `resolved_config`, `checkpoint_identity`, and the reference floors
  (`SCRIPTED_REFERENCE`, `REFERENCE_FINAL_WAVES`).
- `tracking.py` — the `ExperimentTracker`/`TrackedRun` protocols and
  `NoExperimentTracker`. `mlflow_tracking.py` is the optional adapter; MLflow is
  an extra, imported lazily, so a checkout without it still runs.
  `add_tracking_arguments`/`tracked_run` are the `--mlflow-run`/`--run-dir`/
  `--experiment` options and the handle they open, shared by
  `select_checkpoint.py` and `report_arms.py` rather than each defining its
  own copy.
- `metrics.py` — learning-curve points, health counters, collection-window and
  decision-time lines.
- `training_report.py` — `TrainingReport`, which writes a run's artifacts.
- `comparison.py` — `iqm`, `stratified_bootstrap`,
  `stratified_bootstrap_difference`, `bootstrap_difference`, `cohens_d`.
- `wave_statistics.py` — per-wave equivalence analysis between two arms.

**State.** Run identity is immutable and is stamped into every checkpoint and
every record. Durable state is files under the run directory plus whatever the
tracker holds; nothing here is mutated in place.

## 5a. The two loose modules

Two modules sit at the top level of `tower_rl` rather than in a package, because
neither is part of a run:

- `doctor.py` — read-only host, package and Android-device diagnostics, returning
  `CheckResult` rows with a `pass`/`warn`/`fail` status: host and SDK inventory,
  Android tools, the XAPK when one is given, and the device when a serial is
  given. It reads the SDK through `simulation.android_sdk` and the archive
  through `xapk`, so it sits *above* `simulation`; that is why `experiment` is
  forbidden to import it. `scripts/doctor.py` is its one entry point — both the
  XAPK and the serial are optional there — and `tests/unit/test_doctor.py` is
  its other caller; nothing in a run imports it.
- `xapk.py` — metadata-only inspection of a locally supplied XAPK archive
  (manifest, splits, checksums), used by `doctor` and by `tests/unit/test_xapk.py`.
  It never extracts or copies proprietary bytes. The XAPK is reference material;
  the validated runtime is Play-installed.

## 6. Flow: a fleet collection run

`scripts/run_actors.py` measures throughput with scripted policies.

1. `main` builds N `CloneInstance` values and, unless `--cold`, calls
   `simulation.fleet.prepare_pinned_snapshot` once on a writable instance — a
   `-read-only` actor cannot save a snapshot.
2. `simulation.fleet.stagger_bring_up` wraps `collect_episodes` so instance
   *k*'s bring-up starts only once instance *k-1* has signalled ready. The
   signal fires in a `finally`, so a failed bring-up releases the next actor as
   surely as a successful one.
3. `run_fleet` submits one `run_actor` per instance to a thread pool. Each calls
   `collect_episodes`: `simulation.bring_up.bring_up`, then `require_offline`,
   `require_game_activity` and `raise_frame_rate` on its own instance — there is
   no fleet-wide rendezvous, so a ready instance starts collecting while its
   peers still boot (`M1B-E047`).
4. `collect_episodes` runs `scripts/run_episodes.py` as a subprocess against
   that instance's forwarded port. There, an `InstrumentedBridgeClient` feeds an
   `InstrumentedRunAdapter`, which is the `RunPort` an
   `InstrumentedRunEnvironment` drives; `learning.evaluator.evaluate` plays the
   episodes and writes one JSON record.
5. `run_actor` tears its instance down in a `finally` whatever happened, and
   `aggregate` sums throughput and health counters across the outcomes.

## 7. Flow: a training run

`scripts/train.py` trains one arm on the fleet. `BACKBONE` is the constant
`"stacked-dqn"`: one backbone, not a selectable arm.

1. `main` builds the tracker and reads the bridge's expected compatibility, then
   takes one of two paths:
   - `--actors 1` (the default) brings nothing up. It `connect`s to the instance
     the operator already has running, addressed by `--serial` and `--port`, and
     leaves it running afterwards — exactly as it worked before there were
     fleets.
   - `--actors N` calls `simulation.fleet.prepare_pinned_snapshot` first, because
     a `-read-only` fleet cannot save one, then
     `simulation.fleet.bring_up_fleet` over `open_instance`, which brings each
     instance up in sequence and connects it. Training starts only once the
     fleet is up, so the stagger needs no gate. Each instance is appended to
     `started` *before* its bring-up is attempted, so one that fails partway is
     still torn down. A failed bring-up is reported as a `bring_up_failure` and
     costs one actor, not the run.
2. `connect` opens one `InstrumentedBridgeClient` per instance and wraps it as
   `InstrumentedRunAdapter` → `InstrumentedRunEnvironment`, appending each to the
   list `tear_down_fleet` will release.
3. `build_arm` constructs the `PrioritizedSequenceReplay`, the
   `StackedDqnBackbone`, one `Actor` per instance and the `TrainingRun`, with
   the `RunIdentity` resolved and stamped.
4. `train_session` runs the arm until `--budget-decisions` is spent. The
   budget is **cumulative decisions across the fleet**, the one unit of
   training progress: learning happens per decision whatever the game's speed,
   and game time per decision moves with the policy (`M2-P005` diagnostic
   (c): 6.98 against 4.42 game-seconds per decision over matched decisions).
   It is accounted at episode granularity, so the run stops after the episode
   each actor crossed the budget in — past it by at most one episode per
   actor. Every decision-counted schedule reads the same counter: the replay
   ratio (`--gradient-steps-per-decision`), the exploration anneal
   (`--epsilon-anneal-decisions`), the importance exponent (annealed over the
   budget), the kill bars and the selection periods. The n-step anneal alone
   counts gradient steps. Game time is still measured and reported, as a
   statistic. Actors collect concurrently into the one buffer;
   the `Learner` takes gradient steps against the configured replay ratio;
   each actor refreshes its acting copy between its own episodes.
5. `arm.checkpoint` writes the checkpoint, then one pre-registered
   exploration-free evaluation runs on the final weights — after the budget, so
   it costs none of it and cannot be chosen after the fact.
6. The session report and the arm summary are written to the run directory, the
   tracked run is finished in a `finally`, and `tear_down_fleet` puts down every
   bridge and every emulator.

Every collected episode is reported to the tracked run as its own point, keyed
by the decisions spent when it ended — one monotone step axis every series in
the store shares — beside the learner's trailing summaries and the collection
windows that smooth them. Game time travels as a metric rather than as a
second axis: `episode_game_seconds_cumulative` per episode and
`learner_game_seconds` beside the learner's summaries.

`--resume <checkpoint>` makes the run a second segment of an earlier one:
`resume_point` reads the file into a `learning.checkpoint.ResumeState` before a
device is touched — refusing one whose `CheckpointIdentity` names another arm,
profile or schema, and one that has already spent `--budget-decisions`, which
stays the whole run's total — and `build_arm`
restores the weights and optimizer into the backbone, starts the
`TrainingProgressReport` at the parent's counters — decisions and game time
both — so epsilon, beta, the selection periods and the numbered-checkpoint
cadence are derived where a run that never stopped would have them. A
checkpoint before format 4 (`DECISION_BUDGET_FORMAT_VERSION`) is from the
game-time budget era: `load` still reads it for evaluation, but `resume_point`
refuses it by name. `build_arm` continues
the parent's tracked run through `open_run` when it had one, and names the
parent in `resolved_config.parent_checkpoint`; replay is not
persisted, so the buffer re-warms under the loaded policy before learning
restarts.

### The post-hoc selection path

Choosing the strongest checkpoint of a run and reporting it are separate from
training and from each other, because selecting on the episodes a model is then
reported on turns selection noise into a result. `train.py` leaves
`checkpoint-d<decisions, 7 digits>.pt` beside `latest.pt` at every selection
period close and, with `--checkpoint-every-decisions`, on every crossing of
that cadence too — one checkpoint per crossing episode, and no multiple
answered twice, even when one long episode carries the run past several — each
one a candidate that survives the next write. Runs from when the budget was
game time left `checkpoint-gs<game seconds>.pt`; readers take both, and none
parses the name: the decisions are read out of the file.
`run_actors.py --policy checkpoint:<path>` then plays a candidate as an ordinary
arm: `learning.policies.checkpoint_policy` rebuilds the backbone from the
checkpoint's own resolved config on the CPU, and the episodes go through the
same actor, evaluator and per-episode records the scripted and random floors go
through, with the arm's identity written into every actor record.

`scripts/select_checkpoint.py` is a checkpoint-evaluation tool, not the M2 arm
rule. It reads one evaluation directory per candidate — set A — and names the
highest interquartile mean of the final wave, with
`experiment.comparison.stratified_bootstrap` resampling within each actor. A
candidate is identified by the run id and identity hash its records carry, not
by its file name, because two runs at the same period leave identically named
files. What it chose is written to `<run>/selection.json`. Reporting the
selection is another `run_actors.py` run of that one checkpoint into an empty
directory, set B, which `scripts/report_arms.py` reads beside the floors —
given `--selection`, it refuses a set B whose records did not play the model
that was chosen: IQM with stratified intervals per arm, the pairwise difference
of those IQMs with its own stratified interval — the statistic M2-P001's
decision rule is written about, each arm resampled within its own actors — the
difference in means with Cohen's d beneath it as secondary, and the per-wave
comparison handed to `experiment.wave_statistics`. Neither script
starts an emulator, and neither decides a verdict. Given `--mlflow-run`, both
log their results onto the training run they are about, so the greedy curve
lands above the exploring one.

### The spectate path

`scripts/spectate.py` is the one path composed for a human. It brings up a
single instance through exactly the fleet's bring-up — canonical AVD refused,
`-read-only`, `-gpu host`, bridge deployed with its digest confirmed, offline
verified by interface, `tear_down_instance` in a `finally` — with two
differences, both of which are the point. The instance is launched **windowed**
(`emulator_command(..., windowed=True)`, the one caller that omits `-no-window`),
and it runs at 60 Hz, which is real time; the fleet's 120 Hz buys throughput,
which is worth nothing to somebody watching. It refuses to start while any
emulator is running at all, because a windowed real-time session must never
share a host with a measurement.

The seam it watches through is the decision stream.
`InstrumentedRunEnvironment.on_decision` is an optional observer called once per
`step` with a frozen `DecisionView`: episode and decision number, wave, cash,
health, the action as a domain description (`wait`, or `attack:3`), reward,
whether the episode just ended and why. `None` is the default and every
collecting and evaluating path leaves it there, so nothing is built per decision
unless somebody is watching. The panel is therefore a *view* of the same
decisions the episode records are built from, never a second account of them:
the episodes a session plays are written with the evaluator's own
`episode_record`, and the panel's own model — the pure functions that turn
accumulated views into the lines to draw — lives in the script beside the
`curses` rendering, because a terminal panel owns no domain concept. Optional
`--record` runs `adb shell screenrecord` on the guest in three-minute chunks
(the Android tool's own limit) and pulls them at the end; no training or
evaluation path reaches for it, which `tests/unit/test_spectate.py` holds.
