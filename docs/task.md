# Tower-RL — Project Task

## 1. Mission

Build a complete local reinforcement-learning system that learns to play Tier 1 of **The Tower** by controlling genuine, unmodified Android instances of the official APK.

The system must support two primary user experiences:

1. **Training mode** — run multiple headless game instances in parallel, continuously improve a shared model, evaluate checkpoints, and retain the best validated model.
2. **Watch mode** — load the best saved model into one visible Android instance so the user can watch it play the real game and inspect its decisions and performance.

The project is successful only when it operates end to end without routine human intervention. A proof of concept that can tap the game, an isolated RL notebook, or a model trained against a recreated simulator does not satisfy the task.

## 2. Authoritative project objective

For a fixed permanent account state, train an agent to maximize its expected final wave in Tier 1 by choosing in-run upgrades and deciding when to wait.

All experience used for the primary agent must originate from actual gameplay in the official APK. The APK remains authoritative for combat, enemies, timing, upgrades, prices, randomness, and all other game mechanics.

## 3. Required end state

At completion, a user on the target workstation must be able to:

- configure the local APK and a fixed Tier-1 training baseline;
- validate that one Android instance is correctly observable and controllable;
- launch a configurable number of headless training actors;
- pause and safely resume training without losing important progress;
- see current training health and performance metrics;
- automatically produce latest, periodic, and statistically promoted best-model checkpoints;
- evaluate any compatible checkpoint over multiple real Tier-1 episodes;
- launch one visible game instance using the best model with exploration disabled;
- watch the agent start Tier 1, choose upgrades, survive until death, record the result, and optionally begin another episode;
- inspect enough logs and captured evidence to diagnose failures in observation, input, reset, actor health, evaluation, or learning;
- reproduce the setup and normal operations from project documentation.

The completed system must not depend on a human repeatedly starting runs, correcting navigation, buying upgrades, dismissing known screens, selecting checkpoints, or recovering actors during normal operation.

## 4. Target environment

The initial target is one user-owned workstation with:

- 128 GB system RAM;
- an NVIDIA RTX 4090;
- sufficient CPU capacity for multiple Android guests, with the optimal actor count to be established empirically;
- hardware virtualization enabled;
- local storage adequate for Android images, logs, replay data, evaluation records, and model checkpoints.

Do not assume a particular host operating system, CPU model, APK architecture, Android version, screen layout, available in-game speed, or maximum actor count until discovered and documented. If a host choice materially affects feasibility, record the evidence and make the requirement explicit before committing to the production path.

## 5. Scope

### 5.1 V1 gameplay scope

V1 is intentionally narrow:

- official APK only;
- Tier 1 only;
- one explicitly documented, fixed permanent account configuration;
- no permanent progression decisions;
- agent controls in-run upgrade purchases;
- agent may choose to wait;
- an episode begins from the defined Tier-1 start condition and ends when the tower dies or a documented safety limit terminates an invalid/stalled episode;
- primary optimization target is expected final wave;
- training uses multiple genuine Android game instances when the host supports them;
- evaluation and watch mode use genuine Android gameplay.

The initial action set must cover every in-run upgrade that is both available in the fixed account state and safely reachable through the supported game UI. Unsupported or intentionally excluded actions must be enumerated and justified.

### 5.2 Required operating modes

The product must expose stable entry points equivalent to:

```text
tower-rl doctor
tower-rl calibrate
tower-rl train
tower-rl evaluate
tower-rl watch
```

Names may change only if the replacement is equally clear and consistently documented.

- **Doctor** verifies host prerequisites, configured assets, device connectivity, model compatibility, storage, and other required dependencies without starting a long training run.
- **Calibrate** establishes or verifies the supported visual/UI profile and action targets for the configured game/device version.
- **Train** runs data collection and learning, evaluates candidates, checkpoints state, reports progress, and tolerates recoverable actor failures.
- **Evaluate** measures a selected model using exploration-free real-game episodes and records an immutable result tied to the exact checkpoint and environment profile.
- **Watch** visibly runs a selected model—best by default—against the real APK and presents useful decision telemetry without affecting training data or model state.

### 5.3 Required system capabilities

The finished project must include:

- Android instance lifecycle management;
- a canonical fixed account/game baseline and a documented way to verify it;
- reliable Tier-1 episode start, death detection, normal restart, and baseline recovery;
- screen capture and structured observation extraction;
- confidence/validity handling for extracted observations;
- semantic actions mapped to verified UI interactions;
- confirmation that attempted actions had the intended observable effect, where verification is possible;
- valid-action masking or equivalent prevention of known impossible purchases;
- an environment boundary that keeps Android/UI concerns out of the learning algorithm;
- a recurrent, replay-based distributed Q-learning agent suitable for partial observability and expensive real-environment samples;
- parallel actors with configurable exploration behavior;
- sequence-based prioritized replay;
- periodic evaluation separated from exploratory training;
- robust checkpointing for model, optimizer, schedules, counters, configuration, and other state needed for resume;
- best-model promotion based on aggregate evaluation rather than one lucky episode;
- metrics, logs, failure artifacts, and a machine-readable run manifest;
- automated tests and repeatable validation procedures;
- complete operator and developer documentation.

## 6. Explicit non-goals and boundaries

The following are outside V1:

- creating a clone, simulator, approximate physics model, or synthetic replacement for The Tower;
- training the primary policy on fabricated game transitions;
- modifying, patching, repackaging, or redistributing the APK;
- reverse-engineering private game implementation details;
- memory injection, anti-cheat circumvention, network interception, server emulation, or manipulation of online services;
- external time manipulation or speed hacks;
- automated advertisements, purchases, cloud-save manipulation, tournaments, leaderboards, competitive play, or use against other players;
- Workshop, Lab, Card, Module, Perk, event, tournament, or other permanent/meta-progression optimization;
- arbitrary free-form screen-coordinate actions learned by the agent;
- support for tiers other than Tier 1;
- mobile-device deployment or unattended remote/cloud deployment;
- guaranteeing a specific maximum wave, because attainable performance depends on the supplied account state and game version.

Use only the APK and account/save material the user is authorized to possess. Keep the experiment local and isolated from competitive or transactional features.

## 7. Core behavioral contract

### 7.1 Episode contract

Every valid training or evaluation episode must:

1. begin from the declared permanent account baseline;
2. start Tier 1 in a known, verified UI/game state;
3. allow the policy to control all supported in-run purchase decisions and waiting;
4. record observations, chosen actions, action masks, rewards, termination reason, model version, actor identity, and timing information;
5. terminate on confirmed tower death or a clearly classified invalid/stalled condition;
6. record final wave and required diagnostics;
7. return the actor to a verified next-episode state without permanent combat-affecting drift.

Invalid episodes must be excluded from model-quality evaluation and clearly distinguishable from genuine gameplay deaths. Their data may enter training only if the design in `solution.md` explicitly demonstrates that doing so is safe.

### 7.2 Baseline contract

The permanent state used by V1 must be frozen, versioned, and auditable. At minimum, the project must record all visible permanent choices known to affect a run, the game/app version, device profile, display settings, in-game speed, and relevant configuration.

Normal death-to-new-run navigation should preserve natural variation. A golden recovery state must restore actors that drift, become corrupted, enter an unsupported screen, or otherwise fail baseline verification. The implementation must not silently assume that repeated restoration produces suitable randomness; this must be tested.

Training must halt or quarantine affected actors if the baseline cannot be verified.

### 7.3 Observation contract

The policy must receive a normalized, versioned observation schema derived from the visible game interface and controller-owned history. The schema must contain enough information to learn meaningful Tier-1 purchasing behavior and must, at minimum, attempt to represent:

- wave/progression;
- spendable in-run currency;
- tower survivability information visible to the player;
- supported in-run upgrade levels and current prices;
- relevant UI/tab state;
- action availability;
- recent action/timing context needed to disambiguate the current state.

The exact fields, extraction techniques, normalization, temporal sampling, and any image features belong in `solution.md`.

Every observation must carry a validity result. Low-confidence, contradictory, stale, or impossible readings must trigger a documented retry, recovery, quarantine, or termination path rather than silently becoming normal training data.

### 7.4 Action contract

The learned action space must be semantic and versioned. It must contain `WAIT` plus supported `BUY_<UPGRADE>` actions. UI navigation and tap coordinates are adapter concerns and must not be learned outputs.

For each semantic purchase action, the system must know whether it is currently valid, execute the required navigation and interaction, and determine whether the action apparently succeeded. A failed purchase, navigation failure, ambiguous result, and deliberate wait must not be conflated in telemetry.

### 7.5 Reward and objective contract

The authoritative V1 objective is:

> Maximize expected final Tier-1 wave from the fixed permanent account state.

Reward design must stay aligned with this objective and avoid rewarding proxy behaviors merely because they are easy to measure. Any shaping beyond survival/wave progress must be justified in `solution.md`, separately reported in experiment metadata, and shown not to change the intended objective.

### 7.6 Model-selection contract

`best` must mean the strongest checkpoint under a repeatable multi-episode evaluation protocol, not the checkpoint associated with the single highest observed run.

Every promotion decision must retain:

- candidate and incumbent checkpoint identities;
- exact environment/baseline profile;
- number of valid and invalid evaluation episodes;
- final-wave distribution and summary statistics;
- promotion rule and outcome;
- random seeds or other reproducibility information where controllable;
- code/config version.

Mean final wave is the primary V1 ranking metric. Median, lower-tail performance, highest wave, episode duration, and invalid-run rate must also be reported. Ties and uncertainty must be handled conservatively according to a documented rule.

## 8. Quality attributes

### 8.1 Reliability

- No known silent corruption of observations, actions, episode boundaries, replay records, or checkpoints.
- Actor failures are isolated; one failed actor must not normally terminate every healthy actor or corrupt shared learning state.
- Long-running processes recover from anticipated transient failures or stop with an actionable error.
- Training state is written atomically or otherwise protected against partial writes.
- Resume behavior is verified, not assumed.

### 8.2 Reproducibility and traceability

- Every run has a unique identity and immutable resolved configuration.
- Models, replay/schema versions, baseline profiles, evaluations, and source revisions can be correlated.
- Defaults are explicit; important behavior must not depend on undocumented local state.
- The repository never commits the proprietary APK, account data, secrets, large runtime images, or generated training data unless the user explicitly arranges appropriate private storage.

### 8.3 Performance

- Actor count is configurable.
- The project benchmarks aggregate valid environment decisions and episodes per wall-clock hour across increasing actor counts.
- The chosen default actor count maximizes useful aggregate throughput on the target machine without unacceptable instability or starving the learner/evaluator.
- GPU, CPU, memory, storage, capture latency, extraction latency, inference latency, actor lag, invalid observations, and episode throughput are observable.
- No fixed actor-count promise may replace measurement.

### 8.4 Maintainability

- Game/UI integration, environment semantics, learning, evaluation, orchestration, configuration, and reporting have explicit boundaries.
- Versioned schemas fail loudly on incompatible changes.
- Configuration is validated at startup.
- Public commands provide useful help and actionable failures.
- Critical behavior is covered by automated tests; unavoidable APK-dependent tests are clearly separated from fast tests.

### 8.5 Usability

- A technically proficient user can set up, validate, train, resume, evaluate, and watch using the documentation alone.
- Progress output answers: Is it healthy? Is it learning? What is the current best? How fast is it collecting experience? Which actor is failing, if any?
- Watch mode visibly distinguishes selected action, failed action, invalid observation, recovery, and episode end.

## 9. Required project documentation

The orchestrator must maintain documentation as part of the implementation, not as a final afterthought.

### 9.1 `docs/task.md`

This specification, updated only when scope or acceptance criteria genuinely change. Preserve the separation between **what must be achieved** here and **how it will be achieved** in `solution.md`.

### 9.2 `docs/solution.md`

Before substantial implementation, produce the technical plan that explains how this task will be completed. It must resolve the architecture, host/Android strategy, component boundaries, schemas, extraction and action approach, baseline/reset behavior, RL design, process model, data flow, checkpoint/evaluation rules, configuration, observability, test strategy, staged rollout, and major technical tradeoffs.

Implementation-specific decisions, commands, dependency choices, directory structure, protocols, hyperparameters, algorithms, and failure-recovery mechanics belong there rather than in this task.

### 9.3 Other required documentation

Create and maintain, at minimum:

- `README.md` — purpose, status, quick start, and documentation map;
- `docs/setup.md` — prerequisites and complete local setup;
- `docs/operations.md` — calibrate, train, pause/resume, evaluate, watch, recover, and troubleshoot;
- `docs/environment-contract.md` — baseline, observation, action, reward, episode, and termination schemas;
- `docs/experiments.md` — benchmark and learning results, including failed approaches;
- `docs/limitations.md` — known fragility, unsupported screens/features, version compatibility, and safety boundaries.

Use architecture decision records when a consequential choice is expensive to reverse or needs durable rationale.

## 10. Milestones and exit gates

Milestones are outcome gates. The implementation sequence and technical method must be specified in `solution.md` and may evolve as evidence is collected.

### M0 — Feasibility and environment characterized

Exit criteria:

- target host, virtualization support, game APK compatibility, Android/device profile, and supported game version are recorded;
- the game launches in a controllable Android environment at a fixed resolution;
- Tier 1 can be started manually in that environment;
- the intended fixed permanent baseline is identified;
- material feasibility blockers and unsupported assumptions are documented;
- `solution.md` is approved as internally coherent and traces every requirement in this task to a planned component or validation.

### M1 — One fully controlled real-game actor

Exit criteria:

- one program-controlled APK instance can verify readiness, start Tier 1, capture observations, execute every supported semantic action, detect death, record the result, and start the next run;
- calibration detects incompatible UI/device profiles rather than tapping blindly;
- expected non-game screens and transient failures have classified handling;
- baseline drift is detectable and golden recovery is demonstrably functional;
- each supported purchase action has evidence that the intended control was activated;
- a scripted controller completes at least 100 consecutive valid episodes without manual correction during development testing.

### M2 — Environment reliability gate

Exit criteria:

- the automated soak test completes at least **1,000 consecutive episode attempts** without human intervention;
- at least **99%** of attempts are valid game episodes;
- no silent baseline drift, unclassified navigation state, corrupt episode boundary, or undetected invalid observation is observed;
- all invalid attempts are automatically classified and recovered or quarantined;
- recorded episode summaries agree with sampled visual evidence;
- training is not allowed to proceed if this gate is failing.

### M3 — End-to-end learning pipeline

Exit criteria:

- a baseline policy, replay path, learner, checkpoint writer, resume path, and evaluator operate end to end on genuine APK experience;
- replay entries can be traced back to actor, model version, observation/action schema, and episode;
- interrupted training resumes without losing or corrupting the last confirmed state;
- automated tests demonstrate correct masking, sequence boundaries, terminal handling, checkpoint round trips, and evaluation isolation;
- the learning pipeline can run for a meaningful test duration without unbounded resource growth or operator intervention.

### M4 — Distributed recurrent training

Exit criteria:

- multiple actors collect genuine experience concurrently for one central learning process;
- the final V1 recurrent distributed Q-learning configuration is active;
- actor exploration and weight-version behavior are recorded;
- stale, failed, or lagging actors cannot silently contaminate evaluation;
- scale benchmarks cover at least 1, 4, 8, and higher feasible actor counts;
- the selected production actor count and bottleneck analysis are documented;
- a sustained overnight run completes without fatal orchestration failure, checkpoint corruption, runaway storage use, or manual actor recovery.

### M5 — Trustworthy model evaluation and promotion

Exit criteria:

- periodic evaluation runs separately from exploratory training;
- `latest` and `best` have unambiguous, tested semantics;
- candidate promotion uses the documented multi-episode rule;
- an interrupted or partially invalid evaluation cannot replace `best`;
- evaluation reports include the required distribution, validity, environment, and checkpoint metadata;
- at least one trained checkpoint is compared with documented random and simple scripted baselines under the same fixed account state;
- the promoted model shows a reproducible improvement over the random baseline and its result is not based on a single episode.

No fixed wave threshold is required for V1 acceptance; genuine repeatable learning relative to a baseline is required.

### M6 — Complete operator experience

Exit criteria:

- all required commands are implemented and documented;
- `doctor` detects common setup errors before a long run;
- `train` presents health, throughput, learner, replay, evaluation, and best-model status;
- training supports graceful stop and tested resume;
- `evaluate` produces a durable report for a selected checkpoint;
- `watch` defaults to the promoted best checkpoint, disables exploration and learning, controls a visible official APK instance, and displays useful live decisions/telemetry;
- watch-mode episodes do not enter replay or mutate the selected model;
- setup and operation are reproducible from a clean documented environment.

### M7 — Final validation and handoff

Exit criteria:

- all Definition of Done items below pass;
- all automated checks pass from the documented command;
- the end-to-end acceptance run is captured in a final report;
- known limitations and unresolved risks are explicit;
- no placeholder, mock, stub, or manual step remains on a required production path;
- repository status contains only intentional project changes and excludes prohibited/generated assets;
- the project can be handed to another orchestrator or engineer without relying on undocumented conversation context.

## 11. Definition of Done

The project is done only when all of the following are true:

- The official APK is the sole source of primary gameplay experience.
- Tier 1 is played from a versioned fixed permanent account state.
- The 1,000-attempt reliability gate passes with at least 99% valid episodes and no silent corruption.
- All supported in-run actions and `WAIT` are observable, executable, and tested.
- Invalid observations, action failures, navigation errors, deaths, stalls, and baseline drift are separately classified.
- Parallel training runs against multiple real APK instances.
- The implemented learner is recurrent, off-policy, replay-based, and distributed across actors as defined by the accepted `solution.md`.
- Training can stop gracefully and resume from a consistent checkpoint.
- Evaluation is exploration-free, multi-episode, isolated from training, and reproducibly tied to a model/environment version.
- `best` is promoted through the evaluation rule and cannot be overwritten by an incomplete or lucky single run.
- A trained model demonstrates repeatable improvement over the random baseline under the same evaluation conditions.
- Watch mode loads `best` by default and visibly plays the actual APK without learning or adding replay data.
- A sustained overnight training run completes without manual intervention or corrupt artifacts.
- The system exposes sufficient metrics and diagnostics to understand throughput, failures, resource use, and model performance.
- Fast automated tests and APK-dependent integration/soak tests are clearly separated and documented.
- Setup, operation, recovery, evaluation, limitations, and the environment contract are complete and accurate.
- The repository excludes the APK, user account/save data, secrets, Android runtime images, bulk replay data, and generated model artifacts unless explicitly stored in an appropriate private artifact system.
- All required paths are real implementations; no acceptance criterion depends on a TODO, mock, or undocumented manual workaround.

## 12. Orchestrator mandate

The orchestrator owns completion of the entire task, including discovery, planning, implementation, validation, documentation, and final handoff.

The orchestrator must:

- read this task first and treat it as the authoritative scope;
- inspect the actual repository and target environment before assuming their state;
- write or update `docs/solution.md` before substantial implementation;
- maintain traceability from requirements to implementation and tests;
- break work into bounded phases with explicit validation at every gate;
- prefer evidence from the running system over assumptions about the APK, UI, emulator, hardware, or game behavior;
- keep the system runnable throughout development;
- use deterministic scripted policies before interpreting RL behavior;
- protect long-running work with resumable state and bounded storage;
- record failed experiments and contrary evidence, not only successful results;
- finish required integration and documentation rather than stopping after individual components work;
- make reasonable reversible technical decisions autonomously;
- surface a blocker only when it requires missing user-supplied material, access, hardware action, or a genuinely product-defining choice.

The orchestrator must not claim completion based solely on unit tests, mocked Android adapters, a single successful episode, a single high-wave run, a running learner with no demonstrated improvement, or watch-mode playback that still needs manual correction.

## 13. User-supplied prerequisites and legitimate blockers

The following may require user action and should be requested only when actually needed:

- the legitimately obtained APK or a path to it;
- confirmation of the target host OS and CPU if not discoverable;
- enabling hardware virtualization or installing host-level drivers when the agent lacks permission;
- establishing the desired dedicated fixed account state in the game;
- completing any unavoidable first-run or authentication interaction;
- confirming which visible upgrades/features are available in that baseline;
- providing substantial disk capacity or private artifact storage if local capacity is inadequate.

When blocked, report:

1. the exact failed acceptance gate;
2. evidence of the blocker;
3. what was already completed;
4. the smallest user action or decision needed;
5. the precise command or verification that will resume work.

Do not reinterpret a missing prerequisite as permission to replace the actual APK with a simulator or silently narrow the end goal.

## 14. Deferred roadmap

These are potential later projects and must not delay V1:

- optimize coins per hour or multi-objective performance;
- introduce image/playfield features when structured visible state is insufficient;
- optimize permanent Workshop progression with a separate meta-policy;
- add Cards, Labs, Modules, Perks, and other progression systems;
- support additional tiers or account baselines;
- compare alternative RL families after the V1 benchmark exists;
- investigate authorized acceleration methods beyond normal in-game speed;
- support additional hardware hosts or remote actor fleets.

Each deferred item requires its own objective, environment contract, acceptance criteria, and evaluation protocol before implementation.
