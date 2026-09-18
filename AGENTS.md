# AGENTS.md

## Mandatory reading order

Before planning, editing code, or delegating work, read these files completely:

1. `docs/workstation-handoff.md` — the START HERE section at the top says where
   the project actually is and what is already decided
2. `docs/task.md`
3. `docs/solution.md`
4. `docs/architecture.md` — the packages that exist, what each owns, and the
   dependency rule between them
5. Any relevant ADRs under `docs/adr/`
6. The current implementation and tests for the affected area

Do not rely on conversation context as a substitute for the repository documents.

## Authority and document roles

- `docs/task.md` is authoritative for scope, outcomes, milestones, acceptance gates, constraints, and Definition of Done.
- `docs/solution.md` is authoritative for the current technical approach.
- `docs/architecture.md` describes the structure that exists; if it and the code disagree, the code wins and the document is fixed.
- ADRs explain consequential technical decisions and changes.
- Code and configuration must implement those documents; they do not silently redefine them.

Documents are not pinned. When code or measured evidence contradicts a document, the document changes — and every durable claim carries its evidence pointer (a `docs/experiments.md` entry) or goes. Changing `docs/task.md` requires an actual product-scope or acceptance decision and must not be done merely to make implementation easier.

If task and solution conflict, follow the task and reconcile the documents before continuing.

Task state is not prose. It lives on the GitHub board (see Board discipline below); do not add a to-do list, a priority list, or a "next steps" section to any document.

## Module rule

`src/tower_rl/` is four packages in one direction — `environment` is the root, `simulation` and `learning` each import only it and never each other, `experiment` imports `environment` and `learning`, and `scripts/` is the composition root — stated in full in [`docs/architecture.md`](docs/architecture.md) section 1 and enforced by `tests/unit/test_import_contracts.py`. To change a rule, change the constant there and say why beside it; a rule relaxed to make one import compile is the failure that test exists to expose.

## Execution rules

- Work through the milestone gates in `docs/task.md`; do not declare a phase complete without its evidence.
- Keep Android/UI integration separate from the environment and learner.
- Use semantic actions; never make learned screen coordinates part of the policy.
- Treat invalid observations, failed actions, navigation errors, game deaths, stalls, and baseline drift as distinct outcomes.
- Use the game's normal death-to-new-run path for ordinary resets; use the golden baseline for recovery.
- Establish random and scripted baselines before interpreting learning performance.
- Keep evaluation exploration-free and isolated from replay and training.
- Make long-running work resumable and storage-bounded.
- Record failed experiments and contrary evidence in `docs/experiments.md`.
- Keep documentation and the requirement-to-test traceability current as implementation evolves.
- Stop only for a genuine user-owned prerequisite, permission, host-level action, or product-defining choice.

## Public repository safety

This is a public repository. Never commit:

- XAPK, APK, APKS, AAB, OBB, extracted game files, or game assets;
- Android user data, device images, AVDs, snapshots, or account/save state;
- credentials, tokens, signing keys, cloud data, or local environment files;
- screenshots or logs containing account, purchase, advertising, or unrelated personal data;
- replay buffers, checkpoints, trained model files, generated runtime artifacts, or bulk diagnostics.

Before every commit, inspect the complete staged diff and confirm no ignored or sensitive file is force-added. Store only sanitized fixtures that are explicitly safe to publish.

## Validation discipline

Use fast unit/contract tests for normal iteration and clearly tagged opt-in tests for real Android instances. Never run account-affecting integration tests implicitly.

No completion claim may rest only on:

- mocked adapters;
- a successful APK launch;
- one automated episode;
- a running learner;
- a single high-wave result;
- unverified watch-mode playback.

The acceptance evidence must satisfy the exact gates and Definition of Done in `docs/task.md`.

## Completion protocol

At the end of each milestone:

1. run its required checks;
2. record evidence and failures;
3. update traceability;
4. update the solution or ADRs if the design changed;
5. confirm that no task requirement was weakened;
6. leave code, documentation, and runtime state recoverable;
7. report the next unmet gate or exact external blocker.

The final handoff must include supported versions, selected actor count, throughput, evaluation protocol, baseline-versus-best results, soak and overnight results, limitations, and exact operating commands.

## Board discipline

The GitHub project board (project #3, `tower-rl Task Board`) is the single source of truth for task state; every piece of work maps to a board issue, and `.claude/hooks/board-state.sh` surfaces its open items into every turn so this cannot silently drift. Commits that complete an issue use a `Closes #n` trailer; commits that advance one without finishing it use `Refs #n`. Workers cite the issue number they are working against in their reports.
