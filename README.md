# Tower-RL

Tower-RL is a local reinforcement-learning project that trains an agent to play Tier 1 of **The Tower** by controlling genuine, unmodified Android instances of the official game package.

The target system has two primary modes:

- **Train:** multiple headless Android actors collect real gameplay for a central recurrent Q-learning agent.
- **Watch:** the best evaluated model controls one visible Android instance with learning and exploration disabled.

## Project status

**M1 actor implementation — live control slice validated; 100-episode gate in progress.**

The supplied XAPK has been characterized as The Tower 29.0.1, an ARM64 split-APK
set requiring Android API 27 or newer. The validated runtime is the Play-installed
29.0.3 build on an API 36 Google Play ARM64 AVD. A pinned-renderer golden Tier-1
snapshot and bounded offline navigation probe are available; full gameplay
automation and RL remain milestone work.

The RTX workstation now has a validated x86_64 API 36 Google Play AVD using the
pinned Lavapipe/Swangle renderer. Its Play-installed 29.0.3 game baseline is
snapshot-restorable offline. The M1 actor starts Tier 1, extracts structured
observations, recognizes and confirms all supported purchases, handles
transient modals, reaches result, resets through result/home, and restores the
golden baseline. The 100-consecutive-episode M1 gate remains open.

In parallel, ADR 0006 separates a private `instrumented-training` profile from
the unchanged official evaluation profile. Its versioned native bridge now reads
exact game state and executes `WAIT` and earned-cash purchases on Unity's main
thread with game-owned confirmation, verified live against the real package in
`M1B-E001`. Evaluation and watch mode stay on the unchanged, unrooted, pixel-
observed official profile, and no instrumented transition may enter replay until
the remaining M1B parity and quarantine gates pass.

## Authoritative documentation

Future orchestrators and contributors must read these documents completely before implementing changes:

1. [docs/task.md](docs/task.md) — authoritative product scope, required outcomes, milestones, acceptance gates, and Definition of Done.
2. [docs/solution.md](docs/solution.md) — technical architecture and implementation strategy for satisfying the task.
3. [docs/architecture.md](docs/architecture.md) — concise component, interaction, runtime, and dependency view.
4. [docs/experiments.md](docs/experiments.md) — feasibility evidence, benchmarks, and failed experiments.
5. [docs/workstation-handoff.md](docs/workstation-handoff.md) — machine-local snapshot location and workstation reprovisioning procedure.

The task defines **what must be achieved**. The solution defines **how it will be achieved**. A technical discovery may justify updating the solution; it must not silently weaken the task.

Repository-specific agent instructions are in [AGENTS.md](AGENTS.md).

## Important boundaries

- The official APK is the sole authoritative gameplay environment.
- V1 covers Tier 1 with a fixed permanent account state.
- No Tower clone or synthetic gameplay environment.
- No APK modification, speed hacks, anti-cheat bypasses, purchases, ads, tournaments, or leaderboard automation.
- The XAPK/APK, account state, emulator images, screenshots, replay data, and trained models must never be committed.

## Intended commands

The completed V1 will expose commands equivalent to:

```text
tower-rl doctor
tower-rl calibrate
tower-rl train
tower-rl evaluate
tower-rl watch
```

Their required behavior is specified in the documentation. The initial `doctor`
and M0 `probe` slices exist today.

The first M0 `doctor` slice is now available:

```text
uv sync --all-groups
uv run tower-rl doctor \
  --xapk local/the-tower-29-0-1.xapk \
  --serial emulator-5554
```

`doctor`, `probe`, and the M1 actor/reliability runner are implemented at this
stage; training, evaluation, and watch remain later milestone deliverables.

Run the M1 gate only against the provisioned local AVD and keep its report
outside the repository:

```text
uv run python scripts/m1_reliability.py \
  --serial emulator-5554 \
  --snapshot tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation \
  --episodes 100 \
  --output /tmp/tower-rl-m1-100.json
```

For a new workstation, start with the repository preflight and AVD helpers in
[`scripts/`](scripts/), then follow
[`docs/workstation-handoff.md`](docs/workstation-handoff.md). These helpers stop
before Play sign-in and account-bearing snapshot creation.

The probe can validate the baseline and run the bounded no-upgrade navigation
smoke flow, restoring the canonical snapshot afterwards:

```text
uv run tower-rl probe \
  --serial emulator-5554 \
  --navigate \
  --restore-snapshot tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914
```

## Local XAPK

Place the reference XAPK at `local/the-tower-29-0-1.xapk` inside the cloned
workspace if metadata inspection is needed. The entire `local/` directory is
ignored by Git. The validated runtime is acquired through Google Play; the XAPK
is not installed by the workstation handoff procedure.

## License

No license has been selected. This repository does not grant rights to The Tower, its APK, assets, or other third-party material.
