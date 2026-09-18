# Tower-RL

Tower-RL is a local reinforcement-learning project that trains an agent to play Tier 1 of **The Tower** by controlling genuine, unmodified Android instances of the official game package.

The target system has two primary modes:

- **Train:** multiple headless Android actors collect real gameplay for a central recurrent Q-learning agent.
- **Watch:** the best evaluated model controls one visible Android instance with learning and exploration disabled.

## Project status

**M1B complete: the environment is proven and training runs on a real fleet.**

The game is observed and controlled through its own runtime via a private,
versioned native bridge (ADR 0006/0007), not through screenshots — no part of
the decision loop reads a pixel, and no screenshot classifier remains in the
tree. The validated runtime is the Play-installed 29.0.3 build on an x86_64 API
36 Google Play AVD on the RTX workstation, driven offline from a clone AVD that
several `-read-only` instances share.

A fleet of actors collects concurrently, a `stacked-dqn` learner trains against
one prioritized sequence replay under a decision budget, and runs are
checkpointed and evaluated exploration-free. Multi-actor scaling, renderer
equivalence, game-time fidelity and frame-rate limits are all measured; every
claim above has a dated entry in [docs/experiments.md](docs/experiments.md).

Current state, what is decided and what is open:
[docs/workstation-handoff.md](docs/workstation-handoff.md) START HERE, and the
[project board](https://github.com/users/Furkan-rgb/projects/3).

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

## Commands

There is no `tower-rl` console script; the entry points are the scripts under
[`scripts/`](scripts/), each with its own `argparse` interface. The V1 command
surface described in `docs/solution.md` section 11 is a target, not the present.

```text
uv sync --all-groups
uv run ruff check . && uv run mypy && uv run pytest

# one instance up, offline, with the bridge deployed
TOWER_BRIDGE_BUILD_DIR=... uv run python scripts/clone_session.py up \
  --renderer host --cores 4

# a fleet of scripted actors, for throughput
TOWER_BRIDGE_BUILD_DIR=... uv run python scripts/run_actors.py \
  --actors 4 --episodes 20 --renderer host

# a training run on that fleet
TOWER_BRIDGE_BUILD_DIR=... uv run --extra tracking python scripts/train.py \
  --actors 4 --renderer host --budget-decisions 100000
```

For a new workstation, start with the preflight and AVD helpers in
[`scripts/`](scripts/), then follow
[`docs/workstation-handoff.md`](docs/workstation-handoff.md). These helpers stop
before Play sign-in and account-bearing snapshot creation.

## Local XAPK

Place the reference XAPK at `local/the-tower-29-0-1.xapk` inside the cloned
workspace if metadata inspection is needed. The entire `local/` directory is
ignored by Git. The validated runtime is acquired through Google Play; the XAPK
is not installed by the workstation handoff procedure.

## License

No license has been selected. This repository does not grant rights to The Tower, its APK, assets, or other third-party material.
