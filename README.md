# Tower-RL

Tower-RL is a local reinforcement-learning project that trains an agent to play Tier 1 of **The Tower** by controlling genuine, unmodified Android instances of the official game package.

The target system has two primary modes:

- **Train:** multiple headless Android actors collect real gameplay for a central recurrent Q-learning agent.
- **Watch:** the best evaluated model controls one visible Android instance with learning and exploration disabled.

## Project status

**Phase M0 — repository initialization and environment reconnaissance.**

No gameplay automation or RL implementation is considered complete yet. The next technical step is to inspect the locally supplied XAPK, determine its package structure and ABI requirements, and prove that one compatible Android instance can launch it.

## Authoritative documentation

Future orchestrators and contributors must read these documents completely before implementing changes:

1. [docs/task.md](docs/task.md) — authoritative product scope, required outcomes, milestones, acceptance gates, and Definition of Done.
2. [docs/solution.md](docs/solution.md) — technical architecture and implementation strategy for satisfying the task.

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

These entry points do not exist yet. Their required behavior is specified in the documentation.

## Local XAPK

Keep the downloaded XAPK outside Git. The M0 reconnaissance process will inspect it locally, record only non-proprietary compatibility metadata, and determine the correct installation set for one test Android device.

## License

No license has been selected. This repository does not grant rights to The Tower, its APK, assets, or other third-party material.
