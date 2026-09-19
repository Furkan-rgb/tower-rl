# ADR 0006: Separate instrumented training from official evaluation

- **Status:** Accepted
- **Date:** 2026-09-15
- **Supersedes:** ADR 0003 for training throughput only
- **Superseded in part, 2026-09-19:** headed watch mode is `scripts/spectate.py`
  on the instrumented profile, not the official one (`#12`, `M2-E003`)

## Context

The visible controller proves that the real game can be automated, but natural
Tier-1 episodes take several minutes and decision-frequency screenshot/OCR work
limits actor density. Static and isolated live evidence in M1-E007 shows that the
real Unity IL2CPP runtime already owns the exact observations, semantic purchase
methods, and game-time control needed by an RL environment.

On the x86_64 workstation, generic Frida instrumentation does not cross Android's
ARM64 native-translation boundary reliably. A small custom ARM64 library does: it
loaded inside both a disposable re-signed XAPK and the unchanged Play-installed
package, resolved IL2CPP APIs, found `Main`, and read `Main.gameSpeed`. For the
official package, a reversible rooted bind-mount overlay changed only the loaded
`libunity.so` view. Package-manager identity, signed APK bytes, installer identity,
and app data remained unchanged.

The project needs training throughput without weakening the claim that evaluation
measures the real official game.

## Decision

Tower-RL has two explicit execution profiles:

1. `official-evaluation` uses the unchanged, unrooted Play-installed package at a
   validated normal in-game speed. It uses visible state and ordinary input, owns
   best-model promotion, and remains the authoritative behavioral result.
2. `instrumented-training` uses a private rooted clone of that package and a
   reversible local native-library overlay. A versioned ARM64 bridge reads exact
   IL2CPP-owned state and queues semantic game actions for execution on Unity's
   main thread. It may use a higher in-process `Time.timeScale` only after a
   normal-speed parity gate establishes equivalent scripted transitions and
   outcome distributions.

The bridge is an adapter to the real running game, not a simulator or mechanics
reimplementation. The game remains authoritative for costs, purchase validity,
randomness, combat, rewards, death, and reset. Policies continue to emit semantic
actions and never learn addresses, method offsets, or screen coordinates.

Every bridge handshake fails closed unless the package version and hashes, Unity
and metadata versions, bridge protocol/version, environment profile, supported
field/method inventory, and speed profile match an allowlisted compatibility
record. Instrumented transitions are never mixed with incompatible replay.

The following remain prohibited in both profiles:

- changing or redistributing signed APK/XAPK bytes or proprietary extracted data;
- hiding root or bypassing licensing, integrity, authentication, or entitlement;
- editing saves, cloud/account state, currencies, inventory, outcomes, or game
  mechanics;
- automating real-money purchases, advertisements, tournaments, leaderboards,
  competitive/events, or interaction with other players;
- calling game actions from arbitrary native threads;
- promoting a model from instrumented results alone.

Sparse screenshots remain a watchdog during training. If pixel and bridge
lifecycle observations disagree, the transition is invalid and the actor is
quarantined. Headed watch mode and exploration-free evaluation use the official
profile.

## Acceptance gates

Before instrumented experience can train the learner:

1. exact observations cover lifecycle, wave, cash, tower health, action
   availability, all supported upgrade costs/levels, death, and reset;
2. `WAIT` completes only after its bounded interval and a fresh exact
   observation; each supported semantic purchase is dispatched through Unity's
   main-thread message queue and has game-owned before/after confirmation;
3. deterministic normal-speed scripted episodes match the official visible
   controller and terminate/reset reliably;
4. protocol loss, version drift, stale state, thread-affinity failure, and pixel
   disagreement each fail closed;
5. the selected speed passes a documented equivalence and soak gate; and
6. official evaluation remains isolated from training replay and exploration.

## Consequences

- OCR is removed from the training decision loop after parity, reducing latency
  and invalid observations; it remains useful for independent watchdog/evaluation.
- Training can test both actor parallelism and faster game time while evaluation
  preserves an unchanged official reference.
- The native bridge and compatibility profiles become version-sensitive code that
  require focused tests and live revalidation after every game update.
- Rooted AVDs, overlays, extracted libraries, account/save state, logs, and bridge
  binaries remain private machine-local artifacts. Only original bridge source,
  build automation, sanitized schemas, tests, and evidence may enter Git.
- ADR 0003 still governs `official-evaluation`; its blanket normal-time restriction
  no longer governs the separately identified instrumented-training profile.
