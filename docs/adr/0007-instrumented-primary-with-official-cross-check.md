# ADR 0007: Instrumented clone is primary; official profile becomes a cross-check

- **Status:** Accepted
- **Date:** 2026-09-16
- **Amends:** ADR 0006 and the acceptance wording in `docs/task.md`

## Context

ADR 0006 kept the unchanged, unrooted, pixel-observed official instance as the
authoritative behavioral profile, with the instrumented clone confined to
training throughput. Live evidence in `M1B-E001` and `M1B-E002` has since shown
that the instrumented profile observes and controls the real game exactly: the
game itself owns prices, validity, randomness, combat, death, and reset, and the
bridge only reads its state and invokes its own methods.

The screenshot-and-OCR path remains the project's throughput and complexity
bottleneck. Decision-frequency OCR limits actor density, adds invalid
observations, and costs far more engineering than the exact path. The developer
owns this project privately and has decided that using the rooted clone rather
than the unrooted official instance is acceptable for the primary objective,
which is to train the agent.

## Decision

The private `instrumented-training` profile is the primary environment for
training **and** for routine evaluation. OCR leaves the decision loop entirely.

A small `official-evaluation` cross-check is retained. When a checkpoint is
promoted to `best`, a bounded number of exploration-free episodes are replayed on
the unchanged, unrooted official instance and reported alongside the instrumented
result. Its purpose is to detect the failure it is uniquely able to detect: that
instrumentation or an accelerated clock has distorted the game the agent is
being trained against.

A material disagreement between the two profiles invalidates the promotion and
quarantines the configuration rather than being averaged away.

Everything ADR 0006 forbids remains forbidden in both profiles: no modified or
redistributed signed bytes, no integrity or licensing bypass, no save, cloud, or
account mutation, no real-money purchases, advertisements, tournaments, or
interaction with other players, and no calling game actions from arbitrary
native threads.

## Consequences

- The agent's headline result is measured on the instrumented clone. Reports must
  name the profile, bridge version, and game speed, because that result is no
  longer automatically an unmodified-runtime result.
- The cross-check is cheap but not free, and it only runs at promotion.
- Speed remains gated on evidence. A speed is allowed only while final-wave
  distributions under a fixed policy stay comparable to normal speed; the
  measured game-time-per-wall-second ceiling of the host is a separate limit and
  is discovered, not assumed.
- `docs/task.md` keeps the objective, the fixed baseline, the semantic action
  space, and every reliability and evaluation gate. Only the clause naming the
  unrooted official instance as the sole source of primary experience is amended.
