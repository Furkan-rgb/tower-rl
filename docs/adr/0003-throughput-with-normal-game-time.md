# ADR 0003: Scale throughput without manipulating game time

- **Status:** Accepted for official evaluation; superseded by ADR 0006 for training
- **Date:** 2026-09-14

## Context

Real-game episodes are expensive. Faster rendering may reduce emulator overhead,
but render speed is not the same as game-time speed, and external time
manipulation would violate project boundaries and could invalidate game behavior.

## Decision

Use one verified normal in-game speed as part of the fixed official-evaluation
profile. Do not use host clock manipulation, APK modification, or speed hacks in
official evaluation or watch mode. ADR 0006 separately governs the private
instrumented-training profile and its parity-gated in-process Unity time scale.

Improve aggregate experience throughput through:

- VM and compatible graphics acceleration;
- headless training instances at a fixed, readable resolution;
- efficient capture, recognition, and batched inference;
- multiple isolated actors after the M2 reliability gate;
- empirical renderer and actor-count benchmarks on each host.

The selected production actor count maximizes stable valid decisions and episodes
per wall-clock hour while reserving resources for learning and evaluation.

## Consequences

- A faster renderer primarily prevents lag and enables actor density; it does not
  by itself claim faster game simulation.
- Renderer, resolution, game speed, and capture method are versioned compatibility
  fields.
- Every materially different execution profile must pass visual and reliability
  validation.
- The Apple M2 Pro host proves one-device behavior; the 28 GB/RTX 4090 host gets a
  separate scale benchmark and may select a different compatible renderer.
