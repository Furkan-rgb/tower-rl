# ADR 0008: Progression is a deterministic ladder between benchmark phases

- **Status:** Accepted
- **Date:** 2026-09-17
- **Refines:** ADR 0005 and the M8–M11 program in `docs/task.md`

## Context

The frozen V1 baseline is a small problem. The game offers six of its sixty in-run
upgrade slots, scripted play tops out around wave ten, and the account's own speed
ceiling is 1.5 because speed unlocks are themselves progression-gated. Training
several RL backbones against that baseline measures them on a narrow instance of
the real task.

Permanent progression would widen it: more upgrade slots become available, runs
get longer and more strategically interesting, and higher game speeds unlock
legitimately, which is also a throughput lever.

ADR 0005 anticipated progression as a policy-controlled capability with a
fail-closed allowlist and an evaluation gate. That machinery exists for a learned
meta policy. It is not required for the narrower goal of widening the run problem,
and a learned policy is the wrong instrument for irreversible decisions anyway:
exploration on a permanent spend cannot be undone.

## Decision

Permanent progression is executed by a **deterministic, versioned spend ladder**
owned by the controller. It is ordinary scripted navigation over visible,
earned-resource progression, with a fixed and documented spend order. No policy,
learned or otherwise, chooses permanent spending in V1.

The ladder runs **only between benchmark phases, never during a training or
evaluation episode**. Each ladder step mints a new immutable progression profile
with its parent identity, capability inventory, and visible-state fingerprint, as
ADR 0005 already requires. A benchmark is always run wholly within one profile.

This constraint is what makes the idea safe for the benchmark rather than fatal
to it. Coins are earned by playing, so if progression advanced during training the
environment would become both non-stationary and **agent-dependent**: a stronger
backbone would earn more, unlock more, and then appear stronger for two unrelated
reasons. Comparisons across backbones, and reproducibility across reruns, would
both be lost. Freezing the profile during a phase removes that confound entirely.

The alternative of never progressing at all was rejected. It permanently caps the
problem at six available actions and a 1.5 speed ceiling, and it measures the
algorithms on a task narrower than the one the project is actually about.

Everything ADR 0005 forbids remains forbidden: no real-money or store purchase,
advertisement, credential, cloud or save operation, tournament, competitive or
event path, and no unknown or modally ambiguous capability. The ladder is
auditable precisely because it is fixed: what it will spend, and in what order,
is known before it runs.

## Consequences

- The run action space grows across profiles. The policy architecture in
  `solution.md` 9.3 already absorbs that: a shared per-entry scorer plus an
  over-provisioned identity embedding table means a newly available slot is a
  fine-tune with fresh exploration, not a retrain, and not a schema change while
  the slot count is unchanged.
- Benchmark results are reported per profile and are never compared across
  profiles. A wave twelve result under a richer ladder step is not an improvement
  over a wave eight result under the frozen baseline.
- The ladder is itself an experiment artifact: its exact spend order, the profile
  it produced, and the resulting capability inventory are recorded, so a profile
  can be reproduced on a fresh account rather than existing only as device state.
- A learned meta policy remains deferred. If it is ever revisited, ADR 0005's
  allowlist and evaluation gate still govern it; this ADR does not weaken them.
