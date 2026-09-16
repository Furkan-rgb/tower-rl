# ADR 0005: Stage meta progression behind separate contracts and capabilities

- **Status:** Accepted
- **Date:** 2026-09-14
- **Supersedes:** ADR 0001

## Context

Engineering V1 needs a fixed permanent baseline for comparable Tier-1 training
and evaluation. The expanded product also requires ordinary earned-resource
progression, whose decisions are long-horizon, sometimes irreversible, and may
include timed research or deterministic claims.

## Decision

Preserve M0–M7 unchanged as fixed-baseline V1. Deliver progression only in
M8–M11 through separately versioned `MetaObservation`, `MetaAction`, `MetaEnv`,
and `MetaController`; never union their action space with `RunAction`.

Calibration creates a fail-closed capability allowlist. Explicitly calibrated,
allowlisted capabilities may operate without per-action human approval. Unknown,
new, modal-ambiguous, transactional, credential, cloud/save, competitive/event,
or bypass capabilities remain masked. Safe deterministic non-strategic claims may
be controller-owned after verification; strategic irreversible choices remain
evaluation-gated `MetaAction` decisions.

Each successful permanent change produces a new immutable verified progression
profile. Recovery must not silently rewind it. Timed research is progression-only;
fixed-baseline evaluation uses an idle/frozen profile, and replay/evaluation is
isolated by exact profile identity.

## Consequences

- V1 evidence remains reportable and comparable after progression begins.
- Progression is measured by Tier-1 performance per real elapsed time, not V1's
  single-profile final-wave objective.
- Capability evidence, profile lineage, and evaluation gates become mandatory
  acceptance artifacts before autonomous strategic progression is enabled.
