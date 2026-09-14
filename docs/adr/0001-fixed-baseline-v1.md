# ADR 0001: Keep permanent progression outside V1

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

The game has two materially different decision timescales. During a Tier-1 run,
the player spends cash on temporary run upgrades. Between runs, persistent
currencies and elapsed time can change Workshop levels, Lab research, milestones,
Cards, Modules, and other permanent account state.

Combining both levels immediately would make the environment non-stationary,
introduce long-horizon and sometimes irreversible actions, and leave the project
without a precise evaluation baseline.

## Decision

V1 trains only the in-run Tier-1 policy from one versioned, fixed permanent
account baseline. The policy may choose `WAIT` and supported cash-funded run
upgrades. It cannot spend persistent currency, start research, claim milestones,
or optimize other permanent systems.

The architecture preserves a future extension seam for a separate `MetaEnv`,
`MetaAction` schema, and meta-controller. That future project must define its own
objective, safety constraints, and acceptance protocol before implementation.

## Consequences

- V1 evaluation is comparable across episodes and checkpoints.
- A Lab completing or another permanent combat change is baseline drift, not
  ordinary environment variation.
- Persistent balances may change only when shown to have no combat effect without
  a prohibited meta action; automation does not spend them.
- Full account progression is deliberately deferred rather than accidentally
  implemented through navigation code.
