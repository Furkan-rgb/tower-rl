# Tower-RL — Environment Contract v1

This document defines the host-independent boundary between the real-game
environment and a policy. `docs/task.md` defines the required outcomes;
`docs/solution.md` defines the implementation approach. Android transport,
screen coordinates, and permanent-progression operations are outside this
contract.

## Versioning

- Observation schema: `observation-v1`
- Run-action schema: `run-action-v1`
- Every serialized observation and transition carries both versions.
- Incompatible changes require a new version and invalidate old replay.

## Observation

An `Observation` is evidence-bearing and fail-closed. It contains:

| Field | Meaning |
| --- | --- |
| `frame_id`, `captured_at_monotonic` | Device-local provenance and ordering. |
| `screen` | `battle_home_tier_1`, `tier_select`, `tier_1_active_run`, `tier_1_result`, `supported_modal`, or `unknown`. |
| `wave` | Visible wave number with confidence and source region. |
| `cash_normalized` | Spendable in-run currency normalized for the policy; raw text remains diagnostic-only. |
| `health_fraction` | Visible current/max health ratio. |
| `max_health_normalized` | Normalized maximum health when visible. |
| `upgrade_levels` | Per-action observed level readings. |
| `upgrade_costs_normalized` | Per-action current-price readings. |
| `action_mask` | Actions currently valid after freshness and UI checks. |
| `valid`, `invalid_reasons` | Admission decision for replay and environment stepping. |

Each numeric field is a `FieldReading`: value (or `None`), confidence in
`[0,1]`, source frame, region identifier, and an optional reason. Missing or
low-confidence values remain missing; they are never converted to zero.

An active-run observation is invalid unless wave, cash, and health are present.
The validator rejects negative currency, impossible health fractions, stale or
non-monotonic frame timestamps, backward wave movement within an active episode,
unsupported masked actions, schema mismatches, and any explicit invalid reason.
Invalid observations are retried, recovered, quarantined, or terminated; they do
not become an ordinary `WAIT` transition.

## Learned actions

The V1 policy can emit only these semantic `RunAction` values:

```text
WAIT
BUY_HEALTH
BUY_DAMAGE
BUY_ATTACK_SPEED
BUY_CRITICAL_CHANCE
BUY_CRITICAL_FACTOR
```

The action mask is authoritative for the current observation. Navigation (tab
selection, scrolling, opening menus, starting Tier 1, closing supported modals)
is controller-owned and never appears as a policy action. Workshop spending,
Lab research, milestone claims, purchases, advertisements, and other
permanent/meta actions are outside the V1 API.

## Action outcomes

Every requested action produces exactly one typed outcome:

`executed`, `unavailable`, `failed`, `ambiguous`, `navigation_failed`, or
`invalid_observation`.

An action is `executed` only after a fresh post-action observation confirms the
intended visible result. A failed or ambiguous action is not treated as a
successful purchase or as `WAIT`.

## Transition and reward

`StepResult` contains the previous observation, optional next observation,
semantic action, action mask, typed outcome, scalar reward, termination flags,
termination reason, and elapsed real seconds. No transition enters replay until
the next observation is valid.

V1 reward is aligned to the objective of maximizing final Tier-1 wave. The
environment may emit wave-progress and terminal survival reward according to the
versioned reward configuration; any shaping must be separately identified in
experiment metadata and never hide invalid or failed actions.

## Termination and recovery

Termination reasons are distinct: `tower_died`, `user_stop`, `safety_timeout`,
`invalid_observation`, `navigation_failure`, `device_failure`, and
`baseline_drift`. Normal reset follows the game's death-to-new-run path. Golden
snapshot restore is recovery only and must re-verify the fixed baseline before
the next episode.

## Implementation anchors

The canonical typed definitions live in
[`src/tower_rl/domain/contracts.py`](../src/tower_rl/domain/contracts.py), with
the Android port in [`src/tower_rl/ports/android.py`](../src/tower_rl/ports/android.py).
The application probe is in [`src/tower_rl/application/probe.py`](../src/tower_rl/application/probe.py)
and the ADB adapter is in
[`src/tower_rl/infrastructure/adb_probe.py`](../src/tower_rl/infrastructure/adb_probe.py).
Compatibility exports remain at the package root. None of these modules contain
a game clone, private API, or learned coordinate action.
