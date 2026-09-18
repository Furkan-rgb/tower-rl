# Tower-RL — Run Environment Contract v1 and Meta Environment Contract v1

This document defines the host-independent boundary between the real-game
environment and a policy. `docs/task.md` defines the required outcomes;
`docs/solution.md` defines the implementation approach. The run and meta
contracts are intentionally separate: Android transport and screen coordinates
are outside both, and permanent-progression operations are never part of the run
contract.

## Run-contract versioning

- Observation schema: `observation-v1`
- Run-action schema: `run-action-v1`
- Every serialized observation and transition carries both versions.
- Incompatible changes require a new version and invalidate old replay.

## Observation

An `Observation` is evidence-bearing and fail-closed. It contains:

| Field | Meaning |
| --- | --- |
| `source_profile`, `source_sequence`, `captured_at_monotonic` | `official_visual` or `instrumented_bridge` provenance and strict ordering. |
| `compatibility_id` | Exact game/device plus visual profile, or game/library plus bridge/protocol/speed profile. |
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
`[0,1]`, source sequence, evidence identifier (visual region or allowlisted
IL2CPP field), and an optional reason. Exact bridge readings use confidence 1
only while their handshake, heartbeat, lifecycle, sequence, and pixel-watchdog
checks are valid. Missing or low-confidence values remain missing; they are never
converted to zero.

An active-run observation is invalid unless wave, cash, and health are present.
The validator rejects negative currency, impossible health fractions, stale or
non-monotonic source timestamps/sequences, backward wave movement within an active
episode, unsupported masked actions, schema/compatibility mismatches, bridge and
pixel lifecycle disagreement, and any explicit invalid reason.
Invalid observations are retried, recovered, quarantined, or terminated; they do
not become an ordinary `WAIT` transition.

## Run learned actions

The V1 policy can emit only `WAIT` plus `BUY_<UPGRADE>` values from the
versioned, evidence-backed M1 action inventory. The inventory includes every
safely reachable earned-currency in-run upgrade in the supported fixed baseline,
including Utility, and records each discovered action as `supported`, `excluded`,
`unavailable`, or `unsafe`. Only `supported` entries are serialized in
`run-action-v1`; an exclusion cannot silently remove an action from inventory.

The current action IDs are profile data rather than an exhaustive contract list.
For example:

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
is controller-owned and never appears as a policy action. Workshop spending, Lab
research, milestone claims, purchases, advertisements, and other permanent/meta
actions are outside the V1 API and this run contract.

## Action outcomes

Every requested action produces exactly one typed outcome:

`executed`, `unavailable`, `failed`, `ambiguous`, `navigation_failed`, or
`invalid_observation`.

An action is `executed` only after a fresh post-action observation confirms the
intended result. The official profile requires visible confirmation; the
instrumented profile requires game-owned before/after field confirmation from a
Unity-main-thread purchase plus a valid watchdog state. Instrumented `WAIT`
requires its bounded interval and a strictly newer valid observation. A failed
or ambiguous action is not treated as a successful purchase or as `WAIT`.

## Transition and reward

`StepResult` contains the previous observation, optional next observation,
semantic action, action mask, typed outcome, scalar reward, termination flags,
termination reason, and elapsed real seconds. No transition enters replay until
the next observation is valid.

The episode record additionally carries `waves`: one row per wave index the
episode entered, each holding the wave number, whether the episode went on past
it (`completed`, false only for the wave it ended in), the measured game time it
took on the game's own round clock, the decisions taken while it was current,
and the health fraction and log-scaled cash the run held when it began. An
advance that crosses a wave boundary is charged whole to the wave that was
current when it started: the bridge reports one round-clock delta per advance
and cannot say how it split, so the attribution is stated rather than guessed.
The rows partition the episode - their game time and decisions sum to the
episode's own - and exist so behavioural equivalence between two arms can be
judged per wave index rather than on a final wave, whose variance is dominated
by how many waves an episode survived.

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

## Meta environment contract (`meta-observation-v1`, `meta-action-v1`)

The M8–M11 progression bounded context has its own `MetaEnv` and
`MetaController`. It does not reuse, union, cast, or dispatch through
`RunAction`, `Observation`, run replay, or fixed-baseline evaluation records.
Every meta record carries its meta schema versions, exact progression-profile ID,
parent-profile ID where applicable, capability ID, calibration evidence ID, and
resolved configuration identity.

`MetaObservation` contains only visible progression state and controller-owned
history needed to classify an allowlisted capability: verified profile
fingerprint, visible earned-resource balances, available progression controls and
their risk/capability states, relevant timer/research state, claim state, screen
state, field confidence, and invalid reasons. Missing or ambiguous state is not a
default; it masks the capability and produces a classified failure or recovery.

`MetaAction` represents a strategic progression choice, such as a particular
allowlisted earned-resource spend or research schedule. It is admissible only
when its exact calibrated capability is allowlisted and its required progression
evaluation gate has passed. Safe deterministic reward/milestone claiming is not
a `MetaAction` only if `MetaController` has evidence that the claim is
non-strategic, has no selectable alternative, and confirms its visible outcome.

No meta capability may automate real-money/store purchase, advertisements,
credentials, cloud/save controls, tournaments, leaderboards, competitive/event
participation, bypasses, or unknown/new/modally ambiguous paths. An explicitly
calibrated and allowlisted capability may operate autonomously without per-action
human approval; all other capabilities remain masked.

A confirmed permanent change creates a new immutable verified progression profile
linked to the prior profile. Recovery must verify and continue that profile, not
silently restore the parent. Timed research is permitted only in progression
mode. Fixed-baseline run training and evaluation require an idle/frozen profile;
all run episodes are tagged with their exact profile, and incompatible replay or
evaluation is rejected. Progression evaluation measures Tier-1 performance per
real elapsed time from compatible immutable profiles.

## Implementation anchors

The canonical run typed definitions live in
[`src/tower_rl/environment/run_state.py`](../src/tower_rl/environment/run_state.py)
(the observation schema and its validation) and
[`src/tower_rl/environment/run_actions.py`](../src/tower_rl/environment/run_actions.py)
(the action schema), with the episode and reward vocabulary in
[`src/tower_rl/environment/episode.py`](../src/tower_rl/environment/episode.py).
What a decision costs and what it may conclude is
[`src/tower_rl/environment/run_environment.py`](../src/tower_rl/environment/run_environment.py),
and the port it drives an instance through is
[`src/tower_rl/environment/run_port.py`](../src/tower_rl/environment/run_port.py);
the adapter behind that port is
[`src/tower_rl/infrastructure/instrumented_run_adapter.py`](../src/tower_rl/infrastructure/instrumented_run_adapter.py).
None of these modules contain a game clone, private API, or learned coordinate
action.

M8–M11 meta definitions must live in a separate progression contract module and
must not be added as variants to the run types above.
