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

An observation is a frozen `RunState` (`environment/run_state.py`), built from
one exact bridge reading by `RunStateBuilder`. There is no visual profile and no
screen classifier: the only source is the instrumented bridge, and no part of
the decision loop reads a pixel.

| Field | Meaning |
| --- | --- |
| `source_sequence`, `captured_at_monotonic` | Provenance and strict ordering; both must advance between states of one episode. |
| `profile_id` | The exact game/library plus bridge/protocol/speed identity this reading belongs to. |
| `lifecycle` | `active` or `terminal`. `terminal` is what `RunState.terminal` reports. |
| `wave`, `wave_log` | The wave number, raw and log-scaled for the encoder. |
| `cash_log` | Spendable in-run currency, log-scaled. `observation-v1` carries cash only in this form. |
| `health_fraction` | Current/max health ratio. |
| `max_health_log` | Log-scaled maximum health. |
| `game_speed` | The world's speed multiplier as the game reports it. |
| `rows` | One `UpgradeRow` per upgrade action: `cost_log`, `affordability` (clipped at `MAX_AFFORDABILITY_RATIO`), `level`, `max_level`, `headroom`, `unlocked`, `maxed`, `available`. |
| `action_mask` | One flag per entry of `RUN_ACTIONS`, in that order; authoritative for this observation. |
| `valid`, `invalid_reasons` | Admission decision for replay and environment stepping. |
| `schema_version` | `observation-v1`. |

Readings are exact rather than confidence-weighted: the bridge reports a
game-owned field or it reports nothing. A missing value is never converted to
zero — the state is invalid instead.

`validate_transition(previous, current)` rejects a non-advancing source sequence
or capture time, a profile identity or schema version that changed inside an
episode, backward wave movement while both states are active, and any upgrade
level that moved backwards. Invalid observations are retried, recovered,
quarantined, or terminated; they do not become an ordinary `WAIT` transition.

## Run learned actions

The V1 policy can emit only `WAIT` plus `BUY_<UPGRADE>` values from the
versioned, evidence-backed M1 action inventory. The inventory includes every
safely reachable earned-currency in-run upgrade in the supported fixed baseline,
including Utility, and records each discovered action as `supported`, `excluded`,
`unavailable`, or `unsafe`. Only `supported` entries are serialized in
`run-action-v1`; an exclusion cannot silently remove an action from inventory.

The action space is a slot grid rather than a named inventory
(`environment/run_actions.py`): `RUN_ACTIONS` is `WAIT` at index 0 followed by
one `RunActionId(family, slot)` for each of the three `UpgradeFamily` values
(`attack`, `defense`, `utility`) across `SLOTS_PER_FAMILY = 20` slots — 61
indices in all, spelled `attack:3`, `utility:0`, and so on. Twenty slots per
family is what the supported 29.0.3 baseline reports and is the same width the
bridge reads availability over (`kMaskSlotsPerFamily`); a build reporting a
different count is a different action schema and fails closed rather than
silently renumbering.

The action mask is authoritative for the current observation. Navigation (tab
selection, scrolling, opening menus, starting Tier 1, closing supported modals)
is controller-owned and never appears as a policy action. Workshop spending, Lab
research, milestone claims, purchases, advertisements, and other permanent/meta
actions are outside the V1 API and this run contract.

## Action outcomes

Every requested action produces exactly one typed outcome (`ActionOutcome` in
`environment/episode.py`):

`executed`, `unavailable`, `failed`, `ambiguous`, `waited`, or
`invalid_observation`.

An action is `executed` only after a fresh post-action observation confirms the
intended result: game-owned before/after field confirmation from a
Unity-main-thread purchase plus a valid watchdog state. `WAIT` produces `waited`
and requires its bounded advance and a strictly newer valid observation. A
failed or ambiguous action is not treated as a successful purchase or as `WAIT`.

## Decision cadence

Two cadences are in play and they are not the same thing. The *world's* cadence
is `CadenceConfig`: the bridge advances frames until the run ends, the wave
turns, an upgrade becomes newly affordable, health moves by
`health_change_fraction`, or the quiet budget `max_quiet_game_ms` is spent. The
*decision* cadence says which of those stops the policy is asked about.

**The contract is choice points.** A decision is asked for only at a state whose
legal set contains at least one purchase (`RunState.is_choice_point`, read off
the action mask so legality and decision-worthiness cannot drift apart). A stop
whose only legal action is `WAIT` is not a decision — the policy has exactly one
answer available — so the environment takes that answer itself and advances
again, inside `InstrumentedRunEnvironment.step` and inside the reset that
produces an episode's first observation. Reward and game time accrue to the
surrounding decision: the transition covers the whole span. Every advance in the
span is an ordinary advance, checked against the host predicate and charged to
the wave and episode tallies exactly as a decided one is; only the decision is
withheld. 68% of run 1's decisions were such forced slices (`M2-E004`); see
[ADR 0009](adr/0009-decisions-at-choice-points.md).

A run that ends before it offers any purchase is the world ending, not the
pipeline breaking: the reset returns the terminal state and the episode is an
ordinary valid episode that took no decision and submits nothing to replay.
Only a port that produces no state at all fails the reset.

**`every-slice` is the legacy mode.** `DecisionCadence.EVERY_SLICE`, selected by
`--decision-cadence every-slice`, asks at every cadence stop, which is what run
1 collected under. It exists to reproduce run 1's protocol and to replay its
checkpoints, and for nothing else. The cadence is recorded in `resolved_config`,
in evaluation and session records, and in `CheckpointIdentity`, which refuses a
cross-cadence resume or evaluation by name; a checkpoint whose identity names no
cadence is read as `every-slice`.

## Transition and reward

`RunTransition` contains the previous observation, optional next observation,
semantic action, action mask, typed outcome, scalar reward, `terminated` and
`truncated` flags, termination reason, the `DecisionEvent`s that ended the
advance, elapsed real seconds, the game time the advance *requested*
(`requested_game_ms` — this build exposes no live in-run clock, so the honest
record is the request, `M1B-E003`), invalid reasons, and
`reward_schema_version`. Its `admissible` property is what gates replay: both
states valid, a next state present, and no invalid reason.

A transition covers the span between two decisions, which under choice points
may be several advances. `advances` says how many the world was actually given
— zero for a decision that moved nothing, such as a purchase whose settled state
was already a choice point — and `game_ms` is the measured round-clock game time
across them; `events` names
every cadence condition the span met, each once, in the order they were first
met. The reward is the wave progress across the whole span, so a decision held
through two wave boundaries while nothing was affordable is paid for both. A run
that ends inside a span terminates that transition, with the reward the span
earned before the end.

The episode record additionally carries `waves`: one row per wave index the
episode entered, each holding the wave number, whether the episode went on past
it (`completed`, false only for the wave it ended in), the measured game time it
took on the game's own round clock, the decisions taken and the advances made
while it was current, and the health fraction and log-scaled cash the run held
when it began. Decisions are choice points and advances are cadence slices; both
are recorded because they are the same number only under `every-slice`. An
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

Termination reasons are distinct (`TerminationOutcome`): `game_over`,
`operator_stop`, `max_episode_duration`, `observation_invalid`,
`action_pipeline_failed`, `ui_state_lost`, `device_failed`, `baseline_drift`,
and `recovery_failed`.

`VALID_TERMINATIONS` is `{game_over}` alone: only a genuine death is a complete
episode, and everything else is an environment or infrastructure failure
excluded from model-quality measurement. Every summary also carries
`termination_detail`, because an outcome without its reason cannot be diagnosed
later. Normal reset follows the game's death-to-new-run path. Golden snapshot
restore is recovery only and must re-verify the fixed baseline before the next
episode.

## Fidelity

An episode measured in a world that did not run at 1x is not comparable with one
that did, so the environment checks the game's own round clock against the game
time its advances budgeted and fails the episode by name rather than counting it
(`environment/run_environment.py`):

- `MAX_ROUND_CLOCK_RATIO = 1.25` and `GAME_TIME_INFLATED` — the world simulated
  more time than was asked for. Six known-good episodes measured 1.069 of round
  clock per budgeted millisecond; the same six at the account's 1.5x speed
  ceiling measured 1.625 (`M1B-E023`). The ceiling sits between them.
- `MIN_ROUND_CLOCK_RATIO = 0.99` and `GAME_TIME_DEFLATED` — the world simulated
  less. Healthy runs pooled 1.007–1.014; a 150 ms-step arm that under-credited
  simulated time measured 0.987. 1.0 would be the natural floor but leaves no
  room for float noise.
- `MIN_RATIO_EVIDENCE_GAME_MS = 2000.0` — the ratio is taken over the episode so
  far and only once that much game time has been spent; one advance is too short
  a window to judge a clock by. The advance that ends a run is exempt from the
  lower bound explicitly, because its round time legitimately reads zero.
- `ADVANCE_TRUNCATED_BY_WALL` — an advance ended because the bridge ran out of
  wall time rather than because the world did anything. How long the host took
  to render is not part of the decision problem, so such an episode was measured
  under a different problem and is failed, not counted (`M1B-E032`).
- `BRIDGE_EVENT_DIVERGENCE` — the bridge and the host disagree about which
  decision condition fired. The host's `_events_between` stays the only
  definition of what a decision condition is; the transition is recorded invalid
  rather than the host predicate being relaxed to match.
- `DEATH_BOUNDARY_TRANSIENT` — the one inconsistency the bridge may legitimately
  show. Health and the round flag are read separately, so at the instant of
  death health goes negative a moment before game-over flips (`M1B-E008`). It is
  recovered by advancing the world minimally (`MIN_ADVANCE_GAME_MS`), never by
  reading again, and counted as `recovered_transients`.

`advances_cut_short` counts advances the bridge stopped mid-loop on a reading
its settled snapshot then did not corroborate. It is benign — the settled state
is what the agent observes — but counted, because a rise in it says the loop and
the state it reports are drifting apart.

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
[`src/tower_rl/simulation/instrumented_run_adapter.py`](../src/tower_rl/simulation/instrumented_run_adapter.py).
None of these modules contain a game clone, private API, or learned coordinate
action.

M8–M11 meta definitions must live in a separate progression contract module and
must not be added as variants to the run types above.
