# ADR 0009: The environment asks for a decision only at choice points

- **Status:** Accepted
- **Date:** 2026-09-19
- **Refines:** the decision cadence in `docs/solution.md` 9.2c and the run
  contract's transition section

## Context

Run 1's decisions were mostly not decisions. The observation batch recorded for
the plasticity diagnostic (`M2-E004`, 800 states taken from the states run 1's
checkpoints actually acted on) shows that **68% of them had `WAIT` as the only
legal action**: no upgrade was affordable, so the mask offered exactly one
answer and the policy could only give it.

The cadence that produced them is sound on its own terms. `AdvanceUntilEvent`
stops the world when something actionable changes — the run ends, the wave
turns, an upgrade becomes affordable, health moves by the threshold fraction —
or when the quiet budget of 2000 ms of game time runs out. The backstop is what
produces most of the forced slices: in a long stretch where nothing is
affordable the world still stops every two seconds, and the environment still
asks.

Three costs follow. The n-step return is diluted: most of the transitions in a
window are no-ops whose only content is the game time they spent, so the
bootstrap looks through forced steps rather than through choices. Exploration is
spent where it cannot choose anything: an epsilon draw at a WAIT-only state
selects the action the mask already forced. And the decision counter that the
budget, the epsilon schedule and every published rate are expressed in counts
those forced slices, so "200,000 decisions" mostly names moments where nothing
was decided.

## Decision

**A decision is asked for only at a choice point.** A choice point is an
observation whose legal set contains at least one purchase — read straight off
the action mask, so what is legal and what is worth asking about cannot drift
apart (`RunState.is_choice_point`).

`InstrumentedRunEnvironment.step`, and the reset that produces an episode's
first observation, advance repeatedly until the settled observation is a choice
point or the run ends. Each internal advance is an ordinary advance: the same
`CadenceConfig`, the same bridge round trip, the same host-side `_events_between`
predicate and its `BRIDGE_EVENT_DIVERGENCE` check, the same wave and episode
tallies, the same fidelity bounds. **Only the decision is withheld.** The
transition that comes back covers the whole span: its reward is the wave
progress across it, `game_ms` is the measured round-clock time of it, `advances`
is how many advances it took, and `events` names every cadence condition the
span met.

The change lives in the environment. The bridge is untouched, `CadenceConfig` is
untouched, `DecisionEvent` is untouched, and the fake port is untouched: the
cadence of the *world* is not what was wrong.

The old behaviour stays selectable, by name, for one purpose — reproducing run
1's protocol. `InstrumentedRunEnvironment(decision_cadence=...)` takes
`DecisionCadence.CHOICE_POINTS` (the default) or `DecisionCadence.EVERY_SLICE`,
and every entry point exposes it as `--decision-cadence
{choice-points,every-slice}`. The cadence is recorded in `resolved_config`, in
every evaluation and session record, and in `CheckpointIdentity`, whose
`incompatibilities` refuses a cross-cadence resume or evaluation by name. A
checkpoint whose identity names no cadence is read as `every-slice`, because
every file written before this ADR is one.

## Consequences

- **The unit of "decision" changes.** A decision now means a choice. Decision
  counts, decisions per episode and decisions per wave are not comparable with
  run 1's, and neither is anything derived from them — the budget, the epsilon
  horizon, the replay ratio. `EpisodeSummary` therefore carries both units:
  `decisions` counts choice points and `advances` counts the slices played
  through, per episode and per wave, so a run collected under either cadence can
  be read in the other's terms.
- **An episode can now take no decision at all.** A run that dies before it
  ever offers a purchase is the world ending, not the pipeline breaking: the
  reset hands back the terminal state, and the episode is a valid, scored,
  zero-decision episode that submits nothing to replay. The evaluator counts it
  and its final wave; the collection loop counts it as an episode rather than a
  port failure. Because such an episode spends none of the decision budget,
  consecutive ones count toward the same streak that withdraws an actor on
  consecutive port failures, under their own name - an instance whose runs never
  reach a choice point leaves the fleet loudly instead of collecting forever.
- **The budget unit moves to game time in a follow-up.** Spending a budget in
  decisions was already a proxy for spending it in experience; with forced
  slices gone, a decision's game-time cost varies by an order of magnitude
  between one choice point and the next, so a decision budget no longer bounds a
  run's length. That change is deliberately not made here.
- **Run 1's checkpoints are not comparable and cannot be resumed** into a
  choice-point run. The identity refuses it rather than letting the two be
  averaged together, and `--decision-cadence every-slice` is how run 1 is
  reproduced or its checkpoints replayed.
- **Every baseline is re-measured.** The scripted, random and wait floors in
  `REFERENCE_FINAL_WAVES` were measured under `every-slice`. Final wave itself
  is a property of the game and should survive the change, but the decision
  density beside it does not, and the floors are re-measured rather than
  re-labelled.
- **`CheapestFirstPolicy` never holds at a choice point.** By construction it
  buys the cheapest affordable upgrade, and a choice point is exactly a state
  where something is affordable — so under this cadence the scripted baseline
  buys at every decision it is offered and never waits. That is a property of
  that baseline, not of the cadence: it is what "cheapest first" means when the
  environment stops asking at moments where nothing can be bought. It remains a
  legitimate floor; it is no longer a policy that exhibits waiting.
- **`WaitOnlyPolicy` becomes the pure hold.** Its every decision advances one
  slice and then, cash having only grown, lands on another choice point — so it
  still spans the run at the cadence's own granularity.
