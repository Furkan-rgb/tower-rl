# ADR 0010: The agent observes everything the player sees

- **Status:** Accepted
- **Date:** 2026-09-19
- **Supersedes:** the `observation-v1` schema in `docs/environment-contract.md`
- **Board:** #41; evidence from #39

## Context

`observation-v1` showed the policy four run scalars — wave, cash, health
fraction, maximum health — and one row per upgrade slot carrying cost, a clipped
affordability ratio, a level *fraction* and three flags. That is a fraction of
what a person playing the same run reads off the screen.

`docs/solution.md` named the largest gap itself: the agent has no enemy, threat
or boss information a player's view carries, and it was left as the first
candidate observation extension. Two further losses were undocumented. `level`
and `max_level` reached the encoder only as `level/max_level`, so a level 3 of 5
upgrade and a level 30 of 50 one were the same state to the network even though
what they cost next is not. And `affordability` was clipped at ten, which erased
every distinction between "ten times over" and "a hundred times over" — the
difference between an early purchase and a late one.

Until board #39 none of this could be settled from the repository: no IL2CPP
dump exists here, the bridge resolves fields by name, and the only inventory
mechanism was a diagnostics build. #39 ran that build on a device and produced
two things this decision rests on. `FIELD-ENUMERATION.md` lists all 920 fields of
the game's `Main` class by name and declared IL2CPP type, of which 587 are plain
scalars readable through the path the bridge already uses. `FIELD-SAMPLES.md`
records 138 snapshots of 47 named fields over one episode, which is what turned
plausible names into observed values and observed units: `criticalChance` reads
in percent, not in [0, 1]; `closestEnemyDistance` stores a 10000 sentinel for
"no enemy" rather than an absence; `towerHealth` goes negative on the killing
blow; `waveTimer` counts against `waveLengthSeconds + waveCooldownSeconds`. The
same capture found `upgradeName`/`upgradeDescription` and their per-family twins
— the labels that say what slot 7 actually is.

## Decision

`observation-v2` shows the policy what the player sees, and **replaces** v1.
There is no dual support in the production path: v1 checkpoints are already
incompatible by cadence (ADR 0009), and `CheckpointIdentity.observation_schema`
refuses one by name.

Three parts.

**1. The live readings.** Thirty-seven further `Main` fields — the tower's
combat stats, the wave and the threat in it, the economy, and the round clock —
are read every snapshot and carried on the wire under the game's own field
names. The full table, with each field's observed unit and its transform, is in
`docs/environment-contract.md`; the single declaration it is written from is
`LIVE_FIELDS` in `environment/run_state.py`, and the bridge's `kLiveFieldNames`
is the same list in C++.

The rule is **rescale, never summarise**: magnitudes through `log1p`, percents
divided by 100, small counts raw, booleans as 0/1, the nearest-enemy distance
raw with the sentinel translated into an `enemy_present` flag of zero. Every
transform is monotone over the range the game produces, so nothing a player can
read is collapsed away, and each one fixes a range invariant. A reading outside
it is zeroed *and* named `OBSERVATION_OUT_OF_RANGE:<field>` in
`invalid_reasons`, which makes the transition inadmissible and puts an
attributable anomaly in the episode record. That is the M1B-E017 discipline
applied at schema scale: the failure that cost a whole run was a field read at
the wrong width that returned a plausible 0.0, and thirty-seven new fields is
thirty-seven new chances to repeat it.

**2. The upgrade rows.** Raw `level` and `max_level` join the derived fraction,
and `affordability` becomes unclipped `log1p(cash/cost)`.

**3. The labels, for humans only.** `upgradeName` and its siblings are read once
per session by a `slot_labels` bridge command — they are constant for a build —
and appear in the spectate panel and in the `upgrade_rows` block of evaluation
and session records. They are deliberately **not** in the observation tensor.
The policy addresses a slot by its stable index, which is what keeps numbering
stable across progression and game updates; a renamed row must not change what a
checkpoint means.

The protocol version goes to 2. A field the class does not carry, or one
declared at a type the bridge cannot read at its own width, is a named
initialization failure rather than a zero, and a state message that does not
carry exactly the declared set is refused by the host.

## What stays out, and why

- **Boss health and per-enemy state.** `bossHealthBar` and the `Enemy[]` pool
  are object graphs, not scalars on `Main`; reading them means walking managed
  objects the diagnostics build cannot even enumerate today. The boss *flags*
  and the wave's base health and damage are in, which is the threat information
  a decision at a choice point actually turns on. A per-enemy observation is a
  different piece of work with a different risk, and it is not needed to close
  the gap this ADR closes.
- **The upgrade "current → next" preview.** The panel computes it; the game does
  not store it. There is no `upgradeAmount`/`upgradeValue` array anywhere in
  `Main` (#39). The current effect *is* the live stat, which v2 now carries, and
  the next cost is `cost_log`, which v1 already did — so what is missing is one
  arithmetic step the network can learn rather than a reading that was hidden.
- **The `*Enhancement` family, ultimate weapons, and the ~90 per-round
  accounting fields.** Out-of-run permanent scaling is constant within a run;
  ultimates are not in `run-action-v1`, so showing their state would be showing
  the policy a control it does not have; the per-round totals are cumulative and
  belong to reward analysis, not to a state the policy acts on.
- **`tier_unlocked`**, still. Live 29.0.3 reports false for every offered
  upgrade (M1B-E001), so it carries no signal; it stays in the raw reading for
  drift detection.

## Consequences

- Every v1 checkpoint and every v1 replay buffer is invalid. Both were already
  invalid by cadence, so nothing is lost that ADR 0009 had not already retired.
- `SCALAR_FEATURES` grows from 4 to 42 and `ROW_FEATURES` from 7 to 9. The
  network's input widths derive from those constants, so no width is restated
  anywhere; a schema change that needed an edit in `learning/network.py` would
  be the defect this arrangement exists to prevent.
- The wire grows by about 1.5 KB per state message, well inside the frame bound.
- Value verification is a device stage, not a unit test. The panel shows every
  live reading in the game's own unit precisely so it can be held beside the
  HUD.

- **Confirmed on device, `M2-E006` (2026-09-19):** the schema's values were held
  beside the game — zero out-of-range readings, zero invalid episodes, `#39`'s
  capture reproduced field for field.
