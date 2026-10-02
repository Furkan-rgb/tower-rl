# ADR 0019 — The discount horizon is 0.997 per game-second

**Status:** adopted, 2026-10-02. Board `#113`. Supersedes ADR 0013's value
(0.999 per game-second) for both learners. The rest of ADR 0013 stands: the
discount is a task parameter, expressed per game-second, held identical across
learners, with the survival reward scaled so its maximum return stays V_REF.
Neither learner's code default changes: the flag stays required and the run
sets it.

## Context

ADR 0013 chose 0.999 per game-second (a horizon of about 1,000 s, about 28
waves) from a signal-to-noise derivation, which predicted that 0.997 keeps at
most 26% of the optimal signal. `M3-P018` (`docs/experiments.md`) was
pre-registered to test it: DreamerV3 at its official conventions, `M3-P017`'s
launch unchanged except `--discount-per-game-second 0.997`. Its stated
consequence was that if 0.997 won, ADR 0013 would be superseded and R2D2's
discount would follow the result.

## Decision

**The protocol's discount is γ_s = 0.997 per game-second**, a horizon of about
333 game-seconds, about 9.5 waves, for DreamerV3 and R2D2 alike. A run passes
`--discount-per-game-second 0.997`. Where ADR 0013's derivation and this
evidence disagree, the evidence is what is adopted; the derivation is not
re-done here.

## Evidence

- `M3-P018` met its pre-registered bar: period 10 read 44.88 and period 11 read
  46.02, both above 43.09, the highest best period mean among the three 0.999
  controls (`M3-P016` attempt 1 43.09, attempt 3 36.34, `M3-P017` 38.05). Its
  best period reached 59.10 at 1,000,000 decisions, and its pre-registered
  final evaluation there was 63.28, SD 9.04, n=29 valid.
- stacked-dqn at 0.997 reached 55.70 (`M3-P014`, provisional arm-n10, n=10,
  `docs/experiments.md` "M3-P014 ... Stopped, as run"), against about 31 for the
  same recipe at 0.999 (`M3-P015`). That 31 has no results entry in
  `docs/experiments.md`: `M3-P015` was stopped at 404,139 decisions with arm
  period 6 at 34.38, per the `M3-P016` entry, so the figure has no pointer.
- Contrary evidence, kept: stacked-dqn at 0.999 beat its 0.997 twin at 100k
  decisions (`M3-P011` 31.86 against `M3-P012` 28.44, n=1).

## Caveats

- **n=1 against three controls.** One DreamerV3 seed against three 0.999 runs
  (two of them on a different DreamerV3 port, before ADR 0018) shows an effect;
  it does not size it. The DQN comparison is also one run each.
- **Confounded.** The survival reward per game-second is scaled with the
  horizon (ADR 0013, reward scaling), so `M3-P018` changed the horizon and the
  per-second reward together. At 0.997 the scaled reward is three times the
  0.999 run's per game-second at the same maximum return. DreamerV3's return
  normalisation should absorb much of this; it was not measured.
- **The causal claim is weakened.** `M3-P017`'s diagnosis put only 15.5% of
  spend into Defense Absolute from wave 30 on, but that covered its first
  roughly 300,000 decisions; its last period (38.05) had already switched to
  about 79%. `M3-P017` may have been breaking its plateau when the 500k rule
  stopped it. 0.997 reached the switch sooner and went higher; it does not
  follow that 0.999 would never have. (A summary from a lost scratchpad, not
  re-verified.)
- **Training past 1,000,000 decisions did not help.** The developer-approved
  extension's 2,007,970-decision evaluation was 55.77, SD 19.79, n=30 against
  63.28 at 1M: a difference of -7.5, Welch SE about 4.0, with 6 of 30 episodes
  dying at waves 20-23. At 0.997 the policy still kept buying Defense Absolute
  over affordable Thorn Damage (Thorn Damage 5-8% of spend, no trend), so the
  horizon did not change that part of the strategy.
- R2D2 has not run at 0.997 under this protocol; this ADR sets its value from
  DreamerV3's result and stacked-dqn's, not from an R2D2 run.

## Consequences

- `docs/solution.md` section 9.4d states 0.997 as the chosen value and the R2D2
  conventions table follows. ADR 0013 is marked superseded.
- Runs from here pass 0.997. `M3-P015` to `M3-P017` stay recorded at 0.999.
- A checkpoint's resume guard still refuses a different discount than its own,
  so a 0.999 checkpoint does not continue at 0.997.
- Code comments, help text and usage examples that call 0.999 "the protocol's
  value" (`scripts/train.py`, `learning/r2d2.py`, `learning/dreamer.py`) are not
  updated by this ADR.
