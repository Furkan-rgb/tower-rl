# ADR 0020 — One decision a wave, by standing order

**Status:** Proposed, 2026-11-06. Item 2 amended 2026-11-07 on `M3-P021`'s result:
it first kept the mask at affordability, and that form is recorded under Options
rejected. Board `#131`, `#134`. Nothing here is the contract: `choice-points` stays
the default and ADR 0009 and ADR 0019 stand. The pilots `M3-P021` and `M3-P022`
(`docs/experiments.md`) are the spikes whose output is this decision, and the
developer accepts or rejects it.

## Context

The developer asked whether the reward is right. The audit in `docs/experiments.md`
("The credit audit") found the reward's formula correct and aligned, and found
that the setup around it cannot credit what the game needs credited.

At Workshop 10 the agent takes about 22 decisions a wave and discounts at 0.997
per game-second, a horizon of about 9.5 waves (ADR 0019). A purchase that adds one
wave of survival is worth, in return, 0.33 ten waves from death, 0.014 at forty,
0.0002 at eighty. Against `M3-P019`'s return scale (S = 7.82) and DreamerV3's
entropy scale (3e-4), that advantage moves the action odds by 414× at forty waves
from death, 2.1× at sixty and 1.01× at a hundred.

The measured policies show the same shape. In the last 150,000 decisions of
`M3-P019`, Defense Absolute's share over uniform falls from 9–11× within ten waves
of death to 1.2× past sixty, and `M3-P018` falls from 18× to 1.8×. Distance to
death is correlated with wave number, so this is a correlation and not proof. It
does match the arithmetic.

The horizon cannot simply be raised at this cadence. ADR 0013 chose 0.999 per
game-second and it lost to 0.997 (ADR 0019). Imagination stops after 15 decisions,
which is about 28 game-seconds here. At wave 1000 about 94% of an episode's
decisions would sit more than sixty waves from death, where the odds barely move.

## Decision (proposed)

1. **One decision a wave.** A third `DecisionCadence`, `standing-orders`. The
   policy is asked at a wave's first choice point, over the unchanged action mask,
   and its answer stands until the wave turns
   (`docs/environment-contract.md`, "Decision cadence").
2. **Actions stay semantic and the legal set is every row on sale.** `saving-orders`
   is `standing-orders` over every row the run still sells, affordable or not. An
   order buys nothing else, so one for a row out of reach saves for it, and buys it
   the moment the wave has earned the price. Affordability stays in each row's
   features. `standing-orders` stays selectable as `M3-P021`'s recorded control.
3. **The discount is 0.997 per wave**, which is `γ_s = 0.997^(1/35) ≈ 0.99991416`
   per game-second. That gives DreamerV3 its stock horizon of 333 of its own steps,
   here 333 waves. The discount stays a task parameter per game-second (ADR 0013).
4. **The reward is unchanged**: `(1−d)·V_REF` with `d = γ_s^Δt`.
5. **Train ratio 2048.** A wave-long decision yields about 11,000 decisions an hour
   against the learner's ceiling of about 42,000 updates an hour, so 512 would use
   about 13% of it. ADR 0018 requires a deviation to name what forces it, and this
   cadence does: decisions arrive about eight times slower while the update
   ceiling is unchanged. 2048 is 2 updates a decision, about half the ceiling.

On paper, the odds a +1-wave advantage earns rise to e^30 at a hundred waves from
death, e^9 at five hundred and about e^2 at a thousand, at a return scale of about
2.3. That is arithmetic, not a measurement, and the pilot is what tests it.

## Options rejected

- **A logarithmic survival reward** would need a new acceptance decision under
  `docs/task.md` §7.5, and does nothing about the imagination cutoff.
- **A longer discount alone** at this cadence: see ADR 0019's evidence.
- **The affordability mask, saving through `WAIT`.** This ADR's first form, chosen
  because saving through `WAIT` cost a script less than a wave's delay and because
  a wider mask looked like a second mask. It weighed the wrong cost. `M3-P021`'s
  policy took Defense Absolute about 70% of the time at decisions 31–50 when the
  mask offered it, the mask offered it 27–39% of the time, and when it did not the
  policy bought another row 89–93% of the time, which kept the row out of reach. A
  script saves by rule, and a learner has to find a run of `WAIT`s that pay nothing
  until the last one (`docs/experiments.md`, "`M3-P021`: outcome"). No second mask
  was needed: a row already says whether it can be paid for.

## Costs to be measured

- One row a wave is a real restriction. The floor at this cadence is measured
  against the same policy at `choice-points` before the learner runs.
- A checkpoint from one cadence cannot be resumed or evaluated at another.
- **This lifts no account ceiling.** Scripted play at Workshop 10 ends near wave
  102, and a better learner at that level does not make wave 1000 reachable.

## Revisit if

- `M3-P021` does not beat its floors, or beats them only at `choice-points`'
  known level.
- The scripted floor at `standing-orders` falls well below `choice-points`'.
- Wave 1000 needs more than one purchase row a wave.
