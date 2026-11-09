# ADR 0020 — One decision a wave, by standing order

**Status:** Accepted by the developer on 2026-11-09, amended to `filling-orders`:
items 1–4 as written below, with `filling-orders` as the cadence. Board `#167`. Item
5 accepted the same day, amended to batches of 64 × 64 on the cause of `M3-P023`'s
learner failures (`#169`, `#182`). The earlier amendments came from the pilots: item
2 on `M3-P021`'s result (2026-11-07; its first form, the affordability mask, is
under Options rejected), item 5 on the review of the training setup (2026-11-07),
and the ceiling cost on the wave-483 trace (2026-11-08). The pilots `M3-P021`, `M3-P022` and `M3-P023`
(`docs/experiments.md`) are the spikes whose output this decision is. ADR 0009's
choice points still define which stops can be decisions. Item 3 supersedes ADR
0019's value for runs at the order cadences.

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

## Decision

**As accepted, 2026-11-09: the cadence is `filling-orders`.** An order buys the
row's next level the moment the cash reaches its price, and buys nothing else until
then, across as many waves as that takes. It ends early if health falls by
`health_change_fraction` or the row leaves sale. The legal set is item 2's, every
row on sale, so decisions follow purchases: several a wave while levels are cheap,
fewer than one while saving (`docs/environment-contract.md`, "Decision cadence").
The developer chose it over `saving-orders` because the script's bar in
`docs/task.md` §1.1, 101.82, is measured at it. `standing-orders` and `saving-orders`
stay selectable as recorded controls. Items 1 and 2 below are the proposal as the
pilots tested it.

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
5. **Train ratio 2048, in batches of 64 × 64.** A wave-long decision yields about
   11,000 decisions an hour against the learner's ceiling of about 42,000 updates
   of 16 × 64 an hour, so 512 would use about 13% of it. ADR 0018 requires a
   deviation to name what forces it, and this cadence does: decisions arrive about
   eight times slower while the update ceiling is unchanged. 2048 replays 2,048
   steps a decision, about half the ceiling. They are replayed as half an update a
   decision of 64 sequences, four times DreamerV3's 16, because at 16 × 64 the
   actor's two updates a decision moved its top choice on 0.77 of fixed states per
   2,500 decisions in `M3-P023`, on value differences below its own noise. Offline,
   on that run's replay, 64 × 64 moved it on 0.22 against 0.55 at 16 × 64, at about
   the same cost (`docs/experiments.md`, "Why `M3-P023`'s learner failed"). The
   world model learns at 128 of those steps, from one update in sixteen
   (`--world-model-ratio 128`, `docs/solution.md` §9.4c), because at 2048 it
   memorised `M3-P022`'s episodes: a purchase earned no honest credit in
   imagination, and on the same data the world model memorises from ratio 256
   (`docs/experiments.md`, "Review of the training setup"). At 128 it is not the
   cause of the actor's movement (the same entry).

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
- **This lifts no ceiling by itself.** The script's end near wave 102 at Workshop 10 is
  `build-defense-absolute`'s own wall: base damage after Defense % passes the flat
  block (`docs/experiments.md`, "The wall at wave 483"). It is not shown to be the
  account's ceiling. At Workshop 79 a blender, which does not rest on the block,
  ended 40 waves before it, its rows capped in-run by wave 191 (`docs/experiments.md`,
  "Turtle into blender on the block's margin at Workshop 79"). The target is now to
  beat the script at Workshop 10 (`docs/task.md` §1.1, 2026-11-09).

## Revisit if

- `M3-P021` does not beat its floors, or beats them only at `choice-points`'
  known level.
- The scripted floor at `standing-orders` falls well below `choice-points`'.
- A learner at `filling-orders` cannot beat the script for a reason the order
  itself causes, such as a save that the health rule keeps ending early.
