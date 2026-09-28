# ADR 0013 — Discount horizon is a task parameter

**Status:** adopted from the next run (`M3-P015` onward), 2026-09-28. Board
`#85`. Not yet validated by a training run.

## Context

`M3-P009` through `M3-P014` (`docs/experiments.md`) tuned the discount `γ_s`
(`docs/solution.md` §9.4d) as if it were a `stacked-dqn` hyperparameter,
chosen and re-chosen against that learner's own collapse bars. `M3-P011` (γ
0.999) tripped its kill bar at 100k decisions and was read as unstable;
`M3-P012` reverted γ to 0.997 with everything else held, and also failed the
same bar. A specialist offline analysis of the gradient trace
(`state/mlflow.db`, `M3-P011`/`M3-P012`/`M3-P009`, 2026-09-28) found `M3-P011`
was not unstable — its pre-clip gradient norm (22.1/20.1 at matched periods)
tracked the larger value scale at γ 0.999 with no divergence or NaN, and it
beat its exact 0.997 twin `M3-P012` at 100k (31.86 vs 28.44, n=1). The kill
bar that stopped it was built from `M3-P009`'s value, not from a comparison
with its twin (see the correction note in `docs/experiments.md` near
`M3-P011`/`M3-P012`). That reframes the question the discount tuning was
actually answering: not "does the learner tolerate this γ" but "what horizon,
in game time, does crediting a purchase in this game actually need."

The same specialist derivation (`M3-P014` replay dump, mlflow `156c54ce`, 267
complete episodes; `M3-P011`/`M3-P012` as a clean n=1 twin pair) measured that
horizon directly, in game time rather than in decisions, so it does not
depend on how many choice points a policy happens to create.

## Decision

**The discount horizon is a task parameter of the benchmark protocol,
expressed per game-second, and held identical across every learner.**

- **Task vs algorithm parameters.** Task parameters are held identical across
  all learners (`stacked-dqn`, DreamerV3 and later ones), because they define
  the problem: reward, decision cadence, budget, Workshop level, upgrade
  availability, and now the discount horizon. Algorithm parameters — network
  architecture, optimiser, replay details, n-step, and the rest of each
  method's published recipe — stay at each method's own values.
- **Expressed per game-second.** A learner that discounts per decision
  converts the task horizon with its measured game-seconds per decision
  (~1.85, `M3-P014` replay dump). Example: DreamerV3's published discount
  0.997 per environment step is ≈620 game-seconds ≈18 waves here — short of
  the task horizon below — so under this ADR Dreamer is given the task
  horizon converted to its own step, not its paper default.

### Derivation

- **Wave length:** 35.0 game-seconds, stable across waves 2–74 (`M3-P014`
  replay dump).
- **Purchase-to-death delay** (the only point a survival reward can credit a
  purchase), medians: Thorn Damage 1264 s, Defense % 1054 s, Health 853 s,
  Defense Absolute 800 s, Knockback 619–750 s.
- **Credit delay** τ ≈ 1050–1630 s, from the delays above.
- **Noise floor.** The checkpoint-to-checkpoint spread of the action-value gap,
  σ_A, scales as (value scale)^k with k ≈ 0.6–0.9, measured from the
  `M3-P011`/`M3-P012` pair (identical configs except γ).
- **SNR optimum:** γ* = exp(−k/τ) ≈ 0.9994 per game-second.
- **Chosen value: 0.999 per game-second** — a horizon of ≈1000 game-seconds,
  ≈28.5 waves — which keeps ≥69% (mean 88%) of the optimal signal-to-noise
  ratio over the measured (k, τ) range; 0.997 keeps ≤26%. Per decision (~1.85
  game-s), this is ≈0.998, inside the published per-step range used by R2D2
  (0.997) through Agent57 (0.9997) — those figures are per Atari step (1/15
  s) and do not transfer as numbers, only as an order-of-magnitude check.
- **Reward scaling.** The survival reward is scaled linearly so the maximum
  return stays V_REF = 9.51 (`docs/solution.md` §9.4e), keeping the reward
  scale — and with it Huber delta, gradient clip and the |TD|/gradient-norm
  monitors — independent of γ.

### Status and validation

Adopted as the protocol value from `M3-P015` onward. It has not yet been
validated by a training run. Validation criteria, from the developer's
stage-1 target (Defense Absolute paired with Thorns): Thorn spend share and
Thorn purchase rate at waves 30/40 rising clearly above `M3-P014`'s 3.1% /
1.1%, plus `M3-P014`'s collapse guards (`docs/experiments.md`).

### When to re-derive

τ grows with survival, so re-derive (toward the ≈0.9995 optimum) once
near-greedy deaths routinely reach wave ≥55. Re-derive also if the game setup
changes (Workshop level, game version).

## Evidence

- Wave length, purchase-to-death delays, credit delay τ, σ_A scaling, SNR
  optimum and chosen value: specialist derivation, 2026-09-28, `M3-P014`
  replay dump (mlflow `156c54ce`) and the `M3-P011`/`M3-P012` twin pair.
- `M3-P011` mechanism (not unstable, gradient norm tracked value scale, beat
  its twin at 100k): same derivation, §4, from `state/mlflow.db`.
- Reward scaling and V_REF = 9.51: `docs/solution.md` §9.4d–e.

## Consequences

- `docs/solution.md` §9.4d's recipe value (0.999 per game-second) is now the
  protocol's task parameter rather than a `stacked-dqn`-only tuning choice;
  see the pointer added there.
- A learner added later inherits the task horizon by conversion, not by
  re-tuning it against its own collapse bars, the way `M3-P009`–`M3-P012`
  tuned it for `stacked-dqn`.
- `docs/experiments.md`'s `M3-P012` row is corrected to attribute the
  13.5/6.9 well-clipped gradient norm to `M3-P009`, not to the number it had
  been mislabelled with (`M3-P011`'s 22.1/20.1).
