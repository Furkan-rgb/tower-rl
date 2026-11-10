# ADR 0021 — DreamerV3 is evaluated on its top choice

**Status:** Accepted on 2026-11-10 by the Lead, under the developer's delegation of
the call ("I leave the calls up to you", 2026-11-10). Board `#197`.

## Context

`docs/task.md` evaluates a selected model on exploration-free real-game episodes
(the Evaluate stage, and the Definition of Done's "Evaluation is exploration-free").
The official DreamerV3 agent samples its actor in evaluation as in training. It adds
no exploration noise, so the sample is the policy's own entropy, which is its
exploration (`DreamerBackbone.act`). Every DreamerV3 evaluation so far has played
that sample: period means, train.py's final evaluation, and `M3-P024`'s sets A and B
(`docs/solution.md` 9.2b).

On `M3-P023`'s checkpoints the sample departs from the actor's most likely action on
13–17% of decisions, read offline on `#169`'s fixed states (`docs/experiments.md`,
`M3-P024`'s result). That is about 30 of an episode's 200 purchases bought for the
draw, not for the choice. R2D2 is already evaluated on its greedy action, at an
epsilon of 0.

`docs/solution.md` 9.4c had read "exploration-free" as no *added* noise, so that the
sample counted. This record reads the policy's own entropy as exploration too, as
`DreamerBackbone.act` itself describes it. ADR 0018 lets a port leave its reference
code where the environment forces it. Here the force is the task's evaluation rule,
which the sampled play does not meet.

## Decision

1. A DreamerV3 checkpoint is evaluated on its top choice: the most likely valid
   action under its actor, the first of a tie. The latent is still drawn from the
   posterior as acting draws it. That is inference about the state, not a choice
   of action.
2. `run_actors.py`, `run_episodes.py` and `spectate.py` play it as
   `--policy checkpoint-top:<path>`. `checkpoint:<path>` still plays the sample,
   so earlier evaluations can be repeated as they ran. Every record says which was
   played (`top_choice`). `select_checkpoint.py` refuses to mix the two, and
   `report_arms.py` refuses a set B played the other way from set A.
3. Collection is unchanged: actors sample, as the official agent does. So a period's
   mean remains the sampled policy's (`#157`). It orders the checkpoints set A plays.
   It does not report them. train.py's own evaluation of its final weights, a
   readout since `M3-P024`, still samples too, and `#157` labels it as such.

## Options rejected

- **Keep the sample as the evaluated policy, and label it as such** (`#157`). That
  is honest about what was measured, but it measures something the task does not ask
  for.
- **Take the top choice in collection too.** That would remove DreamerV3's only
  exploration, which ADR 0018 does not allow without a cause in the environment.

## Revisit if

- On the same checkpoints, top-choice play is worse than the sample, which would
  mean the actor relies on its own mixing.
- A tie between actions is common enough at evaluation that the first-index rule
  decides episodes.
