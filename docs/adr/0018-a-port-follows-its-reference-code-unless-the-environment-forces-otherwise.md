# ADR 0018 — A port follows its reference code unless the environment forces otherwise

**Status:** adopted in code for DreamerV3, 2026-09-30. Board `#108`. Verified
by unit tests and a GPU smoke run on fake ports; no device run yet.

## Context

The DreamerV3 port (`learning/dreamer.py`, §9.4c) was written against the
official code (danijar/dreamerv3 at e3f02248), but where it met this
project's shared machinery it took the shared machinery's conventions
instead: stacked-dqn's window replay with no latents and no online queue, a
zero-state start for every window, a phantom terminal step because replay
held no terminal observation, masked first-step losses to undo a layout
shift, and the paper's actor unimix where the code applies none. Each was
recorded as a deviation, but the reason given was usually that the shared
code did not support the official behaviour, which is a reason to change the
shared code, not the algorithm. The offline KL(full‖window) measurement
showed the zero-state start put the window's posterior away from the one
acting filtered.

## Decision

**A port of a published method follows its reference code's conventions.
It deviates only where this environment or protocol forces it, and every
deviation names what forces it: an environment contract, an ADR, or the
task's protocol.** "Our shared code does not do that" is not such a reason;
the port gets the component it needs, in its own module, instead.

For DreamerV3 this gave it its own replay (`learning/dreamer_replay.py`,
a port of `embodied/core/replay.py`) with context 1, stored and written-back
latents, the online queue, an item at every step, FIFO capacity 5e6 and the
official warm-up; the driver's step layout with the environment's terminal
observation; first-step losses as `_annotate_batch` leaves them; no actor
unimix; the mask, a boolean key, one-hot in and a two-class categorical out;
and acting on the parameters of the last completed step, loaded before each
decision with the latent carried across, where it had refreshed only at
episode starts. The deviations left each name their cause in the §9.4c table:
whole-episode insertion (ADR 0014, ADR 0017), the stream cut at an
inadmissible transition (the environment contract), the action mask as an
observation key (the environment contract), the per-transition game-time
discount (ADR 0013), and the unused `td_errors` (the `Backbone` contract).

## Consequences

- DreamerV3 checkpoints are format 7 and its replay dumps format 3; an older
  DreamerV3 checkpoint is refused as a resume (a mixed run) and still loads
  for evaluation, with the actor unimix and the 0/1 mask it was trained with
  (`dreamer_mask_one_hot` absent reads as False).
- Every completed learner step now publishes a snapshot of the parameters,
  and an actor loads it without waiting for the step in flight. That replaces
  ADR 0017's publication, which copied the live network under a lock a step
  held and so could wait for one step; R2D2 gets the same no-wait load, network only,
  at its 400-decision cadence. The debt is unchanged in kind and counted in
  credits (ADR 0017).
- Replay costs ~10.6 KB per step at the official float32 entry precision,
  ~10.6 GB per 1M decisions (§9.4c).
- The same rule applies to any later port. It was applied to R2D2 on board
  #111, which replaced stacked-dqn (removed; commit `cb2f324` is the last
  that can resume or evaluate it). R2D2's deviation list is the conventions
  table in `docs/solution.md` section 9.4, one row per convention, each "same"
  or naming what forces it. Its checkpoints are format 8 and its replay dumps
  format 4; the old window buffer's dump formats 1 and 2 are refused.
