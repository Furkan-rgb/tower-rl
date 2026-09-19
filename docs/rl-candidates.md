# Tower-RL — Reinforcement-learning algorithm candidates

## 1. Purpose and authority

This document ranks candidate learning algorithms for the V1 objective and
states how each one would have to be built against the measured Tower-RL
environment. It is a decision document. It does not change `task.md`,
`solution.md`, or the environment contract, and it contains no production code.

Where it disagrees with `solution.md` section 9 it says so explicitly. The most
important disagreement is recorded in section 3.3: R2D2 as specified there is a
large-budget algorithm, and its published configuration is not the right
default for the sample budget this project actually has. Reconciling that is a
Lead decision, not a decision taken here.

Every quantitative claim about published work is cited in section 11. Every
quantitative claim about this environment comes from `docs/experiments.md`,
principally `M1B-E002`. Claims that are engineering judgement rather than
measurement or literature are marked as judgement. The literature does not
contain a result for this environment, this observation space, or this budget,
so most of the per-candidate performance expectations below are judgement, and
are stated as such rather than dressed up as evidence.

## 2. The problem as an RL instance

Before ranking anything it is worth being precise about which properties of the
problem actually discriminate between algorithms. Several of the obvious ones
do not.

### 2.1 Budget arithmetic

Measured, from `M1B-E002`: at game speed 4.0 with a speed-scaled decision
cadence, one free-running actor completes an episode in roughly 68 seconds of
wall clock and takes between 261 and 785 decisions, with an observed mean near
540. Scripted greedy play reaches wave 7 to 10; buying nothing dies at wave 2.

That gives, for one actor:

- about 53 episodes per hour;
- about 28,000 decisions per hour.

Assuming four concurrent actors scale linearly:

- about 210 episodes per hour;
- about 113,000 decisions per hour;
- 10^5 decisions in roughly one hour;
- 10^6 decisions in roughly nine hours;
- 10^6 decisions is roughly 1,850 episodes.

Linear scaling to four actors is an assumption, not a measurement. The
workstation runs the emulator with pinned `lavapipe`, which is CPU rasterization
(`M0-E013`), so four AVDs compete for the same 32 cores rather than for the
4090. Actor scaling must be measured before any comparison protocol is fixed,
because every budget number in section 7 depends on it.

Two consequences follow immediately.

First, the total budget for a full multi-backbone comparison is large in wall
clock. Six candidates at five seeds and 10^6 decisions each is about 270 hours
of continuous four-actor emulation, roughly eleven days, before any evaluation
episodes are spent. This is the reason section 6 stages the work rather than
building everything and comparing at the end.

Second, the budget band of 10^5 to 10^6 decisions is directly comparable in
magnitude to the Atari 100k benchmark, which allows 100,000 agent steps
(400,000 frames), about 1.85 hours of game time. Atari 100k is therefore the
closest published analogue for the budget, and the sample-efficiency literature
built around it is the most relevant body of work. Section 2.5 explains why its
headline margins should nonetheless not be expected to transfer.

### 2.2 Effective horizon is the dominant difficulty

This is the single most important property of the problem and it is easy to
miss, because it is a property of the decision cadence rather than of the game.

At a one-second cadence an episode is 300 to 800 decisions long. Reward is
emitted on wave change. A scripted episode reaching wave 8 emits at most six or
seven non-zero rewards across several hundred decisions. The agent therefore
receives on the order of 10^-2 reward events per decision, and most actions in
an episode are `WAIT`.

The learning problem is consequently not a representation problem. The
observation is small, structured, exact, and already contains the agent's own
build state. The problem is credit assignment across a long horizon with sparse,
low-cardinality reward, under a per-episode outcome distribution with very large
variance (the same scripted policy produced waves 7, 4 and 10 in three
consecutive 4x episodes).

This has a concrete implication that outranks the algorithm choice: coarsening
or event-triggering the decision cadence is probably worth more than any
candidate in section 3. If decisions were triggered on wave change, on an
upgrade becoming newly affordable, on a material health change, or on a
bounded maximum wait, an episode would contain tens of decisions rather than
hundreds. Effective horizon would fall by roughly an order of magnitude, the
number of decisions per reward event would fall with it, and every candidate
below would become more sample-efficient at no cost in real experience. The
environment contract already contemplates this as an optional later mode
(`solution.md` section 7.3).

The recommendation in section 6 is therefore to treat decision cadence as a
first-class experimental variable, measured once with a fixed simple learner,
rather than as a fixed environment property that algorithms must accommodate.

### 2.3 How partially observed this problem really is

The stated partial observability is that the observation carries no enemy,
threat, or boss information. That is real, but it is narrower than it sounds.

The 60 upgrade rows carry level, max level, headroom, cost, unlocked and maxed.
That is a complete and exact description of the agent's own build. Nothing about
the agent's past purchasing needs to be remembered, because the consequences of
every past purchase are visible in the current observation. The classic reason
for recurrence in a POMDP — that the policy must remember what it did — does not
apply here.

What is genuinely hidden is short-horizon threat: what is on screen now, how
hard the current wave is hitting, whether a boss is present. The observable
signature of that is the recent trajectory of health, and secondarily of cash
income. That signal has a short time constant. It is recoverable from a window
of the last few to few tens of decisions, not from episode-scale memory.

Judgement: a stacked short history of run scalars is likely to capture most of
the recoverable hidden state, and a recurrent state is likely to add less here
than its cost in replay complexity and hyperparameter surface. This is a
testable claim and section 6 treats it as the first ablation after the floor is
established, rather than as an assumption baked into the architecture. It is
also the main reason the ranking in section 3 puts a feed-forward agent above a
recurrent one.

### 2.4 What the pause capability actually buys, and what it costs

`M1B-E002` verified that `Pause` and `Unpause` freeze the world exactly: over
six paused seconds cash, health and wave were unchanged and cash resumed
advancing on unpause. Deliberation therefore costs zero game time. This is
genuinely unusual for a real-time environment and it is the precondition for
decision-time search.

Three things temper it.

First, it is not free in wall clock. The same entry measured stepped mode at
roughly 2.3 steps per wall-clock second, about 430 ms per decision, of which
only 80 to 166 ms is unpaused world time. Free-running at speed 4.0 costs about
126 ms of wall clock per decision. Stepped operation is therefore about 3.4
times more expensive per decision today, before any planning time is added.
Wall clock is the binding constraint for this project, so any search method
starts with a measured 3.4x throughput penalty that must be profiled away
before it can be treated as affordable. The entry notes the cause has not been
localised and must be profiled rather than guessed.

Second, pause enables deliberation but not simulation. The environment cannot be
reset to an arbitrary state and cannot be branched, so search cannot use the
real game as its model. Any search method must plan inside a learned model. The
pause therefore does not remove the burden of learning dynamics; it only removes
the latency objection to using a learned model at decision time.

Third, under the choice-point cadence (ADR 0009) that section 2.2's cadence
change produced, the "shallow search does not reach the reward" argument no
longer holds as originally stated, though an honest recompute is weaker than
a first read suggests. The mean legal action set differs by which state set
it is measured over: at a choice point it is **≈4.2 actions including WAIT**
(`M2-E004`, measured over choice-point states); averaged over every
every-slice state, 68% of which were WAIT-only, it falls to ≈2.0. Under
ADR 0009 the agent decides only at choice points, so ≈4.2 is the branching
factor a search method actually faces. At that branching factor, 32
simulations reach a depth k where 4^k ≤ 32, i.e. k ≈ 2.5 — EZ-V2's 32
simulations are exhaustive enumeration to a depth of about **2 to 3**, not
the 5 plies the original argument assumed. `M2-E007` measures 5.00 decisions
per wave under random play and 3.23 under scripted play, so a tree of that
depth spans roughly half to three-quarters of a wave, not a whole one, and
does not reliably reach a reward event at its leaves. The argument that
demoted the MuZero family to rank 5 in section 3.5 is weakened by the
cadence work, not simply removed; section 9 revises the ranking accordingly.

### 2.5 Why published margins should not be expected to transfer

Most of the modern sample-efficiency literature — SPR, SR-SPR, BBF, EfficientZero
and its successors, the transformer world-model line — is measured on Atari
100k, from pixels. A large part of what those methods buy is efficient
representation learning from images: self-predictive latent losses, image
augmentation, large residual encoders. Tower-RL has exact structured
observations of a few hundred numbers. The representation problem those methods
solve does not exist here.

The parts of that literature that should transfer are the optimisation
ingredients rather than the representation ingredients: high update-to-data
ratio, plasticity maintenance, annealed n-step returns, distributional or
normalised value targets, and careful target-network handling. The parts that
should not be assumed to transfer are the self-supervised representation losses
and the augmentation pipelines.

This is the main honest caveat on the entire shortlist. A paper reporting an
interquartile mean of 1.045 against 0.415 on Atari 100k is not predicting a
similar ratio on final Tier-1 wave.

## 3. Ranked shortlist

The ranking is by expected value per unit of engineering risk at this budget,
not by published peak performance. For each candidate: what it is, fit, expected
sample efficiency, implementation cost and risk, masking, partial observability,
and failure modes.

### 3.1 Rank 1 — Masked data-efficient DQN on a stacked history

**What it is.** Double and dueling DQN over the masked 61-action space, with a
shared per-upgrade-row encoder as already designed in `solution.md` section 9.3,
a feed-forward trunk, and a stacked window of recent run scalars instead of
recurrence. The intended optimisation recipe is the one from the Atari 100k
literature rather than the Rainbow or R2D2 defaults: replay ratio of roughly 2
to 8, n-step returns annealed from about 10 down to 3, an EMA target network,
prioritized replay, Huber loss, weight decay, and either periodic
shrink-and-perturb resets or strong normalisation in place of them, with
Munchausen as a two-line optional addition.

What actually runs, as of run 2 (`docs/experiments.md` M2-E007; audit #53):

- Replay ratio **0.25 gradient steps per decision** (`training.py:220`) —
  specified, not implemented — see #53.
- n-step **fixed at 10**, never annealed (`StackedDqnConfig.n_step`,
  `stacked_dqn.py:45`) — specified, not implemented — see #53.
- EMA target network: implemented, decay 0.995 per gradient step.
- Prioritized replay: the machinery is built (`replay.py`), but
  `priority_alpha` runs at **0**, which makes sampling uniform and importance
  weights exactly 1 — specified, not implemented — see #53.
- Huber loss (δ=1): implemented.
- Weight decay: implemented, at `1e-5`; this document does not specify a
  value, so the magnitude is not marked as a deviation.
- Shrink-and-perturb resets / strong normalisation: absent — specified, not
  implemented — see #53.
- Munchausen: not implemented; always described here as optional.

Two further facts govern the current run and are not part of the recipe as
originally written here: the ε ladder anneals to its floor over the first
**2,500 decisions** (`--epsilon-anneal-decisions`, run 2 launch parameters,
M2-E007), and warm-up runs for **100 sequences** (`--warmup-sequences`; under
the choice-point cadence one episode is one sequence, so this is
≈100 episodes ≈ 2,690 decisions). The anneal therefore reaches its floor
before the first gradient step is taken; see #53 for the consequence.

**Fit.** This is the closest match to the problem's actual shape. The
observation is low-dimensional, so no encoder pretraining is needed. The action
space is discrete and masked, which value-based methods handle exactly and
cheaply. Replay makes every expensive real decision reusable many times over,
which is what matters when environment time rather than GPU time is the binding
constraint. BBF reaches an interquartile mean of 1.045 on Atari 100k with a
replay ratio of 8, showing that a purely model-free value-based agent can be
competitive at a budget of this magnitude when the optimisation is right.

**Expected sample efficiency (judgement).** It should beat the always-wait floor
of wave 2 within the first few hundred episodes, because the reward signal for
"spend at all" is strong and immediate. Matching the scripted greedy policy at
wave 7 to 10 is the real test and is uncertain; it may take most of the 10^6
decision budget, and it may not happen at all without the offline bootstrap in
section 3.2 or the cadence change in section 2.2. No published number supports a
sharper estimate.

**Implementation cost and risk.** Lowest of all candidates. It reuses the replay,
evaluation and checkpointing machinery the project needs regardless. The risky
parts are masking correctness and replay-ratio tuning, both of which are
testable offline. Single-machine training is trivially within the 4090's
capacity; the GPU will be idle relative to the emulators.

**Masking.** Value-based masking is the cleanest of any candidate, and also the
easiest to get subtly wrong. The rules are in section 4. The two that matter
most: the dueling mean subtraction must be taken over valid actions only, and
every stored transition must carry both its own mask and the next state's mask,
because the bootstrapped maximum is taken over the next state's valid set.

**Partial observability.** A stacked window of the last k run-scalar vectors,
with k treated as a tuned hyperparameter in the range 4 to 16. Upgrade rows are
supplied for the current step only; they are already a sufficient description of
the build and stacking them multiplies input size by k for no information gain.

**Failure modes to watch.**

- Degenerate `WAIT`. `WAIT` is always valid and is the majority action. A value
  function that is slightly pessimistic about purchases collapses to waiting,
  which survives to wave 2. Monitor the action distribution and the purchase
  rate per episode against the scripted policy, not just return.
- Masked-entry leakage. If masked Q-values are implemented as a large negative
  constant rather than excluded from the loss, gradients flow into invalid
  entries and the shared scorer learns to produce that constant, which corrupts
  the same scorer's outputs for valid rows.
- Replay-ratio instability. High replay ratio with a small and growing buffer
  overfits early data. This is the primacy bias; resets with a preserved buffer
  are the standard mitigation, but see section 8.3 — resets are not
  unconditionally good and can degrade normalised architectures.
- Priority pathology. With reward non-zero only at wave boundaries, TD error
  concentrates on those transitions and prioritized replay can sample almost
  nothing else. Check the priority distribution, not only its mean.
- Episode-length confound in the buffer. Longer episodes contribute more
  transitions, so the buffer is biased toward the behaviour of successful
  episodes. This is usually benign but it interacts badly with the evaluation
  traps in section 7.4.

### 3.2 Rank 2 — The same agent bootstrapped from logged scripted play

**What it is.** Not a separate backbone. It is a modifier applied to rank 1 (and
applicable to ranks 3 and 4): log the non-learning baseline policies the project
already runs — always-wait, random-valid, round-robin-affordable, and the greedy
scripted policy that reaches wave 7 to 10 — into a replay dataset, then train
online with symmetric sampling, drawing half of every batch from that offline
dataset and half from online experience, as in RLPD. Optionally add the
value-reincarnation variant, where a value network is trained against the
scripted policy's returns before online learning starts.

**Fit.** Very strong, and probably the largest single expected gain per unit of
engineering risk in this document. The offline data is nearly free: the project
must run those baselines anyway to establish the floor, and logging them costs
only disk. It directly attacks the problem's worst feature, which is that
random exploration in a 61-action masked space over a 500-step horizon will
almost never discover a wave-8 build by chance. Reincarnating RL argues that
tabula rasa training is a research convention rather than a requirement, and
that reusing prior computation is what large real systems actually do. RLPD
reports up to 2.5x improvements from symmetric sampling with LayerNorm value
regularisation, without pretraining or explicit offline-RL constraints.

**Expected sample efficiency (judgement).** The concrete prediction is that the
agent starts near the scripted policy rather than near the always-wait floor,
and spends its real budget improving on wave 7 to 10 rather than rediscovering
that upgrades should be bought. If the bootstrapped agent cannot beat its own
demonstration data within the budget, that is itself a valuable and early
negative result about the headroom available at this baseline.

**Implementation cost and risk.** Low. Symmetric sampling is a change to the
batch sampler. LayerNorm in the value network is a one-line change. The main
work is a disciplined dataset format carrying schema versions, masks and
profile identity, which the contract already requires of online replay.

**Masking.** Identical to rank 1, with one addition: the offline dataset must
store the masks that were in force when it was collected. A dataset without
masks is unusable for target computation, and recomputing masks after the fact
from stored observations is only safe if the mask is a pure function of the
stored fields. Verify that it is before relying on it.

**Partial observability.** Identical to rank 1. The stacked window must be
reconstructible from the logged trajectories, which means logging whole
trajectories rather than independent transitions.

**Failure modes to watch.**

- Anchoring to the demonstration. The agent matches wave 7 to 10 and stops. The
  diagnostic is whether the online half of the batch ever changes the policy;
  compare against an ablation with the offline fraction annealed to zero.
- Stale profile. Offline data collected under a different game version, speed
  profile or baseline is evidence about a different environment. The contract
  already tags replay with profile and schema identity; enforce it here.
- Optimistic extrapolation. Off-policy value learning on a narrow scripted
  dataset overestimates actions the scripted policy never took. This is what the
  LayerNorm regulariser in RLPD is for; monitor Q magnitudes against realised
  returns.
- Silent budget accounting. Offline episodes cost real wall clock. They must be
  counted in the budget of any candidate that uses them, or the comparison in
  section 7 is unfair by construction.

### 3.3 Rank 3 — Recurrent value-based agent, R2D2 skeleton

**What it is.** The architecture in `solution.md` section 9.3 as written: shared
row encoder, single-layer LSTM, dueling heads, sequence replay with stored
recurrent state and burn-in, prioritized by a mix of maximum and mean absolute
TD error. R2D2's contribution is the machinery for learning recurrent value
functions from replayed sequences: fixed-length overlapping sequences, stored
stale recurrent state, and a burn-in prefix that is unrolled without loss to
re-establish the state.

**Fit.** Mixed, and this is where this document disagrees with the current plan.
The machinery is correct and it is the right way to train a recurrent Q-function
from replay. But R2D2 is a large-budget distributed algorithm; its published
configuration runs 256 actors against billions of frames, and its
hyperparameters — low replay ratio, large batches, slow target updates, n-step
of 5 — are chosen for that regime. Adopting them at 10^5 to 10^6 decisions
imports the wrong operating point. The defensible position is to keep the
sequence-replay skeleton and apply the rank 1 optimisation recipe on top of it,
and to treat recurrence as a hypothesis to be tested against the stacked-history
baseline rather than as the default architecture.

Section 2.3 is the substantive reason for the demotion. The upgrade rows already
encode everything the agent did; the hidden state that remains has a short time
constant. Recurrent model-free RL is a strong POMDP baseline when memory is
actually required — Ni et al. match or beat specialised methods on 18 of 21
environments — but that is a statement about problems where memory is the
difficulty, which this one may not be.

**Expected sample efficiency (judgement).** At best slightly better than rank 1;
plausibly worse at this budget, because sequence replay reduces the effective
number of independent training targets per unit of data and adds
hyperparameters that cannot be afforded to tune. Treat "recurrence helps" as an
open question with a prior near 50 percent.

**Implementation cost and risk.** Moderate. Sequence replay, burn-in, stored
state, and correct handling of episode boundaries and truncation are the most
bug-prone parts of the whole project, and their bugs are quiet: they degrade
learning without crashing. Every one of them is unnecessary if stacking
suffices.

**Masking.** As rank 1, with the sequence complication that the mask is
per-timestep and must be stored and replayed alongside the observation for every
step of the sequence, including burn-in steps. Masking during burn-in does not
affect the loss but does affect the recurrent state, so it cannot be skipped.

**Partial observability.** This is the candidate's entire premise: a learned
recurrent state. If it wins over rank 1 it wins here.

**Failure modes to watch.**

- Representational drift between the stored recurrent state and the current
  parameters. Burn-in mitigates it; it does not eliminate it. At high replay
  ratios the parameters move faster relative to the data, so drift is worse here
  than in the published setting.
- Truncation handling. The termination taxonomy distinguishes `GAME_OVER` from
  `MAX_EPISODE_DURATION` and from environment failures. Bootstrapping through a
  truncation as if it were a terminal state teaches the agent that dying is
  free. This must be explicit and tested.
- Hyperparameter surface. Sequence length, burn-in length, overlap, and
  replay ratio interact. There is no budget to sweep them in the real
  environment, so they must be fixed by argument and held constant.
- Masquerading as progress. A recurrent agent that fails for recurrence-specific
  reasons is easily mistaken for a hard environment. This is exactly why the
  feed-forward floor must exist first.

### 3.4 Rank 4 — DreamerV3

**What it is.** A model-based agent that learns a recurrent latent world model
from experience and trains an actor and critic entirely on imagined rollouts in
that latent space, with symlog transforms, two-hot distributional value targets,
KL balancing with free bits, and normalisation choices that let one fixed
hyperparameter set work across domains. It was published in Nature in 2025 and
covers over 150 tasks with a single configuration, including proprioceptive and
discrete-action domains.

**Fit.** Two genuine strengths here and two genuine problems.

The first strength is robustness. DreamerV3's headline claim is that one
hyperparameter set works everywhere. At this budget, hyperparameter search in
the real environment is unaffordable — a ten-configuration sweep costs ten times
the sample budget — so an algorithm that does not need tuning is worth a large
amount of nominal sample efficiency. This is an underrated argument and it is
the main reason Dreamer appears above the search methods.

The second strength is that it addresses section 2.3 natively. The RSSM latent
is exactly a learned belief state, and imagined rollouts extract far more
gradient signal per real decision than one-step TD.

The first problem is nominal sample efficiency: as reported by the EfficientZero
V2 authors, DreamerV3 reaches an Atari 100k mean of 1.120 and median of 0.490,
against BBF at 2.247 and 0.917 and EZ-V2 at 2.428 and 1.286. At a
budget of this magnitude Dreamer is not the efficiency leader.

The second problem is specific to this environment and is discussed under
masking below: the actor is trained in imagination, where no oracle mask exists.

**Expected sample efficiency (judgement).** Comparable to rank 1 at the top of
the budget, with a better chance of working without tuning and a worse chance of
being the best result if tuning were affordable. Its world model has an easier
job here than on pixels — cash accrual and cost curves are close to
deterministic functions of the state — and a harder one than it looks, because
damage taken is where the stochasticity lives and damage taken is precisely what
is not observed.

**Implementation cost and risk.** High. A correct DreamerV3 is a substantial
piece of engineering, and a subtly incorrect one is hard to distinguish from a
hard environment. Reference implementations exist and should be used rather than
reimplemented. Its GPU cost is the highest of the model-free options but still
far below the 4090's capacity.

**Masking.** This is the decisive implementation question and it is frequently
handled badly. At acting time the oracle mask is available and is applied to the
actor's logits. In imagination there is no oracle: the world model must predict
the mask for every imagined latent state, or the actor will be trained to prefer
actions it can never take and the critic will evaluate a policy that does not
exist.

The mitigating fact is that the mask here is close to a deterministic function of
quantities the world model is already predicting: affordability is cash against
cost, plus `unlocked` and `maxed`. A dedicated mask head trained with binary
cross-entropy should be accurate. The requirements are that the mask head exists,
that predicted masks are applied inside imagination, that mask-head accuracy is
reported as a first-class training metric, and that the disagreement rate between
predicted and oracle masks at acting time is monitored. Without those, mask
error silently becomes policy error.

**Partial observability.** Handled natively by the recurrent latent. No stacking
required, although a short stack of run scalars as model input is harmless and
may speed up early model fitting.

**Failure modes to watch.**

- Mask drift in imagination, as above. Treat as the primary risk.
- Model exploitation. The actor finds imagined states with high predicted value
  that the real game never produces. Classic and hard to detect without
  comparing imagined returns against realised returns; log both.
- Under-modelled stochasticity. Damage is the unobserved stochastic driver. A
  world model that regresses to the mean of damage will systematically
  underestimate death risk and produce an over-aggressive spending policy, which
  is exactly the failure that looks like a plausible strategy.
- Reward sparsity in latent space. With wave change as the only reward, the
  reward head sees few positives; two-hot distributional targets help but the
  class imbalance is real.
- Budget mismatch. Dreamer's published settings assume much longer training. The
  ratio of gradient steps to environment steps must be raised deliberately, and
  that is a tuning decision the "no tuning needed" claim does not cover.

### 3.5 Rank 5 — EfficientZero V2 or Gumbel MuZero with Reanalyse

**What it is.** The MuZero family: learn representation, dynamics, reward, value
and policy functions jointly, and use Monte Carlo tree search in the learned
latent space both to act and to generate policy targets. EfficientZero V2 is the
current sample-efficiency leader on Atari 100k with a reported mean of 2.428 and
median of 1.286, supports low-dimensional inputs, and uses a sampling-based
Gumbel search with 32 simulations plus search-based value estimation. Gumbel
MuZero supplies the policy-improvement guarantee at small simulation budgets
through Gumbel top-k sampling and sequential halving. Reanalyse recomputes fresh
search-based policy and value targets on old trajectories, which is what makes
the family work across data budgets spanning orders of magnitude.

**Fit.** This is the candidate the pause capability is supposed to unlock, and
the honest assessment is that it is the highest-ceiling and highest-risk option,
with two structural reservations specific to Tower-RL.

The arguments in favour are real. The small valid action set is ideal for
search: with about six valid actions at the baseline, 32 simulations cover the
root's full action set several times over, which is precisely the regime where
Gumbel MuZero is strongest. Reanalyse is the single most relevant mechanism in
the entire literature for a project whose real experience is scarce and whose
GPU is idle: it converts spare compute into better targets on data already paid
for. And low-dimensional inputs are explicitly supported.

The reservations are in section 2.4. A few-ply tree at a one-second cadence sees
no reward, so search contributes policy improvement over a learned value rather
than lookahead; and stepped operation currently costs 3.4x the wall clock of
free running before planning time is added. Both of those change substantially
for the better if the cadence work in section 2.2 is done first, which is why
this candidate is sequenced last rather than dismissed.

**Expected sample efficiency (judgement).** Potentially the best of the six if it
is implemented correctly and if the cadence problem is addressed; plausibly the
worst in practice, because the probability of a subtly incorrect implementation
is the highest of any candidate and the budget does not permit the debugging
loop that a correct one requires. The published margin over BBF on Atari 100k
is real but modest relative to the complexity difference, and section 2.5
applies: part of that margin is pixel representation learning that does not
exist here.

**Implementation cost and risk.** Highest. MCTS in a learned latent space,
Reanalyse infrastructure, value and reward transforms, prioritisation, and the
masking work below. Use an existing implementation. Budget for the possibility
that it never reaches a trustworthy state, and make that decision explicitly at
a checkpoint rather than by attrition.

**Masking.** The hardest masking story of the six, and it must be designed, not
retrofitted.

At the root, the oracle mask is available and is applied to the prior and to
Gumbel top-k sampling. Because only about six actions are valid, sampling
without replacement at the root is close to enumeration, which is a genuine
advantage.

At internal nodes, there is no oracle mask, because internal nodes are learned
latents, not real states. Options are to learn a mask head on the dynamics latent
and apply it, or to allow the dynamics model to represent invalid actions as
no-ops and let the value function learn that they are worthless. The first is
correct and costs a head; the second corrupts visit-count policy targets and is
not recommended. Mask-head accuracy at depth should be reported per ply, because
error compounds with depth in a way it does not in Dreamer's flat imagination
rollouts.

Reanalyse adds a third requirement: re-running search on a stored trajectory
needs that trajectory's stored root mask. Masks must be persisted with replay
from the first day, regardless of which candidate is built first. That is cheap
insurance and it is recommended unconditionally.

**Partial observability.** MuZero-family agents conventionally feed a stack of
recent observations and actions into the representation function, and EZ-V2 does
the same for low-dimensional inputs. The same stacked-window answer as rank 1
applies; the dynamics latent then carries state forward within the tree.

**Failure modes to watch.**

- Deterministic dynamics against a stochastic game. Standard MuZero learns a
  deterministic latent transition. Enemy spawns, damage and critical hits are
  stochastic. A deterministic model averages them, which systematically
  underestimates variance and therefore death risk. Stochastic MuZero's
  afterstate factorisation exists for exactly this; if the deterministic variant
  is used, this is the first thing to suspect when the policy is over-aggressive.
- Value-only search. If the tree never reaches a reward, search quality is
  entirely a function of value-function quality, and the search adds compute
  without adding information. Diagnose by comparing the search's recommended
  action against the raw policy prior; if they rarely disagree, the search is not
  earning its cost.
- Wall-clock collapse. The 430 ms stepped-mode overhead plus per-decision
  planning can turn a nine-hour run into a multi-day run. Measure decisions per
  wall-clock hour before committing.
- Silent masking corruption at depth, as above.
- Implementation risk generally. This candidate is the one where "it does not
  work" is most likely to mean "it is wrong" rather than "it does not fit".

### 3.6 Rank 6 — Masked recurrent PPO

**What it is.** On-policy actor-critic with clipped surrogate objective, GAE, an
LSTM or GRU over the observation sequence, truncated backpropagation through
time, and invalid-action masking applied to the logits before the softmax.

**Fit.** Poor as a contender, useful as an instrument. PPO's sample efficiency is
the worst of the six by a wide margin: it discards data after one or a few epochs
and its published successes in comparable discrete domains assume 10^7 to 10^9
steps. At 10^5 to 10^6 decisions it is being asked to work two to four orders of
magnitude below its comfortable operating point. Recurrent PPO is worse still,
because truncated BPTT reduces the number of independent gradient signals per
sample further.

Its value is different. It is the cheapest correct implementation in the list,
it is a different algorithm family from everything above it, and it therefore
fails in different ways. If PPO and the rank 1 agent both fail to beat the
always-wait floor, the environment or the reward is the suspect. If PPO learns
something and the value-based agent does not, the value-based implementation is
the suspect. That cross-check is worth having, and it is worth having cheaply.

**Expected sample efficiency (judgement).** Should beat always-wait. Unlikely to
approach the scripted greedy policy within budget. Do not rank it on outcome; if
it is included in the headline comparison it should be labelled as a reference
point rather than as a competitor, so that beating PPO is not mistaken for a
result.

**Implementation cost and risk.** Lowest of all, with mature masked
implementations widely available.

**Masking.** Set invalid logits to negative infinity before the softmax. Huang
and Ontañón show that this is not a heuristic: the masked policy gradient remains
a valid gradient of the masked policy, and masked entries receive zero gradient.
Penalty-based alternatives scale badly as the invalid set grows, which is
decisive here, where 54 of 60 upgrade actions are invalid at the baseline. A 2026
analysis strengthens the case further by showing that unmasked training
systematically suppresses valid actions at unvisited states through shared
parameters, with exponential decay, and that entropy regularisation only trades
that suppression against sample efficiency while masking removes the trade-off.

Two PPO-specific subtleties matter more than usual here. First, the entropy bonus
must be computed over the valid set, and its maximum value is log of the number
of valid actions, which varies from about 1 to 61 across states. A fixed entropy
coefficient therefore applies a state-dependent pressure and quietly rewards
being in states with many affordable upgrades. Normalise the entropy term by
log of the valid-action count, or accept and document the bias. Second, the mask
used at update time must be the mask that was stored at rollout time; recomputing
it can differ and silently invalidates the importance ratio.

**Partial observability.** Prefer the stacked window over recurrence for the same
reasons as rank 1, and more strongly: recurrent PPO's additional machinery is the
least likely part of this list to pay for itself at this budget.

**Failure modes to watch.**

- Entropy collapse onto `WAIT`, then no recovery, because on-policy methods
  cannot revisit discarded data.
- Advantage estimation across very long episodes: GAE over 500 steps with reward
  at fewer than ten points is high variance regardless of lambda.
- Batch size against episode count. A PPO batch of a few thousand steps is a
  handful of episodes, so gradient noise is dominated by the episode-outcome
  variance documented in section 2.1.
- Being used as the success criterion. It is a control, not a target.

### 3.7 Components that are modifiers, not candidates

Three items named in the brief are best understood as modifiers to the ranks
above rather than as separate backbones.

**Munchausen.** Adding a scaled log-policy term of the current state's action to
the reward implicitly performs KL regularisation between successive policies and
increases the action gap, with a reported median improvement of 45 percent over
DQN across 53 of 60 Atari games. It is cheap and applies to ranks 1, 2 and 3.
Masking interacts with it specifically: the log-policy term comes from a softmax
over Q-values and that softmax must be taken over valid actions only, the term
must be clipped below as the paper prescribes, and — a subtlety with no
published guidance — when the valid set changes between consecutive states the
implicit KL is between distributions over different supports. Treat Munchausen
as an ablation switch, not as part of the baseline, and measure it.

**SPR and self-predictive auxiliary losses.** SPR raised the Atari 100k median
from a previous state of the art to 0.415 using latent future prediction plus
image augmentation, and its descendants SR-SPR and BBF build on it. Judgement:
the augmentation half does not exist here and the representation half is much
less valuable when the observation is already an exact structured state. A latent
transition-prediction auxiliary loss over the upgrade-row encoding might still
regularise usefully, but expecting SPR-sized gains would be unwarranted. Low
priority ablation.

**Periodic resets and high replay ratio.** Resets with a preserved replay buffer
are the standard fix for primacy bias at high replay ratios, and shrink-and-
perturb at 50 percent is what lets BBF run a replay ratio of 8 with a large
network. But this is not unconditional: SimbaV2 reports that hyperspherical
normalisation scales smoothly with update-to-data ratio without resets, and that
resetting can degrade its performance. Since resets deliberately induce periodic
performance drops, and since evaluation here is expensive, treat resets as a
tested option rather than a default, and pair the test with a normalisation
alternative.

## 4. Masking rules that apply regardless of candidate

These are correctness requirements, not tuning choices. Most are cheap to
implement and expensive to discover later.

1. Persist the mask with every stored transition, and persist the next state's
   mask as well. Value targets take a maximum over the next state's valid set.
2. Persist the mask in offline and demonstration datasets from the first
   logged episode, even before any learner exists.
3. Exclude masked entries from the loss rather than assigning them a large
   negative target. A finite sentinel becomes a regression target.
4. Apply the mask at four places, not one: action selection, the bootstrapped
   target, any softmax or entropy computation, and any imagined or planned
   state.
5. For dueling architectures, subtract the mean advantage over valid actions
   only. Averaging over all 61 makes the state-value head absorb a term that
   varies with how many actions happen to be affordable.
6. Make exploration mask-aware. Epsilon-greedy must sample uniformly among valid
   actions. Note that the valid-set size varies with cash, so a fixed epsilon
   has a state-dependent meaning; consider reporting effective exploration as a
   function of valid-set size.
7. For any learned model — world model or dynamics function — add an explicit
   mask-prediction head, apply it inside imagination or search, and report its
   accuracy and its disagreement rate against the oracle mask as first-class
   metrics.
8. Use the stored rollout-time mask at update time; never recompute it.
9. Test that the agent has never emitted an invalid action, as an assertion in
   the environment rather than a metric. The contract already has an
   `unavailable` outcome; a nonzero count is a bug, not a statistic.
10. Log the masked-action frequency and the valid-set size distribution. A
    change in the mask distribution across runs is a baseline-drift signal
    before it is a learning signal.

## 5. Partial observability: recommended default and the ablation that settles it

Default: a stacked window of the last k run-scalar vectors — wave, cash, health
fraction, max health, game speed, last action, last outcome, elapsed real
seconds — concatenated with the current upgrade-row set, with k in the range 4
to 16. Rationale in section 2.3: the build is fully observed, so what memory
must supply is a short-horizon threat estimate, and a stacked window supplies it
without sequence replay.

The ablation that settles it is cheap and should be run early: hold everything
else fixed, sweep k over {1, 4, 8, 16} with the rank 1 agent, and additionally
run the rank 3 recurrent agent at matched optimisation settings. If k = 1 is as
good as k = 16, the environment is more Markovian than assumed and both the
stacking and the recurrence are unnecessary. If performance rises with k and the
recurrent agent does not beat the best k, recurrence is not earning its cost.
Only if the recurrent agent beats the best stacked window does `solution.md`'s
LSTM justify itself.

Note that this ablation is itself expensive: four values of k at three seeds is
twelve runs. It is still cheaper than discovering after six weeks that the
recurrent machinery was never needed.

## 6. Recommended implementation order

The ordering principle is to establish a trustworthy floor before spending
budget on anything that can fail for subtle reasons, and then to order the
remaining work by information gained per unit of engineering risk.

**Step 0 — measure the environment's own variance before building a learner.**

The first experiment should not be an algorithm. Run the scripted greedy policy
for at least 50 episodes and the always-wait policy for at least 20, at the
fixed baseline and fixed speed, and report the full distribution of final waves,
not just the mean. Everything in section 7 depends on the standard deviation of
final wave, and the only estimate available today comes from three episodes.
This costs about one hour of four-actor time and it determines whether any
comparison in this document is affordable at all.

Also measure actor scaling — one, two, four actors — and the current stepped-mode
overhead, since section 2.1 and section 2.4 both rest on assumptions.

**Step 1 — rank 1, the feed-forward masked DQN.** This is the floor. It also
builds every piece of shared infrastructure: replay with masks, evaluation at
epsilon zero, checkpointing, metrics, and the masking test suite from section 4.
Nothing later is trustworthy until this exists and beats always-wait.

**Step 2 — the cadence experiment from section 2.2.** With the rank 1 agent
fixed, compare the current one-second cadence against a coarser fixed cadence
and against the event-triggered mode. Judgement: this is the highest-expected-
value experiment in the plan, because it changes the difficulty of the problem
for every candidate simultaneously, and it is much cheaper than any new
backbone. If it works, re-run step 1 at the new cadence and use that as the
floor.

**Step 3 — rank 2, offline bootstrapping.** Cheap, high expected gain, and it
reuses step 0's logged baseline episodes at no additional environment cost. It
also produces the first honest answer to "is there headroom above the scripted
policy at this baseline", which is a question that could invalidate the entire
programme and should therefore be asked early.

**Step 4 — the partial-observability ablation from section 5, including rank 3.**
This settles whether the recurrent architecture in `solution.md` is justified. It
is placed here rather than earlier because it is only interpretable once the
floor and the cadence are fixed.

**Step 5 — rank 4, DreamerV3.** The first genuinely different hypothesis: that a
learned model extracts more from each real decision than replay does. High cost,
high information. Use a reference implementation. Gate entry on steps 1 to 4
having produced a stable floor, because a Dreamer result is uninterpretable
without one.

**Step 6 — rank 5, EfficientZero V2 or Gumbel MuZero, conditionally.** Enter only
if all three preconditions hold: the cadence work in step 2 shortened the
effective horizon, the stepped-mode overhead was profiled down to something
comparable to free running, and steps 1 to 5 left enough calendar time. Otherwise
record the decision not to build it and why. This is a deliberate decision to
make once, not a task to abandon gradually.

**Throughout, rank 6 (masked PPO) as an optional independent cross-check.** Build
it if and only if the rank 1 agent behaves in a way that is hard to attribute to
either the environment or the learner. It is a diagnostic instrument and should
be budgeted as one.

## 7. Comparison protocol

This section is deliberately the most conservative part of the document. At this
sample size the likeliest outcome of a careless protocol is a confident wrong
conclusion, and the cost of that is weeks of work spent on the wrong backbone.

### 7.1 The budget unit to equalise

Equalise on **environment decisions consumed**, and report wall clock and
episode count alongside it. Never compare at equal episode counts.

The reasoning: at a fixed decision cadence, decisions are proportional to game
time and therefore to the real resource being consumed. Episodes are not a valid
unit, because a better agent survives longer and consumes more decisions per
episode — equalising episodes hands the stronger agent a larger real budget.
Wall clock is the true engineering constraint but is not a fair scientific unit,
because it is contaminated by implementation efficiency, host contention and
emulator scheduling.

There is one important exception that makes reporting wall clock mandatory
rather than optional: any method that pauses to deliberate consumes wall clock
without consuming game time. Section 2.4 measured that penalty at about 3.4x
today. Equalising only on decisions would grant search methods a free pass on
their real cost. Report both, and state explicitly which one a given conclusion
rests on.

Budget accounting must include every environment decision the method consumed,
including episodes logged for offline bootstrapping and every episode spent on
hyperparameter exploration in the real environment. An algorithm tuned over ten
configurations has spent ten times the budget of one that was not.

### 7.2 How many seeds are realistic

Take 10^6 decisions per run, four actors, and section 2.1's throughput: about
nine hours of continuous emulation per training run, plus evaluation.

- Five seeds per candidate is the minimum that supports any run-level inference,
  and is what the small-sample evaluation literature recommends as the practical
  floor with three to ten runs.
- Five seeds across six candidates is roughly 270 hours of training, about
  eleven days continuous, before evaluation.
- Three seeds is what is likely to be affordable for the expensive candidates.
  The consequence must be stated rather than hidden: with three runs and the
  run-to-run variance typical of deep RL, only large effects are detectable, and
  a null result is not evidence of equivalence.

Practical recommendation: five seeds for ranks 1 to 3, three seeds for ranks 4
and 5, and a pre-committed rule that any candidate whose three-seed interval
overlaps the floor is not promoted to five seeds. Record the seed count with
every reported number, and never compare a five-seed candidate's best run
against a three-seed candidate's mean.

Power for the evaluation stage has since been measured, and the estimate this
section originally carried was wrong by roughly a factor of six. Step 0 was run
in `M1B-E006`: fifty episodes of the scripted policy give a final-wave standard
deviation of 1.22 around a mean of 9.74, not the roughly 3 around 7 guessed from
three episodes.

At 80 percent power and alpha 0.05, detecting a one-wave difference needs about
23 evaluation episodes per arm rather than 140, roughly eight minutes at the
observed throughput; two waves needs about six episodes. Half a wave needs about
94 per arm, and is the point below which a difference should not be claimed
cheaply. The original estimate is left described here rather than deleted,
because the lesson it carries — that a variance guessed from three samples can be
off by a large factor and should never be planned against — is the reason step 0
exists.

### 7.3 What to report

- Learning curves against environment decisions, with the run as the resampling
  unit and a stratified bootstrap confidence interval across runs.
- Final performance as the mean and the median final wave over a pre-registered
  number of evaluation episodes at epsilon zero, at a pre-registered checkpoint
  selection rule.
- The full distribution of final waves, as a histogram or survival curve, not
  only a central tendency. Final wave at this baseline takes a small number of
  integer values with a floor at wave 2 and a right tail; means conceal both.
- Every run's individual point estimate, so that readers can see the spread the
  aggregate hides.
- Wall clock and episode count alongside decisions, per section 7.1.
- Purchase rate, action distribution, and masked-action frequency, so that a
  degenerate `WAIT` policy is visible as such rather than as a weak result.
- Tuning cost: how many configurations were tried in the real environment, and
  how many environment decisions that consumed.
- The exact profile identity: game version, speed profile, baseline snapshot,
  schema versions. Results are comparable only within a profile.

Interquartile mean is the right aggregate when many tasks are involved, because
it is robust and statistically efficient across a task-by-run matrix. Tower-RL
has one task. Applying IQM across episodes of one task is not the same procedure
and does not inherit the same justification, so report it, if at all, as a
descriptive statistic about the episode distribution and keep the run-level
stratified bootstrap as the inferential tool.

### 7.4 Statistical traps at this sample size

1. **Pseudo-replication.** The most likely single error. Evaluating one
   checkpoint for 100 episodes and treating those as 100 independent samples of
   the algorithm estimates that checkpoint's mean, not the algorithm's. Run-level
   variance in deep RL typically dominates episode-level variance. The
   resampling unit for any claim about an algorithm is the training run.
2. **Checkpoint cherry-picking.** Taking the maximum over many noisy checkpoint
   evaluations is biased upward, and the bias grows with the number of
   checkpoints evaluated. Pre-register the selection rule — for example, the
   last checkpoint, or the best on a selection set — and evaluate the selected
   checkpoint on held-out episodes that played no part in the selection.
3. **Multiple comparisons.** Six candidates gives fifteen pairwise comparisons.
   At alpha 0.05 the family-wise error probability is large. Either correct for
   it or use a method designed for expensive sequential RL comparisons, such as
   AdaStop's group sequential tests, which adapt the number of runs while
   controlling family-wise error.
4. **Uncounted tuning budget.** Covered in section 7.1; it is listed again
   because it is the trap that most often produces a wrong ranking that survives
   review.
5. **A discrete, floored, heavy-tailed metric.** Final wave is an integer with a
   hard floor at the always-wait outcome. Many policies will sit on that floor,
   which produces a point mass that breaks normality assumptions and makes
   t-tests and normal confidence intervals unreliable. Use nonparametric methods
   and report the distribution.
6. **Temporal confounding on real hardware.** This is the trap the RL literature
   rarely discusses and it is severe here. Running candidate A for two days and
   then candidate B for two days confounds the algorithm with host thermal
   state, background load, emulator version, game version, account drift and
   anything else that changes with time. Interleave candidates and seeds in
   time, randomise the order, and record host conditions per run. A drift check
   — re-running the scripted baseline periodically and confirming it still
   produces the same final-wave distribution — is the cheapest available
   insurance and should be part of the protocol, not an afterthought.
7. **Non-neutral episode exclusion.** The termination taxonomy makes it easy to
   drop episodes. Dropping `OBSERVATION_INVALID` or `DEVICE_FAILED` is neutral.
   Dropping `MAX_EPISODE_DURATION` is not: those are the longest-surviving
   episodes, so excluding them truncates the right tail and penalises exactly
   the agents the project is trying to find. Decide the rule in advance, apply
   it identically to every candidate, and report the exclusion counts.
8. **Peeking and informal early stopping.** Watching curves and stopping a run
   that "looks bad" converts the comparison into an unquantified sequential test.
   If early stopping is wanted, use a method with explicit error control.
9. **Correlated seeds.** Seeds that share a replay buffer, a snapshot, a
   pretrained encoder or an offline dataset are not independent. Offline
   bootstrapping in rank 2 makes this concrete: all seeds share one demonstration
   dataset, so the dataset's idiosyncrasies are common-mode. Note it, and if
   affordable, split the demonstration data across seeds.
10. **Regression to the mean on the winner.** The candidate that wins a noisy
    comparison is, in expectation, overestimated. Re-evaluate the winner on a
    fresh set of episodes before it becomes the project's baseline, and expect
    the number to come down.
11. **Confusing a learning metric for a result.** TD error, loss and Q magnitude
    say nothing about final wave. `solution.md` already states this; it is worth
    restating because at this sample size the temptation to substitute a
    low-variance proxy for a high-variance outcome is strong.

### 7.5 Allocating a scarce evaluation budget

Evaluation episodes cost the same wall clock as training episodes, so evaluation
must be budgeted, not assumed. When several checkpoints or candidates compete for
a fixed number of evaluation episodes, this is a fixed-budget best-arm
identification problem, and sequential halving — evaluate all arms equally,
discard the worst half, repeat — is a near-optimal and very simple allocation.
It is strictly better than evaluating every arm to the same large depth, and it
composes with the pre-registration rule in trap 2 as long as the halving schedule
is fixed in advance.

## 8. Recent work the shortlist does not otherwise cover

### 8.1 Architecture for low-dimensional inputs

Nearly all of the sample-efficiency literature the brief names is pixel-based.
The line most relevant to a structured, low-dimensional observation is SimBa and
SimbaV2, which study what network design lets off-policy RL scale on
proprioceptive inputs. SimbaV2 reports 0.848 normalised return at an
update-to-data ratio of 1, above Simba at 0.818 and BRO at 0.807 with a ratio of
8, and above TD-MPC2 at 0.749. Two findings transfer directly: normalisation
choices matter more than depth on low-dimensional inputs, and hyperspherical
normalisation scales with update-to-data ratio without resets, with resets
actively degrading it. That is a concrete reason not to copy BBF's reset schedule
uncritically into rank 1.

### 8.2 Reusing prior computation rather than starting from scratch

Reincarnating RL frames tabula rasa training as a research convention that large
real systems do not follow, and studies transferring an existing suboptimal
policy into a value-based agent, including on a genuine real-world problem. RLPD
provides the concrete online mechanism. Together they are the literature behind
rank 2, and they are the most directly applicable body of work to a project that
already has a scripted policy reaching wave 7 to 10.

### 8.3 Plasticity loss

The 2024 survey of plasticity loss in deep RL catalogues more than fifty
mitigations and reports that general regularisation techniques often outperform
domain-specific interventions, with LayerNorm and weight decay doing much of the
work. This matters here because high replay ratios on a small buffer are exactly
the regime that induces plasticity loss, and because it gives an alternative to
resets that does not induce periodic performance drops during an already
expensive evaluation schedule.

### 8.4 Stochastic dynamics in learned models

Stochastic MuZero replaces the deterministic latent transition with an afterstate
factorisation — deterministic state to afterstate, stochastic afterstate to next
state — because deterministic models limit performance in environments that are
inherently stochastic or partially observed. Tower-RL is both. If rank 5 is ever
built, this is the variant to build.

### 8.5 World models over symbolic observations

The Atari-100k world-model line has recently extended to mixed symbolic and
continuous observations. Simulus, a 2026 token-based world-model agent from
Technion, Microsoft Research and ByteDance, combines a tokenisation framework
over arbitrary observation and action modalities, uncertainty-driven intrinsic
motivation, prioritized world-model replay and regression-as-classification, and
reports results on Craftax-1M as well as Atari 100k and DMC proprioception. It is
the closest published setting to a structured, symbolic observation like this
one, and it is worth reading before committing to rank 4's architecture.

### 8.6 What actually happens when these methods meet real hardware

A 2026 empirical study of reset-free RL on a physical 1/10-scale vehicle compared
PPO, SAC and TD-MPC2 learning continuously without manual resets. In simulation
SAC with residual learning gave the highest returns; on the real platform that
advantage did not transfer, and only TD-MPC2 consistently beat the classical MPPI
baseline. Residual learning, clearly beneficial in simulation, degraded real
performance. The transferable lesson is not about any of those three algorithms;
it is that simulation rankings do not survive contact with real hardware, which
is the central reason this document ranks by risk rather than by published score.

The wider framing comes from the real-world RL challenge literature, which
formalises the properties — limited samples, partial observability, safety
constraints, non-resettability, delayed and sparse reward — that separate
deployable RL from benchmark RL, and provides a vocabulary for saying which ones
apply here.

### 8.7 Masking, revisited by recent work

Beyond the standard result that masking preserves a valid policy gradient, a 2026
analysis argues the case is stronger than previously understood: in unmasked
training, gradients applied to invalid actions at visited states propagate
through shared parameters and suppress those same actions at unvisited states
where they are valid, with an exponential decay characterisation for softmax
policies, and entropy regularisation only trades that suppression against sample
efficiency. With 54 of 60 upgrade actions invalid at the baseline and the valid
set expanding as the run progresses, this describes a failure this project would
otherwise be exposed to directly: actions that are invalid early and valid later
would be suppressed precisely when they become available.

### 8.8 What is genuinely missing from the literature for this problem

Stated plainly, because pretending otherwise would be the most damaging thing
this document could do:

- There is no published sample-efficiency result for a structured,
  low-dimensional, hard-masked, 61-action, several-hundred-step real-time mobile
  game at 10^5 to 10^6 decisions. Every ranking here is extrapolation.
- There is very little work on semi-MDP effects from variable real decision
  intervals in this kind of setting; `solution.md` section 7.3 already flags
  time-aware discounting as an open question and the literature will not settle
  it.
- There is almost nothing on off-policy checkpoint selection under an evaluation
  budget measured in hours of real play, which is why section 7.5 borrows from
  the bandit literature instead.
- The effect of decision cadence on effective horizon — section 2.2, the largest
  lever identified in this document — is a domain-engineering decision that the
  algorithm literature does not address at all.

## 9. Summary of the ranking

This ranking was re-cut on 2026-09-19 for the regime the project actually
runs in: choice-point cadence (ADR 0009), `observation-v2`, a budget of
360,000 game-seconds ≈ 40,000 decisions ≈ 1,400 episodes, 20.7–27.5 decisions
an episode and 3.2–5.0 a wave (`M2-E007`), a mean legal set at a choice
point of ≈4.2 actions including WAIT (`M2-E004`; ≈2.0 averaged over every
every-slice state, 68% of which were WAIT-only — ≈4.2 is the branching
factor that applies under ADR 0009's choice points), and a
learner that is ~3% busy. Sections 2.2 and 2.4 were written before the cadence
work landed and their conclusions no longer follow; section 3.5's demotion of
the MuZero family rested on section 2.4 and is reversed here.

| Rank | Candidate | Strongest evidence | Fit to this regime | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | **Finish the BBF recipe on the existing `stacked-dqn`** — weight decay 0.1, width, shrink-and-perturb resets, 10→3 n-step and 0.97→0.997 γ anneals, prioritized replay on | Atari-100k IQM 1.045 at RR 8; +0.45 IQM over SR-SPR at *every* replay ratio; every component validated on 29 held-out ALE games (Schwarzer 2023) | Exact masking; replay ratio already between 54:1 and 126:1 depending on the counting convention (see #53), so what is missing is the regularisation and capacity, not the gradient count | 3–5 d | Low |
| 2 | **Offline bootstrap from the logged baselines** (RLPD symmetric sampling + LayerNorm value net) | RLPD reports up to 2.5× from symmetric sampling with no pretraining and no offline-RL constraint (Ball 2023) | Costs **no new device time**: 223 valid baseline episodes already exist from `M2-E007`. Highest expected gain per device-hour in the list, and still never built | 2–3 d | Low |
| 3 | **Gumbel MuZero / EfficientZero-V2 with full Reanalyse**, via LightZero (Apache-2.0) — *the different-paradigm candidate* | EZ-V2 Proprio Control **50k**: mean 723.2 vs DreamerV3 517.1 and SAC 552.0, TD-MPC2 740.9 (Wang 2024) — the only published vector-observation result at our budget. Atari-100k mean 2.428 / median 1.286 | Good structural fit: at the choice-point branching factor of ≈4.2, 32 simulations are exhaustive to a depth of ~2 to 3 (4^k ≤ 32 gives k ≈ 2.5) — roughly half to three-quarters of a wave (3.2–5.0 decisions, `M2-E007`), not a full wave; `action_mask` is first-class in LightZero's env dict; Reanalyse converts idle compute into better targets on scarce data, which is exactly this project's asymmetry | 15–20 d | High |
| 4 | **DreamerV3** (NM512/dreamerv3-torch, MIT; danijar/dreamerv3 JAX, MIT) | One hyperparameter set over 150+ tasks (Nature 2025); but **weakest of the three** on Proprio Control 50k at 517.1, and Atari-100k mean 1.120 / median 0.490 | Robustness is still its real argument, and tuning is unaffordable here. Against it: no native masking, a bespoke mask head in imagination with no reference, and the worst low-dimensional number of the model-based options | 8–12 d | High |
| 5 | Recurrent value-based agent, R2D2 skeleton | Ni et al. 2022 | Further demoted: an episode is now ~27 decisions and `history_length=8` already spans a quarter of one | 5–8 d | Moderate |
| 6 | Masked PPO | Huang & Ontañón 2022 | Control instrument only, unchanged | 2 d | Lowest |

**Not candidates.** SimbaV2, BRO and TD-MPC2 are continuous-action methods
(SimbaV2: 57 continuous-control tasks). Their transferable content is
normalisation architecture for low-dimensional inputs and belongs to rank 1 as
an ablation. The 2026 world-model line (Simulus, EAWM/EASimulus) is pixel
Atari-100k; it is worth reading before rank 3 or 4 is built and is not itself a
candidate here.

**Three decisions recommended to the Lead.** First, that "raise the replay
ratio because compute is free" be retired as a proposal: `training.py`
already runs between 54 and 126 transitions replayed per transition
generated, depending on the counting convention (see #53) — on the order of
SPR's 64 to BBF's 256. Second, that the MuZero family be promoted above
DreamerV3, because the cadence change removed the objection in section 2.4
and because EZ-V2 is the only entry on this list with a published
low-dimensional result at our budget. Third, that `M2-E004`'s absent
plasticity signature not be read as permission to raise width or weight decay
without re-measuring it — it was measured in the regime BBF's resets exist to
leave.

## 10. Run-3 ablation order

From the audit (#53), in the order run 3 should test each item, each
independently measurable against the M2-P002 baselines (random 5.495,
scripted 6.429):

1. **ε schedule + head init.** Fixed Ape-X ladder from step 0 (or anneal on
   gradient steps, starting at warm-up) plus zero-init of the final `Linear`
   layer of `value_head`, `wait_advantage` and `row_advantage`. Expected
   effect: removes the near-constant, WAIT-biased greedy policy that actors
   3–6 play for most of the anneal — the largest single deviation measured
   (D1, #53).
2. **n-step.** 10 → 3 (Ape-X/Rainbow), or the BBF 10→3 anneal already
   specified in section 3.1. Expected effect: removes the off-policy bias in
   the uncorrected n-step bootstrap and raises the action-attributable share
   of the return (D2, #53).
3. **Replay ratio.** 0.25 → 2.0 gradient steps per decision, the value
   section 3.1 specifies and run 1 used. Expected effect: more gradient steps
   behind the network the kill check reads at matched game-seconds; costs
   ~26% of actor wall time (D3, #53).
4. **Reward shaping, as a separate arm.** Potential-based Φ = clipped
   within-wave kill fraction, Φ(terminal) = 0 (Ng, Harada & Russell 1999).
   Developer decision: the benchmark reward stays +1/wave; shaping is an
   ablation only, not a change to the optimised metric. Expected effect:
   denser signal at decision cadence and a derived death penalty, provably
   policy-invariant under this n-step learner (audit #53, Part 3).
5. **Adam ε.** 1e-8 → 1e-3 (R2D2's, matching the lr already used). Expected
   effect: cheap; targets the smallest gradients, which sit in the advantage
   heads (D5, #53).
6. **PER.** `priority_alpha` 0 → 0.6 with the existing β anneal. Expected
   effect: moderate; oversamples the high-|TD| terminal/death transitions
   currently sampled at the background rate (D4, #53).
7. **History length.** k ∈ {1, 4} against 8 — the ablation section 3.1
   already names and that has never been run. Expected effect: settles
   whether the stacked window buys anything at this near-fully-observed
   cadence; not checkpoint-compatible across k (D7, #53).
8. **Masked pooling.** Mean-pool the trunk over `unlocked` rows instead of
   all 60. Expected effect: second-order; removes a channel that is ~88%
   constant given only ~7 of 61 actions are ever valid (D6, #53).
9. **BBF block.** Weight decay 0.1, shrink-and-perturb resets, and the n-step
   and γ anneals together, run as one arm (R3, SOTA-BACKBONES §4). Expected
   effect (pre-registered): post-anneal near-greedy mean final wave ≥ 6.4 (the
   scripted floor) by the third checkpoint; falsified by no improvement over
   run 2's curve at matched game-seconds. Gated on run 2's verdict.
10. **MuZero-family arm.** Gumbel MuZero with full Reanalyse, built behind the
    existing `Backbone` protocol (R6, SOTA-BACKBONES §4). Expected effect
    (pre-registered): beats the best model-free arm at matched game-seconds;
    falsified by failing to do so, or by the search's recommended action
    agreeing with the raw prior on >95% of decisions, which at the
    choice-point branching factor of ≈4.2 would mean the search buys nothing.
    Gated on run 2's verdict.

Items 1–3 are confounded with each other if run together; item 4 is a reward
schema change (`reward-v2`) and must be its own arm.

## 11. Sources

- Kapturowski et al., Recurrent Experience Replay in Distributed Reinforcement
  Learning (R2D2), ICLR 2019.
  https://openreview.net/forum?id=r1lyTjAqYX
- Schwarzer et al., Data-Efficient Reinforcement Learning with Self-Predictive
  Representations (SPR), ICLR 2021. https://arxiv.org/abs/2007.05929
- Schwarzer et al., Bigger, Better, Faster: Human-level Atari with Human-level
  Efficiency (BBF), ICML 2023. https://arxiv.org/abs/2305.19452 and
  https://proceedings.mlr.press/v202/schwarzer23a.html
- Nikishin et al., The Primacy Bias in Deep Reinforcement Learning, ICML 2022.
  https://arxiv.org/abs/2205.07802
- Vieillard et al., Munchausen Reinforcement Learning, NeurIPS 2020.
  https://arxiv.org/abs/2007.14430
- Hafner et al., Mastering Diverse Domains through World Models (DreamerV3),
  arXiv 2023. https://arxiv.org/abs/2301.04104
- Hafner et al., Mastering diverse control tasks through world models, Nature
  640:647-653, 2025. https://www.nature.com/articles/s41586-025-08744-2
- Wang et al., EfficientZero V2: Mastering Discrete and Continuous Control with
  Limited Data, ICML 2024. https://arxiv.org/abs/2403.00564 and
  https://arxiv.org/html/2403.00564v2
- Ye et al., Mastering Atari Games with Limited Data (EfficientZero), NeurIPS
  2021. https://arxiv.org/abs/2111.00210
- Danihelka et al., Policy Improvement by Planning with Gumbel, ICLR 2022.
  https://iclr.cc/virtual/2022/poster/6418
- Antonoglou et al., Planning in Stochastic Environments with a Learned Model
  (Stochastic MuZero), ICLR 2022. https://openreview.net/pdf?id=X6D9bAHhBQ1
- Schrittwieser et al., Online and Offline Reinforcement Learning by Planning
  with a Learned Model (MuZero Reanalyse / Unplugged), NeurIPS 2021.
  https://arxiv.org/abs/2104.06294
- Huang and Ontañón, A Closer Look at Invalid Action Masking in Policy Gradient
  Algorithms, FLAIRS 2022. https://arxiv.org/abs/2006.14171
- Zabounidis et al., Overcoming Valid Action Suppression in Unmasked Policy
  Gradient Algorithms, arXiv, March 2026. https://arxiv.org/abs/2603.09090
- Ni, Eysenbach and Salakhutdinov, Recurrent Model-Free RL Can Be a Strong
  Baseline for Many POMDPs, ICML 2022. https://arxiv.org/abs/2110.05038
- Ball et al., Efficient Online Reinforcement Learning with Offline Data (RLPD),
  ICML 2023. https://arxiv.org/abs/2302.02948
- Agarwal et al., Reincarnating Reinforcement Learning: Reusing Prior
  Computation to Accelerate Progress, NeurIPS 2022.
  https://arxiv.org/abs/2206.01626
- Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical
  Precipice, NeurIPS 2021. https://arxiv.org/abs/2108.13264 and
  https://github.com/google-research/rliable
- Colas, Sigaud and Oudeyer, How Many Random Seeds? Statistical Power Analysis
  in Deep Reinforcement Learning Experiments, arXiv 2018.
  https://arxiv.org/abs/1806.08295
- Mathieu et al., AdaStop: Adaptive Statistical Testing for Sound Comparisons of
  Deep RL Agents, TMLR 2024. https://arxiv.org/abs/2306.10882
- Lee et al., SimBa: Simplicity Bias for Scaling Up Parameters in Deep
  Reinforcement Learning. https://openreview.net/forum?id=jXLiDKsuDo
- Lee et al., Hyperspherical Normalization for Scalable Deep Reinforcement
  Learning (SimbaV2), arXiv 2025. https://arxiv.org/abs/2502.15280
- Klein et al., Plasticity Loss in Deep Reinforcement Learning: A Survey, arXiv
  2024. https://arxiv.org/abs/2411.04832
- Cohen et al., Simulus: Combining Improvements in Sample-Efficient World Model
  Agents, arXiv 2025-2026. https://arxiv.org/abs/2502.11537
- Honda and Hosogaya, Reset-Free Reinforcement Learning for Real-World Agile
  Driving: An Empirical Study, arXiv, April 2026.
  https://arxiv.org/abs/2604.07672
- Dulac-Arnold et al., An Empirical Investigation of the Challenges of
  Real-World Reinforcement Learning, arXiv 2020.
  https://arxiv.org/abs/2003.11881
- Karnin, Koren and Somekh, Almost Optimal Exploration in Multi-Armed Bandits
  (sequential halving), ICML 2013.
  https://proceedings.mlr.press/v28/karnin13.html
