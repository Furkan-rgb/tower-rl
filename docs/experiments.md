# Tower-RL — Experiments and Evidence

This document records feasibility work, benchmarks, failed approaches, and
contrary evidence. An entry records what was observed; it does not advance a
milestone unless the corresponding gate in `task.md` is satisfied.

Do not add proprietary package bytes, extracted assets, account/save state,
personal screenshots, bulk logs, replay, or model artifacts.

## M2-E001 — Walking skeleton: the pipeline runs end to end; the wall-clock budget does not

**Date:** 2026-09-18
**Status:** Pipeline PASSES end to end. M2-P001's throughput assumption is
REFUTED, and a defect in `train.py` is part of why (`#30`). Board `#29`.
**Purpose:** Run the whole Milestone 2 pipeline at the small budget
pre-registered in `M2-P001` ("Skeleton first") — train, numbered checkpoint,
set A, selection, set B, report — and prove the per-episode MLflow view.
Commit `ef053ac`, no code changed on the device.

**Every number below is pipeline evidence, not model evidence.** One training
seed stopped at 25,184 decisions, one candidate checkpoint, 14 episodes an arm,
no scripted arm: none of it says anything about whether `stacked-dqn` learns,
and M2-P001's decision rule is not exercised by it.

**Stage 0 — MLflow (1 min).** `train.py` resolves
`sqlite:///~/.local/state/tower-rl/mlflow.db` from the default `--run-dir`
(`experiment/tracking.py:tracking_uri`). The UI answered on
`http://127.0.0.1:5000`. Pre-flight: zero qemu in `/proc/*/exe`, `adb devices`
empty, `bridge/current` resolving and hashing to `7a98f50b…f99a`.

**Deviation forced by the script, before any device work.** The pre-registered
skeleton's `--checkpoint-every-decisions 25000` is not a whole number of
2,000-decision blocks, so `train.py` refuses it by name. Stage 1 therefore ran
`--block-decisions 2500`. The budgeted run's period of 100,000 is unaffected.

**Stage 1 — training (65 min, 19:48-20:53).** `--actors 7 --budget-decisions
100000 --block-decisions 2500 --checkpoint-every-decisions 25000 --renderer
host`. Fleet 7/7 up in 4.5 min, cold under `-gpu host` as designed: `is at home
and offline` 7/7 and `deployed bridge confirmed
7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a` 7/7. Only
5556-5568; no reference to 5554 or the canonical AVD.

Collected **25,184 decisions in 247 episodes, 246 valid and 1 invalid** (the
invalid one is the episode the interrupt landed in), all seven actor indices
present, 5,193 optimisation steps, mean final wave 4.6, epsilon annealed to
0.05. The run was **stopped at the first numbered checkpoint** rather than run
out to 100,000, on the Lead's decision once the throughput finding below was
in hand; the budget is a script argument, so the stop was a `SIGINT`.
`checkpoint-0025184.pt` and its `.sha256` were written beside `latest.pt`.
Interrupting before the budget means no session report, no arm summary and no
post-budget evaluation: the run directory holds `checkpoints/` and
`manifest.json` only. **The MLflow run ended `FINISHED`** — `train.py`'s
`finally` finishes the tracked run through a `KeyboardInterrupt` — which is
worth knowing, because a run killed this way is indistinguishable in MLflow
from one that spent its budget.

MLflow evidence, read through the client rather than the UI: every `episode_*`
series — `episode_final_wave`, `episode_decisions`, `episode_game_ms`,
`episode_wait_fraction`, `episode_purchases`, `episode_valid`, `episode_actor`,
`episode_epsilon` — at **count 247, equal to the episodes collected**;
`learner_optimisation_steps` and `learner_importance_beta` at 247, and
`learner_gradient_norm`, `learner_weighted_loss`,
`learner_unweighted_mean_absolute_td_error`, `learner_value_fit_correlation` at
202 (they begin once warm-up passes); the 30 s decision-time breakdown in 17
series. The per-episode live view M2-P001 asked for exists and is correct.

**Finding 1: `train.py` never raises the guest frame rate (`#30`).**
`raise_frame_rate` has exactly two callers, `run_actors.py` and
`clone_session.py`; the training path has none. This fleet therefore began
collecting with SurfaceFlinger's stock per-uid game default override of 60 Hz
standing over a 120 Hz display — read off live instance 5556 mid-run as
`activeMode={… vsyncRate=120.00 Hz …}` against `GameFrameRateOverrides
(uid, gameModeOverride, gameDefaultOverride)={10218, 0 60}`. This is exactly
the silent failure `confirm_frame_rate`'s own docstring names. **The rate was
raised by hand during this run, at about 5,000 of the 25,184 decisions**, with
7/7 `confirmed at 120 Hz` on all three readings; the run's throughput series is
therefore two regimes, 60 Hz before that point and 120 Hz after, and only the
second is comparable to anything. The effect was about a factor of two: bridge
round trip 1,400 ms/decision to 730 ms, ~2,500 to ~5,000 decisions/hour per
actor. Every training run before this one collected at 60 Hz and nothing said
so.

**Finding 2: learned-policy throughput is ~4.7x below what M2-P001 budgeted.**
At a confirmed 120 Hz the fleet does **~28,000-30,000 decisions/hour
aggregate** (end-to-end over the whole stage, bring-up and the 60 Hz opening
included, ~24,900), against the 129,000-143,000 M2-P001 took from `M1B-E052`.
`M1B-E052` is a **scripted** measurement. Under the learned policy
`episode_wait_fraction` is ~0.80: most decisions are waits, each advancing the
world until a distant event, so a decision costs far more wall clock than it
does for a policy that buys constantly. Scripted throughput does not transfer
to a training arm. **Consequence: the 100,000-decision skeleton is ~3.3 h, not
~45 min, and the pre-registered 1,000,000-decision run is ~33 h, not ~7 h.**
M2-P001's wall estimates, and the developer's approval priced on them, must be
re-priced before the budgeted run. This is a statement about the fleet under
this policy at this epsilon, not a ceiling: `#30` recovers about half of the
gap on its own, and the wait fraction may fall as the policy sharpens.

**Stage 2 — set A and selection (7 min, 20:56-21:03).** `run_actors.py
--actors 7 --episodes 1 --policy checkpoint:…/checkpoint-0025184.pt --renderer
host --frame-rate-hz 120`. **7 episodes, 7 valid, 0 invalid**, with
`advances_cut_short` 0 and `episodes_not_started_fresh` 0 on every record,
speed-up ~2.0 and ~23 decisions/wave. 7/7 `confirmed at 120 Hz` on all three
readings — the contrast that isolates `#30` to the training path.
`select_checkpoint.py` over the one candidate, `--mlflow-run` given:

    1 checkpoints of stacked-dqn-20260918-195301-dde9c7
    final_wave:
      checkpoint-0025184.pt        IQM   6.00  [  6.00,   6.00]  n=7
    decisions:
      checkpoint-0025184.pt        IQM 133.80  [133.80, 133.80]  n=7
    selected …/checkpoints/checkpoint-0025184.pt

`selection.json` was written beside the run carrying `checkpoint_identity
22867a9ca26e` and `decisions 25184`. With one candidate the interval is
degenerate and the choice is not a choice; what this proves is the mechanism.

**Stage 3 — set B and the report (19 min, 21:14-21:41, plus the report).**
Scripted is skipped in the skeleton, as M2-P001 specifies, so the arms are the
selected checkpoint and random, `--episodes 2` each on its own fleet:
**14/14 valid in each arm, 0 invalid**, 7/7 frame-rate confirmations and 7/7
bridge digests in both. `report_arms.py random=… stacked-dqn=… --selection
…/selection.json --mlflow-run …` accepted set B against the selection — it did
not refuse the arm — and printed:

    final_wave:
      random                       IQM   5.88  [  4.75,   7.00]  n=14
      stacked-dqn                  IQM   6.38  [  5.12,   7.25]  n=14
    decisions:
      random                       IQM 129.88  [108.12, 151.00]  n=14
      stacked-dqn                  IQM 144.25  [119.00, 163.62]  n=14
    pairwise difference in mean final wave:
      random 5.71 vs stacked-dqn 6.21: difference -0.50 [-2.36, +1.36] d=-0.19
        n=14/14 — indistinguishable

with the per-wave comparison over `game_ms`, `decisions`, `health_fraction` and
`cash_log` across 8 wave indices (`decisions` separated at waves 1, 5, 7;
`cash_log` at 3, 5; `health_fraction` at 4), and `logged to tracked run`.
**No claim is drawn from any of it.** At 14 episodes an arm the per-episode
comparison could only detect a 2.8-wave difference, the model arm is a single
25,184-decision checkpoint, and M2-P001's decision rule requires ~60 valid per
arm and a scripted arm.

Post-hoc MLflow series confirmed through the client:
`greedy_final_wave_iqm`, `greedy_final_wave_ci_low` and `_ci_high` all 6.00 at
**step 25184**, the checkpoint's own decision count, and
`report_random_final_wave_iqm` 5.875 and `report_stacked-dqn_final_wave_iqm`
6.375 with their intervals at **step 0** — the greedy curve lands above the
exploring one, as `architecture.md` §7 describes.

**Teardown, after each of the four fleet stages.** Per-serial cleanup on every
live instance before the kill, 7/7 on every line each time: `libunity_sha256
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, `game_frame_rate_override: reset`; then zero qemu
in `/proc/*/exe` and `adb devices` empty. No taps, no screenshots, every
artifact outside the repository.

**Verdict.** The pipeline passes end to end: train → numbered checkpoint →
set A → selection → set B → report, with MLflow carrying the live per-episode
view and both post-hoc curves, and with the fleet coming up 7/7 and going down
clean four times in a row. The budgeted run is not blocked by the machinery. It
is blocked by its own arithmetic: `#30` first, then a re-priced wall estimate
from a measured learned-policy throughput rather than a scripted one.

## M2-P001 — Milestone 2 pre-registered protocol (written before any run)

**Date:** 2026-09-18
**Status:** Pre-registered; no run has started. This entry records the plan and
its decision rule before any data exists, not a result.

**Goal.** One model, `stacked-dqn`, trained on the 7-actor fleet at 120 Hz,
compared against random and scripted under one protocol. Claims are
distributional (IQM with stratified bootstrap intervals, strata = actors); no
claim rests on a single best run. One training seed this round; that is a
stated limitation, not a claim of generality.

**Recipe.** Fleet: N=7 clone instances (`tower_rl_instrumented_api36`,
`-read-only`, cold `-gpu host`, 120 Hz confirmed per instance, bridge digest
confirmed by name, offline by interface). Training: `scripts/train.py --actors
7 --budget-decisions 1000000 --checkpoint-every-decisions 100000` (block size
as default), epsilon/beta schedules as the script's defaults, replay ratio
0.25 gradient-steps/decision (~126:1). Expected wall ≈ 7 h at 129–143k
decisions/hour (`M1B-E052`). Candidates: the 10 numbered checkpoints
`checkpoint-<decisions, 7 digits>.pt`. Set A (selection): each candidate
evaluated greedily on the fleet, `run_actors.py --actors 7 --episodes 2
--policy checkpoint:<path>` (14 episodes per candidate, 140 total, ≈1 h).
Selection: `select_checkpoint.py` — highest IQM of final wave; ties broken by
lower decisions (earlier checkpoint); writes `<run>/selection.json`. Set B
(report): the selected checkpoint, scripted, and random, each `--episodes 9`
on 7 actors (≥60 valid per arm, ≈1.3 h), fresh episodes, into separate
directories; `report_arms.py` with `--mlflow-run` so results land in the
training run, and `--selection <run>/selection.json` so the model arm is
verified to be the selected checkpoint.

**Skeleton first.** Before the budgeted run, the same pipeline at
`--budget-decisions 100000 --checkpoint-every-decisions 25000`, set A
`--episodes 1`, set B `--episodes 2` (≈1.2 h total). Its purpose is to prove
the pipeline end to end and the per-episode MLflow view; its numbers are not
evidence about the model.

**Pre-registered decision rule.** Report IQM and 95% stratified-bootstrap
intervals of final wave per arm on set B, and pairwise bootstrap differences.
The headline claim "`stacked-dqn` beats scripted" is made only if the pairwise
interval of (model − scripted) final-wave IQM excludes zero on set B. "Beats
random" likewise. Per-wave statistics (`wave_statistics.analyse_reports`) are
reported as secondary evidence, not as a decision. Selection on set A never
appears in a claim; only set B does. Any failure of the fleet (an arm below 60
valid) means that arm is re-run whole, not padded.

**What the sample cannot detect.** At ~60 valid per arm, final-wave
differences below roughly 0.5 waves (`M1B-E021`/`M1B-E053` sd ≈ 1.3) are
undetectable; nothing about seed-to-seed variance (one seed); nothing about
robustness to a different image state or frame rate; a selection-set optimism
bias remains in set A numbers, which is why they are not reported.

**Live view.** MLflow run logs per episode (`episode_final_wave`,
`episode_decisions`, `episode_game_ms`, `episode_wait_fraction`,
`episode_purchases`, `episode_valid`, `episode_actor`, epsilon), learner
scalars (`learner_*`), the 30 s decision-time breakdown, numbered checkpoints
as artifacts; post-hoc `greedy_final_wave_iqm` per checkpoint at its decisions
and set-B arm results at step 0. UI: `uv run --extra tracking mlflow ui
--backend-store-uri <tracking uri printed at run start>`.

**Approvals.** Developer approved the ~7 h training budget on 2026-09-18
contingent on the skeleton passing; the ~2.3 h evaluation phase is stated here
so the total (~9.5 h) is on record before the run.

## M1B-E056 — The simulation module drives the device exactly as the scripts did

**Date:** 2026-09-18
**Status:** Passed, all three steps; no behaviour change observed on device

**Purpose:** `#16` moved emulator lifecycle out of `clone_session.py`,
`run_actors.py` and `train.py` into `tower_rl.simulation`, and collapsed the two
`deploy_bridge` wrappers and the two fleet loops to one each. The move was
argued to be behaviour-preserving. That is a claim about a device, and the unit
suite cannot settle it: every emulator, every adb call and every bridge in it is
a double. This is the one run that puts the moved code against the real clone.

Solo run on the free host, branch `worktree-agent-a34563d23c6c7196e` at
`574c6ea`, one instance, `-gpu host`, cold, `-read-only`, 4 cores, 120 Hz.

**1. Cold bring-up through the moved code.** `scripts/clone_session.py up
--read-only --cold --renderer host --cores 4` reached home and exited 0. The
cold path was chosen by name (`renderer 'host' cannot snapshot a Vulkan app`),
which is the pre-existing rule, not a new one. Three read-backs, all as the
scripts produced them before the move:

- Offline by interface: `ip -o -4 addr show` returned `lo` alone. This remains
  the only offline oracle in the tree.
- Guest rate: `confirmed at 120 Hz: display vsync mode 120.00, uid 10218 game
  mode override 120, uid 10218 applied frame rate 120.00` — both levers and the
  applied surface rate agreeing, which is what `confirm_frame_rate` exists to
  refuse an instance for.
- Bridge identity: the deployed artifact read back as
  `7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a`, matching
  `libtower_bridge.so` in the private build directory. The plain
  `adb shell sha256sum` form answered; the `su -c` form was not needed on this
  image.

Every deploy line arrived tagged with its serial (`emulator-5556 deploy: ...`),
which is the surviving `deploy_bridge` — the fleet's capturing variant, now used
by the single-instance CLI too. `adb push` writes its progress to stderr, so
those lines relay under the `error:` marker; that is the wrapper reporting
faithfully, not a failure.

**2. One scripted episode through the normal fleet path.**
`scripts/run_actors.py --actors 1 --episodes 1 --renderer host --cores 4
--frame-rate-hz 120`, exercising `stagger_bring_up` → `collect_episodes` →
`run_episodes.py`, with the sequencer now taking the per-actor step as an
argument rather than closing over the runner's `Namespace`.

1 valid episode, 0 invalid, final wave 8, 17.4 valid episodes/hour at N=1 — a
one-episode rate that carries a whole cold bring-up, so it is not comparable to
the steady-state figures in `M1B-E028` and is recorded only as evidence the path
completed. All four fidelity counters zero: `bridge_event_divergence`,
`stale_or_duplicate`, `advances_cut_short`, `episodes_not_started_fresh`.

The arm key survived the move in both places it is written: the fleet report
carries `frame_rates_hz: [120]` and the actor entry `frame_rate_hz: 120`, and
the actor's own durable record (`emulator-5556.json`) carries
`frame_rate_hz: 120`. That key is what a two-rate comparison groups by, so a
record that lost it could only be attributed by the directory it sat in.

**3. Teardown through `tear_down_instance`.** Run twice — once to put step 1's
instance down, once by the fleet at the end of step 2 — and identical both
times. The per-serial cleanup report verified the instance it names:

    game_frame_rate_override: reset
    libunity_sha256: ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040
    versionCode=1199 minSdk=27 targetSdk=36
    versionName=29.0.3
    installerPackageName=com.android.vending
    libunity_mounts: 0
    bridge_artifacts: removed

The `libunity.so` hash matches the untouched original, so the overlay was
removed rather than left mounted; the installer and version confirm the Play
build was not replaced. After the run: no qemu process under `/proc/*/exe`, and
`adb devices` empty.

**Device safety.** Only `emulator-5556` and only `tower_rl_instrumented_api36`
appear anywhere in either log. `emulator-5554` and the canonical evaluation AVD
were never addressed. Neither log contains `screencap`, `screenshot`,
`uiautomator` or `input tap`; nothing in this path reads a pixel or touches the
screen, which the static guard added with `#16` now holds from source rather
than from a fixture.

**What this does not establish.** One actor, one episode, one image state. It
says the moved code reaches a real device and comes back with the same readings
the scripts produced; it is not a throughput measurement, not a multi-actor
result, and not a substitute for the equivalence gate in `#22`.

## M1B-E055 — Guest resolution does not move host CPU per frame: the render-cost hypothesis is falsified at 120 Hz

**Date:** 2026-09-18
**Status:** NEGATIVE on the stated hypothesis. Prediction falsified: a 3.24x
pixel reduction left emulator-process CPU% within 2.4% of baseline
**Purpose:** Phase 1 of the CPU-per-frame question — test whether host CPU per
guest frame is dominated by rendering cost, which would scale with guest
resolution.

Hypothesis: host CPU per guest frame is dominated by rendering. Prediction if
true: cutting the guest resolution (quarter the pixels) cuts emulator-process
CPU% at a fixed 120 Hz by at least 30%. Falsified if CPU% stays within about
10% of baseline.

One instance (`emulator-5556`, clone AVD `tower_rl_instrumented_api36`,
`-read-only`, cold `-gpu host`, `--cores 4`, `--frame-game-ms 100`, scripted, 3
episodes per configuration), 120 Hz confirmed on three SurfaceFlinger readings
before each arm, bridge deployed, offline verified by interface.

### The measurements

| arm | guest size (px) | pixels vs C0 | emulator CPU% (`/proc` utime+stime over 67 s) | `top -b` cross-check | measured ms/frame | round/budgeted | fidelity counters | valid | dec/wave | mean wave |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 (run a) | 360×640 | 1.00 | **160.8** | 140–200 | 8.177 | 1.00987 | all 0 | 3/3 | 21.95 | 6.67 |
| C0 (run b) | 360×640 | 1.00 | **157.0** | 140–150 | 8.186 | 1.01109 | all 0 | 3/3 | 22.06 | 5.67 |
| C1 | 200×356 | 0.31 | **160.8** | 150–190 | 8.184 | 1.01165 | all 0 | 3/3 | 21.26 | 7.67 |

Fidelity counters are `bridge_event_divergence`, `stale_or_duplicate`,
`advances_cut_short`, `episodes_not_started_fresh` and `invalid_episodes`: zero
in every arm. C0 was run twice because the per-thread sampling below was added
after the first baseline; both baselines are reported rather than one being
discarded, and their 2.4% spread is the honest noise floor for a single
67-second sample.

**Verdict: the prediction is falsified.** 3.24x fewer pixels changed CPU% by
0.0% against the first baseline and +2.4% against the second — inside the
falsification band and inside the baseline's own run-to-run spread, nowhere
near the predicted 30% fall. **C2 (a further halving) was therefore not run**,
per the recipe's own rule that C2 follows only an effect at C1.

Measured ms/frame — advance wall time divided by frames actually stepped —
is 8.177 / 8.186 / 8.184 across the three arms, i.e. identical to three decimal
places and equal to the 120 Hz vsync period (8.33 ms nominal). The guest is
paced by vsync, not by how long a frame takes to render, which is the same
conclusion the CPU figure reaches from the other side.

### Where the CPU actually goes

Per-thread, sampled over the same steady-state window (`top -H -b -n 2 -d 60`;
the second iteration is the 60 s delta and is what is quoted).

Host, emulator process:

| arm | 4 vCPU threads | `RenderThread` (2) | remaining qemu threads | render share of process |
| --- | --- | --- | --- | --- |
| C0 (b) | 37.0 + 34.4 + 32.2 + 28.4 = 132.0 | 9.5 + 2.3 = 11.8 | ~8 | **~7.5%** |
| C1 | 40.1 + 33.8 + 31.1 + 29.7 = 134.7 | 9.5 + 2.2 = 11.7 | ~10 | **~7.3%** |

No thread matching `gfx`, `llvmpipe`, `gl`, `Vk` or `SwiftShader` used
measurable CPU in either arm; under `-gpu host` the GPU work leaves the CPU.
The host-side render share is ~7% of the emulator process and **does not move
with resolution at all**, while ~85% of the process sits in the four vCPU
threads — guest execution, not rasterisation.

Guest, game process (per-thread `/proc/<tid>/stat` deltas over the same window;
guest CPU% is of one guest core):

| thread | C1 (200×356) | C1 first attempt (200×320, no layout fix) |
| --- | --- | --- |
| `UnityMain` | 47.3 | 34.2 |
| `UnityGfxDeviceW` | 20.9 | 18.4 |
| `Job.Worker 0` | 10.2 | 7.1 |
| `UnityChoreograp` | 1.6 | 1.4 |
| other >0.5% | 2.9 | 1.9 |

The C0 guest sample was taken with `top -H` only and its 60 s iteration did not
report `UnityMain` in its ranked rows, so it is not quoted as a comparison
figure; the `/proc`-delta sampler that produced the C1 column was added after
that arm and no extra boot was spent to redo it. What the C1 column does show
is that the game's own render thread (`UnityGfxDeviceW`) is about 30% of the
game's CPU even at 71,200 pixels, so guest-side render cost is not
pixel-bound either at this size.

Instance facts, read from the run rather than assumed: the emulator's own
command line carries `-gpu host`, and the installed `libunity.so` (SHA-256
`ffc1f3ef…dd0040`, read back after cleanup with the overlay gone) reports Unity
`6000.3.15f1`.

### How the resolution was changed, and what the device did with it

No AVD configuration file was edited. The least invasive runtime path was used:
`adb shell wm size WxH` and `wm density D` applied **after** the bridge
deployment and **before** the bring-up's own `launch_game_at_home`, so the game
was force-stopped and launched again by the normal path and laid out at the new
size. No taps, no screenshots. Both were reset (`wm size reset`, `wm density
reset`, read back as 360×640 / 140) before teardown, and the instance is
`-read-only` in any case, so the clone AVD is unmodified.

Two device constraints showed up in doing it, and both are worth recording:

1. **WindowManagerService clamps a forced display size to 200 px per
   dimension.** `wm size 180x320` — the intended exact halving — reported back
   as `Override size: 200x320`, silently changing the aspect ratio from 0.5625
   to 0.625. The floor is why C1 is 200×356 rather than 180×320, and it also
   bounds any future arm: 200 px is the narrowest this lever reaches.
2. **`wm density 70` was refused** (the reading stayed at 140), so the first C1
   attempt ran at 200×320 with the stock density, i.e. with the game laid out
   at 228×366 dp instead of 411×731 dp. **That attempt failed**: the episode
   driver raised `the game did not honour speed_down: lifecycle_timeout` at the
   very first episode boundary, before any episode ran, so its 130% CPU reading
   is of a game idling at home and is not a measurement of anything. The
   corrected arm keeps the logical layout identical — 200×356 at density 78 is
   410×730 dp, against the stock 411×731 dp — and ran 3/3 valid with every
   counter at zero. **A display change that alters the dp layout can break the
   game's own speed control**; one that preserves it did not.

### What this closes and what it does not

It closes the phase-1 question: **resolution is not the lever on host CPU per
guest frame at 120 Hz**, and the render-elimination idea that phase 2 would
have tested is not supported by where the CPU is — ~85% of the emulator process
is vCPU execution and ~7% is host-side rendering, at either resolution.

It is consistent with `M1B-E030`, which reduced pixels 9x (1080×1920 → 360×640)
for only a ~16% fall in per-qemu CPU and attributed its real payoff to
throughput rather than to VRAM. This entry says the remaining pixel reduction
available below 360×640 buys nothing at all.

Single-instance, single 67 s window, 3 episodes per arm: the wave and
decision-density figures are indicative only and no fidelity claim is made from
them beyond the counters being zero. The CPU conclusion does not need more
data — the effect predicted was 30% and the measured difference is inside the
baseline's own repeat spread.

Teardown after each configuration: `instrumented_bridge.sh cleanup
emulator-5556` — override reset, libunity
`ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
versionCode 1199, versionName 29.0.3, installer `com.android.vending`,
`libunity_mounts: 0`, `bridge_artifacts: removed` — then kill, `adb devices`
empty and zero qemu in `/proc/*/exe`. Only `emulator-5556` was addressed; the
canonical AVD and `emulator-5554` were never referenced.

Source: session scratchpad `E055/` (`driver.py`, `C0-360x640.json`,
`C0b-360x640.json`, `C1-180x320.json` (the clamped, failed attempt),
`C1b-200x356.json`, per-arm `*-host-threads.txt`, `*-guest-threads.txt`,
`*-episodes.json`).

## M1B-E054 — Replication of the equivalence fleet in a fresh session: the rule fires nowhere, so `M1B-E053`'s REJECT is not replicated and does not stand

**Date:** 2026-09-18
**Status:** No difference detectable at this n. `M1B-E053`'s provisional REJECT
is **not replicated**; under the rule's own replication clause it does not stand
(board #22 stays open for the Lead's reading of the pair)
**Purpose:** Execute `M1B-E053`'s replication clause — "any REJECT is replicated
once, from this written recipe, in a fresh session, before it is acted on" — by
running the same recipe again, in a session that had not seen `M1B-E053` or its
raw analysis, and applying the pre-registered rule mechanically before reading
either.

**Recipe and parameters.** Session scratchpad `EQUIVALENCE-RECIPE.md`, the same
document `M1B-E053` ran from; no parameter changed. Code at `1cb3e0a`.

    uv run python scripts/run_actors.py --actors 7 --episodes 9 --policy scripted \
        --renderer host --cold --cores 4 --frame-game-ms 100 \
        --frame-rate-hz 60,60,60,60,120,120,120 \
        --output-directory ~/.local/state/tower-rl/equivalence-2026-09-18-replication

N=7, `-read-only`, cold `-gpu host`, offline by interface per instance, scripted,
9 episodes per actor, one deployable bridge. Fleet wall **592.6 s** against
`M1B-E053`'s 605.9 s, inside the 30-minute timebox this run was given.

**Per-instance fate**, each confirmed on all three SurfaceFlinger readings
(display vsync mode, per-uid game mode override, per-uid applied frame rate) at
the rate its index was assigned:

| index | serial | rate | confirmed | valid/attempted | fate |
| --- | --- | --- | --- | --- | --- |
| 0 | emulator-5556 | 60 | yes, 60.00/60/60.00 | 9/9 | clean |
| 1 | emulator-5558 | 60 | yes, 60.00/60/60.00 | 9/9 | clean |
| 2 | emulator-5560 | 60 | yes, 60.00/60/60.00 | 9/9 | clean |
| 3 | emulator-5562 | 60 | yes, 60.00/60/60.00 | 9/9 | clean |
| 4 | emulator-5564 | 120 | yes, 120.00/120/120.00 | 9/9 | clean |
| 5 | emulator-5566 | 120 | yes, 120.00/120/120.00 | 9/9 | clean |
| 6 | emulator-5568 | 120 | yes, 120.00/120/120.00 | 9/9 | clean |

**7 of 7 reporting, no instance lost, no episode invalid.** `M1B-E053`'s
`RunPortError: the instance did not reach an active run` on index 0 **did not
recur**, and neither did its one `stale_or_duplicate`. The instance-loss shape of
the two runs therefore differs: E053 lost its first-index 60 Hz actor whole
(0/9) and one episode at index 6; this run lost nothing. That is the fleet
behaviour `M1B-E052` reports for this image state, not a new fix.

**Arms: 36 valid at 60 Hz, 27 valid at 120 Hz**, both above the pre-registered
floor of 25, with no hand exclusion — every record carried its own
`confirmed at <rate> Hz` line at its assigned rate.

**Fidelity, per arm.** Zero in both arms: `GAME_TIME_INFLATED`,
`GAME_TIME_DEFLATED`, `ADVANCE_TRUNCATED_BY_WALL`, `advances_cut_short`,
`bridge_event_divergence`, `episodes_not_started_fresh`, `stale_or_duplicate`;
`invalid_rate` 0.0 in both. The shared host did not reach the game clock in
either arm, so this is read as a behavioural result rather than a contended one,
on the same grounds as `M1B-E053`.

**Throughput is not a result here either** — the arms shared a host, so no
ms/frame or episodes/hour figure from this run is comparable with `M1B-E045`,
with `M1B-E053`, or between the arms.

**Bridge digest, now confirmed per instance.** All seven instances read back the
deployed
`7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a` at
`/data/user/0/<package>/files/libtower_bridge.so` while up, matching the host
artifact verified before the fleet. `M1B-E053` and `M1B-E046` recorded this
**unconfirmed, not passed** because `adb shell su 0 sha256sum` returned nothing;
the form that works on this image is the repository's own `su_device`,
`adb shell "su -c 'sha256sum <path>'"`. That is a read-back defect, not a bridge
difference: both runs deployed the same host artifact.

**Analysis**, exactly as pre-registered and applied before `M1B-E053` was read —
records grouped by the `frame_rate_hz` each actor's own record carries, then
`wave_statistics.analyse_reports('60hz.json', '120hz.json')`. 36/27 valid
episodes, 95% bootstrap intervals, 80% power; **nine wave indices** reached by ≥2
completed episodes in both arms (wave 10 named as underpowered, 1/0). A single
wave index at this n could detect **d ≥ 0.718**.

| statistic | wave indices compared | separated | pooled d |
| --- | --- | --- | --- |
| `game_ms` | 9 | **wave 9**, 120 Hz higher | +0.010 |
| `decisions` | 9 | **wave 8**, 60 Hz higher | −0.073 |
| `health_fraction` | 9 | none | +0.085 |
| `cash_log` | 9 | **wave 8**, 60 Hz higher | +0.008 |

`game_ms` separated only at wave 9 (−107.00 ms [−178.33, −35.67], d=−1.73,
n=3/3, detectable ≥ 141.3); waves 1–8 ran −22.4 to +20.6 ms against detectable
differences of 47.0 to 118.3 ms, every interval covering zero. `cash_log`
separated only at wave 8 (+1.68 [+0.89, +2.38], d=+2.41, n=8/4), where
`decisions` also separated (+2.75 [+1.88, +3.75], n=8/4) in the same direction
while `game_ms` did not (+13.38 [−66.88, +93.62]) — the four 60 Hz episodes that
reached wave 8 played it longer than the four 120 Hz ones, which is cadence and
sampling at n=8/4, not the game's own clock. Per episode, `final_wave` 6.42 vs
6.52 (detectable ≥ 1.663) and `decisions` 137.8 vs 139.5 (detectable ≥ 33.0),
both indistinguishable. Raw output: session scratchpad
`EQUIVALENCE-ANALYSIS-REPLICATION.txt`.

**VERDICT, applied mechanically under the rule quoted verbatim in `M1B-E053`:
no difference detectable at this n.**

- **Trigger (a), timing, did not fire.** `game_ms` separated at **one** wave
  index (9), and the rule requires more than 100 ms in the same direction at
  **≥2** separated wave indices. The one separation does exceed 100 ms (107.00),
  which E053's `game_ms` never approached, and it sits at the thinnest compared
  depth (n=3/3, detectable ≥ 141.3 ms, i.e. the index could not have resolved a
  one-frame effect at all).
- **Trigger (b), state, did not fire.** `health_fraction` separated at **0 of 9**
  indices (0%); `cash_log` at **1 of 9** (11.1%), at or below the 20% criterion.

**Side by side with `M1B-E053`, per statistic and wave index.**

| statistic | E053 (27/26, 7 indices, d ≥ 0.785) | E054 (36/27, 9 indices, d ≥ 0.718) | same indices? | same direction? |
| --- | --- | --- | --- | --- |
| `game_ms` | separated nowhere; pooled −0.027 | wave 9 only, −107.0 ms (120 Hz higher); pooled +0.010 | no — wave 9 was not compared in E053 (0/1) | n/a |
| `decisions` | nowhere; +0.104 | wave 8 only, +2.75 (60 Hz higher); −0.073 | no — wave 8 was not compared in E053 (4/1) | n/a |
| `health_fraction` | nowhere; −0.173 | nowhere; +0.085 | yes (both nowhere) | n/a |
| `cash_log` | **waves 5 and 6**, both 60 Hz higher; +0.206 | **wave 8 only**, 60 Hz higher; +0.008 | **no** | yes, 60 Hz higher in both |
| rule | **(b) fires, 2/7 = 28.6%** | (b) does not fire, 1/9 = 11.1% | — | — |

At the two indices E053 rejected on, this run found nothing: wave 5 +0.06
[−0.27, +0.37] (E053: +0.34 [+0.02, +0.64]) and wave 6 +0.39 [−0.07, +0.85]
(E053: +0.46 [+0.08, +0.86]). Both point the same way as E053 (60 Hz higher) and
wave 6 is close in magnitude, but neither interval excludes zero, and the pooled
`cash_log` d fell from +0.206 to +0.008. **Stated honestly and without changing
the verdict:** this run's per-index detectable differences at waves 5 and 6
(≥ 0.485 and ≥ 0.649) are *larger* than the effects E053 reported there
(+0.34, +0.46), so its silence at those two indices is a failure to reproduce,
not a demonstration that the effect is absent.

**The combined reading the rule permits.** The rule says a REJECT is acted on
only once replicated from the written recipe in a fresh session. This
replication did not reproduce it: `cash_log` did not separate at E053's indices,
did not clear the 20% criterion anywhere, and no other trigger fired.
**The REJECT is therefore not replicated and does not stand.** What the pair
supports is the published result of the rule's "otherwise" branch — *no
difference detectable at this n* — across 63 and 53 valid episodes, with the two
runs' only separated state indices disjoint, non-monotone, and at 4–8 episodes
per arm. It does not support "no difference": `M1B-E053`'s own caveat about the
20% criterion having no resolution at 7–9 comparable wave indices applies to
this run unchanged, and both runs' separations sit at the shallowest-n depths
where a stray index is exactly what a null looks like.

**What this sample could not detect.** Per-wave effects below d ≈ 0.718 (and
below each printed per-index figure); anything at wave 10 and above (1/0 episodes
reached it); anything about the wave an episode died in, excluded as a fragment;
anything visible only in episode length (`final_wave` could detect ~1.66 waves);
anything about throughput, since the arms shared a host; anything about a rate
other than 60 and 120, a learner attached, an N other than 7, a 4/3 split other
than this one, or a renderer other than `-gpu host`; and anything about
stability, which here cost nothing.

**Teardown:** complete on all seven serials — original `libunity.so` SHA-256
`ffc1f3ef…dd0040` re-verified ×7, `versionCode 1199` / `29.0.3` / installer
`com.android.vending` ×7, `libunity_mounts: 0` ×7, `bridge_artifacts: removed`
×7, `game_frame_rate_override: reset` ×7, `teardown_failure` null ×7, no qemu
process left (checked through `/proc/*/exe`), `adb devices` empty. Per-instance
logcat captured for all seven. 5554 and the canonical AVD were never addressed.

## M1B-E053 — Behavioural equivalence, 60 Hz against 120 Hz, one interleaved fleet: the pre-registered rule says REJECT on cash at two of seven wave indices

**Date:** 2026-09-18
**Status:** REJECT, provisional — the rule's own replication clause is unmet, so
120 Hz is NOT cleared for training (board #22 stays open)
**Purpose:** Decide whether the fleet operating rate (120 Hz, `M1B-E045`) changes
how the *game* behaves relative to the stock 60 Hz. The game is not
deterministic, so equivalence is distributional on the low-variance per-wave
statistics (`experiment/wave_statistics.py`), never on mean final wave (pooled sd
2.27 waves needs ~324 episodes/arm for half a wave).

**Design.** One fleet, two rates, interleaved by instance — not two fleets one
after the other. Both levers that set the rate are per emulator (`-vsync-rate` at
launch, the per-uid `cmd game set --fps` override), so `--frame-rate-hz` now
takes one rate per instance index and the arms share a window, a host, a bridge
and an account state. That removes the order confound a sequential pair could not
have excluded, at the price of a contention limitation stated below.

    uv run python scripts/run_actors.py --actors 7 --episodes 9 --policy scripted \
        --renderer host --cold --cores 4 --frame-game-ms 100 \
        --frame-rate-hz 60,60,60,60,120,120,120 \
        --output-directory ~/.local/state/tower-rl/equivalence-2026-09-18

N=7, `-read-only`, cold `-gpu host`, offline by interface per instance, scripted,
9 episodes per actor, one deployable bridge (host artifact SHA-256
`7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a`, verified on
the host before the fleet). Fleet wall 605.9 s, inside the 45-minute timebox with
34 minutes to spare.

**Per-instance fate**, each confirmed on all three SurfaceFlinger readings
(display vsync mode, per-uid game mode override, per-uid applied frame rate) at
the rate its index was assigned:

| index | serial | rate | confirmed | valid/attempted | fate |
| --- | --- | --- | --- | --- | --- |
| 0 | emulator-5556 | 60 | yes, 60.00/60/60.00 | 0/9 | **lost** at its first `reset`: `RunPortError: the instance did not reach an active run`, after reaching home, offline and confirmed |
| 1 | emulator-5558 | 60 | yes | 9/9 | clean |
| 2 | emulator-5560 | 60 | yes | 9/9 | clean |
| 3 | emulator-5562 | 60 | yes | 9/9 | clean |
| 4 | emulator-5564 | 120 | yes, 120.00/120/120.00 | 9/9 | clean |
| 5 | emulator-5566 | 120 | yes | 9/9 | clean |
| 6 | emulator-5568 | 120 | yes | 8/9 | one invalid: `advance was not confirmed: stale_or_duplicate` |

**Arms: 27 valid at 60 Hz, 26 valid at 120 Hz**, both above the pre-registered
floor of 25 despite the lost instance, which cost the 60 Hz arm 9 episodes and no
partial data — it died before its first episode. One failure, not three
consecutive, so the abort rule was not reached. The cause is **unexplained**: its
logcat shows no kill of the game before teardown's own force-stop, and the
`applied frame rate absent` signature of `M1B-E046`/`M1B-E049` is absent — this
instance was confirmed at its rate and then failed to start a run.

**Fidelity, per arm, and the contention question.** Zero in both arms:
`GAME_TIME_INFLATED`, `GAME_TIME_DEFLATED`, `ADVANCE_TRUNCATED_BY_WALL`,
`advances_cut_short`, `bridge_event_divergence`, `episodes_not_started_fresh`.
Round/budgeted game-time ratio 0.9962–1.0133 (60 Hz) and 0.9892–1.0125 (120 Hz),
both inside the 1.25 guard. One `stale_or_duplicate` in the whole fleet. So the
shared host did not reach the game clock in either arm — which is the only way
contention could have reached these statistics.

**Throughput is not a result here and is not reported as one:** the arms shared a
host, so no ms/frame or episodes/hour figure from this run is comparable with
`M1B-E045` or between the arms. For the record only, median episode wall clock
was 41.6 s at 60 Hz against 26.2 s at 120 Hz.

**Analysis**, exactly as pre-registered — records grouped by the `frame_rate_hz`
each actor's own record carries, then
`wave_statistics.analyse_reports('60hz.json', '120hz.json')`. 27/26 valid
episodes, 95% bootstrap intervals, 80% power; **seven wave indices** were reached
by ≥2 completed episodes in both arms (waves 8 and 9 named as underpowered, 4/1
and 0/1). A single wave index at this n could detect **d ≥ 0.785**.

| statistic | wave indices compared | separated | pooled d |
| --- | --- | --- | --- |
| `game_ms` | 7 | none | −0.027 |
| `decisions` | 7 | none | +0.104 |
| `health_fraction` | 7 | none | −0.173 |
| `cash_log` | 7 | **waves 5 and 6**, both 60 Hz higher | +0.206 |

Per-wave `game_ms` differences ran −28.6 to +128.1 ms against per-wave detectable
differences of 62.8 to 283.5 ms; every interval covered zero. `cash_log`
separated at wave 5 (+0.34 [+0.02, +0.64], d=+0.73, detectable ≥ 0.447) and wave
6 (+0.46 [+0.08, +0.86], d=+0.84, detectable ≥ 0.623), and reversed sign at wave
7 (−0.26, interval covering zero). Per episode, `final_wave` 5.85 vs 5.96
(detectable ≥ 1.664) and `decisions` 127.4 vs 128.4 (detectable ≥ 33.7), both
indistinguishable — the blunt instrument saw nothing, as designed. Raw output:
session scratchpad `EQUIVALENCE-ANALYSIS.txt`.

**The decision rule, pre-registered in writing before the fleet ran, quoted
verbatim:**

> REJECT if **either** trigger fires:
>
> - **(a) Timing.** `game_ms` differs by more than **100 ms** — one decision
>   frame at `--frame-game-ms 100` — in the **same direction** at **≥2 wave
>   indices whose 95% intervals exclude zero**. Game time per rendered frame is
>   fixed by `Time.captureDeltaTime`, so by construction the guest rate must not
>   move the game's own clock *at all*; one frame's worth is a tolerance on the
>   instrument, not a difference we are willing to accept.
> - **(b) State.** `health_fraction` or `cash_log` separates (interval excludes
>   zero) in the **same direction** at **more than 20% of the wave indices
>   compared for that statistic**. 20%, not "any wave": at 95% intervals ~5% of
>   indices separate by chance, the indices are not independent (the same
>   episodes are seen again at each depth), and the fake-port null separated at
>   1 of 62 on `health_fraction` and 7 of 62 on `cash_log`. A handful of
>   separated wave indices is what a null looks like here.
>
> Otherwise the result is **"no difference detectable at this n"** — never "no
> difference" — and is published with the detectable-difference figures beside
> it. `decisions` per wave is reported and interpreted but triggers nothing on
> its own: it is a function of `game_ms` and the cadence, so a decisions shift
> without a timing shift is evidence about the cadence, not about the game.
>
> **Any REJECT is replicated once**, from this written recipe, in a fresh
> session, before it is acted on.

**VERDICT, applied mechanically: REJECT.** Trigger (a) did not fire — `game_ms`
separated at no wave index at all. Trigger (b) did: `cash_log` separated in the
same direction (60 Hz higher) at 2 of 7 compared wave indices, 28.6% against the
20% criterion. The rule does not permit reading that as a null after the fact,
and it is recorded as the rule's answer.

**What is honestly uncertain about that verdict, without changing it.** The 20%
criterion was calibrated on a fake-port null with 62 comparable wave indices,
where one separation is 1.6%. This run reached only 7, where a *single*
separation is already 14.3% and two are 28.6% — the criterion has no resolution
between "a null's usual stray index" and a real effect at this depth. The
separation is also not monotone (wave 7 reverses sign), and `health_fraction`,
the other state statistic and the one a real physics change would move alongside
cash, separated nowhere. This is exactly why the rule carries a replication
clause, and the clause is unmet: **the REJECT is provisional and must not be
acted on until one replication from the written recipe in a fresh session
reproduces it on the same statistic in the same direction.**

**What this sample could not detect.** Per-wave effects below d ≈ 0.785 (and
below the per-wave figures printed beside each line); anything at wave indices 8
and above, which too few episodes reached; anything about the wave an episode
died in, excluded as a fragment; anything visible only in episode length
(`final_wave` could detect ~1.66 waves here); anything about throughput, since
the arms shared a host; anything about a rate other than 60 and 120, a learner
attached, an N other than 7, a 4/3 split other than this one, or a renderer other
than `-gpu host`; and anything about stability, which is `M1B-E046`–`M1B-E052`'s
subject and here only cost the 60 Hz arm one instance.

**Unconfirmed, not passed:** the per-instance read-back of the *deployed* bridge
digest. `adb shell su 0 sha256sum` returned no digest on any of the seven
instances within the run, the same failure `M1B-E046` recorded; the host artifact
was verified before the fleet and is the only digest evidence this run holds.

**Teardown:** complete on all seven serials — original `libunity.so` SHA-256
`ffc1f3ef…dd0040` re-verified ×7, `versionCode 1199` / `29.0.3` / installer
`com.android.vending` ×7, `libunity_mounts: 0` ×7, `bridge_artifacts: removed`
×7, `game_frame_rate_override: reset` ×7, no qemu process left (checked through
`/proc/*/exe`), `adb devices` empty, device offline. 5554 and the canonical AVD
were never addressed.

**Correction to the planning figures.** The recipe sized this run at ~45 minutes
from `M1B-E046`'s per-actor rate; it took 10. Cold `-gpu host` bring-up was
~50–90 s per instance rather than ~165 s (the `M1B-E046` figure predates the
rendezvous removal), and an episode cost 26–42 s rather than 90–165 s. Future
sizing should use these.

## M1B-E052 — 7/7 at 120 Hz with no install round at all: the kill is absent for this image state

**Date:** 2026-09-18
**Status:** 7/7. The equivalence gate `#22` is now runnable; board `#21` stays
open pending it
**Purpose:** Let the fleet arbitrate whether the Play-update kill is still live,
after the staged WebView session was consumed and not re-staged (`M1B-E051`).

Exactly the `M1B-E048` recipe: N=7, `-read-only`, cold `-gpu host`, `--cores 4
--frame-game-ms 100`, 120 Hz, scripted, 2 episodes per instance, boot stagger
with its per-instance backstop, offline verified by interface per instance,
per-instance `adb logcat` captured from boot.

**7/7 reporting, 14 valid episodes, 0 invalid.** Every instance: 2 valid
episodes, `is at home and offline`, `confirmed at 120 Hz` on all three
SurfaceFlinger readings (7 of 7 confirmations in the log), and the deployed
bridge SHA-256 read back from the device as
7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a (7 of 7).
Per instance (index / serial / wall s / valid per hour): 0 / 5556 / 139.0 /
51.8; 1 / 5558 / 159.8 / 45.1; 2 / 5560 / 191.5 / 37.6; 3 / 5562 / 213.6 / 33.7;
4 / 5564 / 288.7 / 24.9; 5 / 5566 / 297.6 / 24.2; 6 / 5568 / 367.0 / 19.6.
Aggregate 137.3 valid episodes/hour; 18,504-20,476 decisions/hour per actor.
Fidelity: `bridge_event_divergence` 0, `stale_or_duplicate` 0,
`advances_cut_short` 0, `episodes_not_started_fresh` 0, `invalid_episodes` 0.

**Install-line count: `installPackageLI` appears ZERO times in all seven
logcats** — 0, 0, 0, 0, 0, 0, 0 — and `Update system package
com.google.android.webview` appears in none of them. The only Finsky lines in
any instance (11-12 each) are `SettingNotFoundException for
download_manager_*`, i.e. settings lookups, not a download. **Zero relaunches
fired**, which this time means nothing needed one rather than that the check was
blind (`M1B-E047`).

Verdict: for this image state the kill is ABSENT, and it is absent at its
source — no install round runs at all. The mechanism the fleet lost actors to
needed a WebView session staged in the base image; `M1B-E051` consumed it and
Play did not stage another. This is a statement about the image as it stands
today, not a guarantee: if Play stages a component again, the same race returns,
and the handoff records how to repeat the bake.

`#22` (behavioural equivalence at 120 Hz) is now runnable on a fleet that comes
up 7/7.

**Consequence taken the same day (2026-09-18), from this entry and `M1B-E049`:**
the post-cut recovery is REMOVED. `relaunch_if_activity_lost` and both of its
call sites are gone, with `RELAUNCH_READY_TIMEOUT` and the `point` label that
existed only to say which of them fired. A relaunch after the network is cut
cannot reach home (`M1B-E049`), and keeping it let one bring-up issue up to nine
launcher intents while `MAX_RELAUNCHES` beside it said two. What stands in its
place reports rather than repairs: the post-cut check names the lost activity in
its failure, and `require_game_activity` refuses the instance before the frame
rate is raised. The DETECTION stays exactly as it is — the resumed-activity
oracle of `M1B-E047` is the part that was actually missing — and so does the one
relaunch that works, inside the readiness wait while the radios are still up
(`fe63abc`). `MAX_RELAUNCHES` now describes the whole budget of a bring-up.

Also corrected the same day: the clone's base image booted ROUTABLE after the
bake (`WifiService starting up with Wi-Fi enabled` and a 10.0.2.16 lease in this
run's own logcats, before each bring-up cut it). One writable boot disabled both
radios and read `ip -o -4 addr show` back as `lo` only, with the game identity
and `libunity.so` SHA-256 unchanged and `reboot -p` as the shutdown, so the base
image is offline at boot as of today.

Teardown: per-serial cleanup before kill on all 7 — override reset, libunity
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040, versionCode
1199, versionName 29.0.3, installer com.android.vending, libunity_mounts 0,
bridge_artifacts removed, 7/7 each; then `adb devices` empty, zero qemu in
/proc/*/exe, samplers and logcat captures reaped. Only 5556-5568 addressed; no
reference to emulator-5554 or the canonical AVD.

Source: session scratchpad artifacts `J/` (fleet.log, fleet.json, actors/,
attend.log, seven `emulator-55xx-logcat.txt`).

## M1B-E051 — The bake does not reach a `-read-only` instance, and one clean run at 120 Hz

**Date:** 2026-09-18
**Status:** The bake is NOT established for the fleet recipe; the fleet run
(stage 4) was not started
**Purpose:** Verify `M1B-E050` under the normal recipe before spending a fleet
run on it.

Solo, `-read-only`, cold `-gpu host`, `--cores 4`, 120 Hz, bridge deployed,
offline by interface, one scripted episode, logcat for the whole boot.

What passed: `is at home and offline` with `lo only` by interface; `confirmed at
120 Hz` on all three SurfaceFlinger readings, re-read after the episode and
still 120.00 / 120 / 120.00; deployed bridge SHA-256 read from the device =
7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a; one valid
episode, 0 invalid, `advances_cut_short` 0, `episodes_not_started_fresh` 0, no
termination detail; and **zero `installPackageLI` lines in the entire boot**,
including its online window.

What refutes the bake: `dumpsys package com.google.android.webview` on this
instance reads versionCode **694313738** (133.0.6943.137), the version the clone
already carried — NOT the 792219908 that `M1B-E050` installed and read back
minutes earlier. `codePath=/data/app/~~JIhSP4kU97OQR_gDmW-xAA==/...`,
`pkgFlags=[... UPDATED_SYSTEM_APP ...]`. The writable session did write to disk
(`userdata-qemu.img.qcow2` mtime moved to the bake's shutdown, and the radio
state it changed persisted), so this is not a lost write in the obvious sense;
what a `-read-only` instance sees is not what the writable session committed.
**Mechanism resolved (same session): the install did not persist.** A WRITABLE
cold boot of the same AVD, reading `dumpsys package com.google.android.webview`
before any other command, returns versionCode **694313738** with the old
`codePath=/data/app/~~JIhSP4kU97OQR_gDmW-xAA==/...` — so this is not about which
image a `-read-only` instance derives from (the read-only path was exonerated).
No `RollbackManager` rollback appears in that boot's logcat (only the service's
own start-up lines) and no `installPackageLI` runs in its first minute. Two
facts narrow it further: `userdata-qemu.img.qcow2` mtime DID move to the bake's
`emu kill` (16:49:59) and did not move during the read-only boot, so writes were
flushed; and the radio-enable made about a minute AFTER the install did persist
into later boots while the install did not. An unflushed write would have lost
both. The reading that survives is that PackageManager discarded the update at
the next boot's package scan, leaving the settings change intact.

A repeat bake with `reboot -p` instead of `emu kill` was attempted inside the
budget and produced no data: with the staged session consumed, Play did not
download or stage the update again within a 180 s online window, so nothing was
installed to test the shutdown path with. Whether `reboot -p` would persist it is
still UNKNOWN. Game identity and `libunity.so` SHA-256 were re-read unchanged on
both of these boots.

The zero-install observation is therefore NOT attributable to the bake: a single
~20 s online window with ~70 s of observation after it is too little to say the
round would have come. The N=7 fleet run (stage 4) was not started, because its
falsifiable expectation ("WebView 792219908 read from the device") is already
refuted here.

Teardown: cleanup per-serial on the live instance — override reset, libunity
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040, versionCode
1199, 29.0.3, installer com.android.vending, mounts 0, artifacts removed — then
kill; `adb devices` empty, zero qemu.

Source: session scratchpad `RENDEZVOUS-REMOVAL-DETAIL.md`, artifacts `I/`.

## M1B-E050 — Letting the guest's WebView update land in the clone base image once

**Date:** 2026-09-18
**Status:** The bake itself succeeded and changed nothing else; its effect on
the fleet recipe is refuted by `M1B-E051`
**Purpose:** Remove the recurring kill at its source by letting the one
legitimate system update install, rather than discarding it every boot.

Reversibility first: with no emulator running and `adb devices` empty, the whole
AVD was copied to `~/.local/state/tower-rl/avd-backup-2026-09-18/` — 35 GB,
`MANIFEST.sha256` over all 53 files, `SIZES.txt` beside it.

The clone was then booted ONCE writable, solo on 5556, cold `-gpu host`,
headless. No bridge deployed, no game launched, no frame rate raised, no taps,
no screenshots, nothing about Play disabled or firewalled. The guest came up
with NO routable interface (the base image carried radios off), so `svc wifi
enable` / `svc data enable` were issued — the only device commands of the
session besides the read-backs.

Play's round ran 16:46:34-16:47:00, 25 `installPackageLI` lines, beginning with
`PackageManager: Update system package com.google.android.webview`, and the
first of them landed BEFORE the network was enabled: the session had been staged
in the base image by an earlier online window and finalised locally on boot,
which is the shape the fleet failures have. After 120 s of quiet, read back from
the device before shutdown:
- `com.google.android.webview` versionCode **792219908**, versionName
  **151.0.7922.199** (the stub entry still reads 694313738, as expected);
- the game: versionCode 1199, versionName 29.0.3, installerPackageName
  com.android.vending — unchanged;
- installed `libunity.so` SHA-256
  **ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040** —
  unchanged, so the bind-mount target is intact;
- per-uid game frame-rate override absent, `cmd game list-configs` `Modes: {}` —
  no game-mode state left on the image.
Clean shutdown via `adb emu kill`, waited out; zero qemu afterwards.

Side effect recorded: the radios this session enabled persist in the base image.
Harmless — every bring-up cuts them itself and verifies offline by interface —
but it is a change to the clone and is noted in `workstation-handoff.md`.

Source: session scratchpad artifacts `I/bake.log`, `I/bake-logcat.txt`.

## M1B-E049 — The offline relaunch cannot reach home, so the kill is detectable but not recoverable after the cut

**Date:** 2026-09-18
**Status:** Falsifies the premise the recovery in `M1B-E047`/`M1B-E048` was
built on
**Purpose:** Prove on device the relaunch that had never once fired in a fleet
run, by forcing the fault by hand.

Solo instance, clone AVD, `-read-only`, cold `-gpu host`, 120 Hz. Bring-up
clean: `lo only` by interface, `confirmed at 120 Hz` on three readings, deployed
bridge SHA-256 7a98f50b…f99a read from the device, pid 4132, resumed activity
present. The fault was then forced with `adb shell am force-stop
com.TechTreeGames.TheTower` and nothing else.

Three seconds later: `pidof` empty, `game_activity_present()` False,
`why_not_ready()` = `the game is not running`. **The corrected oracle sees the
fault**, which the substring version of `M1B-E047` did not — this is the
positive half of the result.

`relaunch_if_activity_lost` then fired, printing its point, and re-issued the
launcher intent offline. The game came back as a process and sat at
`main_unavailable` for the full 180 s cap; the call raised `never became ready`.
This is `M1B-E010` acting on the recovery path: a launch with no network stops
at the Firebase online check and the OFFLINE modal and never reaches the battle
home screen. The bridge answered throughout, so the game was running and `Main`
never initialised — the modal, not a dead process.

Consequences: an offline relaunch is not a recovery. The relaunch added in
`fe63abc` can only work inside the readiness wait, where the radios are still
up; the two points added for `M1B-E047` run after the cut and are therefore
detection only, as is any recovery inside episode collection. A recovery that
works needs a ruling — reopen a brief network window, or treat the instance as
lost and re-run it — and neither is taken here.

Teardown: cleanup per-serial before kill (override reset, libunity
ffc1f3ef…0040, versionCode 1199, 29.0.3, installer com.android.vending, mounts
0, artifacts removed); `adb devices` empty, zero qemu. No relaunch touched a
radio, and every interface reading was `lo only`.

Source: session scratchpad `RENDEZVOUS-REMOVAL-DETAIL.md`, artifact `H/proof.log`.

## M1B-E048 — Retry with a working activity oracle: 5/7, and the Play kill moves into collection

**Date:** 2026-09-18
**Status:** Recorded; board #21 NOT closed
**Purpose:** Repeat `M1B-E047` with the activity oracle corrected, and see where
the kill lands once bring-up covers it.

Same recipe as `M1B-E047` (N=7, cold `-gpu host`, `--cores 4 --frame-game-ms
100`, 120 Hz, scripted, 2 episodes per instance), with `game_activity_present`
reading the RESUMED activity line rather than the whole dump. 5/7 reporting, 10
valid episodes, 0 invalid.

All SEVEN instances reached `is at home and offline` and `confirmed at 120 Hz`
on all three SurfaceFlinger readings (display vsync 120.00, uid 10218 game mode
override 120, uid 10218 applied frame rate 120.00), each immediately after its
own bring-up: 5556 was raised and collecting at 16:20:37 while 5558-5568 were
still booting, which is the rendezvous removal visible in the log. Bring-ups ran
37-40 s apart under the retained stagger. The deployed bridge SHA-256 was
confirmed ON ALL SEVEN instances from the device
(7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a), sampled right
after readiness — the first run in which that check produced a digest. No
relaunch fired at any point.

Reporting actors: 18,603-21,053 decisions/hour; 21.8-50.7 valid episodes/hour;
109.2 aggregate. Every fidelity counter zero: `bridge_event_divergence`,
`stale_or_duplicate`, `advances_cut_short`, `episodes_not_started_fresh`,
`invalid_episodes`, with `ADVANCE_TRUNCATED_BY_WALL`, `GAME_TIME_INFLATED` and
`GAME_TIME_DEFLATED` absent. Host peak load 8.15, peak 4 concurrent qemu (a
2-episode actor finishes before the last instance boots), VRAM 5.2 GB.

Both losses are PAST every point bring-up covers, inside collection:
- emulator-5558: bring-up clean, 120 Hz confirmed 16:21:16; killed at
  16:21:16.03 by `Killing ...TheTower (adj 0): stop com.google.android.webview
  due to installPackageLI` mid-episode; `run_episodes.py` died with
  `BridgeDisconnectedError: bridge is not connected`. Nothing watches the game
  during an episode, so no recovery point exists there.
- emulator-5564: bring-up clean, 120 Hz confirmed 16:23:12; NO Play kill in its
  logcat; `run_episodes.py` failed with `the game did not honour speed_max:
  lifecycle_timeout`. The only kill in its log is our own teardown force-stop.
  UNEXPLAINED and the first occurrence of this failure here; its bring-up
  overlapped four actors already collecting at 120 Hz, an overlap the removed
  rendezvous used to prevent. Recorded as a candidate, not a conclusion.

Reading: covering the two bring-up points did what it was meant to and the kill
is now downstream of bring-up entirely. QUALIFIED by `M1B-E049`: those two
points DETECT the kill correctly, but they cannot recover from it, because a
relaunch after the network is cut cannot reach home. Board #21 stays open, and #22
(behavioural equivalence) is still required before training.

Teardown: per-serial cleanup before kill on all 7 — override reset from a
reading (7/7), libunity_sha256
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040 (7/7),
versionCode 1199, 29.0.3, installer com.android.vending, libunity_mounts 0,
bridge artifacts removed (7/7); then `adb devices` empty and zero qemu. Only
5556-5568 addressed; emulator-5554 and the canonical AVD untouched.

Source: session scratchpad `RENDEZVOUS-REMOVAL-DETAIL.md`, artifacts `G/`.

## M1B-E047 — Rendezvous removed, bring-up covered: 6/7, and the relaunch that never fires is a blind oracle

**Date:** 2026-09-18
**Status:** Recorded; the fix in `fe63abc` is CORRECTED by this entry
**Purpose:** Verify the removal of the fleet-wide rendezvous and the two new
recovery points at N=7, and find out why the relaunch added in `fe63abc` had
never once fired.

N=7, cold `-gpu host`, `--cores 4 --frame-game-ms 100`, 120 Hz, scripted, 2
episodes per instance, per-instance `adb logcat` captured from boot. 6/7
reporting, 12 valid episodes, 0 invalid, every fidelity counter zero. The six
reporting actors each confirmed 120 Hz on all three SurfaceFlinger readings.

emulator-5558 (index 1) failed at the post-cut check —  `the game did not
survive the network being cut: the game is not running` — and the NEW recovery
declined to fire, which is the finding of this entry.

**Correction to `fe63abc` (and to the reading in `M1B-E045`).** The relaunch
added there tested `"UnityPlayerActivity" in dumpsys activity activities`, a
substring of the WHOLE dump. `dumpsys` keeps the killed `ActivityRecord` in its
task history, so the game's component name stays in the dump long after the
activity is gone, and the check reported an activity that did not exist. That is
why the relaunch never fired across three fleet runs — not because the kill was
absent. Measured, from this run's logcat for 5558:

    16:12:32.4 START ...UnityPlayerActivity   16:12:33.8 Displayed +1s500ms
    16:12:59.02 Force stopping com.google.android.webview ...: installPackageLI
    16:12:59.06 Killing 4028:com.TechTreeGames.TheTower (adj 0): stop
                com.google.android.webview due to installPackageLI
    16:12:59.06 Force removing ActivityRecord{...UnityPlayerActivity}: app died

Moments later the post-cut check read `pidof` empty AND the activity check True
on the same instance. Presence now means the game holds the RESUMED activity
(`ResumedActivity` line, package and activity both), which is a reading a dead
record cannot satisfy. Confirmed in `M1B-E048`. The recovery those points then
attempt is detection only: `M1B-E049` shows the offline relaunch cannot reach
home, so nothing in this entry should be read as the kill being survivable
after the cut.

This run also confirms the mechanism recorded in `M1B-E045` a second time, on a
different session: the WebView install is what kills the game, the game
identity is untouched, and the kill lands wherever the instance happens to be
rather than only inside the online window.

The deployed-bridge SHA sampler misfired in this run (it matched the
`sha256sum: ... No such file` text), so the per-instance digest is from
`M1B-E048`, not this one.

Teardown: per-serial cleanup before kill on all 7, every identity line as in
`M1B-E048`; `adb devices` empty and zero qemu afterwards.

Source: session scratchpad `RENDEZVOUS-REMOVAL-DETAIL.md`, artifacts `F/`.

## M1B-E046 — Play-update relaunch fix and two 7-actor cold runs: the relaunch never fired, and the fault moved to the fleet rendezvous

**Date:** 2026-09-18
**Status:** OPEN pending a design change (board #21); NOT cleared for
training (board #22)
**Purpose:** Verify the M1B-E045 diagnosis (Play's WebView update kills the
game inside the online window) is closed by relaunch-on-lost-activity, on a
rebuilt bridge, at N=7, with the two review fixes (guarded cleanup, test-log
pollution) folded in.

Setup: relaunch-on-lost-activity added to the readiness wait (cap 2, reset
after each, no radio touched); the online window polled at 1 s (measured
20-26 s across run 1's seven instances) — the window itself is NOT removable:
`M1B-E010` records a cold launch stopping at a Firebase online check and an
OFFLINE modal, so it belongs to the game alone. Cleanup guarded (`cmd game
reset` can no longer abort teardown under `set -e`) with a real
`gameModeOverride` read-back. Deployable bridge rebuilt containing the string
`wall_ceiling`, SHA-256
`7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a`. N=7 cold
`-gpu host` at 120 Hz, scripted, 2 episodes per instance, run twice.

**Result:** run 1 6/7 reporting, run 2 (the single approved retry) 5/7.
**The relaunch path never fired in either run** — no `lost its activity` line
in either run's log. All three failures were instances that had ALREADY
reached "at home and offline," killed after that point, at the two points the
readiness-wait relaunch does not cover: (a) the single post-cut
`why_not_ready` check, which has no relaunch (run 2, emulator-5558: "the game
did not survive the network being cut: the game is not running"); (b) the
idle gap across `run_actors.await_fleet()`'s fleet rendezvous before
`raise_frame_rate`, where nothing re-checks the game at all — a game with no
surface leaves no per-uid `FrameRateOverrides` entry, which is exactly
"uid 10218 applied frame rate absent" (run 1 emulator-5558, run 2
emulator-5560). No logcat was captured for either failure shape: the failure
watcher keyed on the frame-rate message and teardown preceded it — recorded
as the main evidence gap.

Survivors: every fidelity counter zero in both runs (`bridge_event_divergence`,
`stale_or_duplicate`, `advances_cut_short`, `episodes_not_started_fresh`,
`invalid_episodes` all 0; `ADVANCE_TRUNCATED_BY_WALL` never appeared on the
new bridge). Every reporting actor confirmed 120 Hz on all three
SurfaceFlinger readings (display vsync mode, per-uid game mode override,
per-uid applied frame rate). Per-actor throughput 19.4-20.8 valid
episodes/hour. The per-instance deployed-bridge SHA sampler produced nothing
before teardown (`adb shell su 0 sha256sum` returned no digest in time) — mark
that check unconfirmed, not passed.

Device finding: after a real `cmd game reset`, the per-uid `gameModeOverride`
reads 0, not 60; the cleanup read-back now accepts both as "reset."

Test-pollution fix (module-level `EMULATOR_LOG_DIRECTORY` plus an autouse
fixture pointing it at `tmp_path`) verified two ways: canary content written
to the live 5556/5558 logs survives a full `uv run pytest` invocation, and
making the live logs read-only and running the whole suite raises no
`PermissionError` anywhere.

**CONCLUSION / DECISION:** Play downloads its update during the online window
and installs it whenever it likes afterwards, offline. The fleet-wide
rendezvous (raise every instance's frame rate only after every instance is
ready) parks each already-ready instance idle for up to N x 360 s — exactly
the window in which Play kills it. That rendezvous was added on the
raised-peer hypothesis, which `M1B-E043`'s correction already refuted; it is
therefore pure cost with no remaining justification. **DECISION: remove the
rendezvous** — raise each instance's frame rate immediately after it is ready
and offline, and start its episodes immediately — **and extend relaunch
coverage to the post-cut `why_not_ready` check and to the point of use
immediately before the raise.** Ruling reaffirmed: `com.android.vending` is
not disabled or firewalled anywhere in this path. Status: OPEN pending that
change (board #21); NOT cleared for training (board #22).

Source: session scratchpad `PLAY-UPDATE-FIX-DETAIL.md`.

## M1B-E045 — 120 Hz fleet at N=7: the fleet-safe rate, and the mechanism behind the one recurring loss

**Date:** 2026-09-18
**Status:** 120 Hz established as the fleet operating rate; NOT cleared for
training pending board #22 (behavioural equivalence)
**Purpose:** Run the 120 Hz fallback cell left open by `M1B-E044`, and diagnose
the one instance that failed identically in both runs.

N=7, cold `-gpu host`, `--cores 4 --frame-game-ms 100`, `GUEST_FRAME_RATE_HZ =
120`, run twice. Both runs: 6/7 up, and both runs lost exactly the same
instance the same way — emulator-5558 (index 1), `CloneError: emulator-5558
never became ready: the bridge is not answering: bridge closed the stream`,
preceded by `the game is not running` / `the game is still starting:
main_unavailable`, hanging ~260 s, at game start, before the network cut.
Boot serialisation was shown NOT to decide survival: 5558's hang both times let
the 360 s per-instance backstop lapse for three later instances, which then
launched within 3 s of each other and overlapped further boots — every
overlapping cold `-gpu host` boot at 120 Hz still succeeded.

Survivors (six each run, all `confirmed at 120 Hz`): run A 8.32 / 8.397 / 8.499
/ 8.281 / 8.396 / 8.384 ms/frame (median 8.396); run B 8.44 / 8.488 / 8.437 /
8.429 / 8.484 / 8.410 (median 8.440) — against 8.187 ms solo, a 2–4% fleet
penalty (versus 35–43% at 240 Hz, `M1B-E044`). Per-actor decisions/hour: A
18,236–19,444 (median 19,110), B 18,117–19,056 (median 18,645) — SCRIPTED with
no learner attached, so not directly comparable to the 8,904 60 Hz reference
(`M1B-E031`), which had a learner attached. Fidelity clean both runs:
round/budgeted 1.0083–1.0118 (A) / 1.0092–1.0111 (B); decisions/wave
20.6–23.3; zero `advances_cut_short`, `invalid_episodes`,
`episodes_not_started_fresh`, `bridge_event_divergence`, `stale_or_duplicate`,
`GAME_TIME_INFLATED`/`DEFLATED`, `ADVANCE_TRUNCATED_BY_WALL`. Host: peak VRAM
14,531–14,581 MiB with at most 6 qemu alive; per-qemu CPU max 399–401% of a
3,200% ceiling.

**Diagnosis of the 5558 loss (measured, live reproduction with logcat):**
hypotheses H1 (ports/locks) and H4 (per-slot difference) are refuted by
inspection — no per-serial locks exist pre-run, nothing is bound to
5556–5569, and `CloneInstance` derives only ports and `bridge_host_port` from
index while `emulator_command` is index-identical. Index 1 run ALONE at
120 Hz came up clean, confirmed at 120 Hz. Index 0 then index 1 (two instances
only) reproduced the failure exactly, with full logcat captured while it hung.
Mechanism: the game launches fine (`Displayed +594 ms`, `Fully drawn +4.0 s`),
and at +20 s the guest's Google Play installs a WebView update —
`ActivityManager: Killing ...TheTower (adj 0): stop
com.google.android.webview due to installPackageLI`, `Force removing
ActivityRecord{...UnityPlayerActivity}: app died, no saved state`, followed
by a storm of `installPackageLI` force-stops across system apps. The game
restarts ten seconds later only as a service (`JobInfoSchedulerService`), with
no activity; the launcher stays top-resumed and Android freezes the game
process, so the in-process bridge never answers. The fault is
RATE-INDEPENDENT and lands inside the instance's one online window in the cold
`-gpu host` bring-up path; the snapshot path has no online window and is
therefore not exposed. Game identity is untouched (versionCode 1199, 29.0.3,
installer com.android.vending). Why index 1 is always the victim is explained
but not proven: its online window is the second of a session, landing when
Play's freshly-woken update round installs; a different session timing could
move the victim to a different index.

The 0-byte emulator log that misled two earlier readings of this failure is
TEST POLLUTION, not evidence: `launch_emulator` opens
`/tmp/tower-rl-emulator-<serial>.log` with `"wb"` before `Popen`; unit tests
that patch only `subprocess.Popen` (not the temp directory) exercise
`CloneInstance` indices 0 and 1, so every `uv run pytest` invocation truncates
the real 5556 and 5558 logs to 0 bytes. Demonstrated directly: both files were
backdated, the two unit-test files were run, and both logs came back 0 bytes
with a fresh mtime.

**DECISION: 120 Hz is the fleet operating rate**, chosen on fleet efficiency
(2–4% penalty) and survival evidence, not on the 300 Hz solo peak
(`M1B-E042`). The reading of the fleet result is "7/7 minus a pre-existing,
rate-independent instance fault," not a rate-dependent ceiling. A fix is in
progress: relaunch-on-detection and a shortened online window (board #21). Not
cleared for training until board #22 (behavioural equivalence) runs.

Source: session scratchpad `FLEET-120HZ-CELL1-DETAIL.md`,
`FLEET-5558-DIAGNOSIS-DETAIL.md`.

## M1B-E044 — Fleet stagger-backstop defect fixed; 4/7 at 240 Hz persists regardless

**Date:** 2026-09-18
**Status:** Fix verified by a failing-then-passing test; the fleet-boot
failure it targeted is NOT resolved
**Purpose:** Determine whether `BRING_UP_STAGGER_BACKSTOP` timing explains the
N=7 cold-boot losses seen raising the guest frame rate, and re-run the N=7
ladder once fixed.

`scripts/run_actors.py`'s `stagger_bring_up` timed the 360 s backstop from
fleet start rather than from the previous instance's own bring-up start; every
actor thread starts at fleet start, so `gates[i-1].wait(360)` was in effect a
fleet-start deadline. Cold host bring-up took ~165 s per instance, so it
expired for instances 3–6 and their boots overlapped — the same defect
recorded in `M1B-E028`/`M1B-E029`. Fixed with a per-instance `begun` event:
`await_previous` waits for the previous instance to begin, then waits out the
remainder of that instance's own 360 s window. Test
`test_a_slow_boot_does_not_let_the_backstop_overlap_the_next_bring_up`
(3 instances, backstop patched to 0.3 s, each bring-up 0.25 s) was confirmed
to FAIL against the old fleet-start timing and PASS against the fix. `uv run
pytest` 473 passed, `ruff check .` clean, `mypy` clean.

Re-run, N=7, `--cold --renderer host --cores 4 --frame-game-ms 100`, 2
episodes, `GUEST_FRAME_RATE_HZ = 240`: run-log ordering shows exactly one
bring-up in flight at a time (each `launching` line follows the previous
instance's bring-up conclusion), fleet wall dropped from 773 s to 373.8 s, and
still only 4/7 instances survived — 5558, 5560, 5562 failed with `CloneError:
the game did not survive the network being cut: the game is not running`, with
emulator-log causes `Failed to find ColorBuffer` and `Your GPU cannot be used
for hardware rendering`. Every instance was at the stock 60 Hz game rate for
the whole of every bring-up (the four `confirmed at 240 Hz` lines all come
after the last bring-up concluded), and boots did not overlap this time. So
neither a raised peer's game rate nor simultaneous boots explains these
deaths — both candidates raised in `M1B-E043` are excluded.

**DECISION: the backstop defect is fixed and committed as a correctness fix
in its own right, but it is not the cause of the N=7 frame-rate boot losses.**
The 120 Hz fallback cell and a 60-Hz-with-no-`-vsync-rate`-flag control, needed
to tell whether the launch-time `-vsync-rate 240` display mode itself is
implicated during boot, were not run — out of timebox. The fleet-safe rate
remains open (board #21).

Survivors (4 actors, each confirmed at 240 Hz both levers): 5.599–5.931
ms/frame (median 5.742) against 4.144 ms solo at 240 Hz — roughly 38%
contention tax. Per-actor decisions/hour 8,624 / 14,504 / 13,724 / 11,983
(median 12,854); the fleet-clock aggregate (9,997) is not meaningful against
the 62,327 no-frame-rate-change baseline (3 actors dead, 2 episodes each, most
of the wall is bring-up). Fidelity on the survivors: round/budgeted
1.0063–1.0126, decisions/wave 20.06–25.2, zero
`GAME_TIME_INFLATED`/`GAME_TIME_DEFLATED`/`BRIDGE_EVENT_DIVERGENCE`/
`ADVANCE_TRUNCATED_BY_WALL`/`advances_cut_short`. Host: peak VRAM 9,925 MiB
with at most 4 qemu alive; total qemu CPU mean 595%, max 931% of a 3,200%
ceiling.

Teardown verified per-serial on all 7 instances before kill (`cmd game reset`,
libunity SHA-256 match, `bridge_artifacts: removed`); `adb devices` empty and
no qemu in `/proc/*/exe` afterward. Nothing beyond the backstop fix and its
test was committed — the frame-rate wiring itself does not reach a 7/7 gate.

Source: session scratchpad `FLEET-SERIALISED-BOOT-DETAIL.md`.

**Correction (2026-09-18):** The attribution of these three deaths to
`Failed to find ColorBuffer` / `Your GPU cannot be used for hardware
rendering` is REFUTED: both messages also appear 3–4 times each in the
emulator logs of every healthy 120 Hz survivor (5560/5562/5564/5566/5568) in
`M1B-E045` — they are boot noise common to healthy and failed instances
alike, not a fatal signature, and they diagnose nothing here. What is now
known: 5558's death in this run is most likely the same Play-Store
WebView-update fault characterised in `M1B-E045` — the identical
pre-network-cut, game-not-running signature (`main_unavailable`/`the game is
not running`/`bridge closed the stream`) seen there. 5560's and 5562's
deaths ("the game did not survive the network being cut") remain genuinely
UNEXPLAINED; they are not GPU-diagnosed by this entry's evidence. Separately,
`M1B-E045`'s two 120 Hz runs show that overlapping cold `-gpu host` boots are
not themselves fatal — every overlapping boot in those runs succeeded — so the
boot-overlap serialisation this entry fixed does not decide survival either.
Source: session scratchpad `FLEET-5558-DIAGNOSIS-DETAIL.md`,
`FLEET-120HZ-CELL1-DETAIL.md`.

## M1B-E043 — Fleet N=7 at a raised game rate loses 3/7 instances to a cause that is not a raised peer

**Date:** 2026-09-18
**Status:** OPEN; the fleet-safe frame rate is not established
**Purpose:** Find the fleet-safe game rate once the solo knee (`M1B-E042`)
put 300 Hz within reach, by first shipping the wiring and testing it at N=7.

`scripts/clone_session.py` gained `GUEST_FRAME_RATE_HZ = 240` (one constant
driving both `-vsync-rate` at launch and `raise_frame_rate`, which pins the
per-uid game frame-rate override via `cmd game set --fps` and confirms it by
polling `dumpsys SurfaceFlinger` for activeMode vsyncRate, per-uid
gameModeOverride, and per-uid applied frameRate, up to a 20 s timeout);
`scripts/run_actors.py` gained a fleet-wide rendezvous so no instance is
raised while a peer is still booting, then raises each actor after its own
bring-up. Live-confirmed on one instance: `confirm_frame_rate` fails right
after `cmd game reset` (60.00 Hz reported) and passes after `raise_frame_rate`
(240.00 Hz reported on all three readings). `uv run pytest` 472 passed, `ruff`
and `mypy` clean.

N=7, `--cold --renderer host --cores 4 --frame-game-ms 100`, 2 episodes: 4/7
came up. Failures: 5558 (bridge closed the stream), 5562 and 5568 (`the game
is not running`), with emulator-log causes `Failed to find ColorBuffer` and, on
5568, `Your GPU cannot be used for hardware rendering`. **CRITICAL finding
that overturns the premise this run was designed to test:** every instance was
at the stock 60 Hz game rate for the whole of every bring-up — the first
`confirmed at 240 Hz` line in the log comes after the last bring-up concluded
— so the ColorBuffer deaths reproduce with NO peer running at a raised game
rate, which contradicts the hypothesis that a raised peer caused them.

Two confounds were left uncontrolled by this run and were not yet separable:
(1) this used `--cold --renderer host`, not the restore path the clean 7/7
baseline used, and cold bring-up (~165 s/instance) let the then-unfixed
`BRING_UP_STAGGER_BACKSTOP` (timed from fleet start) expire for instances 3–6,
overlapping their boots — see `M1B-E044`, where fixing this alone did not
change the outcome; (2) `-vsync-rate 240` is applied at launch, so the guest
*display* composites at 240 Hz throughout boot even though the game surface
stays capped at 60 — the "boot happens at 60 as today" assumption behind the
design is false for the display, even though the game rate itself is
confirmed unraised during every failed boot.

Surviving actors (240 Hz confirmed): 5.755 / 5.911 / 6.049 / 5.767 ms/frame
(median 5.839) against 4.144 solo; per-actor decisions/hour 10,345 / 11,026 /
13,664 / 13,733 — not comparable to the 8,904 learner-attached baseline
(`M1B-E031`), since this run is scripted with no learner. VRAM peaked at
11,387 MiB with at most 5 instances alive (~2.2 GiB/instance), consistent with
the no-frame-rate-change baseline — no sign a raised rate costs VRAM, though
this is not a 7-instance measurement. Fidelity on the survivors: round/budgeted
1.0105–1.0119, decisions/wave 20.19–22.22, zero
INFLATED/DEFLATED/`BRIDGE_EVENT_DIVERGENCE`/`ADVANCE_TRUNCATED_BY_WALL`.

Teardown verified per-serial on all 7 instances (`cmd game reset`, libunity
SHA-256 match, `bridge_artifacts: removed`) before kill; `adb devices` empty
afterward. Not committed — the fleet gate was not met.

Source: session scratchpad `FLEET-FRAME-RATE-DETAIL.md`.

**Correction (2026-09-18):** The attribution of these three deaths to
`Failed to find ColorBuffer` / `Your GPU cannot be used for hardware
rendering` is REFUTED: both messages also appear 3–4 times each in the
emulator logs of every healthy 120 Hz survivor (5560/5562/5564/5566/5568) in
`M1B-E045` — they are boot noise common to healthy and failed instances
alike, not a fatal signature, and they diagnose nothing here. What is now
known: 5558's death in this run is most likely the same Play-Store
WebView-update fault characterised in `M1B-E045` — the identical
pre-network-cut, game-not-running signature (`main_unavailable`/`the game is
not running`/`bridge closed the stream`) seen there. 5562's and 5568's
deaths remain genuinely UNEXPLAINED; they are not GPU-diagnosed by this
entry's evidence. Separately, `M1B-E045`'s two 120 Hz runs show that
overlapping cold `-gpu host` boots are not themselves fatal — every
overlapping boot in those runs succeeded — so this entry's confound (1),
uncontrolled boot overlap under the then-unfixed backstop, is now known not to
explain these deaths either. Source: session scratchpad
`FLEET-5558-DIAGNOSIS-DETAIL.md`, `FLEET-120HZ-CELL1-DETAIL.md`.

## M1B-E042 — Solo frame-rate ladder to 300 Hz, then a cliff at 360

**Date:** 2026-09-18
**Status:** Establishes the solo knee; fleet-safe rate left to `M1B-E043`
**Purpose:** Find how far the guest-vsync-timer dial (`M1B-E041`) can be
turned before it stops paying, and whether the poll interval needs to change
with it.

Solo instance, both levers (`-vsync-rate N` at launch, `cmd game set --fps N`
after bring-up) set to the same N each cell, adoption gated on
`dumpsys SurfaceFlinger` before measuring:

| cell | requested Hz | ms/frame | fps | vs nominal | qemu %CPU mean/max | decisions/wave |
| --- | --- | --- | --- | --- | --- | --- |
| 60 (baseline) | 60 | 16.17–16.24 | ~61.8 | +2.8% | ~130–142 | ~21–22 |
| 120 | 120 | 8.187 | 122.14 | +1.8% | 162.2 / 178.6 | 21.7 |
| 144 | 144 | 6.831 | 146.39 | +1.7% | 166.2 / 190.4 | 21.706 |
| 180 | 180 | 5.472 | 182.76 | +1.5% | 173.1 / 185.4 | 22.769 |
| 240 | 240 | 4.144 | 241.30 | +0.5% | 199.2 / 220.0 | 22.400 |
| 300 | 300 | 3.352 | 298.33 | −0.6% | 232.0 / 263.0 | 22.538 |
| 360 | 360 | **10.932** | **91.48** | **−75%** | 147.9 / 162.6 | 21.278 |

Every cell measured (not inferred): 3 valid episodes, 0 invalid, zero
`GAME_TIME_INFLATED`/`GAME_TIME_DEFLATED`/`BRIDGE_EVENT_DIVERGENCE`/
`ADVANCE_TRUNCATED_BY_WALL`. round/budgeted stayed in the same 1.009–1.011
band across every cell up to 360.

**300 Hz is the highest rate that pays; 360 Hz is a cliff, not a plateau.**
At 360, the guest reports full adoption (mode 360, renderRate 360, uid
gameModeOverride 360) yet delivers 91.5 fps — worse than 144 — at *lower* CPU
(147.9% vs 232.0% at 300), i.e. the guest is waiting, not working. **Guest
self-report of an adopted rate is therefore not sufficient evidence the
surface is actually running at that rate; measured fps is the oracle**, the
same lesson `M1B-E040` drew from a different symptom. The overshoot band
also closes monotonically as the requested rate rises (+2.8% at 60 down to
−0.6% at 300), consistent with the timer beginning to be missed just before
it breaks outright at 360.

`kFramePollMicros = 2000` was left unchanged and checked, not assumed safe:
decisions/wave stays flat (21.3–22.8) with no monotone drift against rate —
180 Hz (22.77) and 360 Hz (21.28) bracket the range in the wrong order for a
poll-interval effect — so there is no evidence the poll interval needs to
change at or below 300 Hz.

A first attempt to ship this (`GUEST_FRAME_RATE_HZ` wired into
`clone_session.py`, not committed) broke a 2-instance fleet: with actor 0
running at 240 Hz, actor 1's cold launch died with `Failed to find
ColorBuffer` while actor 0's own rate fell to 225.5 fps (−6%) under the
contention — the first sighting of the fleet problem `M1B-E043` investigates
at N=7.

Source: session scratchpad `FRAME-RATE-KNEE-DETAIL.md`.

## M1B-E041 — Three caps in series gate the frame rate; only the outer two matter, and `45f8ea6`'s claim is confounded

**Date:** 2026-09-18
**Status:** Resolves the mechanism; corrects `M1B-E040`'s open confound and
the commit message of `45f8ea6`
**Purpose:** Establish which of the layers between the guest display and the
rendered frame binds the ~60 fps rate seen throughout prior entries, so the
dial can be turned deliberately.

**Measured.** The AVD's `hw.lcd.vsync=60` is the baseline, not compute: 58.9
fps under lavapipe software rasterisation at 1080×1920 (`M1B-E015`) against
61.7 fps on an RTX 4090 at 360×640, and a 9x render-target shrink moved frame
time only 0.3% (`M1B-E030`) — none of that is a rendering-cost signature.
`-vsync-rate 30` at launch produced exactly 30.67 fps
(`VSYNC-RATE-PROBE-DETAIL.md`), showing the guest vsync timer binds downward
cleanly. Frame production is that guest vsync timer, gated by three caps in
series, of which two must both be lifted:

1. **Launch-time display mode** (`-vsync-rate N` on the emulator command
   line, or the AVD's `hw.lcd.vsync`) sets the guest's physical refresh rate.
2. **SurfaceFlinger's per-uid game frame-rate override**
   (`ro.surface_flinger.game_default_frame_rate_override=60` in the system
   image, read-only) pins the game *surface*, independent of the display
   mode, and can only be lifted at runtime per-uid via `cmd game set --fps N
   com.TechTreeGames.TheTower` — confirmed by `dumpsys SurfaceFlinger`
   (`GameFrameRateOverrides`), not by the write call's return code.
3. **Unity's own pacing** (`QualitySettings.vSyncCount`, `Application.
   targetFrameRate`) inside the app process.

An ablation with the bridge `.so` held byte-identical across all four cells
isolated the first two (`FRAME-RATE-DIAL-REPLICATION-DETAIL.md`,
"R1–R4"): both display mode and override at 120 → 8.187 ms/frame (122.14
fps); mode 120 with no override set → 16.177 ms (override absent, i.e. still
pinned to 60); no mode change with override set to 120 → 16.168 ms (mode
absent); mode and override both at 144 → 6.831 ms. **Both layer 1 and layer 2
are necessary and neither alone suffices; they must agree.** Layer 3
(`vSyncCount = 0`, shipped in commit `45f8ea6`) was not ablated in isolation —
the same `.so` ran in every cell — but cell R3 (layer 2 active, `pacing
vsync=0 target=240` confirmed in the app log, layer 1 absent, display at 60)
measured 16.168 ms/frame, indistinguishable from the long-standing pre-`45f8ea6`
baseline of 16.21–16.24 ms. **Layer 3 shows no independent effect in this
data.**

**Correction to commit `45f8ea6`'s message ("vSyncCount = 0 doubles the
rate"): that claim is CONFOUNDED.** The session that produced the 8.186 ms
result (`UNITY-FRAME-PACING-DETAIL.md`) also ran the scratchpad's
`-vsync-rate 120` wrapper — the guest vsync moved 60→120 in the very same
session the bridge change shipped in, and layer 2's override was found
subsequently absent partway through that run's arm A yet the rate held at
122 fps, which independently argues against layer 3 as the explanation. The
measured rate tracks nominal at +1.7% to +2.8% above the requested Hz in
every ablation cell (`FRAME-RATE-DIAL-REPLICATION-DETAIL.md`), consistent
with layers 1+2 alone accounting for the whole effect.

**None of this reaches production.** Production's `clone_session.
emulator_command` passes no `-vsync-rate`, so the emulator vsync is 60 in
every training run regardless of what the bridge does (`M1B-E040`).

Source: session scratchpad `VSYNC-RATE-PROBE-DETAIL.md`,
`GAME-FRAME-RATE-OVERRIDE-PROBE-DETAIL.md`, `UNITY-FRAME-PACING-DETAIL.md`,
`FRAME-RATE-DIAL-REPLICATION-DETAIL.md`.

## M1B-E040 — The 8.186 ms result did not replicate on the first two attempts; a measurement repeated once is not a replication

**Date:** 2026-09-18
**Status:** Methods finding; resolved by `M1B-E041`'s full recipe
**Purpose:** Record why the 122 fps result from `UNITY-FRAME-PACING-DETAIL.md`
failed to reproduce twice before it did, and what that implies for how a
device result gets accepted.

The 8.186 ms/122.16 fps figure was measured once, in one session, under a
scratchpad wrapper (`-vsync-rate 120`, `-gpu host`) with the bridge's
`vSyncCount = 0` change also active — two levers changed together, one
measurement. Two independent later attempts to reproduce it, each holding
the bridge `.so` byte-identical (sha256 `27471a77…b7ca5`) to that session's
binary, failed:

- A 4-instance fleet at `-vsync-rate 120` under `-gpu host` (matching
  production's renderer choice) measured 16.147–16.183 ms/frame across all
  arms, treatment and control alike — indistinguishable from the pre-change
  60 fps baseline. Isolating one treatment instance alone on the host (no
  fleet contention) still measured 16.00–16.22 ms/frame solo, and qemu %CPU
  (mean 132.7, max 135.7 of a 400% ceiling) was *below* the 8.186 ms run's
  160.8, i.e. producing half as many frames per second, not a compute knee.
  (`VSYNC-HOST-PRODUCTION-VERIFY-DETAIL.md`)
- Separately, under `-gpu lavapipe` (production's actual renderer; `-gpu
  host` cannot snapshot this Vulkan app), the same 120 Hz guest vsync
  produced 16.11–16.21 ms/frame, again unchanged from baseline — ruling out
  host contention by also running one instance solo.
  (`VSYNC-PRODUCTION-EQUIVALENCE-DETAIL.md`)

The result only replicated (`M1B-E041`, R1: 8.187 ms, matching the original
to three decimals) once BOTH the launch-time display mode AND the runtime
`cmd game set --fps` override were set explicitly from the original recipe,
under `-gpu host`, on a solo instance — the two-lever combination the
original session had used but the reproduction attempts had not fully
matched (the host-production-verify attempt set only the display mode, not
the per-uid override).

**Methods finding, recorded so it is not repeated:** a result measured twice
within a single session, under a single ad hoc wrapper, is not a replication.
Guest self-report of the requested mode (`androidboot.qemu.vsync=120` in the
emulator's own launch log) is not evidence the game *surface* ran at 120 Hz —
`dumpsys SurfaceFlinger`'s per-uid `renderRate`, read independently of the
write that requested it, is the evidence that actually discriminated the
working recipe from the two that failed.

Source: session scratchpad `VSYNC-HOST-PRODUCTION-VERIFY-DETAIL.md`,
`VSYNC-PRODUCTION-EQUIVALENCE-DETAIL.md`, `FRAME-RATE-DIAL-REPLICATION-DETAIL.md`.

## M1B-E039 — The game is not deterministic under a fixed action sequence; trajectory-level equivalence is not a valid method

**Date:** 2026-09-18
**Status:** Closes trajectory-replay as an equivalence method; no frame-rate
equivalence evidence exists yet as a result
**Purpose:** Test whether a raised frame rate changes gameplay outcomes by
replaying an identical recorded action sequence at two different rates.

Method: record the scripted policy's action list and a per-decision state
signature (wave, cash_log, health_fraction, max_health_log, upgrade levels,
costs, mask) for 3 episodes on one instance, then replay the identical action
list decision-for-decision on the same instance again, and on a second
instance at a different guest vsync, substituting WAIT wherever the mask
refuses a recorded action.

**The control fails before the treatment can be read.** Same instance, same
frame rate, identical starting state, identical replayed actions: episode 0
diverged at decision 13 — cash_log 2.302585 vs 2.397895, i.e. cash 10 vs 11 —
with one action substitution; episodes 1 and 2 started from states that had
already drifted (a leftover run at a different wave). The 60-vs-120 comparison
(record at 5556/60 Hz vs replay at 5558/120 Hz) diverged at the identical
decision 13, in cash, by the identical magnitude, in all three episodes — the
same signature as the self-divergence, not distinguishable from it.

**Consequence: trajectory-level equivalence is an invalid method for this
game.** Because the control (same instance, same rate) fails on its own,
divergence between two arms cannot be attributed to the frame rate;
equivalence work needs a distributional test on low-variance per-wave
statistics instead, not a trajectory match.

**No equivalence evidence for a raised rate exists as a further consequence.**
The frame-rate arm compared here (5558 at `-vsync-rate 120`) was, independently,
measured running at 16.1 ms/frame under `-gpu lavapipe` — not the 8.186 ms
regime at all (`M1B-E040`) — so even setting the method problem aside, this
run carries no evidence about a genuinely raised rate. Separately, the powered
T1-vs-CONTROL wave-distribution run in `M1B-E038` was, by its own finding,
inert for the same reason (both arms measured 62 fps): so across the whole
session, no equivalence evidence for a raised frame rate exists yet (board
#22).

All arms remained inside the fidelity envelope throughout (round/budgeted
1.0046–1.0117, zero `GAME_TIME_INFLATED`/`DEFLATED`/`BRIDGE_EVENT_DIVERGENCE`/
`ADVANCE_TRUNCATED_BY_WALL`), so the nondeterminism is a game behavior, not a
bridge or harness fault.

Source: session scratchpad `VSYNC-PRODUCTION-EQUIVALENCE-DETAIL.md`.

## M1B-E038 — `frame_game_ms` 150 and 200 rejected on fidelity, powered this time; mean final wave is the wrong instrument for equivalence

**Date:** 2026-09-18
**Status:** Supersedes the underpowered n=5 note in `M1B-E033`; 150 and 200
are now REJECTED, not open
**Purpose:** Re-test `frame_game_ms` 150 and 200 with enough episodes to
detect the fidelity failure `M1B-E033` was too small to see, on a 7-instance
fleet running old and new bridge builds side by side in the same wall-clock
window (`FIDELITY-AB-FRAME-SWEEP-DETAIL.md`).

| arm | frame_game_ms | attempted | valid | invalid | invalid reason | pooled round/budgeted |
| --- | --- | --- | --- | --- | --- | --- |
| CONTROL | 100 | 92 | 92 | 0 | — | 1.00985 |
| T1 | 100 | 84 | 83 | 1 | GAME_TIME_DEFLATED | 1.01015 |
| T2 | 150 | 56 | 10 | 46 (82%) | GAME_TIME_DEFLATED ×46 | 0.98922 |
| T3 | 200 | 68 | 2 | 66 (97%) | GAME_TIME_DEFLATED ×66 | 0.98275 |

**150 and 200 are REJECTED**, and not on wave distribution — they fail the
round-clock fidelity invariant outright, below the 0.99 floor, in a shortfall
that grows monotonically and systematically with `frame_game_ms` (invalid
rate 0% → 82% → 97%). This supersedes `M1B-E033`'s n=5 "open, untested"
verdict at 150 ms: at n=56, 150 is rejected. The admissible ceiling stays at
100; the interval 100 < x < 150 remains untested. Mean final wave for T2/T3
carries no weight (n=10 and n=2 valid episodes) and is not the basis for the
rejection.

**Sensitivity finding, from this session's companion T1-vs-CONTROL wave
comparison at `frame_game_ms` 100** (both arms in fact ran at 62 fps — see
`M1B-E039` — so this is a bridge-binary comparison with the pacing change
inert, not a rate comparison, but its statistics bound what any wave-based
comparison in this harness can detect): pooled sd 2.270 waves, harmonic n
87.5 (CONTROL n=92, T1 n=83) ⇒ the smallest difference detectable at 80%
power is **0.963 waves** (16% of the control mean, Cohen's d ≈ 0.42) —
CONTROL−T1 = −0.543 waves, 95% CI [−1.217, +0.133], contains zero, i.e.
indistinguishable at this n. Detecting a difference as small as 0.5 waves
would need approximately 324 valid episodes per arm. **Mean final wave is
therefore the wrong instrument for equivalence testing at achievable sample
sizes; a distributional or per-wave statistic is needed instead** (see also
`M1B-E039`'s point that trajectory-level equivalence is invalid for a
different reason).

Fidelity counters, all arms, zero across CONTROL/T1/T2/T3 except the
GAME_TIME_DEFLATED counts above:
`GAME_TIME_INFLATED`, `BRIDGE_EVENT_DIVERGENCE`, `ADVANCE_TRUNCATED_BY_WALL`,
`advances_cut_short`. No FATAL EXCEPTION/ANR/tombstone in any of the 7
instances' logcat; no actor failed; no episode hit the 240 s hang deadline.

Source: session scratchpad `FIDELITY-AB-FRAME-SWEEP-DETAIL.md`.

## M1B-E037 — Fleet throughput is non-stationary within a run; the acting-lock removal's real gain is +14.3% at matched phase, not the naive −7%

**Date:** 2026-09-18
**Status:** Confirms `M1B-E031`'s unquantified lock-removal candidate, on a
stray 7-actor 250,000-decision run that was found live (not stuck) during an
unrelated teardown check; not a controlled before/after run
**Purpose:** Determine whether removing the `SharedPolicy` acting lock
(`M1B-E031`, commit `81d883a`) actually bought throughput, and establish what a
throughput comparison must control for.

Per-actor decisions/hour is **not stationary within a single run** — it decays
as the run progresses:

| segment | episodes | steps/decision | decisions/episode | per-actor dec/h |
| --- | --- | --- | --- | --- |
| window 0 (ep 1–100) | 100 | 0.1312 | 101.8 | 10,181 |
| window 1 (ep 101–200) | 100 | 0.2500 | 101.3 | 9,636 |
| post-window-1 (ep 201–244) | 44 | 0.2500 | 140.1 | 8,101 |

Decomposing the decline from the window-0 figure: the learner ramp
(steps/decision 0.131 → 0.250, the buffer warming up) accounts for **−5.4%**
(10,181 → 9,636); deepening episodes (101.3 → 140.1 decisions/episode, the
policy surviving to busier, higher-wave gameplay) accounts for the larger
**−15.9%** (9,636 → 8,101). The second is gameplay cost, not host contention:
a busier world costs more simulated work per decision. Any future throughput
comparison must control for both steps/decision and decisions/episode, or it
measures policy progress rather than host speed.

**The lock-removal result.** The pre-removal reference (`M1B-E031`,
`X812-SCALING-DETAIL.md`) is N=7 at 360×640, 8,904 per-actor decisions/hour,
from a short (~10,000-decision) run where the learner had not yet ramped
(~0.131 steps/decision) — the same phase as this run's window 0. Matched at
that phase: **10,181 vs 8,904 = +14.3% per-actor**. The naive comparison
against this run's later, deeper-phase segments (8,303–8,101 vs 8,904, i.e.
"−7%") is a **phase artifact** and should not be used — it compares a
ramped-learner, deep-episode segment against an unramped, shallow-episode
reference. **DECISION: the acting lock was a real per-actor throughput cost,
and removing it bought approximately +14.3% at matched phase.**

**`wait_fraction` is not host idle time.** Per `training.py:336`,
`wait_fraction` is `waits / decisions` — the fraction of decisions where the
policy chose the wait action. It is a policy metric, not a measurement of host
idleness, and must not be read as such.

Caveats: this run was not a controlled before/after measurement (no
after-the-lock-removal run was taken at the identical episode range as the
before figure); the matched-phase comparison relies on both runs' window-0
segments being phase-comparable, which is supported by both having
steps/decision ≈0.13 and shallow early-wave episodes, but was not verified by
an interleaved design.

Source: session scratchpad `teardown-stray-250k.md`, Addendum 3 ("PHASE 1:
BASELINE CAPTURE of the 7-actor run").

## M1B-E036 — Per-decision wall time is 96.5% bridge round-trip with the host essentially idle; host-side Python optimisation is closed as a lever

**Date:** 2026-09-18
**Status:** Decisive at N=1 and N=4; falsifies the GIL-contention candidate
from `M1B-E031` on its own prediction
**Purpose:** Decompose where a decision's wall time goes, to determine whether
host-side serialisation (a GIL or lock effect) or the emulator itself is the
per-decision cost, and whether host-side optimisation (free-threading, batched
inference, process-based actors, observation encoding) is worth pursuing.

Fresh untrained network each arm, identical settings except `--actors`
(`--backbone stacked-dqn --gradient-steps-per-decision 0.25
--warmup-sequences 1 --epsilon-start 0.05 --epsilon-end 0.05
--epsilon-anneal-decisions 1 --frame-game-ms 100 --renderer host --cores 4`).
Pooled over the delta records (first record of each arm dropped as bring-up
tail):

| arm | bucket | share of wall | wall/cpu | wall ms/decision | cpu ms/decision |
| --- | --- | --- | --- | --- | --- |
| N=1 | bridge_round_trip | 96.53% | 235x | 346.93 | 1.475 |
| N=1 | observation_decode | 0.18% | 1.00 | 0.635 | 0.633 |
| N=1 | policy_forward | 0.93% | 1.00 | 3.353 | 3.348 |
| N=1 | learner_step | 2.30% | 1.03 | 8.271 | 8.035 |
| N=1 | blocked | 0.00% | — | 0.000 | 0.000 |
| N=4 | bridge_round_trip | 96.50% | 453x | 350.65 | 0.774 |
| N=4 | observation_decode | 0.09% | 1.00 | 0.314 | 0.313 |
| N=4 | policy_forward | 0.65% | 1.06 | 2.377 | 2.241 |
| N=4 | learner_step | 2.57% | 1.05 | 9.351 | 8.942 |
| N=4 | blocked | 0.15% | ∞ | 0.539 | 0.000 |

N=1: 8 records, 731 decisions, thread-busy 3.81%, accounting error 0.0 s. N=4:
10 records, 3,310 decisions (four threads), thread-busy 3.41%, accounting
error +3.0e-4 s.

**Verdict: the host is idle and the run is emulator-bound.** `bridge_round_trip`
dominates wall time at both actor counts (96.5%) while consuming essentially
no CPU (wall/cpu 235x at N=1, rising to 453x at N=4 — busier, not worse,
because four threads share the same near-zero CPU cost while waiting).
`observation_decode` and `policy_forward` sit at wall/cpu 1.00–1.06, flat
across actor counts — no serialisation cost appears in either bucket as
actors scale from 1 to 4. `blocked` is 0.15% of wall at N=4 and zero at N=1.
This directly falsifies the GIL-contention candidate left open in `M1B-E031`:
its own prediction was that host-side Python work (observation decode, policy
forward) would show growing wall/cpu divergence or blocking as actor count
rose, and neither happened.

**Consequence, stated as the decision this entry closes: host-side *Python*
work — free-threading, batched inference, process-based actors, observation
encoding — is closed as a throughput lever. This is NOT established for host
CPU capacity spent on emulation itself (renderer choice, `--cores`, instance
count, guest frame rate), which remains a live lever.** The profiler's CPU
clock is `thread_time()` (`CLOCK_THREAD_CPUTIME_ID`), which sees only the
actor's own Python thread; it cannot see CPU the emulator process consumes,
so it has no bearing on emulation-side levers by construction. With 96.5% of
wall time in a bucket that is already near-zero CPU *on the Python side*,
none of the Python-side levers can move the number that matters; the
bottleneck is the emulator's own per-advance wall time.

**Caveat on the comparison.** The two arms did not match on episode depth:
39.9 decisions/episode at N=1 vs 66.9 at N=4 (steps/decision matched to 4
decimal places, 0.2494 vs 0.2499, so the learner ramp is controlled for, but
episode depth is not). This does not affect the per-decision bucket
conclusion above, because the bucket shares and wall/cpu ratios are computed
per decision, not per episode, and depend on the emulator's per-advance cost,
not on how many decisions accumulate before an episode ends; a deeper episode
changes how many decisions are counted, not what each one costs.

**Review findings, 2026-09-18 (independent review of the profiler).**
1. The instrument's discrimination was measured directly, not assumed: eight
   contending pure-Python threads show wall/cpu 6.61, and a torch matmul
   offloading to intra-op workers shows 6.07 on the calling thread — so the
   profiler does detect GIL/thread contention when it is present, and this
   entry's 1.00–1.06 readings for `observation_decode`/`policy_forward` are a
   genuine negative, not an artifact of an instrument that cannot see
   contention. The GIL falsification above stands, and is stronger than
   originally claimed.
2. The profile covers N=1 and N=4 only, where `bridge_round_trip` moved
   346.93 → 350.65 ms/decision (+1.1%). The per-actor decay recorded
   elsewhere out to N=8 (9,498 → 8,479 decisions/hour) was not reproduced
   within these profiled arms, so extending this entry's verdict to N=8 is
   inference, not a measured result. Separately, per-decision CPU roughly
   halved from N=1 to N=4 in three buckets (`observation_decode` 0.635 →
   0.314, `policy_forward` 3.353 → 2.377, bridge CPU 1.475 → 0.774
   ms/decision) — most plausibly warm-up and per-episode costs amortised
   over more decisions, since every host bucket is tiny in both arms either
   way. The "453x, busier not worse" wall/cpu phrasing above describes the
   ratio at N=4; it should not be read as a measured contention effect, since
   the CPU side of that ratio fell rather than held steady.

Source: session scratchpad `DECISION-TIME-PROFILE-DETAIL.md`.

## M1B-E035 — 100,000-decision, 4-actor training run: final evaluation up +1.26 over scripted, within-run trend unresolved, replay occupancy indicated as the constraint

**Date:** 2026-09-18
**Status:** Suggestive, not established (~2.1 standard errors); within-run
trend not resolved at this sample size
**Purpose:** Record the largest training run to date and its learner
diagnostics, to judge whether more decisions or more data (replay capacity/
occupancy) is the next lever.

Configuration as recorded in the source: 4 actors, `stacked-dqn` backbone,
commit `f3c177b`, renderer `-gpu host`, `--cores 4`, `--frame-game-ms 100`,
`--seed 0`, replay capacity 4,096, epsilon anneal over the first 10,000
decisions. **Gap in the source:** gradient-steps-per-decision, batch size,
n-step, discount, and the epsilon start/end values are not recorded in
`FLEET-100K-DETAIL.md` for this run and are not stated here — they are not
carried over from the unrelated profiling run in `M1B-E036`, which used a
fresh untrained network under different settings.

**Final evaluation (30 exploration-free episodes, pre-registered).** Mean
final wave **6.833 ± 0.396** (sd 2.167, median 7, range 1–10, 30/30 valid)
against the scripted floor of 5.57 (+1.26) and random floor 5.35 (+1.48).
Against the 40,000-decision single-actor run's 6.033 (n=30): +0.800, se of
the difference 0.550, **t = 1.45 — about 2.1 combined standard errors from
zero against the floor comparison and 1.45 against the prior run, suggestive
of improvement but not established** at this sample size.

**Within-run trend: unresolved.** Post-anneal collection episodes (630,
windows 1–6): OLS slope +0.068 waves per 100 episodes; first half mean 6.149
(n=315, se 0.142) vs second half 6.362 (n=315, se 0.138), difference
**+0.213 ± 0.198, t = 1.07 — not resolved**, i.e. the run's own collection
curve cannot yet distinguish continued improvement from noise.

**Learner diagnostics.** Value fit correlation rose across the run then
retreated: 0.489 (window 0) → 0.584 (window 4) → 0.535 at run end, at replay
occupancy 2,155 of 4,096 sequences (52.6% full). Mean |TD| fell fairly
steadily, 0.359 → 0.250, over the same span. **Data, not reward, is indicated
as the constraint**: the value-fit retreat coincides with replay occupancy
still short of capacity rather than with any drop in the |TD| error signal,
which kept falling — i.e. the learner was not struggling to fit what it had,
it was working with a replay buffer that had not yet filled, consistent with
the run being data-limited rather than reward- or optimisation-limited.

**Throughput.** Per-actor 9,498 decisions/hour (range 9,445.7–9,575.2 across
the four actors); aggregate 37,993 decisions/hour over the collection clock.

Source: session scratchpad `FLEET-100K-DETAIL.md`.

## M1B-E034 — Stray background shells self-match their own `pgrep`, not each other's processes

**Date:** 2026-09-18
**Status:** NEGATIVE result, corrects a prior contamination hypothesis
**Purpose:** Investigate three Claude Code background shells reported as hung
for roughly 16–18 hours during the scaling work below, and determine whether
they had contaminated any measurement.

### Root cause

Three waiter shells were each blocked in a `until <condition>; do sleep …; done`
loop whose condition could never become true, because it used `pgrep -f
"<pattern>"` to detect completion and `pgrep -f` matches against the full
command line — including the querying shell's own `pgrep -f "<pattern>"`
invocation. The pattern is therefore always "found" and the loop never exits.
One waited on `! pgrep -f "pytest tests/unit/test_train_entry_point"` (the
pytest it was waiting for had finished the day before); the other two waited
on file contents (`-s file`, `grep -qE "passed|failed|error" file`) that had
stopped changing hours earlier and never satisfied. All three were confirmed
stuck (elapsed 16:16:30–18:02:52) and were killed by hand; their `sleep`
children were reaped cleanly with TERM, no orphaned sleeps remained.

### DECISION-relevant finding: no contamination occurred

**This is a negative result, correcting a prior hypothesis that an orphaned
process was stealing host CPU during the actor-scaling runs below.** Combined
CPU time consumed by all three shells over their ~16–18 hours of wall time was
**~28 seconds**. They were sleeping in the loop, not spinning, so they did not
compete for CPU, GPU, or memory with any of the measurements in this stage. No
`pytest`, `torch`, or `multiprocessing/spawn_main` process was left running
anywhere on the host.

The same self-match class of bug also inflates naive process counts taken
during this stage: `pgrep -c -f 'qemu-system-x86_64-headless'` reported 8
qemu instances when only 7 were actually running (the querying shell's own
command line matched the pattern), and `pgrep -c -f 'adb .*fork-server'`
similarly reported 2 adb servers when there was exactly 1. Authoritative
process counting in this stage used `/proc/*/exe` rather than `pgrep -f`.

Source: session scratchpad `teardown-stray-250k.md` (addendum, "long-lived
process sweep and stray-shell kill").

## M1B-E033 — `--frame-game-ms 150` at the shrunk render target: underpowered, not evidence either way

**Date:** 2026-09-18
**Status:** Inconclusive by design (n=5); not a fidelity decision
**Purpose:** Check whether 150 ms per frame remains faithful to the game at
the 360×640 render target introduced in `M1B-E030`, as a candidate throughput
lever beyond the 100 ms chosen in `M1B-E017`/prior entries.

One instance, scripted `CheapestFirstPolicy`, 5 episodes at 100 ms then 5 at
150 ms, same session, both at 360×640. **This is the only test 150 ms has ever
received, and it was 5 episodes on one instance — this entry does not claim
equivalence and does not claim rejection.**

| arm | valid | mean wave | sd | dec/wave | round/budgeted | cut short | speedup | dec/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 ms | 5/5 | 5.80 | 2.95 | 21.59 | 1.01104 | 0 | 4.667 | 10,942 |
| 150 ms | 5/5 | 6.60 | 1.52 | 20.46 | 0.98745 | 0 | 6.307 | 13,879 |

The wave difference (+0.80) is not significant at n=5 (se of the difference
~1.49, t = 0.54). What moved is the game-time accounting: round/budgeted drops
from 1.01104 at 100 ms to 0.98745 at 150 ms, i.e. the game credits 2.3% *less*
simulated time per requested millisecond than budgeted — the opposite
direction from `GAME_TIME_INFLATED`, and a value the one-sided inflation guard
(threshold 1.25) cannot see at all since it never crosses 1. This ratio is the
same phenomenon later formalised as `GAME_TIME_DEFLATED` /
`MIN_ROUND_CLOCK_RATIO` = 0.99 at commit `894c36d`.

Power: at n=5 per arm with per-episode sd 1.5–3.0, the detectable difference
at 80% power is roughly 2.9 waves. `M1B-E018` (in this document) detected 250
ms at +2.12 waves with n=8. **This arm could not have detected an effect the
size of the one that got 250 ms rejected.** 150 ms remains an open, untested
throughput lever, not a validated one.

Source: session scratchpad `X812-fid-150.json`, `X812-fid-100.json`,
`X812-fidelity.py`, `X812-SCALING-DETAIL.md`.

## M1B-E032 — `advances_cut_short` rises with actor count: open, not resolved

**Date:** 2026-09-18
**Status:** OPEN — flagged for investigation, not a resolved benign finding
**Purpose:** Record that `advances_cut_short`, zero in every entry recorded in
this document through `M1B-E028`, is no longer zero once actor count climbs
past four.

| run | episodes | `advances_cut_short` |
| --- | --- | --- |
| 4 actors, 1080×1920 (100k reference) | 731 | 1 |
| 7 actors, 360×640 | 104 | 3 |
| 8 actors, 1080×1920 | 120 | 12 |

An `advances_cut_short` event means the bridge's advance loop stopped on a
mid-loop reading that its own settled snapshot then did not corroborate: it
spent neither its game-time budget nor ended on an event still visible in the
settled state. There is no frame budget; an earlier version of this entry said
there was, and that attribution is refuted. The wall-time ceiling — the one
wall-clock-sensitive term in the loop, `kAdvanceWallBudgetMicros` at 15 s — is
refuted as the cause too: a full advance takes about 20 frames at the measured
16.2 ms, roughly 0.33 s, a 46x margin, and 15 of the 16 cut-short advances came
from episodes whose entire advance wall time was under 14 s. It is therefore
benign as to fidelity: the observation the agent receives is the settled, paused
one, and every other health counter (`BRIDGE_EVENT_DIVERGENCE`, `stale_or_duplicate`,
`GAME_TIME_INFLATED`, `episodes_not_started_fresh`) stayed at 0 across all
three runs, with pooled round/budgeted ratios 1.00634–1.00867 and worst-case
ratios 1.01105–1.01401, indistinguishable from the single- and four-actor
baselines. But the count is not flat, and its cause relative to actor count is
not established by this evidence — treat the trend as open, while treating the
fidelity question as answered: the settled observation is what the agent sees.

**Follow-up taken (2026-09-18).** Because the two conditions the counter could
have meant have opposite severity, the bridge now reports a wall-truncated
advance under its own reason, `wall_ceiling`, instead of letting it fall through
to `budget_exhausted`, and the host fails any episode containing one by name
(`ADVANCE_TRUNCATED_BY_WALL`): an advance cut off by how long the host took is
load-dependent and not comparable with one that was not. With a 46x margin the
invariant should never fire; it exists so that actor scaling large enough to
change that is heard rather than absorbed into this counter. `advances_cut_short`
now means only the benign mid-loop case above. The diagnostics `clockprobe` line
also names the exit that ended each loop (`stopped_on=`).

Source: session scratchpad `X812-SCALING-DETAIL.md` ("Environment health"
table).

## M1B-E031 — Per-actor throughput decays with actor count; cause not separated

**Date:** 2026-09-18
**Status:** OPEN — two candidate causes not distinguished by this evidence
**Purpose:** Explain why per-actor decisions/hour falls as actor count rises
even though per-emulator CPU stays flat and no host resource is near a limit
(see `M1B-E029`).

Per-actor decisions/hour: 9,498 at N=4 (1080×1920) → 8,904 at N=7 (360×640) →
8,479 at N=8 (1080×1920), while per-emulator CPU stays flat at ~96–97% median
across all three counts. A single instance running the same scripted policy
with **no learner attached** reaches 10,942 dec/h at 360×640 — higher than any
fleet actor's per-actor figure — so the emulators themselves are not the
limiter; the marginal loss with N sits in the host-side Python process.

Two candidates are named but **not separated by this evidence**:

1. The former `SharedPolicy` acting lock, under which every actor's forward
   pass and every learner gradient step serialised through one lock; gradient
   steps scale with aggregate decisions at 0.25/decision, so the learner's
   share of lock time roughly doubles from N=4 to N=8. This lock was removed
   at commit `81d883a` (every actor given its own copy of the network),
   **without a controlled before/after measurement** — its benefit, if any,
   is unquantified by this document.
2. GIL contention during host-side observation decoding, independent of any
   lock.

Both are host-side serialisation and would look identical in the CPU, load,
and GPU counters gathered here. What this stage establishes is only that the
bottleneck is host-side, not device-side.

Source: session scratchpad `X812-SCALING-DETAIL.md` ("What binds: host-side,
not the emulators").

## M1B-E030 — The render-target shrink does not save VRAM; it buys throughput

**Date:** 2026-09-18
**Status:** NEGATIVE result on the stated goal, POSITIVE on a different axis
**Purpose:** Shrink the emulator render target to raise the actor-count VRAM
ceiling ahead of the scaling runs in `M1B-E029`.

The `tower_rl_instrumented_api36` clone AVD's `hw.lcd.width` /
`hw.lcd.height` / `hw.lcd.density` were changed from 1080 / 1920 / 420 to
360 / 640 / 140. The logical density-independent size is unchanged at
411×731 dp, so the game's own UI layout is unaffected. Only the clone AVD was
touched; the canonical `tower_rl_api36_play_x86_64` play AVD was not touched.
The original config was backed up as `X812-avd-config.ini.orig` in the session
scratchpad before the change; the change was left in place afterward.

**This did not meaningfully reduce VRAM, which was the reason it was tried.**
Measured per-instance VRAM fell only ~8%, from 2,421 MiB (1080×1920, two-point
fit) to 2,234 MiB (360×640, one instance measured directly). Under `-gpu host`
VRAM is dominated by the game's texture/asset working set, not the
framebuffer — the 1080×1920×32 framebuffer itself is only ~8 MB, so a 9x
reduction in pixel count barely moves total VRAM.

Its actual payoff was throughput, measured on a single scripted instance with
no learner attached: 8,850 → 10,942 decisions/hour (+24%), and per-qemu CPU
165% → 138.7%. That single-instance throughput figure supersedes the
1080×1920 100 ms scripted single-actor figure (8,850 dec/h) recorded in
`M1B-E028` as the current single-instance reference at this profile; the
`M1B-E028` figure remains an accurate record of its own (1080×1920) condition
and is not corrected in place.

Source: session scratchpad `X812-SCALING-DETAIL.md` ("Render target change"
and "Memory and VRAM"), `X812-avd-config.ini.orig`.

## M1B-E029 — Actor-count scaling to eight: the operating point is 7, bound by VRAM

**Date:** 2026-09-18
**Status:** Commit `f3c177b` (measurement build); render target per `M1B-E030`
**Purpose:** Find the actor count this workstation can reliably run, and the
resource that binds it, ahead of longer training runs.

### Measured throughput and bring-up

| N | resolution | per-actor dec/h | aggregate dec/h | bring-up result |
| --- | --- | --- | --- | --- |
| 1 (scripted, no learner, 100 ms) | 360×640 | 10,942 | — | n/a |
| 4 (reference, 100k run) | 1080×1920 | 9,498 | 37,993 | 4/4 up |
| 7 | 360×640 | 8,904 | 62,327 | 7/7 up |
| 8 | 1080×1920 | 8,479 | 67,384 | 8/8 up |
| 8 requested | 360×640 | — | — | **7 came up; the 8th never left `main_unavailable` within its 300 s timeout** |

The 8-at-360×640 bring-up failure is a **measurement**, not an estimate: seven
instances reached ready in 40–52 s each (total 310 s); the eighth was still
polling `main_unavailable` when its 300 s timeout expired. Peak VRAM during
that attempt was 20.3 GiB, 83% of the card's 24,564 MiB.

### What binds: VRAM, not CPU or RAM

VRAM ceiling arithmetic (**inferred, not measured directly** — it is a linear
extrapolation from the measured 2,234 MiB/instance and measured idle/fixed
overhead in `M1B-E030`): 80% ceiling of 24,564 MiB = 19,651 MiB; fixed
overhead ~1,850 MiB (724 MiB idle + ~1,100 MiB learner CUDA context); N =
(19,651 − 1,850) / 2,234 = **7.97**.

CPU and RAM were **not** binding at any count measured here. CPU: 70% of
3,200% = 2,240%, and per-emulator CPU under load was measured at ~97%,
implying headroom to N≈23; total qemu CPU under the 7-actor training load was
measured at 691.7% of 3,200% available (21.6%). RAM: 70% of 112.7 GiB = 78.9
GiB against a measured 6.87 GiB RSS per instance, implying headroom to
N≈11.5.

**DECISION: 7 actors is the reliable operating point.** 8 actors at
1080×1920 did come up cleanly (21.36 GiB, 87% VRAM) earlier the same session,
so the ceiling is marginal rather than a hard wall — but 8 actors at 360×640
lost the eighth actor at bring-up the same day under the same arithmetic
region, so 7 is treated as reliable and 8 as a count that sometimes loses an
actor at bring-up.

Host: i9-14900, 24 physical cores / 32 threads, 125 GiB RAM, RTX 4090 24,564
MiB VRAM.

A defect was also observed here and is recorded, not fixed by this entry: an
instance that fails bring-up is never added to `started`, so
`tear_down_fleet` does not tear it down — `emulator-5570` was left running
with the bridge deployed after the 8-requested run exited 0, and was cleaned
up by hand.

Source: session scratchpad `X812-SCALING-DETAIL.md`.

## M1B-E028 — Multi-actor scaling is linear to four actors

**Date:** 2026-09-17
**Status:** Commit `f251dc8`. Source: this session's scratchpad
`X-SCALING-DETAIL.md` with `X-n{1,2,4}.json`, `X-cpu-n{1,2,4}.log`,
`X-avd-{before,after}.txt`.

Setup: 32-core host, `--cores 4` per instance, `-gpu host`,
`--frame-game-ms 100`, scripted policy, 5 episodes per actor, `--cold`.

| N | Valid/attempted | Aggregate episodes/h | Aggregate decisions/h | Per-actor decisions/h | Speedup | Per-qemu CPU mean/max | Total CPU mean/max | Max 1-min load |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5/5 | 85.6 | 8,850 | 8,850 | 4.63 | 165%/445% | 171%/445% | 1.96 |
| 2 | 10/10 | 117.4 | 18,704 | 9,435 and 9,270 | 4.70–4.73 | 152%/448% | 276%/884% | 4.84 |
| 4 | 15/15 across 3 of 4 actors | 206.9 | 27,781 | 9,150–9,416 | 4.69–4.73 | 156%/498% | 563%/1835% | 10.71 |

Fidelity held at every N: all health counters 0, 30/30 valid,
round/budgeted max 1.014 and mean 1.007–1.011, decisions/wave 20.95–22.0. No
knee was reached in steady state — per-actor decisions/hour is unchanged
from N=1 to N=4, against 563% mean CPU of 3,200% available and disk never
above 2.5%.

What did bind was simultaneous cold bring-up: four concurrent boots peaked
at 1,835% CPU and load 10.71, and one actor (`emulator-5562`) never left
`main_unavailable` within its 300 s timeout while its peers reached home in
60–90 s. Fixed at `f408788` by sequencing bring-ups on readiness.

Verified: several `-read-only` instances co-exist on one AVD, each answering
on its own forwarded port (47652–47655), offline confirmed per instance by
interface; one actor failing left the others intact and still able to tear
down; the base AVD images were byte-identical before and after (qemu opens
them read-only and writes to a private overlay).

Planning figures: ~9,200 decisions/hour per actor, ~37,000 decisions/hour at
four actors, so 100,000 decisions in roughly three hours; ~65–70 valid
episodes/hour per actor.

## M1B-E027 — First RL training run on the real game: flat

**Date:** 2026-09-17
**Status:** `source_revision` recorded as `cddc6e6`, behaviourally `9cc3a233`
(the two intervening commits touched only `scripts/clone_session.py`, its
tests, and the handoff). Source: this session's scratchpad
`M2-TRAINING-DETAIL.md`, `T-health.json`, `T-train.log`. Artifacts under
`~/.local/state/tower-rl/runs/session-20260917-201639/`.

Setup: one arm `stacked-dqn`, 20,000 decisions, `--frame-game-ms 100`,
`-gpu host`, one actor, evaluation every 20 episodes with 5
exploration-free episodes per point, 2.675 hours, MLflow run
`cff5a0870e6146cd81a46d753a5a864d`.

Curve (decisions / wall s / optimisation steps / mean wave / sd / final
waves / valid-invalid / checkpoint):

| Decisions | Wall s | Opt. steps | Mean wave | sd | Final waves | Valid/invalid | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2,543 | 1,219 | 3,748 | 5.80 | 2.17 | 3,7,7,4,8 | 5–0 | `367ed9e14bcb` |
| 5,130 | 2,487 | 8,922 | 5.40 | 2.30 | 3,6,8,3,7 | 5–0 | `6e224e5b4a56` |
| 7,937 | 3,837 | 14,536 | 6.50 | 3.42 | 8,2,10,6 | 4–1 (`action_pipeline_failed`) | `f0ff15fde7ec` |
| 10,800 | 5,162 | 20,262 | 3.80 | 1.92 | 5,1,4,3,6 | 5–0 | `f36187cb3b2c` |
| 13,216 | 6,348 | 25,094 | 5.20 | 3.90 | 1,1,9,8,7 | 5–0 | `1b5fb2399405` |
| 15,967 | 7,708 | 30,596 | 6.00 | 1.87 | 6,9,6,5,4 | 5–0 | `a0a358605b3c` |
| 18,510 | 8,992 | 35,682 | 6.40 | 1.14 | 5,7,6,6,8 | 5–0 | `39dc3f3330b7` |

References: scripted 5.57, random 5.35, wait 1.87.

Trend: OLS slope +0.021 waves per 1,000 decisions, +0.42 across the budget;
first three points average 5.90, last three 5.87; with 5 episodes per point
against a per-episode sd of 2–3, each point carries about ±1 wave of
standard error, so neither the 3.80 dip nor the closing 6.40 is a trend. The
honest reading: after 20,000 decisions no learning is demonstrated; the
learner sits at the scripted/random floor and clearly beats wait, which a
random policy also does.

Training health: loss 0.0626 → 0.0355, gradient norm 5.37 → 2.088, mean
|TD| 0.292 → 0.320 (did not fall); 38,810 optimisation steps; replay 430
sequences of 4,096 capacity, 0 rejected, 0 evicted, 620,960 draws.

Environment health, the entry's other important result: 153/153 collection
episodes valid, all `game_over`, `invalid_detail` empty,
`BRIDGE_EVENT_DIVERGENCE` 0, `stale_or_duplicate` 0, `advances_cut_short` 0
in collection (2 in one evaluation), `episodes_not_started_fresh` 0,
`GAME_TIME_INFLATED` 0, recovered transients 0, round/budgeted ratio 1.0090
pooled with worst episode 1.0123; 70.3 episodes/hour overall, 57.2
collection episodes/hour.

Record the Lead's leading suspects, clearly as hypotheses not conclusions:
too little data (153 episodes), and a replay ratio badly mismatched to it —
each of the 430 sequences was drawn about 1,400 times, and with sequence
length 80, burn-in 40 and n-step 5 each gradient step consumes roughly 560
learnable transitions, so at 2.0 gradient steps per decision that is on the
order of a thousand replayed transitions per generated one where R2D2 uses
roughly 4–8.

## M1B-E026 — Milestone 1 gate: snapshot, renderer equivalence, stability

**Date:** 2026-09-17
**Status:** Commit `9cc3a233`. Source: this session's scratchpad
`M1-GATE-DETAIL.md` with `A1-cold.log`, `A2-restore.log`, `A2-iface.log`,
`A3-episode.json`, `A4-mismatch.log`, `B-host.json`, `B-cpu.log`,
`C-stability.json`, `Z-cleanup.log`. Three gates.

**Gate A — snapshot bring-up, lavapipe.** Cold `up` 44 s saved the keyed
snapshot; second `up` restored in **10.48 s**; a once-per-second interface
poll showed only `lo` with `wifi_on=0`/`mobile_data=0` from the first
reachable sample, and the restore path never deployed or enabled radios; the
restored instance's bridge answered on 47652 and ran an episode 1/1 valid; a
deliberate key mismatch took the cold path. **Defect: under `-gpu host` the
emulator refuses `snapshot save` with `KO: Snapshot save is skipped. Reason:
UNSUPPORTED_VK_APP`, and `save_snapshot` printed that line then reported
success anyway** — the fallback was sound, the success claim was not; fixed
at `cddc6e6`, and snapshots are now declared lavapipe-only.

**Gate B — `-gpu host` equivalence, 5 episodes each.** Host vs lavapipe:
ratio 1.0108 vs 1.0118, decisions/wave 21.0 vs 20.73, mean final wave 6.2
(sd 1.92) vs 6.0 (sd 3.54), 5/5 valid both, `invalid_detail` empty, all
health counters 0, speedup 4.648 vs 4.62; one qemu process at mean 154% CPU,
peak 180%. Caveat: n=5 cannot detect a sub-wave fidelity difference
(M1B-E018 puts that near 97 per arm), so this is consistent equivalence, not
proven equivalence.

**Gate C — stability, `-gpu host`, 25 consecutive episodes.** 25/25 valid,
`invalid_detail` empty, ratio 1.0105 (worst episode 1.013), decisions/wave
21.51, mean final wave 5.8 sd 2.236, **80.6 episodes/hour**, boundary 7.01
s/episode (sd 2.19), all counters 0, no crash.

Conclusion: `-gpu host` adopted as the mandatory training renderer.

## M1B-E025 — The 1x speed pin made effective

**Date:** 2026-09-17
**Status:** Commit `9cc3a233`. Source: this session's scratchpad
`SPEED-PIN-DETAIL.md`, `pin-arm.json`, `pin_probe.py`, `speed_control.py`.

`set_speed` wrote `Main.gameSpeed` and confirmed by reading back the slot it
had just written. `gameSpeed` is the rate the world is running at NOW — the
bridge's pause puts it at 0, and the game restores its own remembered speed
(`Main.gameSpeedMemory`) on unpause, so anything written while the world
stands still is overwritten before the world next moves. That is why a
confirmed pin sat beside a 1.5x world.

The game's own controls DO take: live readings (each a fresh connection's
handshake, taken before the bridge pauses) gave `SpeedChangeUp` 0.0 → 1.0 →
1.5 and `SpeedChangeDown` 1.5 → 1.0; the ladder on this account is 0 – 1.0 –
1.5, with 1.5 both default and ceiling. The pin is now `speed_max` then
exactly one `speed_down`, applied at the episode boundary in `_start_round`.

Hazard, proven accidentally: four presses put the world at 0 and froze the
round clock (two probe timeouts), and `SpeedChangeMax` pressed FROM 0 did
not recover it — only `SpeedChangeUp` did.

`game_speed` is truthful rather than broken: 0.0 is the paused speed, and
read live it tracked every press. The earlier 1.0088 probe at 28d691f could
not be reproduced and is attributed to the remembered speed left at 1.0 in
that app process by a crashed run — state, not a property of that build.

Verification arm, production build, 5 scripted episodes at
`--frame-game-ms 100`: 5/5 valid, ratio **1.0118**, decisions/wave **20.73**,
mean final wave **6.0** (7,8,1,4,10), zero `GAME_TIME_INFLATED`, empty
`invalid_detail`, speedup 4.62; same host and session before the fix, 1.5785
twice.

## M1B-E024 — Verification of the non-visual bring-up, field types, and the pin: three of five stages pass, the fourth fails on the post-advance re-pin, and the diagnostic isolates the cure

**Date:** 2026-09-17
**Status:** A five-stage device chain against commit `28d691f`. Stages 1 and 2
pass outright. Stage 3 fails, but not on the defect it was sent to check — the
`GAME_TIME_INFLATED` guard was never touched, and the failure is a second,
independent bug the same commit introduced. A scratchpad diagnostic isolates
the cure and it is the one the following commit ships. Stages 4 and 5 did not
run, per fail-fast ordering. Host was not quiet: two foreign `pytest`
processes held ~1250% CPU each and load sat near 60 on 32 cores for the whole
session, so every throughput figure recorded here is a lower bound and no
throughput comparison was attempted.

### Stage 1 — non-visual bring-up: PASS

The OFFLINE modal returns `main_unavailable` on every one of six probes over
about 50 s, reconfirmed on the production build after its own cold launch.
`s1-offline-screen.png` confirms the screen is the modal itself — "Checking
Firebase Online Status… 7%" — not a splash. `launch` progressed "the game is
not running" → `main_unavailable` → ready in 46.0 s (diagnostics build) and
45.4 s (production build), offline reverified by interface at every
checkpoint. One transient right after deploy's cold launch: "bridge closed
the stream" for a few seconds while the bridge loaded. The oracle this stage
exists to confirm holds: the splash/OFFLINE modal reports `main_unavailable`,
never `no_initialized_run`.

### Stage 2 — declared field types: PASS

`gameSpeed`, `gameMaxSpeed` and `gameplayTimeThisRound` are all declared
`System.Single`; `playTime` is `System.Double`. No width bug: the bridge
already writes `gameSpeed` from a C++ `float`, so the write width matched the
declared type all along. 940 distinct `Main` fields logged in full
(`s2-main-field-types.txt`).

### Stage 3 — FAIL, on the post-advance re-pin, not on inflation

The run dies deterministically on the second decision of episode 1 with
`BridgeStaleObservationError: command does not bind the latest observation`,
reproduced twice. The traced sequence (`s3-trace.log`):

```text
advance    expected=11 -> result obs seq 12
set_speed  expected=12 -> result obs seq 13   (post-advance re-pin)
advance    expected=12 -> RAISED BridgeStaleObservationError
```

`set_speed` consumes an observation sequence like any other command. The
environment's cached state still names the advance's settled observation
while the bridge — and the client — have moved on to the one the re-pin
produced, so the next advance is refused as stale. The exception escaped
`evaluate` and killed the process rather than being classified as one lost
episode. `GAME_TIME_INFLATED` did **not** appear; the 1.25 threshold was
never exercised.

### Stage 3 diagnostic — the boundary pin alone cures the inflation

A scratchpad monkeypatch (`probe_nopostpin.py`) suppressed only the
post-advance re-pin, leaving the episode-boundary pin and the threshold
untouched; no repository edit. 5 episodes, lavapipe, 100 ms/frame:

| Quantity | Value |
| --- | --- |
| `total_round_seconds / total_budgeted_game_seconds` | 754.35 / 747.80 = **1.0088** |
| `decisions_per_wave` | **22.5** (21 expected at 1x; 15.0 was the inflated reading) |
| Mean final wave | **4.6** (waves 1, 7, 8, 3, 4; sd 2.88) |
| Valid episodes | 5/5 |
| `invalid_detail` | `{}` |
| `BRIDGE_EVENT_DIVERGENCE` | 0 |
| `advances_cut_short` | 0 |
| `episodes_not_started_fresh` | 0 |
| Boundary per episode | 2.3–6.5 s |
| Speedup | 1.282 (host contended) |

The boundary pin alone — with the post-advance re-pin removed — cures the
inflation. Caveats: n=5, the probe deliberately disabled the exact code under
test rather than fixing it, and the host was heavily loaded, so every
throughput figure here is a lower bound, not a measurement of the fix at
rest.

One earlier attempt of this probe failed at reset with "the instance did not
reach an active run", explained by mid-round state the crashed stage-3 run
left behind (`health=-nan game_over=1` in logcat); the retry from a clean
idle home ran to completion.

**Corroborating evidence, taken live.** A handshake taken while the crashed
run's round was still active read `game_speed` **1.5** — direct evidence
that the game holds its account-level 1.5 ceiling during a round unless
pinned, and that the field does witness the running world's rate when read
live; it reads 0.0 only in the paused observations the host normally takes
(idle-home handshakes read 0.0 twice in the same session).

### Stages 4 and 5 — NOT RUN

The `-gpu host` renderer equivalence (stage 4) and multi-actor scaling
(stage 5) were not attempted, per fail-fast ordering on the stage 3 failure.

### Snapshot restore

`nonvisual_baseline_home_offline` restores in 10.4 s: game running, offline,
never connected. Its embedded bridge does not answer the current client
("bridge closed the stream"), so a restored snapshot still needs a redeploy
of the current build, which cold-launches the game and reopens the online
window.

### The fix that followed, commit `cf504b8`

The post-advance re-pin is removed. The episode-boundary pin and the
`GAME_TIME_INFLATED` ratio guard (threshold 1.25) are retained. Stale-sequence
errors are now translated to `RunPortError` so they cost one episode rather
than the run. The adapter now issues no command of its own initiative during
a round, enforced by construction (`_command_between_rounds` raises if a
round is in progress).

### Cleanup

`instrumented_bridge.sh cleanup` run on both the worked instance and the
restored one: libunity SHA-256 `ffc1f3ef…dd0040`, `versionCode 1199`,
`29.0.3`, installer `com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`; emulators killed, `adb devices` empty, no qemu
process. No taps issued, no coins spent, no progression change.

Source data: this session's scratchpad `DEVICE-VERIFY-CHAIN-DETAIL.md`,
`s3-trace.log`, `s3-probe2.json`, `s2-main-field-types.txt`,
`s1-offline-screen.png`.

## M1B-E023 — The 1.5x game-time inflation, diagnosed: it lands within advances, not between them, and traces to the account's own speed ceiling

**Date:** 2026-09-17
**Status:** Analysis of the records `M1B-E022` produced, on the same device
session — no new device run. Explains the `round/budget` ratio of 1.512
flagged there and states its consequence for that run's numbers.

`total_round_seconds / total_budgeted_game_seconds` read **1.512** in
`M1B-E022`'s `-gpu host` arm, against clean prior runs at 1.011. Per-frame
credit was 1.625 against the known-good 1.069 — an inflation of 1.520x,
identical across all six episodes and **both** renderers, spread ±0.002,
even though wall-time per frame differed 3.8x between the two renderers.

**The extra time arrives within advances, not between them.** An inert pause
between advances would scale with wall time and diverge roughly 8 ms/frame
between the two renderers; the observed divergence was 0.1 ms, excluding the
pause hypothesis at about 27x margin. The `round/budget` metric is also
structurally blind to any leakage that happened between advances rather than
inside them, since it only sums per-advance deltas — a second, independent
reason not to read the metric as ruling out an inter-advance cause on its
own, though the divergence measurement already does.

**Cause.** This account's speed ceiling is 1.5, and `gameSpeed` defaults to
1.5. Starting a round through `BattlePanelUI.StartNewRound` (`M1B-E022`) does
not leave the world at 1x the way the old tap path did. `_pin_game_speed`
could not catch this: `game_speed` reads 0.0 in the paused observations the
host normally takes, so the guard the pin used to have — skip if already at
1x — could never see the true rate, and the bridge's `set_speed` confirmation
only reads back the slot it just wrote, not the world's running rate.

**Corroboration.** `decisions_per_wave` fell from 21.1 to 15.0 and mean final
wave rose to 8.0 in the inflated run — the profile of a coarser effective
step, matching the previously-rejected 250 ms arm (`M1B-E018`) rather than
anything about policy quality.

**Consequence, stated plainly.** Any comparison of the `M1B-E022` run against
the 1x-measured floor (`M1B-E021`) would have been invalid, and the higher
waves reported there would have flattered the result rather than reflecting
it. See `M1B-E024` for the device chain that verified the fix, and commit
`28d691f` for the ratio guard this analysis led to.

Source data: `M1B-E022`'s own records — this session's scratchpad
`NONVISUAL-BOUNDARY-DETAIL.md`, `nvb-host.json`, `nvb-lavapipe.json`.

## M1B-E022 — The non-visual episode boundary is found by dumping IL2CPP metadata, not by guessing names, and `-gpu host` clears its throughput arm

**Date:** 2026-09-17
**Status:** Closes the receiver hunt left open since `M1B-E013`: the round-start
control is found, confirmed on device, and the screen tap is retired
entirely. The `-gpu host` renderer arm this entry also ran is later shown
(`M1B-E023`) to have been measured under a 1.52x game-time inflation, so its
wave figure does not survive as reported.

Commit `0a2366f`. Sources: this session's scratchpad
`NONVISUAL-BOUNDARY-DETAIL.md`, `nvb-host.json`, `nvb-lavapipe.json`,
`nvb-logcat-1.txt`.

### The receiver hunt, closed by dumping IL2CPP metadata

The round-start control was found by enumerating the game's own IL2CPP
method inventory (450 `Main` methods, plus a cross-class scan) rather than
by guessing object or method names. It is `BattlePanelUI.StartNewRound`, on
the GameObject named `BattlePanel` — the BATTLE button's own component — and
**not** on `Main`. `Button_GameEndPanelGoHome`, delivered to `Main`, works
(wave 2→0, screen goes home), which proves delivery to `Main` was never the
problem: `Main.StartNewRoundFunction` and `Main.AutoRetryBattle` are
delivered and do nothing. Both are deleted from the adapter rather than kept
as fallbacks. This retires the open question left standing since `M1B-E013`.

`BattlePanel` is unreachable — an inactive object — while the result panel is
up, so the episode boundary is `go_home` then `start_round`, and no retry
control is needed at all.

### The gated screen tap is gone

Nothing in the RL loop reads a pixel any more. Boundary cost fell from 7.25 s
(the tap path) to **1.716 s** measured from a terminal run (0.750 s when the
run was already active). Snapshot `nonvisual_baseline_home_offline` was saved
**and** restored — at home, offline, never connected — closing that
follow-up from the handoff's next-slice list.

### `-gpu host` arm

3/3 valid, `invalid_detail {}`, 0 `advances_cut_short`, 0
`BRIDGE_EVENT_DIVERGENCE`, mean wave 8.0, 15.04 dec/wave, 298 ms/advance,
speedup 5.936, 64.6 episodes/hour, qemu 118–127% CPU. Same-session lavapipe
reference: speedup 1.907, 24.1 episodes/hour, qemu 284–308% CPU — starvation
capped, since other processes on the host held roughly 1500% CPU each during
this arm, against the quiet-host lavapipe reference of 1000–1422% CPU.

**Flagged prominently: this run's `total_round_seconds /
total_budgeted_game_seconds` was 1.512**, and the mean wave of 8.0 is later
shown (`M1B-E023`) to be an artifact of a faster world, not a better policy
or a faithful throughput comparison. Read the `-gpu host` figures above as an
uncorrected measurement pending that diagnosis, not as the renderer verdict.

Source data: this session's scratchpad `NONVISUAL-BOUNDARY-DETAIL.md`,
`nvb-host.json`, `nvb-lavapipe.json`, `nvb-logcat-1.txt`.

## M1B-E021 — The comparison floor: spending beats not spending by a wide margin, the scripted heuristic is not shown to beat random at this sample size, and a boundary deadlock cut the run short

**Date:** 2026-09-17
**Status:** The comparison floor (scripted, random, wait arms) named as the
last item of the `M1B-E020` next-slice list is measured, but the run was
stopped early by the Lead on a priority change; 23 valid episodes per arm —
the stated minimum — were collected first. A blocking defect changed the
run's shape mid-flight and is recorded here in full, since a fix is in
flight in a concurrent commit and this entry should read correctly whether
or not it has landed.

Setup: commit `86fcf3c` on the disposable clone `emulator-5556`, offline
verified by interface, `--frame-game-ms 100`, scripted/random/wait arms.
Because of the defect below, the intended single-process interleaved run was
replaced by 23 interleaved segments of one episode per arm (a randomised
complete block design, `--block 1`), each writing its own report, with
per-episode records pooled using `comparison.py`. 4 segments were lost to the
defect.

### Per-arm results

| Arm | Valid | Mean final wave | sd | dec/ep | dec/wave | purchases/ep | Speed-up | Advance share | ep/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scripted | 23/23 | 5.565 | 2.171 | 121.0 | 21.74 | 19.30 | 5.48 | 0.830 | 89.7 |
| random | 23/23 | 5.348 | 1.824 | 117.7 | 22.00 | 18.70 | 5.33 | 0.829 | 92.7 |
| wait | 23/23 | 1.870 | 0.344 | 27.8 | 14.88 | 0.00 | 5.11 | 0.878 | 325.5 |

Final-wave distributions: scripted `[1, 3, 3, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6,
6, 6, 7, 7, 7, 8, 8, 9, 10]`; random `[2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6,
6, 6, 6, 6, 6, 6, 7, 7, 8, 10]`; wait `[1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
2, 2, 2, 2, 2, 2, 2]`.

**Health.** `BRIDGE_EVENT_DIVERGENCE` 0, `stale_or_duplicate` 0,
`advances_cut_short` 0, `recovered_transients` 0, `episodes_not_started_fresh`
0, `invalid_detail` empty, 69/69 episodes valid across all three arms.
End-to-end 57.5 valid episodes/hour over 72 minutes on one actor.

### Power: scripted-versus-random is under-powered, and no conclusion is drawn about it

At n=23/arm the minimum detectable difference (pooled sd, 80% power, alpha
.05) is 1.1–1.7 waves. Intervals are in this session's scratchpad
`POOLED-FLOOR.txt`. Stated plainly: this sample cannot resolve a
scripted-versus-random difference smaller than about 1.1 waves, and none is
claimed.

### Finding

Spending beats not spending by a wide and unambiguous margin: scripted 5.57
and random 5.35 waves versus wait 1.87 waves. The scripted heuristic does
**not** measurably outperform random choice at this sample size (+0.22
waves, well inside the MDE).

**The Lead's reading, recorded explicitly.** This does not establish that
there is no headroom above the baselines — it establishes that our scripted
heuristic is not a strong bar, and the ceiling remains unknown. This is not
recorded as "choice does not matter"; that would be a stronger claim than
the evidence supports.

**Open question.** Whether runs at this account baseline are simply too
short (≈5.5 waves, ≈19 purchases) for upgrade choice to compound is
unresolved. Powering scripted-versus-random properly for a 1-wave difference
needs ≈97 episodes per arm, which becomes affordable once multi-actor
scaling lands.

### Blocking defect: the episode boundary deadlocks when the run dies inside the pause-settle window

Recorded in full because it is the reason the run's shape changed. Two
78-episode single-process runs died at an episode boundary with
`RunPortError: the instance did not reach an active run in time`, at a rate
of about 1 per 7 boundaries.

Cause, confirmed from code, logcat and a live probe: `AdvanceUntilEvent`
sets the paused flag and dispatches `Pause` **before** reading the settled
snapshot, so a tower death inside the pause-settle window emits a terminal
observation while the bridge believes the world is paused; the bridge then
emits only heartbeats (the sequence-hold behaviour of `484c7e6`), the
client's `read_state` returns that stale terminal reading indefinitely, and
`_resume_a_frozen_run` declines to unpause precisely because the reading is
terminal. A fix is in flight in a concurrent commit.

### Cleanup

Verified: `libunity.so` SHA-256 `ffc1f3ef…dd0040`, `versionCode 1199`,
`29.0.3`, installer `com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`; emulator killed, no qemu process, no adb
device, repo tree clean.

Source data: this session's scratchpad `floor/POOLED-FLOOR.txt`,
`floor/seg-0NN.json`, `floor/BOUNDARY-DEADLOCK.md`, `floor/segments.log`,
`floor/cleanup.log`, `floor/run1-logcat.txt`.

## M1B-E020 — The sequence-race fix verified on device: zero rejections, zero stale reads, and the next largest cost identified

**Date:** 2026-09-17
**Status:** The blocking defect left open in `M1B-E019` — the bridge's idle
observation stream racing a sequence-bound command under host latency — is
verified fixed at the commit that carries the fix. An independent review ran
over the same commit and found further defects in the surrounding lifecycle
code; those are recorded as open findings here regardless of whether a
concurrent fix commit has since landed. `begin_episode` timing is recorded as
the next largest wall-clock cost.

Setup: commit `484c7e6` on the disposable clone `emulator-5556`, offline
verified by interface, `frame_game_ms=100`, diagnostics-ON build — like-for-like
with the `M1B-E019` arms; neither is a diagnostics-off figure. The deployed
binary was verified bit-for-bit: a forced rebuild from the clean `484c7e6` tree
reproduced `libtower_bridge.so` with sha256 `f80696a4…`.

**The defect being verified.** Before the fix, the bridge's idle stream bumped
the observation sequence every ~250 ms, so any command carrying an older
`expected_observation_sequence` was rejected `stale_or_duplicate` — 15 of 35
advances under 1 s of injected host latency (`M1B-E019`). This blocked RL
training, since a trained network's forward pass plus learning step routinely
exceeds 250 ms. The fix: while the bridge has paused a still-active run, it
holds the sequence and emits a heartbeat instead of a fresh observation; the
client answers `read_state()` from its cached last state on a heartbeat.

**V1** (1000 ms host sleep between decisions, 35 advances, 2 episodes): **0 of
35** `stale_or_duplicate` (was 15/35). All 35 confirmed — 30 `budget_exhausted`,
3 health, 1 wave, 1 run_ended. Round-clock delta median 2033 ms (p25=p75=2033,
max 2140, min 214 on a short event-terminated advance); the full-budget median
is unchanged from the no-sleep case, so the fix did not alter simulated time
per decision.

**V2** (3000 ms sleep, ~12 idle intervals per decision, 20 advances): **0 of
20** rejections, no read timeout, no heartbeat/liveness failure, no disconnect,
no exception. Full-budget median round delta again 2033 ms.

**V3** (scripted, 4 episodes): 4/4 valid, `invalid_detail` empty,
`BRIDGE_EVENT_DIVERGENCE` 0, `advances_cut_short` 0, `decisions_per_wave`
21.059, mean final wave 8.5 (median 9, range 6–10), `speedup` 4.627,
`total_round_seconds / total_budgeted_game_seconds` = 1133.683/1120.8 = 1.0115,
per-advance wall 287 ms (205.599 s / 716). Against `M1B-E019` f=100 (1.011,
4.600, 283 ms, 3/4 valid where the single invalid episode was this very race):
indistinguishable on fidelity and throughput, with the race-caused invalid
episode gone. Every episode began cleanly; zero lifecycle failures and zero
bridge-side errors across 3,965 logcat lines.

**V4** (staleness check): across 49 paused reads (31 in V1, 18 in V2),
**zero** returned a state whose sequence differed from the one the bridge was
holding, and zero differed in content from the last settled observation; the
held sequence never drifted during a sleep; after each unpause/advance the
sequence strictly advanced and content changed in all 49 cases. Two reads did
not match the pre-read held sequence — both the first decision of an episode,
where the world is genuinely unpaused and streaming; expected and harmless
since the read precedes the command. The client's `BridgeStaleObservationError`
never fired.

**Boundary cost, recorded for the next slice.** Aggregate boundary 39.4 s over
4 episodes (~9.9 s each) at an advance share of 0.839; directly measured
`begin_episode` took 7.277 s and 7.254 s on the RESULT→RETRY path and 0.256 s
when the run was already active. Note `run_episodes.py` does not time
`begin_episode`, which is why these come from a driver.

**Independent review findings, recorded as open.** A fix commit for some of
these may be landing concurrently with this entry; this reads correctly
whether or not it has — the findings are recorded as review output, not as a
current-state claim:

1. The death-boundary transient retry became a guaranteed no-op, because that
   transient implies the run is active, so the world is paused and the re-read
   returns the identical cached state — the episode is then classified
   `OBSERVATION_INVALID` rather than `GAME_OVER`, which would corrupt the
   validity rate the M1 gate uses. The domain correction is that a frozen
   world resolves a death boundary by processing another frame, not by being
   observed again.
2. `world_paused` was re-derived by a second `RunIsActive()` call rather than
   reported by the advance that decided to pause, leaving a window in which an
   auto-restart could mark a running world as paused.
3. The lifecycle pause flag was set from the action name regardless of
   outcome.
4. The test fake diverged from the bridge rule on `buy_upgrade` and on
   run-ended-under-advance.
5. Pre-existing and now cache-fed: an episode ending host-side while the run
   is still active leaves the world frozen, and `begin_episode` then returns
   on the cached active state, silently continuing the old run.

**Verdict recorded.** The review established that stale data CANNOT reach
training replay — mid-episode observations come only from command-bound
readings, never from `read_state`, and the two failure paths that do read
produce inadmissible transitions that replay rejects wholesale.

**Caveats.** V1/V2 are single runs of 35 and 20 advances, demonstrating
absence of the race at these latencies rather than bounding a rare residual;
untested are a host delay approaching the 120 s read timeout and latency
injected at the episode boundary (which is deliberately unpaused and still
streams).

Source data: this session's scratchpad `verify/DETAIL.md`, `v1-sleep1000.json`,
`v2-sleep3000.json`, `v3-f100.json`, `logcat-v3.txt`.

## M1B-E019 — The round-time law, and a falsified prediction: `round_delta ≈ 1.07 · frame_game_ms · (loop_frames − 1)`, the pause is a genuine freeze, and a sequence race blocks training

**Date:** 2026-09-17
**Status:** The `round/game` defect left unresolved in `M1B-E018` is closed for
reporting purposes. The two-term prediction committed after `M1B-E018` is
falsified in its specific form; a different, empirically fit law is recorded
in its place. A separate, decisive finding rules out one candidate mechanism
outright. A new blocking defect is surfaced and left for a concurrent commit.

Four arms/tests at commit `c07ae35`, which had just fixed settle-frame
mis-accounting by zeroing `captureDeltaTime` before `Pause`, on the disposable
clone `emulator-5556`, offline verified by interface before every measurement,
scripted policy throughout.

### E1 — the ratio sweep, 4 episodes per arm

| f (ms) | round/budgeted | speedup | valid | dec/wave | mean wave | ms/advance |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 1.040 | 2.656 | 4/4 | 20.78 | 8.00 | 540 |
| 100 | 1.011 | 4.600 | 3/4 | 15.91 | 7.67 | 283 |
| 250 | 0.932 | 8.309 | 4/4 | 20.72 | 8.00 | 139 |

`advances_cut_short` and `BRIDGE_EVENT_DIVERGENCE` were zero in every arm. The
one invalid episode (100 ms) was `advance was not confirmed:
stale_or_duplicate`, not a divergence. Four episodes per arm is ample for the
ratio, which aggregates hundreds of advances each — it is **not** ample for
any fidelity or wave claim, and none is made from this arm.

**The committed prediction was falsified in its specific form.** The predicted
flat ratio of approximately 1.07 did not appear: the ratio still falls
monotonically (1.040 / 1.011 / 0.932), though the spread collapsed sharply from
the `M1B-E018` values (0.985 / 0.911 / 0.740) that the fix was meant to
address. The two-term model `deficit = 2.68·f − 110 ms` does not survive this
data and is withdrawn. Recording this plainly: the prediction was made, it was
tested, and it failed.

### E2 — the round-clock regression, 1,572 probed advances

Regressing the chained round-clock delta on `loop_frames` (the chaining is
exact: `t0(N) = t2(N−1)`):

| f (ms) | fit | R² | n | slope/f | intercept/f |
| --- | --- | --- | --- | --- | --- |
| 50 | `53.50·lf − 52.7` | 0.9997 | 569 | 1.070 | −1.053 |
| 100 | `106.89·lf − 103.4` | 0.9987 | 404 | 1.069 | −1.034 |
| 250 | `269.47·lf − 279.5` | 0.9773 | 589 | 1.078 | −1.118 |

The law is **`round_delta ≈ 1.07 · frame_game_ms · (loop_frames − 1)`**: the
game credits about 7% more simulated time per frame than `captureDeltaTime`
requests, and about one frame per advance is never credited at all. Cross-check
against E1: `1.07·(lf−1)/lf` with `lf` = 40/20/8 predicts 1.043/1.016/0.936
against the measured 1.040/1.011/0.932. The earlier "constant 110 ms" term was
in fact `1.07·f` evaluated at the single frame size it was fit against, not a
constant.

Also recorded: `t2 − t1` medians of 53/107/267 ms ≈ 1.068·f — the last loop
frame's credit, which the mid-loop `t1` read precedes. The residual
`(t1 − t0) − f·lf` has medians of +33 / −74 / −395 ms across 50/100/250 ms:
frame-size-proportional, not constant.

### E3 — the decisive test: does the round clock credit paused wall time?

35 advances per condition at `frame_game_ms = 100`, through the normal
adapter/port path. Bridge-reported `round_ms` on full-budget advances: no
sleep, n=30, median 2033 ms; with a deliberate 1000 ms host-side sleep inserted
between decisions, n=14, median 2033 ms (mean 2041, max 2140 — one extra
frame). The chained paused gap `t0(N) − t2(N−1)` was 0.0 ms for all 19 chained
advances under the sleep.

**Conclusion: the game's round clock does not credit paused wall time.** It is
a pure per-frame simulated-time accumulator, and `Pause` is a genuine freeze,
not merely a rendering stop.

### Lead's decisions recorded here

1. The round clock stays the authoritative witness for reported speedup
   (`total_round_seconds / total_wall_seconds`), already in effect since
   `c07ae35`.
2. The advance loop's budget condition stays on `frames × frame_game_ms` and is
   **not** switched to the round clock. The budget is a bound on quiet game
   time, not a measurement — advances are stopped by events rather than by the
   budget, and `M1B-E018` already showed real time per decision flat at about
   1550 ms across all frame sizes. A 7% systematic offset in a bound moves no
   decision moment, and coupling the loop to a game-internal float would add
   complexity for no behavioural gain.

### UNRESOLVED — two mechanisms, neither chased

- **The 1.07 factor is unexplained.** Untested candidates: a permanent
  account-level game-speed modifier (e.g. a lab bonus), a hidden multiplier, or
  a `deltaTime` clamp. Note that the adapter pins the game's own multiplier at
  1.0 and reports it, so a reported-1.0-but-effective-1.07 would indicate a
  separate modifier from the one the adapter controls.
- **The one uncredited frame per advance is unexplained.** Candidate: the
  first frame after `Unpause` does not apply `captureDeltaTime`.

### Blocking finding: a sequence race in the bridge's idle observation stream

15 of 35 advances in the E3 sleep condition were rejected
`stale_or_duplicate` even after a fresh read. The bridge's roughly 250 ms idle
observation stream races any sequence-bound command once host latency
approaches it. This blocks RL training directly: a trained network's forward
pass plus learning step routinely exceeds 250 ms. It surfaces as
`ACTION_PIPELINE_FAILED` — lost episodes, not silent corruption. A fix is in
flight in a concurrent commit.

### Caveats

One host, one build, 4 episodes per E1 arm. The control logcat
(`lc-e3b-ctrl.txt`) also contains the start of the sleep run — its 9.2 s gap
outlier is that boundary — while the sleep-condition logcat
(`lc-e3b-sleep.txt`) is clean.

Source data: `FINDINGS.md`, `E2-statistics.txt`, `e1-f{50,100,250}.json`,
`e3b-{ctrl,sleep}.json`, `logcat-e1.txt`, `lc-e3b-{ctrl,sleep}.txt`, session
scratchpad.

## M1B-E018 — The `frame_game_ms` sweep: decision density is flat, 100 ms is the standing decision, and the round-time witness has its own defect

**Date:** 2026-09-17
**Status:** The sweep this project has been waiting on since `M1B-E017`. Frame
size does not move decision density across the range tested. `250 ms` is
rejected on a detected dynamics difference. `100 ms` is adopted. A second,
unrelated defect in the game-time witness is exposed and left open.

Five arms at commit `38fb276` on the disposable clone `emulator-5556`, offline
verified by interface before every measurement, scripted policy, 8 episodes per
arm, all `run_episodes.py` defaults except `--frame-game-ms`. The arms ran
**sequentially**, in the order 100, 16.7, 250, 50, 100b — **not interleaved**.
That leaves in-game progression drift as an uncontrolled confound across the
run, partly bounded by the repeated 100 ms arm (`100` and `100b`) taken first
and last.

### The 89.3 decisions/episode figure is not the baseline — correcting an expectation this project has been carrying since `M1B-E014`/`M1B-E017`

`M1B-E014`'s 89.3 decisions/episode at 1x, and the requirement in
`docs/workstation-handoff.md` that asked for a match to it, were both measured
through the **old** wall-clock-sleep code path. They are not comparable to
anything measured through the current advance-loop-in-the-bridge path
(`M1B-E016`/`M1B-E017`). Through the current path, every arm in this sweep —
16.7 ms through 250 ms — lands at **124 to 167 decisions/episode**, well above
89.3 regardless of frame size. Quoting 89.3 as a target for the current path
would be an error; the correct reference is the 16.7 ms arm of this sweep,
which is the finest frame tested and the closest surrogate for uncapped
per-frame decisions through the current path. This corrects the expectation
stated in `M1B-E017`'s unresolved section and in `docs/workstation-handoff.md`.

### Per-arm results

| Arm | Valid | dec/ep | dec/wave | Mean final wave (sd) | Speed-up | round/game | Advance share | ms/advance | ep/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.7 ms | 8/8 | 123.8 | 21.06 | 5.88 (2.48) | 0.97 | 1.040 | 0.985 | 1513 | 17.9 |
| 50 ms | 8/8 | 145.3 | 21.13 | 6.88 (2.03) | 2.70 | 0.985 | 0.906 | 533 | 37.3 |
| 100 ms (first) | 7/8 | 103.6 | 18.13 | 5.71 (3.20) | 5.01 | 0.911 | 0.831 | 277 | 67.0 |
| 100 ms (repeat, `100b`) | 8/8 | 129.4 | 21.12 | 6.13 (3.14) | 5.05 | 0.911 | 0.834 | 280 | 65.9 |
| 250 ms | 8/8 | 167.4 | 20.92 | 8.00 (1.51) | 11.62 | 0.740 | 0.722 | 134 | 85.2 |

`BRIDGE_EVENT_DIVERGENCE` and invalid state transitions were zero in every arm,
across all 40 episodes. `advances_cut_short` was zero everywhere. The only
invalid episode in the whole sweep was one `action_pipeline_failed` (`"advance
was not confirmed: stale_or_duplicate"`) in the first 100 ms arm.

### Bootstrap intervals against the 16.7 ms reference

95% intervals, 10,000 resamples, the project's own `comparison.py`:

| Arm | Final wave, diff [95% CI], Cohen's d | dec/wave, diff [95% CI], Cohen's d |
| --- | --- | --- |
| 50 ms | +1.00 [-1.12, +3.12], d=+0.44 | -0.25 [-1.13, +0.69], d=-0.24 |
| 100 ms (`100b`) | +0.25 [-2.38, +2.88], d=+0.09 | +0.36 [-1.17, +2.00], d=+0.21 |
| 250 ms | +2.12 [+0.25, +4.00], d=+1.04 | -0.56 [-1.37, +0.33], d=-0.60 |

Only the 250 ms final-wave interval excludes zero.

### Power: this sample could not have detected a one-wave fidelity loss at any frame size

Reference standard deviations (16.7 ms arm): 2.47 waves, 1.23 decisions/wave.
`required_episodes` for 80% power: **97 per arm** to detect a 1-wave difference,
25 for a 2-wave difference, 11 for a 3-wave difference; 24 per arm for a 1
decision/wave difference, 6 for a 2 decision/wave difference. Eight episodes per
arm is far short of 97. This sample could **not** have detected a one-wave
fidelity loss at any of the frame sizes tested — the absence of a detected
difference at 50 ms and 100 ms is not evidence of equivalence at that
resolution, only an absence of evidence at this sample size.

### Findings

**(a) Decision density is flat.** dec/wave sits at 20.9–21.1 across 16.7, 50,
100b, and 250 ms — a range of 0.2 decisions/wave. Frame size does not move the
decision density the requirement is about, over the range tested.

**(b) `M1B-E017`'s concern does not reproduce and is withdrawn.** That entry's
100 ms sample (mean wave 6.2, 15.9 dec/wave, 5 episodes) is not seen again at
100 ms here — 5.71–6.13 mean wave, 18.1–21.1 dec/wave over 15 episodes across
two 100 ms arms. `M1B-E017`'s unresolved section is corrected in place with a
pointer to this entry rather than by editing its recorded numbers.

**(c) 250 ms is rejected.** Its final-wave difference against the 16.7 ms
reference is +2.12 waves, and the 95% interval excludes zero — the only arm
where that happens. The direction is favourable (episodes run longer at 250 ms,
not shorter), but a detected difference in either direction is still a
detected dynamics difference from the reference. **250 ms is judged not
faithful to the reference and is rejected**, regardless of its direction.

**(d) DECISION: the benchmark runs at `frame_game_ms = 100`.** Grounds: no
detected difference from the 16.7 ms reference at 100 ms; 5 physics steps per
frame against the measured 16-step (`Time.fixedDeltaTime` 20 ms into
`Time.maximumDeltaTime` 333.3 ms) clamp, leaving headroom `M1B-E017` already
established; a smaller game-time accounting error than 250 ms (see below); and
only +29% throughput available from going to 250 ms, because the
episode-boundary cost already dominates above 100 ms — advance share falls from
0.985 at 16.7 ms to 0.722 at 250 ms, i.e. the fixed per-episode boundary, not
the frame, is what limits throughput past 100 ms.

### UNRESOLVED — the round-time witness does not hold at ≈1, and the cause is not yet known

`round/game` — the round clock's own witness of game-time fidelity introduced in
`M1B-E017` — does **not** hold at approximately 1 across this sweep. It reads
1.040 / 0.985 / 0.911 / 0.740 at 16.7 / 50 / 100 / 250 ms: monotone in frame
size and reproducible (the two 100 ms arms agree, 0.911 both times). Budgeted
game time (frames × `frame_game_ms`) systematically exceeds the game's own
round clock as frames coarsen, so the reported `speedup` figure overstates real
game progress — at 250 ms, `speedup` claims 11.62 against 8.59 read from the
round clock itself.

The leading hypothesis under investigation is that the two tail frames of each
advance are counted at full `frame_game_ms` weight after `Pause`, which is
consistent with the shortfall growing at 250 and 100 ms but is not by itself
consistent with the 1.040 excess (game clock running slightly *ahead* of
budget) seen at 16.7 ms, implying a second, opposite-signed effect around
unpause. A specialist analysis of this discrepancy is in flight. This entry
does not present a conclusion on the cause — only that the decision in (d)
above does not depend on resolving it, since 100 ms sits between the two
extremes and was chosen on grounds independent of this defect.

Source data: `sweep-analysis.txt` and `sweep-frame{16.7,50,100,100b,250}.json`
with per-episode sidecars, session scratchpad.

Since this entry, the settle-window frames after `Pause` no longer count toward
`game_ms` and run at real-time pacing, the evaluator's frame-arithmetic total is
named `total_budgeted_game_seconds`, and `speedup` is measured on the round clock
(`total_round_seconds / total_wall_seconds`) rather than on that budget. The
`round/game` ratio quoted above is the same quantity as today's
`total_round_seconds / total_budgeted_game_seconds`; the `speedup` figures quoted
above are the old, budget-based definition.

## M1B-E017 — The bridge-side advance loop runs on the real game at 5x, and the game-time witness was wrong

**Date:** 2026-09-17
**Status:** The mechanism works on the device. One reported number was measuring
the wrong thing and is corrected here. No speed is yet established as admissible.

Five scripted episodes at commit `611667d` on the clone, `--frame-game-ms 100`,
defaults otherwise, offline verified by interface before every measurement. The
one command per decision design of `M1B-E016`'s consequence is now the thing that
actually ran.

| Quantity | Value |
| --- | --- |
| Valid episodes | 5 of 5 (`invalid_detail` empty) |
| Speed-up | 5.013 |
| Frames / advance wall seconds | 8,153 / 135.0 = 60.4 fps |
| Frames per decision | 16.6 |
| Wall milliseconds per advance | 274.4 |
| `advances_cut_short` | 0 |
| `BRIDGE_EVENT_DIVERGENCE` | 0 |
| Decisions per episode | 98.4 |
| Decisions per wave | 15.871 |
| Mean final wave | 6.2 (median 6, range 1 to 10) |
| Episodes per hour | 80.6 |

The per-advance cost is accounted for entirely by frames: 16.6 frames at 16.6 ms
each. The 500 ms pause-settle window is not timing out; it costs about two frames.
Zero `advances_cut_short` means no advance hit the bridge's wall ceiling, and zero
`BRIDGE_EVENT_DIVERGENCE` means the bridge's stopping conditions and the host's
predicate agreed on every one of the 492 decisions.

### The engine ceiling on `frame_game_ms` is measured, not assumed

The one-time diagnostics line read `maximum_delta=0.333333 fixed_delta=0.020000`.
Unity clamps how much game time one frame may advance at `Time.maximumDeltaTime`,
so **333.33 ms is the hard ceiling on `frame_game_ms`**, whatever the protocol
bound says. `Time.fixedDeltaTime` is 20 ms. The 100 ms used here is well under.

### `playTime` was the wrong witness — a bad metric, not a bad mechanism

`total_play_seconds / total_game_seconds` came out 0.168, not the 1.0 the design
requires. That is not evidence against `captureDeltaTime`: `Main.playTime` is the
account-lifetime clock and advances at wall rate regardless of the game clock, so
`play_ms` was measuring wall time. It tracked `total_advance_wall_seconds` to
1.7 percent, and 0.168 is simply one over the measured speed-up of 5.013. The
ratio was arithmetically incapable of saying anything.

Independent, game-owned evidence that `captureDeltaTime` **is** applying: the
game's per-round clock advanced at a median **4.877 game-seconds per wall-second**
(mean 4.451, 595 consecutive sample pairs from the diagnostics log), which agrees
with the measured speed-up of 5.013. `roundTime` tracks it identically.

Corrected in this commit: the command result reports `round_ms` from
`Main.gameplayTimeThisRound` in place of `play_ms`, and the evaluator reports
`total_round_seconds`, whose ratio to `total_game_seconds` must be about 1. A
related earlier reading is also overturned: the note that `roundTime`,
`gameplayTimeThisRound`, and `realTimeThisRound` "all read 0.0 throughout a run"
came from reading `float` fields as `double`. Read as singles they advance.

### The heartbeat defect: every unattended run died at about sixty seconds

`tower_bridge.cpp` accumulates `heartbeat_elapsed` only on the *idle* branch of
the stream loop, and an advance emits its own heartbeat only when it exceeds one
second. With one command permanently in flight at ~275 ms per advance the idle
branch is never reached, so no heartbeat is ever sent and the host's 60-second
check tripped during healthy play — masking the real state behind
`BridgeTimeoutError` on release. The measurement above was taken with a
host-side workaround, not a repository change.

Fixed on the host, where the domain sits: a heartbeat exists to prove the bridge
is alive, and an observation or a command result is strictly stronger proof, so
**any successfully decoded inbound frame renews liveness**. The bridge keeps its
in-advance heartbeat for genuinely quiet long advances.

### UNRESOLVED — fidelity at 100 ms per frame is not decided

Mean final wave was **6.2** against the 9.79 reference of `M1B-E008`, and
decisions per wave **15.9** against roughly 9.1 (89.3 decisions per episode at 1x
in `M1B-E014` over that 9.79). Both gaps are consistent with **either** fidelity
degrading at 100 ms per frame **or** ordinary variance over five episodes — the
standard deviation of final wave here is 3.35, and `M1B-E008` needed about 23
episodes to resolve a one-wave difference. This entry does not decide it. **The
pending `frame_game_ms` sweep decides it, and until it does no speed has been
validated as admissible.**

**Correction (`M1B-E018`):** the sweep this section calls for has since run.
This entry's concern does not reproduce at 100 ms (mean wave 5.71–6.13 over 15
episodes across two arms, not 6.2 over 5). More importantly, the 89.3
decisions/episode figure quoted above as a reference is from the **old**,
pre-`M1B-E016` code path and is not comparable to anything measured through the
current advance-loop-in-the-bridge path — every arm of the `M1B-E018` sweep
lands at 124–167 decisions/episode regardless of frame size. Do not read this
section as still asking for a match to 89.3. See `M1B-E018` for the sweep, the
decision (`frame_game_ms = 100`), and what remains open.

Caveat on the sample: `begin_episode` adopts any non-terminal run, and episode 1
adopted the partial run left by an aborted first attempt, so its decision count is
understated and the minimum final wave of 1 is most likely that episode.

## M1B-E016 — The frame-exact step works, and the bottleneck moves to the round trip

**Date:** 2026-09-17
**Status:** The mechanism works and the game's speed multiplier is now irrelevant
to it, which is what the requirement asked for. Throughput is lower than the
current arrangement, for a reason the measurement identifies precisely.

`Time.captureDeltaTime` makes one rendered frame worth a fixed amount of game
time however long it took to render. The bridge's step now sets the slice, reads
`Time.frameCount`, unpauses, polls until the counter advances by one, pauses, and
restores real-time pacing. The wall-clock sleep remains only as a fallback for
when the engine clock cannot be resolved.

Measured in a live run, 250 ms slices:

| Game speed multiplier | Wall milliseconds per step |
| --- | --- |
| 1 | 59, 66, 74 |
| 16 | 54, 54, 59 |

Every step returned `confirmed/frame_step`.

### What this establishes

**The speed multiplier no longer affects the step.** A sixteen-fold change in the
game's own clock setting moves the cost of a 250 ms step by nothing
distinguishable from noise. Under the old wall-clock sleep the same change moved
game time per step by a factor of sixteen. Decision moments are now exact by
construction rather than by measurement, which is the requirement.

That also means the multiplier stops being a tuning knob. Under frame stepping it
should sit at 1x permanently, and speed comes from elsewhere.

### And where speed now comes from — not the frame

250 ms of game time for about 57 ms of wall clock is **4.4x**. That is lower than
the 8x currently in use, and the reason is visible in the arithmetic: at 58.9 fps
(`M1B-E015`) a frame takes about 17 ms, so roughly 40 ms of each step is host
round trip plus pause and unpause. **The step is round-trip-bound, not
frame-bound.**

This corrects the ceiling estimate in `M1B-E015`. `speed = slice x fps` assumed
frames were the only cost and gave 14.7x. With one host round trip per frame the
real figure is `slice / (frame_time + round_trip)`, which is 4.4x. Raising the
frame rate by uncapping vSync would move 17 ms toward zero and leave the 40 ms
untouched, so it cannot by itself get past roughly 6x.

### The consequence for the design

The remaining cost is one round trip per *slice*, while the agent only needs one
decision per *event*. The environment currently loops, advancing slice after
slice until something actionable changes — roughly eight slices per decision at
the configured backstop — and every one of those slices is a separate command.

Pushing that loop into the bridge is the fix: one command that advances frames
until an event or until a game-time budget expires, then returns the observation.
That is one round trip per decision rather than per slice, and it would put the
frame back in charge of the cost, where the rendering rate and therefore the
renderer choice start to matter again.

Until that exists, 8x with the old free-running path remains the faster option at
56.7 episodes per hour, and `M1B-E014` establishes that its decision moments match
normal-speed play. Frame stepping is correct and slower; free running at 8x is
fast and correct only because 8x happens to sit below the frame limit.

## M1B-E015 — Main exists at the home screen, engine icalls are safe, and the frame rate is 59

**Date:** 2026-09-17
**Status:** Three results from one instrumented deploy. One of them overturns
`M1B-E013`'s explanation of the boundary tap.

### 1. `Main.Instance` is alive at the home screen

The premise carried since `M1B-E004` — that `Main` exists only inside the battle
scene, which is why `UnitySendMessage` has nothing to deliver to from home — is
**wrong**. Logged at a positively classified `battle_home_tier_1` screen:

```text
liveness no_run managed=0x764b4fd3d000 cached_ptr=0x764ba1905790 field=found
```

The managed reference is non-null *and* the native handle is non-zero, so this is
a live component, not Unity's fake null. `Main` is a singleton that persists
across scenes. The receiver exists, and the receiver hunt was aimed at a problem
that does not exist.

`M1B-E013`'s measurement stands — `enable_auto_restart` and `retry` do time out —
but its explanation does not. The cause is something else: the method may not be
on the component attached to the object named `Main`, the object may be inactive
(`UnitySendMessage` uses `GameObject.Find` semantics and cannot see inactive
objects), the call may have unmet preconditions, or the transition may exceed the
30-second lifecycle wait. That is the question to ask next, and it is a much
cheaper question than enumerating the scene.

A note on reading the log correctly: this line is emitted on the `run_unavailable`
path, which at the home screen is reached through the *scalar validation* return,
not the liveness return. `Main` being alive while its wave and health scalars do
not describe a run is exactly right between episodes.

### 2. A direct engine icall from the socket thread is safe

`UnityEngine.Time::get_frameCount()` was resolved through `il2cpp_resolve_icall`,
attributed to `libunity.so` with `dladdr` before being called, and then invoked
roughly four times a second for twenty seconds. No crash; the game process
survived and kept rendering. This is the first direct engine call the bridge has
made from its own thread, and it supports the narrowed rule: engine leaf
accessors are a different category from managed game code.

Every binding the frame-exact step needs resolves, all of them in `libunity.so`:

| Binding | Resolved |
| --- | --- |
| `Time::get_frameCount()` | yes |
| `Time::get_captureDeltaTime()` | yes |
| `Time::set_captureDeltaTime(System.Single)` | yes |
| `Time::get_timeScale()` | yes |
| `Time::get_fixedDeltaTime()` | yes |
| `Application::set_targetFrameRate(System.Int32)` | yes |
| `QualitySettings::set_vSyncCount(System.Int32)` | yes |
| `Object::GetName(UnityEngine.Object)` | **no** (null) |

`il2cpp_stop_gc_world` and `il2cpp_gc_foreach_heap` are both present, so the
stop-the-world heap walk remains available as a fallback. `Object::GetName` not
resolving under that signature removes the cheap route to a GameObject's name —
which no longer matters for the tap, given result 1.

### 3. The frame rate, which is the ceiling on the whole scheme

`frames=1703` at 08:57:11.699 and `frames=2842` at 08:57:31.028: 1,139 frames in
19.33 seconds, **58.9 frames per second**, at the home screen under lavapipe with
`-no-window`.

That fixes the ceiling. Under the frame-exact scheme, speed is `slice x achieved
fps`, so a 250 ms slice at 59 fps is about **14.7x** — with decision moments exact
by construction, against 8x today with decision moments merely *equal* to normal
play. Whether uncapping `vSyncCount` and `targetFrameRate` lifts the rate above
the 60 Hz that is plainly capping it now is the next measurement, and it is what
decides whether this reaches well past 14.7x or settles there.

The measurement was taken at the home screen, where nothing is being simulated. A
busy late wave will render slower, so 14.7x is an upper bound rather than a
promise.

### Incidental: `deploy` cold-launches the game and therefore needs the network

`instrumented_bridge.sh deploy` force-stops and relaunches the package. Offline,
that lands on the Firebase OFFLINE modal (`M1B-E010`) and the game never reaches
home, so the bridge reports `run_unavailable` from a splash screen and a client
that expects a run gets a closed stream. An earlier reading in this session was
taken in exactly that state and briefly looked like evidence that `Main` was
absent. It was evidence about the splash screen. Deploy needs the same
start-online-then-cut treatment `clone_session.py start` performs.

## M1B-E014 — 8x does preserve decision moments; the reference was the thing missing

**Date:** 2026-09-17
**Status:** The requirement is met at 8x and broken at 64x, measured against a
real 1x reference for the first time. This reinstates, on different evidence, the
claim `M1B-E012` withdrew.

`M1B-E012` compared 8x against `M1B-E006`'s table and found 13.4 decisions per
wave where that table said 69, and I withdrew the claim that 8x preserves
decision moments. That withdrawal compared 8x against a number that does not
reproduce, rather than against normal-speed play. Three scripted episodes at 1x
supply the comparator that was missing.

| Speed | Decisions per episode | Decisions per wave | Mean final wave | Wall seconds per episode | Episodes per hour |
| --- | --- | --- | --- | --- | --- |
| 1 | 89.3 | 12.2 | 7.33 | 235 | 15.3 |
| 8 | 88.5 | 13.4 | 6.62 | 63.5 | 56.7 |
| 64 | 42.5 | 4.8 | 8.88 | 17.6 | 205.1 |

Decisions per episode is the cleaner statistic, because decisions per wave
divides by an outcome that varies. On that measure 1x and 8x are
indistinguishable — 89.3 against 88.5 — and 64x delivers less than half.

### What is and is not now established

Established: **the requirement holds at 8x.** The agent gets the same number of
decisions per episode at 8x as at normal speed, which is what "the same decision
moments, only sooner" asks for. It breaks somewhere between 8x and 64x, exactly
where a frame becomes worth more game time than the 250 ms slice.

Not established: the absolute density. Neither 1x nor 8x reaches `M1B-E006`'s 63
to 79 decisions per wave; both sit near 12 to 13. That table remains
unreproducible and must not be quoted. What matters for the requirement is the
*ratio between speeds*, not the absolute value, and the ratio is now measured
against a comparator taken with the same code on the same day.

Sample sizes are three episodes at 1x and eight at each of the others. That is
thin for final wave and adequate for decisions per episode, which is close to a
deterministic property of the cadence rather than an outcome: the 1x-to-8x gap is
under one percent and the 8x-to-64x gap is more than twofold.

### Consequence: training can proceed at 8x now

The interim position in `solution.md` 9.2c assumed satisfying the requirement
meant 1x and roughly seven episodes per hour, which is not a training rate. The
real figure is 8x at 56.7 episodes per hour with decision moments preserved. That
is 3.6 times slower than 64x and entirely usable, so training runs no longer have
to wait for the frame-exact step — they have to run at 8x.

The frame-exact step is still worth building, because it would remove the trade
altogether rather than settling at a point on it, and because `speed = slice x
achieved fps` may reach well beyond 8x. But it is no longer blocking.

## M1B-E013 — Auto-restart is not progression-gated, and the tap still needs a receiver

**Date:** 2026-09-17
**Status:** Negative. Both cheap escapes from the boundary tap are ruled out. The
receiver hunt is required, and it needs a bridge rebuild.

`M1B-E004` found that `StartNewRoundFunction`, `AutoRetryBattle` and
`Button_ToggleAutoRestartBattle` each expired their lifecycle wait when
dispatched from a terminal run, and judged auto-restart "progression-gated at
this baseline". The clone has since drifted from Highest Wave 2 to 11 and from 53
coins to 909 (`M1B-E005`), so the gate was worth re-testing.

Dispatched from a genuine terminal run at wave 11:

```text
enable_auto_restart -> ambiguous / lifecycle_timeout
retry               -> ambiguous / lifecycle_timeout
```

Both still fail. Progression is not the explanation, and the `M1B-E004`
suggestion that auto-restart is gated should be treated as unsupported rather
than merely untested.

**The explanation offered here was also wrong.** This entry proposed that `Main`
does not exist outside the battle scene, so `UnitySendMessage` had no target.
`M1B-E015` measured `Main.Instance` at a positively classified home screen and
found it alive, with a non-zero native handle. The receiver exists. The timeouts
have some other cause — the method may not be on the component attached to that
object, the object may be inactive and therefore invisible to `GameObject.Find`
semantics, preconditions may be unmet, or the transition may exceed the 30-second
wait.

### The free diagnostic is not available

Unity logs `SendMessage: object <name> not found!` when the target is missing,
which would have settled whether `Main` exists at the terminal state and would
have made probing candidate object names free. This build does not emit it: the
`Unity` log tag carries startup output and nothing after, as expected from a
release build with logging stripped. Object existence therefore cannot be probed
from the host, and the enumeration has to happen inside the bridge.

### Consequence

Finding a main-thread receiver is now on the critical path for two separate
things at once — the boundary tap, and setting `Time.captureDeltaTime` for the
frame-exact step (`solution.md` 9.2c). Both need the same thing: a GameObject
that exists outside the battle scene and can be addressed by name. That makes the
next slice a bridge change rather than a host change, and the build environment
is present (NDK 29.0.14206865, build directory retained).

### Incidental: the offline cut is doing real work

While the device sat idle and offline, `PlayCommon` attempted a log upload to
`play.googleapis.com` and failed to connect. Google Play services actively tries
to reach the network on this image, so cutting the radios is not a formality —
it is blocking traffic that would otherwise leave the device.

## M1B-E012 — Decision density measured fresh, and the 8x claim withdrawn

**Date:** 2026-09-17
**Status:** Contrary evidence against my own claim one hour earlier. Lowering the
speed improves decision density but does not restore it, and the `M1B-E006`
density table does not reproduce under current code.

Eight scripted episodes at each speed, same code, same session, same account.

| Speed | Mean final wave | Decisions per episode | Decisions per wave | Wall seconds per episode | Episodes per hour |
| --- | --- | --- | --- | --- | --- |
| 8 | 6.62 | 88.5 | 13.4 | 63.5 | 56.7 |
| 64 | 8.88 | 42.5 | 4.8 | 17.6 | 205.1 |

### The claim being withdrawn

On the strength of `M1B-E006`'s table — 63, 79 and 69 decisions per wave at 1.5x,
4x and 8x, collapsing to 39, 20 and 16 above — I claimed that about 8x preserves
decision moments and could be adopted as the training speed today. That is not
supported.

Freshly measured, 8x yields 13.4 decisions per wave, not 69. It is 2.8 times
better than 64x, so the direction is right and the mechanism is real, but it is
nowhere near the 1.5x reference. Choosing a slower speed does not meet the
requirement; it only makes the violation smaller while costing 3.6 times the
throughput.

The `M1B-E006` numbers were taken under a different code state and should not be
used as a comparator until they are reproduced. Any document quoting 69 decisions
per wave at 8x as a current property is quoting a superseded number.

### What the sample cannot say

Eight episodes per arm. The final-wave means, 6.62 against 8.88, are not a
finding: at a standard deviation near 1.3 this needs 23 episodes per arm to
detect a one-wave difference and 66 at the spread seen during training. The
direction is also opposite to the naive expectation that more decisions produce
better play, which is one more reason not to read it.

Decisions per wave is a different matter. It is a near-deterministic property of
the cadence and the speed rather than a noisy outcome, and 13.4 against 4.8 is
far outside anything eight episodes could produce by chance.

### Why picking a speed was the wrong shape of answer

Game time per frame is `frame_wall_seconds x speed`, so asking the game to run
its own clock faster necessarily makes each frame worth more game time, and the
agent's decisions coarser. Speed and decision density are traded against each
other by construction. The fix is not to find the best point on that trade; it is
to remove the trade, by decoupling game time per frame from wall time per frame
so that speed comes from rendering frames faster rather than from advancing more
game time per frame. `solution.md` 9.2c records the mechanism.

## M1B-E011 — The learning pipeline runs end to end on the real game

**Date:** 2026-09-17
**Status:** Plumbing proven. Not evidence of learning, and far too small a sample
to be. Two real defects found in the first twelve minutes of running it.

The first end-to-end training run on the instrumented clone: both backbones,
interleaved in decision blocks, 800 decisions each at speed 64.

| Quantity | recurrent-q | stacked-dqn |
| --- | --- | --- |
| Decisions | 800 | 810 |
| Episodes | 29 | 25 |
| Valid episodes | 29 | 25 |
| Optimisation steps | 306 | 308 |
| Sequences accepted | 39 | 46 |
| Replay rejections | 0 | 0 |
| Mean recent loss | 0.107 | 0.064 |
| Mean final wave | 6.72 | 7.44 |
| Range | 4 to 11 | 3 to 11 |
| Wall seconds | 353 | 333 |

Whole session: 1,610 decisions, 54 episodes, 54 valid, 614 optimisation steps,
716.7 s. Checkpoints for both arms load cleanly with their checksum sidecars and
carry the identity that produced them — profile
`tower-play-29.0.3-rooted-readonly-v1`, source revision `2a14890`, epsilon
0.050, beta 1.00 at the end of budget.

### What this establishes, and what it does not

It establishes that the pipeline works: the actor collects from the real game,
sequences reach replay and none are rejected, the learner takes gradient steps,
priorities feed back, both backbones run under one budget interleaved on one
device, and checkpoints round-trip.

It establishes nothing about learning. Epsilon anneals from 1.0 to 0.05 across
the budget, so most of these episodes were mostly random. The comparison between
the two arms is what it looks like:

```text
recurrent-q 6.72 vs stacked-dqn 7.44: difference -0.72 [-1.68, +0.24] d=-0.39
n=29/25 — indistinguishable
```

The interval spans zero. At the spread actually observed here, sd 2.04, detecting
a one-wave difference needs **66 episodes per arm**, not 25. Anyone reading 7.44
against 6.72 as "the stacked agent is better" would be reading noise.

For orientation rather than comparison: the scripted policy reaches mean 9.758
(`M1B-E009`) and always-wait dies at wave 2. Mostly-random play reaching 6 to 7
says the action space is forgiving, not that anything was learned.

### It corroborates the decision-density problem from the live loop

| | decisions per episode | decisions per wave |
| --- | --- | --- |
| recurrent-q at 64x | 27.6 | 4.1 |
| stacked-dqn at 64x | 32.4 | 4.4 |
| scripted at 1.5x (`M1B-E006`) | 528 | 63 |

The agents get about four decisions per wave. This is the same effect
`M1B-E006` measured and `solution.md` 9.2c now specifies against: the world runs
away from the policy while it decides, because game time keeps passing during
host latency and is multiplied by the speed. It is not a training bug; it is the
environment handing the agent a much coarser control problem than a normal-speed
player gets.

### Two defects, both found by running something short

- **Checkpoint fingerprinting crashed on the first checkpoint after the first
  gradient step.** `fingerprint` assumed every mapping key was a string; a
  *stepped* optimizer keys its state by integer parameter index. The existing
  tests fingerprint a fresh optimizer, whose state is empty, so nothing caught
  it. Fixed in `2a14890` with a test that steps a real optimizer first.
- **A run was killed by the harness's low-memory watchdog** with 92 GB actually
  available — `free` was low only because 103 GB sat in reclaimable cache. A
  false positive, but the rerun used a replay capacity matched to the run rather
  than the default 4096, which is worth doing anyway: replay holds features as
  Python tuples, and 4096 sequences per arm is on the order of a gigabyte each.

Both were found within twelve minutes of running the pipeline for real, and
neither would have been found sooner by a longer run.

## M1B-E010 — The clone was never offline, and the game will not start without a network

**Date:** 2026-09-17
**Status:** Contrary evidence. Every instrumented run to date, `M1B-E009`
included, executed with a working network connection. A start-online-then-cut
procedure now satisfies the constraint and is verified.

### What was assumed

The operating constraint is that the disposable clone is offline before any
automation. The check used for it was `settings get global airplane_mode_on`
returning `1`, and it did return `1` throughout.

### What is actually true

Airplane mode reads `1` while the wifi radio stays up. On this emulator, before
any change:

```text
airplane_mode_on=1
airplane_mode_radios=cell,bluetooth,uwb,wifi,wimax
wifi_on=2
wlan0    inet 10.0.2.16/24
ping 8.8.8.8 -> 1 packets transmitted, 1 received, rtt 953 ms
```

The setting was written without the broadcast the wifi service acts on, so the
radio never went down. The interface had an address, a route and reachability.
`airplane_mode_on` was therefore never evidence of anything, and the clone has
been online for every instrumented run recorded in this document.

`svc wifi disable` and `svc data disable` do take it down: `wlan0` loses its
address and `ping` returns `Network is unreachable`.

### And then the game would not start

With the device genuinely offline the game stops at its splash screen on a modal
reading *"OFFLINE — You are offline, please check your internet connection and
try again"*, over a progress bar labelled *"Checking Firebase Online Status…"*.
It never reaches the battle home screen, and the adapter correctly refused to
tap: every one of the seven `battle_home_tier_1` anchors disagreed, the screen
classified as `unknown`, and the gate failed closed exactly as designed.

`solution.md` 8.1 already recorded that "offline cold launch after a force-stop
is still unsupported", and the handoff already said to enable airplane mode only
after the game is running. Both were right about the game. What neither caught is
that the mechanism they relied on does not work: airplane mode does not take this
emulator offline, so "enable airplane mode after the game is running" left the
device connected for the whole run rather than for its first twenty seconds.

That is also why the contradiction survived unnoticed. The clone cannot produce a
single valid episode while genuinely offline, so the runs that produced thousands
of them were necessarily online throughout. There was no configuration in which
both the assumption and the results could hold.

### The procedure that satisfies the constraint

The network is needed to *start* the game, not to play it. Verified on this
device:

1. Enable the radio, launch the app, wait for `battle_home_tier_1` — 21 seconds.
2. `svc wifi disable` and `svc data disable`; confirm no IPv4 address on any
   interface but `lo`, and that `ping` is unreachable.
3. The game holds at `battle_home_tier_1` for at least two minutes offline with
   no re-check and no modal.
4. Two scripted episodes then ran to completion offline, reaching waves 10 and
   11, both valid.

So automation can run genuinely offline. What cannot be avoided is a short
online window at application startup, during which the game contacts Firebase
and may do whatever else it does at launch.

### What this does not establish

- Whether the game synced save data, progression or telemetry during the
  startup windows of previous runs. The clone's account has drifted through
  play (`M1B-E005`), and nothing here distinguishes local drift from synced
  drift.
- Whether a longer offline session eventually triggers a re-check. Two minutes
  at home and two full episodes showed none; a four-hour run has not yet been
  observed under a verified-offline device.

### What changed

`scripts/instrumented_bridge.sh deploy` now refuses to run while any interface
other than `lo` holds an IPv4 address, printing the offending interface and the
commands that take it down. `verify` reports routable interfaces alongside the
package identity. A check that can pass while the premise is false is worse than
no check, so the interface is what is tested, not the setting.

## M1B-E009 — The 1,000-episode M2 reliability gate passes

**Date:** 2026-09-17
**Status:** Gate passed — 1,000 of 1,000 attempts valid, no invalid attempt to
classify, no silent corruption

One thousand consecutive scripted episodes ran unattended on the instrumented
clone at requested speed 64. This is the volume the M2 gate asks for, and it is
the first sample large enough to say anything about the tail.

| Quantity | Value |
| --- | --- |
| Episodes attempted | 1,000 |
| Valid episodes | 1,000 |
| Validity | 100 percent |
| Invalid attempts | 0 |
| `invalid_by_reason` | `{}` |
| `invalid_detail` | `{}` |
| Mean final wave | 9.758 |
| Median final wave | 10 |
| Standard deviation | 1.287 |
| Lower quartile | 10 |
| Range | 2 to 11 |
| Decisions | 51,840 |
| Episode wall time | 7,845 s |
| Total wall time | 15,168 s (4 h 13 m) |
| Episodes per hour | 237.3 |

### Does it pass

Yes, on every clause. The gate requires at least 1,000 consecutive attempts at
99 percent validity or better, every invalid attempt classified, and no silent
corruption. There were 1,000 attempts, validity was 100 percent, and the
`invalid_detail` breakdown the gate asks for is empty because there was nothing
to break down — which is the strongest form the clause can take, not an absence
of evidence: the same reporting path produced a populated breakdown in
`M1B-E007` and `M1B-E008`.

Nothing was relaxed to reach it. The validator is the one from `M1B-E008`:
negative health during a genuinely active run is still invalid, health above
maximum is still invalid in any lifecycle, and the death-boundary re-read is
still exactly one retry.

### What this does not pass

M2 in `task.md` is wider than the soak, and two of its exit criteria are still
open. Calling M2 complete on this evidence would be wrong.

- *"the selected training time scale and actor count pass documented parity,
  stability, and aggregate-throughput comparisons against normal-speed
  execution"* — not done. The speed equivalence gate has not been run, so 64 is
  the speed this soak used, not a speed shown to be equivalent to normal
  execution. Actor-count scaling is unmeasured entirely.
- *"recorded episode summaries agree with sampled visual evidence"* — not done
  in this run. Nothing was screenshot-verified against the bridge's summaries
  across these 1,000 episodes.

So: the reliability clauses of M2 pass on this evidence. M2 itself does not, and
training against it may not proceed on this entry alone.

### What changed since 150 episodes

`M1B-E008` measured 99.3 percent over 150 attempts, with its single failure
attributed to the death-boundary transient and the one-retry recovery added in
response. Over 1,000 attempts that failure mode did not produce a single invalid
episode. The recovery is therefore doing what it was built to do rather than
masking a rate that was about to reappear at volume.

The distribution is stable across the two samples, which is the point of quoting
it: mean 9.79 then 9.758, standard deviation 1.26 then 1.287. The
sample-size arithmetic the comparison protocol rests on is unchanged at about 23
evaluation episodes per arm for a one-wave difference, and it now rests on 1,000
episodes rather than 150.

The one number that moved is the minimum, from 5 to 2. At 150 episodes the worst
run reached wave 5; at 1,000 there is a run that died at wave 2. That is what a
longer tail looks like and not a defect — the episode was valid, classified as a
game over, and counted.

### Throughput

237.3 episodes per hour, against 236 measured over 150 episodes. Episode wall
time accounts for 7,845 s of the 15,168 s total, so a little under half the
run's wall clock is spent *between* episodes: the result panel settling, the
gated taps, and the restart. That gap is the obvious target if throughput ever
becomes the binding constraint, and it is device-side rather than model-side.

### A reporting defect this run exposed

The report's `game_speed` field reads `0.0`, and that is wrong in the sense that
it says nothing. `EpisodeSummary.game_speed` is sampled from the final state of
the episode, which is always the terminal one, and the game has stopped time by
then. The run did execute at speed 64 — `requested_speed` records it, and 237
episodes per hour with 51,840 decisions in four hours corroborates it — but the
field that claims to report the speed the episode *ran* at samples the one
instant that is never representative.

This matters for the speed equivalence gate, where the speed an arm actually ran
at is the entire independent variable. Recorded here and fixed rather than
worked around.

### Device

The stage closed as required: `scripts/instrumented_bridge.sh cleanup` restored
the original `libunity.so` (SHA-256 `ffc1f3ef…dd0040`), package identity is
unchanged (`versionCode 1199`, `versionName 29.0.3`, installer
`com.android.vending`), zero remaining mounts, bridge artifacts removed,
airplane mode still on afterwards, and no emulator left running.

## M1B-E008 — 150-episode reliability sample and the death-boundary transient

**Date:** 2026-09-17
**Status:** 99.3 percent validity over 150 episodes; residual attributed and
recovered. Superseded on volume by `M1B-E009`, which passed the full
1,000-attempt gate at 100 percent validity.

With rejection reasons now aggregated into evaluation reports, 150 consecutive
scripted episodes give the first reliability sample worth quoting.

| Quantity | Value |
| --- | --- |
| Episodes attempted | 150 |
| Valid episodes | 149 |
| Validity | 99.3 percent |
| Mean final wave | 9.79 |
| Median final wave | 10 |
| Standard deviation | 1.26 |
| Lower quartile | 10 |
| Range | 5 to 11 |
| Decisions | 7,829 |
| Episodes per hour | 236 |

The standard deviation is 1.26, matching the 1.22 from fifty episodes in
`M1B-E006`, so the sample-size arithmetic that protocol rests on is stable: about
23 evaluation episodes per arm for a one-wave difference.

### The single residual failure

One episode in 150 ended invalid, and its reason was recorded rather than
guessed:

```text
state: negative health in an active run
```

This is the same overkill behaviour as `M1B-E007`, caught one tick earlier. The
bridge reads tower health and the round-active flag separately within a snapshot,
so at the instant of death health has already gone negative while the game has
not yet flipped its game-over flag. The pair is briefly inconsistent, and the
inconsistency is real rather than corrupt: it is what the game looks like for one
moment as the tower dies.

Exclusion was the wrong response to it. Discarding an otherwise complete episode
because one snapshot caught a transition mid-flight loses a genuine game. The
environment now re-reads once when a state is invalid for exactly this reason,
and the settled state is authoritative. Exactly one retry: a state that is still
contradictory on the second read is a real failure and stays invalid, which a
test asserts directly.

The validator itself was not weakened. Negative health during a genuinely active
run remains invalid, health above maximum remains invalid in any lifecycle, and
the recovery is counted so a rising transient rate would be visible rather than
silently absorbed.

### Against the M2 gate

The gate requires at least 1,000 consecutive episode attempts at 99 percent
validity or better with no silent corruption. 150 attempts at 99.3 percent meet
the threshold but not the volume, so this is evidence toward the gate and not a
pass. The 1,000-episode run that followed is recorded in `M1B-E009` and passed;
it took 4 h 13 m at 237 episodes per hour.

## M1B-E007 — The invalid-episode rate was the validator, not the game

**Date:** 2026-09-17
**Status:** Invalid rate reduced from 24 percent to 2.5 percent; the residual is
not yet diagnosed

`M1B-E006` left a 24 percent invalid-episode rate, all classified
`observation_invalid`, with no recorded reason. The first fix was to record the
reason: an outcome without its cause cannot be diagnosed later, and a rate
without reasons cannot be fixed at all. `EpisodeSummary` now carries the
validator text that ended the episode, and evaluation reports aggregate it.

Fifteen instrumented episodes then gave an unambiguous answer. Every invalid
episode failed on exactly one validator, on exactly one observation:

```text
state: health fraction outside [0, 1]
```

always on the final reading of the episode, never in the middle.

### The reading was right and the validator was wrong

The killing blow overkills. The game stores the resulting negative tower health,
so the last observation of a run legitimately reports health below zero, and the
builder was treating that as an impossible reading. A genuine game over was being
classified as a corrupt observation and excluded from the distribution.

This was a modelling error about the game, not noise and not corruption. The fix
encodes the semantics the evidence revealed, rather than widening the bound until
the number improved:

- health above maximum is impossible in any lifecycle and stays invalid;
- health below zero while the run is still `active` is contradictory, because a
  dead tower is not an active run, and stays invalid;
- health below zero on a terminal state is overkill damage, is expected, and is
  clamped to zero without a complaint.

### Effect

Forty episodes after the fix, against fifty before it:

| Quantity | Before | After |
| --- | --- | --- |
| Invalid rate | 24 percent | 2.5 percent |
| Valid episodes | 38 of 50 | 39 of 40 |
| Mean final wave | 9.74 | 9.72 |
| Standard deviation | 1.22 | 1.26 |
| Episodes per hour | 185 | 228 |

The wave distribution is unchanged, which is the expected result: the excluded
episodes were ordinary games all along, so admitting them correctly moves the
validity rate without moving the performance figures. That agreement is itself
evidence the diagnosis was right rather than merely convenient.

One episode in forty still ends invalid. Its reason was not captured because
reason aggregation reached the evaluation report only after that run; it will be
attributable on the next measurement. At 2.5 percent this remains above the M2
gate's 1 percent allowance, so it is the next thing to diagnose rather than a
result to build on.

`docs/rl-candidates.md` has been corrected: its evaluation-power section was
built on the superseded variance estimate and asked for about 140 episodes per
arm where the measured variance asks for about 23. The original estimate is
described rather than deleted, because the lesson that a variance guessed from
three samples can be off by a large factor is exactly why the measurement exists.

Cleanup verified the original `libunity.so` SHA-256, unchanged package identity,
no mounts, no leftover artifacts, airplane mode enabled and no emulator running.

## M1B-E006 — Stage B: pause-stepping reversed, and the variance that sets the protocol

**Date:** 2026-09-17
**Status:** Pipeline runs end to end on the device; scripted variance measured;
a 24 percent invalid-episode rate is the next blocker

The completed pipeline ran against the real clone for the first time. Three
findings, one of which reverses a decision made two entries ago.

### Pause-stepping is withdrawn

`M1B-E003` concluded that above roughly 16x the environment should pause between
decisions, because a 50 ms host round trip is 3.2 seconds of game time at 64x and
the world otherwise runs away from the policy. That reasoning was about decision
density and it was correct about density. It was wrong about cost.

The same scripted policy, same device, same speed:

| Mode | Final wave | Wall seconds per episode | Episodes per hour |
| --- | --- | --- | --- |
| Pause-stepping | 3 | 273 | 13 |
| Free running | 10 | 14.6 | about 245 |

Every slice pays a host round trip and a wall-clock floor, and at a 250 ms slice
an episode needs hundreds of them, so the overhead dominates completely. Pausing
is roughly nineteen times slower and plays worse, because a decision advancing up
to two seconds of game time also buys less often. The default is now free
running; a finite pause threshold remains configurable if decision density is
ever shown to bind.

A related failure was found by accident. A free-running run immediately after a
stepped one produced no valid episode at all, because the stepped session left
the game paused and a paused game outlives the client that paused it. Releasing
the pause is now part of shutting the adapter down.

### The variance that every protocol number depends on

Fifty episodes of the scripted policy at 64x:

| Quantity | Value |
| --- | --- |
| Valid episodes | 38 of 50 |
| Mean final wave | 9.74 |
| Median final wave | 10 |
| Standard deviation | 1.22 |
| Lower quartile | 9 |
| Range | 6 to 11 |
| Episodes per hour | 185 |

The standard deviation is 1.22 waves, not the roughly 3 estimated from three
episodes in `M1B-E003`. That estimate was quoted in `docs/rl-candidates.md` to
argue that about 140 evaluation episodes per arm would be needed; on the measured
variance the requirement is far smaller:

| Difference to detect | Episodes per arm | Wall time at 185 per hour |
| --- | --- | --- |
| 0.5 wave | 94 | 30 minutes |
| 1.0 wave | 23 | 8 minutes |
| 1.5 wave | 10 | 3 minutes |
| 2.0 wave | 6 | 2 minutes |

Two-sample, eighty percent power, five percent significance. Detecting a one-wave
difference costs about eight minutes per arm, which makes seeding and
interleaving arms cheap rather than aspirational. It also sets an honest floor on
what may be claimed: a half-wave difference needs ninety-four episodes per arm
and must not be asserted from fewer.

### The next blocker: a 24 percent invalid-episode rate

Twelve of fifty episodes ended `observation_invalid` rather than `game_over`.
That is the whole reason for a validity taxonomy: those episodes are excluded
from the distribution above rather than quietly averaged into it, so the wave
figures are drawn from genuine episodes only.

It is nonetheless far from the 99 percent validity the M2 gate requires, and it
must be diagnosed before any soak or baseline measurement is trusted. The
classification is recorded but its cause is not yet known; the candidates are the
transition validators in `domain/run_state.py`, a stale observation crossing an
episode boundary, and the free-running stream advancing its sequence between a
read and the command bound to it.

Throughput here was 185 episodes per hour against the 502 measured in
`M1B-E003`, which was taken under host GPU rendering, on a simpler loop, and
without the boundary tap and its six-second settle. Re-measuring throughput under
lavapipe with the real loop remains open.

Cleanup verified the original `libunity.so` SHA-256, unchanged package identity,
no mounts, no leftover artifacts, airplane mode enabled and no emulator running.

## M1B-E005 — Recalibrating the screen gate against the game's own lifecycle

**Date:** 2026-09-17
**Status:** Gate recalibrated and validated on live frames; stage B unblocked

`M1B-E004` left the boundary tap ungateable. Recalibration used the bridge itself
as ground truth rather than assumption: each captured frame was labelled by the
game's own lifecycle, so the anchors were fitted to what the game says it is
showing rather than to what the screen was assumed to be.

Sixty-three frames were collected on the clone under `-gpu lavapipe`: nine at
Battle home, thirty-seven during active runs, and seventeen at the result panel
across three episodes. Home frames were obtained by restarting the app rather
than by tapping, so no ungated tap was needed to break the deadlock.

### What the search found, and why the first answers were rejected

A grid search for pixels constant within a lifecycle state and never seen in the
others produced 3,984 candidates for home. Nearly all were plain background, and
a signature made of background would also match a full-screen modal covering
home, which is precisely the case the gate exists to catch. Those were rejected
in favour of distinctive values.

Requiring a single pixel to separate all three states found exactly one. A single
anchor is what failed in `M1B-E004`, so redundancy was required instead: every
anchor of a screen must match, and one repainted region therefore fails closed
into `unknown` rather than silently matching.

The result panel initially yielded no stable anchor at all across seventeen
frames, with the same pixel varying by up to 240 per channel. The cause is that
the panel animates in and frames were being captured from the moment the bridge
reported terminal. Restricted to frames at least six seconds after termination,
every candidate anchor became exactly stable, spread zero. The adapter's settle
delay is now six seconds for that reason, and classifying earlier correctly
returns `unknown` rather than a screen.

### The calibrated profile

Anchors were then chosen in structurally meaningful places rather than wherever a
pixel happened to be constant: for home the header bar, the title, both panels,
the BATTLE button's border and interior, and the navigation bar; for the result
panel its interior plus both of its buttons; for an active run the health bar,
the upper HUD and the playfield.

Anchoring the result gate on the RETRY button itself is deliberate. The gate then
confirms that the control it is about to press is actually rendered where it is
about to press, rather than inferring it from the surrounding panel.

Validated against all sixty-three live frames, the profile classifies home 9 of 9,
active 37 of 37, and the result panel 12 of 17, where the five it declines are
exactly the mid-animation frames. Declining those is the desired behaviour: a
frame captured during a transition is not a screen, and tapping across a
transition is the `M1-E005` failure.

The profile is versioned `tower-play-29.0.3-clone-wave11-v2` and is bound to the
progression profile it was calibrated against, as ADR 0008 implies. Only the
sampled anchor values are recorded; screenshots carry account state and are not
committed.

Cleanup verified the original `libunity.so` SHA-256, unchanged package identity,
no mounts, no leftover artifacts, airplane mode enabled and no emulator running.

## M1B-E004 — Progression drift breaks the calibrated screen gate

**Date:** 2026-09-17
**Status:** Stage B blocked at the boundary tap; two findings, one of which
reverses the previous entry's renderer recommendation

Wiring the completed pipeline to the real clone stopped before a single episode
ran, for a reason worth more than the episodes would have been.

### Host GPU rendering corrupts the frame

`M1B-E003` recommended `-gpu host` on the strength of boot time and an unbroken
speed ceiling. Under sustained use it renders the game incorrectly: persistent
smearing across large triangular regions, magenta and cyan banding over icons,
and ghosted text. Game logic is unaffected, because the bridge reads exact state
rather than pixels, but the frame is not trustworthy. Screen classification
returned `unknown` and `supported_modal` on a screen that was plainly Battle
home.

That recommendation is withdrawn for any configuration that must classify the
screen. The clone was returned to `-gpu lavapipe`, which renders correctly. The
throughput measurements in `M1B-E003` were taken under host rendering and are
therefore an upper bound that still needs confirming under lavapipe; the earlier
lavapipe sweep did reach 32x with no saturation, so the loss is expected to be
small but is not yet measured.

The safety gate behaved correctly throughout: with the screen unclassifiable, the
adapter refuses to tap rather than tapping anyway.

### The account has drifted out of its documented baseline

Under lavapipe the frame is clean and classification still fails. The cause is
not the renderer.

The documented fixed baseline is Highest Wave 2 with 53 coins. The clone now
reports Highest Wave 11 with 909 coins, and its home screen carries UI that the
baseline did not: a `MILESTONES` button with an unread badge, and a gem and
video-reward widget in the top-left corner. Those appeared because episodes were
played, not because anything was spent.

The calibrated classifier samples three anchors for Battle home. Two still match
exactly. The third, at pixel (10, 200), sampled the dark background at the
baseline and now falls inside the new top-left widget, reading pure white
(255, 255, 255) against an expected (28, 24, 53). One anchor landing on
progression-unlocked UI is enough to make the screen unclassifiable, which
refuses the boundary tap, which prevents any unattended episode from starting.

### What this means

Playing the game necessarily changes visible permanent state. Coins accumulate
and the highest-wave record advances even though nothing combat-affecting was
purchased and no progression was spent, so this is not a violation of the frozen
baseline in the sense ADR 0008 governs. It is nonetheless real drift: the visual
profile is bound to the progression profile, exactly as ADR 0008's profile
identity implies, and the two must be versioned together.

Two consequences follow. Calibration anchors must be chosen in regions that
progression does not repaint, and verified against a live frame rather than
assumed to hold. And the baseline fingerprint must separate combat-affecting
permanent state, which must not change, from earned-record state such as coins
and the highest-wave record, which necessarily accumulates during training; a
fingerprint that fails on the second would fail on every training run.

No episodes were run, nothing was spent, the overlay was unmounted, `libunity.so`
again matched its original SHA-256, package identity was unchanged, airplane mode
was re-enabled and no emulator was left running.

## M1B-E003 — Throughput ceiling, renderer, and decision cadence

**Date:** 2026-09-17
**Status:** Throughput measured to 64x with no saturation; equivalence gate not
yet attempted

Wall-clock environment time is the binding constraint on the whole benchmark, so
this entry establishes what the host can actually deliver. All runs use the same
scripted greedy policy on the private rooted clone, three episodes per
configuration.

### The renderer is an enabler, not a speed-up

Unity clamps how much game time a single frame may advance, so the usable time
scale is bounded by the achieved frame rate. The clone had been running under
software `lavapipe`, chosen when pixel stability mattered for OCR. It needs
pixels only for two boundary classifications per episode, so it was moved to
`-gpu host` on the RTX 4090.

Boot fell from over a minute to 10.3 seconds and the game reached Battle home in
about 30 seconds rather than about 75. More importantly the frame-rate clamp
never became the binding constraint at any speed tested below. The renderer does
not make the simulation faster; it removes the ceiling that would otherwise cap
it. Screen classification still returns `battle_home_tier_1` under host
rendering, so the boundary-tap safety gate survives the change.

`dumpsys SurfaceFlinger --latency` returned no frame rows for the Unity
`SurfaceView` layer, so frame rate was not measured directly. The saturation
point of effective speed-up would imply it, and no saturation was found.

### Measured throughput

| Requested speed | Wall seconds per episode | Episodes per hour | Decisions per episode | Decisions per wave | Final waves |
| --- | --- | --- | --- | --- | --- |
| 1.5 (reference) | 175 | 21 | 528 | 63 | 7, 8, 10 |
| 4 | 57 | 63 | 447 | 79 | 6, 7, 4 |
| 8 | 34 | 105 | 435 | 69 | 6, 6, 7 |
| 16 | 19 | 190 | 273 | 39 | 5, 8, 8 |
| 32 | 11 | 321 | 162 | 20 | 10, 7, 7 |
| 32 (after cadence fix) | 13 | 273 | 230 | 25 | 10, 8, 10 |
| 48 | 9.9 | 365 | 177 | 18 | 10, 10, 10 |
| 64 | 7.2 | 502 | 154 | 16 | 8, 10, 11 |

Effective speed-up held at roughly 66 to 70 percent of nominal at every level and
did not saturate through 64x, which is about 24 times the episode throughput of
the normal-speed reference.

### Decision density, not speed, is what degrades

Decisions per wave fell from 63 at the reference to 16 at 64x. Two separate
causes were found.

The first was the bridge's own cadence floor. Stream and `WAIT` intervals scale
with game speed, but were floored at 20 ms, which binds above 12.5x and cut
decisions per episode at 32x to 162. Lowering the floor to 4 ms raised that to
230 and raised mean final wave from 8.0 to 9.3 in the same configuration.

The second is host round-trip latency and it is now the binding constraint:
roughly 50 ms per decision. At 64x, 50 ms of wall clock is 3.2 seconds of game
time, so the world runs away from the policy while it decides. No cadence
setting can fix this, because the cost is not in the bridge.

This inverts the earlier conclusion in `M1B-E002` that pause-stepping is not
worth its overhead. That was measured at 1.5x, where free-running is cheap. At
32x and above, pausing between decisions is what makes decision density a choice
rather than a consequence of latency, because deliberation then costs no game
time at all. High time scale advances the world; pause controls the cadence;
neither alone is sufficient.

### Equivalence is not established

Mean final wave was 8.33 at the reference and 9.67 at 64x, and every intermediate
configuration fell between. It would be wrong to read that as evidence of
equivalence, or of improvement. The scripted policy's final-wave standard
deviation is roughly three waves and each configuration here has three episodes,
so these distributions are statistically indistinguishable in both directions.
What the data supports is the narrower claim that no gross divergence appeared up
to 64x.

The equivalence gate therefore remains unpassed, and passing it requires first
measuring the scripted policy's own variance over a much larger sample. Until
then no speed above the validated normal-speed reference is admissible for a
result that is reported as a behavioral claim.

## M1B-E002 — Screen-free in-run control, speed, and pause-stepping

**Date:** 2026-09-16
**Status:** In-run control proven without pixels; episode boundary still needs one
tap; speed applies but stepped mode is not yet faster

After `M1B-E001` proved single commands, this entry takes the loop to a whole
episode and probes the throughput levers. The product decision recorded here is
that the private instrumented clone is the primary training and evaluation
environment, with a small official-profile cross-check retained at promotion.

### Discovering semantic members without a metadata dump

A build-flag-gated diagnostic (`TOWER_BRIDGE_DIAGNOSTICS`) enumerates class
members through exported IL2CPP APIs and logs them. It needs no `global-metadata.dat`
extraction and no third-party dumper, and it is absent from an ordinary build.
It reported 450 methods and 920 fields on `Main`, and a substring scan across
every class located members that do not live on `Main`.

### In-run control needs no screen

One greedy scripted episode driven entirely through the bridge reached wave 7
with 21 confirmed purchases in 136.8 seconds, and a second reached wave 8 with 24
purchases in 160.9 seconds. Buying nothing dies at wave 2. Costs are refreshed by
dispatching the game's own `UpgradeCostCalc`, `UpgradeDefenseCostCalc`, and
`UpgradeUtilityCostCalc`, which removes the `M1B-E001` requirement to open each
family tab by hand.

### The episode boundary still needs one tap

`Main` only exists inside the battle scene, so no `Main` method can start a run
from the home screen. From a terminal run, `StartNewRoundFunction`,
`AutoRetryBattle`, and `Button_ToggleAutoRestartBattle` were each dispatched and
each expired its 30-second lifecycle wait without starting a round; the
auto-restart feature appears progression-gated at this baseline. The class scan
located `BattlePanelUI.StartNewRound`, which is the likely handler, but
`UnitySendMessage` addresses a GameObject by name and that object's name is not
yet known.

The loop therefore uses one bridge-gated tap per episode: the bridge's own
terminal state selects the control, and the bridge confirms the new run. No
screenshot or OCR is involved, and at roughly 50 ms against a 30-to-175-second
episode it is not a throughput concern. Finding the correct receiver remains open
work.

### In-run clock: corrected by later evidence

An earlier draft of this entry concluded that the game holds no live clock. That
conclusion was drawn from two fields and was wrong. `roundTime`,
`gameplayTimeThisRound`, and `realTimeThisRound` do all read 0.0 for the whole of
a live run, and are presumably populated only for the end-of-run report. A search
of the full 920-field inventory found `playTime`, which does advance
continuously.

`playTime` is not an in-run game clock. It is account-lifetime and unscaled:
sampled over eight seconds it advanced 8.10, 7.68, and 8.07 at game speeds 1.5,
4.0, and 8.0, a ratio of 1.00, 0.95, and 0.99 against wall time. It therefore
measures real time regardless of how fast the simulation runs.

It is reported as liveness evidence rather than as a policy feature or a game
clock: a hung game process stops advancing it, which no other observed field
proves. Elapsed in-run game time remains controller-owned, and in-run progress is
measured by the game's own wave and cash.

### Speed applies, and cadence must scale with it

This baseline's own speed ceiling is 1.5, consistent with its Highest Wave 2
progression, so `SpeedChangeMax` reports success while leaving `gameSpeed` at
1.5. Writing `gameSpeed` and dispatching the game's own `GameSpeedModifier`
applied 4.0 and 8.0, confirmed by the observed `game_speed`.

A first 4.0 comparison looked worse than 1.5 — final waves 5, 6, 3 against 2, 8,
8 — but the cause was the host loop, not the game. Decision cadence was fixed in
wall-clock time, so a faster game received proportionally fewer decisions per
game second: 112, 199, and 61 decisions per episode against 87, 356, and 355.
After the stream and `WAIT` intervals were made proportional to game speed, 4.0
produced waves 7, 4, and 10 with 572, 261, and 785 decisions. Wall-clock cost
fell from roughly 129 seconds per episode at 1.5 to roughly 68 seconds at 4.0,
an effective speed-up near 1.9 rather than the nominal 2.67. Three episodes per
arm is not a parity result; it is a throughput observation and a demonstration
that a speed-unaware loop silently starves the policy.

### Pause makes the environment turn-based, but is not yet faster

`Pause` and `Unpause` freeze and resume the world exactly: across six paused
seconds cash, health, and wave were unchanged, and cash resumed advancing after
`Unpause`. A `step` command brackets a bounded slice of game time between them,
so policy latency costs no game time.

Measured, stepped mode is currently slower than free running. At roughly 2.3
steps per wall-clock second, each decision costs about 430 ms while only 80 to
166 ms of that is unpaused, so the world is frozen for most of the wall clock
and 25 seconds advanced at most one wave. Making the bridge's pacing wait
interruptible by an inbound command did not change the rate, so the remaining
cost is elsewhere and must be profiled rather than guessed. Until then the
free-running loop with speed-scaled cadence is the faster configuration. A step
window is also floored in wall time, because at a high speed the requested slice
can be shorter than one rendered frame and no world time would pass at all.

### Upgrade inventory

Each of the 60 entries reports family, index, current cost, current level, its
own maximum level, and the `unlocked`, `tier_unlocked`, and `maxed` flags, so a
policy sees exactly which upgrades exist, which are currently offered, what each
costs now, and how much headroom each has. Live ceilings differ sharply per
upgrade: attack 0 caps at 6000 while attack 1, 2, and 3 cap at 99, 79, and 150.
Six of the 60 are offered at this fixed baseline: four attack and two defense,
with utility unavailable. `max_level` had been read and validated but never
serialized; it is now reported, and a level above its own maximum is rejected as
contradictory state.

### Protocol

The handshake now advertises `semantic-v2`. Policy actions remain `wait` and
`buy_upgrade`; `lifecycle`, `set_speed`, and `step` are separate controller-owned
kinds, so navigation and speed can never become learned actions. A run that is
not initialized is reported as its own `run_unavailable` state carrying the same
monotonic sequence, rather than as invented run values or a dropped connection,
which is what lets a controller act between episodes.

### Cleanup

The overlay was unmounted, staged files were removed, `libunity.so` again matched
its original SHA-256, Package Manager still reported 29.0.3, version code 1199,
and `installerPackageName=com.android.vending`, airplane mode was re-enabled, and
no emulator was left running.

### Open before M1B

Profile the per-decision cost and decide between stepped and free-running modes;
find the `BattlePanelUI` receiver so the episode boundary needs no tap; run a
real parity comparison with enough episodes to compare final-wave distributions
at each speed; and establish the actor-count scaling curve. No instrumented
transition may enter replay until parity and quarantine gates pass.

## M1B-E001 — Live semantic command path and family cost coverage

**Date:** 2026-09-16
**Status:** `WAIT` and earned-cash purchases proven live; utility unavailable at
this baseline; family cost coverage requires tab activation

The command slice written in M1-E007 step 3 compiled but had never executed
against the running game. Five defects were found and corrected before any live
claim could be made. The native command parser derived one field offset from a
literal length by hand and was one byte short, so every well-formed command would
have been rejected as malformed. Purchase confirmation required a cash decrease
and treated any cash change as a contradiction, but in-run cash rises
continuously from kills; a real purchase would have been reported ambiguous. The
host client consumed one queued observation per decision while the bridge streams
at a fixed cadence, so it fell progressively behind and bound its commands to
superseded sequences. IL2CPP resolution had been moved to library-load time,
four seconds after `libil2cpp.so` merely appears; that is far earlier than the
proven path and `il2cpp_domain_get` killed the game process twice before a client
ever connected. Finally, the precondition required `tier_unlocked`, which live
29.0.3 reports as false for every upgrade the game actually offers, so no
purchase could ever pass.

Resolution is now deferred to the first client connection and cached, so a
reconnect does not repeat the stabilization delay and the game is demonstrably
initialized before the runtime is touched. In-run availability is `unlocked`,
`maxed`, and a positive cost against current cash; `tier_unlocked` is reported
state, not a gate. Confirmation is the game's own level increment.

On the private rooted clone running the unchanged Play-installed 29.0.3 package
with the reversible overlay, the version-locked handshake passed and exact
observations reported lifecycle, wave, cash, tower health, terminal state, and 60
upgrade entries. The following command outcomes were observed live:

- `WAIT` returned `confirmed` / `wait_elapsed` bound to a strictly newer
  observation;
- a repeated request id returned `rejected` / `stale_or_duplicate`;
- a locked upgrade returned `rejected` / `precondition_failed`;
- an unpriced entry returned `rejected` / `precondition_failed`;
- `attack[2]` returned `confirmed` / `confirmed_state_change` with level 0 to 1,
  cost 4.0 to 6.0, and cash 104.0 to 100.0; and
- `defense[1]` returned `confirmed` / `confirmed_state_change` with level 0 to 1,
  cost 5.0 to 7.0, and cash 97.0 to 92.0.

Sparse pixel evidence agreed with the bridge in both directions. A terminal
bridge observation reporting wave 2 and cash 110.0 matched the visible result
screen, and the confirmed attack purchase was visible as Critical Chance moving
from 1.00% at $4 to 2.00% at $6. One purchase showed cash falling by three while
its cost was four, which is kill income arriving inside the confirmation window
and is exactly why cash is not a confirmation signal.

A material observation gap was found. A family's cost array is only populated
once that family's tab has been opened during the run. Before any tab
interaction the run reported 17 priced attack entries, 0 priced defense entries,
and 0 priced utility entries; opening the defense tab produced 18 priced defense
entries and opening the utility tab produced 13 priced utility entries. Reading
the arrays alone therefore does not satisfy the first ADR 0006 acceptance gate.
Until the game's own refresh path is identified, an actor must open each family
tab once at run start, and any entry without a positive cost must remain masked
and rejected rather than treated as free.

Utility is unavailable at this fixed baseline rather than unsupported: with all
tabs opened, utility reported 13 priced entries and 0 offered entries, while
attack offered 4 and defense offered 2. A utility purchase therefore cannot be
demonstrated from this baseline and is recorded as unavailable with evidence.

Contrary evidence and failures are part of this entry. The clone first ran under
`swiftshader_indirect`, which produced a System UI ANR and an unusable screen;
`-gpu lavapipe` with eight cores ran the same package normally and was used for
all reported results. A deploy that removed the overlay's backing file before
unmounting left the target path resolving to a deleted inode, so the next mount
failed; the deploy and cleanup paths now unmount until the target's SHA-256
matches the original library. One blind coordinate tap, sent without classifying
the screen first, opened Settings and then the Account panel. No account,
credential, link, logout, or cloud action was selected, both panels were closed
with verification between taps, and the account remained not linked. This repeats
the M1-E005 lesson: every tap must follow a positive screen classification, and
the live helpers used for this entry classify the screen before acting.

Reversibility was verified after the run. The overlay was unmounted, the staged
bridge and library files were removed, `libunity.so` again matched its original
SHA-256, and Package Manager still reported version 29.0.3, version code 1199,
and `installerPackageName=com.android.vending`. Airplane mode was re-enabled and
no emulator was left running. The patched library, bridge binary, private build
directory, and live helper scripts remain machine-local and uncommitted.

Remaining before M1B can be claimed: family cost coverage without manual tab
activation or an explicit documented actor step, deterministic normal-speed
scripted parity against the visible controller, protocol-loss and
thread-affinity quarantine behavior, and the speed equivalence gate. No
instrumented transition may enter replay until those pass.

## M1-E006 — Exact-state and accelerated-runtime feasibility review

**Date:** 2026-09-15
**Status:** Static review complete; live reliability work intentionally paused

The user stopped the active M1 reliability retry because six-minute real-time
episodes plus screenshot recognition are not an acceptable production training
path without a stronger scaling result. The runner received a graceful
interrupt, restored the 6 GiB offline baseline, and a final probe verified the
Battle home screen, Tower foreground, airplane mode enabled, no external route,
and no observation errors. The emulator was then stopped through its console.
The scheduled continuation and its active implementation worker were also
stopped so they cannot restart Android implicitly. The interrupted partial run
is not acceptance evidence.

A field review found three materially different integration classes:

1. DeepMind's AndroidEnv supports ordinary unmodified Android applications with
   pixels and touchscreen actions, but its own documentation states that Android
   remains a real-time simulation whose speed cannot be increased. This validates
   actor parallelism and observation reduction as black-box optimizations, not a
   hidden route to faster game time.
2. Unity ML-Agents supports direct structured observations/actions, concurrent
   environments, `time_scale`, and graphics-disabled execution when the Unity
   project/build has been instrumented for ML-Agents. Those controls cannot be
   attached to an arbitrary signed production APK without developer integration
   or modifying/rebuilding the game. The clean exact-and-accelerated architecture
   therefore requires a TechTreeGames-provided training/debug build or supported
   telemetry/control bridge.
3. Tower-specific community tools expose useful but narrower precedents.
   WaveTrace watches the rendered game and performs event-triggered OCR when the
   wave advances. TheTowerSDK decodes `playerInfo.dat`; it documents permanent
   state, completed-run history/battle reports, and a read-only ADB watcher. This
   could replace visual extraction for baseline/progression verification and
   terminal evaluation if a controlled cadence test confirms when the game writes
   the required fields. No reviewed evidence establishes exact high-frequency
   live cash, health, visible run-upgrade state, or an action interface.

Android UI Automator remains worth one bounded audit because it can read and act
on accessibility nodes in release applications. Unity/custom-rendered controls
only appear as useful semantic nodes when the application developer supplies an
accessibility hierarchy, so this is a feasibility check rather than an assumed
solution.

The next decision gate is deliberately bounded and keeps the emulator off until
run: (a) inspect the live accessibility hierarchy, (b) measure save-file write
cadence and fields during one short controlled run without retaining save bytes,
(c) benchmark stable hardware-renderer and headless profiles for aggregate actor
density, and (d) seek a developer-supported bridge if exact faster-than-real-time
simulation is required. Save parsing must remain read-only, local, runtime-only,
and separated from TheTowerSDK's mechanics/formulas; training on those formulas
would create a synthetic approximation and violate the authoritative real-APK
objective.

Sources reviewed:

- https://github.com/google-deepmind/android_env
- https://unity-technologies.github.io/ml-agents/Training-ML-Agents/
- https://unity-technologies.github.io/ml-agents/Python-LLAPI/
- https://developer.android.com/training/testing/other-components/ui-automator
- https://developer.android.com/guide/topics/ui/accessibility/views/custom-views
- https://developer.android.com/studio/run/emulator-commandline
- https://developer.android.com/studio/run/emulator-acceleration
- https://github.com/sbrants/wavetrace
- https://github.com/TmRxJD/TheTowerSDK

## M1-E007 — XAPK/save/runtime instrumentation feasibility audit

**Date:** 2026-09-15
**Status:** Exact read-only production bridge proven live; command and parity gates pending

The locally supplied 29.0.1 XAPK is an official four-split ARM64 Unity IL2CPP
package, not an independently signed repack. Its signing certificate matches the
Play-installed 29.0.3 package exactly. Modifying and re-signing that XAPK would
therefore discard the official signature and risks breaking Play licensing and
signature-bound Google/Firebase integrations. Tower-RL will not bypass those
checks or claim a re-signed package is the official runtime.

Static metadata version 39 was decoded locally with `il2cpp_dumper` 0.7.0. The
29.0.3 release exposes exact run state in `Main.Instance`, including cash, wave,
tower health, game-over and round-active flags, all three run-upgrade cost/level
arrays, and the game-speed fields. Its own attack, defense, and utility purchase
methods can accept the semantic selection used by the UI. `PlayerData` also
contains exact terminal/run fields and upgrade arrays. These findings establish
technical feasibility for a local event-driven bridge that observes the real
compiled game state and invokes the game's normal purchase methods, without
making learned screen coordinates part of the policy.

The game's speed modifier ultimately sets Unity `Time.timeScale`; the method has
an uncapped branch above the normal 5x UI range. An instrumented runtime could
experimentally set a higher speed and invoke that same method. This remains a
speed-altered runtime, not automatically equivalent acceptance evidence. Any
accelerated profile must be compared with normal-speed official evaluation using
fixed scripted actions and available seed/state controls; speed is capped or
rejected when transition order, outcomes, or distributions diverge.

Two tempting passive alternatives failed the live audit:

- Android UI Automator exposed only the full-screen Unity surface and no semantic
  text or controls, so accessibility cannot replace image recognition.
- `playerInfo.dat` decoded successfully and is useful for baseline and terminal
  verification, but during a live Tier-1 run its size, modification time, and hash
  were unchanged after 12 seconds while the controller observed wave 1 and cash
  80. Static disassembly also showed a normal 300-second autosave interval and a
  direct game-over save. Save watching is therefore not a decision-frequency
  observation source, and the file remains read-only.

The current Google Play emulator image is a production build: `adb root` is
disabled and the app is neither debuggable nor profileable. A separate disposable
clone was therefore created from a private copy of the system image and rooted
with Magisk 30.7. The canonical AVD, its snapshots, and the SDK-managed system
image were not modified. The unchanged Play-installed package launched on the
visibly rooted clone without a distinct integrity rejection, although a cold
offline launch still stopped at the already-known entitlement/network panel.

Two generic instrumentation routes were then separated experimentally:

- An x86_64 Frida 17.18 server could enumerate and attach to the ARM64 game
  process through Android's native translation layer. The injected x86 agent
  could not enumerate the translated ARM64 `libil2cpp.so`, so normal Frida module
  lookup and IL2CPP hooks are not viable on this x86_64 AVD.
- An ARM64 Frida Gadget added to a disposable, re-signed XAPK was loaded by the
  ARM linker but aborted in its constructor inside `libndk_translation`. This is
  an observed Frida/native-translation incompatibility, not evidence that custom
  ARM64 code cannot run.

A minimal custom ARM64 shared library proved the narrower mechanism. It was
loaded as a `DT_NEEDED` dependency of `libunity.so`, started a background thread,
resolved `il2cpp_domain_get` and the other required IL2CPP exports, found the
unnamespaced `Main` class dynamically, and read the static `Main.gameSpeed`
field. The re-signed XAPK process remained alive and reported `00.00` on its
pre-game screen. This establishes an exact, non-OCR state path through the real
game runtime without relying on Frida.

The same bridge was then tested against the unchanged Play-installed 29.0.3
package. A Magisk bind mount over only the extracted `libunity.so` supplied the
added dependency while leaving the signed split APKs, package version, installer
identity, and app data unchanged. The process remained alive, package manager
still reported `installerPackageName=com.android.vending`, and the bridge again
read `Main.gameSpeed`. The overlay was removed after the proof; reboot would also
remove it. This is the preferred workstation training route because it preserves
the official package identity and avoids re-signing. It still requires a private
rooted actor clone and is not the canonical evaluation runtime.

The repository bridge implementation was subsequently built for ARM64 with NDK
29.0.14206865 and validated on the rooted 29.0.3 clone. Its version-locked
handshake matched package version/code, official signer, original `libunity.so`,
`libil2cpp.so`, Unity version, metadata version, bridge version, and training
profile. During an active Tier-1 run it returned `lifecycle=active`, wave, cash,
current/max health, round flags, and 60 complete upgrade entries: 20 Attack, 20
Defense, and 20 Utility. A second sample after natural death returned
`lifecycle=terminal`, zero health, and the same complete inventory. These two
samples prove exact active-to-terminal state coverage and the full three-family
inventory shape without OCR.

The bridge also failed closed during startup: before `Main.Instance` existed it
reported an unavailable observation, and before a run initialized all scalar
state it rejected the snapshot rather than emitting plausible defaults. The
initial four-second IL2CPP stabilization delay means the current host client
must allow more than four seconds for its first handshake; subsequent snapshots
are emitted at the configured 250 ms cadence. Startup-state modeling and a
shorter connection path remain implementation work.

Cleanup was verified after a clean reboot. The temporary overlay and bridge
files were absent, `libunity.so` again matched its original SHA-256
`ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
and Package Manager still reported version 29.0.3, version code 1199, and
`installerPackageName=com.android.vending`. No emulator was left running.

The read-only bridge emits versioned observations over an ADB-forwarded local
socket. Wave start, state changes, purchase result, death, and stall/invalid
outcomes remain the target protocol once commands are added. Read-only IL2CPP
access and complete inventory decoding are now proven. Main-thread command
dispatch, earned-cash purchase confirmation, active-run event cadence, reset
handling, and accelerated-time equivalence are not yet proven and must fail
closed until they are. The intended action path queues a semantic command,
executes it on Unity's main thread, sets the game's own upgrade selection, and
invokes the corresponding game method so cost and validity remain owned by the
game.

The implementation sequence resulting from this audit is:

1. Record a product/architecture decision that introduces a separate
   `instrumented-training` profile without weakening official normal-speed
   evaluation.
2. Build the small versioned ARM64 bridge and host socket protocol; expose exact
   lifecycle, wave, cash, health, upgrade-cost/level, and terminal observations.
3. Add a Unity-main-thread command queue for `WAIT` and every supported semantic
   in-run purchase, with before/after confirmation and explicit rejection.
4. Validate normal death/reset and deterministic scripted episodes against the
   existing visible controller, retaining sparse pixels as a watchdog rather
   than decision-frequency OCR.
5. Sweep higher `Time.timeScale` values and actor counts. Accept a speed only
   when scripted transition order and outcome distributions remain equivalent to
   the normal-speed official reference; otherwise cap it.
6. Keep exploration-free best-model evaluation and headed watch mode on the
   unchanged official profile. Instrumented training data and checkpoints record
   the exact game hash, bridge version, speed, and actor profile.

Save inspection remains diagnostic only. `playerInfo.dat` is too infrequently
updated for decisions, and neither the save nor cloud/account state is edited.
No root hiding, integrity bypass, purchase/ad/competitive automation, save
editing, or cloud operation is part of the route. Pixels remain a fail-closed
watchdog and the official headed evaluation path; OCR leaves the training
decision loop only after bridge parity passes.

All proprietary APKs, metadata dumps, save bytes, patched binaries, signing keys,
and runtime artifacts used for this audit remained outside the repository in
private temporary storage. Both disposable test AVDs and the canonical emulator
were stopped after the bounded test.

## M1-E005 — Delayed modal-tap account-UI safety failure

**Date:** 2026-09-15
**Status:** 10-attempt retry stopped after 3 valid episodes; 100 gate not started

The fresh 10-consecutive retry stopped after three valid episodes in 1,118.3
seconds. It failed while waiting for the next active run with a supported modal;
the failure frame showed the Home Settings panel. Tower remained foreground,
the runner restored the offline baseline, and the lifecycle log contained no
crash, ANR, LMK, or process death. The 100-consecutive gate was not started.

A bounded four-episode reproduction again reached three valid episodes and then
failed during result-to-home navigation. Its pre-recovery frame showed Settings,
the Account panel, and an unlinked-cloud-save logout warning. No confirmation,
credential, link, logout, or other account action was selected. This proves the
generic top-right modal-close tap can cross a slow scene transition, land on the
Home Settings control, and allow later lifecycle taps to cascade into unsafe
account UI. Merely delaying the close tap by two modal classifications did not
remove the race.

Lifecycle waits no longer send generic modal taps. A modal may disappear without
input and allow the expected state to be observed; otherwise the bounded wait
fails explicitly. Any future automatic dismissal requires a positively
identified, profile-owned modal subtype and a destination-safe action. Private
reports and frames are `/tmp/tower-rl-m1-10-homewait-retry-20260915.json` and
`/tmp/tower-rl-m1-4-home-no-modal-tap-20260915.json`, with corresponding
filtered logcat files. Both failed runs restored the unchanged 6 GiB offline
baseline.

## M1-E004 — Long-gate result-to-home timing failure

**Date:** 2026-09-15
**Status:** 10-attempt gate stopped after 3 valid episodes; bounded fix passed 2/2

The first 10-consecutive run using the corrected 600-second natural-death
deadline stopped after three valid episodes in 823.8 seconds. The explicit
failure was `result-to-home transition did not reach Battle home`, not a device
failure. The failure frame captured immediately after the controller exhausted
three five-second waits was already a valid Battle home frame. Tower remained
foreground, the filtered lifecycle log contained no crash, ANR, LMK, or process
death, and the runner restored the 6 GiB offline baseline. This proves a late
supported scene transition crossing the controller's deadline rather than
navigation to an unknown screen.

The existing bounded result-to-home attempts now allow 12 seconds each. A
two-episode production-path verification then passed 2/2 genuine WAIT-policy
natural deaths in 687.0 seconds and restored the same baseline with airplane
mode enabled and no route. Reports and lifecycle artifacts are private files at
`/tmp/tower-rl-m1-10-long-20260915.json` and
`/tmp/tower-rl-m1-2-home-wait-20260915.json`, with corresponding failure frame
and filtered logcat files. The required 100-consecutive gate was not started.

## M1-E003 — Wave 3 timeout diagnosis and death-layout correction

**Date:** 2026-09-15
**Status:** Runtime stall rejected; production correction passed one live episode

Bounded host-side traces against the 6 GiB workstation snapshot disproved the
working Wave 3 freeze hypothesis. During an offline sparse-capture control, the
screen remained a valid active run while wave advanced 1 → 2 → 3 → 4 → 5 → 6,
cash advanced 80 → 107, and visible health fell from approximately 98% to 29%.
The run reached the genuine Game Stats death panel after approximately 377
seconds. At Wave 3, Tower remained foreground and input-responsive, its process
CPU counters advanced, RSS was approximately 1.42 GiB, the Unity SurfaceView
and buffers remained present, and ActivityManager/logcat contained no Tower
LMK, OOM, ANR, AndroidRuntime fatal exception, or Unity crash. Because this
complete run occurred with airplane mode enabled and no route, no online or
alternate-renderer variant was needed.

The apparent stall had two controller/profile causes. The fixed 120-second
natural-death deadline expired while a healthy run was still progressing. In
addition, a first post-baseline death adds `New Highest Wave!`, moving the Game
Stats HOME-button border below the previously calibrated normal-result
position; that genuine result classified as `supported_modal`. The controller
now allows 600 seconds and samples at the configured one-second decision
interval. The visual profile recognizes both Game Stats button layouts while
retaining the existing generic-modal rejection.

A production-path one-episode WAIT-policy smoke then passed 1/1 in 351.3
seconds, recognized genuine death, and restored the unchanged 6 GiB golden
snapshot. The final probe was valid Battle home with Tower foreground,
airplane mode enabled, and no route. Private evidence is under
`/tmp/tower-rl-wave3-stall-diag-20260914/`,
`/tmp/tower-rl-wave3-passive-control-20260914/`,
`/tmp/tower-rl-wave3-passive-to-death-20260914/`, and
`/tmp/tower-rl-m1-wave3-fix-live-20260915.json`; logs and screenshots remain
outside the repository. No snapshot or renderer configuration changed, and the
100-consecutive M1 gate was not started.

## M1-E002 — Workstation 6 GiB profile repair and reliability failure

**Date:** 2026-09-14
**Status:** Profile/probe passed; M1 reliability gate failed

The workstation has 125 GiB host RAM, with approximately 93 GiB available at
the start of the repair. The exact AVD is `tower_rl_api36_play_x86_64` and its
configuration is under the user-local Android AVD directory. Before changing
it, the 2 GiB configuration and AVD pointer file were copied to the private
user-local Tower-RL state directory. The only AVD configuration change was
`hw.ramSize=2G` to `hw.ramSize=6144`; the running guest subsequently reported
6,072,056 kB total RAM. The previous snapshot was retained unchanged.

The AVD was stopped through the emulator console and cold-booted with the
pinned `lavapipe` renderer, which selected llvmpipe Vulkan and ANGLE/Swangle
GLES. The official Play-installed 29.0.3 app required ordinary connectivity to
pass its offline startup panel. No sign-in, credential entry, legal acceptance,
purchase, or advertisement was automated. After reaching Battle home,
connectivity was disabled again; airplane mode was enabled and the guest route
table was empty.

The visible baseline remained Tier 1, highest wave 2, 55 coins, 0 gems, x1.00
total coin bonus, and Labs locked. Snapshot
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_6gb_offline_home_20260914_workstation`
was saved separately from the 2 GiB snapshot. It occupies approximately 6.0
GiB locally. The profile probe and bounded Home → active run → result → restored
Home probe passed offline.

A four-episode natural-death diagnostic passed 4/4 in 164.6 seconds with no
failure and restored the new baseline. The subsequent fresh 100-consecutive
gate failed after seven valid episodes with an explicit `DeviceFailureError`;
the failure frame showed the Pixel launcher and the runner restored the new
baseline. The report is
`/tmp/tower-rl-m1-100-6gb-20260914.json` and remains outside the repository.

Follow-up bounded diagnostics did not reproduce the original 2 GiB LMK kill.
Instead, the game repeatedly remained alive and foreground on a visually frozen
Wave 3 active-run frame until the 120-second natural-death deadline. At these
failures Tower RSS was approximately 1.35–2.28 GiB, guest available memory was
approximately 2.15–3.12 GiB, and logs contained no Tower LMK kill, OOM, ANR,
AndroidRuntime fatal exception, or Unity crash. The stall reproduced both from
a freshly launched 6 GiB snapshot and after a cold launch without loading a
snapshot. Reducing death-poll capture frequency from 250 ms to the configured
one-second interval also reproduced the stall, so no controller change was
made from that rejected hypothesis.

The 6 GiB profile therefore fixes the proven guest-capacity deficiency but is
not reliability-qualified. M1 remains open: the complete gate has 7 valid
episodes followed by one `DEVICE_FAILURE`, and the subsequent diagnostic
taxonomy is an active-run stall/timeout. All failure artifacts and lifecycle
logs are private `/tmp` files, and the emulator is left at the new offline
baseline.

## M1-E001 — Workstation actor control slice

**Date:** 2026-09-14
**Status:** In progress; M1 gate remains open

The first real-game controller slice was exercised against the Play-installed
29.0.3 package on the fixed `lavapipe`/ANGLE software-rendered profile. It
restored the golden Tier-1 home snapshot, started a run, extracted wave/cash/
health plus the visible upgrade cards, and recognized `WAIT` and all five
supported purchase actions (`BUY_HEALTH`, `BUY_DAMAGE`, `BUY_ATTACK_SPEED`,
`BUY_CRITICAL_CHANCE`, and `BUY_CRITICAL_FACTOR`). A real critical-chance tap
was confirmed by the observed level/cash transition. Snapshot recovery returned
the emulator to the golden home baseline after the probe.

The implementation is fail-closed on unknown frames and records frame-backed
readings. OCR uses bounded retries because Unity text recognition was
intermittent under the software renderer; the retry path is intentionally local
to visible numeric fields. Focused checks pass: ruff, mypy, and 11 unit tests.

The required M1 evidence is not complete: normal death/result reset and 100
consecutive valid scripted episodes still need to be run and recorded. No
permanent game state was intentionally changed; the workstation AVD is left at
the golden snapshot.

The repeatable gate runner is `scripts/m1_reliability.py`; it checks the complete
purchase action mask, follows a deterministic WAIT policy to genuine game
death, starts the next run through the result/home path, and restores the golden
snapshot in its finalizer. Its output is an explicitly supplied local path and
must remain outside the public repository.

The first 100-attempt run stopped after one valid episode because the result
screen exposed a supported wave-information modal during the result wait. After
that handling was added, a three-attempt run stopped after one valid episode
because the same transient modal appeared during the Home wait. These are
classified UI states, not silent failures; the controller now dismisses the
profile-known modal while waiting for either RESULT or HOME. The 100-episode
gate must be rerun after this fix.

A subsequent gate attempt also encountered a transient `adb: device ... not
found` during the first episode's menu tap. The runner recovered the snapshot;
the ADB adapter now retries bounded transport errors (`not found`, `offline`,
`closed`, or `no devices`) before classifying the attempt as failed. A clean
two-episode run passed after this change, and the 100-episode gate is running
again.

The next three-episode run completed two episodes, then failed when the third
end-run confirmation remained in Active Run. End-run now retries the complete
profile-owned menu/confirmation sequence up to three times and still requires a
classified RESULT frame. The short reliability run should be repeated before
starting the 100-episode gate again.

The reliability runner now also settles for bounded intervals after starting
and resetting runs, so rapid Unity scene transitions are not mistaken for a
ready control surface.

The settle-adjusted three-episode run still completed two valid episodes, then
failed to reach RESULT on the third end-run attempt despite the bounded retry
sequence. This remains an open M1 reliability issue; the actor is not yet
approved for the 100-episode gate. The emulator was restored to the golden
baseline after the failure.

After extending the RESULT wait and adding episode-boundary settling, a fresh
three-episode run passed: all three episodes had the complete action mask,
reached RESULT, reset through the result/home path, and completed with the
golden baseline restored. This is a short reliability check; it does not yet
replace the required 100-episode M1 gate.

The captured result screenshot confirms that the return control is the stable
`HOME` button at the pinned lower-right result-panel location. The controller
now lets the result panel settle for two seconds before retrying that control;
the next short-gate run must verify this timing adjustment.

The diagnostic failure frame exposed the underlying false boundary: a generic
Attack Speed information modal had matched the old broad RESULT heuristic. The
classifier now requires the distinctive Game Stats HOME-button border for
RESULT and classifies other overlays as MODAL. The reliability runner now uses
the deterministic WAIT policy until natural death instead of forcing End Round.
A two-episode natural-death run passed with both valid episodes and the golden
baseline restored. The full 100-episode gate remains outstanding.

The probe restore path was hardened to wait for a valid Battle-home frame after
snapshot load rather than relying on a fixed two-second delay. Final restore
verification passed with airplane mode enabled, no route, foreground The Tower,
and screen `battle_home_tier_1`.

## M0-E001 — Development-host and XAPK characterization

**Date:** 2026-09-14
**Status:** Superseded by M0-E002 after first-run consent
**Purpose:** Establish whether the supplied XAPK can be analyzed, installed, and
launched on the single-device development host.

### Host evidence

- macOS 26.6.2 on Apple M2 Pro (`arm64`)
- 10 physical/logical CPU cores reported
- 16 GiB system memory
- Apple hardware virtualization available
- approximately 88 GiB free before Android SDK installation
- Python, `uv`, and Java present
- Android SDK tools were initially absent

This is the single-device development host. It is not the later 28 GB/RTX 4090
training workstation, whose OS, CPU, virtualization, and storage remain to be
characterized.

### Package evidence

The ignored local archive was inspected without modifying or committing it.

- Archive SHA-256:
  `6496e4f07904723c190c9728e47e621da9bdb40d24b66e1800af1f85211bf0c0`
- Package: `com.TechTreeGames.TheTower`
- Version: `29.0.1` (`versionCode` 1178)
- Minimum SDK: 27
- Target SDK: 36
- Required graphics API: OpenGL ES 3.0
- Launch activity: `com.unity3d.player.UnityPlayerActivity`
- Native ABI supplied: `arm64-v8a`
- All four APKs verify under APK Signature Schemes v2 and v3 with one common
  signing certificate (`SHA-256 b6c646d31fc34415445c6901450fe0a6690d413f9ef67af5ca5a64ce4ae2ee52`)

The archive contains one base APK, its required ARM64 configuration split, a
`gpdeku` install-time feature split, and that feature's ARM64 configuration. All
four share the same package/version identity and form the tested installation
set. Exact safe metadata and per-file checksums are in
`docs/environment-profile.yaml`.

### Android setup and result

Installed:

- Android command-line tools 15.8 / build 15859902
- Android Emulator 37.1.11
- Platform Tools 37.0.1
- Android API 36 platform
- Google APIs API 36 ARM64 system image

Created isolated AVD `tower_rl_api36_arm64` from the Pixel 2 device definition.
The guest reported API 36 and ABI `arm64-v8a`.

The four APKs installed atomically with `adb install-multiple`. Android reported
the expected package, version, SDK range, and primary ABI. A cold launch completed
successfully in approximately 1.6 seconds according to Activity Manager and
reached the real game's first-run EULA/privacy screen. The first captured frame
was 1080×1920 portrait and showed Android's one-time immersive-mode notice above
the game.

### Safety and retained artifacts

- No APK or extracted game file was written into a tracked repository path.
- Temporary extracted APKs and the first screenshot remain outside the repository.
- The screenshot is not suitable as a public test fixture and is not committed.
- The local XAPK remains ignored by Git.

### Blocker and next validation

Accepting the game's EULA/privacy policy is a user-owned legal interaction. No
automation accepted it. The user later completed that step; initialization then
stalled as recorded in M0-E002.

M0 is not complete: the Tier-1 manual-start, fixed baseline, device profile, and
production-host parts of the gate remain open.

## M0-E002 — Google APIs image purchaser stall

**Date:** 2026-09-14
**Status:** Both anonymous emulator trials blocked at purchaser initialization
**Purpose:** Diagnose why the game stopped advancing after first-run consent.

### Observation

The game's visible breadcrumb panel showed successful Firebase initialization,
GDPR acceptance, anonymous PlayFab account creation, cloud-save load, session
creation, remote-settings load, and minimum-version check. The final breadcrumb
remained `Initializing Purchaser` for more than six minutes.

The guest network was connected and Android marked it `INTERNET`, `VALIDATED`,
and unrestricted. The app remained foreground and did not crash.

### Device evidence

The API 36 `google_apis` image contained:

- Google Play services and Google Services Framework;
- a minimal `com.android.vending` version 1.8 package;
- no service resolving
  `com.android.vending.billing.InAppBillingService.BIND`.

The app's log repeatedly reported BillingClient failures while
`Purchaser.Initialize` was active, including attempts to unbind a service that
had not registered. This matches the visible stall and rejects the plain Google
APIs image for The Tower 29.0.1.

### Corrective trial

The incompatible AVD was stopped without deleting its user data. A separate
`tower_rl_api36_play_arm64` AVD was created with the API 36 ARM64 Google Play
image. Its Play Store package resolves the required billing service. The same
four-split XAPK set installed and cold-launched successfully. The user accepted
the EULA/privacy screen on this replacement AVD.

After consent, the replacement also remained at purchaser initialization. Its
log reported `In-app billing API version 3 is not supported on this device`,
followed by billing-service disconnect/death messages. The Play Store updated
itself from version 45.3.21 to 53.0.27 during the trial, but a clean app restart
after that update produced the same result. No Google account was configured on
the AVD, and the game was installed by the ADB shell rather than acquired by that
account through Google Play.

The replacement initially fell back to software graphics because the Mac had
less than the emulator's requested free-memory threshold at launch. Renderer
performance is therefore still uncharacterized and cannot yet support a profile
selection claim.

### Next validation

The next smallest authorized compatibility test is user-owned Play Store
provisioning:

1. sign into the Play Store with a dedicated user-owned Google account;
2. ensure that account can legitimately acquire The Tower from its production
   listing, without making an in-app purchase;
3. relaunch the installed, correctly signed game and observe purchaser setup;
4. if it succeeds, confirm the home/tutorial flow and preserve a named
   post-consent state at a stable screen;
5. if it fails, test a lower supported Play Store API image and record the result.

Do not falsify the installer identity, bypass Play licensing, or automate account
credentials. Whether sign-in/entitlement resolves the error remains a hypothesis,
not a completed compatibility result.

## M0-E003 — Play entitlement and offline snapshot-resume trial

**Date:** 2026-09-14
**Status:** One offline resumed retry cycle passed; offline cold launch failed
**Purpose:** Determine whether a production Play installation resolves purchaser
initialization and whether a post-consent AVD snapshot can support a network-
isolated Tier-1 sample.

### Play installation result

The user signed into Google Play manually. No credentials were automated or
captured. The ADB-installed copy was removed, and the game was acquired and
installed from its production Play listing. Android reported:

- package `com.TechTreeGames.TheTower`;
- version `29.0.3` (`versionCode=1199`);
- installer `com.android.vending`;
- ARM64 base, configuration, `gpdeku`, and `gpdeku` ARM64 configuration splits.

This installation completed purchaser initialization and entered a real Tier-1
run. With no actions, the run ended at wave 2 and displayed the normal result
screen. This confirms that the earlier purchaser stall was specific to the
unentitled ADB-installed path on the tested profile, not a general inability of
the API 36 Play Store image to run the game.

### Recovery snapshots

At the stable wave-2 result screen, the following named emulator snapshots were
created in the local AVD storage, outside the repository:

- `tower_post_consent_play_29_0_3_online_20260914`;
- `tower_post_consent_play_29_0_3_offline_running_20260914`.

The second snapshot was created only after Android airplane mode was enabled,
Wi-Fi and mobile data were disabled, and an external ping failed with `Network is
unreachable`. The running Unity process and result screen survived the transition.

### Offline behavior

A force-stop followed by an offline launcher start did **not** reopen the game.
Google Play displayed its app/licensing panel and the game process exited. A
durable offline cold-start path is therefore not established.

Restoring the online snapshot and immediately disabling connectivity preserved
the already-running game. From that state, a diagnostic tap on `Retry` dismissed
the result panel, ran another short Tier-1 attempt, and returned to the result
panel while airplane mode remained enabled and no route was available. The
untouched offline-running snapshot was restored after the trial.

### Interpretation and remaining gate

This is evidence for exactly one **snapshot-resumed, network-isolated** retry
cycle. It does not establish that cached Play entitlement or sessions survive
long delays, that a force-stopped actor recovers offline, that cloned actors have
independent identities or random streams, or that parallel execution is safe and
reliable. The Play-installed 29.0.3 package is now the runtime candidate; the
locally supplied 29.0.1 XAPK remains characterization input and is not the
candidate runtime package.

M0 remains open pending a fixed baseline, home-to-Tier-1 navigation inventory,
renderer validation, offline snapshot aging/recovery tests, and a controlled
two-actor isolation experiment.

## M0-E004 — Initial Tier-1 golden-baseline candidate

**Date:** 2026-09-14
**Status:** Snapshot created and transport-restored; visual restore gate pending
**Purpose:** Establish the first fixed, no-permanent-spending Tier-1 baseline and
test named-snapshot recovery.

### Baseline establishment

Starting from the offline-running post-consent snapshot, the result screen was
navigated to Battle home. The first-run tutorial required opening Workshop and
claiming an unavoidable 50-coin onboarding grant. The grant increased the visible
unspent balance from 3 to 53 coins. No coins were spent and no permanent combat
upgrade was purchased.

The visible fixed state was inventoried as:

- Tier 1 selected, highest wave 2, total coin bonus `x1.00`;
- 53 unspent coins and 0 gems;
- Workshop attack values: damage 3, attack speed 1.00, critical chance 1.00%,
  critical factor x1.20, with range upgrades locked;
- Workshop defense values: health 5 and health regeneration 0.00/sec, with
  additional defense upgrades locked;
- utility cash bonuses locked;
- Ultimate Weapons informational screen observed, with no weapon selected or
  purchased;
- all post-Workshop progression tabs visibly locked, so no Lab research can be
  active.

The device profile was 1080x1920, 420 dpi, portrait rotation 0, `en-US`, and
approximately 60 Hz. Android airplane mode remained enabled and no external
route was available. SHA-256 hashes of the four Play-installed 29.0.3 APK splits
were recorded in `environment-profile.yaml`; no package bytes were copied into
the repository.

### Snapshot and restore result

The named local snapshot
`tower_golden_t1_v1_play_29_0_3_offline_running_20260914` was created at the
Battle home screen with Tier 1 selected. The emulator reported an approximate
snapshot size of 1.5 GiB. It is stored only in ignored AVD data.

Loading the snapshot restored the game process in the foreground, preserved
airplane mode, and preserved the absence of an external route. The ADB transport
then became unstable when a post-restore screenshot was requested. Reconnecting
ADB restored command access, but another screenshot request reproduced the
disconnect. Earlier captures in the same session succeeded before this restore.

### Interpretation

This artifact is a **golden-baseline candidate**, not yet an admitted M1 recovery
baseline. Snapshot load and nonvisual invariants passed, but the required visual
fingerprint verification did not. The current software-rendered emulator profile
also emitted gfxstream/color-buffer errors and had already shown transient ADB
disconnects. Renderer/capture stability must be fixed or a different validated
graphics profile selected before this snapshot can be promoted to the trusted
golden baseline.

## M0-E005 — Pinned-renderer golden snapshot validation

**Date:** 2026-09-14
**Status:** Passed for the single-device M0 recovery check
**Purpose:** Rebuild the golden snapshot under a pinned renderer and verify that
the visible baseline survives an in-place snapshot restore.

### Procedure and evidence

The AVD was restarted with `-gpu lavapipe`, which resolves to Lavapipe Vulkan and
ANGLE/SwiftShader (`swangle`) GLES. The earlier 1.5-GiB snapshot was not
compatible with alternate renderer combinations; it was retained unchanged. With
connectivity temporarily enabled, the Play-installed 29.0.3 game was relaunched
and reached the same Battle home state. Connectivity was then disabled with
airplane mode, Wi-Fi, and mobile data disabled; an external ping returned
`Network is unreachable`.

A replacement snapshot,
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914`, was
created at Battle home. Its current local AVD footprint is approximately 2.9 GiB
(`ram.bin` plus renderer textures); this account-bearing state is not portable
repository data.

To test restore, the device was navigated away from the baseline and the named
snapshot was loaded in place. After restore:

- the game process was alive and foreground;
- airplane mode remained enabled;
- external network remained unreachable;
- `adb exec-out screencap -p` succeeded without the earlier pull/transport
  failure;
- the restored frame visibly showed Battle home, Tier 1 selected, highest wave 2,
  53 unspent coins, 0 gems, and `x1.00` total coin bonus;
- the representative restored-frame SHA-256 was
  `32ffeb8f5dd1a2d5b5da03b0e694059debe0f49c4ae46162ad2b9e2f44084a25`.

The screenshot is temporary evidence only and is not stored in the repository.

### Interpretation

The pinned-renderer snapshot passes the current single-device golden-baseline
restore gate. It is the canonical `tower-t1-initial-v1` artifact for subsequent
controller work. It remains a local account-bearing AVD snapshot and must not be
copied into Git or treated as server-state rewind. Offline cold launch after a
force-stop remains unsupported; recovery must resume the running snapshot or
temporarily restore connectivity. Parallel identity/randomness isolation and the
M1/M2 reliability gates remain outstanding.

## M0-E006 — Bounded semantic navigation probe

**Date:** 2026-09-14
**Status:** Passed for the single-device smoke path
**Purpose:** Verify that the pinned visual profile can drive the game's safe
Home → Tier 1 → result → Home path without selecting an upgrade or meta action.

The new `tower-rl probe` command captures PNG frames with `adb exec-out`, checks
the foreground package, airplane mode, external route, fixed 1080×1920 frame
size, and a conservative renderer-specific screen profile. It recognizes Home,
active run, the transient Wave Info modal, result, and unknown states. Unknown
or contradictory states fail closed.

The bounded sequence used only the calibrated controller operations: tap Battle,
open the in-run menu, select End Round, confirm, close a possible Wave Info
modal, and tap Home. It completed successfully while offline. The command then
restored the canonical golden snapshot, and a follow-up probe returned the exact
golden representative frame SHA-256
`32ffeb8f5dd1a2d5b5da03b0e694059debe0f49c4ae46162ad2b9e2f44084a25` with the
game foreground and no external route.

Starting a run can award ordinary run coins before End Round is processed, so
navigation probes must always use `--restore-snapshot` when run against the
account-bearing golden device. The probe exposes no purchase or permanent-
progression action. This is a smoke/navigation gate, not the M2 reliability
gate; repeated episodes, death detection, full observation extraction, and
two-actor isolation remain outstanding.

## M0-E007 — Workstation artifact portability assessment

**Date:** 2026-09-14
**Status:** Handoff documented; cross-host transfer intentionally not attempted
**Purpose:** Make the validated Mac setup reproducible on the future RTX 4090
workstation without publishing account-bearing state.

The canonical snapshot resides in the Mac-local AVD snapshot directory under
`$ANDROID_AVD_HOME` (or `$HOME/.android/avd`), in the named snapshot directory
recorded in `docs/workstation-handoff.md`. Its current on-disk footprint is
approximately 2.9 GiB, including `ram.bin`, renderer textures, hardware metadata,
and the snapshot manifest. This is Android user/account state and remains outside
Git.

The workstation ABI, host OS, emulator version, and renderer are not yet known.
Because the current artifact is ARM64 and renderer-pinned to Lavapipe/Swangle,
copying it to an x86_64 RTX 4090 host would not be a supported bootstrap path.
The handoff therefore specifies Play reprovisioning, manual baseline setup, a
new workstation-local snapshot, and probe validation before actor work. No
proprietary package bytes, credentials, screenshots, or emulator data were added
to the repository.

## M0-E008 — Workstation-independent environment contract scaffold

**Date:** 2026-09-14
**Status:** Unit and contract checks passed; real-device extraction remains open
**Purpose:** Make the Android-independent portion of the environment ready for
workstation integration.

Added the versioned `observation-v1` and `run-action-v1` schemas, typed action
outcomes and termination reasons, fail-closed temporal/logical observation
validation, and the `AndroidDevice` protocol boundary. Added a sanitized local
configuration example and contract tests. The policy action set remains semantic
(`WAIT`, `BUY_HEALTH`, and observed attack upgrades); coordinates and navigation
are confined to the controller/device layer.

This scaffold deliberately does not claim M1: there is no full OCR/structured
extractor, purchase confirmation, death detector, or 100-episode reliability
run yet. Those components can now be integrated on the workstation without
changing the policy-facing contract.

## M0-E009 — Repository workstation bootstrap helpers

**Date:** 2026-09-14
**Status:** Read-only preflight and idempotent AVD helpers validated on the Mac
**Purpose:** Move repeatable host setup into the repository while keeping
account-bearing actions manual.

Added `scripts/workstation_preflight.py` for host/SDK/tool/AVD inventory,
`scripts/create_avd.sh` for safe idempotent API-image AVD creation, and
`scripts/launch_avd.sh` for renderer/snapshot launch. Added
`configs/workstation.example.yaml` and documented the complete handoff flow.
The helpers do not install the game, sign into Play, accept legal consent, or
copy snapshots. The Mac preflight reports the expected ARM64/API 36 tools and
AVDs; all repository checks remain green.

## M0-E010 — Domain-driven package boundaries

**Superseded.** The `domain`/`ports`/`application`/`infrastructure` layering and the CLI composition root described here no longer exist; the packages are now `environment`, `simulation`, `learning` and `experiment` with `scripts/` as the composition root, the last move of which is device-verified in `M1B-E056`. The rule is enforced by `tests/unit/test_import_contracts.py` and described in `docs/architecture.md`.

**Date:** 2026-09-14
**Status:** Refactor verified; behavior preserved
**Purpose:** Keep game concepts independent from Android and process details as
the workstation integration grows.

The canonical domain model now lives under `tower_rl.domain`, ports under
`tower_rl.ports`, use-case orchestration under `tower_rl.application`, and ADB
adapters under `tower_rl.infrastructure`. The CLI composes these layers. Root
imports remain compatibility shims only. Unit, lint, type, and live baseline
probe checks pass after the refactor.

## M0-E011 — RTX workstation characterization

**Date:** 2026-09-14
**Status:** Host gate passed; Android provisioning remains open
**Purpose:** Characterize the training workstation before selecting an Android
backend or actor count.

### Observed host

- Ubuntu 26.04.1 LTS, kernel `7.0.0-31-generic`, x86_64;
- Intel Core i9-14900, 24 physical cores / 32 logical CPUs;
- 125 GiB RAM and approximately 1.1 TiB free on the workspace filesystem;
- NVIDIA RTX 4090, driver `595.91.07`, CUDA `13.2`;
- Intel VT-x is enabled; `/dev/kvm` is readable and writable by the current user
  through an explicit device ACL.

### Missing prerequisites

The preflight found no Android SDK root, `adb`, emulator, `sdkmanager`, or
`avdmanager`. No Android device or AVD is currently connected or discoverable.
The expected local XAPK is also absent from the repository's ignored `local/`
directory and the searched local paths, so package metadata cannot be freshly
verified on this host yet.

### Verification

`uv sync --all-groups`, `uv run ruff check .`, `uv run mypy`, and `uv run pytest`
all pass (11 tests). The read-only preflight was run through
`scripts/workstation_preflight.py --json`; the CLI doctor independently reports
the same missing Android tools and XAPK prerequisites.

### Next smallest action

Provision Android command-line tools/emulator and the API 36 Google Play x86_64
image, then place the authorized reference archive at the documented ignored
path (or provide its actual path). Recreate the Play-installed game baseline
manually on this host; the ARM64 Mac snapshot is not portable or supported here.
M0 remains open until one workstation AVD launches the real game, Tier 1 is
manually started, and a new local baseline snapshot passes the probe.

## M0-E012 — Workstation Android stack provisioning

**Date:** 2026-09-14
**Status:** Toolchain and AVD ready; Play provisioning requires user action
**Purpose:** Prepare a compatible Android execution target on the RTX
workstation without touching account state.

Installed under the user's local data directories, without sudo:

- Eclipse Temurin JDK 17.0.20.1;
- Android command-line tools 15.0;
- platform-tools 37.0.1, emulator 37.1.11, API 36 platform/build tools;
- API 36 `google_apis_playstore` x86_64 system image.

Created and booted `tower_rl_api36_play_x86_64` from the Pixel 2 definition.
The device reports x86_64, 1080×1920, 420 dpi, and `sys.boot_completed=1`.
The emulator selected the RTX 4090 through the host renderer; renderer and game
stability remain unvalidated.

At the time of this provisioning entry the Play Store was unauthenticated;
subsequent user-owned sign-in, official game installation/entitlement, legal
consent, and first-run setup are recorded in M0-E013.

## M0-E013 — Workstation renderer and golden-baseline restore

**Date:** 2026-09-14
**Status:** Passed for the workstation single-device M0 recovery check
**Purpose:** Validate a cleanly rendered real game and establish a separate
account-bearing baseline on the x86_64 workstation.

The initial `host` renderer produced visibly corrupted Unity text and UI. The
same AVD was relaunched with pinned `lavapipe`, which selected the llvmpipe
Vulkan device and ANGLE/Swangle GLES. The Play-installed The Tower 29.0.3
(`versionCode=1199`) then rendered cleanly at 1080×1920, 420 dpi, portrait.

At the stable Battle home screen, Tier 1 was selected, highest wave was 2,
unspent coins were 55, gems were 0, total coin bonus was x1.00, and Labs were
locked. Networking was disabled inside the guest; airplane mode was 1 and
`ip route` was empty. The repository probe passed with no reasons.

Snapshot `tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation`
was saved in the local AVD storage. Loading it back into the running emulator
passed: the game remained foreground, the probe passed again, and the restored
PNG SHA-256 exactly matched the pre-restore frame
(`f00ab790b75c44862baa39f84f2ebf0767d600e8af9a6939def1266816f1f643`). The
snapshot directory occupies approximately 2.9 GiB and is ignored local state.

This validates the workstation baseline/recovery artifact, not M1. The XAPK
bytes are still absent from this checkout; previously recorded compatibility
metadata remains in `environment-profile.yaml`, while fresh local XAPK
inspection is still optional reference verification.

## M0-E014 — Workstation bounded Tier-1 navigation probe

**Date:** 2026-09-14
**Status:** Passed; baseline restored
**Purpose:** Verify the safe workstation Home → Tier 1 → result → Home path
without buying upgrades or invoking permanent-progression controls.

`uv run tower-rl probe --serial emulator-5554 --navigate --restore-snapshot
tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation`
completed with valid initial Battle home, active Tier-1 run, result, and final
Battle-home observations. Airplane mode remained enabled and the external route
remained absent throughout. The command restored the workstation snapshot after
the probe; a follow-up probe returned the canonical frame hash
`f00ab790b75c44862baa39f84f2ebf0767d600e8af9a6939def1266816f1f643`.

This closes the workstation-specific M0 navigation/start smoke check. The
available in-run action inventory, speed profile, offline aging behavior, and
multi-actor isolation remain open before M1/M2 work.
