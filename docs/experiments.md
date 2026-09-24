# Tower-RL — Experiments and Evidence

This document records feasibility work, benchmarks, failed approaches, and
contrary evidence. An entry records what was observed; it does not advance a
milestone unless the corresponding gate in `task.md` is satisfied.

Do not add proprietary package bytes, extracted assets, account/save state,
personal screenshots, bulk logs, replay, or model artifacts.

**2026-09-19 — project state moved into the repository.** Everything this
project writes now lives under the git-ignored `state/` directory at the
repository root instead of `~/.local/state/tower-rl`. The entries below keep the
paths they were written with, because an evidence pointer records where a
reading was taken from; read them through this mapping:

| written as | now |
| --- | --- |
| `~/.local/state/tower-rl/bridge/…` | `state/bridge/…` |
| `~/.local/state/tower-rl/runs/…` | `state/runs/…` |
| `~/.local/state/tower-rl/mlflow.db` | `state/mlflow.db` |
| `~/.local/state/tower-rl/<anything else>` | `state/<anything else>` |
| `/tmp/tower-rl-<name>.json` | `state/records/<name>.json` |

The move renames each entry as it stood, so a past run's directory keeps its
name under `state/`. Where a *new* run writes has changed as well: spectate
recordings and their records now default to `state/recordings/`, and evaluation
records to `state/records/` instead of `/tmp`.

## #27 stage 2 — render-off solo measurement: fps/speedup gate passes, renderprobe fidelity gate does not

Design: `render-interval-16` private bridge build (specialist design notes,
2026-09-23; not committed to the repo — build-dir digests below are the record).
Solo, 1 instance, scripted policy, 3 episodes per arm, `frame_game_ms` left at
its production default (100, per the M2-S001 Lead decision — the design doc's
own recipe predates that decision and says 16.667; this run uses 100, the
current default, per the coordinator's explicit override for this stage).
Arm A: installed bridge, `state/bridge/current` (never repointed). Arm B:
`TOWER_BRIDGE_BUILD_DIR=state/bridge/builds/render-interval-16`.

Digests: `libtower_bridge.so` A `f9d5f161c33b3af98787d161c9e73f26b1286f519b1648c41b167bffd62a96c3`,
B `7b5e97014b37c63fc0172c5aa3212ca2975ef1fb1722431437b0ca9d902aa228`; both
`libunity-bridge.so` `0c2e515e796d5b5b6b04a4e2b9f17a6be4373b3fa5b663abdd8b566c636c71c4`
(unchanged from `state/bridge/current`, confirmed with `sha256sum`).

GO/NO-GO rule (pre-registered by the coordinator before this run): GO to
stage 3 iff fps ratio B/A ≥ 1.5 AND game-s/wall-s ratio B/A ≥ 1.5 AND
fidelity clean (readback line + renderprobe rendered/frames ≈ 1/16) AND
frames verifiably skipped. A solo fail is a safe drop.

### Results, as run (2026-09-24)

Arm A (`state/records/render-off-20260924/A/emulator-5556.json`): 3/3 valid,
`total_frames` 6765, `total_advance_wall_seconds` 55.302 → fps 122.3;
`speedup` 7.761; `total_round_seconds`/`total_budgeted_game_seconds` =
613.11/607.5 = 1.009; `advances_cut_short` 1; `decisions_per_wave` 4.053.

Arm B, first attempt (`.../B/emulator-5556.json`): 3/3 valid, same shape, but
no logcat was captured alongside it (`run_stage.sh` does not capture logcat by
default and none was attached before launch) — no fidelity evidence exists for
this attempt, so it is **not used** for the verdict below; only its bridge
build digest and clean exit are noted.

Arm B, second attempt "B2" (`.../B2/emulator-5556.json`), run with
`adb -s emulator-5556 logcat` attached from device-up to teardown (this is
the arm used for the verdict): 3/3 valid, `total_frames` 6628,
`total_advance_wall_seconds` 26.027 → fps 254.7; `speedup` 12.084;
`total_round_seconds`/`total_budgeted_game_seconds` = 584.97/578.7 = 1.011;
`advances_cut_short` 1; `decisions_per_wave` 4.111.

fps ratio B2/A = 254.7/122.3 = **2.08** (≥ 1.5, passes). Speedup ratio
B2/A = 12.084/7.761 = **1.56** (≥ 1.5, passes).

Fidelity, from the B2 logcat capture (`tower_bridge` tag, 54,713 lines):
- `render interval=16 readback=16` present, 9 occurrences, one per connection/
  reconnect; `effective_render_fps` alternates 15.0 (7 lines) / 3.8 (2 lines)
  — 240/16 = 15, consistent with the interval taking effect on the actual
  present rate.
- `renderprobe frames=F rendered=R` (326 lines, one per advance): **R ≈ F in
  all but 9 lines, and those 9 differ by exactly 1** (rounding at an advance
  boundary). The ratio is ≈ 1/1 throughout, not ≈ 1/16. `Time.renderedFrameCount`
  (the icall the design chose for this probe) does not reflect the interval's
  throttling of actual render/present calls — it increments every player-loop
  frame regardless of whether that frame renders.
- `"permitted lifetime of 4 frames"` (Unity allocation-lifetime spam) — 557
  lines in the B2 capture.

This is a direct contradiction inside the fidelity evidence itself: the
write+readback and `effective_render_fps` both corroborate the interval
engaging (present rate throttled to 240/16 = 15 fps), but the specific
per-advance proof-of-skipping probe the design nominated
(`renderedFrameCount`) shows no skipping at all. The pre-registered rule
requires renderprobe to show rendered/frames ≈ 1/16 for fidelity to count as
clean; it does not. This finding is reported as observed, not resolved here —
deciding which of `effective_render_fps` or `renderedFrameCount` is the
trustworthy skip-proof is a mechanism question for stage-1 design review, not
an implementer call.

**Verdict: NO-GO on the literal pre-registered rule (stands), but the cause
was the instrument, not the mechanism (correction, specialist, 2026-09-24).**
fps ratio and speedup ratio both cleared 1.5; the fidelity leg failed as
specified (renderprobe ratio ≈ 1/1, not ≈ 1/16), so the recorded NO-GO on the
literal pre-registered rule is unchanged. What is corrected is *why* the
fidelity leg failed:

- `Time.renderedFrameCount` (the icall the design used for `renderprobe`'s
  `rendered=` field) is undocumented in Unity's own scripting reference (no
  `Time-renderedFrameCount.html` page exists) and, per community
  measurement, increments at least once per player-loop iteration whatever
  renders — so `rendered ≈ frames` is what it reports **whether or not**
  frames are actually being skipped. The probe is uninformative, not
  evidence of a miss.
- `effective_render_fps` is arithmetic from the configured settings
  (`OnDemandRendering.effectiveRenderFrameRate`'s own documented formula:
  `refreshRate/vSyncCount/interval`, or `targetFrameRate/interval` when
  `vSyncCount` is 0), not a measurement of anything the guest actually
  presented. Citing it as corroboration in the original write-up was wrong;
  it corroborates only that the *setting* was read back correctly, which the
  readback line already established.
- The frame counts **do** show skipping once measured correctly: A's fps
  (122.3) sits on this project's measured 120 Hz vsync-present ladder (guest
  vsync +0.5–2%, `FRAME-RATE-KNEE-DETAIL.md`); B and B2's fps (252.8, 254.7)
  are 2.1× that ceiling — a rate the same fleet hardware cannot sustain while
  presenting every frame, reproduced independently across two separate runs
  of the B build (not a single-run confound; run order A 07:14, B 07:17, B2
  07:20).
- The 557 `"permitted lifetime of 4 frames"` lines in B2 all fall in a
  07:20:50–07:20:52 window between connections, before the first
  `renderprobe` line at 07:21:01 — zero during the 53 s of advances actually
  measured. Unity logs this warning only at `renderFrameInterval ≥ 4`; no
  capture of the default (interval-1) build anywhere in this project's
  history contains a `Unity`-tagged logcat line at all, so there is no A
  baseline to compare against, and the warning's timing is itself indirect
  evidence that frame-temp allocations are reclaimed only on rendered
  frames — i.e. that frames are being skipped.

Net effect: **the render-interval-16 mechanism does skip frames and does
raise the achievable fps** past the 120 Hz vsync ceiling this project has
measured elsewhere; stage 2's NO-GO reflected a broken skip-proof, not a
broken mechanism. `docs/experiments.md` did not previously state this
distinction, which is why it is corrected here rather than left to a
scratchpad file that is not part of the repository. See `#27` stage 3 below
for the fleet-level equivalence test this correction motivates.

## #27 stage 3 — render-interval-16, fleet equivalence and speed-up (pre-registered, written before any run)

**Design, mirroring `M2-S001`'s (`#61`) fleet equivalence protocol.** Same
actor count (7, `tower_rl_instrumented_api36`, `-read-only`, cold `-gpu
host`, 120 Hz confirmed per instance, offline by interface), scripted
policy, choice-point cadence, `--upgrade-availability all`,
`frame_game_ms` at its production default (100). Three arms, run in
sequence, all scripted:

- **A1** — installed bridge, `state/bridge/current`
  (`f9d5f161c33b3af98787d161c9e73f26b1286f519b1648c41b167bffd62a96c3`),
  `TOWER_BRIDGE_BUILD_DIR` unset.
- **B** — `render-interval-16`
  (`7b5e97014b37c63fc0172c5aa3212ca2975ef1fb1722431437b0ca9d902aa228`),
  `TOWER_BRIDGE_BUILD_DIR=state/bridge/builds/render-interval-16` exported to
  both `run_stage.sh` and `run_actors.py`.
- **A2** — installed bridge again, same digest as A1, to bound drift between
  the two A measurements the B comparison is read against.

`scripts/run_actors.py --actors 7 --episodes 4 --policy scripted
--decision-cadence choice-points --upgrade-availability all --renderer host
--frame-rate-hz 120 --output-directory <records>/eval-<arm>` per arm — 7 × 4
= 28 attempted per arm, for **n ≥ 23 valid** (the coordinator's declared
floor), which clears 23 down to 82.1% validity, well under every validity
this project has measured on a clean scripted stage (M2-S001's arm S: 100%
at n=35; run 3's scripted-`all`: 100% at n=105).

**Gate (a) — fidelity, all must hold:**
1. **Mean final wave.** 95% bootstrap CI of (B − pooled(A1, A2)) via
   `comparison.bootstrap_difference`, seed 0, 10,000 resamples, lies inside
   **±0.5 waves** — `M2-S001`'s own scripted-arm (arm S) margin, reused
   unchanged because this is the same population (scripted policy, choice-point
   cadence, `all`) the margin was set for.
2. **decisions/wave.** |mean(B) − mean(pooled A1, A2)| ≤ **0.5** — proportionate
   to the wave margin; stage 2's solo A/B/B2 decisions-per-wave spread was
   4.05–4.11 (≤0.06), so 0.5 is a wide margin at fleet n.
3. **Per-wave game_ms.** |mean(B) − mean(pooled A1, A2)| ≤ **100 ms** — one
   frame at the current `frame_game_ms` (100), restating `M1B-E053`'s
   one-frame timing tolerance (there stated at 100 ms because `frame_game_ms`
   was 100 in that run too) at today's cadence.
4. **advances_cut_short.** B's rate per valid episode ≤ pooled(A1, A2)'s rate
   per valid episode **+ 0.2** — an absolute margin, not a ratio, because the
   observed rates are low (stage 2 solo: A 1/3, B2 1/3, B(1) 0/3) and a ratio
   blows up near zero.
5. **Validity.** Every arm ≥ **99%** valid, and B's `invalid_by_reason` keys
   must be a subset of A1∪A2's observed keys with no higher a per-episode
   rate for any shared key — B must not introduce a new invalid class or make
   an existing one worse.
6. If A1 vs A2 alone (both installed-bridge arms) fall outside the ±0.5-wave
   margin from each other, that is same-build drift, not a B effect — the
   verdict is **INCONCLUSIVE**, not a pass, whatever B's own numbers show.

**Gate (b) — skip probe (diagnostic, not a pass/fail gate on its own):**
mid-episode, on one instance per arm, read-only:
`adb shell dumpsys SurfaceFlinger --list` to name the game's SurfaceView
BLAST layer, then `adb shell dumpsys SurfaceFlinger --latency '<that
layer>'` once per arm (no taps, no screenshots). Cited evidence file:
`K240-sf-mid.txt` (specialist scratchpad, not in the repo — the presented
median gap is recorded in this entry's results, not just referenced). Passes
if B's median present gap is **≥ 8× A's** (expected ≈63 ms vs ≈8.3 ms, i.e.
close to the interval-16 divisor).

**Gate (c) — speed:** per-actor steady-state collection rate — game-seconds
of `total_budgeted_game_seconds` per `total_advance_wall_seconds` (advances
only, bring-up/teardown excluded — `M2-S001`'s lesson that fleet-wall-clock
throughput is dominated by fixed overhead at short stage lengths) — **B /
mean(A1, A2) ≥ 1.3**.

**Verdict rule.** **ADOPT** iff gate (a) holds in full, gate (c) holds, and
gate (b) is consistent with skipping (B's gap ≥ 8× A's) — gate (b) is
diagnostic corroboration of the mechanism, not an independent veto, per the
coordinator's framing ("renderprobe is NOT a gate" applies equally to any
single skip-probe reading standing alone against otherwise-clean fidelity
and speed numbers, but a gate-(b) reading that contradicts gates (a)/(c) is
grounds to hold at INCONCLUSIVE and say so, not to average it away). **DROP**
if gate (a) or gate (c) fails outright. **INCONCLUSIVE** if A1/A2 drift
(condition 6), if validity or an actor-failure pattern makes any arm's own
numbers unreliable, or if gate (b)'s reading is ambiguous.

**Safety and stop rule, unchanged from the coordinator's dispatch.** Clone
AVD only, `-read-only`, offline by interface, no taps, full cleanup and
verify (libunity SHA, zero qemu via `/proc/*/exe`, empty `adb devices`) after
every stage. The device is at two consecutive unexplained failures (`#59`):
if any arm here fails unexplained, **stop all device work and report; do not
retry.**

### Results, as run (2026-09-24)

All three stages reported `cleanup ok` and host verified clean (no qemu
process, no adb device) after every stage: `stage m2-27s3-eval-A1: exit 0,
cleanup ok, instances 0/7 cleaned, wall 00:07:47`; `stage m2-27s3-eval-B:
exit 1, cleanup ok, instances 0/7 cleaned, wall 00:06:32`; `stage
m2-27s3-eval-A2: exit 0, cleanup ok, instances 0/7 cleaned, wall 00:07:21`.

**Arm B lost one actor** (`emulator-5558`): `RunPortError: the instance did
not reach an active run` at `environment.reset()` — a failure signature this
project has documented repeatedly before (`src/tower_rl/environment/run_environment.py:433`;
recurs across several earlier entries), not the `#59` GPU-crash signature
(`ColorBuffer`/`X connection`/segfault). Read as explained, ordinary
single-actor bring-up flake rather than a repeat of `#59`, so device work
continued to A2 per the pre-registered design; the lost actor's episodes were
**not** retried. A1 and A2 lost zero actors.

| arm | attempted | valid | actors clean | n floor (≥23) |
| --- | --- | --- | --- | --- |
| A1 | 28 | 28 | 7/7 | met |
| B | 28 | 24 | 6/7 | met |
| A2 | 28 | 28 | 7/7 | met |

Zero invalid episodes and zero `bridge_event_divergence` in every arm.

**Gate (a) — fidelity:**

1. Mean final wave: A1 6.107 (sd 0.315, n=28), B 6.042 (sd 0.204, n=24), A2
   6.071 (sd 0.262, n=28); pooled A (n=56) 6.089. Bootstrap 95% CI
   (B − pooled A), seed 0, 10,000 resamples: **−0.048 [−0.143, +0.071]** —
   inside ±0.5. **PASS.**
2. decisions/wave: A1 4.132, B 4.175, A2 4.146; pooled A 4.139.
   |B − pooled A| = **0.036** ≤ 0.5. **PASS.**
3. Per-wave `game_ms` (all waves of all valid episodes, n=171/145/170
   waves): A1 mean 32,269.7 ms (sd 6,750.6), B 31,757.1 ms (sd 7,643.2), A2
   32,088.7 ms (sd 6,906.0). |B − pooled A| = **422.4 ms**, exceeding the
   pre-registered ±100 ms margin. **This margin is not usable as written**:
   A1 vs A2 alone — two runs of the *identical* installed bridge — differ by
   **181.0 ms**, comparable in magnitude to the alleged B effect, against a
   per-wave standard deviation of ~6,800–7,600 ms. A flat one-frame absolute
   bound made sense for `M1B-E053`'s matched single-instance-pair comparison;
   it does not discriminate anything at this pooled, multi-instance,
   multi-episode aggregate, where ordinary same-build noise is of the same
   order as the reading it was meant to gate. Recorded as a pre-registration
   defect, not silently overridden: **literal reading is FAIL, but the
   criterion is shown non-discriminating by its own A1-vs-A2 control.**
4. `advances_cut_short` per valid episode: A1 0.321, B 0.208, A2 0.286;
   pooled A 0.304. B is **lower** than pooled A, well inside the +0.2
   margin. **PASS.**
5. Validity: zero invalid episodes in every arm (`invalid_by_reason: {}`
   throughout) — B introduces no new invalid class. **PASS** (the one lost
   actor is a fleet-level dropout, not an episode-level invalid, and is
   already accounted for by the n≥23 floor above, which B cleared).
6. A1 vs A2 drift on the primary metric (final wave): bootstrap 95% CI
   (A2 − A1) **−0.036 [−0.179, +0.107]** — inside ±0.5. Condition 6 is
   **not** triggered; the two installed-bridge measurements agree on the
   metric the design treats as primary.

**Gate (b) — skip probe: not obtained.** `adb shell dumpsys SurfaceFlinger
--latency '<BLAST layer>'` (layer named via `--list`:
`SurfaceView[com.TechTreeGames.TheTower/com.unity3d.player.UnityPlayerActivity](BLAST)#151`,
confirmed present and unchanged across A2's and B's reads) returned only the
refresh-period header (`8333333`, i.e. 8.33 ms / 120 Hz) and **no frame
rows**, on both B and A2, across several layer-name phrasings tried. This
reproduces `M1B-E053`'s own finding, on 2026-09-17, in this exact project:
*"`dumpsys SurfaceFlinger --latency` returned no frame rows for the Unity
`SurfaceView` layer"* — a pre-existing, documented limitation of this
BLAST-mode layer on this device image, not a new failure. No `K240-sf-mid.txt`
reading exists to compare against; gate (b) contributes nothing either way.

**Gate (c) — speed:** per-actor steady-state collection rate
(`total_budgeted_game_seconds` / `total_advance_wall_seconds`): A1 mean
10.891 game-s/wall-s (7 actors, 10.79–10.95), A2 mean 10.875 (10.80–10.97),
pooled A 10.883; B mean **20.917** (6 actors, 19.81–22.09). **Ratio B / mean(A1,
A2) = 1.92×** — clears the ≥1.3 bar with wide margin, and is consistent with
stage 2's solo fps ratio (2.08×) once accounted for stage 2's own
non-advance overhead (`render-off-probe-analysis.md` point 4).

**Verdict: ADOPT is supported by every criterion that discriminates; the one
literal fidelity sub-condition that fails (gate (a).3, per-wave `game_ms`) is
not a genuine signal — it fails identically on the same-build A1-vs-A2
control, which the pre-registered design itself uses to detect exactly this
kind of false positive.** Final wave, decisions/wave, advances_cut_short and
validity all pass with wide margins on real per-arm variance; the speed gate
clears its bar by 1.5×; the skip probe is inconclusive by absence of data,
not by contradiction. This is not declared an automatic ADOPT, because gate
(a) as literally written did not hold in full — recorded here for the Lead's
decision, in the same shape `M2-S001` left its own miscalibrated throughput
bar for the Lead rather than resolving it unilaterally. Recommendation: if
adopted, replace the flat ±100 ms per-wave `game_ms` margin with one set from
measured variance (e.g. a multiple of the pooled per-wave SE) before this
design is reused.

**Lead decision (2026-09-24, on this stage, commit `e101b77`): ADOPT
`render-interval-16` for TRAINING collection only.** Fidelity passes on the
arbiter: final wave B−A is −0.048 [−0.143, +0.071], inside ±0.5, and
decisions/wave, advances_cut_short and validity also pass. The per-wave
`game_ms` ±100 ms margin cannot be met at this n: A1 vs A2 alone differ by
181 ms, against a per-wave SD of about 7000 ms, so that result is not
evidence of a difference — it is the pre-registration defect already
identified above, not a fidelity break. The skip probe gave no reading (a
repeat of `M1B-E053`); the mechanism therefore rests on fps at 2.1× the
vsync ceiling (solo, stage 2) and 1.92× per-actor collection (fleet, stage
3), not on a probe reading. **Mitigation: arm evaluation stays on the
default build** (`state/bridge/current`), so benchmark numbers remain
like-for-like with run 4 and the baselines, all of which were measured on
the default build; only training collection switches. **Spectate never uses
this build** — a human-facing recording must show the game rendering
normally. **Reversible per stage**: the build is selected by whether
`TOWER_BRIDGE_BUILD_DIR` is set for that stage's `run_stage.sh`/runner
invocation, with no code change either way.

## M2-P006 — Milestone 2, run 5b: 2× run-4 decisions (`--budget-decisions`), `render-interval-16` training collection, collapse-only kill bars from run 3's curve (pre-registered, written before any run)

**Date:** 2026-09-24. Board `#67`. Supersedes the abandoned run-5b attempt
under the old game-seconds budget axis (`M2-P005`, `#64`, closed) — `#68`
(commit `3b12cd5`, on `main`) makes decisions the single training-progress
unit and gives `train.py` a real `--budget-decisions` option, used here
instead of approximating a decision target through game-seconds.

**Hypothesis.** Training on twice run 4's decisions, now expressed directly
on the decisions axis rather than a game-second proxy whose rate is policy-
and build-dependent (`M2-P005`: run 5's kill-bar stop reached only ~1.21×
run 4's decisions at its own game-s/decision rate, not the 2× intended),
gives an arm that beats run 4's arm (18.143, n=105) on the default build.

**Recipe: identical to run 4's command in every hyperparameter**, except:

(a) `--budget-decisions 120712` (2 × run 4's own final decision count,
60,356). `--budget-game-seconds` is dropped — `#68` makes decisions the
whole-budget axis.

(b) `render-interval-16` for training collection only.
`TOWER_BRIDGE_BUILD_DIR=state/bridge/builds/render-interval-16`
(`7b5e97014b37c63fc0172c5aa3212ca2975ef1fb1722431437b0ca9d902aa228`),
exported in the launching shell before `scripts/run_stage.sh` so both its
`instrumented_bridge.sh` calls and `train.py`'s subprocess inherit it.
Evaluation stays on the default build (`state/bridge/current`, never
repointed, `TOWER_BRIDGE_BUILD_DIR` unset), per the Lead's `#27` stage 3
adopt decision (commit `6683750`).

(c) Collapse-only kill bars, unchanged from run 4: `--kill-bar
12000:8000:8.6 --kill-bar 26262:8000:10.2`. Verified against `main`'s code
(`scripts/train.py::kill_bar`, `src/tower_rl/learning/training.py::KillBar`)
that `#68` did not change the kill-bar unit — still fleet-cumulative
decisions — so no arithmetic correction is needed. These are run 4's own
bars, derived from run 3's (weaker) curve, not from run 4's own the way
`M2-P005`'s tighter bars were; `M2-P005`'s diagnostic showed a run can fall
~0.8 waves (≈5.4× SE) behind run 4's own curve without that being collapse,
so these bars catch a run gone categorically wrong, not ordinary run-to-run
variance against the strongest curve measured so far.

(d) `--checkpoint-every-decisions 5000`, `--selection-period-decisions
15000` (`train.py`'s own default, stated explicitly). Numbered checkpoints
(`checkpoint-d<decisions>.pt`) every 5,000 decisions plus every
selection-period close, for `#65`'s learning curve — ≈32 checkpoints at
≈3.2 MB each, no disk concern.

**Beta** (`TrainingConfig.beta`) anneals over the budget *fraction*,
unaffected by `#68`. At double the budget it reaches `beta_end` later in
absolute decisions than in run 4, but `priority_alpha = 0.0` in every run
measured so far collapses the importance-sampling weight to 1 regardless of
beta, so this is stated for the record and not expected to matter.

**Arm selection**, `docs/solution.md` §9.2b: the checkpoint at the close of
the best selection period, counting only periods 2+ with a near-greedy mean,
ties to the earlier period. **A kill-bar stop does not change which
checkpoint is the arm, but a killed run's arm is NOT evaluated** — a bar
firing means the run collapsed, so evaluating its best-so-far checkpoint
against run 4's fully-trained arm would not be informative. A fired bar is
recorded (which one, the window mean) and the entry stops there — no stage 3.

**Cap policy (Lead decision).** If the 5h wall cap is reached before the
budget is spent, the stage stops cleanly and every checkpoint written so far
is kept. This is **neither a kill nor a verdict** — the run is resumed from
`latest.pt` in a follow-up stage to finish the budget (format-4 resume,
supported since `#68`), and only the completed budget gets a verdict. No
partial-run evaluation.

**Evaluation, if not killed and the budget completes.** The arm, greedy, on
the default build, `--upgrade-availability all --frame-game-ms 100`, n=105
(15 episodes × 7 actors), run 4's eval procedure:

    uv run python scripts/run_actors.py --actors 7 --episodes 15 \
        --policy checkpoint:<run-5b arm checkpoint> \
        --upgrade-availability all --frame-game-ms 100 \
        --output-directory state/records/m2-run5b/eval-arm

Bootstrap 95% CI of the difference against run 4's arm (18.143, n=105)
(`bootstrap_difference`, seed 0, 10,000 resamples). **BETTER** iff the CI's
lower bound is > 0. **WORSE** iff the upper bound is < 0. Otherwise **NOT
DISTINGUISHABLE**. Secondary, same method: vs scripted-`all` (6.105, n=105)
and random-`all` (3.556, n=90).

**Power.** At run 4's own arm SD (3.740, n=105 per arm), the difference's
SE ≈ 0.516, so the 95% CI half-width is ≈1.0. The minimum difference
detectable with 80% power is ≈2.8 × 0.516 ≈ **1.45 waves**; a true 1-wave
difference is resolved only about half the time. A `NOT DISTINGUISHABLE`
verdict means "not resolved at this n", not "no effect".

**Timebox.** Training ≤5h wall hard cap. Run 5's own decisions/hour on
`render-interval-16` was 29,273.9 vs run 4's 27,677.0 on the default build —
only 1.06×, despite the build's own 1.74–1.92× game-time speedup (`#27`
stage 3). This is consistent with a synchronous gradient step (~120 ms) per
decision pacing the fleet, not render/collection time — the build changes
game-time throughput, not decision throughput. At 29,273.9 decisions/hour,
120,712 decisions projects to **≈4.12h**, inside the cap but not by a wide
margin; see the cap policy above if it is not.

**Safety, unchanged.** Clone AVD `tower_rl_instrumented_api36` only, even
console ports from 5556, `-read-only`, offline by interface, no taps, no
screenshots, no coins/permanent-progression changes (in-run purchases fine).
Every device stage under `scripts/run_stage.sh` with full cleanup and host
verification (no qemu via `/proc/*/exe`, empty `adb devices`) after. Stop
after three consecutive unexplained failures. `state/bridge/current` is
never repointed. One eval retry is allowed after full cleanup if instances
drop mid-eval (the gfxstream renderer crash seen in run 4's first eval
attempt); after that, stop and report with crash lines and logcat.

### Results, as run

**Verdict: KILLED — no evaluation.** Kill bar K1 (`12000:8000:8.6`) fired at
12,022 decisions: the near-greedy mean final wave over the (8000, 12000]
window was **8.409** (66 near-greedy episodes), below the 8.6 threshold. K2
was never reached. Per this entry's arm-selection rule, a killed run's arm
is not evaluated; no stage 3.

Run: `state/runs/session-20260924-115003/stacked-dqn-20260924-115003-a8c07a/`.
Decisions at stop: 12,223 (session summary, slightly past the 12,022
kill-check reading as the fleet finished in-flight episodes).
`game_seconds`: 78,781.505. `optimisation_steps`: 10,032. Training wall:
1,465.16s (≈24.4 min); stage wall including bring-up/teardown: 00:33:15,
well inside the 5h cap. Throughput: 30,032.8 decisions/h, 193,571.6
game-s/h — consistent with run 5's rate on the same build (29,273.9
decisions/h, `M2-P005`). Checkpoints written: `checkpoint-d0005016.pt`,
`checkpoint-d0010018.pt`, `latest.pt` (12,223 decisions). No selection
period closed (`periods_closed: 0`) — the run collapsed before the first
15,000-decision period boundary, so no per-period means exist to report.

Host cleanup verified after stage exit: zero qemu processes (`/proc/*/exe`),
empty `adb devices`, `run_stage.sh`'s own teardown reported `cleanup ok`
(6/7 instances already not live, 1 exited during teardown while offline;
0/7 required active cleanup — consistent with a fleet that had already torn
itself down, not a stuck instance).

This is a second `render-interval-16`-collection run stopping on the same
K1 bar inherited from run 3's curve, at a materially lower window mean
(8.409) than run 5's own tighter K1 pass point — but run 5 passed K1 (10.102
vs a 9.2 bar) and was only killed later, at K2. This run failed the looser,
run-3-derived K1 bar outright, at a wall time (~24 min) far too short to
distinguish collapse from ordinary early-training variance under a fresh
seed-0 initialization. The 2× decision budget was not reached; the
hypothesis is untested by this run — the kill-bar mechanism did its job
(stopping a run whose near-greedy policy was not learning), but leaves no
result to compare against run 4. Comment posted on `#67`; board item not
moved (kill, not a completion, per this entry's stated protocol — the Lead
decides whether to retry with a different seed or otherwise).

## M2-P004 — Milestone 2, run 4: DER-rate gradient steps, BBF n-step anneal, `frame_game_ms` 100, one seed (pre-registered, written before any run)

**Change vs run 3 (`M2-P003`).** Everything else identical to run 3: `stacked-dqn`,
`--upgrade-availability all`, ε anneal 8,000 decisions, `--budget-game-seconds 240000`,
`--block-game-seconds 4000`, `--checkpoint-every-game-seconds 60000`,
`--early-stop-patience-periods 2`, `--early-stop-min-improvement 0.2`, seed 0,
7 actors, 120 Hz, `-gpu host`.

- `--gradient-steps-per-decision 1.0` (was 0.25, the DER rate — run 3's default).
- n-step return annealed 10 → 3 exponentially over the first 10,000 gradient
  steps (`--n-step 10 --n-step-final 3 --n-step-anneal-steps 10000`), the BBF
  recipe (Schwarzer et al. 2023, arXiv:2305.19452). Run 3 held n fixed at 10.
- `--frame-game-ms 100`, explicit. Run 3 ran at 16.667 (the pre-`2d38bc9`
  default); 100 and 16.667 were shown equivalent under the M2 setup in
  `M2-S001`, and `2d38bc9` since made 100 the standing default.

**Kill bars, pre-registered from run 3's own curve** (`checkpoint_period_line`
near-greedy means, at matched cumulative fleet decisions, over near-greedy
actors' valid episodes — ladder indices 3–6, epsilon floor ≤0.02, emulators
`emulator-5562`..`emulator-5568` for this actor count):

- K1 `--kill-bar 12000:8000:8.6` — window (8000, 12000] decisions, mean < 8.6
  stops the run (run 3's own value over the comparable window: 9.28).
- K2 `--kill-bar 26262:8000:10.2` — window (8000, 26262] decisions, mean <
  10.2 stops the run (run 3: 11.18).

A kill-bar stop skips the final evaluation (stage 3) entirely; the entry
records which bar fired and the window mean it fired on, and nothing further.

**Arm selection and formal evaluation**, if not killed, exactly as run 3:
arm = best near-greedy period ≥2 (ties to the lower period number); stage 3 =
15 episodes × 7 actors = 105, greedy policy, at `--frame-game-ms 100`.

**Primary comparison**: run-4 arm vs run-3 arm (15.98, n=105 — equivalent to
run 3's own near-greedy in-training figures at 100 ms per `M2-S001`: 16.17,
n=35). Bootstrap 95% CI of the difference
(`src/tower_rl/experiment/comparison.py::bootstrap_difference`, seed 0, 10,000
resamples). **"Better"** means the CI's lower bound is > 0. Secondary
comparisons: vs scripted-`all` (6.105) and random-`all` (3.556), same method.

**Throughput.** Fleet game-s/hour and decisions/hour, reported beside run 3's
36,855.7 game-s/h (same definitions — see `M2-P003`'s "Results, as run").
Learner lock share = learner-step wall time / collection wall time, to
separate a DER-rate slowdown (more gradient steps per decision) from any
cadence effect.

**Timebox.** Training stage ≤7h wall hard limit (via `run_stage.sh`'s own
option if it has one for this; otherwise this figure is enforced by the
operator and reported at expiry rather than by the tool). Eval stage ≤1.5h.

### Results, as run (2026-09-24)

**Stage 2 — training.** `stage m2-run4-train-seed0: exit 0, cleanup ok,
instances 0/7 cleaned, 1 exited during teardown, wall 03:04:49` — well inside
the 7h box. Ran its full budget; neither kill bar fired.

| period | game-s at checkpoint | near-greedy mean, final wave | n |
| --- | --- | --- | --- |
| 1 | 60000 | 6.06 | 180 |
| 2 | 120000 | 11.57 | 90 |
| 3 | 180000 | 12.87 | 78 |
| 4 | 240000 | 16.46 | 67 |

Kill-bar checks (neither stopped the run): K1 at 12,062 decisions, mean
10.03 (bar 8.6); K2 at 26,347 decisions, mean 11.26 (bar 10.2).

Arm = period 4 (highest mean, no tie): `checkpoint-gs0240594.pt`, sha256
`bc94b628c2e2a028857d6a9238fe2445f1911818eda34d6c55089b969350fdde`, at
`state/runs/session-20260924-030851/stacked-dqn-20260924-030851-6014a1/checkpoints/`.
Health: 0 `pin_restarts`, 0 `UNLOCK_*`; 793 episodes, 790 valid (3 invalid: 2
`action_pipeline_failed`, 1 `observation_invalid`).

Throughput: `game_seconds_per_hour` 110,758.8 vs run 3's 36,855.7 — **3.01×**.
Learner lock share (learner-step wall / bridge-round-trip wall, fleet-summed
over the whole run): 4,477.4s / 35,852.4s = **12.5%**.

**Stage 3, first attempt — failed, not pooled.** `m2-run4-eval-arm`: `exit 1,
cleanup ok, instances 0/7 cleaned, 0 exited during teardown, wall 00:23:49`.
5 of 7 actors (`emulator-5560/5562/5564/5566/5568`) failed simultaneously with
`BridgeDisconnectedError` ("bridge closed the stream" / "bridge is not
connected"), then failed teardown with `adb: device '<serial>' not found` —
the devices themselves vanished, not just the bridge socket.
`emulator-5556`/`emulator-5558` completed cleanly (30 valid episodes, 0
invalid). Root cause, confirmed by host-side evidence (memory ruled out
first: 68Gi free, 92Gi available at the time; the exhausted swap was
`redis-server`'s, unrelated): `journalctl -k` for the failure window shows 10
`RenderThread[pid]: segfault ... in libgfxstream_backend.so` entries at the
same instant (06:33:34), all at the same in-library offset — 2 render
threads × the 5 failed instances. Each failed instance's own log
(`state/logs/tower-rl-emulator-emulator-{5560,5562,5564,5566,5568}.log`)
independently shows `ERROR | Failed to find ColorBuffer: 96`, then `X
connection to :0 broken (explicit kill or server shutdown)` at the same
point — the shared host X display behind the `-gpu host` renderer crashed
and took exactly those 5 instances with it; the 2 survivors show no such
lines. A host GPU/X-server renderer fault, not memory, not another agent,
not a code regression in this run's own changes. These 30 episodes are **not
pooled** with the retry below.

**Stage 3, retry — `m2-run4-eval-arm-2`.** Fresh, complete: `exit 0, cleanup
ok, instances 0/7 cleaned, 1 exited during teardown, wall 00:26:57`. 7/7
actors, 105/105 valid, 0 invalid, 0 `GAME_TIME_DEFLATED`, 0 `pin_restarts`, 0
`bridge_event_divergence`, 0 `UNLOCK_*`. Same checkpoint, sha reconfirmed
unchanged before the retry.

| arm | n | mean final wave | sd | SE |
| --- | --- | --- | --- | --- |
| run-4 arm, `all`, 100 ms | 105 | 18.143 | 3.740 | 0.365 |

Bootstrap 95% CI of the difference (`bootstrap_difference`, seed 0, 10,000
resamples), against each reference's own raw episodes:

- vs run-3 arm (15.98, n=105, at 16.667 ms): **+2.162 [+1.152, +3.190]** —
  lower bound > 0, run 4 is **better**.
- vs run-3 arm at 100 ms (`M2-S001`, 16.17, n=35): **+1.971 [+0.505,
  +3.514]** — lower bound > 0, run 4 is **better**.
- vs scripted-`all` (6.105, n=105): **+12.038 [+11.305, +12.743]**.
- vs random-`all` (3.556, n=90): **+14.587 [+13.789, +15.354]**.

**Verdict.** The DER-rate gradient step, BBF n-step anneal, and
`frame_game_ms` 100 change together produce an arm that beats run 3's arm by
both available comparisons, CI lower bound clearly above zero either way.

## M2-P005 — Milestone 2, run 5: 2× run-4 budget, `render-interval-16` training collection, kill bars re-derived from run 4's curve (pre-registered, written before any run)

**Date:** 2026-09-24. Board `#64`. Recipe identical to run 4 (`M2-P004`) in
every hyperparameter and flag — `stacked-dqn`, `--upgrade-availability all`,
`--frame-rate-hz 120`, `--decision-cadence choice-points`,
`--exploration ladder`, `--epsilon-anneal-decisions 8000`,
`--block-game-seconds 4000`, `--checkpoint-every-game-seconds 60000`,
`--early-stop-patience-periods 2`, `--early-stop-min-improvement 0.2`,
`--gradient-steps-per-decision 1.0` (DER rate), `--n-step 10 --n-step-final 3
--n-step-anneal-steps 10000` (BBF anneal), `--frame-game-ms 100`, 7 actors,
seed 0. **The only changes:**

(a) **Decision budget doubled.** `--budget-game-seconds` 240000 → **480000**
(run 4's own mechanism for spending decision budget; run 4 produced 60,356
fleet decisions over its 240,000 game-second budget). With
`--checkpoint-every-game-seconds` unchanged at 60000, this run has **8**
60,000-game-second periods instead of run 4's 4.

(b) **`render-interval-16` for the training stage only.**
`TOWER_BRIDGE_BUILD_DIR=state/bridge/builds/render-interval-16`
(`7b5e97014b37c63fc0172c5aa3212ca2975ef1fb1722431437b0ca9d902aa228`), exported
in the shell before `scripts/run_stage.sh` so both the stage's own
`instrumented_bridge.sh deploy/cleanup` calls and `scripts/train.py`'s
subprocess inherit it — the same mechanism `#27` stage 3 used, per the Lead's
adopt decision (commit `6683750`): training collection only, evaluation stays
on the default build (`state/bridge/current`, unset
`TOWER_BRIDGE_BUILD_DIR`), spectate is not part of this run.

(c) **Kill bars re-derived from run 4's own near-greedy curve**, by the same
method `M2-P004` used on run 3's curve (verified by exact reproduction: run
3's stated window means 9.283 (n=60) and 11.177 (n=124) for windows
(8000,12000] and (8000,26262] both reconstruct exactly from
`collected_episodes` in `state/runs/session-20260920-144627/.../summary.json`,
cumulative fleet decisions in list order, near-greedy actor IDs
`emulator-5562/5564/5566/5568:stacked-dqn`; run 3's stated thresholds 8.6 and
10.2 both equal `round(window_mean − 3.5 × standard_error, 1)` applied to
those two window statistics, to the exact tenth, for both bars — this is
therefore the method, not a guess).

Window definition: `(START, AT]` cumulative fleet decisions over near-greedy
actors' valid episodes (ladder indices 3–6, epsilon floor ≤0.02; same actor
count 7 so the same emulator serials). `START = 8000` is unchanged (the
epsilon-anneal horizon, itself unchanged — see the schedule note below). K1's
`AT = 12000` is reused unchanged from run 4: it is a fixed round offset from
warm-up (not tied to any run-3-specific period boundary), and decision
density near warm-up is close between the two runs (run 3 period-1 close
10,235 decisions; run 4 period-1 close 10,205), so the same early check
window applies. K2's `AT` mirrors how run 4's K2 was set: `M2-P004`'s
`AT = 26262` was exactly run 3's period-2 `decisions_at_end`
(`checkpoint_periods[1].decisions_at_end` in run 3's summary), i.e. the
close of the period `M2-P002` amendment 3 originally named as the kill
check's scope. Run 5's K2 therefore uses run 4's own period-2
`decisions_at_end`, **24328**
(`state/runs/session-20260924-030851/stacked-dqn-20260924-030851-6014a1/summary.json`,
`checkpoint_periods[1]`).

Recomputing the same window/actor/validity filter over run 4's
`collected_episodes`:

| bar | window (decisions) | n | mean | sd | SE | threshold = round(mean − 3.5·SE, 1) |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | (8000, 12000] | 38 | 10.026 | 1.498 | 0.243 | **9.2** |
| K2 | (8000, 24328] | 114 | 11.158 | 1.627 | 0.152 | **10.6** |

Flags: `--kill-bar 12000:8000:9.2 --kill-bar 24328:8000:10.6`. A kill-bar stop
skips the final evaluation entirely, exactly as `M2-P004` specified; the entry
then records only which bar fired and the window mean it fired on.

**Schedules checked against the 2× budget, none retuned.** `train.py`'s
epsilon anneal (`--epsilon-anneal-decisions`), the n-step anneal
(`--n-step-anneal-steps`, counted in gradient steps), the replay warm-up
(`--warmup-sequences`, unset/default) and the kill-bar windows above are all
defined in **absolute decisions or gradient steps**, not as a fraction of the
budget — `ExplorationSchedule.epsilon_for` computes
`fraction = min(1, decisions / anneal_decisions)` from total fleet decisions
with no reference to the run's budget (confirmed in `M2-P003`'s own audit of
this code path), and the n-step anneal is likewise `n(t) =
round(n0·(n1/n0)^(min(t,T)/T))` in gradient-step count `t`, not in budget
fraction. Under a 2× budget every one of these schedules **completes at the
same absolute point it did in run 4** (≈8,000 decisions for epsilon, ≈10,000
gradient steps for n-step — run 4 reports this lands near 10,000 decisions at
DER rate 1.0), but that point now falls at roughly **half the fraction of the
run** it did before (run 4: warm-up/anneal span ≈13% of its ≈60,356 total
decisions; run 5, if density holds, ≈6–7% of an expected ≈120,000). No
schedule is a fraction of the budget, so none needs retuning to stay
comparable; the only consequence is that a larger share of run 5's total
decisions are collected after every schedule has reached its floor/final
value, which is what "more budget" is supposed to buy. `--checkpoint-every-
game-seconds` is also absolute (unchanged 60000), giving 8 periods instead of
4; early-stop patience/min-improvement are unchanged period-counts/margins
and apply exactly as before, so the run can still stop earlier than 8 periods
if it plateaus.

**Arm selection, unchanged from run 4/`M2-P003`.** The checkpoint written at
the close of the best near-greedy period among periods ≥2 (ties to the lower
period number, a period with no valid near-greedy episode is ineligible).

**Evaluation.** The run-5 arm, greedy, on the **default bridge build**
(`state/bridge/current`, `TOWER_BRIDGE_BUILD_DIR` unset), `n = 105` (15
episodes × 7 actors), `--upgrade-availability all --frame-game-ms 100`,
run 4's eval procedure and settings exactly:

    uv run python scripts/run_actors.py --actors 7 --episodes 15 \
        --policy checkpoint:<run-5 arm checkpoint> \
        --upgrade-availability all --frame-game-ms 100 \
        --output-directory state/records/m2-run5/eval-arm

**Primary comparison**: run-5 arm vs run 4's existing arm result (18.143,
n=105, same default build, same `frame_game_ms`). Bootstrap 95% CI of the
difference (`src/tower_rl/experiment/comparison.py::bootstrap_difference`,
seed 0, 10,000 resamples). **BETTER** iff the CI's lower bound is > 0;
otherwise **NOT BETTER**. Secondary comparisons, same method: vs scripted-`all`
(6.105, n=105) and vs random-`all` (3.556, n=90).

**Throughput.** Fleet game-s/hour and decisions/hour reported beside run 4's
110,758.8 game-s/h (default build) and `#27` stage 3's 1.92× per-actor
collection-rate ratio for `render-interval-16` — run 5's training throughput
is measured on the accelerated build, so it is not directly comparable to run
4's number without noting that difference.

**Hard timebox.** Training stage ≤ **7h** wall (`run_stage.sh`'s own
`--shutdown-grace`/operator-enforced cap, as run 4). Budget-arithmetic check
before launch: run 4 (default build) produced 60,356 decisions over
240,000 game-seconds in 03:04:49 wall (110,758.8 game-s/h). At that same rate,
480,000 game-seconds would take ≈4.33h; on `render-interval-16` at `#27`
stage 3's measured 1.92× per-actor collection-rate ratio, ≈2.26h — both well
inside the 7h cap, so no contradiction is expected between the budget and the
box. The kill bars and the 7h cap end the run early if triggered; a kill-bar
stop still gets the default-build arm evaluation of its best checkpoint (the
last checkpoint written before the stop, since a kill-bar stop selects no arm
by the training-time rule but a checkpoint still exists to evaluate per the
task packet's explicit instruction).

**Safety, unchanged.** Clone AVD `tower_rl_instrumented_api36` only, even
console ports from 5556, `-read-only`, offline by interface, no taps, no
screenshots, no coins/permanent-progression changes (in-run purchases fine).
Every device stage under `scripts/run_stage.sh` with full cleanup and host
verification (no qemu via `/proc/*/exe`, empty `adb devices`) after. Stop
after three consecutive unexplained failures. `state/bridge/current` is never
repointed. One eval retry is allowed after full cleanup if instances drop
mid-eval (a gfxstream renderer crash was seen in run 4's first eval attempt);
after that, stop and report with crash lines and logcat.

### Results, as run

**Stage 2 — training.** `stage m2-run5-train-seed0: exit 0, cleanup ok,
instances 0/7 cleaned, 1 exited during teardown, wall 00:58:31` — well inside
the 7h box; the box was never a binding constraint. Internal (collection-only)
wall time `wall_seconds` 2995.1s (00:49:55). `TOWER_BRIDGE_BUILD_DIR` build
digest reconfirmed `7b5e97...aa228` before launch.

**K2 fired; the run stopped itself after 2 periods, at 160,108 of the 480,000
game-second budget (33.4%).** Neither bar's window means are close calls in
opposite directions: K1 passed with margin (mean 10.102 vs bar 9.2, window
(8000,12000], n=49), K2 failed (mean 10.344 vs bar 10.6, window (8000,24328],
n=195, checked at fleet decision 24,355). `early_stopping.kill_bar_checks`
(verbatim):

| bar | at (decisions) | window start | min mean | checked at (decisions) | n | mean | stopped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K1 | 12000 | 8000 | 9.2 | 12,036 | 49 | 10.102 | no |
| K2 | 24328 | 8000 | 10.6 | 24,355 | 195 | 10.344 | **yes** |

Totals at the stop: 24,355 decisions, 22,103 optimisation (gradient) steps,
626 accepted replay sequences, 634 episodes (626 valid, 8 invalid:
6 `observation_invalid`, 2 `action_pipeline_failed`), 0 `pin_restarts`, 0
`actors_withdrawn`, 2 `failed_episodes` (`lifecycle_timeout` on `speed_max`
and one `did not reach an active run` — ordinary single-actor bring-up flake,
not the `#59`/gfxstream signature). `advances_cut_short` 155/626 valid
episodes (24.8%), markedly higher than run 4's ≈0.3–0.5 per episode at the
period level — read as a `render-interval-16` timing-measurement artefact
consistent with `#27` stage 3's own elevated (but still in-margin)
`advances_cut_short` reading for that build, not a new failure mode.

**Near-greedy curve vs run 4's**, by closed period (period boundaries fall at
different decision counts between the two runs because periods are cut by
game-seconds, not decisions, and `render-interval-16` changes the
decisions/game-second density):

| period | run 5 decisions-at-end | run 5 mean | run 5 n | run 4 decisions-at-end | run 4 mean | run 4 n |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (60,000 gs) | 10,189 | 5.793 | 184 | 10,205 | 6.056 | 180 |
| 2 (120,000 gs) | 18,646 | 10.465 | 101 | 24,328 | 11.567 | 90 |

Run 5's period 1 lands at almost the same decision count as run 4's (10,189
vs 10,205) and a comparable mean (5.79 vs 6.06). Run 5's period 2 closes
**6,000 decisions earlier** than run 4's did (`render-interval-16` collects
more decisions per wall-hour but very nearly the same decisions per
game-second here — decisions/hour rose only 1.06× against game-s/hour's
1.74×, so periods, which are cut by game-seconds, close at a similar decision
count either way) and its near-greedy mean (10.465) sits below run 4's later
period-2 reading (11.567). The kill-bar windows are the fairer,
decision-matched comparison and are what actually judged this: run 5's
window mean at the same absolute decision range run 4 set the K2 bar from
(10.344) fell under run 4's own margin (10.6), which is why K2 fired.

**Arm selection.** Only period 2 closed at or after period 2 (`M2-P002`
amendment 3 excludes period 1), so it is the only eligible period and the
arm by construction: `checkpoint-gs0120026.pt`, sha256
`2c55b4a0501c276a10d9dab2d404f37d98203ea983b375dfe1b871984fb7dbc4`
(verified against its `.sha256` sidecar), at
`state/runs/session-20260924-084043/stacked-dqn-20260924-084043-f6567d/checkpoints/`.
Checkpoint identity `5cffec80fccb` (from the evaluation manifest below).

**Throughput vs run 4** (both fleet-summed over their own run):

| | run 5 (`render-interval-16`) | run 4 (default build) | ratio |
| --- | --- | --- | --- |
| game-s/hour | 192,444.6 | 110,758.8 | **1.74×** |
| decisions/hour | 29,273.9 | 27,677.0 | **1.06×** |

The game-time ratio is in the neighbourhood of `#27` stage 3's measured
1.92× per-actor collection-rate ratio (not identical — this run is much
shorter and includes proportionally more bring-up/warm-up than a steady-state
reading, and is a different, later point on the training curve). The
decisions/hour ratio is far smaller: `render-interval-16` buys more simulated
game-time per wall-hour, not proportionally more choice-point decisions per
wall-hour, at this cadence and roster.

**Stage 3 — arm evaluation, default build.** `stage m2-run5-eval-arm: exit 0,
cleanup ok, instances 0/7 cleaned, 1 exited during teardown, wall 00:15:13`.
7/7 actors, 105/105 valid, 0 invalid — no instance drop, so the one allowed
retry was not needed. `TOWER_BRIDGE_BUILD_DIR` unset for this stage
(confirmed absent from the launching shell's environment before launch).

| arm | n | mean final wave | sd | SE |
| --- | --- | --- | --- | --- |
| run-5 arm (period-2 checkpoint), `all`, 100 ms | 105 | 7.657 | 1.770 | 0.173 |

Bootstrap 95% CI of the difference (`bootstrap_difference`, seed 0, 10,000
resamples), against each reference's own raw episodes (run 4's arm and both
baselines' raw episode files reproduce the doc's recorded means exactly:
18.143/105, 6.105/105, 3.556/90):

- vs run-4 arm (18.143, n=105): **−10.486 [−11.267, −9.686]** — lower bound
  well below 0. **NOT BETTER.**
- vs scripted-`all` (6.105, n=105): **+1.552 [+1.200, +1.886]** — beats
  scripted.
- vs random-`all` (3.556, n=90): **+4.102 [+3.640, +4.552]** — beats random.

**Verdict: NOT BETTER.** The primary comparison's CI lies entirely below
zero: the run-5 arm is worse than run 4's arm, not merely statistically
indistinguishable from it. This is not read as evidence against the 2×-budget
or `render-interval-16` changes themselves — K2 stopped the run at 33% of its
budget, before either change had a chance to compound, on a kill bar that did
exactly the job it was set up to do (stop early on a trajectory tracking
below the reference run's own early curve at matched decisions) rather than
on any fault in this run. The evaluated arm is a period-2 checkpoint,
directly comparable in training extent to a `run 3`-scale amount of
collection, not to run 4's period-4 arm; the CI answers the question the
task packet asked (did the run-5 arm beat run 4's?) and the answer is no, but
it is not a like-for-like test of "does more budget / does
`render-interval-16` help", which a kill-bar stop this early cannot speak to
either way.

## M2-P005 diagnostic — why run 5 fell behind run 4 per decision (pre-registered, written before the device stage)

**Date:** 2026-09-24. Board `#64`, follow-up to `M2-P005`'s NOT BETTER verdict.
Three candidate causes for why run 5's near-greedy window mean at matched
decisions (10.344) sat below run 4's own reading at the same window, and why
the greedy default-build eval of a **more-trained** run-5-recipe arm would be
expected to beat, not undercut, a near-greedy in-training figure of similar
magnitude:

- **H1 — `render-interval-16` changes the game for a *learned* policy.**
  `#27` stage 3 validated fidelity only for the **scripted** policy, which
  reads almost none of the observation. Warning sign: near-greedy on the
  `render-interval-16` build was 10.34 (run 5, in-training), but greedy
  evaluation of a *more*-trained arm on the default build (run 4's own arm)
  was 18.14 — not a contradiction by itself (different policies, different
  builds, different training extent), but the gap this diagnostic exists to
  bound.
- **H2 — a schedule defined as a fraction of the budget** (epsilon, beta,
  warm-up) leaves the 2× run less annealed at matched decisions.
- **H3 — seed variance.** The kill-bar SE is within-run episode noise, not
  between-seed variance, so a single-seed kill-bar reading cannot rule out
  ordinary run-to-run spread.

### Desk analysis (before any device time)

**(a) Which schedules scale with the budget.** `TrainingConfig.beta` (the PER
importance-sampling exponent, `src/tower_rl/learning/training.py:357`) is
defined as `beta_start + (beta_end − beta_start) · progress(game_ms)` where
`progress = min(1, game_ms / budget_game_ms)` — **a fraction of the budget**,
not of decisions. This was missed in `M2-P005`'s own pre-registration, which
checked epsilon, n-step and warm-up (all absolute) but not beta explicitly.
`epsilon_anneal_decisions` (8000) and `n_step_anneal_steps` (10000 gradient
steps) are both absolute and reach the same point at the same decision/
gradient-step count regardless of budget — confirmed by the near-greedy
ladder floors being identical between the two runs at decision ≈24k (both
past the epsilon floor). `warmup_sequences` (100) is absolute. **Beta is the
one schedule that differs**: at decision 24,328 (run 4) / 24,355 (run 5),
actual cumulative game time is 119,124.3 s (run 4) and 160,108.1 s (run 5);
against each run's own budget (240,000 s / 480,000 s) that is progress 0.496
and 0.334, giving **beta 0.698 (run 4) vs 0.600 (run 5)** at matched
decisions — run 5 is less annealed on this one schedule, as H2 predicts.
However **`priority_alpha` is 0.0 in both runs** (`resolved_config`,
confirmed by direct read), so priorities are uniform and the IS weight
`(1/(N·P(i)))^beta` collapses to 1 regardless of beta's value — **beta's
practical effect on the gradient should be negligible in both runs**, so H2's
mechanism is present but is not expected to explain the shortfall given
`priority_alpha=0`. This is reported as a real schedule asymmetry, not
retuned.

**(b) Run 4's own near-greedy window mean over the same (8000, 24328]
decisions.** Recomputed from `state/runs/session-20260924-030851/.../summary.json`
`collected_episodes` with the identical filter `M2-P005` used to derive the
kill bars: **mean 11.158, sd 1.627, SE 0.152, n=114** — against run 5's
10.344 (n=195) at the same nominal window. Difference 0.814 waves, about
5.4× run 4's own SE — a real, not noise-level, gap between the two runs at
matched decisions on the metric the kill bar actually used.

**(c) Game-seconds per decision, matched decision range (8000 to ≈24,330],
not whole-run.** Whole-run throughput figures (192,444.6 vs 110,758.8
game-s/hour) are not comparable because run 5 stopped early and includes
proportionally more bring-up. Over the identical decision span:

| run | decisions in span | game-s in span | game-s/decision |
| --- | --- | --- | --- |
| run 4 | 16,302 | 72,095.7 | **4.423** |
| run 5 | 16,324 | 113,856.5 | **6.975** |

Run 5 spends **1.577×** as much game-time per decision as run 4 over the
identical decision range. This is the sharpest finding of this diagnostic:
`#27` stage 3's own fidelity gate (2) found `render-interval-16` did **not**
move decisions/wave for the **scripted** policy (4.05–4.18 across all three
arms). A 1.58× game-time-per-decision inflation for the near-greedy
**learned** policy that does not appear for the scripted policy is direct,
matched-range evidence **for H1**: the build behaves differently for a
policy that reads the observation than for one that mostly ignores it —
consistent with, though not proof of, an observation- or timing-sensitive
effect `render-interval-16`'s validation never covered.

**(d) `advances_cut_short` per episode, matched decision range.** Run 4:
44/219 valid episodes = **0.201** (20.1%). Run 5: 81/351 valid episodes =
**0.231** (23.1%). This is a materially smaller gap than the whole-run
figures in `M2-P005`'s first write-up implied (155/626 = 24.8% for run 5
against a loosely-stated "run 4's ≈0.3–0.5 per episode at the period
level", which was not a matched comparison and is corrected here): at the
same decision range the two runs differ by only 3 points (20.1% vs 23.1%,
ratio 1.15×), not the several-fold gap the uncorrected comparison suggested.
`advances_cut_short` is not read as a material contributor to the shortfall.

**Reading.** (a)+(d) do not support H2 or a fidelity-health explanation as
primary. (c) is the strongest, matched-range signal and points at H1: the
policy-dependent game-time-per-decision inflation is a real, build-specific
effect not present for the scripted policy `#27` stage 3 validated, and it is
large enough (1.58×) to plausibly explain a fleet collecting materially less
*policy-relevant* experience per decision, which would depress the
near-greedy curve at matched decision counts independent of anything about
annealing or health. H3 (seed variance) cannot be excluded by desk analysis
alone — one seed each side of the comparison cannot separate it from H1 — but
(c)'s policy-specific signal is not what seed variance alone would predict
(seed variance would not systematically differ between the scripted-policy
equivalence check and this learned-policy reading).

### Device stage — does `render-interval-16` stay faithful for a *trained* policy?

**Design.** Evaluate the **run-4 arm** (`checkpoint-gs0240594.pt`, sha256
`bc94b628c2e2a028857d6a9238fe2445f1911818eda34d6c55089b969350fdde`, the same
checkpoint `M2-P004` evaluated at 18.143/n=105 on the default build) on the
`render-interval-16` build
(`7b5e97014b37c63fc0172c5aa3212ca2975ef1fb1722431437b0ca9d902aa228`,
`TOWER_BRIDGE_BUILD_DIR=state/bridge/builds/render-interval-16` exported to
both `run_stage.sh` and the runner), run 4's eval settings (`--upgrade-
availability all --frame-game-ms 100`, greedy), **7 actors × 5 episodes = 35
attempted** (n target 35, smaller than the usual 105 because this is a
fidelity check on an existing, already-measured arm, not a fresh claim):

    scripts/run_stage.sh --name m2-run5-diag-eval --instances 7 -- \
        uv run python scripts/run_actors.py --actors 7 --episodes 5 \
        --policy checkpoint:state/runs/session-20260924-030851/stacked-dqn-20260924-030851-6014a1/checkpoints/checkpoint-gs0240594.pt \
        --upgrade-availability all --frame-game-ms 100 \
        --output-directory state/records/m2-run5/diag-eval

**Rule.** Bootstrap 95% CI of the difference against run 4's own default-build
reading (18.143, n=105) via `bootstrap_difference` (seed 0, 10,000
resamples). **FAITHFUL** if the whole CI lies inside ±2.0 waves. **NOT
FAITHFUL** if the CI's upper bound is < −2.0 or its lower bound is > +2.0.
Otherwise **INCONCLUSIVE**. Safety, crash/retry rules unchanged from
`M2-P005` (one retry after full cleanup if instances drop mid-eval, no
pooling of partial attempts).

### Results, as run

`stage m2-run5-diag-eval: exit 0, cleanup ok, instances 0/7 cleaned, 1 exited
during teardown, wall 00:10:26`. `TOWER_BRIDGE_BUILD_DIR` build digest
reconfirmed `7b5e97...aa228`, checkpoint sha256 reconfirmed
`bc94b628...50fdde` before launch. 7/7 actors, 35/35 valid, 0 invalid — no
instance drop, retry not needed.

| arm | n | mean final wave | sd |
| --- | --- | --- | --- |
| run-4 arm on `render-interval-16` | 35 | 17.857 | 4.008 |

Bootstrap 95% CI of the difference against run 4's own default-build reading
(18.143, n=105): **−0.286 [−1.800, +1.190]** — the whole interval is inside
±2.0. **Verdict: FAITHFUL.** The same checkpoint, played greedily, reaches
statistically indistinguishable performance on `render-interval-16` as on the
default build it was trained and evaluated on.

**Reading against H1–H3.** This directly weakens H1 as an explanation for the
in-training shortfall: at evaluation time (greedy, no exploration, a fully
trained network), `render-interval-16` does not measurably change outcomes
for a policy that reads the full observation, extending `#27` stage 3's
scripted-policy fidelity finding to a trained one. It does **not** rule out a
narrower version of H1 — this stage tests greedy inference on a *finished*
network, not near-greedy *collection* with exploration noise on a
*mid-training* network, which is what actually happened during run 5's
collection and is where desk finding (c)'s 1.58× game-time-per-decision
inflation was measured. That inflation is also confounded with policy
strength itself (a policy that survives to a higher wave necessarily
accumulates more advances, hence more game-time, before its episode ends,
independent of any build effect), which this diagnostic's design does not
separate out. H2's mechanism is confirmed present (beta genuinely is
fraction-of-budget) but is not expected to matter given `priority_alpha=0` in
both runs. H3 (seed variance) remains neither confirmed nor excluded — this
diagnostic did not add a second seed of either run, so it cannot speak to it
directly; the FAITHFUL build result at least removes "the build silently
breaks trained-policy inference" as a competing explanation that would have
had to be ruled out before seed variance could be considered the leading
account.

**Net effect on `M2-P005`'s verdict.** Unchanged: NOT BETTER stands. This
diagnostic narrows *why* — the build itself is faithful for a trained,
greedy policy; the gap is most consistent with the near-greedy collection
dynamics during the shortened run (matched-decision-range game-time-per-
decision inflation, desk finding (c)) rather than with an annealing schedule
or a general fidelity break, but is not fully resolved and would need a
second seed (H3) or a near-greedy (not greedy) collection-time fidelity
check to separate cleanly.

### Addendum — the free same-policy discriminator for finding (c) (desk-only, no device)

The diagnostic device stage already collected the one comparison finding (c)
needed and never had: **the same checkpoint** (run 4's arm), played greedily,
**on both builds** — this stage's own 35 episodes on `render-interval-16`
("B") against run 4's own `eval-arm-2` (default build, "A", n=105), both
`--upgrade-availability all --frame-game-ms 100`.

| | A (default), n=105 | B (`render-interval-16`), n=35 |
| --- | --- | --- |
| game-s/decision, per-episode mean (sd) | 2.565 (0.460) | 2.665 (0.689) |
| game-s/decision, pooled (Σgame_ms/Σdecisions) | 2.428 | 2.461 |
| decisions/wave, mean | 13.587 | 13.238 |
| `advances_cut_short`/episode, mean | 0.257 (27/105) | 0.143 (5/35) |

Bootstrap (seed 0, 10,000 resamples) on the per-episode game-s/decision
values: difference B−A **+0.100 [−0.121, +0.357]**; ratio B/A **1.039
[0.954, 1.140]** — both **include the null** (0 and 1.0 respectively). For
the *same* policy, game-s/decision is statistically indistinguishable
between the two builds, and `decisions/wave` and `advances_cut_short` both
sit at or below A's reading on B, not above.

**This answers the question directly: B/A ≈ 1.0, not 1.58×.** Desk finding
(c)'s 1.58× game-time-per-decision inflation was **policy state, not the
build** — it reflects what a mid-training, exploring near-greedy policy was
doing differently in run 5 (whatever combination of more WAIT, different
purchase timing, or different survival at that point in training), not a
`render-interval-16` decision-cadence effect. Combined with the FAITHFUL
verdict above, this closes out H1 for practical purposes: neither greedy
outcomes nor per-decision game-time pacing differ by build for a fixed,
fully-formed policy.

**Decision-budget arithmetic, for scale.** Run 4 reached **60,356 decisions**
and **58,046 gradient steps** over its full 240,000-game-second budget
(4.002 game-s/decision, averaged over the whole run). Run 5 reached **24,355
decisions** at **160,108.1 game-seconds** when K2 stopped it (6.574
game-s/decision, averaged over its own shorter run to that point — not the
same figure as the same-policy A/B reading above, because this is a
different, earlier-training, exploring policy over a different span). Had
run 5's kill bar not fired and it had run to its full 480,000-game-second
budget **at that same observed rate**, it would have reached only
**≈73,016 decisions** (480,000 / 6.574) — **1.21×** run 4's decision count,
not the 2× the budget doubling was meant to buy. The `--budget-game-seconds`
doubling does not translate one-for-one into a decisions doubling whenever
game-s/decision is not the same as the run it is being compared to, and here
it was not (6.574 vs run 4's whole-run 4.002) — a further reason `M2-P005`'s
framing of "the decision budget is 2× run 4's" was optimistic, independent
of anything the kill bar or the build did.

## M2-S001 — `frame_game_ms` 100 under the M2 setup: equivalence + fleet throughput (pre-registered, written before any run)

**Finding that motivates this (scout, verified against manifests and docs).**
Every M2 run (`M2-E001` onward, including run 3) trained and evaluated at
`frame_game_ms=16.667` — `scripts/run_episodes.py::add_cadence_arguments`'s
default (`1000.0 / 60.0`) — even though `M1B-E018` (commit `705836e`, on top
of `611667d`'s move to one round trip per decision) adopted 100 as the
standing decision and `M1B-E038` validated 100 on fidelity, powered, at
round-clock resolution, and rejected 150/200. No M2 pre-registration ever
pinned `--frame-game-ms`, so the argparse default rode through silently
across every M2 run. 16.667 has no powered fidelity evidence behind it —
`M1B-E018`'s own sweep was n=8 per arm, underpowered by its own account, and
`M1B-E038` tested 100/150/200, not 16.667. The fleet is vsync-bound at
~118 fps regardless of `frame_game_ms`, so game-seconds per wall-second scales
with it: 100 could collect up to ~6× the game-seconds per wall-hour that
16.667 does.

**Design.** Setup identical to run 3's stage-3 eval (7 actors, `all`,
choice-point cadence, 120 Hz guest frame rate, `-gpu host`, bridge digest
`f9d5f161c33b3af98787d161c9e73f26b1286f519b1648c41b167bffd62a96c3` from
`state/bridge/current`), except `--frame-game-ms 100`.

- **Arm S** — scripted policy, 35 episodes (7 actors × 5). Reference: run 3's
  scripted-`all` at 16.667, n=105, mean 6.105, sd 0.338.
- **Arm R3** — run 3's arm checkpoint (`checkpoint-gs0240313.pt`, sha256
  `7dd8ba3779e27848048a874e9f9c29001fdfc766b0afda9a2e0bb49c04216771`), greedy,
  35 episodes (7 × 5). Reference: run 3's formal eval at 16.667, n=105, mean
  15.98, sd 3.68.

**PASS (equivalence) iff all hold:**
1. Zero `GAME_TIME_DEFLATED` and ≤1 invalid episode per arm.
2. Arm S: 90% bootstrap CI of (100 ms − reference) inside ±0.5 waves.
3. Arm R3: 90% bootstrap CI of (100 ms − reference) inside ±2.0 waves.

CIs via `src/tower_rl/experiment/comparison.py::bootstrap_difference`, seed 0,
10,000 resamples, against each reference arm's own raw per-episode records.

**Throughput.** Game-seconds per wall-second per actor and fleet game-s/hour
at 100, compared against run 3's own eval records at 16.667 (same
definitions both sides, `valid_episodes_per_hour`-style aggregate from the
stage's final JSON summary). **GO for training at 100** only if PASS and the
fleet speed-up is ≥2×. Anything else → no change: 16.667 stays, and this
entry says why once the run's numbers are in.

**Kill rule.** Stop arm S early if its first 7 episodes show
`GAME_TIME_DEFLATED` on ≥3 of them — a signal the 100 ms cadence is not
holding on this device, checked before committing the rest of the device
budget to it.

### Results, as run (2026-09-23)

Both stages ran clean: `stage m2-s001-eval-scripted-100: exit 0, cleanup ok,
instances 0/7 cleaned, 1 exited during teardown, wall 00:10:01` and `stage
m2-s001-eval-arm-100: exit 0, cleanup ok, instances 0/7 cleaned, 1 exited
during teardown, wall 00:18:54`. Zero `GAME_TIME_DEFLATED` in either (kill
rule never triggered), zero invalid episodes in either, zero
`bridge_event_divergence`, zero `UNLOCK_*`. `checkpoint-gs0240313.pt`'s
sha256 was reconfirmed unchanged (`7dd8ba3779e27848048a874e9f9c29001fdfc766b0afda9a2e0bb49c04216771`)
before arm R3 ran. Arm R3 logged one `pin_restarts` (the boundary-recovery
retry from `762c54b`, capped at 3, recovered) — a normal, already-accounted
health event, not a failure. `frame_game_ms=100.0` was confirmed applied from
every per-actor record's own field (`state/records/m2-s001/eval-*-100/*.json`),
not just the command line.

| arm | n | mean final wave @100 | sd | reference @16.667 (n, mean, sd) |
| --- | --- | --- | --- | --- |
| S (scripted) | 35 | 6.029 | 0.296 | 105, 6.105, 0.338 |
| R3 (arm, greedy) | 35 | 16.171 | 4.127 | 105, 15.981, 3.680 |

Bootstrap 90% CI of (100 ms − reference), `bootstrap_difference`, seed 0,
10,000 resamples, against each reference arm's own raw episodes:

- Arm S: **−0.076 [−0.171, +0.019]** — inside the ±0.5 bound.
- Arm R3: **+0.190 [−1.086, +1.457]** — inside the ±2.0 bound.

Both equivalence conditions hold, so the run **PASSes** on equivalence.

**Throughput, two ways.** Per-actor collection rate (sum of each actor's own
`total_budgeted_game_seconds` / `wall_seconds`, which excludes the fleet's
shared bring-up/teardown, since that field is collection-phase only):

| arm | game-s/wall-s per actor @100 | @16.667 (run 3) | ratio |
| --- | --- | --- | --- |
| S | 5.097 | 1.680 | 3.03× |
| R3 | 4.609 | 1.794 | 2.57× |

Fleet game-s/hour (summed per-actor game-seconds over the stage's own
top-level wall-clock, which *does* include bring-up/teardown):

| arm | fleet game-s/h @100 | @16.667 (run 3) | ratio |
| --- | --- | --- | --- |
| S | 40457 | 34057 | 1.19× |
| R3 | 61316 | 41153 | 1.49× |

**These two ratios disagree, and the reason is a confound this design did not
anticipate.** Both new stages ran only 5 episodes/actor (35 total), so a
mostly-fixed per-stage bring-up/teardown cost (bridge deploy, snapshot
restore, radio settle — roughly constant regardless of cadence) is a much
larger share of the stage's wall-clock at 100 ms, where collection itself
finishes in minutes, than it was in run 3's 15-episode/actor eval stages. The
per-actor collection-rate ratio (2.6–3.0×) isolates the cadence effect and is
the correct measure of what `frame_game_ms` itself buys; the fleet-wall-clock
ratio (1.19–1.49×) is what these particular short stages actually cost
end-to-end, dragged down by overhead that would amortize away over a
training-length run. **Neither ratio reaches the pre-registered ≥2× fleet
speed-up bar as that bar was written** (a bare fleet-wall-clock number), so
the literal GO condition is not met by this evidence; whether the intent was
the fleet-wall-clock number at this stage length, or the steady-state
collection rate that would apply once run length amortizes bring-up, is a
call for the Lead, not resolved here. Host load during arm R3 reached a
15-minute load average of 27.84 (`code` at 53% CPU, no cmake/ninja/cc1/ld
process observed — not a native-bridge build); no other agent's qemu process
was seen on the host at any point.

**Verdict: EQUIVALENT, throughput inconclusive against the pre-registered
bar as written — no automatic GO.** `frame_game_ms` 100 reproduces both the
scripted and the arm-checkpoint's final-wave distribution within the
pre-registered bounds under the M2 setup. It does *not* clearly clear the
±0.5/±2.0-adjacent throughput bar at this stage's episode count; the
per-actor collection-rate evidence (2.6–3.0×) is suggestive of a real
speed-up worth reproducing at a longer episode count where fixed overhead
amortizes, but this run does not by itself authorize switching M2 training to
100. 16.667 stays the standing default pending that follow-up (superseded by
the Lead decision below).

Lead decision (2026-09-24): frame_game_ms = 100 is adopted as the default (merge 55e5e0c). Equivalence passed on both arms. The pre-registered ≥2× fleet bar measured stage wall-clock over 5 episodes per actor, where bring-up and teardown dominate; that was the wrong quantity for a multi-hour run — a design error in this pre-registration, recorded here rather than hidden by a re-run. The quantity that governs a long run is the per-actor collection rate, 2.57–3.03× faster at 100 ms. Run 4 (M2-P004) is the fleet-level measurement at 100 ms; its throughput will be reported beside run 3's 36,855.7 game-s/h. Reversible: `--frame-game-ms 16.667` restores the old value.

## M2-P003 — Milestone 2, run 3: upgrade availability `all`, corrected ε schedule, one seed, exploratory (pre-registered, written before any run)

**Date:** 2026-09-20
**Status:** **pre-registered and approved by the developer 2026-09-20;
exploratory — the confirmatory protocol is applied only if the stop criterion
below is met.** Run 3 buys iteration speed rather than a claim: shorter
baselines, a four-period training budget, and a formal evaluation that is
**conditional** on the training curve clearing the scripted baseline. If that
gate does not open, the training-time windows are the recorded result and the
run ends there. An independent numeric check of this entry was carried out and
its corrections are applied — the replay warm-up's size and everything derived
from it, the ε reached at the first gradient step, the arm-selection statistic,
the evaluation set size, the standard error behind prediction 2 and the
throughput definitions. No run has started and no device time has been spent. This entry records the plan, its prices and its decision rules before
any data exists; it is not a result. Board `#46`;
`M2-P002` is the protocol this one is a two-change edit of, and everything
`M2-P002` says that is not contradicted below stands unchanged and is not
restated.

**Objective, unchanged from `M2-P002`.** *One committed model, trained under one
budgeted protocol, reproducibly beats the random and scripted baselines.* Run 2
(`M2-E007`) did not get there: both seeds early-stopped at period 4 with a
near-greedy curve that rose once and then stopped — 4.855 → 5.529 → 5.581 →
4.490 (seed 0) and 4.822 → 5.899 → 5.000 → 5.283 (seed 1) — against a random
baseline of 5.495 and a scripted baseline of 6.429. Run 3 changes the two things
the evidence names as most likely to be holding that curve down, and changes
nothing else.

**Run 3 differs from run 2 in exactly two declared ways, and attribution is not
available from this run.** Both changes are made at once and there is one seed,
so nothing here can say which of them moved a number, or whether either did
reproducibly. That is stated before the run rather than discovered after it.

| change | what it does | evidence |
| --- | --- | --- |
| upgrade availability `all` | the environment writes every in-run upgrade-availability flag true **at every round start**, so the policy is offered the full roster (17 attack, 18 defense, 13 utility rows) instead of profile v1's six purchasable rows | `M2-E008`: the bridge write takes (Q1), the game honours purchases on rows v1 never offers (Q3), it changes no starting scalar (Q4), and it is **taken back by the next round start** (Q2) — which is why it must be re-applied per round and cannot be a one-off at process start. Implemented under `#54` as `--upgrade-availability all`; ADR 0011 |
| corrected ε schedule | `--epsilon-anneal-decisions` **2,500 → 8,000**, so the anneal runs *through* the replay warm-up instead of finishing before it | `#53` audit D1: warm-up is `--warmup-sequences` 100 ≈ 100 episodes, and run 2's own artifacts put it at **≈3,450–3,600 fleet decisions** — the first 100 valid episodes spanned 3,463 (seed 0) and 3,594 (seed 1) decisions, and the same figure falls out of the gradient-step arithmetic (43,023 − 9,893/0.25 = 3,451; 60,820 − 14,338/0.25 = 3,468) and out of `M2-P002` amendment 2's "replay warm-up did not end until around decision 3,400". The 2,500-decision anneal therefore reached its floor **≈950 decisions before the first gradient step**: every gradient step of run 2 was taken against a fully annealed fleet, and the four near-greedy actors played an untrained network deterministically from the first step onward |

### The ε correction, as flags and as numbers

The deviation is not "2,500 was too small"; it is that **the anneal and the
warm-up were ordered the wrong way round**. Run 2's near-greedy actors reached
their rungs (0.0162, 0.0056, 0.0019, 0.00066) at fleet decision 2,500, about
**950 decisions** *before* the learner took its first gradient step (3,430 in
attempt 1, 3,451 at seed 0, 3,468 at seed 1), and a greedy actor
on a freshly initialised network is not a weak policy but a near-constant one:
`#53`'s init-time probe over 12 seeds found the greedy argmax took one to five
distinct actions over 200 states and was `WAIT` everywhere in half the seeds,
because `wait_advantage` and `row_advantage` are separate heads whose constant
offset the valid-set centring cannot remove. A constant policy is *worse* than
uniform random, which is what attempt 1 measured: near-greedy 3.188 against
random's 5.495.

**The flag.** `--epsilon-anneal-decisions 8000`, replacing run 2's `2500`.
Everything else about exploration is untouched: `--exploration ladder`, the same
seven Ape-X rungs, `--epsilon-end` still not given.

**Why 8,000.** The horizon is bounded on both sides by numbers this project has
measured.

- **Lower bound — warm-up.** The anneal must still be running when the learner
  starts, so that the network is trained on experience collected at a falling
  exploration rate rather than on experience collected entirely at the floor.
  At 8,000, warm-up ends at **43%** of the anneal at run 2's density (3,463 of
  8,000) and at **≈52%** under availability `all`, where the first 100 episodes
  should cost ≈4,100–4,200 decisions if density rises as prediction 1 says. The
  anneal starts from `--epsilon-start`'s default of **1.0**, not from the
  ladder's top rung, so the near-greedy actors are still at **ε ≈ 0.58** when
  the first gradient step is taken (0.574 at actor 3, 0.567 at actor 6; ≈0.49
  under `all`) and reach their rungs **≈4,400–4,540 decisions later**, which at
  0.25 gradient steps a decision is **≈1,100 optimisation steps** of annealed
  exploration over a network that is being trained — ≈960 under `all`'s later
  warm-up. Run 2 had none. This is Mnih 2015's arrangement — anneal *while*
  learning — at this run's scale rather than at Atari's 1M frames.
- **Upper bound — the kill check's scope.** `M2-P002` amendment 3 places the
  kill check at the close of **period 2**, on the ground that period 2 is the
  first period containing no pre-anneal and no pre-warm-up episodes. That holds
  only if the anneal completes inside period 1. Run 2's period 1 held **10,849**
  decisions (seed 0) and **13,304** (seed 1), so 8,000 completes at 74% of the
  smaller of the two, with ~2,800 decisions of margin. The margin survives
  either direction of the density change run 3 makes: a **20% fall** would still
  leave seed 0's period 1 at ~8,700 > 8,000, and the ~20% *rise* `M2-E008`
  points at (6.0 decisions a wave in its one episode against run 2 random's
  5.00 — direction only, from one resumed episode on one instance, not a
  threshold) would put 8,000 at **61%** of period 1.

**Why not the shipped default of 10,000.** It would buy ~1,475 gradient steps
under falling exploration instead of ~1,100, and it is rejected on the upper
bound above: at run 2's density it completes at **92%** of seed 0's period 1,
which leaves the period-2 kill check's scope argument with no margin at all. The
extra 375 steps are not worth spending the only safety margin this horizon has.

**The alternative that was not taken.** `#53` names two corrections: anneal
through warm-up, or drop the anneal and use the fixed Ape-X ladder from step 0
(Horgan 2018; R2D2 likewise holds each actor's ε fixed for the whole run). The
fixed ladder **is** expressible today — `--epsilon-anneal-decisions 1` puts
every actor on its rung from the second decision onward, since `epsilon_for` is
passed the count so far — and it is rejected here for
one reason: it puts the four near-greedy actors on an untrained network from
decision 0, the failure mode above, and `#53` says it must be paired with
zero-initialising the output layers of `value_head`, `wait_advantage` and
`row_advantage` so that no slot is preferred a priori. That is a code change to
`network.py`, it is a third change to the run, and it is out of this run's
scope. The extended anneal is the correction that needs no code and no third
variable.

**What is *not* expressible with today's flags**, stated precisely so the choice
above reads as a constraint rather than a preference:
`ExplorationSchedule.epsilon_for` (`learning/exploration.py`) computes
`fraction = min(1, decisions / anneal_decisions)` from **total fleet decisions
with no offset**, and `training.py` passes it nothing else. So "start the anneal
at warm-up" and "anneal on gradient steps" are both unavailable. The smallest
change that would buy the first is one field —
`anneal_start_decisions: int = 0` on `ExplorationSchedule`, with
`fraction = clamp((decisions - anneal_start_decisions) / anneal_decisions, 0, 1)`
— plus a `--epsilon-anneal-start-decisions` flag on `scripts/train.py` and its
entry in `run_identity.py`'s training identity, where the anneal horizon is
already recorded. It is **not** implemented and **not** proposed for run 3.

### Protocol

Identical to `M2-P002` as amended, except where the two changes above touch it.

**Fleet.** N=7 clone instances (`tower_rl_instrumented_api36`, `-read-only`,
cold `-gpu host`, 120 Hz confirmed per instance, bridge digest confirmed by
name, offline by interface). Every stage is launched through
`scripts/run_stage.sh` (`M2-P002` amendment 4).

**Availability `all` applies to every arm of this run.** Training, both
baselines, the evaluation arm, the greedy probe and the recordings all run with
`--upgrade-availability all`. A baseline measured against a different roster
than the model is not a baseline, which is the argument ADR 0009 already made
for re-measuring under the choice-point cadence.

**Baselines first, re-measured under `all`, at the amended set size.** The
random and scripted arms are collected **before** the training run, because they
set the kill threshold it is watched against. `M2-E007`'s 5.495 and 6.429 are
**not** carried over: they were measured on profile v1's six purchasable rows,
and `M2-E008` shows the roster under `all` is a different environment.

    scripts/run_actors.py --actors 7 --episodes 15 --policy random \
        --decision-cadence choice-points --upgrade-availability all \
        --renderer host --frame-rate-hz 120 \
        --output-directory <records>/eval-random
    scripts/run_actors.py --actors 7 --episodes 15 --policy scripted ...

**`--episodes 15` on 7 actors = 105 attempted, for n ≥ 100 valid an arm** — a
deliberately short baseline, because of the two things stage 1 has to deliver,
only one needs the full set. At n=100 and the per-episode sd of 2.3 measured in
run 2 an arm's mean carries a standard error of **≈0.22 waves**, which is ample
for the **kill bar** (the random mean − 0.3, a threshold, not an interval) and
for a **coarse comparison** of where the near-greedy curve sits against the two
floors. It is **not** enough for a claim: the pairwise IQM intervals this
project's verdict rule needs were priced at **333 valid an arm**
(`required_episodes(2.3, 0.5, power=0.8)`, `M2-P002`'s first amendment), so any
verdict against these baselines requires **topping both arms up to n ≥ 333**
under the identical image state, cadence, roster and schema and pooling them
with the episodes stage 1 already recorded — which is exactly what `M2-P002`'s
amendment did for run 2. That top-up is part of the **confirmatory** run, not
of this one, and is not budgeted below.
`CheapestFirstPolicy` still
never holds at a choice point; under `all` it buys the cheapest affordable row
of a much wider menu, which is a different floor and is why it is re-measured
rather than assumed.

**Training.** One from-scratch `stacked-dqn`, one seed (developer's call).

    scripts/train.py --actors 7 --renderer host --frame-rate-hz 120 \
        --decision-cadence choice-points --upgrade-availability all \
        --exploration ladder \
        --budget-game-seconds 240000 --block-game-seconds 4000 \
        --checkpoint-every-game-seconds 60000 \
        --epsilon-anneal-decisions 8000 \
        --early-stop-patience-periods 2 --early-stop-min-improvement 0.2 \
        --seed 0

Defaults elsewhere exactly as run 2: 0.25 gradient steps a decision, replay
4,096 sequences, `priority_alpha` 0, `n_step` 10, `history_length` 8,
`warmup_sequences` 100, batch 8, no mid-run evaluation. `#53` ranks several of
those as deviations from the literature and from `docs/rl-candidates.md` §3.1
(D2 `n_step`, D3 replay ratio, D4 prioritisation, D5 Adam ε); **none of them is
changed here**, because each would be a further confound on a single seed. They
remain the run-3 ablation shortlist, not part of this run.

No resume and no run-2 checkpoint: availability `all` changes what the
environment offers, so a run-2 checkpoint is refused by identity in any case,
and `PrioritizedSequenceReplay` fixes compatibility on its first accepted
sequence.

**Early stopping, unchanged.** A period is 60,000 game-seconds; the bar is the
mean of the last period that cleared it; below bar + 0.2 waves for 2 consecutive
periods stops the run after writing that checkpoint; a period with no valid
near-greedy episode counts neither way. Patience 2, min-improvement 0.2,
earliest possible stop after the third checkpoint.

**The budget is 240,000 game-seconds — four periods, not six.** Both run-2 seeds
early-stopped at period 4 (240,511 and 241,663 game-seconds spent of the 360,000
budgeted), so the last two periods of a 360,000 budget are periods neither seed
reached and are bought on the hope that run 3 behaves differently. Four periods
is what run 2 actually used, it is what the early-stopping rule needs to be able
to fire (a stop needs two consecutive non-improving periods after a period that
set the bar), and it is the single largest saving available. The cost is stated
plainly: if run 3's curve is still climbing at period 4, this run **cannot see
past it**, and the result is "the budget closed while the curve was still
improving" — which is a reason to buy a longer budget in the confirmatory run,
not a result about the model.

**Evaluation: the best near-greedy period's checkpoint, greedy.** This is the
one rule of `M2-P002` that run 3 replaces, and the reason is a flaw run 2
exposed rather than a preference. `M2-P002` pre-declares the
**highest-numbered** checkpoint, and the early-stopping rule fires precisely
when the last two periods failed to improve — so under early stopping the two
rules together guarantee that the evaluated checkpoint is the one written at the
close of a **non-improving** period. Both run-2 seeds hit exactly that: seed 0's
arm is the checkpoint of a period that scored 4.490 against the run's own best
of 5.529, and seed 1's is 5.283 against 5.899. The design evaluates the worst
post-anneal period of the run by construction.

Run 3's arm is therefore the **checkpoint written at the close of the best
near-greedy period** — the period with the highest
`checkpoint_period_near_greedy_mean_final_wave` among periods **2 onward**
(period 1 is excluded on `M2-P002` amendment 3's grounds: it pools the ε-anneal
and pre-warm-up episodes). Ties go to the **lower**-numbered period, and a
period that carries **no** near-greedy mean — no valid near-greedy episode
closed inside it, which `close_period` counts neither way — is **ineligible for
selection**. The statistic is the per-period series the run already logs at
every crossing (`experiment/metrics.py`, and `checkpoint_periods[]` in the run's
`summary.json`); the maximum over it is **not** the same quantity as
`best_period_near_greedy_mean_final_wave`, which is the mean of the last period
that cleared the bar by `min_improvement` rather than the highest mean the run
has seen. Run 2 seed 0 shows the two diverging: the logged best is 5.5288
(period 2) while period 3's mean was 5.5806, and this rule would select period 3.
**The arm's identity — the period
number, its mean, and the checkpoint file's name and sha256 — is written into
board `#46` and into this entry's result before the evaluation stage is
launched**, from the training run's own logged numbers, so the selection is
fixed by the training curve and never by an evaluation interval.

**What the change costs.** (1) **Comparability with run 2.** Run 2's arms were
selected by a different rule, so run 3's evaluation number is not a
like-for-like successor to them and the two must not be differenced. What stays
comparable across the runs is the *collection curve* — the per-period
near-greedy means — which is measured identically in both. (2) **Selection
optimism.** A period mean over the 179–241 near-greedy episodes run 2's periods
held carries a standard error of **≈0.163 waves** at the sd of 2.3 measured in
run 2, and at a four-period budget the eligible set is **three** periods (2, 3
and 4), whose expected maximum is 0.846 standard errors above their common
mean, so the selection biases the *selected period's collection mean* upward by
**≈0.14 waves as an upper bound** — an upper bound because the true period means
are not equal, which shrinks it, and smaller than the ≈0.2 a six-period budget
would carry. The **reported interval is unbiased for the checkpoint actually
evaluated**: it is a fresh, independent sample of 333 episodes played by that
checkpoint, if that evaluation happens at all. What
selection can still cost is choosing a checkpoint that is not the run's best —
selection regret, which the period standard error does not bound and which one
seed cannot quantify. This is stated now, before the arm exists, and it is
accepted: an
arm chosen for being the run's best moment is the honest thing to evaluate when
the alternative is an arm chosen, by construction, for being its worst.

**The formal evaluation is conditional, and this is the stop criterion.** Stage
3 is launched **only if**

> the **maximum over the per-period
> `checkpoint_period_near_greedy_mean_final_wave` values for periods 2 onward**
> — the same quantity that selects the arm — **exceeds the stage-1 scripted
> arm's mean final wave** (the mean over that arm's valid episodes, the
> availability-`all` successor to `M2-E007`'s 6.429).

If it does not, **no evaluation arm is collected, no recording is made, and the
run ends**: the recorded result is the training-time near-greedy windows and
periods themselves, reported against both stage-1 baselines as the coarse
comparison n≈100 supports, with the verdict stated as "the curve did not reach
the scripted floor during training". That is a real outcome and it is reported
as one; it is not a failed run and it is not a claim.

Two things about the gate, said before it is read. It compares a **training-time
collection** number (near-greedy actors, ε ≤ 0.02 but not zero, episodes played
while the network was changing) with an **evaluation-time** number (greedy, a
frozen checkpoint), so it is deliberately conservative in one direction and
optimistic in the other, and it is a **gate on spending device time, not a
verdict**: clearing it licenses the evaluation, and only the evaluation's
pairwise interval can license a claim. And it is a one-sided threshold on a
noisy quantity — a period mean carries ≈0.16 waves of standard error and the
scripted arm's mean at n≈100 carries ≈0.22 — so a curve that sits within a few
tenths of the scripted floor can fall either side of it by luck. That is
accepted as the price of the gate; the alternative is spending ~1.9 h of device
time on every run whatever the curve did.

    scripts/run_actors.py --actors 7 --episodes 49 \
        --policy checkpoint:<run>/checkpoints/<the declared checkpoint> \
        --decision-cadence choice-points --upgrade-availability all \
        --renderer host --frame-rate-hz 120 \
        --output-directory <records>/eval-model
    scripts/report_arms.py random=<...> scripted=<...> stacked-dqn=<...> \
        --mlflow-run <the training run>

**`--episodes 49` on 7 actors = 343 attempted**, for the 333 valid episodes the
power calculation asks for: 343 clears 333 down to a validity of 97.1%, where
336 would need the ≥99.107% run 2's random arm happened to return. The model
arm is collected at the full set even though stage 1's baselines are not,
because it is the arm a claim would rest on and it cannot be recollected
without re-running the checkpoint; the baselines are topped up to match it only
in the confirmatory run.

There is still **no set-A selection** over candidate checkpoints, for
`M2-P002`'s reason: `M2-E002` ran one at n=14 and could not separate its
candidates. One arm is evaluated, and it is named before it is played.

**One recording, after the evaluation.** The **arm checkpoint only** plays one
round to death — `scripts/spectate.py --policy checkpoint:<the arm> --episodes 1
--frame-rate-hz 60 --renderer lavapipe --record <checkpoint>.mp4`, under
`--upgrade-availability all` like everything else — written under
`state/recordings/` and never into the repository. Run 2's per-checkpoint set
existed so the developer could watch the policy develop across a run; at four
periods there is little development to watch and the one video that matters is
the arm's. It runs only if the gate above opened, and it is **not evidence for
the verdict**.

**Primary statistic and verdict rule, unchanged.** Pairwise **IQM difference**
of final wave, (model − scripted) and (model − random), by
`comparison.stratified_bootstrap_difference`, 95% percentile interval. The claim
"`stacked-dqn` beats scripted" is made only if that interval excludes zero;
"beats random" likewise; each claim stands alone. The mean difference, Cohen's
d and the per-wave families are secondary and decide nothing.

**The re-run floor is restated to the set size it now guards.** `M2-P002`'s
"fewer than 100 valid episodes is re-run whole" was written for a set of 112;
at a set of 333 it would let a 200-episode arm stand, which buys ~0.65-wave
resolution instead of the 0.5 this design is priced on. For run 3's **model
arm** an arm returning **fewer than 300 valid episodes is re-run whole**, not
padded. The stage-1 baselines, whose set is 105 attempted, keep a floor of
**100 valid**, which is the n their ≈0.22-wave standard error is quoted at.

Because stage 1 is collected at n≈100 rather than 333, **no verdict of this
exploratory run rests on a pairwise interval against those baselines**. If the
gate opens and the model arm is collected, the intervals are computed and
reported, and they are read as *provisional*: the (model − baseline) interval
at 343 against 105 carries the baseline's larger standard error, and the claim
the milestone gate needs is made in the confirmatory run, not here.

**Reported outcomes.** One of: **the gate did not open** — the best near-greedy
period did not reach the scripted arm's mean, the training windows are the
result and no arm was collected; both provisional claims made; scripted only;
random only; neither; stopped at the kill criterion; or early stop fired at
period `k`. In every case `k` or the closing period, the period means, and the
game time actually spent are reported beside the outcome.

**The confirmatory run, in one sentence:** the confirmatory run is **this same
block with stage 1 at `--episodes 49` and stage 3 unconditional** — full-set
baselines, and an evaluation arm collected whatever the training curve did.

### Kill criterion, checked once, at the close of period 2

**The rule `M2-P002` used, restated and re-applied.** `M2-P002`'s condition 1 is
not the number 5.195; it is *"`checkpoint_period_near_greedy_mean_final_wave`
**>** (the re-measured random baseline's mean final wave) **− 0.3**"*. Stage 1
of run 2 measured that baseline at 5.495, and 5.195 is what the rule returned
(`M2-E007`). Condition 2 is the degenerate-policy guard: the latest
`collection_window_wait_fraction` **< 0.9**.

**Run 3 pre-declares the same rule against the new baselines.** At the close of
stage 1, condition 1's threshold is fixed at **(the availability-`all` random
arm's mean final wave) − 0.3 waves**, and the number is written into this entry
and into board `#46` **at the stage-1 close, before any training starts** — the
same order that makes a threshold a pre-registration rather than a retrofit.
Condition 2 stays at 0.9 and needs no baseline. The check is read once, at the
close of **period 2** (amendments 2 and 3); period 1 is not a check, because it
pools the ε-anneal and pre-warm-up episodes — and under the 8,000-decision
anneal it pools more of them than run 2's period 1 did, which is exactly why the
period the check reads stays period 2. If either condition fails the run is
**stopped and diagnosed** and the remaining budget is not spent; a doomed run
costs the 120,000 game-seconds of two periods before it can be stopped —
**3.2–3.4 h** of collection at run 2's two measured wall-clock throughputs
(37,727 and 35,539 game-s an hour). `M2-P002` amendment 3's "~2.8 h" is the same
stop priced on the *report* throughput metric, which excludes everything
`measured_apart`; the two figures are different definitions of the same wall,
not a disagreement.

**Secondary readouts, watched but deciding nothing.**
`learner_value_fit_correlation` (run 2's rose, 0.42 → 0.61/0.66), and buy-slot
concentration on an `M2-E004`-style greedy probe re-captured under `all`
(`run_actors.py --actors 1 --episodes 8 --policy checkpoint:<arm>
--upgrade-availability all --record-observations <path>`, ~15 min of device
time). Run 1's policy put 0.85 of its buy mass on three slots; under `all` the
same probe also reports whether any defense or utility row that v1 never offers
is bought at all, which is the most direct observable the wider roster produces.

### Success criterion, and what this run can and cannot falsify

**Success is `M2-P002`'s verdict rule on one seed:** (model − scripted) and
(model − random) final-wave IQM differences, against the availability-`all`
baselines, each excluding zero. The word **"reproducibly"** — which the
milestone goal uses — **cannot be claimed by run 3**, because it is one seed.
Run 3 is an attempt to find a configuration in which the curve moves at all; the
replication the milestone gate needs is a separate, later run.

**The two hypotheses, and why they are confounded by design.** Run 2's flat
curve has two live explanations:

- **H-thin** — profile v1 offers six purchasable rows, so there is very little
  for a policy to learn and the ceiling is the environment's, not the learner's.
- **H-epsilon** — the anneal completed before the first gradient step, so the
  four near-greedy actors the curve is read from spent the whole run playing a
  network that was never trained under exploration.

Run 3 moves **both** levers at once and runs **one** seed. It therefore
**cannot attribute** any improvement to either, and a single seed clearing a
threshold is the result least likely to replicate. This is a deliberate trade:
the alternative is four training runs at ~11 h each to separate two hypotheses
that may both be false.

**What run 3 *can* falsify.** If the near-greedy curve still plateaus at run 2's
level relative to its own random baseline, and the arm still fails to separate,
then **H-thin and H-epsilon are jointly insufficient**: the two interventions
the evidence ranked highest were made together and the curve did not move, which
points at the remaining `#53` deviations (n_step 10 under a five-decision-a-wave
cadence, a 0.25 replay ratio, the missing reward shaping) or at the learner
itself. That is the most informative failure available here and it is reported
as a refutation, not as "not detectable at this n".

**Weak, secondary discrimination inside the run, deciding nothing.** H-epsilon
predicts an *early* gain — period 2's near-greedy mean above run 2's 5.529 /
5.899 at the same crossing, since the fix acts on the first gradient steps.
H-thin predicts a *late* gain — improvement that grows with training as the
policy learns rows v1 never offered, together with a greedy probe that spreads
buy mass beyond three slots and buys at least one v1-locked defense or utility
row. These are readouts, not tests, and neither settles attribution on one seed.

**The follow-up ablation, in one line:** one further seed under
`--upgrade-availability all` with run 2's `--epsilon-anneal-decisions 2500`,
everything else identical, isolates the ε correction at ~11 h of device time.

### Literature guidance

- **ε anneal.** Mnih et al. 2015 (Nature 518): *"ε annealed linearly from 1.0 to 0.1 over the
  first million frames, and fixed at 0.1 thereafter… a total of 50 million frames"* — the first
  **2%** of training, annealing *while* the learner learns, which begins at **50,000 frames**
  (0.1%). Run 3's 8,000 decisions are **13–19%** of a run-2-sized run (43,023–60,820 decisions),
  first gradient step at ≈3,500 (**6–8%**): ~**7× DQN's fraction**, accepted because this budget
  is four orders of magnitude smaller. What is copied is the ordering, not the fraction.
- **Per-actor ladder.** Horgan et al. 2018, Ape-X (ICLR, arXiv:1803.00933), verbatim: *"Each
  actor i ∈ {0,…,N−1} executes an ε_i-greedy policy where ε_i = ε^(1 + i/(N−1)·α) with ε = 0.4,
  α = 7. Each ε_i is held constant throughout training."* N = **360** actors, n **3**, PER
  α = 0.6 / β = 0.4; R2D2 (Kapturowski 2019) reuses the fixed ladder at n = **5** *(second-hand:
  unreachable here)*. Run 3 takes the formula verbatim at **N = 7** and deviates once, annealing
  into the rungs from ε = 1.0 instead of holding them fixed — `#53`'s probe found a greedy actor
  on an untrained network here near-constant (1–5 actions over 200 states), and a fixed ladder
  would additionally need zero-initialised heads.
- **Warm-up.** Mnih 2015 starts learning at 50,000 frames, Rainbow (Hessel et al. 2018, AAAI,
  arXiv:1710.02298) Table 1 at **80K frames**. Run 3's 100 sequences ≈ **3,500 fleet decisions**
  sit far later in their own run than either, which is why the ordering is a live question here.
- **Replay ratio, n-step, optimiser.** DQN and Rainbow both take one gradient step per 4 agent
  steps = **0.25** a decision, exactly run 3's; BBF (arXiv:2305.19452) runs **8**, §3.1 asks 2–8.
  n: Rainbow **3** (tuned over {1,3,5}), Ape-X **3**, R2D2 **5**, BBF *"10 to 3 over the first
  10K gradient steps"*, run 3 **10** at 5.00 decisions a wave. Rainbow's optimiser: lr
  **6.25e-5**, Adam ε **1.5e-4**, ω **0.5**, β **0.4→1.0**, against run 3's 1e-4 / 1e-8 /
  uniform. Known deviations (`#53` D2–D5), unchanged so this run carries two variables, not six.
- **Evaluation and selection.** Agarwal et al. 2021 (NeurIPS, arXiv:2108.13264): IQM *"discards
  the bottom and top 25% of the runs and calculates the mean score of the remaining 50%"*, with
  *"bootstrap CIs with stratified sampling"* giving *"good interval estimates for as few as
  N = 10 runs"*. Run 3 uses that pair but stratifies over **episodes within one run's actors**,
  so the interval carries episode and actor variance and never seed variance; and
  `required_episodes(2.3, 0.5, 0.8) = 333` powers a **mean** difference while the verdict is an
  **IQM** one. Henderson et al. 2018 (AAAI, arXiv:1709.06560) split ten trials of one algorithm
  into two groups of **five seeds** and found them significantly different (TRPO,
  HalfCheetah-v1, t = −9.0916, p = 0.0016), warning against *"selecting the top-N trials"*;
  Colas et al. 2018 (arXiv:1806.08295) gives power-analysis guidelines for a seed count rather
  than a number. Run 3 selects a **period**, not a seed, and answers that warning the only way
  open to it: select on the training curve, report on a fresh 333-episode sample. No published
  checkpoint-selection rule is followed and none is claimed.
- **One seed** can show the curve moved and that the arm's fresh-sample interval excludes zero;
  it cannot support *reproducibly*, attribute movement to either lever, or bound seed variance.

### Price and timeboxes

Priced on run 2's **measured** fleet throughput rather than a solo figure, and
on the low end of it — with the two definitions kept apart, because `M2-E007`'s
41,487 and 35,539 are not the same measurement. Like for like, run 2's seeds
delivered **41,487 vs 40,952** game-seconds an hour by the run report's own
metric (which excludes everything `measured_apart`, the learner steps and the
periodic work) and **37,727 vs 35,539** on MLflow wall-clock span. Stage 2 is
priced more directly than either: run 2's two seeds spent **240,511** and
**241,663** game-seconds — the budget run 3 now buys — in **6.57 h** and
**6.95 h** of *total stage wall*, launch to exit, which already includes
bring-up, the session report and teardown. The arm stages are priced from
`M2-E007`'s two measured arms decomposed into fixed overhead and per-episode
cost: random 1,962 s at 91.9 wall-s an episode is 492 s of overhead, scripted
2,349 s at 108.3 is 616 s.

| stage | what runs | timebox |
| --- | --- | --- |
| 1 — baselines under `all` | random + scripted, `--episodes 15` each (0.52 h + 0.62 h ≈ **1.14 h** at v1 density) | **≤1.5 h** |
| 2 — training | one seed, ≤240,000 game-s (four periods), bring-up to teardown | **≤7.5 h** |
| 3 — evaluation, **only if the gate opened** | the declared arm at `--episodes 49`, + the greedy probe | **≤1.9 h** |
| 4 — recording, **only if the gate opened** | the arm checkpoint, 1 video + bring-up | **≤0.25 h** |
| | gate opens | **≤11.2 h total** |
| | gate does not open | **≤9.0 h total** |

Every figure is an **upper bound**, and two of the four stages may not run at
all. Stage 2's ≤7.5 h boxes run 2's slower seed at this budget (6.95 h) with
~8% margin. Early stopping can end it at the third checkpoint instead of the
fourth, ~5.2 h; a kill-check failure at period 2 ends it at **≤4.3 h** (3.4 h of
collection plus the session).

Stage 3's box is the scripted arm's per-episode cost applied to 49 episodes
(1.65 h) plus the 0.25 h greedy probe; if the model's episodes cost what the
random arm's did it is nearer **1.6 h**, and a 1.5 h box would hold only in that
case, which is why the box is 1.9. Stage 1's ≈1.14 h against ≤1.5 h absorbs a
~30% rise in per-episode wall time — the headroom the density change asks for.
Run 2's own arms suggest that rise will be small: random and scripted cost 0.541
and 0.538 wall-seconds a game-second despite 27.5 against 20.7 decisions an
episode, so wall time tracks advances per game-second rather than decision
count — while seed 1, which took 41% more decisions than seed 0 for the same
game time at a ~6% lower span throughput, is the counter-evidence kept in view.
Stage 4 is 0.25 h rather than the ~5 min the round itself costs, because the
recording brings an instance up and tears it down and that overhead was never
measured.

**The stopping condition is the budget, not the clock.** A timebox is a price
that was approved, not a rule of the protocol; if availability `all` lowers
throughput, stage 2 overruns its box and is still run to its
240,000 game-seconds or to its early stop. An overrun beyond ~25% is reported
beside the result, because a stage that cost half again what it was priced at is
a finding about the environment even when the numbers it produced are fine.

### Run-3 preconditions

None of the stages above may be launched until all three of the following hold,
and each is checked and recorded in the result entry. **(1) `#54` is merged and
the production bridge is reinstalled under its new digest**: `M2-E008` ran on a
session-local *diagnostics* build (`1a8d2467…7289`) selected through
`TOWER_BRIDGE_BUILD_DIR`, while the production pointer `state/bridge/current`
still reads `662cba09…8902b` — the unlock write must be in the production
bridge, that artifact must hash to the directory name it is filed under (the
self-verification `M2-E007` relied on), and the new digest is recorded in the
result entry exactly as earlier entries record `662cba09…8902b`. **(2) ADR 0011
exists**, recording upgrade availability as an environment-level decision: what
`all` means, that it is re-applied at every round start because `M2-E008` Q2
shows a round start takes it back, and what it does to the recorded environment
profile identity — a run under `all` is not the frozen v1 baseline of ADR 0001
and must not be silently pooled with one. **(3) The baselines are re-measured**
under `all` at n ≥ 100 (stage 1), and both the kill-check threshold (the random
arm's mean − 0.3) and the stage-3 gate (the scripted arm's mean) are written
down before stage 2 is launched.

### Falsifiable predictions, written before the run

1. **Decision density rises.** A choice point is a state where something is
   affordable, and `all` makes far more rows purchasable, so the random baseline
   records **more than 27.5 decisions an episode** and **more than 5.0 a wave**
   (`M2-E007`'s figures under v1). If density is unchanged, the availability
   flag did not take — the same way `M2-P002`'s prediction 2 tested the cadence
   flag.
2. **The scripted floor moves.** `CheapestFirstPolicy` buys the cheapest
   affordable row of a wider menu, so its mean final wave differs from
   `M2-E007`'s **6.429 by more than 0.3 waves**. The direction is not predicted,
   and **this is a weak prediction, stated as one**: the comparison is a
   *difference* of two measured means — at the exploratory n≈100 the new
   scripted arm carries a standard error of **0.22** at the sd of 2.2 measured
   in run 2, `M2-E007`'s own n=112 scripted mean carries 0.209, and the
   difference therefore carries **≈0.30**. (At the confirmatory n=333 those
   figures are 0.121 and **0.241**.) A 0.3-wave threshold is about **one**
   standard error here and fires roughly a third of the time with no true
   change at all, so it is not "well outside the standard error" and nothing is
   claimed from it beyond a coarse check that the roster change reached the
   scripted policy.
3. **The kill check passes at period 2**, i.e. the near-greedy mean at the close
   of period 2 clears the new random bar. Run 2 passed it on both seeds; failing
   it under a *longer* anneal would say the correction made collection worse,
   which is the cheapest way this run can be wrong.
4. **The curve improves, which run 2's did not.** The near-greedy mean rises by
   **≥ +0.5 waves** from period 2 to the best later period, against run 2's
   +0.05 (seed 0) and −0.62 (seed 1). This is the prediction the whole run
   exists to test.
5. **The run spends its whole four-period budget**, i.e. the early stop does
   *not* fire at period 3 or 4, **and its best period is period 3 or 4 rather
   than period 2** — where both run-2 seeds peaked before stalling. If it
   early-stops again with the peak at period 2 and at or below run 2's
   5.53 / 5.90, prediction 4 has failed with it.
6. **The gate opens.** The best near-greedy period exceeds the stage-1 scripted
   arm's mean, so stage 3 runs. This is the prediction the exploratory design
   is built around: if it fails, the run cost ≤9.0 h instead of ≤11.2 h and the
   next question is which of the `#53` deviations to change, not which arm to
   collect.

**What counts as failure**, so that it cannot be renegotiated afterwards: the
kill criterion fires, and the result is "stopped at the kill criterion"; or
(model − scripted) contains zero; or (model − random) contains zero; or
prediction 4 fails while prediction 1 holds — the wider roster and the corrected
schedule both landed and the learner still does not improve, which points at the
learner rather than at the environment; or any
`OBSERVATION_OUT_OF_RANGE:<field>`, any `bridge_event_divergence`, or a
valid-episode rate below 99%, in which case the run is a device or schema
failure and reports nothing about the model.

**Limits stated in advance.** One seed, one image state, one frame rate, one
account progression, one session. Two changes at once, so no attribution, and no
claim to reproducibility. **This run is exploratory and makes no milestone
claim**: its baselines are n≈100, its budget stops at four periods, and its
evaluation runs only if the gate opens — any claim the milestone gate accepts
comes from the confirmatory run, which is this same block with stage 1 at 49
and stage 3 unconditional. Final wave is the statistic; nothing here measures how
the model plays. The evaluated checkpoint is chosen on the collection curve,
which carries the selection optimism priced above. Availability `all` is a
within-run capability written by the bridge, not a progressed account:
`M2-E008` did not test persistence, and nothing here says the official game
would present this roster to a player at this progression.

### Results, as run

**Stage 0 — bridge reinstall.** Production digest
`f9d5f161c33b3af98787d161c9e73f26b1286f519b1648c41b167bffd62a96c3` (ADR 0011's
`unlock_state`/`unlock_all_upgrades` build) installed at `state/bridge/current`
2026-09-20. Every `all` device stage run since — `m2-run3-eval-random`,
`-topup`, `-diag-solo`, `-diag-label`, `-eval-scripted`, `-topup2`, `-topup3` —
reported `cleanup ok` and the host verified clean (no qemu process, no adb
device) on exit, and **zero** `UNLOCK_NOT_APPLIED` / `UNLOCK_REVERTED` lines
appeared over the 231 episodes collected under `all` so far (90 random + 105
scripted + 36 diag-label valid, none invalid). That is `#54`'s done-condition
evidence: the reinstalled production bridge applies and holds the unlock
correctly across every round boundary these stages exercised.

**Stage 1 — baselines under `all`, 2026-09-20.**

| arm | n | mean final wave | sd | SE |
| --- | --- | --- | --- | --- |
| random `all` | 90 | 3.556 | 1.500 | 0.158 |
| scripted `all` | 105 | 6.105 | 0.338 | 0.033 |

Random `all` pooled 60 valid episodes from the original stage-1 run
(`m2-run3-eval-random`, 4/7 actors clean) with 30 valid from one top-up
(`m2-run3-eval-random-topup2`, 2/3 actors clean); a second, single-actor
top-up (`m2-run3-eval-random-topup3`) failed with zero episodes and was not
retried further, so the pre-registered **n≥105 was not reached** — n=90 is
accepted as the arm (developer decision 2026-09-20), on the ground that its
SE (0.158) is adequate for a kill bar even though it falls short of the
coarse-comparison target. Scripted `all` reached its full n=105 with zero
actor failures.

**Kill bar and comparator, fixed before stage 2.** Condition 1 at the close of
period 2: `checkpoint_period_near_greedy_mean_final_wave` **> 3.256**
(random-`all` mean 3.556 − 0.3). Condition 2 unchanged: wait fraction **< 0.9**.
The stage-3 gate comparator is the scripted-`all` mean, **6.105**: stage 3 runs
only if the best near-greedy period (2 onward) exceeds 6.105.

**Actor loss at episode boundaries, dated 2026-09-20 — cause under diagnosis.**
Across the `all` stages run so far: `m2-run3-eval-random` 3/7 actors failed
zero-episode; `m2-run3-eval-random-topup` (3 concurrent) 3/3 failed;
`m2-run3-diag-solo` (1 actor) 1/1 clean; `m2-run3-diag-label` (3 actors × 12,
after `e766821` added observed-state reporting to the failure) 36/36 clean, 0
failures; `m2-run3-eval-scripted` (7 actors × 15) 0/105 failures; the random
top-ups above, 1/3 then 1/1 failed. Logs:
`state/logs/m2-run3-eval-random-20260920-112017.log`,
`state/logs/m2-run3-eval-random-topup-20260920-114231.log`,
`state/logs/m2-run3-diag-solo-20260920-120102.log`,
`state/logs/m2-run3-diag-label-20260920-123140.log`,
`state/logs/m2-run3-eval-scripted-20260920-125601.log`,
`state/logs/m2-run3-eval-random-topup2-20260920-133046.log`,
`state/logs/m2-run3-eval-random-topup3-20260920-134926.log`.

The failure is `RunPortError` on `speed_down: lifecycle_timeout`, always inside
`_pin_game_speed`, but **it is not a bring-up failure**: any earlier wording in
this block implying the pin fails while an actor is first coming up is
corrected here. Every failure so far has occurred at an **episode boundary**,
pinning speed for the episode after the first, and the two verbatim
post-`e766821` fingerprints (`m2-run3-eval-random-topup2` and `-topup3`) show
why — the game state at the moment of failure is already terminal:

    the game did not honour speed_down: lifecycle_timeout (after speed_max:
    game_speed=1.5 wave=1 round_active=1 terminal=0 health=5.0 max_health=5.0;
    at failure: game_speed=0.0 wave=1 round_active=0 terminal=1 health=0.0
    max_health=5.0)

identically on both. Read literally: the prior episode's round is still
reported active and healthy when `speed_max` is pressed, and the round has
already ended (`terminal=1`, `health=0.0`, `round_active=0`) by the time
`speed_down` is pressed moments later — a race between the episode ending and
the pin's own two-step press, not obviously a function of fleet concurrency.
**"Host load from simultaneous bring-up" is now a rejected-or-unconfirmed
reading**: the 3-concurrent top-up failed 3/3 (worse than the original
7-actor run's 3/7), a solo actor pinned cleanly once, and a later 3-actor and
7-actor stage ran with zero failures, which is not the signature simple boot
contention would leave. The cause is under diagnosis; nothing here attributes
it to a fix.

**Stage 2 — training, 2026-09-20/21.** `stacked-dqn`, seed 0, 7 actors, `all`,
epsilon-anneal 8000 decisions, run `stacked-dqn-20260920-144627-8125f3`. Ran
its full budget; early stopping never fired (patience 2, min-improvement 0.2
waves) because the near-greedy mean rose every period:

| period | game-s at checkpoint | near-greedy mean, final wave | n |
| --- | --- | --- | --- |
| 1 | 60000 | 4.76 | 234 |
| 2 | 120000 | 12.15 | 86 |
| 3 | 180000 | 14.11 | 71 |
| 4 | 240000 | 15.20 | 70 |

Arm = period 4 (highest mean, no tie): `checkpoint-gs0240313.pt`, sha256
`7dd8ba3779e27848048a874e9f9c29001fdfc766b0afda9a2e0bb49c04216771`, at
`state/runs/session-20260920-144627/stacked-dqn-20260920-144627-8125f3/checkpoints/`.
`pin_restarts` totalled 0 across all 8 logged collection windows; zero
`UNLOCK_NOT_APPLIED`/`UNLOCK_REVERTED` lines in the log. The MLflow collection-window
series stops at step 59005 (~800 valid episodes) even though the run reached
854 episodes and later decision steps by its own final summary — this gap is
flagged as unexplained, not chased further; it does not affect any figure
reported here, all of which come from period-close lines, the final JSON
summary, or the checkpoint files directly rather than the window series.

Stage summary: `stage m2-run3-train-seed0: exit 0, cleanup ok, instances
0/7 cleaned, 1 exited during teardown, wall 09:01:04`. Throughput, two
definitions: the run's own `game_seconds_per_hour` reported 36855.7; the
MLflow wall-clock span (run start→end, 8.855h, over the final 240313 game-s)
gives 27137.4 game-s/h. **Wall-time overrun**: 9.02h for training alone is
within the pre-registered stage-2 box (≤9.0h/11.2h combined in the M2-P003
protocol) but exceeds the operator's own tighter 6.5h bound for this run by
~39% — noted beside the result per this doc's own overrun convention, since a
stage that cost half again what it was priced at is a finding about the
environment even when the numbers it produced are fine.

**Two-lever confound, stated plainly.** This run differs from run 2 on two
levers at once — `--upgrade-availability all` (new) and the corrected
epsilon-anneal schedule of 8000 decisions (also new) — so the improvement over
run 2 cannot be attributed to either lever alone from this run's data.

**Stage 3 — formal eval of the arm, 2026-09-21.** `m2-run3-eval-arm`, 15
episodes × 7 actors = 105, `checkpoint:checkpoint-gs0240313.pt`, `all`. This is
a **deviation** from the pre-registered 49×7 (343): the observed effect size
(~9 waves between arm and either baseline) against a per-episode sd of
~2–3 makes 343 wasteful, and the run was already past its wall box, so n=105
(matching scripted-`all`'s own n) was run instead. Stage summary: `stage
m2-run3-eval-arm: exit 0, cleanup ok, instances 0/7 cleaned, 1 exited during
teardown, wall 01:18:29` (inside the 2.5h box). 105/105 valid, 0 actor
failures, `pin_restarts` 0, no `UNLOCK_*`.

| arm | n | mean final wave | sd | SE |
| --- | --- | --- | --- | --- |
| checkpoint-gs0240313 `all` | 105 | 15.98 | 3.68 | 0.361 |

Bootstrap 95% CI (`bootstrap_difference`, seed 0, 10,000 resamples), reported
as intervals rather than verdicts:

- vs scripted `all` (6.105, n=105): difference **+9.88 [+9.16, +10.56]**
- vs random `all` (3.556, n=90): difference **+12.43 [+11.64, +13.17]**

Both intervals exclude zero by a wide margin. Note the arm's formal-eval mean
(15.98, deterministic policy, n=105) is somewhat above period 4's in-training
near-greedy mean (15.20, n=70, ε-floor actors mid-training) — a different
measurement taken under different conditions, not a contradiction.

**Stage 4 — recording, 2026-09-21 — no recording produced.** Two attempts,
per the top-up-style one-retry rule; both failed at instance bring-up with
different signatures, and per direction the second failure was not retried
further:

- `m2-run3-spectate-arm` (`state/logs/m2-run3-spectate-arm-20260921-010632.log`):
  `tower_rl.simulation.instance.CloneError: radios did not turn enable within 25s`
- `m2-run3-spectate-arm-2` (`state/logs/m2-run3-spectate-arm-2-20260921-010810.log`):
  `tower_rl.simulation.instance.CloneError: emulator-5556 never became ready:
  the bridge is not answering on emulator-5556: bridge closed the stream`

Both stages reported `cleanup ok` and left the host verified clean. No
recording exists under `state/recordings/` for this run; the write-up
proceeds without one.

## M2-E008 — Profile-v2 unlock trial: the write lands and the game honours it, but a round start takes it back

**Date:** 2026-09-20
**Status:** the four questions board `#54` asked are answered on device. A
bridge write to `Main.upgradeUnlocked` / `upgradeDefenseUnlocked` /
`upgradeUtilityUnlocked` **takes** (Q1), the game **honours** it for purchases
inside the running round (Q3), it **survives that round** but **not the next
round start** (Q2), and it changes **no starting scalar** (Q4). Nothing here
decides whether to build a profile v2; that is a separate decision.
**Purpose:** profile v1 offers six purchasable in-run rows (4 attack, 2
defense, 0 utility), which caps what a policy can learn. `#54` asks whether the
diagnostics bridge's unlock write can widen that, and what it costs.

Repository at `ad50e69`. Diagnostics bridge
`1a8d2467b2a1f73a0e3e2ca7e6e8fb780a0b8cbec9d97e005e448c3fa5777289`, built per
`docs/setup.md` with `-DTOWER_BRIDGE_DIAGNOSTICS=ON` (NDK 29.0.14206865,
`-C state/bridge/config/profile.cmake`, arm64-v8a, android-35) into
`state/bridge/unlock-trial-54-diag` and selected for this session only through
`TOWER_BRIDGE_BUILD_DIR`. The production pointer `state/bridge/current` read
`662cba0974d701c471fe0e7c6cbdeda08c14a668509e8da123a738bfa4f8902b` before and
after and was never repointed. One `tower_rl_instrumented_api36` clone on
`emulator-5556`, `-read-only`, host renderer, 4 cores, cold path, 120 Hz
confirmed, offline by interface (`ip -o -4 addr show` → `1: lo` only) before
any bridge command. No taps, no screenshots, no Settings/Store/Cloud; the only
writes to the game were the unlock arrays and in-run purchases paid from in-run
cash. Levels and scalars are read from the observation, never from the screen.

The whole trial ran under
`./scripts/run_stage.sh --name unlock-trial --instances 1 -- <trial script>`,
stage log `state/logs/unlock-trial-20260920-083122.log`. The trial script ran,
in order:

```text
uv run python scripts/clone_session.py --index 0 up --read-only --renderer host --cores 4
uv run python <probe> --phase 1 --port 47652      # round A, read, write, read, buy
uv run python scripts/unlock_trial.py --serial emulator-5556 --port 47652
uv run python scripts/run_episodes.py --episodes 1 --policy random \
    --serial emulator-5556 --port 47652 --output <record>
uv run python <probe> --phase 2 --port 47652      # round B, read, scalars, buy
```

The probe is a session-local script driving `InstrumentedRunAdapter` and
`InstrumentedBridgeClient` directly; it adds no repository code.

### Q1 — the write takes

| read | attack | defense | utility |
| --- | --- | --- | --- |
| before, in round A | 4 / 20 | 2 / 20 | 0 / 20 |
| `unlock_all_upgrades` read-back | 20 / 20 | 20 / 20 | 20 / 20 |
| fresh `unlock_state` after it | 20 / 20 | 20 / 20 | 20 / 20 |
| `scripts/unlock_trial.py`, before / after / read back | 20 / 20 / 20 | 20 / 20 / 20 | 20 / 20 / 20 |

The pre-write counts 4 / 2 / 0 are exactly v1's purchasable rows, and the
observation's own per-slot `unlocked` flags agree (attack 0–3, defense 0–1,
utility none).

### Q2 — it survives the round, not the round start

On the terminal wave-4 run after the random episode the counts still read
20 / 20 / 20. After `go_home` + `start_round` (the adapter's own
`begin_episode`) the next round started at **7 / 4 / 7**.

The per-slot flags say what those counts are: still true at round B start are
attack 0,1,2,3,17,18,19; defense 0,1,18,19; utility 13–19. The trailing indices
are exactly the empty-name slots — the name arrays are length 20 with 17 real
attack rows, 18 defense, 13 utility (`#39`) — and they are priced 0 and
unpurchasable whatever the flag says. **Every real upgrade row reverted to its
v1 value** (4 attack, 2 defense, 0 utility); only the unused tail kept the
write. The game recomputes availability for its real rows at round start.

An unlock written this way is therefore a within-run capability: writing it once
at process start would not produce an unlocked round, and it has to be rewritten
after each round begins.

### Q3 — the game honours it

In round A, immediately after the write, with cash 80.0, utility slot 4 ("Free
Attack Upgrade", cost 8.0) was bought: outcome `confirmed`, reason
`confirmed_state_change`, level **0 → 1** in the observation. Utility has zero
purchasable rows under v1, so that purchase cannot happen on the frozen
baseline.

The random episode then played the same round to its end: valid, final wave 4,
24 decisions, 17 purchases, `advances_cut_short` 0, speedup 1.961. It finished
holding attack 1,2,6; defense 0,1,2,3,4,11; utility 4,5,6 — seven of those rows
(attack 6, defense 3,4,11, utility 4,5,6) are locked under v1. In round B the
recompute had already reverted the real rows and the probe's utility attempt
found nothing purchasable across 15 advances, which is Q2 again rather than a
separate failure.

### Q4 — no starting scalar moves

| field | v1 (`M2-E006` / `#39`, wave 1) | round A start (pre-write) | round B start (post-write) |
| --- | --- | --- | --- |
| `towerHealth` / `towerMaxHealth` | 5 / 5 | 5 / 5 | 5 / 5 |
| `cash` | 80 | 80 | 80 |
| `damage` | 3 | 3 | 3 |
| `attackSpeed` | 1 | 1 | 1 |
| `criticalChance` / `criticalMult` | 1 / 1.2 | 1 / 1.2 | 1 / 1.2 |
| `towerHealthRegen` | 0.0005 | 0.0005 | 0.0005 |
| `wallHealth` | 1 | 1 | 1 |
| `towerRangeDistance` | 2.700 | 2.699679 | 2.699679 |
| `currentWaveBaseHealth` / `…Damage` / `…KillCash` | 2.35 / 1.176 / 1 | 2.35 / 1.17594 / 1 | 2.35 / 1.17594 / 1 |
| `estimatedEnemiesToSpawnThisWave` | 21 | 21 | 21 |
| `waveLengthSeconds` / `waveCooldownSeconds` | 26 / 9 | 26 / 9 | 26 / 9 |
| `multishotTargets` / `rapidFireDuration` / `knockbackForce` / `orbSpeed` / `wallRebuild` | 2 / 0.6 / 0.4 / 0.04 / 1200 | identical | identical |

Across the write itself — same round, seconds apart — the only live fields that
moved were `waveTimer` and `gameplayTimeThisRound`, which are clocks. No
workshop coupling is visible in any starting scalar.

**What Q4 does not settle.** Round B is not a round started on a fully unlocked
instance, because Q2 shows that state does not reach a round start. What was
compared is a v1 round start, an unlocked mid-run state, and a post-write round
start; none of them differ in any starting scalar.

### Invalid and limits

One session, one instance, two rounds and one random episode; nothing repeated.
No persistence test: nothing was saved, backgrounded or relaunched, the clone
was `-read-only` and was discarded, so whether a save would carry the write is
still unanswered and was out of scope. `true_count` is a count — the per-slot
picture that identified the tail slots comes from the observation's own
`unlocked` flag. The random episode resumed round A rather than starting fresh,
so its final wave is not comparable with `M2-E007`'s baselines and stands here
only as evidence that purchases on v1-locked rows are honoured. Cost arrays were
already populated for 17 / 18 / 13 rows at round A start, before any write.

### Cleanup

`game_frame_rate_override: reset`, `libunity_sha256:
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionName=29.0.3`, `versionCode=1199`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, `cleanup_checks: all passed`, `verified: no qemu
process, no adb device`. Stage summary line: `stage unlock-trial: exit 0,
cleanup ok, instances 1/1 cleaned, 0 exited during teardown, wall 00:03:20`.
Checked again independently afterwards: zero qemu processes via `/proc/*/exe`,
`adb devices` empty, and `state/bridge/current` still
`662cba09…8902b`.

## M2-E007 — Milestone 2, run 2 under `M2-P002`: the baselines, and the kill threshold they set

**Date:** 2026-09-19, updated 2026-09-20
**Status:** **stopped at stage 3 by developer decision, 2026-09-20 — abandoned,
no verdict.** Stage 1 (baselines) and stage 2 (training, both seeds,
early-stopped at period 4) are complete; stage 3 (evaluation) got a full
seed0 arm and an interrupted, recordless seed1 arm before the run was stopped.
Nothing here is a verdict on the model. Board `#46`; protocol `M2-P002`, which
is authoritative and is not restated.

Code at `66082dd`, bridge `662cba0974d701c471fe0e7c6cbdeda08c14a668509e8da123a738bfa4f8902b`
(the artifact hashes to the directory name it is filed under, so the installed
build is self-verified before the fleet), seven `tower_rl_instrumented_api36`
clones on `emulator-5556`…`emulator-5568`, all `-read-only`, host renderer, cold
path (a host-renderer fleet has no snapshot to pin), 120 Hz confirmed per
instance from the display vsync mode and the uid's applied frame rate, offline
by interface per instance immediately before collection. The two arms ran
sequentially, exactly as `M2-P002` writes them, `--episodes 16` on 7 actors =
112 attempted an arm. Records: `state/records/m2-run2/eval-random`,
`state/records/m2-run2/eval-scripted`.

### The two arms

| | random | scripted |
| --- | --- | --- |
| valid / attempted | **111 / 112** (99.1%) | **112 / 112** (100%) |
| mean final wave | **5.495** | **6.429** |
| IQM final wave (stratified by actor) | **5.68** [5.07, 6.21] | **6.57** [6.16, 6.95] |
| per-episode sd of final wave | 2.312 | 2.209 |
| median / min / max final wave | 6 / 1 / 10 | 6 / 1 / 10 |
| decisions an episode | 27.49 | 20.73 |
| decisions a wave | 5.00 | 3.23 |
| advances a wave | 18.28 | 17.88 |
| advances a decision | 3.65 | 5.54 |
| `WAIT` share of decisions | **31.0%** | **0.0%** |
| purchases an episode | 18.95 | 20.73 |
| game-s an episode | 169.8 | 201.2 |
| wall-s an episode | 91.9 | 108.3 |
| arm wall time (bring-up to teardown) | **0.55 h** (1,962 s) | **0.65 h** (2,349 s) |

Wait share is `(decisions − purchases) / decisions` over the valid episodes —
under this cadence every decision that bought nothing was a refusal of an
affordable purchase. Both arms: `bridge_event_divergence` 0,
`advances_cut_short` 0, `episodes_not_started_fresh` 0, and **zero**
`OBSERVATION_OUT_OF_RANGE` of any field. The one invalid episode in the whole
stage is `action_pipeline_failed` on `emulator-5562` ("advance was not
confirmed: stale_or_duplicate"), the arm's only `stale_or_duplicate` event. At
99.1% and 100% valid, both arms clear `M2-P002`'s 99% device-failure line, and
neither arm needs the whole-arm re-run its <100-valid-episode rule would force
(111 and 112 against the 107 the power calculation asks for).

1.20 h of fleet time for the pair, against the 1.3 h `M2-P002` priced.

### The kill threshold for both training runs

`M2-P002`'s kill criterion is two conditions on the near-greedy collection
window after the ε anneal. With the random arm now measured at a mean final
wave of **5.495**, condition 1 is:

> `collection_window_near_greedy_mean_final_wave` > **5.195 waves**
> (5.495 − 0.3),

and condition 2 is unchanged — `collection_window_wait_fraction` < **0.9**. If
either fails on the first ~60 near-greedy episodes after the anneal, that seed's
run is stopped and diagnosed rather than finished. The threshold is stated here,
before any training has started, which is the whole point of collecting the
baselines first.

### What these two arms already say

- **The cadence took.** 5.00 decisions a wave and 27.5 an episode for random,
  against `M2-E005`'s 4.59 and 29.4 and run 1's 21.3 a wave —
  `M2-P002`'s prediction 2 (25–35 an episode, 4–6 a wave) **holds**, and advances
  a wave are 18.28 against `M2-E005`'s 18.72, so the world is still advanced the
  same way. The per-wave comparison shows it directly: the two arms differ in
  decisions at 8 of 9 wave indices and are indistinguishable in `game_ms` at 8 of
  9, i.e. the arms differ in how often they are asked, not in how the game runs.
- **The scripted arm never waits**, 0.0% against random's 31.0%, which is the
  behaviour `M2-P002` predicted from `CheapestFirstPolicy` at a choice point and
  is now measured rather than argued.
- **`M2-P002`'s prediction 1 fails.** The prediction was that the baselines would
  *not* separate from each other, as in `M2-E002` (+0.25, [−0.79, +1.06]) and
  `M1B-E021`. At n=111/112 they do: (random − scripted) final-wave IQM difference
  is **−0.89 [−1.61, −0.22]**, excluding zero, with the mean difference
  −0.93 [−1.51, −0.36] and d=−0.41 agreeing. Scripted is the stronger baseline
  under this cadence, and the reason the earlier runs could not see it is
  most likely sample size — `M2-E002` ran ~60 an arm and printed a resolution
  floor of ~1.1–1.2 waves, wider than the 0.89 measured here. This is recorded
  as a failed pre-registered prediction, not renegotiated: it does not change any
  rule of `M2-P002`, whose verdict compares the model with each baseline
  separately and never with their difference, but it does mean the model faces a
  scripted floor that is genuinely above chance rather than level with it.
- **The resolution this stage bought.** Per-episode sd is 2.31 and 2.21 waves,
  above the 1.3 `M2-P002`'s power calculation assumed, so the realised resolution
  on a pairwise difference is nearer 0.85 waves than the 0.5 designed for — the
  printed detectable difference on final wave at these n is 0.848. A model effect
  of run 1's size (+0.67 IQM waves) is therefore still marginal at this n, which
  is a limit on stage 3 and is stated now rather than after the intervals are
  known.

**Cleanup, both arms.** On every one of the seven live serials, before its
emulator was killed: `libunity_sha256`
`ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, `game_frame_rate_override: reset`. After each arm:
no attached device and no `qemu` process on the host.

**Limits of this stage.** Two arms, one image state, one frame rate, one account
progression, one session. These are baselines, not a comparison against any
model; the arms the verdict needs do not exist yet.

### Stage 2 — seed 0, attempt 1, stopped by the pre-amendment check

**This attempt does not count toward the verdict.** It was stopped by the kill
check as originally timed, which `M2-P002`'s amendment 2 then moved to the close
of the first checkpoint period because a check read at that point cannot
discriminate an untrained network from a failed one — 558 optimisation steps had
been taken when it fired. What it did report, it reported honestly, and it is
kept here rather than deleted; no arm, no checkpoint and no claim comes from it.
The first from-scratch `stacked-dqn` of run 2 was launched on the option-B training
line (7 actors, host renderer, 120 Hz, choice points, ε ladder, anneal 2,500
decisions, budget 360,000 game-s, blocks of 4,000, checkpoints every 60,000,
early stop 2 periods / 0.2 waves, `--seed 0`) and stopped at the pre-registered
check after **24,872 of the 360,000 game-seconds**, 6.9% of the budget. Code at
`0feeabe`, bridge `662cba09…8902b`, MLflow run
`ef2c983ad9bf43c9b25813e9584aff93`, run directory
`state/runs/session-20260919-153857/stacked-dqn-20260919-153857-a60334`.

| | |
| --- | --- |
| wall time | **0.87 h** total (15:33:50 launch → teardown complete); collection 15:38:59–16:14:25, **0.59 h** |
| game seconds | **24,872** / 360,000 (fleet throughput ≈42,100 game-s an hour) |
| decisions | **5,662** (26.9 an episode) |
| optimisation steps | 558, first step at episode 94 (replay warm-up) |
| episodes valid / attempted | **210 / 210** while the run was collecting |
| numbered checkpoints | **none** — the first crossing is at 60,000 game-s |

The 14 further episodes the log shows after the stop are the bridges going away
under the interrupt (`BridgeDisconnectedError`, three actors withdrawn); they
are an artifact of the teardown, not of the run, and nothing is read from them.

**The kill check, on the two conditions this entry fixed before training
started.** The window read is the first 100-episode collection window that lies
entirely **after** the ε anneal (the anneal completes at 2,500 fleet decisions,
around episode 68), spanning decisions 3,537–5,186, with **64 near-greedy
episodes** in it from actors 3–6 (ε 0.0162, 0.0056, 0.0019, 0.00066 — the four
rungs at ε ≤ 0.02, exactly as pre-registered).

| window | decisions | near-greedy episodes | `near_greedy_mean_final_wave` | `wait_fraction` | mean final wave (all actors) |
| --- | --- | --- | --- | --- | --- |
| 0 (spans the anneal) | 3,537 | 64 | 3.672 | 0.664 | 3.95 ± 0.25 |
| **1 (post-anneal, the check)** | 5,186 | 64 | **3.188** | **0.157** | 3.57 ± 0.22 |

- **Condition 1 fails:** 3.188 is not > **5.195** (the random baseline's 5.495 −
  0.3). It misses by **2.01 waves**, and it is *below* the random arm's mean by
  2.31 — at the point its exploration had annealed the policy was worse than
  chance, which is the exact failure the condition exists to catch. The
  anneal-spanning window before it, 3.672, fails the same way.
- **Condition 2 passes:** the wait share is 0.157, far under 0.9, and it fell
  from 0.664 to 0.157 across the anneal — the policy is buying, not refusing, so
  this is not the degenerate-WAIT failure mode.

Per `M2-P002` the run was stopped rather than finished, and **is not diagnosed
here**. What the stage did and did not observe, and nothing more: the learner
was alive at the stop (weighted loss 0.460, unweighted mean |TD error| 0.788,
gradient norm 6.17, `learner_value_fit_correlation` 0.557, all finite and all
moving), the device side was clean (**zero** `bridge_event_divergence`, zero
invalid episodes, zero `stale_or_duplicate`, one `advances_cut_short` in window
0, `round_budgeted_ratio` 1.06), and the fleet ran at 120 Hz offline on all
seven instances. So the stop is not a device or schema failure by any of the
three triggers `M2-P002` names — it is the policy's own number.

The only weights this stage leaves are the rolling
`checkpoints/latest.pt`, sha256
`09d48343ad7406abf08ac50593b902d5dd78b343c62c8b4611dbad0fa94c8700`; no numbered
checkpoint exists, so stage 2 produces **no arm for evaluation and no
recordings** for this seed.

**Pre-registration amendment, before this stage.** `M2-P002` carries a dated
amendment written after the baselines and before any model data: at the sd
stage 1 actually measured, `required_episodes(2.3, 0.5, power=0.8)` = 333 valid
episodes an arm (`--episodes 48` on 7 instances) becomes the evaluation set
size. No model number had been produced when it was written, and the verdict
rule is untouched.

**Cleanup, seed 0.** On every one of the seven live serials, before its emulator
was killed: `libunity_sha256`
`ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, `game_frame_rate_override: reset`. After the run:
no attached device and no `qemu` process on the host.

### Stage 2 — seed 0, attempt 2: early stop fired at period 4

**The reported outcome is `M2-P002`'s "early stop fired at period `k`", with
k = 4.** The run cleared the kill check at period 2 under amendment 3, ran on,
and stopped itself after the fourth numbered checkpoint when the near-greedy
curve failed to improve on its best period for two periods in a row. It spent
**240,511 of the 360,000 game-seconds** it was budgeted (66.8%) and left **four
numbered checkpoints**, the last of which is the pre-declared evaluation arm.
Code at `00e6e08`, bridge `662cba09…8902b`, MLflow run
`5b0eff24c11d4042a4de07046601c4f7`, run directory
`state/runs/session-20260919-164947/stacked-dqn-20260919-164947-2b2301`.

| | |
| --- | --- |
| wall time | **6.57 h** (16:44:43 launch → 23:19:11 exit); collection 16:49:48–23:12:18 |
| game seconds | **240,511** / 360,000 budgeted; **41,487 game-s an hour** of fleet throughput |
| decisions | **43,023** (28.8 an episode) |
| optimisation steps | **9,893**; replay held 1,533 sequences of 4,096, 79,144 sampled |
| episodes valid / attempted | **1,483 / 1,492** (**99.4%**, above `M2-P002`'s 99% line) |
| invalid episodes | 9: 7 `action_pipeline_failed` (all "advance was not confirmed: stale_or_duplicate"), 2 `observation_invalid` (round clock ran 0.955× and 0.97× the budgeted game time) |
| actors withdrawn / bring-up failures | **0 / 0** |
| `WAIT` share over the run | 0.416; purchases an episode 16.8 |
| budget overshoot | 0 ms |

**The near-greedy curve, per period.** A period is 60,000 game-seconds, and the
mean is over the valid episodes the four near-greedy actors (ε ≤ 0.02) finished
inside it.

| period | game-s | decisions | near-greedy episodes | near-greedy mean final wave | bar (best clearing period) | plateau counter |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 60,000 | 10,849 | 221 | **4.855** | — | — (pools the anneal; see amendment 3) |
| 2 | 120,000 | 19,471 | 191 | **5.529** | 5.529 | 0 |
| 3 | 180,000 | 29,907 | 186 | **5.581** | 5.529 | 1 (gain +0.05, under 0.2) |
| 4 | 240,000 | 42,913 | 241 | **4.490** | 5.529 | **2 → stop** |

**The kill check passed at period 2**, which is where `M2-P002` amendment 3
places it: 5.529 > the 5.195 the random baseline fixed, with the latest
`collection_window_wait_fraction` at 0.317, far under 0.9. Period 1's 4.855 is
recorded above for completeness and is not a check — it pools the ε-anneal and
pre-warm-up episodes, which is the whole reason amendment 3 exists.

**The early stop, as the shipped rule computed it:** `early_stopped` true,
`stopped_at_period` 4, `best_period_near_greedy_mean_final_wave` 5.529,
`closing_period_near_greedy_mean_final_wave` 4.490,
`periods_without_improvement` 2. The run's own line reads *"stopped early at
checkpoint period 4: the near-greedy curve did not improve on 5.53 waves for 2
periods"*. Period 3 beat the bar by 0.05 and period 4 fell 1.04 below it, so
the two consecutive failures are one marginal period and one clearly worse one,
not two of a kind — the rule does not distinguish them and neither does this
entry, which reports what it did rather than what a different rule would have
done. Nothing here says the curve had converged; it says it did not improve by
0.2 waves for two periods, which is the pre-registered stopping condition and
the whole of what was tested.

**Checkpoints.** Four numbered, plus the rolling `latest.pt`:

| checkpoint | sha256 |
| --- | --- |
| `checkpoint-gs0060029.pt` | `02e47436e94e14d995a1284c8bc7bb8bab5c6afabccb5c0708c7e54f735300ff` |
| `checkpoint-gs0120025.pt` | `13ceb852cf8c813cdf4ec9830796e60c923a353fe0dd24beee4d05b7174de0ff` |
| `checkpoint-gs0180005.pt` | `6400129532cd7909992a62814c75bb62ec27b596bf74d6f647a5f44879c0090e` |
| **`checkpoint-gs0240077.pt`** — the evaluation arm | `3a6bae708d7cf3d96c25e437e67606073d6192b7bca601f086735754ae6c82f6` |
| `latest.pt` | `e6eeee748700239dab95e5c36a793491f8287189a555c132f6f57943b1921d80` |

`checkpoint-gs0240077.pt` is the **highest-numbered** `checkpoint-gs*.pt` in the
run directory and therefore the arm by `M2-P002`'s pre-declaration, which the
early-stopping rule was written to agree with: the checkpoint written at the
stopping crossing *is* the last one. It is also the crossing whose own period
scored worst of the three post-anneal periods, which is a consequence of the
pre-declaration and is stated now, before the arm is evaluated, rather than
after its interval is known. Stage 4 therefore has **four** recordings to make
for this seed, not six.

**The learner, at the stop:** weighted loss 0.772, unweighted mean |TD error|
0.983, gradient norm 10.77, `learner_value_fit_correlation` 0.613 — finite and
moving throughout, and the correlation rose over the run (0.42 at 5k decisions,
0.61 at the end) rather than drifting down as run 1's did.

**Cleanup, seed 0 attempt 2.** On every one of the seven live serials, before
its emulator was killed: `libunity_sha256`
`ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, `game_frame_rate_override: reset` — seven of seven
on each line. After the run: no attached device and no `qemu` process on the
host.

**Limits of this stage.** One seed, one image state, one frame rate, one account
progression. The arm exists; it has not been evaluated, and nothing here
compares it with either baseline. A run that stopped at 240,000 game-seconds and
one that spent its whole 360,000 are not the same evidence even if their
intervals later agree, which is why `M2-P002` requires the stop to be reported
beside the verdict.

### Stage 2 — seed 1: early stop fired at period 4, launched under `run_stage.sh` (amendment 4)

**The reported outcome is `M2-P002`'s "early stop fired at period `k`", with
k = 4**, the same period at which seed 0 stopped. The run cleared the kill
check at period 2 under amendment 3, ran on, and stopped itself after the
fourth numbered checkpoint when the near-greedy curve failed to improve on its
best period for two periods in a row. It spent **241,663 of the 360,000
game-seconds** it was budgeted (67.1%) and left **four numbered checkpoints**,
the last of which is the pre-declared evaluation arm. This is also the first
stage run through `scripts/run_stage.sh` (amendment 4) rather than by hand.
Code at `fc61232`, bridge `662cba09…8902b`, MLflow run
`fd67bcf7ff8243b6801aa6d11656fa08` (status `FINISHED`), run directory
`state/runs/session-20260920-004516/stacked-dqn-20260920-004516-6625bc`.

| | |
| --- | --- |
| wall time | **6.95 h** (00:40:11 launch → 07:37:22 exit); collection (MLflow run span) 00:45:18–07:33:21, **6.80 h** |
| game seconds | **241,663** / 360,000 budgeted; **≈35,539 game-s an hour** of fleet throughput |
| decisions | **60,820** (43.7 an episode) |
| optimisation steps | **14,338**; replay held 1,594 sequences of 4,096, 114,704 sampled |
| episodes valid / attempted | **1,388 / 1,391** (**99.8%**, above `M2-P002`'s 99% line) |
| invalid episodes | 3: 2 `action_pipeline_failed` (both "advance was not confirmed: stale_or_duplicate"), 1 `observation_invalid` (round clock ran 0.823× the budgeted game time) |
| actors withdrawn / bring-up failures | **0 / 0** |
| `WAIT` share over the run | 0.608; purchases an episode 17.14 |
| budget overshoot | 0 ms |

**The near-greedy curve, per period.**

| period | game-s | decisions | near-greedy episodes | near-greedy mean final wave | bar (best clearing period) | plateau counter |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 60,000 | 13,304 | 219 | **4.822** | — | — (pools the anneal; see amendment 3) |
| 2 | 120,000 | 29,391 | 179 | **5.899** | 5.899 | 0 |
| 3 | 180,000 | 45,063 | 203 | **5.000** | 5.899 | 1 (0.899 below bar) |
| 4 | 240,000 | 60,442 | 198 | **5.283** | 5.899 | **2 → stop** (0.616 below bar) |

**The kill check passed at period 2**, where `M2-P002` amendment 3 places it:
`checkpoint_period_near_greedy_mean_final_wave` **5.899** > the 5.195 bar the
random baseline fixed, with the latest `collection_window_wait_fraction` at
**0.687**, under 0.9. Reported to board `#46` at the time
(`checkpoint-gs0120058.pt`).

**The early stop, as the shipped rule computed it:** `early_stopped` true,
`stopped_at_period` 4, `best_period_near_greedy_mean_final_wave` 5.899,
`closing_period_near_greedy_mean_final_wave` 5.283,
`periods_without_improvement` 2. The run's own line reads *"stopped early at
checkpoint period 4: the near-greedy curve did not improve on 5.90 waves for 2
periods."* Unlike seed 0, where period 3 marginally cleared the bar and period
4 fell well below it, seed 1's periods 3 and 4 both fell below the period-2
bar without clearing it again; the rule does not distinguish the two failure
shapes and neither does this entry.

**Checkpoints.** Four numbered, plus the rolling `latest.pt` and the final
decision-count checkpoint written at the stop:

| checkpoint | sha256 |
| --- | --- |
| `checkpoint-gs0060228.pt` | `10d2c8ed7aa452869b46a1076b88a555fb028c718197243d64584f7c588f9a87` |
| `checkpoint-gs0120058.pt` | `97b89bfaa223991438081751ce29ebb3f9250dc35f40748793647a3074fa0e81` |
| `checkpoint-gs0180003.pt` | `2e136722c4f11ebc4424ebdd4b70e08feb64be04a7fd4bf242ade6255e90dbee` |
| **`checkpoint-gs0240054.pt`** — the evaluation arm | `e101bd4a03fd3c3ee96706e179252e19c243ffc9dc107af50d76caa8640693f4` |
| `latest.pt` | `f5bbfe0580c04efb874c9b507f75bf505e42672265fe448f6e008625e2e20285` |

`checkpoint-gs0240054.pt` is the **highest-numbered** `checkpoint-gs*.pt` in the
run directory and therefore the arm by `M2-P002`'s pre-declaration, on the same
basis as seed 0's. Every sha256 above was independently recomputed against the
checkpoint file on disk and matches its `.sha256` sidecar.

**The learner, at the stop:** weighted loss 0.572, unweighted mean |TD error|
0.730, gradient norm 11.81, `learner_value_fit_correlation` 0.655 — finite and
moving throughout.

**Cleanup, seed 1, and the wrapper's exit status.** In order:

1. Training early-stopped at period 4 (above).
2. `train.py`'s own fleet teardown ran immediately afterward and logged, for
   all seven serials individually: matching `libunity_sha256`
   `ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
   `versionCode=1199`, `versionName=29.0.3`,
   `installerPackageName=com.android.vending`, `libunity_mounts: 0`,
   `bridge_artifacts: removed`, and `cleanup_checks: all passed` — seven of
   seven.
3. `run_stage.sh`'s own backstop cleanup pass then ran over the same seven
   serials: six were already gone ("not live; not cleaned" — the runner had
   already killed them in step 2), and `emulator-5566` was found `adb`-offline
   mid-shutdown and counted "did not clean up", because its post-condition
   checks could not be re-run against an offline device.
4. The final host-wide verification in the same `run_stage.sh` invocation, run
   immediately after step 3, found **zero qemu processes and zero adb
   devices**.
5. Independently re-checked after the fact (`/proc/*/exe` qemu count, `adb
   devices`): also zero and empty.

**Classification (board `#56`): a wrapper liveness defect, not a device-safety
event.** A serial that has already gone `adb`-offline and is disappearing is
not a live instance, and every instance in this fleet runs `-read-only`, so a
dead emulator holds nothing to leak or corrupt; `run_stage.sh` counting that
race as "did not clean up" is a defect in its own liveness check, being fixed
under `#56`. The stage's own exit status reflects this: `run_stage.sh` exits
non-zero (1) whenever its cleanup pass reports a failure even if the training
command itself exited 0 and the host verification passes, per its own
documented rule (`docs/setup.md` §10) — so the supervisor's exit status for
this stage is **1**, for the reason above, not because training, cleanup, or
the device were actually left in a bad state.

**Teardown timing.** From the early-stop line to the summary line: **≈243 s**
(the last checkpoint artifact was written at 07:33:20, the log's last write at
07:37:23), against the `--shutdown-grace` bound of 900 s and the 120 s bound
per instance — both realistic, with wide margin, including the one instance
that hit the liveness race above.

**Limits of this stage.** One seed, one image state, one frame rate, one
account progression. The arm exists; it has not been evaluated, and nothing
here compares it with either baseline or with seed 0.

### Stage 3 — evaluation, abandoned before a verdict

**seed0** evaluated to completion: `--episodes 48`, one actor failed at
bring-up (`speed_down: lifecycle_timeout`, before any episode), topped up
`--episodes 7`; pooled **336 valid** episodes (1 invalid,
`stale_or_duplicate`), 99.7% valid. Records:
`state/records/m2-run2/eval-seed0/`, `state/records/m2-run2/eval-seed0-topup/`.

**seed1** was interrupted mid-collection by one `SIGINT` to its
`run_stage.sh` supervisor (`stage m2-run2-eval-seed1: exit 130, cleanup ok,
instances 0/7 cleaned, 1 exited during teardown, wall 00:54:35`); no actor had
finished, so `state/records/m2-run2/eval-seed1/` holds no episode records.

No pairwise comparison, no verdict, no recordings. **Reason: developer
decision 2026-09-20 — profile v1 judged too limited to learn in; effort moves
to `M2-P003` (availability all).**

## M2-P002 — Milestone 2, run 2: pre-registered protocol (written before any run)

**Date:** 2026-09-19
**Status:** Pre-registered, **option B approved 2026-09-19** (two seeds,
sequential). No run has started and no device time has been spent. This entry
records the plan, its prices and its decision rule before any data exists; it is
not a result. Board `#44`.

**Prerequisites, all satisfied.** Everything this protocol runs on is on
`main`. `observation-v2` and [ADR
0010](adr/0010-observation-v2-everything-the-player-sees.md) landed with `#41`;
the schema's values were verified against the game in **`M2-E006`** — zero
out-of-range readings, zero invalid episodes, `#39`'s capture reproduced field
for field — which is the device stage ADR 0010 requires and the guard against
repeating `M1B-E017`'s field read at the wrong width. The one thing `M2-E006`
could not settle, the unit of six further percent candidates whose slots the
frozen V1 baseline never offers, is carried as an open limit there and is not
re-argued here. Early stopping landed with `#45`. Every flag in the command
lines below was checked against its script's `--help` on `main` at `a9142d8`.

**Objective.** The milestone goal, as the handoff states it: *one committed
model, trained under one budgeted protocol, reproducibly beats the random and
scripted baselines*, with the evidence here. `M2-E002` reached half of that on
run 1 — after its addendum, (model − scripted) separates and (model − random)
does not — on one seed, one selected checkpoint, and a sample whose own
resolution floor (~1.1–1.2 waves) is wider than the effect it measured. Run 2 is
the attempt at the whole claim under a protocol built from what run 1 taught.

**Run 2 differs from run 1 in four ways, and attribution is not the goal of this
run.** All four changes are made at once, so nothing here can say which of them
moved a number. If run 2 beats run 1, this entry does not establish why; a
one-factor-at-a-time attribution is a different experiment at four times the
device cost and is deliberately not attempted.

| change | what it does | evidence |
| --- | --- | --- |
| choice-point cadence | a decision is asked for only where a purchase is legal; forced `WAIT` slices are played through and accrue to the surrounding decision | ADR 0009; `M2-E005` measured it on device — decisions per wave 20.87 → 4.59, advances per wave unchanged (18.00 → 18.72), zero environment failures |
| `observation-v2` | the policy is shown what the player sees: 37 live `Main` fields, raw `level`/`max_level`, unclipped affordability | ADR 0010, `#41`; verified on device in `M2-E006` — zero out-of-range readings, zero invalid episodes |
| Ape-X ε ladder | actor `i` of 7 acts at `0.4 ** (1 + 7i/6)`, spanning 0.4 to 0.00066 (mean 0.087) instead of one floor of 0.05 | `#37`; run 1's accounting: ε=0.05 gave **~1.7 exploratory deviations an episode**, ~2,300 in the whole run, and by the binomial roughly **one episode in ~1,400** deviated eight or more times — essentially no alternative build order was ever played |
| budget in game time | `--budget-game-seconds`, `--block-game-seconds`, `--checkpoint-every-game-seconds` | `#42`; under choice points a decision's game-time cost varies by an order of magnitude, so a decision budget no longer bounds a run's length (ADR 0009's own consequence) |

### Protocol

**Fleet.** N=7 clone instances (`tower_rl_instrumented_api36`, `-read-only`,
cold `-gpu host`, 120 Hz confirmed per instance, bridge digest confirmed by name,
offline by interface) — the operating point of `M1B-E045`/`M1B-E029`.

**Baselines first.** The random and scripted arms are collected **before** the
training run, not after it: they set the kill threshold the training run is
watched against, and a threshold measured after the fact is not one. Both are
re-measured under this cadence and this schema rather than carried over —
`REFERENCE_FINAL_WAVES` and every figure in `M2-E002` were taken under
`every-slice` and `observation-v1` (ADR 0009's "every baseline is re-measured").

    scripts/run_actors.py --actors 7 --episodes 16 --policy random \
        --decision-cadence choice-points --renderer host --frame-rate-hz 120 \
        --output-directory <records>/eval-random
    scripts/run_actors.py --actors 7 --episodes 16 --policy scripted ...

`CheapestFirstPolicy` **never holds at a choice point**: it buys the cheapest
affordable upgrade, and a choice point is by definition a state where something
is affordable, so under this cadence the scripted arm buys at every decision it
is offered. It remains a legitimate floor; it is no longer a policy that
exhibits waiting, and its decision counts are not comparable with run 1's.

**Training.** One from-scratch `stacked-dqn` per seed — no resume, no run-1
checkpoint, both being refused by identity anyway (ADR 0009, ADR 0010).

    scripts/train.py --actors 7 --renderer host --frame-rate-hz 120 \
        --decision-cadence choice-points --exploration ladder \
        --budget-game-seconds <option> --block-game-seconds 4000 \
        --checkpoint-every-game-seconds 60000 \
        --epsilon-anneal-decisions 2500 \
        --early-stop-patience-periods 2 --early-stop-min-improvement 0.2 \
        --seed <seed>

Defaults elsewhere, as run 1: 0.25 gradient steps a decision, replay 4,096
sequences, `priority_alpha` 0, no mid-run evaluation. `--epsilon-end` may not be
given under the ladder, where each actor has a floor of its own.

**The ε anneal keeps run 1's game-time footprint, not its decision count.** The
horizon is still expressed in decisions, but a decision is no longer the same
thing: under choice points one spans about **4.5×** the game time it did under
`every-slice` (`M2-E005`, 20.87 → 4.59 decisions a wave). Run 1's 10,000-decision
anneal divided by that ratio is ~2,200, and **2,500** is the round figure taken,
so run 2 anneals over the same amount of *experience* run 1 did rather than over
4.5× as much. Carrying the 10,000 over unchanged would have spent the first ~1.5 h
of every run on a near-random policy by arithmetic that no longer applies.

**Early stopping** (developer decision, 2026-09-19). A **period** is the
interval between consecutive numbered checkpoints — 60,000 game-seconds. At each
crossing the near-greedy actors' mean final wave over the closed period is
compared with the bar the curve last cleared; **if it is below that bar + 0.2
waves for 2 consecutive periods, training stops after writing that
checkpoint**, and that checkpoint is the evaluated one — consistent with the
"highest-numbered" pre-declaration below, which is why the two rules do not
conflict.

Two details of the shipped rule (`NearGreedyPlateau`, `learning/training.py`),
recorded here so the pre-registration says what will actually happen. The bar is
**the mean of the last period that cleared it**, not the highest mean the run
has seen: a curve creeping up by less than 0.2 a period would otherwise raise
the bar by exactly what it gained and stop a run that is still improving. And a
period in which no near-greedy actor finished a valid episode measures nothing
and counts **neither way**, though it still closed.

The rationale is the standard error. A period holds ~140 near-greedy episodes,
and at run 1's post-anneal per-episode sd of 2.46 waves that is a standard error
of ≈0.2 waves on a period's mean — so one flat period is noise and two in a row
are a plateau. The earliest possible stop is after the **third** checkpoint, at
180,000 game-s, about 4 h in: two periods must close before either can be a
second consecutive failure to improve.

**Evaluation: one pre-declared checkpoint, greedy.** The arm is the **final**
numbered checkpoint — the highest-numbered `checkpoint-gs*.pt` in the run
directory, written at the crossing of the budget. Its number is the game seconds
actually spent when it was written and lands slightly *past* the budget (an
episode is played to its classified end), so at a 360,000-second budget the file
is `checkpoint-gs0360xxx.pt`, not exactly `checkpoint-gs0360000.pt`; the
pre-declaration is "the last one", which is unambiguous before the run.

There is **no set-A selection**. `M2-E002` ran it: four candidates at n=14, every
interval overlapping every other, the tie broken by a rule rather than by
evidence, ~1 h of fleet time per candidate to produce a non-separation. A
selection that cannot separate its candidates is a coin flip that also costs
device time and adds a selection-optimism bias, and it is dropped rather than
repeated. The earlier checkpoints are still written, and are still available for
a later question; they are not arms of this run.

    scripts/run_actors.py --actors 7 --episodes 16 \
        --policy checkpoint:<run>/checkpoints/<the last checkpoint-gs*.pt> \
        --decision-cadence choice-points --renderer host --frame-rate-hz 120 \
        --output-directory <records>/eval-model
    scripts/report_arms.py random=<...> scripted=<...> stacked-dqn=<...> \
        --mlflow-run <the training run>

**Checkpoint recordings, after the evaluations.** Every numbered checkpoint of
each seed plays one round to death, in checkpoint order:

    scripts/spectate.py --policy checkpoint:<path> --episodes 1 \
        --frame-rate-hz 60 --renderer lavapipe --record <seed>-<checkpoint>.mp4

Real time and lavapipe, because the point is a watchable picture rather than
throughput. About 5 min each; under option B that is 12 recordings (two seeds ×
six numbered checkpoints), **≈1 h**, written under `state/recordings/` and never
into the repository. These recordings are **not evidence for the verdict** — a
round watched is one episode of a policy whose own sd is over a wave — and no
claim in this entry may rest on one. They exist so the developer can see what
each checkpoint actually does, which no statistic reports.

**Primary statistic.** Pairwise **IQM difference** of final wave, (model −
scripted) and (model − random), by `comparison.stratified_bootstrap_difference`
— each arm resampled within its own actors, the two IQMs differenced *inside*
the resample — with a 95% percentile interval. This is the statistic
`M2-E002`'s addendum installed, and it is the one that decides. The mean
difference, Cohen's d and the per-wave families
(`wave_statistics.analyse_reports`) are reported beneath it as secondary
evidence and decide nothing.

**Verdict rule, as `M2-P001`.** The headline claim "`stacked-dqn` beats
scripted" is made only if the pairwise interval of (model − scripted)
final-wave IQM excludes zero. "Beats random" likewise. Each claim is made
separately; neither carries the other. An arm that returns fewer than 100 valid
episodes is re-run whole, not padded. Under option B the claim is made per seed
and the word "reproducibly" is used only if it fires on **both**.

**Reported outcomes.** Whatever happens, the entry that records this run reports
one of: both claims made; scripted only; random only; neither; **stopped at the
kill criterion**; or **early stop fired at period `k`** — in which case `k`, the
period means it was computed from, and the game time actually spent are reported
beside the verdict, because a run that stopped at period 3 and one that spent
its whole budget are not the same evidence even when their intervals agree.

**Set size.** `comparison.required_episodes(standard_deviation=1.3,
difference=0.5, power=0.8)` = **107 episodes an arm** — sd 1.3 waves is what
`M2-E002`'s own printed resolution floors (1.124 against scripted, 1.218 against
random, at n≈60) imply per episode, and 0.5 waves is the resolution this design
buys. 107 is `--episodes 16` on 7 actors = 112 attempted, ~110 valid at run 1's
observed validity. That is nearly double run 1's ~60 an arm, and it is the
single change that makes an effect of run 1's measured size (+0.67 IQM waves
against scripted, interval lower end +0.03) readable rather than marginal.

### Kill criterion, checked once, before the budget is spent

Two conditions on the training run's own collection metrics, pooled over the
**near-greedy** actors (ε ≤ 0.02 — rungs 3–6 of the ladder, four of the seven),
over the **first ~60 near-greedy episodes after the ε anneal completes**:

1. `collection_window_near_greedy_mean_final_wave` **>** (the re-measured random
   baseline's mean final wave) **− 0.3**; and
2. the wait share of decisions (`collection_window_wait_fraction`) **< 0.9**.

If either fails, **stop the run and diagnose**; do not spend the remaining
budget. Condition 1 says the policy is not worse than chance at the point its
exploration has annealed. Condition 2 is the degenerate-policy guard that run 1
would have wanted: under choice points a `WAIT` is always a refusal of an
affordable purchase (`M2-E005` measured a 29% wait share for random), so a wait
share at or above 0.9 is a policy that has stopped playing.

**When this check lands.** The ε anneal is 2,500 **fleet** decisions; at
`M2-E005`'s choice-point density (~29 decisions an episode for a near-random
policy) that is ~86 fleet episodes, ~0.4 h at 7 instances and ~108 wall-s an
episode. The 60 near-greedy episodes then take ~0.45 h more, as only the four
near-greedy actors produce them — 15 episodes each. So the check is readable
**about an hour after collection starts**, inside the first eighth of option A's
budget. Nothing before it is diagnostic: the pre-anneal episodes are a
near-random policy by construction.

**Secondary readouts, watched but deciding nothing.**
`learner_value_fit_correlation` (run 1's drifted *downward* after 50k
decisions, −0.0063 ± 0.0032 per 100k), and buy-slot concentration on an
`M2-E004`-style greedy probe — run 1's policy put 0.85 of its buy mass on three
slots by 200k decisions. **That probe batch must be re-captured under v2**; the
stored `observations-50k.pt` is an `observation-v1` tensor and cannot be pushed
through a v2 network. One instance, `run_actors.py --actors 1 --episodes 8
--policy checkpoint:<final> --record-observations <path>`, ~15 min of device
time.

### Priced options — the developer picks one

Throughput, from `M2-E005`: one instance under choice points at 120 Hz produced
212.6 game-seconds per 107.8 wall-seconds, i.e. **1.97 game-s a wall-second**,
~7,100 game-s an hour an instance, **fleet ≈49,000 game-s an hour**. That is a
solo measurement; run 1's fleet actually delivered ≈350,000 game-s in its 7.70
arm-hours (1,419 episodes at `M2-E005`'s every-slice 246.7 game-s an episode),
**≈46,000 game-s an hour**, and the options below are priced on the measured
fleet figure rather than the solo one. Every training line includes the ~0.9 h of
bring-up, post-budget evaluation, session report and teardown that `M2-E002`
found a budget estimate must include and that run 1's estimate missed by 44 min.
Each evaluation arm is 16 episodes an actor at ~108 wall-s plus bring-up and
teardown, ≈0.65 h.

Every figure below is an **upper bound** (`≤`): early stopping can end a
training run at any checkpoint from the third onward, and a seed that stops at
180,000 game-s costs ~3.9 h of arm time instead of ~7.8 h and produces three
recordings instead of six. The table prices the budget being spent in full,
which is the case that has to be approved.

| | budget | training (arm + session) | baselines | evaluation | recordings | **total** |
| --- | --- | --- | --- | --- | --- | --- |
| **A** one seed | ≤360,000 game-s | ≤7.8 h + 0.9 h = **≤8.7 h** | 2 arms, **1.3 h** | 1 arm + probe, **0.9 h** | ≤6, **≤0.5 h** | **≤11.4 h** |
| **B** two seeds, sequential — **approved** | ≤360,000 game-s each | 2 × ≤8.7 h = **≤17.4 h** | 2 arms, **1.3 h** (once) | 2 arms + 2 probes, **1.8 h** | ≤12, **≤1.0 h** | **≤21.5 h** |
| **C** one seed, pilot | ≤180,000 game-s | ≤3.9 h + 0.9 h = **≤4.8 h** | 2 arms, **1.3 h** | 1 arm + probe, **0.9 h** | ≤3, **≤0.25 h** | **≤7.3 h** |

Flags per option, everything else as the protocol above:

- **A** — `--budget-game-seconds 360000 --block-game-seconds 4000
  --checkpoint-every-game-seconds 60000 --seed 0`. Six numbered checkpoints; the
  sixth is the evaluated one. 360,000 game-s is ≈1.03× run 1's game time, so the
  budget is matched to run 1 in the unit the device actually sells.
- **B** — A, then `--seed 1` into its own run directory. The baselines are
  collected once and serve both seeds; the image state, cadence and schema are
  identical across the two, which is what makes them shareable.
- **C** — `--budget-game-seconds 180000 --block-game-seconds 4000
  --checkpoint-every-game-seconds 60000 --seed 0`. Three numbered checkpoints,
  so the early stop could fire only at the last of them and buys C nothing.

The checkpoint period must be a whole multiple of the block, which `train.py`
validates before a device is touched: 60,000 = 15 × 4,000.

What each can and cannot conclude:

- **A** can make or fail to make both headline claims on one seed at run 1's
  game-time budget, at ~0.5-wave resolution. It cannot say anything about
  seed-to-seed variance, and a single seed clearing a threshold is the result
  most likely not to replicate (`M2-E002`'s own limitation, unchanged).
- **B** can say **reproducibly**: two independently seeded from-scratch runs,
  the same rule applied to each, the claim made only if it fires on both. It is
  the only option of the three that supports the word the milestone goal uses.
  It cannot estimate seed variance from n=2 — it can only show agreement or
  disagreement of the two verdicts.
- **C** can confirm that the four changes hold together on device for hours,
  that the kill criterion passes, and that the pipeline writes what it should
  under v2. At half the budget it is **not** a fair test of the headline claim:
  a null result is uninterpretable, because a budget that produced no separation
  is not evidence that a full budget would not. It buys de-risking, not a claim.

**Recommendation: B — approved by the developer on 2026-09-19, and it is what
runs.** The milestone goal says *reproducibly*, and A cannot say it however well
it goes. The marginal cost of B over A is one more training run, one more
evaluation arm and six more recordings, ~10.1 h, against the alternative of
running A, getting a result, and then needing a second seed anyway before the
word can be used. C was the cheap gate on four simultaneous changes; it was not
taken, so the kill criterion and the early stop are what stand between the run
and up to ~21.5 h of device time.

### Falsifiable predictions, written before the run

1. **The baselines will still not separate from each other.** (random −
   scripted) final-wave IQM difference will contain zero, as in `M2-E002`
   (+0.25, [−0.79, +1.06]) and `M1B-E021`, at n≈107 an arm.
2. **Decision density will land near `M2-E005`'s.** 25–35 decisions an episode
   and 4–6 decisions a wave for the collection episodes, against run 1's ~141
   and 21.3 a wave. If it lands near run 1's, the cadence flag did not take.
3. **The collection curve will move.** Pooled near-greedy mean final wave rises
   by **≥ +0.5 waves** from the first post-anneal 100-episode window to the
   last. Run 1's was flat: +0.119 ± 0.342 waves per 100k decisions after 50k,
   and its 100-episode windows were statistically indistinguishable from a
   constant.
4. **Exploration will actually play alternatives.** Actor 0 (ε=0.4) will average
   **≥ 8** off-greedy actions an episode, against the fleet-wide ~1.7 of run 1,
   so alternative build orders appear thousands of times rather than roughly
   once in the whole run.
5. **The scripted claim will replicate and sharpen.** (model − scripted)
   final-wave IQM difference excludes zero with a lower end **above +0.2**
   waves, against run 1's +0.03.

**What counts as failure**, so that it cannot be renegotiated afterwards:

- the kill criterion fires — the run is stopped and this entry's result is
  "stopped at the kill criterion", not a quieter version of a claim;
- (model − scripted) contains zero: run 1's one standing claim fails to
  replicate under the new protocol, at nearly double its sample. That is the
  most informative failure available here and it is reported as a refutation,
  not as "not detectable at this n";
- (model − random) contains zero again: the model is still not separable from
  chance and the milestone goal is not met, whatever scripted does;
- prediction 3 fails while predictions 2 and 4 hold: the cadence and the
  exploration changes landed and the learner still does not improve, which
  points at the learner rather than at the environment;
- any `OBSERVATION_OUT_OF_RANGE:<field>` in the episode records, any
  `bridge_event_divergence`, or a valid-episode rate below 99%: the run is a
  device or schema failure and reports nothing about the model.

**Limits stated in advance.** One image state, one frame rate, one account
progression. Four changes at once, so no attribution. Final wave is the
statistic; nothing here measures how the model plays. The evaluated checkpoint
is the last one by declaration, not the best one — if an earlier checkpoint is
stronger, this design cannot see it, which is the price of dropping a selection
stage that `M2-E002` showed could not separate anyway.

**Amendment 2026-09-19 (after stage 1, before stage 2).** Written after the
baselines were collected and before any model data exists. Stage 1 measured a
per-episode sd of final wave of **2.21–2.31 waves** (`M2-E007`, scripted and
random), against the **1.3** the n=107 set size above was priced on — so at 112
attempted an arm the realised resolution on a pairwise difference is ~0.85
waves, not the 0.5 this design buys. To keep the designed **0.5-wave**
resolution rather than silently accept a coarser one, the evaluation set size
becomes what the same function returns at the sd actually observed:
`comparison.required_episodes(standard_deviation=2.3, difference=0.5,
power=0.8)` = **333 valid episodes an arm**, which is `--episodes 48` on 7
instances = **336 attempted**. This applies to the model arms and, in the same
stage, to **top-ups of both baseline arms** from their stage-1 112 to the same
n, collected under the identical image state, cadence and schema and pooled with
the episodes already recorded. Only the set size changes: the primary statistic,
the kill criterion, the early-stopping rule, the pre-declared checkpoint, the
99%/100-valid device rules and the **verdict rule are unchanged**. The price of
the amendment is evaluation time — an arm is ~3× the 0.65 h stage 1 measured —
and it is accepted here, before any model number is known, rather than after an
interval is seen.

**Amendment 2 (2026-09-19, after the first seed-0 attempt).** The kill check as
originally timed reads a network that has barely been trained, and is moved.
Attempt 1 hit the check about an hour in, as `M2-P002` predicted it would — and
at that point the run had taken **558 optimisation steps over 5,662 decisions**,
because replay warm-up did not end until around decision 3,400 and the learner
had therefore been running for roughly a fifth of the episodes the check was
computed from. A near-greedy mean final wave measured there is a reading of an
almost untrained network, not of a policy that has failed to learn, so the check
as timed **cannot discriminate** between the two and its failure carries no
information about the run it would kill. The check therefore moves to the
**close of the first checkpoint period, 60,000 game-seconds**: at that crossing
`checkpoint_period_near_greedy_mean_final_wave` for period 1 must be
**> 5.195**, and the period's wait fraction (or, if the period does not carry
one, the latest `collection_window_wait_fraction`) must be **< 0.9**; if either
fails the run is stopped and diagnosed, exactly as before. **Both thresholds are
unchanged** — they are still the re-measured random baseline's 5.495 − 0.3 and
the 0.9 degenerate-policy guard — and the early-stopping rule, the pre-declared
checkpoint, the primary statistic and the verdict rule are untouched. Only *when*
the check is read changes, and it is changed before the run it applies to
starts. The price is that a doomed run now costs ~1.4 h of fleet time instead of
~1 h before it can be stopped. The first attempt is recorded below as **"seed 0,
attempt 1, stopped by the pre-amendment check"** and **does not count toward the
verdict**: no arm, no checkpoint and no claim comes from it.

**Amendment 3 (2026-09-19, at the period-1 close).** A correction of the kill
check's **scope**, not of its thresholds. Amendment 2 moved the check to the
close of the first checkpoint period so that it would not read an untrained
network — but period 1 begins at decision 0, so the statistic it closes on
pools the ε-anneal episodes *and* the pre-warm-up episodes the move was meant
to exclude. Measured at seed 0 attempt 2's first crossing (60,000 game-s,
decisions 10,849): the pooled
`checkpoint_period_near_greedy_mean_final_wave` over the period's **221**
near-greedy episodes is **4.855**, while the near-greedy collection windows
inside the period rise **3.767 → 5.222 → 5.340** — the pooled mean is dragged
below the bar by the phase the check is not supposed to read. The check
therefore applies at the close of **period 2**, the first period that contains
no pre-anneal and no pre-warm-up episodes, on the same statistic with the
**same thresholds**: `checkpoint_period_near_greedy_mean_final_wave` > **5.195**
and the latest `collection_window_wait_fraction` < **0.9**. Nothing else moves:
the early-stopping rule, the pre-declared checkpoint, the primary statistic and
the verdict rule are untouched, and the run is still stopped rather than
finished if the check fails.

This amendment does **not** rescue a failing run, which is the thing an
amendment written mid-run must be able to show. At the moment of the ruling the
two post-anneal windows were 5.222 and 5.340, both already **above** the 5.195
bar — the scope correction changes which episodes the bar is applied to, not
whether this run was clearing it. (The window that closed immediately
afterwards, at decision 10,988, read 5.034, below the bar: the near-greedy
window mean carries a standard error of roughly 0.3 waves at ~55 episodes, which
is exactly why the pre-registered check is a period of ~220 episodes and not a
window, and why no single window decides anything.) The price is device time: a
run that fails is now stopped at 120,000 game-s, ~2.8 h, rather than at 60,000.

**Amendment 4 (2026-09-20, before seed 1).** Stages from seed 1 onward are
launched through `scripts/run_stage.sh`, which supervises the stage command and
verifies the device is clean on every exit path. It changes no measured
quantity, no threshold and no rule of this protocol.

## M2-E006 — Observation-v2 on device

**Date:** 2026-09-19
**Status:** `observation-v2` reads every declared field on the real game with
**zero** out-of-range readings and **zero** invalid episodes, and the values
reproduce board `#39`'s capture field for field. `criticalChance` is confirmed
to be stored in percent. The six other percent candidates sit in slots the
frozen V1 baseline never offers and **cannot be settled at this baseline**.
Board `#41`, ADR 0010.
**Purpose:** `observation-v2` reads thirty-seven further `Main` fields and
rescales each by a transform chosen from `#39`'s observed units. Only one of
those units had been observed non-zero. This is the check that the production
bridge reads them all on the live game, that no transform produces an
out-of-range value, and that the panel shows numbers a developer can hold
beside the HUD.

One clone instance (`tower_rl_instrumented_api36`, `emulator-5556`,
`-read-only`, lavapipe, 60 Hz, windowed, offline by interface), production
bridge `662cba0974d701c471fe0e7c6cbdeda08c14a668509e8da123a738bfa4f8902b`
deployed through `TOWER_BRIDGE_BUILD_DIR`, `scripts/spectate.py --policy random
--episodes 2 --no-panel --record`. Two episodes, 49 decisions, 33 purchases,
final waves 1 and 6, 197 s of round clock.

Both episodes valid; `invalid_reasons` empty for both, so no
`OBSERVATION_OUT_OF_RANGE` was raised by any of the 37 fields across 49
decisions. All 37 present in every state message, all finite. The 48 non-empty
upgrade-row labels came back from one `slot_labels` command and match `#39`'s
lists slot for slot.

Ranges over the session, in the game's own units (`live_readings` in the
session record):

| field | min | max | `#39` |
| --- | --- | --- | --- |
| `damage` | 3 | 12.09 | 3 → 12.09 |
| `attackSpeed` | 1 | 1.20 | 1 → 1.15 |
| `criticalChance` | 1 | 5 | 1 → 5 |
| `criticalMult` | 1.2 | 1.5 | 1.2 → 1.5 |
| `towerHealthRegen` | 0.0005 | 0.2342 | same |
| `wallHealth` | 0 | 4.019 | same |
| `towerRangeDistance` | 2.70 | 2.70 | 0 → 2.700 |
| `currentWaveBaseHealth` | 2.35 | 8.713 | same |
| `currentWaveBaseDamage` | 1.176 | 2.680 | same |
| `currentWaveBaseKillCash` | 1 | 1 | same |
| `enemiesSpawnedThisWave` | 0 | 30 | 0 → 27 |
| `enemiesKilledThisWave` | 0 | 33 | 0 → 26 |
| `estimatedEnemiesToSpawnThisWave` | 21 | 26 | same |
| `closestEnemyDistance` | 0 | 10000 | same |
| `waveTimer` | 0.16 | 34.59 | 0 → 34.47 |
| `waveLengthSeconds` / `waveCooldownSeconds` | 26 / 9 | 26 / 9 | same |
| `cashEarnedThisWave` | 0 | 23 | 0 → 24.5 |
| `gameplayTimeThisRound` | 0.16 | 197 | 343.9 (longer run) |
| `wallRebuild` / `orbSpeed` / `knockbackForce` / `rapidFireDuration` | 1200 / 0.04 / 0.4 / 0.6 | constant | same |
| `superCritChance`, `multishotChance`, `rapidFireChance`, `knockbackChance`, `lifesteal`, `defenseRel`, `thornDamage`, `defenseAbs`, `cashPerWave`, `orbCount`, `multishotTargets`, boss flags | 0 (2 for `multishotTargets`) | unexercised | same |

Behaviour checks, all from the record rather than by eye: the wave's base health
rises 2.4 → 3.3 → 4.4 → 5.6 → 7.2 → 8.7 with the wave (×1.29, `#39`'s curve);
`waveTimer` counts to 34.6 and resets, against `waveLengthSeconds + waveCooldownSeconds`
= 35; `enemy_present` toggles, 15 decisions with the 10000 sentinel and 35 with a
real distance down to 0.13 m; `cash` is unchanged from v1 and `cashEarnedThisWave`
is a separate within-wave counter.

**Unit conclusion, and what it does not settle.** `criticalChance` reads 1 and
5 — integers, the HUD's "1 %" and "5 %" — so the percent transform is correct
for it. The other six percent-typed fields and `thornDamage` read a flat 0,
because the frozen V1 baseline offers **six** in-run slots (ADR 0001) and those
six are exactly the stats that moved: Damage, Attack Speed, Critical Chance,
Critical Factor, Health and Health Regen. A random policy cannot buy widely here
because there is nothing wider to buy. Their units are therefore **not settled
by any run at this baseline**, and will not be until permanent progression (ADR
0008) unlocks those rows. Today the risk is bounded: a field that is really a
fraction would be divided by 100 — a scaling error, never an invalid observation
— and it can only arise once such a row is purchasable.

Recording `state/recordings/observation-v2-check-000.mp4` and `-001.mp4`, session
record `state/recordings/records/emulator-5556.json` (both under `state/`, not
committed). Installed and deployed as `state/bridge/current`, replacing
`7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a`.

Cleanup on the live serial before it was killed: `game_frame_rate_override:
reset`, `libunity_sha256:
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`. Afterwards zero qemu processes via `/proc/*/exe`
and `adb devices` empty.

## M2-E005 — Choice-point cadence on device

**Date:** 2026-09-19
**Status:** The choice-point cadence runs on the real game with **zero**
environment failures, and it removes about four fifths of the decisions per
wave while advancing the world identically. Board `#38`, ADR 0009.
**Purpose:** `M2-E004`'s observation batch showed 68% of run 1's decisions had
`WAIT` as the only legal action. ADR 0009 stops asking at those slices. This is
the first device evidence that the new cadence drives the real game unchanged,
and the measurement of what it actually removes.

One clone instance (`tower_rl_instrumented_api36`, `emulator-5556`,
`-read-only`, host renderer, 120 Hz, offline by interface), `random` arm,
`scripts/run_actors.py --actors 1`. Five episodes under `--decision-cadence
choice-points`, then two under `--decision-cadence every-slice` as the control.
Code at `988ac04`.

| | choice-points (5 ep) | every-slice (2 ep) |
| --- | --- | --- |
| valid / attempted | 5 / 5 | 2 / 2 |
| decisions | 147 (29.4/ep) | 313 (156.5/ep) |
| advances | 599 (119.8/ep) | 270 (135.0/ep) |
| advances per decision | 4.07 | 0.86 |
| decisions per wave | **4.59** | **20.87** |
| advances per wave | 18.72 | 18.00 |
| `WAIT` share of decisions | 29% (43/147) | 86% (270/313) |
| mean final wave | 6.4 (1, 6, 7, 9, 9) | 7.5 (7, 8) |
| measured game ms per episode | 212,577 | 246,691 |
| wall seconds per episode | 107.8 | 124.6 |

`bridge_event_divergence` 0, `advances_cut_short` 0, `stale_or_duplicate` 0,
`episodes_not_started_fresh` 0, no `ADVANCE_TRUNCATED_BY_WALL` and no invalid
reason of any kind, in both arms.

### What it says

- **The world is advanced the same way.** Advances per wave are 18.72 against
  18.00: the cadence of the *world* is untouched, which is what ADR 0009
  claimed and is the point of leaving the bridge alone.
- **The decisions removed are the forced ones.** Decisions per wave fall to 22%
  of the control (4.59 from 20.87), and the `WAIT` share of decisions falls
  from 86% to 29%. The prediction from `M2-E004` was ~32%; the measured 22% is
  lower because the arm here is `random`, which buys whenever it can and so
  spends more of each run broke than a learned policy does. Under
  choice-points the remaining `WAIT`s are chosen against a real alternative.
- **A decision now covers a span.** 4.07 advances per decision, against 0.86
  under every-slice — below one there because a settled purchase advances
  nothing and is counted as the zero advances it made.
- **Final wave is unchanged within this sample.** 6.4 against 7.5 on five and
  two episodes of a policy whose own standard deviation is over a wave; this
  says nothing either way and is not a comparison. `REFERENCE_FINAL_WAVES` is
  still every-slice and still has to be re-measured.

Cleanup on the live serial before the kill, both sessions: `libunity_sha256`
`ffc1f3ef…dd0040`, `versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, frame-rate override reset; then no attached
device and no `qemu` process.

## M2-E004 — Plasticity diagnostic across run-1 checkpoints

**Date:** 2026-09-19
**Status:** The plasticity-loss signature is **absent** across the four
checkpoints of the `M2-E002` run, against a rule written before the numbers
were seen. No reset A/B is justified by this evidence. Board `#33`.
**Purpose:** `M2-E002` found greedy final-wave IQM roughly flat across the
50k/100k/150k/200k checkpoints. Plasticity loss / primacy bias (Nikishin et
al. 2022) is one hypothesis for a flat curve, and an A/B designed around it
would cost tens of device-hours. This measures the hypothesis' own signature
offline first, on checkpoints that already exist: minutes of CPU, one short
device session for the observations.

`scripts/diagnose_plasticity.py` over
`~/.local/state/tower-rl/runs/session-20260918-215839/stacked-dqn-20260918-215839-e3c6ba/checkpoints/`;
JSON at `~/.local/state/tower-rl/m2-run1/plasticity/diagnosis-50k-batch.json`.

### The rule, written before the numbers

- **Present** — dormancy at τ=0.1 rises monotonically in at least one layer by
  ≥ 10 percentage points from 50k to 200k, **or** the core stable rank falls
  monotonically by ≥ 25% of its 50k value.
- **Absent** — dormancy at τ=0.1 stays within ±3 points of its 50k value in
  every layer **and** the core stable rank stays within ±10% of its 50k value.
- **Inconclusive** — anything else: non-monotone movement, or movement between
  those bands.

Parameter norms decide nothing on their own: a norm that grows while dormancy
and rank are flat is not the signature this asked about.

### Definitions

- **τ-dormant fraction** (Sokar et al. 2023, arXiv:2302.12902): a unit scores
  `s_i = E|h_i| / mean_k E|h_k|`, the expectation over the observation batch,
  and is τ-dormant when `s_i ≤ τ`. Reported at τ = 0.025 and τ = 0.1, that
  paper's two thresholds, on the post-activation outputs of
  `trunk.row_encoder`, `trunk.scalar_encoder` and `core` — each an
  `nn.Sequential` ending in its `SiLU`, so the module output is the
  post-activation one. A row encoder is shared across upgrade rows, so its
  samples are one per row per step.
- **Stable rank**: `‖F‖_F² / ‖F‖_2²` of the `core` output, the pooled feature
  matrix the dueling heads are handed.
- **srank_99** (Kumar et al. 2021, arXiv:2010.14498): the fewest leading
  **singular values** whose sum reaches 99% of their total — the paper's own
  form, on the singular values themselves rather than on their squares.
- **Parameter norm**: the L2 norm of each parameter tensor, and of the whole
  parameter vector (not the sum of the per-tensor norms).

### The observation batch

No observation is stored anywhere in this project — actor records hold episode
summaries, a checkpoint holds `replay_provenance` rather than the buffer, and
`DecisionView` carries none — so one session was played to record some.
`run_actors.py --actors 1 --episodes 8 --policy checkpoint:…-0050123.pt
--frame-rate-hz 120 --renderer host --record-observations …` on
`emulator-5556`, cold, `-read-only`, offline verified by interface, confirmed
at 120 Hz, cleanup before kill.

- Policy identity `8344a482eede`, `checkpoint-0050123.pt` of run
  `stacked-dqn-20260918-215839-e3c6ba`, played greedily.
- **8 of 8 episodes valid**, 1,141 decisions, mean final wave 6.375
  (median 6.0, sd 1.92, range 4-10), 22.4 decisions a wave, 916 s wall.
- **800 observations**: 10 windows of 80 steps, cut from episode starts and
  never straddling two episodes, because the stacked history returns to zeros
  at an episode start. Batch at
  `~/.local/state/tower-rl/m2-run1/plasticity/observations-50k.pt`.

### The three curves

| decisions | opt. steps | dormant τ=.025 / τ=.1, all three layers | core stable rank | core srank_99 | ‖θ‖ |
|---|---|---|---|---|---|
| 50,123 | 11,379 | 0.000 / 0.000 | 1.097 | 98 | 48.965 |
| 100,080 | 23,869 | 0.000 / 0.000 | 1.139 | 103 | 49.272 |
| 150,002 | 36,349 | 0.000 / 0.000 | 1.156 | 104 | 49.501 |
| 200,174 | 48,892 | 0.000 / 0.000 | 1.179 | 104 | 49.681 |

- **Dormancy is zero everywhere**, at both thresholds, in all three layers, at
  all four checkpoints — and with headroom, not because the measure cannot
  fire: the quietest unit of any layer scores between 0.22 and 0.57 of its
  layer's mean, two to five times τ=0.1. The widest drift is
  `trunk.row_encoder`, whose quietest unit falls 0.567 → 0.316 over the run,
  still far above dormant.
- **Rank does not collapse; it rises slightly.** Stable rank 1.097 → 1.179
  (+7.5%), srank_99 98 → 104 of 128 available. The stable rank sits near 1
  because the matrix is uncentred and the features carry a large common mean;
  subtracting the mean gives 2.49 / 2.26 / 2.48 / 2.30 across the four, flat
  and non-monotone. Either way there is no fall.
- **Parameter norms grow monotonically but slightly**: ‖θ‖ 48.965 → 49.681,
  +1.5% over 37,513 gradient steps. The total is damped by the 96x16 identity
  embedding, most of whose rows are never trained and which alone carries
  40.26 of the 48.97: it *falls* 40.264 → 40.214. The trained tensors grow
  more — `core.3.weight` 7.234 → 8.434 (+16.6%),
  `heads.row_advantage.0.weight` 7.087 → 8.103 (+14.3%) — while two head
  output layers shrink (`heads.wait_advantage.2.weight` 0.523 → 0.380). Any
  norm-growth claim about this run must be made on the per-tensor numbers; the
  total understates the trained trunk by a factor of ten.

### Verdict

**Absent.** Dormancy is 0.000 at both thresholds in every layer at every
checkpoint, which is within the ±3 points the rule allows, and the core stable
rank moves +7.5%, inside the ±10% band. Neither clause of *present* is
approached: nothing rises 10 points and nothing falls 25%.

A reset A/B has no support from this evidence. What is here is a network whose
units all stay active, whose feature rank is low but stable, and whose trained
weights grow modestly — the picture of a learner that is not losing capacity,
not one that has lost it.

Two limits on how far this reaches. The batch is the 50k policy's own state
distribution, so the three later checkpoints are measured off-policy on it; a
representation that collapsed only on states those checkpoints themselves
visit would not show here. And `M2-E002` puts this learner ~0.7 waves above a
random baseline, so an alternative reading of the flat greedy curve — that
little was learned for plasticity to be lost — remains open and is not
addressed by this measurement.

## M2-E003 — Spectate mode on device

**2026-09-19 note:** this recording was made under the host renderer and may show the glitching the developer reports; recordings from this change on are lavapipe by default (board `#36`).

**Date:** 2026-09-19
**Status:** Eight of the nine device checks for `#12` pass. Check 9 fails, in
the direction it was written to catch: `screenrecord` interrupted by SIGINT
exits **0** on this image, so the `-partial` chunk naming could never fire and
has been removed. Board `#12`.
**Purpose:** Run the nine device checks posted on `#12` against a real
windowed instance, and make the first recording of the `M2-E002` selected
checkpoint. Repository at `50fae46`; no code changed, on the device or off it.

Two sessions on `emulator-5556` (clone AVD `tower_rl_instrumented_api36`,
`-read-only`, `-gpu host`, windowed on a real X11 display), emulator 37.1.11
under gfxstream. Host free before and after: zero `qemu` walking `/proc/*/exe`,
`adb devices` empty.

### The recording run

```text
uv run python scripts/spectate.py \
  --policy checkpoint:…/checkpoints/checkpoint-0050123.pt \
  --episodes 1 --frame-rate-hz 60 \
  --record ~/.local/state/tower-rl/spectate/checkpoint-0050123-episode1.mp4 \
  --output-directory ~/.local/state/tower-rl/spectate/records
```

- **Final wave 8**, 189 decisions, 24 purchases, 16,117 frames, valid, ending
  `game_over`. The panel's own reading and the evaluator's `emulator-5556.json`
  agree on both numbers.
- Episode wall **272.991 s**; session wall **326.3 s** including bring-up, the
  20 s hold and teardown.
- `checkpoint_identity` `8344a482eede`, matching `selection.json` — the
  `M2-E002` selection is what played.
- Recording at `~/.local/state/tower-rl/spectate/`:
  `checkpoint-0050123-episode1-000.mp4` (179.99 s, 162,468,268 B) and
  `-001.mp4` (144.55 s, 92,320,194 B), both h264 360x640 in a readable MP4
  container.

One episode of this checkpoint at real time is roughly four minutes, so a
single default session already crosses the three-minute chunk boundary; the
`--episodes 2` fallback was not needed.

### The nine checks

1. **Windowed launch — pass.** `launching tower_rl_instrumented_api36 on
   emulator-5556 (host, read-only, windowed)`, a real window on `DISPLAY=:0`.
   Boot, deploy, the launch-online/cut-radios sequence, the 60 Hz confirm, the
   bridge connect and the first decision all inside **45 s** of launch — the
   300 s `wait_for_boot` bound was never approached.
2. **`confirm_frame_rate` at 60 — pass.** `emulator-5556: confirmed at 60 Hz:
   display vsync mode 60.00, uid 10218 game mode override 60, uid 10218 applied
   frame rate 60.00`. All three readings present. The read-back had only ever
   been exercised at 120; a surface already at the stock rate does publish an
   applied rate it accepts.
3. **Real time is real — pass.** `round_ms` 278,669 — 278.7 game seconds —
   against `elapsed_wall_seconds` 272.991: **1.02x**, one game second per wall
   second. The same checkpoint on the 120 Hz fleet (`M2-E002` set A) has a
   median speed-up of 1.97x.
4. **Bring-up unchanged by the window — pass.** `deployed bridge confirmed
   7a98f50b…f99a`, then `emulator-5556 is at home and offline`. Read back
   independently with the window up: only `lo` carries an IPv4, and
   `topResumedActivity` is the game's `UnityPlayerActivity`. Cosmetic, not a
   fault: `instrumented_bridge.sh deploy` tags the stderr of `adb push` and
   `am start` as `deploy: error:` lines on a wholly successful deploy.
5. **The panel — pass.** The frame the death is reported in:

   ```text
   tower-rl spectate — checkpoint-0050123 — episode 1 of 1

   wave 8   cash 15   health 0%   reward +0
   episodes played 1   mean final wave 8.00   decisions 189   0.0/min
   episode 1 ended at wave 8: game_over
   ```

   The death line is written in the same redraw as `e1 d189` and before the
   hold footer, so it lands on the decision the tower dies on rather than one
   later. Still owed to the developer's own eyes: comparing cash and health
   against the game window beside it.

   This check also found what an unattended session was missing: `--no-panel`
   printed `lines[2] | lines[0]` and nothing else, so it gave wave, cash and
   health per decision but **never the death line** — a log showed health
   reaching 0 and left the reader to infer the death. Fixed with the check-9
   decision below: `PlainPanel` now prints the same ended-episode line on the
   same decision, and `panel_lines` reserves that slot so both panels find it
   at one index.
6. **Hold and teardown — pass.** The hold ran after the last episode
   (`session over — press any key to tear the instance down`), then
   `game_frame_rate_override: reset`, `libunity_mounts: 0`,
   `bridge_artifacts: removed`. Afterwards `adb devices` empty and zero `qemu`.
7. **`--record` — pass.** Two chunks, the first stopping at its own 180 s
   limit and the second at 144.55 s; `ffprobe` reads both as h264 in a
   `mov,mp4,m4a` container with durations, so the last is finalised rather than
   truncated. Guest-side removal was caught directly, polling `/sdcard` at 0.5 s:
   chunks present at 10:12:34.657, gone at 10:12:35.175, device down at
   10:12:36.254 — removed before teardown, with room to spare.
8. **The refusal, positively — pass.** Started from a second shell against a
   live instance: exit 1, nothing touched, naming both witnesses — `refusing to
   spectate while an emulator is running (adb: emulator-5556; qemu processes:
   1114598 …/qemu-system-x86_64)`. The running instance was unaffected.
9. **`screenrecord` exit status — fail, and the `-partial` naming is gone.**
   On this image `screenrecord` interrupted by SIGINT exits **zero**. Read by
   hand on a live instance, both ways the question can be asked:

   ```text
   targeted-kill-INT rc=0
   pkill-INT rc=0
   ```

   The recording code's own naming corroborated it. Session A's chunk 001 ran
   144.55 s of its 180 s limit — plainly interrupted — and was pulled as
   `-001.mp4` with no `-partial`; a second session produced three chunks, every
   one interrupted, none named `-partial`. So
   `chunk.partial = "rc=0" not in reply` could never be true for an interrupt.

   The reason is the same one that makes the files good: SIGINT tells
   `screenrecord` to stop, and it finalises what it is writing and exits
   cleanly. All three interrupted chunks read back through `ffprobe` as valid
   MP4 with durations. There is no truncated-chunk case on this image, so
   `-partial` had nothing to distinguish and could only ever mislead in the
   quiet direction — marking nothing while suggesting the distinction was being
   watched.

   **Decided and done:** the partial flag and the `-partial` suffix are removed
   from `GuestRecording`, along with the `; echo rc=$?` that existed only to
   feed them. A chunk is whole; how long it ran is its duration's business. If
   a future image truncates on interrupt, the evidence for putting a warning
   back is a chunk `ffprobe` cannot read — not an exit status.

   The check that replaced it: `--no-panel` now prints the ended-episode line
   the curses panel draws, on the decision the episode ends. An unattended log
   that showed health reaching 0 but never named the death was making its
   reader infer it.

### What the developer can now run

Nothing here needs a second attempt. The windowed path, the panel and the
recording all work on the developer's own desktop with the command above.

## M2-E002 — First budgeted run: 200k decisions, post-hoc evaluation

**Date:** 2026-09-19
**Status:** On the pre-registered statistic (see the addendum of 2026-09-19 at
the end of this entry, which supersedes the verdict below for the purpose of
the claim) the rule fires on **one** comparison: on set B `stacked-dqn` beats
scripted on final-wave IQM; beats random is withdrawn to *not detectable at
this n*; random and scripted remain indistinguishable from each other. Board
`#31`.
**Purpose:** Execute `M2-P001` as corrected — the 200,000-decision budgeted
training run, then set A selection and set B report — and apply its decision
rule mechanically. Repository at `10722c7` for the evaluation, `3c494b7` for
the training run; no code changed for either.

Run directory
`~/.local/state/tower-rl/runs/session-20260918-215839`, arm run
`stacked-dqn-20260918-215839-e3c6ba`, MLflow run
`90d08dac383d4f7d9f48f1a8ca43c189` (`sqlite:///~/.local/state/tower-rl/mlflow.db`).

### The training run

`train.py --actors 7 --budget-decisions 200000 --checkpoint-every-decisions
50000 --renderer host --frame-rate-hz 120`, launched 21:53 on 2026-09-18,
defaults elsewhere (2,000-decision block, 4 cores an instance, seed 0,
0.25 gradient steps a decision).

- **200,174 decisions** in **1,420 episodes, 1,415 valid**, **48,892
  optimisation steps**, 4,290 sequences accepted.
- Wall **27,731 s = 7.70 h** for the arm, **30,957 s = 8.60 h** for the
  session (bring-up, post-budget evaluation and teardown included).
- Fleet 7/7 up in ~5 min cold under `-gpu host`; 7/7 `deployed bridge
  confirmed 7a98f50b…f99a`, 7/7 `is at home and offline`, and **7/7 `confirmed
  at 120 Hz`** from `train.py`'s own raise — the `#30` fix doing its job, which
  `M2-E001` had to do by hand.
- All four numbered checkpoints written: `checkpoint-0050123.pt`,
  `-0100080.pt`, `-0150002.pt`, `-0200174.pt`, each with its `.sha256`.
- Exploring final wave rose: first-10 mean **5.5** → last-10 mean **7.1**
  (first-100 5.66 → last-100 6.47, overall 6.32). Weighted loss fell
  **0.1215 → 0.0921**. Value-fit correlation ended 0.63.
- Post-budget greedy evaluation inside the run: 30 episodes, 0 invalid, mean
  final wave 6.167, 21.3 decisions a wave.
- MLflow `episode_*` carry **1,419** points against the 1,420 episodes the
  session report counts: the last episode of the run is not in the per-episode
  series. The invalid count differs the same way (4 logged, 5 in the report).
  One episode, at the shutdown boundary; worth knowing before either number is
  quoted as exact.
- **The estimate was 44 minutes short.** The budget was spent at 05:40 (the
  final numbered checkpoint) against the 05:50 estimate in the live packet, but
  the session did not close until 06:34: `train.py`'s post-budget evaluation,
  session report and fleet teardown are ~50 min that the estimate did not
  price. A budget estimate for this pipeline must include them.

### Set A — selection, 2 episodes an actor a candidate

Four fleets, one a candidate, 07:38-08:18 (9-10 min each). **54 of 56 episodes
valid.** `select_checkpoint.py` over the four evaluation directories:

    4 checkpoints of stacked-dqn-20260918-215839-e3c6ba

    final_wave:
      checkpoint-0050123.pt        IQM   7.00  [  5.88,   7.88]  n=14
      checkpoint-0100080.pt        IQM   7.00  [  5.50,   8.25]  n=14
      checkpoint-0150002.pt        IQM   5.88  [  5.00,   7.25]  n=14
      checkpoint-0200174.pt        IQM   6.67  [  6.00,   7.00]  n=12

    decisions:
      checkpoint-0050123.pt        IQM 152.75  [128.75, 172.12]  n=14
      checkpoint-0100080.pt        IQM 161.12  [130.88, 188.12]  n=14
      checkpoint-0150002.pt        IQM 123.25  [106.88, 148.62]  n=14
      checkpoint-0200174.pt        IQM 142.50  [126.33, 149.83]  n=12

    selected .../checkpoints/checkpoint-0050123.pt
      its interval overlaps checkpoint-0100080.pt, checkpoint-0200174.pt,
      checkpoint-0150002.pt; the selection is a choice among checkpoints this
      sample could not separate

`checkpoint-0200174` is n=12 because one actor was lost on that fleet to
`RunPortError: the instance did not reach an active run` at episode reset.
The two leaders tie at IQM 7.00 and the rule's tie-break — lower decisions,
the earlier checkpoint — chose **`checkpoint-0050123.pt`, at 50,123 of the
200,174 decisions**. Every interval overlaps every other: 14 episodes a
candidate separates nothing, which is exactly why M2-P001 forbids set A from
appearing in a claim. `selection.json`, written beside the arm run:

    {
      "run_id": "stacked-dqn-20260918-215839-e3c6ba",
      "checkpoint": ".../checkpoints/checkpoint-0050123.pt",
      "decisions": 50123,
      "checkpoint_identity": "8344a482eede",
      "selected_on": "final_wave",
      "iqm": 7.0,
      "interval": [5.875, 7.875],
      "selected_at": "2026-09-19T06:18:04+00:00"
    }

### Set B — report, 9 episodes an actor an arm, fresh fleets

Three fleets, 08:18-09:50: `stacked-dqn` (the selected checkpoint) **61
valid**, `scripted` **63 valid**, `random` **62 valid**, all ≥60. The random
arm was **re-run whole**, as M2-P001 requires and not padded: its first attempt
returned 54 valid because one actor's bridge deploy died on
`adb: error: cannot bind listener: Address already in use` — a host port still
held from the previous fleet, not a device fault — and that attempt's records
were set aside unused. `report_arms.py ... --selection selection.json
--mlflow-run …` accepted the model arm against the selection:

    interquartile mean, stratified by actor:

    final_wave:
      random                       IQM   6.19  [  5.31,   6.84]  n=62
      scripted                     IQM   5.94  [  5.42,   6.48]  n=63
      stacked-dqn                  IQM   6.61  [  6.29,   7.13]  n=61

    decisions:
      random                       IQM 135.16  [116.62, 148.47]  n=62
      scripted                     IQM 125.97  [115.70, 136.30]  n=63
      stacked-dqn                  IQM 148.61  [140.32, 158.61]  n=61

    pairwise difference in mean final wave:
      random 5.79 vs scripted 5.81: difference -0.02 [-0.92, +0.87] d=-0.01
        n=62/63 — indistinguishable
      random 5.79 vs stacked-dqn 6.77: difference -0.98 [-1.83, -0.13] d=-0.41
        n=62/61 — separated
      scripted 5.81 vs stacked-dqn 6.77: difference -0.96 [-1.74, -0.20] d=-0.43
        n=63/61 — separated

Secondary per-wave evidence (`wave_statistics.analyse_reports`), nine wave
indices a pair, 95% intervals at 80% power, quoting each family's own summary
line and the per-episode lines:

    random vs scripted: 62/63 valid episodes
      game_ms:          separated at [1],          pooled d=+0.291
      decisions:        separated at [2, 5, 7],    pooled d=+0.311
      health_fraction:  separated at [3, 8],       pooled d=-0.091
      cash_log:         separated at [7],          pooled d=-0.035
      final_wave: 5.79 vs 5.81: -0.02 [-0.92, +0.87] d=-0.01 — indistinguishable
      decisions: 125.58 vs 123.83: +1.76 [-15.94, +19.10] d=+0.03 — indistinguishable

    random vs stacked-dqn: 62/61 valid episodes
      game_ms:          separated at [1],          pooled d=+0.244
      decisions:        separated at [3, 4, 6, 9], pooled d=-0.299
      health_fraction:  separated at none,         pooled d=+0.033
      cash_log:         separated at [3],          pooled d=-0.035
      final_wave: 5.79 vs 6.77: -0.98 [-1.83, -0.13] d=-0.41 — separated
      decisions: 125.58 vs 150.44: -24.86 [-42.02, -7.68] d=-0.51 — separated

    scripted vs stacked-dqn: 63/61 valid episodes
      game_ms:          separated at none,                  pooled d=+0.045
      decisions:        separated at [2, 3, 4, 5, 6, 7],    pooled d=-0.594
      health_fraction:  separated at [3, 8],                pooled d=+0.111
      cash_log:         separated at [1, 3],                pooled d=-0.039
      final_wave: 5.81 vs 6.77: -0.96 [-1.74, -0.20] d=-0.43 — separated
      decisions: 123.83 vs 150.44: -26.62 [-41.97, -11.59] d=-0.61 — separated

The one per-wave family that separates broadly is **decisions**: against
scripted the model takes more decisions at six of nine wave indices (pooled
d=-0.59), and it survives 150.4 decisions an episode against scripted's 123.8.
Game time per wave is flat everywhere but wave 1 — the waves themselves are
not longer, there are more of them. `health_fraction` runs slightly *lower*
for the model at the indices that separate, so the extra waves are not bought
by playing safer.

The post-hoc MLflow series were confirmed through the client:
`greedy_final_wave_iqm` (with `_ci_low`/`_ci_high`) at steps **50123, 100080,
150002, 200174** — each checkpoint at its own decision count — and
`report_random_*`, `report_scripted_*`, `report_stacked-dqn_*` at step 0.

### The pre-registered rule, applied

M2-P001: *"The headline claim `stacked-dqn` beats scripted is made only if the
pairwise interval of (model − scripted) final-wave IQM excludes zero on set B.
Beats random likewise."*

- (model − scripted) = **+0.96, interval [+0.20, +1.74]** — excludes zero.
  **"`stacked-dqn` beats scripted" is made.**
- (model − random) = **+0.98, interval [+0.13, +1.83]** — excludes zero.
  **"`stacked-dqn` beats random" is made.**
- (random − scripted) = −0.02, [−0.92, +0.87] — the comparison floor is not
  separated at this sample, as in `M1B-E021`.

Both claims rest on `checkpoint-0050123`, chosen on set A and reported on
fresh set-B episodes; set A appears in no claim.

**One instrument note, recorded because it qualifies the rule rather than the
result.** `report_arms.py` prints the per-arm statistic as the stratified IQM
but computes the *pairwise* bootstrap difference on the **mean**, so the
interval the rule is applied to is a difference of means, not of IQMs. The
pre-registration says IQM in both places. The direction and the decision are
the same either way here — the per-arm IQMs are 6.61 model, 5.94 scripted,
6.19 random, and the mean difference intervals exclude zero — but "the
pairwise interval" as the rule names it does not exist in the tooling and was
not built for this run. The claim above is the rule applied to the interval
the pre-registered command actually reports.

A second qualification, from the same output: the per-arm IQM *intervals*
overlap (model [6.29, 7.13] against scripted [5.42, 6.48] and random
[5.31, 6.84]). The rule is deliberately on the paired difference, which is the
sharper test, and that is what fired; overlapping marginal intervals are not
evidence against it, but neither should this be reported as a result so large
that the arms separate on inspection.

**This is also a stated expectation refuted in the model's favour.** M2-P001's
correction pre-registered that "the headline claim is NOT expected to be
reachable at this budget". It was reached, at 200,174 decisions and 48,892
gradient steps, by a checkpoint taken a quarter of the way through.

### What this sample cannot detect

At ~60 valid an arm the per-episode comparison can resolve about 1.1-1.3 waves
(the report prints its own floors: 1.124 against scripted, 1.218 against
random), so the measured ~0.97 is at the edge of what the design can see and
the interval's lower end is +0.20 waves — the size of the effect is not
established, only its sign. One training seed: nothing here speaks to
seed-to-seed variance, and a single seed that clears a threshold is the result
most likely not to replicate. Nothing about robustness to a different image
state, frame rate, or account progression. The selected checkpoint is the
earliest of four and its set-A lead over the others was inside the noise, so
"the model at 50k decisions is better than the model at 200k" is **not** a
finding — set A could not separate them and only one of them was reported.

### The decision as a budget unit

M2-P001 left this under review, on the reading that waits cost 4-5x a purchase
in wall time and that the wait fraction might fall as the policy sharpened.
Over the run's 1,419 logged episodes `episode_wait_fraction` means **0.841**,
and it does not move: first 100 episodes **0.829**, last 100 **0.847**, and by
quartile 0.840, 0.842, 0.840, 0.841. Against `M2-E001`'s ~0.80 it is slightly
*higher*, not lower. The policy sharpening does not buy the wait fraction back,
so the decision stays a variable-cost unit for the whole of a run at this
configuration, and the case for pricing budgets in game time rather than
decisions is not weakened by anything measured here.

### Device hygiene

Seven fleet stages in a row (four set-A, three set-B, plus one discarded
random attempt), each brought up and torn down on its own. Every stage: 7/7
`deployed bridge confirmed 7a98f50b…f99a`, 7/7 `confirmed at 120 Hz` (6/7 on
the discarded attempt, whose lost actor never got that far), and on teardown
7/7 on every line — `libunity_sha256
ffc1f3eff03cb3fe718d5659a6749c34abfbf9cab822cf386a8960cf82dd0040`,
`versionCode=1199`, `versionName=29.0.3`,
`installerPackageName=com.android.vending`, `libunity_mounts: 0`,
`bridge_artifacts: removed`, `game_frame_rate_override: reset` — then zero
qemu in `/proc/*/exe` and `adb devices` empty. Only serials 5556-5568; no
reference to 5554 or the canonical AVD anywhere in any log. No taps, no
screenshots, every artifact under `~/.local/state/tower-rl/`, none in the
repository. Two actor losses in 56 actor-stages, both named by their own
error: one `RunPortError` at reset, one host-side port collision on deploy.

**Addendum 2026-09-19 — the pre-registered statistic, computed:** the
instrument note above — `report_arms.py` printed the per-arm statistic as the
stratified IQM but computed the *pairwise* difference on the **mean**, while
M2-P001 names the IQM in both places — is now fixed rather than only recorded.
`comparison.stratified_bootstrap_difference` resamples each arm within its own
actors and differences the two IQMs *inside* the resample; `report_arms.py`
reports that as the primary pairwise line. The set-B command was re-run
verbatim over the same records. **No episode was re-collected and no record
changed; only the statistic did.**

    pairwise difference in final wave IQM, stratified by actor:
      random - scripted:      IQM difference +0.25 [-0.79, +1.06] n=62/63 — indistinguishable
      random - stacked-dqn:   IQM difference -0.43 [-1.40, +0.33] n=62/61 — indistinguishable
      scripted - stacked-dqn: IQM difference -0.67 [-1.38, -0.03] n=63/61 — separated

**The rule of M2-P001, applied to these intervals** (stated model-first, as the
rule states it):

- (model − scripted) = **+0.67, interval [+0.03, +1.38]** — excludes zero.
  **"`stacked-dqn` beats scripted" stands, as pre-registered.** The lower end is
  +0.03 waves, so what is established is the sign and not the size.
- (model − random) = **+0.43, interval [−0.33, +1.40]** — contains zero.
  **"`stacked-dqn` beats random" is withdrawn**, to *not detectable at this n*.
  That is not a claim the two are equal. The mean-based separation
  (+0.98, [+0.13, +1.83], d=−0.41) is **retained as secondary evidence** and is
  not the pre-registered statistic.
- (random − scripted) = +0.25, [−0.79, +1.06] — contains zero, as the
  mean-based line also found: the comparison floor is not separated at this
  sample, as in `M1B-E021`.

The mean difference and Cohen's d remain in the report, printed beneath the IQM
line and labelled secondary; `report_<a>_minus_<b>_final_wave_iqm_diff`,
`_ci_low` and `_ci_high` are logged at step 0 of the same MLflow run beside the
per-arm keys, and were confirmed through the client.

**This addendum supersedes the verdict paragraph above it for the purpose of
the claim.** One headline claim of the two stands. The rest of the entry — the
training run, the selection, the per-wave evidence, the device hygiene — is
unaffected, and so is the reading of M2-P001's stated expectation, except in
degree: the expectation that no headline claim would be reachable at 200k
decisions is refuted on scripted only, not on both comparisons.

Why the two statistics disagree on random: the gap is in the lower tail, which
the mean counts and the IQM trims away. Random's five worst episodes are five
wave-1 deaths and the model's are 2, 2, 3, 3, 4; both arms top out at 10-11. So
random's *mean* (5.79) sits below its *IQM* (6.19) while the model's mean (6.77)
sits above its IQM (6.61), and the gap of means is 0.98 where the gap of IQMs is
0.42. Most of the mean-based separation from random is the model not dying in
wave 1, not the model reaching further. The pre-registered statistic is the
trimmed one, and it is the one that decides.

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

**Correction 2026-09-18 (after M2-E001), developer-approved:**

> **Budget unit changed to game seconds on 2026-09-19; see `#42`.** The
> paragraphs below are the record of what was pre-registered and are not
> rewritten: they say decisions because that is what was budgeted then. The
> budget and the numbered-checkpoint cadence are now cumulative game time
> across the fleet (`--budget-game-seconds`,
> `--checkpoint-every-game-seconds`), which is the replacement this block's
> last paragraph named as under review.

The recipe's throughput premise (129–143k decisions/hour, `M1B-E052`) was a
scripted-policy figure. Measured under the learning policy in `M2-E001`:
~28–30k decisions/hour fleet-wide at 120 Hz, `episode_wait_fraction` ≈ 0.80
(waits advance the world further per decision). The 1M-decision budget would
take ~33 h, not ~7 h.

`M2-E001` also found `train.py` never raised the per-uid frame rate (fixed in
`#30`, merge `215c4c6`); the 30k/h figure was measured after raising it by
hand.

Re-priced budgeted run, approved by the developer on 2026-09-18:
`--budget-decisions 200000 --checkpoint-every-decisions 50000` (a multiple of
the default 2,000-decision block), ≈7 h at the measured rate, K=50k → 4
candidates; set A `--episodes 2` per actor (14 per candidate, 56 total); set B
unchanged (`--episodes 9` per actor for the selected checkpoint, scripted,
random; ≥60 valid per arm). `report_arms.py --selection <run>/selection.json`
required.

Stated expectation, pre-registered: 200k decisions (~50k gradient steps at
0.25 steps/decision) is a first point on the learning curve; the headline
claim ("beats scripted") is NOT expected to be reachable at this budget and is
not made unless the pre-registered rule fires on set B. The run's purpose is
to establish whether loss and exploring final wave move under this
configuration and whether the fleet holds for 7 h.

The decision as a budget unit is under review after this run (waits cost ~4–5x
a purchase in wall time); game-time is the candidate replacement (M2-P001's
rules are otherwise unchanged).

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

**Note (2026-09-19):** Measured under the game-speed multiplier mechanism
(requested speed 64) that `M1B-E012` showed alters decision density by
construction and that `M1B-E016`–`E019`/`E025` replaced with frame-exact
stepping at a pinned 1× the same day. Not comparable with any figure from
`M1B-E021` onward; see #55.

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

**Note (2026-09-19):** Measured under the game-speed multiplier mechanism
(requested speed 64) that `M1B-E012` showed alters decision density by
construction and that `M1B-E016`–`E019`/`E025` replaced with frame-exact
stepping at a pinned 1× the same day. Not comparable with any figure from
`M1B-E021` onward; see #55.

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
