# Tower-RL — Experiments and Evidence

This document records feasibility work, benchmarks, failed approaches, and
contrary evidence. An entry records what was observed; it does not advance a
milestone unless the corresponding gate in `task.md` is satisfied.

Do not add proprietary package bytes, extracted assets, account/save state,
personal screenshots, bulk logs, replay, or model artifacts.

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
