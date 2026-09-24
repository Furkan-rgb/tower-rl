# Tower-RL — Workstation Handoff

## START HERE — current state, 2026-09-19

This section is the one place that says where the project actually is. Everything
below it is historical and dated; read this first and treat older sections as
context rather than as current truth.

### The protocol the project collects under

Five things define what a run is, and all five are in force together:

- **A decision is asked for only at a choice point** — an observation whose mask
  offers at least one purchase. Forced `WAIT` slices are played through by the
  environment and accrue to the surrounding decision, so one decision spans one
  or more advances ([ADR 0009](adr/0009-decisions-at-choice-points.md);
  `--decision-cadence choice-points` is the default, and `every-slice` is kept
  to reproduce run 1's protocol).
- **The policy is shown what the player sees**: `observation-v2`, 42 run scalars
  — the four run-level readings plus 38 live `Main` fields — and nine per-slot
  row features with raw `level`/`max_level` and unclipped affordability ([ADR
  0010](adr/0010-observation-v2-everything-the-player-sees.md), confirmed on
  device in `M2-E006`).
- **A run's budget is decisions, independent of game speed** —
  `--budget-decisions`, `--checkpoint-every-decisions` and
  `--selection-period-decisions` (`#68`, replacing `#42`'s game-time budget),
  because learning is per decision and game time per decision moves with the
  policy (`M2-P005` diagnostic (c)). The arm is the checkpoint of the best
  near-greedy selection period from period 2 on (`docs/solution.md` §9.2b).
- **Exploration can be an Ape-X ladder**: `--exploration ladder` anneals actor
  `i` of `N` to `0.4 ** (1 + 7 i / (N - 1))` instead of to one floor, and the
  collection curve is then read from the near-greedy actors alone (`#37`). The
  default is still `uniform`.
- **Which upgrade rows are purchasable is configuration, not the image**:
  `--upgrade-availability image|all` ([ADR
  0011](adr/0011-upgrade-availability-is-applied-at-round-start.md), evidence
  `M2-E008`). `image` is the default and is what every baseline so far was
  measured under — the six rows the v1 image offers. `all` reopens every row the
  game really has, applied through the bridge at each round start, because the
  game recomputes its real rows' availability whenever a round begins; a
  "profile v2 base image" could not have carried it. The profile id is the v1
  image's either way, and floors measured under one availability do not read
  against the other.

**Before any `all` run: `state/bridge/current` must be reinstalled.** The
unlock commands moved into the production build, so the installed artifact has
to be a build that has them — digests and the install order are in
[`docs/setup.md`](setup.md) under "Production digests, and the one that has to
be reinstalled". A bridge without them refuses the episode by name rather than
playing a locked run quietly.

A run may also **stop before its budget is spent**: with
`--early-stop-patience-periods` set, the near-greedy mean final wave of each
checkpoint period is compared against the bar the curve last cleared, and a run
that fails to add `--early-stop-min-improvement` waves for that many periods in
a row stops after writing that crossing's checkpoint (`#45`; the default of 0
spends the whole budget). An interrupted run **resumes**: `train.py --resume
<checkpoint.pt>` restores the weights, the optimizer, the game-time and decision
counters and every schedule derived from them, re-warms replay under the loaded
policy, and continues the same whole-run budget (`#32`).

### Watching, recording, and where state lives

`scripts/spectate.py` is the human-facing path: one windowed clone with this
project's bridge deployed, at 60 Hz through `-gpu lavapipe`, refusing to start
while any other emulator is running (`#12`). `--record` writes the guest screen
as mp4 (`#36`) together with `<stem>.decisions.jsonl`, the agent's own decision
log; `scripts/render_recording.py` composes the two into one video with the
action side panel (`#48`).

Everything this project writes lives under the git-ignored `state/` directory at
the repository root — `bridge/`, `runs/`, `records/`, `recordings/`, `logs/`,
`mlflow.db` — rather than under `~/.local/state` or `/tmp` (`#40`);
`docs/setup.md` lists the layout.

### Where run 2 stands

Milestone 2 run 2 is **in progress**. Its protocol is pre-registered as
`M2-P002` in `docs/experiments.md` and is authoritative for the run: option B,
two seeds run sequentially on a seven-instance fleet at 120 Hz under the four
changes above. Stage 1 of 5 is done — the random and scripted baselines are
re-measured under this cadence and this schema and set the kill threshold
(`M2-E007`, board `#46`). Training, evaluation, recordings and the verdict are
not, and nothing measured so far is a verdict on the model.

### How a stage is run

Every device stage — a training seed, an evaluation batch, a recording
session — is launched once through `scripts/run_stage.sh` and then left alone,
rather than watched by whoever started it:

```text
nohup ./scripts/run_stage.sh --name m2-run2-train-seed1 --instances 7 -- \
  uv run --extra tracking python scripts/train.py \
      --actors 7 --renderer host --frame-rate-hz 120 \
      --decision-cadence choice-points --exploration ladder \
      --budget-decisions 120000 --checkpoint-every-decisions 5000 \
      --selection-period-decisions 15000 \
      --epsilon-anneal-decisions 2500 \
      --early-stop-patience-periods 2 --early-stop-min-improvement 0.2 \
      --seed 1 > /dev/null 2>&1 &
```

The stage command after `--` runs exactly as written — the runners still bring
up and tear down their own fleet — and the script guarantees what happens on the
way out. On success, on failure, and on `SIGINT`/`SIGTERM` alike it interrupts
the stage so the runner can tear its own fleet down, runs
`instrumented_bridge.sh cleanup` on every serial still live, kills every
emulator still attached, and verifies the host is empty of qemu processes and
adb devices. It refuses to start against any attached instance that is not the
instrumented clone, not `-read-only`, or not verifiably offline by interface,
and it never kills the canonical evaluation AVD. Signals are ignored once the
teardown has begun, each instance's cleanup being bounded so that cannot hang. Everything
lands in `state/logs/<name>-<timestamp>.log`, ending in one summary line
carrying the stage's exit code, the cleanup result, how many instances were
cleaned and the wall time; the script's own exit status is non-zero if either
the stage or the cleanup failed. `docs/setup.md` section 10 has the detail.

Do not hand-run a stage and then hand-run its cleanup: the abort path is exactly
where that has been got wrong before.

### The goal

A reproducible benchmark on the real game in which a learned model, trained
under one budgeted protocol, reproducibly beats the random and scripted
baselines, with all evidence in `docs/experiments.md`.

The multi-backbone comparison is retired (`#7`, 2026-09-18): the project commits
to one backbone, the `BACKBONE` constant in `scripts/train.py`. The equal-budget
interleaving and the bootstrap/per-wave statistics remain and are what any arm
comparison runs on — they were used for the 60 Hz against 120 Hz equivalence
fleet, where `M1B-E053`'s provisional reject did not replicate in `M1B-E054` and
120 Hz cleared.

### Task tracking is on GitHub, not in this file

Work items, priorities, and status live in GitHub Issues, tracked against two
milestones on this repository:

- [`M1 Foundation`](https://github.com/Furkan-rgb/tower-rl/milestone/1) —
  everything needed before RL training can start.
- [`M2 Training`](https://github.com/Furkan-rgb/tower-rl/milestone/2) —
  running and comparing backbones.

The working view is the project board:
<https://github.com/users/Furkan-rgb/projects/3> (columns `Backlog`, `Next`,
`In progress`, `Blocked`, `Done`). Deferred, unscheduled ideas carry the
`deferred` label and sit in `Backlog` with no milestone. This replaces the
numbered priority lists that used to live in this file — do not re-add an
ordered task list here; open or update an issue instead.

`docs/experiments.md` is the evidence log — every finding, dated, newest
first, including negatives. It is **not** a task list: it records what was
measured, not what to do next.

### What is proven and working

- **The environment.** The game is observed and controlled through its own
  runtime, with no OCR in the decision loop. 1,000 consecutive scripted episodes
  ran at 100 percent validity (`M1B-E009`). That clears M2's *reliability*
  clauses. **M2 is not complete**: its speed/actor-count comparison clause and
  its visual-evidence clause are both open.
- **The learning pipeline, end to end on the real game.** Two backbones,
  interleaved on one device, 54 episodes, 614 optimisation steps, no replay
  rejections, checkpoints round-tripping with identity and checksums
  (`M1B-E011`). This proves plumbing, not learning.
- **One backbone behind the contract suite**: `stacked-dqn`, the rank-1
  candidate from `docs/rl-candidates.md`. `recurrent-q` was the second arm the
  `M1B-E011` run above used; it was removed with the multi-backbone goal (`#7`),
  so `learning/` holds `stacked_dqn.py` alone and `BACKBONE` names it.
- **The comparison machinery**: bootstrap intervals and Cohen's d.
- **Frame-exact stepping** (`M1B-E016`) and **the advance loop inside the
  bridge** (`M1B-E017`), described below.
- **The episode boundary and bring-up are screen-free.** The round-start
  control is `BattlePanelUI.StartNewRound` on the `BattlePanel` GameObject,
  found by dumping IL2CPP metadata rather than guessing names; boundary cost
  fell from 7.25 s to 1.716 s and nothing in the RL loop reads a pixel
  (`M1B-E022`). Bring-up readiness is likewise read from the bridge's own
  `main_unavailable`/`no_initialized_run` reasons, verified again on device
  (`M1B-E024`). The screenshot classifier that was the previous oracle has
  been removed outright (#16): it had no caller left, and a static guard
  under `tests/unit/simulation/` now holds every module that can reach an
  instance to reading no pixel and tapping no coordinate.
- **MLflow experiment tracking** is wired behind a port and on by default;
  run with `uv run --extra tracking` to have it record.

### Standing decisions a new agent must not re-litigate

1. **The game's own speed multiplier is pinned at 1x and is not a speed-up
   mechanism.** A faster game clock makes each rendered frame worth more game
   time, which coarsens the agent's decisions in proportion to the speed gained:
   4.8 decisions per wave at 64x against 12.2 at 1x (`M1B-E012`). Encoded in
   `instrumented_run_adapter.py` as `GAME_SPEED = 1.0` with `_pin_game_speed`
   restoring it at every episode start, and asserted by test. Speed is not a
   parameter anywhere: not in the adapter, the cadence, or any runner's
   arguments. The pin is applied once, at the episode boundary, and nowhere
   else: a post-advance re-pin was tried and found to strand the observation
   sequence a round in progress depends on, killing the process on the very
   next advance (`M1B-E024`). The adapter now refuses to issue a command of
   its own initiative while a round is in progress, enforced by construction.
   Because nothing readable reports the unpaused world's true rate (the field
   reads 0.0 in every paused observation the host takes), the pin is not
   verified by reading it back — it is verified by the effective game time
   per frame, a declared parameter checked against the game's own round clock
   every episode: an episode whose ratio exceeds 1.25 is failed by name and
   excluded, catching exactly the failure mode where starting a round left
   the world at this account's 1.5x ceiling instead of 1x (`M1B-E023`,
   `M1B-E024`).
2. **Speed comes from stepping frames faster.** `Time.captureDeltaTime` makes one
   rendered frame worth a fixed amount of game time however long it took to
   render, so decision moments are identical at any speed *by construction*. The
   bridge implements this; a 250 ms step costs about 57 ms of wall clock whether
   the multiplier is 1 or 16 (`M1B-E016`).
3. **The clone must be offline during automation, verified by interface.**
   `airplane_mode_on` reads 1 while the wifi radio is up and was never evidence
   of anything (`M1B-E010`). Use `svc wifi disable` / `svc data disable` and
   confirm nothing but `lo` holds an IPv4 address. `instrumented_bridge.sh
   deploy` now refuses otherwise.
4. **The game needs a network to *start*, not to play.** It blocks on a Firebase
   check and an OFFLINE modal. `scripts/clone_session.py launch` performs the
   launch-online, wait-for-the-bridge, cut-radios sequence and verifies the
   result by interface. It runs *after* `instrumented_bridge.sh deploy`, because
   deploy cold-launches the game offline and because readiness is the bridge's
   own reading.
5. **Bring-up readiness is non-visual.** The bridge reports `main_unavailable`
   while the game is still starting — the splash, or the OFFLINE modal — and
   `no_initialized_run` once it is up and idle at home, so nothing in the
   automated path classifies a screenshot, and since #16 no screenshot
   classifier remains in the tree at all.
6. **`-gpu host` is the training renderer; snapshots are lavapipe-only.**
   Verified equivalent to lavapipe on game-time ratio, decisions/wave, mean
   wave and unattended stability (see "Done: `-gpu host` is the mandatory
   training renderer" below). The emulator refuses to snapshot a Vulkan app
   under host GPU, so bring-up never attempts a save or restore there;
   `prepare_pinned_snapshot` and `bring_up` in `src/tower_rl/simulation/` check
   the renderer against `SNAPSHOT_CAPABLE_RENDERER` by name. Host bring-up is
   cold every time, by design, and says so in its log
   (`renderer 'host' cannot snapshot a Vulkan app`).

### Done: the advance loop is inside the bridge

Commit `611667d`, measured on the device in `M1B-E017`. One `advance` command per
decision, 5 of 5 episodes valid at 100 ms per frame, speed-up 5.013, 60.4 fps,
274 ms per advance, zero `advances_cut_short`, and zero `BRIDGE_EVENT_DIVERGENCE`
— the bridge's stopping conditions and the host's `_events_between` agreed on
every decision, which is the invariant that keeps the host definition
authoritative. Two defects that run exposed are fixed in the commit that carries
`M1B-E017`: the host now treats **any** inbound frame as liveness (no heartbeat
reaches it while a command is always in flight, so every unattended run died at
about 60 seconds), and the game-time witness is now the game's own per-round
clock, `round_ms`, because `playTime` runs at wall rate and witnessed nothing.

### Done: the `frame_game_ms` sweep, and the standing speed decision

`M1B-E018`. Five arms (16.7, 50, 100, 100 repeat, 250 ms), 8 episodes each,
sequential not interleaved. Decision density (dec/wave) is flat at 20.9–21.1
across 16.7/50/100/250 ms — frame size does not move it over this range. 250 ms
shows a statistically detected final-wave difference from the 16.7 ms reference
and is rejected as not faithful to it, regardless of its (favourable) direction.
**Standing decision: the benchmark runs at `frame_game_ms = 100`.** The prior
target of matching 89.3 decisions/episode (`M1B-E014`) is withdrawn as an error:
that figure is from the old, pre-`M1B-E016` code path, and every arm of the
`M1B-E018` sweep lands at 124–167 decisions/episode through the current path,
regardless of frame size. `M1B-E018` also flags that eight episodes per arm
cannot detect a one-wave fidelity difference at 80% power (97/arm would be
needed), so absence of a detected difference at 50/100 ms is not equivalence,
only an absence of evidence at this sample size.

### Done: the game-time accounting question is closed for reporting purposes

`M1B-E019`. The round clock stays the authoritative witness for reported
speedup, already in effect since `c07ae35`; the advance loop's budget stays on
frame arithmetic (`frames × frame_game_ms`), for the reasons recorded there —
the budget bounds quiet game time rather than measuring it, and advances stop
on events rather than on the budget. The measured law is
`round_delta ≈ 1.07 · frame_game_ms · (loop_frames − 1)`: the game credits
about 7% more simulated time per frame than requested, and about one frame per
advance goes uncredited. Two mechanism questions are open and unchased — the
1.07 factor, and the uncredited frame — see `M1B-E019` for the candidates
considered.

### Done: the comparison floor is measured, under-powered on one question

`M1B-E021`. Scripted, random and wait arms, 23 valid episodes each (the
stated minimum) at commit `86fcf3c`. Spending beats not spending by a wide,
unambiguous margin (scripted 5.57 waves, random 5.35, wait 1.87). The
scripted heuristic is **not** shown to measurably outperform random choice
at this sample size (+0.22 waves, inside the 1.1–1.7 wave MDE at n=23) —
this is under-powered, not a finding of equivalence, and no conclusion is
drawn about scripted-versus-random. The Lead's reading: this shows our
scripted heuristic is not a strong bar, not that upgrade choice doesn't
matter; the ceiling above these baselines remains unknown. The run also
surfaced the boundary deadlock recorded below, which cut it from a single
78-episode interleaved run to 23 pooled one-episode-per-arm segments.

### Done: the boundary tap is retired, and a 1.52x game-time inflation was found and fixed

`M1B-E022`. The round-start control was found by dumping IL2CPP metadata
rather than guessing names — `BattlePanelUI.StartNewRound` on the
`BattlePanel` GameObject, not on `Main` — closing the receiver hunt left open
since `M1B-E013`. Boundary cost fell from 7.25 s to 1.716 s and nothing in
the RL loop reads a pixel. The same device session's `-gpu host` arm cleared
its own health checks (3/3 valid, zero divergence) but was later shown
(`M1B-E023`) to have run under a 1.52x game-time inflation identical across
both renderers, traced to this account's 1.5x speed ceiling surviving the
new round-start path; its wave figure does not stand as a throughput or
fidelity comparison. `M1B-E024` verified the bring-up and field types clean
on device, reproduced the inflation's proximate cause (a post-advance re-pin
that strands the observation sequence and kills the process), and isolated
the cure — the episode-boundary pin alone, with no re-pin, restores the
ratio to 1.0088. Commit `cf504b8` ships that fix: the re-pin is removed, the
`GAME_TIME_INFLATED` ratio guard (threshold 1.25) is retained, stale-sequence
errors now cost one episode instead of the run, and the adapter refuses to
issue any command of its own initiative while a round is in progress.

### Done: `-gpu host` is the mandatory training renderer; snapshots are lavapipe-only

Device-verified at `9cc3a233`. `-gpu host` and `-gpu lavapipe` are equivalent
on the metrics that matter for training: game-time ratio 1.0108 (host) vs.
1.0118 (lavapipe), decisions/wave 21.0 vs. 20.73, mean wave 6.2 vs. 6.0, and
an unattended 25/25-valid-episode run on host at 80.6 episodes/hour with
every health counter at zero. **Standing decision: `-gpu host` is the
renderer training runs use.**

Snapshot save is a separate, narrower capability that does **not** carry
over: the game uses Vulkan, and this emulator refuses to save a snapshot of
a Vulkan app under `-gpu host`, replying `KO: Snapshot save is skipped.
Reason: UNSUPPORTED_VK_APP`. `clone_session.py` used to print that reply and
report success anyway, which meant a later restore would load stale
renderer state from a different renderer, find no running game, and fall
back to the cold path silently — the fallback was correct, the false
success was not. `save_snapshot` now requires the emulator's own `OK` reply
and, cheaply, that the snapshot directory exists, and raises naming the
reason and the renderer otherwise. `bring_up` no longer attempts a save (or
a restore) under any renderer but `lavapipe`; under `-gpu host` it goes
straight to the cold path and says so in one log line. **Consequence: host
bring-up is cold every time** (45-60 s, not the ~10 s a restore costs), and
snapshots remain useful only for lavapipe-driven work such as manual review.

### The current priority order

Superseded by the GitHub board — see "Task tracking is on GitHub, not in
this file" above for the milestones and the board URL. The device-chain and
comparison-floor work referenced here is tracked as milestone issues; the
stale-observation reconnect gap and multi-actor scaling are tracked under
the `deferred` label.

### The boundary deadlock — fixed

`M1B-E021` surfaced a blocking defect: `AdvanceUntilEvent` sets the paused
flag and dispatches `Pause` before reading the settled snapshot, so a tower
death inside the pause-settle window emits a terminal observation while the
bridge believes the world is paused. The bridge then emits only heartbeats,
the client's `read_state` returns the stale terminal reading indefinitely,
and `_resume_a_frozen_run` declines to unpause because the reading is
(wrongly, in this case) terminal. Observed at about 1 failure per 7 episode
boundaries. **Fixed at `328318e`** ("Hold the sequence for a world standing
still, not for a pause pressed"): pause is now reported as the settled state
found it, confirmed once more against the state about to be sent. No
deadlock has been observed since, across 6 boundaries and 2 arms of
device-verified running. Do not reopen this as a live risk without new
evidence.

### The immediate next slice, before this run

**The `stale_or_duplicate` sequence race is verified fixed on device**
(`M1B-E020`). Zero rejections across 35 advances at 1000 ms injected host
latency and 20 advances at 3000 ms (was 15/35 at 1000 ms, `M1B-E019`), with
zero stale or mismatched reads across 49 paused reads and fidelity/throughput
indistinguishable from the pre-race `M1B-E019` figures. An independent review
run over the same commit found further defects in the surrounding lifecycle
code, recorded in `M1B-E020` — a fix commit for some of these may be landing
concurrently with this note, so check `M1B-E020` and recent commits for
status before starting new work here. The most consequential finding: the
death-boundary transient retry became a guaranteed no-op against a paused
frozen world, misclassifying a real `GAME_OVER` as `OBSERVATION_INVALID` and
corrupting the M1 gate's validity rate if left unaddressed. The review also
confirmed stale data cannot reach training replay regardless.

**Episode-boundary overhead** is now the next slice. `M1B-E020` measured
`begin_episode` directly at 7.277 s and 7.254 s on the RESULT→RETRY path (vs.
0.256 s when the run was already active), and advance share — the fraction of
wall clock spent on genuine advances rather than boundary — at 0.839,
confirming the fixed per-episode boundary is now the largest remaining
wall-clock cost now that the sequence race is closed. The 6-second
result-panel settle and the gated boundary tap (item 2 below) are the two
known contributors.

### After that, in order

Superseded by the GitHub board (see above). Two items from this list are
already done and remain recorded in "Done: the boundary tap is retired, and
a 1.52x game-time inflation was found and fixed" above: the boundary tap
receiver was found and the cold-launch/snapshot and `-gpu host` items are
tracked as open issues on the board.

### Claims this session corrected — do not reinstate them

- `M1B-E006`'s decisions-per-wave table (63/79/69) **does not reproduce** and
  must never be quoted as current.
- 8x was adopted then withdrawn as a training speed. It does match 1x density,
  but only because 8x sits below the frame limit by coincidence.
- "A property setter from the socket thread crashed the game twice" was
  **misattributed**. The recorded double crash was `il2cpp_domain_get` at
  library-load time before any client connected. No entry records a
  `runtime_invoke` crash. The rule is: never execute managed game code or Unity
  scene-graph code from the socket thread; engine leaf accessors, attributed to
  `libunity.so` by `dladdr` first, are a different category and are proven safe.
- `Main` existing only inside the battle scene: **disproved** by `M1B-E015`.

### Where things are tracked

| Document | Holds |
| --- | --- |
| `docs/task.md` | Authoritative scope, milestones and gate criteria |
| `docs/solution.md` | Design decisions, including 9.2c on decision moments |
| `docs/experiments.md` | Every finding, newest first, including negatives |
| `docs/rl-candidates.md` | The RL algorithm study and its ranking |
| `docs/adr/` | Architecture decisions 0001-0010 |
| `docs/setup.md` | Host, AVD, bridge build and the `state/` layout |
| `docs/architecture.md` | The packages that exist and the dependency rule |
| `docs/environment-contract.md` | The observation, action and transition contract |
| This section | Current state and what to do next |

### Device state

Nothing is running. The last stage was closed with
`instrumented_bridge.sh cleanup`: original `libunity.so` SHA-256
`ffc1f3ef…dd0040` verified, package identity unchanged (`versionCode 1199`,
`29.0.3`, installer `com.android.vending`), zero mounts, artifacts removed,
device offline, no emulator running.

To resume: `clone_session.py start` (instance up and offline, game not yet
launched), then `./scripts/instrumented_bridge.sh deploy`, then
`clone_session.py launch` (the one online window, ending with the bridge
reporting the game up and idle). `clone_session.py up` does all three.

**The bridge this project deploys lives in `state/bridge/`, inside the
checkout and git-ignored.** One directory per bridge, named for the
SHA-256 of the `libtower_bridge.so` inside it, holding that artifact, the
patched `libunity-bridge.so` and the `CMakeCache.txt` the compatibility identity
is read from; `current` is a symlink to the one that is deployed, so `ls -l`
shows which bridge this project installs and what its digest is; `config/`
beside them holds the private build configuration. Today that is
`7a98f50be6d6c6ec262f332e60511f4e6f84f61da66691c626e7d4bb8ad7f99a`. This
replaces `/tmp/tower-bridge-live.latest`, which pointed into a session
scratchpad and did not survive a reboot — and, before `M1B-E017`, pointed at a
build that could neither deploy nor handshake.

Rebuild and install a new one (the NDK is at
`~/.local/share/android-sdk/ndk/29.0.14206865`; the full recipe, including
configuring from `state/bridge/config/profile.cmake`, is in `docs/setup.md`):

    cmake --build <build dir>
    sha=$(sha256sum <build dir>/libtower_bridge.so | cut -d' ' -f1)
    install -D -t state/bridge/$sha \
      <build dir>/libtower_bridge.so <build dir>/libunity-bridge.so \
      <build dir>/CMakeCache.txt
    sha256sum state/bridge/$sha/libtower_bridge.so   # must be $sha
    ln -sfn $sha state/bridge/current

Copy, verify, *then* move the pointer, in that order. `TOWER_BRIDGE_BUILD_DIR`
still overrides all of it, which is how a bridge under development is deployed
straight out of its build tree.

Resolving the artifact refuses by name if `current` is missing or dangles, or
if `libtower_bridge.so` does not hash to the directory name it is filed under —
an installed build cannot pass as a digest it is not.

**Every deploy reads the deployed bridge back off the device**, through
`adb shell "su -c 'sha256sum /data/user/0/<package>/files/libtower_bridge.so'"`,
and holds it to the host artifact: a deploy either prints
`deployed bridge confirmed <sha256>` or fails by name, on an empty read-back as
well as on a mismatch. The CLI and the fleet share that one form. `su 0
sha256sum` is not it — it returned nothing on all seven instances of two
seven-actor fleets (`M1B-E046`, `M1B-E053`), which is why both had to record the
deployed bridge as unconfirmed; "unconfirmed" is no longer a silent outcome.

The logcat tag is
`tower_bridge`. Always finish with `cleanup`.

### Host quirks worth knowing

- Background tasks are killed by a low-memory watchdog that reads `free` rather
  than `available`; it killed two runs on a machine with 92 GB available. Run
  device measurements in the foreground or detached with `setsid nohup`.
- Piping a command through `tail` masks its exit status. It hid a mypy failure
  at `1be263a` and a crashed training run behind exit code 0. Check each check's
  own status.


## What exists on this Mac

The validated account-bearing snapshot is stored outside the repository at:

```text
${ANDROID_AVD_HOME:-$HOME/.android/avd}/tower_rl_api36_play_arm64.avd/snapshots/tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914/
```

The snapshot is approximately 2.9 GiB on disk (`ram.bin` and renderer
textures). The AVD is API 36 `google_apis_playstore`, ARM64, Pixel 2, 1080×1920,
420 dpi, and was created with Android Emulator 37.1.11. The game is the
Play-installed The Tower 29.0.3 (`versionCode=1199`), not the local 29.0.1
XAPK.

It is intentionally not checked into Git. It contains Android user data,
account/session state, renderer state, and a local game save. The same applies to
the other snapshots under:

```text
${ANDROID_AVD_HOME:-$HOME/.android/avd}/tower_rl_api36_play_arm64.avd/snapshots/
```

## Reopen the validated Mac baseline

On this Mac, with the AVD already provisioned:

```text
emulator @tower_rl_api36_play_arm64 \
  -gpu lavapipe \
  -no-audio \
  -no-boot-anim \
  -snapshot tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914 \
  -no-snapshot-save
```

Then verify:

```text
uv run python scripts/clone_session.py verify
```

(Historical: this step read `uv run tower-rl doctor` and `uv run tower-rl
probe`. There is no `tower-rl` console script and no `probe`; readiness and
network state are reported non-visually.)

The expected state is Battle home, Tier 1 selected, highest wave 2, 53 coins,
0 gems, x1.00 total coin bonus, Labs locked, airplane mode enabled, and no
external route.

## Prepare the RTX 4090 workstation

Treat the workstation as a new device profile. The observed workstation is
Ubuntu 26.04.1 x86_64 with an i9-14900, 125 GiB RAM, RTX 4090, and accessible
KVM. Do not assume this Mac snapshot
will boot there: an RTX 4090 machine is commonly x86_64, while this snapshot is
from an ARM64 AVD and is also tied to the pinned Lavapipe/Swangle renderer.

1. Clone the repository and run `uv sync --all-groups`.
2. Run the repository preflight before changing device state:

   ```text
   mkdir -p runtime
   ./scripts/doctor.py --json > runtime/doctor.json
   ```

   Characterize the host, Android SDK, emulator version, virtualization, ABI,
   storage, and renderer before selecting an AVD.
3. Copy `configs/workstation.example.yaml` to `configs/local.yaml` and fill in
   the discovered values. Create a compatible API 36 Google Play AVD for the
   workstation ABI with the repository helper:

   ```text
   ./scripts/create_avd.sh tower_rl_api36_play_x86_64 \
     'system-images;android-36;google_apis_playstore;x86_64' pixel_2
   ```

   Use the same logical display profile where possible; record any difference in
   `docs/environment-profile.yaml`.
4. Launch the AVD with the renderer selected for that host:

   ```text
   ./scripts/launch_avd.sh tower_rl_api36_play_x86_64 host
   ```

5. Launch the official game from its Google Play listing with the user's manual
   sign-in. Do not automate credentials, purchases, advertisements, or legal
   consent. The local XAPK remains metadata/reference input only.
6. Recreate the semantic baseline manually (Tier 1 selected, no permanent
   Workshop spending, Labs locked), take the device offline only after the game
   is running, and create a new workstation-local snapshot with a unique name.
   Airplane mode is **not** sufficient and never was: the setting reads 1 while
   the wifi radio stays up with a route (`M1B-E010`). Use `adb shell svc wifi
   disable` and `adb shell svc data disable`, then confirm that `ip -o -4 addr
   show` lists nothing but `lo`.
7. Verify that new snapshot by restoring it and reading the bridge back —
   `uv run python scripts/clone_session.py restore <workstation-snapshot>` then
   `verify` — before any actor or learner work. (This step read `tower-rl probe
   --navigate` when a visual probe existed; it does not.)

The workstation snapshot should be created with the renderer that is actually
validated there. Do not copy or reuse the Mac's account-bearing snapshot merely
to save setup time. If a same-OS, same-ABI private transfer is ever attempted,
keep it outside the repository and treat it as an experimental artifact rather
than a supported bootstrap path.

## Portable versus machine-local material

Portable through Git:

- source code, tests, `uv.lock`, and sanitized documentation;
- the XAPK inspection metadata and renderer/profile requirements;
- the probe and restore commands.

Machine-local only:

- XAPK/APK bytes and extracted game files;
- AVD directories, emulator snapshots, Android user data, and account state;
- screenshots, logs, replay, checkpoints, and trained models.

The first workstation milestone is therefore **reprovision and revalidate**, not
snapshot copying. Once it passes, add the workstation profile as a separate
environment version and only then begin multi-actor isolation work.

## Current continuation state — 2026-09-15

Workstation reprovisioning is complete. The authoritative local runtime is the
Play-installed The Tower 29.0.3 build (`versionCode 1199`) on
`tower_rl_api36_play_x86_64`, serial `emulator-5554`, using `-gpu lavapipe` and
the validated ANGLE/Swangle path. The AVD now has `hw.ramSize=6144`. The new
golden candidate is
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_6gb_offline_home_20260914_workstation`.
The prior 2 GiB snapshot
`tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_offline_home_20260914_workstation`
is retained unchanged. The emulator is left at the new snapshot with airplane
mode enabled and no route.

The 6 GiB candidate passes the profile probe, bounded navigation/restore, and
one 4/4 natural-death diagnostic. It does **not** pass M1 reliability: the fresh
100-consecutive run stopped after seven valid episodes with an explicit device
failure and restored the baseline. The subsequent Wave 3 stall interpretation
has been rejected by bounded progression evidence: an offline sparse-capture
run advanced through Wave 6 and died naturally at about 377 seconds. The old
120-second deadline expired during valid gameplay, and the first post-baseline
death's `New Highest Wave!` layout moved the Game Stats HOME-button border and
was classified as a modal. Production now permits 600 seconds, polls at the
configured one-second interval, and recognizes both result layouts. A live
1/1 natural-death smoke passed in 351.3 seconds and restored this same snapshot.
No renderer/profile change or replacement snapshot was required. Do not declare
M1 complete until the unchanged 6 GiB baseline passes the required 100 episodes.

A subsequent 10-episode qualification stopped after three valid episodes because
the result-to-home transition completed just after the controller exhausted its
three five-second waits. The captured failure frame was already valid Battle
home, and there was no device/lifecycle failure. Those bounded waits are now 12
seconds; a production-path 2/2 natural-death verification passed in 687.0
seconds and restored the offline baseline. The 100 gate has not been restarted.

The fresh 10-episode retry also stopped after three valid episodes. Evidence then
showed a delayed generic modal-close tap crossing the result-to-home transition,
opening Home Settings, and allowing subsequent lifecycle taps to reach an
unlinked-cloud-save account warning. No confirmation or account action was
selected, and recovery restored the baseline. Generic modal taps are now
disabled: lifecycle waits observe a modal without input and either continue when
it settles or fail explicitly. The narrow remaining blocker is positive subtype
recognition and a destination-safe action for any modal that truly requires
dismissal. Do not start the 100 gate before that path passes a fresh 10/10 run.

The 2 GiB configuration backup is private local state at
`state/avd-config-backups/20260914-workstation-2gb/`.
To launch the 6 GiB candidate explicitly:

```text
./scripts/launch_avd.sh tower_rl_api36_play_x86_64 lavapipe \
  tower_golden_t1_v1_play_29_0_3_lavapipe_swangle_6gb_offline_home_20260914_workstation
```

The XAPK is no longer required for this workstation runtime. Its sanitized
metadata remains in `docs/environment-profile.yaml`; do not copy proprietary
archive bytes into the repository.

M0 is complete: host/device characterization, Play installation, renderer
selection, offline baseline, snapshot restore, and bounded navigation probe are
recorded in `docs/experiments.md` (M0-E001 through M0-E014).

**Superseded.** The visual M1 implementation this paragraph pointed at —
`src/tower_rl/vision.py`, `src/tower_rl/infrastructure/adb_device.py`,
`src/tower_rl/application/controller.py`, `src/tower_rl/infrastructure/visual_profile.py`
— and the `scripts/m1_reliability.py` gate no longer exist. The game is observed
and controlled through the instrumented bridge instead, with no screenshot
classifier left in the tree (see START HERE, and `M1B-E056` for the last device
verification of the code that replaced it). The current packages and their
dependency rule are `docs/architecture.md`; the reliability gate now in force is
the 1,000-episode scripted run recorded in START HERE.

Before changing code, read `AGENTS.md`, `docs/task.md`, `docs/solution.md`,
`docs/architecture.md`, relevant ADRs, and the current tests for the area.

## Resolved — screen calibration after progression drift — 2026-09-17

Stage B is blocked on one calibration question, recorded rather than guessed at.

The clone's account has drifted from the documented baseline by playing: Highest
Wave 2 to 11, 53 coins to 909, with a `MILESTONES` button and a gem/video widget
now on the home screen. The Battle-home classifier samples pixel (10, 200), which
was background at the baseline and now falls inside the new widget, so the screen
no longer classifies and the adapter correctly refuses the boundary tap.

Resolved in `M1B-E005`, and since superseded entirely: the visual gate described
here was removed with the rest of the screenshot path. It lived in
`src/tower_rl/infrastructure/visual_profile.py`, calibrated against sixty-three
live frames labelled by the game's own lifecycle, with seven anchors for home,
four for the result panel and three for an active run. The result gate anchors on
the RETRY button itself, and the adapter settles six seconds before classifying
because the panel animates in. Only anchor values are recorded, never
screenshots.

Also historical: `M1B-E003` measured throughput under `-gpu host`, which
`M1B-E004` then rejected for rendering the frame incorrectly under the
screen-classification path that existed at the time. Superseded: bring-up and
the episode boundary are now non-visual (see "Standing decisions" above), and
`-gpu host` is device-verified equivalent to lavapipe and is the standing
training renderer (see "Done: `-gpu host` is the mandatory training
renderer").

## Experimental no-OCR continuation — 2026-09-15

The scalability investigation in `M1-E007` established a viable behind-the-GUI
path, but it has not replaced the V1 visual contract yet. Two private disposable
AVDs exist outside Git:

- `tower_rl_instrumented_api36` is a rooted clone with the unchanged
  Play-installed 29.0.3 package.
- `tower_rl_gadget_api36` is a throwaway re-signed-XAPK clone used only to
  isolate native instrumentation behavior.

Frida server attachment works at the x86 process level but cannot enumerate the
translated ARM64 IL2CPP module. ARM64 Frida Gadget aborts under
`libndk_translation`. Do not spend another iteration on those routes unless the
host or Android ABI changes.

A small custom ARM64 dependency does work under translation. It starts inside
the Unity process, resolves IL2CPP exports, finds `Main` dynamically, and reads
`Main.gameSpeed`. It was proven first in a temporary re-signed XAPK and then in
the Play-installed package by placing a reversible Magisk bind mount over the
extracted `libunity.so`. In the second proof, package version 29.0.3,
`installerPackageName=com.android.vending`, signed APK bytes, and app data all
remained unchanged. The mount was removed and the bridge/probe files were deleted
afterward; no emulator is intentionally left running.

**The guest's Google Play update round, and the WebView bake (2026-09-18).**
The clone's Play downloads a `com.google.android.webview` update during the one
online window of a cold bring-up and installs it moments later, killing every
process that holds WebView — the game included (`M1B-E045`, `M1B-E047`). On
2026-09-18 the AVD was backed up whole to
`state/avd-backup-2026-09-18/` (35 GB, `MANIFEST.sha256` over
all 53 files, `SIZES.txt`) and booted ONCE writable and online so that update
could land in the base image: WebView 151.0.7922.199 (versionCode 792219908)
installed, and the game identity was read back unchanged before shutdown
(versionCode 1199, 29.0.3, installer `com.android.vending`, `libunity.so`
SHA-256 `ffc1f3ef…0040`, no per-uid frame-rate override). Nothing about Play was
disabled, frozen or firewalled. **The bake did not reach the fleet recipe**: a
subsequent `-read-only` cold boot reads WebView 694313738 (133.0.6943.137)
again, and so does a fresh WRITABLE boot, so the update did not
persist at all rather than being hidden from read-only instances (`M1B-E051`).
Writes were flushed and no rollback was logged; a settings change made after the
install did persist, so the reading is that PackageManager discarded the update
at the next boot's package scan. The kill nevertheless did not
recur: `M1B-E052` came up 7/7 at 120 Hz with zero `installPackageLI` lines
across all seven logcats, because the staged session the kill needed was
consumed and Play did not stage another. The base image boots ROUTABLE, as run J's own
instances show (`WifiService starting up with Wi-Fi enabled`, then a DHCP lease
of 10.0.2.16, before each bring-up cut it). That was left over from the bake, so
on 2026-09-18 one writable boot disabled both radios and read `ip -o -4 addr
show` back listing only `lo`, with the game identity and `libunity.so` SHA-256
unchanged and `reboot -p` as the shutdown: **radios are off in the base image as
of today.** Either state is safe — every bring-up cuts the radios itself and
verifies offline by interface — but starting offline keeps the online window
short. To
repeat the bake if Play stages another component later: verify no emulator is
running, re-take the backup, boot the clone once writable and online with no
bridge and no game launch, wait for the `installPackageLI` round to finish and
120 s of quiet, read the identity lines back, shut down cleanly, and restore
from the backup if anything but the intended package changed.

ADR 0006 now defines the separate instrumented-training and official-evaluation
profiles. The first production bridge slice lives under `native/tower_bridge/`
with its strict host client in
`src/tower_rl/simulation/instrumented_bridge.py`. A live 29.0.3 run passed
the exact compatibility handshake and returned both active and terminal
observations with all 60 in-run upgrades (20 Attack, 20 Defense, 20 Utility).
Startup observations fail closed until `Main.Instance` and run scalar state are
initialized. The temporary overlay was removed, a clean reboot restored the
original `libunity.so` hash, and no emulator is intentionally left running.

Step 3 of the six-step sequence in `M1-E007` is now proven live and recorded in
`M1B-E001`. On the rooted clone, `WAIT`, duplicate/stale rejection, locked and
unpriced rejection, and confirmed `attack` and `defense` purchases all execute
through Unity's main thread with game-owned before/after evidence, corroborated
by sparse pixels. Utility is unavailable at this fixed baseline, with evidence.
Five defects in the previously unexecuted command slice were corrected first; the
important ones are that IL2CPP resolution must happen on the first client
connection rather than at library load, `tier_unlocked` never gates a purchase,
and cash is not a confirmation signal.

`M1B-E002` then took the loop to whole episodes. In-run control needs no screen
at all: a greedy scripted policy reaches wave 7 to 10 with 20 to 26 confirmed
purchases per episode, driven only through the bridge. Costs are refreshed by the
game's own `Upgrade*CostCalc` methods, so no tab interaction is needed. The
episode boundary still needs one bridge-gated tap, because `Main` exists only in
the battle scene and the known restart entry points are progression-gated here;
`BattlePanelUI.StartNewRound` is the likely receiver but its GameObject name is
unknown. Speed above the account's own 1.5 ceiling applies through the game's
`GameSpeedModifier`, and decision cadence must scale with it or the policy is
silently starved. Pause and unpause freeze the world exactly, but stepped mode
currently costs about 430 ms per decision and is slower than free running; that
overhead needs profiling. ADR 0007 records the decision that the instrumented
clone is now the primary training and evaluation environment, with a bounded
official cross-check at promotion.

Continue from step 4. The open gates are family cost coverage without an
undocumented manual step (a family's cost array only populates after its tab has
been displayed), deterministic normal-speed scripted parity against the visible
controller, protocol-loss and thread-affinity quarantine behavior, and the speed
equivalence gate. Only after parity should higher `Time.timeScale` values or
actor-count scaling be tested, and no instrumented transition may enter replay
before those gates pass. Keep the canonical emulator and all evaluation
unchanged, unrooted, normal-speed, and pixel-observed.

Operate the private clone with `scripts/instrumented_bridge.sh`
(`verify`/`deploy`/`cleanup`), which deploys the installed bridge under
`state/bridge/current` by default; `TOWER_BRIDGE_BUILD_DIR`
overrides that with a private NDK build directory holding `libtower_bridge.so`,
the patched `libunity-bridge.so` and `CMakeCache.txt`, which is how a bridge
under development is deployed. Launch that clone with `-gpu lavapipe`;
`swiftshader_indirect` produced an unusable System UI ANR on this host. Always
finish with `cleanup` and confirm the original `libunity.so` SHA-256, unchanged
package identity, no remaining mounts, and no running emulator. Cleanup reads
that original digest out of the installed `CMakeCache.txt` to verify the
unmount, so an install whose cache records a different original fails cleanup
loudly rather than degrading to a blind `umount`. Never send a tap without first
classifying the screen.

Private/local-only material includes the copied rooted system image, disposable
AVDs, temporary XAPK/APK extraction, IL2CPP dumps, patched native libraries,
temporary signing key, save bytes, logs, and account-bearing state. None may be
committed. The only durable repository evidence is the sanitized experiment and
this handoff.

## Open questions — 2026-09-17

Recorded rather than guessed at, so the next slice does not quietly decide them.

**Resuming an interrupted training run.** `scripts/train.py` writes a complete,
identity-bound checkpoint every 25 episodes and once at the end, and
`learning/checkpoint.py` will refuse to load one whose profile or schema differs.
Nothing reads one back yet: `TrainingRun.run()` always starts from zero
decisions, so resuming would need it to begin from a restored progress counter,
and the replay buffer is in memory and is lost with the process either way. A
multi-hour run interrupted at hour three therefore restarts. The question is
whether resume should restore replay as well — a checkpoint that restores
weights but not replay resumes into a very different learning problem from the
one it stopped in, and recording that honestly matters more than the
convenience. Not blocking: the budget is counted in game seconds, so an
interrupted run is a shorter run rather than a corrupt one. Closed by `#32`
(2026-09-19): `train.py --resume <checkpoint>` restores optimizer, target,
counters and the MLflow run.

**Where the stacked agent's window length should sit.** `docs/rl-candidates.md`
3.1 treats `k` as a tuned hyperparameter between 4 and 16 and section 5 names
`k = 1` as the ablation that settles whether history is needed at all. The
default here is 8, chosen as the midpoint and nothing more. The ablation is a
comparison arm like any other and belongs after the comparison floor (E), not
before it.

**Actor-count scaling.** Still unmeasured. One emulator sustains about 236
episodes per hour at 99.3 percent validity (M1B-E008). Whether two or four
instrumented clones on this host multiply that or contend for the GPU is an
empirical question, and the number that settles it is aggregate *valid* episodes
per hour, not episodes per hour. Measure it before committing to long training
runs, because it changes what an equal game-time budget costs in wall-clock time.

## Directed next steps — 2026-09-17

Superseded by the GitHub board (see "Task tracking is on GitHub, not in this
file" above). Item 2 (finding the round-start receiver) is done — see "Done:
the boundary tap is retired, and a 1.52x game-time inflation was found and
fixed" above. Items 1 (offline-started snapshot) and 3 (`-gpu host`) are
tracked as open issues.


## The frame is the floor on decision granularity — 2026-09-17

A constraint worth stating separately, because it bounds what any amount of
engineering can achieve and it reframes the speed choice.

Unity advances game time per rendered frame. A step shorter than one frame passes
no world time at all, so the finest decision granularity available at speed `s`
is one frame, and one frame is worth `frame_wall_seconds x s` of game time. At
30 fps that is 33 ms of wall clock, so:

| Speed | Game time in one frame |
| --- | --- |
| 1x | 0.03 s |
| 8x | 0.27 s |
| 16x | 0.53 s |
| 64x | 2.1 s |

The environment asks for a 250 ms slice, which a frame exceeds above about 8x.
`M1B-E006` reported decisions per wave of 63, 79 and 69 at 1.5x, 4x and 8x
falling to 39, 20 and 16 above, which appeared to confirm this. **That table does
not reproduce.** Measured fresh in `M1B-E012`, 8x gives 13.4 decisions per wave
and 64x gives 4.8: the direction holds, the magnitude does not, and 8x is not a
speed at which the requirement is met.

Two consequences:

1. **There is no speed that satisfies the requirement.** 8x is 2.8 times better
   than 64x on decision density and 3.6 times worse on throughput, and it still
   reaches only 13.4 decisions per wave (`M1B-E012`). Lowering the speed shrinks
   the violation; it does not remove it, because speed and density are traded
   against each other by construction.
2. **Getting both requires decoupling game time per frame from wall time per
   frame**, which is what `Time.captureDeltaTime` does. `captureDeltaTime` is a
   native-backed property rather than a field, so `field_static_set_value` cannot
   reach it — but it does not need a main-thread receiver either. It is a leaf
   engine binding reachable through `il2cpp_resolve_icall`, as are
   `Time.frameCount`, `QualitySettings.vSyncCount` and
   `Application.targetFrameRate`.

   **A correction to what this section previously claimed.** It said a Unity
   property setter from the socket thread "is the pattern that crashed the game
   twice already". That misattributed the crash. The recorded double crash
   (`M1B-E001`) was `il2cpp_domain_get` called at library-load time, four seconds
   after `libil2cpp.so` appears and *before any client ever connected* — premature
   runtime resolution, not a threading violation and not a property setter. No
   entry in this repository records a `runtime_invoke` crash at all. The rule is
   still right, but its scope is narrower than stated: **never execute managed
   game code or Unity scene-graph code from the socket thread.** Engine leaf
   accessors on `Time` and `QualitySettings` are a different category.

   So the two problems decouple. Only the boundary tap needs an out-of-battle
   receiver.
