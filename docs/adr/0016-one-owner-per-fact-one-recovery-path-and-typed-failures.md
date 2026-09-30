# ADR 0016 — One owner per fact, one recovery path, and typed failures

**Status:** decided, 2026-09-30; **not implemented**. Board `#100`, from the
design study `#98`. Nothing below is in the code yet; the sequence is in
"Order". Evidence labels: **measured** has an evidence pointer (an
`docs/experiments.md` entry or a file and line); **estimate** does not and is
to be confirmed before anything relies on it.

## Context

Five weaknesses share one shape: a fact about the game, or about a failure,
has no owner and is carried by convention.

- **The world hold is a convention the host mirrors.** The bridge streams a new
  state every 250 ms while the world runs (`tower_bridge.cpp:61`,
  `:2652-2669`) and stands still only after it presses `Pause`. Which regime is
  live is a bridge-local flag, reset on every connection (`:2465`) and not
  reported. The adapter re-derives it (`_held_after`, `world_held`), and a
  lifecycle `pause` counts as confirmed when the run is active at the first
  250 ms poll; nothing checks that the clock stopped. The same gap shows as the
  sequence races (`M1B-E020`, `M1B-E024`), a paused game that outlived its
  client (`M1B-E006`) and ADR 0015's round-start race. Between `start_round` and
  the closing pause, about 0.75-1 s of real time passes uncharged (estimate:
  four lifecycle presses at the 250 ms poll).
- **Nothing recovers an instance.** ADR 0015 made an episode retire the run it
  inherits, but the retirement is bounded only by the stall window and a 300 s
  ceiling (measured 19.3 s at wave 12; waves 30-50 are unmeasured, estimated
  at tens of seconds to minutes), and a dead game or bridge ends its actor for
  good (`training.py:710`, "never restarted"). Restarting only the game inside
  an offline instance cannot reach home (`M1B-E049`), and `M1B-E049` left the
  ruling open: reopen a network window, or treat the instance as lost and run
  it again. The game is also killed mid-collection by Google Play's package
  install (`M1B-E048`). `docs/task.md` requires an overnight run with no
  manual actor recovery.
- **Failure meaning is free text, collapsed at the port.** A bridge
  `stale_or_duplicate` on a purchase returns `rejected`, becomes `UNAVAILABLE`,
  then `MASK_LEGAL_REJECTED`: a sequence race is recorded as a mask
  disagreement. `_classify` routes on string prefixes. `DEVICE_FAILED`,
  `BASELINE_DRIFT` and `RECOVERY_FAILED` (`episode.py:36-38`) are declared and
  produced nowhere, and a port failure yields no `TerminationOutcome`.
- **Constants live twice, kept equal by comments:** `kMaskSlotsPerFamily` and
  `SLOTS_PER_FAMILY`, the wall ceiling and `DEFAULT_READ_TIMEOUT_SECONDS`, the
  hold rule and `_held_after`.
- **Resources outside the process are unowned.** qcow2 overlays and emulator
  temp files sit in `/tmp`, a RAM tmpfs (63 GB), and leak on a kill. No memory
  budget exists: seven emulators at 6.87 GiB each (`M1B-E029`), a 22.7 GB
  DreamerV3 replay at 1M decisions and the tmpfs share 125 GB, and `M3-P016` was
  OOM-killed (ADR 0014; the attribution was not isolated).

## Decision

### 1. State ownership

- **The bridge owns the game-world facts.** Every state message reports
  `lifecycle` (`unavailable`, `home`, `run_held`, `run_ended`) and a measured
  `held` flag. `run_running` exists only inside an advance and is never seen
  between commands.
- **The environment owns the episode:** idle, reset, in episode, ended with an
  outcome, reset.
- **Simulation owns the instance:** down, up (offline, bridge deployed), lost,
  relaunching, up.
- **The adapter holds no world state.** `world_held`, `_held_after` and
  `_resume_a_frozen_run` go. `_round_in_progress` stays only as a
  command-authority guard, and the client keeps the sequence (transport). Five
  copies of hold and sequence state become two.

### 2. The bridge enforces the hold

Between commands an active run is held. `EnsureHeld()` runs at the end of every
command handler and at connection start: if a run is active and not held it
presses `Pause`, then requires the round clock to stand still across at least
two rendered frames, so the hold is measured, not assumed. An advance's settle
frames already show this at no extra frames (`:2230-2258`). If the hold cannot
be verified in 500 ms the result is `ambiguous:not_held`. `held` is reported in
every state.

The environment's `_hand_over` keeps `WORLD_NOT_HELD`, now read from
`state.held`. `unpause` leaves the production command allowlist, and `release`
no longer unpauses: every advance already runs `Unpause` then `Pause`
(`:2142`), and `M1B-E006`'s cascade came from host pause-stepping, which is
gone. The stale-sequence class then has no world to occur in.

The bridge, not the adapter, is the place because it alone presses controls and
sees frames, it outlives reconnects, and it deletes a restated rule.

### 3. One recovery path

1. **Ordinary reset:** the game's death path, unchanged (about 1.7 s plus setup,
   `M1B-E022`).
2. **A live run after a non-`GAME_OVER` end:** retire it through the death path,
   time-bounded. Retirement uses `frame_game_ms` up to the bridge's 250 ms
   maximum (`:545`), because a retired run is not data and fidelity does not
   apply (estimate: about 2.5 times fewer frames), and adds a hard wall bound
   beyond which it escalates.
3. **Anything else:** a cold relaunch of the same `-read-only` instance: the
   emulator is torn down and brought up again, not just the game restarted, so
   the boot's radio window that `M1B-E049` showed the game needs is the
   ordinary one, and `M1B-E049`'s open ruling is taken as "run the instance
   again". This covers a lost bridge, a dead game, `main_unavailable`, a failed retirement,
   and N consecutive failed starts. Bring-up follows the existing stagger rule
   (radio window accepted, `require_offline` before any episode) and costs
   about 45-60 s (measured, 40-52 s to ready, `M1B-E029`). It is bounded and
   counted per actor; an actor withdraws only when the bound is spent. Scripts
   inject a `reopen` callable into `simulation`, so `learning` sees only a slow
   reset and the module rule holds.

**A cold `-read-only` relaunch is the golden-baseline recovery.** The base image
is the golden state and the emulator writes only to a private overlay, so a
relaunch discards every overlay write. `AGENTS.md` and `docs/task.md` now name
that mechanism. This clarifies how the requirement is met and does not weaken
it: recovered actors must still restore the baseline and re-verify it before an
episode, and the requirement that repeated restoration not be assumed to give
suitable randomness still has to be tested, now for relaunches.

**Snapshots are rejected** for training starts and for recovery:

- `-gpu host` cannot save a snapshot of a Vulkan app (`docs/setup.md`
  "Renderer", `M1B-E026` A), and a snapshot carries renderer state
  (`bring_up.py:87-92`), so every restoring instance would have to run
  lavapipe: about 1000-1400% CPU per instance and a 3x lower speed-up
  (`M1B-E022`). Seven such instances do not fit 32 threads.
- A restore takes 10.5 s at home (`M1B-E026` A), slower than the roughly 2 s
  death path.
- The game is not deterministic under fixed actions (`M1B-E039`), so a snapshot
  buys no reproducibility.
- A mid-round snapshot would bake in the round-start unlock (ADR 0011), the
  pin and the Workshop levels.

### 4. Typed failures

The environment maps the bridge's reason codes once:

| bridge reason | outcome |
| --- | --- |
| `stale_or_duplicate` | `SEQUENCE_REFUSED`, an action-pipeline failure, never `MASK_LEGAL_REJECTED` |
| `precondition_failed` | `MASK_LEGAL_REJECTED` |
| `confirmation_timeout`, `contradictory` | `ACTION_PIPELINE_FAILED` |
| `wall_ceiling` | `TRUNCATED_BY_WALL` |
| `not_held` | `WORLD_NOT_HELD` |

`RunPortError` carries a `kind`:

| kind | outcome and action |
| --- | --- |
| `BRIDGE_LOST` | `DEVICE_FAILED`, relaunch |
| `GAME_REFUSED` | boundary retry |
| `SETUP_NOT_APPLIED` | refused start |
| `RETIREMENT_FAILED` | `RECOVERY_FAILED`, relaunch |

`DEVICE_FAILED`, `BASELINE_DRIFT` and `RECOVERY_FAILED` are produced this way
or deleted; none stays declared and dormant.

### 5. Resources

- Emulator temp files and overlays move to `state/emulator-tmp/<serial>` (NVMe,
  through `TMPDIR` and `ANDROID_TMP`). `tear_down_instance` removes them and
  `run_stage.sh` sweeps them once no qemu is left. Overlay growth is measured
  first, with `du` during a 7-actor stage.
- `train.py` runs a memory preflight: actors times measured RSS plus the
  replay-capacity estimate against `MemAvailable`. It refuses to start rather
  than leave the choice to systemd-oomd.
- The handshake reports the bridge's constants and the client refuses a
  mismatch.

### 6. Throughput and comparability

Pipeline throughput is benchmarked with a fixed scripted policy, in frames per
second and non-advance milliseconds per decision, never with training
decisions per hour: those follow the policy's game time per decision, which
ranged 1.5-4.3 s across arms. Render-interval-16 gave about 1.8x game time and
1.06x decisions per hour because run 5's policy used 1.58x the game time per
decision of run 4 (`M2-P005` diagnostic (c)); the same checkpoint on both
builds gave B/A 1.039 [0.954, 1.140] (`M2-P005` addendum).

**Comparability rule.** A change is robustness-only if the fixed-policy
benchmark's final wave, decisions per wave and round-clock ratio are unchanged
and only failure rates move. Anything touching frames, settle, the round start
or the learning cadence changes behaviour and needs a pre-registered
equivalence stage first.

Where a decision goes today (default build, fixed greedy policies, 7 actors,
measured from the `m3-p003` to `m3-p006` eval arms): 220-230 ms, of which the
advance is 160-180 ms (17-18 frames at 8.7-9.7 ms, vsync-paced at 120 Hz) and
55-60 ms is outside the advance loop and unattributed (estimate: the adb-forward
path shared by seven devices). Training on render-interval-16 spends about
240 ms: bridge 157-230, learn 24-32, blocked 0-110, policy 3-8 ms, and learning
runs on the finishing actor's thread under the progress lock (`training.py`
progress lock). That reopens the host-side lever `M1B-E036` closed at 0.25
steps per decision.

| Lever | Estimated gain | Validity risk | Confirmation |
| --- | --- | --- | --- |
| Asynchronous learner holding the replay ratio by a debt bound | +15-30% decisions/h per actor | policy lag changes learning; pre-register | one short A/B stage, learn plus blocked toward 0, lag reported |
| Attribute and cut the ~55 ms non-advance overhead (request/response instead of the stream) | up to +20% | none if payloads are unchanged | one `run_actors` stage, `bridge_round_trip` minus the sum of `wall_micros` per actor, N=1 against N=7 |
| An 8th actor | +14% | thin VRAM margin, 87% (`M1B-E029`, measured) | only once relaunch exists; `nvidia-smi` per process |
| Emulator cores, `ncore` 4 against `-cores 8`, render-interval build | unknown | none | solo fixed-policy fps ladder, then N=7 |
| One bridge call per choice-point span | at most 10% at advances per decision 1.0-1.1 | changes settle-tail game time and ADR 0009's per-advance accounting | only if advances per decision stay above 1.3 |
| Purchase poll 50 ms to 2-5 ms | about 2% (purchases are 0.1 per decision) | none | source (`:65`, `:2629`) |
| Raise `frame_game_ms` | none | rejected on fidelity (`M1B-E038`, measured) | none |

### Order

1. **P1 classification** (Python only): the reason table, `RunPortError.kind`,
   `SEQUENCE_REFUSED`, the dormant outcomes produced or deleted.
2. **P2 resources:** `emulator-tmp`, the memory preflight, overlay measurement.
3. **P3 relaunch plus bounded, fast retirement.**
4. **M1 fixed-policy benchmark** (no code): frames per second, non-advance
   milliseconds, boundary seconds per episode, VRAM, overlay size. It becomes
   the comparability reference.
5. **P4 bridge hold:** `EnsureHeld`, measured `held`, `unpause` removed, the
   adapter mirror deleted, constants in the handshake. All four bridge builds
   are rebuilt and deployed per `docs/setup.md`, and an equivalence check runs
   on the eval build against the current floors before the old baselines stay
   comparable. Its one behaviour change, expected negligible, is that the
   uncharged pre-pause second at round start disappears.
6. **T1-T5:** the asynchronous learner (pre-registered), transport per M1, the
   8th actor, emulator cores, span merge only if warranted.

## Consequences

- An instance that loses its game or bridge costs about a minute, not an
  actor; the overnight requirement (`docs/task.md`) becomes reachable.
- `held` is a fact the environment reads instead of a rule it restates; the
  stale-sequence class and `_resume_a_frozen_run` disappear.
- Failure rates become attributable by outcome; a sequence race no longer
  reads as a mask disagreement.
- `ADR 0015`'s retirement and hold stay, and its adapter mirror is superseded by
  the bridge's measured flag once P4 lands.
- P4 changes a bridge binary, so it carries a deployment and an equivalence
  check; P1-P3 are robustness-only under the rule above.

## Unresolved

Named in the design as unmeasured; each is settled by the stage that needs it.

- **Whether the chain-starting cut** (ADR 0015, "Not explained here") survives
  ADR 0015's fixes. One short 7-actor stage with a 0.3 s think delay, reading
  the live invalid reasons, decides whether P4 lands before or after the next
  long run.
- **Retirement duration at waves 30-50.** Estimated, not measured.
- **Overlay growth per instance** during a long stage.
- **Attribution of `M3-P016`'s OOM** between emulators, replay and tmpfs.
- **The ~55 ms non-advance cost per decision:** transport is only the prime
  suspect.
- **Whether 500 ms suffices** to verify a hold on a loaded host.
