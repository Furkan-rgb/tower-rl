# ADR 0015 — An episode never inherits a run, and the agent decides only on a held world

**Status:** adopted, 2026-09-30. Board `#95`. Device evidence:
`docs/experiments.md`, "Invalid cuts no longer cascade".

## Context

M3 training runs showed invalid ends in chains: after one invalid end, the
episodes that followed opened above wave 1 and ended invalid again. There were
also fresh episodes that ended at wave 1 having played no game time. Two
mechanisms are behind this. Both come from the code and both reproduced on a
device:

- **An episode inherited the run it was cut from.** Any end other than
  `GAME_OVER` (an invalid transition, a stall, an operator stop) leaves the
  game's run live. The adapter's `begin_episode` only sends a *finished* run
  home. A live run was unpaused, re-pinned and handed over as the next episode,
  mid-game. On the device, a cut at wave 12 was followed by an episode starting
  at wave 12, twice out of twice.
- **The round was handed over running.** The speed pin is two lifecycle
  presses, and every lifecycle press except `pause` leaves the bridge's world
  running. A running world streams a new observation every 250 ms. A policy
  that took longer than that to choose its first action bound a sequence that
  no longer existed, and the bridge refused it as `stale_or_duplicate`. On the
  device, with 0.3 s of thinking per decision, three fresh episodes out of three
  ended `mask_legal_rejected` at wave 1 with 0 ms of game time. With no
  thinking time, none of five did. The inherited run passed through the same
  unpause and pin, which is why those episodes re-tripped too.

Recovery by golden-snapshot restore was not available. A restore is an
instance-level relaunch owned by `simulation`, no snapshot exists for this
bridge build, and the host renderer cannot take one.

## Decision

**Reset retires any live run through the game's own death, and control returns
to the agent only while the world is held.**

- **Retirement.** Before the Workshop write and the round start,
  `InstrumentedRunEnvironment.reset` reads the port. If a run is still active,
  it is played out with `WAIT` advances until the game ends it:
  - 10 000 ms per advance, the protocol maximum;
  - the cadence's frame;
  - health changes are not a stop condition.

  `begin_episode` then finds a finished run and takes the ordinary
  death-to-new-run path: home, a fresh round at wave 1, and the Workshop,
  availability and setup written and checked again. This is the game's normal
  reset path (AGENTS.md), not the golden baseline.
- **Nothing of the retired run is an episode.** It gets no tally, transition,
  decision or view, so none of it can reach replay. The next episode reports
  only `retired_run_wave` (0 when nothing was retired) and
  `retirement_wall_seconds`.
- **Bound.** Two, both failing `reset` with `RetirementFailed`, a
  `RunPortError`, so the actor counts a failed start and withdraws under the
  existing consecutive-failure rule:
  - the existing `STALLED` window, measured on the game's own round clock
    (`round_ms`), not on the game time the bridge credited: if the round clock
    does not move for `stall_window_wall_seconds` it fails, and frames
    rendering while it stands still (the `GAME_TIME_DEFLATED` signature) are
    not progress;
  - a wall-time ceiling on the whole retirement,
    `RETIREMENT_WALL_CEILING_SECONDS` (300 s).

  The failure carries `retired_run_wave` and `retirement_wall_seconds`, which
  no episode record will, in its message and in the `failed episode start` line
  `train.log` gets. A stop request (`stop_requested`) abandons a retirement
  between advances (`RetirementAbandoned`), so a SIGINT is not held up by it.
  The first advance of a retirement can meet a running world (for example, a
  run left behind by an earlier session's `release`). If that advance is
  refused, retirement reads the latest state and sends it again.
- **The hold.** `InstrumentedRunAdapter._start_round` ends the boundary with the
  game's own `pause`. This is the same `Main.Pause` an advance presses as it
  settles, so the first decision starts from the same held world, with its
  sequence standing, as every later decision. It is one hold per round. It is
  not the per-slice pause-stepping that `M1B-E006` withdrew for its wall cost,
  and it spends no game time.
- **The invariant, enforced in one place.** `RunPort.world_held` reports whether
  the world stands still. The adapter mirrors it from the commands it carries,
  using the bridge's own `world_paused` rule:
  - a confirmed `pause` holds the world;
  - an advance that settles on a live run holds it;
  - any other lifecycle press releases it, and so does an ended run;
  - an advance the bridge could not run (`clock_unavailable`) presses nothing
    and releases it, and so does a bridge connection that failed: the next one
    starts with the world running, held again only by a confirmed `pause`;
  - anything else leaves it as it was.

  `InstrumentedRunEnvironment._hand_over` is the one point where control returns
  to the agent: the end of `reset` and of every `step`. At that point a state
  whose run is in progress on a world that is not held is marked
  `WORLD_NOT_HELD`. The transition is then classified `observation_invalid`,
  and `reset` fails outright.

## Consequences

- An invalid end costs its own episode and one retirement. Measured on the
  device, retirement took 19-30 s of wall time at waves 7-12 (19.3 s for a
  wave-12 cut). Deep cuts at waves 30-50 are estimated, not measured, at 1-3
  minutes, and are bounded by the 300 s ceiling above.
- Outcomes stay distinct:
  - the cut episode keeps its own termination and reason;
  - its last transition is a truncation;
  - the next episode is a fresh one, `starting_wave` 1.
- `begin_episode`'s continue-a-live-run branch (`_resume_a_frozen_run`) is no
  longer reached from `reset`. It stays for any other caller, and a live run it
  resumes is held again by the same closing `pause`.
- The environment's first observation of an episode is unchanged: the hold
  spends no game time. An environment-side alternative, a one-frame advance at
  reset, was rejected because it changed the first observation and the advance
  accounting that `EVERY_SLICE`'s run-1 reproduction depends on.
- **Not explained here:** the cut that *starts* a chain in a long mid-game
  episode. Every path inside an episode already hands over a held world:
  - a `WAIT` settles paused;
  - a purchase does not move the hold;
  - the death-boundary recovery is an advance.

  So a slow policy mid-episode cannot meet this race, going by the bridge
  source. That cut's reason is left to the live invalid-reason logging (`#95`,
  the metrics side).
