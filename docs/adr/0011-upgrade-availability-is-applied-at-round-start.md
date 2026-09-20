# ADR 0011 — Upgrade availability is applied at round start

**Status:** accepted, 2026-09-20. Evidence: `M2-E008`; board `#54`.

## Context

Profile v1 — the frozen image every baseline so far was measured on — offers six
purchasable in-run upgrade rows: 4 attack, 2 defense, 0 utility, out of the 17 /
18 / 13 rows the game really has. That caps what a policy can learn: two thirds
of the decision problem is not on the table, and an entire family is missing
from it.

`#54` asked whether the instrumented bridge's write to `Main.upgradeUnlocked` /
`upgradeDefenseUnlocked` / `upgradeUtilityUnlocked` could widen that, and what
it costs. `M2-E008` answered on device, on one offline `-read-only` clone:

- the write **takes** — 4 / 2 / 0 becomes 20 / 20 / 20 on read-back;
- the game **honours** it: a utility row, of which v1 has none purchasable, was
  bought for in-run cash and its level advanced 0 → 1, and a random episode then
  made 17 purchases including seven rows v1 keeps locked;
- it **survives the round but not the next round start**: after `go_home` +
  `start_round` every *real* row was back at its v1 value and only the empty,
  unpriced tail slots kept the flag. The game recomputes availability for its
  real rows whenever a round begins;
- **no starting scalar moves** — health, cash, damage, attack speed, crit, wave
  scaling, wave length: all identical across the write and across the round it
  started.

The route we had assumed — build a "profile v2" base image that ships with the
rows already open — is not what the evidence supports. There is nothing to put
in an image: an image cannot carry a flag the game recomputes at every round
start, and the write we have is a live-process write, not a save.

## Decision

**Upgrade availability is a property of the environment configuration, not of
the profile image, and it is applied at each round start.**

- `UpgradeAvailability` (`environment/run_environment.py`) has two values.
  `image` — the default — is whatever the profile image provides, which is what
  every baseline so far was measured under. `all` means every row the game
  really has is purchasable.
- Under `all`, `InstrumentedRunEnvironment.reset` issues the unlock through the
  `RunPort` after the round has started and before the first observation is
  returned, and then holds the game to it: the read-back must show at least as
  many true flags per family as that family has real rows, where a real row is a
  slot with a non-empty label from `slot_labels`. It does not retry.
- The profile id is unchanged. The image is still profile v1 under either
  value; what differs is what the environment does at the round start.
- Availability travels in the run identity, in MLflow params, and in the episode
  and session records, and `CheckpointIdentity.incompatibilities` refuses a
  resume or an evaluation across a difference in it.
- Every slot of every family is written, not only the real ones. The legal set
  already excludes an empty row without reference to its flag: an empty-named
  row is priced zero, and `_build_row` masks any row whose cost is not positive
  (`available = active and unlocked and not maxed and priced and cost <= cash`).
  The device agrees — the cost arrays were populated for 17 / 18 / 13 rows
  before any write, and the flagged tail rows stayed unpurchasable. Writing all
  twenty is what the bridge already does and is one rule rather than two.
- The bridge's `unlock_state` and `unlock_all_upgrades` commands move into the
  production build. A capability an ordinary measured run depends on cannot live
  behind a diagnostics flag; this changes the production digest, which is
  recorded in `docs/setup.md` beside the old one.

## Consequences

- **Baselines must be re-measured per availability.** `M2-E007`'s random and
  scripted floors, and every reference in `experiment/run_identity.py`, were
  measured under `image`. They say nothing about a run played under `all`, and a
  comparison across the two measures the availability, not the policy.
- **Two new invalid reasons.** `UNLOCK_NOT_APPLIED` fails the episode at the
  boundary when the unlock does not land — the actor counts it exactly as it
  counts a round that would not open. `UNLOCK_REVERTED` fails a decision when a
  real row that was unlocked at the round start is locked again in a state the
  policy is being asked about: the decision problem changed under the episode,
  and an episode measured across that change is measured against nothing.
- **The unlock is a per-round cost.** One extra round trip at each episode
  boundary, and one command the environment issues of its own initiative inside
  `reset` — the one adapter command not routed through `_command_between_rounds`,
  because it belongs to the round start by construction.
- **Nothing is saved.** The write lands in the live process's heap. Whether an
  autosave would carry it to disk is still unanswered (`M2-E008` did not test
  it), so instances stay disposable and no instance played under `all` is
  promoted to a named snapshot.
- **The "profile v2 base image" route is retired.** It was never possible: the
  game recomputes what such an image would have carried.
