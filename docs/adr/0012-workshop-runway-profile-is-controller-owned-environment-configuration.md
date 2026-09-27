# ADR 0012 — Workshop runway profile is controller-owned environment configuration (baseline v2)

**Status:** accepted, 2026-09-27. Board `#80`. Supersedes ADR 0001 and the
fixed-baseline parts of ADR 0005 (see below). Not yet exercised on device.

## Context

Every run so far was played on the v1 account: highest wave 2, no Workshop
("Enhancement") levels at all. On that account the tower hits a wall around
wave 20 whatever the policy does, so the episodes stop exercising the in-run
decisions that matter later in a round. That wall is a property of the account,
not of the policy.

ADR 0001 kept permanent progression out of V1, and ADR 0005 froze M0–M7 as
fixed-baseline V1 and put every Workshop change behind `MetaAction`. Both were
written before the wall was found. Buying Workshop levels with earned coins
would take a long time and would itself be a `MetaAction` problem. It would also
change the account permanently.

The instrumented bridge can already write game-owned arrays in the live heap of
a disposable read-only instance (ADR 0011). The Workshop's levels are three
such arrays on `Main`: `enhancementLevel`, `enhancementDefenseLevel` and
`enhancementUtilityLevel`, named by `enhancementName*` and bounded by
`enhancementMaxLevel*`.

## Decision

**A fixed Workshop profile is part of the environment configuration. The
controller writes it into the live heap before each round. It is never a policy
action, and it is never persisted.**

- **Purpose: a runway, not a destination.** The profile moves the tower past the
  wave-20 wall so that in-run purchasing is learned over a longer round. It is
  not a claim about which Workshop build is good, and nothing optimises it.
- **It is not a `MetaAction` under ADR 0005.** A `MetaAction` is a strategic,
  earned-resource, permanent choice made by a policy. This profile is a constant
  of the run, chosen by the operator (`--workshop-level N`), paid for by nobody,
  and gone when the process is. `RunAction` is unchanged.
- **Baseline v2 = the v1 account + this profile.** The image, the save and the
  profile id are those of v1. The profile exists only in the memory of the
  running game. It is written on disposable read-only instances only, and no
  instance played under it is promoted to a named snapshot.
- **v2 runs are not comparable with v1 runs.** They measure a different tower.
  The v1 results stay valid as recorded, under v1. `REFERENCE_FINAL_WAVES` and
  the other measured floors are v1 numbers. A v2 arm needs its own floors.
- **Which rows.** All 11 runway rows get the same level N: Damage, Attack Speed,
  Critical Chance, Critical Factor, Health, Health Regen, Defense %, Defense
  Absolute, Thorns, Cash Bonus and Cash / Wave. These are the levers that let a
  tower survive and earn in a normal round. The row names in
  `environment/workshop.WORKSHOP_RUNWAY_ROWS` are the game's in-run labels. They
  are not confirmed on device as Workshop names. "Thorns" is written as
  "Thorn Damage".
- **Rows held at 0, and why.** Each would change what a round *is* rather than
  how long the tower lasts in it:
  - **Orbs** one-shot normal enemies. The round would stop being a fight.
  - **Interest** is worth $0 in this setting. It adds a row and nothing else.
  - **Enemy Level Skips** remove the enemies' scaling, which is the pressure the
    wave count measures.
  - **Death Defy, Recovery and Wall** are extra-life mechanics. They blur the
    death that ends an episode.
  All other rows are left at the account's level, which on v1 is 0.
- **Rows are addressed by name, never by index.** The bridge resolves each name
  against the live `enhancementName*` arrays. It refuses the whole command if a
  name is unknown or matches more than one row, if a name is requested twice,
  or if N is above a row's maximum. The command runs in three passes: resolve
  and validate, write, then read everything back. It reports every row's level
  before and after, and the effect scalars before and after.
- **Placement and enforcement.** `InstrumentedRunEnvironment.reset` writes the
  profile *before* `begin_episode`, and checks it once the round has started.
  - If the write does not land, the episode is refused with
    `WORKSHOP_NOT_APPLIED`.
  - If a row has moved by the time the round has started, every state of the
    episode is invalid with `WORKSHOP_REVERTED`.
- **Identity.** N and the row names are recorded in `resolved_config` and in
  every episode record. N is also in `CheckpointIdentity`.
  `incompatibilities` refuses a checkpoint collected under a different level,
  for both a resume and an evaluation. N is left out of `identity_hash`, the
  same way the protocol fields are, so existing checkpoint keys keep resolving.
  The row names are left out of the checkpoint identity, so that correcting a
  guessed name after the device check does not orphan checkpoints.

### What this supersedes

- **ADR 0001** (already superseded by ADR 0005): its rule that permanent state
  stays frozen at the v1 account.
- **ADR 0005**: the rule that the fixed baseline is an unchanged, frozen v1
  profile, and the rule that Workshop state belongs only to the M8–M11 program.
  The rest of ADR 0005 stands. Strategic, earned-resource progression is still
  staged behind `MetaAction`, capabilities and evaluation gates.

## Consequences

- **Floors per profile.** A v2 arm's curve can only be read against floors
  measured under the same Workshop level.
- **Two new invalid reasons**, `WORKSHOP_NOT_APPLIED` and `WORKSHOP_REVERTED`,
  documented in `docs/environment-contract.md`.
- **A per-round cost.** Each episode boundary gets two extra round trips.
- **Heavier bridge initialisation.** The bridge now requires the Workshop
  fields at initialisation, even when the level is 0. A build of the game
  without them fails closed at bridge start.
- **Unconfirmed until the device check:**
  - the row names;
  - whether a round start reloads the levels from the save (this is what
    `WORKSHOP_REVERTED` would catch);
  - whether the game derives the effect scalars from the levels when the round
    starts;
  - whether the Workshop's tier lock (`enhancementTierUnlocked`) gates the rows.
