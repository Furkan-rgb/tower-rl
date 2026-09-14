# ADR 0004: Use separate post-consent and training baseline states

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

The game has user-owned first-run legal consent, creates or links an online
identity, loads cloud state, and later unlocks permanent systems such as Labs.
Repeating legal setup wastes time, while taking an arbitrary snapshot during
active progression can create silent account drift or conflict with server state.

## Decision

Maintain two local, ignored recovery artifacts:

1. a post-consent setup state captured after the game completes first-run startup;
2. a golden Tier-1 training baseline captured after legitimate desired progression,
   with permanent choices recorded and all Lab research/auto-research idle.

Named emulator snapshots are recovery and setup accelerators, not the ordinary
episode reset. Normal death-to-new-run navigation remains the hot path.

Every snapshot is versioned with the AVD, system image, emulator, APK, UI profile,
and baseline identity. Restore is followed by online/session and visible baseline
verification.

## Consequences

- EULA acceptance and initial account creation do not need to be repeated after
  ordinary restarts of the same preserved AVD.
- A Lab-unlocked baseline is possible only after the user legitimately reaches
  that state; Tower-RL does not edit or fabricate save data.
- No snapshot is captured with active or auto-repeating Lab research.
- Local restore does not claim to rewind cloud or server-authoritative state.
- Concurrent actor clones of one anonymous/online identity remain unsupported
  until an explicit isolation experiment passes.
- Snapshots remain outside Git and can become invalid after emulator, system-image,
  or AVD configuration changes.
