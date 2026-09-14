# ADR 0002: Separate policy intent from UI interaction

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

The policy must choose meaningful game actions while the real game exposes tabs,
scrolling lists, modal screens, and device-specific coordinates. Allowing the
policy to emit coordinates would couple learning to one UI layout and could expose
unintended account-affecting controls.

## Decision

`TowerEnv` accepts only a versioned `RunAction` type. `TowerController` is the V1
run interactor: it maps that semantic intent to bounded screen-state transitions,
ordinary Android input, and verified postconditions.

Run actions, navigation commands, and future meta actions are separate types with
separate authority:

- the learned policy selects run actions;
- the controller privately selects navigation commands;
- meta actions are absent from all V1 policy and environment APIs.

Coordinates and recognition regions live only in versioned UI profiles and the
UI integration layer.

## Consequences

- Policies and learners remain independent of Android and screen geometry.
- UI changes can be handled by recalibration without redefining policy meaning.
- Every attempted purchase has a typed outcome distinct from waiting or navigation
  failure.
- Future permanent progression requires a new explicit capability boundary rather
  than adding dangerous actions to the run enum.
