"""Bringing a fleet of instances up and putting every one of them down again.

One rule runs through all of it: emulators must not boot at the same instant.
Four cold-booting at once pushed host load to 10.71 and total CPU to 1,835%, and
the last of the four never left `main_unavailable` inside its cold-launch
timeout while its peers each reached home alone in 60-90s; steady-state
collection uses 563% of 3,200% available CPU, so the contention is entirely in
the boot (`M1B-E028`). `stagger_bring_up` applies that rule to actors that
collect concurrently, gating each bring-up on the previous one; `bring_up_fleet`
applies the same rule where the fleet is brought up before anything runs, which
needs no gate because the loop is already sequential.

Teardown is the other half and is never conditional: leaving an emulator running
is a device-safety failure, so every step is best-effort and independent, and a
failure is reported after the rest of the fleet has been put down rather than
instead of it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from tower_rl.simulation.bridge import deploy_bridge, run_bridge
from tower_rl.simulation.bring_up import (
    SNAPSHOT_CAPABLE_RENDERER,
    bridge_key,
    bring_up,
    keyed_snapshot_name,
    snapshot_exists,
)
from tower_rl.simulation.instance import CloneInstance, kill_emulator
from tower_rl.simulation.instrumented_bridge import InstrumentedBridgeClient
from tower_rl.simulation.instrumented_run_adapter import InstrumentedRunAdapter

#: A backstop against a readiness signal that never arrives, not the
#: sequencing mechanism itself, and measured from the previous instance's own
#: bring-up rather than from fleet start. `collect_episodes` always signals in a
#: `finally`, and bring_up's own internal waits (`wait_for_boot`,
#: `wait_until_ready`) are already bounded at 300s, so in the ordinary case —
#: including a failed bring-up — this is never reached; it exists only so a
#: bug that skipped the signal could not stall the rest of the fleet forever.
BRING_UP_STAGGER_BACKSTOP = 360.0


def stagger_bring_up[T](
    instances: Sequence[CloneInstance],
    collect_one: Callable[[CloneInstance, Callable[[], None]], T],
) -> Callable[[CloneInstance], T]:
    """Sequence each instance's bring-up after the previous instance's readiness.

    `collect_one` is what one actor does with its own instance: bring it up and
    then run whatever it was given the instance for. It is handed the callback
    that releases the next actor, and must call it once its bring-up
    concludes — success or failure alike, or one dead actor would stall the rest
    of the fleet from ever starting.

    This is the fix for the one failure a fleet of 4 hit on device: booting all
    four emulators at the same instant peaked host load at 10.71 and total CPU
    at 1,835%, and the last of the four never left `main_unavailable` within its
    cold-launch timeout while its peers each reached home alone in 60-90s.
    Steady-state collection uses only 563% of 3,200% available CPU, so the
    contention is entirely in the simultaneous boot, not in running — which is
    why only bring-up is gated here. Once an instance is up, its episode
    collection runs exactly as concurrently as it always has.

    Only bring-up is sequenced. The gates once carried a second, fleet-wide
    rendezvous — no instance raised its rate or began collecting until every
    instance was up — on the reading that a raised peer killed a booting one.
    `M1B-E043` refuted that reading, and the rendezvous was actively harmful: it
    left a ready instance idle for the rest of the fleet's boots, which is when
    the guest's Play installs what it downloaded and kills the game. An actor now
    raises its own instance and collects as soon as its own bring-up returns.

    Sequencing on readiness rather than a fixed sleep means instance i+1 starts
    its bring-up the moment instance i's bring-up actually concludes, not after
    a guessed duration — whether instance i succeeded or failed, since one dead
    actor must not block the rest of the fleet from starting.

    Every actor's thread starts at fleet start, so the backstop has to be timed
    from the previous instance's *own* bring-up, not from when this thread began
    waiting. Timing it from fleet start is what let a 7-instance cold host fleet
    overlap its boots on device: bring-up took ~165s each, so one 360s window
    measured from fleet start had already expired for instances 3-6 before their
    predecessors had even launched, and four emulators booted at once — exactly
    the defect this function exists to prevent.
    """
    gates = [threading.Event() for _ in instances]
    begun = [threading.Event() for _ in instances]
    begun_at = [0.0 for _ in instances]

    def await_previous(instance: CloneInstance) -> None:
        previous = instance.index - 1
        if not begun[previous].wait(BRING_UP_STAGGER_BACKSTOP):
            print(
                f"{instance.serial}: instance {previous} never began its bring-up within "
                f"{BRING_UP_STAGGER_BACKSTOP:.0f}s; starting anyway",
                flush=True,
            )
            return
        remaining = begun_at[previous] + BRING_UP_STAGGER_BACKSTOP - time.monotonic()
        if not gates[previous].wait(max(remaining, 0.0)):
            print(
                f"{instance.serial}: instance {previous} never signalled ready within "
                f"{BRING_UP_STAGGER_BACKSTOP:.0f}s of its own launch; starting anyway",
                flush=True,
            )

    def collect(instance: CloneInstance) -> T:
        if instance.index > 0:
            await_previous(instance)
        begun_at[instance.index] = time.monotonic()
        begun[instance.index].set()
        return collect_one(instance, gates[instance.index].set)

    return collect


def tear_down_instance(instance: CloneInstance) -> None:
    """Remove the bridge and stop the emulator, whatever the actor did.

    Leaving an emulator running is a device-safety failure, so killing it happens
    even when the bridge refuses to clean up, and the bridge's failure is what is
    reported afterwards.
    """
    bridge_error: Exception | None = None
    try:
        run_bridge("cleanup", instance)
    except Exception as error:
        bridge_error = error
    finally:
        kill_emulator(instance)
    if bridge_error is not None:
        raise bridge_error


def prepare_pinned_snapshot(renderer: str, cores: int) -> str:
    """Make sure the fleet has a snapshot to restore, by taking the cold path once.

    A `-read-only` actor cannot save a snapshot, so without this every actor
    would cold-start and the fleet would open N network windows instead of none.
    The preparation runs alone on index 0, writable, before any actor starts —
    two instances must not write the one AVD at the same time — and its instance
    is torn down again whether it succeeded or not.
    """
    instance = CloneInstance()
    name = keyed_snapshot_name(bridge_key())
    if renderer != SNAPSHOT_CAPABLE_RENDERER:
        # A renderer that cannot snapshot has no pinned snapshot to prepare, and
        # `bring_up` would take the cold path on a writable instance for nothing.
        print(f"renderer '{renderer}' cannot snapshot; nothing to pin", flush=True)
        return name
    if snapshot_exists(instance, name):
        return name
    print(f"no snapshot for the current bridge; preparing {name} once", flush=True)
    try:
        bring_up(instance, renderer, deploy=deploy_bridge, cores=cores)
    finally:
        tear_down_instance(instance)
    return name


def bring_up_fleet[T](
    instances: Sequence[CloneInstance],
    open_instance: Callable[[CloneInstance], T],
) -> tuple[list[T], list[str]]:
    """Bring each instance up only after the previous one's bring-up concludes.

    This is `stagger_bring_up`'s rule with nothing left to gate.
    Four emulators cold-booting at the same instant pushed host load to 10.71
    and left the last of them unable to reach home inside its timeout, while
    steady-state collection uses 563% of 3,200% available CPU (M1B-E028): the
    contention is entirely in the boot, so a bring-up must not begin until the
    previous one has concluded. There it is an event each actor waits on,
    because collection starts as soon as an instance is ready; here training
    starts only once the fleet is up, so sequencing the bring-ups is the same
    rule and needs no gate at all.

    A bring-up that fails costs that actor and not the fleet, exactly as a
    failed bring-up there still releases the next actor: it is reported, and the
    next instance is brought up regardless.
    """
    ready: list[T] = []
    failures: list[str] = []
    for instance in instances:
        try:
            ready.append(open_instance(instance))
        except Exception as error:  # noqa: BLE001 - one actor's failure, not the fleet's
            failures.append(f"{instance.serial}: {type(error).__name__}: {error}")
            print(f"{instance.serial}: bring-up failed: {error}", flush=True)
            continue
        print(f"{instance.serial}: ready", flush=True)
    if not ready:
        raise SystemExit(f"no instance of the fleet came up: {'; '.join(failures)}")
    return ready, failures


def tear_down_fleet(
    opened: Sequence[tuple[InstrumentedRunAdapter, InstrumentedBridgeClient]],
    started: Sequence[CloneInstance],
    tear_down: Callable[[CloneInstance], None] = tear_down_instance,
) -> None:
    """Put down every bridge and every instance this run brought up.

    Leaving an emulator running is a safety failure rather than an
    inconvenience, so each step here is independent and best-effort: releasing
    reads the bridge, and on a client that had stopped answering that read
    raised inside the caller's `finally`, skipping every remaining release and
    every teardown and leaving four emulators running with the overlay mounted.
    A failure is reported and the next instance is put down anyway.
    """
    for adapter, client in opened:
        try:
            adapter.release()
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"bridge on port {client.port}: release failed: {error}", flush=True)
        finally:
            client.close()
    for instance in started:
        # Only instances this run brought up are torn down, and one that
        # refuses to clean up must not leave the others running.
        try:
            tear_down(instance)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            print(f"{instance.serial}: teardown failed: {error}", flush=True)
