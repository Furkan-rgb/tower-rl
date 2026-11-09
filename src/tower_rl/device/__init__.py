"""Reaching one real instance of the game: bringing it up, and driving it.

What a run is and what it costs belongs to `environment`; this package is how a
run is made to exist. It owns the emulator instance and its identity, the
bring-up decision that leaves the game started and offline, the guest frame rate
it is paced at, the instrumented bridge deployed into it and the wire client
that speaks to it, and the fleet primitives that bring several instances up
without letting them boot on top of each other.

It depends on `environment` and on nothing else in `tower_rl`: the adapter here
is a `RunPort` implementation, so the environment never learns how a device is
reached, and the learner and the experiment never learn it either.
"""
