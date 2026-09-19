"""Where this project keeps the state it writes: `<repo>/state/`.

Every artifact a run produces or reads back — installed bridge builds and the
private build configuration, runs and their checkpoints, the MLflow store,
evaluation records, spectate recordings — lives under one git-ignored directory
inside the repository. Nothing is written to the host outside the project, so a
checkout is the whole project: what is committed, and the state beside it.

The root is resolved from this file's own location rather than from the current
working directory, because a runner started from anywhere at all must find the
same directory; a path taken from the cwd is how `recordings/` once landed
wherever a session happened to be.

There is deliberately no environment-variable override. One location is the
whole point of the decision, and a test that needs its own directory is given
one explicitly (`tmp_path`) rather than by reaching past the resolver.

This module lives in `environment` because that is the root package: simulation,
experiment and the scripts all may import it, and the location of the project's
state is a fact none of them should each spell out for itself.
"""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """The checkout this package is part of: `src/tower_rl/environment/..`×3."""
    return Path(__file__).resolve().parents[3]


def state_directory() -> Path:
    """The git-ignored directory every artifact this project writes lives under.

    The directory is not created here. What a subdirectory is for belongs to
    whatever owns it — `bridge/` to the bridge installer, `runs/` to training,
    `recordings/` to a spectated session — and each of those makes its own.
    """
    return repository_root() / "state"


__all__ = ["repository_root", "state_directory"]
