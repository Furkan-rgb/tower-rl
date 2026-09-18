"""The dependency direction the packages are allowed to depend on each other in.

The rule is one-directional: an experiment observes training, so it may know
about the environment, the learner and the ports they are driven through, and
nothing about how a device is reached or how a run is started. Read from the
source rather than from an import graph at runtime, so a module that is never
imported is still held to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src"

#: What `tower_rl.experiment` may not reach for. Device adapters and host checks
#: are the simulation side, and a script is where a run is composed: an
#: experiment that imported either could not be run from anywhere else.
FORBIDDEN_TO_EXPERIMENT = ("tower_rl.infrastructure", "tower_rl.doctor")

#: Modules that only exist beside an entry point. `tower_rl.experiment` may not
#: import them at all, by any name.
SCRIPT_MODULES = ("train", "run_episodes", "run_actors", "clone_session", "compare_arms")


def imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, absolute and relative alike."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_experiment_package_reaches_for_no_adapter_and_no_script() -> None:
    package = SOURCE / "tower_rl" / "experiment"
    modules = sorted(package.rglob("*.py"))

    assert modules, "the experiment package must exist to be checked"
    for path in modules:
        for name in imported_modules(path):
            root = name.split(".")[0]
            offence = f"{path.relative_to(SOURCE)} imports {name}"
            assert not name.startswith(FORBIDDEN_TO_EXPERIMENT), offence
            assert root not in SCRIPT_MODULES, offence
