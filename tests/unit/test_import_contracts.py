"""The dependency direction the packages are allowed to depend on each other in.

The rule is one-directional. `tower_rl.environment` is the root: what a run is,
what it is driven through, and what a decision costs - it may know nothing else
in the package. Simulation, learning and the experiment observing them build on
it in that order, so an experiment may know about the environment, the learner
and the ports they are driven through, and nothing about how a device is
reached or how a run is started.

Read from the source rather than from an import graph at runtime, so a module
that is never imported is still held to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src"

#: What `tower_rl.environment` may not reach for: anything else in the package.
#: It is the root, so every other name in `tower_rl` is below it.
FORBIDDEN_TO_ENVIRONMENT = ("tower_rl",)

#: What a use case may not reach for. An experiment observes training; training
#: must not read back from what observes it, or the loop could not run without
#: the reporting it is only measured by.
FORBIDDEN_TO_APPLICATION = ("tower_rl.experiment",)

#: What `tower_rl.experiment` may not reach for. Device adapters and host checks
#: are the simulation side, and a script is where a run is composed: an
#: experiment that imported either could not be run from anywhere else.
FORBIDDEN_TO_EXPERIMENT = ("tower_rl.infrastructure", "tower_rl.doctor")

#: Modules that only exist beside an entry point. No package module may import
#: them at all, by any name.
SCRIPT_MODULES = ("train", "run_episodes", "run_actors", "clone_session", "compare_arms")

def imported_modules(path: Path) -> set[str]:
    """Every `tower_rl` module name this file imports, however it spells it.

    Three spellings reach a module: `import a.b`, `from a.b import c`, and
    `from a import b`, where the imported name is itself a module. The third is
    resolved by joining the name onto the package, so a package cannot escape a
    rule by importing its neighbour as an attribute of `tower_rl`.

    A relative import is resolved against this file's own package, so one that
    climbs out of the package it lives in is reported under the name it
    actually reaches.
    """
    package = path.relative_to(SOURCE).parent.parts
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package[: len(package) - node.level + 1])
            else:
                base = (node.module or "").split(".")
            if node.module and node.level:
                base += node.module.split(".")
            prefix = ".".join(base)
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def offences(package: str, forbidden: tuple[str, ...]) -> list[str]:
    """Every import from `package` that crosses a boundary it may not.

    One package against one set of forbidden prefixes, so a new boundary is a
    constant and a call rather than a new walker.
    """
    directory = SOURCE / "tower_rl" / package
    modules = sorted(directory.rglob("*.py"))
    assert modules, f"the {package} package must exist to be checked"

    found: list[str] = []
    own = f"tower_rl.{package}"
    for path in modules:
        for name in imported_modules(path):
            if name == own or name.startswith(f"{own}."):
                continue
            offence = f"{path.relative_to(SOURCE)} imports {name}"
            if name.startswith(forbidden) or name.split(".")[0] in SCRIPT_MODULES:
                found.append(offence)
    return found


def test_the_environment_package_is_the_root_and_imports_nothing_above_it() -> None:
    """Every other package depends on the environment; it depends on none."""
    assert offences("environment", FORBIDDEN_TO_ENVIRONMENT) == []


def test_no_use_case_reads_back_from_what_observes_it() -> None:
    assert offences("application", FORBIDDEN_TO_APPLICATION) == []
    assert offences("learning", FORBIDDEN_TO_APPLICATION) == []


def test_the_experiment_package_reaches_for_no_adapter_and_no_script() -> None:
    assert offences("experiment", FORBIDDEN_TO_EXPERIMENT) == []


def test_the_walker_sees_every_spelling_an_import_can_take() -> None:
    """The rules are only as good as what the walker can see.

    `from tower_rl import doctor` and a relative import that climbs out of its
    own package both reach a module, and both were once invisible here.
    """
    source = (
        "import tower_rl.infrastructure.adb\n"
        "from tower_rl.experiment import metrics\n"
        "from tower_rl import doctor\n"
        "from ..doctor import find\n"
        "from . import run_state\n"
    )
    path = SOURCE / "tower_rl" / "environment" / "_walker_probe.py"
    try:
        path.write_text(source)
        names = imported_modules(path)
    finally:
        path.unlink()

    assert "tower_rl.infrastructure.adb" in names
    assert "tower_rl.experiment.metrics" in names
    assert "tower_rl.doctor" in names, "the `from tower_rl import doctor` form"
    assert "tower_rl.doctor.find" in names, "a relative import climbing out of the package"
    assert "tower_rl.environment.run_state" in names
