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

#: What `tower_rl.learning` may not reach for: everything in the package except
#: the environment. Stated as an allowance rather than as a list of neighbours,
#: because the rule decided is "learning imports only the environment" - a
#: denylist would have to be extended by hand for every package added, and a
#: package nobody remembered to name would pass by omission.
FORBIDDEN_TO_LEARNING = ("tower_rl",)
ALLOWED_TO_LEARNING = ("tower_rl.environment",)

#: What `tower_rl.simulation` may not reach for: everything in the package except
#: the environment. Stated as an allowance for the same reason learning's is —
#: the rule decided is "simulation imports only the environment", so a package
#: added later is forbidden by default rather than by somebody remembering it.
#: This is what keeps host checks and the learner out of the one place that
#: reaches a device: `find_android_tool` moved down here from `doctor` because
#: of this rule, rather than the rule being widened to let it stay.
FORBIDDEN_TO_SIMULATION = ("tower_rl",)
ALLOWED_TO_SIMULATION = ("tower_rl.environment",)

#: What `tower_rl.experiment` may not reach for. Device adapters and host checks
#: are the simulation side, and a script is where a run is composed: an
#: experiment that imported either could not be run from anywhere else.
FORBIDDEN_TO_EXPERIMENT = ("tower_rl.simulation", "tower_rl.doctor")

#: Modules that only exist beside an entry point. No package module may import
#: them at all, by any name.
SCRIPT_MODULES = ("train", "run_episodes", "run_actors", "clone_session", "compare_arms")

def imported_modules(path: Path, root: Path = SOURCE) -> set[str]:
    """Every `tower_rl` module name this file imports, however it spells it.

    `root` is the source root the file's own package is resolved against, so a
    probe file can be written under a temporary directory. The source tree is
    never written to by a test: a probe left in `src/` by a killed run would
    poison the package, and one written while another suite runs would fail its
    rules on an import it never made.

    Three spellings reach a module: `import a.b`, `from a.b import c`, and
    `from a import b`, where the imported name is itself a module. The third is
    resolved by joining the name onto the package, so a package cannot escape a
    rule by importing its neighbour as an attribute of `tower_rl`.

    A relative import is resolved against this file's own package, so one that
    climbs out of the package it lives in is reported under the name it
    actually reaches.
    """
    package = path.relative_to(root).parent.parts
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


def offences(
    package: str,
    forbidden: tuple[str, ...],
    root: Path = SOURCE,
    allowed: tuple[str, ...] = (),
) -> list[str]:
    """Every import from `package` that crosses a boundary it may not.

    One package against one set of forbidden prefixes, so a new boundary is a
    constant and a call rather than a new walker. `allowed` names the prefixes
    that survive a forbidden one, which is what lets a rule be written as an
    allowlist ("only the environment") instead of a list of neighbours.
    """
    directory = root / "tower_rl" / package
    modules = sorted(directory.rglob("*.py"))
    assert modules, f"the {package} package must exist to be checked"

    found: list[str] = []
    own = f"tower_rl.{package}"
    for path in modules:
        for name in imported_modules(path, root):
            if name == own or name.startswith(f"{own}."):
                continue
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed):
                continue
            offence = f"{path.relative_to(root)} imports {name}"
            if name.startswith(forbidden) or name.split(".")[0] in SCRIPT_MODULES:
                found.append(offence)
    return found


def test_the_environment_package_is_the_root_and_imports_nothing_above_it() -> None:
    """Every other package depends on the environment; it depends on none."""
    assert offences("environment", FORBIDDEN_TO_ENVIRONMENT) == []


def test_learning_reads_back_from_nothing_that_observes_or_drives_it() -> None:
    """Everything about learning lives here, and it knows only the environment."""
    assert offences("learning", FORBIDDEN_TO_LEARNING, allowed=ALLOWED_TO_LEARNING) == []


def test_simulation_knows_how_to_reach_a_device_and_nothing_about_the_run_on_it() -> None:
    """Everything device-facing lives here, and it knows only the environment.

    Not the learner, not the experiment observing it, and not a script: the
    adapter here is a `RunPort`, so a run can be driven without the simulation
    ever learning what is driving it.
    """
    assert offences("simulation", FORBIDDEN_TO_SIMULATION, allowed=ALLOWED_TO_SIMULATION) == []


def test_the_experiment_package_reaches_for_no_adapter_and_no_script() -> None:
    assert offences("experiment", FORBIDDEN_TO_EXPERIMENT) == []


def test_the_walker_sees_every_spelling_an_import_can_take(tmp_path: Path) -> None:
    """The rules are only as good as what the walker can see.

    `from tower_rl import doctor` and a relative import that climbs out of its
    own package both reach a module, and both were once invisible here.
    """
    source = (
        "import tower_rl.simulation.adb\n"
        "from tower_rl.experiment import metrics\n"
        "from tower_rl import doctor\n"
        "from ..doctor import find\n"
        "from . import run_state\n"
    )
    # Under `tmp_path`, never under `src`: the probe is resolved against the
    # root it is written to, so the real package is left alone.
    directory = tmp_path / "tower_rl" / "environment"
    directory.mkdir(parents=True)
    path = directory / "_walker_probe.py"
    path.write_text(source)
    names = imported_modules(path, tmp_path)

    assert "tower_rl.simulation.adb" in names
    assert "tower_rl.experiment.metrics" in names
    assert "tower_rl.doctor" in names, "the `from tower_rl import doctor` form"
    assert "tower_rl.doctor.find" in names, "a relative import climbing out of the package"
    assert "tower_rl.environment.run_state" in names
