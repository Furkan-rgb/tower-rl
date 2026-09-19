"""The dependency direction the packages are allowed to depend on each other in.

The rule is one-directional. `tower_rl.environment` is the root: what a run is,
what it is driven through, and what a decision costs - it may know nothing else
in the package. Simulation, learning and the experiment observing them build on
it in that order, so an experiment may know about the environment, the learner
and the ports they are driven through, and nothing about how a device is
reached or how a run is started.

The scripts sit outside that order entirely: they are the composition root, so
they import packages and no package imports them. The suite is held to the same
line - a test may exercise any package, but may reach an entry point only if it
is that entry point's own test.

Read from the source rather than from an import graph at runtime, so a module
that is never imported is still held to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
PACKAGES = SOURCE / "tower_rl"
TESTS = REPOSITORY / "tests"

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

#: What a test may not reach for: nothing in `tower_rl`. A test may exercise any
#: package, and `fakes` is the suite's own - `conftest.py` puts `tests/` on the
#: path, so the doubles are imported as `fakes.*`. The one rule tests are held
#: to is the standing one below, which `offences` applies to every tree it
#: walks: a test may not import an entry point it is not the test of.
FORBIDDEN_TO_TESTS: tuple[str, ...] = ()

#: Modules that only exist beside an entry point. No package module may import
#: them at all, by any name, and no test may import one either unless it is that
#: script's own test.
SCRIPT_MODULES = (
    "train",
    "run_episodes",
    "run_actors",
    "clone_session",
    "compare_arms",
    "select_checkpoint",
    "report_arms",
    "spectate",
    "diagnose_plasticity",
)

#: The test files allowed to import each entry point: the ones that test that
#: script. Named explicitly rather than derived from the filename, because the
#: names do not follow from each other - `test_multi_actor.py` is the test of
#: `run_actors.py` - and because an entry point picking up a new test reader
#: should be a deliberate line here rather than a rename nobody noticed.
#:
#: A script absent from this mapping is forbidden to every test, which is the
#: direction the default has to point: `run_episodes` and `compare_arms` have no
#: tests today, and listing a name for the file that would test them would be
#: writing down permission for a file nobody has read.
#:
#: `train` has four because the entry point is where a run is composed, and each
#: tests one thing that composition owns: its argument parsing and resolved run
#: identity, its tracker wiring, its report writing, and the session itself.
#: Nothing else in the suite may reach for it.
#: Two files are deliberately listed under several scripts, because the
#: behaviour they hold is itself spread across entry points and lives nowhere
#: else: the evaluation protocol runs a training session, plays its checkpoints
#: through the episode runner and reads them back with the two post-hoc
#: scripts, and the bridge-directory rule is one rule three runners obey.
SCRIPT_TESTS: dict[str, frozenset[str]] = {
    "train": frozenset(
        {
            "test_train_entry_point",
            "test_run_identity",
            "test_tracking",
            "test_training_report",
            # Trains the run whose numbered checkpoints are then selected among.
            "test_evaluation_protocol",
            # Where every runner finds the bridge it expects to be talking to.
            "test_script_bridge_directory",
        }
    ),
    "run_actors": frozenset(
        {
            "test_multi_actor",
            # The fleet's half of playing a checkpoint as an arm.
            "test_checkpoint_arm",
        }
    ),
    "clone_session": frozenset({"test_clone_session_cli"}),
    "run_episodes": frozenset(
        {
            # The policy selector and the actor record it writes live here.
            "test_checkpoint_arm",
            # Plays each arm of the protocol through the same selector.
            "test_evaluation_protocol",
            "test_script_bridge_directory",
            # The observation-batch recorder, which lives beside this entry
            # point and is read back by the diagnostic that test also holds.
            "test_diagnose_plasticity",
        }
    ),
    "compare_arms": frozenset({"test_script_bridge_directory"}),
    "select_checkpoint": frozenset({"test_evaluation_protocol"}),
    "report_arms": frozenset({"test_evaluation_protocol"}),
    # The panel's model, the exclusive-device refusal and the episode loop live
    # in the entry point, because a panel is composition: a view of the decision
    # stream drawn in a terminal, owning no domain concept of its own. This is
    # the one file that reads them.
    "spectate": frozenset({"test_spectate"}),
    # The capture seam and the diagnostic that reads what it writes are one
    # pipeline joined by a file format, and that format is the thing worth
    # holding; `run_episodes` picks up this reader for the wrapper that lives
    # beside its entry point and nothing else.
    "diagnose_plasticity": frozenset({"test_diagnose_plasticity"}),
}

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
    directory: Path,
    forbidden: tuple[str, ...],
    root: Path = SOURCE,
    allowed: tuple[str, ...] = (),
) -> list[str]:
    """Every import from the tree at `directory` that crosses a boundary it may not.

    One tree against one set of forbidden prefixes, so a new boundary is a
    constant and a call rather than a new walker. `root` is the import root the
    tree's own dotted name is taken against, which is what lets the same walker
    hold `src/tower_rl/<package>` and `tests/` to their rules. `allowed` names
    the prefixes that survive a forbidden one, which is what lets a rule be
    written as an allowlist ("only the environment") instead of a list of
    neighbours.

    A script module is refused in every tree, by any name, whatever `forbidden`
    says - unless the file is one of that script's own tests (`SCRIPT_TESTS`).
    """
    modules = sorted(directory.rglob("*.py"))
    assert modules, f"the tree at {directory} must exist to be checked"

    found: list[str] = []
    own = ".".join(directory.relative_to(root).parts)
    for path in modules:
        for name in imported_modules(path, root):
            if name == own or name.startswith(f"{own}."):
                continue
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in allowed):
                continue
            offence = f"{path.relative_to(root)} imports {name}"
            script = name.split(".")[0]
            if script in SCRIPT_MODULES:
                # A script with no entry at all is forbidden to everything: the
                # default has to be refusal, or adding an entry point would
                # quietly open it to the whole suite until somebody remembered
                # to write its rule down.
                if path.stem not in SCRIPT_TESTS.get(script, frozenset()):
                    found.append(offence)
                continue
            if name.startswith(forbidden):
                found.append(offence)
    return found


def test_the_environment_package_is_the_root_and_imports_nothing_above_it() -> None:
    """Every other package depends on the environment; it depends on none."""
    assert offences(PACKAGES / "environment", FORBIDDEN_TO_ENVIRONMENT) == []


def test_learning_reads_back_from_nothing_that_observes_or_drives_it() -> None:
    """Everything about learning lives here, and it knows only the environment."""
    assert offences(PACKAGES / "learning", FORBIDDEN_TO_LEARNING, allowed=ALLOWED_TO_LEARNING) == []


def test_simulation_knows_how_to_reach_a_device_and_nothing_about_the_run_on_it() -> None:
    """Everything device-facing lives here, and it knows only the environment.

    Not the learner, not the experiment observing it, and not a script: the
    adapter here is a `RunPort`, so a run can be driven without the simulation
    ever learning what is driving it.
    """
    assert (
        offences(PACKAGES / "simulation", FORBIDDEN_TO_SIMULATION, allowed=ALLOWED_TO_SIMULATION)
        == []
    )


def test_the_experiment_package_reaches_for_no_adapter_and_no_script() -> None:
    assert offences(PACKAGES / "experiment", FORBIDDEN_TO_EXPERIMENT) == []


def test_no_test_imports_an_entry_point_it_is_not_the_test_of() -> None:
    """A script is composition, and composition has one set of test readers.

    The packages are open to every test; the scripts are not. A test that
    imports `train` to borrow a fixture binds a suite nobody was reading to an
    entry point's argument parsing, which is how a script stops being free to
    change - and it is also how a package's behaviour ends up asserted through a
    composition root instead of through the package that owns it.
    """
    assert offences(TESTS, FORBIDDEN_TO_TESTS, root=REPOSITORY) == []


def test_a_script_nobody_wrote_a_rule_for_is_refused_rather_than_admitted(
    tmp_path: Path,
) -> None:
    """The default for an unlisted entry point is refusal, not permission.

    `run_episodes` and `compare_arms` are in `SCRIPT_MODULES` and absent from
    `SCRIPT_TESTS`, which is the case that decides which way the default points.
    Read from a probe rather than from the real suite, because the property has
    to hold for the entry point somebody adds next, which no file imports yet.
    """
    directory = tmp_path / "tests" / "unit"
    directory.mkdir(parents=True)
    (directory / "test_run_episodes.py").write_text("import run_episodes\n")

    assert offences(tmp_path / "tests", FORBIDDEN_TO_TESTS, root=tmp_path) == [
        "tests/unit/test_run_episodes.py imports run_episodes"
    ], "a script with no rule written for it is open to nobody, not to everybody"


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
