"""Where this project writes: one directory, resolved one way, and nowhere else.

The decision is that no project state lives on the host outside the project —
bridge builds, runs and checkpoints, the MLflow store, evaluation records and
spectate recordings all land under the git-ignored `<repo>/state/`. The resolver
is `tower_rl.environment.project_state`, and these tests hold both halves of the
decision: that the resolver answers with the checkout it is part of, and that no
entry point still names a location outside it.

The defaults are read from the source with `ast` rather than by importing seven
entry points, for the reason `test_import_contracts.py` reads from source too: a
default that nobody imports is still held to the rule, and this file does not
become a test reader of every script in the repository to check one line in each.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tower_rl.environment.project_state import repository_root, state_directory
from tower_rl.simulation import bridge

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY / "scripts"

#: The home-directory location project state used to live in. It may appear in
#: `docs/experiments.md`, where it is part of the record of where a past run's
#: evidence was read from, and nowhere in the code.
FORMER_LOCATION = ".local/state/tower-rl"

#: The argument defaults that name a place this project writes to. Each is an
#: entry point and the arguments of its that must resolve through the state
#: directory; an entry point writing somewhere else is the thing this catches.
WRITING_ARGUMENTS = {
    "train.py": {"--run-dir"},
    "run_actors.py": {"--output", "--output-directory"},
    "run_episodes.py": {"--output"},
    "compare_arms.py": {"--output"},
    "report_arms.py": {"--output", "--output-directory"},
    "select_checkpoint.py": {"--output", "--run-dir"},
    "spectate.py": {"--output-directory"},
}


def default_expressions(source: Path) -> dict[str, str]:
    """Each `add_argument` flag in one script, against its `default=` as source."""
    defaults: dict[str, str] = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        for keyword in node.keywords:
            if keyword.arg == "default":
                for flag in flags:
                    defaults[str(flag)] = ast.unparse(keyword.value)
    return defaults


def test_the_state_directory_is_the_checkout_this_package_is_part_of() -> None:
    """Resolved from the package's own location, so the cwd cannot move it."""
    assert repository_root() == REPOSITORY
    assert state_directory() == REPOSITORY / "state"
    assert state_directory().parent == repository_root()


def test_the_state_directory_is_git_ignored_and_the_resolver_creates_nothing() -> None:
    """It holds artifacts a public repository must never carry, so it is ignored.

    And it is a location, not a side effect: a script that wants a subdirectory
    makes its own, so importing the resolver on a fresh checkout leaves the tree
    exactly as it found it.
    """
    ignored = (REPOSITORY / ".gitignore").read_text().splitlines()
    assert "state/" in ignored
    assert "recordings/" not in ignored, "the old spectate-only rule is subsumed by state/"

    existed = state_directory().exists()
    state_directory()
    assert state_directory().exists() == existed


def test_the_installed_bridge_lives_under_the_state_directory() -> None:
    """`state/bridge/<sha256>/`, with `current` the symlink to the deployed one."""
    assert state_directory() / "bridge" == bridge.BRIDGE_STATE_DIRECTORY


@pytest.mark.parametrize("script", sorted(WRITING_ARGUMENTS), ids=sorted(WRITING_ARGUMENTS))
def test_every_entry_point_defaults_to_writing_inside_the_state_directory(
    script: str,
) -> None:
    """No `/tmp` default, no home-directory default: one place, for all of them.

    `/tmp` was where the evaluation records defaulted, which put a measurement's
    evidence in a directory the host clears on reboot; the home directory was
    where runs and the store went, which put project state outside the project.
    """
    defaults = default_expressions(SCRIPTS / script)
    for flag in WRITING_ARGUMENTS[script]:
        assert flag in defaults, f"{script} no longer has {flag}"
        expression = defaults[flag]
        assert "state_directory()" in expression or "RECORDINGS_DIRECTORY" in expression, (
            f"{script} {flag} defaults to {expression}"
        )
        assert "/tmp" not in expression and "home()" not in expression


def test_nothing_in_the_code_still_points_at_the_former_home_directory_location() -> None:
    """The move is complete only if no source or entry point names the old tree.

    `scripts/migrate_state.py` is the exception and is excluded by name: moving
    the old location is precisely what it is for, and it is the one file that may
    still say where that location was.
    """
    trees = sorted((REPOSITORY / "src").rglob("*.py")) + sorted(SCRIPTS.glob("*"))
    offenders = [
        str(path.relative_to(REPOSITORY))
        for path in trees
        if path.is_file()
        if path.name != "migrate_state.py"
        if FORMER_LOCATION in path.read_text()
    ]
    assert offenders == []
