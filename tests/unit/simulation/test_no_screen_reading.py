"""Nothing that drives the clone may tap a coordinate or read a pixel.

Readiness, the round boundary and every decision are the bridge's own readings.
Screenshot classification was the previous oracle and it is renderer-dependent
and intermittently wrong (`M1B-E024`); a learned screen coordinate would make
the policy a function of a layout rather than of the game. Both were removed,
and this is what keeps them out.

It replaces a fixture that could only refuse the *one* call it stood in for, and
only in the tests that installed it: it read back from a live `AdbDevice` class
that had to keep existing for the guard to have something to patch. Read from
the source instead, so a route added tomorrow in a module no test imports is
still caught, and so nothing has to be kept alive to be forbidden.

Prose is exempt, and deliberately: a comment or a docstring saying "never a
screenshot" is the prohibition being recorded, not broken. Everything a running
line can reach — an identifier, and every string it hands to a shell — is not.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

#: What reading a pixel or touching the screen is spelled as. The adb commands
#: are checked as sequences of words rather than as one token, because they
#: reach the device as separate arguments (`"shell", "input", "tap"`).
FORBIDDEN = ("screencap", "screenshot", "uiautomator", "input tap")


def scanned_files() -> list[Path]:
    """Everything that can reach an instance: the simulation, and the runners."""
    return sorted((ROOT / "src" / "tower_rl" / "simulation").rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    )


def docstring_lines(tree: ast.Module) -> set[int]:
    """Every line a module, class or function docstring occupies.

    A docstring is where the prohibition is written down, so it is the one
    string that may name what it forbids.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def reachable_words(path: Path) -> str:
    """Every word a running line can reach, as one space-separated text.

    Identifiers and string *values*, with comments and docstrings left out and
    punctuation dropped, so `adb(instance, "shell", "input", "tap")` reads as
    `adb instance shell input tap` and the two-word commands are visible.
    """
    source = path.read_text()
    exempt = docstring_lines(ast.parse(source))
    words: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT or token.start[0] in exempt:
            continue
        if token.type == tokenize.NAME:
            words.append(token.string)
        elif token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):  # an f-string part, read as written
                value = token.string
            if isinstance(value, str):
                words.append(value)
    return " ".join(words).lower()


@pytest.mark.parametrize("path", scanned_files(), ids=lambda path: path.name)
def test_no_module_that_drives_the_clone_reads_a_pixel_or_taps(path: Path) -> None:
    text = reachable_words(path)
    found = [term for term in FORBIDDEN if term in text]
    assert found == [], f"{path.relative_to(ROOT)} reaches for {', '.join(found)}"


def test_the_scan_sees_a_tap_spelled_the_way_adb_takes_it(tmp_path: Path) -> None:
    """The guard is only as good as what the scan can see.

    Written to `tmp_path`, never into the tree it guards: a probe left in `src/`
    by a killed run would fail this suite for every later one.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        '"""A docstring may say screencap, because saying so is the rule."""\n'
        "# and so may a comment about uiautomator\n"
        'adb(instance, "shell", "input", "tap", "100", "200")\n'
    )
    text = reachable_words(probe)

    assert "input tap" in text, "an adb tap arrives as separate arguments"
    assert "screencap" not in text, "a docstring is the prohibition, not a breach"
    assert "uiautomator" not in text, "a comment is the prohibition, not a breach"


def test_every_file_that_can_reach_an_instance_is_actually_scanned() -> None:
    """A rule over an empty list passes by saying nothing."""
    names = {path.name for path in scanned_files()}
    assert {"bring_up.py", "frame_rate.py", "instance.py", "bridge.py", "fleet.py"} <= names
    assert {"clone_session.py", "run_actors.py", "run_episodes.py", "train.py"} <= names
