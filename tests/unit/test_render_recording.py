"""Composing a recording with the panel of what the agent did.

Everything here but the last test runs without ffmpeg: what is worth holding is
the placement - where a decision lands once the chunks are one video - and the
two files that placement is turned into, the subtitle the panel is drawn from
and the command that draws it. The encode itself is ffmpeg's, and the one test
that runs it renders two seconds of a colour generator rather than a recording.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import render_recording
from render_recording import Decision

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _decision(**overrides: object) -> Decision:
    fields: dict[str, object] = {
        "video_s": 12.5,
        "chunk": 0,
        "chunk_s": 12.5,
        "episode": 1,
        "decision": 3,
        "wave": 2,
        "cash": 1240.0,
        "health_fraction": 0.62,
        "action": "attack:2",
        "label": "Critical Chance",
        "held_s": 9.8,
        "hud": {
            "damage": 12.09,
            "attackSpeed": 1.05,
            "criticalChance": 5.0,
            "criticalMult": 1.2,
            "towerRangeDistance": 2.7,
            "towerHealthRegen": 0.0,
            "defenseAbs": 0.0,
            "defenseRel": 0.0,
            "wallHealth": 1.0,
            "closestEnemyDistance": 0.13,
            "enemiesKilledThisWave": 14.0,
            "enemiesSpawnedThisWave": 28.0,
            "estimatedEnemiesToSpawnThisWave": 26.0,
            "waveTimer": 22.0,
            "waveLengthSeconds": 26.0,
            "waveCooldownSeconds": 9.0,
            "currentWaveBaseHealth": 8.7,
            "currentWaveBaseDamage": 2.68,
            "gameplayTimeThisRound": 197.0,
        },
        "ended": False,
        "reason": None,
    }
    fields.update(overrides)
    return Decision(**fields)  # type: ignore[arg-type]


def _track(path: Path, decisions: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in decisions))
    return path


# -- reading what the session wrote ----------------------------------------


def test_a_track_is_read_back_as_the_decisions_it_records(tmp_path: Path) -> None:
    track = _track(
        tmp_path / "session.decisions.jsonl",
        [
            {
                "video_s": 1.5, "chunk": 0, "chunk_s": 1.5, "episode": 1, "decision": 1,
                "wave": 1, "cash": 75.0, "health_fraction": 1.0, "action": "wait",
                "label": "Hold", "held_s": 0.5, "hud": {"damage": 3.0}, "ended": False,
            },
            {
                "video_s": 9.0, "chunk": 0, "chunk_s": 9.0, "episode": 1, "decision": 2,
                "wave": 1, "cash": 0.0, "health_fraction": 0.0, "action": "attack:0",
                "label": "Damage", "held_s": 7.5, "hud": {"damage": 5.9}, "ended": True,
                "reason": "game_over",
            },
        ],
    )

    first, second = render_recording.read_decisions(track)

    assert (first.decision, first.label, first.ended) == (1, "Hold", False)
    assert (second.label, second.ended, second.reason) == ("Damage", True, "game_over")
    assert second.hud == {"damage": 5.9}


def test_a_recording_with_no_track_beside_it_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no decision track"):
        render_recording.read_decisions(tmp_path / "session.decisions.jsonl")


def test_the_chunks_are_taken_in_the_order_they_were_recorded(tmp_path: Path) -> None:
    for name in ("session-002.mp4", "session-000.mp4", "session-001.mp4", "other-000.mp4"):
        (tmp_path / name).write_text("video")

    chunks = render_recording.chunk_paths(tmp_path / "session")

    assert [chunk.name for chunk in chunks] == [
        "session-000.mp4", "session-001.mp4", "session-002.mp4",
    ]


def test_a_stem_given_as_a_chunk_filename_is_the_same_recording() -> None:
    """Nobody reading a directory listing types the stem."""
    for spelling in ("state/recordings/run", "state/recordings/run-000.mp4"):
        parsed = render_recording.parse_arguments(["--recording", spelling])
        assert parsed.recording == Path("state/recordings/run")


# -- placing a decision once the chunks are one video ----------------------


def test_a_decision_is_placed_by_its_chunk_and_not_by_wall_time() -> None:
    """The seam loses a second of wall time that the video does not contain.

    Two chunks of 180 and 60 seconds: a decision 19 seconds into the second
    chunk is at 199 in the concatenated video, whatever its `video_s` says -
    and its `video_s` says 200, because the seam cost a second nobody recorded.
    """
    decisions = [
        _decision(chunk=0, chunk_s=30.0, video_s=30.0),
        _decision(chunk=1, chunk_s=19.0, video_s=200.0),
    ]

    assert render_recording.concatenated_starts(decisions, [180.0, 60.0]) == [30.0, 199.0]


def test_a_decision_in_a_chunk_that_was_never_pulled_lands_at_the_end_of_what_there_is() -> None:
    """A lost last chunk must not throw away the decisions taken during it."""
    decisions = [_decision(chunk=2, chunk_s=5.0)]

    assert render_recording.concatenated_starts(decisions, [180.0]) == [180.0]


# -- what the panel says ---------------------------------------------------


def test_the_panel_holds_the_readings_the_watcher_saw_and_the_history_it_could_not() -> None:
    decisions = [
        _decision(decision=number, label=f"Row {number}", held_s=float(number))
        for number in range(1, 20)
    ]

    lines = render_recording.panel_lines(decisions, len(decisions) - 1)
    text = "\n".join(lines)

    assert "episode 1   decision 19" in text
    assert "wave 2   cash 1,240   health 62%" in text
    assert "dmg 12.1" in text and "crit% 5.0" in text
    assert "enemies 14/28 of ~26   nearest 0.13" in text
    assert "wave clock 22.0/26+9s" in text
    history = [line for line in lines if line.startswith((">", " d", "  d"))]
    assert len(history) == render_recording.HISTORY
    assert history[-1].startswith("> d19"), "the current decision is the marked one"
    assert "Row 8" in history[0], "the history is the last twelve, oldest first"


def test_a_history_line_says_which_upgrade_was_bought_the_level_and_the_cost() -> None:
    """The row a watcher could not identify from the label alone."""
    bought = _decision(
        decision=7,
        label="Damage",
        purchase=render_recording.Purchase(
            label="Damage", level_after=4, max_level=20, cost=120.0
        ),
    )

    (line,) = [
        text
        for text in render_recording.panel_lines([bought], 0)
        if text.startswith("> d7")
    ]

    assert line.rstrip() == "> d7    Damage             \u2192 L4/20  -120"


def test_a_history_line_for_a_hold_says_how_long_it_was_held() -> None:
    (line,) = [
        text
        for text in render_recording.panel_lines(
            [_decision(decision=7, label="Hold", action="wait", held_s=2.0)], 0
        )
        if text.startswith("> d7")
    ]

    assert line.rstrip() == "> d7    Hold                 2.0s"


def test_a_row_name_longer_than_the_column_is_cut_rather_than_running_off_the_panel() -> None:
    assert render_recording.short_label("Free Upgrade Chance") == "Free Upgrade Chan\u2026"
    assert render_recording.short_label("Damage") == "Damage"


def test_a_track_written_before_purchases_still_renders_its_history(tmp_path: Path) -> None:
    """The M2-E006 preview was recorded under the old line shape and must keep rendering."""
    line = {
        "video_s": 1.0, "chunk": 0, "chunk_s": 1.0, "episode": 1, "decision": 4,
        "wave": 2, "cash": 300.0, "health_fraction": 0.9, "action": "attack:2",
        "label": "Critical Chance", "held_s": 3.5, "hud": {"damage": 3.0},
        "ended": False,
    }
    path = tmp_path / "old.decisions.jsonl"
    path.write_text(json.dumps(line) + "\n")

    (decision,) = render_recording.read_decisions(path)
    (history,) = [
        text
        for text in render_recording.panel_lines([decision], 0)
        if text.startswith("> d4")
    ]

    assert decision.purchase is None
    assert history.rstrip() == "> d4    Critical Chance      3.5s"


def test_the_death_is_on_the_panel_of_the_decision_that_ended_the_episode() -> None:
    lines = render_recording.panel_lines([_decision(ended=True, reason="game_over")], 0)

    assert "episode 1 ended at wave 2: game_over" in lines


def test_a_reading_the_track_did_not_carry_is_drawn_as_missing_rather_than_guessed() -> None:
    lines = render_recording.panel_lines([_decision(hud={"damage": 3.0})], 0)

    assert "dmg 3.0" in "\n".join(lines)
    assert "aspd -" in "\n".join(lines)


# -- the subtitle the panel is drawn from ----------------------------------


def test_the_subtitle_holds_one_event_per_decision_timed_to_the_next() -> None:
    decisions = [
        _decision(decision=1, chunk=0, chunk_s=0.5, label="Hold", action="wait"),
        _decision(decision=2, chunk=0, chunk_s=12.25, label="Damage"),
    ]

    document = render_recording.ass_document(
        decisions,
        render_recording.concatenated_starts(decisions, [30.0]),
        width=360,
        height=640,
        total_seconds=30.0,
    )
    events = [line for line in document.splitlines() if line.startswith("Dialogue:")]

    assert document.startswith("[Script Info]")
    assert "PlayResX: 920" in document, "the panel is beside the picture, not over it"
    assert "PlayResY: 640" in document
    assert "DejaVu Sans Mono" in document
    assert len(events) == 2
    assert events[0].startswith("Dialogue: 0,0:00:00.50,0:00:12.25,panel,,0,0,0,,")
    assert events[1].startswith("Dialogue: 0,0:00:12.25,0:00:30.00,panel,,0,0,0,,")
    assert "{\\pos(370,10)}" in events[0], "anchored in the padding, left of nothing"
    assert events[1].count("\\N") == len(render_recording.panel_lines(decisions, 1)) - 1
    assert render_recording.HIGHLIGHT_COLOUR in events[1], "the current decision is picked out"


def test_the_clock_the_subtitle_is_timed_on() -> None:
    assert render_recording.ass_time(0.0) == "0:00:00.00"
    assert render_recording.ass_time(12.345) == "0:00:12.34"
    assert render_recording.ass_time(3725.5) == "1:02:05.50"
    assert render_recording.ass_time(-1.0) == "0:00:00.00"


def test_a_brace_out_of_the_game_is_not_read_as_an_override() -> None:
    """ASS spells its own commands in braces; a label may not become one."""
    body = render_recording.ass_text(["  held {\\c&HFF0000&}", "plain"], current=1)

    assert body.startswith("\\h\\h"), "an indent survives as a hard space"
    assert "{\\c&HFF0000&}" not in body
    assert "(/c&HFF0000&)" in body
    assert body.endswith(f"{{\\c{render_recording.TEXT_COLOUR}}}")


# -- the command, assembled -------------------------------------------------


def test_the_render_is_one_pass_that_pads_the_picture_and_draws_into_it(tmp_path: Path) -> None:
    command = render_recording.ffmpeg_command(
        ffmpeg="/usr/bin/ffmpeg",
        concat_list=tmp_path / "chunks.txt",
        subtitles=tmp_path / "panel.ass",
        fonts_directory=tmp_path / "fonts",
        output=tmp_path / "session-panel.mp4",
    )

    assert command == [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(tmp_path / "chunks.txt"),
        "-vf",
        f"pad=iw+560:ih:0:0:color={render_recording.PANEL_BACKGROUND},"
        f"ass={tmp_path / 'panel.ass'}:fontsdir={tmp_path / 'fonts'}",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(tmp_path / "session-panel.mp4"),
    ]
    assert "scale" not in " ".join(command), "the guest picture is never re-encoded larger"


def test_a_path_the_filter_parser_would_read_as_syntax_is_escaped() -> None:
    assert render_recording.escape_filter_path(Path("/tmp/a:b/panel.ass")) == "/tmp/a\\:b/panel.ass"


def test_a_host_without_ffmpeg_is_told_which_tool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="ffmpeg is not on PATH"):
        render_recording.require_tool("ffmpeg")


def test_a_host_with_no_monospace_font_is_told_which_ones_were_looked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        render_recording, "MONOSPACE_FONTS", (("/nowhere/Mono.ttf", "Nowhere Mono"),)
    )

    with pytest.raises(SystemExit, match="Nowhere Mono"):
        render_recording.require_monospace()


# -- one pass of the real thing --------------------------------------------


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="ffmpeg is not installed")
def test_two_seconds_of_colour_come_out_as_a_video_with_a_panel_beside_it(
    tmp_path: Path,
) -> None:
    """The one test that encodes: a synthetic clip, not a recording.

    What it holds is the join between everything above and ffmpeg — that the
    filter, the subtitle and the concat list are accepted as written, and that
    the result is the source picture with the panel's width added to it.
    """
    assert FFMPEG is not None and FFPROBE is not None
    stem = tmp_path / "synthetic"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=10:d=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(stem.with_name("synthetic-000.mp4")),
        ],
        check=True,
        timeout=120.0,
    )
    _track(
        stem.with_name("synthetic.decisions.jsonl"),
        [
            {
                "video_s": 0.2, "chunk": 0, "chunk_s": 0.2, "episode": 1, "decision": 1,
                "wave": 1, "cash": 75.0, "health_fraction": 1.0, "action": "wait",
                "label": "Hold", "held_s": 0.2, "hud": {"damage": 3.0}, "ended": False,
            },
            {
                "video_s": 1.1, "chunk": 0, "chunk_s": 1.1, "episode": 1, "decision": 2,
                "wave": 1, "cash": 0.0, "health_fraction": 0.0, "action": "attack:0",
                "label": "Damage", "held_s": 0.9, "hud": {"damage": 5.9}, "ended": True,
                "reason": "game_over",
            },
        ],
    )

    output = render_recording.render(stem)

    assert output.name == "synthetic-panel.mp4"
    probed = json.loads(
        subprocess.run(
            [
                FFPROBE, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60.0,
        ).stdout
    )
    stream = probed["streams"][0]
    assert (stream["width"], stream["height"]) == (320 + render_recording.PANEL_WIDTH, 240)
