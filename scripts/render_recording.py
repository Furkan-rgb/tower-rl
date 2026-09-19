#!/usr/bin/env python3
"""Put a spectated recording and what the agent did into one video.

`spectate.py --record` leaves two things under `state/recordings/`: the guest
screen as numbered chunks, and `<stem>.decisions.jsonl` — one line per decision,
placed in the video. This composes them: the chunks are concatenated in order
and the game picture is padded to the right with a panel that says, at every
moment, what the agent was looking at and what it had just done.

    uv run python scripts/render_recording.py --recording state/recordings/session

The result is `<stem>-panel.mp4` beside the input. Composition only: nothing
here reads the game, the device, or an episode record, and every number on the
panel was written by the session that played it.

The panel is drawn by `libass` from a generated subtitle file rather than frame
by frame. A decision is one subtitle event that lasts until the next decision,
which is exactly the shape of the data — the panel changes when the agent does
something and not otherwise — and it is one ffmpeg pass over the video instead
of a composite per frame.

**The panel leads the picture slightly.** Times in the jsonl are anchored at
the moment the host asked the guest to record, and the guest's first frame
comes an adb round trip and a process start later — a few hundred milliseconds,
under a second. A chunk seam loses about a second the same way. This is a
human-facing path; nothing measured is read off this alignment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: How wide the panel is, in pixels, beside the guest picture. The guest is
#: never scaled — the output is the source picture plus this — so a rendered
#: recording is the same video it was, with something written next to it.
PANEL_WIDTH = 560
#: Point size of the panel text, and where its first line starts relative to
#: the top-left corner of the panel.
FONT_SIZE = 14
PANEL_MARGIN = 10

#: How many past decisions the panel keeps on screen. Twelve is what reads as
#: "what just happened" without the history becoming the thing being watched.
HISTORY = 12

#: The panel's background, and the two colours the text is drawn in: ASS spells
#: a colour `&HAABBGGRR`, which is BGR with an alpha in front.
PANEL_BACKGROUND = "0x10131a"
TEXT_COLOUR = "&H00D8D8D8"
HIGHLIGHT_COLOUR = "&H0000E5FF"

#: Where a monospace font is looked for, most-preferred first, with the family
#: name `libass` knows it by. A proportional font would make every column of
#: the panel move as the numbers changed, so this is a requirement rather than
#: a preference; if none of these is present the render fails by name.
MONOSPACE_FONTS: tuple[tuple[str, str], ...] = (
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "DejaVu Sans Mono"),
    ("/usr/share/fonts/TTF/DejaVuSansMono.ttf", "DejaVu Sans Mono"),
    ("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", "Liberation Mono"),
    ("/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf", "Noto Sans Mono"),
    ("/System/Library/Fonts/Menlo.ttc", "Menlo"),
    ("/System/Library/Fonts/SFNSMono.ttf", "SF Mono"),
)

#: The encode. The guest picture is copied through at its own resolution and
#: only the padding is new, so the encode is chosen to keep that picture rather
#: than to compress it: x264 at CRF 20 is visually transparent for a 360x640
#: screen capture, `veryfast` because the bottleneck is nobody's, and
#: `yuv420p` because anything else will not play in a browser. No audio: a
#: guest recording has none.
ENCODE = ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p")


@dataclass(frozen=True)
class Decision:
    """One line of the decision track, as the panel needs it."""

    video_s: float
    chunk: int
    chunk_s: float
    episode: int
    decision: int
    wave: int
    cash: float
    health_fraction: float
    action: str
    label: str
    held_s: float
    hud: dict[str, float]
    ended: bool
    reason: str | None


def read_decisions(path: Path) -> list[Decision]:
    """Read the track `spectate.py --record` wrote, in the order it wrote it.

    A line missing a field is a track from another version of the session
    script, and it fails here by name rather than rendering a panel with a hole
    in it.
    """
    if not path.exists():
        raise SystemExit(
            f"no decision track at {path}. A recording made before the track "
            "existed has none; play a session with `spectate.py --record` to get one."
        )
    decisions: list[Decision] = []
    for number, text in enumerate(path.read_text().splitlines(), start=1):
        if not text.strip():
            continue
        line = json.loads(text)
        try:
            decisions.append(
                Decision(
                    video_s=float(line["video_s"]),
                    chunk=int(line["chunk"]),
                    chunk_s=float(line["chunk_s"]),
                    episode=int(line["episode"]),
                    decision=int(line["decision"]),
                    wave=int(line["wave"]),
                    cash=float(line["cash"]),
                    health_fraction=float(line["health_fraction"]),
                    action=str(line["action"]),
                    label=str(line["label"]),
                    held_s=float(line["held_s"]),
                    hud={name: float(value) for name, value in line["hud"].items()},
                    ended=bool(line["ended"]),
                    reason=line.get("reason"),
                )
            )
        except KeyError as missing:
            raise SystemExit(f"{path} line {number} has no {missing}") from missing
    if not decisions:
        raise SystemExit(f"{path} holds no decisions")
    return decisions


def chunk_paths(stem: Path) -> list[Path]:
    """The recording's chunks, in the order they were recorded.

    `spectate.py` numbers them `<stem>-000.mp4`, `<stem>-001.mp4`, and the
    numbering is the order: sorting the names sorts the video.
    """
    chunks = sorted(stem.parent.glob(f"{stem.name}-[0-9][0-9][0-9].mp4"))
    if not chunks:
        raise SystemExit(f"no recording chunks at {stem}-000.mp4")
    return chunks


# -- what the panel says ---------------------------------------------------


def _stat(hud: dict[str, float], wire: str, form: str) -> str:
    return form.format(hud[wire]) if wire in hud else "-"


def panel_lines(decisions: Sequence[Decision], index: int) -> list[str]:
    """The panel as of `decisions[index]`, top to bottom.

    Pure, and the whole of the rendering decision: what is drawn is decided
    here and `libass` only puts it on the picture. The same readings the
    terminal panel showed the watcher live, in the same units, plus the history
    a video can hold and a terminal cannot.
    """
    now = decisions[index]
    hud = now.hud
    lines = [
        f"episode {now.episode}   decision {now.decision}   {now.video_s:,.0f}s",
        "",
        f"wave {now.wave}   cash {now.cash:,.0f}   health {now.health_fraction:.0%}",
        f"held {now.held_s:.1f}s   round {_stat(hud, 'gameplayTimeThisRound', '{:.0f}')}s",
        "",
        f"dmg {_stat(hud, 'damage', '{:.1f}')}   "
        f"aspd {_stat(hud, 'attackSpeed', '{:.2f}')}   "
        f"crit% {_stat(hud, 'criticalChance', '{:.1f}')}   "
        f"critx {_stat(hud, 'criticalMult', '{:.2f}')}",
        f"range {_stat(hud, 'towerRangeDistance', '{:.2f}')}   "
        f"regen {_stat(hud, 'towerHealthRegen', '{:.3f}')}   "
        f"wall {_stat(hud, 'wallHealth', '{:.1f}')}",
        f"defabs {_stat(hud, 'defenseAbs', '{:.1f}')}   "
        f"def% {_stat(hud, 'defenseRel', '{:.1f}')}",
        "",
        f"enemies {_stat(hud, 'enemiesKilledThisWave', '{:.0f}')}/"
        f"{_stat(hud, 'enemiesSpawnedThisWave', '{:.0f}')} of "
        f"~{_stat(hud, 'estimatedEnemiesToSpawnThisWave', '{:.0f}')}   "
        f"nearest {_stat(hud, 'closestEnemyDistance', '{:.2f}')}",
        f"wave clock {_stat(hud, 'waveTimer', '{:.1f}')}/"
        f"{_stat(hud, 'waveLengthSeconds', '{:.0f}')}"
        f"+{_stat(hud, 'waveCooldownSeconds', '{:.0f}')}s   "
        f"base hp {_stat(hud, 'currentWaveBaseHealth', '{:.1f}')}   "
        f"base dmg {_stat(hud, 'currentWaveBaseDamage', '{:.2f}')}",
        "",
    ]
    if now.ended:
        lines.append(f"episode {now.episode} ended at wave {now.wave}: {now.reason or 'unknown'}")
    else:
        lines.append("")
    lines.append("")
    lines.append(f"last {HISTORY} decisions:")
    for position in range(max(0, index - HISTORY + 1), index + 1):
        past = decisions[position]
        marker = ">" if position == index else " "
        lines.append(f"{marker} d{past.decision:<4} {past.label:<18} {past.held_s:>5.1f}s")
    return lines


# -- the subtitle the panel is drawn from ----------------------------------


def ass_time(seconds: float) -> str:
    """ASS's own clock: `H:MM:SS.cc`, centiseconds, hours never padded.

    Truncated rather than rounded, so a panel never starts a centisecond before
    the decision it belongs to - which would leave the one before it ending
    after it began, and libass drawing two panels over each other.
    """
    centiseconds = int(max(0.0, seconds) * 100)
    hours, rest = divmod(centiseconds, 360_000)
    minutes, rest = divmod(rest, 6_000)
    whole, hundredths = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{hundredths:02d}"


def ass_text(lines: Sequence[str], current: int) -> str:
    """The panel's lines as one ASS event body, the current decision picked out.

    Braces are the override syntax, so any that came out of the game are turned
    into brackets rather than escaped; a leading space is `\\h`, because ASS
    drops ordinary ones at the start of a line and the history would lose its
    indent.
    """
    drawn: list[str] = []
    for number, line in enumerate(lines):
        text = line.replace("\\", "/").replace("{", "(").replace("}", ")")
        stripped = text.lstrip(" ")
        text = "\\h" * (len(text) - len(stripped)) + stripped
        if number == current:
            text = f"{{\\c{HIGHLIGHT_COLOUR}}}{text}{{\\c{TEXT_COLOUR}}}"
        drawn.append(text)
    return "\\N".join(drawn)


def ass_document(
    decisions: Sequence[Decision],
    starts: Sequence[float],
    *,
    width: int,
    height: int,
    total_seconds: float,
    panel_width: int = PANEL_WIDTH,
    font_family: str = MONOSPACE_FONTS[0][1],
) -> str:
    """The whole panel as one subtitle file: one event per decision.

    `starts` is each decision's position in the *concatenated* video, which is
    not its `video_s`: the seconds lost at each chunk seam are not in the video
    and `concatenated_starts` takes them out.
    """
    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width + panel_width}",
            f"PlayResY: {height}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
            " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
            " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: panel,{font_family},{FONT_SIZE},{TEXT_COLOUR},{TEXT_COLOUR},&H00000000,"
            "&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    events = []
    for index in range(len(decisions)):
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else max(total_seconds, start + 1.0)
        if end <= start:
            continue
        lines = panel_lines(decisions, index)
        body = ass_text(lines, current=len(lines) - 1)
        position = f"{{\\pos({width + PANEL_MARGIN},{PANEL_MARGIN})}}"
        events.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},panel,,0,0,0,,{position}{body}"
        )
    return header + "\n" + "\n".join(events) + "\n"


def concatenated_starts(
    decisions: Sequence[Decision], durations: Sequence[float]
) -> list[float]:
    """Where each decision lands once the chunks are one video.

    A decision carries the chunk it happened in and how far into that chunk it
    was, because that is the only placement a seam cannot move: `video_s` is
    wall time since the recording began and includes the second each seam loses
    while the next `screenrecord` starts. Here that wall time is dropped and
    the chunk's own offset is added to the chunks before it.

    A decision naming a chunk that was never pulled - a session whose last
    chunk was lost - is placed at the end of what there is rather than dropped.
    """
    offsets = [0.0]
    for duration in durations:
        offsets.append(offsets[-1] + duration)
    placed: list[float] = []
    for decision in decisions:
        chunk = min(max(decision.chunk, 0), len(durations) - 1)
        within = min(decision.chunk_s, durations[chunk])
        if decision.chunk > chunk:  # a chunk that is not in the video
            within = durations[chunk]
        placed.append(offsets[chunk] + within)
    return placed


# -- the tools this needs, by name -----------------------------------------


def require_tool(name: str) -> str:
    """The path to `name`, or a refusal that says what is missing."""
    found = shutil.which(name)
    if found is None:
        raise SystemExit(
            f"{name} is not on PATH, and composing a recording needs it "
            "(Debian/Ubuntu: `sudo apt install ffmpeg`; macOS: `brew install ffmpeg`)"
        )
    return found


def require_monospace() -> tuple[Path, str]:
    """A monospace font present on this host, as a file and a family name."""
    for path, family in MONOSPACE_FONTS:
        if Path(path).exists():
            return Path(path), family
    raise SystemExit(
        "no monospace font found; the panel is columns of numbers and a "
        "proportional font makes them move. Looked for: "
        + ", ".join(f"{family} ({path})" for path, family in MONOSPACE_FONTS)
    )


def probe_video(ffprobe: str, path: Path) -> tuple[int, int, float]:
    """One chunk's picture size and duration, read from the file itself."""
    output = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60.0,
    ).stdout
    probed = json.loads(output)
    stream = probed["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(probed["format"]["duration"])


# -- the render ------------------------------------------------------------


def escape_filter_path(path: Path) -> str:
    """A path as ffmpeg's filter parser reads it: `\\` and `:` mean something."""
    return str(path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def ffmpeg_command(
    *,
    ffmpeg: str,
    concat_list: Path,
    subtitles: Path,
    fonts_directory: Path,
    output: Path,
    panel_width: int = PANEL_WIDTH,
) -> list[str]:
    """The one pass that makes the composed video.

    Concat demuxer in, one filter chain, one encode out: the chunks become one
    stream, the picture is padded to the right by `panel_width` - the guest is
    never scaled, so nothing is re-encoded above the resolution it was captured
    at - and `libass` draws the panel into the padding.
    """
    panel = (
        f"pad=iw+{panel_width}:ih:0:0:color={PANEL_BACKGROUND},"
        f"ass={escape_filter_path(subtitles)}:fontsdir={escape_filter_path(fonts_directory)}"
    )
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-vf", panel,
        "-an",
        *ENCODE,
        "-movflags", "+faststart",
        str(output),
    ]


def render(stem: Path, *, panel_width: int = PANEL_WIDTH) -> Path:
    """Compose `<stem>-NNN.mp4` and `<stem>.decisions.jsonl` into one video."""
    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    font, family = require_monospace()
    chunks = chunk_paths(stem)
    decisions = read_decisions(stem.with_name(f"{stem.name}.decisions.jsonl"))

    probed = [probe_video(ffprobe, chunk) for chunk in chunks]
    width, height, _ = probed[0]
    durations = [duration for _, _, duration in probed]
    document = ass_document(
        decisions,
        concatenated_starts(decisions, durations),
        width=width,
        height=height,
        total_seconds=sum(durations),
        panel_width=panel_width,
        font_family=family,
    )

    output = stem.with_name(f"{stem.name}-panel.mp4")
    with tempfile.TemporaryDirectory(prefix="tower-rl-render-") as scratch:
        concat_list = Path(scratch) / "chunks.txt"
        concat_list.write_text(
            "".join(f"file '{chunk.resolve()}'\n" for chunk in chunks)
        )
        subtitles = Path(scratch) / "panel.ass"
        subtitles.write_text(document)
        subprocess.run(
            ffmpeg_command(
                ffmpeg=ffmpeg,
                concat_list=concat_list,
                subtitles=subtitles,
                fonts_directory=font.parent,
                output=output,
                panel_width=panel_width,
            ),
            check=True,
        )
    return output


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        required=True,
        help="the recording's stem: the path `spectate.py --record` was given, "
        "without the `-000.mp4`",
    )
    arguments = parser.parse_args(argv)
    # A stem given with its suffix, or with a chunk number, is the same
    # recording and is accepted as one: nobody reading a directory listing
    # types the stem.
    stem = arguments.recording
    if stem.suffix == ".mp4":
        stem = stem.with_suffix("")
        if stem.name[-4:-3] == "-" and stem.name[-3:].isdigit():
            stem = stem.with_name(stem.name[:-4])
    arguments.recording = stem
    return arguments


def main() -> int:
    arguments = parse_arguments()
    print(f"composed: {render(arguments.recording)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
