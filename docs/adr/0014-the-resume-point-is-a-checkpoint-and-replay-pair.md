# ADR 0014 — The resume point is a checkpoint and replay pair

**Status:** adopted, 2026-09-30. Board `#97` (under `#58`). Measured on the
workstation (`docs/experiments.md`, "Crash-safe resume point"); not yet used
by a device run.

## Context

`M3-P016` (DreamerV3, 1,000,000 decisions) was killed by systemd-oomd at
425,078 decisions. Its checkpoints survived, but its replay did not: the
buffer was saved only as a run ended, by Python cleanup, which a SIGKILL
never reaches. A resume could only re-warm, which for a run that far in is
a different experiment from the one that was stopped.

Two more gaps sat beside it. `latest.pt` and its checksum sidecar were two
renames, so a kill between them left a checkpoint its sidecar refused. And a
checkpoint recorded no random-stream state, nor did the replay dump record
its sampler's.

## Decision

**Every `latest.pt` is written with a replay dump of its own, at the same
decision count, and names it; the pair is committed so that a kill at any
moment leaves one complete pair.**

- **Layout.** Each save is `<run_dir>/replay/d<decisions, 7 digits>/`
  (`-k` appended for a second save at one count). `latest.pt` records it as
  `paired_replay` (checkpoint format 5). Only one dump is kept: no numbered
  checkpoint has a replay of its own.
- **Commit order.** Capture the buffer under its lock (a tuple of the
  immutable sequences and their priorities, and the sampler state: about
  1 ms); write the dump to `<name>.partial`, fsync every file, rename it into
  place, fsync the directory; write `latest.pt` atomically; then delete every
  other entry in `replay/`. Until `latest.pt`'s rename the old pair is whole;
  after it, the new one is.
- **Checksum sidecar.** Replaced first by one naming both the file in place
  and the new one, then the file renamed, then the sidecar by one naming the
  new file alone. `load` accepts any line, so the file on disk always passes.
- **Cadence.** The pair is written every `--checkpoint-every-episodes`
  (default 25), beside every numbered checkpoint, and once more as the run
  ends. The write holds the progress lock, so learning and episode counting
  wait for it; actors keep playing and adding to the buffer.
- **Full rewrite, not an incremental log.** A full dump at 1,000,000 steps
  takes 11.5 s and adds 0.2 GB of peak memory against a 23 GB process,
  written in 8,192-row chunks with the page cache dropped behind them. That
  is inside the 30 s bound. An append log would save most of those 11.5 s
  but would need eviction records, compaction, and a second format to keep
  consistent with the checkpoint, for no requirement the rewrite misses.
- **Random streams.** The checkpoint carries Python's, NumPy's, torch's CPU
  and (same device count only) CUDA streams; the replay dump (format 2)
  carries its sampler. `build_arm` restores the process streams last, after
  every component is built.
- **Failure.** A replay save that fails moves nothing: it is printed, and
  the previous pair stands. A `latest.pt` without its replay would be a
  resume point that silently loses the buffer.

### What a resume refuses and accepts

- A `latest.pt` naming a dump: that dump must be in `replay/` and record the
  same decision count, or the resume is refused by name ("mismatched pair",
  "resume point is broken"). `replay/` absent altogether is an operator's
  choice to re-warm, as before.
- A numbered checkpoint, or a `latest.pt` from before format 5: reloads a
  complete dump at exactly its decision count, is refused while only dumps
  at other counts are there, and re-warms with none. This is the old guard.
- An old run's single dump saved as `replay/` itself (`M3-P015`) is still
  read, and replaced by the first save of the resumed segment.
  `replay.previous` from that era is still refused.

### What is not restored

- **In-flight episodes.** Actors add an episode only when it ends; the
  episodes being played at the moment of the kill are lost, and the
  emulators restart them from the game's normal new-run path.
- **Uncounted episodes.** An actor that finishes an episode while the pair is
  being written adds it to the buffer before it is counted, so the capture
  can hold up to one such episode per other actor. Closing that would change
  when actors add relative to learning, which is learning behaviour.
- **Per-component streams.** The backbone's acting stream is not saved;
  acting copies are reseeded per actor as the fleet is built.
- **Bit-exact reproducibility is not claimed.** Actor thread scheduling and
  device timing are not replayed; restoring the streams removes one source of
  difference between a resumed run and one that never stopped, not all.

## Consequences

- Disk: one dump (4.8 GB at 1,000,000 steps) plus a second one transiently
  while the next is written.
- Actor-visible stall: learning and counting pause about 12 s per save at
  1,000,000 steps; at the default cadence that is one save per 25 episodes.
- Resuming `M3-P016` itself, which predates this, can only re-warm: its
  `latest.pt` is format 4 and it left no dump.
