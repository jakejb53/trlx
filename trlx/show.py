"""`trlx show <run> [--tui]`, and the run-directory reader behind both displays.

A run directory (SPEC 2.3) is the unit of truth: metrics.jsonl for rows,
config.toml for [ranges] and the run name, checkpoint-N/ directories,
log.txt, preflight.json, verify.json. `load` reads all of them into a
RunState. The TUI calls `load` on every poll, so a run still being written
displays through exactly this code; Phase 5 needs no second path.
"""

import contextlib
import dataclasses
import json
import pathlib
import re
import tomllib

from dataset.progress import stage
from trlx import TrlxError, metrics, ranges, render_lines

CONFIG_FILENAME = "config.toml"
LOG_FILENAME = "log.txt"
PREFLIGHT_FILENAME = "preflight.json"
VERIFY_FILENAME = "verify.json"
CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)$")

# Bytes read from the end of log.txt for the tail pane. A bound so a
# multi-gigabyte log is not re-read every poll.
_LOG_TAIL_BYTES = 65536


# A checkpoint directory and the eval loss logged at its step, None when no
# eval record exists for that step (eval disabled, or intervals differ).
@dataclasses.dataclass(frozen=True)
class Checkpoint:
    step: int
    eval_loss: float | None
    best: bool


# Everything the displays show, read from the run directory at one instant.
@dataclasses.dataclass(frozen=True)
class RunState:
    name: str
    ranges: dict
    rows: list
    checkpoints: list
    log_tail: list
    # Parsed preflight.json and verify.json, None when the file is absent.
    preflight: dict | None
    verify: dict | None
    # Derived from files, not reported by the job: "starting" (no records),
    # "training", "trained" (last step reached max_steps), "verified"
    # (verify.json present).
    phase: str


# Reads [ranges] and run_name from the config.toml snapshot with tomllib alone.
# config.load is not used: it needs the method name and imports torch.
def load_config(run_dir):
    path = pathlib.Path(run_dir) / CONFIG_FILENAME
    try:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    except FileNotFoundError:
        raise TrlxError(f"{path}: no such file; not a run directory")
    except (OSError, UnicodeError) as e:
        raise TrlxError(f"{path}: cannot read: {e}")
    except tomllib.TOMLDecodeError as e:
        raise TrlxError(f"{path}: invalid TOML: {e}")
    if "ranges" not in doc or not isinstance(doc["ranges"], dict):
        raise TrlxError(f"{path}: missing required block [ranges]")
    # run_name is written into the snapshot by config.py's default rule, so
    # its absence means a snapshot trlx did not write.
    name = doc.get("run_name")
    if not isinstance(name, str):
        raise TrlxError(f"{path}: missing 'run_name'")
    return name, ranges.parse(path, doc["ranges"])


# Reads the whole run directory. `log_lines` bounds the tail kept. A missing
# metrics.jsonl yields no rows rather than an error because a live TUI starts
# before the first log step; `show` in line mode checks the file itself.
def load(run_dir, name, range_table, log_lines):
    run_dir = pathlib.Path(run_dir)
    metrics_path = run_dir / metrics.FILENAME
    records = metrics.read(metrics_path) if metrics_path.exists() else []
    rows = ranges.evaluate(records, range_table)
    preflight = _read_json(run_dir / PREFLIGHT_FILENAME)
    verify = _read_json(run_dir / VERIFY_FILENAME)
    return RunState(
        name=name,
        ranges=range_table,
        rows=rows,
        checkpoints=_checkpoints(run_dir, records),
        log_tail=_log_tail(run_dir / LOG_FILENAME, log_lines),
        preflight=preflight,
        verify=verify,
        phase=_phase(rows, verify is not None),
    )


def _phase(rows, verified):
    if verified:
        return "verified"
    if not rows:
        return "starting"
    last = rows[-1]
    return "trained" if last.max_steps and last.step >= last.max_steps else "training"


# checkpoint-N directories in step order, each paired with the eval_loss from
# the eval record at step N. Best is the lowest eval_loss; with none logged,
# nothing is marked.
def _checkpoints(run_dir, records):
    eval_loss = {}
    for rec in records:
        if rec["eval"] and rec["log"].get("eval_loss") is not None:
            eval_loss[rec["step"]] = float(rec["log"]["eval_loss"])
    steps = []
    try:
        entries = list(run_dir.iterdir())
    except OSError as e:
        raise TrlxError(f"{run_dir}: cannot list checkpoints: {e}; select a readable run directory") from e
    for entry in entries:
        m = CHECKPOINT_DIR.match(entry.name)
        if m and entry.is_dir():
            steps.append(int(m.group(1)))
    steps.sort()
    losses = [eval_loss.get(s) for s in steps]
    known = [l for l in losses if l is not None]
    best = min(known) if known else None
    return [Checkpoint(s, l, l is not None and l == best) for s, l in zip(steps, losses)]


# Last `count` lines of the log. Reads only the file's tail; the first line
# of the chunk may be cut mid-line and is dropped when the chunk is partial.
def _log_tail(path, count):
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            partial = size > _LOG_TAIL_BYTES
            if partial:
                f.seek(size - _LOG_TAIL_BYTES)
            chunk = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except OSError as e:
        raise TrlxError(f"{path}: cannot read: {e.strerror or e}")
    lines = chunk.splitlines()
    if partial and lines:
        lines = lines[1:]
    return lines[-count:] if count > 0 else []


# preflight.json / verify.json. Absent is None; present but unreadable is an
# error, since a half-written result must not display as "not run".
def _read_json(path):
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as e:
        raise TrlxError(f"{path}: cannot read: {e}")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise TrlxError(f"{path}: invalid JSON: {e.msg}")
    if not isinstance(doc, dict):
        raise TrlxError(f"{path}: expected a JSON object")
    return doc


# Line mode: every row, once. Fails on a missing metrics.jsonl because a
# finished run without it has nothing to show and saying so is the answer.
def show_lines(run_dir, *, progress=None):
    with stage(progress, f"reading run artifacts from {run_dir}") as activity:
        _, range_table = load_config(run_dir)
        records = metrics.read(pathlib.Path(run_dir) / metrics.FILENAME)
        activity.note(f"read {len(records)} metric records")
    with stage(progress, "rendering saved metrics"):
        print(render_lines.render(ranges.evaluate(records, range_table), range_table))


# TUI mode: polls `load` until quit.
def show_tui(run_dir, *, progress=None):
    from trlx import render_tui

    with stage(progress, f"opening run display for {run_dir}"):
        name, range_table = load_config(run_dir)
    # The existing TUI owns the terminal and continuously displays artifact updates.
    with progress.suspended() if progress is not None else contextlib.nullcontext():
        render_tui.run(lambda log_lines: load(run_dir, name, range_table, log_lines))
