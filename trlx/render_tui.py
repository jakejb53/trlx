"""Full-screen display (SPEC 2.4 --tui): fixed panes, no scrolling, no toggles.

Top to bottom: status bar; metrics table taking the remaining height;
checkpoints; log tail; preflight and verify results. Every pane except the
metrics table has a fixed height, so the metrics pane absorbs the terminal
size. When the terminal cannot hold the layout, a message says what size is
needed instead of silently dropping columns or panes.

The display owns no data: it calls `load(log_lines)` once per poll and draws
the RunState it gets back. That is what makes `show --tui` on a finished run
and the live view during training the same code.
"""

import curses
import json

from trlx import render_lines

# Display constant: how often the run directory is re-read. Not operational
# configuration; the same class of exception as _BACKOFF_BASE_SECONDS in
# dataset/endpoint.py.
POLL_MS = 1000

# Fixed pane heights, excluding each pane's title line.
CHECKPOINT_ROWS = 3
LOG_ROWS = 4
RESULT_ROWS = 2
# status bar + metrics title + metrics header + three titled panes.
_FIXED_ROWS = 1 + 2 + (1 + CHECKPOINT_ROWS) + (1 + LOG_ROWS) + (1 + RESULT_ROWS)
MIN_HEIGHT = _FIXED_ROWS + 1

# Metrics table geometry. The step cell carries the eval mark as a suffix so
# the table needs no separate eval column at 80 columns.
STEP_WIDTH = 6
VALUE_WIDTH = 9
CHANGE_WIDTH = 7
_PAIR_MIN = VALUE_WIDTH + 1 + CHANGE_WIDTH
EVAL_SUFFIX = "e"
BEST_MARK = "*"


# Runs the display until the user quits. `load(log_lines)` returns a
# show.RunState; an exception from it ends the display with the terminal
# restored, and the caller reports it.
def run(load):
    curses.wrapper(_loop, load)


def _loop(stdscr, load):
    try:
        curses.curs_set(0)
    except curses.error:
        pass  # terminal without cursor control; nothing to hide
    stdscr.timeout(POLL_MS)
    while True:
        state = load(LOG_ROWS)
        stdscr.erase()
        _draw(stdscr, state)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return
        # KEY_RESIZE and timeout both fall through to a redraw.


# Column width of one metric's value+change pair: the name must fit as the
# header, and the pair must fit as the cells.
def _pair_width(metric):
    return max(len(metric), _PAIR_MIN)


# Total metrics table width for a [ranges] dict.
def table_width(ranges):
    return STEP_WIDTH + sum(1 + _pair_width(m) for m in ranges)


def _draw(win, state):
    height, width = win.getmaxyx()
    need = table_width(state.ranges)
    if height < MIN_HEIGHT or width < need:
        _put(win, 0, 0, f"terminal too small: need {need}x{MIN_HEIGHT}, have {width}x{height}. q quits.")
        return

    _status_bar(win, state, width)
    y = 1
    metric_rows = height - _FIXED_ROWS
    y = _metrics_pane(win, y, width, state, metric_rows)
    y = _checkpoints_pane(win, y, width, state)
    y = _log_pane(win, y, width, state)
    _results_pane(win, y, width, state)


# Percent complete, steps, epochs, phase; reverse video across the full width.
def _status_bar(win, state, width):
    if state.rows:
        last = state.rows[-1]
        percent = int(100 * last.step / last.max_steps) if last.max_steps else 0
        progress = (f"{percent}%  step {last.step}/{last.max_steps}  "
                    f"epoch {last.epoch:.2f}/{last.num_train_epochs:.2f}")
    else:
        progress = "no metrics yet"
    left = f" {state.name}  {progress}  {state.phase}"
    right = "q quits "
    gap = max(1, width - len(left) - len(right))
    _put(win, 0, 0, (left + " " * gap + right)[:width], curses.A_REVERSE)


# Most recent rows that fit, oldest first, under a header naming each metric
# over its value+change pair.
def _metrics_pane(win, y, width, state, rows_fit):
    _put(win, y, 0, "metrics", curses.A_BOLD)
    header = "step".rjust(STEP_WIDTH)
    for metric in state.ranges:
        header += " " + metric.rjust(_pair_width(metric))
    _put(win, y + 1, 0, header, curses.A_UNDERLINE)
    for i, row in enumerate(state.rows[-rows_fit:]):
        step = f"{row.step}{EVAL_SUFFIX if row.eval else ''}".rjust(STEP_WIDTH)
        line = step
        for metric, cell in row.cells.items():
            pair = f"{render_lines.format_value(cell):>{VALUE_WIDTH}} {render_lines.format_change(cell):>{CHANGE_WIDTH}}"
            line += " " + pair.rjust(_pair_width(metric))
        _put(win, y + 2 + i, 0, line)
    return y + 2 + rows_fit


# Most recent checkpoints that fit. Best (lowest eval loss) carries the mark.
def _checkpoints_pane(win, y, width, state):
    _put(win, y, 0, "checkpoints", curses.A_BOLD)
    if not state.checkpoints:
        _put(win, y + 1, 0, "none")
    for i, ckpt in enumerate(state.checkpoints[-CHECKPOINT_ROWS:]):
        loss = f"eval_loss {ckpt.eval_loss:.3f}" if ckpt.eval_loss is not None else "eval_loss -"
        mark = f"  {BEST_MARK} best" if ckpt.best else ""
        _put(win, y + 1 + i, 0, f"checkpoint-{ckpt.step}  {loss}{mark}"[:width])
    return y + 1 + CHECKPOINT_ROWS


def _log_pane(win, y, width, state):
    _put(win, y, 0, "log", curses.A_BOLD)
    for i, line in enumerate(state.log_tail[-LOG_ROWS:]):
        _put(win, y + 1 + i, 0, line[:width])
    return y + 1 + LOG_ROWS


# One line each for preflight.json and verify.json: the object as compact
# JSON, cut at the terminal edge. The file is the full record.
def _results_pane(win, y, width, state):
    _put(win, y, 0, "preflight / verify", curses.A_BOLD)
    for i, (label, doc) in enumerate((("preflight", state.preflight), ("verify", state.verify))):
        text = json.dumps(doc, separators=(",", ":")) if doc is not None else "not run"
        _put(win, y + 1 + i, 0, f"{label}: {text}"[:width])


# addstr clipped to the window. curses raises when a write touches the
# bottom-right cell even with room for the text; that is not an error here.
def _put(win, y, x, text, attr=0):
    height, width = win.getmaxyx()
    if y >= height or x >= width:
        return
    try:
        win.addstr(y, x, text[: width - x], attr)
    except curses.error:
        pass
