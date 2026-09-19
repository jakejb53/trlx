"""One line per log step (SPEC 2.4 default display).

Plain text, whitespace-aligned, no cursor control, so it is safe to pipe.
Column widths are fixed from [ranges] alone, not from the data, so a header
can be printed before the first record exists and later rows still align;
Phase 5 streams `line` per on_log under a single `header`.
"""

# Fixed widths for the progress columns. Wide enough for six-digit step counts
# and two-decimal epochs; a wider value simply pushes the row right.
STEP_WIDTH = 13
EPOCH_WIDTH = 11
PERCENT_WIDTH = 4
EVAL_WIDTH = 4
CHANGE_WIDTH = 9
# Minimum metric column: "-1234.567!" plus a space.
MIN_METRIC_WIDTH = 10

# Marks. The out-of-range mark trails the value so the number stays parseable
# from the left; the eval mark is its own column so it never shifts metrics.
OUT_OF_RANGE = "!"
EVAL_MARK = "eval"


# Metric column width: at least the name, at least the widest plausible value.
def _metric_width(metric):
    return max(len(metric), MIN_METRIC_WIDTH)


# Header row naming every column. The change column is headed with a delta
# sign so the pairing with its metric is visible.
def header(ranges):
    parts = [
        "step".rjust(STEP_WIDTH),
        "epoch".rjust(EPOCH_WIDTH),
        "%".rjust(PERCENT_WIDTH),
        "".ljust(EVAL_WIDTH),
    ]
    for metric in ranges:
        parts.append(metric.rjust(_metric_width(metric)))
        parts.append("chg".rjust(CHANGE_WIDTH))
    return "  ".join(parts).rstrip()


# Renders one ranges.Row. Missing metrics are blank, never zero.
def line(row):
    percent = int(100 * row.step / row.max_steps) if row.max_steps else 0
    parts = [
        f"{row.step}/{row.max_steps}".rjust(STEP_WIDTH),
        f"{row.epoch:.2f}/{row.num_train_epochs:.2f}".rjust(EPOCH_WIDTH),
        f"{percent}%".rjust(PERCENT_WIDTH),
        (EVAL_MARK if row.eval else "").ljust(EVAL_WIDTH),
    ]
    for metric, cell in row.cells.items():
        parts.append(format_value(cell).rjust(_metric_width(metric)))
        parts.append(format_change(cell).rjust(CHANGE_WIDTH))
    return "  ".join(parts).rstrip()


# Three decimals, out-of-range mark appended. Shared with the TUI so both
# displays spell a value identically.
def format_value(cell):
    if cell.value is None:
        return ""
    text = f"{cell.value:.3f}"
    return text + OUT_OF_RANGE if cell.out_of_range else text


# Signed three-decimal difference; blank for the metric's first value.
def format_change(cell):
    if cell.change is None:
        return ""
    return f"{cell.change:+.3f}"


# Whole display for a finished set of rows: header then every line.
def render(rows, ranges):
    return "\n".join([header(ranges), *(line(r) for r in rows)])
