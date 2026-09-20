"""Plain text metric displays, without cursor control, safe to pipe.

The streaming display keeps headings beside their values and omits absent
metrics. The fixed-layout helpers remain available to existing callers.
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


class Stream:
    # Delta history belongs to ranges.evaluate; this object owns presentation only.
    def __init__(self, range_table, emit, width=120):
        self.ranges = range_table
        self.emit = emit
        self.width = width
        self._layout = None
        self._rows = 0
        self._widths = {}

    # Any intervening terminal message invalidates the visible table heading.
    def interrupt(self):
        self._layout = None

    # Render the already-evaluated cells without replaying or modifying history.
    def record(self, record, row):
        if "train_runtime" in record["log"]:
            self.emit("Training summary:")
            for name, value in record["log"].items():
                self.emit(f"  {name}: {value}")
            self.interrupt()
            return

        names = [name for name in self.ranges if row.cells[name].value is not None]
        if not names:
            return
        progress = [f"{row.step}/{row.max_steps}",
                    f"{row.epoch:.2f}/{row.num_train_epochs:.2f}",
                    f"{int(100 * row.step / row.max_steps) if row.max_steps else 0}%"]
        labels = ["step", "epoch", "%"]
        minimum = [2 * len(str(row.max_steps)) + 1, len(progress[1]), 4]
        for name, value, size in zip(labels, progress, minimum):
            self._widths[name] = max(self._widths.get(name, 0), len(name), len(value), size)
        for name in names:
            cell = row.cells[name]
            self._widths[(name, "value")] = max(
                self._widths.get((name, "value"), 0), 5, len(name), len(format_value(cell)))
            self._widths[(name, "change")] = max(
                self._widths.get((name, "change"), 0), 6, len(format_change(cell)))

        groups = self._groups(names, labels)
        widths = tuple(self._widths.items())
        layout = (row.eval, tuple(names), widths, self.width)
        heading = layout != self._layout or self._rows >= 20
        phase = "Evaluation" if row.eval else "Training"
        if heading:
            self.emit(f"{phase} — chg: difference from previous logged value; "
                      "!: outside configured range")
            self._rows = 0
        for index, group in enumerate(groups, 1):
            # Alternating wrapped layouts need their own headings on every row.
            if len(groups) > 1:
                self.emit(f"{phase} metrics ({index}/{len(groups)}):")
            if heading or len(groups) > 1:
                self.emit(self._columns(labels, group))
            self.emit(self._columns(progress, group, row))
        self._layout = layout
        self._rows += 1

    # Keep each metric beside its change; unusually long names/values stay intact.
    def _groups(self, names, labels):
        prefix = sum(self._widths[name] for name in labels) + 4
        groups = [[]]
        length = prefix
        for name in names:
            size = self._widths[(name, "value")] + self._widths[(name, "change")] + 4
            if groups[-1] and length + size > self.width:
                groups.append([])
                length = prefix
            groups[-1].append(name)
            length += size
        return groups

    # The same width calculation formats headings and data, including wide values.
    def _columns(self, progress, names, row=None):
        parts = [value.rjust(self._widths[name])
                 for name, value in zip(("step", "epoch", "%"), progress)]
        for name in names:
            value = name if row is None else format_value(row.cells[name])
            change = "chg" if row is None else format_change(row.cells[name])
            parts.extend([value.rjust(self._widths[(name, "value")]),
                          change.rjust(self._widths[(name, "change")])])
        return "  ".join(parts).rstrip()
