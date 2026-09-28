"""Plain text metric displays, without cursor control, safe to pipe.

The streaming display keeps headings beside their values and omits absent
metrics. The fixed-layout helpers remain available to existing callers.
"""

import dataclasses
import math

from trlx.ranges import Cell

# Fixed widths for the progress columns. Wide enough for six-digit step counts
# and two-decimal epochs; a wider value simply pushes the row right.
STEP_WIDTH = 13
EPOCH_WIDTH = 11
PERCENT_WIDTH = 4
EVAL_WIDTH = 4
CHANGE_WIDTH = 9
BASELINE_LABEL = "Baseline Δ"
# Minimum metric column: "-1234.567!" plus a space.
MIN_METRIC_WIDTH = 10

# Marks. The out-of-range mark trails the value so the number stays parseable
# from the left; the eval mark is its own column so it never shifts metrics.
OUT_OF_RANGE = "!"
EVAL_MARK = "eval"


# Display names distinguish training loss without changing the stored metric key.
def _metric_label(metric):
    return "training_loss" if metric == "loss" else metric


# Metric column width: at least the name, at least the widest plausible value.
def _metric_width(metric):
    return max(len(_metric_label(metric)), MIN_METRIC_WIDTH)


# Header row names each metric and its adjacent change column.
def header(ranges):
    parts = [
        "step".rjust(STEP_WIDTH),
        "epoch".rjust(EPOCH_WIDTH),
        "%".rjust(PERCENT_WIDTH),
        "".ljust(EVAL_WIDTH),
    ]
    for metric in ranges:
        parts.append(_metric_label(metric).rjust(_metric_width(metric)))
        parts.append("Change".rjust(CHANGE_WIDTH))
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
    # Loss history comes from observed records, including retained records on resume.
    def __init__(self, range_table, emit, width=120):
        self.ranges = range_table
        self.emit = emit
        self.width = width
        self._layout = None
        self._rows = 0
        self._widths = {}
        self._losses = {}
        self._loss_baselines = {}

    # Cache only measured losses. Resume seeds this state from retained records
    # without printing history; carried evaluation values keep their original step.
    def observe(self, record):
        if "quality" in record or "train_runtime" in record["log"]:
            return
        for name in ("loss", "eval_loss"):
            raw = record["log"].get(name)
            if raw is None:
                continue
            value = float(raw)
            # The first finite measurement stays the baseline across carried values and resume.
            if math.isfinite(value):
                self._loss_baselines.setdefault(name, value)
            previous = self._losses.get(name)
            bounds = self.ranges.get(name)
            cell = Cell(value, value - previous[0].value if previous else None,
                        bounds is not None and not bounds[0] <= value <= bounds[1])
            self._losses[name] = (cell, record["step"])

    # Any intervening terminal message invalidates the visible table heading.
    def interrupt(self):
        self._layout = None

    # Render the already-evaluated cells without replaying or modifying history.
    def record(self, record, row):
        self.observe(record)
        if "quality" in record:
            context = record["quality"]
            self.emit(f"Independent quality: {context['phase']}, step {record['step']}, {context['preset']}")
            for name, value in record["log"].items():
                if name.startswith("quality/metric_rows/"):
                    continue
                count = record["log"].get("quality/metric_rows/" + name.removeprefix("quality/"))
                suffix = f" ({count} rows)" if count is not None else ""
                self.emit(f"  {name.removeprefix('quality/')}: {value}{suffix}")
            self.interrupt()
            return
        if "train_runtime" in record["log"]:
            self.emit("Training summary:")
            for name, value in record["log"].items():
                self.emit(f"  {name}: {value}")
            self.interrupt()
            return

        if not any(record["log"].get(name) is not None for name in {*self.ranges, "loss", "eval_loss"}):
            return  # A bookkeeping-only metric event supplies no new table value.
        blank = Cell(None, None, False)
        # Training loss belongs to this log record. Evaluation loss is explicitly
        # carried forward, with eval_step showing its age; neither becomes zero when absent.
        loss = self._losses["loss"][0] if record["log"].get("loss") is not None else blank
        evaluation, eval_step = self._losses.get("eval_loss", (blank, None))
        cells = {"loss": loss, "eval_loss": evaluation,
                 **{name: cell for name, cell in row.cells.items() if name not in {"loss", "eval_loss"}}}
        row = dataclasses.replace(row, cells=cells)
        names = ["loss", "eval_loss", *[name for name in self.ranges
                  if name not in {"loss", "eval_loss"} and cells[name].value is not None]]
        progress = [f"{row.step}/{row.max_steps}",
                    f"{row.epoch:.2f}/{row.num_train_epochs:.2f}",
                    f"{int(100 * row.step / row.max_steps) if row.max_steps else 0}%",
                    str(eval_step) if eval_step is not None else ""]
        labels = ["step", "epoch", "%", "eval_step"]
        minimum = [2 * len(str(row.max_steps)) + 1, len(progress[1]), 4, len("eval_step")]
        for name, value, size in zip(labels, progress, minimum):
            self._widths[name] = max(self._widths.get(name, 0), len(name), len(value), size)
        for name in names:
            cell = row.cells[name]
            self._widths[(name, "value")] = max(
                self._widths.get((name, "value"), 0), 5, len(_metric_label(name)), len(format_value(cell)))
            self._widths[(name, "change")] = max(
                self._widths.get((name, "change"), 0), 6, len(format_change(cell)))
            if name in ("loss", "eval_loss"):
                self._widths[(name, "baseline")] = max(
                    self._widths.get((name, "baseline"), 0), len(BASELINE_LABEL), len(self._baseline_change(name, cell)))

        groups = self._groups(names, labels)
        widths = tuple(self._widths.items())
        layout = (row.eval, tuple(names), widths, self.width)
        heading = layout != self._layout or self._rows >= 20
        phase = "Evaluation" if row.eval else "Training"
        if heading:
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

    # Keep each metric beside all its changes; unusually long names/values stay intact.
    def _groups(self, names, labels):
        prefix = sum(self._widths[name] for name in labels) + 2 * (len(labels) - 1)
        groups = [[]]
        length = prefix
        for name in names:
            size = self._widths[(name, "value")] + self._widths[(name, "change")] + 4
            if name in ("loss", "eval_loss"):
                size += self._widths[(name, "baseline")] + 2
            if groups[-1] and length + size > self.width:
                groups.append([])
                length = prefix
            groups[-1].append(name)
            length += size
        return groups

    # Missing/non-finite losses cannot supply a meaningful baseline difference.
    def _baseline_change(self, name, cell):
        baseline = self._loss_baselines.get(name)
        if baseline is None or cell.value is None or not math.isfinite(cell.value):
            return ""
        return f"{cell.value - baseline:+.3f}"

    # The same width calculation formats headings and data, including wide values.
    def _columns(self, progress, names, row=None):
        parts = [value.rjust(self._widths[name])
                 for name, value in zip(("step", "epoch", "%", "eval_step"), progress)]
        for name in names:
            value = _metric_label(name) if row is None else format_value(row.cells[name])
            change = "Change" if row is None else format_change(row.cells[name])
            parts.extend([value.rjust(self._widths[(name, "value")]),
                          change.rjust(self._widths[(name, "change")])])
            if name in ("loss", "eval_loss"):
                baseline = BASELINE_LABEL if row is None else self._baseline_change(name, row.cells[name])
                parts.append(baseline.rjust(self._widths[(name, "baseline")]))
        return "  ".join(parts).rstrip()
