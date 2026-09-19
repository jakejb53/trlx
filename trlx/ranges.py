"""[ranges]: parsing, and the per-row cells both renderers display.

Stdlib only. `trlx show` reads [ranges] from a run's config.toml snapshot
through this module with tomllib alone, so it must not import config.py
(which imports torch and peft). config.py imports `parse` from here instead;
this is the one place the [ranges] shape is validated.
"""

import dataclasses

from trlx import TrlxError


# Validates a [ranges] table into {metric: (low, high)} in file order. The key
# order is the display column order, so a dict (insertion-ordered) is the
# right container.
def parse(path, table):
    if not table:
        raise TrlxError(f"{path}: [ranges] must name at least one metric")
    ranges = {}
    for metric, bounds in table.items():
        ok = isinstance(bounds, list) and len(bounds) == 2 and all(_is_number(b) for b in bounds)
        if not ok:
            raise TrlxError(f"{path}: [ranges].{metric} must be [low, high], got {bounds!r}")
        low, high = bounds
        if not low < high:
            raise TrlxError(f"{path}: [ranges].{metric} needs low < high, got [{low}, {high}]")
        ranges[metric] = (float(low), float(high))
    return ranges


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# One metric's display cell for one record. `value` is None when the record
# does not carry the metric (an eval record has no `loss`, a train record no
# `eval_loss`). `change` is the difference from the previous record that
# carried the metric, None for the first occurrence. `out_of_range` is judged
# against [ranges] only when a value is present.
@dataclasses.dataclass(frozen=True)
class Cell:
    value: float | None
    change: float | None
    out_of_range: bool


# One displayed row: the record's progress fields plus a Cell per [ranges]
# metric, in [ranges] order.
@dataclasses.dataclass(frozen=True)
class Row:
    step: int
    max_steps: int
    epoch: float
    num_train_epochs: float
    eval: bool
    cells: dict


# Turns metric records (metrics.read) into Rows. The change column compares
# against the previous *logged* value of the metric across rows, not the
# previous row, so a train metric's change is never computed against an eval
# row that lacks it. Values are read from the record's `log` dict, which is
# the trainer's own output; a metric absent from [ranges] is not displayed.
def evaluate(records, ranges):
    previous = {}
    rows = []
    for rec in records:
        cells = {}
        for metric, (low, high) in ranges.items():
            raw = rec["log"].get(metric)
            if raw is None:
                cells[metric] = Cell(None, None, False)
                continue
            value = float(raw)
            change = value - previous[metric] if metric in previous else None
            previous[metric] = value
            cells[metric] = Cell(value, change, not low <= value <= high)
        rows.append(Row(rec["step"], rec["max_steps"], rec["epoch"], rec["num_train_epochs"], rec["eval"], cells))
    return rows
