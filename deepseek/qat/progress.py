"""Live output of a training run: one table row per step, one line per validation.

The run is watched from these lines, so they carry what a decision needs and
nothing else: the step's loss and the running mean over the current epoch, the
latest held-out loss with its change from the previous validation and from the
baseline, the fraction of the model's dense and expert values changed since the
start (over every value of the class, trained or frozen, so runs with different
trainable sets compare; the flip indicator, measured every step by
``deepseek.qat.flips``), how many layers
have any expert change, the sampled master drift per class (``drift-d``,
``drift-e``: mean |master - baseline|, to be read against the FP8 median
half-cell of 9.8e-4 and the FP4 median half-cell of 3.9e-3) with its sign
consistency (``cons-d``, ``cons-e``: 1.0 when every update pushed the same way,
near 0 for a random walk), the per-class gradient norms, and the step's wall time.
Headings repeat every 20 rows. Rates, memory, and clip coefficients stay in the
progress file and the report. A step row's changed fractions describe the masters
as that step began (its forward is what was measured), so they lag the update by
one step; the ``eval`` line and the final report describe the masters after the
last update.

``Progress`` prints on rank 0 unconditionally and, when a report path is given,
appends every record as one JSON line to ``<out stem>-progress.jsonl``, flushed
per record, so a killed run keeps every measurement taken before the kill.
"""

import json
import sys

HEADER_EVERY = 20

# (heading, width). Every cell is right-aligned to its width.
COLUMNS = [
    ("step", 9), ("epoch", 6), ("loss", 8), ("ep-mean", 8), ("Δ", 8),
    ("held-out", 9), ("Δprev", 8), ("Δbase", 8),
    ("dense%", 8), ("expert%", 9), ("layers", 7),
    ("drift-d", 8), ("cons-d", 7), ("drift-e", 8), ("cons-e", 7),
    ("gn-dense", 9), ("gn-expert", 10), ("s/step", 7),
]


class Progress:
    def __init__(self, path: str | None, rank: int):
        self.rank = rank
        self.path = path if rank == 0 else None
        if self.path is not None:
            with open(self.path, "w", encoding="utf-8"):
                pass  # a fresh run starts a fresh file

    def record(self, kind: str, data: dict) -> None:
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, **data}) + "\n")

    def say(self, line: str) -> None:
        if self.rank == 0:
            print(line, file=sys.stderr, flush=True)


def _cell(value, width: int, spec: str = "", signed: bool = False) -> str:
    if value is None:
        return " " * width
    if isinstance(value, str):
        return value.rjust(width)
    text = format(value, ("+" if signed else "") + spec)
    return text.rjust(width)


def _pct(changed: int | None, elements: int | None, decimals: int) -> float | None:
    if changed is None or not elements:
        return None
    return round(100.0 * changed / elements, decimals)


class StepTable:
    """Formats step rows and validation lines, keeping the running state they need."""

    def __init__(self, total_steps: int | None, steps_per_epoch: int | None, expert_layers: int):
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.expert_layers = expert_layers
        self.rows_since_header = HEADER_EVERY  # the first row is preceded by a header
        self._epoch_index: int | None = None
        self._epoch_losses: list[float] = []
        self._previous_mean: float | None = None
        self.baseline: float | None = None
        self.held_out: float | None = None
        self.previous_held_out: float | None = None

    def header(self) -> str:
        return "  ".join(name.rjust(width) for name, width in COLUMNS)

    def note_validation(self, loss: float, baseline: bool = False) -> None:
        if baseline:
            self.baseline = loss
            return
        self.previous_held_out = self.held_out
        self.held_out = loss

    def validation_line(self, step: int) -> str:
        parts = [f"eval step {step}: held-out {self.held_out:.4f}"]
        deltas = []
        if self.previous_held_out is not None:
            deltas.append(f"Δprev {self.held_out - self.previous_held_out:+.4f}")
        if self.baseline is not None:
            deltas.append(f"Δbase {self.held_out - self.baseline:+.4f}")
        if deltas:
            parts.append("(" + ", ".join(deltas) + ")")
        return " ".join(parts)

    # One row for a finished step. `flips` is FlipCounter.totals() of the pass
    # that started this step (the state the step's forward saw), or None;
    # `norms` is the stepper's per-class norms, or None.
    def row(self, step: int, loss: float | None, flips: dict | None, norms: dict | None, seconds: float | None,
            drift: dict | None = None) -> str:
        lines = []
        if self.rows_since_header >= HEADER_EVERY:
            lines.append(self.header())
            self.rows_since_header = 0
        epoch_index = step // self.steps_per_epoch if self.steps_per_epoch else 0
        if epoch_index != self._epoch_index:
            self._epoch_index = epoch_index
            self._epoch_losses = []
        if loss is not None:
            self._epoch_losses.append(loss)
        mean = sum(self._epoch_losses) / len(self._epoch_losses) if self._epoch_losses else None
        delta = None if mean is None or self._previous_mean is None else mean - self._previous_mean
        self._previous_mean = mean if mean is not None else self._previous_mean
        epoch = (step + 1) / self.steps_per_epoch if self.steps_per_epoch else None
        step_text = f"{step}/{self.total_steps}" if self.total_steps else str(step)
        layers = None
        if flips is not None and self.expert_layers:
            changed_layers = sum(1 for k, v in flips.items() if k.startswith("expert_values_changed_layer") and v)
            layers = f"{changed_layers}/{self.expert_layers}"
        cells = [
            _cell(step_text, 9), _cell(epoch, 6, ".2f"), _cell(loss, 8, ".4f"), _cell(mean, 8, ".4f"), _cell(delta, 8, ".4f", signed=True),
            _cell(self.held_out, 9, ".4f"),
            _cell(None if self.held_out is None or self.previous_held_out is None else self.held_out - self.previous_held_out, 8, ".4f", signed=True),
            _cell(None if self.held_out is None or self.baseline is None else self.held_out - self.baseline, 8, ".4f", signed=True),
            _cell(None if flips is None else _pct(flips.get("dense_values_changed"), flips.get("dense_class_elements"), 4), 8, ".4f"),
            _cell(None if flips is None else _pct(flips.get("expert_values_changed"), flips.get("expert_class_elements"), 6), 9, ".6f"),
            _cell(layers, 7),
            _cell(None if drift is None or "dense" not in drift else drift["dense"][0], 8, ".1e"),
            _cell(None if drift is None or "dense" not in drift else drift["dense"][1], 7, ".2f"),
            _cell(None if drift is None or "expert" not in drift else drift["expert"][0], 8, ".1e"),
            _cell(None if drift is None or "expert" not in drift else drift["expert"][1], 7, ".2f"),
            _cell(None if norms is None else norms.get("dense"), 9, ".3f"),
            _cell(None if norms is None else norms.get("expert"), 10, ".3f"),
            _cell(seconds, 7, ".1f"),
        ]
        lines.append("  ".join(cells))
        self.rows_since_header += 1
        return "\n".join(lines)
