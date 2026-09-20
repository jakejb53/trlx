"""metrics.jsonl: the callback that writes it and the reader that parses it.

The file is the only metric source (SPEC 2.3). One JSON object per line, one
line per Trainer.log call:

    {"step": 10, "max_steps": 100, "epoch": 0.5, "num_train_epochs": 2.0,
     "eval": false, "time": 1758200000.0, "log": {"loss": 1.234, ...}}

`log` is the trainer's dict verbatim; trlx never renames or drops its keys.
The top-level fields are trlx's, taken from TrainerState. `eval` is true when
the call carried eval_-prefixed keys, which is how transformers reports an
evaluation pass. `time` is wall-clock seconds for correlating with log.txt.

The callback imports transformers lazily so `show` never pays for it.
"""

import json
import pathlib
import time

from dataset.progress import stage
from trlx import TrlxError

FILENAME = "metrics.jsonl"

# Top-level keys every record carries. The reader rejects a record missing any
# of them so a renderer never has to guess at a partial record.
RECORD_KEYS = ("step", "max_steps", "epoch", "num_train_epochs", "eval", "log")


# Observe actual trainer callbacks without altering its control flags or metric records.
def activity_callback_class():
    from transformers import TrainerCallback

    class ActivityCallback(TrainerCallback):
        # The caller owns the enclosing trainer stage and closes us on failure too.
        def __init__(self, activity):
            self.activity = activity
            self.evaluation = None

        # Resume's completed count is read from the trainer, never guessed from the path.
        def on_train_begin(self, args, state, control, **kwargs):
            self.activity.update(state.global_step, state.max_steps)

        # A finished optimizer step is measured progress, even between metric log steps.
        def on_step_end(self, args, state, control, **kwargs):
            self.activity.update(state.global_step, state.max_steps)
            if control.should_evaluate:
                self.activity.note(f"evaluation scheduled after step {state.global_step}")
            if control.should_save:
                self.activity.note(f"checkpoint save scheduled after step {state.global_step}")

        # There is no evaluation-start callback; the first completed prediction batch
        # is our first evidence of evaluation. Never infer a batch total from the train set.
        def on_prediction_step(self, args, state, control, **kwargs):
            if self.evaluation is None:
                self.evaluation = stage(self.activity, "evaluating", unit="batches", visible=True)
                self.evaluation.__enter__()
            self.evaluation.advance()

        # Evaluation end is explicit even when an empty evaluation produced no batches.
        def on_evaluate(self, args, state, control, **kwargs):
            self.close()
            self.activity.note(f"evaluation completed at step {state.global_step}")

        # on_save runs after the trainer saved; it is not evidence that saving has begun.
        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            # The callback proves saving finished; verify the actual directory rather
            # than labelling the run parent as the saved checkpoint.
            checkpoint = pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint.is_dir():
                raise TrlxError(f"checkpoint save completed but {checkpoint} is missing")
            self.activity.note(f"checkpoint saved: {checkpoint}")

        # Close an outstanding evaluation stage on normal completion or trainer failure.
        def close(self, exc_info=(None, None, None)):
            if self.evaluation is not None:
                self.evaluation.__exit__(*exc_info)
                self.evaluation = None

    return ActivityCallback


# Builds one record from a Trainer.log call. Separated from the callback so it
# can be tested with a fabricated state and without transformers.
def record(state, logs, now):
    return {
        "step": state.global_step,
        "max_steps": state.max_steps,
        # Trainer reports epoch as a float fraction; the state value is the
        # authoritative one, logs["epoch"] is a rounded copy.
        "epoch": state.epoch if state.epoch is not None else 0.0,
        "num_train_epochs": state.num_train_epochs,
        "eval": any(k.startswith("eval_") for k in logs),
        "time": now,
        "log": logs,
    }


# Returns the TrainerCallback class. A function rather than a module-level
# class so importing this module does not import transformers.
def callback_class():
    from transformers import TrainerCallback

    # Rank 0 owns this file (SPEC 2.5): train.py attaches the callback on
    # rank 0 only, so it needs no rank check of its own. Each line is
    # flushed as written so a concurrent `show` sees complete lines plus at
    # most one partial line, which the reader skips.
    class MetricsCallback(TrainerCallback):
        # Delay opening until the first record; rank zero owns this run's metric stream.
        def __init__(self, run_dir):
            self.path = pathlib.Path(run_dir) / FILENAME
            self._file = None

        # Persist and flush each trainer record so live readers see progress immediately.
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            if self._file is None:
                try:
                    self._file = open(self.path, "a", encoding="utf-8")
                except OSError as e:
                    raise TrlxError(f"{self.path}: cannot open for writing: {e.strerror or e}")
            try:
                self._file.write(json.dumps(record(state, logs, time.time())) + "\n")
                self._file.flush()
            except OSError as e:
                raise TrlxError(f"{self.path}: cannot append metrics: {e}; check available space and permissions") from e

        # Surface final flush/close failures before reporting a completed metrics stream.
        def on_train_end(self, args, state, control, **kwargs):
            if self._file is not None:
                try:
                    self._file.close()
                except OSError as e:
                    raise TrlxError(f"{self.path}: cannot finish metrics: {e}; check available space") from e
                self._file = None

    return MetricsCallback


# Reads every record in `path`. A final line that lacks a newline and does not
# parse is the job mid-write and is skipped; any other unparseable or
# incomplete line is corrupt and is an error naming the line. Returns a list.
def read(path):
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TrlxError(f"{path}: no such file")
    except (OSError, UnicodeError) as e:
        raise TrlxError(f"{path}: cannot read: {e}")

    lines = text.split("\n")
    # split leaves "" after a trailing newline; a non-empty tail is partial.
    tail = lines.pop()
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        records.append(_parse(path, number, line))
    if tail.strip():
        try:
            json.loads(tail)
        except json.JSONDecodeError:
            pass  # Only unfinished JSON can be a writer's partial last record.
        else:
            records.append(_parse(path, len(lines) + 1, tail))
    return records


# Validate complete records even at EOF; resume must not silently erase corrupt data.
def _parse(path, number, line):
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as e:
        raise TrlxError(f"{path}: line {number}: invalid JSON: {e.msg}")
    if not isinstance(rec, dict):
        raise TrlxError(f"{path}: line {number}: record is not an object")
    missing = [k for k in RECORD_KEYS if k not in rec]
    if missing:
        raise TrlxError(f"{path}: line {number}: record missing {', '.join(missing)}")
    if not isinstance(rec["log"], dict):
        raise TrlxError(f"{path}: line {number}: 'log' is not an object")
    return rec
