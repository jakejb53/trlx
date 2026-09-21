"""Run allocation and checkpoint rewind, owned by the training supervisor."""

import contextlib
import datetime
import fcntl
import json
import os
import pathlib
import re
import shutil
import tempfile
from dataclasses import dataclass

from dataset.io import DatasetError, write_text
from trlx import TrlxError, metrics, show


# Directory locks need no persistent lock file and disappear when the owner exits.
# Allocation waits for other allocators; a live run cannot be resumed twice.
@contextlib.contextmanager
def locked(directory, wait=False):
    descriptor = None
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        operation = fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB)
        fcntl.flock(descriptor, operation)
    except BlockingIOError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise TrlxError(f"{directory}: another training process owns this run; resume after it exits") from error
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise TrlxError(f"{directory}: cannot lock run directory: {error}") from error
    try:
        yield
    finally:
        os.close(descriptor)


# Names are labels only: the snapshot retains the exact model and dataset inputs.
def _slug(value):
    name = re.sub(r"[^\w.-]+", "-", value.lower()).strip("-._")
    if not name:
        raise TrlxError(f"{value!r}: cannot derive a run-directory name; supply a named model or dataset")
    return name


# Reserve the next number across the whole parent/date, including different models.
# Scan and mkdir share one lock: mkdir alone cannot protect a number in different names.
def allocate(parent, model, dataset):
    parent = pathlib.Path(parent).resolve()
    model_path = pathlib.Path(model)
    # Normalize relative directory notation such as '.' without losing a symlink's label.
    model_name = pathlib.Path(os.path.abspath(model)).name if model_path.is_dir() else model
    data_name = pathlib.Path(dataset.source).stem if dataset.is_file else dataset.source
    suffix = f"{_slug(model_name)}--{_slug(data_name)}"
    try:
        parent.mkdir(parents=True, exist_ok=True)
        with locked(parent, wait=True):
            date = datetime.date.today().strftime("%Y%m%d")
            pattern = re.compile(rf"^{date}-(\d+)--")
            numbers = [int(match[1]) for entry in parent.iterdir() if (match := pattern.match(entry.name))]
            directory = parent / f"{date}-{max(numbers, default=0) + 1}--{suffix}"
            directory.mkdir()
            return directory
    except OSError as error:
        raise TrlxError(f"{parent}: cannot create a run directory: {error}; choose a writable --output-dir") from error


# The checkpoint's metadata, not the digits in its name, selects retained history.
@dataclass(frozen=True)
class Resume:
    checkpoint: pathlib.Path
    step: int

    # A checkpoint and its run snapshot must live in the same run directory.
    @property
    def directory(self):
        return self.checkpoint.parent


# Read checkpoint metadata without loading executable pickle files or model tensors.
def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrlxError(f"{path}: cannot read checkpoint metadata: {error}; select a complete checkpoint") from error
    if not isinstance(value, dict):
        raise TrlxError(f"{path}: checkpoint metadata must be a JSON object; select a complete checkpoint")
    return value


# Check both ordinary and indexed weights before a rewind can discard later work.
# Tensor compatibility is still validated by the trainer when it loads the checkpoint.
def _check_weights(checkpoint):
    for filename in ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin"):
        path = checkpoint / filename
        if path.is_file() and path.stat().st_size:
            metadata = "adapter_config.json" if filename.startswith("adapter_") else "config.json"
            _read_json(checkpoint / metadata)
            return
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = checkpoint / filename
        if not index.is_file():
            continue
        mapping = _read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping or not all(isinstance(v, str) and v for v in mapping.values()):
            raise TrlxError(f"{index}: missing or invalid weight_map; select a complete checkpoint")
        for shard in set(mapping.values()):
            # Index entries name files inside this checkpoint, not another run's artifacts.
            if pathlib.Path(shard).is_absolute() or ".." in pathlib.Path(shard).parts:
                raise TrlxError(f"{index}: weight shard {shard!r} must be a path within the checkpoint")
            path = checkpoint / shard
            if not path.is_file() or not path.stat().st_size:
                raise TrlxError(f"{index}: weight shard {shard!r} is missing or empty; select a complete checkpoint")
        _read_json(checkpoint / "config.json")
        return
    raise TrlxError(f"{checkpoint}: no saved model or adapter weights; select a complete checkpoint")


# Validation is read-only, shared by standalone check and the supervisor before cleanup.
def inspect_checkpoint(checkpoint):
    checkpoint = pathlib.Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise TrlxError(f"{checkpoint}: checkpoint directory does not exist; supply --resume-from-checkpoint CHECKPOINT")
    state = _read_json(checkpoint / "trainer_state.json")
    step = state.get("global_step")
    if type(step) is not int or step < 0:
        raise TrlxError(f"{checkpoint / 'trainer_state.json'}: global_step must be a nonnegative integer")
    named_step = show.CHECKPOINT_DIR.fullmatch(checkpoint.name)
    if named_step and int(named_step[1]) != step:
        raise TrlxError(f"{checkpoint}: directory step {named_step[1]} disagrees with saved global_step {step}")
    try:
        _check_weights(checkpoint)
    except OSError as error:
        raise TrlxError(f"{checkpoint}: cannot inspect saved weights: {error}; select a readable checkpoint") from error
    return Resume(checkpoint, step)


# Run-owned metadata stages by default; direct writes explicitly accept partial history.
def write_atomic(path, text, *, no_staging=False):
    path = pathlib.Path(path)
    if no_staging:
        try:
            write_text(path, text, force=True, no_staging=True)
        except DatasetError as e:
            raise TrlxError(str(e)) from e
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(text)
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# Validate optional quality rounds before any rewind mutation, preserving only retained steps.
def _quality_at_step(directory, step):
    path = directory / show.QUALITY_FILENAME
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise TrlxError(f"{path}: cannot read independent quality evidence before resume: {error}") from error
    retained = []
    for number, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
            value = item["quality"]["step"]
            if type(value) is not int or value < 0:
                raise ValueError("quality.step must be a nonnegative integer")
        except (ValueError, KeyError, TypeError) as error:
            raise TrlxError(f"{path}: line {number}: invalid quality evidence; repair it before resuming: {error}") from error
        if value <= step:
            retained.append(line + "\n")
    return "".join(retained)


# Rewind all step-indexed evidence together; the selected checkpoint remains untouched.
# A checkpoint selection authorizes discarding its abandoned continuation, not its logs.
# Validate evidence before mutation. Multi-file cleanup is not atomic; failures name the affected path.
def rewind(resume, *, no_staging=False):
    directory = resume.directory
    metric_path = directory / metrics.FILENAME
    records = metrics.read(metric_path) if metric_path.exists() else []
    quality_text = _quality_at_step(directory, resume.step)
    retained = []
    for number, record in enumerate(records, 1):
        step = record["step"]
        if type(step) is not int or step < 0:
            raise TrlxError(f"{metric_path}: record {number}: step must be a nonnegative integer; repair metrics before resuming")
        if step <= resume.step:
            retained.append(record)
    # Serialize before deletion so an invalid record cannot interrupt publication.
    text = "".join(json.dumps(record) + "\n" for record in retained)
    removed = []
    path = directory
    try:
        later = [entry for entry in directory.iterdir()
                 if (match := show.CHECKPOINT_DIR.fullmatch(entry.name)) and int(match[1]) > resume.step]
        for path in sorted(later):
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)
            removed.append(path.name)
        for filename in (show.PREFLIGHT_FILENAME, show.VERIFY_FILENAME, show.ASSESSMENT_FILENAME):
            path = directory / filename
            path.unlink(missing_ok=True)
        path = metric_path
        if metric_path.exists():
            write_atomic(metric_path, text, no_staging=no_staging)
        if quality_text is not None:
            path = directory / show.QUALITY_FILENAME
            write_atomic(path, quality_text, no_staging=no_staging)
    except OSError as error:
        raise TrlxError(
            f"{path}: resume cleanup failed: {error}; cleanup may be partial; "
            f"selected checkpoint {resume.checkpoint} is preserved. Retry --resume-from-checkpoint with that path"
        ) from error
    message = (f"resume: {resume.checkpoint} at step {resume.step}; "
               f"discarded {len(records) - len(retained)} later metric records and {len(removed)} later checkpoints; "
               "rewound quality evidence; cleared assessment, preflight, and verify reports")
    return len(retained), message
