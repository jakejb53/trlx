"""[dataset] block -> train and eval `datasets.Dataset` objects.

Files are read by dataset.io (JSONL, JSON array, CSV, Parquet by extension);
HF ids go through `datasets.load_dataset` with the `:split` rule from SPEC 2.2.
Both arrive as a Dataset before the split rule and the column check, so those
run once regardless of source.

Errors are TrlxError naming the dataset source and, where relevant, the
columns found.
"""

import math
from fractions import Fraction

import datasets
from pyarrow import ArrowInvalid, ArrowTypeError

from dataset.io import DatasetError, read_rows
from trlx import TrlxError

# Column sets that satisfy each Method.dataset_format, in TRL's vocabulary. A
# row set passes when it has every column of at least one alternative; extra
# columns are allowed (trainers ignore or forward them). The trainer is the
# authority on what it consumes; this check exists so a wrong file fails here
# with the columns named instead of deep inside the trainer.
_FORMAT_COLUMNS = {
    "language modeling or prompt-completion": ({"text"}, {"messages"}, {"prompt", "completion"}),
    "preference": ({"prompt", "chosen", "rejected"}, {"chosen", "rejected"}),
    "unpaired preference": ({"prompt", "completion", "label"},),
    "prompt-only": ({"prompt"},),
}


# Loads the train set and the eval set (None when eval is disabled) for a
# config.DatasetSpec, validated against `dataset_format`.
def load(spec, dataset_format):
    if spec.split:
        whole = load_ref(spec.source)
        # Round evaluation upward so a positive fraction always reserves data.
        # Refuse an empty side rather than silently changing the requested split.
        # Decimal spelling avoids binary noise such as 100 * 0.07 rounding up to 8.
        eval_rows = math.ceil(whole.num_rows * Fraction(str(spec.eval_fraction)))
        train_rows = whole.num_rows - eval_rows
        if train_rows < 1 or eval_rows < 1:
            raise TrlxError(
                f"{spec.source.source}: [dataset].eval_fraction = {spec.eval_fraction} leaves "
                f"{train_rows} train rows and {eval_rows} eval rows; both must be nonempty "
                f"(dataset has {whole.num_rows} rows)"
            )
        train = whole.select(range(train_rows))
        eval_set = whole.select(range(train_rows, whole.num_rows))
    else:
        train = load_ref(spec.source)
        eval_set = load_ref(spec.eval_source) if spec.eval_source is not None else None

    _check_columns(spec.source.source, train, dataset_format)
    if eval_set is not None:
        source = spec.source.source if spec.split else spec.eval_source.source
        _check_columns(source, eval_set, dataset_format)
    return train, eval_set


# Column marking replay rows in the mixed train set: true on rows from
# [replay].dataset, false on the operator's train rows. Present only when the
# KL term is on; it is the flag the replay trainer's collator carries.
REPLAY_COLUMN = "replay"


# The train set with [replay].dataset mixed in (SPEC 2.9). `fraction` is the
# replay share of the result, so R replay rows join N train rows where
# R / (N + R) = fraction; R is rounded and at least 1. The first R rows of the
# replay dataset in file order are used, matching the ordered train/eval split.
# The two sets must have the same columns: concatenation needs equal
# features, and the trainer decides the row shape from the first example.
# `flag` adds REPLAY_COLUMN; without it the result is plain mixing.
def mix_replay(train, spec, flag):
    replay = load_ref(spec.dataset)
    if set(replay.column_names) != set(train.column_names):
        raise TrlxError(
            f"{spec.dataset.source}: [replay].dataset columns {{{', '.join(sorted(replay.column_names))}}} "
            f"differ from the train set's {{{', '.join(sorted(train.column_names))}}}"
        )
    count = max(1, round(spec.fraction * train.num_rows / (1 - spec.fraction)))
    if count > replay.num_rows:
        raise TrlxError(
            f"{spec.dataset.source}: [replay].fraction = {spec.fraction} needs {count} replay rows for "
            f"{train.num_rows} train rows; the dataset has {replay.num_rows}"
        )
    replay = replay.select(range(count))
    if flag:
        if REPLAY_COLUMN in train.column_names:
            raise TrlxError(f"[replay]: column '{REPLAY_COLUMN}' is reserved for replay KL row markers; "
                            "rename that column in both inputs before enabling replay KL")
        train = train.add_column(REPLAY_COLUMN, [False] * train.num_rows)
        replay = replay.add_column(REPLAY_COLUMN, [True] * count)
    try:
        return datasets.concatenate_datasets([train, replay])
    except ValueError as e:
        # Same column names but different inferred types (a message key present
        # in one file only, say).
        raise TrlxError(f"{spec.dataset.source}: [replay].dataset rows do not match the train set's shape: {e}")


# One config.DatasetRef to a Dataset.
def load_ref(ref):
    if ref.is_file:
        try:
            rows = read_rows(ref.source)
        except DatasetError as e:
            raise TrlxError(str(e))
        if not rows:
            raise TrlxError(f"{ref.source}: dataset is empty")
        try:
            return datasets.Dataset.from_list(rows)
        except (ArrowInvalid, ArrowTypeError) as e:
            raise TrlxError(f"{ref.source}: cannot convert rows to a dataset: {e}; "
                            "use consistent value types within each column") from e
    return _load_hub(ref)


# HF hub id. With `:split` that split is requested directly. Without one, the
# repo must have exactly one split; otherwise the operator has to choose and
# the error lists what is there.
def _load_hub(ref):
    try:
        if ref.split is not None:
            return datasets.load_dataset(ref.source, split=ref.split)
        loaded = datasets.load_dataset(ref.source)
    except (OSError, ValueError) as e:
        where = ref.source if ref.split is None else f"{ref.source}:{ref.split}"
        raise TrlxError(f"{where}: cannot load from the HF hub: {e}")
    names = list(loaded.keys())
    if len(names) != 1:
        raise TrlxError(
            f"{ref.source}: has splits {', '.join(names)}; name one as {ref.source}:<split>"
        )
    return loaded[names[0]]


# Rejects a Dataset whose columns satisfy none of the format's alternatives.
def _check_columns(source, dataset, dataset_format):
    have = set(dataset.column_names)
    alternatives = _FORMAT_COLUMNS[dataset_format]
    if any(need <= have for need in alternatives):
        return
    wanted = " or ".join("{" + ", ".join(sorted(need)) + "}" for need in alternatives)
    raise TrlxError(
        f"{source}: columns {{{', '.join(sorted(have))}}} do not fit the {dataset_format} format; "
        f"need {wanted}"
    )
