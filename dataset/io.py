"""Read and write row datasets by file extension.

Row model: a dataset is a Python list of dicts, fully in memory. Every
subcommand is a function from rows to rows; this module is the only place
that touches file formats.

Formats by extension: .jsonl, .json (array of objects), .csv, .parquet.
"""

import csv
import json
import pathlib

import pyarrow
import pyarrow.parquet


# Raised for any user-facing failure in this package. main() prints its message
# and exits nonzero, so callers never see a traceback for bad input.
class DatasetError(Exception):
    pass


# Reads a JSONL file. A line that is not a JSON object is an error naming the line.
def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON: {e.msg} at column {e.colno}")
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")
            rows.append(row)
    return rows


# One object per line, non-ASCII kept as is so text round-trips readably.
def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# Reads a JSON file that must be an array of objects.
def _read_json(path):
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetError(f"{path}:{e.lineno}: invalid JSON: {e.msg} at column {e.colno}")
    if not isinstance(data, list):
        raise DatasetError(f"{path}: expected a JSON array of objects, got {type(data).__name__}")
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise DatasetError(f"{path}: element {i} is {type(row).__name__}, expected an object")
    return data


# One array, indented one space per level to stay diffable without bloat.
def _write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")


# CSV cells are read as strings; no type inference, since guessing types would
# silently change data.
def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# CSV cannot carry lists or dicts. Refuses rather than JSON-encoding silently.
# Columns are the union over all rows in first-seen order.
def _write_csv(path, rows):
    columns = []
    seen = set()
    for i, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                raise DatasetError(
                    f"cannot write column '{key}' to CSV: row {i} holds a {type(value).__name__}"
                )
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# Whole table to Python values; nested struct columns become dicts and lists.
def _read_parquet(path):
    return pyarrow.parquet.read_table(path).to_pylist()


# from_pylist infers one schema across all rows; rows with conflicting nested
# shapes surface as a pyarrow error, which we translate rather than pass through.
def _write_parquet(path, rows):
    try:
        table = pyarrow.Table.from_pylist(rows)
    except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError) as e:
        raise DatasetError(f"cannot write Parquet: rows do not share one schema: {e}")
    pyarrow.parquet.write_table(table, path)


# Single source of truth for supported formats.
FORMATS = {
    ".jsonl": (_read_jsonl, _write_jsonl),
    ".json": (_read_json, _write_json),
    ".csv": (_read_csv, _write_csv),
    ".parquet": (_read_parquet, _write_parquet),
}


# Maps a path to its (reader, writer) pair. The extension is the only source
# of the format; there is no default, so a missing extension is an error.
def _format(path):
    suffix = pathlib.Path(path).suffix.lower()
    if suffix not in FORMATS:
        known = ", ".join(sorted(FORMATS))
        if not suffix:
            raise DatasetError(f"{path}: a file extension is required to choose the format; one of {known}")
        raise DatasetError(f"{path}: unsupported extension '{suffix}'; expected one of {known}")
    return FORMATS[suffix]


# Reads a dataset file into a list of dicts.
def read_rows(path):
    reader, _ = _format(path)
    if not pathlib.Path(path).is_file():
        raise DatasetError(f"{path}: no such file")
    try:
        return reader(path)
    except OSError as e:
        raise DatasetError(f"{path}: cannot read: {e.strerror or e}")


# Writes rows to path. Refuses to overwrite any of `inputs`, by resolved path,
# because the tool never writes in place.
def write_rows(path, rows, inputs=()):
    _, writer = _format(path)
    target = pathlib.Path(path).resolve()
    for src in inputs:
        if pathlib.Path(src).resolve() == target:
            raise DatasetError(f"{path}: output would overwrite input; the tool never writes in place")
    try:
        writer(path, rows)
    except OSError as e:
        raise DatasetError(f"{path}: cannot write: {e.strerror or e}")
