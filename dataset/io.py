"""Read and write row datasets by file extension.

Row model: a dataset is a Python list of dicts, fully in memory. Every
subcommand is a function from rows to rows; this module is the only place
that touches file formats.

Formats by extension: .jsonl, .json (array of objects), .csv, .parquet.
"""

import csv
import json
import os
import pathlib
import shlex
import shutil
import stat
import sys
import tempfile

import pyarrow
import pyarrow.parquet


# Raised for any user-facing failure in this package. main() prints its message
# and exits nonzero, so callers never see a traceback for bad input.
class DatasetError(Exception):
    pass


# Suggest a separate repair output; shell quoting keeps paths with spaces usable.
def _heal_guidance(path):
    source = pathlib.Path(path)
    destination = source.with_name(source.stem + ".repaired" + source.suffix)
    return f"inspect the syntax or try dataset heal {shlex.quote(str(source))} --out {shlex.quote(str(destination))}"


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
                raise DatasetError(f"{path}:{lineno}: invalid JSON: {e.msg} at column {e.colno}; {_heal_guidance(path)}") from e
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
            raise DatasetError(f"{path}:{e.lineno}: invalid JSON: {e.msg} at column {e.colno}; {_heal_guidance(path)}") from e
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


# Translate file/format failures where the input path and a recovery action are known.
def read_rows(path):
    reader, _ = _format(path)
    try:
        if not pathlib.Path(path).exists():
            raise DatasetError(f"{path}: no such file; supply an existing dataset path")
        if not pathlib.Path(path).is_file():
            raise DatasetError(f"{path}: not a regular file; supply a dataset file, not a directory")
        return reader(path)
    except UnicodeError as e:
        raise DatasetError(f"{path}: cannot decode dataset: {e}; save text datasets as UTF-8") from e
    except (csv.Error, pyarrow.ArrowInvalid, pyarrow.ArrowTypeError) as e:
        raise DatasetError(f"{path}: cannot parse dataset: {e}; repair or export a valid {pathlib.Path(path).suffix} file") from e
    except OSError as e:
        raise DatasetError(f"{path}: cannot read: {e.strerror or e}; check the path and read permissions") from e


# Resolve parent aliases while preserving the final symlink as the replacement object.
# Healing explicitly resolves that final link too, since it repairs the linked target.
def validate_output(path, force=False, *, follow_symlinks=False, directory=False):
    try:
        # Resolve '..' after following parent links, matching filesystem traversal.
        original = pathlib.Path(path)
        target = (original.resolve() if follow_symlinks or original.name in ("", "..")
                  else original.parent.resolve() / original.name)
        exists = target.exists() or target.is_symlink()
        if exists and not force:
            consequence = "replace the symlink itself" if target.is_symlink() else (
                "remove the directory and all its contents" if target.is_dir() else "replace the existing file"
            )
            raise DatasetError(
                f"{path}: output already exists; this would {consequence}. "
                "Use --force to authorize replacement, or choose another output path"
            )
        if not target.parent.is_dir() and (target.parent.exists() or not directory):
            raise DatasetError(f"{path}: output parent {target.parent} is not a directory; create the parent directory first")
        return target
    except (OSError, RuntimeError) as e:
        raise DatasetError(f"{path}: cannot inspect output: {e}; check the path and directory permissions") from e


# Dataset output format is determined before generation, using the requested filename.
def validate_rows_output(path, force=False):
    _format(path)
    return validate_output(path, force)


# Never follow a final symlink during deletion, including links to directories.
def _remove_output(path):
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


# Preserve permissions only for the actual regular file being replaced, not a link target.
def _output_mode(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return stat.S_IMODE(info.st_mode) if stat.S_ISREG(info.st_mode) else None


# Reserve a new destination exclusively, so a late arrival is not silently overwritten.
def _reserve_output(path, directory):
    if directory:
        path.mkdir()
    else:
        with path.open("x", encoding="utf-8"):
            pass


# Keep staging outside the replaced tree; model output may contain all its own inputs.
def _prepare_output(target, writer, directory):
    folder = pathlib.Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    prepared = folder / "result"
    try:
        _reserve_output(prepared, directory)
        writer(prepared)
        mode = _output_mode(target)
        if mode is not None and not directory:
            prepared.chmod(mode)
        return folder
    except BaseException as failure:
        # Preserve the producer failure and retained scratch location if cleanup also fails.
        try:
            shutil.rmtree(folder)
        except OSError as cleanup_error:
            message = f"{target}: preparation failed: {failure}; cannot clean {folder}: {cleanup_error}; inspect and remove the staging directory"
            if isinstance(failure, (OSError, DatasetError)):
                raise DatasetError(message) from failure
            failure.add_note(message)
        raise


# A retained previous output means recovery or cleanup failed; never discard that copy.
def _clean_stage(folder):
    if folder.exists() and not (folder / "previous").exists() and not (folder / "previous").is_symlink():
        shutil.rmtree(folder)


# Publish files atomically. Directory/type replacement uses a recoverable rename sequence;
# it is not an atomic directory exchange and does not coordinate with unrelated processes.
def _publish_prepared(target, folder, force, directory):
    prepared = folder / "result"
    previous = folder / "previous"
    validate_output(target, force, directory=directory)
    if not force:
        # No-force publication never enters the replacement branch, even if another
        # process creates a directory after validation. Reserve exclusively instead.
        if directory:
            target.mkdir()
            try:
                prepared.replace(target)
            except OSError:
                target.rmdir()
                raise
        else:
            os.link(prepared, target)
            prepared.unlink()
        return
    exists = target.exists() or target.is_symlink()
    if not directory and (not exists or not target.is_dir() or target.is_symlink()):
        prepared.replace(target)
        return
    if exists:
        target.rename(previous)
        try:
            prepared.replace(target)
        except OSError as publish_error:
            try:
                previous.replace(target)
            except OSError as restore_error:
                raise DatasetError(
                    f"{target}: publication failed: {publish_error}; restoration failed: {restore_error}. "
                    f"Original output is at {previous}; prepared output is at {prepared}. "
                    "Restore the original or publish the prepared output after correcting the filesystem error"
                ) from publish_error
            raise DatasetError(f"{target}: publication failed: {publish_error}; original output restored; correct the filesystem error and retry") from publish_error
        try:
            _remove_output(previous)
        except OSError as e:
            raise DatasetError(
                f"{target}: new output published, but old-output cleanup failed: {e}; "
                f"old contents remain at {previous}; remove that retained copy after correcting the filesystem error"
            ) from e
    else:
        # Reserve an absent directory before installing the prepared tree.
        target.mkdir()
        try:
            prepared.replace(target)
        except OSError:
            target.rmdir()
            raise


# Callers finish reading inputs before calling this function. Direct mode deliberately
# gives up preservation on failure; force authorizes replacement, not silent link traversal.
def publish_output(path, writer, *, force=False, no_staging=False, directory=False, follow_symlinks=False):
    target = validate_output(path, force, follow_symlinks=follow_symlinks, directory=directory)
    folder = None
    try:
        if directory:
            target.parent.mkdir(parents=True, exist_ok=True)
        if no_staging:
            mode = _output_mode(target)
            if force:
                _remove_output(target)
            _reserve_output(target, directory)
            writer(target)
            # A writable parent permits replacing a read-only file. Restore its mode
            # only after writing so direct mode does not fail reopening its own output.
            if mode is not None and not directory:
                target.chmod(mode)
        else:
            folder = _prepare_output(target, writer, directory)
            _publish_prepared(target, folder, force, directory)
    except (OSError, DatasetError) as e:
        consequence = (
            "direct writing was requested; the original may be lost and the destination may be incomplete"
            if no_staging else "staged output operation failed; inspect the publication status above"
        )
        recovery = ("Use --force to replace the output created by another writer, or choose another path"
                    if isinstance(e, FileExistsError) and not force else
                    "Check available space and filesystem permissions before retrying")
        raise DatasetError(
            f"{path}: cannot write output: {e}; {consequence}. {recovery}"
        ) from e
    finally:
        if folder is not None:
            failure = sys.exception()
            try:
                _clean_stage(folder)
            except OSError as e:
                detail = f"{failure}; " if failure is not None else f"{path}: output published; "
                message = f"{detail}cannot clean staging directory {folder}: {e}; remove it after checking the output"
                if failure is not None and not isinstance(failure, (OSError, DatasetError)):
                    failure.add_note(message)
                else:
                    raise DatasetError(message) from failure


# Text reports/configs and healing share publication policy without changing their content.
def write_text(path, text, *, force=False, no_staging=False, follow_symlinks=False):
    # The destination is already reserved; writing here never chooses replacement policy.
    def write(prepared):
        try:
            prepared.write_text(text, encoding="utf-8")
        except UnicodeError as e:
            raise DatasetError(f"{path}: cannot encode output as UTF-8; correct invalid Unicode characters in the input") from e

    publish_output(path, write, force=force, no_staging=no_staging, follow_symlinks=follow_symlinks)


# Serialization errors describe the operator's output, never the temporary staging filename.
def _rows_writer(path, rows):
    _, writer = _format(path)

    # Unsupported values may originate in input formats or user expressions, not code bugs.
    def write(prepared):
        try:
            writer(prepared, rows)
        except (TypeError, ValueError, UnicodeError, csv.Error, pyarrow.ArrowInvalid, pyarrow.ArrowTypeError) as e:
            raise DatasetError(
                f"{path}: cannot serialize {pathlib.Path(path).suffix} output: {e}; "
                "correct incompatible values or choose a format that supports them (JSONL/Parquet for nested data)"
            ) from e
        except DatasetError as e:
            raise DatasetError(f"{path}: {e}; use JSONL/Parquet for nested values or correct the row schema") from e

    return write


# Inputs have been fully materialized by the caller, so force can replace the input path.
# Keep the inputs argument for existing callers; replacement authorization is destination-based.
def write_rows(path, rows, inputs=(), *, force=False, no_staging=False):
    validate_rows_output(path, force)
    publish_output(path, _rows_writer(path, rows), force=force, no_staging=no_staging)


# Distinct output entries are required even with force: two results cannot occupy one path.
# Final symlinks are separate entries; regular hard links are treated as conflicting outputs.
def _validate_outputs(outputs, force):
    targets = [validate_rows_output(path, force) for path, _ in outputs]
    for i, target in enumerate(targets):
        for other in targets[:i]:
            same_file = (target.is_file() and other.is_file() and not target.is_symlink()
                         and not other.is_symlink() and target.samefile(other))
            if target == other or target in other.parents or other in target.parents or same_file:
                raise DatasetError(f"{target} and {other}: conflicting output destinations; choose distinct --out and --rest paths")
    return targets


# Split prepares every serialization before publishing any result. Publication itself is
# sequential, so a second-output failure must identify results already changed.
def write_many_rows(outputs, inputs=(), *, force=False, no_staging=False):
    outputs = list(outputs)
    prepared = []
    changed = []
    try:
        targets = _validate_outputs(outputs, force)
        if not no_staging:
            for target, (path, rows) in zip(targets, outputs):
                prepared.append(_prepare_output(target, _rows_writer(path, rows), False))
        for i, (target, (path, rows)) in enumerate(zip(targets, outputs)):
            if no_staging:
                publish_output(target, _rows_writer(path, rows), force=force, no_staging=True)
            else:
                _publish_prepared(target, prepared[i], force, False)
            changed.append(str(path))
    except (OSError, DatasetError) as e:
        status = "completed destinations: " + ", ".join(changed) if changed else "no destination completed"
        raise DatasetError(f"cannot write split outputs: {e}; {status}. Check the named destinations before retrying") from e
    finally:
        failure = sys.exception()
        cleanup_errors = []
        for folder in prepared:
            try:
                _clean_stage(folder)
            except OSError as e:
                cleanup_errors.append(f"{folder}: {e}")
        if cleanup_errors:
            # Cleanup cannot erase the original failure or which split result was installed.
            status = "completed destinations: " + ", ".join(changed) if changed else "no destination completed"
            detail = f"{failure}; " if failure is not None else ""
            message = (f"{detail}cannot clean staging directories: {'; '.join(cleanup_errors)}; {status}; "
                       "inspect the destinations before removing retained staging files")
            if failure is not None and not isinstance(failure, (OSError, DatasetError)):
                failure.add_note(message)
            else:
                raise DatasetError(message) from failure
