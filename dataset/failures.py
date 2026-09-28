"""Dependency-free failure evidence shared by command and worker boundaries.

Only explicitly supplied metadata is collected: never frame locals or tensors.
Exception identity and the deepest operation survive wrapping and transport.
"""

import contextlib
import math
import traceback
import uuid


class Error(Exception):
    """An expected operator-facing failure, with optional structured evidence."""

    # Existing positional exception construction remains compatible.
    def __init__(self, *args, context=None, evidence=None):
        super().__init__(*args)
        annotate(self, context=context, evidence=evidence)


# Follow Python's displayed chain, avoiding cycles and suppressed contexts.
def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ if error.__cause__ is not None else (
            None if error.__suppress_context__ else error.__context__)


# Metadata is a JSON tree supplied by the operation, never arbitrary objects.
def _safe(value):
    # Reject unknown objects instead of invoking repr(), which can copy GPU data
    # or disclose input contents. Explicit metadata may contain nested summaries.
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items() if isinstance(key, (str, int))}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return f"unavailable ({type(value).__name__})"


# Attach evidence on the exception so ordinary re-raising preserves ownership.
def annotate(error, *, context=None, evidence=None):
    """Add operation-owned facts; inner annotations win during unwinding."""
    try:
        for name, values in (("context", context), ("evidence", evidence)):
            if values:
                current = getattr(error, "_failure_" + name, {}).copy()
                for key, value in values.items():
                    current.setdefault(key, _safe(value))
                setattr(error, "_failure_" + name, current)
    except Exception:
        error.add_note("Failure metadata collection failed; original error preserved.")
    return error


# Context can be captured by silent library callers as well as CLI stages.
@contextlib.contextmanager
def context(**fields):
    try:
        yield
    except Exception as error:
        annotate(error, context=fields)
        raise


# Transport carries a report, not a synthetic replacement traceback.
def attach(error, report):
    """Attach an already captured worker report without rebuilding its cause."""
    error._failure_report = report
    # The supplied snapshot owns existing fields; later annotations add to it.
    # Stale wrapper metadata must not restore fields deliberately removed from a report.
    error._failure_context = {}
    error._failure_evidence = {}
    return error


# Credentials remain local to the boundary that owns them.
def redact(error, sanitizer):
    """Register process-local credential redaction for all serialized strings."""
    error._failure_sanitizer = sanitizer
    return error


# Redact text and metadata keys while retaining the report/exception protocol's fixed keys.
def _strings(value, sanitizer, *, schema=False):
    if isinstance(value, str):
        redacted = sanitizer(value)
        if not isinstance(redacted, str):
            raise TypeError("credential sanitizer must return text")
        return redacted
    if isinstance(value, list):
        return [_strings(item, sanitizer, schema=schema) for item in value]
    if isinstance(value, dict):
        return {(key if schema else _strings(key, sanitizer)):
                _strings(item, sanitizer, schema=schema and key == "causes") for key, item in value.items()}
    return value


# Capturing is idempotent across wrappers and never inspects traceback locals.
def capture(error, *, context=None, evidence=None):
    """Build a serializable report, keeping diagnostic failures secondary."""
    annotate(error, context=context, evidence=evidence)
    chain = list(_chain(error))
    report = None
    try:
        for cause in reversed(chain):
            transported = getattr(cause, "_failure_report", None)
            if transported is not None:
                report = _safe(transported)
                break
        if report is None:
            identity = next((getattr(cause, "_failure_id", None) for cause in reversed(chain)
                             if getattr(cause, "_failure_id", None)), None) or uuid.uuid4().hex
            report = dict(id=identity, summary=str(error), expected=isinstance(error, Error),
                          context={}, evidence={}, causes=[], traceback="", notes=[])
            for cause in chain:
                cause._failure_id = identity
                frames = traceback.extract_tb(cause.__traceback__)
                location = (f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}"
                            if frames else None)
                report["causes"].append(dict(type=type(cause).__name__, message=str(cause), location=location))
            report["traceback"] = "".join(traceback.format_exception(error))
        for cause in reversed(chain):
            if isinstance(cause, OSError):
                report["evidence"].setdefault("os_error", _safe({
                    "errno": cause.errno, "message": cause.strerror,
                    "path": cause.filename, "other_path": cause.filename2,
                }))
            for name in ("context", "evidence"):
                for key, value in getattr(cause, "_failure_" + name, {}).items():
                    report[name].setdefault(key, _safe(value))
            for note in getattr(cause, "__notes__", ()):
                if note not in report["notes"]:
                    report["notes"].append(str(note))
    except Exception:
        # A broken exception __str__ or diagnostic provider must not mask failure.
        if report is None:
            report = dict(id=uuid.uuid4().hex, summary=f"{type(error).__name__}: details unavailable",
                          expected=isinstance(error, Error), context={}, evidence={}, causes=[],
                          traceback="", notes=[])
        report.setdefault("notes", []).append("Failure report collection incomplete; available original evidence preserved.")
    for cause in chain:
        sanitizer = getattr(cause, "_failure_sanitizer", None)
        if sanitizer is not None:
            try:
                report = _strings(report, sanitizer, schema=True)
            except Exception:
                # Fail closed: a failed sanitizer must never expose its input.
                report = dict(id=report["id"], summary=f"{type(error).__name__}: details withheld",
                              expected=report["expected"], context={}, evidence={}, causes=[], traceback="",
                              notes=["Credential redaction failed; diagnostic details withheld."])
    return report


# Nested evidence stays readable without exposing Python container syntax.
def _facts(value, indent=2):
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = key.replace("_", " ")
            if item is None or item == {} or item == []:
                continue
            if isinstance(item, (dict, list)):
                children = _facts(item, indent + 2)
                if children:
                    lines.append(f"{prefix}{label}:")
                    lines.extend(children)
            elif isinstance(item, int) and not isinstance(item, bool) and (key == "bytes" or key.endswith("_bytes")):
                # Format only at presentation; reports retain exact byte counts
                # for transport, accounting, and later machine inspection.
                divisor, unit = (1024 ** 3, "GiB") if abs(item) >= 1024 ** 3 else (1024 ** 2, "MiB")
                lines.append(f"{prefix}{label}: {item / divisor:.3f} {unit} ({item:,} bytes)")
            elif item is not None:
                lines.append(f"{prefix}{label}: {item}")
        return lines
    if isinstance(value, list):
        if all(isinstance(item, (int, float)) for item in value):
            return [prefix + ", ".join(str(item) for item in value)]
        lines = []
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}item {index + 1}:")
                lines.extend(_facts(item, indent + 2))
            else:
                lines.append(f"{prefix}{item}")
        return lines
    return [prefix + str(value)]


# The terminal and full run log share evidence; only technical detail differs.
def render(report, *, detailed=False):
    """Render the same facts for terminal and log; the log adds technical detail."""
    try:
        lines = [report["summary"]]
        for section in ("context", "evidence"):
            lines.extend(_facts(report.get(section, {})))
        for cause in report.get("causes", [])[1:]:
            lines.append(f"  Caused by {cause['type']}: {cause['message']}")
        causes = report.get("causes", [])
        if causes and causes[-1].get("location"):
            lines.append(f"  Failure location: {causes[-1]['location']}")
        lines.extend(f"  {note}" for note in report.get("notes", []))
        if detailed and report.get("traceback"):
            lines.extend(("", report["traceback"].rstrip()))
        return "\n".join(lines)
    except Exception:
        summary = report.get("summary") if isinstance(report, dict) else None
        lines = [summary if isinstance(summary, str) else "Failure details unavailable",
                 "  Additional diagnostic rendering failed; original summary preserved."]
        if detailed and isinstance(report, dict) and isinstance(report.get("traceback"), str):
            lines.append(report["traceback"])
        return "\n".join(lines)
