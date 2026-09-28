"""Dependency-free failure evidence shared by command and worker boundaries.

Only explicitly supplied metadata is collected: never frame locals or tensors.
Exception identity and the deepest operation survive wrapping and transport.
"""

import contextlib
import errno
import math
import textwrap
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


# Describe storage sizes for people; the full evidence retains exact byte counts.
def _size(value):
    divisor, unit = (1024 ** 3, "GiB") if value >= 1024 ** 3 else (1024 ** 2, "MiB")
    return f"{value / divisor:.3f} {unit}"


# Resolve worker-local device numbers only when the report establishes their mapping.
def _device(context, evidence):
    mapping = context.get("cuda_device_mapping", {})
    # After-exception allocator counters can describe a different current device.
    local = evidence.get("allocation_failure", {}).get("local_cuda_device")
    if local is not None and f"local cuda:{local}" in mapping:
        return f"GPU {mapping[f'local cuda:{local}']}"
    if len(mapping) == 1:
        return f"GPU {next(iter(mapping.values()))}"
    return None


# Explain measured memory pressure without mixing later counters with failure-time figures.
def _memory_lines(report):
    context, evidence = report.get("context", {}), report.get("evidence", {})
    native = evidence.get("allocation_failure", {})
    device = _device(context, evidence)
    title = f"Out of memory on {device}" if device else ("Out of GPU memory" if "CUDA" in report["summary"] or native else "Out of memory")
    operation = context.get("phase") or context.get("operation")
    if operation and operation != "trainer running":
        title += f" during {operation}"
    lines = [title + "."]
    for key in ("model", "destination"):
        if context.get(key):
            lines.append(f"{key.capitalize()}: {context[key]}")
    requested, free = native.get("requested_allocation"), native.get("free_device_memory")
    if requested:
        lines.extend(("", f"The failed allocation needed {requested}" + (f"; {free} was free." if free else ".")))
    if native.get("process_memory") and native.get("device_capacity"):
        lines.append(f"The process was using {native['process_memory']} of the device's {native['device_capacity']}.")
        if native.get("reserved_but_unused"):
            lines.append(f"That includes {native['reserved_but_unused']} reserved by PyTorch but unused.")
    if not requested:
        # Older/other allocators need no parser support to retain their actual explanation.
        causes = report.get("causes", [])
        message = causes[-1]["message"] if causes else report["summary"]
        lines.extend(("", message))

    padded = False
    multiple = False
    for name, counts in evidence.get("batch", {}).items():
        if not name.endswith("attention_mask_counts"):
            continue
        lengths = counts.get("sequence_lengths", [])
        slots = counts.get("padded_token_positions")
        actual = counts.get("actual_token_positions")
        if not lengths or slots is None or actual is None:
            continue
        multiple |= len(lengths) > 1
        padded |= slots > actual
        width = slots // len(lengths)
        label = name.removesuffix("attention_mask_counts").strip("_").replace("_", " ")
        if label:
            lines.extend(("", f"{label.capitalize()} batch:"))
        else:
            lines.append("")
        if slots > actual and lengths.count(width) == 1:
            lines.append(f"One {width:,}-token example expanded this entire {len(lengths)}-example batch.")
        else:
            lines.append(f"Batch: {len(lengths)} examples, {width:,} token positions each.")
        percentage = 100 * (slots - actual) / slots if slots else 0
        lines.extend((f"  Actual tokens: {actual:,}", f"  Padded positions: {slots:,} ({percentage:.0f}% padding)"))

    module = evidence.get("failing_module", {})
    if module.get("class"):
        description = module["class"]
        if ".lora_" in module.get("name", ""):
            description = "LoRA " + description.lower()
        lines.extend(("", f"The failure occurred in {description}."))
        tensors = module.get("input_tensors", [])
        if tensors:
            tensor = tensors[0]
            dtype = {"torch.float32": "FP32", "torch.bfloat16": "BF16", "torch.float16": "FP16"}.get(tensor.get("dtype"), tensor.get("dtype"))
            if dtype and isinstance(tensor.get("bytes"), int):
                lines.append(f"Its {dtype} input occupies {_size(tensor['bytes'])}.")
    inventory = evidence.get("parameter_inventory_after_construction", {})
    if "frozen torch.bfloat16" in inventory and "trainable torch.float32" in inventory:
        lines.append("The model has BF16 frozen weights and FP32 trainable weights.")
    if multiple:
        lines.extend(("", "A smaller per-device batch would reduce activation sizes."))
    if padded:
        lines.append("Padding-free execution could avoid the padding cost if supported by the model and attention implementation.")
    return lines


# Expected error messages already explain many failures; add missing context without duplicating them.
def _general_lines(report):
    context, evidence = report.get("context", {}), report.get("evidence", {})
    causes = report.get("causes", [])
    lines = [report["summary"] or (causes[0]["type"] if causes else "Operation failed.")]
    if context.get("signal") and context.get("source") and report["summary"].startswith(f"{context['source']} exited with code"):
        lines = [f"{context['source'].capitalize()} was terminated by {context['signal']}."]
    operation = context.get("phase") or context.get("operation") or context.get("last_operation")
    if operation and operation not in ("trainer running", "running " + context.get("command", "")) and operation.lower() not in lines[0].lower():
        lines.append(f"While {operation}.")
    for key, label in (("model", "Model"), ("endpoint", "Endpoint"), ("destination", "Destination"), ("source", "Source")):
        value = context.get(key)
        if key == "source" and isinstance(value, str) and value.startswith("rank ") and value.lower() in lines[0].lower():
            continue
        if value is not None and str(value) not in "\n".join(lines):
            lines.append(f"{label}: {value}")
    location = []
    for key in ("record", "line", "column", "field"):
        value = context.get(key)
        if value is not None and f"{key} {value}" not in report["summary"].lower():
            location.append(f"{key} {value}")
    if location:
        lines.append("Location: " + ", ".join(location) + ".")
    if context.get("signal") and context["signal"] not in lines[0]:
        lines.append(f"The process was terminated by {context['signal']}.")
    device = _device(context, evidence)
    if device:
        lines.append(f"Device: {device}.")

    # Retain the deepest useful cause, not every wrapper repeating it.
    if len(causes) > 1:
        cause = causes[-1]
        message = cause.get("message", "")
        if message and message not in "\n".join(lines):
            lines.append(f"Cause: {message}")
    expected, observed = evidence.get("expected_type"), evidence.get("observed_type")
    if expected and observed and not (expected in report["summary"] and observed in report["summary"]):
        lines.append(f"Expected {expected}; received {observed}.")
    changed = evidence.get("completed_destinations", [])
    if changed and not all(str(path) in report["summary"] for path in changed):
        lines.append("Already written: " + ", ".join(map(str, changed)))
    module = evidence.get("failing_module", {})
    if module.get("class"):
        lines.append(f"Failing operation: {module['class']}.")
        shapes = [" × ".join(map(str, tensor["shape"])) for tensor in module.get("input_tensors", []) if tensor.get("shape")]
        if shapes:
            lines.append("Input shapes: " + "; ".join(shapes) + ".")
    os_error = evidence.get("os_error", {})
    path = context.get("destination") or os_error.get("path")
    if path and str(path) not in "\n".join(lines):
        lines.append(f"Path: {path}")
    remedies = {errno.ENOSPC: "Free space on the destination filesystem.",
                errno.EDQUOT: "Free space within the account's quota or increase that quota.",
                errno.EACCES: "Check access permissions for this path.",
                errno.EROFS: "Choose a destination on a writable filesystem."}
    if os_error.get("errno") in remedies:
        remedy = remedies[os_error["errno"]]
        if remedy.lower().rstrip(".") not in report["summary"].lower():
            lines.append(remedy)
    if context.get("attempts", 0) > 1 and "attempt" not in report["summary"].lower():
        lines.append(f"The operation failed after {context['attempts']} attempts.")
    return lines


# Report collection failures explicitly without exposing the surrounding evidence tree.
def _diagnostic_notes(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("diagnostic_error", "diagnostic_limitation") and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _diagnostic_notes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _diagnostic_notes(item)


# Cleanup outcomes and incomplete diagnostics remain visible without printing per-process bookkeeping.
def _terminal(report):
    causes = report.get("causes", [])
    oom = any(cause.get("type") == "OutOfMemoryError" for cause in causes) or bool(report.get("evidence", {}).get("allocation_failure"))
    lines = _memory_lines(report) if oom else _general_lines(report)
    cleanup = report.get("evidence", {}).get("cleanup", {})
    if cleanup:
        remaining = [name for name, state in cleanup.items() if not state.get("process_group_gone")]
        forced = [f"{name}: {', '.join(signal for signal in state.get('signals_sent', []) if signal in ('SIGTERM', 'SIGKILL'))}"
                  for name, state in cleanup.items() if any(signal in ("SIGTERM", "SIGKILL") for signal in state.get("signals_sent", []))]
        lines.append("")
        if remaining:
            lines.append(f"Cleanup could not confirm that these process groups exited: {', '.join(remaining)}.")
        elif forced:
            lines.append(f"Cleanup completed using forced termination ({'; '.join(forced)}).")
        else:
            lines.append("Worker cleanup completed." if all(name.startswith("rank ") for name in cleanup)
                         else "Process cleanup completed.")
    for note in [*_diagnostic_notes(report.get("evidence", {})), *report.get("notes", [])]:
        # Peer aborts are an expected consequence of the primary failure, not another diagnosis.
        if note.startswith("Additional failure from ") and note.endswith("distributed training aborted after a worker failure"):
            continue
        if note not in "\n".join(lines):
            lines.append(note)
    context = report.get("context", {})
    if context.get("incomplete_log"):
        lines.extend(("", f"Incomplete diagnostics log: {context['incomplete_log']}"))
    elif context.get("log"):
        lines.extend(("", f"Full diagnostics: {context['log']}"))
    return "\n".join(textwrap.fill(line, width=100, break_long_words=False, break_on_hyphens=False,
                                  subsequent_indent="  " if line.startswith("  ") else "") if line else "" for line in lines)


# Terminal presentation is selective; detailed logs retain the complete original evidence and traceback.
def render(report, *, detailed=False, include_traceback=False):
    """Render an explanation for people, or complete evidence for the durable log."""
    try:
        if not detailed:
            text = _terminal(report)
            if include_traceback and report.get("traceback"):
                text += "\n\n" + report["traceback"].rstrip()
            return text
        lines = [report["summary"]]
        for section in ("context", "evidence"):
            lines.extend(_facts(report.get(section, {})))
        for cause in report.get("causes", [])[1:]:
            lines.append(f"  Caused by {cause['type']}: {cause['message']}")
        causes = report.get("causes", [])
        if causes and causes[-1].get("location"):
            lines.append(f"  Failure location: {causes[-1]['location']}")
        lines.extend(f"  {note}" for note in report.get("notes", []))
        if report.get("traceback"):
            lines.extend(("", report["traceback"].rstrip()))
        return "\n".join(lines)
    except Exception:
        summary = report.get("summary") if isinstance(report, dict) else None
        lines = [summary if isinstance(summary, str) else "Failure details unavailable",
                 "  Additional diagnostic rendering failed; original summary preserved."]
        if (detailed or include_traceback) and isinstance(report, dict) and isinstance(report.get("traceback"), str):
            lines.append(report["traceback"])
        return "\n".join(lines)
