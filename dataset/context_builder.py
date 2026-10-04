"""Direct, atomic editing of generation Context message arrays."""

import io
import json
import math
import pathlib

from dataset import authoring
from dataset.io import DatasetError, write_text


_MISSING = object()


# Reject links and non-files before Context reads or replacements can follow them.
def _context_path(path, *, must_exist=True):
    target = pathlib.Path(path)
    if target.is_symlink():
        raise DatasetError(f"{path}: Context must be a regular file, not a symlink")
    if must_exist and not target.exists():
        raise DatasetError(f"{path}: no such Context file; create it with dataset context create")
    if target.exists() and not target.is_file():
        raise DatasetError(f"{path}: Context must be a regular file, not a directory")
    return target


# Apply the shared finite-JSON policy while retaining object field order and exact strings.
def _read_json_file(path, description):
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise DatasetError(f"{path}: {description} must be a regular JSON file")
    try:
        with source.open(encoding="utf-8") as stream:
            return authoring.read_json(stream, str(path))
    except OSError as error:
        raise DatasetError(f"{path}: cannot read {description}: {error}") from error


# Python callers can supply values without passing through JSON parsing, so validate recursively.
def _validate_json_value(value, where):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetError(f"{where}: JSON numbers must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DatasetError(f"{where}: JSON object keys must be strings")
            _validate_json_value(item, f"{where}.{key}")
        return
    raise DatasetError(f"{where}: value of type {type(value).__name__} is not valid JSON")


# Function arguments are JSON text inside JSON; validate them without exposing their contents.
def _function_arguments(value, where):
    if not isinstance(value, str):
        raise DatasetError(f"{where}: function.arguments must be a JSON string containing an object")
    try:
        parsed = authoring.read_json(io.StringIO(value), where)
    except DatasetError:
        raise DatasetError(f"{where}: function.arguments must contain one finite JSON object") from None
    if not isinstance(parsed, dict):
        raise DatasetError(f"{where}: function.arguments must contain a JSON object")


# Validate the durable array and all standard function-tool relationships in one ordered pass.
def validate_context(messages, source="Context"):
    if not isinstance(messages, list):
        raise DatasetError(f"{source}: Context must be a JSON array of message objects")
    _validate_json_value(messages, source)
    declared = set()
    pending = {}
    completed = set()
    calls = results = 0
    for index, message in enumerate(messages):
        where = f"{source}: message {index}"
        if not isinstance(message, dict):
            raise DatasetError(f"{where}: each Context entry must be a JSON object")
        role = message.get("role")
        tool_calls = message.get("tool_calls", _MISSING)
        tool_call_id = message.get("tool_call_id", _MISSING)

        if role == "tool":
            if tool_calls is not _MISSING:
                raise DatasetError(f"{where}: a tool-result message cannot declare tool_calls")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise DatasetError(f"{where}: a tool-result message requires a nonempty tool_call_id")
            if tool_call_id in completed:
                raise DatasetError(f"{where}: tool call {tool_call_id!r} already has a result")
            if tool_call_id not in pending:
                raise DatasetError(f"{where}: tool result refers to unknown or non-pending call {tool_call_id!r}")
            del pending[tool_call_id]
            completed.add(tool_call_id)
            results += 1
            continue

        if tool_call_id is not _MISSING:
            raise DatasetError(f"{where}: tool_call_id is valid only on a role='tool' result message")
        if pending:
            ids = ", ".join(repr(item) for item in pending)
            raise DatasetError(f"{where}: non-tool message appears before results for pending calls {ids}")
        if tool_calls is _MISSING:
            continue
        if role != "assistant":
            raise DatasetError(f"{where}: standard tool_calls require role='assistant'")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise DatasetError(f"{where}.tool_calls: expected a nonempty array")
        for call_index, call in enumerate(tool_calls):
            call_where = f"{where}.tool_calls[{call_index}]"
            if not isinstance(call, dict):
                raise DatasetError(f"{call_where}: expected a JSON object")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise DatasetError(f"{call_where}.id: expected a nonempty string")
            if call_id in declared:
                raise DatasetError(f"{call_where}.id: duplicate tool call ID {call_id!r}")
            if call.get("type") != "function":
                raise DatasetError(f"{call_where}.type: expected 'function'")
            function = call.get("function")
            if not isinstance(function, dict):
                raise DatasetError(f"{call_where}.function: expected a JSON object")
            if not isinstance(function.get("name"), str) or not function["name"]:
                raise DatasetError(f"{call_where}.function.name: expected a nonempty string")
            _function_arguments(function.get("arguments"), f"{call_where}.function.arguments")
            declared.add(call_id)
            pending[call_id] = index
            calls += 1
    if pending:
        ids = ", ".join(repr(item) for item in pending)
        raise DatasetError(f"{source}: missing tool results for calls {ids}")
    return {"valid": True, "messages": len(messages), "tool_calls": calls, "tool_results": results}


# Read and validate one durable Context without changing its representation.
def read_context(path):
    target = _context_path(path)
    try:
        with target.open(encoding="utf-8") as stream:
            messages = authoring.read_json(stream, str(path))
    except OSError as error:
        raise DatasetError(f"{path}: cannot read Context: {error}") from error
    validate_context(messages, str(path))
    return messages


# Deterministic formatting makes repeated mutations and review diffs predictable.
def _serialize(messages):
    try:
        return json.dumps(messages, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    except (TypeError, ValueError, UnicodeError) as error:
        raise DatasetError(f"Context cannot be serialized as finite UTF-8 JSON: {error}") from error


# Creation never replaces an existing path; later mutations have their own explicit contract.
def create_context(path, *, progress=None):
    _context_path(path, must_exist=False)
    write_text(path, _serialize([]), progress=progress)
    return {"messages": 0}


# Validate and stage the complete replacement so failed edits preserve the original bytes.
def _mutate(path, change, *, progress=None):
    messages = read_context(path)
    result = change(messages)
    validate_context(messages, str(path))
    _context_path(path)
    write_text(path, _serialize(messages), force=True, progress=progress)
    return {"messages": len(messages), **result}


# Accept insertion positions at every boundary, including append at the current length.
def _insertion_index(index, length, where="index"):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index > length:
        raise DatasetError(f"{where}: expected an integer from 0 through {length}")
    return index


# Resolve a nonempty contiguous range without Python's negative-index behavior.
def _range(index, count, length):
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise DatasetError(f"index: expected a nonnegative integer, got {index!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise DatasetError(f"count: expected a positive integer, got {count!r}")
    if index + count > length:
        raise DatasetError(f"range {index}:{index + count} exceeds Context length {length}")
    return index, index + count


# Insert one caller-supplied message object without restricting provider-owned fields.
def add_message(path, message, at=None, *, progress=None):
    if not isinstance(message, dict):
        raise DatasetError("message: expected one JSON object")

    def change(messages):
        index = len(messages) if at is None else _insertion_index(at, len(messages), "--at")
        messages.insert(index, message.copy())
        return {"inserted": [index]}

    return _mutate(path, change, progress=progress)


# Find every standard call ID without assigning meaning to unrelated provider fields.
def _call_ids(messages):
    result = set()
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    result.add(call["id"])
    return result


# Allocate a reproducible ID from current state while allowing callers to name calls explicitly.
def _call_id(messages, supplied):
    used = _call_ids(messages)
    if supplied is not None:
        if not isinstance(supplied, str) or not supplied:
            raise DatasetError("--call-id: expected a nonempty string")
        if supplied in used:
            raise DatasetError(f"--call-id: duplicate tool call ID {supplied!r}")
        return supplied
    number = 1
    while f"call_{number:04d}" in used:
        number += 1
    return f"call_{number:04d}"


# Build the ordinary paired function-call representation without executing the named tool.
def add_tool_exchange(path, name, arguments, result, call_id=None, at=None, *, progress=None):
    if not isinstance(name, str) or not name:
        raise DatasetError("--name: expected a nonempty tool name")
    if not isinstance(arguments, dict):
        raise DatasetError("tool arguments must be one JSON object")
    _validate_json_value(arguments, "tool arguments")
    if not isinstance(result, str):
        raise DatasetError("tool result content must be text")
    argument_text = json.dumps(arguments, ensure_ascii=False, allow_nan=False,
                               sort_keys=True, separators=(",", ":"))

    def change(messages):
        identifier = _call_id(messages, call_id)
        index = len(messages) if at is None else _insertion_index(at, len(messages), "--at")
        call = {"role": "assistant", "content": None, "tool_calls": [{
            "id": identifier, "type": "function",
            "function": {"name": name, "arguments": argument_text},
        }]}
        tool_result = {"role": "tool", "tool_call_id": identifier, "content": result}
        messages[index:index] = [call, tool_result]
        return {"inserted": [index, index + 1], "call_id": identifier}

    return _mutate(path, change, progress=progress)


# Replace either exact message content or the complete provider-owned message object.
def replace_message(path, index, *, content=_MISSING, message=None, progress=None):
    if (content is _MISSING) == (message is None):
        raise DatasetError("replace requires exactly one of content or message")
    if content is not _MISSING and not isinstance(content, str):
        raise DatasetError("replacement content must be text")
    if message is not None and not isinstance(message, dict):
        raise DatasetError("replacement message must be one JSON object")

    def change(messages):
        start, _ = _range(index, 1, len(messages))
        if message is not None:
            messages[start] = message.copy()
        else:
            updated = messages[start].copy()
            updated["content"] = content
            messages[start] = updated
        return {"replaced": [start]}

    return _mutate(path, change, progress=progress)


# Remove an explicit range and let final validation prevent half-tool exchanges.
def remove_messages(path, index, count=1, *, progress=None):
    def change(messages):
        start, end = _range(index, count, len(messages))
        del messages[start:end]
        return {"removed": {"index": start, "count": count}}

    return _mutate(path, change, progress=progress)


# Move a range using destination coordinates from the array after source removal.
def move_messages(path, source, destination, count=1, *, progress=None):
    def change(messages):
        start, end = _range(source, count, len(messages))
        moving = messages[start:end]
        del messages[start:end]
        target = _insertion_index(destination, len(messages), "destination")
        messages[target:target] = moving
        return {"moved": {"from": start, "to": target, "count": count}}

    return _mutate(path, change, progress=progress)


# Name JSON content types without exposing the content itself in ordinary outlines.
def _content_type(message):
    if "content" not in message:
        return "missing"
    value = message["content"]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "number"


# Produce a compact index that an agent can inspect without loading the full conversation.
def outline_context(path, preview=None):
    if preview is not None and (isinstance(preview, bool) or not isinstance(preview, int) or preview < 0):
        raise DatasetError("--preview: expected a nonnegative integer")
    messages = read_context(path)
    rows = []
    for index, message in enumerate(messages):
        content = message.get("content", _MISSING)
        calls = []
        if isinstance(message.get("tool_calls"), list):
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                calls.append({
                    "id": call.get("id") if isinstance(call, dict) else None,
                    "name": function.get("name") if isinstance(function, dict) else None,
                })
        row = {
            "index": index,
            "role": message.get("role", "<missing>"),
            "content_type": _content_type(message),
            "content_characters": len(content) if isinstance(content, str) else None,
            "tool_calls": calls,
            "tool_call_id": message.get("tool_call_id"),
        }
        if preview is not None and isinstance(content, str):
            row["preview"] = json.dumps(content[:preview], ensure_ascii=False)[1:-1]
        rows.append(row)
    return rows


# Return only the requested region, preserving exact message values for focused review.
def show_messages(path, index, count=1):
    messages = read_context(path)
    start, end = _range(index, count, len(messages))
    selected = messages[start:end]
    return selected[0] if count == 1 else selected


# Read a raw message file through the same finite-JSON boundary used by Context files.
def read_message_file(path):
    value = _read_json_file(path, "message file")
    if not isinstance(value, dict):
        raise DatasetError(f"{path}: message file must contain one JSON object")
    return value


# Read typed tool arguments while retaining their JSON scalar and collection values.
def read_arguments_file(path):
    value = _read_json_file(path, "arguments file")
    if not isinstance(value, dict):
        raise DatasetError(f"{path}: arguments file must contain one JSON object")
    return value
