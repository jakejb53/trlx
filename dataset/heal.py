"""Deterministic JSON and JSONL repairs.

A unit is one line for JSONL or the whole file for JSON. Repairs are text
transforms applied in a fixed order, each operating outside string literals.
The unit is parsed once after all transforms: if it parses, the repairs that
changed the text are reported; if not, the original unit is kept and the
original parse error is reported with line and column.

Transform order: single quotes, Python literals, unquoted keys, trailing
commas, unclosed final brace or bracket. After the transforms, concatenated
objects are split (JSONL) or wrapped (JSON), and a JSONL last line that still
fails and looks cut off is dropped as truncated.
"""

import json
import pathlib
import re

from dataset.io import DatasetError

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}


# Returns (flags, open_string): flags[i] is True when character i is inside a
# double-quoted string, quotes included; open_string is True when the text
# ends inside a string. Callers must use open_string, not flags[-1], to ask
# whether a string is unterminated, since a closing quote is flagged True.
# Backslash escapes inside strings are honoured.
def _scan(text):
    flags = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            flags.append(True)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            flags.append(in_string)
    return flags, in_string


# Converts single-quoted strings to double-quoted. Inner double quotes are
# escaped and \' is unescaped. Returns (text, count).
def _single_quotes(text):
    out = []
    i = 0
    count = 0
    in_double = False
    escaped = False
    while i < len(text):
        ch = text[i]
        if in_double:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
            i += 1
        elif ch == '"':
            in_double = True
            out.append(ch)
            i += 1
        elif ch == "'":
            j = i + 1
            body = []
            closed = False
            while j < len(text):
                c = text[j]
                if c == "\\" and j + 1 < len(text):
                    nxt = text[j + 1]
                    # \' needs no escape inside a double-quoted string; every
                    # other escape sequence is carried through as written.
                    body.append("'" if nxt == "'" else c + nxt)
                    j += 2
                    continue
                if c == "'":
                    closed = True
                    break
                body.append('\\"' if c == '"' else c)
                j += 1
            if not closed:
                # Unterminated single-quoted string: leave it; the parse error
                # will report it.
                out.append(text[i:])
                break
            out.append('"' + "".join(body) + '"')
            count += 1
            i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out), count


# Replaces bare True/False/None outside strings. Whole identifiers are
# consumed at once so a suffix like the `None` in `isNone` is never matched.
def _python_literals(text):
    out = []
    count = 0
    i = 0
    flags, _ = _scan(text)
    while i < len(text):
        if not flags[i]:
            m = _IDENT.match(text, i)
            if m:
                word = m.group()
                if word in _PY_LITERALS:
                    word = _PY_LITERALS[word]
                    count += 1
                out.append(word)
                i = m.end()
                continue
        out.append(text[i])
        i += 1
    return "".join(out), count


# Quotes identifiers that sit in key position: after { or , and before :.
def _unquoted_keys(text):
    out = []
    count = 0
    i = 0
    flags, _ = _scan(text)
    expect_key = False
    while i < len(text):
        ch = text[i]
        if not flags[i]:
            if ch in "{,":
                expect_key = True
            elif expect_key and not ch.isspace():
                m = _IDENT.match(text, i)
                if m:
                    rest = text[m.end() :]
                    if rest.lstrip().startswith(":"):
                        out.append(f'"{m.group()}"')
                        count += 1
                        i = m.end()
                        expect_key = False
                        continue
                expect_key = False
        out.append(ch)
        i += 1
    return "".join(out), count


# Removes a comma followed only by whitespace and then a closer or the end of
# the text. End of text counts because _unclosed runs next and will supply the
# closer for a cut-off write like `{"a": 1,`.
def _trailing_commas(text):
    out = []
    count = 0
    flags, _ = _scan(text)
    for i, ch in enumerate(text):
        if ch == "," and not flags[i]:
            rest = text[i + 1 :].lstrip()
            if not rest or rest.startswith(("}", "]")):
                count += 1
                continue
        out.append(ch)
    return "".join(out), count


# Appends closers for brackets still open at the end of the text. Returns the
# text unchanged when the text ends inside a string, since what the string
# should contain is unknowable.
def _unclosed(text):
    stack = []
    flags, open_string = _scan(text)
    for i, ch in enumerate(text):
        if flags[i]:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return text, 0  # mismatched closer: not ours to guess
    if open_string or not stack:
        return text, 0
    return text.rstrip() + "".join(reversed(stack)), len(stack)


# Decodes a sequence of JSON values with nothing but whitespace between them.
# Returns the list, or None if the text is not such a sequence.
def _decode_sequence(text):
    decoder = json.JSONDecoder()
    values = []
    pos = 0
    n = len(text)
    while True:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        try:
            value, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            return None
        values.append(value)
    return values


# True when the scanner ends inside a string or with brackets still open,
# the signature of a write that was cut off.
def _looks_truncated(text):
    depth = 0
    flags, open_string = _scan(text)
    for i, ch in enumerate(text):
        if not flags[i] and ch in "{[":
            depth += 1
        elif not flags[i] and ch in "}]":
            depth -= 1
    return open_string or depth > 0


_TRANSFORMS = [
    ("single quotes", _single_quotes),
    ("Python literals", _python_literals),
    ("unquoted keys", _unquoted_keys),
    ("trailing commas", _trailing_commas),
    ("unclosed final brace or bracket", _unclosed),
]


# Applies the transforms in order. Returns (text, [(name, count), ...]).
def _repair_text(text):
    applied = []
    for name, fn in _TRANSFORMS:
        text, count = fn(text)
        if count:
            applied.append((name, count))
    return text, applied


# Heals one JSONL line. Returns (output_lines, repairs, error) where
# output_lines is a list of repaired line strings (several for concatenated
# objects, none for a dropped truncated line), repairs is a list of
# descriptions, and error is a message or None. Never raises for bad input.
def _heal_jsonl_line(line, lineno, is_last):
    try:
        json.loads(line)
        return [line], [], None
    except json.JSONDecodeError as e:
        original = e  # the `as` name is unbound after the block; keep a reference
    fixed, applied = _repair_text(line)
    repairs = [f"line {lineno}: {name} ({count})" for name, count in applied]
    values = _decode_sequence(fixed)
    if values is not None and len(values) == 1:
        return [fixed], repairs, None
    if values is not None and len(values) > 1:
        repairs.append(f"line {lineno}: concatenated objects split into {len(values)}")
        return [json.dumps(v, ensure_ascii=False) for v in values], repairs, None
    if is_last and _looks_truncated(line):
        return [], [f"line {lineno}: truncated last line dropped"], None
    return [line], [], f"line {lineno} column {original.colno}: {original.msg}"


# Heals a JSONL file. Returns (output_text, repairs, errors).
def heal_jsonl(text):
    lines = text.split("\n")
    # A trailing newline yields an empty final element that is not a line.
    trailing_newline = text.endswith("\n")
    if trailing_newline:
        lines = lines[:-1]
    last_index = max((i for i, l in enumerate(lines) if l.strip()), default=-1)
    out, repairs, errors = [], [], []
    for i, line in enumerate(lines):
        if not line.strip():
            out.append(line)
            continue
        fixed, line_repairs, error = _heal_jsonl_line(line, i + 1, i == last_index)
        out.extend(fixed)
        repairs.extend(line_repairs)
        if error:
            errors.append(error)
    result = "\n".join(out)
    if trailing_newline or out:
        result += "\n"
    return result, repairs, errors


# Heals a JSON file. Concatenated top-level values become an array.
def heal_json(text):
    try:
        json.loads(text)
        return text, [], []
    except json.JSONDecodeError as e:
        original = e  # the `as` name is unbound after the block; keep a reference
    fixed, applied = _repair_text(text)
    repairs = [f"{name} ({count})" for name, count in applied]
    values = _decode_sequence(fixed)
    if values is not None and len(values) == 1:
        return fixed, repairs, []
    if values is not None and len(values) > 1:
        repairs.append(f"concatenated objects wrapped into an array of {len(values)}")
        return json.dumps(values, ensure_ascii=False, indent=1) + "\n", repairs, []
    return text, [], [f"line {original.lineno} column {original.colno}: {original.msg}"]


# Reads, heals by extension, writes. Returns (repairs, errors); the caller
# prints them and sets the exit code.
def heal_file(src, dst):
    suffix = pathlib.Path(src).suffix.lower()
    if suffix not in (".jsonl", ".json"):
        raise DatasetError(f"{src}: heal handles .jsonl and .json, not '{suffix}'")
    if pathlib.Path(dst).suffix.lower() != suffix:
        raise DatasetError(f"{dst}: output extension must match input '{suffix}'")
    if pathlib.Path(src).resolve() == pathlib.Path(dst).resolve():
        raise DatasetError(f"{dst}: output would overwrite input; the tool never writes in place")
    try:
        with open(src, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise DatasetError(f"{src}: {e.strerror}")
    heal = heal_jsonl if suffix == ".jsonl" else heal_json
    result, repairs, errors = heal(text)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(result)
    return repairs, errors
