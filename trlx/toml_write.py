"""TOML emitter for `trlx init`.

tomllib reads but does not write, and the spec allows no TOML-writing
dependency. This emitter covers exactly what a run config needs: tables,
scalars, arrays, inline tables, comment lines, and commented-out keys.

Output is line-oriented. A Writer accumulates lines; init_cmd decides what to
write and in what order, this module only decides how each item is spelled.
"""

import enum
import json
import re
import textwrap

# Comment lines wrap at this width including the "# " prefix.
WRAP = 78

# TOML bare keys. Anything else is quoted.
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


# Renders a Python value as a TOML value. None is deliberately unsupported:
# TOML has no null, and the caller decides how an absent value is written.
# Any other unsupported type is a bug in the caller, so it raises TypeError.
def format_value(value):
    # bool before int: bool is an int subclass.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        return format_value(value.value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr yields "1e-05", "0.001", "inf", "nan": all valid TOML floats.
        return repr(value)
    if isinstance(value, str):
        # JSON string escapes are a subset of TOML basic-string escapes, and
        # raw non-ASCII is valid in both, so json.dumps produces a valid TOML
        # basic string.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_key(k)} = {format_value(v)}" for k, v in value.items())
        return "{" + items + "}"
    if value is None:
        raise TypeError("None has no TOML spelling; the caller must write the key commented out")
    raise TypeError(f"cannot write a {type(value).__name__} as TOML")


# Bare when TOML allows it, quoted otherwise.
def _key(name):
    return name if _BARE_KEY.match(name) else json.dumps(name, ensure_ascii=False)


class Writer:
    def __init__(self):
        self._lines = []

    def blank(self):
        self._lines.append("")

    # Wraps free text as comment lines. Paragraphs (blank-line separated) are
    # wrapped independently. "%%" is the argparse escape used in some TRL help
    # strings and is unescaped here so the comment reads as prose.
    def comment(self, text):
        text = text.replace("%%", "%")
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                self._lines.append("#")
                continue
            for line in textwrap.wrap(paragraph.strip(), width=WRAP - 2):
                self._lines.append("# " + line)

    # A table header. commented=True writes "# [name]" for an optional block
    # the operator enables by uncommenting; init_cmd writes every key beneath
    # such a header commented as well.
    def table(self, name, commented=False):
        line = f"[{_key(name)}]"
        self._lines.append("# " + line if commented else line)

    # A key line. value=None writes "# name =" with no value: TOML has no
    # null, and uncommenting the line without supplying a value is a parse
    # error rather than a silent placeholder. A None value is therefore
    # always commented, whatever `commented` says.
    def key(self, name, value, commented=False):
        if value is None:
            self._lines.append(f"# {_key(name)} =")
            return
        line = f"{_key(name)} = {format_value(value)}"
        self._lines.append("# " + line if commented else line)

    # The document. Always ends in a newline.
    def text(self):
        return "\n".join(self._lines) + "\n"
