"""Required UTF-8 prompt files and single-pass, data-safe template expansion."""

import pathlib
import re

from dataset.io import DatasetError

_TOKEN = re.compile(r"\[\[(.*?)\]\]|\{([A-Za-z_][A-Za-z_0-9]*)\}", re.DOTALL)
_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z_0-9]*)\}")


# Only template text is interpreted; braces in inserted source data stay literal.
def fill(template, **values):
    # Optional blocks disappear when their input is absent, keeping instructions in files.
    def replace(match):
        block, name = match.groups()
        if block is not None:
            names = _FIELD.findall(block)
            if any(values.get(key) is None for key in names):
                return ""
            return _FIELD.sub(lambda field: str(values[field[1]]), block)
        return str(values[name]) if name in values else match[0]

    return _TOKEN.sub(replace, template)


# Missing files never select packaged defaults; those are exclusively init inputs.
def load(path, *, required=(), allowed=()):
    path = pathlib.Path(path)
    if path.suffix != ".prompt":
        raise DatasetError(f"{path}: prompt file must have a .prompt extension; examples are not active prompts")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise DatasetError(f"{path}: prompt is not valid UTF-8; save it as UTF-8") from error
    except OSError as error:
        raise DatasetError(f"{path}: cannot read UTF-8 prompt: {error}; supply the file or create defaults "
                           "with trlx init --out <unused-config-path> in the intended directory") from error
    if not text.strip():
        raise DatasetError(f"{path}: prompt is empty; provide instructions before running")
    # Plain system rubrics have no template grammar; braces are ordinary authored text.
    if allowed is None:
        return text
    # Optional blocks are deliberately non-nesting; reject incomplete instructions before requests.
    depth = 0
    for delimiter in re.findall(r"\[\[|\]\]", text):
        if (delimiter == "[[" and depth) or (delimiter == "]]" and not depth):
            raise DatasetError(f"{path}: malformed optional block; use non-nested [[...]] blocks")
        depth = 1 if delimiter == "[[" else 0
    if depth:
        raise DatasetError(f"{path}: unclosed optional block; close it with ]]")
    names = set(_FIELD.findall(text))
    missing = set(required) - names
    unknown = names - set(allowed) - set(required)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing placeholders: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown placeholders: " + ", ".join(sorted(unknown)))
        raise DatasetError(f"{path}: {'; '.join(details)}")
    return text
