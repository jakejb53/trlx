"""Loads environment variables from a .env file.

Shared by both CLIs, like io.py and endpoint.py, so the two tools resolve a
key the same way. The file holds secrets only: operational settings belong in
the run config, where the operator can see them and they can be snapshotted
with a run. CLI/config api_key settings name variables rather than values.
The web UI instead receives credentials from its browser workspace and does
not call this loader.
"""

import os

from dataset.io import DatasetError

FILENAME = ".env"


# Reads KEY=value lines into the environment. A variable already set in the
# real environment wins, so an inline override on the command line beats the
# file and nothing silently replaces what the operator exported. A missing
# file is not an error: the file is optional and the variables may be set
# already. Returns the names assigned, never the values.
def load(path=FILENAME):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    except OSError as e:
        raise DatasetError(f"{path}: cannot read: {e.strerror or e}; check the path and permissions")
    except UnicodeError:
        raise DatasetError(f"{path}: environment file is not valid UTF-8; save it as UTF-8")

    assigned = []
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise DatasetError(f"{path}: line {number}: expected KEY=value")
        key, value = key.strip(), value.strip()
        # Quotes are optional and stripped only as a matching pair, so a value
        # that genuinely starts and ends with a quote character is written as
        # "\"literal\"" rather than losing its outer characters by accident.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # Name only the line: either side may contain credential material.
        if "\0" in key or "\0" in value:
            raise DatasetError(f"{path}: line {number}: environment entry contains NUL; remove the NUL character")
        if key not in os.environ:
            os.environ[key] = value
            assigned.append(key)
    return assigned
