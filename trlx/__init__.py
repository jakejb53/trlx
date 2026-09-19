"""trlx library.

TrlxError is the one exception type that reaches cli.main(), which prints its
message and exits 1. Every user-facing failure (bad config, bad input, missing
file, mismatched adapter) is raised as TrlxError at the boundary where the
message can name the key, path, or row involved. Anything else is a bug and
surfaces as a traceback on purpose.
"""


class TrlxError(Exception):
    pass
