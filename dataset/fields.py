"""Field operations and the shared row-expression evaluator.

Expressions are Python, evaluated per row with the row's columns bound as
names. This is a local operator tool over the operator's own files, so eval
is the right amount of machinery; the contract is that any failure becomes a
row-numbered message, never a traceback.
"""

import builtins

from dataset.io import DatasetError
from dataset.progress import stage

# Builtins an expression may use. Everything else from builtins is withheld so
# a typo like `lenght` reports as an unknown name rather than surprising behaviour.
_ALLOWED_BUILTINS = {name: getattr(builtins, name) for name in ("len", "str", "int", "float")}


# Compiles an expression once; a syntax error is reported before any row is touched.
def compile_expression(expr):
    try:
        return compile(expr, "<expression>", "eval")
    except SyntaxError as e:
        raise DatasetError(f"expression `{expr}` is not valid Python: {e.msg} at column {e.offset}; "
                           "quote the complete --add or --where argument in the shell, "
                           "e.g. --add 'size=len(text)' or --where 'len(text) > 0'")


# Evaluates a compiled expression against one row. Columns are names; `row` is
# the whole dict for columns whose names are not identifiers.
def evaluate(code, expr, row, index):
    namespace = dict(row)
    namespace["row"] = row
    try:
        return eval(code, {"__builtins__": _ALLOWED_BUILTINS}, namespace)
    except Exception as e:
        raise DatasetError(f"expression `{expr}` failed on row {index}: {type(e).__name__}: {e}")


# Splits a NAME=VALUE flag argument at the first '=', naming the flag on error.
def split_assignment(arg, flag):
    if "=" not in arg:
        raise DatasetError(f"{flag} expects NAME=VALUE, got '{arg}'; include '=' between the "
                           "column name and value, and quote the complete argument if it contains spaces")
    name, value = arg.split("=", 1)
    if not name:
        raise DatasetError(f"{flag} expects NAME=VALUE, got '{arg}'; put a nonempty column name before '='")
    return name, value


# Applies operations in the fixed order add, remove, rename, swap. Each list
# holds (name, value) pairs already split from the flag argument. A remove,
# rename, or swap naming an absent column is an error naming the row.
def apply(rows, adds=(), removes=(), renames=(), swaps=(), *, progress=None):
    with stage(progress, "applying field operations", total=len(rows), unit="rows") as activity:
        compiled = [(name, expr, compile_expression(expr)) for name, expr in adds]
        out = []
        for i, row in enumerate(rows):
            row = dict(row)
            for name, expr, code in compiled:
                row[name] = evaluate(code, expr, row, i)
            for name in removes:
                if name not in row:
                    raise DatasetError(f"--remove {name}: row {i} has no column '{name}'")
                del row[name]
            for old, new in renames:
                if old not in row:
                    raise DatasetError(f"--rename {old}={new}: row {i} has no column '{old}'")
                row[new] = row.pop(old)
            for a, b in swaps:
                for name in (a, b):
                    if name not in row:
                        raise DatasetError(f"--swap {a}={b}: row {i} has no column '{name}'")
                row[a], row[b] = row[b], row[a]
            out.append(row)
            activity.advance()
        return out
