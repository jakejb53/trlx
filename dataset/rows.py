"""Row-level operations: shuffle, split, sample, mix, filter.

Every function takes and returns lists of dicts and never mutates its input.
Randomness always goes through an explicit seed so runs are reproducible.
"""

import random

from dataset.fields import compile_expression, evaluate
from dataset.io import DatasetError
from dataset.progress import stage


# Seeded Fisher-Yates over a copy; the same seed always yields the same order.
def shuffle(rows, seed, *, progress=None):
    with stage(progress, "shuffling rows"):
        out = list(rows)
        random.Random(seed).shuffle(out)
        return out


# Resolves --n or --fraction to a row count. Exactly one must be given.
def _count(total, n, fraction):
    if (n is None) == (fraction is None):
        raise DatasetError("give exactly one of --n and --fraction")
    if n is not None:
        if n < 0 or n > total:
            raise DatasetError(f"--n {n} is outside 0..{total} (the input has {total} rows)")
        return n
    if not 0.0 <= fraction <= 1.0:
        raise DatasetError(f"--fraction {fraction} is outside 0.0..1.0")
    return round(total * fraction)


# First `count` rows to the first side, remainder to the second, in file order.
# With `key`, rows are grouped by exact key value in first-occurrence order and
# whole groups go to the first side until it holds at least `count` rows, so
# equal keys never straddle the split. The first side may exceed `count`.
def split(rows, n=None, fraction=None, key=None, *, progress=None):
    with stage(progress, "splitting rows", total=len(rows), unit="rows") as activity:
        count = _count(len(rows), n, fraction)
        if key is None:
            first, rest = list(rows[:count]), list(rows[count:])
            activity.update(len(rows))
            return first, rest
        groups = {}
        for i, row in enumerate(rows):
            if key not in row:
                raise DatasetError(f"--key {key}: row {i} has no column '{key}'")
            try:
                groups.setdefault(row[key], []).append(row)
            except TypeError:
                # Lists and dicts cannot be dict keys; exact-match needs a scalar.
                raise DatasetError(
                    f"--key {key}: row {i} value is {type(row[key]).__name__}; the key column must hold scalars"
                )
        first, rest = [], []
        for group in groups.values():
            if len(first) < count:
                first.extend(group)
            else:
                rest.extend(group)
            activity.advance(len(group))
        return first, rest


# Random sample of n rows with a seed, or the first n with head=True.
# Random sampling keeps input order among the chosen rows.
def sample(rows, n, seed=None, head=False, *, progress=None):
    with stage(progress, "sampling rows"):
        if (seed is None) == (not head):
            raise DatasetError("give exactly one of --seed and --head")
        if n < 0 or n > len(rows):
            raise DatasetError(f"--n {n} is outside 0..{len(rows)} (the input has {len(rows)} rows)")
        if head:
            return list(rows[:n])
        chosen = sorted(random.Random(seed).sample(range(len(rows)), n))
        return [rows[i] for i in chosen]


# Concatenates sources in order, taking a seeded random fraction of each.
# `sources` is a list of (rows, fraction). One RNG is shared across sources so
# the whole mix is reproducible from one seed.
def mix(sources, seed, *, progress=None):
    with stage(progress, "mixing datasets", unit="sources") as activity:
        rng = random.Random(seed)
        out = []
        for i, (rows, fraction) in enumerate(sources):
            if not 0.0 <= fraction <= 1.0:
                raise DatasetError(f"fraction {fraction} for input {i} is outside 0.0..1.0")
            count = round(len(rows) * fraction)
            chosen = sorted(rng.sample(range(len(rows)), count))
            out.extend(rows[j] for j in chosen)
            activity.advance()
        return out


# Keeps rows for which every --where expression is truthy and every
# --max-length column is within its limit. Length is characters for strings
# and items for lists; other types are an error naming the row.
def filter_rows(rows, wheres=(), max_lengths=(), *, progress=None):
    with stage(progress, "filtering rows", total=len(rows), unit="rows") as activity:
        compiled = [(expr, compile_expression(expr)) for expr in wheres]
        out = []
        for i, row in enumerate(rows):
            keep = True
            for column, limit in max_lengths:
                if column not in row:
                    raise DatasetError(f"--max-length {column}={limit}: row {i} has no column '{column}'")
                value = row[column]
                if not isinstance(value, (str, list)):
                    raise DatasetError(
                        f"--max-length {column}={limit}: row {i} column is {type(value).__name__}, "
                        "expected a string or list"
                    )
                if len(value) > limit:
                    keep = False
                    break
            if keep:
                for expr, code in compiled:
                    if not evaluate(code, expr, row, i):
                        keep = False
                        break
            if keep:
                out.append(row)
            activity.advance()
        return out
