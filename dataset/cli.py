"""argparse tree and dispatch for the dataset executable.

Every handler reads inputs through dataset.io, calls one module function,
and writes through dataset.io. DatasetError is the one exception type that
reaches main(), which prints its message and exits 1; anything else is a bug
and surfaces as a traceback on purpose.

This package imports nothing from trl or trlx; tests/test_imports.py enforces it.
"""

import argparse
import os
import sys

from dataset import chat, convert, cpt, env, fields, heal, pairs, rows, stats
from dataset.endpoint import Endpoint
from dataset.io import DatasetError, read_rows, write_rows


# File format follows --out's extension; --to additionally reshapes rows.
def _cmd_convert(args):
    data = read_rows(args.input)
    if args.to:
        data = convert.convert(data, args.to)
    write_rows(args.out, data, [args.input])
    return 0


# Seeded reorder.
def _cmd_shuffle(args):
    write_rows(args.out, rows.shuffle(read_rows(args.input), args.seed), [args.input])
    return 0


# Two outputs; --rest is also protected from overwriting --out.
def _cmd_split(args):
    first, rest = rows.split(read_rows(args.input), args.n, args.fraction, args.key)
    write_rows(args.out, first, [args.input])
    write_rows(args.rest, rest, [args.input, args.out])
    print(f"{len(first)} rows to {args.out}, {len(rest)} rows to {args.rest}")
    return 0


# Several inputs, one fraction each, positionally matched.
def _cmd_mix(args):
    try:
        fractions = [float(f) for f in args.fractions.split(",")]
    except ValueError:
        raise DatasetError(f"--fractions must be comma-separated numbers, got '{args.fractions}'")
    if len(fractions) != len(args.input):
        raise DatasetError(f"--fractions has {len(fractions)} values for {len(args.input)} inputs")
    sources = [(read_rows(p), f) for p, f in zip(args.input, fractions)]
    write_rows(args.out, rows.mix(sources, args.seed), args.input)
    return 0


# Splits NAME=VALUE flags here; fields.apply owns the per-row semantics.
def _cmd_fields(args):
    adds = [fields.split_assignment(a, "--add") for a in args.add]
    renames = [fields.split_assignment(a, "--rename") for a in args.rename]
    swaps = [fields.split_assignment(a, "--swap") for a in args.swap]
    data = fields.apply(read_rows(args.input), adds, args.remove, renames, swaps)
    write_rows(args.out, data, [args.input])
    return 0


# Parses COLUMN=N limits here; rows.filter_rows applies them with --where.
def _cmd_filter(args):
    max_lengths = []
    for arg in args.max_length:
        column, limit = fields.split_assignment(arg, "--max-length")
        try:
            max_lengths.append((column, int(limit)))
        except ValueError:
            raise DatasetError(f"--max-length expects COLUMN=INT, got '{arg}'")
    data = read_rows(args.input)
    kept = rows.filter_rows(data, args.where, max_lengths)
    write_rows(args.out, kept, [args.input])
    print(f"kept {len(kept)} of {len(data)} rows")
    return 0


# Random with --seed or first N with --head; rows.sample enforces exactly one.
def _cmd_sample(args):
    write_rows(args.out, rows.sample(read_rows(args.input), args.n, args.seed, args.head), [args.input])
    return 0


# Input is a plain text file, not a dataset, so it bypasses read_rows.
def _cmd_cpt(args):
    data = cpt.cpt_rows(args.input, args.max_tokens)
    write_rows(args.out, data, [args.input])
    print(f"{len(data)} chunks")
    return 0


# One input: distillation split. Two inputs: chosen then rejected alignment.
# Unmatched rows go to stderr so stdout stays a clean summary.
def _cmd_pairs(args):
    if len(args.input) == 1:
        write_rows(args.out, pairs.single(read_rows(args.input[0])), args.input)
        return 0
    if len(args.input) != 2:
        raise DatasetError("pairs takes one messages dataset or two (chosen then rejected)")
    paired, unmatched = pairs.align(read_rows(args.input[0]), read_rows(args.input[1]))
    for u in unmatched:
        print(f"unmatched: {u}", file=sys.stderr)
    if unmatched and args.strict:
        raise DatasetError(f"{len(unmatched)} unmatched rows with --strict")
    write_rows(args.out, paired, args.input)
    print(f"{len(paired)} pairs, {len(unmatched)} unmatched")
    return 0


# Always writes the output; exit 1 when any unit was left unrepaired.
def _cmd_heal(args):
    repairs, errors = heal.heal_file(args.input, args.out)
    for r in repairs:
        print(f"repaired {r}")
    for e in errors:
        print(f"error {e}", file=sys.stderr)
    print(f"{len(repairs)} repairs, {len(errors)} errors")
    return 1 if errors else 0


# Resolves the answers pass to its own endpoint or a copy of the questions
# pass. Exactly one of the two answers flags is an error, never a partial copy.
def _cmd_chat(args):
    if (args.answers_endpoint is None) != (args.answers_model is None):
        raise DatasetError("answers pass needs both --answers-endpoint and --answers-model, or neither")
    api_key = None
    if args.api_key:
        api_key = os.environ.get(args.api_key)
        if not api_key:
            raise DatasetError(f"--api-key: environment variable {args.api_key} is not set")
    q_url, q_model = args.questions_endpoint, args.questions_model
    a_url = args.answers_endpoint if args.answers_endpoint else q_url
    a_model = args.answers_model if args.answers_model else q_model
    print(f"questions: {q_model} at {q_url}")
    print(f"answers:   {a_model} at {a_url}")
    q_ep = Endpoint(q_url, q_model, api_key, args.timeout, args.retries)
    a_ep = Endpoint(a_url, a_model, api_key, args.timeout, args.retries)
    try:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise DatasetError(f"{args.input}: {e.strerror}")
    data, skipped = chat.build(
        text, args.max_tokens, args.n, q_ep, a_ep,
        chat.load_prompt(args.questions_prompt, chat.QUESTIONS_PROMPT),
        chat.load_prompt(args.answers_prompt, chat.ANSWERS_PROMPT),
        args.concurrency, args.strip_reasoning_tags,
    )
    for s in skipped:
        print(f"skipped {s}", file=sys.stderr)
    write_rows(args.out, data, [args.input])
    print(f"{len(data)} rows, {len(skipped)} skipped")
    return 0


# Report only; no output file.
def _cmd_stats(args):
    columns =args.columns.split(",") if args.columns else None
    stats.run(read_rows(args.input), columns, args.model)
    return 0


# Builds the full parser. Kept separate from main so tests can inspect the tree
# without invoking anything.
def build_parser():
    parser = argparse.ArgumentParser(
        prog="dataset", description="Prepare datasets. Reads one input, writes one output."
    )
    sub = parser.add_subparsers(dest="command", metavar="<subcommand>", required=True)

    # Adds the shared input positional and --out. Subcommands with several
    # inputs or no output file pass their own values.
    def add(name, help_text, func, inputs=1, out=True):
        p = sub.add_parser(name, help=help_text)
        if inputs == 1:
            p.add_argument("input", help="input file")
        else:
            p.add_argument("input", nargs="+", help="input files")
        if out:
            p.add_argument("--out", required=True, help="output file; format by extension")
        p.set_defaults(func=func)
        return p

    p = add("convert", "between file formats and TRL dataset formats", _cmd_convert)
    p.add_argument("--to", choices=convert.TARGETS, help="TRL format to convert rows to")

    p = add("shuffle", "reorder rows with a seed", _cmd_shuffle)
    p.add_argument("--seed", type=int, required=True)

    p = add("split", "first N rows or a fraction to --out, remainder to --rest", _cmd_split)
    p.add_argument("--n", type=int, help="rows to the first output")
    p.add_argument("--fraction", type=float, help="fraction of rows to the first output")
    p.add_argument("--rest", required=True, help="output file for the remainder")
    p.add_argument("--key", help="column whose equal values must land on one side")

    p = add("mix", "concatenate several inputs with per-source fractions", _cmd_mix, inputs="+")
    p.add_argument("--fractions", required=True, help="comma-separated fraction per input")
    p.add_argument("--seed", type=int, required=True)

    p = add("fields", "add, remove, rename, or swap fields", _cmd_fields)
    p.add_argument("--add", action="append", default=[], metavar="NAME=EXPR")
    p.add_argument("--remove", action="append", default=[], metavar="NAME")
    p.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--swap", action="append", default=[], metavar="A=B")

    p = add("filter", "keep rows by length limit or expression", _cmd_filter)
    p.add_argument("--where", action="append", default=[], metavar="EXPR", help="Python expression over the row")
    p.add_argument("--max-length", action="append", default=[], metavar="COLUMN=N",
                   help="characters for strings, items for lists")

    p = add("sample", "take N rows, random with seed or head", _cmd_sample)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, help="random sample with this seed")
    p.add_argument("--head", action="store_true", help="first N rows")

    p = add("cpt", "text file to a text-field dataset in token-limited chunks", _cmd_cpt)
    p.add_argument("--max-tokens", type=int, required=True, help=f"chunk limit, {cpt.ESTIMATE_LABEL}")

    p = add("pairs", "two messages datasets (chosen, rejected) to preference pairs; "
            "one to prompt/completion", _cmd_pairs, inputs="+")
    p.add_argument("--strict", action="store_true", help="unmatched rows are fatal")

    add("heal", "deterministic JSON and JSONL repairs", _cmd_heal)

    p = add("chat", "text file to messages via an endpoint", _cmd_chat)
    p.add_argument("--questions-endpoint", required=True, metavar="URL")
    p.add_argument("--questions-model", required=True, metavar="NAME")
    p.add_argument("--questions-prompt", metavar="FILE", help="replaces the built-in instruction")
    p.add_argument("--answers-endpoint", metavar="URL", help="defaults to the questions endpoint")
    p.add_argument("--answers-model", metavar="NAME", help="defaults to the questions model")
    p.add_argument("--answers-prompt", metavar="FILE", help="replaces the built-in instruction")
    p.add_argument("--n", type=int, required=True, help="questions per chunk")
    p.add_argument("--max-tokens", type=int, required=True, help=f"chunk limit, {cpt.ESTIMATE_LABEL}")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--timeout", type=float, required=True, metavar="SECONDS")
    p.add_argument("--retries", type=int, required=True)
    p.add_argument("--api-key", metavar="ENVVAR", help="environment variable holding the key")
    p.add_argument("--strip-reasoning-tags", action="store_true",
                   help="remove an inline reasoning block instead of failing on it")

    p = add("stats", "token length distribution per column; --model adds log-prob", _cmd_stats, out=False)
    p.add_argument("--columns", help="comma-separated columns; default all text columns")
    p.add_argument("--model", help="model path; exact tokens and per-token log-prob")

    return parser


# Entry point of the dataset console script and of `python -m dataset.cli`.
# Returns the process exit code.
def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        # Before any handler runs, so an --api-key variable can come from .env.
        env.load()
        return args.func(args)
    except DatasetError as e:
        print(f"dataset {args.command}: {e}", file=sys.stderr)
        return 1


# Invoked as `python -m dataset.cli` when the project is not installed.
if __name__ == "__main__":
    sys.exit(main())
