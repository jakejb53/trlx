"""argparse tree and dispatch for the dataset executable.

Every handler reads inputs through dataset.io, calls one module function,
and writes through dataset.io. Failures share structured context; unexpected
exceptions also print their traceback because this CLI has no persistent log.

This package imports nothing from trl or trlx; tests/test_imports.py enforces it.
"""

import argparse
import os
import sys

from dataset import chat, context_builder, convert, cpt, env, eval_build, fields, heal, pairs, rows, stats
from dataset.endpoint import Endpoint
from dataset.failures import capture, render
from dataset.io import DatasetError, read_rows, validate_rows_output, validate_rows_outputs, write_many_rows, write_rows
from dataset.progress import Progress, stage
from dataset.prompts import load as load_prompt

# Shared dataset endpoint defaults are an explicit exception recorded in PLAN.md.
# Endpoint itself still receives resolved values; trlx callers are unaffected.
ENDPOINT_CONCURRENCY = 4
ENDPOINT_TIMEOUT = 120
ENDPOINT_RETRIES = 2


# File format follows --out's extension; --to additionally reshapes rows.
def _cmd_convert(args):
    data = read_rows(args.input, progress=args.progress)
    if args.to:
        data = convert.convert(data, args.to, progress=args.progress)
    write_rows(args.out, data, [args.input], force=args.force, no_staging=args.no_staging, progress=args.progress)
    return 0


# Seeded reorder.
def _cmd_shuffle(args):
    data = read_rows(args.input, progress=args.progress)
    write_rows(args.out, rows.shuffle(data, args.seed, progress=args.progress), [args.input],
               force=args.force, no_staging=args.no_staging, progress=args.progress)
    return 0


# Validate both destinations before reading inputs; prepare both before publication.
def _cmd_split(args):
    first_path = validate_rows_output(args.out, force=args.force)
    rest_path = validate_rows_output(args.rest, force=args.force)
    if first_path == rest_path:
        raise DatasetError("--out and --rest name the same destination; choose two distinct output paths")
    data = read_rows(args.input, progress=args.progress)
    first, rest = rows.split(data, args.n, args.fraction, args.key, progress=args.progress)
    write_many_rows([(args.out, first), (args.rest, rest)], [args.input],
                    force=args.force, no_staging=args.no_staging, progress=args.progress)
    print(f"{len(first)} rows to {args.out}, {len(rest)} rows to {args.rest}")
    return 0


# Several inputs, one fraction each, positionally matched.
def _cmd_mix(args):
    try:
        fractions = [float(f) for f in args.fractions.split(",")]
    except ValueError:
        raise DatasetError(f"--fractions must be comma-separated numbers in 0..1, got '{args.fractions}'; "
                           "use one fraction per input, e.g. --fractions 1,0.2 for two input files")
    if len(fractions) != len(args.input):
        raise DatasetError(f"--fractions has {len(fractions)} values for {len(args.input)} inputs; "
                           "supply exactly one fraction per input file, in input order; fractions need not sum to 1")
    sources = [(read_rows(p, progress=args.progress), f) for p, f in zip(args.input, fractions)]
    write_rows(args.out, rows.mix(sources, args.seed, progress=args.progress), args.input,
               force=args.force, no_staging=args.no_staging, progress=args.progress)
    return 0


# Splits NAME=VALUE flags here; fields.apply owns the per-row semantics.
def _cmd_fields(args):
    adds = [fields.split_assignment(a, "--add") for a in args.add]
    renames = [fields.split_assignment(a, "--rename") for a in args.rename]
    swaps = [fields.split_assignment(a, "--swap") for a in args.swap]
    data = read_rows(args.input, progress=args.progress)
    data = fields.apply(data, adds, args.remove, renames, swaps, progress=args.progress)
    write_rows(args.out, data, [args.input], force=args.force, no_staging=args.no_staging, progress=args.progress)
    return 0


# Parses COLUMN=N limits here; rows.filter_rows applies them with --where.
def _cmd_filter(args):
    max_lengths = []
    for arg in args.max_length:
        column, limit = fields.split_assignment(arg, "--max-length")
        try:
            max_lengths.append((column, int(limit)))
        except ValueError:
            raise DatasetError(f"--max-length expects COLUMN=INT, got '{arg}'; "
                               "use e.g. --max-length text=8000 (characters) or --max-length messages=10 (items)")
    data = read_rows(args.input, progress=args.progress)
    kept = rows.filter_rows(data, args.where, max_lengths, progress=args.progress)
    write_rows(args.out, kept, [args.input], force=args.force, no_staging=args.no_staging, progress=args.progress)
    print(f"kept {len(kept)} of {len(data)} rows")
    return 0


# Random with --seed or first N with --head; rows.sample enforces exactly one.
def _cmd_sample(args):
    data = read_rows(args.input, progress=args.progress)
    write_rows(args.out, rows.sample(data, args.n, args.seed, args.head, progress=args.progress), [args.input],
               force=args.force, no_staging=args.no_staging, progress=args.progress)
    return 0


# Input is a plain text file, not a dataset, so it bypasses read_rows.
def _cmd_cpt(args):
    data = cpt.cpt_rows(args.input, args.max_tokens, progress=args.progress)
    write_rows(args.out, data, [args.input], force=args.force, no_staging=args.no_staging, progress=args.progress)
    print(f"{len(data)} chunks")
    return 0


# One input: distillation split. Two inputs: chosen then rejected alignment.
# Unmatched rows go to stderr so stdout stays a clean summary.
def _cmd_pairs(args):
    if len(args.input) == 1:
        data = read_rows(args.input[0], progress=args.progress)
        write_rows(args.out, pairs.single(data, progress=args.progress), args.input,
                   force=args.force, no_staging=args.no_staging, progress=args.progress)
        return 0
    if len(args.input) != 2:
        raise DatasetError("dataset pairs takes one or two input files; use one messages dataset for "
                           "prompt/completion rows, or two files in chosen-then-rejected order for preference pairs")
    chosen = read_rows(args.input[0], progress=args.progress)
    rejected = read_rows(args.input[1], progress=args.progress)
    paired, unmatched = pairs.align(chosen, rejected, progress=args.progress)
    for u in unmatched:
        print(f"unmatched: {u}", file=sys.stderr)
    if unmatched and args.strict:
        raise DatasetError(f"{len(unmatched)} unmatched rows with --strict; no output was written. "
                           "Align the user-turn contents in both inputs, or remove --strict to omit unmatched rows")
    write_rows(args.out, paired, args.input, force=args.force, no_staging=args.no_staging, progress=args.progress)
    print(f"{len(paired)} pairs, {len(unmatched)} unmatched")
    return 0


# Always writes the output; exit 1 when any unit was left unrepaired.
def _cmd_heal(args):
    repairs, errors = heal.heal_file(args.input, args.out, force=args.force,
                                   no_staging=args.no_staging, progress=args.progress)
    for r in repairs:
        print(f"repaired {r}")
    for e in errors:
        print(f"error {e}", file=sys.stderr)
    print(f"{len(repairs)} repairs, {len(errors)} errors")
    return 1 if errors else 0


# Resolves the answers pass to its own endpoint or a copy of the questions
# pass. Exactly one of the two answers flags is an error, never a partial copy.
def _cmd_chat(args):
    if not 0 <= args.eval_n < args.n:
        raise DatasetError(f"--n {args.n} must be a positive question count per source chunk, "
                           f"and --eval-n {args.eval_n} must satisfy 0 <= --eval-n < --n. "
                           "Use --eval-n 0 without --eval-out for training output only, "
                           "or reserve fewer questions than --n for evaluation.")
    if bool(args.eval_n) != bool(args.eval_out):
        raise DatasetError("positive --eval-n and --eval-out must be supplied together; "
                           "to reserve evaluation rows, set --eval-n below --n and --eval-out eval.jsonl; "
                           "for training output only, omit --eval-out and use --eval-n 0")
    # Check both paths, including aliases, before spending any endpoint requests.
    validate_rows_outputs([args.out, args.eval_out] if args.eval_out else [args.out], args.force)
    if (args.answers_endpoint is None) != (args.answers_model is None):
        raise DatasetError("answers pass needs both --answers-endpoint and --answers-model; "
                           "supply the missing flag, or omit both to reuse --questions-endpoint and --questions-model")
    api_key = None
    if args.api_key:
        api_key = os.environ.get(args.api_key)
        if not api_key:
            raise DatasetError("--api-key names an environment variable, not the key itself; "
                               "the selected variable is unset or empty. Export the key in that variable "
                               "and pass its name, or omit --api-key if authentication is not required")
    q_url, q_model = args.questions_endpoint, args.questions_model
    a_url = args.answers_endpoint if args.answers_endpoint else q_url
    a_model = args.answers_model if args.answers_model else q_model
    q_ep = Endpoint(q_url, q_model, api_key, args.timeout, args.retries)
    a_ep = Endpoint(a_url, a_model, api_key, args.timeout, args.retries)
    # Endpoint owns credential-safe URL display as well as request construction.
    print(f"questions: {q_model} at {q_ep.display_url}")
    print(f"answers:   {a_model} at {a_ep.display_url}")
    try:
        with stage(args.progress, f"reading {args.input}"), open(args.input, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise DatasetError(f"{args.input}: cannot read: {e.strerror or e}; check the path and permissions")
    except UnicodeError:
        raise DatasetError(f"{args.input}: input is not valid UTF-8; convert the text to UTF-8")
    data, evaluation, skipped = chat.build(
        text, args.max_tokens, args.n, q_ep, a_ep,
        load_prompt(args.questions_prompt, required=("n", "chunk"), allowed=("n", "chunk")),
        load_prompt(args.answers_prompt, required=("chunk", "question"), allowed=("chunk", "question")),
        args.concurrency, args.strip_reasoning_tags, exclude_reasoning=args.exclude_reasoning,
        eval_n=args.eval_n, progress=args.progress,
    )
    for s in skipped:
        print(f"skipped {s}", file=sys.stderr)
    if args.eval_out:
        write_many_rows([(args.out, data), (args.eval_out, evaluation)], [args.input],
                        force=args.force, no_staging=args.no_staging, progress=args.progress)
        print(f"{len(data)} training rows to {args.out}, {len(evaluation)} evaluation rows to {args.eval_out}, "
              f"{len(skipped)} skipped")
    else:
        write_rows(args.out, data, [args.input], force=args.force, no_staging=args.no_staging, progress=args.progress)
        print(f"{len(data)} rows, {len(skipped)} skipped")
    return 0


# All generation and validation finish before publication, even with --no-staging.
def _cmd_eval_build(args):
    if not args.model.strip():
        raise DatasetError("--model must name the model served by the endpoint")
    api_key = None
    if args.api_key:
        api_key = os.environ.get(args.api_key)
        if not api_key:
            raise DatasetError("--api-key names an environment variable, not the key itself; "
                               "the selected variable is unset or empty. Export the key in that variable "
                               "and pass its name, or omit --api-key if authentication is not required")
    endpoint = Endpoint(args.endpoint, args.model, api_key, args.timeout, args.retries)
    summary_prompt = load_prompt(args.summary_prompt, allowed=None)
    data = eval_build.build(
        read_rows(args.input, progress=args.progress), endpoint, args.max_tokens,
        args.concurrency, args.strip_reasoning_tags, summary_prompt=summary_prompt, progress=args.progress,
    )
    write_rows(args.out, data, [args.input], force=args.force,
               no_staging=args.no_staging, progress=args.progress)
    print(f"{len(data)} summaries written to {args.out}", file=sys.stderr)
    return 0


# Report only; no output file.
def _cmd_stats(args):
    columns =args.columns.split(",") if args.columns else None
    stats.run(read_rows(args.input, progress=args.progress), columns, args.model, progress=args.progress)
    return 0


# Write one machine result without leaking Context content through routine summaries.
def _context_result(value):
    import json

    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


# Read exact message/result content from an explicit value, file, or stdin.
def _context_content(args):
    if args.text is not None:
        return args.text
    source = args.content_file
    if source is None or source == "-":
        try:
            return sys.stdin.read()
        except UnicodeError:
            raise DatasetError("stdin: Context content must be valid UTF-8 text") from None
    try:
        with open(source, encoding="utf-8") as stream:
            return stream.read()
    except (OSError, UnicodeError) as error:
        raise DatasetError(f"{source}: cannot read Context content as UTF-8 text: {error}") from None


# Create one new durable Context array without replacement semantics.
def _cmd_context_create(args):
    return _context_result(context_builder.create_context(args.path, progress=args.progress))


# Insert either a simple role/content message or one exact raw provider message.
def _cmd_context_add(args):
    if args.message_file is not None:
        if args.text is not None or args.content_file is not None:
            raise DatasetError("--message-file cannot be combined with --text or --content-file")
        message = context_builder.read_message_file(args.message_file)
    else:
        if not args.role:
            raise DatasetError("--role: expected a nonempty role")
        message = {"role": args.role, "content": _context_content(args)}
    return _context_result(context_builder.add_message(
        args.path, message, at=args.at, progress=args.progress))


# Parse the convenient flat-string form without inventing types or nested syntax.
def _context_arguments(args):
    if args.arguments_file is not None:
        return context_builder.read_arguments_file(args.arguments_file)
    result = {}
    for assignment in args.arg:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise DatasetError(f"--arg expects nonempty KEY=VALUE, got {assignment!r}")
        if key in result:
            raise DatasetError(f"--arg repeats key {key!r}; supply each tool argument once")
        result[key] = value
    return result


# Insert a fabricated function call and the supplied result as one atomic edit.
def _cmd_context_tool(args):
    return _context_result(context_builder.add_tool_exchange(
        args.path, args.name, _context_arguments(args), _context_content(args),
        call_id=args.call_id, at=args.at, progress=args.progress))


# Replace one complete message or only its content while preserving other fields.
def _cmd_context_replace(args):
    if args.message_file is not None:
        if args.text is not None or args.content_file is not None:
            raise DatasetError("--message-file cannot be combined with --text or --content-file")
        result = context_builder.replace_message(
            args.path, args.index, message=context_builder.read_message_file(args.message_file),
            progress=args.progress)
    else:
        result = context_builder.replace_message(
            args.path, args.index, content=_context_content(args), progress=args.progress)
    return _context_result(result)


# Remove one explicit contiguous range.
def _cmd_context_remove(args):
    return _context_result(context_builder.remove_messages(
        args.path, args.index, args.count, progress=args.progress))


# Move one explicit contiguous range using post-removal destination coordinates.
def _cmd_context_move(args):
    return _context_result(context_builder.move_messages(
        args.path, args.source, args.destination, args.count, progress=args.progress))


# Keep the default outline content-free while offering an exact JSON form for agents.
def _cmd_context_outline(args):
    rows = context_builder.outline_context(args.path, args.preview)
    if args.json:
        return _context_result(rows)
    columns = ["index", "role", "content", "tool calls", "tool result"]
    if args.preview is not None:
        columns.append("preview")
    print("\t".join(columns))
    for row in rows:
        content = (f"{row['content_characters']} chars" if row["content_type"] == "string"
                   else row["content_type"])
        calls = ",".join(f"{call['id']}:{call['name']}" for call in row["tool_calls"])
        values = [str(row["index"]), str(row["role"]), content, calls,
                  str(row["tool_call_id"] or "")]
        if args.preview is not None:
            values.append(row.get("preview", ""))
        print("\t".join(values))
    return 0


# Print only the requested exact message region for focused inspection.
def _cmd_context_show(args):
    import json

    value = context_builder.show_messages(args.path, args.index, args.count)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


# Validate without rewriting the Context and return only compact counts.
def _cmd_context_validate(args):
    messages = context_builder.read_context(args.path)
    return _context_result(context_builder.validate_context(messages, str(args.path)))


# Read one stateless request; the browser alone accepts inline Context and literal credentials.
def _cmd_generate(args):
    import json
    from dataset import authoring

    value = authoring.read_json(sys.stdin, "stdin")
    if isinstance(value, dict) and "context" in value:
        raise DatasetError("stdin: context is supplied only through optional --context-file PATH")
    body = authoring.validate(authoring.Generation, value)
    if args.context_file is not None:
        try:
            with open(args.context_file, encoding="utf-8") as stream:
                context = authoring.read_json(stream, args.context_file)
        except (OSError, ValueError) as error:
            raise DatasetError(f"{args.context_file}: cannot open Context file: {error}") from None
        if not isinstance(context, list) or any(not isinstance(message, dict) for message in context):
            raise DatasetError(f"{args.context_file}: Context must be a JSON array of message objects")
        body.context = context
    # Resolve only after type validation; neither request values nor secrets enter diagnostics.
    if body.api_key:
        if "=" in body.api_key or "\0" in body.api_key:
            raise DatasetError("api_key must name an environment variable, not contain a credential")
        credential = os.environ.get(body.api_key)
        if not credential:
            raise DatasetError("api_key environment variable is unset or empty; set it or omit api_key")
        body.api_key = credential
    result = authoring.generate(body, progress=args.progress)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


# Explicit saves append selected rows; no CLI-owned pending collection or output replacement exists.
def _cmd_save(args):
    import json
    from dataset import authoring

    body = authoring.validate(authoring.Save, authoring.read_json(sys.stdin, "stdin"))
    result = authoring.save(body, progress=args.progress)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


# Import web dependencies only when starting the UI, never for other commands or help.
def _cmd_ui(args):
    from dataset.ui import run

    return run(args.host, args.port)


# Builds the full parser. Kept separate from main so tests can inspect the tree
# without invoking anything.
def build_parser():
    parser = argparse.ArgumentParser(
        prog="dataset",
        description=(
            "Prepare training data without Python: convert, inspect, split, edit,\n"
            "or generate rows. No training config or GPU is needed for preparation.\n\n"
            "Files: .jsonl (one object per line), .json (array of objects),\n"
            ".csv (header row; cells read as strings), or .parquet.\n"
            "cpt/chat read plain text; heal accepts JSON/JSONL only.\n"
            "Replacing an existing output or input requires --force.\n"
            "Replacement acts on symlinks themselves; heal repairs their targets.\n"
            "Progress goes to stderr, with waiting notices after 10 seconds without feedback."
        ),
        epilog=(
            "Start here:\n"
            "  dataset stats data.jsonl\n"
            "  dataset convert data.json --out data.jsonl\n"
            "  dataset shuffle data.jsonl --seed 42 --out shuffled.jsonl\n\n"
            "Run dataset COMMAND --help for inputs, options, and examples.\n"
            "Help never reads datasets, loads models, or contacts endpoints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", title="commands", metavar="COMMAND", required=True)

    # Keep format and overwrite rules beside each command so subcommand help
    # is useful on its own. Handler dispatch and argument semantics stay shared.
    def add(name, help_text, func, description, examples, inputs=1, out=True, input_help=None):
        p = sub.add_parser(
            name, help=help_text, description=description,
            epilog="Examples:\n" + examples,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        input_help = input_help or "dataset file: .jsonl, .json, .csv, or .parquet"
        if inputs == 1:
            p.add_argument("input", help=input_help)
        else:
            p.add_argument("input", nargs="+", help=input_help)
        p.add_argument("--force", action="store_true", help=(
            "authorize replacing existing outputs, including inputs; no additional confirmation"
            if out else "accepted for consistency; stats has no destructive output operation"
        ))
        if out:
            p.add_argument("--out", required=True, metavar="FILE", help=(
                "output file: .jsonl, .json, .csv, or .parquet; existing paths require --force"
                if name != "heal" else
                "same extension as input; existing paths require --force; symlink targets are repaired"
            ))
            p.add_argument("--no-staging", action="store_true", help=(
                "write directly without preparing a temporary output; failure may leave partial output; "
                "does not imply --force"
            ))
        p.set_defaults(func=func)
        return p

    p = add("convert", "change file format or messages/prompt-completion shape", _cmd_convert,
            "Change file format using the output extension; optionally reshape rows.\n"
            "messages holds {role, content} objects and must end with an assistant.\n"
            "prompt and completion must both be strings or both be message lists.\n"
            "Other columns are preserved. CSV output cannot contain lists or objects.",
            "  dataset convert data.json --out data.jsonl\n"
            "  dataset convert chat.jsonl --to prompt-completion --out prompts.jsonl\n"
            "  dataset convert prompts.jsonl --to messages --out chat.parquet")
    p.add_argument("--to", choices=convert.TARGETS,
                   help="target row shape; default: preserve columns; source must be the opposite shape")

    p = add("shuffle", "reorder all rows reproducibly", _cmd_shuffle,
            "Reorder all rows using an explicit random seed. Columns are unchanged.",
            "  dataset shuffle data.jsonl --seed 42 --out shuffled.jsonl")
    p.add_argument("--seed", type=int, required=True, help="integer random seed; same input and seed reproduce order")

    p = add("split", "divide rows into two files, optionally keeping groups together", _cmd_split,
            "Give exactly one of --n or --fraction. Takes rows in input order;\n"
            "shuffle first for a random split. The remainder goes to --rest.\n"
            "With --key, whole groups are kept together in first-occurrence order;\n"
            "the first output may exceed the requested count.",
            "  dataset split data.jsonl --fraction 0.9 --out train.jsonl --rest eval.jsonl\n"
            "  dataset split data.jsonl --n 100 --key source --out first.jsonl --rest rest.jsonl")
    p.add_argument("--n", type=int, metavar="ROWS", help="integer rows to --out, from 0 to input count; excludes --fraction")
    p.add_argument("--fraction", type=float, metavar="FRACTION",
                   help="share of input to --out, 0..1; count = round(rows * fraction); excludes --n")
    p.add_argument("--rest", required=True, metavar="FILE",
                   help="remainder file; format by extension; must differ from --out; existing paths require --force")
    p.add_argument("--key", metavar="COLUMN",
                   help="group by equal scalar column values; default: split individual rows")

    p = add("mix", "sample a fraction of each input and concatenate", _cmd_mix,
            "Fractions select a share of EACH input, not a share of the final mix.\n"
            "Each count is round(source rows * fraction); fractions need not sum to 1.\n"
            "Samples preserve source order and are concatenated in input order.\n"
            "Run shuffle afterward if the final mix should be randomized.",
            "  dataset mix primary.jsonl replay.jsonl --fractions 1,0.2 --seed 42 --out mixed.jsonl",
            inputs="+")
    p.add_argument("--fractions", required=True, metavar="F1,F2,...",
                   help="comma-separated numbers in 0..1; exactly one per input, in the same order")
    p.add_argument("--seed", type=int, required=True, help="integer random seed for reproducible source sampling")

    expressions = (
        "Expressions are Python: columns are names; row['column-name'] accesses\n"
        "any column. Available functions: len, str, int, float. Quote expressions\n"
        "in the shell to preserve Python strings and operators."
    )
    p = add("fields", "add, remove, rename, or swap columns", _cmd_fields,
            "Operations run in this order: add, remove, rename, swap. Each flag may\n"
            "be repeated. With no operations, rows are copied unchanged.\n"
            "Add and rename may replace existing columns; missing sources are errors.\n\n" + expressions,
            "  dataset fields data.jsonl --add 'label=True' --out labeled.jsonl\n"
            "  dataset fields data.jsonl --add 'size=len(text)' --remove id --out sized.jsonl\n"
            "  dataset fields data.jsonl --rename answer=completion --out renamed.jsonl\n"
            "  dataset fields data.jsonl --swap chosen=rejected --out swapped.jsonl")
    p.add_argument("--add", action="append", default=[], metavar="NAME=EXPR",
                   help="set a column from a per-row Python expression; repeatable; default: none")
    p.add_argument("--remove", action="append", default=[], metavar="NAME",
                   help="delete an existing column; repeatable; default: none")
    p.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                   help="rename an existing column; repeatable; default: none")
    p.add_argument("--swap", action="append", default=[], metavar="A=B",
                   help="exchange two existing column values; repeatable; default: none")

    p = add("filter", "keep rows matching all length limits and expressions", _cmd_filter,
            "Keep a row only when ALL supplied conditions pass. With no conditions,\n"
            "all rows are retained. Length limits count characters or list items,\n"
            "not tokens.\n\n" + expressions,
            "  dataset filter data.jsonl --max-length text=8000 --out short.jsonl\n"
            "  dataset filter data.jsonl --where 'label == True' --out positive.jsonl")
    p.add_argument("--where", action="append", default=[], metavar="EXPR",
                   help="keep rows where the Python expression is truthy; repeatable; default: none")
    p.add_argument("--max-length", action="append", default=[], metavar="COLUMN=N",
                   help="inclusive integer limit: string characters or list items; repeatable; default: none")

    p = add("sample", "take a random sample or the first N rows", _cmd_sample,
            "Give exactly one of --seed or --head. Random sampling is without\n"
            "replacement and preserves input order among the selected rows.",
            "  dataset sample data.jsonl --n 100 --seed 42 --out sample.jsonl\n"
            "  dataset sample data.jsonl --n 10 --head --out preview.jsonl")
    p.add_argument("--n", type=int, required=True, metavar="ROWS", help="integer rows to take, from 0 to input count")
    p.add_argument("--seed", type=int, help="integer random seed; required unless --head is used")
    p.add_argument("--head", action="store_true", help="take first N rows instead of sampling; excludes --seed")

    p = add("cpt", "chunk plain text into a text-column training dataset", _cmd_cpt,
            "Prepare continued-pretraining data: each row holds one text chunk.\n"
            "Keeps paragraphs together when possible; splits oversized text at\n"
            "sentences, then characters. Token limits are estimates, not tokenizer counts.",
            "  dataset cpt corpus.txt --max-tokens 2048 --out chunks.jsonl",
            input_help="UTF-8 plain text file")
    p.add_argument("--max-tokens", type=int, required=True, metavar="TOKENS",
                   help=f"positive integer chunk limit ({cpt.ESTIMATE_LABEL}); no tokenizer is loaded")

    p = add("pairs", "make preference pairs or prompt/completion rows", _cmd_pairs,
            "One input: split messages into prompt and final assistant completion.\n"
            "Two inputs: chosen first, rejected second; output prompt/chosen/rejected.\n"
            "Both inputs need messages lists of {role, content}, ending in assistant.\n"
            "Pairs match all user-turn contents, ignoring system turns; duplicate\n"
            "keys pair in order. Unmatched rows are reported and omitted by default.",
            "  dataset pairs answers.jsonl --out prompts.jsonl\n"
            "  dataset pairs chosen.jsonl rejected.jsonl --strict --out preference.jsonl",
            inputs="+")
    p.add_argument("--strict", action="store_true",
                   help="two-input mode: fail before writing if any row is unmatched; default: omit unmatched rows")

    add("heal", "repair common JSON/JSONL syntax errors", _cmd_heal,
        "Repair single quotes, Python literals, unquoted keys, trailing commas,\n"
        "unclosed brackets, and concatenated objects. A truncated final JSONL\n"
        "record may be dropped. Every repair is reported. Unrepairable text is\n"
        "preserved in the output and reported with exit status 1. Review the report.",
        "  dataset heal damaged.jsonl --out repaired.jsonl",
        input_help=".json or .jsonl file; output must use the same extension")

    p = add("chat", "generate question/answer messages from text via an endpoint", _cmd_chat,
            "Chunk text, generate questions, then answer them using their source chunk.\n"
            "Exact duplicate questions are always removed across chunks after list-marker\n"
            "and surrounding-whitespace cleanup, before answering. First occurrence wins;\n"
            "each removal is reported. Case, internal whitespace, and punctuation matter.\n"
            "Outputs messages plus a separate reasoning column unless --exclude-reasoning is set. Empty replies are\n"
            "reported and skipped. Uses an OpenAI-compatible /chat/completions API.\n"
            "Provide both answers endpoint/model flags, or neither to reuse questions.\n"
            "--max-tokens limits source chunks; completion length is controlled by\n"
            "the server. Configure its reasoning parser to keep reasoning separate.\n"
            "Credentials: --api-key names a variable from the environment or .env\n"
            "in the working directory; an exported value takes precedence.",
            "  dataset chat notes.txt --out chat.jsonl \\\n"
            "    --questions-endpoint http://localhost:8000/v1 --questions-model my-model \\\n"
            "    --n 3 --max-tokens 2048 --concurrency 4 --timeout 120 --retries 2\n"
            "  # Add --api-key API_KEY for authentication; pass the variable name, not its value.",
            input_help="UTF-8 plain text file")
    questions = p.add_argument_group("question generation")
    questions.add_argument("--questions-endpoint", required=True, metavar="URL",
                   help="API base URL (e.g. http://localhost:8000/v1); /chat/completions is appended")
    questions.add_argument("--questions-model", required=True, metavar="NAME",
                           help="question model name served by the API")
    questions.add_argument("--questions-prompt", metavar="FILE", default="prompts/chat-questions.prompt",
                   help="UTF-8 .prompt template requiring {n} (question count) and {chunk} (source text); "
                        "[[...]] includes its text when all enclosed placeholders have values; default: "
                        "./prompts/chat-questions.prompt relative to the working directory; missing files fail; "
                        "run trlx init to create defaults or supply a file; no built-in fallback")
    questions.add_argument("--n", type=int, required=True, metavar="QUESTIONS",
                           help="positive integer requested questions per source chunk; fewer may be returned")
    questions.add_argument("--eval-n", type=int, default=0, metavar="QUESTIONS",
                   help="reserve the last N retained questions per chunk for evaluation, after deduplication; "
                        "integer >= 0 and < --n; default: 0 (disabled); positive values require --eval-out; "
                        "short chunks reserve up to N; skipped answers reduce counts without reassignment")
    questions.add_argument("--eval-out", metavar="FILE",
                   help="evaluation output (.jsonl, .json, .csv, .parquet); requires positive --eval-n; "
                        "must differ from --out; existing output requires --force; both outputs stage by default")
    questions.add_argument("--max-tokens", type=int, required=True, metavar="TOKENS",
                           help=f"positive integer SOURCE chunk limit ({cpt.ESTIMATE_LABEL}); not a completion limit")
    answers = p.add_argument_group("answer generation")
    answers.add_argument("--answers-endpoint", metavar="URL",
                   help="answer API base URL; requires --answers-model; default: questions endpoint")
    answers.add_argument("--answers-model", metavar="NAME",
                   help="answer model name; requires --answers-endpoint; default: questions model")
    answers.add_argument("--answers-prompt", metavar="FILE", default="prompts/chat-answers.prompt",
                   help="UTF-8 .prompt template requiring {chunk} (source text) and {question} (generated question); "
                        "[[...]] includes its text when all enclosed placeholders have values; "
                        "default: ./prompts/chat-answers.prompt relative to the working directory; missing files "
                        "fail; run trlx init to create defaults or supply a file; no built-in fallback")
    requests = p.add_argument_group("requests and credentials (both passes)")
    requests.add_argument("--concurrency", type=int, default=ENDPOINT_CONCURRENCY, metavar="REQUESTS",
                   help=f"maximum simultaneous API requests per pass; integer >= 1; default: {ENDPOINT_CONCURRENCY}")
    requests.add_argument("--timeout", type=float, default=ENDPOINT_TIMEOUT, metavar="SECONDS",
                   help=f"finite positive timeout per API request, in seconds; default: {ENDPOINT_TIMEOUT}")
    requests.add_argument("--retries", type=int, default=ENDPOINT_RETRIES, metavar="COUNT",
                   help=f"integer retries after initial request, >= 0; transient failures and malformed responses; "
                        f"exponential backoff; successful requests are retained; default: {ENDPOINT_RETRIES}")
    requests.add_argument("--api-key", metavar="ENVVAR",
                   help="name of variable holding the key, shared by both endpoints; default: no Authorization header")
    requests.add_argument("--strip-reasoning-tags", action="store_true",
                   help="discard a leading inline reasoning block; default: fail; unclosed blocks always fail")
    requests.add_argument("--exclude-reasoning", action="store_true",
                   help="omit the separate reasoning column from output; default: retain it; "
                        "does not strip inline tags or disable model reasoning; use --strip-reasoning-tags independently")

    p = add("eval-build", "generate factual summaries for CPT evaluation", _cmd_eval_build,
            "Summarize every input text chunk through an OpenAI-compatible endpoint.\n"
            "Each input row must contain nonempty string text. Output contains one\n"
            '{"text": "generated summary"} row per input, in the same order.\n'
            "Instructions come from --summary-prompt, sent as the system message;\n"
            "each source text is sent separately as the user message.\n\n"
            "No sampling or re-chunking. Inputs and destination are validated before\n"
            "requests. Empty, malformed, or incomplete responses fail with the source\n"
            "row number; generation failures do not publish an incomplete dataset.\n"
            'The endpoint must report finish_reason="stop" for every summary.\n'
            "Separate endpoint reasoning is excluded from summaries.\n\n"
            "For best results, generate summaries using the same model you'll use this data set to train.\n\n"
            "Generate once and reuse for ordinary next-token-loss evaluation, including\n"
            "the step-zero baseline. Train on all original chunks and use these summaries\n"
            "as --dataset-eval. Progress and final outcome go to stderr.\n"
            "Credentials: --api-key names an environment variable, also loaded from\n"
            ".env in the working directory; exported values take precedence.",
            "  dataset eval-build train.jsonl --out eval.jsonl \\\n"
            "    --endpoint http://localhost:8000/v1 --model my-model --max-tokens 1024\n"
            "  dataset eval-build train.parquet --out eval.parquet \\\n"
            "    --endpoint https://api.example.com/v1 --model my-model \\\n"
            "    --max-tokens 1024 --api-key API_KEY --concurrency 8\n"
            "  trlx sft --model MODEL --no-split \\\n"
            "    --dataset train.jsonl --dataset-eval eval.jsonl \\\n"
            "    --eval-strategy steps --eval-steps 5")
    p.add_argument("--endpoint", required=True, metavar="URL",
                   help="API base URL, including /v1 when required; /chat/completions is appended")
    p.add_argument("--model", required=True, metavar="NAME",
                   help="model name served by the endpoint")
    p.add_argument("--summary-prompt", metavar="FILE", default="prompts/eval-build-summary.prompt",
                   help="UTF-8 .prompt system instruction, no substitution; default: ./prompts/eval-build-summary.prompt "
                        "relative to the working directory; missing files fail; run trlx init to create defaults "
                        "or supply a file; no built-in fallback")
    p.add_argument("--max-tokens", type=int, required=True, metavar="TOKENS",
                   help="positive integer completion-token limit per summary; does not limit source chunk size")
    p.add_argument("--concurrency", type=int, default=ENDPOINT_CONCURRENCY, metavar="REQUESTS",
                   help=f"maximum simultaneous requests; integer >= 1; default: {ENDPOINT_CONCURRENCY}")
    p.add_argument("--timeout", type=float, default=ENDPOINT_TIMEOUT, metavar="SECONDS",
                   help=f"finite positive timeout per request, in seconds; default: {ENDPOINT_TIMEOUT}")
    p.add_argument("--retries", type=int, default=ENDPOINT_RETRIES, metavar="COUNT",
                   help=f"integer retries after initial request, >= 0; transient failures and malformed responses; "
                        f"exponential backoff; successful requests are retained; default: {ENDPOINT_RETRIES}")
    p.add_argument("--api-key", metavar="ENVVAR",
                   help="environment variable holding the API key; default: no Authorization header")
    p.add_argument("--strip-reasoning-tags", action="store_true",
                   help="discard a complete leading inline reasoning block; default: fail; unclosed blocks always fail")

    p = add("stats", "inspect token lengths; optionally score responses with a model", _cmd_stats,
            "Print count/min/mean/median/p90/max per column; no output file.\n"
            f"Without --model, lengths are estimates ({cpt.ESTIMATE_LABEL}).\n"
            "With --model, load weights on CUDA if available, otherwise CPU; report\n"
            "exact tokenizer counts and mean response log-probabilities in nats.\n"
            "Scores completion/chosen/rejected conditioned on prompt, and the final\n"
            "assistant turn in messages. --columns selects length columns only.",
            "  dataset stats data.jsonl\n"
            "  dataset stats pairs.jsonl --columns chosen,rejected --model /models/base",
            out=False)
    p.add_argument("--columns", metavar="NAME,NAME,...",
                   help="comma-separated length columns; default: text/message columns found in first row")
    p.add_argument("--model", metavar="MODEL",
                   help="local model path or Hub ID for exact counts and response scoring; default: estimates only")

    p = sub.add_parser("ui", help="author datasets interactively in a web browser",
        description="Compare editable endpoint responses and save selected examples to JSONL.\n"
                    "Prompts, endpoint settings, credentials, and pending examples persist in browser localStorage.\n"
                    "Save appends unique examples to a path on the server; existing rows are preserved.",
        epilog="Examples:\n  dataset ui --host 127.0.0.1 --port 8000",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="listen address chosen by the operator; no default")
    p.add_argument("--port", required=True, type=int, help="listen port, integer 1..65535; no default")
    p.add_argument("--force", action="store_true", help="accepted for consistency; UI saves preserve existing rows")
    p.set_defaults(func=_cmd_ui)

    p = sub.add_parser("context", help="create and edit reusable generation Context arrays",
        description=(
            "Edit the exact JSON messages array consumed by dataset generate --context-file.\n"
            "Content may come from --text, --content-file, or stdin independently of\n"
            "whether it is represented as an ordinary message or fabricated tool result.\n"
            "Mutations validate and stage the complete replacement; serialize writers to one file."
        ),
        epilog="Run dataset context ACTION --help for action-specific examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = p.add_subparsers(dest="context_action", title="actions", metavar="ACTION", required=True,
                               help="Context action; run dataset context ACTION --help for details")

    # Every content-bearing action uses the same mutually exclusive ingestion choices.
    def content_options(parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--text", help="literal content; default: read complete UTF-8 content from stdin")
        group.add_argument("--content-file", metavar="PATH",
                           help="read exact UTF-8 content from PATH; '-' reads stdin")

    p = actions.add_parser("create", help="create a new empty Context array")
    p.add_argument("path", help="new Context JSON path; parent must exist; path must be absent")
    p.set_defaults(func=_cmd_context_create)

    p = actions.add_parser("add", help="insert an ordinary or raw message")
    p.add_argument("path", help="existing Context JSON array")
    representation = p.add_mutually_exclusive_group(required=True)
    representation.add_argument("--role", help="role for a generated {role, content} message")
    representation.add_argument("--message-file", metavar="PATH",
                                help="UTF-8 JSON file containing one exact message object")
    p.add_argument("--at", type=int, metavar="INDEX", help="zero-based insertion index; default: append")
    content_options(p)
    p.set_defaults(func=_cmd_context_add)

    p = actions.add_parser("tool", help="insert a fabricated function call and supplied result")
    p.add_argument("path", help="existing Context JSON array")
    p.add_argument("--name", required=True, help="nonempty represented function name")
    arguments = p.add_mutually_exclusive_group()
    arguments.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE",
                           help="flat string argument; repeatable; default: empty object")
    arguments.add_argument("--arguments-file", metavar="PATH",
                           help="UTF-8 JSON file containing one argument object")
    p.add_argument("--call-id", help="explicit unique tool-call ID; default: next call_NNNN")
    p.add_argument("--at", type=int, metavar="INDEX", help="zero-based insertion index; default: append")
    content_options(p)
    p.set_defaults(func=_cmd_context_tool)

    p = actions.add_parser("replace", help="replace one message or only its content")
    p.add_argument("path", help="existing Context JSON array")
    p.add_argument("index", type=int, help="zero-based message index")
    p.add_argument("--message-file", metavar="PATH", help="replace with one exact JSON message object")
    content_options(p)
    p.set_defaults(func=_cmd_context_replace)

    p = actions.add_parser("remove", help="remove one contiguous message range")
    p.add_argument("path", help="existing Context JSON array")
    p.add_argument("index", type=int, help="zero-based first message index")
    p.add_argument("--count", type=int, default=1, help="positive messages to remove; default: 1")
    p.set_defaults(func=_cmd_context_remove)

    p = actions.add_parser("move", help="move one contiguous message range")
    p.add_argument("path", help="existing Context JSON array")
    p.add_argument("source", type=int, help="zero-based first source index")
    p.add_argument("destination", type=int,
                   help="zero-based insertion index after removing the source range")
    p.add_argument("--count", type=int, default=1, help="positive messages to move; default: 1")
    p.set_defaults(func=_cmd_context_move)

    p = actions.add_parser("outline", help="summarize messages without printing content")
    p.add_argument("path", help="existing Context JSON array")
    p.add_argument("--json", action="store_true", help="emit the outline as JSON")
    p.add_argument("--preview", type=int, metavar="CHARACTERS",
                   help="include at most this many escaped content characters; default: none")
    p.set_defaults(func=_cmd_context_outline)

    p = actions.add_parser("show", help="print one focused message range")
    p.add_argument("path", help="existing Context JSON array")
    p.add_argument("index", type=int, help="zero-based first message index")
    p.add_argument("--count", type=int, default=1, help="positive messages to show; default: 1")
    p.set_defaults(func=_cmd_context_show)

    p = actions.add_parser("validate", help="validate the complete Context and tool pairing")
    p.add_argument("path", help="existing Context JSON array")
    p.set_defaults(func=_cmd_context_validate)

    p = sub.add_parser("generate", help="generate one editable answer and reasoning from stdin JSON",
        description=(
            "Read one JSON object from stdin; return {answer, reasoning} JSON on stdout.\n"
            "Required: endpoint (HTTP(S) API base), model and user (nonblank strings),\n"
            "sampling (object), timeout (finite positive seconds), retries (integer >= 0).\n"
            "Optional: system (string, default empty), api_key (environment-variable name;\n"
            "omitted or empty means no authentication). Loads secrets from .env.\n"
            "sampling accepts temperature >= 0, top_p in [0,1], integer top_k >= -1,\n"
            "positive integer max_tokens, finite presence_penalty, repetition_penalty > 0.\n"
            "Omitted/null controls are not sent; {} sends no sampling overrides.\n"
            "Context comes only from optional --context-file, never stdin. Messages are\n"
            "system, Context entries unchanged, then user. Generate writes no dataset.\n"
            "Progress/errors go to stderr; failures exit nonzero. No web server is needed."
        ),
        epilog=('Examples:\n  dataset generate --context-file context.json\n'
                '  stdin: {"endpoint":"http://localhost:8000/v1","model":"served-model",\n'
                '          "user":"Explain this.","sampling":{},"timeout":120,"retries":2}\n'
                '  dataset generate --force\n  Supply the same stdin JSON without Context.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--context-file", metavar="PATH",
                   help="optional UTF-8 JSON array of message objects; relative to working directory; default: no Context")
    p.add_argument("--force", action="store_true", help="accepted for consistency; generation writes no dataset")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("save", help="append selected examples from stdin JSON to a dataset",
        description=(
            "Read {path, examples} from stdin; return {added, duplicates} JSON on stdout.\n"
            "path: nonempty .jsonl filename, relative to working directory or absolute;\n"
            "parent must exist; no directories or final symlinks. examples: nonempty array.\n"
            "Each example has messages: exactly one user then one assistant, with string\n"
            "content, plus optional string reasoning. No system or Context is saved.\n"
            "Answer only: omit reasoning. Reasoning only: use empty assistant content.\n"
            "Both: include assistant content and reasoning. Unknown fields are errors.\n"
            "Existing bytes are preserved; exact duplicates, including reasoning presence\n"
            "and text, are skipped. Publication is staged. No --force is required to append.\n"
            "Serialize saves to one destination, including browser and other writers.\n"
            "Progress/errors go to stderr; failures exit nonzero. No web server is needed."
        ),
        epilog=('Examples:\n  dataset save --force\n'
                '  stdin: {"path":"examples.jsonl","examples":[{"messages":[\n'
                '          {"role":"user","content":"Question"},\n'
                '          {"role":"assistant","content":"Edited answer"}]}]}'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="accepted for consistency; saves preserve existing rows")
    p.set_defaults(func=_cmd_save)
    return parser


# Entry point of the dataset console script and of `python -m dataset.cli`.
# Returns the process exit code.
def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        # A long-lived UI is idle between browser requests. Each API operation owns
        # its feedback; CLI environment credentials do not configure this workspace.
        if args.command == "ui":
            return args.func(args)
        with Progress(f"dataset {args.command}") as progress:
            args.progress = progress
            # Before any handler runs, so an --api-key variable can come from .env.
            with stage(progress, "validating inputs and loading credentials"):
                env.load()
                # Split and heal validate their coupled paths in their own handlers.
                if hasattr(args, "out") and args.command not in ("split", "heal"):
                    validate_rows_output(args.out, force=args.force)
            result = args.func(args)
            progress.finish("completed" if result == 0 else "failed")
            return result
    except Exception as e:
        report = capture(e, context={"command": f"dataset {args.command}"})
        print(f"dataset {args.command}: {render(report, include_traceback=not report['expected'])}", file=sys.stderr)
        return 1


# Invoked as `python -m dataset.cli` when the project is not installed.
if __name__ == "__main__":
    sys.exit(main())
