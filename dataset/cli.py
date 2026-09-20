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
from dataset.io import DatasetError, read_rows, validate_rows_output, write_many_rows, write_rows
from dataset.progress import Progress, stage


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
        raise DatasetError(f"--fractions must be comma-separated numbers, got '{args.fractions}'")
    if len(fractions) != len(args.input):
        raise DatasetError(f"--fractions has {len(fractions)} values for {len(args.input)} inputs")
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
            raise DatasetError(f"--max-length expects COLUMN=INT, got '{arg}'")
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
        raise DatasetError("pairs takes one messages dataset or two (chosen then rejected)")
    chosen = read_rows(args.input[0], progress=args.progress)
    rejected = read_rows(args.input[1], progress=args.progress)
    paired, unmatched = pairs.align(chosen, rejected, progress=args.progress)
    for u in unmatched:
        print(f"unmatched: {u}", file=sys.stderr)
    if unmatched and args.strict:
        raise DatasetError(f"{len(unmatched)} unmatched rows with --strict")
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
    data, skipped = chat.build(
        text, args.max_tokens, args.n, q_ep, a_ep,
        chat.load_prompt(args.questions_prompt, chat.QUESTIONS_PROMPT),
        chat.load_prompt(args.answers_prompt, chat.ANSWERS_PROMPT),
        args.concurrency, args.strip_reasoning_tags, progress=args.progress,
    )
    for s in skipped:
        print(f"skipped {s}", file=sys.stderr)
    write_rows(args.out, data, [args.input], force=args.force, no_staging=args.no_staging, progress=args.progress)
    print(f"{len(data)} rows, {len(skipped)} skipped")
    return 0


# Report only; no output file.
def _cmd_stats(args):
    columns =args.columns.split(",") if args.columns else None
    stats.run(read_rows(args.input, progress=args.progress), columns, args.model, progress=args.progress)
    return 0


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
            "Outputs messages plus a separate reasoning column. Empty replies are\n"
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
    questions.add_argument("--questions-prompt", metavar="FILE",
                   help="UTF-8 instruction template with {n} and {chunk}; default: built-in question instruction")
    questions.add_argument("--n", type=int, required=True, metavar="QUESTIONS",
                           help="integer requested questions per source chunk; fewer may be returned")
    questions.add_argument("--max-tokens", type=int, required=True, metavar="TOKENS",
                           help=f"positive integer SOURCE chunk limit ({cpt.ESTIMATE_LABEL}); not a completion limit")
    answers = p.add_argument_group("answer generation")
    answers.add_argument("--answers-endpoint", metavar="URL",
                   help="answer API base URL; requires --answers-model; default: questions endpoint")
    answers.add_argument("--answers-model", metavar="NAME",
                   help="answer model name; requires --answers-endpoint; default: questions model")
    answers.add_argument("--answers-prompt", metavar="FILE",
                   help="UTF-8 template with {chunk} and {question}; default: built-in answer instruction")
    requests = p.add_argument_group("requests and credentials (both passes)")
    requests.add_argument("--concurrency", type=int, required=True, metavar="REQUESTS",
                   help="maximum simultaneous API requests per pass; integer >= 1")
    requests.add_argument("--timeout", type=float, required=True, metavar="SECONDS",
                   help="positive timeout per API request, in seconds")
    requests.add_argument("--retries", type=int, required=True, metavar="COUNT",
                   help="integer retries after initial request, >= 0; transient failures use exponential backoff")
    requests.add_argument("--api-key", metavar="ENVVAR",
                   help="name of variable holding the key, shared by both endpoints; default: no Authorization header")
    requests.add_argument("--strip-reasoning-tags", action="store_true",
                   help="discard a leading inline reasoning block; default: fail; unclosed blocks always fail")

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

    return parser


# Entry point of the dataset console script and of `python -m dataset.cli`.
# Returns the process exit code.
def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
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
    except DatasetError as e:
        print(f"dataset {args.command}: {e}", file=sys.stderr)
        return 1


# Invoked as `python -m dataset.cli` when the project is not installed.
if __name__ == "__main__":
    sys.exit(main())
