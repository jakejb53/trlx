"""Operator CLI: environment defaults, explicit task inputs, and per-run overrides."""

import argparse
import sys

from dataset.env import load as load_env
from dataset.io import DatasetError
from dataset.progress import Progress, stage
from trlx import TrlxError
from trlx.options import HelpFormatter, add_run_settings, add_training_options, overrides, removed_verification_prompts

# Stable TRL trainers only. Order is the order shown in --help.
METHODS = ["sft", "dpo", "grpo", "kto", "rloo", "reward", "distillation"]

STRATEGIES = ["ddp", "fsdp"]


# Init needs only hardware metadata; task inputs belong to the training command.
def _cmd_init(args):
    from trlx import init_cmd

    system = init_cmd.write(args.out, force=args.force, no_staging=args.no_staging, progress=args.progress)
    print(f"wrote {args.out} for {len(system.gpus)} visible GPU(s), {system.cpu_count} logical processor(s)")
    return 0


# Merge uses the model's own architecture and validates the loaded adapter.
def _cmd_merge(args):
    from trlx import merge

    merge.merge(args.base, args.adapter, args.out, force=args.force,
                no_staging=args.no_staging, progress=args.progress)
    return 0


# Method subcommands share one handler; the method is args.command.
def _cmd_train(args):
    from trlx import train

    args.overrides = overrides(args)
    return train.run(args)


# Both views read the persisted run artifacts rather than rebuilding a trainer.
def _cmd_show(args):
    from trlx import show

    if args.tui:
        show.show_tui(args.run, progress=args.progress)
    else:
        show.show_lines(args.run, progress=args.progress)
    return 0


# Check resolves the same config/CLI inputs as training, but launches no workers.
def _cmd_check(args):
    from trlx import train

    args.overrides = overrides(args)
    return train.check(args)


# Exit 1 when a verify check fails: the report has already been printed and
# written, so no message is added here.
def _cmd_verify(args):
    from trlx import verify

    return 0 if verify.run(args.checkpoint, args.base,
                          force=args.force, no_staging=args.no_staging, progress=args.progress).ok else 1


# Generate replay rows locally or through a configured endpoint.
def _cmd_replay_build(args):
    from trlx import replay_build

    replay_build.run(args, progress=args.progress)
    return 0


# Method-specific examples explain additional inputs without selecting an objective.
def _training_examples(method):
    extra = ""
    if method == "distillation":
        extra = " \\\n    --teacher TEACHER"
    elif method in ("grpo", "rloo"):
        extra = " \\\n    --reward json_valid --vllm-server-base-url http://localhost:8000"
    return (
        f"Examples:\n  trlx {method} --model MODEL --dataset DATA{extra}\n"
        f"  trlx {method} --model MODEL --dataset DATA{extra} \\\n"
        "    --learning-rate 1e-5 --output-dir runs/experiment\n\n"
        "Precedence: CLI > [methods." + method + "] > shared config; CLI values never rewrite run.toml.\n"
        "Before loading models or datasets, review the settings printed to stdout.\n"
        "Press Enter to continue or q to quit, including with --tui and when resuming.\n"
        "Input is required; EOF cancels startup with an error.\n"
        "Booleans use --flag / --no-flag. Lists and tables use shell-quoted TOML, for example:\n"
        "  --lora-target-modules '[\"module_a\",\"module_b\"]'\n"
        "  --ranges '{loss=[0,5],eval_loss=[0,5]}'\n"
        "Fresh runs create output_dir/YYYYMMDD-N--model--dataset; N advances across that parent's date.\n"
        f"Resume: trlx {method} --resume-from-checkpoint RUN_DIR/checkpoint-N\n"
        "Resume loads the saved config, keeps the run directory and logs, and automatically removes "
        "metrics/checkpoints after the selected step. Explicit CLI overrides still apply."
    )


# Launch options have no argparse defaults that could overwrite saved settings.
def _run_options(parser, training):
    parser.add_argument("--force", action="store_true", help="authorize destructive output replacement")
    if training:
        parser.add_argument("--no-staging", action="store_true",
                            help="write outputs directly; failures can leave incomplete outputs")
    parser.add_argument("--config", default="run.toml",
                        help="fresh-run settings (default: run.toml); resume loads the selected run's saved config")
    add_run_settings(parser, training)


# Only the selected method's metadata is loaded; root and utility help stay light.
def build_parser(method=None):
    parser = argparse.ArgumentParser(
        prog="trlx", formatter_class=HelpFormatter,
        description="Train with TRL using environment defaults and ordinary CLI options.",
        epilog="Start here:\n  trlx init\n  trlx sft --model MODEL --dataset DATA\n\n"
               "Settings: CLI overrides > selected method section > shared run.toml settings.\n"
               "CLI overrides apply to one run; edit run.toml for persistent changes.\n"
               "Progress goes to stderr; training also records it in log.txt.\n"
               "During loading and verification, training commands show waiting notices after 30 seconds\n"
               "without substantive feedback. Notices stay in log.txt while workers are training.\n"
               "Other commands report waiting after 10 seconds.\n"
               "Inspect a command: trlx sft --help, trlx check sft --help, trlx merge --help.\n"
               "Help never requires a config file, model, dataset, or GPU.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    p = sub.add_parser("init", help="inspect the environment and write defaults for all methods",
                       formatter_class=HelpFormatter,
                       description="Write shared hardware-informed defaults and all method sections. "
                                   "No model, dataset, training method, or calibration run is needed.",
                       epilog="Examples:\n  trlx init\n  trlx init --force\n  trlx init --out experiment.toml\n\n"
                              "Existing files require --force, which replaces saved settings with fresh defaults. "
                              "Training still needs a CUDA GPU. "
                              "Batch/memory defaults are estimates, not a model-fit guarantee.")
    p.add_argument("--out", default="run.toml", help="config file to create (default: run.toml)")
    p.add_argument("--force", action="store_true",
                   help="overwrite the config with fresh environment defaults, discarding saved edits")
    p.add_argument("--no-staging", action="store_true", help="write directly; failure can discard saved settings")
    p.set_defaults(func=_cmd_init)

    # One subparser per method; all share the run flag set.
    purposes = {
        "sft": "supervised fine-tuning: messages, text, or prompt/completion",
        "dpo": "preference training: chosen/rejected answers",
        "kto": "preference training: prompt/completion with a boolean label",
        "reward": "train a reward model on preference pairs",
        "grpo": "policy training with reward functions and a generation server",
        "rloo": "leave-one-out policy training with rewards and a generation server",
        "distillation": "train from a teacher model on prompts",
    }
    for name in METHODS:
        p = sub.add_parser(name, help=purposes[name], formatter_class=HelpFormatter,
                           usage=f"trlx {name} [--config FILE] [--model MODEL] [--dataset DATA] [options]",
                           description=f"{purposes[name]}. Fresh runs use run.toml unless --config is given. "
                                       "Resume uses the run's saved config. Model/data may be saved in the config. "
                                       "All options override one run only.",
                           epilog=_training_examples(name))
        _run_options(p, training=True)
        # Internal: set by the supervisor when spawning worker processes (PLAN.md multi-GPU design).
        p.add_argument("--_rank", type=int, help=argparse.SUPPRESS)
        p.add_argument("--_strategy", choices=["single", *STRATEGIES], help=argparse.SUPPRESS)
        if method == name:
            add_training_options(p, name)
        p.set_defaults(func=_cmd_train)

    p = sub.add_parser("show", help="inspect saved metrics, checkpoints, and reports", formatter_class=HelpFormatter,
                       description="Read a run directory without loading a model. Values include changes and range markers.",
                       epilog="Examples:\n  trlx show RUN_DIR\n  trlx show RUN_DIR --tui\n\n"
                              "RUN_DIR is the generated directory printed at training startup. "
                              "Line mode prints once. The TUI refreshes until q and includes logs/reports.")
    p.add_argument("run", help="run directory containing config.toml and metrics.jsonl")
    p.add_argument("--tui", action="store_true", help="full-screen view (default: print metric lines)")
    p.add_argument("--force", action="store_true", help="accepted for consistency; show only reads files")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("check", help="check a training setup without training", formatter_class=HelpFormatter,
                       usage="trlx check METHOD [--config FILE] --model MODEL --dataset DATA [options]",
                       description="Run preflight on the first selected GPU; the model must fit that GPU. "
                                   "Training performs its own preflight for larger sharded models.",
                       epilog="Examples:\n  trlx check sft --model MODEL --dataset DATA\n"
                              "  trlx check dpo --config preferences.toml --model MODEL --dataset DATA\n\n"
                              "Use trlx check METHOD --help for that method's full override reference.")
    p.add_argument("method", choices=METHODS, help="training method to validate")
    _run_options(p, training=False)
    if method is not None:
        add_training_options(p, method)
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("verify", help="verify a saved checkpoint against its base", formatter_class=HelpFormatter,
                       description="Check checkpoint loading, adapter integrity, and chat-template equality. "
                                   "Writes verify.json; returns nonzero when a check fails.",
                       epilog="Examples:\n  trlx verify CHECKPOINT --base BASE\n\n"
                              "For checkpoints inside a run, --base must match its saved model path. "
                              "All visible GPUs may be used.")
    p.add_argument("checkpoint", help="checkpoint directory")
    p.add_argument("--base", required=True, help="base model path")
    p.add_argument("--prompts", type=removed_verification_prompts, help=argparse.SUPPRESS)
    p.add_argument("--force", action="store_true", help="replace an existing verify.json report")
    p.add_argument("--no-staging", action="store_true", help="write the report directly; failure can discard the old report")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("merge", help="merge a LoRA adapter into its base model", formatter_class=HelpFormatter,
                       description="Validate adapter loading, then save merged model and processor. Existing output requires --force.",
                       epilog="Example:\n  trlx merge --base BASE --adapter ADAPTER --out merged-model")
    p.add_argument("--base", required=True, help="base model path")
    p.add_argument("--adapter", required=True, help="adapter directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--force", action="store_true", help="replace the output directory and all its contents, including inputs")
    p.add_argument("--no-staging", action="store_true",
                   help="write directly; failure can destroy old output; unavailable when replacing an input or its ancestor")
    p.set_defaults(func=_cmd_merge)

    # --model is always the model identifier: a path or HF id locally, the
    # served name at an endpoint. --endpoint selects the endpoint path and
    # brings the connection flags dataset chat uses (SPEC 2.1, 2.10).
    p = sub.add_parser("replay-build", help="generate a messages dataset for replay", formatter_class=HelpFormatter,
                       description="Generate one completion per prompt. Output format follows its extension; "
                                   "existing outputs or inputs require --force. Local generation uses the model's dtype.",
                       epilog="Examples:\n"
                              "  trlx replay-build --model MODEL --prompts prompts.jsonl --out replay.jsonl --max-tokens 256\n"
                              "  trlx replay-build --model MODEL --endpoint URL --prompts prompts.jsonl --out replay.jsonl \\\n"
                              "    --max-tokens 256 --timeout 120 --retries 2 --concurrency 4 --api-key API_KEY\n\n"
                              "The endpoint is an OpenAI-compatible API base (including /v1 when required). "
                              "Endpoint options are rejected without --endpoint. API_KEY names a variable "
                              "from the environment or working-directory .env.")
    p.add_argument("--model", required=True, help="model path or HF id; with --endpoint, the served model name")
    p.add_argument("--endpoint", metavar="URL", help="OpenAI-compatible API base; absent means local generation")
    p.add_argument("--prompts", required=True, help="prompts dataset (prompt or messages column)")
    p.add_argument("--out", required=True, help="output dataset path")
    p.add_argument("--max-tokens", type=int, required=True, help="positive completion token limit per prompt")
    p.add_argument("--timeout", type=float, metavar="SECONDS", help="positive seconds per request; required with --endpoint")
    p.add_argument("--retries", type=int, help="nonnegative retry count after first attempt; required with --endpoint")
    p.add_argument("--concurrency", type=int, help="positive parallel request count; required with --endpoint")
    p.add_argument("--api-key", metavar="ENVVAR", help="endpoint only; environment variable holding the key")
    p.add_argument("--force", action="store_true", help="replace existing output, including the prompts input")
    p.add_argument("--no-staging", action="store_true", help="write directly; failure can discard the old output")
    p.set_defaults(func=_cmd_replay_build)

    return parser


# Discover check's method without mistaking an option value (even "sft") for it.
# The selector knows arities only; the final method parser owns type validation.
def _check_method(argv):
    if not argv or argv in (["--help"], ["-h"]):
        return None
    if argv[0] in METHODS:
        return argv[0]
    selector = argparse.ArgumentParser(prog="trlx check", add_help=False)
    _run_options(selector, training=False)
    selector.add_argument("--help", "-h", action="store_true")
    seen = set(selector._option_string_actions)
    for method in METHODS:
        fields = argparse.ArgumentParser(add_help=False)
        add_training_options(fields, method)
        for action in fields._actions:
            names = [name for name in action.option_strings if name not in seen]
            if not names:
                continue
            seen.update(names)
            kwargs = {"default": argparse.SUPPRESS}
            if action.nargs == 0:
                kwargs["action"] = "store_true"
            else:
                kwargs["nargs"] = action.nargs
            selector.add_argument(*names, **kwargs)
    selector.add_argument("method", choices=METHODS, nargs="?")
    return selector.parse_known_args(argv)[0].method


# Determine the method before building its dynamic options; help exits at parsing.
def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    method = argv[0] if argv and argv[0] in METHODS else None
    if argv and argv[0] == "check":
        method = _check_method(argv[1:])
    return build_parser(method).parse_args(argv)


# Progress stores startup display failures; the training supervisor records them
# once log.txt exists and preserves ownership of the job despite a broken terminal.
def _defer_training_display_error(error):
    pass


# Only execution loads .env or dispatches work; help requires neither.
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Detailed help may inspect library metadata, but help never starts execution feedback.
    if "--help" in argv or "-h" in argv:
        parse_args(argv)
        return 0
    commands = METHODS + ["init", "show", "check", "verify", "merge", "replay-build"]
    command = argv[0] if argv and argv[0] in commands else "command"
    label = f"trlx {command}"
    # Identify internal worker output even during library imports. This only labels
    # feedback; parse_args still owns argument validation and the actual rank value.
    worker_rank = None
    for index, token in enumerate(argv):
        if token == "--_rank" and index + 1 < len(argv):
            worker_rank = argv[index + 1]
        elif token.startswith("--_rank="):
            worker_rank = token.partition("=")[2]
    if worker_rank is not None and worker_rank.isdecimal():
        label += f" rank {worker_rank}"
    from trlx import feedback

    connection = feedback.connect()
    events = connection if connection is not None else (
        feedback.Startup() if command in METHODS and worker_rank is None else None)
    try:
        on_error = _defer_training_display_error if command in METHODS and worker_rank is None else None
        with Progress(label, on_error=on_error, events=events) as progress:
            with stage(progress, "loading command options"):
                args = parse_args(argv)
            rank = getattr(args, "_rank", None)
            progress.command = f"trlx {args.command}" + (f" rank {rank}" if rank is not None else "")
            if rank is not None:
                # Worker stderr is the authoritative log, so its write failures are fatal.
                progress.on_error = None
            args.progress = progress
            if connection is not None:
                feedback.configure_logging(connection)
            # Secrets are loaded before dispatch, without including their values in feedback.
            with stage(progress, "loading credentials"):
                try:
                    load_env()
                except DatasetError as e:
                    raise TrlxError(str(e))
            with stage(progress, f"running {args.command}"):
                result = args.func(args)
            progress.finish("completed" if result == 0 else "failed")
            return result
    except TrlxError as e:
        # A supervisor may have disabled a broken stderr; never redirect errors to metrics stdout.
        if sys.stderr is not None:
            print(f"trlx {command}: {e}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


# launch.py starts workers as `python -m trlx.cli`, so they run under the same
# interpreter whether or not the project is installed.
if __name__ == "__main__":
    sys.exit(main())
