"""argparse tree and dispatch for the trlx executable.

The tree here is the complete command surface from SPEC.md 2.1. Later phases
replace the stub handlers; they do not change the tree.
"""

import argparse
import sys

from dataset.env import load as load_env
from dataset.io import DatasetError
from trlx import TrlxError

# Stable TRL trainers only. Order is the order shown in --help.
METHODS = ["sft", "dpo", "grpo", "kto", "rloo", "reward", "distillation"]

STRATEGIES = ["ddp", "fsdp"]


# Placeholder handler until the owning phase lands. Exits nonzero so a stubbed
# command can never be mistaken for a silent success.
def _not_implemented(args):
    print(f"trlx {args.command}: not implemented", file=sys.stderr)
    return 2


# Handlers import their modules lazily: trlx.trainers imports trl and torch,
# and --help must not pay for that.


def _cmd_init(args):
    from trlx import init_cmd

    init_cmd.write(args.method, args.out)
    print(f"wrote {args.out}")
    return 0


def _cmd_merge(args):
    from trlx import merge

    merge.merge(args.base, args.adapter, args.out)
    return 0


# Method subcommands share one handler; the method is args.command.
def _cmd_train(args):
    from trlx import train

    return train.run(args)


def _cmd_show(args):
    from trlx import show

    if args.tui:
        show.show_tui(args.run)
    else:
        show.show_lines(args.run)
    return 0


def _cmd_check(args):
    from trlx import train

    return train.check(args)


# Exit 1 when a verify check fails: the report has already been printed and
# written, so no message is added here.
def _cmd_verify(args):
    from trlx import config, verify

    prompts = config.dataset_ref("trlx verify", "--prompts", args.prompts) if args.prompts else None
    return 0 if verify.run(args.checkpoint, args.base, prompts).ok else 1


def _cmd_replay_build(args):
    from trlx import replay_build

    replay_build.run(args)
    return 0


# Builds the full parser. Kept separate from main so tests can inspect the tree
# without invoking anything.
def build_parser():
    parser = argparse.ArgumentParser(
        prog="trlx", description="Drive TRL trainers from a run config."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    p = sub.add_parser("init", help="write a run config for a method")
    p.add_argument("method", choices=METHODS)
    p.add_argument("--out", required=True, help="path of the config to write")
    p.set_defaults(func=_cmd_init)

    # One subparser per method; all share the run flag set.
    for method in METHODS:
        p = sub.add_parser(method, help=f"run {method} training from a config")
        p.add_argument("config", help="run config TOML")
        p.add_argument("--tui", action="store_true", help="full-screen display")
        p.add_argument("--gpus", help="comma-separated device indices, e.g. 0,1")
        p.add_argument("--strategy", choices=STRATEGIES, help="override the automatic choice")
        p.add_argument("--no-verify", action="store_true", help="skip post-training verify")
        # Internal: set by the supervisor when spawning worker processes (PLAN.md multi-GPU design).
        p.add_argument("--_rank", type=int, help=argparse.SUPPRESS)
        p.set_defaults(func=_cmd_train)

    p = sub.add_parser("show", help="render a run's metrics.jsonl")
    p.add_argument("run", help="run directory")
    p.add_argument("--tui", action="store_true", help="full-screen display")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("check", help="preflight only")
    p.add_argument("method", choices=METHODS)
    p.add_argument("config", help="run config TOML")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("verify", help="artifact checks only")
    p.add_argument("checkpoint", help="checkpoint directory")
    p.add_argument("--base", required=True, help="base model path")
    p.add_argument("--prompts", help="dataset of prompts for the generation check")
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("merge", help="merge an adapter with adapter-load check")
    p.add_argument("--base", required=True, help="base model path")
    p.add_argument("--adapter", required=True, help="adapter directory")
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=_cmd_merge)

    # --model is always the model identifier: a path or HF id locally, the
    # served name at an endpoint. --endpoint selects the endpoint path and
    # brings the connection flags dataset chat uses (SPEC 2.1, 2.10).
    p = sub.add_parser("replay-build", help="sample a model on prompts into a messages dataset")
    p.add_argument("--model", required=True, help="model path or HF id; with --endpoint, the served model name")
    p.add_argument("--endpoint", metavar="URL", help="OpenAI-compatible API base; absent means local generation")
    p.add_argument("--prompts", required=True, help="prompts dataset (prompt or messages column)")
    p.add_argument("--out", required=True, help="output dataset path")
    p.add_argument("--max-tokens", type=int, required=True, help="completion length limit per prompt")
    p.add_argument("--timeout", type=float, metavar="SECONDS", help="endpoint only; required with --endpoint")
    p.add_argument("--retries", type=int, help="endpoint only; required with --endpoint")
    p.add_argument("--concurrency", type=int, help="endpoint only; required with --endpoint")
    p.add_argument("--api-key", metavar="ENVVAR", help="endpoint only; environment variable holding the key")
    p.set_defaults(func=_cmd_replay_build)

    return parser


# Entry point of the trlx console script and of `python -m trlx.cli`. Returns
# the process exit code. TrlxError is the one exception caught here: it carries
# a message already naming the path or key, so it is printed without a
# traceback. Anything else is a bug.
def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        # Before any handler runs, so an api_key variable named in a run config
        # or on the command line can come from .env. The loader is shared with
        # dataset, so its error is restated as the one exception type this
        # entry point catches.
        try:
            load_env()
        except DatasetError as e:
            raise TrlxError(str(e))
        return args.func(args)
    except TrlxError as e:
        print(f"trlx {args.command}: {e}", file=sys.stderr)
        return 1


# launch.py starts workers as `python -m trlx.cli`, so they run under the same
# interpreter whether or not the project is installed.
if __name__ == "__main__":
    sys.exit(main())
