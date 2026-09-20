"""Operator-facing help stays complete and never starts dataset work."""

import argparse
import builtins
import contextlib
import io
import os
import shlex
import unittest
from unittest.mock import patch

from dataset import cli


# Inspect the real parser tree so new commands must also satisfy help contracts.
def _commands(parser):
    return next(action.choices for action in parser._actions
                if isinstance(action, argparse._SubParsersAction))


class DatasetHelp(unittest.TestCase):
    # Help must exit before .env loading, file access, endpoint setup, or ML imports.
    def test_help_is_available_without_runtime_inputs(self):
        commands = _commands(cli.build_parser())
        original_import = builtins.__import__

        # A lazy model import during help is an interface regression even on a GPU host.
        def reject_model_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "trl", "trlx"}:
                raise AssertionError(f"help imported {name}")
            return original_import(name, *args, **kwargs)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("builtins.__import__", side_effect=reject_model_import))
            for target in ("env.load", "read_rows", "write_rows", "Endpoint"):
                stack.enter_context(patch(f"dataset.cli.{target}", side_effect=AssertionError(target)))
            for argv in [["--help"]] + [[name, "--help"] for name in commands]:
                with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()) as output:
                    with self.assertRaises(SystemExit) as result:
                        cli.main(argv)
                    self.assertEqual(result.exception.code, 0)
                    self.assertIn("usage: dataset", output.getvalue())

    # Each command must describe its purpose, arguments, and at least one invocation.
    def test_commands_and_options_are_documented(self):
        parser = cli.build_parser()
        top_help = parser.format_help()
        for name, command in _commands(parser).items():
            with self.subTest(command=name):
                self.assertIn(name, top_help)
                self.assertTrue(command.description)
                self.assertIn(f"dataset {name} ", command.epilog)
                for action in command._actions:
                    self.assertTrue(action.help, f"{name}: missing help for {action.dest}")

    # Examples are copied from the help itself; parsing them must not dispatch work.
    def test_examples_parse_without_execution(self):
        parser = cli.build_parser()
        for name, command in _commands(parser).items():
            for line in command.epilog.replace("\\\n", " ").splitlines():
                if not line.strip().startswith("dataset "):
                    continue
                with self.subTest(command=name, example=line):
                    args = parser.parse_args(shlex.split(line)[1:])
                    self.assertEqual(args.command, name)
                    self.assertTrue(callable(args.func))

    # Displayed fractions must not mislead operators about evaluation or mixture size.
    def test_sampling_constraints_are_explained(self):
        commands = _commands(cli.build_parser())
        split_help = commands["split"].format_help()
        self.assertIn("exactly one of --n or --fraction", split_help)
        self.assertIn("first output may exceed", split_help)
        mix_help = commands["mix"].format_help()
        self.assertIn("share of EACH input", mix_help)
        self.assertIn("fractions need not sum to 1", mix_help)
        self.assertIn("exactly one of --seed or --head", commands["sample"].format_help())

    # Generation help must distinguish input chunk sizing, endpoint limits, and credentials.
    def test_chat_explains_endpoint_and_reasoning_contract(self):
        help_text = " ".join(_commands(cli.build_parser())["chat"].format_help().split())
        for contract in (
            "completion length is controlled by the server",
            "Provide both answers endpoint/model flags, or neither",
            "separate reasoning column",
            "unclosed blocks always fail",
            "exported value takes precedence",
            "default: no Authorization header",
            "timeout per API request, in seconds",
            "retries after initial request",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, help_text)

    # A 120-column terminal must not wrap tables, usage lines, or copyable examples.
    def test_help_fits_120_columns(self):
        with patch.dict(os.environ, {"COLUMNS": "120"}):
            parser = cli.build_parser()
            for name, command in {"dataset": parser, **_commands(parser)}.items():
                for line in command.format_help().splitlines():
                    with self.subTest(command=name, line=line):
                        self.assertLessEqual(len(line), 120)


# Allow direct execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
