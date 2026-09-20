"""Operator-facing help stays complete and never starts dataset work."""

import argparse
import builtins
import contextlib
import io
import json
import os
import pathlib
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from dataset import cli, stats
from dataset.io import DatasetError

ROOT = pathlib.Path(__file__).resolve().parent


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


class DatasetOutputs(unittest.TestCase):
    # Scratch files stay in the repository; .env and terminal streams are isolated.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)
        self.source = self.root / "input.jsonl"
        self.source.write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8")

    # Exercise the real command boundary without loading the operator's environment.
    def run_cli(self, argv):
        stderr = io.StringIO()
        with patch("dataset.cli.env.load"), contextlib.redirect_stderr(stderr), \
                contextlib.redirect_stdout(io.StringIO()):
            status = cli.main([str(value) for value in argv])
        return status, stderr.getvalue()

    # Existing inputs are ordinary replaceable destinations once force is explicit.
    def test_in_place_requires_force_and_preserves_all_rows(self):
        original = self.source.read_bytes()
        command = ["shuffle", self.source, "--seed", "3", "--out", self.source]
        status, error = self.run_cli(command)
        self.assertEqual(status, 1)
        self.assertIn("--force", error)
        self.assertEqual(self.source.read_bytes(), original)
        status, error = self.run_cli(command + ["--force"])
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(sorted(json.loads(line)["text"] for line in self.source.read_text().splitlines()),
                         ["first", "second"])

    # The low-space path is explicit and never authorizes replacement by itself.
    def test_no_staging_does_not_imply_force(self):
        command = ["convert", self.source, "--out", self.source, "--no-staging"]
        self.assertEqual(self.run_cli(command)[0], 1)
        self.assertEqual(self.run_cli(command + ["--force"]), (0, ""))

    # Output refusal occurs before any remote generation work.
    def test_chat_existing_output_refuses_before_endpoint_setup(self):
        command = ["chat", "unused.txt", "--out", self.source,
                   "--questions-endpoint", "http://localhost:8000/v1", "--questions-model", "model",
                   "--n", "1", "--max-tokens", "32", "--concurrency", "1", "--timeout", "1", "--retries", "0"]
        with patch("dataset.cli.Endpoint", side_effect=AssertionError("endpoint setup")):
            status, error = self.run_cli(command)
        self.assertEqual(status, 1)
        self.assertIn("--force", error)

    # The second output's refusal must not publish the first output.
    def test_split_checks_both_destinations_before_writing(self):
        first = self.root / "first.jsonl"
        status, error = self.run_cli(["split", self.source, "--n", "1", "--out", first, "--rest", self.source])
        self.assertEqual(status, 1)
        self.assertIn("--force", error)
        self.assertFalse(first.exists())

    # Invalid second-output serialization leaves both existing outputs untouched.
    def test_split_prepares_both_outputs_before_replacement(self):
        self.source.write_text('{"text":"first"}\n{"text":["nested"]}\n', encoding="utf-8")
        first, rest = self.root / "first.jsonl", self.root / "rest.csv"
        first.write_text("original first", encoding="utf-8")
        rest.write_text("original rest", encoding="utf-8")
        status, error = self.run_cli(["split", self.source, "--n", "1", "--out", first, "--rest", rest, "--force"])
        self.assertEqual(status, 1)
        self.assertIn("CSV", error)
        self.assertEqual(first.read_text(), "original first")
        self.assertEqual(rest.read_text(), "original rest")

    # Force cannot make one path hold the two distinct products requested by split.
    def test_split_rejects_identical_outputs_before_input_read(self):
        output = self.root / "output.jsonl"
        with patch("dataset.cli.read_rows", side_effect=AssertionError("input read")):
            status, error = self.run_cli(["split", self.source, "--n", "1", "--out", output,
                                          "--rest", output, "--force"])
        self.assertEqual(status, 1)
        self.assertIn("distinct", error)

    # All documented command examples accept the common authorization flag.
    def test_force_and_no_staging_parser_contract(self):
        parser = cli.build_parser()
        for name, command in _commands(parser).items():
            example = next(line for line in command.epilog.replace("\\\n", " ").splitlines()
                           if line.strip().startswith("dataset "))
            argv = shlex.split(example)[1:] + ["--force"]
            if name != "stats":
                argv.append("--no-staging")
            with self.subTest(command=name):
                args = parser.parse_args(argv)
                self.assertTrue(args.force)
                self.assertEqual(getattr(args, "no_staging", False), name != "stats")

    # Invalid UTF-8 plain text becomes a concise path-specific command failure.
    def test_cpt_invalid_utf8(self):
        source = self.root / "input.txt"
        source.write_bytes(b"\xff")
        status, error = self.run_cli(["cpt", source, "--max-tokens", "32", "--out", self.root / "out.jsonl"])
        self.assertEqual(status, 1)
        self.assertIn(str(source), error)
        self.assertIn("UTF-8", error)


class StatsFailures(unittest.TestCase):
    # Exercise resource failures with tiny module substitutes, without loading GPU libraries.
    def test_model_load_and_device_transfer_oom_are_contextual(self):
        oom = type("OutOfMemoryError", (RuntimeError,), {})
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True, OutOfMemoryError=oom), bfloat16="bfloat16")
        for phase in ("load", "transfer"):
            with self.subTest(phase=phase):
                transformers = Mock()
                if phase == "load":
                    transformers.AutoModelForCausalLM.from_pretrained.side_effect = oom("exhausted")
                else:
                    transformers.AutoModelForCausalLM.from_pretrained.return_value.to.side_effect = oom("exhausted")
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                    with self.assertRaisesRegex(DatasetError, "--model example: out of memory loading model on cuda"):
                        stats._load("example")

    # Scoring errors name the exact row/column and suggest feasible recovery actions.
    def test_scoring_oom_names_row_and_column(self):
        oom = type("OutOfMemoryError", (RuntimeError,), {})
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=oom))
        rows = [{"prompt": "p", "completion": "c"}]
        with patch.dict(sys.modules, {"torch": torch}), \
                patch.object(stats, "_encode_pair", return_value=([1], [1, 2])), \
                patch.object(stats, "_logprob", side_effect=oom("exhausted")):
            with self.assertRaisesRegex(DatasetError, "row 0, column 'completion'.*shorten the input"):
                stats.logprobs(rows, Mock(), Mock(), "cuda")

    # Genuine model implementation failures are not converted into resource errors.
    def test_scoring_programmer_error_remains_visible(self):
        oom = type("OutOfMemoryError", (RuntimeError,), {})
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=oom))
        with patch.dict(sys.modules, {"torch": torch}), \
                patch.object(stats, "_encode_pair", return_value=([1], [1, 2])), \
                patch.object(stats, "_logprob", side_effect=RuntimeError("model bug")):
            with self.assertRaisesRegex(RuntimeError, "model bug"):
                stats.logprobs([{"prompt": "p", "completion": "c"}], Mock(), Mock(), "cuda")


# Allow direct execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
