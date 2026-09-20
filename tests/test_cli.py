"""Operator-facing parsing and help, without executing any training or file writes."""

import contextlib
import io
import types
import unittest
from unittest.mock import ANY, patch

from dataset.progress import Progress
from trlx import TrlxError, cli, config, hardware, model, options


class Interface(unittest.TestCase):
    # Initialization has no task inputs; the settings filename is conventional.
    def test_init_needs_no_method(self):
        args = cli.parse_args(["init"])
        self.assertEqual(args.out, "run.toml")
        self.assertFalse(args.force)
        self.assertFalse(hasattr(args, "method"))

    # Explicit overwrite reaches the writer for the selected output path only.
    def test_init_force_is_forwarded(self):
        for flags, no_staging in (([], False), (["--no-staging"], True)):
            with self.subTest(no_staging=no_staging), patch.object(cli, "load_env"), \
                 patch("trlx.init_cmd.write", return_value=hardware.Hardware(8, ())) as write, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["init", "--force", "--out", "alternate.toml", *flags]), 0)
            write.assert_called_once_with("alternate.toml", force=True, no_staging=no_staging, progress=ANY)
            self.assertIsInstance(write.call_args.kwargs["progress"], Progress)

    # Force is an execution control even on commands with no destructive output work.
    def test_force_is_accepted_without_becoming_a_config_override(self):
        commands = [[method] for method in cli.METHODS]
        commands += [["check", method] for method in cli.METHODS]
        commands += [["init"], ["show", "runs/example"], ["verify", "checkpoint", "--base", "base"],
                     ["merge", "--base", "base", "--adapter", "adapter", "--out", "merged"],
                     ["replay-build", "--model", "model", "--prompts", "prompts.jsonl",
                      "--out", "replay.jsonl", "--max-tokens", "10"]]
        for command in commands:
            with self.subTest(command=command):
                args = cli.parse_args([*command, "--force"])
                self.assertTrue(args.force)
                self.assertNotIn("force", options.overrides(args))

    # Direct publication is independently selectable and never implies replacement permission.
    def test_no_staging_does_not_imply_force(self):
        for command in (["init"], ["verify", "checkpoint", "--base", "base"],
                        ["merge", "--base", "base", "--adapter", "adapter", "--out", "merged"],
                        ["replay-build", "--model", "model", "--prompts", "prompts.jsonl",
                         "--out", "replay.jsonl", "--max-tokens", "10"]):
            with self.subTest(command=command):
                args = cli.parse_args([*command, "--no-staging"])
                self.assertTrue(args.no_staging)
                self.assertFalse(args.force)
                self.assertNotIn("no_staging", options.overrides(args))

    # Every method supports the same minimal model/data invocation.
    def test_minimal_training_inputs(self):
        for method in cli.METHODS:
            with self.subTest(method=method):
                args = cli.parse_args([method, "--model", "org/model", "--dataset", "data.jsonl"])
                self.assertEqual(args.config, "run.toml")
                self.assertEqual(options.overrides(args), {
                    "model.path": "org/model", "dataset.source": "data.jsonl",
                })

    # Omitted argparse values cannot shadow persistent operator settings.
    def test_omitted_options_are_not_overrides(self):
        self.assertEqual(options.overrides(cli.parse_args(["sft"])), {})

    # Scalars, booleans, nullable fields and collections retain their config types.
    def test_typed_overrides(self):
        args = cli.parse_args([
            "sft", "--learning-rate", "1e-4", "--max-steps", "20", "--no-bf16",
            "--max-length", "None", "--lora-target-modules", '["one","two"]',
            "--lora-alpha", "16", "--lora-bias", "none", "--lora-adapter-bias",
            "--ranges", '{loss=[0,5]}', "--no-tui", "--no-verify", "--gpus", "all",
        ])
        values = options.overrides(args)
        self.assertEqual(values["learning_rate"], 1e-4)
        self.assertEqual(values["max_steps"], 20)
        self.assertIs(values["bf16"], False)
        self.assertEqual(values["max_length"], "None")
        self.assertEqual(values["peft.target_modules"], ["one", "two"])
        self.assertEqual(values["peft.lora_alpha"], 16)
        self.assertEqual(values["peft.bias"], "none")
        self.assertIs(values["peft.lora_bias"], True)
        self.assertEqual(values["ranges"], {"loss": [0, 5]})
        self.assertIs(values["run.verify"], False)
        self.assertIs(values["run.tui"], False)

    # Repeated explicit rewards replace the configured list in the supplied order.
    def test_reward_factory_and_name(self):
        args = cli.parse_args([
            "grpo", "--reward", "json_valid", "--reward",
            '{name="reference_match",args={column="answer",mode="equals"}}',
        ])
        self.assertEqual(options.overrides(args)["rewards.funcs"], [
            "json_valid", {"name": "reference_match", "args": {"column": "answer", "mode": "equals"}},
        ])

    # Feature-disable switches reject settings that would otherwise be ignored.
    def test_conflicting_lora_and_replay_switches(self):
        for flags in (["--no-lora", "--lora-r", "8"],
                      ["--no-replay", "--replay-fraction", "0.2"]):
            with self.subTest(flags=flags), self.assertRaises(TrlxError):
                options.overrides(cli.parse_args(["sft", *flags]))

    # Preflight accepts the same overrides without a mandatory config positional.
    def test_check_inputs(self):
        args = cli.parse_args(["check", "dpo", "--config", "other.toml", "--model", "model",
                               "--dataset", "data.jsonl", "--preflight-rows", "8"])
        self.assertEqual(args.method, "dpo")
        self.assertEqual(args.config, "other.toml")
        self.assertEqual(options.overrides(args)["preflight.rows"], 8)

    # Method discovery respects preceding option values, including a model called sft.
    def test_check_accepts_options_before_method(self):
        args = cli.parse_args(["check", "--config", "custom.toml", "--model", "sft",
                               "--learning-rate", "1e-5", "dpo", "--dataset", "data.jsonl"])
        self.assertEqual(args.method, "dpo")
        self.assertEqual(args.config, "custom.toml")
        self.assertEqual(options.overrides(args)["model.path"], "sft")

    # Nullable boolean resets are distinct from both disabling and retaining config.
    def test_nullable_boolean_reset(self):
        for flags, expected in ((["--tf32"], True), (["--no-tf32"], False),
                                (["--tf32", "None"], "None"), (["--tf32", "false"], False)):
            with self.subTest(flags=flags):
                self.assertEqual(options.overrides(cli.parse_args(["sft", *flags]))["tf32"], expected)

    # Hidden worker arguments describe launch facts, not temporary user overrides.
    def test_worker_arguments_do_not_change_config(self):
        args = cli.parse_args(["sft", "--config", "snapshot.toml", "--_rank", "0", "--_strategy", "fsdp"])
        self.assertEqual(args._strategy, "fsdp")
        self.assertEqual(options.overrides(args), {})

    # All help paths exit before reading config, probing hardware, or loading models.
    def test_help_is_complete_and_side_effect_free(self):
        paths = [[], ["init"], ["show"], ["check"], ["merge"], ["verify"], ["replay-build"]]
        paths += [[method] for method in cli.METHODS]
        paths += [["check", method] for method in cli.METHODS]
        for path in paths:
            with self.subTest(path=path):
                output = io.StringIO()
                with patch.object(cli, "load_env", side_effect=AssertionError("loaded .env")), \
                     patch.object(config, "_read_toml", side_effect=AssertionError("read config")), \
                     patch.object(hardware, "inspect", side_effect=AssertionError("probed hardware")), \
                     patch.object(model, "load_model", side_effect=AssertionError("loaded model")), \
                     contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit:
                    cli.main([*path, "--help"])
                self.assertEqual(exit.exception.code, 0)
                help_text = output.getvalue()
                self.assertIn("usage:", help_text)
                self.assertLessEqual(max(map(len, help_text.splitlines())), 120)
                if path and path[-1] in cli.METHODS:
                    for option in ("--model", "--dataset", "--learning-rate", "--lora-r", "--eval-fraction"):
                        self.assertIn(option, help_text)
                    self.assertIn("Library default", help_text)
                    self.assertNotIn("--_rank", help_text)
                    self.assertNotIn("Config: dataset.source", help_text)


class CommandProgress(unittest.TestCase):
    # Every public entrypoint must emit before its expensive operation and keep stdout clean.
    def test_every_command_announces_activity_before_work_and_reports_completion(self):
        commands = [([method], "trlx.train.run", 0) for method in cli.METHODS]
        commands += [
            (["init"], "trlx.init_cmd.write", hardware.Hardware(8, ())),
            (["show", "memory-run"], "trlx.show.show_lines", None),
            (["check", "sft"], "trlx.train.check", 0),
            (["verify", "checkpoint", "--base", "base"], "trlx.verify.run", types.SimpleNamespace(ok=True)),
            (["merge", "--base", "base", "--adapter", "adapter", "--out", "merged"], "trlx.merge.merge", None),
            (["replay-build", "--model", "model", "--prompts", "prompts.jsonl", "--out", "replay.jsonl",
              "--max-tokens", "10"], "trlx.replay_build.run", None),
        ]
        for argv, target, result in commands:
            with self.subTest(command=argv[0]):
                stderr, stdout = io.StringIO(), io.StringIO()
                label = f"trlx {argv[0]}"

                # Observe output at dispatch time so completion-only feedback cannot pass.
                def work(*args, **kwargs):
                    output = stderr.getvalue()
                    self.assertIn(f"{label}: starting", output)
                    self.assertIn(f"{label}: running {argv[0]}", output)
                    self.assertNotIn(f"{label}: completed;", output)
                    progress = kwargs.get("progress", getattr(args[0], "progress", None))
                    self.assertIsInstance(progress, Progress)
                    return result

                with patch.object(cli, "load_env"), patch(target, side_effect=work) as operation, \
                     contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.main(argv), 0)
                operation.assert_called_once()
                self.assertIn(f"{label}: completed; elapsed", stderr.getvalue())
                self.assertNotIn("starting", stdout.getvalue())

    # Training and verification return failures without raising, so lifecycle must use their status.
    def test_nonzero_result_reports_failure_without_changing_exit_status(self):
        for argv, target, result, expected in (
            (["sft"], "trlx.train.run", 7, 7),
            (["verify", "checkpoint", "--base", "base"], "trlx.verify.run", types.SimpleNamespace(ok=False), 1),
        ):
            with self.subTest(command=argv[0]), patch.object(cli, "load_env"), \
                 patch(target, return_value=result), contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(cli.main(argv), expected)
                self.assertIn(f"trlx {argv[0]}: failed; elapsed", output.getvalue())
                self.assertNotIn(f"trlx {argv[0]}: completed;", output.getvalue())

    # Shared worker logs need rank attribution from the first emitted line.
    def test_worker_feedback_identifies_rank(self):
        with patch.object(cli, "load_env"), patch("trlx.train.run", return_value=0), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(cli.main(["sft", "--_rank", "3"]), 0)
        self.assertTrue(all(line.startswith("trlx sft rank 3:") for line in output.getvalue().splitlines()))

    # Importing dynamic trainer options can block before normal command dispatch starts.
    def test_startup_is_visible_before_dynamic_command_options_are_loaded(self):
        output = io.StringIO()

        # Stand in for the expensive parse boundary without importing a trainer.
        def parse(argv):
            self.assertEqual(argv, ["sft"])
            self.assertIn("trlx sft: starting", output.getvalue())
            self.assertIn("loading command options", output.getvalue())
            return types.SimpleNamespace(command="sft", _rank=None, func=lambda args: 0)

        with patch.object(cli, "parse_args", side_effect=parse) as parse_args, \
             patch.object(cli, "load_env"), contextlib.redirect_stderr(output):
            self.assertEqual(cli.main(["sft"]), 0)
        parse_args.assert_called_once()

    # Help must retain ordinary argparse output without execution threads or credentials reads.
    def test_help_starts_no_reporter(self):
        for argv in (["--help"], ["sft", "--help"], ["sft", "-h"]):
            with self.subTest(argv=argv), patch.object(cli, "Progress") as reporter, \
                 patch.object(cli, "load_env") as load_env, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as caught:
                    cli.main(argv)
                self.assertEqual(caught.exception.code, 0)
                reporter.assert_not_called()
                load_env.assert_not_called()
                self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
