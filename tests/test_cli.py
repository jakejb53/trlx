"""Operator-facing parsing and help, without executing any training or file writes."""

import contextlib
import io
import unittest
from unittest.mock import patch

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
            write.assert_called_once_with("alternate.toml", force=True, no_staging=no_staging)

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


if __name__ == "__main__":
    unittest.main()
