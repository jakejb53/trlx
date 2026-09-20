"""Startup review correctness without models, datasets, GPUs, or run writes."""

import contextlib
import copy
import io
import shlex
import types
import unittest
from unittest.mock import patch

from dataset.progress import Progress
from trlx import TrlxError, cli, config, options, review


# Real config constructors resolve library defaults on CPU using in-memory inputs.
def configuration(method="sft", extra=None):
    document = {
        "output_dir": "runs/example", "use_cpu": True, "bf16": False,
        "model": {"path": "example-model", "dtype": "float32"},
        "dataset": {"split": True, "dataset": "data.jsonl", "eval_fraction": 0.1},
        "ranges": {"loss": [0, 5]},
    }
    if method in {"grpo", "rloo"}:
        document.update(per_device_train_batch_size=8, num_generations_eval=1,
                        rewards={"funcs": ["json_valid"]})
    if method == "distillation":
        document["teacher"] = {"path": "example-teacher", "dtype": "float32"}
    document.update(extra or {})
    return config.from_document(document, method)


# Extract only complete override fragments, leaving explanatory comments out.
def arguments(text):
    return [line.split("   #", 1)[0].rstrip() for line in text.splitlines() if line.startswith("--")]


class Rendering(unittest.TestCase):
    # All trainer registries are exercised; no model source may be consulted.
    def test_method_selection_and_every_fragment_parses(self):
        expected = {"sft": "--assistant-only-loss", "dpo": "--beta", "kto": "--desirable-weight",
                    "reward": "--center-rewards-coefficient", "grpo": "--scale-rewards",
                    "rloo": "--reward-clip-range", "distillation": "--teacher"}
        for method in cli.METHODS:
            with self.subTest(method=method):
                cfg = configuration(method)
                before = copy.deepcopy(cfg.document)
                output = review.render(cfg, width=120)
                fragments = arguments(output)
                keys = set()
                for fragment in fragments:
                    parsed = cli.parse_args([method, *shlex.split(fragment)])
                    keys.update(options.overrides(parsed))
                self.assertEqual(before, cfg.document)
                positive = output.replace("--no-", "--")
                self.assertIn(expected[method], positive)
                if method != "sft":
                    self.assertNotIn("--assistant-only-loss", positive)
                if method != "kto":
                    self.assertNotIn("--desirable-weight", positive)
                if method != "distillation":
                    self.assertNotIn("--teacher", positive)
                self.assertIn("learning_rate", keys)
                self.assertIn("num_train_epochs", keys)
                self.assertNotIn("fsdp", keys)

    # Actual precedence and post-init defaults supply displayed values, not metadata defaults.
    def test_resolved_defaults_overrides_and_following_intervals(self):
        doc = configuration().document
        doc["learning_rate"] = 0.02
        doc["methods"] = {"sft": {"learning_rate": 0.003, "eval_strategy": "steps", "logging_steps": 7}}
        with patch.object(config, "_read_toml", return_value=doc):
            cfg = config.load("memory.toml", "sft", overrides={"learning_rate": 0.004})
        shown = arguments(review.render(cfg))
        self.assertIn("--learning-rate 0.004", shown)
        self.assertIn("--eval-steps 7", shown)
        self.assertIn("--save-steps 7", shown)
        self.assertIn("--gradient-checkpointing", shown)
        self.assertIn("--loss-type chunked_nll", shown)

    # Layout alignment includes long arguments and all wrapped continuation comments.
    def test_comment_alignment_and_no_value_truncation(self):
        name = "a deliberately long model path with spaces " * 3
        cfg = configuration(extra={"model": {"path": name, "dtype": "float32"}})
        output = review.render(cfg, width=80)
        columns = {line.index("#") for line in output.splitlines() if "#" in line}
        self.assertEqual(len(columns), 1)
        self.assertIn(shlex.quote(name), output)
        self.assertGreater(len(output.splitlines()), len(arguments(output)))

    # Disabled schedules and features cannot advertise controls that do nothing.
    def test_inactive_settings_are_hidden(self):
        cfg = configuration(extra={"max_steps": 12, "save_strategy": "epoch", "save_steps": 8,
                                   "packing_strategy": "bfd", "gradient_checkpointing": False})
        shown = arguments(review.render(cfg))
        self.assertIn("--max-steps 12", shown)
        for prefix in ("--num-train-epochs", "--eval-steps", "--per-device-eval-batch-size",
                       "--save-steps", "--packing-strategy", "--metric-for-best-model"):
            self.assertFalse(any(line.startswith(prefix + " ") for line in shown), prefix)

    # LoRA set values and shell-sensitive text must round-trip through the real parser.
    def test_lora_collections_and_shell_quoting(self):
        cfg = configuration(extra={"peft": {"r": 12, "target_modules": ["one", "two"]},
                                   "run_name": "run's name $(not-a-command) # quoted"})
        parsed_values = {}
        for fragment in arguments(review.render(cfg)):
            parsed_values.update(options.overrides(cli.parse_args(["sft", *shlex.split(fragment)])))
        self.assertEqual(parsed_values["peft.r"], 12)
        self.assertEqual(set(parsed_values["peft.target_modules"]), {"one", "two"})
        self.assertEqual(parsed_values["run_name"], cfg.args.run_name)
        self.assertIn("peft.lora_dropout", parsed_values)

    # Auto generation values belong to workers, not the supervisor's world size.
    def test_automatic_generation_and_dataset_dependent_values(self):
        cfg = configuration("grpo")
        self.assertIsNotNone(cfg.args.generation_batch_size)
        output = review.render(cfg)
        self.assertIn("--generation-batch-size None", output)
        self.assertIn("--steps-per-generation None", output)
        self.assertIn("Automatic:", output)
        sft = review.render(configuration())
        self.assertIn("--completion-only-loss None", sft)
        self.assertIn("prompt/completion", sft)

    # A trlx-owned setting is explained without producing a rejected override.
    def test_forced_settings_are_comments(self):
        cfg = configuration(extra={"replay": {"dataset": "replay.jsonl", "fraction": 0.2, "kl_coef": 0.1}})
        output = review.render(cfg, width=160)
        self.assertNotIn("--loss-type", output)
        self.assertIn("loss_type = 'nll'; set by trlx", output)
        self.assertIn("--replay-kl-coef 0.1", output)
        policy = review.render(configuration("grpo"), width=160)
        self.assertNotIn("--use-vllm", policy)
        self.assertIn("vllm_mode = 'server'", policy)

    # Real sensitive fields and nested endpoint credentials must never reach stdout.
    def test_redaction_preserves_nonsecret_token_settings(self):
        cfg = configuration(extra={"hub_token": "private-hub-secret", "eos_token": "<end>"})
        output = review.render(cfg)
        self.assertNotIn("private-hub-secret", output)
        self.assertIn("<redacted>", output)
        self.assertIn("--eos-token", output)
        cfg = configuration("grpo", {"rewards": {"funcs": [
            {"name": "llm_judge", "args": {"api_key": "private-key", "url": "https://u:private-pass@host/v1?token=private-token"}}
        ]}})
        output = review.render(cfg)
        for secret in ("private-key", "private-pass", "private-token"):
            self.assertNotIn(secret, output)
        self.assertIn("--reward:", output)

    # Explicit settings outside the shortlist stay visible without importing other methods.
    def test_explicit_advanced_setting_is_included(self):
        shown = arguments(review.render(configuration(extra={"adam_beta2": 0.95})))
        self.assertIn("--adam-beta2 0.95", shown)

    # Instantiated nested defaults containing None must remain visible without invalid TOML.
    def test_nested_automatic_defaults_are_explained(self):
        output = review.render(configuration(extra={"accelerator_config": {}}), width=160)
        self.assertIn("--accelerator-config:", output)
        self.assertIn("'dispatch_batches': None", output)
        self.assertFalse(any(line.startswith("--accelerator-config ") for line in arguments(output)))

    # bool|string options require a value, unlike genuine boolean switches.
    def test_mixed_boolean_union_uses_a_value(self):
        for value in (True, False):
            with self.subTest(value=value):
                output = review.render(configuration(extra={"peft": {"init_lora_weights": value}}))
                fragment = next(line for line in arguments(output) if line.startswith("--lora-init-lora-weights "))
                self.assertEqual(fragment, "--lora-init-lora-weights " + str(value).lower())
                parsed = cli.parse_args(["sft", *shlex.split(fragment)])
                self.assertIs(options.overrides(parsed)["peft.init_lora_weights"], value)

    # Generation dictionaries supersede sampling knobs without changing the loss temperature.
    def test_generation_override_precedence(self):
        cfg = configuration("distillation", {"temperature": 0.8, "generation_kwargs": {"top_p": 0.2, "temperature": 0.4}})
        output = review.render(cfg, width=160)
        shown = arguments(output)
        self.assertFalse(any(line.startswith("--top-p ") for line in shown))
        self.assertIn("--temperature 0.8", shown)
        self.assertIn("generation temperature is overridden", output)
        fragment = next(line for line in shown if line.startswith("--generation-kwargs "))
        parsed = cli.parse_args(["distillation", *shlex.split(fragment)])
        self.assertEqual(options.overrides(parsed)["generation_kwargs"], {"top_p": 0.2, "temperature": 0.4})

    # Explicit values for other loss variants are not applied to the selected objective.
    def test_inactive_specialized_loss_settings_are_hidden(self):
        for method, values, hidden in (
            ("dpo", {"loss_type": ["sigmoid"], "discopop_tau": 0.1}, "--discopop-tau"),
            ("grpo", {"loss_type": "bnpo", "sapo_temperature_neg": 2.0}, "--sapo-temperature-neg"),
            ("grpo", {"loss_type": "bnpo", "vespo_k_pos": 2.0}, "--vespo-k-pos"),
        ):
            with self.subTest(method=method, hidden=hidden):
                self.assertNotIn(hidden, review.render(configuration(method, values)))

    # Named rewards use the append action's bare-name spelling, not a quoted TOML string.
    def test_reward_names_round_trip(self):
        output = review.render(configuration("grpo"))
        fragment = next(line for line in arguments(output) if line.startswith("--reward "))
        parsed = cli.parse_args(["grpo", *shlex.split(fragment)])
        self.assertEqual(options.overrides(parsed)["rewards.funcs"], ["json_valid"])


class Input(unittest.TestCase):
    # The render path is tested above; these tests isolate consent and stream handling.
    def setUp(self):
        self.enterContext(patch.object(review, "render", return_value="Settings applied to this run:\n"))
        self.output = self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    # Only a real empty input line authorizes continuation; other text reprompts.
    def test_enter_quit_and_invalid_input(self):
        for text, result in (("\n", True), ("q\n", False), ("Q\n", False), ("yes\n \n\n", True)):
            with self.subTest(text=text), patch.object(review.sys, "stdin", io.StringIO(text)):
                self.assertEqual(review.confirm(None), result)
        self.assertIn("Enter an empty line", self.output.getvalue())

    # EOF is distinct from Enter, including when the prompt receives no terminal input.
    def test_eof_and_missing_streams_stop_startup(self):
        for stream in (io.StringIO(""), None):
            with self.subTest(stream=stream), patch.object(review.sys, "stdin", stream):
                with self.assertRaisesRegex(TrlxError, "training was not started"):
                    review.confirm(None)
        with patch.object(review.sys, "stdout", None), self.assertRaises(TrlxError):
            review.confirm(None)

    # Failed writes or reads cannot silently skip the required review.
    def test_closed_streams_stop_startup(self):
        closed = io.StringIO()
        closed.close()
        for name in ("stdin", "stdout"):
            with self.subTest(stream=name), patch.object(review.sys, name, closed):
                with self.assertRaisesRegex(TrlxError, "could not read stdin or write stdout"):
                    review.confirm(None)

    # Progress heartbeats remain quiet throughout the operator's decision.
    def test_progress_is_suspended_while_reading(self):
        emitted = []
        clock = types.SimpleNamespace(now=0)
        progress = Progress("test", emit=emitted.append, clock=lambda: clock.now)

        # An artificial elapsed wait exercises the actual heartbeat's suppression rule.
        def read():
            clock.now = 100
            progress.waiting()
            self.assertEqual(emitted, [])
            return "\n"

        with patch.object(review.sys, "stdin", types.SimpleNamespace(readline=read)):
            self.assertTrue(review.confirm(None, progress=progress))
        self.assertEqual(progress._suspended, 0)

    # Ctrl-C must unwind the suspended reporter and cannot become consent.
    def test_interrupt_propagates(self):
        with patch.object(review.sys, "stdin") as stream:
            stream.readline.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                review.confirm(None)
