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
    # The review exposes membership selection and its effective data seed only where applicable.
    def test_random_eval_selection_and_seed_are_visible(self):
        cfg = configuration(extra={"data_seed": 0, "dataset": {
            "split": True, "dataset": "data.jsonl", "eval_fraction": .1, "shuffle_eval_data": True}})
        fragments = arguments(review.render(cfg, width=120))
        self.assertIn("--shuffle-eval-data", fragments)
        self.assertIn("--data-seed 0", fragments)
        separate = configuration(extra={"dataset": {"split": False, "dataset_train": "data.jsonl"}})
        self.assertNotIn("shuffle-eval-data", review.render(separate, width=120))

    # All trainer registries are exercised; no model source may be consulted.
    def test_method_selection_and_every_fragment_parses(self):
        expected = {"sft": "--assistant-only-loss", "dpo": "--beta", "kto": "--desirable-weight",
                    "reward": "--center-rewards-coefficient", "grpo": "--scale-rewards",
                    "rloo": "--reward-clip-range", "distillation": "--teacher-dtype"}
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

    # Configured operational controls must stay hidden for every trainer, not just SFT.
    def test_operational_settings_are_hidden_for_every_method(self):
        operational = {
            "run": {"gpus": "all", "strategy": "auto", "tui": False, "verify": True},
            "model": {"path": "example-model", "dtype": "float32",
                      "trust_remote_code": False, "attn_implementation": "sdpa"},
            "run_name": "example-run", "logging_steps": 3, "report_to": [],
            "save_strategy": "steps", "save_steps": 9, "save_total_limit": 2,
            "dataloader_num_workers": 0, "dataloader_pin_memory": True,
            "disable_tqdm": True, "push_to_hub": False,
        }
        hidden = {
            "gpus", "strategy", "tui", "verify", "model", "trust-remote-code",
            "attn-implementation", "dataset", "split", "dataset-eval", "output-dir",
            "run-name", "logging-steps", "report-to", "save-strategy", "save-steps",
            "save-total-limit", "dataloader-num-workers", "dataloader-pin-memory",
            "disable-tqdm", "push-to-hub", "verify-prompts", "teacher",
            "vllm-server-base-url", "vllm-server-host", "vllm-server-port",
            "vllm-server-timeout", "preflight-rows", "offpolicy-logp-per-token",
        }
        for method in cli.METHODS:
            with self.subTest(method=method):
                extra = copy.deepcopy(operational)
                if method in {"grpo", "rloo"}:
                    extra["vllm_server_base_url"] = "http://localhost:8000"
                if method in {"dpo", "kto"}:
                    extra["preflight"] = {"rows": 8, "offpolicy_logp_per_token": -5.0}
                cfg = configuration(method, extra)
                before = copy.deepcopy(cfg.document)
                output = review.render(cfg)
                shown = {shlex.split(line)[0].removeprefix("--").removeprefix("no-")
                         for line in arguments(output)}
                self.assertFalse(shown & hidden, shown & hidden)
                self.assertIn("learning-rate", shown)
                self.assertIn("dtype", shown)
                self.assertTrue(output.startswith("Training tuning settings:\n"))
                self.assertEqual(before, cfg.document)

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
        self.assertNotIn("--save-steps 7", shown)
        self.assertIn("--gradient-checkpointing", shown)
        self.assertIn("--loss-type chunked_nll", shown)

    # Layout alignment includes long arguments and all wrapped continuation comments.
    def test_comment_alignment_and_no_value_truncation(self):
        name = "a deliberately long end token with spaces " * 3
        cfg = configuration(extra={"eos_token": name})
        output = review.render(cfg, width=80)
        columns = {line.index("#") for line in output.splitlines() if "#" in line}
        self.assertEqual(len(columns), 1)
        self.assertIn(shlex.quote(name), output)
        self.assertGreater(len(output.splitlines()), len(arguments(output)))

    # Disabled schedules and features cannot advertise controls that do nothing.
    def test_inactive_settings_are_hidden(self):
        cfg = configuration(extra={"max_steps": 12, "save_strategy": "epoch", "save_steps": 8,
                                   "packing_strategy": "bfd", "gradient_checkpointing": False,
                                   "gradient_checkpointing_kwargs": {"use_reentrant": False}, "data_seed": "None"})
        shown = arguments(review.render(cfg))
        self.assertIn("--max-steps 12", shown)
        for prefix in ("--num-train-epochs", "--eval-steps", "--per-device-eval-batch-size",
                       "--save-steps", "--packing-strategy", "--metric-for-best-model",
                       "--gradient-checkpointing-kwargs", "--data-seed"):
            self.assertFalse(any(line.startswith(prefix + " ") for line in shown), prefix)

    # LoRA set values and shell-sensitive text must round-trip through the real parser.
    def test_lora_collections_and_shell_quoting(self):
        token = "end's token $(not-a-command) # quoted"
        cfg = configuration(extra={"peft": {"r": 12, "target_modules": ["one", "two"]},
                                   "eos_token": token})
        parsed_values = {}
        for fragment in arguments(review.render(cfg)):
            parsed_values.update(options.overrides(cli.parse_args(["sft", *shlex.split(fragment)])))
        self.assertEqual(parsed_values["peft.r"], 12)
        self.assertEqual(set(parsed_values["peft.target_modules"]), {"one", "two"})
        self.assertEqual(parsed_values["eos_token"], token)
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
        self.assertIn("--replay-fraction 0.2", output)
        self.assertNotIn("--replay-dataset", output)
        policy = review.render(configuration("grpo"), width=160)
        self.assertNotIn("--use-vllm", policy)
        self.assertNotIn("vllm_mode", policy)

    # Real sensitive fields and nested endpoint credentials must never reach stdout.
    def test_redaction_preserves_nonsecret_token_settings(self):
        cfg = configuration(extra={"hub_token": "private-hub-secret", "eos_token": "<end>"})
        output = review.render(cfg)
        self.assertNotIn("private-hub-secret", output)
        self.assertNotIn("--hub-token", output)
        self.assertIn("--eos-token", output)
        with patch.object(config, "_read_prompt", return_value="Operator-authored judging rubric."):
            cfg = configuration("grpo", {"rewards": {"funcs": [
                {"name": "llm_judge", "args": {"api_key": "private-key", "rubric_file": "judge.prompt",
                                               "url": "https://u:private-pass@host/v1?token=private-token"}}
            ]}})
        output = review.render(cfg)
        for secret in ("private-key", "private-pass", "private-token"):
            self.assertNotIn(secret, output)
        self.assertIn("--reward:", output)
        self.assertIn("<redacted>", output)

    # Deliberately selected advanced tuning stays visible when explicitly configured.
    def test_explicit_advanced_setting_is_included(self):
        shown = arguments(review.render(configuration(extra={"adam_beta2": 0.95})))
        self.assertIn("--adam-beta2 0.95", shown)

    # Explicit infrastructure dictionaries cannot expand the tuning review.
    def test_nested_infrastructure_is_hidden(self):
        output = review.render(configuration(extra={"accelerator_config": {}}), width=160)
        self.assertNotIn("--accelerator-config", output)
        self.assertNotIn("dispatch_batches", output)

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


class AssessmentRendering(unittest.TestCase):
    # Synthetic full-scan evidence exercises presentation without accessing source datasets.
    def report(self, *, split="train", issue="truncated_rows", count=284):
        return {"profile": {"train": {"rows": 289, "effective_rows": 200, "raw_tokens": 400000,
                                       "retained_tokens": 280000, "discarded_tokens": 120000},
                            "eval": {"rows": 33}},
                "findings": [{"code": f"data.{split}.{issue}", "severity": "warning", "basis": "projected",
                              "summary": "Sequence settings discard tokens", "evidence": {"split": split, issue: list(range(count))},
                              "recommendation": "Inspect the affected rows before changing sequence settings."}]}

    # Count source rows, preserve token totals, and expose effective CLI overrides with explanations.
    def test_truncation_counts_settings_and_immutability(self):
        cfg = configuration(extra={"max_length": 1024, "truncation_mode": "keep_start", "packing": False})
        report = self.report()
        original = copy.deepcopy(report)
        text = review.render_assessment(report, cfg, width=100)
        for fragment in ("284 of 289 rows (98.3%)", "Total source tokens: 400,000",
                         "Tokens retained after preparation: 280,000", "Tokens discarded by all preparation: 120,000",
                         "284 entries; first 5 shown", "0, 1, 2, 3, 4", "--max-length 1024", "--truncation-mode keep_start",
                         "--no-packing", "Keeps the beginning", "--packing-strategy"):
            self.assertIn(fragment, text)
        self.assertNotIn('"truncated_rows":', text)
        self.assertEqual(report, original)

    # Missing token projections cannot masquerade as zero; wrapped prose remains complete.
    def test_eval_unknown_totals_and_narrow_terminal(self):
        cfg = configuration(extra={"max_length": 1024})
        text = review.render_assessment(self.report(split="eval", count=33), cfg, width=60)
        joined = " ".join(text.split())
        self.assertIn("33 of 33 rows (100.0%)", joined)
        self.assertIn("Tokens discarded by all preparation: unknown", joined)
        self.assertIn("within this split", joined)
        self.assertTrue(all(len(line) <= 60 for line in text.splitlines()))

    # Eval packing is independent of training packing; inactive controls must not mislead operators.
    def test_eval_packing_overrides_in_both_directions(self):
        for training, evaluation in ((True, False), (False, True)):
            with self.subTest(training=training, evaluation=evaluation):
                cfg = configuration(extra={"packing": training, "eval_packing": evaluation})
                text = review.render_assessment(self.report(split="eval", count=33), cfg)
                if evaluation:
                    self.assertIn("--eval-packing", text)
                    self.assertIn("--packing-strategy", text)
                    self.assertNotIn("--truncation-mode", text)
                else:
                    self.assertIn("--no-eval-packing", text)
                    self.assertIn("--truncation-mode", text)
                    self.assertNotIn("--packing-strategy", text)

    # Repairing duplicate data is not a reason to suggest unrelated tuning changes.
    def test_data_only_action_and_nested_evidence(self):
        report = self.report(issue="duplicates", count=2)
        finding = report["findings"][0]
        finding.update(basis="measured", summary="Identical examples repeat", recommendation="Confirm repetition is intentional.")
        finding["evidence"] = {"duplicates": [{"rows": [2, 7], "detail": {"source": "original"}}]}
        text = review.render_assessment(report, configuration())
        self.assertIn("Action applies to the data", text)
        self.assertIn("Rows: 2 entries (2, 7)", text)
        self.assertIn("Source: original", text)
        self.assertNotIn("Relevant settings", text)

    # Only problems render across all methods; their relevant settings remain visible.
    def test_all_methods_static_findings_and_setting_associations(self):
        from trlx import assessment

        for method in ("sft", "dpo", "kto", "grpo", "rloo", "reward", "distillation"):
            with self.subTest(method=method):
                cfg = configuration(method, extra={"max_steps": 40, "warmup_steps": 40})
                report = self.report()
                report["profile"]["eval"].update(effective_rows=33, loss_tokens=33759)
                report["findings"] = assessment.static_findings(cfg, report["profile"], 2)
                text = review.render_assessment(report, cfg, width=100)
                for flag in ("--max-steps 40", "--warmup-steps 40"):
                    self.assertIn(flag, text)
                self.assertNotIn("Batch per optimizer update", text)
                self.assertNotIn("Training duration:", text)
                self.assertNotIn("Evaluation coverage", text)
                self.assertNotIn("Training estimates and information", text)
                if method != "sft":
                    self.assertNotIn("--packing", text)

    # Display order follows severity while a max-step budget hides the overridden epoch setting.
    def test_priority_and_max_step_precedence(self):
        from trlx import assessment

        cfg = configuration(extra={"max_steps": 40, "num_train_epochs": 99, "warmup_steps": 40})
        report = self.report()
        report["findings"] = assessment.static_findings(cfg, report["profile"], 2) + report["findings"]
        report["findings"].append({"code": "data.train.errors", "severity": "error", "basis": "projected",
                                   "summary": "Rows could not be profiled", "evidence": {},
                                   "recommendation": "Correct incompatible rows."})
        text = review.render_assessment(report, cfg)
        self.assertLess(text.index("Errors:"), text.index("Warnings:"))
        self.assertIn("--max-steps 40", text)
        self.assertNotIn("--num-train-epochs", text)

    # Empty and information-only reports print nothing, even with quality checks configured.
    def test_no_problems_omits_section_and_preserves_evidence(self):
        from trlx import assessment

        cfg = configuration(extra={"eval_strategy": "epoch", "peft": {"r": 16, "lora_alpha": 8, "use_rslora": True,
                                              "rank_pattern": {"Decoder.Q_proj$": 4}}})
        report = self.report()
        report["findings"] = assessment.static_findings(cfg, report["profile"], 1)
        report["quality"] = {"preset": "qa", "rows": 100}
        for findings in (report["findings"], []):
            report["findings"] = findings
            original = copy.deepcopy(report)
            for will_publish in (True, False):
                self.assertEqual(review.render_assessment(report, cfg, will_publish=will_publish), "")
                self.assertEqual(report, original)

    # Vocabulary token strings are evidence, whereas actual credential settings remain redacted.
    def test_distillation_mismatch_keeps_literal_tokens(self):
        report = self.report()
        report["findings"] = [{"code": "distillation_token_ids", "severity": "warning", "basis": "measured",
                               "summary": "Token IDs differ", "recommendation": "Use compatible vocabularies.",
                               "evidence": {"mismatches": [{"token": "Example_TOKEN", "student_id": 2, "teacher_id": 3}]}}]
        cfg = configuration("distillation")
        text = review.render_assessment(report, cfg)
        self.assertIn("Token: Example_TOKEN", text)
        self.assertIn("--model example-model", text)
        self.assertIn("--teacher example-teacher", text)

    # Scoring-only quality presets must not suggest a generation control they never use.
    def test_quality_context_settings_follow_preset(self):
        for preset in ("language_modeling", "preference", "qa"):
            with self.subTest(preset=preset):
                cfg = types.SimpleNamespace(**vars(configuration()))
                cfg.assessment = types.SimpleNamespace(quality_preset=preset, quality_max_length=1024,
                                                       quality_max_new_tokens=64)
                report = self.report()
                report["findings"] = [{"code": "quality_context_budget", "severity": "warning", "basis": "projected",
                                       "summary": "Quality context exceeds capacity", "evidence": {},
                                       "recommendation": "Review input and generation limits."}]
                text = review.render_assessment(report, cfg)
                self.assertIn("--quality-max-length 1024", text)
                self.assertEqual("--quality-max-new-tokens 64" in text, preset == "qa")


class Input(unittest.TestCase):
    # The render path is tested above; these tests isolate consent and stream handling.
    def setUp(self):
        self.enterContext(patch.object(review, "render", return_value="Training tuning settings:\n"))
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
