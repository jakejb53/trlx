"""Full-profile evidence reaches preflight without a second preparation policy."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset
import torch

from tests.test_data_profile import config, tokenizer
from trlx import data_profile, preflight


class PreflightProfileTests(unittest.TestCase):
    # Matching source content reuses the complete review, including BFD overflow;
    # packing must never suppress the warning about discarded source tokens.
    def test_matching_profile_reused_and_bfd_overflow_reported(self):
        cfg, processor = config(packing=True, max_length=4), tokenizer()
        rows = Dataset.from_list([{"text": "pqrstuvwxyz"}, {"text": "pq"}])
        profile = data_profile.scan(cfg, processor, rows, None)
        report = preflight.Report()
        with patch.object(data_profile, "scan", side_effect=AssertionError("matching full scan must be reused")):
            measured = preflight._check_truncation(cfg, rows, processor, report, profile=profile)
        self.assertIs(measured, profile["train"])
        self.assertEqual(report.facts["dataset_profile"]["discarded_tokens"], 8)
        self.assertTrue(any("1 of 2" in warning and "truncated" in warning for warning in report.warnings))
        self.assertFalse(any("avoids truncation" in line for line in report.lines))

    # Reward loses complete pairs; SFT keeps truncated sequences. Users need the
    # distinction because these settings change sample exposure in different ways.
    def test_reward_filtering_and_sft_truncation_are_distinguished(self):
        processor = tokenizer()
        for method, rows, expected, absent in (
            ("reward", [{"prompt": "pq", "chosen": "rstuvw", "rejected": "x"},
                        {"prompt": "pq", "chosen": "r", "rejected": "s"}], "dropped", "truncated"),
            ("sft", [{"text": "pqrstuvwxyz"}, {"text": "pq"}], "truncated", "dropped"),
        ):
            with self.subTest(method=method):
                cfg, report = config(method, max_length=5), preflight.Report()
                dataset = Dataset.from_list(rows)
                profile = data_profile.scan(cfg, processor, dataset, None)
                preflight._check_truncation(cfg, dataset, processor, report, profile=profile)
                self.assertTrue(any(expected in warning for warning in report.warnings))
                self.assertFalse(any(absent in warning for warning in report.warnings))
                self.assertEqual(report.facts["dataset_profile"]["effective_rows"], 1 if method == "reward" else 2)

    # A different worker dataset invalidates earlier evidence even if row counts
    # match; the fresh scan must measure its content and explain the refresh.
    def test_source_fingerprint_mismatch_explicitly_rescans(self):
        cfg, processor = config(max_length=4), tokenizer()
        original = Dataset.from_list([{"text": "pq"}])
        changed = Dataset.from_list([{"text": "pqrstuvwxyz"}])
        profile = data_profile.scan(cfg, processor, original, None)
        report = preflight.Report()
        with patch.object(data_profile, "scan", wraps=data_profile.scan) as rescanned:
            preflight._check_truncation(cfg, changed, processor, report, profile=profile)
        self.assertEqual(rescanned.call_count, 1)
        self.assertEqual(report.facts["dataset_profile"]["fingerprint"], data_profile.fingerprint(changed))
        self.assertEqual(report.facts["dataset_profile"]["truncated_rows"], [0])
        self.assertTrue(any("no matching pre-run source fingerprint" in line for line in report.lines))

    # Unsupported media identity is retained as evidence, never accepted merely
    # because both the cached and current fingerprints happen to be unknown.
    def test_unverifiable_identity_remains_explicit(self):
        cfg, processor = config(), tokenizer()
        rows = [{"text": "pq", "image": object()}]
        profile = data_profile.scan(cfg, processor, rows, None)
        report = preflight.Report()
        with patch.object(data_profile, "scan", wraps=data_profile.scan) as rescanned:
            measured = preflight._check_truncation(cfg, rows, processor, report, profile=profile)
        self.assertEqual(rescanned.call_count, 1)
        self.assertIsNone(measured["fingerprint"])
        self.assertIsNone(measured["effective_rows"])
        self.assertEqual(measured["identity_errors"][0]["row"], 0)
        self.assertTrue(any("source identity could not be verified" in warning for warning in report.warnings))
        self.assertTrue(any("unresolved rows" in warning for warning in report.warnings))

    # The actual trainer's prepared count is authoritative. Keep the projection
    # visible for comparison, and warn without replacing the trainer's dataset.
    def test_trainer_records_actual_prepared_count_and_disagreement(self):
        cfg, processor = config(), tokenizer()
        raw = Dataset.from_list([{"text": "pq"}, {"text": "rs"}])
        profile = data_profile.scan(cfg, processor, raw, None)
        prepared = Dataset.from_list([{"input_ids": [7, 8, 2], "labels": [7, 8, 2]}])
        trainer = SimpleNamespace(model=object(), processing_class=processor, train_dataset=prepared)
        report = preflight.Report()
        with patch.object(preflight, "_check_targets"), patch.object(preflight, "_check_use_cache"):
            preflight.check_trainer(cfg, trainer, raw, report, profile=profile)
        self.assertEqual(report.facts["assessment_validation"], {"projected_rows": 2, "prepared_rows": 1})
        self.assertTrue(any("Use the trainer count" in warning for warning in report.warnings))
        self.assertEqual(report.facts["example"]["tokens"], 3)
        self.assertIs(trainer.train_dataset, prepared)

    # Off-policy scoring must receive trainer-equivalent response boundaries,
    # including the EOS-only response created by this tokenizer's a+b BPE merge.
    def test_offpolicy_scores_concatenated_tokenization_boundaries(self):
        processor = tokenizer()
        for method in ("dpo", "kto"):
            with self.subTest(method=method):
                cfg = config(method)
                cfg.method.blocks = {"preflight"}
                cfg.preflight = SimpleNamespace(rows=1, offpolicy_logp_per_token=-5.0)
                row = {"prompt": "a", "chosen": "b", "rejected": "c"} if method == "dpo" else {
                    "prompt": "a", "completion": "b", "label": True}
                model, report = torch.nn.Linear(1, 1), preflight.Report()
                model.train()
                with patch.object(preflight, "_mean_logp", return_value=-1.0) as score:
                    preflight.check_offpolicy(cfg, model, processor, Dataset.from_list([row]), report)
                calls = score.call_args_list
                self.assertEqual(len(calls), 2 if method == "dpo" else 1)
                prompt_ids, response_ids = calls[0].args[1:3]
                self.assertEqual(prompt_ids, processor("a")["input_ids"])
                self.assertEqual(response_ids, [processor.eos_token_id])
                self.assertNotEqual(response_ids, processor("b<eos>")["input_ids"])
                if method == "dpo":
                    self.assertEqual(calls[1].args[2], processor("ac<eos>")["input_ids"][len(prompt_ids):])
                self.assertTrue(model.training)
                key = "chosen" if method == "dpo" else "completion"
                self.assertEqual(report.facts["offpolicy"][key]["scored"], 1)

    # Failed observational scoring must still return a model to its original mode.
    def test_offpolicy_failure_restores_model_mode(self):
        cfg, processor = config("kto"), tokenizer()
        cfg.method.blocks = {"preflight"}
        cfg.preflight = SimpleNamespace(rows=1, offpolicy_logp_per_token=-5.0)
        model = torch.nn.Linear(1, 1)
        model.train()
        rows = Dataset.from_list([{"prompt": "pq", "completion": "rs", "label": True}])
        with patch.object(preflight, "_mean_logp", side_effect=RuntimeError("synthetic scoring failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic scoring failure"):
                preflight.check_offpolicy(cfg, model, processor, rows, preflight.Report())
        self.assertTrue(model.training)
