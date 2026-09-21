"""Advisory rules consume synthetic immutable evidence without trainer/model execution."""

import copy
import json
import math
from types import SimpleNamespace
import unittest

from trlx import assessment


# Supply resolved values explicitly; the assessor must not invent operational defaults.
def config(method="sft", **overrides):
    args = SimpleNamespace(
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        num_train_epochs=1.0, max_steps=-1, dataloader_drop_last=False,
        eval_strategy="epoch", generation_batch_size=8, num_generations=2,
        num_iterations=1, steps_per_generation=2, loss_type="sigmoid",
        label_smoothing=0.0, train_sampling_strategy="sequential",
        desirable_weight=1.0, undesirable_weight=1.0, max_length=128,
        get_warmup_steps=lambda steps: math.ceil(steps * 0.1),
        **overrides,
    )
    return SimpleNamespace(method=SimpleNamespace(name=method), args=args, peft=None,
                           document={"generation_batch_size": 8} if method in ("grpo", "rloo") else {})


# Profiles expose prepared row counts separately from original source rows.
def profile(rows=24):
    return {"train": {"rows": rows, "effective_rows": rows, "loss_tokens": rows * 4,
                      "prepared_lengths": {"max": 16}, "prompt_lengths": {"max": 8}},
            "eval": {"effective_rows": 4, "loss_tokens": 16}, "findings": []}


# Real metrics envelopes carry trainer metrics only under log.
def record(step, evaluation=False, **metrics):
    return {"step": step, "time": step * 10, "eval": evaluation, "log": metrics}


# Stable finding codes are the transport's identity, independent of prose changes.
def indexed(findings):
    return {finding["code"]: finding for finding in findings}


class StaticAssessment(unittest.TestCase):
    # The same static entry point supports every registered training family.
    def test_all_methods_batch_budget(self):
        for method in ("sft", "dpo", "kto", "reward", "grpo", "rloo", "distillation"):
            with self.subTest(method=method):
                findings = indexed(assessment.static_findings(config(method), profile(), 2))
                self.assertEqual(findings["effective_batch"]["evidence"]["effective_batch"], 8)
                self.assertEqual(findings["training_budget"]["basis"], "projected")
                self.assertIn("Trainer dataloader", findings["training_budget"]["evidence"]["limitation"])
                json.dumps(list(findings.values()), allow_nan=False)

    # Explicit update budgets override epochs and use the installed args warmup resolver.
    def test_max_steps_and_warmup(self):
        cfg = config()
        cfg.args.max_steps, cfg.args.num_train_epochs = 5, 100
        cfg.args.get_warmup_steps = lambda steps: steps
        findings = indexed(assessment.static_findings(cfg, profile(1), 1))
        self.assertEqual(findings["training_budget"]["evidence"]["updates"], 5)
        self.assertEqual(findings["training_budget"]["evidence"]["budget_source"], "max_steps")
        self.assertIn("warmup_entire_run", findings)

    # Incomplete generation chunks disappear before sharding, including the entire dataset.
    def test_prompt_groups_empty_and_partial(self):
        for method in ("grpo", "rloo", "distillation"):
            with self.subTest(method=method):
                cfg = config(method)
                cfg.args.gradient_accumulation_steps = 4
                findings = indexed(assessment.static_findings(cfg, profile(1), 1))
                self.assertEqual(findings["prompt_groups"]["evidence"]["retained_prompts"], 0)
                self.assertIn("empty_training_batches", findings)
        findings = indexed(assessment.static_findings(config("grpo"), profile(11), 1))
        self.assertEqual(findings["prompt_groups"]["evidence"]["discarded_prompts"], 3)

    # Real supervisor configs resolve world_size=1; selected workers own automatic generation batches.
    def test_real_config_automatic_generation_uses_selected_workers(self):
        from tests.test_review import configuration

        for method in ("grpo", "rloo"):
            with self.subTest(method=method):
                cfg = configuration(method, {"per_device_train_batch_size": 1,
                                              "gradient_accumulation_steps": 8, "num_generations": 8})
                self.assertEqual(cfg.args.world_size, 1)
                self.assertEqual(cfg.args.generation_batch_size, 8)
                before = copy.deepcopy(cfg.document)
                findings = indexed(assessment.static_findings(cfg, profile(9), 2))
                groups = findings["prompt_groups"]["evidence"]
                self.assertEqual(groups["generation_batch_size"], 16)
                self.assertEqual(groups["steps_per_generation"], 8)
                self.assertEqual(groups["prompts_per_group"], 2)
                self.assertEqual(groups["discarded_prompts"], 1)
                self.assertEqual(findings["training_budget"]["evidence"]["updates_per_epoch"], 4)
                self.assertEqual(cfg.args.generation_batch_size, 8)
                self.assertEqual(cfg.document, before)

    # Explicit mutually exclusive controls survive resolution; TOML None preserves automatic semantics.
    def test_real_config_explicit_generation_schedule(self):
        from tests.test_review import configuration

        cases = (({"generation_batch_size": "None", "steps_per_generation": "None"}, 16, 8),
                 ({"steps_per_generation": 16}, 32, 16),
                 ({"generation_batch_size": 24}, 24, 12),
                 ({"generation_batch_size": 16, "steps_per_generation": "None"}, 16, 8),
                 ({"generation_batch_size": "None", "steps_per_generation": 8}, 16, 8))
        for method in ("grpo", "rloo"):
            for explicit, batch, steps in cases:
                with self.subTest(method=method, explicit=explicit):
                    cfg = configuration(method, {"per_device_train_batch_size": 1,
                                                  "gradient_accumulation_steps": 8, "num_generations": 8,
                                                  **explicit})
                    findings = indexed(assessment.static_findings(cfg, profile(9), 2))
                    groups = findings["prompt_groups"]["evidence"]
                    self.assertEqual(groups["generation_batch_size"], batch)
                    self.assertEqual(groups["steps_per_generation"], steps)
                    self.assertEqual(groups["discarded_prompts"], 9 % (batch // 8))

    # A batch valid on the supervisor can be indivisible on selected workers; do not invent groups.
    def test_selected_worker_generation_incompatibility(self):
        from tests.test_review import configuration

        for method in ("grpo", "rloo"):
            with self.subTest(method=method):
                cfg = configuration(method, {"per_device_train_batch_size": 1,
                                              "gradient_accumulation_steps": 8, "num_generations": 8,
                                              "generation_batch_size": 8})
                findings = indexed(assessment.static_findings(cfg, profile(9), 3))
                self.assertIn("generation_schedule_incompatible", findings)
                self.assertNotIn("prompt_groups", findings)
                self.assertNotIn("training_budget", findings)
        cfg = config("grpo")
        cfg.document["steps_per_generation"] = 2
        findings = indexed(assessment.static_findings(cfg, profile(9), 2))
        self.assertIn("mutually exclusive", findings["generation_schedule_incompatible"]["summary"])

    # LoRA defaults are not claimed as layer-wide facts when patterns override them.
    def test_rslora_and_patterns(self):
        cfg = config()
        cfg.peft = SimpleNamespace(r=16, lora_alpha=8, use_rslora=True,
                                   rank_pattern={"layer": 4}, alpha_pattern={"layer": 2})
        result = indexed(assessment.static_findings(cfg, profile(), 1))["lora_scale"]
        self.assertEqual(result["evidence"]["default_scale"], 2)
        self.assertEqual(result["evidence"]["rank_pattern"], {"layer": 4})
        self.assertIn("Patterns override", result["evidence"]["limitation"])

    # Accumulation does not satisfy KTO's physical KL rotation batch requirement.
    def test_kto_batch_and_singleton(self):
        cfg = config("kto")
        cfg.args.per_device_train_batch_size = 1
        cfg.args.gradient_accumulation_steps = 32
        self.assertIn("kto_kl_batch", indexed(assessment.static_findings(cfg, profile(), 1)))
        cfg.args.per_device_train_batch_size = 2
        findings = indexed(assessment.static_findings(cfg, profile(3), 1))
        self.assertIn("kto_singleton_kl", findings)
        cfg.args.loss_type = "apo_zero_unpaired"
        self.assertNotIn("kto_singleton_kl", indexed(assessment.static_findings(cfg, profile(3), 1)))

    # Weighted counts describe data contribution without prescribing a class optimum.
    def test_kto_label_counts(self):
        data = profile()
        data["train"]["label_counts"] = {"desirable": 20, "undesirable": 4}
        cfg = config("kto")
        cfg.args.undesirable_weight = 2
        evidence = indexed(assessment.static_findings(cfg, data, 1))["kto_weighted_balance"]["evidence"]
        self.assertEqual(evidence["weighted_undesirable_count"], 8)

    # Partial labels cannot stand for the full split; parallel map batches have local remainders.
    def test_kto_incomplete_profile_and_parallel_preparation(self):
        cfg, data = config("kto"), profile(3)
        cfg.args.dataset_num_proc = 2
        findings = indexed(assessment.static_findings(cfg, data, 1))
        self.assertNotIn("kto_singleton_kl", findings)
        self.assertIn("kto_parallel_kl_groups", findings)
        data["train"].update(effective_rows=None, loss_tokens=None,
                             label_counts={"desirable": 2, "undesirable": 0},
                             errors=[{"row": 2, "message": "cannot prepare row"}])
        findings = indexed(assessment.static_findings(cfg, data, 1))
        self.assertNotIn("kto_weighted_balance", findings)
        self.assertNotIn("training_budget", findings)
        self.assertNotIn("empty_training_batches", findings)

    # DPO smoothing follows this installed loss implementation, including mixed losses.
    def test_dpo_aot_and_ineffective_smoothing(self):
        cfg = config("dpo")
        cfg.args.loss_type = ["aot", "sigmoid"]
        cfg.args.label_smoothing = 0.1
        findings = indexed(assessment.static_findings(cfg, profile(), 1))
        self.assertIn("dpo_aot_batch", findings)
        self.assertEqual(findings["dpo_inactive_smoothing"]["evidence"]["losses_ignoring_smoothing"], ["sigmoid"])
        cfg.args.use_liger_kernel = True
        self.assertNotIn("dpo_inactive_smoothing", indexed(assessment.static_findings(cfg, profile(), 1)))

    # Reward pair dropping differs materially from truncated continuation training.
    def test_reward_drops_and_model_limits(self):
        data = profile()
        data["train"]["dropped_rows"] = [1, 3]
        findings = indexed(assessment.static_findings(config("reward"), data, 1, model_metadata={"max_position_embeddings": 8}))
        self.assertIn("reward_filtered_pairs", findings)
        self.assertEqual(findings["reward_filtered_pairs"]["basis"], "projected")
        self.assertEqual(findings["evaluation_coverage"]["basis"], "projected")
        self.assertIn("model_context_budget", findings)

    # Real profiler source totals preserve unknown packed loss attribution verbatim.
    def test_profile_sources_and_prompt_context(self):
        cfg, data = config(), profile()
        cfg.replay = SimpleNamespace(fraction=0.25, kl_coef=0.1)
        data["train"]["sources"] = {
            "primary": {"rows": 18, "raw_tokens": 100, "retained_tokens": 96, "loss_tokens": None},
            "replay": {"rows": 6, "raw_tokens": 40, "retained_tokens": 32, "loss_tokens": None}}
        findings = indexed(assessment.static_findings(cfg, data, 1))
        self.assertEqual(findings["replay_exposure"]["evidence"]["sources"], data["train"]["sources"])
        cfg = config("grpo")
        cfg.args.max_completion_length = 8
        cfg.args.max_prompt_length = 6
        findings = indexed(assessment.static_findings(cfg, profile(), 1, model_metadata={"max_position_embeddings": 12}))
        self.assertEqual(findings["model_context_budget"]["evidence"]["requested_tokens"], 14)

    # Metadata mismatches warn; identical dimensions do not prove token-ID equivalence.
    def test_teacher_metadata_and_input_immutability(self):
        cfg, data = config("distillation"), profile()
        original = copy.deepcopy(data)
        findings = indexed(assessment.static_findings(cfg, data, 1,
                           model_metadata={"vocab_size": 16}, teacher_metadata={"vocab_size": 17}))
        self.assertIn("distillation_vocab_size", findings)
        self.assertEqual(data, original)
        self.assertEqual(cfg.args.max_steps, -1)


class RuntimeAssessment(unittest.TestCase):
    # Tests pass every heuristic setting explicitly to avoid hidden operational thresholds.
    def assess(self, records, method="sft", **kwargs):
        return indexed(assessment.runtime_findings(method, records, window=2, min_evaluations=3,
                                                   relative_change=0.1, **kwargs))

    # A numerical failure is immediate even during warmup and outside displayed ranges.
    def test_nonfinite_immediate_every_method(self):
        for method in ("sft", "dpo", "kto", "reward", "grpo", "rloo", "distillation"):
            with self.subTest(method=method):
                findings = self.assess([record(1, hidden=float("nan"))], method, warmup_steps=100, ranges={})
                self.assertIn("nonfinite:hidden", findings)
                json.dumps(list(findings.values()), allow_nan=False)

    # Recovered values resolve numerical warnings, but earlier windows cannot span the failure.
    def test_recovery_restarts_observation_window(self):
        records = [record(1, loss=1), record(2, loss=1), record(3, loss=float("inf")), record(4, loss=2)]
        self.assertNotIn("nonfinite:loss", self.assess(records))
        self.assertNotIn("rising_training_loss", self.assess(records))

    # Sparse metrics are legitimate; missing observations cannot establish a trend.
    def test_missing_data_and_repeat_logs(self):
        self.assertEqual(self.assess([]), {})
        self.assertEqual(self.assess([record(1, loss=1), record(2, learning_rate=0.1)]), {})
        duplicated = [record(1, loss=1)] * 4 + [record(2, loss=2)] * 4
        self.assertNotIn("rising_training_loss", self.assess(duplicated))

    # Distinct windows retain raw values and do not attribute a trend to an optimal LR.
    def test_sustained_loss_with_supporting_evidence(self):
        records = [record(i, loss=value, learning_rate=0.001) for i, value in enumerate([1, 1, 2, 2], 1)]
        before = copy.deepcopy(records)
        findings = self.assess(records)
        self.assertEqual(findings["rising_training_loss"]["evidence"]["loss"]["previous"]["step_range"], [1, 2])
        self.assertEqual(findings["rising_training_loss"]["evidence"]["loss"]["recent"]["values"], [2, 2])
        self.assertIn("separate run", findings["rising_training_loss"]["recommendation"])
        self.assertEqual(records, before)
        self.assertNotIn("rising_training_loss", self.assess(records, warmup_steps=2))

    # On-policy losses and a noisy crossing of the old median do not support convergence claims.
    def test_noisy_and_onpolicy_losses(self):
        noisy = [record(i, loss=value) for i, value in enumerate([1, 1, 0.5, 4], 1)]
        self.assertNotIn("rising_training_loss", self.assess(noisy))
        increasing = [record(i, loss=i) for i in range(1, 5)]
        for method in ("grpo", "rloo", "distillation"):
            self.assertNotIn("rising_training_loss", self.assess(increasing, method))

    # Overfitting needs repeated evaluation deterioration and aligned train improvement.
    def test_possible_overfit_only_aligned_evidence(self):
        records = [record(1, True, eval_loss=1), record(1, loss=4), record(2, loss=3),
                   record(3, True, eval_loss=2), record(3, loss=2), record(4, loss=1),
                   record(4, True, eval_loss=3)]
        self.assertIn("possible_overfitting", self.assess(records))
        self.assertNotIn("possible_overfitting", self.assess(records[:-1]))
        eval_only = [r for r in records if r["eval"]]
        findings = self.assess(eval_only)
        self.assertNotIn("possible_overfitting", findings)
        self.assertIn("worsening_evaluation", findings)

    # Nonmonotonic held-out observations do not imply sustained worsening.
    def test_noisy_evaluation(self):
        self.assertNotIn("worsening_evaluation", self.assess([
            record(1, True, eval_loss=1), record(2, True, eval_loss=0.5), record(3, True, eval_loss=2)]))

    # Reward health is not inferred from scalar reward growth or hidden display selection.
    def test_reward_health_and_cutoffs(self):
        records = [record(i, **{"rewards/example/std": 0.0, "frac_reward_zero_std": 1.0,
                                "completions/clipped_ratio": 0.2}) for i in (1, 2)]
        findings = self.assess(records, "grpo", ranges={})
        self.assertIn("constant_reward:rewards/example/std", findings)
        self.assertEqual(findings["zero_variance_groups"]["severity"], "warning")
        self.assertEqual(findings["generation_cutoffs"]["severity"], "info")
        self.assertNotIn("zero_variance_groups", self.assess(records, "distillation"))

    # Policy metrics are joint descriptive evidence, not a collapse verdict.
    def test_policy_evidence(self):
        records = [record(i, kl=i, entropy=5-i, **{"clip_ratio/region_mean": i / 10}) for i in range(1, 5)]
        evidence = self.assess(records, "rloo")["policy_dynamics"]["evidence"]
        self.assertEqual(set(evidence), {"kl", "entropy", "clip_ratio/region_mean"})

    # User-authored ranges support only claims about those actual bounds.
    def test_configured_ranges(self):
        findings = self.assess([record(1, custom=4), record(2, custom=5)], ranges={"custom": (0, 3)})
        self.assertEqual(findings["configured_range:custom"]["evidence"]["configured_bounds"], [0, 3])

    # A changed evaluation condition hash prevents baseline comparisons across datasets.
    def test_quality_series_and_baseline(self):
        baseline = record(0, **{"quality/accuracy": 0.5})
        baseline["quality"] = {"phase": "baseline", "series": "one", "rows": 10, "status": "complete"}
        current = record(5, **{"quality/accuracy": 0.7})
        current["quality"] = {"phase": "completion", "series": "two", "rows": 10, "status": "complete"}
        self.assertFalse(any(key.startswith("quality_baseline:") for key in self.assess([baseline, current])))
        current["quality"]["series"] = "one"
        evidence = self.assess([baseline, current])["quality_baseline:one"]["evidence"]
        self.assertAlmostEqual(evidence["metrics"]["quality/accuracy"]["delta"], 0.2)

    # Enough comparable quality observations support advice independently of training reward.
    def test_quality_deterioration_and_judge_attribution(self):
        rounds = []
        for step, score in enumerate((0.9, 0.6, 0.3)):
            row = record(step, **{"quality/judge_score": score, "quality/word_count": 10 - step})
            row["quality"] = {"series": "judge", "phase": "baseline" if step == 0 else "scheduled", "status": "complete"}
            rounds.append(row)
        findings = self.assess(rounds, "grpo")
        self.assertTrue(findings["quality_deterioration:judge:judge_score"]["evidence"]["model_judgment"])
        self.assertNotIn("quality_deterioration:judge:word_count", findings)

    # Failed partial scores neither establish a baseline nor count as an evaluation.
    def test_failed_quality_rounds_are_excluded(self):
        rounds = []
        for step, score in enumerate((0.9, 0.6, 0.0)):
            row = record(step, **{"quality/accuracy": score})
            row["quality"] = {"series": "one", "phase": "baseline" if step == 0 else "scheduled",
                              "status": "failed" if step == 2 else "complete"}
            rounds.append(row)
        findings = self.assess(rounds)
        evidence = findings["quality_baseline:one"]["evidence"]
        self.assertEqual(evidence["current_step"], 1)
        self.assertEqual(evidence["metrics"]["quality/accuracy"]["current"], 0.6)
        self.assertNotIn("quality_deterioration:one:accuracy", findings)
        rounds[0]["quality"]["status"] = "failed"
        self.assertNotIn("quality_baseline:one", self.assess(rounds))
        del rounds[0]["quality"]["status"]
        self.assertNotIn("quality_baseline:one", self.assess(rounds))

    # Flatness must cover every observation; medians alone hide oscillations.
    def test_plateau_and_distillation_progress(self):
        self.assertIn("little_loss_change", self.assess([record(i, loss=1) for i in range(1, 5)]))
        self.assertNotIn("little_loss_change", self.assess([
            record(i, loss=value) for i, value in enumerate((0.1, 2, 0.1, 2), 1)]))
        self.assertIn("distillation_loss_trend", self.assess([
            record(i, loss=value) for i, value in enumerate((4, 3, 2, 1), 1)], "distillation"))


if __name__ == "__main__":
    unittest.main()
