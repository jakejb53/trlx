"""Measured facts and deterministic interpretation, without running training."""

import copy
import unittest
from types import SimpleNamespace

from trlx import assessment, review


# Explicit resolved settings keep interpretation independent of operational defaults.
def arguments(**changes):
    values = dict(learning_rate=.0002, max_steps=-1, num_train_epochs=8, lr_scheduler_type="linear",
                  eval_strategy="steps", eval_steps=5, logging_strategy="steps", logging_steps=1,
                  bf16=True, fp16=False, max_completion_length=128, num_generations=4, temperature=1.,
                  get_warmup_steps=lambda steps: 0)
    return SimpleNamespace(**(values | changes))


# Build the same immutable nested metric envelopes the real callback writes.
def losses(values, steps=None):
    steps = steps if steps is not None else range(len(values))
    return [{"step": step, "eval": True, "log": {"eval_loss": value}} for step, value in zip(steps, values)]


# Quality comparisons require matching conditions and complete rounds.
def quality(step, score, *, series="same", status="complete", phase="scheduled"):
    return {"step": step, "eval": True, "quality": {"series": series, "status": status, "phase": phase},
            "log": {"quality/accuracy": score}}


class AssessmentScenarios(unittest.TestCase):
    # Render the actual report so tests catch evidence lost by presentation.
    def report(self, records, method="sft", completed=None, planned=None, args=None, checkpoints=None):
        args = args or arguments()
        completed = max(r["step"] for r in records) if completed is None else completed
        planned = completed if planned is None else planned
        original = copy.deepcopy(records)
        report = assessment.run_assessment(method, records, args, completed_steps=completed,
                    planned_steps=planned, checkpoint_steps=checkpoints)
        rendered = review.render_run_assessment(report, args, final=True, width=160)
        self.assertEqual(records, original)
        return report, " ".join(rendered.split())

    # The user's run has a short last interval; rates, not raw interval deltas, establish slowing gains.
    def test_real_diminishing_run_does_not_prescribe_another_epoch(self):
        steps = [0, 5, 10, 15, 20, 25, 30, 32]
        records = losses([2.655, 1.211, 1.050, .970, .939, .9291242361, .9269463420, .9259175658], steps)
        records += [{"step": step, "eval": False, "log": {"learning_rate": value}}
                    for step, value in ((20, .00008125), (25, .00005), (30, .00001875), (32, .00000625))]
        records.sort(key=lambda r: r["step"])
        report, rendered = self.report(records, checkpoints=[28, 32])
        self.assertIn("diminishing", " ".join(report["final_interpretation"]).lower())
        self.assertIn("32", rendered)
        self.assertNotIn("cannot separate", rendered)
        self.assertNotIn("Checkpoint-32 is available", rendered)
        self.assertNotIn("--num-train-epochs 9", rendered)
        self.assertNotIn("Continue", rendered)

    # A tiny monotone decline is measurable but not evidence for a worthwhile duration increase.
    def test_tiny_decline_is_not_hidden_or_prescribed_as_more_training(self):
        records = losses([1., .99999, .99998, .99997])
        epoch, rendered = self.report(records)
        steps, rendered_steps = self.report(records, args=arguments(max_steps=3))
        self.assertEqual(epoch["final_interpretation"], steps["final_interpretation"])
        self.assertIn("0.99997", rendered)
        self.assertNotIn("does not establish a useful extension", rendered)
        self.assertNotIn("--max-steps 6", rendered_steps)

    # Early gains cannot conceal repeated late deterioration.
    def test_late_reversal_and_checkpoint_availability(self):
        report, rendered = self.report(losses([3., 1.2, 1., .8, .81, .83, .86]), checkpoints=[2, 6])
        self.assertRegex(" ".join(report["final_interpretation"]).lower(), "ris|worsen|deteriorat")
        self.assertNotIn("No verified saved checkpoint", rendered)
        self.assertNotIn("available checkpoints", rendered)
        self.assertNotIn("--max-steps 3", rendered)

    # Different train/evaluation behavior is evidence for inspection, not a causal LR diagnosis.
    def test_rising_training_loss_does_not_override_improving_evaluation(self):
        records = losses([2., 1.8, 1.6, 1.4])
        records += [{"step": i, "eval": False, "log": {"loss": value}}
                    for i, value in enumerate([1., 1.1, 1.2, 1.3])]
        records.sort(key=lambda r: r["step"])
        report, rendered = self.report(records)
        self.assertNotIn("--learning-rate 0.0001", rendered)
        self.assertFalse(any(i["code"] == "rising_training_loss" for i in report["issues"]))
        self.assertRegex(" ".join(report["final_interpretation"]).lower(), "improv|held-out")

    # A display-range warning cannot replace the result or lose the warning's own measurement.
    def test_warnings_keep_outcome_and_support(self):
        args = arguments()
        report = assessment.run_assessment("sft", losses([2., 1.8, 1.6, 1.4]), args,
                   completed_steps=3, planned_steps=3, ranges={"eval_loss": [0, 1]})
        rendered = " ".join(review.render_run_assessment(report, args, final=True, width=160).split())
        for interpretation in report["final_interpretation"]:
            self.assertIn(interpretation, rendered)
        self.assertTrue(any(row["metric"] == "eval_loss" for row in report["recap"]))
        self.assertIn("eval_loss at step 3: 1.4", rendered)
        self.assertIn("Configured range: [0, 1]", rendered)

    # Complete task regressions cannot be overridden by improved loss or training reward.
    def test_conflicting_quality_is_not_overruled_by_loss(self):
        records = losses([2., 1.8, 1.6, 1.4]) + [quality(0, .8, phase="baseline"), quality(3, .7)]
        report, rendered = self.report(records)
        self.assertIn("conflict", " ".join(report["final_interpretation"]).lower())
        self.assertIn("quality/accuracy", rendered)
        self.assertRegex(rendered, "0.8|80")
        self.assertRegex(rendered, "0.7|70")

    # Quality improvement is scoped to completed rounds under matching conditions.
    def test_quality_improvement_and_failed_latest_round(self):
        records = [quality(0, .5, phase="baseline"), quality(2, .7), quality(3, .8)]
        report, rendered = self.report(records, method="grpo")
        self.assertRegex(" ".join(report["final_interpretation"]).lower(), "(quality|task).*improv")
        self.assertTrue(any(row["metric"] == "quality/accuracy" for row in report["recap"]))
        report, rendered = self.report(records + [quality(4, .99, status="failed")], method="grpo")
        self.assertRegex(rendered.lower(), "incomplete|failed")
        self.assertNotIn("0.99", rendered)

    # A completed latest round that omits a score cannot reuse an earlier successful score.
    def test_latest_quality_round_missing_metric_does_not_reuse_score(self):
        latest = quality(4, .9)
        latest["log"] = {}
        records = [quality(0, .5, phase="baseline"), quality(2, .8), latest]
        report, rendered = self.report(records, method="grpo")
        rows = [row for row in report["recap"] if row["metric"] == "quality/accuracy"]
        self.assertTrue(rows)
        self.assertIsNone(rows[0]["latest"])
        self.assertEqual(rows[0]["latest_step"], 4)
        self.assertNotRegex(" ".join(report["final_interpretation"]).lower(),
                            "(quality|task) scores improved")

    # Changing the scorer/dataset series cannot turn unmatched scores into a baseline comparison.
    def test_mismatched_quality_and_stale_endpoint(self):
        records = losses([2., 1.8, 1.6, 1.4]) + [quality(0, .5, phase="baseline"), quality(3, .8, series="new")]
        report, rendered = self.report(records, completed=4, planned=4)
        self.assertRegex(rendered.lower(), "step 4.*(not evaluated|no evaluation)|latest.*step 3")
        self.assertRegex(rendered.lower(), "no completed.*comparison")

    # Successful task comparisons cannot hide absent loss evaluation or a worse baseline loss.
    def test_positive_quality_retains_loss_conflicts(self):
        records = losses([1., 2., 1.8, 1.6]) + [quality(0, .5, phase="baseline"), quality(3, .8)]
        report, rendered = self.report(records)
        self.assertIn("conflict", " ".join(report["final_interpretation"]).lower())
        records.append(quality(4, .9))
        report, rendered = self.report(records, completed=4, planned=4)
        self.assertRegex(rendered.lower(), "step 4.*(not evaluated|no evaluation)|latest.*step 3")

    # Failed or baseline-only independent checks are not a completed quality comparison.
    def test_unavailable_quality_does_not_invent_measurements(self):
        for item in (quality(3, .8, status="failed"), quality(0, .8, phase="baseline")):
            report, rendered = self.report([item], completed=3, planned=3)
            self.assertRegex(rendered.lower(), "no.*(comparison|evaluation)|unavailable|failed")
            self.assertNotIn("Use the independent quality measurements", rendered)

    # An observed schedule alone does not require a causal disclaimer.
    def test_learning_rate_context_uses_interior_observations(self):
        records = losses([2., 1.8, 1.6, 1.4])
        records += [{"step": i, "eval": False, "log": {"learning_rate": value}}
                    for i, value in enumerate([.001, .01, .01, .001])]
        records.sort(key=lambda r: r["step"])
        report, rendered = self.report(records)
        self.assertNotIn("cannot separate schedule effects", rendered)
        self.assertNotIn("Changing duration can change", rendered)
        self.assertNotIn("learning rate stayed", rendered)

    # Preference loss cannot override accuracy regression; policy methods do not optimize monotone loss.
    def test_method_specific_objectives(self):
        for method in ("dpo", "kto", "reward", "grpo", "rloo", "distillation"):
            records = losses([2., 1.8, 1.6, 1.4])
            records += [{"step": i, "eval": False, "log": {"loss": float(i), "reward": float(i)}} for i in range(4)]
            records.sort(key=lambda r: r["step"])
            report, rendered = self.report(records, method=method)
            self.assertNotIn("--learning-rate 0.0001", rendered)
            self.assertNotIn("--num-train-epochs 9", rendered)
        records = losses([2., 1.8, 1.6, 1.4])
        for row, accuracy in zip(records, [.8, .7, .6, .5]):
            row["log"]["eval_rewards/accuracies"] = accuracy
        report, rendered = self.report(records, method="dpo")
        self.assertIn("conflict", " ".join(report["final_interpretation"]).lower())
        self.assertTrue(any(row["metric"] == "eval_rewards/accuracies" for row in report["recap"]))
        self.assertIn("Evaluation preference accuracy", rendered)

    # Numerical issues and insufficient post-warmup data must not produce confident tuning advice.
    def test_warmup_nonfinite_and_short_run(self):
        report, rendered = self.report(losses([2., 1.8, 1.6]), args=arguments(get_warmup_steps=lambda n: n))
        self.assertNotIn("--learning-rate", rendered)
        report, rendered = self.report(losses([2., float("nan"), 1.6]), completed=2, planned=5)
        self.assertIn("2/5", rendered)
        self.assertIn("non-finite", rendered.lower())
        self.assertNotIn("Continue", rendered)

    # Zero variance is not zero reward; cutoff advice requires inspecting the actual answers.
    def test_policy_diagnostics_keep_their_meaning(self):
        records = [{"step": i, "eval": False, "log": {"reward": 10., "reward_std": 0.,
                    "completions/clipped_ratio": .1}} for i in range(3)]
        report, rendered = self.report(records, method="grpo")
        self.assertIn("no within-group variation", rendered)
        self.assertNotIn("reward is zero", rendered)
        self.assertRegex(rendered.lower(), "clip|cutoff|cut short")
        self.assertNotIn("Increase --max-completion-length", rendered)

    # The supplied run distinguishes a latest interval from repeated baseline narration.
    def test_user_run_recap_and_intermediate_interpretation(self):
        records = losses([2.68992, 1.55002, 1.42757, 1.3451], [0, 5, 10, 20])
        for row, accuracy in zip(records, [.503232, .642926, .656229, .66597]):
            row["log"]["eval_mean_token_accuracy"] = accuracy
        for endpoint, initial_step, initial_loss in ((5, 0, 2.68992), (10, 5, 1.55002)):
            with self.subTest(endpoint=endpoint):
                args = arguments()
                report = assessment.run_assessment("sft", records[:2 if endpoint == 5 else 3], args,
                                                   completed_steps=endpoint, planned_steps=20)
                recent = {row["metric"]: row for row in report["recent_recap"]}
                self.assertEqual(recent["eval_loss"]["initial_step"], initial_step)
                self.assertEqual(recent["eval_loss"]["initial"], initial_loss)
                self.assertEqual(recent["eval_loss"]["latest_step"], endpoint)
                interpretation = " ".join(report["interpretation"]).lower()
                self.assertRegex(interpretation, "held-out|prediction|learning")
                self.assertRegex(interpretation, "improv|learn")
                if endpoint == 10:
                    self.assertIn("less improvement per optimizer update", interpretation)
                rendered = review.render_run_assessment(report, args, width=100)
                for boilerplate in ("Monitor", "Continue with", "one interval only", "cannot separate",
                                    "No verified saved checkpoint", "Changing duration"):
                    self.assertNotIn(boilerplate, rendered)
        report, rendered = self.report(records)
        recap = {row["metric"]: row for row in report["recap"]}
        self.assertEqual(recap["eval_loss"]["initial"], 2.68992)
        self.assertEqual(recap["eval_loss"]["latest"], 1.3451)
        self.assertAlmostEqual(recap["eval_loss"]["relative_change"], (1.3451 - 2.68992) / 2.68992)
        accuracy = recap["eval_mean_token_accuracy"]
        self.assertTrue(accuracy["percentage"])
        self.assertAlmostEqual(accuracy["delta"] * 100, 16.2738)
        self.assertRegex(" ".join(report["final_interpretation"]).lower(), "diminish|smaller|slow")
        self.assertIn("93.9%", rendered)
        self.assertNotIn("decision", report)
        self.assertNotIn("final_decision", report)

    # Whole-run train_loss is an average, not the final logged training loss.
    def test_training_recap_uses_logged_endpoints_and_exposes_staleness(self):
        records = [{"step": 1, "eval": False, "log": {"loss": 2.6}},
                   {"step": 7, "eval": False, "log": {"loss": 1.1}},
                   {"step": 10, "eval": False, "log": {"train_loss": 1.8}}]
        records += losses([2.7, 1.4], [0, 8])
        records.sort(key=lambda row: row["step"])
        report, rendered = self.report(records, completed=10, planned=10)
        recap = {row["metric"]: row for row in report["recap"]}
        self.assertEqual(recap["loss"]["initial_kind"], "first logged")
        self.assertEqual(recap["loss"]["initial_step"], 1)
        self.assertEqual(recap["loss"]["latest_step"], 7)
        self.assertEqual(recap["loss"]["latest"], 1.1)
        self.assertEqual(recap["eval_loss"]["latest_step"], 8)
        self.assertIn("first logged", rendered.lower())

    # Invalid endpoints stay missing; older finite values cannot masquerade as final measurements.
    def test_missing_baseline_and_nonfinite_latest_recap(self):
        report, rendered = self.report(losses([2., 1.8], [5, 10]))
        recap = {row["metric"]: row for row in report["recap"]}
        self.assertIsNone(recap["eval_loss"]["initial"])
        self.assertIsNone(recap["eval_loss"]["relative_change"])
        report, rendered = self.report(losses([2., 1.8, float("inf")], [0, 5, 10]))
        recap = {row["metric"]: row for row in report["recap"]}
        self.assertIsNone(recap["eval_loss"]["latest"])
        self.assertEqual(recap["eval_loss"]["latest_step"], 10)
        self.assertIsNone(recap["eval_loss"]["delta"])
        self.assertIn("non-finite", rendered.lower())

    # A recorded invalid baseline is a numerical failure, not an absent evaluation.
    def test_nonfinite_baseline_is_not_reported_as_unrecorded(self):
        report, rendered = self.report(losses([float("inf"), 1.8], [0, 5]))
        recap = {row["metric"]: row for row in report["recap"]}
        self.assertIsNone(recap["eval_loss"]["initial"])
        self.assertEqual(recap["eval_loss"]["initial_step"], 0)
        self.assertEqual(recap["eval_loss"]["latest"], 1.8)
        self.assertIn("non-finite", rendered.lower())
        self.assertNotRegex(rendered.lower(), "no step-zero.*was recorded")

    # Auxiliary measurements from an earlier endpoint cannot explain the latest loss change.
    def test_stale_accuracy_does_not_corroborate_or_conflict_with_latest_loss(self):
        for final_accuracy in (.7, .3):
            with self.subTest(final_accuracy=final_accuracy):
                records = losses([2., 1.8, 1.6], [0, 5, 10])
                records[0]["log"]["eval_mean_token_accuracy"] = .5
                records[1]["log"]["eval_mean_token_accuracy"] = final_accuracy
                report, rendered = self.report(records)
                recap = {row["metric"]: row for row in report["recap"]}
                self.assertEqual(recap["eval_mean_token_accuracy"]["latest_step"], 5)
                self.assertEqual(recap["eval_mean_token_accuracy"]["latest"], final_accuracy)
                for field in ("interpretation", "final_interpretation"):
                    interpretation = " ".join(report[field]).lower()
                    self.assertNotIn("conflict", interpretation)
                    self.assertNotIn("corroborat", interpretation)
                    self.assertNotIn("accuracy improved too", interpretation)

    # A stale quality interval cannot contradict improvement measured in a later interval.
    def test_stale_quality_does_not_invent_cross_objective_conflict(self):
        records = losses([2., 1.8, 1.6], [0, 5, 10])
        records += [quality(0, .8, phase="baseline"), quality(5, .5)]
        records.sort(key=lambda row: row["step"])
        report, rendered = self.report(records)
        for field in ("interpretation", "final_interpretation"):
            interpretation = " ".join(report[field]).lower()
            if field == "final_interpretation":
                self.assertIn("task scores deteriorated", interpretation)
            self.assertIn("last measured at step 5", interpretation)
            self.assertNotIn("conflict", interpretation)
            self.assertNotIn("did not carry over", interpretation)

    # Quality execution counts describe infrastructure, not the model's task performance.
    def test_quality_recap_excludes_infrastructure_counters(self):
        records = [quality(0, .5, phase="baseline"), quality(5, .7)]
        for row in records:
            row["log"].update({"quality/rows": 12, "quality/tokens": 120,
                               "quality/metric_rows/accuracy": 12})
        report, rendered = self.report(records)
        for field in ("recap", "recent_recap"):
            metrics = {row["metric"] for row in report[field]}
            self.assertIn("quality/accuracy", metrics)
            self.assertTrue(metrics.isdisjoint({"quality/rows", "quality/tokens", "quality/metric_rows/accuracy"}))

    # Repeated opposite trends over the same period distinguish overfitting from ineffective learning.
    def test_overfitting_requires_correlated_training_improvement(self):
        for training, expected in (([2., 1.8, 1.6, 1.4], True), ([2., 2.2, 2.4, 2.6], False)):
            with self.subTest(training=training):
                records = losses([1., 1.1, 1.2, 1.3])
                records += [{"step": i, "eval": False, "log": {"loss": value}}
                            for i, value in enumerate(training)]
                records.sort(key=lambda row: row["step"])
                report, rendered = self.report(records)
                codes = {finding["code"] for finding in report["findings"]}
                self.assertEqual("possible_overfitting" in codes, expected)
                if expected:
                    self.assertIn("overfitting", rendered.lower())
                self.assertNotIn("learning rate is too low", rendered.lower())
