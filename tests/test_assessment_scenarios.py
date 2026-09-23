"""Metric comparisons retain measurements without generating runtime advice."""

import copy
import unittest

from trlx import assessment, review


# Use callback-shaped records with explicit optimizer steps, including step-zero baselines.
def losses(values, steps):
    return [{"step": step, "eval": True, "log": {"eval_loss": value}}
            for step, value in zip(steps, values)]


# Comparable quality scores require both matching series and a completed round.
def quality(step, score, *, series="same", status="complete", phase="scheduled"):
    return {"step": step, "eval": True,
            "quality": {"series": series, "status": status, "phase": phase},
            "log": {"quality/accuracy": score}}


class MetricsScenarios(unittest.TestCase):
    # Every scenario checks that report construction leaves authoritative measurements untouched.
    def report(self, records, method="sft", completed=None):
        completed = max(row["step"] for row in records) if completed is None else completed
        original = copy.deepcopy(records)
        result = assessment.run_metrics_report(method, records, completed_steps=completed, planned_steps=70)
        self.assertEqual(records, original)
        return result

    # The supplied reversal run must retain all history at every boundary without interpretation prose.
    def test_complete_user_run_is_retained_at_every_evaluation(self):
        values = [2.68992, 1.53691, 1.39804, 1.33573, 1.31394, 1.30301, 1.29404,
                  1.28905, 1.28791, 1.29056, 1.28809, 1.29544, 1.29964, 1.30676, 1.30562]
        steps = list(range(0, 71, 5))
        for end in range(1, len(values) + 1):
            with self.subTest(step=steps[end - 1]):
                result = self.report(losses(values[:end], steps[:end]))
                self.assertEqual(result["curves"]["evaluation"], list(zip(steps[:end], values[:end])))
                recap = {row["metric"]: row for row in result["recap"]}
                recent = {row["metric"]: row for row in result["recent_recap"]}
                self.assertEqual(recap["eval_loss"]["initial"], values[0])
                self.assertEqual(recap["eval_loss"]["latest"], values[end - 1])
                if end > 1:
                    self.assertEqual(recent["eval_loss"]["initial_step"], steps[end - 2])
                for final in (False, True):
                    rendered = review.render_metrics_report(result, final=final)
                    for prose in ("overfitting", "corroborates", "Continue", "Monitor", "learning rate fell",
                                  "learning extended", "checkpoint"):
                        self.assertNotIn(prose, rendered)

    # Before a previous training endpoint exists, use the first logged loss instead of unavailable.
    def test_first_live_training_comparison_uses_first_logged_measurement(self):
        records = losses([2.7, 1.5], [0, 5])
        records += [{"step": step, "eval": False, "log": {"loss": value}}
                    for step, value in ((1, 2.6), (2, 2.1), (5, 1.55))]
        records.sort(key=lambda row: row["step"])
        result = self.report(records)
        row = next(row for row in result["recent_recap"] if row["metric"] == "loss")
        self.assertEqual((row["initial_step"], row["initial"]), (1, 2.6))
        self.assertEqual((row["latest_step"], row["latest"]), (5, 1.55))

    # Trainer's whole-run average is not an endpoint measurement and cannot extend the curve.
    def test_summary_average_does_not_replace_last_logged_training_loss(self):
        records = [{"step": 1, "eval": False, "log": {"loss": 2.6}},
                   {"step": 7, "eval": False, "log": {"loss": 1.1}},
                   {"step": 10, "eval": False, "log": {"train_loss": 1.8}}]
        result = self.report(records)
        row = next(row for row in result["recap"] if row["metric"] == "loss")
        self.assertEqual(row["initial_kind"], "first logged")
        self.assertEqual((row["initial_step"], row["latest_step"]), (1, 7))
        self.assertEqual(row["latest"], 1.1)
        self.assertEqual(result["curves"]["training"], [(1, 2.6), (7, 1.1)])

    # A first evaluation after training began cannot masquerade as a starting-model baseline.
    def test_missing_baseline_does_not_invent_initial_measurement(self):
        result = self.report(losses([2., 1.8], [5, 10]))
        row = next(row for row in result["recap"] if row["metric"] == "eval_loss")
        self.assertIsNone(row["initial"])
        self.assertIsNone(row["relative_change"])

    # Sparse accuracy logs keep their true measurement step and percentage-point units.
    def test_accuracy_changes_are_percentage_points_and_keep_actual_step(self):
        records = losses([2.7, 1.5, 1.4], [0, 5, 10])
        records[0]["log"]["eval_mean_token_accuracy"] = .503232
        records[1]["log"]["eval_mean_token_accuracy"] = .642926
        result = self.report(records)
        row = next(row for row in result["recap"] if row["metric"] == "eval_mean_token_accuracy")
        self.assertTrue(row["percentage"])
        self.assertEqual(row["latest_step"], 5)
        self.assertAlmostEqual(row["delta"] * 100, 13.9694)

    # A later completed round with a missing score cannot silently reuse an older score.
    def test_quality_matching_complete_rounds_and_missing_score(self):
        records = [quality(0, .5, phase="baseline"), quality(5, .8)]
        result = self.report(records)
        row = next(row for row in result["recap"] if row["metric"] == "quality/accuracy")
        self.assertEqual((row["initial"], row["latest"]), (.5, .8))
        latest = quality(10, .9)
        latest["log"] = {}
        result = self.report(records + [latest])
        row = next(row for row in result["recap"] if row["metric"] == "quality/accuracy")
        self.assertIsNone(row["latest"])
        self.assertEqual(row["latest_step"], 10)

    # Incomplete results and changed scoring conditions cannot create a baseline comparison.
    def test_failed_or_changed_quality_series_does_not_borrow_baseline(self):
        for latest in (quality(10, .99, status="failed"), quality(10, .9, series="new")):
            result = self.report([quality(0, .5, phase="baseline"), latest])
            rows = [row for row in result["recap"] if row["metric"] == "quality/accuracy"]
            self.assertFalse(any(row["initial"] == .5 and row["latest"] in (.9, .99) for row in rows))

    # Execution counters are not model quality measurements.
    def test_quality_recap_excludes_infrastructure_counts(self):
        records = [quality(0, .5, phase="baseline"), quality(5, .7)]
        for row in records:
            row["log"].update({"quality/rows": 12, "quality/tokens": 120,
                               "quality/metric_rows/accuracy": 12})
        result = self.report(records)
        for field in ("recap", "recent_recap"):
            names = {row["metric"] for row in result[field]}
            self.assertIn("quality/accuracy", names)
            self.assertTrue(names.isdisjoint({"quality/rows", "quality/tokens", "quality/metric_rows/accuracy"}))

    # Different objectives retain the same factual reporting contract without inferred diagnoses.
    def test_all_methods_report_losses_without_objective_diagnoses(self):
        for method in ("sft", "dpo", "kto", "reward", "grpo", "rloo", "distillation"):
            with self.subTest(method=method):
                result = self.report(losses([2., 1., 1.5], [0, 5, 10]), method)
                self.assertEqual(result["method"], method)
                self.assertEqual(result["curves"]["evaluation"][-1], (10, 1.5))
                self.assertNotIn("findings", result)
