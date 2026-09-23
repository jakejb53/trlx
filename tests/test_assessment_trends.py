"""Loss history remains factual, complete, and faithful to measurement spacing."""

import unittest

from trlx import assessment


class LossHistoryTests(unittest.TestCase):
    # Explicit steps expose accidental replacement of optimizer spacing with observation indexes.
    def report(self, values, steps):
        rows = [{"step": step, "eval": True, "log": {"eval_loss": value}}
                for step, value in zip(steps, values)]
        return assessment.run_metrics_report("sft", rows, completed_steps=steps[-1], planned_steps=70)

    # Later reversals must not truncate earlier evidence or generate diagnostic conclusions.
    def test_full_history_survives_late_reversal_and_irregular_spacing(self):
        values = [3., 1.2, 1., .8, .81, .83, .86]
        steps = [0, 5, 10, 20, 35, 60, 70]
        result = self.report(values, steps)
        self.assertEqual(result["curves"]["evaluation"], list(zip(steps, values)))
        self.assertEqual(result["curves"]["training"], [])
        for key in ("interpretation", "final_interpretation", "findings", "decision"):
            self.assertNotIn(key, result)

    # Recovery preserves the missing segment so the chart cannot imply continuity through failure.
    def test_nonfinite_gap_is_preserved_after_recovery(self):
        result = self.report([3., 2., float("nan"), .8, .7], [0, 5, 10, 20, 70])
        self.assertEqual(result["curves"]["evaluation"], [(0, 3.), (5, 2.), (10, None), (20, .8), (70, .7)])
        self.assertIn("non-finite", " ".join(result["notices"]).lower())

    # An invalid final measurement remains invalid in both the chart and numeric recap.
    def test_nonfinite_endpoint_is_not_replaced_by_earlier_finite_value(self):
        result = self.report([3., 2., float("inf")], [0, 5, 10])
        self.assertEqual(result["curves"]["evaluation"][-1], (10, None))
        row = next(row for row in result["recap"] if row["metric"] == "eval_loss")
        self.assertIsNone(row["latest"])
        self.assertEqual(row["latest_step"], 10)
