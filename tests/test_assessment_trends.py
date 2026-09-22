"""Recent evidence must not borrow confidence from an earlier training phase."""

import unittest

from trlx.assessment import _metric_trend, runtime_findings


# Build ordinary evaluation observations with explicit optimizer-step positions.
def records(values, steps=None, key="eval_loss"):
    steps = range(len(values)) if steps is None else steps
    return [{"step": step, "eval": True, "log": {key: value}}
            for step, value in zip(steps, values)]


class AssessmentTrendTests(unittest.TestCase):
    # Late reversal must remain visible despite large earlier gains.
    def test_recent_reversal_is_not_hidden_by_early_gains(self):
        trend = _metric_trend(records([3, 1.2, 1, .8, .81, .83, .86]), "eval_loss")
        self.assertEqual(trend["direction"], "up")
        self.assertEqual(trend["recent"]["step_range"], [3, 6])
        self.assertEqual(trend["history"]["values"], [3, 1.2, 1, .8, .81, .83, .86])

    # Single jumps and alternating changes do not establish sustained movement.
    def test_one_outlier_or_one_shift_does_not_establish_sustained_direction(self):
        for values in ([1, 1, .5, 4], [1, .5, 2], [1, 1, 2, 2], [1, 2, 1, 2]):
            self.assertEqual(_metric_trend(records(values), "eval_loss")["direction"], "mixed")

    # Normalize changes by actual training updates, not observation indexes.
    def test_diminishing_changes_use_actual_step_spacing(self):
        trend = _metric_trend(records([2, 1.5, 1.25, 1.125], [0, 5, 10, 20]), "eval_loss")
        self.assertTrue(trend["diminishing"])
        self.assertEqual([interval["rate"] for interval in trend["intervals"]], [-.1, -.05, -.0125])

    # A short final interval can fluctuate within an otherwise slower recent phase.
    def test_short_final_interval_does_not_hide_diminishing_gains(self):
        trend = _metric_trend(records([2.655, 1.211, 1.050, .970, .939, .929, .927, .926],
                                      [0, 5, 10, 15, 20, 25, 30, 32]), "eval_loss")
        self.assertTrue(trend["diminishing"])
        comparison = trend["rate_comparison"]
        self.assertEqual(comparison["previous_step_range"], [5, 20])
        self.assertEqual(comparison["recent_step_range"], [20, 32])
        self.assertGreater(abs(comparison["recent"][-1]), abs(comparison["recent"][-2]))

    # Recovery cannot borrow trend confidence from before a numerical failure.
    def test_nonfinite_gap_resets_recent_evidence_but_preserves_baseline(self):
        trend = _metric_trend(records([3, 2, 1, float("nan"), .8, .7]), "eval_loss")
        self.assertEqual(trend["direction"], "limited")
        self.assertEqual(trend["recent"]["step_range"], [4, 5])
        self.assertEqual(trend["history"]["values"][0], 3)

    # An unresolved numerical failure supplies no current trend direction.
    def test_unresolved_nonfinite_has_no_recent_direction(self):
        trend = _metric_trend(records([3, 2, 1, float("inf")]), "eval_loss")
        self.assertEqual(trend["direction"], "unknown")
        self.assertEqual(trend["intervals"], [])

    # Warmup observations do not inflate post-warmup confidence.
    def test_warmup_does_not_establish_post_warmup_confidence(self):
        trend = _metric_trend(records([3, 2, 1, .8, .7]), "eval_loss", warmup_steps=2)
        self.assertEqual(trend["direction"], "limited")
        self.assertEqual(trend["post_warmup"]["step_range"], [3, 4])

    # Ordinary evaluation is optional when independent task evaluation exists.
    def test_complete_quality_evidence_does_not_demand_ordinary_evaluation(self):
        rows = [{"step": 0, "quality": {"series": "a", "status": "complete", "phase": "baseline"},
                 "log": {"quality/accuracy": .5}}]
        missing = next(f for f in runtime_findings("sft", rows) if f["code"] == "evaluation_missing")
        self.assertEqual(missing["severity"], "info")

    # Failed quality rounds never contribute scores to a matching complete series.
    def test_quality_improvement_requires_matching_complete_rounds(self):
        rows = [{"step": step, "quality": {"series": "a", "status": "complete",
                                           "phase": "baseline" if step == 0 else "evaluation"},
                 "log": {"quality/accuracy": value}}
                for step, value in enumerate([.5, .6, .7])]
        rows.append({"step": 3, "quality": {"series": "a", "status": "failed"},
                     "log": {"quality/accuracy": 0}})
        findings = runtime_findings("sft", rows)
        self.assertTrue(any(f["code"] == "quality_improvement:a:accuracy" for f in findings))
        self.assertFalse(any(f["code"].startswith("quality_deterioration:") for f in findings))
