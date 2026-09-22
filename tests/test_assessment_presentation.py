"""Assessment evidence stays visible and saved-checkpoint claims are verified."""

import contextlib
import io
import pathlib
import types
import unittest
from unittest.mock import patch

from trlx import TrlxError, assessment, metrics, review, run_dirs, show


class Presentation(unittest.TestCase):
    # Distinct evidence sections expose accidental suppression or duplicate support.
    def report(self):
        return dict(method="sft", completed_steps=32, decision="Training budget complete.",
                    final_decision="Compare the saved checkpoints.", support="Combined duplicate support.",
                    final_support="Combined duplicate support.", issues=[], final_issues=[],
                    outcome="Held-out loss improved overall.", recent="Recent gains are diminishing.",
                    schedule="Learning rate is decreasing.", quality=["Independent quality worsened."],
                    checkpoint="Best measured step 30 has no saved checkpoint.")

    # An unrelated range warning cannot hide the outcome or its own measured cause.
    def test_warning_keeps_outcome_and_its_own_evidence(self):
        report = self.report()
        report["final_issues"] = [dict(code="custom_range", message="Outside configured range.",
                                      support="Gradient norm 2 exceeds configured maximum 1.")]
        text = review.render_run_assessment(report, types.SimpleNamespace(), final=True, width=120)
        for expected in (report["final_decision"], report["outcome"], report["recent"], report["schedule"],
                         report["quality"][0], report["checkpoint"], "Outside configured range.",
                         "Gradient norm 2 exceeds configured maximum 1."):
            self.assertIn(expected, text)
        self.assertNotIn("Combined duplicate support", text)

    # Rendering must preserve lifecycle-aware advice supplied by the assessment.
    def test_completed_evaluation_does_not_invent_continue_advice(self):
        report = self.report()
        report["decision"] = "No changes recommended."
        text = review.render_run_assessment(report, types.SimpleNamespace(), width=120)
        self.assertIn("No changes recommended.", text)
        self.assertNotIn("Continue", text)


class CheckpointEvidence(unittest.TestCase):
    # Incomplete saved artifacts qualify availability without suppressing the report.
    def test_invalid_checkpoint_is_excluded_without_suppressing_assessment(self):
        callback = metrics.callback_class()(pathlib.Path(__file__).parent, assessment_settings=object(), method="sft")
        state = types.SimpleNamespace(global_step=32, max_steps=32)
        checkpoints = [types.SimpleNamespace(step=20), types.SimpleNamespace(step=32)]
        output = io.StringIO()
        with patch.object(show, "_checkpoints", return_value=checkpoints), \
             patch.object(run_dirs, "inspect_checkpoint", side_effect=[types.SimpleNamespace(step=20),
                                                                       TrlxError("missing weights")]), \
             patch.object(assessment, "run_assessment", return_value={}) as assess, \
             patch.object(review, "render_run_assessment", return_value="Assessment rendered"), \
             contextlib.redirect_stdout(output):
            callback._report_assessment(types.SimpleNamespace(), state, final=True)
        self.assertEqual(assess.call_args.kwargs["checkpoint_steps"], [20])
        self.assertEqual(assess.call_args.kwargs["checkpoint_errors"], ["missing weights"])
        self.assertIn("Assessment rendered", output.getvalue())
        self.assertNotIn("assessment unavailable", output.getvalue())
