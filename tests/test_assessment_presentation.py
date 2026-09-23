"""Metric recaps and interpretation survive rendering and checkpoint inspection."""

import contextlib
import io
import pathlib
import types
import unittest
from unittest.mock import patch

from trlx import TrlxError, assessment, metrics, review, run_dirs, show


class Presentation(unittest.TestCase):
    # Distinct final and recent facts expose accidental lifecycle mixing.
    def report(self):
        row = dict(metric="eval_loss", label="Evaluation loss", initial=2.5, latest=1.25,
                   initial_step=0, latest_step=32, initial_kind="baseline", delta=-1.25,
                   relative_change=-.5, percentage=False)
        return dict(method="sft", completed_steps=32, issues=[], recap=[row],
                    recent_recap=[row | dict(initial=1.5, initial_step=30, delta=-.25,
                                            relative_change=-1 / 6)],
                    interpretation=["The latest interval improved held-out predictions."],
                    final_interpretation=["Learning extended to held-out text.",
                                          "Later updates produced diminishing gains."])

    # An unrelated range warning cannot hide the outcome or its own measured cause.
    def test_warning_keeps_outcome_and_its_own_evidence(self):
        report = self.report()
        report["issues"] = [dict(code="custom_range", message="Outside configured range.",
                                support="Gradient norm 2 exceeds configured maximum 1.")]
        text = review.render_run_assessment(report, types.SimpleNamespace(), final=True, width=120)
        for expected in (*report["final_interpretation"], "Evaluation loss", "2.5", "1.25", "Outside configured range.",
                         "Gradient norm 2 exceeds configured maximum 1."):
            self.assertIn(expected, text)
        self.assertNotIn(report["interpretation"][0], text)

    # Live reports show the latest interval without inventing recommendations.
    def test_intermediate_selects_recent_facts_and_interpretation(self):
        report = self.report()
        text = review.render_run_assessment(report, types.SimpleNamespace(), width=120)
        self.assertIn(report["interpretation"][0], text)
        self.assertIn("1.5", text)
        self.assertNotIn("2.5", text)
        self.assertNotIn(report["final_interpretation"][0], text)
        self.assertNotIn("Continue", text)

    # A narrow terminal must retain all endpoints and complete interpretation.
    def test_narrow_render_preserves_facts(self):
        report = self.report()
        for width in (40, 80, 160):
            with self.subTest(width=width):
                text = review.render_run_assessment(report, types.SimpleNamespace(), final=True, width=width)
                normalized = " ".join(text.split())
                for expected in ("Evaluation loss", "2.5", "1.25", *report["final_interpretation"]):
                    self.assertIn(expected, normalized)


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
