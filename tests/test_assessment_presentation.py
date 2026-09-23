"""Fixed-size loss charts and factual recap presentation."""

import pathlib
import types
import unittest
from unittest.mock import Mock, patch

from trlx import assessment, metrics, review


class Presentation(unittest.TestCase):
    # Unequal baseline and recent endpoints distinguish live recap selection from final selection.
    def report(self):
        records = [{"step": 0, "eval": True, "log": {"eval_loss": 2.5}},
                   {"step": 30, "eval": True, "log": {"eval_loss": 1.5}},
                   {"step": 32, "eval": True, "log": {"eval_loss": 1.25}}]
        return assessment.run_metrics_report("sft", records, completed_steps=32, planned_steps=40)

    # Lifecycle changes the numeric comparison, never the amount of plotted history.
    def test_live_and_final_select_different_recap_endpoints_but_same_curves(self):
        report = self.report()
        live = review.render_metrics_report(report)
        final = review.render_metrics_report(report, final=True)
        self.assertIn("Previous", live)
        self.assertIn("Initial", final)
        for text in (live, final):
            self.assertIn("Evaluation loss", text)
            self.assertIn("1.25", text)
            self.assertNotIn("Continue", text)
            self.assertNotIn("interpretation", text.lower())
        chart = "\n".join(review._loss_charts(report["curves"], 32))
        self.assertIn(chart, live)
        self.assertIn(chart, final)

    # Removing analysis must not hide recorded numerical failures.
    def test_notices_preserve_factual_numerical_errors(self):
        report = self.report()
        report["notices"] = ["Non-finite eval_loss recorded at step 10."]
        text = review.render_metrics_report(report)
        self.assertIn(report["notices"][0], text)


class LossCharts(unittest.TestCase):
    # Extract plot interiors without coupling tests to numeric label precision.
    def panels(self, training, evaluation, step):
        lines = review._loss_charts({"training": training, "evaluation": evaluation}, step)
        rows = [line for line in lines if line.count("|") == 4]
        self.assertEqual(len(rows), 13)
        self.assertTrue(all(len(line) == 100 for line in rows))
        self.assertTrue(all(len(line) <= 100 for line in lines))
        return [line.split("|")[1] for line in rows], [line.split("|")[3] for line in rows], lines

    # Each panel uses its own value range while growing history only compresses the X axis.
    def test_separate_scales_fill_vertical_range_and_keep_fixed_width(self):
        for end in (5, 70, 10000):
            with self.subTest(end=end):
                left, right, lines = self.panels([(1, 2.6), (end, 1.)], [(0, 100.), (end, 90.)], end)
                self.assertIn("*", left[0])
                self.assertIn("*", left[-1])
                self.assertIn("*", right[0])
                self.assertIn("*", right[-1])
                self.assertIn(str(end), "\n".join(lines))

    # A fitted narrow range must have distinct axis labels, not thirteen identical rounded values.
    def test_narrow_ranges_retain_axis_detail(self):
        _, _, lines = self.panels([], [(0, 1.28809), (70, 1.28791)], 70)
        labels = [line.split("|")[2].strip() for line in lines if line.count("|") == 4]
        self.assertEqual(len(set(labels)), 13)
        self.assertEqual(float(labels[0]), 1.28809)
        self.assertEqual(float(labels[-1]), 1.28791)

    # Finite extreme magnitudes must neither overflow coordinate math nor expand the fixed width.
    def test_large_and_tiny_values_keep_geometry(self):
        for values in ([1e-300, 2e-300], [-1e308, 1e308]):
            with self.subTest(values=values):
                _, panel, lines = self.panels([], list(zip((0, 70), values)), 70)
                self.assertIn("*", panel[0])
                self.assertIn("*", panel[-1])
                self.assertNotIn("inf", "\n".join(lines))

    # Closely spaced updates stay adjacent even when evaluation cadence is irregular.
    def test_optimizer_spacing_not_observation_index_sets_x_position(self):
        _, right, _ = self.panels([], [(0, 3.), (1, 2.), (100, 1.)], 100)
        middle = right[6]
        self.assertIn("*", middle[:2])
        self.assertEqual(right[-1][-1], "*")

    # Connecting across missing values would invent an observed trajectory.
    def test_invalid_observation_breaks_the_line(self):
        _, right, _ = self.panels([], [(0, 3.), (50, None), (100, 1.)], 100)
        self.assertEqual(sum(row.count("*") for row in right), 2)
        self.assertNotIn(".", "".join(right))

    # Compression cannot erase a spike merely because its neighbors share the same column.
    def test_colliding_columns_preserve_vertical_extrema(self):
        _, right, _ = self.panels([], [(1, 0.), (2, 10.), (10000, 5.)], 10000)
        self.assertEqual(right[0][0], "*")
        self.assertEqual(right[-1][0], "*")

    # Degenerate ranges keep valid geometry without fabricating observations.
    def test_empty_single_and_constant_series_are_readable(self):
        for values, step in (([], 0), ([(0, 2.)], 0), ([(0, 2.), (70, 2.)], 70)):
            with self.subTest(values=values):
                _, right, lines = self.panels([], values, step)
                self.assertTrue(all(len(row) == len(right[0]) for row in right))
                if values:
                    self.assertIn("*", "".join(right))
                else:
                    self.assertNotIn("*", "".join(right))
                self.assertNotIn("nan", "\n".join(lines).lower())


class CallbackDelivery(unittest.TestCase):
    # One structured block prevents other workers from interleaving chart lines.
    def test_structured_report_is_one_event_without_checkpoint_inspection(self):
        events = Mock()
        callback = metrics.callback_class()(pathlib.Path(__file__).parent,
                    assessment_settings=object(), method="sft", events=events)
        state = types.SimpleNamespace(global_step=32, max_steps=40)
        with patch.object(assessment, "run_metrics_report", return_value={}), \
             patch.object(review, "render_metrics_report", return_value="Metrics\nchart\n"), \
             patch("builtins.print") as printed:
            callback._report_metrics(types.SimpleNamespace(), state)
        events.assert_called_once_with({"kind": "metric_report", "message": "Metrics\nchart"})
        printed.assert_not_called()
