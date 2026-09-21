"""Assessment evidence survives rendering and follows checkpoint-authorized rewind."""

import copy
import json
import pathlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from trlx import TrlxError, metrics, render_lines, review, run_dirs, show


# Complete metric records exercise the reader rather than bypassing its contract.
def record(step, *, evaluation=False, quality=None):
    result = {"step": step, "max_steps": 40, "epoch": step / 40, "num_train_epochs": 1,
              "eval": evaluation, "time": float(step), "log": {"eval_loss" if evaluation else "loss": 1.5}}
    if quality is not None:
        result.update(eval=True, quality=quality,
                      log={"quality/accuracy": .75, "quality/metric_rows/accuracy": 4})
    return result


# Quality evidence has its own per-row artifact and a shared aggregate metric record.
def quality_context(step, phase="scheduled", **changes):
    return {"step": step, "phase": phase, "preset": "classification", "status": "complete", "series": "same-series", **changes}


class AssessmentLifecycleTests(unittest.TestCase):
    # Every write is disposable repository-local JSON/JSONL; no config is created.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)
        self.addCleanup(scratch.cleanup)
        self.directory = pathlib.Path(scratch.name).resolve()

    # JSONL round-trips preserve individual evidence in addition to aggregate metrics.
    def write_records(self, name, records):
        path = self.directory / name
        path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return path

    # The selected metadata is sufficient here; no model loading is part of rewind.
    def checkpoint(self, step):
        path = self.directory / f"checkpoint-{step}"
        path.mkdir()
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        return path

    # Metrics, independent rounds and checkpoints share one selected boundary;
    # logs and selected checkpoint state must survive for auditability.
    def test_rewind_trims_quality_and_metrics_together(self):
        selected, later = self.checkpoint(20), self.checkpoint(40)
        earlier = self.checkpoint(10)
        baseline = quality_context(0, "baseline")
        scheduled = quality_context(20)
        completion = quality_context(40, "completion")
        metric_rows = [record(0, quality=baseline), record(10), record(20, evaluation=True),
                       record(20, quality=scheduled), record(30), record(40, quality=completion)]
        metric_path = self.write_records(metrics.FILENAME, metric_rows)
        rounds = [{"quality": context, "rows": [{"row": 0, "scores": {"accuracy": 1}}]}
                  for context in (baseline, scheduled, completion)]
        quality_path = self.write_records(show.QUALITY_FILENAME, rounds)
        reports = [self.directory / name for name in (show.ASSESSMENT_FILENAME, show.PREFLIGHT_FILENAME, show.VERIFY_FILENAME)]
        for path in reports:
            path.write_text("{}", encoding="utf-8")
        log = self.directory / show.LOG_FILENAME
        log.write_text("original diagnostic log\n", encoding="utf-8")
        selected_bytes = (selected / "trainer_state.json").read_bytes()
        resume = SimpleNamespace(directory=self.directory, checkpoint=selected, step=20)
        kept, message = run_dirs.rewind(resume)
        self.assertEqual(kept, 4)
        self.assertEqual(metrics.read(metric_path), metric_rows[:4])
        self.assertEqual([json.loads(line) for line in quality_path.read_text().splitlines()], rounds[:2])
        self.assertTrue(earlier.exists())
        self.assertFalse(later.exists())
        self.assertEqual((selected / "trainer_state.json").read_bytes(), selected_bytes)
        self.assertEqual(log.read_text(), "original diagnostic log\n")
        self.assertTrue(all(not path.exists() for path in reports))
        self.assertIn("rewound quality evidence", message)

    # Even a malformed abandoned quality round must be validated before deleting
    # checkpoints or reports; a failed rewind may not destroy the repair evidence.
    def test_corrupt_quality_is_rejected_before_mutation(self):
        selected, later = self.checkpoint(20), self.checkpoint(40)
        metric_path = self.write_records(metrics.FILENAME, [record(20), record(40)])
        reports = [self.directory / name for name in (show.ASSESSMENT_FILENAME, show.PREFLIGHT_FILENAME, show.VERIFY_FILENAME)]
        for path in reports:
            path.write_text("{}", encoding="utf-8")
        quality_path = self.directory / show.QUALITY_FILENAME
        invalid = ["{invalid\n", json.dumps({"quality": {"step": True}}) + "\n",
                   json.dumps({"quality": {"step": -1}}) + "\n", json.dumps({"quality": {}}) + "\n"]
        metric_bytes = metric_path.read_bytes()
        for content in invalid:
            with self.subTest(content=content):
                quality_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(TrlxError, "quality.jsonl.*line 1.*invalid quality evidence"):
                    run_dirs.rewind(SimpleNamespace(directory=self.directory, checkpoint=selected, step=20))
                self.assertEqual(quality_path.read_text(), content)
                self.assertEqual(metric_path.read_bytes(), metric_bytes)
                self.assertTrue(selected.exists() and later.exists())
                self.assertTrue(all(path.exists() for path in reports))

    # Independent quality summaries do not require a normal range-evaluated row,
    # and each score keeps its denominator instead of printing bookkeeping alone.
    def test_line_display_quality_summary_and_denominators(self):
        emitted = []
        stream = render_lines.Stream({"loss": (0, 10)}, emitted.append)
        stream.record(record(20, quality=quality_context(20)), None)
        self.assertIn("Independent quality: scheduled, step 20, classification", emitted)
        self.assertIn("  accuracy: 0.75 (4 rows)", emitted)
        self.assertFalse(any("metric_rows" in line or "Training" in line for line in emitted))

    # Later quality output at the same step must not erase ordinary eval loss or
    # change which checkpoint has the best known held-out loss.
    def test_quality_records_preserve_checkpoint_eval_loss(self):
        self.checkpoint(10)
        self.checkpoint(20)
        first, second = record(10, evaluation=True), record(20, evaluation=True)
        first["log"]["eval_loss"], second["log"]["eval_loss"] = 2.0, 1.0
        records = [first, record(10, quality=quality_context(10)), second, record(20, quality=quality_context(20))]
        checkpoints = show._checkpoints(self.directory, records)
        self.assertEqual([(item.step, item.eval_loss, item.best) for item in checkpoints], [(10, 2.0, False), (20, 1.0, True)])

    # Terminal previews may be concise, but neither nested evidence nor the
    # persisted report is truncated as a side effect of rendering.
    def test_assessment_preview_preserves_full_evidence_and_check_mode(self):
        report = {"profile": {"train": {"rows": 1000}, "eval": {"rows": 100}},
                  "findings": [{"code": "data.train.truncated_rows", "basis": "projected", "severity": "warning",
                                "summary": "Sequence settings discard tokens", "evidence": {"rows": list(range(1000))},
                                "recommendation": "Review the affected rows."}],
                  "quality": {"preset": "qa", "rows": 100}}
        original = copy.deepcopy(report)
        training_text = review.render_assessment(report)
        check_text = review.render_assessment(report, will_publish=False)
        self.assertEqual(report, original)
        self.assertIn('"count": 1000', training_text)
        self.assertIn('"first": [0, 1, 2, 3, 4]', training_text)
        self.assertNotIn("998", training_text)
        self.assertIn("assessment.json after confirmation", training_text)
        self.assertNotIn("assessment.json", check_text)
        self.assertIn("Advisory only", check_text)
        self.assertIn("including when evaluation is disabled", training_text)

    # A complete malformed quality record must fail with source/line context,
    # rather than crashing later inside a live renderer or a resume comparison.
    def test_metrics_rejects_invalid_quality_context(self):
        path = self.directory / metrics.FILENAME
        bad = [None, [], "quality", {}, {"phase": "unknown", "preset": "qa"},
               {"phase": "baseline", "preset": None}, {"phase": "completion", "preset": 7}]
        for context in bad:
            with self.subTest(context=context):
                item = record(1)
                item["quality"] = context
                path.write_text(json.dumps(record(0)) + "\n" + json.dumps(item) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(TrlxError, "metrics.jsonl.*line 2.*quality"):
                    metrics.read(path)

    # Legacy/partial completion metadata is readable, but only explicit complete
    # status can suppress a fresh baseline for the same evaluation series.
    def test_unknown_status_is_readable_but_not_a_completed_baseline(self):
        for status in ("missing", None, "complete"):
            with self.subTest(status=status):
                context = quality_context(0, "baseline", status=status)
                if status == "missing":
                    del context["status"]
                item = record(0, quality=context)
                path = self.write_records(metrics.FILENAME, [item])
                self.assertEqual(metrics.read(path), [item])
                callback = metrics.callback_class()(self.directory, assessment_settings=SimpleNamespace())
                self.assertEqual(callback.has_baseline("same-series"), status == "complete")
                self.assertFalse(callback.has_baseline("different-series"))

    # Every completed round refreshes its measured comparison, even at the same step;
    # intervening training logs still coalesce that unchanged comparison.
    def test_completed_quality_round_refreshes_baseline_evidence(self):
        settings = SimpleNamespace(runtime_window=6, runtime_min_evaluations=3, runtime_relative_change=.1)
        callback = metrics.callback_class()(self.directory, assessment_settings=settings, method="sft", ranges={})
        state = SimpleNamespace(global_step=0, max_steps=40, epoch=0.0, num_train_epochs=1.0)
        args = SimpleNamespace(get_warmup_steps=Mock(return_value=0))
        control = SimpleNamespace(should_training_stop=False)
        self.addCleanup(callback.on_train_end, args, state, control)
        with patch("builtins.print") as emitted:
            callback.quality(args, state, {"quality/accuracy": .5}, quality_context(0, "baseline"))
            emitted.assert_not_called()
            state.global_step = 20
            callback.quality(args, state, {"quality/accuracy": .75}, quality_context(20, "scheduled"))
            self.assertEqual(emitted.call_count, 1)
            callback.on_log(args, state, control, logs={"loss": 1.0})
            self.assertEqual(emitted.call_count, 1)
            callback.quality(args, state, {"quality/accuracy": 1.0}, quality_context(20, "completion"))
            self.assertEqual(emitted.call_count, 2)
            callback.on_log(args, state, control, logs={"loss": .9})
            self.assertEqual(emitted.call_count, 2)
        first, second = [call.args[0] for call in emitted.call_args_list]
        self.assertIn('"current": 0.75', first)
        self.assertIn('"phase": "scheduled"', first)
        self.assertIn('"current": 1.0', second)
        self.assertIn('"phase": "completion"', second)
        self.assertIn('"delta": 0.5', second)
        self.assertFalse(control.should_training_stop)

    # Advice retains exactly the serialized evidence when later trainer callbacks mutate logs.
    def test_callback_records_snapshot_mutable_input_logs(self):
        settings = SimpleNamespace(runtime_window=6, runtime_min_evaluations=3, runtime_relative_change=.1)
        callback = metrics.callback_class()(self.directory, assessment_settings=settings, method="sft", ranges={})
        state = SimpleNamespace(global_step=1, max_steps=40, epoch=0.0, num_train_epochs=1.0)
        args = SimpleNamespace(get_warmup_steps=Mock(return_value=0))
        control = SimpleNamespace(should_training_stop=False)
        self.addCleanup(callback.on_train_end, args, state, control)
        logs = {"loss": 1.0, "details": {"samples": [1, 2]}}
        original = copy.deepcopy(logs)
        callback.on_log(args, state, control, logs=logs)
        logs["loss"] = 99.0
        logs["details"]["samples"].append(3)
        logs["new"] = "later callback mutation"
        self.assertEqual(callback.records[0]["log"], original)
        self.assertEqual(callback.records, metrics.read(self.directory / metrics.FILENAME))
