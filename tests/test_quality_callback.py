"""Advisory callback scheduling and durable evidence without running a trainer."""

import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tests.test_quality import Model, Rows, Tokenizer, settings
from trlx import TrlxError, assessment, metrics, quality, quality_scorers


# Share only in-memory fixtures; the sole disk outputs are the production JSONL evidence files.
class CallbackFixture(unittest.TestCase):
    # Real writers exercise callback ordering and flush behavior inside repository-local scratch.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test-quality-callback-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.settings = settings()
        self.args = SimpleNamespace(load_best_model_at_end=False, eval_strategy="steps",
                                    get_warmup_steps=Mock(return_value=0))
        self.state = SimpleNamespace(global_step=0, max_steps=10, epoch=0.0, num_train_epochs=1.0,
                                     best_model_checkpoint=None)
        self.control = SimpleNamespace(should_training_stop=False, should_epoch_stop=False,
                                       should_log=True, should_evaluate=False, should_save=False)
        self.writer = metrics.callback_class()(self.folder, self.settings, "sft", {})
        self.addCleanup(self.writer.on_train_end, self.args, self.state, self.control)
        self.model = Model()
        self.model.train()
        self.model.dropout.eval()
        self.trainer = SimpleNamespace(processing_class=Tokenizer(),
                                       accelerator=SimpleNamespace(unwrap_model=Mock(return_value=self.model)),
                                       log=Mock(side_effect=AssertionError("quality must not call Trainer.log")))
        self.rows = Rows([{"prompt": "Question", "answer": "Answer"}])
        scored = quality_scorers.score_generation("qa", self.rows[0], "Answer")
        self.results = [{"row": 1, "input": self.rows[0], "output": "Answer", **scored}]
        self.load = self.enterContext(patch.object(quality, "load_data", return_value=self.rows))
        self.series = self.enterContext(patch.object(quality, "series_id", return_value="series-a"))
        self.evaluate = self.enterContext(patch.object(quality, "evaluate", return_value=self.results))
        self.advice = self.enterContext(patch.object(assessment, "run_assessment", wraps=assessment.run_assessment))
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))
        self.callback = quality.callback_class()(self.settings, self.folder, self.writer, 0)
        self.callback.bind(self.trainer)

    # Read the actual atomic publication artifact instead of asserting only mocked calls.
    def raw_rounds(self):
        return [json.loads(line) for line in (self.folder / quality.FILENAME).read_text().splitlines()]

    # Stored baseline fixtures use the same authoritative metric writer as live observations.
    def retain_baseline(self, *, series="series-a", status="complete"):
        self.writer.quality(self.args, self.state, {"quality/score": 0.75},
                            {"phase": "baseline", "preset": "qa", "series": series, "status": status, "step": 0})


class SchedulingTest(CallbackFixture):
    # A scheduled final-step check never substitutes for the separately attributed completion check.
    def test_baseline_scheduled_and_completion(self):
        control = copy.deepcopy(vars(self.control))
        self.assertIsNone(self.callback.on_train_begin(self.args, self.state, self.control, model=self.model))
        self.state.global_step = 10
        self.state.epoch = 1.0
        self.assertIsNone(self.callback.on_evaluate(self.args, self.state, self.control, model=self.model))
        self.assertIsNone(self.callback.on_train_end(self.args, self.state, self.control, model=self.model))
        records = metrics.read(self.folder / metrics.FILENAME)
        self.assertEqual([item["quality"]["phase"] for item in records], ["baseline", "scheduled", "completion"])
        self.assertEqual([item["step"] for item in records], [0, 10, 10])
        self.assertTrue(all(item["quality"]["status"] == "complete" for item in records))
        self.assertEqual([item["quality"]["phase"] for item in self.raw_rounds()], ["baseline", "scheduled", "completion"])
        self.assertEqual(self.evaluate.call_count, 3)
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(vars(self.control), control)
        self.trainer.log.assert_not_called()

    # Disabling ordinary evaluation still leaves baseline and completion quality observations.
    def test_completion_when_eval_disabled(self):
        self.args.eval_strategy = "no"
        self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        self.state.global_step = 10
        self.callback.on_train_end(self.args, self.state, self.control, model=self.model)
        self.assertEqual([item["quality"]["phase"] for item in self.raw_rounds()], ["baseline", "completion"])
        self.assertEqual(self.evaluate.call_count, 2)

    # A retained complete baseline avoids substituting resumed weights for the original comparison.
    def test_matching_complete_baseline_is_retained(self):
        self.retain_baseline()
        self.state.global_step = 4
        self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        self.evaluate.assert_not_called()
        self.assertFalse((self.folder / quality.FILENAME).exists())
        self.assertIn("retained matching baseline", self.output.getvalue())
        self.state.global_step = 10
        self.callback.on_train_end(self.args, self.state, self.control, model=self.model)
        self.evaluate.assert_called_once()
        self.assertEqual(self.raw_rounds()[0]["quality"]["phase"], "completion")

    # Changed inputs start a new baseline at the actual resumed step, never a fictitious step zero.
    def test_changed_series_gets_new_baseline(self):
        self.retain_baseline(series="previous-series")
        self.state.global_step = 4
        self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        self.evaluate.assert_called_once()
        context = self.raw_rounds()[0]["quality"]
        self.assertEqual(context["phase"], "baseline")
        self.assertEqual(context["series"], "series-a")
        self.assertEqual(context["step"], 4)

    # Failed or incomplete evidence cannot suppress collection of a valid baseline.
    def test_failed_baseline_does_not_suppress_new_attempt(self):
        self.retain_baseline(status="failed")
        self.assertFalse(self.writer.has_baseline("series-a"))
        self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        self.evaluate.assert_called_once()
        self.assertTrue(self.writer.has_baseline("series-a"))

    # Final observations identify trainer-selected best weights without selecting any checkpoint themselves.
    def test_completion_attributes_best_checkpoint(self):
        self.args.load_best_model_at_end = True
        self.state.global_step = 10
        self.state.best_model_checkpoint = "runs/example/checkpoint-6"
        self.callback.on_evaluate(self.args, self.state, self.control, model=self.model)
        self.callback.on_train_end(self.args, self.state, self.control, model=self.model)
        scheduled, completion = self.raw_rounds()
        self.assertNotIn("model_checkpoint", scheduled["quality"])
        self.assertEqual(completion["quality"]["model_checkpoint"], "runs/example/checkpoint-6")
        self.assertEqual(self.state.best_model_checkpoint, "runs/example/checkpoint-6")
        self.assertEqual(self.state.global_step, 10)


class FailureTest(CallbackFixture):
    # Partially observed samples survive a scorer failure, but cannot become aggregate quality metrics.
    def test_scoring_failure_is_advisory_and_preserves_state(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        modes = [module.training for module in self.model.modules()]
        generation = self.model.generation_config
        control = copy.deepcopy(vars(self.control))

        # Consume each RNG inside scoring to prove that the outer callback also isolates failure paths.
        def fail(*args, **kwargs):
            random.random()
            np.random.random()
            torch.rand(4)
            self.model.generation_config.temperature = 20
            raise quality.EvaluationError("judge timed out on row 2", copy.deepcopy(self.results))

        self.evaluate.side_effect = fail
        self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        result = self.raw_rounds()[0]
        self.assertEqual(result["quality"]["status"], "failed")
        self.assertIn("row 2", result["quality"]["error"])
        self.assertEqual(result["results"], self.results)
        self.assertFalse((self.folder / metrics.FILENAME).exists())
        self.assertEqual(vars(self.control), control)
        self.assertEqual(random.getstate(), python_state)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
        self.assertEqual(current_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual([module.training for module in self.model.modules()], modes)
        self.assertIs(self.model.generation_config, generation)
        self.assertEqual(generation.temperature, 1.0)
        self.trainer.log.assert_not_called()

    # A failed setup is not cached as ready; completion retries setup and can produce valid evidence.
    def test_completion_retries_after_setup_failure(self):
        self.load.side_effect = [TrlxError("dataset temporarily unavailable"), self.rows]
        self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        self.assertIsNone(self.callback.dataset)
        self.evaluate.assert_not_called()
        self.state.global_step = 10
        self.callback.on_train_end(self.args, self.state, self.control, model=self.model)
        self.assertEqual(self.load.call_count, 2)
        self.evaluate.assert_called_once()
        baseline, completion = self.raw_rounds()
        self.assertEqual(baseline["quality"]["status"], "failed")
        self.assertEqual(baseline["results"], [])
        self.assertEqual(completion["quality"]["phase"], "completion")
        self.assertEqual(completion["quality"]["status"], "complete")
        self.assertIs(self.callback.dataset, self.rows)

    # Publication failure is reported as unavailable evidence and cannot request a training stop.
    def test_raw_publication_failure_does_not_change_control(self):
        control = copy.deepcopy(vars(self.control))
        with patch.object(quality, "publish", side_effect=TrlxError("disk unavailable")):
            self.callback.on_train_begin(self.args, self.state, self.control, model=self.model)
        self.assertEqual(vars(self.control), control)
        self.assertIn("evidence unavailable", self.output.getvalue())
        self.assertFalse((self.folder / metrics.FILENAME).exists())


class MetricsWriterTest(CallbackFixture):
    # Aggregate writes are immediately readable, retain copied context, and bypass Trainer.log.
    def test_quality_metrics_are_durable_and_copy_context(self):
        context = {"phase": "baseline", "preset": "qa", "series": "series-a", "status": "complete", "step": 0}
        values = {"quality/score": 0.5, "quality/rows": 2}
        self.writer.quality(self.args, self.state, values, context)
        context["phase"] = "changed-after-write"
        context["series"] = "changed-series"
        stored = metrics.read(self.folder / metrics.FILENAME)
        self.assertEqual(stored[0]["log"], values)
        self.assertTrue(stored[0]["eval"])
        self.assertEqual(stored[0]["quality"]["phase"], "baseline")
        self.assertEqual(self.writer.records[0]["quality"]["series"], "series-a")
        self.assertIsNot(self.writer.records[0]["quality"], context)
        self.trainer.log.assert_not_called()

    # Resume reads authoritative persisted records and accepts only matching complete baselines.
    def test_retained_baseline_status_and_resume_reload(self):
        for series, status in (("failed", "failed"), ("missing", None), ("complete", "complete")):
            self.retain_baseline(series=series, status=status)
        self.writer.on_train_end(self.args, self.state, self.control)
        reloaded = metrics.callback_class()(self.folder, self.settings, "sft", {})
        self.addCleanup(reloaded.on_train_end, self.args, self.state, self.control)
        self.assertFalse(reloaded.has_baseline("failed"))
        self.assertFalse(reloaded.has_baseline("missing"))
        self.assertFalse(reloaded.has_baseline("unknown"))
        self.assertTrue(reloaded.has_baseline("complete"))

    # Advisory failures do not corrupt durable metrics or seize the trainer's control flags.
    def test_runtime_advisory_failure_isolated(self):
        control = copy.deepcopy(vars(self.control))
        self.advice.side_effect = ValueError("broken diagnostic")
        self.writer.on_log(self.args, self.state, self.control, logs={"loss": 2.0})
        self.state.global_step = 1
        self.writer.on_log(self.args, self.state, self.control, logs={"loss": 1.0})
        self.writer.quality(self.args, self.state, {"quality/score": 0.5},
                            {"phase": "completion", "preset": "qa", "series": "series-a", "status": "complete"})
        self.writer.on_evaluate(self.args, self.state, self.control)
        self.writer.on_evaluate(self.args, self.state, self.control)
        self.assertEqual(len(metrics.read(self.folder / metrics.FILENAME)), 3)
        self.assertEqual(self.advice.call_count, 2)
        self.assertEqual(vars(self.control), control)
        self.assertEqual(self.output.getvalue().count("assessment unavailable"), 2)


if __name__ == "__main__":
    unittest.main()
