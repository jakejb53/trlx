"""Verification report replacement without loading a GPU model."""

import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

from dataset.progress import Progress
from trlx import TrlxError, verify


class VerifyOutput(unittest.TestCase):
    # Only report I/O is real; deterministic model outputs avoid GPU requirements.
    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)))
        self.checkpoint = self.directory / "checkpoint"
        self.checkpoint.mkdir()
        self.report = self.checkpoint / "verify.json"
        self.enterContext(patch("trlx.verify.torch.cuda.is_available", return_value=True))
        self.enterContext(patch("trlx.verify._checkpoint_kind", return_value="causal"))
        self.load = self.enterContext(patch("trlx.verify.model_mod.load_model", return_value=Mock()))
        self.enterContext(patch("trlx.verify.model_mod.load_processor", return_value=Mock()))
        self.enterContext(patch("trlx.verify.generate.prompts_from", return_value=["question"]))
        self.raw_outputs = verify._outputs
        self.outputs = self.enterContext(patch("trlx.verify._outputs", side_effect=[["base"], ["trained"]]))
        self.enterContext(patch("trlx.verify._chat_template_equal", return_value=True))

    # Collision refusal preserves the report and avoids unnecessary model loads.
    def test_existing_report_refuses_before_loading(self):
        self.report.write_text("old")
        with self.assertRaisesRegex(TrlxError, "--force"):
            verify.run(self.checkpoint, "base", None)
        self.load.assert_not_called()
        self.assertEqual(self.report.read_text(), "old")

    # Forced verification replaces the link itself, preserving its former target.
    def test_force_replaces_report_symlink(self):
        target = self.directory / "previous"
        target.write_text("old")
        self.report.symlink_to(target)
        result = verify.run(self.checkpoint, "base", None, force=True)
        self.assertTrue(result.ok)
        self.assertFalse(self.report.is_symlink())
        self.assertEqual(target.read_text(), "old")
        self.assertTrue(json.loads(self.report.read_text())["ok"])

    # Staging choice cannot itself authorize destruction of an existing report.
    def test_direct_mode_still_requires_force(self):
        self.report.write_text("old")
        with self.assertRaisesRegex(TrlxError, "--force"):
            verify.run(self.checkpoint, "base", None, no_staging=True)
        verify.run(self.checkpoint, "base", None, force=True, no_staging=True)
        self.assertTrue(json.loads(self.report.read_text())["ok"])

    # The report is published only after the entire comparison succeeds.
    def test_generation_failure_preserves_previous_report(self):
        self.report.write_text("old")
        self.outputs.side_effect = TrlxError("generation failed")
        with self.assertRaisesRegex(TrlxError, "generation failed"):
            verify.run(self.checkpoint, "base", None, force=True)
        self.assertEqual(self.report.read_text(), "old")

    # Reward-model forward OOM is an operator-facing comparison failure.
    def test_reward_score_oom_is_contextual(self):
        model = Mock()
        model.can_generate.return_value = False
        model.parameters.return_value = iter([verify.torch.nn.Parameter(verify.torch.zeros(1))])
        model.side_effect = verify.torch.cuda.OutOfMemoryError("allocation failed")
        processor = Mock()
        processor.tokenizer.return_value.to.return_value = {}
        with patch("trlx.verify.generate._render", return_value=("prompt", False)):
            with self.assertRaisesRegex(TrlxError, "CUDA memory exhausted comparing reward-model scores"):
                self.raw_outputs(model, processor, ["prompt"])

    # Successful scores alone advance the prompt counter, leaving returned scores unchanged.
    def test_reward_scoring_reports_completed_prompts(self):
        lines = []
        model = Mock()
        model.can_generate.return_value = False
        model.parameters.return_value = iter([verify.torch.nn.Parameter(verify.torch.zeros(1))])
        model.return_value.logits = verify.torch.tensor([[0.25]])
        processor = Mock()
        processor.tokenizer.return_value.to.return_value = {}
        with patch("trlx.verify.generate._render", return_value=("prompt", False)):
            with Progress("verify", emit=lines.append) as progress:
                scores = self.raw_outputs(model, processor, ["first", "second"], progress=progress)
        self.assertEqual(scores, ["score 0.25", "score 0.25"])
        self.assertTrue(any("scoring verification prompts; 0/2 prompts" in line for line in lines))
        self.assertTrue(any("scoring verification prompts; 2/2 prompts; finished" in line for line in lines))
