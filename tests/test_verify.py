"""Verification report replacement without loading a GPU model."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

from trlx import TrlxError, verify


class VerifyOutput(unittest.TestCase):
    # Only report I/O is real; mocked model loads avoid GPU requirements.
    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)))
        self.checkpoint = self.directory / "checkpoint"
        self.checkpoint.mkdir()
        self.report = self.checkpoint / "verify.json"
        self.enterContext(patch("trlx.verify.torch.cuda.is_available", return_value=True))
        self.enterContext(patch("trlx.verify._checkpoint_kind", return_value="causal"))
        self.load = self.enterContext(patch("trlx.verify.model_mod.load_model", return_value=Mock()))
        self.processor = self.enterContext(patch("trlx.verify.model_mod.load_processor", return_value=Mock()))
        self.compare_template = verify._chat_template_equal
        self.enterContext(patch("trlx.verify._chat_template_equal", return_value=True))

    # Verification loads both models but must never generate or score responses.
    def test_full_checkpoint_only_performs_structural_checks(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = verify.run(self.checkpoint, "base")
        self.assertTrue(result.ok)
        self.assertEqual([call.args[0].path for call in self.load.call_args_list],
                         ["base", str(self.checkpoint)])
        self.assertEqual(self.load.return_value.mock_calls, [])
        self.assertNotIn("behaviour", output.getvalue())
        self.assertNotIn("sample", output.getvalue())
        self.assertEqual(json.loads(self.report.read_text()), {
            "checkpoint": str(self.checkpoint), "base": "base", "adapter": None,
            "chat_template_equal": True, "failures": [], "ok": True,
        })

    # Template mismatch remains a verification failure after prompt comparison removal.
    def test_chat_template_mismatch_is_reported(self):
        base_processor, checkpoint_processor = Mock(), Mock()
        base_processor.tokenizer.chat_template = "base template"
        checkpoint_processor.tokenizer.chat_template = "different template"
        self.processor.side_effect = [base_processor, checkpoint_processor]
        with patch("trlx.verify._chat_template_equal", side_effect=self.compare_template):
            result = verify.run(self.checkpoint, "base")
        self.assertFalse(result.ok)
        self.assertFalse(result.chat_template_equal)
        self.assertEqual(result.failures, ["chat template differs from the base's"])
        self.assertFalse(json.loads(self.report.read_text())["ok"])

    # Collision refusal preserves the report and avoids unnecessary model loads.
    def test_existing_report_refuses_before_loading(self):
        self.report.write_text("old")
        with self.assertRaisesRegex(TrlxError, "--force"):
            verify.run(self.checkpoint, "base")
        self.load.assert_not_called()
        self.assertEqual(self.report.read_text(), "old")

    # Forced verification replaces the link itself, preserving its former target.
    def test_force_replaces_report_symlink(self):
        target = self.directory / "previous"
        target.write_text("old")
        self.report.symlink_to(target)
        result = verify.run(self.checkpoint, "base", force=True)
        self.assertTrue(result.ok)
        self.assertFalse(self.report.is_symlink())
        self.assertEqual(target.read_text(), "old")
        self.assertTrue(json.loads(self.report.read_text())["ok"])

    # Staging choice cannot itself authorize destruction of an existing report.
    def test_direct_mode_still_requires_force(self):
        self.report.write_text("old")
        with self.assertRaisesRegex(TrlxError, "--force"):
            verify.run(self.checkpoint, "base", no_staging=True)
        verify.run(self.checkpoint, "base", force=True, no_staging=True)
        self.assertTrue(json.loads(self.report.read_text())["ok"])

    # A failed model load must not overwrite the previous complete report.
    def test_loading_failure_preserves_previous_report(self):
        self.report.write_text("old")
        self.load.side_effect = [Mock(), TrlxError("checkpoint load failed")]
        with self.assertRaisesRegex(TrlxError, "checkpoint load failed"):
            verify.run(self.checkpoint, "base", force=True)
        self.assertEqual(self.report.read_text(), "old")

    # Adapter integrity still determines the verdict without requiring changed output.
    def test_adapter_integrity_remains_required_without_generation(self):
        (self.checkpoint / verify.adapter_check.ADAPTER_FILE).touch()
        for model_count in (1, 0):
            check = verify.adapter_check.AdapterCheck(1, 0.5, model_count, 0.5)
            with self.subTest(model_count=model_count), \
                 patch("trlx.verify.adapter_check.task_type", return_value="CAUSAL_LM"), \
                 patch("trlx.verify.adapter_check.check", return_value=check) as inspect, \
                 patch("trlx.verify.PeftModel.from_pretrained", return_value=Mock()) as load_adapter:
                result = verify.run(self.checkpoint, "base", force=True)
            inspect.assert_called_once_with(self.checkpoint, load_adapter.return_value)
            self.assertEqual(load_adapter.return_value.mock_calls, [])
            self.assertEqual(result.ok, model_count == 1)
            self.assertEqual(result.adapter["ok"], model_count == 1)
            self.assertNotIn("behaviour", result.to_dict())
