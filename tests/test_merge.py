"""Merge publication with model work mocked and real output directories."""

import pathlib
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from trlx import TrlxError, merge


class MergeOutput(unittest.TestCase):
    # Model work is mocked; publication still uses real repository-local directories.
    def setUp(self):
        self.directory = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)))
        self.base = self.directory / "models" / "base"
        self.adapter = self.directory / "models" / "adapter"
        self.base.mkdir(parents=True)
        self.adapter.mkdir()
        (self.base / "original").write_text("base")
        (self.adapter / "original").write_text("adapter")
        self.enterContext(patch("trlx.merge.torch.cuda.is_available", return_value=True))
        self.enterContext(patch("trlx.merge.adapter_check.task_type", return_value="CAUSAL_LM"))
        self.load = self.enterContext(patch("trlx.merge.model.load_model", return_value=Mock()))
        self.peft = self.enterContext(patch("trlx.merge.PeftModel.from_pretrained", return_value=Mock())).return_value
        self.enterContext(patch("trlx.merge.adapter_check.check",
                                return_value=types.SimpleNamespace(ok=True, message=lambda: "adapter ok")))
        self.processor = Mock()
        self.enterContext(patch("trlx.merge.model.load_processor", side_effect=self.load_processor))
        self.peft.merge_and_unload.return_value.save_pretrained.side_effect = self.save_weights
        self.processor.save_pretrained.side_effect = lambda path: (pathlib.Path(path) / "tokenizer.json").write_text("processor")

    # Even direct mode must load the processor while its base is still available.
    def load_processor(self, spec):
        self.assertTrue((self.base / "original").exists())
        return self.processor

    # A small artifact exercises the same publisher as real model serialization.
    def save_weights(self, path):
        (pathlib.Path(path) / "model.safetensors").write_text("merged")

    # A collision is actionable before any expensive model work begins.
    def test_refuses_before_loading(self):
        with self.assertRaisesRegex(TrlxError, "--force"):
            merge.merge(str(self.base), str(self.adapter), self.base)
        self.load.assert_not_called()

    # Force permits exact inputs and their containing directory using default disk staging.
    def test_force_replaces_input_and_ancestor(self):
        for destination in ("base", "adapter", "parent"):
            with self.subTest(destination=destination):
                out = {"base": self.base, "adapter": self.adapter, "parent": self.base.parent}[destination]
                merge.merge(str(self.base), str(self.adapter), out, force=True)
                self.assertEqual((out / "model.safetensors").read_text(), "merged")
                self.assertTrue((out / "tokenizer.json").exists())
                self.assertFalse((out / "original").exists())
                self.base.mkdir(parents=True, exist_ok=True)
                self.adapter.mkdir(parents=True, exist_ok=True)
                (self.base / "original").write_text("base")
                (self.adapter / "original").write_text("adapter")

    # Both model and processor must save before any old output is removed.
    def test_staged_save_failure_preserves_original(self):
        self.processor.save_pretrained.side_effect = OSError("disk full")
        with self.assertRaisesRegex(TrlxError, "disk full"):
            merge.merge(str(self.base), str(self.adapter), self.base, force=True)
        self.assertEqual((self.base / "original").read_text(), "base")
        self.assertFalse((self.base / "model.safetensors").exists())

    # Known serializer and allocation failures preserve staged inputs and name the output.
    def test_expected_save_errors_are_contextual(self):
        for error in (ValueError("invalid configuration"), merge.torch.cuda.OutOfMemoryError("allocation failed")):
            with self.subTest(error=type(error).__name__):
                self.processor.save_pretrained.side_effect = error
                with self.assertRaises(TrlxError) as caught:
                    merge.merge(str(self.base), str(self.adapter), self.base, force=True)
                self.assertIn(str(self.base), str(caught.exception))
                self.assertEqual((self.base / "original").read_text(), "base")

    # Direct mode deliberately gives up old-output preservation on serialization failure.
    def test_direct_save_failure_does_not_restore_original(self):
        out = self.directory / "merged"
        out.mkdir()
        (out / "original").write_text("old merge")
        self.processor.save_pretrained.side_effect = OSError("disk full")
        with self.assertRaisesRegex(TrlxError, "disk full"):
            merge.merge(str(self.base), str(self.adapter), out, force=True, no_staging=True)
        self.assertFalse((out / "original").exists())
        self.assertTrue((self.base / "original").exists())

    # Replacement applies to the named entry, even when it links to the base.
    def test_symlink_output_leaves_target_untouched(self):
        out = self.directory / "output"
        out.symlink_to(self.base, target_is_directory=True)
        merge.merge(str(self.base), str(self.adapter), out, force=True)
        self.assertFalse(out.is_symlink())
        self.assertEqual((self.base / "original").read_text(), "base")

    # Reject incompatible flags before even probing CUDA, preserving every input and output.
    def test_direct_input_replacement_requires_staging(self):
        for out in (self.base, self.adapter, self.base.parent):
            with self.subTest(out=out), patch("trlx.merge.torch.cuda.is_available") as cuda:
                with self.assertRaisesRegex(TrlxError, "Omit --no-staging.*--force"):
                    merge.merge(str(self.base), str(self.adapter), out, force=True, no_staging=True)
                cuda.assert_not_called()
                self.load.assert_not_called()
                self.assertEqual((self.base / "original").read_text(), "base")
                self.assertEqual((self.adapter / "original").read_text(), "adapter")

    # An input alias must not hide that its physical files are being removed.
    def test_direct_input_alias_requires_staging(self):
        alias = self.directory / "base-alias"
        alias.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(TrlxError, "Omit --no-staging"):
            merge.merge(str(alias), str(self.adapter), self.base, force=True, no_staging=True)
        self.load.assert_not_called()

    # Replacing a link used to access the input also breaks subsequent library reads.
    def test_direct_replacement_of_input_access_link_requires_staging(self):
        alias = self.directory / "models-alias"
        alias.symlink_to(self.base.parent, target_is_directory=True)
        with self.assertRaisesRegex(TrlxError, "Omit --no-staging"):
            merge.merge(str(alias / "base"), str(self.adapter), alias, force=True, no_staging=True)
        self.load.assert_not_called()
        self.assertTrue(alias.is_symlink())

    # Resolving only the final target must not hide an output link in an input alias chain.
    def test_direct_replacement_of_intermediate_input_link_requires_staging(self):
        out = self.directory / "output-link"
        out.symlink_to(self.base, target_is_directory=True)
        alias = self.directory / "base-alias"
        alias.symlink_to(out.name, target_is_directory=True)
        with self.assertRaisesRegex(TrlxError, "Omit --no-staging"):
            merge.merge(str(alias), str(self.adapter), out, force=True, no_staging=True)
        self.load.assert_not_called()
        self.assertTrue(out.is_symlink())
        self.assertEqual((alias / "original").read_text(), "base")

    # A separate old merge can be deleted and rewritten without allocating staging storage.
    def test_separate_output_supports_direct_writing(self):
        out = self.directory / "merged"
        out.mkdir()
        (out / "original").write_text("old merge")
        with patch("dataset.io.tempfile.mkdtemp", side_effect=AssertionError("staging used")):
            merge.merge(str(self.base), str(self.adapter), out, force=True, no_staging=True)
        self.assertEqual((out / "model.safetensors").read_text(), "merged")
        self.assertFalse((out / "original").exists())
        self.assertEqual((self.base / "original").read_text(), "base")

    # An unrelated output link is replaced itself, so its target input remains available.
    def test_direct_output_link_does_not_replace_input_target(self):
        out = self.directory / "output"
        out.symlink_to(self.base, target_is_directory=True)
        merge.merge(str(self.base), str(self.adapter), out, force=True, no_staging=True)
        self.assertFalse(out.is_symlink())
        self.assertEqual((self.base / "original").read_text(), "base")

    # Staging leaves original assets available to serializers that reopen them during saving.
    def test_in_place_save_can_read_original_processor_assets(self):
        self.processor.save_pretrained.side_effect = lambda path: (
            pathlib.Path(path) / "tokenizer.json"
        ).write_text((self.base / "original").read_text())
        merge.merge(str(self.base), str(self.adapter), self.base.parent, force=True)
        self.assertEqual((self.base.parent / "tokenizer.json").read_text(), "base")
