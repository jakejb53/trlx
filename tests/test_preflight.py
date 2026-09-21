"""Resume snapshot comparison (SPEC 2.6): the one preflight check that would
fail silently if wrong, by letting a changed config resume a checkpoint.

`compare_snapshot` is a pure function over parsed TOML documents, so no
model, run directory, or GPU is involved.
"""

import io
import pathlib
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import MagicMock, Mock, patch

from trlx import TrlxError, run_dirs, toml_write
from trlx.preflight import Report, _check_example, _check_resume, _check_vllm, compare_snapshot

# Effective inputs after method selection and CLI overrides, and their snapshot:
# run_name resolved, [launch] appended.
CURRENT = {
    "output_dir": "runs/a",
    "learning_rate": 1e-4,
    "resume_from_checkpoint": "runs/a/checkpoint-20",
    "model": {"path": "m", "dtype": "bfloat16"},
    "ranges": {"loss": [0, 5]},
}
SNAPSHOT = {
    "run_name": "a",
    "output_dir": "runs/a",
    "learning_rate": 1e-4,
    "model": {"path": "m", "dtype": "bfloat16"},
    "ranges": {"loss": [0, 5]},
    "launch": {"method": "sft", "strategy": "ddp", "gpus": ["0", "1"]},
}


class CompareSnapshot(unittest.TestCase):
    def test_same_run_resumes(self):
        self.assertEqual(compare_snapshot(CURRENT, SNAPSHOT, "ddp"), [])

    def test_resume_key_and_launch_are_set_aside(self):
        # The snapshot was made without resume_from_checkpoint; the current
        # file must carry it. Neither that nor [launch].gpus is a difference.
        current = dict(CURRENT, resume_from_checkpoint="runs/a/checkpoint-40")
        snapshot = dict(SNAPSHOT, launch={"strategy": "ddp", "gpus": ["1"]})
        self.assertEqual(compare_snapshot(current, snapshot, "ddp"), [])

    # Presentation and device choices can change without changing trained weights.
    def test_run_controls_are_not_training_input_differences(self):
        current = dict(CURRENT, run={"gpus": "1", "strategy": "auto", "tui": True, "verify": False})
        snapshot = dict(SNAPSHOT, run={"gpus": "all", "strategy": "ddp", "tui": False, "verify": True})
        self.assertEqual(compare_snapshot(current, snapshot, "ddp"), [])

    # The checkpoint owns its directory even if the saved path was relative or moved.
    def test_output_directory_is_not_a_training_input_difference(self):
        current = dict(CURRENT, output_dir="runs/relocated")
        self.assertEqual(compare_snapshot(current, SNAPSHOT, "ddp"), [])

    def test_no_strategy_skips_sharding(self):
        # `trlx check` chooses no strategy and passes None.
        self.assertEqual(compare_snapshot(CURRENT, SNAPSHOT, None), [])

    def test_changed_top_level_value(self):
        current = dict(CURRENT, learning_rate=2e-4)
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("learning_rate", diffs[0])

    def test_changed_nested_value(self):
        current = dict(CURRENT, model={"path": "m", "dtype": "float16"})
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("model.dtype", diffs[0])

    def test_added_and_removed_keys(self):
        current = dict(CURRENT, warmup_steps=10)
        del current["learning_rate"]
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 2)
        self.assertTrue(any("warmup_steps" in d for d in diffs))
        self.assertTrue(any("learning_rate" in d for d in diffs))

    def test_sharding_change_is_a_difference(self):
        diffs = compare_snapshot(CURRENT, SNAPSHOT, "fsdp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("strategy", diffs[0])

    def test_single_to_ddp_is_not_a_difference(self):
        # Both unsharded: a one-GPU run resumes on two.
        snapshot = dict(SNAPSHOT, launch={"strategy": "single", "gpus": ["0"]})
        self.assertEqual(compare_snapshot(CURRENT, snapshot, "ddp"), [])

    def test_operator_run_name_is_compared(self):
        # Only a prepended run_name is trlx's; one the operator wrote counts.
        current = dict(CURRENT, run_name="b")
        diffs = compare_snapshot(current, SNAPSHOT, "ddp")
        self.assertEqual(len(diffs), 1)
        self.assertIn("run_name", diffs[0])


class ResumeCheck(unittest.TestCase):
    # Comparison tests isolate metadata validation without creating checkpoint files.
    def setUp(self):
        self.inspect = self.enterContext(patch(
            "trlx.preflight.run_dirs.inspect_checkpoint",
            return_value=run_dirs.Resume(pathlib.Path("runs/a/checkpoint-20"), 20),
        ))

    # Resume needs the already resolved inputs, never a second read of the operator config.
    def config(self, document):
        return SimpleNamespace(
            args=SimpleNamespace(resume_from_checkpoint="runs/a/checkpoint-20", output_dir="runs/a"),
            method=SimpleNamespace(name="sft"), document=document,
        )

    # A temporary CLI override must invalidate resume even when the source file is unchanged.
    def test_resume_compares_effective_inputs(self):
        current = dict(CURRENT, model={"path": "different/model", "dtype": "bfloat16"})
        snapshot = io.BytesIO(toml_write.dumps(SNAPSHOT).encode())
        with patch("trlx.preflight.open", return_value=snapshot) as opening:
            with self.assertRaisesRegex(TrlxError, "resume refused") as caught:
                _check_resume(self.config(current), "operator.toml", "ddp")
        self.assertIn("model.path", str(caught.exception))
        opening.assert_called_once_with(pathlib.Path("runs/a/config.toml"), "rb")

    # Identical argument shapes do not make a checkpoint from another method compatible.
    def test_resume_rejects_different_method(self):
        saved = dict(SNAPSHOT, launch={"method": "dpo", "strategy": "ddp", "gpus": ["0"]})
        with patch("trlx.preflight.open", return_value=io.BytesIO(toml_write.dumps(saved).encode())):
            with self.assertRaisesRegex(TrlxError, "is for dpo, not sft"):
                _check_resume(self.config(CURRENT), "operator.toml", "ddp")

    # Incomplete checkpoints fail before their snapshot can approve a continuation.
    def test_invalid_checkpoint_stops_snapshot_read(self):
        self.inspect.side_effect = TrlxError("trainer_state.json: invalid checkpoint")
        with patch("trlx.preflight.open") as opening:
            with self.assertRaisesRegex(TrlxError, "invalid checkpoint"):
                _check_resume(self.config(CURRENT), "operator.toml", "ddp")
        opening.assert_not_called()


class VllmDiagnostics(unittest.TestCase):
    # The URL and a network-library error can both contain credentials.
    def test_probe_error_omits_url_credentials_and_echoed_reason(self):
        cfg = SimpleNamespace(rewards=["reward"], args=SimpleNamespace(
            vllm_server_base_url="https://user:secret@example.test/v1?token=private",
        ))
        with patch("trlx.preflight.urllib.request.urlopen", side_effect=urllib.error.URLError("secret private")):
            with self.assertRaises(TrlxError) as caught:
                _check_vllm(cfg)
        self.assertIn("example.test/v1/get_world_size", str(caught.exception))
        for secret in ("user", "secret", "private", "token="):
            self.assertNotIn(secret, str(caught.exception))


class ExampleOutput(unittest.TestCase):
    # Exercise the real report formatting while controlling tokenization and masks.
    def example(self, labels, text, trained_text):
        dataset = MagicMock(column_names=["input_ids", "labels"], num_rows=1)
        dataset.__getitem__.return_value = {"input_ids": [1, 2], "labels": labels}
        tokenizer = Mock()
        tokenizer.decode.side_effect = [text, trained_text]
        report = Report()
        _check_example(SimpleNamespace(method=SimpleNamespace(name="sft")),
                       SimpleNamespace(train_dataset=dataset), tokenizer, report)
        output = io.StringIO()
        report.flush(output)
        return report.to_dict(), output.getvalue()

    # Full examples stay in machine-readable facts while the terminal shows counts.
    def test_all_tokens_trained_omits_text_and_preserves_facts(self):
        text = "Café\n\n雪"
        facts, output = self.example([1, 2], text, text)
        self.assertIn("first row: 2 tokens, 2 trained", output)
        self.assertNotIn("Café", output)
        self.assertNotIn("雪", output)
        self.assertNotIn("first row text:", output)
        self.assertIn("all tokens trained", output)
        self.assertNotIn("first row trained text:", output)
        self.assertEqual(facts["example"], {
            "tokens": 2, "trained_tokens": 2, "text": text, "trained_text": text,
        })

    # Equal decoded text does not imply all tokens participate in the loss.
    def test_partial_mask_omits_text_and_preserves_facts(self):
        facts, output = self.example([-100, 2], "é", "é")
        self.assertIn("first row: 2 tokens, 1 trained", output)
        self.assertNotIn("é", output)
        self.assertNotIn("first row text:", output)
        self.assertNotIn("first row trained text:", output)
        self.assertNotIn("all tokens trained", output)
        self.assertEqual(facts["example"], {
            "tokens": 2, "trained_tokens": 1, "text": "é", "trained_text": "é",
        })

    # Omitting example text must not hide the existing empty-label warning.
    def test_no_trained_tokens_keeps_warning(self):
        facts, output = self.example([-100, -100], "prompt", "")
        self.assertIn("first row: 2 tokens, 0 trained", output)
        self.assertNotIn("prompt", output)
        self.assertNotIn("first row trained text:", output)
        self.assertIn("first row has no trained tokens in its label mask", output)
        self.assertNotIn("all tokens trained", output)
        self.assertEqual(facts["example"]["trained_text"], "")


if __name__ == "__main__":
    unittest.main()
