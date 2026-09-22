"""Prompt-file ownership, literal substitution, and run snapshot persistence."""

import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

from dataset import prompts
from dataset.io import DatasetError, write_many_text
from trlx import TrlxError, config, synthetic_eval, train


class PromptFiles(unittest.TestCase):
    # All generated files stay inside the repository and are removed after each test.
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)))

    # Inserted data cannot create placeholders or optional blocks in a second pass.
    def test_substitution_is_literal(self):
        self.assertEqual(prompts.fill("{text}[[ keys: {keys}]]", text="{keys}[[x]]", keys=None), "{keys}[[x]]")
        self.assertEqual(prompts.fill("[[ keys: {keys}]]", keys="{text}"), " keys: {text}")

    # Invalid templates fail before they can be used for model requests.
    def test_template_validation(self):
        path = self.root / "test.prompt"
        for text in ("", "[[ {text}", "[[ [[{text}]] ]]", "{unknown}", "no source"):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(DatasetError):
                    prompts.load(path, required=("text",))
        path.write_text("{text}")
        self.assertEqual(prompts.load(path, required=("text",)), "{text}")

    # Plain rubrics are not templates, so authored brace expressions remain untouched.
    def test_plain_prompt_and_missing_file(self):
        path = self.root / "judge.prompt"
        with self.assertRaisesRegex(DatasetError, "trlx init"):
            prompts.load(path)
        path.write_text("Judge {arbitrary} [[literal]]")
        self.assertEqual(prompts.load(path, allowed=None), path.read_text())

    # Config-relative paths work regardless of the process working directory.
    def test_active_config_paths(self):
        path = self.root / "summary.prompt"
        path.write_text("Summarize {text}")
        dataset = types.SimpleNamespace(synthetic_dataset_eval=True)
        texts = config._prompts(self.root / "run.toml", {"synthetic_eval_summary": "summary.prompt"}, dataset, None)
        self.assertEqual(texts, {"synthetic_eval_summary": "Summarize {text}"})
        path.unlink()
        with self.assertRaisesRegex(TrlxError, "summary.prompt"):
            config._prompts(self.root / "run.toml", {"synthetic_eval_summary": "summary.prompt"}, dataset, None)
        dataset.synthetic_dataset_eval = False
        self.assertEqual(config._prompts(self.root / "run.toml", {}, dataset, None), {})

    # Snapshot publication uses reviewed text and workers resolve only run-owned paths.
    def test_snapshot_and_resume_use_saved_text(self):
        source = self.root / "original.prompt"
        source.write_text("original {text}")
        directory = self.root / "run"
        directory.mkdir()
        cfg = types.SimpleNamespace(
            args=types.SimpleNamespace(run_name="test", resume_from_checkpoint=None),
            method=types.SimpleNamespace(name="sft"),
            document={"prompts": {"synthetic_eval_summary": str(source)},
                      "rewards": {"funcs": [{"name": "llm_judge", "args": {"rubric_file": "source.prompt"}}]}},
            prompts={"synthetic_eval_summary": source.read_text(), "reward_0": "Rate the task"},
        )
        source.write_text("edited after review {text}")
        snapshot = train._write_snapshot(cfg, directory, [0], "single")
        document = config._snapshot(snapshot, "sft")
        self.assertEqual(config._read_prompt(snapshot, document["prompts"]["synthetic_eval_summary"], ("text",)), "original {text}")
        self.assertEqual(config._read_prompt(snapshot, document["rewards"]["funcs"][0]["args"]["rubric_file"]), "Rate the task")
        self.assertEqual(cfg.document["prompts"]["synthetic_eval_summary"], str(source))
        cfg.document = document
        cfg.args.resume_from_checkpoint = str(directory / "checkpoint-1")
        with patch.object(train, "write_many_text", side_effect=AssertionError("resume must not republish prompts")):
            train._write_snapshot(cfg, directory, [0], "single")

    # Synthetic summaries use the selected template and never interpret source placeholders.
    def test_synthetic_prompt_content(self):
        model = types.SimpleNamespace(config=types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace()))
        with patch.object(synthetic_eval.quality, "_encode", return_value=[1]) as encode:
            synthetic_eval._prompt(object(), "source {text}", model, 10, 1, "Custom {text}")
        self.assertEqual(encode.call_args.args[1], "Custom source {text}")

    # Real config preparation loads templates before snapshot handoff and fails on missing copies.
    def test_config_worker_roundtrip(self):
        import tomllib
        from tests.test_config import BASE
        from trlx import init_cmd

        path = self.root / "qa.prompt"
        path.write_text("Only answer the task.")
        document = tomllib.loads(BASE)
        document.update(use_cpu=True, bf16=False, fp16=False)
        document["assessment"] = dict(init_cmd.ASSESSMENT_DEFAULTS,
                                      quality_checks=True, quality_preset="qa", quality_dataset="quality.jsonl")
        document["prompts"] = {"quality_qa": "qa.prompt"}
        cfg = config.from_document(document, "sft", path=self.root / "run.toml")
        directory = self.root / "roundtrip"
        directory.mkdir()
        snapshot = train._write_snapshot(cfg, directory, [0], "single")
        path.unlink()
        worker = config.load(snapshot, "sft", resolved=True)
        self.assertEqual(worker.assessment.prompts, {"quality_qa": "Only answer the task."})
        (directory / "prompts" / "quality-qa.prompt").unlink()
        with self.assertRaisesRegex(TrlxError, "quality-qa.prompt"):
            config.load(snapshot, "sft", resolved=True)

    # Every serialization completes before any existing destination is replaced.
    def test_text_staging_failure_preserves_originals(self):
        first, second = self.root / "one.prompt", self.root / "two.prompt"
        first.write_text("old one")
        second.write_text("old two")
        with self.assertRaises(DatasetError):
            write_many_text([(first, "new"), (second, "\ud800")], force=True)
        self.assertEqual(first.read_text(), "old one")
        self.assertEqual(second.read_text(), "old two")
