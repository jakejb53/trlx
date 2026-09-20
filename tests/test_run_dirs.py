"""Filesystem contracts for isolated runs and checkpoint-authorized rewind."""

import contextlib
import json
import pathlib
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from unittest.mock import patch

from trlx import TrlxError, metrics, run_dirs, show


# Each test owns repository-local scratch; no real run or configuration is changed.
class RunDirsCase(unittest.TestCase):
    # Cleanup remains registered even when an assertion or fixture operation fails.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)
        self.addCleanup(scratch.cleanup)
        self.directory = pathlib.Path(scratch.name).resolve()

    # Real trainer state isolates inspection from model loading and configuration files.
    def checkpoint(self, name="checkpoint-20", step=20):
        checkpoint = self.directory / name
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        return checkpoint

    # Use the actual metric schema, including training and evaluation at the same step.
    def record(self, step, evaluation=False):
        return {"step": step, "max_steps": 100, "epoch": 0.2,
                "num_train_epochs": 2, "eval": evaluation, "time": 123.0,
                "log": {"eval_loss" if evaluation else "loss": 1.5}}


class Allocation(RunDirsCase):
    # A run number belongs to the date and parent, not a model/dataset combination.
    def test_daily_sequence_spans_models_and_datasets(self):
        dataset = types.SimpleNamespace(source="chunks.jsonl", is_file=True)
        with patch("trlx.run_dirs.datetime.date") as today:
            today.today.return_value = date(2026, 9, 20)
            first = run_dirs.allocate(self.directory, "Qwen/Qwen3.8-27B", dataset)
            dataset.source = "other.jsonl"
            second = run_dirs.allocate(self.directory, "Org/Other", dataset)
        self.assertEqual(first.name, "20260920-1--qwen-qwen3.8-27b--chunks")
        self.assertEqual(second.name, "20260920-2--org-other--other")
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())

    # Advancing the calendar starts a new daily sequence without deleting old runs.
    def test_new_date_restarts_counter(self):
        dataset = types.SimpleNamespace(source="chunks.jsonl", is_file=True)
        with patch("trlx.run_dirs.datetime.date") as today:
            today.today.return_value = date(2026, 9, 20)
            first = run_dirs.allocate(self.directory, "Org/Model", dataset)
            today.today.return_value = date(2026, 9, 21)
            second = run_dirs.allocate(self.directory, "Org/Model", dataset)
        self.assertTrue(first.exists())
        self.assertEqual(second.name, "20260921-1--org-model--chunks")

    # Numbering resumes above the largest allocated number, including gaps.
    def test_existing_numbers_and_other_dates(self):
        (self.directory / "20260920-7--old--data").mkdir()
        (self.directory / "20260919-99--old--data").mkdir()
        (self.directory / "notes").mkdir()
        dataset = types.SimpleNamespace(source="chunks.jsonl", is_file=True)
        with patch("trlx.run_dirs.datetime.date") as today:
            today.today.return_value = date(2026, 9, 20)
            allocated = run_dirs.allocate(self.directory, "Org/Model", dataset)
        self.assertEqual(allocated.name, "20260920-8--org-model--chunks")

    # Local model directories use their basename; hosted datasets retain their identity.
    def test_local_model_and_hosted_dataset_labels(self):
        model = self.directory / "My Model"
        model.mkdir()
        dataset = types.SimpleNamespace(source="Team/Training.Data", is_file=False)
        with patch("trlx.run_dirs.datetime.date") as today:
            today.today.return_value = date(2026, 9, 20)
            allocated = run_dirs.allocate(self.directory / "runs", str(model), dataset)
        self.assertEqual(allocated.name, "20260920-1--my-model--team-training.data")

    # Relative model notation and non-ASCII names still produce usable directory labels.
    def test_current_directory_model_and_unicode_dataset(self):
        model = self.directory / "Modèle"
        model.mkdir()
        dataset = types.SimpleNamespace(source="données.jsonl", is_file=True)
        with patch("trlx.run_dirs.datetime.date") as today, contextlib.chdir(model):
            today.today.return_value = date(2026, 9, 20)
            allocated = run_dirs.allocate(self.directory / "runs", ".", dataset)
        self.assertEqual(allocated.name, "20260920-1--modèle--données")

    # Different suffixes must still contend for the same daily counter under concurrency.
    def test_concurrent_allocators_reserve_distinct_daily_numbers(self):
        datasets = [types.SimpleNamespace(source=f"data-{number}.jsonl", is_file=True)
                    for number in range(16)]
        with patch("trlx.run_dirs.datetime.date") as today:
            today.today.return_value = date(2026, 9, 20)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(run_dirs.allocate, self.directory,
                                           f"Org/Model-{number}", dataset)
                           for number, dataset in enumerate(datasets)]
                directories = [future.result() for future in futures]
        numbers = [int(directory.name.split("--", 1)[0].split("-")[1])
                   for directory in directories]
        self.assertEqual(sorted(numbers), list(range(1, 17)))
        self.assertEqual(len(set(directories)), 16)
        self.assertTrue(all(directory.is_dir() for directory in directories))

    # An active owner's directory lock must reject another owner and release on exit.
    def test_live_run_lock_is_exclusive_and_released(self):
        with run_dirs.locked(self.directory):
            with self.assertRaisesRegex(TrlxError, "another training process"):
                with run_dirs.locked(self.directory):
                    self.fail("second owner acquired the same run")
        with run_dirs.locked(self.directory):
            pass


class CheckpointInspection(RunDirsCase):
    # A valid checkpoint resolves to its owning run and authoritative saved step.
    def test_saved_step_and_run_directory(self):
        checkpoint = self.checkpoint(name="saved-checkpoint", step=12)
        with patch("trlx.run_dirs._check_weights") as weights:
            resume = run_dirs.inspect_checkpoint(checkpoint)
        weights.assert_called_once_with(checkpoint)
        self.assertEqual(resume.step, 12)
        self.assertEqual(resume.directory, self.directory)
        self.assertEqual(resume.checkpoint, checkpoint)

    # A boolean is not a step even though Python considers bool an int subclass.
    def test_rejects_invalid_global_steps(self):
        for number, step in enumerate((True, -1, 2.5, "20", None)):
            with self.subTest(step=step):
                checkpoint = self.checkpoint(name=f"invalid-{number}", step=step)
                with self.assertRaisesRegex(TrlxError, "global_step.*nonnegative integer"):
                    run_dirs.inspect_checkpoint(checkpoint)

    # Named checkpoints must not select one step while cleanup uses another.
    def test_rejects_directory_step_mismatch(self):
        checkpoint = self.checkpoint(name="checkpoint-40", step=20)
        with self.assertRaisesRegex(TrlxError, "40.*disagrees.*20"):
            run_dirs.inspect_checkpoint(checkpoint)

    # Missing and malformed metadata must fail before any weight loading or rewind.
    def test_missing_and_invalid_state(self):
        checkpoint = self.directory / "checkpoint-20"
        with self.assertRaisesRegex(TrlxError, "directory does not exist"):
            run_dirs.inspect_checkpoint(checkpoint)
        checkpoint.mkdir()
        with self.assertRaisesRegex(TrlxError, "trainer_state.json.*cannot read"):
            run_dirs.inspect_checkpoint(checkpoint)
        for state in ("{bad", "[]", "{}"):
            with self.subTest(state=state):
                (checkpoint / "trainer_state.json").write_text(state)
                with self.assertRaises(TrlxError):
                    run_dirs.inspect_checkpoint(checkpoint)

    # Empty tensor files cannot make an incomplete checkpoint eligible for rewind.
    def test_missing_or_empty_weights(self):
        checkpoint = self.checkpoint()
        with self.assertRaisesRegex(TrlxError, "no saved model or adapter weights"):
            run_dirs.inspect_checkpoint(checkpoint)
        (checkpoint / "model.safetensors").touch()
        with self.assertRaisesRegex(TrlxError, "no saved model or adapter weights"):
            run_dirs.inspect_checkpoint(checkpoint)

    # Metadata reads are mocked so tests never create configuration files.
    def test_full_and_adapter_weights_require_corresponding_metadata(self):
        for filename, metadata in (("model.safetensors", "config.json"),
                                   ("adapter_model.safetensors", "adapter_config.json")):
            with self.subTest(filename=filename):
                checkpoint = self.checkpoint(name=filename, step=20)
                (checkpoint / filename).write_bytes(b"weights")
                with patch("trlx.run_dirs._read_json", return_value={}) as read:
                    run_dirs._check_weights(checkpoint)
                read.assert_called_once_with(checkpoint / metadata)
                with patch("trlx.run_dirs._read_json", side_effect=TrlxError("invalid metadata")):
                    with self.assertRaisesRegex(TrlxError, "invalid metadata"):
                        run_dirs._check_weights(checkpoint)

    # An index is usable only when its mapping and every referenced shard are complete.
    def test_indexed_weights_validate_mapping_and_shards(self):
        checkpoint = self.checkpoint()
        (checkpoint / "model.safetensors.index.json").touch()
        for mapping in (None, {}, [], {"weight": 3}, {"weight": ""}):
            with self.subTest(mapping=mapping):
                with patch("trlx.run_dirs._read_json", return_value={"weight_map": mapping}):
                    with self.assertRaisesRegex(TrlxError, "invalid weight_map"):
                        run_dirs._check_weights(checkpoint)
        mapping = {"weight_map": {"layer": "model-1.safetensors"}}
        shard = checkpoint / "model-1.safetensors"
        with patch("trlx.run_dirs._read_json", return_value=mapping):
            with self.assertRaisesRegex(TrlxError, "shard.*missing or empty"):
                run_dirs._check_weights(checkpoint)
            shard.touch()
            with self.assertRaisesRegex(TrlxError, "shard.*missing or empty"):
                run_dirs._check_weights(checkpoint)
        shard.write_bytes(b"weights")
        with patch("trlx.run_dirs._read_json", side_effect=[mapping, {}]) as read:
            run_dirs._check_weights(checkpoint)
        self.assertEqual(read.call_args.args, (checkpoint / "config.json",))


class Rewind(RunDirsCase):
    # Rewind keeps both metric types at the boundary, selected weights, and all old logs.
    def test_rewind_retains_boundary_and_discards_only_abandoned_progress(self):
        selected = self.checkpoint()
        earlier = self.checkpoint(name="checkpoint-10", step=10)
        later = self.checkpoint(name="checkpoint-40", step=40)
        foreign = self.directory / "external-checkpoint"
        foreign.mkdir()
        (foreign / "weights.txt").write_text("do not follow symlinks")
        link = self.directory / "checkpoint-50"
        link.symlink_to(foreign, target_is_directory=True)
        log = self.directory / show.LOG_FILENAME
        log.write_text("original log\n")
        for name in (show.PREFLIGHT_FILENAME, show.VERIFY_FILENAME):
            (self.directory / name).write_text("{}")
        records = [self.record(10), self.record(20), self.record(20, True),
                   self.record(30), self.record(40, True)]
        metric_path = self.directory / metrics.FILENAME
        metric_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        retained, message = run_dirs.rewind(run_dirs.Resume(selected, 20))
        self.assertEqual(retained, 3)
        self.assertEqual(metrics.read(metric_path), records[:3])
        self.assertTrue(selected.exists())
        self.assertTrue(earlier.exists())
        self.assertFalse(later.exists())
        self.assertFalse(link.is_symlink())
        self.assertEqual((foreign / "weights.txt").read_text(), "do not follow symlinks")
        self.assertEqual(log.read_text(), "original log\n")
        self.assertFalse((self.directory / show.PREFLIGHT_FILENAME).exists())
        self.assertFalse((self.directory / show.VERIFY_FILENAME).exists())
        self.assertIn("2 later metric records", message)
        self.assertIn("2 later checkpoints", message)
        self.assertIn(str(selected), message)

    # Checkpoints without metrics can resume without inventing historical records.
    def test_rewind_without_metrics(self):
        selected = self.checkpoint()
        retained, _ = run_dirs.rewind(run_dirs.Resume(selected, 20))
        self.assertEqual(retained, 0)
        self.assertFalse((self.directory / metrics.FILENAME).exists())
        self.assertTrue(selected.exists())

    # Corruption and invalid step values fail before any checkpoint or report deletion.
    def test_invalid_metrics_leave_artifacts_untouched(self):
        selected = self.checkpoint()
        later = self.checkpoint(name="checkpoint-40", step=40)
        report = self.directory / show.VERIFY_FILENAME
        report.write_text("{}")
        metric_path = self.directory / metrics.FILENAME
        invalid = ["{bad\n", "{}", json.dumps(self.record(True)) + "\n",
                   json.dumps(self.record(-1)) + "\n", json.dumps(self.record("20")) + "\n"]
        for text in invalid:
            with self.subTest(text=text):
                metric_path.write_text(text)
                with self.assertRaises(TrlxError):
                    run_dirs.rewind(run_dirs.Resume(selected, 20))
                self.assertEqual(metric_path.read_text(), text)
                self.assertTrue(selected.exists())
                self.assertTrue(later.exists())
                self.assertTrue(report.exists())

    # A failed filesystem cleanup names the problem and preserves the resume checkpoint.
    def test_cleanup_failure_preserves_selected_checkpoint(self):
        selected = self.checkpoint()
        later = self.checkpoint(name="checkpoint-40", step=40)
        with patch("trlx.run_dirs.shutil.rmtree", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(TrlxError, "checkpoint-40.*cleanup failed.*preserved"):
                run_dirs.rewind(run_dirs.Resume(selected, 20))
        self.assertTrue((selected / "trainer_state.json").is_file())
        self.assertTrue(later.exists())


class AtomicPublication(RunDirsCase):
    # Replacing text preserves the prior permissions and does not leave staging files.
    def test_atomic_replacement_preserves_mode(self):
        destination = self.directory / "history.txt"
        destination.write_text("old")
        destination.chmod(0o640)
        run_dirs.write_atomic(destination, "new\n")
        self.assertEqual(destination.read_text(), "new\n")
        self.assertEqual(destination.stat().st_mode & 0o777, 0o640)
        self.assertEqual(list(self.directory.iterdir()), [destination])

    # Publication failure must leave the previous artifact intact for a later retry.
    def test_failed_replace_preserves_original(self):
        destination = self.directory / "history.txt"
        destination.write_text("old")
        with patch("trlx.run_dirs.pathlib.Path.replace", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                run_dirs.write_atomic(destination, "new")
        self.assertEqual(destination.read_text(), "old")
        self.assertEqual(list(self.directory.iterdir()), [destination])
