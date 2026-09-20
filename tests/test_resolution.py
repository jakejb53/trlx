"""One effective configuration owns CLI precedence, worker inputs, and snapshots."""

import copy
import pathlib
import tomllib
import types
import unittest
from unittest.mock import Mock, patch

from trlx import TrlxError, cli, config, init_cmd, launch, options, preflight, train
from trlx.hardware import Hardware


class Resolution(unittest.TestCase):
    # Hardware is synthetic; all configuration work remains in memory on CPU.
    def setUp(self):
        self.source = tomllib.loads(init_cmd.render(Hardware(4, ())))

    # Exercise actual argument parsing and config validation with a fake reader.
    def load(self, method="sft", extra=()):
        argv = [method, "--model", "chosen-model", "--dataset", "data.jsonl", "--use-cpu", *extra]
        args = cli.parse_args(argv)
        with patch.object(config, "_read_toml", return_value=self.source):
            return config.load(args.config, method, overrides=options.overrides(args))

    # CLI wins over the selected method, which wins over shared settings.
    def test_precedence_without_source_mutation(self):
        self.source["learning_rate"] = 0.03
        self.source["methods"]["sft"]["learning_rate"] = 0.02
        before = copy.deepcopy(self.source)
        self.assertEqual(self.load().args.learning_rate, 0.02)
        cfg = self.load(extra=["--learning-rate", "0.01", "--lora-r", "16"])
        self.assertEqual(cfg.args.learning_rate, 0.01)
        self.assertEqual(cfg.peft.r, 16)
        self.assertEqual(self.source, before)
        self.assertNotIn("methods", cfg.document)
        self.assertNotIn("teacher", cfg.document)
        self.assertNotIn("rewards", cfg.document)

    # The default document supports every method without manual file edits.
    def test_all_methods_use_shared_config(self):
        for method in cli.METHODS:
            extra = []
            if method == "distillation":
                extra = ["--teacher", "chosen-teacher"]
            if method in ("grpo", "rloo"):
                extra = ["--reward", "json_valid"]
            with self.subTest(method=method):
                cfg = self.load(method, extra)
                self.assertEqual(cfg.model.path, "chosen-model")
                self.assertEqual(cfg.dataset.eval_fraction, 0.1)
                self.assertEqual(cfg.peft.target_modules, "all-linear")

    # Disabling LoRA affects one resolved run and keeps persistent defaults intact.
    def test_no_lora(self):
        cfg = self.load(extra=["--no-lora"])
        self.assertIsNone(cfg.peft)
        self.assertNotIn("peft", cfg.document)
        self.assertIn("peft", self.source)

    # Nullable overrides survive the TOML snapshot and are decoded by each consumer.
    def test_nullable_lora_and_boolean_round_trip(self):
        cfg = self.load(extra=["--lora-target-modules", "None", "--tf32", "None"])
        self.assertIsNone(cfg.peft.target_modules)
        self.assertIsNone(cfg.args.tf32)
        snapshot = self.snapshot(cfg)
        with patch.object(config, "_read_toml", return_value=snapshot):
            worker = config.load("snapshot", "sft", resolved=True)
        self.assertIsNone(worker.peft.target_modules)
        self.assertIsNone(worker.args.tf32)

    # CLI data overrides must also work with an existing separate-file config.
    def test_dataset_override_respects_persistent_split_mode(self):
        self.source["dataset"] = {"split": False, "dataset_train": "old.jsonl", "dataset_eval": "eval.jsonl"}
        cfg = self.load()
        self.assertEqual(cfg.dataset.source.source, "data.jsonl")
        self.assertEqual(cfg.dataset.eval_source.source, "eval.jsonl")

    # Switching to separate files cannot retain the generated fractional-split keys.
    def test_no_split_with_evaluation(self):
        cfg = self.load(extra=["--no-split", "--dataset-eval", "eval.jsonl"])
        self.assertFalse(cfg.dataset.split)
        self.assertTrue(cfg.dataset.eval_enabled)
        self.assertNotIn("eval_fraction", cfg.document["dataset"])
        self.assertEqual(cfg.document["dataset"]["dataset_train"], "data.jsonl")

    # --no-split without an eval file deliberately removes the generated schedule.
    def test_cli_training_only(self):
        cfg = self.load(extra=["--no-split"])
        self.assertIsNone(cfg.dataset.eval_source)
        self.assertEqual(cfg.args.eval_strategy, "no")
        self.assertFalse(any(key.startswith("eval_") for key in cfg.document))

    # Contradictory explicit options are errors, never silently discarded.
    def test_conflicting_dataset_overrides(self):
        for flags in (["--no-split", "--eval-fraction", "0.2"],
                      ["--split", "--dataset-eval", "eval.jsonl"],
                      ["--no-split", "--eval-steps", "10"],
                      ["--no-split", "--dataset-train", "different.jsonl"]):
            with self.subTest(flags=flags), self.assertRaises(TrlxError):
                self.load(extra=flags)

    # Display/GPU overrides retain other persistent run controls.
    def test_run_controls(self):
        self.source["run"]["tui"] = True
        self.source["run"]["verify"] = False
        cfg = self.load(extra=["--no-tui", "--verify", "--gpus", "0"])
        self.assertEqual(config.run_settings(cfg.document), {
            "gpus": "0", "strategy": "auto", "tui": False, "verify": True,
        })

    # Flat configs keep their prior launch defaults when a CLI control is supplied.
    def test_run_override_on_flat_config(self):
        del self.source["run"]
        cfg = self.load(extra=["--tui"])
        self.assertEqual(config.run_settings(cfg.document)["gpus"], "all")
        self.assertTrue(config.run_settings(cfg.document)["tui"])

    # Optional task inputs must be supplied explicitly rather than invented by init.
    def test_missing_teacher_or_reward_is_clear(self):
        for method, expected in (("distillation", "path"), ("grpo", "funcs")):
            with self.subTest(method=method), self.assertRaisesRegex(TrlxError, expected):
                self.load(method)

    # Snapshot writing serializes effective inputs and does not mutate their owner.
    def snapshot(self, cfg):
        before = copy.deepcopy(cfg.document)
        with patch.object(pathlib.Path, "write_text") as write:
            path = train._write_snapshot(cfg, pathlib.Path("memory-run"), ["device"], "single")
        self.assertEqual(path, pathlib.Path("memory-run/config.toml"))
        self.assertEqual(cfg.document, before)
        return tomllib.loads(write.call_args.args[0])

    # A worker must obtain the override from the snapshot after the source changes.
    def test_snapshot_reloads_exact_effective_inputs(self):
        cfg = self.load(extra=["--learning-rate", "0.001", "--lora-r", "16"])
        snapshot = self.snapshot(cfg)
        self.source["methods"]["sft"]["learning_rate"] = 0.1
        with patch.object(config, "_read_toml", return_value=snapshot) as read:
            worker = config.load("memory-run/config.toml", "sft", resolved=True)
        read.assert_called_once_with("memory-run/config.toml")
        self.assertEqual(worker.args.learning_rate, cfg.args.learning_rate)
        self.assertEqual(worker.peft.r, 16)
        self.assertEqual(worker.model.path, "chosen-model")
        self.assertEqual(preflight.compare_snapshot(cfg.document, snapshot, "single"), [])

    # Worker mode rejects method mismatches and late overrides.
    def test_worker_snapshot_constraints(self):
        snapshot = self.snapshot(self.load())
        with patch.object(config, "_read_toml", return_value=snapshot):
            with self.assertRaisesRegex(TrlxError, "does not describe method"):
                config.load("snapshot", "dpo", resolved=True)
            with self.assertRaisesRegex(TrlxError, "cannot override"):
                config.load("snapshot", "sft", resolved=True, overrides={"learning_rate": 0.5})

    # Actual worker entry consumes the snapshot, not the original settings file.
    def test_worker_entry_uses_snapshot(self):
        snapshot = self.snapshot(self.load(extra=["--learning-rate", "0.001"]))
        args = types.SimpleNamespace(command="sft", config="memory-run/config.toml", _rank=0,
                                     _strategy="single", overrides={})
        trainer = Mock()
        with patch.object(config, "_read_toml", return_value=snapshot) as read, \
             patch.object(train, "_attach_logging"), \
             patch.object(train.model_mod, "load_model"), \
             patch.object(train.model_mod, "load_processor"), \
             patch.object(train.data_load, "load", return_value=(Mock(), Mock())), \
             patch.object(train.preflight, "Report"), \
             patch.object(train.preflight, "callback_class"), \
             patch.object(train.preflight, "check_trainer"), \
             patch.object(train.metrics, "callback_class"), \
             patch.object(train, "build_trainer", return_value=trainer) as build:
            self.assertEqual(train._worker(args), 0)
        read.assert_called_once_with(args.config)
        self.assertEqual(build.call_args.args[0].args.learning_rate, 0.001)
        trainer.train.assert_called_once_with(resume_from_checkpoint=None)

    # Every launched worker receives the immutable resolved path and internal strategy.
    def test_spawn_worker_arguments(self):
        with patch.object(launch.subprocess, "Popen") as popen:
            launch.spawn("sft", "memory-run/config.toml", "single", ["device"], Mock())
        argv = popen.call_args.args[0]
        self.assertEqual(argv[4:], ["--config", "memory-run/config.toml", "--_rank", "0", "--_strategy", "single"])


if __name__ == "__main__":
    unittest.main()
