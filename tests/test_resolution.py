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

    # The optional split selection survives CLI resolution and worker snapshots for every trainer.
    def test_random_eval_split_round_trip_for_all_methods(self):
        for method in cli.METHODS:
            extra = ["--shuffle-eval-data", "--data-seed", "0", "--seed", "17"]
            if method == "distillation":
                extra += ["--teacher", "teacher"]
            if method in ("grpo", "rloo"):
                extra += ["--reward", "json_valid"]
            with self.subTest(method=method):
                cfg = self.load(method, extra)
                self.assertTrue(cfg.dataset.shuffle_eval_data)
                with patch.object(config, "_read_toml", return_value=self.snapshot(cfg)):
                    worker = config.load("snapshot", method, resolved=True)
                self.assertEqual(worker.dataset, cfg.dataset)
                self.assertEqual(worker.args.data_seed, 0)
                self.assertEqual(worker.args.seed, 17)

    # Explicit disabling overrides saved membership selection; absence keeps the old default.
    def test_random_eval_split_default_and_override(self):
        self.assertFalse(self.load().dataset.shuffle_eval_data)
        self.source["dataset"]["shuffle_eval_data"] = True
        self.assertTrue(self.load().dataset.shuffle_eval_data)
        self.assertFalse(self.load(extra=["--no-shuffle-eval-data"]).dataset.shuffle_eval_data)

    # Selection applies only to percentage splitting and must be a real boolean.
    def test_random_eval_split_rejects_incompatible_modes_and_types(self):
        for extra in (["--no-split"], ["--no-split", "--dataset-eval", "eval.jsonl"],
                      ["--synthetic-dataset-eval"]):
            with self.subTest(extra=extra), self.assertRaisesRegex(TrlxError, "percentage split"):
                self.load(extra=["--shuffle-eval-data", *extra])
        self.source["dataset"]["shuffle_eval_data"] = "yes"
        with self.assertRaisesRegex(TrlxError, "shuffle_eval_data must be bool"):
            self.load()

    # Exercise the actual assessment loading boundary, including data_seed=0 precedence.
    def test_assessment_passes_effective_split_seed(self):
        for extra, expected in ((["--seed", "17"], 17), (["--seed", "17", "--data-seed", "0"], 0)):
            cfg = self.load(extra=["--shuffle-eval-data", *extra])
            with patch.object(train.model_mod, "load_config"), \
                 patch.object(train, "_assessment_metadata", return_value={}), \
                 patch.object(train.model_mod, "assessment_processor"), \
                 patch.object(train.data_load, "load", side_effect=RuntimeError("loading boundary")) as load:
                with self.assertRaisesRegex(RuntimeError, "loading boundary"):
                    train._assess(cfg, 1)
            self.assertEqual(load.call_args.kwargs["seed"], expected)

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

    # Explicit source roles replace configured split settings and survive worker loading.
    def test_separate_source_flags_select_mode(self):
        for flags, expected_train, expected_eval in (
            (["--dataset-train", "train.jsonl", "--dataset-eval", "eval.jsonl"], "train.jsonl", "eval.jsonl"),
            (["--dataset-train", "train.jsonl"], "train.jsonl", None),
            (["--dataset-eval", "eval.jsonl"], "configured.jsonl", "eval.jsonl"),
        ):
            with self.subTest(flags=flags):
                self.source["dataset"]["dataset"] = "configured.jsonl"
                self.source["dataset"]["shuffle_eval_data"] = True
                args = cli.parse_args(["sft", "--model", "chosen-model", "--use-cpu", *flags])
                with patch.object(config, "_read_toml", return_value=self.source):
                    cfg = config.load(args.config, "sft", overrides=options.overrides(args))
                self.assertFalse(cfg.dataset.split)
                self.assertFalse(cfg.dataset.shuffle_eval_data)
                self.assertNotIn("dataset", cfg.document["dataset"])
                self.assertNotIn("eval_fraction", cfg.document["dataset"])
                self.assertEqual(cfg.dataset.source.source, expected_train)
                self.assertEqual(cfg.dataset.eval_source.source if cfg.dataset.eval_source else None, expected_eval)
                if expected_eval is None:
                    self.assertEqual(cfg.args.eval_strategy, "no")
                    self.assertFalse(any(key.startswith("eval_") for key in cfg.document))
                with patch.object(config, "_read_toml", return_value=self.snapshot(cfg)):
                    worker = config.load("snapshot", "sft", resolved=True)
                self.assertEqual(worker.dataset, cfg.dataset)

    # Contradictory CLI modes identify the supplied flags and explain the remedy.
    def test_separate_source_flags_reject_explicit_split(self):
        for flag in ("--dataset-train", "--dataset-eval"):
            with self.subTest(flag=flag):
                args = cli.parse_args(["sft", "--split", flag, "data.jsonl"])
                with patch.object(config, "_read_toml", return_value=self.source):
                    with self.assertRaisesRegex(TrlxError, f"--split conflicts with {flag}; remove --split"):
                        config.load(args.config, "sft", overrides=options.overrides(args))

    # Mode inference must not silently discard explicit evaluation settings.
    def test_implied_no_split_rejects_incompatible_explicit_settings(self):
        for extra, message in ((["--shuffle-eval-data"], "percentage split"),
                               (["--eval-fraction", "0.2"], "conflicts"),
                               (["--synthetic-dataset-eval"], "synthetic-dataset-eval conflicts")):
            with self.subTest(extra=extra), self.assertRaisesRegex(TrlxError, message):
                self.load(extra=["--dataset-eval", "eval.jsonl", *extra])

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
        with patch.object(train.run_dirs, "write_atomic") as write:
            path = train._write_snapshot(cfg, pathlib.Path("memory-run"), ["device"], "single")
        self.assertEqual(path, pathlib.Path("memory-run/config.toml"))
        self.assertEqual(cfg.document, before)
        return tomllib.loads(write.call_args.args[1])

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
             patch.object(train.data_load, "load", return_value=(Mock(num_rows=3), Mock(num_rows=1))), \
             patch.object(train.preflight, "Report"), \
             patch.object(train.preflight, "callback_class"), \
             patch.object(train.preflight, "check_trainer"), \
             patch.object(train.metrics, "callback_class"), \
             patch.object(train, "build_trainer", return_value=trainer) as build:
            self.assertEqual(train._worker(args), 0)
        read.assert_called_once_with(args.config)
        self.assertEqual(build.call_args.args[0].args.learning_rate, 0.001)
        trainer.train.assert_called_once_with(resume_from_checkpoint=None)

    # All trainers snapshot the allocated directory so workers cannot allocate again.
    def test_all_methods_snapshot_actual_run_directory(self):
        for method in cli.METHODS:
            with self.subTest(method=method):
                extra = ["--teacher", "chosen-teacher"] if method == "distillation" else []
                if method in ("grpo", "rloo"):
                    extra = ["--reward", "json_valid"]
                cfg = self.load(method, extra)
                parent = cfg.args.output_dir
                actual = pathlib.Path(parent) / "20260920-1--chosen-model--data"
                with patch.object(train.run_dirs, "allocate", return_value=actual) as allocate:
                    self.assertEqual(train._create_run_dir(cfg), actual)
                allocate.assert_called_once_with(parent, cfg.model.path, cfg.dataset.source)
                self.assertEqual(cfg.args.run_name, actual.name)
                self.assertEqual(cfg.document["output_dir"], str(actual))
                snapshot = self.snapshot(cfg)
                with patch.object(config, "_read_toml", return_value=snapshot):
                    worker = config.load("snapshot", method, resolved=True)
                self.assertEqual(worker.args.output_dir, str(actual))
                self.assertEqual(worker.args.run_name, actual.name)

    # Explicit run labels remain display metadata rather than choosing directory names.
    def test_allocation_preserves_explicit_run_name(self):
        cfg = self.load(extra=["--run-name", "experiment"])
        actual = pathlib.Path("runs/sft/20260920-1--chosen-model--data")
        with patch.object(train.run_dirs, "allocate", return_value=actual):
            train._create_run_dir(cfg)
        self.assertEqual(cfg.args.run_name, "experiment")
        self.assertEqual(cfg.args.output_dir, str(actual))

    # Explicit resume needs only its saved settings; today's source is never read.
    def test_minimal_resume_uses_snapshot_without_source(self):
        snapshot = self.snapshot(self.load())
        checkpoint = pathlib.Path("memory-run/checkpoint-20").resolve()
        args = cli.parse_args(["sft", "--resume-from-checkpoint", str(checkpoint)])
        with patch.object(config, "_read_toml", return_value=snapshot) as read:
            cfg = config.load(args.config, "sft", overrides=options.overrides(args))
        read.assert_called_once_with(checkpoint.parent / "config.toml")
        self.assertEqual(cfg.model.path, "chosen-model")
        self.assertEqual(cfg.dataset.source.source, "data.jsonl")
        self.assertEqual(cfg.args.resume_from_checkpoint, str(checkpoint))
        self.assertEqual(cfg.args.output_dir, str(checkpoint.parent))
        with patch.object(train.run_dirs, "allocate") as allocate:
            self.assertEqual(train._create_run_dir(cfg), checkpoint.parent)
        allocate.assert_not_called()

    # Resume CLI controls override the snapshot without reviving source-file settings.
    def test_resume_applies_explicit_controls_and_training_overrides(self):
        snapshot = self.snapshot(self.load())
        checkpoint = pathlib.Path("memory-run/checkpoint-20").resolve()
        args = cli.parse_args(["sft", "--resume-from-checkpoint", str(checkpoint),
                               "--gpus", "0", "--tui", "--no-verify", "--learning-rate", "0.007"])
        with patch.object(config, "_read_toml", return_value=snapshot) as read:
            cfg = config.load(args.config, "sft", overrides=options.overrides(args))
        read.assert_called_once_with(checkpoint.parent / "config.toml")
        self.assertEqual(cfg.args.learning_rate, 0.007)
        self.assertEqual(config.run_settings(cfg.document),
                         {"gpus": "0", "strategy": "auto", "tui": True, "verify": False})

    # A configured checkpoint is discovered from source, then its snapshot owns inputs.
    def test_configured_resume_reads_source_then_snapshot(self):
        snapshot = self.snapshot(self.load())
        checkpoint = pathlib.Path("memory-run/checkpoint-20").resolve()
        source = copy.deepcopy(self.source)
        source["resume_from_checkpoint"] = str(checkpoint)
        source["model"] = {"path": "changed-source-model"}
        with patch.object(config, "_read_toml", side_effect=[source, snapshot]) as read:
            cfg = config.load("operator.toml", "sft")
        self.assertEqual([call.args[0] for call in read.call_args_list],
                         ["operator.toml", checkpoint.parent / "config.toml"])
        self.assertEqual(cfg.model.path, "chosen-model")

    # Output overrides cannot relocate a continuation away from the selected checkpoint.
    def test_resume_output_override_names_parent_only(self):
        snapshot = self.snapshot(self.load())
        checkpoint = pathlib.Path("memory-run/checkpoint-20").resolve()
        overrides = {"resume_from_checkpoint": str(checkpoint),
                     "output_dir": str(checkpoint.parent.parent)}
        with patch.object(config, "_read_toml", return_value=snapshot):
            cfg = config.load("unused.toml", "sft", overrides=overrides)
            self.assertEqual(cfg.args.output_dir, str(checkpoint.parent))
            overrides["output_dir"] = str(checkpoint.parent)
            with self.assertRaisesRegex(TrlxError, "omit --output-dir"):
                config.load("unused.toml", "sft", overrides=overrides)

    # Historical schemas are rejected at the snapshot path rather than converted.
    def test_resume_rejects_legacy_dataset_schema(self):
        snapshot = self.snapshot(self.load())
        snapshot["dataset"]["train"] = 60
        snapshot["dataset"].pop("eval_fraction")
        checkpoint = pathlib.Path("memory-run/checkpoint-20").resolve()
        with patch.object(config, "_read_toml", return_value=snapshot):
            with self.assertRaisesRegex(TrlxError, "config.toml.*train"):
                config.load("unused.toml", "sft", overrides={"resume_from_checkpoint": str(checkpoint)})

    # Without method metadata a snapshot cannot safely select a trainer on resume.
    def test_resume_requires_snapshot_method(self):
        snapshot = self.snapshot(self.load())
        snapshot.pop("launch")
        with patch.object(config, "_read_toml", return_value=snapshot):
            with self.assertRaisesRegex(TrlxError, "does not describe method"):
                config.load("unused.toml", "sft", overrides={"resume_from_checkpoint": "memory-run/checkpoint-20"})

    # Every launched worker receives the immutable resolved path and internal strategy.
    def test_spawn_worker_arguments(self):
        collector = Mock()
        launch.spawn("sft", "memory-run/config.toml", "single", ["device"], collector)
        argv = collector.spawn.call_args.args[0]
        self.assertEqual(argv[4:], ["--config", "memory-run/config.toml", "--_rank", "0", "--_strategy", "single"])
        self.assertEqual(collector.spawn.call_args.args[2], "rank 0")


if __name__ == "__main__":
    unittest.main()
