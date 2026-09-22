"""Generated defaults are complete TOML without models, datasets, or GPU use."""

import dataclasses
import pathlib
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from trlx import TrlxError, init_cmd, trainers
from trlx.hardware import Gpu, Hardware


class InitDefaults(unittest.TestCase):
    # Rendering consumes an existing hardware snapshot and must not inspect again.
    def document(self, system):
        with patch("trlx.init_cmd.hardware.inspect", side_effect=AssertionError("unexpected inspection")):
            return tomllib.loads(init_cmd.render(system))

    # A CPU host can prepare defaults but must not claim usable CUDA precision.
    def test_cpu_defaults(self):
        document = self.document(Hardware(8, ()))
        self.assertEqual(document["model"]["dtype"], "float32")
        self.assertFalse(document["bf16"])
        self.assertFalse(document["fp16"])
        self.assertFalse(document["dataloader_pin_memory"])
        self.assertEqual(document["dataloader_num_workers"], 0)

    # Every visible GPU must support native BF16, including heterogeneous systems.
    def test_precision_uses_all_devices(self):
        capable = Gpu(0, "native", 24 * 2**30, 20 * 2**30, True)
        old = Gpu(1, "older", 8 * 2**30, 5 * 2**30, False)
        for devices, dtype, bf16 in [
            ((capable,), "bfloat16", True),
            ((capable, old), "float32", False),
        ]:
            with self.subTest(devices=devices):
                document = self.document(Hardware(16, devices))
                self.assertEqual(document["bf16"], bf16)
                self.assertEqual(document["model"]["dtype"], dtype)
                self.assertEqual(document["methods"]["distillation"]["teacher"]["dtype"], dtype)
                self.assertTrue(document["dataloader_pin_memory"])

    # One config covers all methods while missing task inputs stay absent, never fake paths.
    def test_all_methods_and_explicit_objectives(self):
        document = self.document(Hardware(8, ()))
        self.assertEqual(set(document["methods"]), set(trainers.METHODS))
        self.assertNotIn("path", document["model"])
        self.assertEqual(document["dataset"], {"split": True, "eval_fraction": 0.1})
        for name, method in trainers.METHODS.items():
            with self.subTest(method=name):
                section = document["methods"][name]
                self.assertEqual(section["output_dir"], "runs/" + name)
                self.assertEqual(section["ranges"], init_cmd.RANGES[name])
                fields = {field.name: field for field in dataclasses.fields(method.config_cls)}
                expected_rate = 1e-4 if name in {"sft", "reward", "distillation"} else fields["learning_rate"].default
                self.assertEqual(section["learning_rate"], expected_rate)
                if "rewards" in method.blocks:
                    self.assertEqual(section["rewards"], {})
                    self.assertEqual(section["num_generations_eval"], 1)
                    self.assertGreater(section["num_generations"], 1)
                    self.assertEqual(document["gradient_accumulation_steps"] % section["num_generations"], 0)
                if "preflight" in method.blocks:
                    self.assertEqual(section["preflight"], {"rows": 64, "offpolicy_logp_per_token": -1.0})
                if "teacher" in method.blocks:
                    self.assertNotIn("path", section["teacher"])

    # The first-run preset trains adapters and produces evaluation/checkpoints in one epoch.
    def test_first_run_preset(self):
        document = self.document(Hardware(8, ()))
        self.assertNotIn("verify", document)
        self.assertEqual(document["run"], {
            "gpus": "all", "strategy": "auto", "tui": False, "verify": True,
        })
        self.assertEqual(document["peft"], {
            "r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": "all-linear",
        })
        self.assertEqual(document["num_train_epochs"], 1.0)
        self.assertEqual(document["max_steps"], -1)
        self.assertEqual(document["per_device_train_batch_size"], 1)
        self.assertEqual(document["gradient_accumulation_steps"], 8)
        self.assertEqual(document["per_device_eval_batch_size"], 1)
        self.assertEqual(document["eval_strategy"], "epoch")
        self.assertEqual(document["save_strategy"], "epoch")
        self.assertEqual(document["save_total_limit"], 2)
        self.assertEqual(document["logging_steps"], 1)
        self.assertTrue(document["gradient_checkpointing"])

    # The generated file distinguishes measured facts from unknown model requirements.
    def test_hardware_comments_and_terminal_width(self):
        text = init_cmd.render(Hardware(16, (Gpu(0, "Example GPU", 24 * 2**30, 20 * 2**30, True),)))
        self.assertIn("16 logical CPUs", text)
        self.assertIn("20.0 GiB free /", text)
        self.assertIn("24.0 GiB total", text)
        self.assertIn("not calibrated", text)
        self.assertLessEqual(max(map(len, text.splitlines())), 120)

    # Existing operator settings must be rejected before CUDA inspection or writing.
    def test_write_refuses_existing_file(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            path = pathlib.Path(folder) / "run.toml"
            path.write_text("operator settings", encoding="utf-8")
            with patch("trlx.init_cmd.hardware.inspect") as inspect:
                with self.assertRaisesRegex(TrlxError, "--force"):
                    init_cmd.write(path)
            self.assertEqual(path.read_text(), "operator settings")
        inspect.assert_not_called()

    # Prompt collisions must also fail before inspection or any config publication.
    def test_existing_prompt_rejected_before_inspection(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            path = pathlib.Path(folder) / "custom.toml"
            prompt = path.parent / "prompts" / "chat-questions.prompt"
            prompt.parent.mkdir()
            prompt.write_text("operator instructions", encoding="utf-8")
            with patch("trlx.init_cmd.hardware.inspect") as inspect:
                with self.assertRaisesRegex(TrlxError, "--force"):
                    init_cmd.write(path)
            inspect.assert_not_called()
            self.assertFalse(path.exists())
            self.assertEqual(prompt.read_text(), "operator instructions")

    # Only generated names are replaced; an active operator rubric remains theirs.
    def test_force_resets_prompts_and_preserves_operator_rubric(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            path = pathlib.Path(folder) / "custom.toml"
            prompt = path.parent / "prompts" / "chat-questions.prompt"
            prompt.parent.mkdir()
            prompt.write_text("edited instructions", encoding="utf-8")
            rubric = prompt.parent / "llm-judge.prompt"
            rubric.write_text("operator objective", encoding="utf-8")
            with patch("trlx.init_cmd.hardware.inspect", return_value=Hardware(8, ())):
                init_cmd.write(path, force=True)
            self.assertIn("Write {n} questions", prompt.read_text())
            self.assertEqual(rubric.read_text(), "operator objective")
            document = tomllib.loads(path.read_text())
            for key, value in document["prompts"].items():
                self.assertTrue((path.parent / value).is_file(), key)
            example = (prompt.parent / "llm-judge.prompt.example").read_text()
            self.assertIn("EXAMPLE 1", example)
            self.assertIn("EXAMPLE 2", example)

    # A destination created during hardware inspection must not be silently replaced.
    def test_write_refuses_creation_race(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            path = pathlib.Path(folder) / "run.toml"

            # Simulate another writer claiming the destination after initial validation.
            def inspect(*, progress=None):
                path.write_text("competing settings", encoding="utf-8")
                return Hardware(8, ())

            with patch("trlx.init_cmd.hardware.inspect", side_effect=inspect):
                with self.assertRaisesRegex(TrlxError, "--force"):
                    init_cmd.write(path)
            self.assertEqual(path.read_text(), "competing settings")

    # The CLI receives the same measured system used to produce the persisted defaults.
    def test_write_returns_snapshot_and_defaults_to_run_toml(self):
        system = Hardware(8, ())
        with patch("trlx.init_cmd.validate_output"), patch("trlx.init_cmd.validate_text_outputs"):
            with patch("trlx.init_cmd.hardware.inspect", return_value=system):
                with patch("trlx.init_cmd.write_many_text") as write:
                    self.assertIs(init_cmd.write(), system)
        outputs = dict(write.call_args.args[0])
        self.assertEqual(tomllib.loads(outputs[pathlib.Path("run.toml")]), self.document(system))
        self.assertEqual(set(outputs), {pathlib.Path("run.toml"),
                         *(pathlib.Path("prompts") / name for name in init_cmd.PROMPT_FILES)})
        self.assertEqual(write.call_args.kwargs, {"force": False, "no_staging": False, "progress": None})

    # Forced generation replaces settings explicitly, also permitting a new path.
    def test_force_writes_fresh_defaults(self):
        system = Hardware(8, ())
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
                path = pathlib.Path(folder) / "run.toml"
                if existing:
                    path.write_text("operator settings", encoding="utf-8")
                    mode = path.stat().st_mode & 0o777
                with patch("trlx.init_cmd.hardware.inspect", return_value=system):
                    self.assertIs(init_cmd.write(str(path), force=True), system)
                self.assertEqual(tomllib.loads(path.read_text()), self.document(system))
                prompt_dir = path.parent / "prompts"
                self.assertEqual(set(path.parent.iterdir()), {path, prompt_dir})
                self.assertEqual({item.name for item in prompt_dir.iterdir()}, set(init_cmd.PROMPT_FILES))
                self.assertFalse((prompt_dir / "llm-judge.prompt").exists())
                if existing:
                    self.assertEqual(path.stat().st_mode & 0o777, mode)

    # A failed hardware probe must leave the operator's existing configuration intact.
    def test_force_preserves_config_on_inspection_failure(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            path = pathlib.Path(folder) / "run.toml"
            path.write_text("operator settings", encoding="utf-8")
            with patch("trlx.init_cmd.hardware.inspect", side_effect=TrlxError("hardware query failed")):
                with self.assertRaisesRegex(TrlxError, "hardware query failed"):
                    init_cmd.write(str(path), force=True)
            self.assertEqual(path.read_text(), "operator settings")

    # Failed publication preserves the old file and cleans the temporary replacement.
    def test_force_preserves_config_on_replace_failure(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            path = pathlib.Path(folder) / "run.toml"
            path.write_text("operator settings", encoding="utf-8")
            with patch("trlx.init_cmd.hardware.inspect", return_value=Hardware(8, ())):
                with patch.object(pathlib.Path, "replace", side_effect=OSError("replace failed")):
                    with self.assertRaisesRegex(TrlxError, "replace failed"):
                        init_cmd.write(str(path), force=True)
            self.assertEqual(path.read_text(), "operator settings")
            self.assertEqual(list(path.parent.glob("*.toml")), [path])
            self.assertFalse(any((path.parent / "prompts").glob("*.prompt")))

    # Force replaces the named symlink, including dangling links, without touching its target.
    def test_force_replaces_symlink_in_both_write_modes(self):
        for no_staging in (False, True):
            for dangling in (False, True):
                with self.subTest(no_staging=no_staging, dangling=dangling), \
                     tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
                    target = pathlib.Path(folder) / "saved.toml"
                    if not dangling:
                        target.write_text("saved settings", encoding="utf-8")
                    path = pathlib.Path(folder) / "run.toml"
                    path.symlink_to(target.name)
                    with patch("trlx.init_cmd.hardware.inspect", return_value=Hardware(8, ())):
                        with self.assertRaisesRegex(TrlxError, "--force"):
                            init_cmd.write(path, no_staging=no_staging)
                        init_cmd.write(path, force=True, no_staging=no_staging)
                    self.assertFalse(path.is_symlink())
                    self.assertIn("model", tomllib.loads(path.read_text()))
                    if dangling:
                        self.assertFalse(target.exists())
                    else:
                        self.assertEqual(target.read_text(), "saved settings")

    # The output name authorizes replacing the entire old directory, not just config files.
    def test_force_replaces_directory_in_both_write_modes(self):
        for no_staging in (False, True):
            with self.subTest(no_staging=no_staging), \
                 tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
                path = pathlib.Path(folder) / "run.toml"
                path.mkdir()
                (path / "unrelated").write_text("old contents", encoding="utf-8")
                with patch("trlx.init_cmd.hardware.inspect", return_value=Hardware(8, ())):
                    init_cmd.write(path, force=True, no_staging=no_staging)
                self.assertTrue(path.is_file())
                self.assertIn("model", tomllib.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
