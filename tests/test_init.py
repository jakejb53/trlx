"""Generated defaults are complete TOML without models, datasets, or GPU use."""

import dataclasses
import pathlib
import tempfile
import tomllib
import unittest
from unittest.mock import mock_open, patch

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
        with patch("trlx.init_cmd.pathlib.Path.exists", return_value=True):
            with patch("trlx.init_cmd.hardware.inspect") as inspect:
                with patch("trlx.init_cmd.pathlib.Path.open") as opening:
                    with self.assertRaises(TrlxError) as caught:
                        init_cmd.write()
                    self.assertEqual(str(caught.exception), "run.toml already exists. Use --force to overwrite it.")
        inspect.assert_not_called()
        opening.assert_not_called()

    # Exclusive creation also refuses a competing writer that wins after inspection.
    def test_write_refuses_creation_race(self):
        with patch("trlx.init_cmd.pathlib.Path.exists", return_value=False):
            with patch("trlx.init_cmd.hardware.inspect", return_value=Hardware(8, ())):
                with patch("trlx.init_cmd.pathlib.Path.open", side_effect=FileExistsError):
                    with self.assertRaisesRegex(TrlxError, "Use --force to overwrite it"):
                        init_cmd.write()

    # The CLI receives the same measured system used to produce the persisted defaults.
    def test_write_returns_snapshot_and_defaults_to_run_toml(self):
        system = Hardware(8, ())
        opening = mock_open()
        with patch("trlx.init_cmd.pathlib.Path.exists", return_value=False):
            with patch("trlx.init_cmd.hardware.inspect", return_value=system):
                with patch("trlx.init_cmd.pathlib.Path.open", opening):
                    self.assertIs(init_cmd.write(), system)
        opening.assert_called_once_with("x", encoding="utf-8")
        self.assertEqual(tomllib.loads(opening().write.call_args.args[0]), self.document(system))

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
                self.assertEqual(list(path.parent.iterdir()), [path])
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
                    with self.assertRaisesRegex(TrlxError, "cannot write: replace failed"):
                        init_cmd.write(str(path), force=True)
            self.assertEqual(path.read_text(), "operator settings")
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
