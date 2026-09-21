"""Assessment settings, CLI precedence, and explicit defaults without data/model loading."""

import contextlib
import copy
import io
import os
import pathlib
import tomllib
import unittest
from unittest.mock import patch

from trlx import TrlxError, cli, config, data_load, init_cmd, options
from trlx.hardware import Hardware
from trlx.quality_scorers import PRESETS
from tests.test_review import configuration


# These values are the operator-approved generated contract, not runtime fallbacks.
DEFAULTS = {
    "quality_checks": False, "runtime_window": 20, "runtime_min_evaluations": 3,
    "runtime_relative_change": 0.05, "quality_preset": "None", "quality_dataset": "None",
    "quality_max_length": 2048, "quality_max_new_tokens": 256, "quality_batch_size": 1,
}


# Every supplied judge table must state all connection and generation controls.
def judge(**overrides):
    return {"url": "http://localhost:8080/v1", "model": "judge-model", "api_key": "None",
            "timeout": 30.0, "retries": 0, "max_tokens": 128, **overrides}


# Exercise the real wrapper validator independently of unrelated trainer constructors.
def assessment(method="sft", **overrides):
    return config._assessment("memory.toml", {**DEFAULTS, **overrides}, method)


class AssessmentConfiguration(unittest.TestCase):
    # No operational field may acquire an invisible runtime default, even when disabled.
    def test_every_setting_required(self):
        for key in DEFAULTS:
            with self.subTest(key=key):
                table = dict(DEFAULTS)
                del table[key]
                with self.assertRaisesRegex(TrlxError, key):
                    config._assessment("memory.toml", table, "sft")

    # Library inspection remains possible; executable training/check must enforce the block.
    def test_missing_block_inspection_and_execution_boundary(self):
        cfg = configuration()
        self.assertIsNone(cfg.assessment)
        with self.assertRaisesRegex(TrlxError, r"training and check require \[assessment\]"):
            config.require_assessment(cfg, "memory.toml")
        cfg = configuration(extra={"assessment": dict(DEFAULTS)})
        self.assertIs(config.require_assessment(cfg, "memory.toml"), cfg.assessment)
        self.assertFalse(cfg.assessment.quality_checks)
        self.assertIsNone(cfg.assessment.quality_preset)
        self.assertIsNone(cfg.assessment.quality_dataset)

    # All seven actual config classes accept the universal block without loading data.
    def test_block_all_methods(self):
        with patch.object(data_load, "load", side_effect=AssertionError("data load during config")):
            for method in cli.METHODS:
                with self.subTest(method=method):
                    cfg = configuration(method, {"assessment": dict(DEFAULTS)})
                    self.assertEqual(cfg.assessment.runtime_window, 20)

    # Python bool/int subtype behavior must not loosen the operator-visible types.
    def test_boolean_integer_and_numeric_validation(self):
        cases = {
            "quality_checks": (0, 1, "false", None),
            "runtime_window": (True, 1, 2.0, "2"),
            "runtime_min_evaluations": (False, 1, 3.0),
            "quality_max_length": (True, 1, 2.0),
            "quality_max_new_tokens": (False, 0, -1, 1.0),
            "quality_batch_size": (True, 0, 1.0),
            "runtime_relative_change": (True, 0, -0.1, float("nan"), float("inf"), "0.1"),
        }
        for key, invalid in cases.items():
            for value in invalid:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(TrlxError, key):
                    assessment(**{key: value})
        self.assertEqual(assessment(runtime_relative_change=1).runtime_relative_change, 1.0)

    # Unknown keys fail loudly at each ownership boundary.
    def test_unknown_keys_and_wrong_table_type(self):
        with self.assertRaisesRegex(TrlxError, "typo"):
            assessment(typo=1)
        with self.assertRaisesRegex(TrlxError, "typo"):
            assessment(judge=judge(typo=1))
        with self.assertRaisesRegex(TrlxError, "must be a table"):
            configuration(extra={"assessment": []})
        with self.assertRaisesRegex(TrlxError, "must be a table"):
            assessment(judge="None")

    # Enabled quality needs explicit suitable data and a supported built-in preset.
    def test_presets_and_method_restrictions(self):
        for values in ({}, {"quality_preset": "qa"}, {"quality_dataset": "quality.jsonl"}):
            with self.subTest(values=values), self.assertRaisesRegex(TrlxError, "quality_preset and quality_dataset"):
                assessment(quality_checks=True, **values)
        with self.assertRaisesRegex(TrlxError, "quality_preset"):
            assessment(quality_preset="bespoke")
        for preset in PRESETS:
            method = "reward" if preset == "preference" else "sft"
            extra = {"judge": judge()} if preset in {"instruction_following", "writing"} else {}
            with self.subTest(preset=preset):
                result = assessment(method, quality_checks=True, quality_preset=preset,
                                    quality_dataset="quality.jsonl", **extra)
                self.assertEqual(result.quality_dataset.source, "quality.jsonl")
                self.assertEqual(result.quality_preset, preset)
        for method, preset in (("reward", "qa"), ("sft", "preference")):
            with self.subTest(method=method), self.assertRaisesRegex(TrlxError, "reward trainer requires"):
                assessment(method, quality_checks=True, quality_preset=preset, quality_dataset="quality.jsonl")

    # A partially supplied inactive judge table cannot hide timeout/retry defaults.
    def test_judge_explicit_keys_and_inactive_nullable_fields(self):
        for key in judge():
            table = judge()
            del table[key]
            with self.subTest(key=key), self.assertRaisesRegex(TrlxError, key):
                assessment(judge=table)
        result = assessment(judge=judge(url="None", model=None, api_key="None"))
        self.assertIsNone(result.judge["url"])
        self.assertIsNone(result.judge["model"])
        self.assertIsNone(result.judge["api_key"])
        self.assertEqual(result.judge["timeout"], 30.0)
        for key, values in {"timeout": (True, 0, float("nan"), float("inf")),
                            "retries": (True, -1, 1.5), "max_tokens": (False, 0, 1.5)}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaisesRegex(TrlxError, key):
                    assessment(judge=judge(**{key: value}))

    # Credentials remain environment references; parsing checks the connection but sends no requests.
    def test_judge_endpoint_and_secret_validation(self):
        enabled = dict(quality_checks=True, quality_preset="writing", quality_dataset="quality.jsonl")
        with self.assertRaisesRegex(TrlxError, "endpoint URL and model"):
            assessment(**enabled)
        with self.assertRaisesRegex(TrlxError, "endpoint URL and model"):
            assessment(**enabled, judge=judge(model=None))
        with self.assertRaisesRegex(TrlxError, "endpoint URL"):
            assessment(**enabled, judge=judge(url="file:///secret"))
        with self.assertRaisesRegex(TrlxError, "environment variable") as error:
            assessment(judge=judge(api_key="sk-do-not-print-this-secret"))
        self.assertNotIn("sk-do-not-print-this-secret", str(error.exception))
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(TrlxError, "ASSESSMENT_TEST_KEY is not set"):
            assessment(**enabled, judge=judge(api_key="ASSESSMENT_TEST_KEY"))
        with patch.dict(os.environ, {"ASSESSMENT_TEST_KEY": "local-test-value"}, clear=True):
            result = assessment(**enabled, judge=judge(api_key="ASSESSMENT_TEST_KEY"))
        self.assertEqual(result.judge["api_key"], "ASSESSMENT_TEST_KEY")


class AssessmentCLI(unittest.TestCase):
    # The real parser must preserve explicit booleans and leave absent flags absent.
    def test_positive_negative_and_absent_flags(self):
        for command in (["sft"], ["check", "sft"]):
            for flag, expected in (("--quality-checks", True), ("--no-quality-checks", False)):
                with self.subTest(command=command, flag=flag):
                    overrides = options.overrides(cli.parse_args([*command, flag]))
                    self.assertIs(overrides["assessment.quality_checks"], expected)
            self.assertNotIn("assessment.quality_checks", options.overrides(cli.parse_args(command)))

    # Exercise every public assessment option through actual argparse conversion.
    def test_all_cli_controls(self):
        args = cli.parse_args(["sft", "--assessment-window", "6", "--assessment-min-evaluations", "4",
                               "--assessment-relative-change", "0.2", "--quality-preset", "qa",
                               "--quality-dataset", "quality.jsonl", "--quality-max-length", "128",
                               "--quality-max-new-tokens", "64", "--quality-batch-size", "2",
                               "--quality-judge-url", "None", "--quality-judge-model", "None",
                               "--quality-judge-api-key", "None", "--quality-judge-timeout", "10",
                               "--quality-judge-retries", "2", "--quality-judge-max-tokens", "32"])
        values = options.overrides(args)
        self.assertEqual(values["assessment.runtime_window"], 6)
        self.assertEqual(values["assessment.runtime_min_evaluations"], 4)
        self.assertEqual(values["assessment.runtime_relative_change"], 0.2)
        self.assertEqual(values["assessment.quality_max_length"], 128)
        self.assertEqual(values["assessment.quality_max_new_tokens"], 64)
        self.assertEqual(values["assessment.quality_batch_size"], 2)
        self.assertEqual(values["assessment.judge.retries"], 2)
        source = configuration(extra={"assessment": dict(DEFAULTS)}).document
        with patch.object(config, "_read_toml", return_value=source):
            cfg = config.load("memory.toml", "sft", overrides=values)
        self.assertIsNone(cfg.assessment.judge["api_key"])
        self.assertEqual(cfg.assessment.judge["timeout"], 10.0)

    # The method overlay and CLI merge nested assessment fields without mutating shared input.
    def test_shared_method_cli_precedence(self):
        source = configuration(extra={"assessment": dict(DEFAULTS)}).document
        source["methods"] = {"sft": {"assessment": {"runtime_window": 8, "quality_checks": True,
                                                    "quality_preset": "qa", "quality_dataset": "held-out.jsonl"}}}
        before = copy.deepcopy(source)
        with patch.object(config, "_read_toml", return_value=source):
            selected = config.load("memory.toml", "sft")
            values = options.overrides(cli.parse_args(["sft", "--assessment-window", "5", "--no-quality-checks"]))
            overridden = config.load("memory.toml", "sft", overrides=values)
        self.assertEqual(selected.assessment.runtime_window, 8)
        self.assertTrue(selected.assessment.quality_checks)
        self.assertEqual(overridden.assessment.runtime_window, 5)
        self.assertFalse(overridden.assessment.quality_checks)
        self.assertEqual(overridden.assessment.runtime_min_evaluations, 3)
        self.assertEqual(overridden.assessment.quality_preset, "qa")
        self.assertEqual(source, before)

    # Explicit resume reads assessment settings from its snapshot, never today's run.toml.
    def test_resume_uses_snapshot_then_cli(self):
        snapshot = configuration(extra={"assessment": {**DEFAULTS, "runtime_window": 9}}).document
        snapshot["launch"] = {"method": "sft"}
        checkpoint = pathlib.Path("runs/example/checkpoint-12").absolute()
        values = options.overrides(cli.parse_args(["sft", "--resume-from-checkpoint", str(checkpoint),
                                                   "--assessment-window", "7"]))
        with patch.object(config, "_read_toml", return_value=snapshot) as read:
            cfg = config.load("unused-current.toml", "sft", overrides=values)
        read.assert_called_once_with(checkpoint.parent / "config.toml")
        self.assertEqual(cfg.assessment.runtime_window, 7)
        self.assertEqual(cfg.assessment.quality_max_length, 2048)

    # Help may inspect library metadata but cannot read configuration or load datasets.
    def test_help_no_data_or_configuration_load(self):
        output = io.StringIO()
        with patch.object(data_load, "load", side_effect=AssertionError("help loaded data")), \
             patch.object(config, "_read_toml", side_effect=AssertionError("help read config")), \
             contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit:
            cli.parse_args(["sft", "--help"])
        self.assertEqual(exit.exception.code, 0)
        self.assertIn("--quality-checks", output.getvalue())
        self.assertIn("--no-quality-checks", output.getvalue())

    # Generated settings contain every approved value explicitly; no judge connection is invented.
    def test_init_explicit_defaults(self):
        document = tomllib.loads(init_cmd.render(Hardware(4, ())))
        self.assertEqual(document["assessment"], DEFAULTS)
        self.assertEqual(init_cmd.ASSESSMENT_DEFAULTS, DEFAULTS)


if __name__ == "__main__":
    unittest.main()
