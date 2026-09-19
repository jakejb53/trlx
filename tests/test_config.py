"""Config loader rules from SPEC.md 2.2, and the init round-trip.

Every test writes a TOML file into a temp directory and loads it; no model,
dataset file, or network is touched.
"""

import dataclasses
import pathlib
import tempfile
import unittest

from trlx import TrlxError, config, init_cmd, trainers

# Minimal valid sft config. Tests prepend top-level overrides, so the
# top-level keys come first and blocks after.
BASE = """
output_dir = "runs/test"
[model]
path = "some/model"
dtype = "bfloat16"
[dataset]
split = true
dataset = "data/train.jsonl"
train = 100
[ranges]
loss = [0, 5]
"""


class ConfigCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # Writes `top` (top-level keys) before BASE, or a whole document when
    # `text` is given, and loads it for `method`.
    def load(self, top="", method="sft", text=None):
        path = self.dir / "run.toml"
        path.write_text(text if text is not None else top + BASE)
        return config.load(str(path), method)

    def assertRejected(self, fragment, top="", method="sft", text=None):
        with self.assertRaises(TrlxError) as ctx:
            self.load(top, method, text)
        self.assertIn(fragment, str(ctx.exception))


class TopLevelKeys(ConfigCase):
    def test_unknown_key_rejected(self):
        self.assertRejected("unknown key 'no_such_field'", "no_such_field = 1\n")

    def test_absent_key_takes_dataclass_default(self):
        cfg = self.load()
        expected = {f.name: f for f in dataclasses.fields(trainers.get("sft").config_cls)}
        self.assertEqual(cfg.args.per_device_train_batch_size, expected["per_device_train_batch_size"].default)
        self.assertEqual(cfg.args.learning_rate, expected["learning_rate"].default)

    def test_none_string_accepted_on_optional_field(self):
        self.assertIsNone(self.load('max_length = "None"\n').args.max_length)

    def test_none_string_rejected_on_non_optional_field(self):
        self.assertRejected("does not accept \"None\"", 'learning_rate = "None"\n')

    def test_wrong_scalar_type_rejected(self):
        self.assertRejected("'per_device_train_batch_size' must be int", 'per_device_train_batch_size = "8"\n')

    def test_int_accepted_for_float_field(self):
        self.assertEqual(self.load("learning_rate = 1\n").args.learning_rate, 1)

    def test_model_loading_fields_rejected(self):
        self.assertRejected("[model] block", "trust_remote_code = true\n")
        self.assertRejected("[model] block", "model_init_kwargs = {dtype = \"auto\"}\n")

    def test_output_dir_required(self):
        self.assertRejected("'output_dir' is required", text=BASE.replace('output_dir = "runs/test"\n', ""))

    def test_run_name_defaults_to_directory_name(self):
        self.assertEqual(self.load().args.run_name, "test")
        self.assertEqual(self.load('run_name = "mine"\n').args.run_name, "mine")

    def test_save_steps_defaults_to_eval_steps(self):
        cfg = self.load('eval_strategy = "steps"\neval_steps = 25\n')
        self.assertEqual(cfg.args.save_steps, 25)
        cfg = self.load('eval_strategy = "steps"\neval_steps = 25\nsave_steps = 50\n')
        self.assertEqual(cfg.args.save_steps, 50)
        # transformers resolves an absent eval_steps to logging_steps; the
        # checkpoint interval must follow that resolved value.
        cfg = self.load('eval_strategy = "steps"\nlogging_steps = 7\n')
        self.assertEqual(cfg.args.save_steps, 7)
        # Eval off: the dataclass default stands.
        self.assertEqual(self.load().args.save_steps, 500)

    def test_dataclass_validation_is_reported(self):
        # An invalid enum value is rejected by TrainingArguments.__post_init__
        # and must surface as TrlxError, not a traceback.
        self.assertRejected("rejected the config", 'lr_scheduler_type = "bogus"\n')


class DatasetBlock(ConfigCase):
    def test_split_true_rejects_split_false_keys(self):
        text = BASE.replace("train = 100\n", 'train = 100\ndataset_train = "a.jsonl"\n')
        self.assertRejected("'dataset_train' is for split = false", text=text)

    def test_split_false_rejects_split_true_keys(self):
        text = BASE.replace("split = true\ndataset = \"data/train.jsonl\"\ntrain = 100\n",
                            'split = false\ndataset_train = "a.jsonl"\ntrain = 100\n')
        self.assertRejected("'train' is for split = true", text=text)

    def test_split_false_without_eval_rejects_eval_keys(self):
        text = BASE.replace("split = true\ndataset = \"data/train.jsonl\"\ntrain = 100\n",
                            'split = false\ndataset_train = "a.jsonl"\n')
        self.assertRejected("eval keys are set: eval_steps", text='eval_steps = 10\n' + text)
        cfg = self.load(text=text)
        self.assertFalse(cfg.dataset.eval_enabled)

    def test_split_false_with_eval(self):
        text = BASE.replace("split = true\ndataset = \"data/train.jsonl\"\ntrain = 100\n",
                            'split = false\ndataset_train = "a.jsonl"\ndataset_eval = "b.parquet"\n')
        cfg = self.load(text=text)
        self.assertTrue(cfg.dataset.eval_enabled)
        self.assertEqual(cfg.dataset.eval_source.source, "b.parquet")

    def test_hf_id_with_and_without_split(self):
        ref = config.dataset_ref("run.toml", "[dataset].dataset", "org/name:train")
        self.assertEqual((ref.is_file, ref.source, ref.split), (False, "org/name", "train"))
        ref = config.dataset_ref("run.toml", "[dataset].dataset", "org/name")
        self.assertEqual((ref.is_file, ref.source, ref.split), (False, "org/name", None))
        ref = config.dataset_ref("run.toml", "[dataset].dataset", "dir/rows.parquet")
        self.assertEqual((ref.is_file, ref.source, ref.split), (True, "dir/rows.parquet", None))

    def test_unrecognised_dataset_value(self):
        with self.assertRaises(TrlxError) as ctx:
            config.dataset_ref("run.toml", "[dataset].dataset", "rows.txt")
        self.assertIn("neither a dataset file", str(ctx.exception))

    def test_bool_is_not_a_row_count(self):
        self.assertRejected("[dataset].train must be int", text=BASE.replace("train = 100", "train = true"))


class Blocks(ConfigCase):
    def test_ranges_required(self):
        self.assertRejected("missing required block [ranges]", text=BASE.replace("[ranges]\nloss = [0, 5]\n", ""))

    def test_ranges_shape(self):
        self.assertRejected("must be [low, high]", text=BASE.replace("loss = [0, 5]", "loss = 5"))
        self.assertRejected("needs low < high", text=BASE.replace("loss = [0, 5]", "loss = [5, 0]"))

    def test_block_restricted_to_methods(self):
        self.assertRejected("block [replay] applies to sft, not dpo", method="dpo",
                            text=BASE + '[replay]\ndataset = "r.jsonl"\nfraction = 0.1\nkl_coef = 0\n')

    def test_teacher_required_for_distillation(self):
        self.assertRejected("requires a [teacher] block", method="distillation")

    def test_model_block_keys_and_dtype(self):
        self.assertRejected("[model] has unknown keys: revision", text=BASE.replace('dtype = "bfloat16"', 'dtype = "bfloat16"\nrevision = "x"'))
        self.assertRejected("is not a torch dtype name", text=BASE.replace('dtype = "bfloat16"', 'dtype = "bf16"'))

    def test_peft_block(self):
        cfg = self.load(text=BASE + '[peft]\nr = 4\ntarget_modules = ["a"]\n')
        self.assertEqual(cfg.peft.r, 4)
        self.assertEqual(cfg.peft.task_type, "CAUSAL_LM")
        self.assertRejected("[peft] keys task_type are set by trlx", text=BASE + '[peft]\ntask_type = "SEQ_CLS"\n')
        self.assertRejected("[peft]", text=BASE + '[peft]\nno_such = 1\n')

    def test_peft_absent_means_full_fine_tune(self):
        self.assertIsNone(self.load().peft)

    def test_reward_method_sets_seq_cls(self):
        cfg = self.load(method="reward", text=BASE + '[peft]\nr = 4\n')
        self.assertEqual(cfg.peft.task_type, "SEQ_CLS")

    def test_rewards_entries(self):
        text = BASE + '[rewards]\nfuncs = ["a", {name = "b", args = {k = 1}}]\n'
        cfg = self.load(method="grpo", text=text)
        self.assertEqual([(e.spec, e.args) for e in cfg.rewards], [("a", None), ("b", {"k": 1})])
        self.assertRejected("[rewards].funcs[0] must be a string", method="grpo", text=BASE + "[rewards]\nfuncs = [1]\n")

    # [replay] with kl_coef > 0 forces loss_type = "nll" and rejects the key;
    # kl_coef = 0 leaves loss_type to the operator. fraction is open at 1.
    def test_replay_kl_forces_loss_type(self):
        replay = '[replay]\ndataset = "replay.jsonl"\nfraction = 0.2\nkl_coef = {kl}\n'
        cfg = self.load(text=BASE + replay.format(kl=0.1))
        self.assertEqual(cfg.args.loss_type, "nll")
        self.assertRejected("[replay].kl_coef > 0", text='loss_type = "nll"\n' + BASE + replay.format(kl=0.1))
        cfg = self.load(text='loss_type = "chunked_nll"\n' + BASE + replay.format(kl=0))
        self.assertEqual(cfg.args.loss_type, "chunked_nll")
        self.assertRejected("[replay].fraction must be in (0, 1)", text=BASE + '[replay]\ndataset = "r.jsonl"\nfraction = 1\nkl_coef = 0\n')


class InitRoundTrip(ConfigCase):
    # Every template loads back; placeholders are valid at load time.
    def test_every_method(self):
        for name in trainers.METHODS:
            path = self.dir / f"{name}.toml"
            path.write_text(init_cmd.render(name))
            cfg = config.load(str(path), name)
            self.assertEqual(cfg.method.name, name)
            self.assertEqual(list(cfg.ranges), list(init_cmd.RANGES[name]))

    def test_init_refuses_to_overwrite(self):
        path = self.dir / "x.toml"
        path.write_text("")
        with self.assertRaises(TrlxError):
            init_cmd.write("sft", str(path))


if __name__ == "__main__":
    unittest.main()
