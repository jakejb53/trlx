"""Replay mixing and the replay trainer (SPEC 2.9).

MixReplayTest touches no model. ReplayTrainerTest builds a ReplayTrainer on
Qwen/Qwen3-0.6B with LoRA, the same small model every verification run
uses, and drives compute_loss by hand on batches from the trainer's own
dataloader, so the checks see exactly what training sees: the flag column
surviving to the loss, no KL on batches without replay rows, zero KL before
any update (LoRA B starts at zero, so adapters on and off agree), and a
positive KL after one optimizer step.
"""

import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Outside a distributed run, transformers' Trainer multiplies the per-device
# batch size by the visible device count (nn.DataParallel), which changes both
# how many batches the dataloader yields and which rows share one. A trlx
# worker always sees exactly one device (train.py sets CUDA_VISIBLE_DEVICES per
# rank), so the batch expectations below hold only under that condition. Set
# before torch initialises CUDA; keeps an operator's own selection, first entry.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

import datasets
import torch

from trlx import TrlxError, config, data_load
from trlx.data_load import REPLAY_COLUMN

MODEL = "Qwen/Qwen3-0.6B"


def _row(question, answer):
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


TRAIN = [_row(f"What is {i} plus {i}?", f"{i} plus {i} is {2 * i}.") for i in range(8)]
REPLAY = [_row(f"Name a colour number {i}.", f"Colour {i} is blue.") for i in range(6)]


class MixReplayTest(unittest.TestCase):
    # The KL row marker must not silently replace a column supplied by either input.
    def test_reserved_replay_column_is_contextual_error(self):
        train = datasets.Dataset.from_list([{"text": "train", "replay": "operator value"}])
        replay = datasets.Dataset.from_list([{"text": "replay", "replay": "operator value"}])
        spec = SimpleNamespace(dataset=SimpleNamespace(source="replay.jsonl"), fraction=0.5)
        with patch("trlx.data_load.load_ref", return_value=replay):
            with self.assertRaisesRegex(TrlxError, "replay.*reserved.*rename"):
                data_load.mix_replay(train, spec, flag=True)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def spec(self, rows, fraction, name="replay.jsonl"):
        path = self.dir / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return config.ReplaySpec(config.dataset_ref("t", "[replay].dataset", str(path)), fraction, 0.0)

    def test_count_and_flag(self):
        train = datasets.Dataset.from_list(TRAIN)
        # 8 train rows at fraction 0.2: R / (8 + R) = 0.2 gives R = 2.
        mixed = data_load.mix_replay(train, self.spec(REPLAY, 0.2), True)
        self.assertEqual(mixed.num_rows, 10)
        self.assertEqual(mixed[REPLAY_COLUMN], [False] * 8 + [True] * 2)
        self.assertEqual(mixed[8]["messages"], REPLAY[0]["messages"])
        plain = data_load.mix_replay(train, self.spec(REPLAY, 0.2), False)
        self.assertEqual(plain.num_rows, 10)
        self.assertNotIn(REPLAY_COLUMN, plain.column_names)

    def test_too_few_replay_rows(self):
        train = datasets.Dataset.from_list(TRAIN)
        with self.assertRaisesRegex(TrlxError, "needs 8 replay rows"):
            data_load.mix_replay(train, self.spec(REPLAY, 0.5), True)

    def test_column_mismatch(self):
        train = datasets.Dataset.from_list(TRAIN)
        rows = [{"prompt": "a", "completion": "b"}]
        with self.assertRaisesRegex(TrlxError, "differ from the train set"):
            data_load.mix_replay(train, self.spec(rows, 0.1), True)


class ReplayTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peft import LoraConfig
        from trl import SFTConfig

        from trlx import model as model_mod
        from trlx.replay_trainer import KL_METRIC, ReplayTrainer

        cls.KL_METRIC = KL_METRIC
        cls._tmp = tempfile.TemporaryDirectory()
        cuda = torch.cuda.is_available()
        spec = config.ModelSpec(path=MODEL, dtype="bfloat16" if cuda else "float32", trust_remote_code=None, attn_implementation=None)
        model = model_mod.load_model(spec, model_mod.CAUSAL)
        processor = model_mod.load_processor(spec)
        # Four train rows and two replay rows, batch 2 and no shuffling
        # concern: batches are taken from the dataloader in whatever order
        # and classified by their flag tensor.
        train = datasets.Dataset.from_list(TRAIN[:4] + REPLAY[:2]).add_column(REPLAY_COLUMN, [False] * 4 + [True] * 2)
        args = SFTConfig(
            output_dir=cls._tmp.name, per_device_train_batch_size=2, max_steps=1, logging_steps=1,
            report_to="none", loss_type="nll", bf16=cuda, use_cpu=not cuda, learning_rate=1e-3,
            disable_tqdm=True, max_length=64,
        )
        cls.trainer = ReplayTrainer(
            model=model, args=args, train_dataset=train, processing_class=processor,
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", task_type="CAUSAL_LM"),
            kl_coef=1.0,
        )
        cls.trainer.model.train()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # Batches from the trainer's dataloader, moved to the device.
    def batches(self):
        return [self.trainer._prepare_inputs(b) for b in self.trainer.get_train_dataloader()]

    def test_flag_reaches_every_batch_and_kl_behaviour(self):
        trainer = self.trainer
        batches = self.batches()
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertIn(REPLAY_COLUMN, batch)
            self.assertEqual(batch[REPLAY_COLUMN].dtype, torch.bool)
        # The dataloader shuffles, so only the totals are fixed: two replay
        # rows over three batches always leave a batch without one.
        self.assertEqual(sum(b[REPLAY_COLUMN].sum().item() for b in batches), 2)
        plain = [b for b in batches if not b[REPLAY_COLUMN].any()][0]
        replay = [b for b in batches if b[REPLAY_COLUMN].any()][0]

        # No replay row: the loss is the parent's, and no KL is recorded.
        from trl import SFTTrainer

        with torch.no_grad():
            ours = trainer.compute_loss(trainer.model, dict(plain))
            parent = SFTTrainer.compute_loss(trainer, trainer.model, {k: v for k, v in plain.items() if k != REPLAY_COLUMN})
        self.assertAlmostEqual(ours.item(), parent.item(), places=4)
        self.assertNotIn(self.KL_METRIC, trainer._metrics["train"])

        # Replay rows before any update: adapters at zero agree with adapters
        # off, so the KL is zero and the loss is the plain loss.
        with torch.no_grad():
            loss0 = trainer.compute_loss(trainer.model, dict(replay))
            parent0 = SFTTrainer.compute_loss(trainer, trainer.model, {k: v for k, v in replay.items() if k != REPLAY_COLUMN})
        self.assertAlmostEqual(trainer._metrics["train"][self.KL_METRIC][-1], 0.0, places=6)
        self.assertAlmostEqual(loss0.item(), parent0.item(), places=4)

        # One optimizer step on the replay batch moves the adapters; the KL
        # against the original is then positive and enters the loss.
        trainer.create_optimizer()
        loss = trainer.compute_loss(trainer.model, dict(replay))
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        with torch.no_grad():
            loss1 = trainer.compute_loss(trainer.model, dict(replay))
            parent1 = SFTTrainer.compute_loss(trainer, trainer.model, {k: v for k, v in replay.items() if k != REPLAY_COLUMN})
        kl = trainer._metrics["train"][self.KL_METRIC][-1]
        self.assertGreater(kl, 0.0)
        self.assertGreater(loss1.item(), parent1.item())

        # With the trainer's accumulation-window token count, the KL term is
        # normalised exactly as the NLL is: summed over replay tokens and
        # divided by num_items_in_batch (one process, so no rank scaling).
        # A count far above this batch's own stands in for a longer window.
        replay_tokens = ((replay["labels"][:, 1:] != -100) & replay[REPLAY_COLUMN][:, None]).sum().item()
        window = torch.tensor(1000, device=replay["labels"].device)
        with torch.no_grad():
            loss2 = trainer.compute_loss(trainer.model, dict(replay), num_items_in_batch=window)
            parent2 = SFTTrainer.compute_loss(
                trainer, trainer.model, {k: v for k, v in replay.items() if k != REPLAY_COLUMN}, num_items_in_batch=window
            )
        kl2 = trainer._metrics["train"][self.KL_METRIC][-1]
        expected = trainer.kl_coef * kl2 * replay_tokens / window.item()
        self.assertAlmostEqual((loss2 - parent2).item(), expected, delta=1e-2 * expected)


if __name__ == "__main__":
    unittest.main()
