"""Tests for the reasoning-conversation bridge against the real encoder and
tokenizer: label boundary at <think>, next-token shift, EOS supervision, pad
entry, split selection, and length and format rejection."""

import json
import os
import tempfile
import unittest

from deepseek.qat import data as D

SNAP = "deepseek/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
ROWS = [
    {"messages": [{"role": "user", "content": "Say hi."}, {"role": "assistant", "content": "Hi there."}],
     "reasoning": "The user wants a greeting."},
    {"messages": [{"role": "user", "content": "Count to three."}, {"role": "assistant", "content": "1, 2, 3."}],
     "reasoning": "Simple enumeration."},
    {"messages": [{"role": "user", "content": "Name a color."}, {"role": "assistant", "content": "Blue."}],
     "reasoning": "Any color works."},
]


def _write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@unittest.skipUnless(os.path.isdir(SNAP), "snapshot not present")
class Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained(SNAP)
        cls.enc = D.load_encoder(SNAP)
        cls.tmp = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        cls.path = os.path.join(cls.tmp.name, "rows.jsonl")
        _write(ROWS, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels_supervise_reasoning_answer_and_eos_only(self):
        ds = D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="train", holdout=1)
        ex = ds[0]
        ids, n_prompt = D.encode_row(ROWS[0], self.enc, self.tok)
        self.assertEqual(ex["input_ids"], ids[:-1])
        self.assertEqual(len(ex["labels"]), len(ex["input_ids"]))
        # labels[t] is the next token of input_ids[t]; prompt positions masked.
        for t, (x, y) in enumerate(zip(ex["input_ids"], ex["labels"])):
            if t + 1 < n_prompt:
                self.assertEqual(y, D.IGNORE)
            else:
                self.assertEqual(y, ids[t + 1])
        self.assertEqual(ex["labels"][-1], self.tok.eos_token_id)
        self.assertNotEqual(ex["input_ids"][-1], self.tok.eos_token_id)
        think = self.tok.convert_tokens_to_ids("<think>")
        self.assertEqual(ids[n_prompt - 1], think)
        self.assertEqual(ex["___PAD_TOKEN_IDS___"], {"input_ids": self.tok.pad_token_id, "labels": -100, "attention_mask": 0})
        self.assertEqual(ex["attention_mask"], [1] * len(ex["input_ids"]))

    def test_reasoning_text_is_in_the_supervised_span(self):
        ds = D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="train", holdout=1)
        ex = ds[1]
        supervised = [y for y in ex["labels"] if y != D.IGNORE]
        self.assertIn("Simple enumeration.", self.tok.decode(supervised))
        self.assertIn("1, 2, 3.", self.tok.decode(supervised))

    def test_split_selection(self):
        train = D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="train", holdout=1)
        hold = D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="holdout", holdout=1)
        self.assertEqual((len(train), len(hold)), (2, 1))
        last_ids, _ = D.encode_row(ROWS[2], self.enc, self.tok)
        self.assertEqual(hold[0]["input_ids"], last_ids[:-1])
        with self.assertRaises(ValueError):
            D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="train", holdout=3)
        # "all" takes every row and ignores holdout, even a value train/holdout would reject.
        every = D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="all", holdout=3)
        self.assertEqual(len(every), 3)
        self.assertEqual(every[2]["input_ids"], last_ids[:-1])
        with self.assertRaisesRegex(ValueError, "split must be"):
            D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="eval", holdout=1)

    def test_getitem_returns_copies(self):
        ds = D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=256, split="train", holdout=1)
        a = ds[0]
        a["___PAD_TOKEN_IDS___"].clear()
        a["input_ids"].append(0)
        b = ds[0]
        self.assertTrue(b["___PAD_TOKEN_IDS___"])
        self.assertNotEqual(len(a["input_ids"]), len(b["input_ids"]))

    def test_long_row_and_bad_format_rejected(self):
        with self.assertRaises(ValueError):
            D.ReasoningChatDataset(self.path, SNAP, self.tok, max_tokens=8, split="train", holdout=1)
        bad = os.path.join(self.tmp.name, "bad.jsonl")
        _write([{"messages": [{"role": "user", "content": "x"}], "reasoning": "r"}], bad)
        with self.assertRaises(ValueError):
            D.read_rows(bad)
        _write([{"messages": ROWS[0]["messages"], "reasoning": ""}], bad)
        with self.assertRaises(ValueError):
            D.read_rows(bad)


if __name__ == "__main__":
    unittest.main()
