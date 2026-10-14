"""Reasoning-conversation dataset for the NeMo recipe.

Rows are ``{"messages": [...], "reasoning": "..."}``: one user turn, one assistant
turn, and the assistant's separate reasoning. Each row is encoded with the
snapshot's own ``encoding/encoding_dsv4.py`` in thinking mode, with the reasoning
placed in the assistant message, so the training text is exactly what the model
was post-trained on:

    <BOS><|User|>{user}<|Assistant|><think>{reasoning}</think>{content}<EOS>

Output follows NeMo's shifted next-token format (``formatting_utils``):
``input_ids`` drops the last token, ``labels`` drops the first and masks prompt
positions with -100, so ``labels[t]`` is the target for ``input_ids[t]``. The
prompt is everything through ``<think>``; reasoning, ``</think>``, answer, and
EOS are supervised. Rows above ``max_tokens`` are errors, never truncated.

The encoder module is loaded from the snapshot directory by path so this file
carries no copy of it.
"""

import importlib.util
import json
import os
import sys

from torch.utils.data import Dataset

IGNORE = -100


# Import the snapshot's encoder module without placing its directory on sys.path
# permanently; cached per snapshot path.
_ENCODERS: dict[str, object] = {}


def load_encoder(snapshot: str):
    snapshot = os.path.abspath(snapshot)
    if snapshot not in _ENCODERS:
        path = os.path.join(snapshot, "encoding", "encoding_dsv4.py")
        spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("encoding_dsv4", module)
        spec.loader.exec_module(module)
        _ENCODERS[snapshot] = module
    return _ENCODERS[snapshot]


def read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON: {e}") from None
            msgs = row.get("messages")
            if not isinstance(msgs, list) or len(msgs) != 2 or [m.get("role") for m in msgs] != ["user", "assistant"]:
                raise ValueError(f"{path}:{n}: expected exactly one user and one assistant message")
            if not isinstance(row.get("reasoning"), str) or not row["reasoning"]:
                raise ValueError(f"{path}:{n}: missing or empty reasoning")
            rows.append(row)
    return rows


def encode_row(row: dict, encoder, tokenizer) -> tuple[list[int], int]:
    """Return (token ids of the full text, number of prompt tokens)."""
    msgs = [dict(m) for m in row["messages"]]
    msgs[-1]["reasoning_content"] = row["reasoning"]
    prompt = encoder.encode_messages(msgs[:-1], thinking_mode="thinking")
    full = encoder.encode_messages(msgs, thinking_mode="thinking")
    if not full.startswith(prompt):
        raise ValueError("encoded conversation does not start with its encoded prompt")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens are not a prefix of the full tokenization")
    return ids, len(prompt_ids)


class ReasoningChatDataset(Dataset):
    """``split`` is ``train`` (all rows but the last ``holdout``) or ``holdout``."""

    def __init__(self, path: str, snapshot: str, tokenizer, max_tokens: int, split: str = "train", holdout: int = 16):
        rows = read_rows(path)
        if holdout <= 0 or holdout >= len(rows):
            raise ValueError(f"holdout must be in 1..{len(rows) - 1}, got {holdout}")
        if split == "train":
            rows = rows[:-holdout]
        elif split == "holdout":
            rows = rows[-holdout:]
        else:
            raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")
        encoder = load_encoder(snapshot)
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.examples = []
        for i, row in enumerate(rows):
            ids, n_prompt = encode_row(row, encoder, tokenizer)
            if len(ids) > max_tokens:
                raise ValueError(f"{path} {split} row {i}: {len(ids)} tokens exceeds max_tokens {max_tokens}")
            if ids[-1] != self.eos_id:
                raise ValueError(f"{path} {split} row {i}: encoded text does not end with EOS")
            labels = [IGNORE] * n_prompt + ids[n_prompt:]
            self.examples.append({
                "input_ids": ids[:-1],
                "labels": labels[1:],
                "attention_mask": [1] * (len(ids) - 1),
                "___PAD_TOKEN_IDS___": {"input_ids": self.pad_id, "labels": IGNORE, "attention_mask": 0},
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        # The collater pops the pad entry from the first example in place.
        return {k: (dict(v) if isinstance(v, dict) else list(v)) for k, v in ex.items()}


def build_dataset(path: str, snapshot: str, max_tokens: int, split: str = "train", holdout: int = 16):
    """Recipe entry point: builds the tokenizer from the snapshot."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    return ReasoningChatDataset(path, snapshot, tokenizer, max_tokens, split=split, holdout=holdout)
