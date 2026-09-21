"""Approved scratch validation of installed preference-trainer preparation.

All tokenizers and datasets are synthetic and in memory. No trainer/model is
constructed: these are actual installed preparation methods and collators with
the small context they require, not substitutes for their implementations.
"""

import json
import string
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, disable_progress_bars
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from trl.trainer import dpo_trainer, kto_trainer, reward_trainer


# One explicit BPE merge exposes prompt/completion boundary changes without files.
def make_tokenizer():
    special = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>", "<end>"]
    tokens = special + list(string.ascii_letters + string.digits + " .,!?\n") + ["ab"]
    backend = Tokenizer(BPE({token: i for i, token in enumerate(tokens)}, [("a", "b")], unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>", "<end>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] + '<end>' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    return tokenizer


# Match only the preparation attributes; no hardware allocation or model loading occurs.
def prepare(module, tokenizer, rows, max_length=None, mode="keep_start", calculate_kl=False):
    context = SimpleNamespace(
        _tokenizer=tokenizer,
        calculate_KL=calculate_kl,
        desirable_weight=1.0,
        undesirable_weight=1.0,
    )
    args = SimpleNamespace(
        dataset_num_proc=None,
        max_length=max_length,
        truncation_mode=mode,
        per_device_train_batch_size=2,
    )
    trainer = {
        dpo_trainer: dpo_trainer.DPOTrainer,
        kto_trainer: kto_trainer.KTOTrainer,
        reward_trainer: reward_trainer.RewardTrainer,
    }[module]
    result = trainer._prepare_dataset(context, Dataset.from_list(rows), tokenizer, args, "train")
    assert not result.cache_files, "Synthetic in-memory preparation unexpectedly wrote dataset cache files"
    return result


# Check actual truncation and causal shift, including a fully removed prompt prefix.
def check_dpo(tokenizer):
    rows = [
        {"prompt": "pq", "chosen": "rstuvw", "rejected": "xy"},
        {"prompt": "pqrst", "chosen": "u", "rejected": "v"},
    ]
    start = prepare(dpo_trainer, tokenizer, rows, 4)
    end = prepare(dpo_trainer, tokenizer, rows, 4, "keep_end")
    assert len(start) == 1 and len(end) == 2
    start_batch = dpo_trainer.DataCollatorForPreference(tokenizer.pad_token_id, 4, "keep_start")([start[0]])
    end_batch = dpo_trainer.DataCollatorForPreference(tokenizer.pad_token_id, 4, "keep_end")([end[0]])
    assert start_batch["completion_mask"].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert end_batch["completion_mask"].tolist() == [[1, 1, 1, 1], [0, 1, 1, 1]]
    assert end_batch["completion_mask"][:, 1:].sum(dim=1).tolist() == [3, 3]
    assert start[0]["chosen_ids"] == tokenizer("rstuvw<eos>")["input_ids"]
    assert start_batch["input_ids"][0].tolist() == tokenizer("pqrs")["input_ids"]
    assert end_batch["input_ids"][0].tolist() == tokenizer("uvw<eos>")["input_ids"]
    return {"keep_start_rows": len(start), "keep_end_rows": len(end), "keep_end_shifted_completion_tokens": [3, 3]}


# KTO rotates completions within physical batches, including a singleton final batch.
def check_kto(tokenizer):
    rows = [
        {"prompt": "pq", "completion": "rstuvw", "label": True},
        {"prompt": "pq", "completion": "xy", "label": False},
        {"prompt": "pq", "completion": "z", "label": True},
        {"prompt": "pqrst", "completion": "u", "label": False},
    ]
    data = prepare(kto_trainer, tokenizer, rows, 4, calculate_kl=True)
    assert len(data) == 3 and list(data["label"]) == [True, False, True]
    assert data[0]["KL_completion_ids"] == data[1]["completion_ids"]
    assert data[1]["KL_completion_ids"] == data[0]["completion_ids"]
    assert data[2]["KL_completion_ids"] == data[2]["completion_ids"]
    batch = kto_trainer.DataCollatorForUnpairedPreference(tokenizer.pad_token_id, 4)(list(data))
    assert batch["completion_mask"].tolist() == [[0, 0, 1, 1]] * 3
    assert batch["KL_completion_mask"].tolist() == [[0, 0, 1, 1]] * 3
    assert batch["label"].tolist() == [True, False, True]
    paired = prepare(kto_trainer, tokenizer, [{"prompt": "pq", "chosen": "r", "rejected": "s"}], 4)
    assert list(paired["label"]) == [True, False]
    return {"rows_after_filter": 3, "desirable": 2, "undesirable": 1, "singleton_kl_self_pair": True, "paired_rows_expand_to": 2}


# Reward preparation drops complete preference pairs instead of truncating their tails.
def check_reward(tokenizer):
    rows = [
        {"prompt": "pq", "chosen": "r", "rejected": "s", "margin": 0.5},
        {"prompt": "pq", "chosen": "rstuvw", "rejected": "x", "margin": 0.0},
        {"prompt": "pq", "chosen": "r<eos>", "rejected": "s<eos>", "margin": 0.2},
    ]
    data = prepare(reward_trainer, tokenizer, rows, 5)
    assert len(data) == 2
    assert data[0]["chosen_ids"] == data[1]["chosen_ids"] == tokenizer("pqr<eos>")["input_ids"]
    batch = reward_trainer.DataCollatorForPreference(tokenizer.pad_token_id)(list(data))
    assert batch["attention_mask"].sum(dim=1).tolist() == [4, 4, 4, 4]
    assert batch["margin"].numel() == 2
    pretokenized = prepare(reward_trainer, tokenizer, [{"chosen_ids": [10, 11], "rejected_ids": [12]}], 5)
    assert pretokenized[0]["chosen_ids"] == [10, 11]
    return {"input_pairs": 3, "retained_pairs": 2, "existing_eos_not_duplicated": True, "pretokenized_path": True}


# Boundary merges are warning-only upstream: sliced completions can lose semantic tokens.
def check_boundary_merge(tokenizer):
    prompt_ids = tokenizer("a")["input_ids"]
    full_ids = tokenizer("ab<eos>")["input_ids"]
    assert prompt_ids != full_ids[:len(prompt_ids)]
    with patch.object(dpo_trainer.logger, "warning") as warnings:
        dpo = prepare(dpo_trainer, tokenizer, [{"prompt": "a", "chosen": "b", "rejected": "c"}])[0]
        assert warnings.call_count == 1
    with patch.object(kto_trainer.logger, "warning") as warnings:
        kto = prepare(kto_trainer, tokenizer, [{"prompt": "a", "completion": "b", "label": True}])[0]
        assert any("Mismatch" in str(call) for call in warnings.call_args_list)
    assert dpo["chosen_ids"] == kto["completion_ids"] == [tokenizer.eos_token_id]
    assert dpo["prompt_ids"] + dpo["chosen_ids"] != full_ids
    return {"prefix_mismatch_detected": True, "upstream_continues_with_sliced_completion": True}


# Chat templates insert assistant prefixes; compare against complete rendered conversations.
def check_chat(tokenizer):
    prompt = [{"role": "user", "content": "pq"}]
    chosen = [{"role": "assistant", "content": "rs"}]
    rejected = [{"role": "assistant", "content": "tu"}]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
    full_ids = tokenizer.apply_chat_template(prompt + chosen, tokenize=True, return_dict=True)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids
    dpo = prepare(dpo_trainer, tokenizer, [{"prompt": prompt, "chosen": chosen, "rejected": rejected}])[0]
    kto = prepare(kto_trainer, tokenizer, [{"prompt": prompt, "completion": chosen, "label": True}])[0]
    reward = prepare(reward_trainer, tokenizer, [{"prompt": prompt, "chosen": chosen, "rejected": rejected}])[0]
    assert dpo["prompt_ids"] == kto["prompt_ids"] == prompt_ids
    assert dpo["chosen_ids"] == kto["completion_ids"] == full_ids[len(prompt_ids):]
    assert reward["chosen_ids"] == full_ids
    implicit = prepare(dpo_trainer, tokenizer, [{"chosen": prompt + chosen, "rejected": prompt + rejected}])[0]
    assert implicit["prompt_ids"] == prompt_ids and implicit["chosen_ids"] == dpo["chosen_ids"]
    return {"prompt_tokens": len(prompt_ids), "completion_tokens": len(dpo["chosen_ids"]), "implicit_prompt_extraction": True}


# Emit machine-readable evidence only after every actual-library assertion succeeds.
def main():
    disable_progress_bars()
    tokenizer = make_tokenizer()
    evidence = {
        "dpo": check_dpo(tokenizer),
        "kto": check_kto(tokenizer),
        "reward": check_reward(tokenizer),
        "boundary_merge": check_boundary_merge(tokenizer),
        "chat": check_chat(tokenizer),
    }
    print(json.dumps({"status": "passed", "scope": "synthetic CPU preparation and collators; no model forwards", "evidence": evidence}, indent=2))


if __name__ == "__main__":
    main()
