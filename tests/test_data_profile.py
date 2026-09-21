"""Full-scan evidence and projections against actual installed TRL preparation."""

import copy
import json
import string
import unittest
from types import SimpleNamespace

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from trl import SFTTrainer
from trl.trainer import dpo_trainer, kto_trainer, reward_trainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from trlx.data_profile import compare_sources, fingerprint, response_pairs, scan


# A single BPE merge exposes prompt/response prefix instability without models.
def tokenizer():
    tokens = ["<pad>", "<unk>", "<eos>", "<user>", "<assistant>", "<end>"]
    tokens += list(string.ascii_letters + string.digits + " .,!?\n") + ["ab"]
    backend = Tokenizer(BPE({token: i for i, token in enumerate(tokens)}, [("a", "b")], unk_token="<unk>"))
    result = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
                                    eos_token="<eos>", additional_special_tokens=["<user>", "<assistant>", "<end>"])
    result.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '>' }}"
        "{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<end>' }}{% endgeneration %}"
        "{% else %}{{ message['content'] + '<end>' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    return result


# Explicit synthetic inputs avoid TRL's hardware-dependent config initialization.
def config(method="sft", **overrides):
    settings = dict(eos_token=None, chat_template_path=None, assistant_only_loss=False, completion_only_loss=None,
                    dataset_text_field="text", dataset_kwargs=None, max_length=8, truncation_mode="keep_start",
                    packing=False, eval_packing=None, packing_strategy="bfd", shuffle_dataset=False, seed=1,
                    padding_free=False, dataset_num_proc=None, use_liger_kernel=False,
                    per_device_train_batch_size=2, chat_template_kwargs=None)
    settings.update(overrides)
    return SimpleNamespace(method=SimpleNamespace(name=method), args=SimpleNamespace(**settings))


class DataProfileTests(unittest.TestCase):
    # Compare complete preparation and collator labels, not another count formula.
    def test_sft_preparation_and_collator_parity(self):
        chat = [{"role": "user", "content": "pq"}, {"role": "assistant", "content": "rst"}]
        text = [{"text": "pqrstuvwxyz"}, {"text": "pqr"}, {"text": "st"}]
        completion = [{"prompt": "pqrstuvwxyz", "completion": "pqr"}, {"prompt": "pq", "completion": "rstuvwxyz"}]
        cases = [(text, {"truncation_mode": mode}) for mode in ("keep_start", "keep_end")]
        cases += [(completion, {"completion_only_loss": value}) for value in (None, False, True)]
        cases += [(text, {"packing": True, "packing_strategy": strategy}) for strategy in ("bfd", "bfd_split", "wrapped")]
        cases += [([{"messages": chat}], {"assistant_only_loss": True}),
                  ([{"prompt": chat[:1], "completion": chat[1:]}], {"assistant_only_loss": True}),
                  ([{"input_ids": [4, 5, 6], "completion_mask": [0, 1, 1]}], {"completion_only_loss": True})]
        for rows, overrides in cases:
            with self.subTest(overrides=overrides, rows=rows):
                cfg, processor = config(**overrides), tokenizer()
                dataset = Dataset.from_list(rows)
                profile = scan(cfg, processor, dataset, None)["train"]
                self.assertEqual(profile["errors"], [])
                context = SimpleNamespace(_tokenizer=processor, chat_template=None,
                                          completion_only_loss=cfg.args.completion_only_loss if cfg.args.completion_only_loss is not None
                                          else "prompt" in rows[0])
                actual = SFTTrainer._prepare_dataset(context, dataset, processor, cfg.args, cfg.args.packing, None, "train")
                padding_free = cfg.args.padding_free or cfg.args.packing and cfg.args.packing_strategy in {"bfd", "bfd_split"}
                collator = DataCollatorForLanguageModeling(processor.pad_token_id, padding_free=padding_free)
                batch = collator(list(actual))
                self.assertEqual(profile["effective_rows"], len(actual))
                self.assertEqual(profile["retained_tokens"], sum(len(row["input_ids"]) for row in actual))
                self.assertEqual(profile["loss_tokens"], int((batch["labels"][:, 1:] != -100).sum()))

    # Filtering and collator truncation differ between DPO, KTO, and reward.
    def test_preference_preparation_and_collator_parity(self):
        for method, module in (("dpo", dpo_trainer), ("kto", kto_trainer), ("reward", reward_trainer)):
            for mode in ("keep_start", "keep_end"):
                with self.subTest(method=method, mode=mode):
                    processor = tokenizer()
                    cfg = config(method, max_length=5, truncation_mode=mode)
                    rows = [{"prompt": "pq", "chosen": "rstuvw", "rejected": "x"},
                            {"prompt": "pqrstu", "chosen": "v", "rejected": "w"},
                            {"prompt": "pq", "chosen": "r", "rejected": "s"}]
                    if method == "kto":
                        rows = [{"prompt": row["prompt"], "completion": row["chosen"], "label": i % 2 == 0}
                                for i, row in enumerate(rows)]
                    dataset = Dataset.from_list(rows)
                    profile = scan(cfg, processor, dataset, None)["train"]
                    self.assertEqual(profile["errors"], [])
                    context = SimpleNamespace(_tokenizer=processor, calculate_KL=False, desirable_weight=1, undesirable_weight=1)
                    trainer = {"dpo": dpo_trainer.DPOTrainer, "kto": kto_trainer.KTOTrainer,
                               "reward": reward_trainer.RewardTrainer}[method]
                    actual = trainer._prepare_dataset(context, dataset, processor, cfg.args, "train")
                    if method == "dpo":
                        collator = module.DataCollatorForPreference(processor.pad_token_id, 5, mode)
                    elif method == "kto":
                        collator = module.DataCollatorForUnpairedPreference(processor.pad_token_id, 5)
                    else:
                        collator = module.DataCollatorForPreference(processor.pad_token_id)
                    batch = collator(list(actual))
                    self.assertEqual(profile["effective_rows"], len(actual))
                    self.assertEqual(profile["retained_tokens"], int(batch["attention_mask"].sum()))
                    if method != "reward":
                        self.assertEqual(profile["loss_tokens"], int(batch["completion_mask"][:, 1:].sum()))
                    if method == "kto":
                        self.assertEqual(profile["label_counts"], {"desirable": 2, "undesirable": 0})

    # Boundary merges must match TRL's slicing and retain evidence of the mismatch.
    def test_boundary_merge_is_reported(self):
        for method in ("sft", "dpo", "kto"):
            row = {"prompt": "a", "completion": "b", "label": True} if method != "dpo" else {
                "prompt": "a", "chosen": "b", "rejected": "c"}
            profile = scan(config(method), tokenizer(), [row], None)["train"]
            self.assertEqual(profile["boundary_mismatch_rows"], [0])

    # Off-policy scoring shares trainer slicing even when the completion's first
    # text token merged into the prompt and only EOS remains in the response IDs.
    def test_response_pairs_match_trainer_boundary_slicing(self):
        processor = tokenizer()
        for method, trainer in (("dpo", dpo_trainer.DPOTrainer), ("kto", kto_trainer.KTOTrainer)):
            with self.subTest(method=method):
                row = {"prompt": "a", "chosen": "b", "rejected": "c"} if method == "dpo" else {
                    "prompt": "a", "completion": "b", "label": True}
                context = SimpleNamespace(_tokenizer=processor, calculate_KL=False, desirable_weight=1, undesirable_weight=1)
                prepared = trainer._prepare_dataset(context, Dataset.from_list([row]), processor,
                                                    config(method).args, "train")[0]
                pairs = response_pairs(processor, row, method)
                keys = ("chosen_ids", "rejected_ids") if method == "dpo" else ("completion_ids",)
                self.assertEqual(pairs, [(prepared["prompt_ids"], prepared[key]) for key in keys])
                self.assertEqual(pairs[0][1], [processor.eos_token_id])
                self.assertNotEqual(pairs[0][1], processor("b<eos>")["input_ids"])

    # Explicit labels survive skipped preparation without truncation or packing.
    def test_skip_prepare_and_causal_shift(self):
        rows = [{"input_ids": [1, 2, 3], "labels": [1, -100, -100]}]
        profile = scan(config(max_length=1, packing=True, dataset_kwargs={"skip_prepare_dataset": True}),
                       tokenizer(), rows, None)["train"]
        self.assertEqual(profile["retained_tokens"], 3)
        self.assertEqual(profile["loss_tokens"], 0)
        self.assertEqual(profile["zero_loss_rows"], [0])
        self.assertEqual(profile["truncated_rows"], [])

    # Eval packing is independent, but automatic completion masking is resolved
    # from training shape and reused by the single trainer for both splits.
    def test_evaluation_uses_training_loss_and_own_packing(self):
        report = scan(config(packing=True, eval_packing=False, max_length=12), tokenizer(),
                      [{"text": "pqrst"}], [{"prompt": "pq", "completion": "rst"}])
        self.assertTrue(report["train"]["packing"]["enabled"])
        self.assertFalse(report["eval"]["packing"]["enabled"])
        self.assertEqual(report["eval"]["loss_tokens"], 5)

    # BFD discards overflow even though other packing strategies preserve it.
    def test_packing_discards_are_attributed_to_replay(self):
        report = scan(config(packing=True, max_length=4), tokenizer(),
                      [{"text": "pq"}, {"text": "pqrstuvwxyz"}], None, primary_rows=1)
        profile = report["train"]
        self.assertEqual(profile["truncated_rows"], [1])
        self.assertEqual(profile["discarded_tokens"], 8)
        self.assertEqual(profile["sources"]["replay"]["retained_tokens"], 4)
        self.assertEqual(profile["sources"]["replay"]["loss_tokens"], 3)

    # Wrapped packing gives the next row's first label a predecessor, so that
    # target must not be reported as an ineffective source row.
    def test_wrapped_packing_activates_source_first_token(self):
        rows = [{"input_ids": [10, 11], "labels": [-100, -100]},
                {"input_ids": [12, 13], "labels": [12, -100]}]
        profile = scan(config(packing=True, packing_strategy="wrapped", max_length=4),
                       tokenizer(), rows, None, primary_rows=1)["train"]
        self.assertEqual(profile["zero_loss_rows"], [0])
        self.assertEqual(profile["loss_tokens"], 1)
        self.assertEqual(profile["sources"]["replay"]["loss_tokens"], 1)

    # BFD split resets position IDs on each fragment, masking a target that was
    # trainable in the unsplit source sequence.
    def test_bfd_split_masks_only_target_at_fragment_start(self):
        rows = [{"input_ids": [10, 11, 12, 13, 14], "labels": [-100, -100, -100, -100, 14]}]
        profile = scan(config(packing=True, packing_strategy="bfd_split", max_length=4),
                       tokenizer(), rows, None, primary_rows=0)["train"]
        self.assertEqual(profile["effective_rows"], 2)
        self.assertEqual(profile["retained_tokens"], 5)
        self.assertEqual(profile["zero_loss_rows"], [0])
        self.assertEqual(profile["loss_tokens"], 0)
        self.assertEqual(profile["sources"]["replay"]["loss_tokens"], 0)

    # Distinct source token IDs independently establish attribution from actual
    # trainer/collator output, including shuffled mixed primary/replay blocks.
    def test_shuffled_packing_source_loss_matches_actual_collator(self):
        rows = [{"input_ids": [20 + index] * length,
                 "labels": [20 + index if offset % 2 == 0 else -100 for offset in range(length)]}
                for index, length in enumerate((2, 5, 3, 7))]
        for strategy in ("bfd", "bfd_split", "wrapped"):
            with self.subTest(strategy=strategy):
                cfg = config(packing=True, packing_strategy=strategy, max_length=4, shuffle_dataset=True)
                processor = tokenizer()
                profile = scan(cfg, processor, rows, None, primary_rows=2)["train"]
                context = SimpleNamespace(_tokenizer=processor, chat_template=None, completion_only_loss=False)
                prepared = SFTTrainer._prepare_dataset(context, Dataset.from_list(rows), processor,
                                                       cfg.args, True, None, "train")
                collator = DataCollatorForLanguageModeling(processor.pad_token_id,
                                                          padding_free=strategy in {"bfd", "bfd_split"})
                labels = collator(list(prepared))["labels"][:, 1:].reshape(-1).tolist()
                active = [label for label in labels if label != -100]
                self.assertEqual(profile["loss_tokens"], len(active))
                self.assertEqual(profile["effective_rows"], len(prepared))
                self.assertEqual(profile["zero_loss_rows"], [index for index in range(4) if 20 + index not in active])
                for name, tokens in (("primary", {20, 21}), ("replay", {22, 23})):
                    self.assertEqual(profile["sources"][name]["loss_tokens"], sum(label in tokens for label in active))
                    expected_tokens = sum(token in tokens for row in prepared for token in row["input_ids"])
                    self.assertEqual(profile["sources"][name]["retained_tokens"], expected_tokens)

    # Reward preparation recognizes already encoded pairs and filters whole pairs.
    def test_reward_pretokenized_pairs(self):
        report = scan(config("reward", max_length=2), tokenizer(),
                      [{"chosen_ids": [1, 2], "rejected_ids": [3]},
                       {"chosen_ids": [1, 2, 3], "rejected_ids": [3]}], None)
        self.assertEqual(report["train"]["effective_rows"], 1)
        self.assertEqual(report["train"]["dropped_rows"], [1])
        self.assertIsNone(report["train"]["loss_tokens"])

    # Completion-only masks may remove all targets; invalid input is never counted as zero data.
    def test_mask_errors_are_explicit_unknowns(self):
        profile = scan(config(dataset_kwargs={"skip_prepare_dataset": True}), tokenizer(),
                       [{"input_ids": [1, 2], "completion_mask": [0, 1]}], None)["train"]
        self.assertIsNone(profile["effective_rows"])
        self.assertIn("supply labels", profile["errors"][0]["message"])

    # Exact example overlap and repeated prompts convey different evidence.
    def test_duplicates_conflicts_overlap_and_replay(self):
        rows = [{"prompt": "pq", "chosen": "r", "rejected": "s"},
                {"prompt": "pq", "chosen": "s", "rejected": "r"},
                {"prompt": "pq", "chosen": "r", "rejected": "s"}]
        evaluation = [rows[0], {"prompt": "pq", "chosen": "t", "rejected": "u"}]
        report = scan(config("dpo"), tokenizer(), rows, evaluation, primary_rows=2)
        self.assertEqual(report["train"]["duplicates"], [[0, 2]])
        self.assertEqual(report["train"]["conflicting_labels"], [[0, 1, 2]])
        self.assertEqual(report["overlap"]["identical_examples"], [{"train_rows": [0, 2], "eval_rows": [0]}])
        self.assertEqual(report["overlap"]["prompts"], [{"train_rows": [0, 1, 2], "eval_rows": [0, 1]}])
        self.assertEqual(report["train"]["sources"]["replay"]["rows"], 1)
        self.assertEqual(report["eval"]["rows"], 2)
        json.dumps(report, allow_nan=False)

    # Scan covers all rows, including an outlier beyond common sampling cutoffs.
    def test_full_scan_prompt_trainers(self):
        rows = [{"prompt": "pq"}] * 1001 + [{"prompt": "p" * 25}]
        for method in ("grpo", "rloo", "distillation"):
            report = scan(config(method), tokenizer(), rows, None)
            self.assertEqual(report["train"]["prompt_lengths"]["max"], 25)
            self.assertEqual(report["train"]["rows"], 1002)
            self.assertIsNone(report["train"]["loss_tokens"])
            self.assertEqual(len(report["train"]["duplicates"][0]), 1001)

    # EOS overrides belong to a copy, and resolved template cloning wins its EOS.
    def test_processor_and_inputs_are_not_mutated(self):
        processor = tokenizer()
        rows = [{"text": "pq"}]
        original = copy.deepcopy(rows)
        scan(config(eos_token="<end>"), processor, rows, None)
        self.assertEqual(processor.eos_token, "<eos>")
        self.assertEqual(rows, original)
        report = scan(config(eos_token="missing", chat_template_path="resolved/tokenizer"), processor, rows, None)
        self.assertEqual(report["train"]["errors"], [])

    # Source order and every operator field matter, including a column named replay.
    def test_fingerprint_contract(self):
        rows = [{"prompt": "p", "completion": "q"}, {"prompt": "r", "completion": "s"}]
        marked = [dict(row, replay=True) for row in rows]
        self.assertEqual(fingerprint(rows), fingerprint([dict(reversed(list(row.items()))) for row in rows]))
        self.assertNotEqual(fingerprint(rows), fingerprint(marked))
        self.assertEqual(fingerprint(rows), fingerprint(marked, exclude_columns=("replay",)))
        self.assertNotEqual(fingerprint(rows), fingerprint(rows[::-1]))
        self.assertEqual(fingerprint(rows), scan(config(), tokenizer(), rows, None)["train"]["fingerprint"])
        with self.assertRaisesRegex(ValueError, "cannot fingerprint dataset row 0"):
            fingerprint([{"image": object()}])

    # Internal marker exclusion follows the actual replay KL configuration only.
    def test_scan_distinguishes_internal_and_operator_replay_columns(self):
        rows = [{"text": "pq", "replay": False}, {"text": "pq", "replay": True}]
        cfg = config()
        operator = scan(cfg, tokenizer(), rows, None)["train"]
        self.assertEqual(operator["duplicates"], [])
        self.assertEqual(operator["fingerprint"], fingerprint(rows))
        cfg.replay = SimpleNamespace(kl_coef=1.0)
        internal = scan(cfg, tokenizer(), rows, None)["train"]
        self.assertEqual(internal["duplicates"], [[0, 1]])
        self.assertEqual(internal["fingerprint"], fingerprint(rows, exclude_columns=("replay",)))

    # Quality schema differences still permit exact prompt comparison, without
    # mistaking changed references for identical examples or proving independence.
    def test_compare_quality_sources_without_trainer_preparation(self):
        train = [{"prompt": "pq", "completion": "rs"}, {"prompt": "pq", "completion": "rs"}]
        quality = [{"prompt": "pq", "reference": "tu"}, train[0], {"prompt": "xy", "reference": "z"}]
        result = compare_sources(train, quality, "sft")
        self.assertEqual(result["identical_examples"], [{"train_rows": [0, 1], "eval_rows": [1]}])
        self.assertEqual(result["prompts"], [{"train_rows": [0, 1], "eval_rows": [0, 1]}])
        self.assertEqual(result["identity_errors"], {"train": [], "other": []})
        self.assertIn("no semantic independence", result["scope"])
        marked = [dict(row, replay=True) for row in train]
        excluded = compare_sources(marked, quality, "sft", train_exclude_columns=("replay",))
        self.assertEqual(excluded["identical_examples"], result["identical_examples"])

    # Invalid multimodal identity is explicit while independent structural work continues.
    def test_unsupported_payload_does_not_fabricate_counts(self):
        report = scan(config(), tokenizer(), [{"image": object(), "text": "p"}, {"text": "pq"}], None)
        self.assertIsNone(report["train"]["fingerprint"])
        self.assertIsNone(report["train"]["retained_tokens"])
        self.assertEqual(report["train"]["rows"], 2)
        self.assertEqual(report["train"]["identity_errors"][0]["row"], 0)
