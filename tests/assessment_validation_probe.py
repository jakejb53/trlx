"""Offline synthetic preparation and distributed assessment-state validation."""

import argparse
import contextlib
import copy
import datetime
import json
import os
import pathlib
import random
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

# Standalone spawned workers must import this checkout, with all scratch confined to tests/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["TMPDIR"] = str(pathlib.Path(__file__).resolve().parent)

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, disable_progress_bars
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast


class ProbeConfig(PretrainedConfig):
    model_type = "synthetic_assessment_probe"

    # All architecture dimensions are explicit synthetic inputs, not real-model assumptions.
    def __init__(self, vocab_size=32, hidden_size=16, **kwargs):
        super().__init__(pad_token_id=0, eos_token_id=1, bos_token_id=2,
                         tie_word_embeddings=False, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = 1
        self.use_cache = False


class ProbeModel(PreTrainedModel, GenerationMixin):
    config_class = ProbeConfig
    base_model_prefix = "probe"
    _no_split_modules = []

    # A small dropout-bearing model exercises actual gradients and generation APIs.
    def __init__(self, config):
        super().__init__(config)
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.projection = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # Trainer/tokenizer alignment consults the input vocabulary through this public API.
    def get_input_embeddings(self):
        return self.embedding

    # Generation and TRL inspect the output projection without knowing its module name.
    def get_output_embeddings(self):
        return self.projection

    # Ignore cache/position hints: this synthetic model deliberately needs no KV cache.
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.projection(self.dropout(self.embedding(input_ids)))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


# Build a tokenizer entirely in memory, including an explicit training-compatible template.
def tokenizer():
    vocabulary = {word: index for index, word in enumerate(
        ["<pad>", "<eos>", "<bos>", "<unk>", "<u>", "<a>"] + [f"t{i}" for i in range(26)]
    )}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    result = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
                                     bos_token="<bos>", unk_token="<unk>", model_max_length=128)
    result.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'assistant' %}{{ '<a> ' }}"
        "{% generation %}{{ m['content'] + ' <eos> ' }}{% endgeneration %}"
        "{% else %}{{ '<u> ' + m['content'] + ' ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<a> ' }}{% endif %}"
    )
    return result


# Independent projection of the inspected SFT preparation contract for these fixtures.
def project_sft(rows, processor, args):
    from trl import pack_dataset

    projected = []
    for row in rows:
        completion_mask = None
        assistant_mask = None
        if "prompt" in row:
            prompt, completion = row["prompt"], row["completion"]
            if isinstance(prompt, list):
                prompt_ids = processor.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True)["input_ids"]
                encoded = processor.apply_chat_template(
                    prompt + completion, return_dict=True, return_assistant_tokens_mask=args.assistant_only_loss
                )
                ids = encoded["input_ids"]
                assistant_mask = encoded.get("assistant_masks")
            else:
                prompt_ids = processor(prompt)["input_ids"]
                completion = completion if completion.endswith(processor.eos_token) else completion + processor.eos_token
                ids = processor(prompt + completion)["input_ids"]
            if args.completion_only_loss is not False:
                completion_mask = [0] * len(prompt_ids) + [1] * (len(ids) - len(prompt_ids))
        elif "messages" in row:
            encoded = processor.apply_chat_template(
                row["messages"], return_dict=True, return_assistant_tokens_mask=args.assistant_only_loss
            )
            ids, assistant_mask = encoded["input_ids"], encoded.get("assistant_masks")
        else:
            text = row["text"]
            text = text if text.endswith(processor.eos_token) else text + processor.eos_token
            ids = processor(text)["input_ids"]
        labels = [token if (completion_mask is None or completion_mask[index])
                  and (assistant_mask is None or assistant_mask[index]) else -100
                  for index, token in enumerate(ids)]
        if not args.packing and args.max_length is not None:
            window = slice(None, args.max_length) if args.truncation_mode == "keep_start" else slice(-args.max_length, None)
            ids, labels = ids[window], labels[window]
            if all(label == -100 for label in labels):
                continue
        projected.append({"input_ids": ids, "labels": labels})
    dataset = Dataset.from_list(projected)
    if args.packing:
        dataset = pack_dataset(dataset, args.max_length, args.packing_strategy)
    return dataset


# Compare projected sequences/masks with an actual SFTTrainer and its actual collator.
def check_sft():
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.utils import RepeatSampler

    disable_progress_bars()
    chat = [{"role": "user", "content": "t0 t1"}, {"role": "assistant", "content": "t2 t3"}]
    text_rows = [{"text": "t0 t1 t2 t3 t4 t5 t6 t7"}, {"text": "t0 t1"}]
    prompt_rows = [{"prompt": "t0 t1 t2 t3 ", "completion": "t4 t5"},
                   {"prompt": "t0 ", "completion": "t1 t2 t3 t4"}]
    cases = [
        ("text_start", text_rows, {"max_length": 4}),
        ("text_end", text_rows, {"max_length": 4, "truncation_mode": "keep_end"}),
        ("completion_start", prompt_rows, {"max_length": 4}),
        ("completion_end", prompt_rows, {"max_length": 4, "truncation_mode": "keep_end"}),
        ("full_sequence", prompt_rows, {"max_length": 4, "completion_only_loss": False}),
        ("assistant_mask", [{"messages": chat}], {"max_length": 16, "assistant_only_loss": True}),
        ("chat_completion_mask", [{"prompt": chat[:1], "completion": chat[1:]}], {"max_length": 16}),
        ("combined_masks", [{"prompt": chat[:1], "completion": chat[1:]}],
         {"max_length": 16, "assistant_only_loss": True}),
        *[("packing_" + strategy, text_rows, {"max_length": 4, "packing": True, "packing_strategy": strategy})
          for strategy in ("bfd", "bfd_split", "wrapped")],
    ]
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="assessment-sft-", dir=pathlib.Path(__file__).parent) as work:
        for name, rows, overrides in cases:
            processor = tokenizer()
            args = SFTConfig(output_dir=work, use_cpu=True, bf16=False, fp16=False,
                             loss_type="nll", gradient_checkpointing=False, report_to="none",
                             save_strategy="no", shuffle_dataset=False, **overrides)
            predicted = project_sft(rows, processor, args)
            trainer = SFTTrainer(model=ProbeModel(ProbeConfig()), args=args,
                                 train_dataset=Dataset.from_list(rows), processing_class=processor)
            prepared = trainer.train_dataset
            columns = predicted.column_names
            assert predicted.to_list() == prepared.select_columns(columns).to_list(), name
            batch = trainer.data_collator(prepared.to_list())
            labels = batch["labels"]
            trained = int((labels[:, 1:] != -100).sum())
            tokens = sum(len(row["input_ids"]) for row in prepared)
            if name == "packing_bfd":
                assert tokens == 7, tokens
            if name in {"packing_bfd_split", "packing_wrapped"}:
                assert tokens == 12, tokens
            if name == "completion_start":
                assert prepared.num_rows == 1, prepared.num_rows
            outcomes.append({"case": name, "rows": prepared.num_rows, "tokens": tokens, "loss_tokens": trained})
    # Policy/distillation prompt-block dropping can be checked without a generation server.
    sampler_counts = []
    for rows, group in ((3, 4), (5, 4), (8, 4)):
        sampler = RepeatSampler(list(range(rows)), mini_repeat_count=2, batch_size=group,
                               repeat_count=3, shuffle=False, seed=0)
        emitted = list(sampler)
        assert len(emitted) == rows // group * group * 2 * 3
        sampler_counts.append({"rows": rows, "unique_block": group, "emitted": len(emitted),
                               "distinct_used": len(set(emitted))})
    return {"sft_parity": outcomes, "prompt_sampler": sampler_counts}


# Snapshot RNG across all three generators used by model/data/evaluation code.
def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None)


# Restore only the current worker's device; no unrelated GPU state is touched.
def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3])


# Preserve every module's original mode, including intentionally mixed train/eval modes.
@contextlib.contextmanager
def observational_evaluation(model):
    random_state = rng_state()
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(random_state)


# DTensor snapshots compare local shards without triggering unplanned collectives.
def frozen(value):
    if isinstance(value, torch.Tensor):
        value = value.to_local() if hasattr(value, "to_local") else value
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: frozen(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(frozen(item) for item in value)
    return copy.deepcopy(value)


# Exact equality catches even small hidden state changes before checking the next update.
def equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


# Include pending gradients: an observer must not erase accumulation state.
def training_state(model, optimizer, scheduler):
    return frozen({"weights": model.state_dict(), "gradients": [p.grad for p in model.parameters()],
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "modes": [module.training for module in model.modules()], "rng": rng_state()})


# Run a real optimizer update; preserving this outcome is stronger than matching weights alone.
def step(model, optimizer, scheduler, ids):
    optimizer.zero_grad()
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


# Execute identical synthetic training with/without intermediate successful and failed evaluations.
def check_state(rank, world, mode, port, production=False, peft=False):
    distributed = world > 1
    device = torch.device("cpu" if mode == "cpu" else f"cuda:{rank}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=60))
    try:
        torch.manual_seed(17)
        reference = ProbeModel(ProbeConfig()).to(device)
        observed = ProbeModel(ProbeConfig()).to(device)
        if production:
            # Production must override supported structured-output generation settings safely.
            reference.generation_config.return_dict_in_generate = True
            observed.generation_config.return_dict_in_generate = True
        if peft:
            from peft import LoraConfig, get_peft_model

            targets = [name for name, module in reference.named_modules() if isinstance(module, torch.nn.Linear)]
            peft_config = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1, target_modules=targets, task_type="CAUSAL_LM")
            reference = get_peft_model(reference, copy.deepcopy(peft_config))
            observed = get_peft_model(observed, copy.deepcopy(peft_config))
        observed.load_state_dict(reference.state_dict())
        if mode == "ddp":
            reference = torch.nn.parallel.DistributedDataParallel(reference, device_ids=[rank])
            observed = torch.nn.parallel.DistributedDataParallel(observed, device_ids=[rank])
        elif mode == "fsdp":
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard

            mesh = init_device_mesh("cuda", (world,))
            fully_shard(reference, mesh=mesh)
            fully_shard(observed, mesh=mesh)
        reference.train()
        observed.train()
        optimizers = [torch.optim.AdamW(model.parameters(), lr=0.001) for model in (reference, observed)]
        schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9) for optimizer in optimizers]
        ids = torch.tensor([[6, 7, 8, 9, 1], [10, 11, 12, 13, 1]], device=device)
        start = rng_state()
        step(reference, optimizers[0], schedulers[0], ids)
        restore_rng(start)
        step(observed, optimizers[1], schedulers[1], ids)
        assert equal(frozen(reference.state_dict()), frozen(observed.state_dict())), "initial step diverged"

        # Mixed module modes exercise restoration beyond simply calling model.train().
        base = observed.module if mode == "ddp" else observed
        base.get_input_embeddings().eval()
        ref_base = reference.module if mode == "ddp" else reference
        ref_base.get_input_embeddings().eval()
        if production:
            from trlx import TrlxError, quality

            guard = quality.observational
            evaluation_data = Dataset.from_list([{"prompt": "t0 t1", "answer": "t2"},
                                                 {"prompt": "t3", "answer": "t4"}])
            settings = SimpleNamespace(quality_preset="qa", quality_max_length=128,
                                       quality_max_new_tokens=3, quality_batch_size=2, judge=None)
        else:
            guard = observational_evaluation
        for fail in (False, True):
            before = training_state(observed, optimizers[1], schedulers[1])
            configs = [(module, "generation_config" in vars(module), vars(module).get("generation_config"))
                       for module in base.modules()]
            caught = False
            try:
                with guard(observed):
                    random.random()
                    np.random.random()
                    torch.rand(2, device=device)
                    if production:
                        injected = (patch.object(quality.quality_scorers, "score_generation",
                                                 side_effect=TrlxError("synthetic scorer failure"))
                                    if fail and rank == 0 else contextlib.nullcontext())
                        with injected:
                            results = quality.evaluate(base, tokenizer(), evaluation_data, settings, rank=rank)
                        assert rank != 0 or len(results) == 2
                    else:
                        generated = base.generate(input_ids=ids[:, :2], attention_mask=torch.ones_like(ids[:, :2]),
                                                  max_new_tokens=3, do_sample=False, use_cache=False,
                                                  synced_gpus=mode == "fsdp")
                        assert generated.shape[0] == 2
                        if fail:
                            raise ValueError("synthetic scorer failure")
            except Exception as error:
                if not fail or "synthetic scorer failure" not in str(error):
                    raise
                caught = True
            assert caught == fail, "injected scorer failure was not observed on every rank"
            assert equal(before, training_state(observed, optimizers[1], schedulers[1])), "evaluation mutated state"
            if production:
                for module, owned, original in configs:
                    assert ("generation_config" in vars(module)) == owned, "generation config owner changed"
                    if owned:
                        assert vars(module)["generation_config"] is original, "generation config alias changed"

        if production:
            before = training_state(observed, optimizers[1], schedulers[1])
            original_generate = base.generate

            # Fail on every rank after real generation to exercise FSDP's
            # post-forward cleanup before the next optimizer update.
            def failed_generation(**arguments):
                original_generate(**arguments)
                raise RuntimeError("synthetic generation failure")

            with guard(observed), patch.object(base, "generate", failed_generation):
                try:
                    quality._generate(base, input_ids=ids[:, :2], attention_mask=torch.ones_like(ids[:, :2]),
                                      max_new_tokens=3, do_sample=False, use_cache=False,
                                      synced_gpus=mode == "fsdp")
                except RuntimeError as error:
                    assert "synthetic generation failure" in str(error)
                else:
                    raise AssertionError("generation failure was swallowed")
                assert base.generate is failed_generation, "generation method was not restored"
            assert equal(before, training_state(observed, optimizers[1], schedulers[1])), "failed generation mutated state"

        start = rng_state()
        reference_loss = step(reference, optimizers[0], schedulers[0], ids)
        reference_after = training_state(reference, optimizers[0], schedulers[0])
        restore_rng(start)
        observed_loss = step(observed, optimizers[1], schedulers[1], ids)
        assert equal(reference_after, training_state(observed, optimizers[1], schedulers[1])), "next update diverged"
        print(json.dumps({"mode": mode, "rank": rank, "production": production, "peft": peft, "state_preserved": True,
                          "next_loss": observed_loss, "control_loss": reference_loss}), flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


# Choose an ephemeral local rendezvous port solely for the two synthetic workers.
def free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


# Explicit modes keep CPU preprocessing checks separate from GPU allocations and collectives.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("parity", "cpu", "single", "ddp", "fsdp"))
    parser.add_argument("--production", action="store_true", help="Exercise the actual independent-quality evaluator.")
    parser.add_argument("--peft", action="store_true", help="Wrap the synthetic model with actual LoRA; requires --production.")
    args = parser.parse_args()
    if args.peft and not args.production:
        parser.error("--peft requires --production")
    if args.mode == "parity":
        print(json.dumps(check_sft(), indent=2))
    elif args.mode in {"ddp", "fsdp"}:
        torch.multiprocessing.spawn(check_state, args=(2, args.mode, free_port(), args.production, args.peft), nprocs=2, join=True)
    else:
        check_state(0, 1, args.mode, None, args.production, args.peft)


if __name__ == "__main__":
    main()
