"""Registry: method name -> everything method-specific the other modules need.

This is the only place a method name is mapped to TRL classes. config.py,
init_cmd.py, model.py, and train.py all look up here; none of them branch on
the method name themselves.

Importing this module imports trl, which imports torch. cli.py imports it
inside command handlers so `trlx --help` stays fast.
"""

import dataclasses

import trl
from trl.trainer.distillation_config import DistillationConfig


@dataclasses.dataclass(frozen=True)
class Method:
    name: str
    config_cls: type
    trainer_cls: type
    # "causal": class named by the model's own config. "sequence_classification":
    # transformers' config-keyed sequence-classification mapping (reward).
    model_kind: str
    # peft task type set by trlx on the [peft] block; never operator-visible.
    peft_task_type: str
    # Dataset shape the trainer consumes, in TRL's vocabulary. data_load.py
    # (Phase 5) validates rows against it.
    dataset_format: str
    # Method-specific config blocks allowed beyond UNIVERSAL_BLOCKS.
    blocks: frozenset


# Blocks every method accepts.
UNIVERSAL_BLOCKS = frozenset({"model", "dataset", "peft", "ranges", "verify"})


def _method(name, config_cls, trainer_cls, dataset_format, blocks=(), model_kind="causal"):
    task_type = "SEQ_CLS" if model_kind == "sequence_classification" else "CAUSAL_LM"
    return Method(name, config_cls, trainer_cls, model_kind, task_type, dataset_format, frozenset(blocks))


# Order is the order shown in --help and by init.
METHODS = {
    m.name: m
    for m in (
        _method("sft", trl.SFTConfig, trl.SFTTrainer, "language modeling or prompt-completion", {"replay"}),
        _method("dpo", trl.DPOConfig, trl.DPOTrainer, "preference", {"preflight"}),
        _method("grpo", trl.GRPOConfig, trl.GRPOTrainer, "prompt-only", {"rewards"}),
        _method("kto", trl.KTOConfig, trl.KTOTrainer, "unpaired preference", {"preflight"}),
        _method("rloo", trl.RLOOConfig, trl.RLOOTrainer, "prompt-only", {"rewards"}),
        _method("reward", trl.RewardConfig, trl.RewardTrainer, "preference", model_kind="sequence_classification"),
        _method("distillation", DistillationConfig, trl.DistillationTrainer, "prompt-only", {"teacher"}),
    )
}


# Lookup by name. The CLI restricts choices to METHODS, so a miss is a bug.
def get(name):
    return METHODS[name]
