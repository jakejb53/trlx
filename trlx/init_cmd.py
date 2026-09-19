"""`trlx init <method> --out <path>`: writes a run config template.

Everything about a TRL or peft field (name, default, help text, whether it
admits None) is read from the live dataclass at generation time. This file
holds only what is trlx's own: the curated TrainingArguments subset, the
initial [ranges] per method, the block descriptions, and the placeholder
values for keys that have no meaningful default.

Nothing here names a model family, an architecture, or a machine. Placeholders
are written where the operator must supply a value.
"""

import dataclasses
import pathlib

from peft import LoraConfig
from transformers import TrainingArguments

from trlx import TrlxError, config, trainers
from trlx.toml_write import Writer, format_value

# TrainingArguments fields written by init, in this order. Any other field may
# be added to the config by name.
CURATED = [
    "output_dir", "run_name", "learning_rate", "num_train_epochs", "max_steps",
    "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
    "eval_strategy", "eval_steps", "save_strategy", "save_steps", "save_total_limit",
    "logging_steps", "warmup_steps", "lr_scheduler_type", "weight_decay", "max_grad_norm",
    "optim", "bf16", "gradient_checkpointing", "seed", "resume_from_checkpoint",
]

# trlx's own rules for curated fields, appended to the field's help text.
CURATED_NOTES = {
    "output_dir": "trlx: required. The run directory (SPEC 2.3).",
    "run_name": "trlx: when absent, the last component of output_dir.",
    "save_steps": "trlx: when absent, equal to eval_steps.",
}

# Initial [ranges] per method. Intervals are starting points to be tuned
# against real runs; the metric names are the display columns.
RANGES = {
    "sft": {"loss": [0, 5], "eval_loss": [0, 5], "mean_token_accuracy": [0, 1], "grad_norm": [0, 10]},
    "dpo": {
        "loss": [0, 5], "eval_loss": [0, 5], "rewards/accuracies": [0, 1], "rewards/margins": [0, 10],
        "logps/chosen": [-2000, 0], "grad_norm": [0, 10],
    },
    "kto": {
        "loss": [0, 5], "eval_loss": [0, 5], "rewards/chosen": [-10, 10], "rewards/rejected": [-10, 10],
        "kl": [0, 1], "grad_norm": [0, 10],
    },
    "grpo": {"reward": [-10, 10], "reward_std": [0, 10], "kl": [0, 1], "completions/mean_length": [0, 4096], "grad_norm": [0, 10]},
    "rloo": {"reward": [-10, 10], "reward_std": [0, 10], "kl": [0, 1], "completions/mean_length": [0, 4096], "grad_norm": [0, 10]},
    "reward": {"loss": [0, 5], "eval_loss": [0, 5], "accuracy": [0, 1], "grad_norm": [0, 10]},
    "distillation": {"loss": [0, 5], "eval_loss": [0, 5], "grad_norm": [0, 10]},
}

# [model] and [teacher] keys with the values init writes. `path` is the one
# placeholder; the rest are the values an operator most often wants and can
# change.
MODEL_BLOCK = [
    ("path", "<model path or HF id>", "Local directory or HF model id. The model class is read from its config."),
    ("dtype", "bfloat16", "torch dtype name the weights are loaded in."),
    ("trust_remote_code", False, "Allow model code shipped with the checkpoint."),
    ("attn_implementation", "sdpa", "Attention backend passed to from_pretrained, e.g. sdpa, eager, flash_attention_2."),
]

_TA_FIELDS = {f.name for f in dataclasses.fields(TrainingArguments)}


# Renders the template for one method as TOML text.
def render(method_name):
    method = trainers.get(method_name)
    cls = method.config_cls
    fields = {f.name: f for f in dataclasses.fields(cls)}
    w = Writer()

    w.comment(f"trlx run config for {method.name}. Written by `trlx init {method.name}`.")
    w.comment(
        f"Top-level keys are fields of {cls.__name__}: unknown keys are errors, absent keys take the "
        'dataclass default, and the string "None" sets a field to None where its type allows. '
        "Keys written commented out default to None. Every other TrainingArguments field may be "
        "added by name. The [blocks] are trlx's own."
    )
    w.blank()

    w.comment("Training arguments (transformers.TrainingArguments, curated subset)")
    w.blank()
    for name in CURATED:
        f = fields[name]
        value = "runs/" + method.name if name == "output_dir" else _default(f)
        _field(w, f, value, note=CURATED_NOTES.get(name))

    w.comment(f"{cls.__name__} fields")
    w.blank()
    # Fields config.load rejects are not offered: model loading belongs to
    # [model], and grpo/rloo's vLLM switches are forced on.
    owned = set(config.MODEL_LOADING_FIELDS) | config.STRATEGY_FIELDS
    if "rewards" in method.blocks:
        owned |= set(config.VLLM_FORCED)
    for f in dataclasses.fields(cls):
        if f.name in _TA_FIELDS or f.name in owned:
            continue
        _field(w, f, _default(f))

    _model_block(w, "model", "The model to train.")
    if "teacher" in method.blocks:
        _model_block(w, "teacher", "The teacher for distillation. Same keys as [model].")

    w.comment("Datasets. Files by extension (.jsonl, .json, .csv, .parquet) or HF ids as org/name:split.")
    w.comment("split = true: `dataset` is cut after `train` rows (file order); the rest is eval.")
    w.comment("split = false: `dataset_train` and optional `dataset_eval`; without dataset_eval, "
              "evaluation is disabled and eval_* keys are rejected.")
    w.table("dataset")
    w.key("split", True)
    # A placeholder that is itself a valid reference, so the template loads.
    w.key("dataset", "data/train.jsonl")
    w.key("train", 1000)
    w.key("dataset_train", None)
    w.key("dataset_eval", None)
    w.blank()

    _peft_block(w, method)

    w.comment("Expected interval per metric. Required. Metrics named here are the display columns; "
              "values outside their interval are marked.")
    w.table("ranges")
    for metric, bounds in RANGES[method.name].items():
        w.key(metric, bounds)
    w.blank()

    if "preflight" in method.blocks:
        w.comment("Off-policy warning: responses whose mean per-token log-prob under the starting "
                  "model is below the threshold are reported. `rows` bounds the check to the first "
                  "N train rows; it is one forward pass per response.")
        w.table("preflight")
        w.key("offpolicy_logp_per_token", -1.0)
        w.key("rows", 64)
        w.blank()

    if "rewards" in method.blocks:
        w.comment("Reward functions, in order. Each entry is a bare name from trl.rewards or a trlx "
                  "built-in, {name = ..., args = {...}} for a factory, an HF model path, or "
                  "module:function / path.py:function.")
        w.table("rewards")
        w.key("funcs", ["think_format_reward"])
        w.blank()

    if "replay" in method.blocks:
        w.comment("Replay: mix `dataset` into training so `fraction` of the mixed train set is replay "
                  "rows (built by `trlx replay-build`); kl_coef > 0 adds a KL term against the original "
                  "model on replay batches. With kl_coef > 0, trlx sets loss_type = \"nll\" (the KL needs "
                  "logits) and rejects the key, and use_liger_kernel, packing, and padding_free are refused.")
        w.table("replay", commented=True)
        w.key("dataset", None)
        w.key("fraction", None)
        w.key("kl_coef", None)
        w.blank()

    w.comment("Prompts for the post-training generation check. A built-in set is used when absent.")
    w.table("verify", commented=True)
    w.key("prompts", None)

    return w.text()


# Default value of a dataclass field, materialising factories. MISSING is
# treated as None so the key is written commented out.
def _default(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


# Help text, optional trlx note, then the key. None values come out commented.
def _field(w, f, value, note=None, commented=False):
    w.comment(f.metadata["help"])
    if note:
        w.comment(note)
    w.key(f.name, value, commented=commented)
    w.blank()


def _model_block(w, name, description):
    w.comment(description)
    w.table(name)
    for key, value, help_text in MODEL_BLOCK:
        w.comment(help_text)
        w.key(key, value)
    w.blank()


# [peft] is optional (absent means full fine-tune), so the whole block is
# commented out. Fields are LoraConfig's own, minus those typed as peft
# config objects, which TOML cannot express.
def _peft_block(w, method):
    w.comment("LoRA adapter (peft.LoraConfig fields). Absent means full fine-tune. "
              "Uncomment the header and the fields to set to train an adapter. task_type is set by trlx.")
    w.table("peft", commented=True)
    w.blank()
    for f in dataclasses.fields(LoraConfig):
        if f.name in config.PEFT_OWNED_FIELDS:
            continue
        value = _default(f)
        if value is not None and not _expressible(value):
            continue
        _field(w, f, value, commented=True)


# True when format_value can spell the value; nested dataclass defaults cannot
# be written and are skipped.
def _expressible(value):
    try:
        format_value(value)
    except TypeError:
        return False
    return True


# Writes the template to `out`. Refuses an existing file: a template must never
# replace a config an operator has edited.
def write(method_name, out):
    path = pathlib.Path(out)
    if path.exists():
        raise TrlxError(f"{out}: already exists; init does not overwrite")
    text = render(method_name)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise TrlxError(f"{out}: cannot write: {e.strerror or e}")
