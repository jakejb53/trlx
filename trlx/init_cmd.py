"""`trlx init`: persistent environment defaults for every training method.

Hardware can determine native precision support, but cannot establish model
fit or optimal batch sizes without a model and data. The emitted configuration
distinguishes those measured capabilities from conservative starting settings.
"""

import dataclasses

from dataset.io import DatasetError, validate_output, write_text
from trlx import TrlxError, config, hardware, trainers
from trlx.hardware import Hardware
from trlx.toml_write import Writer

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

# These are visible starting values, not a claim that a particular model fits.
# Native BF16 is a hardware fact; batch sizes require later model/data tuning.
def render(system: Hardware) -> str:
    bf16 = bool(system.gpus) and all(gpu.bf16 for gpu in system.gpus)
    dtype = "bfloat16" if bf16 else "float32"
    w = Writer()
    w.comment("trlx defaults for all training methods. Written by trlx init.")
    w.comment("CLI arguments override this file for one run. Edit this file for persistent changes. "
              "Shared settings go before table headings; methods.NAME overrides shared values.")
    w.comment(f"Detected {system.cpu_count} logical CPUs and {len(system.gpus)} visible CUDA devices.")
    for gpu in system.gpus:
        w.comment(f"GPU {gpu.index}: {gpu.name}; {gpu.free_bytes / 2**30:.1f} GiB free / "
                  f"{gpu.total_bytes / 2**30:.1f} GiB total; native BF16: {gpu.bf16}.")
    w.comment("Memory availability is a snapshot. Without a model and dataset, batch sizing is "
              "conservative, not calibrated; model fit and optimal throughput are unknown.")
    if not system.gpus:
        w.comment("No CUDA GPUs detected. These FP32 defaults can be prepared on a CPU host; "
                  "training requires visible CUDA GPUs.")
    w.blank()

    w.comment("One pass through the training data; -1 means no fixed optimizer-step limit.")
    w.key("num_train_epochs", 1.0)
    w.key("max_steps", -1)
    w.comment("One example per GPU at a time; accumulate eight batches before an optimizer update.")
    w.key("per_device_train_batch_size", 1)
    w.key("gradient_accumulation_steps", 8)
    w.key("per_device_eval_batch_size", 1)
    w.comment("Evaluate and save after each epoch, retain two checkpoints, log every optimizer update.")
    w.key("eval_strategy", "epoch")
    w.key("save_strategy", "epoch")
    w.key("save_total_limit", 2)
    w.key("logging_steps", 1)
    w.key("report_to", "none")
    w.comment("Checkpoint activations to reduce memory. Use BF16 only when all visible GPUs support it natively.")
    w.key("gradient_checkpointing", True)
    w.key("bf16", bf16)
    w.key("fp16", False)
    w.comment("Load data in the trainer process to avoid multiplying unknown dataset memory use.")
    w.key("dataloader_num_workers", 0)
    w.key("dataloader_pin_memory", bool(system.gpus))
    w.blank()

    w.comment("Use all visible GPUs and estimate the launch strategy from the model at training time. "
              "Print line-based progress and run verification after training. CLI flags override these settings.")
    w.table("run")
    for key, value in config.run_settings({}).items():
        w.key(key, value)
    w.blank()

    w.comment("Supply the base model with --model, or persist its local path / Hub id here.")
    _model_block(w, ("model",), dtype)
    w.comment("Supply --dataset with a file or Hub reference. Hold out the final 10% in file order.")
    w.table("dataset")
    w.key("split", True)
    w.key("eval_fraction", 0.1)
    w.key("dataset", None)
    w.blank()

    w.comment("Train a LoRA adapter on all linear layers, including vision layers when present. "
              "Remove this block for full fine-tuning; trlx selects the appropriate task_type.")
    w.table("peft")
    w.key("r", 8)
    w.key("lora_alpha", 16)
    w.key("lora_dropout", 0.05)
    w.key("target_modules", "all-linear")
    w.blank()

    w.comment("Only the chosen training method's section applies. Additional TRL fields may be "
              "added by name; omitted fields use TRL defaults. See trlx METHOD --help.")
    for method in trainers.METHODS.values():
        _method_block(w, method, dtype)
    return w.text()


# Keep the model and distillation teacher at the same hardware-selected dtype.
# Missing paths remain absent so the CLI must supply a real model, never a placeholder.
def _model_block(w, table, dtype):
    w.table_path(*table)
    w.key("path", None)
    w.key("dtype", dtype)
    w.key("trust_remote_code", False)
    w.key("attn_implementation", "sdpa")
    w.blank()


# Method-specific defaults are stored, not reconstructed when launching a run.
# Fields we do not override come from the installed TRL dataclass, without construction.
def _method_block(w, method, dtype):
    fields = {field.name: field for field in dataclasses.fields(method.config_cls)}
    w.table_path("methods", method.name)
    w.key("output_dir", "runs/" + method.name)
    learning_rate = 1e-4 if method.name in {"sft", "reward", "distillation"} else _default(fields["learning_rate"])
    w.key("learning_rate", learning_rate)
    if "rewards" in method.blocks:
        w.comment("Training generations must divide the effective generation batch; "
                  "evaluation uses one generation so a single-prompt batch works.")
        w.key("num_generations", _default(fields["num_generations"]))
        w.key("num_generations_eval", 1)
    w.blank()

    w.comment("Display columns and expected intervals; values outside these ranges are marked.")
    w.table_path("methods", method.name, "ranges")
    for metric, bounds in RANGES[method.name].items():
        w.key(metric, bounds)
    w.blank()

    if "preflight" in method.blocks:
        w.comment("Warn about off-policy responses using up to 64 training rows; "
                  "threshold is mean log probability per token.")
        w.table_path("methods", method.name, "preflight")
        w.key("rows", 64)
        w.key("offpolicy_logp_per_token", -1.0)
        w.blank()
    if "teacher" in method.blocks:
        w.comment("Distillation requires a teacher: supply --teacher or set its path here.")
        _model_block(w, ("methods", method.name, "teacher"), dtype)
    if "rewards" in method.blocks:
        w.comment("Supply an explicit reward objective via CLI or funcs here. "
                  "Entries accept reward names, factories, model paths, or Python callables.")
        w.table_path("methods", method.name, "rewards")
        w.key("funcs", None)
        w.blank()


# Dataclass factories provide defaults without instantiating training arguments,
# which would otherwise initialize devices and validate missing run-specific inputs.
def _default(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None


# Existing settings require explicit replacement. Detect and render first so a
# failed inspection cannot damage them, including with direct publication.
def write(out="run.toml", force=False, no_staging=False) -> Hardware:
    try:
        validate_output(out, force=force)
        system = hardware.inspect()
        text = render(system)
        write_text(out, text, force=force, no_staging=no_staging)
    except DatasetError as exc:
        raise TrlxError(str(exc)) from exc
    return system
