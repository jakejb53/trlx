"""Review relevant, overridable settings before training can load or publish anything."""

import contextlib
import dataclasses
import enum
import re
import shlex
import shutil
import sys
import textwrap
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from trlx import TrlxError, config, options, toml_write


# These lists select presentation only. Values and types always come from the
# resolved configuration and installed libraries, never a second defaults table.
_COMMON = frozenset("""
    num_train_epochs max_steps learning_rate optim lr_scheduler_type warmup_steps
    weight_decay max_grad_norm per_device_train_batch_size gradient_accumulation_steps
    bf16 fp16 gradient_checkpointing seed data_seed train_sampling_strategy
    eval_strategy eval_steps per_device_eval_batch_size save_strategy save_steps
    save_total_limit load_best_model_at_end metric_for_best_model greater_is_better logging_steps
""".split())
_GENERATION = frozenset("""
    max_completion_length temperature top_p top_k min_p repetition_penalty
""".split())
_POLICY = frozenset("""
    num_generations num_generations_eval generation_batch_size steps_per_generation
    beta num_iterations epsilon epsilon_high reward_weights mask_truncated_completions
    disable_dropout sync_ref_model shuffle_dataset
""".split())
_METHOD = {
    "sft": frozenset("max_length truncation_mode packing padding_free completion_only_loss assistant_only_loss loss_type shuffle_dataset".split()),
    "dpo": frozenset("max_length truncation_mode loss_type beta label_smoothing f_divergence_type use_weighting disable_dropout precompute_ref_log_probs sync_ref_model".split()),
    "kto": frozenset("max_length loss_type beta desirable_weight undesirable_weight disable_dropout precompute_ref_log_probs sync_ref_model".split()),
    "reward": frozenset("max_length center_rewards_coefficient disable_dropout".split()),
    "grpo": _GENERATION | _POLICY | frozenset("loss_type scale_rewards multi_objective_aggregation importance_sampling_level vllm_importance_sampling_correction entropy_coef".split()),
    "rloo": _GENERATION | _POLICY | {"normalize_advantages", "reward_clip_range"},
    "distillation": _GENERATION | {"beta", "disable_dropout", "shuffle_dataset", "use_vllm"},
}
_LORA = {"r", "lora_alpha", "lora_dropout", "target_modules", "bias", "modules_to_save"}
_LOSS_FEATURES = {
    "discopop": {"discopop_tau"},
    "sapo": {"sapo_temperature_neg", "sapo_temperature_pos"},
    "vespo": {"vespo_k_pos", "vespo_lambda_pos", "vespo_k_neg", "vespo_lambda_neg"},
}

# Feature details appear only with the feature that consumes them. Explicit
# settings still pass through _applicable so disabled features cannot look active.
_FEATURES = {
    "packing": {"packing_strategy", "eval_packing"},
    "precompute_ref_log_probs": {"precompute_ref_batch_size"},
    "sync_ref_model": {"ref_model_mixup_alpha", "ref_model_sync_steps"},
    "use_liger_kernel": {"use_liger_kernel", "liger_kernel_config"},
    "torch_compile": {"torch_compile", "torch_compile_backend", "torch_compile_mode"},
    "activation_offloading": {"activation_offloading"},
    "use_adaptive_entropy": {"use_adaptive_entropy", "entropy_coef_min", "entropy_coef_max", "entropy_coef_delta", "entropy_target"},
    "vllm_importance_sampling_correction": {"vllm_importance_sampling_mode", "vllm_importance_sampling_clip_max", "vllm_importance_sampling_clip_min"},
    "push_to_hub": {"push_to_hub", "hub_model_id", "hub_private_repo", "hub_strategy", "hub_revision"},
}
_SECRET = re.compile(r"^(?:token|hub_token|push_to_hub_token|api_token)$|(?:^|_)(?:api_key|access_token|refresh_token|password|secret|authorization|credential)$", re.I)
_AUTO_NOTES = {
    "completion_only_loss": "Automatic: completion-only for prompt/completion data; full sequence for language-modeling data.",
    "generation_batch_size": "Automatic: derived in each worker from batch size, GPU count, and steps per generation.",
    "steps_per_generation": "Automatic: follows gradient accumulation unless generation batch size is set.",
    "eval_packing": "Inherits packing.",
    "num_generations_eval": "Inherits num-generations.",
    "epsilon_high": "Inherits epsilon.",
    "peft.target_modules": "Inferred from the model architecture after loading.",
}


# Flatten only config sections; nested trainer dictionaries remain one CLI value.
def _explicit_keys(document):
    blocks = {"model", "teacher", "dataset", "peft", "run", "preflight", "replay", "verify", "rewards"}
    keys = set()
    for key, value in document.items():
        if key in blocks and isinstance(value, dict):
            keys.update(f"{key}.{name}" for name in value)
        else:
            keys.add(key)
    return keys


# Selection follows the chosen trainer and enabled features, not other method sections.
def _selected(cfg):
    selected = set(_COMMON | _METHOD[cfg.method.name]) | _explicit_keys(cfg.document)
    # Metric display ranges do not tune training and can dwarf the review's columns.
    selected.discard("ranges")
    selected.update({"model.path", "model.dtype", "dataset.source", "dataset.split", "dataset.eval_fraction",
                     "dataset.dataset_eval", "output_dir", "resume_from_checkpoint", "rewards.funcs"})
    for feature, details in _FEATURES.items():
        if getattr(cfg.args, feature, False):
            selected.update(details)
    if cfg.peft is not None:
        selected.update("peft." + name for name in _LORA)
    if cfg.teacher is not None:
        selected.update({"teacher.path", "teacher.dtype"})
    if getattr(cfg.args, "use_vllm", False):
        selected.update({"vllm_server_base_url", "vllm_server_host", "vllm_server_port", "vllm_server_timeout"})
    loss = getattr(cfg.args, "loss_type", None)
    losses = [loss] if isinstance(loss, str) else loss or []
    if len(losses) > 1:
        selected.add("loss_weights")
    for loss, details in _LOSS_FEATURES.items():
        if loss in losses:
            selected.update(details)
    if getattr(cfg.args, "f_divergence_type", None) == "alpha_divergence":
        selected.add("f_alpha_divergence_coef")
    return selected


# Omit controls known to be unused even when present in the input document.
def _applicable(key, cfg, controls):
    args = cfg.args
    evaluation = cfg.dataset.eval_enabled and args.eval_strategy != "no"
    if key.startswith("run."):
        return True
    if key.startswith("peft."):
        return cfg.peft is not None
    if key.startswith("teacher."):
        return cfg.teacher is not None
    if key.startswith("replay."):
        return cfg.replay is not None
    if key.startswith("preflight."):
        return cfg.preflight is not None
    if key.startswith("verify."):
        return controls["verify"]
    if key.startswith("rewards."):
        return cfg.rewards is not None
    if key == "dataset.dataset_train":
        return False  # --dataset already addresses the primary source in either mode.
    if key == "dataset.eval_fraction":
        return cfg.dataset.split
    if key == "dataset.dataset_eval":
        return not cfg.dataset.split and cfg.dataset.eval_source is not None
    if key == "num_train_epochs":
        return args.max_steps <= 0
    if key == "max_steps":
        return args.max_steps > 0
    if (key.startswith("eval_") and key != "eval_strategy") or key in {"per_device_eval_batch_size", "num_generations_eval"}:
        if not evaluation:
            return False
    if key == "eval_steps":
        return args.eval_strategy == "steps"
    if key == "save_steps":
        return args.save_strategy == "steps"
    if key == "save_total_limit":
        return args.save_strategy != "no"
    if key == "logging_steps":
        return args.logging_strategy == "steps"
    if key in {"metric_for_best_model", "greater_is_better"}:
        return args.load_best_model_at_end or args.lr_scheduler_type in {"reduce_lr_on_plateau", "greedy"}
    if key in {"resume_from_checkpoint", "ignore_data_skip", "restore_callback_states_from_checkpoint"}:
        return bool(args.resume_from_checkpoint)
    if key == "run_name":
        return cfg.document.get("run_name") not in (None, "None")
    loss = getattr(args, "loss_type", None)
    losses = [loss] if isinstance(loss, str) else loss or []
    for loss, details in _LOSS_FEATURES.items():
        if key in details and loss not in losses:
            return False
    if key == "f_alpha_divergence_coef":
        return args.f_divergence_type == "alpha_divergence"
    # Generation dictionaries override sampling-only controls. Temperature and
    # completion length also affect training, so their separate roles are explained.
    generation = getattr(args, "generation_kwargs", None) or {}
    if key in {"top_p", "top_k", "min_p", "repetition_penalty", "cache_implementation"} and key in generation:
        return False
    for feature, details in _FEATURES.items():
        if key in details and key != feature and not getattr(args, feature, False):
            return False
    if key.startswith("vllm_"):
        if not getattr(args, "use_vllm", False):
            return False
        server = args.vllm_mode == "server"
        if key in {"vllm_gpu_memory_utilization", "vllm_max_model_length", "vllm_tensor_parallel_size", "vllm_enable_sleep_mode"}:
            return not server
        if key.startswith("vllm_server_"):
            if not server:
                return False
            if key in {"vllm_server_host", "vllm_server_port"}:
                return args.vllm_server_base_url is None
            if key == "vllm_server_base_url":
                return args.vllm_server_base_url is not None
    if key == "truncation_mode" and getattr(args, "packing", False):
        return False
    return True


# Convert dataset references back to the syntax accepted by the public CLI.
def _reference(value):
    if isinstance(value, config.DatasetRef):
        return value.source if value.split is None else f"{value.source}:{value.split}"
    return value


# Worker-dependent automatic inputs must not masquerade as supervisor-derived facts.
def _value(key, cfg, controls):
    if key == "ranges":
        return cfg.ranges
    if key in {"generation_batch_size", "steps_per_generation"}:
        value = cfg.document.get(key)
        return None if value == "None" else value
    if "." not in key:
        if key == "output_dir" and cfg.args.resume_from_checkpoint:
            from pathlib import Path

            return str(Path(cfg.args.output_dir).parent)
        return getattr(cfg.args, key)
    block, name = key.split(".", 1)
    if block == "run":
        return controls[name]
    if block == "dataset":
        name = {"dataset_eval": "eval_source"}.get(name, name)
        return _reference(getattr(cfg.dataset, name))
    if block == "verify":
        return _reference(cfg.verify_prompts)
    if block == "rewards":
        return cfg.document["rewards"]["funcs"]
    return _reference(getattr(getattr(cfg, block), name))


# Normalize instantiated library values without mistaking representation changes for redaction.
def _plain(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in (sorted(value) if isinstance(value, set) else value)]
    return value


# Redact credential fields recursively, including credentials embedded in URLs.
# api_key names an environment variable in trlx, but is still withheld here so
# custom reward arguments cannot accidentally print a literal secret.
def _redact(value, key=""):
    if _SECRET.search(key) and value is not None:
        return "<redacted>"
    if isinstance(value, dict):
        return {name: _redact(item, name) for name, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact(item) for item in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, str) and "://" in value:
        try:
            parts = urlsplit(value)
            host = parts.netloc.rsplit("@", 1)[-1]
            netloc = "<redacted>@" + host if "@" in parts.netloc else host
            pairs = parse_qsl(parts.query, keep_blank_values=True)
            safe_pairs = [(name, _redact(item, name)) for name, item in pairs]
            if netloc == parts.netloc and safe_pairs == pairs:
                return value
            query = urlencode(safe_pairs)
            return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
        except ValueError:
            return "<redacted URL>"
    return value


# Nested nulls introduced by library dataclasses have no faithful TOML spelling.
def _contains_none(value):
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_none(item) for item in value)
    return value is None


# Render one valid override fragment; None is legal only for nullable CLI types.
def _argument(setting, value):
    if value is None:
        return f"{setting.flag} None" if config.admits_none(setting.hint) else None
    non_null = [item for item in options._alternatives(setting.hint) if item is not type(None)]
    if isinstance(value, bool) and non_null == [bool]:
        return setting.flag if value else "--no-" + setting.flag[2:]
    if _contains_none(value):
        return None
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, str):
        # A TOML string also protects union fields such as str|list from parsing
        # literal text (e.g. "123" or "None") as a different value.
        alternatives = options._alternatives(setting.hint)
        simple = all(item in (str, type(None)) or isinstance(item, type) and issubclass(item, enum.Enum)
                     for item in alternatives)
        raw = value if setting.key == "rewards.funcs" or simple and not value.startswith('"') and value != "None" else toml_write.format_value(value)
    else:
        raw = toml_write.format_value(value)
    return f"{setting.flag} {shlex.quote(raw)}"


# The first help sentence identifies the control; full reference help stays in --help.
# Automatic and precedence notes take priority over brevity because they affect meaning.
def _description(setting, value, cfg):
    description = setting.description.split(" Library default when absent from config:", 1)[0]
    description = " ".join(description.replace("%%", "%").split())
    description = re.split(r"(?<=[.!?])\s+(?=[A-Z])", description, maxsplit=1)[0]
    if value is None and setting.key in _AUTO_NOTES:
        description = _AUTO_NOTES[setting.key]
    generation = getattr(cfg.args, "generation_kwargs", None) or {}
    if setting.key == "temperature" and "temperature" in generation:
        description = "Training/scoring temperature; generation temperature is overridden by --generation-kwargs."
    if setting.key == "max_completion_length" and any(name in generation for name in ("max_new_tokens", "max_tokens")):
        description = "Trainer completion-length setting; generation length is overridden by --generation-kwargs."
    return f"{description} Type: {options.type_label(setting.hint)}."


# Every row is constructed before column sizing, so long values cannot misalign comments.
def render(cfg, *, width=None):
    controls = config.run_settings(cfg.document)
    selected = _selected(cfg)
    rows = []
    forced = dict(config.VLLM_FORCED) if cfg.rewards is not None else {}
    if "disable_tqdm" in selected:
        forced["disable_tqdm"] = True
    if cfg.method.name == "sft" and cfg.args.packing and cfg.args.packing_strategy == "bfd":
        forced["padding_free"] = True
    if cfg.replay is not None and cfg.replay.kl_coef > 0:
        forced.update(config.REPLAY_KL_FORCED)
    for setting in options.settings(cfg.method.name):
        key = setting.key
        if not (key in selected or key.startswith("run.")) or not _applicable(key, cfg, controls):
            continue
        if key in forced:
            continue
        value = _value(key, cfg, controls)
        values = value if key == "rewards.funcs" else [value]
        for item in map(_plain, values):
            safe = _redact(item, key.rsplit(".", 1)[-1])
            # Redacted or non-nullable automatic values are explanatory comments,
            # never plausible override commands containing placeholders.
            redacted = safe != item
            argument = None if redacted else _argument(setting, safe)
            description = _description(setting, item, cfg)
            if argument is None:
                detail = "automatic/unset" if item is None else repr(safe)
                description = f"{setting.flag}: {detail}. {description}"
            rows.append((argument or "", description))
    if cfg.peft is None:
        rows.append(("--no-lora", "Full fine-tuning; LoRA is disabled. Type: bool."))
    for key, value in forced.items():
        rows.append(("", f"{key} = {value!r}; set by trlx and cannot be overridden for this run."))

    column = max((len(argument) for argument, _ in rows), default=0) + 3
    terminal_width = width if width is not None else shutil.get_terminal_size().columns
    # Keep a readable comment column even when a long argument exceeds the terminal.
    # Values are never shortened: terminal wrapping may occur, but no data is hidden.
    comment_width = max(40, terminal_width - column - 2)
    lines = ["Settings applied to this run:", ""]
    for argument, description in rows:
        pieces = textwrap.wrap(description, width=comment_width, break_long_words=False, break_on_hyphens=False)
        for index, piece in enumerate(pieces):
            lines.append(f"{argument if index == 0 else '':<{column}}# {piece}")
    return "\n".join(lines) + "\n"


# No durable run state exists yet. Missing consent or failed review I/O must stop
# startup, unlike a display failure after workers are already owned by the supervisor.
def confirm(cfg, *, progress=None):
    suspended = progress.suspended() if progress is not None else contextlib.nullcontext()
    with suspended:
        try:
            if sys.stdout is None or sys.stdin is None:
                raise TrlxError("settings review requires readable stdin and writable stdout; training was not started")
            sys.stdout.write(render(cfg))
            while True:
                sys.stdout.write("\nPress Enter to continue or q to quit: ")
                sys.stdout.flush()
                response = sys.stdin.readline()
                if response == "":
                    raise TrlxError("settings review reached EOF; press Enter to continue or q to quit; training was not started")
                answer = response.rstrip("\r\n")
                if answer == "":
                    return True
                if answer.lower() == "q":
                    sys.stdout.write("Training cancelled.\n")
                    sys.stdout.flush()
                    return False
                sys.stdout.write("Enter an empty line to continue or q to quit.\n")
        except (OSError, ValueError) as error:
            raise TrlxError("settings review could not read stdin or write stdout; training was not started") from error
