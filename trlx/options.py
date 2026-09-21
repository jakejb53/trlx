"""Typed, discoverable CLI overrides; absent options never replace config values."""

import argparse
import dataclasses
import enum
import functools
import textwrap
import tomllib
import types
import typing


# Parser and startup review share override names, types, and descriptions.
@dataclasses.dataclass(frozen=True)
class Setting:
    flag: str
    key: str
    hint: object
    description: str


# Preserve paragraphs/examples while bounding help to a normal terminal width.
class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    # Long option names get their own line instead of squeezing descriptions.
    def __init__(self, prog):
        super().__init__(prog, max_help_position=34, width=100)

    # Preserve example indentation while wrapping prose paragraphs for terminals.
    def _fill_text(self, text, width, indent):
        return "\n".join(indent + line if line.startswith("  ") else
                         textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent)
                         for line in text.splitlines())


# The type's alternatives decide whether an unquoted value is literal text.
def _alternatives(hint):
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return typing.get_args(hint)
    return (hint,)


# Parse scalar strings naturally and structured values with TOML, not eval.
# "None" retains the config loader's explicit spelling for nullable fields.
def parse_value(raw, hint):
    alternatives = _alternatives(hint)
    if raw == "None" and type(None) in alternatives:
        return raw
    for item in alternatives:
        if typing.get_origin(item) is typing.Literal and raw in typing.get_args(item):
            return raw
    scalar_string = all(
        item is str or item is type(None) or isinstance(item, type) and issubclass(item, enum.Enum)
        for item in alternatives
    )
    if scalar_string:
        if raw.startswith('"'):
            try:
                return tomllib.loads("value = " + raw)["value"]
            except tomllib.TOMLDecodeError as error:
                raise argparse.ArgumentTypeError(f"invalid quoted string: {error}") from error
        return raw
    try:
        return tomllib.loads("value = " + raw)["value"]
    except tomllib.TOMLDecodeError as error:
        # Union fields such as str|list accept a bare name as well as a TOML array.
        if str in alternatives and not raw.startswith(('[', '{', '"', "'")):
            return raw
        raise argparse.ArgumentTypeError(
            f"expected a {type_label(hint)} value; arrays/tables use quoted TOML syntax: {error}"
        ) from error


# Compact type guidance accompanies the upstream field description in --help.
def type_label(hint):
    labels = []
    for item in _alternatives(hint):
        if item is type(None):
            labels.append("None")
        elif typing.get_origin(item) is typing.Literal:
            labels.append("/".join(str(value) for value in typing.get_args(item)))
        else:
            base = typing.get_origin(item) or item
            labels.append(getattr(base, "__name__", str(base)))
    return " or ".join(labels)


# Nullable booleans retain an explicit reset to the library's automatic behavior.
def _boolean_or_none(raw):
    if raw == "None":
        return raw
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    raise argparse.ArgumentTypeError("expected true, false, or None")


# One field maps its CLI spelling directly to an effective-config key.
def _option(group, flag, key, hint, help_text, *, choices=None):
    # --dataset addresses the primary source in either split mode, not a TOML alias.
    setting = "dataset.dataset or dataset.dataset_train, according to split" if key == "dataset.source" else key
    text = f"{help_text} Config: {setting}; omitted CLI options retain the config value."
    kwargs = {"dest": "override:" + key, "default": argparse.SUPPRESS, "help": text.replace("%", "%%")}
    if choices is not None:
        kwargs["choices"] = choices
    non_null = [item for item in _alternatives(hint) if item is not type(None)]
    if non_null == [bool] and type(None) in _alternatives(hint):
        action = group.add_argument(flag, nargs="?", const=True, type=_boolean_or_none,
                                    metavar="{true,false,None}", **kwargs)
        action.setting = Setting(flag, key, hint, help_text)
        negative = dict(kwargs, help=f"Set {key} to false for this run.")
        group.add_argument("--no-" + flag[2:], action="store_false", **negative)
        return
    if non_null == [bool]:
        kwargs["action"] = argparse.BooleanOptionalAction
    else:
        kwargs["type"] = functools.partial(parse_value, hint=hint)
        kwargs["metavar"] = "VALUE"
        kwargs["help"] += f" Type: {type_label(hint)}."
    action = group.add_argument(flag, **kwargs)
    action.setting = Setting(flag, key, hint, help_text)


# These controls are shared by the launch parser and its pre-launch review.
def add_run_settings(parser, training):
    _option(parser, "--gpus", "run.gpus", str, "Visible device indices, e.g. 0,1, or all.")
    if training:
        _option(parser, "--strategy", "run.strategy", str,
                "Launch strategy; auto is selected after this review; ddp/fsdp require multiple GPUs.",
                choices=["auto", "ddp", "fsdp"])
        _option(parser, "--tui", "run.tui", bool, "Use the full-screen training display.")
        _option(parser, "--verify", "run.verify", bool, "Verify the final checkpoint after training.")


# Build only the selected method's option metadata; no config or model is loaded.
def settings(method_name):
    parser = argparse.ArgumentParser(add_help=False)
    add_run_settings(parser, training=True)
    add_training_options(parser, method_name)
    return [action.setting for action in parser._actions if hasattr(action, "setting")]


# The original field remains authoritative for defaults not specified in config.
def _field_help(field):
    text = field.metadata.get("help", field.name)
    if field.default is not dataclasses.MISSING:
        return f"{text} Library default when absent from config: {field.default!r}."
    return text


# Build method help from field metadata without constructing a trainer/config,
# querying GPUs, reading run.toml, or opening a model/dataset.
def add_training_options(parser, method_name):
    from peft import LoraConfig
    from transformers import TrainingArguments
    from trlx import config, trainers

    method = trainers.get(method_name)
    model = parser.add_argument_group("Model and dataset (CLI values override run.toml)")
    wrapper = [
        ("--model", "model.path", str, "Base model directory or model ID; required unless saved in config."),
        ("--dtype", "model.dtype", str, "Weight dtype: bfloat16, float16, or float32; training precision flags are separate."),
        ("--trust-remote-code", "model.trust_remote_code", bool, "Allow model-provided Python code."),
        ("--attn-implementation", "model.attn_implementation", str, "Attention backend, for example sdpa or eager."),
        ("--dataset", "dataset.source", str, "Training source: .jsonl/.json/.csv/.parquet or org/name:split."),
        ("--eval-fraction", "dataset.eval_fraction", float, "Evaluation share, strictly between 0 and 1; resolved at data load."),
        ("--split", "dataset.split", bool, "Split one source; --no-split uses separate files and removes eval_fraction."),
        ("--dataset-train", "dataset.dataset_train", str, "Training source with --no-split."),
        ("--dataset-eval", "dataset.dataset_eval", str, "Evaluation source with --no-split; omit for training only."),
        ("--verify-prompts", "verify.prompts", str, "Prompts for post-training verification; a built-in set is used if absent."),
    ]
    if "teacher" in method.blocks:
        wrapper += [
            ("--teacher", "teacher.path", str, "Teacher model directory or ID; required for distillation."),
            ("--teacher-dtype", "teacher.dtype", str, "Dtype for the teacher's weights."),
            ("--teacher-trust-remote-code", "teacher.trust_remote_code", bool, "Allow teacher-provided Python code."),
            ("--teacher-attn-implementation", "teacher.attn_implementation", str, "Teacher attention backend."),
        ]
    for flag, key, hint, description in wrapper:
        _option(model, flag, key, hint, description)

    common = parser.add_argument_group("Training settings")
    specific = parser.add_argument_group(f"{method_name} settings")
    shared_names = {field.name for field in dataclasses.fields(TrainingArguments)}
    hints = typing.get_type_hints(method.config_cls)
    owned = set(config.MODEL_LOADING_FIELDS) | config.STRATEGY_FIELDS
    if "rewards" in method.blocks:
        owned |= set(config.VLLM_FORCED)
    # These fields are resolved by the supervisor before TRL sees their values.
    run_help = {
        "output_dir": "Parent for fresh run directories YYYYMMDD-N--model--dataset; resume keeps its existing run.",
        "run_name": "Display label; defaults to the generated run-directory name. Does not select a directory.",
        "resume_from_checkpoint": "Checkpoint directory: load saved settings and resume in place, automatically "
                                  "discarding later metrics/checkpoints and stale reports. Logs are preserved. "
                                  "None disables a configured resume.",
    }
    for field in dataclasses.fields(method.config_cls):
        if not field.init or field.name.startswith("_") or field.name in owned:
            continue
        group = common if field.name in shared_names else specific
        _option(group, "--" + field.name.replace("_", "-"), field.name,
                hints.get(field.name, typing.Any), run_help.get(field.name, _field_help(field)))

    peft = parser.add_argument_group("LoRA adapters")
    peft.add_argument("--no-lora", action="store_true", help="Full fine-tuning for this run; remove the effective [peft] block.")
    hints = typing.get_type_hints(LoraConfig)
    for field in dataclasses.fields(LoraConfig):
        if not field.init or field.name.startswith("_") or field.name in config.PEFT_OWNED_FIELDS:
            continue
        # PEFT has both base-model bias policy and an adapter-bias switch.
        name = "adapter_bias" if field.name == "lora_bias" else field.name.removeprefix("lora_")
        flag = "--lora-" + name.replace("_", "-")
        _option(peft, flag, "peft." + field.name,
                hints.get(field.name, typing.Any), _field_help(field))

    extra = parser.add_argument_group("Metrics and method-specific inputs")
    _option(extra, "--ranges", "ranges", dict, 'Metric ranges as a TOML table, e.g. \'{loss=[0,5], eval_loss=[0,5]}\'.')
    if "preflight" in method.blocks:
        _option(extra, "--preflight-rows", "preflight.rows", int, "Train rows to score for off-policy warnings; positive integer.")
        _option(extra, "--offpolicy-logp-per-token", "preflight.offpolicy_logp_per_token", float, "Off-policy warning threshold.")
    if "rewards" in method.blocks:
        action = extra.add_argument(
            "--reward", action="append", metavar="NAME_OR_TABLE",
            help="Required unless configured. Repeat for multiple rewards; replaces the configured list. "
                 "Use a name, model ID, module:function, or a TOML factory table. "
                 "Example: --reward '{name=\"reference_match\",args={column=\"answer\",mode=\"equals\"}}'. "
                 "A running TRL-compatible vLLM weight-transfer server is also required; set --vllm-server-base-url.",
        )
        action.setting = Setting("--reward", "rewards.funcs", str | dict, action.help)
    if "replay" in method.blocks:
        _option(extra, "--replay-dataset", "replay.dataset", str, "Replay source with the same columns as training data.")
        _option(extra, "--replay-fraction", "replay.fraction", float, "Replay share of mixed training rows, strictly between 0 and 1.")
        _option(extra, "--replay-kl-coef", "replay.kl_coef", float, "Nonnegative KL coefficient; zero is plain mixing.")
        extra.add_argument("--no-replay", action="store_true", help="Disable configured replay for this run.")

    assessment = parser.add_argument_group("Advisory assessment and built-in independent quality checks")
    fields = [
        ("--quality-checks", "quality_checks", bool,
         "Run built-in quality checks at baseline, evaluation points, and completion, even with evaluation disabled; advisory only."),
        ("--assessment-window", "runtime_window", int, "Logged observations per runtime comparison window; at least 2."),
        ("--assessment-min-evaluations", "runtime_min_evaluations", int, "Comparable evaluation observations needed for trend advice; at least 2."),
        ("--assessment-relative-change", "runtime_relative_change", float, "Positive relative-change sensitivity; a heuristic threshold, not statistical confidence."),
        ("--quality-preset", "quality_preset", str | None,
         "Built-in preset: language_modeling, qa, classification, multiple_choice, json, preference, instruction_following, or writing."),
        ("--quality-dataset", "quality_dataset", str | None, "Separate evaluation dataset; required when quality checks are enabled."),
        ("--quality-max-length", "quality_max_length", int, "Quality input/window token limit; at least 2. LM windows overlap by one token; generation prompts are not silently truncated."),
        ("--quality-max-new-tokens", "quality_max_new_tokens", int, "Positive generation token budget per quality example."),
        ("--quality-batch-size", "quality_batch_size", int, "Positive quality generation batch size; LM/preference rows are scored individually."),
    ]
    for flag, key, hint, description in fields:
        _option(assessment, flag, "assessment." + key, hint, description)
    for key, hint, description in (
        ("url", str | None, "OpenAI-compatible judge API base; judging presets only."),
        ("model", str | None, "Served judge model name; judging presets only."),
        ("api_key", str | None, "Judge credential environment-variable name, or None for no authentication; never a literal key."),
        ("timeout", float, "Positive timeout seconds per judge request."),
        ("retries", int, "Nonnegative retry count for transient judge request failures."),
        ("max_tokens", int, "Positive judge response token budget."),
    ):
        _option(assessment, "--quality-judge-" + key.replace("_", "-"), "assessment.judge." + key, hint, description)


# Namespace -> explicit overrides; parser defaults are deliberately excluded.
def overrides(args):
    from trlx import TrlxError

    result = {key.removeprefix("override:"): value for key, value in vars(args).items() if key.startswith("override:")}
    if getattr(args, "no_lora", False):
        if any(key.startswith("peft.") for key in result):
            raise TrlxError("--no-lora cannot be combined with --lora-* settings")
        result["peft"] = None
    if getattr(args, "no_replay", False):
        if any(key.startswith("replay.") for key in result):
            raise TrlxError("--no-replay cannot be combined with --replay-* settings")
        result["replay"] = None
    if getattr(args, "reward", None):
        entries = []
        for raw in args.reward:
            try:
                entry = tomllib.loads("value = " + raw)["value"] if raw.lstrip().startswith("{") else raw
            except tomllib.TOMLDecodeError as error:
                raise TrlxError(f"--reward: invalid TOML factory: {error}") from error
            entries.append(entry)
        result["rewards.funcs"] = entries
    return result
