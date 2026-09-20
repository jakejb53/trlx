"""Typed, discoverable CLI overrides; absent options never replace config values."""

import argparse
import dataclasses
import enum
import functools
import textwrap
import tomllib
import types
import typing


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
def _option(group, flag, key, hint, help_text):
    # --dataset addresses the primary source in either split mode, not a TOML alias.
    setting = "dataset.dataset or dataset.dataset_train, according to split" if key == "dataset.source" else key
    text = f"{help_text} Config: {setting}; omitted CLI options retain the config value."
    kwargs = {"dest": "override:" + key, "default": argparse.SUPPRESS, "help": text.replace("%", "%%")}
    non_null = [item for item in _alternatives(hint) if item is not type(None)]
    if non_null == [bool] and type(None) in _alternatives(hint):
        group.add_argument(flag, nargs="?", const=True, type=_boolean_or_none,
                           metavar="{true,false,None}", **kwargs)
        negative = dict(kwargs, help=f"Set {key} to false for this run.")
        group.add_argument("--no-" + flag[2:], action="store_false", **negative)
        return
    if non_null == [bool]:
        kwargs["action"] = argparse.BooleanOptionalAction
    else:
        kwargs["type"] = functools.partial(parse_value, hint=hint)
        kwargs["metavar"] = "VALUE"
        kwargs["help"] += f" Type: {type_label(hint)}."
    group.add_argument(flag, **kwargs)


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
    for field in dataclasses.fields(method.config_cls):
        if not field.init or field.name.startswith("_") or field.name in owned:
            continue
        group = common if field.name in shared_names else specific
        _option(group, "--" + field.name.replace("_", "-"), field.name,
                hints.get(field.name, typing.Any), _field_help(field))

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
        extra.add_argument(
            "--reward", action="append", metavar="NAME_OR_TABLE",
            help="Required unless configured. Repeat for multiple rewards; replaces the configured list. "
                 "Use a name, model ID, module:function, or a TOML factory table. "
                 "Example: --reward '{name=\"reference_match\",args={column=\"answer\",mode=\"equals\"}}'. "
                 "A running TRL-compatible vLLM weight-transfer server is also required; set --vllm-server-base-url.",
        )
    if "replay" in method.blocks:
        _option(extra, "--replay-dataset", "replay.dataset", str, "Replay source with the same columns as training data.")
        _option(extra, "--replay-fraction", "replay.fraction", float, "Replay share of mixed training rows, strictly between 0 and 1.")
        _option(extra, "--replay-kl-coef", "replay.kl_coef", float, "Nonnegative KL coefficient; zero is plain mixing.")
        extra.add_argument("--no-replay", action="store_true", help="Disable configured replay for this run.")


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
