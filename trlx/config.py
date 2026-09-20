"""Run config loader: TOML file -> RunConfig.

Top-level keys map onto the method's TRL config dataclass; the named blocks
are trlx's own. Every rule in SPEC.md 2.2 is enforced here, at load time, so
train, check, and init round-trips all see one validated object. Nothing in
this module touches a model, a dataset file, or the network; those are
preflight and data_load concerns.

Errors are TrlxError with the config path and the key involved.
"""

import copy
import dataclasses
import enum
import pathlib
import re
import tomllib
import types
import typing

import torch
from peft import LoraConfig, PeftConfig

from dataset.io import FORMATS
from trlx import TrlxError, ranges, trainers

# TRL fields whose only effect is when the trainer loads the model from a path
# itself. trlx loads the model from [model] and [teacher] and passes objects,
# so a value here would be dead config. Rejected, naming the owning block.
MODEL_LOADING_FIELDS = {
    "model_init_kwargs": "model",
    "trust_remote_code": "model",
    "teacher_model_name_or_path": "teacher",
    "teacher_model_revision": "teacher",
    "teacher_model_init_kwargs": "teacher",
}

# The sharding strategy is trlx's decision (SPEC 2.5: automatic, or
# --strategy). An operator value for the field it is set through would fight
# that choice, so it is rejected. fsdp_config stays accepted: it tunes the
# sharding trlx chose.
STRATEGY_FIELDS = {"fsdp"}

# grpo and rloo generate through the TRL vLLM server, always (SPEC 2.10), so
# these two are forced and an operator value is rejected. The server's
# address (vllm_server_host, vllm_server_port, vllm_server_base_url) stays
# operator config.
VLLM_FORCED = {"use_vllm": True, "vllm_mode": "server"}

# The replay KL term needs the training model's logits, and TRL's default
# SFT loss ("chunked_nll") never materialises them (SPEC 5). So [replay] with
# kl_coef > 0 forces the one loss type that does, and an operator value is
# rejected, the same pattern as VLLM_FORCED.
REPLAY_KL_FORCED = {"loss_type": "nll"}

MODEL_KEYS = ("path", "dtype", "trust_remote_code", "attn_implementation")

# PeftConfig-level fields are set by trlx (task_type) or by peft on save; an
# operator value in [peft] would be overridden or meaningless.
PEFT_OWNED_FIELDS = frozenset(f.name for f in dataclasses.fields(PeftConfig))

# HF dataset id: org/name with an optional :split.
_HF_ID = re.compile(r"^([\w.-]+/[\w.-]+)(?::([\w.-]+))?$")


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    path: str
    dtype: str
    # None means the key was absent and transformers' own default applies.
    trust_remote_code: bool | None
    attn_implementation: str | None


# One dataset reference. A file is any path with a dataset.io extension; anything
# else must parse as an HF id. `split` is only ever set for hub ids.
@dataclasses.dataclass(frozen=True)
class DatasetRef:
    source: str
    is_file: bool
    split: str | None


# Mirrors the [dataset] block. A split's evaluation share is resolved against
# the actual row count by data_load, never frozen to a count during init.
@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    split: bool
    source: DatasetRef
    eval_fraction: float | None
    eval_source: DatasetRef | None

    @property
    def eval_enabled(self):
        return self.split or self.eval_source is not None


# [preflight] block. `rows` is how many train rows the off-policy check scores
# from the head of the set; a required key, because the check is a forward
# pass per response and its cost is the operator's to bound (PRINCIPLES:
# no runtime defaults).
@dataclasses.dataclass(frozen=True)
class PreflightSpec:
    offpolicy_logp_per_token: float
    rows: int


# One [rewards].funcs entry. `spec` is the string form (bare name, model path,
# or module:function); `args` is set only for the {name, args} table form.
@dataclasses.dataclass(frozen=True)
class RewardEntry:
    spec: str
    args: dict | None


@dataclasses.dataclass(frozen=True)
class ReplaySpec:
    dataset: DatasetRef
    fraction: float
    kl_coef: float


@dataclasses.dataclass(frozen=True)
class RunConfig:
    method: trainers.Method
    # The instantiated TRL config, with trlx's defaults for run_name and
    # save_steps already applied.
    args: object
    model: ModelSpec
    teacher: ModelSpec | None
    dataset: DatasetSpec
    peft: LoraConfig | None
    # metric name -> (low, high). Also the display column list, in file order.
    ranges: dict
    preflight: PreflightSpec | None
    rewards: list | None
    replay: ReplaySpec | None
    verify_prompts: DatasetRef | None
    # Selected method plus explicit CLI overrides, before launch-only fields.
    # Workers, resume comparison, and the run snapshot share these inputs.
    document: dict


# Loads and validates a run config for `method_name`. `fsdp` is the value for
# the TRL config's fsdp field when the launcher chose sharding; it goes in
# through the constructor because transformers configures FSDP in
# __post_init__, not on attribute assignment.
def load(path, method_name, fsdp=None, overrides=None, resolved=False):
    doc = resolve(path, method_name, overrides, resolved)
    return from_document(doc, method_name, fsdp=fsdp, path=path)


# Instantiate once the supervisor has applied the resolved GPU visibility.
def from_document(doc, method_name, fsdp=None, path="run.toml"):
    method = trainers.get(method_name)
    run_settings(doc, path)

    allowed_blocks = trainers.UNIVERSAL_BLOCKS | method.blocks
    all_blocks = trainers.UNIVERSAL_BLOCKS.union(*(m.blocks for m in trainers.METHODS.values()))
    blocks, top = {}, {}
    for key, value in doc.items():
        if key == "run":
            continue
        if key in all_blocks:
            if key not in allowed_blocks:
                owners = sorted(m.name for m in trainers.METHODS.values() if key in m.blocks)
                raise TrlxError(f"{path}: block [{key}] applies to {', '.join(owners)}, not {method.name}")
            if not isinstance(value, dict):
                raise TrlxError(f"{path}: [{key}] must be a table")
            blocks[key] = value
        else:
            top[key] = value

    for required in ("model", "dataset", "ranges"):
        if required not in blocks:
            raise TrlxError(f"{path}: missing required block [{required}]")
    # Blocks a method cannot run without: the trainer needs the teacher and
    # the reward functions as constructor arguments.
    for needed in ("teacher", "rewards"):
        if needed in method.blocks and needed not in blocks:
            raise TrlxError(f"{path}: {method.name} requires a [{needed}] block")

    dataset = _dataset(path, blocks["dataset"])
    # Parsed before the TRL config: the KL term decides a forced field there.
    replay = _replay(path, blocks["replay"]) if "replay" in blocks else None
    args = _build_args(path, method, top, dataset.eval_enabled, fsdp, replay)
    return RunConfig(
        method=method,
        args=args,
        model=model_spec(path, "model", blocks["model"]),
        teacher=model_spec(path, "teacher", blocks["teacher"]) if "teacher" in blocks else None,
        dataset=dataset,
        peft=_peft(path, method, blocks["peft"]) if "peft" in blocks else None,
        ranges=ranges.parse(path, blocks["ranges"]),
        preflight=_preflight(path, blocks["preflight"]) if "preflight" in blocks else None,
        rewards=_rewards(path, blocks["rewards"]) if "rewards" in blocks else None,
        replay=replay,
        verify_prompts=_verify(path, blocks["verify"]) if "verify" in blocks else None,
        document=doc,
    )


# Launch controls were CLI defaults before [run] became persistently editable.
# Flat per-run configs may omit the block; when present all four keys are required.
def run_settings(doc, path="run.toml"):
    table = doc.get("run", {"gpus": "all", "strategy": "auto", "tui": False, "verify": True})
    if not isinstance(table, dict):
        raise TrlxError(f"{path}: [run] must be a table")
    _check_keys(path, "[run]", table, ("gpus", "strategy", "tui", "verify"))
    gpus = _require(path, "[run]", table, "gpus", str)
    strategy = _require(path, "[run]", table, "strategy", str)
    if not gpus.strip():
        raise TrlxError(f"{path}: [run].gpus must be all or comma-separated visible device indices")
    if strategy not in ("auto", "ddp", "fsdp"):
        raise TrlxError(f"{path}: [run].strategy must be auto, ddp, or fsdp")
    return {"gpus": gpus, "strategy": strategy,
            "tui": _require(path, "[run]", table, "tui", bool),
            "verify": _require(path, "[run]", table, "verify", bool)}


# Merge nested method settings without mutating the operator's document.
def _merge(shared, selected):
    result = copy.deepcopy(shared)
    for key, value in selected.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Resolve persistent settings, then apply only explicitly supplied CLI values.
# A worker reads the already resolved snapshot; it never reopens the source config.
def resolve(path, method_name, overrides=None, resolved=False):
    doc = copy.deepcopy(_read_toml(path))
    if resolved:
        launch = doc.pop("launch", None)
        if not isinstance(launch, dict) or launch.get("method") != method_name:
            raise TrlxError(f"{path}: resolved snapshot does not describe method '{method_name}'")
        if overrides:
            raise TrlxError(f"{path}: workers cannot override a resolved snapshot")
        return doc
    methods = doc.pop("methods", {})
    if not isinstance(methods, dict):
        raise TrlxError(f"{path}: [methods] must be a table")
    unknown = sorted(set(methods) - set(trainers.METHODS))
    if unknown:
        raise TrlxError(f"{path}: unknown methods: {', '.join(unknown)}")
    for name, settings in methods.items():
        if not isinstance(settings, dict):
            raise TrlxError(f"{path}: [methods.{name}] must be a table")
        if "methods" in settings or "launch" in settings:
            raise TrlxError(f"{path}: [methods.{name}] cannot contain methods or launch")
    doc = _merge(doc, methods.get(method_name, {}))
    overrides = dict(overrides or {})
    source = overrides.pop("dataset.source", None)
    if source is not None and any(key in overrides for key in ("dataset.dataset", "dataset.dataset_train")):
        raise TrlxError(f"{path}: --dataset cannot be combined with another explicit training source")
    if "run" not in doc and any(key.startswith("run.") for key in overrides):
        doc["run"] = run_settings(doc, path)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        table = doc
        for part in parts[:-1]:
            table = table.setdefault(part, {})
            if not isinstance(table, dict):
                raise TrlxError(f"{path}: cannot override '{dotted}': parent is not a table")
        if value is None and dotted in ("peft", "replay"):
            table.pop(parts[-1], None)
        else:
            table[parts[-1]] = value
    # Changing split mode explicitly replaces its incompatible source keys.
    # A CLI source is applied afterwards, in the selected mode's vocabulary.
    dataset = doc.get("dataset", {})
    if "dataset.split" in overrides and isinstance(dataset, dict):
        incompatible = ("dataset_train", "dataset_eval") if dataset["split"] else ("dataset", "eval_fraction")
        conflicts = [key for key in incompatible if "dataset." + key in overrides]
        if conflicts:
            raise TrlxError(f"{path}: explicit dataset split mode conflicts with {', '.join(conflicts)}")
        if dataset["split"]:
            previous = dataset.pop("dataset_train", None)
            dataset.pop("dataset_eval", None)
            if previous is not None:
                dataset.setdefault("dataset", previous)
        else:
            previous = dataset.pop("dataset", None)
            dataset.pop("eval_fraction", None)
            if previous is not None:
                dataset.setdefault("dataset_train", previous)
    if source is not None:
        dataset = doc.setdefault("dataset", {})
        if not isinstance(dataset, dict):
            raise TrlxError(f"{path}: [dataset] must be a table")
        key = "dataset" if dataset.get("split") else "dataset_train"
        dataset[key] = source
    if overrides.get("dataset.split") is False and not dataset.get("dataset_eval"):
        # --no-split without a separate eval source explicitly requests training
        # only. Remove the generated eval schedule, but reject contradictory CLI input.
        conflicts = [key for key in overrides if key.startswith("eval_") and
                     not (key == "eval_strategy" and overrides[key] == "no")]
        if conflicts:
            raise TrlxError(f"{path}: evaluation is disabled but CLI sets {', '.join(conflicts)}")
        for key in list(doc):
            if key.startswith("eval_"):
                del doc[key]
    return doc


# Read the operator config or a resolved snapshot; callers own interpretation.
def _read_toml(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise TrlxError(f"{path}: no such file")
    except OSError as e:
        raise TrlxError(f"{path}: cannot read: {e.strerror or e}")
    except tomllib.TOMLDecodeError as e:
        raise TrlxError(f"{path}: invalid TOML: {e}")


# True when a type hint admits None: Optional, X | None, or Any.
def admits_none(hint):
    if hint is typing.Any or hint is type(None):
        return True
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return any(admits_none(a) for a in typing.get_args(hint))
    return False


# Maps top-level keys onto the method's TRL config class and instantiates it.
# Order matters: the "None" rule and rejections run on the raw keys, trlx's
# defaults are filled in, then the dataclass's own __post_init__ validates.
# `replay` is the parsed [replay] block or None; its KL term forces a field.
def _build_args(path, method, top, eval_enabled, fsdp, replay):
    cls = method.config_cls
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    kl_forced = replay is not None and replay.kl_coef > 0
    kwargs = {}
    for key, value in top.items():
        if key in MODEL_LOADING_FIELDS:
            raise TrlxError(
                f"{path}: '{key}' is not accepted at top level; the model is described by the "
                f"[{MODEL_LOADING_FIELDS[key]}] block"
            )
        if key in STRATEGY_FIELDS:
            raise TrlxError(f"{path}: '{key}' is not accepted at top level; the strategy is chosen by trlx or --strategy")
        if key in VLLM_FORCED and "rewards" in method.blocks:
            raise TrlxError(f"{path}: '{key}' is not accepted at top level; {method.name} always uses the TRL vLLM server")
        if key in REPLAY_KL_FORCED and kl_forced:
            raise TrlxError(
                f"{path}: '{key}' is not accepted at top level when [replay].kl_coef > 0; the KL term needs "
                f"logits, so trlx sets {key} = {REPLAY_KL_FORCED[key]!r}"
            )
        if key not in fields:
            raise TrlxError(f"{path}: unknown key '{key}'; not a field of {cls.__name__} or a trlx block")
        # TOML has no null; the string "None" stands in, only where the field's
        # type admits None.
        if value == "None":
            if not admits_none(hints[key]):
                raise TrlxError(f"{path}: '{key}' does not accept \"None\" (type {_type_name(hints[key])})")
            value = None
        elif not _matches(value, hints[key]):
            raise TrlxError(
                f"{path}: '{key}' must be {_type_name(hints[key])}, got {type(value).__name__} {value!r}"
            )
        kwargs[key] = value

    if "output_dir" not in kwargs or kwargs["output_dir"] is None:
        raise TrlxError(f"{path}: 'output_dir' is required")
    if not eval_enabled:
        eval_keys = sorted(k for k in kwargs if k.startswith("eval_"))
        if eval_keys:
            raise TrlxError(
                f"{path}: evaluation is disabled (no eval dataset) but eval keys are set: {', '.join(eval_keys)}"
            )
    # trlx defaults from SPEC 2.2. These are the only values filled in here;
    # every other absent key takes the dataclass default.
    if kwargs.get("run_name") is None:
        kwargs["run_name"] = pathlib.Path(kwargs["output_dir"]).name
    if fsdp is not None:
        kwargs["fsdp"] = fsdp
    if "rewards" in method.blocks:
        kwargs.update(VLLM_FORCED)
    if kl_forced:
        kwargs.update(REPLAY_KL_FORCED)
    save_steps_given = "save_steps" in kwargs

    try:
        args = cls(**kwargs)
    except (ValueError, TypeError) as e:
        # TRL and transformers validate field combinations in __post_init__.
        raise TrlxError(f"{path}: {cls.__name__} rejected the config: {e}")
    # The checkpoint interval follows the eval interval. Applied after
    # __post_init__ because that is where transformers resolves eval_steps
    # (absent eval_steps with eval_strategy = "steps" becomes logging_steps).
    if not save_steps_given and args.eval_strategy == "steps":
        args.save_steps = args.eval_steps
    return args


# Python types a TOML value may have for a hint. Unions are flattened; an int
# is accepted for float; a str enum (SchedulerType, OptimizerNames) accepts
# str. An empty result means the hint is not a TOML-expressible shape trlx
# recognises (Any, a class, a forward ref), and the value is passed through
# unchecked for the dataclass to judge.
def accepted_types(hint):
    if hint is typing.Any:
        return set()
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        accepted = set()
        for arg in typing.get_args(hint):
            accepted |= accepted_types(arg)
        return accepted
    if origin is typing.Literal:
        return {type(a) for a in typing.get_args(hint)}
    base = origin or hint
    if base is bool:
        return {bool}
    if base is float:
        return {int, float}
    if base in (int, str, list, dict, tuple):
        return {base}
    if isinstance(base, type) and issubclass(base, enum.Enum):
        return {str}
    return set()


# Type check of a TOML value against a hint. TOML bools are Python bools,
# which are ints, so a bool only matches when bool is itself accepted.
def _matches(value, hint):
    accepted = accepted_types(hint)
    if not accepted:
        return True
    if isinstance(value, bool):
        return bool in accepted
    return isinstance(value, tuple(accepted))


# Short spelling of a type hint for error messages: "float", "int | None".
def _type_name(hint):
    if isinstance(hint, type):
        return hint.__name__
    return str(hint).replace("typing.", "").replace("transformers.", "").replace("trainer_utils.", "")


# Block helpers. `where` is the display label of the table being checked,
# e.g. "[model]" or "[rewards].funcs[2]", used verbatim in messages.


# Rejects keys outside `allowed`.
def _check_keys(path, where, table, allowed):
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise TrlxError(f"{path}: {where} has unknown keys: {', '.join(unknown)}; allowed: {', '.join(allowed)}")


def _require(path, where, table, key, kind):
    if key not in table:
        raise TrlxError(f"{path}: {where} requires '{key}'")
    return _typed(path, where, table, key, kind)


# Type check for a block value. `kind` is a type or tuple of types as for
# isinstance. bool is an int subclass in Python, so a bool only passes when
# bool itself is the kind asked for; `train = true` must not count as an int.
def _typed(path, where, table, key, kind):
    value = table[key]
    bool_where_not_wanted = isinstance(value, bool) and kind is not bool
    if bool_where_not_wanted or not isinstance(value, kind):
        want = "a number" if isinstance(kind, tuple) else kind.__name__
        raise TrlxError(f"{path}: {where}.{key} must be {want}, got {type(value).__name__}")
    return value


# Shared by [model] and [teacher]; `block` is the block name.
def model_spec(path, block, table):
    where = f"[{block}]"
    _check_keys(path, where, table, MODEL_KEYS)
    if "path" not in table:
        raise TrlxError(f"{path}: {where}.path is required; supply --{block} or save its path in the config")
    dtype = _require(path, where, table, "dtype", str)
    # The dtype is used as getattr(torch, dtype) at load time; check it now so
    # the error names the config key rather than surfacing from transformers.
    if not isinstance(getattr(torch, dtype, None), torch.dtype):
        raise TrlxError(f"{path}: {where}.dtype '{dtype}' is not a torch dtype name (e.g. bfloat16, float16, float32)")
    return ModelSpec(
        path=_require(path, where, table, "path", str),
        dtype=dtype,
        trust_remote_code=_typed(path, where, table, "trust_remote_code", bool) if "trust_remote_code" in table else None,
        attn_implementation=_typed(path, where, table, "attn_implementation", str) if "attn_implementation" in table else None,
    )


# A dataset value is a file when its extension is one dataset.io reads; the
# extension is the only signal, so an HF id can never be mistaken for a file.
def dataset_ref(path, where, value):
    if not isinstance(value, str):
        raise TrlxError(f"{path}: {where} must be a string, got {type(value).__name__}")
    if pathlib.Path(value).suffix.lower() in FORMATS:
        return DatasetRef(source=value, is_file=True, split=None)
    m = _HF_ID.match(value)
    if not m:
        known = ", ".join(sorted(FORMATS))
        raise TrlxError(f"{path}: {where} '{value}' is neither a dataset file ({known}) nor an HF id org/name[:split]")
    return DatasetRef(source=m.group(1), is_file=False, split=m.group(2))


# Validate the split contract without reading data or guessing its size.
def _dataset(path, table):
    block = "[dataset]"
    if "train" in table:
        raise TrlxError(f"{path}: [dataset].train is no longer supported; use eval_fraction (for example 0.1)")
    _check_keys(path, block, table, ("split", "dataset", "eval_fraction", "dataset_train", "dataset_eval"))
    split = _require(path, block, table, "split", bool)
    # The two forms are exclusive; a key from the other form is an error rather
    # than silently ignored.
    if split:
        for wrong in ("dataset_train", "dataset_eval"):
            if wrong in table:
                raise TrlxError(f"{path}: [dataset] split = true uses 'dataset' and 'eval_fraction'; '{wrong}' is for split = false")
        fraction = _require(path, block, table, "eval_fraction", (int, float))
        if not 0 < fraction < 1:
            raise TrlxError(f"{path}: [dataset].eval_fraction must be between 0 and 1, exclusive, got {fraction}")
        if "dataset" not in table:
            raise TrlxError(f"{path}: [dataset] requires 'dataset'; supply --dataset or save the source in the config")
        return DatasetSpec(True, dataset_ref(path, "[dataset].dataset", table["dataset"]), float(fraction), None)
    for wrong in ("dataset", "eval_fraction"):
        if wrong in table:
            raise TrlxError(f"{path}: [dataset] split = false uses 'dataset_train' and 'dataset_eval'; '{wrong}' is for split = true")
    if "dataset_train" not in table:
        raise TrlxError(f"{path}: [dataset] requires 'dataset_train'; supply --dataset or --dataset-train")
    eval_ref = dataset_ref(path, "[dataset].dataset_eval", table["dataset_eval"]) if "dataset_eval" in table else None
    return DatasetSpec(False, dataset_ref(path, "[dataset].dataset_train", table["dataset_train"]), None, eval_ref)


# PEFT validates field combinations after nullable CLI/config values are decoded.
# trlx owns task_type and translates failures into a message naming the config.
def _peft(path, method, table):
    owned = sorted(set(table) & PEFT_OWNED_FIELDS)
    if owned:
        raise TrlxError(f"{path}: [peft] keys {', '.join(owned)} are set by trlx, not the config")
    hints = typing.get_type_hints(LoraConfig)
    values = {key: None if value == "None" and admits_none(hints.get(key)) else value
              for key, value in table.items()}
    try:
        return LoraConfig(task_type=method.peft_task_type, **values)
    except TypeError as e:
        # dataclass __init__ names the offending keyword in its message.
        raise TrlxError(f"{path}: [peft] {e}")
    except ValueError as e:
        raise TrlxError(f"{path}: [peft] rejected by peft: {e}")


# [preflight]: both keys required; the block itself is optional (SPEC 2.2).
def _preflight(path, table):
    _check_keys(path, "[preflight]", table, ("offpolicy_logp_per_token", "rows"))
    value = _require(path, "[preflight]", table, "offpolicy_logp_per_token", (int, float))
    rows = _require(path, "[preflight]", table, "rows", int)
    if rows <= 0:
        raise TrlxError(f"{path}: [preflight].rows must be a positive row count, got {rows}")
    return PreflightSpec(float(value), rows)


# Shape check only: each entry is a string or a {name, args} table. What a
# string means (built-in, trl.rewards, model path, module:function) is
# resolved by rewards.py, which can report it against the real registries.
def _rewards(path, table):
    _check_keys(path, "[rewards]", table, ("funcs",))
    if "funcs" not in table:
        raise TrlxError(f"{path}: [rewards].funcs is required; supply --reward or configure the reward objective")
    funcs = _require(path, "[rewards]", table, "funcs", list)
    if not funcs:
        raise TrlxError(f"{path}: [rewards].funcs must list at least one reward")
    entries = []
    for i, entry in enumerate(funcs):
        if isinstance(entry, str):
            entries.append(RewardEntry(entry, None))
        elif isinstance(entry, dict):
            where = f"[rewards].funcs[{i}]"
            _check_keys(path, where, entry, ("name", "args"))
            name = _require(path, where, entry, "name", str)
            args = _require(path, where, entry, "args", dict)
            entries.append(RewardEntry(name, args))
        else:
            raise TrlxError(f"{path}: [rewards].funcs[{i}] must be a string or {{name, args}}, got {type(entry).__name__}")
    return entries


# [replay]: `fraction` is the replay share of the mixed training set (SPEC
# 2.9), so 1 has no finite row count and the interval is open at both ends.
def _replay(path, table):
    _check_keys(path, "[replay]", table, ("dataset", "fraction", "kl_coef"))
    if "dataset" not in table:
        raise TrlxError(f"{path}: [replay] requires 'dataset'")
    fraction = float(_require(path, "[replay]", table, "fraction", (int, float)))
    if not 0 < fraction < 1:
        raise TrlxError(f"{path}: [replay].fraction must be in (0, 1), got {fraction}")
    kl_coef = float(_require(path, "[replay]", table, "kl_coef", (int, float)))
    if kl_coef < 0:
        raise TrlxError(f"{path}: [replay].kl_coef must be >= 0, got {kl_coef}")
    return ReplaySpec(dataset_ref(path, "[replay].dataset", table["dataset"]), fraction, kl_coef)


def _verify(path, table):
    _check_keys(path, "[verify]", table, ("prompts",))
    if "prompts" not in table:
        return None
    return dataset_ref(path, "[verify].prompts", table["prompts"])
