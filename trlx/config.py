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

from dataset.io import FORMATS, DatasetError
from dataset.prompts import load as load_prompt
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

# Baseline measurement is part of trlx's training lifecycle, not an operator toggle.
BASELINE_FIELDS = {"eval_on_start"}

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
    synthetic_dataset_eval: bool = False
    shuffle_eval_data: bool = False
    include_reasoning: bool = False
    reasoning_only_loss: bool = False

    # Synthetic rows are generated after model placement, but evaluation is enabled now.
    @property
    def eval_enabled(self):
        return self.synthetic_dataset_eval or self.split or self.eval_source is not None


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
    rubric_text: str | None = None


@dataclasses.dataclass(frozen=True)
class ReplaySpec:
    dataset: DatasetRef
    fraction: float
    kl_coef: float


# Independent quality checks have explicit settings; metric interpretation is owned by the assessor.
@dataclasses.dataclass(frozen=True)
class AssessmentSpec:
    quality_checks: bool
    quality_preset: str | None
    quality_dataset: DatasetRef | None
    quality_max_length: int
    quality_max_new_tokens: int
    quality_batch_size: int
    judge: dict | None
    prompts: dict = dataclasses.field(default_factory=dict)


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
    # Selected method plus explicit CLI overrides, before launch-only fields.
    # Workers, resume comparison, and the run snapshot share these inputs.
    document: dict
    # Library-only config inspection can omit the block; training/check enforce it before startup work.
    assessment: AssessmentSpec | None = None
    # Loaded once before review; publication uses these contents, never a second source read.
    prompts: dict = dataclasses.field(default_factory=dict)


# Loads and validates a run config for `method_name`. `fsdp` is the value for
# the TRL config's fsdp field when the launcher chose sharding; it goes in
# through the constructor because transformers configures FSDP in
# __post_init__, not on attribute assignment.
def load(path, method_name, fsdp=None, overrides=None, resolved=False):
    doc = resolve(path, method_name, overrides, resolved)
    return from_document(doc, method_name, fsdp=fsdp, path=path if resolved else source_path(doc, path))


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
                raise TrlxError(f"{path}: block [{key}] applies to {', '.join(owners)}, not {method.name}. "
                                f"Remove it from this method's settings, or move it under the appropriate "
                                f"[methods.NAME.{key}] section so it is not applied to every method.")
            if not isinstance(value, dict):
                raise TrlxError(f"{path}: [{key}] must be a TOML table; put its settings below a [{key}] heading, "
                                f"rather than assigning {key} a scalar value.")
            blocks[key] = value
        else:
            top[key] = value

    for required in ("model", "dataset", "ranges"):
        if required not in blocks:
            raise TrlxError(f"{path}: missing required block [{required}]. " + {
                "model": "Supply --model MODEL --dtype DTYPE (for example float32), or add [model] with path and dtype.",
                "dataset": "For one file, use --dataset FILE --split --eval-fraction FRACTION (0 < FRACTION < 1). "
                           "For already-separated data, use --dataset-train TRAIN and optionally --dataset-eval EVAL.",
                "ranges": "Supply --ranges '{loss=[0,5]}' (example bounds; choose bounds for your metrics), "
                          "or add the metric bounds in a [ranges] table.",
            }[required])
    if "verify" in blocks:
        _verify(path, blocks["verify"])
    # Blocks a method cannot run without: the trainer needs the teacher and
    # the reward functions as constructor arguments.
    for needed in ("teacher", "rewards"):
        if needed in method.blocks and needed not in blocks:
            raise TrlxError(f"{path}: {method.name} requires a [{needed}] block; "
                            + ("supply --teacher MODEL --teacher-dtype DTYPE (for example float32), "
                               "or configure [teacher] with path and dtype."
                               if needed == "teacher" else
                               "supply at least one --reward entry or configure [rewards].funcs with the reward objective."))

    dataset = _dataset(path, blocks["dataset"])
    if dataset.include_reasoning and method_name != "sft":
        raise TrlxError(f"{path}: include_reasoning is supported only for sft. "
                        "Use trlx sft to train the supplied reasoning, or disable include_reasoning and "
                        "reasoning_only_loss in this method's [dataset] settings.")
    if dataset.include_reasoning and dataset.synthetic_dataset_eval:
        raise TrlxError(f"{path}: include_reasoning cannot be combined with synthetic_dataset_eval. "
                        "Reasoning uses messages rows; synthetic evaluation uses raw text rows. "
                        "For reasoning data, pass --no-synthetic-dataset-eval. For synthetic CPT evaluation, "
                        "use --no-reasoning-only-loss --no-include-reasoning with a text dataset.")
    if dataset.synthetic_dataset_eval and method_name != "sft":
        raise TrlxError(f"{path}: synthetic_dataset_eval is supported only for sft with CPT text data. "
                        "Use trlx sft for this feature, or disable synthetic_dataset_eval in [dataset] "
                        "and provide evaluation data supported by the selected method.")
    # Parsed before the TRL config: the KL term decides a forced field there.
    replay = _replay(path, blocks["replay"]) if "replay" in blocks else None
    args = _build_args(path, method, top, dataset.eval_enabled, fsdp, replay)
    if dataset.include_reasoning and (args.dataset_kwargs or {}).get("skip_prepare_dataset"):
        raise TrlxError(f"{path}: include_reasoning requires preparation of raw messages rows. "
                        "Set skip_prepare_dataset = false in dataset_kwargs so trlx can insert reasoning "
                        "and validate its loss mask; supply messages plus a nonempty reasoning string per row.")
    if dataset.synthetic_dataset_eval:
        if isinstance(args.max_length, bool) or not isinstance(args.max_length, int) or args.max_length < 1:
            raise TrlxError(f"{path}: --synthetic-dataset-eval requires a positive --max-length. "
                            "Supply --max-length N with integer N >= 1; this also limits generated summary tokens.")
        if args.dataset_text_field != "text" or args.assistant_only_loss or args.completion_only_loss:
            raise TrlxError(f"{path}: --synthetic-dataset-eval requires CPT text rows with full-sequence loss. "
                            "Use a text-column dataset with --dataset-text-field text --no-assistant-only-loss "
                            "--no-completion-only-loss. For conversational data, use --no-synthetic-dataset-eval "
                            "and supply ordinary evaluation data.")
        if (args.dataset_kwargs or {}).get("skip_prepare_dataset"):
            raise TrlxError(f"{path}: --synthetic-dataset-eval requires preparation of raw CPT text rows. "
                            "Set skip_prepare_dataset = false in dataset_kwargs and supply rows with a text column.")
    assessment = _assessment(path, blocks["assessment"], method_name) if "assessment" in blocks else None
    reward_entries = _rewards(path, blocks["rewards"]) if "rewards" in blocks else None
    prompt_texts = _prompts(path, blocks.get("prompts", {}), dataset, assessment)
    if assessment is not None:
        assessment = dataclasses.replace(assessment, prompts={
            key: text for key, text in prompt_texts.items() if key.startswith("quality_")
        })
    if reward_entries:
        for index, entry in enumerate(reward_entries):
            if entry.spec == "llm_judge":
                values = dict(entry.args or {})
                if "rubric" in values:
                    raise TrlxError(f"{path}: llm_judge rubric was removed; save it in a .prompt file and set rubric_file")
                name = _require(path, f"[rewards].funcs[{index}].args", values, "rubric_file", str)
                text = _read_prompt(path, name, allowed=None)
                prompt_texts[f"reward_{index}"] = text
                reward_entries[index] = dataclasses.replace(entry, rubric_text=text)
    return RunConfig(
        method=method,
        args=args,
        model=model_spec(path, "model", blocks["model"]),
        teacher=model_spec(path, "teacher", blocks["teacher"]) if "teacher" in blocks else None,
        dataset=dataset,
        peft=_peft(path, method, blocks["peft"]) if "peft" in blocks else None,
        ranges=ranges.parse(path, blocks["ranges"]),
        preflight=_preflight(path, blocks["preflight"]) if "preflight" in blocks else None,
        rewards=reward_entries,
        replay=replay,
        document=doc,
        assessment=assessment,
        prompts=prompt_texts,
    )


# Paths belong to the config that supplied them, including saved worker/resume configs.
def _read_prompt(config_path, name, required=(), allowed=()):
    if not name.strip():
        raise TrlxError(f"{config_path}: prompt path must not be empty")
    target = pathlib.Path(name)
    if not target.is_absolute():
        target = pathlib.Path(config_path).absolute().parent / target
    try:
        return load_prompt(target, required=required, allowed=allowed)
    except DatasetError as error:
        raise TrlxError(f"{config_path}: {error}") from error


# Inactive features require no files; active features have no implicit path or text defaults.
def _prompts(path, table, dataset, assessment):
    fields = {
        "synthetic_eval_summary": ("text",),
        "quality_qa": (), "quality_classification": ("labels",),
        "quality_multiple_choice": ("choices",), "quality_json": ("required_fields",),
        "quality_instruction_following_judge": (), "quality_writing_judge": (),
    }
    _check_keys(path, "[prompts]", table, fields)
    for key in table:
        _require(path, "[prompts]", table, key, str)
    active = []
    if dataset.synthetic_dataset_eval:
        active.append("synthetic_eval_summary")
    if assessment is not None and assessment.quality_checks:
        preset = assessment.quality_preset
        if preset in ("qa", "classification", "multiple_choice", "json"):
            active.append(f"quality_{preset}")
        elif preset in ("instruction_following", "writing"):
            active.append(f"quality_{preset}_judge")
    result = {}
    for key in active:
        name = _require(path, "[prompts]", table, key, str)
        result[key] = _read_prompt(path, name, fields[key], allowed=() if fields[key] else None)
    return result


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
        raise TrlxError(f"{path}: [run].gpus must be all or comma-separated visible device indices. "
                        "Use --gpus all or --gpus 0,1 for devices visible to this process; do not supply an empty value.")
    if strategy not in ("auto", "ddp", "fsdp"):
        raise TrlxError(f"{path}: [run].strategy must be auto, ddp, or fsdp. "
                        "Use --strategy auto to let trlx choose, or select --strategy ddp / --strategy fsdp "
                        "with multiple visible GPUs.")
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


# Resolve persistent settings or the selected checkpoint's snapshot, then CLI values.
# An explicit resume never reads today's run.toml; workers cannot resolve inputs again.
def resolve(path, method_name, overrides=None, resolved=False):
    overrides = dict(overrides or {})
    if resolved:
        if overrides:
            raise TrlxError(f"{path}: workers cannot override a resolved snapshot")
        return _snapshot(path, method_name)
    resume = overrides.get("resume_from_checkpoint")
    if resume not in (None, "None", ""):
        doc = {}
    else:
        doc = _method_document(path, method_name)
        resume = overrides.get("resume_from_checkpoint", doc.get("resume_from_checkpoint"))
    if resume not in (None, "None", ""):
        if not isinstance(resume, str):
            raise TrlxError(f"{path}: resume_from_checkpoint must be a checkpoint path, got {resume!r}. "
                            "Use --resume-from-checkpoint RUN/checkpoint-N with an existing checkpoint directory, "
                            "not a boolean or the run's parent directory. Use None to disable a configured resume.")
        checkpoint = pathlib.Path(resume).resolve()
        path = checkpoint.parent / "config.toml"
        doc = _snapshot(path, method_name)
        # output_dir in operator inputs is a parent; a snapshot names the actual run.
        parent = overrides.pop("output_dir", None)
        if parent is not None and pathlib.Path(parent).resolve() != checkpoint.parent.parent:
            raise TrlxError(
                f"--output-dir {parent}: resume continues in {checkpoint.parent}; "
                "omit --output-dir when using --resume-from-checkpoint"
            )
        doc["output_dir"] = str(checkpoint.parent)
        overrides["resume_from_checkpoint"] = str(checkpoint)
    return _apply_overrides(doc, overrides, path)


# The snapshot's method is mandatory; historical schemas receive no implicit conversion.
def _snapshot(path, method_name):
    doc = copy.deepcopy(_read_toml(path))
    launch = doc.pop("launch", None)
    if not isinstance(launch, dict) or launch.get("method") != method_name:
        raise TrlxError(f"{path}: resolved snapshot does not describe method '{method_name}'")
    return doc


# Diagnostics name the snapshot that supplied resume settings, not an unused source file.
def source_path(document, default):
    resume = document.get("resume_from_checkpoint")
    if resume not in (None, "None", ""):
        return pathlib.Path(resume).resolve().parent / "config.toml"
    return default


# Select method settings before looking for a resume configured in the operator's file.
def _method_document(path, method_name):
    doc = copy.deepcopy(_read_toml(path))
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
    return _merge(doc, methods.get(method_name, {}))


# One override path serves fresh runs and resumes; absent CLI values change nothing.
def _apply_overrides(doc, overrides, path):
    # An explicit narrower objective enables reasoning even over an inherited
    # disabled setting; an explicit CLI contradiction must never be hidden.
    if overrides.get("dataset.reasoning_only_loss") is True:
        if overrides.get("dataset.include_reasoning") is False:
            raise TrlxError(f"{path}: --reasoning-only-loss conflicts with --no-include-reasoning: "
                            "reasoning must be included to score its tokens. Remove --no-include-reasoning "
                            "to train reasoning only, or use --no-reasoning-only-loss to train without that objective.")
        overrides["dataset.include_reasoning"] = True
    source = overrides.pop("dataset.source", None)
    if source is not None and any(key in overrides for key in ("dataset.dataset", "dataset.dataset_train")):
        raise TrlxError(f"{path}: --dataset cannot be combined with another explicit training source. "
                        "For one file to split, use --dataset FILE --split --eval-fraction FRACTION "
                        "(0 < FRACTION < 1) and remove --dataset-train. For already-separated files, "
                        "use --dataset-train TRAIN --dataset-eval EVAL and remove --dataset.")
    separate_sources = ["--" + key.replace("_", "-") for key in ("dataset_train", "dataset_eval")
                        if "dataset." + key in overrides]
    if separate_sources:
        if overrides.get("dataset.split") is True:
            raise TrlxError(f"{path}: --split conflicts with {', '.join(separate_sources)}; "
                            "--dataset-train and --dataset-eval declare already-separated sources and disable splitting. "
                            "To split one file, replace --dataset-train with --dataset, remove --dataset-eval if present, "
                            "and keep --split with --eval-fraction FRACTION (0 < FRACTION < 1). "
                            "To use separate files instead, remove --split, --eval-fraction, and --shuffle-eval-data.")
        # Explicit file roles select the same mode and cleanup as --no-split.
        overrides["dataset.split"] = False
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
    if isinstance(dataset, dict) and dataset.get("synthetic_dataset_eval") is True:
        conflicts = [key for key in ("dataset.dataset_eval", "dataset.eval_fraction") if key in overrides]
        if overrides.get("dataset.split") is True:
            conflicts.append("dataset.split")
        if conflicts:
            raise TrlxError(f"{path}: --synthetic-dataset-eval conflicts with "
                            f"{', '.join('--' + key.removeprefix('dataset.').replace('_', '-') for key in conflicts)}. "
                            "Synthetic evaluation generates its own evaluation data from all training rows. "
                            "Remove the conflicting split/evaluation-source options to use it, or pass "
                            "--no-synthetic-dataset-eval to use your split or separate evaluation file.")
        # The synthetic mode replaces configured evaluation sources and retains all
        # primary rows. Persist the resolved form so workers and resume agree.
        previous = dataset.pop("dataset", None)
        if previous is not None:
            dataset.setdefault("dataset_train", previous)
        dataset["split"] = False
        dataset.pop("eval_fraction", None)
        dataset.pop("dataset_eval", None)
    if "dataset.split" in overrides and isinstance(dataset, dict):
        incompatible = ("dataset_train", "dataset_eval") if dataset["split"] else ("dataset", "eval_fraction")
        conflicts = [key for key in incompatible if "dataset." + key in overrides]
        if conflicts:
            raise TrlxError(f"{path}: dataset split mode conflicts with "
                            f"{', '.join('--' + key.replace('_', '-') for key in conflicts)}. "
                            "--dataset-train/--dataset-eval select already-separated sources and imply --no-split. "
                            "To reserve part of one file for evaluation, use --dataset FILE --split "
                            "--eval-fraction FRACTION (0 < FRACTION < 1), replacing --dataset-train and "
                            "removing --dataset-eval or --no-split. For already-separated files, "
                            "remove --eval-fraction and --shuffle-eval-data instead.")
        if dataset["split"]:
            previous = dataset.pop("dataset_train", None)
            dataset.pop("dataset_eval", None)
            if previous is not None:
                dataset.setdefault("dataset", previous)
        else:
            previous = dataset.pop("dataset", None)
            dataset.pop("eval_fraction", None)
            # A configured membership shuffle belongs to the replaced split mode;
            # keep explicit CLI input so validation still rejects a contradiction.
            if "dataset.shuffle_eval_data" not in overrides:
                dataset.pop("shuffle_eval_data", None)
            if previous is not None:
                dataset.setdefault("dataset_train", previous)
    if source is not None:
        dataset = doc.setdefault("dataset", {})
        if not isinstance(dataset, dict):
            raise TrlxError(f"{path}: [dataset] must be a table")
        key = "dataset" if dataset.get("split") else "dataset_train"
        dataset[key] = source
    if (overrides.get("dataset.split") is False and not dataset.get("dataset_eval")
            and not dataset.get("synthetic_dataset_eval")):
        # --no-split without a separate eval source explicitly requests training
        # only. Remove the generated eval schedule, but reject contradictory CLI input.
        conflicts = [key for key in overrides if key.startswith("eval_") and
                     not (key == "eval_strategy" and overrides[key] == "no")]
        if conflicts:
            raise TrlxError(f"{path}: evaluation is disabled but CLI sets "
                            f"{', '.join('--' + key.replace('_', '-') for key in conflicts)}. "
                            "An evaluation schedule does not supply evaluation data. Add --dataset-eval FILE "
                            "for separate data, or use --dataset FILE --split --eval-fraction FRACTION "
                            "(0 < FRACTION < 1), replacing --dataset-train/--no-split. "
                            "For training without evaluation, remove the listed options.")
        for key in list(doc):
            if key.startswith("eval_"):
                del doc[key]
    return doc


# Read the operator config or a resolved snapshot; callers own interpretation.
def _read_toml(path):
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise TrlxError(
            f"{path}: no such config file; check --config or create defaults with trlx init. "
            "For resume, restore the run's original config.toml snapshot."
        ) from e
    except OSError as e:
        raise TrlxError(f"{path}: cannot read config: {e.strerror or e}") from e
    except UnicodeDecodeError as e:
        raise TrlxError(f"{path}: config is not valid UTF-8; save the TOML file as UTF-8") from e
    except tomllib.TOMLDecodeError as e:
        raise TrlxError(f"{path}: invalid TOML: {e}") from e


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
            raise TrlxError(f"{path}: '{key}' is not accepted at top level; the strategy is chosen by trlx or --strategy. "
                            f"Remove '{key}' from the config and use --strategy auto, ddp, or fsdp instead.")
        if key in BASELINE_FIELDS:
            raise TrlxError(f"{path}: '{key}' is managed by trlx; fresh runs automatically evaluate the starting model "
                            f"when evaluation is enabled. Remove '{key}' from the config; use --eval-strategy "
                            "to control subsequent evaluations.")
        if key in VLLM_FORCED and "rewards" in method.blocks:
            raise TrlxError(f"{path}: '{key}' is not accepted at top level; {method.name} always uses the TRL vLLM server. "
                            f"Remove '{key}' from the config and use --vllm-server-base-url to select the TRL server.")
        if key in REPLAY_KL_FORCED and kl_forced:
            raise TrlxError(
                f"{path}: '{key}' is not accepted at top level when [replay].kl_coef > 0; the KL term needs "
                f"logits, so trlx sets {key} = {REPLAY_KL_FORCED[key]!r}. "
                f"Remove --{key.replace('_', '-')} and the '{key}' config entry to keep replay KL, "
                "or use --replay-kl-coef 0 for plain replay mixing."
            )
        if key not in fields:
            raise TrlxError(f"{path}: unknown key '{key}'; not a setting for {method.name} or a trlx block. "
                            f"Use trlx {method.name} --help to find the supported option and its config key; "
                            "TOML keys use underscores, while CLI flags use hyphens.")
        # TOML has no null; the string "None" stands in, only where the field's
        # type admits None.
        if value == "None":
            if not admits_none(hints[key]):
                raise TrlxError(f"{path}: '{key}' does not accept \"None\" (type {_type_name(hints[key])})")
            value = None
        elif not _matches(value, hints[key]):
            # Even malformed credential values stay out of operator diagnostics.
            detail = "" if key in ("hub_token", "push_to_hub_token") else f" {value!r}"
            raise TrlxError(
                f"{path}: '{key}' must be {_type_name(hints[key])}, got {type(value).__name__}{detail}"
            )
        kwargs[key] = value

    if "output_dir" not in kwargs or kwargs["output_dir"] is None:
        raise TrlxError(f"{path}: 'output_dir' is required; supply --output-dir DIR or set output_dir in the config. "
                        "DIR is the parent under which fresh run directories are created.")
    if not eval_enabled:
        eval_keys = sorted(k for k in kwargs if k.startswith("eval_"))
        if eval_keys:
            raise TrlxError(
                f"{path}: evaluation is disabled (no eval dataset) but eval keys are set: {', '.join(eval_keys)}. "
                "Supply --dataset-eval FILE for separate evaluation data, or use --dataset FILE --split "
                "--eval-fraction FRACTION (0 < FRACTION < 1). For training only, use --no-split "
                "and remove explicit evaluation options."
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
    # Resolve save defaults before the dataclass validates coupled save/eval settings.
    # Explicit strategies retain their library defaults when evaluation is disabled.
    save_strategy_given = "save_strategy" in kwargs
    eval_strategy = kwargs.get("eval_strategy", fields["eval_strategy"].default)
    kwargs.setdefault("save_strategy", eval_strategy if eval_strategy != "no" else "steps")
    if "save_steps" not in kwargs and kwargs["save_strategy"] == "steps":
        if eval_strategy == "steps":
            kwargs["save_steps"] = (kwargs.get("eval_steps") or kwargs.get(
                "logging_steps", fields["logging_steps"].default))
        elif not save_strategy_given and eval_strategy == "no":
            # Transformers skips periodic saves at zero but still saves the final step.
            # Keep the steps strategy: "no" would also suppress the final checkpoint.
            kwargs["save_steps"] = 0

    try:
        args = cls(**kwargs)
    except (ValueError, TypeError) as e:
        # TRL and transformers validate field combinations in __post_init__.
        raise TrlxError(f"{path}: {cls.__name__} rejected the config: {e}")
    # The trainer's own startup evaluation runs after distributed preparation and
    # before its first update. Resume keeps the original step-zero measurements.
    args.eval_on_start = bool(eval_enabled and args.eval_strategy != "no" and not args.resume_from_checkpoint)
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


# Missing settings name their owning table; split inputs also explain their CLI forms.
def _require(path, where, table, key, kind):
    if key not in table:
        raise TrlxError(f"{path}: {where} requires '{key}'. " + (
            "Supply --eval-fraction FRACTION with 0 < FRACTION < 1 (0.1 holds out 10%); "
            "use --dataset FILE for the source being split."
            if where == "[dataset]" and key == "eval_fraction" else
            "Use --dataset FILE --split --eval-fraction FRACTION (0 < FRACTION < 1) for one source, "
            "or --dataset-train TRAIN with optional --dataset-eval EVAL for already-separated data."
            if where == "[dataset]" and key == "split" else
            f"Add the '{key}' entry to {where} in the configuration; it has no implicit default."
        ))
    return _typed(path, where, table, key, kind)


# Type check for a block value. `kind` is a type or tuple of types as for
# isinstance. bool is an int subclass in Python, so a bool only passes when
# bool itself is the kind asked for; `train = true` must not count as an int.
def _typed(path, where, table, key, kind):
    value = table[key]
    bool_where_not_wanted = isinstance(value, bool) and kind is not bool
    if bool_where_not_wanted or not isinstance(value, kind):
        if isinstance(kind, tuple):
            want = "a number" if kind == (int, float) else " or ".join(item.__name__ for item in kind)
        else:
            want = kind.__name__
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
    _check_keys(path, block, table, ("split", "dataset", "eval_fraction", "dataset_train", "dataset_eval",
                                    "synthetic_dataset_eval", "shuffle_eval_data", "include_reasoning",
                                    "reasoning_only_loss"))
    split = _require(path, block, table, "split", bool)
    reasoning_only_loss = (_require(path, block, table, "reasoning_only_loss", bool)
                           if "reasoning_only_loss" in table else False)
    include_reasoning = (_require(path, block, table, "include_reasoning", bool)
                         if "include_reasoning" in table else False)
    if reasoning_only_loss:
        if "include_reasoning" in table and not include_reasoning:
            raise TrlxError(f"{path}: reasoning_only_loss requires include_reasoning; "
                            "cannot combine with include_reasoning = false. Use --include-reasoning "
                            "to supply reasoning to the loss, or --no-reasoning-only-loss to disable that objective.")
        # Persist the implication so workers and resume compare the same inputs.
        table["include_reasoning"] = include_reasoning = True
    shuffle = _require(path, block, table, "shuffle_eval_data", bool) if "shuffle_eval_data" in table else False
    if shuffle and not split:
        raise TrlxError(f"{path}: --shuffle-eval-data requires splitting one dataset; it selects which rows "
                        "are held out, rather than shuffling separate training/evaluation files. "
                        "Use --dataset FILE --split --eval-fraction FRACTION (0 < FRACTION < 1), replacing "
                        "--dataset-train and removing --dataset-eval/--no-split. To keep separate sources, "
                        "use --no-shuffle-eval-data or set shuffle_eval_data = false in [dataset].")
    synthetic = (_require(path, block, table, "synthetic_dataset_eval", bool)
                 if "synthetic_dataset_eval" in table else False)
    if synthetic and (split or "dataset_eval" in table):
        raise TrlxError(f"{path}: synthetic_dataset_eval requires split = false without dataset_eval: "
                        "it generates evaluation summaries from dataset_train. Set those [dataset] keys "
                        "accordingly, or disable synthetic_dataset_eval to use your existing evaluation source.")
    # The two forms are exclusive; a key from the other form is an error rather
    # than silently ignored.
    if split:
        for wrong in ("dataset_train", "dataset_eval"):
            if wrong in table:
                raise TrlxError(f"{path}: [dataset] split = true uses 'dataset' and 'eval_fraction'; '{wrong}' is for split = false. "
                                "For one source, use dataset = \"FILE\" and remove dataset_train/dataset_eval. "
                                "For separate files, set split = false, remove dataset/eval_fraction, "
                                "and supply dataset_train and optionally dataset_eval.")
        fraction = _require(path, block, table, "eval_fraction", (int, float))
        if not 0 < fraction < 1:
            raise TrlxError(f"{path}: --eval-fraction / [dataset].eval_fraction must be between 0 and 1, "
                            f"exclusive, got {fraction}. Use a fraction, such as --eval-fraction 0.1 for 10%. "
                            "For no evaluation split, use --no-split and omit --eval-fraction.")
        if "dataset" not in table:
            raise TrlxError(f"{path}: [dataset] requires 'dataset' when splitting. "
                            "Supply --dataset FILE, not --dataset-train, or set dataset = \"FILE\" under [dataset].")
        return DatasetSpec(True, dataset_ref(path, "[dataset].dataset", table["dataset"]), float(fraction), None,
                           shuffle_eval_data=shuffle, include_reasoning=include_reasoning,
                           reasoning_only_loss=reasoning_only_loss)
    for wrong in ("dataset", "eval_fraction"):
        if wrong in table:
            raise TrlxError(f"{path}: [dataset] split = false uses 'dataset_train' and 'dataset_eval'; '{wrong}' is for split = true. "
                            "Use dataset_train = \"FILE\" and remove dataset/eval_fraction for separate data, "
                            "or remove dataset_train/dataset_eval and set split = true with dataset = \"FILE\" "
                            "and 0 < eval_fraction < 1 for a split.")
    if "dataset_train" not in table:
        raise TrlxError(f"{path}: [dataset] requires 'dataset_train'; supply --dataset or --dataset-train")
    eval_ref = dataset_ref(path, "[dataset].dataset_eval", table["dataset_eval"]) if "dataset_eval" in table else None
    return DatasetSpec(False, dataset_ref(path, "[dataset].dataset_train", table["dataset_train"]),
                       None, eval_ref, synthetic, include_reasoning=include_reasoning,
                       reasoning_only_loss=reasoning_only_loss)


# PEFT validates field combinations after nullable CLI/config values are decoded.
# trlx owns task_type and translates failures into a message naming the config.
def _peft(path, method, table):
    owned = sorted(set(table) & PEFT_OWNED_FIELDS)
    if owned:
        raise TrlxError(f"{path}: [peft] keys {', '.join(owned)} are set by trlx for the selected trainer; "
                        "remove these entries from [peft] rather than overriding them.")
    hints = typing.get_type_hints(LoraConfig)
    values = {key: None if value == "None" and admits_none(hints.get(key)) else value
              for key, value in table.items()}
    try:
        return LoraConfig(task_type=method.peft_task_type, **values)
    except TypeError as e:
        # dataclass __init__ names the offending keyword in its message.
        raise TrlxError(f"{path}: [peft] {e}. Check the LoRA option's config key and type in "
                        f"trlx {method.name} --help; for example, --lora-r maps to [peft].r.")
    except ValueError as e:
        raise TrlxError(f"{path}: [peft] rejected by peft: {e}. Correct the named LoRA values in [peft] "
                        f"or their --lora-* overrides; trlx {method.name} --help lists their constraints.")


# [preflight]: both keys required; the block itself is optional (SPEC 2.2).
def _preflight(path, table):
    _check_keys(path, "[preflight]", table, ("offpolicy_logp_per_token", "rows"))
    value = _require(path, "[preflight]", table, "offpolicy_logp_per_token", (int, float))
    rows = _require(path, "[preflight]", table, "rows", int)
    if rows <= 0:
        raise TrlxError(f"{path}: [preflight].rows must be a positive row count, got {rows}; "
                        "set rows to an integer >= 1 for the off-policy data check.")
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
        raise TrlxError(f"{path}: [rewards].funcs must list at least one reward. "
                        "Add a reward name, model path, or module:function using --reward; "
                        "repeat --reward for multiple objectives, or populate funcs in [rewards].")
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
        raise TrlxError(f"{path}: [replay] requires 'dataset'; supply --replay-dataset FILE "
                        "or set dataset in [replay]. Use --no-replay if replay mixing is not intended.")
    fraction = float(_require(path, "[replay]", table, "fraction", (int, float)))
    if not 0 < fraction < 1:
        raise TrlxError(f"{path}: --replay-fraction / [replay].fraction must be in (0, 1), got {fraction}. "
                        "This is replay's share of the mixed training set; 0.2 means 20%. "
                        "Use --no-replay to disable replay rather than setting its fraction to zero.")
    kl_coef = float(_require(path, "[replay]", table, "kl_coef", (int, float)))
    if kl_coef < 0:
        raise TrlxError(f"{path}: --replay-kl-coef / [replay].kl_coef must be >= 0, got {kl_coef}. "
                        "Use 0 for plain replay mixing, or a positive value to add KL regularization.")
    return ReplaySpec(dataset_ref(path, "[replay].dataset", table["dataset"]), fraction, kl_coef)


# Keep empty legacy tables readable, but never silently ignore retired behavior.
def _verify(path, table):
    if "prompts" in table:
        raise TrlxError(f"{path}: [verify].prompts was removed with prompt-based verification; remove this setting")
    _check_keys(path, "[verify]", table, ())


# Inspection can represent a missing block, but executable training/check paths cannot invent defaults.
def require_assessment(cfg, path):
    if cfg.assessment is None:
        raise TrlxError(f"{path}: training and check require [assessment]; add its explicit settings. "
                        "Use trlx init --out <unused-path> to generate an example without replacing this file")
    return cfg.assessment


# Nullable wrapper strings use the same visible 'None' spelling as trainer overrides.
def _assessment_string(path, where, table, key):
    value = _require(path, where, table, key, (str, type(None)))
    return None if value in (None, "None") else value


# Every operational assessment setting is required explicitly, even while independent checks are disabled.
def _assessment(path, table, method):
    from trlx.quality_scorers import PRESETS

    where = "[assessment]"
    _check_keys(path, where, table, {field.name for field in dataclasses.fields(AssessmentSpec)})
    values = {"quality_checks": _require(path, where, table, "quality_checks", bool)}
    minimums = {"quality_max_length": 2,
                "quality_max_new_tokens": 1, "quality_batch_size": 1}
    for key, minimum in minimums.items():
        value = _require(path, where, table, key, int)
        if value < minimum:
            raise TrlxError(f"{path}: --{key.replace('_', '-')} / {where}.{key} must be an integer "
                            f">= {minimum}, got {value}; set this token/batch limit to a valid positive value.")
        values[key] = value
    preset = _assessment_string(path, where, table, "quality_preset")
    if preset is not None and preset not in PRESETS:
        raise TrlxError(f"{path}: --quality-preset / {where}.quality_preset must be None or one of "
                        f"{', '.join(PRESETS)}. Select a preset matching the evaluation data; "
                        "use --no-quality-checks to disable the checks and --quality-preset None to clear an invalid preset.")
    source = _assessment_string(path, where, table, "quality_dataset")
    values["quality_preset"] = preset
    values["quality_dataset"] = dataset_ref(path, where + ".quality_dataset", source) if source is not None else None
    values["judge"] = _assessment_judge(path, table["judge"]) if "judge" in table else None
    if values["quality_checks"]:
        if preset is None or source is None:
            raise TrlxError(f"{path}: --quality-checks requires both --quality-preset PRESET and "
                            "--quality-dataset FILE (or quality_preset/quality_dataset in [assessment]). "
                            "Supply a built-in preset and its matching data, or use --no-quality-checks "
                            "to run without independent quality checks.")
        if (method == "reward") != (preset == "preference"):
            raise TrlxError(f"{path}: the reward trainer requires the preference quality preset; "
                            "generative trainers require a generative or language_modeling preset. "
                            "For trlx reward, use --quality-preset preference with preference data. "
                            "For other trainers, choose another --quality-preset with matching data, "
                            "or disable these checks with --no-quality-checks.")
        if preset in {"instruction_following", "writing"}:
            judge = values["judge"]
            if judge is None or not judge["url"] or not judge["model"]:
                raise TrlxError(f"{path}: {preset} quality checks require a judge endpoint URL and served model name. "
                                "Configure [assessment.judge] with url, model, api_key (environment-variable name or \"None\"), "
                                "timeout > 0, retries >= 0, and max_tokens >= 1. The corresponding CLI flags "
                                "are --quality-judge-url, --quality-judge-model, --quality-judge-api-key, "
                                "--quality-judge-timeout, --quality-judge-retries, and --quality-judge-max-tokens. "
                                "Use --no-quality-checks if no judge is intended.")
            # Validate connection/credential input without contacting the service or running a judge.
            import os
            from dataset.endpoint import Endpoint
            from dataset.io import DatasetError

            name = judge["api_key"]
            key = os.environ.get(name) if name is not None else None
            if name is not None and not key:
                raise TrlxError(f"{path}: [assessment.judge].api_key names an environment variable that is unset or empty. "
                                "Set that variable in the environment or working-directory .env file. "
                                "--quality-judge-api-key takes its name, not the credential; use None for an unauthenticated endpoint.")
            try:
                Endpoint(judge["url"], judge["model"], key, judge["timeout"], judge["retries"])
            except DatasetError as error:
                raise TrlxError(f"{path}: [assessment.judge]: {error}") from error
    return AssessmentSpec(**values)


# Judge connection values are operator inputs; built-in rubric definitions live with the scorers.
def _assessment_judge(path, table):
    import math

    where = "[assessment.judge]"
    if not isinstance(table, dict):
        raise TrlxError(f"{path}: {where} must be a table")
    _check_keys(path, where, table, {"url", "model", "api_key", "timeout", "retries", "max_tokens"})
    values = {key: _assessment_string(path, where, table, key) for key in ("url", "model", "api_key")}
    if values["api_key"] is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", values["api_key"]) is None:
        raise TrlxError(f"{path}: {where}.api_key must name an environment variable, never contain a credential. "
                        "Store the credential in the environment or .env, then pass its variable name with "
                        "--quality-judge-api-key; use None if the endpoint needs no authentication.")
    timeout = _require(path, where, table, "timeout", (int, float))
    if not math.isfinite(timeout) or timeout <= 0:
        raise TrlxError(f"{path}: --quality-judge-timeout / {where}.timeout must be finite positive seconds; "
                        "choose a number greater than zero for each judge request's timeout.")
    values["timeout"] = float(timeout)
    for key, minimum in (("retries", 0), ("max_tokens", 1)):
        value = _require(path, where, table, key, int)
        if value < minimum:
            raise TrlxError(f"{path}: --quality-judge-{key.replace('_', '-')} / {where}.{key} "
                            f"must be an integer >= {minimum}, got {value}.")
        values[key] = value
    return values
