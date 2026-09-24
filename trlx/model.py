"""Model loading by the class the model's own config names.

Never AutoModelForCausalLM: the auto classes pick a class by config type and
task, which for a multimodal checkpoint selects the text-only class and
silently loads a model whose module tree does not match the adapter (SPEC
section 5). The class named in `architectures` is the one the checkpoint was
saved from, so it is the one loaded.

The reward method is the exception: a sequence-classification head is a
different class from the saved one by design, and transformers' own
config-keyed mapping is the authority on which class that is.
"""

import torch
import transformers
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

from dataset.progress import stage
from trlx import TrlxError

SEQUENCE_CLASSIFICATION = "sequence_classification"
CAUSAL = "causal"


# from_pretrained kwargs that a [model] block contributes. Keys the block left
# absent are not passed, so transformers' own defaults apply unchanged.
def _pretrained_kwargs(spec):
    kwargs = {}
    if spec.trust_remote_code is not None:
        kwargs["trust_remote_code"] = spec.trust_remote_code
    return kwargs


# The model's own config. A missing path or unreachable hub id surfaces here
# first, so this is where that error is made readable.
def load_config(spec, *, progress=None):
    try:
        with stage(progress, f"loading model configuration {spec.path}"):
            return AutoConfig.from_pretrained(spec.path, **_pretrained_kwargs(spec))
    except (OSError, ValueError, ImportError) as e:
        raise TrlxError(f"[model].path '{spec.path}': cannot load model config: {e}")


# Resolves the class to load for `kind`, given the model's config.
def model_class(spec, config, kind):
    if kind == SEQUENCE_CLASSIFICATION:
        return _sequence_classification_class(spec, config)
    if kind != CAUSAL:
        raise ValueError(f"unknown model kind {kind!r}")
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise TrlxError(f"[model].path '{spec.path}': config names no architecture; cannot choose a model class")
    name = architectures[0]
    cls = getattr(transformers, name, None)
    if cls is not None:
        return cls
    # A remote-code model's classes are not in transformers; its config lists
    # them in auto_map as "module.Class" and the class is fetched from the
    # checkpoint's own code.
    for ref in (getattr(config, "auto_map", None) or {}).values():
        if ref.rsplit(".", 1)[-1] == name:
            return get_class_from_dynamic_module(ref, spec.path, **_pretrained_kwargs(spec))
    raise TrlxError(
        f"[model].path '{spec.path}': architecture '{name}' is not a transformers class and not in the "
        "config's auto_map"
    )


def _sequence_classification_class(spec, config):
    config_cls = type(config)
    if config_cls in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING:
        return MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING[config_cls]
    ref = (getattr(config, "auto_map", None) or {}).get("AutoModelForSequenceClassification")
    if ref:
        return get_class_from_dynamic_module(ref, spec.path, **_pretrained_kwargs(spec))
    raise TrlxError(
        f"[model].path '{spec.path}': transformers has no sequence-classification class for "
        f"{config_cls.__name__}; this model cannot be trained as a reward model"
    )


# Loads the model described by a [model] or [teacher] block. `kind` is
# Method.model_kind. device_map is passed through when given (merge and
# verify use "auto"); training leaves it None so each rank loads onto its
# own device.
def load_model(spec, kind, device_map=None, *, progress=None):
    config = load_config(spec, progress=progress)
    with stage(progress, f"resolving model class {spec.path}"):
        cls = model_class(spec, config, kind)
    kwargs = _pretrained_kwargs(spec)
    # "auto" is transformers' own "use the checkpoint's dtype"; config.py never
    # accepts it from a run config, merge constructs it directly.
    kwargs["dtype"] = "auto" if spec.dtype == "auto" else getattr(torch, spec.dtype)
    if spec.attn_implementation is not None:
        kwargs["attn_implementation"] = spec.attn_implementation
    if device_map is not None:
        kwargs["device_map"] = device_map
    if kind == SEQUENCE_CLASSIFICATION:
        # A reward model outputs one scalar; TRL's RewardTrainer requires it.
        kwargs["num_labels"] = 1
    try:
        with stage(progress, f"loading model weights {spec.path}", visible=True):
            return cls.from_pretrained(spec.path, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        raise TrlxError(f"[model].path '{spec.path}': CUDA memory exhausted while loading weights; "
                        "free GPU memory or select a smaller model") from e
    except (OSError, ValueError, ImportError) as e:
        raise TrlxError(f"[model].path '{spec.path}': cannot load weights: {e}")


# Tokenizer or processor for the model. AutoProcessor returns the tokenizer
# for text-only checkpoints, so one call covers both.
def load_processor(spec, *, progress=None):
    try:
        with stage(progress, f"loading processor {spec.path}", visible=True):
            return AutoProcessor.from_pretrained(spec.path, **_pretrained_kwargs(spec))
    except (OSError, ValueError, ImportError) as e:
        raise TrlxError(f"[model].path '{spec.path}': cannot load tokenizer or processor: {e}")


# Project tokenizer-side SFT overrides before weights load; the trainer still owns actual preparation.
def assessment_processor(cfg, *, progress=None):
    import inspect
    import pathlib

    processor = load_processor(cfg.model, progress=progress)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos = getattr(cfg.args, "eos_token", None)
    if eos is not None:
        if eos not in tokenizer.get_vocab():
            raise TrlxError(f"eos_token {eos!r} does not exist in the selected tokenizer")
        tokenizer.eos_token = eos
    template = getattr(cfg.args, "chat_template_path", None)
    if template is None:
        return processor
    try:
        with stage(progress, "resolving assessment chat template"):
            path = pathlib.Path(template)
            if path.is_file() and path.suffix in {".jinja", ".j2"}:
                processor.chat_template = path.read_text(encoding="utf-8")
                if getattr(cfg.dataset, "include_reasoning", False):
                    # A replacement template cannot inherit a field mapping from the base template.
                    # Reasoning validation resolves a recognized schema for this effective template.
                    tokenizer.response_template = None
            else:
                from transformers import AddedToken, AutoTokenizer
                from trl.chat_template_utils import clone_chat_template

                source = AutoTokenizer.from_pretrained(template)
                processor.chat_template = source.get_chat_template()
                if getattr(cfg.dataset, "include_reasoning", False):
                    tokenizer.response_template = getattr(source, "response_template", None)
                tokenizer.add_tokens([token for token in source.added_tokens_decoder.values()
                                      if token.content not in tokenizer.get_vocab()])
                tokenizer.eos_token = source.eos_token
                # Match the pinned helper's declared padding rule without instantiating or resizing a model.
                multiple = inspect.signature(clone_chat_template).parameters["resize_to_multiple_of"].default
                size = len(tokenizer.get_vocab())
                projected_size = ((size + multiple - 1) // multiple) * multiple if multiple is not None else size
                index = 0
                while len(tokenizer.get_vocab()) < projected_size:
                    tokenizer.add_tokens([AddedToken(f"<extra_id_{index}>")])
                    index += 1
    except (OSError, UnicodeError, ValueError, ImportError) as error:
        raise TrlxError(f"chat_template_path {template}: cannot prepare assessment tokenizer: {error}") from error
    return processor
