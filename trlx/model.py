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
        with stage(progress, f"loading model weights {spec.path}"):
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
        with stage(progress, f"loading processor {spec.path}"):
            return AutoProcessor.from_pretrained(spec.path, **_pretrained_kwargs(spec))
    except (OSError, ValueError, ImportError) as e:
        raise TrlxError(f"[model].path '{spec.path}': cannot load tokenizer or processor: {e}")
