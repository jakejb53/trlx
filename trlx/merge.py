"""`trlx merge --base <model> --adapter <dir> --out <dir>`.

Loads the base with the class its own config names, attaches the adapter,
runs the adapter-load check, and only then merges. A merge that passed the
check is a merge of the trained weights; one that failed would have written
the base model unchanged, so failure is fatal before anything is written.
"""

import pathlib

import torch
from peft import PeftModel

from trlx import TrlxError, adapter_check, model
from trlx.config import ModelSpec


def merge(base, adapter, out):
    # A GPU is a baseline requirement of trlx; there is no CPU path.
    if not torch.cuda.is_available():
        raise TrlxError("merge needs a CUDA device and none is available")
    out_path = pathlib.Path(out)
    if out_path.exists():
        raise TrlxError(f"{out}: already exists; merge does not overwrite")

    task_type = adapter_check.task_type(adapter)
    kind = model.SEQUENCE_CLASSIFICATION if task_type == "SEQ_CLS" else model.CAUSAL
    # dtype "auto" keeps the base checkpoint's own dtype; merge has no flag
    # for it because a merge must not change the weights' precision.
    spec = ModelSpec(path=base, dtype="auto", trust_remote_code=None, attn_implementation=None)
    base_model = model.load_model(spec, kind, device_map="auto")
    print(f"loaded {type(base_model).__name__} from {base}")

    try:
        peft_model = PeftModel.from_pretrained(base_model, adapter)
    except (OSError, ValueError) as e:
        raise TrlxError(f"{adapter}: cannot load adapter: {e}")

    result = adapter_check.check(adapter, peft_model)
    print(result.message())
    if not result.ok:
        raise TrlxError("adapter check failed; nothing written")

    merged = peft_model.merge_and_unload()
    try:
        merged.save_pretrained(out)
        model.load_processor(spec).save_pretrained(out)
    except OSError as e:
        raise TrlxError(f"{out}: cannot write: {e.strerror or e}")
    print(f"merged model written to {out}")
