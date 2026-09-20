"""`trlx merge --base <model> --adapter <dir> --out <dir>`.

Loads the base with the class its own config names, attaches the adapter,
runs the adapter-load check, and only then merges. A merge that passed the
check is a merge of the trained weights; one that failed would have written
the base model unchanged, so failure is fatal before anything is written.
"""

import pathlib

import torch
from peft import PeftModel

from dataset.io import DatasetError, publish_output, validate_output
from trlx import TrlxError, adapter_check, model
from trlx.config import ModelSpec


# Saving may reopen input assets. Detect both physical containment and paths that
# traverse an output symlink: replacing that link can break reads even if its target survives.
def _check_direct_inputs(target, base, adapter):
    for label, source in (("--base", base), ("--adapter", adapter)):
        path = pathlib.Path(source)
        try:
            if not path.exists():
                continue  # Remote model IDs are resolved by the model loader, not as local directories.
            resolved = path.resolve()
            overlaps = not target.is_symlink() and (target == resolved or target in resolved.parents)
            pending = [path, *path.parents]
            seen = set()
            while pending:
                entry = pending.pop()
                accessed = (entry.resolve() if entry.name in ("", "..")
                            else entry.parent.resolve() / entry.name)
                if accessed in seen:
                    continue
                seen.add(accessed)
                overlaps = overlaps or accessed == target
                # An input can traverse an output link indirectly through another link.
                # Following each link's spelling retains entries that resolve() would hide.
                if accessed.is_symlink():
                    linked = accessed.readlink()
                    if not linked.is_absolute():
                        linked = accessed.parent / linked
                    pending.extend((linked, *linked.parents))
        except (OSError, RuntimeError) as e:
            raise TrlxError(f"{label} {source}: cannot inspect input path: {e}; check the path and permissions") from e
        if overlaps:
            raise TrlxError(
                f"{target}: --no-staging cannot replace {label} {source} or a directory containing it; "
                "saving may still read input files. Omit --no-staging to merge in place with --force, "
                "or choose a separate output directory"
            )


# Validate trained adapter weights before publishing the model and its processor together.
def merge(base, adapter, out, *, force=False, no_staging=False):
    try:
        target = validate_output(out, force, directory=True)
    except DatasetError as e:
        raise TrlxError(str(e)) from e
    if no_staging:
        _check_direct_inputs(target, base, adapter)
    # A GPU is a baseline requirement of trlx; there is no CPU path.
    if not torch.cuda.is_available():
        raise TrlxError("merge needs a CUDA device and none is available")
    task_type = adapter_check.task_type(adapter)
    kind = model.SEQUENCE_CLASSIFICATION if task_type == "SEQ_CLS" else model.CAUSAL
    # dtype "auto" keeps the base checkpoint's own dtype; merge has no flag
    # for it because a merge must not change the weights' precision.
    spec = ModelSpec(path=base, dtype="auto", trust_remote_code=None, attn_implementation=None)
    base_model = model.load_model(spec, kind, device_map="auto")
    print(f"loaded {type(base_model).__name__} from {base}")

    try:
        peft_model = PeftModel.from_pretrained(base_model, adapter)
    except torch.cuda.OutOfMemoryError as e:
        raise TrlxError(f"{adapter}: CUDA memory exhausted loading adapter; free GPU memory") from e
    except (OSError, ValueError) as e:
        raise TrlxError(f"{adapter}: cannot load adapter: {e}")

    result = adapter_check.check(adapter, peft_model)
    print(result.message())
    if not result.ok:
        raise TrlxError("adapter check failed; nothing written")

    try:
        merged = peft_model.merge_and_unload()
    except torch.cuda.OutOfMemoryError as e:
        raise TrlxError(f"{adapter}: CUDA memory exhausted merging adapter; free GPU memory") from e
    # In-place replacement is staged, so save_pretrained can still read input assets.
    processor = model.load_processor(spec)

    # Both artifacts share one publication boundary, so staging protects either failure.
    def save(destination):
        try:
            merged.save_pretrained(destination)
            processor.save_pretrained(destination)
        except torch.cuda.OutOfMemoryError as e:
            raise DatasetError(f"{out}: CUDA memory exhausted saving merged model; free GPU memory") from e
        except ValueError as e:
            raise DatasetError(f"{out}: cannot serialize merged model or processor: {e}; "
                               "check the model and processor configuration") from e

    try:
        publish_output(out, save, force=force, no_staging=no_staging, directory=True)
    except DatasetError as e:
        raise TrlxError(str(e)) from e
    print(f"merged model written to {out}")
