"""`trlx verify <checkpoint> --base <model>` (SPEC 2.7).

One code path for the standalone command and the end of a run: the
supervisor spawns this command as its own process once every worker has
exited, with every selected GPU visible, and the model loads with
device_map="auto" so a checkpoint too large for one GPU still verifies.

Structural checks only; these do not measure training quality:
- base and checkpoint load successfully.
- adapter loaded (LoRA checkpoints only): adapter_check on a fresh load.
- chat template: the checkpoint's equals the base's.

The result is written to verify.json in the run directory when the checkpoint
sits in one (its parent holds config.toml), else in the checkpoint directory,
and printed. Any failed check makes the result not ok; the caller exits 1.
"""

import dataclasses
import json
import pathlib
import tomllib

import torch
from peft import PeftModel

from dataset.io import DatasetError, validate_output, write_text
from dataset.progress import stage
from trlx import TrlxError, adapter_check, config as config_mod, model as model_mod, show


# The verify record: verify.json is this object plus `ok`. `failures` lists
# every failed check in words; empty means every check passed.
@dataclasses.dataclass
class Result:
    checkpoint: str
    base: str
    # adapter_check.AdapterCheck as a dict, None for a full fine-tune.
    adapter: dict | None
    chat_template_equal: bool
    failures: list

    @property
    def ok(self):
        return not self.failures

    # verify.json shape; `ok` is derived, written for readers of the file.
    def to_dict(self):
        return {**dataclasses.asdict(self), "ok": self.ok}


# Runs structural checks without generating responses or scoring prompts.
def run(checkpoint, base, *, force=False, no_staging=False, progress=None):
    ckpt = pathlib.Path(checkpoint)
    if not ckpt.is_dir():
        raise TrlxError(f"{checkpoint}: not a directory")
    run_dir = ckpt.parent if (ckpt.parent / show.CONFIG_FILENAME).is_file() else None
    path = (run_dir if run_dir is not None else ckpt) / show.VERIFY_FILENAME
    try:
        validate_output(path, force)
    except DatasetError as e:
        raise TrlxError(str(e)) from e
    if not torch.cuda.is_available():
        raise TrlxError("verify needs a CUDA device and none is available")
    with stage(progress, "reading verification settings"):
        spec = _base_spec(run_dir, base)
    failures = []

    if (ckpt / adapter_check.ADAPTER_FILE).is_file():
        with stage(progress, f"reading adapter configuration {checkpoint}"):
            kind = model_mod.SEQUENCE_CLASSIFICATION if adapter_check.task_type(ckpt) == "SEQ_CLS" else model_mod.CAUSAL
        base_model = model_mod.load_model(spec, kind, device_map="auto", progress=progress)
        processor = model_mod.load_processor(spec, progress=progress)
        print(f"loaded {type(base_model).__name__} from {base}", flush=True)
        try:
            with stage(progress, f"loading adapter {checkpoint}"):
                peft_model = PeftModel.from_pretrained(base_model, str(ckpt))
        except torch.cuda.OutOfMemoryError as e:
            raise TrlxError(f"{checkpoint}: CUDA memory exhausted loading adapter; free GPU memory") from e
        except (OSError, ValueError) as e:
            raise TrlxError(f"{checkpoint}: cannot load adapter: {e}")
        with stage(progress, f"checking loaded adapter {checkpoint}"):
            check = adapter_check.check(ckpt, peft_model)
        print(check.message(), flush=True)
        adapter = dataclasses.asdict(check) | {"ok": check.ok}
        if not check.ok:
            failures.append("adapter check failed")
    else:
        # A full fine-tune: the checkpoint is a whole model, and its own
        # config says which kind (a reward model was saved as sequence
        # classification), so the base is loaded as the same kind. Base and
        # checkpoint are loaded in turn so only one is resident at a time.
        adapter = None
        ckpt_spec = dataclasses.replace(spec, path=str(ckpt))
        kind = _checkpoint_kind(ckpt_spec, progress=progress)
        base_model = model_mod.load_model(spec, kind, device_map="auto", progress=progress)
        processor = model_mod.load_processor(spec, progress=progress)
        print(f"loaded {type(base_model).__name__} from {base}", flush=True)
        del base_model
        torch.cuda.empty_cache()
        ckpt_model = model_mod.load_model(ckpt_spec, kind, device_map="auto", progress=progress)
        print(f"loaded {type(ckpt_model).__name__} from {checkpoint}", flush=True)

    with stage(progress, "checking chat template") as activity:
        template_equal = _chat_template_equal(spec, ckpt, processor, failures, progress=activity)

    result = Result(str(ckpt), base, adapter, template_equal, failures)
    try:
        write_text(path, json.dumps(result.to_dict(), indent=1) + "\n", force=force,
                   no_staging=no_staging, progress=progress)
    except DatasetError as e:
        raise TrlxError(str(e)) from e
    verdict = "verify passed" if result.ok else "verify failed: " + "; ".join(failures)
    print(f"{verdict}; written to {path}", flush=True)
    return result


# The [model] block of the run when the checkpoint is in a trlx run directory,
# so verify loads the base exactly as training did; otherwise the base path at
# its own dtype, as merge does. The snapshot is read with tomllib because
# config.load rejects the [launch] table trlx appended to it.
def _base_spec(run_dir, base):
    if run_dir is None:
        return config_mod.ModelSpec(path=base, dtype="auto", trust_remote_code=None, attn_implementation=None)
    path = run_dir / show.CONFIG_FILENAME
    try:
        with open(path, "rb") as f:
            table = tomllib.load(f).get("model")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as e:
        raise TrlxError(f"{path}: cannot read snapshot: {e}")
    if not isinstance(table, dict):
        raise TrlxError(f"{path}: snapshot has no [model] block")
    spec = config_mod.model_spec(path, "model", table)
    if spec.path != base:
        raise TrlxError(f"--base {base} is not the run's [model].path {spec.path} ({path})")
    return spec


# Model kind of a full checkpoint: sequence classification when the class
# its config names is the sequence-classification class transformers maps
# that config to, causal otherwise.
def _checkpoint_kind(ckpt_spec, *, progress=None):
    config = model_mod.load_config(ckpt_spec, progress=progress)
    architectures = getattr(config, "architectures", None) or []
    try:
        seq_cls = model_mod.model_class(ckpt_spec, config, model_mod.SEQUENCE_CLASSIFICATION)
    except TrlxError:
        return model_mod.CAUSAL
    return model_mod.SEQUENCE_CLASSIFICATION if architectures[:1] == [seq_cls.__name__] else model_mod.CAUSAL


# The checkpoint's saved chat template against the base's. A checkpoint
# without a saved tokenizer has no template to compare, which fails the check
# rather than passing it silently.
def _chat_template_equal(spec, ckpt, base_processor, failures, *, progress=None):
    try:
        ckpt_processor = model_mod.load_processor(dataclasses.replace(spec, path=str(ckpt)), progress=progress)
    except TrlxError as e:
        failures.append(f"chat template: checkpoint has no loadable tokenizer ({e})")
        return False
    base_template = getattr(getattr(base_processor, "tokenizer", base_processor), "chat_template", None)
    ckpt_template = getattr(getattr(ckpt_processor, "tokenizer", ckpt_processor), "chat_template", None)
    equal = base_template == ckpt_template
    print(f"chat template: {'equal to base' if equal else 'DIFFERS from base'}", flush=True)
    if not equal:
        failures.append("chat template differs from the base's")
    return equal
