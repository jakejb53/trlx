"""`trlx verify <checkpoint> --base <model> [--prompts <dataset>]` (SPEC 2.7).

One code path for the standalone command and the end of a run: the
supervisor spawns this command as its own process once every worker has
exited, with every selected GPU visible, and the model loads with
device_map="auto" so a checkpoint too large for one GPU still verifies.

Three checks:
- adapter loaded (LoRA checkpoints only): adapter_check on a fresh load.
- behaviour changed: outputs from base and checkpoint on the prompts must
  differ on at least one. A model that cannot generate (a reward model) is
  compared on its scores instead.
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

from trlx import TrlxError, adapter_check, config as config_mod, generate, model as model_mod, show


# The verify record: verify.json is this object plus `ok`. `failures` lists
# every failed check in words; empty means every check passed.
@dataclasses.dataclass
class Result:
    checkpoint: str
    base: str
    # adapter_check.AdapterCheck as a dict, None for a full fine-tune.
    adapter: dict | None
    # prompt count, differing count, and every prompt with both outputs.
    behaviour: dict
    chat_template_equal: bool
    failures: list

    @property
    def ok(self):
        return not self.failures

    # verify.json shape; `ok` is derived, written for readers of the file.
    def to_dict(self):
        return {**dataclasses.asdict(self), "ok": self.ok}


# Runs the checks and returns the Result. `prompts_ref` is a config.DatasetRef
# or None for the built-in prompts.
def run(checkpoint, base, prompts_ref):
    ckpt = pathlib.Path(checkpoint)
    if not ckpt.is_dir():
        raise TrlxError(f"{checkpoint}: not a directory")
    if not torch.cuda.is_available():
        raise TrlxError("verify needs a CUDA device and none is available")
    run_dir = ckpt.parent if (ckpt.parent / show.CONFIG_FILENAME).is_file() else None
    spec = _base_spec(run_dir, base)
    prompts = generate.prompts_from(prompts_ref)
    failures = []

    if (ckpt / adapter_check.ADAPTER_FILE).is_file():
        kind = model_mod.SEQUENCE_CLASSIFICATION if adapter_check.task_type(ckpt) == "SEQ_CLS" else model_mod.CAUSAL
        base_model = model_mod.load_model(spec, kind, device_map="auto")
        processor = model_mod.load_processor(spec)
        print(f"loaded {type(base_model).__name__} from {base}", flush=True)
        try:
            peft_model = PeftModel.from_pretrained(base_model, str(ckpt))
        except (OSError, ValueError) as e:
            raise TrlxError(f"{checkpoint}: cannot load adapter: {e}")
        check = adapter_check.check(ckpt, peft_model)
        print(check.message(), flush=True)
        adapter = dataclasses.asdict(check) | {"ok": check.ok}
        if not check.ok:
            failures.append("adapter check failed")
        # Same weights, adapters off: the base's behaviour without a second
        # copy of the model in memory.
        with peft_model.disable_adapter():
            base_outputs = _outputs(peft_model, processor, prompts)
        ckpt_outputs = _outputs(peft_model, processor, prompts)
    else:
        # A full fine-tune: the checkpoint is a whole model, and its own
        # config says which kind (a reward model was saved as sequence
        # classification), so the base is loaded as the same kind. Base and
        # checkpoint are loaded in turn so only one is resident at a time.
        adapter = None
        ckpt_spec = dataclasses.replace(spec, path=str(ckpt))
        kind = _checkpoint_kind(ckpt_spec)
        base_model = model_mod.load_model(spec, kind, device_map="auto")
        processor = model_mod.load_processor(spec)
        print(f"loaded {type(base_model).__name__} from {base}", flush=True)
        base_outputs = _outputs(base_model, processor, prompts)
        del base_model
        torch.cuda.empty_cache()
        ckpt_model = model_mod.load_model(ckpt_spec, kind, device_map="auto")
        print(f"loaded {type(ckpt_model).__name__} from {checkpoint}", flush=True)
        ckpt_outputs = _outputs(ckpt_model, processor, prompts)

    samples = [
        {"prompt": p, "base": b, "checkpoint": c, "differs": b != c}
        for p, b, c in zip(prompts, base_outputs, ckpt_outputs)
    ]
    differing = sum(s["differs"] for s in samples)
    behaviour = {"prompts": len(prompts), "differing": differing, "samples": samples}
    print(f"behaviour: {differing} of {len(prompts)} outputs differ between base and checkpoint", flush=True)
    for s in samples:
        print(f"  prompt:     {json.dumps(_text(s['prompt']))}", flush=True)
        print(f"  base:       {json.dumps(s['base'])}", flush=True)
        print(f"  checkpoint: {json.dumps(s['checkpoint'])}", flush=True)
    if differing == 0:
        failures.append("behaviour unchanged: every output equals the base's")

    template_equal = _chat_template_equal(spec, ckpt, processor, failures)

    result = Result(str(ckpt), base, adapter, behaviour, template_equal, failures)
    out_dir = run_dir if run_dir is not None else ckpt
    path = out_dir / show.VERIFY_FILENAME
    try:
        path.write_text(json.dumps(result.to_dict(), indent=1) + "\n", encoding="utf-8")
    except OSError as e:
        raise TrlxError(f"{path}: cannot write: {e.strerror or e}")
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
    except (OSError, tomllib.TOMLDecodeError) as e:
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
def _checkpoint_kind(ckpt_spec):
    config = model_mod.load_config(ckpt_spec)
    architectures = getattr(config, "architectures", None) or []
    try:
        seq_cls = model_mod.model_class(ckpt_spec, config, model_mod.SEQUENCE_CLASSIFICATION)
    except TrlxError:
        return model_mod.CAUSAL
    return model_mod.SEQUENCE_CLASSIFICATION if architectures[:1] == [seq_cls.__name__] else model_mod.CAUSAL


# Text outputs for the prompts: generated completions, or for a model that
# cannot generate (a sequence-classification reward model) its score per
# prompt, formatted so equal scores compare equal as strings.
def _outputs(model, processor, prompts):
    if model.can_generate():
        return generate.generate(model, processor, prompts)
    tokenizer = getattr(processor, "tokenizer", processor)
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for prompt in prompts:
            text, templated = generate._render(tokenizer, prompt)
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=not templated).to(device)
            scores.append(f"score {model(**encoded).logits[0, 0].item():.6g}")
    return scores


# The checkpoint's saved chat template against the base's. A checkpoint
# without a saved tokenizer has no template to compare, which fails the check
# rather than passing it silently.
def _chat_template_equal(spec, ckpt, base_processor, failures):
    try:
        ckpt_processor = model_mod.load_processor(dataclasses.replace(spec, path=str(ckpt)))
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


# One-line form of a prompt for the printed report; messages become
# "role: content" lines.
def _text(prompt):
    if isinstance(prompt, str):
        return prompt
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in prompt)
