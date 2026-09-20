"""Adapter-load check: lora_B tensors in the adapter file versus the PeftModel.

peft initialises every lora_B to zero, so after training the file's lora_B
tensors are nonzero. If the loaded PeftModel's lora_B count and max magnitude
equal the file's, the file's tensors landed on the model. Zero in the model
with nonzero in the file means the adapter keys did not match the module tree
(the wrong model class was loaded) and peft built fresh zero adapters; a
count mismatch means the same for part of the tree.

The check must run before merge_and_unload, which removes the LoRA modules.
"""

import dataclasses
import json
import math
import pathlib

from safetensors import SafetensorError, safe_open

from trlx import TrlxError

ADAPTER_FILE = "adapter_model.safetensors"

# The file may hold fp32 while the model was loaded in bf16, which rounds to
# about three significant digits; 1% relative tolerance covers that and no
# genuine mismatch is that small (a missed key is a magnitude of zero).
REL_TOL = 1e-2


@dataclasses.dataclass(frozen=True)
class AdapterCheck:
    file_count: int
    file_max: float
    model_count: int
    model_max: float

    @property
    def ok(self):
        counts = self.file_count == self.model_count
        magnitudes = math.isclose(self.file_max, self.model_max, rel_tol=REL_TOL, abs_tol=0.0)
        return counts and magnitudes and self.file_count > 0

    # One line per side, then the verdict with its reason.
    def message(self):
        lines = [
            f"adapter file:  {self.file_count} lora_B tensors, max |value| {self.file_max:.6g}",
            f"loaded model:  {self.model_count} lora_B tensors, max |value| {self.model_max:.6g}",
        ]
        if self.ok:
            lines.append("adapter loaded: counts and magnitudes match")
        elif self.file_count == 0:
            lines.append("adapter check failed: the file holds no lora_B tensors")
        elif self.model_max == 0.0 and self.file_max > 0.0:
            lines.append("adapter check failed: model lora_B are all zero; adapter keys did not match the model "
                         "(wrong model class or wrong base model)")
        elif self.file_count != self.model_count:
            lines.append("adapter check failed: tensor count differs; part of the adapter did not match the model")
        else:
            lines.append("adapter check failed: max magnitude differs")
        return "\n".join(lines)


# Count and max |value| of lora_B tensors in the adapter file. Tensors are
# read one at a time; the file is never loaded whole.
def file_stats(adapter_dir):
    path = pathlib.Path(adapter_dir) / ADAPTER_FILE
    if not path.is_file():
        raise TrlxError(f"{adapter_dir}: no {ADAPTER_FILE}; not an adapter directory")
    count, biggest = 0, 0.0
    try:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                if "lora_B" not in key:
                    continue
                tensor = f.get_tensor(key)
                if tensor.numel() == 0:
                    raise TrlxError(f"{path}: adapter tensor {key} is empty; select a complete adapter")
                count += 1
                biggest = max(biggest, tensor.abs().max().item())
    except (OSError, SafetensorError) as e:
        raise TrlxError(f"{path}: cannot read adapter weights: {e}; select a readable, complete adapter") from e
    return count, biggest


# Count and max |value| of lora_B parameters on a PeftModel, wherever they live.
def model_stats(model):
    count, biggest = 0, 0.0
    for name, param in model.named_parameters():
        if "lora_B" not in name:
            continue
        count += 1
        biggest = max(biggest, param.detach().abs().max().item())
    return count, biggest


# The comparison of SPEC 2.7: file against the loaded PeftModel.
def check(adapter_dir, model):
    file_count, file_max = file_stats(adapter_dir)
    model_count, model_max = model_stats(model)
    return AdapterCheck(file_count, file_max, model_count, model_max)


# task_type from adapter_config.json decides whether the base is loaded as a
# causal or a sequence-classification model; the adapter's saved keys only
# match the class it was trained on.
def task_type(adapter):
    path = pathlib.Path(adapter) / "adapter_config.json"
    try:
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
    except FileNotFoundError:
        raise TrlxError(f"{adapter}: no adapter_config.json; not an adapter directory")
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise TrlxError(f"{path}: cannot read: {e}")
    if not isinstance(document, dict):
        raise TrlxError(f"{path}: expected a JSON object; select a valid adapter directory")
    return document.get("task_type")
