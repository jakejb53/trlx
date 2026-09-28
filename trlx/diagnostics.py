"""Small, operation-owned evidence; never inspect tensor contents after CUDA failure."""

import collections
import contextlib
import os
import re
import sys
import traceback

from dataset import failures

# Collation captures CPU facts before Accelerate transfers batches to a device.
BATCH_CONTEXT = "_trlx_failure_context"


# Shapes and storage sizes are metadata and do not synchronize a device.
def tensor_metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(), "device": str(value.device)}


# This inventory is not peak memory: optimizer state and activations are separate allocations.
def parameter_inventory(model):
    groups = collections.defaultdict(lambda: {"elements": 0, "bytes": 0})
    for parameter in model.parameters():
        key = ("trainable " if parameter.requires_grad else "frozen ") + str(parameter.dtype)
        groups[key]["elements"] += parameter.numel()
        groups[key]["bytes"] += parameter.numel() * parameter.element_size()
    return dict(groups)


# Capture lengths while tensors are on CPU; CUDA-resident input gets shapes only.
def batch_summary(batch):
    import torch

    if not isinstance(batch, dict):
        return {"examples": len(batch), "token_lengths": "unavailable before trainer preprocessing"}
    result = {}
    for name, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        if name.endswith(("input_ids", "attention_mask", "labels", "position_ids")):
            result[name] = tensor_metadata(value)
        if name.endswith("attention_mask") and value.ndim == 2 and value.device.type == "cpu":
            lengths = value.sum(-1).tolist()
            slots = value.numel()
            result[name + "_counts"] = {
                "sequence_lengths": lengths, "actual_token_positions": sum(lengths),
                "padded_token_positions": slots, "padding_percent": round(100 * (slots - sum(lengths)) / slots, 2) if slots else 0,
            }
        if name.endswith("labels") and value.device.type == "cpu":
            result[name + "_unmasked_positions"] = int((value != -100).sum())
    if "attention_mask_counts" not in result:
        positions = batch.get("position_ids")
        if isinstance(positions, torch.Tensor) and positions.ndim == 2 and positions.device.type == "cpu":
            starts = (positions.flatten() == 0).nonzero().flatten().tolist()
            if starts and starts[0] == 0:
                ends = starts[1:] + [positions.numel()]
                result["flattened_sequence_lengths"] = [end - start for start, end in zip(starts, ends)]
        else:
            result["token_lengths"] = "unavailable without a CPU attention mask or sequence boundaries"
    return result


class Collator:
    # Keep the original collation and payload; add only private scalar metadata.
    def __init__(self, wrapped):
        self.wrapped = wrapped

    # Preserve collator metadata used by Trainer save/inspection code.
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "wrapped"), name)

    # Metadata travels with its own batch, including across dataloader worker processes.
    def __call__(self, examples):
        with failures.context(operation="collating batch", examples=len(examples)):
            batch = self.wrapped(examples)
        if isinstance(batch, dict):
            if BATCH_CONTEXT in batch:
                raise failures.Error(f"collator output uses reserved diagnostic key {BATCH_CONTEXT}")
            try:
                summary = batch_summary(batch)
                summary["measurement"] = "CPU collation before device transfer"
            except Exception as error:
                summary = {"diagnostic_error": f"batch summary unavailable: {type(error).__name__}: {error}"}
            batch[BATCH_CONTEXT] = summary
        return batch


# Read allocator bookkeeping only: no device synchronization, mem_get_info, or CUDA initialization.
def allocator_snapshot(device=None):
    import torch

    if not torch.cuda.is_initialized():
        return {"status": "CUDA not initialized; memory measurements unavailable"}
    device = torch.cuda.current_device() if device is None else device
    return {"measurement": "allocator counters after exception; not the allocation-failure instant",
            "local_cuda_device": device,
            "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}


# Evidence collection must never replace the operation's original exception.
def capture_resources(error):
    try:
        # These are rounded failure-time figures supplied by PyTorch, not a
        # reconstruction from later allocator counters. Unknown formats stay raw.
        quantity = r"(\d+(?:\.\d+)?\s+(?:GiB|MiB|KiB|bytes))"
        patterns = {"requested_allocation": r"Tried to allocate " + quantity,
                    "device_capacity": r"total capacity of " + quantity,
                    "free_device_memory": r"of which " + quantity + r" is free",
                    "process_memory": r"this process has " + quantity + r" memory in use",
                    "allocated_by_pytorch": quantity + r" is allocated by PyTorch",
                    "reserved_but_unused": quantity + r" is reserved by PyTorch but unallocated"}
        native = {name: match.group(1) for name, pattern in patterns.items()
                  if (match := re.search(pattern, str(error)))}
        device = re.search(r"GPU (\d+) has a total capacity", str(error))
        if native:
            native["measurement"] = "allocation-failure figures reported by PyTorch (rounded)"
            if device:
                native["local_cuda_device"] = int(device.group(1))
            failures.annotate(error, evidence={"allocation_failure": native})
        failures.annotate(error, evidence={"allocator_after_exception": allocator_snapshot(int(device.group(1)) if device else None)})
    except Exception as diagnostic_error:
        error.add_note(f"Allocator diagnostics unavailable: {type(diagnostic_error).__name__}: {diagnostic_error}")


class ModuleFailure:
    # Store only an identifying name; do not retain inputs, outputs, or the trainer.
    def __init__(self, name):
        self.name = name

    # PyTorch calls always_call hooks while propagating forward exceptions, including recomputation.
    def __call__(self, module, args, kwargs, output):
        error = sys.exception()
        if error is None or output is not None:
            return
        try:
            import torch

            tensors = [tensor_metadata(value) for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
            failures.annotate(error, evidence={"failing_module": {
                "name": self.name, "class": type(module).__name__, "input_tensors": tensors,
            }})
        except Exception as diagnostic_error:
            error.add_note(f"Failing-module diagnostics unavailable: {type(diagnostic_error).__name__}")


# Interpret measured facts only; formulas describe input storage, not an inferred peak allocation.
def interpretation(batch, inventory):
    result = []
    for key, counts in batch.items():
        if key.endswith("attention_mask_counts") and counts["padding_percent"] > 0:
            lengths = counts["sequence_lengths"]
            width = counts['padded_token_positions'] // len(lengths)
            result.append(f"{len(lengths)} sequences are padded to {width} positions each: "
                          f"{counts['actual_token_positions']} actual versus {counts['padded_token_positions']} padded "
                          f"positions ({counts['padding_percent']}% padding). Padding increases activation storage.")
    if "trainable torch.float32" in inventory and "frozen torch.bfloat16" in inventory:
        result.append("Frozen parameters use BF16 and trainable parameters include FP32; BF16 model weights do not "
                      "mean all training operations or temporary tensors use BF16.")
    return result


class TrainerDiagnostics:
    # Preserve the selected TRL/replay trainer; diagnostics wrap, rather than reimplement, its operations.
    def __init__(self, *args, **kwargs):
        with failures.context(operation="constructing trainer"):
            super().__init__(*args, **kwargs)
        self._failure_batch = {}
        self._failure_microbatch = 0
        self._failure_eval_batch = 0
        try:
            self._failure_parameters = parameter_inventory(self.model)
        except Exception as error:
            self._failure_parameters = {"diagnostic_error": f"parameter inventory unavailable: {type(error).__name__}"}
        from torch.utils.data import IterableDataset

        config = self.accelerator.dataloader_config
        dispatch = config.dispatch_batches
        evaluations = self.eval_dataset.values() if isinstance(self.eval_dataset, dict) else [self.eval_dataset]
        dispatch = any(isinstance(dataset, IterableDataset) for dataset in [self.train_dataset, *evaluations]) if dispatch is None else dispatch
        self._failure_batch_limitation = None
        if config.split_batches or dispatch:
            # Accelerate concatenates/slices all payload fields in these modes. Do not
            # inject scalar diagnostics into that tensor-only protocol or mislabel lengths.
            self._failure_batch_limitation = "CPU token counts unavailable with dispatched/split batches; device tensor shapes remain available"
        else:
            self.data_collator = Collator(self.data_collator)
        # Hooks observe exceptions without changing forward results or retaining tensors.
        for name, module in self.model.named_modules():
            module.register_forward_hook(ModuleFailure(name), with_kwargs=True, always_call=True)

    # Consume private metadata before any original trainer or model sees the input mapping.
    def _diagnostic_batch(self, inputs):
        if isinstance(inputs, dict) and BATCH_CONTEXT in inputs:
            return inputs.pop(BATCH_CONTEXT)
        try:
            return batch_summary(inputs)
        except Exception as error:
            return {"diagnostic_error": f"batch summary unavailable: {type(error).__name__}"}

    # Shared capture for training, evaluation, and checkpoint writing; no tensors enter the report.
    @contextlib.contextmanager
    def _diagnostic_operation(self, phase, inputs=None, **context):
        previous = self._failure_batch
        if inputs is not None:
            self._failure_batch = self._diagnostic_batch(inputs)
            if self._failure_batch_limitation:
                self._failure_batch["diagnostic_limitation"] = self._failure_batch_limitation
        try:
            yield
        except Exception as error:
            try:
                import torch

                frames = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
                if phase == "training" and "_engine_run_backward" in frames:
                    phase = "backward recomputation" if "recompute_fn" in frames else "backward"
                failures.annotate(error, context={"phase": phase, "optimizer_step": self.state.global_step, **context})
                if self._failure_batch:
                    failures.annotate(error, evidence={"batch": self._failure_batch})
                if isinstance(error, torch.OutOfMemoryError):
                    failures.annotate(error, evidence={"parameter_inventory_after_construction": self._failure_parameters})
                    capture_resources(error)
                    failures.annotate(error, evidence={"interpretation": interpretation(self._failure_batch, self._failure_parameters)})
            except Exception as diagnostic_error:
                error.add_note(f"Operation diagnostics incomplete: {type(diagnostic_error).__name__}: {diagnostic_error}")
            raise
        finally:
            self._failure_batch = previous

    # Count actual microbatches separately from accumulated optimizer steps.
    def training_step(self, model, inputs, *args, **kwargs):
        self._failure_microbatch += 1
        with self._diagnostic_operation("training", inputs, training_microbatch=self._failure_microbatch):
            return super().training_step(model, inputs, *args, **kwargs)

    # Evaluation has its own batch counter and must not inherit the last training batch.
    def prediction_step(self, model, inputs, *args, **kwargs):
        self._failure_eval_batch += 1
        with self._diagnostic_operation("evaluation", inputs, evaluation_batch=self._failure_eval_batch):
            return super().prediction_step(model, inputs, *args, **kwargs)

    # Capture forward/loss failures before backward, while retaining the outer CPU batch summary.
    def compute_loss(self, model, inputs, *args, **kwargs):
        inputs.pop(BATCH_CONTEXT, None)
        with self._diagnostic_operation("forward/loss computation"):
            return super().compute_loss(model, inputs, *args, **kwargs)

    # Identify checkpoint artifacts using the same directory helper as the installed Trainer.
    def _save_checkpoint(self, model, trial, *args, **kwargs):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        directory = os.path.join(self._get_output_dir(trial=trial), f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        with self._diagnostic_operation("writing checkpoint", destination=directory):
            return super()._save_checkpoint(model, trial, *args, **kwargs)


# A common mixin covers all selected trainer classes, including the replay subclass.
def trainer_class(base):
    return type(f"Diagnostic{base.__name__}", (TrainerDiagnostics, base), {"__module__": __name__})
