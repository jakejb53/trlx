"""Stage 3 check: one (or a few) QAT optimizer steps on the four-layer proxy.

Builds the recipe, disables NeMo's router-bias update, installs the fake-quant
hooks, snapshots the quantized codes and scales of every target and a digest of
every frozen tensor (non-trainable parameters and all buffers), runs the
recipe's own training loop, then requantizes and reports per target:

- codes (bits) and dequantized values and scales that changed, and master
  elements that changed;
- that every frozen tensor is bit-identical to before;
- that the trainable set equals the target set minus the indexer;
- training loss, gradient norm, and learning rate from the recipe's metrics log,
  plus ``lr_group{i}`` for each further optimizer parameter group (the experts
  group under the recipe's ``param_group_overrides``);
- per-step CUDA memory (allocated before and after the step, the step's peak,
  reserved, and the gradient-resident moment: optimizer entry, or the largest
  hook-entry allocation under ``--in-backward``) in ``steps``, with resident
  state by class and series maxima under ``memory``.

Exit nonzero on non-finite loss or gradient norm, a changed frozen tensor, a
trainable-set mismatch, or zero value changes across all targets. Training and
held-out losses are captured per step, with a held-out baseline before the first
update. ``--set KEY=VALUE`` / ``--unset KEY`` override the recipe before it is
built. Launch from the repo root:

  deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy_step \
      --config deepseek/qat/proxy-train.yaml [--out REPORT.json] [--in-backward]

``--in-backward`` steps each parameter inside backward with lagged clipping
(``deepseek.qat.inbackward``) instead of the recipe's post-backward clip and step.

``--expert-selection SCORES.json`` (ESFT-style selection, Handoff 5) trains only
the experts listed per layer under ``layers.<L>.selected`` in a ``route_scores``
report: the stepper restricts each expert parameter to the selected local rows
(no step, no optimizer state for the rest), the flip totals report the selected
experts per layer, and any code or scale change in an unselected expert fails
the run. Requires ``--in-backward``.

Split expert storage (stage 6b) is a recipe setting, ``model.expert_split``
(``--set model.expert_split._target_=deepseek.qat.split_experts.ExpertSplit --set
model.expert_split.scores=SCORES.json`` with a ``route_scores --per-rank``
report): selected experts train as BF16 stacks, unselected experts stay packed
and are covered by the frozen digest; the per-expert tables name rows by the
module's global ids. ``--expert-selection`` is rejected alongside it.

``--dense-start-step S`` (with ``--in-backward``) holds the dense class back
until step S: its gradients are discarded before S and it carries no optimizer
state until then, so the run to S is an experts-only run and dense enters
afterwards at the recipe's dense rate (Pending item 1).

``--dense-tensors attention|shared|none`` declares which dense targets the recipe
trains (the recipe's ``freeze_config`` freezes the rest); frozen targets stay
fake-quantized and are covered by the frozen digest, like the indexer.

``--expert-shares SCORES.json`` (reporting only) joins each routed expert's
token share from a ``route_scores`` report onto the per-expert flip table the
report always carries under ``experts`` (global expert index -> values changed,
elements, selected flag), so flip rate can be read against routing share.

Every step is one table row on rank 0's stderr (``deepseek.qat.progress``):
step, epoch, loss and the running epoch mean, the latest held-out loss with its
change from the previous validation and from baseline, the fraction of dense and
expert values changed since the start, the layers with any expert change, the
per-class gradient norms, and the step's seconds. Every validation adds an
``eval step N`` line. The changed fractions come from ``deepseek.qat.flips``,
which compares each forward's fake quantization against the source on the
device, so they are exact and cost no second pass. With ``--out`` every record
is also appended as one JSON line to ``<out stem>-progress.jsonl``, so a killed
run keeps every measurement taken before the kill. The final report remains the
complete record.

``--export DIR`` (stage 6, stage 7) writes the trained targets as a complete
checkpoint in the layout of the recipe's ``pretrained_model_name_or_path``
(``deepseek.qat.proxy_export``) at every validation and at the end of the run,
under ``DIR/step-N`` with ``DIR/latest`` pointing at the newest; the oldest are
removed beyond ``--keep-exports K`` (default 3). Each export records the hooked
logits on the shared probe token ids as ``<out stem>-step{N}-logits-rank{rank}.pt``
and the replaced tensor names as ``<out stem>-step{N}-replaced.txt`` for
``verify_export --expect @FILE``. Requires ``--out``; ``--force`` replaces an
existing DIR. A failed export stops the run on every rank.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q
from deepseek.qat import inbackward
from deepseek.qat import proxy_export
from deepseek.qat.export import ExportError
from deepseek.qat.flips import FlipCounter
from deepseek.qat.freeze import freeze_gate_bias
from deepseek.qat.progress import Progress, StepTable
from deepseek.qat.overrides import add_override_arguments, apply_overrides
from deepseek.qat.proxy import local_expert_ids, logits as hooked_logits, logits_path, probe_token_ids


# One dense and one expert chunk from the inner optimizer's parameters (all
# groups: dense and experts may sit in separate groups), by size.
def _sample_inner_params(inner):
    params = [p for g in inner.param_groups for p in g["params"] if p.grad is not None]
    dense = [p for p in params if p.ndim == 2]
    expert = [p for p in params if p.ndim == 3]
    return [dense[0], expert[0]] if dense and expert else params[:2]


def _canonical(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


# Optimizer group index of every parameter by canonical name. Must run before any
# forward: FSDP2 with resharding off leaves module attributes pointing at the
# unsharded parameter objects after a forward, so identity against the sharded
# objects the optimizer registered fails from then on; names are stable.
def group_by_name(model, optimizer) -> dict[str, int]:
    group_of_id = {id(q): gi for gi, group in enumerate(optimizer.param_groups) for q in group["params"]}
    return {_canonical(n): group_of_id[id(p)] for n, p in model.named_parameters() if id(p) in group_of_id}


# The learning rate currently set on the group that owns the named parameter.
def _param_lr(optimizer, groups: dict[str, int], name: str) -> float:
    if name not in groups:
        raise RuntimeError(f"{name} is not in any optimizer group")
    return float(optimizer.param_groups[groups[name]]["lr"])


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


# Materialize a parameter consistently before and after training. FSDP2 leaves
# its parameters unsharded after a forward when resharding is off, so dense
# parameters are compared as full tensors (a collective; every rank iterates
# in the same order). Routed-expert parameters are outside FSDP and sharded over
# the expert-parallel mesh, so they are compared as local shards.
def _materialize(name: str, t: torch.Tensor) -> torch.Tensor:
    if ".mlp.experts." in name:
        return _local(t)
    return t.full_tensor() if isinstance(t, DTensor) else t


# Packed codes and scales for a target tensor in its HF orientation, on CPU.
def quantized_state(name: str, tensor: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    w = _materialize(name, tensor).detach()
    if kind == "fp8":
        c, s = Q.quantize_fp8_blocks(w)
        return c.view(torch.uint8).cpu(), s.view(torch.uint8).cpu()
    e = w.shape[0]
    codes, scales = [], []
    for i in range(e):
        if kind == "fp4_gate_up":
            inter = w.shape[2] // 2
            for lo, hi in ((0, inter), (inter, 2 * inter)):
                p, s = FQ.quantize_fp4(w[i, :, lo:hi].T)
                codes.append(p.cpu()); scales.append(s.view(torch.uint8).cpu())
        else:
            p, s = FQ.quantize_fp4(w[i].T)
            codes.append(p.cpu()); scales.append(s.view(torch.uint8).cpu())
    return torch.stack(codes), torch.stack(scales)


def _digest(name: str, t: torch.Tensor) -> str:
    return hashlib.sha256(_materialize(name, t).detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


# Every tensor the export copies from the source unchanged: non-trainable
# parameters and all buffers. Buffers are included because NeMo mutates the
# router correction bias buffer unless freeze_gate_bias has disabled it, and
# a parameter-only check cannot see that.
def _frozen_tensors(model):
    for n, p in model.named_parameters():
        if not p.requires_grad:
            yield n, p
    for n, b in model.named_buffers():
        yield n, b


GB = 1024**3


STEP_EXPORT = re.compile(r"^step-(\d+)$")


# Step exports under an export root, oldest first, as (step, directory name).
def step_exports(root: str) -> list[tuple[int, str]]:
    found = []
    for name in os.listdir(root):
        m = STEP_EXPORT.match(name)
        if m and os.path.isdir(os.path.join(root, name)) and not os.path.islink(os.path.join(root, name)):
            found.append((int(m.group(1)), name))
    return sorted(found)


# Keep the newest `keep` step exports and remove the rest; returns the removed
# names. Staging leftovers and `latest` are not step exports and are untouched.
def retain_exports(root: str, keep: int) -> list[str]:
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")
    removed = []
    for _, name in step_exports(root)[:-keep]:
        shutil.rmtree(os.path.join(root, name))
        removed.append(name)
    return removed


# `latest` is a relative symlink to the newest step export, replaced atomically.
def point_latest(root: str, name: str) -> None:
    link = os.path.join(root, "latest")
    tmp = link + ".tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(name, tmp)
    os.replace(tmp, link)


# Per-step CUDA memory samples around the recipe's optimizer step. The allocator
# peak is reset at every step start so each step has its own peak, and the
# maximum across resets is folded into running_peak so the whole-run figure
# (setup, steps, and validation between steps) keeps its meaning. Samples are
# bytes; the report converts to GB. Reads of the allocator stats never synchronize.
class MemoryMeter:
    def __init__(self, device):
        self.device = device
        self.running_peak = torch.cuda.max_memory_allocated(device)
        self._current: dict | None = None

    def _fold_peak(self) -> None:
        self.running_peak = max(self.running_peak, torch.cuda.max_memory_allocated(self.device))

    def begin_step(self) -> None:
        self._fold_peak()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._current = {"mem_before_step": torch.cuda.memory_allocated(self.device)}

    # Attach a sample taken inside the step (optimizer entry, hook peak).
    def note(self, key: str, value: int) -> None:
        if self._current is None:
            raise RuntimeError(f"memory sample {key} taken outside a step")
        self._current[key] = value

    # Close the step and return its samples as GB metrics with 3-decimal (MB) resolution.
    def end_step(self) -> dict[str, float]:
        if self._current is None:
            raise RuntimeError("end_step without begin_step")
        sample = self._current
        sample["mem_peak_step"] = torch.cuda.max_memory_allocated(self.device)
        sample["mem_after_step"] = torch.cuda.memory_allocated(self.device)
        sample["mem_reserved_after_step"] = torch.cuda.memory_reserved(self.device)
        self._fold_peak()
        self._current = None
        return {f"{k}_gb": round(v / GB, 3) for k, v in sample.items()}

    # Whole-run allocator peak so far, in bytes.
    def peak(self) -> int:
        self._fold_peak()
        return self.running_peak


# Bytes behind a tensor, following tensor subclasses (torchao's OptimState8bit
# reports a float32 appearance dtype, so numel times element_size overstates its
# codes four times) and counting each storage once, so a storage reached through
# two state entries is not double counted.
def _tensor_bytes(t: torch.Tensor, seen: set[int]) -> int:
    flatten = getattr(t, "__tensor_flatten__", None)
    if flatten is not None:
        names, _ = flatten()
        return sum(_tensor_bytes(getattr(t, n), seen) for n in names)
    ptr = t.untyped_storage().data_ptr()
    if ptr in seen:
        return 0
    seen.add(ptr)
    return t.numel() * t.element_size()


# Resident GPU state on this rank between steps, in bytes: parameters by class
# (trainable dense FP8 targets, trainable expert FP4 targets, frozen), buffers,
# and the inner optimizer's state (torchao allocates it lazily, so this is zero
# before the first step). Parameters are counted as local shards, so the caller
# must sample while FSDP2 parameters are sharded: after a backward, not after a
# bare forward, which with reshard_after_forward off leaves them unsharded.
# Dense and frozen FSDP2 parameters also get their unsharded size, resident
# through forward and backward. Expert parameters are sharded over the expert
# mesh and never materialized in full.
def resident_bytes(model, inner) -> dict[str, int]:
    out = {"trainable_dense_shard": 0, "trainable_dense_unsharded": 0, "trainable_expert_shard": 0,
           "frozen_shard": 0, "frozen_unsharded": 0, "buffers": 0, "optimizer_state": 0}
    for n, p in model.named_parameters():
        shard = _local(p).numel() * p.element_size()
        if ".mlp.experts" in n:
            out["trainable_expert_shard" if p.requires_grad else "frozen_shard"] += shard
            continue
        cls = "trainable_dense" if p.requires_grad else "frozen"
        out[f"{cls}_shard"] += shard
        out[f"{cls}_unsharded"] += p.numel() * p.element_size()
    for _, b in model.named_buffers():
        out["buffers"] += _local(b).numel() * b.element_size()
    seen: set[int] = set()
    for st in inner.state.values():
        for v in st.values():
            if isinstance(v, torch.Tensor):
                out["optimizer_state"] += _tensor_bytes(v, seen)
    return out


# Elements whose dequantized value differs between two packed states from
# quantized_state. Values, not bits: the FP4 +0/-0 pair (codes 0 and 8)
# dequantizes identically and must not count. FP4 is walked per expert (and per
# gate/up half) so no full-size float32 tensor is ever held for an expert stack.
def value_changes(before: tuple[torch.Tensor, torch.Tensor], after: tuple[torch.Tensor, torch.Tensor], kind: str) -> int:
    c0, s0 = before
    c1, s1 = after
    if kind == "fp8":
        v0 = Q.dequantize_fp8_blocks(c0.view(torch.float8_e4m3fn), s0.view(torch.float8_e8m0fnu))
        v1 = Q.dequantize_fp8_blocks(c1.view(torch.float8_e4m3fn), s1.view(torch.float8_e8m0fnu))
        return int((v0 != v1).sum())
    return sum(entry_value_changes(before, after))


# Value changes per stacked entry of an FP4 state (one entry per expert for
# down_projs, two per expert for gate_and_up_projs: its w1 and w3 halves).
def entry_value_changes(before: tuple[torch.Tensor, torch.Tensor], after: tuple[torch.Tensor, torch.Tensor]) -> list[int]:
    c0, s0 = before
    c1, s1 = after
    out = []
    for i in range(c0.shape[0]):
        v0 = Q.dequantize_fp4_rows(c0[i], s0[i].view(torch.float8_e8m0fnu))
        v1 = Q.dequantize_fp4_rows(c1[i], s1[i].view(torch.float8_e8m0fnu))
        out.append(int((v0 != v1).sum()))
    return out


# Group per-entry counts by local expert: consecutive entries belong to one expert.
def per_expert(entries: list[int], n_local: int) -> list[int]:
    if n_local <= 0 or len(entries) % n_local:
        raise ValueError(f"{len(entries)} entries do not divide into {n_local} experts")
    k = len(entries) // n_local
    return [sum(entries[i * k : (i + 1) * k]) for i in range(n_local)]


# Per-expert flip rows for one expert parameter: global expert index ->
# (values changed, value elements), from per-entry counts grouped by local row,
# named by the global id of each local row.
def expert_breakdown(entries: list[int], ids: list[int], elements_per_expert: int) -> dict[int, tuple[int, int]]:
    return {e: (n, elements_per_expert) for e, n in zip(ids, per_expert(entries, len(ids)))}


# Read each layer's token shares from a route_scores report, for reporting only.
def load_shares(path: str, layers: list[int], n_experts: int) -> dict[int, list[float]]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    out: dict[int, list[float]] = {}
    for layer in layers:
        entry = report.get("layers", {}).get(str(layer))
        if entry is None or len(entry.get("token_scores", [])) != n_experts:
            raise ValueError(f"{path}: no {n_experts}-expert token_scores for layer {layer}")
        out[layer] = [float(v) for v in entry["token_scores"]]
    return out


# Read an ESFT-style selection from a route_scores report: {layer: sorted global
# expert ids} for exactly the model's layers. A missing layer or an id outside
# the expert range is a configuration error.
def load_selection(path: str, layers: list[int], n_experts: int) -> dict[int, list[int]]:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if int(report.get("n_routed_experts", n_experts)) != n_experts:
        raise ValueError(f"{path}: scored {report.get('n_routed_experts')} experts, model has {n_experts}")
    out: dict[int, list[int]] = {}
    for layer in layers:
        entry = report.get("layers", {}).get(str(layer))
        if entry is None or "selected" not in entry:
            raise ValueError(f"{path}: no selection for layer {layer}")
        ids = sorted(set(int(e) for e in entry["selected"]))
        if any(not (0 <= e < n_experts) for e in ids):
            raise ValueError(f"{path}: layer {layer} selects an expert outside [0, {n_experts})")
        out[layer] = ids
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="write the rank-0 report here as JSON")
    parser.add_argument("--in-backward", action="store_true",
                        help="step each parameter inside backward with lagged clipping at the recipe's max_norm")
    add_override_arguments(parser)
    parser.add_argument("--per-class-clip", action="store_true",
                        help="with --in-backward: clip dense (attention, shared experts) and routed experts by separate norms")
    parser.add_argument("--dense-every", type=int, default=1, metavar="N",
                        help="with --in-backward: update dense parameters on 1 step in N (gradients discarded otherwise, "
                             "Adam moments untouched); experts update every step; implies --per-class-clip")
    parser.add_argument("--dense-start-step", type=int, default=0, metavar="S",
                        help="with --in-backward: dense parameters are not updated before step S (gradients discarded, no optimizer "
                             "state until then); experts update from step 0; implies --per-class-clip")
    parser.add_argument("--trainable-subset", choices=("all", "dense", "experts"), default="all",
                        help="which target classes the recipe is expected to train (ablations freeze the rest via --set); the exact-set check uses it")
    parser.add_argument("--dense-tensors", choices=("all", "attention", "shared", "none"), default="all",
                        help="which dense (FP8) targets the recipe is expected to train: all, the attention projections only, the shared "
                             "experts only, or none; the recipe's freeze_config must freeze the rest, which the exact-set check verifies")
    parser.add_argument("--expert-shares", default=None, metavar="SCORES.json",
                        help="reporting only: join each expert's token share from a route_scores report onto the per-expert flip table")
    parser.add_argument("--expert-selection", default=None, metavar="SCORES.json",
                        help="with --in-backward: train only the experts selected per layer in a route_scores report")
    parser.add_argument("--export", default=None, metavar="DIR",
                        help="export the trained targets as a complete checkpoint in the source layout at every validation and at "
                             "the end, under DIR/step-N with DIR/latest pointing at the newest (requires --out)")
    parser.add_argument("--keep-exports", type=int, default=3, metavar="K",
                        help="with --export: keep the newest K step exports and remove older ones (default 3)")
    parser.add_argument("--force", action="store_true", help="replace an existing --export directory")
    args = parser.parse_args(argv)
    if args.export is not None and not args.out:
        parser.error("--export requires --out: probe logits and the replaced-tensor lists are saved beside the report")
    if args.keep_exports < 1:
        parser.error(f"--keep-exports must be at least 1, got {args.keep_exports}")
    if args.export is not None and os.path.lexists(args.export) and not args.force:
        parser.error(f"--export: {args.export} exists; pass --force to replace it")
    if args.dense_every < 1:
        parser.error(f"--dense-every must be a positive integer, got {args.dense_every}")
    if args.dense_every > 1 and not args.in_backward:
        parser.error("--dense-every needs --in-backward: the update period lives in the stepper")
    if args.dense_start_step < 0:
        parser.error(f"--dense-start-step must be a nonnegative integer, got {args.dense_start_step}")
    if args.dense_start_step > 0 and not args.in_backward:
        parser.error("--dense-start-step needs --in-backward: the start step lives in the stepper")
    if args.expert_selection is not None and not args.in_backward:
        parser.error("--expert-selection needs --in-backward: row restriction lives in the stepper")

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    t0 = time.monotonic()
    cfg = load_yaml_config(args.config)
    overrides = apply_overrides(cfg, args.set, args.unset)
    # The export copies untouched tensors from the checkpoint the model loads,
    # so a train-on-export run exports from that export, not the original snapshot.
    snapshot = cfg.model.config.pretrained_model_name_or_path
    if args.export is not None and not os.path.isfile(os.path.join(snapshot, "model.safetensors.index.json")):
        raise SystemExit(f"--export: the recipe's model path {snapshot} has no model.safetensors.index.json to export from")
    # Split expert storage (stage 6b) freezes unselected experts in packed stacks;
    # the stepper's row restriction is the single-stack form of the same selection.
    if args.expert_selection is not None and cfg.model.get("expert_split", None) is not None:
        raise SystemExit("--expert-selection cannot be combined with model.expert_split: the split storage already fixes the trained experts")
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = recipe.model_parts[0]
    # The export root is created (or, with --force, replaced) once, by rank 0,
    # before anything else; step exports are staged inside it.
    export_root = os.path.abspath(args.export) if args.export is not None else None
    if export_root is not None and rank == 0:
        if os.path.lexists(export_root):
            shutil.rmtree(export_root) if os.path.isdir(export_root) and not os.path.islink(export_root) else os.remove(export_root)
        os.makedirs(export_root)
    if dist.is_initialized():
        dist.barrier()
    prefix = os.path.splitext(args.out)[0] if args.out else None
    progress = Progress(f"{prefix}-progress.jsonl" if prefix else None, rank)
    progress.record("run", {"config": args.config, "overrides": overrides, "in_backward": args.in_backward,
                            "dense_every": args.dense_every, "dense_start_step": args.dense_start_step,
                            "export": export_root, "keep_exports": args.keep_exports})
    # Disable NeMo's per-step router-bias update before any training step.
    gate_factor_before, gates_with_bias = freeze_gate_bias(model)
    state = FQ.install(model)
    report: dict = {"rank": rank, "setup_seconds": round(time.monotonic() - t0, 1)}
    report["gate_bias_update_factor_before"] = gate_factor_before
    report["gates_with_correction_bias"] = gates_with_bias
    report["in_backward"] = args.in_backward
    report["overrides"] = overrides

    # Update-in-backward: the stepper takes over clipping and stepping. The
    # recipe's own step is wrapped so the exact global norm replaces the zero
    # NeMo reports once its clip is disabled.
    # Expert selection: global ids per layer from the scores file, mapped to this
    # rank's local rows through the expert-parallel offset. Parameters are
    # looked up by canonical name so wrapper segments do not matter.
    selection: dict[int, list[int]] | None = None
    local_selected: dict[str, list[int]] = {}  # target param name -> local rows
    if args.expert_selection is not None:
        layer_ids = sorted(int(k) for k in model.model.layers.keys())
        selection = load_selection(args.expert_selection, layer_ids, int(model.config.n_routed_experts))
        for t in state.targets:
            if not t.name.endswith(".mlp.experts"):
                continue
            layer = int(t.name.split(".")[2])
            for attr in t.params:
                ids = local_expert_ids(t.module, getattr(t.module, attr))
                local_selected[f"{t.name}.{attr}"] = [i for i, e in enumerate(ids) if e in selection[layer]]
        report["expert_selection"] = args.expert_selection
        report["selected_per_layer"] = {str(k): len(v) for k, v in selection.items()}
        report["selected_total"] = sum(len(v) for v in selection.values())
    # Split storage: the selection is fixed by the model; report it the same way.
    split = getattr(next(iter(model.model.layers.values())).mlp.experts, "split", None)
    if split is not None:
        report["expert_split"] = split.source
        report["selected_per_layer"] = {str(k): len(v) for k, v in sorted(split.layers.items())}
        report["selected_total"] = sum(len(v) for v in split.layers.values())
        report["packed_per_rank"] = {str(k): split.per_rank_total - split.selected_per_rank(k) for k in sorted(split.layers)}

    def rows_of(name: str) -> list[int] | None:
        return local_selected.get(_canonical(name))

    # Shares are reporting-only but a bad file must fail before training starts.
    shares: dict[int, list[float]] | None = None
    if args.expert_shares is not None:
        shares = load_shares(args.expert_shares, sorted(int(k) for k in model.model.layers.keys()), int(model.config.n_routed_experts))
        report["expert_shares"] = args.expert_shares

    stepper = None
    if args.in_backward:
        stepper = inbackward.from_recipe(recipe, recipe.max_grad_norm, per_class=args.per_class_clip, dense_every=args.dense_every,
                                         dense_start=args.dense_start_step, rows_of=rows_of if selection is not None else None)
        report["per_class_clip"] = args.per_class_clip or args.dense_every > 1 or args.dense_start_step > 0
        report["dense_every"] = args.dense_every
        report["dense_start_step"] = args.dense_start_step

    # Every step is wrapped for memory sampling (Handoff 2: steady-state peaks
    # and the update-in-backward saving need per-step figures). Under
    # update-in-backward the recipe's clip is disabled and the stepper owns
    # clipping, so the recipe's max_grad_norm is withheld; otherwise it passes
    # through unchanged. Memory keys join the step metrics so they stay aligned
    # with the captured loss and norm of the same step.
    meter = MemoryMeter(torch.cuda.current_device())
    original_step = recipe._run_train_optim_step
    # Resident state is sampled once, at the end of the first step: FSDP2 has
    # resharded its dense parameters after backward and torchao's lazily
    # allocated state exists. After training it would be wrong, because the last
    # validation forward leaves FSDP2 parameters unsharded with resharding off.
    resident: dict[str, int] = {}

    def stepping(batches, max_grad_norm=None):
        t_step = time.monotonic()
        meter.begin_step()
        # The step's forward quantizes every trained tensor; the counter compares
        # that against the baselines, so these totals describe the masters as the
        # step began (after the previous update).
        counter.begin_pass()
        sample = original_step(batches, None if stepper is not None else max_grad_norm)
        if not resident:
            resident.update(resident_bytes(model, recipe.optimizer[0].inner))
        if stepper is not None:
            sample.metrics["grad_norm"] = stepper.end_step()
            # Clipping classes' coefficients (one global coefficient unless per-class
            # clipping is on); the reporting norms for dense and experts always.
            for cls in stepper.classes:
                suffix = "" if cls == "all" else f"_{cls}"
                sample.metrics[f"clip_coef_next{suffix}"] = stepper.coefs[cls]
            for cls, norm in stepper.last_report_norms.items():
                sample.metrics[f"grad_norm_{cls}"] = norm
            for cls, stats in stepper.last_drift.items():
                sample.metrics[f"drift_{cls}"] = stats["mean_abs_drift"]
                sample.metrics[f"consistency_{cls}"] = stats["consistency"]
            # Per-tensor drift and sign consistency, the selection data for the
            # next run: in the progress file every step, in the report at the end.
            progress.record("drift_by_tensor", {"step": stepper.step_count - 1, "tensors": stepper.last_drift_by_param})
            meter.note("mem_peak_hook", stepper.last_peak_hook_allocated)
        sample.metrics.update(counter.totals())
        sample.metrics["step_seconds"] = round(time.monotonic() - t_step, 1)
        # Rates of the further parameter groups (group 0 is NeMo's `lr`), read at
        # the same moment NeMo reads `lr`: after its scheduler has advanced, so
        # every rate metric describes the next step.
        for i, group in enumerate(recipe.optimizer[0].param_groups[1:], 1):
            sample.metrics[f"lr_group{i}"] = float(group["lr"])
        sample.metrics.update(meter.end_step())
        return sample

    recipe._run_train_optim_step = stepping

    # Trainable set must be exactly the targets minus the indexer.
    # Expected trainable set: the targets minus the indexer, restricted to the
    # requested class for ablation arms. Frozen targets stay fake-quantized; only
    # the exact-set check and the flip totals follow this selection.
    def in_subset(name: str) -> bool:
        is_expert = ".mlp.experts" in name
        if not (args.trainable_subset == "all" or (args.trainable_subset == "experts") == is_expert):
            return False
        if is_expert:
            return True
        # Dense selection by role (the second selection axis beside the experts'):
        # attention projections or shared experts, the two kinds of dense target.
        role = "attention" if ".self_attn." in name else "shared"
        return args.dense_tensors in ("all", role)

    target_params = {f"{t.name}.{a}" for t in state.targets for a in t.params if "indexer" not in t.name and in_subset(t.name)}
    report["trainable_subset"] = args.trainable_subset
    report["dense_tensors"] = args.dense_tensors
    # Activation checkpointing and compile wrappers add path segments that the
    # target names (taken from the unwrapped hierarchy) do not carry.
    trainable = {_canonical(n) for n, p in model.named_parameters() if p.requires_grad}
    param_groups_by_name = group_by_name(model, recipe.optimizer[0])
    report["trainable_params"] = len(trainable)
    report["trainable_numel"] = sum(_local(p).numel() for n, p in model.named_parameters() if p.requires_grad)
    report["trainable_not_target"] = sorted(trainable - target_params)
    report["target_not_trainable"] = sorted(target_params - trainable)

    # Master copies are kept only for a sample of targets (one dense projection
    # and the gate/up expert tensor per layer): a full copy is 26 GB per rank of
    # host memory, which with two ranks and NeMo's load staging exceeds this host.
    sampled = {t.name for t in state.targets if t.name.endswith(".self_attn.wq_b") or t.name.endswith(".mlp.experts")}
    masters = {}
    for t in state.targets:
        for attr, kind in t.params.items():
            name = f"{t.name}.{attr}"
            if t.name in sampled and attr != "down_projs":
                masters[name] = _materialize(name, getattr(t.module, attr)).detach().cpu().clone()
    # Flip accounting (deepseek.qat.flips): baselines recorded now from the
    # untrained weights; from here every forward's fake quantization is compared
    # against them on the device, so the counts exist after every step.
    # Every routed expert of every backbone layer, from the model config (w1, w3:
    # hidden x inter each; w2: inter x hidden), whether selected or packed.
    mcfg = model.config
    expert_class_elements = 3 * int(mcfg.hidden_size) * int(mcfg.moe_intermediate_size) * int(mcfg.n_routed_experts) * len(model.model.layers)
    counter = FlipCounter(state, target_params, _materialize, torch.device(torch.cuda.current_device()),
                          expert_class_elements=expert_class_elements)
    steps_per_epoch = len(recipe.dataloader) if hasattr(recipe.dataloader, "__len__") else None
    table = StepTable(int(cfg.step_scheduler.max_steps), steps_per_epoch, len(counter.layers))
    frozen_before = {n: _digest(n, t) for n, t in _frozen_tensors(model)}
    report["frozen_params"] = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    report["frozen_buffers"] = sum(1 for _ in model.named_buffers())

    # Capture every training step's metrics in memory on the way to the recipe's logger.
    captured: list[dict] = []
    original_log = recipe.log_train_metrics

    def capturing_log(log_data):
        m = {k: (float(v) if isinstance(v, (int, float, torch.Tensor)) else v) for k, v in log_data.metrics.items()}
        m["step"] = int(log_data.step)
        captured.append(m)
        progress.record("step", m)
        norms = {c: m[f"grad_norm_{c}"] for c in ("dense", "expert") if f"grad_norm_{c}" in m} or None
        drift = {c: (m[f"drift_{c}"], m[f"consistency_{c}"]) for c in ("dense", "expert") if f"drift_{c}" in m} or None
        progress.say(table.row(m["step"], m.get("loss"), m if "dense_value_elements" in m else None, norms, m.get("step_seconds"), drift))
        return original_log(log_data)

    recipe.log_train_metrics = capturing_log

    # Held-out loss: capture NeMo's validation events, and measure the baseline
    # before any update since NeMo's loop has no step-zero evaluation. The
    # validation method switches to eval mode and leaves it to the loop to
    # restore train mode, so that is done here for the baseline.
    captured_val: list[dict] = []
    original_val_log = recipe.log_val_metrics

    # One checkpoint per validation and at the end: the trained targets exported
    # under <export>/step-N with the probe logits and replaced list beside the
    # report, `latest` pointing at the newest, the oldest removed beyond
    # --keep-exports. Killing the run after any validation leaves a complete,
    # verifiable checkpoint. Every rank joins the export collectives and the
    # failure flag is reduced, so the outcome is agreed and no rank is left
    # waiting on a collective. Returns the error message, or None on success.
    exports: list[dict] = []
    last_export_step: list[int | None] = [None]

    def export_at(step: int) -> str | None:
        dest = os.path.join(export_root, f"step-{step}")
        step_prefix = f"{prefix}-step{step}"
        ids = probe_token_ids(int(model.config.vocab_size)).to(next(model.parameters()).device)
        torch.save(hooked_logits(model, ids).cpu(), logits_path(step_prefix, rank))
        t2 = time.monotonic()
        failure = None
        try:
            exported = proxy_export.export_trained(state, target_params, snapshot, dest, force=False)
        except ExportError as e:
            failure = str(e)
        flag = torch.tensor([0 if failure is None else 1], device=torch.cuda.current_device())
        if dist.is_initialized():
            dist.all_reduce(flag)
        if int(flag.item()):
            failure = failure or "export failed on another rank"
            progress.record("export_error", {"step": step, "path": dest, "error": failure})
            progress.say(f"export at step {step} failed: {failure}")
            return failure
        last_export_step[0] = step
        if rank == 0:
            replaced_file = f"{step_prefix}-replaced.txt"
            with open(replaced_file, "w", encoding="utf-8") as f:
                f.write("".join(f"{n}\n" for n in exported.pop("replaced")))
            point_latest(export_root, f"step-{step}")
            removed = retain_exports(export_root, args.keep_exports)
            # A removed export's probe logits (about 265 MB per rank) and replaced
            # list are useless without it; they go with it.
            world = dist.get_world_size() if dist.is_initialized() else 1
            for name in removed:
                gone_prefix = f"{prefix}-step{name.split('-', 1)[1]}"
                for path in [f"{gone_prefix}-replaced.txt", *(logits_path(gone_prefix, r) for r in range(world))]:
                    if os.path.exists(path):
                        os.remove(path)
            exported.update({"step": step, "replaced_file": replaced_file, "probe_logits_prefix": step_prefix,
                             "export_seconds": round(time.monotonic() - t2, 1), "removed": removed})
            exports.append(exported)
            progress.record("export", exported)
            progress.say(f"exported step {step} to {dest} ({exported['replaced_count']} tensors replaced, {exported['export_seconds']} s; "
                         f"latest -> step-{step}; removed {', '.join(exported['removed']) or 'none'})")
        return None

    # Every validation epoch is one counting pass, so its totals describe the
    # masters after the last update (the training pass describes the step start).
    original_val_epoch = recipe._run_validation_epoch

    def counting_val_epoch(*a, **kw):
        counter.begin_pass()
        return original_val_epoch(*a, **kw)

    recipe._run_validation_epoch = counting_val_epoch

    def capturing_val_log(val_name, log_data, metric_logger=None):
        entry = {"step": int(log_data.step), "name": val_name, "val_loss": float(log_data.metrics["val_loss"])}
        entry.update(counter.totals())
        captured_val.append(entry)
        progress.record("validation", entry)
        table.note_validation(entry["val_loss"])
        progress.say(table.validation_line(entry["step"]))
        if export_root is not None:
            # The validation left the model in eval mode, which the probe needs.
            failure = export_at(entry["step"])
            if failure is not None:
                # Mid-run, a broken destination would waste every further step.
                raise SystemExit(f"export at step {entry['step']} failed; stopping the run: {failure}")
        return original_val_log(val_name, log_data, metric_logger)

    recipe.log_val_metrics = capturing_val_log
    for val_name, val_loader in recipe.val_dataloaders.items():
        sample = recipe._run_validation_epoch(val_loader)
        captured_val.append({"step": -1, "name": val_name, "val_loss": float(sample.metrics["val_loss"])})
        progress.record("validation", captured_val[-1])
        table.note_validation(captured_val[-1]["val_loss"], baseline=True)
        progress.say(f"held-out baseline: loss {captured_val[-1]['val_loss']:.4f}")
    for mp in recipe.model_parts:
        mp.train()

    # Diagnostic: before the inner optimizer steps, measure how many gradient
    # elements fall below torchao's smallest 8-bit second-moment level relative
    # to their block's maximum, for the configured and smaller block sizes. Such
    # elements get a zero second moment and an update bounded only by eps.
    grad_stats: dict = {}
    inner = recipe.optimizer[0].inner
    inner_step = inner.step

    def measured_step(closure=None):
        # Every gradient of the step is resident here (post-backward, pre-step):
        # the figure update-in-backward removes.
        meter.note("mem_at_optimizer_step", torch.cuda.memory_allocated(meter.device))
        min_level = 3.25e-7
        for name, chunk in zip(["dense", "expert"], _sample_inner_params(inner)):
            g2 = chunk.grad.detach().float().square().reshape(-1)
            entry = {"elements": g2.numel(), "zero_grad_frac": float((g2 == 0).float().mean())}
            for bs in (256, 64, 32):
                n = g2.numel() - g2.numel() % bs
                blocks = g2[:n].reshape(-1, bs)
                thresh = blocks.amax(dim=1, keepdim=True) * min_level
                entry[f"flush_frac_block{bs}"] = float(((blocks > 0) & (blocks < thresh)).float().mean())
            grad_stats[name] = entry
        return inner_step(closure)

    # Under update-in-backward each inner step sees one parameter's chunks, so
    # the sampled diagnostic has no stable pair to measure; it is skipped.
    if stepper is None:
        inner.step = measured_step

    t1 = time.monotonic()
    recipe.run_train_validation_loop()
    report["grad_stats"] = grad_stats
    report["train_seconds"] = round(time.monotonic() - t1, 1)
    # Whole-run peak including setup and validation; the per-step peaks are in steps.
    report["peak_memory_gb"] = round(meter.peak() / GB, 2)
    step_keys = ("step", "loss", "grad_norm", "lr", "clip_coef_next", "drift_dense", "drift_expert", "consistency_dense", "consistency_expert", "step_seconds", "grad_norm_dense", "grad_norm_expert", "clip_coef_next_dense", "clip_coef_next_expert",
                 "mem_before_step_gb", "mem_peak_step_gb", "mem_after_step_gb", "mem_reserved_after_step_gb", "mem_peak_hook_gb", "mem_at_optimizer_step_gb")
    report["steps"] = [{k: v for k, v in m.items() if k in step_keys or k.startswith("lr_group")} for m in captured]
    # Memory summary (Handoff 2). Resident state comes from the end of the first
    # step (see stepping); the series figures summarize the per-step samples
    # above. mem_after_step is the steady-state resident set (parameters,
    # optimizer state; allocator-held but unused memory excluded); peak minus
    # after is the step's transient: activations, unsharded dense parameters,
    # gradients, fake-quant and optimizer temporaries. resident_total is the
    # sum of the sharded classes, buffers, and state, and should agree with
    # after_step_gb_median to within small library allocations.
    mem_steps = [m for m in captured if "mem_peak_step_gb" in m]
    memory: dict = {f"resident_{k}_gb": round(v / GB, 3) for k, v in resident.items()}
    memory["resident_total_gb"] = round(sum(v for k, v in resident.items() if not k.endswith("_unsharded")) / GB, 3)
    memory["steps_measured"] = len(mem_steps)
    if mem_steps:
        after = sorted(m["mem_after_step_gb"] for m in mem_steps)
        memory["peak_step_gb_max"] = max(m["mem_peak_step_gb"] for m in mem_steps)
        memory["after_step_gb_median"] = after[len(after) // 2]
        memory["after_step_gb_last"] = mem_steps[-1]["mem_after_step_gb"]
        memory["transient_gb_max"] = round(max(m["mem_peak_step_gb"] - m["mem_after_step_gb"] for m in mem_steps), 3)
        # One of the two exists per arm: the hook sample under update-in-backward,
        # the optimizer-entry sample otherwise.
        for key, out in (("mem_peak_hook_gb", "peak_hook_gb_max"), ("mem_at_optimizer_step_gb", "at_optimizer_step_gb_max")):
            if key in mem_steps[-1]:
                memory[out] = max(m[key] for m in mem_steps)
    report["memory"] = memory
    report["validation"] = captured_val
    if stepper is not None:
        report["drift_by_tensor"] = {_canonical(n): v for n, v in stepper.last_drift_by_param.items()}
    last = captured[-1] if captured else {}
    report["loss"] = last.get("loss")
    report["grad_norm"] = last.get("grad_norm")

    # Final flip figures: one explicit counting pass over the trained masters
    # (the training pass describes a step's start, and a validation may not have
    # followed the last update), then the per-tensor and per-expert counts.
    counter.refresh(_materialize)
    tensors = counter.per_tensor()
    flips = {}
    total_code_flips = 0
    total_value_changes = 0
    unselected_changed: list[str] = []
    # Per-expert flips on this rank: layer -> global expert -> [values changed,
    # value elements], summed over the gate/up and down parameters. Merged over
    # ranks below so the report covers every expert.
    local_experts: dict[int, dict[int, list[int]]] = {}
    for t in state.targets:
        for attr, kind in t.params.items():
            name = f"{t.name}.{attr}"
            if name not in tensors:
                continue  # frozen targets (the indexer, an ablation's other class) are covered by the frozen digest
            if kind != "fp8":
                layer = int(t.name.split(".")[2])
                ids = local_expert_ids(t.module, getattr(t.module, attr))
                values, codes, scales = counter.per_expert(name)
                if len(ids) != len(values):
                    raise RuntimeError(f"{name}: {len(ids)} local experts but {len(values)} counted rows")
                per = counter.elements_per_expert(name)
                rows = local_experts.setdefault(layer, {})
                for e, n in zip(ids, values):
                    acc = rows.setdefault(e, [0, 0])
                    acc[0] += n
                    acc[1] += per
                if selection is not None and name in local_selected:
                    # Any bit change in an unselected expert's codes or scales is a
                    # frozen-integrity failure, named by global expert index.
                    chosen = set(local_selected[name])
                    for i, e in enumerate(ids):
                        if i not in chosen and (codes[i] or scales[i]):
                            unselected_changed.append(f"{name}[expert {e}]")
            # Digest of the trained master (dense: full tensor; experts: this
            # rank's shard) so two runs can be compared for equality.
            entry = {"master_sha256": _digest(name, getattr(t.module, attr)), **tensors[name]}
            if name in masters:
                m1 = _materialize(name, getattr(t.module, attr)).detach().cpu()
                delta = (m1.float() - masters[name].float()).abs().reshape(-1)
                lr = _param_lr(recipe.optimizer[0], param_groups_by_name, name)
                entry.update({
                    "master_changed": int((masters[name] != m1).sum()),
                    "master_elements": m1.numel(),
                    "delta_max": float(delta.max()),
                    "delta_over_2lr_frac": float((delta > 2 * lr).float().mean()),
                    "delta_over_20lr_frac": float((delta > 20 * lr).float().mean()),
                    "delta_over_200lr_frac": float((delta > 200 * lr).float().mean()),
                })
                del m1, delta
            flips[name] = entry
            total_code_flips += entry["codes_flipped"]
            total_value_changes += entry["values_changed"]
    report["targets"] = flips
    # Merge every rank's local experts, then join shares and selection flags.
    gathered: list = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
    if dist.is_initialized():
        dist.all_gather_object(gathered, local_experts)
    else:
        gathered[0] = local_experts
    experts_report: dict[str, dict[str, dict]] = {}
    for part in gathered:
        for layer, rows in part.items():
            dst = experts_report.setdefault(str(layer), {})
            for e, (n, m) in rows.items():
                entry = {"values_changed": n, "value_elements": m}
                if shares is not None:
                    entry["token_share"] = shares[layer][e]
                if selection is not None:
                    entry["selected"] = e in selection[layer]
                dst[str(e)] = entry
    report["experts"] = {layer: dict(sorted(rows.items(), key=lambda kv: int(kv[0]))) for layer, rows in sorted(experts_report.items(), key=lambda kv: int(kv[0]))}
    report["total_code_flips_in_trainable"] = total_code_flips
    report["total_value_changes_in_trainable"] = total_value_changes
    # One digest over every target's master digest, in target order, so the
    # short rank-1 line can be compared with another run's rank-1 line.
    report["masters_sha256"] = hashlib.sha256("".join(e["master_sha256"] for e in flips.values()).encode()).hexdigest()
    # A tensor absent before training (a buffer created during the step) has no
    # prior digest and is reported as changed; the frozen set must be fixed.
    report["frozen_changed"] = sorted(n for n, t in _frozen_tensors(model) if _digest(n, t) != frozen_before.get(n))
    report["indexer_changed"] = sorted(n for n in report["frozen_changed"] if "indexer" in n)
    report["unselected_experts_changed"] = unselected_changed

    # Every step's loss and gradient norm must be finite, not only the last.
    finite = bool(captured) and all(
        isinstance(m.get(k), (int, float)) and math.isfinite(m[k]) for m in captured for k in ("loss", "grad_norm")
    )
    ok = (
        finite
        and not report["frozen_changed"]
        and not report["indexer_changed"]
        and not report["unselected_experts_changed"]
        and not report["trainable_not_target"]
        and not report["target_not_trainable"]
        and total_value_changes > 0
    )

    # Final export (stage 6): the probe runs in eval mode on the shared token ids
    # so `proxy --expect-logits` on the reloaded export reproduces it bit for bit.
    # NeMo validates at the end of training, so the last step usually already has
    # its export from the validation hook and is not exported twice. A failed
    # final export is recorded and fails the run without raising, so every rank
    # still reaches the final all-reduce.
    if export_root is not None:
        final_step = captured[-1]["step"] if captured else None
        if final_step is None:
            report["export_error"] = "no training step ran; nothing to export"
            ok = False
        elif last_export_step[0] != final_step:
            model.eval()
            failure = export_at(final_step)
            if failure is not None:
                report["export_error"] = failure
                ok = False
        report["exports"] = exports
        report["export"] = exports[-1] if exports else None
    report["ok"] = ok
    report["elapsed_seconds"] = round(time.monotonic() - t0, 1)
    if rank == 0:
        print(json.dumps(report, indent=1), flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1)
    else:
        print(json.dumps({k: report[k] for k in ("rank", "ok", "frozen_changed", "unselected_experts_changed", "total_value_changes_in_trainable", "masters_sha256")}), flush=True)
    if dist.is_initialized():
        flag = torch.tensor([0 if ok else 1], device=torch.cuda.current_device())
        dist.all_reduce(flag)
        ok = int(flag.item()) == 0
        dist.barrier()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(code)
