"""Stage 3 check: one (or a few) QAT optimizer steps on the four-layer proxy.

Builds the recipe, disables NeMo's router-bias update, installs the fake-quant
hooks, snapshots the quantized codes and scales of every target and a digest of
every frozen tensor (non-trainable parameters and all buffers), runs the
recipe's own training loop, then requantizes and reports per target:

- codes (bits) and dequantized values and scales that changed, and master
  elements that changed;
- that every frozen tensor is bit-identical to before;
- that the trainable set equals the target set minus the indexer;
- training loss and gradient norm from the recipe's metrics log.

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

``--expert-shares SCORES.json`` (reporting only) joins each routed expert's
token share from a ``route_scores`` report onto the per-expert flip table the
report always carries under ``experts`` (global expert index -> values changed,
elements, selected flag), so flip rate can be read against routing share.

``--export DIR`` (stage 6) records the hooked logits on the shared probe token
ids after training, saved as ``<out stem>-logits-rank{rank}.pt``, then writes
the trained targets as a complete checkpoint in the layout of the recipe's
``pretrained_model_name_or_path`` (``deepseek.qat.proxy_export``). The names of
the replaced tensors go to ``<out stem>-replaced.txt`` for ``verify_export
--expect @FILE``. Requires ``--out``; ``--force`` replaces an existing DIR.
"""

import argparse
import hashlib
import json
import math
import os
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
from deepseek.qat.freeze import freeze_gate_bias
from deepseek.qat.overrides import add_override_arguments, apply_overrides
from deepseek.qat.proxy import _ep_offset, logits as hooked_logits, logits_path, probe_token_ids


# One dense and one expert chunk from the inner optimizer's parameter list, by size.
def _sample_inner_params(inner):
    params = [p for p in inner.param_groups[0]["params"] if p.grad is not None]
    dense = [p for p in params if p.ndim == 2]
    expert = [p for p in params if p.ndim == 3]
    return [dense[0], expert[0]] if dense and expert else params[:2]


def _canonical(name: str) -> str:
    return name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


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
# (values changed, value elements), from per-entry counts grouped by local expert.
def expert_breakdown(entries: list[int], n_local: int, offset: int, elements_per_expert: int) -> dict[int, tuple[int, int]]:
    return {offset + i: (n, elements_per_expert) for i, n in enumerate(per_expert(entries, n_local))}


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
    parser.add_argument("--trainable-subset", choices=("all", "dense", "experts"), default="all",
                        help="which targets the recipe is expected to train (ablations freeze the rest via --set); the exact-set check uses it")
    parser.add_argument("--expert-shares", default=None, metavar="SCORES.json",
                        help="reporting only: join each expert's token share from a route_scores report onto the per-expert flip table")
    parser.add_argument("--expert-selection", default=None, metavar="SCORES.json",
                        help="with --in-backward: train only the experts selected per layer in a route_scores report")
    parser.add_argument("--export", default=None, metavar="DIR",
                        help="after training, export the trained targets as a complete checkpoint in the source layout (requires --out)")
    parser.add_argument("--force", action="store_true", help="replace an existing --export directory")
    args = parser.parse_args(argv)
    if args.export is not None and not args.out:
        parser.error("--export requires --out: probe logits and the replaced-tensor list are saved beside the report")
    if args.dense_every < 1:
        parser.error(f"--dense-every must be a positive integer, got {args.dense_every}")
    if args.dense_every > 1 and not args.in_backward:
        parser.error("--dense-every needs --in-backward: the update period lives in the stepper")
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
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = recipe.model_parts[0]
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
                param = getattr(t.module, attr)
                offset = _ep_offset(param)
                n_local = _local(param).shape[0]
                local_selected[f"{t.name}.{attr}"] = [e - offset for e in selection[layer] if offset <= e < offset + n_local]
        report["expert_selection"] = args.expert_selection
        report["selected_per_layer"] = {str(k): len(v) for k, v in selection.items()}
        report["selected_total"] = sum(len(v) for v in selection.values())

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
                                         rows_of=rows_of if selection is not None else None)
        report["per_class_clip"] = args.per_class_clip or args.dense_every > 1
        report["dense_every"] = args.dense_every
        original_step = recipe._run_train_optim_step

        def stepping(batches, max_grad_norm=None):
            sample = original_step(batches, None)
            sample.metrics["grad_norm"] = stepper.end_step()
            # Per-class norms and coefficients when enabled; the global norm stays in grad_norm.
            for cls in stepper.classes:
                suffix = "" if cls == "all" else f"_{cls}"
                sample.metrics[f"clip_coef_next{suffix}"] = stepper.coefs[cls]
                if cls != "all":
                    sample.metrics[f"grad_norm_{cls}"] = stepper.last_norms[cls]
            return sample

        recipe._run_train_optim_step = stepping

    # Trainable set must be exactly the targets minus the indexer.
    # Expected trainable set: the targets minus the indexer, restricted to the
    # requested class for ablation arms. Frozen targets stay fake-quantized; only
    # the exact-set check and the flip totals follow this selection.
    def in_subset(name: str) -> bool:
        is_expert = ".mlp.experts" in name
        return args.trainable_subset == "all" or (args.trainable_subset == "experts") == is_expert

    target_params = {f"{t.name}.{a}" for t in state.targets for a in t.params if "indexer" not in t.name and in_subset(t.name)}
    report["trainable_subset"] = args.trainable_subset
    # Activation checkpointing and compile wrappers add path segments that the
    # target names (taken from the unwrapped hierarchy) do not carry.
    trainable = {_canonical(n) for n, p in model.named_parameters() if p.requires_grad}
    report["trainable_params"] = len(trainable)
    report["trainable_numel"] = sum(_local(p).numel() for n, p in model.named_parameters() if p.requires_grad)
    report["trainable_not_target"] = sorted(trainable - target_params)
    report["target_not_trainable"] = sorted(target_params - trainable)

    # Master copies are kept only for a sample of targets (one dense projection
    # and the gate/up expert tensor per layer): a full copy is 26 GB per rank of
    # host memory, which with two ranks and NeMo's load staging exceeds this host.
    sampled = {t.name for t in state.targets if t.name.endswith(".self_attn.wq_b") or t.name.endswith(".mlp.experts")}
    before = {}
    masters = {}
    for t in state.targets:
        for attr, kind in t.params.items():
            name = f"{t.name}.{attr}"
            before[name] = quantized_state(name, getattr(t.module, attr), kind)
            if t.name in sampled and attr != "down_projs":
                masters[name] = _materialize(name, getattr(t.module, attr)).detach().cpu().clone()
    frozen_before = {n: _digest(n, t) for n, t in _frozen_tensors(model)}
    report["frozen_params"] = sum(1 for _, p in model.named_parameters() if not p.requires_grad)
    report["frozen_buffers"] = sum(1 for _ in model.named_buffers())

    # Capture every training step's metrics in memory on the way to the recipe's logger.
    captured: list[dict] = []
    original_log = recipe.log_train_metrics

    def capturing_log(log_data):
        captured.append({k: (float(v) if isinstance(v, (int, float, torch.Tensor)) else v) for k, v in log_data.metrics.items()})
        return original_log(log_data)

    recipe.log_train_metrics = capturing_log

    # Held-out loss: capture NeMo's validation events, and measure the baseline
    # before any update since NeMo's loop has no step-zero evaluation. The
    # validation method switches to eval mode and leaves it to the loop to
    # restore train mode, so that is done here for the baseline.
    captured_val: list[dict] = []
    original_val_log = recipe.log_val_metrics

    # Value and scale changes since the start, totalled per class (dense FP8,
    # expert FP4) over the trainable targets, so flipping can be read against
    # held-out loss on the same clock. Dense materialization is a collective;
    # every rank runs this from the same validation event in the same order.
    def flip_totals() -> dict:
        totals = {f"{c}_{k}": 0 for c in ("dense", "expert") for k in ("values_changed", "scales_changed", "value_elements")}
        for t in state.targets:
            for attr, kind in t.params.items():
                name = f"{t.name}.{attr}"
                if name not in target_params:
                    continue
                c1, s1 = quantized_state(name, getattr(t.module, attr), kind)
                cls = "dense" if kind == "fp8" else "expert"
                # FP4 entries are dequantized once on the host (the expensive
                # part of this function) and serve both the class total and the
                # per-layer selected count.
                entries = None if kind == "fp8" else entry_value_changes(before[name], (c1, s1))
                totals[f"{cls}_values_changed"] += value_changes(before[name], (c1, s1), kind) if entries is None else sum(entries)
                totals[f"{cls}_scales_changed"] += int((before[name][1] != s1).sum())
                totals[f"{cls}_value_elements"] += c1.numel() if kind == "fp8" else c1.numel() * 2
                if entries is not None:
                    # Per layer, over the selected local experts when a selection
                    # is active (all local experts otherwise); summed over ranks
                    # below so every rank's report carries the same numbers.
                    layer = t.name.split(".")[2]
                    n_local = c1.shape[0] // (2 if kind == "fp4_gate_up" else 1)
                    rows = local_selected.get(name, list(range(n_local))) if selection is not None else list(range(n_local))
                    changes = per_expert(entries, n_local)
                    per_row = c1.numel() * 2 // n_local
                    totals[f"expert_values_changed_layer{layer}"] = totals.get(f"expert_values_changed_layer{layer}", 0) + sum(changes[r] for r in rows)
                    totals[f"expert_value_elements_layer{layer}"] = totals.get(f"expert_value_elements_layer{layer}", 0) + per_row * len(rows)
        if dist.is_initialized():
            keys = sorted(k for k in totals if "_layer" in k)
            if keys:
                vec = torch.tensor([totals[k] for k in keys], dtype=torch.int64, device=torch.cuda.current_device())
                dist.all_reduce(vec)
                totals.update(dict(zip(keys, vec.tolist())))
        return totals

    def capturing_val_log(val_name, log_data, metric_logger=None):
        entry = {"step": int(log_data.step), "name": val_name, "val_loss": float(log_data.metrics["val_loss"])}
        entry.update(flip_totals())
        captured_val.append(entry)
        return original_val_log(val_name, log_data, metric_logger)

    recipe.log_val_metrics = capturing_val_log
    for val_name, val_loader in recipe.val_dataloaders.items():
        sample = recipe._run_validation_epoch(val_loader)
        captured_val.append({"step": -1, "name": val_name, "val_loss": float(sample.metrics["val_loss"])})
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
    report["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    step_keys = ("loss", "grad_norm", "lr", "clip_coef_next", "grad_norm_dense", "grad_norm_expert", "clip_coef_next_dense", "clip_coef_next_expert")
    report["steps"] = [{k: m[k] for k in step_keys if k in m} for m in captured]
    report["validation"] = captured_val
    last = captured[-1] if captured else {}
    report["loss"] = last.get("loss")
    report["grad_norm"] = last.get("grad_norm")

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
            c0, s0 = before[name]
            c1, s1 = quantized_state(name, getattr(t.module, attr), kind)
            entries = None if kind == "fp8" else entry_value_changes((c0, s0), (c1, s1))
            if entries is not None:
                layer = int(t.name.split(".")[2])
                n_local = c1.shape[0] // (2 if kind == "fp4_gate_up" else 1)
                offset = _ep_offset(getattr(t.module, attr))
                rows = local_experts.setdefault(layer, {})
                for e, (n, m) in expert_breakdown(entries, n_local, offset, c1.numel() * 2 // n_local).items():
                    acc = rows.setdefault(e, [0, 0])
                    acc[0] += n
                    acc[1] += m
            if selection is not None and name in local_selected:
                # Any bit change in an unselected expert's codes or scales is a
                # frozen-integrity failure, named by global expert index.
                n_local = c1.shape[0] // (2 if kind == "fp4_gate_up" else 1)
                k = c1.shape[0] // n_local
                offset = _ep_offset(getattr(t.module, attr))
                chosen = set(local_selected[name])
                for i in range(n_local):
                    if i in chosen:
                        continue
                    sl = slice(i * k, (i + 1) * k)
                    if not torch.equal(c0[sl], c1[sl]) or not torch.equal(s0[sl], s1[sl]):
                        unselected_changed.append(f"{name}[expert {offset + i}]")
            code_flips = int((c0 != c1).sum())
            code_elems = c0.numel() if kind == "fp8" else c0.numel() * 2
            entry = {
                # Digest of the trained master (dense: full tensor; experts:
                # this rank's shard) so two runs can be compared for equality.
                "master_sha256": _digest(name, getattr(t.module, attr)),
                "codes_flipped": code_flips,
                "values_changed": value_changes((c0, s0), (c1, s1), kind) if entries is None else sum(entries),
                "code_elements": code_elems,
                "scales_changed": int((s0 != s1).sum()),
                "scale_elements": s0.numel(),
            }
            if name in masters:
                m1 = _materialize(name, getattr(t.module, attr)).detach().cpu()
                delta = (m1.float() - masters[name].float()).abs().reshape(-1)
                lr = float(recipe.optimizer[0].param_groups[0]["lr"])
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
            if name in target_params:
                total_code_flips += code_flips
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
    report["indexer_changed"] = sorted(n for n, e in flips.items() if "indexer" in n and (e["codes_flipped"] or e["scales_changed"]))
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

    # Stage 6: probe logits under the hooks, then export. The probe runs in
    # eval mode on the shared token ids so `proxy --expect-logits` on the
    # reloaded export reproduces it bit for bit. Every rank saves its own
    # logits and joins the export collectives; rank 0 alone writes. A refused
    # or failed export is recorded and fails the run instead of raising, so
    # the other ranks still reach the final all-reduce rather than hanging.
    if args.export is not None:
        prefix = os.path.splitext(args.out)[0]
        model.eval()
        ids = probe_token_ids(int(model.config.vocab_size)).to(next(model.parameters()).device)
        torch.save(hooked_logits(model, ids).cpu(), logits_path(prefix, rank))
        report["probe_logits_path"] = logits_path(prefix, rank)
        t2 = time.monotonic()
        try:
            exported = proxy_export.export_trained(state, target_params, snapshot, args.export, force=args.force)
        except ExportError as e:
            print(f"export failed: {e}", file=sys.stderr, flush=True)
            report["export_error"] = str(e)
            ok = False
        else:
            if rank == 0:
                replaced_file = f"{prefix}-replaced.txt"
                with open(replaced_file, "w", encoding="utf-8") as f:
                    f.write("".join(f"{n}\n" for n in exported.pop("replaced")))
                exported["replaced_file"] = replaced_file
                exported["export_seconds"] = round(time.monotonic() - t2, 1)
                report["export"] = exported
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
