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
"""

import argparse
import hashlib
import json
import math
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q
from deepseek.qat import inbackward
from deepseek.qat.freeze import freeze_gate_bias


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
    total = 0
    for i in range(c0.shape[0]):
        v0 = Q.dequantize_fp4_rows(c0[i], s0[i].view(torch.float8_e8m0fnu))
        v1 = Q.dequantize_fp4_rows(c1[i], s1[i].view(torch.float8_e8m0fnu))
        total += int((v0 != v1).sum())
    return total


# Apply --set KEY=VALUE (YAML scalar) and --unset KEY to a loaded NeMo config
# before the recipe is built, so one recipe file serves every arm of a
# comparison. Returns what was applied for the report. Bad syntax or an
# --unset of a key the recipe does not have is a usage error, not a silent no-op.
def apply_overrides(cfg, sets: list[str], unsets: list[str]) -> dict:
    import yaml
    from nemo_automodel.components.config.loader import ConfigNode

    applied: dict = {"set": {}, "unset": []}
    for item in sets:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        cfg.set_by_dotted(key, yaml.safe_load(raw))
        applied["set"][key] = raw
    for key in unsets:
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.__dict__.get(part)
            if not isinstance(node, ConfigNode):
                raise SystemExit(f"--unset {key}: {part!r} is not a section of the recipe")
        if parts[-1] not in node.__dict__:
            raise SystemExit(f"--unset {key}: key not present in the recipe")
        del node.__dict__[parts[-1]]
        applied["unset"].append(key)
    return applied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="write the rank-0 report here as JSON")
    parser.add_argument("--in-backward", action="store_true",
                        help="step each parameter inside backward with lagged clipping at the recipe's max_norm")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a dotted recipe key with a YAML scalar before building (repeatable)")
    parser.add_argument("--unset", action="append", default=[], metavar="KEY",
                        help="delete a dotted recipe key before building (repeatable)")
    parser.add_argument("--per-class-clip", action="store_true",
                        help="with --in-backward: clip dense (attention, shared experts) and routed experts by separate norms")
    parser.add_argument("--trainable-subset", choices=("all", "dense", "experts"), default="all",
                        help="which targets the recipe is expected to train (ablations freeze the rest via --set); the exact-set check uses it")
    args = parser.parse_args(argv)

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    t0 = time.monotonic()
    cfg = load_yaml_config(args.config)
    overrides = apply_overrides(cfg, args.set, args.unset)
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
    stepper = None
    if args.in_backward:
        stepper = inbackward.from_recipe(recipe, recipe.max_grad_norm, per_class=args.per_class_clip)
        report["per_class_clip"] = args.per_class_clip
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
                totals[f"{cls}_values_changed"] += value_changes(before[name], (c1, s1), kind)
                totals[f"{cls}_scales_changed"] += int((before[name][1] != s1).sum())
                totals[f"{cls}_value_elements"] += c1.numel() if kind == "fp8" else c1.numel() * 2
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
    for t in state.targets:
        for attr, kind in t.params.items():
            name = f"{t.name}.{attr}"
            c0, s0 = before[name]
            c1, s1 = quantized_state(name, getattr(t.module, attr), kind)
            code_flips = int((c0 != c1).sum())
            code_elems = c0.numel() if kind == "fp8" else c0.numel() * 2
            entry = {
                # Digest of the trained master (dense: full tensor; experts:
                # this rank's shard) so two runs can be compared for equality.
                "master_sha256": _digest(name, getattr(t.module, attr)),
                "codes_flipped": code_flips,
                "values_changed": value_changes((c0, s0), (c1, s1), kind),
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
    report["total_code_flips_in_trainable"] = total_code_flips
    report["total_value_changes_in_trainable"] = total_value_changes
    # One digest over every target's master digest, in target order, so the
    # short rank-1 line can be compared with another run's rank-1 line.
    report["masters_sha256"] = hashlib.sha256("".join(e["master_sha256"] for e in flips.values()).encode()).hexdigest()
    # A tensor absent before training (a buffer created during the step) has no
    # prior digest and is reported as changed; the frozen set must be fixed.
    report["frozen_changed"] = sorted(n for n, t in _frozen_tensors(model) if _digest(n, t) != frozen_before.get(n))
    report["indexer_changed"] = sorted(n for n, e in flips.items() if "indexer" in n and (e["codes_flipped"] or e["scales_changed"]))

    # Every step's loss and gradient norm must be finite, not only the last.
    finite = bool(captured) and all(
        isinstance(m.get(k), (int, float)) and math.isfinite(m[k]) for m in captured for k in ("loss", "grad_norm")
    )
    ok = (
        finite
        and not report["frozen_changed"]
        and not report["indexer_changed"]
        and not report["trainable_not_target"]
        and not report["target_not_trainable"]
        and total_value_changes > 0
    )
    report["ok"] = ok
    report["elapsed_seconds"] = round(time.monotonic() - t0, 1)
    if rank == 0:
        print(json.dumps(report, indent=1), flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1)
    else:
        print(json.dumps({k: report[k] for k in ("rank", "ok", "frozen_changed", "total_value_changes_in_trainable", "masters_sha256")}), flush=True)
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
