"""Stage 3 check: one (or a few) QAT optimizer steps on the four-layer proxy.

Builds the recipe, installs the fake-quant hooks, snapshots the quantized codes
and scales of every target and a digest of every frozen parameter, runs the
recipe's own training loop, then requantizes and reports per target:

- codes and scales that flipped, and master elements that changed;
- that every frozen parameter is bit-identical to before;
- that the trainable set equals the target set minus the indexer;
- training loss and gradient norm from the recipe's metrics log.

Exit nonzero on non-finite loss or gradient norm, a changed frozen parameter, a
trainable-set mismatch, or zero flips across all targets. Launch from the repo
root:

  deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy_step \
      --config deepseek/qat/proxy-train.yaml [--out REPORT.json]
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
                p, s = Q.quantize_fp4_rows(w[i, :, lo:hi].T)
                codes.append(p.cpu()); scales.append(s.view(torch.uint8).cpu())
        else:
            p, s = Q.quantize_fp4_rows(w[i].T)
            codes.append(p.cpu()); scales.append(s.view(torch.uint8).cpu())
    return torch.stack(codes), torch.stack(scales)


def _digest(name: str, t: torch.Tensor) -> str:
    return hashlib.sha256(_materialize(name, t).detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="write the rank-0 report here as JSON")
    args = parser.parse_args(argv)

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    t0 = time.monotonic()
    cfg = load_yaml_config(args.config)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = recipe.model_parts[0]
    state = FQ.install(model)
    report: dict = {"rank": rank, "setup_seconds": round(time.monotonic() - t0, 1)}

    # Trainable set must be exactly the targets minus the indexer.
    target_params = {f"{t.name}.{a}" for t in state.targets for a in t.params if "indexer" not in t.name}
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
    frozen_before = {n: _digest(n, p) for n, p in model.named_parameters() if not p.requires_grad}

    # Capture every training step's metrics in memory on the way to the recipe's logger.
    captured: list[dict] = []
    original_log = recipe.log_train_metrics

    def capturing_log(log_data):
        captured.append({k: (float(v) if isinstance(v, (int, float, torch.Tensor)) else v) for k, v in log_data.metrics.items()})
        return original_log(log_data)

    recipe.log_train_metrics = capturing_log

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

    inner.step = measured_step

    t1 = time.monotonic()
    recipe.run_train_validation_loop()
    report["grad_stats"] = grad_stats
    report["train_seconds"] = round(time.monotonic() - t1, 1)
    report["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    report["steps"] = [{k: m.get(k) for k in ("loss", "grad_norm", "lr")} for m in captured]
    last = captured[-1] if captured else {}
    report["loss"] = last.get("loss")
    report["grad_norm"] = last.get("grad_norm")

    flips = {}
    total_code_flips = 0
    for t in state.targets:
        for attr, kind in t.params.items():
            name = f"{t.name}.{attr}"
            c0, s0 = before[name]
            c1, s1 = quantized_state(name, getattr(t.module, attr), kind)
            code_flips = int((c0 != c1).sum())
            code_elems = c0.numel() if kind == "fp8" else c0.numel() * 2
            entry = {
                "codes_flipped": code_flips,
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
    report["targets"] = flips
    report["total_code_flips_in_trainable"] = total_code_flips
    report["frozen_changed"] = sorted(n for n, p in model.named_parameters() if not p.requires_grad and _digest(n, p) != frozen_before[n])
    report["indexer_changed"] = sorted(n for n, e in flips.items() if "indexer" in n and (e["codes_flipped"] or e["scales_changed"]))

    finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (report["loss"], report["grad_norm"]))
    ok = (
        finite
        and not report["frozen_changed"]
        and not report["indexer_changed"]
        and not report["trainable_not_target"]
        and not report["target_not_trainable"]
        and total_code_flips > 0
    )
    report["ok"] = ok
    report["elapsed_seconds"] = round(time.monotonic() - t0, 1)
    if rank == 0:
        print(json.dumps(report, indent=1), flush=True)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1)
    else:
        print(json.dumps({k: report[k] for k in ("rank", "ok", "frozen_changed", "total_code_flips_in_trainable")}), flush=True)
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
