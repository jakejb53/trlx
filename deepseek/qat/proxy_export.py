"""Export a trained proxy as a complete checkpoint in the source layout.

Publication is a snapshot of ``Q(master)``: every trained target is quantized
with the project quantizers (``fakequant.quantize_fp4`` under the training
scale rule, ``quant.quantize_fp8_blocks``) exactly as the training forward and
the flip metric quantize it, and written under its source tensor name through
``export.export``. Every other tensor is copied byte for byte from the source
the model was loaded from, so a train, export, train-on-export chain stays a
complete checkpoint at every link.

Only tensors whose quantized bytes differ from the source are replaced. The
replaced list is therefore the exact set ``verify_export --expect`` must see,
and the report's changed-against-source count is a measurement, not a guess.

Distribution: dense FP8 parameters are FSDP2 DTensors and are materialized in
full on every rank (a collective). Routed experts are DTensors sharded on
dim 0 over the expert-parallel mesh; each rank quantizes its local experts and
the packed codes and scales are gathered to rank 0, which assembles the
replacement dict and writes. Every rank must call ``export_trained``; only
rank 0 returns a report.
"""

import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from deepseek.qat import export as EX
from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q
from deepseek.qat.proxy import Source, hf_names

# HF expert tensor halves of each NeMo expert parameter, in the order the
# stacked NeMo tensor holds them: gate_and_up_projs[e] = cat(w1.T, w3.T).
EXPERT_HALVES = {"fp4_gate_up": ("w1", "w3"), "fp4_down": ("w2",)}


def expert_name(layer: int, e: int, which: str) -> str:
    """HF base name of one routed-expert matrix; add `.weight` or `.scale`."""
    return f"layers.{layer}.ffn.experts.{e}.{which}"


def scale_name(weight_name: str) -> str:
    """The scale tensor paired with a quantized weight tensor."""
    if not weight_name.endswith(".weight"):
        raise ValueError(f"{weight_name}: not a weight tensor name")
    return weight_name[: -len(".weight")] + ".scale"


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


# Quantize this rank's local experts in HF [out, in] orientation. Returns, per
# HF half, stacked packed codes [E_local, out, in/2] uint8 and scales
# [E_local, out, in/32] E8M0 on the parameter's device. One expert at a time
# bounds the float32 temporaries to one matrix.
def quantize_local_experts(w_local: torch.Tensor, kind: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if kind not in EXPERT_HALVES:
        raise ValueError(f"not an expert kind: {kind!r}")
    e = w_local.shape[0]
    if kind == "fp4_gate_up":
        inter = w_local.shape[2] // 2
        slices = {"w1": (0, inter), "w3": (inter, 2 * inter)}
    else:
        slices = {"w2": (0, w_local.shape[2])}
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for which, (lo, hi) in slices.items():
        codes, scales = [], []
        for i in range(e):
            p, s = FQ.quantize_fp4(w_local[i, :, lo:hi].T)
            codes.append(p)
            scales.append(s)
        out[which] = (torch.stack(codes), torch.stack(scales))
    return out


# Gather equal-shaped tensors from every rank of the parameter's expert mesh
# onto global rank 0, in mesh-rank order, so list index r holds the experts
# that start at r * E_local (the same rule as proxy._ep_offset). The mesh must
# be one-dimensional and span the world: that is the layout every proxy
# recipe and the update-in-backward stepper assume, and anything else would
# make the expert-index arithmetic silently wrong. Non-DTensor (unsharded)
# parameters gather trivially. Returns the list on rank 0, None elsewhere.
def gather_expert_shards(t: torch.Tensor, param: torch.Tensor) -> list[torch.Tensor] | None:
    if not isinstance(param, DTensor):
        return [t]
    mesh = param.device_mesh
    if mesh.ndim != 1:
        raise ValueError(f"expert parameter mesh must be one-dimensional, got {mesh.ndim} dims")
    group = mesh.get_group()
    if dist.get_world_size(group) != dist.get_world_size() or dist.get_global_rank(group, 0) != 0:
        raise ValueError("expert parameter mesh must span the world with global rank 0 as its first rank")
    # NCCL has no E8M0 type; callers pass uint8 views.
    if t.dtype != torch.uint8:
        raise ValueError(f"gather expects uint8 payloads, got {t.dtype}")
    rank = dist.get_rank()
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size(group))] if rank == 0 else None
    dist.gather(t, gathered, dst=0, group=group)
    return gathered


# Build the HF replacement entries for one expert half from per-rank stacks:
# global expert e is rank r's local expert i with e = r * E_local + i. Tensors
# are moved to host once per rank stack and sliced as views; scales get back
# their E8M0 dtype so the exporter's header check sees the source dtype.
def assemble_expert_replacements(layer: int, which: str, per_rank: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    replacements: dict[str, torch.Tensor] = {}
    e = 0
    for codes, scales in per_rank:
        codes = codes.cpu()
        scales = scales.cpu().view(torch.float8_e8m0fnu)
        for i in range(codes.shape[0]):
            base = expert_name(layer, e, which)
            replacements[base + ".weight"] = codes[i]
            replacements[base + ".scale"] = scales[i]
            e += 1
    return replacements


# Quantize a dense FP8 target as a full tensor (collective for DTensors) and
# map it to its HF weight and scale names. Every rank computes the same
# result; the caller keeps rank 0's.
def dense_replacements(name: str, param: torch.Tensor) -> dict[str, torch.Tensor]:
    w = param.full_tensor() if isinstance(param, DTensor) else param
    codes, scales = Q.quantize_fp8_blocks(w.detach())
    hf = hf_names(name)[0]
    return {hf: codes.cpu(), scale_name(hf): scales.cpu()}


# Split candidate replacements into those whose bytes differ from the source
# and those identical to it. A shape mismatch compares unequal and is left for
# the exporter to reject with the tensor's name.
def differing_from_source(replacements: dict[str, torch.Tensor], source: Source) -> tuple[dict[str, torch.Tensor], list[str]]:
    kept: dict[str, torch.Tensor] = {}
    unchanged: list[str] = []
    for name, t in replacements.items():
        src = source.tensor(name)
        same = tuple(src.shape) == tuple(t.shape) and torch.equal(
            t.contiguous().view(torch.uint8), src.contiguous().view(torch.uint8)
        )
        if same:
            unchanged.append(name)
        else:
            kept[name] = t
    return kept, unchanged


def export_trained(state: FQ.FakeQuantState, target_params: set[str], snapshot: str, out_dir: str,
                   force: bool = False, log=sys.stderr) -> dict:
    """Export the trained targets of a hooked model. All ranks call this; rank 0 writes.

    `target_params` names the trained parameters (`<target>.<attr>`); other
    targets are left to the source copy. `snapshot` is the checkpoint the model
    was loaded from. Returns the export report on rank 0 and {} elsewhere.
    Raises `export.ExportError` on rank 0 for refused or failed publication.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    candidates: dict[str, torch.Tensor] = {}
    trained = [(t, attr, kind) for t in state.targets for attr, kind in t.params.items() if f"{t.name}.{attr}" in target_params]
    print(f"export: quantizing {len(trained)} trained targets", file=log, flush=True)
    for t, attr, kind in trained:
        name = f"{t.name}.{attr}"
        param = getattr(t.module, attr).detach()
        if kind == "fp8":
            reps = dense_replacements(name, param)
        else:
            layer = int(t.name.split(".")[2])
            reps = {}
            for which, (codes, scales) in quantize_local_experts(_local(param), kind).items():
                per_rank_codes = gather_expert_shards(codes, param)
                per_rank_scales = gather_expert_shards(scales.view(torch.uint8), param)
                if rank == 0:
                    reps.update(assemble_expert_replacements(layer, which, list(zip(per_rank_codes, per_rank_scales))))
                del codes, scales, per_rank_codes, per_rank_scales
        if rank == 0:
            candidates.update(reps)
    if rank != 0:
        return {}
    kept, unchanged = differing_from_source(candidates, Source(snapshot))
    print(f"export: {len(kept)} of {len(candidates)} candidate tensors differ from the source", file=log, flush=True)
    per_shard = EX.export(snapshot, out_dir, kept, force=force, log=log)
    return {
        "path": out_dir,
        "source": snapshot,
        "trained_targets": len(trained),
        "candidate_tensors": len(candidates),
        "replaced_count": len(kept),
        "replaced": sorted(kept),
        "unchanged": sorted(unchanged),
        "per_shard": {shard: len(names) for shard, names in per_shard.items() if names},
    }
