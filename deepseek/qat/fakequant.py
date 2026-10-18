"""Fake quantization of DeepSeek-V4 trained weights inside NeMo AutoModel.

Every loss-bearing forward must use ``dequant(Q(w))`` for each trained tensor,
where ``Q`` is the released quantizer from ``deepseek.qat.quant``. This module
installs that on a built NeMo model without touching its parameters, FSDP2
wrapping, or optimizer: each target module's ``forward`` is replaced by a wrapper
that runs the original forward with the fake-quantized tensors swapped in through
``torch.nn.utils.stateless._reparametrize_module`` (the mechanism behind
``torch.func.functional_call``; the public entry cannot be used because it
re-enters ``Module.__call__`` and would recurse into this wrapper).

Gradients pass through the quantizer unchanged (straight-through estimator), so
the optimizer updates the high-precision masters while the forward sees the grid.

Target orientation. NeMo stores dense projections as ``nn.Linear`` weights in HF
``[out, in]`` layout, so FP8 128x128 blocks apply directly. Routed experts are one
``[E, dim, 2*inter]`` parameter equal to ``cat(w1.T, w3.T)`` per expert and one
``[E, inter, dim]`` equal to ``w2.T``; FP4 groups of 32 run along the HF input
dimension, so each expert matrix is transposed back to ``[out, in]`` before
quantizing. Expert parameters sharded over the expert-parallel mesh are DTensors
sharded on dim 0; they are quantized on the local shard and re-wrapped, which is
exact because no block spans experts.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.utils.stateless import _reparametrize_module

from deepseek.qat import quant as Q

# Experts per chunk when quantizing a stacked expert tensor; bounds the float32
# temporaries to a few GB regardless of the local expert count.
EXPERT_CHUNK = 16

# The FP4 scale rule every training-path quantization uses: the forward here,
# the flip metric in proxy_step, and export. One constant so they cannot drift.
# "min_error" avoids the floor rule's knife edge (quant module docstring).
FP4_SCALE_RULE = "min_error"


def quantize_fp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project FP4 quantizer: quantize_fp4_rows under FP4_SCALE_RULE."""
    return Q.quantize_fp4_rows(w, scale_rule=FP4_SCALE_RULE)


# Observer of the quantizer's output during one tensor's fake quantization: set
# by `observed()` so flip accounting (deepseek.qat.flips) can read the codes,
# scales, and float32 values the forward is built from instead of recomputing
# them. Called once for an FP8 tensor with position None, and once per expert
# chunk and half for FP4 with position (first expert, expert count, half: 0 for
# w1, 1 for w3, None for w2); FP4 rows are HF [experts x out, in] order. None
# when nothing is counting.
_observer = None


def _fp8_dequant_quant(w: torch.Tensor) -> torch.Tensor:
    codes, scales = Q.quantize_fp8_blocks(w)
    values = Q.dequantize_fp8_blocks(codes, scales)
    if _observer is not None:
        _observer("fp8", None, codes, scales, values)
    return values.to(w.dtype)


def _fp4_dequant_quant(w: torch.Tensor, pos: tuple | None = None) -> torch.Tensor:
    packed, scales = quantize_fp4(w)
    values = Q.dequantize_fp4_rows(packed, scales)
    if _observer is not None:
        _observer("fp4", pos, packed, scales, values)
    return values.to(w.dtype)


# [E, dim, 2*inter] -> fake-quantize each expert's w1 and w3 in HF [inter, dim]
# layout -> same NeMo layout. Chunked over experts.
def fake_quant_gate_up(w: torch.Tensor) -> torch.Tensor:
    e, dim, two_inter = w.shape
    inter = two_inter // 2
    out = torch.empty_like(w)
    for start in range(0, e, EXPERT_CHUNK):
        chunk = w[start : start + EXPERT_CHUNK]
        n = chunk.shape[0]
        for half, (lo, hi) in enumerate(((0, inter), (inter, two_inter))):
            hf = chunk[:, :, lo:hi].transpose(1, 2).reshape(n * inter, dim)
            out[start : start + n, :, lo:hi] = _fp4_dequant_quant(hf, (start, n, half)).reshape(n, inter, dim).transpose(1, 2)
    return out


# [E, inter, dim] -> fake-quantize each expert's w2 in HF [dim, inter] layout.
def fake_quant_down(w: torch.Tensor) -> torch.Tensor:
    e, inter, dim = w.shape
    out = torch.empty_like(w)
    for start in range(0, e, EXPERT_CHUNK):
        chunk = w[start : start + EXPERT_CHUNK]
        n = chunk.shape[0]
        hf = chunk.transpose(1, 2).reshape(n * dim, inter)
        out[start : start + n] = _fp4_dequant_quant(hf, (start, n, None)).reshape(n, dim, inter).transpose(1, 2)
    return out


KINDS = {
    "fp8": _fp8_dequant_quant,
    "fp4_gate_up": fake_quant_gate_up,
    "fp4_down": fake_quant_down,
}


class FakeQuant(torch.autograd.Function):
    """dequant(quant(w)) forward; identity backward (straight-through)."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, kind: str) -> torch.Tensor:
        return KINDS[kind](w.detach())

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


def fake_quant(w: torch.Tensor, kind: str) -> torch.Tensor:
    """Apply the kind's fake quantizer to a tensor or a dim-0-sharded DTensor."""
    if isinstance(w, DTensor):
        if any(p.is_shard() and p.dim != 0 for p in w.placements):
            raise ValueError(f"fake quantization needs dim-0 (expert) sharding, got placements {w.placements}")
        local = FakeQuant.apply(w.to_local(), kind)
        return DTensor.from_local(local, w.device_mesh, w.placements, run_check=False)
    return FakeQuant.apply(w, kind)


def observed(w: torch.Tensor, kind: str, observer) -> torch.Tensor:
    """fake_quant with `observer` receiving the quantizer's output for this tensor (None: no observer)."""
    global _observer
    previous = _observer
    _observer = observer
    try:
        return fake_quant(w, kind)
    finally:
        _observer = previous


@dataclass
class Target:
    name: str  # module path within the model
    module: nn.Module
    params: dict[str, str]  # parameter attribute -> kind


@dataclass
class FakeQuantState:
    targets: list[Target]
    originals: dict[str, object] = field(default_factory=dict)
    # Observer factory, (target name, attribute, kind) -> per-tensor observer or
    # None, consulted on every forward; set by flip accounting, None otherwise.
    observer: object = None


# Trained-tensor targets of a DeepSeek-V4 model, by layer, verified against the
# config so a renamed or missing module fails loudly instead of training unquantized.
def find_targets(model: nn.Module) -> list[Target]:
    cfg = model.config
    layers = model.model.layers
    ratios = list(cfg.compress_ratios)
    targets: list[Target] = []
    for key, layer in layers.items():
        i = int(key)
        attn = layer.self_attn
        for proj in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            targets.append(Target(f"model.layers.{i}.self_attn.{proj}", getattr(attn, proj), {"weight": "fp8"}))
        if ratios[i] == 4:
            indexer = attn.compressor.indexer
            targets.append(Target(f"model.layers.{i}.self_attn.compressor.indexer.wq_b", indexer.wq_b, {"weight": "fp8"}))
        elif hasattr(attn, "compressor") and getattr(attn.compressor, "indexer", None) is not None:
            raise ValueError(f"layer {i}: indexer present but compress ratio is {ratios[i]}")
        shared = layer.mlp.shared_experts
        for proj in ("gate_proj", "up_proj", "down_proj"):
            targets.append(Target(f"model.layers.{i}.mlp.shared_experts.{proj}", getattr(shared, proj), {"weight": "fp8"}))
        experts = layer.mlp.experts
        # Split expert storage (stage 6b): a layer with no selected experts holds
        # packed stacks only and has no trained expert tensor to fake-quantize.
        from deepseek.qat.split_experts import SplitExperts  # local: NeMo is only needed on a built model

        if isinstance(experts, SplitExperts) and experts.n_selected == 0:
            continue
        for attr in ("gate_and_up_projs", "down_projs"):
            if not isinstance(getattr(experts, attr, None), torch.Tensor):
                raise ValueError(f"layer {i}: experts module {type(experts).__name__} has no {attr} parameter")
        targets.append(Target(f"model.layers.{i}.mlp.experts", experts,
                              {"gate_and_up_projs": "fp4_gate_up", "down_projs": "fp4_down"}))
    for t in targets:
        for attr in t.params:
            if not isinstance(getattr(t.module, attr, None), torch.Tensor):
                raise ValueError(f"{t.name} has no tensor attribute {attr}")
    return targets


def install(model: nn.Module) -> FakeQuantState:
    """Replace each target module's forward with the fake-quantized version."""
    state = FakeQuantState(find_targets(model))
    for t in state.targets:
        original = t.module.forward
        state.originals[t.name] = original

        def wrapped(*args, _name=t.name, _module=t.module, _params=t.params, _orig=original, **kwargs):
            swapped = {}
            for attr, kind in _params.items():
                observer = state.observer(_name, attr, kind) if state.observer is not None else None
                swapped[attr] = observed(getattr(_module, attr), kind, observer)
            with _reparametrize_module(_module, swapped):
                return _orig(*args, **kwargs)

        t.module.forward = wrapped
    return state


def uninstall(state: FakeQuantState) -> None:
    for t in state.targets:
        if t.name in state.originals:
            del t.module.forward  # restores the class method
    state.originals.clear()


def effective_weights(state: FakeQuantState):
    """Yield (qualified parameter name, fake-quantized tensor) for every target."""
    for t in state.targets:
        for attr, kind in t.params.items():
            yield f"{t.name}.{attr}", fake_quant(getattr(t.module, attr), kind)
