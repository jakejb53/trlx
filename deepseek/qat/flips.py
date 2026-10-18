"""Per-step flip accounting on the device.

Which quantized values and scales of every trained tensor differ from the source
checkpoint is the plan's leading indicator of whether QAT publishes anything. The
fake-quant hook quantizes every trained tensor on every forward; this module
observes that output (``fakequant.observed``) and compares it on the device
against the tensor's baseline codes and scales, kept in pinned host memory and
streamed per chunk. The counts therefore exist after every step, at the cost of
one host-to-device copy of the baseline per step and no second quantization. The
CPU comparison in ``proxy_step`` (``value_changes``, ``entry_value_changes``) is
the reference these counts are tested against; it is not run during training.

Counting is per pass: the caller begins a pass before each training step and
before each validation or probe forward. The first observation of a tensor in a
pass counts; the activation-checkpoint recompute, a second observation of the
same tensor, is skipped. Dense tensors are seen unsharded on every rank, so each
is counted on one owner rank (round robin by position among dense targets);
expert tensors are this rank's local shard. ``totals()`` reduces the per-class
and per-layer counts over the world, so every rank sees the same numbers.

Values are compared after dequantization, as the plan requires: the FP4 +0/-0
pair dequantizes identically and does not count, and a scale change counts the
values it moves. ``codes_flipped`` compares packed bytes (two FP4 codes per byte)
as the CPU reference does.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


@dataclass
class Baseline:
    """Source codes and scales of one tensor in pinned host memory.

    FP8: one entry, codes ``[out, in]`` and scales ``[out/128, in/128]``. FP4: one
    entry per half (w1 and w3 for gate/up, w2 alone for down), codes
    ``[experts x rows_per_expert, in/2]`` and scales ``[..., in/32]`` in the HF row
    order the hook produces.
    """

    kind: str
    experts: int  # local experts (FP4); 0 for FP8
    rows_per_expert: int  # FP4 rows per expert per half; 0 for FP8
    halves: int  # 2 for gate/up, 1 for down and FP8
    codes: list = field(default_factory=list)
    scales: list = field(default_factory=list)

    @property
    def value_elements(self) -> int:
        if self.kind == "fp8":
            return self.codes[0].numel()
        return sum(c.numel() * 2 for c in self.codes)

    @property
    def scale_elements(self) -> int:
        return sum(s.numel() for s in self.scales)

    @property
    def code_elements(self) -> int:
        # The CPU reference counts FP4 code elements as two per packed byte.
        return self.value_elements


@dataclass
class _Counted:
    """One counted tensor: its baseline, device accumulators, and pass state."""

    name: str
    target: object
    attr: str
    kind: str  # fp8 | fp4_gate_up | fp4_down
    layer: int | None  # expert tensors only
    owner: bool  # this rank counts it (dense: owner rank; experts: always)
    chunks: int  # observations per pass (FP8: 1; FP4: ceil(E / EXPERT_CHUNK) x halves)
    base: Baseline | None = None
    seen_chunks: int = 0  # observations in the current pass
    values_changed: torch.Tensor | None = None  # int64 [experts] (FP4) or [1] (FP8), on device
    codes_flipped: torch.Tensor | None = None
    scales_changed: torch.Tensor | None = None


class FlipCounter:
    """Counts changed values, codes, and scales of the trained tensors every pass."""

    def __init__(self, state: FQ.FakeQuantState, target_params: set[str], materialize, device: torch.device, expert_class_elements: int = 0):
        """Record every counted tensor's baseline from the current (untrained) weights.

        `materialize(name, param)` returns the tensor the hook will see: the full
        tensor for dense parameters (a collective every rank must join) and the
        local shard for experts. Dense counting is spread over ranks by position.
        `expert_class_elements` is the model's total routed-expert values, selected
        or not, from its config: the fixed expert denominator.
        """
        self.device = device
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.counted: dict[str, _Counted] = {}
        dense_index = 0
        for t in state.targets:
            for attr, kind in t.params.items():
                name = f"{t.name}.{attr}"
                if name not in target_params:
                    continue
                if kind == "fp8":
                    owner = dense_index % self.world == self.rank
                    dense_index += 1
                    layer = None
                    chunks = 1
                else:
                    owner = True
                    layer = int(t.name.split(".")[2])
                    local = _local(getattr(t.module, attr))
                    chunks = math.ceil(local.shape[0] / FQ.EXPERT_CHUNK) * (2 if kind == "fp4_gate_up" else 1)
                self.counted[name] = _Counted(name, t, attr, kind, layer, owner, chunks)
        self.layers = sorted({c.layer for c in self.counted.values() if c.layer is not None})
        # Class totals over every target, trained or frozen, as the fixed
        # denominator that makes fractions comparable across runs with different
        # trainable sets: dense is every FP8 target's elements, expert is every
        # routed expert of the model (selected BF16 stacks and packed stacks).
        self.class_elements = {"dense": 0, "expert": expert_class_elements}
        for t in state.targets:
            for attr, kind in t.params.items():
                if kind == "fp8":
                    self.class_elements["dense"] += getattr(t.module, attr).numel()
        # Baselines: every rank materializes every dense tensor (collective), the
        # owner records it; experts record their local shard.
        for c in self.counted.values():
            w = materialize(c.name, getattr(c.target.module, c.attr)).detach()
            if c.owner:
                FQ.observed(w, c.kind, self._recorder(c, w))
                if c.base is None or len(c.base.codes) != c.base.halves:
                    raise RuntimeError(f"{c.name}: baseline recording saw {0 if c.base is None else len(c.base.codes)} halves")
                n = c.base.experts if c.kind != "fp8" else 1
                zeros = lambda: torch.zeros(n, dtype=torch.int64, device=device)
                c.values_changed, c.codes_flipped, c.scales_changed = zeros(), zeros(), zeros()
            del w
        state.observer = self.observer
        self._pass_open = False

    # Observer that copies the quantizer's output into pinned host buffers.
    def _recorder(self, c: _Counted, w: torch.Tensor):
        if c.kind == "fp8":
            c.base = Baseline("fp8", 0, 0, 1)
        else:
            e = w.shape[0]
            if c.kind == "fp4_gate_up":
                rows, halves, cols = w.shape[2] // 2, 2, w.shape[1]  # [E, dim, 2*inter]: rows inter, cols dim
            else:
                rows, halves, cols = w.shape[2], 1, w.shape[1]  # [E, inter, dim]: rows dim, cols inter
            c.base = Baseline("fp4", e, rows, halves)
            for _ in range(halves):
                c.base.codes.append(torch.empty((e * rows, cols // 2), dtype=torch.uint8, pin_memory=True))
                c.base.scales.append(torch.empty((e * rows, cols // Q.FP4_GROUP), dtype=torch.uint8, pin_memory=True))

        def record(kind, pos, codes, scales, values):
            if kind == "fp8":
                c.base.codes.append(_u8(codes).cpu().pin_memory())
                c.base.scales.append(_u8(scales).cpu().pin_memory())
                return
            start, n, half = pos
            h = 0 if half is None else half
            r0, r1 = start * c.base.rows_per_expert, (start + n) * c.base.rows_per_expert
            c.base.codes[h][r0:r1].copy_(_u8(codes))
            c.base.scales[h][r0:r1].copy_(_u8(scales))

        return record

    # Observer factory the hook consults per tensor and forward.
    def observer(self, name: str, attr: str, kind: str):
        c = self.counted.get(f"{name}.{attr}")
        if c is None or not c.owner or not self._pass_open or c.seen_chunks >= c.chunks:
            return None
        return lambda k, pos, codes, scales, values: self._count(c, k, pos, codes, scales, values)

    def begin_pass(self) -> None:
        """Start counting a new forward sweep; the first observation of each tensor counts."""
        for c in self.counted.values():
            c.seen_chunks = 0
        self._pass_open = True

    def _count(self, c: _Counted, kind: str, pos, codes, scales, values) -> None:
        if c.seen_chunks == 0:
            c.values_changed.zero_()
            c.codes_flipped.zero_()
            c.scales_changed.zero_()
        dev = values.device
        if kind == "fp8":
            b_codes = c.base.codes[0].to(dev, non_blocking=True)
            b_scales = c.base.scales[0].to(dev, non_blocking=True)
            v0 = Q.dequantize_fp8_blocks(b_codes.view(torch.float8_e4m3fn), b_scales.view(torch.float8_e8m0fnu))
            c.values_changed += (v0 != values).sum()
            c.codes_flipped += (b_codes != _u8(codes)).sum()
            c.scales_changed += (b_scales != _u8(scales)).sum()
        else:
            start, n, half = pos
            h = 0 if half is None else half
            r0, r1 = start * c.base.rows_per_expert, (start + n) * c.base.rows_per_expert
            b_codes = c.base.codes[h][r0:r1].to(dev, non_blocking=True)
            b_scales = c.base.scales[h][r0:r1].to(dev, non_blocking=True)
            v0 = Q.dequantize_fp4_rows(b_codes, b_scales.view(torch.float8_e8m0fnu))
            c.values_changed[start : start + n] += (v0 != values).view(n, -1).sum(1)
            c.codes_flipped[start : start + n] += (b_codes != _u8(codes)).view(n, -1).sum(1)
            c.scales_changed[start : start + n] += (b_scales != _u8(scales)).view(n, -1).sum(1)
        c.seen_chunks += 1

    def refresh(self, materialize) -> None:
        """One explicit counting pass over the current masters, outside any forward.

        `materialize` is the constructor's: dense tensors in full (a collective
        every rank joins), experts as local shards.
        """
        self.begin_pass()
        for c in self.counted.values():
            w = materialize(c.name, getattr(c.target.module, c.attr)).detach()
            if c.owner:
                FQ.observed(w, c.kind, self.observer(c.target.name, c.attr, c.kind))
            del w

    def _check_complete(self) -> None:
        missing = [c.name for c in self.counted.values() if c.owner and c.seen_chunks != c.chunks]
        if missing:
            raise RuntimeError(f"flip accounting: {len(missing)} counted tensors were not fully observed this pass, e.g. {missing[:3]}")

    def totals(self) -> dict:
        """Per-class and per-layer counts of the last pass, reduced over the world.

        Keys match the former validation-time accounting: ``{dense,expert}_values_changed``,
        ``_scales_changed``, ``_codes_flipped``, ``_value_elements`` (the trained
        set, the flip-rate denominator), and per layer ``expert_values_changed_layer{L}``
        and ``expert_value_elements_layer{L}``; plus ``{dense,expert}_class_elements``,
        the class's total values in the model whether trained or not, the
        denominator that is comparable across runs with different trainable sets.
        """
        self._check_complete()
        keys = [f"{cls}_{k}" for cls in ("dense", "expert") for k in ("values_changed", "scales_changed", "codes_flipped", "value_elements")]
        keys += [f"expert_values_changed_layer{L}" for L in self.layers] + [f"expert_value_elements_layer{L}" for L in self.layers]
        vec = torch.zeros(len(keys), dtype=torch.int64, device=self.device)
        at = {k: i for i, k in enumerate(keys)}
        for c in self.counted.values():
            if not c.owner:
                continue
            cls = "dense" if c.kind == "fp8" else "expert"
            vec[at[f"{cls}_values_changed"]] += c.values_changed.sum()
            vec[at[f"{cls}_scales_changed"]] += c.scales_changed.sum()
            vec[at[f"{cls}_codes_flipped"]] += c.codes_flipped.sum()
            vec[at[f"{cls}_value_elements"]] += c.base.value_elements
            if c.layer is not None:
                vec[at[f"expert_values_changed_layer{c.layer}"]] += c.values_changed.sum()
                vec[at[f"expert_value_elements_layer{c.layer}"]] += c.base.value_elements
        if dist.is_initialized():
            dist.all_reduce(vec)
        out = dict(zip(keys, vec.tolist()))
        out["dense_class_elements"] = self.class_elements["dense"]
        out["expert_class_elements"] = self.class_elements["expert"]
        return out

    def per_tensor(self) -> dict[str, dict]:
        """Per-tensor counts of the last pass for every counted tensor, gathered from the owning ranks.

        Expert entries are this rank's shard (as the report has always stated);
        dense entries come from their owner rank so every rank holds all of them.
        """
        self._check_complete()
        mine = {}
        for c in self.counted.values():
            if c.owner and c.kind == "fp8":
                mine[c.name] = {"codes_flipped": int(c.codes_flipped.sum()), "values_changed": int(c.values_changed.sum()),
                                "code_elements": c.base.code_elements, "scales_changed": int(c.scales_changed.sum()),
                                "scale_elements": c.base.scale_elements}
        gathered: list = [None] * self.world
        if dist.is_initialized():
            dist.all_gather_object(gathered, mine)
        else:
            gathered[0] = mine
        out = {}
        for part in gathered:
            out.update(part)
        for c in self.counted.values():
            if c.kind != "fp8":
                out[c.name] = {"codes_flipped": int(c.codes_flipped.sum()), "values_changed": int(c.values_changed.sum()),
                               "code_elements": c.base.code_elements, "scales_changed": int(c.scales_changed.sum()),
                               "scale_elements": c.base.scale_elements}
        return out

    def per_expert(self, name: str) -> tuple[list[int], list[int], list[int]]:
        """(values changed, codes flipped, scales changed) per local expert of an expert tensor, last pass."""
        c = self.counted[name]
        if c.kind == "fp8":
            raise ValueError(f"{name} is a dense tensor")
        return c.values_changed.tolist(), c.codes_flipped.tolist(), c.scales_changed.tolist()

    def elements_per_expert(self, name: str) -> int:
        c = self.counted[name]
        return c.base.value_elements // c.base.experts
