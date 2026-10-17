"""Split expert storage (stage 6b): trained BF16 stack plus frozen packed stack.

Each layer's routed experts are held in two stacks. Experts selected for
training keep NeMo's layout, ``gate_and_up_projs`` ``[R*S, dim, 2*inter]`` and
``down_projs`` ``[R*S, inter, dim]`` in BF16, so the fake-quant hook, the
update-in-backward stepper, the optimizer, and the exporter see the parameter
names and orientation they already handle, only shorter along dim 0. Unselected
experts are held as the source checkpoint's packed FP4 codes and E8M0 scales in
checkpoint orientation, ``*_packed`` ``[R*U, out, in/2]`` int8 and ``*_scales``
``[R*U, out, in/32]``, frozen, and dequantized on the fly inside
``MXFP4GroupedMM`` in forward and backward with no weight gradient. An
unselected expert's ``dequant(Q(w))`` is ``dequant(codes, scales)``, so the two
storages compute the same function as the single BF16 stack they replace.

Shards are even by construction: every expert-parallel rank selects the same
count ``S`` per layer within its own contiguous expert range, so ``Shard(0)``
over the expert mesh gives each rank ``S`` selected and ``U`` packed experts.
``ExpertSplit`` reads that selection from a ``route_scores --per-rank`` report
and swaps the modules and the state-dict adapter into a built model before
sharding; ``R`` is the report's rank count and must equal the expert-parallel
size of the run.

Local slot order is selected first: slots ``0..S-1`` are this rank's selected
experts in ascending global id, slots ``S..S+U-1`` its unselected ones. The
slot table is derived from the mesh rank on first use and cached, never stored
as a buffer: the pre-shard state dict includes persistent buffers and DCP
requires every key in the checkpoint.
"""

import json

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import GroupedExperts, _permute_tokens_for_grouped_mm
from nemo_automodel.components.quantization.mxfp4 import MXFP4_BLOCK_SIZE, MXFP4GroupedMM, dequantize_mxfp4

PACKED_SUFFIXES = ("_packed", "_scales")


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class ExpertSplit:
    """Per-layer, per-rank expert selection from a ``route_scores`` report.

    The report must carry ``ranks`` (written by ``--per-rank``), ``n_routed_experts``,
    and ``layers.<L>.selected``. Every layer must select the same number of experts
    inside each rank's contiguous range ``[r*E/R, (r+1)*E/R)``.
    """

    def __init__(self, scores: str):
        with open(scores, encoding="utf-8") as f:
            report = json.load(f)
        if "ranks" not in report:
            raise ValueError(f"{scores}: no 'ranks' field; split storage needs a selection written by route_scores --per-rank")
        layers = {int(k): [int(e) for e in v["selected"]] for k, v in report.get("layers", {}).items()}
        self._init(layers, int(report["n_routed_experts"]), int(report["ranks"]), source=scores)

    @classmethod
    def from_layers(cls, layers: dict[int, list[int]], n_experts: int, ranks: int) -> "ExpertSplit":
        """Build a split from an in-memory selection (tests and callers without a report)."""
        self = cls.__new__(cls)
        self._init(dict(layers), n_experts, ranks, source="<layers>")
        return self

    def _init(self, layers: dict[int, list[int]], n_experts: int, ranks: int, source: str) -> None:
        if ranks < 1 or n_experts < 1 or n_experts % ranks:
            raise ValueError(f"{source}: {n_experts} experts do not divide over {ranks} ranks")
        if not layers:
            raise ValueError(f"{source}: no layers in the selection")
        self.source = source
        self.n_experts = n_experts
        self.ranks = ranks
        self.per_rank_total = n_experts // ranks
        self.layers: dict[int, list[int]] = {}
        self._selected_sets: dict[int, set[int]] = {}
        for layer, ids in layers.items():
            uniq = sorted(set(ids))
            if len(uniq) != len(ids):
                raise ValueError(f"{source}: layer {layer} lists an expert twice")
            if uniq and not (0 <= uniq[0] and uniq[-1] < n_experts):
                raise ValueError(f"{source}: layer {layer} selects an expert outside [0, {n_experts})")
            counts = [sum(1 for e in uniq if r * self.per_rank_total <= e < (r + 1) * self.per_rank_total) for r in range(ranks)]
            if len(set(counts)) != 1:
                raise ValueError(f"{source}: layer {layer} selects {counts} experts per rank; split storage needs an equal count")
            self.layers[layer] = uniq
            self._selected_sets[layer] = set(uniq)

    def has_layer(self, layer: int) -> bool:
        return layer in self.layers

    def is_selected(self, layer: int, e: int) -> bool:
        return e in self._selected_sets[layer]

    def selected_per_rank(self, layer: int) -> int:
        return len(self.layers[layer]) // self.ranks

    def rank_range(self, rank: int) -> tuple[int, int]:
        if not (0 <= rank < self.ranks):
            raise ValueError(f"rank {rank} outside the {self.ranks}-rank selection")
        return rank * self.per_rank_total, (rank + 1) * self.per_rank_total

    def rank_selected(self, layer: int, rank: int) -> list[int]:
        lo, hi = self.rank_range(rank)
        return [e for e in self.layers[layer] if lo <= e < hi]

    def rank_unselected(self, layer: int, rank: int) -> list[int]:
        lo, hi = self.rank_range(rank)
        chosen = self._selected_sets[layer]
        return [e for e in range(lo, hi) if e not in chosen]

    # Global expert id of every slot of a stack, for one rank's local shard or,
    # with rank None, for the whole stack in rank order (the pre-shard layout).
    def stack_ids(self, layer: int, kind: str, rank: int | None) -> list[int]:
        if kind not in ("selected", "packed"):
            raise ValueError(f"unknown stack kind {kind!r}")
        per_rank = self.rank_selected if kind == "selected" else self.rank_unselected
        ranks = range(self.ranks) if rank is None else [rank]
        return [e for r in ranks for e in per_rank(layer, r)]

    def install(self, model: nn.Module) -> None:
        """Swap every backbone layer's experts module and the state-dict adapter.

        Runs inside the model constructor, before parallelization and loading, so
        the new modules are sharded and filled by NeMo's own machinery.
        """
        from deepseek.qat.split_adapter import SplitStateDictAdapter

        n_experts = int(model.config.n_routed_experts)
        if n_experts != self.n_experts:
            raise ValueError(f"{self.source}: selection covers {self.n_experts} experts, model has {n_experts}")
        if dist.is_initialized() and dist.get_world_size() != self.ranks:
            raise ValueError(f"{self.source}: selection is for {self.ranks} ranks, world size is {dist.get_world_size()}")
        if getattr(model, "mtp", None) is not None:
            raise ValueError("split expert storage covers backbone layers only; build with num_nextn_predict_layers: 0")
        layer_ids = sorted(int(k) for k in model.model.layers.keys())
        if layer_ids != sorted(self.layers):
            raise ValueError(f"{self.source}: selection covers layers {sorted(self.layers)}, model has {layer_ids}")
        adapter = getattr(model, "state_dict_adapter", None)
        if adapter is None:
            raise ValueError("split expert storage needs backend.enable_hf_state_dict_adapter: the adapter routes experts into the stacks")
        for key, layer in model.model.layers.items():
            layer.mlp.experts = SplitExperts(layer.mlp.experts, self, int(key))
        model.state_dict_adapter = SplitStateDictAdapter(adapter, self)


class SplitExperts(GroupedExperts):
    """Routed experts of one layer as a trained BF16 stack and a frozen packed stack.

    Built from the plain ``GroupedExperts`` the model constructed (normally on the
    meta device); parameter contents come from the checkpoint, never from an
    initializer (``_mxfp4_resident`` makes NeMo's skip the module).
    """

    def __init__(self, orig: GroupedExperts, split: ExpertSplit, layer: int):
        if type(orig) is not GroupedExperts:
            raise TypeError(f"split expert storage replaces plain GroupedExperts, got {type(orig).__name__}")
        with torch.device(orig.gate_and_up_projs.device):
            super().__init__(orig.config, backend=None)
        self.use_torch_mm = orig.use_torch_mm
        if not self.use_torch_mm:
            raise NotImplementedError("split expert storage needs backend.experts='torch_mm'; the per-expert loop has no packed variant")
        if self.expert_bias or self.config.apply_router_weight_after_down:
            raise NotImplementedError("split expert storage does not implement expert biases or apply_router_weight_after_down")
        cfg = self.config
        self.split = split
        self.layer = layer
        self.ranks = split.ranks
        self.n_selected = split.selected_per_rank(layer)
        self.n_packed = split.per_rank_total - self.n_selected
        expert_dim, inter = cfg.expert_dim, cfg.moe_inter_dim
        up_dim = inter * 2 if self.is_gated else inter
        if expert_dim % MXFP4_BLOCK_SIZE or inter % MXFP4_BLOCK_SIZE:
            raise ValueError(f"expert dims must be divisible by {MXFP4_BLOCK_SIZE} for packed storage (expert_dim={expert_dim}, inter={inter})")
        device, dtype = orig.gate_and_up_projs.device, cfg.dtype
        # The full-size stacks from GroupedExperts.__init__ are replaced: BF16
        # stacks over the selected experts of every rank, packed stacks over the
        # unselected ones. A stack with no experts is not registered at all.
        del self._parameters["gate_and_up_projs"]
        del self._parameters["down_projs"]
        s, u = self.ranks * self.n_selected, self.ranks * self.n_packed
        if s:
            self.register_parameter("gate_and_up_projs", nn.Parameter(torch.empty(s, expert_dim, up_dim, dtype=dtype, device=device)))
            self.register_parameter("down_projs", nn.Parameter(torch.empty(s, inter, expert_dim, dtype=dtype, device=device)))
        else:
            self.gate_and_up_projs = None
            self.down_projs = None
        if u:
            shapes = {
                "gate_and_up_projs": ((u, up_dim, expert_dim // 2), (u, up_dim, expert_dim // MXFP4_BLOCK_SIZE)),
                "down_projs": ((u, expert_dim, inter // 2), (u, expert_dim, inter // MXFP4_BLOCK_SIZE)),
            }
            for name, (packed_shape, scale_shape) in shapes.items():
                self.register_parameter(name + "_packed", nn.Parameter(torch.empty(packed_shape, dtype=torch.int8, device=device), requires_grad=False))
                self.register_parameter(name + "_scales", nn.Parameter(torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device=device), requires_grad=False))
        self._mxfp4_resident = True
        self._slot_tables: dict[torch.device, torch.Tensor] = {}

    # Any stack parameter: the mesh anchor for rank and expert-parallel size.
    def _anchor(self) -> torch.Tensor:
        for name in ("gate_and_up_projs", "gate_and_up_projs_packed"):
            p = getattr(self, name, None)
            if p is not None:
                return p
        raise RuntimeError(f"layer {self.layer}: experts module holds no stack")

    # Expert-parallel mesh, size, and this rank's index, from the stacks' placement.
    def _ep(self):
        anchor = self._anchor()
        if isinstance(anchor, DTensor):
            mesh = anchor.device_mesh
            if mesh.ndim != 1:
                raise RuntimeError(f"layer {self.layer}: split experts need a one-dimensional expert mesh, got {mesh.ndim} dims")
            size, rank = mesh.size(), mesh.get_local_rank()
        else:
            mesh, size, rank = None, 1, 0
        if size != self.ranks:
            raise RuntimeError(f"layer {self.layer}: selection is for {self.ranks} ranks, stacks are sharded over {size}")
        return mesh, size, rank

    def local_ids(self) -> tuple[list[int], list[int]]:
        """Global expert ids of this rank's selected slots and packed slots, in slot order."""
        _, _, rank = self._ep()
        return self.split.rank_selected(self.layer, rank), self.split.rank_unselected(self.layer, rank)

    # Global expert id -> local slot (selected first), -1 for experts on other ranks.
    def _slot_table(self, device: torch.device, rank: int) -> torch.Tensor:
        table = self._slot_tables.get(device)
        if table is None:
            table = torch.full((self.split.n_experts,), -1, dtype=torch.int64)
            selected, unselected = self.split.rank_selected(self.layer, rank), self.split.rank_unselected(self.layer, rank)
            table[selected] = torch.arange(len(selected))
            table[unselected] = torch.arange(len(unselected)) + len(selected)
            table = table.to(device)
            self._slot_tables[device] = table
        return table

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Same tensor and expert-parallel contract as ``GroupedExperts.forward``."""
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype
        ep_mesh, ep_size, ep_rank = self._ep()
        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            local_num_tokens = x.size(0)
            x, weights, indices, token_mask, gathered_lens = self._gather_ep_inputs(
                x, token_mask, weights, indices, ep_group=ep_group, ep_size=ep_size
            )
        y = self._forward_split(x, token_mask, weights, indices, ep_rank)
        if ep_size > 1:
            # Keep the differentiable all-gather path attached to x without materializing a full-size zero tensor.
            y.add_(x.sum(dtype=torch.float32) * 0.0)
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()
        return y.to(input_dtype)

    # This rank's contribution before the expert-parallel reduction: one
    # permutation over both stacks in slot order, the BF16 grouped matmuls on the
    # selected tokens, the packed grouped matmuls on the rest, one scatter.
    def _forward_split(self, x, token_mask, weights, indices, ep_rank: int) -> torch.Tensor:
        s, u = self.n_selected, self.n_packed
        # Global ids to local slots; a negative id (a dropped assignment) must not
        # wrap around to the last expert, so it stays out of the local range.
        ids = indices.long()
        slots = self._slot_table(indices.device, ep_rank)[ids.clamp_min(0)].masked_fill(ids < 0, -1)
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            slots, weights, token_mask, s + u, 0
        )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        counts = tokens_per_expert.tolist()
        n_sel, total = sum(counts[:s]), sum(counts)
        compute = x.dtype
        if total == 0:
            self._dummy(x, weights, y, compute)
            return y
        permuted_x = x[sorted_token_ids]
        probs = sorted_weights.unsqueeze(-1)
        parts = []
        if s:
            if n_sel:
                gate_up = _local(self.gate_and_up_projs).to(compute)
                down = _local(self.down_projs).to(compute)
                out1 = torch._grouped_mm(permuted_x[:n_sel], gate_up, offs=offs[:s])
                out1 = self.expert_activation_grouped(out1, probs[:n_sel])
                parts.append(torch._grouped_mm(out1, down, offs=offs[:s]))
            else:
                # No token reached a selected expert on this rank this step. The
                # BF16 stack still has to receive a (zero) gradient: the
                # update-in-backward stepper requires every trainable parameter's
                # gradient once per step.
                self._dummy(x, weights, y, compute)
        if u and total > n_sel:
            offs_packed = offs[s:] - n_sel
            out1 = MXFP4GroupedMM.apply(permuted_x[n_sel:], _local(self.gate_and_up_projs_packed), _local(self.gate_and_up_projs_scales), offs_packed)
            out1 = self.expert_activation_grouped(out1, probs[n_sel:])
            parts.append(MXFP4GroupedMM.apply(out1, _local(self.down_projs_packed), _local(self.down_projs_scales), offs_packed))
        out = parts[0] if len(parts) == 1 else torch.cat(parts)
        if out.shape[0] != total:
            raise RuntimeError(f"layer {self.layer}: expert outputs cover {out.shape[0]} of {total} routed tokens")
        y.scatter_add_(0, sorted_token_ids.unsqueeze(1).expand_as(out), out.float())
        return y

    # NeMo's zero contribution through expert 0 so the activations stay in the
    # graph and the stack's parameters get a gradient: the BF16 stack when it
    # exists, otherwise the dequantized packed expert.
    def _dummy(self, x, weights, y, compute) -> None:
        if self.n_selected:
            gate_up0 = _local(self.gate_and_up_projs)[0].to(compute)
            down0 = _local(self.down_projs)[0].to(compute)
        else:
            gate_up0 = dequantize_mxfp4(_local(self.gate_and_up_projs_packed)[0], _local(self.gate_and_up_projs_scales)[0], compute).transpose(-2, -1)
            down0 = dequantize_mxfp4(_local(self.down_projs_packed)[0], _local(self.down_projs_scales)[0], compute).transpose(-2, -1)
        out1 = torch.matmul(x[0] * 0, gate_up0)
        out1 = self.expert_activation_grouped(out1, weights[0, 0, None].unsqueeze(0))
        out2 = torch.matmul(out1, down0)
        y[0] += out2[0]
