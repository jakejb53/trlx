"""State-dict adapter for split expert storage (stage 6b).

NeMo loads a checkpoint after sharding under expert parallelism: ``to_hf`` turns
the model's own state dict into per-expert HF destinations that DCP fills,
``from_hf`` rebuilds the native stacks from what was read. The base DeepSeek-V4
adapter knows one stack per layer whose local slot ``i`` is global expert
``rank * E/R + i``. With ``SplitExperts`` a layer has a selected BF16 stack and a
packed stack whose slots map to global ids through ``ExpertSplit``, so this
subclass owns the expert keys of split layers:

- ``to_hf``: a selected slot yields the base adapter's empty FP4 placeholders
  (int8 codes and E8M0 scales in the on-disk layout, dequantized on the way
  back); a packed slot yields views into the module's own storage, which DCP
  writes straight through, so unselected experts are never dequantized.
- ``from_hf``: selected experts are dequantized and stacked in NeMo's compute
  layout; unselected experts keep their codes and scales and are stacked in
  checkpoint layout. Both stacks are emitted per layer as DTensors on the
  expert mesh.

Every other key is handled by the base adapter unchanged.
"""

import re

import torch
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import _EXPERT_PATTERN, DeepSeekV4StateDictAdapter
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_submesh,
    should_load_expert_for_rank,
)

from deepseek.qat.split_experts import ExpertSplit

_NATIVE_STACK = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_and_up_projs|down_projs)(_packed|_scales)?$")
_HF_EXPERT_SCALE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.scale$")
HALVES = {"gate_and_up_projs": ("w1", "w3"), "down_projs": ("w2",)}


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


class SplitStateDictAdapter(DeepSeekV4StateDictAdapter):
    """DeepSeek-V4 adapter routing each expert into the BF16 or packed stack."""

    def __init__(self, base: DeepSeekV4StateDictAdapter, split: ExpertSplit):
        super().__init__(base.config, base.moe_config, base.backend, dtype=base.dtype,
                         expert_storage_format=base.expert_storage_format)
        # Selected experts load through the dequantizing path; the packed stacks
        # are routed here, not by NeMo's all-or-nothing mxfp4 passthrough.
        if self.expert_storage_format != "bf16":
            raise ValueError(f"split expert storage needs the bf16 expert storage format, got {self.expert_storage_format!r}")
        self.split = split

    # The local rows of a stack tensor and the global expert id of each row.
    # Sharded stacks carry Shard(0) on the expert mesh and map through this
    # rank's selection; an unsharded stack (pre-shard key listing, single rank)
    # holds every rank's rows in rank order.
    def _stack_rows(self, tensor: torch.Tensor, layer: int, kind: str) -> tuple[torch.Tensor, list[int]]:
        if isinstance(tensor, DTensor):
            mesh = tensor.device_mesh
            names = tuple(mesh.mesh_dim_names or ())
            if "ep" not in names:
                raise NotImplementedError(f"split expert stack on a mesh without an 'ep' dimension: {names}")
            for name, placement in zip(names, tensor.placements):
                expected = isinstance(placement, Shard) and placement.dim == 0 if name == "ep" else placement.is_replicate()
                if not expected:
                    raise NotImplementedError(f"split expert stack needs Shard(0) on 'ep' and replication elsewhere, got {tensor.placements} on {names}")
            rank = get_submesh(mesh, ("ep",)).get_local_rank()
            local = tensor.to_local()
        else:
            rank = None
            local = tensor
        ids = self.split.stack_ids(layer, kind, rank)
        if local.shape[0] != len(ids):
            raise ValueError(f"layer {layer} {kind} stack has {local.shape[0]} rows for {len(ids)} experts")
        return local, ids

    def convert_single_tensor_to_hf(self, fqn: str, tensor, **kwargs) -> list[tuple[str, object]]:
        m = _NATIVE_STACK.match(fqn)
        if m is None or not self.split.has_layer(int(m.group(1))):
            return super().convert_single_tensor_to_hf(fqn, tensor, **kwargs)
        layer, which, suffix = int(m.group(1)), m.group(2), m.group(3)
        kind = "selected" if suffix is None else "packed"
        local, ids = self._stack_rows(tensor, layer, kind)
        disk = "scale" if suffix == "_scales" else "weight"
        result = []
        for i, e in enumerate(ids):
            t = local[i]
            if which == "gate_and_up_projs":
                # Compute layout [dim, 2*inter] splits along the last axis and
                # transposes; checkpoint layout [2*inter, in/2 or in/32] splits
                # along the first axis as stored.
                if suffix is None:
                    gate, up = t.split(t.shape[-1] // 2, dim=-1)
                    halves = (gate.T, up.T)
                else:
                    halves = t.split(t.shape[0] // 2, dim=0)
            else:
                halves = (t.T if suffix is None else t,)
            for w, value in zip(HALVES[which], halves):
                result.append((f"layers.{layer}.ffn.experts.{e}.{w}.{disk}", value))
        exclude = kwargs.get("exclude_key_regex")
        if exclude:
            result = [(k, v) for k, v in result if not re.match(exclude, k)]
        if kwargs.get("quantization", False) and suffix is None:
            # DCP validates shape and dtype against the checkpoint before anything
            # is dequantized, so selected experts need placeholders in the on-disk
            # FP4 layout; packed rows already are that layout.
            placeholders = []
            for key, value in result:
                packed, scale = self._build_fp4_expert_placeholders(value)
                placeholders.append((key, packed))
                placeholders.append((key[: -len(".weight")] + ".scale", scale))
            result = placeholders
        return result

    # Unselected experts keep their codes and scales; everything else goes
    # through the base dequantization.
    def _dequantize(self, state_dict: dict) -> dict:
        kept = {}
        for key in list(state_dict):
            m = _EXPERT_PATTERN.match(key)
            if m is None:
                continue
            layer, e = int(m.group(1)), int(m.group(2))
            if self.split.has_layer(layer) and not self.split.is_selected(layer, e):
                kept[key] = state_dict.pop(key)
                scale = key[: -len(".weight")] + ".scale"
                if scale in state_dict:
                    kept[scale] = state_dict.pop(scale)
        out = super()._dequantize(state_dict)
        out.update(kept)
        return out

    def _aggregate_experts(self, state_dict: dict, device_mesh) -> dict:
        n_experts = self.moe_config.n_routed_experts
        if device_mesh is not None:
            ep_mesh = get_submesh(device_mesh, ("ep",)) if "ep" in device_mesh.mesh_dim_names else device_mesh
            if ep_mesh.size() != self.split.ranks:
                raise ValueError(f"{self.split.source}: selection is for {self.split.ranks} ranks, expert mesh has {ep_mesh.size()}")
            ep_rank = ep_mesh.get_local_rank()
            rank = ep_mesh.get_rank()
        else:
            if self.split.ranks != 1:
                raise ValueError(f"{self.split.source}: selection is for {self.split.ranks} ranks, loading without an expert mesh")
            ep_rank, rank = 0, None

        selected: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
        packed: dict[int, dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = {}
        out: dict = {}
        for key in list(state_dict):
            m = _EXPERT_PATTERN.match(key)
            if m is None:
                if _HF_EXPERT_SCALE.match(key) is None:
                    out[key] = state_dict.pop(key)
                continue
            layer, e, which = int(m.group(1)), int(m.group(2)), m.group(3)
            if not self.split.has_layer(layer):
                raise ValueError(f"{key}: layer {layer} is not covered by the expert split")
            value = state_dict.pop(key)
            scale_key = key[: -len(".weight")] + ".scale"
            if not should_load_expert_for_rank(e, device_mesh, n_experts):
                state_dict.pop(scale_key, None)
                continue
            if self.split.is_selected(layer, e):
                selected.setdefault(layer, {}).setdefault(e, {})[which] = _local(value)
                continue
            scale = state_dict.pop(scale_key, None)
            if scale is None:
                raise ValueError(f"{key}: unselected expert has no {scale_key}")
            codes, scale = _local(value), _local(scale)
            if codes.dtype != torch.int8 or scale.dtype != torch.float8_e8m0fnu:
                raise ValueError(f"{key}: packed storage needs int8 codes with E8M0 scales, got {codes.dtype} and {scale.dtype}")
            packed.setdefault(layer, {}).setdefault(e, {})[which] = (codes, scale)
        orphans = sorted(state_dict)
        if orphans:
            raise ValueError(f"expert scale tensors without a weight: {orphans[:5]}")

        for layer in sorted(set(selected) | set(packed)):
            sel_ids = self.split.rank_selected(layer, ep_rank)
            pk_ids = self.split.rank_unselected(layer, ep_rank)
            self._check_complete(layer, "selected", selected.get(layer, {}), sel_ids)
            self._check_complete(layer, "packed", packed.get(layer, {}), pk_ids)
            base = f"model.layers.{layer}.mlp.experts."
            if sel_ids:
                rows = selected.pop(layer)
                gate_up = torch.stack([torch.cat([rows[e]["w1"].T, rows[e]["w3"].T], dim=-1) for e in sel_ids]).to(self.dtype)
                out[base + "gate_and_up_projs"] = create_dtensor_from_local(gate_up, device_mesh, rank)
                down = torch.stack([rows[e]["w2"].T for e in sel_ids]).to(self.dtype)
                out[base + "down_projs"] = create_dtensor_from_local(down, device_mesh, rank)
                del rows, gate_up, down
            if pk_ids:
                rows = packed.pop(layer)
                for part in (0, 1):
                    suffix = "_packed" if part == 0 else "_scales"
                    gate_up = torch.stack([torch.cat([rows[e]["w1"][part], rows[e]["w3"][part]], dim=0) for e in pk_ids])
                    out[base + "gate_and_up_projs" + suffix] = create_dtensor_from_local(gate_up, device_mesh, rank)
                    down = torch.stack([rows[e]["w2"][part] for e in pk_ids])
                    out[base + "down_projs" + suffix] = create_dtensor_from_local(down, device_mesh, rank)
                del rows
        return out

    @staticmethod
    def _check_complete(layer: int, kind: str, rows: dict, ids: list[int]) -> None:
        missing = [e for e in ids if set(rows.get(e, {})) != {"w1", "w2", "w3"}]
        extra = sorted(set(rows) - set(ids))
        if missing or extra:
            raise ValueError(f"layer {layer} {kind} experts: incomplete {missing[:5]}, unexpected {extra[:5]}")
