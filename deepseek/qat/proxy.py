"""Stage 2 parity check: fake-quant hooks at zero delta on the four-layer proxy.

Builds the model through NeMo's training recipe (so FSDP2 and expert-parallel
wrapping match training), loads the real checkpoint, and checks:

1. fixed point: every target's fake-quantized tensor equals the loaded tensor
   bit for bit, on this rank's local shard;
2. source parity: every loaded target equals the dequantized source tensor from
   the snapshot (dense: full tensor on rank 0; experts: this rank's local experts);
3. logits parity: on fixed token ids, the hooked and unhooked forwards produce
   bit-identical logits, and uninstalling restores the unhooked logits.

Any failure exits nonzero. Launch from the repo root:

  deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy \
      --config deepseek/qat/proxy-4layer.yaml --snapshot $SNAP
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.tensor import DTensor

from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q
from deepseek.qat.export import read_shard_header

SHARED = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


# NeMo target name -> HF names of the source tensors it was built from.
def hf_names(nemo_param: str) -> list[str]:
    parts = nemo_param.split(".")
    layer = parts[2]
    rest = parts[3:]
    if rest[:2] == ["mlp", "shared_experts"]:
        return [f"layers.{layer}.ffn.shared_experts.{SHARED[rest[2]]}.weight"]
    if rest[:3] == ["self_attn", "compressor", "indexer"]:
        return [f"layers.{layer}.attn.indexer.{rest[3]}.weight"]
    if rest[0] == "self_attn":
        return [f"layers.{layer}.attn.{rest[1]}.weight"]
    raise ValueError(f"no HF mapping for {nemo_param}")


class Source:
    """Reads dequantized tensors from the snapshot by HF name."""

    def __init__(self, snapshot: str):
        self.snapshot = snapshot
        with open(os.path.join(snapshot, "model.safetensors.index.json"), encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._open: dict[str, object] = {}

    def _tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open:
            if len(self._open) > 4:
                self._open.clear()
            self._open[shard] = safe_open(os.path.join(self.snapshot, shard), framework="pt", device="cpu")
        return self._open[shard].get_tensor(name)

    def dense(self, name: str) -> torch.Tensor:
        w = self._tensor(name)
        s = self._tensor(name[: -len(".weight")] + ".scale")
        return Q.dequantize_fp8_blocks(w, s)

    def expert(self, layer: int, e: int, which: str) -> torch.Tensor:
        base = f"layers.{layer}.ffn.experts.{e}.{which}"
        return Q.dequantize_fp4_rows(self._tensor(base + ".weight"), self._tensor(base + ".scale"))


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def _ep_offset(t: torch.Tensor) -> int:
    """First global expert index held in this rank's local shard of a dim-0-sharded DTensor."""
    if not isinstance(t, DTensor) or not any(p.is_shard() for p in t.placements):
        return 0
    mesh = t.device_mesh
    return mesh.get_local_rank() * t.to_local().shape[0]


def check_fixed_point(state: FQ.FakeQuantState) -> dict:
    bad = {}
    for name, fq in FQ.effective_weights(state):
        target = next(t for t in state.targets if name.startswith(t.name + "."))
        attr = name[len(target.name) + 1 :]
        a = _local(fq)
        b = _local(getattr(target.module, attr)).detach()
        n = int((a != b).sum())
        if n:
            bad[name] = n
    return bad


def check_source(state: FQ.FakeQuantState, source: Source, rank: int) -> dict:
    bad = {}
    for t in state.targets:
        for attr in t.params:
            param = getattr(t.module, attr).detach()
            full_name = f"{t.name}.{attr}"
            if attr == "weight":
                full = param.full_tensor() if isinstance(param, DTensor) else param
                if rank == 0:
                    expected = source.dense(hf_names(full_name)[0]).to(full.dtype)
                    n = int((full.cpu() != expected).sum())
                    if n:
                        bad[full_name] = n
                continue
            layer = int(t.name.split(".")[2])
            local = _local(param).cpu()
            offset = _ep_offset(param)
            for i in range(local.shape[0]):
                e = offset + i
                if attr == "gate_and_up_projs":
                    inter = local.shape[2] // 2
                    exp = torch.cat([source.expert(layer, e, "w1").T, source.expert(layer, e, "w3").T], dim=-1)
                    assert exp.shape[1] == 2 * inter
                else:
                    exp = source.expert(layer, e, "w2").T
                n = int((local[i] != exp.to(local.dtype)).sum())
                if n:
                    bad[f"{full_name}[expert {e}]"] = n
    return bad


def logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return getattr(out, "logits", out).detach().float()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args(argv)

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    t0 = time.monotonic()
    cfg = load_yaml_config(args.config)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = recipe.model_parts[0]
    model.eval()
    device = next(model.parameters()).device
    report = {"rank": rank, "setup_seconds": round(time.monotonic() - t0, 1)}
    report["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})

    state = FQ.install(model)
    report["targets"] = len(state.targets)
    report["fixed_point_mismatches"] = check_fixed_point(state)

    if not args.skip_source_check:
        report["source_mismatches"] = check_source(state, Source(args.snapshot), rank)

    vocab = int(model.config.vocab_size)
    gen = torch.Generator().manual_seed(args.seed)
    input_ids = torch.randint(0, vocab, (args.batch, args.seq), generator=gen).to(device)
    FQ.uninstall(state)
    plain = logits(model, input_ids)
    state = FQ.install(model)
    hooked = logits(model, input_ids)
    FQ.uninstall(state)
    restored = logits(model, input_ids)
    report["logits_shape"] = list(plain.shape)
    report["logits_hooked_equal"] = bool(torch.equal(plain, hooked))
    report["logits_hooked_max_abs_diff"] = float((plain - hooked).abs().max())
    report["logits_hooked_differing"] = int((plain != hooked).sum())
    report["logits_restored_equal"] = bool(torch.equal(plain, restored))
    report["elapsed_seconds"] = round(time.monotonic() - t0, 1)

    ok = (
        not report["fixed_point_mismatches"]
        and not report.get("source_mismatches")
        and report["logits_hooked_equal"]
        and report["logits_restored_equal"]
    )
    report["ok"] = ok
    print(json.dumps(report, indent=1), flush=True)
    if dist.is_initialized():
        flag = torch.tensor([0 if ok else 1], device=device)
        dist.all_reduce(flag)
        ok = int(flag.item()) == 0
        dist.barrier()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(code)
