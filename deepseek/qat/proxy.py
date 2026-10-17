"""Stage 2 parity check: fake-quant hooks at zero delta on the four-layer proxy.

Builds the model through NeMo's training recipe (so FSDP2 and expert-parallel
wrapping match training), loads the real checkpoint, and checks:

1. fixed point: every target's fake-quantized tensor equals the loaded tensor
   bit for bit, on this rank's local shard;
2. source parity: every loaded target equals the dequantized source tensor from
   the snapshot (dense: full tensor on rank 0; experts: this rank's local experts,
   by global id); under split storage (stage 6b) every packed slot also equals
   the source's codes and scales byte for byte;
3. logits parity: on fixed token ids, the hooked and unhooked forwards produce
   bit-identical logits, and uninstalling restores the unhooked logits.

Pointed at an export (``--snapshot`` and the model-path overrides naming the
export), the same checks prove the export reloads exactly; ``--expect-logits``
additionally compares the hooked logits bit for bit with those a ``proxy_step
--export`` run saved before writing it. ``--save-logits PREFIX`` writes this
run's hooked logits in that format, so two builds of the same checkpoint (the
single-stack path and split expert storage, stage 6b gate 2) can be compared.

Any failure exits nonzero. Launch from the repo root:

  deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy \
      --config deepseek/qat/proxy-4layer.yaml --snapshot $SNAP
  deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy \
      --config deepseek/qat/proxy-4layer.yaml --snapshot EXPORT \
      --set model.config.pretrained_model_name_or_path=EXPORT --set model.config.name_or_path=EXPORT \
      --expect-logits REPORT_PREFIX
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
from deepseek.qat.overrides import add_override_arguments, apply_overrides

SHARED = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

# The fixed probe input every parity and reload check shares. proxy_step saves
# logits on these ids before an export; this script recomputes them on the
# reloaded export. One definition so the two sides cannot drift.
PROBE_BATCH = 2
PROBE_SEQ = 256
PROBE_SEED = 0


def probe_token_ids(vocab: int, batch: int = PROBE_BATCH, seq: int = PROBE_SEQ, seed: int = PROBE_SEED) -> torch.Tensor:
    """Deterministic random token ids on CPU for the logits probe."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=gen)


def logits_path(prefix: str, rank: int) -> str:
    """Where a rank's probe logits are saved: beside the report named by `prefix`."""
    return f"{prefix}-logits-rank{rank}.pt"


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

    def tensor(self, name: str) -> torch.Tensor:
        """The stored tensor as written in the shard: codes, scales, or plain weights, no conversion."""
        shard = self.weight_map[name]
        if shard not in self._open:
            if len(self._open) > 4:
                self._open.clear()
            self._open[shard] = safe_open(os.path.join(self.snapshot, shard), framework="pt", device="cpu")
        return self._open[shard].get_tensor(name)

    def dense(self, name: str) -> torch.Tensor:
        w = self.tensor(name)
        s = self.tensor(name[: -len(".weight")] + ".scale")
        return Q.dequantize_fp8_blocks(w, s)

    def expert(self, layer: int, e: int, which: str) -> torch.Tensor:
        base = f"layers.{layer}.ffn.experts.{e}.{which}"
        return Q.dequantize_fp4_rows(self.tensor(base + ".weight"), self.tensor(base + ".scale"))


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def local_expert_ids(module, param: torch.Tensor) -> list[int]:
    """Global expert id of each dim-0 row of this rank's shard of a trained expert parameter.

    Split storage (stage 6b, ``SplitExperts``) maps rows through its selection;
    the single stack holds the contiguous range starting at rank times the local
    count on a one-dimensional expert mesh.
    """
    from deepseek.qat.split_experts import SplitExperts  # local: NeMo is only needed on a built model

    if isinstance(module, SplitExperts):
        return module.local_ids()[0]
    n_local = _local(param).shape[0]
    if not isinstance(param, DTensor) or not any(p.is_shard() for p in param.placements):
        return list(range(n_local))
    offset = param.device_mesh.get_local_rank() * n_local
    return [offset + i for i in range(n_local)]


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
            for i, e in enumerate(local_expert_ids(t.module, param)):
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


# Split storage (stage 6b): every packed slot of every layer must hold the
# source's codes and scales byte for byte, on this rank's shard. Empty when the
# model has no split layers.
def check_packed(model, source: Source) -> dict:
    from deepseek.qat.split_experts import SplitExperts  # local: NeMo is only needed on a built model

    bad = {}
    for key, layer in model.model.layers.items():
        m = layer.mlp.experts
        if not isinstance(m, SplitExperts) or m.n_packed == 0:
            continue
        _, ids = m.local_ids()
        for j, e in enumerate(ids):
            for attr, halves in (("gate_and_up_projs", ("w1", "w3")), ("down_projs", ("w2",))):
                for suffix, disk in (("_packed", "weight"), ("_scales", "scale")):
                    got = _local(getattr(m, attr + suffix)).detach()[j].cpu().view(torch.uint8)
                    exp = torch.cat([source.tensor(f"layers.{key}.ffn.experts.{e}.{w}.{disk}") for w in halves]).view(torch.uint8)
                    n = int((got != exp).sum()) if got.shape == exp.shape else -1
                    if n:
                        bad[f"model.layers.{key}.mlp.experts.{attr}{suffix}[expert {e}]"] = n
    return bad


def logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return getattr(out, "logits", out).detach().float()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--batch", type=int, default=PROBE_BATCH)
    parser.add_argument("--seq", type=int, default=PROBE_SEQ)
    parser.add_argument("--seed", type=int, default=PROBE_SEED)
    parser.add_argument("--skip-source-check", action="store_true")
    add_override_arguments(parser)
    parser.add_argument("--expect-logits", default=None, metavar="PREFIX",
                        help="compare hooked logits bit for bit with PREFIX-logits-rank{rank}.pt saved by proxy_step --export or --save-logits")
    parser.add_argument("--save-logits", default=None, metavar="PREFIX",
                        help="save this rank's hooked logits as PREFIX-logits-rank{rank}.pt for a later --expect-logits (stage 6b gate 2: "
                             "the single-stack path against split storage)")
    args = parser.parse_args(argv)
    if args.save_logits is not None and args.save_logits == args.expect_logits:
        parser.error("--save-logits and --expect-logits name the same prefix")

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    t0 = time.monotonic()
    cfg = load_yaml_config(args.config)
    overrides = apply_overrides(cfg, args.set, args.unset)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = recipe.model_parts[0]
    model.eval()
    device = next(model.parameters()).device
    report = {"rank": rank, "setup_seconds": round(time.monotonic() - t0, 1)}
    report["overrides"] = overrides
    report["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})

    state = FQ.install(model)
    report["targets"] = len(state.targets)
    report["fixed_point_mismatches"] = check_fixed_point(state)

    if not args.skip_source_check:
        source = Source(args.snapshot)
        report["source_mismatches"] = check_source(state, source, rank)
        report["packed_mismatches"] = check_packed(model, source)

    vocab = int(model.config.vocab_size)
    input_ids = probe_token_ids(vocab, args.batch, args.seq, args.seed).to(device)
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

    # Same file format as proxy_step --export, so either source serves --expect-logits.
    if args.save_logits is not None:
        torch.save(hooked.cpu(), logits_path(args.save_logits, rank))
        report["saved_logits_path"] = logits_path(args.save_logits, rank)

    # Reload check: the export holds Q(master) exactly, a fixed point of the
    # quantizer, so the saved logits must be reproduced bit for bit; a missing
    # file or a shape mismatch is a failure, not a skipped check.
    if args.expect_logits is not None:
        path = logits_path(args.expect_logits, rank)
        if not os.path.exists(path):
            raise SystemExit(f"--expect-logits: {path} does not exist")
        saved = torch.load(path, map_location="cpu")
        got = hooked.cpu()
        report["expected_logits_path"] = path
        report["logits_saved_shape"] = list(saved.shape)
        if saved.shape != got.shape:
            raise SystemExit(f"--expect-logits: saved logits {tuple(saved.shape)} do not match probe {tuple(got.shape)}")
        report["logits_match_saved"] = bool(torch.equal(saved, got))
        report["logits_saved_max_abs_diff"] = float((saved - got).abs().max())
        report["logits_saved_differing"] = int((saved != got).sum())
    report["elapsed_seconds"] = round(time.monotonic() - t0, 1)

    ok = (
        not report["fixed_point_mismatches"]
        and not report.get("source_mismatches")
        and not report.get("packed_mismatches")
        and report["logits_hooked_equal"]
        and report["logits_restored_equal"]
        and report.get("logits_match_saved", True)
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
