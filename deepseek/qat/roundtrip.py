"""Read-only round-trip check of the quantizers against a real checkpoint.

For every quantized weight in a DeepSeek-V4-Flash snapshot (FP8 E4M3 dense
weights and packed FP4 routed-expert weights, each with its ``.scale`` tensor):
dequantize with ``deepseek.qat.quant``, re-quantize, and compare.

Two levels are reported. *Value-exact* means the re-quantized tensor dequantizes
to exactly the same values; this is required and any failure exits nonzero.
*Bit-exact* means codes and scales are identical. FP4 is bit-exact under the
floor rule. FP8 legitimately differs on blocks whose largest source code is
exactly 224: the source stores scale 2s with code 224 where the ceiling rule
stores scale s with code 448, the same value. The scale-ratio histogram shows
such differences as +1. Negative-zero FP4 codes and all-zero blocks are
counted because the reference rule for them is not published.

Usage: python -m deepseek.qat.roundtrip SNAPSHOT_DIR [--limit N] [--amax-floor F] [--scale-rule floor|min_error]
"""

import argparse
import collections
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open

from deepseek.qat import quant as Q


# Read every shard header once so tensors can be classified without loading data.
def read_headers(snapshot: str) -> dict[str, tuple[str, str, tuple[int, ...]]]:
    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    headers: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    for shard in sorted(set(weight_map.values())):
        with open(os.path.join(snapshot, shard), "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        header.pop("__metadata__", None)
        for name, meta in header.items():
            headers[name] = (shard, meta["dtype"], tuple(meta["shape"]))
    missing = set(weight_map) - set(headers)
    if missing:
        raise SystemExit(f"{len(missing)} index entries have no header entry, e.g. {sorted(missing)[:3]}")
    return headers


class ClassStats:
    """Counters for one tensor class (fp8 or fp4)."""

    def __init__(self) -> None:
        self.tensors = 0
        self.value_exact = 0
        self.codes_exact = 0
        self.scales_exact = 0
        self.negative_zero_codes = 0
        self.zero_blocks = 0
        self.zero_block_scales: collections.Counter = collections.Counter()
        self.scale_ratio: collections.Counter = collections.Counter()
        self.failures: list[str] = []


def check_fp8(name: str, weight: torch.Tensor, scale: torch.Tensor, floor: float, st: ClassStats) -> None:
    dq = Q.dequantize_fp8_blocks(weight, scale)
    codes, scales = Q.quantize_fp8_blocks(dq, amax_floor=floor)
    _record(name, weight.view(torch.uint8), scale, codes.view(torch.uint8), scales, dq,
            Q.dequantize_fp8_blocks(codes, scales), st)
    blocks = dq.reshape(dq.shape[0] // Q.FP8_BLOCK, Q.FP8_BLOCK, dq.shape[1] // Q.FP8_BLOCK, Q.FP8_BLOCK)
    zero = blocks.abs().amax(dim=(1, 3)) == 0
    _record_zero_blocks(zero, scale, st)


def check_fp4(name: str, packed: torch.Tensor, scale: torch.Tensor, floor: float, st: ClassStats, scale_rule: str = "floor") -> None:
    dq = Q.dequantize_fp4_rows(packed, scale)
    new_packed, scales = Q.quantize_fp4_rows(dq, amax_floor=floor, scale_rule=scale_rule)
    _record(name, packed.view(torch.uint8), scale, new_packed, scales, dq,
            Q.dequantize_fp4_rows(new_packed, scales), st)
    st.negative_zero_codes += int((Q.unpack_fp4_codes(packed) == 8).sum())
    groups = dq.reshape(dq.shape[0], dq.shape[1] // Q.FP4_GROUP, Q.FP4_GROUP)
    _record_zero_blocks(groups.abs().amax(dim=-1) == 0, scale, st)


def _record(name, src_codes, src_scale, new_codes, new_scale, dq, redq, st: ClassStats) -> None:
    st.tensors += 1
    if torch.equal(dq, redq):
        st.value_exact += 1
    else:
        bad = (dq != redq).sum().item()
        st.failures.append(f"{name}: {bad} values differ after re-quantization")
    if torch.equal(src_codes, new_codes):
        st.codes_exact += 1
    src_s = src_scale.float()
    new_s = new_scale.float()
    if torch.equal(src_s, new_s):
        st.scales_exact += 1
    else:
        # Ratio source/new per differing block, as log2, e.g. 1 means the source scale is double ours.
        diff = src_s != new_s
        ratios = torch.log2(src_s[diff] / new_s[diff]).round().to(torch.int64)
        for r, c in zip(*torch.unique(ratios, return_counts=True)):
            st.scale_ratio[int(r)] += int(c)


def _record_zero_blocks(zero_mask: torch.Tensor, src_scale: torch.Tensor, st: ClassStats) -> None:
    n = int(zero_mask.sum())
    if n:
        st.zero_blocks += n
        for s, c in zip(*torch.unique(src_scale.float()[zero_mask], return_counts=True)):
            st.zero_block_scales[float(s)] += int(c)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("snapshot", help="snapshot directory containing model.safetensors.index.json")
    parser.add_argument("--limit", type=int, default=0, help="check at most N tensors per class (0 = all)")
    parser.add_argument("--amax-floor", type=float, default=0.0, help="minimum block absmax before the scale rule")
    parser.add_argument("--scale-rule", choices=Q.FP4_SCALE_RULES, default="floor",
                        help="FP4 scale rule to requantize with; min_error is the training rule and must reproduce the release")
    args = parser.parse_args(argv)

    headers = read_headers(args.snapshot)
    stats = {"fp8": ClassStats(), "fp4": ClassStats()}
    todo: list[tuple[str, str, str]] = []
    for name, (shard, dtype, shape) in headers.items():
        if not name.endswith(".weight"):
            continue
        scale_name = name[: -len(".weight")] + ".scale"
        if dtype == "F8_E4M3" and scale_name in headers:
            todo.append(("fp8", name, scale_name))
        elif dtype == "I8" and scale_name in headers:
            todo.append(("fp4", name, scale_name))
    if args.limit:
        limited = []
        counts = collections.Counter()
        for item in todo:
            if counts[item[0]] < args.limit:
                limited.append(item)
                counts[item[0]] += 1
        todo = limited
    print(f"checking {len(todo)} tensors", file=sys.stderr, flush=True)

    start = time.monotonic()
    last = start
    open_shards: dict[str, object] = {}
    for i, (kind, name, scale_name) in enumerate(todo, 1):
        shard = headers[name][0]
        if shard not in open_shards:
            open_shards.clear()
            open_shards[shard] = safe_open(os.path.join(args.snapshot, shard), framework="pt", device="cpu")
        f = open_shards[shard]
        weight = f.get_tensor(name)
        scale_shard = headers[scale_name][0]
        if scale_shard == shard:
            scale = f.get_tensor(scale_name)
        else:
            with safe_open(os.path.join(args.snapshot, scale_shard), framework="pt", device="cpu") as g:
                scale = g.get_tensor(scale_name)
        if kind == "fp8":
            check_fp8(name, weight, scale, args.amax_floor, stats["fp8"])
        else:
            check_fp4(name, weight, scale, args.amax_floor, stats["fp4"], args.scale_rule)
        now = time.monotonic()
        if now - last >= 10:
            print(f"  {i}/{len(todo)} after {now - start:.0f}s", file=sys.stderr, flush=True)
            last = now

    failed = False
    for kind, st in stats.items():
        print(f"== {kind}: {st.tensors} tensors")
        if not st.tensors:
            continue
        print(f"  value-exact : {st.value_exact}/{st.tensors}")
        print(f"  codes-exact : {st.codes_exact}/{st.tensors}")
        print(f"  scales-exact: {st.scales_exact}/{st.tensors}")
        if st.scale_ratio:
            print("  log2(source scale / our scale) over differing blocks: " + ", ".join(
                f"{r:+d}: {c}" for r, c in sorted(st.scale_ratio.items())))
        if kind == "fp4":
            print(f"  negative-zero codes in source: {st.negative_zero_codes}")
        print(f"  all-zero blocks: {st.zero_blocks}" + (
            " with source scales " + ", ".join(f"{s:g} x{c}" for s, c in sorted(st.zero_block_scales.items()))
            if st.zero_blocks else ""))
        for line in st.failures[:20]:
            print("  FAIL " + line)
        if st.failures:
            failed = True
    print(f"elapsed {time.monotonic() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
