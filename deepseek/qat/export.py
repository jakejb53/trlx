"""Write a checkpoint in the exact layout of a source snapshot, replacing tensors.

A DeepSeek-V4-Flash export must be indistinguishable from the released
checkpoint except for the payloads of trained tensors. Every replacement keeps
its source tensor's name, shape, dtype and byte size, so every shard's header
bytes, tensor order and offsets are copied verbatim from the source; only the
replaced payload ranges are overwritten. Shards with no replacement are
hard-linked to the source's files when both live on one filesystem, and copied
byte for byte otherwise; either way the bytes are the source's, and linking
makes an export cost only its replaced shards, so a run can publish one at
every validation and keep several. Non-weight files (config, tokenizer,
encoding and inference resources, index) are always copied: a hard link would
make an edit inside the export, such as a serving config change, silently edit
the source.

Replacements map a tensor name to a tensor whose storage dtype and byte count
match the source entry; shape is compared on the logical shape recorded in the
header. Packed FP4 experts are stored as I8 in the checkpoint, so a uint8
packed tensor from ``quant.quantize_fp4_rows`` is accepted for an I8 entry.
Scales are ordinary entries and are replaced the same way.

The output is staged in a sibling directory and renamed into place only after
every shard is written and verified, following the repository's publication
policy; an existing destination needs ``force``.
"""

import errno
import json
import os
import shutil
import struct
import sys
import time

import torch

SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I64": torch.int64,
}
# Storage dtypes accepted for each header dtype beyond the exact one.
COMPATIBLE_STORAGE = {"I8": {torch.uint8}}

COPY_BUFFER = 64 << 20


class ExportError(Exception):
    """An export precondition or verification failure with operator context."""


# Parse one shard's header: returns (header_length, header_dict) without
# touching tensor data.
def read_shard_header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return size, header


# Check a replacement against its header entry so a wrong tensor can never be
# written: shape, element dtype, and byte count must all agree.
def _check_replacement(name: str, meta: dict, tensor: torch.Tensor) -> torch.Tensor:
    expected_dtype = SAFETENSORS_DTYPES.get(meta["dtype"])
    if expected_dtype is None:
        raise ExportError(f"{name}: unsupported checkpoint dtype {meta['dtype']}")
    if tensor.dtype != expected_dtype and tensor.dtype not in COMPATIBLE_STORAGE.get(meta["dtype"], set()):
        raise ExportError(f"{name}: replacement dtype {tensor.dtype} does not match checkpoint {meta['dtype']}")
    if tuple(tensor.shape) != tuple(meta["shape"]):
        raise ExportError(f"{name}: replacement shape {tuple(tensor.shape)} does not match checkpoint {tuple(meta['shape'])}")
    start, end = meta["data_offsets"]
    tensor = tensor.detach().contiguous().cpu()
    if tensor.numel() * tensor.element_size() != end - start:
        raise ExportError(f"{name}: replacement is {tensor.numel() * tensor.element_size()} bytes, checkpoint entry is {end - start}")
    return tensor


# Publish an unchanged file: a hard link to the source's real file (the HF cache
# stores snapshot entries as symlinks into its blob store, so the link targets
# the blob), or a byte copy when the destination is on another filesystem.
# Returns "linked" or "copied" for the export's summary.
def _link_or_copy(src: str, dst: str) -> str:
    real = os.path.realpath(src)
    try:
        os.link(real, dst)
        return "linked"
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.copyfile(real, dst)
        return "copied"


# Copy `length` bytes between open files in bounded chunks.
def _copy_range(src, dst, length: int) -> None:
    remaining = length
    while remaining:
        chunk = src.read(min(COPY_BUFFER, remaining))
        if not chunk:
            raise ExportError("source shard ended early while copying")
        dst.write(chunk)
        remaining -= len(chunk)


# Write one shard: header verbatim, then each tensor payload either copied from
# the source or taken from its replacement. Returns the names replaced here.
def write_shard(src_path: str, dst_path: str, replacements: dict[str, torch.Tensor]) -> list[str]:
    header_len, header = read_shard_header(src_path)
    ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    replaced: list[str] = []
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        _copy_range(src, dst, 8 + header_len)
        expected = 0
        for name, meta in ordered:
            start, end = meta["data_offsets"]
            if start != expected:
                raise ExportError(f"{os.path.basename(src_path)}: tensor {name} starts at {start}, expected {expected}; layout has gaps")
            if name in replacements:
                tensor = _check_replacement(name, meta, replacements[name])
                dst.write(tensor.view(torch.uint8).numpy().tobytes())
                src.seek(8 + header_len + end)
                replaced.append(name)
            else:
                _copy_range(src, dst, end - start)
            expected = end
        if src.read(1):
            raise ExportError(f"{os.path.basename(src_path)}: trailing bytes after the last tensor")
    if os.path.getsize(dst_path) != os.path.getsize(src_path):
        raise ExportError(f"{os.path.basename(dst_path)}: size differs from source after writing")
    return replaced


def export(snapshot: str, out_dir: str, replacements: dict[str, torch.Tensor], force: bool = False,
           log=sys.stderr) -> dict[str, list[str]]:
    """Export `snapshot` to `out_dir` with `replacements` applied.

    Returns {shard: [replaced tensor names]}. Every replacement must name a
    tensor in the index; unknown names are an error before anything is written.
    """
    snapshot = os.path.abspath(snapshot)
    out_dir = os.path.abspath(out_dir)
    if os.path.commonpath([snapshot, out_dir]) in (snapshot, out_dir):
        raise ExportError(f"output {out_dir} overlaps the source snapshot {snapshot}")
    if os.path.lexists(out_dir):
        if not force:
            raise ExportError(f"output {out_dir} exists; pass --force to replace it")
    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    unknown = sorted(set(replacements) - set(weight_map))
    if unknown:
        raise ExportError(f"{len(unknown)} replacement names are not in the index, e.g. {unknown[:3]}")
    by_shard: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in replacements.items():
        by_shard.setdefault(weight_map[name], {})[name] = tensor

    staging = out_dir + ".staging"
    if os.path.lexists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    start = time.monotonic()
    result: dict[str, list[str]] = {}
    unchanged = {"linked": 0, "copied": 0}

    def publish_unchanged(src: str, dst: str) -> None:
        unchanged[_link_or_copy(src, dst)] += 1

    try:
        entries = sorted(os.listdir(snapshot))
        shards = [e for e in entries if e.endswith(".safetensors")]
        for i, shard in enumerate(shards, 1):
            reps = by_shard.get(shard, {})
            if not reps:
                # No payload changes: the shard is the source's, byte for byte.
                publish_unchanged(os.path.join(snapshot, shard), os.path.join(staging, shard))
                result[shard] = []
                continue
            print(f"shard {i}/{len(shards)} {shard} ({len(reps)} replaced)", file=log, flush=True)
            result[shard] = write_shard(os.path.join(snapshot, shard), os.path.join(staging, shard), reps)
        for entry in entries:
            if entry.endswith(".safetensors"):
                continue
            src = os.path.join(snapshot, entry)
            dst = os.path.join(staging, entry)
            if os.path.isdir(src):
                shutil.copytree(src, dst, symlinks=False)
            else:
                shutil.copyfile(src, dst)
        written = sum(len(v) for v in result.values())
        if written != len(replacements):
            raise ExportError(f"wrote {written} replacements but {len(replacements)} were requested")
        if os.path.lexists(out_dir):
            if os.path.isdir(out_dir) and not os.path.islink(out_dir):
                shutil.rmtree(out_dir)
            else:
                os.remove(out_dir)
        os.rename(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"exported to {out_dir} in {time.monotonic() - start:.0f}s: {sum(1 for v in result.values() if v)} shards written, "
          f"{unchanged['linked']} unchanged shards hard-linked, {unchanged['copied']} copied", file=log, flush=True)
    return result
