"""Environment metadata for initialization; no model or training allocation."""

from dataclasses import dataclass
import os

from trlx import TrlxError


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    bf16: bool


@dataclass(frozen=True)
class Hardware:
    # Logical processors reported by the OS, not physical CPU packages.
    cpu_count: int
    gpus: tuple[Gpu, ...]


# Query only visible devices, retaining per-device capabilities for mixed systems.
# CUDA memory queries can initialize driver contexts, but never load weights.
def inspect() -> Hardware:
    try:
        import torch
    except ImportError as exc:
        raise TrlxError(f"cannot inspect hardware: importing PyTorch failed: {exc}") from exc

    try:
        cpu_count = os.cpu_count()
    except OSError as exc:
        raise TrlxError(f"cannot inspect CPU count: {exc}") from exc
    # Python explicitly permits an unknown CPU count; assume one worker then.
    if cpu_count is None:
        cpu_count = 1

    try:
        count = torch.cuda.device_count()
    except (RuntimeError, OSError, AssertionError) as exc:
        raise TrlxError(f"cannot enumerate visible CUDA devices: {exc}") from exc
    return Hardware(cpu_count, tuple(_inspect_gpu(torch.cuda, index) for index in range(count)))


# BF16 probing uses the current CUDA device. The context restores it even when
# a query fails, and native support excludes PyTorch's emulation capability.
def _inspect_gpu(cuda, index: int) -> Gpu:
    action = "select device"
    try:
        with cuda.device(index):
            action = "read device name"
            name = cuda.get_device_name(index)
            action = "read free and total memory"
            free_bytes, total_bytes = cuda.mem_get_info(index)
            action = "query native BF16 support"
            bf16 = cuda.is_bf16_supported(including_emulation=False)
            action = "restore previous device"
        return Gpu(index, name, total_bytes, free_bytes, bf16)
    except (RuntimeError, OSError, AssertionError) as exc:
        raise TrlxError(f"cannot inspect CUDA device {index}: {action} failed: {exc}") from exc
