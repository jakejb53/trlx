"""Initialization queries metadata without importing real CUDA or loading models."""

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trlx import TrlxError
from trlx.hardware import Gpu, Hardware, inspect


class FakeCuda:
    # Model different device capabilities and a pre-existing selected device.
    def __init__(self, devices=(), failure=None):
        self.devices = devices
        self.failure = failure
        self.current = 7
        self.visited = []

    # Query failures must propagate instead of looking like an empty machine.
    def device_count(self):
        if self.failure == "count":
            raise RuntimeError("driver unavailable")
        return len(self.devices)

    # Match CUDA's restoring context so tests catch device-selection leaks.
    @contextmanager
    def device(self, index):
        if self.failure == "select":
            raise RuntimeError("selection unavailable")
        previous = self.current
        self.current = index
        self.visited.append(index)
        try:
            yield
        finally:
            self.current = previous

    # Metadata access must address the device currently under inspection.
    def get_device_name(self, index):
        assert index == self.current
        if self.failure == "name":
            raise RuntimeError("name unavailable")
        return self.devices[index].name

    # Preserve CUDA's free-before-total return order to catch swapped fields.
    def mem_get_info(self, index):
        assert index == self.current
        if self.failure == "memory":
            raise RuntimeError("memory unavailable")
        gpu = self.devices[index]
        return gpu.free_bytes, gpu.total_bytes

    # Native support must be queried per device, explicitly excluding emulation.
    def is_bf16_supported(self, *, including_emulation):
        assert including_emulation is False
        if self.failure == "bf16":
            raise RuntimeError("precision unavailable")
        return self.devices[self.current].bf16


class InspectHardware(unittest.TestCase):
    # Inject only the CUDA metadata surface; no real torch import or GPU use.
    def inspect_fake(self, cuda, cpus=12):
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            with patch("trlx.hardware.os.cpu_count", return_value=cpus):
                return inspect()

    # A CPU-only operator can initialize without fabricated GPU capabilities.
    def test_no_gpu(self):
        cuda = FakeCuda()
        self.assertEqual(self.inspect_fake(cuda), Hardware(12, ()))
        self.assertEqual(cuda.visited, [])

    # Unknown CPU count has the explicitly documented one-worker assumption.
    def test_unknown_cpu_count(self):
        self.assertEqual(self.inspect_fake(FakeCuda(), cpus=None), Hardware(1, ()))

    # Keep heterogeneous memory and native precision rather than probing GPU 0 only.
    def test_mixed_gpus_restore_current_device(self):
        devices = (
            Gpu(0, "large", 24 * 2**30, 18 * 2**30, True),
            Gpu(1, "small", 8 * 2**30, 3 * 2**30, False),
        )
        cuda = FakeCuda(devices)
        result = self.inspect_fake(cuda)
        self.assertEqual(result, Hardware(12, devices))
        self.assertEqual(cuda.visited, [0, 1])
        self.assertEqual(cuda.current, 7)
        with self.assertRaises(FrozenInstanceError):
            result.cpu_count = 3
        with self.assertRaises(FrozenInstanceError):
            result.gpus[0].bf16 = False

    # Enumeration errors are actionable errors, never successful CPU-only results.
    def test_enumeration_failure(self):
        with self.assertRaisesRegex(TrlxError, "enumerate visible CUDA devices.*driver unavailable"):
            self.inspect_fake(FakeCuda(failure="count"))

    # Each failure names its device and operation while still restoring context.
    def test_device_query_failures(self):
        actions = {
            "select": "select device",
            "name": "read device name",
            "memory": "read free and total memory",
            "bf16": "query native BF16 support",
        }
        for failure, action in actions.items():
            with self.subTest(failure=failure):
                cuda = FakeCuda((Gpu(0, "gpu", 10, 5, True),), failure=failure)
                with self.assertRaisesRegex(TrlxError, f"CUDA device 0: {action} failed"):
                    self.inspect_fake(cuda)
                self.assertEqual(cuda.current, 7)

    # CPU query failures differ from Python's supported unknown-count result.
    def test_cpu_query_failure(self):
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=FakeCuda())}):
            with patch("trlx.hardware.os.cpu_count", side_effect=OSError("CPU unavailable")):
                with self.assertRaisesRegex(TrlxError, "CPU count.*CPU unavailable"):
                    inspect()

    # Dynamic-library import failures are installation errors, not Python implementation bugs.
    def test_torch_shared_library_failure(self):
        with patch("builtins.__import__", side_effect=OSError("missing shared library")):
            with self.assertRaisesRegex(TrlxError, "importing PyTorch failed.*shared-library dependencies"):
                inspect()


if __name__ == "__main__":
    unittest.main()
