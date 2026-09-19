"""lora_B check against a saved adapter.

The fixture is a plain torch module with two Linear layers wrapped by peft,
so the test needs no model family, no download, and no GPU. What is under
test is the comparison, not peft.
"""

import pathlib
import tempfile
import unittest

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from trlx import TrlxError, adapter_check


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


# Wraps Tiny with LoRA on the given layers and sets every lora_B to `fill`.
# peft initialises lora_B to zero, so a nonzero fill stands in for training.
def _adapter(targets, fill):
    model = get_peft_model(Tiny(), LoraConfig(r=2, target_modules=targets))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(fill)
    return model


class AdapterCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name) / "adapter"
        _adapter(["a", "b"], 0.5).save_pretrained(str(self.dir))

    def tearDown(self):
        self._tmp.cleanup()

    def test_file_stats(self):
        self.assertEqual(adapter_check.file_stats(str(self.dir)), (2, 0.5))

    def test_loaded_adapter_passes(self):
        loaded = PeftModel.from_pretrained(Tiny(), str(self.dir))
        result = adapter_check.check(str(self.dir), loaded)
        self.assertTrue(result.ok, result.message())
        self.assertEqual((result.model_count, result.model_max), (2, 0.5))
        self.assertIn("adapter loaded", result.message())

    def test_zero_model_adapters_fail(self):
        # Fresh peft wrapping with the file's structure but untrained weights:
        # what a key mismatch on load produces.
        result = adapter_check.check(str(self.dir), _adapter(["a", "b"], 0.0))
        self.assertFalse(result.ok)
        self.assertIn("all zero", result.message())

    def test_count_mismatch_fails(self):
        result = adapter_check.check(str(self.dir), _adapter(["a"], 0.5))
        self.assertFalse(result.ok)
        self.assertIn("tensor count differs", result.message())

    def test_magnitude_mismatch_fails(self):
        result = adapter_check.check(str(self.dir), _adapter(["a", "b"], 0.25))
        self.assertFalse(result.ok)
        self.assertIn("max magnitude differs", result.message())

    def test_bf16_rounding_is_tolerated(self):
        # A file in fp32 loaded as bf16 rounds the magnitude slightly.
        result = adapter_check.AdapterCheck(2, 0.123456, 2, 0.1235)
        self.assertTrue(result.ok)

    def test_missing_file(self):
        with self.assertRaises(TrlxError) as ctx:
            adapter_check.file_stats(self._tmp.name)
        self.assertIn("not an adapter directory", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
