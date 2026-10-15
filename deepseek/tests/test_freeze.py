"""CPU tests for freeze_gate_bias on a stub of NeMo's DeepSeek-V4 module layout:
the factor is zeroed and reported, hash gates without a bias are skipped, a
scored gate missing its loaded bias is an error, and a changed layout is an
error rather than a silent no-op."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from deepseek.qat.freeze import freeze_gate_bias


# Scored gate as NeMo builds it: a positive factor and a loaded FP32 bias buffer.
def _scored_gate(factor: float = 1e-3, with_bias: bool = True) -> nn.Module:
    gate = nn.Module()
    gate.bias_update_factor = factor
    if with_bias:
        gate.register_buffer("e_score_correction_bias", torch.zeros(8, dtype=torch.float32))
    else:
        gate.e_score_correction_bias = None
    return gate


# Hash-routing gate: no factor attribute and no correction bias.
def _hash_gate() -> nn.Module:
    return nn.Module()


def _model(gates, factor: float = 1e-3) -> nn.Module:
    inner = nn.Module()
    inner.moe_config = SimpleNamespace(train_gate=True, gate_bias_update_factor=factor)
    inner.layers = nn.ModuleDict({str(i): nn.Module() for i in range(len(gates))})
    for i, gate in enumerate(gates):
        inner.layers[str(i)].mlp = nn.Module()
        inner.layers[str(i)].mlp.gate = gate
    model = nn.Module()
    model.model = inner
    return model


class FreezeGateBias(unittest.TestCase):
    def test_zeroes_factor_and_counts_scored_gates(self):
        model = _model([_hash_gate(), _scored_gate(), _scored_gate()])
        previous, with_bias = freeze_gate_bias(model)
        self.assertEqual(previous, 1e-3)
        self.assertEqual(with_bias, 2)
        self.assertEqual(model.model.moe_config.gate_bias_update_factor, 0.0)
        # The gates' own copies are left alone; only the model-level guard changes.
        self.assertEqual(model.model.layers["1"].mlp.gate.bias_update_factor, 1e-3)

    def test_already_zero_factor_is_reported(self):
        previous, _ = freeze_gate_bias(_model([_scored_gate()], factor=0.0))
        self.assertEqual(previous, 0.0)

    def test_scored_gate_without_bias_is_an_error(self):
        model = _model([_scored_gate(), _scored_gate(with_bias=False)])
        with self.assertRaisesRegex(ValueError, "layer 1"):
            freeze_gate_bias(model)

    def test_missing_moe_config_is_an_error(self):
        model = nn.Module()
        model.model = nn.Module()
        with self.assertRaises(AttributeError):
            freeze_gate_bias(model)

    def test_missing_layers_is_an_error(self):
        model = nn.Module()
        model.model = nn.Module()
        model.model.moe_config = SimpleNamespace(gate_bias_update_factor=1e-3)
        with self.assertRaises(AttributeError):
            freeze_gate_bias(model)


if __name__ == "__main__":
    unittest.main()
