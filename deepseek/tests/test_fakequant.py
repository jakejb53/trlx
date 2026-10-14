"""CPU tests for the fake-quant hooks: straight-through gradient, idempotence on
grid values, expert orientation against HF layout, and forward swapping that
leaves parameters untouched."""

import unittest

import torch
from torch import nn

from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q


def _on_fp4_grid(out: int, inp: int) -> torch.Tensor:
    p, s = Q.quantize_fp4_rows(torch.randn(out, inp))
    return Q.dequantize_fp4_rows(p, s)


class StraightThrough(unittest.TestCase):
    def test_forward_quantizes_and_backward_is_identity(self):
        w = torch.randn(128, 128, requires_grad=True)
        y = FQ.fake_quant(w, "fp8")
        self.assertTrue(torch.equal(y, FQ._fp8_dequant_quant(w.detach())))
        self.assertFalse(torch.equal(y, w.detach()))
        g = torch.randn_like(y)
        y.backward(g)
        self.assertTrue(torch.equal(w.grad, g))

    def test_idempotent_on_grid_values(self):
        c, s = Q.quantize_fp8_blocks(torch.randn(256, 128))
        w = Q.dequantize_fp8_blocks(c, s).bfloat16()
        self.assertTrue(torch.equal(FQ.fake_quant(w, "fp8"), w))
        w4 = _on_fp4_grid(8, 64).bfloat16()
        self.assertTrue(torch.equal(FQ._fp4_dequant_quant(w4), w4))


class ExpertOrientation(unittest.TestCase):
    """NeMo gate_and_up_projs[e] = cat(w1.T, w3.T); down_projs[e] = w2.T."""

    def setUp(self):
        self.e, self.dim, self.inter = 3, 64, 32
        self.w1 = [torch.randn(self.inter, self.dim) for _ in range(self.e)]
        self.w3 = [torch.randn(self.inter, self.dim) for _ in range(self.e)]
        self.w2 = [torch.randn(self.dim, self.inter) for _ in range(self.e)]
        self.gate_up = torch.stack([torch.cat([a.T, b.T], dim=-1) for a, b in zip(self.w1, self.w3)])
        self.down = torch.stack([c.T for c in self.w2])

    def test_gate_up_matches_hf_quantization(self):
        out = FQ.fake_quant_gate_up(self.gate_up)
        for e in range(self.e):
            self.assertTrue(torch.equal(out[e, :, : self.inter].T, FQ._fp4_dequant_quant(self.w1[e])))
            self.assertTrue(torch.equal(out[e, :, self.inter :].T, FQ._fp4_dequant_quant(self.w3[e])))

    def test_down_matches_hf_quantization(self):
        out = FQ.fake_quant_down(self.down)
        for e in range(self.e):
            self.assertTrue(torch.equal(out[e].T, FQ._fp4_dequant_quant(self.w2[e])))

    def test_grid_values_are_fixed_points(self):
        gate_up = torch.stack([torch.cat([_on_fp4_grid(self.inter, self.dim).T, _on_fp4_grid(self.inter, self.dim).T], -1)
                               for _ in range(self.e)])
        self.assertTrue(torch.equal(FQ.fake_quant_gate_up(gate_up), gate_up))
        down = torch.stack([_on_fp4_grid(self.dim, self.inter).T for _ in range(self.e)])
        self.assertTrue(torch.equal(FQ.fake_quant_down(down), down))

    def test_chunking_does_not_change_result(self):
        old = FQ.EXPERT_CHUNK
        try:
            FQ.EXPERT_CHUNK = 2
            a = FQ.fake_quant_gate_up(self.gate_up)
            FQ.EXPERT_CHUNK = 16
            b = FQ.fake_quant_gate_up(self.gate_up)
        finally:
            FQ.EXPERT_CHUNK = old
        self.assertTrue(torch.equal(a, b))


class _Experts(nn.Module):
    def __init__(self, e, dim, inter):
        super().__init__()
        self.gate_and_up_projs = nn.Parameter(torch.randn(e, dim, 2 * inter))
        self.down_projs = nn.Parameter(torch.randn(e, inter, dim))

    def forward(self, x):
        return torch.einsum("ed,edf->ef", x, self.gate_and_up_projs).sum() + self.down_projs.sum()


class Install(unittest.TestCase):
    def test_forward_uses_fake_quant_and_parameters_are_untouched(self):
        lin = nn.Linear(128, 128, bias=False)
        with torch.no_grad():
            lin.weight.mul_(3)
        before = lin.weight.detach().clone()
        x = torch.randn(4, 128)
        plain = lin(x)
        state = FQ.FakeQuantState([FQ.Target("lin", lin, {"weight": "fp8"})])
        original = lin.forward
        state.originals["lin"] = original

        def wrapped(*args, **kwargs):
            swapped = {"weight": FQ.fake_quant(lin.weight, "fp8")}
            with FQ._reparametrize_module(lin, swapped):
                return original(*args, **kwargs)

        lin.forward = wrapped
        hooked = lin(x)
        self.assertTrue(torch.equal(hooked, x @ FQ._fp8_dequant_quant(before).T))
        self.assertFalse(torch.equal(hooked, plain))
        self.assertTrue(torch.equal(lin.weight.detach(), before))
        self.assertIsInstance(lin.weight, nn.Parameter)
        # Gradient reaches the parameter through the straight-through path.
        hooked.sum().backward()
        self.assertIsNotNone(lin.weight.grad)
        FQ.uninstall(state)
        self.assertTrue(torch.equal(lin(x), plain))

    def test_expert_module_forward_swaps_both_parameters(self):
        m = _Experts(2, 64, 32)
        x = torch.randn(2, 64)
        state = FQ.FakeQuantState([FQ.Target("ex", m, {"gate_and_up_projs": "fp4_gate_up", "down_projs": "fp4_down"})])
        orig = m.forward
        state.originals["ex"] = orig

        def wrapped(*args, **kwargs):
            swapped = {a: FQ.fake_quant(getattr(m, a), k) for a, k in state.targets[0].params.items()}
            with FQ._reparametrize_module(m, swapped):
                return orig(*args, **kwargs)

        m.forward = wrapped
        expected = torch.einsum("ed,edf->ef", x, FQ.fake_quant_gate_up(m.gate_and_up_projs.detach())).sum() + \
            FQ.fake_quant_down(m.down_projs.detach()).sum()
        self.assertTrue(torch.equal(m(x), expected))
        FQ.uninstall(state)
        self.assertTrue(torch.equal(m(x), orig(x)))


if __name__ == "__main__":
    unittest.main()
