"""CPU tests for the proxy_step value-change metric: the FP4 +0/-0 pair is not a
value change although its bits differ, a real code change is counted once, a
doubled group scale counts every element whose value moved, and FP8 follows the
same value rule."""

import unittest

import torch

from deepseek.qat import fakequant as FQ
from deepseek.qat import quant as Q
from deepseek.qat.proxy_step import quantized_state, value_changes


def _fp4_state(w: torch.Tensor):
    # One expert, "fp4_down" kind: quantized_state expects [E, inter, dim] and
    # quantizes each expert's transpose, so hand it the transpose of the HF matrix.
    return quantized_state("x.mlp.experts.down_projs", w.T.unsqueeze(0), "fp4_down")


class FP4ValueChanges(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.w = torch.randn(64, 128)
        # Put a run of exact zeros in the first group so sign flips of zero are possible.
        self.w[0, :8] = 0.0

    def test_signed_zero_is_not_a_value_change(self):
        before = _fp4_state(self.w)
        w2 = self.w.clone()
        w2[0, :8] = -0.0
        after = _fp4_state(w2)
        self.assertGreater(int((before[0] != after[0]).sum()), 0, "the packed bits must differ for the test to mean anything")
        self.assertEqual(value_changes(before, after, "fp4_down"), 0)

    def test_one_element_moved_one_cell_counts_once(self):
        # Baseline on the project quantizer's own grid, so it is a fixed point.
        packed, scales = FQ.quantize_fp4(self.w)
        deq = Q.dequantize_fp4_rows(packed, scales)
        before = _fp4_state(deq)
        # Move a mid-sized element to a different grid value without changing
        # the group's absmax (the group's largest-magnitude element stays put).
        row, col = 5, 40
        group = deq[row, 32:64]
        big = int(group.abs().argmax()) + 32
        self.assertNotEqual(big, col)
        w2 = deq.clone()
        step = float(scales[row, 1].float()) * 0.5  # smallest nonzero FP4 code at this scale
        w2[row, col] = w2[row, col] + step if w2[row, col] >= 0 else w2[row, col] - step
        w2[row, col] = torch.clamp(w2[row, col], -float(group.abs().max()), float(group.abs().max()))
        after = _fp4_state(w2)
        self.assertEqual(value_changes(before, after, "fp4_down"), 1)

    def test_scale_doubling_counts_moved_values(self):
        packed, scales = FQ.quantize_fp4(self.w)
        deq = Q.dequantize_fp4_rows(packed, scales)
        before = _fp4_state(deq)
        w2 = deq.clone()
        # Grow the group's largest element enough to double the scale.
        row = 7
        group = w2[row, 0:32]
        big = int(group.abs().argmax())
        w2[row, big] = w2[row, big] * 2.5
        after = _fp4_state(w2)
        c0, s0 = before
        c1, s1 = after
        self.assertEqual(int((s0 != s1).sum()), 1)
        n = value_changes(before, after, "fp4_down")
        self.assertGreater(n, 1)
        self.assertLessEqual(n, 32)


class FP8ValueChanges(unittest.TestCase):
    def test_fp8_same_values_different_scale_code_is_no_change(self):
        torch.manual_seed(1)
        w = torch.randn(256, 256)
        before = quantized_state("x.self_attn.wq_a.weight", w, "fp8")
        codes, scales = Q.quantize_fp8_blocks(w)
        after = quantized_state("x.self_attn.wq_a.weight", Q.dequantize_fp8_blocks(codes, scales), "fp8")
        self.assertEqual(value_changes(before, after, "fp8"), 0)

    def test_fp8_one_element_change(self):
        torch.manual_seed(2)
        w = torch.randn(256, 256)
        codes, scales = Q.quantize_fp8_blocks(w)
        deq = Q.dequantize_fp8_blocks(codes, scales)
        before = quantized_state("x.self_attn.wq_a.weight", deq, "fp8")
        w2 = deq.clone()
        w2[3, 3] = 0.0
        after = quantized_state("x.self_attn.wq_a.weight", w2, "fp8")
        self.assertEqual(value_changes(before, after, "fp8"), 1)


if __name__ == "__main__":
    unittest.main()
