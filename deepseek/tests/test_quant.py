"""Synthetic tests for the exact DeepSeek-V4 weight quantizers.

These cover the rules a wrong implementation would get silently wrong: the
power-of-two scale at exact powers of two, saturation, zero blocks, E2M1 tie
rounding, nibble packing order, FP8 cast parity, and re-quantization fixed
points. The real-checkpoint check lives in ``deepseek.qat.roundtrip``.
"""

import unittest

import torch

from deepseek.qat import quant as Q


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


class ScaleRule(unittest.TestCase):
    def test_exact_powers_of_two_keep_their_exponent(self):
        x = torch.tensor([1.0, 2.0, 0.5, 2.0 ** -100, 2.0 ** 60])
        self.assertEqual(Q.pow2_ceil(x).tolist(), x.tolist())

    def test_nonzero_mantissa_rounds_exponent_up(self):
        x = torch.tensor([1.0001, 3.0, 0.75, 1.5])
        self.assertEqual(Q.pow2_ceil(x).tolist(), [2.0, 4.0, 1.0, 2.0])

    def test_zero_and_denormal_match_reference_bit_rule(self):
        # Reference: exponent field 0 gives 2**-127, plus one when the mantissa is nonzero.
        x = torch.tensor([0.0, 2.0 ** -140])
        self.assertEqual(Q.pow2_ceil(x).tolist(), [2.0 ** -127, 2.0 ** -126])

    def test_amax_floor_applies_before_rule(self):
        amax = torch.tensor([0.0, 1e-6, 100.0])
        s, s_e8m0 = Q.fp8_block_scale(amax, amax_floor=1e-4)
        self.assertEqual(s[0].item(), s[1].item())
        self.assertEqual(s[0].item(), Q.pow2_ceil(torch.tensor([1e-4 / Q.FP8_MAX])).item())
        self.assertEqual(s[2].item(), 0.25)
        self.assertEqual(s_e8m0.float().tolist(), s.tolist())

    def test_fp4_floor_rule(self):
        # 2**(floor(log2(amax)) - 2): amax in [4, 8) gives scale 1, [2, 4) gives 0.5.
        amax = torch.tensor([4.0, 6.0, 7.99, 8.0, 3.0, 2.0, 0.0, 2.0 ** -140])
        s, s_e8m0 = Q.fp4_group_scale(amax, amax_floor=0.0)
        self.assertEqual(s.tolist(), [1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 2.0 ** -127, 2.0 ** -127])
        self.assertEqual(s_e8m0.float().tolist(), s.tolist())
        # The ceiling rule would give amax 6.1 a scale of 2; floor keeps 1 and saturates.
        s2, _ = Q.fp4_group_scale(torch.tensor([6.1]), amax_floor=0.0)
        self.assertEqual(s2.item(), 1.0)

    def test_scales_are_exact_in_e8m0_over_the_whole_range(self):
        # 2**-127 is a float32 denormal, so the reference bit rule maps it to -126.
        x = torch.tensor([2.0 ** -127, 2.0 ** -126, 1.0, 2.0 ** 127])
        s = Q.e8m0_from_exponent(Q.pow2_ceil_exponent(x))
        self.assertEqual(s.view(torch.uint8).tolist(), [1, 1, 127, 254])
        self.assertEqual(s.float().tolist(), [2.0 ** -126, 2.0 ** -126, 1.0, 2.0 ** 127])
        # The smallest E8M0 value is still constructible and converts back exactly.
        smallest = Q.e8m0_from_exponent(torch.tensor([-127], dtype=torch.int32))
        self.assertEqual(smallest.view(torch.uint8).item(), 0)
        self.assertEqual(smallest.float().item(), 2.0 ** -127)
        with self.assertRaises(ValueError):
            Q.e8m0_from_exponent(torch.tensor([128], dtype=torch.int32))


class E2M1Rounding(unittest.TestCase):
    def test_ties_round_to_even_mantissa(self):
        a = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        self.assertEqual(Q.round_to_e2m1_magnitude(a).tolist(), [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])

    def test_nearest_off_tie(self):
        a = torch.tensor([0.2, 0.3, 0.9, 1.1, 1.4, 2.4, 2.6, 3.4, 3.6, 4.9, 5.1, 6.0])
        self.assertEqual(
            Q.round_to_e2m1_magnitude(a).tolist(),
            [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0],
        )


class FP4Packing(unittest.TestCase):
    def test_codes_and_nibble_order(self):
        # One group of 32 with absmax 6 so the scale is exactly 1.
        row = torch.zeros(1, 32)
        row[0, :8] = Q.FP4_MAGNITUDES
        row[0, 8:16] = -Q.FP4_MAGNITUDES
        packed, scales = Q.quantize_fp4_rows(row)
        self.assertEqual(scales.float().item(), 1.0)
        codes = Q.unpack_fp4_codes(packed)[0].tolist()
        # -0.0 keeps its sign bit (code 8), as the hardware cast and the source checkpoint do.
        self.assertEqual(codes[:16], list(range(16)))
        # Element 2k is the low nibble: codes 0 and 1 pack to 0x10, codes 2 and 3 to 0x32.
        self.assertEqual(packed[0, :2].tolist(), [0x10, 0x32])

    def test_int8_input_is_accepted_by_dequantize(self):
        row = torch.randn(4, 64)
        packed, scales = Q.quantize_fp4_rows(row)
        as_int8 = packed.view(torch.int8)
        self.assertTrue(torch.equal(Q.dequantize_fp4_rows(as_int8, scales), Q.dequantize_fp4_rows(packed, scales)))

    def test_largest_element_saturates_under_floor_rule(self):
        # absmax 7 is in [4, 8) so the scale is 1; 7 clamps to 6 (code 7).
        row = torch.zeros(1, 32)
        row[0, 0] = 7.0
        row[0, 1] = -7.0
        packed, scales = Q.quantize_fp4_rows(row)
        self.assertEqual(scales.float().item(), 1.0)
        self.assertEqual(Q.dequantize_fp4_rows(packed, scales)[0, :2].tolist(), [6.0, -6.0])
        self.assertEqual(Q.unpack_fp4_codes(packed)[0, :2].tolist(), [7, 15])
        # A largest code of 3 can only arise with amax < 4*scale, which the floor rule never produces.
        w = torch.randn(64, 256) * 3
        p, s = Q.quantize_fp4_rows(w)
        cmax = (Q.unpack_fp4_codes(p) & 7).reshape(64, 8, 32).amax(-1)
        self.assertTrue(((cmax == 6) | (cmax == 7)).all())

    def test_zero_group_uses_minimum_scale_and_zero_codes(self):
        row = torch.zeros(2, 64)
        packed, scales = Q.quantize_fp4_rows(row)
        self.assertTrue((packed == 0).all())
        self.assertTrue((scales.float() == 2.0 ** -127).all())


class FP8Blocks(unittest.TestCase):
    def test_cast_matches_torch_e4m3_cast(self):
        w = torch.randn(256, 384) * 3
        codes, scales = Q.quantize_fp8_blocks(w)
        blocks = w.reshape(2, 128, 3, 128) / scales.float()[:, None, :, None]
        expected = blocks.to(torch.float8_e4m3fn).reshape(256, 384)
        self.assertTrue(torch.equal(_bits(codes), _bits(expected)))

    def test_largest_element_never_saturates(self):
        w = torch.zeros(128, 128)
        w[3, 7] = 449.0
        codes, scales = Q.quantize_fp8_blocks(w)
        self.assertEqual(scales.float().item(), 2.0)
        self.assertEqual(Q.dequantize_fp8_blocks(codes, scales)[3, 7].item(), 448.0)

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            Q.quantize_fp8_blocks(torch.zeros(100, 128))
        with self.assertRaises(ValueError):
            Q.quantize_fp4_rows(torch.zeros(4, 40))
        with self.assertRaises(ValueError):
            Q.dequantize_fp8_blocks(torch.zeros(128, 128, dtype=torch.float8_e4m3fn), torch.zeros(2, 1, dtype=torch.float8_e8m0fnu))

    def test_non_finite_rejected(self):
        w = torch.zeros(128, 128)
        w[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            Q.quantize_fp8_blocks(w)
        w4 = torch.zeros(1, 32)
        w4[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            Q.quantize_fp4_rows(w4)


class Requantization(unittest.TestCase):
    """Dequantize-then-quantize must preserve values; the second pass is a fixed point."""

    def test_fp8_values_preserved_and_bits_stable(self):
        w = torch.randn(256, 256)
        c1, s1 = Q.quantize_fp8_blocks(w)
        d1 = Q.dequantize_fp8_blocks(c1, s1)
        c2, s2 = Q.quantize_fp8_blocks(d1)
        self.assertTrue(torch.equal(Q.dequantize_fp8_blocks(c2, s2), d1))
        c3, s3 = Q.quantize_fp8_blocks(Q.dequantize_fp8_blocks(c2, s2))
        self.assertTrue(torch.equal(_bits(c2), _bits(c3)) and torch.equal(_bits(s2), _bits(s3)))

    def test_fp4_values_preserved_and_bits_stable(self):
        w = torch.randn(64, 256)
        p1, s1 = Q.quantize_fp4_rows(w)
        d1 = Q.dequantize_fp4_rows(p1, s1)
        p2, s2 = Q.quantize_fp4_rows(d1)
        self.assertTrue(torch.equal(Q.dequantize_fp4_rows(p2, s2), d1))
        p3, s3 = Q.quantize_fp4_rows(Q.dequantize_fp4_rows(p2, s2))
        self.assertTrue(torch.equal(p2, p3) and torch.equal(_bits(s2), _bits(s3)))

    def test_fp4_scale_is_a_fixed_point_of_requantization(self):
        # absmax 3 is in [2, 4) so the scale is 0.5 and 3 encodes as code 6;
        # re-quantizing the dequantized values must reproduce that choice.
        row = torch.zeros(1, 32)
        row[0, :4] = torch.tensor([3.0, 1.5, 0.5, -2.0])
        p1, s1 = Q.quantize_fp4_rows(row)
        d1 = Q.dequantize_fp4_rows(p1, s1)
        p2, s2 = Q.quantize_fp4_rows(d1)
        self.assertEqual(s1.float().item(), 0.5)
        self.assertEqual(s2.float().item(), 0.5)
        self.assertTrue(torch.equal(Q.dequantize_fp4_rows(p2, s2), d1))

    def test_bf16_input_quantizes_like_float32(self):
        w = torch.randn(128, 128).bfloat16()
        c_bf, s_bf = Q.quantize_fp8_blocks(w)
        c_f, s_f = Q.quantize_fp8_blocks(w.float())
        self.assertTrue(torch.equal(_bits(c_bf), _bits(c_f)) and torch.equal(_bits(s_bf), _bits(s_f)))


if __name__ == "__main__":
    unittest.main()
