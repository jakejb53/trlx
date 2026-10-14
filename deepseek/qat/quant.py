"""Exact DeepSeek-V4-Flash weight quantizers.

The released Flash-0731 checkpoint stores dense projections as FP8 E4M3 with one
UE8M0 scale per 128x128 block, and routed experts as E2M1 FP4 packed two per
byte with one UE8M0 scale per 32 elements along the input dimension. The two
formats use different scale rules, both established against the released
tensors by ``deepseek.qat.roundtrip``:

- FP8: scale = 2 ** ceil(log2(absmax / 448)), the rule of the checkpoint's own
  ``inference/kernel.py``; the largest element never saturates.
- FP4: scale = 2 ** (floor(log2(absmax)) - 2), the OCP MX rule; elements above
  6 saturate. The released expert groups only ever have a largest code of 4 or
  6, which the ceiling rule cannot produce, so this is the rule they went
  through.
- Exponents come from IEEE-754 bit arithmetic as the reference
  ``fast_log2_ceil`` does, never from a floating-point log2.
- element = round_to_nearest_even(clamp(value / scale, -format_max, format_max));
- FP4 code = sign<<3 | exponent<<1 | mantissa, element 2k in the low nibble.

Everything is pure PyTorch and runs on CPU or GPU. Inputs are computed in
float32 regardless of their storage dtype, as the reference kernels do.
"""

import torch

FP8_MAX = 448.0
FP8_BLOCK = 128
FP4_MAX = 6.0
FP4_GROUP = 32

# E2M1 magnitudes by code index 0..7; index 8..15 is the same magnitude negated.
FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_VALUES = torch.cat([FP4_MAGNITUDES, -FP4_MAGNITUDES])


# Reject NaN/Inf before quantizing: clamping would silently turn a broken
# master weight into a valid-looking code.
def _check_finite(w: torch.Tensor) -> None:
    if not torch.isfinite(w).all():
        raise ValueError("tensor contains NaN or Inf; refusing to quantize")


# Reference rule: ceil(log2(x)) via the float32 bit pattern. An exact power of
# two keeps its exponent; any nonzero mantissa rounds the exponent up. Zero
# yields -127, the smallest E8M0 exponent, and a denormal yields -126, both
# matching the reference bit manipulation.
def pow2_ceil_exponent(x: torch.Tensor) -> torch.Tensor:
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return exponent - 127 + (mantissa != 0).to(torch.int32)


def pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(x)) as float32."""
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), pow2_ceil_exponent(x))


# E8M0 is the biased exponent byte; building it directly is exact over the whole
# range, whereas torch's float-to-E8M0 cast rounds the float32 denormal 2**-127
# up to 2**-126 and cannot produce the smallest E8M0 value.
def e8m0_from_exponent(k: torch.Tensor) -> torch.Tensor:
    if int(k.min()) < -127 or int(k.max()) > 127:
        raise ValueError("exponent outside the E8M0 range [-127, 127]")
    return (k + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


# floor(log2(x)) via the float32 bit pattern: the unbiased exponent field.
# Zero and denormals both yield -127 so the smallest E8M0 value is used.
def pow2_floor_exponent(x: torch.Tensor) -> torch.Tensor:
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    return torch.where(exponent == 0, torch.full_like(exponent, -127), exponent - 127)


# Exponent k as (2**k float32, E8M0 byte).
def _scale_pair(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ldexp(torch.ones_like(k, dtype=torch.float32), k), e8m0_from_exponent(k)


# FP8 block scale: 2**ceil(log2(amax / 448)), so the largest element never
# saturates. `amax_floor` is the minimum absmax; the reference floors
# activations at 1e-4 but the offline weight rule is unpublished, so callers
# choose it explicitly.
def fp8_block_scale(amax: torch.Tensor, amax_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    if amax_floor > 0:
        amax = amax.clamp_min(amax_floor)
    return _scale_pair(pow2_ceil_exponent(amax / FP8_MAX))


# FP4 group scale: 2**(floor(log2(amax)) - 2), the OCP MX rule the released
# expert weights follow. Elements above 6*scale saturate. The exponent is
# clamped at -127 so an all-zero group gets the smallest E8M0 value.
def fp4_group_scale(amax: torch.Tensor, amax_floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    if amax_floor > 0:
        amax = amax.clamp_min(amax_floor)
    return _scale_pair((pow2_floor_exponent(amax) - 2).clamp_min(-127))


# Round |q| onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} with ties to even
# mantissa. The grid step doubles at 1, 2 and 4, so rounding q/step with
# torch.round (half to even) and scaling back reproduces the hardware
# round-to-nearest-even cast. Callers clamp to [-6, 6] first.
def round_to_e2m1_magnitude(a: torch.Tensor) -> torch.Tensor:
    step = torch.where(a < 2.0, 0.5, torch.where(a < 4.0, 1.0, 2.0))
    return torch.round(a / step) * step


# Map E2M1 magnitudes (already on the grid) to code indexes 0..7.
def _magnitude_to_code(r: torch.Tensor) -> torch.Tensor:
    table = FP4_MAGNITUDES.to(r.device)
    idx = torch.searchsorted(table, r.reshape(-1).contiguous())
    idx = idx.clamp_max(7)
    if not torch.equal(table[idx], r.reshape(-1)):
        raise ValueError("magnitude not on the E2M1 grid")
    return idx.reshape(r.shape).to(torch.uint8)


def quantize_fp8_blocks(w: torch.Tensor, amax_floor: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to FP8 E4M3 with UE8M0 scales per 128x128 block.

    Returns (codes[out, in] float8_e4m3fn, scales[out/128, in/128] float8_e8m0fnu).
    """
    if w.ndim != 2 or w.shape[0] % FP8_BLOCK or w.shape[1] % FP8_BLOCK:
        raise ValueError(f"FP8 block quantization needs a 2-D shape divisible by {FP8_BLOCK}, got {tuple(w.shape)}")
    _check_finite(w)
    out, inp = w.shape
    blocks = w.float().reshape(out // FP8_BLOCK, FP8_BLOCK, inp // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))
    s, s_e8m0 = fp8_block_scale(amax, amax_floor)
    q = (blocks / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    codes = q.to(torch.float8_e4m3fn).reshape(out, inp)
    return codes, s_e8m0


def dequantize_fp8_blocks(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_fp8_blocks; returns float32 [out, in]."""
    out, inp = codes.shape
    if scales.shape != (out // FP8_BLOCK, inp // FP8_BLOCK):
        raise ValueError(f"scale shape {tuple(scales.shape)} does not match codes {tuple(codes.shape)}")
    blocks = codes.float().reshape(out // FP8_BLOCK, FP8_BLOCK, inp // FP8_BLOCK, FP8_BLOCK)
    return (blocks * scales.float()[:, None, :, None]).reshape(out, inp)


def quantize_fp4_rows(w: torch.Tensor, amax_floor: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to packed E2M1 with UE8M0 scales per 32 input elements.

    Returns (packed[out, in/2] uint8 with element 2k in the low nibble,
    scales[out, in/32] float8_e8m0fnu). The sign bit follows the sign of the
    input even when the magnitude rounds to zero, as the hardware cast and the
    released checkpoint do, so -0.0 and small negatives become code 8.
    """
    if w.ndim != 2 or w.shape[1] % FP4_GROUP:
        raise ValueError(f"FP4 quantization needs a 2-D shape with input dim divisible by {FP4_GROUP}, got {tuple(w.shape)}")
    _check_finite(w)
    out, inp = w.shape
    groups = w.float().reshape(out, inp // FP4_GROUP, FP4_GROUP)
    amax = groups.abs().amax(dim=-1)
    s, s_e8m0 = fp4_group_scale(amax, amax_floor)
    q = (groups / s[..., None]).clamp(-FP4_MAX, FP4_MAX)
    magnitude = round_to_e2m1_magnitude(q.abs())
    codes = _magnitude_to_code(magnitude)
    # Sign bit from the input's sign bit, including -0.0: parity with the
    # reference cast, which the source checkpoint's negative-zero codes confirm.
    codes = codes | (torch.signbit(q).to(torch.uint8) << 3)
    flat = codes.reshape(out, inp)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    return packed.contiguous(), s_e8m0


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Expand packed nibbles [out, in/2] to code indexes [out, in] as uint8."""
    if packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def dequantize_fp4_rows(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_fp4_rows; accepts int8 or uint8 packing. Returns float32 [out, in]."""
    codes = unpack_fp4_codes(packed)
    out, inp = codes.shape
    if scales.shape != (out, inp // FP4_GROUP):
        raise ValueError(f"scale shape {tuple(scales.shape)} does not match packed codes {tuple(packed.shape)}")
    values = FP4_VALUES.to(codes.device)[codes.long()]
    groups = values.reshape(out, inp // FP4_GROUP, FP4_GROUP)
    return (groups * scales.float()[..., None]).reshape(out, inp)
