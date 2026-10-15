"""CPU tests for MasterOptimizer on plain BF16 parameters: a step equals FP32
AdamW on an FP32 copy followed by a BF16 round, chunking does not change the
result, step_param equals step, sub-ulp updates accumulate in the FP32 masters
where BF16 masters would stall, and the inherited failure modes hold."""

import copy
import unittest

import torch
from torch import nn

from deepseek.qat.optim import MasterOptimizer, adamw_fp32_master

LR = 1e-3


def _params(seed: int = 0) -> list[nn.Parameter]:
    torch.manual_seed(seed)
    return [nn.Parameter(torch.randn(64, 48).to(torch.bfloat16)), nn.Parameter(torch.randn(24, 16).to(torch.bfloat16))]


def _set_grads(params, seed: int) -> None:
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


class MatchesReference(unittest.TestCase):
    def test_step_equals_fp32_adamw_then_bf16_round(self):
        params = _params()
        masters = [nn.Parameter(p.detach().float().clone()) for p in params]
        opt = MasterOptimizer(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
        ref = torch.optim.AdamW(masters, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8, foreach=False)
        for step in range(3):
            _set_grads(params, step)
            for m, p in zip(masters, params):
                m.grad = p.grad.float()
            ref.step()
            opt.step()
            for m, p in zip(masters, params):
                self.assertTrue(torch.equal(p.detach(), m.detach().to(torch.bfloat16)))
            self.assertTrue(all(p.grad is not None for p in params), "step must not clear the model gradients")

    def test_chunking_is_exact(self):
        a = _params()
        b = copy.deepcopy(a)
        whole = MasterOptimizer(a, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
        chunked = MasterOptimizer(b, max_chunk_elems=48 * 7, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
        self.assertGreater(len(chunked._entries[0][1]), 1)
        for step in range(2):
            _set_grads(a, step)
            _set_grads(b, step)
            whole.step()
            chunked.step()
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x, y))

    def test_step_param_equals_step(self):
        a = _params()
        b = copy.deepcopy(a)
        one = MasterOptimizer(a, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
        each = MasterOptimizer(b, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
        _set_grads(a, 0)
        _set_grads(b, 0)
        one.step()
        for p in b:
            each.step_param(p)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))


class SubUlpAccumulation(unittest.TestCase):
    # A constant gradient with lr far below a BF16 ulp at magnitude 1: the FP32
    # master accumulates and eventually crosses a BF16 rounding boundary, while
    # rounding each step's result to BF16 directly would never move.
    def test_fp32_masters_accumulate_below_bf16_ulp(self):
        p = nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16))
        opt = MasterOptimizer([p], lr=1e-4, weight_decay=0.0, betas=(0.0, 0.0), eps=1e-8)
        start = p.detach().clone()
        moved = False
        for _ in range(100):
            p.grad = torch.ones_like(p)
            opt.step()
            if not torch.equal(p.detach(), start):
                moved = True
                break
        self.assertTrue(moved, "FP32 masters should cross a BF16 ulp within 100 sub-ulp steps")
        # bf16 ulp at 1.0 is 2^-7; 100 steps of 1e-4 is 1e-2, so at most two ulps moved.
        self.assertLess(float((start.float() - p.detach().float()).max()), 2 * 2**-7 + 1e-9)


class Failures(unittest.TestCase):
    def test_factory_and_unsupported_state_dict(self):
        params = _params()
        opt = adamw_fp32_master(params, lr=LR, weight_decay=0.0, betas=[0.9, 0.95])
        self.assertIsInstance(opt, MasterOptimizer)
        self.assertEqual(opt.param_groups[0]["lr"], LR)
        with self.assertRaises(NotImplementedError):
            opt.state_dict()

    def test_step_param_without_gradient_or_unknown_parameter(self):
        params = _params()
        opt = MasterOptimizer(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8)
        with self.assertRaisesRegex(RuntimeError, "no gradient"):
            opt.step_param(params[0])
        stranger = nn.Parameter(torch.zeros(2, 2))
        stranger.grad = torch.zeros(2, 2)
        with self.assertRaisesRegex(RuntimeError, "does not own"):
            opt.step_param(stranger)


if __name__ == "__main__":
    unittest.main()
