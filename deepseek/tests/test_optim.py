"""CPU tests for MasterOptimizer on plain BF16 parameters: a step equals FP32
AdamW on an FP32 copy followed by a BF16 round, chunking does not change the
result, step_param equals step, sub-ulp updates accumulate in the FP32 masters
where BF16 masters would stall, and the inherited failure modes hold. For
LocalShardOptimizer: restrict_rows steps only the selected rows of a stacked
parameter, allocates state only for them, and is refused after a step; NeMo-style
parameter groups are mirrored into the inner optimizer, outer float rates reach
inner tensor rates in place, and restrict_rows works in any group."""

import copy
import unittest

import torch
from torch import nn

from deepseek.qat.optim import LocalShardOptimizer, MasterOptimizer, adamw_fp32_master

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


class RestrictRows(unittest.TestCase):
    # A [4, 8, 6] "expert stack" with rows 1 and 3 selected: the stepped rows
    # match AdamW run on those rows alone, the other rows are bit-identical, and
    # the inner optimizer holds state for exactly the two row views.
    def test_only_selected_rows_step_and_hold_state(self):
        torch.manual_seed(0)
        stack = nn.Parameter(torch.randn(4, 8, 6))
        ref_rows = [nn.Parameter(stack.detach()[r].clone()) for r in (1, 3)]
        opt = LocalShardOptimizer([stack], torch.optim.AdamW, lr=LR, weight_decay=0.0)
        opt.restrict_rows(stack, [3, 1])
        ref = torch.optim.AdamW(ref_rows, lr=LR, weight_decay=0.0)
        before = stack.detach().clone()
        for step in range(3):
            torch.manual_seed(step)
            stack.grad = torch.randn_like(stack)
            for q, r in zip(ref_rows, (1, 3)):
                q.grad = stack.grad[r].clone()
            ref.step()
            opt.step_param(stack)
            stack.grad = None
        for r in (0, 2):
            self.assertTrue(torch.equal(stack.detach()[r], before[r]))
        for q, r in zip(ref_rows, (1, 3)):
            self.assertTrue(torch.equal(stack.detach()[r], q.detach()))
        self.assertEqual(len(opt.inner.state), 2)
        self.assertEqual(len(opt.inner.param_groups[0]["params"]), 2)

    def test_empty_selection_steps_nothing(self):
        stack = nn.Parameter(torch.randn(3, 4, 4))
        opt = LocalShardOptimizer([stack], torch.optim.AdamW, lr=LR, weight_decay=0.0)
        opt.restrict_rows(stack, [])
        before = stack.detach().clone()
        stack.grad = torch.randn_like(stack)
        opt.step_param(stack)
        self.assertTrue(torch.equal(stack.detach(), before))
        self.assertEqual(len(opt.inner.state), 0)

    def test_refused_after_step_or_with_bad_rows(self):
        stack = nn.Parameter(torch.randn(3, 4, 4))
        opt = LocalShardOptimizer([stack], torch.optim.AdamW, lr=LR, weight_decay=0.0)
        with self.assertRaises(ValueError):
            opt.restrict_rows(stack, [0, 3])
        with self.assertRaises(ValueError):
            opt.restrict_rows(stack, [1, 1])
        with self.assertRaises(RuntimeError):
            opt.restrict_rows(nn.Parameter(torch.zeros(2)), [0])
        stack.grad = torch.randn_like(stack)
        opt.step_param(stack)
        with self.assertRaisesRegex(RuntimeError, "before the first optimizer step"):
            opt.restrict_rows(stack, [0])


class ParameterGroups(unittest.TestCase):
    # Two groups as NeMo's param_group_overrides deliver them, the second with
    # lr_mult. The inner groups mirror the outer ones, a float written to an
    # outer group's lr (what OptimizerParamScheduler.step does) reaches the inner
    # group at the next step, and the stepped values equal AdamW at those rates.
    def test_groups_mirror_and_rates_follow_outer_writes(self):
        dense, expert = _params()
        ref_dense, ref_expert = copy.deepcopy(dense), copy.deepcopy(expert)
        opt = LocalShardOptimizer([{"params": [dense]}, {"params": [expert], "lr_mult": 4.0}],
                                  torch.optim.AdamW, lr=LR, weight_decay=0.0, foreach=False)
        self.assertEqual(len(opt.inner.param_groups), 2)
        self.assertEqual(opt.inner.param_groups[1]["lr_mult"], 4.0)
        self.assertIsInstance(opt.param_groups[0]["lr"], float)
        opt.param_groups[1]["lr"] = 4 * LR
        ref = torch.optim.AdamW([{"params": [ref_dense]}, {"params": [ref_expert], "lr": 4 * LR}],
                                lr=LR, weight_decay=0.0, foreach=False)
        for step in range(2):
            _set_grads([dense, expert], step)
            _set_grads([ref_dense, ref_expert], step)
            opt.step()
            ref.step()
        self.assertTrue(torch.equal(dense, ref_dense))
        self.assertTrue(torch.equal(expert, ref_expert))
        self.assertEqual(opt.inner.param_groups[1]["lr"], 4 * LR)

    # torchao keeps each group's lr as a tensor and raises when a float replaces
    # it; the sync must fill that tensor in place rather than rebind the key.
    def test_tensor_lr_is_filled_in_place(self):
        class TensorLR(torch.optim.AdamW):
            def add_param_group(self, group):
                super().add_param_group(group)
                g = self.param_groups[-1]
                if not isinstance(g["lr"], torch.Tensor):
                    g["lr"] = torch.tensor(g["lr"], dtype=torch.float32)

        p = _params()[0]
        opt = LocalShardOptimizer([p], TensorLR, lr=LR, weight_decay=0.0, foreach=False)
        inner_lr = opt.inner.param_groups[0]["lr"]
        self.assertIsInstance(inner_lr, torch.Tensor)
        opt.param_groups[0]["lr"] = 2 * LR
        p.grad = torch.randn_like(p)
        opt.step_param(p)
        self.assertIs(opt.inner.param_groups[0]["lr"], inner_lr)
        self.assertAlmostEqual(float(inner_lr), 2 * LR)

    def test_mixed_list_is_rejected(self):
        dense, expert = _params()
        with self.assertRaises(TypeError):
            LocalShardOptimizer([dense, {"params": [expert]}], torch.optim.AdamW, lr=LR, weight_decay=0.0)

    # restrict_rows must find the parameter's chunk views in whichever inner
    # group holds them, not assume group 0.
    def test_restrict_rows_in_a_later_group(self):
        torch.manual_seed(0)
        dense = nn.Parameter(torch.randn(6, 4))
        stack = nn.Parameter(torch.randn(4, 8, 6))
        opt = LocalShardOptimizer([{"params": [dense]}, {"params": [stack], "lr_mult": 2.0}],
                                  torch.optim.AdamW, lr=LR, weight_decay=0.0)
        opt.restrict_rows(stack, [2])
        self.assertEqual(len(opt.inner.param_groups[0]["params"]), 1)
        self.assertEqual(len(opt.inner.param_groups[1]["params"]), 1)
        before = stack.detach().clone()
        stack.grad = torch.randn_like(stack)
        dense.grad = torch.randn_like(dense)
        opt.step_param(stack)
        opt.step_param(dense)
        for r in (0, 1, 3):
            self.assertTrue(torch.equal(stack.detach()[r], before[r]))
        self.assertFalse(torch.equal(stack.detach()[2], before[2]))
        self.assertEqual(len(opt.inner.state), 2)


if __name__ == "__main__":
    unittest.main()
