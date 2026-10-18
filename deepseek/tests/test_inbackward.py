"""CPU tests for update-in-backward on plain parameters: the hooked path equals a
normal step with the same clip coefficient, the stored norm is the exact global
norm, the coefficient lags one step, gradients are freed, the scaling callback
sees exactly one gradient, a class on an update period skips its optimizer call
without touching its moments, a row selection steps only the selected rows and
excludes the rest from the norm, and a parameter without a gradient is an error."""

import copy
import math
import unittest

import torch
from torch import nn

from deepseek.qat.inbackward import InBackwardStepper
from deepseek.qat.optim import LocalShardOptimizer

MAX_NORM = 0.5
LR = 1e-2


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8))


def _loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).square().mean()


def _global_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.parameters()))


class Equivalence(unittest.TestCase):
    def setUp(self):
        self.hooked = _model()
        self.reference = copy.deepcopy(self.hooked)
        self.opt_hooked = LocalShardOptimizer(list(self.hooked.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        self.opt_ref = torch.optim.AdamW(self.reference.parameters(), lr=LR, weight_decay=0.0)
        self.seen = []
        stepper_self = self

        def scale_grads():
            stepper_self.seen.append(sum(p.grad is not None for p in stepper_self.hooked.parameters()))

        self.stepper = InBackwardStepper([self.hooked], self.opt_hooked, MAX_NORM, scale_grads)
        torch.manual_seed(1)
        self.batches = [torch.randn(4, 16) for _ in range(3)]

    def test_matches_reference_with_lagged_coefficient(self):
        coef = 1.0
        for x in self.batches:
            # Reference: ordinary backward, clip by the previous step's norm, step.
            self.opt_ref.zero_grad()
            _loss(self.reference, x).backward()
            norm = _global_norm(self.reference)
            for p in self.reference.parameters():
                p.grad.mul_(coef)
            self.opt_ref.step()

            # Hooked: the step happens inside backward.
            _loss(self.hooked, x).backward()
            self.assertTrue(all(p.grad is None for p in self.hooked.parameters()), "gradients were not freed")
            reported = self.stepper.end_step()
            self.assertAlmostEqual(reported, norm, places=5)
            self.assertAlmostEqual(self.stepper.coef, min(1.0, MAX_NORM / (norm + 1e-6)), places=6)
            coef = self.stepper.coef

            for a, b in zip(self.hooked.parameters(), self.reference.parameters()):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(self.stepper.step_count, 3)
        # The scaling callback always ran with exactly one gradient alive.
        self.assertEqual(set(self.seen), {1})
        self.assertEqual(len(self.seen), 3 * len(list(self.hooked.parameters())))

    def test_first_step_is_unclipped(self):
        self.assertEqual(self.stepper.coef, 1.0)
        _loss(self.hooked, self.batches[0]).backward()
        norm = self.stepper.end_step()
        self.assertGreater(norm, MAX_NORM, "test data must produce a norm above the limit")
        self.assertLess(self.stepper.coef, 1.0)


class PerClass(unittest.TestCase):
    # Two clipping classes split by module name: each class gets its own norm
    # and coefficient, and the parameters match a reference that clips per class.
    def test_per_class_coefficients_and_parameters(self):
        hooked = _model()
        reference = copy.deepcopy(hooked)
        opt_h = LocalShardOptimizer(list(hooked.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        opt_r = torch.optim.AdamW(reference.parameters(), lr=LR, weight_decay=0.0)
        class_of = lambda name: "a" if name.startswith("0.") else "b"
        tight = 0.05  # below both class norms throughout, so both coefficients are active
        stepper = InBackwardStepper([hooked], opt_h, tight, lambda: None, class_of=class_of)
        self.assertEqual(stepper.classes, ["a", "b"])
        with self.assertRaises(AttributeError):
            stepper.coef
        coefs = {"a": 1.0, "b": 1.0}
        torch.manual_seed(1)
        for x in [torch.randn(4, 16) for _ in range(3)]:
            opt_r.zero_grad()
            _loss(reference, x).backward()
            norms = {c: math.sqrt(sum(float(p.grad.float().square().sum()) for n, p in reference.named_parameters() if class_of(n) == c)) for c in coefs}
            for n, p in reference.named_parameters():
                p.grad.mul_(coefs[class_of(n)])
            opt_r.step()

            _loss(hooked, x).backward()
            stepper.end_step()
            for c in coefs:
                self.assertAlmostEqual(stepper.last_norms[c], norms[c], places=5)
                self.assertAlmostEqual(stepper.coefs[c], min(1.0, tight / (norms[c] + 1e-6)), places=6)
            coefs = dict(stepper.coefs)
            for a, b in zip(hooked.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(a, b))
        # The two classes saw different norms, so the test exercised two coefficients.
        self.assertNotAlmostEqual(stepper.coefs["a"], stepper.coefs["b"], places=3)


class UpdatePeriod(unittest.TestCase):
    # Class "a" updates on 1 step in 2, class "b" every step. The reference skips
    # the optimizer for class "a" on the other steps by dropping its gradients, which
    # leaves torch AdamW's moments and step counters for those parameters untouched;
    # the hooked parameters must match it exactly on every step, and the norm must
    # still include the skipped class's gradient.
    def test_skipped_steps_match_reference_and_leave_moments_alone(self):
        hooked = _model()
        reference = copy.deepcopy(hooked)
        opt_h = LocalShardOptimizer(list(hooked.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        opt_r = torch.optim.AdamW(reference.parameters(), lr=LR, weight_decay=0.0)
        class_of = lambda name: "a" if name.startswith("0.") else "b"
        loose = 1e9  # clipping off, so the comparison isolates the update period
        stepper = InBackwardStepper([hooked], opt_h, loose, lambda: None, class_of=class_of, every={"a": 2})
        torch.manual_seed(1)
        a_params = [p for n, p in reference.named_parameters() if class_of(n) == "a"]
        for step, x in enumerate([torch.randn(4, 16) for _ in range(4)]):
            opt_r.zero_grad()
            _loss(reference, x).backward()
            norm = _global_norm(reference)
            if step % 2:
                for p in a_params:
                    p.grad = None
            opt_r.step()

            _loss(hooked, x).backward()
            self.assertTrue(all(p.grad is None for p in hooked.parameters()), "gradients were not freed")
            self.assertAlmostEqual(stepper.end_step(), norm, places=5)
            for a, b in zip(hooked.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(a, b), f"mismatch at step {step}")
        # Class "a" stepped twice (steps 0 and 2); class "b" four times.
        a_state = [opt_r.state[p]["step"] for p in a_params]
        self.assertTrue(all(float(s) == 2 for s in a_state))
        self.assertEqual(stepper.step_count, 4)

    def test_rejects_unknown_class_and_bad_period(self):
        model = _model()
        opt = LocalShardOptimizer(list(model.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        class_of = lambda name: "a" if name.startswith("0.") else "b"
        with self.assertRaisesRegex(ValueError, "unknown class"):
            InBackwardStepper([model], opt, MAX_NORM, lambda: None, class_of=class_of, every={"c": 2})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            InBackwardStepper([model], opt, MAX_NORM, lambda: None, class_of=class_of, every={"a": 0})


class RowSelection(unittest.TestCase):
    # rows_of selects rows 0..7 of the first Linear's weight: the hooked model
    # matches a reference that zeroes the other rows' gradients before its
    # step (AdamW with a zero gradient and zero moments leaves a row unchanged
    # only if it never had state, so the reference drops those rows from its
    # optimizer entirely), and the reported norm excludes the unselected rows.
    def test_selected_rows_only(self):
        hooked = _model()
        reference = copy.deepcopy(hooked)
        rows = list(range(8))
        opt_h = LocalShardOptimizer(list(hooked.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        ref_params = list(reference.parameters())
        w0 = ref_params[0]
        w0_rows = nn.Parameter(w0.detach()[rows].clone())
        opt_r = torch.optim.AdamW([w0_rows] + ref_params[1:], lr=LR, weight_decay=0.0)
        rows_of = lambda name: rows if name == "0.weight" else None
        stepper = InBackwardStepper([hooked], opt_h, 1e9, lambda: None, rows_of=rows_of)
        torch.manual_seed(1)
        for x in [torch.randn(4, 16) for _ in range(3)]:
            opt_r.zero_grad()
            _loss(reference, x).backward()
            g0 = w0.grad.clone()
            w0.grad = None
            w0_rows.grad = g0[rows]
            norm = math.sqrt(float(g0[rows].float().square().sum()) + sum(float(p.grad.float().square().sum()) for p in ref_params[1:]))
            opt_r.step()
            with torch.no_grad():
                w0[rows] = w0_rows
            _loss(hooked, x).backward()
            self.assertAlmostEqual(stepper.end_step(), norm, places=5)
            for a, b in zip(hooked.parameters(), reference.parameters()):
                self.assertTrue(torch.equal(a, b))
        # State for the 8 row views plus the three unrestricted parameters.
        self.assertEqual(len(opt_h.inner.state), len(rows) + len(ref_params) - 1)

    def test_requires_restrict_rows(self):
        model = _model()

        class NoRestrict:
            param_groups = [{"params": [], "lr": LR}]

            def step_param(self, p):
                pass

        with self.assertRaises(TypeError):
            InBackwardStepper([model], NoRestrict(), MAX_NORM, lambda: None, rows_of=lambda n: [0])


class Failures(unittest.TestCase):
    def test_parameter_without_gradient_is_an_error(self):
        model = _model()
        extra = nn.Linear(4, 4)
        wrapper = nn.ModuleDict({"used": model, "unused": extra})
        opt = LocalShardOptimizer(list(wrapper.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        stepper = InBackwardStepper([wrapper], opt, MAX_NORM, lambda: None)
        _loss(model, torch.randn(2, 16)).backward()
        with self.assertRaisesRegex(RuntimeError, "received no gradient"):
            stepper.end_step()

    def test_rejects_wrong_optimizer_and_bad_norm(self):
        model = _model()
        with self.assertRaises(TypeError):
            InBackwardStepper([model], torch.optim.AdamW(model.parameters()), MAX_NORM, lambda: None)
        opt = LocalShardOptimizer(list(model.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        with self.assertRaises(ValueError):
            InBackwardStepper([model], opt, 0.0, lambda: None)


if __name__ == "__main__":
    unittest.main()


class DriftSamples(unittest.TestCase):
    # A fixed slice of a parameter is tracked: mean |master - baseline| and the
    # sign consistency of its updates, 1.0 when every update pushed the same way.
    def test_consistent_updates_have_consistency_one_and_drift_equals_displacement(self):
        torch.manual_seed(0)
        model = _model()
        opt = LocalShardOptimizer(list(model.parameters()), torch.optim.AdamW, lr=LR, weight_decay=0.0)
        sample_of = lambda name, p: ("dense", (slice(0, 2),)) if name == "0.weight" else None
        stepper = InBackwardStepper([model], opt, 1e9, lambda: None, drift_sample_of=sample_of)
        p0 = model[0].weight.detach()[0:2].clone()
        x = torch.randn(4, 16)
        for _ in range(3):
            _loss(model, x).backward()
            stepper.end_step()
        drift = stepper.last_drift["dense"]
        self.assertAlmostEqual(drift["mean_abs_drift"], float((model[0].weight.detach()[0:2] - p0).abs().mean()), places=6)
        # Same input each step: most updates keep their sign, so |drift| is most of the summed |update|.
        self.assertGreater(drift["consistency"], 0.8)
        self.assertLessEqual(drift["consistency"], 1.0)
        self.assertEqual(stepper.drift_classes, ["dense"])
        self.assertEqual([n for n, q in model.named_parameters() if id(q) in stepper._drift], ["0.weight"])
        per = stepper.last_drift_by_param["0.weight"]
        self.assertAlmostEqual(per["mean_abs_drift"], drift["mean_abs_drift"], places=9)
        self.assertEqual(per["sampled_elements"], 2 * 16)
