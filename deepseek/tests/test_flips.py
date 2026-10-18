"""Device flip accounting against the CPU reference in proxy_step.

CPU-only: the quantizers are pure PyTorch, so the counter's device path runs on
the CPU here. A synthetic target set mimics the hook's three tensor kinds (an FP8
dense weight, an FP4 gate/up stack, an FP4 down stack) on a 32-wide group grid.
"""

import unittest

import torch
from torch import nn

from deepseek.qat import fakequant as FQ
from deepseek.qat.flips import FlipCounter
from deepseek.qat.proxy_step import entry_value_changes, quantized_state, value_changes

torch.manual_seed(0)
E, DIM, INTER = 3, 64, 32  # experts, hidden, intermediate (both multiples of 32)


def _grid_aligned(shape, kind):
    """Random weights that are fixed points of their quantizer, like the source checkpoint."""
    w = torch.randn(*shape) * 0.05
    if kind == "fp8":
        return FQ._fp8_dequant_quant(w.bfloat16())
    if kind == "fp4_gate_up":
        return FQ.fake_quant_gate_up(w.bfloat16())
    return FQ.fake_quant_down(w.bfloat16())


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_and_up_projs = nn.Parameter(_grid_aligned((E, DIM, 2 * INTER), "fp4_gate_up"))
        self.down_projs = nn.Parameter(_grid_aligned((E, INTER, DIM), "fp4_down"))

    def forward(self, x):
        return x


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(_grid_aligned((128, 256), "fp8"))

    def forward(self, x):
        return x


def _state():
    dense, experts = Dense(), Experts()
    targets = [
        FQ.Target("model.layers.0.self_attn.wq_b", dense, {"weight": "fp8"}),
        FQ.Target("model.layers.0.mlp.experts", experts, {"gate_and_up_projs": "fp4_gate_up", "down_projs": "fp4_down"}),
    ]
    state = FQ.FakeQuantState(targets)
    for t in targets:
        original = t.module.forward
        state.originals[t.name] = original

        def wrapped(*args, _name=t.name, _module=t.module, _params=t.params, _orig=original, **kwargs):
            for attr, kind in _params.items():
                observer = state.observer(_name, attr, kind) if state.observer is not None else None
                FQ.observed(getattr(_module, attr), kind, observer)
            return _orig(*args, **kwargs)

        t.module.forward = wrapped
    return state, dense, experts


def _materialize(name, t):
    return t


def _names(state):
    return {f"{t.name}.{a}" for t in state.targets for a in t.params}


class DeviceCounts(unittest.TestCase):
    def setUp(self):
        self.state, self.dense, self.experts = _state()
        self.before = {f"{t.name}.{a}": quantized_state(f"{t.name}.{a}", getattr(t.module, a), k)
                       for t in self.state.targets for a, k in t.params.items()}
        self.counter = FlipCounter(self.state, _names(self.state), _materialize, torch.device("cpu"))

    def _cpu_reference(self):
        out = {}
        for t in self.state.targets:
            for a, k in t.params.items():
                name = f"{t.name}.{a}"
                after = quantized_state(name, getattr(t.module, a), k)
                c0, s0 = self.before[name]
                out[name] = {
                    "values_changed": value_changes(self.before[name], after, k),
                    "codes_flipped": int((c0 != after[0]).sum()),
                    "scales_changed": int((s0 != after[1]).sum()),
                    "entries": None if k == "fp8" else entry_value_changes(self.before[name], after),
                }
        return out

    def _pass(self):
        self.counter.begin_pass()
        for t in self.state.targets:
            t.module.forward(None)

    def test_untrained_weights_count_nothing(self):
        self._pass()
        totals = self.counter.totals()
        self.assertEqual(totals["dense_values_changed"], 0)
        self.assertEqual(totals["expert_values_changed"], 0)
        self.assertEqual(totals["dense_value_elements"], 128 * 256)
        self.assertEqual(totals["expert_value_elements"], E * (DIM * 2 * INTER + INTER * DIM))
        self.assertEqual(totals["expert_value_elements_layer0"], totals["expert_value_elements"])
        # Fixed denominators: every dense target's elements (none frozen here) and
        # every routed expert of the model from the config (not only the local shard).
        self.assertEqual(totals["dense_class_elements"], 128 * 256)
        self.assertEqual(totals["expert_class_elements"], 0)  # not given to this counter
        counter = FlipCounter(self.state, _names(self.state), _materialize, torch.device("cpu"), expert_class_elements=10 ** 12)
        counter.refresh(_materialize)
        self.assertEqual(counter.totals()["expert_class_elements"], 10 ** 12)

    def test_counts_match_cpu_reference_after_perturbation(self):
        with torch.no_grad():
            # A half-cell crossing in the dense weight, a sign flip of zero codes
            # (must not count as a value change), and expert moves in experts 0 and 2.
            w = self.dense.weight
            w[0, 0] += w[0, 0].abs() * 0.3 + 1e-2
            z = (w == 0).nonzero()[:5]
            w[z[:, 0], z[:, 1]] = -0.0
            gu = self.experts.gate_and_up_projs
            gu[0, :8, :4] *= 3.0
            gu[2, 5, INTER + 3] = 1.0
            self.experts.down_projs[2, :, :3] *= 0.0
        self._pass()
        ref = self._cpu_reference()
        tensors = self.counter.per_tensor()
        for name, r in ref.items():
            for key in ("values_changed", "codes_flipped", "scales_changed"):
                self.assertEqual(tensors[name][key], r[key], f"{name} {key}")
        totals = self.counter.totals()
        self.assertEqual(totals["dense_values_changed"], ref["model.layers.0.self_attn.wq_b.weight"]["values_changed"])
        self.assertEqual(totals["expert_values_changed"],
                         ref["model.layers.0.mlp.experts.gate_and_up_projs"]["values_changed"] + ref["model.layers.0.mlp.experts.down_projs"]["values_changed"])
        self.assertGreater(totals["expert_values_changed"], 0)
        # Per expert: gate/up entries alternate w1, w3 per expert in the reference.
        values, _, _ = self.counter.per_expert("model.layers.0.mlp.experts.gate_and_up_projs")
        entries = ref["model.layers.0.mlp.experts.gate_and_up_projs"]["entries"]
        self.assertEqual(values, [entries[2 * i] + entries[2 * i + 1] for i in range(E)])
        values, _, _ = self.counter.per_expert("model.layers.0.mlp.experts.down_projs")
        self.assertEqual(values, ref["model.layers.0.mlp.experts.down_projs"]["entries"])
        self.assertEqual(values[1], 0)

    def test_second_observation_in_a_pass_is_skipped_and_new_pass_recounts(self):
        original = self.dense.weight.detach().clone()
        with torch.no_grad():
            self.dense.weight[1, 1] += 1.0
        self._pass()
        first = self.counter.totals()["dense_values_changed"]
        self.assertGreater(first, 0)
        # The activation-checkpoint recompute: same pass, must not double count.
        for t in self.state.targets:
            t.module.forward(None)
        self.assertEqual(self.counter.totals()["dense_values_changed"], first)
        with torch.no_grad():
            self.dense.weight.copy_(original)  # bf16 cannot undo +1.0 with -1.0 exactly
        self._pass()
        self.assertEqual(self.counter.totals()["dense_values_changed"], 0)

    def test_incomplete_pass_is_an_error(self):
        self.counter.begin_pass()
        self.state.targets[0].module.forward(None)
        with self.assertRaises(RuntimeError):
            self.counter.totals()

    def test_refresh_counts_without_a_forward(self):
        with torch.no_grad():
            self.experts.down_projs[0, 0, :] *= 4.0
        self.counter.refresh(_materialize)
        values, _, _ = self.counter.per_expert("model.layers.0.mlp.experts.down_projs")
        self.assertGreater(values[0], 0)
        self.assertEqual(values[1], 0)
        self.assertEqual(values[2], 0)

    def test_uncounted_targets_are_not_observed(self):
        state, dense, experts = _state()
        counter = FlipCounter(state, {"model.layers.0.mlp.experts.down_projs"}, _materialize, torch.device("cpu"))
        counter.begin_pass()
        for t in state.targets:
            t.module.forward(None)
        totals = counter.totals()
        self.assertEqual(totals["dense_value_elements"], 0)
        self.assertEqual(totals["expert_value_elements"], E * INTER * DIM)
        self.assertNotIn("model.layers.0.self_attn.wq_b.weight", counter.per_tensor())


if __name__ == "__main__":
    unittest.main()
