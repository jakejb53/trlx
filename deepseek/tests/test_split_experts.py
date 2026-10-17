"""CPU tests for split expert storage (stage 6b, gate 1): NeMo's on-the-fly FP4
dequantization agrees with the project quantizer, including the zero-exponent
scale; the selection rules reject uneven shards; the adapter routes a synthetic
two-layer state dict into the right stacks and shapes, without a mesh and on a
one-rank gloo expert mesh; the split forward equals a single-stack reference and
keeps gradients flowing to the BF16 stack only; and a one-layer meta build from
the snapshot config runs through the constructor hook."""

import os
import socket
import unittest

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.quantization.mxfp4 import dequantize_mxfp4

from deepseek.qat import quant as Q
from deepseek.qat.split_adapter import SplitStateDictAdapter
from deepseek.qat.split_experts import ExpertSplit, SplitExperts

SNAP = "deepseek/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062"
E, DIM, INTER = 4, 64, 32


def _moe_config() -> MoEConfig:
    return MoEConfig(n_routed_experts=E, n_shared_experts=1, n_activated_experts=2, n_expert_groups=0, n_limited_groups=0,
                     train_gate=False, gate_bias_update_factor=0.0, aux_loss_coeff=0.0, score_func="sqrtsoftplus",
                     route_scale=1.0, dim=DIM, inter_dim=INTER, moe_inter_dim=INTER, norm_topk_prob=False,
                     dtype=torch.bfloat16, swiglu_limit=10.0)


def _backend() -> BackendConfig:
    return BackendConfig(attn="sdpa", linear="torch", experts="torch_mm", dispatcher="torch", enable_hf_state_dict_adapter=True)


# One synthetic HF expert set: random matrices quantized to the released FP4
# layout, so each expert exists as (codes, scales) and as its dequantized value.
def _hf_experts(layers: int, seed: int = 0) -> dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for layer in range(layers):
        for e in range(E):
            for which, shape in (("w1", (INTER, DIM)), ("w3", (INTER, DIM)), ("w2", (DIM, INTER))):
                out[(layer, e, which)] = Q.quantize_fp4_rows(torch.randn(shape, generator=gen))
    return out


def _deq(entry) -> torch.Tensor:
    return Q.dequantize_fp4_rows(*entry).bfloat16()


class Dequantize(unittest.TestCase):
    def test_matches_project_dequantizer_including_zero_exponent(self):
        packed = torch.randint(0, 256, (8, 32), dtype=torch.uint8).view(torch.int8)
        scales = torch.randint(0, 255, (8, 2), dtype=torch.uint8)
        scales[0, 0] = 0  # E8M0 byte 0: both paths decode 2**-127
        scales = scales.view(torch.float8_e8m0fnu)
        a = dequantize_mxfp4(packed, scales, torch.float32)
        b = Q.dequantize_fp4_rows(packed, scales)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(float(scales[0, 0].float()), 2.0**-127)

    @unittest.skipUnless(os.path.isdir(SNAP), "snapshot not present")
    def test_matches_on_source_tensors(self):
        from deepseek.qat.proxy import Source
        src = Source(SNAP)
        for name in ("layers.3.ffn.experts.17.w1", "layers.3.ffn.experts.200.w2"):
            codes, scales = src.tensor(name + ".weight"), src.tensor(name + ".scale")
            self.assertTrue(torch.equal(dequantize_mxfp4(codes, scales, torch.float32), Q.dequantize_fp4_rows(codes, scales)))


class SelectionRules(unittest.TestCase):
    def test_equal_count_per_rank_and_slot_order(self):
        split = ExpertSplit.from_layers({0: [1, 2], 1: [0, 3]}, n_experts=4, ranks=2)
        self.assertEqual(split.selected_per_rank(0), 1)
        self.assertEqual(split.rank_selected(0, 0), [1])
        self.assertEqual(split.rank_unselected(0, 0), [0])
        self.assertEqual(split.rank_unselected(1, 1), [2])
        self.assertEqual(split.stack_ids(0, "selected", None), [1, 2])
        self.assertEqual(split.stack_ids(1, "packed", None), [1, 2])
        self.assertEqual(split.stack_ids(1, "packed", 1), [2])
        self.assertTrue(split.is_selected(1, 3))
        self.assertFalse(split.is_selected(1, 1))

    def test_rejects_uneven_duplicate_and_out_of_range(self):
        with self.assertRaises(ValueError):
            ExpertSplit.from_layers({0: [0, 1]}, n_experts=4, ranks=2)  # both on rank 0
        with self.assertRaises(ValueError):
            ExpertSplit.from_layers({0: [0, 0]}, n_experts=4, ranks=1)
        with self.assertRaises(ValueError):
            ExpertSplit.from_layers({0: [4]}, n_experts=4, ranks=1)
        with self.assertRaises(ValueError):
            ExpertSplit.from_layers({0: [0]}, n_experts=6, ranks=4)

    def test_report_needs_ranks(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scores.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"n_routed_experts": 4, "layers": {"0": {"selected": [1]}}}, f)
            with self.assertRaises(ValueError):
                ExpertSplit(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"n_routed_experts": 4, "ranks": 1, "layers": {"0": {"selected": [1]}}}, f)
            self.assertEqual(ExpertSplit(path).layers, {0: [1]})


# A model skeleton with the module paths the adapter and the hook address.
class _Skeleton(nn.Module):
    def __init__(self, split: ExpertSplit):
        super().__init__()
        cfg = _moe_config()
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict()
        for layer in split.layers:
            block = nn.Module()
            block.mlp = nn.Module()
            block.mlp.experts = SplitExperts(GroupedExperts(cfg, backend=_backend()), split, layer)
            self.model.layers[str(layer)] = block


def _adapter(split: ExpertSplit) -> SplitStateDictAdapter:
    cfg = DeepseekV4Config(num_hidden_layers=len(split.layers), num_nextn_predict_layers=0, hidden_size=DIM,
                           moe_intermediate_size=INTER, n_routed_experts=E)
    base = DeepSeekV4StateDictAdapter(cfg, _moe_config(), _backend(), dtype=torch.bfloat16)
    return SplitStateDictAdapter(base, split)


# Fill a skeleton's stacks from the synthetic HF experts the way a load would.
def _fill(skeleton: _Skeleton, split: ExpertSplit, hf) -> None:
    with torch.no_grad():
        for key, block in skeleton.model.layers.items():
            layer, m = int(key), block.mlp.experts
            sel, pk = split.rank_selected(layer, 0), split.rank_unselected(layer, 0)
            if sel:
                m.gate_and_up_projs.copy_(torch.stack([torch.cat([_deq(hf[(layer, e, "w1")]).T, _deq(hf[(layer, e, "w3")]).T], -1) for e in sel]))
                m.down_projs.copy_(torch.stack([_deq(hf[(layer, e, "w2")]).T for e in sel]))
            if pk:
                for part, suffix in ((0, "_packed"), (1, "_scales")):
                    gu = torch.stack([torch.cat([hf[(layer, e, "w1")][part], hf[(layer, e, "w3")][part]], 0) for e in pk])
                    getattr(m, "gate_and_up_projs" + suffix).copy_(gu.view(torch.int8) if part == 0 else gu)
                    dn = torch.stack([hf[(layer, e, "w2")][part] for e in pk])
                    getattr(m, "down_projs" + suffix).copy_(dn.view(torch.int8) if part == 0 else dn)


def _hf_state_dict(hf, layers: int) -> dict[str, torch.Tensor]:
    sd = {"layers.0.attn_norm.weight": torch.ones(DIM)}
    for layer in range(layers):
        for e in range(E):
            for which in ("w1", "w3", "w2"):
                codes, scales = hf[(layer, e, which)]
                sd[f"layers.{layer}.ffn.experts.{e}.{which}.weight"] = codes.view(torch.int8).clone()
                sd[f"layers.{layer}.ffn.experts.{e}.{which}.scale"] = scales.clone()
    return sd


class Adapter(unittest.TestCase):
    """Layer 0 selects experts 1 and 2; layer 1 selects expert 0 only."""

    def setUp(self):
        self.split = ExpertSplit.from_layers({0: [1, 2], 1: [0]}, n_experts=E, ranks=1)
        self.hf = _hf_experts(2)
        self.skeleton = _Skeleton(self.split)
        _fill(self.skeleton, self.split, self.hf)
        self.adapter = _adapter(self.split)

    def test_module_shapes(self):
        m0 = self.skeleton.model.layers["0"].mlp.experts
        self.assertEqual(tuple(m0.gate_and_up_projs.shape), (2, DIM, 2 * INTER))
        self.assertEqual(tuple(m0.down_projs.shape), (2, INTER, DIM))
        self.assertEqual(tuple(m0.gate_and_up_projs_packed.shape), (2, 2 * INTER, DIM // 2))
        self.assertEqual(tuple(m0.gate_and_up_projs_scales.shape), (2, 2 * INTER, DIM // 32))
        self.assertEqual(tuple(m0.down_projs_packed.shape), (2, DIM, INTER // 2))
        self.assertEqual(m0.gate_and_up_projs_scales.dtype, torch.float8_e8m0fnu)
        self.assertFalse(m0.gate_and_up_projs_packed.requires_grad)
        self.assertTrue(m0.gate_and_up_projs.requires_grad)
        self.assertEqual(m0.local_ids(), ([1, 2], [0, 3]))
        trainable = {n for n, p in self.skeleton.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {f"model.layers.{l}.mlp.experts.{a}" for l in (0, 1) for a in ("gate_and_up_projs", "down_projs")})

    def test_freeze_selectors_skip_packed_storage(self):
        # NeMo patch 2: the recipe's unfreeze glob must not raise on the int8
        # codes and must leave the E8M0 scales frozen.
        from nemo_automodel.components.utils.model_utils import apply_parameter_freezing
        for p in self.skeleton.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)
        apply_parameter_freezing(self.skeleton, {"freeze_modules": [{"glob": "model"}],
                                                 "unfreeze_modules": [{"glob": "model.layers.*.mlp.experts"}]})
        trainable = {n for n, p in self.skeleton.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {f"model.layers.{l}.mlp.experts.{a}" for l in (0, 1) for a in ("gate_and_up_projs", "down_projs")})

    def _check_to_hf(self, hf_sd, skeleton):
        expected = set()
        for layer in (0, 1):
            for e in range(E):
                for w in ("w1", "w3", "w2"):
                    expected.add(f"layers.{layer}.ffn.experts.{e}.{w}.weight")
                    expected.add(f"layers.{layer}.ffn.experts.{e}.{w}.scale")
        self.assertEqual({k for k in hf_sd if "experts" in k}, expected)
        for layer in (0, 1):
            m = skeleton.model.layers[str(layer)].mlp.experts
            for e in range(E):
                w1, s1 = hf_sd[f"layers.{layer}.ffn.experts.{e}.w1.weight"], hf_sd[f"layers.{layer}.ffn.experts.{e}.w1.scale"]
                w2 = hf_sd[f"layers.{layer}.ffn.experts.{e}.w2.weight"]
                w1, s1, w2 = (t.to_local() if isinstance(t, DTensor) else t for t in (w1, s1, w2))
                self.assertEqual((w1.dtype, s1.dtype), (torch.int8, torch.float8_e8m0fnu))
                self.assertEqual(tuple(w1.shape), (INTER, DIM // 2))
                self.assertEqual(tuple(s1.shape), (INTER, DIM // 32))
                self.assertEqual(tuple(w2.shape), (DIM, INTER // 2))
                if not self.split.is_selected(layer, e):
                    # Passthrough views into the module's storage, holding the codes.
                    packed = m.gate_and_up_projs_packed.to_local() if isinstance(m.gate_and_up_projs_packed, DTensor) else m.gate_and_up_projs_packed
                    self.assertEqual(w1.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr())
                    self.assertTrue(torch.equal(w1, self.hf[(layer, e, "w1")][0].view(torch.int8)))
                    self.assertTrue(torch.equal(w2, self.hf[(layer, e, "w2")][0].view(torch.int8)))

    def test_to_hf_placeholders_and_views(self):
        hf_sd = self.adapter.to_hf(self.skeleton.state_dict(), quantization=True)
        self._check_to_hf(hf_sd, self.skeleton)
        self.assertIn("model.layers.0.mlp.experts.gate_and_up_projs", self.skeleton.state_dict())
        keys = self.adapter.get_hf_state_dict_keys(self.skeleton.state_dict())
        self.assertEqual(len([k for k in keys if "experts" in k and k.endswith(".weight")]), 2 * E * 3)

    def _check_from_hf(self, native, mesh):
        for layer in (0, 1):
            base = f"model.layers.{layer}.mlp.experts."
            sel, pk = self.split.rank_selected(layer, 0), self.split.rank_unselected(layer, 0)
            gu = native[base + "gate_and_up_projs"]
            self.assertEqual(isinstance(gu, DTensor), mesh is not None)
            gu = gu.to_local() if mesh is not None else gu
            self.assertEqual(gu.dtype, torch.bfloat16)
            for i, e in enumerate(sel):
                exp = torch.cat([_deq(self.hf[(layer, e, "w1")]).T, _deq(self.hf[(layer, e, "w3")]).T], -1)
                self.assertTrue(torch.equal(gu[i], exp))
                dn = native[base + "down_projs"]
                dn = dn.to_local() if mesh is not None else dn
                self.assertTrue(torch.equal(dn[i], _deq(self.hf[(layer, e, "w2")]).T))
            gup = native[base + "gate_and_up_projs_packed"]
            gup = gup.to_local() if mesh is not None else gup
            gus = native[base + "gate_and_up_projs_scales"]
            gus = gus.to_local() if mesh is not None else gus
            self.assertEqual((gup.dtype, gus.dtype), (torch.int8, torch.float8_e8m0fnu))
            for j, e in enumerate(pk):
                self.assertTrue(torch.equal(gup[j], torch.cat([self.hf[(layer, e, "w1")][0], self.hf[(layer, e, "w3")][0]]).view(torch.int8)))
                self.assertTrue(torch.equal(gus[j].view(torch.uint8), torch.cat([self.hf[(layer, e, "w1")][1], self.hf[(layer, e, "w3")][1]]).view(torch.uint8)))
                dnp = native[base + "down_projs_packed"]
                dnp = dnp.to_local() if mesh is not None else dnp
                self.assertTrue(torch.equal(dnp[j], self.hf[(layer, e, "w2")][0].view(torch.int8)))
        self.assertIn("model.layers.0.input_layernorm.weight", native)
        self.assertFalse(any(k.endswith(".scale") for k in native))

    def test_from_hf_builds_both_stacks(self):
        native = self.adapter.from_hf(_hf_state_dict(self.hf, 2), device_mesh=None)
        self._check_from_hf(native, None)
        # Loading the rebuilt dict back reproduces the skeleton's stacks exactly.
        fresh = _Skeleton(self.split)
        missing, unexpected = fresh.load_state_dict({k: v for k, v in native.items() if ".mlp.experts." in k}, strict=False)
        self.assertEqual((missing, unexpected), ([], []))
        for (n, a), (_, b) in zip(fresh.state_dict().items(), self.skeleton.state_dict().items()):
            self.assertTrue(torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e8m0fnu else a,
                                        b.view(torch.uint8) if b.dtype == torch.float8_e8m0fnu else b), n)

    def test_from_hf_rejects_incomplete_and_orphans(self):
        sd = _hf_state_dict(self.hf, 2)
        del sd["layers.1.ffn.experts.3.w2.weight"]
        with self.assertRaises(ValueError):
            self.adapter.from_hf(sd, device_mesh=None)
        sd = _hf_state_dict(self.hf, 2)
        del sd["layers.0.ffn.experts.0.w1.scale"]
        with self.assertRaises(ValueError):
            self.adapter.from_hf(sd, device_mesh=None)


class AdapterOnMesh(Adapter):
    """The same routing with the stacks as Shard(0) DTensors on a one-rank gloo expert mesh."""

    @classmethod
    def setUpClass(cls):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
        cls.mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        for block in self.skeleton.model.layers.values():
            m = block.mlp.experts
            for name, param in list(m.named_parameters(recurse=False)):
                # from_local, not distribute_tensor: gloo cannot scatter int8 or E8M0.
                m.register_parameter(name, nn.Parameter(DTensor.from_local(param.data, self.mesh, [Shard(0)]), requires_grad=param.requires_grad))

    def test_to_hf_placeholders_and_views(self):
        hf_sd = self.adapter.to_hf(self.skeleton.state_dict(), quantization=True, device_mesh=self.mesh)
        self._check_to_hf(hf_sd, self.skeleton)

    def test_from_hf_builds_both_stacks(self):
        native = self.adapter.from_hf(_hf_state_dict(self.hf, 2), device_mesh=self.mesh)
        self._check_from_hf(native, self.mesh)

    def test_from_hf_rejects_incomplete_and_orphans(self):
        sd = _hf_state_dict(self.hf, 2)
        del sd["layers.1.ffn.experts.3.w2.weight"]
        with self.assertRaises(ValueError):
            self.adapter.from_hf(sd, device_mesh=self.mesh)


class Forward(unittest.TestCase):
    """Layer 0 selects experts 1 and 2 of 4; one rank, no mesh."""

    def setUp(self):
        torch.manual_seed(0)
        self.split = ExpertSplit.from_layers({0: [1, 2]}, n_experts=E, ranks=1)
        self.hf = _hf_experts(1, seed=1)
        self.skeleton = _Skeleton(self.split)
        _fill(self.skeleton, self.split, self.hf)
        self.module = self.skeleton.model.layers["0"].mlp.experts
        self.reference = GroupedExperts(_moe_config(), backend=_backend())
        with torch.no_grad():
            self.reference.gate_and_up_projs.copy_(torch.stack([torch.cat([_deq(self.hf[(0, e, "w1")]).T, _deq(self.hf[(0, e, "w3")]).T], -1) for e in range(E)]))
            self.reference.down_projs.copy_(torch.stack([_deq(self.hf[(0, e, "w2")]).T for e in range(E)]))

    def _inputs(self, tokens: int, experts: list[int], masked: int = 1):
        x = torch.randn(tokens, DIM).bfloat16().requires_grad_(True)
        pool = torch.tensor(experts)
        indices = pool[torch.randint(0, len(pool), (tokens, 2))]
        weights = torch.softmax(torch.randn(tokens, 2), -1)
        mask = torch.ones(tokens, dtype=torch.bool)
        mask[:masked] = False
        return x, mask, weights, indices

    def _compare(self, experts: list[int], tokens: int = 24):
        x, mask, weights, indices = self._inputs(tokens, experts)
        x_ref = x.detach().clone().requires_grad_(True)
        ref = self.reference(x_ref, mask, weights, indices)
        out = self.module(x, mask, weights, indices)
        self.assertEqual(out.dtype, ref.dtype)
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
        out.float().sum().backward()
        ref.float().sum().backward()
        torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=2e-2, atol=2e-2)
        return out

    def test_mixed_routing_matches_single_stack(self):
        self._compare([0, 1, 2, 3])
        self.assertIsNotNone(self.module.gate_and_up_projs.grad)
        self.assertIsNotNone(self.module.down_projs.grad)
        self.assertIsNone(self.module.gate_and_up_projs_packed.grad)
        self.assertGreater(float(self.module.gate_and_up_projs.grad.abs().sum()), 0.0)
        # The BF16 stack's gradient equals the reference's gradient on the selected experts.
        torch.testing.assert_close(self.module.gate_and_up_projs.grad.float(), self.reference.gate_and_up_projs.grad[[1, 2]].float(), rtol=2e-2, atol=2e-2)

    def test_only_packed_experts_still_grads_bf16_stack(self):
        self._compare([0, 3])
        self.assertIsNotNone(self.module.gate_and_up_projs.grad)
        self.assertEqual(float(self.module.gate_and_up_projs.grad.abs().sum()), 0.0)

    def test_only_selected_experts(self):
        self._compare([1, 2])

    def test_no_valid_tokens(self):
        x, mask, weights, indices = self._inputs(4, [0, 1, 2, 3], masked=4)
        out = self.module(x, mask, weights, indices)
        self.assertEqual(float(out.detach().float().abs().sum()), 0.0)
        out.float().sum().backward()
        self.assertIsNotNone(self.module.gate_and_up_projs.grad)
        self.assertIsNotNone(x.grad)

    def test_fake_quant_swap_reaches_forward(self):
        from deepseek.qat import fakequant as FQ
        x, mask, weights, indices = self._inputs(16, [1, 2])
        plain = self.module(x, mask, weights, indices)
        with torch.no_grad():
            self.module.gate_and_up_projs.mul_(1.01)  # off the grid: fake-quant must change the output
        state = FQ.FakeQuantState([FQ.Target("ex", self.module, {"gate_and_up_projs": "fp4_gate_up", "down_projs": "fp4_down"})])
        original = self.module.forward
        state.originals["ex"] = original

        def wrapped(*args, **kwargs):
            swapped = {a: FQ.fake_quant(getattr(self.module, a), k) for a, k in state.targets[0].params.items()}
            with FQ._reparametrize_module(self.module, swapped):
                return original(*args, **kwargs)

        self.module.forward = wrapped
        hooked = self.module(x, mask, weights, indices)
        FQ.uninstall(state)
        unhooked = self.module(x, mask, weights, indices)
        self.assertFalse(torch.equal(hooked, unhooked))
        self.assertFalse(torch.equal(plain, unhooked))


class NoSelectedStack(unittest.TestCase):
    def test_layer_without_selection_holds_packed_only(self):
        split = ExpertSplit.from_layers({0: []}, n_experts=E, ranks=1)
        m = SplitExperts(GroupedExperts(_moe_config(), backend=_backend()), split, 0)
        self.assertIsNone(m.gate_and_up_projs)
        self.assertEqual(m.n_selected, 0)
        self.assertEqual([n for n, _ in m.named_parameters()], ["gate_and_up_projs_packed", "gate_and_up_projs_scales", "down_projs_packed", "down_projs_scales"])
        self.assertEqual(sum(1 for p in m.parameters() if p.requires_grad), 0)


@unittest.skipUnless(os.path.isdir(SNAP), "snapshot not present")
class MetaBuild(unittest.TestCase):
    def test_one_layer_meta_build_through_constructor_hook(self):
        from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
        config = DeepseekV4Config.from_pretrained(SNAP, num_hidden_layers=1, num_nextn_predict_layers=0)
        split = ExpertSplit.from_layers({0: [3, 7, 130, 140]}, n_experts=int(config.n_routed_experts), ranks=2)
        with torch.device("meta"):
            model = DeepseekV4ForCausalLM(config, backend=_backend(), expert_split=split)
        experts = model.model.layers["0"].mlp.experts
        self.assertIsInstance(experts, SplitExperts)
        self.assertIsInstance(model.state_dict_adapter, SplitStateDictAdapter)
        self.assertEqual(tuple(experts.gate_and_up_projs.shape), (4, config.hidden_size, 2 * config.moe_intermediate_size))
        self.assertEqual(tuple(experts.gate_and_up_projs_packed.shape), (252, 2 * config.moe_intermediate_size, config.hidden_size // 2))
        self.assertTrue(experts.gate_and_up_projs.is_meta)
        # The pre-shard key listing walks the unsharded stacks in rank order.
        keys = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
        self.assertEqual(len([k for k in keys if ".ffn.experts." in k and k.endswith(".weight")]), 3 * 256)
        with self.assertRaises(ValueError):
            ExpertSplit.from_layers({0: [1, 129], 1: [1, 129]}, n_experts=256, ranks=2).install(model)


if __name__ == "__main__":
    unittest.main()
