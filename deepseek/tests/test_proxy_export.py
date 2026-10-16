"""CPU tests for the proxy export path: expert naming and half order, local
expert quantization against the HF matrices, assembly of gathered per-rank
stacks into global expert names with the source dtypes, dense FP8 mapping,
the byte-equal skip against a synthetic snapshot, and verify_export's
``@FILE`` argument expansion."""

import io
import os
import tempfile
import unittest

import torch

from deepseek.qat import export, fakequant as FQ, proxy_export as PE, quant as Q, verify_export
from deepseek.qat.proxy import Source
from deepseek.tests.test_export import _make_snapshot


class Names(unittest.TestCase):
    def test_expert_and_scale_names(self):
        self.assertEqual(PE.expert_name(5, 17, "w2"), "layers.5.ffn.experts.17.w2")
        self.assertEqual(PE.scale_name("layers.5.ffn.experts.17.w2.weight"), "layers.5.ffn.experts.17.w2.scale")
        with self.assertRaises(ValueError):
            PE.scale_name("layers.5.ffn.experts.17.w2.scale")


class LocalExperts(unittest.TestCase):
    """NeMo gate_and_up_projs[e] = cat(w1.T, w3.T); down_projs[e] = w2.T."""

    def setUp(self):
        torch.manual_seed(0)
        self.e, self.dim, self.inter = 3, 64, 32
        self.w1 = [torch.randn(self.inter, self.dim) for _ in range(self.e)]
        self.w3 = [torch.randn(self.inter, self.dim) for _ in range(self.e)]
        self.w2 = [torch.randn(self.dim, self.inter) for _ in range(self.e)]
        self.gate_up = torch.stack([torch.cat([a.T, b.T], dim=-1) for a, b in zip(self.w1, self.w3)])
        self.down = torch.stack([c.T for c in self.w2])

    def test_gate_up_halves_match_hf_quantization(self):
        out = PE.quantize_local_experts(self.gate_up, "fp4_gate_up")
        self.assertEqual(set(out), {"w1", "w3"})
        for which, src in (("w1", self.w1), ("w3", self.w3)):
            codes, scales = out[which]
            self.assertEqual(codes.shape, (self.e, self.inter, self.dim // 2))
            self.assertEqual(scales.shape, (self.e, self.inter, self.dim // Q.FP4_GROUP))
            for i in range(self.e):
                p, s = FQ.quantize_fp4(src[i])
                self.assertTrue(torch.equal(codes[i], p))
                self.assertTrue(torch.equal(scales[i].view(torch.uint8), s.view(torch.uint8)))

    def test_down_matches_hf_quantization(self):
        out = PE.quantize_local_experts(self.down, "fp4_down")
        self.assertEqual(set(out), {"w2"})
        codes, scales = out["w2"]
        for i in range(self.e):
            p, s = FQ.quantize_fp4(self.w2[i])
            self.assertTrue(torch.equal(codes[i], p))
            self.assertTrue(torch.equal(scales[i].view(torch.uint8), s.view(torch.uint8)))

    def test_rejects_non_expert_kind(self):
        with self.assertRaises(ValueError):
            PE.quantize_local_experts(self.down, "fp8")


class Assembly(unittest.TestCase):
    def test_global_expert_index_follows_rank_order(self):
        # Two ranks of two local experts each: rank r's local expert i is global 2r + i.
        per_rank = []
        expected = {}
        for r in range(2):
            codes = torch.randint(0, 256, (2, 8, 16), dtype=torch.uint8)
            scales = torch.randint(100, 140, (2, 8, 1), dtype=torch.uint8)
            per_rank.append((codes, scales))
            for i in range(2):
                expected[2 * r + i] = (codes[i], scales[i])
        reps = PE.assemble_expert_replacements(3, "w1", per_rank)
        self.assertEqual(len(reps), 8)
        for e, (codes, scales) in expected.items():
            w = reps[f"layers.3.ffn.experts.{e}.w1.weight"]
            s = reps[f"layers.3.ffn.experts.{e}.w1.scale"]
            self.assertEqual(w.dtype, torch.uint8)
            self.assertEqual(s.dtype, torch.float8_e8m0fnu)
            self.assertTrue(torch.equal(w, codes))
            self.assertTrue(torch.equal(s.view(torch.uint8), scales))

    def test_dense_maps_to_hf_weight_and_scale(self):
        w = torch.randn(256, 128)
        reps = PE.dense_replacements("model.layers.2.self_attn.wq_b.weight", w)
        codes, scales = Q.quantize_fp8_blocks(w)
        self.assertEqual(set(reps), {"layers.2.attn.wq_b.weight", "layers.2.attn.wq_b.scale"})
        self.assertEqual(reps["layers.2.attn.wq_b.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(reps["layers.2.attn.wq_b.scale"].dtype, torch.float8_e8m0fnu)
        self.assertTrue(torch.equal(reps["layers.2.attn.wq_b.weight"].view(torch.uint8), codes.view(torch.uint8)))
        self.assertTrue(torch.equal(reps["layers.2.attn.wq_b.scale"].view(torch.uint8), scales.view(torch.uint8)))


class SourceComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.src = os.path.join(self.tmp.name, "src")
        os.makedirs(self.src)
        self.tensors = _make_snapshot(self.src)
        self.out = os.path.join(self.tmp.name, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_byte_equal_candidates_are_skipped_including_uint8_for_i8(self):
        changed = torch.randn(128, 128).to(torch.float8_e4m3fn)
        candidates = {
            "layers.0.attn.wq.weight": changed,
            "layers.0.attn.wq.scale": self.tensors["layers.0.attn.wq.scale"].clone(),
            # The packed FP4 export is uint8; the checkpoint stores I8. Same bytes must count as unchanged.
            "layers.0.ffn.experts.0.w1.weight": self.tensors["layers.0.ffn.experts.0.w1.weight"].view(torch.uint8).clone(),
        }
        kept, unchanged = PE.differing_from_source(candidates, Source(self.src))
        self.assertEqual(set(kept), {"layers.0.attn.wq.weight"})
        self.assertEqual(sorted(unchanged), ["layers.0.attn.wq.scale", "layers.0.ffn.experts.0.w1.weight"])
        # The kept set is exactly what the export changes and what --expect @FILE must list.
        export.export(self.src, self.out, kept, log=io.StringIO())
        listing = os.path.join(self.tmp.name, "replaced.txt")
        with open(listing, "w", encoding="utf-8") as f:
            f.write("".join(f"{n}\n" for n in sorted(kept)))
        out = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(out):
            code = verify_export.main([self.src, self.out, "--expect", f"@{listing}"])
        self.assertEqual(code, 0, out.getvalue())

    def test_shape_mismatch_is_not_equal(self):
        candidates = {"layers.0.attn.wq.weight": torch.randn(64, 256).to(torch.float8_e4m3fn)}
        kept, unchanged = PE.differing_from_source(candidates, Source(self.src))
        self.assertEqual(set(kept), {"layers.0.attn.wq.weight"})
        self.assertEqual(unchanged, [])


if __name__ == "__main__":
    unittest.main()
