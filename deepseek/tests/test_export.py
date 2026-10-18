"""Tests for the layout-preserving exporter on synthetic two-shard checkpoints.

Covers: verbatim copy is byte-identical; a replacement changes only its own
payload; dtype, shape, byte-count, and unknown-name rejection; packed FP4
uint8 accepted for I8 entries; refusal without force; staging cleanup.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import torch
from safetensors.torch import save_file

from deepseek.qat import export, verify_export


def _make_snapshot(root: str) -> dict[str, torch.Tensor]:
    """Two shards with mixed dtypes, an index, and a non-weight file."""
    t = {
        "embed.weight": torch.randn(16, 8).bfloat16(),
        "layers.0.attn.wq.weight": torch.randn(128, 128).to(torch.float8_e4m3fn),
        "layers.0.attn.wq.scale": torch.ones(1, 1, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        "layers.0.ffn.experts.0.w1.weight": torch.randint(-128, 127, (4, 16), dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.full((4, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        "layers.0.hc": torch.randn(3),
    }
    shard_a = {k: t[k] for k in ["embed.weight", "layers.0.attn.wq.weight", "layers.0.attn.wq.scale"]}
    shard_b = {k: t[k] for k in t if k not in shard_a}
    save_file(shard_a, os.path.join(root, "model-00001-of-00002.safetensors"))
    save_file(shard_b, os.path.join(root, "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard_a}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard_b})
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        f.write('{"a": 1}')
    os.makedirs(os.path.join(root, "encoding"))
    with open(os.path.join(root, "encoding", "x.py"), "w", encoding="utf-8") as f:
        f.write("pass\n")
    return t


def _verify(source: str, other: str, expect=None) -> tuple[int, str]:
    out = io.StringIO()
    argv = [source, other]
    if expect is not None:
        argv += ["--expect", *expect]
    with redirect_stdout(out):
        code = verify_export.main(argv)
    return code, out.getvalue()


class Exporter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.src = os.path.join(self.tmp.name, "src")
        os.makedirs(self.src)
        self.tensors = _make_snapshot(self.src)
        self.out = os.path.join(self.tmp.name, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_zero_replacements_is_byte_identical(self):
        export.export(self.src, self.out, {}, log=io.StringIO())
        code, text = _verify(self.src, self.out)
        self.assertEqual(code, 0, text)
        self.assertFalse(os.path.exists(self.out + ".staging"))

    def test_replacement_changes_only_its_payload(self):
        new = torch.randn(128, 128).to(torch.float8_e4m3fn)
        result = export.export(self.src, self.out, {"layers.0.attn.wq.weight": new}, log=io.StringIO())
        self.assertEqual(result["model-00001-of-00002.safetensors"], ["layers.0.attn.wq.weight"])
        code, text = _verify(self.src, self.out, expect=["layers.0.attn.wq.weight"])
        self.assertEqual(code, 0, text)
        # The written payload is the replacement, and the other shard is untouched.
        from safetensors.torch import load_file
        got = load_file(os.path.join(self.out, "model-00001-of-00002.safetensors"))
        self.assertTrue(torch.equal(got["layers.0.attn.wq.weight"].view(torch.uint8), new.view(torch.uint8)))
        self.assertEqual(
            verify_export.file_digest(os.path.join(self.src, "model-00002-of-00002.safetensors")),
            verify_export.file_digest(os.path.join(self.out, "model-00002-of-00002.safetensors")),
        )

    def test_unchanged_shards_are_hard_links_and_other_files_are_copies(self):
        # Shard 1 carries the replacement and is a new file; shard 2 shares the
        # source's inode (same filesystem in the test); non-weight files are
        # copies so an edit inside the export cannot reach the source.
        new = torch.randn(128, 128).to(torch.float8_e4m3fn)
        export.export(self.src, self.out, {"layers.0.attn.wq.weight": new}, log=io.StringIO())
        same = lambda rel: os.stat(os.path.join(self.src, rel)).st_ino == os.stat(os.path.join(self.out, rel)).st_ino
        self.assertFalse(same("model-00001-of-00002.safetensors"))
        self.assertTrue(same("model-00002-of-00002.safetensors"))
        self.assertFalse(same("config.json"))
        self.assertFalse(same(os.path.join("encoding", "x.py")))
        self.assertFalse(same("model.safetensors.index.json"))
        # Replacing the export removes links only; the source is intact.
        export.export(self.src, self.out, {}, force=True, log=io.StringIO())
        code, text = _verify(self.src, self.out)
        self.assertEqual(code, 0, text)

    def test_packed_fp4_uint8_accepted_for_i8_entry(self):
        packed = torch.randint(0, 256, (4, 16), dtype=torch.uint8)
        export.export(self.src, self.out, {"layers.0.ffn.experts.0.w1.weight": packed}, log=io.StringIO())
        code, text = _verify(self.src, self.out, expect=["layers.0.ffn.experts.0.w1.weight"])
        self.assertEqual(code, 0, text)

    def test_rejections_write_nothing(self):
        cases = {
            "dtype": {"layers.0.attn.wq.weight": torch.randn(128, 128).bfloat16()},
            "shape": {"layers.0.attn.wq.weight": torch.randn(64, 256).to(torch.float8_e4m3fn)},
            "unknown": {"nope.weight": torch.zeros(1)},
            "scale dtype": {"layers.0.attn.wq.scale": torch.ones(1, 1)},
        }
        for label, rep in cases.items():
            with self.subTest(label):
                with self.assertRaises(export.ExportError):
                    export.export(self.src, self.out, rep, log=io.StringIO())
                self.assertFalse(os.path.lexists(self.out))
                self.assertFalse(os.path.lexists(self.out + ".staging"))

    def test_existing_output_requires_force(self):
        export.export(self.src, self.out, {}, log=io.StringIO())
        with self.assertRaises(export.ExportError):
            export.export(self.src, self.out, {}, log=io.StringIO())
        export.export(self.src, self.out, {}, force=True, log=io.StringIO())
        self.assertEqual(_verify(self.src, self.out)[0], 0)

    def test_output_inside_source_rejected(self):
        with self.assertRaises(export.ExportError):
            export.export(self.src, os.path.join(self.src, "out"), {}, log=io.StringIO())

    def test_verify_reports_non_weight_and_header_differences(self):
        export.export(self.src, self.out, {}, log=io.StringIO())
        with open(os.path.join(self.out, "config.json"), "w", encoding="utf-8") as f:
            f.write('{"a": 2}')
        code, text = _verify(self.src, self.out, expect=[])
        self.assertEqual(code, 1)
        self.assertIn("config.json: differs", text)


if __name__ == "__main__":
    unittest.main()
