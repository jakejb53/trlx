"""Offline regression tests for the authoring scripts' silent-failure points."""
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent / "authoring"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExtractTests(unittest.TestCase):
    def test_rfchtml_keeps_text_after_removed_page_header(self):
        # Mirrors the RFC Editor htmlized layout: a page footer, a page break,
        # a page header, then body text that is the tail of the header span.
        page = (
            '<html><body><pre>'
            '<span class="h3"><a class="selflink" id="section-2.2" href="#section-2.2">2.2</a>.  Response</span>\n'
            '   First paragraph of the section.\n'
            '<span class="grey">Author             Standards Track            [Page 5]</span>\n'
            '</pre><hr class="noprint"/><pre class="newpage"><span id="page-6" class="invisible"> </span>\n'
            '<span class="grey"><a href="./rfc0">RFC 0</a>   Title   June 2013</span>\n\n'
            '   Text after the page header must survive.\n'
            '<span class="h3"><a class="selflink" id="section-2.3" href="#section-2.3">2.3</a>.  Next</span>\n'
            '   Next section.\n'
            '</pre></body></html>'
        )
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "rfc.html"
            p.write_text(page, encoding="utf-8")
            sections = _load("extract").rfchtml_sections(p)
        self.assertIn("First paragraph", sections["section-2.2"])
        self.assertIn("Text after the page header must survive.", sections["section-2.2"])
        self.assertNotIn("Standards Track", sections["section-2.2"])
        self.assertNotIn("June 2013", sections["section-2.2"])
        self.assertIn("Next section.", sections["section-2.3"])

    def test_md_headings_found_past_one_line_fence(self):
        text = (
            "### Alpha\n\nbody a\n\n"
            "```https://example.test/path?x=1```\n\n"
            "### Beta\n\nbody b\n\n"
            "```\nfenced\n### Not A Heading\n```\n\n"
            "### Gamma\n\nbody c\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "doc.md"
            p.write_text(text, encoding="utf-8")
            sections = _load("extract").md_sections(p)
        self.assertEqual(set(sections), {"Alpha", "Beta", "Gamma"})
        self.assertTrue(sections["Beta"].startswith("### Beta"))
        self.assertIn("body b", sections["Beta"])
        self.assertNotIn("body c", sections["Beta"])


class EditTests(unittest.TestCase):
    def test_duplicate_span_is_rejected_and_unique_span_is_applied(self):
        edit = _load("edit")
        raw = {"reasoning": "a b a", "answer": "\n\nx y"}
        with self.assertRaises(AssertionError):
            edit.apply_edits(raw, [{"field": "reasoning", "old": "a", "new": "c"}])
        out = edit.apply_edits(raw, [{"field": "reasoning", "old": "b", "new": "c"}])
        self.assertEqual(out["reasoning"], "a c a")
        self.assertEqual(out["answer"], "x y")
        self.assertEqual(raw["reasoning"], "a b a")


class CountScoreTests(unittest.TestCase):
    def test_spans_match_template_stripped_fields(self):
        cs = _load("count_score")
        r_open, r_close, a_end = "<think>\n", "\n</think>\n\n", "<|im_end|>\n"
        reasoning = "plan the answer\n"          # template strips the trailing newline
        answer = "\n\n## Answer\n\nfinal"          # template strips the leading newlines
        rendered = ("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
                    + r_open + "plan the answer" + r_close + "## Answer\n\nfinal" + a_end)
        sp = cs.spans(rendered, reasoning, answer, r_open, r_close, a_end)
        rs, re_ = sp["reasoning"]
        as_, ae = sp["answer"]
        self.assertEqual(rendered[rs:re_], "plan the answer")
        self.assertEqual(rendered[as_:ae], "## Answer\n\nfinal")
        with self.assertRaises(AssertionError):
            cs.spans(rendered, "not present", answer, r_open, r_close, a_end)


if __name__ == "__main__":
    unittest.main()
