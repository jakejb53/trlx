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

    def test_dtdd_takes_the_definition_and_rejects_a_bare_term(self):
        page = ('<html><head><meta charset="utf-8"></head><body><dl>'
                '<dt id="m.f">m.f(x)<a class="headerlink">¶</a></dt>'
                '<dd><p>Does the thing.</p><p>More.</p></dd>'
                '<dt id="m.g">m.g()</dt></dl></body></html>')
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "doc.html"
            p.write_text(page, encoding="utf-8")
            out = pathlib.Path(d) / "out.txt"
            _load("extract").dtdd(p, out, ["m.f"])
            text = out.read_text(encoding="utf-8")
            with self.assertRaises(AssertionError):
                _load("extract").dtdd(p, out, ["m.g"])   # no <dd> follows
        self.assertTrue(text.startswith("[m.f]\nm.f(x)\n"))
        self.assertNotIn("¶", text)
        self.assertIn("Does the thing.", text)

    def test_clause_removes_only_the_named_subclause(self):
        page = ('<html><body><emu-clause id="sec-a"><h1>A</h1><p>keep one</p>'
                '<emu-clause id="sec-a-x"><h1>A.x</h1><p>drop me</p></emu-clause>'
                '<p>keep two</p></emu-clause></body></html>')
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "spec.html"
            p.write_text(page, encoding="utf-8")
            out = pathlib.Path(d) / "out.txt"
            _load("extract").clause(p, out, ["sec-a,sec-a-x"])
            text = out.read_text(encoding="utf-8")
            with self.assertRaises(AssertionError):
                _load("extract").clause(p, out, ["sec-a,sec-missing"])
        self.assertIn("keep one", text)
        self.assertIn("keep two", text)
        self.assertNotIn("drop me", text)
        self.assertNotIn("\n\n\n", text)


class EditTests(unittest.TestCase):
    def test_derive_widens_to_unique_spans_merges_overlaps_and_round_trips(self):
        edit = _load("edit")
        raw = {"reasoning": "x a x b x a x", "answer": "\n\nkeep. drop this. keep."}
        final = {"reasoning": "x c x b x a x", "answer": "keep. keep."}
        edits = edit.derive_edits(raw, final)
        self.assertEqual(edit.apply_edits(raw, edits), final)
        for e in edits:
            self.assertEqual(raw[e["field"]].count(e["old"]), 1)
        # Two nearby changes whose unique spans overlap must become one edit.
        raw2 = {"reasoning": "p q r s t q r s u", "answer": "z"}
        final2 = {"reasoning": "p Q r S t q r s u", "answer": "z"}
        edits2 = edit.derive_edits(raw2, final2)
        self.assertEqual(edit.apply_edits(raw2, edits2), final2)
        self.assertEqual(len([e for e in edits2 if e["field"] == "reasoning"]), 1)


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


class ReuseTests(unittest.TestCase):
    def test_body_must_match_recorded_sha256(self):
        import hashlib
        reuse = _load("reuse")
        body = "Section text.\n\nMore text.\n"
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        ok = {"role": "tool", "content": f"SOURCE: S\nLOCATION: L\nRETRIEVED: 2026-10-07\nSHA256: {sha}\nREPRESENTATION: R\n\n{body}"}
        header, got = reuse.verified_body(ok)
        self.assertEqual(got, body)
        self.assertTrue(header.startswith("SOURCE: S"))
        inline = dict(ok, content=ok["content"].replace(f"SHA256: {sha}\nREPRESENTATION: R", f"REPRESENTATION: R; SHA256 {sha}"))
        self.assertEqual(reuse.verified_body(inline)[1], body)
        tampered = dict(ok, content=ok["content"].replace("More text.", "Other text."))
        with self.assertRaises(AssertionError):
            reuse.verified_body(tampered)
        unhashed = dict(ok, content=ok["content"].replace(f"SHA256: {sha}\n", ""))
        with self.assertRaises(AssertionError):
            reuse.verified_body(unhashed)
        with self.assertRaises(AssertionError):
            reuse.verified_body({"role": "assistant", "content": ok["content"]})


if __name__ == "__main__":
    unittest.main()
