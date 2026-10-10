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


# A complete minimal dataset-authoring.toml; endpoints point at a closed port and
# every test replaces the network or subprocess call before it would be made.
_CONFIG = """destination = "{dest}"
max_full_sequence_tokens = 4096
[generation]
endpoint = "http://127.0.0.1:9/v1"
model = "test-model"
api_key = ""
system = ""
timeout = 1
retries = 0
[generation.sampling]
max_tokens = 16
[probes]
server_max_model_len = 1000
[probes.tokenization]
url = "http://127.0.0.1:9/tokenize"
detokenize_url = "http://127.0.0.1:9/detokenize"
reasoning_field = "reasoning"
add_generation_prompt = false
continue_final_message = false
add_special_tokens = false
reasoning_open = "<think>\\n"
reasoning_close = "\\n</think>\\n\\n"
assistant_end = "<|im_end|>\\n"
[probes.scoring]
url = "http://127.0.0.1:9/v1/completions"
echo = true
max_tokens = 0
logprobs = 0
add_special_tokens = false
"""


def _write_config(d, dest="unused.jsonl"):
    p = pathlib.Path(d) / "cfg.toml"
    p.write_text(_CONFIG.format(dest=dest), encoding="utf-8")
    return str(p)


class _Stop(Exception):
    """Raised by a replaced call once the value under test has been captured."""


class ConfigReaderTests(unittest.TestCase):
    def test_readers_strip_only_surrounding_newlines(self):
        cfg = _load("config")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "t.txt"
            p.write_text("\n\n Hello\nworld \n\n", encoding="utf-8")
            self.assertEqual(cfg.read_prompt(p), " Hello\nworld ")
            self.assertEqual(cfg.read_field(p), " Hello\nworld ")


class PromptReadingTests(unittest.TestCase):
    """Each tool must send the stripped prompt (and count_score the stripped fields)."""

    def _files(self, d):
        prompt = pathlib.Path(d) / "prompt.txt"
        prompt.write_text("\nWhat now?\n\n", encoding="utf-8")
        return prompt

    def test_gen_request_uses_stripped_prompt(self):
        import json
        import sys
        import types
        from unittest import mock
        gen = _load("gen")
        with tempfile.TemporaryDirectory() as d:
            prompt = self._files(d)
            argv = ["gen.py", "--prompt-file", str(prompt), "--out-dir", d, "--label", "t",
                    "--config", _write_config(d)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(gen.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)):
                self.assertEqual(gen.main(), 0)
            request = json.loads((pathlib.Path(d) / "t.request.json").read_text(encoding="utf-8"))
        self.assertEqual(request["user"], "What now?")

    def test_render_check_sends_stripped_prompt(self):
        import sys
        from unittest import mock
        rc = _load("render_check")
        sent = {}

        def fake_post(url, body, timeout=120):
            sent.update(body)
            raise _Stop

        with tempfile.TemporaryDirectory() as d:
            prompt = self._files(d)
            ctx = pathlib.Path(d) / "ctx.json"
            ctx.write_text("[]", encoding="utf-8")
            argv = ["render_check.py", str(ctx), str(prompt), "--config", _write_config(d)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(rc, "post", fake_post):
                with self.assertRaises(_Stop):
                    rc.main()
        self.assertEqual(sent["messages"][-1], {"role": "user", "content": "What now?"})

    def test_count_score_renders_stripped_prompt_and_fields(self):
        import sys
        from unittest import mock
        cs = _load("count_score")
        seen = {}

        def fake_render(cfg, prompt, reasoning, answer):
            seen.update(prompt=prompt, reasoning=reasoning, answer=answer)
            raise _Stop

        with tempfile.TemporaryDirectory() as d:
            prompt = self._files(d)
            reasoning = pathlib.Path(d) / "r.txt"
            reasoning.write_text("Think.\n", encoding="utf-8")
            answer = pathlib.Path(d) / "a.txt"
            answer.write_text("\n\nAnswer.\n", encoding="utf-8")
            argv = ["count_score.py", str(prompt), str(reasoning), str(answer), "--config", _write_config(d)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(cs, "render", fake_render):
                with self.assertRaises(_Stop):
                    cs.main()
        self.assertEqual(seen, {"prompt": "What now?", "reasoning": "Think.", "answer": "Answer."})


class MakeReviewPromptTests(unittest.TestCase):
    RAW = {"reasoning": "Line one.\nLine two.\nLine three.\n",
           "answer": "Alpha.\nB\nC\nD\nE\nF\nUntouched closing sentence.\n"}

    def _run(self, d, *extra, prompt=True):
        import json
        import subprocess
        import sys
        raw = pathlib.Path(d) / "raw.json"
        raw.write_text(json.dumps(self.RAW), encoding="utf-8")
        ed = pathlib.Path(d) / "edited"
        ed.mkdir(exist_ok=True)
        (ed / "reasoning.txt").write_text("Line one.\nLine two.\nLine three.\n", encoding="utf-8")
        # Only the first line changes, so the closing sentence lies outside the
        # diff's two context lines and must come from the complete answer section.
        (ed / "answer.txt").write_text(self.RAW["answer"].replace("Alpha.", "Alpha fixed."), encoding="utf-8")
        just = pathlib.Path(d) / "just.txt"
        just.write_text("1. Fixed Alpha.\n", encoding="utf-8")
        pfile = pathlib.Path(d) / "prompt.txt"
        pfile.write_text("\n\nSaved prompt text.\n\n", encoding="utf-8")
        out = pathlib.Path(d) / "out.prompt"
        args = [sys.executable, str(ROOT / "make_review_prompt.py"), str(raw), str(ed), str(just), str(out), *extra]
        if prompt:
            args += ["--prompt-file", str(pfile)]
        proc = subprocess.run(args, capture_output=True, text=True)
        return proc, out

    def test_packet_contains_prompt_and_complete_answer_with_and_without_cuts(self):
        for cuts in ([], ["--cuts"]):
            with tempfile.TemporaryDirectory() as d:
                prior = pathlib.Path(d) / "prior.txt"
                prior.write_text("PRIOR TEXT", encoding="utf-8")
                proc, out = self._run(d, "--prior", str(prior), *cuts)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                p = out.read_text(encoding="utf-8")
            self.assertIn("SAVED USER PROMPT (complete)\n\nSaved prompt text.\n\nCOMPLETE UNIFIED DIFF", p)
            answer_section = p.split("EDITED ANSWER (complete)\n\n", 1)[1]
            self.assertTrue(answer_section.startswith("Alpha fixed.\nB\nC\nD\nE\nF\nUntouched closing sentence.\n"))
            self.assertIn("against the saved user prompt", p)
            self.assertEqual("RULES FOR SIZE CUTS" in p, bool(cuts))
            order = [p.index(s) for s in ("EDITING CONTRACT", "PRIOR RULING AND RESPONSE",
                                          "SAVED USER PROMPT (complete)", "COMPLETE UNIFIED DIFF: REASONING")]
            self.assertEqual(order, sorted(order))

    def test_missing_prompt_file_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            proc, out = self._run(d, prompt=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("--prompt-file", proc.stderr)
            self.assertFalse(out.exists())


class SaveTests(unittest.TestCase):
    """save.py against the real `dataset save` command and a temporary destination."""

    def _save(self, d, prompt, reasoning, answer):
        import subprocess
        import sys
        base = pathlib.Path(d)
        (base / "p.txt").write_text(prompt, encoding="utf-8")
        (base / "a.txt").write_text(answer, encoding="utf-8")
        r = "-"
        if reasoning is not None:
            (base / "r.txt").write_text(reasoning, encoding="utf-8")
            r = str(base / "r.txt")
        cfg = _write_config(d, dest=str(base / "dest.jsonl"))
        return subprocess.run([sys.executable, str(ROOT / "save.py"), str(base / "p.txt"), r, str(base / "a.txt"),
                               "--config", cfg], capture_output=True, text=True, cwd=ROOT.parent)

    def _seed(self, d):
        dest = pathlib.Path(d) / "dest.jsonl"
        dest.write_text('{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}\n',
                        encoding="utf-8")
        return dest, dest.read_bytes()

    def test_new_row_saved_verified_and_duplicate_reported(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            dest, before = self._seed(d)
            proc = self._save(d, "\nAsk.\n", "Think.\n", "Answer.\n")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            after = dest.read_bytes()
            self.assertEqual(after[:len(before)], before)
            rows = after.decode("utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[1]),
                             {"messages": [{"role": "user", "content": "Ask."},
                                           {"role": "assistant", "content": "Answer."}],
                              "reasoning": "Think."})
            dup = self._save(d, "\nAsk.\n", "Think.\n", "Answer.\n")
            self.assertEqual(dup.returncode, 3, dup.stdout + dup.stderr)
            self.assertEqual(dest.read_bytes(), after)

    def test_answer_only_row_has_no_reasoning_key(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            dest, _ = self._seed(d)
            proc = self._save(d, "Ask.", None, "Only an answer.")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            row = json.loads(dest.read_bytes().decode("utf-8").splitlines()[-1])
        self.assertNotIn("reasoning", row)
        self.assertEqual(row["messages"][1]["content"], "Only an answer.")

    def test_leading_whitespace_answer_rejected_before_saving(self):
        with tempfile.TemporaryDirectory() as d:
            dest, before = self._seed(d)
            proc = self._save(d, "Ask.", "Think.", " Bad.")
            self.assertEqual(proc.returncode, 1)
            self.assertIn("begins with whitespace", proc.stderr)
            self.assertEqual(dest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
