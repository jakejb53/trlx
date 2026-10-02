"""Stateless authoring through real CLI dispatch, wire serialization, and staged saves."""

import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from dataset import cli


# Saved examples carry edited text, with reasoning presence chosen by the caller.
def example(answer="answer", **extra):
    return {"messages": [{"role": "user", "content": "question"},
                         {"role": "assistant", "content": answer}], **extra}


class AuthoringCLI(unittest.TestCase):
    # Isolate files and credential loading without replacing the command handlers.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)
        self.body = {"endpoint": "http://localhost/v1", "model": "served-model", "user": "question",
                     "sampling": {}, "timeout": 1, "retries": 0}

    # stdout must be a single JSON result; diagnostics are captured separately.
    def invoke(self, command, value, *options, raw=False):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("dataset.cli.env.load"), patch("sys.stdin", io.StringIO(value if raw else json.dumps(value))), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main([command, *options])
        self.assertIn(f"dataset {command}: starting", stderr.getvalue())
        if code == 0:
            self.assertIn(f"dataset {command}: completed; elapsed", stderr.getvalue())
        return code, stdout.getvalue(), stderr.getvalue()

    # Exercise the endpoint serializer, preserving rich Context between system and user.
    def test_context_and_sampling_wire_contract(self):
        context = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "é", "extra": {"x": True}}]
        path = self.root / "context.json"
        path.write_text(json.dumps(context), encoding="utf-8")
        parameters = {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_tokens": 200,
                      "presence_penalty": -0.5, "repetition_penalty": 1.1}
        reply = {"choices": [{"message": {"content": "answer é", "reasoning_content": "thought"}}]}
        with patch.dict(os.environ, {"AUTHORING_TEST_KEY": "private-key"}), \
                patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            code, output, errors = self.invoke("generate", {**self.body, "system": "instruction",
                "sampling": parameters, "api_key": "AUTHORING_TEST_KEY"}, "--context-file", str(path))
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output), {"answer": "answer é", "reasoning": "thought"})
        request = call.call_args.args[0]
        self.assertEqual(json.loads(request.data), {"model": "served-model", "messages": [
            {"role": "system", "content": "instruction"}, *context,
            {"role": "user", "content": "question"}], **parameters})
        self.assertEqual(request.get_header("Authorization"), "Bearer private-key")
        self.assertNotIn("private-key", output + errors)

    # Omission requires neither a Context file nor credentials and sends no sampling defaults.
    def test_optional_inputs_are_absent(self):
        reply = {"choices": [{"message": {"content": "answer"}}]}
        with patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            code, output, errors = self.invoke("generate", self.body)
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output), {"answer": "answer", "reasoning": ""})
        request = call.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(json.loads(request.data), {"model": "served-model", "messages": [
            {"role": "user", "content": "question"}]})

    # File errors and invalid outer shapes fail before any external request, without traceback.
    def test_invalid_context_never_generates(self):
        path = self.root / "context.json"
        for contents in (None, b"\xff", b"{", b"{}", b"[null]", b"[[]]", b"[NaN]",
                         b'[{"role":"tool","content":{"score":1e999}}]'):
            with self.subTest(contents=contents):
                if contents is not None:
                    path.write_bytes(contents)
                with patch("dataset.endpoint.urllib.request.urlopen") as call:
                    code, output, errors = self.invoke("generate", self.body, "--context-file", str(path))
                self.assertEqual(code, 1)
                self.assertEqual(output, "")
                self.assertIn(str(path), errors)
                self.assertNotIn("Traceback", errors)
                call.assert_not_called()

    # Reject unknown fields, inline Context, malformed JSON, and coercion before generation.
    def test_invalid_requests_are_credential_safe(self):
        cases = [[], {**self.body, "context": []}, {**self.body, "surprise": "private-key"},
                 {**self.body, "user": " "}, {**self.body, "api_key": {"secret": "private-key"}},
                 {**self.body, "sampling": {"temperature": "private-key"}},
                 {**self.body, "sampling": {"max_tokens": 0}}, {**self.body, "retries": True},
                 {**self.body, "timeout": 0}]
        for value in cases:
            with self.subTest(value=value), patch("dataset.endpoint.urllib.request.urlopen") as call:
                code, output, errors = self.invoke("generate", value)
                self.assertEqual(code, 1)
                self.assertEqual(output, "")
                self.assertNotIn("private-key", errors)
                self.assertNotIn("Traceback", errors)
                call.assert_not_called()
        for raw in ('{"api_key":"private-key",', '{} {}', '{"timeout":NaN}'):
            code, output, errors = self.invoke("generate", raw, raw=True)
            self.assertEqual(code, 1)
            self.assertEqual(output, "")
            self.assertNotIn("private-key", errors)
            self.assertNotIn("Traceback", errors)

    # A missing variable never turns the supplied name into a literal bearer credential.
    def test_missing_credential_fails_before_request(self):
        with patch.dict(os.environ, {}, clear=True), patch("dataset.endpoint.urllib.request.urlopen") as call:
            code, output, errors = self.invoke("generate", {**self.body, "api_key": "AUTHORING_TEST_KEY"})
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("unset or empty", errors)
        call.assert_not_called()

    # Endpoint failures retain their explanation and redact the resolved secret.
    def test_endpoint_failure_redacts_credentials(self):
        with patch.dict(os.environ, {"AUTHORING_TEST_KEY": "private-key"}), patch(
                "dataset.endpoint.urllib.request.urlopen", side_effect=urllib.error.URLError("private-key refused")):
            code, output, errors = self.invoke("generate", {**self.body, "api_key": "AUTHORING_TEST_KEY"})
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("refused", errors)
        self.assertNotIn("private-key", errors)

    # Saves preserve existing bytes and distinguish omitted, empty, and nonempty reasoning.
    def test_save_shapes_duplicates_and_retry(self):
        path = self.root / "examples.jsonl"
        original = b'{"text": "existing data"}'
        path.write_bytes(original)
        rows = [example(), example("", reasoning="thought"), example(reasoning="thought"), example(reasoning="")]
        body = {"path": str(path), "examples": rows + [rows[0]]}
        code, output, errors = self.invoke("save", body)
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output), {"added": 4, "duplicates": 1})
        saved = path.read_bytes()
        self.assertTrue(saved.startswith(original + b"\n"))
        self.assertEqual([json.loads(line) for line in saved.splitlines()[1:]], rows)
        code, output, errors = self.invoke("save", body)
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output), {"added": 0, "duplicates": 5})
        self.assertEqual(path.read_bytes(), saved)

    # Full request validation precedes publication, including invalid later examples.
    def test_invalid_save_preserves_destination(self):
        path = self.root / "examples.jsonl"
        original = b'{"text":"untouched"}\n'
        path.write_bytes(original)
        for row in (example(reasoning=None), {**example(), "context": []}, {"messages": []}):
            code, output, errors = self.invoke("save", {"path": str(path), "examples": [example(), row]})
            self.assertEqual(code, 1)
            self.assertEqual(output, "")
            self.assertNotIn("Traceback", errors)
            self.assertEqual(path.read_bytes(), original)

    # Failed staged preparation leaves the original dataset available for retry.
    def test_failed_save_preserves_destination(self):
        path = self.root / "examples.jsonl"
        original = b'{"text":"untouched"}\n'
        path.write_bytes(original)
        with patch("dataset.io.os.fsync", side_effect=OSError("disk failure")):
            code, output, errors = self.invoke("save", {"path": str(path), "examples": [example()]})
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("disk failure", errors)
        self.assertEqual(path.read_bytes(), original)


# Support targeted execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
