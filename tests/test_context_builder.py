"""Direct Context-array editing through the Python API and real CLI dispatch."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from dataset import cli, context_builder
from dataset.io import DatasetError


# One valid standard function exchange for validation and mutation tests.
def exchange(call_id="call_0001", *, arguments="{}", result="result"):
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "lookup", "arguments": arguments},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


class ContextBuilder(unittest.TestCase):
    # Keep each test's publication and failure artifacts isolated.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)
        self.path = self.root / "context.json"

    # Creation is deterministic and ordinary messages preserve exact content.
    def test_create_and_add_messages(self):
        self.assertEqual(context_builder.create_context(self.path), {"messages": 0})
        self.assertEqual(self.path.read_text(), "[]\n")
        result = context_builder.add_message(
            self.path, {"role": "user", "content": "héllo\n"})
        self.assertEqual(result, {"messages": 1, "inserted": [0]})
        context_builder.add_message(
            self.path, {"role": "assistant", "content": [
                {"type": "text", "text": "rich"}], "provider": {"x": True}}, at=0)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved[0]["provider"], {"x": True})
        self.assertEqual(saved[1]["content"], "héllo\n")
        self.assertTrue(self.path.read_text().endswith("\n"))

    # Existing paths, links, directories, and missing inputs fail without replacement.
    def test_filesystem_constraints(self):
        context_builder.create_context(self.path)
        with self.assertRaisesRegex(DatasetError, "already exists"):
            context_builder.create_context(self.path)
        directory = self.root / "directory"
        directory.mkdir()
        with self.assertRaisesRegex(DatasetError, "regular file"):
            context_builder.read_context(directory)
        link = self.root / "link.json"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(DatasetError, "symlink"):
            context_builder.read_context(link)
        with self.assertRaisesRegex(DatasetError, "no such Context"):
            context_builder.read_context(self.root / "missing.json")

    # Parsing rejects malformed outer shapes, invalid entries, and non-finite numbers.
    def test_context_shape_and_numbers(self):
        cases = ["{}", "[null]", "[[]]", "[NaN]", '[{"score":1e999}]']
        for number, text in enumerate(cases):
            path = self.root / f"bad-{number}.json"
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(DatasetError):
                context_builder.read_context(path)
        context_builder.create_context(self.path)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(DatasetError, "finite"):
            context_builder.add_message(self.path, {"role": "user", "content": float("inf")})
        self.assertEqual(self.path.read_bytes(), before)

    # Tool construction separates represented arguments from supplied result content.
    def test_tool_exchange_and_call_ids(self):
        context_builder.create_context(self.path)
        first = context_builder.add_tool_exchange(
            self.path, "web_search", {"z": "last", "a": 2}, "page\n")
        second = context_builder.add_tool_exchange(
            self.path, "read_file", {"path": "manual.md"}, "manual", call_id="chosen")
        self.assertEqual(first["call_id"], "call_0001")
        self.assertEqual(second["call_id"], "chosen")
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved[0]["tool_calls"][0]["function"]["arguments"], '{"a":2,"z":"last"}')
        self.assertEqual(saved[1]["content"], "page\n")
        self.assertEqual(context_builder.validate_context(saved), {
            "valid": True, "messages": 4, "tool_calls": 2, "tool_results": 2})
        with self.assertRaisesRegex(DatasetError, "duplicate"):
            context_builder.add_tool_exchange(
                self.path, "lookup", {}, "again", call_id="chosen")

    # Standard tool fields reject malformed, duplicate, unknown, late, and missing results.
    def test_standard_tool_validation_failures(self):
        valid_call, valid_result = exchange()
        cases = [
            ([{"role": "user", "tool_calls": valid_call["tool_calls"]}, valid_result], "role='assistant'"),
            ([{"role": "assistant", "tool_calls": []}], "nonempty array"),
            ([{"role": "assistant", "tool_calls": [{"id": "x", "type": "function",
                "function": {"name": "f", "arguments": "[]"}}]},
              {"role": "tool", "tool_call_id": "x", "content": "r"}], "JSON object"),
            ([valid_result], "unknown or non-pending"),
            ([valid_call], "missing tool results"),
            ([valid_call, {"role": "user", "content": "late"}, valid_result], "before results"),
            ([*exchange(), {**valid_result}], "already has a result"),
            ([*exchange(), *exchange()], "duplicate tool call ID"),
        ]
        for value, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(DatasetError, message):
                context_builder.validate_context(value)

    # Multiple calls in one raw assistant message accept results in either declared order.
    def test_multiple_raw_tool_calls(self):
        calls = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "one", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "two", "arguments": "{}"}},
        ]}
        messages = [calls,
                    {"role": "tool", "tool_call_id": "b", "content": "B"},
                    {"role": "tool", "tool_call_id": "a", "content": "A"}]
        self.assertEqual(context_builder.validate_context(messages)["tool_results"], 2)

    # Replacement, removal, and movement validate the complete resulting conversation.
    def test_mutations_and_failed_pair_edits(self):
        self.path.write_text(json.dumps([
            {"role": "user", "content": "first"}, *exchange(),
            {"role": "assistant", "content": "last"},
        ]))
        context_builder.replace_message(self.path, 2, content="changed\n")
        self.assertEqual(json.loads(self.path.read_text())[2]["content"], "changed\n")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(DatasetError, "missing tool results|unknown or non-pending"):
            context_builder.remove_messages(self.path, 1)
        self.assertEqual(self.path.read_bytes(), before)
        moved = context_builder.move_messages(self.path, 1, 2, count=2)
        self.assertEqual(moved["moved"], {"from": 1, "to": 2, "count": 2})
        saved = json.loads(self.path.read_text())
        self.assertEqual([item["role"] for item in saved], ["user", "assistant", "assistant", "tool"])
        context_builder.remove_messages(self.path, 2, count=2)
        self.assertEqual(len(json.loads(self.path.read_text())), 2)

    # Invalid ranges fail before publication and never use Python negative indexing.
    def test_range_boundaries(self):
        self.path.write_text('[{"role":"user","content":"x"}]')
        before = self.path.read_bytes()
        for operation in (
            lambda: context_builder.remove_messages(self.path, -1),
            lambda: context_builder.remove_messages(self.path, 0, 0),
            lambda: context_builder.remove_messages(self.path, 1),
            lambda: context_builder.move_messages(self.path, 0, 2),
        ):
            with self.assertRaises(DatasetError):
                operation()
            self.assertEqual(self.path.read_bytes(), before)

    # Outline omits content by default, escapes explicit previews, and show stays focused.
    def test_outline_and_show(self):
        self.path.write_text(json.dumps([
            {"role": "user", "content": "line\nsecret"}, *exchange(),
        ]))
        outline = context_builder.outline_context(self.path)
        self.assertNotIn("preview", outline[0])
        self.assertNotIn("secret", json.dumps(outline))
        preview = context_builder.outline_context(self.path, 5)
        self.assertEqual(preview[0]["preview"], "line\\n")
        self.assertEqual(preview[1]["tool_calls"], [{"id": "call_0001", "name": "lookup"}])
        self.assertEqual(context_builder.show_messages(self.path, 0),
                         {"role": "user", "content": "line\nsecret"})
        self.assertEqual(len(context_builder.show_messages(self.path, 1, 2)), 2)

    # Staging failure is reported while the exact original Context remains available.
    def test_publication_failure_preserves_original(self):
        context_builder.create_context(self.path)
        original = self.path.read_bytes()
        with patch("pathlib.Path.write_text", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(DatasetError, "disk failure"):
                context_builder.add_message(self.path, {"role": "user", "content": "new"})
        self.assertEqual(self.path.read_bytes(), original)


class ContextCLI(unittest.TestCase):
    # Exercise real parser dispatch while isolating environment loading and files.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)
        self.path = self.root / "context.json"

    # Capture the CLI's separate machine output and operational diagnostics.
    def invoke(self, action, *arguments, stdin=""):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("dataset.cli.env.load"), patch("sys.stdin", io.StringIO(stdin)), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = cli.main(["context", action, *map(str, arguments)])
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue()

    # Build and edit one Context using every content-ingestion route.
    def test_cli_lifecycle(self):
        code, output, errors = self.invoke("create", self.path)
        self.assertEqual((code, json.loads(output)), (0, {"messages": 0}), errors)
        code, output, errors = self.invoke("add", self.path, "--role", "user", stdin="stdin\n")
        self.assertEqual(code, 0, errors)
        content = self.root / "result.txt"
        content.write_text("web result\n")
        code, output, errors = self.invoke(
            "tool", self.path, "--name", "web_search", "--arg", "query=terms",
            "--content-file", content)
        self.assertEqual(code, 0, errors)
        self.assertEqual(json.loads(output)["call_id"], "call_0001")
        code, output, errors = self.invoke(
            "add", self.path, "--role", "assistant", "--text", "final")
        self.assertEqual(code, 0, errors)
        code, output, errors = self.invoke(
            "replace", self.path, "2", "--text", "corrected")
        self.assertEqual(code, 0, errors)
        code, output, errors = self.invoke("validate", self.path)
        self.assertEqual(json.loads(output), {
            "valid": True, "messages": 4, "tool_calls": 1, "tool_results": 1})
        code, output, errors = self.invoke("outline", self.path, "--json")
        self.assertEqual(len(json.loads(output)), 4)
        self.assertNotIn("web result", output)
        code, output, errors = self.invoke("show", self.path, "1", "--count", "2")
        self.assertEqual([item["role"] for item in json.loads(output)], ["assistant", "tool"])

    # Raw messages and typed argument files retain provider fields and JSON types.
    def test_raw_message_and_arguments_file(self):
        self.invoke("create", self.path)
        message = self.root / "message.json"
        message.write_text(json.dumps({"role": "user", "content": [{"type": "text", "text": "x"}],
                                       "extra": {"enabled": True}}))
        code, _, errors = self.invoke("add", self.path, "--message-file", message)
        self.assertEqual(code, 0, errors)
        arguments = self.root / "arguments.json"
        arguments.write_text(json.dumps({"limit": 3, "filters": ["a"]}))
        code, _, errors = self.invoke(
            "tool", self.path, "--name", "query", "--arguments-file", arguments,
            "--text", "rows")
        self.assertEqual(code, 0, errors)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved[0]["extra"], {"enabled": True})
        self.assertEqual(saved[1]["tool_calls"][0]["function"]["arguments"],
                         '{"filters":["a"],"limit":3}')

    # Bad CLI inputs are concise expected failures and preserve the current file.
    def test_cli_failures_preserve_context(self):
        self.invoke("create", self.path)
        original = self.path.read_bytes()
        cases = [
            ("add", self.path, "--role", "user", "--text", "x", "--content-file", "y"),
            ("tool", self.path, "--name", "x", "--arg", "broken", "--text", "r"),
            ("tool", self.path, "--name", "x", "--arg", "a=1", "--arg", "a=2", "--text", "r"),
            ("remove", self.path, "-1"),
        ]
        for case in cases:
            with self.subTest(case=case):
                code, output, errors = self.invoke(case[0], *case[1:])
                self.assertNotEqual(code, 0)
                self.assertEqual(output, "")
                self.assertNotIn("Traceback", errors)
                self.assertEqual(self.path.read_bytes(), original)

    # A builder-created tool history travels unchanged through existing generation.
    def test_generated_context_wire_compatibility(self):
        self.invoke("create", self.path)
        self.invoke("tool", self.path, "--name", "lookup", "--arg", "q=fact", "--text", "evidence")
        body = {"endpoint": "http://localhost/v1", "model": "served", "user": "Apply it.",
                "sampling": {}, "timeout": 1, "retries": 0}
        reply = {"choices": [{"message": {"content": "answer", "reasoning_content": "thought"}}]}
        with patch("dataset.endpoint.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("dataset.cli.env.load"), patch("sys.stdin", io.StringIO(json.dumps(body))), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = cli.main(["generate", "--context-file", str(self.path)])
        self.assertEqual(code, 0, stderr.getvalue())
        request = json.loads(call.call_args.args[0].data)
        self.assertEqual(request["messages"], [*json.loads(self.path.read_text()),
                                               {"role": "user", "content": "Apply it."}])


# Support focused execution as well as unittest discovery.
if __name__ == "__main__":
    unittest.main()
