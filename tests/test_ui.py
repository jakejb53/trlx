"""Authoring API, non-destructive saves, and browser dataset/request contracts."""

import concurrent.futures
import contextlib
import io
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from fastapi.testclient import TestClient

from dataset import cli
from dataset.endpoint import Endpoint, Reply
from dataset.io import DatasetError, append_examples, read_rows
from dataset.ui import create_app, run


# Construct only the user/assistant data contract; callers opt into reasoning explicitly.
def example(answer="answer", **extra):
    return {"messages": [{"role": "user", "content": "prompt"},
                         {"role": "assistant", "content": answer}], **extra}


class AuthoringAPI(unittest.TestCase):
    # TestClient drives ASGI in memory; all dataset writes stay in repository scratch.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name)
        self.client = TestClient(create_app(), raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.body = dict(endpoint="http://localhost/v1", model="model", api_key="private-key",
                         user="prompt", system="instruction", sampling={}, timeout=1.0, retries=0)

    # Static assets are served through FastAPI and do not embed configuration or secrets.
    def test_page_and_assets(self):
        for path, fragment in (("/", "Dataset Studio"), ("/assets/app.js", "localStorage"),
                               ("/assets/style.css", "@media")):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn(fragment, response.text)

    # Exercise real serialization to the endpoint, including provider-specific fields.
    def test_generation_wire_parameters_and_separate_reasoning(self):
        parameters = dict(temperature=0.7, top_p=0.9, top_k=25, max_tokens=456,
                          presence_penalty=-0.5, repetition_penalty=1.1)
        reply = {"choices": [{"message": {"content": "result", "reasoning_content": "thought"}}]}
        with patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            response = self.client.post("/api/generate", json={**self.body, "sampling": parameters})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"answer": "result", "reasoning": "thought"})
        sent = call.call_args.args[0]
        self.assertEqual(json.loads(sent.data), {"model": "model", "messages": [
            {"role": "system", "content": "instruction"}, {"role": "user", "content": "prompt"}], **parameters})
        self.assertEqual(sent.get_header("Authorization"), "Bearer private-key")

    # Disabled sampling controls and an empty system prompt must truly be absent.
    def test_unset_parameters_are_omitted(self):
        reply = {"choices": [{"message": {"content": "result"}}]}
        with patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as call:
            response = self.client.post("/api/generate", json={**self.body, "system": ""})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(call.call_args.args[0].data), {
            "model": "model", "messages": [{"role": "user", "content": "prompt"}]})

    # The response to an invalid body must never echo its credential input.
    def test_validation_redacts_input_and_rejects_invalid_controls(self):
        for sampling in ({"top_k": 2.5}, {"max_tokens": 0}, {"temperature": -1},
                         {"top_p": 1.1}, {"seed": 1}, {"temperature": "private-key"}):
            with self.subTest(sampling=sampling), patch.object(Endpoint, "complete_full") as generate:
                response = self.client.post("/api/generate", json={**self.body, "sampling": sampling})
                self.assertEqual(response.status_code, 422)
                self.assertNotIn("private-key", response.text)
                generate.assert_not_called()

    # The real UI passes a Progress reporter. Retry logging must not replace a
    # transport/schema failure with AttributeError, on recovery or exhaustion.
    def test_retry_progress_recovers_or_preserves_endpoint_failure(self):
        payload = json.dumps({"choices": [{"message": {"content": "recovered", "reasoning": "thought"}}]}).encode()
        for kind in ("transport", "malformed"):
            for recover in (True, False):
                with self.subTest(kind=kind, recover=recover):
                    failures = ([urllib.error.URLError("connection refused private-key") for _ in range(2)]
                                if kind == "transport" else [io.BytesIO(b"{}"), io.BytesIO(b"{}")])
                    attempts = [failures[0], io.BytesIO(payload) if recover else failures[1]]
                    output = io.StringIO()
                    with patch("dataset.endpoint.urllib.request.urlopen", side_effect=attempts) as call, \
                         patch("dataset.endpoint.time.sleep") as sleep, contextlib.redirect_stderr(output):
                        response = self.client.post("/api/generate", json={**self.body, "retries": 1})
                    self.assertEqual(call.call_count, 2)
                    sleep.assert_called_once_with(1.0)
                    self.assertIn("retry attempt 2/2", output.getvalue())
                    self.assertNotIn("private-key", output.getvalue() + response.text)
                    self.assertNotIn("has no attribute", response.text)
                    if recover:
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertEqual(response.json(), {"answer": "recovered", "reasoning": "thought"})
                    else:
                        self.assertEqual(response.status_code, 400, response.text)
                        self.assertIn("gave up after 2 attempts", response.json()["error"])
                        self.assertIn("connection refused" if kind == "transport" else "message.content",
                                      response.json()["error"])

    # Cards are separate requests: an error does not establish state that poisons another.
    def test_failed_output_does_not_poison_next_output(self):
        with patch.object(Endpoint, "complete_full", side_effect=[DatasetError("endpoint failed"), Reply("ok", "reason")]):
            failed = self.client.post("/api/generate", json=self.body)
            success = self.client.post("/api/generate", json={**self.body, "model": "other"})
        self.assertEqual(failed.status_code, 400)
        self.assertIn("endpoint failed", failed.json()["error"])
        self.assertEqual(success.json(), {"answer": "ok", "reasoning": "reason"})

    # Saved rows preserve omission, reasoning-only empty answers, and exact identity on retry.
    def test_all_saved_shapes_and_retry(self):
        path = self.root / "examples.jsonl"
        rows = [example(), example("", reasoning="thought"), example(reasoning="thought")]
        request = {"path": str(path), "examples": rows + [rows[0]]}
        response = self.client.post("/api/save", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"added": 3, "duplicates": 1})
        self.assertEqual(read_rows(path), rows)
        response = self.client.post("/api/save", json=request)
        self.assertEqual(response.json(), {"added": 0, "duplicates": 4})
        self.assertEqual(read_rows(path), rows)

    # Explicit null reasoning and extra messages cannot corrupt the saved row contract.
    def test_invalid_rows_fail_before_writing(self):
        path = self.root / "examples.jsonl"
        for row in (example(reasoning=None), {"messages": []},
                    {"messages": [{"role": "system", "content": "x"}, {"role": "assistant", "content": "y"}]}):
            response = self.client.post("/api/save", json={"path": str(path), "examples": [row]})
            self.assertEqual(response.status_code, 422)
            self.assertFalse(path.exists())

    # Launch delegates HTTP serving without loading CLI secrets or validating a file output.
    def test_cli_launch_and_port_validation(self):
        with patch("dataset.ui.run", return_value=0) as serve, patch("dataset.cli.env.load") as load:
            self.assertEqual(cli.main(["ui", "--host", "127.0.0.1", "--port", "8000"]), 0)
            serve.assert_called_once_with("127.0.0.1", 8000)
            load.assert_not_called()
        for port in (0, 65536):
            with self.assertRaises(DatasetError):
                run("127.0.0.1", port)


class DatasetAppend(unittest.TestCase):
    # No operator datasets or files outside the repository participate in these checks.
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent)
        self.addCleanup(scratch.cleanup)
        self.path = pathlib.Path(scratch.name) / "examples.jsonl"

    # Preserve original serialization, including unrelated valid rows and missing final newline.
    def test_preserves_original_bytes_and_distinguishes_reasoning_presence(self):
        original = b'{ "text" : "other row" }\n' + json.dumps(example()).encode()
        self.path.write_bytes(original)
        result = append_examples(self.path, [example(), example(reasoning=""), example(reasoning="different")])
        self.assertEqual(result, {"added": 2, "duplicates": 1})
        self.assertTrue(self.path.read_bytes().startswith(original + b"\n"))
        self.assertEqual(len(read_rows(self.path)), 4)

    # A malformed preexisting file must not be silently repaired or overwritten by Save.
    def test_malformed_file_is_preserved(self):
        self.path.write_bytes(b'{"messages":')
        with self.assertRaises(DatasetError):
            append_examples(self.path, [example()])
        self.assertEqual(self.path.read_bytes(), b'{"messages":')

    # Preparation failure leaves the old file intact and an identical retry can succeed.
    def test_failed_preparation_preserves_original_and_retry(self):
        original = json.dumps(example()).encode() + b"\n"
        self.path.write_bytes(original)
        with patch("dataset.io.os.fsync", side_effect=OSError("disk failure")), self.assertRaises(DatasetError):
            append_examples(self.path, [example("new")])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(append_examples(self.path, [example("new")]), {"added": 1, "duplicates": 0})

    # Concurrent Save requests serialize their read/check/publication sequence.
    def test_concurrent_saves_do_not_lose_or_duplicate_rows(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda n: append_examples(self.path, [example(str(n)), example("shared")]), range(8)))
        self.assertEqual(sum(result["added"] for result in results), 9)
        self.assertEqual(len(read_rows(self.path)), 9)

    # Save refuses non-file targets and unsupported formats without modifying their contents.
    def test_invalid_destinations(self):
        self.path.mkdir()
        with self.assertRaises(DatasetError):
            append_examples(self.path, [example()])
        with self.assertRaises(DatasetError):
            append_examples(self.path.with_suffix(".json"), [example()])


class BrowserContracts(unittest.TestCase):
    # Execute the shipped module's pure helpers under Node, without a separate JS dependency.
    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser logic checks")
    def test_browser_rows_sampling_and_workspace_roundtrip(self):
        script = r'''
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source = readFileSync("dataset/ui/app.js", "utf8");
const {newCard, identity, exampleFrom, generationBody, validateWorkspace} =
  await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
const card = newCard();
card.endpoint = "http://example/v1";
card.model = "one";
card.api_key = "key";
card.response = {user: "original prompt", reasoning: "edited reasoning", answer: "edited answer"};
const both = exampleFrom(card);
assert.equal(both.messages[0].content, "original prompt");
assert.equal(both.reasoning, "edited reasoning");
card.includeAnswer = false;
assert.equal(exampleFrom(card).messages[1].content, "");
card.includeAnswer = true;
card.includeReasoning = false;
const answerOnly = exampleFrom(card);
assert.equal(Object.hasOwn(answerOnly, "reasoning"), false);
assert.notEqual(identity(answerOnly), identity({...answerOnly, reasoning: ""}));
card.includeAnswer = false;
assert.throws(() => exampleFrom(card));
assert.deepEqual(generationBody(card, "new prompt", "system").sampling, {});
card.sampling.top_k.enabled = true;
card.sampling.top_k.value = "23";
assert.deepEqual(generationBody(card, "new prompt", "system").sampling, {top_k: 23});
card.sampling.top_k.value = "";
assert.throws(() => generationBody(card, "prompt", ""));
card.sampling.top_k.enabled = false;
assert.deepEqual(generationBody(card, "prompt", "").sampling, {});
const workspace = {version: 1, user: "changed prompt", system: "system", destination: "train.jsonl",
  cards: [card, newCard()], pending: [both, answerOnly]};
assert.deepEqual(validateWorkspace(JSON.parse(JSON.stringify(workspace))), workspace);
assert.throws(() => validateWorkspace({...workspace, version: 2}));
assert.throws(() => validateWorkspace({...workspace, pending: [{messages: []}]}));
'''
        result = subprocess.run(["node", "--input-type=module", "-e", script],
                                cwd=pathlib.Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    # Drive the shipped event handlers with an in-memory DOM/storage boundary. This
    # checks asynchronous ownership and quota recovery, not browser layout/rendering.
    @unittest.skipUnless(shutil.which("node"), "Node is needed for browser event checks")
    def test_browser_events_storage_failure_and_independent_requests(self):
        script = r'''
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const source = readFileSync("dataset/ui/app.js", "utf8");
const {start, newCard} = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
// Minimal DOM boundary retains actual application handlers and child ownership.
class Element {
  // Elements track only the browser operations used by this page.
  constructor() { this.children = []; this.events = {}; this.dataset = {}; this.classList = {toggle() {}}; }
  // Store listeners so tests can await the real async handlers.
  addEventListener(name, callback) { this.events[name] = callback; }
  // Child identity matters for in-flight collection snapshots.
  append(...children) { this.children.push(...children); }
  // Replacement is the render boundary for cards and pending examples.
  replaceChildren() { this.children = []; }
  // Accessibility attributes do not affect these state/ownership assertions.
  setAttribute() {}
  // Route the fixed card selectors to their corresponding elements.
  querySelector(selector) { return this.one[selector]; }
  // Return real mutable editor stand-ins for event registration.
  querySelectorAll(selector) { return this.many[selector]; }
}
// Build the same card control groups that the HTML template supplies.
function cardElement() {
  const element = new Element();
  element.one = Object.fromEntries(["h3", ".sampling", ".add", ".card-message", ".card-status", "details"].map(k => [k, new Element()]));
  element.many = {};
  for (const [selector, field, names] of [
    ["[data-setting]", "setting", ["endpoint", "model", "api_key", "timeout", "retries"]],
    ["[data-include]", "include", ["reasoning", "answer"]],
    ["[data-response]", "response", ["reasoning", "answer"]],
  ]) element.many[selector] = names.map(name => { const input = new Element(); input.dataset[field] = name; return input; });
  return element;
}
const elements = Object.fromEntries(["storage-error", "user", "system", "destination", "clear-user", "clear-system",
  "clear-both", "output-count", "generate", "save", "outputs", "pending", "pending-count", "collection-count",
  "save-status", "generation-status"].map(id => ["#" + id, new Element()]));
elements["#output-template"] = {content: {firstElementChild: {cloneNode: cardElement}}};
globalThis.document = {querySelector: key => elements[key], createElement: () => new Element(), createTextNode: text => text};
const cards = [newCard(), newCard()];
cards.forEach((card, index) => { card.endpoint = "http://example/v1"; card.model = "model-" + index; });
cards[0].sampling.temperature = {enabled: true, value: ""};
let saved = JSON.stringify({version: 1, user: "original", system: "instruction", destination: "train.jsonl", cards, pending: []});
let quota = false;
globalThis.localStorage = {getItem: () => saved, setItem: (key, value) => {
  if (quota) throw new Error("quota exceeded"); saved = value;
}};
const requests = [];
let completeSave;
globalThis.fetch = async (path, options) => {
  const body = JSON.parse(options.body);
  requests.push({path, body});
  if (path === "api/save") return await new Promise(resolve => { completeSave = resolve; });
  return {ok: true, json: async () => ({answer: "answer", reasoning: "reason"})};
};
start();
await elements["#generate"].events.click();
assert.equal(requests.length, 1);
assert.equal(requests[0].body.model, "model-1");
assert.match(elements["#outputs"].children[0].one[".card-message"].textContent, /numeric/);
const card = elements["#outputs"].children[1];
elements["#user"].value = "changed prompt";
elements["#user"].events.input({target: elements["#user"]});
card.one[".add"].events.click();
assert.equal(JSON.parse(saved).pending[0].messages[0].content, "original");
// Local storage is now full: explicit save must still reach the backend.
quota = true;
const inFlight = elements["#save"].events.click();
await Promise.resolve();
assert.equal(requests.at(-1).path, "api/save");
assert.equal(requests.at(-1).body.examples.length, 1);
// A newly edited/added example must survive acknowledgment of the older snapshot.
const answer = card.many["[data-response]"].find(input => input.dataset.response === "answer");
answer.value = "another answer";
answer.events.input();
card.one[".add"].events.click();
completeSave({ok: true, json: async () => ({added: 1, duplicates: 0})});
await inFlight;
assert.equal(elements["#pending-count"].textContent, 1);
assert.match(elements["#storage-error"].textContent, /quota/);
// Failed saves retain the remaining example, and a subsequent retry clears it.
let retry = elements["#save"].events.click();
await Promise.resolve();
completeSave({ok: false, status: 400, json: async () => ({error: "disk full"})});
await retry;
assert.equal(elements["#pending-count"].textContent, 1);
assert.match(elements["#save-status"].textContent, /disk full/);
quota = false;
retry = elements["#save"].events.click();
await Promise.resolve();
completeSave({ok: true, json: async () => ({added: 0, duplicates: 1})});
await retry;
assert.equal(JSON.parse(saved).pending.length, 0);
assert.equal(elements["#pending-count"].textContent, 0);
elements["#clear-system"].events.click();
assert.equal(JSON.parse(saved).system, "");
assert.equal(JSON.parse(saved).user, "changed prompt");
elements["#clear-both"].events.click();
assert.equal(JSON.parse(saved).user, "");
'''
        result = subprocess.run(["node", "--input-type=module", "-e", script],
                                cwd=pathlib.Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
