"""Endpoint failures, bounded per-request retries, and batch result ownership."""

import io
import http.client
import json
import threading
import unittest
import urllib.error
from unittest.mock import Mock, patch

from dataset.endpoint import Endpoint, Reply
from dataset.io import DatasetError
from dataset.progress import Progress


class EndpointErrors(unittest.TestCase):
    # Create a local-looking endpoint; every test that issues a request mocks transport.
    def endpoint(self, **overrides):
        args = dict(url="http://localhost:8000/v1", model="model", api_key=None, timeout=1, retries=0)
        args.update(overrides)
        return Endpoint(**args)

    # Invalid URL syntax fails before urllib can emit low-level exceptions.
    def test_invalid_urls(self):
        for url in ("localhost:8000/v1", "http://", "http://host:bad", "http://host:65536",
                    "http://[broken", "http://host/with space", "file:///tmp/model", None, 3, []):
            with self.subTest(url=url), self.assertRaisesRegex(DatasetError, "endpoint URL"):
                self.endpoint(url=url)

    # Float parsing accepts these values, but none describes a usable timeout.
    def test_timeout_must_be_finite_and_positive(self):
        for timeout in (float("nan"), float("inf"), float("-inf"), 0, -1, None, "bad", True, []):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(DatasetError, "finite positive"):
                self.endpoint(timeout=timeout)

    # Direct reward arguments reach the same integer contract as CLI-parsed options.
    def test_retries_and_concurrency_require_integers(self):
        for value in (None, "1", 1.5, True, -1, []):
            with self.subTest(value=value), self.assertRaisesRegex(DatasetError, "retries must be an integer"):
                self.endpoint(retries=value)
            with self.subTest(value=value), self.assertRaisesRegex(DatasetError, "concurrency must be an integer"):
                self.endpoint().complete_many_full([], value)

    # Header validation errors must not reveal the API key.
    def test_credential_error_omits_value(self):
        with self.assertRaisesRegex(DatasetError, "credential environment variable") as result:
            self.endpoint(api_key="private-secret\nsecond-line")
        self.assertNotIn("private-secret", str(result.exception))

    # Malformed external response types must not reach regex/string consumers.
    def test_reply_field_types(self):
        endpoint = self.endpoint()
        messages = [{"content": None}, {"content": []}, {"content": "ok", "reasoning": []},
                    {"content": "ok", "reasoning_content": 0}]
        for message in messages:
            with self.subTest(message=message), self.assertRaisesRegex(DatasetError, "must be a string"):
                endpoint._extract({"choices": [{"message": message}]})
        reply = endpoint._extract({"choices": [{"message": {"content": "ok", "reasoning": None}}]})
        self.assertEqual((reply.content, reply.reasoning), ("ok", ""))

    # Malformed replies consume the same bounded retry budget as transport failures.
    def test_malformed_responses_retry_then_succeed(self):
        valid = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        invalid = [b"\xff", b"not JSON", b"{}", b'{"choices": []}',
                   b'{"choices": [{"message": {"content": null}}]}',
                   b'{"choices": [{"message": {"content": "ok", "reasoning": []}}]}']
        for data in invalid:
            with self.subTest(data=data), patch("dataset.endpoint.urllib.request.urlopen", side_effect=[
                io.BytesIO(data), io.BytesIO(valid)
            ]) as call, patch("dataset.endpoint.time.sleep") as sleep:
                self.assertEqual(self.endpoint(retries=2).complete([]), "ok")
                self.assertEqual(call.call_count, 2)
                sleep.assert_called_once_with(1.0)

    # Exhaustion names the malformed field, reports every retry, and never loops indefinitely.
    def test_malformed_response_exhaustion(self):
        for retries in (0, 2):
            lines = []
            with self.subTest(retries=retries), patch("dataset.endpoint.urllib.request.urlopen",
                    side_effect=[io.BytesIO(b"{}") for _ in range(retries + 1)]) as call, \
                 patch("dataset.endpoint.time.sleep") as sleep:
                with self.assertRaisesRegex(DatasetError, f"gave up after {retries + 1} attempts.*message.content"):
                    self.endpoint(retries=retries).complete_full([], request="request 19",
                        progress=Mock(note=lines.append))
                self.assertEqual(call.call_count, retries + 1)
                self.assertEqual(sleep.call_count, retries)
                if retries:
                    self.assertIn("request 19:", "\n".join(lines))
                    self.assertIn("retry attempt 3/3 in 2s", "\n".join(lines))

    # A valid but truncated completion requires changed generation settings, not another attempt.
    def test_explicit_incomplete_generation_is_not_retried(self):
        data = json.dumps({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}).encode()
        with patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(data)) as call, \
             patch("dataset.endpoint.time.sleep") as sleep:
            with self.assertRaisesRegex(DatasetError, "increase --max-tokens"):
                self.endpoint(retries=2).complete_full([], require_stop=True)
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()

    # A malformed request result is retried locally; completed neighbors and order are retained.
    def test_batch_retries_only_failed_request(self):
        counts = {"first": 0, "second": 0, "third": 0}
        lock = threading.Lock()
        completed = []

        # Count actual HTTP attempts by input, independent of thread completion order.
        def respond(request, timeout):
            name = json.loads(request.data)["messages"][0]["content"]
            with lock:
                counts[name] += 1
                attempt = counts[name]
            if name == "second" and attempt == 1:
                return io.BytesIO(b"{}")
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": name}}]}).encode())

        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=respond), patch("dataset.endpoint.time.sleep"):
            replies = self.endpoint(retries=1).complete_many_full(
                [[{"role": "user", "content": name}] for name in counts], 3,
                on_complete=lambda index, reply: completed.append(index))
        self.assertEqual([reply.content for reply in replies], list(counts))
        self.assertEqual(counts, {"first": 1, "second": 2, "third": 1})
        self.assertEqual(sorted(completed), [0, 1, 2])

    # Retry count and backoff remain unchanged for a transient connection failure.
    def test_transient_failure_retries_then_succeeds(self):
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=[
            urllib.error.URLError("refused"), io.BytesIO(payload)
        ]) as call, patch("dataset.endpoint.time.sleep") as sleep:
            self.assertEqual(self.endpoint(retries=1).complete([]), "ok")
        self.assertEqual(call.call_count, 2)
        sleep.assert_called_once_with(1.0)

    # A remote error can echo both URL credentials and the Authorization value.
    def test_http_error_redacts_credentials_without_changing_request(self):
        base = "http://operator:password-secret@localhost:8000/v1?token=query-secret"
        key = "api-key-secret"
        endpoint = self.endpoint(url=base, api_key=key)
        body = f"request {endpoint.url}; Authorization: Bearer {key}; password-secret query-secret".encode()
        error = urllib.error.HTTPError(endpoint.url, 401, "unauthorized", {}, io.BytesIO(body))
        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=error) as call:
            with self.assertRaises(DatasetError) as result:
                endpoint.complete([])
        diagnostic = str(result.exception)
        for secret in ("operator", "password-secret", "query-secret", key, "?token="):
            self.assertNotIn(secret, diagnostic)
        self.assertIn("HTTP 401", diagnostic)
        self.assertIn("http://localhost:8000/v1/chat/completions", diagnostic)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, endpoint.url)
        self.assertEqual(request.get_header("Authorization"), "Bearer " + key)
        self.assertEqual(call.call_count, 1)

    # URLError text also comes from an external boundary and can contain credentials.
    def test_connection_error_redacts_echoed_secret(self):
        key = "private-api-key"
        endpoint = self.endpoint(api_key=key)
        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=urllib.error.URLError("echo " + key)):
            with self.assertRaises(DatasetError) as result:
                endpoint.complete([])
        self.assertNotIn(key, str(result.exception))
        self.assertIn("[redacted]", str(result.exception))

    # Schema failures use the safe URL even without an echoed server message.
    def test_response_error_omits_url_userinfo_and_query(self):
        endpoint = self.endpoint(url="http://operator:password-secret@localhost/v1?token=query-secret")
        with self.assertRaises(DatasetError) as result:
            endpoint._extract({})
        for secret in ("operator", "password-secret", "query-secret", "?token="):
            self.assertNotIn(secret, str(result.exception))

    # A broken error-body stream must not hide the final HTTP status or trigger retries.
    def test_http_error_body_read_failure_is_contextual_and_final(self):
        for failure in (OSError("connection dropped"), http.client.IncompleteRead(b"partial")):
            with self.subTest(failure=type(failure).__name__):
                endpoint = self.endpoint(retries=2)
                body = Mock()
                body.read.side_effect = failure
                error = urllib.error.HTTPError(endpoint.url, 401, "unauthorized", {}, body)
                with patch("dataset.endpoint.urllib.request.urlopen", side_effect=error) as call:
                    with self.assertRaisesRegex(DatasetError, "HTTP 401; cannot read error response"):
                        endpoint.complete([])
                self.assertEqual(call.call_count, 1)


class EndpointProgress(unittest.TestCase):
    # Remote diagnostics can echo decoded credentials, percent escapes, or form-encoded spaces.
    def test_encoded_and_decoded_url_secrets_are_redacted_from_retries_and_final_error(self):
        secrets = ("pa%3Ass%2Fword", "pa:ss/word", "query+secret%2Fone", "query%20secret%2Fone", "query secret/one")
        endpoint = Endpoint("http://operator:pa%3Ass%2Fword@localhost/v1?token=query+secret%2Fone",
                            "model", None, 1, 1)
        lines = []
        error = urllib.error.URLError("echo " + " | ".join(secrets))
        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=error), \
             patch("dataset.endpoint.time.sleep"), self.assertRaises(DatasetError) as caught:
            endpoint.complete([], progress=Progress("dataset chat", emit=lines.append))
        feedback = "\n".join(lines)
        diagnostic = str(caught.exception)
        self.assertIn("retry attempt 2/2", feedback)
        self.assertIn("gave up after 2 attempts", diagnostic)
        for text in (feedback, diagnostic):
            self.assertIn("[redacted]", text)
            for secret in secrets:
                self.assertNotIn(secret, text)

    # Operators need the sanitized reason and next attempt before backoff itself blocks.
    def test_retry_reason_attempt_and_delay_are_visible_before_backoff(self):
        key = "private-api-key"
        endpoint = Endpoint("http://localhost/v1", "model", key, 120, 1)
        lines = []
        progress = Progress("dataset chat", emit=lines.append)
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

        # Inspect output at the sleep boundary rather than after the successful response.
        def backoff(delay):
            self.assertEqual(delay, 1.0)
            self.assertIn("connection error: echo [redacted]", lines[-1])
            self.assertIn("request: ", lines[-1])
            self.assertIn("retry attempt 2/2 in 1s", lines[-1])
            self.assertNotIn(key, "\n".join(lines))

        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=[
            urllib.error.URLError("echo " + key), io.BytesIO(payload)
        ]), patch("dataset.endpoint.time.sleep", side_effect=backoff) as sleep:
            self.assertEqual(endpoint.complete([], progress=progress), "ok")
        sleep.assert_called_once()

    # A slow first request must not hide later completions or reorder the resulting dataset.
    def test_later_completion_is_reported_while_first_waits_and_results_keep_input_order(self):
        endpoint = Endpoint("http://localhost/v1", "model", None, 1, 0)
        first_started = threading.Event()
        release_first = threading.Event()
        completion_reported = threading.Event()
        batch_done = threading.Event()
        results, failures = [], []

        # Release the assertion only after the reporter publishes a measured completion.
        def emit(line):
            if "1/2 requests" in line:
                completion_reported.set()

        # Force the second reply to finish while the first remains explicitly blocked.
        def complete(messages, max_tokens=None, **kwargs):
            if messages == "first":
                first_started.set()
                if not release_first.wait(5):
                    raise TimeoutError("test did not release first request")
            elif not first_started.wait(5):
                raise TimeoutError("first request did not start")
            return Reply(messages, "")

        # Retain failures from the batch thread so they cannot disappear as thread warnings.
        def run():
            try:
                results.extend(endpoint.complete_many_full(
                    ["first", "second"], 2, progress=Progress("dataset chat", emit=emit), label="questions"))
            except BaseException as error:
                failures.append(error)
            finally:
                batch_done.set()

        with patch.object(endpoint, "complete_full", side_effect=complete), patch("dataset.progress.COUNT_SECONDS", 0):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(completion_reported.wait(3), "second completion was hidden behind first request")
                self.assertFalse(batch_done.is_set())
            finally:
                release_first.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results, [Reply("first", ""), Reply("second", "")])

    # Terminal failure must be visible during sibling shutdown and cancel requests still queued.
    def test_terminal_failure_is_reported_before_siblings_finish_and_pending_work_is_cancelled(self):
        key = "private-api-key"
        endpoint = Endpoint("http://localhost/v1", "model", key, 1, 0)
        first_started = threading.Event()
        release = threading.Event()
        failure_reported = threading.Event()
        batch_done = threading.Event()
        lines, started, failures = [], [], []

        # Signal from the diagnostic sink while in-flight calls are still blocked.
        def emit(line):
            lines.append(line)
            if "stopping batch:" in line:
                failure_reported.set()

        # Fail one active request and hold its siblings so queued cancellation is observable.
        def complete(messages, max_tokens=None, **kwargs):
            started.append(messages)
            if messages == 1:
                if not first_started.wait(5):
                    raise TimeoutError("first request did not start")
                raise DatasetError("bad response echo " + key)
            if messages == 0:
                first_started.set()
            # A worker may pick up one queued request before cancellation arrives.
            # Hold it as well, so later queued work cannot race through the test.
            if not release.wait(5):
                raise TimeoutError("test did not release sibling request")
            return Reply(str(messages), "")

        # Capture the propagated batch error separately from the earlier user-facing notice.
        def run():
            try:
                endpoint.complete_many_full(range(8), 2, progress=Progress("dataset chat", emit=emit))
            except BaseException as error:
                failures.append(error)
            finally:
                batch_done.set()

        with patch.object(endpoint, "complete_full", side_effect=complete):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(failure_reported.wait(3), "terminal error was hidden behind an in-flight request")
                self.assertFalse(batch_done.is_set())
                self.assertNotIn(key, "\n".join(lines))
                self.assertIn("bad response echo [redacted]", lines[-1])
                self.assertIn("in-flight request(s)", lines[-1])
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], DatasetError)
        self.assertLessEqual(len(started), 3)


if __name__ == "__main__":
    unittest.main()
