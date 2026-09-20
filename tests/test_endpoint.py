"""Expected endpoint failures are actionable without changing retry behavior."""

import io
import http.client
import json
import unittest
import urllib.error
from unittest.mock import Mock, patch

from dataset.endpoint import Endpoint
from dataset.io import DatasetError


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

    # Bad UTF-8/JSON are final response errors rather than transport retries.
    def test_invalid_response_encoding_and_json_are_not_retried(self):
        for data, error in ((b"\xff", "encoding"), (b"not JSON", "not JSON")):
            with self.subTest(data=data), patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(data)) as call:
                with self.assertRaisesRegex(DatasetError, error):
                    self.endpoint(retries=2).complete([])
                self.assertEqual(call.call_count, 1)

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


if __name__ == "__main__":
    unittest.main()
