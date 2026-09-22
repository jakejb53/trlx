"""Waiting diagnostics identify a final straggler without inventing server progress."""

import io
import json
import threading
import unittest
from unittest.mock import Mock, patch

from dataset.endpoint import Endpoint
from dataset.progress import Progress


class EndpointWaiting(unittest.TestCase):
    # Exercise actual request/retry/collector wiring with time advanced only at controlled boundaries.
    def test_final_request_and_retry_are_identified(self):
        clock = Mock(return_value=0.0)
        first_collected = threading.Event()
        lines, waits = [], []
        attempts = {"first": 0, "last": 0}

        # Only the actual batch counter releases the final request's simulated wait.
        def emit(line):
            lines.append(line)
            if "1/2 requests" in line:
                first_collected.set()

        progress = Progress("dataset chat", emit=emit, clock=clock)
        endpoint = Endpoint("https://example.invalid/v1", "model", None, 120, 2)

        # Produce the first answer immediately, then make the last request malformed once.
        def respond(request, timeout):
            name = json.loads(request.data)["messages"][0]["content"]
            attempts[name] += 1
            if name == "last":
                if not first_collected.wait(3):
                    raise AssertionError("first completion was not collected")
                clock.return_value = 101.0 if attempts[name] == 1 else 140.0
                progress.waiting()
                waits.append(lines[-1])
                if attempts[name] == 1:
                    return io.BytesIO(b"{}")
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": name}}]}).encode())

        # A backoff notice is not a new measured completion.
        def backoff(delay):
            clock.return_value = 120.0
            progress.waiting()
            waits.append(lines[-1])

        # Force publication of the first measured count without sleeping in real time.
        def completed(index, reply):
            if index == 0:
                clock.return_value = 1.0

        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=respond), \
             patch("dataset.endpoint.time.sleep", side_effect=backoff):
            replies = endpoint.complete_many_full(
                [[{"role": "user", "content": name}] for name in attempts], 2,
                progress=progress, label="answers", on_complete=completed)
        self.assertEqual([reply.content for reply in replies], ["first", "last"])
        self.assertEqual(attempts, {"first": 1, "last": 2})
        self.assertEqual(len(waits), 3)
        for line in waits:
            self.assertIn("1 completed responses retained in memory", line)
            self.assertIn("0 queued", line)
            self.assertIn("request 2:", line)
            self.assertNotIn("request 1:", line)
            self.assertIn("socket timeout 120s (not a total request deadline)", line)
            self.assertIn("server progress unavailable", line)
        self.assertIn("awaiting endpoint", waits[0])
        self.assertIn("attempt 1/3", waits[0])
        self.assertIn("retry backoff", waits[1])
        self.assertIn("attempt 2/3", waits[1])
        self.assertIn("last measured progress 119.0s ago", waits[1])
        self.assertIn("attempt 2/3, attempt elapsed 20.0s", waits[2])
