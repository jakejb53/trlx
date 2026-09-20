"""Built-in rewards on fixed completions, and the [rewards] resolver."""

import contextlib
import http.server
import io
import json
import pathlib
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from dataset.progress import Progress
from trlx import TrlxError, rewards
from trlx.config import RewardEntry

WHERE = "[rewards].funcs[0]"

CONVERSATIONAL = [{"role": "assistant", "content": "The answer is 42."}]


class ReferenceMatchTest(unittest.TestCase):
    def test_modes(self):
        completions = ["42", "the answer is 42.", "forty-two", CONVERSATIONAL]
        answers = ["42", "42", "42", "the answer is 42"]
        equals = rewards.reference_match(WHERE, {"column": "answer", "mode": "equals"})
        self.assertEqual(equals(completions, answer=answers), [1.0, 0.0, 0.0, 1.0])
        contains = rewards.reference_match(WHERE, {"column": "answer", "mode": "contains"})
        self.assertEqual(contains(completions, answer=answers), [1.0, 1.0, 0.0, 1.0])
        fuzzy = rewards.reference_match(WHERE, {"column": "answer", "mode": "fuzzy", "threshold": 0.8})
        self.assertEqual(fuzzy(["the answer is 41"], answer=["the answer is 42"]), [1.0])

    def test_args_checked(self):
        with self.assertRaises(TrlxError):
            rewards.reference_match(WHERE, {"column": "a"})
        with self.assertRaises(TrlxError):
            rewards.reference_match(WHERE, {"column": "a", "mode": "fuzzy"})
        with self.assertRaises(TrlxError):
            rewards.reference_match(WHERE, {"column": "a", "mode": "equals", "typo": 1})

    def test_missing_column_named(self):
        fn = rewards.reference_match(WHERE, {"column": "answer", "mode": "equals"})
        with self.assertRaisesRegex(TrlxError, "no column 'answer'"):
            fn(["x"], other=["y"])


class RegexTest(unittest.TestCase):
    # Invalid config argument types fail during reward construction with domain errors.
    def test_invalid_argument_types_are_contextual_errors(self):
        for args in ({"pattern": 1}, {"pattern": "(.)", "group": "one", "column": "a"},
                     {"pattern": "(.)", "group": 1, "column": []}):
            with self.subTest(args=args), self.assertRaises(TrlxError):
                rewards.regex(WHERE, args)

    def test_match_and_group(self):
        plain = rewards.regex(WHERE, {"pattern": r"\d+"})
        self.assertEqual(plain(["abc 12", "abc"]), [1.0, 0.0])
        grouped = rewards.regex(WHERE, {"pattern": r"answer: (\d+)", "group": 1, "column": "answer"})
        self.assertEqual(grouped(["answer: 7", "answer: 8", "none"], answer=[7, 7, 7]), [1.0, 0.0, 0.0])

    def test_bad_args(self):
        with self.assertRaises(TrlxError):
            rewards.regex(WHERE, {"pattern": "("})
        with self.assertRaises(TrlxError):
            rewards.regex(WHERE, {"pattern": "x", "group": 1})
        with self.assertRaises(TrlxError):
            rewards.regex(WHERE, {"pattern": "x", "group": 1, "column": "a"})


class PhrasesTest(unittest.TestCase):
    # Phrase lists reject non-text values before any scoring begins.
    def test_invalid_phrase_types_are_contextual_errors(self):
        for values in ("hello", [1], None):
            with self.subTest(values=values), self.assertRaisesRegex(TrlxError, "required.*list of strings"):
                rewards.phrases(WHERE, {"required": values})

    def test_scores(self):
        fn = rewards.phrases(WHERE, {"required": ["Hello", "world"], "forbidden": ["oops"]})
        self.assertEqual(fn(["hello WORLD", "hello oops", "nothing"]), [1.0, -0.5, 0.0])
        with self.assertRaises(TrlxError):
            rewards.phrases(WHERE, {})


class JsonValidTest(unittest.TestCase):
    def test_scores(self):
        fn = rewards.json_valid(WHERE, {})
        self.assertEqual(fn(['{"a": 1}', " [1, 2] ", "{a: 1}", 'text {"a": 1}']), [1.0, 1.0, 0.0, 0.0])
        keyed = rewards.json_valid(WHERE, {"keys": ["a", "b"]})
        self.assertEqual(keyed(['{"a": 1, "b": 2}', '{"a": 1}', "[1]"]), [1.0, 0.0, 0.0])


class LengthWindowTest(unittest.TestCase):
    def test_words(self):
        fn = rewards.length_window(WHERE, {"unit": "words", "low": 2, "high": 4})
        self.assertEqual(fn(["a b c", "a", "a b c d e", "a b c d e f g"]), [1.0, 0.5, 0.5, 0.0])

    def test_args(self):
        with self.assertRaises(TrlxError):
            rewards.length_window(WHERE, {"unit": "words", "low": 4, "high": 2})
        with self.assertRaises(TrlxError):
            rewards.length_window(WHERE, {"unit": "tokens", "low": 1, "high": 2})
        with self.assertRaises(TrlxError):
            rewards.length_window(WHERE, {"unit": "words", "low": 1, "high": 2, "tokenizer": "x"})


# Replies with a score derived from the request so the test can check that
# prompt and completion reached the judge; one canned reply has no number.
class _Judge(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][1]["content"]
        text = "no score here" if "silent" in user else f"Score: {len(user)}"
        reply = {"choices": [{"message": {"content": text}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class LlmJudgeTest(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Judge)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_scores_parsed(self):
        port = self.server.server_address[1]
        fn = rewards.llm_judge(
            WHERE,
            {"url": f"http://127.0.0.1:{port}/v1", "model": "m", "rubric": "r", "timeout": 5, "retries": 0, "concurrency": 2},
        )
        with self.assertLogs("trlx.rewards", level="WARNING") as logs:
            scores = fn(["hello", "silent"], prompts=["p", "p"])
        self.assertEqual(scores[0], float(len("Prompt:\np\n\nResponse:\nhello")))
        self.assertEqual(scores[1], 0.0)
        self.assertIn("no number", logs.output[0])


class ResolveTest(unittest.TestCase):
    def test_kinds(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "extra.py"
            path.write_text("def brevity(completions, **kw):\n    return [0.0 for _ in completions]\n")
            funcs = rewards.resolve(
                [
                    RewardEntry("phrases", {"required": ["x"]}),
                    RewardEntry("think_format_reward", None),
                    RewardEntry(f"{path}:brevity", None),
                    RewardEntry("json:dumps", None),
                    RewardEntry("org/reward-model", None),
                ]
            )
        self.assertEqual([getattr(f, "__name__", f) for f in funcs], ["phrases", "think_format_reward", "brevity", "dumps", "org/reward-model"])

    def test_errors(self):
        with self.assertRaisesRegex(TrlxError, "missing args"):
            rewards.resolve([RewardEntry("reference_match", None)])
        with self.assertRaisesRegex(TrlxError, "args are only for"):
            rewards.resolve([RewardEntry("org/model", {"k": 1})])
        with self.assertRaisesRegex(TrlxError, "no such file"):
            rewards.resolve([RewardEntry("missing.py:f", None)])
        with self.assertRaisesRegex(TrlxError, "no function"):
            rewards.resolve([RewardEntry("json:nope", None)])
        with self.assertRaisesRegex(TrlxError, "rejected args"):
            rewards.resolve([RewardEntry("get_soft_overlong_punishment", {"bogus": 1})])


class JudgeProgressTest(unittest.TestCase):
    # Exercise build_trainer -> resolve -> the retained callable -> real Endpoint
    # batching, with only HTTP and sleeping mocked and no model/training execution.
    def test_training_judge_reports_requests_and_retries_after_resolution(self):
        from trlx import train

        for method in ("grpo", "rloo"):
            with self.subTest(method=method):
                lines = []
                trainer_cls = Mock()
                entry = RewardEntry("llm_judge", {
                    "url": "https://judge.invalid/v1", "model": "judge", "rubric": "score",
                    "timeout": 5, "retries": 1, "concurrency": 1, "max_tokens": 16,
                })
                cfg = types.SimpleNamespace(
                    args=types.SimpleNamespace(), teacher=None, rewards=[entry], replay=None,
                    peft=None, method=types.SimpleNamespace(trainer_cls=trainer_cls),
                )
                replies = iter((TimeoutError(), "Score: 9", "Score: 3"))

                # The stage and total must already be visible before the first HTTP call.
                def respond(request, timeout):
                    self.assertTrue(any("judge requests" in line and "0/2 requests" in line for line in lines))
                    self.assertEqual(json.loads(request.data)["max_tokens"], 16)
                    reply = next(replies)
                    if isinstance(reply, Exception):
                        raise reply
                    return io.BytesIO(json.dumps({"choices": [{"message": {"content": reply}}]}).encode())

                # A retry notice must be visible before entering its backoff wait.
                def backoff(seconds):
                    self.assertEqual(seconds, 1)
                    self.assertTrue(any("timed out" in line and "retry attempt 2/2" in line for line in lines))

                with Progress(f"trlx {method} rank 0", emit=lines.append) as progress, \
                     patch.object(train, "_remove_stock_reporters"), \
                     patch("dataset.endpoint.urllib.request.urlopen", side_effect=respond), \
                     patch("dataset.endpoint.time.sleep", side_effect=backoff):
                    train.build_trainer(cfg, object(), object(), [], None, [], progress=progress)
                    judge = trainer_cls.call_args.kwargs["reward_funcs"][0]
                    self.assertEqual(judge.__name__, "llm_judge")
                    self.assertEqual(judge(["first", "second"], prompts=["p", "q"]), [9.0, 3.0])
                self.assertTrue(any("judge requests" in line and "2/2 requests; finished" in line for line in lines))
                self.assertTrue(all(line.startswith(f"trlx {method} rank 0:") for line in lines))

    # Exhausted judge retries remain contextual training errors and never claim success.
    def test_failed_judge_reports_before_propagating_training_error(self):
        lines = []
        args = {"url": "https://judge.invalid/v1", "model": "judge", "rubric": "score",
                "timeout": 5, "retries": 0, "concurrency": 1}
        with Progress("trlx grpo rank 0", emit=lines.append) as progress:
            judge = rewards.resolve([RewardEntry("llm_judge", args)], progress=progress)[0]
            with patch("dataset.endpoint.urllib.request.urlopen", side_effect=TimeoutError()), \
                 self.assertRaisesRegex(TrlxError, r"\[rewards\].funcs\[0\].*gave up after 1 attempts"):
                judge(["answer"])
        self.assertTrue(any("stopping batch:" in line and "timed out" in line for line in lines))
        self.assertTrue(any("judge requests" in line and "0/1 requests; failed" in line for line in lines))
        self.assertFalse(any("1/1 requests; finished" in line for line in lines))

    # Optional instrumentation must not make direct library reward use print progress.
    def test_judge_without_reporter_preserves_silent_library_use(self):
        reply = json.dumps({"choices": [{"message": {"content": "Score: 4"}}]}).encode()
        args = {"url": "https://judge.invalid/v1", "model": "judge", "rubric": "score",
                "timeout": 5, "retries": 0, "concurrency": 1}
        with contextlib.redirect_stderr(io.StringIO()) as output, \
             patch("dataset.endpoint.urllib.request.urlopen", return_value=io.BytesIO(reply)):
            judge = rewards.resolve([RewardEntry("llm_judge", args)])[0]
            self.assertEqual(judge(["answer"]), [4.0])
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
