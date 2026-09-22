"""Inline-reasoning handling and the reasoning column in dataset chat (SPEC 3).

No network: StubEndpoint stands in for Endpoint and returns canned Reply
objects in request order, which is the order build() relies on.
"""

import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from dataset.chat import build, strip_inline_reasoning
from dataset.endpoint import Endpoint, Reply
from dataset.io import DatasetError
from dataset.progress import Progress
from dataset.prompts import load as load_prompt

TEXT = "Rain falls from clouds. Rivers carry water to the sea."
QUESTIONS_PROMPT = "Generate {n} questions from {chunk}"
ANSWERS_PROMPT = "Use {chunk} to answer {question}"


# Returns the queued replies for each pass. complete_many_full is the only
# method build() calls; replies are consumed in request order.
class StubEndpoint:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete_many_full(self, message_lists, concurrency, max_tokens=None, *, progress=None, label="requests"):
        self.requests.extend(message_lists)
        out, self.replies = self.replies[: len(message_lists)], self.replies[len(message_lists):]
        return out


def run(questions, answers, strip=False, n=1):
    q_ep = StubEndpoint(questions)
    a_ep = StubEndpoint(answers)
    return build(TEXT, 1000, n, q_ep, a_ep, QUESTIONS_PROMPT, ANSWERS_PROMPT, 1, strip)


class StripInlineReasoningTest(unittest.TestCase):
    def test_no_tag_is_returned_unchanged(self):
        self.assertEqual(strip_inline_reasoning("What falls?", "chunk 0", False), "What falls?")

    def test_closed_block_is_fatal_by_default(self):
        with self.assertRaisesRegex(DatasetError, "reasoning parser"):
            strip_inline_reasoning("<think>\nhmm\n</think>\nWhat falls?", "chunk 0", False)

    def test_closed_block_is_removed_when_stripping(self):
        text = strip_inline_reasoning("<think>\nhmm\n</think>\nWhat falls?", "chunk 0", True)
        self.assertEqual(text, "What falls?")

    # Truncation, so there is nothing to strip and no answer to keep.
    def test_unclosed_block_is_fatal_even_when_stripping(self):
        for strip in (False, True):
            with self.assertRaisesRegex(DatasetError, "never closes"):
                strip_inline_reasoning("<think>\nhmm and then", "chunk 0", strip)


class BuildTest(unittest.TestCase):
    # Duplicates within/across chunks never reach the answer endpoint; first context wins.
    def test_duplicate_questions_removed_before_answers(self):
        questions = StubEndpoint([
            Reply("1. What falls?\n2) What falls?\n3. Where does water go?", ""),
            Reply("- What falls?  \n* What flows?\n3. Where does water go?", ""),
            Reply("What flows?", ""),
        ])
        answers = StubEndpoint([Reply("Rain.", ""), Reply("The sea.", ""), Reply("Rivers.", "")])
        with patch("dataset.chat.chunk_text", return_value=["first source", "second source", "third source"]):
            rows, evaluation, skipped = build(TEXT, 1000, 3, questions, answers,
                                  QUESTIONS_PROMPT, ANSWERS_PROMPT, 1)
        self.assertEqual([row["messages"][0]["content"] for row in rows],
                         ["What falls?", "Where does water go?", "What flows?"])
        self.assertEqual([request[0]["content"] for request in answers.requests], [
            "Use first source to answer What falls?", "Use first source to answer Where does water go?",
            "Use second source to answer What flows?",
        ])
        self.assertEqual(len(skipped), 4)
        self.assertIn("chunk 0: duplicate question (first in chunk 0): What falls?", skipped)
        self.assertIn("chunk 2: duplicate question (first in chunk 1): What flows?", skipped)

    # The CLI flags independently control inline content and the output column.
    def test_reasoning_controls_are_independent(self):
        from dataset.cli import build_parser

        for strip in (False, True):
            for exclude in (False, True):
                with self.subTest(strip=strip, exclude=exclude):
                    flags = (["--strip-reasoning-tags"] if strip else []) + (["--exclude-reasoning"] if exclude else [])
                    args = build_parser().parse_args([
                        "chat", "input.txt", "--out", "output.jsonl", "--questions-endpoint", "https://example.invalid",
                        "--questions-model", "test", "--n", "1", "--max-tokens", "1000", *flags,
                    ])
                    answer = "<think>inline</think>Rain falls." if strip else "Rain falls."
                    rows, _, _ = build(TEXT, 1000, 1, StubEndpoint([Reply("What falls?", "question reasoning")]),
                                    StubEndpoint([Reply(answer, "answer reasoning")]), QUESTIONS_PROMPT,
                                    ANSWERS_PROMPT, 1, args.strip_reasoning_tags, exclude_reasoning=args.exclude_reasoning)
                    self.assertEqual(rows[0]["messages"][1]["content"], "Rain falls.")
                    self.assertEqual("reasoning" in rows[0], not exclude)
                    if not exclude:
                        self.assertEqual(rows[0]["reasoning"], "answer reasoning")
        with self.assertRaisesRegex(DatasetError, "reasoning parser"):
            build(TEXT, 1000, 1, StubEndpoint([Reply("What falls?", "")]),
                  StubEndpoint([Reply("<think>inline</think>Rain falls.", "separate")]),
                  QUESTIONS_PROMPT, ANSWERS_PROMPT, 1, exclude_reasoning=True)

    # Inserted source and question text must never be interpreted as placeholders.
    def test_supplied_templates_control_requests_without_recursive_substitution(self):
        questions = StubEndpoint([Reply("What is {chunk}?", "")])
        answers = StubEndpoint([Reply("An answer", "")])
        build("Source {n}", 1000, 2, questions, answers,
              "Ask {n}: {chunk}", "Answer {question} from {chunk}", 1)
        self.assertEqual(questions.requests, [[{"role": "user", "content": "Ask 2: Source {n}"}]])
        self.assertEqual(answers.requests, [[{"role": "user", "content": "Answer What is {chunk}? from Source {n}"}]])

    def test_reasoning_column_from_the_field(self):
        rows, _, skipped = run([Reply("What falls?", "planning the question")],
                            [Reply("Rain falls.", "recalling the text")])
        self.assertEqual(skipped, [])
        self.assertEqual(rows[0]["reasoning"], "recalling the text")
        self.assertEqual(rows[0]["messages"][1]["content"], "Rain falls.")
        # Pass 1's reasoning is scaffolding for a parse and is not carried.
        self.assertNotIn("planning the question", str(rows[0]))

    def test_reasoning_column_empty_without_a_field(self):
        rows, _, _ = run([Reply("What falls?", "")], [Reply("Rain falls.", "")])
        self.assertEqual(rows[0]["reasoning"], "")

    def test_inline_block_in_questions_is_fatal(self):
        with self.assertRaisesRegex(DatasetError, "chunk 0: reply begins with a <think> block"):
            run([Reply("<think>\nhmm\n</think>\nWhat falls?", "")], [Reply("Rain falls.", "")])

    def test_inline_block_in_answers_is_fatal(self):
        with self.assertRaisesRegex(DatasetError, "question: What falls\\?"):
            run([Reply("What falls?", "")], [Reply("<think>\nhmm\n</think>\nRain falls.", "")])

    def test_stripping_recovers_both_passes(self):
        rows, _, skipped = run([Reply("<think>\nhmm\n</think>\nWhat falls?", "")],
                            [Reply("<think>\nhmm\n</think>\nRain falls.", "")], strip=True)
        self.assertEqual(skipped, [])
        self.assertEqual(rows[0]["messages"][0]["content"], "What falls?")
        self.assertEqual(rows[0]["messages"][1]["content"], "Rain falls.")


class BuildProgress(unittest.TestCase):
    # Two-pass generation must expose source size and the active pass before awaiting replies.
    def test_source_count_and_each_pass_are_visible_before_network_work(self):
        lines = []
        observed = []
        questions = Endpoint("http://questions/v1", "question-model", None, 1, 0)
        answers = Endpoint("http://answers/v1", "answer-model", None, 1, 0)

        # Snapshot at the external boundary, before any reply can complete.
        def reply(request, timeout):
            observed.append((request.full_url, list(lines)))
            content = "What falls?" if request.full_url == questions.url else "Rain falls."
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

        with patch("dataset.endpoint.urllib.request.urlopen", side_effect=reply):
            rows, _, skipped = build(TEXT, 1000, 1, questions, answers, QUESTIONS_PROMPT,
                                  ANSWERS_PROMPT, 1, progress=Progress("dataset chat", emit=lines.append))
        self.assertEqual(skipped, [])
        self.assertEqual(rows[0]["messages"][1]["content"], "Rain falls.")
        self.assertEqual([url for url, _ in observed], [questions.url, answers.url])
        question_output = "\n".join(observed[0][1])
        answer_output = "\n".join(observed[1][1])
        self.assertIn("1 source chunks", question_output)
        self.assertIn("questions from question-model", question_output)
        self.assertIn("0/1 requests", question_output)
        self.assertNotIn("answers from answer-model", question_output)
        self.assertIn("questions from question-model", answer_output)
        self.assertIn("1/1 requests; finished", answer_output)
        self.assertIn("answers from answer-model", answer_output)


class PromptErrors(unittest.TestCase):
    # Invalid prompt encodings identify the file before any generation request.
    def test_invalid_utf8_prompt(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as d:
            path = pathlib.Path(d) / "prompt.prompt"
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(DatasetError, "prompt is not valid UTF-8") as result:
                load_prompt(path, required=("n", "chunk"), allowed=("n", "chunk"))
            self.assertIn(str(path), str(result.exception))


if __name__ == "__main__":
    unittest.main()
