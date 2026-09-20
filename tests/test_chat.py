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

from dataset.chat import ANSWERS_PROMPT, QUESTIONS_PROMPT, build, load_prompt, strip_inline_reasoning
from dataset.endpoint import Endpoint, Reply
from dataset.io import DatasetError
from dataset.progress import Progress

TEXT = "Rain falls from clouds. Rivers carry water to the sea."


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
    def test_reasoning_column_from_the_field(self):
        rows, skipped = run([Reply("What falls?", "planning the question")],
                            [Reply("Rain falls.", "recalling the text")])
        self.assertEqual(skipped, [])
        self.assertEqual(rows[0]["reasoning"], "recalling the text")
        self.assertEqual(rows[0]["messages"][1]["content"], "Rain falls.")
        # Pass 1's reasoning is scaffolding for a parse and is not carried.
        self.assertNotIn("planning the question", str(rows[0]))

    def test_reasoning_column_empty_without_a_field(self):
        rows, _ = run([Reply("What falls?", "")], [Reply("Rain falls.", "")])
        self.assertEqual(rows[0]["reasoning"], "")

    def test_inline_block_in_questions_is_fatal(self):
        with self.assertRaisesRegex(DatasetError, "chunk 0: reply begins with a <think> block"):
            run([Reply("<think>\nhmm\n</think>\nWhat falls?", "")], [Reply("Rain falls.", "")])

    def test_inline_block_in_answers_is_fatal(self):
        with self.assertRaisesRegex(DatasetError, "question: What falls\\?"):
            run([Reply("What falls?", "")], [Reply("<think>\nhmm\n</think>\nRain falls.", "")])

    def test_stripping_recovers_both_passes(self):
        rows, skipped = run([Reply("<think>\nhmm\n</think>\nWhat falls?", "")],
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
            rows, skipped = build(TEXT, 1000, 1, questions, answers, QUESTIONS_PROMPT,
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
            path = pathlib.Path(d) / "prompt.txt"
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(DatasetError, "prompt is not valid UTF-8") as result:
                load_prompt(path, QUESTIONS_PROMPT)
            self.assertIn(str(path), str(result.exception))


if __name__ == "__main__":
    unittest.main()
