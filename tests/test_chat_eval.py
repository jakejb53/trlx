"""Per-chunk evaluation membership and coupled chat output publication."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from dataset import chat, cli
from dataset.endpoint import Reply
from dataset.io import DatasetError
from tests.test_chat import StubEndpoint, QUESTIONS_PROMPT, ANSWERS_PROMPT


class ChatEvaluation(unittest.TestCase):
    # Multi-chunk membership is assigned after deduplication and before answer filtering.
    def test_short_chunks_duplicates_and_empty_answers(self):
        questions = StubEndpoint([Reply("a\nb\nc", ""), Reply("a\nd", ""), Reply("e\nf", "")])
        answers = StubEndpoint([Reply(value, "reasoning") for value in ("A", "B", "", "D", "E", "F")])
        with patch.object(chat, "chunk_text", return_value=["one", "two", "three"]):
            train, evaluation, skipped = chat.build("source", 100, 3, questions, answers,
                QUESTIONS_PROMPT, ANSWERS_PROMPT, 2, eval_n=1, exclude_reasoning=True)
        self.assertEqual([row["messages"][0]["content"] for row in train], ["a", "b", "e"])
        self.assertEqual([row["messages"][0]["content"] for row in evaluation], ["d", "f"])
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all("reasoning" not in row for row in train + evaluation))
        self.assertEqual(len(answers.requests), 6)

    # Use real parser, handler, and file writer with endpoint responses supplied in memory.
    def test_cli_writes_nine_training_and_one_evaluation(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            root = pathlib.Path(folder)
            source = root / "source.txt"
            source.write_text("A source paragraph.")
            q, a = root / "q.prompt", root / "a.prompt"
            q.write_text(QUESTIONS_PROMPT)
            a.write_text(ANSWERS_PROMPT)
            training, evaluation = root / "training.jsonl", root / "eval.jsonl"
            argv = ["chat", str(source), "--questions-endpoint", "https://example.invalid/v1",
                    "--questions-model", "model", "--n", "10", "--max-tokens", "100",
                    "--eval-n", "1", "--out", str(training), "--eval-out", str(evaluation),
                    "--questions-prompt", str(q), "--answers-prompt", str(a)]
            questions = StubEndpoint([Reply("\n".join(f"Question {i}?" for i in range(10)), "")])
            answers = StubEndpoint([Reply(f"Answer {i}", "reasoning") for i in range(10)])
            with patch.object(cli, "Endpoint", side_effect=[questions, answers]), \
                 patch.object(cli.env, "load"), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                questions.display_url = answers.display_url = "https://example.invalid/v1"
                self.assertEqual(cli.main(argv), 0)
            train = [json.loads(line) for line in training.read_text().splitlines()]
            evaluate = [json.loads(line) for line in evaluation.read_text().splitlines()]
            self.assertEqual(len(train), 9)
            self.assertEqual(len(evaluate), 1)
            self.assertEqual(evaluate[0]["messages"][0]["content"], "Question 9?")
            self.assertEqual(evaluate[0]["reasoning"], "reasoning")

    # Invalid counts, omitted pairs, aliases, and occupied outputs fail before endpoint setup.
    def test_bad_options_fail_before_requests(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as folder:
            root = pathlib.Path(folder)
            out, evaluation = root / "train.jsonl", root / "eval.jsonl"
            evaluation.write_text("existing")
            base = ["chat", "source.txt", "--questions-endpoint", "https://example.invalid/v1",
                    "--questions-model", "model", "--n", "10", "--max-tokens", "100", "--out", str(out)]
            cases = [(["--eval-n", "-1"], "smaller than --n"),
                     (["--eval-n", "10", "--eval-out", str(evaluation)], "smaller than --n"),
                     (["--eval-n", "1"], "supplied together"),
                     (["--eval-out", str(evaluation)], "supplied together"),
                     (["--eval-n", "1", "--eval-out", str(out)], "conflicting output"),
                     (["--eval-n", "1", "--eval-out", str(evaluation)], "--force")]
            for extra, message in cases:
                with self.subTest(extra=extra), patch.object(cli, "Endpoint") as endpoint:
                    args = cli.build_parser().parse_args(base + extra)
                    with self.assertRaisesRegex(DatasetError, message):
                        cli._cmd_chat(args)
                    endpoint.assert_not_called()
            self.assertFalse(out.exists())
            self.assertEqual(evaluation.read_text(), "existing")
