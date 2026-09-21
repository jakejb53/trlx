"""Concurrent summary logs keep each full source and validated reply together."""

import threading
import unittest
from unittest.mock import patch

from dataset import eval_build
from dataset.endpoint import Endpoint, Reply
from dataset.io import DatasetError
from dataset.progress import Progress


class EvaluationSummaryLogging(unittest.TestCase):
    # Every request is mocked; these tests need no server or output files.
    def setUp(self):
        self.endpoint = Endpoint("http://localhost/v1", "model", None, 1, 0)

    # Row two must be displayed while row one still waits, in one atomic entry;
    # the final dataset must nevertheless retain original source order.
    def test_completed_pairs_are_live_coupled_and_saved_in_input_order(self):
        sources = [{"text": "First source\nwith full original text."},
                   {"text": "Second source: 雪\nAnother line."}]
        first_started = threading.Event()
        second_logged = threading.Event()
        lines, pair_threads = [], []
        collector_thread = threading.get_ident()

        # Publishing the second pair releases the first request, proving live delivery.
        def emit(line):
            lines.append(line)
            if "Summaries generated:" in line:
                pair_threads.append(threading.get_ident())
            if "Summary 2/2:" in line:
                second_logged.set()

        # Keep real endpoint scheduling while forcing a deterministic completion order.
        def complete(messages, max_tokens=None, **kwargs):
            self.assertEqual(max_tokens, 128)
            self.assertTrue(kwargs["require_stop"])
            if messages[-1]["content"] == sources[0]["text"]:
                first_started.set()
                if not second_logged.wait(3):
                    raise RuntimeError("second pair was not displayed while the first request waited")
                return Reply("First summary.", "hidden reasoning")
            if not first_started.wait(3):
                raise RuntimeError("first request did not start")
            return Reply("Second summary.", "hidden reasoning")

        progress = Progress("dataset eval-build", emit=emit, clock=lambda: 0.0)
        with patch.object(self.endpoint, "complete_full", side_effect=complete):
            output = eval_build.build(sources, self.endpoint, 128, 2, progress=progress)
        self.assertIsNone(progress.error)
        self.assertEqual(output, [{"text": "First summary."}, {"text": "Second summary."}])
        pairs = [line for line in lines if "Summaries generated:" in line]
        self.assertEqual(len(pairs), 2)
        for block, row_number, count in ((pairs[0], 2, 1), (pairs[1], 1, 2)):
            summary = output[row_number - 1]["text"]
            expected = (f"Source {row_number}/2:\n{sources[row_number - 1]['text']}\n\n"
                        f"Summary {row_number}/2:\n{summary}\n\nSummaries generated: {count}/2")
            self.assertIn(expected, block)
        self.assertEqual(pair_threads, [collector_thread, collector_thread])
        joined = "\n".join(lines)
        self.assertNotIn("hidden reasoning", joined)
        self.assertNotIn("\r", joined)
        for source in sources:
            self.assertEqual(joined.count(source["text"]), 1)

    # Invalid replies neither produce a successful pair nor advance its accepted count.
    def test_invalid_summaries_fail_before_logging_a_pair(self):
        for content, strip in (("   ", False), ("<think>unfinished", True),
                               ("<think>reasoning</think>summary", False)):
            with self.subTest(content=content, strip=strip):
                lines = []
                progress = Progress("dataset eval-build", emit=lines.append)
                with patch.object(self.endpoint, "complete_full", return_value=Reply(content, "")):
                    with self.assertRaisesRegex(DatasetError, "source row 1"):
                        eval_build.build([{"text": "Source"}], self.endpoint, 128, 1, strip, progress=progress)
                self.assertFalse(any("Summaries generated:" in line for line in lines))
                self.assertTrue(any("stopping batch:" in line for line in lines))

    # The displayed summary must match exactly what the output dataset will contain.
    def test_explicit_reasoning_stripping_applies_before_display(self):
        lines = []
        with patch.object(self.endpoint, "complete_full",
                          return_value=Reply("<think>hidden thoughts</think>  Clean summary.  ", "separate thoughts")):
            output = eval_build.build([{"text": "Full source"}], self.endpoint, 128, 1, True,
                                      progress=Progress("dataset eval-build", emit=lines.append))
        self.assertEqual(output, [{"text": "Clean summary."}])
        pair = next(line for line in lines if "Summaries generated:" in line)
        self.assertIn("Summary 1/1:\nClean summary.\n\nSummaries generated: 1/1", pair)
        self.assertNotIn("thoughts", pair)

    # Whole-source validation still precedes requests and any source/summary pair output.
    def test_invalid_source_prevents_all_requests(self):
        with patch.object(self.endpoint, "complete_full") as complete:
            with self.assertRaisesRegex(DatasetError, "source row 2"):
                eval_build.build([{"text": "Valid"}, {"text": ""}], self.endpoint, 128, 2)
        complete.assert_not_called()
