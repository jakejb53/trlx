"""Progress distinguishes measured work from waiting and owns its watcher lifetime."""

import io
import threading
import unittest
from unittest.mock import Mock, patch

from dataset.progress import Progress, stage


class ProgressTest(unittest.TestCase):
    # Advance time explicitly so waiting assertions do not depend on scheduler timing.
    def setUp(self):
        self.clock = Mock(return_value=0.0)
        self.lines = []
        self.progress = Progress("dataset chat", emit=self.lines.append, clock=self.clock)

    # A stalled request advertises its stage and zero completed work after the interval.
    def test_waiting_describes_current_stage_without_inventing_progress(self):
        with stage(self.progress, "questions", total=3, unit="requests"):
            self.clock.return_value = 9.0
            self.progress.waiting()
            self.assertEqual(len(self.lines), 1)
            self.clock.return_value = 10.0
            self.progress.waiting()
            self.assertIn("waiting: questions; 0/3 requests", self.lines[-1])
            self.assertIn("stage elapsed 10.0s", self.lines[-1])
            self.assertIn("no measured progress yet", self.lines[-1])

    # Retry messages are activity, but must not make stalled work appear to advance.
    def test_retry_notice_does_not_reset_time_since_measured_progress(self):
        with stage(self.progress, "answers", total=4, unit="requests") as activity:
            self.clock.return_value = 2.0
            activity.advance()
            self.clock.return_value = 7.0
            activity.note("request 2: timed out; retry attempt 2/3 in 1s")
            self.assertIn("retry attempt 2/3", self.lines[-1])
            self.clock.return_value = 17.0
            self.progress.waiting()
            self.assertIn("1/4 requests", self.lines[-1])
            self.assertIn("stage elapsed 17.0s", self.lines[-1])
            self.assertIn("last measured progress 15.0s ago", self.lines[-1])

    # Failing a nested operation must retain the enclosing operation's measured history.
    def test_nested_failure_restores_parent_stage_and_its_progress(self):
        with stage(self.progress, "processing rows", total=5) as parent:
            self.clock.return_value = 1.0
            parent.advance()
            with self.assertRaisesRegex(ValueError, "bad reply"):
                with stage(parent, "scoring"):
                    self.clock.return_value = 3.0
                    raise ValueError("bad reply")
            self.assertIn("scoring; failed", self.lines[-1])
            self.clock.return_value = 13.0
            self.progress.waiting()
            self.assertIn("waiting: processing rows; 1/5 items", self.lines[-1])
            self.assertIn("last measured progress 12.0s ago", self.lines[-1])

    # Nested curses ownership suppresses all writes and delays the next waiting notice.
    def test_suspension_keeps_terminal_quiet_until_interactive_viewer_closes(self):
        with stage(self.progress, "viewing artifacts") as activity:
            before = list(self.lines)
            with self.progress.suspended():
                self.clock.return_value = 20.0
                activity.note("viewer is active")
                self.progress.waiting()
                with self.progress.suspended():
                    self.progress.waiting()
                self.progress.waiting()
                self.assertEqual(self.lines, before)
            self.clock.return_value = 29.0
            self.progress.waiting()
            self.assertEqual(self.lines, before)
            self.clock.return_value = 30.0
            self.progress.waiting()
            self.assertIn("waiting: viewing artifacts", self.lines[-1])

    # Completion is final even if another timer tick or cleanup calls finish again.
    def test_finish_is_idempotent_and_prevents_later_heartbeats(self):
        self.clock.return_value = 4.0
        self.progress.finish("failed")
        self.assertIn("failed; elapsed 4.0s", self.lines[-1])
        before = list(self.lines)
        self.clock.return_value = 100.0
        self.progress.waiting()
        self.progress.finish()
        self.assertEqual(self.lines, before)

    # Real background feedback must remain available while the command cannot make calls.
    def test_background_reports_while_command_thread_is_blocked(self):
        waiting = threading.Event()
        lines = []

        # Only an actual waiting message can release the blocked command in this test.
        def emit(line):
            lines.append(line)
            if "waiting: loading model" in line:
                waiting.set()

        # The command blocks on an event: only the independent watcher can release it.
        with patch("dataset.progress.WAIT_SECONDS", 0.01):
            with Progress("trlx merge", emit=emit) as progress:
                with stage(progress, "loading model"):
                    self.assertTrue(waiting.wait(3), "no feedback while command was blocked")
            self.assertFalse(progress._thread.is_alive())
        self.assertIn("no measured progress yet", next(line for line in lines if "waiting:" in line))
        self.assertIn("completed; elapsed", lines[-1])

    # Both exit paths retain their original exception and stop the owned watcher.
    def test_context_reports_failure_and_interruption_without_hiding_exception(self):
        for error, outcome in ((ValueError("bad input"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(outcome=outcome):
                lines = []
                with self.assertRaises(type(error)) as caught:
                    with Progress("dataset stats", emit=lines.append) as progress:
                        raise error
                self.assertIs(caught.exception, error)
                self.assertIn(f"{outcome}; elapsed", lines[-1])
                self.assertFalse(progress._thread.is_alive())

    # Training's protected display handler records one error while subsequent work continues.
    def test_training_sink_failure_calls_handler_once_and_preserves_work(self):
        error = OSError("display closed")
        sink = Mock(side_effect=error)
        failures = []
        with Progress("trlx sft", emit=sink, on_error=failures.append) as progress:
            with stage(progress, "training") as activity:
                activity.advance()
                activity.note("checkpoint saved")
        self.assertEqual(failures, [error])
        sink.assert_called_once()

    # Python's print(file=None) fallback must never mix diagnostic lines into metric stdout.
    def test_missing_stderr_reports_failure_without_redirecting_progress_to_stdout(self):
        output = io.StringIO()
        failures = []
        with patch("dataset.progress.sys.stderr", None), patch("dataset.progress.sys.stdout", output):
            with Progress("trlx sft", on_error=failures.append) as progress:
                with stage(progress, "training") as activity:
                    activity.advance()
                    activity.note("checkpoint saved")
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], BrokenPipeError)
        self.assertIs(progress.error, failures[0])
        self.assertFalse(progress._thread.is_alive())


class StructuredProgress(unittest.TestCase):
    # Metadata distinguishes repeated work without relying on rendered stage text.
    def test_parent_and_sequence_identify_repeated_and_nested_stages(self):
        events = []
        clock = Mock(return_value=0.0)
        progress = Progress("trlx sft", events=events.append, clock=clock)
        with stage(progress, "trainer running") as training:
            for _ in range(2):
                with stage(training, "evaluating", total=3, unit="batches"):
                    pass
        for parent in ("checking base behaviour", "checking checkpoint behaviour"):
            with stage(progress, parent) as checking:
                with stage(checking, "generating completions", total=8, unit="prompts"):
                    pass
        evaluation = [e for e in events if e["kind"] == "start" and e["label"] == "evaluating"]
        self.assertEqual([(e["parent"], e["sequence"]) for e in evaluation],
                         [("trainer running", 1), ("trainer running", 2)])
        generation = [e for e in events if e["kind"] == "start" and e["label"] == "generating completions"]
        self.assertEqual([(e["parent"], e["sequence"]) for e in generation],
                         [("checking base behaviour", 1), ("checking checkpoint behaviour", 1)])
        for event in evaluation + generation:
            ending = next(e for e in events if e["kind"] == "end"
                          and (e["label"], e["parent"], e["sequence"]) ==
                          (event["label"], event["parent"], event["sequence"]))
            self.assertEqual(ending["total"], event["total"])

    # Every optimizer count must reach supervision even when no wall-clock interval elapses.
    def test_every_optimizer_step_is_delivered_at_unchanged_clock(self):
        events = []
        progress = Progress("trlx sft", events=events.append, clock=Mock(return_value=0.0))
        with stage(progress, "trainer running", total=5, unit="steps") as activity:
            for _ in range(5):
                activity.advance()
        counts = [e for e in events if e["kind"] == "count"]
        self.assertEqual([e["completed"] for e in counts], [1, 2, 3, 4, 5])
        self.assertTrue(all(e["measured"] == 0.0 for e in counts))
        self.assertEqual(events[-1]["kind"], "end")
        self.assertEqual(events[-1]["completed"], 5)

    # Per-row preparation cannot flood the transport; its end still carries the exact total.
    def test_row_counts_are_coalesced_but_final_total_is_delivered(self):
        events = []
        clock = Mock(return_value=0.0)
        progress = Progress("trlx sft", events=events.append, clock=clock)
        with stage(progress, "preparing dataset", total=100, unit="rows") as activity:
            for _ in range(50):
                activity.advance()
            self.assertEqual([e for e in events if e["kind"] == "count"], [])
            clock.return_value = 1.0
            activity.advance()
            for _ in range(49):
                activity.advance()
        self.assertEqual([e["completed"] for e in events if e["kind"] == "count"], [51])
        self.assertEqual(events[-1]["kind"], "end")
        self.assertEqual(events[-1]["completed"], 100)
        self.assertIn("100/100 rows", events[-1]["message"])


if __name__ == "__main__":
    unittest.main()
