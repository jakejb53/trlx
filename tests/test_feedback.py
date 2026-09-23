"""CPU subprocess transport and terminal policy without loading training dependencies."""

import contextlib
import io
import logging
import os
import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

from trlx import TrlxError, feedback


class Collection(unittest.TestCase):
    # Children use the current interpreter and may not create bytecode outside the repo.
    def spawn(self, collector, code, source="rank 0"):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return collector.spawn([sys.executable, "-c", code], env, source)

    # A bounded wait detects blocked pipe writers; always reap children and close readers.
    def finish(self, collector, *children):
        try:
            for child in children:
                self.assertEqual(child.wait(timeout=10), 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
            collector.finish()

    # JSON-looking library output cannot impersonate structured warnings or progress.
    def test_structured_pipe_is_separate_from_raw_output(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        child = self.spawn(collector, '''
import json, os
from trlx.feedback import connect
client = connect()
client({"kind": "warning", "message": "structured 雪"})
print(json.dumps({"kind": "warning", "message": "raw JSON"}), flush=True)
os.write(2, "last diagnostic café".encode())
client.close()
''')
        self.finish(collector, child)
        events = collector.take()
        self.assertEqual(len(events), 3)
        self.assertEqual([e["message"] for e in events if e["kind"] == "warning"], ["structured 雪"])
        self.assertTrue(any(e["kind"] == "raw" and '"raw JSON"' in e["message"] for e in events))
        self.assertTrue(any(e["kind"] == "raw" and e["message"] == "last diagnostic café" for e in events))
        self.assertTrue(all(e["source"] == "rank 0" for e in events))
        self.assertIn("[rank 0] last diagnostic café\n", log.getvalue().decode())
        self.assertEqual(collector.take(), [])

    # Carriage returns do not identify disposable text: rapid diagnostics all remain visible.
    def test_raw_carriage_return_diagnostics_survive_collection_and_display(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        child = self.spawn(collector, '''
import os
os.write(2, b"warning before carriage return\\rnew diagnostic\\rfinal output")
''')
        self.finish(collector, child)
        lines = []
        view = feedback.View(lines.append, clock=lambda: 0.0)
        for event in collector.take():
            view.consume(event)
        expected = ["warning before carriage return", "new diagnostic", "final output"]
        self.assertEqual(lines, expected)
        for message in expected:
            self.assertIn(f"[rank 0] {message}\n", log.getvalue().decode())

    # Persistence must retain the severity and library identity supplied by Python logging.
    def test_log_preserves_logging_level_and_logger(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        child = self.spawn(collector, '''
import logging
from trlx.feedback import connect, configure_logging
client = connect()
configure_logging(client)
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)
logger.info("model loaded")
logger.warning("unfamiliar warning")
client.close()
''', source="rank 1")
        self.finish(collector, child)
        self.assertIn("[rank 1] INFO transformers: model loaded\n", log.getvalue().decode())
        self.assertIn("[rank 1] WARNING transformers: unfamiliar warning\n", log.getvalue().decode())
        events = collector.take()
        self.assertEqual([(e["level"], e["logger"]) for e in events],
                         [("INFO", "transformers"), ("WARNING", "transformers")])

    # A multiline report crosses pipe buffers intact and is emitted atomically.
    def test_metric_report_preserves_width_and_per_line_log_attribution(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        child = self.spawn(collector, '''
from trlx.feedback import connect
client = connect()
message = "Metrics — step 70\\n\\n" + "\\n".join("|" + " " * 98 + "|" for _ in range(1000))
client({"kind": "metric_report", "message": message})
client.close()
''')
        self.finish(collector, child)
        events = collector.take()
        self.assertEqual(len(events), 1)
        report = "Metrics — step 70\n\n" + "\n".join("|" + " " * 98 + "|" for _ in range(1000))
        self.assertEqual(events[0]["message"], report)
        lines = []
        view = feedback.View(lines.append)
        view.consume(events[0])
        self.assertEqual(lines, [report])
        self.assertTrue(all(len(line) == 100 for line in lines[0].split("\n")[2:]))
        self.assertEqual(log.getvalue().decode(),
                         "".join(f"[rank 0] {line}\n" for line in report.split("\n")))

    # Both pipes exceed kernel buffering before supervision consumes a single event.
    def test_large_payloads_and_multiple_ranks_drain_without_deadlock(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        code = '''
from trlx.feedback import connect
client = connect()
client({"kind": "note", "message": "雪" * 100000})
print("r" * 300000, flush=True)
client.close()
'''
        children = [self.spawn(collector, code, f"rank {rank}") for rank in range(2)]
        self.finish(collector, *children)
        events = collector.take()
        self.assertEqual(len(events), 4)
        for rank in range(2):
            received = [e for e in events if e["source"] == f"rank {rank}"]
            self.assertEqual(len(received), 2)
            self.assertEqual({e["kind"] for e in received}, {"note", "raw"})
            self.assertEqual(next(e["message"] for e in received if e["kind"] == "note"), "雪" * 100000)
            self.assertEqual(next(e["message"] for e in received if e["kind"] == "raw"), "r" * 300000)
        self.assertEqual(log.getvalue().decode().count("[rank "), 4)

    # A terminal failure discards its queue while all later diagnostics still reach disk.
    def test_disabled_display_keeps_collecting_without_queue(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        collector.accept("supervisor", {"kind": "note", "message": "before failure"})
        collector.disable_display()
        child = self.spawn(collector, '''
from trlx.feedback import connect
client = connect()
client({"kind": "note", "message": "after failure"})
print("x" * 300000, flush=True)
client.close()
''')
        self.finish(collector, child)
        self.assertEqual(collector.take(), [])
        self.assertIn(b"before failure", log.getvalue())
        self.assertIn(b"after failure", log.getvalue())
        self.assertIn(b"x" * 300000, log.getvalue())

    # Log failure is fatal to supervision, but readers must still unblock child writers.
    def test_log_failure_is_reported_after_child_output_drains(self):
        log = Mock()
        log.write.side_effect = OSError("disk full")
        collector = feedback.Collector(log, "run/log.txt")
        child = self.spawn(collector, 'print("x" * 300000, flush=True)')
        with self.assertRaisesRegex(TrlxError, "run/log.txt: cannot record feedback: disk full"):
            self.finish(collector, child)
        self.assertEqual(child.returncode, 0)
        self.assertEqual(collector.take(), [])

    # Malformed internal records are transport errors, never silently reclassified as text.
    def test_invalid_structured_record_is_reported(self):
        collector = feedback.Collector(io.BytesIO(), "log.txt")
        child = self.spawn(collector, '''
import os
from trlx.feedback import PIPE_ENV
os.write(int(os.environ[PIPE_ENV]), b'{"kind":"note"}\\n')
''')
        with self.assertRaisesRegex(TrlxError, "rank 0: cannot collect feedback: invalid feedback record"):
            self.finish(collector, child)


class Presentation(unittest.TestCase):
    # A controlled monotonic clock tests quiet intervals without sleeping.
    def setUp(self):
        self.now = 0.0
        self.lines = []
        self.view = feedback.View(self.lines.append, clock=lambda: self.now)

    # Real structured events carry source identity separately from their diagnostic text.
    def event(self, kind, message, source="rank 0", **fields):
        return {"source": source, "kind": kind, "message": message, **fields}

    # Duplicate warnings print promptly once and later summarize every affected rank.
    def test_warning_deduplication_preserves_counts_and_sources(self):
        warning = self.event("warning", "kernel fallback", logger="transformers", level="WARNING")
        self.view.consume(warning)
        self.view.consume(dict(warning, source="rank 1"))
        self.view.consume(warning)
        self.assertEqual(len(self.lines), 1)
        self.assertIn("kernel fallback", self.lines[0])
        self.view.warning_summary()
        self.assertEqual(len(self.lines), 2)
        self.assertIn("3 times", self.lines[-1])
        self.assertIn("rank 0, rank 1", self.lines[-1])
        self.view.warning_summary()
        self.assertEqual(len(self.lines), 2)

    # Filter one identified dependency notice, keeping unfamiliar warnings and other loggers.
    def test_only_known_dependency_deprecation_is_hidden(self):
        known = "FutureWarning: `torch.distributed.all_gather_into_tensor` is deprecated. Please use `torch.distributed.all_gather_single` instead."
        self.view.consume(self.event("warning", known, logger="py.warnings", level="WARNING"))
        self.assertEqual(self.lines, [])
        for message in ("FutureWarning: another operation is deprecated", "UserWarning: vocabulary assumption",
                        "UserWarning: `torch.distributed.all_gather_into_tensor` is deprecated.",
                        "FutureWarning: `torch.distributed.all_gather_single` is deprecated."):
            self.view.consume(self.event("warning", message, logger="py.warnings", level="WARNING"))
        self.view.consume(self.event("warning", known, logger="other.library", level="WARNING"))
        self.assertEqual(len(self.lines), 5)
        self.assertIn("another operation", self.lines[0])
        self.assertIn("vocabulary assumption", self.lines[1])
        self.assertIn(known, self.lines[-1])

    # Long steps stay quiet while measured counts still maintain progress state.
    def test_training_suppresses_waiting_without_losing_measured_progress(self):
        fields = {"label": "trainer running", "visible": True, "completed": 0, "total": 19, "unit": "steps"}
        self.view.consume(self.event("start", "training", **fields))
        self.now = 29
        self.view.waiting()
        self.assertEqual(self.lines, ["training"])
        self.view.consume(self.event("count", "step completed", **dict(fields, completed=1)))
        self.now = 58
        self.view.waiting()
        self.assertEqual(self.lines, ["training"])
        self.now = 300
        self.view.waiting()
        self.assertEqual(self.lines, ["training"])
        self.assertEqual(self.view.last_feedback, 29)
        self.assertEqual(self.view.stages["rank 0"][0]["completed"], 1)

    # A finished rank cannot re-enable notices while another rank trains or evaluates.
    def test_nested_pauses_stay_quiet_until_all_training_ranks_finish(self):
        training = {"label": "trainer running", "unit": "steps", "visible": False}
        for source in ("rank 0", "rank 1"):
            self.view.consume(self.event("start", "training", source=source, **training))
        self.view.consume(self.event("end", "training finished", **training))
        for label in ("evaluating", "saving checkpoint"):
            pause = {"label": label, "unit": "batches", "visible": False}
            self.view.consume(self.event("start", label, source="rank 1", **pause))
            self.now += 60
            self.view.waiting()
            self.assertEqual(self.lines, [])
            self.view.consume(self.event("end", label, source="rank 1", **pause))
        self.view.consume(self.event("warning", "kernel fallback", logger="library", level="WARNING"))
        self.view.consume(self.event("raw", "unexpected diagnostic"))
        self.assertEqual(len(self.lines), 2)
        self.view.consume(self.event("end", "training finished", source="rank 1", **training))
        self.view.consume(self.event("start", "checking checkpoint", source="verify",
                                     label="checking checkpoint", visible=False))
        self.now += 30
        self.view.waiting()
        self.assertEqual(len(self.lines), 3)
        self.assertIn("waiting: checking checkpoint [verify]", self.lines[-1])

    # Presentation filtering happens after log persistence, including raw heartbeats.
    def test_training_waiting_diagnostics_remain_in_log(self):
        log = io.BytesIO()
        collector = feedback.Collector(log, "log.txt")
        collector.accept("rank 0", {"kind": "start", "message": "training",
                                  "label": "trainer running", "unit": "steps", "visible": False})
        collector.accept("rank 0", {"kind": "waiting", "message": "trainer heartbeat"})
        for event in collector.take():
            self.view.consume(event)
        self.now = 60
        self.view.waiting()
        self.assertEqual(self.lines, [])
        self.assertIn("[rank 0] trainer heartbeat\n", log.getvalue().decode())

    # Equivalent starts are consolidated while a quiet notice identifies diverging ranks.
    def test_rank_stages_are_consolidated_and_divergence_is_visible(self):
        fields = {"label": "loading weights", "visible": True}
        self.view.consume(self.event("start", "loading weights", **fields))
        self.view.consume(self.event("start", "loading weights", source="rank 1", **fields))
        self.assertEqual(self.lines, ["loading weights"])
        self.view.consume(self.event("end", "loaded weights", **fields))
        self.view.consume(self.event("start", "preparing dataset", label="preparing dataset", visible=True))
        self.now = 30
        self.view.waiting()
        self.assertIn("preparing dataset [rank 0]", self.lines[-1])
        self.assertIn("loading weights [rank 1]", self.lines[-1])

    # Deduplication is per occurrence: evaluation repeats, but equivalent ranks share a row.
    def test_repeated_evaluation_displays_each_occurrence_once_across_ranks(self):
        for sequence in (1, 2):
            fields = {"label": "evaluating", "parent": "trainer running", "sequence": sequence,
                      "visible": True, "completed": 0, "total": 3, "unit": "batches"}
            for source in ("rank 0", "rank 1"):
                self.view.consume(self.event("start", "evaluating; 0/3 batches", source=source, **fields))
            for source in ("rank 0", "rank 1"):
                self.view.consume(self.event("end", "evaluating; 3/3 batches; finished",
                                             source=source, **dict(fields, completed=3)))
        self.assertEqual(self.lines, ["evaluating; 0/3 batches", "evaluating; 3/3 batches; finished"] * 2)

    # The same child operation must remain visible under both verification parents.
    def test_generation_under_base_and_checkpoint_parents_is_not_deduplicated(self):
        for parent in ("checking base behaviour", "checking checkpoint behaviour"):
            fields = {"label": "generating completions", "parent": parent, "sequence": 1,
                      "visible": True, "completed": 0, "total": 8, "unit": "prompts"}
            self.view.consume(self.event("start", "generating completions; 0/8 prompts", source="verify", **fields))
            self.view.consume(self.event("end", "generating completions; 8/8 prompts; finished",
                                         source="verify", **dict(fields, completed=8)))
        self.assertEqual(self.lines, ["generating completions; 0/8 prompts",
                                     "generating completions; 8/8 prompts; finished"] * 2)

    # Warning handling cannot suppress subsequent informational output or a distinct warning.
    def test_warning_followed_by_information_and_unknown_warning_preserves_all(self):
        self.view.consume(self.event("warning", "first warning", logger="library", level="WARNING"))
        self.view.consume(self.event("library", "configuration aligned", logger="library", level="INFO"))
        self.view.consume(self.event("raw", "raw information after warning"))
        self.view.consume(self.event("warning", "new unknown warning", logger="library", level="WARNING"))
        self.assertEqual(self.lines, ["WARNING [rank 0] library: first warning",
                                      "configuration aligned", "raw information after warning",
                                      "WARNING [rank 0] library: new unknown warning"])

    # Explicit logging severity, not words in raw text, identifies attributable errors.
    def test_structured_error_retains_rank(self):
        self.view.consume(self.event("warning", "worker failure", source="rank 1",
                                     logger="library", level="ERROR"))
        self.assertEqual(self.lines, ["ERROR [rank 1] library: worker failure"])

    # Expected CLI failures retain their worker identity without labelling ordinary text.
    def test_cli_error_retains_rank_but_command_text_does_not(self):
        self.view.consume(self.event("raw", "trlx sft: CUDA memory exhausted during training",
                                     source="rank 1"))
        self.view.consume(self.event("raw", "trlx show run", source="rank 1"))
        self.assertEqual(self.lines, ["[rank 1] trlx sft: CUDA memory exhausted during training",
                                      "trlx show run"])

    # Internal filesystem and option plumbing cannot interrupt metric headings.
    def test_internal_events_are_log_only_and_unknown_raw_output_is_visible(self):
        self.view.consume(self.event("start", "loading options", label="loading options", visible=False))
        self.view.consume(self.event("diagnostic", "finished operation"))
        self.assertEqual(self.lines, [])
        self.view.consume(self.event("raw", "unexpected library error"))
        self.assertEqual(self.lines, ["unexpected library error"])


class LoggingSetup(unittest.TestCase):
    # Restore process-wide logging exactly, including unrelated pre-existing handlers.
    def test_repeated_setup_preserves_levels_and_file_handlers_without_duplicates(self):
        names = ("transformers", "trl", "py.warnings")
        saved = [(logging.getLogger(name), list(logging.getLogger(name).handlers),
                  logging.getLogger(name).level, logging.getLogger(name).propagate,
                  logging.getLogger(name).disabled) for name in names]
        path = pathlib.Path(__file__).parent / "unused-feedback-test.log"
        file_handler = logging.FileHandler(path, delay=True)
        file_handler.setLevel(logging.CRITICAL + 1)
        events = []
        try:
            for logger, _, _, _, _ in saved:
                logger.handlers = [logging.StreamHandler(io.StringIO()), file_handler]
                logger.setLevel(logging.ERROR)
                logger.disabled = False
            with patch("trlx.feedback.logging.captureWarnings"):
                feedback.configure_logging(events.append)
                feedback.configure_logging(events.append)
            for logger, _, _, _, _ in saved:
                self.assertEqual(logger.level, logging.ERROR)
                self.assertIn(file_handler, logger.handlers)
                self.assertEqual(sum(isinstance(h, feedback.LogHandler) for h in logger.handlers), 1)
                logger.warning("below configured level")
                logger.error("one error")
            self.assertEqual(len(events), len(names))
            self.assertEqual({e["logger"] for e in events}, set(names))
            self.assertTrue(all(e["message"] == "one error" for e in events))
        finally:
            for logger, handlers, level, propagate, disabled in saved:
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate
                logger.disabled = disabled
            file_handler.close()


class WorkerProgress(unittest.TestCase):
    # Only duplicate progress bars are disabled; nonzero ranks still deliver unknown warnings.
    def test_rank_zero_keeps_bars_and_other_ranks_keep_diagnostics(self):
        names = ("transformers", "trl", "py.warnings")
        saved = [(logging.getLogger(name), list(logging.getLogger(name).handlers),
                  logging.getLogger(name).level, logging.getLogger(name).propagate,
                  logging.getLogger(name).disabled) for name in names]
        events = []
        try:
            with patch("datasets.utils.logging.disable_progress_bar") as dataset_bars:
                with patch("transformers.utils.logging.disable_progress_bar") as model_bars:
                    feedback.configure_worker_progress(0)
                    dataset_bars.assert_not_called()
                    model_bars.assert_not_called()
                    feedback.configure_worker_progress(1)
                    dataset_bars.assert_called_once_with()
                    model_bars.assert_called_once_with()
                    with patch("trlx.feedback.logging.captureWarnings"):
                        feedback.configure_logging(events.append)
                    logger = logging.getLogger("transformers")
                    logger.setLevel(logging.WARNING)
                    logger.disabled = False
                    logger.warning("unexpected rank-one diagnostic")
            lines = []
            view = feedback.View(lines.append)
            for event in events:
                view.consume(dict(event, source="rank 1"))
            self.assertEqual(len(lines), 1)
            self.assertIn("[rank 1] transformers: unexpected rank-one diagnostic", lines[0])
        finally:
            for logger, handlers, level, propagate, disabled in saved:
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate
                logger.disabled = disabled


class StartupBuffer(unittest.TestCase):
    # Before a run directory exists, the same thirty-second policy reports blocking startup.
    def test_startup_waits_thirty_seconds_and_reports_active_operation(self):
        startup = feedback.Startup()
        clock = Mock(return_value=0.0)
        startup.view.clock = clock
        startup.view.last_feedback = startup.view.last_notice = 0.0
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            startup({"kind": "start", "message": "resolving model", "label": "resolving model",
                     "visible": True, "parent": None, "sequence": 1})
            for now in (10.0, 20.0, 29.0):
                clock.return_value = now
                startup({"kind": "waiting", "message": "internal ten-second heartbeat"})
            self.assertEqual(output.getvalue(), "resolving model\n")
            clock.return_value = 30.0
            startup({"kind": "waiting", "message": "internal ten-second heartbeat"})
        self.assertIn("waiting: resolving model [supervisor]", output.getvalue())
        self.assertIn("last substantive feedback 30s ago", output.getvalue())
        self.assertNotIn("internal ten-second heartbeat", output.getvalue())
        self.assertIsNone(startup.collector)

    # Early diagnostics reach the eventual log once without replaying public startup text.
    def test_attach_flushes_buffer_without_replaying_display(self):
        startup = feedback.Startup()
        log = io.BytesIO()
        collector = feedback.Collector(log, "run/log.txt")
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            startup({"kind": "diagnostic", "command": "trlx sft", "message": "starting"})
            startup({"kind": "start", "message": "resolving configuration", "visible": False})
            startup({"kind": "start", "message": "inspecting GPUs", "visible": True})
            startup.attach(collector)
        self.assertEqual(output.getvalue(), "trlx sft: starting\ninspecting GPUs\n")
        self.assertEqual(collector.take(), [])
        self.assertEqual(startup.pending, [])
        self.assertEqual(log.getvalue().decode(), "[supervisor] starting\n[supervisor] resolving configuration\n[supervisor] inspecting GPUs\n")
        startup({"kind": "note", "message": "run allocated", "visible": True})
        self.assertEqual([e["message"] for e in collector.take()], ["run allocated"])
        self.assertEqual(log.getvalue().decode().count("[supervisor] starting"), 1)
        startup.detach()


if __name__ == "__main__":
    unittest.main()
