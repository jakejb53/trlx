"""Supervisor behavior when presentation fails; no models or run files are created."""

import contextlib
import io
import pathlib
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

from dataset.progress import Progress
from trlx import TrlxError, cli, launch, render_tui, run_dirs, train


# A process that finishes after several polls, so the real Job exercises its
# worker-to-verify transition without starting training or touching CUDA.
class Process:
    # Keep completion deterministic while allowing several display updates.
    def __init__(self, code=0):
        self.code = code
        self.polls_left = 6
        self.kills = 0
        self.waits = 0

    # Once complete, every later observation returns the same exit code.
    def poll(self):
        if self.polls_left:
            self.polls_left -= 1
            return None
        return self.code

    # Record cancellation separately from ordinary completion.
    def kill(self):
        self.kills += 1
        self.polls_left = 0
        self.code = -9

    # Reaping is observable without blocking a test.
    def wait(self):
        self.waits += 1
        return self.code


# A closed output destination can fail again during interpreter shutdown.
class BrokenStream(io.StringIO):
    # Match the failure of a pipe whose reader has already exited.
    def write(self, text):
        raise BrokenPipeError("reader closed")

    # Keep a buffered failure pending until the supervisor disables the stream.
    def flush(self):
        raise BrokenPipeError("reader closed")


class Supervisor(unittest.TestCase):
    # Replace only external inputs and display I/O; retain the real Job loop.
    def setUp(self):
        self.worker = Process()
        self.verify = Process()
        self.log = io.BytesIO()
        self.stdout = io.StringIO()
        self.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        self.addCleanup(self.stderr.close)
        self.enterContext(patch.object(sys, "stdout", self.stdout))
        self.enterContext(patch.object(sys, "stderr", self.stderr))
        self.enterContext(patch.dict(train.os.environ))
        self.args = types.SimpleNamespace(
            command="sft", config="memory.toml", gpus=None, strategy=None,
            no_verify=False, tui=False, _rank=None,
        )
        cfg = self.cfg = types.SimpleNamespace(
            args=types.SimpleNamespace(run_name="memory", resume_from_checkpoint=None),
            ranges={"loss": (0, 2)}, document={}, method=types.SimpleNamespace(name="sft"),
            model=types.SimpleNamespace(path="memory-model"), verify_prompts=None,
        )
        self._patch("trlx.train.config_mod.resolve", side_effect=self._resolved)
        self._patch("trlx.train.config_mod.from_document", return_value=cfg)
        self.original_confirm = train.review.confirm
        self.confirm = self._patch("trlx.train.review.confirm", return_value=True)
        self.check_config = self._patch("trlx.train.preflight.check_config")
        self._patch("trlx.launch.select_gpus", return_value=([0], 1))
        self._patch("trlx.launch.physical_ids", return_value=["synthetic-device"])
        self.strategy = self._patch("trlx.launch.choose_strategy", return_value=("single", "test"))
        self.create_run = self._patch("trlx.train._create_run_dir", return_value=pathlib.Path("memory-run"))
        self._patch("trlx.train.run_dirs.locked", return_value=contextlib.nullcontext())
        self.snapshot_path = pathlib.Path("memory-run/config.toml")
        self.snapshot = self._patch("trlx.train._write_snapshot", return_value=self.snapshot_path)
        self._patch("trlx.train._final_checkpoint", return_value=pathlib.Path("memory-run/checkpoint-1"))
        self.spawn = self._patch("trlx.launch.spawn", return_value=[self.worker])
        self.spawn_verify = self._patch("trlx.launch.spawn_verify", return_value=self.verify)
        self._patch("trlx.launch.time.sleep")
        self._patch("trlx.train.open", create=True, side_effect=self._open)
        self._patch("pathlib.Path.exists", return_value=True)
        self.records = self._patch("trlx.metrics.read", return_value=[{
            "step": 1, "max_steps": 2, "epoch": 0.5, "num_train_epochs": 1,
            "eval": False, "log": {"loss": 1.0},
        }])
        self._patch("trlx.show.load_config", return_value=("memory", cfg.ranges))
        self.load = self._patch("trlx.show.load", return_value=types.SimpleNamespace(log_tail=[]))
        self.tui = self.enterContext(patch.object(render_tui, "run", side_effect=lambda load: load(4)))

    # Direct supervisor tests choose display mode by changing args; mirror the CLI's
    # resolved launch controls so TUI failures still exercise the actual TUI path.
    def _resolved(self, path, method, overrides):
        return {"run": {"gpus": "all", "strategy": "auto", "tui": self.args.tui,
                        "verify": not self.args.no_verify}}

    # Restore every patched boundary even when a failure propagates out of run.
    def _patch(self, target, **kwargs):
        return self.enterContext(patch(target, **kwargs))

    # Only the log writer and reader reach open; both live entirely in memory.
    def _open(self, path, mode):
        if mode == "ab":
            return contextlib.nullcontext(self.log)
        self.assertEqual(mode, "rb")
        return io.BytesIO(self.log.getvalue() + b"worker log\n")

    # Display failure must retain ownership through verification and reaping.
    def _assert_completed(self, expected=0):
        self.assertEqual(train.run(self.args), expected)
        self.assertEqual(self.spawn.call_args.args[1], str(self.snapshot_path))
        self.spawn_verify.assert_called_once()
        self.assertEqual((self.worker.kills, self.verify.kills), (0, 0))
        self.assertEqual(self.verify.poll(), expected)

    # Exercise real consent while other supervisor tests isolate already-started jobs.
    def _review_input(self, response):
        self.confirm.side_effect = self.original_confirm
        self._patch("trlx.review.render", return_value="Settings applied to this run:\n")
        self.enterContext(patch.object(sys, "stdin", io.StringIO(response)))

    # Both UI modes and resumes must quit before model inspection or any persistent mutation.
    def test_review_quit_precedes_model_inspection_and_run_writes(self):
        self._review_input("q\n" * 4)
        rewind = self._patch("trlx.run_dirs.rewind")
        for tui in (False, True):
            for resume in (None, "memory-run/checkpoint-1"):
                with self.subTest(tui=tui, resume=resume):
                    self.args.tui = tui
                    self.cfg.args.resume_from_checkpoint = resume
                    self.assertEqual(train.run(self.args), 0)
        for operation in (self.strategy, self.check_config, self.create_run, self.snapshot, self.spawn, rewind):
            operation.assert_not_called()
        self.tui.assert_not_called()
        self.assertEqual(self.stdout.getvalue().count("Press Enter to continue or q to quit:"), 4)

    # Confirmation happens once and precedes strategy choice even when the TUI is requested.
    def test_review_enter_precedes_strategy_and_tui(self):
        self.args.tui = True
        self._review_input("\n")
        events = Mock()
        for name, operation in (("review", self.confirm), ("strategy", self.strategy),
                                ("allocate", self.create_run), ("spawn", self.spawn), ("tui", self.tui)):
            events.attach_mock(operation, name)
        self._assert_completed()
        self.assertEqual([call[0] for call in events.mock_calls], ["review", "strategy", "allocate", "spawn", "tui"])

    # Consent is never inferred from EOF or loss of the startup stdout stream.
    def test_review_io_failure_starts_no_job(self):
        self._review_input("")
        with self.assertRaisesRegex(TrlxError, "EOF"):
            train.run(self.args)
        with patch.object(sys, "stdout", BrokenStream()), self.assertRaisesRegex(TrlxError, "settings review"):
            train.run(self.args)
        self.strategy.assert_not_called()
        self.create_run.assert_not_called()
        self.spawn.assert_not_called()

    # Runtime publication controls cross process boundaries without entering snapshots.
    def test_output_controls_reach_workers_and_verify(self):
        self.args.force = True
        self.args.no_staging = True
        self._assert_completed()
        self.assertEqual(self.spawn.call_args.kwargs, {"force": True, "no_staging": True})
        self.assertEqual(self.spawn_verify.call_args.kwargs, {"force": True, "no_staging": True})
        self.assertEqual(self.snapshot.call_args.kwargs, {"no_staging": True})

    # A closed pipe at the header cannot abort workers already launched.
    def test_broken_stdout_continues_through_verification(self):
        self.enterContext(patch.object(sys, "stdout", BrokenStream()))
        self._assert_completed()
        self.assertIsNone(sys.stdout)
        self.assertIn(b"display stopped: BrokenPipeError", self.log.getvalue())
        self.assertIn("display stopped", self.stderr.buffer.getvalue().decode())

    # Losing stderr before launch must still leave a durable notice and a job.
    def test_broken_startup_stderr_continues(self):
        self.enterContext(patch.object(sys, "stderr", BrokenStream()))
        self._assert_completed()
        self.assertIsNone(sys.stderr)
        self.spawn.assert_called_once()
        self.assertIn(b"BrokenPipeError", self.log.getvalue())

    # Reporter failures obey the same protected display boundary as metric rendering.
    def test_reporter_stderr_failure_preserves_verify_status_and_durable_progress(self):
        self.verify.code = 7
        self.enterContext(patch.object(sys, "stderr", BrokenStream()))
        with Progress("trlx sft", on_error=cli._defer_training_display_error) as progress:
            self.args.progress = progress
            self._assert_completed(7)
            progress.finish("failed")
        self.assertIsInstance(progress.error, BrokenPipeError)
        self.assertIsNone(sys.stderr)
        log = self.log.getvalue().decode()
        self.assertIn("display stopped: BrokenPipeError", log)
        self.assertIn("starting training workers", log)
        self.assertIn("starting post-training verification", log)
        self.assertIn("failed: verify exited with code 7", log)
        self.assertFalse(progress._thread.is_alive())

    # Both reporter lifetimes must respect curses ownership while the job continues logging.
    def test_tui_reporters_write_waiting_feedback_to_log_without_touching_terminal(self):
        self.args.tui = True
        clock = Mock(return_value=0.0)
        children = []

        # Share a controllable clock while retaining the real child reporter and its sink.
        def reporter(*args, **kwargs):
            child = Progress(*args, **kwargs, clock=clock)
            children.append(child)
            return child

        # Advance the job and both timers while treating the terminal as owned by curses.
        def render(load):
            before = self.stderr.buffer.getvalue()
            for now in range(10, 91, 10):
                clock.return_value = float(now)
                self.args.progress.waiting()
                children[0].waiting()
                load(4)
                self.assertEqual(self.stderr.buffer.getvalue(), before)

        self.tui.side_effect = render
        self.enterContext(patch.object(train, "Progress", side_effect=reporter))
        with Progress("trlx sft", clock=clock, on_error=cli._defer_training_display_error) as progress:
            self.args.progress = progress
            self._assert_completed()
        self.tui.assert_called_once()
        self.assertEqual(len(children), 1)
        log = self.log.getvalue().decode()
        self.assertIn("[supervisor] waiting", log)
        self.assertIn("starting post-training verification", log)
        self.assertNotIn("display stopped", log)
        self.assertFalse(children[0]._thread.is_alive())

    # Losing the authoritative log asynchronously is a supervision failure requiring cleanup.
    def test_background_progress_log_failure_terminates_owned_worker(self):
        failed_write = threading.Event()
        owner = threading.current_thread()
        original_write = self.log.write

        # Fail only the background heartbeat, after ordinary startup writes have succeeded.
        def write(data):
            if threading.current_thread() is not owner and b"waiting" in data:
                failed_write.set()
                raise OSError("disk full during heartbeat")
            return original_write(data)

        # Keep a running worker alive until the independent reporter hits the bad log.
        def record(*args, **kwargs):
            self.assertTrue(failed_write.wait(3), "background reporter never attempted a log write")

        self.enterContext(patch.object(self.log, "write", side_effect=write))
        self._patch("trlx.render_lines.Stream.record", side_effect=record)
        with patch("dataset.progress.WAIT_SECONDS", 0.01):
            with Progress("trlx sft", on_error=cli._defer_training_display_error) as progress:
                self.args.progress = progress
                with self.assertRaisesRegex(TrlxError, "log.txt.*cannot record feedback.*disk full"):
                    train.run(self.args)
        self.assertTrue(failed_write.is_set())
        self.assertEqual(self.worker.kills, 1)
        self.assertGreater(self.worker.waits, 0)
        self.spawn_verify.assert_not_called()
        self.assertNotIn(b"display stopped", self.log.getvalue())

    # Failed metric reads are attempted once, while worker polling continues.
    def test_metric_read_error_disables_display(self):
        self.records.side_effect = TrlxError("metrics.jsonl: corrupt record")
        self._assert_completed()
        self.records.assert_called_once()
        self.assertIn(b"corrupt record", self.log.getvalue())

    # A renderer bug has the same job-lifetime boundary as an I/O failure.
    def test_line_render_error_preserves_verify_failure(self):
        self.verify.code = 7
        line = self._patch("trlx.render_lines.Stream.record", side_effect=ValueError("invalid metric"))
        self._assert_completed(7)
        line.assert_called_once()
        self.assertIn(b"ValueError: invalid metric", self.log.getvalue())

    # A mid-write report must not be re-read for the final failure log tail.
    def test_tui_report_error_preserves_verify_failure(self):
        self.args.tui = True
        self.verify.code = 7
        self.load.side_effect = TrlxError("preflight.json: invalid JSON")
        self._assert_completed(7)
        self.load.assert_called_once()
        self.assertIn(b"preflight.json: invalid JSON", self.log.getvalue())

    # Curses startup/drawing errors stop presentation and retain supervision.
    def test_tui_render_error_continues(self):
        self.args.tui = True
        self.tui.side_effect = RuntimeError("terminal failed")
        self._assert_completed()
        self.assertIn(b"RuntimeError: terminal failed", self.log.getvalue())

    # Reading the display config is presentation even before curses starts.
    def test_tui_config_error_continues(self):
        self.args.tui = True
        self._patch("trlx.show.load_config", side_effect=TrlxError("snapshot unreadable"))
        self._assert_completed()
        self.tui.assert_not_called()

    # An exception from process polling must reach cleanup, never display recovery.
    def _assert_poll_failure(self, tui):
        self.args.tui = tui
        error = TrlxError("process polling failed")
        self.enterContext(patch.object(launch.Job, "poll", side_effect=error))
        with self.assertRaises(TrlxError) as caught:
            train.run(self.args)
        self.assertIs(caught.exception, error)
        self.assertEqual(self.worker.kills, 1)
        self.spawn_verify.assert_not_called()
        self.assertNotIn(b"display stopped", self.log.getvalue())

    # Job.wait errors are outside the line-rendering exception handler.
    def test_line_poll_failure_remains_fatal(self):
        self._assert_poll_failure(False)

    # Job.poll errors must escape even though curses invokes the callback.
    def test_tui_poll_failure_remains_fatal(self):
        self._assert_poll_failure(True)

    # A failed verify spawn cannot be swallowed or retried as a display error.
    def _assert_verify_start_failure(self, tui):
        self.args.tui = tui
        self.worker.polls_left = 0
        error = TrlxError("cannot start verify")
        self.spawn_verify.side_effect = error
        with self.assertRaises(TrlxError) as caught:
            train.run(self.args)
        self.assertIs(caught.exception, error)
        self.spawn_verify.assert_called_once()
        self.assertNotIn(b"display stopped", self.log.getvalue())

    # Line mode must propagate verify startup errors from Job.wait.
    def test_line_verify_start_failure_remains_fatal(self):
        self._assert_verify_start_failure(False)

    # TUI mode must propagate verify startup errors from its load callback.
    def test_tui_verify_start_failure_remains_fatal(self):
        self._assert_verify_start_failure(True)

    # KeyboardInterrupt is cancellation, including when raised inside a display.
    def _assert_cancelled(self, tui):
        self.args.tui = tui
        if tui:
            self.tui.side_effect = KeyboardInterrupt
        else:
            self._patch("trlx.render_lines.Stream.record", side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            train.run(self.args)
        self.assertEqual(self.worker.kills, 1)
        self.assertEqual(self.worker.waits, 1)
        self.spawn_verify.assert_not_called()

    # A cancellation in line output must still stop and reap the worker.
    def test_line_ctrl_c_cancels(self):
        self._assert_cancelled(False)

    # A cancellation in curses must still stop and reap the worker.
    def test_tui_ctrl_c_cancels(self):
        self._assert_cancelled(True)

    # A missing authoritative log is a supervisor failure, not degraded display.
    def test_display_error_log_failure_remains_fatal(self):
        failed_display = threading.Event()

        # Fail persistence only after the presentation error, not during startup logging.
        def render(*args):
            failed_display.set()
            raise BrokenPipeError("closed")

        # Feedback flushes now occur throughout startup, so count-based failures are brittle.
        def flush():
            if failed_display.is_set():
                raise OSError("disk full")

        self._patch("trlx.render_lines.Stream.record", side_effect=render)
        self.enterContext(patch.object(self.log, "flush", side_effect=flush))
        with self.assertRaisesRegex(TrlxError, "cannot record display failure"):
            train.run(self.args)
        self.assertEqual(self.worker.kills, 1)
        self.spawn_verify.assert_not_called()

    # A failure while printing the final verdict must not replace its exit code.
    def test_final_stderr_failure_preserves_verify_status(self):
        self.verify.code = 7
        original_write = self.stderr.write

        # Leave startup and streaming output usable, then break the final notice.
        def write(text):
            if text.startswith("verify exited"):
                raise BrokenPipeError("verdict pipe closed")
            return original_write(text)

        self.enterContext(patch.object(self.stderr, "write", side_effect=write))
        self._assert_completed(7)
        self.assertIn(b"verdict pipe closed", self.log.getvalue())

    # Ordinary worker failure still stops peers and suppresses verification.
    def test_worker_failure_keeps_existing_handling(self):
        self.worker.code = 3
        self.assertEqual(train.run(self.args), 3)
        self.spawn_verify.assert_not_called()
        self.assertGreater(self.worker.waits, 0)
        self.assertNotIn(b"display stopped", self.log.getvalue())


    # Invalid checkpoint metadata must fail preflight before any run mutation.
    def test_invalid_checkpoint_prevents_rewind_and_worker_spawn(self):
        self.cfg.args.resume_from_checkpoint = "memory-run/checkpoint-20"
        self.check_config.side_effect = lambda *args, **kwargs: train.preflight._check_resume(*args)
        self._patch("trlx.run_dirs.inspect_checkpoint", side_effect=TrlxError("invalid trainer_state.json"))
        rewind = self._patch("trlx.run_dirs.rewind")
        with self.assertRaisesRegex(TrlxError, "invalid trainer_state.json"):
            train.run(self.args)
        self.create_run.assert_not_called()
        rewind.assert_not_called()
        self.snapshot.assert_not_called()
        self.spawn.assert_not_called()
        self.assertEqual(self.log.getvalue(), b"")

    # Reinspection under the directory lock still precedes destructive cleanup.
    def test_checkpoint_reinspection_failure_preserves_history(self):
        self.cfg.args.resume_from_checkpoint = "memory-run/checkpoint-20"
        self.log.write(b"old history\n")
        self._patch("trlx.run_dirs.inspect_checkpoint", side_effect=TrlxError("checkpoint disappeared"))
        rewind = self._patch("trlx.run_dirs.rewind")
        with self.assertRaisesRegex(TrlxError, "checkpoint disappeared"):
            train.run(self.args)
        rewind.assert_not_called()
        self.snapshot.assert_not_called()
        self.spawn.assert_not_called()
        self.assertEqual(self.log.getvalue(), b"old history\n")

    # Workers cannot append progress until cleanup and the replacement snapshot finish.
    def test_resume_cleanup_precedes_snapshot_and_worker_spawn(self):
        self.cfg.args.resume_from_checkpoint = "memory-run/checkpoint-20"
        resume = run_dirs.Resume(pathlib.Path(self.cfg.args.resume_from_checkpoint), 20)
        inspect = self._patch("trlx.run_dirs.inspect_checkpoint", return_value=resume)
        rewind = self._patch("trlx.run_dirs.rewind", return_value=(0, "resume marker"))
        calls = Mock()
        for name, mocked in (("inspect", inspect), ("rewind", rewind),
                             ("snapshot", self.snapshot), ("spawn", self.spawn)):
            calls.attach_mock(mocked, name)
        self._assert_completed()
        self.assertEqual([call[0] for call in calls.mock_calls],
                         ["inspect", "rewind", "snapshot", "spawn"])
        rewind.assert_called_once_with(resume, no_staging=False)
        self.assertIn(b"resume marker", self.log.getvalue())

    # Live output skips retained rows and bytes but derives changes from full history.
    def test_resume_display_uses_history_without_reprinting_it(self):
        self.cfg.args.resume_from_checkpoint = "memory-run/checkpoint-20"
        resume = run_dirs.Resume(pathlib.Path(self.cfg.args.resume_from_checkpoint), 20)
        self._patch("trlx.run_dirs.inspect_checkpoint", return_value=resume)
        self._patch("trlx.run_dirs.rewind", return_value=(1, "resume marker"))
        self.log.write(b"old history\n")
        retained = dict(self.records.return_value[0], step=20)
        continued = dict(retained, step=21, log={"loss": 0.75})
        self.records.return_value = [retained, continued]
        line = self._patch("trlx.render_lines.Stream.record", autospec=True,
                           side_effect=train.render_lines.Stream.record)

        # New child diagnostics arrive through the collector, never by replaying log bytes.
        def spawn(*args, **kwargs):
            args[4].accept("rank 0", {"kind": "raw", "message": "worker log"})
            return [self.worker]

        self.spawn.side_effect = spawn
        self._assert_completed()
        line.assert_called_once()
        row = line.call_args.args[2]
        self.assertEqual(row.step, 21)
        self.assertEqual(row.cells["loss"].change, -0.25)
        shown_log = self.stderr.buffer.getvalue()
        self.assertNotIn(b"old history", shown_log)
        self.assertIn(b"worker log", shown_log)
        self.assertTrue(self.log.getvalue().startswith(b"old history\n"))
        self.assertIn(b"resume marker\n", self.log.getvalue())


class WorkerCleanup(unittest.TestCase):
    # Mock the distributed boundary: these lifecycle checks require neither CUDA nor a group.
    def setUp(self):
        self.args = types.SimpleNamespace(_rank=0)
        self.work = self.enterContext(patch("trlx.train._train_worker", return_value=0))
        self.available = self.enterContext(patch("torch.distributed.is_available", return_value=True))
        self.initialized = self.enterContext(patch("torch.distributed.is_initialized", return_value=True))
        self.destroy = self.enterContext(patch("torch.distributed.destroy_process_group"))
        self.logger = Mock()
        self.enterContext(patch("trlx.train.logging.getLogger", return_value=self.logger))

    # Successful training releases its initialized process group before returning.
    def test_success_destroys_group(self):
        self.assertEqual(train._worker(self.args), 0)
        self.work.assert_called_once_with(self.args)
        self.destroy.assert_called_once_with()

    # A worker that never joined a group must not attempt distributed teardown.
    def test_uninitialized_group_needs_no_cleanup(self):
        self.initialized.return_value = False
        self.assertEqual(train._worker(self.args), 0)
        self.destroy.assert_not_called()

    # Distributed support can be absent even though ordinary training is available.
    def test_unavailable_distributed_needs_no_cleanup(self):
        self.available.return_value = False
        self.assertEqual(train._worker(self.args), 0)
        self.initialized.assert_not_called()
        self.destroy.assert_not_called()

    # Teardown also owns construction failures and user cancellation, without an extra barrier.
    def test_training_failure_and_interrupt_still_destroy_group(self):
        for error in (ValueError("training failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.destroy.reset_mock()
                self.work.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    train._worker(self.args)
                self.assertIs(caught.exception, error)
                self.destroy.assert_called_once_with()

    # Cleanup failure must remain actionable when it is the only failure.
    def test_cleanup_failure_after_success_is_error(self):
        self.destroy.side_effect = RuntimeError("NCCL cleanup failed")
        with self.assertRaisesRegex(TrlxError, "cannot shut down distributed training: NCCL cleanup failed"):
            train._worker(self.args)

    # Secondary teardown and logging errors cannot replace the original training failure.
    def test_original_failure_survives_cleanup_and_reporting_failures(self):
        original = ValueError("original training failure")
        self.work.side_effect = original
        self.destroy.side_effect = RuntimeError("NCCL cleanup failed")
        for reporting_error in (None, OSError("log unavailable")):
            with self.subTest(reporting_error=reporting_error):
                self.logger.error.reset_mock()
                self.logger.error.side_effect = reporting_error
                with self.assertRaises(ValueError) as caught:
                    train._worker(self.args)
                self.assertIs(caught.exception, original)
                self.logger.error.assert_called_once()


class Shutdown(unittest.TestCase):
    # Exercise real pipe buffering and Python's final flush in isolated children.
    def test_broken_standard_streams_preserve_process_exit(self):
        script = """import io, sys
from trlx.train import _report_display_error
sys.stdin.read(1)
try:
    print('probe', file=getattr(sys, sys.argv[1]), flush=True)
except BrokenPipeError as error:
    log = io.BytesIO()
    _report_display_error(error, log, 'memory-run/log.txt')
    assert b'BrokenPipeError' in log.getvalue()
sys.exit(int(sys.argv[2]))
"""
        for stream_name, code in (("stdout", 0), ("stderr", 7)):
            with self.subTest(stream=stream_name):
                proc = subprocess.Popen(
                    [sys.executable, "-B", "-c", script, stream_name, str(code)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                try:
                    # The child waits until its output reader is certainly closed.
                    getattr(proc, stream_name).close()
                    proc.stdin.write(b"x")
                    proc.stdin.close()
                    self.assertEqual(proc.wait(timeout=60), code)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=10)
                    for stream in (proc.stdin, proc.stdout, proc.stderr):
                        stream.close()


if __name__ == "__main__":
    unittest.main()
