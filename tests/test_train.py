"""Supervisor behavior when presentation fails; no models or run files are created."""

import contextlib
import io
import pathlib
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from trlx import TrlxError, launch, render_tui, train


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
        cfg = types.SimpleNamespace(
            args=types.SimpleNamespace(run_name="memory"), ranges={"loss": (0, 2)},
            model=types.SimpleNamespace(path="memory-model"), verify_prompts=None,
        )
        self._patch("trlx.train.config_mod.load", return_value=cfg)
        self._patch("trlx.train.preflight.check_config")
        self._patch("trlx.launch.select_gpus", return_value=([0], 1))
        self._patch("trlx.launch.physical_ids", return_value=["synthetic-device"])
        self._patch("trlx.launch.choose_strategy", return_value=("single", "test"))
        self._patch("trlx.train._create_run_dir", return_value=pathlib.Path("memory-run"))
        self._patch("trlx.train._write_snapshot")
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

    # Restore every patched boundary even when a failure propagates out of run.
    def _patch(self, target, **kwargs):
        return self.enterContext(patch(target, **kwargs))

    # Only the log writer and reader reach open; both live entirely in memory.
    def _open(self, path, mode):
        if mode == "ab":
            return contextlib.nullcontext(self.log)
        self.assertEqual(mode, "rb")
        return io.BytesIO(b"worker log\n")

    # Display failure must retain ownership through verification and reaping.
    def _assert_completed(self, expected=0):
        self.assertEqual(train.run(self.args), expected)
        self.spawn_verify.assert_called_once()
        self.assertEqual((self.worker.kills, self.verify.kills), (0, 0))
        self.assertEqual(self.verify.poll(), expected)

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

    # Failed metric reads are attempted once, while worker polling continues.
    def test_metric_read_error_disables_display(self):
        self.records.side_effect = TrlxError("metrics.jsonl: corrupt record")
        self._assert_completed()
        self.records.assert_called_once()
        self.assertIn(b"corrupt record", self.log.getvalue())

    # A renderer bug has the same job-lifetime boundary as an I/O failure.
    def test_line_render_error_preserves_verify_failure(self):
        self.verify.code = 7
        line = self._patch("trlx.render_lines.line", side_effect=ValueError("invalid metric"))
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
            self._patch("trlx.render_lines.header", side_effect=KeyboardInterrupt)
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
        self._patch("trlx.render_lines.header", side_effect=BrokenPipeError("closed"))
        self.enterContext(patch.object(self.log, "flush", side_effect=[None, OSError("disk full")]))
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
