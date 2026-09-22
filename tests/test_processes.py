"""Real POSIX signal/ownership regressions for supervisor shutdown."""

import os
import subprocess
import sys
import textwrap
import unittest


@unittest.skipUnless(os.name == "posix", "process groups require POSIX")
class Shutdown(unittest.TestCase):
    # Isolate signal handlers from the test runner and bound every regression.
    def run_script(self, source):
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    # User cancellation must outlive the failure grace and leave helpers alone.
    def test_cooperative_leader_finishes_helper_after_grace(self):
        self.run_script('''
            import subprocess, sys
            from trlx import processes
            child = subprocess.Popen([sys.executable, "-u", "-c", """
            import os, signal, time
            cancelled = False
            # Record cancellation while the helper completes its owned work.
            def cancel(signum, frame):
                global cancelled
                cancelled = True
            signal.signal(signal.SIGINT, cancel)
            pid = os.fork()
            if pid == 0:
                signal.signal(signal.SIGINT, lambda *args: os._exit(41))
                time.sleep(.4)
                os._exit(0)
            print('ready', flush=True)
            while not cancelled:
                time.sleep(.01)
            _, status = os.waitpid(pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0, status
            """], stdout=subprocess.PIPE, text=True, start_new_session=True)
            try:
                assert child.stdout.readline().strip() == 'ready'
                processes.register(child, 'rank 0')
                child._trlx_cancel_ready = True
                processes.GRACE_SECONDS = .01
                notices = []
                assert processes.stop([child], notices.append, cancelled=True) == []
                assert child.returncode == 0, child.returncode
                assert child._trlx_shutdown_done
                assert not any('SIGTERM' in item for item in notices)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        ''')

    # A second Ctrl+C overrides a worker that never reaches a safe boundary.
    def test_second_interrupt_forces_group(self):
        self.run_script('''
            import os, signal, subprocess, sys, threading
            from trlx import processes
            child = subprocess.Popen([sys.executable, '-u', '-c',
                "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('ready'); time.sleep(60)"],
                stdout=subprocess.PIPE, text=True, start_new_session=True)
            try:
                assert child.stdout.readline().strip() == 'ready'
                processes.register(child, 'rank 0')
                child._trlx_cancel_ready = True
                timer = threading.Timer(.3, lambda: os.kill(os.getpid(), signal.SIGINT))
                timer.start()
                notices = []
                assert processes.stop([child], notices.append, cancelled=True) == []
                timer.join()
                assert child.returncode == -signal.SIGKILL
                assert child._trlx_shutdown_done
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        ''')

    # Failure cleanup still escalates without requiring a second user signal.
    def test_failure_cleanup_remains_bounded(self):
        self.run_script('''
            import signal, subprocess, sys
            from trlx import processes
            child = subprocess.Popen([sys.executable, '-u', '-c',
                "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('ready'); time.sleep(60)"],
                stdout=subprocess.PIPE, text=True, start_new_session=True)
            try:
                assert child.stdout.readline().strip() == 'ready'
                processes.register(child, 'rank 0')
                processes.GRACE_SECONDS = .01
                assert processes.stop([child], lambda message: None) == []
                assert child.returncode == -signal.SIGTERM
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        ''')

    # Inner Popen ownership scopes transfer cancellation to the whole cohort.
    def test_nested_deferral_completes_outer_body(self):
        self.run_script('''
            import os, signal
            from trlx import processes
            completed = []
            try:
                with processes.defer_interrupt():
                    with processes.defer_interrupt():
                        os.kill(os.getpid(), signal.SIGINT)
                    completed.append('second rank')
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('interrupt lost')
            assert completed == ['second rank']
        ''')

    # Cancellation before handler installation waits for the structured acknowledgement.
    def test_startup_waits_for_worker_handler(self):
        self.run_script('''
            import io, os, sys
            from trlx import feedback
            log = io.BytesIO()
            with feedback.Collector(log, 'memory-log', display=False) as collector:
                child = collector.spawn([sys.executable, '-u', '-c', """
            import json, os, signal, sys, time
            time.sleep(.3)
            signal.signal(signal.SIGINT, lambda *args: sys.exit(130))
            message = json.dumps({'kind': 'cancellation_ready', 'message': 'handler ready'}) + chr(10)
            os.write(int(os.environ['TRLX_FEEDBACK_FD']), message.encode())
            time.sleep(60)
            """], os.environ, 'rank 0')
                assert collector.stop(cancelled=True) == []
                assert child.returncode == 130, child.returncode
                assert child._trlx_cancel_ready
            assert b'handler ready' in log.getvalue()
        ''')

    # A peer failure removes the possibility of coordinated progress during cancellation.
    def test_failed_peer_switches_to_bounded_cleanup(self):
        self.run_script('''
            import signal, subprocess, sys
            from trlx import processes
            child = subprocess.Popen([sys.executable, '-u', '-c',
                "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); print('ready'); time.sleep(60)"],
                stdout=subprocess.PIPE, text=True, start_new_session=True)
            failed = subprocess.Popen([sys.executable, '-c',
                "import time,sys; time.sleep(.2); sys.exit(7)"], start_new_session=True)
            try:
                assert child.stdout.readline().strip() == 'ready'
                for rank, process in enumerate((child, failed)):
                    processes.register(process, f'rank {rank}')
                child._trlx_cancel_ready = True
                processes.GRACE_SECONDS = .01
                notices = []
                assert processes.stop([child, failed], notices.append, cancelled=True) == []
                assert failed.returncode == 7
                assert child.returncode == -signal.SIGTERM
                assert any('worker failed' in item for item in notices)
            finally:
                for process in (child, failed):
                    if process.poll() is None:
                        process.kill()
                    process.wait()
        ''')
