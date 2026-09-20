"""Explicit command progress, with waiting notices independent of blocking work.

Counts describe completed work only. Waiting notices establish that the reporter
is alive, not that the operation it observes is advancing.
"""

import contextlib
import sys
import threading
import time

# Display constants, shared by both CLIs. Frequent counters are coalesced, while
# stage changes, retries, and final outcomes are always printed immediately.
WAIT_SECONDS = 10.0
COUNT_SECONDS = 1.0


class Progress:
    # The command owns this reporter and closes it before its output sink closes.
    def __init__(self, command, *, emit=None, on_error=None, clock=time.monotonic):
        self.command = command
        self.emit = emit if emit is not None else self._stderr
        self.on_error = on_error
        self.clock = clock
        self.started = clock()
        self.last_output = self.started
        self.active = []
        self.error = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._finished = False
        self._suspended = 0

    # Flush each line so redirection does not hide activity behind buffering.
    @staticmethod
    def _stderr(line):
        # print(file=None) would silently redirect diagnostics into metric stdout.
        if sys.stderr is None:
            raise BrokenPipeError("progress stderr is unavailable")
        print(line, file=sys.stderr, flush=True)

    # Start feedback before dispatching expensive command work.
    def __enter__(self):
        self._write("starting")
        self._thread = threading.Thread(target=self._watch, name="command-progress", daemon=True)
        self._thread.start()
        return self

    # Stop the watcher before command resources close; never hide an original error.
    def __exit__(self, exc_type, exc, traceback):
        if not self._finished:
            self.finish("interrupted" if exc_type and issubclass(exc_type, KeyboardInterrupt)
                        else "failed" if exc_type else "completed")
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.error is not None and self.on_error is None and exc_type is None:
            raise self.error

    # Finish explicitly when a command returns a failure code without raising.
    def finish(self, outcome="completed"):
        with self._lock:
            if not self._finished:
                self._write(f"{outcome}; elapsed {self.clock() - self.started:.1f}s")
                self._finished = True
                self._stop.set()

    # Switch training from terminal startup feedback to its authoritative log.
    def set_sink(self, emit, on_error=None):
        with self._lock:
            self.emit = emit
            self.on_error = on_error

    # Supervisors check asynchronous log failures on their normal process poll.
    def check_error(self):
        with self._lock:
            if self.error is not None and self.on_error is None:
                raise self.error

    # Worker log output is already feedback, but is not a measured unit of our own stage.
    def output_seen(self):
        with self._lock:
            self.last_output = self.clock()

    # Curses owns the terminal while an interactive artifact viewer is open.
    @contextlib.contextmanager
    def suspended(self):
        with self._lock:
            self._suspended += 1
        try:
            yield
        finally:
            with self._lock:
                self._suspended -= 1
                self.last_output = self.clock()

    # All sinks, including concurrent request notices, share one serialization lock.
    def _write(self, message):
        with self._lock:
            if self.error is not None or self._suspended:
                return
            try:
                self.emit(f"{self.command}: {message}")
            except Exception as error:
                self.error = error
                if self.on_error is not None:
                    self.on_error(error)
            self.last_output = self.clock()

    # Tests drive this directly with a fake clock; production calls it from the watcher.
    def waiting(self):
        with self._lock:
            now = self.clock()
            if self._finished or self._suspended or now - self.last_output < WAIT_SECONDS:
                return
            if self.active:
                current = self.active[-1]
                detail = current._description()
                since = (f"last measured progress {now - current.last_progress:.1f}s ago"
                         if current.last_progress is not None else "no measured progress yet")
                self._write(f"waiting: {detail}; stage elapsed {now - current.started:.1f}s; {since}")
            else:
                self._write(f"waiting for command; elapsed {now - self.started:.1f}s; no active stage reported")

    # Event.wait permits prompt shutdown even when the heartbeat interval is long.
    def _watch(self):
        while not self._stop.wait(0.25):
            self.waiting()


class Stage:
    # A disabled stage preserves the same call interface for silent library callers.
    def __init__(self, progress, label, total=None, unit="items"):
        self.reporter = progress.reporter if isinstance(progress, Stage) else progress
        self.label = label
        self.total = total
        self.unit = unit
        self.completed = 0
        self.started = 0.0
        self.last_progress = None

    # Publish the boundary before executing the operation, including blocking calls.
    def __enter__(self):
        if self.reporter is not None:
            with self.reporter._lock:
                self.started = self.reporter.clock()
                self.reporter.active.append(self)
                self.reporter._write(self._description())
        return self

    # A nested stage restores the enclosing operation, including on failure.
    def __exit__(self, exc_type, exc, traceback):
        if self.reporter is not None:
            with self.reporter._lock:
                outcome = "failed" if exc_type else "finished"
                self.reporter._write(f"{self._description()}; {outcome} in {self.reporter.clock() - self.started:.1f}s")
                self.reporter.active.remove(self)

    # Unknown totals remain unknown; no percentage is inferred from elapsed time.
    def _description(self):
        count = (f"; {self.completed}/{self.total} {self.unit}" if self.total is not None
                 else f"; {self.completed} {self.unit}" if self.completed else "")
        return self.label + count

    # Increment only after the operation has actually completed its unit of work.
    def advance(self, count=1):
        if self.reporter is not None:
            with self.reporter._lock:
                self.update(self.completed + count)

    # Callers may provide authoritative absolute counts, such as TrainerState.global_step.
    def update(self, completed, total=None):
        if self.reporter is not None:
            with self.reporter._lock:
                if total is not None:
                    self.total = total
                if completed != self.completed:
                    self.last_progress = self.reporter.clock()
                self.completed = completed
                if self.reporter.clock() - self.reporter.last_output >= COUNT_SECONDS:
                    self.reporter._write(self._description())

    # Retry and diagnostic events bypass counter throttling, without claiming progress.
    def note(self, message):
        if self.reporter is not None:
            self.reporter._write(f"{self.label}: {message}")


# Keep reporter plumbing explicit; importing this module starts no background work.
def stage(progress, label, total=None, unit="items"):
    return Stage(progress, label, total, unit)
