"""Explicit command progress, with waiting notices independent of blocking work.

Counts describe completed work only. Waiting notices establish that the reporter
is alive, not that the operation it observes is advancing.
"""

import contextlib
import sys
import threading
import time

from dataset.failures import annotate

# Display constants, shared by both CLIs. Frequent counters are coalesced, while
# stage changes, retries, and final outcomes are always printed immediately.
WAIT_SECONDS = 10.0
COUNT_SECONDS = 1.0


class Progress:
    # The command owns this reporter and closes it before its output sink closes.
    def __init__(self, command, *, emit=None, on_error=None, clock=time.monotonic, events=None):
        self.command = command
        self.emit = emit if emit is not None else self._stderr
        self.on_error = on_error
        self.events = events
        self.clock = clock
        self.started = clock()
        self.last_output = self.started
        self.active = []
        self.sequences = {}
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
        if isinstance(exc, Exception):
            annotate(exc, context={"command": self.command})
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
    def _write(self, message, *, kind="diagnostic", activity=None):
        with self._lock:
            if self.error is not None or self._suspended:
                return
            try:
                if self.events is None:
                    self.emit(f"{self.command}: {message}")
                else:
                    event = {"kind": kind, "message": message, "command": self.command}
                    if activity is not None:
                        event.update(label=activity.label, completed=activity.completed,
                                     total=activity.total, unit=activity.unit, visible=activity.visible,
                                     measured=activity.last_progress, parent=activity.parent,
                                     sequence=activity.sequence)
                    self.events(event)
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
                extra = f"; {current.waiting_detail()}" if current.waiting_detail is not None else ""
                self._write(f"waiting: {detail}; stage elapsed {now - current.started:.1f}s; {since}{extra}", kind="waiting")
            else:
                self._write(f"waiting for command; elapsed {now - self.started:.1f}s; no active stage reported", kind="waiting")

    # Event.wait permits prompt shutdown even when the heartbeat interval is long.
    def _watch(self):
        while not self._stop.wait(0.25):
            self.waiting()


class Stage:
    # A disabled stage preserves the same call interface for silent library callers.
    def __init__(self, progress, label, total=None, unit="items", *, visible=None):
        self.reporter = progress.reporter if isinstance(progress, Stage) else progress
        self.label = label
        self.total = total
        self.unit = unit
        # Measured dataset operations are useful by default; internal bookkeeping
        # stays diagnostic unless its owner explicitly identifies a public phase.
        self.visible = (total is not None or unit != "items") if visible is None else visible
        self.completed = 0
        self.started = 0.0
        self.last_progress = None
        self.parent = None
        self.sequence = 0
        # Operation-owned diagnostics are read only for heartbeats, never counted as progress.
        self.waiting_detail = None

    # Publish the boundary before executing the operation, including blocking calls.
    def __enter__(self):
        if self.reporter is not None:
            with self.reporter._lock:
                self.started = self.reporter.clock()
                self.parent = self.reporter.active[-1].label if self.reporter.active else None
                key = (self.parent, self.label)
                self.sequence = self.reporter.sequences.get(key, 0) + 1
                self.reporter.sequences[key] = self.sequence
                self.reporter.active.append(self)
                self.reporter._write(self._description(), kind="start", activity=self)
        return self

    # A nested stage restores the enclosing operation, including on failure.
    def __exit__(self, exc_type, exc, traceback):
        # Capture before popping the stage; silent library calls need the same
        # operation evidence as visible commands. Inner annotations win.
        if isinstance(exc, Exception):
            annotate(exc, context={"operation": self.label, "completed": self.completed,
                                   "total": self.total, "unit": self.unit})
        if self.reporter is not None:
            with self.reporter._lock:
                outcome = "failed" if exc_type else "finished"
                self.reporter._write(f"{self._description()}; {outcome} in {self.reporter.clock() - self.started:.1f}s",
                                     kind="end", activity=self)
                self.reporter.active.remove(self)

    # Unknown totals remain unknown; no percentage is inferred from elapsed time.
    def _description(self):
        count = (f"; {self.completed}/{self.total} {self.unit}" if self.total is not None
                 else f"; {self.completed} {self.unit}" if self.completed else "")
        return self.label + count

    # Increment only after the operation has actually completed its unit of work.
    def advance(self, count=1):
        if self.reporter is None:
            self.completed += count
        if self.reporter is not None:
            with self.reporter._lock:
                self.update(self.completed + count)

    # Callers may provide authoritative absolute counts, such as TrainerState.global_step.
    def update(self, completed, total=None):
        if self.reporter is None:
            self.completed = completed
            if total is not None:
                self.total = total
        if self.reporter is not None:
            with self.reporter._lock:
                if total is not None:
                    self.total = total
                if completed != self.completed:
                    self.last_progress = self.reporter.clock()
                self.completed = completed
                if (self.reporter.events is not None and self.unit == "steps") or self.reporter.clock() - self.reporter.last_output >= COUNT_SECONDS:
                    # Structured consumers need each measured step, including steps
                    # between Trainer.log calls; they own terminal coalescing.
                    self.reporter._write(self._description(), kind="count", activity=self)

    # Retry and diagnostic events bypass counter throttling, without claiming progress.
    def note(self, message):
        if self.reporter is not None:
            self.reporter._write(f"{self.label}: {message}", kind="note", activity=self)


# Keep reporter plumbing explicit; importing this module starts no background work.
def stage(progress, label, total=None, unit="items", *, visible=None):
    return Stage(progress, label, total, unit, visible=visible)
