"""Training feedback transport and terminal policy; metric values live in metrics.jsonl."""

import codecs
import collections
import json
import logging
import os
import select
import subprocess
import sys
import threading
import time

from trlx import TrlxError

# Internal inherited descriptor, never a persisted setting or operator override.
PIPE_ENV = "TRLX_FEEDBACK_FD"
# Display constants: one quiet-period notice and coalesced measured counters.
WAIT_SECONDS = 30.0
COUNT_SECONDS = 1.0


class Client:
    # A dedicated pipe keeps arbitrary library output from masquerading as events.
    def __init__(self, fd):
        self.fd = fd
        os.set_inheritable(fd, False)
        self.lock = threading.Lock()

    # Serialize full records, including messages larger than an atomic pipe write.
    def __call__(self, event):
        data = (json.dumps(event, ensure_ascii=True) + "\n").encode("utf-8")
        with self.lock:
            while data:
                data = data[os.write(self.fd, data):]

    # The CLI owns this descriptor; it closes after its last completion event.
    def close(self):
        os.close(self.fd)


# Consume the internal descriptor once so subsequently launched commands cannot reuse it.
def connect():
    value = os.environ.pop(PIPE_ENV, None)
    return Client(int(value)) if value is not None else None


class LogHandler(logging.Handler):
    # Library levels remain authoritative; the handler only supplies event identity.
    def __init__(self, events):
        super().__init__()
        self.events = events

    # Unknown warnings and exception details remain visible, with their original text.
    def emit(self, record):
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + logging.Formatter().formatException(record.exc_info)
        self.events({"kind": "warning" if record.levelno >= logging.WARNING else "library",
                     "logger": record.name, "level": record.levelname, "message": message})


# Own console delivery once while retaining file handlers and configured logger levels.
def configure_logging(events=None):
    logging.captureWarnings(True)
    for name in ("transformers", "trl", "py.warnings"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, LogHandler) or (isinstance(handler, logging.StreamHandler)
                    and not isinstance(handler, logging.FileHandler)):
                logger.removeHandler(handler)
        handler = LogHandler(events) if events is not None else logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False


# Library bars describe replicated preparation work; rank zero supplies their
# measured counters. Other ranks retain typed phases, failures, and all diagnostics.
def configure_worker_progress(rank):
    if rank != 0:
        from datasets.utils import logging as dataset_logging
        from transformers.utils import logging as model_logging

        dataset_logging.disable_progress_bar()
        model_logging.disable_progress_bar()


class Startup:
    # Before run allocation, hold diagnostics in memory and show only public feedback.
    def __init__(self):
        self.pending = []
        self.collector = None
        self.view = View(self._print)

    # Startup has no worker renderer yet, but still reports genuinely long waits.
    @staticmethod
    def _print(text):
        if sys.stderr is None:
            raise BrokenPipeError("training stderr is unavailable")
        print(text, file=sys.stderr, flush=True)

    # The outer command alone announces startup and the final command outcome.
    def __call__(self, event):
        if self.collector is None:
            self.pending.append(event)
        else:
            self.collector.accept("supervisor", event)
        message = event["message"]
        if event["kind"] == "diagnostic" and (
                message == "starting" or message.startswith(("completed;", "failed;", "cancelled;", "interrupted;"))):
            self._print(f"{event['command']}: {message}")
        elif self.collector is None:
            self.view.consume(dict(event, source="supervisor"))
            if event["kind"] == "waiting":
                self.view.waiting()

    # Flush startup diagnostics once log.txt exists, without replaying them to the screen.
    def attach(self, collector):
        self.collector = collector
        for event in self.pending:
            collector.accept("supervisor", event, display=False)
        self.pending.clear()

    # Run ownership ends before the outer command emits its final elapsed-time message.
    def detach(self):
        self.collector = None


class Collector:
    # Readers never write to the terminal; log persistence outlives display failure.
    def __init__(self, log_file, log_path, *, display=True):
        self.log_file = log_file
        self.log_path = log_path
        self.display = display
        self.pending = collections.deque()
        self.lock = threading.RLock()
        self.readers = []
        self.error = None

    # The log file's enclosing context outlives every child pipe reader.
    def __enter__(self):
        return self

    # Also clean up children if launch fails before a Job takes ownership.
    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            for process, stop, thread, stream in self.readers:
                if process.poll() is None:
                    process.kill()
                process.wait()
        try:
            self.finish()
        except Exception as error:
            if exc is None:
                raise
            exc.add_note(str(error))

    # Collection and log writes share a lock, preserving complete source-labelled lines.
    def accept(self, source, event, *, display=True):
        with self.lock:
            try:
                identity = (f"{event['level']} {event['logger']}: "
                            if "logger" in event and "level" in event else "")
                text = f"[{source}] {identity}{event['message']}\n"
                self.log_file.write(text.encode("utf-8"))
                self.log_file.flush()
                if self.display and display:
                    self.pending.append(dict(event, source=source))
            except Exception as error:
                self.error = TrlxError(f"{self.log_path}: cannot record feedback: {error}")

    # Start both drainers immediately, before any child can fill either pipe.
    def spawn(self, command, env, source):
        read_fd, write_fd = os.pipe()
        try:
            child_env = dict(env, **{PIPE_ENV: str(write_fd), "PYTHONUNBUFFERED": "1"})
            process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, pass_fds=(write_fd,))
        except BaseException:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        stop = threading.Event()
        for fd, structured in ((read_fd, True), (process.stdout.fileno(), False)):
            thread = threading.Thread(target=self._read, args=(fd, structured, source, stop), daemon=True)
            self.readers.append((process, stop, thread, fd if structured else process.stdout))
            thread.start()
        return process

    # Decode across read boundaries and retain final unterminated diagnostics. A dead
    # worker's descendants must not hold supervision open merely by inheriting stdout.
    def _read(self, fd, structured, source, stop):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        try:
            while True:
                ready = select.select([fd], [], [], 0.1)[0]
                if not ready:
                    if stop.is_set():
                        break
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                pending += decoder.decode(chunk)
                separator = "\n" if structured else None
                # Raw carriage-return progress is isolated per process, never spliced
                # into another rank's output. Every update is retained in the log.
                while "\n" in pending or (separator is None and "\r" in pending):
                    positions = [p for p in (pending.find("\n"), pending.find("\r") if separator is None else -1) if p >= 0]
                    index = min(positions)
                    line, boundary, pending = pending[:index], pending[index], pending[index + 1:]
                    self._line(source, line, structured, boundary == "\r")
            pending += decoder.decode(b"", final=True)
            if pending:
                self._line(source, pending, structured, False)
        except Exception as error:
            with self.lock:
                self.error = TrlxError(f"{source}: cannot collect feedback: {error}")

    # Only the dedicated pipe accepts JSON; arbitrary child output is always raw text.
    def _line(self, source, line, structured, carriage):
        if structured:
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("message"), str):
                raise ValueError("invalid feedback record")
        else:
            if not line:
                return
            # A carriage return is not evidence that text is disposable progress:
            # third-party diagnostics have no type contract and stay visible.
            event = {"kind": "raw", "message": line}
        self.accept(source, event)

    # A log/transport failure belongs to supervision, never the terminal error handler.
    def check(self):
        for process, stop, thread, stream in self.readers:
            if process.poll() is not None:
                stop.set()
        if self.error is not None:
            raise self.error

    # Swap the queue so rendering cannot block pipe readers while writing to a terminal.
    def take(self):
        with self.lock:
            result = list(self.pending)
            self.pending.clear()
            return result

    # Continue draining and logging, without accumulating undisplayable output in memory.
    def disable_display(self):
        with self.lock:
            self.display = False
            self.pending.clear()

    # Call after child exit or termination, before the final display tick and log close.
    def finish(self):
        for process, stop, thread, stream in self.readers:
            stop.set()
        for process, stop, thread, stream in self.readers:
            thread.join()
            if isinstance(stream, int):
                os.close(stream)
            else:
                stream.close()
        self.readers.clear()
        self.check()


class View:
    # State is presentation-only: actual training metric records stay in metrics.jsonl.
    def __init__(self, emit, *, clock=time.monotonic):
        self.emit = emit
        self.clock = clock
        self.last_feedback = clock()
        self.last_notice = self.last_feedback
        self.stages = {}
        self.warnings = {}
        self.shown = set()
        self.counters = {}

    # Substantive messages restart the quiet timer; waiting notices never imply progress.
    def write(self, text):
        self.emit(text)
        self.last_feedback = self.clock()

    # Only this precisely identified dependency deprecation is diagnostic-only.
    @staticmethod
    def _deprecation(event):
        return (event.get("logger") == "py.warnings"
                and "FutureWarning: `torch.distributed.all_gather_into_tensor` is deprecated." in event["message"])

    # Record warning repetitions while immediately showing each new message.
    def consume(self, event):
        source, kind, message = event["source"], event["kind"], event["message"]
        now = self.clock()
        if kind == "warning":
            if self._deprecation(event):
                return
            key = (event.get("logger"), event.get("level"), message)
            if key not in self.warnings:
                self.warnings[key] = [0, set(), 0]
                self.write(f"{event.get('level', 'WARNING')} [{source}] {event.get('logger', '')}: {message}")
            entry = self.warnings[key]
            entry[0] += 1
            entry[1].add(source)
            return
        if kind in ("raw", "library"):
            self.write(f"[{source}] {message}")
            return
        label = event.get("label")
        if kind == "start":
            self.stages.setdefault(source, []).append(dict(event, since=now))
        elif kind == "count":
            for current in reversed(self.stages.get(source, [])):
                if current.get("label") == label:
                    if current.get("completed") != event.get("completed"):
                        self.last_feedback = now
                    current.update(event)
                    break
        elif kind == "end":
            stack = self.stages.get(source, [])
            for index in range(len(stack) - 1, -1, -1):
                if stack[index].get("label") == label:
                    del stack[index]
                    break
        if not event.get("visible"):
            return
        # Optimizer counts maintain liveness but metric rows already display their values.
        if kind == "count" and event.get("unit") == "steps":
            return
        # Equivalent rank operations share a key, but repeated evaluation passes
        # and base/checkpoint generation are distinct occurrences, never duplicates.
        key = ("workers" if source.startswith("rank ") else source,
               event.get("parent"), event.get("sequence"), kind, label,
               event.get("completed"), message if kind == "note" else None)
        if key in self.shown:
            return
        if kind == "count":
            if now - self.counters.get(label, 0) < COUNT_SECONDS:
                return
            self.counters[label] = now
        if kind in ("start", "end", "count", "note"):
            self.shown.add(key)
            self.write(message)

    # Repetitions are attributed without printing the warning text again for every rank.
    def warning_summary(self):
        for (logger, level, message), entry in self.warnings.items():
            count, sources, reported = entry
            if count > 1 and count != reported:
                self.write(f"{level} repeated {count} times across {', '.join(sorted(sources))}: {message}")
                entry[2] = count

    # One notice reports each distinct active operation and the ranks still in it.
    def waiting(self):
        # Training owns the metrics display until every rank leaves its step stage.
        # Inspect whole stacks so nested evaluation/checkpoint pauses stay quiet too;
        # collection has already retained the raw waiting diagnostics in log.txt.
        if any(event.get("unit") == "steps"
               for source, stack in self.stages.items() if source.startswith("rank ")
               for event in stack):
            return
        now = self.clock()
        if now - max(self.last_feedback, self.last_notice) < WAIT_SECONDS:
            return
        groups = {}
        for source, stack in self.stages.items():
            if stack:
                event = stack[-1]
                detail = event.get("label", "working")
                if event.get("total") is not None:
                    detail += f"; {event['completed']}/{event['total']} {event['unit']}"
                elif event.get("completed"):
                    detail += f"; {event['completed']} {event['unit']}"
                if event.get("measured") is not None:
                    detail += f"; last measured progress {max(0, now - event['measured']):.0f}s ago"
                groups.setdefault(detail, []).append(source)
        detail = "; ".join(f"{label} [{', '.join(sources)}]" for label, sources in groups.items())
        self.emit(f"waiting: {detail or 'workers finishing'}; last substantive feedback {now - self.last_feedback:.0f}s ago")
        self.last_notice = now
