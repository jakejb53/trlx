"""POSIX shutdown for process groups created and owned by trlx."""

import contextlib
import os
import signal
import subprocess
import time

# Failure and orphan cleanup is bounded; cooperative user cancellation is not.
GRACE_SECONDS = 60.0
TERM_SECONDS = 5.0
KILL_SECONDS = 5.0
POLL_SECONDS = 0.1


# Defer Ctrl+C until Popen and ownership registration are complete. Unlike a
# blocked signal mask, this does not leave SIGINT blocked in the new child.
@contextlib.contextmanager
def defer_interrupt():
    previous = signal.getsignal(signal.SIGINT)
    interrupted = False
    failed = False

    # The supervisor will unwind as soon as the newly spawned process is owned.
    def remember(signum, frame):
        nonlocal interrupted
        interrupted = True

    remember._trlx_deferred_interrupt = True
    signal.signal(signal.SIGINT, remember)
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
        if interrupted and not failed:
            # A cohort launch nests individual ownership registrations. Transfer
            # the request outward so unspawned peers are not abandoned.
            if getattr(previous, "_trlx_deferred_interrupt", False):
                previous(signal.SIGINT, None)
            else:
                raise KeyboardInterrupt


# Only the launcher may establish this ownership; never derive a group from a
# live PID later, when its leader could have exited and left descendants behind.
def register(process, source):
    process._trlx_group = process.pid
    process._trlx_source = source
    process._trlx_shutdown_done = False
    process._trlx_cancel_ready = False


# Reaping a group leader does not prove that its dataloader children have exited.
def _group_alive(process):
    process.poll()
    try:
        os.killpg(process._trlx_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Do not mistake an inaccessible group for a completed shutdown.
    return True


# Stop every owned group concurrently, retaining the original caller's error.
# Failures are returned and reported rather than replacing cancellation or a worker failure.
def stop(children, report, *, cancelled=False):
    children = [process for process in dict.fromkeys(children)
                if not getattr(process, "_trlx_shutdown_done", False)]
    if not children:
        return []
    pending = []
    errors = []
    forced = False
    previous = signal.getsignal(signal.SIGINT)

    # A second Ctrl+C requests escalation; it must not unwind the shutdown itself.
    def force(signum, frame):
        nonlocal forced
        forced = True

    # Reporting is best effort during cleanup; broken output cannot strand children.
    def notice(message):
        try:
            report(message)
        except Exception as error:
            errors.append(f"cannot report shutdown: {error}")

    # Signal only the private session created at spawn, never the supervisor's group.
    def send(sig):
        for process in pending:
            try:
                os.killpg(process._trlx_group, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"cannot send {sig.name} to {process._trlx_source} (PID {process.pid}): {error}")

    # Poll all ranks together so a slow rank does not multiply the grace period.
    def wait(seconds, *, interruptible=True):
        deadline = time.monotonic() + seconds
        next_notice = time.monotonic() + 10
        while pending:
            pending[:] = [process for process in pending if _group_alive(process)]
            if not pending or time.monotonic() >= deadline or interruptible and forced:
                break
            if time.monotonic() >= next_notice:
                notice("waiting for cleanup: " + ", ".join(process._trlx_source for process in pending))
                next_notice = time.monotonic() + 10
            time.sleep(POLL_SECONDS)

    signal.signal(signal.SIGINT, force)
    try:
        for process in children:
            if getattr(process, "_trlx_shutdown_done", False):
                continue
            group = getattr(process, "_trlx_group", None)
            if group is None or group <= 1 or group == os.getpgrp():
                errors.append(f"refusing to signal an unowned process group for PID {process.pid}")
                continue
            pending.append(process)
        owned = list(pending)
        pending[:] = [process for process in pending if _group_alive(process)]
        if pending:
            if cancelled:
                notice("Clean shutdown in progress; waiting for workers to reach a safe stopping point. "
                       "This may take 60 seconds or longer. Press Ctrl+C again to force termination.")
                # Helpers must finish their current work; only rank leaders own
                # the cancellation handler and coordinated stopping boundaries.
                leaders = [process for process in owned if process._trlx_source.startswith("rank ")]
                next_notice = time.monotonic() + 10
                while not forced:
                    codes = [process.poll() for process in leaders]
                    # A failed peer cannot participate at the next boundary.
                    # Return to bounded failure cleanup instead of waiting forever.
                    if any(code not in (None, 0, 130) for code in codes):
                        notice("worker failed during clean shutdown; switching to bounded failure cleanup")
                        break
                    if all(code is not None for code in codes):
                        break
                    for process, code in zip(leaders, codes):
                        if (code is not None or getattr(process, "_trlx_cancelling", False)
                                or not process._trlx_cancel_ready):
                            continue
                        # The feedback acknowledgement closes the Popen-to-handler
                        # race: default SIGINT must never interrupt worker imports.
                        try:
                            os.kill(process.pid, signal.SIGINT)
                            process._trlx_cancelling = True
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            errors.append(f"cannot request cancellation of {process._trlx_source}: {error}")
                    if time.monotonic() >= next_notice:
                        notice("waiting for clean shutdown: " + ", ".join(
                            process._trlx_source for process in leaders if process.poll() is None))
                        next_notice = time.monotonic() + 10
                    time.sleep(POLL_SECONDS)
                pending[:] = [process for process in pending if _group_alive(process)]
            else:
                notice("stopping " + ", ".join(f"{process._trlx_source} (PID {process.pid})" for process in pending)
                       + "; allowing up to 60s for cleanup. Press Ctrl+C again to force termination.")
        # Once leaders exit, any surviving helpers (or a standalone verifier)
        # receive bounded cleanup. A second interrupt skips directly to SIGKILL.
        if pending and not forced:
            send(signal.SIGINT)
            wait(GRACE_SECONDS)
        if pending and not forced:
            notice("cleanup grace period expired; sending SIGTERM to remaining process groups")
            send(signal.SIGTERM)
            wait(TERM_SECONDS)
        if pending:
            notice("forcing termination of remaining process groups with SIGKILL")
            send(signal.SIGKILL)
            wait(KILL_SECONDS, interruptible=False)
        if pending:
            errors.append("process groups still present after SIGKILL: " + ", ".join(str(p._trlx_group) for p in pending))
        # poll() above normally reaps every leader. Keep this wait bounded even
        # for a process stuck in uninterruptible kernel I/O after SIGKILL.
        deadline = time.monotonic() + KILL_SECONDS
        for process in children:
            if getattr(process, "_trlx_group", None) is None:
                continue
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                errors.append(f"could not reap {process._trlx_source} (PID {process.pid}) after termination")
            else:
                process._trlx_shutdown_done = not _group_alive(process)
        for message in list(errors):
            notice("shutdown error: " + message)
        return errors
    finally:
        signal.signal(signal.SIGINT, previous)
