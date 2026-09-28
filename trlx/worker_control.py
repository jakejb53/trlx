"""Worker-owned distributed teardown, independent of the training thread."""

import os
import select
import threading

from trlx import TrlxError

# Internal launch protocol, not operator configuration. Commands fit in one pipe write.
PIPE_ENV = "TRLX_CONTROL_FD"
ABORT = b"A"
SHUTDOWN = b"S"
POLL_SECONDS = 0.1


class Control:
    # Only the control thread destroys/aborts groups; the training thread finishes first.
    def __init__(self, events):
        self.events = events
        self.fd = int(os.environ.pop(PIPE_ENV))
        os.set_inheritable(self.fd, False)
        self.aborting = threading.Event()
        self.finishing = threading.Event()
        self.initialized = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="trlx-worker-control", daemon=True)

    # Start before model preparation so supervisor requests cannot be lost at startup.
    def __enter__(self):
        self.thread.start()
        return self

    # Configuration owns initial group creation. Never clear its registry concurrently.
    def ready(self):
        self.initialized.set()
        self.check()

    # Call at training boundaries as abort may release a CUDA wait without raising.
    def check(self):
        if self.error is not None:
            raise self.error
        if self.aborting.is_set():
            raise TrlxError("distributed training aborted after a worker failure")

    # Retain the first local cause and publish it before any potentially blocking teardown.
    def _fail(self, error):
        if self.error is None:
            self.error = error
            try:
                self.events({"kind": "worker_failure", "message": str(error) or type(error).__name__})
            except Exception as reporting_error:
                error.add_note(f"cannot report worker failure: {reporting_error}")
        self.aborting.set()

    # Watch all registered NCCL backends, including FSDP subgroups, without CUDA synchronization.
    def _backend_error(self, torch, c10d):
        for group in list(c10d._world.pg_map):
            for device in group._device_types:
                if device.type != "cuda":
                    continue
                backend = group._get_backend(device)
                if isinstance(backend, c10d.ProcessGroupNCCL):
                    error = backend.get_error()
                    if error != torch._C._distributed_c10d.ErrorType.SUCCESS:
                        return TrlxError(f"NCCL process group reported {error}")
        return None

    # Receive out-of-band requests while the main thread may be blocked inside CUDA/NCCL.
    def _run(self):
        import torch
        from torch.distributed import distributed_c10d as c10d

        try:
            while True:
                readable, _, _ = select.select([self.fd], [], [], POLL_SECONDS)
                if readable:
                    command = os.read(self.fd, 1)
                    if command == ABORT:
                        self.aborting.set()
                    elif command == SHUTDOWN:
                        if not self.finishing.is_set():
                            raise TrlxError("supervisor requested shutdown before worker completion")
                        # Every rank has finished training; no peer still needs another collective.
                        if not self.aborting.is_set():
                            if c10d.is_initialized():
                                c10d.destroy_process_group()
                            return
                    elif not command:
                        self._fail(TrlxError("supervisor control pipe closed before worker shutdown"))
                    else:
                        raise TrlxError("invalid supervisor control command")
                if not self.initialized.is_set():
                    continue
                if not self.aborting.is_set():
                    error = self._backend_error(torch, c10d)
                    if error is not None:
                        self._fail(error)
                if self.aborting.is_set():
                    self._abort(c10d)
                    return
        except Exception as error:
            self._fail(TrlxError(f"distributed cleanup failed: {error}"))
            # A monitor or normal-shutdown exception must not strand live communicators.
            self._abort(c10d)

    # Abort owns registry cleanup; never follow it with normal collective destruction.
    def _abort(self, c10d):
        try:
            if c10d.is_initialized():
                c10d._abort_process_group()
        except Exception as error:
            message = f"distributed communicator abort failed: {error}"
            if self.error is None:
                self._fail(TrlxError(message))
            else:
                self.error.add_note(message)
            # Cleanup failure is visible separately from the original training failure.
            try:
                self.events({"kind": "warning", "message": message})
            except Exception as reporting_error:
                self.error.add_note(f"cannot report abort failure: {reporting_error}")
        else:
            try:
                self.events({"kind": "note", "message": "distributed communicator abort completed"})
            except Exception as reporting_error:
                self._fail(TrlxError(f"cannot report completed abort: {reporting_error}"))

    # Success/cancellation waits for cohort approval; failure requests abort without rendezvous.
    def __exit__(self, exc_type, error, traceback):
        if error is not None and not isinstance(error, KeyboardInterrupt):
            self._fail(error)
        self.finishing.set()
        # Main-thread initialization has stopped, even when it raised partway through.
        self.initialized.set()
        if not self.aborting.is_set():
            try:
                self.events({"kind": "worker_finished", "message": "worker ready for distributed shutdown"})
            except Exception as reporting_error:
                self._fail(reporting_error)
        try:
            self.thread.join()
        finally:
            os.close(self.fd)
        if error is None or isinstance(error, KeyboardInterrupt) and self.aborting.is_set():
            self.check()
        elif self.error is not None and self.error is not error:
            error.add_note(str(self.error))


# Abort can unblock native calls successfully; prevent any later training/evaluation/save work.
def callback_class():
    from transformers import TrainerCallback

    class ControlCallback(TrainerCallback):
        # The wrapper owns the lifetime; callbacks only observe its terminal state.
        def __init__(self, control):
            self.control = control

        # All callback boundaries check the same state before other trlx callbacks run.
        def check(self, *args, **kwargs):
            self.control.check()

        on_train_begin = check
        on_epoch_begin = check
        on_step_begin = check
        on_substep_end = check
        on_step_end = check
        on_prediction_step = check
        on_evaluate = check
        on_save = check
        on_train_end = check

    return ControlCallback
