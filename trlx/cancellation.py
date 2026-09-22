"""Worker-owned SIGINT requests, consumed only at matching rank boundaries."""

import contextlib
import os
import signal


_active = None


class Cancelled(KeyboardInterrupt):
    """All participating ranks agreed to stop before leaving a safe boundary."""


class Request:
    # The flag survives dependency exception handlers; only the worker owns it.
    def __init__(self):
        self.pid = os.getpid()
        self.requested = False

    # Never raise or call CUDA from a signal handler: a peer may need our next collective.
    def receive(self, signum, frame):
        self.requested = True


# Install before training-library imports and retain the handler through distributed teardown.
@contextlib.contextmanager
def worker_signals(enabled=True):
    global _active
    if not enabled:
        yield
        return
    previous_request = _active
    request = Request()
    previous_handler = signal.signal(signal.SIGINT, request.receive)
    _active = request
    try:
        yield request
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        _active = previous_request


# Every rank must call this in the same order, outside rank-only work and library
# sections with their own collectives. A local flag must never skip the vote.
def checkpoint(boundary):
    request = _active
    if request is None or request.pid != os.getpid():
        return
    import torch
    import torch.distributed as distributed

    stop = request.requested
    if distributed.is_available() and distributed.is_initialized():
        device = (torch.device("cuda", torch.cuda.current_device())
                  if "nccl" in str(distributed.get_backend()) else torch.device("cpu"))
        vote = torch.tensor([int(stop)], dtype=torch.int32, device=device)
        distributed.all_reduce(vote, op=distributed.ReduceOp.MAX)
        stop = bool(vote.item())
    if stop:
        request.requested = True
        try:
            print(f"cancellation agreed at {boundary}; finishing outstanding GPU work", flush=True)
        except (OSError, ValueError):
            pass  # A broken diagnostic pipe must not prevent agreed GPU cleanup.
        # Model work may use streams other than the cancellation vote's stream.
        # Finish it on every rank before normal process-group destruction begins.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        raise Cancelled()


# Custom callbacks are ordered before preflight, quality, and final reporting.
# Raising only after a shared vote also cancels evaluation, whose loop ignores
# TrainerControl.should_training_stop, without publishing completion-only work.
def callback_class():
    from transformers import TrainerCallback

    class CancellationCallback(TrainerCallback):
        # Preparation and distributed placement have completed on all ranks.
        def on_train_begin(self, args, state, control, **kwargs):
            checkpoint("training start")

        # Check before entering the next epoch's dataloader and model work.
        def on_epoch_begin(self, args, state, control, **kwargs):
            checkpoint("epoch start")

        # All ranks completed batch prefetch before beginning this optimizer step.
        def on_step_begin(self, args, state, control, **kwargs):
            checkpoint("training step start")

        # Nonfinal accumulation microbatches still need a common stopping point.
        def on_substep_end(self, args, state, control, **kwargs):
            checkpoint("training microbatch end")

        # Stop before the trainer schedules evaluation or checkpoint publication.
        def on_step_end(self, args, state, control, **kwargs):
            checkpoint("training step end")

        # Prediction and its metric gathers are finished; do not start another batch.
        def on_prediction_step(self, args, state, control, **kwargs):
            checkpoint("evaluation batch end")

        # Cancel before independent quality callbacks run after ordinary evaluation.
        def on_evaluate(self, args, state, control, **kwargs):
            checkpoint("evaluation end")

        # Let an in-progress checkpoint finish before honoring cancellation.
        def on_save(self, args, state, control, **kwargs):
            checkpoint("checkpoint save end")

        # Do not turn cancellation into successful final assessment or quality work.
        def on_train_end(self, args, state, control, **kwargs):
            checkpoint("training completion")

    return CancellationCallback
