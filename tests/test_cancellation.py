"""Real signal retention and trainer-boundary cancellation, without model downloads."""

import contextlib
import io
import os
import pathlib
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from trlx import cancellation


class SignalRetention(unittest.TestCase):
    # Real SIGINT must survive a dependency's bare-except hashing fallback.
    def test_fingerprinting_retains_signal_until_boundary(self):
        from datasets import fingerprint

        original = fingerprint.Hasher.update
        count = 0

        # Inject at argument hashing, the exact catch-all path observed in the run.
        def interrupt(hasher, value):
            nonlocal count
            count += 1
            if count == 4:
                os.kill(os.getpid(), signal.SIGINT)
            return original(hasher, value)

        previous = signal.getsignal(signal.SIGINT)
        with cancellation.worker_signals() as request, patch.object(fingerprint.Hasher, "update", interrupt):
            result = fingerprint.update_fingerprint("old", "transform", {"fn_kwargs": {"x": 1}})
            self.assertIsInstance(result, str)
            self.assertTrue(request.requested)
            with self.assertRaises(cancellation.Cancelled):
                cancellation.checkpoint("preparation end")
            # Repeated signals during teardown must not interrupt that teardown.
            os.kill(os.getpid(), signal.SIGINT)
        self.assertIs(signal.getsignal(signal.SIGINT), previous)

    # Read-only commands must neither import distributed state here nor cast votes.
    def test_no_worker_scope_leaves_distributed_untouched(self):
        with patch("torch.distributed.is_initialized") as initialized:
            cancellation.checkpoint("standalone check")
        initialized.assert_not_called()

    # A peer's request must stop this rank even when it never received a signal.
    def test_peer_vote_cancels_uninterrupted_rank(self):
        # Model a peer setting the MAX-reduced flag, without launching GPUs.
        def peer_vote(vote, **kwargs):
            vote.fill_(1)

        with cancellation.worker_signals() as request, \
                patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_backend", return_value="gloo"), \
                patch("torch.distributed.all_reduce", side_effect=peer_vote), \
                patch("torch.cuda.is_initialized", return_value=True), \
                patch("torch.cuda.synchronize") as synchronize:
            self.assertFalse(request.requested)
            with self.assertRaises(cancellation.Cancelled):
                cancellation.checkpoint("batch end")
            self.assertTrue(request.requested)
            synchronize.assert_called_once_with()

    # Log I/O failure must not stop CUDA synchronization after a successful vote.
    def test_broken_output_still_synchronizes(self):
        with cancellation.worker_signals() as request, \
                patch("builtins.print", side_effect=BrokenPipeError), \
                patch("torch.cuda.is_initialized", return_value=True), \
                patch("torch.cuda.synchronize") as synchronize:
            request.requested = True
            with self.assertRaises(cancellation.Cancelled):
                cancellation.checkpoint("batch end")
            synchronize.assert_called_once_with()


class TinyModel(torch.nn.Module):
    # A real differentiable model; signal injection occurs inside normal forward work.
    def __init__(self, interrupt_phase=None):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.interrupt_phase = interrupt_phase
        self.training_calls = 0
        self.evaluation_calls = 0

    # The operation finishes after SIGINT so the callback can cancel at its boundary.
    def forward(self, input_ids, labels=None):
        phase = "train" if self.training else "eval"
        if self.training:
            self.training_calls += 1
        else:
            self.evaluation_calls += 1
        if phase == self.interrupt_phase:
            self.interrupt_phase = None
            os.kill(os.getpid(), signal.SIGINT)
        logits = self.linear(input_ids.float())
        loss = ((logits.reshape(-1) - labels.float()) ** 2).mean() if labels is not None else None
        return {"loss": loss, "logits": logits}


class Completion(TrainerCallback):
    # Observe completion-only work without adding any model or artifact behavior.
    def __init__(self):
        self.called = False

    # Cancelled training must bypass normal completion callbacks.
    def on_train_end(self, args, state, control, **kwargs):
        self.called = True


class TrainerBoundaries(unittest.TestCase):
    # All trainer-created output stays in disposable repository-local scratch.
    def make_trainer(self, phase=None):
        directory = self.enterContext(tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent))
        args = TrainingArguments(output_dir=directory, use_cpu=True, report_to="none", save_strategy="no",
                                 eval_strategy="no", max_steps=2, per_device_train_batch_size=1,
                                 per_device_eval_batch_size=1, gradient_accumulation_steps=2,
                                 dataloader_pin_memory=False, disable_tqdm=True, optim="adamw_torch")
        data = [{"input_ids": [1.0, 2.0], "labels": 1.0} for _ in range(8)]
        model = TinyModel(phase)
        completion = Completion()
        trainer = Trainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                          callbacks=[cancellation.callback_class()(), completion])
        return trainer, model, completion

    # A request already pending at train start must not run a forward or final callback.
    def test_preparation_request_stops_before_training(self):
        trainer, model, completion = self.make_trainer()
        with cancellation.worker_signals():
            os.kill(os.getpid(), signal.SIGINT)
            with self.assertRaises(cancellation.Cancelled):
                trainer.train()
        self.assertEqual(model.training_calls, 0)
        self.assertFalse(completion.called)

    # Accumulation need not finish its whole optimizer step to leave safely.
    def test_signal_inside_forward_stops_at_microbatch_boundary(self):
        trainer, model, completion = self.make_trainer("train")
        with cancellation.worker_signals(), self.assertRaises(cancellation.Cancelled):
            trainer.train()
        self.assertEqual(model.training_calls, 1)
        self.assertEqual(trainer.state.global_step, 0)
        self.assertFalse(completion.called)

    # Evaluation ignores should_training_stop; the agreed exception stops its loop.
    def test_evaluation_stops_after_current_batch(self):
        trainer, model, _ = self.make_trainer("eval")
        with cancellation.worker_signals(), self.assertRaises(cancellation.Cancelled):
            trainer.evaluate()
        self.assertEqual(model.evaluation_calls, 1)
        self.assertEqual(trainer.state.log_history, [])

    # With no request, callbacks preserve updates and normal completion.
    def test_uninterrupted_training_completes(self):
        trainer, model, completion = self.make_trainer()
        with cancellation.worker_signals(), contextlib.redirect_stdout(io.StringIO()):
            trainer.train()
        self.assertEqual(trainer.state.global_step, 2)
        self.assertTrue(completion.called)
        self.assertEqual(model.training_calls, 4)


class AuxiliaryBoundaries(unittest.TestCase):
    # Cancellation between quality batches must retain the caller's module modes.
    def test_quality_cancels_before_second_batch(self):
        from tests.test_quality import Model, Rows, Tokenizer, settings
        from trlx import quality

        model = Model(["Answer", "Answer"])
        model.train()
        model.dropout.eval()
        original = model.generate

        # Complete this batch before its pending cancellation is consumed.
        def generate(**kwargs):
            output = original(**kwargs)
            os.kill(os.getpid(), signal.SIGINT)
            return output

        rows = Rows([{"prompt": "Question", "answer": "Answer"}] * 2)
        with patch.object(model, "generate", side_effect=generate), cancellation.worker_signals(), \
                self.assertRaises(cancellation.Cancelled):
            quality.evaluate(model, Tokenizer(), rows, settings(quality_batch_size=1))
        self.assertEqual(len(model.generations), 1)
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)

    # Even a request during the final summary must prevent dataset publication.
    def test_synthetic_final_summary_cancels_before_publication(self):
        from types import SimpleNamespace
        from tests.test_quality import Model, Rows, Tokenizer
        from trlx import synthetic_eval

        model = Model(["Summary"])
        model.train()
        original = model.generate

        # A finished response still does not authorize publishing a cancelled run.
        def generate(**kwargs):
            output = original(**kwargs)
            os.kill(os.getpid(), signal.SIGINT)
            return output

        trainer = SimpleNamespace(model=model, processing_class=Tokenizer(),
                                  accelerator=SimpleNamespace(unwrap_model=lambda value: value))
        callback = synthetic_eval.callback_class()(Rows([{"text": "Source"}]), "unused-run")
        callback.bind(trainer)
        with patch.object(model, "generate", side_effect=generate), \
                patch.object(synthetic_eval, "_prompt", return_value=[10]), \
                patch.object(synthetic_eval, "_summary", return_value={"text": "Summary"}), \
                patch.object(synthetic_eval, "write_rows") as publish, \
                cancellation.worker_signals(), self.assertRaises(cancellation.Cancelled):
            callback.on_train_begin(SimpleNamespace(max_length=32), None, None)
        publish.assert_not_called()
        self.assertFalse(callback.ready)
        self.assertTrue(model.training)
