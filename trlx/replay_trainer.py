"""SFTTrainer with the replay KL term (SPEC 2.9).

Used only when [replay].kl_coef > 0; plain mixing runs on the stock trainer.
Three overrides carry a per-example replay flag from the mixed dataset to the
loss: the signature columns keep data_load.REPLAY_COLUMN through the
trainer's remove_unused_columns, the collator turns it into a batch tensor,
and compute_loss pops it before the model forward.

The KL term is exact forward KL, KL(original || training), over the full
vocabulary at each assistant token of each replay row, summed over the
optimizer step and divided by the step's trained tokens as the SFT loss is,
times kl_coef: kl_coef weights each replay token as the NLL weights each
trained token (_nll_normalisation). Assistant tokens are the positions whose
label is not -100, shifted as cross-entropy shifts them. The original model
is the same PeftModel with adapters disabled for LoRA runs, or a second
loaded copy (`reference_model`) for full fine-tunes. Its forward runs without
labels under no_grad, and both logit tensors are gathered at the replay
assistant positions before the float32 upcast, so what persists is two
(tokens x vocabulary) float32 slices; the reference's full logits are a
transient. A step on which no rank holds a replay row skips the reference
forward; otherwise every rank runs it, because under FSDP it is a
collective (see _replay_term).

The training model's logits exist because config.py forces loss_type =
"nll" for this trainer; TRL's default chunked loss returns none (SPEC 5).

The mean KL per replay token, not the loss term, is logged as `replay_kl`
through the trainer's own metric mechanism, so it reaches metrics.jsonl next
to mean_token_accuracy and can be named in [ranges].
"""

import torch
from trl import SFTTrainer

from trlx.data_load import REPLAY_COLUMN

# Metric name for the logged KL value.
KL_METRIC = "replay_kl"


# Wraps the trainer's collator: the base collator builds the model inputs
# from input_ids and labels and ignores other keys, so the replay flag is
# added here as a bool tensor. Rows without the column (the eval set) are
# not replay rows.
class ReplayCollator:
    def __init__(self, base):
        self.base = base

    def __call__(self, examples):
        batch = self.base(examples)
        batch[REPLAY_COLUMN] = torch.tensor([bool(e.get(REPLAY_COLUMN, False)) for e in examples])
        return batch


class ReplayTrainer(SFTTrainer):
    # `reference_model` is the frozen copy for a full fine-tune, None for a
    # LoRA run, where the adapters are disabled instead. The copy is placed
    # on this rank's device here; it is never prepared by accelerate, so it
    # is unsharded on every rank under every strategy (PLAN.md Phase 7).
    def __init__(self, *args, kl_coef, reference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef
        self.reference_model = reference_model
        if reference_model is not None:
            reference_model.requires_grad_(False)
            reference_model.eval()
            reference_model.to(self.accelerator.device)
        self.data_collator = ReplayCollator(self.data_collator)

    # Keeps the replay column through remove_unused_columns.
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if REPLAY_COLUMN not in self._signature_columns:
            self._signature_columns = list(self._signature_columns) + [REPLAY_COLUMN]

    # The parent's SFT loss plus kl_coef times the replay KL term. The flag
    # is popped before the parent runs so it never reaches the model forward.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        replay = inputs.pop(REPLAY_COLUMN, None)
        # Read before the parent's call, which may drop labels from `inputs`.
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if replay is not None and labels is not None:
            kl = self._replay_term(model, inputs, outputs.logits, labels, replay, num_items_in_batch)
            if kl is not None:
                loss = loss + self.kl_coef * kl
        return (loss, outputs) if return_outputs else loss

    # Collective discipline: every rank takes the same path through here on
    # every call. Ranks hold different batches, the reference forward is a
    # collective under FSDP, and the metric gather is one under every
    # strategy, so whether any rank has a replay assistant token is agreed
    # first, and then every rank runs the forward and the gather, or none
    # does. Eval batches never carry replay rows (data_load.mix_replay adds
    # them to train only), so eval always returns None after the first
    # gather. The logged value is the mean KL per replay token across ranks.
    # Returns the local KL term to add to the loss, normalised like the SFT
    # loss (_nll_normalisation), or None when no rank had a replay row.
    def _replay_term(self, model, inputs, logits, labels, replay, num_items_in_batch):
        # Position t predicts token t + 1, so the masks are over labels[1:].
        trained = labels[:, 1:] != -100
        mask = trained & replay.to(labels.device)[:, None]
        count = mask.sum().float()
        if not bool(self.accelerator.gather(count.view(1)).sum() > 0):
            return None
        kl_sum = self._replay_kl(model, inputs, logits, mask)
        totals = self.accelerator.gather(torch.stack([kl_sum.detach(), count]).view(1, 2)).sum(0)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode][KL_METRIC].append((totals[0] / totals[1]).item())
        denominator, scale = self._nll_normalisation(num_items_in_batch, trained)
        return kl_sum / denominator * scale

    # The denominator and scale factor the SFT loss carries, so the KL term
    # is summed over every replay token of the optimizer step and divided by
    # the step's trained tokens exactly as the NLL is, under any gradient
    # accumulation and world size. Mirrors transformers 5.17 (pinned):
    # the model's cross-entropy divides its token sum by num_items_in_batch
    # (trained tokens of the whole accumulation window, summed across ranks)
    # when the model accepts loss kwargs, else takes a local mean; and
    # Trainer.compute_loss multiplies by the process count under
    # average_tokens_across_devices so DDP's gradient averaging yields a
    # global per-token mean. Trainer exposes neither rule, so a pin bump must
    # re-check Trainer.compute_loss and Trainer.training_step against this.
    def _nll_normalisation(self, num_items_in_batch, trained):
        uses_items = num_items_in_batch is not None and bool(self.model_accepts_loss_kwargs or self.compute_loss_func)
        if not uses_items:
            # Local mean; training_step divides the whole loss by the
            # accumulation steps in this case, the KL term included.
            return trained.sum().clamp(min=1), 1
        scale = 1
        if self.args.average_tokens_across_devices:
            scale = self.accelerator.num_processes
            pc = getattr(self.accelerator, "parallelism_config", None)
            if pc is not None:
                scale //= pc.tp_size
            if self.args.n_gpu > 1:
                scale = self.args.n_gpu
        return num_items_in_batch, scale

    # Summed forward KL at the masked positions (module docstring). `logits`
    # are the training model's from the loss forward, still attached to the
    # graph; the reference logits come from a no-grad forward of the same
    # inputs without labels. An empty mask selects nothing and sums to zero,
    # after the reference forward has run for the ranks that need it.
    def _replay_kl(self, model, inputs, logits, mask):
        ref_inputs = {k: v for k, v in inputs.items() if k not in ("labels", "shift_labels")}
        with torch.no_grad():
            if self.reference_model is not None:
                ref_logits = self.reference_model(**ref_inputs).logits
            else:
                with self.model.disable_adapter():
                    ref_logits = model(**ref_inputs).logits
            ref = torch.log_softmax(ref_logits[:, :-1][mask].float(), dim=-1)
            del ref_logits
        cur = torch.log_softmax(logits[:, :-1][mask].float(), dim=-1)
        return (ref.exp() * (ref - cur)).sum()
