"""Run a torchao low-bit optimizer on the local shards of DTensor parameters.

torchao 0.18's low-bit Adam variants fail on DTensor parameters under torch 2.10
(compiled: fake-tensor conversion of the quantized state; eager: DTensor sharding
propagation of ``lerp``). They work on plain tensors. A parameter's local shard
view shares storage with the DTensor, so updating the shard updates the model,
and Adam is elementwise, so per-shard stepping computes the same values the
DTensor path would.

``LocalShardOptimizer`` builds the inner optimizer over shard views, and at each
step copies every DTensor gradient's local tensor onto its view before stepping.
It exposes ``param_groups`` for learning-rate reads. Optimizer state checkpoints
are not supported; this is for gate runs.
"""

import torch
from torch.distributed.tensor import DTensor


# Dim-0 slices of `t` with at most `max_elems` elements each (at least one row).
def _dim0_slices(t: torch.Tensor, max_elems: int) -> list[slice]:
    rows = t.shape[0]
    per_row = max(1, t.numel() // max(rows, 1))
    step = max(1, max_elems // per_row)
    return [slice(i, min(i + step, rows)) for i in range(0, rows, step)]


class LocalShardOptimizer(torch.optim.Optimizer):
    """Wrap ``factory(params=<local shards>, **kwargs)``; see module docstring."""

    def __init__(self, params, factory, max_chunk_elems: int = 1 << 28, **kwargs):
        originals = [p for p in params]
        if any(isinstance(p, dict) for p in originals):
            raise TypeError("LocalShardOptimizer takes a flat parameter list, not param groups")
        # (parameter, [(chunk view, dim-0 slice)]). A plain parameter is its own
        # single chunk; a DTensor's local shard is split along dim 0 so torchao's
        # step, which materializes several FP32 copies of each tensor it updates,
        # never holds more than max_chunk_elems elements at once. Chunks are views
        # of the shard's storage, and chunk boundaries fall on dim-0 rows, which
        # are multiples of torchao's 256-element state blocks for every target.
        self._entries: list[tuple[torch.Tensor, list[tuple[torch.Tensor, slice]]]] = []
        locals_: list[torch.Tensor] = []
        for p in originals:
            if not isinstance(p, DTensor):
                self._entries.append((p, [(p, slice(None))]))
                locals_.append(p)
                continue
            shard = p.to_local()
            if shard.untyped_storage().data_ptr() != p._local_tensor.untyped_storage().data_ptr():
                raise RuntimeError("local shard view does not share storage with its DTensor parameter")
            shard = shard.detach()
            chunks = []
            for sl in _dim0_slices(shard, max_chunk_elems):
                view = shard[sl].requires_grad_(True)
                chunks.append((view, sl))
                locals_.append(view)
            self._entries.append((p, chunks))
        self.inner = factory(params=locals_, **kwargs)
        # Register the original parameters so torch's Optimizer bookkeeping and
        # NeMo's param_groups reads see the model's parameters and the inner lr.
        super().__init__(originals, dict(self.inner.defaults))
        self.param_groups = self.inner.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        for p, chunks in self._entries:
            if not isinstance(p, DTensor):
                continue
            g = None if p.grad is None else p.grad.to_local().detach()
            for view, sl in chunks:
                view.grad = None if g is None else g[sl]
        return self.inner.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        for p, chunks in self._entries:
            p.grad = None
            if isinstance(p, DTensor):
                for view, _ in chunks:
                    view.grad = None

    def state_dict(self):
        raise NotImplementedError("LocalShardOptimizer does not support optimizer-state checkpoints")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("LocalShardOptimizer does not support optimizer-state checkpoints")


# Recipe entry points. Explicit signatures, not **kwargs: NeMo injects a
# ``foreach`` argument into any factory whose signature accepts arbitrary
# keywords, and torchao's optimizers do not take it.
def adamw8bit(params, lr, weight_decay, betas, bf16_stochastic_round, eps=1e-8):
    """torchao AdamW8bit over local shards."""
    from torchao.optim import AdamW8bit

    return LocalShardOptimizer(params, AdamW8bit, lr=lr, weight_decay=weight_decay, betas=tuple(betas),
                               eps=eps, bf16_stochastic_round=bf16_stochastic_round)


def adamw4bit(params, lr, weight_decay, betas, bf16_stochastic_round, eps=1e-8):
    """torchao AdamW4bit over local shards."""
    from torchao.optim import AdamW4bit

    return LocalShardOptimizer(params, AdamW4bit, lr=lr, weight_decay=weight_decay, betas=tuple(betas),
                               eps=eps, bf16_stochastic_round=bf16_stochastic_round)
