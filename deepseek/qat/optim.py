"""Run a torchao low-bit optimizer on the local shards of DTensor parameters.

torchao 0.18's low-bit Adam variants fail on DTensor parameters under torch 2.10
(compiled: fake-tensor conversion of the quantized state; eager: DTensor sharding
propagation of ``lerp``). They work on plain tensors. A parameter's local shard
view shares storage with the DTensor, so updating the shard updates the model,
and Adam is elementwise, so per-shard stepping computes the same values the
DTensor path would.

``LocalShardOptimizer`` builds the inner optimizer over shard views, and at each
step copies every DTensor gradient's local tensor onto its view before stepping.
``step_param`` does the same for a single parameter so ``deepseek.qat.inbackward``
can step each parameter as its gradient arrives. It exposes ``param_groups`` for
learning-rate reads. Optimizer state checkpoints
are not supported; this is for gate runs.

``MasterOptimizer`` is the FP32-master counterpart: torch AdamW over FP32 copies
of the same chunks, written back to the BF16 shards after each step. Both expose
``step`` and ``step_param``.
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
        self._chunks_by_param = {id(p): chunks for p, chunks in self._entries}
        self.inner = factory(params=locals_, **kwargs)
        # Register the original parameters so torch's Optimizer bookkeeping and
        # NeMo's param_groups reads see the model's parameters and the inner lr.
        super().__init__(originals, dict(self.inner.defaults))
        self.param_groups = self.inner.param_groups

    # Restrict a parameter's stepped rows to `rows` (local dim-0 indices): its
    # chunks become one view per selected row, so the other rows are never
    # stepped and never get optimizer state. This is how ESFT-style expert
    # selection freezes unselected experts inside NeMo's stacked expert
    # parameters. Must run before any step, while the inner optimizer holds no
    # state, so the replaced views leave nothing behind; each row of every
    # target is a multiple of torchao's 256-element state block. An empty
    # selection freezes every row of this shard (a rank holding none of a
    # layer's selected experts).
    @torch.no_grad()
    def restrict_rows(self, p: torch.Tensor, rows: list[int]) -> None:
        old = self._chunks_by_param.get(id(p))
        if old is None:
            raise RuntimeError("restrict_rows called with a parameter this optimizer does not own")
        if len(self.inner.state):
            raise RuntimeError("restrict_rows must be called before the first optimizer step")
        shard = (p.to_local() if isinstance(p, DTensor) else p).detach()
        n = shard.shape[0]
        if any(not (0 <= r < n) for r in rows) or len(set(rows)) != len(rows):
            raise ValueError(f"restrict_rows needs distinct row indices in [0, {n}), got {rows}")
        new = [(shard[r].requires_grad_(True), r) for r in sorted(rows)]
        old_views = {id(v) for v, _ in old}
        params = self.inner.param_groups[0]["params"]
        kept = [q for q in params if id(q) not in old_views]
        if len(kept) != len(params) - len(old):
            raise RuntimeError("inner optimizer parameter list does not match the chunk views")
        self.inner.param_groups[0]["params"] = kept + [v for v, _ in new]
        for i, (q, chunks) in enumerate(self._entries):
            if q is p:
                self._entries[i] = (p, new)
        self._chunks_by_param[id(p)] = new

    # A parameter whose only chunk is itself needs no gradient mapping: its grad
    # is already where the inner optimizer reads it. Every other layout (a
    # DTensor's shard chunks, or row views after restrict_rows) maps the
    # gradient onto the views.
    @staticmethod
    def _is_self_chunk(p: torch.Tensor, chunks) -> bool:
        return len(chunks) == 1 and chunks[0][0] is p

    @staticmethod
    def _local_grad(p: torch.Tensor) -> torch.Tensor:
        return p.grad.to_local().detach() if isinstance(p, DTensor) else p.grad.detach()

    @torch.no_grad()
    def step(self, closure=None):
        for p, chunks in self._entries:
            if self._is_self_chunk(p, chunks):
                continue
            g = None if p.grad is None else self._local_grad(p)
            for view, sl in chunks:
                view.grad = None if g is None else g[sl]
        return self.inner.step(closure)

    # Step one parameter in isolation, for update-in-backward: its gradient is
    # mapped onto its chunk views, the inner optimizer runs (every other chunk
    # has no gradient and is skipped by the optimizer's usual None check), and
    # the chunk gradients are released. The caller owns ``p.grad`` and frees it.
    @torch.no_grad()
    def step_param(self, p: torch.Tensor) -> None:
        chunks = self._chunks_by_param.get(id(p))
        if chunks is None:
            raise RuntimeError("step_param called with a parameter this optimizer does not own")
        if p.grad is None:
            raise RuntimeError("step_param called with a parameter that has no gradient")
        mapped = not self._is_self_chunk(p, chunks)
        if mapped:
            g = self._local_grad(p)
            for view, sl in chunks:
                view.grad = g[sl]
        self.inner.step()
        if mapped:
            for view, _ in chunks:
                view.grad = None

    def zero_grad(self, set_to_none: bool = True):
        for p, chunks in self._entries:
            p.grad = None
            if not self._is_self_chunk(p, chunks):
                for view, _ in chunks:
                    view.grad = None

    def state_dict(self):
        raise NotImplementedError("LocalShardOptimizer does not support optimizer-state checkpoints")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("LocalShardOptimizer does not support optimizer-state checkpoints")


class MasterOptimizer(torch.optim.Optimizer):
    """FP32 masters beside the model's BF16 parameters; see module docstring.

    Each local shard is split along dim 0 like ``LocalShardOptimizer`` and every
    chunk gets an FP32 master copy. A step copies the chunk's gradient to FP32,
    runs torch AdamW on the masters, and writes ``bf16(master)`` back into the
    shard view with round-to-nearest, so the model, the fake-quant hook, and the
    export all see the BF16 image of the master. This is the FP32-master arm of
    the stage 4 comparison: it differs from ``adamw8bit`` only in master and
    optimizer-state precision; the forward is identical.
    """

    def __init__(self, params, max_chunk_elems: int = 1 << 28, **kwargs):
        originals = [p for p in params]
        if any(isinstance(p, dict) for p in originals):
            raise TypeError("MasterOptimizer takes a flat parameter list, not param groups")
        # (parameter, [(shard chunk view, master chunk)]).
        self._entries: list[tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]] = []
        masters: list[torch.Tensor] = []
        for p in originals:
            shard = p.to_local() if isinstance(p, DTensor) else p
            if isinstance(p, DTensor) and shard.untyped_storage().data_ptr() != p._local_tensor.untyped_storage().data_ptr():
                raise RuntimeError("local shard view does not share storage with its DTensor parameter")
            shard = shard.detach()
            chunks = []
            for sl in _dim0_slices(shard, max_chunk_elems):
                view = shard[sl]
                master = view.detach().to(torch.float32).clone().requires_grad_(True)
                chunks.append((view, master))
                masters.append(master)
            self._entries.append((p, chunks))
        self._chunks_by_param = {id(p): chunks for p, chunks in self._entries}
        # foreach=False keeps AdamW's temporaries per chunk (at most one chunk's
        # worth of FP32 at a time) instead of per parameter list.
        self.inner = torch.optim.AdamW(masters, foreach=False, **kwargs)
        super().__init__(originals, dict(self.inner.defaults))
        self.param_groups = self.inner.param_groups

    # Copy one parameter's gradient onto its masters' grads (FP32).
    @staticmethod
    def _load_grad(p: torch.Tensor, chunks) -> None:
        g = (p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).detach()
        # Chunks are consecutive dim-0 slices of the shard, so the gradient is
        # walked in the same order with each chunk's row count.
        rows = 0
        for view, master in chunks:
            n = view.shape[0]
            master.grad = g[rows : rows + n].to(torch.float32)
            rows += n

    # Write the stepped masters back into the BF16 shard views and drop grads.
    @staticmethod
    def _write_back(chunks) -> None:
        for view, master in chunks:
            view.copy_(master)
            master.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        stepped = [(p, chunks) for p, chunks in self._entries if p.grad is not None]
        for p, chunks in stepped:
            self._load_grad(p, chunks)
        out = self.inner.step(closure)
        for _, chunks in stepped:
            self._write_back(chunks)
        return out

    # Same contract as LocalShardOptimizer.step_param.
    @torch.no_grad()
    def step_param(self, p: torch.Tensor) -> None:
        chunks = self._chunks_by_param.get(id(p))
        if chunks is None:
            raise RuntimeError("step_param called with a parameter this optimizer does not own")
        if p.grad is None:
            raise RuntimeError("step_param called with a parameter that has no gradient")
        self._load_grad(p, chunks)
        self.inner.step()
        self._write_back(chunks)

    def zero_grad(self, set_to_none: bool = True):
        for p, chunks in self._entries:
            p.grad = None
            for _, master in chunks:
                master.grad = None

    def state_dict(self):
        raise NotImplementedError("MasterOptimizer does not support optimizer-state checkpoints")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("MasterOptimizer does not support optimizer-state checkpoints")


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


def adamw_fp32_master(params, lr, weight_decay, betas, eps=1e-8):
    """torch AdamW on FP32 masters of the local shards, written back as BF16."""
    return MasterOptimizer(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas), eps=eps)
