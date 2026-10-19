"""Update-in-backward with lagged global-norm clipping.

The B200 memory plan (``ep_size: 8``) cannot hold a full BF16 gradient for each
rank's routed experts (about 69 GB), so every trainable parameter is stepped as
soon as its gradient arrives and the gradient is freed. The hook point is
``register_post_accumulate_grad_hook``: autograd fires it for parameters outside
FSDP, and FSDP2 fires it on the sharded parameter right after reduce-scatter.

Global-norm clipping needs every gradient at once, which update-in-backward
never has. Instead the exact global norm of each step is accumulated as the
gradients arrive (after NeMo's expert-gradient scaling) and reduced once at step
end, and the clip coefficient ``min(1, max_norm / (norm + 1e-6))`` from the
previous step scales the current step's gradients. The first step is unclipped;
Adam's first update is ``lr`` times the gradient's sign regardless of scale.

NeMo's own post-backward scaling stays the source of truth: the hook calls
``scale_grads_and_clip_grad_norm`` with clipping disabled, which scales exactly
the one parameter holding a gradient at that moment. The stepper verifies that
invariant on every call rather than assuming it.

Requirements, checked at construction: one microbatch per step (accumulation
would require holding gradients), and a topology where a single all-reduce over
the world group yields the global norm (``tp = pp = cp = 1`` and the
data-parallel and expert-parallel groups both span the world).

Row selection (``rows_of``): for a parameter whose name maps to a list of local
dim-0 rows, only those rows are stepped and hold optimizer state
(``LocalShardOptimizer.restrict_rows``), and the other rows' gradients are zeroed
before the norm so the class norm counts trained rows only. This freezes
unselected experts inside NeMo's stacked expert parameters for ESFT-style
selection; the selection is computed by the caller, which knows the global
expert index of each local row.

Update period per class (``every``): a class with period N is stepped only on
steps whose index is a multiple of N; on the other steps its gradient still
counts toward the norm and is then freed without an optimizer call, so the
class's Adam moments are untouched. This is the frequency-split lever: under
Adam, 1 step in N at rate r drifts the same expected distance as every step at
r/N, with every applied update at the full rate.

Start step per class (``start``): a class with start S is not stepped before
step S; its gradient counts toward the norm and is freed, as on an off-period
step, so no optimizer state exists for it until S (torchao allocates lazily)
and the other classes' moments are unaffected. The period, if any, counts from
S. This is the delayed dense start (Pending item 1): experts publish first, and
the run to S is identical to an experts-only run by construction.
"""

import math
from typing import Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from deepseek.qat.optim import _dim0_slices

# Elements per float32 temporary when accumulating a gradient's squared norm.
NORM_CHUNK_ELEMS = 1 << 28


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


# Sum of squares in float32, in dim-0 chunks so a 2-billion-element expert
# gradient never materializes a full float32 copy.
def _sum_sq(g: torch.Tensor) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float32, device=g.device)
    for sl in _dim0_slices(g, NORM_CHUNK_ELEMS):
        total += g[sl].float().square().sum()
    return total


class InBackwardStepper:
    """Hooks every trainable parameter; see module docstring."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer,
        max_norm: float,
        scale_grads: Callable[[], None],
        reduce_group=None,
        class_of: Callable[[str], str] | None = None,
        every: dict[str, int] | None = None,
        start: dict[str, int] | None = None,
        rows_of: Callable[[str], list[int] | None] | None = None,
        report_class_of: Callable[[str], str] | None = None,
        drift_sample_of: Callable[[str, torch.Tensor], tuple[str, tuple] | None] | None = None,
    ):
        # Duck-typed: LocalShardOptimizer and MasterOptimizer both provide step_param.
        if not callable(getattr(optimizer, "step_param", None)):
            raise TypeError(f"InBackwardStepper requires an optimizer with step_param (LocalShardOptimizer or MasterOptimizer), got {type(optimizer).__name__}")
        if not (max_norm > 0 and math.isfinite(max_norm)):
            raise ValueError(f"max_norm must be finite and positive, got {max_norm}")
        self.model_parts = model_parts
        self.optimizer = optimizer
        self.max_norm = float(max_norm)
        # Applies NeMo's pipeline/expert gradient scaling to whatever gradients
        # exist; called with exactly one gradient present.
        self._scale_grads = scale_grads
        self._group = reduce_group
        self.step_count = 0
        self._params = [p for mp in model_parts for p in mp.parameters() if p.requires_grad]
        # Clipping classes. With class_of None every parameter is one class and
        # the coefficient is the global one. With a classifier (dense vs expert),
        # each class has its own norm and coefficient, so a class with large
        # gradients does not throttle the drift of a class with small ones.
        self._class_by_param: dict[int, str] = {}
        # Master drift samples (the per-step learning signal the published flips
        # lag by tens of steps): for each sampled parameter, a fixed slice of its
        # local shard with its baseline, its value after the previous step, and
        # the running sum of |update|. drift_sample_of(name, param) returns
        # (class, index tuple into the local shard) or None. Per class, end_step
        # publishes mean |master - baseline| and the sign consistency
        # sum|drift| / sum(sum|update|): 1.0 when every update pushed the same
        # way, near 0 for a random walk.
        self._drift: dict[int, tuple[str, str, tuple, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.drift_classes: list[str] = []
        self.last_drift: dict[str, dict[str, float]] = {}
        # The same per sampled parameter (by canonical name), reduced over ranks:
        # the per-tensor sign consistency that ranks tensors for selection.
        self.last_drift_by_param: dict[str, dict[str, float]] = {}
        # Reporting classes (dense vs expert) are independent of the clipping
        # classes: their norms are published every step in last_report_norms and
        # never affect a coefficient.
        self._report_class_by_param: dict[int, str] = {}
        # Row masks for parameters with a selection: True rows are stepped.
        # The optimizer is restricted here, before any step, so unselected rows
        # never receive optimizer state.
        self._row_mask: dict[int, torch.Tensor] = {}
        for mp in model_parts:
            for name, q in mp.named_parameters():
                if q.requires_grad:
                    self._class_by_param[id(q)] = class_of(name) if class_of is not None else "all"
                    self._report_class_by_param[id(q)] = report_class_of(name) if report_class_of is not None else self._class_by_param[id(q)]
                    sample = drift_sample_of(name, q) if drift_sample_of is not None else None
                    if sample is not None:
                        cls, index = sample
                        base = _local(q).detach()[index].clone()
                        self._drift[id(q)] = (cls, name, index, base, base.clone(), torch.zeros(base.shape, dtype=torch.float32, device=base.device))
                    rows = rows_of(name) if rows_of is not None else None
                    if rows is not None:
                        if not callable(getattr(optimizer, "restrict_rows", None)):
                            raise TypeError(f"row selection needs an optimizer with restrict_rows, got {type(optimizer).__name__}")
                        optimizer.restrict_rows(q, rows)
                        local = _local(q)
                        mask = torch.zeros(local.shape[0], dtype=torch.bool, device=local.device)
                        mask[rows] = True
                        self._row_mask[id(q)] = mask
        self.classes: list[str] = sorted(set(self._class_by_param.values()))
        # Update period per class; absent means every step. A period for a class
        # that does not exist is a configuration error, not a silent no-op.
        self.every: dict[str, int] = {}
        for cls, n in (every or {}).items():
            if cls not in self.classes:
                raise ValueError(f"update period given for unknown class {cls!r}; classes are {self.classes}")
            if not (isinstance(n, int) and n >= 1):
                raise ValueError(f"update period for class {cls!r} must be a positive integer, got {n!r}")
            self.every[cls] = n
        # Start step per class; absent means step 0. Same validation as every.
        self.start: dict[str, int] = {}
        for cls, s in (start or {}).items():
            if cls not in self.classes:
                raise ValueError(f"start step given for unknown class {cls!r}; classes are {self.classes}")
            if not (isinstance(s, int) and s >= 0):
                raise ValueError(f"start step for class {cls!r} must be a nonnegative integer, got {s!r}")
            self.start[cls] = s
        self.coefs: dict[str, float] = {c: 1.0 for c in self.classes}
        self.last_norms: dict[str, float] = {}
        self.report_classes: list[str] = sorted(set(self._report_class_by_param.values()))
        self.drift_classes = sorted({cls for cls, *_ in self._drift.values()})
        self._drift_names = sorted(name for _, name, *_ in self._drift.values())
        self.last_report_norms: dict[str, float] = {}
        self.last_norm: float | None = None  # global norm over all classes
        self._stepped: set[int] = set()
        # Largest CUDA allocation (bytes) observed at hook entry during the
        # current step, when a gradient and the backward's live activations are
        # both resident. The per-step analogue of sampling at the optimizer step
        # with every gradient held; end_step publishes it as last_peak_hook_allocated.
        self.peak_hook_allocated: int = 0
        self.last_peak_hook_allocated: int = 0
        # Per-hook squared norms with the CUDA event that marks their completion.
        # Hooks run on different streams (FSDP2 fires dense hooks inside its
        # reduce-scatter stream, autograd fires expert hooks on its own), so a
        # single running sum across them would race; each contribution is kept
        # with its event and summed at step end after waiting on all of them.
        self._sq_parts: list[tuple[str, str, torch.Tensor, object]] = []
        self._handles = [p.register_post_accumulate_grad_hook(self._on_grad) for p in self._params]

    # Fail fast if another gradient is alive: NeMo's scaling would touch it too,
    # and the norm accounting assumes each gradient is seen exactly once.
    def _check_only_gradient(self, p: torch.Tensor) -> None:
        for mp in self.model_parts:
            for name, q in mp.named_parameters():
                if q is not p and q.grad is not None:
                    raise RuntimeError(f"update-in-backward: {name} holds a gradient while another parameter's hook runs")

    # Single-class convenience: the one coefficient in force.
    @property
    def coef(self) -> float:
        if len(self.classes) != 1:
            raise AttributeError("coef is only defined with a single clipping class; use coefs")
        return self.coefs[self.classes[0]]

    def _on_grad(self, p: torch.Tensor) -> None:
        if p.grad is None:
            raise RuntimeError("update-in-backward: hook fired without a gradient")
        if id(p) in self._stepped:
            raise RuntimeError("update-in-backward: a parameter's gradient arrived twice in one step (gradient accumulation is not supported)")
        self._check_only_gradient(p)
        self._scale_grads()
        g = _local(p.grad)
        # Sampled before the step so the gradient is still counted; no sync.
        if g.is_cuda:
            self.peak_hook_allocated = max(self.peak_hook_allocated, torch.cuda.memory_allocated(g.device))
        cls = self._class_by_param[id(p)]
        # Unselected rows contribute nothing to the norm and are never stepped.
        mask = self._row_mask.get(id(p))
        if mask is not None:
            g.mul_(mask.view(-1, *([1] * (g.ndim - 1))).to(g.dtype))
        sq = _sum_sq(g)
        event = None
        if sq.is_cuda:
            event = torch.cuda.Event()
            event.record()
        self._sq_parts.append((cls, self._report_class_by_param[id(p)], sq, event))
        # A class before its start step or off its update period contributes to
        # the norm above and is otherwise skipped: no optimizer call, so its
        # moments do not advance (and before the start, do not yet exist).
        since_start = self.step_count - self.start.get(cls, 0)
        if since_start >= 0 and since_start % self.every.get(cls, 1) == 0:
            coef = self.coefs[cls]
            if coef != 1.0:
                g.mul_(coef)
            self.optimizer.step_param(p)
            sampled = self._drift.get(id(p))
            if sampled is not None:
                _, _, index, _, prev, abs_sum = sampled
                cur = _local(p).detach()[index]
                abs_sum += (cur.float() - prev.float()).abs()
                prev.copy_(cur)
        p.grad = None
        self._stepped.add(id(p))

    # Finish a step: every trainable parameter must have been stepped; reduce
    # the squared norm over the world and derive the next step's coefficient.
    # Returns this step's exact pre-clip global gradient norm.
    def end_step(self) -> float:
        missing = len(self._params) - len(self._stepped)
        if missing:
            raise RuntimeError(f"update-in-backward: {missing} trainable parameters received no gradient this step")
        # Make the reducing stream wait for every hook's contribution, and tell
        # the allocator those tensors are now used here so their memory is not
        # recycled by the producing stream before this read completes.
        if self._sq_parts and self._sq_parts[0][2].is_cuda:
            current = torch.cuda.current_stream()
            for _, _, sq, event in self._sq_parts:
                current.wait_event(event)
                sq.record_stream(current)
        # One tensor of per-class sums (clipping classes, then reporting classes)
        # so a single collective serves every class.
        device = self._sq_parts[0][2].device
        n_clip = len(self.classes)
        totals = torch.zeros(n_clip + len(self.report_classes), dtype=torch.float32, device=device)
        for cls, rcls, sq, _ in self._sq_parts:
            totals[self.classes.index(cls)] += sq
            totals[n_clip + self.report_classes.index(rcls)] += sq
        if dist.is_initialized():
            dist.all_reduce(totals, group=self._group)
        # Checked after the reduction so every rank sees the same value and
        # raises together instead of one rank leaving the others in the collective.
        all_values = totals.tolist()
        values = all_values[:n_clip]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"update-in-backward: non-finite gradient norm (sums of squares {dict(zip(self.classes, values))}) at step {self.step_count}")
        self.last_norms = {c: math.sqrt(v) for c, v in zip(self.classes, values)}
        self.last_report_norms = {c: math.sqrt(v) for c, v in zip(self.report_classes, all_values[n_clip:])}
        self.coefs = {c: min(1.0, self.max_norm / (n + 1e-6)) for c, n in self.last_norms.items()}
        norm = math.sqrt(sum(values))
        self.last_norm = norm
        self.last_drift = self._drift_stats()
        self.step_count += 1
        self._stepped.clear()
        self._sq_parts = []
        self.last_peak_hook_allocated = self.peak_hook_allocated
        self.peak_hook_allocated = 0
        return norm

    # Per drift class: mean |master - baseline| and sign consistency over every
    # rank's samples (sums reduced over the world).
    def _drift_stats(self) -> dict[str, dict[str, float]]:
        if not self._drift:
            return {}
        device = next(iter(self._drift.values()))[3].device
        # One vector: per class then per parameter, three sums each (|drift|,
        # sum|update|, elements), reduced once. Every rank holds the same name
        # list (dense shards exist on every rank; expert names are per layer).
        n_cls, names = len(self.drift_classes), self._drift_names
        sums = torch.zeros(3 * (n_cls + len(names)), dtype=torch.float64, device=device)
        for cls, name, index, base, prev, abs_sum in self._drift.values():
            drift = (prev.float() - base.float()).abs().sum()
            moved = abs_sum.sum()
            for i in (3 * self.drift_classes.index(cls), 3 * (n_cls + names.index(name))):
                sums[i] += drift
                sums[i + 1] += moved
                sums[i + 2] += base.numel()
        if dist.is_initialized():
            dist.all_reduce(sums, group=self._group)
        v = sums.tolist()

        def stats(i):
            drift, moved, n = v[i : i + 3]
            return {"mean_abs_drift": drift / n if n else 0.0, "consistency": (drift / moved) if moved > 0 else 0.0, "sampled_elements": int(n)}

        self.last_drift_by_param = {name: stats(3 * (n_cls + k)) for k, name in enumerate(names)}
        return {cls: {k: val for k, val in stats(3 * k_cls).items() if k != "sampled_elements"} for k_cls, cls in enumerate(self.drift_classes)}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# Build a stepper for a NeMo training recipe after ``recipe.setup()``. Checks the
# topology the single all-reduce relies on and disables the recipe's own clip
# (``max_grad_norm = None``), since the YAML cannot express "no clipping": an
# absent or null key falls back to 1.0 in the recipe. ``dense_every`` sets the
# dense update period and ``dense_start`` the step dense first updates; either
# needs the dense/expert classifier and therefore implies per-class clipping
# (a no-effect lever, stage 5 phase C).
def from_recipe(recipe, max_norm: float, per_class: bool = False, dense_every: int = 1, dense_start: int = 0,
                rows_of: Callable[[str], list[int] | None] | None = None) -> InBackwardStepper:
    from nemo_automodel.components.training.utils import (
        get_expert_tp_replication_factor,
        scale_grads_and_clip_grad_norm,
    )

    if recipe.step_scheduler.grad_acc_steps != 1:
        raise ValueError(f"update-in-backward needs one microbatch per step; the recipe accumulates {recipe.step_scheduler.grad_acc_steps}")
    dcfg = recipe.cfg.distributed
    for key in ("tp_size", "pp_size", "cp_size"):
        if int(dcfg.get(key, 1)) != 1:
            raise ValueError(f"update-in-backward supports {key} = 1 only; the recipe sets {dcfg.get(key)}")
    world = dist.get_world_size() if dist.is_initialized() else 1
    dp = recipe._get_dp_group_size(include_cp=True)
    if dp != world:
        raise ValueError(f"update-in-backward needs the data-parallel group to span the world; dp {dp}, world {world}")
    moe_mesh = recipe.moe_mesh
    ep_axis = "ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None
    if ep_axis is not None and moe_mesh["ep"].size() != world:
        raise ValueError(f"update-in-backward needs the expert-parallel group to span the world; ep {moe_mesh['ep'].size()}, world {world}")

    def scale_grads() -> None:
        scale_grads_and_clip_grad_norm(
            None,
            recipe.model_parts,
            norm_type=2.0,
            pp_enabled=recipe.pp_enabled,
            device_mesh=recipe.device_mesh,
            moe_mesh=moe_mesh,
            ep_axis_name=ep_axis,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=None,
            dp_group_size=dp,
            expert_tp_replication_factor=get_expert_tp_replication_factor(recipe.model_parts, recipe.device_mesh),
            grad_norm_backend=recipe.cfg.get("clip_grad_norm.backend", "triton"),
        )

    recipe.max_grad_norm = None
    if dense_every < 1:
        raise ValueError(f"dense_every must be a positive integer, got {dense_every}")
    if dense_start < 0:
        raise ValueError(f"dense_start must be a nonnegative integer, got {dense_start}")
    # Dense (attention, shared experts) and routed experts clipped separately.
    by_class = lambda name: "expert" if ".mlp.experts" in name else "dense"
    class_of = by_class if (per_class or dense_every > 1 or dense_start > 0) else None
    every = {"dense": dense_every} if dense_every > 1 else None
    start = {"dense": dense_start} if dense_start > 0 else None
    return InBackwardStepper(recipe.model_parts, recipe.optimizer[0], max_norm, scale_grads, class_of=class_of, every=every, start=start,
                             report_class_of=by_class, rows_of=rows_of, drift_sample_of=drift_sample)


# Drift sample: every dense tensor's shard, its first rows up to one DRIFT_FRACTION
# of them (at least one row), and the first DRIFT_ROWS rows of local expert 0's
# gate/up matrix per layer. About 0.7 GB per rank of baseline, previous value,
# and |update| sums at the full model's shapes. Every dense tensor is sampled so
# its sign consistency ranks it for selection (last_drift_by_param).
DRIFT_ROWS = 512
DRIFT_FRACTION = 8


def drift_sample(name: str, param: torch.Tensor) -> tuple[str, tuple] | None:
    local = _local(param)
    if ".mlp.experts" not in name:
        return "dense", (slice(0, max(1, local.shape[0] // DRIFT_FRACTION)),)
    if name.endswith(".mlp.experts.gate_and_up_projs") and local.ndim == 3 and local.shape[0] > 0:
        return "expert", (0, slice(0, min(DRIFT_ROWS, local.shape[1])))
    return None
