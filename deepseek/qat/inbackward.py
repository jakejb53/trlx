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
        for mp in model_parts:
            for name, q in mp.named_parameters():
                if q.requires_grad:
                    self._class_by_param[id(q)] = class_of(name) if class_of is not None else "all"
        self.classes: list[str] = sorted(set(self._class_by_param.values()))
        self.coefs: dict[str, float] = {c: 1.0 for c in self.classes}
        self.last_norms: dict[str, float] = {}
        self.last_norm: float | None = None  # global norm over all classes
        self._stepped: set[int] = set()
        # Per-hook squared norms with the CUDA event that marks their completion.
        # Hooks run on different streams (FSDP2 fires dense hooks inside its
        # reduce-scatter stream, autograd fires expert hooks on its own), so a
        # single running sum across them would race; each contribution is kept
        # with its event and summed at step end after waiting on all of them.
        self._sq_parts: list[tuple[str, torch.Tensor, object]] = []
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
        cls = self._class_by_param[id(p)]
        sq = _sum_sq(g)
        event = None
        if sq.is_cuda:
            event = torch.cuda.Event()
            event.record()
        self._sq_parts.append((cls, sq, event))
        coef = self.coefs[cls]
        if coef != 1.0:
            g.mul_(coef)
        self.optimizer.step_param(p)
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
        if self._sq_parts and self._sq_parts[0][1].is_cuda:
            current = torch.cuda.current_stream()
            for _, sq, event in self._sq_parts:
                current.wait_event(event)
                sq.record_stream(current)
        # One tensor of per-class sums so a single collective serves every class.
        device = self._sq_parts[0][1].device
        totals = torch.zeros(len(self.classes), dtype=torch.float32, device=device)
        for cls, sq, _ in self._sq_parts:
            totals[self.classes.index(cls)] += sq
        if dist.is_initialized():
            dist.all_reduce(totals, group=self._group)
        # Checked after the reduction so every rank sees the same value and
        # raises together instead of one rank leaving the others in the collective.
        values = totals.tolist()
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError(f"update-in-backward: non-finite gradient norm (sums of squares {dict(zip(self.classes, values))}) at step {self.step_count}")
        self.last_norms = {c: math.sqrt(v) for c, v in zip(self.classes, values)}
        self.coefs = {c: min(1.0, self.max_norm / (n + 1e-6)) for c, n in self.last_norms.items()}
        norm = math.sqrt(sum(values))
        self.last_norm = norm
        self.step_count += 1
        self._stepped.clear()
        self._sq_parts = []
        return norm

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# Build a stepper for a NeMo training recipe after ``recipe.setup()``. Checks the
# topology the single all-reduce relies on and disables the recipe's own clip
# (``max_grad_norm = None``), since the YAML cannot express "no clipping": an
# absent or null key falls back to 1.0 in the recipe.
def from_recipe(recipe, max_norm: float, per_class: bool = False) -> InBackwardStepper:
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
    # Dense (attention, shared experts) and routed experts clipped separately.
    class_of = (lambda name: "expert" if ".mlp.experts" in name else "dense") if per_class else None
    return InBackwardStepper(recipe.model_parts, recipe.optimizer[0], max_norm, scale_grads, class_of=class_of)
