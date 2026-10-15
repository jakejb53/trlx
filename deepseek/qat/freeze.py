"""Keep NeMo's router correction bias frozen during QAT.

NeMo AutoModel's DeepSeek-V4 build hardcodes ``train_gate=True`` and
``gate_bias_update_factor=1e-3`` in the model's ``MoEConfig``; no recipe key
reaches it. After every optimizer step the training recipe calls
``update_moe_gate_bias``, which moves each scored gate's
``e_score_correction_bias`` buffer (loaded from the source ``ffn.gate.bias``)
by the factor per expert according to observed expert load. The published
checkpoint copies frozen tensors byte for byte, so a drifting bias would make
the trained model and the exported model route differently.

``freeze_gate_bias`` zeroes the factor on the model's ``MoEConfig`` so NeMo's
own guard in ``update_moe_gate_bias`` returns before touching any gate. It is
called once after the recipe has built and loaded the model.
"""

from torch import nn


# Zero the gate-bias update factor and confirm the buffers it would have moved
# are present. Returns (previous factor, count of gates carrying the buffer).
# The factor is read before being zeroed so callers can report that the
# default really was active. Gates are classified by duck typing: a gate whose
# ``bias_update_factor`` is positive is one NeMo would update, and it must hold
# a loaded ``e_score_correction_bias``; hash-routing gates have neither and are
# skipped. A missing ``moe_config``, a missing ``layers`` mapping, or an
# updating gate without its buffer is a structural mismatch with the NeMo
# version this module was written against and raises.
def freeze_gate_bias(model: nn.Module) -> tuple[float, int]:
    inner = getattr(model, "model", None)
    moe_config = getattr(inner, "moe_config", None)
    if moe_config is None or not hasattr(moe_config, "gate_bias_update_factor"):
        raise AttributeError("model.model.moe_config.gate_bias_update_factor not found; NeMo DeepSeek-V4 layout changed")
    layers = getattr(inner, "layers", None)
    if layers is None or not hasattr(layers, "items"):
        raise AttributeError("model.model.layers mapping not found; NeMo DeepSeek-V4 layout changed")

    previous = float(moe_config.gate_bias_update_factor)
    with_bias = 0
    for key, block in layers.items():
        gate = getattr(getattr(block, "mlp", None), "gate", None)
        if gate is None or float(getattr(gate, "bias_update_factor", 0.0)) <= 0:
            continue
        if getattr(gate, "e_score_correction_bias", None) is None:
            raise ValueError(f"layer {key}: scored gate has no e_score_correction_bias buffer, so the source gate bias was not loaded")
        with_bias += 1

    # NeMo's update_moe_gate_bias checks this field on every step and returns
    # early when it is not positive; the gates' own copies of the factor are
    # left alone so their load tracking is unchanged.
    moe_config.gate_bias_update_factor = 0.0
    return previous, with_bias
