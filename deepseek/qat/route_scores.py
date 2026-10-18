"""Routing-concentration measurement and ESFT-style expert selection on the proxy.

Runs the base model forward over the training rows with a hook on every layer's
MoE gate and accumulates, per layer and routed expert, the number of top-k slots
the expert received and its summed gate weight over valid tokens. From those it
reports ESFT's two scores (Wang et al. 2024, ``deepseek/ESFT``): ``token_scores``,
each expert's share of routed slots, and ``gate_scores``, its summed gate weight
per token; how many experts cover 20%, 50%, and 80% of routed slots; and the
selection: with ``--top-p``, experts taken in descending score until the
cumulative score reaches ``top_p``; with ``--per-rank N``, the N highest-scoring
experts inside each expert-parallel rank's contiguous expert range, in every
layer, so split expert storage (stage 6b) gets even shards; the report records
the rank count. ``--hash-per-rank M`` gives the hash-routing layers their own
quota (0 freezes them, the per-rank total trains all of them: share does not
predict their flips, Handoff 5.4), default the ``--per-rank`` value. The
selection is the trainable set for ESFT-style training; the coverage says
whether selection concentrates signal on this data or only saves memory.

The forward is the recipe's own validation epoch over the train dataloader, so
batch handling and eval mode are NeMo's; the hook reads ``token_mask`` from the
gate's inputs and ``(weights, indices)`` from its output. Hash-routing layers
route by token id and are scored the same way. Counts are summed over ranks,
since each rank routes its own rows. Launch from the repo root:

  deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.route_scores \\
      --config deepseek/qat/proxy-train.yaml --top-p 0.2 --out SCORES.json
"""

import argparse
import json
import sys
import time

import torch
import torch.distributed as dist

from deepseek.qat.overrides import add_override_arguments, apply_overrides

COVERAGE_FRACTIONS = (0.2, 0.5, 0.8)


# Add one gate call's routing to the running totals: one slot per (valid token,
# selected expert) into `counts`, its gate weight into `gate_sum`. `indices` and
# `weights` are [tokens, topk]; `token_mask` is [tokens] or None (all valid).
def accumulate(counts: torch.Tensor, gate_sum: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
               token_mask: torch.Tensor | None) -> None:
    topk = indices.shape[-1]
    idx = indices.reshape(-1, topk)
    w = weights.reshape(-1, topk).float()
    if token_mask is not None:
        mask = token_mask.reshape(-1).bool()
        if mask.shape[0] != idx.shape[0]:
            raise ValueError(f"token_mask has {mask.shape[0]} entries for {idx.shape[0]} routed tokens")
        idx = idx[mask]
        w = w[mask]
    flat = idx.reshape(-1).long()
    counts.index_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
    gate_sum.index_add_(0, flat, w.reshape(-1).to(gate_sum.dtype))


# ESFT's scores from the totals. `token_scores` sums to 1 over experts (share of
# routed slots); `gate_scores` is the summed weight divided by the token count,
# which is slots / topk.
def scores(counts: torch.Tensor, gate_sum: torch.Tensor, topk: int) -> tuple[list[float], list[float]]:
    slots = float(counts.sum())
    if slots == 0:
        raise ValueError("no routed slots were counted")
    tokens = slots / topk
    return (counts.double() / slots).tolist(), (gate_sum.double() / tokens).tolist()


# Number of experts, taken in descending score, needed to reach each fraction
# of the total score.
def coverage(score: list[float], fractions=COVERAGE_FRACTIONS) -> dict[str, int]:
    ordered = sorted(score, reverse=True)
    total = sum(ordered)
    out = {}
    for f in fractions:
        acc = 0.0
        n = 0
        for v in ordered:
            if acc >= f * total:
                break
            acc += v
            n += 1
        out[str(f)] = n
    return out


# ESFT's selection rule (generate_expert_config.py): experts in descending score
# while the cumulative score is below top_p, so the expert that crosses the
# threshold is included.
def select(score: list[float], top_p: float) -> list[int]:
    if not (0 < top_p <= 1):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    ordered = sorted(range(len(score)), key=lambda e: score[e], reverse=True)
    chosen = []
    acc = 0.0
    for e in ordered:
        if acc >= top_p:
            break
        chosen.append(e)
        acc += score[e]
    return sorted(chosen)


# Even-shard selection for split expert storage (stage 6b): the n highest-scoring
# experts inside each rank's contiguous range [r*E/R, (r+1)*E/R), ties to the
# lower index, so every expert-parallel rank holds the same count per layer.
# n = 0 is a layer with no trained experts (every expert stays packed).
def select_per_rank(score: list[float], n: int, ranks: int) -> list[int]:
    e = len(score)
    if ranks < 1 or e % ranks:
        raise ValueError(f"{e} experts do not divide over {ranks} ranks")
    per_rank = e // ranks
    if not (0 <= n <= per_rank):
        raise ValueError(f"per-rank count must be in [0, {per_rank}], got {n}")
    chosen = []
    for r in range(ranks):
        lo = r * per_rank
        ordered = sorted(range(lo, lo + per_rank), key=lambda i: -score[i])
        chosen.extend(ordered[:n])
    return sorted(chosen)


# Per-layer report: both scores, coverage of the slot share, the largest share,
# and the selection by the chosen score function: cumulative top_p, or per_rank
# experts within each of `ranks` expert ranges.
def summarize_layer(counts: torch.Tensor, gate_sum: torch.Tensor, topk: int, score_function: str,
                    top_p: float | None = None, per_rank: int | None = None, ranks: int = 1) -> dict:
    if (top_p is None) == (per_rank is None):
        raise ValueError("exactly one of top_p and per_rank selects the experts")
    token_scores, gate_scores = scores(counts, gate_sum, topk)
    by_name = {"token": token_scores, "gate": gate_scores}
    score = by_name[score_function]
    return {
        "tokens": int(counts.sum()) // topk,
        "token_scores": [round(v, 6) for v in token_scores],
        "gate_scores": [round(v, 6) for v in gate_scores],
        "coverage_experts": coverage(token_scores),
        "max_token_share": round(max(token_scores), 6),
        "experts_with_zero_slots": int((counts == 0).sum()),
        "selected": select(score, top_p) if top_p is not None else select_per_rank(score, per_rank, ranks),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="write the rank-0 report here as JSON")
    parser.add_argument("--top-p", type=float, default=None, help="cumulative score threshold for the selection")
    parser.add_argument("--per-rank", type=int, default=None, metavar="N",
                        help="select the N highest-scoring experts of each rank's expert range in every layer (even shards for "
                             "split expert storage); the report records the rank count")
    parser.add_argument("--hash-per-rank", type=int, default=None, metavar="M",
                        help="per-rank count for the hash-routing layers only (0 to the per-rank total; default: the --per-rank "
                             "value); requires --per-rank")
    parser.add_argument("--score-function", choices=("token", "gate"), default="token",
                        help="which ESFT score the selection uses (default: token, the share of routed slots)")
    add_override_arguments(parser)
    args = parser.parse_args(argv)
    if (args.top_p is None) == (args.per_rank is None):
        parser.error("exactly one of --top-p and --per-rank is required")
    if args.top_p is not None and not (0 < args.top_p <= 1):
        parser.error(f"--top-p must be in (0, 1], got {args.top_p}")
    if args.per_rank is not None and args.per_rank < 1:
        parser.error(f"--per-rank must be a positive integer, got {args.per_rank}")
    if args.hash_per_rank is not None:
        if args.per_rank is None:
            parser.error("--hash-per-rank requires --per-rank")
        if args.hash_per_rank < 0:
            parser.error(f"--hash-per-rank must be a nonnegative integer, got {args.hash_per_rank}")
    hash_per_rank = args.hash_per_rank if args.hash_per_rank is not None else args.per_rank

    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    t0 = time.monotonic()
    cfg = load_yaml_config(args.config)
    overrides = apply_overrides(cfg, args.set, args.unset)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = recipe.model_parts[0]
    device = next(model.parameters()).device
    n_experts = int(model.config.n_routed_experts)
    topk = int(model.config.num_experts_per_tok)

    # One pair of totals per layer, filled by the gate hooks. The hook sees the
    # gate's positional inputs (x, token_mask, cp_mesh) and its output
    # (weights, indices, aux_loss).
    totals: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    hash_layers: list[int] = []
    handles = []
    for key, layer in model.model.layers.items():
        i = int(key)
        gate = layer.mlp.gate
        if type(gate).__name__ == "DeepseekV4HashGate":
            hash_layers.append(i)
        counts = torch.zeros(n_experts, dtype=torch.int64, device=device)
        gate_sum = torch.zeros(n_experts, dtype=torch.float64, device=device)
        totals[i] = (counts, gate_sum)

        def hook(module, inputs, output, _i=i):
            weights, indices = output[0], output[1]
            token_mask = inputs[1] if len(inputs) > 1 else None
            accumulate(totals[_i][0], totals[_i][1], indices.detach(), weights.detach(), token_mask)

        handles.append(gate.register_forward_hook(hook))

    t1 = time.monotonic()
    recipe._run_validation_epoch(recipe.dataloader)
    for h in handles:
        h.remove()
    if dist.is_initialized():
        for counts, gate_sum in totals.values():
            dist.all_reduce(counts)
            dist.all_reduce(gate_sum)

    # The rank count is the expert-parallel size this selection is even for;
    # the recipe's expert mesh spans the world (inbackward.from_recipe checks it).
    ranks = dist.get_world_size() if dist.is_initialized() else 1
    report = {
        "rank": rank,
        "config": args.config,
        "overrides": overrides,
        "n_routed_experts": n_experts,
        "topk": topk,
        "top_p": args.top_p,
        "per_rank": args.per_rank,
        "hash_per_rank": hash_per_rank,
        "ranks": ranks,
        "score_function": args.score_function,
        "hash_routing_layers": hash_layers,
        "setup_seconds": round(t1 - t0, 1),
        "forward_seconds": round(time.monotonic() - t1, 1),
        # Hash-routing layers take their own per-rank quota; top_p applies alike.
        "layers": {str(i): summarize_layer(c, g, topk, args.score_function, top_p=args.top_p,
                                           per_rank=None if args.top_p is not None else (hash_per_rank if i in hash_layers else args.per_rank),
                                           ranks=ranks)
                   for i, (c, g) in sorted(totals.items())},
    }
    report["selected_total"] = sum(len(v["selected"]) for v in report["layers"].values())
    if rank == 0:
        for i, v in report["layers"].items():
            tag = " (hash)" if int(i) in hash_layers else ""
            print(f"layer {i}{tag}: tokens {v['tokens']}, experts covering 20/50/80% of slots "
                  f"{v['coverage_experts']['0.2']}/{v['coverage_experts']['0.5']}/{v['coverage_experts']['0.8']}, "
                  f"max share {v['max_token_share']:.4f}, zero-slot experts {v['experts_with_zero_slots']}, "
                  f"selected {len(v['selected'])}", file=sys.stderr, flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        rule = (f"top_p {args.top_p}" if args.top_p is not None
                else f"{args.per_rank} per rank over {ranks} ranks, hash layers {hash_per_rank}")
        print(f"wrote {args.out} ({report['selected_total']} experts selected at {rule})", file=sys.stderr, flush=True)
    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    code = main()
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(code)
