# DeepSeek-V4-Flash-0731 full-parameter QAT

Status: active. Replaces the LoRA-and-merge proposal, which was rejected because
small deltas vanish when rounded back onto the source's quantization grids.
`PLAN-deepseek.md` holds stages, measurements, decisions with their evidence, and
the handoff. This file holds the contract.

## 1. Goal

Post-train `deepseek-ai/DeepSeek-V4-Flash-0731` (snapshot
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`) and publish a standalone checkpoint
identical to the source in every file, tensor name, shape, dtype, shard layout,
and scale grid, except the payloads of trained tensors and their scales. No
adapter, no PEFT metadata, no optimizer state. It loads and generates through the
same Transformers and vLLM paths as the source on the same hardware.

## 2. Method

Quantization-aware training on the full trainable set.

- Every loss-bearing forward uses `dequant(Q(master))` for each trained tensor,
  where `Q` is the source format's quantizer (section 3). A straight-through
  estimator carries gradients to the masters.
- Every released tensor is a fixed point of `Q`: quantizing a source tensor
  reproduces its values exactly. A trained tensor therefore changes in the
  published artifact only where its master drifted across a cell boundary.
- Publication is a snapshot of `Q(master)`. There is no post-training
  requantization, merge, or conversion step.

## 3. Quantizers

`deepseek/qat/quant.py` implements both formats exactly, in pure PyTorch.

- FP8 E4M3, 128x128 blocks, E8M0 scale `2**ceil(log2(absmax / 448))`, never
  saturating. Value-exact fixed point on every source tensor. Bit encoding may
  differ from the source only on blocks whose largest code is exactly 224, where
  the source stores scale `2s` with code 224 and the rule stores `s` with 448,
  the same value.
- FP4 E2M1, groups of 32 along the input axis, packed two codes per byte with
  element `2k` in the low nibble, E8M0 scale. Release rule: floor,
  `2**(floor(log2 absmax) - 2)`, saturating at 6. Bit-exact fixed point on every
  source tensor.
- FP4 training rule `min_error`: per group, the floor-rule scale or twice it,
  whichever has the smaller float32 sum of squared errors, ties to the floor
  rule. Stateless. Source groups have zero error at their own scale, so the
  fixed point holds. Used everywhere the project quantizes FP4: the training
  forward, the flip metric, and export. The floor rule is the
  source-reproduction rule only.
- E8M0 scales are built from exponent integers so that `2**-127` is
  representable. FP4 sign bits follow the input sign bit, including `-0.0`.
- `deepseek/qat/roundtrip.py` is the acceptance test: dequantize and requantize
  every quantized tensor of a snapshot and compare values and bits. Against an
  export it runs with `--scale-rule min_error`; under the floor rule, groups
  published at the doubled scale are value-exact but not bit-exact.

## 4. Scope

- SFT on reasoning conversations (section 7).
- Trainable: FP8 and FP4 weights of the 43 backbone layers, except the indexer's
  `wq_b`. Routed experts train only when selected for the run (section 8); the
  selection may be all experts.
- Frozen, byte for byte: MTP, DSpark, embeddings, head, router gates and
  correction biases, hyper-connection parameters, norms, compressor, indexer,
  unselected routed experts, and every buffer.
- Engine: NeMo AutoModel 0.8.0, the source copy in `deepseek/automodel/`, for
  the model definition, FSDP2, expert parallelism, and grouped expert execution.
- Excluded: router, correction-bias, indexer, MTP, and DSpark training;
  conversion to NVFP4 or any other format; any output that is not a fixed point
  of the source quantizers.

## 5. Ownership

`deepseek/qat` owns, and imports nothing from `trlx`:

- Quantizers, round-trip check, exporter, and export verifier.
- Fake-quant hooks, the freeze of NeMo's router-bias update, the optimizer over
  local shards, update-in-backward, expert selection, the data bridge, recipe
  overrides, and the training, parity, scoring, and export entry points.
- Reports, probe logits, replaced-tensor lists, and exports.

NeMo owns model construction and dequantized loading, distributed placement and
collectives, forward, backward, activation checkpointing, expert-gradient
scaling, and the training loop. Two NeMo behaviours are disabled after recipe
setup because the recipe cannot express it: the per-step router correction-bias
update (`deepseek/qat/freeze.py`) and, under update-in-backward, NeMo's own
gradient clip. No NeMo model, parallelism, or kernel code is copied into
`deepseek/qat`.

## 6. Source checkpoint compatibility

The snapshot's safetensors index must be complete with no missing or duplicate
keys. Every tensor is BF16, FP8 E4M3 with an E8M0 `<base>.scale`, packed FP4 I8
with an E8M0 `<base>.scale`, or a small F32/I64 tensor. Block and group geometry
derive from tensor and scale shapes: FP8 scales are `[out/128, in/128]`, FP4
scales `[out, in/32]` over `[out, in/2]` codes. The round-trip check (section 3)
must pass. Inside NeMo, expert tensors are transposed views of the HF `[out, in]`
layout, so the FP4 group runs along the first non-expert axis; the fake-quant
hooks and exporter account for this.

## 7. Dataset

`deepseek/qat/data.py` encodes each row with the snapshot's own
`encoding/encoding_dsv4.py` in thinking mode, with the row's separate `reasoning`
string as the assistant message's reasoning content, and tokenizes with the
snapshot tokenizer. Output is NeMo's shifted format: `input_ids[:-1]`,
`labels[1:]` with prompt positions `-100`, `attention_mask`, and the pad id. The
prompt is everything through `<think>`; reasoning, `</think>`, the answer, and
EOS are supervised. A row above `max_tokens` is an error, never truncated. The
holdout is the last N rows in file order.

## 8. Training

- Fake quantization (`deepseek/qat/fakequant.py`): each target module's forward
  runs under `torch.nn.utils.stateless._reparametrize_module` with
  `dequant(Q(w))` swapped in; backward is identity. Parameters, FSDP2 wrapping,
  and the optimizer are untouched. DTensor parameters are quantized on the local
  shard and re-wrapped.
- Masters: BF16 with stochastic-rounding updates. Optimizer: torchao
  `AdamW8bit`, block 256, run by `deepseek/qat/optim.py` over each parameter's
  local shard in chunks of at most 2^28 elements along dim 0. Chunk views share
  storage with the DTensor.
- Update-in-backward (`deepseek/qat/inbackward.py`): every trainable parameter
  is stepped from its post-accumulate-grad hook and its gradient freed. The exact
  global norm is accumulated as gradients arrive and reduced once per step; the
  previous step's clip coefficient scales the current step; the first step is
  unclipped. Requires one microbatch per step and `tp = pp = cp = 1` with the
  data- and expert-parallel groups spanning the world. Expert-gradient scaling
  remains NeMo's.
- Expert selection: `deepseek/qat/route_scores.py` scores every routed expert by
  the frozen base model's routing over the training rows (`token_scores`, share
  of routed top-k slots; `gate_scores`, summed gate weight) and selects per
  layer in descending score until cumulative `top_p`. The optimizer restricts
  each expert parameter to views of its selected local rows before the first
  step, so unselected rows get no update and no optimizer state; unselected
  rows of each expert gradient are zeroed before the norm.
- Integrity: every run snapshots the quantized codes and scales of every target
  and a digest of every frozen parameter and buffer before training, and checks
  them after. A changed frozen tensor, or any code or scale change in an
  unselected expert, fails the run.

## 9. Artifacts

A training run (`deepseek.qat.proxy_step`) writes its report to `--out`; with
`--export DIR` it also writes probe logits to `<out stem>-logits-rank{r}.pt`, the
replaced tensor names to `<out stem>-replaced.txt`, and the checkpoint to `DIR`.
Reports live in `deepseek/exports/logs/`, exports in `deepseek/exports/`. Exports
are regenerable and may be deleted.

## 10. Export and verification

- `deepseek/qat/export.py` writes every shard with its header copied verbatim.
  A replaced payload must match the source entry's shape, dtype, and byte count.
  Every other payload and every non-weight file is copied byte for byte.
- `deepseek/qat/proxy_export.py` quantizes every trained target with the
  training quantizers (section 3), gathers expert codes and scales to rank 0 over the expert
  mesh, materializes dense tensors in full, and replaces only tensors whose
  bytes differ from the source. The source is the recipe's model path, so a run
  trained on an export exports from that export and every link in a chain is a
  complete checkpoint.
- `deepseek/qat/verify_export.py` hashes two checkpoint trees and names every
  differing shard, tensor, and file; `--expect` asserts that the differing set
  is exactly the replaced list.
- Reload (`deepseek.qat.proxy` against the export): the fixed-point and
  source-parity checks pass, and hooked logits on fixed probe token ids equal
  the saved ones bit for bit.
- Runtime: the published artifact loads through Transformers and vLLM on
  hardware that serves the source and completes deterministic forward and
  generation probes.

## 11. Failure and cancellation

Export is staged and publishes only a complete tree; replacing an existing
destination requires `--force`. An export error is recorded in the report and
fails the run without raising, so every rank reaches the final collective. A
failed run publishes nothing.

## 12. Dependencies and versions

`deepseek/venv/` is created from `deepseek/automodel/uv.lock` with base
dependencies only (torch 2.10.0+cu130, transformers 5.18.0, torchao 0.18.0,
NeMo editable); no `cuda` or `moe` extras. The optimizer's row restriction relies on torchao allocating
optimizer state lazily on the first step; re-verify on any torchao upgrade.

## 13. Deferred extensions

- Publication alternatives if expert QAT does not deliver, ranked in
  `PLAN-deepseek.md`: dense-only QAT, FP8 as the expert publication format,
  stochastic rounding at publication, misaligned grids.
- LoRA-QAT. Its gate arm was dropped before measurement: a LoRA delta after
  tens of steps is about 1e-4 per weight, below both grids' half-cells.
- Methods other than SFT.
- Integration as a `trlx` method.

Each requires its own training, export, and verification contract and must not
weaken the preservation guarantees above.

## 14. Open decisions

- Steady-state memory plan for the full model with the selected trainable set.
- The dense-slowing knob as a recipe setting, with a warmup for any parameter
  class entering training.
- The container for the training host.
- The training dataset, and its distance from the model's own policy.
- An evaluation on prompts outside the SFT distribution.
- The expert score function (`token_scores` or `gate_scores`), and whether the
  full-model selection is taken from `top_p` or measured from per-expert
  gradient-sign consistency and flip fraction.
- Whether to combine selection with dense slowing.
