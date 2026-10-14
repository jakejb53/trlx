# DeepSeek-V4-Flash-0731 full-parameter QAT plan

Goal: post-train `deepseek-ai/DeepSeek-V4-Flash-0731` and publish a standalone
checkpoint identical to the source in every file, tensor name, shape, dtype, shard
layout, and scale grid, except the payloads of trained tensors and their scales. No
adapter, no PEFT metadata, no optimizer state. It must load and generate on the
same hardware as the source, with the 2x RTX PRO 6000 (192 GB) as the reference.

Method: full-parameter quantization-aware training (QAT). Every loss-bearing forward
uses `dequant(Q(master))` for each trained tensor, where `Q` is the exact released
quantizer; a straight-through estimator carries gradients to the masters. Publication
is a snapshot of `Q(master)`. There is no post-training requantization step.

`SPEC-deepseek-lora.md` is superseded by this plan and has not been revised.

## Handoff

Read this first when resuming. Stages 0 to 3 are complete; stage 4 is next. The
dataset, optimizer, trainable set, and mechanism are all decided and verified below.
Every command runs from the repository root `/home/goon/trl/trlx`.

Run all unit tests (CPU, under a minute; the data tests need the snapshot):

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_quant deepseek.tests.test_export \
    deepseek.tests.test_fakequant deepseek.tests.test_data
```

Reproduce the stage 3 one-step check (about 5 minutes, both GPUs):

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
    -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --out deepseek/exports/logs/proxy-stage3.json
```

Reports from this session are in `deepseek/exports/logs/`: `proxy-stage2.log`,
`proxy-stage3-run5.json` (stage 3 acceptance), `proxy-stage3-diag.json` (stage 3
diagnostics). `deepseek/exports/identity/` and `deepseek/exports/one-tensor/` are
the two stage 1 exports, 156 GB each, and can be deleted.

Open items carried into stage 4 and 5, in order:

1. `deepseek/qat/proxy_step.py` counts code flips by comparing packed bits, which
   counts the FP4 +0/-0 pair (codes 0 and 8) as a flip. Change it to compare
   dequantized values before any multi-step measurement.
2. `deepseek/qat/proxy_step.py` keeps host copies of sampled masters; the full set
   (26 GB per rank) exceeded this host's 122 GB alongside NeMo's load staging. Keep
   the sampling.
3. The gate recipe is single-step (`max_steps: 1`, global batch 4). Stage 5 needs
   steps, batch, and a held-out evaluation added.

## Decisions

- Full-parameter QAT is the only path pursued until exhausted. LoRA-QAT is measured
  in the proxy gate as a fallback, not developed further unless full-parameter fails.
- Training hardware: 8x B200 (180 GB each) for the full model. The 8x RTX PRO 6000
  with 1.5 TB host RAM is the fallback and needs custom host-offloaded FP32 masters.
- Development, unit tests, and the four-layer proxy gate run on the owned 2x RTX PRO
  6000 in a host venv. A Docker image is built only for the B200, from the lockfile
  of the venv that passed the gate, and is not tested locally.
- Engine: NeMo AutoModel 0.8.0, the source copy in `deepseek/automodel/` from
  https://github.com/nvidia-nemo/automodel. It supplies the model definition, FSDP2,
  expert parallelism, and grouped expert execution. It provides no weight fake-quant,
  no low-bit optimizer usable on DTensors, and no quantized export; those are this
  project's code.
- Trainable set: every FP8 and FP4 weight of the 43 backbone layers except the
  indexer's `wq_b`. The indexer only produces a discrete top-k selection, so the
  language-model loss gives it no gradient, and NeMo freezes every indexer parameter
  unconditionally. Frozen: MTP, DSpark, embeddings, head, router gates,
  hyper-connection parameters, norms, compressor and indexer, and all buffers.
  Frozen tensors are copied byte for byte at export.
- Master precision: BF16 masters with stochastic-rounding updates and torchao
  `AdamW8bit` (8-bit blockwise moments, block 256) on the B200 and on the proxy.
  FP32 masters do not fit the proxy (27B parameters at 4 bytes plus moments on two
  96 GB GPUs). The FP32-master comparison in stage 4 uses a configuration that fits.
- Optimizer wiring: `deepseek/qat/optim.py` runs the torchao optimizer over each
  parameter's local shard, split along dim 0 into chunks of at most 2^28 elements.
  torchao 0.18 fails on DTensor parameters under torch 2.10 (compiled: fake-tensor
  conversion of the quantized state; eager: sharding propagation of `lerp`), and its
  step materializes several FP32 copies of each tensor it updates, which exceeded GPU
  memory on a 2-billion-element expert tensor. Shard views share storage with the
  DTensor, and chunked updates were verified bit-identical to unchunked ones.
- FP4 weight quantizer: OCP MX floor rule, `scale = 2**(floor(log2 absmax) - 2)`,
  saturating at 6. FP8: `scale = 2**ceil(log2(absmax / 448))`, never saturating.
  Both established against the released tensors (Verified facts).
- Dataset for stages 3 to 5: `data/engineering.jsonl`, 158 rows of one user turn,
  one assistant turn, and a separate `reasoning` string. Chosen for being in the
  exact format and token length the full run will use, not for quality; it targets a
  different model and is still being cleaned. The last 16 rows are held out. The
  gate measures mechanism (flips, loss response, held-out loss), not behavior.
- Everything for this work lives under `deepseek/`. Git is managed by the operator.

## Layout

```
deepseek/
  PLAN-deepseek-lora.md, SPEC-deepseek-lora.md
  automodel/                                   NeMo AutoModel 0.8.0 source
  models--deepseek-ai--DeepSeek-V4-Flash-0731/ HF cache layout; snapshot
                                               7872f01b1d1fe23eabc4c98b48bffcef5a386062
  venv/                                        NeMo environment (uv, Python 3.12.13)
  exports/                                     exported checkpoints (156 GB each), logs/
  qat/                                         project code; imports nothing from trlx
    quant.py          exact FP8/FP4 quantizers and dequantizers
    roundtrip.py      read-only checkpoint round-trip check
    export.py         layout-preserving exporter
    verify_export.py  shard/tensor/file diff of two checkpoints
    fakequant.py      straight-through fake-quant hooks on the NeMo model
    data.py           reasoning-conversation dataset bridge (NeMo shifted format)
    optim.py          torchao optimizer over local shards
    proxy.py          stage 2 parity check (torchrun entry)
    proxy_step.py     stage 3 one-step check and diagnostics (torchrun entry)
    proxy-4layer.yaml stage 2 recipe (mock token ids)
    proxy-train.yaml  training recipe: data bridge, freeze config, optimizer
  tests/                                       unittest, run from the repo root
```

Snapshot path, used below as `$SNAP`:
`deepseek/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062`

## Environment

- `deepseek/venv/`: created from `deepseek/automodel/uv.lock` with
  `UV_PROJECT_ENVIRONMENT=$PWD/deepseek/venv uv sync --frozen` run in
  `deepseek/automodel/`. Base dependencies only: torch 2.10.0+cu130, transformers
  5.18.0, torchao 0.18.0, NeMo editable. No `cuda`/`moe` extras: the fake-quant
  design needs no Transformer Engine, TileLang, or DeepEP.
- Verified: both GPUs are compute capability 12.0 and in torch's arch list;
  matmuls run on both. torchao's C++ extensions are skipped on torch 2.10; its
  Triton paths work (the optimizer step compiles and runs).
- Stages 0 and 1 also run under `/home/goon/trl/venv` (torch 2.13); either works.
- Host: 2x RTX PRO 6000 Blackwell (97,887 MiB each), driver 595.91, CUDA 13.3
  toolkit, 122 GB RAM, 48 cores, 1.4 TB free on the repo filesystem. Host RAM is the
  binding limit for check scripts that copy masters; see Handoff.
- Launch training with `PYTORCH_ALLOC_CONF=expandable_segments:True`; the step
  peaked at 85 GB per GPU with 7 to 11 GB of allocator fragmentation without it.

## Stages

Effort and confidence are session metadata and are not recorded here.

### Stage 0: exact quantizers (complete)

`deepseek/qat/quant.py`: FP8 128x128 block and FP4 32-group quantizers and
dequantizers, pure PyTorch, with E8M0 scales built from exponent integers (torch's
float-to-E8M0 cast cannot produce 2**-127). FP4 sign bits follow the input sign bit
including -0.0. `deepseek/qat/roundtrip.py` checks every quantized tensor of a
snapshot: dequantize, requantize, compare values and bits.

- `deepseek/tests/test_quant.py`: 20 tests pass.
- Full round trip over all 35,718 quantized tensors (47 minutes): value-exact on
  every tensor; FP4 bit-exact on all 35,328; FP8 differs only on the 15,953 blocks
  whose largest code is exactly 224, where the source stores scale 2s with code 224
  and the ceiling rule stores scale s with code 448, the same value; no all-zero
  blocks anywhere.

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_quant
deepseek/venv/bin/python -B -m deepseek.qat.roundtrip $SNAP [--limit N]
```

### Stage 1: layout-preserving exporter (complete)

`deepseek/qat/export.py`: writes the 48 shards with headers copied verbatim,
replaced payloads written from caller tensors that must match the source entry's
shape, dtype, and byte count, and all other payloads and non-weight files copied
byte for byte. Staged, `--force` to replace. `deepseek/qat/verify_export.py` hashes
two trees and names every differing shard, tensor, and file; `--expect` asserts
the exact differing set.

- `deepseek/tests/test_export.py`: 7 tests pass.
- Zero-replacement export of the real snapshot: byte-identical, 116 s.
- One-tensor replacement (`layers.5.ffn.experts.17.w2` codes and scales):
  `--expect` verified exactly those two tensors differ, in one shard, 183 s.

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_export
deepseek/venv/bin/python -B -m deepseek.qat.verify_export $SNAP deepseek/exports/<name> [--expect NAME ...]
```

### Stage 2: four-layer proxy with fake-quant at zero delta (complete)

Build `DeepseekV4Config.from_pretrained($SNAP)` with `num_hidden_layers: 4`,
`num_nextn_predict_layers: 0`, backend `torch` dispatcher, `torch_mm` experts,
`sdpa` attention, FSDP2 with `ep_size: 2`. Layers 0 to 3 cover sliding-window,
compressor plus indexer, compressor only, hash routing, and scored routing.

`deepseek/qat/fakequant.py`: an autograd function returning `dequant(quant(w))`
with identity backward, applied by replacing each target module's forward with
the original run under `torch.nn.utils.stateless._reparametrize_module` (the
context behind `torch.func.functional_call`, whose public entry re-enters the
module call and would recurse) with fake-quantized tensors swapped in.
Parameters, FSDP2 wrapping, and the optimizer are untouched. DTensor parameters
are quantized on the local shard and re-wrapped. Targets per layer: `self_attn`
`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`; `compressor.indexer.wq_b` (hooked but
frozen); shared-expert `gate_proj`, `up_proj`, `down_proj`;
`mlp.experts.gate_and_up_projs` and `down_projs`. Expert tensors are transposed
views of HF `[out, in]` layout (`gate_and_up_projs[e] = cat(w1.T, w3.T)`,
`down_projs[e] = w2.T`), so the FP4 group runs along the first non-expert axis.

Results on both ranks: 37 targets per rank; fake-quantized tensors equal the
loaded tensors bit for bit; loaded dense tensors and every local expert equal the
snapshot's dequantized tensors bit for bit, so NeMo's dequantization agrees with
stage 0 in both orientations; hooked, unhooked, and post-uninstall logits on
fixed token ids are bit-identical. Load: 9.93 GB in about 200 s, nearly all in
NeMo's dequantization; 356 s total. Parameters load as BF16 with FP32 islands.

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_fakequant
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy \
    --config deepseek/qat/proxy-4layer.yaml --snapshot $SNAP
```

### Stage 3: straight-through backward, one update (complete)

`deepseek/qat/data.py`: encodes each row with the snapshot's own
`encoding/encoding_dsv4.py` in thinking mode with the reasoning in the assistant
turn, tokenizes with the snapshot tokenizer, and emits NeMo's shifted format:
`input_ids[:-1]`, `labels[1:]` with prompt positions -100, `attention_mask`, and
the pad-id entry. The prompt is everything through `<think>`; reasoning,
`</think>`, answer, and EOS are supervised. Rows above `max_tokens` are errors.
Measured: 768 to 4,319 tokens per row, mean 2,960, 156 of 158 within 4,096;
prompt tokenization is an exact prefix at the `<think>` boundary.

`deepseek/qat/proxy-train.yaml`: the stage 2 model, the data bridge
(`max_tokens: 4352`, holdout 16), NeMo `freeze_config` freezing `model` and
`lm_head` then unfreezing the targets by glob, activation checkpointing, gradient
clipping at 1.0, `deepseek.qat.optim.adamw8bit` at lr 1e-5 with
`bf16_stochastic_round: true`, global batch 4, one step.

`deepseek/qat/proxy_step.py`: snapshots quantized codes and scales of every
target and a digest of every frozen parameter, runs the recipe's training loop,
then requantizes and reports flips, master deltas, gradient underflow fractions,
and frozen-parameter integrity. Dense parameters are compared as full tensors
because FSDP2 leaves them unsharded after a forward when resharding is off;
parameter names carry a `_checkpoint_wrapped_module` segment under activation
checkpointing and are canonicalized.

Results (`proxy-stage3-run5.json`, `proxy-stage3-diag.json`):

- Trainable set: 40 parameters, 13.1B elements per rank, exactly the targets
  minus the indexer. Loss 16.86, gradient norm 25.70, both finite. Step 21 to
  23 s; peak GPU memory 85.1 GB. No frozen parameter changed by a bit; the
  indexer is unchanged.
- FP8 codes: about 1% changed value, matching Adam's first step (every element
  moves by about the learning rate) against the 0.84% of FP8 elements whose
  half-cell is below 1e-5.
- FP4 codes: 6.4% of codes changed bits, but almost all are the +0/-0 pair
  (13% of FP4 codes are zero and half their masters went slightly negative).
  Real FP4 value changes are about 0.6%, nearly all from groups whose scale
  doubled (see Verified facts). The flip metric must compare values (Handoff 1).
- Master deltas (sampled `wq_b` and `gate_and_up_projs` per layer): 21 to 24% of
  masters changed; about 10% of deltas exceed 2x the learning rate and 0.6%
  exceed 20x, none 200x. The largest, 9.8e-4, are stochastic-rounding jumps of
  one BF16 ulp at large weights, the expected behavior of BF16 masters.
- 8-bit second-moment underflow: gradient elements below torchao's smallest level
  (3.25e-7 of their block maximum) are 0.42% of expert elements at block 256,
  0.27% at 64, 0.22% at 32, and 0.09% for dense. Those elements get a zero second
  moment and an update bounded only by eps 1e-8; measured effect is within the
  0.6% of deltas above 20x. Block 256 is kept for the gate.

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_data
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
    -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --out REPORT.json
```

### Stage 4: memory plan and master-precision comparison (next)

The optimizer and its wiring are done (stage 3). Remaining:

- Update-in-backward for the B200: with `ep_size: 8` each rank owns 32 experts per
  layer outside FSDP; holding a full BF16 gradient for them is about 69 GB. Apply
  the optimizer per parameter as its gradient arrives and free it.
- FP32-master comparison on a configuration that fits two 96 GB GPUs (for example
  one or two layers): same data and steps, BF16 masters with stochastic rounding
  and 8-bit moments against FP32 masters with FP32 AdamW. Compare value flips, loss,
  and held-out loss.

### Stage 5: gate run (planned)

Bounded multi-step runs on the proxy, 20 to 50 steps at global batch 4 to 8 over the
142 training rows: full-parameter with BF16-SR masters, the FP32-master comparison
from stage 4, and rank-8 LoRA merge-in-the-loop. Compare value flips and scale
changes over time, training loss, and held-out loss on the 16 rows. Track groups at
the floor-rule scale boundary (Verified facts). This is the go/no-go verdict for the
B200 run.

### Stage 6: proxy export and reload (planned)

Export the trained proxy with stage 1, verify with `--expect` that only trained
targets differ, reload in a fresh process, and reproduce the final training
forward bit for bit. NeMo-to-HF name mapping for the targets is in
`deepseek/qat/proxy.py` (`hf_names`) and the expert orientation in `fakequant.py`.

### Stage 7: full model on 8x B200 (planned)

Docker image from the gate-passing venv lockfile with a verifier script run first.
One update, bounded convergence run, export, fresh-process equality, Transformers
and vLLM probes on the RTX PRO 6000 pair. Memory plan with `ep_size: 8`: each rank
owns 32 experts per layer outside FSDP; BF16 masters plus 8-bit Adam is about 141 GB
per GPU before activations, 4-bit about 106 GB.

## Verified facts

- Snapshot inventory from safetensors headers: 72,317 tensors, no missing or
  duplicate keys, 166.88 GB. Routed experts: 35,328 I8 tensors of packed FP4,
  296.4B parameters, 148.2 GB plus 9.3 GB E8M0 scales, across 43 layers and 3 MTP
  modules of 256 experts each. FP8 E4M3: 390 tensors, 6.3B parameters. BF16: 445
  tensors, 1.5B parameters. F32/I64: 436 small tensors. Trainable backbone set is
  about 283B parameters. The plan's earlier 155.76B figure counted packed bytes.
- Expert tensor layout: `w1`, `w3` `[2048, 4096]` and `w2` `[4096, 2048]` logical,
  stored `[out, in/2]` I8 with element 2k in the low nibble; scales `[out, in/32]`
  E8M0. Dense FP8 scales are `[out/128, in/128]` E8M0. Scale tensors are named
  `<base>.scale`.
- FP4 scale rule: across 3.9M sampled groups the largest code is only ever 4 or
  6, never 3. The ceiling rule would produce 3 in about a fifth of groups; the
  floor rule cannot. FP8 blocks show the ceiling rule's signature (largest code in
  [224, 448]).
- FP4 code distribution in a sampled expert tensor: 0: 13.0%, 0.5: 12.8%, 1: 22.7%,
  1.5: 8.7%, 2: 17.7%, 3: 12.0%, 4: 9.2%, 6: 3.9%. Codes 0 and 8 (+0 and -0)
  dequantize identically; the checkpoint has 18.76 billion -0 codes.
- Floor-rule scale boundary: a group whose largest code is 4 has absmax exactly
  4 times a power of two, so any upward move of that element doubles the group's
  scale and halves the resolution of all 32 codes. 53% of released expert groups
  are in that state (largest code 4); 47% have largest code 6.
- Quantization half-cells: median FP8 half-cell in `wq_b` is 9.8e-4 and 0.84% of
  elements are below 1e-5; median FP4 half-cell in an expert tensor is 3.9e-3 and
  no element is below 2e-5.
- DeepSeek's reference quantizer is `$SNAP/inference/kernel.py`
  (`fast_round_scale`, `act_quant`, `fp4_act_quant`); `convert.py` holds the FP4
  table and a lossless FP4-to-FP8 cast. The encoder is `$SNAP/encoding/encoding_dsv4.py`
  (`encode_messages(messages, thinking_mode)` with `reasoning_content` on the
  assistant message); special ids: BOS 0, EOS 1, `<think>` 128821, `</think>` 128822.
- NeMo 0.8.0: the checkpoint's `mtp.*` tensors use the DSpark layout, which the
  `deepseek_v4` MTP code does not accept; recipes set `num_nextn_predict_layers: 0`.
  The MXFP4-resident expert path has no weight gradient. Built-in QAT is torchao
  int4/int8 on `nn.Linear` and rejects models with FP32 parameters. Indexer
  parameters are frozen unconditionally. Saving writes dequantized float weights.
  Loading a 4-layer config from the full snapshot reads only the needed tensors
  (9.93 GB). The loss does not shift labels; datasets emit shifted ids. The
  `freeze_config` selector list excludes the root module, so freezing everything
  uses `model` and `lm_head`; `*` crosses dots. NeMo injects `foreach` into any
  optimizer factory whose signature accepts arbitrary keywords.
- Safetensors shards: headers are compact JSON padded with spaces to 8 bytes,
  tensor order equals offset order, no gaps; `json.dumps` with `(",", ":")`
  reproduces them.

## Prior investigation (rented 8x H100, 2026-10-07)

Evidence from the rented host; artifacts under its `/workspace` were not
transferred to this repository.

- NeMo AutoModel at commit 21837edf (0.7.0) trained rank-8 LoRA over attention,
  MXFP4-resident routed experts, and shared experts on the full 43-layer model
  with the torch dispatcher: loss 1.4188 after one update, 43.71 GiB per rank,
  2.6 GB adapter checkpointed and reloaded. HybridEP dispatch hung on that host.
- Post-hoc merge into the source formats was rejected: rank-8 deltas are below
  half a quantization step in most blocks and vanish on requantization. This is
  the finding that motivates QAT.
- A four-layer proxy with the eight shortest rows of a dataset trained in 52 s per
  update at 10.63 GiB per rank on eight ranks.
