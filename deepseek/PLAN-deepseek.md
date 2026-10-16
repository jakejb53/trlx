# DeepSeek-V4-Flash-0731 full-parameter QAT plan

Goal: post-train `deepseek-ai/DeepSeek-V4-Flash-0731` and publish a standalone
checkpoint identical to the source in every file, tensor name, shape, dtype, shard
layout, and scale grid, except the payloads of trained tensors and their scales. No
adapter, no PEFT metadata, no optimizer state. It must load and generate on the
same hardware as the source, with the 2x RTX PRO 6000 (192 GB) as the reference.

Method: full-parameter quantization-aware training (QAT). Every loss-bearing forward
uses `dequant(Q(master))` for each trained tensor, where `Q` is the released
quantizer for FP8 and, for FP4, the released floor rule extended with per-group
error-selected scales (Decisions); every released tensor is a fixed point of `Q`.
A straight-through estimator carries gradients to the masters. Publication
is a snapshot of `Q(master)`. There is no post-training requantization step.

## Handoff

Read this first when resuming. Stages 0 to 6 are complete: routed experts train
under QAT (stage 5), and a trained proxy exports to a complete source-layout
checkpoint that reloads bit-identically, twice in a chain (stage 6). The frequency
split (Handoff 4) passed: slowing dense restores expert flips. ESFT-style expert
selection (Handoff 5) is implemented and measured: selected experts flip at the
split arm's rate with dense at full speed, unselected experts stay byte-identical,
peak memory 48 GB against 70. Next: stage 7 preparation (Handoff 6). The dataset,
optimizer, trainable set, and mechanism are decided and verified below. Every
command runs from the repository root `/home/goon/trl/trlx`. Launch GPU runs from
a plain terminal with `nohup`: the post-training report phase drives host memory
low enough that an agent session's background-process reaper has killed a run.

Run all unit tests (CPU, under a minute; the data tests need the snapshot):

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_quant deepseek.tests.test_export \
    deepseek.tests.test_fakequant deepseek.tests.test_data deepseek.tests.test_freeze \
    deepseek.tests.test_inbackward deepseek.tests.test_optim deepseek.tests.test_proxy_step \
    deepseek.tests.test_proxy_export deepseek.tests.test_route_scores
```

Reproduce the stage 3 one-step check (about 5 minutes, both GPUs):

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
    -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --out deepseek/exports/logs/proxy-stage3.json
```

A stage 5 gate run (about 40 minutes, both GPUs; `--trainable-subset dense|experts`
with a `--set` of `freeze_config.unfreeze_modules` for ablations):

```sh
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
    -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --in-backward \
    --set optimizer.lr=1e-4 --set step_scheduler.max_steps=100 --set step_scheduler.val_every_steps=10 \
    --set step_scheduler.global_batch_size=2 --set step_scheduler.local_batch_size=1 --out REPORT.json
```

Handoff 5 runs (both GPUs). Routing scores and `top_p` selection (about 6 minutes);
the ESFT arm (about 40 minutes); the joint arm with the per-expert flip table:

```sh
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.route_scores \
    --config deepseek/qat/proxy-train.yaml --top-p 0.2 --out deepseek/exports/logs/route-scores-engineering.json
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
    -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --in-backward \
    --expert-selection deepseek/exports/logs/route-scores-engineering.json \
    --set optimizer.lr=1e-4 --set step_scheduler.max_steps=100 --set step_scheduler.val_every_steps=10 \
    --set step_scheduler.global_batch_size=2 --set step_scheduler.local_batch_size=1 --out REPORT.json
# per-expert table against routing share: replace --expert-selection with
#   --expert-shares deepseek/exports/logs/route-scores-engineering.json
# frequency split: add --dense-every 4
```

Reports are in `deepseek/exports/logs/`, named by stage (`proxy-stage2.log`,
`proxy-stage3-*.json`, `proxy-stage4-*.json`, `proxy-stage5-*.json`,
`proxy-stage6-*`) plus `proxy-split-dense4.json` (Handoff 4),
`route-scores-engineering.json`, `proxy-esft-top02.json`, and
`proxy-joint-perexpert.json` (Handoff 5). Exports: `deepseek/exports/identity/` and `one-tensor/` (stage 1),
`stage6-experts/` and `stage6-sequential/` (stage 6), 156 GB each; all can be
deleted and regenerated.

Handoff items, in order:

1. `deepseek/qat/proxy_step.py` keeps host copies of sampled masters; the full set
   (26 GB per rank) exceeded this host's 122 GB alongside NeMo's load staging. Keep
   the sampling.
2. Memory: steady-state peaks exceed first-step peaks (stage 5). Measured so far:
   4-layer proxy, batch 2, in-backward, 70.0 GB; dense-only 45.7 GB. The stage 7
   budget must come from multi-step measurements, and the update-in-backward saving
   itself is still unmeasured (sample allocated memory at end of backward with and
   without it).
3. Gate runs save weights with `proxy_step --export DIR` (stage 6).
4. Frequency split (complete; `proxy_step --in-backward --dense-every 4`, report
   `proxy-split-dense4.json`). Joint training at 1e-4 for 100 steps with dense
   parameters updated on 1 step in 4 (gradients discarded on the other steps, Adam
   moments untouched, per-class clipping implied) and experts every step. Premise
   (stage 5 phase C): a joint run starves experts because dense fits the data in
   about 30 steps and removes the gradient-sign consistency experts need to cross
   cells; slowing dense keeps the signal alive. Under Adam, 1 in N at rate r and
   every step at r/N drift the same expected distance, so this is the per-class
   rate with every dense update at the proven 1e-4; N = 4 is the FP4/FP8 cell-size
   ratio. Result: expert values changed 2.79% at step 100 (joint 1.45%, joint with
   per-class clip 1.79%, experts-only 4.84%), first flips at step 30 instead of
   50, dense 47.4% (54% at every step), held-out 16.18 to 5.13 (joint 4.17,
   experts-only 5.24), expert gradient norm 4.75 to 1.36 (phase C: to 0.81), peak
   70.0 GB. Pass: the signal-lifetime mechanism is confirmed and slowing dense is
   a working lever. It is demoted to a stage 7 tuning knob (the optimum N, rate
   versus frequency, and a flip-rate-keyed dynamic ratio are quality questions the
   proxy cannot rank), because it is a blunt instrument: it slows every dense
   parameter to keep signal alive for every expert, including the long tail the
   task never routes to, whose masters random-walk under Adam at about one
   learning rate per step and would flip cells on noise at production step counts
   (about 3e-3 expected wander over a thousand steps at 1e-4, the median FP4
   half-cell). Experts-first sequencing (stage 6) is measured and not pursued.
5. Decided direction: ESFT-style expert selection (Wang et al. 2024, code in
   `deepseek/ESFT/`, DeepSeek-V2-Lite under HF Transformers; the method, not the
   code, transfers). Score every routed expert by the base model's routing on the
   training rows (`token_scores`: share of routed top-k slots; `gate_scores`:
   summed gate weight), select per layer in descending score until cumulative
   `top_p`, train only the selected experts, freeze the rest byte for byte. Same
   ownership as ours: frozen router and correction bias, selection stable because
   routing is frozen, publication contract unchanged. Two wins: selected experts
   are the ones with enough task tokens to build consistent gradient sign and
   cross FP4 cells (the surgical form of what the frequency split does bluntly),
   and 8-bit optimizer state shrinks with the trainable set (full model: about
   71 GB per rank at `ep_size` 8 for all experts, about 7 GB at 10%), which may
   return the 8x RTX PRO 6000 host from fallback to primary. Caveat: ESFT is a
   BF16 result; its rates and quality numbers say nothing about flips. Steps 1 to
   4 are complete:
   1. Routing concentration (`deepseek/qat/route_scores.py`, report
      `route-scores-engineering.json`): a forward over the 142 training rows
      (421,631 tokens) with a hook on each `mlp.gate` (`forward(x, token_mask,
      cp_mesh)` returns `(weights, indices, aux_loss)`; 256 experts, top-6),
      accumulating ESFT's two scores. Experts covering 20/50/80% of routed slots:
      hash layers 0 to 2 about 25/85/175 (uniform would be 51/128/205), scored
      layer 3 8/42/115 with the top expert at 4.2% of slots, 10.7x uniform. No
      expert has zero slots. `top_p` 0.2 selects 25, 25, 25, 8 = 83 of 1,024.
      Selection concentrates signal on scored layers, 40 of the full model's 43;
      the proxy over-represents hash layers, so layer 3 is the transferable number.
   2. Selection in the proxy stack: `proxy_step --expert-selection SCORES.json`
      (needs `--in-backward`). `LocalShardOptimizer.restrict_rows` rebuilds an
      expert parameter's chunks as one view per selected local row before the
      first step (no step, no 8-bit state for the rest; an empty selection
      freezes the shard); the stepper maps global ids to local rows through the
      expert-parallel offset, calls it at construction, and zeroes unselected
      rows of each expert gradient before the norm; the flip totals report the
      selected experts per layer summed over ranks; any code or scale change in
      an unselected expert fails the run.
   3. Arm (`proxy-esft-top02.json`): selected experts plus dense, joint at 1e-4,
      100 steps, global clip. Held-out 16.18 to 4.12 (joint 4.17, dense-only
      4.135: no loss from selection, the margin is within one seed). Selected
      experts' values changed: layer 3 2.22%, hash layers 1.49 to 1.77%, against
      1.45% over all experts in the joint arm and 2.79% in the split arm; dense
      54.3%. Zero unselected experts changed; frozen set clean. Peak 48.3 GB
      (joint 70.0, dense-only 45.7): expert optimizer state fell from 26 GB per
      rank to about 2 GB. The run's training phase was 3,182 s against 2,318 s
      because the per-layer flip accounting dequantized every FP4 entry twice at
      each validation; fixed (one dequantization per entry): the per-expert run
      below trained in 2,346 s.
   4. Per-expert flips against routing share (`proxy-joint-perexpert.json`: the
      joint arm rerun with `--expert-shares`, reproducing 4.172 and 1.45%; the
      report's `experts` table holds values changed, elements, and share per
      global expert). Scored layer 3, experts ranked by share: ranks 1 to 8
      (`top_p` 0.2) flipped 2.02%, 9 to 42 (to 0.5) 2.13%, 43 to 115 (to 0.8)
      1.57%, the remaining 141 0.91% with 51 of them under 0.5%. The curve is a
      plateau over about the top 40 experts, a decline, then a floor near what an
      Adam random walk produces in 100 steps at 1e-4, so on this data `top_p` 0.2
      was too narrow and 0.5 captured the plateau. Hash layers 0 to 2: flat, 1.4
      to 1.7% at every share (correlation with share under 0.1); hash routing
      gives each expert a fixed vocabulary slice with enough tokens for consistent
      moments, so share-based selection is meaningless there and the choice is
      all 256 or none (7% of the full model's experts). Share is a coarse
      per-expert predictor even on the scored layer (correlation 0.39; the top
      decile holds an expert at 0.08% beside one at 3.5%; one of the eight
      selected flipped 0.51%): what moves an expert is gradient-sign consistency,
      which share only partly captures. Cross-check: the eight selected flipped
      2.02% here and 2.22% in the ESFT arm.
      Conditionality, which bounds all of the above: `data/engineering.jsonl` is
      off-policy for this model (written for another model), and the 4-layer
      proxy's baseline loss of 16.2 is above the uniform 11.8, so much of every
      run fits truncation repair. Both inflate gradient-sign consistency and so
      flip rates everywhere. The structure (plateau, decline, floor on scored
      layers; flat hash layers; share as a coarse predictor) follows from routing
      and Adam and transfers; the positions (the 2% level, the `top_p` 0.5
      boundary, where the noise floor begins) are properties of this dataset on
      this proxy and do not. On near-policy data the plateau shrinks to the few
      experts with the most consistent signal and the floor's share of the layer
      grows. The proxy cannot test this even with model-generated data, since
      truncation repair dominates its gradients. Selection for the full model is
      therefore to be measured there, from per-expert gradient-sign consistency
      (the mean sign of each expert row's gradient over the first steps, available
      in the update-in-backward hook) and per-expert flip fraction, rather than
      inherited as a `top_p` from here; `top_p` by routing share remains the
      starting selection before any such measurement exists.
   Open tuning questions for stage 7: the score function (token or gate), and
   whether to combine selection with the dense-slowing knob.
6. Stage 7 preparation list, each a separate decision: steady-state memory plan
   redone with the selected trainable set; the dense-slowing knob as a recipe
   setting (per-class rate via `optim.py` param groups, or `--dense-every`) with a
   warmup for any class entering training (stage 6: a fresh Adam state at 1e-4
   lifted held-out 5.24 to 5.67 in ten steps); the container from the
   gate-passing lockfile; the dataset, where distance from policy is a condition
   for the method and not only a quality choice: QAT publishes only drift that
   crossed a half-cell, directed drift needs the data to disagree with the model
   consistently, and a dataset built close to the model's own outputs may
   fine-tune a BF16 model gently and publish nothing here but noise flips
   (Handoff 5.4); the per-layer and per-expert flip fractions in the first
   full-model run are how this is checked; an evaluation on prompts outside the
   SFT distribution, since held-out loss on same-distribution rows cannot
   separate dense fitting the data from experts generalizing from it (stage 5).

Alternatives if QAT of the experts does not deliver, ranked by the operator, with
the physics that ranks them. The released expert weights sit at the centers of
their FP4 cells, so any rounding of base plus a small delta back onto that grid
returns the base exactly; a fine-tune survives publication only through flips the
gradient chose (QAT) or through dithering from a misaligned grid.

1. Dense-only QAT: attention and shared experts (6.3B FP8) trained, routed experts
   byte-identical. Proven effective on the proxy; a public attention-only fine-tune
   of this model ships in the native format. Borderline on the 2x RTX PRO 6000
   (about 92 to 97 GB per GPU with MXFP4-resident experts, whose kernel path is
   unverified here); fits the 8x host.
2. FP8 experts as the publication format: a four times finer grid, so experts
   register at ordinary rates; all-FP8 checkpoint about 267 GB, native vLLM FP8
   path, no Blackwell FP4 kernels, does not fit two 96 GB GPUs for serving.
3. Stochastic rounding at publication: in-format dither, expected published weight
   equals the master, at the cost of one quantization's worth of whole-cell noise on
   trained tensors. Measurable on the proxy by rounding a trained arm both ways.
4. INT4 or any misaligned grid: carries the delta in expectation by dithering but
   re-quantizes every weight and leaves the native serving path.

Excluded with reason: BF16 training then FP4 requantization, and FP8 training then
FP4, are post-hoc rounding onto the aligned grid and erase the fine-tune; LoRA at
tens of steps yields deltas about 1e-4, below both grids' half-cells.

## Decisions

- Full-parameter QAT is the only path pursued until exhausted. The LoRA-QAT gate arm
  was dropped before measurement (stage 5) and is not developed further unless
  full-parameter fails.
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
  Frozen tensors are copied byte for byte at export. NeMo's hardcoded per-step
  router-bias update is disabled by `deepseek.qat.freeze.freeze_gate_bias` after
  recipe setup; the frozen-integrity check digests buffers as well as parameters.
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
- Update-in-backward with lagged clipping: `deepseek/qat/inbackward.py` steps every
  trainable parameter from its post-accumulate-grad hook (autograd for expert
  parameters, FSDP2 after reduce-scatter for dense ones) and frees the gradient. The
  exact global norm is accumulated as gradients arrive and reduced once at step end;
  the previous step's `min(1, max_norm / (norm + 1e-6))` scales the current step, and
  the first step is unclipped (Adam's first update is `lr` times the gradient sign).
  NeMo's `scale_grads_and_clip_grad_norm` is called from the hook with clipping off so
  expert-gradient scaling stays NeMo's; the hook verifies exactly one gradient is
  alive. Requires one microbatch per step and `tp = pp = cp = 1` with data- and
  expert-parallel groups spanning the world; NeMo's own clip is set to `None` after
  setup because the YAML cannot disable it.
- FP4 weight quantizer: OCP MX floor rule, `scale = 2**(floor(log2 absmax) - 2)`,
  saturating at 6. FP8: `scale = 2**ceil(log2(absmax / 448))`, never saturating.
  Both established against the released tensors (Verified facts).
- FP4 training scale rule `min_error` (`quantize_fp4_rows(..., scale_rule=
  "min_error")`, used everywhere through `fakequant.quantize_fp4`): per group, the
  floor-rule scale or twice it, whichever has the smaller float32 sum of squared
  errors, ties to the floor rule. Chosen over a per-group floor pinned at the source
  scale because it is stateless (nothing to persist for resume), lets a group whose
  largest weight genuinely shrinks take the finer scale, and fixes the rule's actual
  defect, ignoring saturation error. Released groups have zero error at their own
  scale and are a fixed point: round trip of 200 sampled expert tensors is bit-exact
  in codes and scales. Cost: expert fake-quant runs twice, about 1.2 s per step per
  layer on the proxy (20 steps on one layer: 59 s to 83 s). FP8 is unchanged; its
  ceiling rule's edge is upward, where a doubling keeps the value exact.
- Dataset for stages 3 to 5: `data/engineering.jsonl`, 158 rows of one user turn,
  one assistant turn, and a separate `reasoning` string. Chosen for being in the
  exact format and token length the full run will use, not for quality; it targets a
  different model and is still being cleaned. The last 16 rows are held out. The
  gate measures mechanism (flips, loss response, held-out loss), not behavior.
- Everything for this work lives under `deepseek/`. Git is managed by the operator.

## Layout

```
deepseek/
  PLAN-deepseek.md, SPEC-deepseek.md
  automodel/                                   NeMo AutoModel 0.8.0 source
  ESFT/                                        DeepSeek ESFT reference code (method only)
  models--deepseek-ai--DeepSeek-V4-Flash-0731/ HF cache layout; snapshot
                                               7872f01b1d1fe23eabc4c98b48bffcef5a386062
  venv/                                        NeMo environment (uv, Python 3.12.13)
  exports/                                     exported checkpoints (156 GB each), logs/
  qat/                                         project code; imports nothing from trlx
    quant.py          exact FP8/FP4 quantizers and dequantizers; FP4 min_error scale rule
    roundtrip.py      read-only checkpoint round-trip check
    export.py         layout-preserving exporter
    verify_export.py  shard/tensor/file diff of two checkpoints; --expect @FILE
    fakequant.py      straight-through fake-quant hooks on the NeMo model
    freeze.py         disables NeMo's router correction-bias update
    data.py           reasoning-conversation dataset bridge (NeMo shifted format)
    optim.py          torchao optimizer over local shards; FP32-master AdamW; step_param;
                      restrict_rows (per-row views for expert selection)
    inbackward.py     update-in-backward stepper with lagged clipping, optional per-class norms,
                      per-class update period (frequency split), per-parameter row selection
    overrides.py      --set/--unset recipe overrides shared by the torchrun entries
    proxy_export.py   trained-target export: quantize, gather experts to rank 0,
                      replace only tensors whose bytes differ from the source
    proxy.py          parity check and export reload (torchrun entry); --set/--unset,
                      --expect-logits PREFIX; shared probe token ids
    proxy_step.py     stage 3 to 6 training check and diagnostics (torchrun entry);
                      --in-backward, --per-class-clip, --set/--unset recipe overrides,
                      --trainable-subset, --dense-every N, --expert-selection SCORES.json,
                      --expert-shares SCORES.json, held-out capture with per-class and
                      per-layer flip totals, per-expert flip table in the report,
                      --export DIR [--force] with probe logits and replaced list beside --out
    route_scores.py   routing-concentration measurement and ESFT top_p selection (torchrun entry)
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

Results on both ranks: 37 target modules per rank (41 parameter tensors, 40
trainable plus the frozen indexer); fake-quantized tensors equal the
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
  Real FP4 value changes are about 0.6%, nearly all from knife-edge groups whose
  scale halved (see Verified facts; this was first recorded as "doubled" and
  corrected in stage 4). The flip metric must compare values.
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

### Stage 4: memory plan and master-precision comparison (complete)

The optimizer and its wiring are done (stage 3).

- Router-bias freeze. NeMo's DeepSeek-V4 build hardcodes a 1e-3 per-step update of
  each scored gate's `e_score_correction_bias` buffer (Verified facts), which stage 3
  could not see because its frozen check covered parameters only. `freeze.py` zeroes
  the model-level factor after setup; `proxy_step.py` calls it and digests every
  buffer alongside the frozen parameters. One-step check with the freeze
  (`deepseek/exports/logs/proxy-stage4-freeze.json`): factor 1e-3 before, one scored
  gate on the proxy (layers 0 to 2 hash-route), 68 frozen parameters and 6 buffers
  unchanged by a bit, all other figures equal to stage 3.
- Update-in-backward (Decisions). Motivation: with `ep_size: 8` each B200 rank owns
  32 experts per layer outside FSDP, and a full BF16 gradient for them is about
  69 GB. `proxy_step.py --in-backward` enables it. Equivalence check with stochastic
  rounding off in both runs and the reference clip at 1e9 (so both first steps are
  unclipped): all 41 target masters bit-identical on rank 0 and the rank-1 aggregate
  digest identical; identical loss and flips; norms 25.700846943 (NeMo) and
  25.700847638 (stepper). Reports `deepseek/exports/logs/proxy-stage4-eq-reference.json`
  and `proxy-stage4-eq-inbackward.json`. Peak memory unchanged (Handoff 2).

- FP32-master comparison. `deepseek.qat.optim.adamw_fp32_master` holds FP32 masters
  per shard chunk beside the BF16 parameters and writes `bf16(master)` back after
  each torch AdamW step, so the forward, hook, and export see `Q(bf16(master))`.
  One layer (four layers of FP32 state is about 156 GB per rank), 20 steps at
  global batch 4, held-out loss at baseline and after updates 5, 10, 15, 20, both
  arms `--in-backward`:

  ```sh
  PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
      -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --in-backward \
      --set model.config.num_hidden_layers=1 --set step_scheduler.max_steps=20 \
      --set step_scheduler.val_every_steps=5 --out REPORT.json
  # FP32 arm adds:
      --set optimizer._target_=deepseek.qat.optim.adamw_fp32_master --unset optimizer.bf16_stochastic_round
  ```

  Results, floor rule (`deepseek/exports/logs/proxy-stage4-master-bf16sr.json`,
  `-fp32.json`): step 0 identical. BF16-SR: training loss 17.775 to 16.335,
  held-out 17.295 to 15.347, peak 41.2 GB. FP32: 17.775 to 17.040, held-out 17.295
  to 16.162, peak 71.7 GB. Dense FP8 value changes alike (2.17M vs 2.10M of 33.5M
  in `wq_b`). Expert gate/up group scales changed: 2.92M of 67.1M (4.35%) under
  BF16-SR, 0.25M (0.37%) under FP32, 1.06 values per changed scale in both arms.
  Results, `min_error` rule, BF16-SR (`proxy-stage4-minerror-bf16sr.json`): expert
  scale changes 0, expert value changes 0 (gate/up) and 6 (down); dense changes as
  before; training loss 17.775 to 17.121, held-out 17.295 to 16.248; peak 41.2 GB.
  Reading: under the floor rule every expert value change was a knife-edge clobber
  (largest weight 4s to 3s), and the BF16-SR arm's lower loss came from those
  clobbers, not learning: the two BF16-SR arms differ only in clobbers and differ
  by 0.8 to 0.9 nats, while the clobber-free arm tracks the FP32 arm. Shrinking the
  largest expert weights lowers the loss of a one-layer truncated model whose loss
  is above uniform. Master precision itself made little difference in 20 steps.
  At lr 1e-5, 20 steps move a master by at most 2e-4, below every FP4 half-cell
  (median 3.9e-3, none below 2e-5), so routed experts cannot change value; dense
  FP8 does train (6.5% of `wq_b` values). The lagged clip coefficient stayed in
  0.035 to 0.051 with no spikes. A one-layer loss is a mechanism signal, not quality.

### Stage 5: gate run (complete)

The gate answers whether routed experts train under QAT, not whether a prior recipe
transfers. The rank-8 LoRA arm is removed: a LoRA delta after tens of steps at 1e-4
is about 1e-4 per weight, below both grids' half-cells, which is the prior H100
finding restated.

Proxy memory: the 4-layer proxy at local batch 2 runs out of memory from step 2 on,
because the 8-bit moments (26 GB per rank) are resident before backward from the
second step while a first step allocates them progressively. Gate runs use global
batch 2, local batch 1 (steady-state peak 70.0 GB). First-step peaks understate
steady state; the stage 7 budget must use multi-step measurements.

Phase A (complete): learning-rate sweep, full trainable set, 100 steps, held-out
every 10 steps with per-class value and scale changes at each point
(`deepseek/exports/logs/proxy-stage5-sweep-lr{1e-5,3e-5,1e-4,3e-4}.json`):

| lr | held-out end | dense changed | expert changed | expert scales | norm max |
|---|---|---|---|---|---|
| 1e-5 | 6.21 | 19.4% | 0.0000% | 0 | 28.0 |
| 3e-5 | 4.56 | 31.1% | 0.0010% | 1 | 28.0 |
| 1e-4 | 4.17 | 54.1% | 1.45% | 1,041 | 28.0 |
| 3e-4 | 6.04 | 82.7% | 12.3% | 732,173 | 88.4 |

Baseline held-out 16.18. Expert flips begin where drift reaches the small-cell end
of the half-cell distribution (step 70 at 3e-5, step 30 at 1e-4) and then grow
smoothly; at 1e-4 nothing destabilizes and the scale changes are upward doublings.
At 3e-4 the model is rewritten: held-out stalls at 7.1 from step 30 to 70 and ends
worse than 3e-5. The usable regime is around 1e-4. Dense value changes saturate
near 54% at that rate. The learning rate sets flip frequency; the quantizer sets
flip size. A per-class rate (experts about 4x dense, the cell-size ratio) and a
schedule are the natural next levers if phase B supports the premise.

The first 1e-5 arm (`-normbug.json`) ran with a stream race in the stepper's norm
accumulation (NaN norms, clipping effectively off); fixed before the sweep proper.

Phase B (complete): at 1e-4, 100 steps, experts-only (dense frozen) and dense-only
(experts frozen) arms via `--trainable-subset` and a `--set` of the unfreeze list
(`proxy-stage5-ablation-{experts-only,dense-only}.json`), against the 1e-4 sweep arm:

| arm | held-out end | dense changed | expert changed | step time | peak |
|---|---|---|---|---|---|
| full set | 4.172 | 54.1% | 1.45% | 23 s | 70.0 GB |
| dense-only | 4.135 | 54.3% | 0 | 13 s | 45.7 GB |
| experts-only | 5.242 | 0 | 4.84% | 23 s | 69.5 GB |

Experts train under QAT: alone, with dense verified unchanged, held-out fell from
16.18 to 5.24, flat for 20 steps while masters crossed cells (0.01% flipped), then
five nats between steps 20 and 30 as flips passed 1%, then steadily with the flip
fraction. The flips were chosen by the straight-through gradient in whole FP4 cells.
Experts did not add measurable value on top of dense on this proxy: dense-only
matches the full set at every checkpoint. Confound: one global clip scaled by the
dense gradient norm (coefficient about 0.04) held the joint arm's experts to 1.45%
flips against 4.84% alone (coefficient near 1). The proxy's 264M dense parameters can
absorb the 142-row dataset by themselves, so the marginal value of experts in the
full model, where dense is 6.3B of 290B, is not a question it can settle. The
flip fraction, not the loss, is the leading indicator in this regime. The clipping
confound named above was tested in phase C and was not the cause.

Phase C (complete): full set at 1e-4 with per-class clipping (`--per-class-clip`:
dense and expert norms reduced separately, each class clipped by its own coefficient;
`proxy-stage5-perclass-lr1e-4.json`). Held-out 4.145; experts flipped 1.79% with
their coefficient at 1.0 for most of the run, against 4.84% alone. Clipping was not
the limiter: Adam is invariant to a constant gradient scaling except through `eps`
and the early moment estimates, which also explains why lagged and exact clipping
agreed in stage 4. What limits expert flips in a joint run is gradient-sign
consistency, and dense learning removes it: the expert gradient norm fell from 4.75
to 0.81 as dense took held-out from 16 to 5 in 30 steps, and masters that are no
longer pushed consistently stop reaching cell boundaries. The ceiling is the data:
dense alone 4.135, experts alone 5.24, every joint arm 4.14 to 4.17 on 142 rows.
Per-class clipping is a documented no-effect lever and stays in the code as such;
per-class learning rate still sets drift speed under Adam but cannot raise a
data-set ceiling on this proxy.

Verdict: go on the mechanism. Routed FP4 experts train under QAT: with dense frozen
and verified unchanged, held-out fell from 16.18 to 5.24 through gradient-chosen
whole-cell flips with no instability. The marginal value of experts over dense is
not measurable on this proxy and not refuted; it needs a model whose dense fraction
cannot absorb the dataset alone (the full model, dense 6.3B of 290B) and a dataset
with more in it than 142 rows. Stage 6 needs an export path at the end of a
`proxy_step` run, since gate runs save no weights.

### Stage 6: proxy export and reload (complete)

Export the trained proxy with stage 1, verify that only trained targets differ,
reload in a fresh process, and reproduce the final training forward bit for bit.

- `deepseek/qat/proxy_export.py`: every rank quantizes its trained targets with the
  project quantizers (`fakequant.quantize_fp4`, `quant.quantize_fp8_blocks`); expert
  codes and scales are gathered to rank 0 over the expert mesh (one-dimensional,
  spanning the world, global expert `e = rank * E_local + i`); dense tensors are
  materialized in full. Names: dense through `proxy.hf_names`, experts to
  `layers.{L}.ffn.experts.{e}.{w1,w3,w2}.{weight,scale}`. Rank 0 compares every
  candidate's bytes with the source and replaces only those that differ, so the
  replaced list is the exact `verify_export --expect` set and the report's
  changed-against-source count is a measurement. The source is the recipe's
  `pretrained_model_name_or_path`, so a train-on-export run exports from that
  export and the chain stays a complete checkpoint at every link.
- `proxy_step.py --export DIR [--force]` (requires `--out`): after the report checks,
  records hooked logits in eval mode on the shared probe ids (`proxy.probe_token_ids`,
  batch 2, seq 256, seed 0) to `<out stem>-logits-rank{r}.pt`, exports, and writes
  the replaced names to `<out stem>-replaced.txt`. An `ExportError` is recorded in
  the report and fails the run without raising, so other ranks reach the final
  all-reduce.
- `proxy.py --snapshot EXPORT --set model.config.pretrained_model_name_or_path=EXPORT
  --set model.config.name_or_path=EXPORT --expect-logits PREFIX`: the existing
  fixed-point and source-parity checks run against the export; the hooked logits must
  equal the saved ones bit for bit. `deepseek/qat/overrides.py` holds `--set/--unset`
  for both entries.
- `deepseek/tests/test_proxy_export.py`: 9 tests (naming, half order, local expert
  quantization against HF matrices, rank-order assembly and dtypes, dense mapping,
  byte-equal skip including uint8 against I8, `--expect @FILE`).

Results (`deepseek/exports/logs/proxy-stage6-*`):

- Experts-only at 1e-4, 100 steps, `--export deepseek/exports/stage6-experts`:
  held-out 16.18 to 5.242 (phase B: 5.24); 8 trained targets, 6,144 candidate
  tensors, 5,888 replaced, 256 unchanged, every one a scale (`min_error` keeps the
  source scale until a flip crosses a bin). Export 181 s. `verify_export` against
  `$SNAP` with `--expect @proxy-stage6-experts-replaced.txt`: exactly those 5,888
  differ. Reload: fixed point and source parity against the export clean on both
  ranks; probe logits equal the saved ones bit for bit, zero differing elements.
- Sequential experiment: dense-only at 1e-4 for 100 steps on that export,
  `--export deepseek/exports/stage6-sequential`. Step-zero held-out 5.242 (equals the
  experts-only end, a third confirmation that export and reload are exact). 32
  trained targets, 64 of 64 candidates replaced. Differ against `stage6-experts`:
  exactly those 64. Reload: bit-identical logits again. Held-out: 5.24, 5.67 at
  step 9, then 5.40, 5.00, 4.78, 4.69, 4.73, 4.78, 4.81, 4.80, 4.75, 4.62.
- Reading. Dense changed 53.9% of its values on the experts export against 54.3%
  from base and 54.1% joint: the dense flip fraction does not depend on how much
  loss remains, because Adam normalizes gradient scale and the learning rate sets
  drift per step. This refutes the premise that dense would stop once experts had
  fit (former Handoff 4). The 4.62 endpoint against dense-only's 4.14 is a fact
  with a named confound, not a ranking: a fresh Adam state at 1e-4 moves every
  dense master by about one learning rate per step at first and lifted held-out
  from 5.24 to 5.67 in ten steps, so any class entering training needs a warmup.
  Whether experts plus dense beats dense alone remains unanswerable on this proxy
  (stage 5).

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_proxy_export
# train with export (experts-only arm shown; see Handoff for the override lists)
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 \
    -m deepseek.qat.proxy_step --config deepseek/qat/proxy-train.yaml --in-backward \
    --trainable-subset experts --set 'freeze_config.unfreeze_modules=[{glob: "model.layers.*.mlp.experts"}]' \
    --set optimizer.lr=1e-4 --set step_scheduler.max_steps=100 --set step_scheduler.val_every_steps=10 \
    --set step_scheduler.global_batch_size=2 --set step_scheduler.local_batch_size=1 \
    --out deepseek/exports/logs/REPORT.json --export deepseek/exports/EXPORT
deepseek/venv/bin/python -B -m deepseek.qat.verify_export $SNAP deepseek/exports/EXPORT \
    --expect @deepseek/exports/logs/REPORT-replaced.txt
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy --config deepseek/qat/proxy-4layer.yaml \
    --snapshot deepseek/exports/EXPORT --set model.config.pretrained_model_name_or_path=deepseek/exports/EXPORT \
    --set model.config.name_or_path=deepseek/exports/EXPORT --expect-logits deepseek/exports/logs/REPORT
# train on an export: add the two model-path overrides to the proxy_step command
```

### Stage 7: full model on 8x B200 or B300 (planned)

Docker image from the gate-passing venv lockfile with a verifier script run first.
Routing scores over the training data on the training host, then selection, then
one update, bounded convergence run, export, fresh-process equality, Transformers
and vLLM probes on the RTX PRO 6000 pair. Memory plan with `ep_size: 8` and
ESFT-style selection (Handoff 5): each rank owns 32 experts per layer outside FSDP
as dequantized BF16, about 72 GB of weights per rank for the full model, which now
dominates; 8-bit moments are under 2 GB for dense and a few GB for the selected
experts (about 71 GB per rank if every expert were trained, the pre-selection
figure of 141 GB per GPU with all experts no longer applies). Weights fall further
only if unselected experts could stay in MXFP4, which NeMo's one-tensor-per-layer
expert layout does not allow. These are resident-state figures; the steady-state
peak adds activations, `min_error` fake-quant temporaries during recompute, and
torchao step temporaries, and must be measured on a multi-step run (Handoff 2).
With selection the 8x RTX PRO 6000 (96 GB) host is borderline on activations rather
than ruled out; the B200 is comfortable, and a B300 host is available. The
preparation decisions are listed in Handoff 6. Expert flips arrive tens of steps
after dense changes at the same rate (stage 5), and a joint run at one rate starves
them once dense fits (phase C). Selection is the primary remedy: the task-routed
experts flip at full dense speed (Handoff 5.3). Dense slowing is a tuning knob
(Handoff 4). Production runs need more steps than a dense recipe, and the per-layer
and per-expert flip fractions are the leading indicators to watch, not the loss.

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
  4 times a power of two, the bottom of its scale bin. Any downward move of that
  element, however small, halves the group's scale; the element then needs code 8,
  saturates at 6, and drops from 4s to 3s, while the other 31 values stay exact.
  An upward move changes nothing until the element reaches 8s, where the scale
  doubles and about 13 of 32 values move. Checked directly on a synthetic group
  with `quantize_fp4_rows` in stage 4; the stage 4 comparison measured 1.06 values
  changed per changed scale. 53% of released expert groups are in that state
  (largest code 4); 47% have largest code 6.
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
  optimizer factory whose signature accepts arbitrary keywords. The DeepSeek-V4
  model build hardcodes `train_gate=True` and `gate_bias_update_factor=1e-3` in its
  `MoEConfig` with no recipe key, and the training recipe calls
  `update_moe_gate_bias` after every optimizer step, moving each scored gate's
  `e_score_correction_bias` buffer (loaded from `ffn.gate.bias`) by the factor per
  expert; hash-routing gates (`num_hash_layers: 3`) have no correction bias.
  Expert gradients are divided by the data-parallel over expert-shard ratio before
  clipping, and the optimizer and scheduler step after clipping.
- torchao 0.18 `_AdamBase` allocates `exp_avg` and `exp_avg_sq` lazily inside
  `step`, on the first step a parameter arrives with a gradient; construction and
  `add_param_group` allocate nothing. `LocalShardOptimizer.restrict_rows` relies on
  this to swap a parameter's chunk views before the first step and refuses once
  any state exists. Re-verify on a torchao upgrade.
- Safetensors shards: headers are compact JSON padded with spaces to 8 bytes,
  tensor order equals offset order, no gaps; `json.dumps` with `(",", ":")`
  reproduces them.
- Adam is invariant to a constant scaling of its gradients except through `eps` and
  the early moment estimates. Consequences measured in stage 5: a clip coefficient
  of 0.04 versus 1.0 barely changes drift; lagged and exact clipping agree; a
  per-class clip does not change expert flip rates. The learning rate sets drift per
  step (flip frequency); the quantizer sets flip size.
- In QAT from grid-aligned weights, a parameter changes only when its master has
  drifted a half-cell, so loss is flat while masters accumulate and then moves in
  bursts; the flip fraction leads the loss. Routing sparsity (each expert sees about
  2% of tokens) lowers per-expert sample counts by about fifty times and so delays
  consistent drift; it does not change what experts can learn.
- First-step peak memory understates steady state: optimizer state allocated
  progressively during a first backward is fully resident before every later one.

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
