# DeepSeek-V4-Flash-0731 full-parameter QAT plan

Goal: post-train `deepseek-ai/DeepSeek-V4-Flash-0731` and publish a standalone
checkpoint identical to the source in every file, tensor name, shape, dtype, shard
layout, and scale grid, except the payloads of trained tensors and their scales. No
adapter, no PEFT metadata, no optimizer state. It must load and generate on
hardware that serves the source.

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
split arm's rate with dense at full speed, unselected experts stay byte-identical.
The steady-state memory plan (Handoff 2, Handoff 6
item 1) is measured: per-step peaks, the update-in-backward saving, and a labelled
full-model estimate (stage 7). The dense-slowing knob and warmup are recipe
settings (Handoff 6 item 2). Stage 6b, split expert storage, which halves
the per-GPU requirement and makes 4x B200 or 8x RTX PRO 6000 viable hosts, is
complete: a split-module run trains the selected experts at the ESFT arm's
rates, exports a verified checkpoint that reloads bit-identically, and peaks at
44.4 GiB against 53.4. Next is Handoff 6 from item 3. The dataset, optimizer,
trainable set, and mechanism are decided and verified below. Every command runs
from the repository root
`/home/goon/trl/trlx`. Launch GPU runs from a plain terminal with `nohup`: the
post-training report phase drives host memory low enough that an agent
session's background-process reaper has killed a run.
Rank 0 prints the full report to stdout, so a log redirect is a second copy of `--out`.

Run all unit tests (CPU, under a minute; the data tests need the snapshot):

```sh
deepseek/venv/bin/python -B -m unittest deepseek.tests.test_quant deepseek.tests.test_export \
    deepseek.tests.test_fakequant deepseek.tests.test_data deepseek.tests.test_freeze \
    deepseek.tests.test_inbackward deepseek.tests.test_optim deepseek.tests.test_proxy_step \
    deepseek.tests.test_proxy_export deepseek.tests.test_route_scores deepseek.tests.test_split_experts
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
`proxy-joint-perexpert.json` (Handoff 5), and `proxy-mem-2layer-inbackward.json`,
`proxy-mem-2layer-reference.json`, `proxy-mem-2layer-inbackward-rerun.json`, and
`proxy-mem-esft-top02.json` (Handoff 2), and `proxy-mem-2layer-inbackward-groups.json`
(Handoff 6 item 2), and `route-scores-perrank10.json` and `proxy-stage6b-*` (stage 6b).
Exports: `deepseek/exports/identity/` and `one-tensor/` (stage 1),
`stage6-experts/` and `stage6-sequential/` (stage 6), and `stage6b-split/`
(stage 6b), 156 GB each; all can be deleted and regenerated.

Handoff items, in order:

1. `deepseek/qat/proxy_step.py` keeps host copies of sampled masters; the full set
   (26 GB per rank) exceeded this host's 122 GB alongside NeMo's load staging. Keep
   the sampling.
2. Memory (complete; GiB per rank, local batch 1, rows to 4,352 tokens). `proxy_step`
   reports per-step allocated before/after, step peak, reserved, and the
   gradient-resident moment (optimizer entry, or the hook-entry peak under
   `--in-backward`) in `steps`, with resident state by class under `memory`.
   Saving pair, 2 layers, full set, 20 steps at 1e-4 (`proxy-mem-2layer-reference.json`,
   `-inbackward.json`): resident after step 26.196 in both arms; optimizer entry
   38.442, which is resident plus exactly the BF16 gradient of the 6,574,571,520
   trainable elements (12.246); step-peak maxima 56.821 against 46.538, so
   update-in-backward saves 10.283 of peak, the gradient set minus the one gradient
   live in the hook (hook-entry max 41.956), at no time cost (225 s both). Step-1
   peaks (41.5, 36.7) understate steady state by 9 to 14; peaks rise with row length,
   so a dataset-wide maximum needs a run over the whole file. Reserved exceeds peak
   allocated by 8 to 10 under `expandable_segments`. Budget, 4 layers, ESFT top 0.2,
   100 steps (`proxy-mem-esft-top02.json`): resident 29.169 after the step; the class
   accounting sums to 29.153 (experts 24.000, 8-bit state 3.119, buffers 0.017, and
   half of the dense 0.984 and frozen 3.050 full-tensor figures, since the report's
   shard figures are the full tensors), the same 16 MiB residual as the 2-layer
   rerun; step-peak max
   53.445 (43 to 53 by batch), hook-entry max 47.579, transient max 24.276
   (activations, unsharded dense 2.0, one expert gradient up to 4.0, fake-quant
   temporaries during recompute, torchao step temporaries), reserved 60.168.
   Resolved: a rerun of the 2-layer in-backward arm
   (`proxy-mem-2layer-inbackward-rerun.json`) reproduced every per-step memory
   figure to the megabyte, every loss, and the master digest, so the current code
   is deterministic and a measured peak is a requirement. The Handoff 5.3 run of
   this configuration reported 48.28 on an uncommitted working tree between commits
   9b89c39 and 225d80f whose state is not recoverable; 53.45 is the figure for the
   code going forward. The rerun also confirmed the resident accounting:
   `resident_total_gb` 26.180 against 26.196 allocated after the step.
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
   71 GB per rank at `ep_size` 8 for all experts, about 7 GB at 10%), which
   lowers the per-GPU memory requirement. Caveat: ESFT is a
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
   redone with the selected trainable set (complete, item 2); the dense-slowing
   knob as a recipe setting with a warmup for any class entering training
   (stage 6: a fresh Adam state at 1e-4 lifted held-out 5.24 to 5.67 in ten
   steps) (complete: `proxy-train.yaml` carries an `optimizer.param_group_overrides`
   entry putting `mlp\.experts` in its own group with `lr_mult`, and an
   `lr_scheduler` block with `lr_decay_style: constant` and `lr_warmup_steps`;
   NeMo's scheduler applies both, so dense at r/N with experts at r is base
   `lr = r/N` and `lr_mult: N`, and warmup is linear from `init_lr` (default
   0.1 x lr) over `lr_warmup_steps`, which must stay below `max_steps`. The base
   recipe sets `lr_mult: 1.0` and warmup 0, and the 2-layer in-backward arm under
   it reproduced the master digest 54a0a65f of the ungrouped run
   (`proxy-mem-2layer-inbackward-groups.json`). `--dense-every` remains as the
   frequency-split form; a class entering mid-run is outside this mechanism. The
   values of N and the warmup length are stage 7 tuning questions); the container from the
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
   of this model ships in the native format. About 92 to 97 GB per GPU on two
   GPUs with MXFP4-resident experts, whose kernel path is unverified here.
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
- Training host requirements for the full model: a rank count dividing 256, with
  the expert-parallel group spanning the world (`tp = pp = cp = 1`); per-GPU memory
  equal to the steady-state peak measured for the selected trainable set at that
  rank count (Handoff 2 measured the proxy; the full-model figure is the stage 7
  estimate until the first full-model run's `memory` block measures it; resident
  expert weights are 516 GiB of dequantized BF16 divided by the rank count); host
  RAM for NeMo's dequantized load staging and the master sampling (unquantified for
  the full model); no format-specific kernels. The stage 6b hardware estimate bounds
  the host search; the working target host is a separate decision below, and the
  first full-model run's `memory` block remains the requirement.
- Development, unit tests, and the four-layer proxy gate run on the owned host
  (Environment) in a venv. A container is built for the training host from the
  lockfile of the venv that passed the gate and is not tested locally.
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
  Routed experts train only when selected for the run by ESFT-style routing scores
  (Handoff 5); the selection may be all experts. The full-model selection is
  measured there from per-expert gradient-sign consistency and flip fraction,
  with `top_p` by routing share as the starting point. Unselected experts are
  frozen byte for byte and get no optimizer state.
- Master precision: BF16 masters with stochastic-rounding updates and torchao
  `AdamW8bit` (8-bit blockwise moments, block 256) for the full run and on the proxy.
  FP32 masters do not fit the proxy (27B parameters at 4 bytes plus moments on two
  96 GB GPUs). The FP32-master comparison in stage 4 uses a configuration that fits.
- Optimizer wiring: `deepseek/qat/optim.py` runs the torchao optimizer over each
  parameter's local shard, split along dim 0 into chunks of at most 2^28 elements.
  torchao 0.18 fails on DTensor parameters under torch 2.10 (compiled: fake-tensor
  conversion of the quantized state; eager: sharding propagation of `lerp`), and its
  step materializes several FP32 copies of each tensor it updates, which exceeded GPU
  memory on a 2-billion-element expert tensor. Shard views share storage with the
  DTensor, and chunked updates were verified bit-identical to unchunked ones.
  `LocalShardOptimizer` mirrors NeMo's parameter groups: its own `param_groups`
  hold the model's parameters with float rates, which NeMo's LR scheduler reads
  and writes, and it copies them into the inner groups before every inner step,
  because torchao keeps its rate as a tensor and raises on a float.
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
- Expert storage (stage 6b): each layer's routed experts are split into a trained
  BF16 stack (selected experts) and a frozen packed stack (unselected experts as
  the source's FP4 codes and E8M0 scales), dequantized on the fly through NeMo's
  `MXFP4GroupedMM`. Equivalent by construction: an unselected expert's
  `dequant(Q(w))` is `dequant(codes, scales)`, and selected experts keep the same
  masters, hook, and optimizer. Shards are even: every expert-parallel rank selects
  the same count per layer, by score within its own expert range (`route_scores
  --per-rank N`); hash layers stay all-or-none. The selected stacks keep the names
  `gate_and_up_projs` and `down_projs`, so the fake-quant, stepper, optimizer, and
  export contract is unchanged, only shorter along dim 0.
- The vendored NeMo copy may be patched and its module code adapted into
  `deepseek/qat`. Each patch is listed in the stage that introduces it.
- Working target host: 4x B200 (NVLink, about 40% margin on the stage 6b estimate,
  fits a 30% selection). 8x RTX PRO 6000 is the fallback (PCIe, about 25% margin
  at 8% selection). The first full-model run's `memory` block remains the
  requirement.
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
                      restrict_rows (per-row views for expert selection); NeMo parameter
                      groups with rate sync into torchao's tensor rates
    inbackward.py     update-in-backward stepper with lagged clipping, optional per-class norms,
                      per-class update period (frequency split), per-parameter row selection,
                      hook-entry peak memory sample
    overrides.py      --set/--unset recipe overrides shared by the torchrun entries
    proxy_export.py   trained-target export: quantize, gather experts to rank 0,
                      replace only tensors whose bytes differ from the source
    proxy.py          parity check and export reload (torchrun entry); --set/--unset,
                      --expect-logits PREFIX, --save-logits PREFIX; shared probe token ids;
                      packed-stack byte check under split storage
    proxy_step.py     stage 3 to 6 training check and diagnostics (torchrun entry);
                      --in-backward, --per-class-clip, --set/--unset recipe overrides,
                      --trainable-subset, --dense-every N, --expert-selection SCORES.json,
                      --expert-shares SCORES.json, held-out capture with per-class and
                      per-layer flip totals, per-expert flip table in the report,
                      per-step memory series and resident accounting by class, per-group
                      rate metrics,
                      --export DIR [--force] with probe logits and replaced list beside --out
    route_scores.py   routing-concentration measurement; ESFT top_p or --per-rank N selection (torchrun entry)
    split_experts.py  stage 6b: ExpertSplit (selection file, install hook) and SplitExperts, the per-layer
                      module with a trained BF16 stack and a frozen packed stack dequantized on the fly
    split_adapter.py  stage 6b: DeepSeekV4StateDictAdapter subclass routing each expert into the BF16
                      or packed stack per rank on load and export
    proxy-4layer.yaml stage 2 recipe (mock token ids)
    proxy-train.yaml  training recipe: data bridge, freeze config, optimizer with the
                      experts parameter group (lr_mult), constant-rate scheduler with warmup
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
- Update-in-backward (Decisions). Motivation: with `ep_size: 8` each rank owns
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

### Stage 6b: split expert storage (complete; gate 4 optional)

Routed experts of each layer live in two stacks: selected experts as trained BF16
parameters, unselected experts as the source's packed FP4 codes and E8M0 scales
(Decisions). Unselected experts are 92% of resident bytes at an 8% selection and
cost about 0.53 bytes per parameter packed against 2 as BF16: expert storage
falls from 516 GiB to about 167, the load no longer dequantizes them (NeMo's
dequantized load staging was the unquantified host-RAM item), and export and
integrity for unselected experts become byte copies and digests.

NeMo supplies the pieces: `GroupedExpertsMXFP4` and `MXFP4ExpertStorageMixin`
(`components/moe/quantized_experts.py`) keep `*_packed` and `*_scales` as frozen
parameters; `MXFP4GroupedMM` (`components/quantization/mxfp4.py`) dequantizes
inside forward and backward, returns the activation gradient, and produces no
weight gradient; the adapter's `expert_storage_format = "mxfp4"` aggregates
per-expert codes and scales without unpacking, byte-identical to the HF layout
(gate and up concatenated along the output dim); the LoRA variant shows trainable
and packed compute in one grouped forward. What NeMo lacks is two storage kinds
in one layer, and a hook between construction and sharding: with `ep_size > 1`
the build applies EP and FSDP first (`_should_load_before_shard` is false under
expert parallelism) and loads the checkpoint afterwards through DCP into the
sharded parameters, with the adapter's `to_hf` supplying per-expert placeholders
from each rank's local shard and `from_hf` rebuilding the stacks.

Components (built):

- `deepseek/qat/split_experts.py`: `ExpertSplit` reads a `route_scores --per-rank`
  report (`ranks`, `n_routed_experts`, `layers.<L>.selected`), requires an equal
  count per rank per layer inside each rank's contiguous expert range, and
  `install(model)` swaps every backbone layer's experts module and the state-dict
  adapter (the model must have no MTP and must enable the HF adapter).
  `SplitExperts(GroupedExperts)`: BF16 stacks keep the names `gate_and_up_projs`
  and `down_projs` (global dim 0 is ranks times the per-rank count); packed stacks
  are `*_packed` and `*_scales` in checkpoint orientation; `_mxfp4_resident = True`
  so NeMo's initializer skips the module; a stack with no experts is not
  registered. Local slot to global expert id (selected first, ascending) is derived
  from the DTensor mesh rank and the selection on first use and cached, not stored
  as a buffer: the pre-shard state dict includes persistent buffers and DCP
  requires every key in the checkpoint. Forward remaps global ids to local slots,
  calls `_permute_tokens_for_grouped_mm` once, splits at the selected token count,
  runs `torch._grouped_mm` on the BF16 stack and `MXFP4GroupedMM` on the packed
  stack, and scatters once. A packed stack with no tokens is skipped; a BF16 stack
  with no tokens runs NeMo's zero-contribution dummy through its expert 0, because
  the update-in-backward stepper fails a step in which a trainable parameter
  received no gradient.
- `deepseek/qat/split_adapter.py`: `SplitStateDictAdapter(DeepSeekV4StateDictAdapter)`
  overrides `convert_single_tensor_to_hf` for split layers (per-slot ids from
  `ExpertSplit`; selected rows become the base adapter's FP4 placeholders,
  packed rows become passthrough views into the module's storage that DCP writes
  through), `_dequantize` (unselected experts keep codes and scales), and
  `_aggregate_experts` (two stacks per layer as DTensors on the expert mesh, with
  completeness and orphan checks). The base splitter asserts a local count of
  experts divided by ranks and cannot be reused for the stacks.
- NeMo patch 1, `components/models/deepseek_v4/model.py`: `DeepseekV4ForCausalLM.__init__`
  pops `expert_split` from its kwargs and calls `install(self)` after the adapter
  and MTP are created. The recipe key is `model.expert_split`, a `_target_` block
  instantiating `ExpertSplit` with `scores: PATH`, so NeMo imports nothing from
  the project; `--set` creates the nested block.
- NeMo patch 2, `components/utils/model_utils.py`: the freeze selector sets
  `requires_grad` per parameter instead of `module.requires_grad_`. Freezing
  applies to every dtype; unfreezing only to floating-point parameters wider than
  one byte. The plan's earlier "per floating-point parameter" rule was wrong: torch
  reports E8M0 as floating point and accepts `requires_grad_(True)` on it, so the
  `unfreeze_modules` glob would have made the packed scales trainable.
- `route_scores.py --per-rank N` (exactly one of `--top-p` and `--per-rank`): the N
  highest-scoring experts of each rank's range, ties to the lower index, every
  layer; the report records `ranks` (the world size) and `per_rank`. The
  full-model all-or-none policy for hash layers (Handoff 5.4) is a separate
  selection file.
- `fakequant.find_targets` skips a split layer with no selected stack.
- `proxy.py`: `local_expert_ids(module, param)` replaces the `rank * E_local + i`
  rule; `check_packed` compares every packed slot with the source bytes
  (`packed_mismatches` in the report, part of `ok`); `--save-logits PREFIX` writes
  the hooked probe logits in the `--expect-logits` format. `proxy_export.py`:
  `expert_ids_per_rank` names gathered rows; export quantizes selected stacks only,
  and unselected experts are copied from the source like every frozen tensor.
- `proxy_step.py`: per-expert tables read the module's ids; packed stacks are
  covered by the frozen digest and counted as `frozen_shard` in the memory
  report; `--expert-selection` together with `model.expert_split` is rejected; the
  report records `expert_split`, `selected_per_layer`, `selected_total`, and
  `packed_per_rank`.
- `proxy-train.yaml` is unchanged: `model.expert_split` is absent so every
  existing arm is unchanged; runs add it with the two `--set` keys below.

Unchanged by evidence: `inbackward.py` and `optim.py` select parameters by
`requires_grad` and the `.mlp.experts` name; NeMo's expert-gradient scaling
identifies expert parameters by the `ep` axis on their DTensor mesh, which both
stacks carry.

Gates, in order, each blocking the next:

1. CPU (passed; `deepseek/tests/test_split_experts.py`, 22 tests): `dequantize_mxfp4`
   equals `quant.py` on sampled source tensors and on a zero-exponent scale (both
   on-the-fly paths decode E8M0 byte 0 as `2**-127`; NeMo's load-time dequantization
   decodes it as 0; the checkpoint has no zero scales); the selection rules reject
   uneven shards; the adapter routes a synthetic two-layer state dict into the
   right stacks and shapes without a mesh and on a one-rank gloo expert mesh
   (`DTensor.from_local`, since gloo cannot scatter int8 or E8M0); the forward on
   random tokens equals a single-stack reference within bf16 tolerance (mixed,
   packed-only, selected-only, and no-token routing) with gradients on the BF16
   stack only; the freeze selectors leave the packed storage frozen; a one-layer
   meta build from the snapshot config runs through the constructor hook and the
   pre-shard key listing. CPU `torch._grouped_mm` accepts empty and zero-row groups.
2. GPU, zero delta (passed): on the split module (`route-scores-perrank10.json`,
   20 experts per layer, 10 per rank), fixed-point, source-parity, and packed
   checks are clean on both ranks; setup 19 s against 169 s because the load
   dequantizes nothing. Probe logits (`proxy-stage6b-single-logits-rank{0,1}.pt`
   against `proxy-stage6b-split-logits-rank{0,1}.pt`): max abs diff 0.878, mean
   9.4e-3, argmax agreement 99.2%, mean KL 2.8e-4; row 0 bit-identical through
   position 10, row 1 through position 2, every later position differs.
   Calibration: the unmodified single-stack path with the probe batch changed
   from 2 to 1 (`proxy --batch 1`, `proxy-stage6b-single-b1-logits-rank0.pt`,
   row 0 against row 0) differs more: max 0.962, mean 1.25e-2, argmax 98.8%,
   mean KL 2.8e-4. Kernel check on 8 real layer-3 experts (ids 100 to 107) and
   4,096 random bf16 tokens in uneven groups: `MXFP4GroupedMM` and `_grouped_mm`
   on the contiguous BF16 stack differ in 0.0001% of outputs by one bf16
   rounding (up to 6 ulps at near-cancelling values), with identical error
   against fp64 (mean 0.25 ulp). The difference is cuBLAS reduction order on the
   transposed operand, entering rarely and amplified by routing and attention.
3. GPU, training and artifact (passed; `proxy-stage6b-export.json`, export
   `deepseek/exports/stage6b-split`): one split-module run on
   `route-scores-perrank10.json` (80 experts, 20 per layer) with `--export`,
   then `verify_export` against the replaced list and a reload under the split
   path. Against the ESFT arm (`proxy-mem-esft-top02.json`, 83 experts, single
   stack): held-out 16.18 to 4.105 (ESFT 4.121), within 0.04 at all twelve
   validation points; selected experts changed 1.63 / 1.46 / 1.77 / 2.16% on
   layers 0 to 3 (ESFT 1.64 / 1.49 / 1.77 / 2.22%); dense 54.22% (54.25%);
   unselected changed 0, frozen clean. Export: 544 candidates, 445 replaced, 99
   unchanged, all scales; `verify_export --expect` exact; reload fixed-point,
   source-parity, and packed checks clean with probe logits bit-identical.
   Memory (GiB per rank): peak step 44.36 (ESFT 53.45), resident after step
   12.19 (29.17: experts 1.88 against 24.0, frozen shard 7.40 against 3.05,
   optimizer state 2.40 against 3.12), hook-entry peak 25.09 (47.58), transient
   32.15 (24.28). Time: training 509 s (ESFT 2,297 s), setup 21 s (175 s),
   export 158 s. The transient grew by 7.9: `dequantize_mxfp4` builds two
   full-size fp32 temporaries before the bf16 cast, so a layer's packed stack
   costs about 2.5 times its bf16 size while it is dequantized; it scales with
   packed experts per rank per layer (118 here). The earlier form of this gate,
   a single-stack baseline on the same selection with identical `masters_sha256`
   and a reload under the single-stack path, was dropped: masters cannot agree
   across kernel paths (gate 2, Verified facts), and the cross-path reload adds
   nothing to `verify_export` plus the same-path reload.
4. GPU, estimate (optional, unblocking): a 16-layer proxy memory run at 2 ranks
   (a 16-layer `--per-rank` selection first, both with
   `--set model.config.num_hidden_layers=16`), giving the 4x B200 column a
   measured transient. Its only value is avoiding a wrong rental; the first live
   step measures the same thing.

Gate commands (both GPUs, `nohup` from a plain terminal; `$SNAP` as above):

```sh
SPLIT="--set model.expert_split._target_=deepseek.qat.split_experts.ExpertSplit --set model.expert_split.scores=deepseek/exports/logs/route-scores-perrank10.json"
COMMON="--config deepseek/qat/proxy-train.yaml --in-backward --set optimizer.lr=1e-4 --set step_scheduler.max_steps=100 \
    --set step_scheduler.val_every_steps=10 --set step_scheduler.global_batch_size=2 --set step_scheduler.local_batch_size=1"
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.route_scores \
    --config deepseek/qat/proxy-train.yaml --per-rank 10 --out deepseek/exports/logs/route-scores-perrank10.json
# gate 2
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy --config deepseek/qat/proxy-4layer.yaml \
    --snapshot $SNAP --save-logits deepseek/exports/logs/proxy-stage6b-single
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy --config deepseek/qat/proxy-4layer.yaml \
    --snapshot $SNAP $SPLIT --expect-logits deepseek/exports/logs/proxy-stage6b-single
# gate 3
PYTORCH_ALLOC_CONF=expandable_segments:True deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy_step $COMMON \
    $SPLIT --out deepseek/exports/logs/proxy-stage6b-export.json --export deepseek/exports/stage6b-split
deepseek/venv/bin/python -B -m deepseek.qat.verify_export $SNAP deepseek/exports/stage6b-split \
    --expect @deepseek/exports/logs/proxy-stage6b-export-replaced.txt
deepseek/venv/bin/torchrun --nproc-per-node 2 -m deepseek.qat.proxy --config deepseek/qat/proxy-4layer.yaml \
    --snapshot deepseek/exports/stage6b-split $SPLIT --set model.config.pretrained_model_name_or_path=deepseek/exports/stage6b-split \
    --set model.config.name_or_path=deepseek/exports/stage6b-split --expect-logits deepseek/exports/logs/proxy-stage6b-export
```

Hardware estimate at 8% selection (GiB reserved per rank, same model as Handoff 2;
the per-layer dequantization transient measured 7.9 at 118 packed experts per
rank per layer in gate 3, so about 4 at 4 ranks and 2 at 8, inside the allowance
below): 2 ranks
about 170 (B300 class only); 4 ranks about 104 (B200: fits; RTX PRO 6000: no);
8 ranks about 71 (RTX PRO 6000: fits; B200: fits); 16 ranks about 54. Selection
fraction matters here: at 30%, 8 ranks is about 99 and 4x B200 about 130. The
local 2x RTX PRO 6000 host still cannot train the full model (about 170 at 2
ranks); it gains a 15 to 20 layer proxy.

Not in scope: uneven shards by per-layer `top_p`; MXFP4 compute for selected
experts; selection scoring changes beyond the per-rank quota.

### Stage 7: full model (planned)

Container from the gate-passing venv lockfile with a verifier script run first.
Routing scores over the training data on the training host, then selection, then
one update, bounded convergence run, export, fresh-process equality, Transformers
and vLLM probes on hardware that serves the source. Memory plan with the
expert-parallel group spanning the world and ESFT-style selection (Handoff 5):
each rank owns its share of every layer's experts outside FSDP as dequantized
BF16: 43 layers of 6,442,450,944 elements, 516 GiB (the earlier 593 GB counted the
three unloaded MTP modules), so 64.5 GiB at eight ranks and 32.3 at sixteen, which
dominates; 8-bit moments are about 1.5 GiB for dense and about 5.3 for experts at
the 8% selection of Handoff 5 (about 66 at eight ranks if every expert were
trained, so the pre-selection figure of 141 GB per GPU no longer applies). Stage 6b
keeps unselected experts packed, cutting expert storage to about 167 GiB; its
figures are in that stage, and the ones below are for the single-stack path.
Estimate (labelled; the first
full-model run's `memory` block is the measurement), from the proxy's Handoff 2
figures at local batch 1 and rows to 4,352 tokens: resident about 73 GiB at eight
ranks (experts 64.5, dense shard 1.5, frozen 0.35, dense state 1.5, expert state
5.3); transient about 32 to 39 (the proxy's 24.3, plus dense and frozen parameters
held unsharded through forward and backward growing from 2.0 to about 12.7 under
`reshard_after_forward: false`, plus checkpoint-saved activations for 39 more
layers, minus four times smaller per-rank expert gradients and fake-quant copies);
peak allocated about 105 to 112 at eight ranks and 120 to 127 reserved, outside a
96 GB GPU. Two levers: `distributed.moe.reshard_after_forward: true` removes most
of the unsharded-dense term (about 95 to 101 allocated at eight ranks), and
sixteen ranks give about 69 to 76 allocated (58 to 65 with resharding) and 80 to 87
reserved, inside a 96 GB (89.4 GiB) GPU. The measured peak of the first full-model
run is the per-GPU memory requirement and selects the host. The preparation
decisions are listed in Handoff 6. Expert flips arrive tens of steps
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
  Measured in Handoff 2: 9 to 14 GiB at two layers.
- NeMo `param_group_overrides` only stores `lr_mult` on a group; the multiplier is
  applied by `OptimizerParamScheduler.step`, so it is inert without an
  `lr_scheduler` block. The scheduler calls `step(0)` at build and `step(1)` after
  each optimizer step, so every rate describes the next step, which is compatible
  with update-in-backward. torchao's Adam keeps each group's `lr` as a tensor and
  raises when a float replaces it, which NeMo's scheduler does.
- After a bare forward with `reshard_after_forward: false`, FSDP2 module
  attributes point at the unsharded parameter objects, not the sharded ones the
  optimizer registered; parameter lookups across that boundary must use names.
- FSDP2 dense and frozen parameters are sharded between steps and unsharded
  through forward and backward under `reshard_after_forward: false`; a bare
  forward (validation) leaves them unsharded until the next backward. Resident
  accounting must therefore sample after a backward (`proxy_step` samples at the
  end of the first step).
- The BF16 gradient set equals trainable elements times two bytes exactly:
  12.246 GiB for 6,574,571,520 elements, measured as optimizer-entry allocation
  minus steady resident state on the two-layer proxy.
- Probe logits are bit-exact only along one kernel path. A change of reduction
  order alone (probe batch 2 to 1 on the unmodified single-stack path) moves
  logits by up to 0.96 with 98.8% argmax agreement, because one bf16 rounding
  difference is amplified by top-k routing and attention. Cross-path comparisons
  are made at the kernel level and calibrated against that run (stage 6b gate 2).

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
