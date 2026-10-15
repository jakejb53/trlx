# Proposed DeepSeek-V4 LoRA integration

Status: proposed; not part of the current `trlx` contract.

## 1. Goal

`trlx` post-trains an official DeepSeek-V4-Flash Hugging Face checkpoint with
LoRA and publishes a standalone merged checkpoint in the same external format
as that source checkpoint. The result has no runtime adapter dependency and
loads through the same Transformers and vLLM model paths as the source.

The training representation is not constrained. The base may be dequantized
for training provided publication restores every output tensor to the source
tensor's role, name, shape, dtype, quantization geometry, and scale format.

## 2. Initial scope

- SFT with a frozen official DeepSeek-V4-Flash base and trainable LoRA adapters.
- Official dated revisions whose metadata names `DeepseekV4ForCausalLM` and
  whose tensor formats pass the compatibility checks in section 5.
- NeMo AutoModel as the training engine.
- FSDP2 and the tensor, pipeline, context, data, and expert parallel layouts
  supported by the selected NeMo version.
- LoRA on explicitly supported attention, compressor, and other non-routed-
  expert projections.
- Adapter checkpoints for resume and a merged official-format checkpoint at
  successful completion.

The initial scope excludes full-weight training, reward-model training,
routed-expert LoRA, router or mutable-buffer training, MTP/DSpark training,
and conversion to NVIDIA ModelOpt NVFP4 or another checkpoint format.

## 3. Ownership

`trlx` owns:

- Configuration resolution, input validation, dataset preparation, review,
  run allocation, supervision, cancellation, diagnostics, and terminal output.
- The source checkpoint identity and compatibility report.
- Translation of approved settings into an engine request.
- Normalization of engine events and metrics into `log.txt` and
  `metrics.jsonl`.
- Adapter artifact collection and resume identity.
- Streaming merge, source-format requantization, publication, and final
  verification.

NeMo AutoModel owns:

- Model execution and the dequantized training representation.
- Distributed model placement and collectives.
- Forward, backward, gradient checkpointing, optimizer, and scheduler work.
- LoRA injection for the approved target list.
- Distributed training checkpoints and HF-compatible adapter export.

The integration uses NeMo's supported invocation boundary. `trlx` does not
copy NeMo model, parallelism, or kernel implementations into this repository.

## 4. Engine configuration and protocol

The run configuration explicitly selects the external engine:

```toml
[engine]
name = "nemo_automodel"
command = ["/absolute/path/to/python", "-m", "trlx_nemo_worker"]

[engine.distributed]
fsdp2 = true
tp_size = 1
pp_size = 1
cp_size = 1
ep_size = 1
```

`command` is an argv list, not a shell string. Every key is required when the
block is present. The operator owns installation of a compatible NeMo
environment; the project dependency pins remain unchanged.

Before allocating or rewinding a run, `trlx` executes a read-only engine probe
that reports the engine, NeMo, PyTorch, Transformers, PEFT, CUDA, and kernel
versions and validates the requested topology. Missing capabilities and
unsupported combinations are fatal.

The engine worker receives one versioned JSON request describing resolved
paths, model identity, prepared datasets, LoRA settings, trainer settings,
parallel topology, output paths, and resume state. It emits versioned JSONL
events to stdout and raw diagnostics to stderr. Unknown request or event keys
are errors. Secrets are passed by environment-variable name and never written
to requests, snapshots, logs, or engine configuration.

The engine process belongs to the run's worker process group and follows the
existing cancellation and bounded failure-cleanup contract. A worker failure
preserves its structured cause and engine stderr in `log.txt`.

## 5. Source checkpoint compatibility

The supervisor resolves the exact source revision and records its repository
ID or local path, commit, model type, architecture, configuration digest,
weight-index digest, tensor inventory, and tokenizer digest.

Compatibility requires:

- Architecture `DeepseekV4ForCausalLM`.
- A complete safetensors index with no missing or duplicate tensor keys.
- Every tensor classified as BF16, FP8 E4M3 plus its inverse-scale tensor,
  packed E2M1 FP4 plus its inverse-scale tensor, or an explicitly supported
  integer/buffer type.
- FP8 block geometry and FP4 scale geometry derivable unambiguously from the
  tensor and scale shapes.
- UE8M0 or other scale dtypes explicitly recognized by the merger.
- Every requested LoRA target mapped losslessly between source HF names,
  NeMo names, and exported PEFT names.
- No requested target inside routed-expert packed tensors in the initial
  implementation.

An unknown architecture field, tensor format, quantization layout, target,
conversion mapping, or source revision fails before training. There is no
fallback to logical-BF16 size estimates or a different model artifact.

## 6. Dataset preparation

`trlx` remains authoritative for dataset loading, splitting, replay selection,
reasoning inclusion, reasoning-only labels, packing eligibility, source
fingerprints, and assessment. The engine consumes a run-owned prepared
representation and does not reload the operator's source datasets.

The prepared representation contains the exact token IDs and labels selected
by `trlx`, plus fields required for source identity and supported engine
collation. NeMo must not reapply a chat template, independently truncate rows,
resplit data, or change labels. Startup profiling and engine-observed row and
trained-token counts must agree before the first optimizer update.

## 7. LoRA target contract

The initial target set is an explicit allowlist resolved against the loaded
model. It may contain supported attention, compressor, shared non-expert, and
output projections. Bare `all-linear` is rejected for this backend because it
does not state whether packed routed-expert parameters are intended.

Every target must have:

- A unique HF source tensor or supported tensor group.
- A unique NeMo module and parameter mapping.
- A unique exported PEFT key and target metadata entry.
- A supported merge and source-format requantization path.

The preflight report lists every resolved target, source tensor format, adapter
shape, trainable count, and merge path. Zero targets, partially resolved target
groups, unsupported expert targets, and trainable non-adapter parameters are
fatal. Router correction biases and other mutable base buffers remain frozen
and must match the source after training and resume.

## 8. Training artifacts and resume

The run directory additionally contains:

```text
<run>/
  engine-request.json
  engine-compatibility.json
  engine-checkpoints/
  adapter/                 final HF-compatible adapter
  merge-report.json
  merged/                  standalone official-format checkpoint
```

Engine checkpoints contain the adapter, optimizer, scheduler, random-number,
sampler, and distributed state required for exact continuation. They do not
publish a second copy of the dequantized base model.

Resume requires the same source-checkpoint identity, prepared-data identity,
target mapping, LoRA settings, engine compatibility identity, and distributed
state schema. Existing `trlx` rewind rules apply. A completed or partial merge
is never a resume source.

## 9. Merge and requantization

Merge reads the immutable source checkpoint and final adapter, stages a
complete output, and publishes only after all structural and numerical checks
pass. It operates shard by shard and tensor by tensor; it must not materialize
the whole model in host or device memory.

For each source tensor:

- An untouched tensor is copied without numerical conversion and its tensor
  payload digest must equal the source digest.
- An adapted BF16 tensor is dequantized if necessary, updated by its LoRA
  delta in at least FP32 accumulation, cast back to BF16, and written under the
  original key.
- An adapted FP8 tensor is dequantized, updated by its LoRA delta in at least
  FP32 accumulation, and requantized to the source E4M3 format using the
  source block geometry and scale dtype.
- Packed FP4 expert tensors and scales are untouched in the initial scope.

Requantization recomputes scales from the merged tensor; it never reuses stale
source scales. Rounding, saturation, zero-block, non-finite, and power-of-two
UE8M0 rules are explicit and deterministic. An updated tensor whose shape is
not divisible by its source block geometry is an error.

The output preserves the source weight-index mapping, shard names, tensor key
set, configuration, tokenizer, generation configuration, encoding resources,
and other required repository files. Only adapted tensor payloads, their scale
payloads, and provenance metadata may differ. No LoRA keys or PEFT metadata
remain in `merged/`.

The merge report records source and adapter identities, every changed tensor,
source and output formats, scale geometry, saturation counts, non-finite
counts, payload digests, and publication paths. Existing force, overlap,
staging, and recovery rules apply.

## 10. Verification

Verification has four required stages:

1. **Structure:** output files, configuration, tokenizer, index, keys, shapes,
   dtypes, and scale grids satisfy the source-format contract.
2. **Preservation:** every untouched tensor payload is identical to the source;
   every changed tensor is named in the merge report; the output contains no
   adapter state.
3. **Numerics:** on fixed calibration inputs, compare the dequantized
   base-plus-adapter reference with the merged/requantized output using logits,
   token loss, and generated tokens. Non-finite values and unreported output
   changes are fatal.
4. **Runtime:** load the published artifact through the supported Transformers
   and vLLM paths and complete deterministic forward and generation probes.

Quantization makes exact logit equality impossible. Acceptance thresholds for
logit error, loss change, and generation agreement are versioned verification
inputs established by calibration against an unchanged round-trip control.
The control dequantizes and requantizes the source without an adapter; adapter
verification must distinguish merge error from that baseline quantization
error. Thresholds are recorded in the snapshot and merge report and may not be
silently relaxed.

Verification also proves that at least one adapted tensor changed, adapter
effects survive requantization, and representative prompts differ from the
unmodified source when the dequantized reference differs.

## 11. Failure and cancellation

Expected engine, checkpoint, conversion, quantization, storage, and validation
failures produce contextual `TrlxError` reports without Python tracebacks on
the terminal. Reports identify the operation, rank, source tensor or row,
affected path, and retained recovery artifact where applicable.

Cancellation stops training cooperatively through the engine worker group.
Merge cancellation stops before publication and retains only an identified
staged recovery directory. Cancellation never publishes a partial merged
checkpoint.

## 12. Dependency and version policy

The NeMo engine environment and merger compatibility matrix are explicit and
versioned. A newer model, NeMo, Transformers, PEFT, PyTorch, CUDA, or kernel
version is unsupported until its compatibility probes and reference tests
pass. Compatibility is based on observed capabilities and tensor contracts,
not version ordering alone.

`trlx` does not install or update the external engine automatically. The
operator-provided engine command is invoked only for a run that explicitly
selects it.

## 13. Deferred extensions

- Routed-expert LoRA and official E2M1 FP4 merge/requantization.
- Router, correction-bias, indexer, MTP, or DSpark training.
- Full-weight DeepSeek-V4 training.
- DPO, KTO, GRPO, RLOO, reward, and distillation through the NeMo engine.
- Pipeline-specific quality evaluation or generation during training.
- Automatic conversion to ModelOpt NVFP4 or another non-source format.

Each extension requires a separate target, checkpoint, merge, verification,
and resume contract. It must not weaken the initial preservation guarantees.

## 14. Open decisions

- The first approved LoRA target allowlist.
- Whether the external engine runs in a host environment or a configured
  container boundary.
- The engine request/event schema and distribution mechanism.
- Calibration datasets and numerical acceptance thresholds for each supported
  checkpoint revision.
- Whether successful training always performs merge and verification or
  permits an explicit adapter-only completion mode.
