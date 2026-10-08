# trlx

**TRL training through an intuitive CLI, without bespoke Python training scripts.**
`trlx` brings seven TRL trainers together with reusable configuration, LoRA and full
fine-tuning, multi-GPU execution, evaluation, and live metrics. Its companion
`dataset` CLI creates and manages the training and evaluation data those trainers need.
Its [interactive web UI](#interactive-dataset-authoring) lets you compare responses
from different endpoints, edit answers and reasoning, and save selected examples
to a training dataset. The [stateless CLI](#stateless-cli-authoring) exposes generation
and saving for scripts and agents. Its [reusable Context builder](#reusable-context-builder)
incrementally assembles source-backed conversation histories—including fabricated
tool calls and supplied results—directly in the format used for generation. The
[agent-assisted authoring workflow](#agent-assisted-dataset-authoring) turns those
pieces into a process: an agent builds a source-backed Context that teaches the
target model a topic, generates on-policy reasoning and answers, repairs them under
a strict editing contract with likelihood scoring and an adversarial review, and
saves verified rows. The `authoring/` scripts implement each check.

## Why trlx exists

Using TRL directly gives you Python trainer APIs; you assemble the surrounding workflow
for loading models and datasets, configuring trainers, launching distributed jobs,
recording results, and checking saved artifacts. trlx makes that workflow available
through commands and a TOML configuration, so experiments do not each need a custom script.

TRL supplies the trainers. trlx exposes their method-specific settings as CLI options
and combines them with the tools needed before, during, and after training:

| Need | What trlx provides |
| --- | --- |
| Choose a training objective | SFT, DPO, KTO, reward modeling, GRPO, RLOO, and distillation, with trainer settings and per-run overrides. |
| Prepare suitable data | Format conversion, text chunking, reusable source-backed generation Contexts, endpoint-generated Q&A and evaluation summaries, preference pairs, filtering, splitting, mixing, sampling, repair, and statistics. |
| Check a setup before committing to training | Full dataset inspection, a review of effective tuning settings, and preflight checks for model, adapter, dataset, and trainer compatibility. |
| Use available GPUs | Hardware-aware initialization, automatic data-parallel or sharded launch, and explicit device/strategy overrides without a separate launcher configuration. |
| See what training is doing | Live metric tables or a full-screen TUI, loss charts, measured progress, complete diagnostic logs, and saved metrics that can be viewed later. |
| Measure results | Step-zero evaluation, held-out or synthetic evaluation data, and optional independent quality benchmarks with per-example evidence. |
| Train on reasoning or preserve earlier behavior | Native reasoning inclusion, reasoning-only loss, and SFT replay with optional KL regularization. |
| Continue and export work | Run-owned configuration and prompt snapshots, checkpoint resume, artifact verification, and LoRA merging. |
| Handle failures and replacement safely | Contextual errors, coordinated cancellation and distributed failure cleanup, explicit replacement permission, and staged output publication. |

You can keep shared defaults, specialize them by method, and change individual runs
from the command line. The supported scope is the seven trainers below; some settings
are managed by trlx or restricted when features conflict. Model classes come from
model metadata, rather than a hardcoded model family. `dataset` also works independently
of TRL, so its preparation tools are useful outside a trlx training run.

## Guide

- [Start here](#start-here) and [installing updates](#installing-updates)
- [Start the web UI and author datasets interactively](#interactive-dataset-authoring)
- [Author through the CLI](#stateless-cli-authoring) and [author datasets with an agent](#agent-assisted-dataset-authoring)
- [Prepare datasets](#dataset-cheat-sheet) and [generate data through endpoints](#tutorial-endpoint-generation-and-credentials)
- [Choose a trainer](#training-cheat-sheet) and [configure runs](#tutorial-temporary-and-persistent-settings)
- [Train on reasoning](#sft-reasoning-supervision), [use rewards](#tutorial-rewards), and [mix replay data](#tutorial-sft-replay)
- [Configure LoRA and GPUs](#tutorial-lora-memory-and-gpus)
- [Choose evaluation data](#tutorial-evaluation-and-data-sources) and [run quality checks](#settings-assessment-and-independent-quality-checks)
- [Validate a setup](#validation-before-and-after-training) and [monitor or stop training](#monitoring-cancellation-and-failures)
- [Resume, inspect, and merge results](#tutorial-checkpoints-resume-and-results)
- [Customize prompt files](#prompt-files) and [control output replacement](#output-replacement-and-staging)

## Start here

Activate the Python environment you intend to use. It needs Python 3.11+ and a
PyTorch build suitable for your CUDA setup; training needs a visible CUDA GPU.
From the repository root, install and start:

```sh
python -m pip install .
trlx init
trlx sft --model MODEL --dataset DATA
```

`init` also creates `prompts/` beside its output config. Existing config or generated
prompt files require `--force`, which resets their contents, including edited prompts.
Other files, including an operator-authored `llm-judge.prompt`, are preserved.
For an existing installation, see "Installing updates" below.

Replace `MODEL` with your model directory or model ID, and `DATA` with a dataset file
or `org/name:split`. For SFT, data needs `messages`, `text`, or `prompt` + `completion`.
The default split requires at least two rows. See the method/data cheat sheet below.

`init` writes `run.toml` once for your environment and all seven training methods.
It enables LoRA, holds out 10% of the data for evaluation, and supplies training
defaults. No config editing is required. Training reads `./run.toml` automatically,
runs one epoch by default, and saves each run in its own directory under `runs/METHOD`.
For example: `runs/sft/20260920-1--qwen-qwen3.8-27b--chunks/`. Training prints the actual
directory at startup; use that path wherever `RUN_DIR` appears below.

Training first inspects model metadata/tokenizers and scans every effective dataset row. It then prints
tuning settings and an assessment with supporting evidence. Press Enter to load weights and continue,
or `q` to quit before run files change. Metadata/data downloads and library cache writes can occur first.
This review also applies to `--tui` and resume. EOF or failed review I/O stops startup with an error;
`--force` does not skip the review. Quit and relaunch with different overrides to change settings.

**CLI options change one run. Editing `run.toml` changes future runs.**
For example, add `--learning-rate 5e-5` for one run or edit `learning_rate` under
`[methods.sft]` to keep that value. Use `--output-dir runs/experiment` to change the parent
directory for new runs.

```sh
trlx show RUN_DIR
trlx --help
trlx sft --help
dataset --help
```

Help lists commands, options, defaults, constraints, and examples; it needs no
config, model, dataset, or GPU. The sections below are reference and optional tuning.

## Installing updates

Editing or updating the checkout does not update an installed copy of the tools.
With the same Python environment active, run this from the repository root to
refresh the installed code and both commands:

```sh
python -m pip install --force-reinstall --no-deps .
```

This keeps installed runtime dependencies and `run.toml` unchanged. If the dependency
requirements in `pyproject.toml` changed, run the normal installation command first
to update them.

## Dataset cheat sheet

`dataset` works independently of TRL. Files may be JSONL (one object per line), JSON
(an array of objects), CSV, or Parquet. The output extension selects the file format.
CSV cells are strings; use JSONL or Parquet for nested messages. Replacing an existing
output or input requires `--force`. File-transform commands support `--no-staging`
for direct writes. Authoring saves append unique JSONL examples without `--force`
and always stage publication; see [CLI authoring](#stateless-cli-authoring).

```sh
# Convert the file format, or reshape prompt/completion rows into messages.
dataset convert data.json --out data.jsonl
dataset convert prompts.jsonl --to messages --out chat.jsonl
dataset convert chat.jsonl --to prompt-completion --out prompts.jsonl

# Reorder, divide, combine, or sample rows.
dataset shuffle data.jsonl --seed 42 --out shuffled.jsonl
dataset split shuffled.jsonl --fraction 0.9 --out train.jsonl --rest eval.jsonl
dataset mix a.jsonl b.jsonl --fractions 1,0.25 --seed 42 --out mixed.jsonl
dataset sample data.jsonl --n 20 --seed 42 --out sample.jsonl
dataset sample data.jsonl --n 10 --head --out preview.jsonl

# Edit columns or select rows.
dataset fields data.jsonl --add 'length=len(text)' --out sized.jsonl
dataset fields data.jsonl --add 'source="manual"' --out tagged.jsonl
dataset fields data.jsonl --remove unused --rename answer=completion --out edited.jsonl
dataset fields pairs.jsonl --swap chosen=rejected --out swapped.jsonl
dataset filter data.jsonl --where 'label == True' --out positive.jsonl
dataset filter data.jsonl --max-length text=4000 --out short.jsonl

# Chunk plain text, build preference pairs, or extract prompt/completion rows.
dataset cpt source.txt --max-tokens 1024 --out chunks.jsonl
dataset pairs chosen.jsonl rejected.jsonl --strict --out pairs.jsonl
dataset pairs messages.jsonl --out prompts.jsonl

# Repair syntax, inspect lengths, or score responses with a model.
dataset heal broken.jsonl --out repaired.jsonl
dataset stats data.jsonl
dataset stats pairs.jsonl --columns chosen,rejected --model MODEL

# Generate question/answer data from text: complete example in the endpoint tutorial.
dataset chat --help
dataset fields --help
```

- `split` takes either `--n` or `--fraction`; `--key COLUMN` keeps groups together
  and may exceed the requested count. It preserves input order.
- `mix` fractions select a share of each input, not a share of the output.
  Sources are concatenated; shuffle afterward to interleave.
- `fields` and `filter` expressions use column names or `row["column-name"]`;
  `len`, `str`, `int`, and `float` are available. Field operations run in the order
  add, remove, rename, swap. Flags may repeat; all filter conditions must pass.
- Length limits count characters or list items. `cpt`, `chat`, and plain `stats`
  estimate tokens as characters / 3.5; `stats --model` loads weights for exact
  token counts and response log-probabilities.
- `pairs` requires `messages` ending in an assistant turn. Two inputs align by
  user-turn content; unmatched rows are omitted unless `--strict` makes them fatal.
- `heal` reports every repair, may drop a truncated final JSONL row, and writes
  output even when errors remain; remaining errors produce a nonzero exit.

## Training cheat sheet

Choose the method on the training command; there is no method-specific initialization.

| Method       | Use                           | Required data                                  |
| ------------ | ----------------------------- | ---------------------------------------------- |
| sft          | Supervised fine-tuning        | messages, text, or prompt + completion         |
| dpo          | Learn from preference pairs   | chosen + rejected; prompt for explicit prompts |
| kto          | Learn from individual ratings | prompt + completion + boolean label            |
| reward       | Train a reward model          | Same preference pairs as DPO                   |
| grpo         | Train a policy using rewards  | prompt + any reward-specific columns           |
| rloo         | Train a policy using rewards  | prompt + any reward-specific columns           |
| distillation | Learn from a teacher model    | prompt; also supply --teacher                  |

Conversational columns contain lists of `{role, content}` messages. An SFT JSONL row:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

Models whose architecture registers a bundled Python message encoder do not need a Jinja chat template.
DeepSeek-V4 is detected from its model metadata and uses its native encoding automatically across SFT,
DPO, KTO, GRPO, RLOO, and distillation. Its default training encoding adds no reasoning-effort instruction;
explicit chat-template kwargs remain available for generation requests. `reward` additionally requires the
selected checkpoint to provide a sequence-classification model class, which the base DeepSeek-V4-Flash-0731
checkpoint does not provide. An explicit `chat_template_path` overrides automatic encoding.

The normal command for SFT, DPO, KTO, or reward-model training is:

```sh
trlx sft --model MODEL --dataset DATA
trlx dpo --model MODEL --dataset DATA
trlx kto --model MODEL --dataset DATA
trlx reward --model MODEL --dataset DATA
trlx distillation --model MODEL --dataset DATA --teacher TEACHER
```

GRPO and RLOO also need an explicit reward and a running generation server;
see "Rewards" below. Each method creates run subdirectories under `runs/METHOD` unless overridden.

```sh
# Check the setup without training; accepts the same model/data/trainer overrides.
trlx check sft --model MODEL --dataset DATA

# Live full-screen training, or inspect an existing run.
trlx sft --model MODEL --dataset DATA --tui
trlx show RUN_DIR --tui

# Verify a checkpoint or produce a merged model from an adapter.
trlx verify CHECKPOINT --base BASE
trlx merge --base BASE --adapter ADAPTER --out merged-model

# Generate replay data locally.
trlx replay-build --model MODEL --prompts prompts.jsonl --out replay.jsonl --max-tokens 256

# Discover method-specific options and utility commands.
trlx check dpo --help
trlx replay-build --help
```

## SFT reasoning supervision

Reasoning datasets can retain a model's explanation separately from its final answer.
trlx maps that explanation into the model's native chat format and validates which
tokens contribute to loss. Choose whether to train on the answer as well:

```sh
# Include reasoning and the answer; exclude user/system tokens from loss.
trlx sft --model MODEL --dataset reasoning.jsonl --include-reasoning --assistant-only-loss

# Train only on reasoning and its native boundaries.
trlx sft --model MODEL --dataset reasoning.jsonl --reasoning-only-loss
```

`reasoning.jsonl` must contain raw text `messages` rows with a separate nonempty
`reasoning` string. For example, with a model whose template supports reasoning:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}],"reasoning":"Adding two to two gives four."}
```

For SFT datasets with a separate `reasoning` column, add `--include-reasoning` to train on it.
Each train/evaluation/replay row must contain nonempty reasoning and exactly one assistant response,
at the end of its text conversation. The model's response metadata and chat template determine the
native field; unsupported templates, conflicting fields, or selected reasoning lost to masks/packing
are errors. Tokenizer offset mappings are required for validation. Source datasets are not rewritten.
When a row exceeds `--max-length`, native template structure is preserved and reasoning gets the
token budget first, followed by the final answer, then the beginning of user/system content.
Reasoning that cannot fit is truncated at its end. This happens automatically without truncation notices.
The option defaults off, persists as `[dataset].include_reasoning`, and can be overridden with
`--no-include-reasoning`. Use `--assistant-only-loss` independently to exclude user tokens from loss.
Already embedded native reasoning continues through the normal template path without this option.

Use `--reasoning-only-loss` to score only the reasoning and its native opening/closing boundaries.
It implies `--include-reasoning`; user/system text, the final answer, and final-answer termination
tokens are masked. The same mask applies to evaluation and replay, so loss and token accuracy now
measure reasoning tokens. When fitting oversized rows, masked final-answer content is omitted.
Explicit, nonempty boundaries must be verifiable from the template metadata;
unsupported boundaries and tokens crossing into the answer are errors. The dataset format is unchanged;
the final answer may be empty if the template supports it. The option defaults off and persists as
`[dataset].reasoning_only_loss`; `--no-reasoning-only-loss` disables it. Explicitly disabling reasoning
inclusion while enabling reasoning-only loss is an error. `--assistant-only-loss` can remain enabled.

Reasoning inclusion is incompatible with synthetic CPT evaluation and skipped dataset
preparation. The length limit must fit the native template structure and a reasoning token.

## Output replacement and staging

Uppercase names are placeholders; replace them with your own inputs. Paths are
relative to the working directory. Every command accepts `--force`. Without it,
destructive operations are refused with their consequences and the flag needed to proceed.
With it, the requested replacement proceeds without another confirmation, including replacing
inputs or directories containing unrelated files. Destructive replacement replaces symlinks
themselves and leaves their targets alone; healing follows symlinks to repair their targets.

| Command                  | Replacement authorized by --force                         |
| ------------------------ | --------------------------------------------------------- |
| trlx init                | Config and named generated prompt files, reset to defaults |
| trlx merge               | Entire output path, including base/adapter input paths    |
| trlx replay-build        | Existing output dataset, including an input path          |
| trlx verify              | Existing verify.json report                               |
| dataset writers          | Existing outputs, including inputs; heal repairs them     |
| Training                 | Auto verify report; new runs and resume rewind unchanged  |
| check/show/dataset stats | No destructive output                                     |

`init`, `merge`, `replay-build`, `verify`, and dataset writers prepare complete outputs in
staging space by default. Add `--no-staging` to write directly after required inputs are read.
It saves staging disk space but a write failure can lose the original and leave incomplete output.
Existing destinations still require `--force`; `--no-staging` does not grant replacement permission.
Merge requires disk staging when replacing its base, adapter, or a directory containing either.
For those in-place merges, omit `--no-staging`; an incompatible request is rejected before loading inputs.
Directory replacement and publishing both split outputs are not atomic transactions.
Training also accepts `--no-staging` for its config snapshot, resume evidence rewrites, assessment,
quality, preflight, and verification reports. Trainer checkpoints are saved directly in either mode.

```sh
# Replace a separate old merge directly; MERGED must not contain BASE or ADAPTER.
trlx merge --base BASE --adapter ADAPTER --out MERGED --force --no-staging

# Replace the base model using default disk staging.
trlx merge --base BASE --adapter ADAPTER --out BASE --force
```

## Validation before and after training

Validation happens at several stages, with different inputs and costs:

| Stage | What it checks | When it runs |
| --- | --- | --- |
| Startup inspection and review | Configuration, model metadata/tokenizers, and all effective train/evaluation/replay rows; tuning findings include measured evidence and preparation projections. | Before weight loading and run allocation or resume rewind; synthetic evaluation rows are generated later. |
| Preflight | Loaded model and prepared data, resolved LoRA targets, trainable parameters, label masks, truncation, and method-specific compatibility. | Before training, or by running `trlx check METHOD`. |
| Checkpoint verification | Model loading, adapter integrity, and chat-template consistency with the base. | After successful training unless disabled, or by running `trlx verify`. |

```sh
# Load and check a setup without training or prompting for review.
trlx check sft --model MODEL --dataset DATA

# Inspect a saved checkpoint; --force is needed only to replace an existing report.
trlx verify CHECKPOINT --base BASE
```

Preflight reports resolved LoRA targets and the trainable parameter count. It warns
about response truncation, an empty trained-token mask in the first prepared example,
missing/EOS padding tokens, and gradient checkpointing combined with caching.
DPO and KTO also score starting-model response log-probabilities on the configured
preflight sample to flag potentially off-policy data. GRPO and RLOO check for the
TRL vLLM server needed for weight synchronization.

Fatal checks stop the job with a contextual error; advisory warnings remain visible
and allow training to continue. `preflight.json` retains the findings, example text,
and trained-token information. Standalone `check` uses one GPU and requires the model
to fit there; larger sharded models receive preflight inside their training run.

Verification runs in a separate process after the training workers exit. For adapters,
it compares the count and maximum magnitude of saved `lora_B` tensors with the loaded
adapter. Full checkpoints must load successfully, and the saved chat template must
match the base. Results go to `verify.json`; a failed verification makes the command
exit nonzero. Generation quality is measured by evaluation and quality checks below.

## Monitoring, cancellation, and failures

Line mode displays training and evaluation tables with step/epoch progress, metric
values, and changes from prior measurements. Headings repeat after interruptions;
narrow terminals use column groups so values remain visible. `[ranges]` selects
additional metrics and marks values outside configured intervals.

Use `--tui` for a full-screen view of progress, recent metrics, checkpoints, diagnostics,
and validation results. Inspect saved results with `trlx show RUN_DIR` or add `--tui`.
`metrics.jsonl` is the authoritative metric history; `log.txt` retains full diagnostics
with their sources. Display failure stops presentation while supervision continues.

Both tools report current operations, measured counts when available, retries, and
final outcome with elapsed time. Opaque operations have no inferred percentage.
After ten seconds without feedback, ordinary command waiting notices show elapsed
time and the last measured progress. Training's supervisor uses 30-second terminal
notices outside training; while workers train, waiting notices remain in the log.
Repeated warnings and rank progress are consolidated in terminal output.

The first Ctrl+C requests clean shutdown at a shared worker boundary. Workers finish
their current coordinated GPU work before releasing resources; this may take 60 seconds
or longer without automatic timeout escalation. A second Ctrl+C forces termination
of owned process groups. Cancellation skips completion-only work and verification,
reports shutdown progress, and exits with status 130.

A fatal worker failure stops the other workers and requests NCCL communicator abort.
Cleanup preserves the original error, reports a 30-second abort/exit limit, and escalates
to termination signals if cleanup cannot complete. GPU recovery is a separate manual
operation, described [below](#recovering-stuck-gpu-activity).

## Settings assessment and independent quality checks

Choose the measurement that answers your question:

| Measurement | Data | What it tells you |
| --- | --- | --- |
| Ordinary evaluation | A held-out split or a separate evaluation dataset | The trainer's loss and available method-specific metrics, including a step-zero baseline. |
| Synthetic CPT evaluation | One generated summary per primary `text` row | Ordinary evaluation on summaries while all original rows train; these summaries derive from training material. |
| Independent quality checks | A separate benchmark dataset and a built-in preset | Task scores such as QA accuracy, JSON validity, or judged writing quality, with individual evidence. |
| Checkpoint verification | Saved checkpoint and base model | Whether the artifacts load and pass integrity/template checks. |

All seven trainers provide a full pre-run scan. Startup findings distinguish
measurements, preparation projections, and heuristics, and include their evidence. They never change
training settings or select a checkpoint. Complete pre-run evidence is in `assessment.json`;
training metrics remain in `metrics.jsonl`. Startup shows only problems, including disabled evaluation.
Every evaluation and completion shows factual metric comparisons and training/evaluation loss charts,
without assessment prose or tuning advice. The two charts occupy 100 columns with 13 plot rows;
each uses its own observed Y range and the complete run through the current optimizer step, including
resume history. Missing data is explicit; non-finite measurements break chart lines.
Intermediate reports compare previous and latest values; final reports compare initial and latest values.
Training starts at the first logged loss, not the whole-run average. Accuracy changes use percentage points.
Reports appear inline and in `log.txt` and add no evaluation passes. Ordinary terminal output omits rank
prefixes; explicit diagnostics and log lines retain source attribution.
Line-mode metric tables retain their column headers, use `Change` for differences, and omit repeated legend sentences.
They show `training_loss`, the latest measured `eval_loss`, and its `eval_step` together, including on
resume and in `trlx show`. Blank evaluation fields mean no evaluation has been recorded yet.

When ordinary evaluation is enabled, fresh runs measure the starting model at step zero using the
same evaluation data as later evaluations. This adds one evaluation pass; resume keeps the original
baseline. No extra flag or quality benchmark is needed. Final evaluation comparisons use that baseline;
missing baselines and measurements before the final step are identified by their recorded steps.

Training and `check` require the following explicit block, supplied by new `trlx init` configurations.
Add it to older configs; initialization with `--force` replaces the entire file.

```toml
[assessment]
quality_checks = false
quality_preset = "None"
quality_dataset = "None"
quality_max_length = 2048
quality_max_new_tokens = 256
quality_batch_size = 1
```

Evaluation frequency controls metric-report timing; reports use the recorded history without
minimum observation counts. Ordinary evaluation and metric reports do not require `--quality-checks`.

Independent checks are optional and require a separate evaluation dataset and a built-in preset:

```sh
trlx sft --model MODEL --dataset DATA --quality-checks \
  --quality-preset qa --quality-dataset HELDOUT
```

`HELDOUT` is a supported dataset file or Hub reference with the columns below. `prompt` accepts text
or text messages; generative presets also accept `messages` instead. No custom scorer is needed;
generation and judge presets use the editable prompt files created by `init`.

| Preset | Evaluation rows | Reported evidence |
| --- | --- | --- |
| `language_modeling` | `text`, `messages`, or `prompt` + `completion` | Full-sequence loss and token-weighted perplexity |
| `qa` | `prompt` + `answer` string or `answers` list | Normalized exact match and whitespace-token F1 |
| `classification` | `prompt`, `labels` list, correct `label` | Accuracy, per-class counts/results, invalid/ambiguous answers |
| `multiple_choice` | `prompt`, `choices` mapping labels to text, correct `answer` label | Accuracy, per-class results, invalid/ambiguous answers |
| `json` | `prompt`; optional `required_fields` list and `reference` object | JSON validity, required fields, supplied reference-value correctness |
| `preference` | `chosen`, `rejected`; optional `prompt` | Reward-model ranking accuracy, ties, and margins; reward trainer only |
| `instruction_following`, `writing` | `prompt` | File-defined rubric ratings and rationale from a configured judge |

QA normalization uses Unicode normalization, case folding, punctuation boundaries, and whitespace.
Classification requires a complete permitted label; multiple choice also permits label-only forms such
as `(A)` or `Answer: A`. JSON parsing requires a complete JSON response, with no duplicate keys or
non-finite numbers; supplied reference fields must match recursively. Extra top-level fields are allowed.
Generation adds preset format instructions and supplied choices/labels, never the expected answer.
These text presets do not assess image/audio content. Repetition, empty outputs, and token-limit cutoffs
are diagnostics, not universal quality scores.

Checks run at baseline, ordinary evaluation points, and completion—even with ordinary evaluation disabled.
All evaluation rows are used. `--quality-max-length` controls input/window size; LM uses one-token-overlap
windows, while generation/preference inputs exceeding the limit fail visibly instead of being truncated.
`--quality-max-new-tokens` bounds greedy generation and `--quality-batch-size` controls generation batches.
Individual observations and failed-check evidence are retained in `quality.jsonl`; aggregate metrics use
the `quality/` prefix in `metrics.jsonl`. Scorer failures remain unavailable evidence, not zero scores.
Resume reuses a baseline only for matching data, tokenizer semantics, scorer, and evaluation settings.

Judging presets additionally require `[assessment.judge]` with explicit `url`, `model`, `api_key`,
`timeout`, `retries`, and `max_tokens`, or their `--quality-judge-*` CLI forms. `api_key` names a credential
environment variable; use `"None"` for an unauthenticated endpoint. Timeout must be positive, retries
nonnegative, and the response token budget positive. Judge ratings are model judgments, not ground truth.
Quality checks add inference/scoring work; `--no-quality-checks` disables them for one run.

## Tutorial: temporary and persistent settings

Settings resolve in this order: **CLI > selected method section > shared config**.
For example, `[methods.sft]` applies to SFT only; shared settings apply across methods.
Training saves its resolved settings with the run and never rewrites `run.toml`.

```sh
# Try a lower learning rate and three epochs for one new run.
trlx sft --model MODEL --dataset DATA \
  --learning-rate 5e-5 --num-train-epochs 3 --output-dir runs/three-epochs

# Use another persistent config; NEW_DIR must be an existing empty directory.
trlx init --out NEW_DIR/experiment.toml
trlx sft --config NEW_DIR/experiment.toml --model MODEL --dataset DATA

# Booleans have positive/negative forms; lists and tables use shell-quoted TOML.
trlx sft --model MODEL --dataset DATA --no-verify
trlx sft --model MODEL --dataset DATA --lora-target-modules '["MODULE_A","MODULE_B"]'

# Nullable booleans also accept None to clear a configured true/false value.
trlx sft --model MODEL --dataset DATA --tf32 None
```

Replace `MODULE_A` and `MODULE_B` with module names from your model.

To persist settings, edit the existing entries in the generated file. This excerpt
shows where they belong; keep the other generated settings and avoid duplicate keys.

```toml
# Shared trainer settings go before any [section] heading.
num_train_epochs = 3                     # Passes through the training data.
max_steps = -1                           # A positive value overrides the epoch count.
per_device_train_batch_size = 1          # Examples processed at once on each GPU.
gradient_accumulation_steps = 8          # Batches accumulated before a weight update.

[model]
path = "models/base"                     # Optional: avoids repeating --model.

[dataset]
split = true
dataset = "data.jsonl"                   # Optional: avoids repeating --dataset.
eval_fraction = 0.1                     # Fraction held out for evaluation.

[methods.sft]
output_dir = "runs/three-epochs"
learning_rate = 5e-5                     # Update scale; larger values change weights faster.
```

With model and dataset saved, `trlx sft` is sufficient. Trainer fields use their TRL
names in TOML and hyphenated names on the CLI. Omitted trainer fields use TRL defaults;
unknown keys are errors. Nullable trainer and LoRA fields accept the string `"None"`.

Other useful sections: `[run]` controls `gpus`, `strategy`, `tui`, and `verify`;
`[peft]` controls LoRA. `[methods.METHOD.ranges]` selects displayed metrics and expected
intervals. DPO/KTO `preflight` sections control off-policy warnings. Distillation's
`teacher` section holds its teacher model. Verification checks loading, adapter integrity, and chat-template
consistency. Prompt comparison is removed; delete `[verify].prompts`, `--verify-prompts`, or `verify --prompts` if present.

## Tutorial: evaluation and data sources

The generated `eval_fraction = 0.1` reserves the final 10% of rows for evaluation.
The count is rounded up when loading the actual data; both partitions must remain
nonempty. Evaluation measures held-out performance without updating model weights.
Add `--shuffle-eval-data` to select the same number of evaluation rows randomly from
the whole source, excluding them from training. Both sets retain their original relative
row order. The selection uses `--data-seed` when set, otherwise `--seed`, and repeats for
the same source and seed. This option requires percentage-split mode and works for all trainers.
To persist it, set `shuffle_eval_data = true` in `[dataset]`.

```sh
# Change the held-out share for one run.
trlx sft --model MODEL --dataset DATA --eval-fraction 0.2

# Randomly hold out the same share instead of taking rows from the end.
trlx sft --model MODEL --dataset DATA --eval-fraction 0.2 --shuffle-eval-data

# Use separate training/evaluation files.
trlx sft --model MODEL --dataset-train train.jsonl --dataset-eval eval.jsonl

# Train on all rows, without evaluation.
trlx sft --model MODEL --no-split --dataset data.jsonl
```

Either `--dataset-train` or `--dataset-eval` selects separate-file mode, overriding
the configured split default and removing inherited split-only settings. Explicit
`--split` conflicts with these options. Training-only mode also removes the configured
evaluation schedule when no separate or synthetic evaluation source is available.

To persist separate files, use `split = false` with `dataset_train` and optional
`dataset_eval` in `[dataset]`; remove `dataset` and `eval_fraction`.
Without a separate or synthetic evaluation dataset, also remove top-level `eval_*`
settings. The CLI's `--no-split` handles that removal automatically.
The old `train = N` setting is rejected. Dataset IDs use `org/name:split`;
a dataset with multiple splits requires an explicit split name.

For SFT with CPT `text` rows, generate evaluation summaries with the model already
loaded for training:

```sh
trlx sft --model MODEL --dataset train.jsonl --synthetic-dataset-eval --max-length 1024
```

This trains on all primary source rows and generates one factual prose summary per
row before the step-zero evaluation. The positive `--max-length` value also limits
generated tokens per summary. The flag replaces configured splitting or evaluation
sources; explicit `--split`, `--eval-fraction`, and `--dataset-eval` conflict with it.
The ordinary evaluation schedule is retained. Startup skips evaluation-data inspection;
`trlx check sft` validates the setup without generating summaries.

Summaries are saved only in this run as `synthetic-eval.jsonl` and reused unchanged
on resume. A missing file prevents resume before rewind. Replay rows are not summarized.
Generation failures stop training; source prompts are not silently truncated. Model
response templates separate reasoning from summary content; unparsed tagged responses
are errors. No endpoint, separate model load, or cross-run cache is involved.

## Tutorial: LoRA, memory, and GPUs

LoRA is enabled by default: rank 8, alpha 16, dropout 0.05, and `all-linear` targets.
It trains small adapters instead of all base weights. Higher rank adds trainable
capacity and memory use. `all-linear` may include vision layers in multimodal models.

```sh
trlx sft --model MODEL --dataset DATA --lora-r 16 --lora-alpha 32
trlx sft --model MODEL --dataset DATA --no-lora
trlx sft --model MODEL --dataset DATA --gpus 0
trlx sft --model MODEL --dataset DATA --gpus 0,1 --strategy fsdp
```

`init` detects the visible hardware; it does not inspect models/data or run calibration.
It selects native BF16 when every visible GPU supports it, otherwise FP32. Small
batches and gradient checkpointing are conservative starting settings, not a fit or
throughput guarantee. Default training uses one example per GPU and accumulates eight
batches per update. Nominal effective batch = per-GPU batch x accumulation x GPU count.

Training uses all visible GPUs by default and estimates whether to use data parallelism
or sharding. `--strategy auto` retains that choice; `ddp` or `fsdp` forces a strategy
and requires multiple GPUs. `--gpus` indexes the currently visible devices.
No separate Accelerate configuration or launcher is needed.

`check` loads the model on the first selected GPU, so it must fit there. Larger
sharded runs receive preflight during training. If moving the config to different
hardware, review precision settings or run `init --out NEW_CONFIG` on that host.

### Understanding failures

Failures report the original cause, the operation in progress, and relevant evidence. Training errors
identify the rank and visible GPU; batch failures include available shapes and token counts. OOM
reports add parameter dtypes, allocator memory, and measured padding overhead. Allocator counters
collected after an exception are labelled separately from the original allocation-failure figures.
Unavailable measurements are identified rather than estimated.

Training `log.txt` retains the complete evidence, exception chain, and traceback. Terminal reports
lead with the failure, explain relevant facts, and summarize successful cleanup in one line. Internal
field trees and routine peer-abort messages stay in the log; cleanup problems remain visible. Startup and standalone
commands report on stderr, including a traceback for unexpected errors. Reports exclude tensor
contents and frame locals and retain credential redaction.

### Recovering stuck GPU activity

If a failed run has exited but a GPU still reports high utilization with no compute
processes, [recover_gpu.py](recover_gpu.py) can restore idle operation by running a
tiny CUDA kernel, synchronizing, and explicitly destroying its temporary context.
It requires Python and an NVIDIA driver with an Ampere-or-newer GPU; it does not
reset hardware or stop monitoring services.

From the repository root, identify the affected GPU's PCI bus ID:

```sh
nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
```

Replace `PCI_BUS_ID` below with that address (for example, `0000:21:00.0`), then
verify utilization returns to idle:

```sh
timeout --signal=TERM --kill-after=5s 45s python -u recover_gpu.py PCI_BUS_ID
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,pstate --format=csv
```

## Tutorial: checkpoints, resume, and results

The defaults evaluate and save at each epoch's end, retain two checkpoints, and log
every optimizer update. Generated configs omit explicit save strategy/interval settings:
saving follows the evaluation schedule, or saves only at completion when evaluation
is disabled. Set `--eval-strategy steps --eval-steps N` for evaluation and checkpoints
every N optimizer steps, where N is a positive integer. Explicit `--save-strategy` and
`--save-steps` settings override this coupling, including settings retained in an older
config. `save_strategy = "no"` is rejected.

Every fresh run uses `output_dir/YYYYMMDD-N--model--dataset/`. The number advances across
all models and datasets under that parent for the local date, starting above existing numbers.
Model IDs are lowercased with `/` replaced by `-`; local models use their directory name.
Dataset files use their filename stem. `run_name` changes only the display label.

```sh
# RUN_DIR is the generated directory printed at startup; select an existing checkpoint.
trlx sft --resume-from-checkpoint RUN_DIR/checkpoint-100

# Verify saved artifacts against the base, or merge the LoRA adapter.
trlx verify RUN_DIR/checkpoint-100 --base MODEL
trlx merge --base MODEL --adapter RUN_DIR/checkpoint-100 --out merged-model
```

Resume loads the run's saved `config.toml`, not today's `run.toml`; model and dataset
arguments need not be repeated. Explicit CLI overrides still apply. Only the current
snapshot schema is supported. Training settings must match; GPU selection and display
controls may change, but switching between sharded and unsharded training is rejected.
Assessment settings may also change; independent quality baselines are reused only
when their data, tokenizer, scorer, prompts, and evaluation conditions match.
Active prompts are copied into the run's `prompts/` directory and referenced by its
saved config. Workers and resume use those copies. Missing required copies fail before
resume cleanup; changes to quality prompt contents invalidate baseline reuse.

Resume continues in the original directory. After validation it automatically removes
metrics, checkpoints, and quality rounds beyond the selected saved step and clears stale
assessment, preflight, and verification reports. No `--force` is needed. Earlier metrics
remain; logs append a resume marker and new output. The live line display prints the
continuation; `trlx show RUN_DIR` includes
the retained history. `trlx check` validates resume inputs without performing cleanup.

Each run retains the inputs and evidence needed to inspect its results:

| Artifact | Contents |
| --- | --- |
| `config.toml` | Resolved method, trainer settings, selected GPUs, and launch strategy. |
| `prompts/` | Copies of active prompt files referenced by the snapshot. |
| `metrics.jsonl` | Authoritative training/evaluation history and aggregate quality metrics. |
| `log.txt` | Complete operational and library diagnostics with source attribution. |
| `assessment.json` | Pre-run data inspection and tuning findings with evidence. |
| `preflight.json` | Model/data checks, resolved LoRA targets, warnings, and prepared-example details. |
| `quality.jsonl` | Individual quality-check results and partial failure evidence when enabled. |
| `synthetic-eval.jsonl` | Generated CPT summaries when enabled; retained for evaluation and resume. |
| `checkpoint-N/` | Trainer checkpoint at the recorded optimizer step. |
| `verify.json` | Artifact loading, adapter-integrity, and chat-template checks when verification runs. |

`trlx merge` loads the base using its model metadata, checks the loaded adapter, and
writes a merged model to `--out`. An existing destination requires `--force`; replacing
the base or adapter requires staging. Keep the original run to retain its metrics and
diagnostics alongside the exported model.

`--tui` enables the full-screen display; `--no-tui` prints lines. Press `q` to close
the display while training continues; Ctrl-C cancels. Display failures are recorded
in `log.txt` and stop presentation without abandoning supervision.

## Tutorial: rewards

GRPO/RLOO require a reward objective and a TRL-compatible vLLM weight-transfer server.
Install vLLM separately and run `trl vllm-serve --model MODEL` on dedicated GPUs.
Use `CUDA_VISIBLE_DEVICES` to select the server GPUs and exclude them from training.
For a task whose objective is producing valid JSON:

```sh
trlx grpo --model MODEL --dataset DATA --reward json_valid \
  --vllm-server-base-url http://localhost:8000
trlx rloo --model MODEL --dataset DATA --reward json_valid \
  --vllm-server-base-url http://localhost:8000
```

Choose a reward suited to your task. Repeat `--reward` to combine rewards; doing so
replaces the configured list. Factories accept a shell-quoted TOML table:

```sh
trlx grpo --model MODEL --dataset DATA \
  --reward '{name="reference_match",args={column="answer",mode="equals"}}' \
  --vllm-server-base-url http://localhost:8000
```

| Built-in reward | Arguments                                                                       |
| --------------- | ------------------------------------------------------------------------------- |
| reference_match | column, mode: equals / contains / fuzzy; fuzzy also needs threshold (0..1)      |
| regex           | pattern; optional group and column together                                     |
| phrases         | required and/or forbidden: lists of phrases                                     |
| json_valid      | Optional keys: list of required JSON keys                                       |
| length_window   | unit: words / tokens, low, high; tokens also needs tokenizer                    |
| llm_judge       | url, model, rubric_file, timeout, retries, concurrency; optional api_key, max_tokens |

`llm_judge` requires your own `.prompt` file. `prompts/llm-judge.prompt.example`
contains authoring guidance and example rubrics; it is never loaded automatically.
Set `rubric_file` to your authored file, relative to the config directory (absolute
paths also work). The former inline `rubric` argument is rejected.

Rewards may also name a `trl.rewards` function, reward model, or
`module:function` / `path.py:function`. `--reward-weights '[1.0,0.5]'` weights two entries
in order. Persist factories under the selected method:

```toml
[methods.grpo.rewards]
funcs = [{name = "reference_match", args = {column = "answer", mode = "equals"}}]
```

## Tutorial: SFT replay

Replay mixes examples of the original behavior into training to help retain it.
Generate `replay.jsonl` with `replay-build`, then supply:

```sh
trlx sft --model MODEL --dataset DATA \
  --replay-dataset replay.jsonl --replay-fraction 0.2 --replay-kl-coef 0
```

The fraction requests a share of the final training mixture, strictly between 0 and 1.
Rounding and a minimum of one replay row can change that share for small datasets.
Replay must match training columns and never enters evaluation. `--no-replay` disables
configured replay for one run. Persist these values in `[methods.sft.replay]` as
`dataset`, `fraction`, and `kl_coef`.

A positive `kl_coef` additionally penalizes drift from the original model. It requires
omitting `loss_type` and disabling `use_liger_kernel`, `packing`, and `padding_free`.
Add `replay_kl` to `[methods.sft.ranges]` to display that metric.

## Interactive dataset authoring

From the repository root, start the FastAPI interface:

```sh
python -m dataset.cli ui --host 127.0.0.1 --port 8000
```

Both arguments are required; `--port` accepts 1..65535. Open the chosen address
in your browser. Deployment and access controls are the operator's responsibility.

Enter a user prompt and optional system prompt, then configure at least two
output cards with their own endpoint URL, model, and optional API key. Enable
only the sampling overrides you want sent. Each card has editable timeout and
retry settings, initially 120 seconds and 2 retries. Generate compares independent
responses; reasoning and answers can both be edited. Prompts remain populated,
with separate clear buttons for user, system, or both.

Choose reasoning, answer, or both and use each card's **Add to dataset** button.
Any number of candidates can be added. The user is always saved; the system
prompt is excluded. Excluding reasoning omits its field; excluding the answer
leaves an empty assistant message. Adding captures the current user prompt, including
edits made after generation. Later edits do not change examples already pending.

The pending collection, prompts, edited responses, endpoint settings, and API keys
persist in this browser's localStorage for the same origin. **Save dataset** appends
unique examples to the selected `.jsonl` path on the server. Relative paths use the
server's working directory; the parent must exist. Existing contents are preserved
through staged publication. Exact duplicates include reasoning presence and text.
Successful saves clear submitted examples; failures retain them for retry. Use one
UI process for a destination and avoid concurrent writes from other applications.

### Reusable Context builder

`dataset context` incrementally builds the exact messages array accepted by
`generate --context-file`. The JSON file is the durable state; no separate
workspace or export step exists.

```sh
dataset context create security-context.json

# Add ordinary messages from stdin, a file, or short inline text.
dataset context add security-context.json --role user < research-question.txt
dataset context add security-context.json --role assistant --content-file analysis.txt
dataset context add security-context.json --role user --text "Apply those findings."

# Represent supplied data as fabricated retrieval without executing that tool.
dataset context tool security-context.json \
  --name web_search \
  --arg 'query=JWT verification requirements' \
  --content-file search-results.txt
dataset context tool security-context.json \
  --name read_file \
  --arg 'path=manual.md' \
  --content-file manual.md

# Inspect structure without printing the full content, then validate it.
dataset context outline security-context.json
dataset context show security-context.json 3 --count 2
dataset context validate security-context.json
```

Content ingestion and message representation are independent: a result read
from any local file or stdin may be represented using any caller-selected tool
name and arguments. The builder only creates messages; it does not fetch URLs,
run commands, query services, or generate responses. Mutations use staged
replacement and must be serialized for one Context file. See
[SPEC-context-packages.md](SPEC-context-packages.md) for insertion, replacement,
range editing, raw-message, validation, and formatting contracts.

### Stateless CLI authoring

Use `generate` and `save` independently of the web server. Each reads one JSON
object from stdin and returns JSON on stdout; progress/errors go to stderr.
Keep those streams separate when capturing responses. Run these commands from the
repository root with your project environment's Python executable:

```sh
python -m dataset.cli generate --context-file context.json <<'JSON'
{"endpoint":"http://localhost:8000/v1","model":"served-model","user":"Explain this.","system":"","sampling":{},"timeout":3600,"retries":0}
JSON
```

Replace endpoint/model with your served model. `--context-file` is optional;
when supplied it must name a UTF-8 JSON array of message objects. Its entries
are sent unchanged between system and user. Relative paths use the working
directory. Context is not accepted in stdin JSON.

`endpoint`, `model`, `user`, `sampling`, `timeout`, and `retries` are required.
`system` is optional and defaults to empty. For authentication, add
`"api_key":"ENVIRONMENT_VARIABLE"`; the CLI resolves it from the environment or
`.env`, with existing environment values taking precedence. Omission means no
authentication. `sampling: {}` sends no overrides. The example allows one hour
and disables automatic retries so long reasoning runs can finish. Supported
controls and constraints are in `python -m dataset.cli generate --help`.

Generation returns `{"answer":"...","reasoning":"..."}` without saving it.
Repeat for other candidates, edit/select responses, then explicitly save:

```sh
python -m dataset.cli save <<'JSON'
{"path":"examples.jsonl","examples":[{"messages":[{"role":"user","content":"Explain this."},{"role":"assistant","content":"Edited answer"}],"reasoning":"Edited reasoning"}]}
JSON
```

Store reasoning as a separate top-level string, without manually added reasoning
delimiters. Omit `reasoning` for answer-only examples; use empty assistant content
for reasoning-only examples. Save accepts exactly one user turn followed by one
assistant turn, without system or Context, and returns `{"added":N,"duplicates":N}`.
Exact duplicates include reasoning presence and text. Existing bytes are preserved
through staged publication; no `--force` is needed. The destination parent must
exist. Run saves to the same destination sequentially, including browser saves.
The CLI retains no workspace or pending collection.

## Agent-assisted dataset authoring

The authoring workflow produces supervised examples in the target model's own
voice: each row is a self-contained user prompt, the model's reasoning, and its
answer, generated by the target model and repaired only where defective. An
agent runs the workflow by following [DATASET-AUTHORING.md](DATASET-AUTHORING.md);
invoke that file explicitly to start a session. The agent reads the saved
session settings from `dataset-authoring.toml` (endpoint, model, destination,
full-sequence token limit, and the tokenizer and scoring probes for the model),
confirms them, and asks for the topic, the batch size, the likelihood acceptance
policy, whether to use Context-assisted generation, and how to dispatch the
reviewer. Approval of the scenario list then authorizes the whole batch without
per-row approval.

### How a row is made

1. **Generate.** The agent sends the prompt, and optionally a Context, through
   `dataset generate` and keeps the untouched response.
2. **Edit under contract.** Only defects are repaired: false claims, grammar,
   contradictions, drafting residue, and references to material that will not be
   in the saved row. Edits are exact literal replacements applied once to the
   untouched text; sound reasoning is preserved verbatim.
3. **Validate.** The complete example is rendered with the training tokenizer
   and template and counted against the limit. The reasoning and answer spans
   are scored for mean log-probability before and after editing, including a
   fixed-context comparison that isolates answer wording changes; candidates
   must stay within the agreed threshold.
4. **Review.** An adversarial reviewer compares the untouched and edited
   reasoning against the contract and rules `ACCEPT` or `REVISE`. The ruling is
   binding.
5. **Save and verify.** `dataset save` appends the row; the agent confirms the
   saved fields match the scored candidate byte for byte and that earlier rows
   are unchanged.

When an edited example still exceeds the token limit, the "Fitting the token
limit" rule in DATASET-AUTHORING.md applies: regenerate with a better Context or a
narrower prompt when the reasoning does not demonstrate understanding or the
prompt asked too much; otherwise cut the answer from the end, never good
reasoning.

### Context-assisted generation

Rather than packing facts into the saved prompt, the agent builds a Context: a
research conversation, in the exact messages format `generate --context-file`
accepts, in which user questions motivate retrieval, fabricated tool calls return
verified source passages, and assistant turns analyze them. The Context teaches
the target model the topic so that its own reasoning demonstrates understanding;
the Context itself is excluded from the saved row. Three gates precede
generation: a provenance gate (every tool result comes from a retrieval or
execution that actually happened, with its location recorded), a printed
sufficiency challenge (the case for, the case against, and a `READY` or
`NOT READY` adjudication), and a rendering check that the Context, the prompt,
and the generation reserve fit the model's context window. Saved Contexts live in
`contexts/NAME.json` with a `contexts/NAME.txt` description and are reused across
scenarios on the same topic.

### Authoring tools

`authoring/` holds the scripts that perform the mechanical steps. They read
`dataset-authoring.toml`; source extraction needs the `authoring` extra:

```sh
python -m pip install '.[authoring]'
```

| Script | Step | Example |
| --- | --- | --- |
| `extract.py` | Structural excerpts from RFC HTML, HTML ids, definition entries, clauses minus subclauses, Markdown headings, and PDF pages | `python authoring/extract.py rfchtml rfc6960.html out.txt section-2.2 section-3.2` |
| `reuse.py` | Copy a hash-verified tool result out of an existing Context for reuse | `python authoring/reuse.py contexts/TOPIC.json 6 out.txt` |
| `ctxbuild.py` | Build a Context from a script, with provenance headers | imported by a build script; see its docstring |
| `render_check.py` | Mandatory rendering and budget check | `python authoring/render_check.py contexts/TOPIC.json prompt.txt` |
| `gen.py` | One generation with the saved settings | `python authoring/gen.py --prompt-file prompt.txt --context-file contexts/TOPIC.json --out-dir gen --label row1` |
| `count_score.py` | Token count and span log-probabilities | `python authoring/count_score.py prompt.txt reasoning.txt answer.txt --score` |
| `edit.py` | Exact literal edits with uniqueness checks and a diff; `derive` builds the edit list from an edited copy | `python authoring/edit.py gen/row1.out edits.json edited/` |
| `make_review_prompt.py` | Assemble the reviewer prompt | `python authoring/make_review_prompt.py gen/row1.out edited/ justifications.txt review.prompt` |
| `review.sh` | Run the tool-call reviewer and read its ruling | `authoring/review.sh review.prompt review1` |

Each script's `--help` documents its inputs, and `tests/test_authoring.py`
covers their silent-failure points. Token counting and likelihood scoring use the
endpoint's tokenizer and completions API as recorded in `[probes]`; they apply to
training only when that rendering matches the training template. See
[SFT reasoning supervision](#sft-reasoning-supervision) for training on the
saved reasoning field.

Independent generations can run concurrently within the endpoint's capacity;
the agent launches separate `generate` requests, and concurrency is orchestrated
by the caller, not a batch option on `generate`. Keep saves to the same
destination sequential, including saves from other processes or the browser.

## Tutorial: endpoint generation and credentials

`ENDPOINT` is an OpenAI-compatible API base URL, including `/v1` when required;
`MODEL` is the served model name. CLI commands other than `dataset ui` load optional `KEY=value` entries from
`.env` in the working directory. Exported environment values take precedence.
`--api-key API_KEY` names the variable holding the key; it never takes the secret value.
Reward factories use `api_key = "API_KEY"`. Omit the option for unauthenticated servers.

```sh
dataset chat source.txt --out chat.jsonl \
  --questions-endpoint ENDPOINT --questions-model MODEL \
  --n 3 --max-tokens 1024 --concurrency 4 --timeout 120 --retries 2 \
  --api-key API_KEY
```

This generates questions from chunks, then answers each question using its source.
Exact duplicate questions are always removed across the run after list-marker and
surrounding-whitespace cleanup, before answer requests. The first occurrence is kept;
each removal is reported. Case, internal whitespace, and punctuation remain significant.
Here `--max-tokens` limits input chunks; the server controls answer length. Provide
both `--answers-endpoint` and `--answers-model` to use a different answer model.
`--questions-prompt FILE` defaults to `./prompts/chat-questions.prompt` and requires
`{chunk}` and `{n}`; `--answers-prompt FILE` defaults to `./prompts/chat-answers.prompt`
and requires `{chunk}` and `{question}`. Paths are relative to the working directory.

Output contains `messages` and separate `reasoning` by default. `--exclude-reasoning`
omits the reasoning column; it does not disable model reasoning. Configure the server's reasoning
parser; inline reasoning is rejected unless `--strip-reasoning-tags` removes a complete
leading block. An unclosed block is always an error. The two flags are independent;
use both to strip inline tags and omit the separate reasoning column.

To reserve one of ten questions per chunk for evaluation, add
`--n 10 --eval-n 1 --out training.jsonl --eval-out eval.jsonl` to `dataset chat`.
`--eval-n` defaults to zero; a positive value must be smaller than `--n` and requires
`--eval-out`. After deduplication, the last requested number of retained questions
per chunk goes to evaluation and the rest to training. Short chunks contribute up
to that number; skipped answers reduce counts without reassigning questions.
Both destinations are validated before requests and staged before either is published
unless `--no-staging` is set. Existing outputs require `--force`.

For a synthetic CPT evaluation dataset:

```sh
dataset eval-build train.jsonl --out eval.jsonl \
  --endpoint ENDPOINT --model MODEL --max-tokens 1024

trlx sft --model MODEL --no-split \
  --dataset train.jsonl --dataset-eval eval.jsonl \
  --eval-strategy steps --eval-steps 5
```

For best results, generate summaries using the same model you'll use this data set to train.

Each input row must contain nonempty string `text`; output contains one factual prose
summary in `text` per input row, in the same order. JSONL, JSON, CSV, and Parquet are
supported. `--max-tokens` is a positive completion-token limit, not a source chunk limit.
`--summary-prompt FILE` defaults to `./prompts/eval-build-summary.prompt`; its contents
are the system message, with each source row supplied separately as the user message.
Train on all original chunks; generate summaries once and reuse them for the step-zero
baseline and subsequent ordinary evaluations through `--dataset-eval`.

Inputs and destination are validated before requests. Empty or malformed replies and
any `finish_reason` other than `"stop"` (including missing metadata) fail with the source
row number, without publishing an incomplete dataset. Separate reasoning is excluded;
inline reasoning follows the same `--strip-reasoning-tags` policy as `chat`.
Existing outputs require `--force`; `--no-staging` controls publication only.

`dataset chat` and `dataset eval-build` share optional request settings:
`--concurrency` defaults to 4 (positive integer), `--timeout` to 120 seconds (finite and
positive), and `--retries` to 2 (nonnegative integer). Both accept `--api-key ENVVAR`.
Retries apply per request to transport failures, retryable HTTP statuses, and malformed
response encoding, JSON, or message fields. Successful requests remain in memory and
are not repeated. Retry notices name the failed request, cause, attempt, and backoff.
Exhausting the retry budget still fails the batch; results are not saved for a later run.
Explicit incomplete-generation results remain errors rather than automatic retries.
Waiting notices identify every active request by number, elapsed request/attempt time,
retry attempt, and whether it is awaiting the endpoint or backing off. They report queued
requests and completed responses retained in memory. Server-side progress is unavailable;
the socket timeout is not a total request deadline.

For replay generation through an endpoint:

```sh
trlx replay-build --model MODEL --endpoint ENDPOINT \
  --prompts prompts.jsonl --out replay.jsonl --max-tokens 256 \
  --timeout 120 --retries 2 --concurrency 4 --api-key API_KEY
```

Prompts need `prompt` or `messages`. Here `--max-tokens` limits the completion.
Endpoint mode requires `--timeout` (seconds per request), `--retries` (after the first
attempt), and `--concurrency` (simultaneous requests). Without `--endpoint`, generation
is local and endpoint-only options are rejected.

## Prompt files

`trlx init` copies these editable UTF-8 files into `prompts/` beside `--out`.
Training paths in `[prompts]` are relative to the config directory; dataset CLI
paths are relative to the working directory. Absolute paths are accepted.

| File | Training config key / dataset option | Required placeholders |
| --- | --- | --- |
| `chat-questions.prompt` | `--questions-prompt` | `{chunk}`, `{n}` |
| `chat-answers.prompt` | `--answers-prompt` | `{chunk}`, `{question}` |
| `eval-build-summary.prompt` | `--summary-prompt` | None |
| `synthetic-eval-summary.prompt` | `synthetic_eval_summary` | `{text}` |
| `quality-qa.prompt` | `quality_qa` | None |
| `quality-classification.prompt` | `quality_classification` | `{labels}` |
| `quality-multiple-choice.prompt` | `quality_multiple_choice` | `{choices}` |
| `quality-json.prompt` | `quality_json` | `{required_fields}` |
| `quality-instruction-following-judge.prompt` | `quality_instruction_following_judge` | None |
| `quality-writing-judge.prompt` | `quality_writing_judge` | None |

For example, `[prompts]` contains
`quality_qa = "prompts/quality-qa.prompt"`. Only enabled features require their
files. Missing, unreadable, empty, invalid UTF-8, or invalid templates fail before
requests or model loading. Installed templates are used only by `init`; runtime
never falls back to them. The additional `llm-judge.prompt.example` is guidance,
not an active prompt.

Substitution is one pass: inserted source text is never interpreted as a template.
`{labels}` is a JSON list; `{choices}` is one `label: text` line per choice.
The JSON preset uses `[[ The JSON object must contain these top-level keys: {required_fields}.]]`:
the optional block is omitted when the row has no required fields, otherwise the
placeholder receives their JSON list. Keep each template's required placeholders.

For an existing configuration, create a fresh scratch directory and run
`trlx init --out SCRATCH/run.toml`, where `SCRATCH` already exists and contains
neither that config nor generated prompt files. Copy the needed prompt files and
`[prompts]` entries into your existing setup; author a separate judge rubric if
using `llm_judge`. Existing runs need their required prompt files and saved paths
before they can resume.

## Further reference

Run `trlx COMMAND --help` or `dataset COMMAND --help` for complete option references.
With dependencies installed, you can use the checkout directly:
`python -m trlx.cli` and `python -m dataset.cli` replace the installed command names.

[SPEC.md](SPEC.md) describes the behavior; [PLAN.md](PLAN.md#second-pass-review-of-phases-4-and-5)
tracks known issues, including reward scoring, distillation memory estimates, and display limitations.
