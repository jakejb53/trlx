# trlx

TRL training without bespoke Python or a setup worksheet. `trlx` handles configuration,
GPU launch, live metrics, and verification; `dataset` prepares the training data.

## Start here

Activate the Python environment you intend to use. It needs Python 3.11+ and a
PyTorch build suitable for your CUDA setup; training needs a visible CUDA GPU.
From the repository root, install and start:

```sh
python -m pip install .
trlx init
trlx sft --model MODEL --dataset DATA
```

If `run.toml` already exists, `trlx init --force` replaces it with fresh defaults
for this environment, discarding saved edits. Without `--force`, the file is preserved.
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

Both tools report their current operation, completed counts when available, retries,
and final outcome to stderr. After ten seconds without feedback, a waiting notice
shows elapsed time and the last measured progress; it does not claim work is advancing.
Training also records feedback in `log.txt`, visible in the TUI's log pane.

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
trlx verify CHECKPOINT --base BASE --prompts prompts.jsonl
trlx merge --base BASE --adapter ADAPTER --out merged-model

# Generate replay data locally.
trlx replay-build --model MODEL --prompts prompts.jsonl --out replay.jsonl --max-tokens 256

# Discover method-specific options and utility commands.
trlx check dpo --help
trlx replay-build --help
```

Uppercase names are placeholders; replace them with your own inputs. Paths are
relative to the working directory. Every command accepts `--force`. Without it,
destructive operations are refused with their consequences and the flag needed to proceed.
With it, the requested replacement proceeds without another confirmation, including replacing
inputs or directories containing unrelated files. Destructive replacement replaces symlinks
themselves and leaves their targets alone; healing follows symlinks to repair their targets.

| Command                  | Replacement authorized by --force                         |
| ------------------------ | --------------------------------------------------------- |
| trlx init                | Existing config path, replaced with fresh defaults        |
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

## Settings assessment and independent quality checks

All seven trainers provide a full pre-run scan and runtime recommendations. Findings distinguish
measurements, preparation projections, and heuristics, and include their evidence. They never change
training settings, stop a run, or select a checkpoint. Complete pre-run evidence is in `assessment.json`;
runtime notices appear inline and in the TUI log pane. Training metrics remain in `metrics.jsonl`.

Training and `check` require the following explicit block, supplied by new `trlx init` configurations.
Add it to older configs; initialization with `--force` replaces the entire file.

```toml
[assessment]
quality_checks = false
runtime_window = 20
runtime_min_evaluations = 3
runtime_relative_change = 0.05
quality_preset = "None"
quality_dataset = "None"
quality_max_length = 2048
quality_max_new_tokens = 256
quality_batch_size = 1
```

The runtime window counts logged observations. The relative-change threshold is heuristic sensitivity,
not a confidence level. Override these with `--assessment-window`, `--assessment-min-evaluations`, and
`--assessment-relative-change`. Missing metrics or insufficient observations do not establish a finding.

Independent checks are optional and require a separate evaluation dataset and a built-in preset:

```sh
trlx sft --model MODEL --dataset DATA --quality-checks \
  --quality-preset qa --quality-dataset HELDOUT
```

`HELDOUT` is a supported dataset file or Hub reference with the columns below. `prompt` accepts text
or text messages; generative presets also accept `messages` instead. No custom scorer or rubric is needed.

| Preset | Evaluation rows | Reported evidence |
| --- | --- | --- |
| `language_modeling` | `text`, `messages`, or `prompt` + `completion` | Full-sequence loss and token-weighted perplexity |
| `qa` | `prompt` + `answer` string or `answers` list | Normalized exact match and whitespace-token F1 |
| `classification` | `prompt`, `labels` list, correct `label` | Accuracy, per-class counts/results, invalid/ambiguous answers |
| `multiple_choice` | `prompt`, `choices` mapping labels to text, correct `answer` label | Accuracy, per-class results, invalid/ambiguous answers |
| `json` | `prompt`; optional `required_fields` list and `reference` object | JSON validity, required fields, supplied reference-value correctness |
| `preference` | `chosen`, `rejected`; optional `prompt` | Reward-model ranking accuracy, ties, and margins; reward trainer only |
| `instruction_following`, `writing` | `prompt` | Built-in rubric ratings and rationale from a configured judge |

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

## Dataset cheat sheet

`dataset` works independently of TRL. Files may be JSONL (one object per line), JSON
(an array of objects), CSV, or Parquet. The output extension selects the file format.
CSV cells are strings; use JSONL or Parquet for nested messages. Replacing an existing
output or input requires `--force`. All writers support `--no-staging` for direct writes.

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

## Tutorial: temporary and persistent settings

Settings resolve in this order: **CLI > selected method section > shared config**.
For example, `[methods.sft]` applies to SFT only; shared settings apply across methods.
Training saves its resolved settings with the run and never rewrites `run.toml`.

```sh
# Try a lower learning rate and three epochs for one new run.
trlx sft --model MODEL --dataset DATA \
  --learning-rate 5e-5 --num-train-epochs 3 --output-dir runs/three-epochs

# Use another persistent config.
trlx init --out experiment.toml
trlx sft --config experiment.toml --model MODEL --dataset DATA

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
`teacher` section holds its teacher model. `[verify]` can set a `prompts` dataset.

## Tutorial: evaluation and data sources

The generated `eval_fraction = 0.1` reserves the final 10% of rows for evaluation.
The count is rounded up when loading the actual data; both partitions must remain
nonempty. Evaluation measures held-out performance without updating model weights.
Splitting preserves file order; shuffle first when a random split is appropriate.

```sh
# Change the held-out share for one run.
trlx sft --model MODEL --dataset DATA --eval-fraction 0.2

# Use separate training/evaluation files.
trlx sft --model MODEL --no-split --dataset train.jsonl --dataset-eval eval.jsonl

# Train on all rows, without evaluation.
trlx sft --model MODEL --no-split --dataset data.jsonl
```

To persist separate files, use `split = false` with `dataset_train` and optional
`dataset_eval` in `[dataset]`; remove `dataset` and `eval_fraction`.
Without `dataset_eval`, also remove top-level `eval_*` settings. The CLI's
`--no-split` handles that removal automatically when no evaluation source is set.
The old `train = N` setting is rejected. Dataset IDs use `org/name:split`;
a dataset with multiple splits requires an explicit split name.

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

## Tutorial: checkpoints, resume, and results

The defaults evaluate and save at each epoch's end, retain two checkpoints, and log
every optimizer update. Use `--eval-strategy steps --eval-steps N` and
`--save-strategy steps --save-steps N` for intermediate evaluation/checkpoints.
`save_strategy = "no"` is rejected.

Every fresh run uses `output_dir/YYYYMMDD-N--model--dataset/`. The number advances across
all models and datasets under that parent for the local date, starting above existing numbers.
Model IDs are lowercased with `/` replaced by `-`; local models use their directory name.
Dataset files use their filename stem. `run_name` changes only the display label.

```sh
# RUN_DIR is the generated directory printed at startup; select an existing checkpoint.
trlx sft --resume-from-checkpoint RUN_DIR/checkpoint-100

# Compare a trained checkpoint with the base, or merge its LoRA adapter.
trlx verify RUN_DIR/checkpoint-100 --base MODEL
trlx merge --base MODEL --adapter RUN_DIR/checkpoint-100 --out merged-model
```

Resume loads the run's saved `config.toml`, not today's `run.toml`; model and dataset
arguments need not be repeated. Explicit CLI overrides still apply. Only the current
snapshot schema is supported. Training settings must match; GPU selection and display
controls may change, but switching between sharded and unsharded training is rejected.

Resume continues in the original directory. After validation it automatically removes
metrics and checkpoints beyond the selected saved step and clears stale preflight/verify
reports. No `--force` is needed. Earlier metrics remain; logs append a resume marker and
new output. The live line display prints the continuation; `trlx show RUN_DIR` includes
the retained history. `trlx check` validates resume inputs without performing cleanup.

Run artifacts include `config.toml` (resolved settings), `metrics.jsonl`, `log.txt`,
`preflight.json`, `checkpoint-N` directories, and `verify.json` when verification runs.
Verification checks loading, changed outputs/scores, and chat-template consistency.
A verification failure returns nonzero even if training completed.

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
| llm_judge       | url, model, rubric, timeout, retries, concurrency; optional api_key, max_tokens |

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

## Tutorial: endpoint generation and credentials

`ENDPOINT` is an OpenAI-compatible API base URL, including `/v1` when required;
`MODEL` is the served model name. Both tools load optional `KEY=value` entries from
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
Here `--max-tokens` limits input chunks; the server controls answer length. Provide
both `--answers-endpoint` and `--answers-model` to use a different answer model.
`--questions-prompt FILE` templates use `{chunk}` and `{n}`; `--answers-prompt FILE`
templates use `{chunk}` and `{question}`.

Output contains `messages` and separate `reasoning`. Configure the server's reasoning
parser; inline reasoning is rejected unless `--strip-reasoning-tags` removes a complete
leading block. An unclosed block is always an error.

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

## Further reference

Run `trlx COMMAND --help` or `dataset COMMAND --help` for complete option references.
With dependencies installed, you can use the checkout directly:
`python -m trlx.cli` and `python -m dataset.cli` replace the installed command names.

[SPEC.md](SPEC.md) describes the behavior; [PLAN.md](PLAN.md#second-pass-review-of-phases-4-and-5)
tracks known issues, including reward scoring, distillation memory estimates, and display limitations.
