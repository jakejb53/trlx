# trlx

Two executables that replace the TRL CLI. `trlx` drives TRL trainers from a run config and displays metrics. `dataset` prepares datasets and has no TRL dependency.

Design rule: do what makes sense, not what the HF/ML ecosystem does.

Design rule: general-purpose. No model family, architecture, module name, path, device, or machine fact is assumed in source or tests. Model facts come from explicit inputs or model metadata; `init` obtains environment facts from hardware inspection. Section 5 records one test model, not assumptions the tools rely on.

Project-specific runtime-default exceptions: absent trainer settings use TRL dataclass defaults; `run_name` uses the output directory's name; a step-based checkpoint interval follows the evaluation interval (§2.2). Flat configs without `[run]` retain the original CLI defaults: all visible GPUs, automatic strategy, line display, and verification enabled. A present `[run]` requires all its keys. Other operational defaults are written explicitly by `init`.

## 1. Repository

```
trlx/              repo root
  pyproject.toml
  trlx/            library for trlx
  dataset/         library for dataset; imports nothing from trl or trlx
  tests/           unittest
```

- Dependencies: trl, transformers, peft, accelerate, datasets. Pinned to exact versions (trl 1.13.0 at time of writing).
- No other dependencies. CLI: argparse. TUI: curses. Config read: tomllib. Config write: own emitter. HTTP: urllib.
- `trlx` and `dataset` are console entry points declared in `pyproject.toml`, installed onto PATH by `pip install`. Until the project is installed, both run from the repo root as `python -m trlx.cli` and `python -m dataset.cli`.
- Secrets: every `api_key` setting names an environment variable and never holds a key. Both tools load `KEY=value` lines from `.env` in the working directory at startup; a variable already set in the environment wins. The file carries secrets only, never operational settings.

### 1.1 Destructive operations and --force

Every command accepts `--force`. `--force` authorizes the requested destructive operation, including
replacing inputs or directories containing other files. Without it, refuse, explain the consequences,
and offer `--force`. With it, perform the operation without further confirmation. Destructive operations
replace symlinks themselves; healing follows them to repair their targets.

`init`, `merge`, `replay-build`, standalone `verify`, and dataset writers stage complete outputs
before publication by default. `--no-staging` writes directly after reading required inputs;
it saves staging space but a failure can lose the original and leave incomplete output.
It does not authorize replacement: existing destinations still require `--force`.
Merge requires disk staging when its output replaces the base, adapter, or a directory containing
either, including input path aliases. Such a request with `--no-staging` is rejected before loading
or modifying anything, with guidance to omit the option. Separate merge outputs support direct writing.
Merge replacement covers the entire output directory, including base/adapter inputs or their
ancestors and unrelated contents. Nonempty directory replacement and split's two publications
are not atomic transactions; failures identify changed destinations and any recovery paths.
Split validates both destinations before either write and, when staging, prepares both before publication.
Fresh training allocates a new run; checkpoint resume retains its automatic rewind behavior (§2.3).
Training's `--no-staging` controls config snapshots, resume metric rewrites, assessment and quality
evidence, preflight reports, and verification reports. Trainer checkpoints are saved directly in either mode.

### 1.2 Command feedback

Both CLIs report startup, current operations, measured counts when available,
retries, and final outcome with elapsed time to flushed stderr. After ten seconds
without feedback, report the active operation, elapsed time, and time since its
last measured progress; waiting notices do not establish that work is advancing.
Opaque operations have no inferred percentage. Counter output is coalesced to
one second; these intervals are display constants. Help does not start reporting.

Training records complete diagnostics in `log.txt`. The supervisor collects typed
feedback and raw child output separately and owns terminal presentation. Line mode
shows useful operations, measured progress, findings, and unknown output; internal
bookkeeping stays in the log. Identical warnings and rank progress are consolidated,
with warning counts and affected ranks retained. Library preparation bars come from
rank zero; other ranks still report their phases and diagnostics. Only the known PyTorch
`all_gather_into_tensor` FutureWarning is log-only. Waiting notices are log-only
while any worker is training, including evaluation and checkpoint pauses. Outside
training, terminal notices follow 30 seconds without substantive feedback and
identify active operations and available progress by rank.
The TUI retains its log pane. Collection continues after display failure. Metrics
remain authoritative in `metrics.jsonl`. Publication success follows publication
and cleanup. Endpoint batches report completions as observed while preserving
input order in their returned results.

## 2. trlx

### 2.1 Commands

| Command | Does |
|---|---|
| `trlx init [--out <path>] [--force]` | Writes `run.toml` by default, with shared environment-informed settings and all method sections. Existing files require `--force`, which replaces saved settings with freshly generated defaults. No method, model, dataset, or calibration run is required. |
| `trlx <method> [--config <path>] [--model <model>] [--dataset <data>] [options]` | Uses `run.toml` by default. Explicit CLI settings override this run only. Runs preflight, training, and verification. Model/data must come from config or CLI. |
| `trlx show <run> [--tui]` | Renders a run's `metrics.jsonl` with the same renderers. |
| `trlx check <method> [--config <path>] [options]` | Preflight only, with the same training-setting overrides. |
| `trlx verify <checkpoint> --base <model>` | Artifact checks only. |
| `trlx merge --base <model> --adapter <dir> --out <dir>` | Merge with adapter-load check. |
| `trlx replay-build --model <path\|name> [--endpoint <url>] --prompts <dataset> --out <path> --max-tokens <n>` | Samples a model on prompts, writes a `messages` dataset. `--model` is a local path or HF id, or with `--endpoint` the served model name; the endpoint takes the connection flags of `dataset chat`. |

Methods: `sft`, `dpo`, `grpo`, `kto`, `rloo`, `reward`, `distillation`. Stable TRL trainers only.

Training options use hyphenated field names; LoRA fields use `--lora-*` (`lora_alpha` becomes `--lora-alpha`). Boolean options have positive and negative forms; lists and tables use shell-quoted TOML. `--no-lora` and `--no-replay` remove those features for one run. Repeated `--reward` entries replace the reward list; distillation exposes `--teacher`. Conflicting explicit options are errors.

Fresh and resumed training inspect model metadata, tokenizers, and all effective dataset rows
(except synthetic evaluation rows, §2.2), then
print curated tuning settings and an advisory assessment before weight loading, run allocation, or
resume rewind. Metadata/data downloads and library cache writes may precede confirmation.
Paths, launch/display controls, reporting, checkpoint storage, and routine infrastructure
settings are omitted even when explicitly configured. Each override
uses CLI syntax with aligned description/type comments; inactive settings are omitted, automatic
values are explained, and secrets are redacted. Forced tuning settings appear as explanatory comments.
Both display modes require Enter to continue or `q` to cancel successfully. Other input repeats the
prompt; EOF or review I/O failure stops startup with an error. Progress notices pause during review.
Workers and `check` do not prompt. Display failures after launch retain supervisor ownership.

Both CLIs provide top-level command descriptions and command-level help with inputs, options, types, defaults, constraints, and examples. Help reads no run config and performs no hardware inspection, model loading, dataset access, or training. Detailed training help derives fields from installed library metadata.

`init` records native BF16 support and GPU memory metadata, selects BF16 only when all visible GPUs support it, otherwise FP32, and emits conservative batching/checkpointing defaults. LoRA is active with rank 8, alpha 16, dropout 0.05, and `all-linear` targets. It writes `eval_fraction = 0.1`; no model/data/reward objective is guessed. CPU-only initialization is allowed; training requires CUDA. Model fit remains a launch-time estimate.

### 2.2 Run config

TOML. One file holds persistent defaults for all methods. Precedence is explicit CLI values, then the selected `[methods.<name>]` section, then shared top-level settings. Nested method tables are merged by key. Unselected method settings are not passed to TRL. A flat per-run config remains supported.

- Top-level keys map onto the method's TRL config dataclass. Unknown key: error. Absent key: dataclass default. Nullable trainer and LoRA fields accept the string `"None"`; nullable CLI booleans also accept an explicit `None` value.
- `output_dir` is the parent for fresh runs; init writes `runs/<method>` in each method section. Each run gets its own subdirectory (§2.3). `run_name` is a display label, defaulting to that generated directory name. For step evaluation, an omitted checkpoint interval follows the eval interval.
- `[run]`: `gpus` (`all` or comma-separated visible indices), `strategy` (`auto`, `ddp`, `fsdp`), `tui` and `verify` (booleans). CLI forms are `--gpus`, `--strategy`, `--tui`/`--no-tui`, `--verify`/`--no-verify`.
- `[model]`: `path`, `dtype`, `trust_remote_code`, `attn_implementation`. Class is read from the model's own config, never hardcoded. `reward` resolves the sequence-classification variant of that architecture.
- `[teacher]`: `distillation` only. Same keys as `[model]`.
- `[dataset]`:
  - `split = true`: `dataset` and `eval_fraction` strictly between 0 and 1. At load time, the final `ceil(row_count * eval_fraction)` rows evaluate; earlier rows train. Both sides must be nonempty. The old `train` key is rejected.
  - Optional `shuffle_eval_data = true` / `--shuffle-eval-data` selects that same evaluation count randomly without replacement instead of from the end. Only valid with `split = true`. Uses `data_seed` when set, otherwise `seed`; preserves source order within both disjoint sets. Omission or false retains the end split.
  - `split = false`: `dataset_train`, optional `dataset_eval`. Without `dataset_eval` or synthetic evaluation, evaluation is disabled and `eval_*` fields are rejected.
  - Key mismatch with `split` is an error.
  - CLI `--dataset` selects the primary source in either mode. `--no-split` removes fractional-split keys; without a separate or synthetic evaluation source, it also removes the configured evaluation schedule. Contradictory explicit evaluation options are rejected.
  - SFT CPT `text` rows only: `--synthetic-dataset-eval` / `synthetic_dataset_eval = true` replaces configured splitting and evaluation sources, trains on all primary rows, and retains the evaluation schedule. Explicit CLI `--split`, `--eval-fraction`, or `--dataset-eval` conflicts. The resolved snapshot uses `split = false`, `dataset_train`, and `synthetic_dataset_eval`.
    Generate one factual prose summary per primary row from the loaded model after distributed placement, before the step-zero evaluation or optimizer updates; replay rows are excluded. The positive `max_length` is also the generated-token limit. Full prompts must fit the model context; no silent truncation. Model response templates separate reasoning; unparsed tagged responses, empty summaries, and token-limit exhaustion without EOS are fatal.
    Rank zero saves `synthetic-eval.jsonl` in the run directory (§1.1 publication policy). All evaluations and resume reuse those summaries. Missing saved data prevents resume before rewind; no regeneration, hashes, or cross-run cache. Startup skips synthetic evaluation-data inspection; `check` validates without generating. Absence of this option disables the feature.
  - HF ids carry a split as `org/name:split`. Without one, a single-split repo is accepted; a multi-split repo is an error listing the splits.
- `[peft]`: LoraConfig fields. Absent means full fine-tune.
- `[ranges]`: expected interval per metric. Required. Missing block is a fatal error. Metrics named here are the display columns.
- `[preflight]`: `offpolicy_logp_per_token`, the per-token log-prob threshold for the off-policy warning, and `rows`, how many train rows it scores. `dpo` and `kto` only.
- `[rewards]`: `grpo` and `rloo`. `funcs`: list of entries, each a bare name from `trl.rewards` or trlx built-ins, `{name, args}` for factories, an HF model path, or `module:function` / `path.py:function`.
- `[replay]`: `sft` only. `dataset`, `fraction` (replay share of the mixed train set, in (0, 1)), `kl_coef`.
- `[assessment]`: required for training and `check`; explicit independent quality-check
  settings per §2.11. Shared, method-specific, and CLI precedence applies. Missing required keys are errors.

Dataset files: JSONL, JSON array, CSV, Parquet, by extension. Applies everywhere a dataset is named.

### 2.3 Run directory

All methods allocate `<output_dir>/YYYYMMDD-N--model--dataset/` and print its path at startup.
The local date and parent share one counter: N is one above the highest existing number for that date,
regardless of model or dataset. Allocation is serialized across concurrent launches.
Model labels use the model ID or local directory basename; dataset labels use the primary file's stem or
hub ID. Labels are lowercase, with characters outside letters, digits, `.`, `_`, and `-` replaced
by `-`; leading/trailing punctuation is removed. Exact inputs remain in the snapshot.

```
<run>/
  config.toml      resolved method + CLI settings, including chosen strategy and GPUs
  metrics.jsonl    one record per log step, written by the trlx callback; the only metric source
  log.txt          complete operational and library diagnostics, with child source attribution
  preflight.json
  assessment.json  full pre-run scan, metadata, and recommendation evidence
  quality.jsonl    independent evaluation rounds with individual inputs, outputs, and scores
  synthetic-eval.jsonl  optional run-owned CPT summaries, retained across resume
  verify.json
  checkpoint-N/    TRL checkpoint
```

The supervisor resolves inputs once and writes the snapshot before spawning workers. Its `output_dir`
is the actual run directory. Workers read that snapshot, not the operator's source file. The source config
is never rewritten by training. `[launch]` records method, actual strategy, and physical GPU identifiers;
it is reserved for snapshots. One supervisor owns the run through verification.

`resume_from_checkpoint` selects an existing run and loads its saved snapshot, then applies explicit CLI
overrides. Explicit CLI resume does not read the operator's config. Only the current snapshot schema is
supported. After config and checkpoint metadata validation, resume automatically removes metrics and
checkpoint directories and quality rounds beyond the saved `trainer_state.json` global_step, and clears
stale assessment, preflight, and verification reports. Records through that step remain. Logs are
preserved with an appended resume marker. Assessment settings may change on resume; quality baselines
are reused only when dataset, tokenizer, scorer, and evaluation conditions match.
The selected checkpoint is preserved; no `--force` is required for this rewind. Live line output starts
at the continuation; `show` and the TUI retain access to historical metrics. Check-only execution never rewinds.

### 2.4 Display

Values to three decimals. Each `[ranges]` metric has a change column: difference from the previous logged value of that metric. Out-of-range values per `[ranges]` are marked.

Default: training and evaluation tables show step/total, epoch/total, percent complete,
and present `[ranges]` metrics with change columns. Headings precede the first row,
repeat after interruptions or 20 rows, and change with the columns. Narrow terminals
use labelled column groups without dropping values. Final trainer statistics appear
as named summary values. Output is plain text and safe to pipe.
Change columns are named `Change`; no repeated legend sentence accompanies the headers.
Line displays always include `training_loss` and the latest measured `eval_loss`, with `eval_step`
identifying that measurement. Evaluation values remain blank until measured and are retained across
subsequent training rows, resume, and `show`. Other metrics continue to follow `[ranges]`.

Preflight example text and trained-token text remain in `preflight.json`; terminal
output retains token counts, mask information, and warnings. Checkpoint completion identifies the verified
checkpoint directory; only the saving rank announces it. One final command outcome
includes elapsed time, with artifact locations displayed alongside the results.

`--tui`: fixed layout, no scrolling, no toggles. Status bar: percent complete, steps done/total, epochs done/total, phase. Metrics table: most recent rows that fit. Checkpoints: step, eval loss, best marked. Log tail. Preflight and verify results. Stays until quit.

### 2.5 Multi-GPU

- All visible GPUs by default. `--gpus` takes device indices.
- trlx starts its own worker processes. No `accelerate launch`, no accelerate config file.
- A supervisor process starts one worker per selected GPU. The supervisor owns the run directory, `config.toml`, `log.txt`, and the display, and never loads a model. Rank 0 owns the metric callback, `metrics.jsonl`, and preflight. All ranks report attributed diagnostics and progress. Workers destroy initialized process groups on exit; cleanup failures must not replace an existing training failure. A worker exiting nonzero stops the others and the supervisor exits with that code. Verify runs as a further process after every worker has exited, with all selected GPUs visible, and its exit code is the job's.
- Workers and verification run in private process groups owned from creation. Cancellation, partial
  startup, and failure share one shutdown path: SIGINT with 60 seconds for cleanup, then SIGTERM with
  5 seconds, then SIGKILL for survivors. A second Ctrl+C forces escalation without abandoning cleanup.
  Descendants are signalled too; final reaping and output draining are bounded. Shutdown progress and
  failures are reported without replacing the original failure. Cancelled runs never start verification
  and return 130 without a cancellation traceback.
- Strategy: trlx chooses data-parallel when the model at its dtype fits one selected GPU with headroom, sharded otherwise. A sharded run whose per-rank estimate, the training state divided by the rank count plus any unsharded original copy (2.9), still exceeds the smallest selected GPU is refused, forced or not. `--strategy` overrides. Choice printed at startup and recorded in the snapshot.

### 2.6 Preflight

Runs before the trainer, sharing its loaded model and dataset. `trlx check` runs it alone. Fatal checks stop the job. Warnings print and continue.

Fatal:
- `[ranges]` missing or unknown config keys.
- LoRA target modules resolve to no module, or trainable parameter count is zero.
- `grpo`/`rloo`: TRL vLLM server unreachable, or the server answering is not the TRL server.
- `[replay]` with `kl_coef > 0` alongside `use_liger_kernel`, `packing`, or `padding_free`.
- `save_strategy = "no"`: nothing would be left to verify or merge.
- Resume from a different method or changed effective training settings, including CLI overrides. `resume_from_checkpoint`, `output_dir`, display/launch controls, assessment settings, and the GPU list are excluded; actual sharding is compared except under `trlx check`, which chooses no strategy.

Warnings:
- Preference methods: mean per-token log-prob of chosen and rejected (`completion` for `kto`) under the starting model below the threshold in `[preflight]` (off-policy data), over the first `[preflight].rows` train rows.
- Rows whose response is cut by `max_length`, with counts.
- First prepared example has no trained tokens in its label mask.
- Pad token missing or equal to EOS.
- Gradient checkpointing with `use_cache`.

Reported: resolved LoRA target modules counted per submodule path, and trainable parameter count. Written to `preflight.json` with the warnings.

`trlx check` runs in one process on the first GPU and needs the model to fit it; a larger model's preflight runs inside the training run.

### 2.7 Verify

Runs after training on the final checkpoint unless `--no-verify`, as its own process with every selected GPU visible. `trlx verify` is that process and runs it alone. The base is loaded per the run's `[model]` block when the checkpoint sits in a run directory, else at the checkpoint's own dtype.

- Adapter loaded (LoRA checkpoints): `lora_B` tensor count and max magnitude in `adapter_model.safetensors` equal those in the loaded PeftModel.
- Full checkpoints must load successfully. No prompt generation or reward-score comparison is performed.
- Chat template in checkpoint equals the base's.

Result written to `verify.json` in the run directory, or in the checkpoint directory when it is not in a run, and shown as the last output of the job. Nonzero exit on failure.
Standalone verification requires `--force` to replace an existing report and supports `--no-staging` (§1.1).
`verify --prompts`, training `--verify-prompts`, and `[verify].prompts` are rejected as removed settings.

### 2.8 Rewards

Built into trlx, resolved by bare name alongside `trl.rewards`, each a factory taking `args`:

- `reference_match`: equals, contains, or fuzzy match against a column after normalisation.
- `regex`: pattern match; optional group compared to a column.
- `phrases`: reward for required phrases, penalty for forbidden ones.
- `json_valid`: parseable JSON, optional required keys.
- `length_window`: target range in tokens or words, linear falloff outside.
- `llm_judge`: OpenAI-compatible endpoint, rubric prompt, parsed score. Batching, timeout, retry.

### 2.9 Replay (SFT)

- `[replay].dataset` is mixed into training so that `fraction` of the mixed train set is replay rows: the first R rows of the replay dataset in file order, R chosen so R / (train rows + R) = `fraction`. The replay dataset must have the same columns as the train set. Replay rows never enter eval.
- `kl_coef > 0` adds a KL term on replay batches between the training model and the original model: exact forward KL, original against training, over the full vocabulary at each assistant token of each replay row, normalised like the SFT loss, per trained token of the step, gathered at those positions before the float32 upcast. The mean KL per replay token is logged as `replay_kl`. Original model: same PeftModel with adapters disabled for LoRA runs; a second loaded copy for full fine-tunes, frozen and unsharded on every rank under every strategy.
- `kl_coef > 0` is implemented as an SFTTrainer subclass overriding `compute_loss`, signature columns, and the collator to carry a per-example replay flag. It forces `loss_type = "nll"` (the KL needs logits, and TRL's default loss materialises none, section 5) and rejects the key at top level. Incompatible with `use_liger_kernel`, `packing`, and `padding_free`.
- `kl_coef = 0` is plain mixing on the stock trainer with no restriction.

### 2.10 Endpoints

Only `grpo` and `rloo` policy generation requires the TRL vLLM server (per-step weight sync). Every other generation over a network (`llm_judge`, independent quality judging, `dataset chat`, `replay-build --endpoint`) accepts any OpenAI-compatible endpoint.

### 2.11 Assessment

All seven methods scan every training/evaluation row, including selected replay rows, before review,
except synthetic evaluation rows (§2.2).
Findings identify measured facts, preparation projections, or heuristics, with evidence and recommendations.
The trainer remains authoritative on prepared data; preflight checks projections against actual row counts.
Startup and `check` display only warnings and errors, with relevant settings. Disabled ordinary evaluation
is a warning. After every evaluation, rank zero assesses the full chronological metric history, including
any quality results from that evaluation. Recency weighting, noise handling, and warmup interpretation
are internal analysis policy. Sparse evidence limits conclusions without suppressing the assessment.
Assessment never changes settings, optimizer/scheduler state, control flags, or
checkpoint selection. Scores do not establish an optimal learning rate or universally correct reward.
At training completion, the same analysis produces a final report after any enabled quality checks.
Reports appear inline and in `log.txt`; `metrics.jsonl` remains authoritative. Assessment performs no
additional evaluation. Completion alone does not establish model improvement.
Displayed assessments lead with a recommendation and only the measurements supporting it. Routine
observations, analysis labels, and generic disclaimers are omitted; actionable problems remain visible.
The final assessment states the whole-run outcome and next action, rather than repeating interim advice.
Fresh runs with ordinary evaluation enabled first evaluate at step zero, before any optimizer update,
using the trainer's normal held-out data and scoring. `eval_on_start` is managed by trlx, not a setting.
The baseline stays in `metrics.jsonl` across resume; resumed runs do not replace it. Independent quality
checks do not repeat their own baseline round for this ordinary step-zero evaluation.
Loss assessments compare against the step-zero baseline and identify the best measured step. Suggested
experiments name the changed flag and proposed value. An unevaluated endpoint is not presented as a
measured final result; a small loss decrease alone does not establish success or justify longer training.

`[assessment]` requires `quality_checks`,
`quality_preset`, `quality_dataset` (both nullable using `"None"`), `quality_max_length` (at least 2),
`quality_max_new_tokens`, and `quality_batch_size` (both positive). `init` writes explicit defaults;
quality checks start disabled. CLI overrides include `--quality-checks` / `--no-quality-checks`.

Independent checks require a separate dataset and a built-in preset: `language_modeling`, `qa`,
`classification`, `multiple_choice`, `json`, `preference`, `instruction_following`, or `writing`.
Only reward training uses `preference`; the other presets assess generative models. No custom scorer
or authored rubric is required. Judging presets require `[assessment.judge]`: explicit `url`, `model`,
`api_key` (environment-variable name or `"None"`), positive `timeout`, nonnegative `retries`, and positive
`max_tokens`. Judge results are model judgments. Invalid judge output is a failed check, not a zero score.

When enabled, quality checks run before the first update, at ordinary evaluation events, and at completion
even if ordinary evaluation is disabled. Every benchmark row is assessed; generation is greedy and input
limits never silently discard examples. LM loss is token-weighted over full sequences with one-token
window overlap. Expected scoring failures are reported without changing training control.
Evaluation preserves per-module modes, generation-configuration ownership, and random-number state;
distributed ranks coordinate forwards, while rank zero scores and publishes results.
Aggregate quality metrics use `metrics.jsonl`; `quality.jsonl` retains individual and partial failure evidence.
Reports use staged publication by default, with `--no-staging` selecting direct writing.

## 3. dataset

Dataset transforms read inputs and write output files; `split` writes two and `stats` only reports.
Formats: JSONL, JSON array, CSV, Parquet, by extension; a path without a recognised extension is an error.
Replacing an existing output or input requires `--force`; writers support `--no-staging` (§1.1).

| Subcommand | Does |
|---|---|
| `convert` | Between file formats, and between TRL dataset formats (prompt/completion and messages). |
| `shuffle` | Reorder rows with a seed. |
| `split` | First N rows or a fraction to one output, remainder to another. Optional exact-match key column so equal keys land on one side. |
| `mix` | Concatenate several inputs with per-source fractions. |
| `fields` | Add constant or derived field, remove, rename, swap. |
| `filter` | Keep rows by length limit on a column or a simple predicate. |
| `sample` | Take N rows, random with seed or head. |
| `cpt` | Text file to a `text` field dataset. Paragraph-aware chunks up to a token limit estimated by a conservative heuristic. |
| `pairs` | Two `messages` datasets into `prompt`/`chosen`/`rejected`, aligned by user-turn content with system messages excluded. Unmatched rows reported and dropped; `--strict` makes them fatal. One `messages` dataset into `prompt`/`completion` for distillation. |
| `heal` | Deterministic JSON and JSONL repairs: truncated last line, trailing commas, unclosed final brace or bracket, single quotes, unquoted keys, Python literals, concatenated objects. Ambiguous errors are reported with line and column and left alone. Every repair is listed. |
| `chat` | Text file to `messages` via endpoint. Pass 1: chunk plus instruction yields N questions. Pass 2: each question plus its chunk yields the answer. Each pass has its own endpoint and model flags, and a built-in instruction replaceable by `--prompt <file>`. Unparseable replies reported and skipped. Rows carry the answer's reasoning in a `reasoning` column, from the endpoint's reasoning field, `""` when it returns none. The endpoint must return reasoning in that field; a reply with reasoning inline in the content is fatal unless `--strip-reasoning-tags` removes it. |
| `stats` | Token length distribution per column. With `--model`, per-token log-prob of response columns. |
| `eval-build` | One factual prose summary per input `text` row via an OpenAI-compatible endpoint; writes ordered `text` rows for ordinary CPT evaluation. |

`dataset eval-build INPUT --out OUTPUT --endpoint URL --model NAME --max-tokens N`
requires nonempty string `text` in every row and a positive completion-token limit.
It does not sample, re-chunk, or withhold source rows. Built-in instructions preserve key
facts, names, numbers, and relationships without invented facts, commentary, or Q&A formatting.
Inputs and destination are validated before requests. Empty, malformed, or incomplete
responses fail with the source row number; every summary requires `finish_reason = "stop"`.
No incomplete generation set is published. Separate reasoning is excluded; inline reasoning
uses `chat`'s `--strip-reasoning-tags` policy. Publication follows §1.1.
Train on all original chunks; generate summaries once and reuse them as `--dataset-eval`
for the step-zero baseline and subsequent ordinary evaluations.

For best results, generate summaries using the same model you'll use this data set to train.

`chat` and `eval-build` share explicit runtime-default exceptions: concurrency 4,
timeout 120 seconds, retries 2. Optional CLI overrides require positive integer concurrency,
finite positive timeout, and nonnegative integer retries. `--api-key` names an environment
variable; neither command needs `run.toml`.

## 4. Tests

`unittest`, standard library. Targeted at components that fail silently: config loader `None` and type rules, `heal`, `pairs`, metric reader, `lora_B` check, built-in rewards, replay trainer loss on a tiny model.

## 5. Verified facts

- Qwen3.8-27B is multimodal. The trainer loads it with the image-text-to-text class; adapter keys are `base_model.model.model.language_model.layers.*`. Loading with the causal-LM class silently builds zero adapters. Merge must use the class from the base config.
- `all-linear` targets the vision tower too. Text-only target list: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_a in_proj_b in_proj_z out_proj`.
- Adapter load check: compare count and max magnitude of `lora_B` tensors in `adapter_model.safetensors` against the loaded PeftModel. Zero in model with nonzero in file means key mismatch.
- Trainer log fields: SFT emits `loss`, `eval_loss`, `mean_token_accuracy`; DPO emits `rewards/*`, `logps/chosen`, `logps/rejected` (summed over response tokens), `logits/*`; all emit `grad_norm`, `learning_rate`, `epoch`, `num_tokens`.
- `assistant_only_loss` applies to SFT only.
- GRPO and RLOO require the TRL vLLM server for the policy, not stock vLLM, because of per-step weight sync.
- SFTTrainer with `use_liger_kernel` sets `skip_logits=True` in training, so no logits exist for a KL term.
- SFTTrainer's signature columns are `input_ids`, `labels`, `seq_lengths`; other columns are dropped.
- SFTConfig 1.13 `loss_type` defaults to `chunked_nll`, which patches the model forward to compute cross-entropy from hidden states in chunks and returns `logits=None`. Only `nll` materialises logits. Extra dataset columns survive dataset preparation and are dropped by the signature-column rule alone.
- DPOConfig 1.13 accepts `loss_type` as a list with `loss_weights`. No custom code needed.
- GRPOTrainer and RLOOTrainer accept `reward_funcs` as callables, model paths, or loaded models, weighted by `reward_weights`.
- Missing `causal_conv1d` and `flash-linear-attention` fall back to slow reference kernels for this model.
