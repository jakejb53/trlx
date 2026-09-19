# trlx

Two executables that replace the TRL CLI. `trlx` drives TRL trainers from a run config and displays metrics. `dataset` prepares datasets and has no TRL dependency.

Design rule: do what makes sense, not what the HF/ML ecosystem does.

Design rule: general-purpose. Both tools run on any machine and any model transformers can load. No model family, architecture, module name, path, device, or machine fact is hardcoded in source, in `init` output, or in tests. Such facts come from the run config or the model's own metadata. Section 5 records facts about one test model, not assumptions the tools rely on.

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
- Secrets: every `api_key` setting names an environment variable and never holds a key. Both tools load `KEY=value` lines from `.env` in the working directory at startup; a variable already set in the environment wins. The file carries secrets only, never operational settings, and is not committed.

## 2. trlx

### 2.1 Commands

| Command | Does |
|---|---|
| `trlx init <method> --out <path>` | Writes a run config for the method with every trainer-specific field, a curated subset of training arguments, and all blocks below. Each field carries its TRL docstring as a comment. Fields whose default is `None` are written commented out. |
| `trlx <method> <config> [--tui] [--gpus i,j] [--strategy ddp\|fsdp] [--no-verify]` | Runs preflight, trains, runs verify. Displays metrics until the job completes. |
| `trlx show <run> [--tui]` | Renders a run's `metrics.jsonl` with the same renderers. |
| `trlx check <method> <config>` | Preflight only. |
| `trlx verify <checkpoint> --base <model> [--prompts <dataset>]` | Artifact checks only. |
| `trlx merge --base <model> --adapter <dir> --out <dir>` | Merge with adapter-load check. |
| `trlx replay-build --model <path\|name> [--endpoint <url>] --prompts <dataset> --out <path> --max-tokens <n>` | Samples a model on prompts, writes a `messages` dataset. `--model` is a local path or HF id, or with `--endpoint` the served model name; the endpoint takes the connection flags of `dataset chat`. |

Methods: `sft`, `dpo`, `grpo`, `kto`, `rloo`, `reward`, `distillation`. Stable TRL trainers only.

`--tui` shows a full-screen view that stays until the user quits. Without it, output is one line per log step.

### 2.2 Run config

TOML. One file describes the whole job.

- Top-level keys map onto the method's TRL config dataclass. Unknown key: error. Absent key: dataclass default. String `"None"`: `None`, accepted only where the field type admits `None`.
- `output_dir` required. `run_name` defaults to the directory name. Checkpoint interval defaults to the eval interval.
- `[model]`: `path`, `dtype`, `trust_remote_code`, `attn_implementation`. Class is read from the model's own config, never hardcoded. `reward` resolves the sequence-classification variant of that architecture.
- `[teacher]`: `distillation` only. Same keys as `[model]`.
- `[dataset]`:
  - `split = true`: `dataset` (file path or HF id) and `train` (row count, file order). Remaining rows are eval.
  - `split = false`: `dataset_train`, optional `dataset_eval`. No `dataset_eval` disables evaluation and rejects `eval_*` fields.
  - Key mismatch with `split` is an error.
  - HF ids carry a split as `org/name:split`. Without one, a single-split repo is accepted; a multi-split repo is an error listing the splits.
- `[peft]`: LoraConfig fields. Absent means full fine-tune.
- `[ranges]`: expected interval per metric. Required. Missing block is a fatal error. Metrics named here are the display columns.
- `[preflight]`: `offpolicy_logp_per_token`, the per-token log-prob threshold for the off-policy warning, and `rows`, how many train rows it scores. `dpo` and `kto` only.
- `[rewards]`: `grpo` and `rloo`. `funcs`: list of entries, each a bare name from `trl.rewards` or trlx built-ins, `{name, args}` for factories, an HF model path, or `module:function` / `path.py:function`.
- `[replay]`: `sft` only. `dataset`, `fraction` (replay share of the mixed train set, in (0, 1)), `kl_coef`.
- `[verify]`: `prompts` dataset for the post-training generation check.

Dataset files: JSONL, JSON array, CSV, Parquet, by extension. Applies everywhere a dataset is named.

### 2.3 Run directory

```
<run>/
  config.toml      snapshot, including chosen strategy and GPUs
  metrics.jsonl    one record per log step, written by the trlx callback; the only metric source
  log.txt          TRL and transformers output, always written; also passed to stderr without --tui
  preflight.json
  verify.json
  checkpoint-N/    TRL checkpoint
```

### 2.4 Display

Values to three decimals. Each `[ranges]` metric has a change column: difference from the previous logged value of that metric. Out-of-range values per `[ranges]` are marked.

Default: one line per log step: step/total, epoch/total, percent complete, the `[ranges]` metrics with change columns. Eval rows marked. No progress bar. Safe to pipe.

`--tui`: fixed layout, no scrolling, no toggles. Status bar: percent complete, steps done/total, epochs done/total, phase. Metrics table: most recent rows that fit. Checkpoints: step, eval loss, best marked. Log tail. Preflight and verify results. Stays until quit.

### 2.5 Multi-GPU

- All visible GPUs by default. `--gpus` takes device indices.
- trlx starts its own worker processes. No `accelerate launch`, no accelerate config file.
- A supervisor process starts one worker per selected GPU. The supervisor owns the run directory, `config.toml`, `log.txt`, and the display, and never loads a model. Rank 0 owns the callback, `metrics.jsonl`, and preflight. Other ranks train silently. A worker exiting nonzero stops the others and the supervisor exits with that code. Verify runs as a further process after every worker has exited, with all selected GPUs visible, and its exit code is the job's.
- Strategy: trlx chooses data-parallel when the model at its dtype fits one selected GPU with headroom, sharded otherwise. A sharded run whose per-rank estimate, the training state divided by the rank count plus any unsharded original copy (2.9), still exceeds the smallest selected GPU is refused, forced or not. `--strategy` overrides. Choice printed at startup and recorded in the snapshot.

### 2.6 Preflight

Runs before the trainer, sharing its loaded model and dataset. `trlx check` runs it alone. Fatal checks stop the job. Warnings print and continue.

Fatal:
- `[ranges]` missing or unknown config keys.
- LoRA target modules resolve to no module, or trainable parameter count is zero.
- `grpo`/`rloo`: TRL vLLM server unreachable, or the server answering is not the TRL server.
- `[replay]` with `kl_coef > 0` alongside `use_liger_kernel`, `packing`, or `padding_free`.
- `save_strategy = "no"`: nothing would be left to verify or merge.
- Resume from checkpoint whose saved config differs from the run config. Compared on parsed values; `resume_from_checkpoint` itself and the GPU list are not compared, sharding (FSDP or not) is, except under `trlx check`, which chooses no strategy.

Warnings:
- Preference methods: mean per-token log-prob of chosen and rejected (`completion` for `kto`) under the starting model below the threshold in `[preflight]` (off-policy data), over the first `[preflight].rows` train rows.
- Rows whose response is cut by `max_length`, with counts.
- Rendered example with label mask; no assistant tokens in the mask.
- Pad token missing or equal to EOS.
- Gradient checkpointing with `use_cache`.

Reported: resolved LoRA target modules counted per submodule path, and trainable parameter count. Written to `preflight.json` with the warnings.

`trlx check` runs in one process on the first GPU and needs the model to fit it; a larger model's preflight runs inside the training run.

### 2.7 Verify

Runs after training on the final checkpoint unless `--no-verify`, as its own process with every selected GPU visible. `trlx verify` is that process and runs it alone. The base is loaded per the run's `[model]` block when the checkpoint sits in a run directory, else at the checkpoint's own dtype.

- Adapter loaded (LoRA checkpoints): `lora_B` tensor count and max magnitude in `adapter_model.safetensors` equal those in the loaded PeftModel.
- Behaviour changed: generate from base and from checkpoint on `[verify].prompts` (built-in set if absent); report differing count. Zero differing is a failure. A model that cannot generate (`reward`) is compared on its scores.
- Chat template in checkpoint equals the base's.

Result written to `verify.json` in the run directory, or in the checkpoint directory when it is not in a run, and shown as the last output of the job. Nonzero exit on failure.

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

Only `grpo` and `rloo` policy generation requires the TRL vLLM server (per-step weight sync). Every other generation over a network (`llm_judge`, `dataset chat`, `replay-build --endpoint`) accepts any OpenAI-compatible endpoint.

## 3. dataset

Each subcommand reads one input and writes one output. Formats: JSONL, JSON array, CSV, Parquet, by extension; a path without a recognised extension is an error. Never writes in place.

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
