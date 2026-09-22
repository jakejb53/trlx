# trlx implementation plan

Contract: `SPEC.md`. This file records what each phase builds and how it is verified. Phases are sequential; each leaves the tool usable.

## Status

Development progress is tracked in this file. Each phase heading below carries its status when work on it starts.

Current status: Phases 1-8, startup settings review, settings assessment, training output improvements, cooperative cancellation, synthetic CPT evaluation, optional random evaluation splitting, assessment recommendation overhaul, and checkpoint defaults complete. Optional acceleration recommendations remain planned. Full-scale cancellation and synthetic evaluation execution validation remain with the operator. Five stale line-renderer tests remain unresolved (assessment overhaul below).
Phase 8, startup settings review, settings assessment, and training output improvements record current work; the addendum records earlier work and supersedes historical Phases 1-7.

### Session notes

- Invoke the venv's executables by absolute path; a relative path triggers a site.py prefix warning on every call.
- Nothing is installed. Both tools run from the repo root as `python -m trlx.cli` and `python -m dataset.cli` (addendum, Packaging).
- All `trlx` subcommands are implemented. The seven training methods are wired to `train.run`.
- `tests/test_imports.py` enforces the import rule by `ast` scan of source, not by importing.
- `_BACKOFF_BASE_SECONDS` in `dataset/endpoint.py` is a constant by decision, an accepted exception to the no-runtime-defaults principle.
- `data.json` and `swap_dataset_fields.py`, once the real-data fixtures for `dataset` checks, are no longer in the repo root.
- `dataset stats --model` and `dataset chat` against a real endpoint are verified; see the addendum.
- Preflight reports resolved LoRA targets per submodule path (SPEC 2.6, ruled in Phase 6); no warning judges where they landed.
- The `config.toml` snapshot must contain `run_name` and `[ranges]`; `show.load_config` reads only those, with tomllib, and never calls `config.load`.
- The `metrics.jsonl` record schema is documented in the `trlx/metrics.py` docstring. Phase 5 attaches `metrics.callback_class()(run_dir)`, streams line output with `render_lines.header` and `render_lines.line`, and shows the live TUI through `show.load` and `render_tui.run`.
- `POLL_MS` in `trlx/render_tui.py` is a display constant, the same accepted exception as `_BACKOFF_BASE_SECONDS`.
- Six-metric `[ranges]` defaults (`dpo`, `kto`) do not fit the TUI at 80 columns; it reports the width needed. The line display is unaffected.
- `TrlxError` in `trlx/__init__.py` is the one exception `cli.main` catches (exit 1), mirroring `DatasetError`.
- `config.py` rejects `model_init_kwargs`, `trust_remote_code`, and `teacher_model_*` at top level; `[model]` and `[teacher]` own model description and `model.py` passes loaded objects to the trainer. It also type-checks top-level scalars against the dataclass hints.
- `init` writes `[peft]`, `[replay]`, `[verify]` fully commented out; required blocks active with placeholders that load. `init` refuses an existing file; `merge` refuses an existing output directory.
- transformers 5.17 resolves absent `eval_steps` to `logging_steps` in `__post_init__`; `config.py` applies the `save_steps` default after instantiation for that reason.
- `DistillationTrainer` 1.13 needs only a `prompt` column and ignores others; `dataset pairs` single-input output works as is. Relevant to Phase 5 `data_load.py`.
- `train.py` sets `CUDA_VISIBLE_DEVICES` to the selected device before any CUDA context exists. With more than one GPU visible, transformers' Trainer wraps the model in `nn.DataParallel` and peft fails with a cross-device error.
- The `config.toml` snapshot is the operator's file byte for byte, with `run_name` prepended when absent and a `[launch]` table (`strategy`, `gpus`) appended. `config.load` rejects the snapshot (`launch` is an unknown key); the Phase 6 resume check must read it with tomllib.
- Verification fixtures in the repo root, 80 synthetic arithmetic rows each: `sft.jsonl` (`messages`), `dpo.jsonl` (conversational `prompt`/`chosen`/`rejected`, also used by `reward`), `kto.jsonl` (`prompt`/`completion`/`label`), `distill.jsonl` (`prompt`). Configs `sft-lines.toml`, `dpo-lines.toml`, `kto-lines.toml`, `reward-lines.toml`, `distill-lines.toml`, each writing `runs/<name>`. The TUI was driven under a `pty` from a script that sends `q` once `checkpoint-20` exists.
- `dpo` log field names confirmed: the trainer logs every `[ranges]` default in `init_cmd.py` plus `rewards/chosen`, `rewards/rejected`, `logps/rejected`, `logits/*`, `entropy`, `mean_token_accuracy`. `sft` confirmed the same way in slice 1.
- `config.load(path, method, fsdp=None)`: the launcher's sharding choice enters through the constructor because transformers configures FSDP in `__post_init__`. A top-level `fsdp` key in a run config is rejected (`config.STRATEGY_FIELDS`).
- The supervisor holds about 550 MiB on the first selected GPU for the run: `config.load` initializes CUDA because transformers validates `bf16` against a real device, and hiding the GPUs makes that validation fail.
- `sft-lines.toml` is the fixture for every GPU count; only `--gpus` changes between checks.
- `RewardTrainer` 1.13 accepts the conversational `prompt`/`chosen`/`rejected` shape. `DistillationConfig` has no `max_length` field; `config.load` rejects it by name. `kto`, `reward`, and `distillation` log every `[ranges]` default in `init_cmd.py`.
- vllm 0.28.0 is installed in the venv for the TRL vLLM server; it moved torch to 2.13.0 and left the five pins unchanged. Server used for verification, on the GPU not training: `CUDA_VISIBLE_DEVICES=1 trl vllm-serve --model Qwen/Qwen3-0.6B --port 8000 --gpu_memory_utilization 0.3`; ready when `GET /health` returns 200 (the trailing-slash path redirects).
- `use_vllm` and `vllm_mode` are forced (`config.VLLM_FORCED`) for methods with a `[rewards]` block, rejected at top level, and skipped by `init`, the same pattern as `fsdp`. `[rewards]` is required for those methods. `grpo` and `rloo` log every `[ranges]` default plus `rewards/<name>/mean` and `/std` per reward function, named after the callable.
- Fixtures for `grpo`/`rloo`: `grpo.jsonl` (conversational `prompt` plus `answer`), `rewards_extra.py` (`brevity`, a `path.py:function` entry), `grpo-lines.toml`, `rloo-lines.toml`. Qwen3's `<think>` prefix fills `max_completion_length` on every step, so rewards stay near zero; the runs verify the pipeline, not learning.
- After training, transformers logs `train_runtime` and friends as a final `on_log`; it renders as a row with no metric cells.
- Preflight stages (`preflight.py` docstring): config-only checks in the supervisor before the run directory exists and in `trlx check`; trainer checks on rank 0 after `train.build_trainer`; the off-policy forward in `PreflightCallback.on_train_begin` on every rank, a collective under FSDP2 (`callback_handler.model` is the in-place sharded module). Rank 0 writes `preflight.json` after each of the last two.
- Verify is `launch.Job`: the supervisor spawns `python -m trlx.cli verify <run>/checkpoint-N --base <[model].path>` once every worker exited 0, with all selected GPUs visible, output to `log.txt`; `--no-verify` is supervisor-only. `verify.run` reads `[model]` from the run snapshot with tomllib and refuses a `--base` that is not the run's path.
- `trlx check` runs one process on the first visible GPU and loads the model like a single-GPU worker (CPU, placed by the Trainer). Not `device_map="auto"`: accelerate's dispatch hooks turn `forward` into a partial that TRL 1.13 SFTTrainer's chunked-CE patch cannot wrap. It chooses no strategy, so `compare_snapshot` gets `None` and skips the sharding comparison. The Trainer creates `output_dir` on construction; `check` removes it again when it did not exist before and is still empty.
- TRL 1.13 SFTTrainer prepares the dataset into `input_ids` and `labels` (-100 where untrained) and truncates to `max_length` before the trainer holds it; the label mask comes from `labels`, and truncation counts are measured on the raw rows. The warning wording follows the trainer: `truncation_mode = "keep_end"` cuts the prompt, `packing` skips truncation, `reward` drops long rows. The off-policy check applies the same mode and excludes (and counts) responses with no token left after the cut, which the trainer drops.
- The vLLM probe is GET `/get_world_size`, a route only `trl vllm-serve` has; a stock vLLM server answers `/health` and 404 there. transformers 5.17 builds the FSDP plugin with `state_dict_type` `FULL_STATE_DICT` and FSDP version 2, so FSDP checkpoints are consolidated and load with `from_pretrained`. `DefaultFlowCallback` saves at the last step under a step save strategy.
- `resume_from_checkpoint` is typed `str | None` in transformers 5; the operator gives the checkpoint path, never `true`.
- `VLLM_PROBE_SECONDS` in `trlx/preflight.py` and `MAX_NEW_TOKENS` in `trlx/generate.py` are constants by decision, the same accepted exception as `POLL_MS`.
- Fixtures `dpo-lines.toml` and `kto-lines.toml` carry a `[preflight]` block (`rows = 16`). The TUI was driven by a pty script sending `q` once `verify.json` exists.
- `tests/test_preflight.py` covers `compare_snapshot` only; it is the one preflight component that would fail silently.
- `verify.run` on a full checkpoint reads the checkpoint's own config to load base and checkpoint as the same kind (`_checkpoint_kind`): a full fine-tune of `reward` is compared on scores like a LoRA one.
- Second-pass review of Phase 6 done; its two bugs (full-checkpoint kind, NaN rows in the off-policy check) and wording items are fixed.
- `[replay].kl_coef > 0` forces `loss_type = "nll"` (`config.REPLAY_KL_FORCED`), selects `replay_trainer.ReplayTrainer`, and adds the `replay` flag column; `kl_coef = 0` is plain mixing on the stock `SFTTrainer` with no column.
- `ReplayTrainer._replay_term` agrees across ranks whether any rank has replay tokens before the reference forward and the metric gather; a rank-local branch there deadlocks ddp.
- `ReplayTrainer._nll_normalisation` copies transformers 5.17's loss scaling (`Trainer.compute_loss`, `Trainer.training_step`); re-check it on any transformers pin change.
- Fixtures for replay, in the repo root: `replay.jsonl` (`trlx replay-build --model Qwen/Qwen3-0.6B --prompts sft.jsonl --out replay.jsonl --max-tokens 48`) and `sft-replay-lines.toml` (LoRA, `fraction = 0.2`, `kl_coef = 0.1`, `replay_kl` in `[ranges]`).
- Stock vLLM server for the `replay-build --endpoint` check: `CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-0.6B --port 8000 --gpu-memory-utilization 0.3 --max-model-len 2048`, API base `http://127.0.0.1:8000/v1`.

## Pins

From the project venv at planning time. `pyproject.toml` pins these exactly.

| Package | Version |
|---|---|
| trl | 1.13.0 |
| transformers | 5.17.0 |
| peft | 0.21.0 |
| accelerate | 1.15.0 |
| datasets | 5.0.1 |

torch (2.13.0 installed, set by vllm 0.28.0) is not pinned by trlx; it is installed by the operator for their CUDA. `requires-python >= 3.11` for `tomllib`.

## Module map

Flat. No subpackages.

```
trlx/
  cli.py            argparse tree, dispatch
  config.py         load TOML, map onto TRL dataclass, None rule, [dataset] rules, unknown-key errors
  toml_write.py     emitter for init: tables, scalars, lists, comment lines, commented-out keys
  init_cmd.py       per-method template: curated TrainingArguments, trainer fields with docstrings, blocks
  trainers.py       registry: method -> (Config class, Trainer class, model-class rule, dataset format)
  model.py          load by class named in the model config; sequence-classification variant for reward
  data_load.py      [dataset] block -> train and eval Datasets; file formats; HF id with :split
  launch.py         GPU selection, strategy choice, worker spawn, rank-0 ownership
  train.py          run directory, config snapshot, log.txt, preflight -> train -> verify
  metrics.py        TrainerCallback writing metrics.jsonl; reader
  ranges.py         [ranges] parsing, out-of-range evaluation, change columns
  render_lines.py   one line per log step
  render_tui.py     curses fixed layout
  show.py           trlx show
  preflight.py      checks in 2.6
  verify.py         checks in 2.7
  generate.py       local generation, shared by verify and replay-build
  adapter_check.py  lora_B count and magnitude comparison
  merge.py
  rewards.py        built-ins in 2.8 and resolver for [rewards] entries
  replay_trainer.py SFTTrainer subclass
  replay_build.py
dataset/
  cli.py
  io.py             read and write JSONL, JSON array, CSV, Parquet by extension
  endpoint.py       OpenAI-compatible chat completions client: batching, timeout, retry
  env.py            .env loader; every api_key setting names a variable
  convert.py
  rows.py           shuffle, split, sample, mix, filter
  fields.py
  cpt.py
  pairs.py
  heal.py
  chat.py
  stats.py
tests/
```

Import rule: `dataset/` imports nothing from `trlx/` or `trl`. `trlx/` may import `dataset/io.py`, `dataset/endpoint.py`, `dataset/env.py`, and `dataset/progress.py`. A test asserts the rule by scanning imports.

## Config schema

Top level: fields of the method's TRL config class.

Curated TrainingArguments written by `init`: `output_dir`, `run_name`, `learning_rate`, `num_train_epochs`, `max_steps`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `gradient_accumulation_steps`, `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `save_total_limit`, `logging_steps`, `warmup_steps`, `lr_scheduler_type`, `weight_decay`, `max_grad_norm`, `optim`, `bf16`, `gradient_checkpointing`, `seed`, `resume_from_checkpoint`. Any other TrainingArguments field may be added by name.

| Block | Keys |
|---|---|
| `[model]` | `path`, `dtype`, `trust_remote_code`, `attn_implementation` |
| `[teacher]` | same as `[model]`; `distillation` only |
| `[dataset]` | `split`; `dataset`, `train`; or `dataset_train`, `dataset_eval` |
| `[peft]` | LoraConfig fields |
| `[ranges]` | `<metric> = [low, high]` |
| `[preflight]` | `offpolicy_logp_per_token`, `rows`; `dpo`, `kto` only |
| `[rewards]` | `funcs = [...]` entries per spec 2.2; `grpo`, `rloo` only |
| `[replay]` | `dataset`, `fraction`, `kl_coef`; `sft` only |
| `[verify]` | `prompts` |

`[ranges]` metrics per method, written by `init` with initial intervals to be tuned against real runs:

| Method | Metrics |
|---|---|
| sft | `loss`, `eval_loss`, `mean_token_accuracy`, `grad_norm` |
| dpo | `loss`, `eval_loss`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `grad_norm` |
| kto | `loss`, `eval_loss`, `rewards/chosen`, `rewards/rejected`, `kl`, `grad_norm` |
| grpo, rloo | `reward`, `reward_std`, `kl`, `completions/mean_length`, `grad_norm` |
| reward | `loss`, `eval_loss`, `accuracy`, `grad_norm` |
| distillation | `loss`, `eval_loss`, `grad_norm` |

Exact log field names are confirmed against each trainer in Phase 5 before the defaults are written.

## Multi-GPU design

- `--gpus` absent: all devices in `torch.cuda.device_count()`. Present: the listed indices.
- The supervisor (`train.py`, no `--_rank`) validates the config, chooses the strategy, writes the run directory and `config.toml`, opens `log.txt`, and spawns one worker per selected GPU via `subprocess` running `python -m trlx.cli <method> <config> --_rank <r> [--strategy fsdp]`. Worker stdout and stderr go to `log.txt`. The supervisor runs the display from the run directory, waits, kills the rest when one worker exits nonzero, and exits with that code. It never loads a model but holds a CUDA context on the first selected device: transformers validates `bf16` against a real device when the config is instantiated.
- Worker environment: `CUDA_VISIBLE_DEVICES` its one physical device, and for more than one GPU `RANK`, `WORLD_SIZE`, `LOCAL_RANK=0`, `MASTER_ADDR=127.0.0.1`, `MASTER_PORT` (a free port). One GPU: no distributed variables, strategy `single`.
- Strategy rule: parameter count from instantiating the model class on the `meta` device, times the `[model].dtype` size, times 1.5 for peft runs or 8 for full fine-tunes, compared with the smallest selected GPU's total memory. Fits: `ddp`. Otherwise: `fsdp`, passed to `config.load` and set through the TRL config's `fsdp` field at construction. A top-level `fsdp` key in a run config is rejected. Multipliers are constants in `launch.py`.
- Rank 0 only: callback, `metrics.jsonl`, preflight report. Verify: a separate process after the workers, all selected GPUs visible (`launch.Job`).

## Phases

### Phase 1: scaffold (complete)

Files: `pyproject.toml`, `bin/trlx`, `bin/dataset`, `trlx/__init__.py`, `trlx/cli.py`, `dataset/__init__.py`, `dataset/cli.py`, `tests/__init__.py`, `tests/test_imports.py`.

- `pyproject.toml`: setuptools build backend, `[tool.setuptools] packages = ["trlx", "dataset"]`, `script-files = ["bin/trlx", "bin/dataset"]`. setuptools 84.0.0 honours `script-files` under `pyproject.toml` (confirmed; schema marks it discouraged but supported).
- `pip install -e .` from the repo root.

Verify: `trlx --help` and `dataset --help` list their subcommands. `python -m unittest` passes.

### Phase 2: dataset tool (complete)

Files: all of `dataset/`, `tests/test_heal.py`, `tests/test_pairs.py`.

Order: `io.py`, `rows.py`, `fields.py`, `convert.py`, `sample`, `stats.py` (lengths only), `heal.py`, `pairs.py`, `cpt.py`, `endpoint.py`, `chat.py`, `stats.py --model`.

Verify:
- Done: `dataset convert data.json` through JSONL and Parquet and back, rows and content equal.
- Done: `dataset fields --swap chosen=rejected` on `data.json` matches the `rename_columns` in `swap_dataset_fields.py`. The script's HF `train_test_split` is not reproducible without the HF download.
- Done: `dataset split` on the shuffled result, two outputs, counts sum to input.
- Done: `dataset heal` on hand-made files for each repair class; report lists each repair.
- Done: `dataset pairs` on two small `messages` files with one unmatched row each side; `--strict` exits nonzero.
- Done against a mock server only: `dataset chat` with a short text file. Not run against a real endpoint.
- Not run: `dataset stats --model`.
- Done: unit tests for `heal` and `pairs`.

### Phase 3: config, init, model loading, merge (complete)

Files: `trlx/config.py`, `trlx/toml_write.py`, `trlx/init_cmd.py`, `trlx/trainers.py`, `trlx/model.py`, `trlx/adapter_check.py`, `trlx/merge.py`, `tests/test_config.py`, `tests/test_adapter_check.py`. Also `trlx/__init__.py` (`TrlxError`) and `trlx/cli.py` (`init`, `merge` wired; `TrlxError` caught).

Verify:
- Done: `trlx init <method> --out` for all seven methods; each output loads back through `config.py` without error. Also a unit test.
- Done: config tests: unknown key rejected, absent key yields dataclass default, `"None"` accepted only on Optional fields, `[dataset]` key mismatches rejected, HF id with and without `:split`. `tests/test_adapter_check.py` uses a plain torch module wrapped by peft, no model download.
- Done: `trlx merge --base Qwen/Qwen3.8-27B --adapter <adapter> --out <scratch>` (a LoRA adapter for that base, outside the repo) loads with the multimodal class from the base config and passes the `lora_B` check. Running the check against a causal-LM load of the same adapter reports the mismatch and exits nonzero.
- Second-pass review of the Phase 3 files waived by user decision.

### Phase 4: metrics and display (complete)

Files: `trlx/metrics.py`, `trlx/ranges.py`, `trlx/render_lines.py`, `trlx/render_tui.py`, `trlx/show.py`, `tests/test_metrics.py`.

Verify:
- Done: `trlx show` on a hand-built run directory prints one line per step with change columns, eval rows marked, out-of-range marked.
- Done: `trlx show --tui` on the same directory renders all panes at 80x24 and at 120x40, stays until `q`.
- Done: metric reader test: partial last line (job still writing) is skipped, not an error.

### Phase 5: training (complete)

Files: `trlx/train.py`, `trlx/data_load.py`, `trlx/launch.py`, `trlx/cli.py` method subcommands.

Order: single-GPU `sft` end to end (done); then `dpo` (done); then multi-GPU launch and strategy (done); then `kto`, `reward`, `distillation` (done); then `rewards.py`, `grpo` and `rloo` (done). All methods are wired to `train.run`. `train.py` is split into the supervisor (`_supervise`) and worker (`_worker`) paths of the multi-GPU design. Preflight and verify calls are marked seams in `train._worker` for Phase 6.

Verify:
- Done for `sft`: `Qwen/Qwen3-0.6B`, LoRA, 20 steps, GPU 0, both display modes, on `sft.jsonl` in the repo root. Run directory contains `config.toml`, `metrics.jsonl`, `log.txt`, `checkpoint-N`. `trlx show` renders the result. Non-empty `output_dir` refused.
- Done for `dpo`: same settings on `dpo.jsonl`, both display modes, no `train.py` change needed.
- Done: `sft` with `--gpus 0`, `--gpus 1`, and `--gpus 0,1` (auto-chose `ddp`), both display modes on two GPUs; `metrics.jsonl` written once, both ranks' output in `log.txt`; `--gpus 1` observed via `nvidia-smi` mid-run with GPU 0 at 2 MiB and 0%. A missing dataset on two GPUs propagated as `worker rank 0 exited with code 1`, exit 1.
- Done: `--strategy fsdp` forced on the 0.6B LoRA run completes without an operator `fsdp_config`.
- Done: `kto`, `reward`, `distillation` each complete 20 steps on GPU 0 on their fixtures; `kto` also on two GPUs. `distillation` used `Qwen/Qwen3-0.6B` as both teacher and student.
- Done: `grpo` and `rloo`, 20 steps on GPU 0 against the TRL vLLM server on GPU 1, `[rewards]` holding one built-in (`reference_match`), one TRL reward (`think_format_reward`), and one `path.py:function` (`rewards_extra.py:brevity`); each logged separately.
- Done: each built-in reward has a unit test on fixed completions in `tests/test_rewards.py`; `llm_judge` against a stdlib mock endpoint, not a live one.
- Done: log field names confirmed for all seven trainers; no `[ranges]` default in `init_cmd.py` needed correction.

### Phase 6: preflight, verify (complete)

Files: `trlx/preflight.py`, `trlx/verify.py`, `trlx/generate.py`, `tests/test_preflight.py`; `trlx/cli.py` (`check <method> <config>`, `verify`), `trlx/train.py` (`check`, `build_trainer`, preflight stages, verify hand-off), `trlx/launch.py` (`spawn_verify`, `Job`), `trlx/config.py` (`[preflight].rows`, `model_spec`), `trlx/adapter_check.py` (`task_type`, from `merge.py`), `trlx/data_load.py` (`load_ref`), `trlx/show.py` (`CHECKPOINT_DIR`), `trlx/init_cmd.py`. `rewards.py` landed in Phase 5.

Verify:
- Done: `trlx check sft` on `sft-lines.toml` prints the LoRA breakdown, trainable count, truncation count, and rendered example; with `assistant_only_loss = true` the trained text is the assistant turn only; with a wrong target module name it exits 1 naming `[peft].target_modules`; exit leaves no run directory.
- Done: `trlx check dpo` and `kto` report the off-policy check (chosen -1.26, rejected -2.22, completion -1.82 mean per-token log-prob on the fixtures). `trlx check grpo` against a stock vLLM server on port 8000 exits 1 in seconds naming the route.
- Done: `sft` on GPU 0, both display modes: `preflight.json`, then verify spawned after training, adapter loaded, 8 of 8 outputs differ, chat template equal, `verify.json`, exit 0. `sft` and `dpo` with `--gpus 0,1 --strategy fsdp`: the off-policy forward ran as a collective and verify loaded across both GPUs. `reward` on GPU 0: verify compared scores.
- Done: resume with `learning_rate` changed refused naming the key; the same config resumed a one-GPU run on two GPUs.
- Done: standalone `trlx verify` on a checkpoint copied outside any run, `--prompts sft.jsonl` (`messages` column), 80 of 80 differ; `--base` not matching the run snapshot refused.
- Done: full checkpoints outside a run, built from `Qwen/Qwen3-0.6B` without training: a sequence-classification save verifies on scores with both sides loaded as that class; a causal save identical to the base fails with `behaviour unchanged`, exit 1.
- Not run: `trlx verify <adapter> --base Qwen/Qwen3.8-27B` (the adapter from Phase 3). Both GPUs held 87 GiB of a vLLM server not started by trlx during the session; the 27B does not fit the remainder. The breakdown line for `all-linear` on the multimodal model is expected to show the vision tower; unverified.
- Not run: `llm_judge` against a running endpoint (unit-tested against a mock in Phase 5).

### Phase 7: replay (complete)

Files: `trlx/replay_trainer.py`, `trlx/replay_build.py`, `tests/test_replay.py`; `trlx/config.py` (`REPLAY_KL_FORCED`, `[replay]` parsed before the TRL config), `trlx/data_load.py` (`mix_replay`, `REPLAY_COLUMN`), `trlx/train.py` (`_mix_replay`, replay trainer and reference copy in `build_trainer`), `trlx/launch.py` (`COPY_MULTIPLIER`, fsdp per-rank refusal), `trlx/cli.py` (`replay-build`), `trlx/preflight.py` (`padding_free`), `trlx/init_cmd.py`, `tests/test_config.py`, `SPEC.md`.

Verify:
- Done: `trlx replay-build` from `Qwen/Qwen3-0.6B` locally and from a stock vLLM endpoint, 80 `messages` rows each on the `sft.jsonl` prompts.
- Done: `tests/test_replay.py`: mixing count and flag; the flag reaches `compute_loss` on every batch; a batch without replay rows equals the parent's loss; KL zero before any update and positive after one; the KL term normalised by `num_items_in_batch`.
- Done: 20-step runs, exit 0: `kl_coef = 0` on GPU 0; with `kl_coef > 0`, each logging `replay_kl`, LoRA on one GPU in both display modes with verify passing, on two GPUs under ddp, forced fsdp, and ddp with gradient accumulation 2; full fine-tune on GPU 0 (`--no-verify`).
- Done: refusals for `loss_type` with `kl_coef > 0`; `use_liger_kernel`, `packing`, `padding_free` with `kl_coef > 0`; a fraction needing more rows than the replay dataset has; mismatched columns; `replay-build` endpoint flags without `--endpoint` and `--endpoint` without them. `launch.choose_strategy`'s fsdp refusal exercised with patched memory figures.
- Not run: a full fine-tune with `kl_coef > 0` under fsdp; verify on a full fine-tune replay checkpoint.
- Second-pass review of Phase 7 done; the KL scaling bug and the `padding_free` gap are fixed.

### Phase 8: actionable errors and consistent --force (complete)

Make both tools explain failures and offer direct recovery without requiring source-code inspection.
Run directories, resume, output replacement, and the error audit are complete.

Completed: run directories and resume (2026-09-20)

- All seven trainers create `output_dir/YYYYMMDD-N--model--dataset/`. `output_dir` is the parent;
  allocation serializes one daily counter across that parent's existing runs. Model IDs are normalized,
  local models use their directory basename, and dataset files use their stem. `run_name` is a display
  label, defaulting to the generated directory name. Startup prints the actual path.
- The snapshot and workers use the actual run directory. Resume loads the selected run's saved snapshot
  and applies explicit CLI overrides, without needing today's source config for explicit CLI resume.
  Current schema only: existing runs are disposable; no legacy conversion is required or implemented.
- Resume validates checkpoint metadata and saved weights' presence, then automatically removes metrics
  and checkpoints beyond the saved `global_step` and clears stale preflight/verify reports. It preserves
  the selected checkpoint, earlier metrics, and logs; a resume marker precedes new log output. No force
  is required. Live lines print the continuation; `show` and the TUI retain historical metrics.
- `trlx check` validates without rewinding. Existing method, effective-training-setting, and sharding
  checks remain. Snapshot and metrics replacement is staged by default; complete malformed metric records are errors.
- Files: new `trlx/run_dirs.py`; `trlx/train.py`, `trlx/config.py`, `trlx/preflight.py`, `trlx/metrics.py`,
  `trlx/cli.py`, `trlx/options.py`; new `tests/test_run_dirs.py` and updated resolution, preflight, train,
  and metrics tests. `README.md`, `SPEC.md`, and command help describe the contract.
- Verification: 108 distinct focused tests passed with mocked training and repository-local scratch.
  Independent review found no confirmed blockers. Live GPU training and custom FSDP checkpoint layouts
  remain unverified. From the repo root with the project environment:

  ```sh
  PYTHONDONTWRITEBYTECODE=1 python -B -m unittest tests.test_run_dirs tests.test_resolution tests.test_preflight tests.test_train tests.test_cli tests.test_metrics tests.test_imports
  PYTHONDONTWRITEBYTECODE=1 python -B -m unittest tests.test_run_dirs
  ```

Completed: output replacement and actionable errors (2026-09-20)

- All 25 subcommands accept `--force`; output writers support `--no-staging` per SPEC 1.1.
  `dataset/io.py` owns destination validation, staged publication, and direct writing for both packages.
  Destructive replacement replaces symlinks themselves; healing repairs their targets. Replacing a
  hard link leaves its other names unchanged. Directory failures report retained recovery paths;
  split validates both destinations, stages both by default, and reports partial publication.
- `trlx merge --force` supports replacing the base, adapter, or their containing directory using disk
  staging. `--no-staging` rejects these overlaps before loading or mutation, including chained input
  symlink aliases. Separate outputs retain direct writing.
- Fresh-run allocation and automatic resume rewind remain intact. Execution flags stay outside saved
  training settings. Training forwards output controls to workers and verification; run-owned metadata
  is staged by default, while trainer checkpoints remain direct writes.
- Expanded contextual errors for config/encoding, dataset formats and serialization, model/adapter
  loading, scoring, reward arguments, replay-column conflicts, logs/metrics, and endpoint failures.
  Endpoint diagnostics redact credentials; malformed credential-valued config fields do not echo values.
  Expected failures retain nonzero exits; programming errors remain distinguishable.
- Command help, `README.md`, and `SPEC.md` describe the implemented contract. New suites:
  `tests/test_io.py`, `tests/test_endpoint.py`, `tests/test_merge.py`, `tests/test_verify.py`,
  `tests/test_replay_build.py`; existing CLI, config, dataset, and runtime suites were extended.
- Verification: the 299-test focused run and 14 reward tests passed. After the final merge restriction,
  56 targeted tests passed. Independent reviews completed; reported findings were fixed and re-reviewed.
  Training/merge lifecycle tests mocked model/GPU work; live GPU training/merging and custom FSDP layouts
  remain unverified. The full replay-trainer loss suite was not rerun. Reward tests use a local mock
  HTTP server and passed outside the socket-restricted sandbox. Runtime configs and existing runs
  were not changed; tests used repository-local scratch. Installation remains the operator's task.

Verification commands, from the repo root with the project Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_io tests.test_dataset_cli tests.test_heal tests.test_env tests.test_chat \
  tests.test_endpoint tests.test_cli tests.test_init tests.test_config tests.test_resolution \
  tests.test_hardware tests.test_merge tests.test_verify tests.test_replay_build tests.test_train \
  tests.test_run_dirs tests.test_preflight tests.test_metrics tests.test_data_load \
  tests.test_adapter_check tests.test_imports tests.test_pairs tests.test_replay.MixReplayTest -q
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest tests.test_rewards -v
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_merge tests.test_cli tests.test_io tests.test_imports -q
```

### Startup settings review (complete, 2026-09-20)

- Fresh and resumed training show curated tuning settings for the selected trainer on stdout as CLI
  overrides with aligned description/type comments. Values include resolved defaults; automatic
  and forced settings are explained, inactive controls omitted, and credentials redacted.
- After config validation and the full assessment scan, Enter continues and `q` cancels before weight
  loading, run allocation, or resume rewind. Both display modes prompt; workers and `check` do not.
  EOF or review I/O failure stops startup. Progress pauses while awaiting input; post-launch
  display failures retain existing supervision behavior.
- Files: new `trlx/review.py` and `tests/test_review.py`; updated `trlx/options.py`, `trlx/train.py`,
  `trlx/cli.py`, `tests/test_train.py`, `tests/test_cli.py`, `SPEC.md`, and `README.md`.
- Verification: 92 focused tests passed, including real CLI/config rendering with mocked GPU and
  run boundaries. Independent review findings were fixed and re-reviewed with no remaining blockers.
  Live GPU training was not run. Runtime configs and existing runs were unchanged.

From the repo root, with the project environment's Python executable:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_review tests.test_train tests.test_cli tests.test_resolution tests.test_imports -q
```

Completed: tuning-only review (2026-09-20)

- All seven trainers omit paths, launch/display controls, reporting, checkpoint storage, and routine
  infrastructure settings even when explicitly configured. Method-specific tuning and enabled-feature
  details remain visible; advanced tuning fields require deliberate inclusion in the presentation lists.
- Updated `trlx/review.py`, `tests/test_review.py`, and `SPEC.md`. A comparable SFT configuration
  displays 31 controls. Training behavior, runtime configurations, and existing runs are unchanged.
- Verification: all 20 review tests passed, including selection across all seven trainers. No live GPU
  training was run. From the repo root with the project environment's Python executable:

  ```sh
  PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest tests.test_review -q
  ```

### Settings assessment (complete, 2026-09-20)

- All seven trainers scan model metadata, tokenizers, and all effective train/eval/replay rows before
  confirmation. Method-specific findings distinguish measurements, projections, and heuristics.
- Assessments run after every evaluation and at completion, interpreting the recorded history internally
  without changing training settings or control.
- Optional built-in quality presets cover language modeling, QA, classification, multiple choice, JSON,
  reward preferences, instruction following, and writing; the last two use a configured judge and shipped rubrics.
  Checks run at baseline, scheduled evaluations, and completion even when ordinary evaluation is disabled.
- Added the approved `[assessment]` defaults to `run.toml` and `trlx init`, with CLI overrides;
  quality checks default off. Training and `check` require the block, including older configs/snapshots.
- `assessment.json` retains pre-run evidence, `quality.jsonl` retains sample evidence, and `metrics.jsonl`
  holds quality aggregates. Resume trims evidence to the checkpoint and compares only matching baselines.
- Core additions: `trlx/data_profile.py`, `trlx/assessment.py`, `trlx/quality_scorers.py`, and `trlx/quality.py`;
  configuration, trainer callbacks, review/display, resume, tests, `SPEC.md`, and `README.md` updated.
- Validation covered 367 tests; stale assertions were corrected and affected tests passed on rerun.
  Synthetic CPU, single-GPU, DDP, and LoRA/FSDP checks preserved training state and the next update,
  including FSDP generation-error cleanup. No full-scale model run or live judge endpoint was tested.

Completed: assessment readability (2026-09-20)

- Startup and `check` assessments show only warnings and errors, grouped by severity; the entire
  section is omitted when neither exists. Evidence uses readable fields, affected-row percentages,
  token totals, and explicit unknowns. Relevant CLI settings
  accompany findings with resolved values and explanations, including method and evaluation overrides.
- Full report evidence is preserved; long terminal lists have counted previews. Runtime notices retain
  their existing format. Files: `trlx/review.py`, `trlx/train.py`, `tests/test_review.py`,
  and `tests/test_assessment_lifecycle.py`.
- Verification: 102 focused tests passed; independent review findings were fixed and re-reviewed
  with no remaining blockers. All 102 tests passed again after the warnings-and-errors-only follow-up,
  including empty output and evidence preservation checks. No live GPU training was run.
  From the repo root with the project Python:

  ```sh
  PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
    tests.test_review tests.test_assessment_lifecycle tests.test_assessment tests.test_train -q
  ```

### Training output improvements (complete, 2026-09-20)

- `trlx/feedback.py` separates typed worker feedback from raw output. The supervisor records both
  in `log.txt`, retains useful information inline, consolidates warnings and rank progress, and
  keeps waiting notices log-only while workers train; other phases retain 30-second notices. Unknown diagnostics
  remain visible; library preparation bars come from rank zero. Collection survives display failure.
- Live metrics have nearby, repeated headings, separate training/evaluation tables, wrapped column
  groups, and named final statistics. `metrics.jsonl` remains authoritative, including resume deltas.
- Preflight example text remains in `preflight.json`; terminal output retains counts and mask warnings.
  Verification checks loading, adapter integrity, and chat-template consistency without prompt comparisons.
  Checkpoint notices identify the saved directory.
  Workers clean up initialized process groups without replacing an existing training failure.
- Updated CLI help and `SPEC.md`; runtime configurations and existing runs were unchanged.
- Verification: 208 focused tests passed, including CPU subprocess transport, concurrent output,
  warning policy, dataset counts, rendering, resume, and mocked worker cleanup. Independent review
  findings were fixed and re-reviewed with no remaining confirmed blockers. Live GPU/NCCL behavior
  remains unverified.

From the repo root, using the project environment's Python executable:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_feedback tests.test_progress tests.test_metrics tests.test_train tests.test_cli \
  tests.test_preflight tests.test_verify tests.test_resolution tests.test_review tests.test_imports \
  tests.test_dataset_cli tests.test_data_load -q
```

Completed: training feedback and final assessment (2026-09-20)

- Removed prompt generation/scoring from verification. Retired `verify --prompts`, training
  `--verify-prompts`, and `[verify].prompts` produce explicit removal errors; existing configs are not rewritten.
- Startup warns about disabled ordinary evaluation and projected observation counts below assessment
  requirements. Unknown and resumed schedules do not receive definite coverage projections.
- Rank zero emits one final assessment after evaluation and enabled quality checks, using existing
  metrics and settings. It reports supported conclusions, evidence gaps, and relevant controls in
  terminal output and `log.txt`; no extra evaluation or automatic setting changes occur.
- Validation: 289 focused tests passed, then 78 checks after final edits. Independent review findings
  were fixed and re-reviewed with no remaining blockers. No live GPU training was run.
- Read-only validation against `runs/sft/20260920-5--jakejb53-qwen3.8-27b-heretic--books-cpt`
  confirmed 19/40 required training observations and 1/3 required evaluations. Runtime configs and
  existing run artifacts were unchanged. `SPEC.md`, `README.md`, and CLI help describe the new contract.

From the repo root with the project Python:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_feedback tests.test_preflight tests.test_verify tests.test_cli tests.test_config \
  tests.test_init tests.test_resolution tests.test_train tests.test_assessment tests.test_review \
  tests.test_assessment_lifecycle tests.test_quality_callback tests.test_metrics -q
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_review tests.test_assessment_lifecycle tests.test_train tests.test_imports -q
```

Completed: assessment after every evaluation (2026-09-20)

- Supersedes the configurable-window assessment behavior above. Every evaluation and training
  completion use one assessment path over the full chronological history, with recency, noise,
  and warmup interpretation handled internally. Limited evidence qualifies conclusions without silencing reports.
- Removed `runtime_window`, `runtime_min_evaluations`, `runtime_relative_change`, their CLI options,
  generated defaults, and coverage warnings. Disabled ordinary evaluation remains a startup warning.
- Updated `trlx/assessment.py`, `metrics.py`, `review.py`, `config.py`, `options.py`, `init_cmd.py`,
  `README.md`, and `SPEC.md`. No compatibility handling or existing-run changes.
- `run.toml` was not modified. The operator will regenerate it with `trlx init --force` from the repo root;
  this replaces `run.toml` with fresh defaults.
- Validation is left to the operator. No tests were updated or run for this revision; earlier test
  results do not validate it. Existing assessment tests still reference the removed controls.

Completed: actionable guidance, visible evaluation loss, and baseline (2026-09-20)

- Assessments lead with an action and supporting measurements. Final guidance compares with the
  step-zero baseline, identifies the best measured step, and proposes specific next-run settings.
  Routine narration, generic disclaimers, and isolated gradient increases are omitted; actionable problems remain.
- Line-mode tables show `training_loss`, the latest measured `eval_loss`, and `eval_step` together.
  Resume seeds display state from retained metrics; `show` uses the same renderer. Unmeasured losses
  remain blank. Repeated legends are removed and change columns are named `Change`.
- Fresh runs with ordinary evaluation enabled automatically evaluate before the first update.
  `eval_on_start` is managed internally. Resume preserves the original baseline; independent quality
  checks do not duplicate their baseline round at the step-zero ordinary evaluation.
- Updated assessment, review, line rendering, supervisor, saved-run display, configuration, CLI
  ownership, quality callbacks, `README.md`, and `SPEC.md`. `run.toml` and existing runs were unchanged.
- Validation remains with the operator. No tests were updated or run for these changes.

Completed: cooperative cancellation (2026-09-21)

- First Ctrl+C records worker cancellation; ranks agree at matching preparation, training, evaluation,
  and generation/scoring boundaries, finish outstanding GPU work, and destroy process groups normally.
  The supervisor announces that clean shutdown may take 60 seconds or longer, without automatic
  escalation. Second Ctrl+C forces owned process groups. Cancellation skips completion-only work and
  verification and exits 130. Failure, verifier, and leftover-helper cleanup remain bounded.
- Launch registers the whole rank cohort before propagating interruption. Workers acknowledge handler
  readiness before receiving SIGINT; helper processes finish normally. A failed peer during cancellation
  switches to bounded failure cleanup. NCCL watchdog settings are unchanged; no experimental abort API
  or automatic GPU recovery probe is used.
- Added `trlx/cancellation.py`; updated CLI, training, processes, launch, feedback, preflight, quality,
  synthetic evaluation, `README.md`, and `SPEC.md`. Added cancellation/process tests and updated affected
  supervisor/CLI tests. Operational configuration and existing runs were unchanged.
- Validation: 160 focused tests passed; six tiny single-GPU/DDP/FSDP training/evaluation cancellation
  checks passed. Both GPUs returned to 0% utilization without recovery. Independent review found no
  remaining blockers. Full-scale model cancellation remains untested.
- Broader validation exposed the unrelated stale
  `tests.test_quality_callback.MetricsWriterTest.test_runtime_advisory_failure_isolated` expectation
  for the removed runtime assessment behavior; it remains unresolved.

From the repo root, using the project environment's Python executable:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_cancellation tests.test_processes tests.test_feedback tests.test_train \
  tests.test_cli tests.test_imports tests.test_preflight tests.test_quality \
  tests.test_quality_callback.SchedulingTest -q
```

### Synthetic CPT evaluation (complete, 2026-09-21)

- `trlx sft --synthetic-dataset-eval` supports raw CPT `text` rows. All primary rows train;
  the loaded model generates one summary per row before step-zero evaluation, after distributed
  placement. The existing positive `max_length` limits generated tokens. Replay rows are excluded.
- Startup skips synthetic evaluation-data inspection. The flag replaces configured splitting or
  evaluation sources while retaining the evaluation schedule; explicit conflicting CLI options fail.
  `trlx check sft` validates without generating.
- Rank zero saves `<run>/synthetic-eval.jsonl`. Evaluation and resume reuse it unchanged;
  missing data prevents resume before rewind. No hashes or cross-run cache. Generation preserves
  training state; incomplete summaries fail rather than becoming evaluation data.
- Standalone `dataset eval-build INPUT --out OUTPUT --endpoint URL --model MODEL --max-tokens N`
  remains available for JSONL, JSON, CSV, and Parquet. It preserves row order, uses existing reasoning
  and publication policies, and requires `finish_reason="stop"` without changing the `Reply` contract.
  Positive `N` limits completion tokens. `chat` and `eval-build` share optional concurrency 4,
  timeout 120 seconds, retries 2: approved runtime-default exceptions.
- Added `trlx/synthetic_eval.py` and `dataset/eval_build.py`; updated `trlx/train.py`, `config.py`,
  `options.py`, `assessment.py`, `review.py`, `dataset/cli.py`, `dataset/endpoint.py`, `README.md`,
  and `SPEC.md`. Runtime configuration files and existing runs were not modified.
- Independent source review completed. The inherited generation-stopping issue was fixed and
  re-reviewed with no remaining reported defects. No tests, endpoint requests, or training runs
  were executed. Execution validation remains with the operator.

### Random evaluation split (complete, 2026-09-21)

- All trainers support `--shuffle-eval-data` / `[dataset].shuffle_eval_data` for standard percentage
  splitting. Samples without replacement using the existing evaluation fraction and rounding;
  selected rows are excluded from training. Both sets retain source order. Default remains the tail split.
- Selection uses `data_seed` when set, otherwise `seed`, consistently across assessment, workers,
  and `check`. The option survives run snapshots and requires `split = true`.
- Updated `trlx/data_load.py`, `config.py`, `options.py`, `train.py`, `review.py`, focused tests,
  `README.md`, and `SPEC.md`. No operational configuration files changed.
- Validation: 125 focused tests passed. The broader supervisor suite's 7 failures and 17 errors
  reproduced with the original loading calls; those process-control fixtures were subsequently corrected
  and the supervisor suite passed during cooperative cancellation work above.

### Assessment recommendation overhaul (complete, 2026-09-22)

- Supersedes earlier runtime assessment rules. Reports separate whole-run outcome from recent
  behavior over up to three measured intervals, using actual step spacing. Full history remains
  available; outliers, warmup, non-finite recovery, and diminishing gains qualify conclusions.
- Advice incorporates recorded learning rates and matching independent quality comparisons,
  preserves conflicting signals and warning evidence, and never automatically halves learning rate,
  doubles steps, or adds an epoch. Completed runs receive completion-aware advice.
- Best measured steps are distinguished from checkpoints verified through metadata and saved-weight
  checks. Assessments remain advisory and perform no additional evaluation or training-state changes.
- Updated `trlx/assessment.py`, `trlx/review.py`, `trlx/metrics.py`, assessment/config/lifecycle/quality
  callback tests, `README.md`, and `SPEC.md`. Added `tests/test_assessment_trends.py`,
  `tests/test_assessment_scenarios.py`, and `tests/test_assessment_presentation.py`.
- Validation: 305 of 310 tests passed. Independent review found no remaining assessment blockers.
  Read-only replay of `runs/sft/20260922-2--jakejb53-qwen3.8-27b-heretic--friends-train`
  reports diminishing gains and available checkpoint 32 without prescribing another epoch.
  No live training was run; operator configs and existing run artifacts were unchanged.
- Five unrelated stale tests in `tests.test_metrics.StreamingLines` remain unresolved:
  `test_columns_change_repeats_headings`, `test_narrow_terminal_groups_keep_names_and_values`,
  `test_present_columns_and_phase_headings`, `test_repeats_headings_after_twenty_rows_and_interrupt`,
  and `test_resume_retains_history_without_replaying_old_rows`. Their assertions expect the old table format.

Validation command, from the repo root using the project environment's absolute Python executable:

```sh
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tests" python -B -m unittest \
  tests.test_assessment tests.test_assessment_trends tests.test_assessment_scenarios \
  tests.test_assessment_config tests.test_assessment_lifecycle tests.test_quality_callback \
  tests.test_assessment_presentation tests.test_review tests.test_metrics tests.test_train \
  tests.test_quality tests.test_quality_scorers tests.test_preflight tests.test_resolution tests.test_imports -q
```

### Checkpoint defaults (complete, 2026-09-22)

- `init` omits `save_strategy` and `save_steps`. Saving follows evaluation by default;
  disabled evaluation uses `save_strategy = "steps"`, `save_steps = 0` for final-only saving.
  Explicit save settings override the defaults. The installed trainer owns final saving.
- Updated `trlx/init_cmd.py`, `trlx/config.py`, `tests/test_init.py`, `tests/test_config.py`,
  `tests/test_resolution.py`, and `SPEC.md`.
- Validation: 37 tests passed: `tests.test_resolution` and five rendering-only `InitDefaults`
  tests (`test_cpu_defaults`, `test_precision_uses_all_devices`,
  `test_all_methods_and_explicit_objectives`, `test_first_run_preset`,
  `test_hardware_comments_and_terminal_width`). Checks cover CLI overrides, snapshots, and
  installed trainer save decisions through step 1001. No live training was run.
- Existing `run.toml` was not modified; its explicit `save_strategy = "epoch"` still overrides
  these defaults. Removing that setting requires separate configuration approval.

### Additional TODO: optional acceleration recommendations (planned)

- During `trlx init`, use GPU architecture and installed Python, PyTorch, and CUDA versions to recommend
  applicable optional acceleration packages, including `causal-conv1d` and `flash-linear-attention`.
  No model selection, model weights, or dataset is required. Explain which model architectures benefit;
  distinguish verified compatibility from unknown compatibility rather than promising installability.
- During `trlx check` and training, inspect the selected model's configuration before loading weights
  to identify applicable accelerators. Distinguish missing packages from installed packages that fail
  to import, and provide actionable guidance for either condition.
- Recommendations are advisory: no automatic installation, blocking prompt, or suppression of failures.
  Document optional accelerators and their purpose concisely in `README.md` and command help.
- Expected files: `trlx/hardware.py`, `trlx/init_cmd.py`, `trlx/cli.py`, `trlx/preflight.py`,
  `tests/test_hardware.py`, `tests/test_init.py`, `tests/test_preflight.py`, `README.md`, and `SPEC.md`.
  Verify recommendations with mocked environments and model configurations, without loading weights
  or installing packages. Implementation awaits approval.

## Addendum: post-Phase-7 session (2026-09-19)

Work after Phases 1-7 were complete; this addendum supersedes their historical descriptions.
Phase 8 above records subsequent work.

### Document corrections

- SPEC 2.6: the resume check compares sharding except under `trlx check`,
  which chooses no strategy.
- `replay_kl` is logged only by runs with `kl_coef > 0` (Phase 7 verify line).
- SPEC and PLAN carry no absolute paths and no `../` paths. Paths inside the
  repo are relative to the repo root; anything outside it is generic (`python`,
  `pip`, `<adapter>`).

### Packaging (supersedes Phase 1)

- The editable install was removed: `pip uninstall -y trlx`, and
  `trlx.egg-info/` deleted. Nothing of this project is installed outside the
  repo. Verified: no `trlx` or `__editable__` entry in site-packages, and
  `import trlx` from another directory raises `ModuleNotFoundError`.
- `script-files` and the `bin/trlx`, `bin/dataset` shims are gone. pip copied
  those files into the venv at install time, which is a copy of repo files
  outside the repo. `[project.scripts]` in `pyproject.toml` declares
  `trlx = "trlx.cli:main"` and `dataset = "dataset.cli:main"`; pip generates
  wrappers instead.
- Until the project is installed, both tools run from the repo root as
  `python -m trlx.cli` and `python -m dataset.cli`. `dataset/cli.py` gained the
  `__main__` guard it never had; without it the module entry printed nothing
  and exited 0.
- Installing is a packaging step for the end of the project, not a development
  prerequisite.

### Secrets

- `dataset/env.py` loads `KEY=value` lines from `.env` in the working
  directory at the start of both CLIs. A variable already set in the
  environment wins; a missing file is not an error; a line without `=` is
  fatal with its number.
- Every `api_key` setting names an environment variable. `trlx/rewards.py`
  `llm_judge` passed its `api_key` value straight to `Endpoint`, which would
  have put a live key in the run config and its snapshot; it now reads the
  named variable and fails when it is unset.
- `.env` holds secrets only.
- The import rule now admits `dataset.env` alongside `dataset.io` and
  `dataset.endpoint`; `tests/test_imports.py` carries the same list.

### dataset chat: reasoning

- Reasoning must arrive in the endpoint's reasoning field
  (`reasoning` or `reasoning_content`), which means a reasoning parser on a
  self-hosted server. A reply whose content opens an XML-style tag that also
  closes is fatal, naming the tag and both remedies;
  `--strip-reasoning-tags` removes the block instead. An unclosed tag is
  truncation and is fatal either way.
- Rows carry the answer's reasoning in a `reasoning` column beside `messages`,
  `""` when the endpoint returns none. Pass 1's reasoning is discarded: that
  pass produces question strings, not data.
- `dataset/endpoint.py` gained `Reply(content, reasoning)` with `complete_full`
  and `complete_many_full`. `complete` and `complete_many` still return content
  strings, so the trlx callers are unchanged.
- Sampled through OpenRouter: `deepseek-v4.1-flash`, `glm-5.3-flash`,
  `gemini-3.8-flash`, `kimi-k3` and `grok-4.6` populate the reasoning field;
  `gpt-5.6-luna`, `claude-sonnet-5` and `gemma-4-31b-it` return none. None
  returns reasoning inline. Inline reasoning comes from a server started
  without a reasoning parser, so it is a server property, not a model family's.

### Tests

- `tests/test_replay.py` pins `CUDA_VISIBLE_DEVICES` to one device at module
  scope. With more than one GPU visible, transformers' Trainer multiplies the
  per-device batch size by the device count (`nn.DataParallel`), which changed
  the dataloader's batch count and failed the test. A trlx worker always sees
  one device.
- New: `tests/test_chat.py` (inline-reasoning guard, reasoning column) and
  `tests/test_env.py` (the `.env` loader). 116 tests pass.

### Verification completed from the Phase 5-7 "not run" list

- `dataset stats --model` on `sft.jsonl`, `dpo.jsonl` and `kto.jsonl`: exact
  token counts and per-token log-probs on all three column shapes. `chosen`
  -1.3, `rejected` -2.3 and `completion` -1.8 agree with the preflight
  off-policy figures.
- `dataset chat` against a live endpoint: the run that exposed the inline
  reasoning bug, then the guard's error, then `--strip-reasoning-tags`
  recovering real questions and answers from local vLLM, then
  `deepseek-v4.1-flash` through OpenRouter with the `reasoning` column
  populated. Fixtures `chat-input.txt`, `chat.jsonl`, `chat-openrouter.jsonl`.
- Full fine-tune with `kl_coef > 0` under fsdp on two GPUs: fixture
  `sft-replay-full-fsdp.toml`, `replay_kl` 0.13 rising to 1.42 over 20 steps,
  exit 0. Verify then passed on that checkpoint.
- Verify on a full fine-tune replay checkpoint, also standalone on
  `runs/sft-replay-full/checkpoint-20`: 8 of 8 outputs differ, chat template
  equal, exit 0.
- `llm_judge` against a running endpoint: fixture `grpo-judge-lines.toml`,
  one `llm_judge` entry on `google/gemma-4-31b-it` through OpenRouter with
  `api_key = "OPENROUTER_API_KEY"` resolved from `.env`. Called directly, it
  scored a correct answer 10, a wrong answer 0 and `<think>` filler 0. A
  20-step `grpo` run on GPU 0 against the TRL vLLM server on GPU 1 logged
  `rewards/llm_judge/mean` 1.25-2.50 with `judge_std` 2.50-5.00, `grad_norm`
  up to 1.09, and verify passed. `max_completion_length` must leave room for
  an answer after Qwen3's `<think>` block: at 64 every completion was filler,
  every score 0, `reward_std` 0, the adapter never moved, and verify
  correctly failed with `behaviour unchanged`. 256 fixed it.

- `trlx verify <adapter> --base Qwen/Qwen3.8-27B` on the Phase 3 adapter: the
  adapter loaded with the multimodal class from the base config, 606 `lora_B`
  tensors and max magnitude 0.01074434258043766 equal in file and model, 8 of
  8 outputs differ. The chat template check failed, correctly: the operator had
  since installed a customized template on the base (27,159 characters,
  `template_version = "qwen3.8-froggeric-v22.4"`) while the checkpoint carries
  the 8,952-character template saved at training time. Exit 1. This is the
  template check's failing branch on a real artifact; the passing branch is
  covered by every 0.6B run.
- The `all-linear` breakdown on the multimodal model, fixture
  `check-27b-lines.toml`: `model.visual.blocks` 108 modules,
  `model.visual.merger` 2, `model.language_model.layers` 496; 62,365,440
  trainable of 27,419,094,000. The vision tower is targeted, as SPEC section 5
  records.

### Second-pass review of Phases 4 and 5

Done by three reviewers over `metrics.py`, `ranges.py`, `render_lines.py`,
`render_tui.py`, `show.py` (Phase 4), `train.py`, `launch.py`,
`data_load.py`, `rewards.py` and `tests/test_rewards.py` (Phase 5).
Follow-up source verification used TRL 1.13.0, transformers 5.17.0,
torch 2.13.0 and datasets 5.0.1. In-memory probes reproduced display failure
paths, metric reading/writing errors, rendering and resume output defects,
reward parsing/validation errors, and replay-count arithmetic. No live
training or GPU failure reproduction was performed in that verification;
remaining runtime questions are identified below. Findings remain outstanding
unless marked otherwise; line references are from the original review.

Tier 1, silently wrong training or destroyed work:

1. **Complete:** `train.py` isolates display failures from job
   supervision. A failed display stops and records its error in `log.txt`
   and usable stderr; training and verification continue with their exit
   status preserved, including after broken-stream shutdown flushes.
   Ctrl-C, job polling, verification startup, and authoritative-log failures
   retain supervisor cleanup. `tests/test_train.py`: 17 CPU-only tests pass
   with `python -B -m unittest tests.test_train -v`, including real pipe
   closure in child processes. Independent review found no blocking issues.
   **Completed in Phase 8:** `preflight.json` and `verify.json` use staged publication
   by default; explicit `--no-staging` selects direct writing.
2. `train.py:147` raises on a nonzero verify exit before loading results on
   that poll, closing the TUI. SPEC 2.4 says the TUI stays until quit.
3. `rewards.py:30` `_NUMBER` reads a word-internal hyphen as a minus sign, so
   an `llm_judge` reply naming a model scores negative: "As GPT-4, I rate
   this 9" gives -4.0. "Score: .5" gives 5.0. Neither reaches the
   no-number warning. Selecting the first number is the documented policy;
   fixing numeric token syntax alone would still select 4 instead of 9.
4. `rewards.py:140,163` iterate `required`, `forbidden` and `keys` without
   checking they are lists. `required = "hello"` becomes five
   single-character phrases and scores "h e l l o" 1.0. Non-string phrase
   elements raise `AttributeError`; JSON keys supplied as a string become
   character keys, while invalid elements can fail membership or raise
   `TypeError`, depending on their type.
5. `rewards.py:83` checks that a fuzzy `threshold` is numeric but not that it
   is in [0, 1], which the error message promises. `threshold = 5` starts
   cleanly and returns 0.0 for every completion for the whole run. The same
   argument is accepted and ignored in the other two modes.
6. `launch.py:110` `_estimate` never reads `cfg.teacher`, while
   `train.py:271` loads a second full model per rank for `distillation`. The
   estimate omits the teacher and can choose an unsuitable strategy or
   permit a run that OOMs; the outcome depends on the workload and hardware.
   SPEC 2.5 and the Multi-GPU design section above state the rule without
   the teacher, so the contract needs the fix too.

Tier 2, wrong output, lost information, or a traceback on operator input:

- Each built-in factory names its closure after the built-in kind. TRL's
  GRPO and RLOO trainers append same-name metrics into one list and average
  them, so two `phrases` entries share `rewards/phrases/mean` and cannot be
  addressed separately in `[ranges]`. Training retains separate weighted
  reward columns. The `rewards.py` docstring incorrectly claims separate names.
- `render_tui.py:120` sizes value and change columns as minimums, so an
  out-of-range or large value overflows `table_width`, passes the width
  guard, and is clipped at the terminal edge. The module docstring promises
  a size message instead of silent clipping.
- **Complete in Phase 8:** `metrics.py` rejects complete malformed final records;
  only an invalid JSON tail without a newline is skipped as a partial write.
- `render_tui.py:131` shows three checkpoint rows while `best` is computed
  over all of them, so the best checkpoint can be off-screen and unmarked.
- **Complete in Phase 8:** resumed line output skips historical log bytes and metric rows,
  while retaining earlier metrics for change-column calculations.
- `rewards.py:312` passes an unresolved bare name to TRL, which calls
  `AutoModelForSequenceClassification.from_pretrained`; the eventual error
  depends on the name and lacks trlx's config-entry context. `config.py:62`
  `_HF_ID` describes dataset references, not a general model-path contract.
  TRL exposes reward submodules as attributes, so `hasattr(trl.rewards, name)`
  also accepts modules without checking that they are callable.
- `data_load.py:75` floors the replay count at 1, so a small `fraction` on a
  small train set can deliver nine times the requested share with no line
  saying so. SPEC 2.9 states the equality without an exception.
- `data_load.py:100` treats an empty file as fatal but lets an empty hub
  split through.
- A local path with an unrecognised extension is treated as a hub reference
  when it matches `_HF_ID`: `data/input.txt` does, while `./data/input.txt`
  and `nested/data/input.txt` receive the supported-format error.
- Tracebacks on operator input: `data_load.py:83` when the train set already
  has a `replay` column (`Dataset.add_column` raises `ValueError` before
  trlx's handler); `rewards.py:116` for a non-integer or negative
  `group`; `rewards.py:240` for a non-numeric `timeout` or `retries`;
  `train.py:341` when `output_dir` names an existing file;
  `metrics.py:68` when a write or flush fails; and `train.py:187`, where a
  worker OOM or rendezvous timeout has no corresponding error handler.
  `train.py:253` handles OOM around standalone preflight, not rendezvous errors.
- `launch.py:242` kills workers with `SIGKILL` and no process group, so a
  worker's own dataloader children are never signalled and can outlive the
  run holding GPU memory, and surviving ranks cannot tear down NCCL.
  The direct-worker-only kill path is source-confirmed; retained GPU memory
  was not reproduced.
- `train.py:110` returns a negative exit code for a signal-killed worker,
  which wraps at the shell boundary; specifically, SIGKILL's -9 becomes 247.

Tier 3, contract and comment drift:

- SPEC 2.5 says other ranks train silently. `launch.py:142` sets
  `LOCAL_RANK=0` in every worker. With the default `log_on_each_node = true`,
  transformers keys `should_log` off the local index, so every rank logs
  into the shared `log.txt` with no rank prefix. Phase 5 recorded this as
  observed, so the spec sentence is what is wrong. Checkpoint saving is unaffected.
- Stale or contradictory comments: `metrics.py:36` states `logs["epoch"]` is
  a rounded copy, but transformers 5.17 assigns `state.epoch` directly;
  `launch.py:20` and `launch.py:74` disagree about whether the multiplier
  covers activations. The TUI's size guarantee concerns columns and panes;
  checkpoint, log and result text is truncated, with result truncation
  explicitly documented.
- Functions with no preceding comment: `metrics.py:107`, `metrics.py:60`,
  `metrics.py:71`, `ranges.py:32`, `render_tui.py:49`, `render_tui.py:77`,
  `render_tui.py:138`, `show.py:100`, `rewards.py:70`.
- `launch.py:36` `_POLL_SECONDS` and `train.py:51` `_FAILURE_TAIL_LINES` are
  operational constants not recorded in the session notes as accepted
  exceptions, unlike the four that are.
- `train.py:168` and `show.py:112` each scan the run directory for
  checkpoints; one source of truth would serve both.
- **Complete in Phase 8:** the line-mode log reader starts after the already-printed
  startup notice, so the strategy line is not duplicated.
- `normalise` removes ASCII punctuation only. Unicode punctuation can make
  otherwise equivalent punctuated and unpunctuated answers differ; matching
  punctuation on both sides still matches. The ASCII restriction is undocumented.

Test gaps named by the review:

- No test covers `show.py` or `render_tui` geometry. Width checks need to
  compare rendered cells with `table_width` and `_pair_width`; checkpoint
  visibility and display lifecycle defects need separate coverage.
- `llm_judge` number parsing is tested with one reply shape, so both
  parsing defects pass the suite. No unreachable-endpoint, timeout or
  retry-exhaustion test, and no `api_key` test in either direction.
- No empty-completion test for any built-in reward. `length_window` scores
  an empty string 0.889 with `low = 1, high = 10`, consistent with its
  documented linear falloff; this is a coverage gap, not an established defect.
- `reference_match` fuzzy has one assertion above one threshold; `regex` has
  no test for group 0, a negative group, a non-participating group or a
  wrongly typed group; `phrases` has no bare-string or all-forbidden case;
  `length_window` tests token-mode validation but not token counting;
  `resolve` has no unknown-bare-name test.

Additional verification and open decisions:

- `ranges.py:75` raises `ValueError` for a nonnumeric selected metric in an
  in-memory probe. Whether supported trainers naturally emit such values
  remains unverified.
- `launch.py:43` counts devices before `CUDA_VISIBLE_DEVICES` is rewritten.
  When NVML discovery fails, PyTorch falls back to a CUDA runtime call that
  initializes the driver. The ordering hazard is source-supported; incorrect
  mapping on an affected host still needs a controlled GPU reproduction.
- `SequenceMatcher`'s default autojunk heuristic changes fuzzy scores at
  length 200: `"x" + "a" * 199` versus `"y" + "a" * 199` yields 0.0,
  compared with 0.995 when autojunk is disabled. The current reward rejects
  that pair at threshold 0.8. The effect is reproduced; changing the
  matching policy requires approval.
- Changes to the judge's first-number policy, Unicode normalization, or
  empty-completion length scoring also require separate behavior decisions.

Reviewed and found clean: change-column arithmetic and the TUI row
geometry; interval boundaries including NaN; empty and absent files in the
readers; rank-0 ownership of `metrics.jsonl` and the preflight report; the
logical-to-physical GPU mapping, including under a pre-set
`CUDA_VISIBLE_DEVICES` (subject to the unresolved NVML case above); verify
gating on every worker exiting 0; zombie reaping; `[dataset]` split rules
and the eval remainder; reward resolution
order; `json_valid` key logic; `length_window` falloff arithmetic;
`llm_judge` concurrency, ordering and retry wiring; and the `api_key`
change made earlier today.

### Facts

- `SFTConfig` 1.13 defaults `gradient_checkpointing` to `True`, where
  `TrainingArguments` defaults it to `False`. A config that omits the key gets
  the trainer class's default, so preflight's `use_cache` warning fires on an
  sft run that never mentions checkpointing.

### Session notes

- The TUI was not re-run this session; display code is unchanged.
- Stock vLLM server used for `dataset chat`:
  `CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-0.6B --port 8000
  --gpu-memory-utilization 0.3 --max-model-len 2048`, ready when `GET /health`
  returns 200. It has no reasoning parser, which is why it returns reasoning
  inline.
