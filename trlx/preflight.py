"""Preflight checks (SPEC 2.6).

Checks are grouped by what they need, and each group runs where that is
available:

- `check_config`: config and filesystem only. The supervisor runs it before
  the run directory exists, and `trlx check` runs it first. Every check here
  is fatal, so nothing is written; a failure is a TrlxError.
- `check_trainer`: the built trainer, before training, rank 0. The trainer
  has wrapped the model with peft and tokenized the dataset, so these checks
  see what training will see. Zero LoRA targets and zero trainable
  parameters are fatal; the rest are warnings and facts.
- `check_offpolicy`: one forward pass per response under the starting
  model. Runs from a TrainerCallback at on_train_begin on every rank,
  because under FSDP the placed model is a collective and every rank must
  join the forward; rank 0 alone publishes the facts and warnings. Each rank
  can report its own progress through the reporter supplied by its worker.

A Report collects warnings and facts. Rank 0 writes it to preflight.json
after `check_trainer` and again after `check_offpolicy`, so the file is
complete once training starts. Unknown config keys and a missing [ranges]
block are not checked here: config.load rejects them before any of this
runs.
"""

import dataclasses
import json
import pathlib
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request

import torch

from dataset.io import DatasetError, write_text
from dataset.progress import stage
from dataset import failures
from dataset.endpoint import diagnostic_redactor
from trlx import TrlxError, cancellation, run_dirs, show

# Seconds allowed for the vLLM server health probe. A connection that takes
# longer is treated as unreachable; the same accepted display-constant
# exception as POLL_MS (PLAN.md session notes).
VLLM_PROBE_SECONDS = 5

# Route probed on the TRL vLLM server: one that only the TRL server serves
# (its client reads it before weight sync), so a stock vLLM server on the
# same port, which answers /health but has no weight sync, is told apart.
VLLM_PROBE_PATH = "/get_world_size"


# Warnings and facts from the checks that ran. `facts` is keyed by check
# name and holds whatever that check measured; preflight.json is the
# warnings list plus the facts, flat. `lines` is text not yet printed:
# each stage prints what it added and clears it, so the log shows every
# check in order without repeating earlier ones.
@dataclasses.dataclass
class Report:
    warnings: list = dataclasses.field(default_factory=list)
    facts: dict = dataclasses.field(default_factory=dict)
    lines: list = dataclasses.field(default_factory=list)

    def warn(self, text):
        self.warnings.append(text)
        self.lines.append(f"preflight warning: {text}")

    def note(self, text):
        self.lines.append(f"preflight: {text}")

    def to_dict(self):
        return {"warnings": list(self.warnings), **self.facts}

    # Prints pending lines to `out` and clears them.
    def flush(self, out=sys.stderr):
        for line in self.lines:
            print(line, file=out, flush=True)
        self.lines.clear()

    # Writes preflight.json, replacing an earlier write from a previous stage.
    def write(self, run_dir, *, no_staging=False, progress=None):
        path = pathlib.Path(run_dir) / show.PREFLIGHT_FILENAME
        try:
            # Both stages belong to the same run; updating its report is already authorized.
            write_text(path, json.dumps(self.to_dict(), indent=1) + "\n", force=True,
                       no_staging=no_staging, progress=progress)
        except DatasetError as e:
            raise TrlxError(str(e)) from e


# Config-only fatal checks. `config_path` is the operator's file and
# `strategy` the launcher's choice for this run, or None from `trlx check`,
# which makes no such choice.
def check_config(cfg, config_path, strategy, *, progress=None):
    with stage(progress, "checking preflight configuration") as activity:
        _check_save_strategy(cfg)
        _check_replay(cfg)
        _check_resume(cfg, config_path, strategy)
        _check_vllm(cfg, progress=activity)


# A run that never saves leaves nothing for verify or merge; better refused
# now than discovered after training.
def _check_save_strategy(cfg):
    if cfg.args.save_strategy == "no":
        raise TrlxError('save_strategy = "no" leaves no checkpoint to verify or merge; '
                        'use --save-strategy steps --save-steps 0 for a final checkpoint only, '
                        'or --save-strategy epoch to save each epoch')


# The replay KL term needs logits on every batch and one flag per batch row;
# liger skips the logits, and packing and padding_free join the examples of
# a batch into one row, so the per-example replay flag no longer lines up
# with the logits (SPEC 2.9).
def _check_replay(cfg):
    if cfg.replay is None or cfg.replay.kl_coef <= 0:
        return
    for field in ("use_liger_kernel", "packing", "padding_free"):
        if getattr(cfg.args, field, False):
            raise TrlxError(f"[replay].kl_coef > 0 is incompatible with {field} = true: "
                            "replay KL needs separate examples and their logits; "
                            f"use --no-{field.replace('_', '-')} to retain KL, "
                            "or --replay-kl-coef 0 for replay mixing without KL")


# Resume compares the effective inputs, including temporary CLI overrides;
# comparing the source file alone would miss a changed model or learning rate.
def _check_resume(cfg, config_path, strategy):
    if not cfg.args.resume_from_checkpoint:
        return
    resume = run_dirs.inspect_checkpoint(cfg.args.resume_from_checkpoint)
    snapshot_path = resume.directory / show.CONFIG_FILENAME
    try:
        with open(snapshot_path, "rb") as f:
            snapshot = tomllib.load(f)
    except FileNotFoundError as e:
        raise TrlxError(f"--resume-from-checkpoint requires the run's saved config, but {e.filename} "
                        "does not exist; select a checkpoint inside its original trlx run directory "
                        "with config.toml, or restore the missing snapshot")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as e:
        raise TrlxError(f"{snapshot_path}: cannot read snapshot: {e}")
    saved_method = snapshot.get("launch", {}).get("method")
    if saved_method != cfg.method.name:
        raise TrlxError(f"resume refused: {snapshot_path} is for {saved_method}, not {cfg.method.name}; "
                        "use the saved run's training subcommand, or select a checkpoint from "
                        f"a {cfg.method.name} run")
    diffs = compare_snapshot(cfg.document, snapshot, strategy)
    if diffs:
        raise TrlxError(
            f"resume refused: the run config differs from {snapshot_path}:\n  " + "\n  ".join(diffs)
            + "\nRemove conflicting CLI overrides to retain the saved settings; "
            "to train with different settings, start a fresh run without --resume-from-checkpoint."
        )


# Differences between the current config document and the run snapshot, as
# human-readable lines; empty means the same run. Set aside before comparing:
# `resume_from_checkpoint` on both sides (the operator must set it to resume
# at all), `output_dir` (the checkpoint selects the actual run), `[run]` controls,
# and `[assessment]` (observational only; quality series track changed criteria),
# the snapshot's `[launch]` table (compared on
# sharding: an FSDP checkpoint and an unsharded one differ in format, while
# single and ddp are both unsharded and a different GPU set resumes fine),
# and the snapshot's `run_name` when the current file has none (trlx
# recorded the resolved value). Everything else, including nested tables,
# must match value for value.
def compare_snapshot(current, snapshot, strategy):
    current = dict(current)
    snapshot = dict(snapshot)
    current.pop("resume_from_checkpoint", None)
    snapshot.pop("resume_from_checkpoint", None)
    current.pop("output_dir", None)
    snapshot.pop("output_dir", None)
    current.pop("run", None)
    snapshot.pop("run", None)
    current.pop("assessment", None)
    snapshot.pop("assessment", None)
    launch = snapshot.pop("launch", None) or {}
    if current.get("run_name") in (None, "None"):
        current.pop("run_name", None)
        snapshot.pop("run_name", None)
    diffs = _diff_tables(current, snapshot, "")
    saved_strategy = launch.get("strategy")
    if None not in (saved_strategy, strategy) and (saved_strategy == "fsdp") != (strategy == "fsdp"):
        diffs.append(f"strategy: snapshot {saved_strategy}, now {strategy}; sharded and unsharded checkpoints differ")
    return diffs


# Recursive key-by-key comparison; `prefix` is the dotted path of the table.
def _diff_tables(current, snapshot, prefix):
    diffs = []
    for key in sorted(set(current) | set(snapshot)):
        name = f"{prefix}{key}"
        if key not in current:
            diffs.append(f"{name}: in snapshot ({snapshot[key]!r}), not in the run config")
        elif key not in snapshot:
            diffs.append(f"{name}: in the run config ({current[key]!r}), not in snapshot")
        elif isinstance(current[key], dict) and isinstance(snapshot[key], dict):
            diffs.extend(_diff_tables(current[key], snapshot[key], f"{name}."))
        elif current[key] != snapshot[key]:
            diffs.append(f"{name}: snapshot {snapshot[key]!r}, now {current[key]!r}")
    return diffs


# grpo and rloo generate through the TRL vLLM server (SPEC 2.10), not stock
# vLLM (SPEC 5). Probed before the model loads so an absent or wrong server
# fails in seconds, not after a long load. The address is resolved the way
# GRPOTrainer resolves it: base_url when set, else host and port.
def _check_vllm(cfg, *, progress=None):
    if cfg.rewards is None:
        return
    base = cfg.args.vllm_server_base_url or f"http://{cfg.args.vllm_server_host}:{cfg.args.vllm_server_port}"
    try:
        sanitize = diagnostic_redactor(base)
    except ValueError:
        # Preserve the safe validation explanation without raw malformed URL text.
        raise TrlxError("vllm_server_base_url: invalid URL; supply the TRL vLLM server's HTTP(S) address") from None
    try:
        return _probe_vllm(base, progress=progress)
    except Exception as error:
        failures.redact(error, sanitize)
        raise


# Probe failures retain their underlying network cause under the caller's URL redactor.
def _probe_vllm(base, *, progress=None):
    url = base.rstrip("/") + VLLM_PROBE_PATH
    try:
        parsed = urllib.parse.urlsplit(base)
    except ValueError as e:
        raise TrlxError("vllm_server_base_url: invalid URL; supply the TRL vLLM server's HTTP(S) address") from e
    # Network exceptions can echo credentials as well as the URL itself.
    display_base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    display_url = display_base.rstrip("/") + VLLM_PROBE_PATH
    try:
        with stage(progress, f"probing TRL vLLM server {display_url}"):
            with urllib.request.urlopen(url, timeout=VLLM_PROBE_SECONDS) as response:
                status = response.status
    except urllib.error.HTTPError as e:
        raise TrlxError(
            f"{display_base} answered {e.code} on {VLLM_PROBE_PATH}: a server is listening but it is not the TRL vLLM "
            "server (start it with `trl vllm-serve`)"
        )
    except urllib.error.URLError as e:
        raise TrlxError(f"TRL vLLM server unreachable at {display_url}; check its address, network access, "
                        "and that `trl vllm-serve` is running") from e
    except OSError as e:
        raise TrlxError(f"TRL vLLM server unreachable at {display_url} ({type(e).__name__}); "
                        "check network access and that `trl vllm-serve` is running") from e
    except (ValueError, UnicodeError) as e:
        raise TrlxError("vllm_server_base_url: invalid request URL; check the address and credentials") from e
    if status != 200:
        raise TrlxError(f"TRL vLLM server at {display_url} answered {status}, not 200")


# Checks on the built trainer, before training. Fatal ones raise; the rest
# go to `report`.
def check_trainer(cfg, trainer, train_set, report, *, progress=None, profile=None):
    model = trainer.model
    tokenizer = _tokenizer(trainer.processing_class)
    with stage(progress, "checking trainable parameters and LoRA targets"):
        _check_targets(cfg, model, report)
    with stage(progress, "checking tokenizer and model cache settings"):
        _check_pad_token(tokenizer, report)
        _check_use_cache(cfg, model, report)
    projected = _check_truncation(cfg, train_set, trainer.processing_class, report, progress=progress, profile=profile)
    actual_rows = trainer.train_dataset.num_rows
    report.facts["assessment_validation"] = {"projected_rows": projected["effective_rows"], "prepared_rows": actual_rows}
    if projected["effective_rows"] is not None and actual_rows != projected["effective_rows"]:
        report.warn(f"assessment projected {projected['effective_rows']} prepared rows; the trainer prepared {actual_rows}. "
                    "Use the trainer count; projected update/token estimates need review")
    with stage(progress, "checking prepared training example"):
        _check_example(cfg, trainer, tokenizer, report)


# The tokenizer behind a processor; a text-only checkpoint's processor is
# the tokenizer itself.
def _tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


# LoRA targets and trainable parameters. The breakdown groups adapted
# modules by the path above the first numeric component, so a layer stack
# reads as one line ("model.language_model.layers: 448") and the operator
# sees at a glance which towers `target_modules` reached. Nothing here knows
# what a tower is; the names are the model's own (SPEC design rule).
def _check_targets(cfg, model, report):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if cfg.peft is not None:
        from peft.tuners.lora import LoraLayer

        groups = {}
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                group = _group_name(name)
                groups[group] = groups.get(group, 0) + 1
        if not groups:
            raise TrlxError(f"[peft].target_modules {cfg.peft.target_modules!r} resolve to no module in the model")
        report.facts["lora_targets"] = groups
        for group, count in groups.items():
            report.note(f"LoRA targets: {group}: {count} modules")
    if trainable == 0:
        raise TrlxError("trainable parameter count is zero")
    report.facts["parameters"] = {"trainable": trainable, "total": total}
    report.note(f"trainable parameters: {trainable:,} of {total:,}")


# Module path with peft's wrapper prefix removed, cut before the first
# numeric component; a path with no index is its parent path.
def _group_name(name):
    parts = name.split(".")
    if parts[:2] == ["base_model", "model"]:
        parts = parts[2:]
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[:i])
    return ".".join(parts[:-1])


# A missing pad token breaks batching; one equal to EOS hides the end of
# text from the loss mask.
def _check_pad_token(tokenizer, report):
    pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
    if pad is None:
        report.warn("tokenizer has no pad token")
    elif pad == eos:
        report.warn(f"pad token equals EOS ({tokenizer.pad_token!r}); padding and end of text are indistinguishable")


# `use_cache` lives on the text config of a multimodal model; transformers'
# `get_text_config` resolves it for any config.
def _check_use_cache(cfg, model, report):
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if cfg.args.gradient_checkpointing and getattr(text_config, "use_cache", False):
        report.warn("gradient_checkpointing with use_cache = true; the cache is wasted work under checkpointing")


# Reuse the full-scan preparation contract; packing, filtering, and token boundaries are method-specific.
def _check_truncation(cfg, train_set, processor, report, *, progress=None, profile=None):
    from trlx import data_profile

    replay = getattr(cfg, "replay", None)
    excluded = ("replay",) if replay is not None and replay.kl_coef > 0 else ()
    try:
        identity = data_profile.fingerprint(train_set, exclude_columns=excluded)
    except ValueError as error:
        identity = None
        report.warn(f"assessment source identity could not be verified: {error}")
    if profile is None or identity is None or profile["train"]["fingerprint"] != identity:
        report.note("profiling current worker inputs; no matching pre-run source fingerprint is available")
        profile = data_profile.scan(cfg, processor, train_set, None, progress=progress)
    measured = profile["train"]
    report.facts["dataset_profile"] = measured
    if measured["errors"]:
        report.warn(f"dataset preparation assessment has {len(measured['errors'])} unresolved rows; "
                    "see dataset_profile.errors in preflight.json")
    for key, action in (("truncated_rows", "truncated"), ("dropped_rows", "dropped"), ("zero_loss_rows", "left without loss-bearing tokens")):
        rows = measured[key]
        if rows:
            report.warn(f"assessment projects {len(rows)} of {measured['rows']} input rows {action} by {cfg.method.name} preparation")
    report.note(f"dataset preparation projection: {measured['effective_rows']} rows; "
                f"{measured['loss_tokens']} loss-bearing tokens (None means not established)")
    return measured


# First train row as the trainer prepared it, with its label mask: TRL
# writes -100 into `labels` for every token the loss ignores, so the trained
# tokens are the rest. sft only; it is the one method with a mask. A row
# with no trained token is the "no assistant tokens" warning of SPEC 2.6.
def _check_example(cfg, trainer, tokenizer, report):
    if cfg.method.name != "sft":
        return
    dataset = trainer.train_dataset
    if not {"input_ids", "labels"} <= set(dataset.column_names) or dataset.num_rows == 0:
        return
    row = dataset[0]
    ids = row["input_ids"]
    trained = [t for t, label in zip(ids, row["labels"]) if label != -100]
    example = {
        "tokens": len(ids),
        "trained_tokens": len(trained),
        "text": tokenizer.decode(ids),
        "trained_text": tokenizer.decode(trained),
    }
    # Full text remains available in preflight.json without flooding the terminal.
    report.facts["example"] = example
    report.note(f"first row: {example['tokens']} tokens, {example['trained_tokens']} trained")
    # Token counts, not decoded-text equality, establish whether the mask trains
    # every token; distinct token sequences can decode to identical text.
    if trained and len(trained) == len(ids):
        report.note("first row: all tokens trained")
    if not trained:
        report.warn("first row has no trained tokens in its label mask")


# Off-policy check for dpo and kto: mean per-token log-prob of each response
# under the starting model, over the first [preflight].rows train rows. A
# response below the threshold is one the starting model finds unlikely,
# which is what off-policy data looks like. Runs on every rank (see module
# docstring); the model is put in eval mode for the forwards and restored.
def check_offpolicy(cfg, model, processor, train_set, report, *, progress=None):
    from trlx import data_profile

    if "preflight" not in cfg.method.blocks:
        return
    if cfg.preflight is None:
        report.note("off-policy check skipped: no [preflight] block")
        return
    threshold = cfg.preflight.offpolicy_logp_per_token
    count = min(cfg.preflight.rows, train_set.num_rows)
    max_length = getattr(cfg.args, "max_length", None)
    keep_end = getattr(cfg.args, "truncation_mode", "keep_start") == "keep_end"
    scores, skipped = {}, {}
    was_training = model.training
    model.eval()
    try:
        with stage(progress, "checking off-policy responses", total=count, unit="rows") as activity, torch.no_grad():
            for row in train_set.select(range(count)):
                # Both responses of the preceding row finished on every rank.
                cancellation.checkpoint("off-policy row boundary")
                # Use the same concatenated tokenization and prefix slicing as training and the full scan.
                pairs = data_profile.response_pairs(processor, row, cfg.method.name)
                names = ("completion",) if len(pairs) == 1 else ("chosen", "rejected")
                for name, (prompt_ids, response_ids) in zip(names, pairs):
                    value = _mean_logp(model, prompt_ids, response_ids, max_length, keep_end)
                    if value is None:
                        skipped[name] = skipped.get(name, 0) + 1
                    else:
                        scores.setdefault(name, []).append(value)
                # All responses in the row have been checked; every rank must still
                # execute the same forwards because FSDP participates collectively.
                activity.advance()
    finally:
        model.train(was_training)
    cancellation.checkpoint("off-policy scoring end")
    facts = {"rows": count, "threshold": threshold}
    for name in ("chosen", "rejected", "completion"):
        values = scores.get(name)
        if values is None:
            continue
        below = sum(v < threshold for v in values)
        facts[name] = {"mean_logp_per_token": sum(values) / len(values), "below_threshold": below,
                       "scored": len(values), "no_response_tokens": skipped.get(name, 0)}
        line = f"{name}: mean per-token log-prob {facts[name]['mean_logp_per_token']:.3f} over {len(values)} rows"
        if skipped.get(name):
            line += f" ({skipped[name]} with no response tokens after max_length excluded)"
        if below:
            report.warn(f"off-policy: {below} of {len(values)} {name} responses score below {threshold} ({line})")
        else:
            report.note(f"off-policy check: {line}, none below {threshold}")
    report.facts["offpolicy"] = facts


# Mean log-prob per response token for one prompt and response, with the
# joined sequence cut at max_length as the trainer cuts it: the tail under
# keep_start, the head under keep_end. None when no response token survives
# the cut (the trainer drops such rows). The first token of the sequence has
# no predecessor to predict it from, so an empty prompt starts at token 1.
def _mean_logp(model, prompt_ids, response_ids, max_length, keep_end):
    ids = prompt_ids + response_ids
    start = len(prompt_ids)
    if max_length is not None and len(ids) > max_length:
        if keep_end:
            excess = len(ids) - max_length
            ids, start = ids[excess:], max(start - excess, 0)
        else:
            ids = ids[:max_length]
    start = max(start, 1)
    if len(ids) <= start:
        return None
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids=input_ids).logits[0, start - 1 : -1].float()
    targets = input_ids[0, start:]
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return logp.mean().item()


# TrainerCallback running check_offpolicy at on_train_begin on every rank.
# `report` is rank 0's; other ranks pass one that is discarded.
def callback_class():
    from transformers import TrainerCallback

    class PreflightCallback(TrainerCallback):
        # Each worker owns its progress reporter; report publication remains rank 0's responsibility.
        def __init__(self, cfg, processor, train_set, report, run_dir, rank, *, no_staging=False, progress=None):
            self.cfg = cfg
            self.processor = processor
            self.train_set = train_set
            self.report = report
            self.run_dir = run_dir
            self.rank = rank
            self.no_staging = no_staging
            self.progress = progress

        # Forward passes run on every rank; only the authoritative report is flushed and written once.
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            check_offpolicy(self.cfg, model, self.processor, self.train_set, self.report, progress=self.progress)
            if self.rank == 0:
                self.report.flush()
                self.report.write(self.run_dir, no_staging=self.no_staging, progress=self.progress)
            # The rank-zero publication completes before any subsequent callback.
            cancellation.checkpoint("preflight callback end")

    return PreflightCallback
