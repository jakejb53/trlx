"""Startup assessment rules and factual runtime metric reports.

No rule owns training control or mutates its inputs. Projected schedules are
estimates until the trainer constructs its dataloader. Runtime reports expose
recorded measurements without interpreting learning or recommending settings.
"""

import copy
import math


# Keep the persisted contract shared by static and runtime observations.
def _finding(code, severity, basis, summary, evidence, recommendation=None):
    return dict(code=code, severity=severity, basis=basis, summary=summary,
                evidence=evidence, recommendation=recommendation)


# Model metadata can be a serialized config or the config object itself.
def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


# Booleans and non-finite values are not usable quantitative observations.
def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# Match GRPOConfig/RLOOConfig generation resolution using the selected worker count.
# Supervisor args have already resolved automatic fields with world_size=1, so
# only the original explicit document can distinguish operator input from defaults.
def _generation_schedule(cfg, gpu_count):
    explicit = cfg.document
    generation_batch = explicit.get("generation_batch_size")
    steps = explicit.get("steps_per_generation")
    generation_batch = None if generation_batch in (None, "None") else generation_batch
    steps = None if steps in (None, "None") else steps
    global_batch = cfg.args.per_device_train_batch_size * gpu_count
    if generation_batch is not None and steps is not None:
        return None, None, "generation_batch_size and steps_per_generation are mutually exclusive"
    if generation_batch is None:
        steps = cfg.args.gradient_accumulation_steps if steps is None else steps
        generation_batch = global_batch * steps
    elif generation_batch % global_batch:
        return None, None, "generation_batch_size must be divisible by the selected global per-step batch"
    else:
        steps = generation_batch // global_batch
    if generation_batch <= 0 or steps <= 0:
        return None, None, "generation_batch_size and steps_per_generation must be positive"
    if generation_batch % cfg.args.num_generations:
        return None, None, "generation_batch_size must be divisible by num_generations"
    return generation_batch, steps, None


# Describe the exact mathematical consequence without claiming token-equivalent batches.
def _batch_budget(cfg, profile, gpu_count):
    args, method = cfg.args, cfg.method.name
    physical = args.per_device_train_batch_size
    accumulation = args.gradient_accumulation_steps
    batch = physical * gpu_count * accumulation
    rows = profile["train"].get("effective_rows")
    findings = [_finding("effective_batch", "info", "projected",
                         f"A full optimizer update spans {batch} batch entries across {gpu_count} workers.",
                         {"per_device_batch": physical, "data_parallel_workers": gpu_count,
                          "gradient_accumulation_steps": accumulation, "effective_batch": batch,
                          "limitation": "Assumes one data-parallel worker per selected GPU (DDP/FSDP). Entries are "
                          "packed blocks, preference examples, or generated completions depending on the trainer; "
                          "partial batches and repeated prompts do not represent distinct examples."})]
    sampler_rows = rows
    multiplier = 1
    dataloader_batch = physical * gpu_count
    if method in ("grpo", "rloo", "distillation") and rows is not None:
        # These installed trainers discard incomplete RepeatSampler chunks before
        # Accelerate shards the dataloader. Accumulation cannot rescue a short chunk.
        if method == "distillation":
            generation_batch, generations = batch, 1
            steps_per_generation = accumulation
            multiplier = accumulation
            dataloader_batch *= accumulation
        else:
            generation_batch, steps_per_generation, error = _generation_schedule(cfg, gpu_count)
            generations = args.num_generations
            if error is not None:
                sampler_rows = None
                findings.append(_finding("generation_schedule_incompatible", "warning", "projected", error,
                                         {"data_parallel_workers": gpu_count, "per_device_batch": physical,
                                          "num_generations": generations,
                                          "generation_batch_size": cfg.document.get("generation_batch_size"),
                                          "steps_per_generation": cfg.document.get("steps_per_generation")},
                                         "Review the explicit generation settings for the selected worker count; a valid prompt-group projection is unavailable."))
            else:
                multiplier = generations * args.num_iterations * steps_per_generation
                dataloader_batch *= steps_per_generation
        if generation_batch is not None:
            group = generation_batch // generations
            sampler_rows = rows - rows % group
            findings.append(_finding(
                "prompt_groups", "warning" if sampler_rows < rows else "info", "projected",
                f"Generation uses {group} distinct prompts per full group; {rows - sampler_rows} prompts are dropped per epoch.",
                {"prepared_rows": rows, "prompts_per_group": group, "retained_prompts": sampler_rows,
                 "discarded_prompts": rows - sampler_rows, "generations_per_prompt": generations,
                 "generation_batch_size": generation_batch, "steps_per_generation": steps_per_generation,
                 "limitation": "Dropped identities depend on sampler order; repeated generations are not new prompts."},
                "Reduce the generation batch or supply more prompts if discarding these rows is undesirable."
                if sampler_rows < rows else None))
    updates_per_epoch = None
    if sampler_rows is not None:
        microbatches = (sampler_rows * multiplier / dataloader_batch)
        microbatches = math.floor(microbatches) if getattr(args, "dataloader_drop_last", False) else math.ceil(microbatches)
        updates_per_epoch = math.ceil(microbatches / accumulation)
        if updates_per_epoch == 0:
            findings.append(_finding("empty_training_batches", "warning", "projected",
                                     "The prepared data cannot produce a training batch.",
                                     {"effective_rows": rows, "retained_sampler_rows": sampler_rows,
                                      "dataloader_drop_last": getattr(args, "dataloader_drop_last", False)},
                                     "Supply more usable rows or reduce the batch/group size before training."))
    max_steps = getattr(args, "max_steps", -1)
    steps = max_steps if max_steps > 0 else (
        math.ceil(updates_per_epoch * args.num_train_epochs) if updates_per_epoch is not None else None)
    if steps is not None:
        warmup = args.get_warmup_steps(steps)
        findings.append(_finding(
            "training_budget", "info", "projected",
            f"Training is projected to run {steps} optimizer updates, including {warmup} warmup updates.",
            {"updates": steps, "updates_per_epoch": updates_per_epoch,
             "budget_source": "max_steps" if max_steps > 0 else "num_train_epochs",
             "num_train_epochs": args.num_train_epochs, "max_steps": max_steps, "warmup_updates": warmup,
             "limitation": "Trainer dataloader construction is authoritative; distributed padding, custom sampling, "
             "resume position, and automatic batch-size changes can alter this full-run projection."}))
        if warmup >= steps and steps > 0:
            findings.append(_finding("warmup_entire_run", "warning", "projected",
                                     "Warmup occupies the entire projected training budget.",
                                     {"warmup_updates": warmup, "total_updates": steps},
                                     "Review warmup_steps against the intended learning-rate schedule and run length."))
    return findings


# Evaluation and completion report recorded facts only; no trainer settings or controls are changed.
def run_metrics_report(method, records, *, completed_steps, planned_steps, ranges=None):
    records = [r for r in records if _finite(r.get("step")) and 0 <= r["step"] <= completed_steps]
    recap, recent = _metric_comparisons(method, records, ranges)
    notices = [f"{key}: non-finite value {value} at step {r['step']}."
               for r in records for key, value in r.get("log", {}).items()
               if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value)]
    quality = [r for r in records if r.get("quality")]
    if quality and quality[-1]["quality"].get("status") != "complete":
        notices.append(f"Independent quality check at step {quality[-1]['step']} was incomplete or failed.")
    curves = {}
    for name, key, evaluation in (("training", "loss", False), ("evaluation", "eval_loss", True)):
        # A non-finite observation is a gap, not permission to connect across a failed measurement.
        # Last recorded value wins at repeated steps, as it does in the metric recap.
        points = {r["step"]: r["log"][key] if _finite(r["log"][key]) else None
                  for r in records if bool(r.get("eval")) == evaluation and not r.get("quality")
                  and key in r.get("log", {})}
        curves[name] = sorted(points.items())
    return dict(method=method, completed_steps=completed_steps, planned_steps=planned_steps,
                recap=recap, recent_recap=recent, curves=curves, notices=notices)


# A recap preserves the actual first/last records, including failed endpoints.
# Filtering to finite values here would silently substitute older successful measurements.
def _metric_row(records, key, label, *, percentage=False, previous_step=None):
    observations = {r["step"]: r for r in records if (key in r.get("log", {}) or r.get("quality"))
                    and _finite(r.get("step"))}
    ordered = [observations[step] for step in sorted(observations)]
    initial = ordered[0] if ordered else None
    if previous_step is not None:
        # Before the first evaluation, training has no step-zero loss; name its first logged step instead.
        initial = next((r for r in reversed(ordered) if r["step"] <= previous_step), initial)
    latest = ordered[-1] if ordered else None
    # Failed quality rounds retain diagnostics but never contribute a score.
    def value(record):
        if record is None or (record.get("quality") and record["quality"].get("status") != "complete"):
            return None
        raw = record.get("log", {}).get(key)
        return raw if _finite(raw) else None

    first, last = value(initial), value(latest)
    delta = last - first if (first is not None and last is not None
                            and initial["step"] != latest["step"]) else None
    return dict(metric=key, label=label, initial=first, latest=last,
                initial_step=initial["step"] if initial else None,
                latest_step=latest["step"] if latest else None,
                initial_kind="baseline" if initial and ((initial.get("eval") and initial["step"] == 0) or
                             initial.get("quality", {}).get("phase") == "baseline") else "first logged",
                delta=delta, relative_change=delta / abs(first) if delta is not None and first else None,
                percentage=percentage)


# Select relevant measured metrics and their display units, without judging their direction.
def _metric_comparisons(method, records, ranges):
    training = [r for r in records if not r.get("eval") and not r.get("quality")]
    evaluation = [r for r in records if r.get("eval") and not r.get("quality")]
    eval_steps = sorted({r["step"] for r in evaluation if _finite(r.get("step"))})
    previous = eval_steps[-2] if len(eval_steps) > 1 else None
    definitions = [("loss", "Training loss", training, False),
                   ("eval_loss", "Evaluation loss", evaluation, False)]
    extras = {
        "sft": [("mean_token_accuracy", "Training token accuracy", True),
                ("eval_mean_token_accuracy", "Evaluation token accuracy", True)],
        "dpo": [("eval_rewards/accuracies", "Evaluation preference accuracy", True),
                ("rewards/accuracies", "Training preference accuracy", True),
                ("rewards/margins", "Training reward margin", False)],
        "kto": [("rewards/chosen", "Chosen reward", False),
                ("rewards/rejected", "Rejected reward", False), ("kl", "KL", False)],
        "reward": [("eval_accuracy", "Evaluation accuracy", True),
                   ("accuracy", "Training accuracy", True)],
        "grpo": [("reward", "Training reward", False), ("reward_std", "Reward spread", False),
                 ("kl", "KL", False)],
        "rloo": [("reward", "Training reward", False), ("reward_std", "Reward spread", False),
                 ("kl", "KL", False)],
        "distillation": [],
    }
    for key, label, percentage in extras[method]:
        source = evaluation if key.startswith("eval_") else training
        if any(key in r.get("log", {}) for r in source):
            definitions.append((key, label, source, percentage))
    known = {entry[0] for entry in definitions}
    for key in ranges or {}:
        # Infrastructure and step diagnostics are already in the live metrics tables.
        if key in known or key in {"grad_norm", "learning_rate", "epoch", "num_tokens"} or key.startswith("quality/"):
            continue
        source = evaluation if key.startswith("eval_") else training
        if any(key in r.get("log", {}) for r in source):
            definitions.append((key, key, source, False))
    quality = [r for r in records if r.get("quality")]
    if quality:
        active = quality[-1]["quality"].get("series")
        matching = [r for r in quality if r["quality"].get("series") == active]
        baseline = next((r for r in matching if r["quality"].get("phase") == "baseline"
                         and r["quality"].get("status") == "complete"), None)
        # A new unmatched series cannot borrow an older series' baseline.
        source = matching[matching.index(baseline):] if baseline else [matching[-1]]
        # Quality writers also emit sample/token accounting; those are not performance scores.
        keys = sorted({key for r in source for key in r.get("log", {}) if key.startswith("quality/")
                       and key not in {"quality/rows", "quality/tokens", "quality/scored_rows"}
                       and not key.startswith("quality/metric_rows/")
                       and not (key.startswith("quality/class/") and key.endswith("/rows"))})
        for key in keys:
            name = key.removeprefix("quality/")
            percent = (name in {"accuracy", "exact_match", "token_f1", "json_valid", "required_fields_present"}
                       or (name.startswith("class/") and name.endswith("/accuracy")))
            definitions.append((key, key, source, percent))
    recap, recent = [], []
    for key, label, source, percentage in definitions:
        row = _metric_row(source, key, label, percentage=percentage)
        if key.startswith("eval_") and row["initial_step"] != 0:
            row.update(initial=None, initial_step=None, delta=None, relative_change=None)
        if key.startswith("quality/") and (not source or source[0].get("quality", {}).get("phase") != "baseline"):
            row.update(initial=None, initial_step=None, delta=None, relative_change=None)
        recap.append(row)
        recent.append(_metric_row(source, key, label, percentage=percentage,
                                  previous_step=previous) if previous is not None else dict(row))
    return recap, recent


# Report method-specific contracts of the installed trainers, not universal tuning targets.
def _objective(cfg, profile):
    args, method, train = cfg.args, cfg.method.name, profile["train"]
    findings = []
    physical = args.per_device_train_batch_size
    if method == "kto":
        kl = args.loss_type != "apo_zero_unpaired"
        if kl and (physical <= 1 or args.train_sampling_strategy != "sequential"):
            findings.append(_finding(
                "kto_kl_batch", "warning", "projected",
                "KTO's KL estimator requires a per-device batch above one and sequential sampling.",
                {"per_device_train_batch_size": physical, "gradient_accumulation_steps": args.gradient_accumulation_steps,
                 "train_sampling_strategy": args.train_sampling_strategy, "loss_type": args.loss_type},
                "Use a physical batch above one and sequential sampling for this loss; accumulation does not satisfy the requirement."))
        rows = train.get("effective_rows")
        workers = getattr(args, "dataset_num_proc", None)
        if kl and rows is not None and physical > 1 and workers not in (None, 1):
            # Dataset.map partitions preprocessing across workers before batching;
            # the full split's remainder does not identify per-partition singletons.
            findings.append(_finding("kto_parallel_kl_groups", "info", "projected",
                                     "KTO rotates KL completions within each preprocessing worker's batches.",
                                     {"effective_rows": rows, "preparation_batch_size": physical,
                                      "dataset_num_proc": workers,
                                      "limitation": "Partition boundaries determine singleton self-pairs; aggregate profile counts cannot establish their number."},
                                     "Inspect prepared KL pairs if the selected batch and preprocessing partitioning leave singleton groups."))
        if kl and rows is not None and physical > 1 and workers in (None, 1) and rows % physical == 1:
            findings.append(_finding("kto_singleton_kl", "warning", "projected",
                                     "The final KTO preparation group contains one row and pairs its KL completion with itself.",
                                     {"effective_rows": rows, "preparation_batch_size": physical},
                                     "Review the physical batch size or usable row count; KL pair rotation happens during preparation."))
        counts = train.get("label_counts")
        # The profiler preserves successful-row label counts when other rows fail;
        # those partial counts cannot establish the complete training class balance.
        if counts and not train.get("errors"):
            positive, negative = counts["desirable"], counts["undesirable"]
            findings.append(_finding("kto_weighted_balance", "warning" if not positive or not negative else "info", "projected",
                                     "KTO label counts and class weights determine the relative contributions before model-dependent losses.",
                                     {"label_counts": counts, "desirable_weight": args.desirable_weight,
                                      "undesirable_weight": args.undesirable_weight,
                                      "weighted_desirable_count": positive * args.desirable_weight,
                                      "weighted_undesirable_count": negative * args.undesirable_weight},
                                     "Review both label populations and weights against the intended preference objective; equal weighting is not a universal optimum."))
    elif method == "dpo":
        losses = args.loss_type if isinstance(args.loss_type, (tuple, list)) else [args.loss_type]
        if set(losses) & {"aot", "aot_unpaired"}:
            findings.append(_finding("dpo_aot_batch", "warning" if physical == 1 else "info", "projected",
                                     f"AOT's distribution sorting sees {physical} preference pairs per device, not the accumulated batch.",
                                     {"loss_type": losses, "per_device_batch": physical,
                                      "gradient_accumulation_steps": args.gradient_accumulation_steps},
                                     "Compare a larger physical batch if distribution-level alignment is intended; accumulation cannot enlarge the sorted sample."))
        smoothing = getattr(args, "label_smoothing", 0)
        unsupported = sorted(set(losses) - {"robust", "exo_pair", "aot", "aot_unpaired"})
        if smoothing and unsupported and not getattr(args, "use_liger_kernel", False):
            findings.append(_finding("dpo_inactive_smoothing", "warning", "projected",
                                     "Some selected DPO losses do not use label_smoothing in the installed trainer.",
                                     {"label_smoothing": smoothing, "losses_ignoring_smoothing": unsupported},
                                     "Choose a loss that implements the intended smoothing or remove this ineffective setting."))
    elif method == "reward" and train.get("dropped_rows"):
        findings.append(_finding("reward_filtered_pairs", "warning", "projected",
                                 "Reward preparation is projected to remove overlength preference pairs instead of truncating them.",
                                 {"dropped_rows": train["dropped_rows"], "effective_rows": train.get("effective_rows"),
                                  "max_length": args.max_length},
                                 "Inspect excluded pairs for systematic coverage changes before changing max_length."))
    return findings


# Combine full-scan evidence with consequences of resolved settings; never infer task intent.
def static_findings(cfg, profile, gpu_count, *, model_metadata=None, teacher_metadata=None):
    if not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("assessment gpu_count must be a positive data-parallel worker count")
    findings = copy.deepcopy(profile.get("findings", []))
    findings.extend(_batch_budget(cfg, profile, gpu_count))
    findings.extend(_objective(cfg, profile))
    args = cfg.args
    sources = profile["train"].get("sources")
    if sources and getattr(cfg, "replay", None) is not None:
        # Row sampling fraction and loss-token exposure are different quantities;
        # packed mixed-source blocks cannot attribute their shifted losses exactly.
        findings.append(_finding("replay_exposure", "info", "projected",
                                 "Selected replay rows and token exposure describe the data balance before model-dependent losses.",
                                 {"sources": copy.deepcopy(sources), "configured_fraction": cfg.replay.fraction,
                                  "kl_coef": cfg.replay.kl_coef,
                                  "limitation": "Unknown loss-token attribution remains null; row fractions do not imply equal loss contributions."},
                                 "Compare primary-task and replay held-out results before adjusting replay fraction or KL regularization."))
    evaluation = profile.get("eval")
    strategy = getattr(args, "eval_strategy", "no")
    synthetic = getattr(cfg.dataset, "synthetic_dataset_eval", False)
    # An uninspected synthetic set is pending, not evidence that evaluation is off.
    disabled = (not evaluation and not synthetic) or strategy == "no"
    findings.append(_finding("evaluation_coverage", "warning" if disabled else "info", "projected",
                             "Ordinary evaluation is disabled; its held-out metrics will not be recorded."
                             if disabled else ("Synthetic evaluation data inspection is skipped; summaries belong to this run."
                                               if synthetic else "Evaluation rows and loss-token coverage are projected from the full dataset scan."),
                             {"train_rows": profile["train"].get("effective_rows"),
                              "eval_rows": evaluation.get("effective_rows") if evaluation else (None if synthetic else 0),
                              "eval_loss_tokens": evaluation.get("loss_tokens") if evaluation else None,
                              "eval_strategy": str(strategy)},
                             "Enable evaluation with suitable held-out data to assess generalization."
                             if disabled else None))
    peft = getattr(cfg, "peft", None)
    if peft is not None:
        rank, alpha = peft.r, peft.lora_alpha
        rslora = getattr(peft, "use_rslora", False)
        rank_pattern, alpha_pattern = getattr(peft, "rank_pattern", {}), getattr(peft, "alpha_pattern", {})
        findings.append(_finding("lora_scale", "info", "projected",
                                 "LoRA adapter scaling follows alpha/sqrt(rank)." if rslora else "LoRA adapter scaling follows alpha/rank.",
                                 {"rank": rank, "alpha": alpha, "use_rslora": rslora,
                                  "default_scale": alpha / (math.sqrt(rank) if rslora else rank) if rank > 0 else None,
                                  "rank_pattern": rank_pattern, "alpha_pattern": alpha_pattern,
                                  "limitation": "Patterns override individual layer scales; adapter variants and initialization affect behavior. "
                                  "This factor is not a learning-rate optimum or a measurement of trainable capacity."}))
    # Configuration capacity is a boundary, not a guarantee of useful long-context quality.
    maximum = _get(model_metadata, "max_position_embeddings")
    if _finite(maximum) and maximum > 0:
        if cfg.method.name in ("grpo", "rloo", "distillation"):
            prompt = profile["train"].get("prompt_lengths", {}).get("max")
            prompt_cap = getattr(args, "max_prompt_length", None)
            if prompt is not None and prompt_cap is not None:
                prompt = min(prompt, prompt_cap)
            completion = getattr(args, "max_completion_length", None)
            requested = prompt + completion if prompt is not None and completion is not None else None
        else:
            requested = profile["train"].get("prepared_lengths", {}).get("max")
        if requested is not None and requested > maximum:
            findings.append(_finding("model_context_budget", "warning", "projected",
                                     "The prepared or generation sequence budget exceeds the model's declared position count.",
                                     {"requested_tokens": requested, "max_position_embeddings": maximum,
                                      "rope_scaling": _get(model_metadata, "rope_scaling")},
                                     "Check the model's documented context/scaling support before changing sequence or generation limits."))
    if cfg.method.name == "distillation":
        student_vocab, teacher_vocab = _get(model_metadata, "vocab_size"), _get(teacher_metadata, "vocab_size")
        if student_vocab is not None and teacher_vocab is not None and student_vocab != teacher_vocab:
            findings.append(_finding("distillation_vocab_size", "warning", "measured",
                                     "Student and teacher declare different vocabulary sizes for token-level distillation.",
                                     {"student_vocab_size": student_vocab, "teacher_vocab_size": teacher_vocab},
                                     "Use compatible token-ID meanings and output vocabularies; matching sizes alone do not prove compatibility."))
    return findings
