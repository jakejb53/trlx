"""Pure advisory rules over resolved settings, full profiles, and recorded metrics.

No rule owns training control or mutates its inputs. Projected schedules are
estimates until the trainer constructs its dataloader; observed metric changes
are evidence for comparison experiments, not causal diagnoses or optimal values.
"""

import copy
import math
import statistics


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
    findings.append(_finding("evaluation_coverage", "info", "projected",
                             "Evaluation rows and loss-token coverage are projected from the full dataset scan.",
                             {"train_rows": profile["train"].get("effective_rows"),
                              "eval_rows": evaluation.get("effective_rows") if evaluation else 0,
                              "eval_loss_tokens": evaluation.get("loss_tokens") if evaluation else None,
                              "eval_strategy": str(strategy)},
                             "Enable evaluation with suitable held-out data to assess generalization."
                             if not evaluation or strategy == "no" else None))
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


# Read the authoritative nested metrics payload; no [ranges] selection limits analysis.
def _series(records, key, warmup_steps=0):
    points = {}
    for record in records:
        step = record.get("step")
        value = record.get("log", {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            # Recovery starts a new observation interval; do not bridge a numerical
            # failure with earlier values or count repeated logs as extra updates.
            points.clear()
        if _finite(step) and step > warmup_steps and _finite(value):
            points[step] = (step, value, record.get("time"))
    return list(points.values())


# Persist raw observations as well as summaries so a recommendation can be audited.
def _observations(points):
    return {"step_range": [points[0][0], points[-1][0]],
            "time_range": [points[0][2], points[-1][2]],
            "values": [p[1] for p in points], "steps": [p[0] for p in points],
            "median": statistics.median(p[1] for p in points)}


# Two non-overlapping windows avoid comparing overlapping moving averages.
def _trend(points, window, relative_change):
    if len(points) < 2 * window:
        return None
    older, newer = points[-2 * window:-window], points[-window:]
    before, after = statistics.median(p[1] for p in older), statistics.median(p[1] for p in newer)
    delta = after - before
    relative = delta / abs(before) if before else None
    # A zero reference has no meaningful relative change; exact constancy remains
    # meaningful. Require every new observation beyond the old median for a trend.
    above = delta > 0 and all(p[1] > before for p in newer)
    below = delta < 0 and all(p[1] < before for p in newer)
    qualifies = relative is not None and abs(relative) >= relative_change
    direction = "up" if qualifies and above else "down" if qualifies and below else "mixed"
    return {"previous": _observations(older), "recent": _observations(newer),
            "relative_change": relative, "relative_change_threshold": relative_change,
            "threshold_meaning": "Operator-configured descriptive heuristic, not statistical confidence.",
            "direction": direction}


# Compare quality only against a baseline with the same explicit conditions hash.
def _quality_findings(records, min_evaluations, relative_change):
    groups = {}
    for record in records:
        context = record.get("quality") or {}
        # Failed or incomplete rounds can retain partial diagnostics, but none of
        # their scores establish a baseline or count toward quality trends.
        if context.get("series") and context.get("status") == "complete":
            groups.setdefault(context["series"], []).append(record)
    findings = []
    for series, rounds in groups.items():
        baseline = next((r for r in rounds if r["quality"].get("phase") == "baseline"), None)
        latest = rounds[-1]
        if baseline is None or latest is baseline:
            continue
        changes = {}
        for key, value in latest.get("log", {}).items():
            start = baseline.get("log", {}).get(key)
            if key.startswith("quality/") and _finite(start) and _finite(value):
                changes[key] = {"baseline": start, "current": value, "delta": value - start}
        if changes:
            findings.append(_finding(f"quality_baseline:{series}", "info", "measured",
                                     "Independent quality metrics can be compared with this run's matching baseline.",
                                     {"series": series, "baseline_step": baseline["step"], "current_step": latest["step"],
                                      "baseline_context": baseline["quality"], "current_context": latest["quality"],
                                      "metrics": changes},
                                     "Review task scores and individual results together; model-judge scores remain judgments, and diagnostics alone do not establish task quality."))
        # Only built-in metrics with known semantics support directional advice.
        # Unknown/custom metrics still receive the exact baseline deltas above.
        higher_better = {"exact_match", "token_f1", "accuracy", "json_valid", "required_fields_present",
                         "reference_values_correct", "judge_score", "judge_instruction_adherence",
                         "judge_coherence", "judge_task_quality", "judge_clarity", "preference_accuracy"}
        lower_better = {"loss", "perplexity", "invalid", "ambiguous"}
        for key in sorted(changes):
            name = key.removeprefix("quality/")
            if name not in higher_better | lower_better:
                continue
            points = _series(rounds, key, -1)
            if len(points) < min_evaluations:
                continue
            points = points[-min_evaluations:]
            first, last = points[0][1], points[-1][1]
            relative = (last - first) / abs(first) if first else None
            direction = -1 if name in higher_better else 1
            deteriorating = all(direction * (b[1] - a[1]) > 0 for a, b in zip(points, points[1:]))
            if relative is not None and direction * relative >= relative_change and deteriorating:
                findings.append(_finding(f"quality_deterioration:{series}:{name}", "warning", "heuristic",
                                         f"{key} repeatedly deteriorated under matching evaluation conditions.",
                                         {"series": series, "metric": key, **_observations(points),
                                          "relative_change": relative, "relative_change_threshold": relative_change,
                                          "min_evaluations": min_evaluations, "model_judgment": name.startswith("judge_")},
                                         "Inspect the individual held-out results; compare a shorter run or adjusted optimization in a separate experiment. Training reward increases do not override this independent evidence."))
    return findings


# Report sustained observations; never infer optimal settings, reward alignment, or causation.
def runtime_findings(method, records, *, window, min_evaluations, relative_change, warmup_steps=0, ranges=None):
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError("assessment window must be an integer of at least two metric observations")
    if not isinstance(min_evaluations, int) or isinstance(min_evaluations, bool) or min_evaluations < 2:
        raise ValueError("assessment min_evaluations must be an integer of at least two")
    if not _finite(relative_change) or relative_change <= 0:
        raise ValueError("assessment relative_change must be finite and positive")
    findings, latest = [], {}
    for record in records:
        for key, value in record.get("log", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                latest[key] = (value, record)
    # A recovered metric resolves the warning; other newly observed metrics do not
    # erase a still-unresolved non-finite observation from the authoritative stream.
    for key, (value, record) in sorted(latest.items()):
        if not math.isfinite(value):
            findings.append(_finding(f"nonfinite:{key}", "warning", "measured",
                                     f"{key} reported a non-finite value.",
                                     {"metric": key, "value": str(value), "step": record.get("step"), "time": record.get("time")},
                                     "Inspect the affected batch, precision, gradients, and scorer output before attributing this to learning rate."))
    training = [r for r in records if not r.get("eval") and not r.get("quality")]
    trends = {key: _trend(_series(training, key, warmup_steps), window, relative_change) for key in latest}
    loss = trends.get("loss")
    # On-policy loss tracks a changing distribution/objective and is not a monotone
    # measure of progress. Preference loss also cannot establish answer quality.
    if method in ("sft", "dpo", "kto", "reward") and loss and loss["direction"] == "up":
        evidence = {"loss": loss, "warmup_steps": warmup_steps}
        for key in ("grad_norm", "learning_rate"):
            if trends.get(key):
                evidence[key] = trends[key]
        findings.append(_finding("rising_training_loss", "warning", "heuristic",
                                 "Training loss increased across two post-warmup observation windows.", evidence,
                                 "Inspect data and gradient trends; if instability persists, compare a lower learning rate in a separate run. These observations do not identify an optimal rate."))
    if method in ("sft", "dpo", "kto", "reward") and loss:
        before = loss["previous"]["median"]
        values = loss["previous"]["values"] + loss["recent"]["values"]
        # A narrow range across every observation supports a descriptive plateau;
        # matching medians amid large oscillations do not establish stalled learning.
        flat = all(value == 0 for value in values) if before == 0 else all(
            abs(value - before) / abs(before) < relative_change for value in values)
        if flat:
            findings.append(_finding("little_loss_change", "info", "heuristic",
                                     "Logged training loss changed little across two post-warmup windows.",
                                     {"loss": loss, "warmup_steps": warmup_steps},
                                     "Compare held-out progress and the learning-rate schedule before changing duration or rate; a flat training objective alone does not establish convergence."))
    if method == "distillation" and loss and loss["direction"] != "mixed":
        findings.append(_finding("distillation_loss_trend", "info", "heuristic",
                                 "The logged distillation objective changed across post-warmup windows.",
                                 {"loss": loss, "warmup_steps": warmup_steps},
                                 "Compare teacher agreement and held-out task scores under matching conditions; the generated training distribution changes, so loss alone cannot establish student quality."))
    gradient = trends.get("grad_norm")
    if gradient and gradient["direction"] == "up":
        findings.append(_finding("rising_gradient_norm", "info", "heuristic",
                                 "Gradient norms increased across two post-warmup windows.", {"grad_norm": gradient},
                                 "Check loss, clipping, and batch composition together; growing norms alone do not prove divergence."))
    evaluation_points = _series([r for r in records if r.get("eval")], "eval_loss", warmup_steps)
    if method in ("sft", "dpo", "kto", "reward") and len(evaluation_points) >= min_evaluations:
        points = evaluation_points[-min_evaluations:]
        first, last = points[0][1], points[-1][1]
        relative = (last - first) / abs(first) if first else None
        # Align train windows to the evaluated interval; unrelated older training
        # improvements cannot support a current generalization-gap interpretation.
        aligned = [p for p in _series(training, "loss", warmup_steps) if points[0][0] <= p[0] <= points[-1][0]]
        aligned_trend = _trend(aligned, window, relative_change)
        consistently_worse = all(b[1] > a[1] for a, b in zip(points, points[1:]))
        if relative is not None and relative >= relative_change and consistently_worse:
            evidence = {"eval_loss": _observations(points), "relative_change": relative,
                        "relative_change_threshold": relative_change, "min_evaluations": min_evaluations}
            if aligned_trend and aligned_trend["direction"] == "down":
                evidence["training_loss"] = aligned_trend
                findings.append(_finding("possible_overfitting", "warning", "heuristic",
                                         "Training loss improved while held-out loss repeatedly worsened over the same interval.", evidence,
                                         "Check that evaluation data and masking/objective remain comparable; compare shorter training or stronger regularization in another run. This pattern can indicate overfitting."))
            else:
                findings.append(_finding("worsening_evaluation", "warning", "heuristic",
                                         "Held-out loss repeatedly worsened across the configured minimum evaluation count.", evidence,
                                         "Inspect held-out examples and training stability; this observation alone does not distinguish overfitting from other causes."))
    if method in ("grpo", "rloo", "distillation"):
        findings.extend(_generation_findings(training, trends, window, warmup_steps, method))
    for key, bounds in (ranges or {}).items():
        points = _series(records, key, warmup_steps)[-window:]
        if len(points) < window:
            continue
        low, high = bounds
        outside = (low is not None and all(p[1] < low for p in points)) or (high is not None and all(p[1] > high for p in points))
        if outside:
            findings.append(_finding(f"configured_range:{key}", "warning", "measured",
                                     f"{key} stayed outside its configured range for {window} observations.",
                                     {"metric": key, "configured_bounds": list(bounds), **_observations(points)},
                                     "Review the observations against the task; configured display ranges are operator guidance, not universal quality limits."))
    findings.extend(_quality_findings(records, min_evaluations, relative_change))
    return findings


# Exact boundaries (zero variance/cutoffs) describe signal health without hidden thresholds.
def _generation_findings(records, trends, window, warmup_steps, method):
    findings = []
    keys = {key for record in records for key in record.get("log", {})}
    for key in sorted(keys):
        points = _series(records, key, warmup_steps)[-window:]
        if len(points) < window:
            continue
        zero_std = key == "reward_std" or key.startswith("rewards/") and key.endswith("/std")
        if method in ("grpo", "rloo") and zero_std and all(p[1] == 0 for p in points):
            findings.append(_finding(f"constant_reward:{key}", "warning", "measured",
                                     f"{key} was zero throughout the observed window.",
                                     {"metric": key, **_observations(points)},
                                     "Inspect generated answers and this reward's requirements; a constant component cannot distinguish sampled answers, but other components may still provide signal."))
        if method in ("grpo", "rloo") and key == "frac_reward_zero_std" and any(p[1] > 0 for p in points):
            all_groups = all(p[1] == 1 for p in points)
            findings.append(_finding("zero_variance_groups", "warning" if all_groups else "info", "measured",
                                     "Some sampled prompt groups have no within-group reward variation." if not all_groups else
                                     "Every reported prompt group has zero reward variation throughout the observed window.",
                                     {"metric": key, **_observations(points)},
                                     "Inspect rewards and completions for constant scoring, uniformly solved/unsolved prompts, and insufficient sampling diversity. Reward variation does not establish reward quality."))
        if key == "completions/clipped_ratio" and any(p[1] > 0 for p in points):
            findings.append(_finding("generation_cutoffs", "warning" if all(p[1] == 1 for p in points) else "info", "measured",
                                     "Generated completions reached their token limit during the observed window.",
                                     {"metric": key, **_observations(points)},
                                     "Inspect clipped answers; compare a larger completion budget if useful answers are cut short, accounting for added memory and generation time."))
    # Drift, entropy, and clipping are interpreted together; none independently
    # proves collapse or bad quality. Missing evidence stays missing.
    evidence = {key: trend for key, trend in trends.items()
                if trend and (key in ("kl", "entropy") or key.startswith("clip_ratio/"))}
    if any(t["direction"] != "mixed" for t in evidence.values()):
        findings.append(_finding("policy_dynamics", "info", "heuristic",
                                 "Observed policy statistics changed across post-warmup windows.", evidence,
                                 "Review KL, entropy, policy clipping, and independent task scores together before comparing learning rate, regularization, or sampling settings. Direction alone does not establish improvement or collapse."))
    return findings
