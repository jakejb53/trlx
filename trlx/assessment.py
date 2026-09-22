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


# Evaluation and completion share one assessment of all durable observations.
def run_assessment(method, records, args, *, completed_steps, planned_steps, ranges=None,
                   checkpoint_steps=None, checkpoint_errors=()):
    warmup = args.get_warmup_steps(planned_steps)
    findings = runtime_findings(method, records, warmup_steps=warmup, ranges=ranges)
    training = [r for r in records if not r.get("eval") and not r.get("quality")]
    evaluation = [r for r in records if r.get("eval") and not r.get("quality")]
    invalid = [{"step": r.get("step"), "metric": key, "value": str(value)}
               for r in records for key, value in r.get("log", {}).items()
               if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value)]
    # A later finite value does not erase an earlier numerical failure from the run's conclusion.
    if invalid:
        findings.append(_finding("final_numerical_issues", "warning", "measured",
                                 "Non-finite metric values occurred during this run.", {"observations": invalid},
                                 "Inspect the affected steps before using this checkpoint; later recovery does not invalidate the earlier failure."))
    final_train = next((r["log"]["train_loss"] for r in reversed(training) if "train_loss" in r["log"]), None)
    latest_eval = evaluation[-1] if evaluation else None
    # Preserve exact evidence, but exclude throughput and cumulative counters from the conclusion.
    eval_values = {key: value for key, value in (latest_eval["log"] if latest_eval else {}).items() if key.startswith("eval_")
                   and key not in {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second", "eval_num_tokens"}}
    report = {"method": method, "completed_steps": completed_steps, "planned_steps": planned_steps,
            "train_loss": final_train, "evaluation_records": len(evaluation), "evaluation_metrics": eval_values,
            "evaluation_step": latest_eval["step"] if latest_eval else None,
            "metric_records": len(records), "nonfinite_observations": len(invalid), "findings": findings}
    report.update(_run_recommendation(method, findings, training, evaluation, args, completed_steps, planned_steps,
                                     records, checkpoint_steps, checkpoint_errors))
    return report


# Metric summaries preserve small nonzero values instead of displaying a false zero.
def _display_number(value):
    return f"{value:.6g}"


# Changes retain enough precision to distinguish a small measured gain from zero.
def _change_text(first, last):
    delta = last - first
    relative = f", {100 * delta / abs(first):+.3g}%" if first else ""
    return f"{_display_number(first)} → {_display_number(last)} (change {delta:+.6g}{relative})"


# Recent descriptions name the actual interval; fitted direction is not practical significance.
def _recent_text(trend, label):
    if trend is None:
        return f"No finite {label} observations."
    recent = trend.get("post_warmup", trend["recent"])
    if recent["observations"] < 2:
        return f"Too little recent {label} evidence to establish a direction."
    start, end = recent["step_range"]
    values = recent["values"]
    direction = {"down": "falling", "up": "rising", "mixed": "mixed", "flat": "within numerical tolerance",
                 "limited": "one interval only", "unknown": "undetermined"}[trend["direction"]]
    shape = " The change per optimizer step has slowed." if trend["diminishing"] else ""
    return (f"Recent {label}, steps {start}–{end}: {_change_text(values[0], values[-1])}; "
            f"{direction}.{shape}")


# Schedule observations qualify attribution; changing duration can also change the LR trajectory.
def _schedule_context(training, args, completed_steps, planned_steps, trend):
    warmup = args.get_warmup_steps(planned_steps)
    if warmup and completed_steps <= warmup:
        return f"Warmup is still active ({completed_steps}/{warmup} steps); post-warmup behavior is not established."
    recent = trend.get("post_warmup", trend["recent"]) if trend else {}
    start = recent.get("step_range", [0])[0]
    rates = [p for p in _series(training, "learning_rate") if start <= p[0] <= completed_steps]
    if len(rates) < 2:
        return None
    first, last = rates[0][1], rates[-1][1]
    if all(point[1] == first for point in rates):
        return f"Recorded learning rate stayed at {_display_number(last)} over steps {rates[0][0]}–{rates[-1][0]}."
    if first == last:
        return (f"Recorded learning rate varied from {_display_number(min(p[1] for p in rates))} to "
                f"{_display_number(max(p[1] for p in rates))} over steps {rates[0][0]}–{rates[-1][0]}; "
                "equal endpoints do not establish a constant schedule.")
    direction = "fell" if last < first else "rose"
    return (f"Recorded learning rate {direction} from {_display_number(first)} to {_display_number(last)} "
            f"over steps {rates[0][0]}–{rates[-1][0]}; loss changes alone cannot separate schedule effects from learning saturation.")


# Only the latest quality series is relevant to current advice; failed rounds never become scores.
def _quality_context(records, findings, completed_steps):
    rounds = [r for r in records if r.get("quality")]
    if not rounds:
        return [], False, False, False
    latest = rounds[-1]
    context = latest["quality"]
    series = context.get("series")
    if context.get("status") != "complete":
        return [f"Independent quality check at step {latest['step']} was incomplete or failed; no current quality conclusion."], False, False, True
    comparisons = [f for f in findings if f["code"] == f"quality_baseline:{series}"]
    if not comparisons:
        return [f"Independent quality at step {latest['step']} has no completed comparison against a matching baseline yet."], False, False, True
    evidence = comparisons[-1]["evidence"]
    higher, lower = QUALITY_HIGHER_BETTER, QUALITY_LOWER_BETTER
    lines, better, worse = [], False, False
    for key, change in evidence["metrics"].items():
        name = key.removeprefix("quality/")
        if name not in higher | lower:
            continue
        gain = change["delta"] if name in higher else -change["delta"]
        # This tolerance prevents numerical artifacts, not a claim of practical significance.
        tolerance = 1e-6 * max(1., abs(change["baseline"]), abs(change["current"]))
        better |= gain > tolerance
        worse |= gain < -tolerance
        # Aggregate judge score is accompanied only by criteria that moved against it.
        if name.startswith("judge_") and name != "judge_score" and gain >= 0:
            continue
        lines.append(f"{key}, steps {evidence['baseline_step']}–{evidence['current_step']}: "
                     f"{_change_text(change['baseline'], change['current'])}." +
                     (" Model judgment." if name.startswith("judge_") else ""))
    current_findings = [f for f in findings if f.get("evidence", {}).get("series") == series]
    worse |= any(f["code"].startswith("quality_deterioration:") for f in current_findings)
    stale = latest["step"] != completed_steps
    if stale:
        lines.append(f"Independent quality was last measured at step {latest['step']}, not step {completed_steps}.")
    return lines, better, worse, stale or not lines


# Availability is supplied from verified artifacts, never inferred from an evaluation step number.
def _checkpoint_context(points, checkpoint_steps, checkpoint_errors):
    if not points:
        return None
    best = min(points, key=lambda p: p[1])
    text = f"Lowest measured evaluation loss: {_display_number(best[1])} at step {best[0]}."
    if best[0] == 0:
        text += " This is the starting-model baseline."
    elif checkpoint_steps is None:
        text += " Saved checkpoint availability has not been checked."
    elif best[0] in checkpoint_steps:
        text += f" Checkpoint-{best[0]} is available for comparison."
    else:
        text += " No verified saved checkpoint is available at that measured step."
    if checkpoint_steps is not None:
        candidates = [p for p in points if p[0] in checkpoint_steps]
        if candidates:
            saved = min(candidates, key=lambda p: p[1])
            if saved[0] != best[0]:
                text += f" Lowest measured loss among available checkpoints: {_display_number(saved[1])} at step {saved[0]}."
    if checkpoint_errors:
        text += " Checkpoint inspection: " + "; ".join(checkpoint_errors)
    return text


# Keep finding qualifications and attach the measurements that support each warning.
def _assessment_issues(findings, active_quality_series):
    issues = []
    for finding in findings:
        code, evidence = finding["code"], finding.get("evidence") or {}
        if code.startswith("nonfinite:"):
            continue  # The run-wide numerical finding preserves both current and recovered failures.
        if code.startswith("quality_") and evidence.get("series") != active_quality_series:
            continue
        signal = code in {"generation_cutoffs", "zero_variance_groups"}
        if signal and not evidence.get("values", [0])[-1]:
            continue
        if finding["severity"] not in {"warning", "error"} and not signal:
            continue
        support = None
        if "metric" in evidence and evidence.get("values"):
            support = f"{evidence['metric']} at step {evidence['steps'][-1]}: {_display_number(evidence['values'][-1])}."
            if "configured_bounds" in evidence:
                support += f" Configured range: {evidence['configured_bounds']}."
        elif evidence.get("eval_loss"):
            support = _recent_text(evidence["eval_loss"], "evaluation loss")
        elif evidence.get("loss"):
            support = _recent_text(evidence["loss"], "training loss")
        elif evidence.get("trend"):
            support = _recent_text(evidence["trend"], evidence.get("metric", "metric"))
        elif evidence.get("observations") and code == "final_numerical_issues":
            support = "; ".join(f"{r['metric']}={r['value']} at step {r['step']}" for r in evidence["observations"])
        issues.append({"code": code.split(":")[0],
                       "message": finding["summary"] + (" " + finding["recommendation"] if finding.get("recommendation") else ""),
                       "support": support})
    issues.sort(key=lambda item: 0 if item["code"] == "final_numerical_issues" else 1)
    return issues


# Compare total outcome, recent trajectory, and independent evidence before choosing an action.
# No observed slope identifies an optimal duration or learning-rate value.
def _run_recommendation(method, findings, training, evaluation, args, completed_steps, planned_steps,
                        records, checkpoint_steps, checkpoint_errors):
    fixed = method in {"sft", "dpo", "kto", "reward"}
    key = "eval_loss" if fixed else "loss" if method == "distillation" else "reward"
    label = "evaluation loss" if fixed else "distillation training loss" if method == "distillation" else "training reward"
    source = evaluation if fixed else training
    points = _series(source, key)
    trend = _metric_trend(source, key, args.get_warmup_steps(planned_steps))
    baseline = next((p for p in points if p[0] == 0), None) if fixed else None
    quality, quality_better, quality_worse, quality_uncertain = _quality_context(records, findings, completed_steps)
    quality_records = [r for r in records if r.get("quality")]
    active_series = quality_records[-1]["quality"].get("series") if quality_records else None
    issues = _assessment_issues(findings, active_series)
    direction = trend["direction"] if trend else "unknown"
    recent = _recent_text(trend, label) if points else None
    outcome = f"No finite {label} measurements were recorded."
    if points:
        first = baseline or points[0]
        prefix = "Baseline → latest" if baseline else "First → latest measured"
        outcome = f"{prefix} {label}, steps {first[0]}–{points[-1][0]}: {_change_text(first[1], points[-1][1])}."
        if len(points) == 1:
            outcome = f"{'Baseline' if baseline else 'First measured'} {label}: {_display_number(first[1])} at step {first[0]}."
        if fixed and baseline is None:
            outcome += " No step-zero baseline; total training gain is unknown."
    schedule = _schedule_context(training, args, completed_steps, planned_steps, trend)
    checkpoint = _checkpoint_context(points, checkpoint_steps, checkpoint_errors) if fixed else None
    # A conflicting training trend remains visible even when it is not an instability warning.
    for finding in findings:
        if finding["code"] == "rising_training_loss" and finding["severity"] == "info":
            recent = (recent + " " if recent else "") + _recent_text(finding["evidence"]["loss"], "training loss")
            recent += " Training and held-out losses describe different samples; these measurements do not establish instability."
    # Auxiliary held-out scores can disagree with the optimized loss; never hide that conflict.
    auxiliary = {"sft": "eval_mean_token_accuracy", "dpo": "eval_rewards/accuracies", "reward": "eval_accuracy"}.get(method)
    auxiliary_worse = False
    if auxiliary:
        scores = _series(evaluation, auxiliary)
        if len(scores) > 1:
            quality.append(f"{auxiliary}, steps {scores[0][0]}–{scores[-1][0]}: {_change_text(scores[0][1], scores[-1][1])}.")
            auxiliary_worse = scores[-1][1] < scores[0][1]
    final = "Review held-out task results before deciding whether to train longer."
    live = "Monitor the next evaluation before changing settings; recent evidence is limited."
    if fixed and points:
        improved = baseline is not None and points[-1][1] < baseline[1]
        if len(points) == 1 and baseline is not None:
            live = "Starting-model baseline recorded. Training can begin."
            final = "Only the starting-model baseline was measured. Evaluate the trained model before judging the result."
        elif direction == "up":
            live = final = "Recent held-out loss is rising. Compare measured earlier results before extending training."
        elif direction == "down":
            live = "Held-out loss is falling. Monitor its measured gains through the planned evaluations."
            if trend["diminishing"]:
                final = "Held-out loss improved with diminishing recent gains. Retain the result for task-level comparison; do not extend on these gains alone."
            else:
                final = ("Held-out loss improved. Compare the measured recent gain against task quality before choosing a longer-run experiment; "
                         "the slope alone does not establish a useful extension.")
        elif direction in {"flat", "mixed"}:
            live = "Recent held-out results are flat or mixed. Inspect the measurements before changing settings."
            final = "Retain the measured results for comparison; there is no clear recent loss trend supporting longer training."
        if baseline is None:
            final = "No starting-model baseline is available. Compare measured checkpoints, but do not infer total training improvement or an optimal duration."
        elif len(points) > 1 and not improved:
            final = "The latest held-out loss did not beat the starting model. Compare the baseline and earlier measured results; do not increase duration on this loss evidence."
    elif not fixed:
        live = f"Assess task scores alongside {label}; its movement alone does not justify changing settings."
        final = f"Compare held-out task results before selecting a model or increasing duration; {label} alone is insufficient."
    elif (quality_records and quality_records[-1]["quality"].get("status") == "complete" and
          any(f["code"] == f"quality_baseline:{active_series}" for f in findings)):
        live = final = "Use the independent quality measurements for task-specific comparison; ordinary evaluation loss is unavailable."
    else:
        live = final = "No completed held-out comparison is available. Obtain suitable evaluation before judging improvement or increasing duration."
    # Quality regressions override favorable optimization metrics, without declaring their cause.
    if quality_worse or auxiliary_worse:
        live = final = "The measured scores conflict or show deterioration. Compare the affected task results and available checkpoints before extending training."
    elif quality_better and not quality_uncertain:
        if fixed and (direction == "up" or (baseline is not None and points[-1][1] > baseline[1])):
            live = final = "Independent task scores improved while held-out loss deteriorated recently or relative to baseline. Compare both objectives before choosing a checkpoint or changing settings."
        else:
            live = "Independent task scores improved. Monitor the planned evaluations and the size of further gains."
            final = "Independent task scores improved. Retain the measured result for comparison; further training is an experiment, not an established benefit."
    if quality_uncertain and quality_records:
        final += " Obtain a complete matching quality comparison for the endpoint before relying on task-quality advice."
    if any(issue["code"] == "final_numerical_issues" for issue in issues):
        live = final = "Investigate the recorded non-finite metrics before relying on this run's results; later finite values do not erase the failure."
    # Independent scores cannot make an unmeasured ordinary-loss endpoint appear measured.
    if fixed and points and points[-1][0] != completed_steps:
        final += f" Ordinary loss at step {completed_steps} was not evaluated; measure it before drawing a final loss conclusion."
    # Training completion and premature exit never emit advice to wait for another in-run evaluation.
    if completed_steps < planned_steps:
        final = f"Training ended at {completed_steps}/{planned_steps} updates. Review the stopping reason before increasing the budget. " + final
    else:
        live = final
    if completed_steps == 0 and planned_steps > 0 and baseline is not None:
        live = "Starting-model baseline recorded. Training can begin." if not issues else live
    if completed_steps >= planned_steps or "experiment" in final:
        scheduler = getattr(args, "lr_scheduler_type", None)
        scheduler = getattr(scheduler, "value", scheduler)
        if scheduler not in (None, "constant", "constant_with_warmup"):
            schedule = (schedule + " " if schedule else "") + "Changing duration can change the learning-rate trajectory even with the same configured learning rate."
    support = " ".join(part for part in (outcome, recent, schedule) if part)
    return {"decision": live, "support": support, "issues": issues,
            "final_decision": final, "final_support": support, "final_issues": list(issues),
            "outcome": outcome, "recent": recent, "schedule": schedule, "quality": quality, "checkpoint": checkpoint}


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


# Read the authoritative nested metrics payload; no [ranges] selection limits analysis.
def _series(records, key):
    points = {}
    for record in records:
        step = record.get("step")
        value = record.get("log", {}).get(key)
        # Invalid values are reported separately, not grounds to erase earlier
        # history. Repeated logs at one optimizer step are not new observations.
        if _finite(step) and _finite(value):
            points[step] = (step, value, record.get("time"))
    return [points[step] for step in sorted(points)]


# Persist raw observations as well as summaries so a recommendation can be audited.
def _observations(points):
    return {"step_range": [points[0][0], points[-1][0]],
            "time_range": [points[0][2], points[-1][2]],
            "values": [p[1] for p in points], "steps": [p[0] for p in points],
            "median": statistics.median(p[1] for p in points)}


# Weighted least squares separates drift from observed scatter. Recency weights
# decay over the recorded step span: the oldest point retains 1/16 the newest's
# weight. This is a descriptive heuristic, never a statistical confidence claim.
def _fit(points, *, recent=False):
    if len(points) < 2:
        return {"direction": "unknown", "observations": len(points)}
    span = points[-1][0] - points[0][0]
    xs = [(point[0] - points[0][0]) / span for point in points]
    ys = [point[1] for point in points]
    weights = [2 ** (4 * (x - 1)) if recent else 1.0 for x in xs]
    total = sum(weights)
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    variance = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    change = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys)) / variance
    noise = math.sqrt(sum(w * (y - mean_y - change * (x - mean_x)) ** 2
                          for w, x, y in zip(weights, xs, ys)) / total)
    # Two points establish only a difference. With more evidence, require drift
    # larger than twice residual scatter; a numerical tolerance avoids noise-floor claims.
    tolerance = 1e-6 * max(1.0, abs(statistics.median(ys)))
    direction = "limited" if len(points) == 2 else (
        "flat" if max(ys) - min(ys) <= tolerance else
        ("up" if change > 0 else "down") if abs(change) > max(2 * noise, tolerance) else "mixed")
    if direction in {"up", "down"}:
        changes = [b - a for a, b in zip(ys, ys[1:])]
        signs = [1 if delta > tolerance else -1 for delta in changes if abs(delta) > tolerance]
        sign = 1 if direction == "up" else -1
        supporting = sum(delta * sign > tolerance for delta in changes)
        # A fitted slope driven by one outlier is not sustained movement. Require
        # two corroborating changes, a majority of intervals, and agreement in
        # the latest two nonzero changes. This is descriptive, not significance.
        if supporting < 2 or supporting <= len(changes) / 2 or signs[-2:] != [sign, sign]:
            direction = "mixed"
    return {"direction": direction, "observations": len(points), "fitted_change": change, "residual_scatter": noise}


# Overall history remains auditable, but recent evidence covers at most three
# successive intervals. Early large gains must not mask a late reversal.
def _trend(points, warmup_steps=0, *, recent_points=None):
    if not points:
        return None
    segment = points if recent_points is None else recent_points
    recent = segment[-4:]
    result = {"history": _observations(points), "overall": _fit(points),
              "recent": {**(_observations(recent) if recent else {}), **_fit(recent)}}
    current = result["recent"]
    if warmup_steps > 0:
        segment = [point for point in segment if point[0] > warmup_steps]
        after = segment[-4:]
        current = {**(_observations(after) if after else {}), **_fit(after)}
        result["post_warmup"] = current
        result["warmup_updates"] = warmup_steps
        recent = after
    intervals = [{"step_range": [a[0], b[0]], "delta": b[1] - a[1],
                  "rate": (b[1] - a[1]) / (b[0] - a[0])}
                 for a, b in zip(recent, recent[1:])]
    rates = [interval["rate"] for interval in intervals]
    result["intervals"] = intervals
    # This describes the shape, not whether gains are practically worthwhile.
    result["diminishing"] = (len(rates) == 3 and
                             (all(rate > 0 for rate in rates) or all(rate < 0 for rate in rates)) and
                             all(abs(a) > abs(b) for a, b in zip(rates, rates[1:])))
    if len(segment) >= 7:
        prior = segment[-7:-3]
        previous_rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(prior, prior[1:])]
        combined = previous_rates + rates
        # Compare adjacent windows as well as successive intervals: a short final
        # interval can fluctuate while all recent rates remain below earlier ones.
        result["rate_comparison"] = {"previous": previous_rates, "recent": rates,
                                     "previous_step_range": [prior[0][0], prior[-1][0]],
                                     "recent_step_range": [recent[0][0], recent[-1][0]]}
        result["diminishing"] |= ((all(rate > 0 for rate in combined) or all(rate < 0 for rate in combined)) and
                                  max(abs(rate) for rate in rates) < min(abs(rate) for rate in previous_rates))
    result["direction"] = current["direction"]
    return result


# Preserve the full finite history while restarting recent evidence after invalid data.
def _metric_trend(records, key, warmup_steps=0):
    points = _series(records, key)
    latest_invalid = None
    for record in records:
        step = record.get("step")
        value = record.get("log", {}).get(key)
        if (_finite(step) and isinstance(value, (int, float)) and
                not isinstance(value, bool) and not math.isfinite(value)):
            latest_invalid = step if latest_invalid is None else max(step, latest_invalid)
    # Recovery begins a new recent segment; earlier finite values still document
    # the baseline and full history but cannot establish uninterrupted progress.
    segment = points if latest_invalid is None else [p for p in points if p[0] > latest_invalid]
    return _trend(points, warmup_steps, recent_points=segment)


# Shared metric semantics keep measured comparisons and trend advice consistent.
QUALITY_HIGHER_BETTER = {"exact_match", "token_f1", "accuracy", "json_valid", "required_fields_present",
                         "reference_values_correct", "judge_score", "judge_instruction_adherence",
                         "judge_coherence", "judge_task_quality", "judge_clarity", "preference_accuracy"}
QUALITY_LOWER_BETTER = {"loss", "perplexity", "invalid", "ambiguous"}


# Compare quality only against a baseline with the same explicit conditions hash.
def _quality_findings(records):
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
        for key in sorted(changes):
            name = key.removeprefix("quality/")
            if name not in QUALITY_HIGHER_BETTER | QUALITY_LOWER_BETTER:
                continue
            trend = _metric_trend(rounds, key)
            worsening = "down" if name in QUALITY_HIGHER_BETTER else "up"
            if trend and trend["direction"] == worsening:
                findings.append(_finding(f"quality_deterioration:{series}:{name}", "warning", "heuristic",
                                         f"{key} repeatedly deteriorated under matching evaluation conditions.",
                                         {"series": series, "metric": key, "trend": trend, **changes[key],
                                          "model_judgment": name.startswith("judge_")},
                                         "Inspect the individual held-out results; compare a shorter run or adjusted optimization in a separate experiment. Training reward increases do not override this independent evidence."))
            elif trend and trend["direction"] == ("up" if name in QUALITY_HIGHER_BETTER else "down"):
                findings.append(_finding(f"quality_improvement:{series}:{name}", "info", "heuristic",
                                         f"{key} shows a recent improving trend under matching evaluation conditions.",
                                         {"series": series, "metric": key, "trend": trend, **changes[key],
                                          "model_judgment": name.startswith("judge_")},
                                         "This supports improvement on the measured task; inspect individual results and conflicting scores before generalizing to other tasks."))
    return findings


# Sparse or noisy evidence limits a conclusion, never whether an assessment appears.
def _trend_summary(label, trend):
    if trend is None:
        return f"{label}: no finite observations were recorded."
    direction = trend["direction"]
    if direction in {"unknown", "limited"}:
        phase = "post-warmup " if "post_warmup" in trend else ""
        return f"{label}: too little {phase}history to establish a sustained trend yet."
    descriptions = {"up": "rising", "down": "falling", "flat": "nearly unchanged", "mixed": "noisy, with no clear direction"}
    overall = trend["overall"]["direction"]
    text = f"{label}: the recent trend is {descriptions[direction]}."
    if trend.get("diminishing"):
        text += " Changes per optimizer step show diminishing magnitude."
    if overall in {"up", "down"} and direction in {"up", "down"} and overall != direction:
        text += f" The overall history is {descriptions[overall]}, so the recent direction has reversed."
    return text


# Interpret full ordered histories with shared trend rules; recommendations remain advisory.
def runtime_findings(method, records, *, warmup_steps=0, ranges=None):
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
    trends = {key: _metric_trend(training, key, warmup_steps) for key in latest}
    loss = trends.get("loss")
    # On-policy loss tracks a changing distribution/objective and is not a monotone
    # measure of progress. Preference loss also cannot establish answer quality.
    fixed_objective = method in {"sft", "dpo", "kto", "reward"}
    if fixed_objective:
        rising = loss is not None and loss["direction"] == "up"
        evidence = {"loss": loss, "warmup_updates": warmup_steps}
        for key in ("grad_norm", "learning_rate"):
            if trends.get(key):
                evidence[key] = trends[key]
        findings.append(_finding("rising_training_loss" if rising else "training_progress", "warning" if rising else "info", "heuristic",
                                 _trend_summary("Training loss", loss), evidence,
                                 "Inspect batch composition, gradients, and the learning-rate schedule; persistent instability can justify comparing a lower learning rate."
                                 if rising else "Interpret training loss alongside held-out results; fitting the training objective does not establish general task quality."))
    else:
        findings.append(_finding("generated_objective", "info", "heuristic",
                                 _trend_summary("Training objective", loss), {"loss": loss},
                                 "This method trains on generated responses. Interpret reward, KL, clipping, and held-out results together; training loss alone cannot establish model improvement."))
    gradient = trends.get("grad_norm")
    if gradient and gradient["direction"] == "up":
        findings.append(_finding("rising_gradient_norm", "info", "heuristic",
                                 "Gradient norms show a rising recent trend.", {"grad_norm": gradient},
                                 "Check loss, clipping, and batch composition together; growing norms alone do not prove divergence."))
    evaluation = [r for r in records if r.get("eval") and not r.get("quality")]
    evaluation_trend = _metric_trend(evaluation, "eval_loss", warmup_steps)
    if fixed_objective and loss and loss["direction"] == "up" and evaluation_trend and evaluation_trend["direction"] == "down":
        for finding in findings:
            if finding["code"] == "rising_training_loss":
                finding["severity"] = "info"
                finding["summary"] += " Held-out loss is falling; these measurements do not establish instability."
                finding["evidence"]["eval_loss"] = evaluation_trend
                finding["recommendation"] = "Inspect batch composition and objective comparability before changing optimization; training and evaluation losses describe different samples."
    if not evaluation:
        quality_available = any((r.get("quality") or {}).get("status") == "complete" for r in records)
        findings.append(_finding("evaluation_missing", "info" if quality_available else "warning", "measured", "No ordinary evaluation metrics were recorded.", {},
                                 "Independent quality evaluations are available; ordinary held-out loss was not measured."
                                 if quality_available else "Review --eval-strategy and the held-out dataset configuration."))
    elif fixed_objective:
        evidence = {"eval_loss": evaluation_trend}
        if evaluation_trend and evaluation_trend["direction"] == "up":
            # Fit every training observation in the evaluation window, rather
            # than silently comparing a shorter tail with the evaluation trend.
            # An invalid training value prevents this uninterrupted comparison.
            interval = evaluation_trend.get("post_warmup", evaluation_trend["recent"])["step_range"]
            aligned = [r for r in training if _finite(r.get("step")) and interval[0] <= r["step"] <= interval[1]]
            aligned_trend = _metric_trend(aligned, "loss", warmup_steps)
            aligned_points = _series(aligned, "loss")
            invalid = any(isinstance(r.get("log", {}).get("loss"), (int, float)) and
                          not _finite(r["log"]["loss"]) for r in aligned)
            aligned_fit = _fit(aligned_points) if not invalid else {"direction": "unknown"}
            if aligned_trend and aligned_fit["direction"] == "down":
                evidence["training_loss"] = aligned_trend
                evidence["aligned_training_fit"] = {**_observations(aligned_points), **aligned_fit,
                                                    "evaluation_step_range": interval}
                findings.append(_finding("possible_overfitting", "warning", "heuristic",
                                         "Training loss is falling within the recent evaluation interval while held-out loss is rising.", evidence,
                                         "Check that evaluation data and masking/objective remain comparable; compare shorter training or stronger regularization in another run. This pattern can indicate overfitting."))
            else:
                findings.append(_finding("worsening_evaluation", "warning", "heuristic",
                                         _trend_summary("Held-out loss", evaluation_trend), evidence,
                                         "Inspect held-out examples and training stability; this observation alone does not distinguish overfitting from other causes."))
        else:
            findings.append(_finding("evaluation_progress", "info", "heuristic", _trend_summary("Held-out loss", evaluation_trend), evidence,
                                     "One evaluation measures performance but cannot establish a trend. Decreasing held-out loss supports progress on this objective, not unmeasured tasks."))
    if method in ("grpo", "rloo", "distillation"):
        findings.extend(_generation_findings(training, trends, method))
    for key, bounds in (ranges or {}).items():
        points = _series(records, key)
        if not points:
            continue
        low, high = bounds
        outside = (low is not None and points[-1][1] < low) or (high is not None and points[-1][1] > high)
        if outside:
            findings.append(_finding(f"configured_range:{key}", "warning", "measured",
                                     f"The latest {key} is outside its configured range.",
                                     {"metric": key, "configured_bounds": list(bounds), **_observations(points)},
                                     "Review the observations against the task; configured display ranges are operator guidance, not universal quality limits."))
    findings.extend(_quality_findings(records))
    return findings


# Exact boundaries (zero variance/cutoffs) describe signal health without hidden thresholds.
def _generation_findings(records, trends, method):
    findings = []
    keys = {key for record in records for key in record.get("log", {})}
    for key in sorted(keys):
        points = _series(records, key)
        if not points:
            continue
        zero_std = key == "reward_std" or key.startswith("rewards/") and key.endswith("/std")
        if method in ("grpo", "rloo") and zero_std and points[-1][1] == 0:
            findings.append(_finding(f"constant_reward:{key}", "warning", "measured",
                                     f"The latest {key} is zero; this reward currently provides no within-group variation.",
                                     {"metric": key, **_observations(points)},
                                     "Inspect generated answers and this reward's requirements; a constant component cannot distinguish sampled answers, but other components may still provide signal."))
        if method in ("grpo", "rloo") and key == "frac_reward_zero_std" and any(p[1] > 0 for p in points):
            all_groups = all(p[1] == 1 for p in points)
            findings.append(_finding("zero_variance_groups", "warning" if all_groups else "info", "measured",
                                     "Some sampled prompt groups have no within-group reward variation." if not all_groups else
                                     "Every reported prompt group has zero reward variation throughout the recorded history.",
                                     {"metric": key, **_observations(points)},
                                     "Inspect rewards and completions for constant scoring, uniformly solved/unsolved prompts, and insufficient sampling diversity. Reward variation does not establish reward quality."))
        if key == "completions/clipped_ratio" and any(p[1] > 0 for p in points):
            findings.append(_finding("generation_cutoffs", "warning" if all(p[1] == 1 for p in points) else "info", "measured",
                                     "Generated completions reached their token limit during this run.",
                                     {"metric": key, **_observations(points)},
                                     "Inspect clipped answers; compare a larger completion budget if useful answers are cut short, accounting for added memory and generation time."))
    # Drift, entropy, and clipping are interpreted together; none independently
    # proves collapse or bad quality. Missing evidence stays missing.
    evidence = {key: trend for key, trend in trends.items()
                if trend and (key in ("kl", "entropy") or key.startswith("clip_ratio/"))}
    if any(t["direction"] in {"up", "down"} for t in evidence.values()):
        findings.append(_finding("policy_dynamics", "info", "heuristic",
                                 "Policy statistics show directional changes in the recorded history.", evidence,
                                 "Review KL, entropy, policy clipping, and independent task scores together before comparing learning rate, regularization, or sampling settings. Direction alone does not establish improvement or collapse."))
    return findings
