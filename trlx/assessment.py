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
def run_assessment(method, records, args, *, completed_steps, planned_steps, ranges=None):
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
    report.update(_run_recommendation(method, findings, training, evaluation, args, completed_steps, planned_steps))
    return report


# Metric summaries preserve small nonzero values instead of displaying a false zero.
def _display_number(value):
    return f"{value:.3g}" if value and abs(value) < .001 else f"{value:.3f}"


# Select operator actions from established findings; routine observations remain
# available in the report and raw metrics without competing with the recommendation.
def _run_recommendation(method, findings, training, evaluation, args, completed_steps, planned_steps):
    messages = {
        "final_numerical_issues": "Numerical instability detected. Check the failed steps and precision settings before using this checkpoint.",
        "possible_overfitting": "Likely overfitting. Shorten the next run or return to an earlier checkpoint.",
        "worsening_evaluation": "Evaluation loss is getting worse. Don’t extend this run; compare an earlier checkpoint.",
        "rising_training_loss": "Training loss is rising. Try a lower learning rate on the next run.",
        "evaluation_missing": "No evaluation results. Enable evaluation before the next run.",
        "generation_cutoffs": "Responses are being cut short. Increase --max-completion-length.",
        "zero_variance_groups": "Some prompt groups produce identical rewards. Review reward scoring and response diversity.",
    }
    issues = []
    for finding in findings:
        code = finding["code"]
        evidence = finding.get("evidence") or {}
        # A resolved clipping/variance observation needs no current corrective action.
        if code in {"generation_cutoffs", "zero_variance_groups"} and not evidence.get("values", [0])[-1]:
            continue
        if code in messages:
            issues.append({"code": code, "message": messages[code]})
        elif code.startswith("constant_reward:"):
            metric = evidence["metric"]
            issues.append({"code": "constant_reward", "message": f"{metric} is zero. Review reward scoring and response diversity."})
        elif code.startswith("quality_deterioration:"):
            issues.append({"code": "quality_deterioration", "message": f"{evidence['metric']} is deteriorating. Shorten the run or compare an earlier checkpoint."})
        elif code.startswith("configured_range:"):
            metric, value = evidence["metric"], evidence["values"][-1]
            bounds = evidence["configured_bounds"]
            issues.append({"code": "configured_range", "message": f"{metric} is {_display_number(value)}, outside {bounds}. Review this result and its configured range."})
        elif finding["severity"] in {"error", "warning"} and not code.startswith("nonfinite:"):
            issues.append({"code": code, "message": finding["summary"]})
    # Numerical failures take priority; a duplicate current-nonfinite notice adds
    # no action beyond the full-run numerical finding already selected above.
    priority = {"final_numerical_issues": 0, "possible_overfitting": 1, "worsening_evaluation": 2}
    issues.sort(key=lambda item: priority.get(item["code"], 3))
    for issue in issues:
        if issue["code"] == "rising_training_loss" and _finite(args.learning_rate) and args.learning_rate > 0:
            issue["message"] = (f"Training loss is rising. Next experiment: --learning-rate {args.learning_rate / 2:.12g} "
                                f"(currently {args.learning_rate:.12g}); keep the duration unchanged.")
    points = _series(evaluation, "eval_loss")
    baseline = next((point for point in points if point[0] == 0), None)
    support = None
    decision = "No changes recommended."
    # SFT/preference objectives have meaningful held-out losses. Policy and
    # distillation objectives do not inherit that interpretation merely by name.
    if method in {"sft", "dpo", "kto", "reward"} and points:
        current = points[-1][1]
        if len(points) == 1:
            if baseline is not None:
                decision = "Starting-model baseline recorded. Training can begin."
                support = f"Baseline evaluation loss: {_display_number(current)}."
            else:
                support = f"First evaluation: loss {_display_number(current)}."
        else:
            previous = points[-2][1]
            reference = baseline[1] if baseline is not None else previous
            relative = (current - reference) / abs(reference) if reference else None
            change = (f" ({100 * relative:+.1f}%)" if not relative or abs(relative) >= .0005 else
                      f" ({100 * relative:+.2g}%)") if relative is not None else ""
            label = "Baseline → current evaluation loss" if baseline is not None else "Evaluation loss"
            support = f"{label}: {_display_number(reference)} → {_display_number(current)}{change}."
            if current < previous:
                decision = "Evaluation loss improved. Continue this run without changing settings."
            elif current > previous:
                decision = "Evaluation loss rose at the last check. Watch the next evaluation before changing settings."
            else:
                decision = "Evaluation loss is unchanged. Continue to the next evaluation before changing settings."
    # Completion judges the whole run, not just its last evaluation interval.
    # A final recommendation must not ask the operator to wait for another eval.
    final_issues = list(issues)
    final_decision = "No setting changes recommended for the next run."
    final_support = support
    if method in {"sft", "dpo", "kto", "reward"} and points:
        if baseline is None:
            final_issues.append({"code": "baseline_missing",
                                 "message": "No starting-model baseline was recorded. This run cannot establish total training gain; check the startup evaluation before rerunning."})
        elif len(points) == 1:
            final_issues.append({"code": "evaluation_sparse",
                                 "message": "Only the starting-model baseline was measured. Evaluate during training before judging the result."})
        else:
            first, last = baseline[1], points[-1][1]
            relative = (last - first) / abs(first) if first else None
            change = (f" ({100 * relative:+.1f}%)" if not relative or abs(relative) >= .0005 else
                      f" ({100 * relative:+.2g}%)") if relative is not None else ""
            best = min(points, key=lambda point: point[1])
            final_support = (f"Baseline → last evaluation loss: {_display_number(first)} → {_display_number(last)}{change}. "
                             f"Best: {_display_number(best[1])} at step {best[0]}.")
            direction = _trend(points, args.get_warmup_steps(planned_steps))["direction"]
            # Only recommend an extension when the measured endpoint is the
            # completed run and the full-history analysis still supports progress.
            if points[-1][0] != completed_steps:
                final_issues.append({"code": "evaluation_sparse",
                                     "message": f"The final checkpoint was not evaluated; the last measurement is from step {points[-1][0]}. Next run, use --eval-strategy epoch to measure the epoch endpoint."})
            elif completed_steps < planned_steps:
                final_decision = f"Training stopped at {completed_steps}/{planned_steps} updates. Review why it stopped before increasing the budget."
            elif best[0] == 0:
                rate = args.learning_rate
                action = (f"Next experiment: --learning-rate {rate / 2:.12g} (currently {rate:.12g}); keep the duration unchanged."
                          if last > first and _finite(rate) and rate > 0 else
                          "Do not add more epochs with the same setup; review the dataset and training objective.")
                final_issues = [item for item in final_issues if item["code"] not in {"possible_overfitting", "worsening_evaluation", "rising_training_loss"}]
                final_issues.append({"code": "starting_model_best", "message": "Training did not beat the starting model. Use the starting model. " + action})
            elif best[1] < last and direction == "up":
                final_issues = [item for item in final_issues if item["code"] not in {"possible_overfitting", "worsening_evaluation"}]
                final_issues.append({"code": "shorter_run", "message": f"Evaluation is deteriorating after its best result at step {best[0]}. Next experiment: --max-steps {int(best[0])} (this run planned {planned_steps}); keep the other settings unchanged."})
            elif last < first and direction == "down" and last == best[1]:
                if args.max_steps > 0:
                    budget = f"--max-steps {2 * args.max_steps} (currently {args.max_steps})"
                else:
                    budget = f"--num-train-epochs {args.num_train_epochs + 1:.12g} (currently {args.num_train_epochs:.12g})"
                final_decision = ("Held-out loss improved and was still falling at the end. "
                                  f"Next experiment: {budget}; keep learning rate and all other settings unchanged.")
            elif last < first:
                final_decision = "Held-out loss improved, but further improvement is not clear. Keep this checkpoint; do not extend the duration yet."
            else:
                final_decision = "Held-out loss did not improve. Do not train longer; review the data and learning rate first."
    elif method in {"grpo", "rloo", "distillation"} and not issues:
        key = "loss" if method == "distillation" else "reward"
        values = _series(training, key)
        if len(values) > 1:
            first, last = values[0][1], values[-1][1]
            improved = last < first if key == "loss" else last > first
            objective = "distillation loss" if key == "loss" else "training reward"
            final_decision = (f"The recorded {objective} improved. Keep this checkpoint; do not extend the run based on training metrics alone." if improved else
                              f"The recorded {objective} did not improve. Revisit the objective and data before training longer.")
            final_support = f"{objective.capitalize()}: {_display_number(first)} → {_display_number(last)}."
        else:
            final_decision = "The run recorded too little objective data to judge progress. Log training metrics more frequently next time."
    return {"decision": decision, "support": support, "issues": issues,
            "final_decision": final_decision, "final_support": final_support,
            "final_issues": final_issues}


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
    return {"direction": direction, "observations": len(points), "fitted_change": change, "residual_scatter": noise}


# Every finite observation contributes to the overall and recency-weighted fits.
# Warmup remains visible; post-warmup behavior is interpreted as a separate phase.
def _trend(points, warmup_steps=0):
    if not points:
        return None
    result = {"history": _observations(points), "overall": _fit(points), "recent": _fit(points, recent=True)}
    current = result["recent"]
    if warmup_steps > 0:
        after = [point for point in points if point[0] > warmup_steps]
        current = _fit(after, recent=True)
        result["post_warmup"] = current
        result["warmup_updates"] = warmup_steps
    result["direction"] = current["direction"]
    return result


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
        higher_better = {"exact_match", "token_f1", "accuracy", "json_valid", "required_fields_present",
                         "reference_values_correct", "judge_score", "judge_instruction_adherence",
                         "judge_coherence", "judge_task_quality", "judge_clarity", "preference_accuracy"}
        lower_better = {"loss", "perplexity", "invalid", "ambiguous"}
        for key in sorted(changes):
            name = key.removeprefix("quality/")
            if name not in higher_better | lower_better:
                continue
            trend = _trend(_series(rounds, key))
            worsening = "down" if name in higher_better else "up"
            if trend and trend["direction"] == worsening:
                findings.append(_finding(f"quality_deterioration:{series}:{name}", "warning", "heuristic",
                                         f"{key} repeatedly deteriorated under matching evaluation conditions.",
                                         {"series": series, "metric": key, "trend": trend,
                                          "model_judgment": name.startswith("judge_")},
                                         "Inspect the individual held-out results; compare a shorter run or adjusted optimization in a separate experiment. Training reward increases do not override this independent evidence."))
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
    trends = {key: _trend(_series(training, key), warmup_steps) for key in latest}
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
    evaluation_points = _series(evaluation, "eval_loss")
    evaluation_trend = _trend(evaluation_points, warmup_steps)
    if not evaluation:
        findings.append(_finding("evaluation_missing", "warning", "measured", "No ordinary evaluation metrics were recorded.", {},
                                 "Review --eval-strategy and the held-out dataset configuration."))
    elif fixed_objective:
        evidence = {"eval_loss": evaluation_trend}
        if evaluation_trend and evaluation_trend["direction"] == "up":
            # Overfitting needs opposing training/eval directions over the same
            # observed interval; retain earlier history in the training conclusion.
            aligned = [p for p in _series(training, "loss") if evaluation_points[0][0] <= p[0] <= evaluation_points[-1][0]]
            aligned_trend = _trend(aligned, warmup_steps)
            if aligned_trend and aligned_trend["direction"] == "down":
                evidence["training_loss"] = aligned_trend
                findings.append(_finding("possible_overfitting", "warning", "heuristic",
                                         "Recent training loss is falling while held-out loss is rising over the same observed interval.", evidence,
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
