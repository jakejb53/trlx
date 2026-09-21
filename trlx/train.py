"""`trlx <method> [options]`: supervisor and worker sides of a training run.

Supervisor (no --_rank): select GPUs -> validate configuration -> inspect metadata
and fully scan datasets -> review settings/assessment and await consent -> choose
strategy -> allocate or rewind -> write snapshot/assessment -> spawn workers.
The supervisor owns the directory through verification; workers never allocate it.
It never loads model weights. It does hold a CUDA context on the first selected
device, because transformers validates bf16 against a real device when the
config is instantiated.

Worker (--_rank r): load the config, the model, the datasets, and train.
Rank 0 owns metrics.jsonl, quality evidence, and the preflight report. The supervisor drains
raw output and typed feedback separately, recording both in log.txt.

Preflight runs in stages where its inputs exist: the config-only checks in
the supervisor before the run directory is made, the trainer checks on rank
0 before training, and the forward-pass check on every rank at train begin
(preflight.py). Verify runs once every worker has exited, as its own process
on all selected GPUs (launch.Job), so the trained checkpoint is loaded from
disk the way an operator would load it.

metrics.jsonl is the single metric source (SPEC 2.3). Live line mode presents
useful feedback; complete diagnostics remain in log.txt for the TUI and inspection.
"""

import contextlib
import json
import logging
import os
import pathlib
import shutil
import sys

from dataset.progress import Progress, stage
from trlx import (
    TrlxError,
    assessment,
    config as config_mod,
    data_load,
    data_profile,
    feedback,
    launch,
    metrics,
    model as model_mod,
    preflight,
    quality,
    ranges,
    render_lines,
    review,
    run_dirs,
    show,
    toml_write,
)

# Lines of log.txt shown after a worker failure in TUI mode, where the log was
# not mirrored while the display was up.
_FAILURE_TAIL_LINES = 30


# Entry point for the method subcommands. `args` is the argparse namespace.
# Returns the process exit code.
def run(args):
    if args._rank is not None:
        return _worker(args)
    return _supervise(args)


# Supervisor side.
def _supervise(args):
    progress = getattr(args, "progress", None)
    with stage(progress, "resolving training configuration"):
        document = config_mod.resolve(args.config, args.command, getattr(args, "overrides", None))
    source = config_mod.source_path(document, args.config)
    controls = config_mod.run_settings(document, source)
    args.tui = controls["tui"]
    args.no_verify = not controls["verify"]
    gpu_flag = None if controls["gpus"] == "all" else controls["gpus"]
    strategy_flag = None if controls["strategy"] == "auto" else controls["strategy"]
    with stage(progress, "inspecting selected GPUs"):
        gpus, count = launch.select_gpus(gpu_flag)
        physical = launch.physical_ids(gpus, count)
    # From here the supervisor sees only the selected devices: config.load
    # initializes CUDA, and the memory query in choose_strategy indexes them.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)
    with stage(progress, "validating trainer settings"):
        cfg = config_mod.from_document(document, args.command, path=source)
    config_mod.require_assessment(cfg, source)
    assessment_report = _assess(cfg, len(physical), progress=progress)
    # Metadata and full data scans inform consent; weights and run mutation still wait.
    # Workers consume the saved snapshot later and never repeat this prompt.
    if not review.confirm(cfg, progress=progress, assessment=assessment_report):
        if progress is not None:
            progress.finish("cancelled")
        return 0
    with stage(progress, "estimating model memory and selecting strategy"):
        strategy, why = launch.choose_strategy(strategy_flag, cfg, physical)
    # Config-only preflight (SPEC 2.6) before anything is written: a fatal
    # check must not leave a half-made run directory behind.
    preflight.check_config(cfg, source, strategy, progress=progress)
    startup = f"strategy: {strategy} ({why}); GPUs {','.join(physical)}"

    with stage(progress, "allocating run directory"):
        run_dir = _create_run_dir(cfg)
    # Retain exclusive ownership through verification; another supervisor must not
    # rewind files while this job's workers or verification are still using them.
    with stage(progress, f"acquiring run ownership: {run_dir}"), run_dirs.locked(run_dir):
        try:
            return _run_job(args, cfg, run_dir, physical, strategy, startup, assessment_report=assessment_report)
        except OSError as e:
            raise TrlxError(f"{e.filename or run_dir}: training supervisor I/O failed: {e}; "
                            "check available space and permissions") from e


# Text-config facts are explicit metadata, not inferred capabilities or estimates of learning quality.
def _assessment_metadata(model_config):
    text = model_config.get_text_config()
    names = ("model_type", "vocab_size", "max_position_embeddings", "rope_scaling")
    return {name: getattr(text, name, None) for name in names}


# Read and profile every effective row before confirmation; model weights remain worker-owned.
def _assess(cfg, gpu_count, *, progress=None):
    metadata = _assessment_metadata(model_mod.load_config(cfg.model, progress=progress))
    processor = model_mod.assessment_processor(cfg, progress=progress)
    train_set, eval_set = data_load.load(cfg.dataset, cfg.method.dataset_format, progress=progress)
    primary_rows = train_set.num_rows
    train_set = _mix_replay(cfg, train_set, progress=progress)
    profile = data_profile.scan(cfg, processor, train_set, eval_set, primary_rows=primary_rows, progress=progress)
    teacher_metadata = None
    teacher_findings = []
    if cfg.teacher is not None:
        teacher_metadata = _assessment_metadata(model_mod.load_config(cfg.teacher, progress=progress))
        teacher_processor = model_mod.load_processor(cfg.teacher, progress=progress)
        student = getattr(processor, "tokenizer", processor).get_vocab()
        teacher = getattr(teacher_processor, "tokenizer", teacher_processor).get_vocab()
        mismatches = [{"token": token, "student_id": student.get(token), "teacher_id": teacher.get(token)}
                      for token in sorted(student.keys() | teacher.keys()) if student.get(token) != teacher.get(token)]
        if mismatches:
            teacher_findings.append({"code": "distillation_token_ids", "severity": "warning", "basis": "measured",
                                     "summary": "Teacher and student token-to-ID mappings differ.",
                                     "evidence": {"mismatches": mismatches},
                                     "recommendation": "Use compatible token-ID meanings for token-level distillation; equal vocabulary sizes are insufficient."})
    findings = assessment.static_findings(cfg, profile, gpu_count, model_metadata=metadata,
                                         teacher_metadata=teacher_metadata) + teacher_findings
    report = {"version": 1, "method": cfg.method.name, "profile": profile,
              "model": metadata, "teacher": teacher_metadata, "findings": findings, "quality": None}
    if cfg.assessment.quality_checks:
        independent = quality.load_data(cfg.assessment, progress=progress)
        report["quality"] = {"preset": cfg.assessment.quality_preset, "rows": independent.num_rows,
                             "fingerprint": data_profile.fingerprint(independent),
                             "overlap": data_profile.compare_sources(train_set, independent, cfg.method.name,
                                                                       train_exclude_columns=("replay",) if _replay_kl_on(cfg) else ())}
        inputs = quality.inspect_inputs(cfg.assessment, processor, independent, metadata, progress=progress)
        findings.extend(inputs.pop("findings"))
        report["quality"].update(inputs)
        overlap = report["quality"]["overlap"]
        if overlap["identical_examples"] or overlap["prompts"]:
            findings.append({"code": "quality_overlap", "severity": "warning", "basis": "measured",
                             "summary": "Independent evaluation data overlaps the training inputs.",
                             "evidence": overlap,
                             "recommendation": "Use held-out task examples; exact prompt overlap and identical examples are reported separately."})
    return report


# Startup feedback reaches the terminal until the supervisor starts its display.
# The inner reporter stops before log_file closes; child processes own their reporters.
@contextlib.contextmanager
def _job_feedback(args, collector):
    parent = getattr(args, "progress", None)
    if parent is None:
        yield None
        return

    startup = parent.events if isinstance(parent.events, feedback.Startup) else None
    if startup is not None:
        startup.attach(collector)
    try:
        with parent.suspended(), Progress(f"trlx {args.command} supervisor",
                events=lambda event: collector.accept("supervisor", event)) as progress:
            yield progress
    finally:
        if startup is not None:
            startup.detach()


# The directory is owned and config preflight has passed before history is changed.
def _run_job(args, cfg, run_dir, physical, strategy, startup, *, assessment_report=None):
    log_path = run_dir / show.LOG_FILENAME
    with open(log_path, "ab") as log_file, \
            feedback.Collector(log_file, log_path, display=not args.tui) as collector, \
            _job_feedback(args, collector) as progress:
        retained = 0
        startup = f"run directory: {run_dir}\n{startup}"
        if cfg.args.resume_from_checkpoint:
            with stage(progress, "validating and rewinding checkpoint resume"):
                resume = run_dirs.inspect_checkpoint(cfg.args.resume_from_checkpoint)
                retained, marker = run_dirs.rewind(resume, no_staging=getattr(args, "no_staging", False))
            startup = f"{marker}\n{startup}"
        with stage(progress, "writing resolved configuration snapshot"):
            snapshot = _write_snapshot(cfg, run_dir, physical, strategy,
                                       no_staging=getattr(args, "no_staging", False))
        if assessment_report is not None:
            # Publication follows confirmation and resume rewind; workers see exactly the reviewed evidence.
            run_dirs.write_atomic(run_dir / show.ASSESSMENT_FILENAME,
                                  json.dumps(assessment_report, ensure_ascii=False, indent=1) + "\n",
                                  no_staging=getattr(args, "no_staging", False))
        try:
            log_file.write((startup + "\n").encode("utf-8"))
            log_file.flush()
        except OSError as e:
            raise TrlxError(f"{log_path}: cannot write startup log: {e}; check available space and permissions") from e
        display_failed = False

        # The log remains authoritative after presentation stops. A failure to
        # record this notice is a supervisor error, not another display error.
        def on_display_error(error):
            nonlocal display_failed
            display_failed = True
            collector.disable_display()
            with collector.lock:
                _report_display_error(error, log_file, log_path)

        parent_progress = getattr(args, "progress", None)
        if parent_progress is not None and parent_progress.error is not None:
            on_display_error(parent_progress.error)
        try:
            if sys.stderr is not None:
                print(startup, file=sys.stderr, flush=True)
        except Exception as error:
            on_display_error(error)
        with stage(progress, "starting training workers", total=len(physical), unit="workers") as activity:
            workers = launch.spawn(args.command, str(snapshot), strategy, physical, collector,
                                   force=getattr(args, "force", False), no_staging=getattr(args, "no_staging", False))
            activity.update(len(workers))
        start_verify = None
        if not args.no_verify:
            # Called by the Job once the workers are done: the checkpoint to
            # verify exists only then.
            def start_verify():
                with stage(progress, "starting post-training verification", visible=True):
                    checkpoint = _final_checkpoint(run_dir)
                    return launch.spawn_verify(checkpoint, cfg.model.path, physical, collector,
                                               force=getattr(args, "force", False),
                                               no_staging=getattr(args, "no_staging", False))

        job = launch.Job(workers, start_verify, progress=progress, feedback=collector)
        try:
            with stage(progress, "supervising training workers and verification"):
                if display_failed:
                    failure = job.wait(lambda: None)
                elif args.tui:
                    failure = _supervise_tui(run_dir, job, on_display_error)
                else:
                    failure = _supervise_lines(run_dir, cfg.ranges, log_path, job, on_display_error,
                                               printed=retained)
            if failure is not None:
                label, code = failure
                # Final diagnostics are display work too: a broken stream or
                # unreadable report must not replace the job's exit status.
                try:
                    if sys.stderr is not None:
                        print(f"{label} exited with code {code}; see {log_path}", file=sys.stderr, flush=True)
                    if args.tui and not display_failed:
                        state = show.load(run_dir, cfg.args.run_name, cfg.ranges, _FAILURE_TAIL_LINES)
                        for line in state.log_tail:
                            print(line, file=sys.stderr, flush=True)
                except Exception as error:
                    on_display_error(error)
        except BaseException as error:
            # Cancellation and supervisor failures still own process cleanup.
            # Display exceptions have already been handled at their boundary.
            try:
                failures = job.terminate(terminal=True)
                if failures:
                    error.add_note("; ".join(failures))
            except Exception as cleanup_error:
                error.add_note(f"shutdown also failed: {cleanup_error}")
            raise
        if progress is not None:
            progress.finish("completed" if failure is None else f"failed: {failure[0]} exited with code {failure[1]}")
    return 0 if failure is None else failure[1]


# Records loss of presentation before disabling unusable standard streams.
# Log I/O failures propagate: supervision requires the authoritative run log.
def _report_display_error(error, log_file, log_path):
    message = (
        f"display stopped: {type(error).__name__}: {error}; "
        f"training and verification remain supervised; see {log_path}"
    )
    try:
        log_file.write((message + "\n").encode("utf-8"))
        log_file.flush()
    except OSError as exc:
        raise TrlxError(f"{log_path}: cannot record display failure: {exc}") from exc

    # Python flushes standard streams at shutdown; a retained broken pipe
    # would replace even a successful job exit with status 120. None disables
    # that stream without closing a descriptor owned by the caller.
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None:
            try:
                stream.flush()
            except Exception:
                setattr(sys, name, None)
    if sys.stderr is not None:
        try:
            print(message, file=sys.stderr, flush=True)
        except Exception:
            sys.stderr = None


# Line mode: render failures disable only this callback. Job.wait and its
# process polling stay outside the presentation exception boundary. The collector
# supplies only current output; retained metrics still supply resume change columns.
def _supervise_lines(run_dir, range_table, log_path, job, on_display_error, printed=0):
    metrics_path = run_dir / metrics.FILENAME
    active = True
    primed = False

    # Metric output remains stdout; diagnostic interruptions restore the next header.
    def metric_line(text):
        print(text, flush=True)

    stream = render_lines.Stream(range_table, metric_line, width=shutil.get_terminal_size().columns)

    # One renderer owns both streams, so messages cannot bisect a metric row.
    def message(text):
        stream.interrupt()
        if sys.stderr is None:
            raise BrokenPipeError("training stderr is unavailable")
        print(text, file=sys.stderr, flush=True)

    view = feedback.View(message)

    # Once a read or render fails, do not retry it on subsequent job polls.
    def tick():
        nonlocal printed, active, primed
        if not active:
            return
        try:
            for event in job.feedback.take():
                view.consume(event)
            if metrics_path.exists():
                records = metrics.read(metrics_path)
                if not primed:
                    # Retained evaluation measurements remain visible on resume,
                    # but earlier metric rows must not be printed a second time.
                    for record in records[:printed]:
                        stream.observe(record)
                    primed = True
                rows = ranges.evaluate(records, range_table)
                stream.width = shutil.get_terminal_size().columns
                for record, row in zip(records[printed:], rows[printed:]):
                    stream.record(record, row)
                    view.last_feedback = view.clock()
                printed = len(rows)
            view.waiting()
        except Exception as error:
            active = False
            job.feedback.disable_display()
            on_display_error(error)

    failure = job.wait(tick)
    if active:
        try:
            view.warning_summary()
            message(f"run artifacts: {run_dir}; diagnostics: {log_path}")
        except Exception as error:
            on_display_error(error)
    return failure


# TUI mode: the same display as `trlx show --tui`, polling the run directory.
# Process failure is checked on every poll and surfaces as an exception from
# `load`, which ends the display with the terminal restored. On a normal
# finish the display stays until quit; if the user quits early the job keeps
# going and the supervisor waits for it. Display errors also end only the
# display; process-polling errors retain their supervisor semantics.
def _supervise_tui(run_dir, job, on_display_error):
    failure = None
    poll_error = None
    last_log = None

    # Polling is embedded in the TUI callback. Remember its error separately
    # so the outer curses error boundary cannot classify it as presentation.
    def load(log_lines):
        nonlocal failure, poll_error, last_log
        try:
            failure = job.poll()
        except Exception as error:
            poll_error = error
            raise
        if failure is not None:
            raise TrlxError(f"{failure[0]} exited with code {failure[1]}")
        state = show.load(run_dir, name, range_table, log_lines)
        if state.log_tail != last_log and job.progress is not None:
            job.progress.output_seen()
        last_log = state.log_tail
        return state

    try:
        from trlx import render_tui

        name, range_table = show.load_config(run_dir)
        render_tui.run(load)
    except Exception as error:
        if poll_error is not None:
            raise poll_error
        if failure is not None:
            return failure
        on_display_error(error)
    else:
        if not job.done():
            try:
                print(f"display closed; the job continues, follow with: trlx show {run_dir}", file=sys.stderr, flush=True)
            except Exception as error:
                on_display_error(error)
    return job.wait(lambda: None)


# Verify the latest saved checkpoint by step number, after training has exited.
# Preflight refuses save_strategy = "no" so a run must leave an artifact.
def _final_checkpoint(run_dir):
    steps = []
    for entry in run_dir.iterdir():
        m = show.CHECKPOINT_DIR.match(entry.name)
        if m and entry.is_dir():
            steps.append((int(m.group(1)), entry))
    if not steps:
        raise TrlxError(f"{run_dir}: training finished but wrote no checkpoint; nothing to verify")
    return max(steps)[1]


# Worker side. Everything printed here lands in log.txt.
def _worker(args):
    import torch.distributed as distributed

    failure = None
    try:
        return _train_worker(args)
    except BaseException as error:
        failure = error
        raise
    finally:
        # Every worker owns its process group, including failures during trainer
        # construction. No extra barrier: another rank may already have failed.
        if distributed.is_available() and distributed.is_initialized():
            try:
                distributed.destroy_process_group()
            except Exception as error:
                if failure is None:
                    raise TrlxError(f"cannot shut down distributed training: {error}") from error
                try:
                    logging.getLogger("trl").error("distributed cleanup also failed: %s", error)
                except Exception as reporting_error:
                    # Even a broken diagnostic pipe cannot replace the training error.
                    failure.add_note(f"distributed cleanup failed: {error}; reporting failed: {reporting_error}")


# Load and train inside the process-group lifetime owned by _worker.
def _train_worker(args):
    import torch

    rank = args._rank
    progress = getattr(args, "progress", None)
    fsdp = "full_shard" if args._strategy == "fsdp" else None
    with stage(progress, "loading resolved worker configuration"):
        cfg = config_mod.load(args.config, args.command, fsdp=fsdp,
                              overrides=getattr(args, "overrides", None), resolved=True)
    settings = config_mod.require_assessment(cfg, args.config)
    run_dir = pathlib.Path(cfg.args.output_dir)
    _attach_logging(progress)
    feedback.configure_worker_progress(rank)

    model = model_mod.load_model(cfg.model, cfg.method.model_kind, progress=progress)
    processor = model_mod.load_processor(cfg.model, progress=progress)
    train_set, eval_set = data_load.load(cfg.dataset, cfg.method.dataset_format, progress=progress)
    train_set = _mix_replay(cfg, train_set, progress=progress)
    if rank == 0:
        print(f"dataset: {train_set.num_rows} training rows; "
              f"{eval_set.num_rows if eval_set is not None else 0} evaluation rows", flush=True)

    # Preflight (SPEC 2.6): rank 0 holds the report; every rank carries the
    # callback because the forward-pass check is a collective under FSDP.
    # Other ranks' reports are discarded.
    report = preflight.Report()
    callbacks = [preflight.callback_class()(cfg, processor, train_set, report, run_dir, rank,
                                            no_staging=getattr(args, "no_staging", False), progress=progress)]
    writer = metrics.callback_class()(run_dir, settings, cfg.method.name, cfg.ranges) if rank == 0 else None
    quality_callback = None
    if settings.quality_checks:
        quality_callback = quality.callback_class()(settings, run_dir, writer, rank,
                                                    no_staging=getattr(args, "no_staging", False), progress=progress)
        callbacks.append(quality_callback)
    if rank == 0:
        # Completion quality must publish before this callback closes the sole metrics writer.
        callbacks.append(writer)
    trainer = build_trainer(cfg, model, processor, train_set, eval_set, callbacks, progress=progress)
    if quality_callback is not None:
        quality_callback.bind(trainer)
    if rank == 0:
        # Flushed even when a check is fatal: the lines already noted (the
        # LoRA breakdown, say) are the context for the failure.
        try:
            preview = show._read_json(run_dir / show.ASSESSMENT_FILENAME)
            preflight.check_trainer(cfg, trainer, train_set, report, progress=progress,
                                    profile=preview["profile"] if preview is not None else None)
        finally:
            report.flush()
        report.write(run_dir, no_staging=getattr(args, "no_staging", False), progress=progress)
    try:
        with stage(progress, "trainer running", unit="steps", visible=True) as activity:
            activity_callback = metrics.activity_callback_class()(activity)
            trainer.add_callback(activity_callback)
            try:
                trainer.train(resume_from_checkpoint=cfg.args.resume_from_checkpoint)
            finally:
                activity_callback.close(sys.exc_info())
    except torch.cuda.OutOfMemoryError as e:
        raise TrlxError("CUDA memory exhausted during training; reduce batch size or sequence length, "
                        "or use more GPU memory") from e
    except OSError as e:
        raise TrlxError(f"{e.filename or run_dir}: training I/O failed: {e}; "
                        "check available space and permissions") from e
    return 0


# `trlx check <method> [--config <path>]`: preflight alone (SPEC 2.6).
# One process on the first selected GPU, loading and building the trainer as a
# single-GPU worker does, so the checks see what training would see; the
# Trainer places the model. Not device_map="auto": accelerate's dispatch
# hooks replace `forward` with a partial, which TRL's SFTTrainer cannot
# patch. A model too large for one GPU is refused with a message; its
# preflight runs inside the training run under the chosen strategy. No
# strategy is chosen here, so the resume check compares sharding only in a
# run. Nothing is written; the run directory belongs to a training run.
def check(args):
    import torch

    progress = getattr(args, "progress", None)
    with stage(progress, "resolving check configuration"):
        document = config_mod.resolve(args.config, args.method, getattr(args, "overrides", None))
    source = config_mod.source_path(document, args.config)
    controls = config_mod.run_settings(document, source)
    gpu_flag = None if controls["gpus"] == "all" else controls["gpus"]
    with stage(progress, "inspecting selected GPU"):
        gpus, count = launch.select_gpus(gpu_flag)
        physical = launch.physical_ids(gpus[:1], count)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical[0]
    with stage(progress, "validating trainer settings"):
        cfg = config_mod.from_document(document, args.method, path=source)
    config_mod.require_assessment(cfg, source)
    assessment_report = _assess(cfg, 1, progress=progress)
    print(review.render_assessment(assessment_report, cfg, will_publish=False), flush=True)
    print(f"check: one process on GPU {physical[0]}", file=sys.stderr)
    preflight.check_config(cfg, source, None, progress=progress)
    print("preflight: config checks passed", file=sys.stderr)
    _attach_logging()

    model = model_mod.load_model(cfg.model, cfg.method.model_kind, progress=progress)
    processor = model_mod.load_processor(cfg.model, progress=progress)
    train_set, eval_set = data_load.load(cfg.dataset, cfg.method.dataset_format, progress=progress)
    train_set = _mix_replay(cfg, train_set, progress=progress)
    report = preflight.Report()
    # The Trainer creates output_dir on construction. check writes nothing,
    # so a directory that did not exist before is removed if still empty.
    run_dir = pathlib.Path(cfg.args.output_dir)
    existed = run_dir.exists()
    try:
        trainer = build_trainer(cfg, model, processor, train_set, eval_set, [], progress=progress)
        preflight.check_trainer(cfg, trainer, train_set, report, progress=progress, profile=assessment_report["profile"])
        preflight.check_offpolicy(cfg, trainer.model, processor, train_set, report, progress=progress)
    except torch.cuda.OutOfMemoryError:
        raise TrlxError(
            f"[model].path '{cfg.model.path}' does not fit GPU {physical[0]} for a standalone check; "
            "the same preflight runs inside the training run under its strategy"
        )
    finally:
        report.flush()
        if not existed and run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()
    print(f"preflight: passed with {len(report.warnings)} warning(s)", file=sys.stderr)
    return 0


# The method's trainer over loaded objects, as every worker and `check`
# build it. Metrics remain in metrics.jsonl; operational feedback goes to the log.
def build_trainer(cfg, model, processor, train_set, eval_set, callbacks, *, progress=None):
    cfg.args.disable_tqdm = True
    extra = {}
    if cfg.teacher is not None:
        # distillation: the teacher is a loaded object, never a path, so
        # [teacher] governs its dtype and attention implementation too.
        extra["teacher_model"] = model_mod.load_model(cfg.teacher, model_mod.CAUSAL, progress=progress)
    if cfg.rewards is not None:
        from trlx import rewards

        with stage(progress, "resolving reward functions"):
            extra["reward_funcs"] = rewards.resolve(cfg.rewards, progress=progress)
    trainer_cls = cfg.method.trainer_cls
    if _replay_kl_on(cfg):
        # The KL term needs the replay subclass (SPEC 2.9). LoRA runs compare
        # against the same model with adapters off; a full fine-tune needs
        # the original weights as a second copy, loaded per [model] like the
        # training model. Plain mixing (kl_coef = 0) uses the stock trainer.
        from trlx import replay_trainer

        trainer_cls = replay_trainer.ReplayTrainer
        extra["kl_coef"] = cfg.replay.kl_coef
        if cfg.peft is None:
            extra["reference_model"] = model_mod.load_model(cfg.model, cfg.method.model_kind, progress=progress)
    from peft.utils.error import NoMatchingPeftModuleError

    try:
        with stage(progress, "constructing trainer and preparing datasets", visible=True):
            trainer = trainer_cls(
                model=model,
                args=cfg.args,
                train_dataset=train_set,
                eval_dataset=eval_set,
                processing_class=processor,
                peft_config=cfg.peft,
                callbacks=callbacks,
                **extra,
            )
    except NoMatchingPeftModuleError as e:
        # peft refuses target_modules that match nothing while wrapping the
        # model (SPEC 2.6 fatal); reported against the config key.
        raise TrlxError(f"[peft].target_modules {cfg.peft.target_modules!r}: {e}")
    _remove_stock_reporters(trainer)
    return trainer


# True when the run carries the replay KL term, which is what selects the
# replay trainer, the flag column, and the reference copy.
def _replay_kl_on(cfg):
    return cfg.replay is not None and cfg.replay.kl_coef > 0


# The train set with [replay].dataset mixed in, or unchanged without a
# [replay] block. The flag column is added only for the KL term: the stock
# trainer would carry an unused column through packing otherwise.
def _mix_replay(cfg, train_set, *, progress=None):
    if cfg.replay is None:
        return train_set
    return data_load.mix_replay(train_set, cfg.replay, _replay_kl_on(cfg), progress=progress)


# transformers installs PrinterCallback (tqdm disabled) or ProgressCallback
# (tqdm enabled); either prints the raw log dict on every step. That would be
# noise in log.txt; metrics.jsonl carries the same data.
def _remove_stock_reporters(trainer):
    from transformers import PrinterCallback, ProgressCallback

    for cls in (PrinterCallback, ProgressCallback):
        trainer.remove_callback(cls)


# Only the supervisor allocates runs. Replace the operator's parent with the actual
# directory in both trainer args and the document before workers see the snapshot.
def _create_run_dir(cfg):
    if cfg.args.resume_from_checkpoint:
        run_dir = pathlib.Path(cfg.args.resume_from_checkpoint).resolve().parent
    else:
        run_dir = run_dirs.allocate(cfg.args.output_dir, cfg.model.path, cfg.dataset.source)
        if cfg.document.get("run_name") in (None, "None"):
            cfg.args.run_name = run_dir.name
    cfg.args.output_dir = str(run_dir)
    cfg.document["output_dir"] = str(run_dir)
    return run_dir


# Snapshot the selected method and CLI values once, before spawning workers.
# The operator's file is untouched; workers consume this exact hand-off instead.
# Atomic publication preserves a readable resume source if writing fails.
def _write_snapshot(cfg, run_dir, physical, strategy, *, no_staging=False):
    dest = run_dir / show.CONFIG_FILENAME
    document = dict(cfg.document)
    document["run_name"] = cfg.args.run_name
    document["launch"] = {"method": cfg.method.name, "strategy": strategy, "gpus": list(physical)}
    try:
        run_dirs.write_atomic(dest, toml_write.dumps(document), no_staging=no_staging)
    except OSError as e:
        raise TrlxError(f"{dest}: cannot write snapshot: {e.strerror or e}")
    return dest


# Replace console delivery once; library levels and file handlers remain intact.
def _attach_logging(progress=None):
    feedback.configure_logging(progress.events if progress is not None else None)
