"""`trlx <method> [options]`: supervisor and worker sides of a training run.

Supervisor (no --_rank): select GPUs -> validate the config -> choose the
strategy -> allocate a fresh run or rewind the selected checkpoint's run ->
write config.toml -> spawn workers (launch.py) -> display and propagate exit.
The supervisor owns the directory through verification; workers never allocate it.
It never loads a model. It does hold a CUDA context on the first selected
device, because transformers validates bf16 against a real device when the
config is instantiated.

Worker (--_rank r): load the config, the model, the datasets, and train.
Rank 0 owns metrics.jsonl and the preflight report. stdout and stderr are
log.txt, wired by the supervisor.

Preflight runs in stages where its inputs exist: the config-only checks in
the supervisor before the run directory is made, the trainer checks on rank
0 before training, and the forward-pass check on every rank at train begin
(preflight.py). Verify runs once every worker has exited, as its own process
on all selected GPUs (launch.Job), so the trained checkpoint is loaded from
disk the way an operator would load it.

Both display modes read the run directory only: metrics.jsonl is the single
metric source (SPEC 2.3), and log.txt is mirrored to stderr in line mode.
"""

import logging
import os
import pathlib
import sys

from trlx import (
    TrlxError,
    config as config_mod,
    data_load,
    launch,
    metrics,
    model as model_mod,
    preflight,
    ranges,
    render_lines,
    run_dirs,
    show,
    toml_write,
)

# Library loggers whose output is log.txt. Python warnings are routed through
# logging so they land in the file too.
_LOGGERS = ("transformers", "trl", "py.warnings")

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
    document = config_mod.resolve(args.config, args.command, getattr(args, "overrides", None))
    source = config_mod.source_path(document, args.config)
    controls = config_mod.run_settings(document, source)
    args.tui = controls["tui"]
    args.no_verify = not controls["verify"]
    gpu_flag = None if controls["gpus"] == "all" else controls["gpus"]
    strategy_flag = None if controls["strategy"] == "auto" else controls["strategy"]
    gpus, count = launch.select_gpus(gpu_flag)
    physical = launch.physical_ids(gpus, count)
    # From here the supervisor sees only the selected devices: config.load
    # initializes CUDA, and the memory query in choose_strategy indexes them.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)
    cfg = config_mod.from_document(document, args.command, path=source)
    strategy, why = launch.choose_strategy(strategy_flag, cfg, physical)
    # Config-only preflight (SPEC 2.6) before anything is written: a fatal
    # check must not leave a half-made run directory behind.
    preflight.check_config(cfg, source, strategy)
    startup = f"strategy: {strategy} ({why}); GPUs {','.join(physical)}"

    run_dir = _create_run_dir(cfg)
    # Retain exclusive ownership through verification; another supervisor must not
    # rewind files while this job's workers or verification are still using them.
    with run_dirs.locked(run_dir):
        try:
            return _run_job(args, cfg, run_dir, physical, strategy, startup)
        except OSError as e:
            raise TrlxError(f"{e.filename or run_dir}: training supervisor I/O failed: {e}; "
                            "check available space and permissions") from e


# The directory is owned and config preflight has passed before history is changed.
def _run_job(args, cfg, run_dir, physical, strategy, startup):
    log_path = run_dir / show.LOG_FILENAME
    with open(log_path, "ab") as log_file:
        retained = 0
        startup = f"run directory: {run_dir}\n{startup}"
        if cfg.args.resume_from_checkpoint:
            resume = run_dirs.inspect_checkpoint(cfg.args.resume_from_checkpoint)
            retained, marker = run_dirs.rewind(resume, no_staging=getattr(args, "no_staging", False))
            startup = f"{marker}\n{startup}"
        snapshot = _write_snapshot(cfg, run_dir, physical, strategy,
                                   no_staging=getattr(args, "no_staging", False))
        try:
            log_file.write((startup + "\n").encode("utf-8"))
            log_file.flush()
        except OSError as e:
            raise TrlxError(f"{log_path}: cannot write startup log: {e}; check available space and permissions") from e
        # Capture the boundary before workers start, including fast first writes.
        # Startup is printed below; only subsequent log bytes need mirroring.
        log_offset = log_file.tell()
        display_failed = False

        # The log remains authoritative after presentation stops. A failure to
        # record this notice is a supervisor error, not another display error.
        def on_display_error(error):
            nonlocal display_failed
            display_failed = True
            _report_display_error(error, log_file, log_path)

        try:
            print(startup, file=sys.stderr, flush=True)
        except Exception as error:
            on_display_error(error)
        workers = launch.spawn(args.command, str(snapshot), strategy, physical, log_file,
                               force=getattr(args, "force", False), no_staging=getattr(args, "no_staging", False))
        start_verify = None
        if not args.no_verify:
            # Called by the Job once the workers are done: the checkpoint to
            # verify exists only then.
            def start_verify():
                checkpoint = _final_checkpoint(run_dir)
                prompts = _dataset_arg(cfg.verify_prompts)
                return launch.spawn_verify(checkpoint, cfg.model.path, prompts, physical, log_file,
                                           force=getattr(args, "force", False),
                                           no_staging=getattr(args, "no_staging", False))

        job = launch.Job(workers, start_verify)
        try:
            if display_failed:
                failure = job.wait(lambda: None)
            elif args.tui:
                failure = _supervise_tui(run_dir, job, on_display_error)
            else:
                failure = _supervise_lines(run_dir, cfg.ranges, log_path, job, on_display_error,
                                           printed=retained, log_offset=log_offset)
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
        except BaseException:
            # Cancellation and supervisor failures still own process cleanup.
            # Display exceptions have already been handled at their boundary.
            job.terminate()
            raise
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
# process polling stay outside the presentation exception boundary. Resume offsets
# suppress old output, while evaluating all retained metrics preserves change columns.
def _supervise_lines(run_dir, range_table, log_path, job, on_display_error, printed=0, log_offset=0):
    try:
        print(render_lines.header(range_table), flush=True)
        log_reader = open(log_path, "rb")
        log_reader.seek(log_offset)
    except Exception as error:
        on_display_error(error)
        return job.wait(lambda: None)
    metrics_path = run_dir / metrics.FILENAME
    active = True

    # Once a read or render fails, do not retry it on subsequent job polls.
    def tick():
        nonlocal printed, active
        if not active:
            return
        try:
            chunk = log_reader.read()
            if chunk:
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
            if metrics_path.exists():
                rows = ranges.evaluate(metrics.read(metrics_path), range_table)
                for row in rows[printed:]:
                    print(render_lines.line(row), flush=True)
                printed = len(rows)
        except Exception as error:
            active = False
            on_display_error(error)

    try:
        return job.wait(tick)
    finally:
        try:
            log_reader.close()
        except Exception as error:
            on_display_error(error)


# TUI mode: the same display as `trlx show --tui`, polling the run directory.
# Process failure is checked on every poll and surfaces as an exception from
# `load`, which ends the display with the terminal restored. On a normal
# finish the display stays until quit; if the user quits early the job keeps
# going and the supervisor waits for it. Display errors also end only the
# display; process-polling errors retain their supervisor semantics.
def _supervise_tui(run_dir, job, on_display_error):
    failure = None
    poll_error = None

    # Polling is embedded in the TUI callback. Remember its error separately
    # so the outer curses error boundary cannot classify it as presentation.
    def load(log_lines):
        nonlocal failure, poll_error
        try:
            failure = job.poll()
        except Exception as error:
            poll_error = error
            raise
        if failure is not None:
            raise TrlxError(f"{failure[0]} exited with code {failure[1]}")
        return show.load(run_dir, name, range_table, log_lines)

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


# A config.DatasetRef back in its config spelling, for a command line.
def _dataset_arg(ref):
    if ref is None:
        return None
    return ref.source if ref.split is None else f"{ref.source}:{ref.split}"


# Worker side. Everything printed here lands in log.txt.
def _worker(args):
    import torch

    rank = args._rank
    fsdp = "full_shard" if args._strategy == "fsdp" else None
    cfg = config_mod.load(args.config, args.command, fsdp=fsdp,
                          overrides=getattr(args, "overrides", None), resolved=True)
    run_dir = pathlib.Path(cfg.args.output_dir)
    _attach_logging()

    model = model_mod.load_model(cfg.model, cfg.method.model_kind)
    processor = model_mod.load_processor(cfg.model)
    train_set, eval_set = data_load.load(cfg.dataset, cfg.method.dataset_format)
    train_set = _mix_replay(cfg, train_set)

    # Preflight (SPEC 2.6): rank 0 holds the report; every rank carries the
    # callback because the forward-pass check is a collective under FSDP.
    # Other ranks' reports are discarded.
    report = preflight.Report()
    callbacks = [preflight.callback_class()(cfg, processor, train_set, report, run_dir, rank,
                                            no_staging=getattr(args, "no_staging", False))]
    if rank == 0:
        callbacks.append(metrics.callback_class()(run_dir))
    trainer = build_trainer(cfg, model, processor, train_set, eval_set, callbacks)
    if rank == 0:
        # Flushed even when a check is fatal: the lines already noted (the
        # LoRA breakdown, say) are the context for the failure.
        try:
            preflight.check_trainer(cfg, trainer, train_set, report)
        finally:
            report.flush()
        report.write(run_dir, no_staging=getattr(args, "no_staging", False))
    try:
        trainer.train(resume_from_checkpoint=cfg.args.resume_from_checkpoint)
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

    document = config_mod.resolve(args.config, args.method, getattr(args, "overrides", None))
    source = config_mod.source_path(document, args.config)
    controls = config_mod.run_settings(document, source)
    gpu_flag = None if controls["gpus"] == "all" else controls["gpus"]
    gpus, count = launch.select_gpus(gpu_flag)
    physical = launch.physical_ids(gpus[:1], count)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical[0]
    cfg = config_mod.from_document(document, args.method, path=source)
    print(f"check: one process on GPU {physical[0]}", file=sys.stderr)
    preflight.check_config(cfg, source, None)
    print("preflight: config checks passed", file=sys.stderr)
    _attach_logging()

    model = model_mod.load_model(cfg.model, cfg.method.model_kind)
    processor = model_mod.load_processor(cfg.model)
    train_set, eval_set = data_load.load(cfg.dataset, cfg.method.dataset_format)
    train_set = _mix_replay(cfg, train_set)
    report = preflight.Report()
    # The Trainer creates output_dir on construction. check writes nothing,
    # so a directory that did not exist before is removed if still empty.
    run_dir = pathlib.Path(cfg.args.output_dir)
    existed = run_dir.exists()
    try:
        trainer = build_trainer(cfg, model, processor, train_set, eval_set, [])
        preflight.check_trainer(cfg, trainer, train_set, report)
        preflight.check_offpolicy(cfg, trainer.model, processor, train_set, report)
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
# build it. SPEC 2.4: no progress bar; metrics.jsonl is the only reporter.
def build_trainer(cfg, model, processor, train_set, eval_set, callbacks):
    cfg.args.disable_tqdm = True
    extra = {}
    if cfg.teacher is not None:
        # distillation: the teacher is a loaded object, never a path, so
        # [teacher] governs its dtype and attention implementation too.
        extra["teacher_model"] = model_mod.load_model(cfg.teacher, model_mod.CAUSAL)
    if cfg.rewards is not None:
        from trlx import rewards

        extra["reward_funcs"] = rewards.resolve(cfg.rewards)
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
            extra["reference_model"] = model_mod.load_model(cfg.model, cfg.method.model_kind)
    from peft.utils.error import NoMatchingPeftModuleError

    try:
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
def _mix_replay(cfg, train_set):
    if cfg.replay is None:
        return train_set
    return data_load.mix_replay(train_set, cfg.replay, _replay_kl_on(cfg))


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


# Worker logging: library loggers to stderr, which the supervisor wired to
# log.txt. Handlers are added, never replaced, so the operator's log_level on
# the TRL config still governs verbosity.
def _attach_logging():
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in _LOGGERS:
        logging.getLogger(name).addHandler(handler)
