"""`trlx <method> <config>`: supervisor and worker sides of a training run.

Supervisor (no --_rank): select GPUs -> validate the config -> choose the
strategy -> create the run directory and config.toml -> open log.txt ->
spawn one worker per GPU (launch.py) -> display -> wait and propagate exit.
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

import json
import logging
import os
import pathlib
import sys
import tomllib

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
    show,
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
    gpus, count = launch.select_gpus(args.gpus)
    physical = launch.physical_ids(gpus, count)
    # From here the supervisor sees only the selected devices: config.load
    # initializes CUDA, and the memory query in choose_strategy indexes them.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)
    cfg = config_mod.load(args.config, args.command)
    strategy, why = launch.choose_strategy(args.strategy, cfg, physical)
    # Config-only preflight (SPEC 2.6) before anything is written: a fatal
    # check must not leave a half-made run directory behind.
    preflight.check_config(cfg, args.config, strategy)
    startup = f"strategy: {strategy} ({why}); GPUs {','.join(physical)}"
    print(startup, file=sys.stderr)

    run_dir = _create_run_dir(cfg.args)
    _write_snapshot(args.config, run_dir, physical, strategy, cfg.args.run_name)
    log_path = run_dir / show.LOG_FILENAME
    with open(log_path, "ab") as log_file:
        log_file.write((startup + "\n").encode("utf-8"))
        log_file.flush()
        workers = launch.spawn(args.command, args.config, strategy, physical, log_file)
        start_verify = None
        if not args.no_verify:
            # Called by the Job once the workers are done: the checkpoint to
            # verify exists only then.
            def start_verify():
                checkpoint = _final_checkpoint(run_dir)
                prompts = _dataset_arg(cfg.verify_prompts)
                return launch.spawn_verify(checkpoint, cfg.model.path, prompts, physical, log_file)

        job = launch.Job(workers, start_verify)
        try:
            if args.tui:
                failure = _supervise_tui(run_dir, job)
            else:
                failure = _supervise_lines(run_dir, cfg.ranges, log_path, job)
        except BaseException:
            job.terminate()
            raise
    if failure is None:
        return 0
    label, code = failure
    print(f"{label} exited with code {code}; see {log_path}", file=sys.stderr)
    if args.tui:
        state = show.load(run_dir, cfg.args.run_name, cfg.ranges, _FAILURE_TAIL_LINES)
        for line in state.log_tail:
            print(line, file=sys.stderr)
    return code


# Line mode: header once, then every metrics.jsonl row not yet printed, on
# each poll, plus new log.txt bytes copied to stderr.
def _supervise_lines(run_dir, range_table, log_path, job):
    print(render_lines.header(range_table), flush=True)
    metrics_path = run_dir / metrics.FILENAME
    printed = 0
    with open(log_path, "rb") as log_reader:

        def tick():
            nonlocal printed
            chunk = log_reader.read()
            if chunk:
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
            if metrics_path.exists():
                rows = ranges.evaluate(metrics.read(metrics_path), range_table)
                for row in rows[printed:]:
                    print(render_lines.line(row), flush=True)
                printed = len(rows)

        return job.wait(tick)


# TUI mode: the same display as `trlx show --tui`, polling the run directory.
# Process failure is checked on every poll and surfaces as an exception from
# `load`, which ends the display with the terminal restored. On a normal
# finish the display stays until quit; if the user quits early the job keeps
# going and the supervisor waits for it.
def _supervise_tui(run_dir, job):
    from trlx import render_tui

    name, range_table = show.load_config(run_dir)
    failure = []

    def load(log_lines):
        found = job.poll()
        if found is not None:
            failure.append(found)
            raise TrlxError(f"{found[0]} exited with code {found[1]}")
        return show.load(run_dir, name, range_table, log_lines)

    try:
        render_tui.run(load)
    except TrlxError:
        if not failure:
            raise
        return failure[0]
    if not job.done():
        print(f"display closed; the job continues, follow with: trlx show {run_dir}", file=sys.stderr)
    return job.wait(lambda: None)


# The highest-numbered checkpoint-N in the run directory: the final weights,
# since the Trainer saves at the last step under a step save strategy and
# preflight refuses save_strategy = "no".
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
    rank = args._rank
    fsdp = "full_shard" if args.strategy == "fsdp" else None
    cfg = config_mod.load(args.config, args.command, fsdp=fsdp)
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
    callbacks = [preflight.callback_class()(cfg, processor, train_set, report, run_dir, rank)]
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
        report.write(run_dir)
    trainer.train(resume_from_checkpoint=cfg.args.resume_from_checkpoint)
    return 0


# `trlx check <method> <config>`: preflight alone (SPEC 2.6). One process on
# the first visible GPU, loading and building the trainer exactly as a
# single-GPU worker does, so the checks see what training would see; the
# Trainer places the model. Not device_map="auto": accelerate's dispatch
# hooks replace `forward` with a partial, which TRL's SFTTrainer cannot
# patch. A model too large for one GPU is refused with a message; its
# preflight runs inside the training run under the chosen strategy. No
# strategy is chosen here, so the resume check compares sharding only in a
# run. Nothing is written; the run directory belongs to a training run.
def check(args):
    import torch

    gpus, count = launch.select_gpus(None)
    physical = launch.physical_ids(gpus[:1], count)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical[0]
    cfg = config_mod.load(args.config, args.method)
    print(f"check: one process on GPU {physical[0]}", file=sys.stderr)
    preflight.check_config(cfg, args.config, None)
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


# Creates output_dir. An existing non-empty directory is refused unless the
# run resumes from a checkpoint, so two runs never write into one directory.
def _create_run_dir(args):
    run_dir = pathlib.Path(args.output_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_from_checkpoint:
        raise TrlxError(f"{run_dir}: output_dir exists and is not empty; set resume_from_checkpoint to continue it")
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise TrlxError(f"{run_dir}: cannot create output_dir: {e.strerror or e}")
    return run_dir


# config.toml snapshot: the operator's file byte for byte, then a [launch]
# table with the facts trlx decided (SPEC 2.3). The copy keeps the operator's
# comments; nothing of theirs is re-emitted. show.load_config needs run_name
# at top level, and config.py fills it in when absent, so the resolved value
# is prepended in that case: prepended, because a top-level key written after
# the operator's tables would belong to the last table. GPUs are the physical
# ids as CUDA_VISIBLE_DEVICES spells them, which may be UUIDs, hence strings.
def _write_snapshot(config_path, run_dir, physical, strategy, run_name):
    dest = run_dir / show.CONFIG_FILENAME
    try:
        with open(config_path, "rb") as f:
            original = f.read()
        # config.load already parsed this file, so a second parse cannot fail
        # on syntax; it only answers whether the operator set run_name.
        header = b"" if "run_name" in tomllib.loads(original.decode("utf-8")) else (
            f"# resolved by trlx\nrun_name = {json.dumps(run_name)}\n\n".encode("utf-8")
        )
        gpus = ", ".join(json.dumps(g) for g in physical)
        launch_table = f"\n[launch]\nstrategy = {json.dumps(strategy)}\ngpus = [{gpus}]\n".encode("utf-8")
        with open(dest, "wb") as f:
            f.write(header + original + launch_table)
    except OSError as e:
        raise TrlxError(f"{dest}: cannot write snapshot: {e.strerror or e}")


# Worker logging: library loggers to stderr, which the supervisor wired to
# log.txt. Handlers are added, never replaced, so the operator's log_level on
# the TRL config still governs verbosity.
def _attach_logging():
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in _LOGGERS:
        logging.getLogger(name).addHandler(handler)
