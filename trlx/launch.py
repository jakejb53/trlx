"""GPU selection, strategy choice, and worker processes (SPEC 2.5).

The supervisor (train.py, no --_rank) never trains. It spawns one worker per
selected GPU; rank 0 owns metrics.jsonl and preflight. Each worker sees
exactly its own device through CUDA_VISIBLE_DEVICES, so LOCAL_RANK is 0
everywhere and RANK distinguishes them. Verify is a further process started
once every worker has exited, with every selected device visible (SPEC 2.7).
"""

import os
import socket
import sys
import time

import torch

from trlx import TrlxError, model as model_mod, processes

# Bytes-per-parameter multipliers for the fit rule (PLAN.md multi-GPU design):
# a peft run holds frozen weights plus a small adapter and its optimizer
# state; a full fine-tune holds weights, gradients, and two optimizer moments
# in fp32 plus activations. Compared against the smallest selected GPU.
PEFT_MULTIPLIER = 1.5
FULL_MULTIPLIER = 8

# A full fine-tune with the replay KL term keeps a frozen copy of the
# original weights on every rank (SPEC 2.9): weights at dtype, no gradients
# or optimizer state, and never sharded.
COPY_MULTIPLIER = 1

# The one process-group rendezvous address trlx uses: all workers are local.
MASTER_ADDR = "127.0.0.1"

# How often the supervisor checks worker liveness, in seconds.
_POLL_SECONDS = 0.5


# Device selection. --gpus absent: every visible device; present: the listed
# indices, positions in the currently visible set. Validated against the
# device count so a typo fails here rather than as a CUDA error mid-load.
# Returns (indices, visible_count).
def select_gpus(flag):
    count = torch.cuda.device_count()
    if count == 0:
        raise TrlxError("no CUDA device visible")
    if flag is None:
        return list(range(count)), count
    try:
        indices = [int(x) for x in flag.split(",")]
    except ValueError:
        raise TrlxError(f"--gpus '{flag}': expected comma-separated integers")
    bad = [i for i in indices if not 0 <= i < count]
    if bad:
        raise TrlxError(f"--gpus {flag}: device {bad[0]} does not exist ({count} visible)")
    if len(set(indices)) != len(indices):
        raise TrlxError(f"--gpus {flag}: repeated index")
    return indices, count


# Physical device ids for the selected positions, honouring an operator's own
# CUDA_VISIBLE_DEVICES rather than overwriting it blindly.
def physical_ids(gpus, count):
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = existing.split(",") if existing else [str(i) for i in range(count)]
    return [visible[i] for i in gpus]


# Strategy for the run. `override` is --strategy. One GPU is "single": no
# process group, no sharding. Otherwise the fit rule decides between ddp and
# fsdp: ddp when the whole training state plus any original copy fits the
# smallest selected GPU; else fsdp, where the training state is spread over
# the ranks but the copy is not, and a per-rank estimate that still does not
# fit refuses the run, forced or not. Both estimates are parameter-count
# heuristics that ignore activations (PRINCIPLES: labelled estimates).
# Returns (strategy, explanation) where the explanation names the numbers
# the decision rested on, for the startup line and the log.
def choose_strategy(override, cfg, physical):
    if len(physical) == 1:
        if override is not None:
            raise TrlxError(f"--strategy {override}: one GPU selected; strategy applies to multi-GPU runs")
        return "single", "one GPU"
    weights, multiplier, copy = _estimate(cfg)
    need = weights * (multiplier + copy)
    smallest = min(_total_memory(physical))
    explanation = (
        f"model needs about {need / 2**30:.1f} GiB at {cfg.model.dtype} x{multiplier}"
        f"{' (peft)' if cfg.peft is not None else ' (full fine-tune)'}"
        f"{' plus an original copy' if copy else ''}, "
        f"smallest GPU has {smallest / 2**30:.1f} GiB"
    )
    strategy = override if override is not None else ("ddp" if need < smallest else "fsdp")
    if strategy == "fsdp":
        world = len(physical)
        per_rank = weights * multiplier / world + weights * copy
        if per_rank >= smallest:
            raise TrlxError(
                f"sharded over {world} GPUs each rank still needs about {per_rank / 2**30:.1f} GiB; "
                f"{explanation}"
            )
        explanation += f"; sharded over {world} GPUs about {per_rank / 2**30:.1f} GiB per rank"
    if override is not None:
        return override, f"forced by --strategy; {explanation}"
    return strategy, explanation


# Parameter bytes at [model].dtype, the fit multiplier for the training
# state, and the copy multiplier (COPY_MULTIPLIER or 0). The model class is
# instantiated on the meta device: shapes only, no weights, no GPU.
def _estimate(cfg):
    config = model_mod.load_config(cfg.model)
    cls = model_mod.model_class(cfg.model, config, cfg.method.model_kind)
    with torch.device("meta"):
        params = sum(p.numel() for p in cls(config).parameters())
    itemsize = getattr(torch, cfg.model.dtype).itemsize
    multiplier = PEFT_MULTIPLIER if cfg.peft is not None else FULL_MULTIPLIER
    kl_copy = cfg.peft is None and cfg.replay is not None and cfg.replay.kl_coef > 0
    return params * itemsize, multiplier, COPY_MULTIPLIER if kl_copy else 0


# Total memory of each selected device. The supervisor's CUDA_VISIBLE_DEVICES
# is already the selected set (train.py sets it before config.load, which
# itself initializes CUDA: transformers validates bf16 against a real
# device), so visible index i is the i-th selected GPU.
def _total_memory(physical):
    return [torch.cuda.get_device_properties(i).total_memory for i in range(len(physical))]


# Starts one worker per selected device over the supervisor's resolved snapshot.
# The collector drains feedback and raw output into log.txt. With one
# device no distributed variables are set, so accelerate runs single-process.
# Complete the cohort before propagating Ctrl+C: a subset cannot rendezvous.
@processes.defer_interrupt()
def spawn(method, config_path, strategy, physical, collector, *, force=False, no_staging=False):
    world = len(physical)
    port = _free_port() if world > 1 else None
    procs = []
    for rank, device in enumerate(physical):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
        if world > 1:
            env.update(
                RANK=str(rank),
                WORLD_SIZE=str(world),
                LOCAL_RANK="0",
                MASTER_ADDR=MASTER_ADDR,
                MASTER_PORT=str(port),
            )
        # Internal launch facts are separate from user configuration overrides.
        cmd = [sys.executable, "-m", "trlx.cli", method, "--config", str(config_path),
               "--_rank", str(rank), "--_strategy", strategy]
        if force:
            cmd.append("--force")
        if no_staging:
            cmd.append("--no-staging")
        try:
            procs.append(collector.spawn(cmd, env, f"rank {rank}"))
        except OSError as e:
            collector.stop(terminal=True)
            raise TrlxError(f"cannot start worker rank {rank}: {e.strerror or e}")
    return procs


# Detect the first failing rank, then stop its peers with bounded cleanup time.
# Preserve the original code even when other ranks subsequently exit on SIGINT.
def check(procs, *, feedback=None):
    for rank, proc in enumerate(procs):
        code = proc.poll()
        if code is not None and code != 0:
            terminate(procs, feedback=feedback)
            return rank, code
    return None


# True while any of `procs` is still running.
def running(procs):
    return any(proc.poll() is None for proc in procs)


# Starts the verify process for a finished run (SPEC 2.7): the standalone
# command on the run's final checkpoint, with every selected device visible
# so a model larger than one GPU can spread across them. Output joins
# log.txt like the workers'.
def spawn_verify(checkpoint, base, physical, collector, *, force=False, no_staging=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(physical))
    cmd = [sys.executable, "-m", "trlx.cli", "verify", str(checkpoint), "--base", base]
    if force:
        cmd.append("--force")
    if no_staging:
        cmd.append("--no-staging")
    try:
        return collector.spawn(cmd, env, "verify")
    except OSError as e:
        raise TrlxError(f"cannot start verify: {e.strerror or e}")


# The supervisor's processes: the workers, then verify once every worker has
# exited 0. `start_verify` returns the verify Popen when called, or is None
# for --no-verify; it is called late because the checkpoint to verify exists
# only after training. A failure is (label, code) for the message and exit.
class Job:
    # Own worker/verify lifetimes; a progress-log failure remains a supervisor failure.
    def __init__(self, workers, start_verify, *, progress=None, feedback=None):
        self.workers = workers
        self.start_verify = start_verify
        self.verify = None
        self.progress = progress
        self.feedback = feedback
        self._failure = None
        self._cancelled = False

    # Advance supervision; failed workers and the verify transition may require bounded cleanup.
    def poll(self):
        if self._failure is not None:
            return self._failure
        if self._cancelled:
            return "cancelled", 130
        if self.feedback is not None:
            self.feedback.check()
        if self.progress is not None:
            self.progress.check_error()
        failure = check(self.workers, feedback=self.feedback)
        if failure is not None:
            self._failure = (f"worker rank {failure[0]}", failure[1])
            self.start_verify = None
            return self._failure
        if self.verify is None and self.start_verify is not None and not running(self.workers):
            # Worker exit alone does not prove descendant processes released CUDA.
            errors = terminate(self.workers, feedback=self.feedback)
            if errors:
                raise TrlxError("cannot start verification after incomplete worker cleanup: " + "; ".join(errors))
            self.verify = self.start_verify()
        if self.verify is not None:
            code = self.verify.poll()
            if code is not None and code != 0:
                self._failure = ("verify", code)
                return self._failure
        return None

    # True once nothing remains to run: workers exited (any code), and verify
    # was not wanted or has exited (any code; a failure is reported by poll).
    def done(self):
        if self._failure is not None or self._cancelled:
            return True
        if running(self.workers):
            return False
        if self.start_verify is None:
            return True
        return self.verify is not None and self.verify.poll() is not None

    # Blocks until everything exits or something fails. `tick` runs on every
    # poll and once more at the end so the caller's display sees the final
    # state.
    def wait(self, tick):
        while not self.done():
            tick()
            failure = self.poll()
            if failure is not None:
                self._finish_feedback(failure)
                tick()
                return failure
            time.sleep(_POLL_SECONDS)
        failure = self.poll()
        self._finish_feedback(failure)
        tick()
        return failure

    # Cleanup diagnostics cannot overwrite the worker/verification failure already observed.
    def _finish_feedback(self, failure):
        if self.feedback is None:
            return
        try:
            self.feedback.finish()
        except Exception as error:
            if failure is None:
                raise
            self.feedback.shutdown_notice(f"shutdown error: {error}")

    # Cancellation revokes verification before cleanup; the collector also knows
    # children created during a spawn interrupted before its return value was assigned.
    def terminate(self, *, terminal=None, cancelled=False):
        self._cancelled = True
        self.start_verify = None
        if self.feedback is not None:
            return self.feedback.stop(terminal=terminal, cancelled=cancelled)
        return terminate(self.workers + ([self.verify] if self.verify is not None else []), cancelled=cancelled)


# All shutdown entry points use the same group-aware policy.
def terminate(procs, *, feedback=None, cancelled=False):
    if feedback is not None:
        return feedback.stop(procs, cancelled=cancelled)

    # A caller without a collector still receives explicit shutdown diagnostics.
    def report(message):
        if sys.stderr is not None:
            print(message, file=sys.stderr, flush=True)

    return processes.stop(procs, report, cancelled=cancelled)


# An unused TCP port for the rendezvous, released just before the workers
# bind it. The window between is the usual accepted race.
def _free_port():
    with socket.socket() as s:
        s.bind((MASTER_ADDR, 0))
        return s.getsockname()[1]
